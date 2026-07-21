# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build Dataset APIs for Material Agent."""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from world_understanding.utils.credentials import (
    path_exists_with_safe_diagnostics,
    redact_sensitive_path,
)
from world_understanding.utils.result_projection import (
    project_result_metadata,
    retain_safe_result_path,
)
from world_understanding.utils.safe_repr import SecretSafeReprMixin

from material_agent.api.types import APIResult

logger = logging.getLogger(__name__)

_USD_BUILD_FAILURE_MESSAGE = "USD dataset build failed"
_PDF_VECTORSTORE_BUILD_FAILURE_MESSAGE = "PDF vectorstore build failed"
_PREPARE_DATASET_FAILURE_MESSAGE = "Prepare dataset failed"


def _log_diagnostic_path(label: str, value: Path) -> None:
    """Log a runtime path through the credential-safe diagnostic projection."""
    logger.info("%s: %s", label, redact_sensitive_path(value))


def _projected_int(mapping: dict[str, Any], key: str) -> int:
    """Read an integer from projected metadata without widening the schema."""
    value = mapping.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _projected_mapping(value: Any) -> dict[str, Any] | None:
    """Keep a projected mapping field in its declared public shape."""
    return value if isinstance(value, dict) else None


def _projected_mapping_list(value: Any) -> list[dict[str, Any]]:
    """Keep only mapping entries from an already projected result list."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _projected_string_list(value: Any) -> list[str]:
    """Keep only exact string entries from an already projected result list."""
    if not isinstance(value, list):
        return []
    return [item for item in value if type(item) is str]


# ============================================================================
# USD Dataset Building API
# ============================================================================


@dataclass
class BuildDatasetUsdInput:
    """Input parameters for USD dataset building API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        source_override: Optional path to USD file or directory (overrides config)
        output_dir_override: Optional output directory (overrides config)
        extract_metadata: Extract prim metadata
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    source_override: Path | None = None  # Can be file or directory
    output_dir_override: Path | None = None
    extract_metadata: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Validate inputs."""
        # Handle config as either Path or dict
        if isinstance(self.config, dict):
            if not self.config:
                raise ValueError("Config dictionary cannot be empty")
        else:
            self.config = Path(self.config)
            if not path_exists_with_safe_diagnostics(
                self.config,
                label="USD dataset configuration",
            ):
                raise FileNotFoundError("Config file not found")

        if self.source_override:
            self.source_override = Path(self.source_override)

        if self.output_dir_override:
            self.output_dir_override = Path(self.output_dir_override)


@dataclass(repr=False)
class BuildDatasetUsdOutput(SecretSafeReprMixin, APIResult):
    """Output results from USD dataset building API."""

    dataset_path: Path | None = None
    num_prims: int = 0
    num_images: int = 0
    batch_results: dict[str, dict[str, Any]] | None = None  # For batch processing
    raw_result: dict[str, Any] | None = None


async def abuild_dataset_usd(params: BuildDatasetUsdInput) -> BuildDatasetUsdOutput:
    """Build a dataset from USD file(s) by rendering views of each prim.

    This command will intelligently handle both single file and batch processing:
    - If config has 'usd_path': processes a single USD file
    - If config has 'usd_dir': processes all USD files in that directory

    For batch processing, subdirectories will be created for each USD file.

    Args:
        params: USD dataset building input parameters

    Returns:
        BuildDatasetUsdOutput with results or error information
    """
    import yaml

    logger.info("Starting USD dataset building via API")
    if isinstance(params.config, dict):
        logger.info("Using in-memory config dictionary")
    else:
        logger.info("Using configuration file")

    try:
        # Load config - either from file or use provided dict
        if isinstance(params.config, dict):
            config_data = params.config
        else:
            with open(params.config, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

        # Determine if source override points to a directory or file
        is_batch_mode = False
        if params.source_override:
            if params.source_override.is_dir():
                is_batch_mode = True
        elif "usd_dir" in config_data:
            is_batch_mode = True
        elif "usd_path" not in config_data:
            raise ValueError(
                "Configuration must contain either 'usd_path' (for single file) "
                "or 'usd_dir' (for batch processing)"
            )

        if is_batch_mode:
            return await _build_dataset_usd_batch(params, config_data)
        else:
            return await _build_dataset_usd_single(params)

    except Exception:
        # Config parsers, workflow backends, and filesystem APIs may reflect
        # credential-bearing paths or values in exception text. The API owns
        # this diagnostic boundary, so neither the exception nor its traceback
        # is copied into logs or the returned result.
        logger.error(_USD_BUILD_FAILURE_MESSAGE)
        return BuildDatasetUsdOutput(
            success=False,
            error=_USD_BUILD_FAILURE_MESSAGE,
        )


async def _build_dataset_usd_single(
    params: BuildDatasetUsdInput,
) -> BuildDatasetUsdOutput:
    """Build dataset from a single USD file."""
    from material_agent.workflows import (
        create_usd_data_preparation_workflow_from_config,
    )

    logger.info("Processing single USD file")

    workflow = create_usd_data_preparation_workflow_from_config()

    initial_context: dict[str, Any] = {}

    # Add config as either path or dict
    if isinstance(params.config, dict):
        initial_context["config_dict"] = params.config
    else:
        initial_context["config_path"] = params.config

    if params.source_override:
        initial_context["source_override"] = params.source_override
        logger.info("Using USD source override")

    if params.output_dir_override:
        initial_context["output_dir_override"] = params.output_dir_override
        logger.info("Using output directory override")

    if params.extract_metadata:
        initial_context["extract_prim_metadata"] = params.extract_metadata
        logger.info("Metadata extraction enabled")

    # Run workflow
    logger.info("Executing dataset build workflow")
    result = await workflow.arun(initial_context)
    safe_result = project_result_metadata(result)

    return BuildDatasetUsdOutput(
        success=True,
        dataset_path=retain_safe_result_path(result.get("dataset_path")),
        num_prims=_projected_int(safe_result, "num_prims"),
        num_images=_projected_int(safe_result, "num_images"),
        raw_result=safe_result,
    )


async def _build_dataset_usd_batch(
    params: BuildDatasetUsdInput, config_data: dict[str, Any]
) -> BuildDatasetUsdOutput:
    """Build datasets from multiple USD files in a directory."""
    from material_agent.batch_processor import process_usd_batch
    from material_agent.workflows import (
        create_usd_data_preparation_workflow_from_config,
    )

    logger.info("Detected batch processing mode")

    # Get USD directory
    if params.source_override and params.source_override.is_dir():
        usd_dir = params.source_override
        logger.info("Using USD directory override")
    elif "usd_dir" in config_data:
        # For file-based config, resolve relative to config file
        # For dict-based config, use as-is (must be absolute or relative to cwd)
        if isinstance(params.config, Path):
            config_dir = params.config.parent
            usd_dir = config_dir / Path(config_data["usd_dir"])
            usd_dir = usd_dir.resolve()
        else:
            usd_dir = Path(config_data["usd_dir"])
        logger.info("Using usd_dir from config")
    else:
        raise ValueError("Batch mode requires usd_dir in config or --source directory")

    # Get output directory
    if params.output_dir_override:
        batch_output_dir = params.output_dir_override
    elif "output_dir" in config_data:
        # For file-based config, resolve relative to config file
        # For dict-based config, use as-is (must be absolute or relative to cwd)
        if isinstance(params.config, Path):
            config_dir = params.config.parent
            batch_output_dir = config_dir / Path(config_data["output_dir"])
            batch_output_dir = batch_output_dir.resolve()
        else:
            batch_output_dir = Path(config_data["output_dir"])
    else:
        batch_output_dir = Path("output")

    # Check if USD directory exists
    if not usd_dir.exists():
        raise FileNotFoundError("USD directory not found")

    # Create workflow once
    workflow = create_usd_data_preparation_workflow_from_config()

    # Prepare base context
    base_context: dict[str, Any] = {}

    # Add config as either path or dict
    if isinstance(params.config, dict):
        base_context["config_dict"] = params.config
    else:
        base_context["config_path"] = params.config

    if params.extract_metadata:
        base_context["extract_prim_metadata"] = params.extract_metadata

    # Run batch processor
    batch_result = await process_usd_batch(
        usd_dir=usd_dir,
        batch_output_dir=batch_output_dir,
        workflow_runner=lambda ctx: workflow.arun(ctx),
        base_context=base_context,
    )
    if not isinstance(batch_result, dict):
        raise ValueError("Batch processor returned an invalid result")
    raw_results = batch_result["results"]
    raw_successful_builds = batch_result["num_files_processed"]
    raw_failed_builds = batch_result["num_files_failed"]
    if (
        not isinstance(raw_results, dict)
        or not isinstance(raw_successful_builds, int)
        or isinstance(raw_successful_builds, bool)
        or not isinstance(raw_failed_builds, int)
        or isinstance(raw_failed_builds, bool)
    ):
        raise ValueError("Batch processor returned invalid result metadata")
    safe_batch_result = project_result_metadata(batch_result)

    safe_results = _projected_mapping(safe_batch_result.get("results")) or {}
    results = {
        key: value
        for key, value in safe_results.items()
        if type(key) is str and isinstance(value, dict)
    }
    successful_builds = _projected_int(safe_batch_result, "num_files_processed")
    failed_builds = _projected_int(safe_batch_result, "num_files_failed")

    logger.info(
        f"Batch processing complete: {successful_builds} successful, "
        f"{failed_builds} failed"
    )

    return BuildDatasetUsdOutput(
        success=raw_failed_builds == 0,
        batch_results=results,
        raw_result=safe_batch_result,
    )


# ============================================================================
# PDF VectorStore Building API
# ============================================================================


@dataclass
class BuildDatasetPdfVectorstoreInput:
    """Input parameters for PDF vectorstore building API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        source_override: Optional path to PDF file or directory (overrides config)
        output_dir_override: Optional output directory (overrides config)
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    source_override: Path | None = None  # PDF file or directory
    output_dir_override: Path | None = None
    verbose: bool = False

    def __post_init__(self):
        """Validate inputs."""
        # Handle config as either Path or dict
        if isinstance(self.config, dict):
            if not self.config:
                raise ValueError("Config dictionary cannot be empty")
        else:
            self.config = Path(self.config)
            if not path_exists_with_safe_diagnostics(
                self.config,
                label="PDF vectorstore configuration",
            ):
                raise FileNotFoundError("Config file not found")

        if self.source_override:
            self.source_override = Path(self.source_override)

        if self.output_dir_override:
            self.output_dir_override = Path(self.output_dir_override)


@dataclass(repr=False)
class BuildDatasetPdfVectorstoreOutput(SecretSafeReprMixin, APIResult):
    """Output results from PDF vectorstore building API."""

    vectorstore_path: Path | None = None
    num_documents_indexed: int = 0
    num_texts: int = 0
    num_images: int = 0
    embedding_dimension: int = 0
    extraction_result: dict[str, Any] | None = None
    split_result: dict[str, Any] | None = None
    raw_result: dict[str, Any] | None = None


async def abuild_dataset_pdf_vectorstore(
    params: BuildDatasetPdfVectorstoreInput,
) -> BuildDatasetPdfVectorstoreOutput:
    """Build a multimodal vector store from PDF documents.

    This command processes PDF files to extract content (text, images, tables),
    splits them by type, and creates a searchable vector store.

    Args:
        params: PDF vectorstore building input parameters

    Returns:
        BuildDatasetPdfVectorstoreOutput with results or error information
    """
    logger.info("Starting PDF vectorstore building via API")
    if isinstance(params.config, dict):
        logger.info("Using in-memory config dictionary")
    else:
        _log_diagnostic_path("Configuration file", params.config)

    if params.source_override:
        _log_diagnostic_path("Source override", params.source_override)
    if params.output_dir_override:
        _log_diagnostic_path("Output directory override", params.output_dir_override)

    try:
        # Import workflow factory
        from material_agent.workflows.factory import (
            create_pdf_vectorstore_workflow_from_config,
        )

        # Prepare initial context with config and overrides
        initial_context: dict[str, Any] = {
            "source_override": (
                str(params.source_override) if params.source_override else None
            ),
            "output_dir_override": (
                str(params.output_dir_override) if params.output_dir_override else None
            ),
            "verbose": params.verbose,
        }

        # Add config as either path or dict
        if isinstance(params.config, dict):
            initial_context["config_dict"] = params.config
        else:
            initial_context["config_path"] = str(params.config)

        # Create workflow
        workflow = create_pdf_vectorstore_workflow_from_config()

        # Run the workflow
        logger.info("Processing PDFs and building vector store...")
        result = await workflow.arun(initial_context=initial_context)
        safe_result = project_result_metadata(result)

        # Check if workflow completed successfully
        if result.get("workflow_completed"):
            logger.info("PDF vectorstore workflow completed successfully")

            vectorstore_result = (
                _projected_mapping(safe_result.get("vectorstore_result")) or {}
            )

            return BuildDatasetPdfVectorstoreOutput(
                success=True,
                vectorstore_path=retain_safe_result_path(
                    result.get("vectorstore_result", {}).get("save_path")
                    if isinstance(result.get("vectorstore_result"), dict)
                    else None
                ),
                num_documents_indexed=_projected_int(
                    vectorstore_result, "num_documents_indexed"
                ),
                num_texts=_projected_int(vectorstore_result, "num_texts"),
                num_images=_projected_int(vectorstore_result, "num_images"),
                embedding_dimension=_projected_int(
                    vectorstore_result, "embedding_dimension"
                ),
                extraction_result=_projected_mapping(
                    safe_result.get("extraction_result")
                ),
                split_result=_projected_mapping(safe_result.get("split_result")),
                raw_result=safe_result,
            )
        else:
            safe_result["error"] = _PDF_VECTORSTORE_BUILD_FAILURE_MESSAGE
            logger.error(_PDF_VECTORSTORE_BUILD_FAILURE_MESSAGE)
            return BuildDatasetPdfVectorstoreOutput(
                success=False,
                error=_PDF_VECTORSTORE_BUILD_FAILURE_MESSAGE,
                raw_result=safe_result,
            )

    except Exception:
        logger.error(_PDF_VECTORSTORE_BUILD_FAILURE_MESSAGE)
        return BuildDatasetPdfVectorstoreOutput(
            success=False,
            error=_PDF_VECTORSTORE_BUILD_FAILURE_MESSAGE,
        )


# ============================================================================
# Prepare Dataset API
# ============================================================================


@dataclass
class BuildDatasetPrepareDatasetInput:
    """Input parameters for prepare dataset API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        vector_store_override: Optional path to vector store (overrides config)
        dataset_override: Optional path to dataset directory (overrides config)
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    vector_store_override: Path | None = None
    dataset_override: Path | None = None
    verbose: bool = False

    def __post_init__(self):
        """Validate inputs."""
        # Handle config as either Path or dict
        if isinstance(self.config, dict):
            if not self.config:
                raise ValueError("Config dictionary cannot be empty")
        else:
            self.config = Path(self.config)
            if not path_exists_with_safe_diagnostics(
                self.config,
                label="prepare dataset configuration",
            ):
                raise FileNotFoundError("Config file not found")

        if self.vector_store_override:
            self.vector_store_override = Path(self.vector_store_override)

        if self.dataset_override:
            self.dataset_override = Path(self.dataset_override)


@dataclass(repr=False)
class BuildDatasetPrepareDatasetOutput(SecretSafeReprMixin, APIResult):
    """Output results from prepare dataset API."""

    dataset_jsonl_path: Path | None = None
    dataset_entries: list[dict[str, Any]] = field(default_factory=list)
    failed_models: list[str] = field(default_factory=list)
    raw_result: dict[str, Any] | None = None


async def abuild_dataset_prepare_dataset(
    params: BuildDatasetPrepareDatasetInput,
) -> BuildDatasetPrepareDatasetOutput:
    """Prepare dataset with CMF specifications for benchmark or prediction.

    This command prepares datasets by extracting CMF specifications
    for model numbers using a document vector store. It can prepare either
    benchmark datasets (with ground truth) or prediction datasets (without
    ground truth).

    Args:
        params: Prepare dataset input parameters

    Returns:
        BuildDatasetPrepareDatasetOutput with results or error information
    """
    logger.info("Starting prepare dataset via API")
    if isinstance(params.config, dict):
        logger.info("Using in-memory config dictionary")
    else:
        _log_diagnostic_path("Configuration file", params.config)

    if params.vector_store_override:
        _log_diagnostic_path("Vector store override", params.vector_store_override)
    if params.dataset_override:
        _log_diagnostic_path("Dataset override", params.dataset_override)

    try:
        # Import workflow factory
        from material_agent.workflows.factory import (
            create_prepare_dataset_workflow_from_config,
        )

        # Create config-driven workflow
        logger.info("Creating prepare dataset workflow...")
        workflow = create_prepare_dataset_workflow_from_config()

        # Prepare initial context with config and overrides
        initial_context: dict[str, Any] = {
            "vector_store_override": (
                str(params.vector_store_override)
                if params.vector_store_override
                else None
            ),
            "dataset_override": (
                str(params.dataset_override) if params.dataset_override else None
            ),
            "verbose": params.verbose,
        }

        # Add config as either path or dict
        if isinstance(params.config, dict):
            initial_context["config_dict"] = params.config
        else:
            initial_context["config_path"] = str(params.config)

        # Run the workflow
        logger.info("Running prepare dataset workflow...")
        result = await workflow.arun(initial_context=initial_context)
        safe_result = project_result_metadata(result)

        # Check if workflow completed successfully
        if result.get("dataset_entries") is not None:
            dataset_entries = _projected_mapping_list(
                safe_result.get("dataset_entries")
            )
            failed_models = _projected_string_list(safe_result.get("failed_models"))
            dataset_jsonl_path = retain_safe_result_path(
                result.get("dataset_jsonl_path")
            )

            logger.info(
                f"Dataset preparation completed: {len(dataset_entries)} entries, "
                f"{len(failed_models)} failed"
            )

            return BuildDatasetPrepareDatasetOutput(
                success=True,
                dataset_jsonl_path=dataset_jsonl_path,
                dataset_entries=dataset_entries,
                failed_models=failed_models,
                raw_result=safe_result,
            )
        else:
            error_msg = "Prepare dataset workflow did not complete successfully"
            safe_result["error"] = error_msg
            logger.error(error_msg)
            return BuildDatasetPrepareDatasetOutput(
                success=False,
                error=error_msg,
                raw_result=safe_result,
            )

    except Exception:
        logger.error(_PREPARE_DATASET_FAILURE_MESSAGE)
        return BuildDatasetPrepareDatasetOutput(
            success=False,
            error=_PREPARE_DATASET_FAILURE_MESSAGE,
        )


def build_dataset_usd(params: BuildDatasetUsdInput) -> BuildDatasetUsdOutput:
    """Build dataset from USD files synchronously."""
    return asyncio.run(abuild_dataset_usd(params))


def build_dataset_pdf_vectorstore(
    params: BuildDatasetPdfVectorstoreInput,
) -> BuildDatasetPdfVectorstoreOutput:
    """Build PDF vector store synchronously."""
    return asyncio.run(abuild_dataset_pdf_vectorstore(params))


def build_dataset_prepare_dataset(
    params: BuildDatasetPrepareDatasetInput,
) -> BuildDatasetPrepareDatasetOutput:
    """Prepare dataset synchronously."""
    return asyncio.run(abuild_dataset_prepare_dataset(params))
