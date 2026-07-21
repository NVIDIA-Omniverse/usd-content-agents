# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configure API for Material Agent."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from world_understanding.utils.credentials import path_exists_with_safe_diagnostics
from world_understanding.utils.result_projection import (
    project_result_metadata,
    retain_safe_result_path,
    retain_safe_result_text,
)
from world_understanding.utils.safe_repr import SecretSafeReprMixin

from material_agent.api.diagnostics import diagnostic_path
from material_agent.api.types import APIResult

logger = logging.getLogger(__name__)

_CONFIGURE_FAILURE_MESSAGE = "Configuration creation failed"
_CONFIG_EXISTS_MESSAGE = "Configuration file already exists; use force=True"


def _raise_config_exists() -> NoReturn:
    """Raise the fixed conflict from a frame with no caller-owned paths."""
    raise FileExistsError(_CONFIG_EXISTS_MESSAGE) from None


@dataclass
class ConfigureInput:
    """Input parameters for configure API.

    Args:
        output_config_path: Path where the new config will be written
        materials_manifest: Path to materials manifest YAML file
        reference_images: List of reference image paths
        force: Overwrite existing file if it exists
        verbose: Enable verbose output

    Note:
        This API always generates a config file (not in-memory),
        so output_config_path is always required as a Path.
    """

    output_config_path: Path
    materials_manifest: Path | None = None
    reference_images: list[str] | None = None
    force: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Validate inputs."""
        self.output_config_path = Path(self.output_config_path)

        # Check if file already exists and force is not set
        if not self.force and path_exists_with_safe_diagnostics(
            self.output_config_path,
            label="output configuration file",
        ):
            del self
            _raise_config_exists()


@dataclass(repr=False)
class ConfigureOutput(SecretSafeReprMixin, APIResult):
    """Output results from configure API."""

    config_path: Path | None = None
    pipeline_name: str | None = None
    input_usd_path: str | None = None
    materials_library_path: str | None = None
    output_usd_path: str | None = None
    dataset_dir: str | None = None
    predictions_dir: str | None = None
    raw_result: dict[str, Any] | None = None


async def arun_configure(params: ConfigureInput) -> ConfigureOutput:
    """Create a new pipeline configuration file interactively.

    This command guides you through creating a pipeline configuration
    by asking for essential parameters and auto-populating the rest
    with sensible defaults.

    Args:
        params: Configure input parameters

    Returns:
        ConfigureOutput with results or error information
    """
    logger.info("Starting configuration creation via API")
    logger.info(
        "Output configuration path: %s",
        diagnostic_path(params.output_config_path),
    )

    workflow: Any = None
    initial_context: dict[str, Any] | None = None
    result: Any = None
    safe_result: dict[str, Any] | None = None
    config_exists = False
    try:
        # Import workflow factory
        from material_agent.workflows import create_configure_workflow

        # Create workflow
        logger.info("Creating configuration workflow...")
        workflow = create_configure_workflow()

        # Prepare initial context
        initial_context = {
            "output_config_path": str(params.output_config_path),
            "force": params.force,
            "verbose": params.verbose,
        }
        if params.materials_manifest:
            initial_context["materials_manifest"] = str(params.materials_manifest)
        if params.reference_images:
            initial_context["reference_images"] = params.reference_images

        # Run the workflow
        logger.info("Running configuration wizard...")
        result = await workflow.arun(initial_context=initial_context)
        safe_result = project_result_metadata(result)

        # Check if configuration was created successfully
        if result.get("config_created"):
            config_path = retain_safe_result_path(result.get("config_path"))
            pipeline_name = retain_safe_result_text(safe_result.get("pipeline_name"))
            input_usd_path = retain_safe_result_text(
                result.get("input_usd_path"), path_context=True
            )
            materials_library_path = retain_safe_result_text(
                result.get("materials_library_path"), path_context=True
            )
            output_usd_path = retain_safe_result_text(
                result.get("output_usd_path"), path_context=True
            )
            dataset_dir = retain_safe_result_text(
                result.get("dataset_dir"), path_context=True
            )
            predictions_dir = retain_safe_result_text(
                result.get("predictions_dir"), path_context=True
            )

            logger.info("Configuration file created successfully")

            return ConfigureOutput(
                success=True,
                config_path=config_path,
                pipeline_name=pipeline_name,
                input_usd_path=input_usd_path,
                materials_library_path=materials_library_path,
                output_usd_path=output_usd_path,
                dataset_dir=dataset_dir,
                predictions_dir=predictions_dir,
                raw_result=safe_result,
            )
        else:
            error_msg = "Configuration workflow did not complete successfully"
            logger.error(error_msg)
            return ConfigureOutput(
                success=False,
                error=error_msg,
                raw_result=safe_result,
            )

    except FileExistsError:
        config_exists = True
    except Exception:
        logger.error(_CONFIGURE_FAILURE_MESSAGE)
        return ConfigureOutput(
            success=False,
            error=_CONFIGURE_FAILURE_MESSAGE,
        )

    assert config_exists
    del initial_context, params, result, safe_result, workflow
    _raise_config_exists()


def run_configure(params: ConfigureInput) -> ConfigureOutput:
    """Create configuration synchronously.

    This is a wrapper around the async implementation for backward
    compatibility.

    Args:
        params: Configure input parameters

    Returns:
        ConfigureOutput with results or error information
    """
    return asyncio.run(arun_configure(params))


async def aconfigure(
    output_config_path: Path,
    materials_manifest: Path | None = None,
    reference_images: list[str] | None = None,
    force: bool = False,
    verbose: bool = False,
) -> ConfigureOutput:
    """Async convenience function for configure API.

    Args:
        output_config_path: Path where config will be saved
        materials_manifest: Path to materials manifest YAML file
        reference_images: List of reference image paths
        force: Overwrite existing file
        verbose: Enable verbose output

    Returns:
        ConfigureOutput with results
    """
    params = ConfigureInput(
        output_config_path=output_config_path,
        materials_manifest=materials_manifest,
        reference_images=reference_images,
        force=force,
        verbose=verbose,
    )
    return await arun_configure(params)


def configure(
    output_config_path: Path,
    materials_manifest: Path | None = None,
    reference_images: list[str] | None = None,
    force: bool = False,
    verbose: bool = False,
) -> ConfigureOutput:
    """Sync convenience function for configure API.

    This delegates to the async version for implementation reuse.

    Args:
        output_config_path: Path where config will be saved
        materials_manifest: Path to materials manifest YAML file
        reference_images: List of reference image paths
        force: Overwrite existing file
        verbose: Enable verbose output

    Returns:
        ConfigureOutput with results
    """
    return asyncio.run(
        aconfigure(
            output_config_path, materials_manifest, reference_images, force, verbose
        )
    )
