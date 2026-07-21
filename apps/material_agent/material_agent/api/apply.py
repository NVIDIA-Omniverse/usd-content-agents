# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply API for Material Agent."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from world_understanding.utils.result_projection import (
    project_result_metadata,
    retain_safe_result_path,
    retain_safe_result_text,
)
from world_understanding.utils.safe_repr import SecretSafeReprMixin

from material_agent.api.diagnostics import (
    diagnostic_path,
    normalize_required_config,
)
from material_agent.api.types import APIResult, AssignmentStats, DownloadStats

logger = logging.getLogger(__name__)

_APPLY_FAILURE_MESSAGE = "Material apply failed"


def _projected_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _projected_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if type(item) is str]


def _projected_mapping(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _projected_mapping_list(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, dict)]


@dataclass
class ApplyInput:
    """Input parameters for apply API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        input_usd_override: Optional path to override input USD from config
        predictions_override: Optional path to override predictions from config
        output_usd_override: Optional path to override output USD from config
        layer_only: Output layer only (not full stage)
        render_enabled: Enable rendering after apply
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    input_usd_override: Path | None = None
    predictions_override: Path | None = None
    output_usd_override: Path | None = None
    layer_only: bool = False
    render_enabled: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Validate inputs."""
        self.config = normalize_required_config(self.config)

        if self.input_usd_override:
            self.input_usd_override = Path(self.input_usd_override)

        if self.predictions_override:
            self.predictions_override = Path(self.predictions_override)

        if self.output_usd_override:
            self.output_usd_override = Path(self.output_usd_override)


@dataclass(repr=False)
class ApplyOutput(SecretSafeReprMixin, APIResult):
    """Output results from apply API."""

    output_usd_path: Path | None = None
    unique_materials: list[str] | None = None
    matched_materials: dict[str, list[Any]] | None = None
    resolved_materials: dict[str, str] | None = None
    materials_applied: dict[str, Any] | None = None
    material_profile_result: dict[str, Any] | None = None
    resolved_material_profile: str | None = None
    material_profile_warnings: list[dict[str, Any]] | None = None
    material_profile_errors: list[dict[str, Any]] | None = None
    assignment_stats: AssignmentStats | None = None
    download_stats: DownloadStats | None = None
    rendered_image_paths: list[Path] | None = None
    rendering_skipped: bool = True
    layer_only: bool = False
    raw_result: dict[str, Any] | None = None


async def arun_apply(params: ApplyInput) -> ApplyOutput:
    """Apply predicted materials to a USD file asynchronously.

    This is the core async implementation. The sync version delegates to this.
    This is equivalent to: pipeline --only apply

    Args:
        params: Apply input parameters

    Returns:
        ApplyOutput with results or error information
    """
    logger.info("Starting apply via API")
    if isinstance(params.config, dict):
        logger.info("Using in-memory config dictionary")
    else:
        logger.info("Configuration file: %s", diagnostic_path(params.config))

    if params.input_usd_override:
        logger.info(
            "Input USD override: %s",
            diagnostic_path(params.input_usd_override),
        )
    if params.predictions_override:
        logger.info(
            "Predictions override: %s",
            diagnostic_path(params.predictions_override),
        )
    if params.output_usd_override:
        logger.info(
            "Output USD override: %s",
            diagnostic_path(params.output_usd_override),
        )

    try:
        # Import the pipeline API to reuse logic
        from material_agent.api.pipeline import PipelineInput, arun_pipeline

        # Create pipeline params with only=apply
        pipeline_params = PipelineInput(
            config=params.config,
            skip_steps=[],
            only_steps=["apply"],
            resume=False,
            dry_run=False,
            clean=False,
            verbose=params.verbose,
        )

        # Run pipeline asynchronously
        pipeline_result = await arun_pipeline(pipeline_params)

        if pipeline_result.success:
            # Extract apply-specific results
            runtime_apply_result = pipeline_result.step_results.get("apply", {})
            if not isinstance(runtime_apply_result, dict):
                raise ValueError("Apply pipeline returned invalid result metadata")
            apply_result = project_result_metadata(runtime_apply_result)

            # Convert assignment stats
            assignment_stats_dict = (
                _projected_mapping(apply_result.get("assignment_stats")) or {}
            )
            assignment_stats = (
                AssignmentStats(
                    materials_created=_projected_int(
                        assignment_stats_dict, "materials_created"
                    ),
                    materials_applied=_projected_int(
                        assignment_stats_dict, "materials_applied"
                    ),
                    total_prims=_projected_int(assignment_stats_dict, "total_prims"),
                    failed=_projected_int(assignment_stats_dict, "failed"),
                    bound_prim_ids=_projected_string_list(
                        assignment_stats_dict.get("bound_prim_ids")
                    ),
                    unbound_prim_ids=_projected_string_list(
                        assignment_stats_dict.get("unbound_prim_ids")
                    ),
                )
                if assignment_stats_dict
                else None
            )

            # Convert download stats
            download_stats_dict = (
                _projected_mapping(apply_result.get("download_stats")) or {}
            )
            download_stats = (
                DownloadStats(
                    found_local=_projected_int(download_stats_dict, "found_local"),
                    downloaded=_projected_int(download_stats_dict, "downloaded"),
                    failed=_projected_int(download_stats_dict, "failed"),
                    skipped=_projected_int(download_stats_dict, "skipped"),
                )
                if download_stats_dict
                else None
            )

            # Convert rendered images to Paths
            rendered_image_paths = (
                [
                    safe_path
                    for value in runtime_apply_result.get("rendered_image_paths", [])
                    if (safe_path := retain_safe_result_path(value)) is not None
                ]
                if isinstance(runtime_apply_result.get("rendered_image_paths"), list)
                else None
            )
            if not rendered_image_paths:
                rendered_image_paths = None

            unique_materials = (
                _projected_string_list(apply_result.get("unique_materials"))
                if isinstance(apply_result.get("unique_materials"), list)
                else None
            )
            matched_materials_data = _projected_mapping(
                apply_result.get("matched_materials")
            )
            matched_materials = (
                {
                    key: value
                    for key, value in matched_materials_data.items()
                    if type(key) is str and isinstance(value, list)
                }
                if matched_materials_data is not None
                else None
            )
            resolved_materials_data = _projected_mapping(
                apply_result.get("resolved_materials")
            )
            resolved_materials = (
                {
                    key: value
                    for key, value in resolved_materials_data.items()
                    if type(key) is str and type(value) is str
                }
                if resolved_materials_data is not None
                else None
            )

            return ApplyOutput(
                success=True,
                output_usd_path=retain_safe_result_path(
                    runtime_apply_result.get("output_usd_path")
                ),
                unique_materials=unique_materials,
                matched_materials=matched_materials,
                resolved_materials=resolved_materials,
                materials_applied=_projected_mapping(
                    apply_result.get("materials_applied")
                ),
                material_profile_result=_projected_mapping(
                    apply_result.get("material_profile_result")
                ),
                resolved_material_profile=retain_safe_result_text(
                    apply_result.get("resolved_material_profile")
                ),
                material_profile_warnings=_projected_mapping_list(
                    apply_result.get("material_profile_warnings")
                ),
                material_profile_errors=_projected_mapping_list(
                    apply_result.get("material_profile_errors")
                ),
                assignment_stats=assignment_stats,
                download_stats=download_stats,
                rendered_image_paths=rendered_image_paths,
                rendering_skipped=(
                    apply_result.get("rendering_skipped")
                    if isinstance(apply_result.get("rendering_skipped"), bool)
                    else True
                ),
                layer_only=(
                    apply_result.get("layer_only")
                    if isinstance(apply_result.get("layer_only"), bool)
                    else False
                ),
                raw_result=apply_result,
            )
        else:
            return ApplyOutput(
                success=False,
                error=_APPLY_FAILURE_MESSAGE,
            )

    except Exception:
        logger.error(_APPLY_FAILURE_MESSAGE)
        return ApplyOutput(
            success=False,
            error=_APPLY_FAILURE_MESSAGE,
        )


def run_apply(params: ApplyInput) -> ApplyOutput:
    """Apply predicted materials to a USD file synchronously.

    This is a wrapper around the async implementation for backward compatibility.

    Args:
        params: Apply input parameters

    Returns:
        ApplyOutput with results or error information
    """
    return asyncio.run(arun_apply(params))


async def aapply(
    config: Path | dict[str, Any],
    input_usd_override: Path | None = None,
    predictions_override: Path | None = None,
    output_usd_override: Path | None = None,
    layer_only: bool = False,
    render_enabled: bool = False,
    verbose: bool = False,
) -> ApplyOutput:
    """Async convenience function for apply API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        input_usd_override: Optional path to override input USD from config
        predictions_override: Optional path to override predictions from config
        output_usd_override: Optional path to override output USD from config
        layer_only: Output layer only (not full stage)
        render_enabled: Enable rendering after apply
        verbose: Enable verbose output

    Returns:
        ApplyOutput with results
    """
    params = ApplyInput(
        config=config,
        input_usd_override=input_usd_override,
        predictions_override=predictions_override,
        output_usd_override=output_usd_override,
        layer_only=layer_only,
        render_enabled=render_enabled,
        verbose=verbose,
    )
    return await arun_apply(params)


def apply(
    config: Path | dict[str, Any],
    input_usd_override: Path | None = None,
    predictions_override: Path | None = None,
    output_usd_override: Path | None = None,
    layer_only: bool = False,
    render_enabled: bool = False,
    verbose: bool = False,
) -> ApplyOutput:
    """Sync convenience function for apply API.

    This delegates to the async version for implementation reuse.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        input_usd_override: Optional path to override input USD from config
        predictions_override: Optional path to override predictions from config
        output_usd_override: Optional path to override output USD from config
        layer_only: Output layer only (not full stage)
        render_enabled: Enable rendering after apply
        verbose: Enable verbose output

    Returns:
        ApplyOutput with results
    """
    return asyncio.run(
        aapply(
            config,
            input_usd_override,
            predictions_override,
            output_usd_override,
            layer_only,
            render_enabled,
            verbose,
        )
    )
