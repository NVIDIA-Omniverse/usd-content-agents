# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD data preparation configuration task."""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import (
    load_config_mapping_from_context,
    log_config_source,
)
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    create_directory_with_safe_diagnostics,
    redact_sensitive_config,
    resolve_path_with_safe_diagnostics,
)
from world_understanding.utils.object_store import ObjectStore

from .defaults import USD_RENDERING_DEFAULTS

logger = logging.getLogger(__name__)


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"{field_name} must be a positive integer, got {type(value).__name__}"
        )
    return int(value)


def _strict_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean, got {type(value).__name__}")
    return value


class USDDataPrepConfigTask(Task):
    """Load and validate USD data preparation configuration from YAML."""

    def __init__(self) -> None:
        self.name = "USDDataPrepConfig"
        self.description = "Load and validate USD data preparation configuration"

    def run(
        self,
        context: dict[str, Any],
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """Load configuration and populate context for USD data preparation.

        Expected context inputs:
            - config_path: Path to YAML configuration file
            - config_dict: Inline configuration dictionary
            - source_override: Optional USD path override
            - output_dir_override: Optional output directory override
            - prim_filters: Optional filters for prim selection
            - extract_prim_metadata: Optional flag to extract metadata

        Updates context with:
            - usd_path: Path to USD file
            - output_dir: Output directory for dataset
            - render_output_dir: Directory for rendered images
            - dataset_output_dir: Directory for dataset manifest
            - prim_filters: Filters for prim selection
            - extract_metadata: Whether to extract prim metadata
            - renderer_config: Renderer configuration

        Path Resolution:
            - All relative paths in config file are treated as relative to
              the config file location
            - Command-line overrides are relative to the current working
              directory
            - Absolute paths are used as-is
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        config_path_value = context.get("config_path")
        missing_config_file = False
        if context.get("config_dict") is None and config_path_value:
            try:
                missing_config_file = not Path(config_path_value).exists()
            except OSError:
                # The shared loader will normalize the actual read failure.
                pass

        config, config_path = load_config_mapping_from_context(
            context,
            default_config_path=Path.cwd() / "config_dict.yaml",
            allow_empty=True,
            allow_missing_file=True,
            missing_path_message="Either config_path or config_dict must be provided",
            parse_error_message=("Invalid USD data preparation YAML: {config_path}"),
            config_dict_non_mapping_message=(
                "config_dict must be a dictionary when provided"
            ),
            file_non_mapping_message=(
                "USD data preparation config must be a mapping at the "
                "document root: {config_path}"
            ),
        )
        log_config_source(context, listener.info, label="USD")
        if missing_config_file:
            listener.warning("Configuration file not found; using defaults")

        # Get USD path (from override or config)
        source_override = context.get("source_override")
        if source_override:
            # Command-line overrides are relative to current directory
            usd_path = Path(source_override)
            if not usd_path.is_absolute():
                usd_path = resolve_path_with_safe_diagnostics(
                    usd_path,
                    label="USD source override",
                )
            listener.info(
                f"Using USD path override: {redact_sensitive_config(usd_path)}"
            )
        elif config.get("usd_path"):
            usd_path = Path(config["usd_path"])
            # Config paths are relative to config file location
            if not usd_path.is_absolute():
                usd_path = config_path.parent / usd_path
        else:
            # USD path is required - no default
            raise ValueError(
                "USD path not specified. Please provide 'usd_path' in the configuration file "
                "or use --usd-path command line option."
            )

        context["usd_path"] = usd_path

        # Get output directory (from override or config)
        output_dir_override = context.get("output_dir_override")
        if output_dir_override:
            # Command-line overrides are relative to current directory
            output_dir = Path(output_dir_override)
            if not output_dir.is_absolute():
                output_dir = resolve_path_with_safe_diagnostics(
                    output_dir,
                    label="USD output directory override",
                )
            listener.info(
                "Using output directory override: "
                f"{redact_sensitive_config(output_dir)}"
            )
        elif config.get("output_dir"):
            output_dir = Path(config["output_dir"])
            # Config paths are relative to config file location
            if not output_dir.is_absolute():
                output_dir = config_path.parent / output_dir
        else:
            # Default output dir is relative to config file location
            output_dir = config_path.parent / "output/dataset"

        context["output_dir"] = output_dir
        context["render_output_dir"] = output_dir / "renders"

        # Note: dataset files (dataset.json, prims.jsonl) will be saved
        # directly in output_dir, not in a subdirectory

        # Create output directories
        create_directory_with_safe_diagnostics(
            context["render_output_dir"],
            label="USD render output directory",
        )
        create_directory_with_safe_diagnostics(
            output_dir,
            label="USD dataset output directory",
        )

        # Get prim filters
        context["prim_filters"] = config.get(
            "prim_filters",
            context.get(
                "prim_filters",
                {
                    "types": ["UsdGeom.Mesh"],
                    "skip_instances": True,
                    "skip_prototypes": False,
                },
            ),
        )

        # Get metadata extraction flag
        context["extract_metadata"] = config.get(
            "extract_metadata", context.get("extract_prim_metadata", False)
        )

        # Get display color extraction flag
        context["extract_display_color"] = config.get(
            "extract_display_color", context.get("extract_display_color", False)
        )

        # Get display color statistics flag
        context["include_display_color_statistics"] = config.get(
            "include_display_color_statistics",
            context.get("include_display_color_statistics", False),
        )

        # Get material bindings extraction flag
        context["extract_material_bindings"] = config.get(
            "extract_material_bindings", context.get("extract_material_bindings", True)
        )

        # Get hierarchy extraction flag
        context["extract_hierarchy"] = config.get(
            "extract_hierarchy", context.get("extract_hierarchy", True)
        )

        # Get USD model building flag
        context["build_usd_model"] = config.get(
            "build_usd_model", context.get("build_usd_model", True)
        )

        # Get USD model export flag
        context["export_usd_model"] = config.get(
            "export_usd_model", context.get("export_usd_model", True)
        )

        for evidence_policy_key in (
            "fail_on_blank_dataset_renders",
            "fail_on_missing_prim_images",
        ):
            if evidence_policy_key in config:
                context[evidence_policy_key] = _strict_bool(
                    config[evidence_policy_key],
                    evidence_policy_key,
                )

        # Get skip existing flag (for resuming renders)
        # Check both "resume" (unified pipeline) and "skip_existing" (direct usage)
        # They mean the same thing: skip already rendered prims
        # Priority: context[skip_existing] > context[resume] > config[skip_existing] > False
        skip_existing = context.get(
            "skip_existing", context.get("resume", config.get("skip_existing", False))
        )
        context["skip_existing"] = skip_existing

        # Get skip existing materials flag
        # Filter prims with direct material bindings during traversal
        # Priority: context[skip_existing_materials] > config[skip_existing_materials] > False
        skip_existing_materials = context.get(
            "skip_existing_materials",
            config.get("skip_existing_materials", False),
        )
        context["skip_existing_materials"] = skip_existing_materials

        # Get batch size for rendering efficiency
        context["batch_size"] = _positive_int(
            config.get("batch_size", context.get("batch_size", 10)),
            "batch_size",
        )

        # Get async render request concurrency. This is separate from num_workers:
        # num_workers controls local task/thread parallelism, while this limits
        # simultaneous remote render requests in the async traversal path.
        context["max_concurrent_requests"] = _positive_int(
            config.get(
                "max_concurrent_requests", context.get("max_concurrent_requests", 128)
            ),
            "max_concurrent_requests",
        )

        # Get number of workers for parallel batch processing
        # Check both "max_workers" (unified pipeline) and "num_workers" (direct usage)
        context["num_workers"] = _positive_int(
            config.get(
                "num_workers", context.get("max_workers", context.get("num_workers", 1))
            ),
            "num_workers",
        )

        # Get renderer configuration - merge with defaults
        # Check context first (for unified pipeline), then config file
        renderer_config_override = context.get("renderer", config.get("renderer", {}))
        if not isinstance(renderer_config_override, dict):
            raise ValueError("renderer must be a mapping when provided")
        context["renderer_config"] = {
            **USD_RENDERING_DEFAULTS,  # Start with centralized defaults
            **renderer_config_override,  # Override with user config or context
        }
        for dimension_key in ("image_width", "image_height"):
            context["renderer_config"][dimension_key] = _positive_int(
                context["renderer_config"][dimension_key],
                f"renderer.{dimension_key}",
            )

        listener.info("USD configuration loaded:")
        listener.info(f"  USD path: {redact_sensitive_config(context['usd_path'])}")
        listener.info(
            f"  Output directory: {redact_sensitive_config(context['output_dir'])}"
        )
        listener.info(f"  Extract metadata: {context['extract_metadata']}")
        listener.info(
            f"  Extract display color: {context.get('extract_display_color', False)}"
        )
        listener.info(
            f"  Extract material bindings: {context['extract_material_bindings']}"
        )
        listener.info(f"  Extract hierarchy: {context['extract_hierarchy']}")
        listener.info(
            f"  Build USD model: {redact_sensitive_config(context['build_usd_model'])}"
        )
        listener.info(
            "  Export USD model: "
            f"{redact_sensitive_config(context['export_usd_model'])}"
        )
        listener.info(f"  Skip existing: {context['skip_existing']}")
        listener.info(f"  Batch size: {context['batch_size']}")
        listener.info(
            f"  Max concurrent requests: {context['max_concurrent_requests']}"
        )
        listener.info(f"  Number of workers: {context['num_workers']}")
        listener.info(
            "  Renderer backend: "
            f"{redact_sensitive_config(context['renderer_config']['backend'])}"
        )
        camera_type = context["renderer_config"].get("camera_view_type", "corner")
        listener.info(f"  Camera view type: {camera_type}")

        return context
