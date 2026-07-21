# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified configuration task for all Physics Agent operations.

This task replaces all individual config tasks with a single, unified approach.
It loads the configuration, validates it, resolves paths, and prepares everything
needed for pipeline execution.
"""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import (
    RendererConfig,
    load_config_mapping_from_context,
    log_config_source,
)
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    ensure_no_inline_secrets,
    redact_sensitive_path,
)

from physics_agent.api.defaults import PREDICT_DEFAULTS
from physics_agent.config.path_resolver import ProjectPathResolver
from physics_agent.config.schema import (
    STEP_ORDER,
    get_default_config,
    get_step_defaults,
)
from physics_agent.config.usd_suffixes import default_apply_physics_output_suffix
from physics_agent.config.validator import ConfigValidator

logger = logging.getLogger(__name__)


class UnifiedPipelineConfigTask(Task):
    """Unified config loader for all pipeline and step operations.

    This task handles:
    1. Loading and parsing YAML configuration
    2. Validating structure and conventions
    3. Resolving all paths automatically
    4. Building complete step configs with auto-wired paths

    The same task is used whether running:
    - Full pipeline: physics-agent run config.yaml
    - Single step: physics-agent predict config.yaml (equivalent to pipeline --only predict)
    """

    def __init__(self):
        """Initialize the unified config task."""
        self.name = "UnifiedConfigLoading"
        self.description = "Load and validate unified pipeline configuration"
        self.validator = ConfigValidator()

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load and validate unified configuration.

        Args:
            context: Workflow context containing:
                - config_path: Path to YAML config file
                - skip_steps: Optional list of steps to skip
                - only_steps: Optional list of steps to run exclusively
            object_store: Optional object store (not used)

        Returns:
            Updated context with:
                - config: Full configuration dictionary
                - path_resolver: ProjectPathResolver instance
                - steps_to_run: List of steps to execute
                - step_configs: Dictionary of configs for each step
                - project_name: Project name
                - working_dir: Working directory path

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If configuration file not found
        """
        config, config_path = self._load_config(context)
        log_config_source(context, logger.info, label="unified")

        # Merge with defaults
        config = self._merge_with_defaults(config)

        # Inject session_id from context if provided (for --session-id CLI option)
        if "session_id" in context and context["session_id"]:
            ensure_no_inline_secrets(
                context["session_id"],
                context="session identifier",
                path_context=True,
            )
            if "project" not in config:  # pragma: no cover - defaults ensure project
                config["project"] = {}
            config["project"]["session_id"] = context["session_id"]
            logger.debug(
                "Injected session_id from context: %s",
                redact_sensitive_path(context["session_id"]),
            )

        # Validate configuration
        try:
            self.validator.validate(config)
        except ValueError:
            logger.error("Configuration validation failed")
            raise

        # Create path resolver
        try:
            path_resolver = ProjectPathResolver(config, config_path)
            path_resolver.validate_input_paths()
        except (FileNotFoundError, ValueError):
            logger.error("Path resolution failed")
            raise

        # Determine which steps to run
        steps_to_run = self._determine_steps(config, context)

        # Build step configs with auto-wired paths
        step_configs = self._build_step_configs(steps_to_run, config, path_resolver)

        # Log configuration summary
        self._log_summary(config, path_resolver, steps_to_run)

        # Update context
        context.update(
            {
                "config": config,
                "path_resolver": path_resolver,
                "steps_to_run": steps_to_run,
                "step_configs": step_configs,
                "project_name": config["project"]["name"],
                "session_id": path_resolver.session_id,
                "working_dir": path_resolver.working_dir,
                "config_path": config_path,
            }
        )
        working_dir_base = getattr(path_resolver, "working_dir_base", None)
        if working_dir_base is not None:
            context["working_dir_base"] = working_dir_base

        return context

    def _load_config(self, context: dict[str, Any]) -> tuple[dict[str, Any], Path]:
        """Load an isolated unified config and its relative-path anchor."""
        return load_config_mapping_from_context(
            context,
            default_config_path=Path.cwd() / "config_dict.yaml",
            missing_path_message=(
                "Neither config_path nor config_dict provided in context"
            ),
            parse_error_message="Failed to parse YAML configuration: {config_path}",
            empty_message="Configuration is empty",
            config_dict_non_mapping_message="config_dict must be a mapping",
            file_non_mapping_message=(
                "Configuration must be a YAML mapping, got {type_name}"
            ),
        )

    def _merge_with_defaults(self, config: dict[str, Any]) -> dict[str, Any]:
        """Merge user config with defaults.

        Args:
            config: User configuration

        Returns:
            Merged configuration
        """
        defaults = get_default_config()

        # Merge top-level sections
        for section in ["project", "input", "advanced"]:
            # Handle case where section is missing or None (YAML with only comments)
            if section not in config or config[section] is None:
                config[section] = {}

            # Merge defaults into user config (user values take precedence)
            for key, value in defaults[section].items():
                if key not in config[section]:
                    config[section][key] = value

        # Ensure steps section exists
        if "steps" not in config or config["steps"] is None:
            config["steps"] = {}

        return config

    def _determine_steps(
        self, config: dict[str, Any], context: dict[str, Any]
    ) -> list[str]:
        """Determine which steps to run based on config and context.

        Args:
            config: Full configuration
            context: Workflow context with skip_steps/only_steps

        Returns:
            List of step names to execute
        """
        skip_steps = set(context.get("skip_steps", []))
        only_steps = context.get("only_steps", [])

        steps_config = config.get("steps") or {}
        steps_to_run = []

        for step_name in STEP_ORDER:
            step_config = steps_config.get(step_name, {})

            # Check if step is enabled in config
            # If 'enabled' is explicitly set, use that value
            # Otherwise, implicitly enable if step has configuration
            enabled = step_config.get("enabled")
            if enabled is None:
                # Implicitly enable if step has any configuration besides 'enabled'
                has_config = any(k != "enabled" for k in step_config.keys())
                enabled = has_config
                if has_config:
                    logger.debug(
                        "Step '%s' implicitly enabled (has configuration)", step_name
                    )

            if not enabled:
                logger.debug("Step '%s' is not enabled", step_name)
                continue

            # Apply skip filter
            if step_name in skip_steps:
                logger.info("Skipping step: %s (--skip)", step_name)
                continue

            # Apply only filter
            if only_steps and step_name not in only_steps:
                logger.debug("Skipping step: %s (not in --only)", step_name)
                continue

            steps_to_run.append(step_name)

        if not steps_to_run:
            raise ValueError(
                "No steps enabled in configuration. "
                "Please add step configuration in the 'steps' section. "
                "Steps are automatically enabled when configured, "
                "or you can explicitly set 'enabled: true'."
            )
        return steps_to_run

    def _build_step_configs(
        self,
        steps_to_run: list[str],
        config: dict[str, Any],
        path_resolver: ProjectPathResolver,
    ) -> dict[str, dict[str, Any]]:
        """Build complete configs for each step with auto-wired paths.

        Args:
            steps_to_run: List of steps to run
            config: Full configuration
            path_resolver: Path resolver instance

        Returns:
            Dictionary mapping step names to their complete configs
        """
        step_configs = {}
        steps_section = config.get("steps") or {}

        for step_name in steps_to_run:
            # Get step-specific config from user
            user_step_config = steps_section.get(step_name, {})

            # Merge with defaults
            step_config = self._merge_step_config(step_name, user_step_config)

            # Auto-wire paths based on step type
            step_config = self._autowire_paths(
                step_name, step_config, path_resolver, config
            )

            # Validate step requirements
            self.validator.validate_step_requirements(step_name, step_config, config)

            step_configs[step_name] = step_config

        return step_configs

    def _merge_step_config(
        self, step_name: str, user_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge user step config with defaults.

        Args:
            step_name: Name of the step
            user_config: User-provided step configuration

        Returns:
            Merged step configuration
        """
        defaults = get_step_defaults(step_name)
        return self._deep_merge(defaults, user_config)

    def _deep_merge(
        self, defaults: dict[str, Any], user_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Recursively merge user config into defaults.

        Args:
            defaults: Default configuration
            user_config: User-provided configuration

        Returns:
            Merged configuration with user values taking precedence
        """
        merged = defaults.copy()

        for key, value in user_config.items():
            if (
                isinstance(value, dict)
                and key in merged
                and isinstance(merged[key], dict)
            ):
                # Recursively merge nested dicts
                merged[key] = self._deep_merge(merged[key], value)
            else:
                # Overwrite with user value
                merged[key] = value

        return merged

    def _autowire_paths(
        self,
        step_name: str,
        step_config: dict[str, Any],
        path_resolver: ProjectPathResolver,
        full_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Auto-wire paths into step config.

        Args:
            step_name: Name of the step
            step_config: Step configuration
            path_resolver: Path resolver instance
            full_config: Full configuration

        Returns:
            Step configuration with auto-wired paths
        """
        if step_name == "optimize_usd":
            step_config["input_usd_path"] = str(path_resolver.input_usd)
            optimize_output_dir = path_resolver.get_step_output_dir("optimize_usd")
            step_config["output_usd_path"] = str(
                optimize_output_dir / (path_resolver.input_usd.stem + "_optimized.usdc")
            )
            self._normalize_optimize_usd_config(step_config)

        elif step_name == "build_dataset_usd":
            step_config["usd_path"] = str(path_resolver.input_usd)
            step_config["output_dir"] = str(path_resolver.get_usd_dataset_dir())

            # Parse rendering config with unified parser
            if (
                "renderer" in step_config
                and "rendering_modes" in step_config["renderer"]
            ):
                try:
                    # Create RendererConfig from step config
                    renderer_cfg = RendererConfig(**step_config["renderer"])

                    # Parse rendering modes using unified parser
                    rendering_modes_raw = step_config["renderer"]["rendering_modes"]
                    modes_config = renderer_cfg.get_rendering_modes_config(
                        rendering_modes_raw
                    )

                    # Store parsed config for future use
                    step_config["renderer"]["_rendering_modes_config"] = modes_config

                    # Get list of mode names for splitting RGB/sensor
                    mode_names = list(modes_config.keys())

                    # Define known sensor modes
                    sensor_modes_list = [
                        "linear_depth",
                        "depth",
                        "instance_id_segmentation",
                    ]

                    # Split modes into RGB and sensor categories
                    rgb_modes = []
                    sensor_modes = []
                    for mode in mode_names:
                        if mode in sensor_modes_list:
                            sensor_modes.append(mode)
                        else:
                            rgb_modes.append(mode)

                    # Add split modes to renderer config
                    step_config["renderer"]["rgb_rendering_modes"] = rgb_modes
                    step_config["renderer"]["sensor_rendering_modes"] = sensor_modes

                    logger.info(
                        "Parsed rendering config: %d modes (RGB=%d, Sensor=%d)",
                        len(mode_names),
                        len(rgb_modes),
                        len(sensor_modes),
                    )

                except ValueError:
                    logger.error("Failed to parse rendering config")
                    raise ValueError("Invalid renderer configuration") from None
                except Exception:
                    logger.error("Failed to create RendererConfig")
                    raise RuntimeError(
                        "Unable to create renderer configuration"
                    ) from None

        elif step_name == "identify_asset":
            step_config["usd_path"] = str(path_resolver.input_usd)
            step_config["output_dir"] = str(
                path_resolver.get_step_output_dir("identify_asset")
            )

        elif step_name == "build_dataset_prepare_dataset":
            step_config["usd_dir"] = str(path_resolver.get_usd_dataset_dir())
            step_config["dataset"] = str(path_resolver.get_dataset_dir())

            # Provide models list - use "." to indicate flat structure
            step_config["models"] = ["."]

            # Inject reference images
            if path_resolver.reference_images:
                step_config["reference_images"] = [
                    str(img) for img in path_resolver.reference_images
                ]

        elif step_name == "predict":
            step_config["dataset"] = str(
                path_resolver.get_step_dataset_file("build_dataset_prepare_dataset")
            )
            step_config["output_dir"] = str(path_resolver.get_predictions_dir())

            # Ensure output_key is set (configurable)
            if "output_key" not in step_config:
                step_config["output_key"] = PREDICT_DEFAULTS.get(
                    "output_key", "classification"
                )

        elif step_name == "restore_usd":
            # Wire original USD path and output path.
            # optimization_metadata and predictions_path are auto-wired at runtime
            # by the executor from previous step outputs.
            step_config["original_usd_path"] = str(path_resolver.input_usd)
            step_config["output_predictions_path"] = str(
                path_resolver.working_dir / "restored_predictions.jsonl"
            )

        elif step_name == "apply_physics":
            # Start from the original input. At runtime the executor rewires
            # this to optimize_usd.optimized_usd_path when optimization ran,
            # so physics authoring targets writable deinstanced prims.
            step_config["usd_path"] = str(path_resolver.input_usd)
            # Output USD goes into the physics step output dir
            physics_dir = path_resolver.get_step_output_dir("apply_physics")
            stem = path_resolver.input_usd.stem if path_resolver.input_usd else "output"
            if "output_usd_path" not in step_config:
                input_suffix = (
                    path_resolver.input_usd.suffix.lower()
                    if path_resolver.input_usd
                    else ".usd"
                )
                # USDZ packaging bundles referenced assets. Omniverse USDZs
                # often keep MDL shaders as runtime-resolved bare asset paths,
                # so unified-pipeline autowiring uses a USDA layer by default.
                suffix = default_apply_physics_output_suffix(input_suffix)
                step_config["output_usd_path"] = str(
                    physics_dir / f"{stem}_physics{suffix}"
                )
            # predictions_path is auto-wired at runtime by the executor.

        return step_config

    def _normalize_optimize_usd_config(self, step_config: dict[str, Any]) -> None:
        """Move public optimize_usd fields into the subtask config shape.

        Unified pipeline configs keep optimizer options directly under
        ``steps.optimize_usd`` for readability.  The reusable lower-level
        optimizer task reads those same fields from ``optimization_config``.
        Normalize here so API-generated and hand-authored configs behave the
        same way.
        """
        optimization_keys = {
            "api_key",
            "aws_vpc_mode",
            "backend",
            "base_url",
            "extract_geom_subset_indices",
            "flatten_prototypes",
            "max_retries",
            "poll_seconds",
            "s3_bucket",
            "s3_profile",
            "s3_region",
            "scene_optimizer_settings",
            "stage_timeout",
            "timeout",
            "wait_for_assets",
        }

        optimization_config = step_config.get("optimization_config")
        if isinstance(optimization_config, dict):
            normalized_config = dict(optimization_config)
        else:
            normalized_config = {}

        for key in optimization_keys:
            if key in step_config:
                incoming_value = step_config.pop(key)
                existing_value = normalized_config.get(key)
                if isinstance(existing_value, dict) and isinstance(
                    incoming_value, dict
                ):
                    normalized_config[key] = self._deep_merge(
                        existing_value, incoming_value
                    )
                else:
                    normalized_config[key] = incoming_value

        if normalized_config:
            step_config["optimization_config"] = normalized_config

    def _log_summary(
        self,
        config: dict[str, Any],
        path_resolver: ProjectPathResolver,
        steps_to_run: list[str],
    ) -> None:
        """Log configuration summary.

        Args:
            config: Full configuration
            path_resolver: Path resolver instance
            steps_to_run: List of steps to run
        """
        logger.info("=" * 70)
        logger.info("Configuration Summary")
        logger.info("=" * 70)
        logger.info("Project: %s", redact_sensitive_path(config["project"]["name"]))
        logger.info("")
        logger.info(
            "Session ID: %s",
            redact_sensitive_path(path_resolver.session_id),
        )
        logger.info("")
        if config["project"].get("description"):
            logger.info(
                "Description: %s",
                redact_sensitive_path(config["project"]["description"]),
            )
        logger.info(
            "Working directory: %s",
            redact_sensitive_path(path_resolver.working_dir),
        )
        logger.info("Input USD: %s", redact_sensitive_path(path_resolver.input_usd))

        logger.info("Steps to run: %s", ", ".join(steps_to_run))
        logger.info("=" * 70)
