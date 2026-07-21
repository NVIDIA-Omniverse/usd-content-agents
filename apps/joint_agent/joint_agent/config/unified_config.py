# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified configuration task for all Joint Agent operations.

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

from joint_agent.api.defaults import (
    DEFAULT_PROP_ARTICULATION_SYSTEM_PROMPT,
    DEFAULT_PROP_ARTICULATION_USER_PROMPT,
    DEFAULT_VLM_IMAGE_PROMPTS,
    PREDICT_DEFAULTS,
)
from joint_agent.config.model_aliases import normalize_analyze_structure_model_alias
from joint_agent.config.path_resolver import ProjectPathResolver
from joint_agent.config.schema import (
    PROMPT_PROFILE_PROP_ARTICULATION,
    STEP_ORDER,
    get_default_config,
    get_step_defaults,
)
from joint_agent.config.validator import ConfigValidator
from joint_agent.joint_rigger_options import (
    DEFAULT_JOINT_RIGGER_ADAPTER,
    PREDICTION_FREE_JOINT_RIGGER_ADAPTERS,
    PREDICTION_OPTIONAL_JOINT_RIGGER_ADAPTERS,
)

logger = logging.getLogger(__name__)


class UnifiedPipelineConfigTask(Task):
    """Unified config loader for all pipeline and step operations.

    This task handles:
    1. Loading and parsing YAML configuration
    2. Validating structure and conventions
    3. Resolving all paths automatically
    4. Building complete step configs with auto-wired paths

    The same task is used whether running:
    - Full pipeline: joint-agent run config.yaml
    - Single step: joint-agent predict config.yaml (equivalent to pipeline --only predict)
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

        # Merge with defaults
        config = self._merge_with_defaults(config)

        # Inject session_id from context if provided (for --session-id CLI option)
        if "session_id" in context and context["session_id"]:
            ensure_no_inline_secrets(
                context["session_id"],
                context="session identifier",
                path_context=True,
            )
            if "project" not in config:
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
        """Load an isolated config and retain its relative-path anchor."""
        config, config_path = load_config_mapping_from_context(
            context,
            default_config_path=Path.cwd() / "config_dict.yaml",
            allow_empty=False,
            missing_path_message=(
                "Neither config_path nor config_dict provided in context"
            ),
            parse_error_message=("Failed to parse YAML configuration: {config_path}"),
            empty_message="Configuration is empty",
            config_dict_non_mapping_message="config_dict must be a mapping",
            file_non_mapping_message="Configuration must be a mapping",
        )
        log_config_source(context, logger.info, label="unified")
        return config, config_path

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
            if step_name == "consistency_pass" and "output_key" not in user_step_config:
                predict_config = steps_section.get("predict", {})
                step_config["output_key"] = predict_config.get(
                    "output_key", PREDICT_DEFAULTS.get("output_key", "classification")
                )
            if (
                step_name == "infer_articulation_candidates"
                and "output_key" not in user_step_config
            ):
                predict_config = steps_section.get("predict", {})
                step_config["output_key"] = predict_config.get(
                    "output_key", PREDICT_DEFAULTS.get("output_key", "classification")
                )

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
        if step_name == "apply_joint_rigger" and "adapter" not in user_config:
            raise ValueError(
                "apply_joint_rigger.adapter is required when the step is selected"
            )
        if step_name == "analyze_structure":
            user_config = normalize_analyze_structure_model_alias(user_config)
        defaults = get_step_defaults(step_name)
        if (
            step_name == "apply_joint_rigger"
            and user_config.get("adapter") == "owned_core"
        ):
            for option in ("apply_masses", "apply_collision"):
                if option not in user_config:
                    defaults[option] = False
        if (
            step_name == "build_dataset_prepare_dataset"
            and user_config.get("prompt_profile") == PROMPT_PROFILE_PROP_ARTICULATION
        ):
            defaults["prompts"] = {
                "system": DEFAULT_PROP_ARTICULATION_SYSTEM_PROMPT,
                "user": DEFAULT_PROP_ARTICULATION_USER_PROMPT,
                "vlm_image_prompts": DEFAULT_VLM_IMAGE_PROMPTS.copy(),
            }
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

        elif step_name == "analyze_structure":
            step_config["usd_path"] = str(path_resolver.input_usd)
            step_config["output_dir"] = str(
                path_resolver.get_step_output_dir("analyze_structure")
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

        elif step_name == "consistency_pass":
            consistency_dir = path_resolver.get_step_output_dir("consistency_pass")
            if step_config.get("predictions_path"):
                predictions_path = path_resolver.resolve_config_path(
                    step_config["predictions_path"]
                )
                if predictions_path:
                    step_config["predictions_path"] = str(predictions_path)

            if step_config.get("output_predictions_path"):
                output_predictions_path = path_resolver.resolve_config_path(
                    step_config["output_predictions_path"]
                )
                if output_predictions_path:
                    step_config["output_predictions_path"] = str(
                        output_predictions_path
                    )
            else:
                output_predictions_path = (
                    consistency_dir / "consistent_predictions.jsonl"
                )
                step_config["output_predictions_path"] = str(output_predictions_path)
            if step_config.get("output_stats_path"):
                output_stats_path = path_resolver.resolve_config_path(
                    step_config["output_stats_path"]
                )
                if output_stats_path:
                    step_config["output_stats_path"] = str(output_stats_path)
            else:
                output_predictions_path = Path(step_config["output_predictions_path"])
                step_config["output_stats_path"] = str(
                    output_predictions_path.with_suffix(".stats.json")
                )

        elif step_name == "infer_articulation_candidates":
            candidates_dir = path_resolver.get_step_output_dir(
                "infer_articulation_candidates"
            )
            if step_config.get("predictions_path"):
                predictions_path = path_resolver.resolve_config_path(
                    step_config["predictions_path"]
                )
                if predictions_path:
                    step_config["predictions_path"] = str(predictions_path)
            if step_config.get("prim_metadata_path"):
                prim_metadata_path = path_resolver.resolve_config_path(
                    step_config["prim_metadata_path"]
                )
                if prim_metadata_path:
                    step_config["prim_metadata_path"] = str(prim_metadata_path)
            if step_config.get("dataset_path"):
                dataset_path = path_resolver.resolve_config_path(
                    step_config["dataset_path"]
                )
                if dataset_path:
                    step_config["dataset_path"] = str(dataset_path)

            if step_config.get("output_candidates_path"):
                output_candidates_path = path_resolver.resolve_config_path(
                    step_config["output_candidates_path"]
                )
                if output_candidates_path:
                    step_config["output_candidates_path"] = str(output_candidates_path)
            else:
                step_config["output_candidates_path"] = str(
                    candidates_dir / "articulation_candidates.json"
                )

            if step_config.get("output_report_path"):
                output_report_path = path_resolver.resolve_config_path(
                    step_config["output_report_path"]
                )
                if output_report_path:
                    step_config["output_report_path"] = str(output_report_path)
            else:
                output_candidates_path = Path(step_config["output_candidates_path"])
                step_config["output_report_path"] = str(
                    output_candidates_path.with_suffix(".html")
                )
            if step_config.get("output_adjudications_path"):
                output_adjudications_path = path_resolver.resolve_config_path(
                    step_config["output_adjudications_path"]
                )
                if output_adjudications_path:
                    step_config["output_adjudications_path"] = str(
                        output_adjudications_path
                    )

        elif step_name == "restore_usd":
            # Wire original USD path and output path.
            # optimization_metadata and predictions_path are auto-wired at runtime
            # by the executor from previous step outputs.
            step_config["original_usd_path"] = str(path_resolver.input_usd)
            step_config["output_predictions_path"] = str(
                path_resolver.working_dir / "restored_predictions.jsonl"
            )

        elif step_name == "apply_joint_rigger":
            joint_rigger_dir = path_resolver.get_step_output_dir("apply_joint_rigger")
            adapter = step_config.get("adapter", DEFAULT_JOINT_RIGGER_ADAPTER)

            if step_config.get("input_usd_path"):
                input_usd_path = path_resolver.resolve_config_path(
                    step_config["input_usd_path"]
                )
                if input_usd_path:
                    step_config["input_usd_path"] = str(input_usd_path)
            else:
                input_usd_path = path_resolver.input_usd
                step_config["input_usd_path"] = str(input_usd_path)
            suffix = input_usd_path.suffix if input_usd_path else ".usd"
            if not suffix:
                suffix = ".usd"

            if step_config.get("predictions_path"):
                predictions_path = path_resolver.resolve_config_path(
                    step_config["predictions_path"]
                )
                if predictions_path:
                    step_config["predictions_path"] = str(predictions_path)
            elif (
                adapter not in PREDICTION_FREE_JOINT_RIGGER_ADAPTERS
                and adapter not in PREDICTION_OPTIONAL_JOINT_RIGGER_ADAPTERS
            ):
                step_config["predictions_path"] = str(
                    path_resolver.working_dir / "restored_predictions.jsonl"
                )

            if step_config.get("articulation_candidates_path"):
                articulation_candidates_path = path_resolver.resolve_config_path(
                    step_config["articulation_candidates_path"]
                )
                if articulation_candidates_path:
                    step_config["articulation_candidates_path"] = str(
                        articulation_candidates_path
                    )
            elif adapter not in PREDICTION_FREE_JOINT_RIGGER_ADAPTERS:
                step_config["articulation_candidates_path"] = str(
                    path_resolver.get_step_output_dir("infer_articulation_candidates")
                    / "articulation_candidates.json"
                )

            if step_config.get("output_usd_path"):
                output_usd_path = path_resolver.resolve_config_path(
                    step_config["output_usd_path"]
                )
                if output_usd_path:
                    step_config["output_usd_path"] = str(output_usd_path)
            else:
                step_config["output_usd_path"] = str(
                    joint_rigger_dir / f"rigged{suffix}"
                )

            if step_config.get("diagnostics_path"):
                diagnostics_path = path_resolver.resolve_config_path(
                    step_config["diagnostics_path"]
                )
                if diagnostics_path:
                    step_config["diagnostics_path"] = str(diagnostics_path)
            else:
                step_config["diagnostics_path"] = str(
                    joint_rigger_dir / "joint_rigger_diagnostics.json"
                )

            if step_config.get("validation_path"):
                validation_path = path_resolver.resolve_config_path(
                    step_config["validation_path"]
                )
                if validation_path:
                    step_config["validation_path"] = str(validation_path)
            else:
                step_config["validation_path"] = str(
                    joint_rigger_dir / "joint_rigger_validation.json"
                )

        elif step_name == "author_physics_schemas":
            physics_dir = path_resolver.get_step_output_dir("author_physics_schemas")
            joint_rigger_dir = path_resolver.get_step_output_dir("apply_joint_rigger")
            apply_config = (full_config.get("steps") or {}).get(
                "apply_joint_rigger", {}
            )
            if step_config.get("input_usd_path"):
                input_usd_path = path_resolver.resolve_config_path(
                    step_config["input_usd_path"]
                )
            else:
                configured_rigged_path = apply_config.get("output_usd_path")
                input_usd_path = (
                    path_resolver.resolve_config_path(configured_rigged_path)
                    if configured_rigged_path
                    else None
                )
                if input_usd_path is None:
                    configured_rigger_input = apply_config.get("input_usd_path")
                    rigger_input_path = (
                        path_resolver.resolve_config_path(configured_rigger_input)
                        if configured_rigger_input
                        else path_resolver.input_usd
                    )
                    input_suffix = (
                        rigger_input_path.suffix if rigger_input_path else ".usd"
                    ) or ".usd"
                    input_usd_path = joint_rigger_dir / f"rigged{input_suffix}"
            if input_usd_path:
                step_config["input_usd_path"] = str(input_usd_path)
            output_suffix = input_usd_path.suffix if input_usd_path else ".usd"
            if not output_suffix:
                output_suffix = ".usd"

            configured_stage2_diagnostics = apply_config.get("diagnostics_path")
            default_stage2_diagnostics = (
                path_resolver.resolve_config_path(configured_stage2_diagnostics)
                if configured_stage2_diagnostics
                else joint_rigger_dir / "joint_rigger_diagnostics.json"
            )
            configured_stage2_validation = apply_config.get("validation_path")
            default_stage2_validation = (
                path_resolver.resolve_config_path(configured_stage2_validation)
                if configured_stage2_validation
                else joint_rigger_dir / "joint_rigger_validation.json"
            )
            for path_key, default_path in (
                ("stage2_diagnostics_path", default_stage2_diagnostics),
                ("stage2_validation_path", default_stage2_validation),
                ("authoring_plan_path", physics_dir / "authoring_plan.json"),
                (
                    "output_usd_path",
                    physics_dir / f"physics_ready{output_suffix}",
                ),
                ("diagnostics_path", physics_dir / "diagnostics.json"),
                ("validation_path", physics_dir / "validation.json"),
            ):
                configured_path = step_config.get(path_key)
                if configured_path:
                    resolved_path = path_resolver.resolve_config_path(configured_path)
                    if resolved_path:
                        step_config[path_key] = str(resolved_path)
                else:
                    step_config[path_key] = str(default_path)

        return step_config

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
