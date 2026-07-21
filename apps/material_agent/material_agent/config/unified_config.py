# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified configuration task for all Material Agent operations.

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
    redact_sensitive_config,
    redact_sensitive_path,
    resolve_path_with_safe_diagnostics,
)

from material_agent.api.defaults import (
    ITERATION_DEFAULTS,
    PREDICT_DEFAULTS,
    apply_defaults,
)
from material_agent.config.path_resolver import ProjectPathResolver
from material_agent.config.schema import (
    STEP_ORDER,
    get_default_config,
    get_step_defaults,
)
from material_agent.config.validator import ConfigValidator
from material_agent.materials import (
    material_entries_with_fallback,
    material_mapping_with_fallback,
)
from material_agent.prompt_security import format_material_names_for_prompt
from material_agent.simready import (
    SimReadyCatalogError,
    build_material_entries,
    load_manifest,
)
from material_agent.tasks.prepare_dataset import (
    _TRUSTED_PREPARE_SYSTEM_PROMPT_TEMPLATE_CONFIG_KEY,
    render_system_prompt_from_prepare_config,
)

logger = logging.getLogger(__name__)


class UnifiedPipelineConfigTask(Task):
    """Unified config loader for all pipeline and step operations.

    This task handles:
    1. Loading and parsing YAML configuration
    2. Validating structure and conventions
    3. Resolving all paths automatically
    4. Injecting materials into appropriate steps
    5. Building complete step configs with auto-wired paths

    The same task is used whether running:
    - Full pipeline: material-agent pipeline config.yaml
    - Single step: material-agent predict config.yaml (equivalent to pipeline --only predict)
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
                - materials_data: Parsed materials information
                - steps_to_run: List of steps to execute
                - step_configs: Dictionary of configs for each step
                - project_name: Project name
                - working_dir: Working directory path

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If configuration file not found
        """
        config, config_path = load_config_mapping_from_context(
            context,
            default_config_path=Path.cwd() / "config_dict.yaml",
            missing_path_message="Neither config_path nor config_dict provided in context",
            parse_error_message="Failed to parse YAML configuration: {config_path}",
            empty_message="Configuration is empty",
            config_dict_non_mapping_message=(
                "config_dict must be a mapping, got {type_name}"
            ),
            file_non_mapping_message=(
                "Configuration must be a mapping, got {type_name}"
            ),
        )
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

        # Parse materials data
        materials_data = self._parse_materials(config, path_resolver)

        # Determine which steps to run
        steps_to_run = self._determine_steps(config, context)

        # Build step configs with auto-wired paths
        step_configs = self._build_step_configs(
            steps_to_run, config, path_resolver, materials_data
        )

        # Log configuration summary
        self._log_summary(config, path_resolver, materials_data, steps_to_run)

        # Update context
        context.update(
            {
                "config": config,
                "path_resolver": path_resolver,
                "materials_data": materials_data,
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

    def _merge_with_defaults(self, config: dict[str, Any]) -> dict[str, Any]:
        """Merge user config with defaults.

        Args:
            config: User configuration

        Returns:
            Merged configuration
        """
        defaults = get_default_config()

        # Merge top-level sections
        for section in ["project", "input", "output", "materials", "advanced"]:
            # Handle case where section is missing or None (YAML with only comments)
            if section not in config or config[section] is None:
                config[section] = {}

            # Special handling for materials section
            if section == "materials":
                # If user specified 'path' (external file), don't merge inline defaults
                if "path" in config["materials"]:
                    # Only add library_path default if not present
                    if "library_path" not in config["materials"]:
                        config["materials"]["library_path"] = None
                    # Skip entries default - will be loaded from external file
                    continue

            # Merge defaults into user config (user values take precedence)
            for key, value in defaults[section].items():
                if key not in config[section]:
                    config[section][key] = value

        # Ensure steps section exists
        if "steps" not in config or config["steps"] is None:
            config["steps"] = {}

        return config

    def _parse_materials(
        self, config: dict[str, Any], path_resolver: ProjectPathResolver
    ) -> dict[str, Any] | None:
        """Parse materials section from config.

        Supports two modes:
        1. External file: materials: {path: "materials/file.yaml"}
        2. Inline: materials: {library_path: "...", entries: [...]}

        Args:
            config: Full configuration
            path_resolver: Path resolver instance

        Returns:
            Parsed materials data or None if not present
        """
        materials_section = config.get("materials")
        if not materials_section:
            return None

        # Check if materials are referenced from external file
        materials_yaml_dir: Path | None = None
        if "path" in materials_section:
            materials_file_path = materials_section["path"]
            logger.info("Loading materials from external file")

            # Resolve path relative to config file
            resolved_path = path_resolver._resolve_path(materials_file_path)

            # Remember the materials YAML directory for resolving library_path
            materials_yaml_dir = resolved_path.parent

            materials_section, _ = load_config_mapping_from_context(
                {"config_path": resolved_path},
                missing_file_message="Materials file not found: {config_path}",
                parse_error_message="Failed to parse materials file: {config_path}",
                empty_message="Materials file is empty: {config_path}",
                file_non_mapping_message=(
                    "Materials file must contain a mapping, got {type_name}"
                ),
            )

            logger.info("Successfully loaded materials from external file")
        else:
            pass

        if "simready" in materials_section and not materials_section.get("entries"):
            return self._parse_simready_materials(
                materials_section["simready"],
                path_resolver,
            )

        # Now parse materials (whether from file or inline)
        if not materials_section.get("entries"):
            return None

        # Parse library path — resolve relative to the materials YAML file
        # when loaded from an external file, otherwise relative to the
        # pipeline config (via path_resolver).
        library_path = materials_section.get("library_path")
        if library_path:
            if materials_yaml_dir is not None:
                library_path = str(
                    resolve_path_with_safe_diagnostics(
                        materials_yaml_dir / library_path,
                        label="material library path",
                    )
                )
            else:
                library_path = str(path_resolver._resolve_path(library_path))

        # Parse entries
        entries = materials_section["entries"]

        logger.info("Loaded %d materials", len(entries))
        if library_path:
            logger.info(
                "Material library: %s",
                redact_sensitive_path(library_path),
            )

        parsed_materials = {
            "library_path": library_path,
            "entries": entries,
        }
        simready = materials_section.get("simready")
        if isinstance(simready, dict):
            parsed_materials["simready"] = dict(simready)
        return parsed_materials

    def _parse_simready_materials(
        self,
        simready_config: str | dict[str, Any],
        path_resolver: ProjectPathResolver,
    ) -> dict[str, Any]:
        """Parse a SimReady material-library shortcut from unified config."""
        if isinstance(simready_config, str):
            library_id = simready_config
            options: dict[str, Any] = {}
        elif isinstance(simready_config, dict):
            options = dict(simready_config)
            library_id = str(options.get("library_id") or "").strip()
        else:
            raise ValueError("'materials.simready' must be a string or mapping")

        if not library_id:
            raise ValueError("'materials.simready.library_id' is required")
        safe_library_id = str(redact_sensitive_config(library_id))

        manifest_path = options.get("manifest_path")
        if manifest_path:
            manifest_path = str(path_resolver._resolve_path(manifest_path))
        cache_dir = options.get("cache_dir")
        if cache_dir:
            cache_dir = str(path_resolver._resolve_path(cache_dir))
        else:
            cache_dir = str(path_resolver.working_dir / "simready-cache")

        allowed_categories = self._parse_simready_allowed_categories(
            options.get("allowed_categories")
        )
        split_archives_enabled = bool(options.get("split_archives_enabled", False))

        try:
            manifest = load_manifest(manifest_path)
            release_tag = options.get("release_tag")
            if release_tag and str(manifest.get("release_tag")) != str(release_tag):
                raise SimReadyCatalogError(
                    f"SimReady manifest release tag {manifest.get('release_tag')!r} "
                    f"does not match configured tag {release_tag!r}"
                )
            entries = build_material_entries(
                manifest,
                library_id,
                allowed_categories=allowed_categories,
                split_archives_enabled=split_archives_enabled,
            )
        except SimReadyCatalogError:
            raise ValueError(
                "Unable to load the configured SimReady material library"
            ) from None

        if not entries:
            raise ValueError(
                f"SimReady material library has no entries: {safe_library_id}"
            )

        simready = {
            "library_id": library_id,
            "release_tag": str(manifest.get("release_tag") or ""),
            "manifest_path": manifest_path,
            "cache_dir": cache_dir,
            "split_archives_enabled": split_archives_enabled,
        }
        if allowed_categories:
            simready["allowed_categories"] = sorted(allowed_categories)

        logger.info(
            "Loaded %d SimReady materials from %s",
            len(entries),
            safe_library_id,
        )
        return {
            "library_path": "",
            "entries": entries,
            "simready": simready,
        }

    def _parse_simready_allowed_categories(self, raw: Any) -> set[str] | None:
        if raw is None:
            return None
        if isinstance(raw, str):
            values = [item.strip() for item in raw.split(",")]
        elif isinstance(raw, list):
            values = [str(item).strip() for item in raw]
        else:
            raise ValueError(
                "'materials.simready.allowed_categories' must be a string or list"
            )
        categories = {item for item in values if item}
        return categories or None

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
            #
            # Special case: expand_cluster_predictions is auto-enabled when
            # cluster_prims is enabled, since it's a required dependent step
            # that propagates cluster representative predictions to all members.
            enabled = step_config.get("enabled")
            if enabled is None:
                if step_name == "expand_cluster_predictions":
                    # Implicit enable has two paths:
                    # 1. Step has its own config keys → enable it directly
                    #    (standalone use, e.g. predictions_path + cluster_map_path).
                    # 2. No own config → auto-enable when cluster_prims is enabled,
                    #    since expand_cluster_predictions is a required follow-up
                    #    that needs no configuration of its own.
                    has_own_config = any(k != "enabled" for k in step_config.keys())
                    if has_own_config:
                        enabled = True
                        logger.debug(
                            "Step 'expand_cluster_predictions' implicitly enabled "
                            "(has configuration)"
                        )
                    else:
                        cluster_config = steps_config.get("cluster_prims", {})
                        cluster_enabled = cluster_config.get("enabled")
                        if cluster_enabled is None:
                            cluster_enabled = any(
                                k != "enabled" for k in cluster_config.keys()
                            )
                        if cluster_enabled:
                            # Only auto-enable when a prediction step will also run.
                            # If neither predict nor benchmark is enabled, there will
                            # be no predictions file to expand.
                            def _is_step_enabled(name: str) -> bool:
                                cfg = steps_config.get(name, {})
                                explicit = cfg.get("enabled")
                                if explicit is not None:
                                    return bool(explicit)
                                return any(k != "enabled" for k in cfg.keys())

                            enabled = _is_step_enabled("predict") or _is_step_enabled(
                                "benchmark"
                            )
                        else:
                            enabled = False
                        if enabled:
                            logger.debug(
                                "Step 'expand_cluster_predictions' auto-enabled "
                                "(cluster_prims and a prediction step are enabled)"
                            )
                else:
                    # Implicitly enable if step has any configuration besides 'enabled'
                    has_config = any(k != "enabled" for k in step_config.keys())
                    enabled = has_config
                    if has_config:
                        logger.debug(
                            "Step '%s' implicitly enabled (has configuration)",
                            step_name,
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
                "Steps are automatically enabled when configured, or you can explicitly set 'enabled: true'."
            )

        return steps_to_run

    def _build_step_configs(
        self,
        steps_to_run: list[str],
        config: dict[str, Any],
        path_resolver: ProjectPathResolver,
        materials_data: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        """Build complete configs for each step with auto-wired paths.

        Args:
            steps_to_run: List of steps to run
            config: Full configuration
            path_resolver: Path resolver instance
            materials_data: Parsed materials data

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
                step_name, step_config, path_resolver, materials_data, config
            )

            if (
                step_name in ("predict", "benchmark")
                and "system_prompt" not in step_config
            ):
                prepare_config = step_configs.get("build_dataset_prepare_dataset")
                if prepare_config is not None:
                    step_config["system_prompt"] = (
                        render_system_prompt_from_prepare_config(prepare_config)
                    )
                elif step_name == "predict":
                    configured_prepare = steps_section.get(
                        "build_dataset_prepare_dataset", {}
                    )
                    configured_prompts = (
                        configured_prepare.get("prompts", {})
                        if isinstance(configured_prepare, dict)
                        else {}
                    )
                    if (
                        isinstance(configured_prompts, dict)
                        and "vlm_system" in configured_prompts
                    ):
                        # A predict-only run cannot execute the prepare step, but it
                        # can still authenticate a persisted custom prompt against
                        # the template from the trusted pipeline configuration. The
                        # predict loader rebuilds that prompt with the structured
                        # material names stored in dataset.json before accepting it.
                        step_config[
                            _TRUSTED_PREPARE_SYSTEM_PROMPT_TEMPLATE_CONFIG_KEY
                        ] = configured_prompts["vlm_system"]
                else:
                    configured_prepare = steps_section.get(
                        "build_dataset_prepare_dataset"
                    )
                    if isinstance(configured_prepare, dict):
                        # BenchmarkConfigTask does not reload dataset.json. Rebuild
                        # the omitted prepare step so a benchmark-only rerun receives
                        # the same trusted, material-scoped prompt as a full run.
                        benchmark_prepare_config = self._merge_step_config(
                            "build_dataset_prepare_dataset",
                            configured_prepare,
                        )
                        benchmark_prepare_config = self._autowire_paths(
                            "build_dataset_prepare_dataset",
                            benchmark_prepare_config,
                            path_resolver,
                            materials_data,
                            config,
                        )
                        step_config["system_prompt"] = (
                            render_system_prompt_from_prepare_config(
                                benchmark_prepare_config
                            )
                        )

            # Validate step requirements
            self.validator.validate_step_requirements(
                step_name, step_config, config, steps_to_run
            )

            step_configs[step_name] = step_config

            # After configuring optimize_usd, update path_resolver to use optimized file
            # for all subsequent tasks
            if step_name == "optimize_usd":
                original_input = path_resolver.input_usd
                path_resolver.input_usd = Path(step_config["output_usd_path"])
                logger.info(
                    "Auto-wiring: Updated input USD from %s to optimized %s for downstream tasks",
                    redact_sensitive_path(original_input),
                    redact_sensitive_path(path_resolver.input_usd),
                )

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
        materials_data: dict[str, Any] | None,
        full_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Auto-wire paths into step config.

        Args:
            step_name: Name of the step
            step_config: Step configuration
            path_resolver: Path resolver instance
            materials_data: Parsed materials data
            full_config: Full configuration

        Returns:
            Step configuration with auto-wired paths
        """
        if step_name == "validate_input":
            # Pre-validation: validate the original input USD
            step_config["input_usd_path"] = str(path_resolver.input_usd)
            step_config["output_dir"] = str(
                path_resolver.get_step_output_dir("validate_input")
            )
            if "validation_config" not in step_config:
                step_config["validation_config"] = {}

        elif step_name == "validate_output":
            # Post-validation: validate the output USD after material assignment
            step_config["input_usd_path"] = str(path_resolver.output_usd)
            step_config["output_dir"] = str(
                path_resolver.get_step_output_dir("validate_output")
            )
            if "validation_config" not in step_config:
                step_config["validation_config"] = {}
            # Reject on_failure=fix for validate_output (fix only applies to input)
            if step_config.get("on_failure") == "fix":
                raise ValueError(
                    "on_failure='fix' is not supported for validate_output. "
                    "Use 'warn' or 'block'."
                )
            # baseline_validation is injected at runtime by the executor

        elif step_name == "optimize_usd":
            # Input is the original USD file
            step_config["input_usd_path"] = str(path_resolver.input_usd)
            # Output is saved in the optimized directory
            optimized_dir = path_resolver.get_step_output_dir("optimize_usd")
            step_config["output_usd_path"] = str(optimized_dir / "optimized_input.usd")
            # Extract optimization_config if present in user config
            if "optimization_config" not in step_config:
                step_config["optimization_config"] = {}

        elif step_name == "render_preview":
            step_config["usd_path"] = str(path_resolver.input_usd)
            step_config["output_dir"] = str(
                path_resolver.get_step_output_dir("render_preview")
            )

        elif step_name == "identify_asset":
            step_config["usd_path"] = str(path_resolver.input_usd)
            step_config["output_dir"] = str(
                path_resolver.get_step_output_dir("identify_asset")
            )
            if "vlm" in step_config and "vlm_config" not in step_config:
                step_config["vlm_config"] = step_config["vlm"]
            if isinstance(step_config.get("vlm"), dict):
                step_config.pop("vlm", None)
            if "vlm_config" not in step_config:
                raise ValueError(
                    "steps.identify_asset.vlm is required when identify_asset "
                    "is enabled. Set backend and model explicitly."
                )
            if (
                not step_config.get("reference_images")
                and path_resolver.reference_images
            ):
                step_config["reference_images"] = [
                    str(img) for img in path_resolver.reference_images
                ]

        elif step_name == "generate_reference_image":
            step_config["output_dir"] = str(
                path_resolver.get_step_output_dir("generate_reference_image")
            )
            # Inject input.reference_images if the step doesn't already have them
            if (
                not step_config.get("reference_images")
                and path_resolver.reference_images
            ):
                step_config["reference_images"] = [
                    str(img) for img in path_resolver.reference_images
                ]

        elif step_name == "generate_material_library":
            step_config["input_usd_path"] = str(path_resolver.input_usd)
            step_config["output_dir"] = str(
                path_resolver.get_step_output_dir("generate_material_library")
            )

            plan_path = step_config.get("material_generation_plan_path")
            if plan_path:
                step_config["material_generation_plan_path"] = str(
                    path_resolver._resolve_path(plan_path)
                )

            if path_resolver.reference_images and not step_config.get(
                "reference_images"
            ):
                step_config["reference_images"] = [
                    str(img) for img in path_resolver.reference_images
                ]
            elif step_config.get("reference_images"):
                step_config["reference_images"] = [
                    str(path_resolver._resolve_path(path))
                    for path in step_config["reference_images"]
                ]

        elif step_name == "build_dataset_usd":
            step_config["usd_path"] = str(path_resolver.input_usd)
            step_config["output_dir"] = str(path_resolver.get_usd_dataset_dir())

            # Scope prim traversal to a subtree when prim_path is set
            if path_resolver.prim_path:
                prim_filters = step_config.setdefault("prim_filters", {})
                prim_filters["root_prim"] = path_resolver.prim_path

            # Parse rendering config with unified parser (supports both old and new formats!)
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

                    # IMPORTANT: Keep rendering_modes in original format (dict or list)
                    # The renderer's parse_camera_configuration() needs the dict format
                    # for per-mode camera settings. Don't convert dict to list!
                    # (rendering_modes_raw is already in step_config["renderer"]["rendering_modes"])

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

        elif step_name == "build_dataset_pdf_vectorstore":
            # Source is user-provided (external data)
            # Resolve it relative to config file
            if step_config.get("source"):
                step_config["source"] = str(
                    path_resolver._resolve_path(step_config["source"])
                )
            step_config["output_dir"] = str(path_resolver.get_vectorstore_dir())

        elif step_name == "build_dataset_prepare_dataset":
            step_config["usd_dir"] = str(path_resolver.get_usd_dataset_dir())
            # Note: PrepareDatasetConfigTask looks for "dataset", not "dataset_path"
            step_config["dataset"] = str(path_resolver.get_dataset_dir())

            # Provide models list - use "." to indicate flat structure
            # (usd_dir/. resolves to usd_dir itself)
            step_config["models"] = ["."]

            # Optional vectorstore (only if enabled)
            steps_config_section = full_config.get("steps") or {}
            if "build_dataset_pdf_vectorstore" in steps_config_section:
                if steps_config_section["build_dataset_pdf_vectorstore"].get("enabled"):
                    step_config["vector_store"] = str(
                        path_resolver.get_vectorstore_dir() / "vector_store"
                    )

            # Inject reference images
            if path_resolver.reference_images:
                step_config["reference_images"] = [
                    str(img) for img in path_resolver.reference_images
                ]

            # Inject reference PDFs
            if path_resolver.reference_pdfs:
                step_config["reference_pdfs"] = [
                    str(pdf) for pdf in path_resolver.reference_pdfs
                ]

            # Inject materials list
            if materials_data:
                step_config["materials_list"] = [
                    entry["name"]
                    for entry in material_entries_with_fallback(
                        materials_data["entries"]
                    )
                ]

                # Inject formatted materials for prompts
                if "prompts" in step_config:
                    materials_formatted = self._format_materials_for_prompt(
                        material_entries_with_fallback(materials_data["entries"])
                    )
                    step_config["_materials_formatted"] = materials_formatted

        elif step_name in ["predict", "benchmark"]:
            step_config["dataset"] = str(
                path_resolver.get_step_dataset_file("build_dataset_prepare_dataset")
            )
            step_config["output_dir"] = str(path_resolver.get_predictions_dir())

            # A trusted prepare-step prompt is injected by _build_step_configs when
            # both steps run together. Reused default dataset prompts are rebuilt and
            # completely validated by the predict task.
            # Remove legacy system_prompt_file if present in user config
            step_config.pop("system_prompt_file", None)

        elif step_name == "validate_predictions":
            step_config["predictions_path"] = str(
                path_resolver.get_step_predictions_file()
            )
            # Inject material names from materials_data
            if materials_data and "material_names" not in step_config:
                step_config["material_names"] = [
                    entry["name"]
                    for entry in material_entries_with_fallback(
                        materials_data["entries"]
                    )
                ]

        elif step_name == "harmonize_predictions":
            step_config["predictions_path"] = str(
                path_resolver.get_step_predictions_file()
            )
            # Inject material names from materials_data (same as validate_predictions)
            if materials_data and "material_names" not in step_config:
                step_config["material_names"] = [
                    entry["name"]
                    for entry in material_entries_with_fallback(
                        materials_data["entries"]
                    )
                ]

        elif step_name == "create_materials":
            step_config["source_usd"] = str(path_resolver.input_usd)
            config_dir = getattr(path_resolver, "config_dir", None)
            if config_dir is not None:
                step_config["_config_dir"] = str(config_dir)
            step_config["predictions_path"] = str(
                path_resolver.get_step_predictions_file()
            )
            output_dir = path_resolver.get_step_output_dir("create_materials")
            step_config["output_dir"] = str(output_dir)
            step_config["output_predictions_path"] = str(
                output_dir / "created_predictions.jsonl"
            )
            if not any(
                key in step_config
                for key in (
                    "material_profile",
                    "shader_target",
                    "material_authoring_target",
                )
            ):
                output_config = full_config.get("output") or {}
                step_config["material_profile"] = (
                    getattr(path_resolver, "material_profile", None)
                    or output_config.get("material_profile")
                    or output_config.get("shader_target")
                    or output_config.get("material_authoring_target")
                    or "auto"
                )

        elif step_name == "apply":
            step_config["input_usd_path"] = str(path_resolver.input_usd)
            step_config["predictions_path"] = str(
                path_resolver.get_step_predictions_file()
            )
            step_config["output_usd_path"] = str(path_resolver.output_usd)
            step_config["layer_only"] = path_resolver.layer_only
            step_config["flatten_output"] = path_resolver.flatten_output
            if not any(
                key in step_config
                for key in (
                    "material_profile",
                    "shader_target",
                    "material_authoring_target",
                )
            ):
                output_config = full_config.get("output") or {}
                step_config["material_profile"] = (
                    getattr(path_resolver, "material_profile", None)
                    or output_config.get("material_profile")
                    or output_config.get("shader_target")
                    or output_config.get("material_authoring_target")
                    or "auto"
                )

            # Inject materials mapping
            if materials_data:
                step_config["materials_mapping"] = self._build_materials_mapping(
                    materials_data
                )

        elif step_name == "refine":
            # Similar to apply but with iterations
            step_config["input_usd_path"] = str(path_resolver.input_usd)
            step_config["output_usd_path"] = str(path_resolver.output_usd)
            step_config["iterations_dir"] = str(
                path_resolver.get_step_output_dir("refine")
            )

            # Auto-wire dataset file from prepare_dataset step
            step_config["dataset"] = str(
                path_resolver.get_step_dataset_file("build_dataset_prepare_dataset")
            )

            # System prompt is stored in dataset.json (v0.2 format)
            # The refine task will read it from there
            # Remove legacy system_prompt_file from nested predict config if present
            if "predict" in step_config:
                step_config["predict"].pop("system_prompt_file", None)

            # Inject reference images for judge (if available)
            if path_resolver.reference_images:
                if "judge" not in step_config:
                    step_config["judge"] = {}
                step_config["judge"]["reference_images"] = [
                    str(img) for img in path_resolver.reference_images
                ]

            # Extract nested VLM configs to top level for ModelProvisioningTask
            # The ModelProvisioningTask expects vlm/llm_judge at the top level

            # If no predict section exists, create one with defaults for minimal configs
            if "predict" not in step_config or not step_config["predict"]:
                # Apply PREDICT_DEFAULTS to empty predict config for minimal configs
                step_config["predict"] = apply_defaults({}, PREDICT_DEFAULTS)

            if "predict" in step_config and "vlm" in step_config["predict"]:
                # Only override if top-level vlm is empty (from defaults)
                if not step_config.get("vlm") or step_config.get("vlm") == {}:
                    step_config["vlm"] = step_config["predict"]["vlm"]

            # NOTE: predict.llm (parsing LLM) is intentionally NOT hoisted to
            # top level.  ModelProvisioningTask would create it on every run,
            # but VLM responses already return structured output — the parsing
            # LLM is never invoked.  If a step needs it, it reads from its own
            # step config directly.

            if "predict" in step_config and "max_workers" in step_config["predict"]:
                if "max_workers" not in step_config:
                    step_config["max_workers"] = step_config["predict"]["max_workers"]

            # Extract iteration settings to top level
            if "iteration" in step_config:
                if "max_iterations" in step_config["iteration"]:
                    step_config["max_iterations"] = step_config["iteration"][
                        "max_iterations"
                    ]
                if "save_intermediate" in step_config["iteration"]:
                    step_config["save_intermediate"] = step_config["iteration"][
                        "save_intermediate"
                    ]

            # If no judge section exists, create one with defaults for minimal configs
            if "judge" not in step_config or not step_config["judge"]:
                # Apply judge defaults for minimal configs
                step_config["judge"] = apply_defaults({}, ITERATION_DEFAULTS["judge"])

            if "judge" in step_config and "vlm" in step_config["judge"]:
                # Judge VLM goes to vlm_judge for ModelProvisioningTask
                if (
                    not step_config.get("vlm_judge")
                    or step_config.get("vlm_judge") == {}
                ):
                    step_config["vlm_judge"] = step_config["judge"]["vlm"]
            elif "judge" in step_config:
                # If judge section exists but has no vlm, use judge defaults as llm_judge
                if (
                    not step_config.get("llm_judge")
                    or step_config.get("llm_judge") == {}
                ):
                    step_config["llm_judge"] = step_config["judge"]

            # Inject materials into nested apply config
            if "apply" not in step_config:
                step_config["apply"] = {}
            if not any(
                key in step_config["apply"]
                for key in (
                    "material_profile",
                    "shader_target",
                    "material_authoring_target",
                )
            ):
                output_config = full_config.get("output") or {}
                step_config["apply"]["material_profile"] = (
                    getattr(path_resolver, "material_profile", None)
                    or output_config.get("material_profile")
                    or output_config.get("shader_target")
                    or output_config.get("material_authoring_target")
                    or "auto"
                )

            if materials_data:
                step_config["apply"]["materials_mapping"] = (
                    self._build_materials_mapping(materials_data)
                )

        elif step_name == "restore_usd":
            # Input is the output from apply step
            step_config["input_usd_path"] = str(path_resolver.output_usd)

            # Output goes to restored directory
            restored_dir = path_resolver.get_step_output_dir("restore_usd")
            step_config["output_usd_path"] = str(restored_dir / "restored_output.usd")

            # optimization_metadata will be injected at runtime by executor
            # restore_config comes from user config
            if "restore_config" not in step_config:
                step_config["restore_config"] = {}

        elif step_name == "render":
            # Render step takes a USD and renders it
            # Input USD can be from apply step output or explicitly specified
            # If not specified, it will be auto-wired from previous step outputs
            # by UnifiedPipelineExecutorTask
            if "input_usd_path" not in step_config:
                # Default to output USD if not specified
                # (will be auto-wired by executor if apply step ran)
                step_config["input_usd_path"] = str(path_resolver.output_usd)
            else:
                # Resolve user-provided path
                step_config["input_usd_path"] = str(
                    path_resolver._resolve_path(step_config["input_usd_path"])
                )

            # Scope camera to prim_path when set
            if path_resolver.prim_path:
                step_config["prim_path"] = path_resolver.prim_path

            # Output path for rendered images - use same directory as output USD
            if "output_path" not in step_config:
                step_config["output_path"] = str(path_resolver.output_usd.parent)
            else:
                # Resolve user-provided path
                step_config["output_path"] = str(
                    path_resolver._resolve_path(step_config["output_path"])
                )

        return step_config

    def _build_materials_mapping(
        self, materials_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Build materials mapping from materials data.

        Args:
            materials_data: Parsed materials data

        Returns:
            Materials mapping dictionary
        """
        mapping = {}

        # Add library path if present
        if materials_data.get("library_path"):
            mapping["material_library_path"] = materials_data["library_path"]

        # Add name -> binding mappings
        for entry in material_entries_with_fallback(materials_data["entries"]):
            mapping[entry["name"]] = entry["binding"]

        return material_mapping_with_fallback(mapping)

    def _format_materials_for_prompt(self, entries: list[dict[str, Any]]) -> str:
        """Format material names as untrusted prompt data.

        Args:
            entries: List of material entries

        Returns:
            Formatted string for prompt substitution
        """
        return format_material_names_for_prompt(entries)

    def _log_summary(
        self,
        config: dict[str, Any],
        path_resolver: ProjectPathResolver,
        materials_data: dict[str, Any] | None,
        steps_to_run: list[str],
    ) -> None:
        """Log configuration summary.

        Args:
            config: Full configuration
            path_resolver: Path resolver instance
            materials_data: Parsed materials data
            steps_to_run: List of steps to run
        """
        logger.info("=" * 70)
        logger.info("Configuration Summary")
        logger.info("=" * 70)
        logger.info("Project: %s", redact_sensitive_path(config["project"]["name"]))
        logger.info("")
        logger.info(
            "🔑 Session ID: %s",
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
        logger.info(
            "Input USD: %s",
            (
                redact_sensitive_path(path_resolver.input_usd)
                if path_resolver.input_usd is not None
                else None
            ),
        )
        logger.info("Output USD: %s", redact_sensitive_path(path_resolver.output_usd))
        if path_resolver.output_usd:
            logger.info(
                "Output directory: %s",
                redact_sensitive_path(path_resolver.output_usd.parent),
            )

        if materials_data:
            logger.info("Materials: %d defined", len(materials_data["entries"]))
            if materials_data.get("library_path"):
                logger.info(
                    "  Library: %s",
                    redact_sensitive_path(materials_data["library_path"]),
                )

        logger.info("Steps to run: %s", ", ".join(steps_to_run))
        logger.info("=" * 70)
