# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration validation for the unified config system.

This module validates that the unified configuration follows the required structure
and conventions. Most importantly, it ensures that no path configurations are
present in step configs (paths must be auto-derived).
"""

import logging
from collections.abc import Collection
from typing import Any

from world_understanding.agentic.config.unknown_keys import (
    build_nested_config_key_schema,
    warn_unknown_nested_config_keys,
)
from world_understanding.utils.credentials import (
    redact_sensitive_config,
    redact_sensitive_path,
)

from material_agent.config.schema import (
    MUTUALLY_EXCLUSIVE_STEPS,
    OPTIONAL_SECTIONS,
    REQUIRED_FIELDS,
    REQUIRED_SECTIONS,
    STEP_ORDER,
    get_default_config,
    get_step_defaults,
)

logger = logging.getLogger(__name__)

_OPEN_STEP_KEY_SCHEMAS = frozenset(
    {"build_dataset_prepare_dataset", "validate_predictions"}
)


class ConfigValidator:
    """Validates unified configuration structure and conventions.

    This validator ensures:
    1. Required sections and fields are present
    2. No path keys are present in step configs
    3. Materials are properly defined
    4. Mutually exclusive steps are not both enabled
    """

    # Path-like keys that are FORBIDDEN in step configs
    FORBIDDEN_PATH_KEYS = {
        "usd_path",
        "usd_dir",
        "output_dir",
        "dataset",
        "dataset_path",
        "predictions_path",
        "source_usd",
        "input_usd_path",
        "output_usd_path",
        "vector_store",
        "vector_store_path",
        "vectorstore",
    }

    # Exceptions: step-specific keys that are allowed (external sources)
    ALLOWED_EXTERNAL_SOURCES = {
        "build_dataset_pdf_vectorstore": {"source"},  # External PDF source
        "generate_material_library": {
            "material_generation_plan_path",
        },
    }

    # Behavior flags that contain "path" or "dir" in their names but are NOT path configs
    # These are boolean/config flags, not filesystem paths
    ALLOWED_PATH_LIKE_BEHAVIOR_FLAGS = {
        "include_prim_path_context",  # Boolean flag for including prim path in context
        "include_geometric_context",  # Boolean flag (could be confused with path)
    }

    def __init__(self) -> None:
        """Initialize the validator."""
        pass

    def validate(self, config: dict[str, Any]) -> None:
        """Validate the entire configuration.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_structure(config)
        key_schema = build_nested_config_key_schema(
            get_default_config(),
            STEP_ORDER,
            get_step_defaults,
        )
        key_schema["steps"]["predict"].update(
            {
                "report": {},
                "system_prompt": None,
                "system_prompt_file": None,
            }
        )
        warn_unknown_nested_config_keys(
            config,
            key_schema,
            logger,
            strict_paths={
                ("steps", step_name)
                for step_name in STEP_ORDER
                if step_name not in _OPEN_STEP_KEY_SCHEMAS
            },
        )
        self._validate_required_fields(config)
        self._validate_materials(config)
        self._validate_steps(config)
        self._validate_mutually_exclusive_steps(config)

        logger.info("Configuration validation passed")

    def _validate_structure(self, config: dict[str, Any]) -> None:
        """Validate top-level structure.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If structure is invalid
        """
        # Check for required sections
        for section in REQUIRED_SECTIONS:
            if section not in config:
                raise ValueError(
                    f"Missing required section: '{section}'\n"
                    f"The unified config must have: {', '.join(REQUIRED_SECTIONS)}"
                )

        # Check for unknown sections
        valid_sections = set(REQUIRED_SECTIONS + OPTIONAL_SECTIONS)
        for section in config.keys():
            if section not in valid_sections:
                logger.warning(
                    "Unknown config section: '%s' (will be ignored)",
                    redact_sensitive_config(section),
                )

    def _validate_required_fields(self, config: dict[str, Any]) -> None:
        """Validate required fields in each section.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If required fields are missing
        """
        for section, required_fields in REQUIRED_FIELDS.items():
            # Handle case where section exists but is None (YAML with only comments)
            section_config = config.get(section) or {}
            for field in required_fields:
                if field not in section_config or section_config[field] is None:
                    raise ValueError(
                        f"Missing required field: '{section}.{field}'\n"
                        f"This field must be specified in your config."
                    )

    def _validate_materials(self, config: dict[str, Any]) -> None:
        """Validate materials section.

        Supports two modes:
        1. External file: materials: {path: "materials/file.yaml"}
        2. Inline: materials: {library_path: "...", entries: [...]}

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If materials section is invalid
        """
        materials = config.get("materials")
        if not materials:
            logger.warning(
                "No 'materials' section found. This is required for predict/apply steps."
            )
            return

        if not isinstance(materials, dict):
            raise ValueError("'materials' section must be a dictionary")

        # Check if materials are referenced from external file
        if "path" in materials:
            # External file reference - just validate that path is a string
            if not isinstance(materials["path"], str):
                raise ValueError("'materials.path' must be a string")

            # Check that no inline entries are also specified
            if "entries" in materials:
                raise ValueError(
                    "Cannot specify both 'materials.path' and 'materials.entries'. "
                    "Use either external file (path) OR inline entries, not both."
                )

            logger.info(
                "Materials will be loaded from external file: %s",
                redact_sensitive_path(materials["path"]),
            )
            return

        if "simready" in materials and not materials.get("entries"):
            simready = materials["simready"]
            if isinstance(simready, str):
                if not simready.strip():
                    raise ValueError("'materials.simready' must not be empty")
                return
            if not isinstance(simready, dict):
                raise ValueError("'materials.simready' must be a string or mapping")
            library_id = simready.get("library_id")
            if not isinstance(library_id, str) or not library_id.strip():
                raise ValueError("'materials.simready.library_id' is required")
            return

        # Inline materials - validate entries
        entries = materials.get("entries", [])
        if not isinstance(entries, list):
            raise ValueError("'materials.entries' must be a list")

        if not entries:
            logger.warning("'materials.entries' is empty. No materials defined.")
            return

        # Validate each material entry
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Material entry {i} must be a dictionary")

            # Check required fields
            if "name" not in entry:
                raise ValueError(f"Material entry {i} missing 'name' field")

            if "binding" not in entry:
                raise ValueError(f"Material entry {i} missing 'binding' field")

        logger.info(f"Validated {len(entries)} material entries")

    def _validate_steps(self, config: dict[str, Any]) -> None:
        """Validate steps section.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If steps configuration is invalid
        """
        steps = config.get("steps") or {}
        if not isinstance(steps, dict):
            raise ValueError("'steps' section must be a dictionary")

        # Validate each step
        for step_name, step_config in steps.items():
            if step_name not in STEP_ORDER:
                logger.warning(
                    "Unknown step: '%s' (valid steps: %s)",
                    redact_sensitive_config(step_name),
                    ", ".join(STEP_ORDER),
                )
                continue

            if not isinstance(step_config, dict):
                raise ValueError(f"Step '{step_name}' config must be a dictionary")

            # Validate no forbidden path keys
            self._validate_no_path_keys(step_name, step_config)

    def _validate_no_path_keys(
        self, step_name: str, step_config: dict[str, Any]
    ) -> None:
        """Ensure no path keys in step config.

        Args:
            step_name: Name of the step
            step_config: Step configuration dictionary

        Raises:
            ValueError: If forbidden path keys are found
        """
        allowed_keys = self.ALLOWED_EXTERNAL_SOURCES.get(step_name, set())

        for key in step_config.keys():
            safe_key = redact_sensitive_config(key)
            # Check exact matches
            if key in self.FORBIDDEN_PATH_KEYS and key not in allowed_keys:
                raise ValueError(
                    f"Path configuration not allowed in step '{step_name}': "
                    f"'{safe_key}'\n\n"
                    f"In the unified config system, paths are auto-derived from:\n"
                    f"  - project.working_dir\n"
                    f"  - input.usd_path\n"
                    f"  - output.usd_path\n\n"
                    f"Please remove '{safe_key}' from steps.{step_name} configuration.\n"
                    f"Only behavior configuration is allowed in step configs."
                )

            # Check for path-like patterns
            path_patterns = ["_path", "_dir"]
            if any(pattern in key.lower() for pattern in path_patterns):
                # Skip if it's a known behavior flag (not an actual path config)
                if key in self.ALLOWED_PATH_LIKE_BEHAVIOR_FLAGS:
                    continue

                if key not in allowed_keys:
                    raise ValueError(
                        f"Suspicious path-like key in step '{step_name}': "
                        f"'{safe_key}'\n\n"
                        f"Only behavior configuration is allowed in step configs.\n"
                        f"Paths are auto-derived from project.working_dir.\n\n"
                        f"Please remove '{safe_key}' or move it to the appropriate "
                        "top-level section:\n"
                        f"  - input.usd_path for input USD files\n"
                        f"  - output.usd_path for output USD files\n"
                        f"  - project.working_dir for intermediate files location"
                    )

    def _validate_mutually_exclusive_steps(self, config: dict[str, Any]) -> None:
        """Validate that mutually exclusive steps are not both enabled.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If mutually exclusive steps are both enabled
        """
        steps = config.get("steps") or {}

        for exclusive_group in MUTUALLY_EXCLUSIVE_STEPS:
            enabled_steps = []
            for step_name in exclusive_group:
                step_config = steps.get(step_name, {})
                enabled = step_config.get("enabled")
                if enabled is None:
                    # Implicitly enable if step has any configuration besides 'enabled'
                    has_config = any(k != "enabled" for k in step_config.keys())
                    enabled = has_config
                if enabled:
                    enabled_steps.append(step_name)

            if len(enabled_steps) > 1:
                raise ValueError(
                    f"Mutually exclusive steps cannot both be enabled: "
                    f"{', '.join(enabled_steps)}\n"
                    f"Please enable only one of: {', '.join(exclusive_group)}"
                )

    def validate_step_requirements(
        self,
        step_name: str,
        step_config: dict[str, Any],
        config: dict[str, Any],
        current_steps: Collection[str] | None = None,
    ) -> None:
        """Validate that a specific step has all its requirements.

        Args:
            step_name: Name of the step
            step_config: Step configuration
            config: Full configuration
            current_steps: Steps scheduled for the current run, after filters

        Raises:
            ValueError: If step requirements are not met
        """
        # Check if materials are required but not defined
        if step_name in ["predict", "benchmark", "apply", "refine"]:
            materials = config.get("materials") or {}
            generated_materials = self._is_step_enabled(
                (config.get("steps") or {}).get("generate_material_library", {})
            )
            create_materials_enabled = self._is_step_enabled(
                (config.get("steps") or {}).get("create_materials", {})
            )
            create_materials_scheduled = (
                current_steps is None or "create_materials" in current_steps
            )
            created_materials = (
                step_name
                in [
                    "apply",
                    "refine",
                ]
                and create_materials_enabled
                and create_materials_scheduled
            )
            # Materials can be either external file (path) or inline (entries)
            has_materials = materials and (
                materials.get("path")
                or materials.get("entries")
                or materials.get("simready")
            )
            if not has_materials and not generated_materials and not created_materials:
                raise ValueError(
                    f"Step '{step_name}' requires materials to be defined.\n"
                    f"Please add a 'materials' section with either:\n"
                    f"  - path: 'materials/file.yaml' (external file)\n"
                    f"  - entries: [...] (inline definition)\n"
                    f"  - simready: simready-light"
                )

        # Check step-specific requirements
        if step_name == "build_dataset_pdf_vectorstore":
            if step_config.get("enabled"):
                if not step_config.get("source"):
                    raise ValueError(
                        f"Step '{step_name}' is enabled but missing required field 'source'\n"
                        f"Please specify the PDF source directory in steps.{step_name}.source"
                    )

        if step_name == "benchmark":
            if step_config.get("enabled"):
                if not step_config.get("llm_judge"):
                    raise ValueError(
                        f"Step '{step_name}' requires 'llm_judge' configuration"
                    )

        if step_name in {"predict", "apply", "benchmark"}:
            self._validate_allow_empty_predictions(step_name, step_config)
        if step_name == "apply":
            self._validate_fail_on_unknown_material(step_name, step_config)

        if step_name == "refine":
            apply_config = step_config.get("apply", {})
            if isinstance(apply_config, dict):
                self._validate_allow_empty_predictions("refine.apply", apply_config)
                self._validate_fail_on_unknown_material("refine.apply", apply_config)

        logger.debug(f"Step '{step_name}' requirements validated")

    def _is_step_enabled(self, step_config: dict[str, Any]) -> bool:
        enabled = step_config.get("enabled")
        if enabled is not None:
            return bool(enabled)
        return any(key != "enabled" for key in step_config)

    def _validate_allow_empty_predictions(
        self, step_name: str, config: dict[str, Any]
    ) -> None:
        value = config.get("allow_empty_predictions", False)
        if not isinstance(value, bool):
            raise ValueError(
                f"{step_name}.allow_empty_predictions must be a boolean, "
                f"got {type(value).__name__}"
            )

    def _validate_fail_on_unknown_material(
        self, step_name: str, config: dict[str, Any]
    ) -> None:
        value = config.get("fail_on_unknown_material", False)
        if not isinstance(value, bool):
            raise ValueError(
                f"{step_name}.fail_on_unknown_material must be a boolean, "
                f"got {type(value).__name__}"
            )
