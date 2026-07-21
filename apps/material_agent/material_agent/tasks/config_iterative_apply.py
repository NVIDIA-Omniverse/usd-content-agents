# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading task for iterative apply workflows.

NOTE: This is a compatibility shim for the old workflow system.
The unified config system (UnifiedPipelineConfigTask) is preferred.
"""

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
    path_exists_with_safe_diagnostics,
    read_text_with_safe_diagnostics,
    redact_sensitive_path,
    resolve_path_with_safe_diagnostics,
)

from material_agent.api.defaults import (
    ITERATION_DEFAULTS,
    PREDICT_DEFAULTS,
    apply_defaults,
)
from material_agent.materials import (
    material_entries_with_fallback,
    material_mapping_with_fallback,
)
from material_agent.tasks.config_loader import (
    load_config_from_context,
    resolve_config_relative_path,
)

logger = logging.getLogger(__name__)


class IterativeApplyConfigTask(Task):
    """Compatibility config task for iterative apply workflows."""

    def __init__(self):
        """Initialize the iterative apply config loading task."""
        self.name = "IterativeApplyConfigLoading"
        self.description = "Load iterative apply configuration from YAML file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Load iterative apply configuration.

        Args:
            context: Workflow context containing config_dict or config_path
            object_store: Optional object store (not used)

        Returns:
            Updated context with loaded configuration
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        config, config_path = load_config_from_context(context)
        log_config_source(context, listener.info, label="iterative apply")

        # Normalize the compatibility workflow's complete path contract while
        # the source anchor is available.  ``load_config_from_context`` returns
        # an isolated copy, so the caller-owned dictionary remains unchanged.
        for field in ("input_usd_path", "output_usd_path", "dataset"):
            if config.get(field):
                config[field] = resolve_config_relative_path(
                    config[field],
                    config_path,
                )

        # Pass through the config
        context["config"] = config
        context["input_usd_path"] = config.get("input_usd_path")
        context["output_usd_path"] = config.get("output_usd_path")
        # Set final_output_usd_path for IterativeApplyCompletionTask
        context["final_output_usd_path"] = config.get("output_usd_path")
        context["dataset_path"] = config.get("dataset")

        # Get iteration settings from nested 'iteration' section or top-level
        iteration_config = config.get("iteration", {})
        context["max_iterations"] = iteration_config.get(
            "max_iterations"
        ) or config.get("max_iterations", 5)
        save_intermediate = iteration_config.get("save_intermediate", True)
        context["save_intermediate"] = save_intermediate

        # Map iterations_dir / intermediate_dir to intermediate_output_dir
        iterations_dir = iteration_config.get("intermediate_dir") or config.get(
            "iterations_dir"
        )
        if iterations_dir:
            iterations_dir = resolve_config_relative_path(
                iterations_dir,
                config_path,
            )
            if iteration_config.get("intermediate_dir"):
                iteration_config["intermediate_dir"] = iterations_dir
            else:
                config["iterations_dir"] = iterations_dir
            context["intermediate_output_dir"] = iterations_dir
            context["iterations_dir"] = (
                iterations_dir  # Keep for backward compatibility
            )

        # Extract settings from nested predict config with defaults applied
        predict_config = config.get("predict", {})
        if predict_config.get("system_prompt_file"):
            predict_config["system_prompt_file"] = resolve_config_relative_path(
                predict_config["system_prompt_file"],
                config_path,
            )
        predict_config_with_defaults = apply_defaults(predict_config, PREDICT_DEFAULTS)

        context["vlm_config"] = predict_config_with_defaults.get("vlm", {})
        context["llm_config"] = predict_config_with_defaults.get("llm", {})
        context["max_workers"] = predict_config_with_defaults.get("max_workers", 64)
        context["prediction_batch_size"] = predict_config_with_defaults.get(
            "prediction_batch_size", 1
        )
        allow_empty_predictions = predict_config_with_defaults.get(
            "allow_empty_predictions", False
        )
        if not isinstance(allow_empty_predictions, bool):
            raise ValueError(
                "predict.allow_empty_predictions must be a boolean, got "
                f"{type(allow_empty_predictions).__name__}"
            )
        context["allow_empty_predictions"] = allow_empty_predictions

        # Add VLM and LLM configs to main config for ModelProvisioningTask
        config["vlm"] = predict_config_with_defaults.get("vlm", {})
        config["llm"] = predict_config_with_defaults.get("llm", {})

        # Load system prompt from file if system_prompt_file is provided
        system_prompt = predict_config.get("system_prompt")
        system_prompt_file = predict_config.get("system_prompt_file")

        if system_prompt_file and not system_prompt:
            # Load from file
            system_prompt_path = Path(system_prompt_file)
            if path_exists_with_safe_diagnostics(
                system_prompt_path,
                label="system prompt file",
            ):
                system_prompt = read_text_with_safe_diagnostics(
                    system_prompt_path,
                    label="system prompt file",
                )
                listener.info(
                    "Loaded system prompt from: "
                    f"{redact_sensitive_path(system_prompt_path)}"
                )
            else:
                listener.warning(
                    "System prompt file not found: "
                    f"{redact_sensitive_path(system_prompt_path)}, will use default"
                )

        # Store system prompt in both locations for compatibility
        context["system_prompt"] = system_prompt
        # VLMInferenceTask expects it here (context["config"] already set above)
        context["config"]["system_prompt"] = system_prompt

        # Extract report compression configuration from predict config
        report_config = predict_config.get("report", {})
        if isinstance(report_config, dict):
            if "image_max_size" in report_config:
                context["report_image_max_size"] = report_config["image_max_size"]
            if "image_format" in report_config:
                context["report_image_format"] = report_config["image_format"]
            if "image_quality" in report_config:
                context["report_image_quality"] = report_config["image_quality"]

        # Extract settings from nested apply config
        apply_config = config.get("apply", {})
        context["layer_only"] = apply_config.get("layer_only", False)
        context["flatten_output"] = apply_config.get("flatten_output", True)
        apply_allow_empty_predictions = apply_config.get(
            "allow_empty_predictions", False
        )
        if not isinstance(apply_allow_empty_predictions, bool):
            raise ValueError(
                "apply.allow_empty_predictions must be a boolean, got "
                f"{type(apply_allow_empty_predictions).__name__}"
            )
        context["apply_allow_empty_predictions"] = apply_allow_empty_predictions
        apply_fail_on_unknown_material = apply_config.get(
            "fail_on_unknown_material", False
        )
        if not isinstance(apply_fail_on_unknown_material, bool):
            raise ValueError(
                "apply.fail_on_unknown_material must be a boolean, got "
                f"{type(apply_fail_on_unknown_material).__name__}"
            )
        context["apply_fail_on_unknown_material"] = apply_fail_on_unknown_material
        context["aws_profile"] = apply_config.get("aws_profile")
        context["usd_search_config"] = apply_config.get("usd_search", {})

        # Build materials_mapping from top-level materials section or apply config
        materials_mapping = apply_config.get("materials_mapping", {})
        if not materials_mapping:
            materials_mapping = self._load_materials_mapping(
                config, config_path, listener
            )
        context["materials_mapping"] = materials_mapping

        # Extract render settings (now a sibling of apply, not nested within it)
        render_config = config.get("render", {})
        context["render_enabled"] = render_config.get("enabled", False)
        context["render_config"] = render_config

        # Extract judge settings with defaults applied
        judge_config = config.get("judge", {})
        reference_images = judge_config.get("reference_images", [])
        if isinstance(reference_images, list):
            judge_config["reference_images"] = [
                resolve_config_relative_path(reference_image, config_path)
                for reference_image in reference_images
            ]
        judge_config_with_defaults = apply_defaults(
            judge_config, ITERATION_DEFAULTS["judge"]
        )
        context["judge_config"] = judge_config_with_defaults
        context["reference_images"] = judge_config.get("reference_images", [])

        # Add judge config to main config for ModelProvisioningTask.
        # When the judge has a VLM, set vlm_judge; otherwise set llm_judge.
        if "vlm" in judge_config_with_defaults:
            config["vlm_judge"] = judge_config_with_defaults["vlm"]
        else:
            config["llm_judge"] = judge_config_with_defaults

        return context

    def _load_materials_mapping(
        self, config: dict[str, Any], config_path: Path, listener: Any
    ) -> dict[str, str]:
        """Load materials mapping from top-level materials section.

        Supports:
        - materials.path: Path to external materials YAML file
        - materials.library_path + materials.entries: Inline definition

        Args:
            config: Full configuration dictionary
            config_path: Path to the config file (for resolving relative paths)
            listener: Event listener for logging

        Returns:
            Dictionary mapping material names to bindings, plus
            material_library_path key for library-based materials
        """
        materials_config = config.get("materials", {})
        if not materials_config:
            return {}

        config_dir = config_path.parent

        # If materials.path points to external YAML, load it
        materials_path = materials_config.get("path")
        if materials_path:
            materials_yaml_path = Path(materials_path)
            if not materials_yaml_path.is_absolute():
                materials_yaml_path = config_dir / materials_yaml_path
            if path_exists_with_safe_diagnostics(
                materials_yaml_path,
                label="materials configuration",
            ):
                listener.info("Loading materials from file")
                materials_config, _ = load_config_mapping_from_context(
                    {"config_path": materials_yaml_path},
                    allow_empty=True,
                    parse_error_message=(
                        "Unable to parse materials configuration: {config_path}"
                    ),
                    file_non_mapping_message=(
                        "Materials configuration must contain a mapping, got "
                        "{type_name}"
                    ),
                )
                # Resolve library_path relative to the materials YAML
                if materials_config.get("library_path"):
                    lib_path = Path(materials_config["library_path"])
                    if not lib_path.is_absolute():
                        lib_path = materials_yaml_path.parent / lib_path
                    materials_config["library_path"] = str(
                        resolve_path_with_safe_diagnostics(
                            lib_path,
                            label="material library path",
                        )
                    )
            else:
                listener.warning("Materials file not found")
                return {}

        # Convert library_path + entries into materials_mapping dict
        library_path = materials_config.get("library_path")
        entries = materials_config.get("entries", [])
        if not library_path or not entries:
            return {}

        # Resolve library_path relative to config dir if needed
        if not Path(library_path).is_absolute():
            library_path = str(config_dir / library_path)

        mapping: dict[str, str] = {"material_library_path": library_path}
        for entry in material_entries_with_fallback(entries):
            name = entry.get("name", "")
            binding = entry.get("binding", "")
            if name and binding:
                mapping[name] = binding
        mapping = material_mapping_with_fallback(mapping)

        listener.info(
            f"Loaded {len(mapping) - 1} materials from library: "
            f"{Path(library_path).name}"
        )
        return mapping
