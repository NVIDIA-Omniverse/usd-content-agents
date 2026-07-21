# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration validator for Joint Agent."""

import logging
from typing import Any

from world_understanding.agentic.config.unknown_keys import (
    build_nested_config_key_schema,
    warn_unknown_nested_config_keys,
)
from world_understanding.utils.credentials import redact_sensitive_config

from joint_agent.config.schema import (
    REQUIRED_FIELDS,
    REQUIRED_SECTIONS,
    STEP_ORDER,
    SUPPORTED_PROMPT_PROFILES,
    get_default_config,
    get_step_defaults,
)
from joint_agent.joint_rigger_options import (
    DEFAULT_CANDIDATE_READINESS_POLICY,
    DEFAULT_JOINT_RIGGER_ADAPTER,
    DEFAULT_MISSING_DEPENDENCY_POLICY,
    SUPPORTED_CANDIDATE_READINESS_POLICIES,
    SUPPORTED_INTERNAL_JOINT_RIGGER_ADAPTERS,
    SUPPORTED_MISSING_DEPENDENCY_POLICIES,
    format_allowed_values,
)

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validator for Joint Agent configuration."""

    def validate(self, config: dict[str, Any]) -> None:
        """Validate the configuration structure.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required sections
        for section in REQUIRED_SECTIONS:
            if section not in config:
                raise ValueError(f"Missing required section: '{section}'")

        # Check required fields in each section
        for section, fields in REQUIRED_FIELDS.items():
            if section not in config:
                continue
            section_config = config[section]
            if section_config is None:
                section_config = {}
            for field in fields:
                if field not in section_config or section_config[field] is None:
                    raise ValueError(f"Missing required field: '{section}.{field}'")

        key_schema = build_nested_config_key_schema(
            get_default_config(),
            STEP_ORDER,
            get_step_defaults,
        )
        key_schema["steps"]["predict"]["report"] = {}
        # The structure-analysis step accepts ``llm`` as the legacy/API-facing
        # alias for its model configuration. Keep it in the diagnostic schema
        # without adding it to defaults, where a default ``vlm`` would take
        # precedence over a caller-provided ``llm`` during config merging.
        key_schema["steps"]["analyze_structure"]["llm"] = {}
        warn_unknown_nested_config_keys(
            config,
            key_schema,
            logger,
            strict_paths={
                *(("steps", step_name) for step_name in STEP_ORDER),
                ("steps", "infer_articulation_candidates", "adjudication"),
            },
        )

        # Validate steps section if present
        steps = config.get("steps", {})
        if steps:
            self._validate_steps(steps)

    def _validate_steps(self, steps: dict[str, Any]) -> None:
        """Validate steps configuration.

        Args:
            steps: Steps configuration dictionary
        """
        valid_steps = set(STEP_ORDER)

        for step_name in steps.keys():
            if step_name not in valid_steps:
                logger.warning(
                    "Unknown step '%s' in configuration. Valid steps: %s",
                    redact_sensitive_config(step_name),
                    ", ".join(sorted(valid_steps)),
                )

    def validate_step_requirements(
        self,
        step_name: str,
        step_config: dict[str, Any],
        full_config: dict[str, Any],
    ) -> None:
        """Validate requirements for a specific step.

        Args:
            step_name: Name of the step
            step_config: Step configuration
            full_config: Full configuration dictionary
        """
        # Step-specific validation
        if step_name == "predict":
            # Ensure VLM config is present
            if "vlm" not in step_config:
                logger.warning(
                    "predict step has no 'vlm' configuration - using defaults"
                )

            # Validate output_key if present
            output_key = step_config.get("output_key")
            if output_key and not isinstance(output_key, str):
                raise ValueError(
                    f"predict.output_key must be a string, got {type(output_key)}"
                )

            completion_retries = step_config.get("completion_retries", 3)
            if (
                isinstance(completion_retries, bool)
                or not isinstance(completion_retries, int)
                or completion_retries < 0
            ):
                raise ValueError(
                    "predict.completion_retries must be a non-negative integer"
                )

        if step_name == "build_dataset_prepare_dataset":
            prompt_profile = step_config.get("prompt_profile")
            if (
                prompt_profile is not None
                and prompt_profile not in SUPPORTED_PROMPT_PROFILES
            ):
                allowed = ", ".join(sorted(SUPPORTED_PROMPT_PROFILES))
                raise ValueError(
                    "build_dataset_prepare_dataset.prompt_profile must be one of: "
                    f"{allowed}"
                )

        if step_name == "consistency_pass":
            output_key = step_config.get("output_key")
            if output_key and not isinstance(output_key, str):
                raise ValueError(
                    "consistency_pass.output_key must be a string, "
                    f"got {type(output_key)}"
                )

            min_group_size = step_config.get("min_group_size", 2)
            if not isinstance(min_group_size, int) or min_group_size < 2:
                raise ValueError("consistency_pass.min_group_size must be >= 2")

            min_majority_fraction = step_config.get("min_majority_fraction", 0.6)
            if (
                isinstance(min_majority_fraction, bool)
                or not isinstance(min_majority_fraction, int | float)
                or not (0 < min_majority_fraction <= 1)
            ):
                raise ValueError(
                    "consistency_pass.min_majority_fraction must be in (0, 1]"
                )

            harmonize_motion_profiles = step_config.get(
                "harmonize_motion_profiles", False
            )
            if not isinstance(harmonize_motion_profiles, bool):
                raise ValueError(
                    "consistency_pass.harmonize_motion_profiles must be a boolean"
                )

        if step_name == "infer_articulation_candidates":
            output_key = step_config.get("output_key")
            if output_key is not None and not isinstance(output_key, str):
                raise ValueError(
                    "infer_articulation_candidates.output_key must be a string, "
                    f"got {type(output_key)}"
                )
            prim_metadata_path = step_config.get("prim_metadata_path")
            if prim_metadata_path is not None and not isinstance(
                prim_metadata_path, str
            ):
                raise ValueError(
                    "infer_articulation_candidates.prim_metadata_path must be a "
                    f"string, got {type(prim_metadata_path)}"
                )
            dataset_path = step_config.get("dataset_path")
            if dataset_path is not None and not isinstance(dataset_path, str):
                raise ValueError(
                    "infer_articulation_candidates.dataset_path must be a "
                    f"string, got {type(dataset_path)}"
                )
            output_adjudications_path = step_config.get("output_adjudications_path")
            if output_adjudications_path is not None and not isinstance(
                output_adjudications_path, str
            ):
                raise ValueError(
                    "infer_articulation_candidates.output_adjudications_path "
                    f"must be a string, got {type(output_adjudications_path)}"
                )

            candidate_joint_types = step_config.get("candidate_joint_types", [])
            if not isinstance(candidate_joint_types, list) or not all(
                isinstance(value, str) for value in candidate_joint_types
            ):
                raise ValueError(
                    "infer_articulation_candidates.candidate_joint_types "
                    "must be a list of strings"
                )
            adjudication = step_config.get("adjudication", {})
            if adjudication is not None and not isinstance(adjudication, dict):
                raise ValueError(
                    "infer_articulation_candidates.adjudication must be a dictionary"
                )
            if isinstance(adjudication, dict):
                enabled = adjudication.get("enabled", False)
                if not isinstance(enabled, bool):
                    raise ValueError(
                        "infer_articulation_candidates.adjudication.enabled "
                        "must be a boolean"
                    )
                reconcile_topology = adjudication.get("reconcile_topology", False)
                if not isinstance(reconcile_topology, bool):
                    raise ValueError(
                        "infer_articulation_candidates.adjudication."
                        "reconcile_topology must be a boolean"
                    )
                model_key = adjudication.get("model_key", "llm")
                if not isinstance(model_key, str):
                    raise ValueError(
                        "infer_articulation_candidates.adjudication.model_key "
                        "must be a string"
                    )
                if model_key not in {"llm", "vlm", "llm_judge", "vlm_judge"}:
                    raise ValueError(
                        "infer_articulation_candidates.adjudication.model_key "
                        "must be one of: llm, vlm, llm_judge, vlm_judge"
                    )
                min_confidence = adjudication.get("min_confidence", "high")
                if min_confidence not in {"high", "medium", "low"}:
                    raise ValueError(
                        "infer_articulation_candidates.adjudication.min_confidence "
                        "must be one of: high, medium, low"
                    )
                max_images = adjudication.get("max_images", 16)
                if (
                    isinstance(max_images, bool)
                    or not isinstance(max_images, int)
                    or max_images < 0
                ):
                    raise ValueError(
                        "infer_articulation_candidates.adjudication.max_images "
                        "must be a non-negative integer"
                    )
                max_adjudications = adjudication.get("max_adjudications", 8)
                if (
                    isinstance(max_adjudications, bool)
                    or not isinstance(max_adjudications, int)
                    or max_adjudications < 0
                ):
                    raise ValueError(
                        "infer_articulation_candidates.adjudication."
                        "max_adjudications must be a non-negative integer"
                    )
                require_source_images = adjudication.get("require_source_images", False)
                if not isinstance(require_source_images, bool):
                    raise ValueError(
                        "infer_articulation_candidates.adjudication."
                        "require_source_images must be a boolean"
                    )
                if require_source_images and model_key not in {"vlm", "vlm_judge"}:
                    raise ValueError(
                        "infer_articulation_candidates.adjudication."
                        "require_source_images requires model_key vlm or vlm_judge"
                    )
                if reconcile_topology and not enabled:
                    raise ValueError(
                        "infer_articulation_candidates.adjudication."
                        "reconcile_topology requires enabled: true"
                    )
                if reconcile_topology and model_key not in {"vlm", "vlm_judge"}:
                    raise ValueError(
                        "infer_articulation_candidates.adjudication."
                        "reconcile_topology requires model_key vlm or vlm_judge"
                    )
                if reconcile_topology and min_confidence != "high":
                    raise ValueError(
                        "infer_articulation_candidates.adjudication."
                        "reconcile_topology requires min_confidence: high"
                    )
                if reconcile_topology and not require_source_images:
                    raise ValueError(
                        "infer_articulation_candidates.adjudication."
                        "reconcile_topology requires require_source_images: true"
                    )
                if reconcile_topology and max_images == 0:
                    raise ValueError(
                        "infer_articulation_candidates.adjudication."
                        "reconcile_topology requires max_images greater than zero"
                    )
                temperature = adjudication.get("temperature", 0.0)
                if (
                    isinstance(temperature, bool)
                    or not isinstance(temperature, int | float)
                    or temperature < 0
                ):
                    raise ValueError(
                        "infer_articulation_candidates.adjudication.temperature "
                        "must be a non-negative number"
                    )
                max_tokens = adjudication.get("max_tokens", 4096)
                if (
                    isinstance(max_tokens, bool)
                    or not isinstance(max_tokens, int)
                    or max_tokens <= 0
                ):
                    raise ValueError(
                        "infer_articulation_candidates.adjudication.max_tokens "
                        "must be a positive integer"
                    )

        if step_name == "apply_joint_rigger":
            adapter = step_config.get("adapter", DEFAULT_JOINT_RIGGER_ADAPTER)
            if adapter not in SUPPORTED_INTERNAL_JOINT_RIGGER_ADAPTERS:
                raise ValueError(
                    "apply_joint_rigger.adapter must be one of: "
                    f"{format_allowed_values(SUPPORTED_INTERNAL_JOINT_RIGGER_ADAPTERS)}"
                )

            on_missing_dependency = step_config.get(
                "on_missing_dependency",
                DEFAULT_MISSING_DEPENDENCY_POLICY,
            )
            if on_missing_dependency not in SUPPORTED_MISSING_DEPENDENCY_POLICIES:
                raise ValueError(
                    "apply_joint_rigger.on_missing_dependency must be one of: "
                    f"{format_allowed_values(SUPPORTED_MISSING_DEPENDENCY_POLICIES)}"
                )
            on_unready_candidates = step_config.get(
                "on_unready_candidates",
                DEFAULT_CANDIDATE_READINESS_POLICY,
            )
            if on_unready_candidates not in SUPPORTED_CANDIDATE_READINESS_POLICIES:
                raise ValueError(
                    "apply_joint_rigger.on_unready_candidates must be one of: "
                    f"{format_allowed_values(SUPPORTED_CANDIDATE_READINESS_POLICIES)}"
                )

            for path_key in (
                "input_usd_path",
                "predictions_path",
                "articulation_candidates_path",
                "output_usd_path",
                "diagnostics_path",
                "validation_path",
            ):
                value = step_config.get(path_key)
                if value is not None and not isinstance(value, str):
                    raise ValueError(
                        f"apply_joint_rigger.{path_key} must be a string, "
                        f"got {type(value)}"
                    )
            if "joint_rigger_template" in step_config:
                joint_rigger_template = step_config["joint_rigger_template"]
                if (
                    not isinstance(joint_rigger_template, str)
                    or not joint_rigger_template.strip()
                ):
                    raise ValueError(
                        "apply_joint_rigger.joint_rigger_template must be a "
                        "non-empty string"
                    )
            for bool_key in ("apply_masses", "apply_collision"):
                if bool_key not in step_config:
                    continue
                value = step_config[bool_key]
                if not isinstance(value, bool):
                    raise ValueError(
                        f"apply_joint_rigger.{bool_key} must be a boolean, "
                        f"got {type(value)}"
                    )
            if adapter == "owned_core" and (
                step_config.get("apply_masses", False)
                or step_config.get("apply_collision", False)
            ):
                raise ValueError(
                    "apply_joint_rigger owned_core is topology-only; apply_masses "
                    "and apply_collision must both be false"
                )

        if step_name == "author_physics_schemas":
            for path_key in (
                "input_usd_path",
                "stage2_diagnostics_path",
                "stage2_validation_path",
                "authoring_plan_path",
                "output_usd_path",
                "diagnostics_path",
                "validation_path",
            ):
                value = step_config.get(path_key)
                if value is not None and not isinstance(value, str):
                    raise ValueError(
                        f"author_physics_schemas.{path_key} must be a string, "
                        f"got {type(value)}"
                    )
