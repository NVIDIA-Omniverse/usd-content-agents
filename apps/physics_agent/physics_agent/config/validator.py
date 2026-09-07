# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration validator for Physics Agent."""

import logging
import math
from typing import Any, cast

from world_understanding.agentic.config.unknown_keys import (
    build_nested_config_key_schema,
    warn_unknown_nested_config_keys,
)
from world_understanding.utils.credentials import redact_sensitive_config

from physics_agent.config.schema import (
    REQUIRED_FIELDS,
    REQUIRED_SECTIONS,
    STEP_ORDER,
    get_default_config,
    get_step_defaults,
)
from physics_agent.functions.mass_scale_quality import VALID_MASS_SCALE_POLICIES
from physics_agent.integrations.vomp_defaults import DEFAULT_VOMP_RENDER_CONFIG

logger = logging.getLogger(__name__)

# Allowed values for apply_physics.collision_approx. Shared with the
# per-step ConfigTask so both validation paths stay in sync.
VALID_COLLISION_APPROX = frozenset(
    {
        "convexHull",
        "convexDecomposition",
        "boundingCube",
        "boundingSphere",
        "meshSimplification",
        "none",
    }
)

_VOMP_ARTIFACT_KEYS = frozenset(
    {
        "config",
        "geometry_checkpoint_dir",
        "matvae_checkpoint_dir",
        "normalization_params_path",
    }
)
_VOMP_RENDER_KEYS = frozenset(DEFAULT_VOMP_RENDER_CONFIG)


def _is_hex_digest(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return cast(int, value)


class ConfigValidator:
    """Validator for Physics Agent configuration."""

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
        warn_unknown_nested_config_keys(
            config,
            key_schema,
            logger,
            strict_paths={("steps", step_name) for step_name in STEP_ORDER},
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
            allow_empty_predictions = step_config.get("allow_empty_predictions", False)
            if not isinstance(allow_empty_predictions, bool):
                raise ValueError(
                    "predict.allow_empty_predictions must be a boolean, got "
                    f"{type(allow_empty_predictions).__name__}"
                )

        elif step_name == "apply_physics":
            collision_approx = step_config.get("collision_approx", "convexHull")
            if collision_approx not in VALID_COLLISION_APPROX:
                raise ValueError(
                    "apply_physics.collision_approx must be one of "
                    f"{sorted(VALID_COLLISION_APPROX)}; got an unsupported value"
                )
            mass_scale_policy = step_config.get("mass_scale_policy", "skip_mass")
            if mass_scale_policy not in VALID_MASS_SCALE_POLICIES:
                raise ValueError(
                    "apply_physics.mass_scale_policy must be one of "
                    f"{sorted(VALID_MASS_SCALE_POLICIES)}; got an unsupported value"
                )
            allow_empty_predictions = step_config.get("allow_empty_predictions", False)
            if not isinstance(allow_empty_predictions, bool):
                raise ValueError(
                    "apply_physics.allow_empty_predictions must be a boolean, got "
                    f"{type(allow_empty_predictions).__name__}"
                )
            approved_dependency_roots = step_config.get("approved_dependency_roots")
            if approved_dependency_roots is not None and (
                not isinstance(approved_dependency_roots, list)
                or not approved_dependency_roots
                or not all(
                    isinstance(root, str) and root for root in approved_dependency_roots
                )
            ):
                raise ValueError(
                    "apply_physics.approved_dependency_roots must be a non-empty "
                    "list of paths"
                )

        elif step_name == "vomp_mass":
            for field in ("target_prim", "runtime_root"):
                value = step_config.get(field)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"vomp_mass.{field} must be a non-empty string")
            expected_revision = step_config.get("expected_revision")
            if not _is_hex_digest(expected_revision, length=40):
                raise ValueError(
                    "vomp_mass.expected_revision must be a full lowercase "
                    "40-character commit SHA"
                )
            artifact_hashes = step_config.get("expected_artifact_sha256")
            if (
                not isinstance(artifact_hashes, dict)
                or set(artifact_hashes) != _VOMP_ARTIFACT_KEYS
                or not all(
                    _is_hex_digest(value, length=64)
                    for value in artifact_hashes.values()
                )
            ):
                raise ValueError(
                    "vomp_mass.expected_artifact_sha256 must pin every required artifact"
                )
            if step_config.get("attention_backend") not in {
                "xformers",
                "sdpa",
                "naive",
            }:
                raise ValueError("vomp_mass.attention_backend is unsupported")
            try:
                timeout_seconds = float(step_config.get("timeout_seconds", 0.0))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("vomp_mass runtime limits must be numeric") from exc
            _require_positive_int(
                step_config.get("max_complete_voxels"),
                field="vomp_mass.max_complete_voxels",
            )
            if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
                raise ValueError("vomp_mass.timeout_seconds must be positive")
            render = step_config.get("render")
            if not isinstance(render, dict):
                raise ValueError("vomp_mass.render must be a mapping")
            unknown_render_keys = sorted(
                str(key) for key in render if key not in _VOMP_RENDER_KEYS
            )
            if unknown_render_keys:
                raise ValueError(
                    "vomp_mass.render contains unsupported key(s): "
                    + ", ".join(unknown_render_keys)
                )
            for field in (
                "num_views",
                "image_width",
                "image_height",
                "num_sensor_updates",
            ):
                _require_positive_int(
                    render.get(field),
                    field=f"vomp_mass.render.{field}",
                )
            render_scalars: dict[str, float] = {}
            for field in ("radius", "fov_degrees"):
                try:
                    value = float(render.get(field, 0.0))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"vomp_mass.render.{field} must be finite"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(f"vomp_mass.render.{field} must be finite")
                render_scalars[field] = value
            if render_scalars["radius"] <= 0.5:
                raise ValueError("vomp_mass.render.radius must be above 0.5")
            if not 1.0 < render_scalars["fov_degrees"] < 179.0:
                raise ValueError(
                    "vomp_mass.render.fov_degrees must be between 1 and 179"
                )
            if render.get("render_mode") not in {"rt1", "rt2", "pt"}:
                raise ValueError("vomp_mass.render.render_mode is unsupported")
            if render.get("material_target") not in {
                "auto",
                "preview_surface",
                "openpbr_materialx",
            }:
                raise ValueError("vomp_mass.render.material_target is unsupported")
