# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Central defaults and constants for Joint Agent.

This module defines all default values and constants used across the Joint Agent system.
Having everything in one place ensures consistency across CLI, API, and workflows.
"""

import logging
import os
from typing import Any

from world_understanding.functions.graphics.rendering_backend_factory import (
    validate_rendering_backend_name,
)
from world_understanding.utils.environment import parse_int_env

from joint_agent.joint_rigger_options import (
    DEFAULT_CANDIDATE_READINESS_POLICY,
    DEFAULT_JOINT_RIGGER_ADAPTER,
    DEFAULT_MISSING_DEPENDENCY_POLICY,
    DEFAULT_USD_JOINT_RIGGER_APPLY_COLLISION,
    DEFAULT_USD_JOINT_RIGGER_APPLY_MASSES,
    DEFAULT_USD_JOINT_RIGGER_TEMPLATE,
)
from joint_agent.prompts.prop_articulation import (
    render_prop_articulation_system_prompt,
    render_prop_articulation_user_prompt,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Pipeline Step Names (Constants)
# ============================================================================

# All valid pipeline step names in execution order
PIPELINE_STEP_NAMES = [
    "optimize_usd",
    "identify_asset",
    "analyze_structure",
    "build_dataset_usd",
    "build_dataset_prepare_dataset",
    "predict",
    "consistency_pass",
    "infer_articulation_candidates",
    "restore_usd",
    "apply_joint_rigger",
    "author_physics_schemas",
]

# Step name constants for type-safe references
STEP_OPTIMIZE_USD = "optimize_usd"
STEP_IDENTIFY_ASSET = "identify_asset"
STEP_ANALYZE_STRUCTURE = "analyze_structure"
STEP_BUILD_DATASET_USD = "build_dataset_usd"
STEP_BUILD_DATASET_PREPARE_DATASET = "build_dataset_prepare_dataset"
STEP_PREDICT = "predict"
STEP_CONSISTENCY_PASS = "consistency_pass"
STEP_INFER_ARTICULATION_CANDIDATES = "infer_articulation_candidates"
STEP_RESTORE_USD = "restore_usd"
STEP_APPLY_JOINT_RIGGER = "apply_joint_rigger"
STEP_AUTHOR_PHYSICS_SCHEMAS = "author_physics_schemas"


# ============================================================================
# Rendering Defaults
# ============================================================================

# Default camera directions for rendering (two opposite corners)
DEFAULT_CAMERA_DIRECTIONS = ["+x+y+z", "-x-y-z"]

# Dataset rendering defaults
DEFAULT_RENDER_BACKEND = os.environ.get("JA_RENDER_BACKEND", "remote")
DEFAULT_DATASET_IMAGE_SIZE = [512, 512]

# USD prim count warning threshold
DEFAULT_USD_PRIM_WARNING_THRESHOLD = 1000


# ============================================================================
# Model Defaults
# ============================================================================

DEFAULT_VLM_BACKEND = os.environ.get("JA_VLM_BACKEND", "nim")
DEFAULT_VLM_MODEL = os.environ.get("JA_VLM_MODEL", "qwen/qwen3.5-397b-a17b")
DEFAULT_VLM_TEMPERATURE = 1.0
DEFAULT_VLM_MAX_TOKENS = parse_int_env(
    "JA_VLM_MAX_TOKENS", 24576, minimum=1, logger=logger
)
DEFAULT_VLM_REASONING_EFFORT = "high"  # for reasoning-capable models (e.g. gpt-5)
DEFAULT_VLM_MAX_WORKERS = 64
DEFAULT_PREDICTION_COMPLETION_RETRIES = 3

DEFAULT_LLM_BACKEND = os.environ.get("JA_LLM_BACKEND", "nim")
DEFAULT_LLM_MODEL = os.environ.get("JA_LLM_MODEL", "qwen/qwen3.5-397b-a17b")
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_LLM_MAX_TOKENS = 512


# ============================================================================
# Prediction Defaults
# ============================================================================

PREDICT_DEFAULTS = {
    "vlm": {
        "backend": DEFAULT_VLM_BACKEND,
        "model": DEFAULT_VLM_MODEL,
        "temperature": DEFAULT_VLM_TEMPERATURE,
        "max_tokens": DEFAULT_VLM_MAX_TOKENS,
        "reasoning_effort": DEFAULT_VLM_REASONING_EFFORT,
    },
    # No separate parser LLM by default. Predict falls back to llm = vlm
    # unless the caller explicitly configures a dedicated llm backend/model.
    "max_workers": DEFAULT_VLM_MAX_WORKERS,
    "completion_retries": DEFAULT_PREDICTION_COMPLETION_RETRIES,
    "output_key": "classification",  # Configurable output key for predictions
}


# ============================================================================
# Identify Asset Defaults
# ============================================================================

IDENTIFY_ASSET_DEFAULTS: dict[str, Any] = {
    "renderer": {
        "backend": DEFAULT_RENDER_BACKEND,
        "image_width": 512,
        "image_height": 512,
        "cameras": ["+x+y+z", "-x+y+z", "-x-y+z", "+x-y+z"],
    },
    "vlm": {
        "backend": DEFAULT_VLM_BACKEND,
        "model": DEFAULT_VLM_MODEL,
        "temperature": 0.3,
        "max_tokens": 4096,
    },
    "prompts": {
        "system": (
            "You are an expert at identifying 3D objects from rendered images. "
            "Analyze the composition views and identify what this object is. "
            "Consider the overall shape, proportions, components visible, "
            "and functional purpose of the object.\n\n"
            "Respond with JSON:\n"
            '{"asset_type": "category (e.g., vehicle, tool, appliance, robot, '
            'furniture, industrial_equipment)", '
            '"asset_subtype": "specific type (e.g., forklift, drill, sedan)", '
            '"asset_description": "brief description of the object", '
            '"confidence": "high/medium/low", '
            '"reasoning": "explanation of identification"}'
        ),
    },
}


# ============================================================================
# Pipeline Defaults
# ============================================================================

PIPELINE_DEFAULTS = {
    "advanced": {
        "keep_temp_files": True,
        "log_level": "INFO",
    },
}


# ============================================================================
# Dataset Building Defaults
# ============================================================================

USD_DATASET_DEFAULTS = {
    "renderer": {
        "backend": DEFAULT_RENDER_BACKEND,
        "image_size": DEFAULT_DATASET_IMAGE_SIZE,
        "camera_view_type": "corner",
        "camera_directions": DEFAULT_CAMERA_DIRECTIONS,
        "num_views": 2,
        "rendering_modes": {
            "composition": {
                "margin": 6.0,
                "cameras": ["+x", "+y"],
                "camera_focus_mode": "stage",
                "skip_occluded_images": False,
                "use_original_materials": True,
            },
            "prim_only": {
                "margin": 1.2,
                "cameras": ["+x+y+z", "-x-y-z"],
                "camera_focus_mode": "prim",
                "use_original_materials": True,
            },
            "prim_only_original": {
                "base_mode": "prim_only",
                "margin": 1.2,
                "cameras": ["+x+y+z", "-x-y-z"],
                "camera_focus_mode": "prim",
                "use_original_materials": True,
            },
        },
        # batch_size/num_workers are tuned for local GPU backends (warp/ovrtx).
        # For NVCF, override with a smaller batch_size (e.g. 4) and more
        # num_workers (e.g. 32) to match NVCF's parallel-request model.
        "batch_size": 256,
        "num_workers": 1,
        "skip_existing": True,
    },
    "metadata": {
        "extract_metadata": True,
        "extract_display_color": False,
        "extract_hierarchy": True,
        "build_usd_model": True,
        "export_usd_model": True,
    },
}

# Default VLM image prompts for dataset preparation
PREPARE_DATASET_PROMPTS_DEFAULTS = {
    "vlm_image_prompts": {
        "composition": "This is a rendered part of interest highlighted with an orange outline, with the rest of the parts of the object rendered in muted colors.",
        "prim_only": "This is a rendered part of interest only without highlighting.",
        "prim_only_original": "This is a rendered part of interest only with original materials preserved.",
        # Default prompt for reference images
        "reference_images": "This is a reference image of the asset.",
    }
}


# ============================================================================
# Service-Level Pipeline Defaults (used by joint_agent_service)
# ============================================================================

# Generic fallback prompts for legacy/general Joint Agent configs. Service-level
# 0.5 prop-articulation defaults opt into the prop prompt constants below.
DEFAULT_SYSTEM_PROMPT = """\
You are an expert 3D asset analyst specializing in identifying components from
rendered images. Analyze the overall object context, the isolated component
views, and any available geometric context.

Respond with JSON containing:
{
  "asset_type": "what the overall object is",
  "component_type": "functional category of this component",
  "component_name": "descriptive name of what this part is",
  "material": "best-effort primary material if inferable",
  "confidence": "high|medium|low",
  "reasoning": "brief evidence-based explanation"
}

Answer format:
<reasoning>your analysis</reasoning>
<answer>your JSON</answer>
"""

DEFAULT_USER_PROMPT = """\
Please identify and classify this 3D component from the rendered images.
"""

DEFAULT_PROP_ARTICULATION_SYSTEM_PROMPT = render_prop_articulation_system_prompt()
DEFAULT_PROP_ARTICULATION_USER_PROMPT = render_prop_articulation_user_prompt()

DEFAULT_VLM_IMAGE_PROMPTS: dict[str, str] = {
    "composition": (
        "This is an orthographic view of the complete 3D model with the "
        "component highlighted, showing original materials and textures."
    ),
    "prim_only": "This is a rendered view of the isolated component only.",
    "prim_only_original": (
        "This is a rendered view of the isolated component with its original "
        "materials, textures, and colors preserved."
    ),
    "reference_images": "This is a reference image of the complete asset.",
}


def build_default_pipeline_config(
    session_id: str,
    usd_path: str,
    working_dir: str,
    user_prompt: str | None = None,
    render_backend: str | None = None,
    vlm_backend: str | None = None,
    vlm_model: str | None = None,
    llm_backend: str | None = None,
    llm_model: str | None = None,
    joint_rigger_output_suffix: str = ".usd",
) -> dict[str, Any]:
    """Build a complete pipeline config dict from server defaults.

    The joint-agent pipeline always includes structure analysis (LLM hierarchy
    analysis) for articulated bodies like robot arms.

    Args:
        session_id: Unique session identifier.
        usd_path: Absolute path to the input USD file.
        working_dir: Working directory for intermediate artifacts.
        user_prompt: Optional user prompt override.  When *None* the
            ``DEFAULT_USER_PROMPT`` is used.
        render_backend: Rendering backend to use ("remote", "warp", "ovrtx",
            or "mock").
            When *None*, uses ``DEFAULT_RENDER_BACKEND``.
        vlm_backend / vlm_model: Override the ``predict`` step VLM. When
            *None*, uses ``DEFAULT_VLM_BACKEND`` / ``DEFAULT_VLM_MODEL``.
        llm_backend / llm_model: Override the ``analyze_structure`` LLM
            (hierarchy naming). When *None*, uses ``DEFAULT_LLM_BACKEND``
            / ``DEFAULT_LLM_MODEL``.
        joint_rigger_output_suffix: USD-family suffix for the disabled-by-default
            Joint Rigger output. The public owned-core service uses ``.usdz``.

    Returns:
        Full pipeline configuration dictionary ready for ``execute_pipeline_async``.
    """
    backend = validate_rendering_backend_name(render_backend or DEFAULT_RENDER_BACKEND)
    if joint_rigger_output_suffix not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise ValueError(
            "joint_rigger_output_suffix must be one of: .usd, .usda, .usdc, .usdz"
        )

    vlm_backend = vlm_backend or DEFAULT_VLM_BACKEND
    vlm_model = vlm_model or DEFAULT_VLM_MODEL
    llm_backend = llm_backend or DEFAULT_LLM_BACKEND
    llm_model = llm_model or DEFAULT_LLM_MODEL

    config: dict[str, Any] = {
        "project": {
            "name": session_id,
            "session_id": session_id,
            "working_dir": working_dir,
        },
        "input": {
            "usd_path": usd_path,
        },
        "steps": {
            "optimize_usd": {
                "enabled": False,
            },
            "identify_asset": {
                "enabled": True,
                "renderer": {
                    "backend": backend,
                },
            },
            "analyze_structure": {
                "enabled": True,
                "strategy": "auto",
                "use_prompt_library": False,
                "robot_id": None,
                "llm": {
                    "backend": llm_backend,
                    "model": llm_model,
                    "temperature": DEFAULT_LLM_TEMPERATURE,
                    "max_tokens": DEFAULT_LLM_MAX_TOKENS,
                },
            },
            "build_dataset_usd": {
                "enabled": True,
                "renderer": {
                    "backend": backend,
                    "image_width": 512,
                    "image_height": 512,
                    "camera_view_type": "corner",
                    "rendering_modes": {
                        "composition": {
                            "margin": 6.0,
                            "cameras": ["+x", "+y"],
                            "camera_focus_mode": "stage",
                            "skip_occluded_images": False,
                            "use_original_materials": True,
                        },
                        "prim_only": {
                            "margin": 1.2,
                            "cameras": ["+x+y+z", "-x-y-z"],
                            "camera_focus_mode": "prim",
                            "use_original_materials": True,
                        },
                        "prim_only_original": {
                            "base_mode": "prim_only",
                            "margin": 1.2,
                            "cameras": ["+x+y+z", "-x-y-z"],
                            "camera_focus_mode": "prim",
                            "use_original_materials": True,
                        },
                    },
                    "should_highlight_prim": False,
                    "should_assign_random_colors": True,
                },
                "prim_filters": {
                    # Render the USD geometric-primitive base schema so Stage 1
                    # covers meshes and native Cube/Sphere/etc. geometry.
                    "types": ["UsdGeom.Gprim"],
                    "skip_instances": False,
                    "skip_prototypes": False,
                    # Stage 1 evidence excludes authored guide/proxy helpers,
                    # transparent registration geometry, and prims that cannot be
                    # framed by the focused cameras.
                    "allowed_purposes": ["default", "render"],
                    # Recover guide geometry only for a nearest authored rigid-body
                    # owner that would otherwise have no observable Stage 1 row.
                    # Guide helpers beside default/render geometry stay excluded.
                    "rigid_body_purpose_fallbacks": ["guide"],
                    "skip_fully_transparent": True,
                    "skip_unusable_bbox": True,
                },
                "extract_hierarchy": True,
                "extract_metadata": True,
                "skip_existing": True,
                # NVCF parallelises across HTTP requests; local GPU backends
                # benefit from large batches and a single worker process.
                "batch_size": 4 if backend == "remote" else 256,
                "num_workers": 32 if backend == "remote" else 1,
            },
            "build_dataset_prepare_dataset": {
                "enabled": True,
                "include_prim_path_context": True,
                "include_scene_prim_path_context": True,
                "scene_prim_path_limit": 80,
                "include_geometric_context": True,
                "include_repetition_context": True,
                "repetition_signature_depth": 2,
                "repetition_sibling_limit": 5,
                "prompts": {
                    "system": DEFAULT_PROP_ARTICULATION_SYSTEM_PROMPT,
                    "user": (
                        user_prompt
                        if user_prompt
                        else DEFAULT_PROP_ARTICULATION_USER_PROMPT
                    ),
                    "vlm_image_prompts": DEFAULT_VLM_IMAGE_PROMPTS.copy(),
                },
            },
            "predict": {
                "enabled": True,
                "vlm": {
                    "backend": vlm_backend,
                    "model": vlm_model,
                    "temperature": 0.3,
                    "max_tokens": DEFAULT_VLM_MAX_TOKENS,
                },
                "max_workers": DEFAULT_VLM_MAX_WORKERS,
                "completion_retries": DEFAULT_PREDICTION_COMPLETION_RETRIES,
                "output_key": "classification",
                "report": {
                    "image_max_size": 256,
                    "image_format": "jpeg",
                    "image_quality": 75,
                },
            },
            "consistency_pass": {
                "enabled": True,
                "output_key": "classification",
                "predictions_path": f"{working_dir}/predictions/predictions.jsonl",
                # The service artifact endpoint serves the post-consistency file;
                # the pass loads predictions before rewriting this path in place.
                "output_predictions_path": (
                    f"{working_dir}/predictions/predictions.jsonl"
                ),
                "output_stats_path": f"{working_dir}/predictions/predictions.stats.json",
                "min_group_size": 2,
                "min_majority_fraction": 0.6,
                "harmonize_fields": ["axis_hint"],
                "harmonize_motion_profiles": True,
                "add_role": True,
                "add_instance_id": True,
                "signature_depth": 2,
            },
            "infer_articulation_candidates": {
                "enabled": True,
                "output_key": "classification",
                "predictions_path": f"{working_dir}/predictions/predictions.jsonl",
                "output_candidates_path": (
                    f"{working_dir}/predictions/articulation_candidates.json"
                ),
                "output_report_path": (
                    f"{working_dir}/predictions/articulation_candidates.html"
                ),
                # Keep the stable service cache path explicit so regenerating
                # this step alone still has the Stage 1 source renders.
                "dataset_path": f"{working_dir}/dataset/dataset.jsonl",
                "output_adjudications_path": (
                    f"{working_dir}/predictions/"
                    "articulation_candidate_adjudications.json"
                ),
                "candidate_joint_types": ["revolute", "prismatic", "spherical"],
                # Fail-closed source-image review first attempts one complete,
                # validated physical-link reconciliation for structurally blocked
                # assets, then reruns the unchanged strict Stage 2 inference.
                "adjudication": {
                    "enabled": True,
                    "reconcile_topology": True,
                    "model_key": "vlm",
                    "min_confidence": "high",
                    "temperature": 0.0,
                    "max_tokens": 8192,
                    "max_images": 64,
                    "max_adjudications": 8,
                    "require_source_images": True,
                },
                "vlm": {
                    "backend": vlm_backend,
                    "model": vlm_model,
                },
            },
            "apply_joint_rigger": {
                "enabled": False,
                "adapter": DEFAULT_JOINT_RIGGER_ADAPTER,
                "on_missing_dependency": DEFAULT_MISSING_DEPENDENCY_POLICY,
                "on_unready_candidates": DEFAULT_CANDIDATE_READINESS_POLICY,
                "joint_rigger_template": DEFAULT_USD_JOINT_RIGGER_TEMPLATE,
                "apply_masses": DEFAULT_USD_JOINT_RIGGER_APPLY_MASSES,
                "apply_collision": DEFAULT_USD_JOINT_RIGGER_APPLY_COLLISION,
                "input_usd_path": usd_path,
                "articulation_candidates_path": (
                    f"{working_dir}/predictions/articulation_candidates.json"
                ),
                "output_usd_path": (
                    f"{working_dir}/joint_rigger/rigged{joint_rigger_output_suffix}"
                ),
                "diagnostics_path": (
                    f"{working_dir}/joint_rigger/joint_rigger_diagnostics.json"
                ),
                "validation_path": (
                    f"{working_dir}/joint_rigger/joint_rigger_validation.json"
                ),
            },
            "author_physics_schemas": {
                "enabled": False,
                "input_usd_path": (
                    f"{working_dir}/joint_rigger/rigged{joint_rigger_output_suffix}"
                ),
                "stage2_diagnostics_path": (
                    f"{working_dir}/joint_rigger/joint_rigger_diagnostics.json"
                ),
                "stage2_validation_path": (
                    f"{working_dir}/joint_rigger/joint_rigger_validation.json"
                ),
                "authoring_plan_path": (
                    f"{working_dir}/physics_authoring/authoring_plan.json"
                ),
                "output_usd_path": (
                    f"{working_dir}/physics_authoring/physics_ready.usd"
                ),
                "diagnostics_path": (
                    f"{working_dir}/physics_authoring/diagnostics.json"
                ),
                "validation_path": (f"{working_dir}/physics_authoring/validation.json"),
            },
        },
        "advanced": {
            "keep_temp_files": True,
            "log_level": "INFO",
        },
    }

    return config


# ============================================================================
# Helper Functions
# ============================================================================


def apply_defaults(config: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Apply default values to config recursively.

    Only adds defaults for missing keys, never overwrites user-provided values.

    Args:
        config: User configuration dictionary
        defaults: Default values to apply

    Returns:
        Config with defaults applied

    Example:
        >>> user_config = {"vlm": {"model": "qwen/qwen3.5-397b-a17b"}}
        >>> full_config = apply_defaults(user_config, PREDICT_DEFAULTS)
        >>> # Result:
        >>> # {"vlm": {"model": "qwen/qwen3.5-397b-a17b",
        >>> #          "backend": "nim", ...}}
    """
    result = config.copy()

    for key, default_value in defaults.items():
        if key not in result:
            # Key missing entirely - add default
            result[key] = default_value
        elif isinstance(default_value, dict) and isinstance(result[key], dict):
            # Both are dicts - recurse
            result[key] = apply_defaults(result[key], default_value)
        # else: user provided value - don't override

    return result


def get_predict_config_with_defaults(
    user_config: dict[str, Any],
) -> dict[str, Any]:
    """Get predict config with defaults applied.

    Args:
        user_config: Minimal user configuration

    Returns:
        Complete configuration with defaults

    Example:
        >>> minimal = {"vlm": {"model": "example-vlm-model"}, "dataset": "data.jsonl"}
        >>> full = get_predict_config_with_defaults(minimal)
        >>> # VLM backend, temperature, etc. auto-filled
    """
    return apply_defaults(user_config, PREDICT_DEFAULTS)


def get_minimal_required_fields() -> dict[str, list[str]]:
    """Get truly minimal required fields for each config type.

    These are the ONLY fields users must provide - everything else has defaults.

    Returns:
        Dictionary mapping config type to list of required field paths
    """
    return {
        "predict": [
            "dataset",  # Only dataset is required!
            # VLM backend/model have defaults
        ],
        "pipeline": [
            "project.name",
            "input.usd_path",
        ],
    }
