# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration schema definitions for the unified config system.

This module defines the expected structure of the unified pipeline configuration
for the Joint Agent. Unlike Material Agent, Joint Agent focuses on general
asset classification without material-specific operations.
"""

from typing import Any

from joint_agent.api.defaults import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_IMAGE_PROMPTS,
    DEFAULT_VLM_MAX_TOKENS,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_REASONING_EFFORT,
    DEFAULT_VLM_TEMPERATURE,
    IDENTIFY_ASSET_DEFAULTS,
)
from joint_agent.joint_rigger_options import (
    DEFAULT_CANDIDATE_READINESS_POLICY,
    DEFAULT_JOINT_RIGGER_ADAPTER,
    DEFAULT_MISSING_DEPENDENCY_POLICY,
    DEFAULT_USD_JOINT_RIGGER_APPLY_COLLISION,
    DEFAULT_USD_JOINT_RIGGER_APPLY_MASSES,
    DEFAULT_USD_JOINT_RIGGER_TEMPLATE,
)

PROMPT_PROFILE_GENERIC = "generic"
PROMPT_PROFILE_PROP_ARTICULATION = "prop_articulation"
SUPPORTED_PROMPT_PROFILES = {
    PROMPT_PROFILE_GENERIC,
    PROMPT_PROFILE_PROP_ARTICULATION,
}

# Step name to directory name mapping
STEP_OUTPUT_DIRS = {
    "optimize_usd": "optimized",
    "build_dataset_usd": "dataset/usd",
    "identify_asset": "identification",
    "analyze_structure": "structure",
    "build_dataset_prepare_dataset": "dataset",
    "predict": "predictions",
    "consistency_pass": "consistency",
    "infer_articulation_candidates": "articulation_candidates",
    "restore_usd": "restored",
    "apply_joint_rigger": "joint_rigger",
    "author_physics_schemas": "physics_authoring",
}

# Step execution order
STEP_ORDER = [
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

# Required top-level sections
REQUIRED_SECTIONS = ["project", "input"]

# Required fields in each section
REQUIRED_FIELDS = {
    "project": ["name"],
    "input": ["usd_path"],
}

# Optional top-level sections
OPTIONAL_SECTIONS = ["steps", "advanced"]


def get_default_config() -> dict[str, Any]:
    """Get default configuration structure.

    Returns:
        Dictionary with default configuration values
    """
    return {
        "project": {
            "name": "joint_agent_project",
            "session_id": None,  # Will auto-generate UUID if not provided
            "working_dir": None,  # Will default to .sessions/{session_id}
            "description": "",
        },
        "input": {
            "usd_path": None,  # Required
            "reference_images": [],
        },
        "steps": {},
        "advanced": {
            "keep_temp_files": True,
            "log_level": "INFO",
        },
    }


def get_step_defaults(step_name: str) -> dict[str, Any]:
    """Get default configuration for a specific step.

    Args:
        step_name: Name of the step

    Returns:
        Dictionary with default step configuration
    """
    defaults: dict[str, dict[str, Any]] = {
        "optimize_usd": {
            "enabled": False,  # Disabled by default, opt-in
            # optimization_config defaults are applied by OptimizeUSDConfigTask
            # via SceneOptimizerSettings model (deinstance + split + deduplicate)
        },
        "build_dataset_usd": {
            "enabled": True,
            "renderer": {
                "backend": "remote",
                "image_width": 512,
                "image_height": 512,
                "cull_style": "back",
                "camera_view_type": "corner",
                "rendering_modes": {
                    "prim_only": {
                        "margin": 1.2,
                        "cameras": ["+x+y+z", "-x-y-z"],
                        "camera_focus_mode": "prim",
                    },
                    "composition": {
                        "margin": 6.0,
                        "cameras": ["+x", "+y", "+z"],
                        "camera_focus_mode": "stage",
                        "skip_occluded_images": False,
                    },
                },
                "should_highlight_prim": False,
                "should_assign_random_colors": True,
                "highlight_color": [0.7, 0.0, 0.0],
                "other_color_range": [0.1, 0.2],
            },
            "prim_filters": {
                "types": ["UsdGeom.Mesh"],
                "skip_instances": False,
                "skip_prototypes": False,
            },
            "extract_hierarchy": True,
            "extract_metadata": True,
            "skip_existing": True,
            "batch_size": 4,
            "num_workers": 32,
        },
        "identify_asset": {
            "enabled": True,
            "renderer": IDENTIFY_ASSET_DEFAULTS["renderer"],
            "vlm": IDENTIFY_ASSET_DEFAULTS["vlm"],
            "prompts": IDENTIFY_ASSET_DEFAULTS["prompts"],
        },
        "analyze_structure": {
            "enabled": False,  # Opt-in, used for articulated bodies
            "strategy": "auto",  # auto | hierarchy | geometric | skip
            "use_prompt_library": False,
            "robot_id": None,
            "vlm": {
                "backend": DEFAULT_VLM_BACKEND,
                "model": DEFAULT_VLM_MODEL,
                "temperature": 0.1,
                "max_tokens": 8192,
            },
        },
        "build_dataset_prepare_dataset": {
            "enabled": True,
            "prompt_profile": PROMPT_PROFILE_GENERIC,
            "include_prim_path_context": True,
            "include_scene_prim_path_context": True,
            "scene_prim_path_limit": 80,
            "include_geometric_context": True,
            "include_repetition_context": True,
            "repetition_signature_depth": 2,
            "repetition_sibling_limit": 5,
            "prompts": {
                "system": DEFAULT_SYSTEM_PROMPT,
                "user": DEFAULT_USER_PROMPT,
                "vlm_image_prompts": DEFAULT_VLM_IMAGE_PROMPTS.copy(),
            },
        },
        "predict": {
            "enabled": True,
            "vlm": {
                "backend": DEFAULT_VLM_BACKEND,
                "model": DEFAULT_VLM_MODEL,
                "temperature": DEFAULT_VLM_TEMPERATURE,
                "max_tokens": DEFAULT_VLM_MAX_TOKENS,
                "reasoning_effort": DEFAULT_VLM_REASONING_EFFORT,
            },
            "llm": {},  # Optional LLM for response parsing
            "max_workers": 64,
            "completion_retries": 3,
            "output_key": "classification",  # Configurable output key
        },
        "consistency_pass": {
            "enabled": False,
            "output_key": "classification",
            "min_group_size": 2,
            "min_majority_fraction": 0.6,
            "harmonize_fields": [],
            "harmonize_motion_profiles": False,
            "add_role": True,
            "add_instance_id": True,
            "signature_depth": 2,
        },
        "infer_articulation_candidates": {
            "enabled": False,
            "output_key": "classification",
            "candidate_joint_types": ["revolute", "prismatic", "spherical"],
            "llm": {},
            "vlm": {},
            "adjudication": {
                "enabled": False,
                "reconcile_topology": False,
                "model_key": "llm",
                "min_confidence": "high",
                "temperature": 0.0,
                "max_tokens": 4096,
                "max_images": 16,
                "max_adjudications": 8,
                "require_source_images": False,
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
        },
        "author_physics_schemas": {
            "enabled": False,
        },
    }

    return defaults.get(step_name, {"enabled": True})
