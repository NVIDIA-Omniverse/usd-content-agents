# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration schema definitions for the unified config system.

This module defines the expected structure of the unified pipeline configuration
for the Physics Agent. Unlike Material Agent, Physics Agent focuses on general
asset classification without material-specific operations.
"""

from typing import Any

from physics_agent.api.defaults import (
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

# Step name to directory name mapping
STEP_OUTPUT_DIRS = {
    "optimize_usd": "optimized",
    "build_dataset_usd": "dataset/usd",
    "identify_asset": "identification",
    "build_dataset_prepare_dataset": "dataset",
    "predict": "predictions",
    "restore_usd": "restored",
    "apply_physics": "physics",
}

# Step execution order
STEP_ORDER = [
    "optimize_usd",
    "identify_asset",
    "build_dataset_usd",
    "build_dataset_prepare_dataset",
    "predict",
    "restore_usd",
    "apply_physics",
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
            "name": "physics_agent_project",
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
    defaults = {
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
                        "use_original_materials": True,
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
        "build_dataset_prepare_dataset": {
            "enabled": True,
            "include_prim_path_context": True,
            "include_geometric_context": True,
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
            "output_key": "classification",  # Configurable output key
            "allow_empty_predictions": False,
        },
        "apply_physics": {
            "enabled": True,
            # usd_path, predictions_path, output_usd_path are auto-wired by the pipeline
            "collision_approx": "convexHull",
            "mass_scale_policy": "skip_mass",
            "allow_empty_predictions": False,
        },
    }

    return defaults.get(step_name, {"enabled": True})
