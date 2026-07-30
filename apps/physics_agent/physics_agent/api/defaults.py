# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Central defaults and constants for Physics Agent.

This module defines all default values and constants used across the Physics Agent system.
Having everything in one place ensures consistency across CLI, API, and workflows.
"""

import logging
import os
from typing import Any

from world_understanding.functions.graphics.rendering_backend_factory import (
    validate_rendering_backend_name,
)
from world_understanding.utils.credentials import format_env_reference
from world_understanding.utils.environment import parse_float_env, parse_int_env

logger = logging.getLogger(__name__)


# ============================================================================
# Pipeline Step Names (Constants)
# ============================================================================

# All valid pipeline step names in execution order
PIPELINE_STEP_NAMES = [
    "optimize_usd",
    "identify_asset",
    "build_dataset_usd",
    "build_dataset_prepare_dataset",
    "predict",
    "restore_usd",
    "apply_physics",
]

# Step name constants for type-safe references
STEP_OPTIMIZE_USD = "optimize_usd"
STEP_IDENTIFY_ASSET = "identify_asset"
STEP_BUILD_DATASET_USD = "build_dataset_usd"
STEP_BUILD_DATASET_PREPARE_DATASET = "build_dataset_prepare_dataset"
STEP_PREDICT = "predict"
STEP_RESTORE_USD = "restore_usd"
STEP_APPLY_PHYSICS = "apply_physics"


# ============================================================================
# Rendering Defaults
# ============================================================================

# Default camera directions for rendering (two opposite corners)
DEFAULT_CAMERA_DIRECTIONS = ["+x+y+z", "-x-y-z"]

# Dataset rendering defaults
DEFAULT_RENDER_BACKEND = os.environ.get("PA_RENDER_BACKEND", "remote")
DEFAULT_DATASET_IMAGE_SIZE = [512, 512]

# USD prim count warning threshold
DEFAULT_USD_PRIM_WARNING_THRESHOLD = 1000


# ============================================================================
# Model Defaults
# ============================================================================

DEFAULT_VLM_BACKEND = os.environ.get("PA_VLM_BACKEND", "nim")
DEFAULT_VLM_MODEL = os.environ.get("PA_VLM_MODEL", "google/gemma-4-31b-it")
DEFAULT_VLM_TEMPERATURE = parse_float_env(
    "PA_VLM_TEMPERATURE", 1.0, minimum=0.0, maximum=2.0, logger=logger
)
DEFAULT_VLM_MAX_TOKENS = parse_int_env(
    "PA_VLM_MAX_TOKENS", 24576, minimum=1, logger=logger
)
DEFAULT_VLM_REASONING_EFFORT = os.environ.get(
    "PA_VLM_REASONING_EFFORT", "high"
)  # for reasoning-capable models (e.g. gpt-5)
# Bound per-pipeline inference concurrency to avoid accidental resource exhaustion.
DEFAULT_VLM_MAX_WORKERS = parse_int_env(
    "PA_VLM_MAX_WORKERS", 64, minimum=1, maximum=256, logger=logger
)
DEFAULT_VLM_BASE_URL = os.environ.get("PA_VLM_BASE_URL") or None
DEFAULT_VLM_API_KEY = os.environ.get("PA_VLM_API_KEY") or None
DEFAULT_VLM_API_KEY_ENV = os.environ.get("PA_VLM_API_KEY_ENV") or None

# Judge-specific invoke defaults. Keep this separate from the VLM construction
# token budget so visual judging can use the same compact critique contract as
# material-agent while the base VLM remains configured for larger tasks.
DEFAULT_JUDGE_TEMPERATURE = parse_float_env(
    "PA_JUDGE_TEMPERATURE", 0.0, minimum=0.0, maximum=2.0, logger=logger
)
DEFAULT_JUDGE_MAX_TOKENS = parse_int_env(
    "PA_JUDGE_MAX_TOKENS", 2048, minimum=1, logger=logger
)


def _vlm_endpoint_config() -> dict[str, str]:
    config: dict[str, str] = {}
    if DEFAULT_VLM_BASE_URL:
        config["base_url"] = DEFAULT_VLM_BASE_URL
    if DEFAULT_VLM_API_KEY_ENV:
        config["api_key_env"] = format_env_reference(DEFAULT_VLM_API_KEY_ENV)
    elif DEFAULT_VLM_API_KEY:
        config["api_key_env"] = format_env_reference("PA_VLM_API_KEY")
    return config


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
        **_vlm_endpoint_config(),
    },
    # No separate parser LLM by default. Predict falls back to llm = vlm
    # unless the caller explicitly configures a dedicated llm backend/model.
    "max_workers": DEFAULT_VLM_MAX_WORKERS,
    "output_key": "classification",  # Configurable output key for predictions
    "allow_empty_predictions": False,
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
        "temperature": DEFAULT_VLM_TEMPERATURE,
        "max_tokens": 4096,
        **_vlm_endpoint_config(),
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
            "prim_only_original": {
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

DEFAULT_REFERENCE_IMAGE_PROMPT = "This is a reference image of the asset."


# ============================================================================
# Service-Level Pipeline Defaults (used by physics_agent_service)
# ============================================================================

# These prompts are identical across all asset config YAMLs.
DEFAULT_SYSTEM_PROMPT = """\
You are an expert 3D asset analyst specializing in identifying components and materials
from rendered images. Your task is to analyze 3D model parts and classify them.

**Important: Rendering Context**
The rendered images may use randomized or artificial colors that do NOT represent
actual materials. Do not rely on color to determine material type. Instead, reason
about the component's shape, geometry, function, and what it would be made of
in the real world.

**Instructions:**
1. First, examine the composition images to identify what the overall object is
2. Then examine the isolated component images to identify the specific part
3. Reason about what material this component would realistically be made of,
   considering the overall object, the part's function, and common manufacturing
4. Estimate physical properties based on the identified material
5. Use the geometric context (dimensions) to inform your analysis

**Component Classification:**
Identify the component by its function and form. Examples:
- structural: Frames, housings, enclosures, bodies, bases
- mechanical: Gears, levers, hinges, springs, fasteners
- electrical: Wires, connectors, circuits, contacts, terminals
- optical: Lenses, bulbs, reflectors, diffusers, displays
- thermal: Heat sinks, fins, insulation, vents
- decorative: Covers, trim, labels, textures
- fastener: Screws, bolts, nuts, clips, brackets
- seal: Gaskets, O-rings, caps, plugs
- other: Components that don't fit above categories

**Material Classification:**
Identify the primary material based on the component's function and real-world context:
- metal: Toolheads, structural frames, barrels, nozzles, springs, load-bearing parts, fasteners
- glass: Lenses, transparent covers, vials, bulbs, display surfaces
- plastic: Lightweight housings, consumer product bodies, non-structural covers, buttons, knobs
- rubber: Grips, seals, gaskets, tires, vibration dampeners, flexible components
- ceramic: Insulators, high-temperature components, decorative tiles
- fabric: Woven or fibrous components, insulation wraps, straps
- wood: Handles, structural supports, frames in hand tools and furniture
- composite: Layered or reinforced components, carbon fiber, fiberglass

**Physical Properties to Estimate:**
Based on the identified material, estimate:
- density: kg/m\u00b3 (wood ~500-800, plastic ~900-1400, rubber ~1100-1200, glass ~2500, metal ~2700-8000)
- estimated_mass_kg: Estimated mass in kg. Formula: mass = density * bbox_volume * fill_factor. \
The fill factor accounts for how much of the bounding box is actually solid material. \
IMPORTANT: Most real-world objects are hollow, thin-walled, or have significant void space — \
do NOT treat them as solid. Fill factor guidelines by construction type: solid castings/blocks ~0.7-0.9, \
dense machined parts ~0.3-0.5, thin shells/panels/plates ~0.05-0.15, hollow tubes/pipe frames ~0.01-0.05, \
sheet metal enclosures ~0.005-0.02. Always sanity-check your mass estimate against what a person could \
realistically lift or what the object would weigh in everyday experience.
- static_friction: 0.0-1.0 (plastic ~0.3-0.5, glass ~0.4, metal ~0.5-0.7, wood ~0.5-0.6, rubber ~0.8-1.0)
- dynamic_friction: 0.0-1.0 (typically 70-90% of static friction)
- restitution: 0.0-1.0 (wood ~0.3-0.5, metal ~0.3, plastic ~0.4, glass ~0.5, rubber ~0.8)

The unitless coefficients dynamicFriction and staticFriction are defined by the Coulomb friction \
model. Permissible values are non-negative, and the fallback value is zero.

The coefficient of restitution is the ratio of the final to initial relative velocity between \
two objects after they collide. Permissible values are non-negative, and the fallback value is zero.

These friction and restitution coefficients are defined per material rather than per material pair. \
Physics simulation combines the coefficients of two interacting colliders into a single value \
by averaging them.

**Response Format:**
Analyze the images and respond with JSON:
{{
  "asset_type": "what the overall object is (e.g., tool, appliance, robot)",
  "component_type": "category from list above",
  "component_name": "descriptive name of what this part is",
  "material": "material type from list above",
  "physical_properties": {{
    "density": estimated density in kg/m\u00b3,
    "estimated_mass_kg": estimated mass in kg,
    "static_friction": coefficient (0.0-1.0),
    "dynamic_friction": coefficient (0.0-1.0),
    "restitution": coefficient (0.0-1.0)
  }},
  "confidence": "high/medium/low",
  "reasoning": "brief explanation including why this material fits the component's function"
}}

Answer format:
<reasoning>your analysis</reasoning>
<answer>your JSON</answer>
"""

DEFAULT_USER_PROMPT = """\
Please identify and classify this 3D component.

Analyze the rendered images to determine:
1. What is this component? (its function and purpose)
2. What material is it made of?
3. What are its physical properties?
"""

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
    "reference_images": DEFAULT_REFERENCE_IMAGE_PROMPT,
}

# Default VLM image prompts for dataset preparation.
PREPARE_DATASET_PROMPTS_DEFAULTS = {
    "vlm_image_prompts": DEFAULT_VLM_IMAGE_PROMPTS.copy(),
}


def build_default_pipeline_config(
    session_id: str,
    usd_path: str,
    working_dir: str,
    user_prompt: str | None = None,
    render_backend: str | None = None,
    optimize_usd: bool = False,
    enable_deinstance: bool = True,
    enable_split: bool = False,
    enable_deduplicate: bool = False,
) -> dict[str, Any]:
    """Build a complete pipeline config dict from server defaults.

    Args:
        session_id: Unique session identifier.
        usd_path: Absolute path to the input USD file.
        working_dir: Working directory for intermediate artifacts.
        user_prompt: Optional user prompt override.  When *None* the
            ``DEFAULT_USER_PROMPT`` is used.
        render_backend: Rendering backend to use ("remote", "warp", "ovrtx",
            or "mock").
            When *None*, uses ``DEFAULT_RENDER_BACKEND``.
        optimize_usd: Enable the Scene Optimizer step.  When ``True``,
            ``restore_usd`` is also enabled to map predictions back to
            original prim paths.
        enable_deinstance: Enable deinstance operation (default ``True``).
            Only used when ``optimize_usd`` is ``True``.
        enable_split: Enable split-meshes operation (default ``False``).
            Only used when ``optimize_usd`` is ``True``.
        enable_deduplicate: Enable deduplicate operation (default ``False``).
            Only used when ``optimize_usd`` is ``True``.

    Returns:
        Full pipeline configuration dictionary ready for ``execute_pipeline_async``.
    """
    backend = validate_rendering_backend_name(render_backend or DEFAULT_RENDER_BACKEND)

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
                "enabled": optimize_usd,
                **(
                    {
                        "backend": "local",
                        "scene_optimizer_settings": {
                            "enable_deinstance": enable_deinstance,
                            # Public API flag is shorter; Scene Optimizer expects
                            # the operation-specific key name.
                            "enable_split_meshes": enable_split,
                            "enable_deduplicate": enable_deduplicate,
                            "generate_report": True,
                            "capture_stats": True,
                        },
                        "flatten_prototypes": False,
                    }
                    if optimize_usd
                    else {}
                ),
            },
            "identify_asset": {
                "enabled": True,
                "renderer": {
                    "backend": backend,
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
                        "prim_only_original": {
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
                    "types": ["UsdGeom.Mesh"],
                    "skip_instances": False,
                    "skip_prototypes": False,
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
                "include_geometric_context": True,
                "prompts": {
                    "system": DEFAULT_SYSTEM_PROMPT,
                    "user": user_prompt if user_prompt else DEFAULT_USER_PROMPT,
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
                    **_vlm_endpoint_config(),
                },
                "max_workers": DEFAULT_VLM_MAX_WORKERS,
                "output_key": "classification",
                "allow_empty_predictions": False,
                "report": {
                    "image_max_size": 256,
                    "image_format": "jpeg",
                    "image_quality": 75,
                },
            },
            "restore_usd": {
                "enabled": optimize_usd,
            },
            "apply_physics": {
                "enabled": True,
                # usd_path and output_usd_path are auto-wired at build time
                # from input.usd_path and working_dir; predictions_path is
                # auto-wired at runtime by the pipeline executor from prior
                # step outputs (preferring restore_usd, falling back to
                # predict).
                "collision_approx": "convexHull",
                # Do not silently bake predicted MassAPI.mass when the predict
                # step detected an implausible scale-driven mass estimate.
                "mass_scale_policy": "skip_mass",
                "allow_empty_predictions": False,
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
        >>> user_config = {"vlm": {"model": "example-vlm-model"}}
        >>> full_config = apply_defaults(user_config, PREDICT_DEFAULTS)
        >>> # Result: {"vlm": {"model": "example-vlm-model", "backend": "nim", ...}}
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
