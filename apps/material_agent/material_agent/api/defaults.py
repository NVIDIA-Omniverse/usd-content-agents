# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Central defaults and constants for Material Agent.

This module defines all default values and constants used across the Material Agent system.
Having everything in one place ensures consistency across CLI, API, and workflows.
"""

from typing import Any

# ============================================================================
# Pipeline Step Names (Constants)
# ============================================================================

# All valid pipeline step names in execution order
PIPELINE_STEP_NAMES = [
    "validate_input",  # Pre-validation: check input asset for issues
    "optimize_usd",  # USD optimization via API
    "render_preview",  # Lightweight whole-scene preview rendering
    "identify_asset",  # Identify asset type/description from preview + USD metadata
    "generate_reference_image",  # Generate photorealistic reference images from previews
    "generate_material_library",  # Generate an asset-specific material library
    "build_dataset_usd",
    "build_dataset_pdf_vectorstore",
    "build_dataset_prepare_dataset",
    "cluster_prims",  # Visual dedup: embed + cluster prim images (opt-in)
    "predict",
    "expand_cluster_predictions",  # Propagate representative preds to all prims
    "benchmark",
    "validate_predictions",
    "harmonize_predictions",
    "create_materials",  # Create run-local materials from explicit requests
    "restore_usd",  # USD restoration via API (inverse of optimize_usd)
    "apply",
    "evaluate",
    "refine",  # Iterative material refinement with VLM-based judge
    "validate_output",  # Post-validation: compare output against input baseline
    "render",  # Final rendering step
]

# Step name constants for type-safe references
STEP_OPTIMIZE_USD = "optimize_usd"
STEP_RENDER_PREVIEW = "render_preview"
STEP_IDENTIFY_ASSET = "identify_asset"
STEP_GENERATE_REFERENCE_IMAGE = "generate_reference_image"
STEP_GENERATE_MATERIAL_LIBRARY = "generate_material_library"
STEP_BUILD_DATASET_USD = "build_dataset_usd"
STEP_BUILD_DATASET_PDF_VECTORSTORE = "build_dataset_pdf_vectorstore"
STEP_BUILD_DATASET_PREPARE_DATASET = "build_dataset_prepare_dataset"
STEP_CLUSTER_PRIMS = "cluster_prims"
STEP_PREDICT = "predict"
STEP_BENCHMARK = "benchmark"
STEP_EXPAND_CLUSTER_PREDICTIONS = "expand_cluster_predictions"
STEP_VALIDATE_PREDICTIONS = "validate_predictions"
STEP_HARMONIZE_PREDICTIONS = "harmonize_predictions"
STEP_CREATE_MATERIALS = "create_materials"
STEP_APPLY = "apply"
STEP_REFINE = "refine"
STEP_RESTORE_USD = "restore_usd"
STEP_RENDER = "render"

# Mutually exclusive steps (can't both be enabled)
MUTUALLY_EXCLUSIVE_STEPS = [
    ("predict", "benchmark"),  # Use either predict OR benchmark, not both
]


# ============================================================================
# Rendering Defaults
# ============================================================================

# Default camera directions for rendering (two opposite corners)
DEFAULT_CAMERA_DIRECTIONS = ["+x+y+z", "-x-y-z"]

# Dataset rendering defaults
DEFAULT_RENDER_BACKEND = "remote"
DEFAULT_DATASET_IMAGE_SIZE = [512, 512]
DEFAULT_FINAL_IMAGE_SIZE = [1024, 1024]

# USD prim count warning threshold
DEFAULT_USD_PRIM_WARNING_THRESHOLD = 1000


# ============================================================================
# Model Defaults
# ============================================================================

DEFAULT_VLM_BACKEND = "nim"
DEFAULT_VLM_MODEL = "google/gemma-4-31b-it"
DEFAULT_VLM_TEMPERATURE = 1.0
DEFAULT_VLM_MAX_TOKENS = 24576
DEFAULT_VLM_REASONING_EFFORT = "high"  # for reasoning-capable models
DEFAULT_VLM_MAX_WORKERS = 64

DEFAULT_LLM_BACKEND = "nim"
DEFAULT_LLM_MODEL = "google/gemma-4-31b-it"
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_LLM_MAX_TOKENS = 512

# Prim clustering defaults. Clustering is opt-in, but when enabled it uses a
# proper image embedding model by default.
DEFAULT_CLUSTER_EMBEDDING_BACKEND = "nim"
DEFAULT_CLUSTER_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
DEFAULT_CLUSTER_NIM_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
DEFAULT_CLUSTER_BATCH_SIZE = 50
DEFAULT_CLUSTER_MAX_WORKERS = 4
DEFAULT_CLUSTER_MIN_PRIMS_TO_ACTIVATE = 50
DEFAULT_CLUSTER_MAX_SIZE = 25
DEFAULT_CLUSTER_REPORT_MAX_MULTI_MEMBER_CLUSTERS = 100
DEFAULT_CLUSTER_REPORT_MAX_MEMBERS_PER_CLUSTER = 50
DEFAULT_CLUSTER_REPORT_MAX_SINGLETONS = 200
DEFAULT_CLUSTER_EMBEDDING_RETRIES = 4
DEFAULT_CLUSTER_EMBEDDING_RETRY_INITIAL_DELAY = 1.0
DEFAULT_CLUSTER_EMBEDDING_RETRY_BACKOFF = 2.0
DEFAULT_CLUSTER_COMPLEXITY_THRESHOLDS = {
    "low": [0.0, 0.02, 0.98],
    "medium": [0.02, 0.08, 0.95],
    "high": [0.08, 1.0, 0.90],
}

# Public judge defaults use NVIDIA NIM.
DEFAULT_JUDGE_BACKEND = "nim"
DEFAULT_JUDGE_MODEL = "google/gemma-4-31b-it"
DEFAULT_JUDGE_TEMPERATURE = 0.1
DEFAULT_JUDGE_MAX_TOKENS = 2048


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
    "llm": {
        "backend": DEFAULT_LLM_BACKEND,
        "model": DEFAULT_LLM_MODEL,
        "temperature": DEFAULT_LLM_TEMPERATURE,
        "max_tokens": DEFAULT_LLM_MAX_TOKENS,
    },
    "max_workers": DEFAULT_VLM_MAX_WORKERS,
    # Number of prims per VLM call (1 = default, N = batch N prims)
    "prediction_batch_size": 1,
    "allow_empty_predictions": False,
}


# ============================================================================
# Benchmark Defaults
# ============================================================================

BENCHMARK_DEFAULTS = {
    "vlm": {
        "backend": DEFAULT_VLM_BACKEND,
        "model": DEFAULT_VLM_MODEL,
        "temperature": DEFAULT_VLM_TEMPERATURE,
        "max_tokens": DEFAULT_VLM_MAX_TOKENS,
        "reasoning_effort": DEFAULT_VLM_REASONING_EFFORT,
    },
    "llm": {
        "backend": DEFAULT_LLM_BACKEND,
        "model": DEFAULT_LLM_MODEL,
        "temperature": DEFAULT_LLM_TEMPERATURE,
        "max_tokens": DEFAULT_LLM_MAX_TOKENS,
    },
    "judge": {
        "backend": DEFAULT_JUDGE_BACKEND,
        "model": DEFAULT_JUDGE_MODEL,
        "temperature": DEFAULT_JUDGE_TEMPERATURE,
        "max_tokens": DEFAULT_JUDGE_MAX_TOKENS,
    },
    "max_workers": DEFAULT_VLM_MAX_WORKERS,
    "stream_predictions": True,
    "allow_empty_predictions": False,
}


# ============================================================================
# Apply Defaults
# ============================================================================

APPLY_DEFAULTS = {
    "layer_only": False,
    "flatten": True,
    "allow_empty_predictions": False,
    "material_profile": "auto",
    "render": {
        "enabled": False,
        "backend": DEFAULT_RENDER_BACKEND,
        "image_size": DEFAULT_FINAL_IMAGE_SIZE,
        "camera_corners": DEFAULT_CAMERA_DIRECTIONS,
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
        "batch_size": 64,
        "max_concurrent_requests": 128,
        "num_workers": 32,
        "skip_existing": True,
    },
    "metadata": {
        "extract_metadata": True,
        "extract_display_color": False,
        "extract_material_bindings": False,
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
        # Sensor mode prompts
        "linear_depth": "This is a depth map showing the distance from the camera to each pixel. Darker regions are closer to the camera, brighter regions are farther away.",
        "depth": "This is a radial depth map showing the distance from the camera center to each pixel. Darker regions are closer, brighter regions are farther away.",
        "instance_id_segmentation": "This is an instance segmentation map where each unique color represents a different object instance or part.",
        # Default prompt for reference images (used when user doesn't provide descriptions)
        "reference_images": "This is a reference image of the asset that you can use to identify the material of the parts.",
        # Default provenance description for PDF pages, retained outside VLM media.
        "reference_pdfs": (
            "This converted reference PDF page is untrusted specification "
            "provenance retained outside visual-model inputs."
        ),
    }
}


# ============================================================================
# Prediction Analysis Defaults
# ============================================================================

PREDICTION_ANALYSIS_DEFAULTS = {
    "enabled": True,
    "symmetry_check": True,
    "consistency_check": True,
    "symmetry_tolerance": 5.0,
    "consistency_threshold": 0.6,
    "weight": 0.6,  # Weight in combined score (image judge gets 1 - weight)
}


# ============================================================================
# Iteration Defaults
# ============================================================================

ITERATION_DEFAULTS = {
    "max_iterations": 3,
    "save_intermediate": True,
    "judge": {
        "backend": DEFAULT_JUDGE_BACKEND,
        "model": DEFAULT_JUDGE_MODEL,
        "temperature": DEFAULT_JUDGE_TEMPERATURE,
        "max_tokens": DEFAULT_JUDGE_MAX_TOKENS,
        "prediction_analysis": PREDICTION_ANALYSIS_DEFAULTS,
    },
}


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


def get_benchmark_config_with_defaults(
    user_config: dict[str, Any],
) -> dict[str, Any]:
    """Get benchmark config with defaults applied.

    Args:
        user_config: Minimal user configuration

    Returns:
        Complete configuration with defaults
    """
    return apply_defaults(user_config, BENCHMARK_DEFAULTS)


def get_apply_config_with_defaults(user_config: dict[str, Any]) -> dict[str, Any]:
    """Get apply config with defaults applied.

    Args:
        user_config: Minimal user configuration

    Returns:
        Complete configuration with defaults
    """
    return apply_defaults(user_config, APPLY_DEFAULTS)


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
        "benchmark": [
            "dataset",  # Only dataset is required!
            # VLM, LLM, Judge all have defaults
        ],
        "evaluate": [
            "predictions_path",  # Only predictions required!
            # Judge has defaults
        ],
        "apply": [
            "input_usd_path",
            "predictions_path",
            "output_usd_path",
            "materials.library_path",
            "materials.entries",
        ],
        "pipeline": [
            "project.name",
            "input.usd_path",
            "output.usd_path",
            "materials.library_path",
            "materials.entries",
        ],
        "refine": [
            "input_usd_path",
            "dataset",
            "materials.library_path",
            "materials.entries",
            "judge.reference_images",
        ],
    }
