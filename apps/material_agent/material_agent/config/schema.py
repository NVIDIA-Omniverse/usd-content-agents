# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration schema definitions for the unified config system.

This module defines the expected structure of the unified pipeline configuration.
"""

import copy
from typing import Any

from world_understanding.functions.graphics.validate_usd import (
    MATERIAL_VALIDATION_CATEGORIES,
)

from material_agent.api.defaults import (
    DEFAULT_CLUSTER_BATCH_SIZE,
    DEFAULT_CLUSTER_COMPLEXITY_THRESHOLDS,
    DEFAULT_CLUSTER_EMBEDDING_BACKEND,
    DEFAULT_CLUSTER_EMBEDDING_MODEL,
    DEFAULT_CLUSTER_EMBEDDING_RETRIES,
    DEFAULT_CLUSTER_EMBEDDING_RETRY_BACKOFF,
    DEFAULT_CLUSTER_EMBEDDING_RETRY_INITIAL_DELAY,
    DEFAULT_CLUSTER_MAX_SIZE,
    DEFAULT_CLUSTER_MAX_WORKERS,
    DEFAULT_CLUSTER_MIN_PRIMS_TO_ACTIVATE,
    DEFAULT_CLUSTER_REPORT_MAX_MEMBERS_PER_CLUSTER,
    DEFAULT_CLUSTER_REPORT_MAX_MULTI_MEMBER_CLUSTERS,
    DEFAULT_CLUSTER_REPORT_MAX_SINGLETONS,
    DEFAULT_JUDGE_BACKEND,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TEMPERATURE,
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_MAX_TOKENS,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_REASONING_EFFORT,
    DEFAULT_VLM_TEMPERATURE,
)

# Step name to directory name mapping
STEP_OUTPUT_DIRS = {
    "validate_input": "validation/input",
    "optimize_usd": "optimized",
    "render_preview": "preview",
    "identify_asset": "identification",
    "generate_reference_image": "generated_refs",
    "generate_material_library": "generated_material_library",
    "build_dataset_usd": "dataset/usd",
    "build_dataset_pdf_vectorstore": "vectorstore",
    "build_dataset_prepare_dataset": "dataset",
    "cluster_prims": "clusters",
    "predict": "predictions",
    "expand_cluster_predictions": "predictions",
    "benchmark": "predictions",
    "create_materials": "created_materials",
    "evaluate": "evaluation",
    "refine": "iterations",
    "restore_usd": "restored",
    "validate_output": "validation/output",
    "render": "renders",
}

# Step execution order
STEP_ORDER = [
    "validate_input",  # Pre-validation: establish baseline before any processing
    "optimize_usd",
    "render_preview",
    "identify_asset",
    "generate_reference_image",
    "generate_material_library",
    "build_dataset_usd",
    "build_dataset_pdf_vectorstore",
    "build_dataset_prepare_dataset",
    "cluster_prims",
    "predict",
    "expand_cluster_predictions",
    "benchmark",
    "validate_predictions",  # Validate/repair VLM predictions against material library
    "harmonize_predictions",  # Resolve conflicts for instanced parts
    "create_materials",  # Create run-local materials before restore/apply
    "restore_usd",  # Restore predictions before applying materials
    "apply",
    "evaluate",
    "refine",
    "validate_output",  # Post-validation: compare against baseline after assignment
    "render",
]

# Mutually exclusive step groups
MUTUALLY_EXCLUSIVE_STEPS = [
    ["predict", "benchmark"],  # Can't run both predict and benchmark
    ["apply", "refine"],  # Can't run both apply and refine (refine includes apply)
]

# Required top-level sections
REQUIRED_SECTIONS = ["project", "input", "output"]

# Required fields in each section
REQUIRED_FIELDS = {
    "project": ["name"],
    "input": ["usd_path"],
    "output": [],  # output.usd_path is now optional (auto-derived from session_id)
}

# Optional top-level sections
OPTIONAL_SECTIONS = ["materials", "steps", "advanced"]


def get_default_config() -> dict[str, Any]:
    """Get default configuration structure.

    Returns:
        Dictionary with default configuration values
    """
    return {
        "project": {
            "name": "material_agent_project",
            "session_id": None,  # Will auto-generate UUID if not provided
            "working_dir": None,  # Will default to .sessions/{session_id} if session_id is used
            "description": "",
        },
        "input": {
            "usd_path": None,  # Required
            "reference_images": [],
        },
        "output": {
            # usd_path is auto-derived as .{session_id}/output/output.usd
            # Only include it if you want to override the default
            "layer_only": False,
            "flatten_output": True,
            "material_profile": "auto",
        },
        "materials": {
            "library_path": None,
            "entries": [],
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
                "other_color_range": [0.35, 0.35],
            },
            "prim_filters": {
                "types": [
                    "UsdGeom.Mesh",
                    "UsdGeom.Cube",
                    "UsdGeom.Cylinder",
                    "UsdGeom.Capsule",
                    "UsdGeom.Sphere",
                    "UsdGeom.Cone",
                ],
                "skip_instances": True,
                "skip_prototypes": False,
            },
            "extract_hierarchy": True,
            "extract_metadata": True,
            "extract_material_bindings": False,
            "skip_existing": True,
            "batch_size": 64,
            "max_concurrent_requests": 128,
            "num_workers": 32,
        },
        "build_dataset_pdf_vectorstore": {
            "enabled": False,
            "source": None,  # Required if enabled
            "embedding": {
                "service": "nim",
                "model": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
            },
            "chunk_size": 512,
            "chunk_overlap": 50,
            "image_embedding_type": "text",
        },
        "build_dataset_prepare_dataset": {
            "enabled": True,
            "include_ground_truth": False,
            "include_prim_path_context": True,
            "include_geometric_context": True,
            "prompts": {
                "vlm_image_prompts": {
                    "composition": "This is an orthographic view of the object with the part of interest highlighted with an orange outline.",
                    "prim_only": "This is a rendered part of interest only without highlighting.",
                    "linear_depth": "This is a depth map showing the distance from the camera to each pixel. Darker regions are closer to the camera, brighter regions are farther away.",
                    "depth": "This is a radial depth map showing the distance from the camera center to each pixel. Darker regions are closer, brighter regions are farther away.",
                    "instance_id_segmentation": "This is an instance segmentation map where each unique color represents a different object instance or part.",
                }
            },  # Default prompts with vlm_image_prompts
            "llm": {},  # Optional LLM for spec extraction
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
            "prediction_batch_size": 1,  # Prims per VLM call (1 = default)
            "allow_empty_predictions": False,
        },
        "cluster_prims": {
            "enabled": False,
            "embedding_service": DEFAULT_CLUSTER_EMBEDDING_BACKEND,
            "embedding_model": DEFAULT_CLUSTER_EMBEDDING_MODEL,
            "batch_size": DEFAULT_CLUSTER_BATCH_SIZE,
            "max_workers": DEFAULT_CLUSTER_MAX_WORKERS,
            "min_prims_to_activate": DEFAULT_CLUSTER_MIN_PRIMS_TO_ACTIVATE,
            "max_cluster_size": DEFAULT_CLUSTER_MAX_SIZE,
            "complexity_thresholds": copy.deepcopy(
                DEFAULT_CLUSTER_COMPLEXITY_THRESHOLDS
            ),
            "embedding_retries": DEFAULT_CLUSTER_EMBEDDING_RETRIES,
            "embedding_retry_initial_delay": DEFAULT_CLUSTER_EMBEDDING_RETRY_INITIAL_DELAY,
            "embedding_retry_backoff": DEFAULT_CLUSTER_EMBEDDING_RETRY_BACKOFF,
            "report": {
                "enabled": True,
                "image_max_size": 128,
                "image_format": "jpeg",
                "image_quality": 75,
                "max_multi_member_clusters": (
                    DEFAULT_CLUSTER_REPORT_MAX_MULTI_MEMBER_CLUSTERS
                ),
                "max_members_per_cluster": (
                    DEFAULT_CLUSTER_REPORT_MAX_MEMBERS_PER_CLUSTER
                ),
                "max_singletons": DEFAULT_CLUSTER_REPORT_MAX_SINGLETONS,
            },
        },
        "expand_cluster_predictions": {
            "enabled": True,
        },
        "benchmark": {
            "enabled": False,
            "vlm": {
                "backend": DEFAULT_VLM_BACKEND,
                "model": DEFAULT_VLM_MODEL,
                "temperature": DEFAULT_VLM_TEMPERATURE,
                "max_tokens": DEFAULT_VLM_MAX_TOKENS,
                "reasoning_effort": DEFAULT_VLM_REASONING_EFFORT,
            },
            "llm": {},
            "llm_judge": {
                "backend": DEFAULT_JUDGE_BACKEND,
                "model": DEFAULT_JUDGE_MODEL,
                "temperature": DEFAULT_JUDGE_TEMPERATURE,
                "max_tokens": DEFAULT_JUDGE_MAX_TOKENS,
            },
            "max_workers": 64,
            "allow_empty_predictions": False,
        },
        "evaluate": {
            "enabled": False,
            "llm_judge": {
                "backend": "nim",
                "model": "google/gemma-4-31b-it",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            "success_threshold": 4.0,
            "generate_html_report": True,
        },
        "apply": {
            "enabled": True,
            "usd_search": {},  # Optional: USD Search config
            "llm": {},  # Optional: for LLM-enhanced search
            "aws_profile": None,
            "local_material_source_dir": None,
            "allow_empty_predictions": False,
            "fail_on_unknown_material": False,
        },
        "refine": {
            "enabled": False,
            "max_iterations": 5,
            "vlm": {},
            "llm_judge": {},
            "apply": {
                "allow_empty_predictions": False,
                "fail_on_unknown_material": False,
            },  # Nested apply config
        },
        "validate_input": {
            "enabled": False,
            "on_failure": "warn",  # "warn", "block", or "fix"
            "validation_config": {
                "categories": list(MATERIAL_VALIDATION_CATEGORIES),
                "stage_timeout": 180.0,
            },
        },
        "validate_output": {
            "enabled": False,
            "on_failure": "warn",  # "warn" or "block"
            "validation_config": {
                "categories": list(MATERIAL_VALIDATION_CATEGORIES),
                "stage_timeout": 180.0,
            },
        },
        "optimize_usd": {
            "enabled": False,
            "optimization_config": {
                "scene_optimizer_settings": {
                    "enable_deinstance": True,
                    "enable_split_meshes": True,
                    "enable_deduplicate": True,
                    "generate_report": True,
                    "capture_stats": True,
                    "verbose": False,
                    "wait_for_assets": False,
                    "stage_timeout": 180.0,
                    "output_format": "usdc",
                    "extract_geom_subset_indices": True,
                }
            },
        },
        "generate_reference_image": {
            "enabled": False,
            "image_gen": {
                "backend": "gemini",
                "model": "gemini-3-pro-image-preview",
            },
            "prompt": "",
            "num_images": 1,
            "reference_images": [],
        },
        "generate_material_library": {
            "enabled": False,
            "material_generation_plan_path": None,
            "material_generation_plan": None,
            "vlm": {
                "backend": DEFAULT_VLM_BACKEND,
                "model": DEFAULT_VLM_MODEL,
                "temperature": DEFAULT_VLM_TEMPERATURE,
                "max_tokens": DEFAULT_VLM_MAX_TOKENS,
                "reasoning_effort": DEFAULT_VLM_REASONING_EFFORT,
            },
            "texture_generation": {
                "texture_size": 1024,
                "backend": None,
                "model": None,
                "base_url": None,
                "api_key": None,
                "api_key_env_var": None,
                "seed": None,
                "color_correct_albedo": True,
                "albedo_color_correction_strength": 1.0,
            },
            "write_material_generation_plan": True,
            "include_generation_metadata": True,
            "reference_images": [],
            "material_guidance": "",
        },
        "create_materials": {
            "enabled": False,
            "backend": "fake",
            "fake_behavior": "success",
            "creation_requests": [],
            "fail_on_error": True,
            "overwrite": False,
        },
        "identify_asset": {
            "enabled": False,
        },
        "render_preview": {
            "enabled": False,
            "backend": "remote",
            "image_width": 512,
            "image_height": 512,
            "cameras": ["+x+y+z"],
            "camera_margin": 1.0,
            "background_color": [
                1.0,
                1.0,
                1.0,
            ],  # RGB 0.0-1.0 (same scale as render step)
            "should_reset_materials": True,
            "use_lights": True,
            "flatten_before_render": False,
            "prim_filters": {},  # Empty = show all; same schema as build_dataset_usd
        },
        "render": {
            "enabled": True,
            "backend": "remote",
            "image_width": 1024,
            "image_height": 1024,
            # camera_corners: list[str] - one or multiple viewing angles
            # e.g., ["+x+y+z"] or ["+x+y+z", "-x-y-z"] for before/after views
            "camera_corners": ["+x+y+z"],
            "camera_margin": 1.2,  # 1.0 if above is applied
            "background_color": [1.0, 1.0, 1.0],  # RGB values 0.0-1.0 (white)
            "flatten_before_render": True,  # Whether to flatten the USD before rendering
        },
    }

    return defaults.get(step_name, {"enabled": True})
