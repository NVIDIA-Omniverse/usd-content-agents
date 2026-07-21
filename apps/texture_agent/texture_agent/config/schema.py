# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration schema for the texture agent pipeline."""

from texture_agent.config.rendering_backends import (
    DEFAULT_TEXTURE_RENDERING_BACKEND,
)

# Step execution order
STEP_ORDER = [
    "prepare_uvs",
    "discover_materials",
    "plan_textures",
    "generate_prompts",
    "render_previews",
    "generate_textures",
    "blend_textures",
    "apply_textures",
    "render",
]

# Step name to output directory mapping
STEP_OUTPUT_DIRS = {
    "prepare_uvs": "prepared",
    "discover_materials": "discovery",
    "generate_prompts": "prompts",
    "render_previews": "previews",
    "generate_textures": "generated",
    "blend_textures": "textures",
    "apply_textures": "output",
    "render": "renders",
}

# Default configuration values
DEFAULTS = {
    "texture": {
        "backend": "simple_image_gen",
        "detail_policy": "default",
        "model": None,
        "size": 1024,
        "uv_policy": "generate_missing",
        "uv_scope": "stage",
        "uv_generation_mode": "projection",
        "uv_projection": "box",
        "uv_normalize_out_of_range": False,
        "uv_rebake_source_albedo": False,
        "uv_rebake_size": None,
        "uv_atlas_distortion_threshold": 3.0,
        "uv_atlas_enable_packing": True,
    },
    "variations": {
        "count": 1,
    },
    "auto_prompt": {
        "enabled": False,
    },
    "steps": {
        "prepare_uvs": {"enabled": True},
        "discover_materials": {"enabled": True},
        "plan_textures": {"enabled": True},
        "generate_prompts": {"enabled": True},
        "render_previews": {
            "enabled": True,
            "backend": DEFAULT_TEXTURE_RENDERING_BACKEND,
            "image_width": 512,
            "image_height": 512,
        },
        "generate_textures": {
            "enabled": True,
            "max_workers": 4,
            "skip_existing": True,
        },
        "blend_textures": {
            "enabled": True,
            "default_opacity": 0.85,
            "output_size": 1024,
        },
        "apply_textures": {"enabled": True},
        "render": {
            "enabled": True,
            "backend": DEFAULT_TEXTURE_RENDERING_BACKEND,
            "image_width": 1024,
            "image_height": 1024,
        },
    },
}
