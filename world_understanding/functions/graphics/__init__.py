# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graphics functions for World Understanding.

This module contains functions for graphics operations including USD rendering.
"""

from __future__ import annotations

import logging

from world_understanding.functions.graphics.comfyui_workflows import (
    execute_comfyui_workflow,
)
from world_understanding.functions.graphics.image_editing import edit_image_with_comfyui
from world_understanding.functions.graphics.pdf_to_images import convert_pdf_to_images
from world_understanding.functions.graphics.render_time_sampled_usd import (
    render_time_sampled_usd,
)
from world_understanding.functions.graphics.rendering import (
    prepare_prims_with_composition,
    prepare_render_prims,
    render_from_prepared_composition,
    render_from_prepared_prims,
    render_prims,
    render_prims_with_composition,
)
from world_understanding.functions.graphics.usd_camera import (
    extract_camera_parameters,
    load_camera_json,
    project_point,
    save_camera_json,
    unproject_pixel,
)
from world_understanding.functions.graphics.usd_scene_analysis import detect_objects
from world_understanding.functions.graphics.uv_generation import (
    ProjectionType,
    generate_atlas_uvs,
    generate_projection_uvs,
)
from world_understanding.utils.image_utils import (
    base64_to_image,
    image_to_base64,
    load_image,
    save_image,
)
from world_understanding.utils.usd.stage import (
    create_stage,
    create_stage_with_file,
    create_temp_stage,
    duplicate_stage,
    export_stage_to_string,
    flatten_stage,
    get_stage_info,
    load_stage,
    load_stage_from_string,
    merge_stages,
    remove_animation,
    save_stage,
)

from . import render_nvcf, render_nvcf_async, render_remote, render_remote_async

_logger = logging.getLogger(__name__)

try:
    from . import render_ovrtx
except ImportError:  # pragma: no cover - optional GPU backend dependency
    _logger.debug("render_ovrtx unavailable (ovrtx not installed)")

try:
    from . import render_warp
except ImportError:  # pragma: no cover - optional GPU backend dependency
    _logger.debug("render_warp unavailable (warp-lang not installed)")

__all__ = [
    # USD rendering functions
    "prepare_render_prims",
    "render_prims",
    "prepare_prims_with_composition",
    "render_prims_with_composition",
    "render_from_prepared_prims",
    "render_from_prepared_composition",
    "render_time_sampled_usd",
    # Image IO functions
    "load_image",
    "save_image",
    "image_to_base64",
    "base64_to_image",
    # USD Stage IO functions
    "create_stage",
    "create_stage_with_file",
    "create_temp_stage",
    "duplicate_stage",
    "export_stage_to_string",
    "flatten_stage",
    "get_stage_info",
    "load_stage",
    "load_stage_from_string",
    "merge_stages",
    "remove_animation",
    "save_stage",
    # ComfyUI functions
    "execute_comfyui_workflow",
    "edit_image_with_comfyui",
    # USD camera functions
    "extract_camera_parameters",
    "project_point",
    "unproject_pixel",
    "save_camera_json",
    "load_camera_json",
    # PDF functions
    "convert_pdf_to_images",
    # Scene analysis functions
    "detect_objects",
    # UV generation functions
    "ProjectionType",
    "generate_atlas_uvs",
    "generate_projection_uvs",
    # Rendering backend modules
    "render_remote",
    "render_remote_async",
    "render_nvcf",
    "render_nvcf_async",
]

# Add optional GPU backend modules to __all__ if available
if "render_ovrtx" in dir():
    __all__.append("render_ovrtx")
if "render_warp" in dir():
    __all__.append("render_warp")
