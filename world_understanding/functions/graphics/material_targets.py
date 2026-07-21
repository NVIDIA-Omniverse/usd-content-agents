# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared render material target policy."""

from __future__ import annotations

from typing import Literal, cast, get_args

RenderMaterialTarget = Literal[
    "auto",
    "display_color",
    "preview_surface",
    "openpbr_materialx",
    "omnipbr_mdl",
]

RENDER_MATERIAL_TARGETS = frozenset(get_args(RenderMaterialTarget))

_ALIASES = {
    "displaycolor": "display_color",
    "preview": "preview_surface",
    "usd_preview_surface": "preview_surface",
    "usdpreviewsurface": "preview_surface",
    "openpbr": "openpbr_materialx",
    "materialx": "openpbr_materialx",
    "materialx_openpbr": "openpbr_materialx",
    "native": "auto",
    "mdl": "omnipbr_mdl",
    "omnipbr": "omnipbr_mdl",
}


def normalize_render_material_target(
    material_target: str | None,
) -> RenderMaterialTarget:
    """Normalize a user-facing material target token."""
    if material_target is None:
        return "auto"

    target = material_target.strip().lower().replace("-", "_")
    if not target:
        return "auto"
    target = _ALIASES.get(target, target)

    if target not in RENDER_MATERIAL_TARGETS:
        supported = ", ".join(sorted(RENDER_MATERIAL_TARGETS))
        raise ValueError(
            f"Unsupported material_target '{material_target}'. "
            f"Supported values: {supported}"
        )
    return cast(RenderMaterialTarget, target)


def preview_fallbacks_enabled_for_material_target(
    material_target: str | None,
    *,
    legacy_add_preview_fallbacks: bool | None = None,
) -> bool:
    """Return whether render export should author PreviewSurface fallbacks.

    ``auto`` preserves authored/native material outputs by default. The legacy
    ``add_preview_fallbacks`` flag only affects ``auto`` when callers set it
    explicitly.
    """
    target = normalize_render_material_target(material_target)
    if target == "preview_surface":
        return True
    if target == "auto" and legacy_add_preview_fallbacks is not None:
        return legacy_add_preview_fallbacks
    return False
