# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material authoring profile names and normalization helpers."""

from typing import Literal

MaterialProfile = Literal[
    "auto",
    "display_color",
    "preview_surface",
    "openpbr_materialx",
    "omnipbr_mdl",
]

MATERIAL_PROFILES: tuple[MaterialProfile, ...] = (
    "auto",
    "display_color",
    "preview_surface",
    "openpbr_materialx",
    "omnipbr_mdl",
)

_ALIASES: dict[str, MaterialProfile] = {
    "": "auto",
    "auto": "auto",
    "native": "auto",
    "preserve": "auto",
    "display": "display_color",
    "displaycolor": "display_color",
    "display-color": "display_color",
    "display_color": "display_color",
    "basic": "display_color",
    "preview": "preview_surface",
    "previewsurface": "preview_surface",
    "preview-surface": "preview_surface",
    "preview_surface": "preview_surface",
    "usdpreview": "preview_surface",
    "usd_preview_surface": "preview_surface",
    "openpbr": "openpbr_materialx",
    "open_pbr": "openpbr_materialx",
    "openpbr_materialx": "openpbr_materialx",
    "openpbr-materialx": "openpbr_materialx",
    "materialx": "openpbr_materialx",
    "mtlx": "openpbr_materialx",
    "mdl": "omnipbr_mdl",
    "omnipbr": "omnipbr_mdl",
    "omnipbr_mdl": "omnipbr_mdl",
    "omni_pbr_mdl": "omnipbr_mdl",
    "rtx": "omnipbr_mdl",
}


def normalize_material_profile(profile: str | None) -> MaterialProfile:
    """Normalize a user-supplied material authoring profile.

    ``None`` and empty strings preserve backward compatibility by mapping to
    ``auto``.
    """
    if profile is None:
        return "auto"

    key = str(profile).strip().lower().replace(" ", "_")
    if key in _ALIASES:
        return _ALIASES[key]

    valid = ", ".join(MATERIAL_PROFILES)
    raise ValueError(
        f"Unsupported material_profile '{profile}'. Expected one of: {valid}"
    )
