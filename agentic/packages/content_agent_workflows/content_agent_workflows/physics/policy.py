# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physics workflow inference policy shared by skills and wrappers."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsMaterialProfile:
    """Baseline physics defaults for one inferred material family."""

    family: str
    density: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    volume_fraction: float
    rationale: str


_PROFILES: dict[str, PhysicsMaterialProfile] = {
    "glass": PhysicsMaterialProfile(
        family="glass",
        density=2500.0,
        static_friction=0.50,
        dynamic_friction=0.35,
        restitution=0.45,
        volume_fraction=0.04,
        rationale="Glass geometry is usually a thin shell, so use glass density with a low filled-volume fraction; glass itself is a hard elastic solid, so the wall that hits the ground still rebounds moderately.",
    ),
    "metal": PhysicsMaterialProfile(
        family="metal",
        density=7800.0,
        static_friction=0.55,
        dynamic_friction=0.42,
        restitution=0.55,
        volume_fraction=0.12,
        rationale="Metal geometry is dense and moderately rough; small fixtures and threads are often partly hollow or thin.",
    ),
    "plastic": PhysicsMaterialProfile(
        family="plastic",
        density=1050.0,
        static_friction=0.60,
        dynamic_friction=0.45,
        restitution=0.55,
        volume_fraction=0.40,
        rationale="Plastic parts are light with moderate friction and a clear rebound.",
    ),
    "rubber": PhysicsMaterialProfile(
        family="rubber",
        density=1100.0,
        static_friction=1.00,
        dynamic_friction=0.80,
        restitution=0.75,
        volume_fraction=0.35,
        rationale="Rubber has high friction and noticeable rebound, with flexible or hollow molded geometry.",
    ),
    "wood": PhysicsMaterialProfile(
        family="wood",
        density=650.0,
        static_friction=0.55,
        dynamic_friction=0.40,
        restitution=0.45,
        volume_fraction=0.60,
        rationale="Wood is medium density with moderate friction.",
    ),
    "soft": PhysicsMaterialProfile(
        family="soft",
        density=150.0,
        static_friction=0.80,
        dynamic_friction=0.65,
        restitution=0.05,
        volume_fraction=0.90,
        rationale="Foam, fabric, and other soft porous materials absorb the impact and land dead with essentially no rebound.",
    ),
    "generic": PhysicsMaterialProfile(
        family="generic",
        density=1000.0,
        static_friction=0.50,
        dynamic_friction=0.40,
        restitution=0.45,
        volume_fraction=0.25,
        rationale="Fallback profile for unknown material cues on ordinary rigid bodies: mid-range density and friction, with the moderate rebound hard solids show against a hard floor.",
    ),
}


def material_volume_fractions() -> dict[str, float]:
    """Typical filled fraction of the bounding-box volume per material family.

    Inspection bounds report axis-aligned bounding-box volume, not the solid
    volume a part occupies, so mass estimates from those bounds must scale by
    how much of the box the family's typical geometry actually fills (a glass
    bottle is a thin shell; a wood block is nearly solid).
    """

    return {family: profile.volume_fraction for family, profile in _PROFILES.items()}


def infer_material_profile(*labels: str | None) -> PhysicsMaterialProfile:
    """Infer a material profile from prim/material labels."""

    # Split camel-case transitions before lowercasing so USD names like
    # FoamPad or SeatCushion tokenize into their words instead of one blob.
    text = " ".join(
        re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label or "") for label in labels
    ).lower()
    if any(token in text for token in ("rubber", "tire", "elastomer")):
        return _PROFILES["rubber"]
    if any(token in text for token in ("plastic", "polymer", "cap")):
        return _PROFILES["plastic"]
    if any(
        token in text
        for token in (
            "metal",
            "steel",
            "aluminum",
            "aluminium",
            "screw",
            "filament",
            "bolt",
            "copper",
        )
    ):
        return _PROFILES["metal"]
    if any(token in text for token in ("wood", "plywood", "timber")):
        return _PROFILES["wood"]
    if any(token in text for token in ("glass", "chamber")):
        return _PROFILES["glass"]
    # Soft porous materials are the inelastic exception the raised
    # rigid-family rebounds must not absorb; cues like "foam pad" or
    # "fabric cover" would otherwise fall to generic. Checked AFTER the
    # explicit rigid families so a compound rigid name that happens to
    # carry a soft word — "metal pillow block", a standard bearing
    # housing — keeps its rigid profile, and matched on whole words
    # (split on non-letters, simple plurals folded) so "cloth" does not
    # classify a clothes_rack or clothing_hook as a 150 kg/m^3
    # dead-rebound softbody while "SeatCushions" still matches.
    words = {
        word[:-1] if word.endswith("s") and len(word) > 4 else word
        for word in re.split(r"[^a-z]+", text)
    }
    if words & {
        "foam",
        "styrofoam",
        "fabric",
        "cloth",
        "textile",
        "cotton",
        "polyester",
        "denim",
        "wool",
        "sponge",
        "felt",
        "cushion",
        "pillow",
    }:
        return _PROFILES["soft"]
    return _PROFILES["generic"]


def physics_policy_prompt() -> str:
    """Return a compact policy prompt for agent-side finalizers."""

    lines = [
        "Infer physics properties per mesh/component from geometry, material bindings, names, and existing schemas.",
        "Author densities and collision shapes from the inferred material family, with realistic per-family restitution; escalate beyond the baseline only when runtime evidence justifies it.",
        "Record density, estimated_mass_kg, static_friction, dynamic_friction, restitution, collision approximation, and rationale for every authored collider.",
        "Validate the authored USD with a real simulation runtime when available; record unavailable runtimes as validation evidence, not as silent success.",
    ]
    return "\n".join(f"- {line}" for line in lines)
