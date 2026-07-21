# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physics workflow inference policy shared by skills and wrappers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsMaterialProfile:
    """Conservative physics defaults for one inferred material family."""

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
        restitution=0.08,
        volume_fraction=0.04,
        rationale="Glass shell geometry is usually thin and brittle, so use glass density with a low filled-volume fraction and low restitution.",
    ),
    "metal": PhysicsMaterialProfile(
        family="metal",
        density=7800.0,
        static_friction=0.55,
        dynamic_friction=0.42,
        restitution=0.12,
        volume_fraction=0.12,
        rationale="Metal geometry is dense and moderately rough; small fixtures and threads are often partly hollow or thin.",
    ),
    "plastic": PhysicsMaterialProfile(
        family="plastic",
        density=1050.0,
        static_friction=0.60,
        dynamic_friction=0.45,
        restitution=0.20,
        volume_fraction=0.40,
        rationale="Plastic parts are light with moderate friction and mild rebound.",
    ),
    "rubber": PhysicsMaterialProfile(
        family="rubber",
        density=1100.0,
        static_friction=1.00,
        dynamic_friction=0.80,
        restitution=0.35,
        volume_fraction=0.35,
        rationale="Rubber has high friction and noticeable rebound, with flexible or hollow molded geometry.",
    ),
    "wood": PhysicsMaterialProfile(
        family="wood",
        density=650.0,
        static_friction=0.55,
        dynamic_friction=0.40,
        restitution=0.18,
        volume_fraction=0.60,
        rationale="Wood is medium density with moderate friction.",
    ),
    "generic": PhysicsMaterialProfile(
        family="generic",
        density=1000.0,
        static_friction=0.50,
        dynamic_friction=0.40,
        restitution=0.15,
        volume_fraction=0.25,
        rationale="Fallback profile for unknown material cues; values are conservative and solver-friendly.",
    ),
}


def infer_material_profile(*labels: str | None) -> PhysicsMaterialProfile:
    """Infer a material profile from prim/material labels."""

    text = " ".join(label or "" for label in labels).lower()
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
    return _PROFILES["generic"]


def physics_policy_prompt() -> str:
    """Return a compact policy prompt for agent-side finalizers."""

    lines = [
        "Infer physics properties per mesh/component from geometry, material bindings, names, and existing schemas.",
        "Use conservative densities and collision shapes unless runtime evidence justifies a stronger claim.",
        "Record density, estimated_mass_kg, static_friction, dynamic_friction, restitution, collision approximation, and rationale for every authored collider.",
        "Validate the authored USD with a real simulation runtime when available; record unavailable runtimes as validation evidence, not as silent success.",
    ]
    return "\n".join(f"- {line}" for line in lines)
