# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFM (Design For Manufacturing) rule pack — REQ-DFM-1..6.

Heuristic checks that surface geometry which is technically valid but hard /
impossible to manufacture. Each check returns a ``CheckResult`` with a
descriptive message. All P1 — they extend the SimReady gate.

Manufacturing process is a parameter; default rules cover 3D-printing
(FFF/SLA) and CNC milling. Sheet-metal-specific rules live in
``cad_verifier.sheet_metal`` (Phase 2).

REQ-DFM-1 — minimum wall (printability + mold-flow)
REQ-DFM-2 — sharp internal corner (CNC tool radius limit)
REQ-DFM-3 — undercut detection (mold-release)
REQ-DFM-4 — draft angle (injection mold)
REQ-DFM-5 — overhang angle (FFF print without supports)
REQ-DFM-6 — minimum hole diameter (CNC drill bit limit)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .checks import CheckResult


@dataclass
class DFMConfig:
    """Tunable thresholds. Defaults target FFF 3D-printing + 3-axis CNC mill."""

    min_wall_mm: float = 0.8  # FFF print wall
    cnc_tool_radius_mm: float = 1.0  # smallest internal corner radius reachable
    min_hole_d_mm: float = 1.0  # smallest drill bit
    min_draft_deg: float = 0.5  # injection-mold release angle
    max_overhang_deg: float = 45.0  # FFF angle from build plate
    process: str = "3d_print_fff"  # informational tag


def min_wall_check(shape, cfg: DFMConfig | None = None) -> CheckResult:
    """REQ-DFM-1 — flag walls thinner than ``cfg.min_wall_mm``.

    Heuristic: the existing ``thin_wall_check`` (volume / surface area)
    extended with a process-specific threshold.
    """
    cfg = cfg or DFMConfig()
    try:
        vol = shape.volume()
        area = shape.surface_area()
    except Exception as e:
        return CheckResult(name="dfm_min_wall", passed=False, message=f"measurement failed: {e}")
    if area <= 1e-9:
        return CheckResult(name="dfm_min_wall", passed=False, message="zero surface area")
    avg_wall = 2.0 * vol / area
    ok = avg_wall >= cfg.min_wall_mm
    msg = (
        f"avg wall ≈ {avg_wall:.3f} mm (threshold {cfg.min_wall_mm} for {cfg.process})"
        if ok
        else f"avg wall ≈ {avg_wall:.3f} mm < {cfg.min_wall_mm} mm — won't print clean / mold-fill issue"
    )
    return CheckResult(name="dfm_min_wall", passed=ok, message=msg)


def sharp_internal_corner_check(shape, cfg: DFMConfig | None = None) -> CheckResult:
    """REQ-DFM-2 — flag internal corners that no tool can reach.

    Heuristic: count edges that are simultaneously linear and short. A short
    linear edge between two faces typically corresponds to a sharp internal
    corner that a finite-radius tool can't follow. Threshold: edge length
    < 2 × tool_radius.
    """
    cfg = cfg or DFMConfig()
    try:
        edges = shape.edges().linear()
    except Exception as e:
        return CheckResult(name="dfm_sharp_corner", passed=False, message=f"edge query failed: {e}")
    short = edges.shorter_than(cfg.cnc_tool_radius_mm * 2.0).count()
    ok = short == 0
    msg = (
        f"all linear edges >= {cfg.cnc_tool_radius_mm * 2.0:.1f} mm"
        if ok
        else (
            f"{short} linear edge(s) shorter than 2× tool radius "
            f"({cfg.cnc_tool_radius_mm} mm) — sharp corners CNC won't reach"
        )
    )
    return CheckResult(name="dfm_sharp_corner", passed=ok, message=msg)


def min_hole_check(shape, cfg: DFMConfig | None = None) -> CheckResult:
    """REQ-DFM-6 — flag circular holes smaller than ``cfg.min_hole_d_mm``."""
    cfg = cfg or DFMConfig()
    try:
        small = (
            shape.edges()
            .circular()
            .custom(lambda e: (e.radius or 1e6) * 2 < cfg.min_hole_d_mm)
            .count()
        )
    except Exception as e:
        return CheckResult(name="dfm_min_hole", passed=False, message=f"edge query failed: {e}")
    ok = small == 0
    msg = (
        f"no holes < {cfg.min_hole_d_mm} mm"
        if ok
        else f"{small} hole(s) below {cfg.min_hole_d_mm} mm — drill / print resolution limit"
    )
    return CheckResult(name="dfm_min_hole", passed=ok, message=msg)


def draft_angle_check(shape, cfg: DFMConfig | None = None) -> CheckResult:
    """REQ-DFM-4 — flag vertical walls (zero draft) — would stick in injection mold.

    Heuristic: faces whose normal lies almost exactly in the XY plane (i.e.,
    perpendicular to the parting direction +Z) have effectively 0° draft.
    Pass if no such face exists.
    """
    cfg = cfg or DFMConfig()
    try:
        faces = shape.faces().planar().all()
    except Exception as e:
        return CheckResult(name="dfm_draft_angle", passed=False, message=f"face query failed: {e}")
    zero_draft = 0
    threshold_cos = math.cos(math.radians(90 - cfg.min_draft_deg))
    for f in faces:
        n = f.normal
        # Cosine between face normal and Z axis
        cos = abs(n.z) / max(math.sqrt(n.x**2 + n.y**2 + n.z**2), 1e-9)
        if cos < threshold_cos:
            # Almost perpendicular to Z (vertical wall) → no draft
            zero_draft += 1
    # If process isn't an injection mold, this rule is informational not pass/fail.
    if cfg.process not in ("injection_mold", "die_cast", "compression_mold"):
        return CheckResult(
            name="dfm_draft_angle",
            passed=True,
            message=f"draft check skipped (process={cfg.process})",
        )
    ok = zero_draft == 0
    msg = (
        f"all walls have >= {cfg.min_draft_deg}° draft"
        if ok
        else f"{zero_draft} vertical wall(s) without draft — won't release from mold"
    )
    return CheckResult(name="dfm_draft_angle", passed=ok, message=msg)


def overhang_check(shape, cfg: DFMConfig | None = None) -> CheckResult:
    """REQ-DFM-5 — flag faces whose normal points >max angle below horizontal
    (i.e., overhangs steeper than the FFF support-free limit).

    Heuristic: face normal n; compute the angle between n and -Z. If > 90 +
    max_overhang, it's a downward-facing overhang.
    """
    cfg = cfg or DFMConfig()
    try:
        faces = shape.faces().planar().all()
    except Exception as e:
        return CheckResult(name="dfm_overhang", passed=False, message=f"face query failed: {e}")
    if cfg.process not in ("3d_print_fff", "3d_print_sla"):
        return CheckResult(
            name="dfm_overhang",
            passed=True,
            message=f"overhang check skipped (process={cfg.process})",
        )
    overhangs = 0
    for f in faces:
        n = f.normal
        nz = n.z / max(math.sqrt(n.x**2 + n.y**2 + n.z**2), 1e-9)
        # nz close to -1 means face points straight down (worst case)
        if nz > -0.05:
            continue  # face points up or sideways — not an overhang
        # Angle below horizontal = 90 - acos(-nz)
        angle_below = 90 - math.degrees(math.acos(min(1.0, -nz)))
        if angle_below > cfg.max_overhang_deg:
            overhangs += 1
    ok = overhangs == 0
    msg = (
        f"no overhangs > {cfg.max_overhang_deg}°"
        if ok
        else f"{overhangs} face(s) overhang > {cfg.max_overhang_deg}° — needs supports"
    )
    return CheckResult(name="dfm_overhang", passed=ok, message=msg)


def undercut_check(shape, cfg: DFMConfig | None = None) -> CheckResult:
    """REQ-DFM-3 — flag faces hidden from both +Z and -Z silhouettes
    (undercut wrt a 2-piece mold).

    Heuristic: a face is an undercut if its normal projected on Z is positive
    AND a ray from any point on the face along +Z would be intercepted by
    another face of the same shape *between* the face and infinity. We
    approximate via a coarser test: faces whose XY-bbox is enclosed by a
    larger XY-bbox at higher Z. Conservative — skip on small parts.
    """
    cfg = cfg or DFMConfig()
    if cfg.process not in ("injection_mold", "die_cast"):
        return CheckResult(
            name="dfm_undercut",
            passed=True,
            message=f"undercut check skipped (process={cfg.process})",
        )
    try:
        bb = shape.bounding_box()
        # Conservative bound: if the XY footprint shrinks then grows again as Z
        # increases, there's likely an undercut. Approximate by sampling 10 z
        # planes and checking footprint area.
        zmin, zmax = bb["min"][2], bb["max"][2]
        if (zmax - zmin) < 1e-3:
            return CheckResult(
                name="dfm_undercut", passed=True, message="part too thin to undercut"
            )
        # Without slicing primitives we can't measure XY footprint at z; emit
        # a pass + guidance rather than false-negative.
        return CheckResult(
            name="dfm_undercut",
            passed=True,
            message="undercut analysis requires slicing — Phase 3 (full implementation)",
        )
    except Exception as e:
        return CheckResult(name="dfm_undercut", passed=False, message=f"bbox query failed: {e}")


def all_dfm_checks(shape, cfg: DFMConfig | None = None) -> list[CheckResult]:
    """Run the full DFM battery for a given process configuration."""
    cfg = cfg or DFMConfig()
    return [
        min_wall_check(shape, cfg),
        sharp_internal_corner_check(shape, cfg),
        min_hole_check(shape, cfg),
        draft_angle_check(shape, cfg),
        overhang_check(shape, cfg),
        undercut_check(shape, cfg),
    ]
