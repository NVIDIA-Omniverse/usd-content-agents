# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static geometric checks over backend-neutral shape objects.

These run fast (sub-second each) and don't require a simulator. Physics
drop-test (via Isaac Sim through Kit Inspector) is a Phase 2 addition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class ShapeLike(Protocol):
    """Minimum shape contract required by the static verifier."""

    def volume(self) -> float: ...

    def surface_area(self) -> float: ...

    def bounding_box(self) -> dict[str, tuple[float, float, float]]: ...

    def is_valid(self) -> bool: ...

    def intersect(self, other: ShapeLike) -> ShapeLike: ...


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""


@dataclass
class VerifyReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        p = sum(1 for c in self.checks if c.passed)
        return f"{p}/{len(self.checks)} checks passed"


def has_volume(shape: ShapeLike, min_volume: float = 1e-6) -> CheckResult:
    vol = shape.volume()
    return CheckResult(
        name="has_volume",
        passed=vol > min_volume,
        message=f"volume={vol:.6f}" if vol > min_volume else f"volume {vol:.6f} <= {min_volume}",
    )


def volume_check(
    shape: ShapeLike, min_volume: float = 0.0, max_volume: float | None = None
) -> CheckResult:
    vol = shape.volume()
    ok = vol >= min_volume
    if max_volume is not None:
        ok = ok and vol <= max_volume
    msg = f"volume={vol:.4f}" if ok else f"volume {vol:.4f} out of [{min_volume}, {max_volume}]"
    return CheckResult(name="volume_check", passed=ok, message=msg)


def bbox_check(shape: ShapeLike, min_size: float = 1e-3, max_size: float = 1e6) -> CheckResult:
    bb = shape.bounding_box()
    dx = bb["max"][0] - bb["min"][0]
    dy = bb["max"][1] - bb["min"][1]
    dz = bb["max"][2] - bb["min"][2]
    sizes = (dx, dy, dz)
    ok = all(min_size <= s <= max_size for s in sizes)
    msg = f"extents={sizes}" if ok else f"extents {sizes} out of [{min_size}, {max_size}]"
    return CheckResult(name="bbox_check", passed=ok, message=msg)


def manifold_check(shape: ShapeLike) -> CheckResult:
    """Weakly check validity through the owning backend.

    Full manifold analysis requires additional OCP ``ShapeAnalysis`` evidence
    and is deferred to Phase 2.
    """
    ok = shape.is_valid()
    return CheckResult(
        name="manifold_check",
        passed=ok,
        message="valid" if ok else "shape.isValid() returned False",
    )


def interference_check(a: ShapeLike, b: ShapeLike, tolerance: float = 1e-6) -> CheckResult:
    try:
        inter = a.intersect(b)
        overlap = inter.volume()
    except Exception as e:
        return CheckResult(
            name="interference_check", passed=False, message=f"intersect failed: {e}"
        )
    ok = overlap <= tolerance
    msg = f"overlap={overlap:.6f}" if ok else f"overlap {overlap:.6f} > {tolerance}"
    return CheckResult(name="interference_check", passed=ok, message=msg)


def thin_wall_check(
    shape: ShapeLike,
    min_wall_mm: float = 0.6,
) -> CheckResult:
    """Heuristic thin-wall detection.

    Compares part volume against a "thick-coat" estimator: if the part is
    significantly thicker than ``min_wall_mm`` everywhere, then eroding it
    by ``min_wall_mm/2`` should still leave a positive volume. We approximate
    the erosion via a scale based on bbox + thickness ratio because this check
    does not invoke a topology-changing 3D offset. Returns a soft pass when the ratio
    of (volume / surface_area) — a length scale — is above ``min_wall_mm/2``.
    """
    try:
        vol = shape.volume()
        area = shape.surface_area()
    except Exception as e:
        return CheckResult(name="thin_wall_check", passed=False, message=f"measurement failed: {e}")
    if area <= 1e-9:
        return CheckResult(name="thin_wall_check", passed=False, message="zero surface area")
    # 2 * volume / surface_area is a rough average wall thickness for plate-ish parts
    avg_thickness = 2.0 * vol / area
    ok = avg_thickness >= min_wall_mm
    msg = (
        f"avg wall ≈ {avg_thickness:.3f} mm"
        if ok
        else f"avg wall ≈ {avg_thickness:.3f} mm < min {min_wall_mm} mm"
    )
    return CheckResult(name="thin_wall_check", passed=ok, message=msg)


def oversize_check(
    shape: ShapeLike,
    max_dim_mm: float = 10_000.0,
) -> CheckResult:
    """Reject parts whose any-axis bbox extent exceeds ``max_dim_mm``.

    Default 10 m catches accidental unit errors (cm / inch / m authored as mm).
    """
    bb = shape.bounding_box()
    extents = (
        bb["max"][0] - bb["min"][0],
        bb["max"][1] - bb["min"][1],
        bb["max"][2] - bb["min"][2],
    )
    longest = max(extents)
    ok = longest <= max_dim_mm
    msg = (
        f"longest axis = {longest:.1f} mm"
        if ok
        else f"longest axis {longest:.1f} mm > {max_dim_mm} (unit error?)"
    )
    return CheckResult(name="oversize_check", passed=ok, message=msg)


def self_intersect_check(shape: ShapeLike) -> CheckResult:
    """Detect a malformed solid where the boolean-self produces a different volume."""
    try:
        v0 = shape.volume()
        if v0 < 1e-9:
            return CheckResult(name="self_intersect_check", passed=False, message="zero volume")
        # `shape ∪ shape` should equal `shape` — if not, geometry has issues.
        vu = shape.union(shape).volume()
        ratio = abs(vu - v0) / v0
        ok = ratio < 0.01
        msg = (
            f"vol stable ({v0:.3f}=={vu:.3f})"
            if ok
            else f"self-union shifts vol by {ratio * 100:.1f}% (non-manifold?)"
        )
        return CheckResult(name="self_intersect_check", passed=ok, message=msg)
    except Exception as e:
        return CheckResult(name="self_intersect_check", passed=False, message=f"check failed: {e}")


def has_planar_mounting_face_check(
    shape: ShapeLike,
    min_face_area: float = 100.0,
) -> CheckResult:
    """At least one planar face larger than ``min_face_area`` mm² exists.

    Useful gate for parts that need to mount flat (brackets, plates, etc.).
    Optional — call directly rather than including in `all_checks`.
    """
    try:
        faces = shape.faces().planar().area_between(min_face_area, 1e9).count()
    except Exception as e:
        return CheckResult(
            name="planar_mounting_face", passed=False, message=f"face query failed: {e}"
        )
    ok = faces > 0
    msg = (
        f"{faces} planar face(s) >= {min_face_area} mm²"
        if ok
        else f"no planar face exceeds {min_face_area} mm²"
    )
    return CheckResult(name="planar_mounting_face", passed=ok, message=msg)


def all_checks(shape: ShapeLike) -> VerifyReport:
    """Run the standard battery of static checks against a shape."""
    return VerifyReport(
        checks=[
            has_volume(shape),
            manifold_check(shape),
            bbox_check(shape),
            volume_check(shape),
            oversize_check(shape),
            self_intersect_check(shape),
            thin_wall_check(shape),
        ]
    )
