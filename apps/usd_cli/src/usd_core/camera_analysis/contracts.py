# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed, USD-independent contracts for camera analysis."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

Matrix4 = tuple[tuple[float, float, float, float], ...]
AnalysisRole = Literal["obstacle", "floor", "target", "helper"]

# Shared hard limits are import-light so USD ingestion can reject oversized raw
# topology before NumPy triangulation and Newton can enforce the same final model cap.
MAX_ANALYSIS_SHAPES = 100_000
MAX_ANALYSIS_VERTICES = 5_000_000
MAX_ANALYSIS_TRIANGLES = 10_000_000
MAX_ANALYSIS_GEOMETRY_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class SceneAnalysisPolicy:
    """USD-independent shape-selection and analysis-role policy.

    Paths use exact USD descendant semantics (``/Shelf`` does not match
    ``/Shelf2``). ``include_paths`` limits admitted shapes when nonempty;
    ``exclude_paths`` removes matching shapes. Camera discovery is intentionally
    independent of these geometry filters: the IR records every camera admitted by
    stage visibility and purpose policy, and callers select cameras separately.
    Supported helpers remain represented in the IR for provenance and identity, but
    visibility backends must not let them occlude; unsupported boundable helpers are
    explicitly ignored instead of being mistaken for obstacles. Target and floor
    paths assign their corresponding analysis roles; memberships may overlap (a target
    can also be a requested floor surface), while ``ShapeIR.analysis_role`` retains the
    primary helper/target/floor precedence. Every other admitted shape is an obstacle.
    """

    scope_paths: tuple[str, ...] = ()
    include_paths: tuple[str, ...] = ()
    exclude_paths: tuple[str, ...] = ()
    helper_paths: tuple[str, ...] = ()
    target_paths: tuple[str, ...] = ()
    floor_paths: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, list[str]]:
        """Return the canonical JSON-compatible policy configuration."""

        return {
            "scope_paths": list(self.scope_paths),
            "include_paths": list(self.include_paths),
            "exclude_paths": list(self.exclude_paths),
            "helper_paths": list(self.helper_paths),
            "target_paths": list(self.target_paths),
            "floor_paths": list(self.floor_paths),
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MeshResource:
    """One reusable triangle mesh in canonical metre space."""

    resource_id: str
    vertices: np.ndarray
    indices: np.ndarray


@dataclass(frozen=True)
class ShapeIR:
    """One composed shape instance and its stable USD identity.

    ``transform`` follows USD's row-vector convention.  It maps a local point in
    canonical metres to canonical Z-up world metres.  Mesh resources remain shared;
    instances carry their own transform instead of being flattened into one world-space
    triangle soup.
    """

    shape_id: int
    prim_path: str
    kind: str
    transform: Matrix4
    mesh_resource_id: str | None = None
    parameters: dict[str, float] = field(default_factory=dict)
    purpose: str = "default"
    instance_proxy: bool = False
    analysis_role: AnalysisRole = "obstacle"
    in_scope: bool = False


@dataclass(frozen=True)
class CameraIR:
    """A camera observed from the composed USD stage.

    The IR retains the authored projection even when it is not supported by the
    pinhole visibility model.  That lets unrelated cameras coexist with geometry
    analysis while callers that explicitly select one receive a precise error.
    """

    prim_path: str
    transform: Matrix4
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    clipping_range_m: tuple[float, float]
    horizontal_aperture_offset_mm: float = 0.0
    vertical_aperture_offset_mm: float = 0.0
    projection: str = "perspective"


@dataclass(frozen=True)
class SceneAnalysisIR:
    """Canonical, typed scene input shared by analytic and RTX backends."""

    meters_per_unit: float
    source_up_axis: str
    canonical_up_axis: str
    stage_to_canonical: Matrix4
    canonical_to_stage: Matrix4
    meshes: tuple[MeshResource, ...]
    shapes: tuple[ShapeIR, ...]
    cameras: tuple[CameraIR, ...]
    source_digest: str
    policy: SceneAnalysisPolicy = field(default_factory=SceneAnalysisPolicy)

    @property
    def shape_path_by_id(self) -> dict[int, str]:
        return {shape.shape_id: shape.prim_path for shape in self.shapes}


@dataclass(frozen=True)
class RayHits:
    """Closest-hit result returned by a visibility backend."""

    distances_m: np.ndarray
    shape_ids: np.ndarray
    normals: np.ndarray | None = None


@dataclass(frozen=True)
class CameraPose:
    """A camera pose and pinhole model in canonical metre space."""

    prim_path: str | None
    position_m: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    forward: tuple[float, float, float]
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    clipping_range_m: tuple[float, float] = (0.01, 1.0e6)
    horizontal_aperture_offset_mm: float = 0.0
    vertical_aperture_offset_mm: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.focal_length_mm) or self.focal_length_mm <= 0.0:
            raise ValueError("camera focal length must be finite and positive")

    @property
    def horizontal_tan_half_fov(self) -> float:
        return self.horizontal_aperture_mm / (2.0 * self.focal_length_mm)

    @property
    def vertical_tan_half_fov(self) -> float:
        return self.vertical_aperture_mm / (2.0 * self.focal_length_mm)

    @property
    def horizontal_tan_bounds(self) -> tuple[float, float]:
        half = self.horizontal_aperture_mm * 0.5
        return (
            (-half + self.horizontal_aperture_offset_mm) / self.focal_length_mm,
            (half + self.horizontal_aperture_offset_mm) / self.focal_length_mm,
        )

    @property
    def vertical_tan_bounds(self) -> tuple[float, float]:
        half = self.vertical_aperture_mm * 0.5
        return (
            (-half + self.vertical_aperture_offset_mm) / self.focal_length_mm,
            (half + self.vertical_aperture_offset_mm) / self.focal_length_mm,
        )


@dataclass(frozen=True)
class CameraBatchRequest:
    """Pinhole camera observations requested from a visibility backend.

    Output image axes are ``(camera, y, x)``. Channels are opt-in so a backend
    can skip unused sensor work.
    """

    cameras: tuple[CameraPose, ...]
    width: int
    height: int
    include_depth: bool = True
    include_shape_ids: bool = True
    include_normals: bool = False
    include_albedo: bool = False


@dataclass(frozen=True)
class CameraObservation:
    """Structured single-world camera result in canonical metre space.

    ``depth_m`` is Euclidean ray-hit distance, matching OVRTX
    ``DistanceToCameraSD`` rather than forward/image-plane depth. Depth and
    SceneAnalysisIR shape-ID misses use ``-1``. Normals are world-space vectors
    and zero on misses. Each camera's clipping range is applied along its forward
    axis before all channels are returned. Albedo is unshaded sRGB uint8 RGBA and
    transparent black on misses. Every present scalar channel has shape
    ``(C, H, W)``; normals have shape ``(C, H, W, 3)`` and albedo has shape
    ``(C, H, W, 4)``. Channels not requested by ``CameraBatchRequest`` are
    ``None``.
    """

    depth_m: np.ndarray | None = None
    shape_ids: np.ndarray | None = None
    normals: np.ndarray | None = None
    albedo_rgba: np.ndarray | None = None


@runtime_checkable
class VisibilityBackend(Protocol):
    """Import-light contract shared by analytic and verification backends."""

    @property
    def versions(self) -> dict[str, str | bool]: ...

    def evaluate_rays(
        self,
        origins_m: np.ndarray,
        directions: np.ndarray,
        *,
        include_normals: bool = False,
    ) -> RayHits: ...

    def render_cameras(self, request: CameraBatchRequest) -> CameraObservation: ...
