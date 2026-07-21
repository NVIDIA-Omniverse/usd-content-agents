# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Automatic UV generation for meshes without texture coordinates.

Provides multiple UV projection modes for meshes that lack UV unwraps
(common in CAD-derived USD assets from STEP/IGES files).
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

logger = logging.getLogger(__name__)

UV_REPORT_SCHEMA_VERSION = "texture-agent-uv-report.v1"
DIAGNOSTIC_SCHEMA_VERSION = "texture-agent-diagnostic.v1"


class UVProjectionMode(StrEnum):
    """UV projection modes for meshes without UVs."""

    BOX = "box"
    """Box (triplanar) projection: projects from the axis with largest face area.
    Good for CAD parts, mechanical components, and boxy geometry."""

    PLANAR = "planar"
    """Planar projection from the dominant face direction.
    Good for flat or mostly-flat surfaces."""


class UVPreparePolicy(StrEnum):
    """Policies for prepare_uvs mutation behavior."""

    VALIDATE = "validate"
    """Only inspect UVs and fail when meshes are not UV-ready."""

    PRESERVE_OR_FIX = "preserve_or_fix"
    """Preserve valid UVs and apply only safe repairs to existing UVs."""

    GENERATE_MISSING = "generate_missing"
    """Preserve valid UVs, apply safe repairs, and project UVs for missing ones."""

    FORCE_PROJECTION = "force_projection"
    """Project UVs for every mesh, including meshes that already have UVs."""


_FACE_VARYING = "faceVarying"
_SUPPORTED_INTERPOLATIONS = {_FACE_VARYING, "vertex", "varying"}


def make_uv_diagnostic(
    *,
    code: str,
    severity: str,
    prim_path: str,
    message: str,
    recommended_action: str,
    stage: str = "prepare_uvs",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable UV diagnostic record."""
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "code": code,
        "severity": severity,
        "stage": stage,
        "prim_path": prim_path,
        "material_name": None,
        "message": message,
        "recommended_action": recommended_action,
        "details": details or {},
    }


def _mesh_topology_counts(mesh: UsdGeom.Mesh) -> dict[str, int]:
    points = mesh.GetPointsAttr().Get() or []
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get() or []
    return {
        "point_count": len(points),
        "face_count": len(face_vertex_counts),
        "face_vertex_count": len(face_vertex_indices),
    }


def _expected_count_for_interpolation(
    interpolation: str,
    topology: dict[str, int],
) -> int | None:
    if interpolation == _FACE_VARYING:
        return topology["face_vertex_count"]
    if interpolation in ("vertex", "varying"):
        return topology["point_count"]
    if interpolation == "uniform":
        return topology["face_count"]
    if interpolation == "constant":
        return 1
    return None


def _uv_array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.array(value, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return arr.reshape((-1, 2))


def _uv_range(values: np.ndarray) -> dict[str, list[float]] | None:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values).all(axis=1)]
    if finite.size == 0:
        return None
    return {
        "min": [float(finite[:, 0].min()), float(finite[:, 1].min())],
        "max": [float(finite[:, 0].max()), float(finite[:, 1].max())],
    }


def _indices_are_valid(indices: Any, value_count: int) -> bool:
    if indices is None:
        return False
    idx = np.array(indices, dtype=np.int64)
    if idx.size == 0:
        return False
    return bool(idx.min() >= 0 and idx.max() < value_count)


def _face_varying_compatible(report: dict[str, Any]) -> bool:
    expected = report["face_vertex_count"]
    if expected <= 0:
        return False
    if report["indexed"]:
        return (
            report["index_count"] == expected
            and report["indices_valid"]
            and report["value_count"] > 0
        )
    return report["value_count"] == expected


def inspect_uvs_for_mesh(prim: Usd.Prim) -> dict[str, Any]:
    """Inspect a mesh prim's UV readiness.

    The returned dictionary is intentionally JSON-serializable so it can be
    written directly into ``uv_report.json`` and surfaced by service status APIs.
    """
    mesh = UsdGeom.Mesh(prim)
    topology = _mesh_topology_counts(mesh)
    api = UsdGeom.PrimvarsAPI(prim)
    st = api.GetPrimvar("st")

    report: dict[str, Any] = {
        "prim_path": str(prim.GetPath()),
        **topology,
        "has_uvs": False,
        "interpolation": "",
        "indexed": False,
        "value_count": 0,
        "index_count": 0,
        "expected_count": None,
        "indices_valid": True,
        "uv_range": None,
        "out_of_range": False,
        "has_non_finite": False,
        "issues": [],
        "diagnostics": [],
        "recommended_action": "preserve",
        "status": "valid",
    }

    issues: list[str] = report["issues"]
    diagnostics: list[dict[str, Any]] = report["diagnostics"]
    prim_path = str(prim.GetPath())
    if topology["point_count"] == 0 or topology["face_vertex_count"] == 0:
        issues.append("UV_EMPTY_TOPOLOGY")
        diagnostics.append(
            make_uv_diagnostic(
                code="UV_EMPTY_TOPOLOGY",
                severity="error",
                prim_path=prim_path,
                message="Mesh has no usable face topology for UV preparation.",
                recommended_action="Provide mesh topology before texture generation.",
                details=topology,
            )
        )
        report["status"] = "invalid"
        report["recommended_action"] = "provide_mesh_topology"
        return report

    if not st or not st.IsDefined() or st.Get() is None:
        issues.append("UV_MISSING_ST")
        diagnostics.append(
            make_uv_diagnostic(
                code="UV_MISSING_ST",
                severity="error",
                prim_path=prim_path,
                message="Mesh has no primvars:st UV coordinates.",
                recommended_action=(
                    "Set texture.uv_policy=generate_missing or provide a UV-ready asset."
                ),
            )
        )
        report["status"] = "missing"
        report["recommended_action"] = "generate_missing"
        return report

    values = _uv_array(st.Get())
    interpolation = st.GetInterpolation() or ""
    indices = st.GetIndices() if st.IsIndexed() else None
    report.update(
        {
            "has_uvs": len(values) > 0,
            "interpolation": interpolation,
            "indexed": bool(st.IsIndexed()),
            "value_count": int(len(values)),
            "index_count": int(len(indices) if indices is not None else 0),
            "expected_count": _expected_count_for_interpolation(
                interpolation, topology
            ),
        }
    )

    if len(values) == 0:
        issues.append("UV_BAD_VALUE_COUNT")
        diagnostics.append(
            make_uv_diagnostic(
                code="UV_BAD_VALUE_COUNT",
                severity="error",
                prim_path=prim_path,
                message="Mesh primvars:st exists but has no UV values.",
                recommended_action=(
                    "Regenerate missing UVs or provide a UV-ready asset."
                ),
                details={"value_count": 0},
            )
        )
        report["status"] = "missing"
        report["recommended_action"] = "generate_missing"
        return report

    if st.IsIndexed():
        report["indices_valid"] = _indices_are_valid(indices, len(values))
        if not report["indices_valid"]:
            issues.append("UV_BAD_INDEX_COUNT")
            diagnostics.append(
                make_uv_diagnostic(
                    code="UV_BAD_INDEX_COUNT",
                    severity="error",
                    prim_path=prim_path,
                    message="Indexed UV primvar has missing or out-of-range indices.",
                    recommended_action="Fix the indexed primvar or force projection UVs.",
                    details={
                        "value_count": report["value_count"],
                        "index_count": report["index_count"],
                    },
                )
            )

    finite_mask = np.isfinite(values).all(axis=1)
    report["has_non_finite"] = not bool(finite_mask.all())
    if report["has_non_finite"]:
        issues.append("UV_NAN_INF")
        diagnostics.append(
            make_uv_diagnostic(
                code="UV_NAN_INF",
                severity="error",
                prim_path=prim_path,
                message="UV values contain NaN or infinite coordinates.",
                recommended_action="Fix the source UVs or force projection UVs.",
            )
        )

    report["uv_range"] = _uv_range(values)
    if report["uv_range"] is not None:
        uv_min = report["uv_range"]["min"]
        uv_max = report["uv_range"]["max"]
        report["out_of_range"] = (
            uv_min[0] < 0.0 or uv_min[1] < 0.0 or uv_max[0] > 1.0 or uv_max[1] > 1.0
        )
        if report["out_of_range"]:
            issues.append("UV_OUT_OF_RANGE")
            diagnostics.append(
                make_uv_diagnostic(
                    code="UV_OUT_OF_RANGE",
                    severity="warning",
                    prim_path=prim_path,
                    message="UV coordinates extend outside the [0, 1] range.",
                    recommended_action=(
                        "Preserve tiled UVs or enable texture.uv_normalize_out_of_range."
                    ),
                    details={"uv_range": report["uv_range"]},
                )
            )

    expected = report["expected_count"]
    actual_count = report["index_count"] if report["indexed"] else report["value_count"]
    if interpolation == "constant":
        if _face_varying_compatible(report):
            issues.append("UV_REPAIRABLE_CONSTANT_INTERPOLATION")
            diagnostics.append(
                make_uv_diagnostic(
                    code="UV_BAD_INTERPOLATION",
                    severity="warning",
                    prim_path=prim_path,
                    message=(
                        "UV interpolation is constant but value/index counts are "
                        "compatible with faceVarying."
                    ),
                    recommended_action="Repair interpolation to faceVarying.",
                    details={
                        "interpolation": interpolation,
                        "value_count": report["value_count"],
                        "index_count": report["index_count"],
                        "expected_count": report["face_vertex_count"],
                    },
                )
            )
            report["recommended_action"] = "fix_interpolation"
        else:
            issues.append("UV_BAD_INTERPOLATION")
            diagnostics.append(
                make_uv_diagnostic(
                    code="UV_BAD_INTERPOLATION",
                    severity="error",
                    prim_path=prim_path,
                    message=(
                        "UV interpolation is constant and cannot be safely changed "
                        "to faceVarying because counts are incompatible."
                    ),
                    recommended_action="Fix the source UVs or force projection UVs.",
                    details={
                        "interpolation": interpolation,
                        "value_count": report["value_count"],
                        "index_count": report["index_count"],
                        "expected_count": report["face_vertex_count"],
                    },
                )
            )
    elif interpolation not in _SUPPORTED_INTERPOLATIONS:
        issues.append("UV_BAD_INTERPOLATION")
        diagnostics.append(
            make_uv_diagnostic(
                code="UV_BAD_INTERPOLATION",
                severity="error",
                prim_path=prim_path,
                message=f"Unsupported UV interpolation: {interpolation!r}.",
                recommended_action="Fix the source UV interpolation or force projection UVs.",
                details={"interpolation": interpolation},
            )
        )
    elif expected is not None and actual_count != expected:
        issues.append("UV_BAD_VALUE_COUNT")
        diagnostics.append(
            make_uv_diagnostic(
                code="UV_BAD_VALUE_COUNT",
                severity="error",
                prim_path=prim_path,
                message="UV value/index count does not match interpolation topology.",
                recommended_action="Fix the source UVs or force projection UVs.",
                details={
                    "interpolation": interpolation,
                    "actual_count": actual_count,
                    "expected_count": expected,
                },
            )
        )

    blocking = {
        "UV_BAD_INTERPOLATION",
        "UV_BAD_INDEX_COUNT",
        "UV_NAN_INF",
        "UV_BAD_VALUE_COUNT",
    }
    if any(issue in blocking for issue in issues):
        report["status"] = "invalid"
        if report["recommended_action"] == "preserve":
            report["recommended_action"] = "force_projection_or_fix_asset"
    elif "UV_REPAIRABLE_CONSTANT_INTERPOLATION" in issues:
        report["status"] = "repairable"

    return report


def inspect_uvs_for_stage(stage: Usd.Stage) -> dict[str, Any]:
    """Inspect UV readiness for every mesh in a stage."""
    meshes = [
        inspect_uvs_for_mesh(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh) and not prim.IsInstanceProxy()
    ]
    summary = {
        "total_meshes": len(meshes),
        "valid": sum(1 for item in meshes if item["status"] == "valid"),
        "repairable": sum(1 for item in meshes if item["status"] == "repairable"),
        "missing": sum(1 for item in meshes if item["status"] == "missing"),
        "invalid": sum(1 for item in meshes if item["status"] == "invalid"),
        "out_of_range": sum(1 for item in meshes if item["out_of_range"]),
        "non_finite": sum(1 for item in meshes if item["has_non_finite"]),
        "indexed": sum(1 for item in meshes if item["indexed"]),
    }
    return {
        "schema_version": UV_REPORT_SCHEMA_VERSION,
        "summary": summary,
        "meshes": meshes,
    }


def _compute_face_normals(
    points: np.ndarray, fvi: np.ndarray, fvc: np.ndarray
) -> np.ndarray:
    """Compute per-face normals from mesh topology."""
    normals = []
    idx = 0
    for count in fvc:
        if count < 3:
            normals.append(np.array([0.0, 1.0, 0.0]))
            idx += count
            continue
        v0 = points[fvi[idx]]
        v1 = points[fvi[idx + 1]]
        v2 = points[fvi[idx + 2]]
        n = np.cross(v1 - v0, v2 - v0)
        length = np.linalg.norm(n)
        if length > 0:
            n = n / length
        else:
            n = np.array([0.0, 1.0, 0.0])
        normals.append(n)
        idx += count
    return np.array(normals)


def generate_box_uvs(
    points: np.ndarray,
    fvi: np.ndarray,
    fvc: np.ndarray,
    margin: float = 0.05,
) -> np.ndarray:
    """Generate box-projected UVs for a mesh.

    Projects each face-vertex from the axis direction that best matches
    the face normal (triplanar projection, picking the dominant axis).

    Args:
        points: Vertex positions (N, 3).
        fvi: Face vertex indices.
        fvc: Face vertex counts.
        margin: UV margin from edges (default 0.05).

    Returns:
        UV array (len(fvi), 2) in [margin, 1-margin].
    """
    face_normals = _compute_face_normals(points, fvi, fvc)

    uvs = np.zeros((len(fvi), 2), dtype=np.float32)

    # Compute global bounding box for consistent scaling
    all_pts = points[fvi]
    bbox_min = all_pts.min(axis=0)
    bbox_max = all_pts.max(axis=0)
    bbox_range = bbox_max - bbox_min
    bbox_range[bbox_range == 0] = 1.0

    idx = 0
    for face_idx, count in enumerate(fvc):
        normal = face_normals[face_idx]
        abs_normal = np.abs(normal)
        dominant = np.argmax(abs_normal)

        # Choose projection axes (the two non-dominant axes)
        if dominant == 0:  # X dominant → project on YZ
            ax_u, ax_v = 1, 2
        elif dominant == 1:  # Y dominant → project on XZ
            ax_u, ax_v = 0, 2
        else:  # Z dominant → project on XY
            ax_u, ax_v = 0, 1

        for i in range(count):
            pt = points[fvi[idx + i]]
            u = (pt[ax_u] - bbox_min[ax_u]) / bbox_range[ax_u]
            v = (pt[ax_v] - bbox_min[ax_v]) / bbox_range[ax_v]
            uvs[idx + i] = [
                u * (1.0 - 2 * margin) + margin,
                v * (1.0 - 2 * margin) + margin,
            ]

        idx += count

    return uvs


def generate_planar_uvs(
    points: np.ndarray,
    fvi: np.ndarray,
    margin: float = 0.05,
) -> np.ndarray:
    """Generate planar-projected UVs from the direction of least variance.

    Args:
        points: Vertex positions (N, 3).
        fvi: Face vertex indices.
        margin: UV margin.

    Returns:
        UV array (len(fvi), 2).
    """
    face_verts = points[fvi]
    bbox_min = face_verts.min(axis=0)
    bbox_max = face_verts.max(axis=0)
    bbox_range = bbox_max - bbox_min
    bbox_range[bbox_range == 0] = 1.0

    # Pick the 2 axes with largest extent
    smallest_axis = np.argmin(bbox_range)
    axes = [i for i in range(3) if i != smallest_axis]

    normalized = (face_verts - bbox_min) / bbox_range
    uvs = normalized[:, axes]
    uvs = uvs * (1.0 - 2 * margin) + margin

    return uvs.astype(np.float32)


def generate_uvs_for_mesh(
    prim: Usd.Prim,
    mode: UVProjectionMode = UVProjectionMode.BOX,
    overwrite_existing: bool = False,
) -> bool:
    """Generate UVs for a single mesh prim.

    Args:
        prim: A UsdGeom.Mesh prim.
        mode: UV projection mode.
        overwrite_existing: If true, replace an existing ``primvars:st``.

    Returns:
        True if UVs were generated, False if UVs already existed or the mesh
        cannot be projected.
    """
    api = UsdGeom.PrimvarsAPI(prim)
    st = api.GetPrimvar("st")
    if not overwrite_existing and st and st.IsDefined() and st.Get() is not None:
        existing = np.array(st.Get())
        if len(existing) > 0:
            return False  # UVs already exist

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    fvi = mesh.GetFaceVertexIndicesAttr().Get()
    fvc = mesh.GetFaceVertexCountsAttr().Get()

    if points is None or fvi is None or fvc is None:
        return False

    pts = np.array(points)
    indices = np.array(fvi)
    counts = np.array(fvc)

    if len(indices) == 0:
        return False

    if mode == UVProjectionMode.BOX:
        uvs = generate_box_uvs(pts, indices, counts)
    elif mode == UVProjectionMode.PLANAR:
        uvs = generate_planar_uvs(pts, indices)
    else:
        raise ValueError(f"Unknown UV projection mode: {mode}")

    vt_uvs = Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uvs])

    st_pv = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
    st_pv.SetInterpolation("faceVarying")
    if st_pv.IsIndexed():
        st_pv.BlockIndices()
    st_pv.Set(vt_uvs)

    return True


def generate_uvs_for_stage(
    stage: Usd.Stage,
    mode: UVProjectionMode = UVProjectionMode.BOX,
    overwrite_existing: bool = False,
    target_prim_paths: list[str] | tuple[str, ...] | None = None,
) -> int:
    """Generate projection UVs for meshes in a stage.

    Args:
        stage: USD stage to process.
        mode: UV projection mode.
        overwrite_existing: If true, replace existing UVs on every mesh.
        target_prim_paths: Optional mesh prim paths or ancestor paths to limit
            mutation to a subset of the stage.

    Returns:
        Number of meshes that received new UVs.
    """
    count = 0
    target_paths = _normalize_target_paths(target_prim_paths)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not _prim_matches_target_paths(prim, target_paths):
            continue
        if not _prepare_mutable_prim(prim, "UV generation"):
            continue
        if generate_uvs_for_mesh(prim, mode, overwrite_existing=overwrite_existing):
            count += 1
            logger.debug("Generated %s UVs for %s", mode.value, prim.GetPath())

    if count > 0:
        logger.info("Generated %s UVs for %d meshes", mode.value, count)
    return count


def _normalize_target_paths(
    target_prim_paths: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if not target_prim_paths:
        return ()
    normalized: list[str] = []
    for raw_path in target_prim_paths:
        path = str(raw_path).strip()
        if path:
            normalized.append(path.rstrip("/"))
    return tuple(dict.fromkeys(normalized))


def _prim_matches_target_paths(prim: Usd.Prim, target_paths: tuple[str, ...]) -> bool:
    if not target_paths:
        return True
    prim_path = str(prim.GetPath())
    return any(
        prim_path == target_path or prim_path.startswith(f"{target_path}/")
        for target_path in target_paths
    )


def fix_uv_interpolation(
    stage: Usd.Stage,
    target_prim_paths: list[str] | tuple[str, ...] | None = None,
) -> int:
    """Safely fix meshes with compatible ``constant`` UV interpolation.

    A ``constant`` primvar with one UV value is not face-varying data. This
    function only flips interpolation when the value or index count already
    matches face-vertex topology.

    Returns:
        Number of meshes fixed.
    """
    count = 0
    target_paths = _normalize_target_paths(target_prim_paths)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not _prim_matches_target_paths(prim, target_paths):
            continue
        if not _prepare_mutable_prim(prim, "UV interpolation repair"):
            continue
        api = UsdGeom.PrimvarsAPI(prim)
        st = api.GetPrimvar("st")
        if not st or not st.IsDefined() or st.GetInterpolation() != "constant":
            continue
        report = inspect_uvs_for_mesh(prim)
        if report["status"] == "repairable":
            st.SetInterpolation("faceVarying")
            count += 1

    if count > 0:
        logger.info("Fixed UV interpolation on %d meshes", count)
    return count


def normalize_uvs(
    stage: Usd.Stage,
    margin: float = 0.025,
    target_prim_paths: list[str] | tuple[str, ...] | None = None,
) -> int:
    """Normalize UV coordinates to [margin, 1-margin] for meshes with out-of-range UVs.

    Returns:
        Number of meshes normalized.
    """
    count = 0
    target_paths = _normalize_target_paths(target_prim_paths)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not _prim_matches_target_paths(prim, target_paths):
            continue
        if not _prepare_mutable_prim(prim, "UV normalization"):
            continue
        api = UsdGeom.PrimvarsAPI(prim)
        st = api.GetPrimvar("st")
        if not st or not st.IsDefined():
            continue
        uvs = np.array(st.Get())
        if len(uvs) == 0:
            continue
        if not np.isfinite(uvs).all():
            logger.warning("Skipping UV normalization for non-finite UVs: %s", prim)
            continue

        u_min, u_max = uvs[:, 0].min(), uvs[:, 0].max()
        v_min, v_max = uvs[:, 1].min(), uvs[:, 1].max()

        # Only normalize if out of [0, 1] range
        if u_min >= 0 and u_max <= 1 and v_min >= 0 and v_max <= 1:
            continue

        u_range = u_max - u_min or 1.0
        v_range = v_max - v_min or 1.0
        uvs[:, 0] = (uvs[:, 0] - u_min) / u_range * (1 - 2 * margin) + margin
        uvs[:, 1] = (uvs[:, 1] - v_min) / v_range * (1 - 2 * margin) + margin

        vt_uvs = Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uvs])
        st.Set(vt_uvs)
        count += 1

    if count > 0:
        logger.info("Normalized UVs on %d meshes", count)
    return count


def _prepare_mutable_prim(prim: Usd.Prim, action: str) -> bool:
    if prim.IsInstanceProxy():
        logger.debug(
            "Skipping %s for read-only instance proxy %s", action, prim.GetPath()
        )
        return False
    if prim.IsInstance() or prim.IsInstanceable():
        prim.SetInstanceable(False)
        logger.debug(
            "Disabled instanceability before %s for %s", action, prim.GetPath()
        )
    return True
