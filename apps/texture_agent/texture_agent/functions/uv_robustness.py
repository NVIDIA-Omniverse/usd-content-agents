# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UV robustness evidence helpers for Texture Agent evaluation runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pxr import Gf, Usd, UsdGeom

from texture_agent.functions.uv_generation import inspect_uvs_for_stage
from texture_agent.tasks.prepare_uvs import PrepareUVsTask, UVPreparationError

UV_ROBUSTNESS_SCHEMA_VERSION = "texture-agent-uv-robustness.v1"
UV_ROBUSTNESS_MANIFEST_SCHEMA_VERSION = "texture-agent-uv-robustness-manifest.v1"


def _as_float_array(value: Any, *, width: int) -> np.ndarray:
    if value is None:
        return np.empty((0, width), dtype=np.float64)
    arr = np.array(value, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, width), dtype=np.float64)
    return arr.reshape((-1, width))


def _polygon_area_3d(points: np.ndarray) -> float:
    # Fan triangulation is a stable proxy for robustness comparisons. It is not
    # an exact polygon area for non-convex faces.
    if len(points) < 3:
        return 0.0
    origin = points[0]
    area = 0.0
    for idx in range(1, len(points) - 1):
        area += 0.5 * float(
            np.linalg.norm(np.cross(points[idx] - origin, points[idx + 1] - origin))
        )
    return area


def _polygon_area_2d(points: np.ndarray) -> float:
    # Fan triangulation is a stable proxy for robustness comparisons. It is not
    # an exact polygon area for non-convex faces.
    if len(points) < 3:
        return 0.0
    origin = points[0]
    area = 0.0
    for idx in range(1, len(points) - 1):
        a = points[idx] - origin
        b = points[idx + 1] - origin
        area += 0.5 * abs(float(a[0] * b[1] - a[1] * b[0]))
    return area


def _mesh_world_points(prim: Usd.Prim, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.array(
        [
            transform.Transform(
                Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
            )
            for point in points
        ],
        dtype=np.float64,
    )


def _expanded_face_vertex_uvs(
    prim: Usd.Prim,
    face_vertex_indices: np.ndarray,
    point_count: int,
) -> np.ndarray | None:
    st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    if not st or not st.IsDefined() or st.Get() is None:
        return None

    values = _as_float_array(st.Get(), width=2)
    if len(values) == 0:
        return None

    interpolation = st.GetInterpolation() or ""
    indices = None
    if st.IsIndexed():
        raw_indices = st.GetIndices()
        if raw_indices is None:
            return None
        indices = np.array(raw_indices, dtype=np.int64)
        if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(values):
            return None

    face_vertex_count = len(face_vertex_indices)
    if interpolation == "faceVarying":
        if indices is not None and len(indices) == face_vertex_count:
            return values[indices]
        if indices is None and len(values) == face_vertex_count:
            return values
        return None

    if interpolation in {"vertex", "varying"}:
        if indices is not None:
            if len(indices) == face_vertex_count:
                return values[indices]
            if len(indices) == point_count:
                point_uvs = values[indices]
                return point_uvs[face_vertex_indices]
            return None
        if len(values) == point_count:
            return values[face_vertex_indices]

    return None


def _edge_discontinuity_counts(
    face_vertex_indices: np.ndarray,
    face_vertex_counts: np.ndarray,
    face_vertex_uvs: np.ndarray,
) -> tuple[int, int]:
    edges: dict[tuple[int, int], list[tuple[tuple[float, float], tuple[float, float]]]]
    edges = defaultdict(list)
    cursor = 0
    for count in face_vertex_counts:
        face_indices = face_vertex_indices[cursor : cursor + count]
        face_uvs = face_vertex_uvs[cursor : cursor + count]
        for offset in range(count):
            next_offset = (offset + 1) % count
            point_a = int(face_indices[offset])
            point_b = int(face_indices[next_offset])
            uv_a = tuple(float(v) for v in face_uvs[offset])
            uv_b = tuple(float(v) for v in face_uvs[next_offset])
            if point_a <= point_b:
                key = (point_a, point_b)
                ordered_uvs = (uv_a, uv_b)
            else:
                key = (point_b, point_a)
                ordered_uvs = (uv_b, uv_a)
            edges[key].append(ordered_uvs)
        cursor += int(count)

    shared_edges = 0
    discontinuous_edges = 0
    for edge_uvs in edges.values():
        if len(edge_uvs) < 2:
            continue
        shared_edges += 1
        reference = edge_uvs[0]
        if any(
            not (
                np.allclose(reference[0], candidate[0], atol=1e-5)
                and np.allclose(reference[1], candidate[1], atol=1e-5)
            )
            for candidate in edge_uvs[1:]
        ):
            discontinuous_edges += 1
    return shared_edges, discontinuous_edges


def measure_uv_robustness_for_mesh(prim: Usd.Prim) -> dict[str, Any]:
    """Measure stretch and seam proxies for one mesh prim."""
    mesh = UsdGeom.Mesh(prim)
    points = _mesh_world_points(
        prim,
        _as_float_array(mesh.GetPointsAttr().Get(), width=3),
    )
    face_vertex_indices = np.array(
        mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64
    )
    face_vertex_counts = np.array(
        mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64
    )
    face_vertex_uvs = _expanded_face_vertex_uvs(
        prim,
        face_vertex_indices,
        point_count=len(points),
    )

    result: dict[str, Any] = {
        "prim_path": str(prim.GetPath()),
        "face_count": int(len(face_vertex_counts)),
        "measured_face_count": 0,
        "zero_area_uv_faces": 0,
        "zero_area_world_faces": 0,
        "world_area": 0.0,
        "uv_area": 0.0,
        "uv_to_world_area_ratio_median": None,
        "uv_to_world_area_ratio_p95": None,
        "uv_to_world_area_ratio_max": None,
        "stretch_p95_over_median": None,
        "stretch_max_over_median": None,
        "texel_density_cv": None,
        "shared_edges": 0,
        "uv_discontinuous_shared_edges": 0,
        "uv_discontinuous_shared_edge_ratio": None,
        "measurable": face_vertex_uvs is not None,
    }
    if (
        face_vertex_uvs is None
        or len(points) == 0
        or len(face_vertex_indices) == 0
        or len(face_vertex_counts) == 0
    ):
        return result

    ratios: list[float] = []
    densities: list[float] = []
    cursor = 0
    for count in face_vertex_counts:
        indices = face_vertex_indices[cursor : cursor + count]
        face_points = points[indices]
        face_uvs = face_vertex_uvs[cursor : cursor + count]
        world_area = _polygon_area_3d(face_points)
        uv_area = _polygon_area_2d(face_uvs)
        result["world_area"] += world_area
        result["uv_area"] += uv_area
        if world_area <= 1e-12:
            result["zero_area_world_faces"] += 1
        elif uv_area <= 1e-12:
            result["zero_area_uv_faces"] += 1
        else:
            ratios.append(uv_area / world_area)
            densities.append(float(np.sqrt(uv_area / world_area)))
            result["measured_face_count"] += 1
        cursor += int(count)

    if ratios:
        ratio_arr = np.array(ratios, dtype=np.float64)
        median = float(np.median(ratio_arr))
        p95 = float(np.percentile(ratio_arr, 95))
        maximum = float(np.max(ratio_arr))
        result["uv_to_world_area_ratio_median"] = median
        result["uv_to_world_area_ratio_p95"] = p95
        result["uv_to_world_area_ratio_max"] = maximum
        if median > 1e-12:
            result["stretch_p95_over_median"] = p95 / median
            result["stretch_max_over_median"] = maximum / median

    if densities:
        density_arr = np.array(densities, dtype=np.float64)
        mean_density = float(np.mean(density_arr))
        if mean_density > 1e-12:
            result["texel_density_cv"] = float(np.std(density_arr) / mean_density)

    shared_edges, discontinuous_edges = _edge_discontinuity_counts(
        face_vertex_indices,
        face_vertex_counts,
        face_vertex_uvs,
    )
    result["shared_edges"] = shared_edges
    result["uv_discontinuous_shared_edges"] = discontinuous_edges
    if shared_edges:
        result["uv_discontinuous_shared_edge_ratio"] = (
            discontinuous_edges / shared_edges
        )
    return result


def evaluate_usd_uv_robustness(
    usd_path: Path | str,
    *,
    asset_id: str,
    mode_id: str,
) -> dict[str, Any]:
    """Evaluate UV readiness and robustness metrics for a USD file."""
    path = Path(usd_path)
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise FileNotFoundError(f"Failed to open USD stage: {path}")

    uv_report = inspect_uvs_for_stage(stage)
    mesh_metrics = [
        measure_uv_robustness_for_mesh(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh) and not prim.IsInstanceProxy()
    ]
    measurable = [item for item in mesh_metrics if item["measurable"]]
    zero_area_uv_faces = sum(int(item["zero_area_uv_faces"]) for item in measurable)
    discontinuous_edges = sum(
        int(item["uv_discontinuous_shared_edges"]) for item in measurable
    )
    shared_edges = sum(int(item["shared_edges"]) for item in measurable)
    stretch_values = [
        float(item["stretch_p95_over_median"])
        for item in measurable
        if item["stretch_p95_over_median"] is not None
    ]
    density_values = [
        float(item["texel_density_cv"])
        for item in measurable
        if item["texel_density_cv"] is not None
    ]

    return {
        "schema_version": UV_ROBUSTNESS_SCHEMA_VERSION,
        "asset_id": asset_id,
        "mode_id": mode_id,
        "usd_path": str(path),
        "uv_report": uv_report,
        "robustness_summary": {
            "measurable_meshes": len(measurable),
            "zero_area_uv_faces": zero_area_uv_faces,
            "shared_edges": shared_edges,
            "uv_discontinuous_shared_edges": discontinuous_edges,
            "uv_discontinuous_shared_edge_ratio": (
                discontinuous_edges / shared_edges if shared_edges else None
            ),
            "stretch_p95_over_median_max": (
                max(stretch_values) if stretch_values else None
            ),
            "texel_density_cv_max": max(density_values) if density_values else None,
        },
        "meshes": mesh_metrics,
    }


def load_uv_robustness_manifest(path: Path | str) -> dict[str, Any]:
    """Load and validate an issue #364 UV robustness manifest."""
    manifest_path = Path(path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    schema = manifest.get("schema_version")
    if schema != UV_ROBUSTNESS_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Invalid UV robustness manifest schema "
            f"{schema!r}; expected {UV_ROBUSTNESS_MANIFEST_SCHEMA_VERSION!r}"
        )
    return manifest


def evaluate_uv_robustness_manifest(
    manifest_path: Path | str,
    output_dir: Path | str,
    *,
    repo_root: Path | str | None = None,
    asset_ids: set[str] | None = None,
    mode_ids: set[str] | None = None,
    fail_on_missing_required: bool = True,
) -> dict[str, Any]:
    """Run the configured UV mode matrix and write a JSON evidence report."""
    manifest = load_uv_robustness_manifest(manifest_path)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": UV_ROBUSTNESS_SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "output_dir": str(out_dir),
        "missing_required_assets": [],
        "assets": [],
    }

    modes = [
        mode
        for mode in manifest.get("modes", [])
        if mode_ids is None or mode.get("id") in mode_ids
    ]
    task = PrepareUVsTask()

    for asset in manifest.get("assets", []):
        asset_id = str(asset.get("id", "")).strip()
        if not asset_id or (asset_ids is not None and asset_id not in asset_ids):
            continue

        source_usd = str(asset.get("source_usd") or "").strip()
        source_path = root / source_usd if source_usd else None
        asset_record: dict[str, Any] = {
            "id": asset_id,
            "role": asset.get("role"),
            "uv_condition": asset.get("uv_condition"),
            "source_usd": str(source_path) if source_path is not None else "",
            "required": bool(asset.get("required", False)),
            "modes": [],
        }
        if source_path is None or not source_path.is_file():
            if source_path is None:
                asset_record["error"] = "source_usd is required"
            elif source_path.exists():
                asset_record["error"] = "source_usd must point to a USD file"
            else:
                asset_record["error"] = "source_usd file is missing"
            if asset_record["required"]:
                asset_record["status"] = "missing_required"
                report["missing_required_assets"].append(asset_id)
            else:
                asset_record["status"] = "missing"
            report["assets"].append(asset_record)
            continue

        asset_record["status"] = "evaluated"
        asset_record["source_report"] = evaluate_usd_uv_robustness(
            source_path,
            asset_id=asset_id,
            mode_id="source",
        )

        for mode in modes:
            mode_id = str(mode.get("id", "")).strip()
            if not mode_id:
                continue
            work_dir = out_dir / asset_id / mode_id
            texture_config = dict(mode.get("texture_config", {}))
            if (
                asset.get("target_prim_paths")
                and "uv_target_prim_paths" not in texture_config
            ):
                texture_config["uv_target_prim_paths"] = list(
                    asset["target_prim_paths"]
                )
            if asset.get("target_prim_paths") and "uv_scope" not in texture_config:
                texture_config["uv_scope"] = "target_prims"
            context = {
                "usd_path": str(source_path),
                "working_dir": str(work_dir),
                "texture_config": texture_config,
                "material_textures": asset.get("material_textures", {}),
            }

            mode_record: dict[str, Any] = {
                "id": mode_id,
                "description": mode.get("description"),
                "texture_config": texture_config,
            }
            try:
                result_context = task.run(context)
                prepared_path = Path(result_context["usd_path"])
                mode_record["status"] = "completed"
                mode_record["prepared_usd"] = str(prepared_path)
                mode_record["uv_preparation"] = result_context.get("uv_preparation")
                mode_record["report"] = evaluate_usd_uv_robustness(
                    prepared_path,
                    asset_id=asset_id,
                    mode_id=mode_id,
                )
            except (
                FileNotFoundError,
                OSError,
                RuntimeError,
                UVPreparationError,
                ValueError,
            ) as err:
                mode_record["status"] = "failed"
                mode_record["error"] = str(err)
            asset_record["modes"].append(mode_record)

        report["assets"].append(asset_record)

    report_path = out_dir / "uv_robustness_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    if report["missing_required_assets"] and fail_on_missing_required:
        missing = ", ".join(report["missing_required_assets"])
        raise FileNotFoundError(
            f"Required UV robustness assets are missing: {missing}. "
            f"Report written to {report_path}"
        )
    return report
