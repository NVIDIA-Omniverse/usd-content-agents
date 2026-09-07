# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorded OpenVDB 10 to source-locked OpenVDB 13 migration sentinel."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh

from geometry_repair.workers.sdf_reconstruction import (
    SdfReconstructionControls,
    reconstruct_sdf_mesh,
)

_BASELINE_PATH = Path(__file__).parent / "fixtures/openvdb10_to_13_parity_baseline.json"
_V10_MESH_PATH = Path(__file__).parent / "fixtures/openvdb10_grid128_canonical_mesh.npz"
_OPENVDB_RUNTIME_SPEC = importlib.util.find_spec("openvdb_runtime")
if _OPENVDB_RUNTIME_SPEC is None or _OPENVDB_RUNTIME_SPEC.origin is None:
    pytest.skip(
        "openvdb_runtime is required for the OpenVDB parity sentinel",
        allow_module_level=True,
    )
_OPENVDB_PACKAGE_ROOT = Path(_OPENVDB_RUNTIME_SPEC.origin).resolve().parent


def _baseline() -> dict[str, Any]:
    payload = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _promoted_x86_64_release() -> dict[str, Any]:
    packaged_lock = _OPENVDB_PACKAGE_ROOT / "_release_lock.json"
    source_lock = _OPENVDB_PACKAGE_ROOT.parent / "native" / "release-lock.json"
    lock_path = packaged_lock if packaged_lock.is_file() else source_lock
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "world-understanding.sdf-native-release-lock.v1"
    selected = payload["platforms"].get("x86_64")
    assert isinstance(selected, dict), "OpenVDB parity fixture has no x86_64 release record"
    assert selected.get("status") == "promoted"
    return selected


def _array_record(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.name,
        "shape": [int(value) for value in contiguous.shape],
        "sha256": hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest(),
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _openvdb10_mesh() -> tuple[np.ndarray, np.ndarray]:
    with np.load(_V10_MESH_PATH, allow_pickle=False) as archive:
        assert archive.files == ["vertices", "triangles"]
        vertices = np.array(archive["vertices"], dtype=np.float32, order="C", copy=True)
        triangles = np.array(archive["triangles"], dtype=np.int32, order="C", copy=True)
    return vertices, triangles


def _fixture_meshes() -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    ground_truth = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    keep = np.ones(len(ground_truth.faces), dtype=bool)
    keep[0] = False
    source = ground_truth.copy()
    source.update_faces(keep)
    source.remove_unreferenced_vertices()
    return source, ground_truth


def _bounded_surface_points(mesh: trimesh.Trimesh, limit: int = 4096) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape((-1, 3))
    triangles = np.asarray(mesh.triangles, dtype=np.float64).reshape((-1, 3, 3))
    centroids = triangles.mean(axis=1)
    points = np.vstack((vertices, centroids))
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, num=limit, dtype=np.int64)
    return points[indices]


def _directed_surface_distances(surface: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    _closest, distances, _triangle_ids = trimesh.proximity.closest_point_naive(surface, points)
    return np.asarray(distances, dtype=np.float64)


def _semantic_metrics(
    vertices: np.ndarray,
    triangles: np.ndarray,
    ground_truth: trimesh.Trimesh,
) -> dict[str, Any]:
    output = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    output_to_truth = _directed_surface_distances(ground_truth, _bounded_surface_points(output))
    truth_to_output = _directed_surface_distances(output, _bounded_surface_points(ground_truth))
    return {
        "vertex_count": len(vertices),
        "triangle_count": len(triangles),
        "watertight": bool(output.is_watertight),
        "winding_consistent": bool(output.is_winding_consistent),
        "component_count": len(output.split(only_watertight=False)),
        "bounds_min": [float(value) for value in output.bounds[0]],
        "bounds_max": [float(value) for value in output.bounds[1]],
        "surface_area": float(output.area),
        "signed_volume": float(output.volume),
        "output_to_ground_truth_surface_distance": {
            "p99": float(np.quantile(output_to_truth, 0.99)),
            "max": float(np.max(output_to_truth)),
        },
        "ground_truth_to_output_surface_distance": {
            "p99": float(np.quantile(truth_to_output, 0.99)),
            "max": float(np.max(truth_to_output)),
        },
    }


def _assert_relative(actual: float, expected: float, tolerance: float) -> None:
    assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=0.0)


def _assert_semantic_parity(
    actual: dict[str, Any],
    expected: dict[str, Any],
    tolerances: dict[str, Any],
) -> None:
    assert tolerances["watertight_required"] is True
    assert tolerances["winding_consistent_required"] is True
    assert tolerances["component_count_must_match"] is True
    assert tolerances["positive_volume_required"] is True
    assert actual["watertight"] is expected["watertight"] is True
    assert actual["winding_consistent"] is expected["winding_consistent"] is True
    assert actual["component_count"] == expected["component_count"]
    assert actual["signed_volume"] > 0.0 and expected["signed_volume"] > 0.0
    _assert_relative(
        actual["surface_area"], expected["surface_area"], tolerances["relative_surface_area"]
    )
    _assert_relative(
        actual["signed_volume"], expected["signed_volume"], tolerances["relative_volume"]
    )
    assert np.allclose(
        actual["bounds_min"],
        expected["bounds_min"],
        rtol=0.0,
        atol=tolerances["bounds_absolute_m"],
    )
    assert np.allclose(
        actual["bounds_max"],
        expected["bounds_max"],
        rtol=0.0,
        atol=tolerances["bounds_absolute_m"],
    )
    for direction in (
        "output_to_ground_truth_surface_distance",
        "ground_truth_to_output_surface_distance",
    ):
        for statistic in ("p99", "max"):
            assert (
                abs(actual[direction][statistic] - expected[direction][statistic])
                <= (tolerances["surface_distance_delta_m"])
            )


def test_recorded_openvdb_major_parity_observation_is_self_consistent() -> None:
    baseline = _baseline()
    recorded_runtime = baseline["openvdb13_observation"]["runtime"]
    promoted_release = _promoted_x86_64_release()
    assert baseline["schema_version"] == "geometry-repair.openvdb-major-parity.v1"
    assert baseline["status"] == "passed"
    comparison = baseline["recorded_comparison"]
    assert comparison["verdict"] == "pass"
    assert comparison["bidirectional_surface_distance_m"] == 0.0
    parity_gates = {
        key: value
        for key, value in comparison.items()
        if key not in {"bidirectional_surface_distance_m", "verdict"}
    }
    assert parity_gates, "recorded_comparison must declare boolean parity gates"
    assert all(value is True for value in parity_gates.values()), parity_gates
    assert baseline["openvdb10_observation"]["adapter"]["reported_library_version"] == [
        10,
        0,
        1,
    ]
    assert recorded_runtime["library_version"] == [13, 0, 0]
    assert promoted_release["source_lock_sha256"] == recorded_runtime["source_lock_sha256"]
    assert promoted_release["wheel"]["sha256"] == recorded_runtime["wheel_sha256"]
    extension_members = {
        name: digest
        for name, digest in promoted_release["wheel_native_members"].items()
        if name.startswith("openvdb/lib/openvdb.") and name.endswith(".so")
    }
    assert len(extension_members) == 1
    assert next(iter(extension_members.values())) == recorded_runtime["module_sha256"]
    assert baseline["openvdb13_observation"]["canonical_arrays_equal_to_openvdb10"] is True
    assert baseline["direct_surface_distance_method"] == {
        "closest_point_implementation": "trimesh.proximity.closest_point_naive",
        "gate_statistic": "maximum",
        "max_query_points_per_direction": 4096,
        "query_points": "all_vertices_and_triangle_centroids_then_even_index_subsample",
    }
    assert (
        baseline["openvdb10_observation"]["output_arrays"]
        == baseline["openvdb13_observation"]["output_arrays"]
    )
    assert (
        baseline["openvdb10_observation"]["semantic_metrics"]
        == baseline["openvdb13_observation"]["semantic_metrics"]
    )
    capture = baseline["openvdb10_observation"]["capture"]
    artifact = capture["canonical_mesh_artifact"]
    assert artifact["path"] == _V10_MESH_PATH.name
    assert artifact["sha256"] == _file_sha256(_V10_MESH_PATH)
    v10_vertices, v10_triangles = _openvdb10_mesh()
    assert (
        _array_record(v10_vertices)
        == baseline["openvdb10_observation"]["output_arrays"]["vertices"]
    )
    assert (
        _array_record(v10_triangles)
        == baseline["openvdb10_observation"]["output_arrays"]["triangles"]
    )

    source, ground_truth = _fixture_meshes()
    _assert_semantic_parity(
        _semantic_metrics(v10_vertices, v10_triangles, ground_truth),
        baseline["openvdb10_observation"]["semantic_metrics"],
        baseline["approved_tolerances"],
    )
    controls = baseline["controls"]
    usable_cells = controls["max_grid_dimension"] - 2 * math.ceil(controls["half_width"]) - 2
    canonical_source_vertices = np.asarray(source.vertices, dtype=np.float32).astype(np.float64)
    expected_voxel_size = float(np.linalg.norm(np.ptp(canonical_source_vertices, axis=0))) / (
        usable_cells
    )
    assert controls["voxel_size"] == pytest.approx(expected_voxel_size, rel=0.0, abs=1.0e-15)
    assert (
        _array_record(np.asarray(source.vertices, dtype=np.float32))
        == baseline["fixture"]["source_arrays"]["vertices"]
    )
    assert (
        _array_record(np.asarray(source.faces, dtype=np.int32))
        == baseline["fixture"]["source_arrays"]["triangles"]
    )
    assert (
        _array_record(np.asarray(ground_truth.vertices, dtype=np.float32))
        == baseline["fixture"]["ground_truth_arrays"]["vertices"]
    )
    assert (
        _array_record(np.asarray(ground_truth.faces, dtype=np.int32))
        == baseline["fixture"]["ground_truth_arrays"]["triangles"]
    )


def _assert_live_openvdb13_parity() -> None:
    baseline = _baseline()
    source, ground_truth = _fixture_meshes()
    source_vertices = np.asarray(source.vertices, dtype=np.float32)
    source_triangles = np.asarray(source.faces, dtype=np.int32)
    controls = SdfReconstructionControls.model_validate(baseline["controls"])
    result = reconstruct_sdf_mesh(source_vertices, source_triangles, controls)
    v10_vertices, v10_triangles = _openvdb10_mesh()
    runtime = result.evidence["selected_backend_identity"]["provenance"]
    recorded_runtime = baseline["openvdb13_observation"]["runtime"]
    for field in (
        "distribution_version",
        "library_version",
        "module_sha256",
        "source_commit",
        "source_lock_sha256",
    ):
        assert runtime[field] == recorded_runtime[field]

    assert (
        result.evidence["selected_backend_identity"]["provenance"]["source_lock_sha256"]
        == recorded_runtime["source_lock_sha256"]
    )
    assert result.evidence["algorithm"]["route"] == "signed_topology_closing"
    assert result.evidence["algorithm"]["fallback_used"] is False
    assert len(result.vertices) == len(v10_vertices)
    assert len(result.triangles) == len(v10_triangles)

    v10_surface = trimesh.Trimesh(vertices=v10_vertices, faces=v10_triangles, process=False)
    v13_surface = trimesh.Trimesh(
        vertices=result.vertices,
        faces=result.triangles,
        process=False,
    )
    distance_method = baseline["direct_surface_distance_method"]
    point_limit = distance_method["max_query_points_per_direction"]
    v13_to_v10 = _directed_surface_distances(
        v10_surface,
        _bounded_surface_points(v13_surface, limit=point_limit),
    )
    v10_to_v13 = _directed_surface_distances(
        v13_surface,
        _bounded_surface_points(v10_surface, limit=point_limit),
    )
    direct_tolerance = baseline["approved_tolerances"]["direct_bidirectional_surface_distance_m"]
    assert float(np.max(v13_to_v10)) <= direct_tolerance
    assert float(np.max(v10_to_v13)) <= direct_tolerance

    actual_metrics = _semantic_metrics(result.vertices, result.triangles, ground_truth)
    _assert_semantic_parity(
        actual_metrics,
        baseline["openvdb13_observation"]["semantic_metrics"],
        baseline["approved_tolerances"],
    )

    assert (
        _array_record(result.vertices)
        == baseline["openvdb13_observation"]["output_arrays"]["vertices"]
    )
    assert (
        _array_record(result.triangles)
        == baseline["openvdb13_observation"]["output_arrays"]["triangles"]
    )


def test_openvdb13_matches_recorded_openvdb10_semantics() -> None:
    try:
        importlib.metadata.version("openvdb")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("the live OpenVDB 13 parity gate requires the native wheel")

    environment = os.environ.copy()
    environment.pop("PYTHONOPTIMIZE", None)
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--live-openvdb13-parity"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, (
        "isolated live OpenVDB 13 parity gate failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--live-openvdb13-parity"]:
        raise SystemExit("usage: test_openvdb13_parity.py --live-openvdb13-parity")
    _assert_live_openvdb13_parity()
