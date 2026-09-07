# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporary, non-final PhysX validation for generated collision geometry."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    open_confined_directory,
    open_confined_regular_file,
    validated_artifact_relative_key,
    write_bytes_to_confined,
)

from .artifacts import atomic_write_json
from .mesh_io import load_meshes


def _author_temporary_proxy(
    collision_path: Path,
    output_path: Path,
    *,
    orientation_xyzw: tuple[float, float, float, float],
) -> Path:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    meshes, _ = load_meshes(collision_path, include_guide_purpose=True)
    if not meshes:
        raise RuntimeError("Temporary collision proxy found no mesh bodies")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/GeometryRepairRuntimeProxy")
    stage.SetDefaultPrim(root.GetPrim())
    all_vertices = np.concatenate([mesh.world_vertices_m for mesh in meshes], axis=0)
    center = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) * 0.5
    x, y, z, w = orientation_xyzw
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(1.0)
    for index, source in enumerate(meshes):
        mesh = UsdGeom.Mesh.Define(stage, f"/GeometryRepairRuntimeProxy/Collider_{index:04d}")
        rotated_vertices = (source.world_vertices_m - center) @ rotation.T + center
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(rotated_vertices.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(source.triangles), 3, dtype=np.int32))
        )
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(source.triangles.astype(np.int32).reshape(-1))
        )
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        if len(rotated_vertices):
            minimum = rotated_vertices.min(axis=0)
            maximum = rotated_vertices.max(axis=0)
            mesh.CreateExtentAttr(
                Vt.Vec3fArray(
                    [
                        Gf.Vec3f(*[float(value) for value in minimum]),
                        Gf.Vec3f(*[float(value) for value in maximum]),
                    ]
                )
            )
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    layer_data = dict(stage.GetRootLayer().customLayerData or {})
    layer_data["geometryRepairTemporaryProxy"] = True
    layer_data["claimScope"] = "temporary_geometry_loadability_only"
    stage.GetRootLayer().customLayerData = layer_data
    stage.GetRootLayer().Save()
    return output_path


_RIGID_ORIENTATIONS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("identity", (0.0, 0.0, 0.0, 1.0)),
    ("quarter_turn_x", (2**-0.5, 0.0, 0.0, 2**-0.5)),
    ("quarter_turn_y", (0.0, 2**-0.5, 0.0, 2**-0.5)),
    ("diagonal", (0.27059805, 0.27059805, 0.0, 0.92387953)),
)
_MAX_RUNTIME_REPORT_BYTES = 8 * 1024 * 1024
_MAX_RUNTIME_TRAJECTORY_BYTES = 64 * 1024 * 1024


def _runtime_artifact_key(
    runtime_root: Path,
    value: object,
    *,
    label: str,
    expected_key: str,
) -> tuple[Path, str]:
    """Bind a runtime-reported path to one fixed artifact below its output root."""

    root = runtime_root.expanduser().resolve(strict=True)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"usd-cli physics simulate omitted its {label}")
    candidate = Path(value).expanduser()
    absolute = Path(os.path.abspath(candidate if candidate.is_absolute() else root / candidate))
    try:
        relative = absolute.relative_to(root).as_posix()
        key = validated_artifact_relative_key(relative)
    except ValueError as exc:
        raise RuntimeError(
            f"usd-cli physics simulate returned {label} outside its output root"
        ) from exc
    if key != expected_key:
        raise RuntimeError(f"usd-cli physics simulate returned unexpected {label}: {key!r}")
    return root, key


def _read_runtime_artifact_text(
    runtime_root: Path,
    value: object,
    *,
    label: str,
    expected_key: str,
    max_bytes: int,
) -> tuple[Path, str]:
    root, key = _runtime_artifact_key(
        runtime_root,
        value,
        label=label,
        expected_key=expected_key,
    )
    try:
        with open_confined_directory(root) as root_descriptor:
            with open_confined_regular_file(root_descriptor, key) as (stream, metadata):
                if metadata.st_size > max_bytes:
                    raise RuntimeError(f"usd-cli physics simulate returned oversized {label}")
                data = stream.read(max_bytes + 1)
    except (ArtifactPathError, OSError, ValueError) as exc:
        raise RuntimeError(f"usd-cli physics simulate returned unsafe {label}: {exc}") from exc
    if len(data) > max_bytes:
        raise RuntimeError(f"usd-cli physics simulate returned oversized {label}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"usd-cli physics simulate returned non-UTF-8 {label}") from exc
    return root / key, text


def _quaternion_rotation_matrix(quaternion_xyzw: list[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = max(float(np.linalg.norm([x, y, z, w])), 1e-30)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _exact_proxy_ground_penetration(
    proxy_path: Path,
    runtime_report_path: Path,
    *,
    runtime_root: Path,
) -> dict[str, float | int]:
    """Measure ground penetration from actual collider vertices at every pose."""

    meshes, _ = load_meshes(proxy_path, include_guide_purpose=True)
    vertices = np.concatenate([mesh.world_vertices_m for mesh in meshes], axis=0)
    _report_path, report_text = _read_runtime_artifact_text(
        runtime_root,
        str(runtime_report_path),
        label="runtime report",
        expected_key="runtime_validation_report.json",
        max_bytes=_MAX_RUNTIME_REPORT_BYTES,
    )
    report = json.loads(report_text)
    _trajectory_path, trajectory_text = _read_runtime_artifact_text(
        runtime_root,
        report.get("trajectory_jsonl"),
        label="trajectory",
        expected_key="trajectory.jsonl",
        max_bytes=_MAX_RUNTIME_TRAJECTORY_BYTES,
    )
    poses: list[list[float]] = []
    for line in trajectory_text.splitlines():
        if line.strip():
            payload = json.loads(line)
            poses.append([float(value) for value in payload["pose"]])
    if not len(vertices) or not poses:
        raise RuntimeError("exact collision penetration check has no vertices or poses")

    maximum_penetration_m = 0.0
    for pose in poses:
        if len(pose) != 7 or not np.all(np.isfinite(pose)):
            raise RuntimeError("exact collision penetration check found an invalid pose")
        rotation = _quaternion_rotation_matrix(pose[3:7])
        world_vertices = vertices @ rotation.T + np.asarray(pose[:3], dtype=np.float64)
        maximum_penetration_m = max(
            maximum_penetration_m,
            max(0.0, -float(world_vertices[:, 2].min())),
        )
    return {
        "exact_maximum_ground_penetration_m": maximum_penetration_m,
        "exact_collision_vertex_count": int(len(vertices)),
        "evaluated_pose_count": len(poses),
    }


def _fake_trajectory(
    *,
    rest_position: list[float],
    world_up: list[float],
    duration_s: float,
    sample_fps: int,
    drop_height_m: float,
) -> list[tuple[float, list[float], list[float]]]:
    """Produce deterministic test-only motion without claiming solver evidence."""

    sample_count = max(2, int(round(duration_s * sample_fps)) + 1)
    up_index = max(range(3), key=lambda index: abs(float(world_up[index])))
    trajectory: list[tuple[float, list[float], list[float]]] = []
    for index in range(sample_count):
        time_s = duration_s * index / (sample_count - 1)
        alpha = index / (sample_count - 1)
        height_offset = max(drop_height_m * (1.0 - alpha) ** 2, 0.0)
        pose = [float(value) for value in rest_position] + [0.0, 0.0, 0.0, 1.0]
        pose[up_index] += height_offset
        velocity = [0.0] * 6
        velocity[up_index] = -2.0 * drop_height_m * (1.0 - alpha) / max(duration_s, 1e-6)
        trajectory.append((float(time_s), pose, velocity))
    return trajectory


def _author_drop_settle_scenario(
    physics_usd: Path,
    output_scene: Path,
    *,
    drop_height_m: float,
) -> dict[str, Any]:
    """Author Geometry Repair's explicit one-body gravity scenario."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    stage = Usd.Stage.Open(str(physics_usd))
    if stage is None:
        raise RuntimeError(f"Failed to open temporary collision proxy: {physics_usd}")
    body = stage.GetDefaultPrim()
    if not body or not body.IsValid():
        raise RuntimeError("Temporary collision proxy has no default body prim")
    body_path = body.GetPath().pathString
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    bounds = cache.ComputeWorldBound(body).ComputeAlignedRange()
    minimum = [float(value) for value in bounds.GetMin()]
    maximum = [float(value) for value in bounds.GetMax()]
    if not all(math.isfinite(value) for value in [*minimum, *maximum]):
        raise RuntimeError("Temporary collision proxy has invalid bounds")
    translation = [0.0, 0.0, float(drop_height_m) - minimum[2]]
    UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(*translation))

    physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    half_extent = max(maximum[index] - minimum[index] for index in range(3)) * 4.0 + 1.0
    ground = UsdGeom.Mesh.Define(stage, "/GroundPlane")
    ground.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(-half_extent, -half_extent, 0.0),
                Gf.Vec3f(half_extent, -half_extent, 0.0),
                Gf.Vec3f(half_extent, half_extent, 0.0),
                Gf.Vec3f(-half_extent, half_extent, 0.0),
            ]
        )
    )
    ground.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    ground.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    ground.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    output_scene.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output_scene))
    rest_position = [0.0, 0.0, -minimum[2]]
    return {
        "body_prim_path": body_path,
        "body_pattern": body_path,
        "rest_position": rest_position,
        "world_up": [0.0, 0.0, 1.0],
        "drop_height_m_resolved": float(drop_height_m),
    }


def _fake_runtime_validation(
    physics_usd: Path,
    output_dir: Path,
    *,
    duration_s: float,
    sample_fps: int,
    drop_height_m: float,
) -> dict[str, Any]:
    """Exercise usd-cli's scene/recording primitives with a synthetic test engine."""

    from usd_core import physics_runtime

    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / "drop_settle_scene.usda"
    scene_info = _author_drop_settle_scenario(
        physics_usd,
        scene_path,
        drop_height_m=drop_height_m,
    )
    resolved_drop_height = float(scene_info["drop_height_m_resolved"])
    trajectory = _fake_trajectory(
        rest_position=[float(value) for value in scene_info["rest_position"]],
        world_up=[float(value) for value in scene_info["world_up"]],
        duration_s=duration_s,
        sample_fps=sample_fps,
        drop_height_m=resolved_drop_height,
    )
    trajectory_path = Path(
        physics_runtime.author_trajectory_jsonl(
            trajectory,
            str(output_dir / "trajectory.jsonl"),
        )
    )
    recording_path = Path(
        physics_runtime.author_trajectory_usda(
            str(scene_path),
            trajectory,
            str(scene_info["body_prim_path"]),
            str(output_dir / "recording.usda"),
            fps=sample_fps,
        )
    )
    report_path = output_dir / "runtime_validation_report.json"
    report: dict[str, Any] = {
        "engine": "fake",
        "executor": "synthetic-test-only",
        "ok": True,
        "n_bodies": 1,
        "scene_usd": str(scene_path),
        "recording_usda": str(recording_path),
        "trajectory_jsonl": str(trajectory_path),
        "drop_height_m": resolved_drop_height,
        "metrics": physics_runtime.trajectory_metrics(
            trajectory,
            scene_info["rest_position"],
            scene_info["world_up"],
        ),
        "failures": [],
        "warnings": ["fake runtime evidence is synthetic and test-only"],
        "scene_info": scene_info,
        "report_path": str(report_path),
    }
    atomic_write_json(report_path, report)
    return report


def _usd_cli_runtime_validation(
    physics_usd: Path,
    output_dir: Path,
    *,
    engine: Literal["fake", "ovphysx"],
    duration_s: float,
    dt: float,
    sample_fps: int,
    drop_height_m: float,
) -> dict[str, Any]:
    """Invoke workflow-neutral usd-cli runtime primitives for one proxy scene."""

    if engine == "fake":
        return _fake_runtime_validation(
            physics_usd,
            output_dir,
            duration_s=duration_s,
            sample_fps=sample_fps,
            drop_height_m=drop_height_m,
        )

    from usd_core.session import Session

    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / "drop_settle_scene.usda"
    scene_info = _author_drop_settle_scenario(
        physics_usd,
        scene_path,
        drop_height_m=drop_height_m,
    )
    response = Session.open(scene_path).physics_simulate(
        scene=str(scene_path),
        body=str(scene_info["body_prim_path"]),
        rest_position=[float(value) for value in scene_info["rest_position"]],
        world_up=[float(value) for value in scene_info["world_up"]],
        engine="ovphysx",
        duration=duration_s,
        dt=dt,
        fps=sample_fps,
        output=str(output_dir),
        relax_ovphysx_address_space_limit=True,
    )
    if not response.data:
        detail = "; ".join(issue.message for issue in response.issues)
        raise RuntimeError(detail or "usd-cli physics simulate returned no evidence")
    report = dict(response.data)
    report["scene_info"] = scene_info
    report["drop_height_m"] = float(drop_height_m)
    report["runtime_report"] = report.get("report_path")
    return report


def _simulate_explicit_scene(
    scene: Path,
    output_dir: Path,
    *,
    body_path: str,
    rest_position: list[float],
    world_up: list[float],
    duration_s: float,
    dt: float,
    sample_fps: int,
) -> tuple[dict[str, Any], list[tuple[float, list[float], list[float]]]]:
    """Invoke neutral usd-cli simulation and load its raw trajectory artifact."""

    from usd_core.session import Session

    response = Session.open(scene).physics_simulate(
        scene=str(scene),
        body=body_path,
        rest_position=rest_position,
        world_up=world_up,
        engine="ovphysx",
        duration=duration_s,
        dt=dt,
        fps=sample_fps,
        output=str(output_dir),
        relax_ovphysx_address_space_limit=True,
    )
    if not response.data:
        detail = "; ".join(issue.message for issue in response.issues)
        raise RuntimeError(detail or "usd-cli physics simulate returned no evidence")
    report = dict(response.data)
    trajectory_path, trajectory_text = _read_runtime_artifact_text(
        output_dir,
        report.get("trajectory_jsonl"),
        label="trajectory",
        expected_key="trajectory.jsonl",
        max_bytes=_MAX_RUNTIME_TRAJECTORY_BYTES,
    )
    report["trajectory_jsonl"] = str(trajectory_path)
    trajectory: list[tuple[float, list[float], list[float]]] = []
    for line in trajectory_text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        trajectory.append(
            (
                float(row["t"]),
                [float(value) for value in row["pose"]],
                [float(value) for value in row["vel"]],
            )
        )
    return report, trajectory


def _collision_runtime_policy(
    report: dict[str, Any],
    *,
    runtime_root: Path,
) -> dict[str, Any]:
    """Apply Geometry Repair's one-body gravity-response acceptance policy."""

    failures = [str(item) for item in report.get("failures") or []]
    warnings = [str(item) for item in report.get("warnings") or []]
    loaded_body_count = int(report.get("n_bodies") or 0)
    if loaded_body_count != 1:
        failures.append(
            f"Collision proxy runtime loaded {loaded_body_count} bodies; expected exactly 1."
        )

    trajectory_path, trajectory_text = _read_runtime_artifact_text(
        runtime_root,
        report.get("trajectory_jsonl"),
        label="trajectory",
        expected_key="trajectory.jsonl",
        max_bytes=_MAX_RUNTIME_TRAJECTORY_BYTES,
    )
    report["trajectory_jsonl"] = str(trajectory_path)
    poses: list[list[float]] = []
    for line in trajectory_text.splitlines():
        if line.strip():
            payload = json.loads(line)
            poses.append([float(value) for value in payload["pose"]])
    if not poses:
        failures.append("Collision proxy runtime produced no trajectory poses.")
    else:
        world_up = np.asarray(
            (report.get("scene_info") or {}).get("world_up") or [0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(world_up))
        if not math.isfinite(norm) or norm <= 0.0:
            failures.append("Collision proxy runtime reported an invalid world-up axis.")
        else:
            world_up /= norm
            downward_displacement = float(
                np.dot(
                    np.asarray(poses[0][:3], dtype=np.float64)
                    - np.asarray(poses[-1][:3], dtype=np.float64),
                    world_up,
                )
            )
            minimum_response = min(
                max(float(report.get("drop_height_m") or 0.0) * 0.1, 1e-6),
                0.001,
            )
            if downward_displacement < minimum_response:
                failures.append(
                    "Collision proxy did not exhibit the required gravity response: "
                    f"downward displacement {downward_displacement:.6f} m, "
                    f"minimum {minimum_response:.6f} m."
                )

    report["failures"] = failures
    report["warnings"] = warnings
    report["ok"] = not failures
    report["acceptance"] = {
        "expected_body_count": 1,
        "max_ground_penetration_m": None,
        "require_gravity_response": True,
    }
    report_root, report_key = _runtime_artifact_key(
        runtime_root,
        report.get("report_path"),
        label="report path",
        expected_key="runtime_validation_report.json",
    )
    report_path = report_root / report_key
    report["report_path"] = str(report_path)
    report["runtime_report"] = str(report_path)
    report_bytes = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )
    with open_confined_directory(report_root) as root_descriptor:
        write_bytes_to_confined(
            root_descriptor,
            report_key,
            report_bytes,
            overwrite=True,
        )
    return report


def validate_collision_runtime(
    collision_path: str | Path,
    output_dir: str | Path,
    *,
    engine: Literal["fake", "ovphysx"],
) -> dict:
    """Cook and step collision geometry through workflow-neutral usd-cli runtime."""

    output = Path(output_dir).expanduser().resolve()
    scenarios = []
    failures: list[str] = []
    warnings: list[str] = []
    for scenario_id, orientation in _RIGID_ORIENTATIONS:
        scenario_dir = output / scenario_id
        runtime_root = scenario_dir / "usd_cli_runtime"
        proxy = _author_temporary_proxy(
            Path(collision_path).expanduser().resolve(),
            scenario_dir / "temporary_collision_runtime_proxy.usda",
            orientation_xyzw=orientation,
        )
        report = _collision_runtime_policy(
            _usd_cli_runtime_validation(
                proxy,
                runtime_root,
                engine=engine,
                duration_s=1.5,
                dt=1.0 / 240.0,
                sample_fps=30,
                drop_height_m=0.02,
            ),
            runtime_root=runtime_root,
        )
        scenario_failures = [str(item) for item in report.get("failures") or []]
        scenario_warnings = [str(item) for item in report.get("warnings") or []]
        metrics: dict[str, float | int] = {}
        runtime_report_path = report.get("runtime_report")
        try:
            if not runtime_report_path:
                raise RuntimeError("shared runtime response omitted runtime_report")
            metrics = _exact_proxy_ground_penetration(
                proxy,
                Path(str(runtime_report_path)),
                runtime_root=runtime_root,
            )
            penetration = float(metrics["exact_maximum_ground_penetration_m"])
            if penetration > 0.005:
                scenario_failures.append(
                    "Exact collision vertices penetrated the ground by "
                    f"{penetration:.6f} m (limit 0.005000 m)."
                )
        except Exception as exc:
            scenario_failures.append(
                f"Exact collision penetration evidence was unavailable: {type(exc).__name__}: {exc}"
            )
        failures.extend(f"{scenario_id}: {item}" for item in scenario_failures)
        warnings.extend(f"{scenario_id}: {item}" for item in scenario_warnings)
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "scenario_kind": "rigid_drop_settle",
                "status": "fail" if scenario_failures else "pass",
                "orientation_xyzw": list(orientation),
                "runtime_report_path": runtime_report_path,
                "temporary_proxy_path": str(proxy),
                "metrics": metrics,
                "failures": scenario_failures,
                "warnings": scenario_warnings,
            }
        )
    summary_path = atomic_write_json(
        output / "runtime_scenarios.json",
        {
            "schema_version": "geometry-repair.runtime-scenarios.v1",
            "engine": engine,
            "status": "fail" if failures else "pass",
            "scenarios": scenarios,
            "failures": failures,
            "warnings": warnings,
        },
    )
    return {
        "status": "fail" if failures else "pass",
        "engine": engine,
        "temporary_proxy_path": scenarios[0]["temporary_proxy_path"],
        "runtime_report_path": str(summary_path),
        "runtime_scenarios": scenarios,
        "failures": failures,
        "warnings": warnings,
    }


def _select_static_contact_triangle(collision_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    meshes, _ = load_meshes(collision_path, include_guide_purpose=True)
    candidates: list[tuple[float, str, int, np.ndarray, np.ndarray]] = []
    all_vertices = []
    for mesh in meshes:
        all_vertices.append(mesh.world_vertices_m)
        triangles = mesh.triangles[
            np.all(
                (mesh.triangles >= 0) & (mesh.triangles < len(mesh.world_vertices_m)),
                axis=1,
            )
        ]
        coordinates = mesh.world_vertices_m[triangles]
        cross = np.cross(
            coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]
        )
        areas = np.linalg.norm(cross, axis=1) * 0.5
        normals = np.divide(
            cross,
            np.maximum(np.linalg.norm(cross, axis=1), 1e-30)[:, None],
        )
        for index, (triangle, normal, area) in enumerate(
            zip(coordinates, normals, areas, strict=True)
        ):
            if float(normal[2]) >= 0.5 and float(area) > 0.0:
                candidates.append((float(area), mesh.path, index, triangle, normal))
    if not candidates or not all_vertices:
        raise RuntimeError("static collision has no upward-facing triangle for a contact canary")
    _area, _path, _index, triangle, normal = max(
        candidates,
        key=lambda item: (item[0], item[1], -item[2]),
    )
    vertices = np.concatenate(all_vertices, axis=0)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    return triangle.mean(axis=0), normal, diagonal


def _author_static_contact_scene(
    collision_path: Path,
    output_path: Path,
) -> tuple[Path, dict[str, float]]:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    meshes, _ = load_meshes(collision_path, include_guide_purpose=True)
    contact_point, _normal, diagonal = _select_static_contact_triangle(collision_path)
    radius = max(diagonal * 0.02, 1e-4)
    drop = max(diagonal * 0.05, radius)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/StaticEnvironment")
    for index, source in enumerate(meshes):
        mesh = UsdGeom.Mesh.Define(stage, f"/StaticEnvironment/Collider_{index:04d}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(source.world_vertices_m.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(source.triangles), 3, dtype=np.int32))
        )
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(source.triangles.astype(np.int32).reshape(-1))
        )
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    probe = UsdGeom.Sphere.Define(stage, "/ContactProbe")
    probe.CreateRadiusAttr(radius)
    probe.CreateExtentAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(-radius, -radius, -radius),
                Gf.Vec3f(radius, radius, radius),
            ]
        )
    )
    position = contact_point + np.asarray([0.0, 0.0, radius + drop])
    UsdGeom.Xformable(probe).AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in position]))
    UsdPhysics.CollisionAPI.Apply(probe.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(probe.GetPrim())
    UsdPhysics.MassAPI.Apply(probe.GetPrim()).CreateMassAttr(1.0)
    physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    stage.SetDefaultPrim(probe.GetPrim())
    stage.GetRootLayer().Save()
    return output_path, {
        "contact_height_m": float(contact_point[2]),
        "probe_radius_m": radius,
        "initial_height_m": float(position[2]),
        "asset_diagonal_m": diagonal,
    }


def validate_static_collision_runtime(
    collision_path: str | Path,
    output_dir: str | Path,
    *,
    engine: Literal["fake", "ovphysx"],
) -> dict:
    """Cook a static triangle collider and execute a representative contact canary."""

    output = Path(output_dir).expanduser().resolve()
    scene, geometry = _author_static_contact_scene(
        Path(collision_path).expanduser().resolve(),
        output / "static_contact_scene.usda",
    )
    failures: list[str] = []
    warnings: list[str] = []
    if engine == "fake":
        warnings.append("fake static contact evidence is test-only")
        trajectory_count = 2
        final_height = geometry["contact_height_m"] + geometry["probe_radius_m"]
    else:
        _report, trajectory = _simulate_explicit_scene(
            scene,
            output / "usd_cli_simulation",
            body_path="/ContactProbe",
            rest_position=[
                0.0,
                0.0,
                geometry["contact_height_m"] + geometry["probe_radius_m"],
            ],
            world_up=[0.0, 0.0, 1.0],
            duration_s=1.5,
            dt=1.0 / 240.0,
            sample_fps=60,
        )
        trajectory_count = len(trajectory)
        finite = all(
            np.isfinite(float(value))
            for _time, pose, velocity in trajectory
            for value in [*pose, *velocity]
        )
        if not trajectory:
            failures.append("static contact canary produced no trajectory")
            final_height = None
        elif not finite:
            failures.append("static contact canary produced non-finite trajectory values")
            final_height = None
        else:
            final_height = float(trajectory[-1][1][2])
            expected = geometry["contact_height_m"] + geometry["probe_radius_m"]
            tolerance = max(geometry["probe_radius_m"] * 1.5, geometry["asset_diagonal_m"] * 0.01)
            if final_height < expected - tolerance:
                failures.append(
                    f"contact probe fell through the static collider: final z={final_height:.6g}, "
                    f"expected at least {expected - tolerance:.6g}"
                )
    report_path = atomic_write_json(
        output / "static_contact_runtime_report.json",
        {
            "schema_version": "geometry-repair.static-contact-runtime.v1",
            "engine": engine,
            "scene_usd": str(scene),
            "status": "fail" if failures else "pass",
            "geometry": geometry,
            "trajectory_sample_count": trajectory_count,
            "final_probe_height_m": final_height,
            "failures": failures,
            "warnings": warnings,
        },
    )
    scenario = {
        "scenario_id": "static_contact",
        "scenario_kind": "static_contact",
        "status": "fail" if failures else "pass",
        "runtime_report_path": str(report_path),
        "temporary_proxy_path": str(scene),
        "failures": failures,
        "warnings": warnings,
    }
    return {
        "status": scenario["status"],
        "engine": engine,
        "temporary_proxy_path": str(scene),
        "runtime_report_path": str(report_path),
        "runtime_scenarios": [scenario],
        "failures": failures,
        "warnings": warnings,
    }


def _author_contact_insertion_scene(
    receiver_path: Path,
    moving_path: Path,
    output_path: Path,
    *,
    insertion_axis: tuple[float, float, float],
    seated_root_translation_m: tuple[float, float, float],
    separation_m: float,
    approach_speed_m_s: float,
    moving_mass_kg: float,
) -> Path:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    receiver_all, _ = load_meshes(receiver_path, include_guide_purpose=True)
    moving_all, _ = load_meshes(moving_path, include_guide_purpose=True)
    receiver_meshes = [mesh for mesh in receiver_all if mesh.role == "collision"] or [
        mesh for mesh in receiver_all if mesh.role == "render"
    ]
    moving_meshes = [mesh for mesh in moving_all if mesh.role == "collision"] or [
        mesh for mesh in moving_all if mesh.role == "render"
    ]
    if not receiver_meshes or not moving_meshes:
        raise ValueError("contact insertion requires receiver and moving collision meshes")
    axis = np.asarray(insertion_axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("insertion_axis must have non-zero length")
    axis /= norm
    seated_translation = np.asarray(seated_root_translation_m, dtype=np.float64)
    if seated_translation.shape != (3,) or not np.isfinite(seated_translation).all():
        raise ValueError("seated_root_translation_m must contain three finite values")
    initial_offset = seated_translation - axis * separation_m
    initial_velocity = axis * approach_speed_m_s

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/Receiver")
    for index, source in enumerate(receiver_meshes):
        mesh = UsdGeom.Mesh.Define(stage, f"/Receiver/Collider_{index:04d}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(source.world_vertices_m.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(source.triangles), 3, dtype=np.int32))
        )
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(source.triangles.astype(np.int32).reshape(-1))
        )
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")

    moving_root = UsdGeom.Xform.Define(stage, "/Male")
    moving_root.AddTranslateOp().Set(Gf.Vec3d(*[float(item) for item in initial_offset]))
    rigid = UsdPhysics.RigidBodyAPI.Apply(moving_root.GetPrim())
    rigid.CreateVelocityAttr(Gf.Vec3f(*[float(item) for item in initial_velocity]))
    UsdPhysics.MassAPI.Apply(moving_root.GetPrim()).CreateMassAttr(moving_mass_kg)
    for index, source in enumerate(moving_meshes):
        mesh = UsdGeom.Mesh.Define(stage, f"/Male/Collider_{index:04d}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(source.world_vertices_m.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(source.triangles), 3, dtype=np.int32))
        )
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(source.triangles.astype(np.int32).reshape(-1))
        )
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
            "convexDecomposition"
        )
    physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(0.0)
    stage.SetDefaultPrim(moving_root.GetPrim())
    layer_data = dict(stage.GetRootLayer().customLayerData or {})
    layer_data.update(
        {
            "claimScope": "geometry_repair.contact_rich.runtime_validation",
            "geometryRepairInsertionAxis": ",".join(f"{float(item):.17g}" for item in axis),
            "geometryRepairInitialSeparationM": float(separation_m),
            "geometryRepairSeatedRootTranslationM": ",".join(
                f"{float(item):.17g}" for item in seated_translation
            ),
        }
    )
    stage.GetRootLayer().customLayerData = layer_data
    stage.GetRootLayer().Save()
    return output_path


def validate_contact_insertion_runtime(
    receiver_path: str | Path,
    moving_path: str | Path,
    output_dir: str | Path,
    *,
    engine: Literal["ovphysx"],
    insertion_axis: tuple[float, float, float],
    seated_root_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    separation_m: float = 0.04,
    approach_speed_m_s: float = 0.05,
    duration_s: float = 1.5,
    moving_mass_kg: float = 0.1,
    seat_tolerance_m: float = 0.003,
    penetration_tolerance_m: float = 0.001,
    lateral_tolerance_m: float = 0.001,
) -> dict:
    """Execute a dynamic approach-to-seat canary and measure contact response."""

    if engine != "ovphysx":
        raise ValueError("contact insertion currently requires authoritative ovphysx")
    if separation_m <= 0.0 or approach_speed_m_s <= 0.0 or duration_s <= 0.0:
        raise ValueError("contact insertion distances, speed, and duration must be positive")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    axis = np.asarray(insertion_axis, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12 or not np.isfinite(axis_norm):
        raise ValueError("insertion_axis must have finite non-zero length")
    axis /= axis_norm
    seated_translation = np.asarray(seated_root_translation_m, dtype=np.float64)
    if seated_translation.shape != (3,) or not np.isfinite(seated_translation).all():
        raise ValueError("seated_root_translation_m must contain three finite values")
    scene = _author_contact_insertion_scene(
        Path(receiver_path).expanduser().resolve(),
        Path(moving_path).expanduser().resolve(),
        output / "contact_insertion_scene.usda",
        insertion_axis=tuple(float(item) for item in axis),
        seated_root_translation_m=tuple(float(item) for item in seated_translation),
        separation_m=separation_m,
        approach_speed_m_s=approach_speed_m_s,
        moving_mass_kg=moving_mass_kg,
    )
    runtime_response, trajectory = _simulate_explicit_scene(
        scene,
        output / "usd_cli_simulation",
        body_path="/Male",
        rest_position=[float(item) for item in seated_translation],
        world_up=[0.0, 0.0, 1.0],
        duration_s=duration_s,
        dt=1.0 / 240.0,
        sample_fps=120,
    )
    failures: list[str] = []
    if not trajectory:
        failures.append("contact insertion produced no solver trajectory")
        positions = np.empty((0, 3), dtype=np.float64)
        velocities = np.empty((0, 3), dtype=np.float64)
    else:
        positions = np.asarray([sample[1][:3] for sample in trajectory], dtype=np.float64)
        velocities = np.asarray([sample[2][:3] for sample in trajectory], dtype=np.float64)
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            failures.append("contact insertion produced non-finite position or velocity values")
    metrics: dict[str, float | int | None] = {
        "trajectory_sample_count": len(trajectory),
        "separation_m": separation_m,
        "approach_speed_m_s": approach_speed_m_s,
        "duration_s": duration_s,
    }
    if len(positions):
        start = positions[0]
        final = positions[-1]
        progress = float(np.dot(final - start, axis))
        final_from_seat = final - seated_translation
        signed_seat_offset = float(np.dot(final_from_seat, axis))
        penetration = max(0.0, signed_seat_offset)
        short_of_seat = max(0.0, -signed_seat_offset)
        lateral = final_from_seat - axis * signed_seat_offset
        lateral_error = float(np.linalg.norm(lateral))
        axial_velocities = velocities @ axis
        final_axial_speed = abs(float(axial_velocities[-1]))
        contact_response = bool(
            progress >= separation_m - seat_tolerance_m
            and final_axial_speed <= approach_speed_m_s * 0.25
        )
        metrics.update(
            {
                "axial_progress_m": progress,
                "signed_seat_offset_m": signed_seat_offset,
                "short_of_seat_m": short_of_seat,
                "penetration_m": penetration,
                "final_lateral_error_m": lateral_error,
                "final_axial_speed_m_s": final_axial_speed,
                "contact_response_detected": contact_response,
            }
        )
        if short_of_seat > seat_tolerance_m:
            failures.append(f"moving part stopped {short_of_seat:.6g} m short of the seated pose")
        if penetration > penetration_tolerance_m:
            failures.append(f"moving part penetrated {penetration:.6g} m beyond the seated pose")
        if lateral_error > lateral_tolerance_m:
            failures.append(f"moving part lateral drift {lateral_error:.6g} m exceeds tolerance")
        if not contact_response:
            failures.append("solver trajectory did not show a bounded seated contact response")
    report_path = atomic_write_json(
        output / "contact_insertion_runtime_report.json",
        {
            "schema_version": "geometry-repair.contact-insertion-runtime.v1",
            "engine": engine,
            "status": "fail" if failures else "pass",
            "scene_usd": str(scene),
            "receiver_path": str(Path(receiver_path).expanduser().resolve()),
            "moving_path": str(Path(moving_path).expanduser().resolve()),
            "insertion_axis": [float(item) for item in axis],
            "seated_root_translation_m": [float(item) for item in seated_translation],
            "metrics": metrics,
            "failures": failures,
            "runtime_response": runtime_response,
            "claim_boundary": (
                "This is a fixed-property contact canary, not final friction, force, "
                "controller, or task-policy certification."
            ),
        },
    )
    return {
        "status": "fail" if failures else "pass",
        "engine": engine,
        "scene_usd": str(scene),
        "runtime_report_path": str(report_path),
        "metrics": metrics,
        "failures": failures,
    }
