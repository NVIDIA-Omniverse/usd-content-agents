# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused USD-ingest and Newton camera-analysis correctness tests."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom
from usd_core.camera_analysis import coverage as coverage_module
from usd_core.camera_analysis import look_at
from usd_core.camera_analysis import scene as scene_module
from usd_core.camera_analysis.authoring import author_camera_rig
from usd_core.camera_analysis.calibration import camera_calibration
from usd_core.camera_analysis.contracts import (
    CameraBatchRequest,
    CameraObservation,
    CameraPose,
    RayHits,
    SceneAnalysisPolicy,
    VisibilityBackend,
)
from usd_core.camera_analysis.coverage import (
    GridSurface,
    frustum_mask,
    sample_surface,
    validate_visibility_region_workload,
)
from usd_core.camera_analysis.evidence import _newton_ovrtx_parity, semantic_label
from usd_core.camera_analysis.look_at import LookAtConfig, LookAtResult
from usd_core.camera_analysis.newton_backend import NewtonVisibilityBackend
from usd_core.camera_analysis.placement import (
    _analytic_surface_samples,
    _coverage_candidates,
    _geometry_clearance_mask,
    _target_surface_samples,
    place_cameras_look_at,
    place_max_coverage,
    validate_look_at_workload,
)
from usd_core.camera_analysis.scene import (
    _source_digest,
    analysis_bounds,
    build_scene_analysis_ir,
    camera_pose,
    transform_points,
)


def _stage(*, meters_per_unit: float = 1.0, up_axis: str = "Z") -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.SetStageUpAxis(
        stage,
        UsdGeom.Tokens.y if up_axis == "Y" else UsdGeom.Tokens.z,
    )
    return stage


def test_public_analysis_contract_surface_is_import_light() -> None:
    source = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(source), env.get("PYTHONPATH", "")) if item
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from usd_core.camera_analysis import ("
            "AnalysisRole, CameraBatchRequest, CameraObservation, "
            "SceneAnalysisPolicy, VisibilityBackend); "
            "import sys; "
            "assert AnalysisRole and CameraBatchRequest and CameraObservation and "
            "SceneAnalysisPolicy and VisibilityBackend; "
            "assert 'newton' not in sys.modules and 'warp' not in sys.modules",
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )


def test_backend_rejects_unqualified_versions_before_preflight_or_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from usd_core.camera_analysis import newton_backend

    stage = _stage()
    UsdGeom.Cube.Define(stage, "/Cube")
    scene = build_scene_analysis_ir(stage)
    monkeypatch.setattr(
        newton_backend,
        "backend_versions",
        lambda: {"newton": "1.5.0", "warp": "1.15.0", "qualified": False},
    )
    monkeypatch.setattr(
        newton_backend.NewtonVisibilityBackend,
        "_preflight_scene",
        lambda _self: pytest.fail(
            "scene preflight must not start with an unqualified backend"
        ),
    )
    monkeypatch.setattr(
        newton_backend,
        "_lazy_imports",
        lambda: pytest.fail("Newton/Warp must not import when versions are unqualified"),
    )

    with pytest.raises(
        newton_backend.CameraAnalysisUnavailable,
        match=r"requires qualified Newton 1\.5\.x and Warp 1\.16\.x.*warp=1\.15\.0",
    ):
        newton_backend.NewtonVisibilityBackend(scene)


def test_visibility_region_workload_rejects_aggregate_checkerboard_output() -> None:
    rows = columns = 174
    checkerboard = np.indices((rows, columns)).sum(axis=0) % 2 == 0
    accessible = np.ones_like(checkerboard)

    assert (
        validate_visibility_region_workload(checkerboard[None, ...], accessible)
        == 60_552
    )
    with pytest.raises(ValueError, match="visibility_region_complexity_limit"):
        validate_visibility_region_workload(
            np.repeat(checkerboard[None, ...], 4, axis=0), accessible
        )


def _triangle_mesh(stage: Usd.Stage, path: str) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [Gf.Vec3f(-0.5, -0.5, 0.0), Gf.Vec3f(0.5, -0.5, 0.0), Gf.Vec3f(0.0, 0.5, 0.0)]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return mesh


def test_usd_schema_defaults_are_preserved_in_canonical_meters() -> None:
    stage = _stage(meters_per_unit=0.01, up_axis="Y")
    UsdGeom.Cube.Define(stage, "/Cube")
    UsdGeom.Sphere.Define(stage, "/Sphere")
    UsdGeom.Capsule.Define(stage, "/Capsule")
    UsdGeom.Cylinder.Define(stage, "/Cylinder")
    UsdGeom.Cone.Define(stage, "/Cone")
    UsdGeom.Plane.Define(stage, "/Plane")
    UsdGeom.Camera.Define(stage, "/Camera")

    scene = build_scene_analysis_ir(stage)
    shapes = {shape.prim_path: shape for shape in scene.shapes}
    assert shapes["/Cube"].parameters == pytest.approx(
        {"hx": 0.01, "hy": 0.01, "hz": 0.01}
    )
    assert shapes["/Sphere"].parameters == pytest.approx({"radius": 0.01})
    assert shapes["/Capsule"].parameters == pytest.approx(
        {"radius": 0.005, "half_height": 0.005}
    )
    assert shapes["/Cylinder"].parameters == pytest.approx(
        {"radius": 0.01, "half_height": 0.01}
    )
    assert shapes["/Cone"].parameters == pytest.approx(
        {"radius": 0.01, "half_height": 0.01}
    )
    assert shapes["/Plane"].parameters == pytest.approx({"width": 0.02, "length": 0.02})
    camera = scene.cameras[0]
    assert camera.focal_length_mm == pytest.approx(50.0)
    assert camera.horizontal_aperture_mm == pytest.approx(20.955)
    assert camera.vertical_aperture_mm == pytest.approx(15.2908)
    assert camera.clipping_range_m == pytest.approx((0.01, 10_000.0))


def test_shape_path_filters_do_not_filter_stage_camera_inventory() -> None:
    stage = _stage()
    UsdGeom.Cube.Define(stage, "/Included/Geometry")
    UsdGeom.Cube.Define(stage, "/Included/Excluded/Geometry")
    UsdGeom.Camera.Define(stage, "/Included/Camera")
    UsdGeom.Camera.Define(stage, "/Included/Excluded/Camera")
    UsdGeom.Camera.Define(stage, "/OutsideCamera")

    scene = build_scene_analysis_ir(
        stage,
        policy=SceneAnalysisPolicy(
            include_paths=("/Included",),
            exclude_paths=("/Included/Excluded",),
        ),
    )

    assert [shape.prim_path for shape in scene.shapes] == ["/Included/Geometry"]
    assert [camera.prim_path for camera in scene.cameras] == [
        "/Included/Camera",
        "/Included/Excluded/Camera",
        "/OutsideCamera",
    ]


def test_explicit_zero_meters_per_unit_fails_closed() -> None:
    stage = _stage()
    stage.SetMetadata("metersPerUnit", 0.0)
    UsdGeom.Cube.Define(stage, "/Cube")

    with pytest.raises(ValueError, match="metersPerUnit must be finite and positive"):
        build_scene_analysis_ir(stage)


def test_mesh_holes_and_left_handed_orientation_survive_triangulation() -> None:
    stage = _stage()
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(1, 0, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(0, 1, 0),
            Gf.Vec3f(2, 0, 0),
            Gf.Vec3f(2, 1, 0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 1, 4, 5])
    mesh.CreateHoleIndicesAttr([1])
    mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    scene = build_scene_analysis_ir(stage)
    assert len(scene.meshes) == 1
    assert scene.meshes[0].indices.tolist() == [0, 2, 1, 0, 3, 2]


@pytest.mark.parametrize("failure", ["concave", "nonplanar"])
def test_mesh_ngons_fail_when_fan_triangulation_changes_the_surface(
    failure: str,
) -> None:
    stage = _stage()
    mesh = UsdGeom.Mesh.Define(stage, "/UnsafeFace")
    points = [
        Gf.Vec3f(0, 0, 0),
        Gf.Vec3f(2, 0, 0),
        Gf.Vec3f(2, 2, 0),
        Gf.Vec3f(1, 1, 0),
        Gf.Vec3f(0, 2, 0),
    ]
    if failure == "nonplanar":
        points = points[:4]
        points[2] = Gf.Vec3f(2, 2, 0.5)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(points)])
    mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    with pytest.raises(ValueError, match=rf"unsupported_{failure}_mesh_face"):
        build_scene_analysis_ir(stage)


def test_source_digest_streams_binary_flattening_without_usda_text() -> None:
    payload = b"PXR-USDC\x00streamed-composed-stage"

    class FlatLayer:
        def ExportToString(self):
            raise AssertionError("source digest must not materialize USDA text")

        def Export(self, path: str) -> bool:
            with open(path, "wb") as stream:  # noqa: PTH123 - fake USD exporter
                stream.write(payload)
            return True

    class Stage:
        def Flatten(self):
            return FlatLayer()

    assert _source_digest(Stage()) == "sha256:" + hashlib.sha256(payload).hexdigest()


def test_analysis_bounds_exclude_distant_helpers() -> None:
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/Scope")
    floor = UsdGeom.Cube.Define(stage, "/Scope/Floor")
    floor.CreateSizeAttr(10.0)
    helper = UsdGeom.Cube.Define(stage, "/Scope/Guide")
    helper.AddTranslateOp().Set(Gf.Vec3d(1000.0, 0.0, 0.0))
    scene = build_scene_analysis_ir(
        stage,
        policy=SceneAnalysisPolicy(
            scope_paths=("/Scope",),
            floor_paths=("/Scope/Floor",),
            helper_paths=("/Scope/Guide",),
        ),
    )

    minimum, maximum = analysis_bounds(scene, roots=("/Scope",))
    assert minimum == pytest.approx([-5.0, -5.0, -5.0])
    assert maximum == pytest.approx([5.0, 5.0, 5.0])


def test_analysis_bounds_scans_a_shared_packed_mesh_resource_once() -> None:
    stage = _stage()
    _triangle_mesh(stage, "/Target")
    scene = build_scene_analysis_ir(stage)
    resource = scene.meshes[0]
    shape = scene.shapes[0]

    class TrackingVertices:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.array_calls = 0

        def __len__(self) -> int:
            return len(self.values)

        def __array__(
            self,
            dtype: np.dtype | None = None,
            copy: bool | None = None,
        ) -> np.ndarray:
            del copy
            self.array_calls += 1
            assert dtype is None, "bounds must not copy every vertex to float64"
            return self.values

    tracking = TrackingVertices(resource.vertices)
    translated_transform = np.asarray(shape.transform, dtype=np.float64).copy()
    translated_transform[3, 0] += 10.0
    shared_scene = replace(
        scene,
        meshes=(replace(resource, vertices=tracking),),
        shapes=(
            shape,
            replace(
                shape,
                shape_id=shape.shape_id + 1,
                prim_path="/TargetCopy",
                transform=tuple(
                    tuple(float(value) for value in row) for row in translated_transform
                ),
            ),
        ),
    )

    minimum, maximum = analysis_bounds(shared_scene)

    assert tracking.array_calls == 1
    assert minimum == pytest.approx([-0.5, -0.5, 0.0])
    assert maximum == pytest.approx([10.5, 0.5, 0.0])


def test_analysis_bounds_caps_distinct_instance_transform_vertex_visits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage()
    _triangle_mesh(stage, "/Target")
    scene = build_scene_analysis_ir(stage)
    resource = scene.meshes[0]
    shape = scene.shapes[0]

    class TrackingVertices:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.array_calls = 0

        def __len__(self) -> int:
            return len(self.values)

        def __array__(
            self,
            dtype: np.dtype | None = None,
            copy: bool | None = None,
        ) -> np.ndarray:
            del dtype, copy
            self.array_calls += 1
            return self.values

    tracking = TrackingVertices(resource.vertices)
    scaled_transform = np.asarray(shape.transform, dtype=np.float64).copy()
    scaled_transform[0, 0] = 2.0
    varied_scene = replace(
        scene,
        meshes=(replace(resource, vertices=tracking),),
        shapes=(
            shape,
            replace(
                shape,
                shape_id=shape.shape_id + 1,
                prim_path="/TargetCopy",
                transform=tuple(
                    tuple(float(value) for value in row) for row in scaled_transform
                ),
            ),
        ),
    )
    monkeypatch.setattr(scene_module, "MAX_ANALYSIS_BOUNDS_VERTEX_VISITS", 3)

    with pytest.raises(ValueError, match="analysis_bounds_workload_limit"):
        analysis_bounds(varied_scene)
    assert tracking.array_calls == 0


def test_probe_ceiling_scans_shared_mesh_once_across_distinct_transforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage()
    _triangle_mesh(stage, "/Target")
    scene = build_scene_analysis_ir(stage)
    resource = scene.meshes[0]
    shape = scene.shapes[0]

    class TrackingVertices:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.array_calls = 0

        def __len__(self) -> int:
            return len(self.values)

        def __array__(
            self,
            dtype: np.dtype | None = None,
            copy: bool | None = None,
        ) -> np.ndarray:
            del copy
            self.array_calls += 1
            assert dtype is None, "ceiling must not copy every vertex to float64"
            return self.values

    tracking = TrackingVertices(resource.vertices)
    occurrences = []
    for index in range(6):
        angle = math.radians(index * 10.0)
        cosine, sine = math.cos(angle), math.sin(angle)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(
            [
                [cosine, 0.0, -sine],
                [0.0, 1.0, 0.0],
                [sine, 0.0, cosine],
            ]
        )
        transform[3, 2] = float(index)
        occurrences.append(
            replace(
                shape,
                shape_id=index,
                prim_path=f"/Target{index}",
                transform=tuple(
                    tuple(float(value) for value in row) for row in transform
                ),
            )
        )
    shared_scene = replace(
        scene,
        meshes=(replace(resource, vertices=tracking),),
        shapes=tuple(occurrences),
    )
    monkeypatch.setattr(
        scene_module,
        "MAX_ANALYSIS_BOUNDS_VERTEX_VISITS",
        len(resource.vertices) * 5,
    )

    # Exact global bounds multiply the shared vertex count by six distinct
    # linear transforms and hit the simulated five-resource visit ceiling.
    with pytest.raises(ValueError, match="analysis_bounds_workload_limit"):
        analysis_bounds(shared_scene)
    assert tracking.array_calls == 0

    ceiling_m = coverage_module._admitted_scene_ceiling_m(shared_scene)
    exact_ceiling_m = max(
        float(transform_points(resource.vertices, occurrence.transform)[:, 2].max())
        for occurrence in occurrences
    )
    assert tracking.array_calls == 1
    assert ceiling_m >= exact_ceiling_m


@pytest.mark.parametrize(
    "schema_name", ["Cube", "Sphere", "Capsule", "Cylinder", "Cone", "Plane"]
)
def test_probe_ceiling_matches_analytic_shape_bounds(schema_name: str) -> None:
    stage = _stage()
    schema = getattr(UsdGeom, schema_name)
    shape = schema.Define(stage, "/Shape")
    shape.AddRotateYOp().Set(37.0)
    shape.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 2.0))
    helper = UsdGeom.Cube.Define(stage, "/HighHelper")
    helper.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 100.0))
    scene = build_scene_analysis_ir(
        stage,
        policy=SceneAnalysisPolicy(helper_paths=("/HighHelper",)),
    )

    assert coverage_module._admitted_scene_ceiling_m(scene) == pytest.approx(
        analysis_bounds(scene)[1][2]
    )


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_analysis_bounds_keep_a_thin_rotated_mesh_floor_sampleable() -> None:
    stage = _stage()
    floor = UsdGeom.Mesh.Define(stage, "/Floor")
    floor.CreatePointsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(10.0, 10.0, 0.0),
            Gf.Vec3f(10.001, 9.999, 0.0),
        ]
    )
    floor.CreateFaceVertexCountsAttr([3])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2])
    floor.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    floor.AddRotateZOp().Set(-45.0)
    scene = build_scene_analysis_ir(
        stage,
        policy=SceneAnalysisPolicy(scope_paths=("/Floor",), floor_paths=("/Floor",)),
    )
    shape = scene.shapes[0]
    world_vertices = transform_points(scene.meshes[0].vertices, shape.transform)

    bounds = analysis_bounds(scene, roots=("/Floor",))

    assert bounds[0] == pytest.approx(world_vertices.min(axis=0))
    assert bounds[1] == pytest.approx(world_vertices.max(axis=0))
    sampled = sample_surface(
        scene,
        NewtonVisibilityBackend(scene, device="cpu"),
        bounds,
        scope_path="/Floor",
        grid=32,
    )
    assert sampled.accessible.sum() > 0


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_scoped_floor_probe_starts_above_external_admitted_obstacles() -> None:
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World")
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.CreateSizeAttr(1.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.05))
    floor.AddScaleOp().Set(Gf.Vec3f(4.0, 2.0, 0.1))
    shelf = UsdGeom.Cube.Define(stage, "/World/OverheadShelf")
    shelf.CreateSizeAttr(1.0)
    shelf.AddTranslateOp().Set(Gf.Vec3d(-0.5, 0.0, 3.0))
    shelf.AddScaleOp().Set(Gf.Vec3f(0.4, 1.8, 0.4))
    scene = build_scene_analysis_ir(
        stage,
        policy=SceneAnalysisPolicy(
            scope_paths=("/World/Floor",),
            floor_paths=("/World/Floor",),
        ),
    )
    floor_bounds = analysis_bounds(scene, roots=("/World/Floor",))
    assert floor_bounds[1][2] == pytest.approx(0.0, abs=1.0e-8)
    assert analysis_bounds(scene)[1][2] == pytest.approx(3.2)

    surface = sample_surface(
        scene,
        NewtonVisibilityBackend(scene),
        floor_bounds,
        scope_path="/World/Floor",
        grid=4,
    )

    # The shelf is outside the floor Gprim scope and starts well above the old
    # one-metre probe clearance, but remains an admitted occluder.  It covers
    # the second column; the rest of the scoped floor stays accessible.
    assert surface.accessible[:, 1].tolist() == [False, False]
    assert surface.accessible[:, [0, 2, 3]].all()


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_look_at_unobstructed_sphere_passes_default_occlusion_threshold() -> None:
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Sphere.Define(stage, "/World/Target")
    scene = build_scene_analysis_ir(
        stage,
        policy=SceneAnalysisPolicy(target_paths=("/World/Target",)),
    )

    placement = place_cameras_look_at(
        scene,
        NewtonVisibilityBackend(scene, device="cpu"),
        (np.asarray([-1.0, -1.0, -1.0]), np.asarray([1.0, 1.0, 1.0])),
        target_path="/World/Target",
    )

    assert placement.report["occlusion_threshold"] == 0.4
    assert placement.report["passed"] is True
    assert placement.report["selected_count"] == 4
    assert len(placement.poses) == 4
    for camera in placement.report["cameras"]:
        assert camera["visible_target_samples"] == camera["target_sample_count"]
        assert camera["occlusion_fraction"] == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_look_at_counts_out_of_frame_target_samples_as_not_visible() -> None:
    stage = _stage()
    UsdGeom.Sphere.Define(stage, "/Target")
    scene = build_scene_analysis_ir(
        stage,
        policy=SceneAnalysisPolicy(target_paths=("/Target",)),
    )
    backend = NewtonVisibilityBackend(scene, device="cpu")
    bounds = analysis_bounds(scene, roots=("/Target",))
    parameters = {
        "target_path": "/Target",
        "camera_count": 1,
        "candidate_count": 8,
        "min_distance_m": 1.1,
        "max_distance_m": 1.1,
        "focal_length_mm": 1000.0,
        "aperture_mm": 36.0,
    }

    permissive = place_cameras_look_at(
        scene,
        backend,
        bounds,
        occlusion_threshold=1.0,
        **parameters,
    )
    camera = permissive.report["cameras"][0]
    assert camera["in_frame_target_samples"] == 0
    assert camera["visible_target_samples"] == 0
    assert camera["occlusion_fraction"] == pytest.approx(1.0)

    strict = place_cameras_look_at(
        scene,
        backend,
        bounds,
        occlusion_threshold=0.0,
        **parameters,
    )
    assert strict.report["passed"] is False
    assert strict.report["selected_count"] == 0
    assert strict.report["valid_candidate_count"] == 0


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_mesh_target_samples_hit_target_and_not_prefix_occluder() -> None:
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World")
    _triangle_mesh(stage, "/World/Target")
    policy = SceneAnalysisPolicy(target_paths=("/World/Target",))
    clear_scene = build_scene_analysis_ir(stage, policy=policy)
    target_shape = next(
        shape for shape in clear_scene.shapes if shape.prim_path == "/World/Target"
    )
    samples = _target_surface_samples(clear_scene, (target_shape,))
    origin = np.asarray([0.0, 0.0, 2.0])
    directions = samples - origin
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    origins = np.repeat(origin[None, :], len(samples), axis=0)
    clear_hits = NewtonVisibilityBackend(clear_scene, device="cpu").evaluate_rays(
        origins, directions
    )
    assert set(clear_hits.shape_ids.tolist()) == {target_shape.shape_id}

    occluder = UsdGeom.Cube.Define(stage, "/World/TargetBackdrop")
    occluder.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
    occluder.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 0.5))
    blocked_scene = build_scene_analysis_ir(stage, policy=policy)
    blocked_target = next(
        shape for shape in blocked_scene.shapes if shape.prim_path == "/World/Target"
    )
    blocked_samples = _target_surface_samples(blocked_scene, (blocked_target,))
    blocked_directions = blocked_samples - origin
    blocked_directions /= np.linalg.norm(blocked_directions, axis=1)[:, None]
    blocked_hits = NewtonVisibilityBackend(blocked_scene, device="cpu").evaluate_rays(
        origins, blocked_directions
    )
    assert blocked_target.shape_id not in set(blocked_hits.shape_ids.tolist())
    assert {
        blocked_scene.shape_path_by_id[int(shape_id)]
        for shape_id in blocked_hits.shape_ids
    } == {"/World/TargetBackdrop"}


def test_mesh_target_samples_use_world_area_after_anisotropic_scale() -> None:
    stage = _stage()
    mesh = UsdGeom.Mesh.Define(stage, "/Target")
    # These triangles have equal local area but lie in different planes. Scaling
    # local X by nine grows only the XY triangle's canonical-world area.
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
            Gf.Vec3f(2.0, 0.0, 0.0),
            Gf.Vec3f(2.0, 1.0, 0.0),
            Gf.Vec3f(2.0, 0.0, 1.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.AddScaleOp().Set(Gf.Vec3f(9.0, 1.0, 1.0))
    scene = build_scene_analysis_ir(
        stage, policy=SceneAnalysisPolicy(target_paths=("/Target",))
    )
    target = next(shape for shape in scene.shapes if shape.prim_path == "/Target")

    samples = _target_surface_samples(scene, (target,))

    # Midpoint-stratifying 32 samples over a 9:1 cumulative area split assigns
    # 29 samples to the first face and three to the second. Weighting the shared
    # resource in local space would incorrectly split these 16:16.
    assert np.count_nonzero(samples[:, 0] < 10.0) == 29
    assert np.count_nonzero(samples[:, 0] > 10.0) == 3


def test_look_at_caps_area_scans_across_varied_scale_mesh_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from usd_core.camera_analysis import placement

    stage = _stage()
    UsdGeom.Xform.Define(stage, "/Target")
    _triangle_mesh(stage, "/Target/Mesh")
    scene = build_scene_analysis_ir(stage)
    shape = scene.shapes[0]

    translated = np.asarray(shape.transform, dtype=np.float64).copy()
    translated[3, 0] = 2.0
    translated_shape = replace(
        shape,
        shape_id=1,
        prim_path="/Target/Translated",
        transform=tuple(tuple(float(value) for value in row) for row in translated),
    )
    monkeypatch.setattr(placement, "MAX_LOOK_AT_MESH_AREA_TRIANGLE_VISITS", 2)
    same_metric_scene = replace(scene, shapes=(shape, translated_shape))
    assert (
        len(validate_look_at_workload(same_metric_scene, "/Target", candidate_count=4))
        == 2
    )

    varied = translated.copy()
    varied[0, 0] = 2.0
    varied_scene = replace(
        scene,
        shapes=(
            shape,
            replace(
                translated_shape,
                transform=tuple(tuple(float(value) for value in row) for row in varied),
            ),
        ),
    )
    with pytest.raises(ValueError, match="look_at_surface_sampling_workload_limit"):
        validate_look_at_workload(varied_scene, "/Target", candidate_count=4)


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_long_capsule_samples_barrel_and_measures_central_occluder() -> None:
    stage = _stage()
    capsule = UsdGeom.Capsule.Define(stage, "/Target")
    capsule.CreateRadiusAttr(1.0)
    capsule.CreateHeightAttr(8.0)
    scene = build_scene_analysis_ir(
        stage, policy=SceneAnalysisPolicy(target_paths=("/Target",))
    )
    target = next(shape for shape in scene.shapes if shape.prim_path == "/Target")
    local_samples = _analytic_surface_samples(target, 32)
    assert np.count_nonzero(np.abs(local_samples[:, 2]) < 3.5) >= 16

    samples = transform_points(local_samples, target.transform)
    origin = np.asarray([10.0, 0.0, 0.0])
    directions = samples - origin
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    origins = np.repeat(origin[None, :], len(samples), axis=0)
    clear_hits = NewtonVisibilityBackend(scene).evaluate_rays(origins, directions)
    clear_visible = int(np.count_nonzero(clear_hits.shape_ids == target.shape_id))
    assert clear_visible == len(samples)

    occluder = UsdGeom.Cube.Define(stage, "/BarrelOccluder")
    occluder.AddTranslateOp().Set(Gf.Vec3d(5.0, 0.0, 0.0))
    occluder.AddScaleOp().Set(Gf.Vec3f(0.25, 2.0, 1.5))
    blocked_scene = build_scene_analysis_ir(
        stage, policy=SceneAnalysisPolicy(target_paths=("/Target",))
    )
    blocked_target = next(
        shape for shape in blocked_scene.shapes if shape.prim_path == "/Target"
    )
    blocked_hits = NewtonVisibilityBackend(blocked_scene).evaluate_rays(
        origins, directions
    )
    blocked_visible = int(
        np.count_nonzero(blocked_hits.shape_ids == blocked_target.shape_id)
    )
    assert 0 < blocked_visible < clear_visible


def test_triangulation_preallocates_packed_output_without_python_input_lists(
    monkeypatch,
) -> None:
    class ArrayOnly:
        def __init__(self, values) -> None:
            self.values = np.asarray(values, dtype=np.int32)

        def __array__(self, dtype=None, copy=None):
            return np.array(self.values, dtype=dtype, copy=bool(copy))

        def __iter__(self):
            raise AssertionError("triangulation must not create Python input lists")

    allocations = []
    original_empty = np.empty

    def recorded_empty(shape, *, dtype):
        allocations.append((shape, dtype))
        return original_empty(shape, dtype=dtype)

    monkeypatch.setattr(scene_module.np, "empty", recorded_empty)
    result = scene_module._triangulate(
        ArrayOnly([4]),
        ArrayOnly([0, 1, 2, 3]),
    )

    assert allocations == [((2, 3), np.int32)]
    assert result.dtype == np.int32
    assert result.tolist() == [0, 1, 2, 0, 2, 3]


@pytest.mark.parametrize("case", ["short_face", "bad_hole", "bad_hidden_index"])
def test_malformed_mesh_topology_fails_explicitly(case: str) -> None:
    stage = _stage()
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr([Gf.Vec3f(0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
    if case == "short_face":
        mesh.CreateFaceVertexCountsAttr([2])
        mesh.CreateFaceVertexIndicesAttr([0, 1])
    else:
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr(
            [0, 1, 99] if case == "bad_hidden_index" else [0, 1, 2]
        )
        mesh.CreateHoleIndicesAttr([0 if case == "bad_hidden_index" else 1])

    with pytest.raises(ValueError, match="face|hole|missing point"):
        build_scene_analysis_ir(stage)


def test_subdivision_mesh_fails_before_topology_materialization(monkeypatch) -> None:
    stage = _stage()
    mesh = UsdGeom.Mesh.Define(stage, "/CurvedControlCage")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-1, -1, 0),
            Gf.Vec3f(1, -1, 0),
            Gf.Vec3f(1, 1, 1),
            Gf.Vec3f(-1, 1, 0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    materialized = False

    def unexpected_materialization(_mesh):
        nonlocal materialized
        materialized = True
        raise AssertionError(
            "subdivision validation must precede topology materialization"
        )

    monkeypatch.setattr(scene_module, "_raw_mesh_data", unexpected_materialization)
    with pytest.raises(
        ValueError,
        match="unsupported_subdivision_scheme.*CurvedControlCage.*catmullClark",
    ):
        build_scene_analysis_ir(stage)
    assert materialized is False

    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(-1, -1, 0),
            Gf.Vec3f(1, -1, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(-1, 1, 0),
        ]
    )
    monkeypatch.undo()
    assert [shape.prim_path for shape in build_scene_analysis_ir(stage).shapes] == [
        "/CurvedControlCage"
    ]


def test_visible_unsupported_boundable_requires_explicit_helper_policy() -> None:
    stage = _stage()
    UsdGeom.Cube.Define(stage, "/Floor")
    curves = UsdGeom.BasisCurves.Define(stage, "/CableObstacle")
    curves.CreatePointsAttr([Gf.Vec3f(-1, 0, 1), Gf.Vec3f(1, 0, 1)])
    curves.CreateCurveVertexCountsAttr([2])

    with pytest.raises(
        ValueError,
        match=r"unsupported_geometry_type: /CableObstacle \(BasisCurves\)",
    ):
        build_scene_analysis_ir(
            stage,
            policy=SceneAnalysisPolicy(
                scope_paths=("/Floor",), floor_paths=("/Floor",)
            ),
        )

    scene = build_scene_analysis_ir(
        stage,
        policy=SceneAnalysisPolicy(
            scope_paths=("/Floor",),
            floor_paths=("/Floor",),
            helper_paths=("/CableObstacle",),
        ),
    )
    assert [shape.prim_path for shape in scene.shapes] == ["/Floor"]


def test_computed_visibility_purpose_and_point_instancer_policy() -> None:
    stage = _stage()
    hidden = UsdGeom.Xform.Define(stage, "/Hidden")
    hidden.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdGeom.Cube.Define(stage, "/Hidden/Cube")
    guide = UsdGeom.Cube.Define(stage, "/Guide")
    guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    render = UsdGeom.Cube.Define(stage, "/Render")
    render.CreatePurposeAttr(UsdGeom.Tokens.render)
    render_hidden = UsdGeom.Cube.Define(stage, "/RenderHidden")
    render_hidden.CreatePurposeAttr(UsdGeom.Tokens.render)
    UsdGeom.VisibilityAPI.Apply(render_hidden.GetPrim()).CreateRenderVisibilityAttr(
        UsdGeom.Tokens.invisible
    )

    scene = build_scene_analysis_ir(stage)
    assert [shape.prim_path for shape in scene.shapes] == ["/Render"]

    instanced = _stage()
    point_instancer = UsdGeom.PointInstancer.Define(instanced, "/Instances")
    UsdGeom.Cube.Define(instanced, "/Prototype")
    point_instancer.CreatePrototypesRel().SetTargets([Sdf.Path("/Prototype")])
    point_instancer.CreateProtoIndicesAttr([0])
    point_instancer.CreatePositionsAttr([Gf.Vec3f(0)])
    with pytest.raises(ValueError, match="unsupported_point_instancer"):
        build_scene_analysis_ir(instanced)
    # Scope names the operational surface; it does not exclude external obstacles
    # that can still occlude that surface.
    with pytest.raises(ValueError, match="unsupported_point_instancer"):
        build_scene_analysis_ir(
            instanced, policy=SceneAnalysisPolicy(scope_paths=("/Prototype",))
        )


@pytest.mark.parametrize("ignored_by", ["visibility", "include", "exclude", "helper"])
def test_point_instancer_rejection_respects_explicit_ignore_policy(
    ignored_by: str,
) -> None:
    stage = _stage()
    UsdGeom.Cube.Define(stage, "/Floor")
    point_instancer = UsdGeom.PointInstancer.Define(stage, "/Instances")
    policy = SceneAnalysisPolicy()
    if ignored_by == "visibility":
        point_instancer.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    elif ignored_by == "include":
        policy = SceneAnalysisPolicy(include_paths=("/Floor",))
    elif ignored_by == "exclude":
        policy = SceneAnalysisPolicy(exclude_paths=("/Instances",))
    else:
        policy = SceneAnalysisPolicy(helper_paths=("/Instances",))

    scene = build_scene_analysis_ir(stage, policy=policy)

    assert [shape.prim_path for shape in scene.shapes] == ["/Floor"]


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_analysis_roles_keep_shelf_tops_inaccessible_and_helpers_non_occluding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World")
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.CreateSizeAttr(1.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.05))
    floor.AddScaleOp().Set(Gf.Vec3f(4.0, 2.0, 0.1))
    for name, x in (("Shelf", -0.5), ("Helper", 0.5)):
        cube = UsdGeom.Cube.Define(stage, f"/World/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.5))
        cube.AddScaleOp().Set(Gf.Vec3f(0.4, 1.8, 0.4))
    UsdGeom.Sphere.Define(stage, "/World/Target").AddTranslateOp().Set(
        Gf.Vec3d(10.0, 0.0, 0.5)
    )
    UsdGeom.Cube.Define(stage, "/World/Excluded")
    UsdGeom.Cube.Define(stage, "/Outside")

    broad_policy = SceneAnalysisPolicy(
        scope_paths=("/World", "/World"),
        include_paths=("/World",),
        exclude_paths=("/World/Excluded",),
        helper_paths=("/World/Helper",),
        target_paths=("/World/Target",),
    )
    scene = build_scene_analysis_ir(stage, policy=broad_policy)
    roles = {shape.prim_path: shape.analysis_role for shape in scene.shapes}
    assert roles == {
        "/World/Floor": "obstacle",
        "/World/Helper": "helper",
        "/World/Shelf": "obstacle",
        "/World/Target": "target",
    }
    assert all(shape.in_scope for shape in scene.shapes)
    assert scene.policy.scope_paths == ("/World",)

    backend = NewtonVisibilityBackend(scene)
    helper_id = next(
        shape.shape_id for shape in scene.shapes if shape.analysis_role == "helper"
    )
    assert helper_id not in backend._shape_ir_by_newton_id.values()
    path_checks = 0
    original_path_check = coverage_module.is_path_at_or_below

    def count_path_check(path: str, root: str) -> bool:
        nonlocal path_checks
        path_checks += 1
        return original_path_check(path, root)

    monkeypatch.setattr(coverage_module, "is_path_at_or_below", count_path_check)
    surface = sample_surface(
        scene,
        backend,
        (np.asarray([-2.0, -1.0, -0.1]), np.asarray([2.0, 1.0, 0.8])),
        scope_path="/World",
        grid=4,
    )
    assert path_checks == len(scene.shape_path_by_id)
    # The shelf blocks the inferred bottom floor at x=-0.5. The helper at
    # x=+0.5 is absent from the collision model, so the floor remains accessible.
    assert surface.accessible[:, 1].tolist() == [False, False]
    assert surface.accessible[:, 2].tolist() == [True, True]

    declared_floor = build_scene_analysis_ir(
        stage,
        policy=SceneAnalysisPolicy(
            scope_paths=("/World",),
            exclude_paths=("/World/Excluded", "/Outside"),
            helper_paths=("/World/Helper",),
            target_paths=("/World/Target",),
            floor_paths=("/World/Floor",),
        ),
    )
    declared_roles = {
        shape.prim_path: shape.analysis_role for shape in declared_floor.shapes
    }
    assert declared_roles["/World/Floor"] == "floor"
    assert declared_roles["/World/Shelf"] == "obstacle"


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_native_instances_share_mesh_but_keep_sorted_occurrence_identity() -> None:
    stage = _stage()
    stage.CreateClassPrim("/_Prototype")
    _triangle_mesh(stage, "/_Prototype/Mesh")
    for name, x in (("I2", 2.0), ("I1", 1.0)):
        instance = stage.DefinePrim(f"/{name}", "Xform")
        instance.GetReferences().AddInternalReference("/_Prototype")
        instance.SetInstanceable(True)
        UsdGeom.Xformable(instance).AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.0))

    scene = build_scene_analysis_ir(stage)
    assert len(scene.meshes) == 1
    assert [(shape.shape_id, shape.prim_path) for shape in scene.shapes] == [
        (0, "/I1/Mesh"),
        (1, "/I2/Mesh"),
    ]
    assert all(shape.instance_proxy for shape in scene.shapes)
    assert scene.shapes[0].mesh_resource_id == scene.shapes[1].mesh_resource_id
    assert np.allclose(np.asarray(scene.shapes[0].transform)[3, :3], [1.0, 0.0, 0.0])
    assert np.allclose(np.asarray(scene.shapes[1].transform)[3, :3], [2.0, 0.0, 0.0])

    backend = NewtonVisibilityBackend(scene)
    hits = backend.evaluate_rays(
        np.asarray([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
        np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
    )
    assert hits.shape_ids.tolist() == [0, 1]


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_mirrored_native_mesh_instance_keeps_signed_shared_scale() -> None:
    stage = _stage()
    stage.CreateClassPrim("/_Prototype")
    mesh = UsdGeom.Mesh.Define(stage, "/_Prototype/Mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    instance = stage.DefinePrim("/Mirror", "Xform")
    instance.GetReferences().AddInternalReference("/_Prototype")
    instance.SetInstanceable(True)
    xformable = UsdGeom.Xformable(instance)
    xformable.AddTranslateOp().Set(Gf.Vec3d(3.0, 0.0, 0.0))
    xformable.AddScaleOp().Set(Gf.Vec3f(-1.0, 1.0, 1.0))

    scene = build_scene_analysis_ir(stage)
    assert len(scene.meshes) == 1
    assert len(scene.meshes[0].vertices) == 3

    class NoBakeBackend(NewtonVisibilityBackend):
        max_mesh_vertices = 3

    backend = NoBakeBackend(scene)
    centroid = transform_points(
        np.mean(scene.meshes[0].vertices, axis=0, keepdims=True),
        scene.shapes[0].transform,
    )[0]
    hit = backend.evaluate_rays(
        np.asarray([centroid + [0.0, 0.0, 1.0]]),
        np.asarray([[0.0, 0.0, -1.0]]),
    )
    assert hit.shape_ids.tolist() == [0]


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_mirrored_analytic_primitive_uses_symmetry_preserving_rotation() -> None:
    stage = _stage()
    cube = UsdGeom.Cube.Define(stage, "/Cube")
    UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(-2.0, 1.0, 1.0))
    backend = NewtonVisibilityBackend(build_scene_analysis_ir(stage))
    hit = backend.evaluate_rays(
        np.asarray([[0.0, 0.0, 3.0]]),
        np.asarray([[0.0, 0.0, -1.0]]),
    )
    assert hit.distances_m[0] == pytest.approx(2.0, abs=1.0e-5)


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_mesh_shear_is_baked_but_analytic_shear_is_rejected() -> None:
    mesh_stage = _stage()
    mesh = _triangle_mesh(mesh_stage, "/Mesh")
    shear = Gf.Matrix4d(1.0)
    shear.SetRow(0, Gf.Vec4d(1.0, 0.5, 0.0, 0.0))
    UsdGeom.Xformable(mesh).AddTransformOp().Set(shear)
    mesh_scene = build_scene_analysis_ir(mesh_stage)
    backend = NewtonVisibilityBackend(mesh_scene)
    centroid = transform_points(
        np.mean(mesh_scene.meshes[0].vertices, axis=0, keepdims=True),
        mesh_scene.shapes[0].transform,
    )[0]
    hit = backend.evaluate_rays(
        np.asarray([centroid + [0.0, 0.0, 1.0]]),
        np.asarray([[0.0, 0.0, -1.0]]),
    )
    assert hit.shape_ids.tolist() == [0]

    analytic_stage = _stage()
    cube = UsdGeom.Cube.Define(analytic_stage, "/Cube")
    UsdGeom.Xformable(cube).AddTransformOp().Set(shear)
    with pytest.raises(ValueError, match="unsupported_affine_transform.*shear"):
        NewtonVisibilityBackend(build_scene_analysis_ir(analytic_stage))


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_nonuniform_sphere_uses_ellipsoid_geometry() -> None:
    stage = _stage()
    sphere = UsdGeom.Sphere.Define(stage, "/Sphere")
    UsdGeom.Xformable(sphere).AddScaleOp().Set(Gf.Vec3f(2.0, 1.0, 1.0))
    backend = NewtonVisibilityBackend(build_scene_analysis_ir(stage))
    hit = backend.evaluate_rays(
        np.asarray([[3.0, 0.0, 0.0]]),
        np.asarray([[-1.0, 0.0, 0.0]]),
    )
    assert hit.distances_m[0] == pytest.approx(1.0, abs=1.0e-5)


@pytest.mark.usefixtures("qualified_camera_analysis_metadata")
def test_axially_scaled_capsule_is_rejected_before_newton_allocation(
    monkeypatch,
) -> None:
    stage = _stage()
    capsule = UsdGeom.Capsule.Define(stage, "/Capsule")
    UsdGeom.Xformable(capsule).AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 2.0))
    scene = build_scene_analysis_ir(stage)
    imported = False

    def unexpected_import():
        nonlocal imported
        imported = True
        raise AssertionError("capsule validation must precede Newton allocation")

    monkeypatch.setattr(
        "usd_core.camera_analysis.newton_backend._lazy_imports", unexpected_import
    )
    with pytest.raises(
        ValueError,
        match="unsupported_affine_transform.*nonuniform capsule scale",
    ):
        NewtonVisibilityBackend(scene)
    assert imported is False


@pytest.mark.parametrize(
    ("schema", "kind"),
    [(UsdGeom.Cylinder, "cylinder"), (UsdGeom.Cone, "cone")],
)
@pytest.mark.usefixtures("qualified_camera_analysis_metadata")
def test_elliptical_radial_shape_is_rejected_before_newton_allocation(
    monkeypatch, schema, kind: str
) -> None:
    stage = _stage()
    shape = schema.Define(stage, f"/{kind.title()}")
    UsdGeom.Xformable(shape).AddScaleOp().Set(Gf.Vec3f(2.0, 1.0, 1.0))
    scene = build_scene_analysis_ir(stage)
    imported = False

    def unexpected_import():
        nonlocal imported
        imported = True
        raise AssertionError(f"{kind} validation must precede Newton allocation")

    monkeypatch.setattr(
        "usd_core.camera_analysis.newton_backend._lazy_imports", unexpected_import
    )
    with pytest.raises(
        ValueError,
        match=rf"unsupported_affine_transform.*elliptical {kind} cross-section",
    ):
        NewtonVisibilityBackend(scene)
    assert imported is False


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_transform_only_stage_change_is_observed_after_backend_rebuild() -> None:
    stage = _stage()
    cube = UsdGeom.Cube.Define(stage, "/Cube")
    initial_scene = build_scene_analysis_ir(stage)
    initial_backend = NewtonVisibilityBackend(initial_scene)
    downward = np.asarray([[0.0, 0.0, -1.0]])
    assert initial_backend.evaluate_rays(
        np.asarray([[0.0, 0.0, 3.0]]), downward
    ).shape_ids.tolist() == [0]

    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(4.0, 0.0, 0.0))
    rebuilt_scene = build_scene_analysis_ir(stage)
    rebuilt_backend = NewtonVisibilityBackend(rebuilt_scene)

    assert rebuilt_scene.source_digest != initial_scene.source_digest
    assert rebuilt_backend.evaluate_rays(
        np.asarray([[0.0, 0.0, 3.0]]), downward
    ).shape_ids.tolist() == [-1]
    assert rebuilt_backend.evaluate_rays(
        np.asarray([[4.0, 0.0, 3.0]]), downward
    ).shape_ids.tolist() == [0]


@pytest.mark.usefixtures("qualified_camera_analysis_metadata")
def test_scene_and_candidate_limits_fail_before_backend_allocation(monkeypatch) -> None:
    stage = _stage()
    _triangle_mesh(stage, "/Mesh")
    scene = build_scene_analysis_ir(stage)
    imported = False

    def unexpected_import():
        nonlocal imported
        imported = True
        raise AssertionError("Newton import must follow deterministic preflight")

    monkeypatch.setattr(
        "usd_core.camera_analysis.newton_backend._lazy_imports",
        unexpected_import,
    )

    class TinyBackend(NewtonVisibilityBackend):
        max_mesh_vertices = 2

    with pytest.raises(ValueError, match="mesh vertices"):
        TinyBackend(scene)
    assert imported is False

    class UncalledBackend:
        versions = {}

        def evaluate_rays(self, *_args, **_kwargs):
            raise AssertionError("surface sampling must follow workload preflight")

    with pytest.raises(ValueError, match="candidate evaluation.*rays"):
        place_max_coverage(
            scene,
            UncalledBackend(),
            (np.zeros(3), np.asarray([10.0, 10.0, 1.0])),
            scope_path="/Mesh",
            grid=100,
            candidate_count=512,
        )

    class OversizedInput:
        def __len__(self) -> int:
            return 4

        def __array__(self):
            raise AssertionError("dtype conversion must follow the ray-count guard")

    uninitialized = object.__new__(NewtonVisibilityBackend)
    uninitialized.max_rays = 3
    with pytest.raises(ValueError, match="4 rays; limit is 3"):
        uninitialized.evaluate_rays(OversizedInput(), OversizedInput())


def test_raw_scene_limits_precede_mesh_materialization_digest_and_newton(
    monkeypatch,
) -> None:
    stage = _stage()
    _triangle_mesh(stage, "/Mesh")
    touched = {"materialized": False, "digest": False, "newton": False}

    class OversizedPoints:
        def __len__(self) -> int:
            return scene_module.MAX_ANALYSIS_VERTICES + 1

        def __iter__(self):
            raise AssertionError("oversized points must not be materialized")

    monkeypatch.setattr(
        scene_module,
        "_raw_mesh_data",
        lambda _mesh: scene_module._RawMeshData(
            points=OversizedPoints(),
            face_counts=(3,),
            face_indices=(0, 1, 2),
            holes=(),
        ),
    )

    def unexpected_materialization(*_args, **_kwargs):
        touched["materialized"] = True
        raise AssertionError("raw cap must precede mesh materialization")

    def unexpected_digest(*_args, **_kwargs):
        touched["digest"] = True
        raise AssertionError("raw cap must precede stage flattening")

    def unexpected_newton_import():
        touched["newton"] = True
        raise AssertionError("raw cap must precede Newton import")

    monkeypatch.setattr(scene_module, "_mesh_resource", unexpected_materialization)
    monkeypatch.setattr(scene_module, "_source_digest", unexpected_digest)
    monkeypatch.setattr(
        "usd_core.camera_analysis.newton_backend._lazy_imports",
        unexpected_newton_import,
    )

    with pytest.raises(ValueError, match="raw mesh points.*limit"):
        build_scene_analysis_ir(stage)
    assert touched == {"materialized": False, "digest": False, "newton": False}


def test_aperture_offsets_shift_the_analytic_frustum() -> None:
    camera = CameraPose(
        prim_path=None,
        position_m=(0.0, 0.0, 0.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, 1.0),
        focal_length_mm=1.0,
        horizontal_aperture_mm=2.0,
        vertical_aperture_mm=2.0,
        horizontal_aperture_offset_mm=1.0,
    )
    mask, _, _ = frustum_mask(
        camera,
        np.asarray([[-1.0, 0.0, 1.0], [0.5, 0.0, 1.0]]),
    )
    assert mask.tolist() == [False, True]


def test_max_coverage_candidates_clear_the_requested_xy_standoff() -> None:
    x_values = np.linspace(-3.0, 3.0, 4)
    y_values = np.linspace(-0.5, 0.5, 2)
    xx, yy = np.meshgrid(x_values, y_values)
    points = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
    surface = GridSurface(
        rows=2,
        columns=4,
        x_min_m=-4.0,
        x_max_m=4.0,
        y_min_m=-1.0,
        y_max_m=1.0,
        points_m=points,
        accessible=np.ones((2, 4), dtype=bool),
    )
    standoff = 3.0
    candidates, _target = _coverage_candidates(
        surface,
        candidate_count=32,
        height_m=2.0,
        standoff_m=standoff,
        seed=5,
        focal_length_mm=18.0,
        aperture_mm=36.0,
    )

    for candidate in candidates:
        x, y, _z = candidate.position_m
        dx = max(surface.x_min_m - x, 0.0, x - surface.x_max_m)
        dy = max(surface.y_min_m - y, 0.0, y - surface.y_max_m)
        assert math.hypot(dx, dy) >= standoff


def test_candidate_clearance_rejects_a_pose_inside_supported_geometry() -> None:
    def pose(position: tuple[float, float, float]) -> CameraPose:
        return CameraPose(
            prim_path=None,
            position_m=position,
            right=(1.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            forward=(0.0, 1.0, 0.0),
            focal_length_mm=35.0,
            horizontal_aperture_mm=36.0,
            vertical_aperture_mm=36.0,
        )

    class ProbeBackend:
        def evaluate_rays(self, origins, directions, *, include_normals=False):
            assert include_normals is True
            count = len(origins)
            distances = np.full(count, -1.0)
            shape_ids = np.full(count, -1, dtype=np.int64)
            normals = np.zeros((count, 3))
            # The first pose has opposed outward-facing exit hits on one closed
            # shape. The second pose misses in every direction.
            distances[:6] = 1.0
            shape_ids[:6] = 7
            normals[:6] = np.asarray(directions[:6])
            return RayHits(distances, shape_ids, normals)

    keep = _geometry_clearance_mask(
        [pose((0.0, 0.0, 0.0)), pose((10.0, 0.0, 0.0))], ProbeBackend()
    )

    assert keep.tolist() == [False, True]


def test_typed_look_at_facade_forwards_one_immutable_config(monkeypatch) -> None:
    expected_report = {"method": "look_at", "passed": True}
    forwarded = {}

    def fake_placement(scene, backend, bounds_m, **kwargs):
        forwarded.update(kwargs)
        assert scene == "scene"
        assert backend == "backend"
        assert bounds_m[0].tolist() == [0.0, 0.0, 0.0]
        return SimpleNamespace(
            report=expected_report,
            poses=(),
            masks=None,
        )

    monkeypatch.setattr(look_at, "_place_cameras_look_at", fake_placement)
    config = LookAtConfig(
        target_path="/World/Target",
        camera_count=3,
        yaw_ranges="10,20",
        occlusion_threshold=0.25,
        min_height_m=1.0,
        max_height_m=2.0,
        min_look_down_deg=10.0,
        max_look_down_deg=45.0,
        xy_bounds_m=((-3.0, 3.0), (-4.0, 4.0)),
        seed=7,
    )
    result = look_at.place_cameras_look_at(
        "scene",  # type: ignore[arg-type]
        "backend",  # type: ignore[arg-type]
        (np.zeros(3), np.ones(3)),
        config=config,
    )

    assert isinstance(result, LookAtResult)
    assert result.report is expected_report
    assert forwarded["target_path"] == "/World/Target"
    assert forwarded["camera_count"] == 3
    assert forwarded["yaw_ranges"] == "10,20"
    assert forwarded["occlusion_threshold"] == 0.25
    assert forwarded["min_height_m"] == 1.0
    assert forwarded["max_height_m"] == 2.0
    assert forwarded["min_look_down_deg"] == 10.0
    assert forwarded["max_look_down_deg"] == 45.0
    assert forwarded["xy_bounds_m"] == ((-3.0, 3.0), (-4.0, 4.0))
    assert forwarded["seed"] == 7


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_public_sensor_tiled_camera_observation_on_cpu(monkeypatch) -> None:
    stage = _stage()
    UsdGeom.Cube.Define(stage, "/Box")
    scene = build_scene_analysis_ir(stage)
    backend = NewtonVisibilityBackend(scene, device="cpu")
    assert isinstance(backend, VisibilityBackend)
    pose = CameraPose(
        prim_path=None,
        position_m=(0.0, 0.0, 3.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, -1.0),
        focal_length_mm=20.0,
        horizontal_aperture_mm=36.0,
        vertical_aperture_mm=36.0,
        clipping_range_m=(0.01, 10.0),
    )
    observation = backend.render_cameras(
        CameraBatchRequest(
            cameras=(pose,),
            width=5,
            height=5,
            include_normals=True,
            include_albedo=True,
        )
    )
    assert observation.depth_m is not None
    assert observation.shape_ids is not None
    assert observation.normals is not None
    assert observation.albedo_rgba is not None
    assert observation.depth_m.shape == (1, 5, 5)
    assert observation.depth_m[0, 2, 2] == pytest.approx(2.0, abs=1.0e-5)
    assert observation.shape_ids[0, 2, 2] == 0
    assert observation.normals[0, 2, 2] == pytest.approx([0.0, 0.0, 1.0])
    assert observation.albedo_rgba.shape == (1, 5, 5, 4)
    assert observation.albedo_rgba.dtype == np.uint8
    assert np.any(observation.albedo_rgba[0, 2, 2] != 0)
    assert observation.depth_m[0, 0, 0] == -1.0
    assert observation.shape_ids[0, 0, 0] == -1
    assert observation.albedo_rgba[0, 0, 0].tolist() == [0, 0, 0, 0]

    sensor = backend._sensors[True]
    original_update = sensor.update
    render_configs = []

    def recording_update(*args, **kwargs):
        render_configs.append(kwargs["render_config"])
        return original_update(*args, **kwargs)

    monkeypatch.setattr(sensor, "update", recording_update)
    albedo_only = backend.render_cameras(
        CameraBatchRequest(
            cameras=(pose,),
            width=3,
            height=3,
            include_depth=False,
            include_shape_ids=False,
            include_albedo=True,
        )
    )
    assert render_configs[0].enable_textures is True
    assert render_configs[0].output_color_space == backend.newton.utils.ColorSpace.SRGB
    assert albedo_only.depth_m is None
    assert albedo_only.shape_ids is None
    assert albedo_only.normals is None
    assert albedo_only.albedo_rgba is not None

    clipped = backend.render_cameras(
        CameraBatchRequest(
            cameras=(replace(pose, clipping_range_m=(0.01, 1.5)), pose),
            width=1,
            height=1,
            include_depth=False,
            include_normals=True,
        )
    )
    assert clipped.depth_m is None
    assert clipped.shape_ids is not None
    assert clipped.shape_ids[:, 0, 0].tolist() == [-1, 0]
    assert clipped.normals is not None
    assert clipped.normals[0, 0, 0].tolist() == [0.0, 0.0, 0.0]
    assert clipped.normals[1, 0, 0].tolist() == pytest.approx([0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="include_depth must be boolean"):
        backend.render_cameras(
            CameraBatchRequest(
                cameras=(pose,),
                width=1,
                height=1,
                include_depth="yes",  # type: ignore[arg-type]
                include_shape_ids=False,
            )
        )


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_newton_cpu_cuda_ray_and_sensor_parity() -> None:
    wp = pytest.importorskip("warp")
    if wp.get_cuda_device_count() < 1:
        pytest.skip("CUDA device is unavailable")

    stage = _stage()
    UsdGeom.Cube.Define(stage, "/Box")
    scene = build_scene_analysis_ir(stage)
    cpu = NewtonVisibilityBackend(scene, device="cpu")
    cuda = NewtonVisibilityBackend(scene, device="cuda:0")
    origins = np.asarray(
        [[0.0, 0.0, 3.0], [3.0, 0.0, 3.0], [0.5, 0.5, 3.0]],
        dtype=np.float32,
    )
    directions = np.repeat(
        np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32), len(origins), axis=0
    )
    cpu_hits = cpu.evaluate_rays(origins, directions, include_normals=True)
    cuda_hits = cuda.evaluate_rays(origins, directions, include_normals=True)
    np.testing.assert_array_equal(cuda_hits.shape_ids, cpu_hits.shape_ids)
    np.testing.assert_allclose(cuda_hits.distances_m, cpu_hits.distances_m, atol=1.0e-5)
    np.testing.assert_allclose(cuda_hits.normals, cpu_hits.normals, atol=1.0e-5)

    pose = CameraPose(
        prim_path=None,
        position_m=(0.0, 0.0, 3.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, -1.0),
        focal_length_mm=20.0,
        horizontal_aperture_mm=36.0,
        vertical_aperture_mm=36.0,
        clipping_range_m=(0.01, 10.0),
    )
    request = CameraBatchRequest(cameras=(pose,), width=5, height=5)
    cpu_observation = cpu.render_cameras(request)
    cuda_observation = cuda.render_cameras(request)
    np.testing.assert_array_equal(cuda_observation.shape_ids, cpu_observation.shape_ids)
    np.testing.assert_allclose(
        cuda_observation.depth_m, cpu_observation.depth_m, atol=1.0e-5
    )


def test_newton_ovrtx_parity_treats_nonpositive_depth_as_background(tmp_path) -> None:
    ovrtx_depth = np.asarray(
        [[[0.0], [2.0]], [[-1.0], [1.0e30]]],
        dtype=np.float32,
    )
    ovrtx_semantic = np.zeros((2, 2, 1), dtype=np.uint32)
    depth_path = tmp_path / "depth.npy"
    semantic_path = tmp_path / "semantic.npy"
    np.save(depth_path, ovrtx_depth, allow_pickle=False)
    np.save(semantic_path, ovrtx_semantic, allow_pickle=False)
    observation = CameraObservation(
        depth_m=np.asarray([[[-1.0, 2.0], [-1.0, -1.0]]], dtype=np.float32),
        shape_ids=np.asarray([[[-1, 0], [-1, -1]]], dtype=np.int32),
    )

    class Scene:
        shape_path_by_id = {0: "/Box"}

    parity = _newton_ovrtx_parity(
        "/Camera",
        analytic_observation=observation,
        scene=Scene(),
        aovs={"SemanticSegmentation": {"statistics": {"semantic_labels": []}}},
        artifact_map={
            "DistanceToCameraSD": str(depth_path),
            "SemanticSegmentation": str(semantic_path),
        },
        semantic_labels={},
    )
    assert parity["passed"] is True
    assert parity["depth"]["ovrtx_valid_pixels"] == 1
    assert parity["depth"]["newton_valid_pixels"] == 1
    assert parity["depth"]["valid_mask_iou"] == 1.0
    assert parity["depth"]["max_error_m"] == 0.0


def test_floor_semantic_parity_binds_the_exact_accessible_shape_path(tmp_path) -> None:
    depth_path = tmp_path / "floor-depth.npy"
    semantic_path = tmp_path / "floor-semantic.npy"
    np.save(
        depth_path,
        np.ones((1, 2, 1), dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        semantic_path,
        np.asarray([[[7], [0]]], dtype=np.uint32),
        allow_pickle=False,
    )
    floor_path = "/World/Floor"
    floor_label = semantic_label(floor_path, "floor")

    parity = _newton_ovrtx_parity(
        "/World/Rig/Camera_001",
        analytic_observation=CameraObservation(
            depth_m=np.ones((1, 1, 2), dtype=np.float32),
            shape_ids=np.asarray([[[0, 1]]], dtype=np.int32),
        ),
        scene=SimpleNamespace(shape_path_by_id={0: floor_path, 1: "/World/Shelf"}),
        aovs={
            "SemanticSegmentation": {
                "statistics": {
                    "semantic_labels": [
                        {
                            "id": 7,
                            "label": f"usd_cli: {floor_label};",
                            "pixels": 1,
                        }
                    ]
                }
            }
        },
        artifact_map={
            "DistanceToCameraSD": str(depth_path),
            "SemanticSegmentation": str(semantic_path),
        },
        semantic_labels={floor_path: floor_label},
    )

    assert parity["semantics"]["newton_target_pixels"] == 1
    assert parity["semantics"]["mask_iou"] == pytest.approx(1.0)
    assert parity["passed"] is True


def test_camera_offsets_round_trip_into_camera_pose() -> None:
    stage = _stage(meters_per_unit=0.01)
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.CreateHorizontalApertureOffsetAttr(1.25)
    camera.CreateVerticalApertureOffsetAttr(-0.75)
    pose = camera_pose(build_scene_analysis_ir(stage).cameras[0])
    assert pose.horizontal_aperture_offset_mm == pytest.approx(1.25)
    assert pose.vertical_aperture_offset_mm == pytest.approx(-0.75)


def test_camera_pose_rejects_retained_orthographic_camera_with_typed_error() -> None:
    stage = _stage()
    camera = UsdGeom.Camera.Define(stage, "/SurveyOrtho")
    camera.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
    camera_ir = build_scene_analysis_ir(stage).cameras[0]

    assert camera_ir.projection == "orthographic"
    with pytest.raises(ValueError, match="unsupported_camera_projection"):
        camera_pose(camera_ir)


@pytest.mark.parametrize("focal_length", [0.0, -1.0, math.nan, math.inf])
def test_camera_pose_rejects_nonpositive_or_nonfinite_focal_length(
    focal_length: float,
) -> None:
    with pytest.raises(ValueError, match="focal length must be finite and positive"):
        CameraPose(
            prim_path=None,
            position_m=(0.0, 0.0, 3.0),
            right=(1.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            forward=(0.0, 0.0, -1.0),
            focal_length_mm=focal_length,
            horizontal_aperture_mm=36.0,
            vertical_aperture_mm=24.0,
        )


def test_rig_authoring_rejects_instance_proxy_target_before_editing() -> None:
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World")
    stage.CreateClassPrim("/_RigPrototype")
    UsdGeom.Xform.Define(stage, "/_RigPrototype/Rig")
    instance = stage.DefinePrim("/World/Instance", "Xform")
    instance.GetReferences().AddInternalReference("/_RigPrototype")
    instance.SetInstanceable(True)
    proxy_path = "/World/Instance/Rig"
    assert stage.GetPrimAtPath(proxy_path).IsInstanceProxy()
    scene = build_scene_analysis_ir(stage)
    pose = CameraPose(
        prim_path=None,
        position_m=(0.0, 0.0, 3.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, -1.0),
        focal_length_mm=35.0,
        horizontal_aperture_mm=36.0,
        vertical_aperture_mm=24.0,
    )
    report = {
        "method": "look_at",
        "source_digest": scene.source_digest,
        "cameras": [{"id": "camera-001", "look_at": [0.0, 0.0, 0.0]}],
    }
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(ValueError, match="instance proxy and is read-only"):
        author_camera_rig(
            stage,
            scene,
            (pose,),
            report,
            author_under=proxy_path,
            on_existing="replace",
        )

    assert stage.GetRootLayer().ExportToString() == before


def test_rig_authoring_disables_regular_instance_ancestor() -> None:
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World")
    stage.CreateClassPrim("/_RigPrototype")
    instance = stage.DefinePrim("/World/Instance", "Xform")
    instance.GetReferences().AddInternalReference("/_RigPrototype")
    instance.SetInstanceable(True)
    scene = build_scene_analysis_ir(stage)
    pose = CameraPose(
        prim_path=None,
        position_m=(0.0, 0.0, 3.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, -1.0),
        focal_length_mm=35.0,
        horizontal_aperture_mm=36.0,
        vertical_aperture_mm=24.0,
    )
    report = {
        "method": "look_at",
        "source_digest": scene.source_digest,
        "cameras": [{"id": "camera-001", "look_at": [0.0, 0.0, 0.0]}],
    }

    authored = author_camera_rig(
        stage,
        scene,
        (pose,),
        report,
        author_under="/World/Instance/Rig",
    )

    assert not stage.GetPrimAtPath("/World/Instance").IsInstance()
    assert stage.GetPrimAtPath(authored["camera_paths"][0]).IsA(UsdGeom.Camera)


def test_rig_lens_units_round_trip_identically_on_meter_and_centimeter_stages() -> None:
    def authored_camera(meters_per_unit: float):
        stage = _stage(meters_per_unit=meters_per_unit)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Cube.Define(stage, "/World/Target")
        scene = build_scene_analysis_ir(stage)
        pose = CameraPose(
            prim_path=None,
            position_m=(0.0, 0.0, 3.0),
            right=(1.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            forward=(0.0, 0.0, -1.0),
            focal_length_mm=35.0,
            horizontal_aperture_mm=36.0,
            vertical_aperture_mm=24.0,
            clipping_range_m=(0.1, 100.0),
            horizontal_aperture_offset_mm=1.2,
            vertical_aperture_offset_mm=-0.6,
        )
        report = {
            "method": "look_at",
            "source_digest": scene.source_digest,
            "seed": 0,
            "target_path": "/World/Target",
            "cameras": [{"id": "camera-001", "look_at": [0.0, 0.0, 0.0]}],
        }
        authored = author_camera_rig(
            stage,
            scene,
            (pose,),
            report,
            author_under="/World/Rig",
        )
        path = authored["camera_paths"][0]
        camera = UsdGeom.Camera(stage.GetPrimAtPath(path))
        rebuilt_scene = build_scene_analysis_ir(stage)
        extracted = next(
            item for item in rebuilt_scene.cameras if item.prim_path == path
        )
        calibration = camera_calibration(
            stage,
            rebuilt_scene,
            path,
            resolution=(640, 480),
        )
        return stage, camera, extracted, calibration

    _meter_stage, meter_camera, meter_ir, meter_export = authored_camera(1.0)
    _centimeter_stage, centimeter_camera, centimeter_ir, centimeter_export = (
        authored_camera(0.01)
    )

    assert meter_camera.GetFocalLengthAttr().Get() == pytest.approx(0.35)
    assert centimeter_camera.GetFocalLengthAttr().Get() == pytest.approx(35.0)
    assert meter_camera.GetHorizontalApertureAttr().Get() == pytest.approx(0.36)
    assert centimeter_camera.GetHorizontalApertureAttr().Get() == pytest.approx(36.0)
    for camera_ir in (meter_ir, centimeter_ir):
        assert camera_ir.focal_length_mm == pytest.approx(35.0)
        assert camera_ir.horizontal_aperture_mm == pytest.approx(36.0)
        assert camera_ir.vertical_aperture_mm == pytest.approx(24.0)
        assert camera_ir.horizontal_aperture_offset_mm == pytest.approx(1.2)
        assert camera_ir.vertical_aperture_offset_mm == pytest.approx(-0.6)
    assert meter_export["intrinsics"]["focal_length_m"] == pytest.approx(0.035)
    assert centimeter_export["intrinsics"]["focal_length_m"] == pytest.approx(0.035)
    np.testing.assert_allclose(
        meter_export["intrinsics"]["K"],
        centimeter_export["intrinsics"]["K"],
    )

    def camera_create_values(meters_per_unit: float):
        from usd_core.session import Session

        session = Session()
        session._stage = _stage(meters_per_unit=meters_per_unit)
        world = UsdGeom.Xform.Define(session._stage, "/World")
        session._stage.SetDefaultPrim(world.GetPrim())
        session._index_prims()
        response = session.camera_create(name="Lens", focal=35.0, aperture=36.0)
        assert response.ok
        camera = UsdGeom.Camera(
            session._stage.GetPrimAtPath(response.summary["camera"])
        )
        extracted = next(
            item
            for item in build_scene_analysis_ir(session._stage).cameras
            if item.prim_path == response.summary["camera"]
        )
        return (
            float(camera.GetFocalLengthAttr().Get()),
            float(camera.GetHorizontalApertureAttr().Get()),
            extracted,
        )

    meter_raw_focal, meter_raw_aperture, meter_created_ir = camera_create_values(1.0)
    cm_raw_focal, cm_raw_aperture, cm_created_ir = camera_create_values(0.01)
    assert meter_raw_focal == pytest.approx(0.35)
    assert meter_raw_aperture == pytest.approx(0.36)
    assert cm_raw_focal == pytest.approx(35.0)
    assert cm_raw_aperture == pytest.approx(36.0)
    for camera_ir in (meter_created_ir, cm_created_ir):
        assert camera_ir.focal_length_mm == pytest.approx(35.0)
        assert camera_ir.horizontal_aperture_mm == pytest.approx(36.0)
