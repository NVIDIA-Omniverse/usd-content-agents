# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage-aware camera rig regression tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

from usd_core.camera import author_camera
from usd_core.camera_analysis.authoring import author_camera_rig
from usd_core.camera_analysis.contracts import CameraPose
from usd_core.camera_analysis.coverage import GridSurface, visibility_regions
from usd_core.camera_analysis.newton_backend import NewtonVisibilityBackend
from usd_core.camera_analysis.scene import (
    build_scene_analysis_ir,
    camera_pose,
    canonical_bounds,
    is_path_at_or_below,
)
from usd_core.session import Session
from usd_core.spatial import get_bbox


def _scene(path: Path, *, meters_per_unit: float = 1.0, up_axis: str = "Z") -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.SetStageUpAxis(
        stage, UsdGeom.Tokens.y if up_axis == "Y" else UsdGeom.Tokens.z
    )
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.GetSizeAttr().Set(2.0 / meters_per_unit)
    floor_xform = UsdGeom.Xformable(floor)
    if up_axis == "Y":
        floor_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -1.0 / meters_per_unit, 0.0))
        floor_xform.AddScaleOp().Set(Gf.Vec3f(5.0, 0.05, 5.0))
        camera_position = (0.0, 10.0 / meters_per_unit, 0.0)
    else:
        floor_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -1.0 / meters_per_unit))
        floor_xform.AddScaleOp().Set(Gf.Vec3f(5.0, 5.0, 0.05))
        camera_position = (0.0, 0.0, 10.0 / meters_per_unit)
    target = UsdGeom.Cube.Define(stage, "/World/Target")
    target.GetSizeAttr().Set(1.0 / meters_per_unit)
    if up_axis == "Y":
        target.AddTranslateOp().Set(Gf.Vec3d(0.0, 1.0 / meters_per_unit, 0.0))
    else:
        target.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0 / meters_per_unit))
    author_camera(
        stage,
        "/World/Existing",
        camera_position,
        (0.0, 0.0, 0.0),
        focal=18.0,
        h_ap=36.0,
        v_ap=36.0,
    )
    stage.GetRootLayer().Save()
    return path


def test_scene_ir_normalizes_z_up_meters_and_y_up_centimeters(tmp_path):
    z_stage = Usd.Stage.Open(str(_scene(tmp_path / "z.usda")))
    y_stage = Usd.Stage.Open(
        str(_scene(tmp_path / "y.usda", meters_per_unit=0.01, up_axis="Y"))
    )
    z_ir = build_scene_analysis_ir(z_stage)
    y_ir = build_scene_analysis_ir(y_stage)
    z_floor = next(shape for shape in z_ir.shapes if shape.prim_path == "/World/Floor")
    y_floor = next(shape for shape in y_ir.shapes if shape.prim_path == "/World/Floor")
    assert z_ir.canonical_up_axis == y_ir.canonical_up_axis == "Z"
    assert z_floor.parameters == y_floor.parameters
    z_bounds = canonical_bounds(get_bbox(z_stage, "/World/Floor"), z_ir)
    y_bounds = canonical_bounds(get_bbox(y_stage, "/World/Floor"), y_ir)
    assert np.allclose(z_bounds[0], y_bounds[0], atol=1e-6)
    assert np.allclose(z_bounds[1], y_bounds[1], atol=1e-6)
    assert np.allclose(
        camera_pose(
            next(
                camera
                for camera in z_ir.cameras
                if camera.prim_path.endswith("Existing")
            )
        ).position_m,
        camera_pose(
            next(
                camera
                for camera in y_ir.cameras
                if camera.prim_path.endswith("Existing")
            )
        ).position_m,
    )


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_newton_backend_hits_analytic_shape_with_stable_identity(tmp_path):
    stage = Usd.Stage.Open(str(_scene(tmp_path / "scene.usda")))
    scene = build_scene_analysis_ir(stage)
    backend = NewtonVisibilityBackend(scene)
    hits = backend.evaluate_rays(
        np.asarray([[0.0, 0.0, 5.0], [20.0, 20.0, 5.0]]),
        np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
    )
    assert hits.distances_m[0] > 0.0
    assert scene.shape_path_by_id[int(hits.shape_ids[0])] in {
        "/World/Floor",
        "/World/Target",
    }
    assert hits.distances_m[1] < 0.0
    assert hits.shape_ids[1] == -1


def test_visibility_regions_preserve_hole_and_area():
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    points = np.zeros((5, 5, 3), dtype=np.float64)
    grid = GridSurface(5, 5, 0.0, 5.0, 0.0, 5.0, points, np.ones((5, 5), dtype=bool))
    regions = visibility_regions(mask, grid)
    assert len(regions) == 1
    assert len(regions[0]["holes"]) == 1
    assert regions[0]["area_m2"] == pytest.approx(24.0)
    assert regions[0]["outline"][0] == regions[0]["outline"][-1]


def test_exact_target_descendant_matching():
    assert is_path_at_or_below("/World/Shelf/Part", "/World/Shelf")
    assert not is_path_at_or_below("/World/Shelf2/Part", "/World/Shelf")


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_coverage_skips_unselected_orthographic_camera_and_rejects_explicit_use(
    tmp_path,
):
    session = Session.open(_scene(tmp_path / "orthographic.usda"))
    unrelated = UsdGeom.Camera.Define(session._stage, "/World/UnrelatedOrtho")
    unrelated.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
    scene = build_scene_analysis_ir(session._stage)
    projections = {camera.prim_path: camera.projection for camera in scene.cameras}
    assert projections == {
        "/World/Existing": "perspective",
        "/World/UnrelatedOrtho": "orthographic",
    }

    implicit = session.camera_coverage(
        scope="/World/Floor",
        target=0.1,
        grid=4,
    )
    assert implicit.ok
    assert [item["camera"] for item in implicit.data["cameras"]] == ["/World/Existing"]

    explicit = session.camera_coverage(
        scope="/World/Floor",
        cameras=["/World/UnrelatedOrtho"],
        target=0.1,
        grid=4,
    )
    assert not explicit.ok
    assert "unsupported_camera_projection" in explicit.issues[0].message


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_session_coverage_preview_author_undo_redo_and_export(tmp_path):
    scene_path = _scene(tmp_path / "scene.usda")
    session = Session.open(scene_path)
    unrelated = UsdGeom.Camera.Define(session._stage, "/World/UnrelatedOrtho")
    unrelated.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
    active_before = session._active_cam
    lower_slab = UsdGeom.Cube.Define(session._stage, "/World/LowerSiblingSlab")
    lower_slab.AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.1))
    lower_slab.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -2.0))

    coverage = session.camera_coverage(
        scope="/World/Floor",
        cameras=["/World/Existing"],
        target=0.5,
        grid=8,
    )
    assert coverage.ok
    assert coverage.data["schema"] == "usd-cli.camera-coverage.v1"
    assert coverage.data["analysis_policy"]["floor_paths"] == ["/World/Floor"]
    assert coverage.data["analysis_policy_digest"].startswith("sha256:")
    assert coverage.data["grid"]["surface_planar"] is True
    assert (
        coverage.data["grid"]["surface_max_deviation_m"]
        <= coverage.data["grid"]["surface_planarity_tolerance_m"]
    )
    assert coverage.summary["passed"] is True
    assert session._active_cam == active_before

    preview = session.camera_place(
        method="max_coverage",
        scope="/World/Floor",
        target_coverage=0.5,
        max_cameras=2,
        candidates=8,
        grid=8,
        seed=17,
    )
    assert preview.ok
    assert preview.summary["preview"] is True
    assert not session._stage.GetPrimAtPath("/World/Rig").IsValid()
    repeated_preview = session.camera_place(
        method="max_coverage",
        scope="/World/Floor",
        target_coverage=0.5,
        max_cameras=2,
        candidates=8,
        grid=8,
        seed=17,
    )
    assert repeated_preview.ok
    assert repeated_preview.data["cameras"] == preview.data["cameras"]
    assert repeated_preview.data["achieved_coverage"] == pytest.approx(
        preview.data["achieved_coverage"]
    )

    viewer_revision_before = session._viewer_revision
    session._viewer_line_geometry_publications["stale"] = {"digest": "old"}
    authored = session.camera_place(
        method="max_coverage",
        scope="/World/Floor",
        target_coverage=0.5,
        max_cameras=2,
        candidates=8,
        grid=8,
        seed=17,
        author_under="/World/Rig",
    )
    assert authored.ok
    assert authored.summary["preview"] is False
    assert session._stage.GetPrimAtPath("/World/Rig").IsValid()
    assert session._active_cam == active_before
    assert len(session.history.entries()) == 1
    assert session._viewer_revision == viewer_revision_before + 1
    assert session._viewer_line_geometry_publications == {}
    rig_prim = session._stage.GetPrimAtPath("/World/Rig")
    persisted_policy = json.loads(
        rig_prim.GetCustomDataByKey("usdCameraRigAnalysisPolicyJson")
    )
    assert persisted_policy == authored.data["analysis_policy"]
    assert (
        rig_prim.GetCustomDataByKey("usdCameraRigAnalysisPolicyDigest")
        == authored.data["analysis_policy_digest"]
    )

    rig_audit = session.camera_coverage(
        scope="/World/Floor",
        cameras=["/World/Rig"],
        target=0.5,
        grid=8,
    )
    assert rig_audit.ok
    assert len(rig_audit.data["cameras"]) == authored.summary["cameras"]
    assert rig_audit.data["coverage_fraction"] == pytest.approx(
        authored.data["achieved_coverage"], abs=1.0e-12
    )

    undone = session.undo()
    assert undone.ok
    rig_after_undo = session._stage.GetPrimAtPath("/World/Rig")
    assert not rig_after_undo.IsValid() or not rig_after_undo.IsActive()
    redone = session.redo()
    assert redone.ok
    assert session._stage.GetPrimAtPath("/World/Rig").IsValid()

    destination = tmp_path / "rig.json"
    exported = session.camera_rig_export(
        "/World/Rig",
        res=[640, 480],
        include_visibility=True,
        grid=8,
        output=str(destination),
    )
    assert exported.ok
    document = json.loads(destination.read_text())
    assert document["schema"] == "usd-cli.camera-rig.v1"
    assert document["source"]["canonical_up_axis"] == "Z"
    assert len(document["cameras"]) == authored.summary["cameras"]
    assert document["source"]["analysis_policy"] == authored.data["analysis_policy"]
    assert (
        document["source"]["analysis_policy_digest"]
        == authored.data["analysis_policy_digest"]
    )
    assert (
        document["coverage"]["analysis_policy_digest"]
        == authored.data["analysis_policy_digest"]
    )
    assert document["coverage"]["coverage_fraction"] == pytest.approx(
        authored.data["achieved_coverage"], abs=1.0e-12
    )
    camera = document["cameras"][0]
    assert camera["calibration"]["eligible"] is True
    assert camera["calibration"]["projection_verification"]["passed"] is True

    saved_path = tmp_path / "scene_with_rig.usda"
    saved = session.save(str(saved_path))
    assert saved.ok
    reopened = Usd.Stage.Open(str(saved_path))
    assert reopened.GetPrimAtPath("/World/Rig").IsValid()
    reopened_cameras = [
        prim.GetPath().pathString
        for prim in reopened.Traverse()
        if prim.IsA(UsdGeom.Camera)
        and prim.GetPath().pathString.startswith("/World/Rig/")
    ]
    assert len(reopened_cameras) == authored.summary["cameras"]
    assert (
        reopened.GetPrimAtPath("/World/Rig").GetCustomDataByKey("usdCameraRigMethod")
        == "max_coverage"
    )


def test_authored_world_poses_survive_a_transformed_parent_scope(tmp_path):
    stage = Usd.Stage.Open(str(_scene(tmp_path / "scene.usda")))
    parent = UsdGeom.Xform.Define(stage, "/World/Translated")
    parent.AddTranslateOp().Set(Gf.Vec3d(100.0, -25.0, 7.0))
    scene = build_scene_analysis_ir(stage)
    pose = CameraPose(
        prim_path=None,
        position_m=(2.0, 3.0, 4.0),
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
        "seed": 3,
        "target_path": "/World/Target",
        "cameras": [{"id": "camera-001", "look_at": [2.0, 3.0, 0.0]}],
    }

    authored = author_camera_rig(
        stage,
        scene,
        (pose,),
        report,
        author_under="/World/Translated/Rig",
    )

    rig = UsdGeom.Xform(stage.GetPrimAtPath(authored["rig_path"]))
    assert rig.GetResetXformStack() is True
    camera_prim = stage.GetPrimAtPath(authored["camera_paths"][0])
    world = UsdGeom.XformCache().GetLocalToWorldTransform(camera_prim)
    assert tuple(world.ExtractTranslation()) == pytest.approx(pose.position_m)


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_look_at_is_deterministic_and_n_or_nothing(tmp_path):
    session = Session.open(_scene(tmp_path / "scene.usda"))
    first = session.camera_place(
        method="look_at",
        target="/World/Target",
        cameras=3,
        candidates=18,
        yaw_ranges="0,360",
        occlusion_threshold=0.9,
        height=0.0,
        seed=11,
    )
    second = session.camera_place(
        method="look_at",
        target="/World/Target",
        cameras=3,
        candidates=18,
        yaw_ranges="0,360",
        occlusion_threshold=0.9,
        height=0.0,
        seed=11,
    )
    assert first.ok and second.ok
    assert first.data["cameras"] == second.data["cameras"]
    assert len(first.data["cameras"]) == 3
    assert all(item["occlusion_fraction"] <= 0.9 for item in first.data["cameras"])

    impossible = session.camera_place(
        method="look_at",
        target="/World/Target",
        cameras=4,
        candidates=4,
        yaw_ranges="0,0",
        occlusion_threshold=0.0,
        height=0.0,
        seed=11,
    )
    assert not impossible.ok
    assert impossible.data["cameras"] == []
    assert impossible.data["stop_reason"] in {
        "insufficient_unoccluded_views",
        "no_valid_candidate",
    }


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_look_at_rig_export_derives_separate_floor_policy(tmp_path):
    session = Session.open(_scene(tmp_path / "scene.usda"))
    authored = session.camera_place(
        method="look_at",
        target="/World/Target",
        cameras=2,
        candidates=16,
        yaw_ranges="0,360",
        occlusion_threshold=0.9,
        height=0.0,
        seed=13,
        author_under="/World/LookAtRig",
    )
    assert authored.ok, authored.issues
    assert authored.data["analysis_policy"]["target_paths"] == ["/World/Target"]
    assert authored.data["analysis_policy"]["floor_paths"] == []

    output = tmp_path / "look-at-rig.json"
    exported = session.camera_rig_export(
        "/World/LookAtRig",
        res=[64, 48],
        include_visibility=True,
        scope="/World/Floor",
        grid=4,
        output=str(output),
    )
    assert exported.ok, exported.issues
    policy = exported.data["document"]["source"]["analysis_policy"]
    assert policy["scope_paths"] == ["/World/Floor"]
    assert policy["floor_paths"] == ["/World/Floor"]
    assert policy["target_paths"] == ["/World/Target"]
    assert exported.data["document"]["coverage"]["analysis_policy"] == policy
    assert (
        exported.data["document"]["coverage"]["analysis_policy_digest"]
        == exported.data["document"]["source"]["analysis_policy_digest"]
    )
    floor_z = exported.data["document"]["coverage"]["grid"]["surface_z_m"]
    assert all(
        camera["homography"]["floor_z_m"] == pytest.approx(floor_z)
        for camera in exported.data["document"]["cameras"]
    )


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_look_at_target_can_also_be_the_visibility_floor(tmp_path):
    session = Session.open(_scene(tmp_path / "scene.usda"))
    authored = session.camera_place(
        method="look_at",
        target="/World/Target",
        cameras=1,
        candidates=12,
        yaw_ranges="0,360",
        occlusion_threshold=0.9,
        height=0.0,
        seed=29,
        author_under="/World/LookAtTargetRig",
    )
    assert authored.ok, authored.issues

    output = tmp_path / "look-at-target-floor.json"
    exported = session.camera_rig_export(
        "/World/LookAtTargetRig",
        res=[64, 48],
        include_visibility=True,
        scope="/World/Target",
        grid=4,
        output=str(output),
    )

    assert exported.ok, exported.issues
    policy = exported.data["document"]["source"]["analysis_policy"]
    assert policy["target_paths"] == ["/World/Target"]
    assert policy["floor_paths"] == ["/World/Target"]
    assert exported.data["document"]["coverage"]["grid"][
        "accessible_surface_paths"
    ] == ["/World/Target"]


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_broad_scope_homography_uses_accessible_floor_not_tall_obstacle(tmp_path):
    session = Session.open(_scene(tmp_path / "scene.usda"))
    destination = tmp_path / "broad-scope-rig.json"

    exported = session.camera_rig_export(
        "/World",
        res=[64, 48],
        include_visibility=True,
        scope="/World",
        grid=8,
        output=str(destination),
    )

    assert exported.ok, exported.issues
    document = exported.data["document"]
    surface = document["coverage"]["grid"]
    assert surface["surface_planar"] is True
    assert surface["accessible_surface_paths"] == ["/World/Floor"]
    # The floor slab top is -0.95 m; /World/Target reaches +1.5 m and must not
    # become the calibration plane merely because it expands the scope bbox.
    assert surface["surface_z_m"] == pytest.approx(-0.95, abs=1.0e-6)
    assert document["cameras"][0]["homography"]["floor_z_m"] == pytest.approx(
        -0.95, abs=1.0e-6
    )


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_placement_envelopes_patch_size_and_true_target_distance(tmp_path):
    session = Session.open(_scene(tmp_path / "scene.usda"))

    coverage = session.camera_place(
        method="max_coverage",
        scope="/World/Floor",
        target_coverage=0.1,
        max_cameras=2,
        candidates=16,
        patch_size=2.0,
        height=3.0,
        min_look_down=10.0,
        max_look_down=40.0,
        seed=19,
    )
    assert coverage.ok
    assert coverage.data["grid"]["rows"] == 5
    assert coverage.data["grid"]["columns"] == 5
    assert coverage.data["config"]["patch_size_m"] == 2.0
    assert coverage.data["config"]["min_look_down_deg"] == 10.0
    assert coverage.data["config"]["max_look_down_deg"] == 40.0
    assert coverage.data["cameras"]
    for camera in coverage.data["cameras"]:
        position = np.asarray(camera["canonical_position_m"])
        target = np.asarray(camera["look_at"])
        delta = position - target
        look_down = np.degrees(np.arctan2(delta[2], np.linalg.norm(delta[:2])))
        assert 10.0 <= look_down <= 40.0

    look_at = session.camera_place(
        method="look_at",
        target="/World/Target",
        cameras=3,
        candidates=36,
        min_distance=2.0,
        max_distance=2.5,
        min_height=1.0,
        max_height=1.5,
        min_look_down=20.0,
        max_look_down=50.0,
        min_x=-3.0,
        max_x=3.0,
        min_y=-3.0,
        max_y=3.0,
        occlusion_threshold=0.9,
        seed=23,
    )
    assert look_at.ok, look_at.issues
    assert look_at.data["config"]["height_reference"] == "target_center_z"
    assert look_at.data["config"]["xy_bounds_m"] == [
        [-3.0, 3.0],
        [-3.0, 3.0],
    ]
    assert len(look_at.data["cameras"]) == 3
    for camera in look_at.data["cameras"]:
        position = np.asarray(camera["canonical_position_m"])
        target = np.asarray(camera["look_at"])
        delta = position - target
        distance = np.linalg.norm(delta)
        look_down = np.degrees(np.arctan2(delta[2], np.linalg.norm(delta[:2])))
        assert 2.0 <= distance <= 2.5
        assert 1.0 <= delta[2] <= 1.5
        assert 20.0 <= look_down <= 50.0
        assert -3.0 <= position[0] <= 3.0
        assert -3.0 <= position[1] <= 3.0


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_look_at_exact_count_never_uses_coincident_top_down_poses(tmp_path):
    session = Session.open(_scene(tmp_path / "scene.usda"))

    response = session.camera_place(
        method="look_at",
        target="/World/Target",
        cameras=2,
        candidates=12,
        min_distance=2.0,
        max_distance=2.0,
        min_height=2.0,
        max_height=2.0,
        min_look_down=90.0,
        max_look_down=90.0,
        occlusion_threshold=1.0,
        seed=29,
    )

    assert not response.ok
    assert response.data["cameras"] == []
    assert response.data["valid_candidate_count"] <= 1
    assert response.data["duplicate_rejected_candidate_count"] == 11
    assert response.data["stop_reason"] in {
        "insufficient_unoccluded_views",
        "no_valid_candidate",
    }


@pytest.mark.usefixtures("qualified_camera_analysis_runtime")
def test_read_only_preview_allowed_but_authoring_blocked(tmp_path):
    session = Session.open(_scene(tmp_path / "scene.usda"), name="reader")
    session.read_only = True
    preview = session.camera_place(
        method="max_coverage",
        scope="/World/Floor",
        target_coverage=0.5,
        candidates=8,
        grid=8,
    )
    assert preview.ok
    authored = session.camera_place(
        method="max_coverage",
        scope="/World/Floor",
        candidates=8,
        grid=8,
        author_under="/World/Rig",
    )
    assert not authored.ok
    assert "read-only" in authored.issues[0].message


def test_cli_import_and_help_do_not_import_newton_or_warp():
    source = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(source), env.get("PYTHONPATH", "")) if item
    )
    code = (
        "import sys; import usd_cli.main; "
        "assert 'newton' not in sys.modules; assert 'warp' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
