# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``physics_agent.tuning.scenarios._scene_builder``.

The scene builder is the parent-side helper that turns a patched
physics USD into a derivative scene USD with a UsdPhysics.Scene,
ground plane, body initial pose, and cameras. It's normally exercised
end-to-end by ``OvPhysXBackend.evaluate``; these tests poke directly
at the corner cases that a full backend run would only hit
intermittently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

pxr = pytest.importorskip("pxr")  # noqa: F841
from pxr import (  # type: ignore[import-untyped]  # noqa: E402
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdShade,
)

from physics_agent.tuning.scenarios._scene_builder import (  # noqa: E402
    build_drop_settle_scene,
    build_freeform_scene,
)


@pytest.fixture
def scene_builder_caplog(
    caplog: pytest.LogCaptureFixture,
) -> pytest.LogCaptureFixture:
    """Wire ``caplog`` directly to the scene_builder module logger.

    pytest's default ``caplog`` installs its handler on the root logger
    and relies on propagation to capture child-logger records. Two
    pathologies have to be defended against simultaneously:

    1. Some earlier test in the broader ``apps/physics_agent/tests``
       suite leaves the ``physics_agent`` logger tree with
       ``propagate=False`` (e.g. by invoking ``setup_agent_logging``).
       In that case the warnings never reach root and
       ``caplog.records`` is empty.
    2. When this file is the ONLY file under test (``propagate=True``
       by default), naively attaching ``caplog.handler`` *also* lets
       the record propagate to root and be captured a second time, so
       a single ``logger.warning`` shows up twice in ``caplog.records``
       and breaks ``len(matching) == 1`` asserts.

    The robust fix is to attach the handler directly AND temporarily
    set ``propagate = False`` for the duration of the test, restoring
    both on teardown. That makes test outcomes independent of suite
    ordering and of side-effects from upstream logger setup.
    """
    target = logging.getLogger("physics_agent.tuning.scenarios._scene_builder")
    prev_propagate = target.propagate
    prev_level = target.level
    try:
        target.addHandler(caplog.handler)
        target.setLevel(logging.WARNING)
        target.propagate = False
        yield caplog
    finally:
        target.removeHandler(caplog.handler)
        target.propagate = prev_propagate
        target.setLevel(prev_level)


def _patched_physics_usd(tmp_path: Path) -> Path:
    """Minimal USD with a single rigid body — stand-in for the
    output of ``patch_physics_usd``."""
    p = tmp_path / "patched.usda"
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    body = UsdGeom.Sphere.Define(stage, "/World/Body")
    body.CreateRadiusAttr(0.25)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr().Set(1.0)
    stage.GetRootLayer().Save()
    return p


def _nested_rigid_body_usd(tmp_path: Path) -> Path:
    """RoboCasa-style topology: wrapper rigid body, nested object rigid
    body, and collision/visible mesh under the nested object."""
    from pxr import Gf, Vt

    p = tmp_path / "nested_rigid_body.usda"
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    wrapper = UsdGeom.Xform.Define(stage, "/World/Body")
    UsdPhysics.RigidBodyAPI.Apply(wrapper.GetPrim())

    obj = UsdGeom.Xform.Define(stage, "/World/Body/Object")
    UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/World/Body/Object/Geom")
    pts = [
        (-0.05, -0.05, 0.0),
        (0.05, -0.05, 0.0),
        (0.05, 0.05, 0.0),
        (-0.05, 0.05, 0.0),
        (-0.05, -0.05, 0.155),
        (0.05, -0.05, 0.155),
        (0.05, 0.05, 0.155),
        (-0.05, 0.05, 0.155),
    ]
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*pt) for pt in pts]))
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            0,
            1,
            5,
            4,
            2,
            3,
            7,
            6,
            0,
            3,
            7,
            4,
            1,
            2,
            6,
            5,
        ]
    )
    mesh.CreateExtentAttr([Gf.Vec3f(-0.05, -0.05, 0.0), Gf.Vec3f(0.05, 0.05, 0.155)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr().Set(1.0)

    stage.GetRootLayer().Save()
    return p


def _patched_physics_usd_with_helper_bbox(tmp_path: Path) -> Path:
    from pxr import Gf

    p = tmp_path / "helper_bbox.usda"
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    body = UsdGeom.Xform.Define(stage, "/World/Body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr().Set(1.0)

    collider = UsdGeom.Cube.Define(stage, "/World/Body/Collider")
    collider.CreateSizeAttr(1.0)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)

    helper = UsdGeom.Cube.Define(stage, "/World/Body/reg_bbox")
    helper.CreateSizeAttr(4.0)
    UsdGeom.Xformable(helper.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -2.0))
    stage.GetRootLayer().Save()
    return p


def test_freeform_scene_uses_first_rigid_body_by_traversal_order(
    tmp_path: Path,
) -> None:
    """Tune assumes the patched USD is a clean single-body asset.

    Nested/multi-body inputs preserve the historical traversal-order
    selection contract instead of trying to infer intent from descendant
    collision topology.
    """
    patched = _nested_rigid_body_usd(tmp_path)
    out = tmp_path / "scene.usda"

    info = build_freeform_scene(
        patched,
        out,
        target={"duration_s": 1.0, "initial_pose": {"position": [0.0, 0.0, 0.155]}},
    )

    assert info["body_prim_path"] == "/World/Body"
    assert info["body_pattern"] == "/World/Body"
    assert info["world_up"] == [0.0, 0.0, 1.0]
    assert info["bbox_min_local_stage"][2] == pytest.approx(0.0, abs=1e-6)
    assert info["bbox_max_local_stage"][2] == pytest.approx(0.155, abs=1e-6)

    rec = Usd.Stage.Open(str(out))
    assert rec.GetPrimAtPath(Sdf.Path("/SceneRoot/GroundPlane")).IsValid()


def test_freeform_scene_records_bbox_after_selected_body_transform(
    tmp_path: Path,
) -> None:
    """Selection stays traversal-order based, while bbox metadata still
    reflects transforms on the selected rigid-body subtree."""
    from pxr import Gf

    patched = _nested_rigid_body_usd(tmp_path)
    stage = Usd.Stage.Open(str(patched))
    body = stage.GetPrimAtPath("/World/Body")
    xformable = UsdGeom.Xformable(body)
    xformable.AddRotateZOp().Set(90.0)
    xformable.AddScaleOp().Set(Gf.Vec3f(2.0, 1.0, 1.0))
    stage.GetRootLayer().Save()
    out = tmp_path / "scene.usda"

    info = build_freeform_scene(
        patched,
        out,
        target={"duration_s": 1.0, "initial_pose": {"position": [0.0, 0.0, 0.155]}},
    )

    assert info["body_prim_path"] == "/World/Body"
    assert info["body_pattern"] == "/World/Body"
    assert info["bbox_min_local_stage"] == pytest.approx([-0.05, -0.05, 0.0])
    assert info["bbox_max_local_stage"] == pytest.approx([0.05, 0.05, 0.155])
    assert info["bbox_local_stage_scale"] == pytest.approx([2.0, 1.0, 1.0])
    assert info["meters_per_unit"] == pytest.approx(1.0)


def test_drop_settle_scene_uses_collider_bounds_for_pose_local_acceptance(
    tmp_path: Path,
) -> None:
    patched = _patched_physics_usd_with_helper_bbox(tmp_path)

    info = build_drop_settle_scene(
        patched,
        tmp_path / "scene.usda",
        drop_height_m=1.0,
        gravity=-9.81,
    )

    assert info["bbox_min_local_stage"] == pytest.approx([-0.5, -0.5, -0.5])
    assert info["bbox_max_local_stage"] == pytest.approx([0.5, 0.5, 0.5])


def test_drop_settle_scene_reports_composed_scale_for_pose_local_bounds(
    tmp_path: Path,
) -> None:
    from pxr import Gf

    patched = _patched_physics_usd(tmp_path)
    stage = Usd.Stage.Open(str(patched))
    UsdGeom.Xformable(stage.GetPrimAtPath("/World")).AddScaleOp().Set(
        Gf.Vec3f(2.0, 3.0, 4.0)
    )
    body_xformable = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Body"))
    matrix = Gf.Matrix4d(1.0)
    matrix.SetScale(Gf.Vec3d(0.5, 2.0, 1.5))
    body_xformable.AddTransformOp().Set(matrix)
    stage.GetRootLayer().Save()

    info = build_drop_settle_scene(
        patched,
        tmp_path / "scene.usda",
        drop_height_m=1.0,
        gravity=-9.81,
    )

    assert info["bbox_local_stage_scale"] == pytest.approx([1.0, 6.0, 6.0])


def test_freeform_scene_with_rotation_authors_rotate_op(tmp_path: Path) -> None:
    """A freeform target with ``initial_pose.rotation`` exercises
    ``_set_body_rotation``. The previous code referenced a non-existent
    ``UsdGeom.XformOp.TypeRotateXYX`` enum that would AttributeError on
    the very first freeform tune; this regression guards the fix.
    """
    patched = _patched_physics_usd(tmp_path)
    out = tmp_path / "scene.usda"

    target = {
        "duration_s": 1.0,
        "initial_pose": {
            "position": [0.0, 0.5, 0.0],
            "rotation": [0.1, 0.2, 0.3],  # XYZ Euler radians
        },
    }
    info = build_freeform_scene(patched, out, target=target)

    assert out.exists()
    rec = Usd.Stage.Open(str(out))
    body = rec.GetPrimAtPath(Sdf.Path(info["body_prim_path"]))
    op_types = [op.GetOpType() for op in UsdGeom.Xformable(body).GetOrderedXformOps()]
    # Some flavor of rotate-Euler must have been authored.
    rotate_types = {
        UsdGeom.XformOp.TypeRotateXYZ,
        UsdGeom.XformOp.TypeRotateXZY,
        UsdGeom.XformOp.TypeRotateYXZ,
        UsdGeom.XformOp.TypeRotateYZX,
        UsdGeom.XformOp.TypeRotateZXY,
        UsdGeom.XformOp.TypeRotateZYX,
    }
    assert rotate_types.intersection(op_types), (
        f"freeform target.initial_pose.rotation did not author a rotate op; "
        f"present ops: {op_types!r}"
    )


def test_freeform_scene_without_rotation_skips_rotate_op(tmp_path: Path) -> None:
    """When the target omits ``initial_pose.rotation`` the body keeps
    its existing op set unchanged — the fix must not eagerly add a
    rotate op for translate-only freeform scenarios."""
    patched = _patched_physics_usd(tmp_path)
    out = tmp_path / "scene.usda"

    target = {
        "duration_s": 1.0,
        "gravity": -1.25,
        "initial_pose": {"position": [0.0, 0.5, 0.0]},
    }
    info = build_freeform_scene(patched, out, target=target)
    assert info["gravity_magnitude_m_per_s2"] == pytest.approx(1.25)

    rec = Usd.Stage.Open(str(out))
    body = rec.GetPrimAtPath(Sdf.Path(info["body_prim_path"]))
    op_types = [op.GetOpType() for op in UsdGeom.Xformable(body).GetOrderedXformOps()]
    rotate_types = {
        UsdGeom.XformOp.TypeRotateXYZ,
        UsdGeom.XformOp.TypeRotateXZY,
        UsdGeom.XformOp.TypeRotateYXZ,
        UsdGeom.XformOp.TypeRotateYZX,
        UsdGeom.XformOp.TypeRotateZXY,
        UsdGeom.XformOp.TypeRotateZYX,
    }
    assert not rotate_types.intersection(op_types)


def _z_up_cm_scale_ladder_usd(tmp_path: Path) -> Path:
    """A SimReady-style stand-in: Z-up, metersPerUnit=0.01, single
    Mesh body with bbox [0,0,0]→[50,30,200] cm (a tall cabinet shape).

    Mirrors the apply_physics output that triggered the demo bugs:
    Z-up upAxis + cm-scale geometry + single RigidBodyAPI on a parent
    Xform.
    """
    p = tmp_path / "patched_zup_cm.usda"
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)  # cm

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    body = UsdGeom.Xform.Define(stage, "/World/Body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr().Set(5.0)

    # Mesh child with vertices in cm — a 50×30×200 cm cabinet.
    from pxr import Gf, Vt

    mesh = UsdGeom.Mesh.Define(stage, "/World/Body/Geom")
    pts = [
        (0, 0, 0),
        (50, 0, 0),
        (50, 30, 0),
        (0, 30, 0),  # bottom face
        (0, 0, 200),
        (50, 0, 200),
        (50, 30, 200),
        (0, 30, 200),  # top face
    ]
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            0,
            1,
            5,
            4,
            2,
            3,
            7,
            6,
            0,
            3,
            7,
            4,
            1,
            2,
            6,
            5,
        ]
    )
    mesh.CreateExtentAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(50, 30, 200)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

    stage.GetRootLayer().Save()
    return p


def test_drop_settle_scene_handles_z_up_cm_stage(tmp_path: Path) -> None:
    """Scene-builder must honor ``upAxis=Z`` and ``metersPerUnit=0.01``:

    * Result fields (``bbox_size_m``, ``rest_position``,
      ``drop_height_m_resolved``) report meters even when source is cm.
    * Authored scene has ``metersPerUnit=1.0`` (metric bake) so PhysX's
      stage-unit gravity equals real Earth m/s².
    * Gravity direction is ``(0, 0, -1)`` (Z-up), magnitude 9.81.
    * Body's bbox-min on Z lands at ``drop_height_m`` above the
      ground (corner-origin asset).
    * ``rest_position`` is the body's translate when settled, i.e.
      ``-bbox_min_pre_metric`` along Z (here ``0.0`` because the source
      mesh has bbox-min at origin).
    """
    src = _z_up_cm_scale_ladder_usd(tmp_path)
    out = tmp_path / "scene.usda"

    info = build_drop_settle_scene(src, out, drop_height_m=None, gravity=-9.81)

    # User-facing dimensions are meters (50×30×200 cm → 0.5×0.3×2.0 m).
    assert info["bbox_size_m"][0] == pytest.approx(0.5, abs=1e-3)
    assert info["bbox_size_m"][1] == pytest.approx(0.3, abs=1e-3)
    assert info["bbox_size_m"][2] == pytest.approx(2.0, abs=1e-3)
    assert info["drop_height_m_resolved"] == pytest.approx(2.0, abs=1e-3)

    # Output stage is metric and Z-up; gravity direction matches up-axis.
    rec = Usd.Stage.Open(str(out))
    assert UsdGeom.GetStageMetersPerUnit(rec) == pytest.approx(1.0)
    assert UsdGeom.GetStageUpAxis(rec) == UsdGeom.Tokens.z
    ps = UsdPhysics.Scene(rec.GetPrimAtPath("/PhysicsScene"))
    direction = ps.GetGravityDirectionAttr().Get()
    assert direction[0] == pytest.approx(0.0)
    assert direction[1] == pytest.approx(0.0)
    assert direction[2] == pytest.approx(-1.0)
    assert ps.GetGravityMagnitudeAttr().Get() == pytest.approx(9.81, abs=1e-3)

    # Body's bbox-min on Z lands at drop_height_m above ground.
    body = rec.GetPrimAtPath(Sdf.Path(info["body_prim_path"]))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bb = cache.ComputeWorldBound(body).ComputeAlignedRange()
    assert bb.GetMin()[2] == pytest.approx(2.0, abs=1e-3)
    # And the top sits at drop_h + bbox_h on Z.
    assert bb.GetMax()[2] == pytest.approx(4.0, abs=1e-3)

    # rest_position: corner-origin → translate at rest = (0, 0, 0).
    assert info["rest_position"][0] == pytest.approx(0.0)
    assert info["rest_position"][1] == pytest.approx(0.0)
    assert info["rest_position"][2] == pytest.approx(0.0, abs=1e-6)
    assert info["bbox_local_stage_space"] == "pose_local"
    assert info["bbox_local_stage_scale"] == pytest.approx([1.0, 1.0, 1.0])
    assert info["meters_per_unit"] == pytest.approx(1.0)
    assert info["gravity_magnitude_m_per_s2"] == pytest.approx(9.81)
    assert info["bbox_min_local_stage"][2] == pytest.approx(0.0, abs=1e-6)
    assert info["bbox_max_local_stage"][2] == pytest.approx(2.0, abs=1e-3)

    # ``world_up`` must be authoritative — Z-up here regardless of the
    # rest_position carrying no signal. Downstream consumers (the judge's
    # ``_best_trial_summary``) rely on this for corner-origin assets
    # because ``infer_world_up([0, 0, 0])`` falls back to Y-up.
    assert info["world_up"] == [0.0, 0.0, 1.0]


def test_drop_settle_scene_y_up_centered_unchanged(tmp_path: Path) -> None:
    """Y-up + meters + centered single-mesh body still works — the
    pre-existing case must not regress when we generalized the
    placement to be origin-aware. A centered mesh has bbox-min at
    -bbox_h/2 (Y), so post-translation the bbox-min lands at drop_h
    and the rest position is +bbox_h/2."""
    src = _patched_physics_usd(tmp_path)  # Y-up, mpu default, 0.5 m sphere
    out = tmp_path / "scene.usda"

    info = build_drop_settle_scene(src, out, drop_height_m=1.0, gravity=-9.81)

    rec = Usd.Stage.Open(str(out))
    assert UsdGeom.GetStageUpAxis(rec) == UsdGeom.Tokens.y

    ps = UsdPhysics.Scene(rec.GetPrimAtPath("/PhysicsScene"))
    direction = ps.GetGravityDirectionAttr().Get()
    assert (direction[0], direction[1], direction[2]) == pytest.approx((0.0, -1.0, 0.0))

    body = rec.GetPrimAtPath(Sdf.Path(info["body_prim_path"]))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bb = cache.ComputeWorldBound(body).ComputeAlignedRange()
    # Bbox-min on Y at drop_h=1.0; sphere radius=0.25 so max at 1.5.
    assert bb.GetMin()[1] == pytest.approx(1.0, abs=1e-3)
    assert bb.GetMax()[1] == pytest.approx(1.5, abs=1e-3)

    # Centered sphere: rest translate = +bbox_h/2 on the up-axis.
    assert info["rest_position"][1] == pytest.approx(0.25, abs=1e-3)
    assert info["bbox_local_stage_space"] == "pose_local"
    assert info["bbox_min_local_stage"][1] == pytest.approx(-0.25, abs=1e-3)
    assert info["bbox_max_local_stage"][1] == pytest.approx(0.25, abs=1e-3)

    # Y-up stage → world_up reports Y.
    assert info["world_up"] == [0.0, 1.0, 0.0]


def _patched_physics_usd_with_nested_scene(tmp_path: Path) -> Path:
    """Mirror what ``apply_physics`` produces for an Isaac SimReady asset
    (``/RootNode`` default-prim wrapper): a rigid body with a
    ``UsdPhysics.Scene`` *nested under the default prim* rather than at
    stage root. The scene builder must deactivate this nested scene so
    the harness-owned ``/PhysicsScene`` is the single authority — the
    real-world failure mode is the rigid body binding to the nested
    scene while the harness's GroundPlane binds to ``/PhysicsScene``,
    leaving the body in free-fall."""
    p = tmp_path / "patched_with_nested_scene.usda"
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = UsdGeom.Xform.Define(stage, "/RootNode")
    stage.SetDefaultPrim(root.GetPrim())
    body = UsdGeom.Sphere.Define(stage, "/RootNode/Body")
    body.CreateRadiusAttr(0.25)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr().Set(1.0)
    UsdPhysics.Scene.Define(stage, "/RootNode/PhysicsScene")
    stage.GetRootLayer().Save()
    return p


def test_drop_settle_scene_dedups_pre_existing_nested_physicsscene(
    tmp_path: Path,
) -> None:
    """Pre-existing ``UsdPhysics.Scene`` at a path other than
    ``/PhysicsScene`` (e.g. ``/RootNode/PhysicsScene`` from
    apply_physics on an Isaac SimReady asset) must be deactivated so
    the simulation runs in a single scene. Without dedup the rigid
    body binds to the nested scene and the GroundPlane (at
    ``/SceneRoot/GroundPlane``) binds to the harness scene at
    ``/PhysicsScene`` — they're in different physics worlds and never
    collide."""
    src = _patched_physics_usd_with_nested_scene(tmp_path)
    out = tmp_path / "scene.usda"

    build_drop_settle_scene(src, out, drop_height_m=1.0, gravity=-9.81)

    rec = Usd.Stage.Open(str(out))

    # The pre-existing nested scene survives composition but is
    # deactivated.
    nested = rec.GetPrimAtPath(Sdf.Path("/RootNode/PhysicsScene"))
    assert nested.IsValid()
    assert nested.IsActive() is False

    # The harness scene at /PhysicsScene is active and carries the
    # scenario gravity.
    harness = UsdPhysics.Scene(rec.GetPrimAtPath(Sdf.Path("/PhysicsScene")))
    assert harness.GetPrim().IsActive() is True
    assert harness.GetGravityMagnitudeAttr().Get() == pytest.approx(9.81, abs=1e-3)

    # Exactly one ACTIVE UsdPhysics.Scene in the composed stage.
    active_scenes = [
        p for p in rec.TraverseAll() if p.IsA(UsdPhysics.Scene) and p.IsActive()
    ]
    assert len(active_scenes) == 1
    assert active_scenes[0].GetPath() == Sdf.Path("/PhysicsScene")


def _camera_op_types(stage: Usd.Stage, camera_path: str) -> list:
    """Return the xform-op type list authored on the camera prim. The
    side-view helper authors a single ``xformOp:transform`` 4×4
    matrix; the corner-view helper authors the same but at a tilted
    angle. We assert presence of a transform op (rather than orient /
    translate split) because the camera helpers in
    ``world_understanding.utils.usd.camera`` use that representation.
    """
    cam = stage.GetPrimAtPath(Sdf.Path(camera_path))
    return [op.GetOpType() for op in UsdGeom.Xformable(cam).GetOrderedXformOps()]


def test_drop_settle_scene_default_camera_is_corner_view(tmp_path: Path) -> None:
    """When no ``cameras`` is supplied the scene builder authors a
    tilted corner view (``+x+y+z``), not a top-down ``-z``. A drop
    along the ladder's height axis would visualise as an
    indistinguishable bird's-eye projection in the previous default;
    the corner-view default is what a reviewer needs to see the fall.
    """
    src = _z_up_cm_scale_ladder_usd(tmp_path)
    out = tmp_path / "scene.usda"

    info = build_drop_settle_scene(src, out, cameras=None)

    # Default corner direction string survives through the scene-builder
    # default machinery and lands on the camera prim path.
    assert info["camera_paths"] == ["/Cameras/plus_xplus_yplus_z"]

    # And the camera looks tilted: the 3×3 of its transform isn't an
    # axis-aligned reflection (it would be for a cardinal "-z" cam, where
    # the basis is exactly the world axes ± a sign). For a corner view
    # the basis vectors mix two or three axes, so each row has at least
    # two non-trivial components.
    rec = Usd.Stage.Open(str(out))
    op_types = _camera_op_types(rec, "/Cameras/plus_xplus_yplus_z")
    assert UsdGeom.XformOp.TypeTransform in op_types
    cam = rec.GetPrimAtPath(Sdf.Path("/Cameras/plus_xplus_yplus_z"))
    matrix = next(
        op.Get()
        for op in UsdGeom.Xformable(cam).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform
    )
    # The basis has 3 rows. For an axis-aligned cardinal camera, each
    # row has exactly one non-zero component on the x/y/z axis pick.
    # For a corner view (+x+y+z), the look direction `(1,1,1)/sqrt(3)`
    # produces a basis where each row has 2-3 non-trivial entries.
    cardinal_rows = 0
    for row_idx in range(3):
        row = [abs(matrix[row_idx][i]) for i in range(3)]
        nonzero = sum(1 for v in row if v > 1e-3)
        if nonzero == 1:
            cardinal_rows += 1
    assert cardinal_rows < 3, (
        f"basis looks axis-aligned (cardinal-rows={cardinal_rows}/3); expected "
        "tilted basis from corner-view camera"
    )


def test_drop_settle_scene_cardinal_camera_routes_to_side_view(tmp_path: Path) -> None:
    """An explicit cardinal direction like ``+x`` should route through
    ``add_focused_side_view_camera`` (the routing fork in
    ``_author_cameras``), not the corner helper."""
    src = _z_up_cm_scale_ladder_usd(tmp_path)
    out = tmp_path / "scene.usda"

    info = build_drop_settle_scene(src, out, cameras=["+x"])

    assert info["camera_paths"] == ["/Cameras/plus_x"]

    # The side-view +x camera puts the camera on the +X axis looking
    # along -X. Its basis matrix has exactly one non-trivial component
    # per row (it's an axis-aligned camera).
    rec = Usd.Stage.Open(str(out))
    cam = rec.GetPrimAtPath(Sdf.Path("/Cameras/plus_x"))
    matrix = next(
        op.Get()
        for op in UsdGeom.Xformable(cam).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform
    )
    cardinal_rows = 0
    for row_idx in range(3):
        row = [abs(matrix[row_idx][i]) for i in range(3)]
        nonzero = sum(1 for v in row if v > 1e-3)
        if nonzero == 1:
            cardinal_rows += 1
    assert cardinal_rows == 3, (
        f"basis is not axis-aligned (cardinal-rows={cardinal_rows}/3); the "
        "cardinal direction should route through the side-view helper"
    )


def _camera_transform(stage: Usd.Stage, camera_path: str) -> Any:
    """Pull the camera's authored world-space transform (``xformOp:transform``)."""
    cam = stage.GetPrimAtPath(Sdf.Path(camera_path))
    return next(
        op.Get()
        for op in UsdGeom.Xformable(cam).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform
    )


def _camera_translate(stage: Usd.Stage, camera_path: str) -> tuple[float, float, float]:
    """Pull the camera's world-space translate out of its ``xformOp:transform``.

    Row 3 of a row-major USD ``GfMatrix4d`` is the translate.
    """
    matrix = _camera_transform(stage, camera_path)
    return (float(matrix[3][0]), float(matrix[3][1]), float(matrix[3][2]))


def _camera_forward(stage: Usd.Stage, camera_path: str) -> tuple[float, float, float]:
    """World-space forward direction of the camera.

    USD cameras look down their local ``-Z``. The world-space basis
    columns sit in the upper 3×3 of the ``xformOp:transform`` matrix
    (row-major: ``matrix[row][col]`` and rows are the camera-local
    axis directions in world space, so ``matrix[2]`` is the camera's
    local ``+Z`` in world space). Forward = ``-matrix[2]`` of the top
    three components.
    """
    matrix = _camera_transform(stage, camera_path)
    return (-float(matrix[2][0]), -float(matrix[2][1]), -float(matrix[2][2]))


def test_drop_settle_camera_ground_bias_shifts_lookat(tmp_path: Path) -> None:
    """``target.camera_ground_bias_fraction`` lerps the camera's look-at
    on the stage's up-axis from the body's bbox center toward the
    ground plane (up-axis coord 0). The camera position is computed
    from the bbox corner + a direction-aligned distance and is
    independent of the look-at override; only the camera's rotation
    changes (the camera looks lower). So the test checks the camera's
    forward direction on the up-axis: with the look-at shifted toward
    the ground, ``forward_z`` (Z-up) must become more negative.

    Uses the Z-up ladder fixture so the up-axis is Z. Body bbox is
    50×30×200 cm → height ~2.0 m. drop_height_m=1.0 → body bbox-min
    at Z=1.0 m, body center at Z=2.0 m. With ``ground_bias_fraction=0.75``
    the look-at Z lands at 2.0 * (1 - 0.75) = 0.5 m, so the camera
    tilts down more than the no-bias baseline.
    """
    src = _z_up_cm_scale_ladder_usd(tmp_path)
    out_baseline = tmp_path / "scene_baseline.usda"
    out_biased = tmp_path / "scene_biased.usda"

    info_b = build_drop_settle_scene(
        src, out_baseline, drop_height_m=1.0, gravity=-9.81
    )
    info_g = build_drop_settle_scene(
        src,
        out_biased,
        drop_height_m=1.0,
        gravity=-9.81,
        camera_ground_bias_fraction=0.75,
    )

    cam_path = "/Cameras/plus_xplus_yplus_z"
    assert info_b["camera_paths"] == [cam_path]
    assert info_g["camera_paths"] == [cam_path]

    # Camera position is unchanged (only look-at moves), so the
    # translates must match within float epsilon.
    bx, by, bz = _camera_translate(Usd.Stage.Open(str(out_baseline)), cam_path)
    gx, gy, gz = _camera_translate(Usd.Stage.Open(str(out_biased)), cam_path)
    for axis_name, baseline, biased in [("X", bx, gx), ("Y", by, gy), ("Z", bz, gz)]:
        assert biased == pytest.approx(baseline, abs=1e-6), (
            f"camera {axis_name} shifted ({baseline:.4f} → {biased:.4f}); "
            "bias should rotate the camera, not move its position"
        )

    # Camera forward on Z must be more negative (more downward) when
    # biased toward the ground.
    bfx, bfy, bfz = _camera_forward(Usd.Stage.Open(str(out_baseline)), cam_path)
    gfx, gfy, gfz = _camera_forward(Usd.Stage.Open(str(out_biased)), cam_path)
    assert gfz < bfz - 0.05, (
        f"biased camera forward Z {gfz:.4f} not meaningfully more negative "
        f"than baseline {bfz:.4f}; expected the bias to tilt the camera "
        "downward toward the ground"
    )
    # Forward X/Y change is OK (the rotation is repointing), but the
    # off-up-axis target coords were NOT overridden so the look-at's
    # X/Y are still body center — the rotation difference should be
    # roughly a tilt, not a yaw. Sanity: the camera still mostly looks
    # toward the body in the horizontal plane (forward X and Y same
    # sign as baseline).
    assert (gfx * bfx) > 0 and (gfy * bfy) > 0, (
        "bias rotated the camera horizontally too — expected only up-axis tilt"
    )


def test_camera_ground_bias_fraction_rejects_out_of_range(tmp_path: Path) -> None:
    """``camera_ground_bias_fraction`` outside [0.0, 1.0] is invalid;
    silently clamping would surprise users (e.g. negative biases that
    push the look-at above the body, or bias > 1 that drives it
    underground). Fail loud at scene-build time."""
    src = _patched_physics_usd(tmp_path)
    out = tmp_path / "scene.usda"

    with pytest.raises(ValueError, match="ground_bias_fraction"):
        build_drop_settle_scene(
            src, out, drop_height_m=0.5, camera_ground_bias_fraction=-0.1
        )
    with pytest.raises(ValueError, match="ground_bias_fraction"):
        build_drop_settle_scene(
            src, out, drop_height_m=0.5, camera_ground_bias_fraction=1.5
        )


def test_drop_settle_camera_ground_bias_with_cardinal_direction_warns(
    tmp_path: Path, scene_builder_caplog: pytest.LogCaptureFixture
) -> None:
    """``add_focused_side_view_camera`` does NOT accept ``target_x/y/z``
    (only the corner-view helper does), so the previous
    ``**target_kwargs`` spread crashed with ``TypeError`` when the
    scenario combined ``camera_ground_bias_fraction`` with a cardinal
    direction like ``"+x"``. The build must now succeed (the bias is
    dropped for cardinal directions) and a single ``WARNING`` is
    emitted so the silent drop is auditable in logs.
    """
    src = _patched_physics_usd(tmp_path)
    out = tmp_path / "scene.usda"

    with scene_builder_caplog.at_level(
        "WARNING", logger="physics_agent.tuning.scenarios._scene_builder"
    ):
        info = build_drop_settle_scene(
            src,
            out,
            drop_height_m=0.5,
            cameras=["+x"],
            camera_ground_bias_fraction=0.5,
        )

    assert info["camera_paths"] == ["/Cameras/plus_x"]
    rec = Usd.Stage.Open(str(out))
    assert rec.GetPrimAtPath(Sdf.Path("/Cameras/plus_x")).IsValid()

    matching = [
        r
        for r in scene_builder_caplog.records
        if "ground_bias_fraction" in r.message and "cardinal" in r.message
    ]
    assert len(matching) == 1, (
        "expected exactly one warning about ground_bias_fraction being "
        f"ignored for the cardinal direction; got {len(matching)} matches "
        f"out of {[r.message for r in scene_builder_caplog.records]!r}"
    )


def test_drop_settle_camera_ground_bias_warns_once_for_multiple_cardinals(
    tmp_path: Path, scene_builder_caplog: pytest.LogCaptureFixture
) -> None:
    """When several cardinal directions trigger the ignored-bias path,
    only one warning is logged per ``build_drop_settle_scene`` call so
    the log doesn't get spammed for scenarios that author many
    cardinal views."""
    src = _patched_physics_usd(tmp_path)
    out = tmp_path / "scene.usda"

    with scene_builder_caplog.at_level(
        "WARNING", logger="physics_agent.tuning.scenarios._scene_builder"
    ):
        info = build_drop_settle_scene(
            src,
            out,
            drop_height_m=0.5,
            cameras=["+x", "+y", "+z"],
            camera_ground_bias_fraction=0.5,
        )

    assert info["camera_paths"] == [
        "/Cameras/plus_x",
        "/Cameras/plus_y",
        "/Cameras/plus_z",
    ]
    matching = [
        r
        for r in scene_builder_caplog.records
        if "ground_bias_fraction" in r.message and "cardinal" in r.message
    ]
    assert len(matching) == 1, (
        f"expected one warning across three cardinal cameras; got {len(matching)}"
    )


def _patched_physics_usd_with_instance_proxy_scene(tmp_path: Path) -> Path:
    """Build a stage where the ``UsdPhysics.Scene`` arrives via an
    *instanced* reference, so the composed stage exposes it as a
    read-only instance proxy. The scene builder's dedup path must skip
    SetActive on these (instance proxies silently no-op SetActive) and
    log a warning instead of falsely claiming dedup succeeded."""
    proto_path = tmp_path / "_proto_with_scene.usda"
    proto = Usd.Stage.CreateNew(str(proto_path))
    UsdGeom.SetStageUpAxis(proto, UsdGeom.Tokens.y)
    proto_root = UsdGeom.Xform.Define(proto, "/Prototype")
    proto.SetDefaultPrim(proto_root.GetPrim())
    UsdPhysics.Scene.Define(proto, "/Prototype/PhysicsScene")
    proto.GetRootLayer().Save()

    p = tmp_path / "patched_with_instance_proxy_scene.usda"
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    root = UsdGeom.Xform.Define(stage, "/RootNode")
    stage.SetDefaultPrim(root.GetPrim())

    body = UsdGeom.Sphere.Define(stage, "/RootNode/Body")
    body.CreateRadiusAttr(0.25)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr().Set(1.0)

    wrapper = UsdGeom.Xform.Define(stage, "/RootNode/Wrapper")
    wrapper.GetPrim().GetReferences().AddReference(str(proto_path))
    wrapper.GetPrim().SetInstanceable(True)

    stage.GetRootLayer().Save()
    return p


def test_drop_settle_skips_instance_proxy_physicsscene(
    tmp_path: Path, scene_builder_caplog: pytest.LogCaptureFixture
) -> None:
    """A pre-existing ``UsdPhysics.Scene`` reached through an instanced
    reference becomes an instance proxy in the composed stage.
    ``SetActive(False)`` on instance proxies is a silent no-op in USD,
    so the dedup loop must skip them with a logged warning rather than
    pretend to have deactivated them. Asserts the proxy *is still
    active* (we didn't accidentally mutate it) and that the harness
    scene is the sole one targeted for activation by the harness."""
    src = _patched_physics_usd_with_instance_proxy_scene(tmp_path)
    out = tmp_path / "scene.usda"

    with scene_builder_caplog.at_level(
        "WARNING", logger="physics_agent.tuning.scenarios._scene_builder"
    ):
        build_drop_settle_scene(src, out, drop_height_m=0.5, gravity=-9.81)

    rec = Usd.Stage.Open(str(out))
    harness = UsdPhysics.Scene(rec.GetPrimAtPath(Sdf.Path("/PhysicsScene")))
    assert harness.GetPrim().IsValid() and harness.GetPrim().IsActive()
    assert harness.GetGravityMagnitudeAttr().Get() == pytest.approx(9.81, abs=1e-3)

    # Proxy scene must remain present and active — the skip path is a
    # no-op on the proxy itself, NOT a sneaky activate-then-deactivate
    # cycle. Traverse with the instance-proxy predicate to see proxies.
    proxy_range = Usd.PrimRange.Stage(
        rec, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    )
    proxy_scenes = [
        p for p in proxy_range if p.IsA(UsdPhysics.Scene) and p.IsInstanceProxy()
    ]
    assert len(proxy_scenes) == 1, (
        f"expected the original instance-proxy UsdPhysics.Scene to survive "
        f"the build; got {len(proxy_scenes)} proxy-scene prims"
    )
    assert proxy_scenes[0].IsActive(), (
        "instance-proxy PhysicsScene should still be active (proxy is "
        "read-only; we cannot toggle it) — found inactive"
    )

    # Same active-scene-count invariant as the nested-scene test: the
    # harness scene is the only one the dedup loop activated.
    active_scenes = [
        p
        for p in Usd.PrimRange.Stage(
            rec, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        )
        if p.IsA(UsdPhysics.Scene) and p.IsActive()
    ]
    # Both the harness scene and the proxy scene are "active" in USD's
    # sense (proxy by construction; harness by authorship). The
    # important invariant is that no NON-proxy non-target scene is
    # silently left active.
    non_proxy_active = [p for p in active_scenes if not p.IsInstanceProxy()]
    assert len(non_proxy_active) == 1
    assert non_proxy_active[0].GetPath() == Sdf.Path("/PhysicsScene")

    matching = [
        r
        for r in scene_builder_caplog.records
        if "instance-proxy UsdPhysics.Scene" in r.message
    ]
    assert len(matching) >= 1, (
        "expected a warning about skipping the instance-proxy PhysicsScene; "
        f"got log records: {[r.message for r in scene_builder_caplog.records]!r}"
    )


def _patched_physics_usd_with_coexisting_scenes(tmp_path: Path) -> Path:
    """Stage with BOTH an instance-proxy ``UsdPhysics.Scene`` AND a
    non-proxy nested scene authored directly on the root layer. Real
    SimReady-style assets can hit this combination when one sub-asset
    is instanced and the wrapper's own physics scene is direct. The
    dedup loop must take BOTH branches in a single traversal: warn-and-
    skip on the proxy, deactivate the non-proxy. We assemble both in
    one fixture so the two branches are exercised on the same
    ``_author_physics_scene`` call."""
    proto_path = tmp_path / "_proto_with_scene.usda"
    proto = Usd.Stage.CreateNew(str(proto_path))
    UsdGeom.SetStageUpAxis(proto, UsdGeom.Tokens.y)
    proto_root = UsdGeom.Xform.Define(proto, "/Prototype")
    proto.SetDefaultPrim(proto_root.GetPrim())
    UsdPhysics.Scene.Define(proto, "/Prototype/PhysicsScene")
    proto.GetRootLayer().Save()

    p = tmp_path / "patched_with_coexisting_scenes.usda"
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    root = UsdGeom.Xform.Define(stage, "/RootNode")
    stage.SetDefaultPrim(root.GetPrim())

    body = UsdGeom.Sphere.Define(stage, "/RootNode/Body")
    body.CreateRadiusAttr(0.25)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr().Set(1.0)

    # Non-proxy nested scene (direct authorship).
    UsdPhysics.Scene.Define(stage, "/RootNode/PhysicsScene")

    # Proxy scene via an instanceable reference to the prototype.
    wrapper = UsdGeom.Xform.Define(stage, "/RootNode/Wrapper")
    wrapper.GetPrim().GetReferences().AddReference(str(proto_path))
    wrapper.GetPrim().SetInstanceable(True)

    stage.GetRootLayer().Save()
    return p


def test_drop_settle_handles_proxy_and_nonproxy_scenes_coexisting(
    tmp_path: Path, scene_builder_caplog: pytest.LogCaptureFixture
) -> None:
    """When a stage contains BOTH a non-proxy nested ``UsdPhysics.Scene``
    AND an instance-proxy scene, ``_author_physics_scene`` must (a)
    deactivate the non-proxy nested scene so the harness wins and (b)
    log the skip warning for the proxy without crashing. Both branches
    of the dedup loop are exercised on a single call."""
    src = _patched_physics_usd_with_coexisting_scenes(tmp_path)
    out = tmp_path / "scene.usda"

    with scene_builder_caplog.at_level(
        "WARNING", logger="physics_agent.tuning.scenarios._scene_builder"
    ):
        build_drop_settle_scene(src, out, drop_height_m=0.5, gravity=-9.81)

    rec = Usd.Stage.Open(str(out))

    # Harness scene is the active authority.
    harness = UsdPhysics.Scene(rec.GetPrimAtPath(Sdf.Path("/PhysicsScene")))
    assert harness.GetPrim().IsValid() and harness.GetPrim().IsActive()

    # Non-proxy nested scene was deactivated.
    nested = rec.GetPrimAtPath(Sdf.Path("/RootNode/PhysicsScene"))
    assert nested.IsValid() and nested.IsActive() is False

    # Proxy scene remains untouched (still active by construction).
    proxy_range = Usd.PrimRange.Stage(
        rec, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    )
    proxy_scenes = [
        p for p in proxy_range if p.IsA(UsdPhysics.Scene) and p.IsInstanceProxy()
    ]
    assert len(proxy_scenes) == 1 and proxy_scenes[0].IsActive()

    # Among NON-proxy scenes, only the harness is active.
    non_proxy_active = [
        p
        for p in Usd.PrimRange.Stage(
            rec, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        )
        if p.IsA(UsdPhysics.Scene) and not p.IsInstanceProxy() and p.IsActive()
    ]
    assert len(non_proxy_active) == 1
    assert non_proxy_active[0].GetPath() == Sdf.Path("/PhysicsScene")

    # Warning fired for the proxy.
    matching = [
        r
        for r in scene_builder_caplog.records
        if "instance-proxy UsdPhysics.Scene" in r.message
    ]
    assert len(matching) >= 1


def test_freeform_scene_camera_ground_bias_shifts_lookat(tmp_path: Path) -> None:
    """``target.camera_ground_bias_fraction`` is also honored by
    ``build_freeform_scene`` — the kwarg threads from the freeform
    target dict through ``_author_cameras`` identically to drop_settle.
    A regression that disconnects the freeform key from the kwarg
    would otherwise ship green because every other bias test uses the
    drop_settle path. Mirrors
    ``test_drop_settle_camera_ground_bias_shifts_lookat`` but on the
    freeform builder."""
    src = _z_up_cm_scale_ladder_usd(tmp_path)
    out_baseline = tmp_path / "freeform_baseline.usda"
    out_biased = tmp_path / "freeform_biased.usda"

    target_baseline: dict[str, Any] = {
        "duration_s": 1.0,
        "initial_pose": {"position": [0.0, 0.0, 1.0]},
    }
    target_biased: dict[str, Any] = {
        "duration_s": 1.0,
        "initial_pose": {"position": [0.0, 0.0, 1.0]},
        "camera_ground_bias_fraction": 0.75,
    }
    info_b = build_freeform_scene(src, out_baseline, target=target_baseline)
    info_g = build_freeform_scene(src, out_biased, target=target_biased)

    cam_path = "/Cameras/plus_xplus_yplus_z"
    assert info_b["camera_paths"] == [cam_path]
    assert info_g["camera_paths"] == [cam_path]

    # Camera position unchanged (only look-at moves, so rotation changes).
    bx, by, bz = _camera_translate(Usd.Stage.Open(str(out_baseline)), cam_path)
    gx, gy, gz = _camera_translate(Usd.Stage.Open(str(out_biased)), cam_path)
    for axis_name, baseline, biased in [("X", bx, gx), ("Y", by, gy), ("Z", bz, gz)]:
        assert biased == pytest.approx(baseline, abs=1e-6), (
            f"freeform camera {axis_name} shifted ({baseline:.4f} → {biased:.4f}); "
            "bias should rotate the camera, not move its position"
        )

    # Camera forward on the up-axis (Z here) must be more negative with bias.
    _bfx, _bfy, bfz = _camera_forward(Usd.Stage.Open(str(out_baseline)), cam_path)
    _gfx, _gfy, gfz = _camera_forward(Usd.Stage.Open(str(out_biased)), cam_path)
    assert gfz < bfz - 0.05, (
        f"freeform biased camera forward Z {gfz:.4f} not meaningfully more "
        f"negative than baseline {bfz:.4f}; expected the bias to tilt the "
        "camera downward toward the ground"
    )


def test_camera_ground_bias_fraction_accepts_string_floats(tmp_path: Path) -> None:
    """YAML or LLM-authored ``camera_ground_bias_fraction`` values may
    arrive as strings (e.g. quoted ``"0.5"`` in YAML, or LLM JSON
    output that wraps numbers in strings). The validation path coerces
    via ``float()`` so callers don't need to pre-cast, and non-numeric
    values raise ``ValueError`` (not ``TypeError``) for a consistent
    error contract."""
    src = _patched_physics_usd(tmp_path)
    out = tmp_path / "scene.usda"

    info = build_drop_settle_scene(
        src, out, drop_height_m=0.5, camera_ground_bias_fraction="0.5"
    )
    assert info["camera_paths"] == ["/Cameras/plus_xplus_yplus_z"]

    with pytest.raises(ValueError, match="ground_bias_fraction"):
        build_drop_settle_scene(
            src, out, drop_height_m=0.5, camera_ground_bias_fraction="not-a-number"
        )


@pytest.mark.parametrize("bias_str", ["-0.1", "1.5", "2.0", "-1.0"])
def test_camera_ground_bias_fraction_rejects_out_of_range_strings(
    tmp_path: Path, bias_str: str
) -> None:
    """String-form out-of-range bias values must raise ``ValueError``
    after the ``float()`` coercion, mirroring the numeric-form
    rejection. Otherwise an LLM authoring ``"1.5"`` in JSON would slip
    past the range check via the string-coercion path."""
    src = _patched_physics_usd(tmp_path)
    out = tmp_path / "scene.usda"
    with pytest.raises(ValueError, match="ground_bias_fraction"):
        build_drop_settle_scene(
            src, out, drop_height_m=0.5, camera_ground_bias_fraction=bias_str
        )


@pytest.mark.parametrize("bias_value", [True, False])
def test_camera_ground_bias_fraction_rejects_bool(
    tmp_path: Path, bias_value: bool
) -> None:
    """``bool`` is a subclass of ``int`` in Python; ``float(True) ==
    1.0`` and ``float(False) == 0.0``. Both pass the [0.0, 1.0] bounds
    check, but neither matches plausible caller intent (a YAML or LLM
    that emits ``true`` almost certainly means "enable the feature",
    not "lerp 100% toward the ground"). Reject explicitly so the
    misconfig surfaces as a ``ValueError``."""
    src = _patched_physics_usd(tmp_path)
    out = tmp_path / "scene.usda"
    with pytest.raises(ValueError, match="bool"):
        build_drop_settle_scene(
            src, out, drop_height_m=0.5, camera_ground_bias_fraction=bias_value
        )


def test_ground_plane_is_outside_body_subtree(tmp_path: Path) -> None:
    """Round 12 (CX P1#1): the ground plane and its physics material
    must NOT be authored under the rigid-body parent prim; otherwise the
    plane inherits ``UsdPhysics.RigidBodyAPI`` from the parent and the
    asset never falls.

    The bundled ``_patched_physics_usd`` fixture authors the rigid body
    on ``/World/Body`` (the legacy single-prim layout). Tighter assets
    like the lightbulb put the body on the stage default-prim itself
    (``/light_bulb_01``); both layouts must keep the plane out of the
    body subtree.
    """
    patched = _patched_physics_usd(tmp_path)
    out = tmp_path / "scene.usda"
    info = build_drop_settle_scene(patched, out, drop_height_m=0.5)

    rec = Usd.Stage.Open(str(out))
    body_path = Sdf.Path(info["body_prim_path"])

    # Walk every prim with CollisionAPI; one of them is the body, and at
    # least one must be the ground plane sitting OUTSIDE the body
    # subtree.
    plane_outside_body_count = 0
    for prim in rec.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        prim_path = prim.GetPath()
        if prim_path == body_path or prim_path.HasPrefix(body_path):
            continue
        # Plane prim — confirm it does NOT inherit rigid body from its
        # ancestors.
        anc = prim.GetParent()
        inherits_rigid_body = False
        while anc.IsValid() and anc.GetPath() != Sdf.Path.absoluteRootPath:
            if anc.HasAPI(UsdPhysics.RigidBodyAPI):
                inherits_rigid_body = True
                break
            anc = anc.GetParent()
        assert not inherits_rigid_body, (
            f"ground-plane-like prim {prim_path} has a rigid-body "
            f"ancestor; ground would not stay static"
        )
        plane_outside_body_count += 1

    assert plane_outside_body_count >= 1, (
        "no static collider authored outside the rigid-body subtree; "
        "the ground plane is missing or wrongly nested under the body"
    )


def test_ground_plane_material_is_matte_preview_surface(tmp_path: Path) -> None:
    """The synthetic physics ground must also carry a visual material.

    A physics-only ``UsdShade.Material`` leaves RTX renderers to infer
    visual defaults, which can turn the ground plane glossy under the
    default HDRI dome. Keep the turntable-style dark matte preview
    shader while preserving the physics material attributes.
    """
    patched = _patched_physics_usd(tmp_path)
    out = tmp_path / "scene.usda"

    build_drop_settle_scene(patched, out, drop_height_m=0.5)

    rec = Usd.Stage.Open(str(out))
    material = UsdShade.Material.Get(rec, Sdf.Path("/SceneRoot/GroundPlaneMaterial"))
    assert material

    surface = material.GetSurfaceOutput()
    assert surface.GetAttr().GetConnections() == [
        Sdf.Path("/SceneRoot/GroundPlaneMaterial/PreviewSurface.outputs:surface")
    ]

    shader = UsdShade.Shader.Get(
        rec, Sdf.Path("/SceneRoot/GroundPlaneMaterial/PreviewSurface")
    )
    assert shader
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx(
        (0.14, 0.14, 0.14)
    )
    assert shader.GetInput("roughness").Get() == pytest.approx(0.86)
    assert shader.GetInput("metallic").Get() == pytest.approx(0.0)
    assert shader.GetInput("useSpecularWorkflow").Get() == 1
    assert tuple(shader.GetInput("specularColor").Get()) == pytest.approx(
        (0.0, 0.0, 0.0)
    )

    physics_material = UsdPhysics.MaterialAPI(material.GetPrim())
    assert physics_material.GetStaticFrictionAttr().Get() == pytest.approx(0.5)
    assert physics_material.GetDynamicFrictionAttr().Get() == pytest.approx(0.5)
    assert physics_material.GetRestitutionAttr().Get() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Regression: kimbyn 2026-05-12 — pre-translated assets must land above ground
# ---------------------------------------------------------------------------


def _patched_physics_usd_with_translate(
    tmp_path: Path, translate: tuple[float, float, float]
) -> Path:
    """Same as :func:`_patched_physics_usd` but pre-authors a translate
    on the body so we can probe the bbox-min-aware placement logic
    against a transformed asset (the case kimbyn's review flagged).

    Sets ``metersPerUnit = 1.0`` explicitly so the scene-builder's
    ``_bake_metric_units`` (which rescales pre-existing translates by
    ``mpu``) is a no-op — otherwise the default 0.01 (cm) USD unit
    would silently turn the test's intended 1m-up translate into 1cm.
    """
    from pxr import Gf

    p = tmp_path / "patched_translated.usda"
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    body = UsdGeom.Sphere.Define(stage, "/World/Body")
    body.CreateRadiusAttr(0.25)
    UsdGeom.Xformable(body.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*translate))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr().Set(1.0)
    stage.GetRootLayer().Save()
    return p


def test_drop_settle_preserves_pre_existing_body_translate(tmp_path: Path) -> None:
    """An asset whose body is already translated upward must still land
    its bottom at ``drop_height_m`` above the ground, not below it.

    Before kimbyn's 2026-05-12 fix, ``_set_body_translation`` REPLACED
    the existing translate with the gap-closing delta. For a sphere of
    radius 0.25 pre-translated to y=1.0, ``world bbox_min.y = 0.75``;
    the delta ``drop_h - 0.75`` would then OVERWRITE the existing 1.0
    instead of being added to it, dropping the body below ground for
    any ``drop_h < 0.75``. The fix is to add the delta to the current
    translate so ``new_world_bbox_min.y == drop_h`` regardless of any
    pre-existing offset.
    """
    drop_h = 0.5
    pre_translate = (0.0, 1.0, 0.0)  # body already 1m up on Y
    patched = _patched_physics_usd_with_translate(tmp_path, pre_translate)
    out = tmp_path / "scene.usda"

    build_drop_settle_scene(patched, out, drop_height_m=drop_h, gravity=-9.81)

    rec = Usd.Stage.Open(str(out))
    body = rec.GetPrimAtPath(Sdf.Path("/World/Body"))
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_],
    )
    # ComputeAlignedRange (NOT GetRange) returns the axis-aligned bbox in
    # world coordinates — matches what the production placement code uses
    # via ``_bbox_minmax_stage_units`` → ``get_bbox_from_prim``.
    bbox_min = cache.ComputeWorldBound(body).ComputeAlignedRange().GetMin()
    # Y is up here. bbox_min.y must equal drop_h within float tolerance.
    assert abs(bbox_min[1] - drop_h) < 1e-5, (
        f"body bottom landed at y={bbox_min[1]:.6f}, expected {drop_h}; "
        "pre-existing translate was not preserved as delta"
    )


def test_freeform_default_placement_preserves_pre_existing_body_translate(
    tmp_path: Path,
) -> None:
    """Parallel of the drop_settle test for freeform's default
    placement branch (no explicit ``initial_pose.position``).
    """
    pre_translate = (0.0, 0.7, 0.0)
    patched = _patched_physics_usd_with_translate(tmp_path, pre_translate)
    out = tmp_path / "scene.usda"

    target = {"duration_s": 1.0}  # no initial_pose → default placement branch
    build_freeform_scene(patched, out, target=target)

    rec = Usd.Stage.Open(str(out))
    body = rec.GetPrimAtPath(Sdf.Path("/World/Body"))
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_],
    )
    bbox_min = cache.ComputeWorldBound(body).ComputeAlignedRange().GetMin()
    # Freeform's default places the body bottom at bbox_height above ground.
    # For a sphere radius 0.25, bbox_height = 0.5 → bbox_min.y == 0.5.
    expected_bottom = 0.5
    assert abs(bbox_min[1] - expected_bottom) < 1e-5, (
        f"freeform default placement: bottom at y={bbox_min[1]:.6f}, "
        f"expected {expected_bottom}; pre-existing translate not preserved"
    )


# ---------------------------------------------------------------------------
# Regression: kimbyn 2026-05-12 — fractional camera dirs → USD-safe prim names
# ---------------------------------------------------------------------------


def test_fractional_camera_direction_authors_valid_prim_path(tmp_path: Path) -> None:
    """A camera direction like ``+x-0.5y+z`` (documented fractional
    weighting for diagonal views) used to produce
    ``/Cameras/plus_xminus_0.5yplus_z`` — an ill-formed Sdf.Path
    because ``.`` is illegal in USD prim names. The sanitizer must
    map non-alphanumerics to underscores so the resulting path is
    valid.
    """
    patched = _patched_physics_usd(tmp_path)
    out = tmp_path / "scene.usda"

    info = build_drop_settle_scene(
        patched,
        out,
        drop_height_m=0.5,
        gravity=-9.81,
        cameras=["+x-0.5y+z"],
    )

    cam_paths = info.get("camera_paths") or []
    assert len(cam_paths) == 1
    cam_path = cam_paths[0]
    # USD must accept the path as well-formed.
    sdf_path = Sdf.Path(cam_path)
    assert not sdf_path.isEmpty, (
        f"sanitized camera path {cam_path!r} is still ill-formed under Sdf"
    )
    # And no '.' may remain in the prim name segment.
    assert "." not in cam_path.split("/")[-1], (
        f"prim name component still contains '.': {cam_path!r}"
    )
    # The camera prim must actually exist on the recorded scene.
    rec = Usd.Stage.Open(str(out))
    cam_prim = rec.GetPrimAtPath(sdf_path)
    assert cam_prim and cam_prim.IsValid(), (
        f"sanitized camera path {cam_path!r} did not resolve to a real prim"
    )


def test_digit_leading_camera_direction_authors_valid_prim_path(
    tmp_path: Path,
) -> None:
    """Round 20 adversarial follow-up: an LLM-authored scenario could
    omit the leading ``+`` / ``-`` sign on the first axis term and emit
    e.g. ``0.5x+0.5y+0.5z``. Without a guard, ``_sanitize_prim_name``
    would produce ``0_5xplus_0_5yplus_0_5z`` — Sdf rejects digit-leading
    prim names as ill-formed. The sanitizer must prefix an underscore
    when the cleaned token starts with a digit so the resulting path
    is always well-formed regardless of caller convention.
    """
    patched = _patched_physics_usd(tmp_path)
    out = tmp_path / "scene.usda"

    info = build_drop_settle_scene(
        patched,
        out,
        drop_height_m=0.5,
        gravity=-9.81,
        cameras=["0.5x+0.5y+0.5z"],
    )

    cam_paths = info.get("camera_paths") or []
    assert len(cam_paths) == 1
    cam_path = cam_paths[0]
    # Sdf must accept the path. ``0_5x...`` (digit-leading) would fail;
    # the underscore-prefix guard turns it into ``_0_5x...``.
    sdf_path = Sdf.Path(cam_path)
    assert not sdf_path.isEmpty, (
        f"sanitized camera path {cam_path!r} is still ill-formed under Sdf"
    )
    # Prim name component must not start with a digit.
    prim_name = cam_path.split("/")[-1]
    assert not prim_name[0].isdigit(), (
        f"prim name {prim_name!r} starts with a digit; Sdf will reject it"
    )
    # And the prim must actually exist on the recorded scene.
    rec = Usd.Stage.Open(str(out))
    cam_prim = rec.GetPrimAtPath(sdf_path)
    assert cam_prim and cam_prim.IsValid(), (
        f"digit-leading camera path {cam_path!r} did not resolve to a real prim"
    )
