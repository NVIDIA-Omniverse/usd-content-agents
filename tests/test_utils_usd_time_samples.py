# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Roundtrip tests for ``world_understanding.utils.usd.time_samples``.

These cover the new ``add_pose_velocity_trajectory`` /
``read_pose_velocity_trajectory`` pair — the path the physics-tune
recording uses to persist raw simulator pose + velocity as time-sampled
USD attributes (issue #50). Verifies:

* USD authoring uses split ``xformOp:translate`` + ``xformOp:orient``
  (not a single matrix), so usdview shows raw values directly.
* Velocity attributes are the standard ``UsdPhysics.RigidBodyAPI``
  ``physics:velocity`` / ``physics:angularVelocity`` schema attrs.
* The read-back returns the same numbers the daemon emitted, so
  metric functions in
  ``world_understanding.functions.physics.trajectory`` work
  identically against the in-flight dict and the persisted USD.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")  # noqa: F841 — usd-core is a dev dep
from pxr import (  # type: ignore[import-untyped]  # noqa: E402
    Gf,
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
)

from world_understanding.utils.usd import (
    time_samples as time_samples_utils,  # noqa: E402
)
from world_understanding.utils.usd.time_samples import (  # noqa: E402
    add_pose_velocity_trajectory,
    add_xform_time_sample,
    iter_time_samples,
    read_pose_velocity_trajectory,
)


def _make_stage_with_body(tmp_path: Path) -> tuple[Usd.Stage, str]:
    """Create a minimal stage with a single rigid-body Xform prim."""
    p = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/World")
    body = UsdGeom.Xform.Define(stage, "/World/Body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    return stage, "/World/Body"


def _identity_quat() -> list[float]:
    return [0.0, 0.0, 0.0, 1.0]


def test_roundtrip_pose_and_velocity(tmp_path: Path) -> None:
    """Author a 3-sample trajectory and read it back unchanged."""
    stage, body_path = _make_stage_with_body(tmp_path)
    body = stage.GetPrimAtPath(Sdf.Path(body_path))

    trajectory = [
        (0.0, [0.0, 1.0, 0.0] + _identity_quat(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (1.0, [0.1, 0.5, 0.0] + _identity_quat(), [0.1, -0.5, 0.0, 0.0, 1.0, 0.0]),
        (2.0, [0.2, 0.0, 0.0] + _identity_quat(), [0.2, 0.0, 0.0, 0.0, 0.0, 0.5]),
    ]
    add_pose_velocity_trajectory(body, trajectory)

    out = read_pose_velocity_trajectory(stage, body_path)
    assert len(out) == len(trajectory)
    for (t_in, pose_in, vel_in), (t_out, pose_out, vel_out) in zip(
        trajectory, out, strict=True
    ):
        assert t_out == pytest.approx(t_in, abs=1e-9)
        for a, b in zip(pose_in, pose_out, strict=True):
            assert b == pytest.approx(a, abs=1e-5)  # Quatf == 32-bit
        for a, b in zip(vel_in, vel_out, strict=True):
            assert b == pytest.approx(a, abs=1e-5)  # Vec3f == 32-bit


def test_xform_op_order_is_translate_then_orient(tmp_path: Path) -> None:
    """No ``xformOp:transform`` matrix; ops are split for usdview clarity."""
    stage, body_path = _make_stage_with_body(tmp_path)
    body = stage.GetPrimAtPath(Sdf.Path(body_path))

    add_pose_velocity_trajectory(
        body,
        [(0.0, [0.0, 1.0, 0.0] + _identity_quat(), [0.0] * 6)],
    )

    xformable = UsdGeom.Xformable(body)
    op_types = [op.GetOpType() for op in xformable.GetOrderedXformOps()]
    assert UsdGeom.XformOp.TypeTransform not in op_types
    assert op_types == [
        UsdGeom.XformOp.TypeTranslate,
        UsdGeom.XformOp.TypeOrient,
    ]


def test_preexisting_scale_op_preserved(tmp_path: Path) -> None:
    """A body with an authored ``xformOp:scale`` (e.g. unit-cube scaled
    to 0.5 m by ``apply_physics``) keeps its scale through trajectory
    authoring — the simulator's pose drives translate+orient and the
    scale tail-op stays in xformOpOrder so the body still renders at
    the right size."""
    stage, body_path = _make_stage_with_body(tmp_path)
    body = stage.GetPrimAtPath(Sdf.Path(body_path))
    from pxr import Gf

    UsdGeom.Xformable(body).AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))

    add_pose_velocity_trajectory(
        body,
        [(0.0, [0.0, 1.0, 0.0] + _identity_quat(), [0.0] * 6)],
    )

    op_types = [op.GetOpType() for op in UsdGeom.Xformable(body).GetOrderedXformOps()]
    # Scale survives; pose ops go in front of it.
    assert UsdGeom.XformOp.TypeScale in op_types
    assert op_types[:2] == [
        UsdGeom.XformOp.TypeTranslate,
        UsdGeom.XformOp.TypeOrient,
    ]


def test_preexisting_matrix_transform_dropped(tmp_path: Path) -> None:
    """A pre-existing ``xformOp:transform`` matrix must NOT survive —
    it would stack on top of the simulator's pose. Scale (a non-pose
    op) survives; the matrix transform does not."""
    stage, body_path = _make_stage_with_body(tmp_path)
    body = stage.GetPrimAtPath(Sdf.Path(body_path))
    from pxr import Gf

    matrix_op = UsdGeom.Xformable(body).AddTransformOp()
    matrix_op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(99.0, 99.0, 99.0)))

    add_pose_velocity_trajectory(
        body,
        [(0.0, [0.0, 1.0, 0.0] + _identity_quat(), [0.0] * 6)],
    )

    op_types = [op.GetOpType() for op in UsdGeom.Xformable(body).GetOrderedXformOps()]
    assert UsdGeom.XformOp.TypeTransform not in op_types


def test_preexisting_pose_ops_are_reused_and_old_samples_cleared(
    tmp_path: Path,
) -> None:
    stage, body_path = _make_stage_with_body(tmp_path)
    body = stage.GetPrimAtPath(Sdf.Path(body_path))
    xformable = UsdGeom.Xformable(body)
    translate_op = xformable.AddTranslateOp()
    orient_op = xformable.AddOrientOp()
    translate_op.Set(Gf.Vec3d(99.0, 99.0, 99.0), time=Usd.TimeCode(99.0))
    orient_op.Set(
        Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)),
        time=Usd.TimeCode(99.0),
    )
    rb_api = UsdPhysics.RigidBodyAPI(body)
    rb_api.CreateVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0)).Set(
        Gf.Vec3f(9.0, 9.0, 9.0),
        time=Usd.TimeCode(99.0),
    )
    rb_api.CreateAngularVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0)).Set(
        Gf.Vec3f(8.0, 8.0, 8.0),
        time=Usd.TimeCode(99.0),
    )

    add_pose_velocity_trajectory(
        body,
        [(0.0, [1.0, 2.0, 3.0] + _identity_quat(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
    )

    ordered = UsdGeom.Xformable(body).GetOrderedXformOps()
    assert ordered[0].GetAttr().GetPath() == translate_op.GetAttr().GetPath()
    assert ordered[1].GetAttr().GetPath() == orient_op.GetAttr().GetPath()
    assert translate_op.GetAttr().GetTimeSamples() == [0.0]
    assert orient_op.GetAttr().GetTimeSamples() == [0.0]
    assert rb_api.GetVelocityAttr().GetTimeSamples() == [0.0]
    assert rb_api.GetAngularVelocityAttr().GetTimeSamples() == [0.0]


def test_velocity_attrs_are_rigidbodyapi(tmp_path: Path) -> None:
    """``physics:velocity`` + ``physics:angularVelocity`` come from the
    standard ``UsdPhysics.RigidBodyAPI`` schema, not custom attributes."""
    stage, body_path = _make_stage_with_body(tmp_path)
    body = stage.GetPrimAtPath(Sdf.Path(body_path))

    add_pose_velocity_trajectory(
        body,
        [(0.0, [0.0, 0.0, 0.0] + _identity_quat(), [1.0, 0.0, 0.0, 0.0, 0.0, 2.0])],
    )

    rb_api = UsdPhysics.RigidBodyAPI(body)
    assert bool(rb_api), "RigidBodyAPI must remain applied on the body"
    vel_attr = rb_api.GetVelocityAttr()
    angvel_attr = rb_api.GetAngularVelocityAttr()
    assert vel_attr is not None
    assert angvel_attr is not None
    assert vel_attr.GetTimeSamples() == [0.0]
    assert angvel_attr.GetTimeSamples() == [0.0]


def test_orient_uses_quatf_real_first(tmp_path: Path) -> None:
    """The orient op stores ``Quatf(real=qw, imaginary=Vec3f(qx,qy,qz))``;
    the read-back must reverse that into pose7's ``[qx,qy,qz,qw]`` order.
    A 90° rotation around Z exercises the channel mapping."""
    stage, body_path = _make_stage_with_body(tmp_path)
    body = stage.GetPrimAtPath(Sdf.Path(body_path))

    half = math.pi / 4
    qz = math.sin(half)
    qw = math.cos(half)
    pose7 = [0.0, 0.0, 0.0, 0.0, 0.0, qz, qw]
    add_pose_velocity_trajectory(body, [(0.0, pose7, [0.0] * 6)])

    out = read_pose_velocity_trajectory(stage, body_path)
    assert len(out) == 1
    _, pose_out, _ = out[0]
    assert pose_out[3] == pytest.approx(0.0, abs=1e-5)
    assert pose_out[4] == pytest.approx(0.0, abs=1e-5)
    assert pose_out[5] == pytest.approx(qz, abs=1e-5)
    assert pose_out[6] == pytest.approx(qw, abs=1e-5)


def test_read_returns_empty_when_no_time_samples(tmp_path: Path) -> None:
    stage, body_path = _make_stage_with_body(tmp_path)
    assert read_pose_velocity_trajectory(stage, body_path) == []


def test_read_defaults_missing_channels_and_rejects_bad_prims(tmp_path: Path) -> None:
    stage, body_path = _make_stage_with_body(tmp_path)
    body = stage.GetPrimAtPath(Sdf.Path(body_path))

    with pytest.raises(ValueError, match="not found"):
        read_pose_velocity_trajectory(stage, "/Missing")

    scope = stage.DefinePrim(Sdf.Path("/World/Scope"), "Scope")
    assert scope
    with pytest.raises(ValueError, match="not Xformable"):
        read_pose_velocity_trajectory(stage, "/World/Scope")

    orient = UsdGeom.Xformable(body).AddOrientOp()
    orient.Set(
        Gf.Quatf(0.5, Gf.Vec3f(0.1, 0.2, 0.3)),
        time=Usd.TimeCode(2.0),
    )
    assert read_pose_velocity_trajectory(stage, body_path) == [
        (
            pytest.approx(2.0 / stage.GetTimeCodesPerSecond()),
            [
                0.0,
                0.0,
                0.0,
                pytest.approx(0.1),
                pytest.approx(0.2),
                pytest.approx(0.3),
                pytest.approx(0.5),
            ],
            [0.0] * 6,
        )
    ]

    second_stage_dir = tmp_path / "second"
    second_stage_dir.mkdir()
    stage2, body_path2 = _make_stage_with_body(second_stage_dir)
    body2 = stage2.GetPrimAtPath(Sdf.Path(body_path2))
    UsdGeom.Xformable(body2).AddTranslateOp().Set(
        Gf.Vec3d(1.0, 2.0, 3.0),
        time=Usd.TimeCode(4.0),
    )
    assert read_pose_velocity_trajectory(stage2, body_path2) == [
        (
            pytest.approx(4.0 / stage2.GetTimeCodesPerSecond()),
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            [0.0] * 6,
        )
    ]


def test_rejects_wrong_arity(tmp_path: Path) -> None:
    stage, body_path = _make_stage_with_body(tmp_path)
    body = stage.GetPrimAtPath(Sdf.Path(body_path))

    with pytest.raises(ValueError, match="3-tuples"):
        add_pose_velocity_trajectory(
            body,
            [(0.0, [0.0, 1.0, 0.0] + _identity_quat())],  # type: ignore[list-item]
        )
    with pytest.raises(ValueError, match="vel6"):
        add_pose_velocity_trajectory(
            body,
            [(0.0, [0.0, 1.0, 0.0] + _identity_quat(), [0.0] * 5)],
        )


def test_rejects_non_xformable(tmp_path: Path) -> None:
    p = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    # Scope is NOT Xformable.
    scope_prim = stage.DefinePrim(Sdf.Path("/World"), "Scope")
    with pytest.raises(TypeError, match="Xformable"):
        add_pose_velocity_trajectory(
            scope_prim,
            [(0.0, [0.0, 0.0, 0.0] + _identity_quat(), [0.0] * 6)],
        )


def test_deprecated_matrix_time_sample_helpers(tmp_path: Path) -> None:
    stage, body_path = _make_stage_with_body(tmp_path)
    body = stage.GetPrimAtPath(Sdf.Path(body_path))
    pose = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]

    with pytest.raises(ValueError, match="pose7"):
        time_samples_utils._matrix_from_pose7([1.0, 2.0])
    with pytest.raises(TypeError, match="Matrix4d"):
        time_samples_utils._matrix_to_pose7(object())

    matrix = time_samples_utils._matrix_from_pose7(pose)
    assert time_samples_utils._matrix_to_pose7(matrix) == pytest.approx(pose)

    add_xform_time_sample(body, 3.0, pose)
    add_xform_time_sample(body, 4.0, [2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0])

    samples = list(iter_time_samples(stage))
    assert [time for time, _poses in samples] == [3.0, 4.0]
    assert samples[0][1][body_path] == pytest.approx(pose)

    scope_prim = stage.DefinePrim(Sdf.Path("/World/NonXform"), "Scope")
    with pytest.raises(TypeError, match="Xformable"):
        add_xform_time_sample(scope_prim, 0.0, pose)


def test_iter_time_samples_skips_non_transform_and_none_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakePrim:
        def __init__(self, path: str, kind: str) -> None:
            self.path = path
            self.kind = kind

        def GetPath(self) -> str:
            return self.path

    class _FakeAttr:
        def GetTimeSamples(self) -> list[float]:
            return [1.0]

        def Get(self, timecode: Usd.TimeCode) -> None:
            return None

    class _FakeOp:
        def __init__(self, op_type: object) -> None:
            self._op_type = op_type

        def GetOpType(self) -> object:
            return self._op_type

        def GetAttr(self) -> _FakeAttr:
            return _FakeAttr()

    class _FakeXformable:
        def __init__(self, prim: _FakePrim) -> None:
            self.prim = prim

        def __bool__(self) -> bool:
            return self.prim.kind != "not_xformable"

        def GetOrderedXformOps(self) -> list[_FakeOp]:
            if self.prim.kind == "translate_only":
                return [_FakeOp(UsdGeom.XformOp.TypeTranslate)]
            return [_FakeOp(UsdGeom.XformOp.TypeTransform)]

    class _FakeStage:
        def Traverse(self) -> list[_FakePrim]:
            return [
                _FakePrim("/Scope", "not_xformable"),
                _FakePrim("/Translate", "translate_only"),
                _FakePrim("/NoneTransform", "transform_none"),
            ]

    monkeypatch.setattr(UsdGeom, "Xformable", _FakeXformable)

    assert list(iter_time_samples(_FakeStage())) == []


def test_read_back_returns_seconds_after_fps_authoring(tmp_path: Path) -> None:
    """Recorder authors at frame timecodes (``t * fps``) and sets
    ``timeCodesPerSecond=fps``; ``read_pose_velocity_trajectory`` must
    return wall-clock seconds, not frame indices, so downstream metrics
    that key on `times` (settle_time, duration_s) get the right scale.
    """
    from physics_agent.recording.recorder import author_trajectory_usda

    stage, body_path = _make_stage_with_body(tmp_path)
    # Save scene to disk so the recorder can open it.
    scene_path = tmp_path / "scene.usda"
    stage.GetRootLayer().Export(str(scene_path))

    fps = 30
    trajectory = [
        (
            i * 0.1,
            [0.0, 1.0 - i * 0.05, 0.0] + _identity_quat(),
            [float(i), 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        for i in range(5)
    ]
    recording_path = tmp_path / "recording.usda"
    author_trajectory_usda(scene_path, trajectory, body_path, recording_path, fps=fps)

    rec_stage = Usd.Stage.Open(str(recording_path))
    assert rec_stage.GetTimeCodesPerSecond() == pytest.approx(float(fps))

    out = read_pose_velocity_trajectory(rec_stage, body_path)
    assert len(out) == len(trajectory)
    # Times come back in seconds, NOT frames. With fps=30, the samples
    # come back at the recorder's quantized seconds (round(t*fps)/fps),
    # not at frame indices 0, 3, 6, 9, 12.
    expected_times = [round(i * 0.1 * fps) / fps for i in range(5)]
    actual_times = [t for t, _, _ in out]
    for got, want in zip(actual_times, expected_times, strict=True):
        assert got == pytest.approx(want, abs=1e-6)
    # Pose AND velocity values must round-trip too — not just timecode
    # arity. A regression that zeroed translate/velocity time samples
    # would otherwise pass an arity-only check.
    for (_t_in, pose_in, vel_in), (_t_out, pose_out, vel_out) in zip(
        trajectory, out, strict=True
    ):
        for a, b in zip(pose_in, pose_out, strict=True):
            assert b == pytest.approx(a, abs=1e-5)
        for a, b in zip(vel_in, vel_out, strict=True):
            assert b == pytest.approx(a, abs=1e-5)


def test_metric_function_reads_back_same_numbers(tmp_path: Path) -> None:
    """``max_linear_speed`` reads identical numbers from a recording USD
    and from the daemon's in-flight dict — the recording is the
    persisted view of raw simulator output."""
    from world_understanding.functions.physics.trajectory import max_linear_speed

    stage, body_path = _make_stage_with_body(tmp_path)
    body = stage.GetPrimAtPath(Sdf.Path(body_path))

    trajectory = [
        (
            i * 0.1,
            [0.0, 1.0, 0.0] + _identity_quat(),
            [float(i), 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        for i in range(5)
    ]
    add_pose_velocity_trajectory(body, trajectory)

    expected = max_linear_speed(trajectory)
    persisted = read_pose_velocity_trajectory(stage, body_path)
    assert max_linear_speed(persisted) == pytest.approx(expected, abs=1e-5)
