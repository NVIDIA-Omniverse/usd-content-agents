# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``physics_agent.recording.recorder``.

The recorder authors a time-sampled USD recording from a daemon-shaped trajectory
of ``(t, pose7, vel6)`` 3-tuples (issue #50). Exercises:

* Source ``scene_usd`` is opened in memory and exported to ``output_path``;
  the on-disk source is never modified.
* ``timeCodesPerSecond`` / ``framesPerSecond`` / start+endTimeCode all
  reflect the ``fps`` argument and the trajectory bounds.
* Legacy ``(t, pose7)`` 2-tuples are rejected with a clear message — the
  daemon emits velocity per sample now and there is no finite-difference
  fallback.
* Trajectories longer than the duration cap are truncated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")  # noqa: F841
from pxr import (  # type: ignore[import-untyped]  # noqa: E402
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
)

from physics_agent.recording.recorder import (  # noqa: E402
    author_trajectory_jsonl,
    author_trajectory_usda,
)


def _identity_quat() -> list[float]:
    return [0.0, 0.0, 0.0, 1.0]


def _scene_usd(tmp_path: Path) -> tuple[Path, str]:
    """Author a minimal scene USD with a single rigid-body Xform."""
    p = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/World")
    body = UsdGeom.Xform.Define(stage, "/World/Body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.GetRootLayer().Save()
    return p, "/World/Body"


def _trajectory(n: int = 3, *, dt: float = 0.5) -> list:
    """N samples at ``dt`` spacing — well under the 2.0s recording cap
    (``n=3, dt=0.5`` → t in {0.0, 0.5, 1.0})."""
    return [
        (
            i * dt,
            [0.0, 1.0 - i * 0.1, 0.0] + _identity_quat(),
            [0.0, -float(i), 0.0, 0.0, 0.0, 0.0],
        )
        for i in range(n)
    ]


def test_recording_authored_with_time_samples(tmp_path: Path) -> None:
    scene, body_path = _scene_usd(tmp_path)
    out = tmp_path / "recording.usda"

    result = author_trajectory_usda(scene, _trajectory(), body_path, out, fps=30)
    assert result == out
    assert out.exists()

    rec = Usd.Stage.Open(str(out))
    body = rec.GetPrimAtPath(Sdf.Path(body_path))
    xformable = UsdGeom.Xformable(body)
    op_types = [op.GetOpType() for op in xformable.GetOrderedXformOps()]
    # Split form (translate + orient), no matrix.
    assert UsdGeom.XformOp.TypeTranslate in op_types
    assert UsdGeom.XformOp.TypeOrient in op_types
    assert UsdGeom.XformOp.TypeTransform not in op_types

    # Velocity attrs are time-sampled and from RigidBodyAPI.
    rb_api = UsdPhysics.RigidBodyAPI(body)
    assert bool(rb_api)
    assert rb_api.GetVelocityAttr().GetTimeSamples()
    assert rb_api.GetAngularVelocityAttr().GetTimeSamples()


def test_source_scene_usd_unmodified(tmp_path: Path) -> None:
    """``author_trajectory_usda`` must export to ``output_path`` only —
    the source file is opened in-memory and never written back."""
    scene, body_path = _scene_usd(tmp_path)
    before = scene.read_bytes()

    out = tmp_path / "recording.usda"
    author_trajectory_usda(scene, _trajectory(), body_path, out, fps=30)

    after = scene.read_bytes()
    assert before == after


def test_fps_and_timecodes_set(tmp_path: Path) -> None:
    """Stage timecode metadata reflects ``fps`` and trajectory bounds."""
    scene, body_path = _scene_usd(tmp_path)
    out = tmp_path / "recording.usda"

    author_trajectory_usda(scene, _trajectory(n=3, dt=0.5), body_path, out, fps=30)

    rec = Usd.Stage.Open(str(out))
    assert rec.GetTimeCodesPerSecond() == pytest.approx(30.0)
    assert rec.GetFramesPerSecond() == pytest.approx(30.0)
    # Frames at t in {0.0, 0.5, 1.0} → frame indices {0, 15, 30}.
    assert rec.GetStartTimeCode() == pytest.approx(0.0)
    assert rec.GetEndTimeCode() == pytest.approx(30.0)


def test_two_tuple_trajectory_rejected(tmp_path: Path) -> None:
    """Legacy ``(t, pose7)`` shape must error — finite-differencing
    velocity downstream would silently corrupt ``physics:velocity``
    samples."""
    scene, body_path = _scene_usd(tmp_path)
    out = tmp_path / "recording.usda"
    legacy = [
        (0.0, [0.0, 1.0, 0.0] + _identity_quat()),
        (1.0, [0.0, 0.5, 0.0] + _identity_quat()),
    ]
    with pytest.raises(ValueError, match="3-tuples"):
        author_trajectory_usda(scene, legacy, body_path, out)  # type: ignore[arg-type]


def test_trajectory_truncated_at_duration_cap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Issue #50 caps recording duration at 2.0s; samples beyond are
    dropped and a warning is emitted. The boundary sample at exactly
    t == 2.0 SURVIVES because the truncation predicate is
    ``t > _DEFAULT_MAX_DURATION_S`` (strict).
    """
    scene, body_path = _scene_usd(tmp_path)
    out = tmp_path / "recording.usda"

    # 4s of samples at 0.5s spacing — t in {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5}.
    long_traj = [
        (i * 0.5, [0.0, 1.0, 0.0] + _identity_quat(), [0.0] * 6) for i in range(8)
    ]
    # Pin caplog to the actual emitting logger. ``physics_agent.cli``'s
    # CLI tests (run earlier in the same session) install a logging
    # config that sets ``logger.propagate = False`` on the
    # ``physics_agent`` parent logger, after which caplog's root-attached
    # handler never sees ``physics_agent.recording.recorder`` records and
    # the assertion below mis-fires (CodeRabbit Round 11 bring-up of the
    # CI flake).
    import logging as _logging

    rec_logger = _logging.getLogger("physics_agent.recording.recorder")
    with caplog.at_level("WARNING", logger=rec_logger.name):
        rec_logger.addHandler(caplog.handler)
        try:
            author_trajectory_usda(scene, long_traj, body_path, out, fps=30)
        finally:
            rec_logger.removeHandler(caplog.handler)

    rec = Usd.Stage.Open(str(out))
    # 2.0s @ 30fps == frame 60. The boundary sample is included; everything
    # past it is dropped, so EndTimeCode equals 60 exactly.
    assert rec.GetEndTimeCode() == pytest.approx(60.0)
    # Five daemon samples should survive: t in {0.0, 0.5, 1.0, 1.5, 2.0} →
    # frame indices {0, 15, 30, 45, 60} → 5 distinct timecodes on the
    # translate op.
    body = rec.GetPrimAtPath(Sdf.Path(body_path))
    translate_op = next(
        op
        for op in UsdGeom.Xformable(body).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    )
    assert len(translate_op.GetAttr().GetTimeSamples()) == 5
    assert any("truncated" in record.message for record in caplog.records)


def test_trajectory_respects_explicit_duration_cap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Scenario evaluators can extend recording evidence to their full duration."""
    scene, body_path = _scene_usd(tmp_path)
    out = tmp_path / "recording.usda"
    long_traj = [
        (i * 0.5, [0.0, 1.0, 0.0] + _identity_quat(), [0.0] * 6) for i in range(8)
    ]

    import logging as _logging

    rec_logger = _logging.getLogger("physics_agent.recording.recorder")
    with caplog.at_level("WARNING", logger=rec_logger.name):
        rec_logger.addHandler(caplog.handler)
        try:
            author_trajectory_usda(
                scene,
                long_traj,
                body_path,
                out,
                fps=30,
                max_duration_s=3.0,
            )
        finally:
            rec_logger.removeHandler(caplog.handler)

    rec = Usd.Stage.Open(str(out))
    assert rec.GetEndTimeCode() == pytest.approx(90.0)
    body = rec.GetPrimAtPath(Sdf.Path(body_path))
    translate_op = next(
        op
        for op in UsdGeom.Xformable(body).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    )
    assert len(translate_op.GetAttr().GetTimeSamples()) == 7
    assert any("3.00s duration cap" in record.message for record in caplog.records)


def test_invalid_body_prim_path_rejected(tmp_path: Path) -> None:
    scene, _ = _scene_usd(tmp_path)
    out = tmp_path / "recording.usda"
    with pytest.raises(ValueError, match="not present"):
        author_trajectory_usda(scene, _trajectory(), "/World/Missing", out)


def test_empty_trajectory_rejected(tmp_path: Path) -> None:
    scene, body_path = _scene_usd(tmp_path)
    out = tmp_path / "recording.usda"
    with pytest.raises(ValueError, match="empty"):
        author_trajectory_usda(scene, [], body_path, out)


def _multi_mesh_scene_usd(tmp_path: Path) -> tuple[Path, str]:
    """Author a scene with the canonical multi-mesh rigid-body topology:
    a parent Xform carrying ``RigidBodyAPI`` and three static Mesh
    leaves underneath. Mirrors the structure ``apply_physics`` produces
    for monolithic assets like the ladder (one body + N colliders).
    """
    p = tmp_path / "multi_mesh_scene.usda"
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/World")
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body)
    # Static child meshes — local translates only, no time samples.
    for i, lx in enumerate((-1.5, 0.0, 1.5)):
        cube_path = f"/World/Body/Cube{i}"
        cube = UsdGeom.Cube.Define(stage, cube_path).GetPrim()
        UsdPhysics.CollisionAPI.Apply(cube)
        xf = UsdGeom.Xformable(cube)
        translate_op = xf.AddTranslateOp()
        from pxr import Gf  # type: ignore[import-untyped]

        translate_op.Set(Gf.Vec3d(lx, 0.0, 0.0))
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.GetRootLayer().Save()
    return p, "/World/Body"


def test_recording_on_parent_xform_with_child_meshes(tmp_path: Path) -> None:
    """The recorder authors time-sampled translate+orient on whatever
    ``body_prim_path`` resolves to — including a non-leaf Xform whose
    descendants are Mesh leaves. Per OpenUSD's transform-composition
    rules, leaf world transforms inherit the parent's time samples;
    the no-bug investigation at
    ``~/ovrtx_parent_xform_animation_no_bug.md`` empirically confirms
    ovrtx 0.2.0 + Hydra render this case correctly.

    This is the topology ``apply_physics`` produces for monolithic
    assets (one ``RigidBodyAPI`` on the asset's default prim, N
    ``CollisionAPI`` Mesh children). The regression test pins the
    contract so future code that switches to a flat-only authoring
    model gets caught here rather than silently breaking dropped-asset
    recordings.
    """
    scene, body_path = _multi_mesh_scene_usd(tmp_path)
    out = tmp_path / "recording.usda"

    author_trajectory_usda(scene, _trajectory(n=3, dt=0.5), body_path, out, fps=30)
    rec = Usd.Stage.Open(str(out))

    # Time samples land on the parent body, NOT on the leaf meshes.
    body = rec.GetPrimAtPath(Sdf.Path(body_path))
    body_xf_ops = {
        op.GetOpType(): op for op in UsdGeom.Xformable(body).GetOrderedXformOps()
    }
    assert UsdGeom.XformOp.TypeTranslate in body_xf_ops
    assert UsdGeom.XformOp.TypeOrient in body_xf_ops
    body_translate_samples = (
        body_xf_ops[UsdGeom.XformOp.TypeTranslate].GetAttr().GetTimeSamples()
    )
    assert len(body_translate_samples) == 3

    for i in range(3):
        cube = rec.GetPrimAtPath(f"/World/Body/Cube{i}")
        assert cube.IsValid(), f"child cube {i} missing in recording"
        for op in UsdGeom.Xformable(cube).GetOrderedXformOps():
            assert not op.GetAttr().GetTimeSamples(), (
                f"child {cube.GetPath()} should be static; recorder must not "
                "author time samples on leaves when body_prim is a parent."
            )

    # XformCache composes parent + child local transforms into the
    # expected world position at each authored timecode. _trajectory
    # writes body translate (0, 1.0 - 0.1*i, 0); cubes are at local
    # x in {-1.5, 0.0, 1.5}. World == body + child.local.
    cache = UsdGeom.XformCache()
    for i, t_seconds in enumerate((0.0, 0.5, 1.0)):
        cache.SetTime(Usd.TimeCode(t_seconds * 30))
        body_world = cache.GetLocalToWorldTransform(body).ExtractTranslation()
        expected_body_y = 1.0 - 0.1 * i
        assert body_world[1] == pytest.approx(expected_body_y, abs=1e-5)
        for j, lx in enumerate((-1.5, 0.0, 1.5)):
            cube = rec.GetPrimAtPath(f"/World/Body/Cube{j}")
            cube_world = cache.GetLocalToWorldTransform(cube).ExtractTranslation()
            assert cube_world[0] == pytest.approx(lx, abs=1e-5)
            assert cube_world[1] == pytest.approx(expected_body_y, abs=1e-5)


# ---------------------------------------------------------------------------
# author_trajectory_jsonl — judge-readable companion to the USD recording
# ---------------------------------------------------------------------------


def test_jsonl_writes_one_line_per_quantized_frame(tmp_path: Path) -> None:
    """Frame quantization mirrors author_trajectory_usda. With three
    samples at t in {0.0, 0.5, 1.0} and fps=30, frames {0, 15, 30}
    survive — three lines, each parses, fields are exactly
    {frame, t, pose, vel} with the right lengths.
    """
    import json

    out = tmp_path / "trajectory.jsonl"
    result = author_trajectory_jsonl(_trajectory(n=3, dt=0.5), out, fps=30)
    assert result == out
    assert out.exists()

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    parsed = [json.loads(line) for line in lines]
    assert [p["frame"] for p in parsed] == [0, 15, 30]
    assert parsed[0]["t"] == pytest.approx(0.0)
    assert parsed[1]["t"] == pytest.approx(0.5)
    assert parsed[2]["t"] == pytest.approx(1.0)
    for p in parsed:
        assert set(p.keys()) == {"frame", "t", "pose", "vel"}
        assert len(p["pose"]) == 7  # px,py,pz,qx,qy,qz,qw
        assert len(p["vel"]) == 6  # vx,vy,vz,wx,wy,wz


def test_jsonl_two_tuple_trajectory_rejected(tmp_path: Path) -> None:
    """Same legacy-format rejection as the USD writer — the validator
    is shared via _validate_and_truncate."""
    out = tmp_path / "trajectory.jsonl"
    legacy = [
        (0.0, [0.0, 1.0, 0.0] + _identity_quat()),
        (1.0, [0.0, 0.5, 0.0] + _identity_quat()),
    ]
    with pytest.raises(ValueError, match="3-tuples"):
        author_trajectory_jsonl(legacy, out)  # type: ignore[arg-type]


def test_jsonl_truncated_at_duration_cap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Issue #50's 2.0s duration cap applies to JSONL too. With samples
    at t in {0.0..3.5} the boundary t=2.0 sample SURVIVES (strict ``>``)
    and t > 2.0 are dropped — five lines, frames {0, 15, 30, 45, 60}.
    """
    import json

    out = tmp_path / "trajectory.jsonl"
    long_traj = [
        (i * 0.5, [0.0, 1.0, 0.0] + _identity_quat(), [0.0] * 6) for i in range(8)
    ]
    # See test_trajectory_truncated_at_duration_cap above for the
    # propagate=False / caplog-handler attachment rationale.
    import logging as _logging

    rec_logger = _logging.getLogger("physics_agent.recording.recorder")
    with caplog.at_level("WARNING", logger=rec_logger.name):
        rec_logger.addHandler(caplog.handler)
        try:
            author_trajectory_jsonl(long_traj, out, fps=30)
        finally:
            rec_logger.removeHandler(caplog.handler)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    frames = [json.loads(line)["frame"] for line in lines]
    assert frames == [0, 15, 30, 45, 60]
    assert any("truncated" in record.message for record in caplog.records)


def test_jsonl_respects_explicit_duration_cap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import json

    out = tmp_path / "trajectory.jsonl"
    long_traj = [
        (i * 0.5, [0.0, 1.0, 0.0] + _identity_quat(), [0.0] * 6) for i in range(8)
    ]

    import logging as _logging

    rec_logger = _logging.getLogger("physics_agent.recording.recorder")
    with caplog.at_level("WARNING", logger=rec_logger.name):
        rec_logger.addHandler(caplog.handler)
        try:
            author_trajectory_jsonl(
                long_traj,
                out,
                fps=30,
                max_duration_s=3.0,
            )
        finally:
            rec_logger.removeHandler(caplog.handler)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 7
    frames = [json.loads(line)["frame"] for line in lines]
    assert frames == [0, 15, 30, 45, 60, 75, 90]
    assert any("3.00s duration cap" in record.message for record in caplog.records)


def test_jsonl_empty_trajectory_rejected(tmp_path: Path) -> None:
    out = tmp_path / "trajectory.jsonl"
    with pytest.raises(ValueError, match="empty"):
        author_trajectory_jsonl([], out)


def test_jsonl_and_usda_cover_same_frames(tmp_path: Path) -> None:
    """USD and JSONL artifacts must align frame-for-frame so a consumer
    reading either gets the same sample grid. Authored with the same
    fps on the same trajectory."""
    import json

    scene, body_path = _scene_usd(tmp_path)
    usd_path = tmp_path / "recording.usda"
    jsonl_path = tmp_path / "trajectory.jsonl"

    traj = _trajectory(n=3, dt=0.5)
    author_trajectory_usda(scene, traj, body_path, usd_path, fps=30)
    author_trajectory_jsonl(traj, jsonl_path, fps=30)

    rec = Usd.Stage.Open(str(usd_path))
    body = rec.GetPrimAtPath(Sdf.Path(body_path))
    translate_op = next(
        op
        for op in UsdGeom.Xformable(body).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    )
    usd_frames = sorted(int(s) for s in translate_op.GetAttr().GetTimeSamples())

    jsonl_frames = [
        json.loads(line)["frame"]
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]

    assert usd_frames == jsonl_frames
