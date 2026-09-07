# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, exact undo/redo coverage for camera-rig authoring history."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pxr import Gf, Usd, UsdGeom
from usd_core.camera_analysis.contracts import CameraPose
from usd_core.session import Session


def _session(tmp_path) -> Session:
    path = tmp_path / "history.usda"
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 0.1))
    UsdGeom.Camera.Define(stage, "/World/Existing")
    stage.GetRootLayer().Save()
    return Session.open(path)


@pytest.fixture(autouse=True)
def _fixed_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    from usd_core.camera_analysis import newton_backend, placement

    monkeypatch.setattr(
        newton_backend,
        "require_qualified_backend_versions",
        lambda: {"newton": "1.5.0", "warp": "1.16.0", "qualified": True},
    )
    monkeypatch.setattr(
        newton_backend,
        "NewtonVisibilityBackend",
        lambda *_args, **_kwargs: object(),
    )

    def place(scene, *_args, **_kwargs):
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
            "schema": "usd-cli.camera-placement.v1",
            "method": "max_coverage",
            "source_digest": scene.source_digest,
            "seed": 0,
            "scope_path": "/World/Floor",
            "cameras": [{"id": "camera-001", "look_at": [0.0, 0.0, 0.0]}],
            "passed": True,
            "selected_count": 1,
            "stop_reason": "target_reached",
            "achieved_coverage": 1.0,
        }
        return SimpleNamespace(poses=(pose,), report=report)

    monkeypatch.setattr(placement, "place_max_coverage", place)


def _place(session: Session, path: str, *, on_existing: str = "error"):
    response = session.camera_place(
        "max_coverage",
        scope="/World/Floor",
        target_coverage=0.5,
        max_cameras=1,
        candidates=4,
        grid=4,
        author_under=path,
        on_existing=on_existing,
    )
    assert response.ok, response.issues
    return response


def _assert_exact_round_trip(
    session: Session,
    *,
    root_before: str,
    session_before: str,
    root_after: str,
    session_after: str,
    edit_target: str,
    active_camera: str | None,
) -> None:
    other_layer = (
        session._stage.GetSessionLayer()
        if edit_target == "root"
        else session._stage.GetRootLayer()
    )
    session._stage.SetEditTarget(other_layer)
    session._active_cam = None

    undone = session.undo()
    assert undone.ok, undone.issues
    assert session._stage.GetRootLayer().ExportToString() == root_before
    assert session._stage.GetSessionLayer().ExportToString() == session_before
    assert session._stage_edit_layer_name() == edit_target
    assert session._active_cam == active_camera

    session._stage.SetEditTarget(other_layer)
    session._active_cam = None
    redone = session.redo()
    assert redone.ok, redone.issues
    assert session._stage.GetRootLayer().ExportToString() == root_after
    assert session._stage.GetSessionLayer().ExportToString() == session_after
    assert session._stage_edit_layer_name() == edit_target
    assert session._active_cam == active_camera


def test_new_rig_history_retains_only_the_targeted_subtree(tmp_path) -> None:
    session = _session(tmp_path)
    unrelated = UsdGeom.Xform.Define(session._stage, "/World/Unrelated")
    unrelated.GetPrim().SetCustomDataByKey("largePayload", "x" * 100_000)
    root_before = session._stage.GetRootLayer().ExportToString()
    session_before = session._stage.GetSessionLayer().ExportToString()
    active_camera = session._active_cam

    _place(session, "/World/Rig")

    root_after = session._stage.GetRootLayer().ExportToString()
    session_after = session._stage.GetSessionLayer().ExportToString()
    operation = session.history.entries()[0]
    assert operation.command == "camera.place"
    undo_change = operation.inverse["undo"]
    redo_change = operation.inverse["redo"]
    assert undo_change["kind"] == redo_change["kind"] == "layer_subtree_snapshot"
    assert undo_change["path"] == redo_change["path"] == "/World/Rig"
    assert undo_change["fragment"] is None
    assert "root" not in undo_change and "session" not in undo_change
    retained_bytes = sum(
        len(change["fragment"] or "") for change in (undo_change, redo_change)
    )
    assert retained_bytes < len(root_before) // 20

    _assert_exact_round_trip(
        session,
        root_before=root_before,
        session_before=session_before,
        root_after=root_after,
        session_after=session_after,
        edit_target="root",
        active_camera=active_camera,
    )


def test_replace_round_trips_session_layer_order_target_and_active_camera(
    tmp_path,
) -> None:
    session = _session(tmp_path)
    root_before = session._stage.GetRootLayer().ExportToString()
    session._stage.SetEditTarget(session._stage.GetSessionLayer())
    UsdGeom.Xform.Define(session._stage, "/World/A")
    old_rig = UsdGeom.Xform.Define(session._stage, "/World/Rig")
    old_rig.GetPrim().SetCustomDataByKey("identity", "old-rig")
    UsdGeom.Camera.Define(session._stage, "/World/Rig/LegacyCamera")
    UsdGeom.Xform.Define(session._stage, "/World/Z")
    session_before = session._stage.GetSessionLayer().ExportToString()
    active_camera = session._active_cam

    response = _place(session, "/World/Rig", on_existing="replace")

    assert response.summary["rig"] == "/World/Rig"
    root_after = session._stage.GetRootLayer().ExportToString()
    session_after = session._stage.GetSessionLayer().ExportToString()
    operation = session.history.entries()[0]
    assert operation.inverse["undo"]["layer"] == "session"
    assert operation.inverse["redo"]["layer"] == "session"

    _assert_exact_round_trip(
        session,
        root_before=root_before,
        session_before=session_before,
        root_after=root_after,
        session_after=session_after,
        edit_target="session",
        active_camera=active_camera,
    )
    child_names = [
        child.name
        for child in session._stage.GetSessionLayer()
        .GetPrimAtPath("/World")
        .nameChildren
    ]
    assert child_names == ["A", "Z", "Rig"]


def test_append_history_tracks_the_resolved_rig_path(tmp_path) -> None:
    session = _session(tmp_path)
    UsdGeom.Xform.Define(session._stage, "/World/Rig")
    root_before = session._stage.GetRootLayer().ExportToString()
    session_before = session._stage.GetSessionLayer().ExportToString()
    active_camera = session._active_cam

    response = _place(session, "/World/Rig", on_existing="append")

    assert response.summary["rig"] == "/World/Rig_2"
    root_after = session._stage.GetRootLayer().ExportToString()
    session_after = session._stage.GetSessionLayer().ExportToString()
    operation = session.history.entries()[0]
    assert operation.inverse["undo"]["path"] == "/World/Rig_2"
    assert operation.inverse["redo"]["path"] == "/World/Rig_2"
    _assert_exact_round_trip(
        session,
        root_before=root_before,
        session_before=session_before,
        root_after=root_after,
        session_after=session_after,
        edit_target="root",
        active_camera=active_camera,
    )


def test_instance_ancestor_opinion_is_part_of_targeted_history(tmp_path) -> None:
    session = _session(tmp_path)
    session._stage.CreateClassPrim("/_RigPrototype")
    instance = session._stage.DefinePrim("/World/Instance", "Xform")
    instance.GetReferences().AddInternalReference("/_RigPrototype")
    instance.SetInstanceable(True)
    root_before = session._stage.GetRootLayer().ExportToString()
    session_before = session._stage.GetSessionLayer().ExportToString()
    active_camera = session._active_cam

    _place(session, "/World/Instance/Rig")

    assert not session._stage.GetPrimAtPath("/World/Instance").IsInstance()
    root_after = session._stage.GetRootLayer().ExportToString()
    session_after = session._stage.GetSessionLayer().ExportToString()
    operation = session.history.entries()[0]
    undo_instanceable = operation.inverse["undo"]["instanceable"]
    redo_instanceable = operation.inverse["redo"]["instanceable"]
    assert undo_instanceable == [
        {"path": "/World/Instance", "authored": True, "value": True}
    ]
    assert redo_instanceable == [
        {"path": "/World/Instance", "authored": True, "value": False}
    ]

    _assert_exact_round_trip(
        session,
        root_before=root_before,
        session_before=session_before,
        root_after=root_after,
        session_after=session_after,
        edit_target="root",
        active_camera=active_camera,
    )
