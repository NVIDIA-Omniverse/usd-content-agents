# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cooperative cancellation boundaries for camera-analysis jobs."""

from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pxr import Usd, UsdGeom
from typer.testing import CliRunner

from usd_cli import main
from usd_core.camera_analysis.cancellation import (
    CameraAnalysisCancelled,
    cancellation_scope,
    check_cancelled,
    defer_cancellation,
)
from usd_core.camera_analysis.coverage import GridSurface
from usd_core.config import Config
from usd_core.models import Response
from usd_core.session import Session
from usd_server import app as server_app

_HEADERS = {"x-usd-cli-token": "camera-cancel-token"}


def _request(http: TestClient, command: str, payload: dict | None = None) -> dict:
    response = http.post(
        "/cmd",
        json={"command": command, "payload": payload or {}},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, session_type: type):
    monkeypatch.setattr(server_app, "Session", session_type)
    return server_app.build_app(
        Config(project_dir=tmp_path),
        token=_HEADERS["x-usd-cli-token"],
        instance_id="camera-cancellation-test",
    )


def _memory_session(tmp_path: Path) -> Session:
    session = Session(Config(project_dir=tmp_path))
    session._stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(session._stage, "/World")
    session._stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Cube.Define(session._stage, "/World/Floor")
    session._index_prims()
    return session


def test_cancellation_scope_is_typed_and_resets() -> None:
    event = threading.Event()
    with cancellation_scope(event):
        check_cancelled()
        event.set()
        with pytest.raises(CameraAnalysisCancelled, match="camera analysis cancelled"):
            check_cancelled()
    check_cancelled()


def test_cancellation_can_be_deferred_through_an_atomic_boundary() -> None:
    event = threading.Event()
    event.set()
    with cancellation_scope(event):
        with defer_cancellation():
            check_cancelled()
        with pytest.raises(CameraAnalysisCancelled):
            check_cancelled()


def test_visibility_polygon_extraction_checks_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from usd_core.camera_analysis import coverage

    rows = columns = 128
    mask = np.indices((rows, columns)).sum(axis=0) % 2 == 0
    grid = GridSurface(
        rows=rows,
        columns=columns,
        x_min_m=0.0,
        x_max_m=float(columns),
        y_min_m=0.0,
        y_max_m=float(rows),
        points_m=np.zeros((rows, columns, 3), dtype=np.float64),
        accessible=np.ones((rows, columns), dtype=bool),
    )
    event = threading.Event()
    original_check = coverage.check_cancelled
    calls = 0

    def cancel_during_extraction() -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            event.set()
        original_check()

    monkeypatch.setattr(coverage, "check_cancelled", cancel_during_extraction)
    with cancellation_scope(event), pytest.raises(CameraAnalysisCancelled):
        coverage.visibility_regions(mask, grid)
    assert 4 <= calls <= 5
    assert event.is_set()


def test_queued_cancel_uses_future_cancel_and_never_dispatches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSession:
        calls = 0

        def __init__(self, _config, name: str = "default") -> None:
            self.name = name
            self.read_only = False
            self._stage_path = None

        def camera_place(self, method: str, preview: bool = False) -> Response:
            del method, preview
            type(self).calls += 1
            return Response(command="camera.place")

    app = _app(monkeypatch, tmp_path, FakeSession)
    release = threading.Event()
    blocker_started = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def block_worker() -> None:
        blocker_started.set()
        assert release.wait(timeout=10.0)

    pool.submit(block_worker)
    assert blocker_started.wait(timeout=2.0)
    app.state.job_pool = pool
    try:
        with TestClient(app) as http:
            detached = _request(
                http,
                "camera.place",
                {"method": "max_coverage", "preview": True, "detach": True},
            )
            job = detached["summary"]["job"]
            cancelled = _request(http, "cancel", {"job": job})
            assert cancelled["ok"] is True
            assert cancelled["summary"] == {
                "job": job,
                "state": "cancelled",
                "cancel_requested": True,
                "queued": True,
            }
            waited = _request(http, "wait", {"job": job, "timeout": 5})
            assert waited["ok"] is False
            assert waited["summary"]["error_type"] == "cancelled"
            listed = _request(http, "jobs")
            row = next(item for item in listed["data"]["jobs"] if item["job"] == job)
            assert row["state"] == "cancelled"
            late = _request(http, "cancel", {"job": job})
            assert late["summary"]["late_cancel_ignored"] is True
            assert FakeSession.calls == 0
            assert server_app._IN_FLIGHT[0] == 0
            assert http.get("/health", headers=_HEADERS).json()["busy"] is False
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_running_loop_cancellation_is_cancelled_not_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started = threading.Event()
    continue_to_next_candidate = threading.Event()

    class FakeSession:
        candidate_visits: list[int] = []
        stage_state = "unchanged"
        history_state: tuple = ()
        output_state = "absent"

        def __init__(self, _config, name: str = "default") -> None:
            self.name = name
            self.read_only = False
            self._stage_path = None

        def camera_place(self, method: str, preview: bool = False) -> Response:
            del method, preview
            for candidate in range(2):
                check_cancelled()
                type(self).candidate_visits.append(candidate)
                if candidate == 0:
                    started.set()
                    assert continue_to_next_candidate.wait(timeout=10.0)
            return Response(command="camera.place")

    app = _app(monkeypatch, tmp_path, FakeSession)
    with TestClient(app) as http:
        detached = _request(
            http,
            "camera.place",
            {"method": "max_coverage", "preview": True, "detach": True},
        )
        job = detached["summary"]["job"]
        assert started.wait(timeout=2.0)
        cancelling = _request(http, "cancel", {"job": job})
        assert cancelling["summary"]["state"] == "cancelling"
        pending = next(
            item
            for item in _request(http, "jobs")["data"]["jobs"]
            if item["job"] == job
        )
        assert pending["state"] == "cancelling"
        continue_to_next_candidate.set()
        waited = _request(http, "wait", {"job": job, "timeout": 5})
        assert waited["ok"] is False
        assert waited["summary"]["error_type"] == "cancelled"
        row = next(
            item
            for item in _request(http, "jobs")["data"]["jobs"]
            if item["job"] == job
        )

    assert row["state"] == "cancelled"
    assert FakeSession.candidate_visits == [0]
    assert FakeSession.stage_state == "unchanged"
    assert FakeSession.history_state == ()
    assert FakeSession.output_state == "absent"


def test_max_coverage_stops_before_the_next_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from usd_core.camera_analysis import placement

    surface = GridSurface(
        rows=2,
        columns=2,
        x_min_m=-1.0,
        x_max_m=1.0,
        y_min_m=-1.0,
        y_max_m=1.0,
        points_m=np.asarray(
            [
                [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]],
                [[-0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
            ],
            dtype=np.float64,
        ),
        accessible=np.ones((2, 2), dtype=bool),
    )
    monkeypatch.setattr(placement, "sample_surface", lambda *_args, **_kwargs: surface)
    monkeypatch.setattr(
        placement,
        "_geometry_clearance_mask",
        lambda poses, _backend: np.ones(len(poses), dtype=bool),
    )
    event = threading.Event()
    visits = 0

    def visibility(*_args, **_kwargs) -> np.ndarray:
        nonlocal visits
        visits += 1
        event.set()
        return np.ones((2, 2), dtype=bool)

    monkeypatch.setattr(placement, "visibility_mask", visibility)
    with cancellation_scope(event), pytest.raises(CameraAnalysisCancelled):
        placement.place_max_coverage(
            SimpleNamespace(),
            SimpleNamespace(device="cpu"),
            (np.asarray([-1.0, -1.0, 0.0]), np.asarray([1.0, 1.0, 0.1])),
            scope_path="/World/Floor",
            grid=2,
            candidate_count=4,
        )
    assert visits == 1


@pytest.mark.usefixtures("qualified_warp_runtime")
def test_greedy_warp_scoring_stops_before_the_next_gain_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from usd_core.camera_analysis import warp_scoring

    event = threading.Event()
    wp, _kernels = warp_scoring._lazy_runtime()
    original_synchronize = wp.synchronize_device
    synchronizations = 0

    def synchronize(device=None) -> None:
        nonlocal synchronizations
        original_synchronize(device)
        synchronizations += 1
        # Packing, two initial reductions, covered-count, gain scoring, then the
        # first selected-mask update. Cancellation is observed immediately after
        # that safe sync and before another gain iteration can launch.
        if synchronizations == 6:
            event.set()

    monkeypatch.setattr(wp, "synchronize_device", synchronize)
    masks = np.asarray(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    with cancellation_scope(event), pytest.raises(CameraAnalysisCancelled):
        warp_scoring.greedy_select_masks(
            masks,
            np.ones(4, dtype=bool),
            per_cell=1,
            max_cameras=3,
            target_coverage=1.0,
            minimum_gain=0.0,
            device="cpu",
        )
    assert synchronizations == 6


@pytest.mark.usefixtures("qualified_camera_analysis_metadata")
def test_cancel_immediately_before_authoring_leaves_no_state_or_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from usd_core.camera_analysis import authoring, newton_backend, placement, scene

    session = _memory_session(tmp_path)
    root_before = session._stage.GetRootLayer().ExportToString()
    session_before = session._stage.GetSessionLayer().ExportToString()
    history_before = session.history.snapshot_state()
    epoch_before = session._mutation_epoch
    output = tmp_path / "cancelled-placement.json"
    cancel_event = threading.Event()
    report = {
        "schema": "usd-cli.camera-placement.v1",
        "backend": {"qualified": True},
        "cameras": [{}],
        "passed": True,
        "selected_count": 1,
        "stop_reason": "target_reached",
        "achieved_coverage": 1.0,
    }
    evaluation = SimpleNamespace(poses=(object(),), report=report)
    monkeypatch.setattr(
        scene, "build_scene_analysis_ir", lambda _stage, **_kwargs: object()
    )
    monkeypatch.setattr(
        scene,
        "analysis_bounds",
        lambda _scene, **_kwargs: (np.zeros(3), np.ones(3)),
    )
    monkeypatch.setattr(
        newton_backend, "NewtonVisibilityBackend", lambda *_args, **_kwargs: object()
    )

    def finish_analysis(*_args, **_kwargs):
        cancel_event.set()
        return evaluation

    monkeypatch.setattr(placement, "place_max_coverage", finish_analysis)
    monkeypatch.setattr(
        authoring,
        "author_camera_rig",
        lambda *_args, **_kwargs: pytest.fail("cancelled request must not author"),
    )

    with cancellation_scope(cancel_event), pytest.raises(CameraAnalysisCancelled):
        session.camera_place(
            method="max_coverage",
            scope="/World/Floor",
            author_under="/World/Rig",
            output=str(output),
        )

    assert session._stage.GetRootLayer().ExportToString() == root_before
    assert session._stage.GetSessionLayer().ExportToString() == session_before
    assert session.history.snapshot_state() == history_before
    assert session._mutation_epoch == epoch_before
    assert not session._stage.GetPrimAtPath("/World/Rig").IsValid()
    assert not output.exists()


def test_completed_job_ignores_late_cancel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSession:
        def __init__(self, _config, name: str = "default") -> None:
            self.name = name
            self.read_only = False
            self._stage_path = None

        def camera_place(self, method: str, preview: bool = False) -> Response:
            del method, preview
            return Response(command="camera.place", summary={"completed": True})

    with TestClient(_app(monkeypatch, tmp_path, FakeSession)) as http:
        detached = _request(
            http,
            "camera.place",
            {"method": "max_coverage", "preview": True, "detach": True},
        )
        job = detached["summary"]["job"]
        completed = _request(http, "wait", {"job": job, "timeout": 5})
        assert completed["ok"] is True
        late = _request(http, "cancel", {"job": job})
    assert late["summary"] == {
        "job": job,
        "state": "done",
        "cancel_requested": False,
        "late_cancel_ignored": True,
    }


def test_cancel_cli_dispatches_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []

    def dispatch(command: str, payload: dict) -> Response:
        calls.append((command, payload))
        return Response(command=command)

    monkeypatch.setattr(main, "dispatch", dispatch)
    result = CliRunner().invoke(main.app, ["cancel", "j7"])
    assert result.exit_code == 0
    assert calls == [("cancel", {"job": "j7"})]
