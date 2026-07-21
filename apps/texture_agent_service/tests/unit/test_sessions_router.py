# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from ...service.routers import pipeline_router, sessions_router
from ...service.runtime import ProgressEvent, StepState
from ...service.runtime import bus as bus_module
from ...service.runtime.bus import get_event_bus, init_event_bus
from ...service.session.manager import SessionManager


def _build_session_app(tmp_path: Path) -> tuple[TestClient, SessionManager]:
    manager = SessionManager(tmp_path)
    init_event_bus(manager)
    pipeline_router.set_session_manager(manager)
    sessions_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    app.include_router(sessions_router.router)
    return TestClient(app), manager


def test_delete_missing_session_returns_json_error(tmp_path: Path) -> None:
    client, _ = _build_session_app(tmp_path)

    response = client.delete("/sessions/missing-session")

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"
    assert response.json()["detail"] == "Session not found"


def test_invalid_session_id_routes_return_not_found(tmp_path: Path) -> None:
    client, manager = _build_session_app(tmp_path)

    requests = [
        ("GET", "/sessions/%2E%2E"),
        ("DELETE", "/sessions/%2E%2E"),
        ("GET", "/pipeline/%2E%2E/status"),
        ("GET", "/pipeline/%2E%2E/results"),
        ("POST", "/pipeline/%2E%2E/cancel"),
        ("GET", "/pipeline/%2E%2E/event-log"),
    ]

    for method, path in requests:
        response = client.request(method, path)
        assert response.status_code == 404

    assert manager.storage_path.exists()


def test_delete_session_openapi_documents_json_errors(tmp_path: Path) -> None:
    client, _ = _build_session_app(tmp_path)
    responses = client.app.openapi()["paths"]["/sessions/{session_id}"]["delete"][
        "responses"
    ]

    for status_code in ("404", "409", "500"):
        content = responses[status_code]["content"]

        assert list(content) == ["application/json"]
        assert content["application/json"]["schema"] == {}


def test_delete_session_clears_runtime_status_snapshot(tmp_path: Path) -> None:
    client, manager = _build_session_app(tmp_path)
    sid = "completed-session"
    manager.create_session(sid)
    asyncio.run(
        get_event_bus().emit(
            ProgressEvent(
                session_id=sid,
                step="render",
                state=StepState.RUNNING,
                percent=50,
            )
        )
    )

    assert client.get(f"/pipeline/{sid}/status").status_code == 200

    response = client.delete(f"/sessions/{sid}")

    assert response.status_code == 204
    assert get_event_bus().get_snapshot(sid) is None
    assert client.get(f"/pipeline/{sid}/status").status_code == 404


def test_delete_running_session_returns_conflict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, manager = _build_session_app(tmp_path)
    sid = "running-session"
    manager.create_session(sid)

    class RunningJobRegistry:
        def is_running(self, session_id: str) -> bool:
            return session_id == sid

    monkeypatch.setattr(
        sessions_router,
        "get_job_registry",
        lambda: RunningJobRegistry(),
    )

    response = client.delete(f"/sessions/{sid}")

    assert response.status_code == 409
    assert response.headers["content-type"] == "application/json"
    assert response.json()["detail"] == (
        "Cannot delete an active session. Cancel it and wait for the worker "
        "to stop before deleting."
    )
    assert manager.session_exists(sid) is True


def test_delete_worker_locked_session_returns_conflict(tmp_path: Path) -> None:
    client, manager = _build_session_app(tmp_path)
    sid = "worker-locked-session"
    manager.create_session(sid)

    with manager.worker_lock(sid):
        response = client.delete(f"/sessions/{sid}")

    assert response.status_code == 409
    assert response.headers["content-type"] == "application/json"
    assert response.json()["detail"] == (
        "Cannot delete an active session. A worker is still writing artifacts "
        "for this session."
    )
    assert manager.session_exists(sid) is True


def test_delete_corrupt_stalled_marker_returns_conflict(tmp_path: Path) -> None:
    client, manager = _build_session_app(tmp_path)
    sid = "corrupt-stalled-session"
    manager.create_session(sid)
    marker_path = manager.get_session_dir(sid) / ".worker.stalled"
    marker_path.write_text("{", encoding="utf-8")

    response = client.delete(f"/sessions/{sid}")

    assert response.status_code == 409
    assert response.headers["content-type"] == "application/json"
    assert response.json()["detail"] == (
        "Cannot delete an active session. A worker is still writing artifacts "
        "for this session."
    )
    assert marker_path.exists() is True
    assert manager.session_exists(sid) is True


def test_delete_stale_cancelling_session_succeeds(tmp_path: Path) -> None:
    client, manager = _build_session_app(tmp_path)
    sid = "cancelling-session"
    manager.create_session(sid)
    asyncio.run(
        get_event_bus().emit(
            ProgressEvent(
                session_id=sid,
                step="render",
                state=StepState.RUNNING,
                percent=50,
            )
        )
    )
    manager.update_session(sid, {"status": "cancelling"})

    response = client.delete(f"/sessions/{sid}")

    assert response.status_code == 204
    assert manager.session_exists(sid) is False
    assert get_event_bus().get_snapshot(sid) is None


def test_session_detail_normalizes_non_list_completed_steps(tmp_path: Path) -> None:
    client, manager = _build_session_app(tmp_path)
    sid = "non-list-completed-steps"
    manager.create_session(sid)
    manager.update_session(
        sid,
        {
            "completed_steps": {"bad": "shape"},
            "status": "failed",
            "error": "failed under /tmp/internal/path",
            "failed_step_stats": {"message": "see /tmp/internal/path"},
            "partial_results": {"errors": [{"message": "secret path"}]},
            "results": {"warnings": ["secret path"]},
        },
    )

    response = client.get(f"/sessions/{sid}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["completed_steps"] == []
    assert payload["status"] == "failed"


def test_build_session_view_handles_missing_metadata_and_bad_dates() -> None:
    class ViewManager:
        def __init__(self, metadata: dict[str, Any] | None) -> None:
            self.metadata = metadata

        def session_exists(self, session_id: str) -> bool:
            return True

        def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
            return self.metadata

    old_manager = sessions_router.session_manager
    old_bus = bus_module._event_bus
    try:
        sessions_router.set_session_manager(ViewManager(None))  # type: ignore[arg-type]
        init_event_bus(None)
        assert sessions_router._build_session_view("missing-metadata") is None

        sessions_router.set_session_manager(
            ViewManager(
                {
                    "session_id": "bad-date",
                    "status": "pending",
                    "created_at": "not-a-date",
                    "updated_at": "not-a-date",
                    "config": {},
                }
            )
        )  # type: ignore[arg-type]
        view = sessions_router._build_session_view("bad-date")
        assert view is not None
        assert view["created_at"] == "not-a-date"
    finally:
        sessions_router.session_manager = old_manager
        bus_module._event_bus = old_bus


def test_session_summary_list_filters_invalid_rows_and_bad_dates(
    tmp_path: Path,
) -> None:
    class MetadataListManager:
        def list_session_metadata(self) -> list[dict[str, Any] | None]:
            return [
                None,
                {"session_id": 123, "status": "pending"},
                {
                    "session_id": "terminal",
                    "status": "failed",
                    "created_at": "not-a-date",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "config": {"original_filename": "scene.usd"},
                },
                {
                    "session_id": "active",
                    "status": "pending",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "config": {},
                },
            ]

    old_manager = sessions_router.session_manager
    old_bus = bus_module._event_bus
    try:
        sessions_router.set_session_manager(MetadataListManager())  # type: ignore[arg-type]
        bus = init_event_bus(None)
        bus._state["terminal"] = {"status": "running"}
        bus._state["active"] = {"status": "running"}

        summaries = sessions_router._build_session_summary_list(
            sessions_router.get_session_manager()
        )

        status_by_id = {summary.session_id: summary.status for summary in summaries}
        assert status_by_id == {"active": "running", "terminal": "failed"}
    finally:
        sessions_router.session_manager = old_manager
        bus_module._event_bus = old_bus


async def test_delete_session_retries_then_returns_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NeverDeletesManager:
        def __init__(self) -> None:
            self.delete_calls = 0

        def session_exists(self, _session_id: str) -> bool:
            return True

        def is_worker_active(self, _session_id: str) -> bool:
            return False

        def delete_session(self, _session_id: str) -> bool:
            self.delete_calls += 1
            return False

    class EmptyRegistry:
        def is_running(self, _session_id: str) -> bool:
            return False

    async def no_sleep(_seconds: float) -> None:
        return None

    manager = NeverDeletesManager()
    old_manager = sessions_router.session_manager
    try:
        sessions_router.set_session_manager(manager)  # type: ignore[arg-type]
        monkeypatch.setattr(
            sessions_router, "get_job_registry", lambda: EmptyRegistry()
        )
        monkeypatch.setattr(sessions_router.asyncio, "sleep", no_sleep)

        with pytest.raises(HTTPException) as exc:
            await sessions_router.delete_session("stuck")

        assert exc.value.status_code == 500
        assert manager.delete_calls == 3
    finally:
        sessions_router.session_manager = old_manager


async def test_delete_session_retry_races_to_active_or_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RaceManager:
        def __init__(self, *, active_after_failure: bool) -> None:
            self.active_after_failure = active_after_failure
            self.delete_calls = 0

        def session_exists(self, _session_id: str) -> bool:
            return self.delete_calls == 0 or self.active_after_failure

        def is_worker_active(self, _session_id: str) -> bool:
            return self.delete_calls > 0 and self.active_after_failure

        def delete_session(self, _session_id: str) -> bool:
            self.delete_calls += 1
            return False

    class EmptyRegistry:
        def is_running(self, _session_id: str) -> bool:
            return False

    async def no_sleep(_seconds: float) -> None:
        return None

    old_manager = sessions_router.session_manager
    try:
        monkeypatch.setattr(
            sessions_router, "get_job_registry", lambda: EmptyRegistry()
        )
        monkeypatch.setattr(sessions_router.asyncio, "sleep", no_sleep)

        active_manager = RaceManager(active_after_failure=True)
        sessions_router.set_session_manager(active_manager)  # type: ignore[arg-type]
        with pytest.raises(HTTPException) as exc:
            await sessions_router.delete_session("active-race")
        assert exc.value.status_code == 409

        missing_manager = RaceManager(active_after_failure=False)
        sessions_router.set_session_manager(missing_manager)  # type: ignore[arg-type]
        assert await sessions_router.delete_session("missing-race") is None
    finally:
        sessions_router.session_manager = old_manager


async def test_list_sessions_offloads_store_reads_from_event_loop() -> None:
    event_loop_thread = threading.get_ident()

    class ThreadAssertingManager:
        def list_sessions(self) -> list[str]:
            assert threading.get_ident() != event_loop_thread
            return ["threaded-session"]

        def get_session_metadata(self, session_id: str) -> dict[str, Any]:
            assert threading.get_ident() != event_loop_thread
            return {
                "session_id": session_id,
                "status": "pending",
                "created_at": "2026-05-22T00:00:00+00:00",
                "updated_at": "2026-05-22T00:00:00+00:00",
                "config": {},
            }

    old_manager = sessions_router.session_manager
    old_bus = bus_module._event_bus
    try:
        sessions_router.set_session_manager(ThreadAssertingManager())  # type: ignore[arg-type]
        init_event_bus(None)

        response = await sessions_router.list_sessions()

        assert [session.session_id for session in response.sessions] == [
            "threaded-session"
        ]
    finally:
        sessions_router.session_manager = old_manager
        bus_module._event_bus = old_bus
