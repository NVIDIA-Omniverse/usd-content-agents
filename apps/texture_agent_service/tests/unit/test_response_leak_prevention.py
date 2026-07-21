# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end leak-prevention tests for the public response surface.

Drives a forced-failure pipeline run whose injected exception carries an
NVCF function-invocation URL and an absolute session-storage path, then
asserts neither substring appears in any of:

- the persisted session metadata (write-time scrubbing),
- the FAILED ``ProgressEvent`` (SSE),
- ``GET /pipeline/{session_id}/status`` and ``/results``,
- ``GET /sessions`` and ``/sessions/{session_id}``.

These tests pin both filesystem-path and NVCF-URL leak prevention.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ...service.routers import pipeline_router, sessions_router
from ...service.runtime import bus as bus_module
from ...service.session.manager import SessionManager
from ...service.workers import executor

NVCF_URL = "https://abc12345-def6-4789-9abc-def012345678.invocation.api.nvcf.nvidia.com/v2/nvcf/exec/functions/abc/versions/v1"
ABS_SESSION_PATH_PREFIX = "/var/texture-agent/sessions"


def _build_app(manager: SessionManager) -> FastAPI:
    app = FastAPI()
    pipeline_router.set_session_manager(manager)
    sessions_router.set_session_manager(manager)
    app.include_router(pipeline_router.router)
    app.include_router(sessions_router.router)
    return app


class _LeakyTask:
    """Stand-in that injects NVCF URLs + absolute session paths into both
    the per-unit error records and the top-level threshold-gate exception."""

    name = "GenerateTextures"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        context["generated_textures"] = {}
        context["generate_textures_errors"] = [
            {
                "material": "Steel",
                "type": "HTTPStatusError",
                "status": 500,
                "message": (
                    f"NVCF request failed with HTTP 500: Server error for url '{NVCF_URL}'"
                ),
            },
            {
                "material": "Wood",
                "type": "RuntimeError",
                "status": None,
                "message": (
                    "missing albedo at "
                    f"{ABS_SESSION_PATH_PREFIX}/abcdef/cache/textures/wood_albedo.png"
                ),
            },
        ]
        context["generate_textures_failed_count"] = 2
        context["generate_textures_attempted_count"] = 2
        raise RuntimeError(
            f"2/2 texture generation requests failed via {NVCF_URL}: HTTP 500"
        )


class GenerateTexturesTask:
    """Successful-enough step that still carries per-material failures."""

    name = "GenerateTextures"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        context["generated_textures"] = {"Steel": {"albedo": "steel.png"}}
        context["generate_textures_errors"] = [
            {
                "material": "Wood",
                "type": "HTTPStatusError",
                "status": 500,
                "message": f"HTTP 500 for url '{NVCF_URL}'",
            },
            {
                "material": "Glass",
                "type": "RuntimeError",
                "status": None,
                "message": (
                    "missing albedo at "
                    f"{ABS_SESSION_PATH_PREFIX}/partial/cache/textures/glass.png"
                ),
            },
        ]
        context["generate_textures_failed_count"] = 2
        context["generate_textures_attempted_count"] = 3
        return context


def _stub_factory(context, skip=None, only=None):
    return [_LeakyTask()]


def _partial_success_factory(context, skip=None, only=None):
    return [GenerateTexturesTask()]


@pytest.fixture
def manager(tmp_path: Path) -> SessionManager:
    return SessionManager(storage_path=tmp_path / "sessions", ttl_hours=24)


@pytest.fixture
def app(manager: SessionManager) -> FastAPI:
    return _build_app(manager)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _drive_failure(
    manager: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
    session_id: str = "leak-001",
) -> None:
    """Run the executor with a leaky stub task, persisting failure state."""
    # Ensure the executor's sanitizer reads the test session-root.
    monkeypatch.setattr(
        executor.service_config,
        "session_storage_path",
        str(manager.storage_path),
        raising=False,
    )

    manager.create_session(session_id, config={})
    session_dir = manager.get_session_dir(session_id)

    bus_module._event_bus = None
    bus = bus_module.init_event_bus(manager)

    captured: list[Any] = []
    original_emit = bus.emit

    async def capture(event):
        captured.append(event)
        await original_emit(event)

    bus.emit = capture  # type: ignore[method-assign]

    async def _run() -> None:
        with pytest.raises(RuntimeError):
            await executor._execute_pipeline_inner(
                session_id=session_id,
                config_dict={"input": {"usd_path": "/tmp/in.usd"}},
                session_manager=manager,
                event_bus=bus,
                session_dir=session_dir,
                only_steps=None,
                skip_steps=None,
                create_texture_pipeline_workflow=_stub_factory,
            )

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()

    # Stash the captured events on the manager for assertions.
    manager._captured_events = captured  # type: ignore[attr-defined]


def _drive_partial_success(
    manager: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
    session_id: str = "partial-live-001",
) -> None:
    """Run a completed step whose stats contain partial-failure diagnostics."""
    monkeypatch.setattr(
        executor.service_config,
        "session_storage_path",
        str(manager.storage_path),
        raising=False,
    )

    manager.create_session(session_id, config={})
    session_dir = manager.get_session_dir(session_id)

    bus_module._event_bus = None
    bus = bus_module.init_event_bus(manager)

    captured: list[Any] = []
    original_emit = bus.emit

    async def capture(event):
        captured.append(event)
        await original_emit(event)

    bus.emit = capture  # type: ignore[method-assign]

    async def _run() -> None:
        await executor._execute_pipeline_inner(
            session_id=session_id,
            config_dict={"input": {"usd_path": "/tmp/in.usd"}},
            session_manager=manager,
            event_bus=bus,
            session_dir=session_dir,
            only_steps=None,
            skip_steps=None,
            create_texture_pipeline_workflow=_partial_success_factory,
        )

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()

    manager._captured_events = captured  # type: ignore[attr-defined]


def _assert_clean(payload: object, where: str) -> None:
    text = repr(payload)
    assert "nvcf.nvidia.com" not in text, f"{where}: NVCF URL leaked"
    assert "abc12345" not in text, f"{where}: NVCF function id leaked"
    assert ABS_SESSION_PATH_PREFIX not in text, f"{where}: absolute session path leaked"


class TestWriteTimeScrubbing:
    def test_persisted_metadata_does_not_leak_nvcf_or_paths(
        self,
        manager: SessionManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _drive_failure(manager, monkeypatch)
        metadata = manager.get_session_metadata("leak-001")
        assert metadata is not None
        _assert_clean(metadata.get("error"), "metadata.error")
        _assert_clean(metadata.get("failed_step_stats"), "metadata.failed_step_stats")

    def test_failed_progress_event_does_not_leak(
        self,
        manager: SessionManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _drive_failure(manager, monkeypatch)
        captured = manager._captured_events  # type: ignore[attr-defined]
        failed = [
            ev
            for ev in captured
            if getattr(getattr(ev, "state", None), "value", None) == "failed"
        ]
        assert failed, "expected a FAILED ProgressEvent"
        ev = failed[-1]
        _assert_clean(ev.message, "failed_event.message")
        _assert_clean(ev.extra, "failed_event.extra")


class TestCompletedStepScrubbing:
    def test_completed_progress_event_does_not_leak_partial_errors(
        self,
        manager: SessionManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _drive_partial_success(manager, monkeypatch)
        captured = manager._captured_events  # type: ignore[attr-defined]
        completed_steps = [
            ev
            for ev in captured
            if getattr(getattr(ev, "state", None), "value", None) == "completed"
            and ev.step == "generate_textures"
            and not (ev.extra or {}).get("pipeline_completed")
        ]
        assert completed_steps, "expected a step COMPLETED ProgressEvent"
        _assert_clean(completed_steps[-1].extra, "completed_event.extra")

    def test_status_snapshot_does_not_leak_completed_step_stats(
        self,
        manager: SessionManager,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _drive_partial_success(manager, monkeypatch)

        snapshot = bus_module.get_event_bus().get_snapshot("partial-live-001")
        assert snapshot is not None
        _assert_clean(snapshot["completed_steps"], "event_bus.completed_steps")

        resp = client.get("/pipeline/partial-live-001/status")
        assert resp.status_code == 200
        _assert_clean(resp.json()["completed_steps"], "GET /pipeline/{id}/status")


class TestReadTimeScrubbing:
    """Older session.json files (written before the write-time scrubber
    landed) may still hold raw URLs / paths. Read-time sanitization is a
    second line of defense."""

    def _seed_dirty_metadata(self, manager: SessionManager, session_id: str) -> None:
        manager.create_session(
            session_id,
            config={
                "usd_path": f"{ABS_SESSION_PATH_PREFIX}/{session_id}/input/scene.usd"
            },
        )
        manager.update_session(
            session_id,
            {
                "status": "failed",
                "error": f"failed at {NVCF_URL}",
                "failed_step": "generate_textures",
                "completed_steps": [
                    {
                        "name": "generate_textures",
                        "display_name": "Generating PBR Textures",
                        "started_at": "2026-04-30T00:00:00+00:00",
                        "completed_at": "2026-04-30T00:00:03+00:00",
                        "duration_seconds": 3,
                        "stats": {
                            "textures_generated": 1,
                            "textures_failed": 1,
                            "errors": [
                                {
                                    "material": "Wood",
                                    "type": "HTTPStatusError",
                                    "status": 500,
                                    "message": f"HTTP 500 for url '{NVCF_URL}'",
                                },
                                {
                                    "material": "Glass",
                                    "type": "RuntimeError",
                                    "status": None,
                                    "message": (
                                        "missing at "
                                        f"{ABS_SESSION_PATH_PREFIX}/{session_id}"
                                        "/cache/textures/glass.png"
                                    ),
                                },
                            ],
                        },
                    }
                ],
                "failed_step_stats": {
                    "textures_generated": 0,
                    "textures_failed": 1,
                    "errors": [
                        {
                            "material": "Steel",
                            "type": "HTTPStatusError",
                            "status": 500,
                            "message": f"HTTP 500 for url '{NVCF_URL}'",
                        }
                    ],
                },
            },
        )

    def test_get_pipeline_status_sanitizes(
        self,
        manager: SessionManager,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ...service import config as config_module

        monkeypatch.setattr(
            config_module.config,
            "session_storage_path",
            str(manager.storage_path),
            raising=False,
        )
        self._seed_dirty_metadata(manager, "stale-001")
        resp = client.get("/pipeline/stale-001/status")
        assert resp.status_code == 200
        _assert_clean(resp.json(), "GET /pipeline/{id}/status")

    def test_get_pipeline_results_sanitizes(
        self,
        manager: SessionManager,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ...service import config as config_module

        monkeypatch.setattr(
            config_module.config,
            "session_storage_path",
            str(manager.storage_path),
            raising=False,
        )
        self._seed_dirty_metadata(manager, "stale-002")
        resp = client.get("/pipeline/stale-002/results")
        assert resp.status_code == 200
        _assert_clean(resp.json(), "GET /pipeline/{id}/results")

    def test_get_sessions_list_sanitizes(
        self,
        manager: SessionManager,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ...service import config as config_module

        monkeypatch.setattr(
            config_module.config,
            "session_storage_path",
            str(manager.storage_path),
            raising=False,
        )
        self._seed_dirty_metadata(manager, "stale-003")
        resp = client.get("/sessions")
        assert resp.status_code == 200
        # Old session.json with raw ``usd_path`` must not surface in the
        # whitelisted SessionConfigSummary.
        body = resp.json()
        assert body["total"] >= 1
        _assert_clean(body, "GET /sessions")

    def test_get_session_detail_sanitizes(
        self,
        manager: SessionManager,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ...service import config as config_module

        monkeypatch.setattr(
            config_module.config,
            "session_storage_path",
            str(manager.storage_path),
            raising=False,
        )
        self._seed_dirty_metadata(manager, "stale-004")
        resp = client.get("/sessions/stale-004")
        assert resp.status_code == 200
        body = resp.json()
        # Whitelist drops legacy ``usd_path``; sanitizer scrubs error +
        # failed_step_stats.
        assert "usd_path" not in body.get("config", {})
        _assert_clean(body, "GET /sessions/{id}")


class TestPartialSuccessLeakPrevention:
    """Threshold-not-hit runs finish as ``status=completed`` but persist
    per-unit ``errors`` records carrying ``str(httpx.HTTPStatusError)``.
    The leak surface is ``metadata.results.errors[<step>][*].message``
    -- echoed in ``GET /pipeline/{id}/results.stats`` and
    ``GET /sessions/{id}.results``."""

    def _seed_completed_with_partial_failures(
        self, manager: SessionManager, session_id: str
    ) -> None:
        manager.create_session(session_id, config={})
        manager.update_session(
            session_id,
            {
                "status": "completed",
                "results": {
                    "materials_found": 3,
                    "textures_generated": 2,
                    "textures_generated_failed": 1,
                    "textures_failed": 1,
                    "errors": {
                        "generate_textures": [
                            {
                                "material": "Steel",
                                "type": "HTTPStatusError",
                                "status": 500,
                                "message": (f"HTTP 500 for url '{NVCF_URL}'"),
                            },
                            {
                                "material": "Wood",
                                "type": "RuntimeError",
                                "status": None,
                                "message": (
                                    "missing albedo at "
                                    f"{ABS_SESSION_PATH_PREFIX}/{session_id}"
                                    "/cache/textures/wood_albedo.png"
                                ),
                            },
                        ]
                    },
                },
                "duration_seconds": 42,
                "completed_at": "2026-04-30T01:00:00+00:00",
            },
        )

    def test_get_pipeline_results_sanitizes_completed_partial(
        self,
        manager: SessionManager,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ...service import config as config_module

        monkeypatch.setattr(
            config_module.config,
            "session_storage_path",
            str(manager.storage_path),
            raising=False,
        )
        self._seed_completed_with_partial_failures(manager, "partial-001")
        resp = client.get("/pipeline/partial-001/results")
        assert resp.status_code == 200
        _assert_clean(resp.json(), "GET /pipeline/{id}/results (completed)")

    def test_get_session_detail_sanitizes_completed_partial(
        self,
        manager: SessionManager,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ...service import config as config_module

        monkeypatch.setattr(
            config_module.config,
            "session_storage_path",
            str(manager.storage_path),
            raising=False,
        )
        self._seed_completed_with_partial_failures(manager, "partial-002")
        resp = client.get("/sessions/partial-002")
        assert resp.status_code == 200
        _assert_clean(resp.json(), "GET /sessions/{id} (completed)")


class TestEventLogLeakPrevention:
    """``GET /pipeline/{id}/event-log`` reads ``event_log.jsonl``
    written by the event bus. Pre-fix sessions persisted raw
    ``ProgressEvent.message`` / ``extra`` containing NVCF URLs;
    sanitize at read time so legacy logs don't replay the leak."""

    def _seed_dirty_event_log(self, manager: SessionManager, session_id: str) -> None:
        import json

        manager.create_session(session_id, config={})
        log_file = manager.get_session_dir(session_id) / "event_log.jsonl"
        events = [
            {
                "session_id": session_id,
                "step": "generate_textures",
                "state": "running",
                "message": "Starting GenerateTextures",
                "extra": None,
            },
            {
                "session_id": session_id,
                "step": "generate_textures",
                "state": "failed",
                "message": (
                    f"NVCF request failed with HTTP 500: Server error for "
                    f"url '{NVCF_URL}'"
                ),
                "extra": {
                    "textures_failed": 2,
                    "errors": [
                        {
                            "material": "Steel",
                            "message": f"HTTP 500 for {NVCF_URL}",
                        },
                        {
                            "material": "Wood",
                            "message": (
                                "missing at "
                                f"{ABS_SESSION_PATH_PREFIX}/{session_id}"
                                "/cache/textures/wood_albedo.png"
                            ),
                        },
                    ],
                },
            },
        ]
        with open(log_file, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

    def test_get_event_log_sanitizes_legacy_entries(
        self,
        manager: SessionManager,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ...service import config as config_module

        monkeypatch.setattr(
            config_module.config,
            "session_storage_path",
            str(manager.storage_path),
            raising=False,
        )
        self._seed_dirty_event_log(manager, "log-001")
        resp = client.get("/pipeline/log-001/event-log")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        _assert_clean(body, "GET /pipeline/{id}/event-log")

    def test_get_event_log_passes_through_clean_entries(
        self,
        manager: SessionManager,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import json

        from ...service import config as config_module

        monkeypatch.setattr(
            config_module.config,
            "session_storage_path",
            str(manager.storage_path),
            raising=False,
        )
        manager.create_session("log-002", config={})
        log_file = manager.get_session_dir("log-002") / "event_log.jsonl"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "session_id": "log-002",
                        "step": "discover_materials",
                        "state": "completed",
                        "message": "Discovered 3 materials",
                        "extra": {"materials_found": 3},
                    }
                )
                + "\n"
            )
        resp = client.get("/pipeline/log-002/event-log")
        assert resp.status_code == 200
        body = resp.json()
        assert body["events"][0]["message"] == "Discovered 3 materials"
        assert body["events"][0]["extra"] == {"materials_found": 3}
