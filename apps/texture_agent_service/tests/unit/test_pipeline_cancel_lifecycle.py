# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancellation route lifecycle regressions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ...service.routers import pipeline_router
from ...service.runtime import bus as bus_module
from ...service.session.manager import SessionManager


class _EmptyRegistry:
    def __init__(self) -> None:
        self.cancel_called = False

    def is_running(self, session_id: str) -> bool:
        return False

    async def cancel(self, session_id: str) -> bool:
        self.cancel_called = True
        return False


def _build_pipeline_client(manager: SessionManager) -> TestClient:
    app = FastAPI()
    pipeline_router.set_session_manager(manager)
    app.include_router(pipeline_router.router)
    return TestClient(app)


@pytest.mark.parametrize("status", ["running", "cancelling"])
def test_cancel_remote_worker_writes_shared_marker_without_local_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    """POST /cancel must work when another process owns the worker task."""
    sid = f"remote-worker-{status}"
    manager = SessionManager(tmp_path / "sessions", ttl_hours=2)
    manager.create_session(sid)
    manager.update_session(sid, {"status": status})
    bus_module._event_bus = None
    bus = bus_module.init_event_bus(manager)

    registry = _EmptyRegistry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)

    client = _build_pipeline_client(manager)

    with manager.worker_lock(sid):
        response = client.post(f"/pipeline/{sid}/cancel")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelling"
    assert registry.cancel_called is False
    assert manager.is_cancelled(sid) is True

    metadata = manager.get_session_metadata(sid)
    assert metadata is not None
    assert metadata["status"] == "cancelling"

    snapshot = bus.get_snapshot(sid)
    assert snapshot is not None
    assert snapshot["status"] == "cancelling"


def _seed_terminal_session_with_cancel_marker(
    storage_path: Path,
    session_id: str,
    *,
    terminal_status: str,
) -> SessionManager:
    """Create a session in a terminal state with a stale `.cancel` marker.

    Mirrors the on-disk shape produced by request_cancellation followed by
    natural completion: the durable cancellation marker is left behind even
    though the worker stopped, and config.yaml is present so /regenerate
    can load it.
    """
    manager = SessionManager(storage_path, ttl_hours=2)
    session_dir = manager.create_session(session_id)
    manager.update_session(session_id, {"status": terminal_status})
    (session_dir / "input").mkdir(parents=True, exist_ok=True)
    (session_dir / "input" / "scene.usd").write_text("#usda 1.0\n", encoding="utf-8")
    config_path = session_dir / "input" / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"session_id": session_id},
                "input": {"usd_path": "scene.usd"},
                "steps": {
                    "prepare_uvs": {"enabled": True},
                    "discover_materials": {"enabled": True},
                    "generate_prompts": {"enabled": True},
                    "render_previews": {"enabled": False},
                    "generate_textures": {"enabled": True},
                    "blend_textures": {"enabled": True},
                    "apply_textures": {"enabled": True},
                    "render": {"enabled": False},
                },
            }
        )
    )
    manager.request_cancellation(session_id)
    manager.update_session(session_id, {"status": terminal_status})
    assert manager.is_cancelled(session_id) is True
    return manager


@pytest.mark.parametrize("terminal_status", ["cancelled", "failed", "completed"])
def test_regenerate_clears_stale_cancel_marker_before_register(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    """Accepted regenerate must clear the prior run's `.cancel` before register.

    Otherwise the executor's between-step is_cancelled() checkpoint sees the
    stale marker and immediately cancels the new run.
    """
    sid = f"regenerate-stale-cancel-{terminal_status}"
    manager = _seed_terminal_session_with_cancel_marker(
        tmp_path, sid, terminal_status=terminal_status
    )
    observed: dict[str, bool] = {}

    class _StubRegistry:
        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            observed["cancel_marker_when_registered"] = manager.is_cancelled(session_id)
            coro.close()
            if on_finished is not None:
                on_finished()

    class _StubBus:
        def clear_session_state(self, *args: Any, **kwargs: Any) -> None:
            return None

        async def seed_pending_session(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _StubBus())

    client = _build_pipeline_client(manager)
    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
    )

    assert response.status_code == 202, response.text
    assert observed["cancel_marker_when_registered"] is False
    assert manager.is_cancelled(sid) is False


@pytest.mark.parametrize("terminal_status", ["cancelled", "failed", "completed"])
def test_create_existing_session_clears_stale_cancel_marker_before_register(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    """Reusing an existing session via POST /pipeline must clear stale cancel."""
    sid = f"create-stale-cancel-{terminal_status}"
    manager = _seed_terminal_session_with_cancel_marker(
        tmp_path, sid, terminal_status=terminal_status
    )
    observed: dict[str, bool] = {}

    class _StubRegistry:
        def is_running(self, session_id: str) -> bool:
            return False

        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            observed["cancel_marker_when_registered"] = manager.is_cancelled(session_id)
            coro.close()
            if on_finished is not None:
                on_finished()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())

    client = _build_pipeline_client(manager)
    response = client.post("/pipeline", data={"session_id": sid})

    assert response.status_code == 202, response.text
    assert observed["cancel_marker_when_registered"] is False
    assert manager.is_cancelled(sid) is False


def test_reset_session_for_new_run_fresh_clears_bookkeeping(
    tmp_path: Path,
) -> None:
    """`fresh=True` must zero progress bookkeeping the executor would otherwise see."""
    sid = "reset-fresh-clears-bookkeeping"
    manager = _seed_terminal_session_with_cancel_marker(
        tmp_path, sid, terminal_status="failed"
    )
    manager.update_session(
        sid,
        {
            "completed_steps": ["prepare_uvs", "discover_materials"],
            "preview_images": ["preview_a.png", "preview_b.png"],
            "overall_progress": {
                "current_step": 3,
                "total_steps": 8,
                "percent": 37,
                "estimated_remaining_seconds": 90,
            },
        },
    )
    bus_module._event_bus = None
    bus_module.init_event_bus(manager)

    pipeline_router._reset_session_for_new_run(manager, sid, fresh=True)

    metadata = manager.get_session_metadata(sid)
    assert metadata is not None
    assert metadata["status"] == "pending"
    assert metadata["completed_steps"] == []
    assert metadata["preview_images"] == []
    assert metadata["overall_progress"] == {
        "current_step": 0,
        "total_steps": 9,
        "percent": 0,
        "estimated_remaining_seconds": None,
    }
    assert metadata.get("error") is None
    assert metadata.get("failed_step") is None
    assert manager.is_cancelled(sid) is False


def test_reset_session_for_new_run_regenerate_keeps_bookkeeping(
    tmp_path: Path,
) -> None:
    """`fresh=False` (regenerate) preserves completed_steps and progress."""
    sid = "reset-regenerate-keeps-bookkeeping"
    manager = _seed_terminal_session_with_cancel_marker(
        tmp_path, sid, terminal_status="failed"
    )
    completed = ["prepare_uvs", "discover_materials"]
    previews = ["preview_a.png"]
    progress = {
        "current_step": 3,
        "total_steps": 8,
        "percent": 37,
        "estimated_remaining_seconds": 90,
    }
    manager.update_session(
        sid,
        {
            "completed_steps": completed,
            "preview_images": previews,
            "overall_progress": progress,
        },
    )
    bus_module._event_bus = None
    bus_module.init_event_bus(manager)

    pipeline_router._reset_session_for_new_run(manager, sid, fresh=False)

    metadata = manager.get_session_metadata(sid)
    assert metadata is not None
    assert metadata["status"] == "pending"
    assert metadata["completed_steps"] == completed
    assert metadata["preview_images"] == previews
    assert metadata["overall_progress"] == progress
    assert manager.is_cancelled(sid) is False


@pytest.mark.parametrize("terminal_status", ["cancelled", "failed", "completed"])
def test_create_existing_session_cancellable_while_queued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    """An accepted retry of a terminal session must be cancellable before it runs.

    Without the run-state reset the cancel route reads the persisted terminal
    status, returns 400, and the queued retry runs to completion despite the
    user's cancel request — losing the cancel contract entirely.
    """
    sid = f"create-queued-cancellable-{terminal_status}"
    manager = _seed_terminal_session_with_cancel_marker(
        tmp_path, sid, terminal_status=terminal_status
    )

    class _QueuedRegistry:
        def __init__(self) -> None:
            self.cancel_called = False

        def is_running(self, session_id: str) -> bool:
            return False

        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            coro.close()
            if on_finished is not None:
                on_finished()

        async def cancel(self, session_id: str) -> bool:
            self.cancel_called = True
            return False

    registry = _QueuedRegistry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)

    bus_module._event_bus = None
    bus_module.init_event_bus(manager)

    client = _build_pipeline_client(manager)
    accept = client.post("/pipeline", data={"session_id": sid})
    assert accept.status_code == 202, accept.text

    cancel = client.post(f"/pipeline/{sid}/cancel")
    assert cancel.status_code == 200, cancel.text

    metadata = manager.get_session_metadata(sid)
    assert metadata is not None
    assert metadata["status"] == "cancelling"
    assert manager.is_cancelled(sid) is True


def test_create_existing_session_register_failure_restores_prior_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A register failure after reset must restore the prior terminal view.

    Without rollback the reused session would be stuck in `pending` with
    completed_steps/preview_images/overall_progress/error/failed_step
    all wiped — and no executor coroutine ever scheduled.
    """
    sid = "create-existing-register-fails"
    manager = _seed_terminal_session_with_cancel_marker(
        tmp_path, sid, terminal_status="failed"
    )
    prior_metadata = {
        "status": "failed",
        "error": "old failure",
        "failed_step": "generate_textures",
        "failed_step_stats": {"old": True},
        "failed_at": "2026-04-30T00:00:00+00:00",
        "partial_results": {"old": True},
        "completed_steps": ["prepare_uvs", "discover_materials"],
        "preview_images": ["preview_a.png"],
        "overall_progress": {
            "current_step": 3,
            "total_steps": 8,
            "percent": 37,
            "estimated_remaining_seconds": 90,
        },
    }
    manager.update_session(sid, prior_metadata)
    bus_module._event_bus = None
    bus_module.init_event_bus(manager)

    class _FailingRegistry:
        def is_running(self, session_id: str) -> bool:
            return False

        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            coro.close()
            raise RuntimeError("synthetic register failure")

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _FailingRegistry())

    client = _build_pipeline_client(manager)
    with pytest.raises(RuntimeError, match="synthetic register failure"):
        client.post("/pipeline", data={"session_id": sid})

    metadata = manager.get_session_metadata(sid)
    assert metadata is not None
    for key, value in prior_metadata.items():
        assert metadata[key] == value, f"{key} not restored"
    assert manager.is_worker_active(sid) is False


def test_create_existing_session_pre_reset_failure_leaves_prior_state_intact(
    tmp_path: Path,
) -> None:
    """A failure before the reset point must NOT mutate prior session state.

    Locks in the deferred-reset invariant: validation/config-write failures
    happen with the original metadata still in place, so the user sees the
    prior terminal view unchanged.
    """
    sid = "create-pre-reset-failure"
    manager = _seed_terminal_session_with_cancel_marker(
        tmp_path, sid, terminal_status="failed"
    )
    prior_metadata = {
        "status": "failed",
        "error": "old failure",
        "failed_step": "generate_textures",
        "failed_step_stats": {"old": True},
        "completed_steps": ["prepare_uvs"],
        "preview_images": ["preview_a.png"],
    }
    manager.update_session(sid, prior_metadata)
    bus_module._event_bus = None
    bus_module.init_event_bus(manager)

    session_dir = manager.get_session_dir(sid)
    for ext in (".usd", ".usda", ".usdc", ".usdz"):
        candidate = session_dir / "input" / f"scene{ext}"
        if candidate.exists():
            candidate.unlink()

    client = _build_pipeline_client(manager)
    response = client.post("/pipeline", data={"session_id": sid})
    assert response.status_code == 400, response.text
    assert "Input USD not found" in response.json()["detail"]

    metadata = manager.get_session_metadata(sid)
    assert metadata is not None
    for key, value in prior_metadata.items():
        assert metadata[key] == value, f"{key} mutated by pre-reset failure"
    assert manager.is_cancelled(sid) is True


def test_regenerate_pre_reset_failure_leaves_prior_state_intact(
    tmp_path: Path,
) -> None:
    """Regenerate failures before the reset point must NOT mutate prior state."""
    sid = "regenerate-pre-reset-failure"
    manager = _seed_terminal_session_with_cancel_marker(
        tmp_path, sid, terminal_status="failed"
    )
    prior_metadata = {
        "status": "failed",
        "error": "old failure",
        "failed_step": "generate_textures",
        "failed_step_stats": {"old": True},
        "completed_steps": ["prepare_uvs"],
        "preview_images": ["preview_a.png"],
    }
    manager.update_session(sid, prior_metadata)
    bus_module._event_bus = None
    bus_module.init_event_bus(manager)

    config_path = manager.get_session_dir(sid) / "input" / "config.yaml"
    config_path.unlink()

    client = _build_pipeline_client(manager)
    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
    )
    assert response.status_code == 400, response.text
    assert "Original config not found" in response.json()["detail"]

    metadata = manager.get_session_metadata(sid)
    assert metadata is not None
    for key, value in prior_metadata.items():
        assert metadata[key] == value, f"{key} mutated by pre-reset failure"
    assert manager.is_cancelled(sid) is True


def test_regenerate_clears_cancel_marker_under_worker_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clear must happen with the worker lock held so peers cannot race.

    A second SessionManager pointing at the same storage path is denied the
    worker lock for the entire window from acquire through register, which
    also covers the clear_cancellation call performed inside that window.
    """
    sid = "regenerate-cancel-cleared-under-lock"
    manager = _seed_terminal_session_with_cancel_marker(
        tmp_path, sid, terminal_status="cancelled"
    )
    peer = SessionManager(tmp_path, ttl_hours=2)
    observed: dict[str, bool] = {}

    class _StubRegistry:
        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            try:
                with peer.worker_lock(session_id, timeout=0):
                    observed["peer_acquired_during_register"] = True
            except Exception:
                observed["peer_acquired_during_register"] = False
            observed["cancel_marker_when_registered"] = manager.is_cancelled(session_id)
            coro.close()
            if on_finished is not None:
                on_finished()

    class _StubBus:
        def clear_session_state(self, *args: Any, **kwargs: Any) -> None:
            return None

        async def seed_pending_session(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _StubBus())

    client = _build_pipeline_client(manager)
    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
    )

    assert response.status_code == 202, response.text
    assert observed["cancel_marker_when_registered"] is False
    assert observed["peer_acquired_during_register"] is False
