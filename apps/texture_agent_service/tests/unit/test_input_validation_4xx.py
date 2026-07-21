# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""4xx-hygiene regression tests for the Texture Agent service.

Covers three regressions against the public service surface:

* ``/pipeline/{sid}/regenerate`` accepted requests
  for steps that are disabled in the loaded session config (no rendering
  backend in the default docker-compose deploy disables ``render`` and
  ``render_previews``). The workflow factory silently dropped those
  steps, so the API returned 202 with no real work performed.
* The same endpoint accepted ``{"steps": []}``
  with HTTP 202 and an empty ``"Regenerating steps: "`` message instead
  of 422.
* ``POST /pipeline`` returned a plain 400 (and
  in the QA repro, dropped the TCP connection mid-multipart) when
  ``material_textures_json`` was malformed, instead of a structured 422
  matching FastAPI's request-validation contract used elsewhere.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from ...service.routers import pipeline_router
from ...service.session.manager import SessionManager


class _ChunkedUpload:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class _NamedUpload(_ChunkedUpload):
    def __init__(self, filename: str | None, chunks: list[bytes]) -> None:
        super().__init__(chunks)
        self.filename = filename


def _default_steps_disabling_render() -> dict[str, Any]:
    """Mirror of the defaults emitted by build_default_pipeline_config()."""
    return {
        "prepare_uvs": {"enabled": True},
        "discover_materials": {"enabled": True},
        "generate_prompts": {"enabled": True},
        "render_previews": {"enabled": False},
        "generate_textures": {"enabled": True},
        "blend_textures": {"enabled": True},
        "apply_textures": {"enabled": True},
        "render": {"enabled": False},
    }


async def test_stream_copy_removes_oversized_upload_after_close(
    tmp_path: Path,
) -> None:
    dest = tmp_path / "input.usda"
    upload = _ChunkedUpload([b"aaaa", b"bbbb"])

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._stream_copy(
            upload,
            dest,
            chunk_size=4,
            max_bytes=5,
        )

    assert exc_info.value.status_code == 413
    assert not dest.exists()


def test_upload_usd_oversize_removes_created_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    client = _build_test_client(manager)
    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 1)

    response = client.post(
        "/pipeline/upload-usd",
        files={
            "usd_file": (
                "scene.usd",
                b"#usda 1.0\n" + b"x" * (1024 * 1024),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 413
    assert manager.list_sessions() == []


def _seed_completed_session(
    storage_path: Path,
    session_id: str,
    steps_cfg: dict[str, Any],
) -> SessionManager:
    """Create a session in 'completed' status with a config.yaml on disk."""
    manager = SessionManager(storage_path, ttl_hours=2)
    session_dir = manager.create_session(session_id)
    manager.update_session(session_id, {"status": "completed"})

    config_path = session_dir / "input" / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"session_id": session_id},
                "input": {"usd_path": "scene.usd"},
                "steps": steps_cfg,
            }
        )
    )
    return manager


def _build_test_client(manager: SessionManager) -> TestClient:
    """Build a TestClient with just the pipeline router wired up.

    Bypasses the full FastAPI lifespan (which would touch global config,
    NVIDIA API key checks, periodic cleanup tasks, etc.).
    """
    app = FastAPI()
    pipeline_router.set_session_manager(manager)
    app.include_router(pipeline_router.router)
    return TestClient(app)


def test_regenerate_rejects_disabled_render_step(tmp_path: Path) -> None:
    """Regenerate with a single disabled step returns 422 with a clear detail."""
    sid = "session-render-disabled"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = client.post(f"/pipeline/{sid}/regenerate", json={"steps": ["render"]})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "render" in detail
    assert "disabled" in detail.lower()


def test_default_pipeline_config_preserves_service_auto_prompting() -> None:
    """Service-created configs should continue auto-prompting missing materials."""
    config = pipeline_router.build_default_pipeline_config(
        session_id="session-auto-prompt",
        usd_path="/tmp/scene.usd",
        working_dir="/tmp/work",
        material_textures={"Steel": {"prompt": "brushed steel"}},
        user_prompt="aged",
    )

    assert config["auto_prompt"]["enabled"] is True
    assert config["auto_prompt"]["user_prompt"] == "aged"


def test_default_pipeline_config_can_disable_service_auto_prompting() -> None:
    """Explicit validation runs can request strict material_textures scope."""
    config = pipeline_router.build_default_pipeline_config(
        session_id="session-strict-scope",
        usd_path="/tmp/scene.usd",
        working_dir="/tmp/work",
        material_textures={"Steel": {"prompt": "brushed steel"}},
        user_prompt="aged",
        auto_prompt_enabled=False,
    )

    assert config["auto_prompt"]["enabled"] is False
    assert config["material_textures"] == {"Steel": {"prompt": "brushed steel"}}


def test_legacy_service_config_migration_preserves_auto_prompting() -> None:
    """Regenerate should keep auto-prompting and add current service guards."""
    config = {"auto_prompt": {"user_prompt": "aged"}}

    pipeline_router._preserve_legacy_service_auto_prompting(config)

    assert config["auto_prompt"]["enabled"] is True
    assert config["auto_prompt"]["max_generated_materials"] == 64


def test_legacy_service_config_migration_preserves_disabled_auto_prompting() -> None:
    """Regenerate should not add auto-prompt guards to explicitly disabled configs."""
    config = {"auto_prompt": {"enabled": False, "user_prompt": "aged"}}

    pipeline_router._preserve_legacy_service_auto_prompting(config)

    assert config["auto_prompt"] == {"enabled": False, "user_prompt": "aged"}


def test_regenerate_rejects_disabled_render_previews_step(tmp_path: Path) -> None:
    """render_previews disabled by default in service deploy must also be rejected."""
    sid = "session-render-previews-disabled"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["render_previews"]}
    )

    assert response.status_code == 422
    assert "render_previews" in response.json()["detail"]


def test_regenerate_rejects_mixed_request_listing_all_disabled(tmp_path: Path) -> None:
    """A mixed request (one valid + two disabled) must list every offender."""
    sid = "session-mixed"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = client.post(
        f"/pipeline/{sid}/regenerate",
        json={"steps": ["generate_textures", "render", "render_previews"]},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    # Both disabled steps are surfaced; the enabled one is not flagged.
    assert "render" in detail
    assert "render_previews" in detail
    assert "generate_textures" not in detail


def test_regenerate_accepts_enabled_step_when_others_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sanity: requesting only an enabled step must NOT 422.

    We stub the job registry so the test does not actually launch the
    pipeline executor -- the validation guard must let the request through
    before registration happens.
    """
    sid = "session-enabled-only"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())

    class _StubRegistry:
        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            # Close the unawaited coroutine to avoid RuntimeWarning. The
            # production registry would schedule it; the test only cares
            # about the 202 acknowledgement.
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

    client = _build_test_client(manager)
    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
    )

    assert response.status_code == 202, response.text


def test_regenerate_hydrates_cache_for_incremental_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = "session-regenerate-hydrate-cache"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    sync_prefixes: list[str] = []

    def recording_sync_from_store(session_id: str, prefix: str = "") -> int:
        sync_prefixes.append(prefix)
        return 0

    class _StubRegistry:
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

    class _StubBus:
        def clear_session_state(self, *args: Any, **kwargs: Any) -> None:
            return None

        async def seed_pending_session(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(manager, "sync_from_store", recording_sync_from_store)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _StubBus())

    client = _build_test_client(manager)
    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
    )

    assert response.status_code == 202, response.text
    assert sync_prefixes == ["input/", "cache/"]


def test_regenerate_clears_stale_bus_state_before_register(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new regenerate run must not let post-register cleanup erase new events."""
    sid = "session-regenerate-clear-before-register"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    manager.update_session(
        sid,
        {
            "status": "failed",
            "error": "old failure",
            "failed_step": "generate_textures",
            "failed_step_stats": {"old": True},
            "partial_results": {"old": True},
        },
    )
    calls: list[str] = []

    class _StubBus:
        cleared = False

        def clear_session_state(self, session_id: str) -> None:
            assert session_id == sid
            calls.append("clear")
            self.cleared = True

        async def seed_pending_session(self, session_id: str) -> None:
            assert session_id == sid
            calls.append("seed")

    bus = _StubBus()

    class _StubRegistry:
        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            calls.append("register")
            assert bus.cleared is True
            metadata = manager.get_session_metadata(session_id)
            assert metadata is not None
            assert metadata["status"] == "pending"
            assert metadata.get("error") is None
            assert metadata.get("failed_step") is None
            coro.close()
            if on_finished is not None:
                on_finished()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: bus)

    client = _build_test_client(manager)
    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
    )

    assert response.status_code == 202, response.text
    assert calls == ["clear", "seed", "register"]


def test_regenerate_register_failure_restores_prior_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clearing before register still rolls disk diagnostics back on failure."""
    sid = "session-regenerate-register-fails"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    old_diagnostics = {
        "status": "failed",
        "error": "old failure",
        "failed_step": "generate_textures",
        "failed_step_stats": {"old": True},
        "failed_at": "2026-04-30T00:00:00+00:00",
        "partial_results": {"old": True},
    }
    manager.update_session(sid, old_diagnostics)

    class _StubBus:
        def clear_session_state(self, session_id: str) -> None:
            assert session_id == sid

        async def seed_pending_session(self, session_id: str) -> None:
            assert session_id == sid

    class _FailingRegistry:
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
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _StubBus())

    client = _build_test_client(manager)
    with pytest.raises(RuntimeError, match="synthetic register failure"):
        client.post(
            f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
        )

    metadata = manager.get_session_metadata(sid)
    assert metadata is not None
    for key, value in old_diagnostics.items():
        assert metadata[key] == value
    assert manager.is_worker_active(sid) is False


def test_create_existing_session_rejects_worker_lock(tmp_path: Path) -> None:
    """A draining worker lock blocks same-session pipeline restart."""
    sid = "session-worker-locked"
    manager = SessionManager(tmp_path, ttl_hours=2)
    manager.create_session(sid)
    client = _build_test_client(manager)

    with manager.worker_lock(sid):
        response = client.post("/pipeline", data={"session_id": sid})

    assert response.status_code == 409
    assert "worker" in response.json()["detail"].lower()


def test_create_existing_shared_session_defers_hydration_until_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submitting a shared session should not download input before 202."""
    from ...service.storage import LocalSessionStore

    sid = "session-defer-hydration"
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=2, store=shared_store)
    manager.create_session(sid)
    manager.update_session(
        sid,
        {
            "config": {
                "has_usd_upload": True,
                "input_extension": ".usd",
                "original_filename": "scene.usd",
            }
        },
    )
    released: list[str] = []
    sync_called = False
    real_release_worker_lock = manager.release_worker_lock

    def failing_sync_from_store(session_id: str, prefix: str = "") -> int:
        nonlocal sync_called
        sync_called = True
        raise RuntimeError("synthetic hydration failure")

    def recording_release(worker_lock: Any, session_id: str) -> None:
        released.append(session_id)
        real_release_worker_lock(worker_lock, session_id)

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
            coro.close()
            if on_finished is not None:
                on_finished()

    monkeypatch.setattr(manager, "sync_from_store", failing_sync_from_store)
    monkeypatch.setattr(manager, "release_worker_lock", recording_release)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())

    client = _build_test_client(manager)
    response = client.post("/pipeline", data={"session_id": sid})

    assert response.status_code == 202, response.text
    assert sync_called is False
    assert released == [sid]


def test_create_existing_session_reserves_worker_lock_before_202(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An accepted existing-session job blocks cross-process deletion immediately."""
    sid = "session-reserve-before-ack"
    manager = SessionManager(tmp_path, ttl_hours=2)
    manager.create_session(sid)
    session_dir = manager.get_session_dir(sid)
    (session_dir / "input" / "scene.usd").write_text("#usda 1.0\n", encoding="utf-8")
    peer_manager = SessionManager(tmp_path, ttl_hours=2)
    observed: dict[str, bool] = {}
    real_find_input_usd = pipeline_router._find_input_usd

    def racing_find_input_usd(session_dir: Path) -> Path | None:
        observed["delete_blocked_before_read"] = (
            peer_manager.delete_session(sid) is False
        )
        return real_find_input_usd(session_dir)

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
            observed["delete_blocked"] = (
                peer_manager.delete_session(session_id) is False
            )
            coro.close()
            if on_finished is not None:
                on_finished()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "_find_input_usd", racing_find_input_usd)

    client = _build_test_client(manager)
    response = client.post("/pipeline", data={"session_id": sid})

    assert response.status_code == 202, response.text
    assert observed["delete_blocked_before_read"] is True
    assert observed["delete_blocked"] is True
    with peer_manager.worker_lock(sid, timeout=0):
        pass
    assert manager.session_exists(sid) is True


def test_regenerate_reserves_worker_lock_before_session_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regeneration blocks deletion before reading metadata/config from disk."""
    sid = "session-regenerate-reserve-before-read"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    peer_manager = SessionManager(tmp_path, ttl_hours=2)
    observed: dict[str, bool] = {}
    real_get_metadata = manager.get_session_metadata

    def racing_get_metadata(session_id: str) -> dict[str, Any] | None:
        if session_id == sid and "delete_blocked_before_read" not in observed:
            observed["delete_blocked_before_read"] = (
                peer_manager.delete_session(session_id) is False
            )
        return real_get_metadata(session_id)

    class _StubRegistry:
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

    class _StubBus:
        def clear_session_state(self, *args: Any, **kwargs: Any) -> None:
            return None

        async def seed_pending_session(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(manager, "get_session_metadata", racing_get_metadata)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _StubBus())

    client = _build_test_client(manager)
    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
    )

    assert response.status_code == 202, response.text
    assert observed["delete_blocked_before_read"] is True
    with peer_manager.worker_lock(sid, timeout=0):
        pass
    assert manager.session_exists(sid) is True


def test_regenerate_rejects_worker_lock(tmp_path: Path) -> None:
    """A draining worker lock blocks same-session regeneration."""
    sid = "session-regenerate-worker-locked"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    with manager.worker_lock(sid):
        response = client.post(
            f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
        )

    assert response.status_code == 409
    assert "worker" in response.json()["detail"].lower()


def test_regenerate_missing_session_does_not_create_session_dir(
    tmp_path: Path,
) -> None:
    """A 404 regenerate request must not leave hidden lock-only sessions."""
    sid = "missing-regenerate-session"
    manager = SessionManager(tmp_path, ttl_hours=2)
    client = _build_test_client(manager)

    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"
    assert not (tmp_path / sid).exists()


# ---------------------------------------------------------------------------
# Empty steps[] must be 422, not 202.
# ---------------------------------------------------------------------------


def test_regenerate_rejects_empty_steps_array(tmp_path: Path) -> None:
    """``{"steps": []}`` must hit pydantic's min_length validator and 422."""
    sid = "session-empty-steps"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = client.post(f"/pipeline/{sid}/regenerate", json={"steps": []})

    assert response.status_code == 422
    detail = response.json()["detail"]
    # FastAPI/pydantic v2 emits a list of structured errors for body
    # validation failures. The min_length=1 violation is type=too_short.
    assert isinstance(detail, list)
    assert any(item.get("type") == "too_short" for item in detail)
    assert any(item.get("loc") == ["body", "steps"] for item in detail)


# ---------------------------------------------------------------------------
# Malformed material_textures_json must be a
# structured 422, not a plain 400 (or worse, a connection drop).
# ---------------------------------------------------------------------------


def _make_minimal_usd_bytes() -> bytes:
    """Return a tiny but-valid .usda payload for multipart upload tests."""
    return b'#usda 1.0\n(\n    defaultPrim = "World"\n)\n\ndef Xform "World" {}\n'


class _NoopRegistry:
    def __init__(self, *, running: bool = False, cancel_result: bool = True) -> None:
        self.running = running
        self.cancel_result = cancel_result

    def is_running(self, session_id: str) -> bool:
        return self.running

    async def cancel(self, session_id: str) -> bool:
        return self.cancel_result

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


class _NoopBus:
    def __init__(self, snapshot: dict[str, Any] | None = None) -> None:
        self.snapshot = snapshot
        self.emitted: list[Any] = []

    def clear_session_state(self, *args: Any, **kwargs: Any) -> None:
        return None

    def get_snapshot(self, session_id: str) -> dict[str, Any] | None:
        return self.snapshot

    def get_queue(self, session_id: str) -> asyncio.Queue[Any]:
        return asyncio.Queue()

    async def seed_pending_session(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def emit(self, event: Any) -> None:
        self.emitted.append(event)


def _stub_pipeline_registration(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any] | None = None,
    *,
    running: bool = False,
) -> None:
    registry = _NoopRegistry(running=running)

    def fake_execute_pipeline_async(**kwargs: Any) -> Any:
        if captured is not None:
            captured["config"] = kwargs["config_dict"]

        async def noop() -> None:
            return None

        return noop()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _NoopBus())
    monkeypatch.setattr(
        pipeline_router, "execute_pipeline_async", fake_execute_pipeline_async
    )


def test_download_s3_to_session_validates_uri_and_extension(tmp_path: Path) -> None:
    for uri, expected in (
        ("https://bucket/scene.usd", "Invalid S3 URI"),
        ("s3://bucket/path/scene.txt", "Invalid USD file type"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            pipeline_router._download_s3_to_session(uri, tmp_path)
        assert exc_info.value.status_code == 400
        assert expected in str(exc_info.value.detail)


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (FileNotFoundError("missing"), 404, "not found"),
        (PermissionError("denied"), 403, "Access denied"),
        (RuntimeError("backend down"), 502, "Failed to download"),
    ],
)
def test_download_s3_to_session_translates_download_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status_code: int,
    detail: str,
) -> None:
    def fail_download(_uri: str, _path: Path) -> None:
        raise error

    monkeypatch.setattr(pipeline_router.config, "s3_allowed_buckets", "bucket")
    monkeypatch.setattr(pipeline_router, "download_file_from_s3", fail_download)

    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._download_s3_to_session("s3://bucket/path/scene.usd", tmp_path)

    assert exc_info.value.status_code == status_code
    assert detail in str(exc_info.value.detail)


def test_download_s3_to_session_success_and_size_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write_usd(_uri: str, path: Path) -> None:
        path.write_text("#usda 1.0\n", encoding="utf-8")

    monkeypatch.setattr(pipeline_router.config, "s3_allowed_buckets", "bucket")
    monkeypatch.setattr(pipeline_router, "download_file_from_s3", write_usd)
    path = pipeline_router._download_s3_to_session(
        "s3://bucket/path/scene.usda",
        tmp_path / "ok",
    )
    assert path.name == "scene.usda"
    assert path.exists()

    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 0)
    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._download_s3_to_session(
            "s3://bucket/path/scene.usd",
            tmp_path / "too-large",
        )
    assert exc_info.value.status_code == 413
    assert not (tmp_path / "too-large" / "input" / "scene.usd").exists()


def test_pipeline_router_small_helpers_cover_edge_cases(tmp_path: Path) -> None:
    assert (
        pipeline_router._input_usd_path_from_metadata(
            tmp_path, {"config": ["bad-shape"]}
        )
        is None
    )
    assert (
        pipeline_router._input_usd_path_from_metadata(
            tmp_path, {"config": {"input_extension": ".txt"}}
        )
        is None
    )
    assert pipeline_router._input_usd_path_from_metadata(
        tmp_path, {"config": {"input_extension": ".usdz"}}
    ) == (tmp_path / "input" / "scene.usdz")

    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._normalize_uri_list(["ok.png", 3], field_name="images")
    assert exc_info.value.status_code == 422
    assert exc_info.value.detail[0]["loc"] == ["form", "images", 1]


async def test_reserve_worker_slot_handles_value_error_and_stalled_worker() -> None:
    class ValueErrorManager:
        def acquire_worker_lock(self, session_id: str, timeout: int) -> Any:
            raise ValueError("bad session")

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._reserve_worker_slot(ValueErrorManager(), "sid")  # type: ignore[arg-type]
    assert exc_info.value.status_code == 404

    class StalledManager:
        def __init__(self) -> None:
            self.released: list[str] = []

        def acquire_worker_lock(self, session_id: str, timeout: int) -> object:
            return object()

        def is_worker_stalled(self, session_id: str) -> bool:
            return True

        def release_worker_lock(self, worker_lock: Any, session_id: str) -> None:
            self.released.append(session_id)

    stalled = StalledManager()
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._reserve_worker_slot(stalled, "stalled")  # type: ignore[arg-type]
    assert exc_info.value.status_code == 409
    assert stalled.released == ["stalled"]


def test_active_snapshot_status_normalizes_dates_and_completed_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = {
        "status": "running",
        "created_at": "2026-01-01T00:00:00",
        "elapsed_seconds": 5,
        "completed_steps": {"bad": "shape"},
        "preview_images": ["preview.png"],
        "overall_progress": {"current_step": 1, "total_steps": 3, "percent": 33},
        "failed_step_stats": {"message": "/var/texture-agent/sessions/x"},
    }
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _NoopBus(snapshot))
    monkeypatch.setattr(
        pipeline_router, "get_job_registry", lambda: _NoopRegistry(running=True)
    )

    status = pipeline_router._active_snapshot_status("sid")

    assert status is not None
    assert status.completed_steps == []
    assert status.preview_images == ["/artifacts/sid/preview/preview.png"]
    assert status.elapsed_seconds >= 5

    snapshot["created_at"] = "not-a-date"
    status = pipeline_router._active_snapshot_status("sid")
    assert status is not None
    assert status.elapsed_seconds == 5


async def test_save_reference_image_upload_validates_extension_and_default_name(
    tmp_path: Path,
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._save_reference_image_upload(
            _NamedUpload("bad.gif", [b"img"]),
            tmp_path / "bad",
        )  # type: ignore[arg-type]
    assert exc_info.value.status_code == 400

    uri = await pipeline_router._save_reference_image_upload(
        _NamedUpload(None, [b"png-bytes"]),
        tmp_path / "ok",
    )  # type: ignore[arg-type]
    assert uri is not None
    assert uri.endswith("/input/reference_images/reference_image.png")


async def test_worker_slot_callbacks_cover_no_token_and_emit_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    assert (
        pipeline_router._heartbeat_worker_slot_callback(
            manager,
            "sid",
            object(),
        )()
        is None
    )

    class MissingManager:
        def update_session(self, session_id: str, metadata: dict[str, Any]) -> None:
            raise FileNotFoundError(session_id)

    pipeline_router._cancel_never_started_callback(MissingManager(), "missing")()  # type: ignore[arg-type]

    manager.create_session("no-loop")
    pipeline_router._cancel_never_started_callback(manager, "no-loop")()
    assert manager.get_session_metadata("no-loop")["status"] == "cancelled"

    class FailingBus(_NoopBus):
        async def emit(self, event: Any) -> None:
            raise RuntimeError("emit failed")

    manager.create_session("emit-fails")
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: FailingBus())
    pipeline_router._cancel_never_started_callback(manager, "emit-fails")()
    await asyncio.sleep(0)


def test_cancel_never_started_callback_without_running_loop(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    manager.create_session("no-loop")

    pipeline_router._cancel_never_started_callback(manager, "no-loop")()

    assert manager.get_session_metadata("no-loop")["status"] == "cancelled"


def test_restore_session_after_reset_failure_edge_cases(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingManager:
        def update_session(self, session_id: str, snapshot: dict[str, Any]) -> None:
            raise RuntimeError("restore failed")

    pipeline_router._restore_session_after_reset_failure(
        FailingManager(),  # type: ignore[arg-type]
        "sid",
        {},
    )
    pipeline_router._restore_session_after_reset_failure(
        FailingManager(),  # type: ignore[arg-type]
        "sid",
        {"status": "failed"},
    )
    assert "Failed to restore session metadata" in caplog.text


def test_default_pipeline_config_uses_direct_image_gen_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_router.config, "image_gen_api_key_env", None)
    monkeypatch.setattr(pipeline_router.config, "image_gen_api_key", "secret-key")

    config = pipeline_router.build_default_pipeline_config(
        session_id="sid",
        usd_path="/tmp/scene.usd",
        working_dir="/tmp/work",
    )

    assert config["texture"]["image_gen"]["api_key"] == "secret-key"


def test_upload_usd_immediate_rejects_missing_or_conflicting_inputs(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    client = _build_test_client(manager)

    response = client.post("/pipeline/upload-usd")
    assert response.status_code == 400
    assert "Either usd_file or s3_uri" in response.json()["detail"]

    response = client.post(
        "/pipeline/upload-usd",
        data={"s3_uri": "s3://bucket/path/scene.usd"},
        files={"usd_file": ("scene.usd", _make_minimal_usd_bytes())},
    )
    assert response.status_code == 400
    assert "either" in response.json()["detail"].lower()


@pytest.mark.parametrize("route", ["/pipeline", "/pipeline/upload-usd"])
@pytest.mark.parametrize(
    "allowed_buckets,s3_uri",
    [
        ("", "s3://approved/private/scene.usdz"),
        ("approved", "s3://foreign/private/scene.usdz"),
    ],
)
def test_client_s3_uri_rejects_foreign_bucket_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    allowed_buckets: str,
    s3_uri: str,
) -> None:
    """Texture rejects S3 inputs before acquiring a possibly remote store."""
    manager = SessionManager(tmp_path, ttl_hours=2)
    client = _build_test_client(manager)
    monkeypatch.setattr(pipeline_router.config, "s3_allowed_buckets", allowed_buckets)
    manager_calls = 0
    download_calls = 0

    def fail_manager():  # type: ignore[no-untyped-def]
        nonlocal manager_calls
        manager_calls += 1
        raise AssertionError("S3 policy must precede session-store access")

    def fail_download(*_args: Any, **_kwargs: Any) -> None:
        nonlocal download_calls
        download_calls += 1
        raise AssertionError("foreign S3 bucket must not be downloaded")

    monkeypatch.setattr(pipeline_router, "get_session_manager", fail_manager)
    monkeypatch.setattr(pipeline_router, "download_file_from_s3", fail_download)

    response = client.post(
        route,
        data={"s3_uri": s3_uri},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "S3 URI is not permitted by the service's configured bucket allowlist"
    )
    assert manager_calls == 0
    assert download_calls == 0


def test_upload_usd_immediate_downloads_s3_and_cleans_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    client = _build_test_client(manager)
    monkeypatch.setattr(pipeline_router.config, "s3_allowed_buckets", "bucket")

    def fake_download(s3_uri: str, session_dir: Path) -> Path:
        path = session_dir / "input" / "scene.usdc"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#usda 1.0\n", encoding="utf-8")
        return path

    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", fake_download)
    response = client.post(
        "/pipeline/upload-usd",
        data={"s3_uri": "s3://bucket/path/original.usdc"},
    )
    assert response.status_code == 201, response.text
    sid = response.json()["session_id"]
    metadata = manager.get_session_metadata(sid)
    assert metadata is not None
    assert metadata["config"]["original_filename"] == "original.usdc"
    assert metadata["config"]["input_extension"] == ".usdc"

    def raise_http(_s3_uri: str, _session_dir: Path) -> Path:
        raise HTTPException(status_code=404, detail="missing")

    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", raise_http)
    response = client.post(
        "/pipeline/upload-usd",
        data={"s3_uri": "s3://bucket/path/missing.usd"},
    )
    assert response.status_code == 404

    def raise_generic(_s3_uri: str, _session_dir: Path) -> Path:
        raise RuntimeError("download exploded")

    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", raise_generic)
    response = client.post(
        "/pipeline/upload-usd",
        data={"s3_uri": "s3://bucket/path/error.usd"},
    )
    assert response.status_code == 500
    assert "download exploded" in response.json()["detail"]


def test_upload_usd_immediate_rejects_invalid_file_type(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    client = _build_test_client(manager)

    response = client.post(
        "/pipeline/upload-usd",
        files={"usd_file": ("scene.txt", b"not usd")},
    )

    assert response.status_code == 400
    assert "Invalid USD file type" in response.json()["detail"]


def test_create_pipeline_rejects_missing_conflicting_and_invalid_inputs(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    client = _build_test_client(manager)

    response = client.post("/pipeline")
    assert response.status_code == 400
    assert "One of usd_file, session_id, or s3_uri" in response.json()["detail"]

    response = client.post("/pipeline", data={"session_id": "missing"})
    assert response.status_code == 404

    response = client.post(
        "/pipeline",
        files={"usd_file": ("scene.txt", b"not usd")},
    )
    assert response.status_code == 400
    assert "Invalid USD file type" in response.json()["detail"]


def test_create_pipeline_rejects_inline_secret_before_durable_config(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    client = _build_test_client(manager)
    sentinel = "texture-custom-parameter-secret-713"

    response = client.post(
        "/pipeline",
        files={"usd_file": ("scene.usd", _make_minimal_usd_bytes())},
        data={"backend_custom_parameters_json": ('{"api_key": "' + sentinel + '"}')},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Pipeline configuration is invalid"}
    assert sentinel not in response.text
    assert manager.list_sessions() == []
    assert not list(tmp_path.rglob("config.yaml"))


def test_create_pipeline_session_id_precedes_unused_s3_uri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing session must ignore lower-priority stray S3/file sources."""
    session_id = "session-source-precedence"
    manager = SessionManager(tmp_path, ttl_hours=2)
    session_dir = manager.create_session(session_id)
    input_path = session_dir / "input" / "scene.usd"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("#usda 1.0\n", encoding="utf-8")
    original_s3_uri = "s3://approved/original/scene.usdz"
    manager.update_session(
        session_id,
        {
            "config": {
                "original_filename": "original.usdz",
                "has_usd_upload": False,
                "s3_uri": original_s3_uri,
            }
        },
    )
    client = _build_test_client(manager)
    _stub_pipeline_registration(monkeypatch)

    def fail_validation(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("unused S3 URI reached authorization")

    def fail_download(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("unused S3 URI reached the downloader")

    monkeypatch.setattr(
        pipeline_router,
        "_validate_and_authorize_s3_usd_uri",
        fail_validation,
    )
    monkeypatch.setattr(pipeline_router, "download_file_from_s3", fail_download)

    response = client.post(
        "/pipeline",
        files={"usd_file": ("stray.usd", b"#usda 1.0\n")},
        data={
            "session_id": session_id,
            "s3_uri": "s3://foreign/private/scene.usdz",
        },
    )

    assert response.status_code == 202, response.text
    assert response.json()["session_id"] == session_id
    metadata = manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["config"]["s3_uri"] == original_s3_uri
    assert metadata["config"]["original_filename"] == "original.usdz"
    assert metadata["config"]["has_usd_upload"] is False


def test_create_pipeline_rejects_running_and_shared_active_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = "session-running"
    manager = SessionManager(tmp_path, ttl_hours=2)
    manager.create_session(sid)
    client = _build_test_client(manager)

    _stub_pipeline_registration(monkeypatch, running=True)
    response = client.post("/pipeline", data={"session_id": sid})
    assert response.status_code == 409
    assert "already running" in response.json()["detail"]

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _NoopRegistry())
    monkeypatch.setattr(manager, "uses_shared_store", lambda: True)
    monkeypatch.setattr(manager, "is_worker_active", lambda _sid: True)
    response = client.post("/pipeline", data={"session_id": sid})
    assert response.status_code == 409
    assert "another instance" in response.json()["detail"]


def test_create_pipeline_s3_and_file_modes_start_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    client = _build_test_client(manager)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(pipeline_router.config, "s3_allowed_buckets", "bucket")
    _stub_pipeline_registration(monkeypatch, captured)

    def fake_download(s3_uri: str, session_dir: Path) -> Path:
        path = session_dir / "input" / "scene.usdz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_make_minimal_usd_bytes())
        return path

    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", fake_download)
    response = client.post(
        "/pipeline",
        data={
            "s3_uri": "s3://bucket/path/scene.usdz",
            "reference_image_uris_json": '[" file:///tmp/ref.png ", "", 3]',
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == [
        "form",
        "reference_image_uris_json",
        2,
    ]

    response = client.post(
        "/pipeline",
        data={
            "s3_uri": "s3://bucket/path/scene.usdz",
            "reference_image_uris_json": '[" file:///tmp/ref.png ", ""]',
            "turntable_video_uri": " file:///tmp/turntable.mp4 ",
            "multiview_image_uris_json": '["file:///tmp/a.png"]',
            "backend_custom_parameters_json": '{"cfg": 7}',
            "texture_backend": "service",
            "texture_endpoint": " http://texture ",
            "backend_engine": " step1x ",
            "detail_policy": "surface_only",
            "uv_policy": " force_projection ",
            "uv_scope": " target_prims ",
            "uv_backend": " python ",
            "uv_projection": " box ",
            "uv_overwrite_existing": "true",
            "uv_rebake_source_albedo": "true",
            "uv_rebake_size": "256",
            "uv_normalize_out_of_range": "false",
        },
    )
    assert response.status_code == 202, response.text
    config = captured["config"]
    assert config["input"]["usd_path"].endswith("scene.usdz")
    assert config["texture"]["reference_image_uris"] == ["file:///tmp/ref.png"]
    assert config["texture"]["turntable_video_uri"] == "file:///tmp/turntable.mp4"
    assert config["texture"]["multiview_image_uris"] == ["file:///tmp/a.png"]
    assert config["texture"]["custom_parameters"] == {"cfg": 7}
    assert config["texture"]["endpoint"] == "http://texture"
    assert config["texture"]["engine"] == "step1x"
    assert config["texture"]["detail_policy"] == "surface_only"
    assert config["texture"]["uv_rebake_size"] == 256

    captured.clear()
    response = client.post(
        "/pipeline",
        files={"usd_file": ("scene.usda", _make_minimal_usd_bytes())},
    )
    assert response.status_code == 202, response.text
    assert captured["config"]["input"]["usd_path"].endswith("scene.usda")


def test_create_pipeline_s3_and_upload_failures_clean_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    client = _build_test_client(manager)
    monkeypatch.setattr(pipeline_router.config, "s3_allowed_buckets", "bucket")

    def raise_http(_s3_uri: str, _session_dir: Path) -> Path:
        raise HTTPException(status_code=404, detail="missing")

    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", raise_http)
    response = client.post(
        "/pipeline",
        data={"s3_uri": "s3://bucket/path/missing.usd"},
    )
    assert response.status_code == 404

    def raise_generic(_s3_uri: str, _session_dir: Path) -> Path:
        raise RuntimeError("download exploded")

    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", raise_generic)
    response = client.post(
        "/pipeline",
        data={"s3_uri": "s3://bucket/path/error.usd"},
    )
    assert response.status_code == 500
    assert "download exploded" in response.json()["detail"]

    async def raise_stream_copy(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("read exploded")

    monkeypatch.setattr(pipeline_router, "_stream_copy", raise_stream_copy)
    response = client.post(
        "/pipeline",
        files={"usd_file": ("scene.usd", _make_minimal_usd_bytes())},
    )
    assert response.status_code == 500
    assert "read exploded" in response.json()["detail"]


def test_create_pipeline_existing_session_reference_upload_and_metadata_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = "session-reference-upload"
    manager = SessionManager(tmp_path, ttl_hours=2)
    session_dir = manager.create_session(sid)
    (session_dir / "input" / "scene.usd").write_text("#usda 1.0\n", encoding="utf-8")
    manager.update_session(sid, {"config": "legacy-bad-shape"})
    client = _build_test_client(manager)
    captured: dict[str, Any] = {}
    _stub_pipeline_registration(monkeypatch, captured)

    response = client.post(
        "/pipeline",
        data={"session_id": sid, "reference_image_uris_json": '["file:///tmp/a.png"]'},
        files={"reference_image_file": ("ref.png", b"png")},
    )

    assert response.status_code == 202, response.text
    refs = captured["config"]["texture"]["reference_image_uris"]
    assert refs[0] == "file:///tmp/a.png"
    assert refs[1].endswith("/input/reference_images/reference_image.png")
    metadata = manager.get_session_metadata(sid)
    assert metadata is not None
    assert metadata["config"]["original_filename"] is None


def test_pipeline_status_and_results_disk_views_normalize_edges(
    tmp_path: Path,
) -> None:
    from ...service.routers import sessions_router

    manager = SessionManager(tmp_path, ttl_hours=2)
    pipeline_router.set_session_manager(manager)
    sessions_router.set_session_manager(manager)
    sid = "session-disk-status"
    manager.create_session(sid)
    manager.update_session(
        sid,
        {
            "status": "completed",
            "completed_steps": {"bad": "shape"},
            "results": {"manifest_available": False},
        },
    )

    status = asyncio.run(pipeline_router.get_pipeline_status(sid))
    assert status.completed_steps == []

    manager.update_session(sid, {"status": "running"})
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pipeline_router.get_pipeline_results(sid))
    assert exc_info.value.status_code == 202


def test_cancel_pipeline_rejects_terminal_and_failed_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    pipeline_router.set_session_manager(manager)
    sid = "session-cancel"
    manager.create_session(sid)
    manager.update_session(sid, {"status": "completed"})

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pipeline_router.cancel_pipeline(sid))
    assert exc_info.value.status_code == 400

    manager.update_session(sid, {"status": "running"})
    monkeypatch.setattr(
        pipeline_router,
        "get_job_registry",
        lambda: _NoopRegistry(running=True, cancel_result=False),
    )
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _NoopBus())
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(pipeline_router.cancel_pipeline(sid))
    assert exc_info.value.status_code == 400
    assert "Failed to cancel" in exc_info.value.detail


def test_stream_progress_events_rejects_remote_running_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RemoteManager:
        def __init__(self, *, shared: bool) -> None:
            self.shared = shared

        def session_exists(self, session_id: str) -> bool:
            return True

        def get_session_metadata(self, session_id: str) -> dict[str, Any]:
            return {"status": "running"}

        def uses_shared_store(self) -> bool:
            return self.shared

        def get_session_dir(self, session_id: str) -> Path:
            return tmp_path / "not-local"

    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _NoopBus())
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _NoopRegistry())

    for shared in (True, False):
        pipeline_router.set_session_manager(RemoteManager(shared=shared))  # type: ignore[arg-type]
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(pipeline_router.stream_progress_events("remote"))
        assert exc_info.value.status_code == 503


def test_regenerate_reaches_metadata_404_and_running_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_reserve(_manager: Any, _session_id: str) -> object:
        return object()

    monkeypatch.setattr(pipeline_router, "_reserve_worker_slot", fake_reserve)

    manager = SessionManager(tmp_path, ttl_hours=2)
    pipeline_router.set_session_manager(manager)
    monkeypatch.setattr(manager, "release_worker_lock", lambda *_args: None)

    client = _build_test_client(manager)
    response = client.post(
        "/pipeline/missing-after-reserve/regenerate",
        json={"steps": ["generate_textures"]},
    )
    assert response.status_code == 404

    sid = "session-regenerate-running"
    manager.create_session(sid)
    manager.update_session(sid, {"status": "pending"})
    response = client.post(
        f"/pipeline/{sid}/regenerate",
        json={"steps": ["generate_textures"]},
    )
    assert response.status_code == 400
    assert "pending" in response.json()["detail"]


def test_create_pipeline_rejects_malformed_material_textures_json(
    tmp_path: Path,
) -> None:
    """Malformed JSON in the form field returns a structured 422 detail list."""
    sid = "session-for-malformed-json"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    # Reuse the existing session so the handler does not try to download
    # / persist a fresh upload through the broader filesystem path.
    client = _build_test_client(manager)

    response = client.post(
        "/pipeline",
        files={
            "usd_file": (
                "scene.usda",
                _make_minimal_usd_bytes(),
                "application/octet-stream",
            ),
        },
        data={
            "session_id": sid,
            "material_textures_json": "NOT_JSON_AT_ALL",
        },
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert any(item.get("type") == "json_invalid" for item in detail)
    assert any(item.get("loc") == ["form", "material_textures_json"] for item in detail)


def test_create_pipeline_accepts_empty_material_textures_json(
    tmp_path: Path,
) -> None:
    """Sanity: empty / whitespace-only material_textures_json must not 422.

    The existing parser intentionally treats an empty form value as "no
    overrides", and customers rely on that. The 6127700 fix tightens
    *malformed* JSON without regressing the empty-string allowance.
    """
    sid = "session-for-empty-json"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = client.post(
        "/pipeline",
        files={
            "usd_file": (
                "scene.usda",
                _make_minimal_usd_bytes(),
                "application/octet-stream",
            ),
        },
        data={"session_id": sid, "material_textures_json": "   "},
    )

    # The handler accepts the request and proceeds toward pipeline
    # registration. We only assert that *parsing* did not 422 -- a
    # downstream failure (e.g., 409 because the executor stub is not
    # wired) is fine for this test's contract.
    assert response.status_code != 422, response.text


# ---------------------------------------------------------------------------
# Syntactically valid but structurally invalid material_textures_json must
# also return 422 at submit, not
# fail asynchronously inside the pipeline after the 202.
# ---------------------------------------------------------------------------


def _post_with_material_textures_json(
    client: TestClient,
    sid: str,
    payload: str,
    *,
    extra_data: dict[str, str] | None = None,
) -> Any:
    data = {"session_id": sid, "material_textures_json": payload}
    if extra_data:
        data.update(extra_data)
    return client.post(
        "/pipeline",
        files={
            "usd_file": (
                "scene.usda",
                _make_minimal_usd_bytes(),
                "application/octet-stream",
            ),
        },
        data=data,
    )


def test_create_pipeline_rejects_top_level_list_material_textures(
    tmp_path: Path,
) -> None:
    """``[]`` is valid JSON but the wire shape is dict[str, dict]."""
    sid = "session-mt-list"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(client, sid, "[]")

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert any(item.get("type") == "dict_type" for item in detail)
    assert any(item.get("loc") == ["form", "material_textures_json"] for item in detail)


def test_create_pipeline_rejects_top_level_scalar_material_textures(
    tmp_path: Path,
) -> None:
    """``42`` decodes to int -- not a dict, must 422."""
    sid = "session-mt-scalar"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(client, sid, "42")

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(item.get("type") == "dict_type" for item in detail)


def test_create_pipeline_rejects_dict_with_scalar_value_material_textures(
    tmp_path: Path,
) -> None:
    """``{"Steel":"rust"}`` -- top level is dict but the value is a str.

    Per-material overrides must themselves be objects (prompt/opacity
    fields). A scalar leaf value would crash the prompt-expansion code
    later in the pipeline.
    """
    sid = "session-mt-scalar-leaf"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client, sid, '{"Steel": "rust", "Wood": {"prompt": "ok"}}'
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(item.get("type") == "dict_type" for item in detail)
    # The offending key is named in the loc; the well-formed key is not.
    assert any(
        item.get("loc") == ["form", "material_textures_json", "Steel"]
        for item in detail
    )
    assert not any(
        item.get("loc") == ["form", "material_textures_json", "Wood"] for item in detail
    )


def test_create_pipeline_accepts_empty_dict_material_textures(
    tmp_path: Path,
) -> None:
    """``{}`` is the documented "no overrides" shape -- must not 422."""
    sid = "session-mt-empty-dict"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(client, sid, "{}")

    assert response.status_code != 422, response.text


def test_create_pipeline_rejects_material_missing_prompt(
    tmp_path: Path,
) -> None:
    """An explicit material override needs a prompt before job acceptance."""
    sid = "session-mt-missing-prompt"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(client, sid, '{"Steel": {}}')

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        item.get("loc") == ["form", "material_textures_json", "Steel", "prompt"]
        and item.get("type") == "missing"
        for item in detail
    )


def test_create_pipeline_rejects_material_prompt_list(
    tmp_path: Path,
) -> None:
    """Prompt must be a non-empty string, not an arbitrary JSON value."""
    sid = "session-mt-list-prompt"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client, sid, '{"Steel": {"prompt": ["rust"]}}'
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        item.get("loc") == ["form", "material_textures_json", "Steel", "prompt"]
        for item in detail
    )


def test_create_pipeline_rejects_out_of_range_material_opacity(
    tmp_path: Path,
) -> None:
    """Opacity must be numeric and bounded before the job is registered."""
    sid = "session-mt-bad-opacity"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client, sid, '{"Steel": {"prompt": "rust", "opacity": 1.5}}'
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        item.get("loc") == ["form", "material_textures_json", "Steel", "opacity"]
        for item in detail
    )


def test_create_pipeline_rejects_unknown_material_override_field(
    tmp_path: Path,
) -> None:
    """Unknown material override fields are rejected before job acceptance."""
    sid = "session-mt-extra-field"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client, sid, '{"Steel": {"prompt": "rust", "roughness": 0.2}}'
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        item.get("loc") == ["form", "material_textures_json", "Steel", "roughness"]
        and item.get("type") == "extra_forbidden"
        for item in detail
    )


def test_create_pipeline_rejects_unknown_material_detail_policy(
    tmp_path: Path,
) -> None:
    sid = "session-mt-bad-detail-policy"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client,
        sid,
        '{"Steel": {"prompt": "rust", "detail_policy": "bake_traces"}}',
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        item.get("loc") == ["form", "material_textures_json", "Steel", "detail_policy"]
        for item in detail
    )


def test_create_pipeline_rejects_unknown_global_detail_policy(
    tmp_path: Path,
) -> None:
    sid = "session-global-bad-detail-policy"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client,
        sid,
        '{"Steel": {"prompt": "rust"}}',
        extra_data={"detail_policy": "bake_traces"},
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(item.get("loc") == ["body", "detail_policy"] for item in detail)


def test_create_pipeline_rejects_blank_material_key(
    tmp_path: Path,
) -> None:
    """Material override keys must name a real discovered material."""
    sid = "session-mt-blank-material"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client, sid, '{"   ": {"prompt": "rust"}}'
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        "Material override keys must be non-empty" in item.get("msg", "")
        for item in detail
    )


def test_create_pipeline_rejects_blank_per_prim_key(
    tmp_path: Path,
) -> None:
    """Per-prim override keys must identify a prim path or leaf name."""
    sid = "session-mt-blank-prim"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client,
        sid,
        '{"Steel": {"prompt": "rust", "per_prim": {"   ": {"opacity": 0.5}}}}',
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        "Per-prim override keys must be non-empty" in item.get("msg", "")
        for item in detail
    )


def test_create_pipeline_accepts_per_prim_and_enables_per_prim_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A documented per_prim override must switch the generated config mode."""
    sid = "session-mt-per-prim"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    session_dir = manager.get_session_dir(sid)
    (session_dir / "input" / "scene.usd").write_text("#usda 1.0\n", encoding="utf-8")
    captured: dict[str, Any] = {}

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
            coro.close()
            if on_finished is not None:
                on_finished()

    def fake_execute_pipeline_async(**kwargs: Any) -> Any:
        captured["config"] = kwargs["config_dict"]

        async def noop() -> None:
            return None

        return noop()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(
        pipeline_router, "execute_pipeline_async", fake_execute_pipeline_async
    )
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client,
        sid,
        (
            '{"Steel": {"prompt": "rust", "per_prim": '
            '{"/World/Rung_01": {"opacity": 0.65}}}}'
        ),
    )

    assert response.status_code == 202, response.text
    config = captured["config"]
    assert config["texture"]["mode"] == "per_prim"
    assert (
        config["material_textures"]["Steel"]["per_prim"]["/World/Rung_01"]["opacity"]
        == 0.65
    )


def test_create_pipeline_material_only_override_keeps_default_texture_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Material-only overrides should not opt a new run into per-prim mode."""
    sid = "session-mt-material-only"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    session_dir = manager.get_session_dir(sid)
    (session_dir / "input" / "scene.usd").write_text("#usda 1.0\n", encoding="utf-8")
    captured: dict[str, Any] = {}

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
            coro.close()
            if on_finished is not None:
                on_finished()

    def fake_execute_pipeline_async(**kwargs: Any) -> Any:
        captured["config"] = kwargs["config_dict"]

        async def noop() -> None:
            return None

        return noop()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(
        pipeline_router, "execute_pipeline_async", fake_execute_pipeline_async
    )
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client,
        sid,
        '{"Steel": {"prompt": "brushed steel", "opacity": 0.75}}',
    )

    assert response.status_code == 202, response.text
    assert "mode" not in captured["config"]["texture"]


@pytest.mark.parametrize(
    ("form_value", "expected_enabled"),
    [("false", False), ("true", True)],
)
def test_create_pipeline_auto_prompt_enabled_form_sets_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    form_value: str,
    expected_enabled: bool,
) -> None:
    """REST clients can explicitly choose strict or auto-prompting scope."""
    sid = f"session-mt-auto-prompt-{form_value}"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    session_dir = manager.get_session_dir(sid)
    (session_dir / "input" / "scene.usd").write_text("#usda 1.0\n", encoding="utf-8")
    captured: dict[str, Any] = {}

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
            coro.close()
            if on_finished is not None:
                on_finished()

    def fake_execute_pipeline_async(**kwargs: Any) -> Any:
        captured["config"] = kwargs["config_dict"]

        async def noop() -> None:
            return None

        return noop()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(
        pipeline_router, "execute_pipeline_async", fake_execute_pipeline_async
    )
    client = _build_test_client(manager)

    response = client.post(
        "/pipeline",
        files={
            "usd_file": (
                "scene.usda",
                _make_minimal_usd_bytes(),
                "application/octet-stream",
            ),
        },
        data={
            "session_id": sid,
            "material_textures_json": (
                '{"Aluminum_Matte": {"prompt": "weathered aluminum"}}'
            ),
            "auto_prompt_enabled": form_value,
        },
    )

    assert response.status_code == 202, response.text
    assert captured["config"]["auto_prompt"]["enabled"] is expected_enabled
    assert captured["config"]["material_textures"] == {
        "Aluminum_Matte": {"prompt": "weathered aluminum"}
    }


def test_regenerate_rejects_invalid_material_textures(
    tmp_path: Path,
) -> None:
    """Regenerate uses the same material override schema as POST /pipeline."""
    sid = "session-regenerate-mt-invalid"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = client.post(
        f"/pipeline/{sid}/regenerate",
        json={
            "steps": ["generate_textures"],
            "material_textures": {"Steel": {"opacity": "opaque"}},
        },
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        item.get("loc") == ["body", "material_textures", "Steel", "prompt"]
        for item in detail
    )
    assert any(
        item.get("loc") == ["body", "material_textures", "Steel", "opacity"]
        for item in detail
    )


def test_regenerate_per_prim_override_enables_per_prim_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regenerate uses per_prim overrides to switch texture mode too."""
    sid = "session-regenerate-mt-per-prim"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    captured: dict[str, Any] = {}

    class _StubRegistry:
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

    class _StubBus:
        def clear_session_state(self, *args: Any, **kwargs: Any) -> None:
            return None

        async def seed_pending_session(self, *args: Any, **kwargs: Any) -> None:
            return None

    def fake_execute_pipeline_async(**kwargs: Any) -> Any:
        captured["config"] = kwargs["config_dict"]

        async def noop() -> None:
            return None

        return noop()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _StubBus())
    monkeypatch.setattr(
        pipeline_router, "execute_pipeline_async", fake_execute_pipeline_async
    )
    client = _build_test_client(manager)

    response = client.post(
        f"/pipeline/{sid}/regenerate",
        json={
            "steps": ["generate_textures"],
            "material_textures": {
                "Steel": {
                    "prompt": "rust",
                    "per_prim": {"/World/Rung_01": {"opacity": 0.65}},
                }
            },
        },
    )

    assert response.status_code == 202, response.text
    assert captured["config"]["texture"]["mode"] == "per_prim"


def test_regenerate_material_only_override_preserves_per_prim_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regenerate should not downgrade a stored per-prim session config."""
    sid = "session-regenerate-mt-preserve-per-prim"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    session_dir = manager.get_session_dir(sid)
    config_path = session_dir / "input" / "config.yaml"
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_data["texture"] = {"mode": "per_prim", "backend": "mock"}
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    captured: dict[str, Any] = {}

    class _StubRegistry:
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

    class _StubBus:
        def clear_session_state(self, *args: Any, **kwargs: Any) -> None:
            return None

        async def seed_pending_session(self, *args: Any, **kwargs: Any) -> None:
            return None

    def fake_execute_pipeline_async(**kwargs: Any) -> Any:
        captured["config"] = kwargs["config_dict"]

        async def noop() -> None:
            return None

        return noop()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _StubBus())
    monkeypatch.setattr(
        pipeline_router, "execute_pipeline_async", fake_execute_pipeline_async
    )
    client = _build_test_client(manager)

    response = client.post(
        f"/pipeline/{sid}/regenerate",
        json={
            "steps": ["generate_textures"],
            "material_textures": {
                "Steel": {"prompt": "brushed steel", "opacity": 0.75}
            },
        },
    )

    assert response.status_code == 202, response.text
    assert captured["config"]["texture"]["mode"] == "per_prim"
    assert captured["config"]["material_textures"]["Steel"] == {
        "prompt": "brushed steel",
        "opacity": 0.75,
    }
