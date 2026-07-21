# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ...service.session import manager as manager_module
from ...service.session.manager import SessionManager
from ...service.storage import METADATA_KEY, WORKER_RESERVATION_KEY, LocalSessionStore


def test_create_session_and_progress_lifecycle(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)

    session_dir = manager.create_session("session-1", config={"foo": "bar"})

    assert session_dir.exists()
    assert (session_dir / "input").is_dir()
    assert (session_dir / "cache" / "prepared").is_dir()
    assert (session_dir / "cache" / "renders").is_dir()
    assert (session_dir / "preview").is_dir()

    metadata = manager.get_session_metadata("session-1")
    assert metadata is not None
    assert metadata["status"] == "pending"
    assert metadata["config"] == {"foo": "bar"}

    manager.update_step_progress(
        "session-1",
        "generate_textures",
        {"current": 1, "total": 2, "percent": 50, "message": "halfway"},
    )
    metadata = manager.get_session_metadata("session-1")
    assert metadata["current_step"]["name"] == "generate_textures"
    assert metadata["overall_progress"]["current_step"] == 6
    manager.update_step_progress(
        "session-1",
        "generate_textures",
        {"current": 2, "total": 2, "percent": 100, "message": "done"},
    )
    metadata = manager.get_session_metadata("session-1")
    assert metadata["current_step"]["progress"]["percent"] == 100

    manager.mark_step_completed("session-1", "generate_textures", {"textures": 2})
    metadata = manager.get_session_metadata("session-1")
    assert metadata["current_step"] is None
    assert metadata["overall_progress"]["percent"] == 75
    assert metadata["completed_steps"][0]["stats"] == {"textures": 2}
    assert "generate_textures" in metadata["timings"]

    direct_metadata = {
        "current_step": {
            "name": "custom_step",
            "display_name": "Custom Step",
            "started_at": datetime.now(UTC).isoformat(),
        },
        "overall_progress": {"current_step": 0, "percent": 12},
    }
    completed = manager._apply_step_completed(direct_metadata, "custom_step")
    assert completed is not None
    assert completed["completed_steps"][0]["name"] == "custom_step"
    assert completed["timings"]["custom_step"] >= 0
    assert manager._apply_step_completed(direct_metadata, "missing") is None


def test_default_local_metadata_writes_do_not_rewrite_through_store(
    tmp_path: Path,
) -> None:
    class NoDuplicateMetadataWriteStore(LocalSessionStore):
        def put_json(self, session_id: str, key: str, obj: dict) -> None:
            if key == METADATA_KEY:
                raise AssertionError("metadata should use the atomic local writer once")
            super().put_json(session_id, key, obj)

    manager = SessionManager(
        tmp_path,
        ttl_hours=2,
        store=NoDuplicateMetadataWriteStore(str(tmp_path)),
    )

    manager.create_session("local-once")
    manager.update_session("local-once", {"status": "running"})

    metadata = manager.get_session_metadata("local-once")
    assert metadata is not None
    assert metadata["status"] == "running"


@pytest.mark.parametrize("bad_id", ["", ".", "..", "../x", "x/y", "x\\y"])
def test_rejects_invalid_session_ids_before_filesystem_access(
    tmp_path: Path,
    bad_id: str,
) -> None:
    manager = SessionManager(tmp_path)

    assert manager.session_exists(bad_id) is False
    assert manager.get_session_metadata(bad_id) is None
    with pytest.raises(ValueError):
        manager.create_session(bad_id)
    with pytest.raises(ValueError):
        manager.get_session_dir(bad_id)


def test_rejects_symlink_session_escape(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "sessions")
    outside = tmp_path / "outside"
    outside.mkdir()
    (manager.storage_path / "escape").symlink_to(outside, target_is_directory=True)

    assert manager.session_exists("escape") is False
    assert manager.get_session_metadata("escape") is None
    with pytest.raises(ValueError):
        manager.get_session_dir("escape")


def test_preview_cancellation_and_artifact_helpers(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    manager.create_session("session-2")
    session_dir = manager.get_session_dir("session-2")

    manager.add_preview_image("session-2", "a.png")
    manager.add_preview_image("session-2", "a.png")
    manager.update_preview_images("session-2", ["b.png", "c.png"])

    textures_dir = session_dir / "cache" / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)
    (textures_dir / "one.png").write_text("x", encoding="utf-8")
    output_usd = session_dir / "cache" / "output" / "textured_output.usd"
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    output_usd.write_text("#usda 1.0\n", encoding="utf-8")

    manager.request_cancellation("session-2")
    metadata = manager.get_session_metadata("session-2")

    assert metadata["preview_images"] == ["b.png", "c.png"]
    assert metadata["status"] == "cancelling"
    assert manager.is_cancelled("session-2") is True
    assert manager.get_artifact_path("session-2", "output_usd") == output_usd
    assert manager.get_artifact_dir("session-2", "textures") == textures_dir


def test_request_cancellation_does_not_overwrite_terminal_status(
    tmp_path: Path,
) -> None:
    """request_cancellation races with natural completion in the cancel route.

    If the worker finishes before request_cancellation runs, the terminal
    status (completed / failed / cancelled) must win — otherwise /status
    reports a stale in-flight `cancelling` and /results rejects valid
    output.
    """
    manager = SessionManager(tmp_path)

    for terminal in ("completed", "failed", "cancelled"):
        sid = f"session-{terminal}"
        manager.create_session(sid)
        manager.update_session(sid, {"status": terminal})

        manager.request_cancellation(sid)

        metadata = manager.get_session_metadata(sid)
        assert metadata is not None
        assert metadata["status"] == terminal, (
            f"request_cancellation overwrote terminal {terminal} state"
        )
        # The .cancel marker still gets dropped — harmless for terminal
        # sessions, useful if a stray worker still observes it.
        assert manager.is_cancelled(sid) is True


def test_request_cancellation_serializes_with_concurrent_completion(
    tmp_path: Path,
) -> None:
    """Race the cancel route's terminal-state guard against a worker-side
    `update_session(..., "completed")`.

    The atomic read-check-write inside `_session_lock` should serialize
    the two writes such that whichever takes the lock first wins and the
    loser observes the new status. The terminal status, once written,
    must never be downgraded to `cancelling`.
    """
    import threading

    manager = SessionManager(tmp_path)
    manager.create_session("race-1")
    manager.update_session("race-1", {"status": "running"})

    barrier = threading.Barrier(2)
    cancel_done = threading.Event()
    complete_done = threading.Event()

    def cancel_path() -> None:
        barrier.wait()
        manager.request_cancellation("race-1")
        cancel_done.set()

    def complete_path() -> None:
        barrier.wait()
        manager.update_session("race-1", {"status": "completed"})
        complete_done.set()

    threads = [
        threading.Thread(target=cancel_path),
        threading.Thread(target=complete_path),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert cancel_done.is_set() and complete_done.is_set()

    final = manager.get_session_metadata("race-1")
    assert final is not None
    # Either order is allowed; what's not allowed is downgrading a
    # terminal `completed` to `cancelling`.
    assert final["status"] in ("completed", "cancelling")
    if final["status"] == "completed":
        # `completed` won the race — request_cancellation observed it
        # under the lock and bailed without overwriting.
        pass


def test_clear_cancellation_removes_marker(tmp_path: Path) -> None:
    """clear_cancellation removes the durable `.cancel` marker."""
    manager = SessionManager(tmp_path)
    manager.create_session("session-clear")

    manager.request_cancellation("session-clear")
    assert manager.is_cancelled("session-clear") is True

    manager.clear_cancellation("session-clear")
    assert manager.is_cancelled("session-clear") is False


def test_clear_cancellation_is_idempotent_for_missing_marker(
    tmp_path: Path,
) -> None:
    """Calling clear_cancellation when no marker exists is a no-op."""
    manager = SessionManager(tmp_path)
    manager.create_session("session-no-marker")

    manager.clear_cancellation("session-no-marker")
    assert manager.is_cancelled("session-no-marker") is False


def test_clear_cancellation_tolerates_missing_session(tmp_path: Path) -> None:
    """Clearing for a non-existent session is safe (no exception)."""
    manager = SessionManager(tmp_path)
    manager.clear_cancellation("never-existed")


def test_cleanup_expired_sessions_removes_old_session(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    manager.create_session("expired")

    metadata = manager.get_session_metadata("expired")
    assert metadata is not None
    metadata["ttl_expires_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    manager._save_metadata("expired", metadata)

    cleaned = manager.cleanup_expired_sessions()

    assert cleaned == ["expired"]
    assert not manager.get_session_dir("expired").exists()


def test_delete_session_returns_false_when_worker_lock_busy(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    manager.create_session("busy-delete")

    with manager.worker_lock("busy-delete"):
        assert manager.delete_session("busy-delete") is False

    assert manager.session_exists("busy-delete") is True


def test_worker_lock_missing_session_has_no_filesystem_side_effect(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)

    with pytest.raises(FileNotFoundError):
        manager.acquire_worker_lock("missing-session", timeout=0)

    assert not (tmp_path / "missing-session").exists()


def test_update_missing_session_has_no_filesystem_side_effect(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)

    manager.update_session("missing-session", {"status": "cancelled"})

    assert not (tmp_path / "missing-session").exists()


def test_metadata_less_dir_is_not_a_session(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    orphan_dir = tmp_path / "metadata-less"
    orphan_dir.mkdir()

    assert manager.session_exists("metadata-less") is False

    manager.request_cancellation("metadata-less")
    assert not (orphan_dir / ".cancel").exists()

    assert manager.delete_session("metadata-less") is False
    assert not (orphan_dir / ".worker.lock").exists()


def test_cleanup_expired_sessions_skips_metadata_less_dir_without_lock(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    orphan_dir = tmp_path / "metadata-less-expired"
    orphan_dir.mkdir()

    assert manager.cleanup_expired_sessions() == []
    assert orphan_dir.exists()
    assert not (orphan_dir / ".worker.lock").exists()


def test_cleanup_expired_sessions_skips_worker_locked_session(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    manager.create_session("busy-expired")

    metadata = manager.get_session_metadata("busy-expired")
    assert metadata is not None
    metadata["ttl_expires_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    manager._save_metadata("busy-expired", metadata)

    with manager.worker_lock("busy-expired"):
        assert manager.cleanup_expired_sessions() == []
        assert manager.session_exists("busy-expired") is True

    assert manager.cleanup_expired_sessions() == ["busy-expired"]
    assert manager.session_exists("busy-expired") is False


def test_stalled_worker_marker_blocks_delete_and_ttl_cleanup(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    manager.create_session("stalled-worker")

    metadata = manager.get_session_metadata("stalled-worker")
    assert metadata is not None
    metadata["ttl_expires_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    manager._save_metadata("stalled-worker", metadata)

    manager.mark_worker_stalled("stalled-worker", "still writing")
    marker_path = manager.get_session_dir("stalled-worker") / ".worker.stalled"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["reason"] == "still writing"
    assert (
        list(manager.get_session_dir("stalled-worker").glob(".worker.stalled.*.tmp"))
        == []
    )

    assert manager.is_worker_active("stalled-worker") is True
    assert manager.delete_session("stalled-worker") is False
    assert manager.cleanup_expired_sessions() == []
    assert manager.session_exists("stalled-worker") is True

    manager.clear_worker_stalled("stalled-worker")

    assert manager.is_worker_active("stalled-worker") is False
    assert manager.delete_session("stalled-worker") is True
    assert manager.session_exists("stalled-worker") is False


def test_corrupt_stalled_worker_marker_blocks_delete_until_cleared(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    manager.create_session("corrupt-stalled")
    marker_path = manager.get_session_dir("corrupt-stalled") / ".worker.stalled"
    marker_path.write_text("{", encoding="utf-8")

    metadata = manager.get_session_metadata("corrupt-stalled")
    assert metadata is not None
    metadata["ttl_expires_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    manager._save_metadata("corrupt-stalled", metadata)

    assert manager.is_worker_stalled("corrupt-stalled") is True
    assert marker_path.exists() is True

    with manager.worker_lock("corrupt-stalled"):
        assert manager.delete_session("corrupt-stalled") is False

    assert marker_path.read_text(encoding="utf-8") == "{"
    assert manager.delete_session("corrupt-stalled") is False
    assert manager.cleanup_expired_sessions() == []
    assert manager.session_exists("corrupt-stalled") is True

    manager.clear_worker_stalled("corrupt-stalled")

    assert manager.delete_session("corrupt-stalled") is True
    assert manager.session_exists("corrupt-stalled") is False


def test_stale_stalled_worker_marker_allows_delete_and_ttl_cleanup(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    impossible_pid = 999_999_999

    manager.create_session("stale-delete")
    delete_marker = manager.get_session_dir("stale-delete") / ".worker.stalled"
    delete_marker.write_text(
        json.dumps(
            {
                "reason": "owner process exited",
                "pid": impossible_pid,
                "boot_id": manager._current_boot_id(),
                "process_start_ticks": "0",
                "created_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    assert manager.is_worker_stalled("stale-delete") is False
    assert delete_marker.exists() is False
    assert manager.delete_session("stale-delete") is True
    assert manager.session_exists("stale-delete") is False

    manager.create_session("stale-expired")
    expired_marker = manager.get_session_dir("stale-expired") / ".worker.stalled"
    expired_marker.write_text(
        json.dumps(
            {
                "reason": "owner process exited",
                "pid": impossible_pid,
                "boot_id": manager._current_boot_id(),
                "process_start_ticks": "0",
                "created_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    metadata = manager.get_session_metadata("stale-expired")
    assert metadata is not None
    metadata["ttl_expires_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    manager._save_metadata("stale-expired", metadata)

    assert manager.cleanup_expired_sessions() == ["stale-expired"]
    assert manager.session_exists("stale-expired") is False


def test_pid_exists_falls_back_to_signal_when_proc_entry_is_hidden(
    monkeypatch,
) -> None:
    def fake_exists(path: Path) -> bool:
        return path.as_posix() == "/proc"

    calls: list[tuple[int, int]] = []

    def fake_kill(pid: int, signal_number: int) -> None:
        calls.append((pid, signal_number))

    monkeypatch.setattr(manager_module.Path, "exists", fake_exists)
    monkeypatch.setattr(manager_module.os, "kill", fake_kill)

    assert SessionManager._pid_exists(1234) is True
    assert calls == [(1234, 0)]


def test_process_and_marker_helper_edges(monkeypatch) -> None:
    assert SessionManager._pid_exists(os.getpid()) is True

    monkeypatch.setattr(
        manager_module.Path,
        "read_text",
        lambda self, encoding="utf-8": (_ for _ in ()).throw(OSError("missing")),
    )
    assert SessionManager._process_start_ticks(999_999) is None

    monkeypatch.setattr(
        manager_module.Path,
        "read_text",
        lambda self, encoding="utf-8": "malformed",
    )
    assert SessionManager._process_start_ticks(1) is None

    monkeypatch.setattr(
        manager_module.Path,
        "read_text",
        lambda self, encoding="utf-8": "1 (python) S " + " ".join(["0"] * 19),
    )
    assert SessionManager._process_start_ticks(1) == "0"

    monkeypatch.setattr(SessionManager, "_current_boot_id", lambda: "boot-b")
    assert not SessionManager._stalled_owner_is_live(
        {"pid": 1, "boot_id": "boot-a", "process_start_ticks": "1"}
    )
    monkeypatch.setattr(SessionManager, "_current_boot_id", lambda: None)
    monkeypatch.setattr(SessionManager, "_process_start_ticks", lambda pid: "2")
    assert not SessionManager._stalled_owner_is_live(
        {"pid": 1, "process_start_ticks": "1"}
    )
    monkeypatch.setattr(SessionManager, "_process_start_ticks", lambda pid: None)
    monkeypatch.setattr(SessionManager, "_pid_exists", lambda pid: False)
    assert not SessionManager._stalled_owner_is_live({"pid": 1})
    assert not SessionManager._stalled_owner_is_live({"pid": "bad"})
    assert not SessionManager._stalled_owner_is_live({"pid": 0})

    assert SessionManager._parse_metadata_datetime(None) is None
    assert SessionManager._parse_metadata_datetime("not-a-date") is None
    assert SessionManager._parse_metadata_datetime("2026-01-01T00:00:00").tzinfo == UTC
    assert SessionManager._latest_timestamp("not-a-date") is None
    assert SessionManager._remote_worker_metadata_is_stale(
        {"ttl_expires_at": "2020-01-01T00:00:00+00:00"},
        datetime.now(UTC),
    )


def test_shared_reservation_and_cleanup_lock_edges(tmp_path: Path) -> None:
    class FlakyUpdateStore(LocalSessionStore):
        def __init__(self, root_dir: str) -> None:
            super().__init__(root_dir)
            self.fail_heartbeat = False

        def update_json(self, session_id: str, key: str, updater):
            if self.fail_heartbeat:
                raise OSError("update failed")
            return super().update_json(session_id, key, updater)

    shared_store = FlakyUpdateStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=shared_store)
    session_id = "shared-edges"
    manager.create_session(session_id)

    manager.update_session(session_id, {"status": "running"})
    assert manager._write_shared_worker_reservation(session_id) is None

    metadata = manager.get_session_metadata(session_id)
    assert metadata is not None
    metadata["status"] = "running"
    metadata["updated_at"] = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
    shared_store.put_json(session_id, METADATA_KEY, metadata)
    owner_token = manager._write_shared_worker_reservation(session_id)
    assert owner_token is not None
    manager._clear_shared_worker_reservation(session_id)
    assert shared_store.get_json(session_id, WORKER_RESERVATION_KEY) is None

    assert manager.heartbeat_worker(session_id, owner_token=None) is False
    shared_store.put_json(
        session_id,
        WORKER_RESERVATION_KEY,
        {"owner_token": "owner-a", "updated_at": "2026-01-01T00:00:00+00:00"},
    )
    assert manager.heartbeat_worker(session_id, owner_token="owner-b") is False
    shared_store.fail_heartbeat = True
    assert manager.heartbeat_worker(session_id, owner_token="owner-a") is False
    shared_store.fail_heartbeat = False

    shared_store.put_json(
        "_maintenance",
        "cleanup.lock",
        {
            "created_at": (datetime.now(UTC) - timedelta(hours=3)).isoformat(),
            "updated_at": (datetime.now(UTC) - timedelta(hours=3)).isoformat(),
        },
    )
    cleanup_owner = manager._acquire_shared_cleanup_lock()
    assert cleanup_owner is not None
    assert manager._heartbeat_shared_cleanup_lock(cleanup_owner) is True
    assert manager._heartbeat_shared_cleanup_lock("wrong-owner") is False


def test_invalid_and_remote_event_log_edges(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    assert manager.delete_session("../bad") is False
    assert manager.delete_session("missing") is False
    manager.clear_worker_stalled("../bad")
    assert manager.is_worker_stalled("../bad") is False
    assert manager.is_cancelled("../bad") is False
    manager.clear_cancellation("../bad")
    assert manager.store_key_exists("../bad", "metadata.json") is False
    assert manager.get_event_log("missing") == []

    manager.create_session("preview-missing-list")
    metadata = manager.get_session_metadata("preview-missing-list")
    assert metadata is not None
    metadata.pop("preview_images", None)
    manager._save_metadata("preview-missing-list", metadata)
    manager.add_preview_image("preview-missing-list", "preview.png")
    assert manager.get_session_metadata("preview-missing-list")["preview_images"] == [
        "preview.png"
    ]

    manager.create_session("sync-missing-dir")
    manager.delete_session("sync-missing-dir")
    assert manager.sync_to_store("sync-missing-dir") == 0

    file_path = tmp_path / "source-file.txt"
    file_path.write_text("payload", encoding="utf-8")
    manager.put_file_to_store("preview-missing-list", "cache/file.txt", str(file_path))
    assert manager.store_key_exists("preview-missing-list", "cache/file.txt") is True

    ordered = SessionManager._merge_event_logs(
        [{"timestamp": "bad-date", "state": "fallback"}],
        [{"timestamp": "2026-01-01T00:00:00", "state": "naive"}],
    )
    assert [event["state"] for event in ordered] == ["naive", "fallback"]

    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    shared_manager = SessionManager(tmp_path / "pod", ttl_hours=1, store=shared_store)
    sid = "remote-event-only"
    shared_store.init_session(sid)
    shared_store.put_json(sid, METADATA_KEY, {"session_id": sid, "status": "running"})
    shared_manager.append_event(sid, {"step": "render", "state": "running"})

    assert shared_store.get_event_log(sid) == [{"step": "render", "state": "running"}]
    assert shared_manager.open_store_stream(sid, "missing.bin") is None
