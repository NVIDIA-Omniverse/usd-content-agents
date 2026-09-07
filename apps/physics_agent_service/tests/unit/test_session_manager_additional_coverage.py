# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import io
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from ...service.session import manager as manager_mod
from ...service.session.manager import SessionManager
from ...service.storage import local_store as local_store_mod
from ...service.storage.base import (
    METADATA_KEY,
    CompletedSessionSnapshot,
    SessionGeneration,
)
from ...service.storage.local_store import LocalSessionStore


class _FailingDeleteStore(LocalSessionStore):
    async def delete_session(self, session_id: str) -> None:
        raise RuntimeError("store delete failed")

    async def delete_session_if_terminal(self, session_id: str) -> bool:
        raise RuntimeError("store terminal delete failed")


class _StreamStore(LocalSessionStore):
    async def open_read(self, session_id: str, key: str) -> io.BytesIO:
        return io.BytesIO((self._session_dir(session_id) / key).read_bytes())


class _GenerationAdoptionStore:
    def __init__(self) -> None:
        self.generation = SessionGeneration(2, "owner-2")
        self.adopted: tuple[str, str, SessionGeneration] | None = None

    async def begin_generation(self, _session_id: str) -> SessionGeneration:
        return self.generation

    async def adopt_local_generation(
        self,
        session_id: str,
        local_session_dir: str,
        generation: SessionGeneration,
    ) -> None:
        self.adopted = (session_id, local_session_dir, generation)


class _FailingGenerationAdoptionStore(_GenerationAdoptionStore):
    async def adopt_local_generation(
        self,
        session_id: str,
        local_session_dir: str,
        generation: SessionGeneration,
    ) -> None:
        raise RuntimeError("local marker unavailable")


class _CancelledGenerationAdoptionStore(_GenerationAdoptionStore):
    async def adopt_local_generation(
        self,
        session_id: str,
        local_session_dir: str,
        generation: SessionGeneration,
    ) -> None:
        raise asyncio.CancelledError


def _sid() -> str:
    return str(uuid4())


@pytest.mark.asyncio
async def test_begin_generation_adopts_proven_local_cache(tmp_path: Path) -> None:
    store = _GenerationAdoptionStore()
    manager = SessionManager(tmp_path, store=store)  # type: ignore[arg-type]
    session_id = _sid()

    generation = await manager.begin_generation(session_id)

    assert generation == store.generation
    assert store.adopted == (
        session_id,
        str(manager.get_session_dir(session_id)),
        generation,
    )


@pytest.mark.asyncio
async def test_begin_generation_defers_adoption_error_for_caller_rollback(
    tmp_path: Path,
) -> None:
    store = _FailingGenerationAdoptionStore()
    manager = SessionManager(tmp_path, store=store)  # type: ignore[arg-type]
    claim_observed = asyncio.Event()

    async def begin_then_suspend() -> None:
        assert await manager.begin_generation(_sid()) == store.generation
        claim_observed.set()
        await asyncio.sleep(0)

    task = asyncio.create_task(begin_then_suspend())
    await claim_observed.wait()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_begin_generation_defers_adoption_cancellation_for_caller_rollback(
    tmp_path: Path,
) -> None:
    store = _CancelledGenerationAdoptionStore()
    manager = SessionManager(tmp_path, store=store)  # type: ignore[arg-type]
    claim_observed = asyncio.Event()

    async def begin_then_suspend() -> None:
        assert await manager.begin_generation(_sid()) == store.generation
        claim_observed.set()
        await asyncio.sleep(0)

    task = asyncio.create_task(begin_then_suspend())
    await claim_observed.wait()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store_type", "error_type"),
    [
        (_FailingGenerationAdoptionStore, RuntimeError),
        (_CancelledGenerationAdoptionStore, asyncio.CancelledError),
    ],
)
async def test_begin_generation_reraises_when_not_running_in_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    store_type,
    error_type: type[BaseException],
) -> None:
    manager = SessionManager(
        tmp_path,
        store=store_type(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(manager_mod.asyncio, "current_task", lambda: None)

    with pytest.raises(error_type):
        await manager.begin_generation(_sid())


@pytest.mark.asyncio
async def test_generation_lease_compatibility_and_cancellation_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyStore:
        pass

    class ImmediateEvent:
        async def wait(self) -> None:
            return None

    legacy_manager = SessionManager(tmp_path / "legacy", store=LegacyStore())  # type: ignore[arg-type]
    monkeypatch.setattr(manager_mod.asyncio, "Event", ImmediateEvent)
    assert await legacy_manager.maintain_generation_lease(_sid()) is True

    class GenerationStore:
        async def owns_active_generation(self, _session_id: str) -> bool:
            return True

    generation_manager = SessionManager(
        tmp_path / "generation",
        store=GenerationStore(),  # type: ignore[arg-type]
    )
    task = asyncio.create_task(
        generation_manager.maintain_generation_lease(_sid(), interval_seconds=60)
    )
    await asyncio.sleep(0)
    task.cancel()
    assert await task is False


@pytest.mark.asyncio
async def test_generation_lease_retries_transient_renewal_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GenerationStore:
        def __init__(self) -> None:
            self.calls = 0

        async def owns_active_generation(self, _session_id: str) -> bool:
            self.calls += 1
            if self.calls < 3:
                raise OSError("transient renewal failure")
            return False

    store = GenerationStore()
    manager = SessionManager(
        tmp_path / "transient-generation",
        store=store,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(manager_mod, "_LEASE_HEARTBEAT_RETRY_SECONDS", 0)

    await manager.maintain_generation_lease(_sid(), interval_seconds=60)
    assert store.calls == 3


@pytest.mark.asyncio
async def test_generation_lease_keeps_retrying_through_persistent_outage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GenerationStore:
        def __init__(self) -> None:
            self.calls = 0

        async def owns_active_generation(self, _session_id: str) -> bool:
            self.calls += 1
            if self.calls <= 5:
                raise OSError("persistent renewal failure")
            return False

    store = GenerationStore()
    manager = SessionManager(
        tmp_path / "failed-generation",
        store=store,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(manager_mod, "_LEASE_HEARTBEAT_RETRY_SECONDS", 0)

    await manager.maintain_generation_lease(_sid(), interval_seconds=60)
    assert store.calls == 6


@pytest.mark.asyncio
async def test_generation_lease_rechecks_ownership_at_capacity_handoff(
    tmp_path: Path,
) -> None:
    class GenerationStore:
        def __init__(self) -> None:
            self.calls = 0

        async def owns_active_generation(self, _session_id: str) -> bool:
            self.calls += 1
            return self.calls == 1

    store = GenerationStore()
    manager = SessionManager(
        tmp_path / "handoff-generation",
        store=store,  # type: ignore[arg-type]
    )
    capacity_ready = asyncio.Event()
    heartbeat = asyncio.create_task(
        manager.maintain_generation_lease(
            _sid(),
            interval_seconds=60,
            capacity_ready=capacity_ready,
        )
    )
    while store.calls == 0:
        await asyncio.sleep(0)
    capacity_ready.set()
    assert await heartbeat is False
    assert store.calls == 2


@pytest.mark.asyncio
async def test_generation_lease_capacity_handoff_ready_and_timeout_paths(
    tmp_path: Path,
) -> None:
    class GenerationStore:
        def __init__(self, results: list[bool]) -> None:
            self.results = results

        async def owns_active_generation(self, _session_id: str) -> bool:
            return self.results.pop(0)

    ready = asyncio.Event()
    ready.set()
    ready_manager = SessionManager(
        tmp_path / "ready-handoff",
        store=GenerationStore([True]),  # type: ignore[arg-type]
    )
    assert (
        await ready_manager.maintain_generation_lease(
            _sid(),
            capacity_ready=ready,
        )
        is True
    )

    timeout_manager = SessionManager(
        tmp_path / "timeout-handoff",
        store=GenerationStore([True, False]),  # type: ignore[arg-type]
    )
    assert (
        await timeout_manager.maintain_generation_lease(
            _sid(),
            interval_seconds=0,
            capacity_ready=asyncio.Event(),
        )
        is False
    )


@pytest.mark.asyncio
async def test_manager_delegates_completed_publication_snapshot(tmp_path: Path) -> None:
    session_id = _sid()
    expected = CompletedSessionSnapshot(
        metadata={"status": "completed"},
        artifact_keys=("cache/physics/scene_physics.usda",),
        downloaded_count=1,
    )

    class Store:
        async def sync_completed_publication_to_local(
            self,
            session_id: str,
            destination_dir: str,
            *,
            prefix: str,
        ) -> CompletedSessionSnapshot:
            assert session_id == expected_session_id
            assert destination_dir == str(tmp_path / "snapshot")
            assert prefix == "cache/physics/"
            return expected

    expected_session_id = session_id
    manager = SessionManager(tmp_path / "manager", store=Store())  # type: ignore[arg-type]
    assert (
        await manager.snapshot_completed_publication(
            session_id,
            tmp_path / "snapshot",
            prefix="cache/physics/",
        )
        == expected
    )


@pytest.mark.asyncio
async def test_metadata_update_legacy_store_fallbacks(tmp_path: Path) -> None:
    class LegacyStore:
        def __init__(self) -> None:
            self.metadata: dict | None = {"value": 1}
            self.writes: list[dict] = []

        async def get_json(self, _session_id: str, _key: str):
            return self.metadata

        async def put_json(self, _session_id: str, _key: str, value: dict) -> None:
            self.writes.append(value)
            self.metadata = value

    store = LegacyStore()
    manager = SessionManager(tmp_path, store=store)  # type: ignore[arg-type]
    sid = _sid()

    updated = await manager._update_metadata_document(
        sid,
        lambda current: {**current, "value": 2},
    )
    assert updated == {"value": 2}
    assert store.writes == [{"value": 2}]
    assert await manager._update_metadata_document(sid, lambda _current: None) == {
        "value": 2
    }
    store.metadata = None
    assert await manager._update_metadata_document(sid, lambda value: value) is None


@pytest.mark.asyncio
async def test_conditional_session_updates_and_generation_cancellation_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    sid = _sid()
    await manager.create_session(sid)
    metadata = await manager.get_session_metadata(sid)
    metadata["created_at"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
    await manager.store.put_json(sid, METADATA_KEY, metadata)

    async def conditional_update(
        session_id: str,
        key: str,
        updater,
    ):
        current = await manager.store.get_json(session_id, key)
        assert current is not None
        return updater(current)

    monkeypatch.setattr(
        manager.store,
        "update_json_if_not_cancelled",
        conditional_update,
        raising=False,
    )
    assert await manager.update_session_if_not_cancelled(sid, {"status": "completed"})

    async def reject_update(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        manager.store,
        "update_json_if_not_cancelled",
        reject_update,
        raising=False,
    )
    assert not await manager.update_session_if_not_cancelled(sid, {"status": "failed"})

    async def reject_cancellation(*_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr(
        manager.store,
        "request_generation_cancellation",
        reject_cancellation,
        raising=False,
    )
    assert not await manager.request_pipeline_cancellation(sid)
    await manager.request_cancellation(sid)

    async def lost_generation(_session_id: str) -> bool:
        return False

    monkeypatch.setattr(
        manager.store,
        "owns_active_generation",
        lost_generation,
        raising=False,
    )
    assert await manager.is_cancelled(sid)


@pytest.mark.asyncio
async def test_noop_step_completion_and_nonterminal_delete(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    sid = _sid()
    await manager.create_session(sid)

    await manager.mark_step_completed(sid, "not-current")
    await manager.store.put_bytes(sid, ".cancel", b"")
    assert not await manager.update_session_if_not_cancelled(
        sid,
        {"status": "completed"},
    )
    assert not await manager.delete_terminal_session(sid)


def test_session_manager_suffix_helpers_cover_config_shapes() -> None:
    assert manager_mod._usd_suffix_from_path(None) is None
    assert manager_mod._usd_suffix_from_path("scene.txt") is None
    assert manager_mod._usd_suffix_from_path("scene.USDA") == ".usda"
    assert (
        manager_mod._configured_output_usd_suffix(
            {"step_configs": {"apply_physics": {"output_usd_path": "out.usdc"}}}
        )
        == ".usdc"
    )
    assert (
        manager_mod._configured_output_usd_suffix(
            {"apply_physics": {"output_usd_path": "out.usd"}}
        )
        == ".usd"
    )
    assert (
        manager_mod._configured_output_usd_suffix({"steps": {"apply_physics": {}}})
        is None
    )


@pytest.mark.asyncio
async def test_session_manager_missing_metadata_paths(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    sid = _sid()

    await manager.update_session(sid, {"status": "running"})
    await manager.update_step_progress(sid, "predict", {"percent": 50})
    await manager.mark_step_completed(sid, "predict")
    await manager.add_preview_image(sid, "preview.png")
    await manager.update_preview_images(sid, ["preview.png"])
    await manager.request_cancellation(sid)

    assert await manager.sync_to_store(sid) == 0


@pytest.mark.asyncio
async def test_session_manager_naive_datetimes_and_unknown_step(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    sid = _sid()
    await manager.create_session(sid)

    metadata = await manager.get_session_metadata(sid)
    metadata["created_at"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
    metadata["current_step"] = {
        "name": "custom",
        "display_name": "custom",
        "started_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        "progress": {},
        "elapsed_seconds": 0,
    }
    metadata.pop("completed_steps", None)
    await manager.store.put_json(sid, METADATA_KEY, metadata)

    await manager.update_session(sid, {"status": "running"})
    await manager.update_step_progress(sid, "custom", {"percent": 42})
    await manager.mark_step_completed(sid, "custom")

    updated = await manager.get_session_metadata(sid)
    assert updated["elapsed_seconds"] >= 0
    assert updated["overall_progress"]["percent"] == 0
    assert updated["completed_steps"][0]["name"] == "custom"

    updated.pop("preview_images", None)
    await manager.store.put_json(sid, METADATA_KEY, updated)
    await manager.add_preview_image(sid, "preview.png")
    updated = await manager.get_session_metadata(sid)
    assert updated["preview_images"] == ["preview.png"]

    manager_mod.STEP_NUMBER["unweighted"] = 6
    try:
        await manager.update_step_progress(sid, "unweighted", {"percent": 42})
        updated = await manager.get_session_metadata(sid)
        assert updated["overall_progress"]["percent"] == 42
    finally:
        manager_mod.STEP_NUMBER.pop("unweighted", None)


@pytest.mark.asyncio
async def test_session_manager_artifact_suffix_fallbacks(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    sid = _sid()
    session_dir = await manager.create_session(sid)
    metadata = await manager.get_session_metadata(sid)
    metadata["config"] = {"input_config": {"usd_path": "scene.usd"}}
    await manager.store.put_json(sid, METADATA_KEY, metadata)
    assert await manager.get_artifact_path(sid, "output_usd") is None

    physics_dir = session_dir / "cache" / "physics"
    (physics_dir / "scene_physics.usda").write_text("#usda\n", encoding="utf-8")
    (physics_dir / "scene_physics.usdc").write_bytes(b"usd")

    path = await manager.get_artifact_path(sid, "output_usd")
    assert path is not None
    assert path.suffix in {".usda", ".usdc"}

    assert await manager.get_artifact_path(sid, "missing") is None
    assert await manager.list_artifact_keys(sid, "dataset") == []

    await manager.store.put_bytes(sid, "cache/physics/scene_physics.usda", b"#usda")
    await manager.store.put_bytes(sid, "cache/physics/scene_physics.usdc", b"usd")
    keys = await manager.list_artifact_keys(sid, "output_usd")
    assert "cache/physics/scene_physics.usda" in keys
    stream = await manager.get_artifact_stream(sid, "output_usd")
    assert stream is not None


@pytest.mark.asyncio
async def test_session_manager_expected_suffix_from_input_files_and_store_keys(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    sid = _sid()
    session_dir = await manager.create_session(sid)
    (session_dir / "input" / "scene.usd").write_text("usd", encoding="utf-8")
    assert await manager._expected_output_usd_suffix(sid, session_dir) == ".usd"

    (session_dir / "input" / "scene.usd").unlink()
    (session_dir / "input" / "arbitrary.usdz").write_text("usd", encoding="utf-8")
    assert await manager._expected_output_usd_suffix(sid, session_dir) == ".usda"

    sid2 = _sid()
    session_dir2 = await manager.create_session(sid2)
    shutil.rmtree(session_dir2 / "input")
    await manager.store.put_bytes(sid2, "input/scene.usdc", b"usd")
    assert await manager._expected_output_usd_suffix(sid2, session_dir2) == ".usdc"

    sid3 = _sid()
    session_dir3 = await manager.create_session(sid3)
    shutil.rmtree(session_dir3 / "input")
    await manager.store.put_bytes(sid3, "input/random.usdc", b"usd")
    assert await manager._expected_output_usd_suffix(sid3, session_dir3) == ".usdc"

    split_store = LocalSessionStore(str(tmp_path / "split-store"))
    split_manager = SessionManager(tmp_path / "split-local", store=split_store)
    sid4 = _sid()
    session_dir4 = await split_manager.create_session(sid4)
    shutil.rmtree(session_dir4 / "input")
    await split_store.put_bytes(sid4, "input/scene.usda", b"usd")
    assert (
        await split_manager._expected_output_usd_suffix(sid4, session_dir4) == ".usda"
    )

    sid5 = _sid()
    session_dir5 = await split_manager.create_session(sid5)
    shutil.rmtree(session_dir5 / "input")
    await split_store.put_bytes(sid5, "input/random.usdz", b"usd")
    assert (
        await split_manager._expected_output_usd_suffix(sid5, session_dir5) == ".usda"
    )


@pytest.mark.asyncio
async def test_session_manager_artifact_stream_paths(tmp_path: Path) -> None:
    store = _StreamStore(str(tmp_path / "store"))
    manager = SessionManager(tmp_path / "local", store=store)
    sid = _sid()
    await manager.create_session(sid)
    await store.put_bytes(sid, "cache/predictions/predictions.jsonl", b"{}\n")
    stream = await manager.get_artifact_stream(sid, "predictions")
    assert stream is not None
    assert stream.read() == b"{}\n"

    assert await manager.get_artifact_stream(sid, "unknown") is None
    assert await manager.get_artifact_stream(sid, "output_usd", key="bad") is None


@pytest.mark.asyncio
async def test_session_manager_delete_failures_and_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing = SessionManager(
        tmp_path / "failing", store=_FailingDeleteStore(str(tmp_path / "s"))
    )
    sid = _sid()
    await failing.create_session(sid)
    assert await failing.delete_session(sid) is False
    with pytest.raises(manager_mod.SessionStoreDeletionError):
        await failing.delete_terminal_session(sid)

    manager = SessionManager(tmp_path / "retry")
    sid = _sid()
    await manager.create_session(sid)
    calls = {"count": 0}
    real_remove_confined_tree = local_store_mod.remove_confined_tree

    def flaky_remove_confined_tree(path, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("busy")
        return real_remove_confined_tree(path, *args, **kwargs)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(
        local_store_mod,
        "remove_confined_tree",
        flaky_remove_confined_tree,
    )
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    assert await manager.delete_session(sid) is True
    assert calls["count"] == 2

    sid = _sid()
    await manager.create_session(sid)
    assert await manager.delete_session(sid) is True

    split_store = LocalSessionStore(str(tmp_path / "delete-store"))
    split = SessionManager(tmp_path / "delete-local", store=split_store)
    sid = _sid()
    await split.create_session(sid)
    assert await split.delete_session(sid) is True


@pytest.mark.asyncio
async def test_session_manager_cleanup_expired_sessions(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    expired = _sid()
    fresh = _sid()
    await manager.create_session(expired)
    await manager.create_session(fresh)
    expired_meta = await manager.get_session_metadata(expired)
    expired_meta["status"] = "completed"
    expired_meta["ttl_expires_at"] = (
        datetime.now(UTC) - timedelta(hours=1)
    ).isoformat()
    await manager.store.put_json(expired, METADATA_KEY, expired_meta)
    fresh_meta = await manager.get_session_metadata(fresh)
    fresh_meta["ttl_expires_at"] = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    await manager.store.put_json(fresh, METADATA_KEY, fresh_meta)
    await manager.store.init_session(_sid())

    assert await manager.cleanup_expired_sessions() == 1
    assert not await manager.session_exists(expired)
    assert await manager.session_exists(fresh)

    manager = SessionManager(tmp_path / "cleanup-extra")
    missing = _sid()
    naive = _sid()
    await manager.create_session(naive)
    naive_meta = await manager.get_session_metadata(naive)
    naive_meta["status"] = "failed"
    naive_meta["ttl_expires_at"] = (
        (datetime.now(UTC) - timedelta(seconds=1)).replace(tzinfo=None).isoformat()
    )
    await manager.store.put_json(naive, METADATA_KEY, naive_meta)

    async def list_extra_sessions() -> list[str]:
        return [missing, naive]

    manager.list_sessions = list_extra_sessions  # type: ignore[method-assign]
    assert await manager.cleanup_expired_sessions() == 1


@pytest.mark.asyncio
async def test_session_manager_cleanup_preserves_active_sessions(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    expired_running = _sid()
    expired_pending = _sid()
    expired_completed = _sid()

    for sid, status in (
        (expired_running, "running"),
        (expired_pending, "pending"),
        (expired_completed, "completed"),
    ):
        await manager.create_session(sid)
        metadata = await manager.get_session_metadata(sid)
        metadata["status"] = status
        metadata["ttl_expires_at"] = (
            datetime.now(UTC) - timedelta(hours=1)
        ).isoformat()
        await manager.store.put_json(sid, METADATA_KEY, metadata)

    assert await manager.cleanup_expired_sessions() == 1
    assert await manager.session_exists(expired_running)
    assert await manager.session_exists(expired_pending)
    assert not await manager.session_exists(expired_completed)


@pytest.mark.asyncio
async def test_session_manager_stale_cache_skips_active_sessions(
    tmp_path: Path,
) -> None:
    class RecordingCleanupStore(LocalSessionStore):
        skip_session_ids: set[str] | None = None

        async def cleanup_stale_local_sessions(
            self,
            local_storage_path: str,
            max_age_hours: float = 24.0,
            skip_session_ids: set[str] | None = None,
        ) -> int:
            self.skip_session_ids = skip_session_ids
            return 0

    store = RecordingCleanupStore(str(tmp_path / "store"))
    manager = SessionManager(tmp_path / "local", store=store)
    running = _sid()
    completed = _sid()
    await manager.create_session(running)
    await manager.create_session(completed)
    await manager.update_session(running, {"status": "running"})
    await manager.update_session(completed, {"status": "completed"})

    assert await manager.cleanup_stale_local_cache(max_age_hours=1) == 0
    assert store.skip_session_ids == {running}


@pytest.mark.asyncio
async def test_session_manager_sync_from_store_creates_local_dir(
    tmp_path: Path,
) -> None:
    store = LocalSessionStore(str(tmp_path / "store"))
    manager = SessionManager(tmp_path / "local", store=store)
    sid = _sid()
    await store.put_bytes(sid, "cache/dataset/dataset.jsonl", b"{}\n")

    assert await manager.sync_from_store(sid, prefix="cache/dataset/") == 1
    assert (
        tmp_path / "local" / sid / "cache" / "dataset" / "dataset.jsonl"
    ).read_bytes() == b"{}\n"
