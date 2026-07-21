# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomicity tests for the session.json run lease."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from ...service.session.manager import (
    ACTIVE_RUN_EXPIRES_AT_FIELD,
    ACTIVE_RUN_ID_FIELD,
    SessionManager,
)
from ...service.storage.base import METADATA_KEY
from ...service.storage.local_store import LocalSessionStore
from ...service.workers import executor

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class PausingMetadataCasStore(LocalSessionStore):
    """Pause one metadata CAS so another manager can publish first."""

    def __init__(self, root_dir: str) -> None:
        super().__init__(root_dir)
        self.pause_next_metadata_cas = False
        self.metadata_cas_started = asyncio.Event()
        self.resume_metadata_cas = asyncio.Event()

    async def compare_and_swap_bytes(
        self,
        session_id: str,
        key: str,
        expected: bytes,
        replacement: bytes | None,
        content_type: str | None = None,
    ) -> bool:
        if key == METADATA_KEY and self.pause_next_metadata_cas:
            self.pause_next_metadata_cas = False
            self.metadata_cas_started.set()
            await self.resume_metadata_cas.wait()
        return await super().compare_and_swap_bytes(
            session_id,
            key,
            expected,
            replacement,
            content_type,
        )


class CoordinatedMetadataCasStore(LocalSessionStore):
    """Coordinate metadata writers and inject deterministic CAS contention."""

    def __init__(self, root_dir: str) -> None:
        super().__init__(root_dir)
        self.metadata_cas_attempts = 0
        self.reject_next_metadata_cas = 0
        self._metadata_cas_barrier: asyncio.Barrier | None = None

    def coordinate_next_metadata_cas(self, parties: int = 2) -> None:
        assert self._metadata_cas_barrier is None
        self._metadata_cas_barrier = asyncio.Barrier(parties)

    async def compare_and_swap_bytes(
        self,
        session_id: str,
        key: str,
        expected: bytes,
        replacement: bytes | None,
        content_type: str | None = None,
    ) -> bool:
        if key == METADATA_KEY:
            self.metadata_cas_attempts += 1
            if self.reject_next_metadata_cas:
                self.reject_next_metadata_cas -= 1
                return False
            barrier = self._metadata_cas_barrier
            if barrier is not None:
                position = await barrier.wait()
                if position == 0:
                    self._metadata_cas_barrier = None
        return await super().compare_and_swap_bytes(
            session_id,
            key,
            expected,
            replacement,
            content_type,
        )


def _manager(
    storage_path: Path,
    store: LocalSessionStore,
    clock: list[datetime],
) -> SessionManager:
    manager = SessionManager(
        storage_path,
        store=store,
        run_claim_lease_seconds=10,
        run_claim_heartbeat_seconds=1,
    )
    manager._claim_now = lambda: clock[0]  # type: ignore[method-assign]
    return manager


async def test_expired_report_claim_is_reclaimed_across_instances(
    tmp_path: Path,
) -> None:
    store = LocalSessionStore(str(tmp_path / "shared-store"))
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    owner = _manager(tmp_path / "owner", store, clock)
    successor = _manager(tmp_path / "successor", store, clock)
    session_id = str(uuid4())
    owner_run = "a" * 32
    successor_run = "b" * 32
    publication_id = "c" * 32

    await owner.create_session(session_id)
    await owner.update_session(
        session_id,
        {
            "status": "completed",
            "cache_publications": {
                "dataset": publication_id,
                "predictions": publication_id,
            },
        },
    )

    assert await owner.reserve_legacy_cache_run(session_id, owner_run)
    assert not await successor.reserve_legacy_cache_run(session_id, successor_run)

    clock[0] += timedelta(seconds=11)
    assert await successor.reserve_legacy_cache_run(session_id, successor_run)
    metadata = await successor.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata[ACTIVE_RUN_ID_FIELD] == successor_run
    assert not await owner.release_run(session_id, owner_run)
    assert await successor.release_run(session_id, successor_run)

    malformed = await successor.get_session_metadata(session_id)
    assert malformed is not None
    malformed[ACTIVE_RUN_ID_FIELD] = "d" * 32
    malformed[ACTIVE_RUN_EXPIRES_AT_FIELD] = 123
    await store.put_json(session_id, METADATA_KEY, malformed)
    assert not await owner.reserve_legacy_cache_run(session_id, "e" * 32)


async def test_successor_claim_fences_paused_stale_terminal_write(
    tmp_path: Path,
) -> None:
    store = PausingMetadataCasStore(str(tmp_path / "shared-store"))
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    owner = _manager(tmp_path / "owner", store, clock)
    successor = _manager(tmp_path / "successor", store, clock)
    session_id = str(uuid4())
    owner_run = "a" * 32
    successor_run = "b" * 32

    await owner.create_session(session_id)
    assert await owner.reserve_run(session_id, owner_run)
    assert await owner.update_session_for_run(
        session_id,
        owner_run,
        {"status": "running"},
    )
    clock[0] += timedelta(seconds=11)

    store.pause_next_metadata_cas = True
    stale_terminalization = asyncio.create_task(
        owner.terminalize_and_release_run(
            session_id,
            owner_run,
            {"status": "failed", "error": "stale terminal write"},
        )
    )
    await asyncio.wait_for(store.metadata_cas_started.wait(), timeout=2)
    try:
        assert await successor.reserve_run(session_id, successor_run)
    finally:
        store.resume_metadata_cas.set()

    assert await asyncio.wait_for(stale_terminalization, timeout=2) is False
    metadata = await successor.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata[ACTIVE_RUN_ID_FIELD] == successor_run
    assert ACTIVE_RUN_EXPIRES_AT_FIELD in metadata
    assert metadata["status"] == "running"
    assert "error" not in metadata
    assert not await store.exists(session_id, ".active_run")
    assert await successor.release_run(session_id, successor_run)


async def test_concurrent_reserve_renew_and_release_use_one_metadata_lease(
    tmp_path: Path,
) -> None:
    store = CoordinatedMetadataCasStore(str(tmp_path / "shared-store"))
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    managers = [_manager(tmp_path / f"pod-{index}", store, clock) for index in range(3)]
    session_id = str(uuid4())
    run_ids = ("a" * 32, "b" * 32)
    await managers[0].create_session(session_id)

    store.coordinate_next_metadata_cas()
    accepted = await asyncio.wait_for(
        asyncio.gather(
            managers[0].reserve_run(session_id, run_ids[0]),
            managers[1].reserve_run(session_id, run_ids[1]),
        ),
        timeout=2,
    )
    assert accepted.count(True) == 1
    winner_index = accepted.index(True)
    winner = managers[winner_index]
    peer = managers[1 - winner_index]
    run_id = run_ids[winner_index]

    metadata = await winner.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata[ACTIVE_RUN_ID_FIELD] == run_id
    assert datetime.fromisoformat(metadata[ACTIVE_RUN_EXPIRES_AT_FIELD]) == (
        clock[0] + timedelta(seconds=10)
    )
    assert not await store.exists(session_id, ".active_run")

    clock[0] += timedelta(seconds=4)
    attempts_before = store.metadata_cas_attempts
    store.reject_next_metadata_cas = 1
    assert await winner.renew_run(session_id, run_id)
    assert store.metadata_cas_attempts - attempts_before == 2

    clock[0] += timedelta(seconds=1)
    store.coordinate_next_metadata_cas()
    renewed, released = await asyncio.wait_for(
        asyncio.gather(
            winner.renew_run(session_id, run_id),
            peer.release_run(session_id, run_id),
        ),
        timeout=2,
    )
    assert renewed in (True, False)
    assert released is True
    metadata = await winner.get_session_metadata(session_id)
    assert metadata is not None
    assert ACTIVE_RUN_ID_FIELD not in metadata
    assert ACTIVE_RUN_EXPIRES_AT_FIELD not in metadata

    attempts_before = store.metadata_cas_attempts
    store.reject_next_metadata_cas = 1
    assert await managers[2].reserve_run(session_id, "c" * 32)
    assert store.metadata_cas_attempts - attempts_before == 2

    attempts_before = store.metadata_cas_attempts
    store.reject_next_metadata_cas = 1
    assert await managers[0].release_run(session_id, "c" * 32)
    assert store.metadata_cas_attempts - attempts_before == 2


async def test_paused_cancellation_cannot_downgrade_completed_run(
    tmp_path: Path,
) -> None:
    store = PausingMetadataCasStore(str(tmp_path / "shared-store"))
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    owner = _manager(tmp_path / "owner", store, clock)
    finisher = _manager(tmp_path / "finisher", store, clock)
    session_id = str(uuid4())
    run_id = "a" * 32

    await owner.create_session(session_id)
    assert await owner.reserve_run(session_id, run_id)
    assert await owner.update_session_for_run(
        session_id,
        run_id,
        {"status": "running"},
    )

    store.pause_next_metadata_cas = True
    cancellation = asyncio.create_task(owner.request_cancellation(session_id, run_id))
    await asyncio.wait_for(store.metadata_cas_started.wait(), timeout=2)
    try:
        assert await finisher.terminalize_and_release_run(
            session_id,
            run_id,
            {"status": "completed"},
        )
    finally:
        store.resume_metadata_cas.set()

    assert await asyncio.wait_for(cancellation, timeout=2) is False
    metadata = await finisher.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "completed"
    assert ACTIVE_RUN_ID_FIELD not in metadata


async def test_cancel_marker_is_not_consumed_before_metadata_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PausingMetadataCasStore(str(tmp_path / "shared-store"))
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    owner = _manager(tmp_path / "owner", store, clock)
    canceller = _manager(tmp_path / "canceller", store, clock)
    session_id = str(uuid4())
    run_id = "a" * 32

    await owner.create_session(session_id)
    assert await owner.reserve_run(session_id, run_id)
    assert await owner.update_session_for_run(
        session_id,
        run_id,
        {"status": "running"},
    )

    store.pause_next_metadata_cas = True
    cancellation = asyncio.create_task(
        canceller.request_cancellation(session_id, run_id)
    )
    await asyncio.wait_for(store.metadata_cas_started.wait(), timeout=1)
    monkeypatch.setattr(executor, "_RUN_CANCELLATION_POLL_SECONDS", 0.01)
    guard = asyncio.create_task(executor.maintain_run_claim(owner, session_id, run_id))
    await asyncio.sleep(0.03)
    assert not guard.done()

    store.resume_metadata_cas.set()
    assert await asyncio.wait_for(cancellation, timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(guard, timeout=0.2)


async def test_accepted_cancellation_fences_late_successful_completion(
    tmp_path: Path,
) -> None:
    store = LocalSessionStore(str(tmp_path / "shared-store"))
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    owner = _manager(tmp_path / "owner", store, clock)
    canceller = _manager(tmp_path / "canceller", store, clock)
    session_id = str(uuid4())
    run_id = "a" * 32

    await owner.create_session(session_id)
    assert await owner.reserve_run(session_id, run_id)
    assert await owner.update_session_for_run(
        session_id,
        run_id,
        {"status": "running"},
    )

    assert await canceller.request_cancellation(session_id, run_id)
    assert not await owner.terminalize_and_release_run(
        session_id,
        run_id,
        {"status": "completed"},
    )
    assert await owner.terminalize_and_release_run(
        session_id,
        run_id,
        {"status": "cancelled", "failed_step": "cancelled"},
    )

    metadata = await canceller.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "cancelled"
    assert metadata["failed_step"] == "cancelled"
    assert ACTIVE_RUN_ID_FIELD not in metadata


async def test_paused_cancellation_cannot_target_successor_run(
    tmp_path: Path,
) -> None:
    store = PausingMetadataCasStore(str(tmp_path / "shared-store"))
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    owner = _manager(tmp_path / "owner", store, clock)
    successor = _manager(tmp_path / "successor", store, clock)
    session_id = str(uuid4())
    owner_run = "a" * 32
    successor_run = "b" * 32

    await owner.create_session(session_id)
    assert await owner.reserve_run(session_id, owner_run)
    assert await owner.update_session_for_run(
        session_id,
        owner_run,
        {"status": "running"},
    )

    store.pause_next_metadata_cas = True
    cancellation = asyncio.create_task(
        owner.request_cancellation(session_id, owner_run)
    )
    await asyncio.wait_for(store.metadata_cas_started.wait(), timeout=2)
    clock[0] += timedelta(seconds=11)
    try:
        assert await successor.reserve_run(session_id, successor_run)
    finally:
        store.resume_metadata_cas.set()

    assert await asyncio.wait_for(cancellation, timeout=2) is False
    metadata = await successor.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "running"
    assert metadata[ACTIVE_RUN_ID_FIELD] == successor_run
    assert await successor.is_cancelled(session_id, successor_run) is False
    assert await successor.is_cancelled(session_id, owner_run) is True
