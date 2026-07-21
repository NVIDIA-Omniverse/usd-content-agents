# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distributed metadata fencing tests for multi-instance regeneration."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from ...service.artifact_lineage import initial_artifact_validity
from ...service.runtime.bus import EventBus
from ...service.runtime.events import ProgressEvent, StepState
from ...service.session import manager as manager_module
from ...service.session.manager import (
    RegenerationClaim,
    RegenerationClaimConflictError,
    SessionManager,
)
from ...service.storage import local_store as local_store_module
from ...service.storage.base import (
    METADATA_KEY,
    JsonPreconditionError,
    SessionMetadataContentionError,
)
from ...service.storage.local_store import LocalSessionStore


class _ConflictOnceStore(LocalSessionStore):
    """Inject one metadata CAS loss while retaining the real local store."""

    fail_next_replace = False

    async def replace_json_if_version(
        self,
        session_id: str,
        key: str,
        obj: dict,
        expected_version: str | None,
    ) -> str:
        if self.fail_next_replace:
            self.fail_next_replace = False
            raise JsonPreconditionError("injected contention")
        return await super().replace_json_if_version(
            session_id,
            key,
            obj,
            expected_version,
        )


class _RemoteKindStore(LocalSessionStore):
    @property
    def kind(self) -> str:
        return "remote-test"


def _manager_pair(tmp_path: Path) -> tuple[SessionManager, SessionManager]:
    shared_root = tmp_path / "shared"
    return (
        SessionManager(
            tmp_path / "pod-a",
            store=LocalSessionStore(str(shared_root)),
        ),
        SessionManager(
            tmp_path / "pod-b",
            store=LocalSessionStore(str(shared_root)),
        ),
    )


async def _create_completed_session(manager: SessionManager) -> str:
    session_id = str(uuid4())
    await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "completed"}, sync_files=False)
    return session_id


async def _expire_claim(manager: SessionManager, session_id: str) -> None:
    snapshot = await manager.get_session_metadata_versioned(session_id)
    assert snapshot.value is not None
    assert snapshot.version is not None
    metadata = dict(snapshot.value)
    raw_claim = dict(metadata["regeneration_claim"])
    raw_claim["lease_expires_at"] = (
        datetime.now(UTC) - timedelta(seconds=1)
    ).isoformat()
    metadata["regeneration_claim"] = raw_claim
    await manager.store.replace_json_if_version(
        session_id,
        METADATA_KEY,
        metadata,
        snapshot.version,
    )


async def _replace_metadata_fields_unchecked(
    manager: SessionManager,
    session_id: str,
    updates: dict[str, Any],
) -> None:
    """Install deliberately malformed/historical fenced test fixtures."""
    snapshot = await manager.get_session_metadata_versioned(session_id)
    assert snapshot.value is not None
    assert snapshot.version is not None
    metadata = dict(snapshot.value)
    metadata.update(updates)
    await manager.store.replace_json_if_version(
        session_id,
        METADATA_KEY,
        metadata,
        snapshot.version,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_json_compare_and_swap_rejects_stale_versions(
    tmp_path: Path,
) -> None:
    store = LocalSessionStore(str(tmp_path))
    created_version = await store.replace_json_if_version(
        "session", METADATA_KEY, {"value": 1}, None
    )
    assert (await store.get_json_versioned("session", METADATA_KEY)).version == (
        created_version
    )

    with pytest.raises(JsonPreconditionError):
        await store.replace_json_if_version(
            "session", METADATA_KEY, {"value": "stale-create"}, None
        )

    next_version = await store.replace_json_if_version(
        "session", METADATA_KEY, {"value": 2}, created_version
    )
    assert next_version != created_version
    with pytest.raises(JsonPreconditionError):
        await store.replace_json_if_version(
            "session", METADATA_KEY, {"value": "stale-update"}, created_version
        )
    assert await store.get_json("session", METADATA_KEY) == {"value": 2}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_json_lock_wait_does_not_block_event_loop(tmp_path: Path) -> None:
    store = LocalSessionStore(str(tmp_path))
    await store.put_json("session", METADATA_KEY, {"value": 1})
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_lock() -> None:
        with store._json_lock("session", METADATA_KEY):
            lock_held.set()
            release_lock.wait(timeout=2)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert await asyncio.to_thread(lock_held.wait, 1)
    threading.Timer(0.1, release_lock.set).start()

    read_task = asyncio.create_task(store.get_json_versioned("session", METADATA_KEY))
    ticker_iterations = 0

    async def tick_while_waiting() -> None:
        nonlocal ticker_iterations
        while not read_task.done():
            ticker_iterations += 1
            await asyncio.sleep(0.001)

    try:
        result, _ = await asyncio.gather(read_task, tick_while_waiting())
    finally:
        release_lock.set()
        await asyncio.to_thread(holder.join, 1)

    assert result.value == {"value": 1}
    assert ticker_iterations > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_repeatedly_cancelled_local_json_write_drains_before_propagating(
    tmp_path: Path,
) -> None:
    store = LocalSessionStore(str(tmp_path))
    await store.put_json("session", METADATA_KEY, {"value": 1})
    snapshot = await store.get_json_versioned("session", METADATA_KEY)
    assert snapshot.version is not None
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_lock() -> None:
        with store._json_lock("session", METADATA_KEY):
            lock_held.set()
            release_lock.wait(timeout=2)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert await asyncio.to_thread(lock_held.wait, 1)
    write_task = asyncio.create_task(
        store.replace_json_if_version(
            "session",
            METADATA_KEY,
            {"value": 2},
            snapshot.version,
        )
    )

    try:
        await asyncio.sleep(0.02)
        assert write_task.cancel()
        await asyncio.sleep(0.02)
        assert not write_task.done()
        assert write_task.cancel()
        await asyncio.sleep(0.02)
        assert not write_task.done()
        release_lock.set()
        with pytest.raises(asyncio.CancelledError):
            await write_task
    finally:
        release_lock.set()
        await asyncio.to_thread(holder.join, 1)

    assert await store.get_json("session", METADATA_KEY) == {"value": 2}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_local_io_remains_cancelled_when_worker_fails(
    tmp_path: Path,
) -> None:
    store = LocalSessionStore(str(tmp_path))
    operation_started = threading.Event()
    release_operation = threading.Event()

    def failing_operation() -> None:
        operation_started.set()
        release_operation.wait(timeout=2)
        raise OSError("injected worker failure")

    operation_task = asyncio.create_task(store._run_locked_io(failing_operation))
    assert await asyncio.to_thread(operation_started.wait, 1)

    try:
        assert operation_task.cancel()
        await asyncio.sleep(0.02)
        assert not operation_task.done()
        release_operation.set()
        with pytest.raises(asyncio.CancelledError):
            await operation_task
    finally:
        release_operation.set()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_json_lock_timeout_is_retryable_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSessionStore(str(tmp_path))
    await store.put_json("session", METADATA_KEY, {"value": 1})
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_lock() -> None:
        with store._json_lock("session", METADATA_KEY):
            lock_held.set()
            release_lock.wait(timeout=2)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert await asyncio.to_thread(lock_held.wait, 1)
    monkeypatch.setattr(local_store_module, "_JSON_LOCK_TIMEOUT_SECONDS", 0.01)

    try:
        with pytest.raises(SessionMetadataContentionError):
            await store.get_json_versioned("session", METADATA_KEY)
    finally:
        release_lock.set()
        await asyncio.to_thread(holder.join, 1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metadata_cas_exhaustion_uses_contention_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    await manager.create_session(session_id)
    attempts = 0

    async def always_conflict(*_args: Any, **_kwargs: Any) -> str:
        nonlocal attempts
        attempts += 1
        raise JsonPreconditionError("injected contention")

    monkeypatch.setattr(manager_module, "_MAX_CAS_ATTEMPTS", 2)
    monkeypatch.setattr(manager.store, "replace_json_if_version", always_conflict)

    with pytest.raises(SessionMetadataContentionError):
        await manager.update_session(session_id, {"status": "running"})

    assert attempts == 2


@pytest.mark.unit
def test_claim_parser_and_metadata_update_defensive_contracts() -> None:
    assert RegenerationClaim.from_metadata({}) is None
    assert RegenerationClaim.from_metadata({"regeneration_claim": {}}) is None
    assert (
        RegenerationClaim.from_metadata(
            {
                "regeneration_claim": {
                    "generation": 1,
                    "token": "token",
                    "lease_expires_at": "not-a-date",
                }
            }
        )
        is None
    )
    parsed = RegenerationClaim.from_metadata(
        {
            "regeneration_claim": {
                "generation": 1,
                "token": "token",
                "lease_expires_at": "2030-01-01T00:00:00",
            }
        }
    )
    assert parsed is not None
    assert parsed.lease_expires_at.tzinfo is UTC

    metadata: dict[str, Any] = {"created_at": "not-a-date", "remove": True}
    SessionManager._apply_metadata_updates(
        metadata,
        {"value": 1},
        ("remove",),
        now=datetime.now(UTC),
    )
    assert metadata["value"] == 1
    assert "remove" not in metadata
    with pytest.raises(ValueError, match="Fenced metadata fields"):
        SessionManager._validate_fenced_updates(
            {"published_artifacts": {}},
            (),
        )
    assert not SessionManager._claim_matches(
        {},
        parsed,
        now=datetime.now(UTC),
    )
    claim_metadata = {
        "regeneration_claim": {
            "generation": parsed.generation,
            "token": parsed.token,
            "active": True,
            "lease_expires_at": parsed.lease_expires_at.isoformat(),
        }
    }
    assert SessionManager._claim_matches(
        claim_metadata,
        parsed,
        now=datetime.now(UTC),
        require_active_lease=False,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metadata_batch_and_cancellation_marker_compatibility(
    tmp_path: Path,
) -> None:
    remote_store = _RemoteKindStore(str(tmp_path / "remote"))
    manager = SessionManager(tmp_path / "local", store=remote_store)
    session_id = await _create_completed_session(manager)
    assert (await manager.get_session_metadata_batch([session_id]))[0] is not None

    assert not await manager.is_cancelled(session_id)
    await manager.request_cancellation(session_id)
    assert await manager.is_cancelled(session_id)
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "cancelling"

    await manager.clear_cancellation(session_id)
    assert not await manager.is_cancelled(session_id)
    await manager.restore_cancellation(session_id, cancelled=True)
    assert await manager.is_cancelled(session_id)
    assert (manager.get_session_dir(session_id) / ".cancel").exists()
    await manager.restore_cancellation(session_id, cancelled=False)
    assert not await manager.is_cancelled(session_id)
    assert not (manager.get_session_dir(session_id) / ".cancel").exists()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_claim_validation_conflicts_and_missing_session_paths(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    missing_id = str(uuid4())
    with pytest.raises(RegenerationClaimConflictError, match="not found"):
        await manager.claim_regeneration(
            missing_id,
            expected_version="missing",
        )
    fake_claim = RegenerationClaim(
        generation=1,
        token="missing",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
    )
    assert not await manager.update_session_for_claim(
        missing_id, fake_claim, {"status": "running"}
    )
    assert not await manager.renew_regeneration_claim(missing_id, fake_claim)
    assert not await manager.cancel_regeneration_claim(missing_id, fake_claim)
    assert not await manager.cancel_expired_regeneration_claim(missing_id, fake_claim)
    assert not await manager.is_regeneration_cancel_requested(missing_id, fake_claim)
    assert not await manager.finalize_regeneration_claim(
        missing_id,
        fake_claim,
        updates={"status": "completed"},
    )
    with pytest.raises(ValueError, match="different session"):
        await manager.abort_regeneration_claim(
            missing_id,
            fake_claim,
            restore_metadata={"session_id": str(uuid4())},
        )
    assert not await manager.abort_regeneration_claim(
        missing_id,
        fake_claim,
        restore_metadata={"session_id": missing_id},
    )

    session_id = await _create_completed_session(manager)
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    with pytest.raises(ValueError, match="positive"):
        await manager.claim_regeneration(
            session_id,
            expected_version=planned.version,
            lease_seconds=0,
        )
    with pytest.raises(RegenerationClaimConflictError, match="changed"):
        await manager.claim_regeneration(
            session_id,
            expected_version="stale-version",
        )
    claim = await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        lease_seconds=60,
    )
    latest = await manager.get_session_metadata_versioned(session_id)
    assert latest.version is not None
    with pytest.raises(RegenerationClaimConflictError, match="already active"):
        await manager.claim_regeneration(
            session_id,
            expected_version=latest.version,
        )
    with pytest.raises(ValueError, match="positive"):
        await manager.renew_regeneration_claim(
            session_id,
            claim,
            lease_seconds=0,
        )

    running_id = await _create_completed_session(manager)
    await manager.update_session(
        running_id,
        {"status": "running"},
        sync_files=False,
    )
    running_plan = await manager.get_session_metadata_versioned(running_id)
    assert running_plan.version is not None
    with pytest.raises(RegenerationClaimConflictError, match="while pipeline"):
        await manager.claim_regeneration(
            running_id,
            expected_version=running_plan.version,
        )

    malformed_id = await _create_completed_session(manager)
    await _replace_metadata_fields_unchecked(
        manager,
        malformed_id,
        {"regeneration_claim": {"active": True}},
    )
    malformed_plan = await manager.get_session_metadata_versioned(malformed_id)
    assert malformed_plan.version is not None
    with pytest.raises(RegenerationClaimConflictError, match="invalid active"):
        await manager.claim_regeneration(
            malformed_id,
            expected_version=malformed_plan.version,
        )

    historical_id = await _create_completed_session(manager)
    await _replace_metadata_fields_unchecked(
        manager,
        historical_id,
        {"published_artifacts": {"generation": 5}},
    )
    historical_plan = await manager.get_session_metadata_versioned(historical_id)
    assert historical_plan.version is not None
    historical_claim = await manager.claim_regeneration(
        historical_id,
        expected_version=historical_plan.version,
        lease_seconds=60,
    )
    assert historical_claim.generation == 6


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fenced_operations_retry_conditional_contention(tmp_path: Path) -> None:
    store = _ConflictOnceStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "local", store=store)
    session_id = await _create_completed_session(manager)

    store.fail_next_replace = True
    await manager.update_session(
        session_id,
        {
            "ordinary_update": True,
            "artifact_validity": {
                **initial_artifact_validity(),
                "raw_predictions": True,
            },
            "prediction_lineage_token": "R1",
        },
        sync_files=False,
    )
    report_key = "reports/R1/build/report.html"
    await manager.put_bytes_to_store(session_id, report_key, b"report")
    store.fail_next_replace = True
    assert await manager.publish_prediction_report_if_lineage_matches(
        session_id,
        "R1",
        report_key,
    )
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None

    store.fail_next_replace = True
    with pytest.raises(RegenerationClaimConflictError, match="won"):
        await manager.claim_regeneration(
            session_id,
            expected_version=planned.version,
            lease_seconds=60,
        )
    claim = await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={
            "artifact_validity": {
                **initial_artifact_validity(),
                "raw_predictions": True,
            },
            "prediction_lineage_token": "R1",
        },
        lease_seconds=60,
    )

    store.fail_next_replace = True
    assert await manager.update_session_for_claim(
        session_id,
        claim,
        {"status": "running"},
    )
    store.fail_next_replace = True
    assert await manager.renew_regeneration_claim(
        session_id,
        claim,
        lease_seconds=60,
    )
    store.fail_next_replace = True
    assert await manager.cancel_regeneration_claim(session_id, claim)

    artifact_key = f"{claim.artifact_prefix}/output/scene.usd"
    await manager.put_bytes_to_store(session_id, artifact_key, b"scene")
    store.fail_next_replace = True
    assert await manager.finalize_regeneration_claim(
        session_id,
        claim,
        updates={"status": "cancelled"},
        artifact_map={"output_usd": artifact_key},
    )

    lineage_session = await _create_completed_session(manager)
    await manager.update_session(
        lineage_session,
        {
            "artifact_validity": {
                **initial_artifact_validity(),
                "raw_predictions": True,
            }
        },
        sync_files=False,
    )
    store.fail_next_replace = True
    assert await manager.capture_prediction_lineage(lineage_session)

    abort_session = await _create_completed_session(manager)
    abort_plan = await manager.get_session_metadata_versioned(abort_session)
    assert abort_plan.value is not None
    assert abort_plan.version is not None
    abort_claim = await manager.claim_regeneration(
        abort_session,
        expected_version=abort_plan.version,
        lease_seconds=60,
    )
    store.fail_next_replace = True
    assert await manager.abort_regeneration_claim(
        abort_session,
        abort_claim,
        restore_metadata=abort_plan.value,
    )

    expired_session = await _create_completed_session(manager)
    expired_plan = await manager.get_session_metadata_versioned(expired_session)
    assert expired_plan.version is not None
    expired_claim = await manager.claim_regeneration(
        expired_session,
        expected_version=expired_plan.version,
        lease_seconds=60,
    )
    await _expire_claim(manager, expired_session)
    store.fail_next_replace = True
    assert await manager.cancel_expired_regeneration_claim(
        expired_session,
        expired_claim,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fenced_publication_validation_fails_closed(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session_id = await _create_completed_session(manager)
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    claim = await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        lease_seconds=60,
    )

    with pytest.raises(ValueError, match="Fenced metadata fields"):
        await manager.update_session_for_claim(
            session_id,
            claim,
            {"published_artifacts": {}},
        )
    with pytest.raises(ValueError, match="named strings"):
        await manager.finalize_regeneration_claim(
            session_id,
            claim,
            updates={"status": "completed"},
            artifact_map={"": f"{claim.artifact_prefix}/empty"},
        )
    with pytest.raises(ValueError, match="immutable prefix"):
        await manager.finalize_regeneration_claim(
            session_id,
            claim,
            updates={"status": "completed"},
            artifact_map={"output": "output/mutable.usd"},
        )
    with pytest.raises(FileNotFoundError, match="missing regeneration artifact"):
        await manager.finalize_regeneration_claim(
            session_id,
            claim,
            updates={"status": "completed"},
            artifact_map={"output": f"{claim.artifact_prefix}/missing.usd"},
        )

    with pytest.raises(ValueError, match="non-empty path segment"):
        await manager.publish_prediction_report_if_lineage_matches(
            session_id,
            "bad/lineage",
            "reports/bad/lineage/build/report.html",
        )
    with pytest.raises(ValueError, match="immutable lineage"):
        await manager.publish_prediction_report_if_lineage_matches(
            session_id,
            "R1",
            "reports/R1/report.html",
        )
    assert not await manager.publish_prediction_report_if_lineage_matches(
        session_id,
        "R1",
        "reports/R1/build/missing.html",
    )
    orphan_id = str(uuid4())
    orphan_report_key = "reports/R1/build/orphan.html"
    await manager.put_bytes_to_store(orphan_id, orphan_report_key, b"orphan")
    assert not await manager.publish_prediction_report_if_lineage_matches(
        orphan_id,
        "R1",
        orphan_report_key,
    )

    missing_id = str(uuid4())
    assert await manager.capture_prediction_lineage(missing_id) is None
    no_predictions_id = await _create_completed_session(manager)
    await manager.update_session(
        no_predictions_id,
        {"artifact_validity": initial_artifact_validity()},
        sync_files=False,
    )
    assert await manager.capture_prediction_lineage(no_predictions_id) is None
    assert await manager.finalize_regeneration_claim(
        session_id,
        claim,
        updates={"status": "failed"},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_two_managers_racing_same_version_have_exactly_one_claim_winner(
    tmp_path: Path,
) -> None:
    pod_a, pod_b = _manager_pair(tmp_path)
    session_id = await _create_completed_session(pod_a)
    snapshots = await asyncio.gather(
        pod_a.get_session_metadata_versioned(session_id),
        pod_b.get_session_metadata_versioned(session_id),
    )
    assert snapshots[0].version == snapshots[1].version
    assert snapshots[0].version is not None

    async def attempt(manager: SessionManager):
        try:
            return await manager.claim_regeneration(
                session_id,
                expected_version=snapshots[0].version,
                lease_seconds=60,
            )
        except RegenerationClaimConflictError as exc:
            return exc

    outcomes = await asyncio.gather(attempt(pod_a), attempt(pod_b))
    winners = [item for item in outcomes if isinstance(item, RegenerationClaim)]
    conflicts = [
        item for item in outcomes if isinstance(item, RegenerationClaimConflictError)
    ]
    assert len(winners) == 1
    assert len(conflicts) == 1

    metadata = await pod_b.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["regeneration_claim"]["token"] == winners[0].token
    assert metadata["regeneration_claim"]["active"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generic_metadata_writes_cannot_bypass_fenced_fields_or_live_claim(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = await _create_completed_session(manager)

    for field in (
        "regeneration_claim",
        "published_artifacts",
        "prediction_report_publication",
        "terminal_events_quiesced",
    ):
        with pytest.raises(ValueError, match=field):
            await manager.update_session(
                session_id,
                {field: {}},
                sync_files=False,
            )
        with pytest.raises(ValueError, match=field):
            await manager.update_session(
                session_id,
                {},
                remove_fields=(field,),
                sync_files=False,
            )

    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        lease_seconds=60,
    )
    with pytest.raises(RegenerationClaimConflictError, match="active regeneration"):
        await manager.update_session(
            session_id,
            {"status": "completed"},
            sync_files=False,
        )
    with pytest.raises(RegenerationClaimConflictError, match="active regeneration"):
        await manager.add_preview_image(session_id, "stale.png")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_regeneration_waits_for_standard_terminal_event_quiescence(
    tmp_path: Path,
) -> None:
    pod_a, pod_b = _manager_pair(tmp_path)
    session_id = str(uuid4())
    await pod_a.create_session(session_id)
    bus = EventBus()
    bus.set_session_manager(pod_a)
    await bus.seed_pending_session(session_id)

    assert await pod_a.finalize_standard_pipeline(
        session_id,
        {
            "status": "completed",
            "results": {"owner": "standard"},
            "coverage": None,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    blocked_plan = await pod_b.get_session_metadata_versioned(session_id)
    assert blocked_plan.version is not None
    with pytest.raises(RegenerationClaimConflictError, match="terminal events"):
        await pod_b.claim_regeneration(
            session_id,
            expected_version=blocked_plan.version,
            lease_seconds=60,
        )

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="pipeline",
            state=StepState.COMPLETED,
            percent=100,
            overall_percent=100,
            extra={"pipeline_completed": True, "coverage": None},
        )
    )
    assert await pod_a.mark_terminal_events_quiesced(
        session_id,
        expected_status="completed",
    )
    ready_plan = await pod_b.get_session_metadata_versioned(session_id)
    assert ready_plan.version is not None
    claim = await pod_b.claim_regeneration(
        session_id,
        expected_version=ready_plan.version,
        lease_seconds=60,
    )
    metadata = await pod_a.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["regeneration_claim"]["token"] == claim.token


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancel_request_wins_over_completed_claim_finalize(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = await _create_completed_session(manager)
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    claim = await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "running"},
        lease_seconds=60,
    )

    assert await manager.cancel_regeneration_claim(session_id, claim)
    assert not await manager.finalize_regeneration_claim(
        session_id,
        claim,
        updates={"status": "completed", "results": {"stale": True}},
    )
    assert await manager.finalize_regeneration_claim(
        session_id,
        claim,
        updates={"status": "cancelled", "cancelled_at": datetime.now(UTC).isoformat()},
    )
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "cancelled"
    assert metadata["regeneration_claim"]["cancel_requested"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_standard_cancellation_wins_over_completed_finalize(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    await manager.create_session(session_id)
    await manager.update_session(
        session_id,
        {"status": "cancelling"},
        sync_files=False,
    )

    assert not await manager.finalize_standard_pipeline(
        session_id,
        {"status": "completed", "results": {"stale": True}},
    )
    assert await manager.finalize_standard_pipeline(
        session_id,
        {"status": "cancelled", "cancelled_at": datetime.now(UTC).isoformat()},
    )
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "cancelled"
    assert metadata["terminal_events_quiesced"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finalize_preserves_concurrent_same_lineage_report_publication(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = await _create_completed_session(manager)
    validity = initial_artifact_validity()
    validity["raw_predictions"] = True
    await manager.update_session(
        session_id,
        {
            "artifact_validity": validity,
            "prediction_lineage_token": "R1",
        },
        sync_files=False,
    )
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    claim = await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "running"},
        lease_seconds=60,
    )
    report_key = "reports/R1/concurrent/report.html"
    await manager.put_bytes_to_store(session_id, report_key, b"report")

    stale_validity = dict(validity)
    assert await manager.publish_prediction_report_if_lineage_matches(
        session_id,
        "R1",
        report_key,
    )
    assert await manager.finalize_regeneration_claim(
        session_id,
        claim,
        updates={"status": "completed", "artifact_validity": stale_validity},
    )
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["prediction_report_publication"]["key"] == report_key
    assert metadata["artifact_validity"]["prediction_report"] is True


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add", "remove"])
async def test_generated_reference_cas_loses_to_cross_pod_pipeline_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    pod_a, pod_b = _manager_pair(tmp_path)
    session_id = str(uuid4())
    await pod_a.create_session(session_id)
    await pod_a.update_session(session_id, {"status": "ready"}, sync_files=False)
    if operation == "remove":
        assert await pod_a.add_generated_reference_image(
            session_id,
            {"id": "ref", "key": "input/ref.png"},
        )

    blocked = asyncio.Event()
    resume = asyncio.Event()
    original_replace = pod_a.store.replace_json_if_version
    paused = False

    async def pause_reference_replace(
        replace_session_id: str,
        key: str,
        obj: dict,
        expected_version: str | None,
    ) -> str:
        nonlocal paused
        references = obj.get("generated_reference_images")
        should_pause = (
            key == METADATA_KEY
            and replace_session_id == session_id
            and isinstance(references, list)
            and (
                (operation == "add" and references)
                or (operation == "remove" and not references)
            )
        )
        if should_pause and not paused:
            paused = True
            blocked.set()
            await resume.wait()
        return await original_replace(
            replace_session_id,
            key,
            obj,
            expected_version,
        )

    monkeypatch.setattr(pod_a.store, "replace_json_if_version", pause_reference_replace)
    if operation == "add":
        reference_task = asyncio.create_task(
            pod_a.add_generated_reference_image(
                session_id,
                {"id": "ref", "key": "input/ref.png"},
            )
        )
    else:
        reference_task = asyncio.create_task(
            pod_a.remove_generated_reference_image(session_id, "ref")
        )
    await blocked.wait()
    await pod_b.update_session(
        session_id,
        {"status": "pending"},
        sync_files=False,
    )
    resume.set()
    with pytest.raises(RegenerationClaimConflictError, match="session is ready"):
        await reference_task

    metadata = await pod_b.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "pending"
    references = metadata.get("generated_reference_images", [])
    assert (len(references) == 0) if operation == "add" else (len(references) == 1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_expired_takeover_fences_stale_worker_and_artifact_publish(
    tmp_path: Path,
) -> None:
    pod_a, pod_b = _manager_pair(tmp_path)
    session_id = await _create_completed_session(pod_a)
    planned = await pod_a.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    stale_claim = await pod_a.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "running"},
        lease_seconds=60,
    )
    await _expire_claim(pod_a, session_id)

    takeover_plan = await pod_b.get_session_metadata_versioned(session_id)
    assert takeover_plan.version is not None
    active_claim = await pod_b.claim_regeneration(
        session_id,
        expected_version=takeover_plan.version,
        updates={"status": "running"},
        lease_seconds=60,
    )
    assert active_claim.generation == stale_claim.generation + 1

    assert not await pod_a.update_session_for_claim(
        session_id,
        stale_claim,
        {"status": "failed", "results": {"worker": "stale"}},
    )
    assert not await pod_a.renew_regeneration_claim(session_id, stale_claim)
    assert not await pod_a.cancel_regeneration_claim(session_id, stale_claim)

    stale_key = f"{stale_claim.artifact_prefix}/output/scene.usd"
    active_key = f"{active_claim.artifact_prefix}/output/scene.usd"
    await pod_a.put_bytes_to_store(session_id, stale_key, b"stale")
    await pod_b.put_bytes_to_store(session_id, active_key, b"active")
    assert await pod_b.finalize_regeneration_claim(
        session_id,
        active_claim,
        updates={"status": "completed", "results": {"worker": "active"}},
        artifact_map={"output_usd": active_key},
    )
    assert not await pod_a.finalize_regeneration_claim(
        session_id,
        stale_claim,
        updates={"status": "completed", "results": {"worker": "stale"}},
        artifact_map={"output_usd": stale_key},
    )

    metadata = await pod_a.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "completed"
    assert metadata["results"] == {"worker": "active"}
    assert metadata["published_artifacts"] == {
        "generation": active_claim.generation,
        "token": active_claim.token,
        "prefix": active_claim.artifact_prefix,
        "artifacts": {"output_usd": active_key},
        "published_at": metadata["published_artifacts"]["published_at"],
    }
    assert metadata["regeneration_claim"]["active"] is False
    assert not await pod_b.update_session_for_claim(
        session_id,
        active_claim,
        {"status": "failed"},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_claim_renew_and_cancel_are_token_scoped(tmp_path: Path) -> None:
    pod_a, pod_b = _manager_pair(tmp_path)
    session_id = await _create_completed_session(pod_a)
    planned = await pod_a.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    claim = await pod_a.claim_regeneration(
        session_id,
        expected_version=planned.version,
        lease_seconds=60,
    )

    assert await pod_b.renew_regeneration_claim(
        session_id,
        claim,
        lease_seconds=120,
    )
    assert await pod_b.cancel_regeneration_claim(session_id, claim)
    assert await pod_a.is_regeneration_cancel_requested(session_id, claim)
    metadata = await pod_a.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "cancelling"
    assert metadata["regeneration_claim"]["cancel_requested"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_expired_cancel_recovery_and_takeover_are_one_cas_race(
    tmp_path: Path,
) -> None:
    pod_a, pod_b = _manager_pair(tmp_path)
    session_id = await _create_completed_session(pod_a)
    planned = await pod_a.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    expired_claim = await pod_a.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "running"},
        lease_seconds=60,
    )
    assert not await pod_b.cancel_expired_regeneration_claim(
        session_id,
        expired_claim,
    )
    await _expire_claim(pod_a, session_id)
    takeover_plan = await pod_b.get_session_metadata_versioned(session_id)
    assert takeover_plan.version is not None

    async def attempt_takeover():
        try:
            return await pod_b.claim_regeneration(
                session_id,
                expected_version=takeover_plan.version,
                lease_seconds=60,
            )
        except RegenerationClaimConflictError as exc:
            return exc

    cancel_outcome, takeover_outcome = await asyncio.gather(
        pod_a.cancel_expired_regeneration_claim(session_id, expired_claim),
        attempt_takeover(),
    )
    takeover_won = isinstance(takeover_outcome, RegenerationClaim)
    assert cancel_outcome is not takeover_won

    metadata = await pod_a.get_session_metadata(session_id)
    assert metadata is not None
    if cancel_outcome:
        assert metadata["status"] == "cancelled"
        assert metadata["can_cancel"] is False
        assert metadata["regeneration_claim"]["active"] is False
        assert metadata["regeneration_claim"]["cancel_requested"] is True
    else:
        assert takeover_won
        assert metadata["regeneration_claim"]["token"] == takeover_outcome.token


@pytest.mark.unit
@pytest.mark.asyncio
async def test_abort_claim_restores_snapshot_and_keeps_generation_fence(
    tmp_path: Path,
) -> None:
    pod_a, pod_b = _manager_pair(tmp_path)
    session_id = await _create_completed_session(pod_a)
    await pod_a.update_session(
        session_id,
        {"results": {"run": "original"}, "completed_at": "original-time"},
        sync_files=False,
    )
    planned = await pod_a.get_session_metadata_versioned(session_id)
    assert planned.value is not None
    assert planned.version is not None
    restore_metadata = dict(planned.value)
    claim = await pod_a.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "pending", "results": {"run": "new"}},
        remove_fields=("completed_at",),
        lease_seconds=60,
    )
    assert await pod_b.abort_regeneration_claim(
        session_id,
        claim,
        restore_metadata=restore_metadata,
    )

    restored = await pod_a.get_session_metadata(session_id)
    assert restored is not None
    assert restored["status"] == "completed"
    assert restored["results"] == {"run": "original"}
    assert restored["completed_at"] == "original-time"
    assert restored["regeneration_claim"]["generation"] == claim.generation
    assert restored["regeneration_claim"]["active"] is False
    assert not await pod_a.abort_regeneration_claim(
        session_id,
        claim,
        restore_metadata=restore_metadata,
    )

    next_plan = await pod_b.get_session_metadata_versioned(session_id)
    assert next_plan.version is not None
    next_claim = await pod_b.claim_regeneration(
        session_id,
        expected_version=next_plan.version,
        lease_seconds=60,
    )
    assert next_claim.generation == claim.generation + 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_report_finishing_last_cannot_replace_current_pointer(
    tmp_path: Path,
) -> None:
    pod_a, pod_b = _manager_pair(tmp_path)
    session_id = await _create_completed_session(pod_a)
    valid_predictions = {
        **initial_artifact_validity(),
        "raw_predictions": True,
    }
    await pod_a.update_session(
        session_id,
        {
            "prediction_lineage_token": "R1",
            "artifact_validity": valid_predictions,
        },
        sync_files=False,
    )
    stale_key = "reports/R1/build-stale/prediction_report.html"
    await pod_a.put_bytes_to_store(session_id, stale_key, b"R1")

    await pod_b.update_session(
        session_id,
        {
            "prediction_lineage_token": "R2",
            "artifact_validity": valid_predictions,
        },
        sync_files=False,
    )
    current_key = "reports/R2/build-current/prediction_report.html"
    await pod_b.put_bytes_to_store(session_id, current_key, b"R2")

    assert await pod_b.publish_prediction_report_if_lineage_matches(
        session_id,
        "R2",
        current_key,
    )
    assert not await pod_a.publish_prediction_report_if_lineage_matches(
        session_id,
        "R1",
        stale_key,
    )
    metadata = await pod_a.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["prediction_report_publication"] == {
        "prediction_lineage_token": "R2",
        "key": current_key,
        "published_at": metadata["prediction_report_publication"]["published_at"],
    }
    assert metadata["artifact_validity"]["prediction_report"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_legacy_prediction_lineage_creation_is_multi_manager_cas(
    tmp_path: Path,
) -> None:
    pod_a, pod_b = _manager_pair(tmp_path)
    session_id = await _create_completed_session(pod_a)
    await pod_a.update_session(
        session_id,
        {
            "artifact_validity": {
                **initial_artifact_validity(),
                "raw_predictions": True,
            }
        },
        remove_fields=("prediction_lineage_token",),
        sync_files=False,
    )

    lineages = await asyncio.gather(
        pod_a.capture_prediction_lineage(session_id),
        pod_b.capture_prediction_lineage(session_id),
    )
    assert lineages[0]
    assert lineages[0] == lineages[1]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_standard_terminal_and_metadata_helpers_retry_cas_contention(
    tmp_path: Path,
) -> None:
    store = _ConflictOnceStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "local", store=store)
    session_id = str(uuid4())
    await manager.create_session(session_id)

    store.fail_next_replace = True
    assert await manager.finalize_standard_pipeline(
        session_id,
        {"status": "completed", "results": {}, "coverage": None},
    )
    store.fail_next_replace = True
    assert await manager.mark_terminal_events_quiesced(
        session_id,
        expected_status="completed",
    )

    await manager.update_session(session_id, {"status": "running"})
    store.fail_next_replace = True
    await manager.update_step_progress(
        session_id,
        "predict",
        {"percent": 25},
    )
    store.fail_next_replace = True
    await manager.mark_step_completed(session_id, "predict", {"count": 1})
    store.fail_next_replace = True
    await manager.add_preview_image(session_id, "preview.png")
    store.fail_next_replace = True
    await manager.update_preview_images(session_id, ["replacement.png"])

    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["preview_images"] == ["replacement.png"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_standard_terminal_quiescence_guard_edges(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    missing = str(uuid4())
    assert not await manager.finalize_standard_pipeline(
        missing,
        {"status": "failed", "error": "gone"},
    )
    assert not await manager.mark_terminal_events_quiesced(
        missing,
        expected_status="failed",
    )

    session_id = str(uuid4())
    await manager.create_session(session_id)
    assert not await manager.mark_terminal_events_quiesced(
        session_id,
        expected_status="completed",
    )
    assert not await manager.mark_terminal_events_quiesced(
        session_id,
        expected_status="pending",
    )
    await _replace_metadata_fields_unchecked(
        manager,
        session_id,
        {"terminal_events_quiesced": True},
    )
    assert await manager.mark_terminal_events_quiesced(
        session_id,
        expected_status="pending",
    )


@pytest.mark.unit
def test_publication_resolvers_fail_closed_after_pointer_contract_exists() -> None:
    assert (
        SessionManager.resolve_published_artifact_key(
            {},
            "output_usd",
            legacy_key="output/scene.usd",
        )
        == "output/scene.usd"
    )
    publication = {
        "prefix": "runs/2-token",
        "artifacts": {"output_usd": "runs/2-token/output/scene.usd"},
    }
    assert (
        SessionManager.resolve_published_artifact_key(
            {"published_artifacts": "invalid"},
            "output_usd",
        )
        is None
    )
    assert (
        SessionManager.resolve_published_artifact_key(
            {"published_artifacts": {"prefix": 2, "artifacts": []}},
            "output_usd",
        )
        is None
    )
    assert (
        SessionManager.resolve_published_artifact_key(
            {"published_artifacts": publication},
            "output_usd",
            legacy_key="output/stale.usd",
        )
        == "runs/2-token/output/scene.usd"
    )
    assert (
        SessionManager.resolve_published_artifact_key(
            {"published_artifacts": publication},
            "missing",
            legacy_key="output/stale.usd",
        )
        is None
    )
    assert (
        SessionManager.resolve_published_artifact_key(
            {
                "published_artifacts": {
                    "prefix": "runs/2-token",
                    "artifacts": {"output_usd": "output/stale.usd"},
                }
            },
            "output_usd",
        )
        is None
    )

    assert SessionManager.resolve_prediction_report_key({}) == (
        "cache/predictions/prediction_report.html"
    )
    assert (
        SessionManager.resolve_prediction_report_key(
            {"prediction_report_publication": "invalid"}
        )
        is None
    )
    assert (
        SessionManager.resolve_prediction_report_key(
            {
                "prediction_lineage_token": "R2",
                "prediction_report_publication": {
                    "prediction_lineage_token": "R2",
                    "key": "reports/R2/build/report.html",
                },
            }
        )
        == "reports/R2/build/report.html"
    )
    assert (
        SessionManager.resolve_prediction_report_key(
            {
                "prediction_lineage_token": "R2",
                "prediction_report_publication": {
                    "prediction_lineage_token": "R1",
                    "key": "reports/R1/build/report.html",
                },
            }
        )
        is None
    )
    assert (
        SessionManager.resolve_prediction_report_key(
            {
                "prediction_lineage_token": "R2",
                "prediction_report_publication": {
                    "prediction_lineage_token": "R2",
                    "key": "",
                },
            }
        )
        is None
    )
