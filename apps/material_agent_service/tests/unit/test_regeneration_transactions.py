# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused regeneration reservation, rollback, and dependency tests."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi import HTTPException

from ...service.artifact_lineage import initial_artifact_validity
from ...service.models.requests import PipelineStep, RegenerateRequest
from ...service.routers import pipeline_router
from ...service.runtime.bus import EventBus
from ...service.runtime.events import ProgressEvent, StepState
from ...service.runtime.registry import JobRegistry
from ...service.session.manager import CANCEL_KEY, RegenerationClaim, SessionManager
from ...service.storage.base import METADATA_KEY, JsonPreconditionError
from ...service.storage.local_store import LocalSessionStore
from ...service.workers import executor


class _RemoteKindStore(LocalSessionStore):
    """Local test implementation with remote-store trust semantics."""

    @property
    def kind(self) -> str:
        return "remote-test"


async def _expire_regeneration_claim(
    manager: SessionManager,
    session_id: str,
) -> None:
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


async def _ready_apply_session(
    manager: SessionManager,
) -> tuple[str, Path, Path]:
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    (session_dir / "input" / "scene.usda").write_text(
        "#usda 1.0\n",
        encoding="utf-8",
    )
    raw_path = session_dir / "cache" / "predictions" / "predictions.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text('{"id": "/Root", "material": "Steel"}\n')
    state_path = session_dir / "cache" / ".pipeline_state.json"
    state_path.write_text(
        json.dumps(
            {
                "completed_steps": ["predict", "apply"],
                "failed_steps": [],
                "step_errors": {},
                "step_outputs": {
                    "predict": {"predictions_path": str(raw_path)},
                    "apply": {"output_usd_path": "old-output.usd"},
                },
                "current_step": None,
            }
        ),
        encoding="utf-8",
    )
    await manager.update_session(
        session_id,
        {
            "status": "completed",
            "results": {"old": True},
            "coverage": None,
            "completed_at": "2026-01-01T00:00:00+00:00",
            "artifact_validity": {
                **initial_artifact_validity(),
                "raw_predictions": True,
            },
        },
    )
    return session_id, session_dir, state_path


def _install_regeneration_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    manager: SessionManager,
    event_bus: EventBus,
    registry: object,
) -> None:
    import material_agent.api as material_api

    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: manager)
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: event_bus)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    monkeypatch.setattr(
        material_api,
        "build_unified_pipeline_config",
        lambda **kwargs: {
            "input": {"usd_path": kwargs["input_usd_path"]},
            "output": {"usd_path": kwargs["output_usd_path"]},
            "steps": {step: {} for step in kwargs["enabled_steps"]},
        },
    )


@pytest.mark.asyncio
async def test_registration_failure_rolls_back_every_preparation_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id, _session_dir, state_path = await _ready_apply_session(manager)
    await manager.store.put_bytes(session_id, CANCEL_KEY, b"")
    metadata_before = await manager.get_session_metadata(session_id)
    state_before = state_path.read_bytes()

    bus = EventBus()
    bus._state[session_id] = {"status": "completed", "marker": ["old"]}
    queue = bus.get_queue(session_id)
    old_event = ProgressEvent(
        session_id=session_id,
        step="pipeline",
        state=StepState.COMPLETED,
    )
    await queue.put(old_event)
    event_state_before = dict(bus._state[session_id])

    class _FailAfterPreparation:
        async def register(
            self,
            registered_session_id: str,
            coro: Any,
            *,
            before_start,
        ) -> None:
            assert registered_session_id == session_id
            await before_start()
            coro.close()
            raise RuntimeError("scheduler unavailable")

    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        bus,
        _FailAfterPreparation(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(
            session_id,
            RegenerateRequest(steps=[PipelineStep.APPLY]),
        )
    assert exc_info.value.status_code == 503
    metadata_after = await manager.get_session_metadata(session_id)
    aborted_claim = metadata_after.pop("regeneration_claim")
    metadata_after.pop("updated_at")
    metadata_before.pop("updated_at")
    assert metadata_after == metadata_before
    assert aborted_claim["active"] is False
    assert aborted_claim["aborted_at"]
    assert state_path.read_bytes() == state_before
    assert await manager.is_cancelled(session_id)
    assert bus.get_snapshot(session_id) == event_state_before
    assert bus.get_queue(session_id) is queue
    assert await queue.get() is old_event


@pytest.mark.asyncio
async def test_preparation_heartbeat_store_error_cancels_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RenewalFailureManager(SessionManager):
        async def renew_regeneration_claim(
            self,
            session_id: str,
            claim: RegenerationClaim,
            *,
            lease_seconds: float = 300.0,
        ) -> bool:
            del session_id, claim, lease_seconds
            raise RuntimeError("claim store unavailable")

    manager = _RenewalFailureManager(tmp_path)
    session_id, _session_dir, state_path = await _ready_apply_session(manager)
    state_before = state_path.read_bytes()

    class _PrepareOnlyRegistry:
        async def register(
            self,
            registered_session_id: str,
            coro: Any,
            *,
            before_start,
        ) -> None:
            assert registered_session_id == session_id
            try:
                await before_start()
            finally:
                coro.close()

    async def block_hydration(*_args: Any, **_kwargs: Any) -> None:
        await asyncio.sleep(60)

    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        EventBus(),
        _PrepareOnlyRegistry(),
    )
    monkeypatch.setattr(
        pipeline_router,
        "_REGENERATION_PREPARATION_HEARTBEAT_SECONDS",
        0.0,
    )
    monkeypatch.setattr(
        pipeline_router,
        "_hydrate_regeneration_inputs",
        block_hydration,
    )

    with pytest.raises(asyncio.CancelledError):
        await pipeline_router.regenerate_pipeline(
            session_id,
            RegenerateRequest(steps=[PipelineStep.APPLY]),
        )

    metadata = await manager.get_session_metadata(session_id)
    assert metadata["status"] == "completed"
    assert metadata["results"] == {"old": True}
    assert metadata["regeneration_claim"]["active"] is False
    assert metadata["regeneration_claim"]["aborted_at"]
    assert state_path.read_bytes() == state_before


@pytest.mark.asyncio
async def test_successful_preparation_clears_prior_cancel_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id, _session_dir, _state_path = await _ready_apply_session(manager)
    await manager.store.put_bytes(session_id, CANCEL_KEY, b"")
    bus = EventBus()

    class _AcceptAfterPreparation:
        async def register(
            self,
            registered_session_id: str,
            coro: Any,
            *,
            before_start,
        ) -> None:
            assert registered_session_id == session_id
            await before_start()
            coro.close()

    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        bus,
        _AcceptAfterPreparation(),
    )

    response = await pipeline_router.regenerate_pipeline(
        session_id,
        RegenerateRequest(steps=[PipelineStep.APPLY]),
    )
    assert response.status == "pending"
    assert not await manager.is_cancelled(session_id)
    metadata = await manager.get_session_metadata(session_id)
    assert metadata["status"] == "pending"


@pytest.mark.asyncio
async def test_cancel_semaphore_waiting_regeneration_preserves_prior_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id, _session_dir, _state_path = await _ready_apply_session(manager)
    metadata_before = await manager.get_session_metadata(session_id)
    registry = JobRegistry(max_concurrent=1)
    blocker_release = asyncio.Event()

    async def blocker() -> None:
        await blocker_release.wait()

    await registry.register("blocking-session", blocker())
    worker_started = asyncio.Event()

    async def unexpected_worker(**_kwargs: Any) -> None:
        worker_started.set()

    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        EventBus(),
        registry,
    )
    monkeypatch.setattr(
        pipeline_router,
        "execute_pipeline_async",
        unexpected_worker,
    )
    regeneration_task = asyncio.create_task(
        pipeline_router.regenerate_pipeline(
            session_id,
            RegenerateRequest(steps=[PipelineStep.APPLY]),
        )
    )
    try:
        for _ in range(200):
            if registry.is_reserved(session_id):
                break
            await asyncio.sleep(0.005)
        assert registry.is_reserved(session_id)

        response = await pipeline_router.cancel_pipeline(session_id)

        assert response["status"] == "completed"
        assert response["message"] == "Queued regeneration cancelled before start"
        with pytest.raises(asyncio.CancelledError):
            await regeneration_task
        assert not worker_started.is_set()
        metadata_after = await manager.get_session_metadata(session_id)
        assert metadata_after == metadata_before
        assert not await manager.is_cancelled(session_id)
    finally:
        if not regeneration_task.done():
            regeneration_task.cancel()
        blocker_release.set()
        blocking_task = registry.get_task("blocking-session")
        if blocking_task is not None:
            await blocking_task


@pytest.mark.asyncio
async def test_stale_plan_is_rejected_before_destructive_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id, _session_dir, state_path = await _ready_apply_session(manager)
    state_before = state_path.read_bytes()
    bus = EventBus()

    class _AdvanceThenPrepare:
        async def register(
            self,
            registered_session_id: str,
            coro: Any,
            *,
            before_start,
        ) -> None:
            await manager.update_session(session_id, {"results": {"new": True}})
            try:
                await before_start()
            finally:
                coro.close()

    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        bus,
        _AdvanceThenPrepare(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(
            session_id,
            RegenerateRequest(steps=[PipelineStep.APPLY]),
        )
    assert exc_info.value.status_code == 409
    assert state_path.read_bytes() == state_before
    metadata = await manager.get_session_metadata(session_id)
    assert metadata["status"] == "completed"
    assert metadata["results"] == {"new": True}
    assert bus.get_snapshot(session_id) is None


@pytest.mark.asyncio
async def test_regeneration_rejects_provisional_standard_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id, _session_dir, _state_path = await _ready_apply_session(manager)
    # This is the status-only write a standard worker's EventBus may persist
    # before the executor stores authoritative failure diagnostics.
    await manager.update_session(
        session_id,
        {
            "status": "failed",
            "results": None,
        },
        remove_fields=("failed_at", "failed_step", "error"),
    )
    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        EventBus(),
        JobRegistry(max_concurrent=1),
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(
            session_id,
            RegenerateRequest(steps=[PipelineStep.APPLY]),
        )

    assert exc_info.value.status_code == 409
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert "regeneration_claim" not in metadata

    # The old standard worker remains free to finish its own terminal write;
    # there is no newly claimed generation for it to corrupt.
    await manager.update_session(
        session_id,
        {
            "status": "failed",
            "failed_at": datetime.now(UTC).isoformat(),
            "failed_step": "predict",
            "error": "authoritative old-worker failure",
        },
    )
    finalized = await manager.get_session_metadata(session_id)
    assert finalized is not None
    assert finalized["error"] == "authoritative old-worker failure"
    assert "regeneration_claim" not in finalized


@pytest.mark.asyncio
async def test_regeneration_rejects_ready_upload_session_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "ready"})
    metadata_before = await manager.get_session_metadata(session_id)
    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        EventBus(),
        JobRegistry(max_concurrent=1),
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(
            session_id,
            RegenerateRequest(steps=[PipelineStep.APPLY]),
        )
    assert exc_info.value.status_code == 400
    assert "ready" in str(exc_info.value.detail)
    assert await manager.get_session_metadata(session_id) == metadata_before


@pytest.mark.asyncio
async def test_regenerate_route_takes_over_an_expired_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id, _session_dir, _state_path = await _ready_apply_session(manager)
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    expired_claim = await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "running"},
        lease_seconds=60,
    )
    await _expire_regeneration_claim(manager, session_id)

    class _AcceptAfterPreparation:
        async def register(
            self,
            registered_session_id: str,
            coro: Any,
            *,
            before_start,
        ) -> None:
            assert registered_session_id == session_id
            await before_start()
            coro.close()

    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        EventBus(),
        _AcceptAfterPreparation(),
    )

    response = await pipeline_router.regenerate_pipeline(
        session_id,
        RegenerateRequest(steps=[PipelineStep.APPLY]),
    )

    assert response.status == "pending"
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["regeneration_claim"]["active"] is True
    assert metadata["regeneration_claim"]["generation"] == (
        expired_claim.generation + 1
    )
    assert metadata["regeneration_claim"]["token"] != expired_claim.token


@pytest.mark.asyncio
async def test_cancel_route_recovers_an_expired_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id, _session_dir, _state_path = await _ready_apply_session(manager)
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "running"},
        lease_seconds=60,
    )
    await _expire_regeneration_claim(manager, session_id)

    class _IdleRegistry:
        def is_running(self, _session_id: str) -> bool:
            return False

        def is_reserved(self, _session_id: str) -> bool:
            return False

    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: manager)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _IdleRegistry())

    response = await pipeline_router.cancel_pipeline(session_id)

    assert response["status"] == "cancelled"
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "cancelled"
    assert metadata["can_cancel"] is False
    assert metadata["regeneration_claim"]["active"] is False
    assert metadata["regeneration_claim"]["cancel_requested"] is True


@pytest.mark.asyncio
async def test_no_claim_cancel_cas_retries_against_remote_claim_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "running"})

    class _RemoteOnlyRegistry:
        def is_running(self, _session_id: str) -> bool:
            return False

        def is_reserved(self, _session_id: str) -> bool:
            return False

    original_replace = manager.store.replace_json_if_version
    winner_token = str(uuid4())
    race_injected = False

    async def inject_claim_before_cancel_cas(
        requested_session_id: str,
        key: str,
        obj: dict,
        expected_version: str | None,
    ) -> str:
        nonlocal race_injected
        if (
            not race_injected
            and key == METADATA_KEY
            and obj.get("status") == "cancelling"
        ):
            current = await manager.store.get_json_versioned(
                requested_session_id,
                key,
            )
            assert current.value is not None
            assert current.version == expected_version
            winner = dict(current.value)
            now = datetime.now(UTC)
            winner["status"] = "running"
            winner["regeneration_claim"] = {
                "generation": 1,
                "token": winner_token,
                "active": True,
                "claimed_at": now.isoformat(),
                "renewed_at": now.isoformat(),
                "lease_expires_at": (now + timedelta(seconds=60)).isoformat(),
                "cancel_requested": False,
            }
            await original_replace(
                requested_session_id,
                key,
                winner,
                expected_version,
            )
            race_injected = True
            raise JsonPreconditionError("remote claim won cancellation CAS")
        return await original_replace(
            requested_session_id,
            key,
            obj,
            expected_version,
        )

    monkeypatch.setattr(
        manager.store,
        "replace_json_if_version",
        inject_claim_before_cancel_cas,
    )
    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: manager)
    monkeypatch.setattr(
        pipeline_router,
        "get_job_registry",
        lambda: _RemoteOnlyRegistry(),
    )

    response = await pipeline_router.cancel_pipeline(session_id)

    assert response["status"] == "cancelling"
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["regeneration_claim"]["token"] == winner_token
    assert metadata["regeneration_claim"]["active"] is True
    assert metadata["regeneration_claim"]["cancel_requested"] is True
    assert metadata["status"] == "cancelling"
    assert not await manager.is_cancelled(session_id)


@pytest.mark.asyncio
async def test_cancel_route_returns_success_when_worker_finalizes_marker_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "running"})

    class _RemoteOnlyRegistry:
        def is_running(self, _session_id: str) -> bool:
            return False

        def is_reserved(self, _session_id: str) -> bool:
            return False

    original_replace = manager.store.replace_json_if_version
    race_injected = False

    async def finalize_before_cancel_cas(
        requested_session_id: str,
        key: str,
        obj: dict,
        expected_version: str | None,
    ) -> str:
        nonlocal race_injected
        if (
            not race_injected
            and key == METADATA_KEY
            and obj.get("status") == "cancelling"
        ):
            current = await manager.store.get_json_versioned(
                requested_session_id,
                key,
            )
            assert current.value is not None
            assert current.version == expected_version
            terminal = dict(current.value)
            terminal.update(
                {
                    "status": "cancelled",
                    "cancelled_at": datetime.now(UTC).isoformat(),
                    "can_cancel": False,
                }
            )
            await original_replace(
                requested_session_id,
                key,
                terminal,
                expected_version,
            )
            race_injected = True
            raise JsonPreconditionError("worker finalized cancellation first")
        return await original_replace(
            requested_session_id,
            key,
            obj,
            expected_version,
        )

    monkeypatch.setattr(
        manager.store,
        "replace_json_if_version",
        finalize_before_cancel_cas,
    )
    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: manager)
    monkeypatch.setattr(
        pipeline_router,
        "get_job_registry",
        lambda: _RemoteOnlyRegistry(),
    )

    response = await pipeline_router.cancel_pipeline(session_id)

    assert response["status"] == "cancelled"
    assert response["message"] == "Pipeline cancellation completed"
    assert not await manager.is_cancelled(session_id)


def test_dependency_closure_requires_real_unplanned_upstream_evidence() -> None:
    invalid = {"artifact_validity": initial_artifact_validity()}
    cases = [
        (["build_dataset_prepare_dataset"], {"build_dataset_prepare_dataset"}),
        (["cluster_prims"], {"cluster_prims"}),
        (["predict"], {"predict"}),
    ]
    for steps, invalidated in cases:
        with pytest.raises(HTTPException, match="no current cached evidence"):
            pipeline_router._validate_regeneration_dependency_closure(
                steps,
                invalidated,
                optimize_usd_enabled=False,
                metadata=invalid,
            )

    with pytest.raises(HTTPException, match="invalidated by an earlier"):
        pipeline_router._validate_regeneration_dependency_closure(
            ["build_dataset_usd", "apply"],
            set(pipeline_router.STEP_ORDER),
            optimize_usd_enabled=False,
            metadata=invalid,
        )
    with pytest.raises(HTTPException, match="invalidated by an earlier"):
        pipeline_router._validate_regeneration_dependency_closure(
            ["predict", "render"],
            set(pipeline_router.STEP_ORDER[10:]),
            optimize_usd_enabled=False,
            metadata={
                **invalid,
                "_regeneration_step_evidence": {
                    "build_dataset_prepare_dataset",
                },
            },
        )

    reusable = {
        "artifact_validity": {
            **initial_artifact_validity(),
            "raw_predictions": True,
            "applied_output_usd": True,
        },
        "_regeneration_step_evidence": {"predict", "apply"},
    }
    pipeline_router._validate_regeneration_dependency_closure(
        ["apply"],
        {"apply", "render"},
        optimize_usd_enabled=False,
        metadata=reusable,
    )
    pipeline_router._validate_regeneration_dependency_closure(
        ["render"],
        {"render"},
        optimize_usd_enabled=False,
        metadata=reusable,
    )


@pytest.mark.asyncio
async def test_large_scene_guard_is_read_only_for_both_metadata_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for updates in (
        {"pipeline_type": "large_scene"},
        {"config": {"large_scene": True}},
    ):
        manager = SessionManager(tmp_path / str(uuid4()))
        session_id = str(uuid4())
        session_dir = await manager.create_session(session_id)
        state_path = session_dir / "cache" / ".pipeline_state.json"
        state_path.write_text('{"marker": "unchanged"}', encoding="utf-8")
        await manager.update_session(session_id, {"status": "completed", **updates})
        metadata_before = await manager.get_session_metadata(session_id)
        bus = EventBus()
        monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: manager)
        monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: bus)

        with pytest.raises(HTTPException) as exc_info:
            await pipeline_router.regenerate_pipeline(
                session_id,
                RegenerateRequest(steps=[PipelineStep.APPLY]),
            )
        assert exc_info.value.status_code == 400
        assert await manager.get_session_metadata(session_id) == metadata_before
        assert state_path.read_text(encoding="utf-8") == '{"marker": "unchanged"}'
        assert bus.get_snapshot(session_id) is None


@pytest.mark.asyncio
async def test_remote_checkpoint_and_inputs_overwrite_stale_or_missing_local_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod-local", store=shared_store)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    (session_dir / "input" / "scene.usda").write_text(
        "#usda 1.0\n",
        encoding="utf-8",
    )
    remote_raw = b'{"id": "/Remote", "material": "Steel"}\n'
    await shared_store.put_bytes(
        session_id,
        "cache/predictions/predictions.jsonl",
        remote_raw,
    )
    remote_state = {
        "completed_steps": ["predict", "apply"],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {
            "predict": {"predictions_path": "remote-raw.jsonl"},
            "apply": {"output_usd_path": "old.usd"},
        },
    }
    await shared_store.put_bytes(
        session_id,
        "cache/.pipeline_state.json",
        json.dumps(remote_state).encode(),
    )
    await manager.update_session(
        session_id,
        {
            "status": "completed",
            "results": {},
            "coverage": None,
            "completed_at": datetime.now(UTC).isoformat(),
            "artifact_validity": {
                **initial_artifact_validity(),
                "raw_predictions": True,
            },
        },
        sync_files=False,
    )
    local_raw = session_dir / "cache" / "predictions" / "predictions.jsonl"
    local_raw.parent.mkdir(parents=True, exist_ok=True)
    local_raw.write_bytes(b"stale-local")
    local_state = session_dir / "cache" / ".pipeline_state.json"
    assert not local_state.exists()

    class _AcceptAfterPreparation:
        async def register(
            self,
            registered_session_id: str,
            coro: Any,
            *,
            before_start,
        ) -> None:
            await before_start()
            coro.close()

    bus = EventBus()
    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        bus,
        _AcceptAfterPreparation(),
    )
    response = await pipeline_router.regenerate_pipeline(
        session_id,
        RegenerateRequest(steps=[PipelineStep.APPLY]),
    )

    assert response.status == "pending"
    assert local_raw.read_bytes() == remote_raw
    local_checkpoint = json.loads(local_state.read_text(encoding="utf-8"))
    assert local_checkpoint["completed_steps"] == ["predict"]
    assert set(local_checkpoint["step_outputs"]) == {"predict"}


@pytest.mark.asyncio
async def test_remote_store_never_uses_stale_local_regeneration_evidence(
    tmp_path: Path,
) -> None:
    store = _RemoteKindStore(str(tmp_path / "remote"))
    manager = SessionManager(tmp_path / "pod-local", store=store)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    stale_scene = session_dir / "input" / "scene.usda"
    stale_scene.parent.mkdir(parents=True, exist_ok=True)
    stale_scene.write_text("#usda 1.0\n# stale local only\n", encoding="utf-8")
    stale_raw = session_dir / "cache" / "predictions" / "predictions.jsonl"
    stale_raw.parent.mkdir(parents=True, exist_ok=True)
    stale_raw.write_text('{"id": "/Stale"}\n', encoding="utf-8")
    stale_preview = session_dir / "cache" / "preview" / "same.png"
    stale_preview.parent.mkdir(parents=True, exist_ok=True)
    stale_preview.write_bytes(b"stale-preview")
    stale_checkpoint = session_dir / "cache" / ".pipeline_state.json"
    stale_checkpoint.write_text(
        json.dumps(
            {
                "completed_steps": ["build_dataset_usd", "predict"],
                "step_outputs": {
                    "build_dataset_usd": {"num_images": 1, "output_dir": "stale"},
                    "predict": {"predictions_path": str(stale_raw)},
                },
            }
        ),
        encoding="utf-8",
    )
    await manager.update_session(
        session_id,
        {
            "status": "completed",
            "preview_images": ["same.png"],
            "artifact_validity": {
                **initial_artifact_validity(),
                "raw_predictions": True,
                "previews": True,
            },
        },
        sync_files=False,
    )

    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert (
        await pipeline_router._read_regeneration_checkpoint(
            manager,
            session_id,
            session_dir,
        )
        == {}
    )
    validity = await pipeline_router._derive_regeneration_artifact_validity(
        manager,
        session_id,
        session_dir,
        metadata,
    )
    assert not validity["raw_predictions"]
    assert not validity["previews"]
    assert not await pipeline_router._derive_regeneration_step_evidence(
        manager,
        session_id,
        session_dir,
        validity,
    )
    with pytest.raises(HTTPException, match="Input USD not found"):
        await pipeline_router._plan_regeneration_input_bundle(
            manager,
            session_id,
            session_dir,
            metadata,
        )

    await pipeline_router._hydrate_regeneration_inputs(
        manager,
        session_id,
        session_dir,
        ["apply"],
        optimize_usd_enabled=False,
    )
    assert not stale_raw.exists()


@pytest.mark.asyncio
async def test_apply_only_regeneration_carries_upstream_artifacts_immutably(
    tmp_path: Path,
) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    pod_a = SessionManager(tmp_path / "pod-a", store=shared_store)
    pod_b = SessionManager(tmp_path / "pod-b", store=shared_store)
    session_id = str(uuid4())
    await pod_a.create_session(session_id)
    canonical_artifacts = {
        "cache/dataset/dataset.jsonl": b'{"id": "/Remote"}\n',
        "cache/dataset/usd/prims.jsonl": b'{"id": "/Remote"}\n',
        "cache/predictions/predictions.jsonl": (
            b'{"id": "/Remote", "material": "Steel"}\n'
        ),
    }
    for key, data in canonical_artifacts.items():
        await shared_store.put_bytes(session_id, key, data)
    await shared_store.put_bytes(
        session_id,
        "cache/.pipeline_state.json",
        b'{"completed_steps": ["predict"], "stale": true}',
    )
    await pod_a.update_session(
        session_id,
        {
            "status": "completed",
            "artifact_validity": {
                **initial_artifact_validity(),
                "raw_predictions": True,
            },
        },
        sync_files=False,
    )
    planned = await pod_a.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    claim = await pod_a.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "running"},
    )
    current = await pod_b.get_session_metadata(session_id)
    assert current is not None
    artifact_map: dict[str, str] = {}
    carried_validity = await executor._carry_forward_regeneration_artifacts(
        pod_b,
        session_id,
        claim,
        current,
        dict(current["artifact_validity"]),
        artifact_map,
        {"apply"},
    )
    assert carried_validity["raw_predictions"] is True
    assert set(canonical_artifacts) <= set(artifact_map)
    assert "cache/.pipeline_state.json" not in artifact_map
    assert await pod_b.finalize_regeneration_claim(
        session_id,
        claim,
        updates={
            "status": "completed",
            "artifact_validity": carried_validity,
        },
        artifact_map=artifact_map,
    )

    for key in canonical_artifacts:
        await shared_store.delete_file(session_id, key)
    metadata = await pod_a.get_session_metadata(session_id)
    assert metadata is not None
    assert (
        pod_a.resolve_published_artifact_key(
            metadata,
            "cache/.pipeline_state.json",
            legacy_key="cache/.pipeline_state.json",
        )
        is None
    )
    for logical_key, expected in canonical_artifacts.items():
        immutable_key = pod_a.resolve_published_artifact_key(
            metadata,
            logical_key,
            legacy_key=logical_key,
        )
        assert immutable_key == artifact_map[logical_key]
        assert await pod_a.read_from_store(session_id, immutable_key) == expected

    pod_c = SessionManager(tmp_path / "pod-c", store=shared_store)
    await pipeline_router._hydrate_regeneration_inputs(
        pod_c,
        session_id,
        pod_c.get_session_dir(session_id),
        ["apply"],
        optimize_usd_enabled=False,
    )
    assert (
        pod_c.get_session_dir(session_id)
        / "cache"
        / "predictions"
        / "predictions.jsonl"
    ).read_bytes() == canonical_artifacts["cache/predictions/predictions.jsonl"]


@pytest.mark.asyncio
async def test_apply_only_regeneration_preserves_lineage_report_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id, _session_dir, _state_path = await _ready_apply_session(manager)
    lineage = str(uuid4())
    await manager.update_session(
        session_id,
        {
            "prediction_lineage_token": lineage,
            "artifact_validity": {
                **initial_artifact_validity(),
                "raw_predictions": True,
            },
        },
    )
    report_key = f"reports/{lineage}/{uuid4()}/prediction_report.html"
    await manager.put_bytes_to_store(session_id, report_key, b"<html>current</html>")
    assert await manager.publish_prediction_report_if_lineage_matches(
        session_id,
        lineage,
        report_key,
    )

    class _AcceptAfterPreparation:
        async def register(
            self,
            registered_session_id: str,
            coro: Any,
            *,
            before_start,
        ) -> None:
            assert registered_session_id == session_id
            await before_start()
            coro.close()

    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        EventBus(),
        _AcceptAfterPreparation(),
    )
    response = await pipeline_router.regenerate_pipeline(
        session_id,
        RegenerateRequest(steps=[PipelineStep.APPLY]),
    )

    assert response.status == "pending"
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["artifact_validity"]["prediction_report"] is True
    assert manager.resolve_prediction_report_key(metadata) == report_key


@pytest.mark.asyncio
async def test_cross_instance_regeneration_reconstructs_and_hydrates_full_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import material_agent.api as material_api

    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod-b", store=shared_store)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)

    remote_objects = {
        "input/scene.usda": b"#usda 1.0\n# authoritative remote scene\n",
        "input/reference_images/reference_0000.png": b"remote-image",
        "input/reference_images/descriptions.json": b'["remote finish"]',
        "input/reference_pdfs/reference_0000.pdf": b"%PDF-remote",
        "materials/materials.yaml": (
            b"library_path: custom.usda\n"
            b"entries:\n  - name: RemoteSteel\n    binding: /Looks/RemoteSteel\n"
        ),
        "materials/custom.usda": b"#usda 1.0\n# remote custom library\n",
        "cache/generated_material_library/materials.yaml": (
            b"library_path: material_library.usda\n"
            b"entries:\n  - name: GeneratedRemote\n"
            b"    binding: /Looks/GeneratedRemote\n"
        ),
        "cache/generated_material_library/material_library.usda": (
            b"#usda 1.0\n# generated remote library\n"
        ),
        "cache/predictions/predictions.jsonl": (
            b'{"id": "/Root", "material": "RemoteSteel"}\n'
        ),
    }
    for key, data in remote_objects.items():
        await shared_store.put_bytes(session_id, key, data)
    await shared_store.put_bytes(
        session_id,
        "cache/.pipeline_state.json",
        json.dumps(
            {
                "completed_steps": ["predict"],
                "failed_steps": [],
                "step_errors": {},
                "step_outputs": {
                    "predict": {"predictions_path": "remote-predictions.jsonl"}
                },
            }
        ).encode(),
    )
    await manager.update_session(
        session_id,
        {
            "status": "completed",
            "results": {},
            "coverage": None,
            "completed_at": datetime.now(UTC).isoformat(),
            "config": {
                "enable_material_generation": True,
                "optimize_usd": False,
            },
            "artifact_validity": {
                **initial_artifact_validity(),
                "raw_predictions": True,
            },
        },
        sync_files=False,
    )
    assert not (session_dir / "input" / "scene.usda").exists()

    captured_build: dict[str, Any] = {}
    captured_config: dict[str, Any] = {}

    def build_config(**kwargs: Any) -> dict[str, Any]:
        captured_build.update(kwargs)
        config = {
            "input": {"usd_path": kwargs["input_usd_path"]},
            "output": {"usd_path": kwargs["output_usd_path"]},
            "steps": {step: {} for step in kwargs["enabled_steps"]},
        }
        captured_config.update(config)
        return config

    class _AcceptAfterPreparation:
        async def register(
            self,
            registered_session_id: str,
            coro: Any,
            *,
            before_start,
        ) -> None:
            assert registered_session_id == session_id
            await before_start()
            coro.close()

    bus = EventBus()
    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        bus,
        _AcceptAfterPreparation(),
    )
    monkeypatch.setattr(material_api, "build_unified_pipeline_config", build_config)

    response = await pipeline_router.regenerate_pipeline(
        session_id,
        RegenerateRequest(steps=[PipelineStep.APPLY]),
    )

    assert response.status == "pending"
    assert Path(captured_build["input_usd_path"]) == (
        session_dir / "input" / "scene.usda"
    )
    assert captured_build["materials_library_path"] == str(
        (session_dir / "materials" / "custom.usda").resolve()
    )
    assert captured_build["materials_entries"][0]["name"] == "RemoteSteel"
    assert captured_config["input"]["reference_images"] == [
        str(session_dir / "input" / "reference_images" / "reference_0000.png")
    ]
    assert captured_config["input"]["reference_pdfs"] == [
        str(session_dir / "input" / "reference_pdfs" / "reference_0000.pdf")
    ]
    for key, expected in remote_objects.items():
        assert (session_dir / key).read_bytes() == expected
    checkpoint = json.loads(
        (session_dir / "cache" / ".pipeline_state.json").read_text(encoding="utf-8")
    )
    generated = checkpoint["step_outputs"]["generate_material_library"]
    assert generated["generated_material_library_path"] == str(
        (
            session_dir
            / "cache"
            / "generated_material_library"
            / "material_library.usda"
        ).resolve()
    )


@pytest.mark.asyncio
async def test_preview_evidence_uses_current_and_legacy_directories(
    tmp_path: Path,
) -> None:
    for preview_dir_name in ("cache/preview", "preview"):
        manager = SessionManager(tmp_path / preview_dir_name.replace("/", "-"))
        session_id = str(uuid4())
        session_dir = await manager.create_session(session_id)
        preview_path = session_dir / preview_dir_name / "current.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_bytes(b"png")
        state_path = session_dir / "cache" / ".pipeline_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "completed_steps": ["build_dataset_usd"],
                    "step_outputs": {"build_dataset_usd": {"num_images": 1}},
                }
            ),
            encoding="utf-8",
        )
        metadata = await manager.get_session_metadata(session_id)
        metadata["preview_images"] = ["current.png"]
        validity = await pipeline_router._derive_regeneration_artifact_validity(
            manager,
            session_id,
            session_dir,
            metadata,
        )
        assert validity["previews"]


@pytest.mark.asyncio
async def test_event_bus_stale_claim_cannot_persist_terminal_status(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "completed"})
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    claim = await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "running"},
    )
    bus = EventBus()
    bus.set_session_manager(manager)
    await bus.seed_pending_session(session_id, regeneration_claim=claim)
    assert await manager.finalize_regeneration_claim(
        session_id,
        claim,
        updates={"status": "completed", "results": {"winner": True}},
    )

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="predict",
            state=StepState.FAILED,
            message="late stale failure",
        )
    )

    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "completed"
    assert metadata["results"] == {"winner": True}


@pytest.mark.asyncio
async def test_event_bus_drops_local_progress_after_claim_takeover(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "completed"})
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    stale_claim = await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "running"},
        lease_seconds=60,
    )
    bus = EventBus()
    bus.set_session_manager(manager)
    await bus.seed_pending_session(session_id, regeneration_claim=stale_claim)
    snapshot_before = json.loads(json.dumps(bus.get_snapshot(session_id)))

    await _expire_regeneration_claim(manager, session_id)
    takeover_plan = await manager.get_session_metadata_versioned(session_id)
    assert takeover_plan.version is not None
    active_claim = await manager.claim_regeneration(
        session_id,
        expected_version=takeover_plan.version,
        updates={"status": "running"},
        lease_seconds=60,
    )
    assert active_claim.token != stale_claim.token

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="build_dataset_usd",
            state=StepState.RUNNING,
            percent=75,
            extra={"rendered_images": ["stale.png"]},
        )
    )

    assert bus.get_snapshot(session_id) == snapshot_before
    assert bus.get_queue(session_id).empty()
    assert await bus.get_fenced_snapshot(session_id) is None


@pytest.mark.asyncio
async def test_event_bus_drops_nonterminal_progress_after_expired_claim_cancel(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "completed"})
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    stale_claim = await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "running"},
        lease_seconds=60,
    )
    bus = EventBus()
    bus.set_session_manager(manager)
    await bus.seed_pending_session(session_id, regeneration_claim=stale_claim)
    snapshot_before = json.loads(json.dumps(bus.get_snapshot(session_id)))
    await _expire_regeneration_claim(manager, session_id)
    assert await manager.cancel_expired_regeneration_claim(session_id, stale_claim)

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="build_dataset_usd",
            state=StepState.RUNNING,
            percent=50,
            extra={"rendered_images": ["stale-after-cancel.png"]},
        )
    )

    assert bus.get_snapshot(session_id) == snapshot_before
    assert bus.get_queue(session_id).empty()
    assert await bus.get_fenced_snapshot(session_id) is None


@pytest.mark.asyncio
async def test_pinned_sse_drops_event_queued_before_claim_takeover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "completed"})
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    stale_claim = await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "running"},
        lease_seconds=60,
    )
    bus = EventBus()
    bus.set_session_manager(manager)
    await bus.seed_pending_session(session_id, regeneration_claim=stale_claim)
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: bus)
    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: manager)
    response = await pipeline_router.stream_progress_events(session_id)
    await bus.get_queue(session_id).put(
        ProgressEvent(
            session_id=session_id,
            step="predict",
            state=StepState.RUNNING,
            percent=90,
            message="queued on old pod",
        )
    )

    await _expire_regeneration_claim(manager, session_id)
    takeover_plan = await manager.get_session_metadata_versioned(session_id)
    assert takeover_plan.version is not None
    await manager.claim_regeneration(
        session_id,
        expected_version=takeover_plan.version,
        updates={"status": "running"},
        lease_seconds=60,
    )

    with pytest.raises(StopAsyncIteration):
        await anext(response.body_iterator)


@pytest.mark.asyncio
async def test_new_sse_on_non_owner_rejects_pending_active_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id, _session_dir, _state_path = await _ready_apply_session(manager)
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "pending"},
        lease_seconds=60,
    )
    non_owner_bus = EventBus()
    non_owner_bus.set_session_manager(manager)
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: non_owner_bus)
    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: manager)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.stream_progress_events(session_id)
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_old_standard_terminal_backlog_is_fenced_after_remote_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    pod_a = SessionManager(tmp_path / "pod-a", store=shared_store)
    pod_b = SessionManager(tmp_path / "pod-b", store=shared_store)
    session_id = str(uuid4())
    await pod_a.create_session(session_id)
    old_bus = EventBus()
    old_bus.set_session_manager(pod_a)
    await old_bus.seed_pending_session(session_id)
    assert await pod_a.finalize_standard_pipeline(
        session_id,
        {
            "status": "completed",
            "results": {"owner": "standard"},
            "coverage": None,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    terminal_event = ProgressEvent(
        session_id=session_id,
        step="pipeline",
        state=StepState.COMPLETED,
        percent=100,
        extra={"pipeline_completed": True, "coverage": None},
    )
    await old_bus.emit(terminal_event)
    assert await pod_a.mark_terminal_events_quiesced(
        session_id,
        expected_status="completed",
    )
    planned = await pod_b.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    await pod_b.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "pending"},
        lease_seconds=60,
    )

    assert await old_bus.get_fenced_snapshot(session_id) is None
    assert not await old_bus.queued_event_is_current(terminal_event)
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: old_bus)
    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: pod_a)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.stream_progress_events(session_id)
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_event_bus_allows_only_matching_persisted_terminal_claim_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "completed"})
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    claim = await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "running"},
    )
    bus = EventBus()
    bus.set_session_manager(manager)
    await bus.seed_pending_session(session_id, regeneration_claim=claim)
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: bus)
    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: manager)
    response = await pipeline_router.stream_progress_events(session_id)
    running = ProgressEvent(
        session_id=session_id,
        step="predict",
        state=StepState.RUNNING,
        percent=50,
    )
    await bus.emit(running)
    assert await manager.finalize_regeneration_claim(
        session_id,
        claim,
        updates={"status": "completed", "coverage": {"policy": "allow_partial"}},
    )
    terminal = ProgressEvent(
        session_id=session_id,
        step="pipeline",
        state=StepState.COMPLETED,
        percent=100,
        extra={
            "pipeline_completed": True,
            "coverage": {"policy": "allow_partial"},
        },
    )
    assert await bus.event_is_current(terminal)
    await bus.emit(terminal)
    running_item = await anext(response.body_iterator)
    terminal_item = await anext(response.body_iterator)
    done_item = await anext(response.body_iterator)
    assert running_item["event"] == "progress"
    assert json.loads(running_item["data"])["state"] == "running"
    assert terminal_item["event"] == "progress"
    assert json.loads(terminal_item["data"])["state"] == "completed"
    assert done_item["event"] == "done"
    with pytest.raises(StopAsyncIteration):
        await anext(response.body_iterator)
    assert await bus.get_fenced_snapshot(session_id) is None
