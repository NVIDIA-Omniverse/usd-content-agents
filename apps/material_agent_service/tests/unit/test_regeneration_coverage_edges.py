# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused edge coverage for regeneration fencing and publication paths."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastapi import HTTPException

from ...service.artifact_lineage import initial_artifact_validity
from ...service.models.requests import PipelineStep, RegenerateRequest
from ...service.routers import pipeline_router
from ...service.runtime.bus import EventBus
from ...service.runtime.events import StepState
from ...service.session.manager import RegenerationClaim, SessionManager
from ...service.storage.base import METADATA_KEY, JsonPreconditionError, VersionedJson
from ...service.workers import executor


async def _ready_apply_session(manager: SessionManager) -> tuple[str, Path]:
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    scene = session_dir / "input" / "scene.usda"
    scene.write_text("#usda 1.0\n", encoding="utf-8")
    predictions = session_dir / "cache" / "predictions" / "predictions.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text('{"id": "/Root", "material": "Steel"}\n')
    state = session_dir / "cache" / ".pipeline_state.json"
    state.write_text(
        json.dumps(
            {
                "completed_steps": ["predict"],
                "failed_steps": [],
                "step_errors": {},
                "step_outputs": {
                    "predict": {"predictions_path": str(predictions)},
                },
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
    return session_id, session_dir


def _install_regeneration_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    manager: SessionManager,
    event_bus: EventBus,
    registry: object,
    *,
    captured_configs: list[dict[str, Any]] | None = None,
) -> None:
    import material_agent.api as material_api

    def build_config(**kwargs: Any) -> dict[str, Any]:
        steps: dict[str, dict[str, Any]] = {}
        for step in kwargs["enabled_steps"]:
            if step == "build_dataset_usd":
                steps[step] = {
                    "renderer": {},
                    "num_workers": 1,
                    "max_concurrent_requests": 1,
                }
            else:
                steps[step] = {}
        config = {
            "input": {"usd_path": kwargs["input_usd_path"]},
            "output": {"usd_path": kwargs["output_usd_path"]},
            "steps": steps,
        }
        if captured_configs is not None:
            config["steps"].update(
                {
                    "build_dataset_prepare_dataset": {"prompts": {}},
                    "render": {},
                }
            )
            captured_configs.append(config)
        return config

    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: manager)
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: event_bus)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    monkeypatch.setattr(material_api, "build_unified_pipeline_config", build_config)


class _Registry:
    def __init__(
        self,
        *,
        reserved: set[str] | None = None,
        running: set[str] | None = None,
        register_error: BaseException | None = None,
        after_prepare: Any | None = None,
    ) -> None:
        self.reserved = reserved or set()
        self.running = running or set()
        self.register_error = register_error
        self.after_prepare = after_prepare
        self.cancelled: list[str] = []

    def is_reserved(self, session_id: str) -> bool:
        return session_id in self.reserved

    def is_running(self, session_id: str) -> bool:
        return session_id in self.running

    async def cancel(self, session_id: str) -> bool:
        self.cancelled.append(session_id)
        return session_id in self.reserved or session_id in self.running

    async def register(
        self,
        _session_id: str,
        coro: Any,
        *,
        before_start: Any,
    ) -> None:
        try:
            await before_start()
            if self.after_prepare is not None:
                self.after_prepare()
            if self.register_error is not None:
                raise self.register_error
        finally:
            coro.close()


@pytest.mark.asyncio
async def test_cancel_pipeline_retries_claim_and_cancels_running_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "completed"})
    snapshot = await manager.get_session_metadata_versioned(session_id)
    assert snapshot.version is not None
    await manager.claim_regeneration(
        session_id,
        expected_version=snapshot.version,
        updates={"status": "running"},
    )
    registry = _Registry(running={session_id})
    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: manager)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    original_cancel = manager.cancel_regeneration_claim
    attempts = 0

    async def lose_once(
        requested_session_id: str,
        claim: RegenerationClaim,
    ) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return False
        return await original_cancel(requested_session_id, claim)

    monkeypatch.setattr(manager, "cancel_regeneration_claim", lose_once)
    response = await pipeline_router.cancel_pipeline(session_id)
    assert response["status"] == "cancelling"
    assert attempts == 2
    assert registry.cancelled == [session_id]


@pytest.mark.asyncio
async def test_cancel_pipeline_reports_claim_vanished_during_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "completed"})
    snapshot = await manager.get_session_metadata_versioned(session_id)
    assert snapshot.version is not None
    await manager.claim_regeneration(
        session_id,
        expected_version=snapshot.version,
        updates={"status": "running"},
    )
    original_read = manager.get_session_metadata_versioned
    reads = 0

    async def vanish_after_initial_read(
        requested_session_id: str,
    ) -> VersionedJson:
        nonlocal reads
        reads += 1
        if reads == 1:
            return await original_read(requested_session_id)
        return VersionedJson(value=None, version=None)

    async def lose_claim_cancellation(
        _requested_session_id: str,
        _claim: RegenerationClaim,
    ) -> bool:
        return False

    monkeypatch.setattr(
        manager,
        "get_session_metadata_versioned",
        vanish_after_initial_read,
    )
    monkeypatch.setattr(
        manager,
        "cancel_regeneration_claim",
        lose_claim_cancellation,
    )
    monkeypatch.setattr(
        pipeline_router,
        "get_session_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        pipeline_router,
        "get_job_registry",
        lambda: _Registry(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.cancel_pipeline(session_id)

    assert exc_info.value.status_code == 404
    assert reads == 2


@pytest.mark.asyncio
async def test_cancel_pipeline_reservation_and_cas_race_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: manager)

    reserved_id = str(uuid4())
    await manager.create_session(reserved_id)
    await manager.update_session(reserved_id, {"status": "pending"})
    reserved_registry = _Registry(reserved={reserved_id})
    monkeypatch.setattr(
        pipeline_router,
        "get_job_registry",
        lambda: reserved_registry,
    )
    response = await pipeline_router.cancel_pipeline(reserved_id)
    assert response["status"] == "cancelled"
    metadata = await manager.get_session_metadata(reserved_id)
    assert metadata is not None and metadata["can_cancel"] is False

    vanished_id = str(uuid4())
    await manager.create_session(vanished_id)
    await manager.update_session(vanished_id, {"status": "pending"})
    vanished_registry = _Registry(reserved={vanished_id})
    monkeypatch.setattr(
        pipeline_router,
        "get_job_registry",
        lambda: vanished_registry,
    )
    original_get = manager.get_session_metadata_versioned
    reads = 0

    async def vanish_after_reservation(requested_session_id: str):
        nonlocal reads
        reads += 1
        if reads == 2:
            return SimpleNamespace(value=None, version=None)
        return await original_get(requested_session_id)

    monkeypatch.setattr(
        manager,
        "get_session_metadata_versioned",
        vanish_after_reservation,
    )
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.cancel_pipeline(vanished_id)
    assert exc_info.value.status_code == 404
    monkeypatch.setattr(manager, "get_session_metadata_versioned", original_get)

    missing_after_cas_id = str(uuid4())
    await manager.create_session(missing_after_cas_id)
    await manager.update_session(missing_after_cas_id, {"status": "pending"})
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _Registry())
    original_replace = manager.store.replace_json_if_version

    async def lose_and_delete(*_args: Any, **_kwargs: Any) -> None:
        await manager.store.delete_session(missing_after_cas_id)
        raise JsonPreconditionError("lost")

    monkeypatch.setattr(manager.store, "replace_json_if_version", lose_and_delete)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.cancel_pipeline(missing_after_cas_id)
    assert exc_info.value.status_code == 404
    monkeypatch.setattr(manager.store, "replace_json_if_version", original_replace)

    contended_id = str(uuid4())
    await manager.create_session(contended_id)
    await manager.update_session(contended_id, {"status": "pending"})

    async def always_lose(*_args: Any, **_kwargs: Any) -> None:
        raise JsonPreconditionError("contended")

    monkeypatch.setattr(manager.store, "replace_json_if_version", always_lose)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.cancel_pipeline(contended_id)
    assert exc_info.value.status_code == 409
    assert not await manager.is_cancelled(contended_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("descriptions", "expected_prompt"),
    [
        (["front"], "This is a reference image: front"),
        ([], "This is reference image 1 of the asset you will match this look exactly"),
    ],
)
async def test_regeneration_reference_prompt_render_and_heartbeat_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    descriptions: list[str],
    expected_prompt: str,
) -> None:
    manager = SessionManager(tmp_path)
    session_id, session_dir = await _ready_apply_session(manager)
    reference = session_dir / "input" / "reference_images" / "reference_0.png"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(b"png")
    (reference.parent / "descriptions.json").write_text(json.dumps(descriptions))
    configs: list[dict[str, Any]] = []
    registry = _Registry()
    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        EventBus(),
        registry,
        captured_configs=configs,
    )

    async def yielding_hydration(*_args: Any, **_kwargs: Any) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(
        pipeline_router,
        "_hydrate_regeneration_inputs",
        yielding_hydration,
    )
    response = await pipeline_router.regenerate_pipeline(
        session_id,
        RegenerateRequest(steps=[PipelineStep.APPLY]),
    )
    assert response.status == "pending"
    config = configs[0]
    prompts = config["steps"]["build_dataset_prepare_dataset"]["prompts"]
    assert prompts["vlm_image_prompts"]["reference_images"] == [expected_prompt]
    assert config["steps"]["render"]["image_size"] == [512, 512]


@pytest.mark.asyncio
async def test_regeneration_rejects_unquiesced_terminal_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id, _ = await _ready_apply_session(manager)
    snapshot = await manager.get_session_metadata_versioned(session_id)
    assert snapshot.value is not None and snapshot.version is not None
    metadata = dict(snapshot.value)
    metadata["terminal_events_quiesced"] = False
    await manager.store.replace_json_if_version(
        session_id,
        METADATA_KEY,
        metadata,
        snapshot.version,
    )
    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: manager)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(
            session_id,
            RegenerateRequest(steps=[PipelineStep.APPLY]),
        )
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cancel", "renew_false", "renew_then_cancel"])
async def test_preparation_heartbeat_cancels_owner_for_each_fencing_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    class _HeartbeatManager(SessionManager):
        cancel_checks = 0

        async def is_regeneration_cancel_requested(
            self,
            session_id: str,
            claim: RegenerationClaim,
        ) -> bool:
            del session_id, claim
            self.cancel_checks += 1
            if mode == "cancel":
                return True
            return mode == "renew_then_cancel" and self.cancel_checks > 1

        async def renew_regeneration_claim(
            self,
            session_id: str,
            claim: RegenerationClaim,
            *,
            lease_seconds: float = 300.0,
        ) -> bool:
            del session_id, claim, lease_seconds
            return mode == "renew_then_cancel"

    manager = _HeartbeatManager(tmp_path)
    session_id, _ = await _ready_apply_session(manager)
    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        EventBus(),
        _Registry(),
    )
    monkeypatch.setattr(
        pipeline_router,
        "_REGENERATION_PREPARATION_HEARTBEAT_SECONDS",
        0.0,
    )

    async def block_hydration(*_args: Any, **_kwargs: Any) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(
        pipeline_router,
        "_hydrate_regeneration_inputs",
        block_hydration,
    )
    task = asyncio.create_task(
        pipeline_router.regenerate_pipeline(
            session_id,
            RegenerateRequest(steps=[PipelineStep.APPLY]),
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("abort_mode", ["raise", "false"])
async def test_regeneration_rollback_handles_lost_or_failing_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    abort_mode: str,
) -> None:
    manager = SessionManager(tmp_path)
    session_id, _ = await _ready_apply_session(manager)

    def install_abort_fault() -> None:
        async def abort(*_args: Any, **_kwargs: Any) -> bool:
            if abort_mode == "raise":
                raise RuntimeError("abort unavailable")
            return False

        monkeypatch.setattr(manager, "abort_regeneration_claim", abort)

    registry = _Registry(
        register_error=HTTPException(status_code=418, detail="registration failed"),
        after_prepare=install_abort_fault,
    )
    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        EventBus(),
        registry,
    )
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(
            session_id,
            RegenerateRequest(steps=[PipelineStep.APPLY]),
        )
    assert exc_info.value.status_code == 418


@pytest.mark.asyncio
async def test_regeneration_rollback_cleanup_failures_are_non_masking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id, session_dir = await _ready_apply_session(manager)
    state_path = session_dir / "cache" / ".pipeline_state.json"
    bus = EventBus()
    original_write_bytes = Path.write_bytes

    def install_cleanup_faults() -> None:
        def fail_checkpoint_restore(path: Path, data: bytes) -> int:
            if path.name == ".pipeline_state.regeneration-rollback.json":
                raise OSError("checkpoint unavailable")
            return original_write_bytes(path, data)

        async def fail_cancel_restore(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("cancel restore unavailable")

        async def fail_bus_restore(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("bus restore unavailable")

        monkeypatch.setattr(Path, "write_bytes", fail_checkpoint_restore)
        monkeypatch.setattr(manager, "restore_cancellation", fail_cancel_restore)
        monkeypatch.setattr(bus, "restore_session", fail_bus_restore)

    registry = _Registry(
        register_error=RuntimeError("scheduler failed"),
        after_prepare=install_cleanup_faults,
    )
    _install_regeneration_dependencies(monkeypatch, manager, bus, registry)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(
            session_id,
            RegenerateRequest(steps=[PipelineStep.APPLY]),
        )
    assert exc_info.value.status_code == 503
    assert state_path.exists()


@pytest.mark.asyncio
async def test_carry_forward_filters_and_validity_downgrades() -> None:
    claim = RegenerationClaim(
        generation=2,
        token="token",
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )

    class _CarryManager:
        def __init__(self) -> None:
            self.copied: dict[str, bytes] = {}

        def resolve_published_artifact_key(
            self,
            metadata: dict[str, Any],
            logical_key: str,
            *,
            legacy_key: str,
        ) -> str | None:
            del metadata, legacy_key
            if logical_key in {"none-source", "cache/predictions/predictions.jsonl"}:
                return None
            return f"prior/{logical_key}"

        async def read_from_store(self, _session_id: str, key: str) -> bytes | None:
            return None if key.endswith("missing-data") else b"prior"

        async def put_bytes_to_store(
            self,
            _session_id: str,
            key: str,
            data: bytes,
        ) -> None:
            self.copied[key] = data

    prior_keys = {
        "already",
        "cache/dataset/usd/prims.jsonl",
        "cache/preview/old.png",
        "cache/generated_material_library/materials.yaml",
        "cache/predictions/prediction_report.html",
        "cache/predictions/predictions.jsonl",
        "none-source",
        "missing-data",
    }
    metadata = {
        "published_artifacts": {
            "artifacts": {key: f"prior/{key}" for key in prior_keys}
        }
    }
    manager = _CarryManager()
    validity = {key: True for key in initial_artifact_validity()}
    artifact_map = {"already": "runs/2-token/already"}
    verified = await executor._carry_forward_regeneration_artifacts(
        manager,  # type: ignore[arg-type]
        "sid",
        claim,
        metadata,
        validity,
        artifact_map,
        {"build_dataset_usd", "generate_material_library"},
    )
    assert verified["raw_predictions"] is False
    assert verified["prediction_report"] is False
    assert verified["previews"] is False
    assert not any("old.png" in key for key in artifact_map)

    await executor._carry_forward_regeneration_artifacts(
        manager,  # type: ignore[arg-type]
        "sid",
        claim,
        {
            "published_artifacts": {
                "artifacts": {"cache/dataset/dataset.jsonl": "prior/dataset"}
            }
        },
        initial_artifact_validity(),
        {},
        {"build_dataset_prepare_dataset"},
    )
    await executor._carry_forward_regeneration_artifacts(
        manager,  # type: ignore[arg-type]
        "sid",
        claim,
        {
            "published_artifacts": {
                "artifacts": {"preview/old.png": "prior/preview/old.png"}
            }
        },
        initial_artifact_validity(),
        {},
        set(),
    )


@pytest.mark.asyncio
async def test_executor_claim_wrappers_monitors_and_generated_auxiliary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "completed"})
    snapshot = await manager.get_session_metadata_versioned(session_id)
    assert snapshot.version is not None
    claim = await manager.claim_regeneration(
        session_id,
        expected_version=snapshot.version,
        updates={"status": "running"},
    )

    assert await executor._update_pipeline_session(
        manager,
        session_id,
        {"current_step": "predict"},
        regeneration_claim=claim,
    )
    assert await executor._finalize_pipeline_session(
        manager,
        session_id,
        {"status": "completed", "results": {}},
        regeneration_claim=claim,
        artifact_map={},
    )
    await executor._mark_standard_terminal_events_quiesced(
        manager,
        session_id,
        regeneration_claim=claim,
        expected_status="completed",
    )

    generated = session_dir / "cache" / "generated_material_library"
    generated.mkdir(parents=True)
    (generated / "materials.yaml").write_text("materials: {}\n")
    assert executor._artifact_file_signature(generated) is None
    artifact_map: dict[str, str] = {}
    await executor._promote_current_run_artifacts(
        manager,
        session_id,
        session_dir,
        ["generate_material_library"],
        {"generate_material_library": {}},
        regeneration_claim=claim,
        artifact_map=artifact_map,
        baseline_signatures={},
    )
    assert "cache/generated_material_library/materials.yaml" in artifact_map

    class _PollManager:
        calls = 0

        async def is_cancelled(self, _session_id: str) -> bool:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("cancellation-poll-secret-727")
            return True

    original_sleep = executor.asyncio.sleep

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(executor.asyncio, "sleep", no_sleep)
    owner = asyncio.create_task(original_sleep(60))
    with caplog.at_level(logging.ERROR):
        await executor._poll_standard_pipeline_cancellation(
            _PollManager(),  # type: ignore[arg-type]
            session_id,
            owner,
        )
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert "cancellation-poll-secret-727" not in caplog.text
    assert "code=pipeline_cancellation_poll_failed" in caplog.text

    class _MonitorManager:
        checks = 0

        async def is_regeneration_cancel_requested(
            self,
            _session_id: str,
            _claim: RegenerationClaim,
        ) -> bool:
            self.checks += 1
            return self.checks > 1

        async def renew_regeneration_claim(
            self,
            _session_id: str,
            _claim: RegenerationClaim,
            **_kwargs: Any,
        ) -> bool:
            return True

    owner = asyncio.create_task(original_sleep(60))
    await executor._monitor_regeneration_claim(
        _MonitorManager(),  # type: ignore[arg-type]
        session_id,
        claim,
        owner,
    )
    with pytest.raises(asyncio.CancelledError):
        await owner

    class _InvalidFinalizeManager:
        async def is_regeneration_cancel_requested(self, *_args: Any) -> bool:
            return False

        async def renew_regeneration_claim(self, *_args: Any, **_kwargs: Any) -> bool:
            return False

        async def get_session_metadata(self, _session_id: str) -> dict[str, Any]:
            return {
                "status": "completed",
                "regeneration_claim": {
                    "generation": claim.generation,
                    "token": claim.token,
                    "lease_expires_at": claim.lease_expires_at.isoformat(),
                    "active": False,
                    "finalized_at": "invalid",
                },
            }

    owner = asyncio.create_task(original_sleep(60))
    await executor._monitor_regeneration_claim(
        _InvalidFinalizeManager(),  # type: ignore[arg-type]
        session_id,
        claim,
        owner,
    )
    with pytest.raises(asyncio.CancelledError):
        await owner


@pytest.mark.asyncio
async def test_claimed_pipeline_failure_publishes_preview_and_finalizes_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    await manager.update_session(
        session_id,
        {
            "status": "completed",
            "artifact_validity": initial_artifact_validity(),
        },
    )
    snapshot = await manager.get_session_metadata_versioned(session_id)
    assert snapshot.version is not None
    claim = await manager.claim_regeneration(
        session_id,
        expected_version=snapshot.version,
        updates={"status": "running"},
    )
    bus = EventBus()
    bus.set_session_manager(manager)
    await bus.seed_pending_session(session_id, regeneration_claim=claim)
    bus._state[session_id]["preview_images"] = ["current.png"]
    monkeypatch.setattr(executor, "get_event_bus", lambda: bus)

    async def failed_run(_pipeline_input: Any) -> SimpleNamespace:
        preview = session_dir / "cache" / "preview" / "current.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        preview.write_bytes(b"current")
        return SimpleNamespace(
            success=False,
            error="failed after preview",
            completed_steps=["build_dataset_usd"],
            step_results={"build_dataset_usd": {"num_images": 1}},
            raw_result={},
        )

    monkeypatch.setattr(executor, "arun_pipeline", failed_run)
    await executor._execute_pipeline_inner(
        session_id,
        {"steps": {"build_dataset_usd": {}}},
        manager,
        regeneration_claim=claim,
    )
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "failed"
    assert metadata["preview_images"] == ["current.png"]
    assert metadata["published_artifacts"]["artifacts"]
    failure_event = await bus.get_queue(session_id).get()
    assert failure_event.state == StepState.FAILED
    assert failure_event.message == "material_pipeline_result_failed"
    assert failure_event.extra == {"pipeline_failed": True}


@pytest.mark.asyncio
async def test_sse_timeout_stops_when_local_fenced_snapshot_vanishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _VanishingBus:
        def __init__(self) -> None:
            self.snapshot_reads = 0
            self.queue: asyncio.Queue = asyncio.Queue()

        async def get_fenced_snapshot(self, _session_id: str) -> dict | None:
            self.snapshot_reads += 1
            return {"status": "pending"} if self.snapshot_reads == 1 else None

        def get_queue(self, _session_id: str) -> asyncio.Queue:
            return self.queue

    async def timeout_wait_for(awaitable: object, timeout: float) -> None:
        del timeout
        close = getattr(awaitable, "close", None)
        if close is not None:
            close()
        raise TimeoutError

    bus = _VanishingBus()
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: bus)
    monkeypatch.setattr(pipeline_router.asyncio, "wait_for", timeout_wait_for)
    response = await pipeline_router.stream_progress_events("session")
    with pytest.raises(StopAsyncIteration):
        await anext(response.body_iterator)


@pytest.mark.asyncio
async def test_regeneration_rollback_without_prior_checkpoint_keeps_it_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    (session_dir / "input" / "scene.usda").write_text("#usda 1.0\n")
    await manager.update_session(
        session_id,
        {
            "status": "completed",
            "results": {},
            "coverage": None,
            "completed_at": "2026-01-01T00:00:00+00:00",
        },
    )
    _install_regeneration_dependencies(
        monkeypatch,
        manager,
        EventBus(),
        _Registry(register_error=RuntimeError("scheduler failed")),
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(
            session_id,
            RegenerateRequest(steps=[PipelineStep.BUILD_DATASET]),
        )
    assert exc_info.value.status_code == 503
    assert not (session_dir / "cache" / ".pipeline_state.json").exists()


@pytest.mark.asyncio
async def test_claim_monitor_propagates_external_cancellation() -> None:
    claim = RegenerationClaim(
        generation=1,
        token="token",
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    entered_check = asyncio.Event()

    class _BlockingManager:
        async def is_regeneration_cancel_requested(
            self,
            _session_id: str,
            _claim: RegenerationClaim,
        ) -> bool:
            entered_check.set()
            await asyncio.Future()
            return False

        async def renew_regeneration_claim(
            self,
            _session_id: str,
            _claim: RegenerationClaim,
            **_kwargs: Any,
        ) -> bool:
            return True

    owner = asyncio.create_task(asyncio.sleep(60))
    monitor = asyncio.create_task(
        executor._monitor_regeneration_claim(
            _BlockingManager(),  # type: ignore[arg-type]
            "session",
            claim,
            owner,
        )
    )
    await entered_check.wait()
    monitor.cancel()
    with pytest.raises(asyncio.CancelledError):
        await monitor
    assert not owner.done()
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
