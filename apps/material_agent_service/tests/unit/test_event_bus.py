# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for service progress event state."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from ...service.runtime.bus import EventBus, EventBusSessionSnapshot
from ...service.runtime.events import ProgressEvent, StepState
from ...service.session.manager import RegenerationClaim
from ...service.workers.executor import (
    _cluster_telemetry_attributes,
    _merge_completed_steps_from_result,
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_bus_tracks_dynamic_total_steps_for_injected_steps():
    """Progress state should reflect steps injected at execution time."""
    bus = EventBus()
    session_id = "test-session"
    steps = [
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "cluster_prims",
        "predict",
        "expand_cluster_predictions",
    ]

    for index, step in enumerate(steps, 1):
        await bus.emit(
            ProgressEvent(
                session_id=session_id,
                step=step,
                state=StepState.RUNNING,
                percent=0,
                extra={"step_index": index, "total_steps": len(steps)},
            )
        )
        await bus.emit(
            ProgressEvent(
                session_id=session_id,
                step=step,
                state=StepState.COMPLETED,
                percent=100,
                extra={"step_index": index, "total_steps": len(steps)},
            )
        )

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="pipeline",
            state=StepState.COMPLETED,
            percent=100,
            extra={"pipeline_completed": True, "coverage": {}},
        )
    )

    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["overall_progress"]["current_step"] == len(steps)
    assert snapshot["overall_progress"]["total_steps"] == len(steps)
    assert snapshot["overall_progress"]["percent"] == 100
    assert snapshot["completed_steps"][2]["display_name"] == "Clustering Prims"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_bus_completed_step_stats_are_json_safe():
    """Task outputs can contain Path objects, but status snapshots must be JSON-safe."""
    bus = EventBus()
    session_id = "test-json-safe-session"

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="build_dataset_prepare_dataset",
            state=StepState.RUNNING,
            percent=0,
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="build_dataset_prepare_dataset",
            state=StepState.COMPLETED,
            percent=100,
            extra={"outputs": {"dataset_path": Path("/tmp/material-agent/dataset")}},
        )
    )

    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    stats = snapshot["completed_steps"][0]["stats"]
    assert stats["outputs"]["dataset_path"] == "/tmp/material-agent/dataset"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_bus_claim_guard_and_queue_edge_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    claim = RegenerationClaim(
        generation=1,
        token="token",
        lease_expires_at=now + timedelta(minutes=5),
    )
    event = ProgressEvent(
        session_id="claimed",
        step="pipeline",
        state=StepState.RUNNING,
    )
    bus = EventBus()
    bus.bind_regeneration_claim(event.session_id, claim)
    monkeypatch.setattr(bus, "_get_session_manager", lambda: None)
    assert await bus.get_fenced_snapshot(event.session_id) is None
    assert not await bus.event_is_current(event)
    assert not await bus.queued_event_is_current(event)

    class _MetadataManager:
        metadata: dict | None = None

        async def get_session_metadata(self, _session_id: str) -> dict | None:
            return self.metadata

    manager = _MetadataManager()
    bus.set_session_manager(manager)  # type: ignore[arg-type]
    monkeypatch.setattr(bus, "_get_session_manager", lambda: manager)
    assert await bus.get_fenced_snapshot(event.session_id) is None
    assert not await bus.event_is_current(event)
    assert not await bus.queued_event_is_current(event)

    raw_claim = {
        "generation": claim.generation,
        "token": claim.token,
        "lease_expires_at": claim.lease_expires_at.isoformat(),
        "active": True,
    }
    manager.metadata = {"status": "running", "regeneration_claim": raw_claim}
    assert await bus.event_is_current(event)
    assert await bus.queued_event_is_current(event)
    bus._regeneration_claims.pop(event.session_id)
    assert not await bus.event_is_current(event)
    bus.bind_regeneration_claim(event.session_id, claim)

    manager.metadata = {
        "status": "cancelled",
        "regeneration_claim": {
            **raw_claim,
            "active": False,
            "finalized_at": now.isoformat(),
        },
    }
    cancelled = ProgressEvent(
        session_id=event.session_id,
        step="pipeline",
        state=StepState.CANCELLED,
        extra={"pipeline_cancelled": True},
    )
    assert await bus.event_is_current(cancelled)

    manager.metadata["regeneration_claim"]["aborted_at"] = now.isoformat()
    assert not await bus.queued_event_is_current(cancelled)
    manager.metadata["regeneration_claim"].pop("aborted_at")
    manager.metadata["regeneration_claim"].pop("finalized_at")
    assert not await bus.queued_event_is_current(cancelled)
    manager.metadata["regeneration_claim"]["finalized_at"] = "not-a-date"
    assert not await bus.queued_event_is_current(cancelled)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_bus_restore_queue_and_rendered_image_edges() -> None:
    bus = EventBus()
    session_id = "restore"
    claim = RegenerationClaim(
        generation=2,
        token="restore-token",
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    queue = bus.get_queue(session_id)
    stale = ProgressEvent(
        session_id=session_id,
        step="stale",
        state=StepState.RUNNING,
    )
    restored = ProgressEvent(
        session_id=session_id,
        step="restored",
        state=StepState.RUNNING,
    )
    queue.put_nowait(stale)
    await bus.restore_session(
        session_id,
        EventBusSessionSnapshot(
            state_present=False,
            state=None,
            queue_present=True,
            queue=queue,
            queued_events=(restored,),
            regeneration_claim=claim,
        ),
    )
    assert queue.qsize() == 1
    assert queue.get_nowait() is restored
    assert bus._regeneration_claims[session_id] == claim

    queue.put_nowait(stale)
    await bus.restore_session(
        session_id,
        EventBusSessionSnapshot(
            state_present=False,
            state=None,
            queue_present=False,
            queue=None,
            queued_events=(),
            regeneration_claim=None,
        ),
    )
    assert session_id not in bus._queues

    live_session = "rendered"
    await bus.emit(
        ProgressEvent(
            session_id=live_session,
            step="render",
            state=StepState.RUNNING,
            extra={"rendered_images": [42, "nested/current.png", "current.png"]},
        )
    )
    snapshot = bus.get_snapshot(live_session)
    assert snapshot is not None
    assert snapshot["preview_images"] == ["current.png"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_bus_overall_progress_includes_cluster_steps():
    bus = EventBus()
    session_id = "test-cluster-progress-session"

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="cluster_prims",
            state=StepState.RUNNING,
            percent=50,
        )
    )
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["overall_progress"]["percent"] == 52

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="expand_cluster_predictions",
            state=StepState.RUNNING,
            percent=100,
        )
    )
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["overall_progress"]["percent"] == 90


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_bus_does_not_publish_provisional_completion():
    bus = EventBus()
    session_id = "test-provisional-completion"

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="apply",
            state=StepState.RUNNING,
            percent=100,
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="apply",
            state=StepState.COMPLETED,
            percent=100,
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="pipeline",
            state=StepState.COMPLETED,
            percent=100,
            extra={"pipeline_completed": True},
        )
    )

    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "running"
    assert snapshot["overall_progress"]["percent"] == 100
    assert [step["name"] for step in snapshot["completed_steps"]] == ["apply"]
    assert "completed_at" not in snapshot


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_bus_records_late_completed_step_after_pipeline_completion():
    """A fast final step can complete after the pipeline terminal event."""
    bus = EventBus()
    session_id = "test-late-completed-step-session"

    for index, step in enumerate(
        [
            "build_dataset_usd",
            "build_dataset_prepare_dataset",
            "cluster_prims",
            "predict",
            "expand_cluster_predictions",
        ],
        1,
    ):
        await bus.emit(
            ProgressEvent(
                session_id=session_id,
                step=step,
                state=StepState.RUNNING,
                percent=0,
                extra={"step_index": index, "total_steps": 5},
            )
        )
        if step == "expand_cluster_predictions":
            continue
        await bus.emit(
            ProgressEvent(
                session_id=session_id,
                step=step,
                state=StepState.COMPLETED,
                percent=100,
                extra={"step_index": index, "total_steps": 5},
            )
        )

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="pipeline",
            state=StepState.COMPLETED,
            percent=100,
            extra={"pipeline_completed": True, "coverage": {}},
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="expand_cluster_predictions",
            state=StepState.COMPLETED,
            percent=100,
            extra={"step_index": 5, "total_steps": 5},
        )
    )

    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert [step["name"] for step in snapshot["completed_steps"]] == [
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "cluster_prims",
        "predict",
        "expand_cluster_predictions",
    ]
    assert snapshot["overall_progress"]["current_step"] == 5
    assert snapshot["overall_progress"]["total_steps"] == 5
    assert snapshot["overall_progress"]["percent"] == 100


@pytest.mark.unit
def test_executor_merges_result_completed_steps_missing_from_event_snapshot():
    """Final persistence should include fast terminal steps missed by EventBus."""
    completed_steps = [
        {
            "name": "predict",
            "display_name": "Running VLM Predictions",
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:00:01+00:00",
            "duration_seconds": 1,
            "stats": {"successful": 1},
        }
    ]
    result = SimpleNamespace(
        completed_steps=["predict", "expand_cluster_predictions"],
        step_results={
            "expand_cluster_predictions": {
                "predictions_path": Path("/tmp/predictions.jsonl")
            }
        },
    )

    merged = _merge_completed_steps_from_result(
        completed_steps,
        result,
        {"expand_cluster_predictions": 0.003},
    )

    assert [step["name"] for step in merged] == [
        "predict",
        "expand_cluster_predictions",
    ]
    assert merged[-1]["display_name"] == "Expanding Cluster Predictions"
    assert (
        merged[-1]["stats"]["outputs"]["predictions_path"] == "/tmp/predictions.jsonl"
    )


@pytest.mark.unit
def test_cluster_telemetry_attributes_are_sanitized():
    attrs = _cluster_telemetry_attributes(
        {
            "steps": {
                "cluster_prims": {
                    "embedding_service": "nim",
                    "embedding_model": "nvidia/llama-nemotron-embed-vl-1b-v2",
                    "api_key": "secret",
                }
            }
        },
        {
            "cluster_prims_ran": True,
            "cluster_total_prims": 10,
            "cluster_count": 5,
            "cluster_representative_count": 5,
            "cluster_reduction_percent": 50.0,
            "cluster_max_size": 25,
            "cluster_capped_count": 0,
        },
    )

    assert attrs["maa.clustering.enabled"] is True
    assert attrs["maa.clustering.embedding_backend"] == "nim"
    assert (
        attrs["maa.clustering.embedding_model"]
        == "nvidia/llama-nemotron-embed-vl-1b-v2"
    )
    assert attrs["maa.clustering.total_prims"] == 10
    assert "secret" not in repr(attrs)
