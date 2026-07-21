# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the per-step except path persists structured per-unit errors.

When ``GenerateTexturesTask`` / ``BlendTexturesTask`` mutate the pipeline
context with structured ``*_errors`` records before raising the threshold-
gate ``RuntimeError``, the executor's per-step except block must surface
those records on the FAILED ``ProgressEvent`` and in the persisted session
metadata. Otherwise the highest-value failure mode (the threshold gate
firing) loses the very diagnostics this code path was added to provide.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from ...service.models.responses import PipelineError
from ...service.runtime import bus as bus_module
from ...service.workers import executor


class _StubSessionManager:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.updates: list[dict[str, Any]] = []
        self.cancelled = False

    def get_session_dir(self, session_id: str) -> Path:
        return self.session_dir

    def update_session(self, session_id: str, updates: dict[str, Any]) -> None:
        self.updates.append(updates)

    def is_cancelled(self, session_id: str) -> bool:
        return self.cancelled

    def session_exists(self, session_id: str) -> bool:
        return True


class GenerateTexturesTask:  # name mirrors the real task so executor's
    # ``_TASK_CLASS_TO_STEP`` mapping resolves it to ``generate_textures``.
    """Stand-in for ``GenerateTexturesTask`` that mutates context with the
    same ``generate_textures_errors`` shape and then raises the threshold-
    gate ``RuntimeError`` from ``_raise_if_above_threshold``."""

    name = "GenerateTextures"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        context["generated_textures"] = {}
        context["generate_textures_errors"] = [
            {
                "material": "Aluminum_Brushed",
                "type": "RuntimeError",
                "status": 403,
                "message": "HTTP 403 Forbidden",
            },
            {
                "material": "Rubber_Black_Matte",
                "type": "RuntimeError",
                "status": 403,
                "message": "HTTP 403 Forbidden",
            },
        ]
        context["generate_textures_failed_count"] = 2
        context["generate_textures_attempted_count"] = 2
        raise RuntimeError(
            "2/2 texture generation requests failed via nim "
            "(failure rate 100% >= threshold 100%). First errors:\n"
            "  - Aluminum_Brushed: [RuntimeError 403] HTTP 403 Forbidden\n"
            "  - Rubber_Black_Matte: [RuntimeError 403] HTTP 403 Forbidden"
        )


def _stub_factory(context, skip=None, only=None):
    return [GenerateTexturesTask()]


async def test_failed_step_persists_structured_errors_in_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``manager.update_session`` for the failed step must include
    ``failed_step_stats`` carrying the structured ``errors`` list and
    ``textures_failed`` count -- not just ``str(e)``."""
    session_id = "fail-001"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    manager = _StubSessionManager(session_dir)

    monkeypatch.setattr(bus_module, "_event_bus", None)
    bus_module.init_event_bus(manager)

    with pytest.raises(RuntimeError, match=r"2/2 texture generation requests failed"):
        await executor._execute_pipeline_inner(
            session_id=session_id,
            config_dict={"input": {"usd_path": "/tmp/in.usd"}},
            session_manager=manager,
            event_bus=bus_module.get_event_bus(),
            session_dir=session_dir,
            only_steps=None,
            skip_steps=None,
            create_texture_pipeline_workflow=_stub_factory,
        )

    failed_updates = [u for u in manager.updates if u.get("status") == "failed"]
    assert failed_updates, "expected at least one failed-status update"
    failed = failed_updates[-1]
    assert failed["failed_step"] == "generate_textures"
    assert failed["failed_step_stats"]["textures_generated"] == 0
    assert failed["failed_step_stats"]["textures_failed"] == 2
    assert failed["failed_step_stats"]["errors"] == [
        {
            "material": "Aluminum_Brushed",
            "type": "RuntimeError",
            "status": 403,
            "message": "HTTP 403 Forbidden",
        },
        {
            "material": "Rubber_Black_Matte",
            "type": "RuntimeError",
            "status": 403,
            "message": "HTTP 403 Forbidden",
        },
    ]


async def test_failed_step_emits_structured_errors_on_progress_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The FAILED ``ProgressEvent`` ``extra`` must carry the structured
    errors so SSE consumers see per-material failures without grepping
    container logs."""
    session_id = "fail-002"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    manager = _StubSessionManager(session_dir)

    monkeypatch.setattr(bus_module, "_event_bus", None)
    bus = bus_module.init_event_bus(manager)

    captured: list[Any] = []
    original_emit = bus.emit

    async def capture(event):
        captured.append(event)
        await original_emit(event)

    bus.emit = capture  # type: ignore[method-assign]

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

    failed_events = [
        ev
        for ev in captured
        if getattr(getattr(ev, "state", None), "value", None) == "failed"
    ]
    assert failed_events, "expected at least one FAILED progress event"
    failed_event = failed_events[-1]
    assert failed_event.step == "generate_textures"
    assert failed_event.extra is not None
    assert failed_event.extra["textures_failed"] == 2
    assert len(failed_event.extra["errors"]) == 2
    assert failed_event.extra["errors"][0]["status"] == 403


async def test_outer_executor_does_not_mask_handled_step_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outer executor guard must not emit a second pipeline-level FAILED.

    The per-step handler already persisted ``failed_step`` and structured
    stats. Re-emitting FAILED for ``step="pipeline"`` would overwrite the bus
    snapshot and hide the actual failed task from /status.
    """
    from texture_agent.workflows import factory as workflow_factory

    session_id = "fail-outer-001"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    manager = _StubSessionManager(session_dir)

    monkeypatch.setattr(bus_module, "_event_bus", None)
    bus = bus_module.init_event_bus(manager)

    captured: list[Any] = []
    original_emit = bus.emit

    async def capture(event):
        captured.append(event)
        await original_emit(event)

    bus.emit = capture  # type: ignore[method-assign]
    monkeypatch.setattr(
        workflow_factory,
        "create_texture_pipeline_workflow",
        _stub_factory,
    )

    with pytest.raises(RuntimeError, match=r"2/2 texture generation requests failed"):
        await executor.execute_pipeline_async(
            session_id=session_id,
            config_dict={"input": {"usd_path": "/tmp/in.usd"}},
            session_manager=manager,
        )

    failed_events = [
        ev
        for ev in captured
        if getattr(getattr(ev, "state", None), "value", None) == "failed"
    ]
    assert [ev.step for ev in failed_events] == ["generate_textures"]

    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["failed_step"] == "generate_textures"
    assert snapshot["failed_step_stats"]["textures_failed"] == 2

    failed_updates = [u for u in manager.updates if u.get("status") == "failed"]
    assert failed_updates[-1]["failed_step"] == "generate_textures"


def test_pipeline_error_model_exposes_failed_step_stats() -> None:
    """``GET /result/{session_id}`` after a threshold-gated failure must
    surface the structured per-unit errors via ``PipelineError``. SSE
    consumers see the FAILED event live; REST polling clients only see
    what this model carries."""
    err = PipelineError(
        session_id="s",
        error_message="2/2 failed",
        failed_step="generate_textures",
        failed_step_stats={
            "textures_generated": 0,
            "textures_failed": 2,
            "errors": [
                {
                    "material": "Aluminum_Brushed",
                    "type": "RuntimeError",
                    "status": 403,
                    "message": "HTTP 403",
                }
            ],
        },
    )

    dumped = err.model_dump()
    assert dumped["failed_step_stats"]["textures_failed"] == 2
    assert dumped["failed_step_stats"]["errors"][0]["status"] == 403


def test_pipeline_error_model_defaults_failed_step_stats_none() -> None:
    """Backward-compat: callers that don't set ``failed_step_stats`` get
    ``None`` (matches the existing ``partial_results`` shape)."""
    err = PipelineError(
        session_id="s",
        error_message="boom",
        failed_step="apply_textures",
    )
    assert err.failed_step_stats is None


class _UpstreamErrorsCarryingTask:
    """Stand-in for an apply_textures step that ran cleanly but produced
    no output USD (because upstream gen/blend silently emitted zero
    textures with structured error records on context). Mirrors the
    customer-visible scenario where the threshold gate is below 1.0 and
    partial generation failures slip past it."""

    name = "ApplyTextures"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        # Upstream populated structured errors but didn't raise (threshold
        # was below 1.0). apply_textures produced no output_usd_paths.
        context["generate_textures_errors"] = [
            {
                "material": "Aluminum_Brushed",
                "type": "RuntimeError",
                "status": 403,
                "message": "HTTP 403 Forbidden",
            }
        ]
        context["generate_textures_failed_count"] = 1
        context["generated_textures"] = {}
        context["output_usd_paths"] = []  # triggers validation_error
        return context


# Match _TASK_CLASS_TO_STEP mapping for apply_textures
class ApplyTexturesTask(_UpstreamErrorsCarryingTask):
    pass


def _validation_factory(context, skip=None, only=None):
    return [ApplyTexturesTask()]


async def test_validation_error_path_persists_upstream_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The validation-error branch (apply_textures empty-output) must
    persist ``failed_step_stats`` containing the upstream gen/blend
    error list. Without this, REST clients polling ``/result`` only see
    prose -- the structured errors this MR adds are unreachable for
    apply_textures-validation failures."""
    session_id = "val-001"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    manager = _StubSessionManager(session_dir)

    monkeypatch.setattr(bus_module, "_event_bus", None)
    bus_module.init_event_bus(manager)

    with pytest.raises(RuntimeError, match="no output USD files"):
        await executor._execute_pipeline_inner(
            session_id=session_id,
            config_dict={"input": {"usd_path": "/tmp/in.usd"}},
            session_manager=manager,
            event_bus=bus_module.get_event_bus(),
            session_dir=session_dir,
            only_steps=None,
            skip_steps=None,
            create_texture_pipeline_workflow=_validation_factory,
        )

    failed_updates = [u for u in manager.updates if u.get("status") == "failed"]
    assert failed_updates, "expected validation-error to flip status to failed"
    failed = failed_updates[-1]
    stats = failed.get("failed_step_stats")
    assert stats is not None, "validation-error path must persist failed_step_stats"
    assert stats["manifest_available"] is True
    assert Path(stats["manifest_path"]).name == "artifacts_manifest.json"
    assert "upstream_errors" in stats
    assert stats["upstream_errors"]["generate_textures"]["count"] == 1
    assert stats["upstream_errors"]["generate_textures"]["errors"][0]["status"] == 403


async def test_apply_hydration_runs_off_the_asyncio_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached-map and plan hydration performs blocking disk/USD work."""
    session_id = "hydrate-thread-001"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    manager = _StubSessionManager(session_dir)
    event_loop_thread = threading.get_ident()
    hydration_threads: list[int] = []

    monkeypatch.setattr(bus_module, "_event_bus", None)
    bus_module.init_event_bus(manager)
    monkeypatch.setattr(
        executor,
        "_hydrate_cached_apply_context",
        lambda context: hydration_threads.append(threading.get_ident()),
    )

    with pytest.raises(RuntimeError, match="no output USD files"):
        await executor._execute_pipeline_inner(
            session_id=session_id,
            config_dict={"input": {"usd_path": "/tmp/in.usd"}},
            session_manager=manager,
            event_bus=bus_module.get_event_bus(),
            session_dir=session_dir,
            only_steps=None,
            skip_steps=None,
            create_texture_pipeline_workflow=_validation_factory,
        )

    assert len(hydration_threads) == 1
    assert hydration_threads[0] != event_loop_thread
