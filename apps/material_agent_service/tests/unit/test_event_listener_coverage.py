# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for FastAPIEventListener event mapping and helper paths."""

from __future__ import annotations

import asyncio
import builtins
import json
from pathlib import Path

import pytest
from PIL import Image

from ...service.events import listener as listener_module
from ...service.events.listener import FastAPIEventListener
from ...service.runtime.events import ProgressEvent, StepState


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []
        self.claims: list[object | None] = []

    async def emit_for_owner(
        self,
        event: ProgressEvent,
        *,
        regeneration_claim: object | None,
    ) -> bool:
        self.events.append(event)
        self.claims.append(regeneration_claim)
        return True


class _FakeLoop:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ProgressEvent]] = []

    def call_soon_threadsafe(self, callback, event: ProgressEvent) -> None:
        self.calls.append((callback, event))


@pytest.fixture
def fake_bus(monkeypatch: pytest.MonkeyPatch) -> _FakeBus:
    bus = _FakeBus()
    monkeypatch.setattr(listener_module, "get_event_bus", lambda: bus)
    return bus


def _listener(
    fake_bus: _FakeBus,
    *,
    session_dir: Path | None = None,
    loop: object | None = None,
    icons: dict[str, str] | None = None,
) -> FastAPIEventListener:
    return FastAPIEventListener(
        "session-12345678",
        session_dir=session_dir,
        loop=loop,  # type: ignore[arg-type]
        session_material_icons=icons,
    )


@pytest.mark.unit
def test_listener_init_logging_and_event_emit_paths(
    fake_bus: _FakeBus,
    caplog: pytest.LogCaptureFixture,
) -> None:
    listener = _listener(fake_bus)
    assert listener.loop is None

    listener.info("info")
    listener.debug("debug")
    listener.warning("warning")
    listener.error("error")

    loop = _FakeLoop()
    listener.loop = loop  # type: ignore[assignment]
    listener.event("step.started", {"step_name": "predict"})
    assert listener.current_step == "predict"
    assert len(loop.calls) == 1
    assert loop.calls[0][1].step == "predict"
    listener.event("step.cancelled", {"step_name": "predict"})
    listener.event("task.cancelled", {"task_name": "VLMInference"})
    listener.event("workflow.cancelled", {"step_name": "predict"})
    assert len(loop.calls) == 1

    listener.loop = None
    with caplog.at_level("WARNING"):
        listener._emit_event_threadsafe(
            ProgressEvent(
                session_id="session-12345678",
                step="predict",
                state=StepState.RUNNING,
            )
        )
    assert "No event loop found" in caplog.text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_listener_async_loop_and_emit_sync(fake_bus: _FakeBus) -> None:
    listener = _listener(fake_bus)
    event = ProgressEvent(
        session_id="session-12345678",
        step="predict",
        state=StepState.RUNNING,
    )

    listener._emit_event_threadsafe(event)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert fake_bus.events == [event]

    listener._emit_event_sync(event)
    await asyncio.sleep(0)
    assert fake_bus.events == [event, event]
    assert fake_bus.claims == [None, None]


@pytest.mark.unit
def test_listener_maps_lifecycle_and_workflow_events(fake_bus: _FakeBus) -> None:
    listener = _listener(fake_bus)

    started = listener._map_event_to_progress("step.started", {"step_name": "apply"})
    assert started is not None
    assert started.state == StepState.RUNNING
    assert started.message == "Starting apply"

    assert (
        listener._map_event_to_progress("task.started", {"task_name": "VLMInference"})
        is None
    )
    task_started = listener._map_event_to_progress(
        "task.started", {"task_name": "TextureTask"}
    )
    assert task_started is not None
    assert task_started.step == "TextureTask"

    completed = listener._map_event_to_progress(
        "task.completed", {"task_name": "TextureTask"}
    )
    assert completed is not None
    assert completed.state == StepState.COMPLETED

    failed = listener._map_event_to_progress(
        "step.failed", {"step_name": "apply", "error": "bad material"}
    )
    assert failed is not None
    assert failed.state == StepState.FAILED
    assert failed.message == "bad material"

    assert (
        listener._map_event_to_progress("task.cancelled", {"task_name": "TextureTask"})
        is None
    )
    assert (
        listener._map_event_to_progress("step.cancelled", {"step_name": "apply"})
        is None
    )

    workflow_done = listener._map_event_to_progress("workflow.completed", {})
    assert workflow_done is None

    listener.current_step = "predict"
    workflow_failed = listener._map_event_to_progress("workflow.failed", {})
    assert workflow_failed is not None
    assert workflow_failed.step == "predict"

    assert (
        listener._map_event_to_progress(
            "workflow.cancelled", {"step_name": "scene_collect"}
        )
        is None
    )

    assert listener._map_event_to_progress("workflow.started", {}) is None
    assert listener._map_event_to_progress("unknown.event", {}) is None


@pytest.mark.unit
def test_listener_bounds_scene_sized_completion_evidence(fake_bus: _FakeBus) -> None:
    listener = _listener(fake_bus)
    original = {
        "step_name": "apply",
        "outputs": {
            "restore_stats": {
                "mapping_complete": True,
                "restored_prim_sources": {"/Root/A": "/Optimized/A"},
                "uncovered_originals": ["/Root/B"],
                "unconsumed_predictions": ["/Optimized/C"],
                "mapping_warnings": ["duplicate /Root/A"],
            },
            "assignment_stats": {
                "total_prims": 2,
                "bound_prim_ids": ["/Root/A"],
                "unbound_prim_ids": ["/Root/B"],
            },
        },
    }

    completed = listener._map_event_to_progress("step.completed", original)

    assert completed is not None
    assert completed.extra == {
        "step_name": "apply",
        "outputs": {
            "restore_stats": {
                "mapping_complete": True,
                "restored_prim_source_count": 1,
                "uncovered_original_count": 1,
                "unconsumed_prediction_count": 1,
                "mapping_warning_count": 1,
            },
            "assignment_stats": {
                "total_prims": 2,
                "bound_prim_count": 1,
                "unbound_prim_count": 1,
            },
        },
    }
    assert "restored_prim_sources" in original["outputs"]["restore_stats"]
    unchanged = listener._map_event_to_progress(
        "task.completed",
        {"task_name": "TextureTask", "outputs": "not-a-mapping"},
    )
    assert unchanged is not None
    assert unchanged.extra == {
        "task_name": "TextureTask",
        "outputs": "not-a-mapping",
    }


@pytest.mark.unit
def test_listener_maps_progress_prediction_and_rendering_events(
    fake_bus: _FakeBus,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    dataset_dir = session_dir / "cache" / "dataset"
    renders_dir = dataset_dir / "usd" / "renders" / "mesh"
    renders_dir.mkdir(parents=True)
    render_image = renders_dir / "mesh_I3_prim_only.png"
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(render_image)
    dataset_file = dataset_dir / "dataset.jsonl"
    dataset_file.write_text(
        json.dumps(
            {
                "id": "entry-1",
                "images": [
                    "usd/renders/mesh/reference.png",
                    "usd/renders/mesh/mesh_I3_prim_only.png",
                ],
            }
        )
        + "\n"
    )

    listener = _listener(
        fake_bus,
        session_dir=session_dir,
        icons={"Copper": "icons/copper.png"},
    )

    assert (
        listener._map_event_to_progress(
            "step.progress", {"step_name": "predict", "percent": 1}
        )
        is None
    )
    assert (
        listener._map_event_to_progress(
            "task.progress",
            {"task_name": "USDPrimTraversalAndRendering", "current": 0, "total": 10},
        )
        is None
    )

    render_progress = listener._map_event_to_progress(
        "task.progress",
        {
            "task_name": "USDPrimTraversalAndRendering",
            "current": 1,
            "total": 2,
            "percentage": "50",
            "message": "rendering",
        },
    )
    assert render_progress is not None
    assert render_progress.step == "build_dataset_usd"
    assert render_progress.percent == 50
    assert render_progress.extra["rendered_images"]

    prediction = listener._map_event_to_progress(
        "prediction.completed",
        {
            "step_name": "predict",
            "entry_id": "entry-1",
            "material": "Copper",
            "response_snippet": "<reasoning>because copper</reasoning>",
        },
    )
    assert prediction is not None
    assert prediction.message == "Predicted Event: Copper"
    assert prediction.extra["material_icon"].endswith("/materials/icon/Copper")
    assert prediction.extra["preview_image"].endswith("mesh_I3_prim_only.png")
    assert prediction.extra["reasoning"] == "because copper"

    no_reasoning = listener._map_event_to_progress(
        "prediction.completed",
        {
            "step_name": "predict",
            "entry_id": "missing-entry",
            "response_snippet": "plain text",
        },
    )
    assert no_reasoning is not None
    assert "reasoning" not in (no_reasoning.extra or {})

    assert (
        listener._map_event_to_progress(
            "rendering.progress", {"current": 0, "total": 2}
        )
        is None
    )
    assert listener._map_event_to_progress("rendering.progress", {"current": 1}) is None

    extra_image = renders_dir / "new_for_progress.png"
    Image.new("RGB", (8, 8), color=(99, 1, 2)).save(extra_image)
    rendering = listener._map_event_to_progress(
        "rendering.progress", {"current": 2, "total": 4, "percent": 60}
    )
    assert rendering is not None
    assert rendering.step == "build_dataset_usd"
    assert rendering.extra["rendered_images"]

    completed_image = renders_dir / "new_for_completed.png"
    Image.new("RGB", (8, 8), color=(2, 99, 1)).save(completed_image)
    assert listener._map_event_to_progress("rendering.completed", {}) is None
    all_done_image = renders_dir / "new_for_all_done.png"
    Image.new("RGB", (8, 8), color=(1, 2, 99)).save(all_done_image)
    all_done = listener._map_event_to_progress(
        "rendering.all_completed", {"total_prims": 3, "total_images": 6}
    )
    assert all_done is not None
    assert all_done.state == StepState.COMPLETED
    assert all_done.extra["rendered_images"]

    original_import = builtins.__import__

    def fail_config_import(name, *args, **kwargs):
        if name.endswith(".config"):
            raise RuntimeError("icon boom")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_config_import)
    assert listener._get_material_icon("Broken") is None


@pytest.mark.unit
def test_listener_global_icon_and_simple_thumbnail_fallback(
    fake_bus: _FakeBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _listener(fake_bus)

    from ...service import config as config_module

    monkeypatch.setattr(
        config_module.config,
        "material_icons",
        {"GlobalMaterial": "icons/global.png"},
    )
    assert listener._get_material_icon("GlobalMaterial") == (
        "/materials/icon/GlobalMaterial"
    )
    assert listener._get_thumbnail_filename("usd/renders/mesh/color.png").endswith(
        "color.png"
    )


@pytest.mark.unit
def test_listener_dataset_reasoning_and_thumbnail_helpers(
    fake_bus: _FakeBus,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    listener = _listener(fake_bus)
    assert listener._get_reasoning_from_predictions("entry") is None
    assert listener._scan_for_new_thumbnails("build_dataset_usd") == []
    listener.dataset_cache["already"] = {"id": "already"}
    listener._load_dataset_cache()
    assert listener.dataset_cache == {"already": {"id": "already"}}

    session_dir = tmp_path / "session"
    listener = _listener(fake_bus, session_dir=session_dir)
    assert listener._get_prim_image_from_dataset("missing") is None
    assert listener._get_reasoning_from_predictions("missing") is None
    assert listener._scan_for_new_thumbnails("predict") == []
    assert listener._scan_for_new_thumbnails("build_dataset_usd") == []

    dataset_dir = session_dir / "cache" / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "dataset.jsonl").write_text(
        "\n"
        + json.dumps({"id": "entry-1", "images": ["reference.png", "color.png"]})
        + "\n"
        + json.dumps({"id": "entry-2", "images": ["reference.png"]})
        + "\n"
    )
    assert listener._get_prim_image_from_dataset("entry-1") == "color.png"
    assert listener._get_prim_image_from_dataset("entry-2") is None

    predictions_dir = session_dir / "cache" / "predictions"
    predictions_dir.mkdir(parents=True)
    (predictions_dir / "predictions.jsonl").write_text(
        "\n"
        + json.dumps(
            {
                "id": "entry-1",
                "materials": {
                    "original_response": "<reasoning>full reasoning</reasoning>"
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "entry-2",
                "vlm_response": {"original_response": "<reasoning>unterminated"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "entry-3",
                "vlm_response": {"original_response": "plain"},
            }
        )
        + "\n"
    )
    assert listener._get_reasoning_from_predictions("entry-1") == "full reasoning"
    assert listener._get_reasoning_from_predictions("entry-2") is None
    assert listener._get_reasoning_from_predictions("entry-3") is None

    preview_dir = session_dir / "cache" / "preview"
    preview_dir.mkdir(parents=True)
    assert listener._get_thumbnail_filename("usd/renders/mesh/color.png").endswith(
        "color.png"
    )

    monkeypatch.setattr(
        listener_module,
        "normalize_render_image_path",
        lambda _path: (_ for _ in ()).throw(RuntimeError("bad path")),
    )
    assert listener._get_thumbnail_filename("bad") is None

    broken_dataset = dataset_dir / "dataset.jsonl"
    broken_dataset.write_text("{bad json")
    listener.dataset_cache = {}
    listener._load_dataset_cache()
    assert listener.dataset_cache == {}

    broken_predictions = predictions_dir / "predictions.jsonl"
    broken_predictions.write_text("{bad json")
    assert listener._get_reasoning_from_predictions("entry-1") is None


@pytest.mark.unit
def test_listener_scan_thumbnail_existing_and_invalid_files(
    fake_bus: _FakeBus,
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    renders_dir = session_dir / "cache" / "dataset" / "usd" / "renders"
    preview_dir = session_dir / "cache" / "preview"
    renders_dir.mkdir(parents=True)
    preview_dir.mkdir(parents=True)

    existing_image = renders_dir / "existing.png"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(existing_image)
    existing_preview = listener_module.resolve_preview_filename(
        preview_dir, "existing.png"
    )
    (preview_dir / existing_preview).write_bytes(b"already here")

    invalid_image = renders_dir / "invalid.png"
    invalid_image.write_text("not an image")

    listener = _listener(fake_bus, session_dir=session_dir)
    assert listener._scan_for_new_thumbnails("build_dataset_usd") == []
    assert existing_preview in listener.thumbnailed_images

    listener.thumbnailed_images.add(existing_preview)
    assert listener._scan_for_new_thumbnails("build_dataset_usd") == []
