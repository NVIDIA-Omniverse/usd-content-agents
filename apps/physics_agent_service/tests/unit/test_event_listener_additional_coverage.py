# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from PIL import Image

from ...service.events.listener import FastAPIEventListener
from ...service.runtime.events import StepState


def test_event_listener_logging_methods_do_not_raise() -> None:
    listener = FastAPIEventListener("session-1234", loop=None)
    listener.info("info")
    listener.debug("debug")
    listener.warning("warning")
    listener.error("error")


def test_event_listener_maps_lifecycle_and_workflow_events(tmp_path: Path) -> None:
    listener = FastAPIEventListener("session-1234", session_dir=tmp_path, loop=None)

    vlm_step = listener._map_event_to_progress(
        "step.started",
        {"step_name": "VLMInference"},
    )
    assert vlm_step is not None
    assert vlm_step.step == "predict"

    task_event = listener._map_event_to_progress(
        "task.started",
        {"task_name": "CustomTask", "message": "go"},
    )
    assert task_event is not None
    assert task_event.step == "CustomTask"
    assert task_event.state == StepState.RUNNING

    assert (
        listener._map_event_to_progress("task.started", {"task_name": "VLMInference"})
        is None
    )

    completed = listener._map_event_to_progress(
        "task.completed", {"task_name": "CustomTask"}
    )
    assert completed is not None
    assert completed.state == StepState.COMPLETED

    failed = listener._map_event_to_progress(
        "step.failed", {"step_name": "predict", "error": "boom"}
    )
    assert failed is not None
    assert failed.state == StepState.FAILED
    assert failed.message == "boom"

    workflow_completed = listener._map_event_to_progress(
        "workflow.completed", {"count": 2}
    )
    assert workflow_completed is not None
    assert workflow_completed.extra["pipeline_completed"] is True

    workflow_failed = listener._map_event_to_progress(
        "workflow.failed", {"message": "bad"}
    )
    assert workflow_failed is not None
    assert workflow_failed.state == StepState.FAILED
    assert workflow_failed.message == "bad"

    assert listener._map_event_to_progress("workflow.started", {}) is None
    assert listener._map_event_to_progress("unknown.event", {}) is None


def test_event_listener_can_suppress_failure_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = FastAPIEventListener(
        "session-1234",
        loop=None,
        suppress_failure_events=True,
    )
    emitted = []
    monkeypatch.setattr(listener, "_emit_event_threadsafe", emitted.append)

    listener.event("step.started", {"step_name": "predict"})
    listener.event("step.failed", {"step_name": "predict", "error": "sensitive"})
    listener.event("workflow.failed", {"error": "sensitive"})

    assert len(emitted) == 1
    assert emitted[0].state == StepState.RUNNING

    listener.event("task.started", {"task_name": "VLMInference"})
    assert listener.canonical_current_step == "predict"


def test_event_listener_progress_filters_and_reasoning(tmp_path: Path) -> None:
    listener = FastAPIEventListener("session-1234", session_dir=tmp_path, loop=None)

    assert (
        listener._map_event_to_progress(
            "task.progress", {"task_name": "VLMInference", "current": 1, "total": 2}
        )
        is None
    )
    assert (
        listener._map_event_to_progress(
            "step.progress",
            {"step_name": "build_dataset_usd", "current": 0, "total": 3},
        )
        is None
    )

    event = listener._map_event_to_progress(
        "prediction.completed",
        {
            "step_name": "predict",
            "current": 1,
            "total": 2,
            "percentage": "75",
            "entry_id": "/World/Cube",
            "response_snippet": "<reasoning>looks heavy</reasoning><answer>x</answer>",
            "classification": "rigid_body",
        },
    )
    assert event is not None
    assert event.step == "predict"
    assert event.percent == 75
    assert event.extra["classification"] == "rigid_body"
    assert event.extra["reasoning"] == "looks heavy"
    assert event.extra["response_snippet"].startswith("<reasoning>")


def test_event_listener_progress_adds_rendered_and_preview_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = FastAPIEventListener("session-1234", session_dir=tmp_path, loop=None)
    renders = tmp_path / "cache" / "dataset" / "usd" / "renders"
    cube_dir = renders / "cube"
    cube_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(0, 255, 0)).save(cube_dir / "step.png")

    step_event = listener._map_event_to_progress(
        "step.progress",
        {
            "step_name": "build_dataset_usd",
            "current": 1,
            "total": 2,
            "percent": 50,
        },
    )
    assert step_event is not None
    assert step_event.extra["rendered_images"]

    listener = FastAPIEventListener("session-1234", session_dir=tmp_path, loop=None)
    Image.new("RGB", (16, 16), color=(0, 0, 255)).save(cube_dir / "rendering.png")
    rendering_event = listener._map_event_to_progress(
        "rendering.progress",
        {"current": 1, "total": 2, "percent": 50},
    )
    assert rendering_event is not None
    assert rendering_event.extra["rendered_images"]

    listener = FastAPIEventListener("session-1234", session_dir=tmp_path, loop=None)
    Image.new("RGB", (16, 16), color=(255, 255, 0)).save(cube_dir / "done.png")
    completed = listener._map_event_to_progress(
        "rendering.all_completed",
        {"total_prims": 1, "total_images": 3},
    )
    assert completed is not None
    assert completed.extra["rendered_images"]

    monkeypatch.setattr(
        listener, "_get_prim_image_from_dataset", lambda _entry: "view.png"
    )
    monkeypatch.setattr(listener, "_get_thumbnail_filename", lambda _image: "thumb.png")
    prediction = listener._map_event_to_progress(
        "prediction.completed",
        {"step_name": "predict", "entry_id": "/a", "current": 1, "total": 1},
    )
    assert prediction is not None
    assert prediction.extra["preview_image"] == "thumb.png"


def test_event_listener_dataset_lookup_and_thumbnail_fallbacks(tmp_path: Path) -> None:
    listener = FastAPIEventListener("session-1234", session_dir=tmp_path, loop=None)
    dataset_dir = tmp_path / "cache" / "dataset"
    dataset_dir.mkdir(parents=True)
    dataset_path = dataset_dir / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "/a", "images": ["reference.png", "prim_only.png"]}),
                json.dumps({"id": "/b", "images": ["view.png"]}),
                json.dumps({"id": "/c", "images": []}),
            ]
        ),
        encoding="utf-8",
    )

    assert listener._get_prim_image_from_dataset("/a") == "prim_only.png"
    assert listener._get_prim_image_from_dataset("/b") == "view.png"
    assert listener._get_prim_image_from_dataset("/missing") is None
    assert listener._get_prim_image_from_dataset("/c") is None

    filename = listener._get_thumbnail_filename("prim_only.png")
    assert filename is not None
    assert filename.endswith(".png")

    no_dir = FastAPIEventListener("session-1234", loop=None)
    assert no_dir._get_prim_image_from_dataset("/a") is None
    assert no_dir._scan_for_new_thumbnails("predict") == []


def test_event_listener_load_dataset_cache_handles_bad_json(tmp_path: Path) -> None:
    listener = FastAPIEventListener("session-1234", session_dir=tmp_path, loop=None)
    dataset_dir = tmp_path / "cache" / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "dataset.jsonl").write_text("{bad json\n", encoding="utf-8")

    listener._load_dataset_cache()

    assert listener.dataset_cache == {}


def test_event_listener_scans_and_dedupes_thumbnails(tmp_path: Path) -> None:
    listener = FastAPIEventListener("session-1234", session_dir=tmp_path, loop=None)
    renders = tmp_path / "cache" / "dataset" / "usd" / "renders" / "cube"
    renders.mkdir(parents=True)
    img_path = renders / "view.png"
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(img_path)

    first = listener._scan_for_new_thumbnails("build_dataset_usd")
    second = listener._scan_for_new_thumbnails("build_dataset_usd")

    assert len(first) == 1
    assert second == []
    assert (tmp_path / "cache" / "preview" / first[0]).is_file()

    progress = listener._map_event_to_progress(
        "rendering.progress",
        {"current": 1, "total": 2, "percent": 50},
    )
    assert progress is not None
    assert progress.step == "build_dataset_usd"

    assert (
        listener._map_event_to_progress(
            "rendering.progress", {"current": 0, "total": 2}
        )
        is None
    )
    assert listener._map_event_to_progress("rendering.completed", {}) is None
    all_done = listener._map_event_to_progress(
        "rendering.all_completed", {"total_prims": 1, "total_images": 1}
    )
    assert all_done is not None
    assert all_done.state == StepState.COMPLETED

    preview_dir = tmp_path / "cache" / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    assert listener._get_thumbnail_filename("cube/view.png") is not None

    listener = FastAPIEventListener("session-1234", session_dir=tmp_path, loop=None)
    renders = tmp_path / "cache" / "dataset" / "usd" / "renders"
    existing = renders / "existing.png"
    Image.new("RGB", (16, 16), color=(0, 0, 0)).save(existing)
    unique = listener._get_thumbnail_filename("existing.png")
    assert unique is not None
    (preview_dir / unique).write_bytes(b"already there")
    assert listener._scan_for_new_thumbnails("build_dataset_usd") == []


def test_event_listener_scan_handles_thumbnail_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = FastAPIEventListener("session-1234", session_dir=tmp_path, loop=None)
    renders = tmp_path / "cache" / "dataset" / "usd" / "renders"
    renders.mkdir(parents=True)
    (renders / "broken.png").write_text("not an image", encoding="utf-8")
    assert listener._scan_for_new_thumbnails("build_dataset_usd") == []

    def bad_rglob(_self: Path, _pattern: str):
        raise OSError("scan failed")

    monkeypatch.setattr(Path, "rglob", bad_rglob)
    assert listener._scan_for_new_thumbnails("build_dataset_usd") == []


@pytest.mark.asyncio
async def test_event_listener_event_schedules_on_loop() -> None:
    loop = asyncio.get_running_loop()
    listener = FastAPIEventListener("session-1234", loop=loop)

    listener.event("step.started", {"step_name": "predict"})

    async def wait_for_snapshot() -> dict:
        for _ in range(100):
            snapshot = listener.event_bus.get_snapshot("session-1234")
            if snapshot is not None:
                return snapshot
            await asyncio.sleep(0.01)
        raise AssertionError("event snapshot did not update")

    snapshot = await asyncio.wait_for(wait_for_snapshot(), timeout=1.5)
    assert snapshot["current_step"]["name"] == "predict"


def test_event_listener_emit_without_loop_is_noop() -> None:
    listener = FastAPIEventListener("session-1234", loop=None)
    listener.loop = None
    listener._emit_event_threadsafe(
        listener._map_event_to_progress("step.started", {"step_name": "predict"})
    )
