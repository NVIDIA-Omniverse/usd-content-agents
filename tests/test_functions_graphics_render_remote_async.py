# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for async remote rendering response handling."""

from __future__ import annotations

import asyncio
import itertools
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from world_understanding.functions.graphics import render_remote_async
from world_understanding.functions.graphics.render_remote import RenderingStatus

_ONE_PIXEL_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="


def test_legacy_render_nvcf_async_exposes_remote_helpers() -> None:
    """Old async NVCF module imports should keep working during the rename window."""
    from world_understanding.functions.graphics import render_nvcf_async

    for name in (
        "get_global_nvcf_render_limit",
        "global_nvcf_render_slot",
        "_reset_global_nvcf_render_semaphore_for_tests",
    ):
        legacy_helper = getattr(render_nvcf_async, name)
        remote_helper = getattr(render_remote_async, name)

        assert callable(legacy_helper)
        assert legacy_helper.__name__ == remote_helper.__name__


def test_global_remote_render_limit_env_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", "not-an-int")
    assert render_remote_async.get_global_remote_render_limit() is None

    monkeypatch.setenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", "0")
    assert render_remote_async.get_global_remote_render_limit() is None

    monkeypatch.delenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.setenv("MA_RENDER_GLOBAL_MAX_CONCURRENT_REQUESTS", "2")
    assert render_remote_async.get_global_remote_render_limit() == 2


@pytest.mark.asyncio
async def test_render_cameras_retries_response_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_execute_nvcf_request_async(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 1:
            return {"status": "exception", "error": "renderer worker unavailable"}
        return {
            "status": "success",
            "images": {"0": {"/Camera": {"images": _ONE_PIXEL_PNG}}},
        }

    monkeypatch.setattr(
        render_remote_async,
        "execute_nvcf_request_async",
        fake_execute_nvcf_request_async,
    )

    result = await render_remote_async.render_cameras_from_url(
        usd_url="https://example.com/scene.usda",
        cameras=["/Camera"],
        api_key="test-api-key",
        base_url="https://example.com",
        max_retries=1,
        retry_delay=0.0,
    )

    assert len(calls) == 2
    assert result["successful_cameras"] == 1
    assert result["failed_cameras"] == 0
    assert result["results"][0]["status"] == RenderingStatus.success
    assert result["results"][0]["frame_count"] == 1


@pytest.mark.asyncio
async def test_render_cameras_applies_retry_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    sleeps: list[float] = []

    async def fake_execute_nvcf_request_async(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 1:
            return {"status": "exception", "error": "renderer worker unavailable"}
        return {
            "status": "success",
            "images": {"0": {"/Camera": {"images": _ONE_PIXEL_PNG}}},
        }

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        render_remote_async,
        "execute_nvcf_request_async",
        fake_execute_nvcf_request_async,
    )
    monkeypatch.setattr(render_remote_async.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(render_remote_async.random, "uniform", lambda _low, high: high)

    result = await render_remote_async.render_cameras_from_url(
        usd_url="https://example.com/scene.usda",
        cameras=["/Camera"],
        api_key="test-api-key",
        base_url="https://example.com",
        max_retries=1,
        retry_delay=2.0,
        retry_jitter=0.25,
    )

    assert calls[0]["retry_jitter"] == 0.25
    assert sleeps == [2.5]
    assert result["successful_cameras"] == 1


@pytest.mark.asyncio
async def test_render_cameras_frame_range_sensors_mapping_and_missing_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_execute_nvcf_request_async(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "status": "success",
            "images": {
                "1": {
                    "Camera": {"images": "img-a", "linear_depth": "depth-a"},
                    "/Other/Named": {"images": "img-b", "linear_depth": "depth-b"},
                },
                "2": {
                    "Camera": {"images": "img-c", "linear_depth": "depth-c"},
                },
            },
        }

    monkeypatch.setattr(
        render_remote_async,
        "execute_nvcf_request_async",
        fake_execute_nvcf_request_async,
    )
    monkeypatch.setattr(
        render_remote_async,
        "base64_to_image",
        lambda value: Image.new("RGB", (1, 1), "red"),
    )
    monkeypatch.setattr(
        render_remote_async,
        "base64_to_numpy",
        lambda value, dtype: np.array([1], dtype=dtype),
    )

    result = await render_remote_async.render_cameras_from_url(
        usd_url="https://example.com/scene.usda",
        cameras=["/Camera", "/Rig/Named", "/Missing"],
        api_key="test-api-key",
        base_url="https://example.com",
        frames="1:2",
        sensors=["linear_depth"],
        semaphore=asyncio.Semaphore(1),
    )

    assert calls[0]["params"]["render_settings"]["frame_range"] == {
        "start": 1,
        "end": 2,
    }
    assert result["successful_cameras"] == 2
    assert result["failed_cameras"] == 1
    assert result["results"][0]["frame_count"] == 2
    assert set(result["results"][0]["sensors"]["linear_depth"]) == {1, 2}
    assert result["results"][1]["camera"] == "/Rig/Named"
    assert result["results"][2]["status"] == RenderingStatus.exception


@pytest.mark.asyncio
async def test_render_cameras_v2_conversion_and_no_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute_nvcf_request_async(**kwargs: Any) -> dict[str, Any]:
        return {"schema": "v2"}

    monkeypatch.setattr(
        render_remote_async,
        "execute_nvcf_request_async",
        fake_execute_nvcf_request_async,
    )
    monkeypatch.setattr(render_remote_async, "_is_v2_response", lambda result: True)
    monkeypatch.setattr(
        render_remote_async,
        "_convert_v2_to_v1",
        lambda result: {
            "status": "success",
            "images": {"0": {"/Camera": {"images": _ONE_PIXEL_PNG}}},
        },
    )

    converted = await render_remote_async.render_cameras_from_url(
        usd_url="https://example.com/scene.usda",
        cameras=["/Camera"],
        api_key="test-api-key",
        base_url="https://example.com",
    )
    assert converted["successful_cameras"] == 1

    monkeypatch.setattr(render_remote_async, "_is_v2_response", lambda result: False)
    no_response = await render_remote_async.render_cameras_from_url(
        usd_url="https://example.com/scene.usda",
        cameras=["/Camera"],
        api_key="test-api-key",
        base_url="https://example.com",
        max_retries=-1,
    )
    assert no_response["failed_cameras"] == 1
    assert "without a response" in no_response["results"][0]["error"]


@pytest.mark.asyncio
async def test_render_cameras_logs_queue_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGlobalSemaphore:
        def release(self) -> None:
            pass

    async def fake_execute_nvcf_request_async(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "success",
            "images": {"0": {"/Camera": {"images": _ONE_PIXEL_PNG}}},
        }

    async def fake_acquire_threading_semaphore(
        semaphore: threading.BoundedSemaphore,
    ) -> None:
        return None

    times = itertools.chain([0.0, 0.0, 0.10, 0.10, 0.20, 0.30], itertools.repeat(0.30))
    monkeypatch.setattr(render_remote_async.time, "time", lambda: next(times))
    monkeypatch.setattr(
        render_remote_async,
        "_get_global_nvcf_render_semaphore",
        lambda: FakeGlobalSemaphore(),
    )
    monkeypatch.setattr(
        render_remote_async,
        "_acquire_threading_semaphore",
        fake_acquire_threading_semaphore,
    )
    monkeypatch.setattr(
        render_remote_async,
        "execute_nvcf_request_async",
        fake_execute_nvcf_request_async,
    )

    result = await render_remote_async.render_cameras_from_url(
        usd_url="https://example.com/scene.usda",
        cameras=["/Camera"],
        api_key="test-api-key",
        base_url="https://example.com",
    )

    assert result["successful_cameras"] == 1


@pytest.mark.asyncio
async def test_render_cameras_does_not_retry_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_execute_nvcf_request_async(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "load_error", "error": "invalid USD"}

    monkeypatch.setattr(
        render_remote_async,
        "execute_nvcf_request_async",
        fake_execute_nvcf_request_async,
    )

    result = await render_remote_async.render_cameras_from_url(
        usd_url="https://example.com/bad.usda",
        cameras=["/Camera"],
        api_key="test-api-key",
        base_url="https://example.com",
        max_retries=3,
        retry_delay=0.0,
    )

    assert len(calls) == 1
    assert result["successful_cameras"] == 0
    assert result["failed_cameras"] == 1
    assert result["results"][0]["status"] == RenderingStatus.load_error
    assert "invalid USD" in result["results"][0]["error"]


@pytest.mark.asyncio
async def test_render_composition_single_and_multi_camera_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []

    async def fake_render_cameras_from_url(
        usd_url: str,
        cameras: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((usd_url, tuple(cameras)))
        return {
            "total_cameras": len(cameras),
            "successful_cameras": len(cameras),
            "failed_cameras": 0,
            "total_render_time": float(len(cameras)),
            "results": [
                {"camera": camera, "status": RenderingStatus.success}
                for camera in cameras
            ],
        }

    monkeypatch.setattr(
        render_remote_async,
        "render_cameras_from_url",
        fake_render_cameras_from_url,
    )

    highlight, plain = await render_remote_async.render_composition_from_url(
        highlight_url="https://example.com/highlight.usda",
        plain_url="https://example.com/plain.usda",
        cameras=["/CamA", "/CamB"],
        api_key="test-api-key",
        base_url="https://example.com",
    )

    assert calls[:4] == [
        ("https://example.com/highlight.usda", ("/CamA",)),
        ("https://example.com/plain.usda", ("/CamA",)),
        ("https://example.com/highlight.usda", ("/CamB",)),
        ("https://example.com/plain.usda", ("/CamB",)),
    ]
    assert highlight["successful_cameras"] == 2
    assert plain["successful_cameras"] == 2
    assert len(highlight["results"]) == 2

    calls.clear()
    (
        highlight_multi,
        plain_multi,
    ) = await render_remote_async.render_composition_from_url(
        highlight_url="https://example.com/highlight.usda",
        plain_url="https://example.com/plain.usda",
        cameras=["/CamA", "/CamB"],
        api_key="test-api-key",
        base_url="https://example.com",
        single_camera_per_request=False,
    )

    assert calls == [
        ("https://example.com/highlight.usda", ("/CamA", "/CamB")),
        ("https://example.com/plain.usda", ("/CamA", "/CamB")),
    ]
    assert highlight_multi["successful_cameras"] == 2
    assert plain_multi["successful_cameras"] == 2


@pytest.mark.asyncio
async def test_render_cameras_forwards_material_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_execute_nvcf_request_async(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "status": "success",
            "images": {"0": {"/Camera": {"images": _ONE_PIXEL_PNG}}},
        }

    monkeypatch.setattr(
        render_remote_async,
        "execute_nvcf_request_async",
        fake_execute_nvcf_request_async,
    )

    result = await render_remote_async.render_cameras_from_url(
        usd_url="https://example.com/scene.usda",
        cameras=["/Camera"],
        api_key="test-api-key",
        base_url="https://example.com",
        material_target="openpbr_materialx",
    )

    assert result["successful_cameras"] == 1
    assert calls[0]["params"]["render_settings"]["material_target"] == (
        "openpbr_materialx"
    )


@pytest.mark.asyncio
async def test_global_remote_render_limit_serializes_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_requests = 0
    max_active_requests = 0
    calls = 0

    async def fake_execute_nvcf_request_async(**kwargs: Any) -> dict[str, Any]:
        nonlocal active_requests, max_active_requests, calls
        calls += 1
        active_requests += 1
        max_active_requests = max(max_active_requests, active_requests)
        await asyncio.sleep(0.01)
        active_requests -= 1
        return {
            "status": "success",
            "images": {"0": {"/Camera": {"images": _ONE_PIXEL_PNG}}},
        }

    monkeypatch.setenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", "1")
    render_remote_async._reset_global_remote_render_semaphore_for_tests()
    monkeypatch.setattr(
        render_remote_async,
        "execute_nvcf_request_async",
        fake_execute_nvcf_request_async,
    )

    try:
        results = await asyncio.gather(
            render_remote_async.render_cameras_from_url(
                usd_url="https://example.com/scene-a.usda",
                cameras=["/Camera"],
                api_key="test-api-key",
                base_url="https://example.com",
            ),
            render_remote_async.render_cameras_from_url(
                usd_url="https://example.com/scene-b.usda",
                cameras=["/Camera"],
                api_key="test-api-key",
                base_url="https://example.com",
            ),
        )
    finally:
        render_remote_async._reset_global_remote_render_semaphore_for_tests()

    assert calls == 2
    assert max_active_requests == 1
    assert [result["successful_cameras"] for result in results] == [1, 1]


def test_save_images_parallel_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert render_remote_async.save_images_parallel([]) == 0

    output_path = tmp_path / "nested" / "image.png"
    assert (
        render_remote_async.save_images_parallel(
            [(Image.new("RGB", (1, 1), "blue"), output_path)],
            max_workers=1,
        )
        == 1
    )
    assert output_path.exists()

    class BadImage:
        def save(self, path: Path) -> None:
            raise OSError("cannot save")

    assert (
        render_remote_async.save_images_parallel(
            [(BadImage(), tmp_path / "bad.png")],
            max_workers=1,
        )
        == 0
    )

    class RaisingFuture:
        def result(self) -> bool:
            raise RuntimeError("future failed")

    class FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def submit(self, func, task):
            return RaisingFuture()

    monkeypatch.setattr(render_remote_async, "ThreadPoolExecutor", FakeExecutor)
    assert (
        render_remote_async.save_images_parallel(
            [(Image.new("RGB", (1, 1), "green"), tmp_path / "future.png")],
            max_workers=1,
        )
        == 0
    )


def test_global_remote_render_slot_without_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.delenv("MA_RENDER_GLOBAL_MAX_CONCURRENT_REQUESTS", raising=False)
    render_remote_async._reset_global_remote_render_semaphore_for_tests()

    with render_remote_async.global_remote_render_slot() as waited:
        assert waited == 0.0


def test_global_remote_render_slot_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", "1")
    render_remote_async._reset_global_remote_render_semaphore_for_tests()

    try:
        with render_remote_async.global_remote_render_slot():
            with pytest.raises(
                render_remote_async.RemoteRenderingSlotTimeoutError,
                match="global remote render slot",
            ):
                with render_remote_async.global_remote_render_slot(
                    timeout_seconds=0.001,
                ):
                    pass
    finally:
        render_remote_async._reset_global_remote_render_semaphore_for_tests()
