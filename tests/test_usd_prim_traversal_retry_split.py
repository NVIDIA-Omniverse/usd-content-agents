# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Any

import pytest

from world_understanding.agentic.usd_tasks.prim_traversal import (
    USDPrimTraversalAndRenderingTask,
)
from world_understanding.functions.graphics.rendering import RemoteRenderingBackend


class _Listener:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, **_: Any) -> None:
        self.warnings.append(message)


class _SplitTask(USDPrimTraversalAndRenderingTask):
    def _process_batch(
        self,
        batch_start: int,
        batch_end: int,
        prims_to_render: list[str],
        prim_data: dict[str, dict[str, Any]],
        prepared_stages: dict[str, dict[str, Any]],
        rendering_backend: Any,
        render_mode: str,
        render_output_dir: Path,
        output_dir: Path,
        num_total_tasks: int,
        batch_size: int,
        listener: Any,
        sensor_modes: list[str] | None = None,
        image_height: int = 512,
    ) -> tuple[int, list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        if batch_end - batch_start > 1:
            return 0, [], {}
        prim_path = prims_to_render[batch_start]
        return 1, [], {prim_path: [{"path": f"{prim_path}.png"}]}

    async def _process_batch_async(
        self,
        batch_start: int,
        batch_end: int,
        prims_to_render: list[str],
        prim_data: dict[str, dict[str, Any]],
        prepared_stages: dict[str, dict[str, Any]],
        rendering_backend: Any,
        render_mode: str,
        render_output_dir: Path,
        output_dir: Path,
        num_total_tasks: int,
        batch_size: int,
        listener: Any,
        sensor_modes: list[str] | None = None,
        image_height: int = 512,
        semaphore: Any = None,
    ) -> tuple[
        int, int, str, int, list[dict[str, Any]], dict[str, list[dict[str, Any]]]
    ]:
        if batch_end - batch_start > 1:
            return batch_start, batch_end, render_mode, 0, [], {}
        prim_path = prims_to_render[batch_start]
        return (
            batch_start,
            batch_end,
            render_mode,
            1,
            [],
            {prim_path: [{"path": f"{prim_path}.png"}]},
        )


def _remote_backend() -> RemoteRenderingBackend:
    return RemoteRenderingBackend.__new__(RemoteRenderingBackend)


def test_process_batch_with_retry_split_recovers_zero_image_batches() -> None:
    task = _SplitTask()
    listener = _Listener()
    prims = ["/A", "/B", "/C", "/D"]

    total_images, failures, prim_images = task._process_batch_with_retry_split(
        0,
        len(prims),
        prims,
        {p: {} for p in prims},
        {},
        _remote_backend(),
        "prim_only",
        Path("renders"),
        Path("out"),
        1,
        len(prims),
        listener,
    )

    assert total_images == len(prims)
    assert failures == []
    assert set(prim_images) == set(prims)
    assert listener.warnings


def test_process_batch_with_retry_split_does_not_retry_local_zero_images() -> None:
    task = _SplitTask()
    listener = _Listener()
    prims = ["/A", "/B"]

    total_images, failures, prim_images = task._process_batch_with_retry_split(
        0,
        len(prims),
        prims,
        {p: {} for p in prims},
        {},
        object(),
        "prim_only",
        Path("renders"),
        Path("out"),
        1,
        len(prims),
        listener,
    )

    assert total_images == 0
    assert failures == []
    assert prim_images == {}
    assert listener.warnings == []


@pytest.mark.asyncio
async def test_process_batch_async_with_retry_split_recovers_zero_image_batches() -> (
    None
):
    task = _SplitTask()
    listener = _Listener()
    prims = ["/A", "/B", "/C", "/D"]

    result = await task._process_batch_async_with_retry_split(
        0,
        len(prims),
        prims,
        {p: {} for p in prims},
        {},
        _remote_backend(),
        "prim_only",
        Path("renders"),
        Path("out"),
        1,
        len(prims),
        listener,
    )

    assert result[:4] == (0, len(prims), "prim_only", len(prims))
    assert result[4] == []
    assert set(result[5]) == set(prims)
    assert listener.warnings


@pytest.mark.asyncio
async def test_process_batch_async_with_retry_split_does_not_retry_local_zero_images() -> (
    None
):
    task = _SplitTask()
    listener = _Listener()
    prims = ["/A", "/B"]

    result = await task._process_batch_async_with_retry_split(
        0,
        len(prims),
        prims,
        {p: {} for p in prims},
        {},
        object(),
        "prim_only",
        Path("renders"),
        Path("out"),
        1,
        len(prims),
        listener,
    )

    assert result[:4] == (0, len(prims), "prim_only", 0)
    assert result[4] == []
    assert result[5] == {}
    assert listener.warnings == []
