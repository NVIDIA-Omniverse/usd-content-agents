# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility tests for the deprecated async NVCF render module."""

from __future__ import annotations

from typing import Any

import pytest

from world_understanding.functions.graphics import (
    render_nvcf_async,
    render_remote_async,
)

_ONE_PIXEL_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="


def test_render_nvcf_async_exposes_remote_async_compatibility_api() -> None:
    for name in (
        "render_cameras_from_url",
        "render_composition_from_url",
        "get_global_nvcf_render_limit",
        "global_nvcf_render_slot",
        "_reset_global_nvcf_render_semaphore_for_tests",
    ):
        assert callable(getattr(render_nvcf_async, name))
        assert hasattr(render_remote_async, name)


@pytest.mark.asyncio
async def test_render_nvcf_async_monkeypatches_request_helper(
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
        render_nvcf_async,
        "execute_nvcf_request_async",
        fake_execute_nvcf_request_async,
    )

    result = await render_nvcf_async.render_cameras_from_url(
        usd_url="https://example.com/scene.usda",
        cameras=["/Camera"],
        api_key="test-api-key",
        base_url="https://example.com",
    )

    assert len(calls) == 1
    assert result["successful_cameras"] == 1
