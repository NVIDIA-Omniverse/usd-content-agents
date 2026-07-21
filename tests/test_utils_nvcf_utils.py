# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import random
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from world_understanding.utils import nvcf_utils
from world_understanding.utils.nvcf_utils import (
    execute_nvcf_request_async,
    get_base_url,
    is_service_base_url,
    resolve_endpoint_or_function_id,
)


def test_get_base_url_resolves_render_function_id_when_endpoint_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.setenv(
        "NVCF_RENDER_FUNCTION_ID",
        "12345678-1234-1234-1234-123456789abc",
    )

    assert (
        get_base_url(None, "RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID")
        == "https://12345678-1234-1234-1234-123456789abc.invocation.api.nvcf.nvidia.com"
    )


def test_get_base_url_prefers_endpoint_over_function_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.local:8001")
    monkeypatch.setenv(
        "NVCF_RENDER_FUNCTION_ID",
        "12345678-1234-1234-1234-123456789abc",
    )

    assert (
        get_base_url(None, "RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID")
        == "http://renderer.local:8001"
    )


def test_get_base_url_preserves_uppercase_http_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RENDER_ENDPOINT", "HTTPS://renderer.local:8001")
    monkeypatch.setenv(
        "NVCF_RENDER_FUNCTION_ID",
        "12345678-1234-1234-1234-123456789abc",
    )

    assert (
        get_base_url(None, "RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID")
        == "HTTPS://renderer.local:8001"
    )


def test_get_base_url_preserves_http_endpoint_when_urlsplit_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_urlsplit(value: str) -> object:
        raise ValueError(f"invalid URL for test: {value}")

    monkeypatch.setattr(nvcf_utils, "urlsplit", reject_urlsplit)

    assert (
        get_base_url(
            "https://renderer.local:8001",
            "RENDER_ENDPOINT",
            "NVCF_RENDER_FUNCTION_ID",
        )
        == "https://renderer.local:8001"
    )


def test_get_base_url_normalizes_schemeless_host_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RENDER_ENDPOINT", "renderer.local:8001")
    monkeypatch.setenv(
        "NVCF_RENDER_FUNCTION_ID",
        "12345678-1234-1234-1234-123456789abc",
    )

    assert (
        get_base_url(None, "RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID")
        == "http://renderer.local:8001"
    )


def test_get_base_url_accepts_url_in_function_id_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.setenv("NVCF_RENDER_FUNCTION_ID", "https://renderer.local:8001")

    assert (
        get_base_url(None, "RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID")
        == "https://renderer.local:8001"
    )


def test_get_base_url_accepts_explicit_function_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)

    assert (
        get_base_url(
            "12345678-1234-1234-1234-123456789abc",
            "RENDER_ENDPOINT",
            "NVCF_RENDER_FUNCTION_ID",
        )
        == "https://12345678-1234-1234-1234-123456789abc.invocation.api.nvcf.nvidia.com"
    )


def test_get_base_url_requires_endpoint_or_function_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)

    with pytest.raises(ValueError, match="RENDER_ENDPOINT"):
        get_base_url(None, "RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID")


def test_is_service_base_url_detects_service_urls() -> None:
    assert is_service_base_url("HTTPS://renderer.local:8001")
    assert is_service_base_url("renderer.local:8001")
    assert not is_service_base_url("12345678-1234-1234-1234-123456789abc")


def test_resolve_endpoint_or_function_id_expands_function_id() -> None:
    assert (
        resolve_endpoint_or_function_id("12345678-1234-1234-1234-123456789abc")
        == "https://12345678-1234-1234-1234-123456789abc.invocation.api.nvcf.nvidia.com"
    )


@pytest.mark.asyncio
async def test_execute_nvcf_request_async_applies_retry_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    request = httpx.Request("POST", "https://example.com/render")

    error_response = MagicMock()
    error_response.status_code = 503
    error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "server unavailable",
        request=request,
        response=httpx.Response(503, request=request),
    )

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.headers = {"content-type": "application/json"}
    success_response.json.return_value = {"status": "success"}
    success_response.raise_for_status.return_value = None

    class FakeAsyncClient:
        responses = [error_response, success_response]

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, *args: Any, **kwargs: Any):
            return self.responses.pop(0)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(random, "uniform", lambda _low, high: high)

    result = await execute_nvcf_request_async(
        url="https://example.com/render",
        headers={},
        params={},
        api_key="test-key",
        timeout=10,
        max_retries=1,
        retry_delay=2.0,
        retry_jitter=0.25,
    )

    assert sleeps == [2.5]
    assert result == {"status": "success"}
