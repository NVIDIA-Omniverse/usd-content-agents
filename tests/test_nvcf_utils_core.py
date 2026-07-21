# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Targeted coverage for NVCF utility edge paths."""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from types import SimpleNamespace
from typing import Any

import pytest
import requests

from world_understanding.utils import nvcf_utils


def _zip_response(payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("result.response", json.dumps(payload))
    return buffer.getvalue()


class PollResponse:
    def __init__(
        self,
        status_code: int,
        headers: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
        content: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload or {}
        self.content = content

    def json(self) -> dict[str, Any]:
        return self._payload


class PollClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def get(self, url: str, headers: dict[str, str]) -> PollResponse:
        self.calls.append({"url": url, "headers": headers})
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def test_api_key_and_url_resolution_edges(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    with caplog.at_level("WARNING"):
        assert nvcf_utils.get_nvcf_api_key() == ""
    assert "NGC_API_KEY is not set" in caplog.text
    assert nvcf_utils.get_nvcf_api_key("provided") == "provided"

    assert nvcf_utils.is_service_base_url("") is False
    assert nvcf_utils.is_service_base_url("localhost:8000") is True
    assert nvcf_utils.is_service_base_url("localhost:not-a-port") is False
    assert nvcf_utils.is_service_base_url("http://[bad") is True
    assert nvcf_utils._has_schemeless_service_location("[bad") is False
    assert nvcf_utils._has_schemeless_service_location(":8000") is False
    assert nvcf_utils.resolve_endpoint_or_function_id("localhost:8000") == (
        "http://localhost:8000"
    )
    assert nvcf_utils.s3_uri_to_https_url("s3://bucket/key/file.txt", "us-west-2") == (
        "https://bucket.s3.us-west-2.amazonaws.com/key/file.txt"
    )
    assert nvcf_utils.create_nvcf_headers("", 30) == {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    assert nvcf_utils.create_nvcf_headers("key", 30, poll_seconds=5) == {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer key",
        "NVCF-POLL-SECONDS": "5",
        "nvcf-feature-enable-gateway-timeout": "true",
    }


def test_parse_zip_response_success() -> None:
    assert nvcf_utils.parse_zip_response(_zip_response({"success": True})) == {
        "success": True
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("result.txt", "no response")
    assert nvcf_utils.parse_zip_response(buffer.getvalue()) is None


def test_poll_nvcf_status_edge_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    zip_status, zip_result = asyncio.run(
        nvcf_utils.poll_nvcf_status(
            PollClient(
                [PollResponse(200, {"content-type": "application/zip"}, content=b"bad")]
            ),
            "req-zip",
            api_key="",
            poll_seconds=5,
            timeout=10,
        )
    )
    assert (zip_status, zip_result) == (200, None)

    json_status, json_result = asyncio.run(
        nvcf_utils.poll_nvcf_status(
            PollClient(
                [
                    PollResponse(202),
                    PollResponse(
                        200,
                        {"content-type": "application/json"},
                        {"done": True},
                    ),
                ]
            ),
            "req-json",
            api_key="key",
            poll_seconds=5,
            timeout=10,
        )
    )
    assert (json_status, json_result) == (200, {"done": True})

    unexpected_status, unexpected_result = asyncio.run(
        nvcf_utils.poll_nvcf_status(
            PollClient([PollResponse(200, {"content-type": "text/plain"})]),
            "req-text",
            api_key="key",
            poll_seconds=5,
            timeout=10,
        )
    )
    assert (unexpected_status, unexpected_result) == (200, None)

    status, result = asyncio.run(
        nvcf_utils.poll_nvcf_status(
            PollClient([PollResponse(418)]),
            "req-status",
            api_key="key",
            poll_seconds=5,
            timeout=10,
        )
    )
    assert (status, result) == (418, None)

    timeout_status, timeout_result = asyncio.run(
        nvcf_utils.poll_nvcf_status(
            PollClient([PollResponse(504)]),
            "req-timeout",
            api_key="key",
            poll_seconds=5,
            timeout=10,
        )
    )
    assert (timeout_status, timeout_result) == (504, None)

    error_status, error_result = asyncio.run(
        nvcf_utils.poll_nvcf_status(
            PollClient([RuntimeError("network")]),
            "req-error",
            api_key="key",
            poll_seconds=5,
            timeout=10,
        )
    )
    assert (error_status, error_result) == (500, None)

    times = iter([0.0, 0.0, 11.0, 11.0])
    monkeypatch.setattr(nvcf_utils, "time", SimpleNamespace(time=lambda: next(times)))
    timed_out_status, timed_out_result = asyncio.run(
        nvcf_utils.poll_nvcf_status(
            PollClient([PollResponse(202)]),
            "req-client-timeout",
            api_key="key",
            poll_seconds=5,
            timeout=10,
        )
    )
    assert (timed_out_status, timed_out_result) == (504, None)


class AsyncResponse:
    def __init__(
        self,
        status_code: int,
        headers: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
        content: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload or {}
        self.content = content

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class FakeAsyncClient:
    responses: list[AsyncResponse] = []
    posts: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None

    async def post(
        self, url: str, headers: dict[str, str], json: dict[str, Any]
    ) -> AsyncResponse:
        self.posts.append({"url": url, "headers": headers, "json": json})
        return self.responses.pop(0)


def test_execute_nvcf_request_async_retry_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    FakeAsyncClient.responses = [
        AsyncResponse(202),
        AsyncResponse(200, {"content-type": "application/json"}, {"ok": True}),
    ]
    result = asyncio.run(
        nvcf_utils.execute_nvcf_request_async(
            ".invocation.api.nvcf.nvidia.com",
            {"NVCF-POLL-SECONDS": "7"},
            {"x": 1},
            api_key="key",
            timeout=1,
            max_retries=1,
            retry_delay=0,
            retry_jitter=0,
        )
    )
    assert result == {"ok": True}

    async def poll_fail_once(**_kwargs: Any) -> tuple[int, dict[str, Any] | None]:
        if not hasattr(poll_fail_once, "called"):
            poll_fail_once.called = True  # type: ignore[attr-defined]
            return 504, None
        return 200, {"polled": True}

    monkeypatch.setattr(nvcf_utils, "poll_nvcf_status", poll_fail_once)
    FakeAsyncClient.responses = [
        AsyncResponse(202, {"nvcf-reqid": "req-1"}),
        AsyncResponse(202, {"nvcf-reqid": "req-2"}),
    ]
    result = asyncio.run(
        nvcf_utils.execute_nvcf_request_async(
            "https://func.invocation.api.nvcf.nvidia.com",
            {},
            {},
            api_key="key",
            timeout=1,
            max_retries=1,
            retry_delay=0,
            retry_jitter=0,
        )
    )
    assert result == {"polled": True}

    FakeAsyncClient.responses = [
        AsyncResponse(200, {"content-type": "application/zip"}, content=b"bad"),
        AsyncResponse(200, {"content-type": "text/plain"}, {"fallback": True}),
    ]
    result = asyncio.run(
        nvcf_utils.execute_nvcf_request_async(
            "http://local",
            {},
            {},
            api_key="",
            timeout=1,
            max_retries=1,
            retry_delay=0,
            retry_jitter=0,
        )
    )
    assert result == {"fallback": True}

    FakeAsyncClient.responses = [
        AsyncResponse(
            200,
            {"content-type": "application/zip"},
            content=_zip_response({"zip": True}),
        )
    ]
    result = asyncio.run(
        nvcf_utils.execute_nvcf_request_async(
            "http://local",
            {},
            {},
            api_key="",
            timeout=1,
            max_retries=0,
        )
    )
    assert result == {"zip": True}

    FakeAsyncClient.responses = [
        AsyncResponse(200, {"content-type": "application/zip"}, content=b"bad")
    ]
    with pytest.raises(RuntimeError, match="Failed to parse ZIP response"):
        asyncio.run(
            nvcf_utils.execute_nvcf_request_async(
                "http://local",
                {},
                {},
                api_key="",
                timeout=1,
                max_retries=0,
            )
        )


def test_sync_retry_delay_malformed_url_and_unexpected_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(nvcf_utils.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(nvcf_utils.time, "sleep", sleeps.append)

    class SyncResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

    calls = {"count": 0}

    def flaky_post(*_args: Any, **_kwargs: Any) -> SyncResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.ConnectionError("temporary")
        return SyncResponse()

    monkeypatch.setattr(nvcf_utils.requests, "post", flaky_post)
    response = nvcf_utils.execute_nvcf_request_with_retry(
        ".invocation.api.nvcf.nvidia.com",
        {},
        {},
        timeout=1,
        max_retries=1,
        retry_delay=0.5,
        retry_jitter=0,
    )
    assert response.status_code == 200
    assert sleeps == [0.5]

    def success_post(*_args: Any, **_kwargs: Any) -> SyncResponse:
        return SyncResponse()

    monkeypatch.setattr(nvcf_utils.requests, "post", success_post)
    response = nvcf_utils.execute_nvcf_request_with_retry(
        "https://func.invocation.api.nvcf.nvidia.com",
        {},
        {},
        timeout=1,
        max_retries=0,
    )
    assert response.status_code == 200

    fallback = nvcf_utils.execute_nvcf_request_with_retry(
        "http://local",
        {},
        {},
        timeout=1,
        max_retries=-1,
        error_response_factory=lambda message, elapsed: {
            "message": message,
            "elapsed": elapsed,
        },
    )
    assert fallback["message"] == "NVCF request failed: unexpected error"
