# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from unittest.mock import patch

import httpx
import pytest

from texture_agent.functions.rest_client import RestTextureVariationClient
from texture_agent.functions.texture_generation import (
    Conditioning,
    TextureVariationConfig,
)


def test_rest_client_headers_submit_status_and_wait_false() -> None:
    captured: dict[str, object] = {}

    class AcceptedClient:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict) -> httpx.Response:
            return httpx.Response(
                202,
                json={"job_id": "job", "status": "queued", "progress": 7},
                request=httpx.Request("POST", url),
            )

    with patch("httpx.Client", AcceptedClient):
        status = RestTextureVariationClient(
            "http://texture-service/",
            api_key="secret",
        ).generate(
            "file:///asset.usd",
            Conditioning(text_prompt="steel"),
            TextureVariationConfig(),
            wait=False,
        )

    assert status.status == "queued"
    assert captured["headers"] == {"Authorization": "Bearer secret"}

    class BadStatusClient(AcceptedClient):
        def post(self, url: str, json: dict) -> httpx.Response:
            return httpx.Response(
                500,
                text="broken",
                request=httpx.Request("POST", url),
            )

    with patch("httpx.Client", BadStatusClient):
        failed = RestTextureVariationClient("http://texture-service").generate(
            "file:///asset.usd",
            Conditioning(text_prompt="steel"),
            TextureVariationConfig(),
            wait=False,
        )
    assert failed.status == "failed"
    assert failed.error_message == "HTTP 500: broken"


def test_submit_retry_delay_and_deadline_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RestTextureVariationClient(
        "http://texture-service",
        submit_retry_timeout_sec=0,
        submit_retry_base_delay_sec=0.2,
        submit_retry_max_delay_sec=0.5,
    )
    response = httpx.Response(
        429,
        headers={"Retry-After": "not-a-number"},
        request=httpx.Request("POST", "http://texture-service/jobs"),
    )
    assert client._submit_retry_delay(response, 2) == 0.5

    class AlwaysBusy:
        def post(self, url: str, json: dict) -> httpx.Response:
            return response

    assert (
        client._post_with_backpressure_retry(
            AlwaysBusy(),  # type: ignore[arg-type]
            "http://texture-service/jobs",
            {},
        ).status_code
        == 429
    )

    request = httpx.Request("POST", "http://texture-service/jobs")

    class AlwaysTransportError:
        def post(self, url: str, json: dict) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

    with pytest.raises(httpx.ConnectError):
        client._post_with_backpressure_retry(
            AlwaysTransportError(),  # type: ignore[arg-type]
            "http://texture-service/jobs",
            {},
        )


def test_get_status_and_cancel_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    class StatusClient:
        def __init__(self, response: httpx.Response) -> None:
            self.response = response
            self.health_response = httpx.Response(
                200,
                json={
                    "status": "healthy",
                    "ready": True,
                    "accepting_jobs": True,
                    "active_jobs": 0,
                    "queued_jobs": 0,
                    "max_workers": 1,
                    "gpu_available": True,
                },
                request=httpx.Request("GET", "http://texture-service/health"),
            )

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def get(self, url: str) -> httpx.Response:
            if url.endswith("/health"):
                return self.health_response
            return self.response

        def delete(self, url: str) -> httpx.Response:
            return self.response

    not_found = httpx.Response(
        404,
        request=httpx.Request("GET", "http://texture-service/job"),
    )
    with patch("httpx.Client", lambda *args, **kwargs: StatusClient(not_found)):
        status = RestTextureVariationClient("http://texture-service").get_status(
            "missing"
        )
    assert status.status == "failed"
    assert status.error_message == (
        "Job not found; texture service health: "
        "status=healthy, ready=True, accepting_jobs=True, active_jobs=0, "
        "queued_jobs=0, max_workers=1, gpu_available=True"
    )

    conflict = httpx.Response(
        409,
        request=httpx.Request("DELETE", "http://texture-service/job"),
    )
    with patch("httpx.Client", lambda *args, **kwargs: StatusClient(conflict)):
        with pytest.raises(ValueError, match="terminal state"):
            RestTextureVariationClient("http://texture-service").cancel("job")

    server_error = httpx.Response(
        500,
        request=httpx.Request("DELETE", "http://texture-service/job"),
    )
    with patch("httpx.Client", lambda *args, **kwargs: StatusClient(server_error)):
        with pytest.raises(httpx.HTTPStatusError):
            RestTextureVariationClient("http://texture-service").cancel("job")


def test_service_health_summary_handles_sparse_and_malformed_health() -> None:
    client = RestTextureVariationClient("http://texture-service")

    class HealthClient:
        def __init__(self, response: httpx.Response) -> None:
            self.response = response

        def get(self, url: str) -> httpx.Response:
            return self.response

    assert (
        client._service_health_summary(
            HealthClient(  # type: ignore[arg-type]
                httpx.Response(
                    503,
                    request=httpx.Request("GET", "http://texture-service/health"),
                )
            )
        )
        == "HTTP 503"
    )

    assert (
        client._service_health_summary(
            HealthClient(  # type: ignore[arg-type]
                httpx.Response(
                    200,
                    text="not\njson",
                    request=httpx.Request("GET", "http://texture-service/health"),
                )
            )
        )
        == "non-JSON response 'not json'"
    )

    assert (
        client._service_health_summary(
            HealthClient(  # type: ignore[arg-type]
                httpx.Response(
                    200,
                    json={"status": "", "error": None, "detail": "ignored"},
                    request=httpx.Request("GET", "http://texture-service/health"),
                )
            )
        )
        is None
    )

    non_dict_health_responses = (
        httpx.Response(
            200,
            content=b"null",
            request=httpx.Request("GET", "http://texture-service/health"),
        ),
        httpx.Response(
            200,
            json=42,
            request=httpx.Request("GET", "http://texture-service/health"),
        ),
        httpx.Response(
            200,
            json=["ready"],
            request=httpx.Request("GET", "http://texture-service/health"),
        ),
    )
    for response in non_dict_health_responses:
        assert (
            client._service_health_summary(
                HealthClient(response)  # type: ignore[arg-type]
            )
            is None
        )


def test_generate_polling_logs_progress_and_backs_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    monkeypatch.setattr(time, "sleep", lambda delay: sleeps.append(delay))

    class PollingClient:
        def __init__(self, *args, **kwargs) -> None:
            self.statuses = [
                {"job_id": "job", "status": "processing", "progress": 10},
                {"job_id": "job", "status": "completed", "progress": 100},
            ]

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict) -> httpx.Response:
            return httpx.Response(
                202,
                json={"job_id": "job", "status": "queued", "progress": 0},
                request=httpx.Request("POST", url),
            )

        def get(self, url: str) -> httpx.Response:
            return httpx.Response(
                200,
                json=self.statuses.pop(0),
                request=httpx.Request("GET", url),
            )

    with patch("httpx.Client", PollingClient):
        final = RestTextureVariationClient("http://texture-service").generate(
            "file:///asset.usd",
            Conditioning(text_prompt="steel"),
            TextureVariationConfig(),
            timeout_sec=60,
        )

    assert final.status == "completed"
    assert sleeps == [2.0, 3.0]


def test_post_with_backpressure_returns_when_retry_delay_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RestTextureVariationClient(
        "http://texture-service",
        submit_retry_timeout_sec=10,
    )
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "http://texture-service/jobs"),
    )

    class BusyClient:
        def post(self, url: str, json: dict) -> httpx.Response:
            return response

    monkeypatch.setattr(client, "_submit_retry_delay", lambda resp, attempt: 0.0)

    assert (
        client._post_with_backpressure_retry(
            BusyClient(),  # type: ignore[arg-type]
            "http://texture-service/jobs",
            {},
        )
        is response
    )


def test_parse_status_defaults_and_filters_maps() -> None:
    status = RestTextureVariationClient._parse_status(
        {
            "result": {
                "maps": {
                    "albedo": {"uri": "file:///a.png"},
                    "empty": {"uri": ""},
                    "malformed": "bad",
                },
                "generated_textures": {},
            }
        }
    )

    assert status.job_id == ""
    assert status.status == "failed"
    assert status.result is not None
    assert set(status.result.maps) == {"albedo"}
    assert status.result.generated_textures.albedo is None
