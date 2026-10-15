# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the hosted NIM transport policy."""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from importlib.metadata import version
from typing import Any

import pytest
import requests
from urllib3 import HTTPResponse
from urllib3.exceptions import MaxRetryError

from world_understanding.functions.models import nim_transport
from world_understanding.functions.models.backends.public import nim
from world_understanding.functions.models.vision_language_models import NvidiaNIMVLM

_LLAMA_VISION = "meta/llama-3.2-11b-vision-instruct"


def _chat_model(factory: str, **kwargs: Any) -> Any:
    if factory == "vlm":
        return NvidiaNIMVLM(api_key="nvapi-test", model=_LLAMA_VISION, **kwargs)
    return nim.create_nim_chat(api_key="nvapi-test", model=_LLAMA_VISION, **kwargs)


def _sdk_client(model: Any) -> Any:
    return getattr(model, "chat_model", model)._client


def _async_sdk_client(model: Any) -> Any:
    return getattr(model, "chat_model", model)._async_client


@pytest.mark.parametrize("factory", ["vlm", "chat"])
def test_hosted_nim_stays_on_configured_integrate_url(factory: str) -> None:
    from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE

    assert version("langchain-nvidia-ai-endpoints") == "1.4.3"
    # The locked SDK reroutes this model away from the configured base URL.
    assert "/v1/gr/" in (MODEL_TABLE[_LLAMA_VISION].endpoint or "")

    model = _chat_model(factory)

    for client in (_sdk_client(model), _async_sdk_client(model)):
        assert (
            client.infer_url == "https://integrate.api.nvidia.com/v1/chat/completions"
        )
        assert client.interval == 1.0
    adapter = _sdk_client(model).get_session_fn().get_adapter("https://x")
    assert adapter.max_retries is nim_transport._HOSTED_RETRY


def test_hosted_nim_keeps_sdk_endpoint_for_nv_vlm_models() -> None:
    """nv-vlm models need the SDK's own payload format on their own endpoint."""
    vlm = NvidiaNIMVLM(
        api_key="nvapi-test", model="microsoft/phi-3-vision-128k-instruct"
    )

    assert "/vlm/" in vlm.chat_model._client.infer_url
    assert vlm.chat_model._client.interval == 1.0


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://integrate.api.nvidia.com:443/v1",
        "https://INTEGRATE.api.nvidia.com/v1",
        "https://ai.api.nvidia.com:443/v1",
    ],
)
def test_hosted_policy_matches_equivalent_hosted_urls(endpoint_url: str) -> None:
    client = _sdk_client(_chat_model("vlm", base_url=endpoint_url))

    assert client.interval == 1.0
    assert type(client) in nim_transport._slot_bounded_types.values()
    adapter = client.get_session_fn().get_adapter("https://x")
    assert adapter.max_retries is nim_transport._HOSTED_RETRY


@pytest.mark.parametrize("factory", ["vlm", "chat"])
def test_self_hosted_nim_transport_is_untouched(factory: str) -> None:
    model = _chat_model(factory, base_url="http://localhost:8000/v1")

    client = _sdk_client(model)
    assert client.infer_url == "http://localhost:8000/v1/chat/completions"
    assert client.interval == 0.02
    adapter = client.get_session_fn().get_adapter("https://x")
    assert adapter.max_retries is not nim_transport._HOSTED_RETRY
    assert type(client) not in nim_transport._slot_bounded_types.values()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://INTEGRATE.api.nvidia.com:443/v1/chat/completions",
            "https://integrate.api.nvidia.com/v1/chat/completions",
        ),
        ("http://localhost:80/v1/", "http://localhost/v1"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1"),
    ],
)
def test_canonical_endpoint_spells_equivalent_urls_once(
    url: str, expected: str
) -> None:
    assert nim_transport.canonical_endpoint(url) == expected


def test_hosted_retry_policy_backs_off_on_throttling_only() -> None:
    retry = nim_transport._HOSTED_RETRY

    for status in (429, 503):
        assert retry.is_retry("POST", status)
    for status in (400, 500, 502, 504):
        assert not retry.is_retry("POST", status)
    throttled = HTTPResponse(status=429, headers={"Retry-After": "21600"})
    assert retry.get_retry_after(throttled) == retry.backoff_max
    for attempt in range(5):
        retry = retry.increment("POST", "/v1/chat/completions")
        floor = 2.0 * 2**attempt
        assert floor <= retry.get_backoff_time() <= floor + retry.backoff_jitter
    with pytest.raises(MaxRetryError):
        retry.increment("POST", "/v1/chat/completions")


@pytest.mark.parametrize("method", ["get_req", "get_req_stream"])
@pytest.mark.parametrize(("limit", "expected"), [("2", 2), ("0", 6)])
def test_hosted_client_holds_one_slot_per_whole_request(
    monkeypatch: pytest.MonkeyPatch, method: str, limit: str, expected: int
) -> None:
    """The cap covers a whole request: retries, 202 polling and stream reads."""
    monkeypatch.setenv(nim_transport.MAX_CONCURRENCY_ENV, limit)
    client = _sdk_client(_chat_model("vlm"))
    active = peak = 0
    full = threading.Condition()

    def chunks() -> Iterator[dict[str, Any]]:
        nonlocal active, peak
        with full:
            active += 1
            peak = max(peak, active)
            full.notify_all()
            # Hold every admitted request until the expected overlap is reached.
            assert full.wait_for(lambda: peak >= expected, timeout=10)
        yield {}
        with full:
            active -= 1

    sdk_client_type = type(client).__mro__[1]
    monkeypatch.setattr(sdk_client_type, "get_req", lambda *_a, **_k: list(chunks()))
    monkeypatch.setattr(sdk_client_type, "get_req_stream", lambda *_a, **_k: chunks())
    threads = [
        threading.Thread(target=lambda: list(getattr(client, method)({})))
        for _ in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak == expected


def test_hosted_slot_is_released_after_errors_and_closed_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(nim_transport.MAX_CONCURRENCY_ENV, "1")
    client = _sdk_client(_chat_model("vlm"))
    sdk_client_type = type(client).__mro__[1]

    def failing(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("[429] Too Many Requests")

    monkeypatch.setattr(sdk_client_type, "get_req", failing)
    monkeypatch.setattr(
        sdk_client_type, "get_req_stream", lambda *_a, **_k: iter([{}, {}])
    )
    with pytest.raises(RuntimeError):
        client.get_req({})
    stream = client.get_req_stream({})
    next(stream)
    stream.close()

    semaphore = nim_transport._hosted_semaphore()
    assert semaphore is not None
    assert semaphore.acquire(blocking=False)
    semaphore.release()


@pytest.mark.parametrize(("raw", "expected"), [("", 4), ("  ", 4), ("x", 4), ("3", 3)])
def test_hosted_concurrency_env_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    monkeypatch.setenv(nim_transport.MAX_CONCURRENCY_ENV, raw)

    assert nim_transport._max_concurrency() == expected


def test_hosted_slot_logs_queue_wait(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class BusySemaphore:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def acquire(self, blocking: bool = True) -> bool:
            self.calls.append(blocking)
            return blocking

        def release(self) -> None:
            self.calls.append(False)

    busy = BusySemaphore()
    monkeypatch.setattr(nim_transport, "_hosted_semaphore", lambda: busy)

    with caplog.at_level("DEBUG", logger=nim_transport.__name__):
        with nim_transport._hosted_slot():
            pass

    assert busy.calls == [False, True, False]
    assert "Waited" in caplog.text


def test_hosted_client_is_wrapped_once() -> None:
    client = _sdk_client(_chat_model("vlm"))
    bounded = type(client)

    assert nim_transport._slot_bounded_client_type(bounded) is bounded


def test_hosted_retry_ignores_malformed_retry_after() -> None:
    malformed = HTTPResponse(status=429, headers={"Retry-After": "soon"})

    assert nim_transport._HOSTED_RETRY.get_retry_after(malformed) is None


def test_hosted_read_timeout_stays_a_non_retried_read_timeout() -> None:
    """A read timeout must surface as ReadTimeout, which callers never retry."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    accepted: list[socket.socket] = []
    acceptor = threading.Thread(
        target=lambda: accepted.append(listener.accept()[0]), daemon=True
    )
    acceptor.start()
    session = nim_transport._hosted_session(True)
    session.mount("http://", session.get_adapter("https://x"))
    try:
        with pytest.raises(requests.exceptions.ReadTimeout):
            session.post(
                f"http://127.0.0.1:{listener.getsockname()[1]}/v1/chat/completions",
                json={},
                timeout=0.2,
            )
    finally:
        acceptor.join(timeout=1)
        for connection in accepted:
            connection.close()
        listener.close()
    assert len(accepted) == 1
