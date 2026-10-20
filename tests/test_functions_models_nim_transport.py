# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the hosted NIM transport policy."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from typing import Any

import pytest
import requests
from urllib3 import HTTPResponse
from urllib3.exceptions import MaxRetryError

from world_understanding.functions.models import nim_transport
from world_understanding.functions.models.backends.public import nim
from world_understanding.functions.models.vision_language_models import NvidiaNIMVLM
from world_understanding.utils.model_timeout import NonRetryableVLMTimeoutError

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


def test_hosted_retry_policy_backs_off_on_throttling_and_server_errors() -> None:
    retry = nim_transport._HOSTED_RETRY

    for status in (429, 500, 502, 503, 504):
        assert retry.is_retry("POST", status)
    for status in (400, 401, 404, 410, 422):
        assert not retry.is_retry("POST", status)
    assert retry.read == 2
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


def test_hosted_exhausted_read_timeouts_stay_a_read_timeout() -> None:
    """Retried read timeouts end as ReadTimeout, which callers never retry."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    accepted: list[socket.socket] = []

    def accept() -> None:
        while True:
            try:
                accepted.append(listener.accept()[0])
            except OSError:
                return

    acceptor = threading.Thread(target=accept, daemon=True)
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
        listener.close()
        acceptor.join(timeout=1)
        for connection in accepted:
            connection.close()
    assert len(accepted) == 1 + nim_transport._HOSTED_RETRY.read


# ---------------------------------------------------------------------------
# Hung and failed hosted attempts
# ---------------------------------------------------------------------------

_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": _LLAMA_VISION,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class _ScriptedNim:
    """Local chat-completions server that answers each POST from a script.

    Actions: ``ok``, ``hang`` (never answer), an HTTP status such as ``500``,
    ``202`` (accepted; every status poll then hangs), ``stream`` and
    ``stream-hang`` (one chunk, then silence).
    """

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.posts = 0
        self.polls = 0
        self.release = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                pass

            def _send_json(self, status: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                if status == 202:
                    self.send_header("NVCF-REQID", "req-1")
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                owner.posts += 1
                action = owner.script.pop(0) if owner.script else "ok"
                if action == "hang":
                    owner.release.wait(10)
                elif action == "ok":
                    self._send_json(200, _COMPLETION)
                elif action in {"stream", "stream-hang"}:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    chunk = {"choices": [{"index": 0, "delta": {"content": "ok"}}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    if action == "stream-hang":
                        owner.release.wait(10)
                    else:
                        self.wfile.write(b"data: [DONE]\n\n")
                else:
                    status = int(action)
                    self._send_json(
                        status, {"status": status, "title": f"status {status}"}
                    )

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                owner.polls += 1
                owner.release.wait(10)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def scripted_nim(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[..., tuple[_ScriptedNim, Any, Any]]]:
    """Point a hosted-configured client at a scripted local server."""
    monkeypatch.setattr(nim_transport, "_attempt_timeout_seconds", lambda: 0.3)
    monkeypatch.setattr(
        nim_transport,
        "_HOSTED_RETRY",
        nim_transport._HOSTED_RETRY.new(backoff_factor=0, backoff_jitter=0),
    )
    servers: list[_ScriptedNim] = []

    def start(
        script: list[str], *, factory: str = "chat", timeout: float | None = None
    ) -> tuple[_ScriptedNim, Any, Any]:
        server = _ScriptedNim(script)
        servers.append(server)
        endpoint = {} if timeout is None else {"timeout": timeout}
        model = _chat_model(factory, **endpoint)
        client = _sdk_client(model)
        verify = client.verify_ssl
        explicit = client.get_session_fn.keywords["explicit_timeout"]

        def session() -> requests.Session:
            hosted = nim_transport._hosted_session(verify, explicit_timeout=explicit)
            hosted.mount("http://", hosted.get_adapter("https://x"))
            return hosted

        client.get_session_fn = session
        client.base_url = f"{server.url}/v1"
        client.polling_url_tmpl = f"{server.url}/v1/status/{{request_id}}"
        return server, model, client

    yield start
    for server in servers:
        server.close()


@pytest.mark.parametrize("first", ["hang", "500", "502", "504", "429", "503"])
def test_hosted_attempt_is_retried_after_a_hang_or_server_error(
    scripted_nim: Callable[..., tuple[_ScriptedNim, Any, Any]],
    first: str,
) -> None:
    server, _model, client = scripted_nim([first, "ok"])

    started = time.monotonic()
    response = client.get_req({})

    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert server.posts == 2
    # The hung POST reads for the attempt cap, not the 600 s default deadline.
    assert time.monotonic() - started < 5


def test_hosted_vlm_exhausted_hangs_stay_a_terminal_vlm_timeout(
    scripted_nim: Callable[..., tuple[_ScriptedNim, Any, Any]],
) -> None:
    server, vlm, _client = scripted_nim(["hang"] * 3, factory="vlm")

    with pytest.raises(NonRetryableVLMTimeoutError):
        vlm.generate("describe", images=[])

    assert server.posts == 1 + nim_transport._HOSTED_RETRY.read


def test_hosted_exhausted_server_errors_raise_the_sdk_error(
    scripted_nim: Callable[..., tuple[_ScriptedNim, Any, Any]],
) -> None:
    attempts = 1 + nim_transport._HOSTED_RETRY.status
    server, _model, client = scripted_nim(["500"] * attempts)

    with pytest.raises(Exception, match=r"\[500\]"):
        client.get_req({})

    assert server.posts == attempts


def test_hosted_accepted_request_is_never_posted_again(
    scripted_nim: Callable[..., tuple[_ScriptedNim, Any, Any]],
) -> None:
    """202 polling keeps the client timeout; the accepted POST is not repeated."""
    server, _model, client = scripted_nim(["202", "ok"], timeout=0.5)
    client.interval = 0.01

    with pytest.raises((requests.exceptions.ReadTimeout, TimeoutError)):
        client.get_req({})

    assert server.posts == 1
    assert server.polls >= 1


def test_hosted_stream_is_retried_only_before_its_first_byte(
    scripted_nim: Callable[..., tuple[_ScriptedNim, Any, Any]],
) -> None:
    server, _model, client = scripted_nim(["500", "stream"])

    assert list(client.get_req_stream({}))
    assert server.posts == 2

    stalled, _model, stalled_client = scripted_nim(["stream-hang", "stream"])
    with pytest.raises(requests.exceptions.RequestException):
        list(stalled_client.get_req_stream({}))
    assert stalled.posts == 1


@pytest.mark.parametrize(
    ("timeout", "expected"),
    [
        (None, (180.0, 180.0)),
        (600.0, (180.0, 180.0)),
        (120.0, (120.0, 120.0)),
        ((5.0, 600.0), (5.0, 180.0)),
    ],
)
def test_hosted_post_attempt_reads_for_at_most_the_attempt_cap(
    timeout: Any, expected: tuple[float, float]
) -> None:
    assert nim_transport._bounded_timeout(timeout, 180.0) == expected


@pytest.mark.parametrize("factory", ["vlm", "chat"])
@pytest.mark.parametrize(
    ("timeout", "expected_read"),
    [(None, 180.0), (600.0, 600.0), (120.0, 120.0)],
)
def test_hosted_attempt_cap_applies_only_to_the_default_deadline(
    monkeypatch: pytest.MonkeyPatch,
    factory: str,
    timeout: float | None,
    expected_read: float,
) -> None:
    """A timeout the caller chose is never shortened; the default is split."""
    monkeypatch.delenv(nim_transport.ATTEMPT_TIMEOUT_ENV, raising=False)
    sent: list[Any] = []

    def send(
        _adapter: Any, request: requests.PreparedRequest, **kwargs: Any
    ) -> requests.Response:
        sent.append(kwargs.get("timeout"))
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.request = request
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    endpoint = {} if timeout is None else {"timeout": timeout}
    client = _sdk_client(_chat_model(factory, **endpoint))

    client.get_session_fn().post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json={},
        timeout=client.timeout,
    )

    read = sent[-1][1] if isinstance(sent[-1], tuple) else sent[-1]
    assert read == expected_read
