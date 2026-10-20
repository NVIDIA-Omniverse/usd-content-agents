# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hosted NIM transport policy for LangChain ChatNVIDIA clients.

The locked ``langchain-nvidia-ai-endpoints`` 1.4.3 SDK has three behaviours
that break hosted NIM workloads:

* its static model table reroutes some vision models from the configured
  ``integrate.api.nvidia.com`` base URL to ``ai.api.nvidia.com/v1/gr/...``,
  which can fail (HTTP 451 or 500) where the integrate route works;
* it never retries, and it raises on HTTP 429 without the response headers;
* it polls an HTTP 202 request status every 20 ms.

For hosted NVIDIA endpoints this module keeps requests on the configured base
URL, bounds concurrent synchronous requests per process (each slot is held
for the whole request, including 202 polling and streaming), retries 429/503
with jittered exponential backoff (honouring ``Retry-After`` when sent), and
polls 202 responses once per second. Self-hosted NIM endpoints are untouched.

Hosted endpoints also hang or fail fast on a minority of requests. The same
retry policy therefore retries HTTP 500/502/504 and
read timeouts. When the client uses the default deadline, each POST attempt
reads for at most ``WU_NIM_ATTEMPT_TIMEOUT_SECONDS`` so that a hang can be
retried; a timeout the caller chose is never shortened and applies to every
attempt. Completions have no side effects, so a repeated request costs tokens
only. 202 polling keeps the client's whole timeout.
"""

import contextlib
import functools
import logging
import os
import random
import threading
import time
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.exceptions import InvalidHeader, MaxRetryError, ReadTimeoutError
from urllib3.util.retry import Retry

from world_understanding.functions.models.request_timeout import (
    DEFAULT_HOSTED_NIM_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS,
)
from world_understanding.utils.environment import parse_float_env

logger = logging.getLogger(__name__)

_INTEGRATE_HOST = "integrate.api.nvidia.com"
_HOSTED_HOSTS = frozenset({_INTEGRATE_HOST, "ai.api.nvidia.com"})
_CHAT_COMPLETIONS_PATH = "{base_url}/chat/completions"
_POLL_INTERVAL_SECONDS = 1.0

MAX_CONCURRENCY_ENV = "WU_NIM_HOSTED_MAX_CONCURRENT_REQUESTS"
DEFAULT_MAX_CONCURRENCY = 4

ATTEMPT_TIMEOUT_ENV = "WU_NIM_ATTEMPT_TIMEOUT_SECONDS"
DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 180.0


class _HostedRetry(Retry):
    """Back off from the first retry; urllib3 retries the first failure at once."""

    def get_backoff_time(self) -> float:
        backoff = super().get_backoff_time()
        if backoff:
            return backoff
        return min(
            self.backoff_max,
            self.backoff_factor + random.uniform(0, self.backoff_jitter),
        )

    def get_retry_after(self, response: Any) -> float | None:
        # A long Retry-After would otherwise hold a hosted slot for hours, and a
        # malformed one would replace the HTTP error with InvalidHeader.
        try:
            retry_after = super().get_retry_after(response)
        except InvalidHeader:
            return None
        return None if retry_after is None else min(retry_after, self.backoff_max)


# 429 and 503 (and, by urllib3's rule, 413 with Retry-After) mean the service
# turned the request away. Hosted chat completions also hang or fail fast with
# 500, 502 or 504; they have no side effects, so those statuses
# and up to two read timeouts are retried too. A final 5xx comes back as the
# response, so the SDK raises its usual error; an exhausted read stays a
# ReadTimeout (see _HostedAdapter), which callers never retry.
_HOSTED_RETRY = _HostedRetry(
    total=5,
    connect=2,
    read=2,
    other=0,
    status=5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=None,
    backoff_factor=2.0,
    backoff_jitter=2.0,
    backoff_max=60.0,
    respect_retry_after_header=True,
    raise_on_status=False,
)

_semaphore_lock = threading.Lock()
_semaphore: threading.BoundedSemaphore | None = None
_semaphore_limit: int | None = None


def _max_concurrency() -> int:
    raw = os.environ.get(MAX_CONCURRENCY_ENV, "")
    if not raw.strip():
        return DEFAULT_MAX_CONCURRENCY
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using %d", MAX_CONCURRENCY_ENV, raw, DEFAULT_MAX_CONCURRENCY
        )
        return DEFAULT_MAX_CONCURRENCY


def _hosted_semaphore() -> threading.BoundedSemaphore | None:
    """Return the process-wide hosted NIM slot pool, or None when unlimited."""
    global _semaphore, _semaphore_limit
    limit = _max_concurrency()
    if limit <= 0:
        return None
    with _semaphore_lock:
        if _semaphore is None or _semaphore_limit != limit:
            _semaphore = threading.BoundedSemaphore(limit)
            _semaphore_limit = limit
        return _semaphore


@contextlib.contextmanager
def _hosted_slot() -> Iterator[None]:
    semaphore = _hosted_semaphore()
    if semaphore is None:
        yield
        return
    if not semaphore.acquire(blocking=False):
        # Queue time is outside every request deadline; make it visible.
        started = time.monotonic()
        semaphore.acquire()
        logger.debug(
            "Waited %.1f s for a hosted NIM request slot (%s=%s)",
            time.monotonic() - started,
            MAX_CONCURRENCY_ENV,
            _semaphore_limit,
        )
    try:
        yield
    finally:
        semaphore.release()


_slot_bounded_types: dict[type, type] = {}


def _slot_bounded_client_type(client_type: type) -> type:
    """Hold one hosted slot per logical request: retries, 202 polling, stream.

    A stream keeps its slot until it is exhausted or closed, so do not make a
    second hosted call from inside a stream consumer.
    """
    with _semaphore_lock:
        if client_type in _slot_bounded_types.values():
            return client_type
        if client_type not in _slot_bounded_types:

            class _SlotBoundedClient(client_type):
                def get_req(self, *args: Any, **kwargs: Any) -> Any:
                    with _hosted_slot():
                        return super().get_req(*args, **kwargs)

                def get_req_stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
                    with _hosted_slot():
                        yield from super().get_req_stream(*args, **kwargs)

            _slot_bounded_types[client_type] = _SlotBoundedClient
        return _slot_bounded_types[client_type]


def _attempt_timeout_seconds() -> float:
    return parse_float_env(
        ATTEMPT_TIMEOUT_ENV,
        DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
        minimum=10.0,
        maximum=3600.0,
        logger=logger,
    )


def _bounded_timeout(timeout: Any, limit: float) -> tuple[float, float]:
    """Cap a requests timeout (number, pair or None) at ``limit`` seconds."""
    connect, read = timeout if isinstance(timeout, tuple) else (timeout, timeout)
    return (
        limit if connect is None else min(float(connect), limit),
        limit if read is None else min(float(read), limit),
    )


class _HostedAdapter(HTTPAdapter):
    """Bound each POST attempt; keep an exhausted read a ReadTimeout.

    The SDK passes its one ``timeout`` to the POST and to every 202 poll. With
    the default deadline only the POST is capped, so a hung attempt can be
    retried while polling keeps the whole budget. An explicit caller timeout is
    left as is for every attempt. requests reports retries exhausted by read
    timeouts as a ConnectionError; restore ReadTimeout so timeouts stay
    terminal for callers.
    """

    def __init__(self, *args: Any, explicit_timeout: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.explicit_timeout = explicit_timeout

    def send(
        self, request: requests.PreparedRequest, *args: Any, **kwargs: Any
    ) -> requests.Response:
        if request.method == "POST" and not self.explicit_timeout:
            kwargs["timeout"] = _bounded_timeout(
                kwargs.get("timeout"), _attempt_timeout_seconds()
            )
        try:
            return super().send(request, *args, **kwargs)
        except requests.exceptions.ConnectionError as error:
            exhausted = error.args[0] if error.args else None
            if isinstance(exhausted, MaxRetryError) and isinstance(
                exhausted.reason, ReadTimeoutError
            ):
                raise requests.exceptions.ReadTimeout(
                    exhausted, request=request
                ) from error
            raise


def _hosted_session(
    verify: bool | str, *, explicit_timeout: bool = False
) -> requests.Session:
    session = requests.Session()
    session.verify = verify
    session.mount(
        "https://",
        _HostedAdapter(max_retries=_HOSTED_RETRY, explicit_timeout=explicit_timeout),
    )
    return session


def default_nim_request_timeout(base_url: str | None) -> float:
    """Return the NIM request deadline used when the caller sets none.

    Hosted NIM retries hung attempts inside a longer deadline; self-hosted
    endpoints get one attempt and keep the shared hosted-model default.
    """
    if not base_url or hosted_nim_host(base_url):
        return DEFAULT_HOSTED_NIM_REQUEST_TIMEOUT_SECONDS
    return DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS


def hosted_nim_host(base_url: str | None) -> str | None:
    """Return the hosted NVIDIA NIM hostname of ``base_url``, or None.

    ``.hostname`` drops an explicit port and lowercases, unlike ``.netloc``.
    """
    host = urlparse(base_url or "").hostname
    return host if host in _HOSTED_HOSTS else None


def canonical_endpoint(url: str) -> str:
    """Spell an endpoint URL one way: lowercase host, no default port."""
    parts = urlparse(url)
    scheme = parts.scheme.lower()
    port = parts.port
    if port is None or (scheme, port) in {("https", 443), ("http", 80)}:
        netloc = parts.hostname or ""
    else:
        netloc = f"{parts.hostname}:{port}"
    return f"{scheme}://{netloc}{parts.path.rstrip('/')}"


def _configure_nim_chat_transport(
    chat_model: Any, *, label: str, explicit_timeout: bool = False
) -> None:
    """Apply the hosted NIM transport policy to a ChatNVIDIA instance.

    ``explicit_timeout`` says the caller chose the client timeout, so hosted
    attempts read for that whole timeout instead of the attempt cap.
    """
    for client_attr in ("_client", "_async_client"):
        client: Any = getattr(chat_model, client_attr, None)
        base_url = getattr(client, "base_url", None)
        if not isinstance(base_url, str):
            continue
        host = hosted_nim_host(base_url)
        if host is None:
            continue
        # nv-vlm models need the SDK's own payload format on their own endpoint.
        model_type = getattr(getattr(client, "model", None), "model_type", None)
        if (
            host == _INTEGRATE_HOST
            and model_type != "nv-vlm"
            and client.infer_path != _CHAT_COMPLETIONS_PATH
        ):
            logger.info(
                "%s keeps %s on %s instead of the SDK model-table endpoint %s",
                label,
                getattr(chat_model, "model", "the model"),
                base_url,
                client.infer_path,
            )
            client.infer_path = _CHAT_COMPLETIONS_PATH
        client.interval = _POLL_INTERVAL_SECONDS
        if client_attr == "_client":
            # The async client uses aiohttp; NvidiaNIMVLM only calls the sync one.
            client.get_session_fn = functools.partial(
                _hosted_session, client.verify_ssl, explicit_timeout=explicit_timeout
            )
            client.__class__ = _slot_bounded_client_type(type(client))
