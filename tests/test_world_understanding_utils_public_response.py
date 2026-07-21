# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for public service response sanitization."""

from __future__ import annotations

import json
from typing import cast

import pytest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from world_understanding.utils import public_response
from world_understanding.utils.public_response import (
    PublicJsonResponseSanitizationMiddleware,
    sanitize_public_response_payload,
)


def test_public_payload_projects_paths_and_internal_endpoints() -> None:
    session_id = "12345678-1234-1234-1234-123456789abc"
    root = "/var/material-agent/sessions"
    absolute_path = f"{root}/{session_id}/cache/optimized/input.usd"
    sibling_path = f"{root}_backup/retained.usd"
    payload = {
        "library_path": absolute_path,
        "generated_files": [absolute_path],
        "source_payload_file": absolute_path,
        "error_message": (
            f"render failed for {absolute_path} via "
            "http://ovrtx-rendering-api:8000/render"
        ),
        "cluster_error": (
            "request to http://render.graphics.svc.cluster.local:8000/render failed"
        ),
        "ipv6_loopback_error": "request to http://[::1]:8080/render failed",
        "ipv6_ula_error": "request to http://[fd00::1]:8080/render failed",
        "nvcf_deployment_error": (
            "request to https://function-id.invocation.api.nvcf.nvidia.com/run failed"
        ),
        "nvcf_status_error": (
            "polling https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/request-id failed"
        ),
        "connection_pool_error": (
            "HTTPConnectionPool(host='ovrtx-rendering-api', port=8000): exhausted"
        ),
        "link_local_error": "request to http://169.254.169.254/latest failed",
        "uppercase_internal_url": "HTTPS://RENDER.GRAPHICS.SVC.CLUSTER.LOCAL/run",
        "sibling_path_message": sibling_path,
        "external_url": "https://api.example.test/v1/results",
        "external_ipv6_url": "https://[2606:4700:4700::1111]/dns-query",
        "external_pool_error": (
            "HTTPSConnectionPool(host='api.example.test', port=443): exhausted"
        ),
        "target_prim_path": "/World/Tire",
    }

    sanitized = sanitize_public_response_payload(payload, session_roots=(root,))

    session_uri = f"session://{session_id}/cache/optimized/input.usd"
    assert sanitized["library_path"] == session_uri
    assert sanitized["generated_files"] == [session_uri]
    assert sanitized["source_payload_file"] == session_uri
    assert sanitized["error_message"] == (
        "render failed for <session> via <internal-endpoint>"
    )
    assert sanitized["cluster_error"] == ("request to <internal-endpoint> failed")
    assert sanitized["ipv6_loopback_error"] == ("request to <internal-endpoint> failed")
    assert sanitized["ipv6_ula_error"] == "request to <internal-endpoint> failed"
    assert sanitized["nvcf_deployment_error"] == (
        "request to <internal-endpoint> failed"
    )
    assert sanitized["nvcf_status_error"] == "polling <internal-endpoint> failed"
    assert sanitized["connection_pool_error"] == "<internal-endpoint>: exhausted"
    assert sanitized["link_local_error"] == "request to <internal-endpoint> failed"
    assert sanitized["uppercase_internal_url"] == "<internal-endpoint>"
    assert sanitized["sibling_path_message"] == sibling_path
    assert sanitized["external_url"] == payload["external_url"]
    assert sanitized["external_ipv6_url"] == payload["external_ipv6_url"]
    assert sanitized["external_pool_error"] == payload["external_pool_error"]
    assert sanitized["target_prim_path"] == "/World/Tire"


async def _invoke_middleware(
    middleware: PublicJsonResponseSanitizationMiddleware,
    *,
    method: str = "GET",
    extensions: dict[str, object] | None = None,
) -> list[Message]:
    """Invoke middleware with a minimal HTTP exchange and collect messages."""
    messages: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    scope = cast(
        Scope,
        {"type": "http", "method": method, "extensions": extensions or {}},
    )
    await middleware(scope, receive, send)
    return messages


@pytest.mark.asyncio
async def test_json_middleware_sanitizes_serialized_response() -> None:
    root = "/var/physics-agent/sessions"
    session_id = "12345678-1234-1234-1234-123456789abc"

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        body = json.dumps(
            {"identification_path": f"{root}/{session_id}/cache/id.json"}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    middleware = PublicJsonResponseSanitizationMiddleware(
        cast(ASGIApp, app),
        session_roots=(root,),
    )
    messages = await _invoke_middleware(middleware)

    body = json.loads(messages[1]["body"])
    assert body["identification_path"] == (f"session://{session_id}/cache/id.json")
    headers = dict(messages[0]["headers"])
    assert int(headers[b"content-length"]) == len(messages[1]["body"])


@pytest.mark.asyncio
async def test_json_middleware_bounds_buffering_and_fails_closed() -> None:
    internal_path = "/var/physics-agent/sessions/session-id/cache/secret.json"

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        body = json.dumps({"output_path": internal_path}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        midpoint = len(body) // 2
        await send(
            {
                "type": "http.response.body",
                "body": body[:midpoint],
                "more_body": True,
            }
        )
        await send({"type": "http.response.body", "body": body[midpoint:]})

    middleware = PublicJsonResponseSanitizationMiddleware(
        cast(ASGIApp, app),
        max_body_bytes=32,
    )
    messages = await _invoke_middleware(middleware)

    assert messages[0]["status"] == 500
    assert internal_path.encode() not in messages[1]["body"]
    assert json.loads(messages[1]["body"]) == {
        "detail": "Public JSON response sanitization failed"
    }
    headers = dict(messages[0]["headers"])
    assert int(headers[b"content-length"]) == len(messages[1]["body"])


@pytest.mark.asyncio
async def test_json_middleware_contains_recursive_sanitization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        body = b'{"status":"ok"}'
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def fail_sanitization(*_args: object, **_kwargs: object) -> None:
        raise RecursionError("nested response")

    monkeypatch.setattr(
        public_response,
        "_sanitize_prepared_public_response_payload",
        fail_sanitization,
    )
    middleware = PublicJsonResponseSanitizationMiddleware(cast(ASGIApp, app))
    messages = await _invoke_middleware(middleware)

    assert messages[0]["status"] == 500
    assert json.loads(messages[1]["body"]) == {
        "detail": "Public JSON response sanitization failed"
    }


@pytest.mark.asyncio
async def test_json_middleware_contains_unicode_encoding_failure() -> None:
    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b'{"detail":"\\ud800"}'})

    middleware = PublicJsonResponseSanitizationMiddleware(cast(ASGIApp, app))
    messages = await _invoke_middleware(middleware)

    assert messages[0]["status"] == 500
    assert json.loads(messages[1]["body"]) == {
        "detail": "Public JSON response sanitization failed"
    }


@pytest.mark.parametrize(
    ("status", "method"),
    [(100, "GET"), (204, "GET"), (205, "GET"), (304, "GET"), (200, "HEAD")],
)
@pytest.mark.asyncio
async def test_json_middleware_preserves_body_forbidden_empty_response(
    status: int,
    method: str,
) -> None:
    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", b"0"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    middleware = PublicJsonResponseSanitizationMiddleware(cast(ASGIApp, app))
    messages = await _invoke_middleware(middleware, method=method)

    assert messages[0]["status"] == status
    assert messages[1]["body"] == b""


@pytest.mark.asyncio
async def test_ndjson_middleware_sanitizes_split_records() -> None:
    root = "/var/physics-agent/sessions"
    internal_path = f"{root}/session-id/cache/predictions.jsonl"
    records = (
        json.dumps({"output_path": internal_path})
        + "\n"
        + json.dumps(
            {
                "error": (
                    "request to https://function-id.invocation.api.nvcf.nvidia.com/run"
                )
            }
        )
        + "\n"
    ).encode()

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/x-ndjson"),
                    (b"content-length", str(len(records)).encode()),
                ],
            }
        )
        split = len(records) // 2
        await send(
            {
                "type": "http.response.body",
                "body": records[:split],
                "more_body": True,
            }
        )
        await send({"type": "http.response.body", "body": records[split:]})

    middleware = PublicJsonResponseSanitizationMiddleware(
        cast(ASGIApp, app),
        session_roots=(root,),
    )
    messages = await _invoke_middleware(middleware)

    headers = dict(messages[0]["headers"])
    assert b"content-length" not in headers
    body = b"".join(message.get("body", b"") for message in messages[1:])
    sanitized_records = [json.loads(line) for line in body.splitlines()]
    assert sanitized_records == [
        {"output_path": "session://session-id/cache/predictions.jsonl"},
        {"error": "request to <internal-endpoint>"},
    ]
    assert internal_path.encode() not in body
    assert b"nvcf.nvidia.com" not in body


@pytest.mark.asyncio
async def test_sse_middleware_sanitizes_split_json_data_records() -> None:
    root = "/var/material-agent/sessions"
    internal_path = f"{root}/session-id/cache/output.usd"
    event = (
        "event: progress\r\n"
        "data: "
        + json.dumps(
            {
                "output_path": internal_path,
                "message": (
                    "polling https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/id"
                ),
            }
        )
        + "\r\n\r\n"
    ).encode()

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream; charset=utf-8")],
            }
        )
        split = event.index(b"output.usd")
        await send(
            {
                "type": "http.response.body",
                "body": event[:split],
                "more_body": True,
            }
        )
        await send({"type": "http.response.body", "body": event[split:]})

    middleware = PublicJsonResponseSanitizationMiddleware(
        cast(ASGIApp, app),
        session_roots=(root,),
    )
    messages = await _invoke_middleware(middleware)

    body = b"".join(message.get("body", b"") for message in messages[1:])
    data_line = next(line for line in body.splitlines() if line.startswith(b"data:"))
    payload = json.loads(data_line.removeprefix(b"data: "))
    assert payload == {
        "output_path": "session://session-id/cache/output.usd",
        "message": "polling <internal-endpoint>",
    }
    assert internal_path.encode() not in body
    assert b"nvcf.nvidia.com" not in body


@pytest.mark.parametrize(
    "content_type",
    [
        b"application/json",
        b"application/x-ndjson",
        b"text/event-stream",
        b"multipart/byteranges; boundary=public-test",
    ],
)
@pytest.mark.asyncio
async def test_structured_range_responses_fail_closed(content_type: bytes) -> None:
    secret = b"/var/material-agent/sessions/session-id/manifest.json"

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 206,
                "headers": [
                    (b"content-type", content_type),
                    (b"content-range", b"bytes 0-10/100"),
                    (b"accept-ranges", b"bytes"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": secret})

    middleware = PublicJsonResponseSanitizationMiddleware(cast(ASGIApp, app))
    messages = await _invoke_middleware(middleware)

    assert messages[0]["status"] == 416
    headers = dict(messages[0]["headers"])
    assert b"content-range" not in headers
    assert b"accept-ranges" not in headers
    assert secret not in messages[1]["body"]
    assert json.loads(messages[1]["body"]) == {
        "detail": "Public JSON response sanitization failed"
    }


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        (
            b"application/json",
            b'{"output_path":"/var/material-agent/sessions/id/out.json"}',
        ),
        (
            b"application/x-ndjson",
            b'{"output_path":"/var/material-agent/sessions/id/out.json"}\n',
        ),
        (
            b"text/event-stream",
            b'data: {"output_path":"/var/material-agent/sessions/id/out.json"}\n\n',
        ),
    ],
)
@pytest.mark.asyncio
async def test_identity_encoded_structured_responses_are_sanitized(
    content_type: bytes,
    body: bytes,
) -> None:
    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", content_type),
                    (b"content-encoding", b"identity"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    middleware = PublicJsonResponseSanitizationMiddleware(cast(ASGIApp, app))
    messages = await _invoke_middleware(middleware)

    sanitized_body = b"".join(message.get("body", b"") for message in messages[1:])
    assert b"/var/material-agent/sessions" not in sanitized_body
    assert b"session://id/out.json" in sanitized_body


@pytest.mark.parametrize(
    "content_type",
    [b"application/json", b"application/x-ndjson", b"text/event-stream"],
)
@pytest.mark.asyncio
async def test_compressed_structured_responses_fail_closed(
    content_type: bytes,
) -> None:
    secret = b"/var/material-agent/sessions/id/out.json"

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", content_type),
                    (b"content-encoding", b"gzip"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": secret})

    middleware = PublicJsonResponseSanitizationMiddleware(cast(ASGIApp, app))
    messages = await _invoke_middleware(middleware)

    assert messages[0]["status"] == 500
    headers = dict(messages[0]["headers"])
    assert b"content-encoding" not in headers
    assert secret not in messages[1]["body"]
    assert json.loads(messages[1]["body"]) == {
        "detail": "Public JSON response sanitization failed"
    }


@pytest.mark.asyncio
async def test_middleware_disables_pathsend_before_structured_response() -> None:
    internal_path = "/var/physics-agent/sessions/session-id/cache/secret.json"

    async def app(scope: Scope, _receive: Receive, send: Send) -> None:
        assert "http.response.pathsend" not in scope.get("extensions", {})
        body = json.dumps({"output_path": internal_path}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    middleware = PublicJsonResponseSanitizationMiddleware(cast(ASGIApp, app))
    messages = await _invoke_middleware(
        middleware,
        extensions={"http.response.pathsend": {}},
    )

    assert internal_path.encode() not in messages[1]["body"]
    assert json.loads(messages[1]["body"]) == {
        "output_path": "session://session-id/cache/secret.json"
    }
