# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for public service response sanitization."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import cast
from urllib.parse import quote

import pytest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from world_understanding.utils import public_response
from world_understanding.utils.public_response import (
    PublicJsonResponseSanitizationMiddleware,
    sanitize_public_response_payload,
)


@pytest.mark.parametrize(
    "separator_pattern",
    (
        public_response._WINDOWS_SEPARATOR_PATTERN,
        public_response._POSIX_SEPARATOR_PATTERN,
    ),
)
def test_separator_patterns_reject_long_runs_without_backtracking(
    separator_pattern: str,
) -> None:
    pattern = re.compile(rf"{separator_pattern}var")
    started = time.perf_counter()

    assert pattern.search("/" * 1000 + "x") is None
    assert time.perf_counter() - started < 1.0


def test_public_payload_projects_paths_and_internal_endpoints() -> None:
    session_id = "12345678-1234-1234-1234-123456789abc"
    root = "/var/material-agent/sessions"
    absolute_path = f"{root}/{session_id}/cache/optimized/input.usd"
    encoded_path = quote(absolute_path, safe="")
    lowercase_encoded_path = encoded_path.replace("%2F", "%2f")
    sibling_path = f"{root}_backup/retained.usd"
    payload = {
        "library_path": absolute_path,
        "generated_files": [absolute_path],
        "source_payload_file": absolute_path,
        "error_message": (
            f"render failed for {absolute_path} via "
            "http://ovrtx-rendering-api:8000/render"
        ),
        "encoded_path_message": f"failed for {encoded_path}",
        "lowercase_encoded_path_message": f"failed for {lowercase_encoded_path}",
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
    assert sanitized["encoded_path_message"] == "failed for <session>"
    assert sanitized["lowercase_encoded_path_message"] == "failed for <session>"
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


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX symlink semantics")
def test_public_payload_redacts_canonical_path_for_symlinked_session_root(
    tmp_path: Path,
) -> None:
    session_id = "12345678-1234-1234-1234-123456789abc"
    real_root = tmp_path / "real-sessions"
    real_root.mkdir()
    alias_root = tmp_path / "configured-sessions"
    alias_root.symlink_to(real_root, target_is_directory=True)
    emitted_path = real_root / session_id / "cache" / "input.usd"

    sanitized = sanitize_public_response_payload(
        {
            "output_path": str(emitted_path),
            "error": f"failed for {emitted_path}",
        },
        session_roots=(alias_root,),
    )

    assert sanitized == {
        "output_path": f"session://{session_id}/cache/input.usd",
        "error": "failed for <session>",
    }


def test_public_payload_sanitizes_windows_session_paths_portably() -> None:
    session_id = "12345678-1234-1234-1234-123456789abc"
    root = r"C:\ProgramData\NVIDIA\material-agent\sessions"
    absolute_path = rf"{root}\{session_id}\cache\optimized\input.usd"
    mixed_path = (
        rf"C:/ProgramData\NVIDIA/material-agent\sessions/{session_id}\cache/input.usd"
    )
    case_variant = absolute_path.lower()
    escaped_path = absolute_path.replace("\\", "\\\\")
    encoded_path = quote(absolute_path, safe="")
    double_encoded_path = quote(encoded_path, safe="")
    unicode_escaped_path = absolute_path.replace("\\", r"\u005c").replace(
        ":", r"\u003a"
    )
    sibling_path = rf"{root}_backup\retained.usd"
    payload = {
        "output_path": absolute_path,
        "generated_files": [mixed_path, encoded_path],
        "error_message": f"failed for {absolute_path}",
        "case_variant_message": f"failed for {case_variant}",
        "escaped_message": f"failed for {escaped_path}",
        "encoded_message": f"failed for {encoded_path}",
        "double_encoded_message": f"failed for {double_encoded_path}",
        "unicode_escaped_message": f"failed for {unicode_escaped_path}",
        "sibling_path_message": sibling_path,
        "target_prim_path": "/World/Tire",
    }

    sanitized = sanitize_public_response_payload(payload, session_roots=(root,))

    session_uri = f"session://{session_id}/cache/optimized/input.usd"
    assert sanitized["output_path"] == session_uri
    assert sanitized["generated_files"] == [
        f"session://{session_id}/cache/input.usd",
        session_uri,
    ]
    assert sanitized["error_message"] == "failed for <session>"
    assert sanitized["case_variant_message"] == "failed for <session>"
    assert sanitized["escaped_message"] == "failed for <session>"
    assert sanitized["encoded_message"] == "failed for <session>"
    assert sanitized["double_encoded_message"] == "failed for <session>"
    assert sanitized["unicode_escaped_message"] == "failed for <session>"
    assert sanitized["sibling_path_message"] == sibling_path
    assert sanitized["target_prim_path"] == "/World/Tire"


def test_public_payload_sanitizes_lowercase_encoded_unicode_posix_root() -> None:
    root = "/var/séssions"
    absolute_path = f"{root}/session-id/cache/input.usd"
    encoded_path = quote(absolute_path, safe="")
    lowercase_encoded_path = (
        encoded_path.replace("%2F", "%2f").replace("%C3", "%c3").replace("%A9", "%a9")
    )
    case_variant = absolute_path.replace("/séssions/", "/Séssions/")

    sanitized = sanitize_public_response_payload(
        {
            "error_message": f"failed for {lowercase_encoded_path}",
            "case_variant_message": f"failed for {case_variant}",
        },
        session_roots=(root,),
    )

    assert sanitized["error_message"] == "failed for <session>"
    assert sanitized["case_variant_message"] == f"failed for {case_variant}"


def test_public_payload_sanitizes_windows_unc_session_paths_portably() -> None:
    session_id = "12345678-1234-1234-1234-123456789abc"
    root = r"\\render-share\content agent\sessions"
    absolute_path = rf"{root}\{session_id}\cache\input.usd"
    mixed_path = rf"//render-share/content agent\sessions/{session_id}/cache\input.usd"
    payload = {
        "source_file": mixed_path,
        "error": f"unable to read {absolute_path}",
    }

    sanitized = sanitize_public_response_payload(payload, session_roots=(root,))

    session_uri = f"session://{session_id}/cache/input.usd"
    assert sanitized == {
        "source_file": session_uri,
        "error": "unable to read <session>",
    }


def test_public_payload_preserves_literal_percent_sequences_in_session_root() -> None:
    root = r"C:\sessions%20archive"
    absolute_path = rf"{root}\session-id\cache\input.usd"
    encoded_path = quote(absolute_path, safe="")

    sanitized = sanitize_public_response_payload(
        {
            "paths": [absolute_path, encoded_path],
            "messages": [absolute_path, encoded_path],
        },
        session_roots=(root,),
    )

    session_uri = "session://session-id/cache/input.usd"
    assert sanitized == {
        "paths": [session_uri, session_uri],
        "messages": ["<session>", "<session>"],
    }


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


@pytest.mark.asyncio
async def test_sse_middleware_sanitizes_lowercase_unicode_escaped_raw_path() -> None:
    escaped_path = r"/var/material\u002dagent/sessions/session-id/cache/output.usd"
    event = f"data: failed for {escaped_path}\n\n".encode()

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await send({"type": "http.response.body", "body": event})

    middleware = PublicJsonResponseSanitizationMiddleware(cast(ASGIApp, app))
    messages = await _invoke_middleware(middleware)

    body = b"".join(message.get("body", b"") for message in messages[1:])
    assert body == b"data: failed for <session>\n\n"
    assert b"material\\u002dagent" not in body


@pytest.mark.asyncio
async def test_sse_middleware_sanitizes_unicode_escaped_non_ascii_root() -> None:
    root = "/var/matérial-agent/sessions"
    escaped_path = r"/var/mat\u00e9rial-agent/sessions/session-id/cache/output.usd"
    event = f"data: failed for {escaped_path}\n\n".encode()

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await send({"type": "http.response.body", "body": event})

    middleware = PublicJsonResponseSanitizationMiddleware(
        cast(ASGIApp, app),
        session_roots=(root,),
    )
    messages = await _invoke_middleware(middleware)

    body = b"".join(message.get("body", b"") for message in messages[1:])
    assert body == b"data: failed for <session>\n\n"
    assert b"mat\\u00e9rial-agent" not in body


@pytest.mark.asyncio
async def test_sse_middleware_sanitizes_non_bmp_surrogate_escaped_root() -> None:
    root = "/var/material-\U0001f680-agent/sessions"
    escaped_path = (
        r"/var/material-\ud83d\ude80-agent/sessions/session-id/cache/output.usd"
    )
    event = f"data: failed for {escaped_path}\n\n".encode()

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await send({"type": "http.response.body", "body": event})

    middleware = PublicJsonResponseSanitizationMiddleware(
        cast(ASGIApp, app),
        session_roots=(root,),
    )
    messages = await _invoke_middleware(middleware)

    body = b"".join(message.get("body", b"") for message in messages[1:])
    assert body == b"data: failed for <session>\n\n"
    assert b"\\ud83d\\ude80" not in body.lower()


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
