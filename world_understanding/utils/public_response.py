# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanitize JSON emitted by public service response boundaries."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from ipaddress import ip_address
from pathlib import Path
from typing import Any, cast

from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_PUBLIC_JSON_RESPONSE_BYTES = 16 * 1024 * 1024
_SANITIZATION_FAILURE_BODY = b'{"detail":"Public JSON response sanitization failed"}'
_DOCKER_SESSION_ROOTS = (
    "/var/material-agent/sessions",
    "/var/joint-agent/sessions",
    "/var/physics-agent/sessions",
    "/var/texture-agent/sessions",
)
_INTERNAL_ENDPOINT_RE = re.compile(
    r"https?://(?P<host>\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9_.-]+)"
    r"(?::[0-9]+)?(?:[/?#][^\s'\"]*)?",
    re.IGNORECASE,
)
_HTTP_CONNECTION_POOL_RE = re.compile(
    r"\b(?:HTTP|HTTPS)ConnectionPool\(host=(?P<quote>['\"])"
    r"(?P<host>[^'\"]+)(?P=quote),\s*port=[^)]+\)",
    re.IGNORECASE,
)
_INTERNAL_DNS_SUFFIXES = (
    ".svc",
    ".svc.cluster.local",
    ".cluster.local",
    ".local",
    ".internal",
    ".localdomain",
)


def _is_internal_host(host: str) -> bool:
    """Classify service-only, NVCF, and non-global IP endpoint hosts."""
    normalized = host.strip("[]").rstrip(".").lower()
    try:
        return not ip_address(normalized).is_global
    except ValueError:
        pass
    if normalized == "localhost" or "." not in normalized:
        return True
    if normalized == "api.nvcf.nvidia.com" or normalized.endswith(
        ".invocation.api.nvcf.nvidia.com"
    ):
        return True
    if normalized.endswith(_INTERNAL_DNS_SUFFIXES):
        return True
    return False


def _redact_internal_endpoint(match: re.Match[str]) -> str:
    """Redact classified endpoint URLs while preserving public URLs."""
    return (
        "<internal-endpoint>"
        if _is_internal_host(match.group("host"))
        else match.group(0)
    )


def _redact_http_connection_pool(match: re.Match[str]) -> str:
    """Redact urllib3 connection-pool diagnostics for internal hosts."""
    return (
        "<internal-endpoint>"
        if _is_internal_host(match.group("host"))
        else match.group(0)
    )


def _normalized_roots(session_roots: Iterable[str | Path]) -> tuple[Path, ...]:
    """Resolve unique configured and container session roots longest-first."""
    roots: list[Path] = []
    for value in (*session_roots, *_DOCKER_SESSION_ROOTS):
        path = Path(value).resolve(strict=False)
        if path not in roots:
            roots.append(path)
    return tuple(sorted(roots, key=lambda path: len(str(path)), reverse=True))


def _root_redaction_patterns(roots: tuple[Path, ...]) -> tuple[re.Pattern[str], ...]:
    """Compile bounded descendant matchers for trusted session roots."""
    return tuple(
        re.compile(
            rf"{re.escape(str(root).rstrip('/'))}"
            rf"(?![A-Za-z0-9_.-])(?:/[^\s'\"]*)?"
        )
        for root in roots
    )


def _session_uri(value: str, roots: tuple[Path, ...]) -> str | None:
    """Project one absolute path under a session root to a session URI."""
    path = Path(value)
    if not path.is_absolute():
        return None
    normalized = path.resolve(strict=False)
    for root in roots:
        try:
            relative = normalized.relative_to(root)
        except ValueError:
            continue
        if not relative.parts:
            return "session://"
        session_id, *remainder = relative.parts
        suffix = "/".join(remainder)
        return (
            f"session://{session_id}/{suffix}" if suffix else f"session://{session_id}"
        )
    return None


def _sanitize_text(
    value: str,
    root_patterns: tuple[re.Pattern[str], ...],
) -> str:
    """Redact internal endpoints and session-root descendants from text."""
    cleaned = _INTERNAL_ENDPOINT_RE.sub(_redact_internal_endpoint, value)
    cleaned = _HTTP_CONNECTION_POOL_RE.sub(_redact_http_connection_pool, cleaned)
    for root_pattern in root_patterns:
        cleaned = root_pattern.sub("<session>", cleaned)
    return cleaned


def _sanitize_prepared_public_response_payload(
    payload: Any,
    *,
    roots: tuple[Path, ...],
    root_patterns: tuple[re.Pattern[str], ...],
) -> Any:
    """Sanitize a payload with precomputed session-root state."""

    def sanitize(value: Any, *, path_field: bool = False) -> Any:
        """Recursively detach and sanitize JSON-compatible containers."""
        if isinstance(value, str):
            if path_field:
                session_uri = _session_uri(value, roots)
                if session_uri is not None:
                    return session_uri
            return _sanitize_text(value, root_patterns)
        if isinstance(value, list):
            return [sanitize(item, path_field=path_field) for item in value]
        if isinstance(value, dict):
            return {
                key: sanitize(
                    child,
                    path_field=(
                        isinstance(key, str)
                        and (
                            key in {"path", "paths", "files"}
                            or key.endswith(("_path", "_paths", "_file", "_files"))
                        )
                        and not key.endswith("_prim_path")
                    ),
                )
                for key, child in value.items()
            }
        return value

    return sanitize(payload)


def sanitize_public_response_payload(
    payload: Any,
    *,
    session_roots: Iterable[str | Path] = (),
) -> Any:
    """Return a detached payload without service-local paths or hostnames."""
    roots = _normalized_roots(session_roots)
    return _sanitize_prepared_public_response_payload(
        payload,
        roots=roots,
        root_patterns=_root_redaction_patterns(roots),
    )


def _serialize_sanitized_json(
    body: bytes,
    *,
    roots: tuple[Path, ...],
    root_patterns: tuple[re.Pattern[str], ...],
) -> bytes:
    """Parse, sanitize, and compactly serialize one JSON value."""
    payload = json.loads(body)
    sanitized = _sanitize_prepared_public_response_payload(
        payload,
        roots=roots,
        root_patterns=root_patterns,
    )
    return json.dumps(
        sanitized,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class _StreamingRecordSanitizer:
    """Incrementally sanitize newline-delimited JSON or SSE data records."""

    def __init__(
        self,
        mode: str,
        *,
        roots: tuple[Path, ...],
        root_patterns: tuple[re.Pattern[str], ...],
        max_record_bytes: int,
    ) -> None:
        self.mode = mode
        self.roots = roots
        self.root_patterns = root_patterns
        self.max_record_bytes = max_record_bytes
        self.buffer = bytearray()
        self.discarding_oversized_record = False

    def _failure_record(self, ending: bytes = b"\n") -> bytes:
        """Return one format-valid fail-closed record."""
        if self.mode == "sse":
            return b"data: " + _SANITIZATION_FAILURE_BODY + ending
        return _SANITIZATION_FAILURE_BODY + ending

    def _sanitize_line(self, line: bytes) -> bytes:
        """Sanitize one complete or final partial streaming record."""
        if line.endswith(b"\r\n"):
            content, ending = line[:-2], b"\r\n"
        elif line.endswith(b"\n"):
            content, ending = line[:-1], b"\n"
        else:
            content, ending = line, b""

        if self.mode == "ndjson":
            if not content.strip():
                return line
            try:
                sanitized = _serialize_sanitized_json(
                    content,
                    roots=self.roots,
                    root_patterns=self.root_patterns,
                )
            except (
                UnicodeDecodeError,
                UnicodeEncodeError,
                json.JSONDecodeError,
                RecursionError,
                TypeError,
                ValueError,
            ):
                return self._failure_record(ending or b"\n")
            return sanitized + ending

        if not content.startswith(b"data:"):
            return line
        data = content[5:]
        separator = b""
        if data.startswith(b" "):
            separator, data = b" ", data[1:]
        if not data:
            return line
        try:
            sanitized = _serialize_sanitized_json(
                data,
                roots=self.roots,
                root_patterns=self.root_patterns,
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            try:
                sanitized = _sanitize_text(
                    data.decode("utf-8"),
                    self.root_patterns,
                ).encode("utf-8")
            except (UnicodeDecodeError, UnicodeEncodeError):
                sanitized = _SANITIZATION_FAILURE_BODY
        except (UnicodeEncodeError, RecursionError, TypeError, ValueError):
            sanitized = _SANITIZATION_FAILURE_BODY
        return b"data:" + separator + sanitized + ending

    def feed(self, chunk: bytes, *, final: bool) -> bytes:
        """Consume a response chunk and return sanitized complete records."""
        output = bytearray()
        if self.discarding_oversized_record:
            record_end = chunk.find(b"\n")
            if record_end < 0:
                return b""
            chunk = chunk[record_end + 1 :]
            self.discarding_oversized_record = False

        self.buffer.extend(chunk)
        while True:
            record_end = self.buffer.find(b"\n")
            if record_end < 0:
                break
            line = bytes(self.buffer[: record_end + 1])
            del self.buffer[: record_end + 1]
            if len(line) > self.max_record_bytes:
                output.extend(self._failure_record())
            else:
                output.extend(self._sanitize_line(line))

        if final:
            if self.buffer:
                if len(self.buffer) > self.max_record_bytes:
                    output.extend(self._failure_record())
                else:
                    output.extend(self._sanitize_line(bytes(self.buffer)))
                self.buffer.clear()
            self.discarding_oversized_record = False
        elif len(self.buffer) > self.max_record_bytes:
            output.extend(self._failure_record())
            self.buffer.clear()
            self.discarding_oversized_record = True

        return bytes(output)


def _media_type(content_type: bytes) -> bytes:
    """Return a normalized response media type without parameters."""
    return content_type.partition(b";")[0].strip().lower()


def _is_json_media_type(media_type: bytes) -> bool:
    """Return whether a media type contains one complete JSON value."""
    return media_type == b"application/json" or media_type.endswith(b"+json")


class PublicJsonResponseSanitizationMiddleware:
    """Sanitize public JSON, NDJSON, and SSE response boundaries."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        session_roots: Iterable[str | Path] = (),
        max_body_bytes: int = MAX_PUBLIC_JSON_RESPONSE_BYTES,
    ) -> None:
        """Precompute fixed sanitization state for one service application."""
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self.roots = _normalized_roots(session_roots)
        self.root_patterns = _root_redaction_patterns(self.roots)
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Sanitize structured responses and fail closed on unsafe ranges."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_message: Message | None = None
        body_chunks: list[bytes] = []
        body_size = 0
        body_overflow = False
        sanitize_json = False
        encoded_response_blocked = False
        range_response_blocked = False
        stream_sanitizer: _StreamingRecordSanitizer | None = None

        async def send_sanitized(message: Message) -> None:
            """Intercept one ASGI response message."""
            nonlocal body_overflow, body_size, encoded_response_blocked
            nonlocal range_response_blocked
            nonlocal sanitize_json, start_message, stream_sanitizer
            if message["type"] == "http.response.start":
                headers = {
                    key.lower(): value for key, value in message.get("headers", [])
                }
                content_type = headers.get(b"content-type", b"").lower()
                media_type = _media_type(content_type)
                content_encoding = headers.get(b"content-encoding", b"").strip().lower()
                status = int(message["status"])
                body_forbidden = (
                    100 <= status < 200
                    or status in {204, 205, 304}
                    or scope.get("method") == "HEAD"
                )
                range_response_blocked = status == 206 and (
                    _is_json_media_type(media_type)
                    or media_type == b"application/x-ndjson"
                    or media_type == b"text/event-stream"
                    or media_type == b"multipart/byteranges"
                )
                structured_media_type = (
                    _is_json_media_type(media_type)
                    or media_type == b"application/x-ndjson"
                    or media_type == b"text/event-stream"
                )
                encoded_response_blocked = (
                    not body_forbidden
                    and structured_media_type
                    and content_encoding not in {b"", b"identity"}
                )
                if range_response_blocked or encoded_response_blocked:
                    blocked_start = dict(message)
                    blocked_start["status"] = 416 if range_response_blocked else 500
                    blocked_start["headers"] = [
                        (key, value)
                        for key, value in message.get("headers", [])
                        if key.lower()
                        not in {
                            b"accept-ranges",
                            b"content-length",
                            b"content-md5",
                            b"content-range",
                            b"content-type",
                            b"content-encoding",
                            b"etag",
                        }
                    ]
                    blocked_start["headers"].extend(
                        [
                            (b"content-type", b"application/json"),
                            (
                                b"content-length",
                                str(len(_SANITIZATION_FAILURE_BODY)).encode("ascii"),
                            ),
                        ]
                    )
                    await send(blocked_start)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": _SANITIZATION_FAILURE_BODY,
                        }
                    )
                    return

                stream_mode = None
                if media_type == b"application/x-ndjson":
                    stream_mode = "ndjson"
                elif media_type == b"text/event-stream":
                    stream_mode = "sse"
                if (
                    not body_forbidden
                    and stream_mode is not None
                    and content_encoding in {b"", b"identity"}
                ):
                    stream_sanitizer = _StreamingRecordSanitizer(
                        stream_mode,
                        roots=self.roots,
                        root_patterns=self.root_patterns,
                        max_record_bytes=self.max_body_bytes,
                    )
                    stream_start = dict(message)
                    stream_start["headers"] = [
                        (key, value)
                        for key, value in message.get("headers", [])
                        if key.lower()
                        not in {b"content-length", b"content-md5", b"etag"}
                    ]
                    await send(stream_start)
                    return

                sanitize_json = (
                    not body_forbidden
                    and _is_json_media_type(media_type)
                    and content_encoding in {b"", b"identity"}
                )
                if sanitize_json:
                    start_message = message
                else:
                    await send(message)
                return

            if range_response_blocked or encoded_response_blocked:
                return

            if message["type"] != "http.response.body":
                await send(message)
                return

            if stream_sanitizer is not None:
                stream_message = dict(message)
                stream_message["body"] = stream_sanitizer.feed(
                    message.get("body", b""),
                    final=not message.get("more_body", False),
                )
                await send(stream_message)
                return

            if not sanitize_json:
                await send(message)
                return

            chunk = message.get("body", b"")
            body_size += len(chunk)
            if body_size > self.max_body_bytes:
                body_overflow = True
                body_chunks.clear()
            elif not body_overflow:
                body_chunks.append(chunk)
            if message.get("more_body", False):
                return

            processing_failed = body_overflow
            if body_overflow:
                body = _SANITIZATION_FAILURE_BODY
            else:
                body = b"".join(body_chunks)
                try:
                    body = _serialize_sanitized_json(
                        body,
                        roots=self.roots,
                        root_patterns=self.root_patterns,
                    )
                except (
                    UnicodeDecodeError,
                    UnicodeEncodeError,
                    json.JSONDecodeError,
                    RecursionError,
                    TypeError,
                    ValueError,
                ):
                    processing_failed = True
                    body = _SANITIZATION_FAILURE_BODY

            if start_message is None:
                raise RuntimeError("JSON response body arrived before response start")
            start_message = dict(start_message)
            if processing_failed:
                start_message["status"] = 500
            start_message["headers"] = [
                (key, value)
                for key, value in start_message.get("headers", [])
                if key.lower() not in {b"content-length", b"content-md5", b"etag"}
            ]
            start_message["headers"].append(
                (b"content-length", str(len(body)).encode("ascii"))
            )
            await send(start_message)
            await send({"type": "http.response.body", "body": body})

        app_scope = scope
        extensions = scope.get("extensions")
        if extensions and "http.response.pathsend" in extensions:
            app_scope = cast(Scope, dict(scope))
            app_scope["extensions"] = {
                name: value
                for name, value in extensions.items()
                if name != "http.response.pathsend"
            }
        await self.app(app_scope, receive, send_sanitized)
