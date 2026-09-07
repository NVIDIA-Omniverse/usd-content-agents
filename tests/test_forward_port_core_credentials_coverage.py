# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward-port regressions for credential and public-response hardening edges."""

from __future__ import annotations

import errno
import json
import traceback
from pathlib import Path
from typing import cast

import pytest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from world_understanding.utils import credentials, public_response


def _percent_encode_every_byte(value: str, rounds: int) -> str:
    for _ in range(rounds):
        value = "".join(f"%{byte:02X}" for byte in value.encode())
    return value


def _stream_sanitizer(
    mode: str, *, max_record_bytes: int = 1024
) -> public_response._StreamingRecordSanitizer:
    roots = public_response._normalized_roots(())
    return public_response._StreamingRecordSanitizer(
        mode,
        roots=roots,
        root_patterns=public_response._root_redaction_patterns(roots),
        max_record_bytes=max_record_bytes,
    )


def test_credential_key_and_reference_edge_classification() -> None:
    # A compact ``tokenv2`` rotation suffix is still a credential field, while
    # empty path/reference keys and non-string metadata keys remain ordinary.
    assert credentials._is_sensitive_config_key("tokenv2")
    assert not credentials._is_path_reference_key("")
    assert not credentials._credential_container_scalar_is_reference(7, "value")
    assert not credentials._credential_container_scalar_is_reference("", "value")


def test_terminal_percent_decoding_is_bounded_and_fail_closed() -> None:
    assert credentials._decode_nested_percent_bytes_once("%25") == ("%", True)
    assert credentials._decode_nested_percent_bytes_once("%GG") == ("%GG", False)

    deeply_encoded_percent = _percent_encode_every_byte(
        "%", credentials._MAX_URI_DECODE_ROUNDS + 1
    )
    budget = credentials._ScanBudget(credentials.DEFAULT_CREDENTIAL_SCAN_LIMITS)
    canonical, exhausted = credentials._canonicalize_terminal_percent_encoding(
        deeply_encoded_percent,
        budget=budget,
    )
    assert canonical == "%25"
    assert exhausted is True
    assert budget.decode_work > 0


def test_nested_uri_query_and_fragment_credentials_fail_closed() -> None:
    assert credentials._contains_uri_candidate("https://example.test/public")
    assert credentials._url_component_has_inline_secret(
        "redirect=https://user:opaque-secret@example.test/private",
        depth=credentials._MAX_URI_NESTING_DEPTH,
    )

    secret = "query-fragment-secret-818"
    query_uri = f"https://example.test/object?access_token={secret}"
    fragment_uri = f"https://example.test/object#access_token={secret}"
    assert credentials._is_url_with_inline_secret(query_uri)
    assert credentials._is_url_with_inline_secret(fragment_uri)
    assert credentials.find_inline_secret_paths(
        {"query_uri": query_uri, "fragment_uri": fragment_uri}
    ) == ("query_uri", "fragment_uri")


def test_normalized_path_with_unresolved_authority_encoding_fails_closed() -> None:
    encoded_percent = _percent_encode_every_byte(
        "%", credentials._MAX_URI_DECODE_ROUNDS + 1
    )
    ambiguous_path = f"https:/example{encoded_percent}.test/object"

    assert credentials._path_text_has_inline_secret(ambiguous_path)
    assert credentials.redact_sensitive_path(ambiguous_path) == "<redacted>"


def test_authorization_placeholders_and_malformed_assignments_remain_safe() -> None:
    assert not credentials._has_inline_authorization("Bearer redacted")
    assert not credentials._has_inline_sensitive_assignment(r"'\x': opaque-secret")

    oversized_key = '"' + ("a" * 129) + '": opaque-secret'
    assert not credentials._has_inline_sensitive_assignment(oversized_key)


def test_deeply_encoded_sensitive_assignment_fails_closed() -> None:
    encoded_colon = _percent_encode_every_byte(
        ":", (credentials._MAX_URI_DECODE_ROUNDS * 2) + 1
    )
    secret = "deep-assignment-secret-818"
    serialized = f"api_key{encoded_colon}{secret}"

    assert credentials._has_inline_sensitive_assignment(serialized)
    assert credentials.find_inline_secret_paths({"description": serialized}) == (
        "description",
    )


def test_encoded_pem_separators_and_boundary_rejections() -> None:
    assert credentials._consume_pem_separator("%2520rest", 0, allow_encoded=True) == (
        5,
        True,
    )
    assert credentials._consume_pem_separator("%2521rest", 0, allow_encoded=True) == (
        0,
        False,
    )

    begin = credentials._PEM_BEGIN_PREFIX
    assert (
        credentials._parse_pem_boundary(
            f"{begin} -----", 0, begin, allow_encoded_whitespace=False
        )
        is None
    )
    assert (
        credentials._parse_pem_boundary(
            f"{begin} PRIVATE!KEY-----",
            0,
            begin,
            allow_encoded_whitespace=False,
        )
        is None
    )
    assert (
        credentials._parse_pem_boundary(
            f"{begin} {'A' * (credentials._MAX_PEM_LABEL_LENGTH + 1)}-----",
            0,
            begin,
            allow_encoded_whitespace=False,
        )
        is None
    )
    assert (
        credentials._parse_pem_boundary(
            f"{begin} PRIVATE KEY",
            0,
            begin,
            allow_encoded_whitespace=False,
        )
        is None
    )


def test_private_key_placeholder_body_boundaries_are_exact() -> None:
    assert not credentials._pem_body_is_explicit_placeholder(
        "x" * (credentials._MAX_PEM_PLACEHOLDER_BODY_LENGTH + 1)
    )
    assert credentials._pem_body_is_explicit_placeholder("\n > \n")
    assert not credentials._pem_body_is_explicit_placeholder("<redacted>\n${KEY}")

    quoted_placeholder = (
        "> -----BEGIN PRIVATE KEY-----\n> <redacted>\n> -----END PRIVATE KEY-----"
    )
    assert credentials._is_explicit_private_key_placeholder_value(quoted_placeholder)
    assert not credentials._is_explicit_private_key_placeholder_value(
        "-----BEGIN CERTIFICATE-----\n<redacted>\n-----END CERTIFICATE-----"
    )
    assert not credentials._is_explicit_private_key_placeholder_value(
        "-----BEGIN PRIVATE KEY-----\n<redacted>"
    )

    invalid_end_then_valid_end = (
        "-----BEGIN PRIVATE KEY-----\n<redacted>\n"
        "-----NOT AN END-----\n-----END PRIVATE KEY-----"
    )
    assert not credentials._is_explicit_private_key_placeholder_value(
        invalid_end_then_valid_end
    )
    assert not credentials._is_explicit_private_key_placeholder_value(
        "-----BEGIN RSA PRIVATE KEY-----\n<redacted>\n-----END PRIVATE KEY-----"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("no marker", False),
        ("-----NOT A BEGIN-----", False),
        ("-----BEGIN!", False),
        ("-----BEGIN%20CERTIFICATE-----", False),
        ("-----BEGIN%20PRIVATE%20KEY-----%20", True),
    ],
)
def test_unresolved_encoded_private_key_markers_fail_closed(
    value: str, expected: bool
) -> None:
    assert credentials._has_unresolved_encoded_private_key_marker(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("user:opaque-secret@", False),
        ("user:opaque-secret@[::1]", True),
        ("user:opaque-secret@[broken/path", False),
    ],
)
def test_authorityless_userinfo_requires_a_complete_host(
    value: str, expected: bool
) -> None:
    assert credentials._has_userinfo_without_authority(value) is expected


def test_diagnostic_mapping_paths_are_value_free_for_unusual_keys() -> None:
    class OpaqueKey:
        pass

    assert credentials._diagnostic_mapping_path("root", True, 0) == "root[bool:true]"
    assert credentials._diagnostic_mapping_path("root", None, 1) == "root[null]"
    assert credentials._diagnostic_mapping_path("root", OpaqueKey(), 2) == (
        "root[key#2]"
    )

    secret = "mapping-key-secret-818"
    findings = credentials.find_inline_secret_paths(
        {True: {"api_key": secret}, None: {"api_key": secret}}
    )
    assert findings == ("[bool:true].api_key", "[null].api_key")
    assert secret not in repr(findings)


def test_recursive_and_unordered_credential_containers_are_safe() -> None:
    recursive_list: list[object] = []
    recursive_list.append(recursive_list)
    assert credentials.find_inline_secret_paths(recursive_list) == ()
    redacted_list = credentials.redact_sensitive_config(recursive_list)
    assert redacted_list[0] is redacted_list

    secret = "unordered-container-secret-818"
    config = {"credentials": {"slot": {secret}}}
    assert credentials.find_inline_secret_paths(config) == ("credentials.slot",)
    assert credentials.redact_sensitive_config(config) == {
        "credentials": {"slot": "<redacted>"}
    }
    assert secret not in repr(credentials.redact_sensitive_config(config))


def test_redaction_preserves_shared_and_recursive_tuple_identity() -> None:
    shared_tuple = ("safe",)
    redacted_shared = credentials.redact_sensitive_config([shared_tuple, shared_tuple])
    assert redacted_shared[0] is redacted_shared[1]

    holder: list[object] = []
    recursive_tuple = (holder,)
    holder.append(recursive_tuple)
    redacted_tuple = credentials.redact_sensitive_config(recursive_tuple)
    assert redacted_tuple[0][0] is redacted_tuple


def test_safe_path_resolution_oserror_does_not_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "resolve-path-secret-818"
    value = Path(f"cache/user:{secret}@example.test/result.usd")

    def fail_resolve(_path: Path, strict: bool = False) -> Path:
        del strict
        raise OSError(errno.ELOOP, "symlink loop", str(value))

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    with pytest.raises(OSError) as exc_info:
        credentials.resolve_path_with_safe_diagnostics(value, label="result path")

    observable = "".join(traceback.format_exception(exc_info.value))
    assert exc_info.value.errno == errno.ELOOP
    assert exc_info.value.filename == "<redacted>"
    assert secret not in observable
    assert "Unable to resolve result path" in observable


def test_safe_directory_runtime_failure_does_not_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "mkdir-path-secret-818"
    value = Path(f"cache/user:{secret}@example.test/output")

    def fail_mkdir(
        _path: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
    ) -> None:
        del mode, parents, exist_ok
        raise RuntimeError(f"cannot create {value}")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    with pytest.raises(RuntimeError) as exc_info:
        credentials.create_directory_with_safe_diagnostics(
            value, label="output directory"
        )

    observable = "".join(traceback.format_exception(exc_info.value))
    assert str(exc_info.value) == "Unable to create output directory: <redacted>"
    assert secret not in observable
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_public_path_projection_handles_relative_and_root_paths() -> None:
    root = "/var/physics-agent/sessions"
    payload = {"path": "relative/result.usd", "output_path": root}

    assert public_response.sanitize_public_response_payload(
        payload, session_roots=(root,)
    ) == {"path": "relative/result.usd", "output_path": "session://"}


def test_stream_failure_records_and_partial_lines_are_format_valid() -> None:
    failure = public_response._SANITIZATION_FAILURE_BODY
    ndjson = _stream_sanitizer("ndjson")
    sse = _stream_sanitizer("sse")

    assert ndjson._failure_record(b"\r\n") == failure + b"\r\n"
    assert sse._failure_record() == b"data: " + failure + b"\n"
    assert ndjson._sanitize_line(b'{"ok":true}') == b'{"ok":true}'
    assert ndjson._sanitize_line(b"\n") == b"\n"
    assert ndjson._sanitize_line(b"not-json\n") == failure + b"\n"
    assert sse._sanitize_line(b"data:\n") == b"data:\n"


def test_sse_invalid_json_fallback_redacts_or_fails_closed() -> None:
    failure = public_response._SANITIZATION_FAILURE_BODY
    sanitizer = _stream_sanitizer("sse")
    internal = b"http://renderer.internal:8000/private"

    assert sanitizer._sanitize_line(b"data: " + internal + b"\n") == (
        b"data: <internal-endpoint>\n"
    )
    assert sanitizer._sanitize_line(b"data: \xff\n") == (b"data: " + failure + b"\n")
    assert internal not in sanitizer._sanitize_line(b"data: " + internal + b"\n")


def test_sse_serialization_value_error_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = b"serialization-secret-818"

    def fail_serialization(*_args: object, **_kwargs: object) -> bytes:
        raise ValueError(secret.decode())

    monkeypatch.setattr(
        public_response, "_serialize_sanitized_json", fail_serialization
    )
    output = _stream_sanitizer("sse")._sanitize_line(
        b'data: {"detail":"' + secret + b'"}\n'
    )
    assert output == b"data: " + public_response._SANITIZATION_FAILURE_BODY + b"\n"
    assert secret not in output


def test_stream_record_size_bounds_discard_and_recover() -> None:
    failure = public_response._SANITIZATION_FAILURE_BODY + b"\n"

    buffered = _stream_sanitizer("ndjson", max_record_bytes=4)
    assert buffered.feed(b"12345", final=False) == failure
    assert buffered.feed(b"still-oversized", final=False) == b""
    assert buffered.feed(b"discarded\n{}\n", final=False) == b"{}\n"

    complete = _stream_sanitizer("ndjson", max_record_bytes=4)
    assert complete.feed(b"12345\n", final=False) == failure

    final_oversized = _stream_sanitizer("ndjson", max_record_bytes=4)
    assert final_oversized.feed(b"12345", final=True) == failure

    final_partial = _stream_sanitizer("ndjson", max_record_bytes=4)
    assert final_partial.feed(b"{}", final=True) == b"{}"


async def _invoke_non_http_or_auxiliary_message(
    app: ASGIApp, scope: Scope
) -> list[Message]:
    messages: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    middleware = public_response.PublicJsonResponseSanitizationMiddleware(app)
    await middleware(scope, receive, send)
    return messages


@pytest.mark.asyncio
async def test_public_middleware_passes_through_non_http_scope() -> None:
    async def app(scope: Scope, _receive: Receive, send: Send) -> None:
        assert scope["type"] == "lifespan"
        await send({"type": "lifespan.startup.complete"})

    messages = await _invoke_non_http_or_auxiliary_message(
        cast(ASGIApp, app), cast(Scope, {"type": "lifespan"})
    )
    assert messages == [{"type": "lifespan.startup.complete"}]


@pytest.mark.asyncio
async def test_public_middleware_passes_through_auxiliary_http_message() -> None:
    auxiliary: Message = {
        "type": "http.response.trailers",
        "headers": [(b"x-public", b"ok")],
        "more_trailers": False,
    }

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send(auxiliary)

    scope = cast(Scope, {"type": "http", "method": "GET", "extensions": {}})
    messages = await _invoke_non_http_or_auxiliary_message(cast(ASGIApp, app), scope)
    assert messages[0]["status"] == 200
    assert messages[1] == auxiliary


def test_public_failure_body_never_contains_test_secrets() -> None:
    # Keep a final audit assertion close to the direct private-helper coverage:
    # all failure projections are fixed public JSON, never exception text.
    body = json.loads(public_response._SANITIZATION_FAILURE_BODY)
    assert body == {"detail": "Public JSON response sanitization failed"}
    assert "secret-818" not in repr(body)
