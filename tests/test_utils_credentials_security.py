# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
import json
import threading
import traceback
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

import pytest
import yaml

from world_understanding.utils import credentials as credentials_module
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    confined_cleanup_path,
    is_pipeline_temp_path,
    open_confined_directory,
    open_held_confined_artifact,
    remove_legacy_pipeline_temp,
    visible_local_artifact_key,
    write_bytes_to_confined,
)
from world_understanding.utils.credentials import (
    CREDENTIAL_SCAN_LIMIT_MESSAGE,
    DEFAULT_CREDENTIAL_SCAN_LIMITS,
    CredentialScanLimitError,
    CredentialScanLimits,
    InlineSecretError,
    aensure_no_inline_secrets,
    ensure_no_inline_secrets,
    find_inline_secret_paths,
    format_env_reference,
    parse_env_reference,
    path_exists_with_safe_diagnostics,
    path_is_file_with_safe_diagnostics,
    read_text_with_safe_diagnostics,
    redact_sensitive_config,
    redact_sensitive_path,
    resolve_endpoint_api_key,
)
from world_understanding.utils.session_paths import (
    confined_session_path,
    confined_storage_child_path,
    safe_listed_session_ids,
)


def _percent_encode(value: str, rounds: int) -> str:
    for _ in range(rounds):
        value = quote(value, safe="")
    return value


def _percent_encode_all_bytes(value: str) -> str:
    return "".join(f"%{byte:02X}" for byte in value.encode())


def _assert_scan_within_declared_byte_budget(
    small_value: str,
    large_value: str,
    *,
    expected_paths: tuple[str, ...],
) -> None:
    def scan(value: str) -> tuple[str, ...]:
        byte_budget = len(b"description") + len(
            value.encode("utf-8", errors="surrogatepass")
        )
        return find_inline_secret_paths(
            {"description": value},
            limits=replace(DEFAULT_CREDENTIAL_SCAN_LIMITS, max_bytes=byte_budget),
        )

    assert scan(small_value) == expected_paths
    assert scan(large_value) == expected_paths


@pytest.mark.parametrize(
    ("value", "limits"),
    [
        ("safe", replace(DEFAULT_CREDENTIAL_SCAN_LIMITS, max_nodes=0)),
        ("safe", replace(DEFAULT_CREDENTIAL_SCAN_LIMITS, max_bytes=0)),
        ("safe", replace(DEFAULT_CREDENTIAL_SCAN_LIMITS, max_decode_work=0)),
        (
            "https://example.test/path",
            replace(DEFAULT_CREDENTIAL_SCAN_LIMITS, max_uri_candidates=0),
        ),
        (
            "https://example.test/?a=1&b=2",
            replace(DEFAULT_CREDENTIAL_SCAN_LIMITS, max_query_branches=1),
        ),
        (
            [["safe"]],
            replace(DEFAULT_CREDENTIAL_SCAN_LIMITS, max_container_depth=1),
        ),
    ],
    ids=["nodes", "bytes", "decode", "uri", "query", "depth"],
)
def test_total_scan_budget_exhaustion_is_constant_and_fail_closed(
    value: object,
    limits: CredentialScanLimits,
) -> None:
    with pytest.raises(CredentialScanLimitError) as exc_info:
        find_inline_secret_paths(value, limits=limits)

    assert str(exc_info.value) == CREDENTIAL_SCAN_LIMIT_MESSAGE


def test_structured_credential_containers_allow_references_but_reject_live_leaves() -> (
    None
):
    safe = {
        "credentials": {
            "primary": {"env": "${NVIDIA_API_KEY}", "provider": "nim"},
            "fallback": {"ref": "managed/nim-key"},
        }
    }
    assert find_inline_secret_paths(safe) == ()
    assert redact_sensitive_config(safe) == safe

    assert find_inline_secret_paths(
        {"credentials": {"primary": {"value": "live-secret"}}}
    ) == ("credentials.primary.value",)
    assert find_inline_secret_paths(
        {"credentials": {"primary": {"api_key": "live-secret"}}}
    ) == ("credentials.primary.api_key",)

    for container in (["shortsecret"], ("shortsecret",)):
        config = {"credentials": container}
        assert find_inline_secret_paths(config) == ("credentials[0]",)
        assert redact_sensitive_config(config) == {
            "credentials": type(container)(("<redacted>",))
        }


def test_heterogeneous_mapping_keys_have_distinct_canonical_paths() -> None:
    config = {
        1: {"api_key": "integer-key-secret"},
        "1": {"api_key": "string-key-secret"},
    }

    assert find_inline_secret_paths(config) == (
        "[1].api_key",
        '["1"].api_key',
    )

    huge_key = 1 << 100_000
    assert find_inline_secret_paths({huge_key: {"api_key": "huge-key-secret"}}) == (
        "[int#0].api_key",
    )


def test_scanner_and_redactor_are_cycle_safe_and_share_fail_closed_limits() -> None:
    config: dict[str, object] = {"api_key": "cycle-secret"}
    config["self"] = config

    assert find_inline_secret_paths(config) == ("api_key",)
    redacted = redact_sensitive_config(config)
    assert redacted["api_key"] == "<redacted>"
    assert redacted["self"] is redacted

    with pytest.raises(CredentialScanLimitError) as exc_info:
        redact_sensitive_config(
            config,
            limits=replace(DEFAULT_CREDENTIAL_SCAN_LIMITS, max_nodes=0),
        )
    assert str(exc_info.value) == CREDENTIAL_SCAN_LIMIT_MESSAGE


@pytest.mark.parametrize(
    ("limit_name", "counter_name", "config"),
    [
        ("max_nodes", "nodes", {"description": "safe"}),
        ("max_bytes", "bytes", {"description": "safe"}),
        ("max_decode_work", "decode_work", {"description": "safe"}),
        (
            "max_uri_candidates",
            "uri_candidates",
            {"description": "https://example.test/path"},
        ),
        (
            "max_query_branches",
            "query_branches",
            {"description": "https://example.test/?a=1&b=2"},
        ),
    ],
    ids=["nodes", "bytes", "decode", "uri", "query"],
)
def test_redaction_scan_and_projection_share_one_aggregate_budget(
    limit_name: str,
    counter_name: str,
    config: dict[str, str],
) -> None:
    scan_budget = credentials_module._ScanBudget(DEFAULT_CREDENTIAL_SCAN_LIMITS)
    assert (
        credentials_module.find_inline_secret_paths(
            config,
            _scan_budget=scan_budget,
        )
        == ()
    )
    scan_only_work = getattr(scan_budget, counter_name)
    assert scan_only_work > 0
    exact_scan_limits = replace(
        DEFAULT_CREDENTIAL_SCAN_LIMITS,
        **{limit_name: scan_only_work},
    )

    # The public scanner fits exactly, but redaction must charge its repeated
    # traversal and detector work to that same counter instead of resetting it.
    assert find_inline_secret_paths(config, limits=exact_scan_limits) == ()
    with pytest.raises(CredentialScanLimitError) as exc_info:
        redact_sensitive_config(config, limits=exact_scan_limits)
    assert str(exc_info.value) == CREDENTIAL_SCAN_LIMIT_MESSAGE


class _ExpandingRedactionMapping(Mapping[str, object]):
    """Expose deeper data only when redaction performs its second traversal."""

    def __init__(self) -> None:
        self._reads = 0

    def __getitem__(self, key: str) -> object:
        if key != "value":
            raise KeyError(key)
        self._reads += 1
        return "safe" if self._reads == 1 else [["safe"]]

    def __iter__(self) -> Iterator[str]:
        return iter(("value",))

    def __len__(self) -> int:
        return 1


def test_redaction_projection_enforces_depth_after_preflight() -> None:
    limits = replace(DEFAULT_CREDENTIAL_SCAN_LIMITS, max_container_depth=1)
    assert (
        find_inline_secret_paths(
            _ExpandingRedactionMapping(),
            limits=limits,
        )
        == ()
    )

    with pytest.raises(CredentialScanLimitError) as exc_info:
        redact_sensitive_config(_ExpandingRedactionMapping(), limits=limits)
    assert str(exc_info.value) == CREDENTIAL_SCAN_LIMIT_MESSAGE


def test_redaction_preserves_cycles_and_structured_credential_references() -> None:
    config: dict[str, object] = {
        "credentials": {
            "primary": {"env": "${NVIDIA_API_KEY}", "provider": "nim"},
            "fallback": {"value": "live-secret"},
        }
    }
    config["self"] = config

    redacted = redact_sensitive_config(config)

    assert redacted["self"] is redacted
    assert redacted["credentials"]["primary"] == {
        "env": "${NVIDIA_API_KEY}",
        "provider": "nim",
    }
    assert redacted["credentials"]["fallback"] == {"value": "<redacted>"}


@pytest.mark.asyncio
async def test_async_scanner_helper_runs_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_thread_id = threading.get_ident()
    worker_thread_ids: list[int] = []
    started = threading.Event()
    release = threading.Event()

    def blocking_guard(*_args: object, **_kwargs: object) -> None:
        worker_thread_ids.append(threading.get_ident())
        started.set()
        release.wait(timeout=2)

    monkeypatch.setattr(
        credentials_module,
        "ensure_no_inline_secrets",
        blocking_guard,
    )
    task = asyncio.create_task(aensure_no_inline_secrets({"value": "safe"}))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 1), timeout=2)
        assert not task.done()
        assert worker_thread_ids != [loop_thread_id]
    finally:
        release.set()
    await task


def test_redact_sensitive_path_preserves_runtime_text_but_hides_credentials() -> None:
    benign_path = Path("assets/models/chair.usd")
    secret = "path-helper-secret-token-713"
    credential_path = Path(f"cache/user:{secret}@assets.example.test/model.usd")

    assert redact_sensitive_path(benign_path) == str(benign_path)
    assert redact_sensitive_path(credential_path) == "<redacted>"
    assert secret not in redact_sensitive_path(credential_path)


def test_safe_path_exists_preserves_oserror_semantics_without_path_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "exists-helper-secret-713"
    credential_path = Path(f"cache/user:{secret}@assets.example.test/model.usd")

    def raise_name_too_long(path: Path) -> bool:
        raise OSError(errno.ENAMETOOLONG, "File name too long", str(path))

    monkeypatch.setattr(Path, "exists", raise_name_too_long)

    with pytest.raises(OSError) as exc_info:
        path_exists_with_safe_diagnostics(
            credential_path,
            label="credential-bearing artifact",
        )

    assert exc_info.value.errno == errno.ENAMETOOLONG
    assert exc_info.value.filename == "<redacted>"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    observable = "".join(traceback.format_exception(exc_info.value))
    assert secret not in observable
    assert "<redacted>" in observable


def test_safe_path_is_file_preserves_raw_io_and_value_safe_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "is-file-helper-secret-713"
    credential_path = Path(f"cache/user:{secret}@assets.example.test/model.usd")
    inspected_paths: list[Path] = []

    def raise_name_too_long(path: Path) -> bool:
        inspected_paths.append(path)
        raise OSError(errno.ENAMETOOLONG, "File name too long", str(path))

    monkeypatch.setattr(Path, "is_file", raise_name_too_long)

    with pytest.raises(OSError) as exc_info:
        path_is_file_with_safe_diagnostics(
            credential_path,
            label="credential-bearing source",
        )

    assert inspected_paths == [credential_path]
    assert exc_info.value.errno == errno.ENAMETOOLONG
    assert exc_info.value.filename == "<redacted>"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    observable = "".join(traceback.format_exception(exc_info.value))
    assert secret not in observable
    assert "<redacted>" in observable


def test_safe_text_read_preserves_is_a_directory_without_path_leak(
    tmp_path: Path,
) -> None:
    secret = "read-helper-secret-713"
    credential_dir = tmp_path / f"user:{secret}@assets.example.test"
    credential_dir.mkdir()

    with pytest.raises(IsADirectoryError) as exc_info:
        read_text_with_safe_diagnostics(
            credential_dir,
            label="credential-bearing text file",
        )

    assert exc_info.value.errno == errno.EISDIR
    assert exc_info.value.filename == "<redacted>"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    observable = "".join(traceback.format_exception(exc_info.value))
    assert secret not in observable
    assert "<redacted>" in observable


def test_find_inline_secret_paths_is_recursive_and_value_free() -> None:
    config = {
        "steps": [
            {"vlm": {"api_key": "x"}},
            {"auth": {"client_secret": "sentinel-client-secret"}},
            {"headers": {"Authorization": "Bearer sentinel"}},
        ],
        "aws": {"secret_access_key": "sentinel-aws"},
    }

    paths = find_inline_secret_paths(config)

    assert paths == (
        "steps[0].vlm.api_key",
        "steps[1].auth.client_secret",
        "steps[2].headers.Authorization",
        "aws.secret_access_key",
    )
    assert "sentinel" not in repr(paths)


def test_sensitive_key_with_structured_value_fails_closed_with_typed_error() -> None:
    config = {"api_key": {"region": "us", "timeout": 30}}

    assert find_inline_secret_paths(config) == ("api_key",)
    assert redact_sensitive_config(config) == {"api_key": "<redacted>"}
    with pytest.raises(InlineSecretError, match="api_key") as exc_info:
        ensure_no_inline_secrets(config, context="structured credential field")

    assert isinstance(exc_info.value, ValueError)
    assert exc_info.value.paths == ("api_key",)
    assert "region" not in str(exc_info.value)
    assert "timeout" not in str(exc_info.value)


@pytest.mark.parametrize(
    "key",
    [
        "x-api-key",
        "secretKey",
        "access_key_id",
        "accountKey",
        "auth_header",
        "authorizationHeader",
        "private-key",
        "AWSAccessKeyId",
        "HTTPAuthorizationHeader",
        "client_secret_value",
        "apiSecretValue",
        "api_keys",
        "APIKeys",
        "api_tokens",
        "access_tokens",
        "refreshTokens",
        "tokens",
        "passwords",
        "secrets",
        "private_keys",
        "signingKeys",
        "authorizationHeaders",
        "access_key_ids",
        "client_secrets",
        "session_key",
        "storage_key",
        "subscription_key",
        "serviceAccountKey",
        "SharedAccessKey",
        "signing-key-string",
    ],
)
def test_find_inline_secret_paths_covers_common_credential_key_styles(
    key: str,
) -> None:
    assert find_inline_secret_paths({key: "s"}) == (key,)


@pytest.mark.parametrize(
    "key",
    [
        "api_key_2",
        "apiKey2",
        "secret_v2",
        "backup_api_key_1",
        "api_key_2_value",
        "clientSecretValueV3",
    ],
)
def test_rotated_credential_key_variants_are_detected(key: str) -> None:
    assert find_inline_secret_paths({key: "rotation-secret"}) == (key,)
    assert redact_sensitive_config({key: "rotation-secret"}) == {key: "<redacted>"}


@pytest.mark.parametrize("key", ["max_tokens_2", "custom_tokens_2", "tokenizer2"])
def test_rotated_credential_key_detection_preserves_benign_fields(key: str) -> None:
    config = {key: "ordinary-setting"}

    assert find_inline_secret_paths(config) == ()
    assert redact_sensitive_config(config) == config


def test_inline_secret_detection_preserves_explicit_references_and_false_positives() -> (
    None
):
    config = {
        "api_key_env": "${CUSTOM_API_KEY}",
        "client_secret_env_value": "${CLIENT_SECRET}",
        "password_file": "/run/secrets/password",
        "credential_fields": ["client_secret"],
        "client_secrets_file": "/run/secrets/oauth-client.json",
        "storage_key_env": "${AZURE_STORAGE_KEY}",
        "max_tokens": 1024,
        "max_completion_tokens": 2048,
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "custom_tokens": ["ordinary", "metadata"],
        "tokenizer": "fast",
        "secretary": "person",
        "cache_key": "ordinary-cache-key",
        "cache_keys": ["ordinary-cache-key"],
        "output_keys": ["classification"],
        "public_keys": ["public-material"],
        "keys": ["ordinary-lookup-key"],
        "key": "ordinary-lookup-key",
        "placeholder": {"api_key": "YOUR_API_KEY_HERE"},
        "local": {"api_key": "not-used"},
    }

    assert find_inline_secret_paths(config) == ()
    ensure_no_inline_secrets(config)


@pytest.mark.parametrize(
    "value",
    [
        "YOUR_API_KEY_HERE",
        "your-api-key",
        "YOUR_NIM_API_KEY",
        "YOUR_NGC_API_KEY",
        "not-used",
    ],
)
def test_durable_secret_guards_exempt_explicit_placeholders(value: str) -> None:
    config = {"api_key": value}

    assert find_inline_secret_paths(config) == ()
    assert redact_sensitive_config(config) == config
    ensure_no_inline_secrets(config)


@pytest.mark.parametrize(
    "value", ["your_live_credential_123", "your-live-credential-123"]
)
def test_durable_secret_guards_do_not_trust_placeholder_prefix(value: str) -> None:
    config = {"api_key": value}

    assert find_inline_secret_paths(config) == ("api_key",)
    assert redact_sensitive_config(config) == {"api_key": "<redacted>"}
    with pytest.raises(ValueError) as exc_info:
        ensure_no_inline_secrets(config)
    assert value not in str(exc_info.value)


def test_credential_env_fields_require_explicit_durable_references() -> None:
    config = {
        "api_key_env": "nvapi-secret-value",
        "nested": {"secretEnv": "another-secret-value"},
        "token_env": "${CUSTOM_TOKEN}",
        "password_env_var": "${CUSTOM_PASSWORD}",
        "credential_env_vars": [
            "${PRIMARY_CREDENTIAL}",
            "$FALLBACK_CREDENTIAL",
        ],
        "api_key_env_2": "${ROTATED_API_KEY}",
        "client_secret_env_v2": "$ROTATED_CLIENT_SECRET",
        "direct": {"api_key": "${CUSTOM_API_KEY}"},
        "secretary_env": "ordinary-setting",
    }

    assert find_inline_secret_paths(config) == (
        "api_key_env",
        "nested.secretEnv",
        "credential_env_vars",
        "client_secret_env_v2",
        "direct.api_key",
    )
    assert redact_sensitive_config(config) == {
        "api_key_env": "<redacted>",
        "nested": {"secretEnv": "<redacted>"},
        "token_env": "${CUSTOM_TOKEN}",
        "password_env_var": "${CUSTOM_PASSWORD}",
        "credential_env_vars": "<redacted>",
        "api_key_env_2": "${ROTATED_API_KEY}",
        "client_secret_env_v2": "<redacted>",
        "direct": {"api_key": "<redacted>"},
        "secretary_env": "ordinary-setting",
    }


def test_credential_edge_values_remain_safe_and_non_secret() -> None:
    config = {
        "value": "ordinary-setting",
        "env": "ordinary-setting",
        "api_key_env": [123],
        "none_key": {"api_key": None},
        "false_key": {"api_key": False},
        "empty_key": {"api_key": "  "},
        "redacted_key": {"api_key": "<redacted>"},
    }
    original = deepcopy(config)

    assert find_inline_secret_paths(config) == ("api_key_env",)
    redacted = redact_sensitive_config((config, "plain"))
    assert redacted == (
        {
            **config,
            "api_key_env": "<redacted>",
        },
        "plain",
    )
    assert config == original
    assert redacted[0] is not config


def test_ensure_no_inline_secrets_reports_paths_not_values() -> None:
    secret = "never-print-this-sentinel"

    with pytest.raises(ValueError) as exc_info:
        ensure_no_inline_secrets(
            {"nested": {"apiKey": secret}}, context="pipeline publication"
        )

    message = str(exc_info.value)
    assert "pipeline publication" in message
    assert "nested.apiKey" in message
    assert secret not in message


def test_ensure_no_inline_secrets_bounds_reported_paths() -> None:
    config = [{"api_key": f"secret-{index}"} for index in range(9)]

    with pytest.raises(ValueError, match=r"and 1 more") as exc_info:
        ensure_no_inline_secrets(config)

    assert "[7].api_key" in str(exc_info.value)
    assert "[8].api_key" not in str(exc_info.value)


def test_redact_sensitive_config_preserves_shape_without_secret_fragments() -> None:
    secret = "abcd-super-secret-wxyz"
    config = {
        "api_key": secret,
        "nested": [{"access_token": "t"}, {"max_tokens": 3}],
        "api_key_env": "${SAFE_ENV_NAME}",
    }

    redacted = redact_sensitive_config(config)

    assert redacted == {
        "api_key": "<redacted>",
        "nested": [{"access_token": "<redacted>"}, {"max_tokens": 3}],
        "api_key_env": "${SAFE_ENV_NAME}",
    }
    assert secret[:4] not in repr(redacted)
    assert secret[-4:] not in repr(redacted)


def test_redact_sensitive_config_separates_aliases_by_credential_context() -> None:
    opaque_secret = "QWxwaGFCZXRhR2FtbWE"
    shared_mapping = {"slot": opaque_secret}
    shared_list = [opaque_secret]
    shared_tuple = (opaque_secret,)
    config = {
        "ordinary_mapping": shared_mapping,
        "credentials": shared_mapping,
        "ordinary_list": shared_list,
        "tokens": shared_list,
        "ordinary_tuple": shared_tuple,
        "access_tokens": shared_tuple,
    }

    redacted = redact_sensitive_config(config)

    assert redacted["ordinary_mapping"] == {"slot": opaque_secret}
    assert redacted["credentials"] == {"slot": "<redacted>"}
    assert redacted["ordinary_list"] == [opaque_secret]
    assert redacted["tokens"] == ["<redacted>"]
    assert redacted["ordinary_tuple"] == (opaque_secret,)
    assert redacted["access_tokens"] == ("<redacted>",)


@pytest.mark.parametrize("is_bound", [False, True])
def test_identifier_shaped_env_value_is_never_trusted_as_durable_reference(
    monkeypatch: pytest.MonkeyPatch,
    is_bound: bool,
) -> None:
    ambiguous_value = "MYTOKEN123ABC"
    if is_bound:
        monkeypatch.setenv(ambiguous_value, "attacker-controlled-dummy")
    else:
        monkeypatch.delenv(ambiguous_value, raising=False)
    config = {"api_key_env": ambiguous_value}

    assert find_inline_secret_paths(config) == ("api_key_env",)
    assert redact_sensitive_config(config) == {"api_key_env": "<redacted>"}
    with pytest.raises(InlineSecretError) as exc_info:
        ensure_no_inline_secrets(config, context="durable pipeline config")

    assert ambiguous_value not in str(exc_info.value)


@pytest.mark.parametrize("is_bound", [False, True])
def test_explicit_env_reference_persists_only_name_independent_of_environment(
    monkeypatch: pytest.MonkeyPatch,
    is_bound: bool,
) -> None:
    env_name = "CUSTOM_BOUND_PIPELINE_KEY"
    resolved_secret = "never-persist-resolved-env-value"
    if is_bound:
        monkeypatch.setenv(env_name, resolved_secret)
    else:
        monkeypatch.delenv(env_name, raising=False)
    reference = format_env_reference(env_name)
    config = {"api_key_env": reference}

    assert find_inline_secret_paths(config) == ()
    assert redact_sensitive_config(config) == config
    ensure_no_inline_secrets(config, context="durable pipeline config")

    serialized = yaml.safe_dump(config)
    assert reference in serialized
    assert resolved_secret not in serialized


def test_env_reference_helpers_are_deterministic_and_resolve_legacy_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = "CUSTOM_PIPELINE_KEY"
    resolved_secret = "resolved-only-at-runtime"
    explicit = "${CUSTOM_PIPELINE_KEY}"

    assert format_env_reference(env_name) == explicit
    assert format_env_reference(explicit) == explicit
    assert parse_env_reference(explicit) == env_name
    assert parse_env_reference(env_name) is None
    assert parse_env_reference(env_name, allow_legacy_bare=True) == env_name
    assert parse_env_reference("$CUSTOM_PIPELINE_KEY") is None
    with pytest.raises(ValueError, match="environment reference") as exc_info:
        format_env_reference("not a variable name")
    assert "not a variable name" not in str(exc_info.value)

    monkeypatch.setenv(env_name, resolved_secret)
    assert resolve_endpoint_api_key(api_key_env=explicit) == resolved_secret
    assert resolve_endpoint_api_key(api_key_env=env_name) == resolved_secret


@pytest.mark.parametrize(
    "reference",
    ["NEVER_DISCLOSE_MISSING_ENV_713", "${NEVER_DISCLOSE_MISSING_ENV_713}"],
)
@pytest.mark.parametrize("api_key", [None, "YOUR_API_KEY"])
def test_required_env_failure_is_value_free_and_has_no_exception_chain(
    monkeypatch: pytest.MonkeyPatch,
    reference: str,
    api_key: str | None,
) -> None:
    """Legacy and explicit references stay runtime inputs, not diagnostics."""
    env_name = "NEVER_DISCLOSE_MISSING_ENV_713"
    monkeypatch.delenv(env_name, raising=False)

    with pytest.raises(ValueError) as exc_info:
        resolve_endpoint_api_key(
            api_key=api_key,
            api_key_env=reference,
            require_env=True,
        )

    error = exc_info.value
    diagnostic = f"{error!s}\n{error!r}\n{error.args!r}"
    assert diagnostic == (
        "configured API key environment variable is not set or empty\n"
        "ValueError('configured API key environment variable is not set or empty')\n"
        "('configured API key environment variable is not set or empty',)"
    )
    assert env_name not in diagnostic
    assert reference not in diagnostic
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback_cursor = error.__traceback__
    resolver_frames: list[dict[str, object]] = []
    while traceback_cursor is not None:
        if traceback_cursor.tb_frame.f_code.co_name == "resolve_endpoint_api_key":
            resolver_frames.append(dict(traceback_cursor.tb_frame.f_locals))
        traceback_cursor = traceback_cursor.tb_next
    assert resolver_frames
    assert env_name not in repr(resolver_frames)
    assert reference not in repr(resolver_frames)
    if api_key is not None:
        assert api_key not in repr(resolver_frames)


@pytest.mark.parametrize(
    "reference",
    ["NEVER_DISCLOSE_MISSING_ENV_727", "${NEVER_DISCLOSE_MISSING_ENV_727}"],
)
def test_required_env_uses_valid_inline_fallback(
    monkeypatch: pytest.MonkeyPatch,
    reference: str,
) -> None:
    monkeypatch.delenv("NEVER_DISCLOSE_MISSING_ENV_727", raising=False)
    inline_fallback = "valid-inline-fallback-727"

    assert (
        resolve_endpoint_api_key(
            api_key=inline_fallback,
            api_key_env=reference,
            require_env=True,
        )
        == inline_fallback
    )


def test_preferred_required_env_does_not_use_inline_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = "NEVER_DISCLOSE_PREFERRED_ENV_727"
    inline_fallback = "never-disclose-preferred-inline-fallback"
    monkeypatch.delenv(env_name, raising=False)

    with pytest.raises(ValueError) as exc_info:
        resolve_endpoint_api_key(
            api_key=inline_fallback,
            api_key_env=env_name,
            prefer_env=True,
            require_env=True,
        )

    diagnostic = repr(exc_info.value)
    assert env_name not in diagnostic
    assert inline_fallback not in diagnostic


@pytest.mark.parametrize(
    "secret_key",
    [
        (
            "https://assets.example.test/object.usd?"
            "X-Amz-Signature=never-disclose-mapping-signature"
        ),
        "https://mapping-user:never-disclose-userinfo@example.test/object.usd",
        "artifact.usd?X-Amz-Signature=never-disclose-relative-signature",
        "artifact#access_token=never-disclose-fragment-token",
    ],
)
def test_mapping_key_credentials_redact_atomically_without_disclosure(
    secret_key: str,
) -> None:
    config = {
        "nested": {
            "ordinary": "setting",
            secret_key: "route-name",
        }
    }

    assert find_inline_secret_paths(config) == ("nested",)
    assert redact_sensitive_config(config) == {"nested": "<redacted>"}
    with pytest.raises(ValueError) as exc_info:
        ensure_no_inline_secrets(config, context="generated mapping")

    message = str(exc_info.value)
    assert "nested" in message
    assert secret_key not in message
    assert "never-disclose" not in message
    assert secret_key not in yaml.safe_dump(redact_sensitive_config(config))


def test_composite_mapping_key_credentials_use_value_free_parent_path() -> None:
    secret = "never-disclose-composite-signature"
    secret_key = (
        "signed-route",
        f"https://assets.example.test/object.usd?sig={secret}",
    )
    config = {"routes": {secret_key: "primary", "fallback": "public"}}

    assert find_inline_secret_paths(config) == ("routes",)
    assert redact_sensitive_config(config) == {"routes": "<redacted>"}
    with pytest.raises(ValueError) as exc_info:
        ensure_no_inline_secrets(config)
    assert secret not in str(exc_info.value)


def test_yaml_set_signed_url_is_detected_and_redacted_after_round_trip() -> None:
    secret = "never-persist-this-set-signature"
    config = yaml.safe_load(
        "reference_images: !!set\n"
        "  ? 'https://assets.example.test/image.png?"
        f"X-Amz-Signature={secret}'\n"
    )
    round_tripped = yaml.safe_load(yaml.safe_dump(config))

    assert isinstance(round_tripped["reference_images"], set)
    assert find_inline_secret_paths(round_tripped) == ("reference_images",)
    assert redact_sensitive_config(round_tripped) == {"reference_images": "<redacted>"}
    with pytest.raises(ValueError) as exc_info:
        ensure_no_inline_secrets(round_tripped, context="generated scene config")
    assert "reference_images" in str(exc_info.value)
    assert secret not in str(exc_info.value)
    assert secret not in yaml.safe_dump(redact_sensitive_config(round_tripped))


def test_unordered_container_credentials_use_one_stable_parent_path() -> None:
    first_secret = "never-persist-first-signature"
    second_secret = "never-persist-second-signature"
    config = {
        "nested": [
            frozenset(
                {
                    f"https://assets.example.test/a?sig={first_secret}",
                    f"https://assets.example.test/b?sig={second_secret}",
                }
            )
        ]
    }

    assert find_inline_secret_paths(config) == ("nested[0]",)
    assert redact_sensitive_config(config) == {"nested": ["<redacted>"]}


def test_unordered_redaction_reuses_the_caller_bounded_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_limits = replace(DEFAULT_CREDENTIAL_SCAN_LIMITS, max_nodes=8)
    monkeypatch.setattr(
        credentials_module,
        "DEFAULT_CREDENTIAL_SCAN_LIMITS",
        replace(DEFAULT_CREDENTIAL_SCAN_LIMITS, max_nodes=1),
    )
    original_scan = credentials_module.find_inline_secret_paths
    scan_limits: list[CredentialScanLimits | None] = []

    def counted_scan(*args: object, **kwargs: object) -> tuple[str, ...]:
        scan_limits.append(kwargs.get("limits"))
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(credentials_module, "find_inline_secret_paths", counted_scan)
    config = {"links": {"https://assets.example.test/image.png?sig=live-secret"}}

    assert credentials_module.redact_sensitive_config(
        config,
        limits=custom_limits,
    ) == {"links": "<redacted>"}
    assert scan_limits == [custom_limits]


@pytest.mark.parametrize("value", [{"alpha", "beta"}, frozenset({"alpha", "beta"})])
def test_benign_unordered_containers_remain_credential_free(
    value: set[str] | frozenset[str],
) -> None:
    config = {"labels": value}

    assert find_inline_secret_paths(config) == ()
    ensure_no_inline_secrets(config)
    redacted = redact_sensitive_config(config)
    assert redacted == config
    if isinstance(value, set):
        assert redacted["labels"] is not value


@pytest.mark.parametrize(
    "binary_value",
    [
        b"https://user:binary-secret@example.test/artifact",
        bytearray(b"https://example.test/artifact?sig=binary-secret"),
        memoryview(b"Authorization: Bearer binary-secret-token-713"),
        "https://user:utf16-secret@example.test/artifact".encode("utf-16"),
        "https://example.test/artifact?sig=utf32-secret".encode("utf-32"),
        b"opaque-binary-configuration-value",
    ],
)
def test_binary_scalars_cannot_cross_the_durable_scanner(
    binary_value: bytes | bytearray | memoryview,
) -> None:
    config = {"payload": binary_value}

    assert find_inline_secret_paths(config) == ("payload",)
    assert redact_sensitive_config(config) == {"payload": "<redacted>"}
    with pytest.raises(InlineSecretError) as exc_info:
        ensure_no_inline_secrets(config, context="binary durable value")

    observable = str(exc_info.value)
    assert "payload" in observable
    assert "binary-secret" not in observable


def test_binary_text_credential_mapping_key_is_an_atomic_boundary() -> None:
    secret_key = b"https://user:binary-key-secret@example.test/config"
    config = {secret_key: "ordinary"}

    assert find_inline_secret_paths(config) == ("$",)
    assert redact_sensitive_config(config) == "<redacted>"


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.test/v1",
        "https://bearer-token@example.test/v1",
        "https://example.test/v1?api_key=live-key",
        "https://example.test/v1?key=google-api-key",
        "https://example.test/object?X-Amz-Signature=deadbeef",
        "https://example.test/container?sv=1&sig=azure-signature",
        "https://example.test/callback#access_token=oauth-token",
        "https://user:password@[invalid/v1",
        "https://[invalid/v1?access_token=oauth-token",
        "wss://socket-token@stream.example.test/events",
        "ws://stream.example.test/events?access_token=socket-token",
        "ftp://user:password@files.example.test/archive",
        "s3://access:secret@bucket/object.usd",
        "https://live-secret-713%20%40example.com/path",
        "https://live-secret-713%09%40example.com/path",
        "https://live-secret-713%22%40example.com/path",
        "https://live-secret-713%2f%40example.com/path",
        "https://live-secret-713%2520%2540example.com/path",
        "connect using wss://socket-token@stream.example.test/events now",
        (
            "https://gateway.example.test/callback?redirect="
            "https%3A%2F%2Fuser%3Apassword%40private.example.test%2Fv1"
        ),
        (
            "https://gateway.example.test/callback?redirect="
            "https%253A%252F%252Fuser%253Apassword%2540private.example.test%252Fv1"
        ),
        "https://example.test/v1?model=latest;access_token=socket-token",
        (
            "https://gateway.example.test/callback?redirect="
            "%2F%2Fuser%3Apassword%40private.example.test%2Fv1"
        ),
        (
            "see https://public.example.test and "
            "https%253A%252F%252Fuser%253Apassword%2540private.example.test%252Fv1"
        ),
    ],
)
def test_durable_secret_guards_reject_credentials_embedded_in_urls(url: str) -> None:
    config = {"nested": {"base_url": url}}

    assert find_inline_secret_paths(config) == ("nested.base_url",)
    assert redact_sensitive_config(config) == {"nested": {"base_url": "<redacted>"}}
    with pytest.raises(ValueError) as exc_info:
        ensure_no_inline_secrets(config)
    assert "nested.base_url" in str(exc_info.value)
    assert url not in str(exc_info.value)


@pytest.mark.parametrize(
    "value",
    [
        "Authorization: Bearer never-persist-authorization",
        "Authorization: Bearer a1-b",
        "Authorization%3A%20Bearer%20a1-b",
        "Authorization%253A%2520Bearer%2520a1-b",
        "authorization=never-persist-direct-header",
        "use Bearer never-persist-bare-token for this request",
        "use Bearer\nnever-persist-line-token for this request",
        "Bearer\r\nnever-persist-crlf-token",
        _percent_encode(
            "use Bearer\nnever-persist-deep-line-token for this request",
            5,
        ),
        "use Bearer a1-b for this request",
        "Bearer abc",
        "use Bearer secret for this request",
        "user:never-persist-dsn@example.test/database",
        "user:never-persist-single-label@redis",
        "123456:987654@redis",
        "24:60@redis",
        "registry.example:5000@namespace",
        _percent_encode("user:never-persist-deep-single-label@redis", 5),
        "postgres:user:never-persist-opaque-dsn@example.test/database",
        "sip:user:never-persist-sip-dsn@host",
        "postgres%3Auser%3Anever-persist-encoded-dsn%40example.test%2Fdatabase",
        _percent_encode(
            "postgres:user:never-persist-deep-dsn@host/database",
            5,
        ),
    ],
)
def test_durable_secret_guards_reject_free_text_credentials(value: str) -> None:
    config = {"user_prompt": value}

    assert find_inline_secret_paths(config) == ("user_prompt",)
    assert redact_sensitive_config(config) == {"user_prompt": "<redacted>"}
    with pytest.raises(ValueError) as exc_info:
        ensure_no_inline_secrets(config, context="durable prompt")
    assert value not in str(exc_info.value)
    assert "never-persist" not in str(exc_info.value)


def test_encoded_dsn_shaped_long_nonmatch_is_scanned_in_bounded_time() -> None:
    # The encoded delimiters are deliberately reversed (``@`` before ``:``),
    # forcing a complete non-match scan without containing a credential.
    def make_value(segment_length: int) -> str:
        segment = "a" * segment_length
        return f"{segment}%40{segment}%3A{segment}"

    _assert_scan_within_declared_byte_budget(
        make_value(5_000),
        make_value(20_000),
        expected_paths=(),
    )


@pytest.mark.parametrize(
    "value",
    [
        "api_key=assignment-secret-713",
        "api-key: assignment-secret-713",
        "x-api-key:assignment-secret-713",
        "OPENAI_API_KEY = assignment-secret-713",
        "AWSAccessKeyId: assignment-secret-713",
        "refresh-token=assignment-secret-713",
        "api_key_2=assignment-secret-713",
        "password='assignment secret 713'",
        'client_secret: "assignment secret 713"',
        "credentials follow:\nclientSecret=assignment-secret-713",
        "api_key=your_live_credential_123",
        "api_key%3Dencoded-assignment-secret-713",
        "OPENAI_API_KEY%253Dencoded-assignment-secret-713",
        _percent_encode("client_secret=deep-assignment-secret-713", 5),
    ],
)
def test_free_text_sensitive_assignments_are_rejected(value: str) -> None:
    config = {"description": value}

    assert find_inline_secret_paths(config) == ("description",)
    assert redact_sensitive_config(config) == {"description": "<redacted>"}
    with pytest.raises(InlineSecretError) as exc_info:
        ensure_no_inline_secrets(config, context="durable description")
    assert "description" in str(exc_info.value)
    assert "assignment-secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    "connection_string",
    [
        (
            "DefaultEndpointsProtocol=https;AccountName=durabletest;"
            "AccountKey=QXp1cmVTdG9yYWdlU2VjcmV0NzEzLys9PQ==;"
            "EndpointSuffix=core.windows.net"
        ),
        (
            "Endpoint=sb://durabletest.servicebus.windows.net/;"
            "SharedAccessKeyName=producer;"
            "SharedAccessKey=U2VydmljZUJ1c1NlY3JldDcxMy8rPT0="
        ),
        (
            "DefaultEndpointsProtocol=https;AccountName=encoded;"
            "AccountKey%3DQXp1cmVFbmNvZGVkU2VjcmV0NzEzPQ=="
        ),
        (
            "Endpoint=sb://durabletest.servicebus.windows.net/;"
            "SharedAccessSignature=sr=durabletest&sig=SASSECRET713&se=9999999999"
        ),
    ],
)
def test_azure_connection_string_secrets_are_rejected_value_safely(
    connection_string: str,
) -> None:
    durable_event = {"event": {"message": connection_string}}

    assert find_inline_secret_paths(durable_event) == ("event.message",)
    assert redact_sensitive_config(durable_event) == {
        "event": {"message": "<redacted>"}
    }
    with pytest.raises(InlineSecretError) as exc_info:
        ensure_no_inline_secrets(durable_event, context="durable event")
    diagnostic = str(exc_info.value)
    assert "event.message" in diagnostic
    assert connection_string not in diagnostic
    assert "QXp1cm" not in diagnostic
    assert "U2Vydmlj" not in diagnostic
    assert "SASSECRET713" not in diagnostic


@pytest.mark.parametrize(
    "key",
    [
        "connection_string",
        "connectionString",
        "storage_connection_string_2",
        "connection_string_value",
        "connection_string_value_v2_literal",
    ],
)
def test_opaque_connection_string_fields_are_sensitive_by_name(key: str) -> None:
    opaque_secret = "opaque-connection-credential-727"
    config = {key: opaque_secret}

    assert find_inline_secret_paths(config) == (key,)
    assert redact_sensitive_config(config) == {key: "<redacted>"}
    with pytest.raises(InlineSecretError) as exc_info:
        ensure_no_inline_secrets(config)
    assert opaque_secret not in str(exc_info.value)


def test_connection_string_reference_field_remains_non_secret() -> None:
    config = {"connection_string_env": "${DATABASE_CONNECTION_STRING}"}

    assert find_inline_secret_paths(config) == ()
    assert redact_sensitive_config(config) == config


@pytest.mark.parametrize(
    "value",
    [
        (
            "DefaultEndpointsProtocol=https;AccountName=template;"
            "AccountKey=${AZURE_STORAGE_ACCOUNT_KEY};"
            "EndpointSuffix=core.windows.net"
        ),
        "AccountKey=YOUR_API_KEY_HERE",
        "AccountKey=YOUR_ACCOUNT_KEY_HERE",
        "SharedAccessKey=your-shared-access-key-here",
        "SharedAccessSignature=${AZURE_SERVICE_BUS_SAS}",
        "SharedAccessSignature: required.",
        "SharedAccessSignatureName=producer",
        "AccountKey: required.",
        "AccountKeyName=primary-rotation",
        (
            "Endpoint=sb://durabletest.servicebus.windows.net/;"
            "SharedAccessKeyName=producer"
        ),
        "AccountName=public-account-name",
    ],
)
def test_azure_connection_string_references_and_documentation_remain_safe(
    value: str,
) -> None:
    config = {"description": value}

    assert find_inline_secret_paths(config) == ()
    assert redact_sensitive_config(config) == config
    ensure_no_inline_secrets(config)


@pytest.mark.parametrize(
    "serialized",
    [
        json.dumps({"client_secret": "LIVESECRET713"}),
        r'{"client\u005fsecret": "LIVESECRET713"}',
        json.dumps({"client secret": "LIVESECRET713"}),
        json.dumps({"private key": "LIVESECRET713"}),
        json.dumps({"shared access signature": "LIVESECRET713"}),
        yaml.safe_dump({"private key": "LIVESECRET713"}, sort_keys=False),
        yaml.safe_dump({"shared access signature": "LIVESECRET713"}, sort_keys=False),
        yaml.safe_dump({"service account key": "LIVESECRET713"}, sort_keys=False),
        yaml.safe_dump({"api_key": {"value": "LIVESECRET713"}}, sort_keys=False),
        yaml.safe_dump({"client_secret": ["LIVESECRET713"]}, sort_keys=False),
        yaml.safe_dump({"private key": {"value": "LIVESECRET713"}}, sort_keys=False),
        json.dumps({"AccountKey": "LIVESECRET713"}),
        json.dumps({"password": "LIVESECRET713"}),
        json.dumps({"api_key": "LIVESECRET713"}),
        "{'client_secret': 'LIVESECRET713'}",
        "{'AccountKey': \"LIVESECRET713\"}",
        "settings = {'password': 'LIVESECRET713'}",
        json.dumps(
            {
                "private_key": (
                    "-----BEGIN PRIVATE KEY-----\n"
                    "YOUR_PRIVATE_KEY_HERE\n"
                    "-----END PRIVATE KEY-----\n"
                    "LIVESECRET713"
                )
            }
        ),
        json.dumps(
            {
                "private_key": (
                    "-----BEGIN PRIVATE KEY-----LIVESECRET713-----END PRIVATE KEY-----"
                )
            }
        ),
    ],
)
def test_quoted_structured_secret_assignments_are_rejected_value_safely(
    serialized: str,
) -> None:
    config = {"event": {"message": serialized}}

    assert find_inline_secret_paths(config) == ("event.message",)
    assert redact_sensitive_config(config) == {"event": {"message": "<redacted>"}}
    with pytest.raises(InlineSecretError) as exc_info:
        ensure_no_inline_secrets(config, context="durable structured event")
    diagnostic = str(exc_info.value)
    assert "event.message" in diagnostic
    assert serialized not in diagnostic
    assert "LIVESECRET713" not in diagnostic


@pytest.mark.parametrize(
    "serialized",
    [
        json.dumps({"client_secret": "example"}),
        json.dumps({"api_key": "YOUR_API_KEY_HERE"}),
        json.dumps({"AccountKey": "${AZURE_STORAGE_ACCOUNT_KEY}"}),
        "{'password': None}",
        "{'client_secret': '<redacted>'}",
        'The field "client_secret": "example" is documentation.',
        json.dumps({"public_key": "identifier"}),
    ],
)
def test_quoted_structured_references_and_documentation_remain_safe(
    serialized: str,
) -> None:
    config = {"description": serialized}

    assert find_inline_secret_paths(config) == ()
    assert redact_sensitive_config(config) == config
    ensure_no_inline_secrets(config)


def test_deep_encoded_quoted_structured_secret_fails_closed() -> None:
    serialized = _percent_encode(
        json.dumps({"client_secret": "LIVESECRET713"}),
        5,
    )

    assert find_inline_secret_paths({"event": serialized}) == ("event",)
    assert redact_sensitive_config({"event": serialized}) == {"event": "<redacted>"}
    with pytest.raises(InlineSecretError) as exc_info:
        ensure_no_inline_secrets({"event": serialized}, context="durable event")
    assert "LIVESECRET713" not in str(exc_info.value)


@pytest.mark.parametrize(
    "serialized",
    [
        '{"client_secret" : "LIVESECRET713"}',
        '{"client_secret"\t:\t"LIVESECRET713"}',
        '{"client_secret"\n:\n"LIVESECRET713"}',
        '{"client_secret"\r\n:\r\n"LIVESECRET713"}',
        "{'client_secret' = 'LIVESECRET713'}",
    ],
)
@pytest.mark.parametrize("encoding_rounds", [0, 5])
def test_quoted_assignment_json_whitespace_is_detected_raw_and_encoded(
    serialized: str,
    encoding_rounds: int,
) -> None:
    value = _percent_encode(serialized, encoding_rounds)

    assert find_inline_secret_paths({"event": value}) == ("event",)
    assert redact_sensitive_config({"event": value}) == {"event": "<redacted>"}
    with pytest.raises(InlineSecretError) as exc_info:
        ensure_no_inline_secrets({"event": value}, context="durable event")
    assert "LIVESECRET713" not in str(exc_info.value)


@pytest.mark.parametrize("quote_layers", [3, 4, 7])
def test_fully_byte_encoded_quoted_assignment_detects_boundary_and_exhaustion(
    quote_layers: int,
) -> None:
    raw = '{"client_secret"\n:\n"LIVESECRET713"}'
    encoded = _percent_encode(_percent_encode_all_bytes(raw), quote_layers)

    assert find_inline_secret_paths({"event": encoded}) == ("event",)
    assert redact_sensitive_config({"event": encoded}) == {"event": "<redacted>"}
    with pytest.raises(InlineSecretError) as exc_info:
        ensure_no_inline_secrets({"event": encoded}, context="durable event")
    assert "LIVESECRET713" not in str(exc_info.value)


def test_quoted_assignment_scan_scales_with_long_benign_field() -> None:
    def make_value(length: int) -> str:
        return json.dumps(
            {
                "description": "x" * length,
                "client_secret": "LIVESECRET713",
            }
        )

    _assert_scan_within_declared_byte_budget(
        make_value(10_000),
        make_value(40_000),
        expected_paths=("description",),
    )


def test_fully_byte_encoded_assignment_probe_scales_linearly() -> None:
    def make_value(length: int) -> str:
        raw = json.dumps(
            {
                "description": "x" * length,
                "client_secret": "LIVESECRET713",
            }
        )
        return _percent_encode(_percent_encode_all_bytes(raw), 4)

    _assert_scan_within_declared_byte_budget(
        make_value(2_000),
        make_value(8_000),
        expected_paths=("description",),
    )


@pytest.mark.parametrize(
    "private_key_block",
    [
        (
            "-----BEGIN PRIVATE KEY-----\n"
            "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC713PRIVATEKEYPAYLOAD==\n"
            "-----END PRIVATE KEY-----"
        ),
        (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA713RSAPRIVATEKEYPAYLOAD+/==\n"
            "-----END RSA PRIVATE KEY-----"
        ),
        (
            "-----BEGIN OPENSSH PRIVATE KEY-----\\n"
            "b3BlbnNzaC1rZXktdjEAAAAA713PRIVATEKEYPAYLOAD\\n"
            "-----END OPENSSH PRIVATE KEY-----"
        ),
        (
            "-----BEGIN PGP PRIVATE KEY BLOCK-----\n"
            "lQOYBG713PGPPRIVATEKEYPAYLOAD+/==\n"
            "-----END PGP PRIVATE KEY BLOCK-----"
        ),
    ],
)
def test_private_key_blocks_in_free_text_are_rejected_value_safely(
    private_key_block: str,
) -> None:
    durable_checkpoint = {"steps": {"result": private_key_block}}

    assert find_inline_secret_paths(durable_checkpoint) == ("steps.result",)
    assert redact_sensitive_config(durable_checkpoint) == {
        "steps": {"result": "<redacted>"}
    }
    with pytest.raises(InlineSecretError) as exc_info:
        ensure_no_inline_secrets(durable_checkpoint, context="durable checkpoint")
    diagnostic = str(exc_info.value)
    assert "steps.result" in diagnostic
    assert private_key_block not in diagnostic
    assert "PRIVATEKEYPAYLOAD" not in diagnostic


def test_embedded_serialized_and_unicode_prefixed_private_keys_are_rejected() -> None:
    private_key_block = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIE-LIVEPEMSECRET713-PAYLOAD==\n"
        "-----END PRIVATE KEY-----"
    )
    wrapped_values = [
        json.dumps({"private_key": private_key_block}),
        f"diagnostic prefix {private_key_block}",
        "\n".join(f"> {line}" for line in private_key_block.splitlines()),
        f"{'ß' * 2_000}{private_key_block}",
    ]

    for wrapped in wrapped_values:
        config = {"checkpoint": {"output": wrapped}}
        assert find_inline_secret_paths(config) == ("checkpoint.output",)
        assert redact_sensitive_config(config) == {
            "checkpoint": {"output": "<redacted>"}
        }
        with pytest.raises(InlineSecretError) as exc_info:
            ensure_no_inline_secrets(config, context="durable checkpoint")
        diagnostic = str(exc_info.value)
        assert "checkpoint.output" in diagnostic
        assert "LIVEPEMSECRET713" not in diagnostic
        assert wrapped not in diagnostic


@pytest.mark.parametrize(
    "partial_block",
    [
        "-----BEGIN PRIVATE KEY-----\nLIVEPEMSECRET713",
        (
            "-----BEGIN PRIVATE KEY-----\n"
            "LIVEPEMSECRET713\n"
            "-----END RSA PRIVATE KEY-----"
        ),
        "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE",
        (
            "-----BEGIN PRIVATE KEY-----"
            "MIIEowIBAAKCAQEA713LIVEPAYLOAD=="
            "-----END PRIVATE KEY-----"
        ),
        "-----BEGIN PRIVATE KEY----- MIIE713LIVEPAYLOAD== -----END PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----MIIE713LIVEPAYLOAD==",
        ("-----BEGIN PRIVATE KEY-----\n-----BEGIN PRIVATE KEY-----\n"),
    ],
)
def test_partial_or_nested_private_key_material_fails_closed(
    partial_block: str,
) -> None:
    config = {"artifact": partial_block}

    assert find_inline_secret_paths(config) == ("artifact",)
    assert redact_sensitive_config(config) == {"artifact": "<redacted>"}
    with pytest.raises(InlineSecretError) as exc_info:
        ensure_no_inline_secrets(config, context="durable artifact")
    assert "LIVEPEMSECRET713" not in str(exc_info.value)
    assert "YOUR_PRIVATE_KEY_HERE" not in str(exc_info.value)


@pytest.mark.parametrize("rounds", [5, 8])
def test_deep_percent_encoded_private_key_material_fails_closed(rounds: int) -> None:
    private_key_block = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIE-LIVEPEMSECRET713-PAYLOAD==\n"
        "-----END PRIVATE KEY-----"
    )
    encoded = _percent_encode(private_key_block, rounds)

    assert find_inline_secret_paths({"artifact": encoded}) == ("artifact",)
    assert redact_sensitive_config({"artifact": encoded}) == {"artifact": "<redacted>"}
    with pytest.raises(InlineSecretError) as exc_info:
        ensure_no_inline_secrets({"artifact": encoded}, context="durable artifact")
    diagnostic = str(exc_info.value)
    assert "LIVEPEMSECRET713" not in diagnostic
    assert encoded not in diagnostic


@pytest.mark.parametrize("quote_layers", [3, 4, 7])
def test_fully_byte_encoded_private_key_detects_boundary_and_exhaustion(
    quote_layers: int,
) -> None:
    private_key_block = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIE-LIVEPEMSECRET713-PAYLOAD==\n"
        "-----END PRIVATE KEY-----"
    )
    encoded = _percent_encode(
        _percent_encode_all_bytes(private_key_block),
        quote_layers,
    )

    assert find_inline_secret_paths({"artifact": encoded}) == ("artifact",)
    assert redact_sensitive_config({"artifact": encoded}) == {"artifact": "<redacted>"}
    with pytest.raises(InlineSecretError) as exc_info:
        ensure_no_inline_secrets({"artifact": encoded}, context="durable artifact")
    assert "LIVEPEMSECRET713" not in str(exc_info.value)


def test_repeated_private_begin_markers_reject_without_quadratic_scan() -> None:
    marker = "-----BEGIN PRIVATE KEY-----\n"
    _assert_scan_within_declared_byte_budget(
        marker * 1_000,
        marker * 4_000,
        expected_paths=("description",),
    )


def test_sequential_safe_placeholder_blocks_scan_linearly() -> None:
    def make_value(count: int) -> str:
        return "\n".join(
            (
                f"-----BEGIN ALG{index} PRIVATE KEY-----\n"
                "YOUR_PRIVATE_KEY_HERE\n"
                f"-----END ALG{index} PRIVATE KEY-----"
            )
            for index in range(count)
        )

    _assert_scan_within_declared_byte_budget(
        make_value(250),
        make_value(1_000),
        expected_paths=(),
    )


@pytest.mark.parametrize(
    "raw",
    [
        (
            "-----BEGIN PRIVATE KEY-----\n"
            "YOUR_PRIVATE_KEY_HERE\n"
            "-----END PRIVATE KEY-----"
        ),
        (
            "-----BEGIN CERTIFICATE-----\n"
            "MIIC8DCCAdigAwIBAgIUCERTIFICATEPUBLICDATA==\n"
            "-----END CERTIFICATE-----"
        ),
        (
            "The markers -----BEGIN PRIVATE KEY----- and "
            "-----END PRIVATE KEY----- identify private-key material."
        ),
        '{"client_secret": "example"}',
        '{"api_key": "YOUR_API_KEY_HERE"}',
        '{"AccountKey": "${AZURE_STORAGE_ACCOUNT_KEY}"}',
        "Percent encoding is documented here without an assignment.",
    ],
)
def test_fully_byte_encoded_placeholders_and_documentation_remain_safe(
    raw: str,
) -> None:
    encoded = _percent_encode(_percent_encode_all_bytes(raw), 4)
    config = {"documentation": encoded}

    assert find_inline_secret_paths(config) == ()
    assert redact_sensitive_config(config) == config
    ensure_no_inline_secrets(config)


@pytest.mark.parametrize(
    "value",
    [
        (
            "-----BEGIN PUBLIC KEY-----\n"
            "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8APUBLICKEYDATA==\n"
            "-----END PUBLIC KEY-----"
        ),
        (
            "-----BEGIN RSA PUBLIC KEY-----\n"
            "MIIBCgKCAQEARSA_PUBLIC_KEY_DATA==\n"
            "-----END RSA PUBLIC KEY-----"
        ),
        (
            "-----BEGIN CERTIFICATE-----\n"
            "MIIC8DCCAdigAwIBAgIUCERTIFICATEPUBLICDATA==\n"
            "-----END CERTIFICATE-----"
        ),
        (
            "-----BEGIN PGP PUBLIC KEY BLOCK-----\n"
            "mQINBGPUBLICKEYBLOCKDATA==\n"
            "-----END PGP PUBLIC KEY BLOCK-----"
        ),
        json.dumps(
            {
                "certificate": (
                    "-----BEGIN CERTIFICATE-----\n"
                    "MIIC8DCCAdigAwIBAgIUCERTIFICATEPUBLICDATA==\n"
                    "-----END CERTIFICATE-----"
                )
            }
        ),
        (
            "The markers -----BEGIN PRIVATE KEY----- and "
            "-----END PRIVATE KEY----- identify private-key material."
        ),
        json.dumps(
            {
                "private_key": (
                    "-----BEGIN PRIVATE KEY-----\n"
                    "YOUR_PRIVATE_KEY_HERE\n"
                    "-----END PRIVATE KEY-----"
                )
            }
        ),
        _percent_encode(
            "-----BEGIN PRIVATE KEY-----\n"
            "YOUR_PRIVATE_KEY_HERE\n"
            "-----END PRIVATE KEY-----",
            4,
        ),
        (
            "-----BEGIN PRIVATE KEY-----\n"
            "YOUR_PRIVATE_KEY_HERE\n"
            "-----END PRIVATE KEY-----"
        ),
        ("-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----"),
        (
            "-----BEGIN OPENSSH PRIVATE KEY-----\n"
            "${SSH_PRIVATE_KEY}\n"
            "-----END OPENSSH PRIVATE KEY-----"
        ),
        ("-----BEGIN PRIVATE KEY-----\n{{ PRIVATE_KEY }}\n-----END PRIVATE KEY-----"),
        "private key: YOUR_PRIVATE_KEY_HERE\n",
        "shared access signature: required\n",
        "api_key:\n",
        "api_key:\n# schema-only declaration\nother: value\n",
    ],
)
def test_public_pem_and_explicit_private_key_placeholders_remain_safe(
    value: str,
) -> None:
    config = {"documentation": value}

    assert find_inline_secret_paths(config) == ()
    assert redact_sensitive_config(config) == config
    ensure_no_inline_secrets(config)


@pytest.mark.parametrize(
    "value",
    [
        "api_key",
        "api_key=",
        "api-key:   ",
        "api_key: # configured externally",
        "api_key=YOUR_API_KEY_HERE",
        "api_key='your-api-key'",
        "x-api-key: not-used",
        "OPENAI_API_KEY=$OPENAI_API_KEY",
        "OPENAI_API_KEY: '${OPENAI_API_KEY}'",
        "password: <redacted>",
        "client_secret: example",
        "x-api-key: required.",
        "api_key: credential",
        "api_key: null",
        "password=false",
        "password == supplied_at_runtime",
        "Password authentication is required",
        "max_tokens=4096",
        "tokenizer: fast",
        "secretary=Alice",
        "api_key_env=OPENAI_API_KEY",
        "client_secret_file=/run/secrets/client",
        "credential_fields=client_secret",
        "public_key=identifier",
        "Question?token=word",
    ],
)
def test_free_text_sensitive_assignment_detection_preserves_safe_text(
    value: str,
) -> None:
    config = {"description": value}

    assert find_inline_secret_paths(config) == ()
    assert redact_sensitive_config(config) == config
    ensure_no_inline_secrets(config)


@pytest.mark.parametrize(
    "value",
    [
        "Contact user@example.test for access",
        "Bearer token authentication is supported",
        "Use Bearer authentication for this endpoint",
        "Bearer credentials are configured externally",
        "The bearer must authenticate before use",
        "Bearer examples appear below",
        "Authorization guidance is documented here",
        "Authorization is required by policy",
        "Authorization: Bearer $NVIDIA_API_KEY",
        "Authorization: Bearer ${NVIDIA_API_KEY}",
        "Authorization: Bearer ",
        "mailto:help@example.test",
        "What?key=answer",
        "Question?token=word",
        "The URL query?sig=example",
        "foo#key=answer",
        "10:30@room-a",
        "time:10:30@room-a",
        "postgres:public-host/database",
    ],
)
def test_free_text_credential_detection_preserves_benign_text(value: str) -> None:
    config = {"user_prompt": value}

    assert find_inline_secret_paths(config) == ()
    assert redact_sensitive_config(config) == config


@pytest.mark.parametrize(
    "value",
    [
        "pass Authorization: Bearer <your-token>",
        "Authorization: Bearer <access-token>",
        "See https://user@example.com/path for the public example",
        "https://example.com/object?X-Amz-Signature=<signature>",
        "https://example.com/object?sig=%3Csignature%3E",
    ],
)
def test_free_text_credential_placeholders_remain_safe(value: str) -> None:
    config = {"user_prompt": value}

    assert find_inline_secret_paths(config) == ()
    assert redact_sensitive_config(config) == config
    ensure_no_inline_secrets(config)


@pytest.mark.parametrize(
    "value",
    [
        "Authorization: Bearer live-secret-713",
        "https://opaque-token@example.com/path",
        "https://user:password@example.com/path",
        "https://example.com/object?sig=live-secret-713",
    ],
)
def test_documentation_exemptions_do_not_allow_live_credentials(value: str) -> None:
    config = {"user_prompt": value}

    assert find_inline_secret_paths(config) == ("user_prompt",)
    assert redact_sensitive_config(config) == {"user_prompt": "<redacted>"}


@pytest.mark.parametrize(
    ("key", "value", "expected_path", "expected_redacted"),
    [
        (
            "path",
            "/artifact?access_token=relative-secret-713",
            "path",
            "<redacted>",
        ),
        (
            "artifact_path",
            "artifact.usd?X-Amz-Signature=relative-secret-713",
            "artifact_path",
            "<redacted>",
        ),
        (
            "uri",
            "./artifact#token=relative-secret-713",
            "uri",
            "<redacted>",
        ),
        (
            "urls",
            ["../x?key=relative-secret-713"],
            "urls[0]",
            ["<redacted>"],
        ),
        (
            "reference_images",
            ["/asset?access_token=relative-secret-713"],
            "reference_images[0]",
            ["<redacted>"],
        ),
        (
            "reference_pdfs",
            ["doc.pdf?X-Amz-Signature=relative-secret-713"],
            "reference_pdfs[0]",
            ["<redacted>"],
        ),
        (
            "dataset",
            "../data?key=relative-secret-713",
            "dataset",
            "<redacted>",
        ),
        (
            "working_dir",
            "./work#token=relative-secret-713",
            "working_dir",
            "<redacted>",
        ),
        (
            "output_dir",
            "/out?sig=relative-secret-713",
            "output_dir",
            "<redacted>",
        ),
        (
            "source",
            "./docs?access_token=relative-secret-713",
            "source",
            "<redacted>",
        ),
        (
            "endpoint",
            "/v1?api_key=relative-secret-713",
            "endpoint",
            "<redacted>",
        ),
        (
            "composition_images",
            ["preview.png?X-Amz-Signature=relative-secret-713"],
            "composition_images[0]",
            ["<redacted>"],
        ),
    ],
)
def test_path_reference_fields_reject_relative_bearer_values(
    key: str,
    value: str | list[str],
    expected_path: str,
    expected_redacted: str | list[str],
) -> None:
    config = {key: value}

    assert find_inline_secret_paths(config) == (expected_path,)
    assert redact_sensitive_config(config) == {key: expected_redacted}


def test_path_reference_fields_preserve_public_query_values() -> None:
    config = {
        "path": "artifact.usd?model=latest",
        "urls": ["../x?public_key=identifier"],
        "composition_images": ["preview.png?model=latest"],
    }

    assert find_inline_secret_paths(config) == ()
    assert redact_sensitive_config(config) == config


def test_path_context_resets_inside_structured_reference_records() -> None:
    config = {
        "references": [
            {
                "description": "What?key=answer",
                "path": "asset.usd?model=latest",
            }
        ]
    }

    assert find_inline_secret_paths(config) == ()
    assert redact_sensitive_config(config) == config


def test_url_secret_detection_preserves_noncredential_parameters_and_references() -> (
    None
):
    config = {
        "base_url": (
            "https://example.test/v1?model=latest&max_tokens=1024"
            "&object_key=scene.usd&public_key=identifier"
        ),
        "empty_signature": "https://example.test/object?X-Amz-Signature=",
        "path_only": "https://example.test/token/public",
        "embedded_public": "See wss://stream.example.test/events for details",
        "encoded_public_redirect": (
            "https://gateway.example.test/callback?redirect="
            "https%3A%2F%2Fpublic.example.test%2Fv1%3Fmodel%3Dlatest"
        ),
        "public_protocol_relative": "//cdn.example.test/assets?v=1",
        "encoded_example_user": "https://user%40example.test/path",
        "nested_encoded_example_user": "https://user%2520%2540example.test/path",
        "benign_semicolon": "https://example.test/search?matrix=a;b&model=latest",
    }

    assert find_inline_secret_paths(config) == ()
    assert redact_sensitive_config(config) == config
    ensure_no_inline_secrets(config)


def test_url_secret_detection_handles_deep_redirect_nesting() -> None:
    nested_url = "https://user:password@private.example.test/v1"
    for _ in range(20):
        nested_url = (
            "https://gateway.example.test/callback?redirect="
            f"{quote(nested_url, safe='')}"
        )

    assert find_inline_secret_paths({"callback": nested_url}) == ("callback",)


def test_url_secret_detection_fails_closed_beyond_nesting_bound() -> None:
    nested_url = "https://public.example.test/v1?model=latest"
    for _ in range(20):
        nested_url = (
            "https://gateway.example.test/callback?redirect="
            f"{quote(nested_url, safe='')}"
        )

    # The deepest URI cannot be proven credential-free within bounded work.
    assert find_inline_secret_paths({"callback": nested_url}) == ("callback",)


def test_url_secret_detection_fails_closed_on_excessive_percent_encoding() -> None:
    encoded_url = "https://user:password@private.example.test/v1"
    for _ in range(12):
        encoded_url = quote(encoded_url, safe="")

    assert find_inline_secret_paths({"callback": encoded_url}) == ("callback",)


@pytest.mark.parametrize(
    ("once_encoded_component", "location"),
    [
        ("%40", "authority"),
        ("%61ccess_token", "query"),
        ("%70ublic-segment", "path"),
    ],
    ids=["authority-delimiter", "query-key", "unresolved-public-segment"],
)
def test_url_secret_detection_fails_closed_on_unresolved_visible_uri_content(
    once_encoded_component: str,
    location: str,
) -> None:
    """No URI component may cross the bounded decoder unresolved."""
    encoded_component = _percent_encode(once_encoded_component, 4)
    if location == "authority":
        value = f"https://user:password{encoded_component}private.example.test/v1"
    elif location == "query":
        value = f"https://public.example.test/v1?{encoded_component}=opaque"
    else:
        value = f"https://public.example.test/{encoded_component}"

    assert find_inline_secret_paths({"callback": value}) == ("callback",)


def test_url_secret_detection_preserves_bounded_nested_public_redirects() -> None:
    nested_url = "//public.example.test/v1?model=latest"
    for _ in range(3):
        nested_url = (
            "https://gateway.example.test/callback?redirect="
            f"{quote(nested_url, safe='')}"
        )

    config = {"callback": nested_url}
    assert find_inline_secret_paths(config) == ()
    ensure_no_inline_secrets(config)


@pytest.mark.parametrize(
    "raw_template",
    [
        "https://assets.example.test/model.usd?X-Amz-Signature={secret}",
        "https://{secret}@assets.example.test/model.usd",
    ],
    ids=["signed-query", "username-only-userinfo"],
)
def test_path_normalization_cannot_hide_uri_credentials(
    tmp_path: Path,
    raw_template: str,
) -> None:
    secret = "path-normalization-secret-713"
    raw = raw_template.format(secret=secret)
    path = Path(raw)
    resolved_path = (tmp_path / path).resolve()

    for value in (raw, path, str(path), resolved_path, str(resolved_path)):
        assert find_inline_secret_paths({"artifact": value}) == ("artifact",)
        assert redact_sensitive_path(value) == "<redacted>"
        with pytest.raises(ValueError) as exc_info:
            ensure_no_inline_secrets({"artifact": value})
        assert secret not in str(exc_info.value)


@pytest.mark.parametrize(
    "path",
    [
        ".pipeline_temp/predict.yaml",
        "cache/.pipeline_temp/nested/config.yaml",
        Path("/sessions/one/cache/.pipeline_temp/config.yaml"),
        r"cache\.pipeline_temp\config.yaml",
    ],
)
def test_is_pipeline_temp_path_matches_complete_path_component(
    path: str | Path,
) -> None:
    assert is_pipeline_temp_path(path)


@pytest.mark.parametrize(
    "path",
    ["pipeline_temp/config.yaml", "cache/.pipeline_temporary/file", "output/result"],
)
def test_is_pipeline_temp_path_rejects_similar_names(path: str) -> None:
    assert not is_pipeline_temp_path(path)


def test_remove_legacy_pipeline_temp_removes_retained_tree(tmp_path: Path) -> None:
    temp_tree = tmp_path / ".pipeline_temp"
    nested = temp_tree / "nested"
    nested.mkdir(parents=True)
    (nested / "config.yaml").write_text("api_key: secret", encoding="utf-8")

    assert remove_legacy_pipeline_temp(tmp_path)
    assert not temp_tree.exists()
    assert not remove_legacy_pipeline_temp(tmp_path)


def test_remove_legacy_pipeline_temp_unlinks_symlink_without_following(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    (tmp_path / ".pipeline_temp").symlink_to(outside, target_is_directory=True)

    assert remove_legacy_pipeline_temp(tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / ".pipeline_temp").exists()


def test_descriptor_confined_artifact_reads_reject_aliases_and_windows_keys(
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "session"
    safe_dir = session_root / "input"
    safe_dir.mkdir(parents=True)
    safe_file = safe_dir / "result.json"
    safe_file.write_text("{}", encoding="utf-8")
    reserved_file = session_root / "cache" / ".pipeline_temp" / "config.yaml"
    reserved_file.parent.mkdir(parents=True)
    reserved_file.write_text("credential", encoding="utf-8")
    outside_file = tmp_path / "outside.json"
    outside_file.write_text("{}", encoding="utf-8")

    (session_root / "safe-alias.json").symlink_to(safe_file)
    (session_root / "reserved-alias.yaml").symlink_to(reserved_file)
    (session_root / "outside-alias.json").symlink_to(outside_file)

    assert visible_local_artifact_key(session_root, r"input\result.json") == (
        "input/result.json"
    )
    artifact = open_held_confined_artifact(session_root, r"input\result.json")
    try:
        assert artifact.stream.read() == b"{}"
    finally:
        artifact.stream.close()
    for alias in ("safe-alias.json", "reserved-alias.yaml", "outside-alias.json"):
        with pytest.raises(ArtifactPathError):
            open_held_confined_artifact(session_root, alias)
    assert (
        visible_local_artifact_key(session_root, r"cache\.pipeline_temp\config.yaml")
        is None
    )
    assert visible_local_artifact_key(session_root, "../outside.json") is None


def test_reserved_artifact_writes_and_cleanup_fail_value_free(tmp_path: Path) -> None:
    secret = "cleanup-boundary-secret-727"
    with open_confined_directory(tmp_path) as root_descriptor:
        with pytest.raises(ValueError) as write_error:
            write_bytes_to_confined(
                root_descriptor,
                f"cache/.pipeline_temp/{secret}.yaml",
                b"rejected",
            )
    with pytest.raises(ValueError) as cleanup_error:
        confined_cleanup_path(tmp_path / secret, tmp_path / "owned")

    for error in (write_error.value, cleanup_error.value):
        assert secret not in str(error)
        assert error.__cause__ is None


def test_session_path_rejects_valid_uuid_symlink_to_outside(tmp_path: Path) -> None:
    session_id = "11111111-1111-4111-8111-111111111111"
    storage_root = tmp_path / "sessions"
    storage_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (storage_root / session_id).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError) as exc_info:
        confined_session_path(storage_root, session_id)

    assert exc_info.value.__cause__ is None


def test_session_child_helpers_do_not_resolve_checked_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "11111111-1111-4111-8111-111111111111"
    storage_root = tmp_path / "sessions"
    storage_root.mkdir()
    original_resolve = Path.resolve

    def guarded_resolve(path: Path, strict: bool = False) -> Path:
        if path.parent == storage_root and path.name in {session_id, "legacy"}:
            raise AssertionError("checked children must remain lexical")
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)

    assert confined_session_path(storage_root, session_id) == storage_root / session_id
    assert (
        confined_storage_child_path(storage_root, "legacy") == storage_root / "legacy"
    )


def test_session_listing_projects_only_unique_uuid_identifiers() -> None:
    safe_a = "11111111-1111-4111-8111-111111111111"
    safe_b = "22222222-2222-4222-8222-222222222222"

    assert safe_listed_session_ids(
        [safe_b, "../escape", ".pipeline_temp", safe_a, safe_b, 7]
    ) == [safe_a, safe_b]
