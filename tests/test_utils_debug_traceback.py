# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the session-local redacted failure diagnostics."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from world_understanding.utils.debug_traceback import (
    STEP_FAILURE_DEBUG_RELATIVE_PATH,
    append_step_failure_debug_entry,
    format_scrubbed_exception,
    scrub_secret_text,
)


def test_scrub_secret_text_removes_obvious_secret_shapes() -> None:
    text = (
        "request to https://api.example.com failed: "
        "nvapi-AbC123_def-456 rejected; "
        "Authorization: Bearer sk.live.token-987; "
        'api_key="inline-secret-value" and API-KEY: another-secret '
        "and token=tok_value password=hunter2"
    )
    scrubbed = scrub_secret_text(text)

    assert "nvapi-AbC123_def-456" not in scrubbed
    assert "sk.live.token-987" not in scrubbed
    assert "inline-secret-value" not in scrubbed
    assert "another-secret" not in scrubbed
    assert "tok_value" not in scrubbed
    assert "hunter2" not in scrubbed
    assert "nvapi-[REDACTED]" in scrubbed
    # Non-secret context is preserved.
    assert "https://api.example.com" in scrubbed


def test_scrub_secret_text_redacts_prefixed_env_var_secret_names() -> None:
    text = (
        "startup failed: OPENAI_API_KEY=sk-abc123 rejected; "
        "GITHUB_TOKEN: ghp_xxx expired; "
        "AWS_SECRET_ACCESS_KEY='wJalrXUtnFEMIsecret' revoked; "
        "DB_PASSWORD=hunter3 and SSH-KEY: id-rsa-material"
    )
    scrubbed = scrub_secret_text(text)

    assert "sk-abc123" not in scrubbed
    assert "ghp_xxx" not in scrubbed
    assert "wJalrXUtnFEMIsecret" not in scrubbed
    assert "hunter3" not in scrubbed
    assert "id-rsa-material" not in scrubbed
    assert "OPENAI_API_KEY=[REDACTED]" in scrubbed
    assert "GITHUB_TOKEN: [REDACTED]" in scrubbed
    assert "AWS_SECRET_ACCESS_KEY=[REDACTED]" in scrubbed
    assert "DB_PASSWORD=[REDACTED]" in scrubbed
    assert "SSH-KEY: [REDACTED]" in scrubbed


def test_scrub_secret_text_redacts_hyphen_infixed_key_names() -> None:
    # Key names cannot dodge the match by splitting their syllables with
    # "-" or "_" (or by carrying a trailing identifier segment).
    text = (
        "pass-word=hunter9; pass_word: hunter8; PASS-WORD: hunter7; "
        "se-cret=split-secret-1; to-ken: split-token-2; "
        "pass-phrase=split-phrase-3; passwd=split-passwd-4; "
        "password-hash=split-hash-5; a-p-i-k-e-y=split-key-6"
    )
    scrubbed = scrub_secret_text(text)

    for leaked in (
        "hunter9",
        "hunter8",
        "hunter7",
        "split-secret-1",
        "split-token-2",
        "split-phrase-3",
        "split-passwd-4",
        "split-hash-5",
        "split-key-6",
    ):
        assert leaked not in scrubbed
    assert "pass-word=[REDACTED]" in scrubbed
    assert "se-cret=[REDACTED]" in scrubbed
    assert "password-hash=[REDACTED]" in scrubbed


def test_scrub_secret_text_redacts_quoted_json_credentials() -> None:
    text = (
        'provider rejected payload {"api_key": "json-secret-1", '
        '"model": "gemma", "auth_token":"json-secret-2"}'
    )
    scrubbed = scrub_secret_text(text)

    assert "json-secret-1" not in scrubbed
    assert "json-secret-2" not in scrubbed
    assert '"model": "gemma"' in scrubbed


def test_scrub_secret_text_redacts_signed_url_parameters() -> None:
    text = (
        "download failed for https://bucket.s3.amazonaws.com/asset.usdz"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=300"
        "&X-Amz-Signature=deadbeef0123456789 and "
        "https://account.blob.core.windows.net/c/b?sv=2020-08-04"
        "&sig=azure%2Fsig%2Bvalue also ?key=google-api-key-value"
    )
    scrubbed = scrub_secret_text(text)

    assert "deadbeef0123456789" not in scrubbed
    assert "azure%2Fsig%2Bvalue" not in scrubbed
    assert "google-api-key-value" not in scrubbed
    assert "X-Amz-Signature=[REDACTED]" in scrubbed
    assert "sig=[REDACTED]" in scrubbed
    assert "key=[REDACTED]" in scrubbed
    # Non-credential URL parts are preserved.
    assert "https://bucket.s3.amazonaws.com/asset.usdz" in scrubbed
    assert "X-Amz-Algorithm=AWS4-HMAC-SHA256" in scrubbed
    assert "X-Amz-Expires=300" in scrubbed
    assert "sv=2020-08-04" in scrubbed


def test_format_scrubbed_exception_never_includes_message_content() -> None:
    # Shapes no scrubber can recognize: an unlabelled password and a short
    # bearer token (built without in-file literals so on-disk source lines
    # cannot reintroduce them).
    plain_password = "hun" + "ter2"
    short_bearer = "Bearer " + "abc"
    message = (
        f"login failed for {plain_password}; Authorization: {short_bearer}; "
        'also {"api_key":"exc-json-secret"} and '
        "https://b.s3.amazonaws.com/a?X-Amz-Signature=exc-sig-secret"
    )

    def _boom() -> None:
        raise RuntimeError(message)

    try:
        _boom()
    except RuntimeError as error:
        formatted = format_scrubbed_exception(error)

    # The exception message is withheld entirely: none of its content —
    # recognizable secret shapes or not — reaches the persisted rendering.
    assert "Traceback (most recent call last)" in formatted
    assert "RuntimeError" in formatted
    assert plain_password not in formatted
    assert short_bearer not in formatted
    assert "exc-json-secret" not in formatted
    assert "exc-sig-secret" not in formatted
    assert "login failed" not in formatted
    assert "exception message withheld" in formatted


def test_format_scrubbed_exception_handles_missing_traceback() -> None:
    # A never-raised exception instance still yields a structured entry
    # instead of raising (the append boundary would otherwise swallow it).
    formatted = format_scrubbed_exception(ValueError("x"))

    assert "Traceback (most recent call last)" in formatted
    assert "[no traceback frames available]" in formatted
    assert "ValueError" in formatted
    assert "exception message withheld" in formatted


def test_scrub_secret_text_redacts_additional_credential_shapes() -> None:
    text = (
        "bare token sk-proj-Abc123DefGhi456 leaked\n"
        "userinfo https://svc-account:p4ssw0rd@internal.example.com/v1 dialed\n"
        "Set-Cookie: session=0a1b2c3d4e5f; Path=/\n"
        "auth header was Basic dXNlcjpwYXNzd29yZDEyMw== today\n"
        "Authorization: Basic dXNlcjpwYXNzd29yZDEyMw== sent\n"
        "github token ghp_AbC123def456GHI789 revoked"
    )
    scrubbed = scrub_secret_text(text)

    assert "sk-proj-Abc123DefGhi456" not in scrubbed
    assert "p4ssw0rd" not in scrubbed
    assert "://[REDACTED]@internal.example.com" in scrubbed
    assert "0a1b2c3d4e5f" not in scrubbed
    assert "Set-Cookie: [REDACTED]" in scrubbed
    # Basic credentials lose the base64 payload, not just the scheme word.
    assert "dXNlcjpwYXNzd29yZDEyMw" not in scrubbed
    assert "Basic [REDACTED]" in scrubbed
    assert "ghp_AbC123def456GHI789" not in scrubbed


def test_scrub_secret_text_redacts_entropy_like_tokens() -> None:
    # No recognizable prefix or key name: the conservative last pass still
    # removes long mixed-alphanumeric tokens.
    text = "opaque blob A1b2C3d4E5f6G7h8I9j0K1L2 returned by provider"
    scrubbed = scrub_secret_text(text)

    assert "A1b2C3d4E5f6G7h8I9j0K1L2" not in scrubbed
    assert "[REDACTED]" in scrubbed
    assert "returned by provider" in scrubbed


def test_scrub_secret_text_redacts_bare_bearer_tokens() -> None:
    scrubbed = scrub_secret_text("header was 'Bearer sk.live.token-987' today")
    assert "sk.live.token-987" not in scrubbed
    assert "Bearer [REDACTED]" in scrubbed


def test_scrub_secret_text_keeps_ordinary_text_unchanged() -> None:
    text = "ValueError: dataset.jsonl has 0 rows (expected > 0)"
    assert scrub_secret_text(text) == text


def test_format_scrubbed_exception_has_traceback_without_locals() -> None:
    secret_local = "nvapi-frame-local-secret"

    def _boom() -> None:
        credential = secret_local  # noqa: F841 - must not leak via locals
        raise ValueError("provider rejected api_key=nvapi-frame-local-secret")

    try:
        _boom()
    except ValueError as error:
        formatted = format_scrubbed_exception(error)

    assert "Traceback (most recent call last)" in formatted
    assert "_boom" in formatted
    assert "ValueError" in formatted
    assert "nvapi-frame-local-secret" not in formatted


def test_append_step_failure_debug_entry_appends_entries(tmp_path: Path) -> None:
    try:
        raise RuntimeError("first failure api_key=nvapi-first-secret")
    except RuntimeError as error:
        debug_path = append_step_failure_debug_entry(tmp_path, "predict", error)

    assert debug_path == tmp_path / STEP_FAILURE_DEBUG_RELATIVE_PATH
    assert debug_path is not None and debug_path.exists()

    try:
        raise ValueError("second failure")
    except ValueError as error:
        assert append_step_failure_debug_entry(tmp_path, "render", error) == debug_path

    content = debug_path.read_text(encoding="utf-8")
    assert "step failure: predict" in content
    assert "step failure: render" in content
    assert "RuntimeError" in content
    assert "ValueError" in content
    assert "nvapi-first-secret" not in content
    # Entries carry an ISO timestamp.
    assert content.count("] step failure: ") == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permission semantics")
def test_append_step_failure_debug_entry_tightens_existing_permissions(
    tmp_path: Path,
) -> None:
    debug_path = tmp_path / STEP_FAILURE_DEBUG_RELATIVE_PATH
    debug_path.parent.mkdir(parents=True)
    debug_path.write_text("pre-existing entry\n", encoding="utf-8")
    debug_path.chmod(0o644)

    try:
        raise RuntimeError("failure with api_key=nvapi-permissive-secret")
    except RuntimeError as error:
        assert append_step_failure_debug_entry(tmp_path, "predict", error) == debug_path

    # A pre-existing permissive log is tightened before the append.
    assert stat.S_IMODE(debug_path.stat().st_mode) == 0o600
    content = debug_path.read_text(encoding="utf-8")
    assert content.startswith("pre-existing entry\n")
    assert "step failure: predict" in content
    assert "nvapi-permissive-secret" not in content


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permission semantics")
def test_append_step_failure_debug_entry_creates_private_file(
    tmp_path: Path,
) -> None:
    try:
        raise RuntimeError("fresh failure")
    except RuntimeError as error:
        debug_path = append_step_failure_debug_entry(tmp_path, "predict", error)

    assert debug_path is not None
    assert stat.S_IMODE(debug_path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink semantics")
def test_append_step_failure_debug_entry_refuses_symlinked_debug_dir(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    working_dir = tmp_path / "work"
    working_dir.mkdir()
    temp_dir = working_dir / ".pipeline_temp"
    temp_dir.mkdir()
    (temp_dir / "debug").symlink_to(outside, target_is_directory=True)

    try:
        raise RuntimeError("boom behind a symlinked debug directory")
    except RuntimeError as error:
        assert append_step_failure_debug_entry(working_dir, "predict", error) is None

    # Nothing escaped the session: the symlink target stays untouched.
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink semantics")
def test_append_step_failure_debug_entry_refuses_symlinked_log_file(
    tmp_path: Path,
) -> None:
    outside_log = tmp_path / "outside.log"
    outside_log.write_text("untouched\n", encoding="utf-8")
    working_dir = tmp_path / "work"
    debug_dir = (working_dir / STEP_FAILURE_DEBUG_RELATIVE_PATH).parent
    debug_dir.mkdir(parents=True)
    (debug_dir / STEP_FAILURE_DEBUG_RELATIVE_PATH.name).symlink_to(outside_log)

    try:
        raise RuntimeError("boom behind a symlinked log file")
    except RuntimeError as error:
        assert append_step_failure_debug_entry(working_dir, "predict", error) is None

    assert outside_log.read_text(encoding="utf-8") == "untouched\n"


def test_debug_log_path_is_excluded_from_session_sync() -> None:
    from world_understanding.utils.artifacts import (
        is_pipeline_temp_path,
        validated_artifact_relative_key,
    )

    relative = STEP_FAILURE_DEBUG_RELATIVE_PATH.as_posix()
    # The service session stores (local and S3) skip any path that enters
    # ``.pipeline_temp`` when listing/synchronizing session files, so the
    # debug log is never uploaded or downloaded as a session artifact.
    assert is_pipeline_temp_path(relative)
    assert is_pipeline_temp_path(Path("cache") / STEP_FAILURE_DEBUG_RELATIVE_PATH)
    # The canonical artifact-key validation rejects the namespace outright.
    with pytest.raises(ValueError):
        validated_artifact_relative_key(relative)


@pytest.mark.skipif(not hasattr(os, "fchmod"), reason="requires os.fchmod")
def test_append_step_failure_debug_entry_fails_closed_without_fchmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    debug_path = tmp_path / STEP_FAILURE_DEBUG_RELATIVE_PATH
    debug_path.parent.mkdir(parents=True)
    debug_path.write_text("pre-existing entry\n", encoding="utf-8")
    debug_path.chmod(0o644)

    def _fchmod_fails(fd: int, mode: int) -> None:
        raise OSError("fchmod denied")

    monkeypatch.setattr(os, "fchmod", _fchmod_fails)
    try:
        raise RuntimeError("boom with api_key=nvapi-fail-closed-secret")
    except RuntimeError as error:
        assert append_step_failure_debug_entry(tmp_path, "predict", error) is None

    # Tightening happens BEFORE the write: when it fails, no bytes land on
    # the still-permissive file.
    assert debug_path.read_text(encoding="utf-8") == "pre-existing entry\n"


def test_append_step_failure_debug_entry_bounds_huge_messages(
    tmp_path: Path,
) -> None:
    huge_payload = "payload " + ("x" * 5_000_000)
    try:
        raise RuntimeError(huge_payload)
    except RuntimeError as error:
        debug_path = append_step_failure_debug_entry(tmp_path, "predict", error)

    assert debug_path is not None
    # The message is never persisted, so a multi-megabyte embedded payload
    # cannot reach the log at all.
    assert debug_path.stat().st_size < 20_000
    content = debug_path.read_text(encoding="utf-8")
    assert "xxxxx" not in content
    assert "step failure: predict" in content
    assert "exception message withheld" in content


def test_append_step_failure_debug_entry_skips_at_log_size_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import world_understanding.utils.debug_traceback as debug_traceback

    monkeypatch.setattr(debug_traceback, "_MAX_LOG_BYTES", 1024)
    debug_path = tmp_path / STEP_FAILURE_DEBUG_RELATIVE_PATH
    debug_path.parent.mkdir(parents=True)
    debug_path.write_text("x" * 1024, encoding="utf-8")
    debug_path.chmod(0o600)

    try:
        raise RuntimeError("beyond the size cap")
    except RuntimeError as error:
        assert append_step_failure_debug_entry(tmp_path, "predict", error) is None

    # The capped log is left exactly as it was.
    assert debug_path.stat().st_size == 1024


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires os.mkfifo")
def test_append_step_failure_debug_entry_refuses_fifo_log(tmp_path: Path) -> None:
    debug_path = tmp_path / STEP_FAILURE_DEBUG_RELATIVE_PATH
    debug_path.parent.mkdir(parents=True)
    os.mkfifo(debug_path)

    try:
        raise RuntimeError("boom behind a FIFO log path")
    except RuntimeError as error:
        # A reader-less FIFO must fail the non-blocking open immediately
        # instead of hanging the failure path.
        assert append_step_failure_debug_entry(tmp_path, "predict", error) is None

    assert stat.S_ISFIFO(debug_path.lstat().st_mode)


def test_append_step_failure_debug_entry_without_working_dir() -> None:
    try:
        raise RuntimeError("no session directory")
    except RuntimeError as error:
        assert append_step_failure_debug_entry(None, "predict", error) is None
