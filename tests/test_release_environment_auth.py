# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release-gate tests for numeric environment and model authentication handling."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from world_understanding.agentic.base_pipeline_executor import (
    safe_step_failure_message,
)
from world_understanding.agentic.config.model_credentials import (
    validate_selected_model_credentials,
)
from world_understanding.utils.model_auth import (
    MODEL_AUTHENTICATION_FAILURE_MESSAGE,
    SANITIZED_EXCEPTION_FAILURE_MESSAGE,
    ModelAuthenticationFailure,
    ModelAuthenticationLogFilter,
    is_model_authentication_error,
    raise_for_model_authentication,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_defaults_import(
    package: str,
    values: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in (
        "PA_VLM_TEMPERATURE",
        "PA_VLM_MAX_TOKENS",
        "PA_VLM_MAX_WORKERS",
        "PA_JUDGE_TEMPERATURE",
        "PA_JUDGE_MAX_TOKENS",
        "JA_VLM_MAX_TOKENS",
    ):
        env.pop(name, None)
    env.update(values)
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(REPO_ROOT),
            str(REPO_ROOT / "apps" / "physics_agent"),
            str(REPO_ROOT / "apps" / "joint_agent"),
        )
    )
    fields = (
        "DEFAULT_VLM_TEMPERATURE, DEFAULT_VLM_MAX_TOKENS, "
        "DEFAULT_VLM_MAX_WORKERS, DEFAULT_JUDGE_TEMPERATURE, "
        "DEFAULT_JUDGE_MAX_TOKENS"
        if package == "physics_agent"
        else "DEFAULT_VLM_MAX_TOKENS"
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, logging; logging.basicConfig(); "
                f"from {package}.api.defaults import {fields}; "
                "print(json.dumps({name: globals()[name] for name in "
                f"{[field.strip() for field in fields.split(',')]!r}}}))"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_physics_numeric_environment_overrides_are_valid_in_fresh_process() -> None:
    result = _run_defaults_import(
        "physics_agent",
        {
            "PA_VLM_TEMPERATURE": "1.5",
            "PA_VLM_MAX_TOKENS": "8192",
            "PA_VLM_MAX_WORKERS": "8",
            "PA_JUDGE_TEMPERATURE": "0.25",
            "PA_JUDGE_MAX_TOKENS": "1024",
        },
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "DEFAULT_VLM_TEMPERATURE": 1.5,
        "DEFAULT_VLM_MAX_TOKENS": 8192,
        "DEFAULT_VLM_MAX_WORKERS": 8,
        "DEFAULT_JUDGE_TEMPERATURE": 0.25,
        "DEFAULT_JUDGE_MAX_TOKENS": 1024,
    }
    assert "Invalid PA_" not in result.stderr


@pytest.mark.parametrize(
    ("values", "expected_name"),
    [
        ({"PA_VLM_TEMPERATURE": "secret-bad-value"}, "PA_VLM_TEMPERATURE"),
        ({"PA_VLM_MAX_TOKENS": ""}, "PA_VLM_MAX_TOKENS"),
        ({"PA_VLM_MAX_WORKERS": "0"}, "PA_VLM_MAX_WORKERS"),
        ({"PA_VLM_MAX_WORKERS": "1000000"}, "PA_VLM_MAX_WORKERS"),
        ({"PA_JUDGE_TEMPERATURE": "3"}, "PA_JUDGE_TEMPERATURE"),
        ({"PA_JUDGE_MAX_TOKENS": "-1"}, "PA_JUDGE_MAX_TOKENS"),
    ],
)
def test_physics_invalid_numeric_environment_falls_back_without_value_leak(
    values: dict[str, str], expected_name: str
) -> None:
    result = _run_defaults_import("physics_agent", values)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "DEFAULT_VLM_TEMPERATURE": 1.0,
        "DEFAULT_VLM_MAX_TOKENS": 24576,
        "DEFAULT_VLM_MAX_WORKERS": 64,
        "DEFAULT_JUDGE_TEMPERATURE": 0.0,
        "DEFAULT_JUDGE_MAX_TOKENS": 2048,
    }
    warning_lines = [
        line
        for line in result.stderr.splitlines()
        if f"Invalid {expected_name};" in line
    ]
    assert len(warning_lines) == 1
    warning = warning_lines[0]
    assert "expected" in warning
    assert "Using default" in warning
    for rejected_value in values.values():
        if rejected_value:
            assert rejected_value not in warning


def test_joint_invalid_numeric_environment_falls_back_in_fresh_process() -> None:
    result = _run_defaults_import(
        "joint_agent", {"JA_VLM_MAX_TOKENS": "secret-bad-value"}
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"DEFAULT_VLM_MAX_TOKENS": 24576}
    assert "JA_VLM_MAX_TOKENS" in result.stderr
    assert "secret-bad-value" not in result.stderr


class AuthenticationError(RuntimeError):
    status_code = 401


def _raised_authentication_error(message: str) -> AuthenticationError:
    try:
        raise AuthenticationError(message)
    except AuthenticationError as error:
        return error


def test_authentication_failure_is_stable_and_detached() -> None:
    error = _raised_authentication_error(
        "401 provider body contained bearer-secret and request internals"
    )

    assert error.__traceback__ is not None
    assert is_model_authentication_error(error)
    assert safe_step_failure_message(error) == MODEL_AUTHENTICATION_FAILURE_MESSAGE
    with pytest.raises(ModelAuthenticationFailure) as exc_info:
        raise_for_model_authentication(error)
    assert str(exc_info.value) == MODEL_AUTHENTICATION_FAILURE_MESSAGE
    assert "bearer-secret" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_authentication_log_filter_removes_body_and_traceback() -> None:
    error = _raised_authentication_error("401 provider body contained bearer-secret")
    assert error.__traceback__ is not None
    record = logging.LogRecord(
        "test",
        logging.ERROR,
        __file__,
        1,
        "provider failed: %s",
        (error,),
        (type(error), error, error.__traceback__),
    )

    assert ModelAuthenticationLogFilter().filter(record)
    assert record.getMessage() == MODEL_AUTHENTICATION_FAILURE_MESSAGE
    assert record.exc_info is None
    assert "bearer-secret" not in record.getMessage()


def test_authentication_log_filter_sanitizes_truncated_exception_group() -> None:
    error = ExceptionGroup(
        "large provider failure",
        [
            AuthenticationError("401 provider body contained bearer-secret"),
            *(RuntimeError(f"ordinary failure {index}") for index in range(20)),
        ],
    )
    record = logging.LogRecord(
        "test",
        logging.ERROR,
        __file__,
        1,
        "provider failed after a large exception group",
        (),
        (type(error), error, error.__traceback__),
    )

    assert ModelAuthenticationLogFilter().filter(record)
    assert record.getMessage() == SANITIZED_EXCEPTION_FAILURE_MESSAGE
    assert record.exc_info is None
    assert "bearer-secret" not in record.getMessage()


def test_authentication_log_filter_does_not_mislabel_truncated_non_auth_group() -> None:
    error = ExceptionGroup(
        "large ordinary failure",
        [RuntimeError(f"ordinary failure {index}") for index in range(20)],
    )
    record = logging.LogRecord(
        "test",
        logging.ERROR,
        __file__,
        1,
        "operation failed after a large exception group",
        (),
        (type(error), error, error.__traceback__),
    )

    assert ModelAuthenticationLogFilter().filter(record)
    assert record.getMessage() == SANITIZED_EXCEPTION_FAILURE_MESSAGE
    assert record.getMessage() != MODEL_AUTHENTICATION_FAILURE_MESSAGE
    assert record.exc_info is None


def test_selected_step_preflight_is_backend_aware_and_respects_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    config = {
        "project": {"working_dir": str(tmp_path / "must-not-exist")},
        "steps": {
            "predict": {
                "enabled": True,
                "vlm": {"backend": "nim", "model": "example"},
            },
            "apply": {"enabled": True},
        },
    }

    validate_selected_model_credentials(
        config,
        tmp_path / "config.yaml",
        [],
        ["apply"],
        get_step_defaults=lambda _step: {},
    )
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        validate_selected_model_credentials(
            config,
            tmp_path / "config.yaml",
            [],
            ["predict"],
            get_step_defaults=lambda _step: {},
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_selected_step_preflight_omits_local_no_auth_hint_for_remote_nim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    config = {
        "steps": {
            "predict": {
                "enabled": True,
                "vlm": {
                    "backend": "nim",
                    "model": "example",
                    "base_url": "https://nim.example.com/v1",
                },
            }
        }
    }

    with pytest.raises(ValueError) as exc_info:
        validate_selected_model_credentials(
            config,
            tmp_path / "config.yaml",
            [],
            [],
            get_step_defaults=lambda _step: {},
        )

    message = str(exc_info.value)
    assert "endpoint-scoped api_key or api_key_env" in message
    assert "not-used" not in message


def test_selected_step_preflight_does_not_fallback_past_missing_api_key_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "hosted-provider-key")
    monkeypatch.delenv("SELECTED_ENDPOINT_KEY", raising=False)
    config = {
        "steps": {
            "predict": {
                "enabled": True,
                "vlm": {
                    "backend": "openai",
                    "model": "example",
                    "api_key_env": "SELECTED_ENDPOINT_KEY",
                },
            }
        }
    }

    with pytest.raises(ValueError) as exc_info:
        validate_selected_model_credentials(
            config,
            tmp_path / "config.yaml",
            [],
            [],
            get_step_defaults=lambda _step: {},
        )
    message = str(exc_info.value)
    assert "configured api_key_env" in message
    assert "SELECTED_ENDPOINT_KEY" not in message
    assert "hosted-provider-key" not in message
