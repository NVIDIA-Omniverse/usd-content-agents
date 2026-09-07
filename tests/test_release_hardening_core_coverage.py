# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Close small core-coverage seams introduced by release hardening."""

from __future__ import annotations

import logging
import os

import pytest

from world_understanding.agentic import tasks, workflows
from world_understanding.functions.models import token_limits, vision_language_models
from world_understanding.utils import environment, model_auth
from world_understanding.utils.safe_repr import SecretSafeReprMixin
from world_understanding.utils.session_paths import confined_storage_child_path


class _SecretSafeValue(SecretSafeReprMixin):
    pass


class _UnprintableError(RuntimeError):
    def __str__(self) -> str:
        raise RuntimeError("unprintable")


class _InvalidPathLike(os.PathLike[str]):
    def __fspath__(self) -> str:
        raise OSError("unavailable")


def test_release_hardening_small_defensive_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _SecretSafeValue()
    assert repr(value) == "_SecretSafeValue(<redacted>)"
    assert str(value) == "_SecretSafeValue(<redacted>)"

    assert environment._range_description("value", None, 5) == (
        "value less than or equal to 5"
    )
    assert environment._range_description("value", None, None) == "value"

    monkeypatch.setenv("WU_TEST_FLOAT", "-1")
    assert (
        environment.parse_float_env(
            "WU_TEST_FLOAT",
            2.0,
            minimum=0.0,
            logger=logging.getLogger(__name__),
        )
        == 2.0
    )

    assert model_auth.is_model_authentication_error(_UnprintableError()) is False
    assert (
        model_auth.is_model_authentication_error(RuntimeError("invalid api key"))
        is True
    )

    assert token_limits.model_output_token_cap(None) is None

    def unexpected_span_lookup() -> None:
        raise AssertionError("empty token kwargs must return before span lookup")

    monkeypatch.setattr(
        vision_language_models,
        "get_current_span",
        unexpected_span_lookup,
    )
    vision_language_models._record_effective_openai_max_tokens({})

    assert tasks._diagnostic_name(object()) == "<unavailable>"
    assert workflows._diagnostic_name(object()) == "<unavailable>"

    with pytest.raises(ValueError, match="escapes configured root"):
        confined_storage_child_path(_InvalidPathLike(), "session")  # type: ignore[arg-type]
