# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable diagnostics for model-provider authentication failures."""

from __future__ import annotations

import logging
import re
from typing import Any

MODEL_AUTHENTICATION_FAILURE_MESSAGE = (
    "Model authentication failed. Verify the endpoint-scoped API key for the "
    "selected model endpoint; hosted providers use their documented credential "
    "environment variable, while documented local no-auth endpoints require "
    "the explicit no-auth setting."
)
SANITIZED_EXCEPTION_FAILURE_MESSAGE = (
    "Operation failed. Detailed diagnostics were sanitized."
)

_AUTH_ERROR_NAMES = {
    "authenticationerror",
    "permissiondeniederror",
    "unauthorizederror",
}
_AUTH_TEXT_PATTERN = re.compile(
    r"(?:\b401\b|\bunauthori[sz]ed\b|invalid[_ -]?api[_ -]?key|"
    r"incorrect[_ -]?api[_ -]?key|authentication (?:error|failed|failure))",
    re.IGNORECASE,
)
_EXCEPTION_CHAIN_LIMIT = 16


class ModelAuthenticationFailure(RuntimeError):
    """Detached, value-free provider authentication failure."""

    def __init__(self) -> None:
        super().__init__(MODEL_AUTHENTICATION_FAILURE_MESSAGE)


def _exception_chain(
    error: BaseException,
) -> tuple[tuple[BaseException, ...], bool]:
    pending = [error]
    seen: set[int] = set()
    chain: list[BaseException] = []
    while pending and len(seen) < _EXCEPTION_CHAIN_LIMIT:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    truncated = any(id(error) not in seen for error in pending)
    return tuple(chain), truncated


def _status_code(error: BaseException) -> int | None:
    for source in (error, getattr(error, "response", None)):
        if source is None:
            continue
        value = getattr(source, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def is_model_authentication_error(value: Any) -> bool:
    """Return whether a value represents an authentication/authorization failure."""
    if isinstance(value, BaseException):
        chain, _truncated = _exception_chain(value)
        for error in chain:
            if isinstance(error, ModelAuthenticationFailure):
                return True
            if type(error).__name__.lower() in _AUTH_ERROR_NAMES:
                return True
            if _status_code(error) in (401, 403):
                return True
            try:
                message = str(error)
            except Exception:
                continue
            if _AUTH_TEXT_PATTERN.search(message):
                return True
        return False
    return isinstance(value, str) and bool(_AUTH_TEXT_PATTERN.search(value))


def raise_for_model_authentication(error: BaseException) -> None:
    """Replace an auth exception with a detached, value-free failure."""
    if is_model_authentication_error(error):
        raise ModelAuthenticationFailure() from None


def public_model_failure_message(error: BaseException, fallback: str) -> str:
    """Return the stable auth message or a caller-owned generic fallback."""
    if is_model_authentication_error(error):
        return MODEL_AUTHENTICATION_FAILURE_MESSAGE
    return fallback


class ModelAuthenticationLogFilter(logging.Filter):
    """Remove provider bodies and traceback graphs from auth log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        error = record.exc_info[1] if record.exc_info else None
        chain_truncated = False
        if isinstance(error, BaseException):
            _chain, chain_truncated = _exception_chain(error)
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive logging boundary
            message = ""
        authentication_failure = is_model_authentication_error(
            error
        ) or is_model_authentication_error(message)
        if chain_truncated or authentication_failure:
            record.msg = (
                MODEL_AUTHENTICATION_FAILURE_MESSAGE
                if authentication_failure
                else SANITIZED_EXCEPTION_FAILURE_MESSAGE
            )
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return True
