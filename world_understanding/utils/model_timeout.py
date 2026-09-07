# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable timeout types and durable terminal-failure markers."""

from typing import Any

NON_RETRYABLE_VLM_TIMEOUT_MESSAGE = (
    "VLM request timed out without verified remote completion"
)
TERMINAL_VLM_TIMEOUT_RESUME_MESSAGE = (
    "Pipeline resume is blocked after an unverified VLM timeout; "
    "restart with clean=True before dispatching another model request"
)
TERMINAL_VLM_TIMEOUT_CONTEXT_KEY = "terminal_vlm_timeout"

_TERMINAL_FAILURE_SCHEMA_VERSION = 1
_TERMINAL_FAILURE_ERROR_TYPE = "NonRetryableVLMTimeoutError"
_EXCEPTION_GRAPH_LIMIT = 32

# Keep this module independent of optional provider SDKs. Matching an entry
# anywhere in the exception class MRO recognizes SDK subclasses without
# importing (and thereby requiring) their packages.
_MODEL_TIMEOUT_CLASS_IDENTITIES = frozenset(
    {
        ("anthropic", "APITimeoutError"),
        ("botocore.exceptions", "ConnectTimeoutError"),
        ("botocore.exceptions", "ReadTimeoutError"),
        ("google.api_core.exceptions", "DeadlineExceeded"),
        ("httpx", "TimeoutException"),
        ("openai", "APITimeoutError"),
        ("requests.exceptions", "Timeout"),
        ("urllib3.exceptions", "TimeoutError"),
    }
)


class NonRetryableVLMTimeoutError(TimeoutError):
    """A local VLM deadline expired without proof the remote work stopped."""


def _is_model_timeout_instance(error: BaseException) -> bool:
    """Return whether one exception has a recognized timeout type."""
    if isinstance(error, TimeoutError):
        return True
    return any(
        (error_type.__module__, error_type.__name__) in _MODEL_TIMEOUT_CLASS_IDENTITIES
        for error_type in type(error).__mro__
    )


def is_model_timeout_error(error: BaseException) -> bool:
    """Recognize provider timeouts through bounded cause/context traversal.

    OpenAI, Anthropic, Bedrock, Gemini, NIM, and Gradio surface different
    timeout classes. Some adapters wrap the provider exception, so inspect the
    exception graph while avoiding provider imports, cycles, and unbounded
    traversal. A surfaced transport deadline does not prove that remote model
    work stopped and is therefore unsafe to retry automatically.
    """
    pending = [error]
    seen: set[int] = set()
    while pending and len(seen) < _EXCEPTION_GRAPH_LIMIT:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if _is_model_timeout_instance(current):
            return True
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return False


def make_terminal_vlm_timeout_marker(failed_step: str) -> dict[str, Any]:
    """Return a value-safe checkpoint marker for an unverified remote timeout."""
    if type(failed_step) is not str or not failed_step or len(failed_step) > 128:
        raise ValueError(
            "Terminal VLM timeout failed_step must be a non-empty string "
            "of at most 128 characters"
        ) from None
    return {
        "schema_version": _TERMINAL_FAILURE_SCHEMA_VERSION,
        "error_type": _TERMINAL_FAILURE_ERROR_TYPE,
        "terminal": True,
        "failed_step": failed_step,
        "recovery": {"clean": True, "resume": False},
    }


def is_terminal_vlm_timeout_marker(value: Any) -> bool:
    """Return whether *value* is the exact supported terminal marker shape."""
    if type(value) is not dict:
        return False
    failed_step = value.get("failed_step")
    if type(failed_step) is not str or not failed_step or len(failed_step) > 128:
        return False
    return value == make_terminal_vlm_timeout_marker(failed_step)


def raise_for_terminal_vlm_timeout_result(result: dict[str, Any]) -> None:
    """Restore terminal timeout semantics from a value-safe workflow result."""
    if is_terminal_vlm_timeout_marker(result.get(TERMINAL_VLM_TIMEOUT_CONTEXT_KEY)):
        raise NonRetryableVLMTimeoutError(NON_RETRYABLE_VLM_TIMEOUT_MESSAGE) from None
