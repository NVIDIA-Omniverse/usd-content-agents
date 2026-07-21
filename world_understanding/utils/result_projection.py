# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credential-safe projections for published result metadata."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePath
from typing import Any

from world_understanding.utils.credentials import (
    redact_sensitive_config,
    redact_sensitive_path,
)

_RUNTIME_ONLY_RESULT_KEYS = frozenset(
    {
        "cancel_checker",
        "config_dict",
        "event_listener",
        "listener",
        "object_store",
        "path_resolver",
        "workflow",
    }
)
_UNTRUSTED_DIAGNOSTIC_KEY_TOKENS = frozenset(
    {
        "cause",
        "detail",
        "details",
        "error",
        "errors",
        "exception",
        "exceptions",
        "failure",
        "failures",
        "message",
        "msg",
        "msgs",
        "trace",
        "traces",
        "traceback",
        "tracebacks",
    }
)


def _is_untrusted_diagnostic_key(key: str) -> bool:
    """Identify fields whose text may originate in a provider exception."""
    snake_case = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", key)
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", snake_case)
    tokens = tuple(
        token for token in re.split(r"[^A-Za-z0-9]+", snake_case.lower()) if token
    )
    return bool(tokens and tokens[-1] in _UNTRUSTED_DIAGNOSTIC_KEY_TOKENS)


def _diagnostic_value_requires_omission(value: Any) -> bool:
    """Preserve value-free state/count fields but omit text-bearing payloads."""
    value_type = type(value)
    if value is None or value_type in {bool, int, float}:
        return False
    if value_type in {dict, list, tuple, set, frozenset} and not value:
        return False
    return True


def project_result_metadata(value: Any) -> dict[str, Any]:
    """Return a detached, credential-safe mapping for a public result object.

    Workflow contexts intentionally retain caller configuration and runtime
    collaborators while work is executing. Those objects are not result data.
    At every nesting level, copy only built-in primitives and containers plus
    ``pathlib`` paths; omit runtime-only keys, provider-authored diagnostic
    text, callables, unsupported objects, and cycles without rendering them.
    Code-owned callers may add fixed public error codes/messages after this
    projection. The projection never mutates the caller-owned mapping.
    """
    if type(value) is not dict:
        return {}

    omitted = object()
    active_container_ids: set[int] = set()

    def detach(item: Any) -> Any:
        """Copy supported data without rendering unsupported runtime objects."""
        item_type = type(item)
        if item_type in {type(None), bool, int, float, str, bytes, bytearray}:
            return item
        if isinstance(item, PurePath):
            return item

        if item_type not in {dict, list, tuple, set, frozenset}:
            return omitted

        item_id = id(item)
        if item_id in active_container_ids:
            return omitted
        active_container_ids.add(item_id)
        try:
            if item_type is dict:
                detached_mapping: dict[str, Any] = {}
                for key, child in item.items():
                    if (
                        type(key) is not str
                        or key in _RUNTIME_ONLY_RESULT_KEYS
                        or (
                            _is_untrusted_diagnostic_key(key)
                            and _diagnostic_value_requires_omission(child)
                        )
                    ):
                        continue
                    detached_child = detach(child)
                    if detached_child is not omitted:
                        detached_mapping[key] = detached_child
                return detached_mapping

            detached_items = []
            for child in item:
                detached_child = detach(child)
                if detached_child is not omitted:
                    detached_items.append(detached_child)
            if item_type is tuple:
                return tuple(detached_items)
            if item_type is set:
                return set(detached_items)
            if item_type is frozenset:
                return frozenset(detached_items)
            return detached_items
        finally:
            active_container_ids.remove(item_id)

    result_data = detach(value)
    if not isinstance(result_data, dict):
        return {}
    projected = redact_sensitive_config(result_data)
    return projected if isinstance(projected, dict) else {}


def retain_safe_result_path(value: str | os.PathLike[str] | None) -> Path | None:
    """Preserve an operational result path only when its diagnostic form is safe."""
    if isinstance(value, str) and type(value) is str:
        raw_text = value
    elif isinstance(value, PurePath):
        raw_text = os.fspath(value)
    else:
        return None
    if not raw_text or redact_sensitive_path(value) != raw_text:
        return None
    return Path(value)


def retain_safe_result_text(
    value: Any,
    *,
    path_context: bool = False,
) -> str | None:
    """Preserve a text result only when projection would leave it unchanged."""
    if not isinstance(value, str) or type(value) is not str:
        return None
    projected = redact_sensitive_config(value, _path_context=path_context)
    return value if projected == value else None
