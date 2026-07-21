# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actionable diagnostics for unknown nested configuration keys."""

from __future__ import annotations

import copy
import difflib
import logging
from collections.abc import Callable, Collection, Mapping
from typing import Any

from world_understanding.utils.credentials import redact_sensitive_path


def build_nested_config_key_schema(
    root_defaults: Mapping[str, Any],
    step_names: list[str],
    get_step_defaults: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Build an isolated root-and-step key schema from runtime defaults."""
    schema = copy.deepcopy(dict(root_defaults))
    schema["steps"] = {
        step_name: copy.deepcopy(get_step_defaults(step_name))
        for step_name in step_names
    }
    return schema


def warn_unknown_nested_config_keys(
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
    logger: logging.Logger,
    *,
    strict_paths: Collection[tuple[str, ...]],
) -> None:
    """Warn for unknown keys in explicitly closed behavior mappings.

    Top-level section and step-name diagnostics remain the responsibility of each
    agent validator so their established messages stay compatible. Only mappings
    named in ``strict_paths`` are closed; model-provider, renderer, and other
    extension mappings therefore remain open unless a caller explicitly closes
    them.
    """
    strict_path_set = frozenset(strict_paths)
    for section_name, section_value in config.items():
        if section_name == "steps":
            _warn_step_keys(
                section_value,
                schema.get("steps"),
                logger,
                strict_paths=strict_path_set,
            )
            continue
        if not isinstance(section_name, str):
            continue
        section_schema = schema.get(section_name)
        section_path = (section_name,)
        if (
            _contains_strict_path(section_path, strict_path_set)
            and isinstance(section_value, Mapping)
            and isinstance(section_schema, Mapping)
        ):
            _warn_mapping_keys(
                section_value,
                section_schema,
                path=section_path,
                logger=logger,
                strict_paths=strict_path_set,
            )


def _warn_step_keys(
    steps: Any,
    step_schemas: Any,
    logger: logging.Logger,
    *,
    strict_paths: frozenset[tuple[str, ...]],
) -> None:
    if not isinstance(steps, Mapping) or not isinstance(step_schemas, Mapping):
        return
    for step_name, step_value in steps.items():
        if not isinstance(step_name, str):
            continue
        step_schema = step_schemas.get(step_name)
        step_path = ("steps", step_name)
        if (
            _contains_strict_path(step_path, strict_paths)
            and isinstance(step_value, Mapping)
            and isinstance(step_schema, Mapping)
        ):
            _warn_mapping_keys(
                step_value,
                step_schema,
                path=step_path,
                logger=logger,
                strict_paths=strict_paths,
            )


def _warn_mapping_keys(
    value: Mapping[Any, Any],
    schema: Mapping[str, Any],
    *,
    path: tuple[str, ...],
    logger: logging.Logger,
    strict_paths: frozenset[tuple[str, ...]],
) -> None:
    supported_keys = tuple(key for key in schema if isinstance(key, str))
    is_strict = path in strict_paths

    for key, nested_value in value.items():
        if not isinstance(key, str):
            if is_strict:
                logger.warning(
                    "Unknown configuration key '%s.<non-string-key>'; the key "
                    "will be ignored and defaults may apply",
                    ".".join(path),
                )
            continue

        if key not in schema:
            if not is_strict or _is_runtime_wiring_key(path, key):
                continue
            suggestion = _unambiguous_suggestion(key, supported_keys)
            full_path = ".".join((*path, key))
            safe_full_path = redact_sensitive_path(full_path)
            if suggestion is None:
                logger.warning(
                    "Unknown configuration key '%s'; the key will be ignored "
                    "and defaults may apply",
                    safe_full_path,
                )
            else:
                logger.warning(
                    "Unknown configuration key '%s'; did you mean '%s'? The key "
                    "will be ignored and defaults may apply",
                    safe_full_path,
                    ".".join((*path, suggestion)),
                )
            continue

        nested_schema = schema[key]
        nested_path = (*path, key)
        if (
            _contains_strict_path(nested_path, strict_paths)
            and isinstance(nested_value, Mapping)
            and isinstance(nested_schema, Mapping)
        ):
            _warn_mapping_keys(
                nested_value,
                nested_schema,
                path=nested_path,
                logger=logger,
                strict_paths=strict_paths,
            )


def _contains_strict_path(
    path: tuple[str, ...],
    strict_paths: frozenset[tuple[str, ...]],
) -> bool:
    """Return whether ``path`` is strict or contains a strict descendant."""
    return any(candidate[: len(path)] == path for candidate in strict_paths)


def _unambiguous_suggestion(
    key: str,
    supported_keys: tuple[str, ...],
) -> str | None:
    matches = difflib.get_close_matches(key, supported_keys, n=2, cutoff=0.75)
    if len(matches) == 1:
        return matches[0]
    if len(matches) != 2:
        return None
    best_ratio = difflib.SequenceMatcher(None, key, matches[0]).ratio()
    runner_up_ratio = difflib.SequenceMatcher(None, key, matches[1]).ratio()
    return matches[0] if best_ratio - runner_up_ratio >= 0.15 else None


def _is_runtime_wiring_key(path: tuple[str, ...], key: str) -> bool:
    """Return whether a step key is a supported artifact-wiring surface."""
    if len(path) != 2 or path[0] != "steps":
        return False
    return key.endswith(("_dir", "_dirs", "_path", "_paths")) or key in {
        "config",
        "dataset",
        "models",
        "source",
        "usd_path",
    }
