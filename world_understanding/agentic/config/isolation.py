# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identity-safe isolation for in-memory workflow configuration."""

from __future__ import annotations

import json
import math
from enum import Enum
from pathlib import PosixPath, PurePosixPath, PureWindowsPath, WindowsPath
from typing import Any

UNSUPPORTED_YAML_CONFIG_MESSAGE = "Unsupported YAML-equivalent configuration value"
_STDLIB_PATH_TYPES = frozenset({PosixPath, PurePosixPath, PureWindowsPath, WindowsPath})


def _yaml_config_sort_key(value: Any) -> tuple[int, str]:
    """Return a deterministic key for an already-normalized value."""
    type_rank = {
        type(None): 0,
        bool: 1,
        int: 2,
        float: 3,
        str: 4,
        list: 5,
        dict: 6,
    }
    return (
        type_rank[type(value)],
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _normalize_yaml_config_value(
    value: Any,
    *,
    active_container_ids: set[int],
) -> Any:
    """Normalize one trusted built-in config shape without rendering objects."""
    if isinstance(value, Enum):
        return _normalize_yaml_config_value(
            value.value,
            active_container_ids=active_container_ids,
        )

    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise TypeError from None
        return value
    if value_type in _STDLIB_PATH_TYPES:
        return str(value)
    if value_type not in {dict, list, tuple, set, frozenset}:
        raise TypeError from None

    container_id = id(value)
    if container_id in active_container_ids:
        raise TypeError from None
    active_container_ids.add(container_id)
    try:
        if value_type is dict:
            normalized: dict[str, Any] = {}
            for key, child in value.items():
                normalized_key = _normalize_yaml_config_value(
                    key,
                    active_container_ids=active_container_ids,
                )
                if type(normalized_key) not in {
                    type(None),
                    bool,
                    int,
                    float,
                    str,
                }:
                    raise TypeError from None
                string_key = str(normalized_key)
                if string_key in normalized:
                    raise TypeError from None
                normalized[string_key] = _normalize_yaml_config_value(
                    child,
                    active_container_ids=active_container_ids,
                )
            return normalized

        if value_type in {list, tuple}:
            return [
                _normalize_yaml_config_value(
                    child,
                    active_container_ids=active_container_ids,
                )
                for child in value
            ]

        normalized_items = [
            _normalize_yaml_config_value(
                child,
                active_container_ids=active_container_ids,
            )
            for child in value
        ]
        return sorted(normalized_items, key=_yaml_config_sort_key)
    finally:
        active_container_ids.remove(container_id)


def normalize_yaml_config_value(value: Any) -> Any:
    """Return a deterministic YAML-equivalent config or fail value-free.

    Runtime configurations may contain provisioned clients, locks, sessions,
    or other opaque objects. Calling ``str``/``repr`` (including from a sort
    key) at this boundary can disclose credentials owned by those objects.
    Accept only explicit scalar/path/enum values and built-in containers;
    reject every other leaf with one code-owned diagnostic.
    """
    normalized: Any = None
    failed = False
    try:
        normalized = _normalize_yaml_config_value(
            value,
            active_container_ids=set(),
        )
    except Exception:
        failed = True

    if failed:
        del normalized, value
        raise TypeError(UNSUPPORTED_YAML_CONFIG_MESSAGE) from None
    return normalized


def clone_config_containers(value: Any) -> Any:
    """Clone built-in containers while retaining opaque leaves by identity.

    ``copy.deepcopy`` is the wrong boundary for runtime configuration because a
    config may carry an already-provisioned client, lock, session, or other
    opaque object.  This helper isolates only mutable/config-shaped built-ins
    (``dict``, ``list``, ``set`` and their immutable tuple/frozenset peers).
    A per-call memo preserves aliases and recursive container graphs.  Every
    invocation owns a fresh memo, so concurrent loads cannot share clones.
    """
    memo: dict[int, Any] = {}

    def clone(item: Any) -> Any:
        item_type = type(item)
        if item_type not in {dict, list, tuple, set, frozenset}:
            return item

        item_id = id(item)
        cached = memo.get(item_id)
        if cached is not None:
            return cached

        if item_type is dict:
            cloned_dict: dict[Any, Any] = {}
            memo[item_id] = cloned_dict
            for key, child in item.items():
                cloned_dict[clone(key)] = clone(child)
            return cloned_dict

        if item_type is list:
            cloned_list: list[Any] = []
            memo[item_id] = cloned_list
            cloned_list.extend(clone(child) for child in item)
            return cloned_list

        if item_type is set:
            cloned_set: set[Any] = set()
            memo[item_id] = cloned_set
            cloned_set.update(clone(child) for child in item)
            return cloned_set

        # Tuples can participate in an indirect cycle (tuple -> list -> tuple).
        # Clone children first, then honor a tuple clone installed by that
        # recursive path, matching deepcopy's alias-preserving behavior without
        # ever copying opaque leaves.
        if item_type is tuple:
            cloned_items = [clone(child) for child in item]
            recursive_clone = memo.get(item_id)
            if recursive_clone is not None:
                return recursive_clone
            cloned_tuple = tuple(cloned_items)
            memo[item_id] = cloned_tuple
            return cloned_tuple

        cloned_frozenset = frozenset(clone(child) for child in item)
        memo[item_id] = cloned_frozenset
        return cloned_frozenset

    return clone(value)


__all__ = [
    "UNSUPPORTED_YAML_CONFIG_MESSAGE",
    "clone_config_containers",
    "normalize_yaml_config_value",
]
