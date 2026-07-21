# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared confinement rules for service session identifiers and paths."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path, PureWindowsPath
from typing import TypeGuard

_SESSION_ID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def is_safe_session_id(session_id: object) -> TypeGuard[str]:
    """Return whether a value has the server-generated UUID session shape."""
    return isinstance(session_id, str) and bool(
        _SESSION_ID_PATTERN.fullmatch(session_id)
    )


def safe_listed_session_ids(session_ids: Iterable[object]) -> list[str]:
    """Project an untrusted backend listing to unique, valid session IDs.

    Backends can contain historical or externally injected prefixes.  Those
    names must never become URL identifiers or inputs to derived local paths.
    """
    return sorted({value for value in session_ids if is_safe_session_id(value)})


def confined_session_path(base_dir: str | Path, session_id: str) -> Path:
    """Return one lexical session child beneath the resolved storage root."""
    if not is_safe_session_id(session_id):
        raise ValueError("Invalid session identifier")
    try:
        base = Path(base_dir).resolve(strict=False)
        candidate = base / session_id
        if candidate.is_symlink():
            raise ValueError("Session roots cannot be symlinks")
        candidate.relative_to(base)
    except (OSError, RuntimeError, ValueError):
        raise ValueError("Session path escapes configured storage root") from None
    return candidate


def validated_storage_child_name(name: object) -> str:
    """Return one portable storage child name without touching the filesystem."""

    windows_name = PureWindowsPath(name) if isinstance(name, str) else None
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or (windows_name is not None and windows_name.drive)
    ):
        raise ValueError("Invalid storage child identifier")
    return name


def confined_storage_child_path(base_dir: str | Path, name: str) -> Path:
    """Confine one legacy storage child without granting traversal semantics.

    Service managers require UUID session identifiers. Storage adapters retain
    support for simple historical/test identifiers, but those identifiers are
    still a single portable path component and can never escape ``base_dir``.
    """
    name = validated_storage_child_name(name)
    try:
        base = Path(base_dir).resolve(strict=False)
        candidate = base / name
        if candidate.is_symlink():
            raise ValueError("Storage child roots cannot be symlinks")
        candidate.relative_to(base)
    except (OSError, RuntimeError, ValueError):
        raise ValueError("Storage child path escapes configured root") from None
    return candidate
