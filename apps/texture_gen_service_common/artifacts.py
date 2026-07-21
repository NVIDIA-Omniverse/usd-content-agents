# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact URI helpers for texture generation services."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

_WINDOWS_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_windows_drive_path(value: str) -> bool:
    return bool(_WINDOWS_DRIVE_PATH_RE.match(value))


def local_file_uri(path: str | Path) -> str:
    """Return a normalized file URI for a local artifact."""
    return Path(path).expanduser().resolve().as_uri()


def local_path_from_file_uri(uri: str) -> Path | None:
    """Resolve a local file URI or bare local path.

    Returns ``None`` for non-local schemes. The returned path may not exist.
    """
    if _is_windows_drive_path(uri):
        return Path(uri).expanduser()

    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        return None
    raw_path = uri
    if parsed.scheme == "file":
        host = unquote(parsed.netloc)
        raw_path = unquote(parsed.path)
        if host in {"", "localhost"}:
            if re.match(r"^/[A-Za-z]:[\\/]", raw_path):
                raw_path = raw_path[1:]
        elif _is_windows_drive_path(host):
            raw_path = f"{host}{raw_path}"
        elif re.fullmatch(r"[A-Za-z]:", host):
            raw_path = f"{host}{raw_path}"
        else:
            raw_path = f"/{host}{raw_path}"
    path = Path(raw_path).expanduser()
    if parsed.scheme == "file" or path.exists():
        return path
    decoded = unquote(raw_path)
    if decoded != raw_path:
        decoded_path = Path(decoded).expanduser()
        if decoded_path.exists():
            return decoded_path
    return path


def require_visible_file(uri: str, *, label: str = "artifact") -> Path:
    """Resolve a visible local file artifact or raise a structured ValueError."""
    path = local_path_from_file_uri(uri)
    if path is None:
        raise ValueError(
            f"{label} is not a local file URI reachable by this process: {uri}"
        )
    if not path.is_file():
        raise ValueError(f"{label} is not visible by this process: {path}")
    return path.resolve()
