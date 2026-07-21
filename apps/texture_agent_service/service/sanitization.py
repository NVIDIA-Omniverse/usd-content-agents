# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public-surface response sanitization.

Strips internals that must not leak to the public API:

- NVCF function deployment URLs (``https://<id>.invocation.api.nvcf.nvidia.com/...``)
  embedded in error messages by ``str(httpx.HTTPStatusError)``.
- Absolute filesystem paths under the configured session storage root
  (``/var/texture-agent/sessions/<sid>/...`` in the docker image).

Sanitizers operate on the JSON-serializable payloads built by the routers
and on the strings persisted into ``session.json`` / event log / SSE
events; the underlying exceptions and container logs keep their original
text so operators can still diagnose incidents.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

# Matches an NVCF function-invocation URL, with or without a path tail.
# The function id is the leftmost label of the host (e.g. ``abc12345``); we
# replace the entire URL so the id never appears anywhere in the response.
_NVCF_URL_RE = re.compile(
    r"https?://[a-zA-Z0-9_-]+\.invocation\.api\.nvcf\.nvidia\.com[^\s'\"]*"
)

# Generic NVCF status / API URLs (e.g. ``https://api.nvcf.nvidia.com/...``).
_NVCF_API_URL_RE = re.compile(r"https?://api\.nvcf\.nvidia\.com[^\s'\"]*")

_NVCF_REDACTION = "<nvcf-endpoint>"
_PATH_REDACTION = "<session>"


def _path_redactor(storage_root: str | None) -> re.Pattern[str] | None:
    """Build a regex that matches absolute paths under the session root.

    Both the configured ``session_storage_path`` and the fixed
    ``/var/texture-agent/sessions`` Docker default are stripped, since
    the docker image bakes the latter even when local development uses
    a different root.
    """
    roots: list[str] = []
    if storage_root:
        roots.append(storage_root.rstrip("/"))
    if "/var/texture-agent/sessions" not in roots:
        roots.append("/var/texture-agent/sessions")
    if not roots:  # pragma: no cover - docker default root is always appended
        return None
    pattern = "|".join(re.escape(r) for r in roots)
    # Match the root plus any non-whitespace tail (path segments, filenames).
    return re.compile(rf"(?:{pattern})(?:/[^\s'\"]*)?")


def sanitize_message(message: str, storage_root: str | None = None) -> str:
    """Redact NVCF URLs and absolute session paths from a free-form string.

    Safe to call on any user-visible diagnostic string (per-unit error
    ``message``, top-level ``error``, SSE event ``message``).
    """
    if not message:
        return message
    cleaned = _NVCF_URL_RE.sub(_NVCF_REDACTION, message)
    cleaned = _NVCF_API_URL_RE.sub(_NVCF_REDACTION, cleaned)
    path_re = _path_redactor(storage_root)
    if path_re is not None:
        cleaned = path_re.sub(_PATH_REDACTION, cleaned)
    return cleaned


def _sanitize_value(value: Any, storage_root: str | None) -> Any:
    if isinstance(value, str):
        return sanitize_message(value, storage_root)
    if isinstance(value, dict):
        return {k: _sanitize_value(v, storage_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(v, storage_root) for v in value]
    return value


def sanitize_payload(payload: Any, storage_root: str | None = None) -> Any:
    """Recursively sanitize any JSON-serializable public response payload."""
    return _sanitize_value(deepcopy(payload), storage_root)


def sanitize_step_stats(
    stats: dict[str, Any] | None,
    storage_root: str | None = None,
) -> dict[str, Any] | None:
    """Recursively sanitize a ``failed_step_stats`` payload.

    The structure is shallow but heterogeneous: ``errors`` is sometimes a
    list (per-step extract) and sometimes a dict-of-lists (final summary
    in ``_extract_final_stats``). Recursing handles both without coupling
    to either shape.
    """
    if stats is None:
        return None
    cleaned: dict[str, Any] = _sanitize_value(deepcopy(stats), storage_root)
    return cleaned
