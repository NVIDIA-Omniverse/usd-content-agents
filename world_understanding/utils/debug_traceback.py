# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-local redacted failure diagnostics.

Public pipeline failure surfaces (logs, events, telemetry, checkpoints) stay
value-free by design: provider and validator exceptions routinely embed
prompts, URLs, and credentials in their text. That redaction intent is
legitimate, but it previously left NOTHING on disk that recorded the real
cause of a step failure.

This module adds the missing session-local debug artifact:

- :func:`scrub_secret_text` removes recognizable secret material (provider
  token shapes, ``Bearer``/``Basic`` credentials, URL userinfo, cookies,
  key-style assignments, signed-URL parameters) from free-form text and
  finishes with a conservative pass that redacts any remaining
  entropy-looking token. It is best-effort by nature and must never be the
  only guarantee.
- :func:`format_scrubbed_exception` renders a *structured* entry that is
  value-free by construction — exception categories (code-defined type
  names) and frame locations/source lines read from files on disk, never
  exception values, messages, or frame locals. The exception message itself
  is deliberately NOT persisted: regex matching cannot prove arbitrary text
  credential-safe (``login failed for hunter2`` passes every pattern), so
  free text is kept off this surface entirely.
- :func:`append_step_failure_debug_entry` appends one timestamped, bounded
  entry per step failure to
  ``<working_dir>/.pipeline_temp/debug/step_failures.log``. The
  ``.pipeline_temp`` namespace is the canonical session-sync exclusion
  (:func:`world_understanding.utils.artifacts.is_pipeline_temp_path`), so the
  artifact is never listed, uploaded, or downloaded as a session file, and it
  is removed at the next pipeline startup by the legacy temp cleanup —
  bounded local retention for credential-adjacent diagnostics.

:func:`scrub_secret_text` is applied to the structured rendering as defence
in depth only (source lines on disk can embed literal token shapes); it is
never the mechanism that makes a surface credential-safe.
"""

from __future__ import annotations

import logging
import os
import re
import stat
import traceback

try:  # pragma: no cover - POSIX-only module
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore[assignment]
from datetime import UTC, datetime
from pathlib import Path

from world_understanding.utils.artifacts import (
    open_confined_directory,
)
from world_understanding.utils.credentials import (
    _is_sensitive_url_parameter,
)

logger = logging.getLogger(__name__)

#: Lives under ``.pipeline_temp`` so the canonical session-sync exclusion
#: keeps the artifact out of session listing/upload (see module docstring).
STEP_FAILURE_DEBUG_RELATIVE_PATH = (
    Path(".pipeline_temp") / "debug" / "step_failures.log"
)

_SECRET_REPLACEMENT = "[REDACTED]"

#: Bounds. The structured entry is small by construction; the entry and the
#: log itself are explicitly capped so deep recursion or repeated retries
#: cannot exhaust session storage.
_MAX_TRACEBACK_FRAMES = 50
_MAX_ENTRY_CHARS = 65536
_MAX_LOG_BYTES = 1_000_000
_MAX_EXCEPTION_CHAIN = 8

# Recognizable secret shapes to scrub from debug text. Order matters: the
# specific token formats run before the generic key=value assignment
# patterns, and the conservative entropy pass runs last (see
# ``scrub_secret_text``).
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # NVIDIA API keys (nvapi-...).
    (re.compile(r"nvapi-[A-Za-z0-9_-]+"), f"nvapi-{_SECRET_REPLACEMENT}"),
    # Bare provider token shapes (sk-..., sk-proj-..., pk-..., rk-...).
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{8,}"), _SECRET_REPLACEMENT),
    # GitHub / GitLab token shapes (ghp_..., github_pat_..., glpat-...).
    (
        re.compile(r"\b(?:gh[pousr]_|github_pat_|glpat[-_])[A-Za-z0-9_-]{8,}"),
        _SECRET_REPLACEMENT,
    ),
    # Authorization credentials: "Bearer <token>" and "Basic <base64>". The
    # value is captured with the scheme word so a Basic credential loses its
    # base64 payload, not just the word "Basic".
    (
        re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{4,}"),
        rf"\1 {_SECRET_REPLACEMENT}",
    ),
    # URL userinfo (https://user:secret@host/...): the whole userinfo is
    # authentication material.
    (re.compile(r"://[^/\s@\"'<>]+@"), f"://{_SECRET_REPLACEMENT}@"),
    # Cookie material: the entire header value is bearer material.
    (
        re.compile(r"(?i)\b(set-cookie|cookie)(\s*[:=]\s*)[^\r\n]+"),
        rf"\1\2{_SECRET_REPLACEMENT}",
    ),
    # Key-style assignments: api_key=..., api-key: "...", token=..., plus
    # identifier-prefixed environment-variable forms such as
    # OPENAI_API_KEY=... or GITHUB_TOKEN: ... A leading \b alone cannot match
    # those because the preceding "_" is a word character, so an optional
    # identifier prefix is matched explicitly. The optional quote before the
    # separator also covers serialized-JSON fields ({"api_key": "secret"}),
    # whose closing key quote would otherwise break the match. Key-name cores
    # tolerate a "-" or "_" between EVERY pair of characters (pass-word,
    # se_cret, to-ken cannot dodge the match), and a trailing identifier
    # segment (password-hash=...) is folded into the key name as well.
    (
        re.compile(
            r"(?i)([A-Za-z0-9_-]*(?:"
            + "|".join(
                "[-_]?".join(keyword_name)
                for keyword_name in (
                    "apikey",
                    "accesstoken",
                    "authtoken",
                    "authorization",
                    "passphrase",
                    "password",
                    "passwd",
                    "secret",
                    "token",
                    "key",
                )
            )
            + r")[A-Za-z0-9_-]*)"
            r"([\"']?\s*[=:]\s*)"
            r"(\"[^\"]*\"|'[^']*'|[^\s\"'&,;]+)"
        ),
        rf"\1\2{_SECRET_REPLACEMENT}",
    ),
)

# URL query/fragment parameters (?X-Amz-Signature=..., &sig=..., ?key=...).
# Which parameter names carry bearer material is decided by the repository's
# credential-aware URL detection (``_is_sensitive_url_parameter``), so this
# scrub stays consistent with the durable-artifact credential guards.
_URL_QUERY_PARAMETER_RE = re.compile(r"([?&;#])([A-Za-z0-9_.%~+-]+)(=)([^\s&;#\"'<>]+)")

# Conservative last-pass redaction: any remaining long mixed alphanumeric run
# is treated as potential secret material. Punctuation such as ``-``, ``.``
# and ``/`` intentionally splits runs so ordinary URLs, paths, timestamps,
# and parameter names survive; over-redaction of genuine identifiers is an
# accepted cost for this session-local diagnostic.
_ENTROPY_TOKEN_RE = re.compile(r"[A-Za-z0-9+_]{20,}")


def _redact_url_query_parameters(text: str) -> str:
    """Redact values of credential-bearing URL query/fragment parameters."""

    def _redact(match: re.Match[str]) -> str:
        if _is_sensitive_url_parameter(match.group(2)):
            return f"{match.group(1)}{match.group(2)}={_SECRET_REPLACEMENT}"
        return match.group(0)

    return _URL_QUERY_PARAMETER_RE.sub(_redact, text)


def _redact_entropy_like_tokens(text: str) -> str:
    """Redact remaining long mixed-alphanumeric tokens (conservative pass)."""

    def _redact(match: re.Match[str]) -> str:
        token = match.group(0)
        if any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token):
            return _SECRET_REPLACEMENT
        return token

    return _ENTROPY_TOKEN_RE.sub(_redact, text)


def scrub_secret_text(text: str) -> str:
    """Return ``text`` with recognizable and entropy-like secrets replaced.

    This is a lightweight scrub for session-local debug artifacts. It
    intentionally errs on the side of over-redaction: known token shapes,
    ``Bearer``/``Basic`` credentials, URL userinfo, cookie headers, key-style
    assignments, and signed-URL parameters are redacted first, then a
    conservative pass redacts any remaining long mixed-alphanumeric token.
    It remains best-effort: callers must not treat the result as
    credential-safe by itself (see :func:`format_scrubbed_exception`).
    """
    scrubbed = _redact_url_query_parameters(text)
    for pattern, replacement in _SECRET_PATTERNS:
        scrubbed = pattern.sub(replacement, scrubbed)
    return _redact_entropy_like_tokens(scrubbed)


def _safe_exception_categories(error: BaseException) -> list[str]:
    """Return bounded code-defined type names along the exception chain."""
    categories: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while (
        current is not None
        and id(current) not in seen
        and len(categories) < _MAX_EXCEPTION_CHAIN
    ):
        seen.add(id(current))
        name = type(current).__name__
        if len(name) > 128 or not name.isidentifier():
            name = "Exception"
        categories.append(name)
        current = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )
    return categories


def format_scrubbed_exception(error: BaseException) -> str:
    """Render a bounded, structured traceback for ``error``.

    The output is value-free **by construction**, not by pattern matching:
    exception *categories* are code-defined type names, and frame locations
    plus source lines come from :func:`traceback.extract_tb`, which reads
    source files on disk — never exception values and never frame locals.
    The exception message is deliberately NOT included: arbitrary exception
    text cannot be proven credential-safe by a scrubber (``login failed for
    hunter2`` or ``Authorization: Bearer abc`` pass every pattern), so free
    text stays off this surface entirely. The rendering is still passed
    through :func:`scrub_secret_text` as defence in depth, because source
    lines on disk can embed literal token shapes.

    Traceback-less errors (e.g. a never-raised exception instance) still
    yield a structured category entry instead of raising.
    """
    error_traceback = getattr(error, "__traceback__", None)
    if error_traceback is not None:
        frames = traceback.extract_tb(error_traceback)
    else:
        frames = []
    omitted = max(0, len(frames) - _MAX_TRACEBACK_FRAMES)
    if omitted:
        frames = frames[-_MAX_TRACEBACK_FRAMES:]
    lines: list[str] = ["Traceback (most recent call last):"]
    if omitted:
        lines.append(f"  [{omitted} earlier frame(s) omitted]")
    if not frames:
        lines.append("  [no traceback frames available]")
    for frame in frames:
        lines.append(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}')
        if frame.line:
            lines.append(f"    {frame.line}")
    lines.append(
        "exception categories (outermost first): "
        + " <- ".join(_safe_exception_categories(error))
    )
    lines.append(
        "exception message withheld: free text is not credential-safe "
        "(value-free structured fields only)"
    )
    return scrub_secret_text("\n".join(lines)) + "\n"


def append_step_failure_debug_entry(
    working_dir: str | os.PathLike[str] | None,
    step_name: str,
    error: BaseException,
) -> Path | None:
    """Append one bounded, scrubbed step-failure traceback to the debug log.

    Writes to ``<working_dir>/.pipeline_temp/debug/step_failures.log`` in
    append mode with one timestamped entry per failure. The ``.pipeline_temp``
    location keeps the artifact inside the canonical session-sync exclusion:
    it is never listed or uploaded with session files, and the next pipeline
    startup removes it, bounding local retention. Best-effort: any failure to
    persist the debug artifact is swallowed (with a value-free warning) so
    diagnostics can never change pipeline control flow or mask the original
    step failure. Appends stop once the log reaches ``_MAX_LOG_BYTES``.

    Returns:
        The debug log path on success, or ``None`` when nothing was written.
    """
    if working_dir is None:
        return None
    try:
        debug_path = Path(working_dir) / STEP_FAILURE_DEBUG_RELATIVE_PATH
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        entry = (
            f"{'=' * 78}\n"
            f"[{timestamp}] step failure: {scrub_secret_text(step_name)}\n"
            f"{'-' * 78}\n"
            f"{format_scrubbed_exception(error)}"
            "\n"
        )
        if len(entry) > _MAX_ENTRY_CHARS:
            entry = entry[:_MAX_ENTRY_CHARS] + "\n[entry truncated]\n"
        entry_bytes = entry.encode("utf-8")
        # Keep the artifact private to the session owner and confine every
        # path component: a symlinked working directory, ``.pipeline_temp``
        # or ``debug`` directory, or log file must never redirect the append
        # outside the session. ``open_confined_directory`` holds and
        # no-follows each component; the leaf open below no-follows and must
        # be a regular file. O_NONBLOCK keeps a pre-existing FIFO at the log
        # path from blocking the open before fstat can reject it (a
        # reader-less FIFO fails the write-only open immediately with ENXIO);
        # the flag is cleared once the leaf is validated as a regular file.
        with open_confined_directory(debug_path.parent, create=True) as debug_directory:
            descriptor = os.open(
                debug_path.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_APPEND
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                0o600,
                dir_fd=debug_directory,
            )
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise OSError("Debug log must be a regular file")
                if fcntl is not None:
                    flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
                    fcntl.fcntl(
                        descriptor, fcntl.F_SETFL, flags & ~getattr(os, "O_NONBLOCK", 0)
                    )
                # The creation mode applies only when the file is first
                # created. Enforce owner-only permissions on a pre-existing
                # log BEFORE any bytes land on this descriptor, so a
                # permissive 0o644 log is tightened before new content is
                # readable and an fchmod failure writes nothing at all.
                if hasattr(os, "fchmod"):
                    os.fchmod(descriptor, 0o600)
                if metadata.st_size + len(entry_bytes) > _MAX_LOG_BYTES:
                    logger.warning(
                        "Step-failure debug log reached its size cap; skipping append"
                    )
                    return None
                view = memoryview(entry_bytes)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:  # pragma: no cover - regular-file invariant
                        raise OSError("Could not append debug entry bytes")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return debug_path
    except Exception:  # pragma: no cover - defensive diagnostic boundary
        logger.warning(
            "Unable to persist step-failure debug artifact for step diagnostics"
        )
        return None
