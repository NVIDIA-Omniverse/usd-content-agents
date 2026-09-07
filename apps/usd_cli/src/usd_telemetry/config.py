# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for usd-cli-tel.

Everything comes from USD_CLI_TEL_* environment variables — the wrapper
deliberately consumes zero argv so the wrapped usd-cli command line passes
through byte-for-byte, regardless of what flags usd-cli grows later.

Variables:
    USD_CLI_TEL_DISABLED       "1"/"true" → pure exec passthrough, no telemetry.
    USD_CLI_TEL_TARGET         Wrapped binary (default: "usd-cli").
    USD_CLI_TEL_BACKENDS       Comma list of backends (default: "file").
                               Known: file, otlp, none.
    USD_CLI_TEL_FILE           JSONL path for the file backend
                               (default: $XDG_STATE_HOME/usd-cli-tel/telemetry.jsonl,
                               falling back to ~/.local/state/...).
    USD_CLI_TEL_FILE_MAX_BYTES Rotate telemetry.jsonl → telemetry.jsonl.1 past this
                               size (default: 52428800 = 50 MiB; 0 disables rotation).
    USD_CLI_TEL_OTLP_ENDPOINT  OTLP/HTTP traces URL (default:
                               http://localhost:4318/v1/traces).
    USD_CLI_TEL_OTLP_HEADERS   Extra headers, "k=v,k2=v2" (e.g. auth).
    USD_CLI_TEL_OTLP_TIMEOUT   POST timeout seconds (default: 3.0).
    USD_CLI_TEL_TRACE_ID       32-hex trace id to correlate a whole session's
                               invocations under one trace (default: random per call).
                               A valid W3C TRACEPARENT env var takes precedence.
    USD_CLI_TEL_ATTRS          Extra span attributes, "k=v,k2=v2" (e.g. a
                               workflow run id). Recorded verbatim on every span.
    USD_CLI_TEL_ACTION_LOCK    Optional filesystem lease used to serialize
                               higher-level agent shell actions. Consecutive
                               wrapper calls from the same parent process share
                               the lease until that parent exits.
    USD_CLI_TEL_ACTION_LOCK_TIMEOUT
                               Maximum seconds to wait for another live action
                               owner (default: 3600).
    USD_CLI_TEL_DEBUG          "1"/"true" → telemetry errors print to stderr
                               instead of being swallowed.

Standard (non-USD_CLI_TEL) env consumed for distributed tracing:
    TRACEPARENT / TRACESTATE   W3C trace context from the parent process. When
                               valid, the span joins that trace with the given
                               parent span id, and the wrapped usd-cli child is
                               exec'd with TRACEPARENT rewritten to name this
                               wrapper span as the parent (for future
                               daemon-emitted spans).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on"}


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def default_log_path() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(state_home) if state_home else Path.home() / ".local" / "state"
    return base / "usd-cli-tel" / "telemetry.jsonl"


@dataclass
class Config:
    disabled: bool = False
    target: str = "usd-cli"
    backends: list[str] = field(default_factory=lambda: ["file"])
    log_path: Path = field(default_factory=default_log_path)
    file_max_bytes: int = 50 * 1024 * 1024
    otlp_endpoint: str = "http://localhost:4318/v1/traces"
    otlp_headers: dict[str, str] = field(default_factory=dict)
    otlp_timeout: float = 3.0
    trace_id: str | None = None
    extra_attrs: dict[str, str] = field(default_factory=dict)
    action_lock_path: Path | None = None
    action_lock_timeout: float = 3600.0
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        env = os.environ
        backends = [
            b.strip().lower()
            for b in env.get("USD_CLI_TEL_BACKENDS", "file").split(",")
            if b.strip()
        ] or ["file"]

        headers: dict[str, str] = {}
        for pair in env.get("USD_CLI_TEL_OTLP_HEADERS", "").split(","):
            if "=" in pair:
                k, _, v = pair.partition("=")
                if k.strip():
                    headers[k.strip()] = v.strip()

        trace_id = env.get("USD_CLI_TEL_TRACE_ID", "").strip().lower() or None
        if trace_id is not None:
            ok = len(trace_id) == 32 and all(c in "0123456789abcdef" for c in trace_id)
            if not ok:
                trace_id = None  # malformed → ignore, never break the call

        extra_attrs: dict[str, str] = {}
        for pair in env.get("USD_CLI_TEL_ATTRS", "").split(","):
            if "=" in pair:
                k, _, v = pair.partition("=")
                if k.strip():
                    extra_attrs[k.strip()] = v.strip()

        def _num(name: str, cast, default):
            try:
                return cast(env[name])
            except (KeyError, ValueError):
                return default

        return cls(
            disabled=_flag("USD_CLI_TEL_DISABLED"),
            target=env.get("USD_CLI_TEL_TARGET", "usd-cli").strip() or "usd-cli",
            backends=backends,
            log_path=Path(env.get("USD_CLI_TEL_FILE", "").strip() or default_log_path()),
            file_max_bytes=_num("USD_CLI_TEL_FILE_MAX_BYTES", int, 50 * 1024 * 1024),
            otlp_endpoint=env.get(
                "USD_CLI_TEL_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"
            ).strip(),
            otlp_headers=headers,
            otlp_timeout=_num("USD_CLI_TEL_OTLP_TIMEOUT", float, 3.0),
            trace_id=trace_id,
            extra_attrs=extra_attrs,
            action_lock_path=(
                Path(action_lock)
                if (action_lock := env.get("USD_CLI_TEL_ACTION_LOCK", "").strip())
                else None
            ),
            action_lock_timeout=max(
                0.0,
                _num("USD_CLI_TEL_ACTION_LOCK_TIMEOUT", float, 3600.0),
            ),
            debug=_flag("USD_CLI_TEL_DEBUG"),
        )
