# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the OTel-shaped invocation record.

One JSON-serializable dict per usd-cli invocation, using OpenTelemetry span
field names and semantics (trace_id/span_id, unix-nano timestamps, status,
flat attributes map). The file backend writes it as-is (one JSONL line);
the OTLP backend re-serializes the same dict into an OTLP/HTTP payload.

Everything here is heuristic-over-argv by design: the wrapper never imports
usd_cli, so a new usd-cli flag can at worst misattribute `usd.command` —
`process.command_args` always carries the full ground truth.
"""

from __future__ import annotations

import os
import secrets
import socket
import sys

from . import SCHEMA_VERSION, __version__

# Current usd-cli global options that take a separate value argument. Used only
# to locate the subcommand token; unknown future flags degrade gracefully (the
# full argv is recorded regardless).
_GLOBAL_VALUE_FLAGS = {"--server", "--session", "--timeout"}

_SCENE_SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")

# Values following these flags are redacted in process.command_args.
_SECRET_FLAG_MARKERS = ("token", "key", "secret", "password", "auth")


def new_trace_id() -> str:
    return secrets.token_hex(16)


def new_span_id() -> str:
    return secrets.token_hex(8)


def _is_hex(value: str) -> bool:
    return bool(value) and all(c in "0123456789abcdef" for c in value)


def parse_traceparent(value: str | None) -> tuple[str, str] | None:
    """Parse a W3C `traceparent` value → (trace_id, parent_span_id) or None.

    Malformed values are ignored (never break the call): version must be two
    hex chars and not "ff", trace id 32 non-zero hex, parent span id 16
    non-zero hex.
    """

    if not value:
        return None
    parts = value.strip().lower().split("-")
    if len(parts) < 4:
        return None
    version, trace_id, span_id = parts[0], parts[1], parts[2]
    if len(version) != 2 or not _is_hex(version) or version == "ff":
        return None
    if len(trace_id) != 32 or not _is_hex(trace_id) or trace_id == "0" * 32:
        return None
    if len(span_id) != 16 or not _is_hex(span_id) or span_id == "0" * 16:
        return None
    return trace_id, span_id


def format_traceparent(trace_id: str, span_id: str) -> str:
    """Render a W3C `traceparent` naming this span as the parent (sampled)."""

    return f"00-{trace_id}-{span_id}-01"


# Well-known parent-agent env vars → span attributes. Neither Claude Code nor
# Codex injects W3C trace context into tool subprocesses today; these session
# keys are the correlation handle they DO expose. Allowlist only — arbitrary
# caller attributes go through USD_CLI_TEL_ATTRS instead.
_PARENT_AGENT_ENV_ATTRS = {
    "CLAUDE_CODE_SESSION_ID": "parent.claude_code.session_id",
}


def parent_agent_attributes(env: dict[str, str] | None = None) -> dict[str, str]:
    source = os.environ if env is None else env
    return {
        attr: source[var]
        for var, attr in _PARENT_AGENT_ENV_ATTRS.items()
        if source.get(var)
    }


def _is_secret_flag(flag: str) -> bool:
    name = flag.lstrip("-").lower()
    return any(marker in name for marker in _SECRET_FLAG_MARKERS)


def redact_args(argv: list[str]) -> list[str]:
    out: list[str] = []
    redact_next = False
    for tok in argv:
        if redact_next:
            out.append("<redacted>")
            redact_next = False
            continue
        if tok.startswith("-"):
            flag, eq, value = tok.partition("=")
            if _is_secret_flag(flag):
                if eq:
                    out.append(f"{flag}=<redacted>")
                    continue
                redact_next = True
            out.append(tok)
            continue
        out.append(tok)
    return out


def parse_command(argv: list[str]) -> tuple[str | None, dict[str, str]]:
    """First bare token = subcommand; also lift --session/--server values."""
    lifted: dict[str, str] = {}
    command: str | None = None
    skip_value_of: str | None = None
    for tok in argv:
        if skip_value_of is not None:
            lifted[skip_value_of] = tok
            skip_value_of = None
            continue
        if tok.startswith("-"):
            flag, eq, value = tok.partition("=")
            if flag in _GLOBAL_VALUE_FLAGS:
                if eq:
                    lifted[flag] = value
                else:
                    skip_value_of = flag
            continue
        if command is None:
            command = tok
            continue
    return command, lifted


def sniff_scenes(argv: list[str]) -> list[dict[str, object]]:
    """Tokens that look like USD files; stat the ones that exist on disk."""
    scenes: list[dict[str, object]] = []
    seen: set[str] = set()
    for tok in argv:
        # `--stage=foo.usd` and bare `foo.usd` both count.
        candidate = tok.partition("=")[2] if tok.startswith("-") and "=" in tok else tok
        if not candidate.lower().endswith(_SCENE_SUFFIXES) or candidate in seen:
            continue
        seen.add(candidate)
        entry: dict[str, object] = {"path": candidate}
        try:
            st = os.stat(candidate)
            entry["size_bytes"] = st.st_size
            entry["mtime_unix"] = int(st.st_mtime)
        except OSError:
            entry["exists"] = False
        scenes.append(entry)
    return scenes


def build_record(
    *,
    argv: list[str],
    executable: str | None,
    exit_code: int,
    start_unix_nano: int,
    end_unix_nano: int,
    stderr_tail: str,
    trace_id: str | None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    trace_state: str | None = None,
    extra_attributes: dict[str, str] | None = None,
) -> dict:
    command, lifted = parse_command(argv)
    scenes = sniff_scenes(argv)

    attributes: dict[str, object] = {
        "process.command_args": redact_args(argv),
        "process.exit_code": exit_code,
        "process.pid": os.getpid(),
        "usd.command": command or "<none>",
        "host.name": socket.gethostname(),
        "telemetry.sdk.name": "usd-cli-tel",
        "telemetry.sdk.version": __version__,
        "telemetry.schema_version": SCHEMA_VERSION,
        "os.type": sys.platform,
    }
    if executable:
        attributes["process.executable.path"] = executable
    if "--session" in lifted:
        attributes["usd.session"] = lifted["--session"]
    if "--server" in lifted:
        attributes["usd.server"] = lifted["--server"]
    if scenes:
        attributes["usd.scene.path"] = scenes[0]["path"]
        if "size_bytes" in scenes[0]:
            attributes["usd.scene.size_bytes"] = scenes[0]["size_bytes"]
        if len(scenes) > 1:
            attributes["usd.scene.paths"] = [s["path"] for s in scenes]

    attributes.update(parent_agent_attributes())
    if extra_attributes:
        attributes.update(extra_attributes)

    ok = exit_code == 0
    status: dict[str, object] = {"code": "OK" if ok else "ERROR"}
    if not ok:
        if stderr_tail:
            status["message"] = stderr_tail
        attributes["error.type"] = f"exit_code:{exit_code}"

    record = {
        "trace_id": trace_id or new_trace_id(),
        "span_id": span_id or new_span_id(),
        "name": f"usd-cli.{command}" if command else "usd-cli",
        "kind": "SPAN_KIND_CLIENT",
        "start_time_unix_nano": start_unix_nano,
        "end_time_unix_nano": end_unix_nano,
        "duration_ms": round((end_unix_nano - start_unix_nano) / 1e6, 3),
        "status": status,
        "attributes": attributes,
    }
    if parent_span_id:
        record["parent_span_id"] = parent_span_id
    if trace_state:
        record["trace_state"] = trace_state
    return record
