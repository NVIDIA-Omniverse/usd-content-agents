# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a Response to stdout: plain text by default, JSON envelope with --json."""

from __future__ import annotations

import json
import sys

from usd_core.models import Response
from usd_cli.state import G

# Exit codes (cli-design.md §4)
EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2
EXIT_UNREACHABLE = 3
# Daemon alive but busy (error_type "busy") — never restart it. The timed-out command
# may still complete on the daemon (outcome UNKNOWN): verify state (snapshot/history)
# before retrying a MUTATING command, or it may execute twice.
EXIT_BUSY = 4
# Uniform ceiling for plain-text bodies: enough for real trees/tables, far below
# the harness truncators that made oversized output unusable AND expensive.
_TEXT_CAP_LINES = 400


def _error_exit_code(resp: Response) -> int:
    error_type = resp.summary.get("error_type")
    if error_type == "busy":
        return EXIT_BUSY
    if error_type in {"startup", "authentication", "transport"}:
        return EXIT_UNREACHABLE
    return EXIT_RUNTIME


def emit(resp: Response) -> int:
    if G.json:
        d = resp.to_dict()
        # Agents parse this; humans have jq. Compact separators + dropped empty
        # collections cut --json output ~40% (round-7 audit: 295 kB of pure
        # indentation + hundreds of empty issues/artifacts arrays paid per run).
        # ok/command/schema_version stay — cheap, and parsers rely on them.
        for k in ("summary", "data", "artifacts", "issues"):
            if not d.get(k):
                d.pop(k, None)
        blob = json.dumps(d, separators=(",", ":"))
        if len(blob) > 2_000_000:
            # round 8: an uncapped `--json find --type Xform` emitted 28 MB on
            # ONE line; downstream sed/jq turned it into multi-million-token
            # dumps. Warn on stderr — stdout stays a clean JSON contract.
            print(f"warning: {len(blob) / 1e6:.1f} MB of JSON on one line — "
                  "narrow the query (--type/--under/--count) or expect "
                  "line-based tools (sed/grep) to choke on it", file=sys.stderr)
        print(blob)
        if resp.ok:
            return EXIT_OK
        return _error_exit_code(resp)

    # plain text
    is_stub = bool(resp.summary.get("stub"))
    if not G.quiet:
        if is_stub:
            body = resp.data.get("would_send", {})
            print(f"» would run: {body.get('command')} {body.get('payload', {})}")
        elif resp.summary:
            parts = " | ".join(f"{k}: {v}" for k, v in resp.summary.items())
            if parts:
                print(parts)

    if not is_stub:
        # -q + a refs list = pipe mode: bare refs, one per line (`for r in $(usd-cli -q
        # find …)`), regardless of how rich the display text is
        if G.quiet and resp.data.get("refs"):
            print("\n".join(resp.data["refs"]))
        else:
            # text body (snapshot tree, resolve listing, describe, …) — capped:
            # usd-cli must never emit unbounded listings (round-7: single outputs of
            # 200k tokens were generated, then thrown away by harness truncation)
            text = resp.data.get("text")
            if text:
                lines = text.split("\n")
                if len(lines) > _TEXT_CAP_LINES:
                    print("\n".join(lines[:_TEXT_CAP_LINES]))
                    # round 8: the old "or use --json" hint steered an agent
                    # straight into a 140k-token uncapped JSON dump — narrowing
                    # is the advice; --json comes with a size warning.
                    print(f"… (+{len(lines) - _TEXT_CAP_LINES} more lines of "
                          f"{len(lines)} — narrow the query (--type/--under/"
                          f"--depth). --json has NO cap: expect "
                          f"~{len(lines)}+ lines)")
                else:
                    print(text)
        for art in resp.artifacts:
            label = f" ({art.label})" if art.label else ""
            print(f"{art.kind}: {art.path}{label}")

        # segmentation legend: ref → color, so the seg image is machine-readable
        legend = resp.data.get("segmentation_legend")
        if legend:
            shown = list(legend.items())[:12]
            for ref, rgb in shown:
                print(f"  ■ {ref}  rgb({rgb[0]},{rgb[1]},{rgb[2]})")
            if len(legend) > len(shown):
                lf = resp.data.get("segmentation_legend_file")
                where = f"full legend: {lf}" if lf else "--json for the full legend"
                print(f"  … {len(legend) - len(shown)} more ({where})")

    for issue in resp.issues:  # always shown — incl. the stub "where's the daemon" hint
        stream = sys.stderr if issue.severity in ("warn", "error") else sys.stdout
        prefix = {"info": "", "warn": "warning: ", "error": "error: "}[issue.severity]
        print(f"{prefix}{issue.message}", file=stream)

    if not resp.ok:
        return _error_exit_code(resp)
    return EXIT_OK
