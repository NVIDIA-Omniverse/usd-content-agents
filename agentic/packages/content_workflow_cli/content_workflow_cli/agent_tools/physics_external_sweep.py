# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request one budgeted external-runtime (BYOR) tuning sweep from the broker.

This is the agent-facing client of the external tuning sweep broker. It does
NOT import engine code: the broker reserves budget atomically before running
the qualification-gated, judge-free ``run_external_tune`` sweep, and this
client just submits the request, polls until the sweep reaches a terminal
state, and prints the sweep record (evidence packet location, rendered frame
paths and digests) as JSON on stdout.

External sweeps are sequential: a second ``run`` while one sweep is still
running is refused with the budget exit code.

Exit codes:
  0  sweep succeeded (evidence packet written to the iteration directory)
  2  bad input / broker rejected the request
  3  budget, sequencing, or phase deadline refused the reservation
  4  sweep cancelled or per-sweep deadline exceeded
  5  sweep failed inside the engine or external runtime
  6  broker unreachable
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Any

import requests

from content_workflow_cli.external_tuning_broker import (
    DEFAULT_EXTERNAL_SWEEP_DEADLINE_SECONDS,
    TERMINAL_SWEEP_STATUSES,
)

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_BUDGET_REFUSED = 3
EXIT_DEADLINE = 4
EXIT_SWEEP_FAILED = 5
EXIT_BROKER_UNREACHABLE = 6
# Transient sequencing refusal (HTTP 503): the previous sweep's runtime is
# still terminating. Wait and retry — distinct from terminal budget
# exhaustion (exit 3) so deadline recovery stays reachable.
EXIT_SWEEP_IN_PROGRESS = 7

DEFAULT_POLL_SECONDS = 15.0


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a positive number, got {value!r}"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive finite number, got {value!r}"
        )
    return parsed


def _emit(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


def _json_body(response: requests.Response) -> dict[str, Any] | None:
    """Return the response's JSON object body, or None when it is not one.

    The broker always answers with a JSON object; anything else (an HTML
    proxy error page, a truncated body) means we are not talking to a
    healthy broker and must map to a documented exit code instead of
    raising an unhandled JSONDecodeError out of main().
    """

    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_active_search(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.active_search and args.active_search_file:
        raise argparse.ArgumentTypeError(
            "pass either --active-search or --active-search-file, not both"
        )
    raw: str | None = None
    if args.active_search:
        raw = args.active_search
    elif args.active_search_file:
        with open(args.active_search_file, encoding="utf-8") as handle:
            raw = handle.read()
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"active_search is not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("active_search must be a JSON object")
    return parsed


def _list_sweeps(args: argparse.Namespace) -> int:
    """Recovery surface: list every reserved sweep and its terminal state.

    A client whose blocking ``run`` call was killed (timeout, crash) must
    still cite the sweep it reserved; this recovers the sweep_id and record
    without reserving new budget.
    """

    try:
        response = requests.get(
            f"{args.broker_url.rstrip('/')}/sweeps", timeout=args.request_timeout
        )
    except requests.RequestException as exc:
        _emit({"error": f"broker unreachable: {exc}"})
        return EXIT_BROKER_UNREACHABLE
    payload = _json_body(response)
    if payload is None:
        _emit(
            {
                "error": "broker returned a non-JSON sweep-list response",
                "status_code": response.status_code,
            }
        )
        return EXIT_BROKER_UNREACHABLE
    _emit(payload)
    return EXIT_OK if response.status_code == 200 else EXIT_BAD_INPUT


def _budget(args: argparse.Namespace) -> int:
    try:
        response = requests.get(
            f"{args.broker_url.rstrip('/')}/budget", timeout=args.request_timeout
        )
    except requests.RequestException as exc:
        _emit({"error": f"broker unreachable: {exc}"})
        return EXIT_BROKER_UNREACHABLE
    payload = _json_body(response)
    if payload is None:
        _emit(
            {
                "error": "broker returned a non-JSON budget response",
                "status_code": response.status_code,
            }
        )
        return EXIT_BROKER_UNREACHABLE
    _emit(payload)
    return EXIT_OK if response.status_code == 200 else EXIT_BAD_INPUT


def _broker_sweep_deadline(broker: str, args: argparse.Namespace) -> float:
    """Fetch the broker's configured per-sweep deadline for the poll bound.

    The broker is the authority on its own deadline (``GET /budget``); the
    ``--sweep-deadline-seconds`` flag is only the fallback when the budget
    endpoint cannot be read, so the two can never silently disagree.
    """

    try:
        response = requests.get(f"{broker}/budget", timeout=args.request_timeout)
    except requests.RequestException:
        return args.sweep_deadline_seconds
    payload = _json_body(response)
    if response.status_code != 200 or payload is None:
        return args.sweep_deadline_seconds
    value = payload.get("sweep_deadline_seconds")
    if isinstance(value, int | float) and math.isfinite(value) and value > 0:
        return float(value)
    return args.sweep_deadline_seconds


def _request_sweep(args: argparse.Namespace) -> int:
    broker = args.broker_url.rstrip("/")
    sweep_deadline_seconds = _broker_sweep_deadline(broker, args)
    request_body: dict[str, Any] = {"output_dir": args.output_dir}
    try:
        active_search = _parse_active_search(args)
    except argparse.ArgumentTypeError as exc:
        _emit({"error": str(exc)})
        return EXIT_BAD_INPUT
    if active_search is not None:
        request_body["active_search"] = active_search
    if args.max_trials is not None:
        request_body["max_trials"] = args.max_trials
    try:
        response = requests.post(
            f"{broker}/sweeps", json=request_body, timeout=args.request_timeout
        )
    except requests.RequestException as exc:
        _emit({"error": f"broker unreachable: {exc}"})
        return EXIT_BROKER_UNREACHABLE
    payload = _json_body(response)
    if payload is None:
        _emit(
            {
                "error": "broker returned a non-JSON sweep response",
                "status_code": response.status_code,
            }
        )
        return EXIT_BROKER_UNREACHABLE
    if response.status_code == 503:
        _emit(payload)
        return EXIT_SWEEP_IN_PROGRESS
    if response.status_code == 409:
        _emit(payload)
        return EXIT_BUDGET_REFUSED
    if response.status_code != 200:
        _emit(payload)
        return EXIT_BAD_INPUT

    sweep_id = payload["sweep_id"]
    # The per-sweep deadline makes the broker cancel its own sweeps; cap the
    # polling loop at 2x that so this client always exits even if the broker
    # crashes between state transitions.
    max_polls = max(1, int(sweep_deadline_seconds / args.poll_seconds * 2))
    polls = 0
    while payload.get("status") not in TERMINAL_SWEEP_STATUSES:
        if polls >= max_polls:
            _emit(
                {"error": "sweep polling timed out without reaching a terminal status"}
            )
            return EXIT_SWEEP_FAILED
        time.sleep(args.poll_seconds)
        polls += 1
        try:
            poll = requests.get(
                f"{broker}/sweeps/{sweep_id}", timeout=args.request_timeout
            )
        except requests.RequestException as exc:
            _emit({"error": f"broker unreachable while polling: {exc}"})
            return EXIT_BROKER_UNREACHABLE
        poll_payload = _json_body(poll)
        if poll_payload is None:
            _emit(
                {
                    "error": "broker returned a non-JSON poll response",
                    "status_code": poll.status_code,
                }
            )
            return EXIT_BROKER_UNREACHABLE
        if poll.status_code != 200 or "status" not in poll_payload:
            # A broker error body (a 404/500 {"error": ...}) has no status
            # and will never become terminal; fail fast instead of burning
            # the full polling budget before reporting it.
            _emit(poll_payload)
            return EXIT_BAD_INPUT
        payload = poll_payload

    _emit(payload)
    status = payload.get("status")
    if status == "succeeded":
        return EXIT_OK
    if status in {"cancelled", "deadline_exceeded"}:
        return EXIT_DEADLINE
    return EXIT_SWEEP_FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="content-workflow-physics-external-sweep",
        description=(
            "Request one budgeted, judge-free external-runtime (BYOR) tuning "
            "sweep from the wrapper-owned broker and wait for its evidence "
            "packet."
        ),
    )
    parser.add_argument(
        "--broker-url",
        required=True,
        help="Sweep broker base URL (provided in the session task prompt).",
    )
    parser.add_argument(
        "--poll-seconds",
        type=_positive_float,
        default=DEFAULT_POLL_SECONDS,
        help="Polling interval while the sweep runs.",
    )
    parser.add_argument(
        "--request-timeout",
        type=_positive_float,
        default=30.0,
        help="Per-HTTP-request timeout in seconds.",
    )
    parser.add_argument(
        "--sweep-deadline-seconds",
        type=_positive_float,
        default=DEFAULT_EXTERNAL_SWEEP_DEADLINE_SECONDS,
        help=(
            "Fallback per-sweep wall-clock deadline for bounding the "
            "polling loop, used only when the broker's GET /budget (the "
            "authority on its own configured deadline) cannot be read."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sweep = subparsers.add_parser(
        "run", help="Reserve budget and run one external tuning sweep."
    )
    sweep.add_argument(
        "--output-dir",
        required=True,
        help="Iteration directory under the run directory.",
    )
    sweep.add_argument(
        "--active-search",
        default=None,
        help=(
            'Inline JSON object {"param": {"min": ..., "max": ...}, ...} '
            "naming the active parameter search. Omit to use the runtime "
            "config's declared search. Parameters omitted here are pinned to "
            "the previous sweep's winning values by the broker."
        ),
    )
    sweep.add_argument(
        "--active-search-file",
        default=None,
        help="Path to a JSON file with the same shape as --active-search.",
    )
    sweep.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Requested trials (capped by the broker's per-sweep budget).",
    )
    sweep.set_defaults(handler=_request_sweep)

    budget = subparsers.add_parser(
        "budget", help="Show the remaining sweep budget and deadlines."
    )
    budget.set_defaults(handler=_budget)

    list_sweeps = subparsers.add_parser(
        "list",
        help=(
            "List every reserved sweep and its state (recovery: recover a "
            "sweep_id after a killed run call without reserving new budget)."
        ),
    )
    list_sweeps.set_defaults(handler=_list_sweeps)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
