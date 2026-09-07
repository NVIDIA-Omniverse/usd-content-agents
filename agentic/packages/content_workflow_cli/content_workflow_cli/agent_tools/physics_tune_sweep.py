# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request one budgeted physics tuning sweep from the wrapper-owned broker.

This is the agent-facing client of the physics tuning sweep broker. It does
NOT import engine code: the broker reserves budget atomically before running
the sanitized, judge-free ``run_tune`` sweep, and this client just submits
the request, polls until the sweep reaches a terminal state, and prints the
evidence packet location as JSON on stdout.

Exit codes:
  0  sweep succeeded (evidence packet written)
  2  bad input / broker rejected the request
  3  budget or phase deadline refused the reservation
  4  sweep cancelled or per-sweep deadline exceeded
  5  sweep failed inside the engine
  6  broker unreachable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

import requests

from content_workflow_cli.tuning_broker import DEFAULT_SWEEP_DEADLINE_SECONDS

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_BUDGET_REFUSED = 3
EXIT_DEADLINE = 4
EXIT_SWEEP_FAILED = 5
EXIT_BROKER_UNREACHABLE = 6

DEFAULT_POLL_SECONDS = 10.0
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "deadline_exceeded"}


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


def _download_artifact(
    *,
    url: str,
    output_value: str,
    request_timeout: float,
    expected_sha256: str | None = None,
    allow_extensionless: bool = False,
) -> tuple[Path, str]:
    """Download one broker artifact into a newly created local file."""

    output_path = Path(output_value).expanduser().absolute()
    with requests.get(
        url,
        timeout=request_timeout,
        stream=True,
    ) as download:
        if download.status_code != 200:
            try:
                detail = str(download.json().get("error") or "")
            except (AttributeError, ValueError):
                detail = ""
            raise ValueError(detail or f"broker returned HTTP {download.status_code}")
        expected_suffix = str(download.headers.get("X-Content-Suffix") or "").lower()
        if allow_extensionless and not expected_suffix:
            # Closure members may legitimately be extensionless USD assets;
            # the name (and digest) must still be preserved exactly.
            if output_path.suffix:
                raise ValueError(
                    "output path must preserve the broker artifact's extensionless name"
                )
        elif not expected_suffix or output_path.suffix.lower() != expected_suffix:
            raise ValueError(
                "output path must preserve the broker artifact suffix "
                f"{expected_suffix!r}"
            )
        header_sha256 = str(download.headers.get("X-Content-SHA256") or "")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        digest = hashlib.sha256()
        fd = os.open(output_path, flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                for chunk in download.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    stream.write(chunk)
                    digest.update(chunk)
        except Exception:
            output_path.unlink(missing_ok=True)
            raise

    observed_sha256 = digest.hexdigest()
    if (
        not header_sha256
        or observed_sha256 != header_sha256
        or (expected_sha256 is not None and observed_sha256 != expected_sha256)
    ):
        output_path.unlink(missing_ok=True)
        raise ValueError("broker artifact digest mismatch")
    return output_path, observed_sha256


def _materialize(args: argparse.Namespace) -> int:
    try:
        response = requests.post(
            f"{args.broker_url.rstrip('/')}/sweeps/{args.sweep_id}/materialize",
            json={"trial_index": args.trial_index},
            timeout=args.request_timeout,
        )
    except requests.RequestException as exc:
        _emit({"error": f"broker unreachable: {exc}"})
        return EXIT_BROKER_UNREACHABLE
    payload = response.json()
    if response.status_code != 200 or (
        not args.output_usd and not args.output_recording
    ):
        _emit(payload)
        return EXIT_OK if response.status_code == 200 else EXIT_BAD_INPUT
    created_outputs: list[Path] = []
    try:
        if args.output_usd:
            output_path, digest = _download_artifact(
                url=(
                    f"{args.broker_url.rstrip('/')}/sweeps/{args.sweep_id}/"
                    f"materialized/{args.trial_index}"
                ),
                output_value=args.output_usd,
                request_timeout=args.request_timeout,
                expected_sha256=str(payload.get("usd_sha256") or ""),
            )
            created_outputs.append(output_path)
            payload["exported_usd_path"] = str(output_path)
            payload["exported_usd_sha256"] = digest
            # A localized candidate references its "*_assets" sidecar
            # relatively; download the dependency closure alongside the
            # root so the exported bundle composes outside the broker.
            closure_url = (
                f"{args.broker_url.rstrip('/')}/sweeps/{args.sweep_id}/"
                f"materialized/{args.trial_index}/closure"
            )
            closure_response = requests.get(closure_url, timeout=args.request_timeout)
            if closure_response.status_code != 200:
                try:
                    detail = str(closure_response.json().get("error") or "")
                except (AttributeError, ValueError):
                    detail = ""
                raise ValueError(
                    detail
                    or "broker returned HTTP "
                    f"{closure_response.status_code} for the candidate closure"
                )
            closure_payload = closure_response.json()
            if not isinstance(closure_payload, dict) or not isinstance(
                closure_payload.get("members") or [], list
            ):
                raise ValueError(
                    "broker returned a malformed closure payload: "
                    f"{type(closure_payload).__name__}"
                )
            closure_members = closure_payload.get("members") or []
            # Preflight the member list before any download: duplicate
            # relative paths or a member colliding with the root USD name
            # would otherwise surface as a late partial-export failure.
            seen_relative: set[str] = set()
            for member in closure_members:
                # Validate the payload SHAPE before anything downstream
                # touches it: a malformed broker response must surface as
                # the documented bad-input error, not a TypeError traceback
                # from the download loop.
                if (
                    not isinstance(member, dict)
                    or not isinstance(member.get("index"), int)
                    or isinstance(member.get("index"), bool)
                    or not isinstance(member.get("sha256"), str)
                    or not member.get("sha256")
                    or not isinstance(member.get("relative_path"), str)
                ):
                    raise ValueError(
                        f"broker returned a malformed closure member: {member!r}"
                    )
                relative_probe = str(member.get("relative_path") or "")
                if "." in PurePosixPath(relative_probe).parts:
                    raise ValueError(
                        "broker returned a non-normalized closure member "
                        f"path: {relative_probe!r}"
                    )
                if relative_probe in seen_relative:
                    raise ValueError(
                        "broker returned duplicate closure member path: "
                        f"{relative_probe!r}"
                    )
                seen_relative.add(relative_probe)
            if output_path.name in seen_relative:
                raise ValueError(
                    "broker returned a closure member colliding with the "
                    f"root USD name: {output_path.name!r}"
                )
            exported_members: list[dict[str, str]] = []
            for member in closure_members:
                relative_raw = str(member.get("relative_path") or "")
                relative = PurePosixPath(relative_raw)
                if (
                    not relative_raw
                    or relative.is_absolute()
                    or any(part in ("..", "") for part in relative.parts)
                ):
                    raise ValueError(
                        f"broker returned an unsafe closure member path: "
                        f"{relative_raw!r}"
                    )
                member_output = output_path.parent.joinpath(*relative.parts)
                member_path, member_digest = _download_artifact(
                    url=f"{closure_url}/{int(member['index'])}",
                    output_value=str(member_output),
                    request_timeout=args.request_timeout,
                    expected_sha256=str(member.get("sha256") or ""),
                    allow_extensionless=True,
                )
                created_outputs.append(member_path)
                exported_members.append(
                    {
                        "relative_path": relative_raw,
                        "path": str(member_path),
                        "sha256": member_digest,
                    }
                )
            payload["exported_sidecar_members"] = exported_members
        if args.output_recording:
            recording_path, recording_digest = _download_artifact(
                url=(
                    f"{args.broker_url.rstrip('/')}/sweeps/{args.sweep_id}/"
                    f"recordings/{args.trial_index}"
                ),
                output_value=args.output_recording,
                request_timeout=args.request_timeout,
            )
            created_outputs.append(recording_path)
            payload["exported_recording_path"] = str(recording_path)
            payload["exported_recording_sha256"] = recording_digest
    except (OSError, requests.RequestException, ValueError) as exc:
        for path in created_outputs:
            path.unlink(missing_ok=True)
        _emit({**payload, "error": f"candidate export failed: {exc}"})
        return EXIT_BAD_INPUT
    _emit(payload)
    return EXIT_OK


def _request_sweep(args: argparse.Namespace) -> int:
    broker = args.broker_url.rstrip("/")
    request_body = {
        "scenario_path": args.scenario,
        "physics_usd": args.physics_usd,
        "output_dir": args.output_dir,
    }
    if args.engine:
        request_body["engine"] = args.engine
    if args.optimizer:
        request_body["optimizer"] = args.optimizer
    if args.max_trials is not None:
        request_body["max_trials"] = args.max_trials
    if args.rebuilt_decision:
        request_body["rebuilt_decision_path"] = args.rebuilt_decision
    try:
        response = requests.post(
            f"{broker}/sweeps", json=request_body, timeout=args.request_timeout
        )
    except requests.RequestException as exc:
        _emit({"error": f"broker unreachable: {exc}"})
        return EXIT_BROKER_UNREACHABLE
    payload = response.json()
    if response.status_code == 409:
        _emit(payload)
        return EXIT_BUDGET_REFUSED
    if response.status_code != 200:
        _emit(payload)
        return EXIT_BAD_INPUT

    sweep_id = payload["sweep_id"]
    # Guard against a broker that stalls without reaching a terminal status.
    # The per-sweep deadline ensures the broker cancels its own sweeps, but
    # a crash between state transitions could leave the status permanently
    # non-terminal. Cap at 2× the sweep deadline so the client always exits.
    max_polls = max(1, int(args.sweep_deadline_seconds / args.poll_seconds * 2))
    polls = 0
    while payload.get("status") not in TERMINAL_STATUSES:
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
        payload = poll.json()

    _emit(payload)
    status = payload.get("status")
    if status == "succeeded":
        return EXIT_OK
    if status in {"cancelled", "deadline_exceeded"}:
        return EXIT_DEADLINE
    return EXIT_SWEEP_FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="content-workflow-physics-tune-sweep",
        description=(
            "Request one budgeted, judge-free physics tuning sweep from the "
            "wrapper-owned broker and wait for its evidence packet."
        ),
    )
    parser.add_argument(
        "--broker-url",
        required=True,
        help="Sweep broker base URL (provided in the tuning task prompt).",
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
        default=DEFAULT_SWEEP_DEADLINE_SECONDS,
        help=(
            "Expected per-sweep wall-clock deadline (used to bound the polling "
            "loop; should match the broker's configured value)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sweep = subparsers.add_parser(
        "run", help="Reserve budget and run one tuning sweep."
    )
    sweep.add_argument("--scenario", required=True, help="Scenario YAML path.")
    sweep.add_argument(
        "--physics-usd", required=True, help="Sweep input physics USD path."
    )
    sweep.add_argument(
        "--output-dir",
        required=True,
        help="Iteration directory under the run directory.",
    )
    sweep.add_argument(
        "--engine",
        default=None,
        help=(
            "Informational only: the engine is broker-enforced and a "
            "mismatching value is rejected. Omit to use the broker's engine."
        ),
    )
    sweep.add_argument("--optimizer", default=None, help="Override optimizer.")
    sweep.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Requested trials (capped by the broker's per-sweep budget).",
    )
    sweep.add_argument(
        "--rebuilt-decision",
        default=None,
        help=(
            "Required when --physics-usd is a rebuilt artifact rather than "
            "the finalized input: path to the revise_patch decision file "
            "whose rebuilt_physics_usd_sha256 binds it."
        ),
    )
    sweep.set_defaults(handler=_request_sweep)

    materialize = subparsers.add_parser(
        "materialize",
        help="Materialize one top-K candidate trial into an immutable USD.",
    )
    materialize.add_argument("--sweep-id", required=True)
    materialize.add_argument("--trial-index", type=int, required=True)
    materialize.add_argument(
        "--output-usd",
        default=None,
        help=(
            "Optional child-writable path for a digest-verified copy used by "
            "usd-cli candidate simulation and rendering."
        ),
    )
    materialize.add_argument(
        "--output-recording",
        default=None,
        help=(
            "Optional child-writable path for the digest-verified recording "
            "produced by this exact scenario trial."
        ),
    )
    materialize.set_defaults(handler=_materialize)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
