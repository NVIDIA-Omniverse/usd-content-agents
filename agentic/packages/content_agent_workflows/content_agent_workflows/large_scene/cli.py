# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for durable large-scene orchestration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from content_agent_workflows.common.artifacts import seal_phase_result

from .gates import load_phase_result
from .models import PHASE_ORDER, PhaseName
from .state import (
    HandoffValidationError,
    LargeSceneStateError,
    begin_phase,
    complete_phase,
    create_run,
    fail_phase,
    invalidate_from,
    load_run_state,
    revise_additional_instructions,
    validate_phase_handoff,
)


def _phase(value: str) -> PhaseName:
    if value not in PHASE_ORDER:
        raise argparse.ArgumentTypeError(
            f"phase must be one of: {', '.join(PHASE_ORDER)}"
        )
    return value  # type: ignore[return-value]


def _add_run_state(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-state", type=Path, required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Coordinate decomposition, asset tasks, and collection."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create")
    _add_run_state(create_parser)
    create_parser.add_argument("--run-id", required=True)
    create_parser.add_argument("--source-scene", type=Path, required=True)
    create_parser.add_argument("--task", action="append", required=True)
    create_parser.add_argument(
        "--input-artifact", type=Path, action="append", default=[]
    )
    create_parser.add_argument(
        "--additional-instructions",
        help="Scene-level user guidance carried through every workflow phase.",
    )
    create_parser.add_argument(
        "--additional-instructions-file",
        type=Path,
        help="File containing scene-level user guidance.",
    )
    create_parser.add_argument("--actor", default="agent")

    status_parser = subparsers.add_parser("status")
    _add_run_state(status_parser)

    for command in ("begin-phase", "validate-handoff", "complete-phase"):
        command_parser = subparsers.add_parser(command)
        _add_run_state(command_parser)
        command_parser.add_argument("--phase", type=_phase, required=True)
        if command != "begin-phase":
            command_parser.add_argument("--result", type=Path, required=True)
        if command != "validate-handoff":
            command_parser.add_argument("--actor", default="agent")

    seal_parser = subparsers.add_parser("seal-result")
    seal_parser.add_argument("--phase", type=_phase, required=True)
    seal_parser.add_argument("--result", type=Path, required=True)

    fail_parser = subparsers.add_parser("fail-phase")
    _add_run_state(fail_parser)
    fail_parser.add_argument("--phase", type=_phase, required=True)
    fail_parser.add_argument("--reason", required=True)
    fail_parser.add_argument("--actor", default="agent")

    invalidate_parser = subparsers.add_parser("invalidate-from")
    _add_run_state(invalidate_parser)
    invalidate_parser.add_argument("--phase", type=_phase, required=True)
    invalidate_parser.add_argument("--reason", required=True)
    invalidate_parser.add_argument("--actor", default="agent")

    revise_parser = subparsers.add_parser("revise-instructions")
    _add_run_state(revise_parser)
    instruction_group = revise_parser.add_mutually_exclusive_group(required=True)
    instruction_group.add_argument("--additional-instructions")
    instruction_group.add_argument("--additional-instructions-file", type=Path)
    revise_parser.add_argument("--reason", required=True)
    revise_parser.add_argument("--actor", default="agent")
    return parser


def _additional_instructions(args: argparse.Namespace) -> str | None:
    instructions = args.additional_instructions
    if args.additional_instructions_file is not None:
        path = args.additional_instructions_file.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"--additional-instructions-file does not exist: {path}")
        file_text = path.read_text(encoding="utf-8").strip()
        instructions = (
            f"{instructions.rstrip()}\n{file_text}"
            if instructions and file_text
            else instructions or file_text
        )
    return instructions.strip() if instructions and instructions.strip() else None


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "create":
            output = create_run(
                args.run_state,
                run_id=args.run_id,
                source_scene=args.source_scene,
                requested_tasks=args.task,
                request_artifact_paths=args.input_artifact,
                additional_instructions=_additional_instructions(args),
                actor=args.actor,
            )
        elif args.command == "status":
            output = load_run_state(args.run_state)
        elif args.command == "begin-phase":
            output = begin_phase(
                args.run_state,
                args.phase,
                actor=args.actor,
            )
        elif args.command == "validate-handoff":
            output = validate_phase_handoff(
                args.run_state,
                args.phase,
                args.result,
            )
            print(output.model_dump_json(indent=2))
            return 0 if output.valid else 1
        elif args.command == "complete-phase":
            output = complete_phase(
                args.run_state,
                args.phase,
                args.result,
                actor=args.actor,
            )
        elif args.command == "seal-result":
            result = load_phase_result(args.phase, args.result)
            output = seal_phase_result(result, args.result)
        elif args.command == "fail-phase":
            output = fail_phase(
                args.run_state,
                args.phase,
                reason=args.reason,
                actor=args.actor,
            )
        elif args.command == "invalidate-from":
            output = invalidate_from(
                args.run_state,
                args.phase,
                reason=args.reason,
                actor=args.actor,
            )
        elif args.command == "revise-instructions":
            instructions = _additional_instructions(args)
            if instructions is None:
                raise ValueError("revise-instructions requires non-empty instructions")
            output = revise_additional_instructions(
                args.run_state,
                additional_instructions=instructions,
                reason=args.reason,
                actor=args.actor,
            )
        else:  # pragma: no cover - argparse enforces the command set
            raise AssertionError(f"Unhandled command: {args.command}")
    except HandoffValidationError as exc:
        print(exc.report.model_dump_json(indent=2))
        return 1
    except LargeSceneStateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(output.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
