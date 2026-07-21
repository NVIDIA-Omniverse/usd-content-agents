# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin module command for Workflow 2 deterministic state operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runtime import (
    AssetTaskRuntimeError,
    begin_work_item,
    commit_work_item,
    fail_work_item,
    finalize_processing_run,
    get_work_item,
    prepare_processing_run,
    processing_status,
    record_plan,
    waive_work_item,
)


def _output_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operate the deterministic state of Workflow 2."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest-catalog", type=Path, required=True)
    prepare.add_argument("--task-catalog", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--input-digest", required=True)

    status = subparsers.add_parser("status")
    _output_dir(status)

    plan = subparsers.add_parser("record-plan")
    _output_dir(plan)
    plan.add_argument("--plan-file", type=Path, required=True)

    show = subparsers.add_parser("show-item")
    _output_dir(show)
    show.add_argument("--work-item-id", required=True)

    begin = subparsers.add_parser("begin-item")
    _output_dir(begin)
    begin.add_argument("--work-item-id", required=True)
    begin.add_argument("--actor", default="agent")

    commit = subparsers.add_parser("commit-item")
    _output_dir(commit)
    commit.add_argument("--work-item-id", required=True)
    commit.add_argument("--result", type=Path, required=True)
    commit.add_argument("--validation", type=Path, required=True)
    commit.add_argument("--ledger-entry", type=Path, required=True)
    commit.add_argument("--actor", default="agent")

    fail = subparsers.add_parser("fail-item")
    _output_dir(fail)
    fail.add_argument("--work-item-id", required=True)
    fail.add_argument("--reason", required=True)
    fail.add_argument("--actor", default="agent")

    waive = subparsers.add_parser("waive-item")
    _output_dir(waive)
    waive.add_argument("--work-item-id", required=True)
    waive.add_argument("--reason", required=True)
    waive.add_argument("--accepted-by", required=True)
    waive.add_argument("--actor", default="agent")

    finalize = subparsers.add_parser("finalize")
    _output_dir(finalize)
    return parser


def _json_output(payload: object) -> None:
    if hasattr(payload, "model_dump"):
        document = payload.model_dump(mode="json")
    elif isinstance(payload, tuple):
        document = [item.model_dump(mode="json") for item in payload]
    else:
        document = payload
    print(json.dumps(document, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            output = prepare_processing_run(
                manifest_catalog_path=args.manifest_catalog,
                task_catalog_path=args.task_catalog,
                output_dir=args.output_dir,
                input_digest=args.input_digest,
            )
        elif args.command == "status":
            output = processing_status(args.output_dir)
        elif args.command == "record-plan":
            output = record_plan(args.output_dir, args.plan_file)
        elif args.command == "show-item":
            output = get_work_item(args.output_dir, args.work_item_id)
        elif args.command == "begin-item":
            output = begin_work_item(
                args.output_dir, args.work_item_id, actor=args.actor
            )
        elif args.command == "commit-item":
            output = commit_work_item(
                args.output_dir,
                args.work_item_id,
                result_path=args.result,
                validation_path=args.validation,
                ledger_entry_path=args.ledger_entry,
                actor=args.actor,
            )
        elif args.command == "fail-item":
            output = fail_work_item(
                args.output_dir,
                args.work_item_id,
                reason=args.reason,
                actor=args.actor,
            )
        elif args.command == "waive-item":
            output = waive_work_item(
                args.output_dir,
                args.work_item_id,
                reason=args.reason,
                accepted_by=args.accepted_by,
                actor=args.actor,
            )
        elif args.command == "finalize":
            output = finalize_processing_run(args.output_dir)
        else:  # pragma: no cover
            raise AssertionError(f"Unhandled command: {args.command}")
    except AssetTaskRuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _json_output(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
