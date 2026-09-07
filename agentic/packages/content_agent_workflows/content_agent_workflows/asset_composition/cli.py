# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal transition CLI for the composed single-asset workflow."""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from .coordinator import run_asset_coordinator_transition
from .execution import execute_active_leaf
from .models import ALL_STAGE_NAMES, ASSET_EXECUTION_GRAPH_SCHEMA_VERSION, StageName
from .state import (
    AssetCompositionStateError,
    _load_execution_graph,
    begin_leaf,
    begin_stage,
    build_combined_report,
    cancel_leaf,
    cancel_stage,
    complete_leaf,
    complete_stage,
    create_run,
    execute_cad_modeling_stage,
    execute_geometry_stage,
    fail_leaf,
    fail_stage,
    finalize_graph_run,
    freeze_execution_graph,
    leaf_directory,
    load_run_state,
    load_verified_run,
    record_coordinator_evidence_review,
    record_coordinator_plan,
    record_review_decisions,
    recover_leaf,
    recover_stage,
    require_review,
    stage_directory,
    validate_terminal,
)


def _stage(value: str) -> StageName:
    if value not in ALL_STAGE_NAMES:
        raise argparse.ArgumentTypeError(
            f"stage must be one of: {', '.join(ALL_STAGE_NAMES)}"
        )
    return cast(StageName, value)


def _leaf_id(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*\.v[1-9][0-9]*", value):
        raise argparse.ArgumentTypeError("leaf ID must be a stable versioned public ID")
    return value


def _add_state(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-state", type=Path, required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Coordinate CAD modeling, Geometry, Joint, Material, Texture, "
            "Physics, Validation, and Finalization through an outer-supplied "
            "agentic graph or an explicit fixed compatibility run."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    _add_state(create)
    create.add_argument("--run-id", required=True)
    create.add_argument("--request", type=Path, required=True)
    create.add_argument("--source-asset", type=Path, required=True)
    create.add_argument("--actor", default="content-workflow-cli")
    create.add_argument("--include-geometry-stage", action="store_true")
    create.add_argument("--include-cad-modeling-stage", action="store_true")

    status = subparsers.add_parser("status")
    _add_state(status)

    freeze_graph = subparsers.add_parser("freeze-graph")
    _add_state(freeze_graph)
    freeze_graph.add_argument("--graph", type=Path, required=True)
    freeze_graph.add_argument("--actor", default="asset-coordinator")

    leaf_dir = subparsers.add_parser("leaf-dir")
    _add_state(leaf_dir)
    leaf_dir.add_argument("--leaf", type=_leaf_id, required=True)
    leaf_dir.add_argument("--attempt", type=int)

    begin_graph_leaf = subparsers.add_parser("begin-leaf")
    _add_state(begin_graph_leaf)
    begin_graph_leaf.add_argument("--leaf", type=_leaf_id, required=True)
    begin_graph_leaf.add_argument("--actor", default="asset-coordinator")

    invoke_graph_leaf = subparsers.add_parser("invoke-leaf")
    _add_state(invoke_graph_leaf)
    invoke_graph_leaf.add_argument("--leaf", type=_leaf_id, required=True)
    invoke_graph_leaf.add_argument("--invocation", type=Path, required=True)
    invoke_graph_leaf.add_argument("--result", type=Path, required=True)

    complete_graph_leaf = subparsers.add_parser("complete-leaf")
    _add_state(complete_graph_leaf)
    complete_graph_leaf.add_argument("--leaf", type=_leaf_id, required=True)
    complete_graph_leaf.add_argument("--invocation", type=Path, required=True)
    complete_graph_leaf.add_argument("--result", type=Path, required=True)
    complete_graph_leaf.add_argument(
        "--native-terminal-receipt",
        type=Path,
        help="Compatibility-only graph-v1 native receipt.",
    )
    complete_graph_leaf.add_argument(
        "--operation-index",
        type=Path,
        action="append",
        default=[],
        help="Compatibility-only graph-v1 operation index.",
    )
    complete_graph_leaf.add_argument(
        "--evidence-index",
        type=Path,
        action="append",
        default=[],
        help="Compatibility-only graph-v1 evidence index.",
    )
    complete_graph_leaf.add_argument(
        "--evidence",
        type=Path,
        action="append",
        default=[],
        help="Compatibility-only graph-v1 evidence artifact.",
    )
    complete_graph_leaf.add_argument(
        "--saved-stage-readback",
        type=Path,
        action="append",
        default=[],
        help="Compatibility-only graph-v1 saved-stage readback.",
    )
    complete_graph_leaf.add_argument(
        "--native-disposition",
        choices=("passed", "not_evaluated"),
        help="Compatibility-only graph-v1 disposition.",
    )
    complete_graph_leaf.add_argument(
        "--resource-claim",
        action="append",
        default=[],
        help="Compatibility-only graph-v1 resource claim.",
    )
    complete_graph_leaf.add_argument(
        "--resource-release",
        type=Path,
        action="append",
        default=[],
        help="Compatibility-only graph-v1 resource release.",
    )
    complete_graph_leaf.add_argument(
        "--summary",
        help="Compatibility-only graph-v1 terminal summary.",
    )
    complete_graph_leaf.add_argument("--actor", default="asset-coordinator")

    for command in ("fail-leaf", "cancel-leaf"):
        action = subparsers.add_parser(command)
        _add_state(action)
        action.add_argument("--leaf", type=_leaf_id, required=True)
        action.add_argument("--reason", required=True)
        action.add_argument("--invocation", type=Path)
        action.add_argument("--result", type=Path)
        action.add_argument(
            "--native-terminal-receipt",
            type=Path,
            help="Compatibility-only graph-v1 native receipt.",
        )
        action.add_argument(
            "--operation-index",
            type=Path,
            action="append",
            default=[],
            help="Compatibility-only graph-v1 operation index.",
        )
        action.add_argument(
            "--evidence-index",
            type=Path,
            action="append",
            default=[],
            help="Compatibility-only graph-v1 evidence index.",
        )
        action.add_argument(
            "--evidence",
            type=Path,
            action="append",
            default=[],
            help="Compatibility-only graph-v1 evidence artifact.",
        )
        action.add_argument(
            "--saved-stage-readback",
            type=Path,
            action="append",
            default=[],
            help="Compatibility-only graph-v1 saved-stage readback.",
        )
        action.add_argument(
            "--resource-claim",
            action="append",
            default=[],
            help="Compatibility-only graph-v1 resource claim.",
        )
        action.add_argument(
            "--resource-release",
            type=Path,
            action="append",
            default=[],
            help="Compatibility-only graph-v1 resource release.",
        )
        action.add_argument("--actor", default="asset-coordinator")

    recover_graph_leaf = subparsers.add_parser("recover-leaf")
    _add_state(recover_graph_leaf)
    recover_graph_leaf.add_argument("--leaf", type=_leaf_id, required=True)
    recover_graph_leaf.add_argument("--reason", required=True)
    recover_graph_leaf.add_argument("--actor", default="operator")

    finalize_graph = subparsers.add_parser("finalize-graph")
    _add_state(finalize_graph)
    finalize_graph.add_argument("--parent-release-receipt", type=Path)
    finalize_graph.add_argument("--parent-command-receipt-journal", type=Path)
    finalize_graph.add_argument("--parent-command-receipt-checkpoint", type=Path)
    finalize_graph.add_argument(
        "--resource-release",
        type=Path,
        action="append",
        default=[],
        help="Compatibility-only graph-v1 parent resource release alias.",
    )
    finalize_graph.add_argument("--actor", default="asset-coordinator")

    stage_dir = subparsers.add_parser("stage-dir")
    _add_state(stage_dir)
    stage_dir.add_argument("--stage", type=_stage, required=True)

    begin = subparsers.add_parser("begin-stage")
    _add_state(begin)
    begin.add_argument("--stage", type=_stage, required=True)
    begin.add_argument("--actor", default="agent")

    execute_geometry = subparsers.add_parser("execute-geometry")
    _add_state(execute_geometry)
    execute_geometry.add_argument("--actor", default="asset-geometry-executor")

    execute_cad = subparsers.add_parser("execute-cad-modeling")
    _add_state(execute_cad)
    execute_cad.add_argument("--actor", default="asset-cad-modeling-executor")
    execute_cad.add_argument(
        "--geometry-authoring-command",
        default=None,
        help=(
            "Installed external geometry-authoring provider command. Defaults "
            "to CONTENT_AGENT_GEOMETRY_AUTHORING_COMMAND, then "
            "geometry-authoring-provider."
        ),
    )

    plan = subparsers.add_parser("record-plan")
    _add_state(plan)
    plan.add_argument("--plan-file", type=Path, required=True)
    plan.add_argument("--actor", default="agent")

    evidence_review = subparsers.add_parser("record-evidence-review")
    _add_state(evidence_review)
    evidence_review.add_argument("--review-file", type=Path, required=True)
    evidence_review.add_argument("--actor", default="agent")

    review = subparsers.add_parser("require-review")
    _add_state(review)
    review.add_argument("--candidates", type=Path, required=True)
    review.add_argument("--actor", default="agent")

    decisions = subparsers.add_parser("record-review")
    _add_state(decisions)
    decisions.add_argument("--decisions", type=Path, required=True)
    decisions.add_argument("--reviewer", required=True)

    complete = subparsers.add_parser("complete-stage")
    _add_state(complete)
    complete.add_argument("--stage", type=_stage, required=True)
    complete.add_argument("--output-asset", type=Path, required=True)
    complete.add_argument("--evidence", type=Path, action="append", default=[])
    complete.add_argument("--summary", required=True)
    complete.add_argument("--actor", default="agent")

    for command in ("fail-stage", "cancel-stage", "recover-stage"):
        action = subparsers.add_parser(command)
        _add_state(action)
        action.add_argument("--stage", type=_stage, required=True)
        action.add_argument("--reason", required=True)
        action.add_argument(
            "--actor",
            default="operator" if command == "recover-stage" else "agent",
        )

    terminal = subparsers.add_parser("validate-terminal")
    _add_state(terminal)
    report = subparsers.add_parser("build-report")
    _add_state(report)
    report.add_argument("--final-asset", type=Path, required=True)
    report.add_argument("--validation-summary", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output: BaseModel
    try:
        agentic_commands = {
            "freeze-graph",
            "leaf-dir",
            "begin-leaf",
            "invoke-leaf",
            "complete-leaf",
            "fail-leaf",
            "cancel-leaf",
            "recover-leaf",
            "finalize-graph",
        }
        compatibility_commands = {
            "stage-dir",
            "begin-stage",
            "record-plan",
            "record-evidence-review",
            "require-review",
            "record-review",
            "complete-stage",
            "fail-stage",
            "cancel-stage",
            "recover-stage",
            "build-report",
        }
        if args.command in agentic_commands | compatibility_commands:
            # Mode routing accepts no artifact claim. Each dispatched state
            # surface owns authoritative verification where its contract needs it.
            run = load_run_state(args.run_state)
            expected_mode = (
                "agentic" if args.command in agentic_commands else "compatibility_fixed"
            )
            if run.selected_mode != expected_mode:
                raise AssetCompositionStateError(
                    f"{args.command} requires {expected_mode} mode, not "
                    f"{run.selected_mode}"
                )
        if args.command == "create":
            output = create_run(
                args.run_state,
                run_id=args.run_id,
                request_path=args.request,
                source_asset=args.source_asset,
                actor=args.actor,
                include_geometry_stage=args.include_geometry_stage,
                include_cad_modeling_stage=args.include_cad_modeling_stage,
            )
        elif args.command == "status":
            output = load_verified_run(args.run_state)
        elif args.command == "freeze-graph":
            output = freeze_execution_graph(
                args.run_state,
                graph_path=args.graph,
                actor=args.actor,
            )
        elif args.command == "leaf-dir":
            print(leaf_directory(args.run_state, args.leaf, attempt=args.attempt))
            return 0
        elif args.command == "begin-leaf":
            output = begin_leaf(args.run_state, args.leaf, actor=args.actor)
        elif args.command == "invoke-leaf":
            output = execute_active_leaf(
                args.run_state,
                args.leaf,
                invocation_path=args.invocation,
                result_path=args.result,
            )
        elif args.command == "complete-leaf":
            output = complete_leaf(
                args.run_state,
                args.leaf,
                invocation_path=args.invocation,
                result_path=args.result,
                native_terminal_receipt_path=args.native_terminal_receipt,
                operation_index_paths=args.operation_index,
                evidence_index_paths=args.evidence_index,
                evidence_paths=args.evidence,
                saved_stage_readback_paths=args.saved_stage_readback,
                native_disposition=args.native_disposition,
                resource_claims=args.resource_claim,
                resource_release_paths=args.resource_release,
                summary=args.summary,
                actor=args.actor,
            )
        elif args.command == "fail-leaf":
            output = fail_leaf(
                args.run_state,
                args.leaf,
                reason=args.reason,
                invocation_path=args.invocation,
                result_path=args.result,
                native_terminal_receipt_path=args.native_terminal_receipt,
                operation_index_paths=args.operation_index,
                evidence_index_paths=args.evidence_index,
                evidence_paths=args.evidence,
                saved_stage_readback_paths=args.saved_stage_readback,
                resource_claims=args.resource_claim,
                resource_release_paths=args.resource_release,
                actor=args.actor,
            )
        elif args.command == "cancel-leaf":
            output = cancel_leaf(
                args.run_state,
                args.leaf,
                reason=args.reason,
                invocation_path=args.invocation,
                result_path=args.result,
                native_terminal_receipt_path=args.native_terminal_receipt,
                operation_index_paths=args.operation_index,
                evidence_index_paths=args.evidence_index,
                evidence_paths=args.evidence,
                saved_stage_readback_paths=args.saved_stage_readback,
                resource_claims=args.resource_claim,
                resource_release_paths=args.resource_release,
                actor=args.actor,
            )
        elif args.command == "recover-leaf":
            output = run_asset_coordinator_transition(
                args.run_state,
                transition=lambda: recover_leaf(
                    args.run_state,
                    args.leaf,
                    reason=args.reason,
                    actor=args.actor,
                ),
            )
        elif args.command == "finalize-graph":
            run = load_verified_run(args.run_state)
            graph = (
                _load_execution_graph(run.execution_graph)
                if run.execution_graph is not None
                else None
            )
            if (
                graph is not None
                and graph.schema_version == ASSET_EXECUTION_GRAPH_SCHEMA_VERSION
            ):
                raise AssetCompositionStateError(
                    "graph v2 finalization is launcher-owned after parent teardown; "
                    "the finalize-graph CLI is compatibility-only for graph v1"
                )
            output = finalize_graph_run(
                args.run_state,
                resource_release_paths=args.resource_release,
                parent_release_receipt_path=args.parent_release_receipt,
                parent_command_receipt_journal_path=(
                    args.parent_command_receipt_journal
                ),
                parent_command_receipt_checkpoint_path=(
                    args.parent_command_receipt_checkpoint
                ),
                actor=args.actor,
            )
        elif args.command == "stage-dir":
            print(stage_directory(args.run_state, args.stage))
            return 0
        elif args.command == "begin-stage":
            output = begin_stage(args.run_state, args.stage, actor=args.actor)
        elif args.command == "execute-geometry":
            output = execute_geometry_stage(args.run_state, actor=args.actor)
            print(output.model_dump_json(indent=2))
            return 0 if output.success else 1
        elif args.command == "execute-cad-modeling":
            output = execute_cad_modeling_stage(
                args.run_state,
                actor=args.actor,
                geometry_authoring_command=(
                    tuple(shlex.split(args.geometry_authoring_command))
                    if args.geometry_authoring_command
                    else None
                ),
            )
            print(output.model_dump_json(indent=2))
            return 0 if output.success else 1
        elif args.command == "record-plan":
            output = record_coordinator_plan(
                args.run_state,
                plan_path=args.plan_file,
                actor=args.actor,
            )
        elif args.command == "record-evidence-review":
            output = record_coordinator_evidence_review(
                args.run_state,
                review_path=args.review_file,
                actor=args.actor,
            )
        elif args.command == "require-review":
            output = require_review(
                args.run_state,
                candidates_path=args.candidates,
                actor=args.actor,
            )
        elif args.command == "record-review":
            output = run_asset_coordinator_transition(
                args.run_state,
                transition=lambda: record_review_decisions(
                    args.run_state,
                    decisions_path=args.decisions,
                    reviewer=args.reviewer,
                ),
            )
        elif args.command == "complete-stage":
            output = complete_stage(
                args.run_state,
                args.stage,
                output_asset=args.output_asset,
                evidence_paths=args.evidence,
                summary=args.summary,
                actor=args.actor,
            )
        elif args.command == "fail-stage":
            output = fail_stage(
                args.run_state,
                args.stage,
                reason=args.reason,
                actor=args.actor,
            )
        elif args.command == "cancel-stage":
            output = cancel_stage(
                args.run_state,
                args.stage,
                reason=args.reason,
                actor=args.actor,
            )
        elif args.command == "recover-stage":
            output = run_asset_coordinator_transition(
                args.run_state,
                transition=lambda: recover_stage(
                    args.run_state,
                    args.stage,
                    reason=args.reason,
                    actor=args.actor,
                ),
            )
        elif args.command == "validate-terminal":
            output = validate_terminal(args.run_state)
            print(output.model_dump_json(indent=2))
            return 0 if output.valid else 1
        elif args.command == "build-report":
            output = build_combined_report(
                args.run_state,
                final_asset=args.final_asset,
                validation_summary=args.validation_summary,
                output_path=args.output,
            )
        else:  # pragma: no cover - argparse enforces the command set
            raise AssertionError(f"Unhandled command: {args.command}")
    except (AssetCompositionStateError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(output.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
