"""Prepare native Validation, without launching a model, renderer or solver.

The command plan invokes the installed workflow later. No approval/assessment is
created here. Native failure and source/task identity mismatches fail closed.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

MODEL = "gpt-6-astra"
EFFORT = "ultra"
CAPABILITIES = (
    "validation.render_valid",
    "validation.physics_sane",
    "validation.physical_behavior",
)
SPEC_SHA256 = "90cde935ce539607c6a770f6ff0ee8cc8c5bc37ba2f18c76802e5cdfe70bf1cd"
EXPECTED_SEEDS = (11, 23, 47, 83, 131)
TASK = """Validate the exact authored drawer USD. Select exactly the three provider-free capabilities validation.render_valid, validation.physics_sane, and validation.physical_behavior, all required with all_required_evidence and explicit dependencies. Use native OVRTX canonical rendering; no external VLM or other model leaf is configured. physical_behavior must consume the supplied strict native_physics_bundle; no legacy approve record, waiver, skipped required check or status promotion is permitted. This required behavior gate covers only actual native mounted runtime stability and its recorded visual review. It does not establish articulated opening, loaded-payload retention or the independent five-seed task. Preserve the original Geometry conditional handoff and all prior failed Physics attempts. The separately hash-bound five-seed task is part of the final capstone conjunction, not evidence automatically evaluated by this native adapter. Planning/check exit0 is not terminal native Validation acceptance: actual collected evidence, outer AstraUltra typed assessment, independent AstraUltra review and native terminal receipt are still required. No universal SimReady claim."""


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def check_task(asset, expected_sha, report_path, bindings_path, spec_path):
    require(sha(asset) == expected_sha, "Final asset differs from requested SHA256.")
    spec = read(spec_path)
    require(sha(spec_path) == SPEC_SHA256, "Unrecognized source-clear specification.")
    report, bindings = read(report_path), read(bindings_path)
    require(
        Path(bindings["final_usd"]).resolve(strict=True) == asset.resolve(),
        "Task binding identifies a different asset path.",
    )
    require(
        report.get("status") in {"PASS", "FAIL"}
        and isinstance(report.get("pass"), bool),
        "Task report is incomplete.",
    )
    require(
        report["inputs"]["usd_sha256"] == expected_sha,
        "Task report identifies a different asset digest.",
    )
    require(
        report["inputs"]["bindings_sha256"] == sha(bindings_path),
        "Task bindings digest is stale.",
    )
    require(
        report["inputs"]["protocol_sha256"] == SPEC_SHA256,
        "Task specification digest mismatch.",
    )
    trials = report.get("trials", [])
    require(
        sorted(t.get("seed", -1) for t in trials) == list(EXPECTED_SEEDS),
        "Exactly five distinct prescribed trials are required.",
    )
    paths = [asset, report_path, bindings_path, spec_path]
    for trial in trials:
        require(
            trial.get("status") in {"PASS", "FAIL"}
            and isinstance(trial.get("pass"), bool),
            "Trial incomplete.",
        )
        require(
            (trial["status"] == "PASS") == trial["pass"],
            "Trial status/pass contradiction.",
        )
        path = report_path.parent / f"seed_{trial['seed']}" / "trial_report.json"
        require(
            read(path) == trial, "Aggregate trial differs from actual per-seed report."
        )
        paths.append(path)
    all_pass = all(t["pass"] for t in trials)
    require(
        report["pass"] == all_pass and (report["status"] == "PASS") == all_pass,
        "Task aggregate contradicts trials.",
    )
    return {
        "task_status": report["status"],
        "task_pass": all_pass,
        "trials_passed": sum(t["pass"] for t in trials),
        "artifacts": [binding(p) for p in paths],
    }


def command_plan(cli, repo, asset, physics_run, output, policy):
    run = output / "native_run"
    args = [
        str(cli),
        "validate",
        "run",
        "--usd",
        str(asset),
        "--task",
        TASK,
        "--output-dir",
        str(run),
        "--base-dir",
        str(physics_run),
        "--repo-root",
        str(repo),
        "--policy-file",
        str(policy),
        "--runner",
        "codex",
        "--model",
        MODEL,
        "--model-reasoning-effort",
        EFFORT,
        "--render-backend",
        "ovrtx",
        "--render-width",
        "1024",
        "--render-height",
        "768",
        "--child-timeout",
        "1800",
        "--fail-on-warn",
        "--json",
    ]
    for capability in CAPABILITIES:
        args += ["--required-capability", capability]
    return {
        "run": args,
        "collect": [
            str(cli),
            "validate",
            "collect-evidence",
            "--output-dir",
            str(run),
            "--json",
        ],
        "assess": [
            str(cli),
            "validate",
            "assess",
            "--output-dir",
            str(run),
            "--assessment",
            str(output / "outer_assessment.json"),
            "--json",
        ],
        "review": [
            str(cli),
            "validate",
            "review-assessment",
            "--output-dir",
            str(run),
            "--review",
            str(output / "independent_review.json"),
            "--json",
        ],
    }


def prepare(args):
    asset, run, repo, output = (
        p.resolve() for p in (args.asset, args.physics_run, args.repo, args.output)
    )
    require(
        not output.exists(),
        "Use a new output directory; no prior evidence is overwritten.",
    )
    task = check_task(
        asset,
        args.expected_asset_sha256,
        args.task_report.resolve(),
        args.bindings.resolve(),
        args.spec.resolve(),
    )
    manifest_path = run / "workflow_run_manifest.json"
    require(
        read(manifest_path).get("status") == "pass",
        "Native Physics workflow is not pass; retain failure and obtain fresh honest evidence.",
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    require(
        commit == args.expected_commit,
        "Runtime repository differs from explicitly selected qualified commit.",
    )
    for package in [
        repo,
        repo / "agentic/packages/content_agent_workflows",
        repo / "agentic/packages/content_workflow_cli",
        repo / "apps/usd_cli/src",
    ]:
        sys.path.insert(0, str(package))
    from world_understanding.functions.physics.native_behavior_validation import (
        prepare_native_physics_bundle,
        validate_native_physics_bundle,
    )

    bundle = prepare_native_physics_bundle(
        asset=asset,
        assessment=run / "physics_behavior_assessment.json",
        validation_evidence=run / "validation_evidence.json",
        run_dir=run,
    )
    original_inputs = task["artifacts"] + [
        binding(manifest_path),
        binding(args.geometry_evidence.resolve()),
        *bundle["artifacts"],
    ]
    code_files = [
        "world_understanding/functions/physics/native_behavior_validation.py",
        "world_understanding/agentic/validation_scaffold.py",
        "world_understanding/validation/scaffold_runner.py",
        "agentic/packages/content_workflow_cli/content_workflow_cli/cli.py",
        "agentic/packages/content_workflow_cli/content_workflow_cli/validation_runner.py",
        "agentic/packages/content_agent_workflows/content_agent_workflows/validation/coordinator.py",
        "agentic/packages/content_agent_workflows/content_agent_workflows/validation/standalone_assessment.py",
        "agentic/packages/content_agent_workflows/content_agent_workflows/validation/embedded_assessment.py",
    ]
    original_inputs += [binding(repo / name) for name in code_files]
    require(
        read(args.geometry_evidence).get("sim_ready_status") == "conditional",
        "Expected historical Geometry conditional evidence is missing or changed in meaning.",
    )
    policy = {
        "behavior_evidence_required": True,
        "expect_physics": True,
        "canonical_visual_evidence": True,
        "runtime_render_usd": True,
        "physical_behavior_evidence": [
            {
                "path": str(output / "native_physics_bundle.json"),
                "kind": "simulation_json",
                "role": "native_physics_bundle",
                "required": True,
            }
        ],
    }
    cli = repo / ".venv/bin/content-workflow-cli"
    require(cli.is_file(), "Installed native CLI executable is missing.")
    commands = command_plan(cli, repo, asset, run, output, output / "policy.json")
    # Parse every future command using the real installed CLI, without handling it.
    from content_workflow_cli.cli import build_parser

    parser = build_parser()
    for command in commands.values():
        parser.parse_args(command[1:])
    output.mkdir(parents=True)
    for name, value in [
        ("native_physics_bundle.json", bundle),
        ("policy.json", policy),
        ("commands.json", commands),
    ]:
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    validation = validate_native_physics_bundle(
        output / "native_physics_bundle.json", usd_paths=[asset]
    )
    require(
        validation["status"] == "passed",
        "New bundle failed exact native consumer verification.",
    )
    from content_agent_workflows.validation.embedded_assessment import (
        ValidationCoordinatorAssessment,
        ValidationCoordinatorReviewDraft,
    )

    for name, model in [
        ("assessment.schema.json", ValidationCoordinatorAssessment),
        ("review.schema.json", ValidationCoordinatorReviewDraft),
    ]:
        (output / name).write_text(
            json.dumps(model.model_json_schema(), indent=2) + "\n"
        )
    for item in original_inputs:
        require(
            binding(Path(item["path"])) == item, "An input changed during preparation."
        )
    receipt = {
        "schema_version": "capstone-native-validation-preparation.v1",
        "status": "PREPARED_NOT_EXECUTED",
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "code_commit": commit,
        "repository": str(repo),
        "asset": binding(asset),
        "task": task,
        "native_bundle_check": validation,
        "input_bindings": original_inputs,
        "prepared_files": [binding(p) for p in sorted(output.iterdir())],
        "native_models_renderers_solvers_launched": False,
        "native_terminal_acceptance": False,
        "final_capstone_acceptance": False,
        "scope": "Precheck only. A clean native bundle does not establish task acceptance; conditional Geometry and prior failures remain historical limitations.",
    }
    (output / "preparation_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "asset",
        "physics-run",
        "task-report",
        "bindings",
        "spec",
        "geometry-evidence",
        "repo",
        "output",
    ]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-asset-sha256", required=True)
    p.add_argument("--expected-commit", required=True)
    args = p.parse_args()
    result = prepare(args)
    print(
        "Preparation receipt SHA256: " + sha(args.output / "preparation_receipt.json")
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "model",
                    "reasoning_effort",
                    "code_commit",
                    "native_terminal_acceptance",
                    "final_capstone_acceptance",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
