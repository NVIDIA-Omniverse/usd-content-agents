"""Synthetic launcher qualification; real native closure/parser, zero providers."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import pytest
from prepare_validation import check_geometry_handoff, check_task, command_plan, prepare, sha, SPEC_SHA256
from run_stage import verify_preparation, launch_environment

pytest_plugins = ["test_native_physics_behavior_validation"]
REPO = Path(os.environ["CAPSTONE_TEST_REPO"]).resolve()
SPEC = Path(os.environ["CAPSTONE_TEST_SPEC"]).resolve()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def inputs(native, tmp_path):
    # Existing qualified synthetic producer fixture uses the real native verifier.
    asset = native["asset"]
    run = native["run_dir"]
    # Canonical native names, preserving references and recomputing actual closure.
    import shutil

    shutil.copyfile(native["assessment"], run / "physics_behavior_assessment.json")
    validation_text = (
        native["validation_evidence"]
        .read_text()
        .replace(
            str(native["assessment"]), str(run / "physics_behavior_assessment.json")
        )
    )
    (run / "validation_evidence.json").write_text(validation_text)
    write(run / "workflow_run_manifest.json", {"status": "pass"})
    bindings = write(tmp_path / "bindings.json", {"final_usd": str(asset)})
    geometry = write(tmp_path / "geometry.json", {
        "schema_version": "content-agent-workflows.validation-evidence.v1",
        "workflow": "geometry",
        "asset": "original_geometry.usdc",
        "target_runtime": "isaac-lab",
        "validation_tier": "T1_basic_stability",
        "sim_ready_status": "not_evaluated",
        "metadata": {
            "schema_version": "content-agent-workflows.geometry.v3",
            "handoff_ready": "conditional",
            "formal_simready_mode": "skip",
        },
        "warnings": ["The mesh contains zero area faces."],
        "unresolved_issues": ["The mesh contains zero area faces."],
    })
    trials = []
    for seed in [11, 23, 47, 83, 131]:
        trial = {"seed": seed, "status": "PASS", "pass": True}
        trials.append(trial)
        write(tmp_path / f"task/seed_{seed}/trial_report.json", trial)
    report = write(
        tmp_path / "task/report.json",
        {
            "status": "PASS",
            "pass": True,
            "inputs": {
                "usd_sha256": sha(asset),
                "bindings_sha256": sha(bindings),
                "protocol_sha256": SPEC_SHA256,
            },
            "trials": trials,
        },
    )
    return argparse.Namespace(
        asset=asset,
        expected_asset_sha256=sha(asset),
        physics_run=run,
        task_report=report,
        bindings=bindings,
        spec=SPEC,
        geometry_evidence=geometry,
        repo=REPO,
        output=tmp_path / "prepared",
        expected_commit=subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
    )


def task(args):
    return check_task(
        args.asset,
        args.expected_asset_sha256,
        args.task_report,
        args.bindings,
        args.spec,
    )


def test_real_native_synthetic_preparation_no_launch(inputs):
    result = prepare(inputs)
    assert result["status"] == "PREPARED_NOT_EXECUTED"
    assert not result["native_models_renderers_solvers_launched"]
    assert (
        not result["native_terminal_acceptance"]
        and not result["final_capstone_acceptance"]
    )
    policy = json.loads((inputs.output / "policy.json").read_text())
    assert policy["canonical_visual_evidence"] and policy["runtime_render_usd"]
    verify_preparation(inputs.output, sha(inputs.output / "preparation_receipt.json"))
    assert result["geometry_handoff"]["sim_ready_status"] == "not_evaluated"
    assert result["geometry_handoff"]["handoff_ready"] == "conditional"


@pytest.mark.parametrize("mutation", [
    "missing_handoff", "unconditional_handoff", "missing_metadata_schema",
    "wrong_metadata_schema", "wrong_workflow", "top_level_conditional",
    "missing_simready", "wrong_evidence_schema",
])
def test_geometry_handoff_mismatch_rejected_without_output(inputs, mutation):
    data = json.loads(inputs.geometry_evidence.read_text())
    if mutation == "missing_handoff":
        del data["metadata"]["handoff_ready"]
    elif mutation == "unconditional_handoff":
        data["metadata"]["handoff_ready"] = "ready"
    elif mutation == "missing_metadata_schema":
        del data["metadata"]["schema_version"]
    elif mutation == "wrong_metadata_schema":
        data["metadata"]["schema_version"] = "content-agent-workflows.geometry.v2"
    elif mutation == "wrong_workflow":
        data["workflow"] = "physics_authoring"
    elif mutation == "top_level_conditional":
        data["sim_ready_status"] = "conditional"
    elif mutation == "missing_simready":
        del data["sim_ready_status"]
    else:
        data["schema_version"] = "unrecognized"
    write(inputs.geometry_evidence, data)
    before = sha(inputs.geometry_evidence)
    with pytest.raises(ValueError, match="Geometry v3 conditional handoff"):
        prepare(inputs)
    assert sha(inputs.geometry_evidence) == before
    assert not inputs.output.exists()


def test_actual_retained_geometry04_receipt(inputs):
    actual = Path(os.environ["CAPSTONE_TEST_GEOMETRY"]).resolve()
    before = sha(actual)
    result = check_geometry_handoff(actual)
    assert result["workflow"] == "geometry"
    assert result["sim_ready_status"] == "not_evaluated"
    assert result["handoff_ready"] == "conditional"
    inputs.geometry_evidence = actual
    prepared = prepare(inputs)
    assert prepared["geometry_handoff"] == result
    assert not prepared["native_terminal_acceptance"]
    assert not prepared["final_capstone_acceptance"]
    assert sha(actual) == before


def test_failed_native_manifest_rejected(inputs):
    write(inputs.physics_run / "workflow_run_manifest.json", {"status": "fail"})
    with pytest.raises(ValueError, match="not pass"):
        prepare(inputs)
    assert not inputs.output.exists()


def test_unresolved_real_typed_assessment_rejected(inputs):
    p = inputs.physics_run / "physics_behavior_assessment.json"
    d = json.loads(p.read_text())
    d["status"] = "unresolved_issues"
    d["unresolved_issues"] = ["synthetic unresolved issue"]
    write(p, d)
    with pytest.raises(ValueError, match="unresolved"):
        prepare(inputs)


def test_task_different_asset_rejected(inputs):
    inputs.expected_asset_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        task(inputs)


def test_stale_bindings_rejected(inputs):
    d = json.loads(inputs.bindings.read_text())
    d["extra"] = "changed"
    write(inputs.bindings, d)
    with pytest.raises(ValueError, match="stale"):
        task(inputs)


def test_task_missing_seed_rejected(inputs):
    d = json.loads(inputs.task_report.read_text())
    d["trials"].pop()
    write(inputs.task_report, d)
    with pytest.raises(ValueError, match="five distinct"):
        task(inputs)


def test_task_contradicting_aggregate_rejected(inputs):
    d = json.loads(inputs.task_report.read_text())
    d["pass"] = False
    write(inputs.task_report, d)
    with pytest.raises(ValueError, match="contradicts"):
        task(inputs)


def test_failed_task_is_retained_not_promoted(inputs):
    d = json.loads(inputs.task_report.read_text())
    d.update(status="FAIL", pass_=False)
    d.pop("pass_")
    d["pass"] = False
    d["trials"][0].update(status="FAIL")
    d["trials"][0]["pass"] = False
    write(inputs.task_report.parent / "seed_11/trial_report.json", d["trials"][0])
    write(inputs.task_report, d)
    result = prepare(inputs)
    assert (
        result["task"]["task_pass"] is False
        and result["final_capstone_acceptance"] is False
    )


def test_stale_prepared_bundle_rejected(inputs):
    prepare(inputs)
    receipt_sha = sha(inputs.output / "preparation_receipt.json")
    (inputs.output / "policy.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        verify_preparation(inputs.output, receipt_sha)


def test_no_overwrite(inputs):
    inputs.output.mkdir()
    with pytest.raises(ValueError, match="new output"):
        prepare(inputs)


def test_cli_shape_model_required_checks(inputs):
    commands = command_plan(
        REPO / ".venv/bin/content-workflow-cli",
        REPO,
        inputs.asset,
        inputs.physics_run,
        inputs.output,
        inputs.output / "policy.json",
    )
    run = commands["run"]
    assert "gpt-6-astra" in run and "ultra" in run
    assert "--direct-executor" not in run and "--template" not in run
    assert run.count("--required-capability") == 3 and "--fail-on-warn" in run
    assert set(commands) == {"run", "collect", "assess", "review"}


def test_reviewed_repo_precedes_inherited_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/synthetic/old/source")
    result = launch_environment(REPO)
    assert result["PYTHONPATH"].split(os.pathsep)[:4] == [
        str(REPO),
        str(REPO / "agentic/packages/content_agent_workflows"),
        str(REPO / "agentic/packages/content_workflow_cli"),
        str(REPO / "apps/usd_cli/src"),
    ]
    assert result["PYTHONPATH"].endswith("/synthetic/old/source")


def test_actual_stage_subprocess_receives_reviewed_context(inputs, monkeypatch):
    import run_stage
    from types import SimpleNamespace

    receipt = prepare(inputs)
    expected = sha(inputs.output / "preparation_receipt.json")
    monkeypatch.setattr(run_stage, "verify_preparation", lambda root, digest: receipt)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write(b"synthetic no-process result")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_stage.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_stage.sys if hasattr(run_stage, "sys") else __import__("sys"),
        "argv",
        [
            "run_stage",
            "--prepared",
            str(inputs.output),
            "--preparation-sha256",
            expected,
            "--stage",
            "run",
            "--execute",
        ],
    )
    with pytest.raises(SystemExit) as result:
        run_stage.main()
    assert result.value.code == 0 and len(calls) == 1
    assert calls[0][1]["cwd"] == str(REPO)
    assert calls[0][1]["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(REPO)
    record = json.loads((inputs.output / "run.execution.json").read_text())
    assert record["terminal_acceptance_inferred_from_exit_code"] is False
