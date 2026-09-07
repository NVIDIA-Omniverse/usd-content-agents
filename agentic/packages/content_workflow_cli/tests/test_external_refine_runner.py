# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end agent-managed external refinement tests with a fake engine and
a scripted child agent: qualification pause, accept-and-publish, honest stop,
and fail-closed verification."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests
import yaml
from content_agent_workflows.physics.external_tuning_contract import sha256_file
from world_understanding.utils.file_locking import exclusive_descriptor_lock

from content_workflow_cli.external_refine_runner import (
    PhysicsExternalRefineConfig,
    _bundle_relative_tune_results,
    run_physics_external_refine,
)

APPROVAL = "sha256:" + "0" * 64


def test_bundle_relative_tune_results_rejects_relative_escape(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    tune_dir = private_root / "tune"
    final_dir = tmp_path / "final"
    tune_dir.mkdir(parents=True)
    final_dir.mkdir()
    (private_root / "escaped.json").write_text("{}", encoding="utf-8")
    (tmp_path / "escaped.json").write_text("{}", encoding="utf-8")

    assert (
        _bundle_relative_tune_results(
            "../escaped.json",
            tune_dir=tune_dir,
            private_root=private_root,
            final_dir=final_dir,
        )
        is None
    )


def _write_runtime_config(tmp_path: Path) -> Path:
    adapter = tmp_path / "adapter.py"
    adapter.write_text("print('adapter')\n", encoding="utf-8")
    config = {
        "schema_version": 1,
        "task": "gear_assembly",
        "runtime": {
            "python": sys.executable,
            "script": str(adapter),
            "cwd": str(tmp_path),
            "timeout_s": 60,
            "fingerprint_paths": [str(adapter)],
            "trial": {"target": 0.1},
        },
        "parameters": {"restitution": {"min": 0.0, "max": 1.0}},
        "objective": {
            "name": "grasp_slip",
            "unit": "m",
            "direction": "minimize",
            "failure_penalty": 1.0e12,
        },
        "optimizer": {"name": "random", "max_trials": 2, "seed": 7},
        "qualification": {"nominal_params": {"restitution": 0.5}, "seed": 1000},
        "evidence": {
            "artifact_name": "frames",
            "renderer": "isaac_sim_kit_rtx",
            "media_type": "application/json",
            "width": 320,
            "height": 240,
            "fps": 30,
            "min_frames": 2,
            "require_motion": True,
            "camera": {"position": [1.0, 1.0, 1.0], "target": [0.0, 0.0, 0.0]},
            "recording_artifact_name": "recording_usd",
            "playback_renderer": "ovrtx",
            "max_duration_seconds": 4.0,
            "num_sensor_updates": 8,
            "render_mode": "rt2",
        },
    }
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _fake_tune_runner(tune_input: Any) -> Any:
    output_dir = Path(tune_input.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if tune_input.approval_digest is None:
        qualification_path = output_dir / "qualification.json"
        qualification_path.write_text("{}", encoding="utf-8")
        qualification_frames = output_dir / "qualification_frames.json"
        qualification_frames.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            status="awaiting_approval",
            cancelled=False,
            error=None,
            qualification_digest=APPROVAL,
            qualification_path=qualification_path,
            artifacts={"qualification_frames": qualification_frames},
        )
    render_dir = output_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(2):
        frame = render_dir / f"frame_{index:04d}.png"
        frame.write_bytes(b"PNGDATA-%d" % index)
        frames.append(frame)
    (render_dir / "render_response_metadata.json").write_text(
        '{"renderer": "ovrtx"}', encoding="utf-8"
    )
    for name, content in (
        ("run_spec.json", "{}"),
        ("best_params.json", json.dumps({"restitution": 0.8})),
        ("history.jsonl", ""),
        # Absolute engine-run paths, like the real engine serializes: the
        # recording is published into the bundle, the approval artifact is
        # broker-private and deliberately not carried.
        (
            "external_tune_results.json",
            json.dumps(
                {
                    "artifacts": {
                        "best_recording": str(output_dir / "best_recording.usd"),
                        "approval": str(
                            output_dir.parent / "qualification" / "approval.json"
                        ),
                    }
                }
            ),
        ),
        ("best_recording.usd", "#usda 1.0\n"),
    ):
        (output_dir / name).write_text(content, encoding="utf-8")
    trial = SimpleNamespace(
        trial_index=0,
        params={"restitution": 0.8},
        optimizer_score=0.05,
        backend_metrics={},
        duration_seconds=1.0,
        failed=False,
        error=None,
        objective_value=0.05,
        success=True,
        replicas=[],
    )
    return SimpleNamespace(
        success=True,
        status="completed",
        cancelled=False,
        error=None,
        best_params={"restitution": 0.8},
        best_objective=0.05,
        best_score=0.05,
        n_trials=1,
        history=[trial],
        rendered_frames=frames,
        artifacts={"best_recording": output_dir / "best_recording.usd"},
        selected_evidence={
            "trial_index": 0,
            # The engine pins the published recording's digest; the broker
            # verifies its copy against exactly this value.
            "published_recording": {
                "path": "best_recording.usd",
                "sha256": "sha256:"
                + hashlib.sha256(
                    (output_dir / "best_recording.usd").read_bytes()
                ).hexdigest(),
            },
        },
    )


def _broker_url(kwargs: dict[str, Any]) -> str:
    # The sandbox allowlist must carry hostnames (runner.py and
    # claude_bridge.mjs push these verbatim into network.allowedDomains); a
    # full URL there silently blocks the loopback broker for --runner claude.
    hosts = kwargs.get("extra_allowed_hosts") or []
    assert hosts == ["127.0.0.1"], (
        "the sandbox allowlist must carry the broker hostname, not a URL"
    )
    match = re.search(r"http://127\.0\.0\.1:\d+", str(kwargs["prompt"]))
    assert match, "the child prompt must carry the full broker endpoint"
    return match.group(0)


def _run_one_sweep(broker_url: str, run_dir: Path) -> dict[str, Any]:
    response = requests.post(
        f"{broker_url}/sweeps",
        json={"output_dir": str(run_dir / "tuning" / "iter_1")},
        timeout=10.0,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    while payload["status"] == "running":
        payload = requests.get(
            f"{broker_url}/sweeps/{payload['sweep_id']}", timeout=10.0
        ).json()
    return payload


def _write_accept_artifacts(run_dir: Path, sweep: dict[str, Any]) -> None:
    selected = {
        "sweep_id": sweep["sweep_id"],
        "best_params": sweep["best_params"],
        "evidence_sha256": sweep["evidence_sha256"],
        "recording_sha256": sweep["recording_sha256"],
    }
    decision = {
        "schema_version": "content-agents.physics-external-tuning-decision.v1",
        "iteration": 1,
        "decision": "accept",
        "sweep_id": sweep["sweep_id"],
        "evidence_path": sweep["evidence_path"],
        "evidence_sha256": sweep["evidence_sha256"],
        "reviewed_frames": sweep["frames"],
        "selected": selected,
        "prior_decision_sha256": None,
        "rationale": "frames show the goal behavior",
    }
    decision_path = run_dir / "raw" / "physics_external_tuning_decision_1.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    result = {
        "schema_version": "content-agents.physics-external-tuning-result.v1",
        "status": "accepted",
        "selected": selected,
        "decision_paths": [str(decision_path)],
        "final_decision_sha256": sha256_file(decision_path),
        "rationale": "accepted after visual review",
    }
    (run_dir / "raw" / "physics_external_tuning_result.json").write_text(
        json.dumps(result), encoding="utf-8"
    )


def _accepting_child(**kwargs: Any) -> int:
    run_dir = kwargs["run_dir"]
    sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
    assert sweep["status"] == "succeeded"
    _write_accept_artifacts(run_dir, sweep)
    return 0


def _config(
    tmp_path: Path, *, approval_digest: str | None
) -> PhysicsExternalRefineConfig:
    return PhysicsExternalRefineConfig(
        repo_root=tmp_path,
        runtime_config=_write_runtime_config(tmp_path),
        user_prompt="keep the gear grasped",
        output_dir=tmp_path / "run",
        approval_digest=approval_digest,
        max_iterations=2,
        max_trials=2,
    )


@pytest.fixture(autouse=True)
def _skip_qualification_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    import physics_agent.tuning.external as external

    monkeypatch.setattr(
        external,
        "validate_qualification_approval",
        lambda spec, *, qualification_dir, approval_digest: qualification_dir
        / "qualification.json",
    )


def test_qualification_phase_pauses_for_approval(tmp_path: Path) -> None:
    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=None), tune_runner=_fake_tune_runner
    )
    assert result.status == "awaiting_approval"
    assert result.returncode == 0
    assert result.qualification_digest == APPROVAL
    summary = json.loads(
        (result.run_dir / "raw" / "external_qualification.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["qualification_digest"] == APPROVAL
    if sys.platform == "win32":
        pytest.skip("POSIX directory mode bits are unavailable on Windows")
    # The child session's raw-directory guard fails closed on any non-0700
    # raw/, and this runner has no _prepare_run_dir ahead of it: a plain
    # umask-inherited mkdir at the call site would regress the finalize
    # render probe.
    raw_mode = stat.S_IMODE((result.run_dir / "raw").stat().st_mode)
    assert raw_mode == 0o700


def test_accepted_run_publishes_final_bundle(tmp_path: Path) -> None:
    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=_accepting_child,
    )
    assert result.status == "accepted", result.reasons
    assert result.validated
    assert result.returncode == 0
    final_dir = result.final_dir
    assert final_dir is not None and final_dir.is_dir()
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    for name in (
        "best_params.json",
        "best_recording.usd",
        "external_tune_results.json",
        "physics_external_contract.json",
        "result.json",
        "evidence.json",
        "decisions/physics_external_tuning_decision_1.json",
    ):
        assert name in manifest
        published = final_dir / name
        assert published.is_file()
        assert sha256_file(published) == manifest[name]
    conclusion = json.loads(
        (result.run_dir / "raw" / "external_refine_conclusion.json").read_text(
            encoding="utf-8"
        )
    )
    assert conclusion["validated"] is True
    # Whole-phase ceiling transparency is durably recorded (no deadline was
    # configured here, so no overrun can be claimed).
    assert conclusion["phase_deadline_seconds"] is None
    assert conclusion["phase_deadline_exceeded"] is False
    assert (result.run_dir / "raw" / "external_sweep_ledger.jsonl").is_file()
    # The published tune results are a portable snapshot: broker-private
    # absolute paths are rewritten bundle-relative (or dropped when the
    # bundle deliberately does not carry the artifact).
    published_results = json.loads(
        (final_dir / "external_tune_results.json").read_text(encoding="utf-8")
    )
    assert published_results["artifacts"]["best_recording"] == "best_recording.usd"
    assert published_results["artifacts"]["approval"] is None
    # The portable path index resolves every absolute reference the
    # digest-bound copies carry to the bundle's own files.
    portable = json.loads(
        (final_dir / "portable_paths.json").read_text(encoding="utf-8")
    )
    paths = portable["paths"]
    decision_source = str(
        result.run_dir / "raw" / "physics_external_tuning_decision_1.json"
    )
    assert paths[decision_source] == (
        "decisions/physics_external_tuning_decision_1.json"
    )
    evidence = json.loads((final_dir / "evidence.json").read_text(encoding="utf-8"))
    for frame in evidence["frames"]:
        assert paths[frame["path"]] is not None
        assert paths[frame["path"]].startswith("render/")
    assert paths[evidence["recording_path"]] == "best_recording.usd"
    # The engine's selected-evidence recording descriptors resolve to the
    # published copy by digest instead of dangling after workspace release.
    published_descriptor = evidence["selected_evidence"]["published_recording"]
    assert paths[published_descriptor["path"]] == "best_recording.usd"
    # The decisions' cited evidence packet resolves to the bundle copy.
    published_decision = json.loads(
        (final_dir / "decisions" / "physics_external_tuning_decision_1.json").read_text(
            encoding="utf-8"
        )
    )
    assert paths[published_decision["evidence_path"]] == "evidence.json"


def test_tampered_final_decision_digest_fails_verification(tmp_path: Path) -> None:
    """result.final_decision_sha256 must match the last decision file; a
    mismatched digest converts the run to tool_failure."""

    def child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
        _write_accept_artifacts(run_dir, sweep)
        result_path = run_dir / "raw" / "physics_external_tuning_result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["final_decision_sha256"] = "a" * 64
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=child,
    )
    assert result.status == "tool_failure"
    assert any("final_decision_sha256" in reason for reason in result.reasons)


def test_unaccounted_second_sweep_blocks_acceptance(tmp_path: Path) -> None:
    """An accept for sweep 1 must not publish while a reserved sweep 2 is
    absent from the decision chain."""

    def child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        broker_url = _broker_url(kwargs)
        sweep = _run_one_sweep(broker_url, run_dir)
        _write_accept_artifacts(run_dir, sweep)
        # Reserve and finish a second sweep the chain never cites.
        response = requests.post(
            f"{broker_url}/sweeps",
            json={"output_dir": str(run_dir / "tuning" / "iter_2")},
            timeout=30.0,
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        while payload["status"] == "running":
            payload = requests.get(
                f"{broker_url}/sweeps/{payload['sweep_id']}", timeout=10.0
            ).json()
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=child,
    )
    assert result.status == "tool_failure"
    assert any("unaccounted" in reason for reason in result.reasons)


def test_revise_search_must_match_the_following_sweep(tmp_path: Path) -> None:
    """A revise_search that declares search A while the next sweep ran
    search B voids the chain: the narrative must match the broker record."""

    def child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        broker_url = _broker_url(kwargs)
        sweep_one = _run_one_sweep(broker_url, run_dir)
        revise = {
            "schema_version": ("content-agents.physics-external-tuning-decision.v1"),
            "iteration": 1,
            "decision": "revise_search",
            "sweep_id": sweep_one["sweep_id"],
            "evidence_sha256": sweep_one["evidence_sha256"],
            "next_active_search": {"restitution": {"min": 0.2, "max": 0.4}},
            "prior_decision_sha256": None,
            "rationale": "narrow the search",
        }
        first_path = run_dir / "raw" / "physics_external_tuning_decision_1.json"
        first_path.write_text(json.dumps(revise), encoding="utf-8")
        # Run the second sweep with a DIFFERENT search than declared.
        response = requests.post(
            f"{broker_url}/sweeps",
            json={
                "output_dir": str(run_dir / "tuning" / "iter_2"),
                "active_search": {"restitution": {"min": 0.5, "max": 0.9}},
            },
            timeout=30.0,
        )
        assert response.status_code == 200, response.text
        sweep_two = response.json()
        while sweep_two["status"] == "running":
            sweep_two = requests.get(
                f"{broker_url}/sweeps/{sweep_two['sweep_id']}", timeout=10.0
            ).json()
        selected = {
            "sweep_id": sweep_two["sweep_id"],
            "best_params": sweep_two["best_params"],
            "evidence_sha256": sweep_two["evidence_sha256"],
            "recording_sha256": sweep_two["recording_sha256"],
        }
        accept = {
            "schema_version": ("content-agents.physics-external-tuning-decision.v1"),
            "iteration": 2,
            "decision": "accept",
            "sweep_id": sweep_two["sweep_id"],
            "evidence_path": sweep_two["evidence_path"],
            "evidence_sha256": sweep_two["evidence_sha256"],
            "reviewed_frames": sweep_two["frames"],
            "selected": selected,
            "prior_decision_sha256": sha256_file(first_path),
            "rationale": "frames show the goal behavior",
        }
        second_path = run_dir / "raw" / "physics_external_tuning_decision_2.json"
        second_path.write_text(json.dumps(accept), encoding="utf-8")
        result = {
            "schema_version": ("content-agents.physics-external-tuning-result.v1"),
            "status": "accepted",
            "selected": selected,
            "decision_paths": [str(first_path), str(second_path)],
            "final_decision_sha256": sha256_file(second_path),
            "rationale": "accepted after visual review",
        }
        (run_dir / "raw" / "physics_external_tuning_result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=child,
    )
    assert result.status == "tool_failure"
    assert any("next_active_search" in reason for reason in result.reasons)


def test_rerun_archives_prior_final_deliverable(tmp_path: Path) -> None:
    """A rerun into the same --output-dir that concludes non-accepted must
    not destroy the previous accepted deliverable: after the broker released
    its workspace, final/ is the only copy, so it is archived to final.<n>/."""

    def silent_child(**_kwargs: Any) -> int:
        return 0

    config = _config(tmp_path, approval_digest=APPROVAL)
    prior_final = config.output_dir / "final"
    prior_final.mkdir(parents=True)
    (prior_final / "best_params.json").write_text(
        '{"restitution": 0.8}', encoding="utf-8"
    )

    result = run_physics_external_refine(
        config,
        tune_runner=_fake_tune_runner,
        child_agent_runner=silent_child,
    )
    assert result.status == "unresolved"
    assert result.final_dir is None
    run_dir = config.output_dir.expanduser().resolve()
    # The archive lives BESIDE the run directory, outside the child-writable
    # workspace, so a new session cannot corrupt or delete it.
    archived = run_dir.parent / f"{run_dir.name}.final.1" / "best_params.json"
    assert archived.read_text(encoding="utf-8") == '{"restitution": 0.8}'
    assert not prior_final.exists()


def test_failed_approval_validation_preserves_prior_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a validated run releases the broker workspace, final/ is the
    only copy of the accepted deliverable. A rerun that fails approval
    validation (typo'd digest, edited adapter) must fail BEFORE the stale-
    artifact wipe, leaving the previous deliverable untouched."""

    import physics_agent.tuning.external as external

    config = _config(tmp_path, approval_digest=APPROVAL)
    final_dir = config.output_dir / "final"
    final_dir.mkdir(parents=True)
    keepsake = final_dir / "best_params.json"
    keepsake.write_text('{"restitution": 0.8}', encoding="utf-8")
    prior_decision = (
        config.output_dir / "raw" / "physics_external_tuning_decision_1.json"
    )
    prior_decision.parent.mkdir(parents=True, exist_ok=True)
    prior_decision.write_text("{}", encoding="utf-8")

    def reject(*_args: Any, **_kwargs: Any) -> Path:
        raise ValueError("approval digest sha256:0000... does not match")

    monkeypatch.setattr(external, "validate_qualification_approval", reject)
    with pytest.raises(ValueError, match="does not match"):
        run_physics_external_refine(config, tune_runner=_fake_tune_runner)

    assert keepsake.read_text(encoding="utf-8") == '{"restitution": 0.8}'
    assert prior_decision.is_file()


def test_concurrent_runs_sharing_output_dir_are_refused(tmp_path: Path) -> None:
    """Two invocations sharing --output-dir would wipe each other's
    artifacts and race on result filenames; the second must be refused while
    the first holds the per-output-dir lock."""

    config = _config(tmp_path, approval_digest=APPROVAL)
    run_dir = config.output_dir.expanduser().resolve()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir.parent / f".{run_dir.name}.external-refine.lock"
    with lock_path.open("a+b") as held:
        with exclusive_descriptor_lock(held.fileno()):
            with pytest.raises(RuntimeError, match="already active"):
                run_physics_external_refine(config, tune_runner=_fake_tune_runner)


def test_accepted_result_must_enumerate_decision_paths(tmp_path: Path) -> None:
    """An accepted result with decision_paths omitted must fail verification:
    the published result.json has to identify the exact chain it summarizes."""

    def child_without_decision_paths(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
        assert sweep["status"] == "succeeded"
        _write_accept_artifacts(run_dir, sweep)
        result_path = run_dir / "raw" / "physics_external_tuning_result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["decision_paths"] = []
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=child_without_decision_paths,
    )
    assert result.status == "tool_failure"
    assert any("enumerate decision_paths" in reason for reason in result.reasons)


def test_child_runner_exception_concludes_tool_failure(tmp_path: Path) -> None:
    """An operational child-runner failure (timeout, provider startup) must
    conclude as tool_failure with durable artifacts, not escape as a bare
    exception that skips verification and the conclusion record."""

    def exploding_child(**_kwargs: Any) -> int:
        raise TimeoutError("child session timed out")

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=exploding_child,
    )
    assert result.status == "tool_failure"
    assert not result.validated
    assert result.returncode == 1
    assert any("child runner failed" in reason for reason in result.reasons)
    conclusion = json.loads(
        (result.run_dir / "raw" / "external_refine_conclusion.json").read_text(
            encoding="utf-8"
        )
    )
    assert conclusion["status"] == "tool_failure"


def test_honest_stop_publishes_nothing(tmp_path: Path) -> None:
    def stopping_child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
        decision = {
            "schema_version": "content-agents.physics-external-tuning-decision.v1",
            "iteration": 1,
            "decision": "stop",
            "sweep_id": sweep["sweep_id"],
            "evidence_sha256": sweep["evidence_sha256"],
            "prior_decision_sha256": None,
            "rationale": "objective cannot express the goal",
        }
        decision_path = run_dir / "raw" / "physics_external_tuning_decision_1.json"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")
        result = {
            "schema_version": "content-agents.physics-external-tuning-result.v1",
            "status": "stopped",
            "decision_paths": [str(decision_path)],
            "final_decision_sha256": sha256_file(decision_path),
            "rationale": "stopping honestly",
        }
        (run_dir / "raw" / "physics_external_tuning_result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=stopping_child,
    )
    assert result.status == "stopped"
    assert not result.validated
    assert result.returncode == 1
    assert result.final_dir is None
    assert not (result.run_dir / "final").exists()


def test_tampered_frame_fails_closed(tmp_path: Path) -> None:
    def tampering_child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
        _write_accept_artifacts(run_dir, sweep)
        # Tamper with a reviewed frame after writing the accept decision.
        Path(sweep["frames"][0]["path"]).write_bytes(b"tampered")
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=tampering_child,
    )
    assert result.status == "tool_failure"
    assert not result.validated
    assert result.returncode == 1
    assert any("reviewed frames" in reason for reason in result.reasons)
    assert not (result.run_dir / "final").exists()


def test_mutated_tune_artifact_fails_publication_closed(tmp_path: Path) -> None:
    """Core tune artifacts are digest-pinned when the sweep succeeds: a
    post-sweep mutation of the engine's best_params.json (e.g. a lingering
    BYOR descendant) must fail publication closed, not ship changed bytes in
    a validated bundle."""

    tune_dirs: list[Path] = []

    def capturing_tune_runner(tune_input: Any) -> Any:
        # The engine's tune directory lives in the broker-private
        # workspace; remember it so the child can mutate it post-sweep.
        tune_dirs.append(Path(tune_input.output_dir))
        return _fake_tune_runner(tune_input)

    def mutating_child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
        _write_accept_artifacts(run_dir, sweep)
        # Rewrite the engine's private tune output after the sweep succeeded.
        for tune_dir in tune_dirs:
            tune_params = tune_dir / "best_params.json"
            if tune_params.is_file():
                tune_params.write_text(
                    json.dumps({"restitution": 0.0}), encoding="utf-8"
                )
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=capturing_tune_runner,
        child_agent_runner=mutating_child,
    )
    assert result.status == "tool_failure"
    assert not result.validated
    assert any("tune artifact best_params.json" in reason for reason in result.reasons)
    assert result.final_dir is None


def _outputs_tune_runner(tune_input: Any) -> Any:
    """Fake engine runner that also promotes one declared winner output."""

    result = _fake_tune_runner(tune_input)
    if tune_input.approval_digest is None:
        return result
    artifact = Path(tune_input.output_dir) / "outputs" / "policy" / "policy.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"policy": true}', encoding="utf-8")
    result.published_outputs = {
        "policy": {
            "path": "outputs/policy/policy.json",
            "sha256": "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    }
    return result


def test_declared_outputs_publish_digest_verified(tmp_path: Path) -> None:
    """Declared winner outputs land in the final bundle, republished against
    the digests the engine pinned when it promoted them."""

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_outputs_tune_runner,
        child_agent_runner=_accepting_child,
    )
    assert result.status == "accepted", result.reasons
    final_dir = result.final_dir
    assert final_dir is not None
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "outputs/policy/policy.json" in manifest
    published = final_dir / "outputs" / "policy" / "policy.json"
    assert sha256_file(published) == manifest["outputs/policy/policy.json"]


def test_removed_declared_output_fails_publication_closed(tmp_path: Path) -> None:
    """Every declared winner output is required at publication: a lingering
    BYOR descendant deleting it after the sweep succeeds must fail closed
    rather than the bundle silently omitting it."""

    tune_dirs: list[Path] = []

    def capturing_tune_runner(tune_input: Any) -> Any:
        tune_dirs.append(Path(tune_input.output_dir))
        return _outputs_tune_runner(tune_input)

    def deleting_child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
        _write_accept_artifacts(run_dir, sweep)
        for tune_dir in tune_dirs:
            artifact = tune_dir / "outputs" / "policy" / "policy.json"
            if artifact.is_file():
                artifact.unlink()
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=capturing_tune_runner,
        child_agent_runner=deleting_child,
    )
    assert result.status == "tool_failure"
    assert not result.validated
    assert any("declared output" in reason for reason in result.reasons)
    assert result.final_dir is None


def test_result_status_must_match_the_terminal_decision(tmp_path: Path) -> None:
    """A chain that ends in accept must not conclude under a different
    terminal status: the published result would narrate a conclusion the
    decision chain never made."""

    def mismatching_child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
        _write_accept_artifacts(run_dir, sweep)
        result_path = run_dir / "raw" / "physics_external_tuning_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["status"] = "stopped"
        del result["selected"]  # stopped results must not carry a selection
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=mismatching_child,
    )
    assert result.status == "tool_failure"
    assert not result.validated
    assert any("ends in accept" in reason for reason in result.reasons)
    assert not (result.run_dir / "final").exists()


def test_identical_frames_keep_distinct_portable_paths(tmp_path: Path) -> None:
    """A fully settled rollout renders byte-identical frames; the portable
    path index must still resolve each source frame to its own published
    render/ copy, not collapse them onto the first digest match."""

    def settled_tune_runner(tune_input: Any) -> Any:
        result = _fake_tune_runner(tune_input)
        for frame in getattr(result, "rendered_frames", []) or []:
            frame.write_bytes(b"PNGDATA-settled")
        return result

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=settled_tune_runner,
        child_agent_runner=_accepting_child,
    )
    assert result.status == "accepted", result.reasons
    final_dir = result.final_dir
    assert final_dir is not None
    portable = json.loads(
        (final_dir / "portable_paths.json").read_text(encoding="utf-8")
    )
    paths = portable["paths"]
    evidence = json.loads((final_dir / "evidence.json").read_text(encoding="utf-8"))
    frames = evidence["frames"]
    assert len(frames) == 2
    assert {frame["sha256"] for frame in frames} == {frames[0]["sha256"]}
    for frame in frames:
        assert paths[frame["path"]] == f"render/{Path(frame['path']).name}"


def test_final_stop_may_conclude_budget_exhausted(tmp_path: Path) -> None:
    """The skill and prompt direct the agent to stop with a rationale when a
    reservation is refused; that refusal is budget exhaustion, so a final
    stop decision paired with budget_exhausted is an honest conclusion, not
    tool_failure."""

    def exhausted_child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
        decision = {
            "schema_version": "content-agents.physics-external-tuning-decision.v1",
            "iteration": 1,
            "decision": "stop",
            "sweep_id": sweep["sweep_id"],
            "evidence_sha256": sweep["evidence_sha256"],
            "prior_decision_sha256": None,
            "rationale": "next reservation was refused; budget is exhausted",
        }
        decision_path = run_dir / "raw" / "physics_external_tuning_decision_1.json"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")
        result = {
            "schema_version": "content-agents.physics-external-tuning-result.v1",
            "status": "budget_exhausted",
            "decision_paths": [str(decision_path)],
            "final_decision_sha256": sha256_file(decision_path),
            "rationale": "no sweeps remaining",
        }
        (run_dir / "raw" / "physics_external_tuning_result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=exhausted_child,
    )
    assert result.status == "budget_exhausted"
    assert not result.validated
    assert not any("ends in stop" in reason for reason in result.reasons)
    assert result.final_dir is None
    assert not (result.run_dir / "final").exists()


def test_unbound_revise_of_succeeded_sweep_does_not_void_the_run(
    tmp_path: Path,
) -> None:
    """The contract keeps evidence_sha256 optional for revise_search (a
    failed sweep never publishes evidence, and the model cannot know the
    sweep outcome at write time); an intermediate revise that omits the
    digest must not convert an otherwise fully verified accepted run into
    tool_failure at conclusion."""

    def child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        broker_url = _broker_url(kwargs)
        sweep_one = _run_one_sweep(broker_url, run_dir)
        next_search = {"restitution": {"min": 0.2, "max": 0.4}}
        revise = {
            "schema_version": "content-agents.physics-external-tuning-decision.v1",
            "iteration": 1,
            "decision": "revise_search",
            "sweep_id": sweep_one["sweep_id"],
            # Deliberately no evidence_path/evidence_sha256.
            "next_active_search": next_search,
            "prior_decision_sha256": None,
            "rationale": "narrow the search",
        }
        first_path = run_dir / "raw" / "physics_external_tuning_decision_1.json"
        first_path.write_text(json.dumps(revise), encoding="utf-8")
        response = requests.post(
            f"{broker_url}/sweeps",
            json={
                "output_dir": str(run_dir / "tuning" / "iter_2"),
                "active_search": next_search,
            },
            timeout=30.0,
        )
        assert response.status_code == 200, response.text
        sweep_two = response.json()
        while sweep_two["status"] == "running":
            sweep_two = requests.get(
                f"{broker_url}/sweeps/{sweep_two['sweep_id']}", timeout=10.0
            ).json()
        selected = {
            "sweep_id": sweep_two["sweep_id"],
            "best_params": sweep_two["best_params"],
            "evidence_sha256": sweep_two["evidence_sha256"],
            "recording_sha256": sweep_two["recording_sha256"],
        }
        accept = {
            "schema_version": "content-agents.physics-external-tuning-decision.v1",
            "iteration": 2,
            "decision": "accept",
            "sweep_id": sweep_two["sweep_id"],
            "evidence_path": sweep_two["evidence_path"],
            "evidence_sha256": sweep_two["evidence_sha256"],
            "reviewed_frames": sweep_two["frames"],
            "selected": selected,
            "prior_decision_sha256": sha256_file(first_path),
            "rationale": "frames show the goal behavior",
        }
        second_path = run_dir / "raw" / "physics_external_tuning_decision_2.json"
        second_path.write_text(json.dumps(accept), encoding="utf-8")
        result = {
            "schema_version": "content-agents.physics-external-tuning-result.v1",
            "status": "accepted",
            "selected": selected,
            "decision_paths": [str(first_path), str(second_path)],
            "final_decision_sha256": sha256_file(second_path),
            "rationale": "accepted after visual review",
        }
        (run_dir / "raw" / "physics_external_tuning_result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return 0

    config = _config(tmp_path, approval_digest=APPROVAL)
    result = run_physics_external_refine(
        config,
        tune_runner=_fake_tune_runner,
        child_agent_runner=child,
    )
    assert result.status == "accepted", result.reasons
    assert result.validated


def test_mutated_tune_results_fails_publication_closed(tmp_path: Path) -> None:
    """external_tune_results.json is documented output: its portable rewrite
    must start from bytes matching the digest pinned at sweep success, so a
    post-sweep rewrite (or deletion) fails publication closed."""

    tune_dirs: list[Path] = []

    def capturing_tune_runner(tune_input: Any) -> Any:
        tune_dirs.append(Path(tune_input.output_dir))
        return _fake_tune_runner(tune_input)

    def mutating_child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
        _write_accept_artifacts(run_dir, sweep)
        for tune_dir in tune_dirs:
            results = tune_dir / "external_tune_results.json"
            if results.is_file():
                results.write_text('{"artifacts": {}}', encoding="utf-8")
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=capturing_tune_runner,
        child_agent_runner=mutating_child,
    )
    assert result.status == "tool_failure"
    assert not result.validated
    assert any("external_tune_results.json" in reason for reason in result.reasons)
    assert result.final_dir is None


def test_tampered_operator_contract_voids_acceptance(tmp_path: Path) -> None:
    """physics_external_contract.json is the raw acceptance criterion the
    review is judged against; a child edit must void the accept rather than
    publish a result claiming to satisfy a falsified behavior goal."""

    def contract_editing_child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
        _write_accept_artifacts(run_dir, sweep)
        contract_path = run_dir / "raw" / "physics_external_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["behavior_goal"] = "anything at all"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=contract_editing_child,
    )
    assert result.status == "tool_failure"
    assert not result.validated
    assert any(
        "physics_external_contract.json changed" in reason for reason in result.reasons
    )
    assert result.final_dir is None


def test_missing_result_is_unresolved(tmp_path: Path) -> None:
    def silent_child(**kwargs: Any) -> int:
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=silent_child,
    )
    assert result.status == "unresolved"
    assert result.returncode == 1


def test_nonzero_child_without_result_is_tool_failure(tmp_path: Path) -> None:
    def crashing_child(**kwargs: Any) -> int:
        return 3

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=crashing_child,
    )
    assert result.status == "tool_failure"
    assert result.returncode == 1


def test_whitespace_only_user_prompt_is_rejected(tmp_path: Path) -> None:
    """A blank behavior goal would let the workflow accept and publish a
    result with no meaningful acceptance criterion at all."""

    from dataclasses import replace

    config = replace(_config(tmp_path, approval_digest=APPROVAL), user_prompt="   ")
    with pytest.raises(ValueError, match="behavior goal"):
        run_physics_external_refine(config, tune_runner=_fake_tune_runner)
    # Fail-fast: nothing was staged or wiped.
    assert not (tmp_path / "run").exists()


def test_video_reference_is_rejected_before_external_run_cleanup(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    run_dir = tmp_path / "run"
    final_marker = run_dir / "final/keep.txt"
    final_marker.parent.mkdir(parents=True)
    final_marker.write_text("keep", encoding="utf-8")
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"video")
    config = replace(
        _config(tmp_path, approval_digest=APPROVAL), reference_images=[video]
    )

    with pytest.raises(ValueError, match="Unsupported reference image extension"):
        run_physics_external_refine(config, tune_runner=_fake_tune_runner)

    assert final_marker.read_text(encoding="utf-8") == "keep"


def test_reference_media_is_digest_bound_and_published(tmp_path: Path) -> None:
    """Staged reference media are the review's comparison inputs: their
    digests are taken before the child launches and the exact reviewed copies
    are published with the accepted bundle."""

    from dataclasses import replace

    reference = tmp_path / "goal.png"
    reference.write_bytes(b"REFPNG")
    config = replace(
        _config(tmp_path, approval_digest=APPROVAL), reference_images=[reference]
    )
    result = run_physics_external_refine(
        config, tune_runner=_fake_tune_runner, child_agent_runner=_accepting_child
    )
    assert result.status == "accepted", result.reasons
    final_dir = result.final_dir
    assert final_dir is not None
    published = final_dir / "reference_media" / "image_01_goal.png"
    assert published.read_bytes() == b"REFPNG"
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reference_media/image_01_goal.png"] == sha256_file(published)


def test_tampered_reference_media_voids_acceptance(tmp_path: Path) -> None:
    """A child that rewrites the staged reference media judged against
    different comparison inputs than the operator supplied; the accept must
    not survive."""

    from dataclasses import replace

    reference = tmp_path / "goal.png"
    reference.write_bytes(b"REFPNG")

    def tampering_child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        staged = run_dir / "reference_media" / "image_01_goal.png"
        assert staged.is_file()
        staged.write_bytes(b"SWAPPED")
        sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
        _write_accept_artifacts(run_dir, sweep)
        return 0

    config = replace(
        _config(tmp_path, approval_digest=APPROVAL), reference_images=[reference]
    )
    result = run_physics_external_refine(
        config, tune_runner=_fake_tune_runner, child_agent_runner=tampering_child
    )
    assert result.status == "tool_failure"
    assert not result.validated
    assert result.final_dir is None
    assert any("reference media" in reason for reason in result.reasons)


def test_skipped_workspace_release_is_recorded_in_conclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a lingering BYOR runtime keeps the broker workspace alive past
    close(), a validated run must record the leaked path in the conclusion
    instead of reporting the workspace as gone."""

    from content_workflow_cli.external_tuning_broker import ExternalTuningBroker

    monkeypatch.setattr(ExternalTuningBroker, "release_private_dir", lambda self: False)
    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=_accepting_child,
    )
    assert result.status == "accepted", result.reasons
    conclusion = json.loads(
        (result.run_dir / "raw" / "external_refine_conclusion.json").read_text(
            encoding="utf-8"
        )
    )
    assert conclusion["validated"] is True
    leaked = conclusion["broker_private_dir"]
    assert leaked is not None and Path(leaked).is_dir()


def test_final_render_evidence_is_published_from_verified_copies(
    tmp_path: Path,
) -> None:
    """final/render/ carries the broker's digest-verified evidence copies,
    not files from the engine's mutable tune directory."""

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=_accepting_child,
    )
    assert result.status == "accepted", result.reasons
    final_dir = result.final_dir
    assert final_dir is not None
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    evidence = json.loads((final_dir / "evidence.json").read_text(encoding="utf-8"))
    published_frames = sorted(
        name for name in manifest if name.startswith("render/frame_")
    )
    assert published_frames
    assert sorted(frame["sha256"] for frame in evidence["frames"]) == sorted(
        manifest[name] for name in published_frames
    )
    assert "render/render_response_metadata.json" in manifest


def test_failed_child_session_cannot_conclude_accepted(tmp_path: Path) -> None:
    """A result written before the session died (timeout, crash, nonzero
    exit) may predate work the session was still doing; it must not be
    verified and published as a clean acceptance."""

    def accepting_then_failing_child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
        _write_accept_artifacts(run_dir, sweep)
        return 3

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=accepting_then_failing_child,
    )
    assert result.status == "tool_failure"
    assert not result.validated
    assert result.final_dir is None
    assert not (result.run_dir / "final").exists()
    assert any("failed session" in reason for reason in result.reasons)


def test_remote_playback_renderer_is_refused_before_qualification(
    tmp_path: Path,
) -> None:
    """The broker rejects unattested render provenance at evidence
    publication; the operator must be refused at config load instead of
    after a full sweep of customer-simulator wall clock."""

    # Edit the runtime config after _config() writes it (same path).
    refine_config = _config(tmp_path, approval_digest=None)
    runtime_config = refine_config.runtime_config
    config = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    config["evidence"]["playback_renderer"] = "remote"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="OVRTX"):
        run_physics_external_refine(refine_config, tune_runner=_fake_tune_runner)
    # Refused before qualification ran.
    assert not (tmp_path / "run" / "qualification").exists()


def test_bare_stop_after_a_sweep_is_an_invalid_audit_artifact(
    tmp_path: Path,
) -> None:
    """A child that completes a sweep and then writes only a bare stopped
    result hides executed work; results summarizing executed sweeps must
    bind the decision chain."""

    def bare_stopping_child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        _run_one_sweep(_broker_url(kwargs), run_dir)
        result = {
            "schema_version": "content-agents.physics-external-tuning-result.v1",
            "status": "stopped",
            "rationale": "stopping without accounting for the sweep",
        }
        (run_dir / "raw" / "physics_external_tuning_result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=bare_stopping_child,
    )
    assert result.status == "tool_failure"
    assert any("final_decision_sha256" in reason for reason in result.reasons)


def test_bare_stop_with_zero_sweeps_is_honest(tmp_path: Path) -> None:
    """A refusal before any sweep is reserved has no executed work to bind;
    the bare terminal status stands."""

    def refusing_child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        result = {
            "schema_version": "content-agents.physics-external-tuning-result.v1",
            "status": "stopped",
            "rationale": "objective cannot express the goal",
        }
        (run_dir / "raw" / "physics_external_tuning_result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=refusing_child,
    )
    assert result.status == "stopped"
    assert result.final_dir is None


def test_published_history_is_portable(tmp_path: Path) -> None:
    """final/history.jsonl is regenerated from the verified evidence packet;
    it must not carry broker-private replica paths that dangle once the
    workspace is released."""

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=_accepting_child,
    )
    assert result.status == "accepted", result.reasons
    final_dir = result.final_dir
    assert final_dir is not None
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "history.jsonl" in manifest
    history_lines = [
        json.loads(line)
        for line in (final_dir / "history.jsonl")
        .read_text(encoding="utf-8")
        .split("\n")
        if line
    ]
    evidence = json.loads((final_dir / "evidence.json").read_text(encoding="utf-8"))
    assert history_lines == evidence["history"]
    assert "physics-external-tuning-broker" not in (
        final_dir / "history.jsonl"
    ).read_text(encoding="utf-8")


def test_stop_citing_a_succeeded_sweep_without_digest_is_honest(
    tmp_path: Path,
) -> None:
    """The contract keeps evidence_sha256 optional for stop decisions; a
    stop that names the succeeded sweep it learned from but omits the digest
    must conclude as the honest stop it is, not tool_failure."""

    def stopping_child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
        assert sweep["status"] == "succeeded"
        decision = {
            "schema_version": "content-agents.physics-external-tuning-decision.v1",
            "iteration": 1,
            "decision": "stop",
            "sweep_id": sweep["sweep_id"],
            "prior_decision_sha256": None,
            "rationale": "the objective plateaued; stopping",
        }
        decision_path = run_dir / "raw" / "physics_external_tuning_decision_1.json"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")
        result = {
            "schema_version": "content-agents.physics-external-tuning-result.v1",
            "status": "stopped",
            "decision_paths": [str(decision_path)],
            "final_decision_sha256": sha256_file(decision_path),
            "rationale": "stopping honestly",
        }
        (run_dir / "raw" / "physics_external_tuning_result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return 0

    result = run_physics_external_refine(
        _config(tmp_path, approval_digest=APPROVAL),
        tune_runner=_fake_tune_runner,
        child_agent_runner=stopping_child,
    )
    assert result.status == "stopped", result.reasons
    assert not result.validated
    assert result.final_dir is None


def test_rerun_clears_stale_reference_media(tmp_path: Path) -> None:
    """A rerun with different reference arguments must not leave the prior
    run's staged media visible to the child: they would be absent from the
    current digest set and final bundle."""

    from dataclasses import replace

    first_reference = tmp_path / "old_goal.png"
    first_reference.write_bytes(b"OLDREF")
    config = replace(
        _config(tmp_path, approval_digest=APPROVAL),
        reference_images=[first_reference],
    )
    result = run_physics_external_refine(
        config, tune_runner=_fake_tune_runner, child_agent_runner=_accepting_child
    )
    assert result.status == "accepted", result.reasons
    staged_old = result.run_dir / "reference_media" / "image_01_old_goal.png"
    assert staged_old.is_file()

    second_reference = tmp_path / "new_goal.png"
    second_reference.write_bytes(b"NEWREF")

    def asserting_child(**kwargs: Any) -> int:
        run_dir = kwargs["run_dir"]
        staged = sorted(path.name for path in (run_dir / "reference_media").iterdir())
        assert staged == ["image_01_new_goal.png"], staged
        sweep = _run_one_sweep(_broker_url(kwargs), run_dir)
        _write_accept_artifacts(run_dir, sweep)
        return 0

    config = replace(
        _config(tmp_path, approval_digest=APPROVAL),
        reference_images=[second_reference],
    )
    result = run_physics_external_refine(
        config, tune_runner=_fake_tune_runner, child_agent_runner=asserting_child
    )
    assert result.status == "accepted", result.reasons
    assert not (result.run_dir / "reference_media" / "image_01_old_goal.png").exists()
