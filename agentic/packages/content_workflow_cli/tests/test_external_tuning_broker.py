# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""External tuning broker tests: budget, sequencing, pinned-parameter
carryover, evidence publication with digests, and claim verification."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import content_workflow_cli.external_tuning_broker as external_tuning_broker_module
from content_workflow_cli.external_tuning_broker import (
    ExternalTuningBroker,
    sha256_file,
)
from content_workflow_cli.tuning_broker import BrokerError


def _make_spec(tmp_path: Path) -> Any:
    from physics_agent.tuning.external import (
        ExternalRuntime,
        ExternalTuneSpec,
        QualificationSettings,
        QualifiedParameter,
    )
    from physics_agent.tuning.external.types import EvidenceSettings
    from physics_agent.tuning.types import (
        OptimizerSettings,
        TunableParam,
        TuningObjective,
    )

    script = tmp_path / "adapter.py"
    script.write_text("print('adapter')\n", encoding="utf-8")
    runtime = ExternalRuntime(
        python=Path(sys.executable),
        script=script,
        cwd=tmp_path,
        fingerprint_paths=(script,),
    )
    return ExternalTuneSpec(
        task="gear_assembly",
        runtime=runtime,
        parameter_catalog=(
            QualifiedParameter(name="restitution"),
            QualifiedParameter(name="dynamic_friction"),
        ),
        params=(TunableParam(name="restitution", min_value=0.0, max_value=1.0),),
        objective=TuningObjective(name="grasp_slip", unit="m"),
        optimizer=OptimizerSettings(name="random", max_trials=4, seed=7),
        qualification=QualificationSettings(
            nominal_params={"restitution": 0.5, "dynamic_friction": 0.6}
        ),
        evidence=EvidenceSettings(),
    )


def _fake_trial(index: int, objective: float) -> SimpleNamespace:
    return SimpleNamespace(
        trial_index=index,
        params={"restitution": 0.1 * (index + 1)},
        optimizer_score=objective,
        backend_metrics={},
        duration_seconds=1.0,
        failed=False,
        error=None,
        objective_value=objective,
        success=True,
        replicas=[],
    )


def _fake_external_runner(
    *,
    best_params: dict[str, float] | None = None,
    status: str = "completed",
    success: bool = True,
    frames: int = 3,
    identical_frames: int = 2,
    write_recording: bool = True,
    tamper_recording: bool = False,
    write_tune_files: bool = True,
    drop_frame: bool = False,
    seen_inputs: list[Any] | None = None,
) -> Any:
    def runner(tune_input: Any) -> Any:
        if seen_inputs is not None:
            seen_inputs.append(tune_input)
        output_dir = Path(tune_input.output_dir)
        render_dir = output_dir / "render"
        render_dir.mkdir(parents=True, exist_ok=True)
        rendered: list[Path] = []
        for index in range(frames):
            frame = render_dir / f"frame_{index:04d}.png"
            # The first `identical_frames` frames share bytes to exercise
            # distinct-frame deduplication.
            content = (
                b"PNGDATA-same" if index < identical_frames else (b"PNGDATA-%d" % index)
            )
            frame.write_bytes(content)
            rendered.append(frame)
        if drop_frame and rendered:
            # Simulate a listed frame disappearing before publication.
            rendered[0].unlink()
        (render_dir / "render_response_metadata.json").write_text(
            '{"renderer": "ovrtx"}', encoding="utf-8"
        )
        artifacts: dict[str, Any] = {}
        selected_evidence: dict[str, Any] = {"trial_index": 1, "replica_seed": 7}
        if write_recording:
            recording = output_dir / "best_recording.usd"
            recording.write_text("#usda 1.0\n", encoding="utf-8")
            artifacts["best_recording"] = recording
            # The engine pins the published recording's digest; the broker
            # verifies its copy against exactly this value.
            selected_evidence["published_recording"] = {
                "path": "best_recording.usd",
                "sha256": "sha256:"
                + hashlib.sha256(recording.read_bytes()).hexdigest(),
            }
            if tamper_recording:
                # Simulate a lingering BYOR descendant rewriting the
                # recording after the engine pinned its digest.
                recording.write_text("#usda 1.0\n# tampered\n", encoding="utf-8")
        params = dict(best_params or {"restitution": 0.8})
        if write_tune_files:
            # Core tune artifacts are mandatory for a succeeded sweep: the
            # broker pins their digests when publishing evidence.
            (output_dir / "run_spec.json").write_text("{}", encoding="utf-8")
            (output_dir / "best_params.json").write_text(
                json.dumps(params), encoding="utf-8"
            )
            (output_dir / "external_tune_results.json").write_text(
                json.dumps({"artifacts": {}}), encoding="utf-8"
            )
        return SimpleNamespace(
            success=success,
            status=status,
            cancelled=False,
            error=None if success else "external tune failed",
            best_params=params,
            best_objective=0.05,
            best_score=0.05,
            n_trials=2,
            history=[_fake_trial(0, 0.4), _fake_trial(1, 0.05)],
            rendered_frames=rendered,
            artifacts=artifacts,
            selected_evidence=selected_evidence,
        )

    return runner


def _start_broker(
    tmp_path: Path, runner: Any, **overrides: Any
) -> ExternalTuningBroker:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "run_dir": run_dir,
        "spec": _make_spec(tmp_path),
        "qualification_dir": run_dir / "qualification",
        "approval_digest": "sha256:" + "0" * 64,
        "max_sweeps": 3,
        "max_trials_per_sweep": 4,
        "private_dir": tmp_path / "private",
        "tune_runner": runner,
    }
    kwargs.update(overrides)
    broker = ExternalTuningBroker(**kwargs)
    return broker


def _wait_terminal(
    broker: ExternalTuningBroker, sweep_id: str, timeout: float = 10.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        view = broker.sweep_view(sweep_id)
        if view["status"] != "running":
            return view
        time.sleep(0.02)
    raise AssertionError(f"sweep {sweep_id} never reached a terminal status")


def test_successful_sweep_publishes_digest_bound_evidence(tmp_path: Path) -> None:
    seen: list[Any] = []
    broker = _start_broker(tmp_path, _fake_external_runner(seen_inputs=seen))
    iter_dir = broker.run_dir / "tuning" / "iter_1"
    view = broker.request_sweep({"output_dir": str(iter_dir)})
    view = _wait_terminal(broker, view["sweep_id"])

    assert view["status"] == "succeeded"
    assert view["iteration"] == 1
    # First sweep pins the qualified nominal parameters.
    assert view["pinned_params"] == {"restitution": 0.5, "dynamic_friction": 0.6}
    # Default active search comes from the runtime config's declared search.
    assert view["active_search"] == {"restitution": {"min": 0.0, "max": 1.0}}
    # The engine received the approval digest, qualification dir, and pins.
    tune_input = seen[0]
    assert tune_input.approval_digest == "sha256:" + "0" * 64
    assert tune_input.fixed_params == view["pinned_params"]
    assert tune_input.render_winning_trial is True

    evidence_path = Path(view["evidence_path"])
    assert evidence_path.is_file()
    assert view["evidence_sha256"] == sha256_file(evidence_path)
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert payload["best_params"] == {"restitution": 0.8}
    assert payload["objective"]["name"] == "grasp_slip"
    # Frames are copied into the child-readable iteration dir with digests;
    # byte-identical frames collapse in distinct_frames.
    assert len(view["frames"]) == 3
    assert len(view["distinct_frames"]) == 2
    for frame in view["frames"]:
        frame_path = Path(frame["path"])
        assert frame_path.is_file()
        assert iter_dir in frame_path.parents
        assert sha256_file(frame_path) == frame["sha256"]
    assert view["recording_sha256"] is not None
    assert Path(view["recording_path"]).is_file()

    ok, reason = broker.verify_claim(
        {"sweep_id": view["sweep_id"], "evidence_sha256": view["evidence_sha256"]}
    )
    assert ok, reason
    ok, _ = broker.verify_reviewed_frames(view["sweep_id"], view["frames"][:1])
    assert ok
    # The packet carries reproducibility provenance for the frames: the
    # renderer identity/settings bound to the exact source recording digest.
    provenance = payload["render_provenance"]
    assert provenance["renderer"] == "ovrtx"
    assert provenance["source_recording_sha256"] == view["recording_sha256"]
    broker.close()


def test_sweep_without_winner_recording_fails_closed(tmp_path: Path) -> None:
    """Frames are rendered from the winner recording; publishing them without
    that recording would make the evidence non-reproducible, so the broker
    fails the sweep instead of degrading."""

    broker = _start_broker(tmp_path, _fake_external_runner(write_recording=False))
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    view = _wait_terminal(broker, view["sweep_id"])
    assert view["status"] == "failed"
    assert "recording" in (view["error"] or "")
    broker.close()


def test_broker_preconditions_bind_the_engine_contract(tmp_path: Path) -> None:
    """The broker's mandatory evidence preconditions (core tune filenames,
    render-metadata sidecar name, descriptor shape, selected-evidence keys)
    are engine-produced artifacts; binding them to the engine's own symbols
    makes an engine-side rename fail here, in unit tests, instead of hours
    into a customer-simulator run."""

    import inspect

    from physics_agent.tuning.external import runner as engine_runner
    from physics_agent.tuning.external.artifacts import (
        BEST_PARAMS,
        BEST_RECORDING,
        RESULTS,
        RUN_SPEC,
        file_descriptor,
    )
    from world_understanding.functions.graphics.render_time_sampled_usd import (
        RENDER_RESPONSE_METADATA_FILENAME,
    )

    from content_workflow_cli.external_tuning_broker import CORE_TUNE_ARTIFACT_NAMES

    # Core tune artifacts the broker pins at success are the engine's own
    # canonical filenames, and the recording/sidecar names the broker
    # requires are the ones the engine writes.
    assert CORE_TUNE_ARTIFACT_NAMES == (RUN_SPEC, BEST_PARAMS, RESULTS)
    assert BEST_RECORDING == "best_recording.usd"
    assert RENDER_RESPONSE_METADATA_FILENAME == "render_response_metadata.json"

    # Descriptor shape the broker/publication parse: a relative path plus a
    # "sha256:"-prefixed digest.
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"bytes")
    descriptor = file_descriptor(artifact, relative_path="outputs/x/artifact.bin")
    assert descriptor["path"] == "outputs/x/artifact.bin"
    assert descriptor["sha256"].startswith("sha256:")
    assert len(descriptor["sha256"].removeprefix("sha256:")) == 64

    # selected_evidence keys the broker verifies and publication indexes are
    # literals in the engine's winner-selection code.
    engine_source = inspect.getsource(engine_runner)
    assert '"published_recording"' in engine_source
    assert '"scored_recording"' in engine_source


def test_recording_rewritten_after_engine_pinning_fails_closed(
    tmp_path: Path,
) -> None:
    """The engine pins the winning recording's digest when it validates and
    publishes best_recording.usd; a rewrite between engine return and the
    broker's evidence copy must fail the sweep, or the accepted bundle could
    carry a recording that never produced the reviewed frames."""

    broker = _start_broker(tmp_path, _fake_external_runner(tamper_recording=True))
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    view = _wait_terminal(broker, view["sweep_id"])
    assert view["status"] == "failed"
    assert "engine-pinned" in (view["error"] or "")
    broker.close()


def test_missing_rendered_frame_fails_closed(tmp_path: Path) -> None:
    """Every engine-returned frame is part of the winner rollout: publishing
    the remainder would present truncated motion as complete review evidence,
    so a listed frame missing at publication fails the sweep."""

    broker = _start_broker(tmp_path, _fake_external_runner(drop_frame=True))
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    view = _wait_terminal(broker, view["sweep_id"])
    assert view["status"] == "failed"
    assert "frame is missing" in (view["error"] or "")
    broker.close()


def test_missing_core_tune_artifact_fails_closed(tmp_path: Path) -> None:
    """A sweep that cannot pin run_spec.json/best_params.json must not become
    succeeded: final publication publishes exactly the pinned set, so a
    missing core artifact would otherwise ship an accepted-but-incomplete
    bundle."""

    broker = _start_broker(tmp_path, _fake_external_runner(write_tune_files=False))
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    view = _wait_terminal(broker, view["sweep_id"])
    assert view["status"] == "failed"
    assert "core tune artifact" in (view["error"] or "")
    broker.close()


def test_pre_planted_evidence_symlink_cannot_redirect_publication(
    tmp_path: Path,
) -> None:
    """A child that knows the sweep_id could pre-create the evidence dir (or
    its ``frames`` subdir) as a symlink while the sweep runs; the no-follow
    publication path must refuse it rather than write through the link."""

    outside = tmp_path / "outside"
    outside.mkdir()
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Windows symlink creation is unavailable: {exc}")
    else:
        probe.unlink()
    planted = threading.Event()

    def planting_runner(tune_input: Any) -> Any:
        # Block the engine until the test has planted the link, so the
        # symlink deterministically exists before publication starts.
        assert planted.wait(timeout=30)
        return _fake_external_runner()(tune_input)

    broker = _start_broker(tmp_path, planting_runner)
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    sweep_id = view["sweep_id"]
    iter_dir = broker.run_dir / "tuning" / "iter_1"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / f"evidence-{sweep_id}").symlink_to(outside)
    planted.set()
    view = _wait_terminal(broker, sweep_id)
    assert view["status"] == "failed"
    assert not any(outside.iterdir()), "publication must not follow the link"
    broker.close()


def test_pre_planted_frames_symlink_cannot_redirect_publication(
    tmp_path: Path,
) -> None:
    """Rejecting only the evidence root is not enough: a child can create the
    evidence directory itself with ``frames`` pre-planted as a symlink, and
    the frame copies must still refuse to write through it."""

    outside = tmp_path / "outside"
    outside.mkdir()
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Windows symlink creation is unavailable: {exc}")
    else:
        probe.unlink()
    planted = threading.Event()

    def planting_runner(tune_input: Any) -> Any:
        assert planted.wait(timeout=30)
        return _fake_external_runner()(tune_input)

    broker = _start_broker(tmp_path, planting_runner)
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    sweep_id = view["sweep_id"]
    evidence_dir = broker.run_dir / "tuning" / "iter_1" / f"evidence-{sweep_id}"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "frames").symlink_to(outside)
    planted.set()
    view = _wait_terminal(broker, sweep_id)
    assert view["status"] == "failed"
    assert not any(outside.iterdir()), "publication must not follow the link"
    broker.close()


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows reparse APIs")
def test_windows_evidence_link_swap_after_validation_cannot_redirect_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reparse swap after path validation must lose the exclusive NT create."""

    outside = tmp_path / "outside"
    outside.mkdir()
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Windows symlink creation is unavailable: {exc}")
    else:
        probe.unlink()

    real_reject = external_tuning_broker_module._reject_child_output_links

    def swap_after_validation(run_dir: Path, output_dir: Path) -> Path:
        checked = real_reject(run_dir, output_dir)
        if checked.name.startswith("evidence-external-sweep-"):
            checked.symlink_to(outside, target_is_directory=True)
        return checked

    monkeypatch.setattr(
        external_tuning_broker_module,
        "_reject_child_output_links",
        swap_after_validation,
    )
    broker = _start_broker(tmp_path, _fake_external_runner())
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    view = _wait_terminal(broker, view["sweep_id"])

    assert view["status"] == "failed"
    assert "evidence publication failed" in (view["error"] or "")
    assert "O_DIRECTORY" not in (view["error"] or "")
    assert not any(outside.iterdir()), "publication must not follow the swapped link"
    broker.close()


def test_pinned_params_advance_on_success(tmp_path: Path) -> None:
    seen: list[Any] = []
    broker = _start_broker(
        tmp_path,
        _fake_external_runner(best_params={"restitution": 0.8}, seen_inputs=seen),
    )
    first = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    _wait_terminal(broker, first["sweep_id"])
    second = broker.request_sweep(
        {
            "output_dir": str(broker.run_dir / "tuning" / "iter_2"),
            "active_search": {"dynamic_friction": {"min": 0.1, "max": 0.9}},
        }
    )
    view = _wait_terminal(broker, second["sweep_id"])
    # The second sweep pins the first sweep's winner for restitution.
    assert view["pinned_params"]["restitution"] == 0.8
    assert view["active_search"] == {"dynamic_friction": {"min": 0.1, "max": 0.9}}
    assert seen[1].fixed_params["restitution"] == 0.8
    broker.close()


def test_failed_sweep_keeps_pins_and_records_error(tmp_path: Path) -> None:
    broker = _start_broker(
        tmp_path,
        _fake_external_runner(status="optimization_failed", success=False),
    )
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    view = _wait_terminal(broker, view["sweep_id"])
    assert view["status"] == "failed"
    assert view["error"] == "external tune failed"
    assert view["engine_status"] == "optimization_failed"
    second = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_2")}
    )
    assert second["pinned_params"] == {
        "restitution": 0.5,
        "dynamic_friction": 0.6,
    }
    broker.close()


def test_success_without_frames_fails_closed(tmp_path: Path) -> None:
    broker = _start_broker(tmp_path, _fake_external_runner(frames=0))
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    view = _wait_terminal(broker, view["sweep_id"])
    assert view["status"] == "failed"
    assert "rendered winner frames" in (view["error"] or "")
    broker.close()


def test_sweeps_are_sequential(tmp_path: Path) -> None:
    release = threading.Event()

    def blocking_runner(tune_input: Any) -> Any:
        release.wait(timeout=10.0)
        return _fake_external_runner()(tune_input)

    broker = _start_broker(tmp_path, blocking_runner)
    first = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    with pytest.raises(BrokerError, match="sequential") as excinfo:
        broker.request_sweep({"output_dir": str(broker.run_dir / "tuning" / "iter_2")})
    # 503, not 409: the refusal is transient, and the sweep client must map
    # it to retry-later guidance rather than terminal budget exhaustion.
    assert excinfo.value.status == 503
    release.set()
    _wait_terminal(broker, first["sweep_id"])
    broker.close()


def test_sweeps_use_a_private_qualification_copy(tmp_path: Path) -> None:
    """The engine writes an approval audit record into qualification_dir on
    every sweep with symlink-following opens; the broker must therefore hand
    the engine a broker-private copy, never the child-writable run-dir path,
    or a pre-planted approval.json.tmp link redirects that unsandboxed
    write."""

    run_dir = tmp_path / "run"
    qualification_dir = run_dir / "qualification"
    qualification_dir.mkdir(parents=True)
    (qualification_dir / "qualification.json").write_text("{}", encoding="utf-8")
    seen: list[Any] = []
    broker = _start_broker(tmp_path, _fake_external_runner(seen_inputs=seen))
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    _wait_terminal(broker, view["sweep_id"])
    engine_qualification = Path(seen[0].qualification_dir)
    assert not engine_qualification.is_relative_to(broker.run_dir)
    assert engine_qualification.is_relative_to(broker.private_dir)
    assert (engine_qualification / "qualification.json").is_file()
    broker.close()


def test_spec_declared_trial_ceiling_caps_the_sweep(tmp_path: Path) -> None:
    """The runtime config's optimizer.max_trials is a ceiling the broker
    caps against, never replaces: a BYOR spec declaring few trials (each
    launches a customer simulator) must not silently run more under the
    wrapper's larger per-sweep budget."""

    seen: list[Any] = []
    broker = _start_broker(
        tmp_path,
        _fake_external_runner(seen_inputs=seen),
        max_trials_per_sweep=9,
    )
    view = broker.request_sweep(
        {
            "output_dir": str(broker.run_dir / "tuning" / "iter_1"),
            "max_trials": 9,
        }
    )
    view = _wait_terminal(broker, view["sweep_id"])
    assert view["status"] == "succeeded"
    # The spec fixture declares optimizer.max_trials == 4.
    assert view["max_trials"] == 4
    assert seen[0].config.optimizer.max_trials == 4
    broker.close()


def test_budget_exhaustion(tmp_path: Path) -> None:
    broker = _start_broker(tmp_path, _fake_external_runner(), max_sweeps=1)
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    _wait_terminal(broker, view["sweep_id"])
    with pytest.raises(BrokerError, match="budget exhausted"):
        broker.request_sweep({"output_dir": str(broker.run_dir / "tuning" / "iter_2")})
    broker.close()


def test_active_search_validation(tmp_path: Path) -> None:
    broker = _start_broker(tmp_path, _fake_external_runner())
    with pytest.raises(BrokerError, match="'min' and 'max'"):
        broker.request_sweep(
            {
                "output_dir": str(broker.run_dir / "tuning" / "iter_1"),
                "active_search": {"restitution": {"min": 0.1}},
            }
        )
    with pytest.raises(BrokerError, match="output_dir is required"):
        broker.request_sweep({})
    # Catalog membership and bound ordering are rejected synchronously,
    # BEFORE any budget is reserved: a typo'd parameter name or inverted
    # bounds must not permanently consume one of the few sweeps.
    with pytest.raises(BrokerError, match="not in the .* catalog"):
        broker.request_sweep(
            {
                "output_dir": str(broker.run_dir / "tuning" / "iter_1"),
                "active_search": {"unknown": {"min": 0.0, "max": 1.0}},
            }
        )
    with pytest.raises(BrokerError, match="min < max"):
        broker.request_sweep(
            {
                "output_dir": str(broker.run_dir / "tuning" / "iter_1"),
                "active_search": {"restitution": {"min": 0.9, "max": 0.1}},
            }
        )
    assert broker.budget.sweeps_reserved == 0
    # Reserved wrapper-owned run entries are refused as sweep output dirs.
    with pytest.raises(BrokerError, match="reserved run entry"):
        broker.request_sweep({"output_dir": str(broker.run_dir / "final")})
    broker.close()


def test_zero_deadline_aborts_before_engine(tmp_path: Path) -> None:
    calls: list[Any] = []
    broker = _start_broker(
        tmp_path,
        _fake_external_runner(seen_inputs=calls),
        sweep_deadline_seconds=0.000001,
    )
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    view = _wait_terminal(broker, view["sweep_id"])
    assert view["status"] == "deadline_exceeded"
    assert not calls
    broker.close()


def test_verify_claim_rejects_tampered_evidence(tmp_path: Path) -> None:
    broker = _start_broker(tmp_path, _fake_external_runner())
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    view = _wait_terminal(broker, view["sweep_id"])
    evidence_path = Path(view["evidence_path"])
    evidence_path.write_text(
        evidence_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    ok, reason = broker.verify_claim(
        {"sweep_id": view["sweep_id"], "evidence_sha256": view["evidence_sha256"]}
    )
    assert not ok
    assert "modified after" in reason
    broker.close()


@pytest.mark.parametrize(
    "target", ["best_recording.usd", "render_response_metadata.json"]
)
def test_verify_claim_rejects_tampered_digest_bound_files(
    tmp_path: Path, target: str
) -> None:
    """Every digest-addressed published file is rehashed at verification, not
    just evidence.json: the recording and render-response metadata are the
    reproducibility anchors of the frames, so a post-sweep edit to either
    must fail the claim closed."""

    broker = _start_broker(tmp_path, _fake_external_runner())
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    view = _wait_terminal(broker, view["sweep_id"])
    assert view["status"] == "succeeded"
    tampered = Path(view["evidence_path"]).parent / target
    tampered.write_text(tampered.read_text(encoding="utf-8") + " ", encoding="utf-8")
    ok, reason = broker.verify_claim(
        {"sweep_id": view["sweep_id"], "evidence_sha256": view["evidence_sha256"]}
    )
    assert not ok
    assert "modified after" in reason
    broker.close()


def test_verify_reviewed_frames_rejects_tampering(tmp_path: Path) -> None:
    broker = _start_broker(tmp_path, _fake_external_runner())
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    view = _wait_terminal(broker, view["sweep_id"])
    sweep_id = view["sweep_id"]

    ok, reason = broker.verify_reviewed_frames(sweep_id, [])
    assert not ok and "mandatory" in reason
    ok, reason = broker.verify_reviewed_frames(
        sweep_id, [{"path": "/nowhere.png", "sha256": "0" * 64}]
    )
    assert not ok and "not published" in reason
    frame = dict(view["frames"][0])
    Path(frame["path"]).write_bytes(b"tampered")
    ok, reason = broker.verify_reviewed_frames(sweep_id, [frame])
    assert not ok and "modified after" in reason
    broker.close()


def test_http_surface(tmp_path: Path) -> None:
    requests = pytest.importorskip("requests")
    broker = _start_broker(tmp_path, _fake_external_runner())
    broker.start()
    try:
        budget_response = requests.get(f"{broker.url}/budget", timeout=5.0)
        assert budget_response.headers["X-Content-Type-Options"] == "nosniff"
        assert budget_response.json()["max_sweeps"] == 3
        response = requests.post(
            f"{broker.url}/sweeps",
            json={"output_dir": str(broker.run_dir / "tuning" / "iter_1")},
            timeout=5.0,
        )
        assert response.status_code == 200
        sweep_id = response.json()["sweep_id"]
        _wait_terminal(broker, sweep_id)
        view = requests.get(f"{broker.url}/sweeps/{sweep_id}", timeout=5.0).json()
        assert view["status"] == "succeeded"
        missing = requests.get(f"{broker.url}/sweeps/nope", timeout=5.0)
        assert missing.status_code == 404
    finally:
        broker.close()


def test_remote_playback_renderer_fails_closed(tmp_path: Path) -> None:
    """A "remote" playback renderer is any REST render service; its response
    carries no verifiable OVRTX identity, so evidence publication must fail
    closed instead of accepting frames from an unattested endpoint."""

    import dataclasses

    from physics_agent.tuning.external.types import EvidenceSettings

    spec = dataclasses.replace(
        _make_spec(tmp_path), evidence=EvidenceSettings(playback_renderer="remote")
    )
    broker = _start_broker(tmp_path, _fake_external_runner(), spec=spec)
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    view = _wait_terminal(broker, view["sweep_id"])
    assert view["status"] == "failed"
    assert "OVRTX" in (view["error"] or "")
    broker.close()


def test_metadata_renderer_mismatch_fails_closed(tmp_path: Path) -> None:
    """The response metadata records the renderer that actually produced the
    frames; an engine that rendered through a different backend than the
    configured OVRTX one must not publish evidence."""

    ovrtx_runner = _fake_external_runner()

    def warp_metadata_runner(tune_input: Any) -> Any:
        result = ovrtx_runner(tune_input)
        metadata = (
            Path(tune_input.output_dir) / "render" / "render_response_metadata.json"
        )
        metadata.write_text('{"renderer": "warp"}', encoding="utf-8")
        return result

    broker = _start_broker(tmp_path, warp_metadata_runner)
    view = broker.request_sweep(
        {"output_dir": str(broker.run_dir / "tuning" / "iter_1")}
    )
    view = _wait_terminal(broker, view["sweep_id"])
    assert view["status"] == "failed"
    assert "OVRTX" in (view["error"] or "")
    broker.close()


def test_release_private_dir_reports_skipped_release(tmp_path: Path) -> None:
    """A worker thread that outlives close() (a lingering BYOR runtime) makes
    the release a no-op; the caller must learn the workspace is still on disk
    so the leaked path gets reported instead of silently forgotten."""

    broker = _start_broker(tmp_path, _fake_external_runner(), private_dir=None)
    private_dir = broker.private_dir
    assert private_dir.is_dir()

    release_gate = threading.Event()
    worker = threading.Thread(target=release_gate.wait, daemon=True)
    worker.start()
    broker._workers.append(worker)
    try:
        assert broker.release_private_dir() is False
        assert private_dir.is_dir()
    finally:
        release_gate.set()
    worker.join(timeout=5.0)
    assert broker.release_private_dir() is True
    assert not private_dir.exists()
    # Idempotent once released.
    assert broker.release_private_dir() is True
    broker.close()


def test_release_private_dir_never_removes_an_unowned_dir(tmp_path: Path) -> None:
    """An operator-provided private directory is not the broker's to delete;
    releasing must leave it alone while reporting nothing is left to clean."""

    broker = _start_broker(tmp_path, _fake_external_runner())
    assert broker.release_private_dir() is True
    assert (tmp_path / "private").is_dir()
    broker.close()


def test_release_private_dir_reports_partial_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rmtree can fail partway (e.g. a non-traversable directory left by the
    external runtime); success is confirmed by the directory actually being
    gone, so a leaked workspace is reported instead of recorded as
    released."""

    import shutil as shutil_module

    broker = _start_broker(tmp_path, _fake_external_runner(), private_dir=None)
    private_dir = broker.private_dir
    assert private_dir.is_dir()

    monkeypatch.setattr(shutil_module, "rmtree", lambda *args, **kwargs: None)
    assert broker.release_private_dir() is False
    assert private_dir.is_dir()

    monkeypatch.undo()
    assert broker.release_private_dir() is True
    assert not private_dir.exists()
    broker.close()
