# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep broker tests: protocol-enforced budget, sanitization, deadlines,
trust boundaries (engine, sweep input, digests), and top-K materialization."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests
import yaml

from content_workflow_cli.tuning_broker import (
    BrokerError,
    PhysicsTuningBroker,
    SweepRecord,
    sha256_file,
)


def _write_scenario(path: Path, *, vlm_check: str | None = None) -> Path:
    scenario: dict[str, Any] = {
        "name": "drop_settle",
        "metric": "settle_distance",
        "target": {"drop_height_m": 0.5, "duration_s": 1.0},
        "parameters": [
            {"name": "dynamic_friction", "min": 0.05, "max": 1.0},
        ],
    }
    if vlm_check is not None:
        scenario["target"]["vlm_check"] = vlm_check
        scenario["judge"] = {"temperature": 0.2}
    path.write_text(yaml.safe_dump(scenario), encoding="utf-8")
    return path


def _fake_trial(index: int, score: float, *, failed: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        trial_index=index,
        params={"dynamic_friction": 0.1 * (index + 1)},
        score=score,
        backend_metrics={"recording_usd": f"/tmp/trial_{index}/recording.usd"},
        failed=failed,
        error="boom" if failed else None,
    )


FAKE_RESOLVED_BINDINGS = [
    {"name": "dynamic_friction", "binding_kind": "usd_attribute"},
    {"name": "contact_ke", "binding_kind": "simulator_parameter"},
]


def _fake_result(
    output_dir: Path,
    *,
    success: bool = True,
    cancelled: bool = False,
    trials: list[SimpleNamespace] | None = None,
    write_tune_results: bool = True,
) -> SimpleNamespace:
    tuned = output_dir / "tuned_physics.usd"
    tuned.parent.mkdir(parents=True, exist_ok=True)
    tuned.write_text("#usda 1.0\n", encoding="utf-8")
    if write_tune_results:
        # Mirrors physics_agent's tune_results.json: the resolved scenario
        # parameter bindings the sweep ran with live under scenario.extra.
        (output_dir / "tune_results.json").write_text(
            json.dumps(
                {
                    "scenario": {
                        "extra": {"resolved_parameter_bindings": FAKE_RESOLVED_BINDINGS}
                    }
                }
            ),
            encoding="utf-8",
        )
    history = trials if trials is not None else [_fake_trial(0, 0.5)]
    best = min(
        (trial for trial in history if not trial.failed),
        key=lambda trial: trial.score,
        default=None,
    )
    return SimpleNamespace(
        success=success,
        cancelled=cancelled,
        error=None if success else "tune failed",
        history=history,
        artifacts={"tuned_physics.usd": tuned},
        best_params=dict(best.params) if best else {},
        best_score=best.score if best else float("inf"),
        n_trials=len(history),
        optimizer_used="random",
        engine_used="fake",
        needs_refinement=False,
    )


def _make_broker(tmp_path: Path, **overrides: Any) -> PhysicsTuningBroker:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    captured: list[Any] = []

    def tune_runner(tune_input: Any) -> Any:
        captured.append(tune_input)
        return _fake_result(Path(tune_input.output_dir))

    kwargs: dict[str, Any] = {
        "run_dir": run_dir,
        "engine": "fake",
        "optimizer": "random",
        "max_sweeps": 2,
        "max_trials_per_sweep": 10,
        "sweep_deadline_seconds": 30.0,
        "private_dir": tmp_path / "private",
        "tune_runner": tune_runner,
    }
    kwargs.update(overrides)
    broker = PhysicsTuningBroker(**kwargs)
    broker.captured_tune_inputs = captured  # type: ignore[attr-defined]
    return broker


def _request_and_wait(
    broker: PhysicsTuningBroker, payload: dict[str, Any], timeout: float = 10.0
) -> dict[str, Any]:
    view = broker.request_sweep(payload)
    sweep_id = view["sweep_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        view = broker.sweep_view(sweep_id)
        if view["status"] != "running":
            return view
        time.sleep(0.02)
    raise AssertionError(f"sweep {sweep_id} did not finish: {view}")


def _sweep_payload(broker: PhysicsTuningBroker, tmp_path: Path) -> dict[str, Any]:
    scenario = _write_scenario(tmp_path / "scenario.yaml")
    physics = tmp_path / "physics.usda"
    physics.write_text("#usda 1.0\n", encoding="utf-8")
    return {
        "scenario_path": str(scenario),
        "physics_usd": str(physics),
        "output_dir": str(broker.run_dir / "tuning" / "iter_1"),
    }


def test_budget_reserved_atomically_before_sweep_and_refused_when_exhausted(
    tmp_path: Path,
) -> None:
    broker = _make_broker(tmp_path, max_sweeps=1)
    payload = _sweep_payload(broker, tmp_path)
    view = _request_and_wait(broker, payload)
    assert view["status"] == "succeeded"

    payload["output_dir"] = str(broker.run_dir / "tuning" / "iter_2")
    with pytest.raises(BrokerError) as excinfo:
        broker.request_sweep(payload)
    assert excinfo.value.status == 409
    assert "budget exhausted" in str(excinfo.value)


def test_failed_sweep_still_consumes_reserved_budget(tmp_path: Path) -> None:
    def failing_runner(_tune_input: Any) -> Any:
        raise RuntimeError("engine exploded")

    broker = _make_broker(tmp_path, max_sweeps=1, tune_runner=failing_runner)
    payload = _sweep_payload(broker, tmp_path)
    view = _request_and_wait(broker, payload)
    assert view["status"] == "failed"
    assert "engine exploded" in view["error"]
    assert broker.budget_view()["sweeps_remaining"] == 0
    payload["output_dir"] = str(broker.run_dir / "tuning" / "iter_2")
    with pytest.raises(BrokerError):
        broker.request_sweep(payload)


def test_broker_rejects_protected_parameters_before_reserving_budget(
    tmp_path: Path,
) -> None:
    broker = _make_broker(tmp_path, forbidden_parameters={"mass_scale"})
    payload = _sweep_payload(broker, tmp_path)
    scenario_path = Path(str(payload["scenario_path"]))
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario["parameters"].append({"name": "mass_scale", "min": 0.5, "max": 2.0})
    scenario_path.write_text(yaml.safe_dump(scenario), encoding="utf-8")

    with pytest.raises(BrokerError, match="wrapper-protected.*mass_scale") as excinfo:
        broker.request_sweep(payload)

    assert excinfo.value.status == 400
    assert broker.budget_view()["sweeps_reserved"] == 0


def test_broker_rejects_unknown_optimizer_override_before_reserving_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.optimization import registry as reg

    monkeypatch.setattr(reg, "_optimizer_plugins_scanned", True)
    broker = _make_broker(tmp_path)
    payload = _sweep_payload(broker, tmp_path)
    payload["optimizer"] = "does-not-exist"

    with pytest.raises(BrokerError, match="Unknown optimizer") as excinfo:
        broker.request_sweep(payload)

    assert excinfo.value.status == 400
    assert broker.budget_view()["sweeps_reserved"] == 0


def test_broker_rejects_unavailable_plugin_optimizer_override_before_reserving_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.optimization import registry as reg

    plugin = reg.OptimizerPlugin(
        runner=lambda *a, **kw: None,
        is_available=lambda: False,
        unavailable_message="fake-opt requires the fake-opt-client package.",
    )
    monkeypatch.setitem(reg._optimizer_plugins, "fake-opt", plugin)
    monkeypatch.setattr(reg, "_optimizer_plugins_scanned", True)

    broker = _make_broker(tmp_path)
    payload = _sweep_payload(broker, tmp_path)
    payload["optimizer"] = "fake-opt"

    with pytest.raises(BrokerError, match="fake-opt-client") as excinfo:
        broker.request_sweep(payload)

    assert excinfo.value.status == 400
    assert broker.budget_view()["sweeps_reserved"] == 0


def test_broker_can_disable_rebuilt_physics_inputs(tmp_path: Path) -> None:
    initial = tmp_path / "initial.usda"
    initial.write_text("#usda 1.0\n# initial\n", encoding="utf-8")
    broker = _make_broker(
        tmp_path,
        initial_physics_usd=initial,
        allow_rebuilt_inputs=False,
    )
    payload = _sweep_payload(broker, tmp_path)
    Path(str(payload["physics_usd"])).write_text(
        "#usda 1.0\n# rebuilt\n", encoding="utf-8"
    )

    with pytest.raises(
        BrokerError, match="rebuilt physics USD inputs are disabled"
    ) as excinfo:
        broker.request_sweep(payload)

    assert excinfo.value.status == 400
    assert broker.budget_view()["sweeps_reserved"] == 0


def test_sweep_is_sanitized_to_pure_inner_loop(tmp_path: Path) -> None:
    broker = _make_broker(tmp_path)
    scenario = _write_scenario(tmp_path / "scenario.yaml", vlm_check="always")
    physics = tmp_path / "physics.usda"
    physics.write_text("#usda 1.0\n", encoding="utf-8")
    view = _request_and_wait(
        broker,
        {
            "scenario_path": str(scenario),
            "physics_usd": str(physics),
            "output_dir": str(broker.run_dir / "tuning" / "iter_1"),
            "max_trials": 500,
        },
    )
    assert view["status"] == "succeeded"
    tune_input = broker.captured_tune_inputs[0]  # type: ignore[attr-defined]
    assert tune_input.enable_judge is False
    assert tune_input.user_prompt is None
    # Requested trials are capped by the per-sweep budget.
    assert tune_input.max_trials == 10
    sanitized = yaml.safe_load(Path(tune_input.scenario).read_text(encoding="utf-8"))
    assert sanitized["target"]["vlm_check"] == "off"
    assert "judge" not in sanitized


def test_sweep_deadline_cancels_cooperatively(tmp_path: Path) -> None:
    def slow_runner(tune_input: Any) -> Any:
        cancel: threading.Event = tune_input.cancel_event
        cancel.wait(timeout=10.0)
        return _fake_result(Path(tune_input.output_dir), success=False, cancelled=True)

    broker = _make_broker(tmp_path, sweep_deadline_seconds=0.2, tune_runner=slow_runner)
    view = _request_and_wait(broker, _sweep_payload(broker, tmp_path))
    assert view["status"] == "deadline_exceeded"


def test_sweep_deadline_refuses_late_success_from_noncooperative_runner(
    tmp_path: Path,
) -> None:
    def noncooperative_runner(tune_input: Any) -> Any:
        time.sleep(0.2)
        return _fake_result(Path(tune_input.output_dir), success=True)

    broker = _make_broker(
        tmp_path,
        sweep_deadline_seconds=0.05,
        tune_runner=noncooperative_runner,
    )
    view = _request_and_wait(broker, _sweep_payload(broker, tmp_path))

    assert view["status"] == "deadline_exceeded"
    with pytest.raises(BrokerError) as excinfo:
        broker.materialize(view["sweep_id"], trial_index=0)
    assert excinfo.value.status == 409


def test_expired_reservation_refuses_success_before_timer_runs(tmp_path: Path) -> None:
    broker = _make_broker(tmp_path)
    payload = _sweep_payload(broker, tmp_path)
    record = SweepRecord(
        sweep_id="sweep-expired",
        iter_dir=Path(payload["output_dir"]),
        work_dir=tmp_path / "private" / "sweep-expired",
        scenario_path=Path(payload["scenario_path"]),
        scenario_sha256=sha256_file(payload["scenario_path"]),
        physics_usd=Path(payload["physics_usd"]),
        physics_usd_sha256=sha256_file(payload["physics_usd"]),
        engine="fake",
        optimizer="random",
        max_trials=1,
    )

    broker._run_sweep(  # noqa: SLF001 - verify the worker-start race directly
        record,
        time.monotonic() - 0.01,
        threading.Event(),
    )

    assert record.status == "deadline_exceeded"


def test_phase_deadline_refuses_new_reservations(tmp_path: Path) -> None:
    broker = _make_broker(tmp_path, phase_deadline_seconds=0.01)
    time.sleep(0.05)
    with pytest.raises(BrokerError) as excinfo:
        broker.request_sweep(_sweep_payload(broker, tmp_path))
    assert excinfo.value.status == 409
    assert "phase deadline" in str(excinfo.value)


def test_evidence_packet_exposes_top_k_candidates(tmp_path: Path) -> None:
    trials = [
        _fake_trial(0, 0.9),
        _fake_trial(1, 0.1),
        _fake_trial(2, 0.5),
        _fake_trial(3, 2.0, failed=True),
    ]

    def runner_with_history(tune_input: Any) -> Any:
        return _fake_result(Path(tune_input.output_dir), trials=trials)

    broker = _make_broker(tmp_path, top_k=2, tune_runner=runner_with_history)
    view = _request_and_wait(broker, _sweep_payload(broker, tmp_path))
    evidence = json.loads(Path(view["evidence_path"]).read_text(encoding="utf-8"))
    candidates = evidence["candidates"]
    assert [entry["trial_index"] for entry in candidates] == [1, 2]
    assert candidates[0]["rank"] == 1
    assert candidates[0]["recording"] == "/tmp/trial_1/recording.usd"
    assert view["evidence_sha256"] == sha256_file(view["evidence_path"])


def test_materialize_only_allows_top_k_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trials = [_fake_trial(0, 0.9), _fake_trial(1, 0.1)]

    def runner_with_history(tune_input: Any) -> Any:
        return _fake_result(Path(tune_input.output_dir), trials=trials)

    import physics_agent.tuning.usd_patch as usd_patch

    captured_bindings: list[Any] = []

    def fake_patch(
        input_usd: Path,
        output_usd: Path,
        params: dict,
        *,
        bindings: Any = None,
        **_: Any,
    ) -> Path:
        captured_bindings.append(bindings)
        Path(output_usd).parent.mkdir(parents=True, exist_ok=True)
        Path(output_usd).write_text(
            f"#usda 1.0\n# params {sorted(params.items())}\n", encoding="utf-8"
        )
        return Path(output_usd)

    monkeypatch.setattr(usd_patch, "patch_physics_usd", fake_patch)
    broker = _make_broker(tmp_path, top_k=1, tune_runner=runner_with_history)
    view = _request_and_wait(broker, _sweep_payload(broker, tmp_path))
    sweep_id = view["sweep_id"]

    entry = broker.materialize(sweep_id, 1)
    assert Path(entry["usd_path"]).is_file()
    assert entry["usd_sha256"] == sha256_file(entry["usd_path"])
    # Candidates are patched through the sweep's resolved bindings so bound
    # backend parameters (contact_ke/contact_kd) survive materialization.
    assert captured_bindings == [FAKE_RESOLVED_BINDINGS]
    # Cached on repeat.
    assert broker.materialize(sweep_id, 1) == entry
    # Trial 0 is not a top-1 candidate.
    with pytest.raises(BrokerError) as excinfo:
        broker.materialize(sweep_id, 0)
    assert excinfo.value.status == 404
    with pytest.raises(BrokerError):
        broker.materialize("sweep-nonexistent", 1)


def test_materialize_honors_candidate_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Candidates are materialized in the promotion target's format."""

    import physics_agent.tuning.usd_patch as usd_patch

    def fake_patch(
        input_usd: Path,
        output_usd: Path,
        params: dict,
        *,
        bindings: Any = None,
        **_: Any,
    ) -> Path:
        Path(output_usd).parent.mkdir(parents=True, exist_ok=True)
        Path(output_usd).write_text("#usda 1.0\n", encoding="utf-8")
        return Path(output_usd)

    monkeypatch.setattr(usd_patch, "patch_physics_usd", fake_patch)
    broker = _make_broker(tmp_path, candidate_suffix=".usdc")
    view = _request_and_wait(broker, _sweep_payload(broker, tmp_path))

    entry = broker.materialize(view["sweep_id"], 0)
    assert Path(entry["usd_path"]).suffix == ".usdc"


def test_selected_candidate_replay_context_survives_artifact_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import physics_agent.tuning.usd_patch as usd_patch

    def fake_patch(
        input_usd: Path,
        output_usd: Path,
        params: dict[str, float],
        **_kwargs: Any,
    ) -> Path:
        del input_usd, params
        output_usd.parent.mkdir(parents=True, exist_ok=True)
        output_usd.write_text("#usda 1.0\n# selected\n", encoding="utf-8")
        return output_usd

    def runner_with_recording(tune_input: Any) -> Any:
        output_dir = Path(tune_input.output_dir)
        trial_dir = output_dir / ".tune_scenes" / "trial_seed_42"
        trial_dir.mkdir(parents=True, exist_ok=True)
        recording = trial_dir / "recording.usd"
        scene = trial_dir / "scene.usd"
        patched = trial_dir / "patched_physics.usd"
        for path in (recording, scene, patched):
            path.write_text("#usda 1.0\n", encoding="utf-8")
        trial = _fake_trial(0, 0.1)
        trial.backend_metrics = {
            "recording_usd": str(recording),
            "scene_usd": str(scene),
            "patched_usd": str(patched),
        }
        return _fake_result(output_dir, trials=[trial])

    monkeypatch.setattr(usd_patch, "patch_physics_usd", fake_patch)
    broker = _make_broker(tmp_path, tune_runner=runner_with_recording)
    payload = _sweep_payload(broker, tmp_path)
    original_scenario_sha256 = sha256_file(payload["scenario_path"])
    view = _request_and_wait(broker, payload)
    blocked_publish_dir = (
        Path(str(payload["output_dir"])) / "broker_artifacts" / view["sweep_id"]
    )
    blocked_publish_dir.mkdir(parents=True)
    with pytest.raises(BrokerError, match="destination already exists"):
        broker.publish_artifacts()
    materialized = broker.materialize(view["sweep_id"], 0)

    context = broker.selected_candidate_replay_context(
        view["sweep_id"],
        0,
        expected_usd_sha256=materialized["usd_sha256"],
    )

    assert context["trial_seed"] == 42
    assert context["scenario_sha256"] == original_scenario_sha256
    assert Path(context["scenario_path"]).is_relative_to(broker.run_dir)
    assert Path(context["sanitized_scenario_path"]).is_relative_to(broker.run_dir)
    assert not Path(context["scenario_path"]).is_relative_to(blocked_publish_dir)
    assert context["materialized_usd_sha256"] == materialized["usd_sha256"]


def test_candidate_recording_export_is_confined_to_broker_workspace(
    tmp_path: Path,
) -> None:
    def runner_with_recording(tune_input: Any) -> Any:
        output_dir = Path(tune_input.output_dir)
        recording = output_dir / ".tune_scenes" / "trial_seed_42" / "recording.usd"
        recording.parent.mkdir(parents=True, exist_ok=True)
        recording.write_text("#usda 1.0\n# rollout\n", encoding="utf-8")
        trial = _fake_trial(0, 0.1)
        trial.backend_metrics = {"recording_usd": str(recording)}
        return _fake_result(output_dir, trials=[trial])

    broker = _make_broker(tmp_path, tune_runner=runner_with_recording)
    view = _request_and_wait(broker, _sweep_payload(broker, tmp_path))

    recording, digest = broker.candidate_recording_file(view["sweep_id"], "0")

    assert recording.is_relative_to((tmp_path / "private").resolve())
    assert digest == sha256_file(recording)


def test_candidate_recording_export_rejects_outside_path(tmp_path: Path) -> None:
    outside = tmp_path / "outside-recording.usd"
    outside.write_text("#usda 1.0\n", encoding="utf-8")

    def runner_with_outside_recording(tune_input: Any) -> Any:
        trial = _fake_trial(0, 0.1)
        trial.backend_metrics = {"recording_usd": str(outside)}
        return _fake_result(Path(tune_input.output_dir), trials=[trial])

    broker = _make_broker(tmp_path, tune_runner=runner_with_outside_recording)
    view = _request_and_wait(broker, _sweep_payload(broker, tmp_path))

    with pytest.raises(BrokerError, match="escapes the broker workspace"):
        broker.candidate_recording_file(view["sweep_id"], "0")


def test_sweep_rejects_physics_usd_with_external_layer_dependencies(
    tmp_path: Path,
) -> None:
    """Sweeps run from a private workspace, so a USD whose geometry arrives
    through a relative reference would silently compose empty there."""

    broker = _make_broker(tmp_path)
    payload = _sweep_payload(broker, tmp_path)
    geo = tmp_path / "geo.usda"
    geo.write_text('#usda 1.0\ndef Mesh "M" {}\n', encoding="utf-8")
    referencing = tmp_path / "physics_with_ref.usda"
    referencing.write_text(
        '#usda 1.0\ndef Xform "World" ( references = @./geo.usda@</M> ) {}\n',
        encoding="utf-8",
    )
    payload["physics_usd"] = str(referencing)

    view = _request_and_wait(broker, payload)

    assert view["status"] == "failed"
    assert "self-contained" in view["error"]


def test_sweep_accepts_self_contained_physics_usd(tmp_path: Path) -> None:
    """The normal flattened workflow output still sweeps."""

    broker = _make_broker(tmp_path)
    view = _request_and_wait(broker, _sweep_payload(broker, tmp_path))

    assert view["status"] == "succeeded"


def test_broker_rejects_unsupported_candidate_suffix(tmp_path: Path) -> None:
    """.usdz is a package format; a plain layer export cannot author it."""

    with pytest.raises(ValueError, match="Unsupported candidate suffix"):
        _make_broker(tmp_path, candidate_suffix=".usdz")


def test_verify_claim_rejects_unknown_and_mismatched_sweeps(tmp_path: Path) -> None:
    broker = _make_broker(tmp_path)
    view = _request_and_wait(broker, _sweep_payload(broker, tmp_path))
    ok, _reason = broker.verify_claim(
        {
            "sweep_id": view["sweep_id"],
            "scenario_sha256": view["scenario_sha256"],
            "evidence_sha256": view["evidence_sha256"],
        }
    )
    assert ok

    ok, reason = broker.verify_claim({"sweep_id": "sweep-forged"})
    assert not ok
    assert "no broker record" in reason

    ok, reason = broker.verify_claim(
        {
            "sweep_id": view["sweep_id"],
            "scenario_sha256": view["scenario_sha256"],
            "evidence_sha256": "0" * 64,
        }
    )
    assert not ok
    assert "mismatch" in reason

    # Digest bindings are mandatory: omitting a digest the broker recorded
    # is a rejection, not a skipped comparison.
    ok, reason = broker.verify_claim(
        {"sweep_id": view["sweep_id"], "scenario_sha256": view["scenario_sha256"]}
    )
    assert not ok
    assert "evidence_sha256 missing" in reason


def test_output_dir_outside_run_dir_is_rejected(tmp_path: Path) -> None:
    broker = _make_broker(tmp_path)
    payload = _sweep_payload(broker, tmp_path)
    payload["output_dir"] = str(tmp_path / "escape")
    with pytest.raises(BrokerError) as excinfo:
        broker.request_sweep(payload)
    assert excinfo.value.status == 400


def test_child_symlink_in_output_dir_is_rejected_without_touching_target(
    tmp_path: Path,
) -> None:
    broker = _make_broker(tmp_path)
    payload = _sweep_payload(broker, tmp_path)
    iter_dir = Path(payload["output_dir"])
    iter_dir.mkdir(parents=True)
    target = tmp_path / "parent_owned.json"
    original = '{"owner": "wrapper"}\n'
    target.write_text(original, encoding="utf-8")
    (iter_dir / "evidence.json").symlink_to(target)

    with pytest.raises(BrokerError) as excinfo:
        broker.request_sweep(payload)

    assert excinfo.value.status == 400
    assert "symbolic link" in str(excinfo.value)
    assert target.read_text(encoding="utf-8") == original
    assert broker.budget_view()["sweeps_reserved"] == 0


def test_symlink_created_after_reservation_cannot_capture_broker_writes(
    tmp_path: Path,
) -> None:
    runner_started = threading.Event()
    release_runner = threading.Event()

    def blocked_runner(tune_input: Any) -> Any:
        runner_started.set()
        assert release_runner.wait(timeout=10.0)
        return _fake_result(Path(tune_input.output_dir))

    broker = _make_broker(tmp_path, tune_runner=blocked_runner)
    payload = _sweep_payload(broker, tmp_path)
    view = broker.request_sweep(payload)
    assert runner_started.wait(timeout=2.0)

    iter_dir = Path(payload["output_dir"])
    iter_dir.mkdir(parents=True)
    target = tmp_path / "parent_owned.json"
    original = '{"owner": "wrapper"}\n'
    target.write_text(original, encoding="utf-8")
    (iter_dir / "evidence.json").symlink_to(target)
    release_runner.set()

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        view = broker.sweep_view(view["sweep_id"])
        if view["status"] != "running":
            break
        time.sleep(0.02)

    assert view["status"] == "succeeded"
    assert target.read_text(encoding="utf-8") == original
    assert Path(view["evidence_path"]).parent != iter_dir
    assert view["evidence"]["sweep_id"] == view["sweep_id"]
    with pytest.raises(BrokerError) as excinfo:
        broker.publish_artifacts()
    assert excinfo.value.status == 400
    assert target.read_text(encoding="utf-8") == original

    (iter_dir / "evidence.json").unlink()
    broker.publish_artifacts()
    published = broker.sweep_view(view["sweep_id"])
    published_dir = Path(published["published_dir"])
    assert published_dir.is_dir()
    assert Path(published["published_evidence_path"]).is_file()
    assert published_dir != Path(view["evidence_path"]).parent


def test_http_surface_reserves_and_serves_sweeps(tmp_path: Path) -> None:
    broker = _make_broker(tmp_path, max_sweeps=1)
    broker.start()
    try:
        payload = _sweep_payload(broker, tmp_path)
        response = requests.post(f"{broker.url}/sweeps", json=payload, timeout=10)
        assert response.status_code == 200
        sweep_id = response.json()["sweep_id"]
        deadline = time.monotonic() + 10
        status = "running"
        while time.monotonic() < deadline and status == "running":
            status = requests.get(f"{broker.url}/sweeps/{sweep_id}", timeout=10).json()[
                "status"
            ]
            time.sleep(0.02)
        assert status == "succeeded"

        refused = requests.post(f"{broker.url}/sweeps", json=payload, timeout=10)
        assert refused.status_code == 409

        budget_response = requests.get(f"{broker.url}/budget", timeout=10)
        assert budget_response.headers["X-Content-Type-Options"] == "nosniff"
        assert budget_response.json()["sweeps_remaining"] == 0
    finally:
        broker.close()


def test_ledger_records_are_hmac_signed(tmp_path: Path) -> None:
    broker = _make_broker(tmp_path)
    _request_and_wait(broker, _sweep_payload(broker, tmp_path))
    ledger_path = tmp_path / "private" / "sweep_ledger.jsonl"
    lines = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines
    assert all(
        entry["hmac_sha256"] and len(entry["hmac_sha256"]) == 64 for entry in lines
    )
    assert lines[-1]["record"]["status"] == "succeeded"


def test_verify_claim_accepts_failed_sweep_references(tmp_path: Path) -> None:
    """A revise/stop decision may cite the failed sweep it learned from."""

    def failing_runner(_tune_input: Any) -> Any:
        raise RuntimeError("backend rejected scenario params")

    broker = _make_broker(tmp_path, tune_runner=failing_runner)
    view = _request_and_wait(broker, _sweep_payload(broker, tmp_path))
    assert view["status"] == "failed"

    ok, reason = broker.verify_claim(
        {"sweep_id": view["sweep_id"], "scenario_sha256": view["scenario_sha256"]}
    )
    assert ok, reason
    # Promotion safety is preserved elsewhere: a failed sweep can be cited
    # but never materialized into an accepted candidate.
    with pytest.raises(BrokerError) as excinfo:
        broker.materialize(view["sweep_id"], 0)
    assert excinfo.value.status == 409


def test_engine_override_is_rejected(tmp_path: Path) -> None:
    """The tune engine is broker-enforced; a mismatch is an explicit error."""

    broker = _make_broker(tmp_path)  # engine="fake"
    payload = _sweep_payload(broker, tmp_path)
    payload["engine"] = "ovphysx"
    with pytest.raises(BrokerError) as excinfo:
        broker.request_sweep(payload)
    assert excinfo.value.status == 400
    assert "broker-enforced" in str(excinfo.value)

    # Passing the broker's own engine is fine (informational).
    payload["engine"] = "fake"
    view = _request_and_wait(broker, payload)
    assert view["status"] == "succeeded"
    assert view["engine"] == "fake"


def _write_revise_patch_decision(
    run_dir: Path,
    rebuilt_usd: Path,
    sweep: dict[str, Any],
    *,
    digest: str | None = None,
) -> Path:
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    decision_path = run_dir / "raw" / "physics_tuning_decision_1.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.physics-tuning-decision.v1",
                "iteration": 1,
                "decision": "revise_patch",
                "sweep_id": sweep["sweep_id"],
                "scenario_sha256": sweep["scenario_sha256"],
                "evidence_sha256": sweep["evidence_sha256"],
                "revised_patch_path": str(run_dir / "raw" / "patch_v1.json"),
                "rebuilt_physics_usd": str(rebuilt_usd),
                "rebuilt_physics_usd_sha256": digest or sha256_file(rebuilt_usd),
                "prior_decision_sha256": None,
                "rationale": "collider approximation was wrong",
            }
        ),
        encoding="utf-8",
    )
    return decision_path


def test_sweep_input_must_be_finalized_or_digest_bound_rebuild(
    tmp_path: Path,
) -> None:
    finalized = tmp_path / "finalized.usda"
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")
    broker = _make_broker(tmp_path, max_sweeps=5, initial_physics_usd=finalized)

    # The finalized input itself is always accepted.
    payload = _sweep_payload(broker, tmp_path)
    payload["physics_usd"] = str(finalized)
    view = _request_and_wait(broker, payload)
    assert view["status"] == "succeeded"

    # An arbitrary other USD without a citing decision is rejected.
    other = tmp_path / "other.usda"
    other.write_text("#usda 1.0\n# other\n", encoding="utf-8")
    payload = _sweep_payload(broker, tmp_path)
    payload["physics_usd"] = str(other)
    payload["output_dir"] = str(broker.run_dir / "tuning" / "iter_2")
    with pytest.raises(BrokerError) as excinfo:
        broker.request_sweep(payload)
    assert excinfo.value.status == 400
    assert "revise_patch" in str(excinfo.value)

    # A rebuilt USD digest-bound to a revise_patch decision is accepted.
    decision_path = _write_revise_patch_decision(broker.run_dir, other, view)
    payload["rebuilt_decision_path"] = str(decision_path)
    view = _request_and_wait(broker, payload)
    assert view["status"] == "succeeded"

    # A digest mismatch (file changed after the decision) is rejected.
    other.write_text("#usda 1.0\n# mutated\n", encoding="utf-8")
    payload["output_dir"] = str(broker.run_dir / "tuning" / "iter_3")
    with pytest.raises(BrokerError) as excinfo:
        broker.request_sweep(payload)
    assert excinfo.value.status == 400
    assert "digest" in str(excinfo.value)


def test_rebuilt_input_rejects_fabricated_decision_chain(tmp_path: Path) -> None:
    finalized = tmp_path / "finalized.usda"
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")
    broker = _make_broker(tmp_path, max_sweeps=5, initial_physics_usd=finalized)

    # Establish a real sweep record, then try to authorize a rebuild through
    # a standalone decision that cites an invented sweep instead.
    payload = _sweep_payload(broker, tmp_path)
    payload["physics_usd"] = str(finalized)
    _request_and_wait(broker, payload)
    rebuilt = tmp_path / "rebuilt.usda"
    rebuilt.write_text("#usda 1.0\n# rebuilt\n", encoding="utf-8")
    forged = {
        "sweep_id": "sweep-forged",
        "scenario_sha256": "a" * 64,
        "evidence_sha256": "b" * 64,
    }
    decision_path = _write_revise_patch_decision(broker.run_dir, rebuilt, forged)

    payload = _sweep_payload(broker, tmp_path)
    payload.update(
        {
            "physics_usd": str(rebuilt),
            "output_dir": str(broker.run_dir / "tuning" / "iter_2"),
            "rebuilt_decision_path": str(decision_path),
        }
    )
    with pytest.raises(BrokerError) as excinfo:
        broker.request_sweep(payload)
    assert excinfo.value.status == 400
    assert "broker-backed" in str(excinfo.value)
    assert "no broker record" in str(excinfo.value)


def test_rebuilt_input_rejects_nonpositive_decision_iteration(tmp_path: Path) -> None:
    finalized = tmp_path / "finalized.usda"
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")
    broker = _make_broker(tmp_path, initial_physics_usd=finalized)
    rebuilt = tmp_path / "rebuilt.usda"
    rebuilt.write_text("#usda 1.0\n# rebuilt\n", encoding="utf-8")
    decision_path = broker.run_dir / "raw" / "physics_tuning_decision_0.json"
    decision_path.parent.mkdir(parents=True)
    decision_path.write_text("{}", encoding="utf-8")

    payload = _sweep_payload(broker, tmp_path)
    payload.update(
        {
            "physics_usd": str(rebuilt),
            "output_dir": str(broker.run_dir / "tuning" / "iter_1"),
            "rebuilt_decision_path": str(decision_path),
        }
    )
    with pytest.raises(BrokerError) as excinfo:
        broker.request_sweep(payload)
    assert excinfo.value.status == 400
    assert "iteration must be at least 1" in str(excinfo.value)


def test_rebuilt_input_enforces_broker_iteration_cap(tmp_path: Path) -> None:
    finalized = tmp_path / "finalized.usda"
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")
    rebuilt = tmp_path / "rebuilt.usda"
    rebuilt.write_text("#usda 1.0\n# rebuilt\n", encoding="utf-8")
    broker = _make_broker(
        tmp_path,
        max_sweeps=2,
        initial_physics_usd=finalized,
    )
    first_sweep_payload = _sweep_payload(broker, tmp_path)
    first_sweep_payload["physics_usd"] = str(finalized)
    first_sweep = _request_and_wait(broker, first_sweep_payload)
    assert first_sweep["status"] == "succeeded"

    second_sweep_payload = _sweep_payload(broker, tmp_path)
    second_sweep_payload.update(
        {
            "physics_usd": str(finalized),
            "output_dir": str(broker.run_dir / "tuning" / "iter_2"),
        }
    )
    second_sweep = _request_and_wait(broker, second_sweep_payload)
    assert second_sweep["status"] == "succeeded"

    raw_dir = broker.run_dir / "raw"
    raw_dir.mkdir(parents=True)
    first = raw_dir / "physics_tuning_decision_1.json"
    first.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.physics-tuning-decision.v1",
                "iteration": 1,
                "decision": "revise_scenario",
                "sweep_id": first_sweep["sweep_id"],
                "scenario_sha256": first_sweep["scenario_sha256"],
                "evidence_sha256": first_sweep["evidence_sha256"],
                "next_scenario_path": "tuning/iter_2/scenario.yaml",
                "prior_decision_sha256": None,
                "rationale": "revise the search",
            }
        ),
        encoding="utf-8",
    )
    second = raw_dir / "physics_tuning_decision_2.json"
    second.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.physics-tuning-decision.v1",
                "iteration": 2,
                "decision": "revise_patch",
                "sweep_id": second_sweep["sweep_id"],
                "scenario_sha256": second_sweep["scenario_sha256"],
                "evidence_sha256": second_sweep["evidence_sha256"],
                "revised_patch_path": str(raw_dir / "patch_v2.json"),
                "rebuilt_physics_usd": str(rebuilt),
                "rebuilt_physics_usd_sha256": sha256_file(rebuilt),
                "prior_decision_sha256": sha256_file(first),
                "rationale": "rebuild the collider",
            }
        ),
        encoding="utf-8",
    )
    # Model a resumed run whose configured cap is stricter than the broker
    # records on disk; chain validation must reject the second judged sweep.
    broker.budget.max_sweeps = 1
    payload = _sweep_payload(broker, tmp_path)
    payload.update(
        {
            "physics_usd": str(rebuilt),
            "output_dir": str(broker.run_dir / "tuning" / "iter_3"),
            "rebuilt_decision_path": str(second),
        }
    )

    with pytest.raises(BrokerError) as excinfo:
        broker.request_sweep(payload)

    assert excinfo.value.status == 400
    assert "configured max_iterations=1 was exhausted" in str(excinfo.value)


def test_materialize_refused_without_recovered_bindings(
    tmp_path: Path,
) -> None:
    """No tune_results.json → the candidate cannot be reproduced exactly."""

    def runner_without_results(tune_input: Any) -> Any:
        return _fake_result(Path(tune_input.output_dir), write_tune_results=False)

    broker = _make_broker(tmp_path, tune_runner=runner_without_results)
    view = _request_and_wait(broker, _sweep_payload(broker, tmp_path))
    assert view["status"] == "succeeded"
    assert view["bindings_recovered"] is False
    with pytest.raises(BrokerError) as excinfo:
        broker.materialize(view["sweep_id"], 0)
    assert excinfo.value.status == 409
    assert "cannot be reproduced exactly" in str(excinfo.value)


def test_phase_deadline_caps_active_sweep_deadline(tmp_path: Path) -> None:
    """A sweep reserved near the phase cutoff cannot run past it."""

    def slow_runner(tune_input: Any) -> Any:
        cancel: threading.Event = tune_input.cancel_event
        cancel.wait(timeout=10.0)
        return _fake_result(Path(tune_input.output_dir), success=False, cancelled=True)

    broker = _make_broker(
        tmp_path,
        sweep_deadline_seconds=30.0,
        phase_deadline_seconds=0.3,
        tune_runner=slow_runner,
    )
    started = time.monotonic()
    view = _request_and_wait(broker, _sweep_payload(broker, tmp_path))
    assert view["status"] == "deadline_exceeded"
    # Cancelled by the remaining phase time, not the 30s per-sweep ceiling.
    assert time.monotonic() - started < 10.0


def test_close_cancels_active_sweeps(tmp_path: Path) -> None:
    releases: list[float] = []
    runner_started = threading.Event()

    def slow_runner(tune_input: Any) -> Any:
        runner_started.set()
        cancel: threading.Event = tune_input.cancel_event
        cancel.wait(timeout=10.0)
        releases.append(time.monotonic())
        return _fake_result(Path(tune_input.output_dir), success=False, cancelled=True)

    broker = _make_broker(tmp_path, tune_runner=slow_runner)
    view = broker.request_sweep(_sweep_payload(broker, tmp_path))
    sweep_id = view["sweep_id"]
    assert runner_started.wait(timeout=2.0)
    started = time.monotonic()
    broker.close()
    assert releases and releases[0] - started < 9.0
    assert broker.sweep_view(sweep_id)["status"] in {"cancelled", "deadline_exceeded"}


def test_close_refuses_late_success_from_noncooperative_runner(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def noncooperative_runner(tune_input: Any) -> Any:
        started.set()
        assert release.wait(timeout=10.0)
        return _fake_result(Path(tune_input.output_dir), success=True)

    broker = _make_broker(tmp_path, tune_runner=noncooperative_runner)
    view = broker.request_sweep(_sweep_payload(broker, tmp_path))
    assert started.wait(timeout=2.0)

    broker.close(wait_for_workers=False)
    release.set()
    broker._workers[-1].join(timeout=2.0)  # noqa: SLF001 - wait for late result

    assert broker.sweep_view(view["sweep_id"])["status"] == "cancelled"
    with pytest.raises(BrokerError) as excinfo:
        broker.materialize(view["sweep_id"], trial_index=0)
    assert excinfo.value.status == 409


def test_close_defers_owned_private_dir_cleanup_until_worker_exits(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def noncooperative_runner(tune_input: Any) -> Any:
        started.set()
        assert release.wait(timeout=10.0)
        return _fake_result(Path(tune_input.output_dir), success=True)

    broker = _make_broker(
        tmp_path,
        private_dir=None,
        tune_runner=noncooperative_runner,
    )
    private_dir = broker._private_dir  # noqa: SLF001 - assert lifecycle ownership
    broker.request_sweep(_sweep_payload(broker, tmp_path))
    assert started.wait(timeout=2.0)

    broker.close(wait_for_workers=False)
    assert private_dir.is_dir()
    release.set()
    broker._workers[-1].join(timeout=2.0)  # noqa: SLF001 - wait for cleanup

    assert not private_dir.exists()


def test_verify_claim_rejects_artifacts_modified_after_the_sweep(
    tmp_path: Path,
) -> None:
    broker = _make_broker(tmp_path)
    payload = _sweep_payload(broker, tmp_path)
    view = _request_and_wait(broker, payload)
    claim = {
        "sweep_id": view["sweep_id"],
        "scenario_sha256": view["scenario_sha256"],
        "evidence_sha256": view["evidence_sha256"],
    }
    ok, reason = broker.verify_claim(claim)
    assert ok, reason

    # Rewrite the evidence packet in place: recorded digests still match the
    # claim, but the bytes on disk no longer match the record.
    evidence_path = Path(view["evidence_path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["best_score"] = 0.0
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    ok, reason = broker.verify_claim(claim)
    assert not ok
    assert "modified after" in reason


def _plant_sidecar(
    physics: Path, tmp_path: Path, *, symlink_member: bool = False
) -> Path:
    """Sidecar with a real texture and optionally a symlinked member."""
    sidecar = physics.with_name(physics.stem + "_assets")
    (sidecar / "textures").mkdir(parents=True)
    (sidecar / "textures" / "albedo.png").write_bytes(b"png-bytes")
    if symlink_member:
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"external")
        (sidecar / "swapped.bin").symlink_to(outside)
    return sidecar


def test_trusted_snapshot_stays_in_broker_private_storage(tmp_path: Path) -> None:
    """The digest-checked snapshot must live in broker-private storage (the
    original run directory stays child-writable while the sweep runs), the
    SOURCE directory reaches the tune runner as an explicit approved
    dependency root, and the localized sidecar is mirrored file-by-file
    without dereferencing planted symlinks."""

    broker = _make_broker(tmp_path)
    payload = _sweep_payload(broker, tmp_path)
    physics = Path(payload["physics_usd"])
    sidecar = _plant_sidecar(physics, tmp_path)

    view = _request_and_wait(broker, payload)
    assert view["status"] == "succeeded"

    tune_input = broker.captured_tune_inputs[0]  # type: ignore[attr-defined]
    trusted = Path(tune_input.physics_usd)
    # Broker-private, never the child-writable source directory.
    assert trusted.parent != physics.parent
    trusted.parent.relative_to((tmp_path / "private").resolve())
    # No snapshot leaf is ever created next to the original physics USD.
    strays = [
        path
        for path in physics.parent.iterdir()
        if "trusted" in path.name or path.name.startswith(".tuning-broker")
    ]
    assert strays == []
    # The source directory is admitted as an EXPLICIT approved root instead.
    assert [Path(root) for root in tune_input.approved_dependency_roots] == [
        physics.parent
    ]
    # Sidecar mirrored under its original name.
    mirrored = trusted.parent / sidecar.name / "textures" / "albedo.png"
    assert mirrored.read_bytes() == b"png-bytes"


def test_sweep_fails_closed_on_symlinked_sidecar_member(tmp_path: Path) -> None:
    """A symlinked sidecar member is tampering, not a portable export shape:
    the no-follow snapshot copy must refuse it and fail the sweep rather
    than dereference child-chosen bytes into broker storage."""

    broker = _make_broker(tmp_path)
    payload = _sweep_payload(broker, tmp_path)
    physics = Path(payload["physics_usd"])
    _plant_sidecar(physics, tmp_path, symlink_member=True)
    view = _request_and_wait(broker, payload)
    assert view["status"] == "failed"
    assert "unsafe entries" in (view.get("error") or "")


def test_materialize_localizes_the_dependency_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage.Flatten() inside patch_physics_usd can anchor asset paths to
    absolute snapshot/source locations; materialization must REWRITE those
    into the candidate's own sidecar (not just copy directories around) and
    the finished candidate must compose entirely from its own directory."""

    trials = [_fake_trial(0, 0.9), _fake_trial(1, 0.1)]
    captured_inputs: list[Any] = []

    def runner_with_history(tune_input: Any) -> Any:
        captured_inputs.append(tune_input)
        return _fake_result(Path(tune_input.output_dir), trials=trials)

    import physics_agent.tuning.usd_patch as usd_patch

    texture_holder: dict[str, Path] = {}

    def fake_patch(
        input_usd: Path,
        output_usd: Path,
        params: dict,
        *,
        bindings: Any = None,
        **_: Any,
    ) -> Path:
        # Mimic Flatten(): an ABSOLUTE asset path anchored at the broker
        # snapshot sidecar, exactly the shape plain dir-copying cannot fix.
        texture = texture_holder["texture"]
        Path(output_usd).parent.mkdir(parents=True, exist_ok=True)
        Path(output_usd).write_text(
            "#usda 1.0\n"
            'def Material "M" {\n'
            f"    asset inputs:file = @{texture}@\n"
            "}\n",
            encoding="utf-8",
        )
        return Path(output_usd)

    monkeypatch.setattr(usd_patch, "patch_physics_usd", fake_patch)
    broker = _make_broker(tmp_path, top_k=2, tune_runner=runner_with_history)
    payload = _sweep_payload(broker, tmp_path)
    physics = Path(payload["physics_usd"])
    sidecar = _plant_sidecar(physics, tmp_path)
    view = _request_and_wait(broker, payload)
    assert view["status"] == "succeeded"
    tune_input = captured_inputs[0]
    texture_holder["texture"] = (
        Path(tune_input.physics_usd).parent / sidecar.name / "textures" / "albedo.png"
    )

    entry = broker.materialize(view["sweep_id"], 1)
    candidate = Path(entry["usd_path"])
    text = candidate.read_text(encoding="utf-8")
    # The absolute reference was rewritten into the candidate's OWN sidecar.
    assert str(texture_holder["texture"]) not in text
    # Portable naming contract: "<output-name>_assets".
    assert f"./{candidate.name}_assets/" in text
    localized = list((candidate.parent / f"{candidate.name}_assets").rglob("*"))
    assert any(
        member.is_file() and member.read_bytes() == b"png-bytes" for member in localized
    )
    # And the digest covers the REWRITTEN bytes.
    from content_workflow_cli.tuning_broker import sha256_file

    assert entry["usd_sha256"] == sha256_file(candidate)


def test_sweep_fails_closed_on_fifo_sidecar_member(tmp_path: Path) -> None:
    """A pre-planted FIFO in the sidecar must fail the sweep promptly, not
    park the broker worker in a blocking open."""

    import os as _os

    broker = _make_broker(tmp_path)
    payload = _sweep_payload(broker, tmp_path)
    physics = Path(payload["physics_usd"])
    sidecar = physics.with_name(physics.stem + "_assets")
    sidecar.mkdir(parents=True)
    _os.mkfifo(sidecar / "trap.png")

    view = _request_and_wait(broker, payload)
    assert view["status"] == "failed"
    assert "unsafe entries" in (view.get("error") or "")


def test_snapshot_copy_refuses_symlinked_ancestor_directories(
    tmp_path: Path,
) -> None:
    """O_NOFOLLOW on the leaf alone is not enough: a child can replace an
    already-enumerated ancestor directory with a symlink and route the copy
    anywhere the broker can read. Every component must refuse links."""

    from content_workflow_cli.tuning_broker import _copy_regular_file_nofollow

    root = tmp_path / "source_root"
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    (secret_dir / "albedo.png").write_bytes(b"secret-bytes")
    (root / "assets").mkdir(parents=True)
    (root / "assets" / "albedo.png").write_bytes(b"real-bytes")
    dest = tmp_path / "dest" / "albedo.png"
    # Baseline: the honest layout copies.
    _copy_regular_file_nofollow(root, Path("assets/albedo.png"), dest)
    assert dest.read_bytes() == b"real-bytes"

    # Swap the ANCESTOR for a symlink after enumeration.
    (root / "assets" / "albedo.png").unlink()
    (root / "assets").rmdir()
    (root / "assets").symlink_to(secret_dir)
    with pytest.raises(BrokerError, match="link-substituted|unreadable"):
        _copy_regular_file_nofollow(
            root, Path("assets/albedo.png"), tmp_path / "dest2" / "albedo.png"
        )
    assert not (tmp_path / "dest2" / "albedo.png").exists()


def test_snapshot_copy_rejects_hard_linked_members(tmp_path: Path) -> None:
    """Multi-link regular files are rejected at the sweep-input boundary:
    the child shares the broker's UID, so fs.protected_hardlinks permits
    aliasing same-UID files from outside the approved roots into the
    sidecar, and the snapshot copy would export them. Dedupe-by-hardlink
    layouts live in parent-owned artifact stores that never feed this
    path - the sweep-input contract is plain single-link files."""

    from content_workflow_cli.tuning_broker import (
        BrokerError as _BrokerError,
    )
    from content_workflow_cli.tuning_broker import (
        _copy_regular_file_nofollow,
    )

    root = tmp_path / "source_root"
    (root / "assets").mkdir(parents=True)
    original = root / "assets" / "albedo.png"
    original.write_bytes(b"png-bytes")
    import os as _os

    _os.link(original, root / "assets" / "albedo_alias.png")
    with pytest.raises(_BrokerError, match="single-link"):
        _copy_regular_file_nofollow(
            root, Path("assets/albedo.png"), tmp_path / "dest" / "albedo.png"
        )


def test_localize_rewrites_absolute_reference_inside_candidate_dir(
    tmp_path: Path,
) -> None:
    """An absolute asset identifier that already resolves INSIDE the
    candidate directory must still be rewritten to the candidate-relative
    form: the candidate outlives broker.close() under a different path, so
    an absolute reference into the broker work directory would dangle after
    promotion even though it verifies locally."""

    candidate_dir = tmp_path / "candidates"
    sidecar = candidate_dir / "trial_0001.usda_assets"
    sidecar.mkdir(parents=True)
    texture = sidecar / "albedo.png"
    texture.write_bytes(b"png-bytes")
    candidate = candidate_dir / "trial_0001.usda"
    candidate.write_text(
        f'#usda 1.0\ndef Material "M" {{\n    asset inputs:file = @{texture}@\n}}\n',
        encoding="utf-8",
    )

    # ComputeAllDependencies at the end of the impl re-verifies
    # self-containment; it must still pass on the rewritten layer.
    PhysicsTuningBroker._localize_candidate_dependencies_impl(  # noqa: SLF001
        candidate, copy_roots=()
    )

    text = candidate.read_text(encoding="utf-8")
    assert str(texture) not in text
    # Sdf normalizes the "./" prefix away when saving.
    assert "trial_0001.usda_assets/albedo.png" in text


def test_materialized_candidate_closure_is_exported_with_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A localized root references its broker-private sidecar; the closure
    listing and file endpoints must expose exactly those members so the
    exported candidate composes outside the broker workspace."""

    trials = [_fake_trial(0, 0.9), _fake_trial(1, 0.1)]
    captured_inputs: list[Any] = []

    def runner_with_history(tune_input: Any) -> Any:
        captured_inputs.append(tune_input)
        return _fake_result(Path(tune_input.output_dir), trials=trials)

    import physics_agent.tuning.usd_patch as usd_patch

    texture_holder: dict[str, Path] = {}

    def fake_patch(
        input_usd: Path,
        output_usd: Path,
        params: dict,
        *,
        bindings: Any = None,
        **_: Any,
    ) -> Path:
        texture = texture_holder["texture"]
        Path(output_usd).parent.mkdir(parents=True, exist_ok=True)
        Path(output_usd).write_text(
            "#usda 1.0\n"
            'def Material "M" {\n'
            f"    asset inputs:file = @{texture}@\n"
            "}\n",
            encoding="utf-8",
        )
        return Path(output_usd)

    monkeypatch.setattr(usd_patch, "patch_physics_usd", fake_patch)
    broker = _make_broker(tmp_path, top_k=2, tune_runner=runner_with_history)
    payload = _sweep_payload(broker, tmp_path)
    physics = Path(payload["physics_usd"])
    sidecar = _plant_sidecar(physics, tmp_path)
    view = _request_and_wait(broker, payload)
    assert view["status"] == "succeeded"
    tune_input = captured_inputs[0]
    texture_holder["texture"] = (
        Path(tune_input.physics_usd).parent / sidecar.name / "textures" / "albedo.png"
    )

    entry = broker.materialize(view["sweep_id"], 1)
    candidate = Path(entry["usd_path"])
    members = broker.materialized_candidate_closure(view["sweep_id"], "1")
    assert len(members) == 1
    member = members[0]
    assert member["index"] == 0
    relative = str(member["relative_path"])
    assert relative.startswith(f"{candidate.name}_assets/")
    member_path = candidate.parent / relative
    assert member_path.read_bytes() == b"png-bytes"
    assert member["sha256"] == sha256_file(member_path)

    served_path, served_digest = broker.materialized_candidate_closure_file(
        view["sweep_id"], "1", "0"
    )
    assert served_path == member_path
    assert served_digest == member["sha256"]

    with pytest.raises(BrokerError):
        broker.materialized_candidate_closure_file(view["sweep_id"], "1", "5")


def test_materialize_exempts_resolver_owned_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bare MDL tokens and non-file resolver URIs intentionally stay
    runtime-resolved: localization must leave them unrewritten and the
    self-containment verification must not fail the candidate over them."""

    trials = [_fake_trial(0, 0.9), _fake_trial(1, 0.1)]

    def runner_with_history(tune_input: Any) -> Any:
        return _fake_result(Path(tune_input.output_dir), trials=trials)

    import physics_agent.tuning.usd_patch as usd_patch

    def fake_patch(
        input_usd: Path,
        output_usd: Path,
        params: dict,
        *,
        bindings: Any = None,
        **_: Any,
    ) -> Path:
        Path(output_usd).parent.mkdir(parents=True, exist_ok=True)
        Path(output_usd).write_text(
            "#usda 1.0\n"
            'def Material "M" {\n'
            "    asset inputs:mdl = @OmniPBR.mdl@\n"
            "    asset inputs:remote = @omniverse://server/materials/base.usd@\n"
            "}\n",
            encoding="utf-8",
        )
        return Path(output_usd)

    monkeypatch.setattr(usd_patch, "patch_physics_usd", fake_patch)
    broker = _make_broker(tmp_path, top_k=2, tune_runner=runner_with_history)
    payload = _sweep_payload(broker, tmp_path)
    view = _request_and_wait(broker, payload)
    assert view["status"] == "succeeded"

    entry = broker.materialize(view["sweep_id"], 1)
    text = Path(entry["usd_path"]).read_text(encoding="utf-8")
    assert "OmniPBR.mdl" in text
    assert "omniverse://server/materials/base.usd" in text


def test_sweep_input_gate_exempts_resolver_owned_layers(tmp_path: Path) -> None:
    """_assert_self_contained_usd applies the same resolver-owned exemption
    as localization/promotion: a bare MDL token or non-file URI in the
    unresolved list is not a self-containment violation."""

    from content_workflow_cli.tuning_broker import _assert_self_contained_usd

    physics = tmp_path / "physics.usda"
    physics.write_text(
        "#usda 1.0\n"
        'def Material "M" {\n'
        "    asset inputs:mdl = @OmniPBR.mdl@\n"
        "    asset inputs:remote = @omniverse://server/materials/base.usd@\n"
        "}\n",
        encoding="utf-8",
    )
    _assert_self_contained_usd(physics)  # must not raise
