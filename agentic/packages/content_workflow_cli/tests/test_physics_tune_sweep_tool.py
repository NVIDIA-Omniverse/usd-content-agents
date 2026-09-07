# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-facing sweep client CLI tests against a live broker."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from content_workflow_cli.agent_tools import physics_tune_sweep
from content_workflow_cli.tuning_broker import PhysicsTuningBroker


def _fake_tune_runner(tune_input: Any) -> Any:
    output_dir = Path(tune_input.output_dir)
    tuned = output_dir / "tuned_physics.usd"
    tuned.parent.mkdir(parents=True, exist_ok=True)
    tuned.write_text("#usda 1.0\n", encoding="utf-8")
    recording = output_dir / ".tune_scenes" / "trial_seed_42" / "recording.usd"
    recording.parent.mkdir(parents=True, exist_ok=True)
    recording.write_text("#usda 1.0\n# exact scenario rollout\n", encoding="utf-8")
    # Resolved-bindings artifact required for exact candidate materialization.
    (output_dir / "tune_results.json").write_text(
        json.dumps({"scenario": {"extra": {}}}), encoding="utf-8"
    )
    trial = SimpleNamespace(
        trial_index=0,
        params={"restitution": 0.8},
        score=0.2,
        backend_metrics={"recording_usd": str(recording)},
        failed=False,
        error=None,
    )
    return SimpleNamespace(
        success=True,
        cancelled=False,
        error=None,
        history=[trial],
        artifacts={"tuned_physics.usd": tuned},
        best_params=dict(trial.params),
        best_score=trial.score,
        n_trials=1,
        optimizer_used="random",
        engine_used="fake",
        needs_refinement=False,
    )


def _start_broker(tmp_path: Path, **overrides: Any) -> PhysicsTuningBroker:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "run_dir": run_dir,
        "engine": "fake",
        "optimizer": "random",
        "max_sweeps": 1,
        "max_trials_per_sweep": 5,
        "private_dir": tmp_path / "private",
        "tune_runner": _fake_tune_runner,
    }
    kwargs.update(overrides)
    broker = PhysicsTuningBroker(**kwargs)
    broker.start()
    return broker


def _inputs(tmp_path: Path, run_dir: Path) -> tuple[Path, Path, Path]:
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        yaml.safe_dump(
            {
                "name": "drop_settle",
                "metric": "settle_distance",
                "parameters": [{"name": "restitution", "min": 0.0, "max": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    physics = tmp_path / "physics.usda"
    physics.write_text("#usda 1.0\n", encoding="utf-8")
    return scenario, physics, run_dir / "tuning" / "iter_1"


def test_run_subcommand_waits_for_evidence_and_exits_zero(
    tmp_path: Path, capsys: Any
) -> None:
    broker = _start_broker(tmp_path)
    try:
        scenario, physics, iter_dir = _inputs(tmp_path, broker.run_dir)
        exit_code = physics_tune_sweep.main(
            [
                "--broker-url",
                broker.url,
                "--poll-seconds",
                "0.05",
                "run",
                "--scenario",
                str(scenario),
                "--physics-usd",
                str(physics),
                "--output-dir",
                str(iter_dir),
            ]
        )
        assert exit_code == physics_tune_sweep.EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "succeeded"
        assert Path(payload["evidence_path"]).is_file()
        assert payload["scenario_sha256"]
    finally:
        broker.close()


def test_run_subcommand_reports_budget_refusal_as_exit_three(
    tmp_path: Path, capsys: Any
) -> None:
    broker = _start_broker(tmp_path, max_sweeps=0)
    try:
        scenario, physics, iter_dir = _inputs(tmp_path, broker.run_dir)
        exit_code = physics_tune_sweep.main(
            [
                "--broker-url",
                broker.url,
                "run",
                "--scenario",
                str(scenario),
                "--physics-usd",
                str(physics),
                "--output-dir",
                str(iter_dir),
            ]
        )
        assert exit_code == physics_tune_sweep.EXIT_BUDGET_REFUSED
        payload = json.loads(capsys.readouterr().out)
        assert "budget exhausted" in payload["error"]
    finally:
        broker.close()


def test_broker_unreachable_exits_six(tmp_path: Path, capsys: Any) -> None:
    scenario, physics, iter_dir = _inputs(tmp_path, tmp_path / "run")
    exit_code = physics_tune_sweep.main(
        [
            "--broker-url",
            "http://127.0.0.1:9",
            "--request-timeout",
            "0.2",
            "run",
            "--scenario",
            str(scenario),
            "--physics-usd",
            str(physics),
            "--output-dir",
            str(iter_dir),
        ]
    )
    assert exit_code == physics_tune_sweep.EXIT_BROKER_UNREACHABLE
    assert "unreachable" in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--poll-seconds", "0"),
        ("--poll-seconds", "nan"),
        ("--sweep-deadline-seconds", "-1"),
    ],
)
def test_parser_rejects_nonpositive_or_nonfinite_polling_values(
    option: str, value: str
) -> None:
    with pytest.raises(SystemExit):
        physics_tune_sweep.build_parser().parse_args(
            [
                "--broker-url",
                "http://127.0.0.1:1",
                option,
                value,
                "run",
                "--scenario",
                "scenario.yaml",
                "--physics-usd",
                "physics.usda",
                "--output-dir",
                "output",
            ]
        )


def test_materialize_subcommand_returns_candidate_usd(
    tmp_path: Path, capsys: Any, monkeypatch: Any
) -> None:
    import physics_agent.tuning.usd_patch as usd_patch_module

    def fake_patch(
        input_usd: Path, output_usd: Path, params: dict, **_kwargs: Any
    ) -> Path:
        Path(output_usd).parent.mkdir(parents=True, exist_ok=True)
        Path(output_usd).write_text("#usda 1.0\n# tuned\n", encoding="utf-8")
        return Path(output_usd)

    monkeypatch.setattr(usd_patch_module, "patch_physics_usd", fake_patch)
    broker = _start_broker(tmp_path)
    try:
        scenario, physics, iter_dir = _inputs(tmp_path, broker.run_dir)
        assert (
            physics_tune_sweep.main(
                [
                    "--broker-url",
                    broker.url,
                    "--poll-seconds",
                    "0.05",
                    "run",
                    "--scenario",
                    str(scenario),
                    "--physics-usd",
                    str(physics),
                    "--output-dir",
                    str(iter_dir),
                ]
            )
            == physics_tune_sweep.EXIT_OK
        )
        sweep_id = json.loads(capsys.readouterr().out)["sweep_id"]
        exit_code = physics_tune_sweep.main(
            [
                "--broker-url",
                broker.url,
                "materialize",
                "--sweep-id",
                sweep_id,
                "--trial-index",
                "0",
            ]
        )
        assert exit_code == physics_tune_sweep.EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert Path(payload["usd_path"]).is_file()
        assert len(payload["usd_sha256"]) == 64

        exported = iter_dir / "review" / f"trial_0{Path(payload['usd_path']).suffix}"
        exported_recording = iter_dir / "review" / "trial_0_recording.usd"
        exit_code = physics_tune_sweep.main(
            [
                "--broker-url",
                broker.url,
                "materialize",
                "--sweep-id",
                sweep_id,
                "--trial-index",
                "0",
                "--output-usd",
                str(exported),
                "--output-recording",
                str(exported_recording),
            ]
        )
        assert exit_code == physics_tune_sweep.EXIT_OK
        exported_payload = json.loads(capsys.readouterr().out)
        assert exported_payload["exported_usd_path"] == str(exported.resolve())
        assert exported_payload["exported_usd_sha256"] == payload["usd_sha256"]
        assert exported.read_text(encoding="utf-8") == "#usda 1.0\n# tuned\n"
        assert exported_payload["exported_recording_path"] == str(
            exported_recording.resolve()
        )
        assert len(exported_payload["exported_recording_sha256"]) == 64
        assert (
            exported_recording.read_text(encoding="utf-8")
            == "#usda 1.0\n# exact scenario rollout\n"
        )

        exported_bytes = exported.read_bytes()
        assert (
            physics_tune_sweep.main(
                [
                    "--broker-url",
                    broker.url,
                    "materialize",
                    "--sweep-id",
                    sweep_id,
                    "--trial-index",
                    "0",
                    "--output-usd",
                    str(exported),
                ]
            )
            == physics_tune_sweep.EXIT_BAD_INPUT
        )
        assert "candidate export failed" in json.loads(capsys.readouterr().out)["error"]
        assert exported.read_bytes() == exported_bytes
    finally:
        broker.close()


def test_materialize_exports_the_localized_sidecar_closure(
    tmp_path: Path, capsys: Any, monkeypatch: Any
) -> None:
    """`materialize --output-usd` must export the dependency closure with
    the root: a localized candidate references its broker-private
    "*_assets" sidecar relatively and cannot compose from the root alone."""

    import physics_agent.tuning.usd_patch as usd_patch_module
    from pxr import UsdUtils

    captured_inputs: list[Any] = []

    def capturing_tune_runner(tune_input: Any) -> Any:
        captured_inputs.append(tune_input)
        return _fake_tune_runner(tune_input)

    texture_holder: dict[str, Path] = {}

    def fake_patch(
        input_usd: Path, output_usd: Path, params: dict, **_kwargs: Any
    ) -> Path:
        # Absolute reference into the broker snapshot: the shape only
        # closure localization (not directory copying) can export.
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

    monkeypatch.setattr(usd_patch_module, "patch_physics_usd", fake_patch)
    broker = _start_broker(tmp_path, tune_runner=capturing_tune_runner)
    try:
        scenario, physics, iter_dir = _inputs(tmp_path, broker.run_dir)
        sidecar = physics.with_name(physics.stem + "_assets")
        (sidecar / "textures").mkdir(parents=True)
        (sidecar / "textures" / "albedo.png").write_bytes(b"png-bytes")
        assert (
            physics_tune_sweep.main(
                [
                    "--broker-url",
                    broker.url,
                    "--poll-seconds",
                    "0.05",
                    "run",
                    "--scenario",
                    str(scenario),
                    "--physics-usd",
                    str(physics),
                    "--output-dir",
                    str(iter_dir),
                ]
            )
            == physics_tune_sweep.EXIT_OK
        )
        sweep_id = json.loads(capsys.readouterr().out)["sweep_id"]
        texture_holder["texture"] = (
            Path(captured_inputs[0].physics_usd).parent
            / sidecar.name
            / "textures"
            / "albedo.png"
        )
        exported = iter_dir / "review" / "trial_0.usda"
        exit_code = physics_tune_sweep.main(
            [
                "--broker-url",
                broker.url,
                "materialize",
                "--sweep-id",
                sweep_id,
                "--trial-index",
                "0",
                "--output-usd",
                str(exported),
            ]
        )
        assert exit_code == physics_tune_sweep.EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        members = payload["exported_sidecar_members"]
        assert len(members) == 1
        exported_member = Path(members[0]["path"])
        assert exported_member.read_bytes() == b"png-bytes"
        # The exported bundle composes OUTSIDE the broker workspace: every
        # dependency resolves inside the export directory.
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(exported))
        assert not unresolved
        export_root = exported.parent.resolve()
        for identifier in [layer.identifier for layer in layers] + [
            str(item) for item in assets
        ]:
            Path(identifier).resolve().relative_to(export_root)
    finally:
        broker.close()
