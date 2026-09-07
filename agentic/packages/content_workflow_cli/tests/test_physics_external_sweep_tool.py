# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-facing external sweep client CLI tests against a live broker."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from content_workflow_cli.agent_tools import physics_external_sweep
from content_workflow_cli.external_tuning_broker import ExternalTuningBroker


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
        parameter_catalog=(QualifiedParameter(name="restitution"),),
        params=(TunableParam(name="restitution", min_value=0.0, max_value=1.0),),
        objective=TuningObjective(name="grasp_slip", unit="m"),
        optimizer=OptimizerSettings(name="random", max_trials=4, seed=7),
        qualification=QualificationSettings(nominal_params={"restitution": 0.5}),
        evidence=EvidenceSettings(),
    )


def _fake_runner(tune_input: Any) -> Any:
    output_dir = Path(tune_input.output_dir)
    render_dir = output_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    frame = render_dir / "frame_0000.png"
    frame.write_bytes(b"PNGDATA")
    (render_dir / "render_response_metadata.json").write_text(
        '{"renderer": "ovrtx"}', encoding="utf-8"
    )
    # Frames are rendered from the winner recording, so a succeeded sweep
    # always has one; the broker refuses evidence without it.
    recording = output_dir / "best_recording.usd"
    recording.write_text("#usda 1.0\n", encoding="utf-8")
    # Core tune artifacts are mandatory for a succeeded sweep: the broker
    # pins their digests when publishing evidence.
    (output_dir / "run_spec.json").write_text("{}", encoding="utf-8")
    (output_dir / "best_params.json").write_text(
        json.dumps({"restitution": 0.8}), encoding="utf-8"
    )
    (output_dir / "external_tune_results.json").write_text(
        json.dumps({"artifacts": {}}), encoding="utf-8"
    )
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
        best_params=dict(trial.params),
        best_objective=0.05,
        best_score=0.05,
        n_trials=1,
        history=[trial],
        rendered_frames=[frame],
        artifacts={"best_recording": recording},
        selected_evidence={
            "published_recording": {
                "path": "best_recording.usd",
                "sha256": "sha256:"
                + hashlib.sha256(recording.read_bytes()).hexdigest(),
            }
        },
    )


@pytest.fixture()
def broker(tmp_path: Path) -> Any:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    broker = ExternalTuningBroker(
        run_dir=run_dir,
        spec=_make_spec(tmp_path),
        qualification_dir=run_dir / "qualification",
        approval_digest="sha256:" + "0" * 64,
        max_sweeps=1,
        max_trials_per_sweep=4,
        private_dir=tmp_path / "private",
        tune_runner=_fake_runner,
    )
    broker.start()
    yield broker
    broker.close()


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, Any]:
    code = physics_external_sweep.main(argv)
    output = capsys.readouterr().out
    return code, json.loads(output)


def test_run_sweep_success(broker: Any, capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _run(
        [
            "--broker-url",
            broker.url,
            "--poll-seconds",
            "0.05",
            "run",
            "--output-dir",
            str(broker.run_dir / "tuning" / "iter_1"),
            "--active-search",
            json.dumps({"restitution": {"min": 0.2, "max": 0.9}}),
            "--max-trials",
            "2",
        ],
        capsys,
    )
    assert code == physics_external_sweep.EXIT_OK
    assert payload["status"] == "succeeded"
    assert payload["active_search"] == {"restitution": {"min": 0.2, "max": 0.9}}
    assert payload["max_trials"] == 2
    assert payload["frames"]


def test_budget_refusal_exit_code(
    broker: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(
        [
            "--broker-url",
            broker.url,
            "--poll-seconds",
            "0.05",
            "run",
            "--output-dir",
            str(broker.run_dir / "tuning" / "iter_1"),
        ],
        capsys,
    )
    assert code == physics_external_sweep.EXIT_OK
    code, payload = _run(
        [
            "--broker-url",
            broker.url,
            "run",
            "--output-dir",
            str(broker.run_dir / "tuning" / "iter_2"),
        ],
        capsys,
    )
    assert code == physics_external_sweep.EXIT_BUDGET_REFUSED
    assert "budget exhausted" in payload["error"]


def test_budget_subcommand(broker: Any, capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _run(["--broker-url", broker.url, "budget"], capsys)
    assert code == physics_external_sweep.EXIT_OK
    assert payload["max_sweeps"] == 1


def test_list_subcommand_recovers_reserved_sweeps(
    broker: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """A client whose blocking run call was killed must still cite the sweep
    it reserved: `list` recovers every reserved sweep's id and record without
    reserving new budget."""

    code, payload = _run(
        [
            "--broker-url",
            broker.url,
            "--poll-seconds",
            "0.05",
            "run",
            "--output-dir",
            str(broker.run_dir / "tuning" / "iter_1"),
        ],
        capsys,
    )
    assert code == physics_external_sweep.EXIT_OK
    code, listing = _run(["--broker-url", broker.url, "list"], capsys)
    assert code == physics_external_sweep.EXIT_OK
    sweeps = listing["sweeps"]
    assert len(sweeps) == 1
    assert sweeps[0]["sweep_id"] == payload["sweep_id"]
    assert sweeps[0]["status"] == "succeeded"


def test_invalid_active_search_json(
    broker: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = _run(
        [
            "--broker-url",
            broker.url,
            "run",
            "--output-dir",
            str(broker.run_dir / "tuning" / "iter_1"),
            "--active-search",
            "not-json",
        ],
        capsys,
    )
    assert code == physics_external_sweep.EXIT_BAD_INPUT
    assert "not valid JSON" in payload["error"]


class _StubResponse:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def test_poll_error_body_fails_fast(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A broker error body (404/500 {"error": ...}) has no terminal status
    and never will; the client must exit with the bad-input code on the
    first such poll instead of burning the full polling budget (~4 h at the
    defaults) before reporting it."""

    polls: list[str] = []

    monkeypatch.setattr(
        physics_external_sweep.requests,
        "post",
        lambda url, **_: _StubResponse(200, {"sweep_id": "s1", "status": "running"}),
    )

    def fake_get(url: str, **_: Any) -> _StubResponse:
        if url.endswith("/budget"):
            # The poll bound is derived from the broker's own deadline.
            return _StubResponse(200, {"sweep_deadline_seconds": 1.0})
        polls.append(url)
        return _StubResponse(404, {"error": "unknown sweep_id 's1'"})

    monkeypatch.setattr(physics_external_sweep.requests, "get", fake_get)
    code, payload = _run(
        [
            "--broker-url",
            "http://127.0.0.1:1",
            "--poll-seconds",
            "0.01",
            "run",
            "--output-dir",
            "/tmp/nowhere",
        ],
        capsys,
    )
    assert code == physics_external_sweep.EXIT_BAD_INPUT
    assert payload["error"] == "unknown sweep_id 's1'"
    assert len(polls) == 1


def test_non_json_broker_response_maps_to_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-JSON body (an HTML proxy error page) must map to the
    broker-unreachable exit code instead of raising JSONDecodeError out of
    main()."""

    monkeypatch.setattr(
        physics_external_sweep.requests,
        "post",
        lambda url, **_: _StubResponse(502, ValueError("not JSON")),
    )
    code, payload = _run(
        ["--broker-url", "http://127.0.0.1:1", "run", "--output-dir", "/tmp/nowhere"],
        capsys,
    )
    assert code == physics_external_sweep.EXIT_BROKER_UNREACHABLE
    assert "non-JSON" in payload["error"]

    monkeypatch.setattr(
        physics_external_sweep.requests,
        "get",
        lambda url, **_: _StubResponse(502, ValueError("not JSON")),
    )
    code, payload = _run(["--broker-url", "http://127.0.0.1:1", "budget"], capsys)
    assert code == physics_external_sweep.EXIT_BROKER_UNREACHABLE
    assert "non-JSON" in payload["error"]


def test_broker_unreachable(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _run(
        [
            "--broker-url",
            "http://127.0.0.1:9",
            "--request-timeout",
            "0.2",
            "run",
            "--output-dir",
            "/tmp/nowhere",
        ],
        capsys,
    )
    assert code == physics_external_sweep.EXIT_BROKER_UNREACHABLE
    assert "unreachable" in payload["error"]
