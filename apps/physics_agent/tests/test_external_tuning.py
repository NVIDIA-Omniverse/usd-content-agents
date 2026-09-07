# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract and workflow tests for trusted local external tuning."""

from __future__ import annotations

import asyncio
import copy
import json
import subprocess
import sys
import textwrap
import venv
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from physics_agent.tuning import (
    OptimizationOutput,
    OptimizerSettings,
    ReplicaRecord,
    TrialRecord,
    TunableParam,
    TuningCancelledError,
    TuningConfigError,
    TuningError,
    TuningObjective,
)
from physics_agent.tuning.external import (
    ExternalTuneConfigError,
    ExternalTuneInput,
    ExternalTuneOutput,
    ExternalTuneSpec,
    arun_external_tune,
    load_external_tune_spec,
    run_external_tune,
)
from physics_agent.tuning.external.artifacts import (
    collect_public_tune_artifacts,
    file_descriptor,
)
from physics_agent.tuning.external.backend import (
    ExternalTrialError,
    ExternalTrialExecutor,
)
from physics_agent.tuning.external.fingerprint import (
    _python_environment,
    build_runtime_fingerprint,
)


@pytest.fixture(autouse=True)
def lightweight_runtime_environment_fingerprint(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Avoid hashing the test environment except in probe-specific tests."""

    real_probe_tests = {
        "test_python_environment_fingerprint_detects_installed_file_edits",
        "test_python_environment_fingerprint_fails_closed",
        "test_python_environment_fingerprint_uses_declared_runtime",
        "test_runtime_fingerprint_reports_process_spawn_failure",
    }
    if request.node.name in real_probe_tests:
        yield
        return

    from physics_agent.tuning.external import fingerprint

    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        fingerprint,
        "_python_environment",
        lambda spec: {
            "executable": str(spec.runtime.python),
            "packages": [],
        },
    )
    try:
        yield
    finally:
        patcher.undo()


def _installed_package(environment: dict[str, Any], *, name: str) -> dict[str, Any]:
    return next(
        package for package in environment["packages"] if package["name"] == name
    )


def test_trial_record_preserves_replica_outcomes() -> None:
    replicas = [
        ReplicaRecord(seed=1, objective_value=0.2, success=True),
        ReplicaRecord(
            seed=2,
            objective_value=None,
            success=False,
            error="simulation failed",
        ),
    ]
    record = TrialRecord(
        trial_index=3,
        params={"gain": 0.4},
        score=1.0e12,
        replicas=replicas,
    )

    assert not record.success
    assert record.to_dict()["replicas"] == [replica.to_dict() for replica in replicas]


@pytest.fixture(autouse=True)
def playback_render_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Render deterministic test PNGs without starting OvRTX."""

    from PIL import Image, ImageDraw
    from world_understanding.functions import graphics

    calls: list[Path] = []

    def render_time_sampled_usd(
        recording_path: Path,
        output_dir: Path,
        **kwargs: object,
    ) -> list[Path]:
        calls.append(Path(recording_path).resolve())
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        width = int(kwargs["image_width"])
        height = int(kwargs["image_height"])
        frame_paths: list[Path] = []
        for index in range(6):
            frame = Image.new("RGB", (width, height), (0, 0, 0))
            draw = ImageDraw.Draw(frame)
            start = 4 + index * 3
            draw.rectangle((start, 8, start + 12, 24), fill=(20, 180, 240))
            frame_path = destination / f"frame_{index:04d}.png"
            frame.save(frame_path)
            frame_paths.append(frame_path)
        return frame_paths

    monkeypatch.setattr(graphics, "render_time_sampled_usd", render_time_sampled_usd)
    return calls


def _write_trial_adapter(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            r"""
            import argparse
            import json
            import time
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--request", required=True)
            parser.add_argument("--result", required=True)
            args = parser.parse_args()

            request = json.loads(Path(args.request).read_text())
            value = float(request["params"]["gain"])
            trial = request.get("trial", {})
            if trial.get("log_bytes"):
                print("x" * int(trial["log_bytes"]))
            if trial.get("sleep_s"):
                time.sleep(float(trial["sleep_s"]))
            artifacts_dir = Path(request["artifacts_dir"])
            artifact = artifacts_dir / "metrics.json"
            artifact.write_text(json.dumps({"gain": value, "seed": request["seed"]}))
            artifacts = {"metrics": str(artifact)}
            metadata = {"applied_params": request["params"]}

            evidence = request.get("evidence")
            missing_evidence = trial.get("missing_evidence")
            if evidence is not None and not missing_evidence:
                manifest = artifacts_dir / f"{request['purpose']}_frames.json"
                if trial.get("corrupt_evidence"):
                    manifest.write_text("not-json")
                else:
                    from PIL import Image, ImageDraw

                    width = int(evidence["width"])
                    height = int(evidence["height"])
                    fps = float(evidence["fps"])
                    frame_dir = artifacts_dir / f"{request['purpose']}_frames"
                    frame_dir.mkdir(parents=True, exist_ok=True)
                    frames = []
                    for index in range(6):
                        frame = Image.new("RGB", (width, height), (0, 0, 0))
                        if not trial.get("blank_evidence"):
                            draw = ImageDraw.Draw(frame)
                            start = 4 + index * 3
                            draw.rectangle(
                                (start, 8, start + 12, 24), fill=(20, 180, 240)
                            )
                        frame_path = frame_dir / f"frame_{index:04d}.png"
                        frame.save(frame_path)
                        frames.append(
                            {
                                "path": frame_path.relative_to(artifacts_dir).as_posix(),
                                "timestamp_seconds": index / fps,
                            }
                        )
                    manifest.write_text(
                        json.dumps(
                            {
                                "schema_version": "physics-agent.qualification-frames.v1",
                                "renderer": evidence["renderer"],
                                "width": width,
                                "height": height,
                                "fps": fps,
                                "frames": frames,
                            }
                        )
                    )
                artifacts[evidence["artifact_name"]] = str(manifest)
                metadata["evidence"] = {
                    "renderer": (
                        "wrong_renderer"
                        if trial.get("wrong_renderer")
                        else evidence["renderer"]
                    ),
                    "width": evidence["width"],
                    "height": evidence["height"],
                    "fps": evidence["fps"],
                }

            recording = request.get("recording")
            if recording is not None and not trial.get("missing_recording"):
                recording_path = artifacts_dir / "recording.usda"
                if trial.get("corrupt_recording"):
                    recording_path.write_text("not a USD")
                else:
                    fps = float(recording["fps"])
                    recording_path.write_text(
                        "#usda 1.0\n"
                        "(\n"
                        "    startTimeCode = 0\n"
                        "    endTimeCode = 5\n"
                        f"    framesPerSecond = {fps}\n"
                        f"    timeCodesPerSecond = {fps}\n"
                        ")\n"
                        'def Xform "World"\n'
                        "{\n"
                        '    def Cube "Cube"\n'
                        "    {\n"
                        "        double3 xformOp:translate.timeSamples = {\n"
                        "            0: (0, 0, 1),\n"
                        "            1: (0, 0, 0.8),\n"
                        "            2: (0, 0, 0.6),\n"
                        "            3: (0, 0, 0.4),\n"
                        "            4: (0, 0, 0.2),\n"
                        "            5: (0, 0, 0.1),\n"
                        "        }\n"
                        "        quatf xformOp:orient.timeSamples = {\n"
                        "            0: (1, 0, 0, 0),\n"
                        "            1: (1, 0, 0, 0),\n"
                        "            2: (1, 0, 0, 0),\n"
                        "            3: (1, 0, 0, 0),\n"
                        "            4: (1, 0, 0, 0),\n"
                        "            5: (1, 0, 0, 0),\n"
                        "        }\n"
                        "        uniform token[] xformOpOrder = [\n"
                        '            "xformOp:translate", "xformOp:orient"\n'
                        "        ]\n"
                        "    }\n"
                        '    def Camera "PlaybackCamera"\n'
                        "    {\n"
                        "        double3 xformOp:translate = (1, -1, 1)\n"
                        '        uniform token[] xformOpOrder = ["xformOp:translate"]\n'
                        "    }\n"
                        "}\n"
                    )
                artifacts[recording["artifact_name"]] = str(recording_path)
                metadata["recording"] = {
                    "media_type": recording["media_type"],
                    "fps": recording["fps"],
                    "frame_count": 6,
                }

            if trial.get("failed_low_objective") and value < 0.5:
                objective_value = 0.0
                success = False
            else:
                objective_value = abs(value - 0.8)
                success = True
            objective = dict(request["objective"])
            objective["value"] = objective_value
            if trial.get("wrong_objective"):
                objective["name"] = "changed_objective"
            artifact_path = artifact
            if trial.get("escape_artifact"):
                artifact_path = artifacts_dir.parent / "escaped.json"
                artifact_path.write_text("{}")
            metric_distance = (
                -objective_value
                if trial.get("misleading_metric")
                else objective_value
            )
            metrics = {"distance": metric_distance}
            if trial.get("non_finite_metric"):
                metrics["bad"] = float("nan")
            result = {
                "status": "ok",
                "objective": objective,
                "success": success,
                "metrics": metrics,
                "metadata": metadata,
                "artifacts": {**artifacts, "metrics": str(artifact_path)},
            }
            if trial.get("missing_objective"):
                result.pop("objective")
            Path(args.result).write_text(json.dumps(result))
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _config(script: Path, **trial: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "task": "quadratic_contract",
        "runtime": {
            "python": sys.executable,
            "script": str(script),
            "cwd": str(script.parent),
            "timeout_s": 10,
            "fingerprint_paths": [str(script)],
            "trial": trial,
        },
        "parameters": {"gain": {"min": 0.0, "max": 1.0}},
        "objective": {
            "name": "absolute_error",
            "unit": "normalized",
            "direction": "minimize",
        },
        "optimizer": {
            "name": "random",
            "max_trials": 12,
            "seed": 7,
            "replicas": 2,
        },
        "qualification": {
            "nominal_params": {"gain": 0.75},
            "seed": 100,
        },
        "evidence": {
            "artifact_name": "frames",
            "renderer": "isaac_sim_kit_rtx",
            "media_type": "application/json",
            "width": 64,
            "height": 48,
            "fps": 10,
            "min_frames": 3,
            "require_motion": True,
            "min_frame_stddev": 1.0,
            "min_motion_score": 0.1,
            "recording_artifact_name": "recording_usd",
            "playback_renderer": "ovrtx",
            "max_duration_seconds": 1.0,
            "num_sensor_updates": 1,
            "render_mode": "rt2",
            "camera": {
                "position": [1.0, -1.0, 0.8],
                "target": [0.0, 0.0, 0.2],
            },
        },
    }


def _evidence_config(script: Path, **trial: object) -> dict[str, object]:
    return _config(script, **trial)


def _qualify_and_run(
    config: dict[str, object],
    output_dir: Path,
    *,
    render_winning_trial: bool = False,
):
    qualified = run_external_tune(
        ExternalTuneInput(config=config, output_dir=output_dir)
    )
    assert qualified.success
    assert qualified.status == "awaiting_approval"
    assert qualified.qualification_digest
    return run_external_tune(
        ExternalTuneInput(
            config=config,
            output_dir=output_dir,
            approval_digest=qualified.qualification_digest,
            render_winning_trial=render_winning_trial,
        )
    )


def test_external_tuning_reuses_shared_tuning_contracts(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)

    spec = load_external_tune_spec(_config(script))

    assert isinstance(spec.params[0], TunableParam)
    assert isinstance(spec.objective, TuningObjective)
    assert isinstance(spec.optimizer, OptimizerSettings)
    assert isinstance(ExternalTuneOutput(success=True), OptimizationOutput)
    assert issubclass(ExternalTuneConfigError, TuningConfigError)
    assert issubclass(ExternalTrialError, TuningError)


def test_external_tuning_requires_frames_and_recording_contract(
    tmp_path: Path,
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    del config["evidence"]

    with pytest.raises(
        ExternalTuneConfigError,
        match="requires qualification PNG-frame and rollout recording",
    ):
        load_external_tune_spec(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fps", 61, "fps must be positive and <= 60"),
        ("playback_renderer", "warp", "playback_renderer must be remote or ovrtx"),
        ("playback_renderer", "nvcf", "playback_renderer must be remote or ovrtx"),
    ],
)
def test_external_tuning_rejects_unrenderable_recording_settings(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    config["evidence"][field] = value  # type: ignore[index]

    with pytest.raises(ExternalTuneConfigError, match=message):
        load_external_tune_spec(config)


def test_external_tuning_accepts_remote_playback_renderer(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    config["evidence"]["playback_renderer"] = "remote"  # type: ignore[index]

    spec = load_external_tune_spec(config)

    assert spec.evidence is not None
    assert spec.evidence.playback_renderer == "remote"


def test_external_tuning_requires_qualification_approval(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output_dir = tmp_path / "run"

    qualified = run_external_tune(
        ExternalTuneInput(config=_config(script), output_dir=output_dir)
    )

    assert qualified.success
    assert qualified.status == "awaiting_approval"
    assert qualified.qualification_path == output_dir / "qualification.json"
    assert not (output_dir / "optimization").exists()


def test_approval_allows_changed_search_and_optimizer_settings(
    tmp_path: Path,
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output_dir = tmp_path / "run"
    qualified_config = _config(script)
    qualified = run_external_tune(
        ExternalTuneInput(config=qualified_config, output_dir=output_dir)
    )
    updated_config = copy.deepcopy(qualified_config)
    updated_config["parameters"] = {"gain": {"min": 0.25, "max": 0.9}}
    updated_config["optimizer"] = {
        "name": "random",
        "max_trials": 2,
        "seed": 23,
        "replicas": 1,
    }

    result = run_external_tune(
        ExternalTuneInput(
            config=updated_config,
            output_dir=output_dir,
            approval_digest=qualified.qualification_digest,
        )
    )

    assert result.success
    assert result.status == "completed"
    assert result.n_trials == 2
    assert all(0.25 <= row.params["gain"] <= 0.9 for row in result.history)


def test_qualification_publishes_digest_bound_frame_evidence(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output_dir = tmp_path / "run"

    qualified = run_external_tune(
        ExternalTuneInput(config=_evidence_config(script), output_dir=output_dir)
    )

    assert qualified.success
    assert qualified.status == "awaiting_approval"
    assert qualified.artifacts["qualification_frames"] == (
        output_dir / "qualification_frames.json"
    )
    qualification = json.loads(qualified.qualification_path.read_text(encoding="utf-8"))
    descriptor = qualification["record"]["artifact_metadata"]["frames"]
    assert descriptor["path"] == "qualification_frames.json"
    assert descriptor["renderer"] == "isaac_sim_kit_rtx"
    assert descriptor["sha256"].startswith("sha256:")
    assert descriptor["frame_count"] == 6
    assert len(descriptor["frames"]) == 6
    assert all(item["media_type"] == "image/png" for item in descriptor["frames"])


def test_approval_rejects_changed_qualification_frames(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _evidence_config(script)
    output_dir = tmp_path / "run"
    qualified = run_external_tune(
        ExternalTuneInput(config=config, output_dir=output_dir)
    )
    manifest = qualified.artifacts["qualification_frames"]
    manifest.write_bytes(manifest.read_bytes() + b"changed-after-review")

    result = run_external_tune(
        ExternalTuneInput(
            config=config,
            output_dir=output_dir,
            approval_digest=qualified.qualification_digest,
        )
    )

    assert not result.success
    assert result.status == "qualification_evidence_changed"
    assert not (output_dir / "optimization").exists()


@pytest.mark.parametrize(
    "trial_key",
    ["missing_evidence", "corrupt_evidence", "blank_evidence", "wrong_renderer"],
)
def test_qualification_rejects_invalid_frame_evidence(
    tmp_path: Path, trial_key: str
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)

    result = run_external_tune(
        ExternalTuneInput(
            config=_evidence_config(script, **{trial_key: True}),
            output_dir=tmp_path / "run",
        )
    )

    assert not result.success
    assert result.status == "qualification_failed"


def test_failed_qualification_removes_stale_published_frames(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    stale_manifest = output_dir / "qualification_frames.json"
    stale_manifest.write_bytes(b"stale")
    stale_frames = output_dir / "qualification_frames"
    stale_frames.mkdir()
    (stale_frames / "frame_0000.png").write_bytes(b"stale")

    result = run_external_tune(
        ExternalTuneInput(
            config=_evidence_config(script, missing_evidence=True),
            output_dir=output_dir,
        )
    )

    assert result.status == "qualification_failed"
    assert not stale_manifest.exists()
    assert not stale_frames.exists()


def test_new_qualification_removes_stale_approval(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    stale_approval = output_dir / "approval.json"
    stale_approval.write_text('{"approval_source": "stale"}', encoding="utf-8")

    result = run_external_tune(
        ExternalTuneInput(config=_evidence_config(script), output_dir=output_dir)
    )

    assert result.status == "awaiting_approval"
    assert not stale_approval.exists()


def test_external_tuning_runs_only_declared_trials(
    tmp_path: Path,
    playback_render_calls: list[Path],
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output_dir = tmp_path / "run"
    config = _evidence_config(script)
    qualified = run_external_tune(
        ExternalTuneInput(config=config, output_dir=output_dir)
    )

    result = run_external_tune(
        ExternalTuneInput(
            config=config,
            output_dir=output_dir,
            approval_digest=qualified.qualification_digest,
            render_winning_trial=True,
        )
    )

    assert result.success
    assert result.status == "completed"
    assert result.artifacts["qualification_frames"].is_file()
    requests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output_dir.rglob("request.json")
    ]
    purposes = {
        purpose: sum(request["purpose"] == purpose for request in requests)
        for purpose in {request["purpose"] for request in requests}
    }
    assert purposes == {"qualification": 1, "optimization": 24}
    assert all(
        (request["purpose"] == "qualification") == ("evidence" in request)
        for request in requests
    )
    assert all(
        (request["purpose"] == "optimization") == ("recording" in request)
        for request in requests
    )
    assert len(requests) == 1 + 12 * 2
    best_trial = next(row for row in result.history if row.params == result.best_params)
    expected_recording = (
        output_dir / best_trial.replicas[0].artifacts["recording_usd"]
    ).resolve()
    assert playback_render_calls == [expected_recording]
    published_recording = output_dir / "best_recording.usd"
    assert result.artifacts["best_recording"] == published_recording
    assert published_recording.read_bytes() == expected_recording.read_bytes()
    assert len(result.rendered_frames) == 6
    assert all(
        path.is_file() and path.suffix == ".png" for path in result.rendered_frames
    )
    assert result.render_error is None
    assert not list(output_dir.rglob("*.mp4"))
    saved = json.loads(result.artifacts["results"].read_text(encoding="utf-8"))
    assert saved["artifacts"]["qualification_frames"] == "qualification_frames.json"
    assert saved["artifacts"]["best_recording"] == "best_recording.usd"
    assert saved["selected_evidence"]["trial_index"] == best_trial.trial_index
    assert saved["selected_evidence"]["replica_index"] == 0
    assert saved["selected_evidence"]["seed"] == best_trial.replicas[0].seed
    assert saved["rendered_frames"] == [
        str(path.relative_to(output_dir)) for path in result.rendered_frames
    ]
    assert saved["render_error"] is None


def test_external_tuning_renders_at_recording_fps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions import graphics

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    evidence = config["evidence"]
    assert isinstance(evidence, dict)
    evidence["fps"] = 29.97
    config["optimizer"] = {
        "name": "random",
        "max_trials": 1,
        "seed": 7,
        "replicas": 1,
    }
    original_render = graphics.render_time_sampled_usd
    rendered_fps: list[object] = []

    def capture_fps(*args: Any, **kwargs: Any) -> list[Path]:
        rendered_fps.append(kwargs["fps"])
        return original_render(*args, **kwargs)

    monkeypatch.setattr(graphics, "render_time_sampled_usd", capture_fps)

    result = _qualify_and_run(
        config,
        tmp_path / "run",
        render_winning_trial=True,
    )

    assert result.success
    assert len(rendered_fps) == 1
    assert float(rendered_fps[0]) == pytest.approx(29.97)


def test_external_tuning_reuses_common_replica_seeds(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output_dir = tmp_path / "run"

    result = _qualify_and_run(_config(script), output_dir)

    assert result.success
    assert result.status == "completed"
    assert len(result.history) == 12
    assert result.n_trials == 12
    assert all(
        [replica.seed for replica in candidate.replicas] == [7, 8]
        for candidate in result.history
    )
    assert result.best_objective is not None
    assert result.best_objective < 0.15
    assert not (output_dir / "final_validation").exists()


def test_external_tuning_does_not_render_winner_by_default(
    tmp_path: Path,
    playback_render_calls: list[Path],
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output_dir = tmp_path / "run"

    result = _qualify_and_run(_config(script), output_dir)

    assert result.success
    assert result.artifacts["best_recording"].is_file()
    assert result.rendered_frames == []
    assert result.render_error is None
    assert playback_render_calls == []
    assert not (output_dir / "render").exists()


@pytest.mark.parametrize("trial_key", ["missing_recording", "corrupt_recording"])
def test_external_tuning_rejects_missing_or_invalid_winner_recording(
    tmp_path: Path,
    trial_key: str,
    playback_render_calls: list[Path],
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script, **{trial_key: True})
    config["optimizer"] = {
        "name": "random",
        "max_trials": 1,
        "seed": 7,
        "replicas": 1,
    }

    result = _qualify_and_run(
        config,
        tmp_path / "run",
        render_winning_trial=True,
    )

    assert not result.success
    assert result.status == "optimization_failed"
    assert playback_render_calls == []


def test_external_tuning_rejects_recording_changed_during_playback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions import graphics

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    config["optimizer"] = {
        "name": "random",
        "max_trials": 1,
        "seed": 7,
        "replicas": 1,
    }

    def mutate_recording(
        recording_path: Path,
        _output_dir: Path,
        **_kwargs: object,
    ) -> list[Path]:
        path = Path(recording_path)
        path.write_text(path.read_text(encoding="utf-8") + "\n# changed\n")
        return []

    monkeypatch.setattr(graphics, "render_time_sampled_usd", mutate_recording)

    result = _qualify_and_run(
        config,
        tmp_path / "run",
        render_winning_trial=True,
    )

    assert not result.success
    assert result.status == "render_failed"
    assert "changed after execution" in str(result.error)
    assert not (tmp_path / "run" / "render").exists()


def test_external_tuning_treats_optional_render_failure_as_non_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions import graphics

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)

    def fail_render(*_args: object, **_kwargs: object) -> list[Path]:
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(graphics, "render_time_sampled_usd", fail_render)

    result = _qualify_and_run(
        _config(script),
        tmp_path / "run",
        render_winning_trial=True,
    )

    assert result.success
    assert result.status == "completed"
    assert result.rendered_frames == []
    assert result.render_error == "RuntimeError"
    assert result.artifacts["best_recording"].is_file()
    assert not (tmp_path / "run" / "render").exists()


def test_unsuccessful_low_objective_cannot_win(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)

    result = _qualify_and_run(
        _config(script, failed_low_objective=True), tmp_path / "run"
    )

    assert result.success
    assert result.best_params["gain"] >= 0.5
    failed = [candidate for candidate in result.history if not candidate.success]
    assert failed
    assert all(candidate.optimizer_score == 1.0e12 for candidate in failed)


def test_diagnostic_metrics_do_not_affect_candidate_selection(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)

    result = _qualify_and_run(_config(script, misleading_metric=True), tmp_path / "run")

    assert result.success
    selected = next(row for row in result.history if row.params == result.best_params)
    selected_metric = float(selected.replicas[0].metrics["distance"])
    all_metrics = [
        float(replica.metrics["distance"])
        for candidate in result.history
        for replica in candidate.replicas
    ]
    assert result.best_objective == min(
        candidate.objective_value for candidate in result.history
    )
    assert selected_metric > min(all_metrics)


@pytest.mark.parametrize(
    "trial_key",
    ["missing_objective", "wrong_objective", "escape_artifact", "non_finite_metric"],
)
def test_qualification_fails_closed_on_invalid_evidence(
    tmp_path: Path, trial_key: str
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)

    result = run_external_tune(
        ExternalTuneInput(
            config=_config(script, **{trial_key: True}),
            output_dir=tmp_path / "run",
        )
    )

    assert not result.success
    assert result.status == "qualification_failed"


def test_runtime_fingerprint_reports_process_spawn_failure(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    non_executable = tmp_path / "not-executable"
    non_executable.write_text("not executable", encoding="utf-8")
    config = _config(script)
    config["runtime"]["python"] = str(non_executable)  # type: ignore[index]

    result = run_external_tune(
        ExternalTuneInput(config=config, output_dir=tmp_path / "run")
    )

    assert not result.success
    assert result.status == "failed"
    assert result.error == "external Python runtime fingerprint probe failed"
    assert result.qualification_path is None


def test_qualification_enforces_log_limit_after_process_exit(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script, log_bytes=4096)
    config["runtime"]["max_log_bytes"] = 128  # type: ignore[index]

    result = run_external_tune(
        ExternalTuneInput(config=config, output_dir=tmp_path / "run")
    )

    assert not result.success
    assert result.status == "qualification_failed"


@pytest.mark.parametrize("field", ["name", "unit"])
def test_external_config_rejects_non_string_objective_identity(
    tmp_path: Path, field: str
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    config["objective"][field] = 123  # type: ignore[index]

    with pytest.raises(ExternalTuneConfigError, match=f"objective.{field}"):
        load_external_tune_spec(config)


def test_runtime_python_preserves_virtualenv_invocation_path(
    tmp_path: Path,
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    bin_dir = tmp_path / "customer-env" / "bin"
    bin_dir.mkdir(parents=True)
    launcher = bin_dir / "python"
    launcher.symlink_to(sys.executable)
    config = _config(script)
    config["runtime"]["python"] = str(launcher)  # type: ignore[index]

    spec = load_external_tune_spec(config)

    assert spec.runtime.python == launcher.absolute()
    assert spec.runtime.python != launcher.resolve()


def test_external_config_rejects_unknown_fields(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    config["surprise"] = True

    with pytest.raises(ExternalTuneConfigError, match="unknown field"):
        load_external_tune_spec(config)


@pytest.mark.parametrize("field", ["measurements", "scoring"])
def test_external_config_rejects_removed_scoring_fields(
    tmp_path: Path, field: str
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    config[field] = {}

    with pytest.raises(ExternalTuneConfigError, match="unknown field"):
        load_external_tune_spec(config)


def test_external_config_requires_scalar_objective(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    del config["objective"]

    with pytest.raises(ExternalTuneConfigError, match="objective must be an object"):
        load_external_tune_spec(config)


def test_results_artifact_matches_output(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output_dir = tmp_path / "run"
    result = _qualify_and_run(_config(script), output_dir)

    saved = json.loads(result.artifacts["results"].read_text(encoding="utf-8"))
    assert saved["best_params"] == result.best_params
    assert saved["objective"]["best_value"] == result.best_objective
    assert saved["status"] == "completed"
    assert saved["started_at"]
    assert saved["completed_at"]
    best_params = json.loads((output_dir / "best_params.json").read_text())
    assert set(best_params) == {"best_score", "params"}
    assert best_params["params"] == result.best_params
    assert best_params["best_score"] == result.best_score
    run_spec = json.loads((output_dir / "run_spec.json").read_text(encoding="utf-8"))
    assert run_spec["active_parameters"] == [
        {"integer": False, "max": 1.0, "min": 0.0, "name": "gain"}
    ]
    assert run_spec["fixed_params"] == {}
    assert run_spec["optimizer"] == {
        "max_trials": 12,
        "replica_seed": None,
        "replica_seeds": [7, 8],
        "replicas": 2,
        "requested": "random",
        "resolved": "random",
        "seed": 7,
    }
    history = json.loads((output_dir / "history.jsonl").read_text().splitlines()[0])
    assert history["mode"] == "external_runtime"
    assert history["score"] == history["optimizer_score"]
    assert history["failed"] is False
    assert history["duration_seconds"] > 0
    replica = result.history[0].replicas[0]
    artifact = Path(replica.artifacts["metrics"])
    assert not artifact.is_absolute()
    assert (output_dir / artifact).is_file()
    assert replica.trial_dir is not None
    assert not Path(replica.trial_dir).is_absolute()


def test_external_tune_promotes_allowlisted_winner_outputs(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output_dir = tmp_path / "run"
    config = _config(script)
    config["publish_artifacts"] = ["metrics"]

    qualified = run_external_tune(
        ExternalTuneInput(config=config, output_dir=output_dir)
    )
    qualification = json.loads((output_dir / "qualification.json").read_text())
    assert qualification["input_fingerprint"]["contract"]["publish_artifacts"] == [
        "metrics"
    ]

    result = run_external_tune(
        ExternalTuneInput(
            config=config,
            output_dir=output_dir,
            approval_digest=qualified.qualification_digest,
        )
    )

    assert result.success
    promoted = output_dir / "outputs" / "metrics" / "metrics.json"
    assert result.artifacts["output.metrics"] == promoted
    assert result.published_outputs["metrics"]["path"] == (
        "outputs/metrics/metrics.json"
    )
    assert result.published_outputs["metrics"]["sha256"].startswith("sha256:")
    optimization_requests = [
        json.loads(path.read_text())
        for path in output_dir.rglob("request.json")
        if json.loads(path.read_text())["purpose"] == "optimization"
    ]
    assert optimization_requests
    assert all(
        request["publish_artifacts"] == ["metrics"] for request in optimization_requests
    )
    approval = json.loads((output_dir / "approval.json").read_text())
    assert approval["approval_source"] == "local"
    assert approval["recorded_by_process_user"]
    assert "approved_by" not in approval


def test_external_tune_removes_partial_winner_bundle_on_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tuning.external import runner

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output_dir = tmp_path / "run"
    config = _config(script)
    config["publish_artifacts"] = ["metrics"]
    qualified = run_external_tune(
        ExternalTuneInput(config=config, output_dir=output_dir)
    )

    def fail_descriptor(path: Path, *, relative_path: str) -> dict[str, object]:
        del path, relative_path
        raise OSError("publish failed")

    monkeypatch.setattr(runner, "file_descriptor", fail_descriptor)
    result = run_external_tune(
        ExternalTuneInput(
            config=config,
            output_dir=output_dir,
            approval_digest=qualified.qualification_digest,
        )
    )

    assert not result.success
    assert result.status == "recording_failed"
    assert not (output_dir / "best_recording.usd").exists()
    assert not (output_dir / "outputs").exists()
    assert not (output_dir / ".outputs.tmp").exists()


@pytest.mark.parametrize(
    ("publish_artifacts", "message"),
    [
        (["metrics", "metrics"], "must be unique"),
        (["recording_usd"], "must not duplicate evidence"),
        (["../escape"], "name is invalid"),
    ],
)
def test_external_config_rejects_invalid_publish_artifacts(
    tmp_path: Path,
    publish_artifacts: list[str],
    message: str,
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    config["publish_artifacts"] = publish_artifacts

    with pytest.raises(ExternalTuneConfigError, match=message):
        load_external_tune_spec(config)


def test_external_tune_persists_config_failure(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output_dir = tmp_path / "run"
    config = _config(script)
    del config["objective"]

    result = run_external_tune(ExternalTuneInput(config=config, output_dir=output_dir))

    assert not result.success
    assert result.status == "failed"
    saved = json.loads((output_dir / "external_tune_results.json").read_text())
    assert saved["status"] == "failed"
    assert "objective must be an object" in saved["error"]


def test_external_config_rejects_non_json_trial_values(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    config["runtime"]["trial"] = {"opaque": object()}  # type: ignore[index]

    with pytest.raises(ExternalTuneConfigError, match="JSON-serializable"):
        load_external_tune_spec(config)


@pytest.mark.parametrize(
    "params",
    [
        {"other": 0.5},
        {"gain": float("nan")},
        {"gain": 2.0},
    ],
)
def test_backend_rejects_invalid_candidate_before_process_start(
    tmp_path: Path, params: dict[str, float]
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    spec = load_external_tune_spec(_config(script))
    output_dir = tmp_path / "run"
    executor = ExternalTrialExecutor(spec, output_dir)

    with pytest.raises(ExternalTrialError):
        executor.run(params, seed=1, purpose="optimization", index=0)

    assert not output_dir.exists()


def test_optimization_cancellation_persists_completed_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    original_run = ExternalTrialExecutor.run

    def cancel_during_optimization(
        self: ExternalTrialExecutor,
        params: dict[str, float],
        seed: int,
        *,
        purpose: str,
        index: int,
    ):
        if purpose == "optimization" and index >= 2:
            raise TuningCancelledError("cancel optimization")
        return original_run(
            self,
            params,
            seed,
            purpose=purpose,
            index=index,
        )

    monkeypatch.setattr(ExternalTrialExecutor, "run", cancel_during_optimization)

    result = _qualify_and_run(_config(script), tmp_path / "run")

    assert not result.success
    assert result.status == "cancelled"
    assert result.cancelled
    assert len(result.history) == 1
    assert not result.best_params
    saved = json.loads(result.artifacts["results"].read_text(encoding="utf-8"))
    assert saved["status"] == "cancelled"
    assert len(saved["history"]) == 1
    assert "final_validation" not in saved


def test_approved_run_requires_existing_qualification(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output_dir = tmp_path / "run"

    result = run_external_tune(
        ExternalTuneInput(
            config=_config(script),
            output_dir=output_dir,
            approval_digest="sha256:" + "a" * 64,
        )
    )

    assert not result.success
    assert result.status == "qualification_missing"
    assert not (output_dir / "qualification.json").exists()
    saved = json.loads((output_dir / "external_tune_results.json").read_text())
    assert saved["status"] == "qualification_missing"


def test_approved_run_does_not_overwrite_changed_qualification(
    tmp_path: Path,
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output_dir = tmp_path / "run"
    qualified = run_external_tune(
        ExternalTuneInput(config=_config(script), output_dir=output_dir)
    )
    qualification_path = output_dir / "qualification.json"
    reviewed_evidence = qualification_path.read_bytes()
    script.write_text(
        script.read_text(encoding="utf-8") + "\n# changed after review\n",
        encoding="utf-8",
    )

    result = run_external_tune(
        ExternalTuneInput(
            config=_config(script),
            output_dir=output_dir,
            approval_digest=qualified.qualification_digest,
        )
    )

    assert not result.success
    assert result.status == "fingerprint_changed"
    assert qualification_path.read_bytes() == reviewed_evidence
    assert not (output_dir / "optimization").exists()


def test_fingerprint_covers_base_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    spec = load_external_tune_spec(_config(script))
    monkeypatch.setenv("PATH", "/first")
    first = build_runtime_fingerprint(spec)
    monkeypatch.setenv("PATH", "/second")
    second = build_runtime_fingerprint(spec)

    assert "PATH" in first["environment"]
    assert first["digest"] != second["digest"]


def test_fingerprint_distinguishes_unset_from_empty_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    spec = load_external_tune_spec(_config(script))
    variable = "PHYSICS_AGENT_OPTIONAL_RUNTIME_VALUE"
    spec = replace(
        spec,
        runtime=replace(spec.runtime, pass_env=(variable,)),
    )
    monkeypatch.delenv(variable, raising=False)
    unset = build_runtime_fingerprint(spec)
    monkeypatch.setenv(variable, "")
    empty = build_runtime_fingerprint(spec)

    assert unset["environment"][variable] == "unset"
    assert empty["environment"][variable].startswith("sha256:")
    assert unset["digest"] != empty["digest"]


def test_fingerprint_detects_installed_runtime_package_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning.external import fingerprint

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    spec = load_external_tune_spec(_config(script))
    runtime_version = "6.0.1.0"

    def python_environment(_spec: ExternalTuneSpec) -> dict[str, object]:
        return {
            "packages": [{"name": "isaacsim", "version": runtime_version}],
        }

    monkeypatch.setattr(fingerprint, "_python_environment", python_environment)
    original = build_runtime_fingerprint(spec)
    runtime_version = "6.0.2.0"
    upgraded = build_runtime_fingerprint(spec)

    assert original["python_environment"] != upgraded["python_environment"]
    assert original["digest"] != upgraded["digest"]


def test_python_environment_fingerprint_uses_declared_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning.external import fingerprint

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    spec = load_external_tune_spec(_config(script))
    spec = replace(
        spec,
        runtime=replace(
            spec.runtime,
            python_args=("-I",),
            pass_env=("PHYSICS_AGENT_RUNTIME_TOKEN",),
        ),
    )
    monkeypatch.setenv("PHYSICS_AGENT_RUNTIME_TOKEN", "allowed")
    monkeypatch.setenv("PHYSICS_AGENT_NOT_ALLOWED", "excluded")
    captured: dict[str, object] = {}

    def run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        captured["argv"] = argv
        captured.update(kwargs)
        return SimpleNamespace(
            stdout=(
                "runtime launcher message\n"
                'PHYSICS_AGENT_RUNTIME_FINGERPRINT_V1={"packages": []}\n'
            )
        )

    monkeypatch.setattr(fingerprint.subprocess, "run", run)

    environment = fingerprint._python_environment(spec)

    assert environment == {"packages": []}
    assert captured["argv"][:3] == [str(spec.runtime.python), "-I", "-c"]
    assert captured["cwd"] == str(spec.runtime.cwd)
    assert captured["timeout"] == spec.runtime.timeout_s
    assert captured["shell"] is False
    process_environment = captured["env"]
    assert isinstance(process_environment, dict)
    assert process_environment["PHYSICS_AGENT_RUNTIME_TOKEN"] == "allowed"
    assert "PHYSICS_AGENT_NOT_ALLOWED" not in process_environment


def test_python_environment_fingerprint_detects_installed_file_edits(
    tmp_path: Path,
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    environment = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False).create(environment)
    site_packages = (
        environment
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    package = site_packages / "demo_runtime"
    package.mkdir()
    installed_file = package / "__init__.py"
    installed_file.write_text("VALUE = 1\n", encoding="utf-8")
    (site_packages / "demo_alias").symlink_to(package, target_is_directory=True)
    metadata = site_packages / "demo_runtime-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: demo-runtime\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (metadata / "RECORD").write_text(
        "\n".join(
            [
                "demo_runtime/__init__.py,,",
                "demo_alias/optional.py,,",
                "demo_runtime-1.0.dist-info/METADATA,,",
                "demo_runtime-1.0.dist-info/RECORD,,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec = load_external_tune_spec(_config(script))
    spec = replace(
        spec,
        runtime=replace(spec.runtime, python=environment / "bin" / "python"),
    )

    original = _installed_package(
        _python_environment(spec),
        name="demo-runtime",
    )
    installed_file.write_text("VALUE = 2\n", encoding="utf-8")
    edited = _installed_package(
        _python_environment(spec),
        name="demo-runtime",
    )

    assert original["version"] == edited["version"] == "1.0"
    assert original["record_digest"] == edited["record_digest"]
    assert original["installed_files_digest"] != edited["installed_files_digest"]
    assert original["installed_file_count"] == 4

    with (metadata / "RECORD").open("a", encoding="utf-8") as stream:
        stream.write("demo_runtime/missing.py,,\n")
    with pytest.raises(RuntimeError, match="fingerprint probe failed"):
        _python_environment(spec)


def test_python_environment_fingerprint_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning.external import fingerprint

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    spec = load_external_tune_spec(_config(script))

    monkeypatch.setattr(
        fingerprint.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["python"])
        ),
    )
    with pytest.raises(RuntimeError, match="fingerprint probe failed"):
        fingerprint._python_environment(spec)

    monkeypatch.setattr(
        fingerprint.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="launcher output only\n"),
    )
    with pytest.raises(RuntimeError, match="returned invalid output"):
        fingerprint._python_environment(spec)

    monkeypatch.setattr(
        fingerprint.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="PHYSICS_AGENT_RUNTIME_FINGERPRINT_V1={broken}\n"
        ),
    )
    with pytest.raises(RuntimeError, match="fingerprint probe failed"):
        fingerprint._python_environment(spec)

    monkeypatch.setattr(
        fingerprint.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="PHYSICS_AGENT_RUNTIME_FINGERPRINT_V1=[]\n"
        ),
    )
    with pytest.raises(RuntimeError, match="returned invalid data"):
        fingerprint._python_environment(spec)


@pytest.mark.parametrize("nested_directory", ["output", "qualification"])
def test_external_tune_rejects_run_directories_inside_fingerprint_root(
    tmp_path: Path,
    nested_directory: str,
) -> None:
    fingerprint_root = tmp_path / "runtime"
    fingerprint_root.mkdir()
    script = fingerprint_root / "trial.py"
    _write_trial_adapter(script)
    spec = load_external_tune_spec(_config(script))
    spec = replace(
        spec,
        runtime=replace(spec.runtime, fingerprint_paths=(fingerprint_root,)),
    )
    output_dir = (
        fingerprint_root / "run" if nested_directory == "output" else tmp_path / "run"
    )
    qualification_dir = (
        fingerprint_root / "qualification"
        if nested_directory == "qualification"
        else None
    )

    result = run_external_tune(
        ExternalTuneInput(
            config=spec,
            output_dir=output_dir,
            qualification_dir=qualification_dir,
        )
    )

    assert not result.success
    assert result.status == "invalid_output_layout"
    assert f"{nested_directory}_dir must not be inside" in str(result.error)
    assert not output_dir.exists()
    if qualification_dir is not None:
        assert not qualification_dir.exists()


def test_external_tune_allows_run_directories_outside_fingerprint_root(
    tmp_path: Path,
) -> None:
    fingerprint_root = tmp_path / "runtime"
    fingerprint_root.mkdir()
    script = fingerprint_root / "trial.py"
    _write_trial_adapter(script)
    spec = load_external_tune_spec(_config(script))
    spec = replace(
        spec,
        runtime=replace(spec.runtime, fingerprint_paths=(fingerprint_root,)),
    )

    result = run_external_tune(
        ExternalTuneInput(
            config=spec,
            output_dir=tmp_path / "run",
            qualification_dir=tmp_path / "qualification",
        )
    )

    assert result.success
    assert result.status == "awaiting_approval"


def test_external_config_loads_relative_paths_from_yaml(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    runtime = config["runtime"]
    assert isinstance(runtime, dict)
    runtime.update(
        {
            "script": "trial.py",
            "cwd": ".",
            "fingerprint_paths": ["trial.py"],
        }
    )
    config_path = tmp_path / "external.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    spec = load_external_tune_spec(config_path)

    assert spec.runtime.script == script
    assert spec.runtime.cwd == tmp_path


@pytest.mark.parametrize("contents", ["missing", "invalid: ["])
def test_external_config_reports_unreadable_yaml(tmp_path: Path, contents: str) -> None:
    config_path = tmp_path / "external.yaml"
    if contents == "invalid: [":
        config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ExternalTuneConfigError):
        load_external_tune_spec(config_path)


def test_external_artifact_helpers_cover_fallback_and_allowlist(
    tmp_path: Path,
) -> None:
    unknown = tmp_path / "artifact.unknown-extension"
    unknown.write_bytes(b"payload")
    assert file_descriptor(unknown, relative_path=unknown.name)["media_type"] == (
        "application/octet-stream"
    )

    (tmp_path / "qualification.json").write_text("{}", encoding="utf-8")
    (tmp_path / "private.json").write_text("{}", encoding="utf-8")
    render = tmp_path / "render"
    render.mkdir()
    (render / "frame.png").write_bytes(b"png")
    (tmp_path / "qualification-link.json").symlink_to(tmp_path / "qualification.json")

    assert collect_public_tune_artifacts(tmp_path) == [
        "qualification.json",
        "render/frame.png",
    ]


def test_runtime_fingerprint_walks_directories_and_skips_nonfiles(
    tmp_path: Path,
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    fingerprint_root = tmp_path / "runtime"
    fingerprint_root.mkdir()
    (fingerprint_root / "config.json").write_text("{}", encoding="utf-8")
    excluded = fingerprint_root / ".git"
    excluded.mkdir()
    (excluded / "ignored").write_text("ignored", encoding="utf-8")
    (fingerprint_root / "dangling").symlink_to(fingerprint_root / "missing")
    spec = load_external_tune_spec(_config(script))
    spec = replace(
        spec,
        runtime=replace(spec.runtime, fingerprint_paths=(fingerprint_root,)),
    )

    fingerprint = build_runtime_fingerprint(spec)

    assert str((fingerprint_root / "config.json").absolute()) in fingerprint["files"]
    assert all(".git" not in path for path in fingerprint["files"])


def test_runtime_fingerprint_rejects_nested_directory_symlink(
    tmp_path: Path,
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    fingerprint_root = tmp_path / "runtime"
    fingerprint_root.mkdir()
    linked_target = tmp_path / "linked-runtime"
    linked_target.mkdir()
    (linked_target / "plugin.py").write_text("VERSION = 1\n", encoding="utf-8")
    (fingerprint_root / "plugins").symlink_to(
        linked_target,
        target_is_directory=True,
    )
    spec = load_external_tune_spec(_config(script))
    spec = replace(
        spec,
        runtime=replace(spec.runtime, fingerprint_paths=(fingerprint_root,)),
    )

    with pytest.raises(ValueError, match="must not contain directory symlinks"):
        build_runtime_fingerprint(spec)


def test_external_runtime_rejects_invalid_environment_name(tmp_path: Path) -> None:
    from physics_agent.tuning.external.types import ExternalRuntime

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)

    with pytest.raises(ValueError, match="invalid pass_env"):
        ExternalRuntime(
            python=Path(sys.executable),
            script=script,
            cwd=tmp_path,
            fingerprint_paths=(script,),
            pass_env=("BAD-NAME",),
        )


def test_integer_search_values_are_rounded_and_tune_input_exposes_settings(
    tmp_path: Path,
) -> None:
    from physics_agent.tuning.optimizers import _params_from_vector
    from physics_agent.tuning.types import TuneInput

    search = SimpleNamespace(params=(TunableParam("count", 1, 5, integer=True),))

    assert _params_from_vector(search, np.array([0.6])) == {"count": 3.0}
    settings = TuneInput(
        scenario=tmp_path / "scenario.yaml",
        physics_usd=tmp_path / "asset.usd",
        output_dir=tmp_path,
        optimizer="random",
        max_trials=3,
        seed=11,
    ).optimizer_settings
    assert settings == OptimizerSettings(name="random", max_trials=3, seed=11)


def test_external_integer_parameters_do_not_inherit_builtin_friction_coupling(
    tmp_path: Path,
) -> None:
    from physics_agent.tuning.optimizers import (
        _params_from_vector,
        run_random_optimizer,
    )

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    config["parameters"] = {
        "static_friction": {"min": 1, "max": 3, "integer": True},
        "dynamic_friction": {"min": 4, "max": 6, "integer": True},
    }
    qualification = config["qualification"]
    assert isinstance(qualification, dict)
    qualification["nominal_params"] = {
        "static_friction": 2,
        "dynamic_friction": 5,
    }
    spec = load_external_tune_spec(config)

    assert _params_from_vector(spec, np.array([0.5, 0.5])) == {
        "static_friction": 2.0,
        "dynamic_friction": 5.0,
    }
    samples: list[dict[str, float]] = []
    run_random_optimizer(
        spec,
        lambda params: samples.append(dict(params)) or 0.0,
        max_trials=3,
        seed=17,
    )
    assert len(samples) == 3
    assert all(
        sample["dynamic_friction"] > sample["static_friction"] for sample in samples
    )


class _Process:
    def __init__(self, *, wait_timeout: bool = False) -> None:
        self.pid = 123
        self.returncode = 0
        self.wait_timeout = wait_timeout
        self.wait_calls = 0

    def poll(self) -> None:
        return None

    def wait(self, timeout: float) -> None:
        self.wait_calls += 1
        if self.wait_timeout and self.wait_calls == 1:
            raise subprocess.TimeoutExpired("trial", timeout)


def test_terminate_group_handles_missing_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tuning.external import backend

    monkeypatch.setattr(
        backend.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )

    backend._terminate_group(_Process())


@pytest.mark.parametrize("missing_on_kill", [False, True])
def test_terminate_group_escalates_after_grace_period(
    monkeypatch: pytest.MonkeyPatch, missing_on_kill: bool
) -> None:
    from physics_agent.tuning.external import backend

    signals: list[int] = []

    def killpg(_pid: int, sent_signal: int) -> None:
        signals.append(sent_signal)
        if missing_on_kill and len(signals) == 2:
            raise ProcessLookupError

    monkeypatch.setattr(backend.os, "killpg", killpg)
    process = _Process(wait_timeout=True)

    backend._terminate_group(process)

    assert signals[:2] == [backend.signal.SIGTERM, backend.signal.SIGKILL]
    assert process.wait_calls == (1 if missing_on_kill else 2)


def test_backend_trial_directory_collision(tmp_path: Path) -> None:
    from physics_agent.tuning.external import backend

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    executor = ExternalTrialExecutor(
        load_external_tune_spec(_config(script)), tmp_path / "run"
    )
    base = tmp_path / "run" / "optimization" / "trial_0000_seed_1"
    base.mkdir(parents=True)
    base.with_name(base.name + "_attempt_2").mkdir()
    assert executor._trial_dir("optimization", 0, 1).name.endswith("_attempt_3")
    relative_artifact = base / "artifact.bin"
    relative_artifact.write_bytes(b"artifact")
    assert (
        backend._artifact_path(
            "artifact.bin",
            name="artifact",
            root=base.resolve(),
        )
        == relative_artifact.resolve()
    )


def test_backend_process_cancellation_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning.external import backend

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    spec = load_external_tune_spec(_config(script))
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    process = _Process()
    popen_calls: list[dict[str, Any]] = []

    def popen(*_args: Any, **kwargs: Any) -> _Process:
        popen_calls.append(kwargs)
        return process

    monkeypatch.setattr(backend.subprocess, "Popen", popen)
    monkeypatch.setattr(backend, "_terminate_group", lambda _process: None)

    cancelled = SimpleNamespace(is_set=lambda: True)
    executor = ExternalTrialExecutor(spec, tmp_path / "run", cancel_event=cancelled)
    with pytest.raises(TuningCancelledError):
        executor._run_process(["trial"], stdout_path=stdout, stderr_path=stderr)

    executor = ExternalTrialExecutor(
        replace(spec, runtime=replace(spec.runtime, timeout_s=0.01)),
        tmp_path / "run",
    )
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(backend.time, "monotonic", lambda: next(ticks))
    with pytest.raises(ExternalTrialError, match="timed out"):
        executor._run_process(["trial"], stdout_path=stdout, stderr_path=stderr)

    assert len(popen_calls) == 2
    assert all(call["shell"] is False for call in popen_calls)
    assert all(call["start_new_session"] is True for call in popen_calls)


def test_backend_keeps_process_stderr_out_of_public_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    output = tmp_path / "run"
    secret = "adapter-secret-token"

    def fail_process(
        _self: ExternalTrialExecutor,
        _argv: list[str],
        *,
        stdout_path: Path,
        stderr_path: Path,
    ) -> int:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(f"adapter exploded: {secret}", encoding="utf-8")
        return 3

    monkeypatch.setattr(ExternalTrialExecutor, "_run_process", fail_process)
    result = run_external_tune(
        ExternalTuneInput(config=_config(script), output_dir=output)
    )

    qualification = json.loads(
        (output / "qualification.json").read_text(encoding="utf-8")
    )
    assert result.status == "qualification_failed"
    assert qualification["record"]["error"] == "external trial exited with status 3"
    assert secret not in json.dumps(qualification)
    stderr_logs = list(output.rglob("stderr.log"))
    assert len(stderr_logs) == 1
    assert secret in stderr_logs[0].read_text(encoding="utf-8")


def test_backend_rejects_invalid_metadata_and_escaped_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    spec = load_external_tune_spec(_config(script))
    executor = ExternalTrialExecutor(spec, tmp_path / "run")
    payload = {
        "status": "ok",
        "success": False,
        "metrics": {},
        "metadata": [],
        "artifacts": {},
    }

    def write_result(argv: list[str], *, stdout_path: Path, stderr_path: Path) -> int:
        del stdout_path, stderr_path
        request_path = Path(argv[argv.index("--request") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        artifacts_dir = Path(request["artifacts_dir"])
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"outside")
        (artifacts_dir / "escaped.bin").symlink_to(outside)
        result_path = Path(argv[argv.index("--result") + 1])
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return 0

    monkeypatch.setattr(executor, "_run_process", write_result)
    with pytest.raises(ExternalTrialError, match="metadata must be an object"):
        executor.run({"gain": 0.5}, seed=1, purpose="optimization", index=0)

    payload["metadata"] = {"applied_params": {"gain": 0.5}}
    with pytest.raises(ExternalTrialError, match="resolves outside"):
        executor.run({"gain": 0.5}, seed=2, purpose="optimization", index=1)


def test_runner_private_validation_and_serialization_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning.external import runner

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    spec = load_external_tune_spec(_config(script))

    events: list[str] = []
    runner._emit(
        SimpleNamespace(event=lambda event_type, _data: events.append(event_type)),
        "trial",
        {},
    )
    assert events == ["trial"]
    assert runner._cancelled(SimpleNamespace(is_set=lambda: True))
    with pytest.raises(TypeError, match="is_set"):
        runner._cancelled(object())

    outside = tmp_path / "outside.usd"
    outside.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    escaped_replica = ReplicaRecord(
        seed=1,
        objective_value=0.1,
        success=True,
        artifacts={"frames": "../outside.json", "recording_usd": "../outside.usd"},
        artifact_metadata={"frames": {}, "recording_usd": {}},
    )
    with pytest.raises(ExternalTrialError, match="evidence escaped"):
        runner._publish_evidence(escaped_replica, spec, output, purpose="qualification")
    escaped_record = TrialRecord(
        trial_index=0,
        params={"gain": 0.5},
        score=0.1,
        replicas=[escaped_replica],
    )
    with pytest.raises(ExternalTrialError, match="recording escaped"):
        runner._resolve_best_recording(escaped_record, spec, output)
    assert runner._discover_camera_paths(outside) is None

    external_output = ExternalTuneOutput(
        success=False,
        output_dir=output,
        artifacts={"outside": outside},
    )
    assert runner._result_payload(external_output, spec)["artifacts"]["outside"] == str(
        outside
    )
    assert (
        runner._persist_unhandled_result(
            ExternalTuneOutput(success=False, output_dir=None)
        ).output_dir
        is None
    )

    persisted = runner._persist_result(
        ExternalTuneOutput(success=False, output_dir=output),
        spec,
        output,
    )
    assert persisted.started_at and persisted.completed_at

    monkeypatch.setattr(
        runner,
        "inspect_usd_recording",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad recording")),
    )
    source = tmp_path / "source.usd"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    with pytest.raises(ExternalTrialError, match="could not be published"):
        runner._publish_best_recording(source, {}, spec, output)
    assert not (output / ".best_recording.tmp.usd").exists()


def test_runner_rejects_escaped_publish_output_and_render_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions import graphics

    from physics_agent.tuning.external import runner

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    config["publish_artifacts"] = ["metrics"]
    spec = load_external_tune_spec(config)
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    replica = ReplicaRecord(
        seed=1,
        objective_value=0.1,
        success=True,
        artifacts={"metrics": "../outside.json"},
        artifact_metadata={"metrics": {}},
    )
    record = TrialRecord(
        trial_index=0,
        params={"gain": 0.5},
        score=0.1,
        replicas=[replica],
    )
    with pytest.raises(ExternalTrialError, match="escaped"):
        runner._promote_winner_outputs(record, spec, output)

    recording = output / "recording.usd"
    recording.write_text("#usda 1.0\n", encoding="utf-8")
    descriptor = {"path": "recording.usd", "fps": 30.0}
    monkeypatch.setattr(
        runner, "inspect_usd_recording", lambda *_args, **_kwargs: descriptor
    )
    monkeypatch.setattr(
        graphics,
        "render_time_sampled_usd",
        lambda *_args, **_kwargs: [outside],
    )
    with pytest.raises(RuntimeError, match="outside render directory"):
        runner._render_best_recording(
            recording_path=recording,
            descriptor=descriptor,
            spec=spec,
            output_dir=output,
        )

    def missing_png(
        _recording: Path, render_dir: Path, **_kwargs: object
    ) -> list[Path]:
        render_dir.mkdir(parents=True, exist_ok=True)
        invalid = render_dir / "frame.txt"
        invalid.write_text("not png", encoding="utf-8")
        return [invalid]

    monkeypatch.setattr(graphics, "render_time_sampled_usd", missing_png)
    with pytest.raises(RuntimeError, match="PNG"):
        runner._render_best_recording(
            recording_path=recording,
            descriptor=descriptor,
            spec=spec,
            output_dir=output,
        )

    config["evidence"]["playback_renderer"] = "remote"  # type: ignore[index]
    remote_spec = load_external_tune_spec(config)
    forwarded: dict[str, object] = {}

    def capture_renderer(
        _recording: Path, render_dir: Path, **kwargs: object
    ) -> list[Path]:
        forwarded.update(kwargs)
        render_dir.mkdir(parents=True, exist_ok=True)
        frame = render_dir / "frame.png"
        frame.write_bytes(b"png")
        return [frame]

    monkeypatch.setattr(graphics, "render_time_sampled_usd", capture_renderer)
    runner._render_best_recording(
        recording_path=recording,
        descriptor=descriptor,
        spec=remote_spec,
        output_dir=output,
    )
    assert forwarded["renderer"] == "remote"


def test_runner_qualification_parsers_and_cached_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tuning.external import runner

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    spec = load_external_tune_spec(_config(script))
    output = tmp_path / "run"
    output.mkdir()

    assert runner._stored_evidence_is_valid({}, SimpleNamespace(evidence=None), output)
    for qualification in (
        {},
        {"record": []},
        {"record": {"artifacts": []}},
        {"record": {"artifacts": {}, "artifact_metadata": []}},
        {
            "record": {
                "artifacts": {},
                "artifact_metadata": {},
                "metadata": [],
            }
        },
    ):
        assert not runner._stored_evidence_is_valid(qualification, spec, output)
    assert runner._qualification_artifacts(
        output / "qualification.json", {"record": []}, spec, output
    ) == {"qualification": output / "qualification.json"}
    assert runner._qualification_artifacts(
        output / "qualification.json",
        {"record": {"artifacts": []}},
        spec,
        output,
    ) == {"qualification": output / "qualification.json"}

    malformed = output / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert runner._load_qualification(malformed) is None
    malformed.write_text("[]", encoding="utf-8")
    assert runner._load_qualification(malformed) is None
    malformed.write_text('{"qualification_digest":"wrong"}', encoding="utf-8")
    assert runner._load_valid_qualification(malformed) is None

    invalid_relative = {
        "record": {
            "artifacts": {spec.evidence.artifact_name: 1},
            "artifact_metadata": {spec.evidence.artifact_name: {}},
            "metadata": {},
        }
    }
    assert not runner._stored_evidence_is_valid(invalid_relative, spec, output)
    missing_frames = {
        "record": {
            "artifacts": {spec.evidence.artifact_name: "missing.json"},
            "artifact_metadata": {spec.evidence.artifact_name: {}},
            "metadata": {},
        }
    }
    assert not runner._stored_evidence_is_valid(missing_frames, spec, output)
    assert runner._qualification_artifacts(
        output / "qualification.json",
        {},
        SimpleNamespace(evidence=None),
        output,
    ) == {"qualification": output / "qualification.json"}

    from pxr import Usd

    with monkeypatch.context() as patch:
        patch.setattr(Usd, "Stage", SimpleNamespace(Open=lambda _path: None))
        assert runner._discover_camera_paths(tmp_path / "invalid.usd") is None

    first = run_external_tune(ExternalTuneInput(config=spec, output_dir=output))
    assert first.status == "awaiting_approval"
    second = run_external_tune(ExternalTuneInput(config=spec, output_dir=output))
    assert second.qualification_digest == first.qualification_digest


def test_run_qualification_propagates_cancellation(tmp_path: Path) -> None:
    from physics_agent.tuning.external import runner

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    spec = load_external_tune_spec(_config(script))
    output = tmp_path / "run"
    output.mkdir()

    class Executor:
        def run(self, *_args: object, **_kwargs: object) -> ReplicaRecord:
            raise TuningCancelledError("cancel")

    with pytest.raises(TuningCancelledError, match="cancel"):
        runner._run_qualification(
            spec,
            Executor(),  # type: ignore[arg-type]
            output,
            build_runtime_fingerprint(spec),
        )


def test_external_tune_rejects_wrong_approval_digest(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    output = tmp_path / "run"
    qualified = run_external_tune(ExternalTuneInput(config=config, output_dir=output))

    result = run_external_tune(
        ExternalTuneInput(
            config=config,
            output_dir=output,
            approval_digest="sha256:" + "0" * 64,
        )
    )

    assert qualified.status == "awaiting_approval"
    assert result.status == "approval_rejected"


def test_external_tune_detects_fingerprint_changes_at_both_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning.external import runner

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    config["optimizer"] = {
        "name": "random",
        "max_trials": 1,
        "seed": 7,
        "replicas": 1,
    }
    output = tmp_path / "run"
    qualified = run_external_tune(ExternalTuneInput(config=config, output_dir=output))
    original = runner.build_runtime_fingerprint
    calls = 0

    def changed_before_optimization(spec):
        nonlocal calls
        calls += 1
        result = original(spec)
        if calls == 2:
            result = {**result, "digest": "sha256:" + "f" * 64}
        return result

    monkeypatch.setattr(
        runner, "build_runtime_fingerprint", changed_before_optimization
    )
    result = run_external_tune(
        ExternalTuneInput(
            config=config,
            output_dir=output,
            approval_digest=qualified.qualification_digest,
        )
    )
    assert result.status == "fingerprint_changed"

    monkeypatch.setattr(runner, "build_runtime_fingerprint", original)
    output = tmp_path / "run-during"
    qualified = run_external_tune(ExternalTuneInput(config=config, output_dir=output))
    calls = 0

    def changed_after_optimization(spec):
        nonlocal calls
        calls += 1
        result = original(spec)
        if calls == 3:
            result = {**result, "digest": "sha256:" + "e" * 64}
        return result

    monkeypatch.setattr(runner, "build_runtime_fingerprint", changed_after_optimization)
    result = run_external_tune(
        ExternalTuneInput(
            config=config,
            output_dir=output,
            approval_digest=qualified.qualification_digest,
        )
    )
    assert result.status == "fingerprint_changed"


def test_external_tune_cancellation_after_optimizer_and_after_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning.external import runner

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    config["optimizer"] = {
        "name": "random",
        "max_trials": 1,
        "seed": 7,
        "replicas": 1,
    }
    output = tmp_path / "after-optimizer"
    qualified = run_external_tune(ExternalTuneInput(config=config, output_dir=output))
    monkeypatch.setattr(runner, "get_runner", lambda _name: lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_cancelled", lambda _event: True)
    result = run_external_tune(
        ExternalTuneInput(
            config=config,
            output_dir=output,
            approval_digest=qualified.qualification_digest,
        )
    )
    assert result.status == "cancelled"

    monkeypatch.undo()
    output = tmp_path / "after-publish"
    qualified = run_external_tune(ExternalTuneInput(config=config, output_dir=output))
    published = False
    original_promote = runner._promote_winner_outputs

    def promote(*args: object, **kwargs: object):
        nonlocal published
        result = original_promote(*args, **kwargs)
        published = True
        return result

    monkeypatch.setattr(runner, "_promote_winner_outputs", promote)
    monkeypatch.setattr(runner, "_cancelled", lambda _event: published)
    result = run_external_tune(
        ExternalTuneInput(
            config=config,
            output_dir=output,
            approval_digest=qualified.qualification_digest,
        )
    )
    assert result.status == "cancelled"
    assert result.error == "external tuning cancelled after optimization"


def test_external_tune_cancellation_after_winner_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning.external import runner

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    config = _config(script)
    config["optimizer"] = {
        "name": "random",
        "max_trials": 1,
        "seed": 7,
        "replicas": 1,
    }
    output = tmp_path / "run"
    qualified = run_external_tune(ExternalTuneInput(config=config, output_dir=output))
    render_finished = False
    frame = output / "render" / "frame.png"

    def render_best_recording(**_kwargs: object) -> list[Path]:
        nonlocal render_finished
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"png")
        render_finished = True
        return [frame]

    monkeypatch.setattr(runner, "_render_best_recording", render_best_recording)
    cancel_event = SimpleNamespace(is_set=lambda: render_finished)

    result = run_external_tune(
        ExternalTuneInput(
            config=config,
            output_dir=output,
            approval_digest=qualified.qualification_digest,
            render_winning_trial=True,
            cancel_event=cancel_event,
        )
    )

    assert result.status == "cancelled"
    assert result.cancelled
    assert result.rendered_frames == [frame]
    assert result.error == "external tuning cancelled after optimization"


def test_arun_external_tune_uses_async_wrapper(tmp_path: Path) -> None:
    script = tmp_path / "trial.py"
    _write_trial_adapter(script)

    result = asyncio.run(
        arun_external_tune(
            ExternalTuneInput(config=_config(script), output_dir=tmp_path / "run")
        )
    )

    assert result.status == "awaiting_approval"


def test_usd_recording_inspection_handles_proxy_scope_and_matrix_ops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pxr import Usd, UsdGeom

    from physics_agent.tuning.external.evidence import inspect_usd_recording

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    settings = load_external_tune_spec(_config(script)).evidence
    assert settings is not None
    recording = tmp_path / "recording.usd"
    recording.write_text("#usda 1.0\n", encoding="utf-8")
    sample_times = [0.0, 1.0, 2.0]

    class Attribute:
        def GetTimeSamples(self) -> list[float]:
            return sample_times

    class Op:
        def GetAttr(self) -> Attribute:
            return Attribute()

        def GetOpType(self) -> str:
            return "transform"

    class Prim:
        def __init__(
            self,
            path: str,
            *,
            instance_proxy: bool = False,
            camera: bool = False,
            xformable: bool = False,
        ) -> None:
            self.path = path
            self.instance_proxy = instance_proxy
            self.camera = camera
            self.xformable = xformable

        def IsInstanceProxy(self) -> bool:
            return self.instance_proxy

        def IsA(self, schema: object) -> bool:
            if schema == "camera":
                return self.camera
            return self.xformable

        def GetPath(self) -> str:
            return self.path

        def GetAttributes(self) -> list[Attribute]:
            return [Attribute()]

        def GetOrderedXformOps(self) -> list[Op]:
            return [Op()]

    class Stage:
        def GetTimeCodesPerSecond(self) -> float:
            return settings.fps

        def GetFramesPerSecond(self) -> float:
            return settings.fps

        def HasAuthoredTimeCodeRange(self) -> bool:
            return True

        def GetStartTimeCode(self) -> float:
            return 0.0

        def GetEndTimeCode(self) -> float:
            return 2.0

        def TraverseAll(self) -> list[Prim]:
            return [
                Prim("/Proxy", instance_proxy=True),
                Prim("/Scope"),
                Prim("/Camera", camera=True, xformable=True),
            ]

    monkeypatch.setattr(Usd, "Stage", SimpleNamespace(Open=lambda _path: Stage()))
    monkeypatch.setattr(UsdGeom, "Camera", "camera")
    monkeypatch.setattr(UsdGeom, "Xformable", lambda prim: prim)
    monkeypatch.setattr(
        UsdGeom,
        "XformOp",
        SimpleNamespace(
            TypeOrient="orient",
            TypeRotateX="rotateX",
            TypeRotateY="rotateY",
            TypeRotateZ="rotateZ",
            TypeRotateXYZ="rotateXYZ",
            TypeRotateXZY="rotateXZY",
            TypeRotateYXZ="rotateYXZ",
            TypeRotateYZX="rotateYZX",
            TypeRotateZXY="rotateZXY",
            TypeRotateZYX="rotateZYX",
            TypeTranslate="translate",
            TypeTransform="transform",
        ),
    )

    descriptor = inspect_usd_recording(
        recording,
        settings,
        relative_path="recording.usd",
    )

    assert descriptor["pose_prim_paths"] == ["/Camera"]
    assert descriptor["camera_prim_paths"] == ["/Camera"]


@pytest.mark.parametrize(
    ("dependency_kind", "recording_text"),
    [
        (
            "sublayer",
            '#usda 1.0\n(subLayers = [@dependency.usda@])\ndef Xform "World" {}\n',
        ),
        (
            "reference",
            "#usda 1.0\n"
            'def Xform "World" (references = @dependency.usda@</Thing>) {}\n',
        ),
        (
            "payload",
            '#usda 1.0\ndef Xform "World" (payload = @dependency.usda@</Thing>) {}\n',
        ),
        (
            "asset",
            "#usda 1.0\n"
            'def Xform "World"\n'
            "{\n"
            "    custom asset inputs:file = @texture.png@\n"
            "}\n",
        ),
    ],
)
def test_usd_recording_rejects_external_dependencies(
    tmp_path: Path,
    dependency_kind: str,
    recording_text: str,
) -> None:
    from physics_agent.tuning.external.evidence import (
        EvidenceValidationError,
        inspect_usd_recording,
    )

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    settings = load_external_tune_spec(_config(script)).evidence
    assert settings is not None
    (tmp_path / "dependency.usda").write_text(
        '#usda 1.0\ndef Xform "Thing" {}\n',
        encoding="utf-8",
    )
    (tmp_path / "texture.png").write_bytes(b"texture")
    recording = tmp_path / f"{dependency_kind}.usda"
    recording.write_text(recording_text, encoding="utf-8")

    with pytest.raises(EvidenceValidationError, match="self-contained single-file"):
        inspect_usd_recording(
            recording,
            settings,
            relative_path=recording.name,
        )


def test_usd_recording_rejects_malformed_dependency_identifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import UsdUtils

    from physics_agent.tuning.external.evidence import (
        EvidenceValidationError,
        inspect_usd_recording,
    )

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    settings = load_external_tune_spec(_config(script)).evidence
    assert settings is not None
    recording = tmp_path / "recording.usda"
    recording.write_text(
        '#usda 1.0\ndef Xform "World" {}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        UsdUtils,
        "ComputeAllDependencies",
        lambda _path: (
            [SimpleNamespace(realPath="", identifier="\0")],
            [],
            [],
        ),
    )

    with pytest.raises(EvidenceValidationError, match="external layers"):
        inspect_usd_recording(
            recording,
            settings,
            relative_path=recording.name,
        )


def test_usd_recording_rejects_unresolved_dependency(tmp_path: Path) -> None:
    from physics_agent.tuning.external.evidence import (
        EvidenceValidationError,
        inspect_usd_recording,
    )

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    settings = load_external_tune_spec(_config(script)).evidence
    assert settings is not None
    recording = tmp_path / "unresolved.usda"
    recording.write_text(
        '#usda 1.0\n(subLayers = [@missing.usda@])\ndef Xform "World" {}\n',
        encoding="utf-8",
    )

    with pytest.raises(EvidenceValidationError, match="unresolved paths"):
        inspect_usd_recording(
            recording,
            settings,
            relative_path=recording.name,
        )


def test_usd_recording_rejects_stale_openusd_layer_cache(tmp_path: Path) -> None:
    from pxr import Usd

    from physics_agent.tuning.external.evidence import (
        EvidenceValidationError,
        inspect_usd_recording,
    )

    script = tmp_path / "trial.py"
    _write_trial_adapter(script)
    settings = load_external_tune_spec(_config(script)).evidence
    assert settings is not None
    dependency = tmp_path / "dependency.usda"
    dependency.write_text(
        '#usda 1.0\ndef Xform "Thing" {}\n',
        encoding="utf-8",
    )
    recording = tmp_path / "recording.usda"
    recording.write_text(
        '#usda 1.0\ndef Xform "World" {}\n',
        encoding="utf-8",
    )
    held_stage = Usd.Stage.Open(str(recording))
    assert held_stage is not None
    recording.write_text(
        '#usda 1.0\ndef Xform "World" (references = @dependency.usda@</Thing>) {}\n',
        encoding="utf-8",
    )

    with pytest.raises(EvidenceValidationError, match="cached layer"):
        inspect_usd_recording(
            recording,
            settings,
            relative_path=recording.name,
        )
