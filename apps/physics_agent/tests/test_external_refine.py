# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract and loop tests for trusted external-runtime refinement."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from physics_agent.tasks.judge_tune import JudgeResult
from physics_agent.tuning import (
    OptimizerSettings,
    ReplicaRecord,
    TrialRecord,
    TunableParam,
    TuningObjective,
)
from physics_agent.tuning.external import (
    ExternalRefineInput,
    ExternalRefineOutput,
    ExternalTuneInput,
    ExternalTuneOutput,
    arun_external_refine,
    load_external_tune_spec,
    run_external_refine,
    run_external_tune,
)
from physics_agent.tuning.external.fingerprint import build_runtime_fingerprint
from physics_agent.tuning.visual_evidence import JudgeVisualEvidence


@pytest.fixture(autouse=True)
def lightweight_runtime_environment_fingerprint() -> Iterator[None]:
    """Keep loop tests focused on refinement rather than environment I/O."""

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


@pytest.fixture(autouse=True)
def playback_render_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Render deterministic PNGs from evaluated recordings without starting OvRTX."""

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


def _write_adapter(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            r"""
            import argparse
            import json
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--request", required=True)
            parser.add_argument("--result", required=True)
            args = parser.parse_args()

            request = json.loads(Path(args.request).read_text())
            gain = float(request["params"]["gain"])
            bias = float(request["params"]["bias"])
            metrics = {
                "distance": abs(gain - 0.8),
                "value": gain,
                "bias_readback": bias,
            }
            objective = dict(request["objective"])
            objective["value"] = abs(gain - 0.8)
            artifacts_dir = Path(request["artifacts_dir"])
            artifacts = {}
            metadata = {"applied_params": request["params"]}
            evidence = request.get("evidence")
            if evidence is not None:
                from PIL import Image, ImageDraw

                width = int(evidence["width"])
                height = int(evidence["height"])
                fps = float(evidence["fps"])
                frame_dir = artifacts_dir / f"{request['purpose']}_frames"
                frame_dir.mkdir(parents=True, exist_ok=True)
                frames = []
                for index in range(6):
                    frame = Image.new("RGB", (width, height), (0, 0, 0))
                    draw = ImageDraw.Draw(frame)
                    start = 2 + index * 4
                    draw.rectangle(
                        (start, 5, start + 10, 18), fill=(20, 180, 240)
                    )
                    frame_path = frame_dir / f"frame_{index:04d}.png"
                    frame.save(frame_path)
                    frames.append(
                        {
                            "path": frame_path.relative_to(artifacts_dir).as_posix(),
                            "timestamp_seconds": index / fps,
                        }
                    )
                manifest = artifacts_dir / f"{request['purpose']}_frames.json"
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
                    "renderer": evidence["renderer"],
                    "width": width,
                    "height": height,
                    "fps": fps,
                }
            recording = request.get("recording")
            if recording is not None:
                recording_path = artifacts_dir / "recording.usd"
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
            for name in request.get("publish_artifacts", []):
                published_path = artifacts_dir / f"{name}.json"
                published_path.write_text(
                    json.dumps({"name": name, "params": request["params"]})
                )
                artifacts[name] = str(published_path)
            result = {
                "status": "ok",
                "success": True,
                "objective": objective,
                "metrics": metrics,
                "metadata": metadata,
                "artifacts": artifacts,
            }
            Path(args.result).write_text(json.dumps(result))
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _config(script: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "scalar_objective_test",
        "runtime": {
            "python": sys.executable,
            "script": str(script),
            "cwd": str(script.parent),
            "timeout_s": 10,
            "fingerprint_paths": [str(script)],
            "trial": {"scene": "fixed"},
        },
        "parameters": {
            "gain": {"min": 0.0, "max": 1.0},
            "bias": {"min": 1.0, "max": 3.0},
        },
        "objective": {
            "name": "absolute_gain_error",
            "unit": "normalized",
            "direction": "minimize",
            "failure_penalty": 1000.0,
        },
        "optimizer": {
            "name": "random",
            "max_trials": 4,
            "seed": 7,
            "replicas": 1,
            "replica_seed": 100,
        },
        "qualification": {
            "nominal_params": {"gain": 0.75, "bias": 2.0},
            "seed": 1000,
        },
        "evidence": {
            "artifact_name": "frames",
            "renderer": "isaac_sim_kit_rtx",
            "media_type": "application/json",
            "width": 48,
            "height": 32,
            "fps": 10,
            "min_frames": 2,
            "require_motion": True,
            "min_frame_stddev": 0.1,
            "min_motion_score": 0.01,
            "camera": {
                "position": [1.0, 1.0, 1.0],
                "target": [0.0, 0.0, 0.0],
            },
            "recording_artifact_name": "recording_usd",
            "playback_renderer": "ovrtx",
            "max_duration_seconds": 1.0,
            "num_sensor_updates": 1,
            "render_mode": "rt2",
        },
    }


class _VLM:
    def __init__(self, decisions: list[str]):
        self.decisions = list(decisions)
        self.calls: list[dict[str, Any]] = []

    def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        decision = self.decisions.pop(0)
        return json.dumps(
            {
                "score": 1.0 if decision == "approve" else 0.4,
                "decision": decision,
                "reasoning": f"test says {decision}",
            }
        )


class _Chat:
    pass


def test_search_changes_do_not_change_qualification_fingerprint(
    tmp_path: Path,
) -> None:
    script = tmp_path / "adapter.py"
    _write_adapter(script)
    spec = load_external_tune_spec(_config(script))
    changed = replace(
        spec,
        params=(TunableParam("gain", 0.1, 0.4),),
        optimizer=OptimizerSettings(
            name="random",
            max_trials=2,
            seed=88,
            replicas=2,
            replica_seed=100,
        ),
    )

    assert (
        build_runtime_fingerprint(spec)["digest"]
        == build_runtime_fingerprint(changed)["digest"]
    )
    changed_runtime = replace(
        changed,
        runtime=replace(changed.runtime, trial={"scene": "different"}),
    )
    assert (
        build_runtime_fingerprint(spec)["digest"]
        != build_runtime_fingerprint(changed_runtime)["digest"]
    )
    changed_objective = replace(
        changed,
        objective=TuningObjective("different_objective", "normalized"),
    )
    assert (
        build_runtime_fingerprint(spec)["digest"]
        != build_runtime_fingerprint(changed_objective)["digest"]
    )


def test_trial_metrics_are_optional_diagnostics(
    tmp_path: Path,
) -> None:
    script = tmp_path / "adapter.py"
    _write_adapter(script)
    script.write_text(
        script.read_text(encoding="utf-8").replace(
            '"metrics": metrics,', '"metrics": {},'
        ),
        encoding="utf-8",
    )

    result = run_external_tune(
        ExternalTuneInput(config=_config(script), output_dir=tmp_path / "run")
    )

    assert result.success
    assert result.status == "awaiting_approval"


def test_approved_iteration_allows_bounds_excluding_nominal_and_active_subset(
    tmp_path: Path,
) -> None:
    script = tmp_path / "adapter.py"
    _write_adapter(script)
    spec = load_external_tune_spec(_config(script))
    qualification_dir = tmp_path / "qualification"
    qualified = run_external_tune(
        ExternalTuneInput(config=spec, output_dir=qualification_dir)
    )
    changed = replace(
        spec,
        params=(TunableParam("gain", 0.1, 0.4),),
    )
    result = run_external_tune(
        ExternalTuneInput(
            config=changed,
            output_dir=tmp_path / "iteration",
            qualification_dir=qualification_dir,
            approval_digest=qualified.qualification_digest,
            fixed_params={"gain": 0.75, "bias": 2.0},
        )
    )

    assert result.success
    assert result.n_trials == 4
    assert all(set(record.params) == {"gain", "bias"} for record in result.history)
    assert all(record.params["bias"] == 2.0 for record in result.history)
    assert all(0.1 <= record.params["gain"] <= 0.4 for record in result.history)
    run_spec = json.loads((tmp_path / "iteration" / "run_spec.json").read_text())
    assert run_spec["fixed_params"] == {"bias": 2.0}


def test_wrong_approval_digest_does_not_delete_existing_iterations(
    tmp_path: Path,
) -> None:
    script = tmp_path / "adapter.py"
    _write_adapter(script)
    output_dir = tmp_path / "refine"
    qualified = run_external_refine(
        ExternalRefineInput(
            config=_config(script),
            output_dir=output_dir,
            user_prompt="goal",
        )
    )
    sentinel = output_dir / "iter_1" / "keep.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("keep", encoding="utf-8")

    result = run_external_refine(
        ExternalRefineInput(
            config=_config(script),
            output_dir=output_dir,
            user_prompt="goal",
            approval_digest="sha256:" + "0" * 64,
            chat_model=_Chat(),
            vlm_model=_VLM(["approve"]),
        )
    )

    assert qualified.status == "awaiting_approval"
    assert not result.success
    assert "digest does not match" in str(result.error)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_external_refine_rejects_nested_output_before_writing(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    script = runtime_root / "adapter.py"
    _write_adapter(script)
    config = _config(script)
    config["runtime"]["fingerprint_paths"] = [str(runtime_root)]
    output_dir = runtime_root / "output"

    result = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output_dir,
            user_prompt="goal",
        )
    )

    assert not result.success
    assert result.status == "invalid_output_layout"
    assert "output_dir must not be inside" in str(result.error)
    assert not output_dir.exists()


@pytest.mark.parametrize("changed_input", ["runtime", "qualification_frames"])
def test_external_refine_validates_qualification_before_cleanup(
    tmp_path: Path,
    changed_input: str,
) -> None:
    script = tmp_path / "adapter.py"
    _write_adapter(script)
    config = _config(script)
    output_dir = tmp_path / "refine"
    qualified = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output_dir,
            user_prompt="goal",
        )
    )
    stale_iteration = output_dir / "iter_1" / "keep.txt"
    stale_iteration.parent.mkdir(parents=True)
    stale_iteration.write_text("iteration", encoding="utf-8")
    stale_final = output_dir / "final" / "keep.txt"
    stale_final.parent.mkdir()
    stale_final.write_text("final", encoding="utf-8")

    if changed_input == "runtime":
        script.write_text(
            script.read_text(encoding="utf-8") + "\n# changed\n",
            encoding="utf-8",
        )
        expected_error = "runtime inputs changed"
    else:
        qualification_frames = qualified.artifacts["qualification_frames"]
        qualification_frames.write_bytes(qualification_frames.read_bytes() + b"changed")
        expected_error = "frame evidence changed"

    result = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output_dir,
            user_prompt="goal",
            approval_digest=qualified.qualification_digest,
            chat_model=_Chat(),
            vlm_model=_VLM(["approve"]),
        )
    )

    assert not result.success
    assert expected_error in str(result.error)
    assert stale_iteration.read_text(encoding="utf-8") == "iteration"
    assert stale_final.read_text(encoding="utf-8") == "final"


def test_external_refine_input_coerces_max_iterations(tmp_path: Path) -> None:
    params = ExternalRefineInput(
        config={},
        output_dir=tmp_path,
        user_prompt="goal",
        max_iterations=2.0,  # type: ignore[arg-type]
    )

    assert params.max_iterations == 2
    assert isinstance(params.max_iterations, int)


def test_external_refine_qualifies_once_and_publishes_existing_iteration(
    tmp_path: Path,
    playback_render_calls: list[Path],
) -> None:
    script = tmp_path / "adapter.py"
    _write_adapter(script)
    config = _config(script)
    config["publish_artifacts"] = ["tuned_config"]
    output_dir = tmp_path / "refine"
    qualified = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output_dir,
            user_prompt="match the desired behavior",
            max_iterations=1,
        )
    )
    assert qualified.status == "awaiting_approval"
    stale_final = output_dir / "final"
    stale_final.mkdir()
    stale_iteration = output_dir / "iter_99"
    stale_iteration.mkdir()
    stale_reference = output_dir / "reference_media" / "stale.txt"
    stale_reference.parent.mkdir()
    stale_reference.write_text("stale", encoding="utf-8")

    vlm = _VLM(["approve"])
    result = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output_dir,
            user_prompt="match the desired behavior",
            approval_digest=qualified.qualification_digest,
            max_iterations=1,
            vlm_model=vlm,
        )
    )

    assert result.success
    assert result.validated
    assert result.status == "completed"
    assert not stale_iteration.exists()
    assert result.termination_reason == "approved"
    assert len(result.iterations) == 1
    assert result.final_dir is not None
    assert not (result.final_dir / "reference_media").exists()
    final_frames = sorted((result.final_dir / "render").glob("*.png"))
    assert len(final_frames) == 6
    assert not list(result.final_dir.rglob("*.mp4"))
    assert (result.final_dir / "best_recording.usd").is_file()
    assert (result.final_dir / "best_params.json").is_file()
    assert (result.final_dir / "objective.json").is_file()
    assert (result.final_dir / "run_spec.json").is_file()
    assert (result.final_dir / "search.json").is_file()
    assert (result.final_dir / "history.jsonl").is_file()
    assert (result.final_dir / "external_tune_results.json").is_file()
    assert (result.final_dir / "judge.json").is_file()
    assert (result.final_dir / "refine_request.json").is_file()
    assert (result.final_dir / "manifest.json").is_file()
    assert (
        result.final_dir / "outputs" / "tuned_config" / "tuned_config.json"
    ).is_file()
    assert not (result.final_dir / "optimization").exists()
    assert not (result.final_dir / "scoring.json").exists()
    assert not (result.final_dir / "measurements.json").exists()
    assert (
        len(list((output_dir / "qualification" / "qualification").glob("trial_*"))) == 1
    )
    request_purposes = {
        json.loads(path.read_text())["purpose"]
        for path in output_dir.rglob("request.json")
    }
    assert "baseline" not in request_purposes
    assert "final_validation" not in request_purposes
    assert "selected_evidence" not in request_purposes
    assert len(playback_render_calls) == 1
    assert not (output_dir / "iter_1" / "selected_frames").exists()
    generated_paths = [
        Path(path)
        for caption, path in vlm.calls[0]["image_caption_pairs"]
        if caption.startswith("Generated Physics Output")
    ]
    assert generated_paths
    assert all(
        path.parent == output_dir / "iter_1" / "render" for path in generated_paths
    )
    final_best_params = json.loads((result.final_dir / "best_params.json").read_text())
    assert set(final_best_params) == {"best_score", "params"}
    final_result = json.loads((result.final_dir / "result.json").read_text())
    assert final_result["selected_evidence"]["replica_index"] == 0
    assert final_result["selected_evidence"]["seed"] == 100
    assert final_result["selected_evidence"]["scored_recording"]["path"] == (
        "best_recording.usd"
    )
    assert final_result["published_outputs"]["tuned_config"]["sha256"].startswith(
        "sha256:"
    )
    final_history = [
        json.loads(line)
        for line in (result.final_dir / "history.jsonl").read_text().splitlines()
    ]
    assert final_history
    assert all(
        not {"artifacts", "artifact_metadata", "trial_dir"} & set(replica)
        for trial in final_history
        for replica in trial["replicas"]
    )
    final_tune = json.loads(
        (result.final_dir / "external_tune_results.json").read_text()
    )
    assert final_tune["history"] == final_history
    assert final_tune["selected_evidence"]["scored_recording"]["path"] == (
        "best_recording.usd"
    )
    assert final_tune["artifacts"]["best_recording"] == "best_recording.usd"
    final_judge = json.loads((result.final_dir / "judge.json").read_text())
    visual = final_judge["extra"]["visual_evidence"]
    assert all(
        item["path"].startswith("render/") for item in visual["generated_images"]
    )
    assert visual["comparison_image"] is None
    manifest = json.loads((result.final_dir / "manifest.json").read_text())
    assert "optimization" not in "\n".join(manifest["artifacts"])
    assert "outputs/tuned_config/tuned_config.json" in manifest["artifacts"]
    summary = json.loads((output_dir / "external_refine_summary.json").read_text())
    assert summary["validated"] is True
    assert summary["request"]["user_prompt"] == "match the desired behavior"
    assert summary["request"]["settings"]["max_iterations"] == 1
    assert summary["artifacts"]["manifest"] == "final/manifest.json"
    assert (
        result.artifacts["best_recording"].read_bytes()
        == playback_render_calls[0].read_bytes()
    )


def test_external_refine_fails_closed_when_winner_png_render_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions import graphics

    script = tmp_path / "adapter.py"
    _write_adapter(script)
    config = _config(script)
    output_dir = tmp_path / "refine"
    qualified = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output_dir,
            user_prompt="match the desired behavior",
        )
    )

    def fail_render(*_args: object, **_kwargs: object) -> list[Path]:
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(graphics, "render_time_sampled_usd", fail_render)
    vlm = _VLM(["approve"])
    result = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output_dir,
            user_prompt="match the desired behavior",
            approval_digest=qualified.qualification_digest,
            max_iterations=1,
            vlm_model=vlm,
        )
    )

    assert not result.success
    assert result.termination_reason == "error"
    assert (
        result.iterations[0].error
        == "VLM unavailable: generated evidence failed: RuntimeError"
    )
    assert vlm.calls == []
    assert (output_dir / "iter_1" / "external_tune_results.json").is_file()
    assert not (output_dir / "iter_1" / "render").exists()


def test_external_refine_changes_search_bounds_but_not_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "adapter.py"
    _write_adapter(script)
    config = _config(script)
    output_dir = tmp_path / "refine"
    qualified = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output_dir,
            user_prompt="target value 0.3",
        )
    )

    refined_payload = {
        "search": {"parameters": {"gain": {"min": 0.1, "max": 0.4}}},
        "reasoning": "focus the search around the promising region",
    }
    monkeypatch.setattr(
        "physics_agent.tuning.external.refiner.generate_chat_response",
        lambda *_args, **_kwargs: {"response": json.dumps(refined_payload)},
    )
    result = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output_dir,
            user_prompt="target value 0.3",
            approval_digest=qualified.qualification_digest,
            max_iterations=2,
            chat_model=_Chat(),
            vlm_model=_VLM(["continue", "approve"]),
        )
    )

    assert result.success
    assert result.termination_reason == "approved"
    assert len(result.iterations) == 2
    assert result.iterations[0].judge is not None
    assert result.iterations[0].judge["programmatic_score"] == 1.0
    assert result.iterations[0].judge["llm_score"] == 0.4
    assert result.iterations[0].judge["score"] == pytest.approx(0.76)
    assert [iteration.n_trials for iteration in result.iterations] == [4, 4]
    assert result.iterations[1].active_search == {"gain": {"min": 0.1, "max": 0.4}}
    assert result.iterations[1].objective == result.iterations[0].objective
    assert result.iterations[1].objective == {
        "name": "absolute_gain_error",
        "unit": "normalized",
        "direction": "minimize",
    }


def test_external_refine_helpers_cover_events_timeouts_and_paths(
    tmp_path: Path,
) -> None:
    from physics_agent.tuning.external import refine

    events: list[str] = []
    refine._emit(
        SimpleNamespace(event=lambda event_type, _data: events.append(event_type)),
        "refine",
        {},
    )
    assert events == ["refine"]
    assert refine._run_with_timeout(lambda value: value + 1, 2, timeout_seconds=0) == 3
    with pytest.raises(RuntimeError, match="provider failed"):
        refine._run_with_timeout(
            lambda: (_ for _ in ()).throw(RuntimeError("provider failed")),
            timeout_seconds=1,
        )

    with pytest.raises(FileNotFoundError, match="reference media"):
        ExternalRefineInput(
            config={},
            output_dir=tmp_path,
            user_prompt="goal",
            reference_images=[tmp_path / "missing.png"],
        )

    video = tmp_path / "reference.mp4"
    video.write_bytes(b"video")
    with pytest.raises(ValueError, match="Unsupported reference media extension"):
        ExternalRefineInput(
            config={},
            output_dir=tmp_path,
            user_prompt="goal",
            reference_images=[video],
        )

    class Model:
        model = "test-model"

    assert refine._model_identity(Model())["model"] == "test-model"
    output = ExternalRefineOutput(success=False, output_dir=None)
    assert refine._persist_summary(output) is output

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    payload = refine._summary_payload(
        ExternalRefineOutput(
            success=True,
            output_dir=root,
            artifacts={"outside": outside},
        )
    )
    assert payload["artifacts"]["outside"] == str(outside.resolve())


def test_portable_judge_payload_validates_generated_and_comparison_paths(
    tmp_path: Path,
) -> None:
    from physics_agent.tuning.external import refine

    iteration = tmp_path / "iter_1"
    render = iteration / "render"
    render.mkdir(parents=True)
    frame = render / "frame.png"
    frame.write_bytes(b"png")
    judge_path = iteration / "judge.json"

    judge_path.write_text('{"decision":"continue"}', encoding="utf-8")
    assert refine._portable_judge_payload(judge_path, iteration) == {
        "decision": "continue"
    }

    payload = {
        "extra": {
            "visual_evidence": {
                "generated_images": [None, {"path": str(tmp_path / "outside.png")}],
                "comparison_image": None,
            }
        }
    }
    judge_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frame escaped"):
        refine._portable_judge_payload(judge_path, iteration)

    payload["extra"]["visual_evidence"]["generated_images"] = [{"path": str(frame)}]
    payload["extra"]["visual_evidence"]["comparison_image"] = str(
        tmp_path / "outside-comparison.png"
    )
    judge_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="comparison escaped"):
        refine._portable_judge_payload(judge_path, iteration)

    comparison = iteration / "comparison.png"
    payload["extra"]["visual_evidence"]["comparison_image"] = str(comparison)
    judge_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="comparison is missing"):
        refine._portable_judge_payload(judge_path, iteration)

    comparison.write_bytes(b"png")
    portable = refine._portable_judge_payload(judge_path, iteration)
    visual = portable["extra"]["visual_evidence"]
    assert visual["generated_images"][0]["path"] == "render/frame.png"
    assert visual["comparison_image"] == "comparison.png"

    reference = tmp_path / "reference_media" / "images" / "reference_image_01.png"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"png")
    payload["extra"]["visual_evidence"]["reference_images"] = [
        None,
        {"caption": "missing path"},
        {"caption": "reference", "path": str(reference)},
    ]
    judge_path.write_text(json.dumps(payload), encoding="utf-8")
    portable = refine._portable_judge_payload(judge_path, iteration)
    references = portable["extra"]["visual_evidence"]["reference_images"]
    assert references[:2] == [None, {"caption": "missing path"}]
    assert references[2]["path"] == ("reference_media/images/reference_image_01.png")


def test_portable_request_payload_skips_malformed_optional_media(
    tmp_path: Path,
) -> None:
    from physics_agent.tuning.external import refine

    request_with_invalid_container = {"reference_media": []}
    assert (
        refine._portable_request_payload(request_with_invalid_container, tmp_path)
        == request_with_invalid_container
    )

    root = tmp_path / "refine"
    reference = root / "reference_media" / "images" / "reference_image_02.png"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"png")
    request = {
        "reference_media": {
            "images": [
                {"caption": "missing path"},
                {"path": str(tmp_path / "original.png")},
            ],
        }
    }

    portable = refine._portable_request_payload(request, root)

    assert portable["reference_media"]["images"] == [
        {"caption": "missing path"},
        {"path": "reference_media/images/reference_image_02.png"},
    ]
    assert request["reference_media"]["images"][1]["path"] == str(
        tmp_path / "original.png"
    )


@pytest.mark.parametrize(
    ("path_kind", "message"),
    [
        ("escaped", "escaped"),
        ("unmanaged", "outside managed"),
        ("missing", "missing"),
        ("symlink", "must not use symlinks"),
    ],
)
def test_portable_judge_payload_rejects_invalid_reference_paths(
    tmp_path: Path,
    path_kind: str,
    message: str,
) -> None:
    from physics_agent.tuning.external import refine

    iteration = tmp_path / "iter_1"
    iteration.mkdir()
    judge_path = iteration / "judge.json"
    if path_kind == "escaped":
        reference = tmp_path.parent / f"{tmp_path.name}-outside.png"
        reference.write_bytes(b"png")
    elif path_kind == "unmanaged":
        reference = tmp_path / "other" / "reference.png"
        reference.parent.mkdir()
        reference.write_bytes(b"png")
    elif path_kind == "missing":
        reference = tmp_path / "reference_media" / "images" / "missing.png"
    else:
        outside = tmp_path.parent / f"{tmp_path.name}-symlink-target.png"
        outside.write_bytes(b"png")
        reference = tmp_path / "reference_media" / "images" / "linked.png"
        reference.parent.mkdir(parents=True)
        reference.symlink_to(outside)
    judge_path.write_text(
        json.dumps(
            {
                "extra": {
                    "visual_evidence": {
                        "reference_images": [
                            {"caption": "reference", "path": str(reference)}
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        refine._portable_judge_payload(judge_path, iteration)


def test_publish_final_cleans_partial_temporary_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning.external import refine

    script = tmp_path / "adapter.py"
    _write_adapter(script)
    spec = load_external_tune_spec(_config(script))
    root = tmp_path / "refine"
    root.mkdir()

    def fail_build(*, final_dir: Path, **_kwargs: object) -> dict[str, Path]:
        final_dir.mkdir()
        (final_dir / "partial").write_text("partial", encoding="utf-8")
        raise ValueError("bundle failed")

    monkeypatch.setattr(refine, "_build_final", fail_build)
    with pytest.raises(ValueError, match="bundle failed"):
        refine._publish_final(
            root=root,
            iteration_dir=root / "iter_1",
            spec=spec,
            tune=ExternalTuneOutput(success=False),
            request={},
        )
    assert not (root / ".final.tmp").exists()


def test_build_final_rejects_rendered_frame_outside_iteration(
    tmp_path: Path,
) -> None:
    from physics_agent.tuning.external import refine

    script = tmp_path / "adapter.py"
    _write_adapter(script)
    spec = load_external_tune_spec(_config(script))
    iteration = tmp_path / "iter_1"
    iteration.mkdir()
    for name in ("run_spec.json", "search.json", "best_params.json", "judge.json"):
        (iteration / name).write_text("{}", encoding="utf-8")
    recording = iteration / "best_recording.usd"
    recording.write_text("#usda 1.0\n", encoding="utf-8")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")
    tune = ExternalTuneOutput(
        success=True,
        status="completed",
        best_score=0.1,
        best_objective=0.1,
        artifacts={"best_recording": recording},
        rendered_frames=[outside],
    )

    with pytest.raises(ValueError, match="frame escaped"):
        refine._build_final(
            final_dir=tmp_path / "final",
            iteration_dir=iteration,
            spec=spec,
            tune=tune,
            request={},
        )


def test_external_refine_cancels_before_and_during_iterations(
    tmp_path: Path,
) -> None:
    script = tmp_path / "adapter.py"
    _write_adapter(script)
    config = _config(script)

    cancelled = SimpleNamespace(is_set=lambda: True)
    during_qualification = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=tmp_path / "qualification-cancelled",
            user_prompt="goal",
            cancel_event=cancelled,
        )
    )
    assert during_qualification.status == "cancelled"
    assert during_qualification.termination_reason == "cancelled"
    assert during_qualification.cancelled

    before = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=tmp_path / "before",
            user_prompt="goal",
            approval_digest="sha256:" + "a" * 64,
            cancel_event=cancelled,
        )
    )
    assert before.status == "cancelled"

    output = tmp_path / "during"
    qualified = run_external_refine(
        ExternalRefineInput(config=config, output_dir=output, user_prompt="goal")
    )

    class CancelAtLoop:
        calls = 0

        def is_set(self) -> bool:
            self.calls += 1
            return self.calls >= 2

    during = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output,
            user_prompt="goal",
            approval_digest=qualified.qualification_digest,
            cancel_event=CancelAtLoop(),
        )
    )
    assert during.status == "cancelled"
    assert during.iterations == []


def test_external_refine_records_failed_tune_and_missing_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning.external import refine

    script = tmp_path / "adapter.py"
    _write_adapter(script)
    config = _config(script)
    output = tmp_path / "failed"
    qualified = run_external_refine(
        ExternalRefineInput(config=config, output_dir=output, user_prompt="goal")
    )
    monkeypatch.setattr(
        refine,
        "run_external_tune",
        lambda _params: ExternalTuneOutput(
            success=False,
            status="optimization_failed",
            error="optimizer failed",
        ),
    )
    failed = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output,
            user_prompt="goal",
            approval_digest=qualified.qualification_digest,
        )
    )
    assert failed.termination_reason == "error"
    assert failed.iterations[0].error == "optimizer failed"

    output = tmp_path / "no-frames"
    monkeypatch.undo()
    qualified = run_external_refine(
        ExternalRefineInput(config=config, output_dir=output, user_prompt="goal")
    )
    monkeypatch.setattr(
        refine,
        "run_external_tune",
        lambda _params: ExternalTuneOutput(
            success=True,
            status="completed",
            best_params={"gain": 0.8, "bias": 2.0},
            best_score=0.0,
            best_objective=0.0,
        ),
    )
    no_frames = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output,
            user_prompt="goal",
            approval_digest=qualified.qualification_digest,
            vlm_model=_VLM(["approve"]),
        )
    )
    assert no_frames.termination_reason == "error"
    assert "renderer produced no frames" in str(no_frames.iterations[0].error)


def test_external_refine_publishes_max_iteration_and_comparison(
    tmp_path: Path,
) -> None:
    from PIL import Image

    script = tmp_path / "adapter.py"
    _write_adapter(script)
    config = _config(script)
    reference = tmp_path / "reference.png"
    Image.new("RGB", (48, 32), (128, 128, 128)).save(reference)
    output = tmp_path / "refine"
    qualified = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output,
            user_prompt="goal",
            reference_images=[reference],
        )
    )

    result = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output,
            user_prompt="goal",
            approval_digest=qualified.qualification_digest,
            reference_images=[reference],
            max_iterations=1,
            vlm_model=_VLM(["continue"]),
        )
    )

    assert result.success
    assert result.termination_reason == "max_iterations"
    assert result.status == "completed"
    assert not result.validated
    assert result.final_dir is not None
    assert (result.final_dir / "comparison.png").is_file()
    final_reference = (
        result.final_dir / "reference_media" / "images" / "reference_image_01.png"
    )
    assert final_reference.read_bytes() == reference.read_bytes()
    final_judge = json.loads((result.final_dir / "judge.json").read_text())
    assert (
        final_judge["extra"]["visual_evidence"]["reference_images"][0]["path"]
        == "reference_media/images/reference_image_01.png"
    )
    final_request = json.loads((result.final_dir / "refine_request.json").read_text())
    assert final_request["reference_media"]["images"][0]["path"] == (
        "reference_media/images/reference_image_01.png"
    )
    summary = json.loads((output / "external_refine_summary.json").read_text())
    assert summary["validated"] is False


def test_external_refine_rejects_terminal_result_without_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning.external import refine

    script = tmp_path / "adapter.py"
    _write_adapter(script)
    config = _config(script)
    output = tmp_path / "refine"
    qualified = run_external_refine(
        ExternalRefineInput(config=config, output_dir=output, user_prompt="goal")
    )
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"png")
    recording = tmp_path / "recording.usd"
    recording.write_text("#usda 1.0\n", encoding="utf-8")
    history = [
        TrialRecord(
            trial_index=0,
            params={"gain": 0.8, "bias": 2.0},
            score=0.0,
            objective_value=None,
            replicas=[ReplicaRecord(seed=1, objective_value=None, success=True)],
        )
    ]
    monkeypatch.setattr(
        refine,
        "run_external_tune",
        lambda _params: ExternalTuneOutput(
            success=True,
            status="completed",
            best_params={"gain": 0.8, "bias": 2.0},
            best_score=0.0,
            best_objective=None,
            history=history,
            rendered_frames=[frame],
            artifacts={"best_recording": recording},
        ),
    )
    monkeypatch.setattr(
        refine,
        "run_external_judge",
        lambda **_kwargs: JudgeResult(
            decision="approve",
            score=1.0,
            programmatic_score=1.0,
            llm_score=1.0,
            reasoning="approved",
            iterations=1,
        ),
    )

    result = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output,
            user_prompt="goal",
            approval_digest=qualified.qualification_digest,
        )
    )

    assert not result.success
    assert result.iterations[0].error == "selected candidate has no objective value"


def test_external_refine_cancels_after_judge_before_final_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from threading import Event

    from physics_agent.tuning.external import refine

    script = tmp_path / "adapter.py"
    _write_adapter(script)
    config = _config(script)
    output = tmp_path / "refine"
    qualified = run_external_refine(
        ExternalRefineInput(config=config, output_dir=output, user_prompt="goal")
    )
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"png")
    recording = tmp_path / "recording.usd"
    recording.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setattr(
        refine,
        "run_external_tune",
        lambda _params: ExternalTuneOutput(
            success=True,
            status="completed",
            best_params={"gain": 0.8, "bias": 2.0},
            best_score=0.0,
            best_objective=0.0,
            rendered_frames=[frame],
            artifacts={"best_recording": recording},
        ),
    )
    cancel_event = Event()

    def approve_and_cancel(**_kwargs: object) -> JudgeResult:
        cancel_event.set()
        return JudgeResult(
            decision="approve",
            score=1.0,
            programmatic_score=1.0,
            llm_score=1.0,
            reasoning="approved",
            iterations=1,
        )

    monkeypatch.setattr(refine, "run_external_judge", approve_and_cancel)

    result = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output,
            user_prompt="goal",
            approval_digest=qualified.qualification_digest,
            cancel_event=cancel_event,
        )
    )

    assert result.status == "cancelled"
    assert result.termination_reason == "cancelled"
    assert result.cancelled
    assert result.final_dir is None
    assert not (output / "final").exists()
    assert result.iterations[0].error == "external refinement cancelled after judge"


def test_external_refine_cancels_during_final_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from threading import Event

    from physics_agent.tuning.external import refine

    script = tmp_path / "adapter.py"
    _write_adapter(script)
    config = _config(script)
    output = tmp_path / "refine"
    qualified = run_external_refine(
        ExternalRefineInput(config=config, output_dir=output, user_prompt="goal")
    )
    cancel_event = Event()
    publish_final = refine._publish_final

    def publish_and_cancel(**kwargs: Any) -> tuple[Path, dict[str, Path]]:
        published = publish_final(**kwargs)
        cancel_event.set()
        return published

    monkeypatch.setattr(refine, "_publish_final", publish_and_cancel)

    result = run_external_refine(
        ExternalRefineInput(
            config=config,
            output_dir=output,
            user_prompt="goal",
            approval_digest=qualified.qualification_digest,
            max_iterations=1,
            vlm_model=_VLM(["approve"]),
            cancel_event=cancel_event,
        )
    )

    assert result.status == "cancelled"
    assert result.termination_reason == "cancelled"
    assert result.cancelled
    assert result.final_dir is None
    assert result.final_objective is None
    assert result.final_optimizer_loss is None
    assert not (output / "final").exists()
    assert result.iterations[0].error == (
        "external refinement cancelled during final publication"
    )


@pytest.mark.parametrize(
    "response",
    [
        None,
        "",
        "not json",
        "prefix {broken} suffix",
        "[]",
        '{"score": true, "decision": "approve", "reasoning": "ok"}',
        '{"score": NaN, "decision": "approve", "reasoning": "ok"}',
        '{"score": 1, "decision": "maybe", "reasoning": "ok"}',
        '{"score": 1, "decision": "approve", "reasoning": ""}',
    ],
)
def test_external_judge_rejects_invalid_responses(response: Any) -> None:
    from physics_agent.tuning.external import judge

    assert judge._parse_response(response) is None


def test_external_judge_programmatic_and_availability_failures(
    tmp_path: Path,
) -> None:
    from physics_agent.tuning.external import judge

    script = tmp_path / "adapter.py"
    _write_adapter(script)
    spec = load_external_tune_spec(_config(script))
    assert judge._programmatic_score(ExternalTuneOutput(success=False)) == (
        0.0,
        "optimization did not produce a usable history",
        True,
    )

    failed_history = [
        TrialRecord(
            trial_index=0,
            params={"gain": 0.5},
            score=float("inf"),
            failed=True,
            replicas=[
                ReplicaRecord(
                    seed=1,
                    objective_value=None,
                    success=False,
                    error="failed",
                )
            ],
        )
    ]
    broken = ExternalTuneOutput(
        success=True,
        history=failed_history,
        best_score=float("inf"),
        best_objective=None,
    )
    score, critique, hard_failure = judge._programmatic_score(broken)
    assert score == 0.0
    assert "trials failed" in critique
    assert "recording is unavailable" in critique
    assert "objective is non-finite" in critique
    assert hard_failure

    for evidence, model, expected in (
        (
            JudgeVisualEvidence(reference_error="bad reference"),
            _VLM(["approve"]),
            "reference evidence failed",
        ),
        (JudgeVisualEvidence(), _VLM(["approve"]), "produced no judge frames"),
    ):
        result = judge.run_external_judge(
            spec=spec,
            output=broken,
            user_prompt="goal",
            vlm_model=model,
            visual_evidence=evidence,
            score_threshold=0.7,
            iteration=1,
        )
        assert result.llm_unavailable
        assert expected in result.llm_critique

    frame = tmp_path / "frame.png"
    frame.write_bytes(b"png")
    result = judge.run_external_judge(
        spec=spec,
        output=broken,
        user_prompt="goal",
        vlm_model=None,
        visual_evidence=JudgeVisualEvidence(generated_image_paths=(frame,)),
        score_threshold=0.7,
        iteration=1,
    )
    assert "no VLM model supplied" in result.llm_critique

    invalid_model = SimpleNamespace(
        generate_with_image_caption_pairs=lambda **_kwargs: "invalid response"
    )
    result = judge.run_external_judge(
        spec=spec,
        output=broken,
        user_prompt="goal",
        vlm_model=invalid_model,
        visual_evidence=JudgeVisualEvidence(generated_image_paths=(frame,)),
        score_threshold=0.7,
        iteration=1,
    )
    assert "not valid judge JSON" in result.llm_critique


def test_external_refiner_extracts_wrapped_json_and_rejects_bad_wrappers() -> None:
    from physics_agent.tuning.external import refiner

    payload = refiner._json_object(
        'answer: {"search":{"parameters":{"gain":{"min":0,"max":1}}},'
        '"reasoning":"focus"} end'
    )
    assert payload["reasoning"] == "focus"
    with pytest.raises(refiner.ExternalRefineError, match="not JSON"):
        refiner._json_object("no object here")
    with pytest.raises(refiner.ExternalRefineError, match="not valid JSON"):
        refiner._json_object("prefix {broken} suffix")


def test_arun_external_refine_uses_async_wrapper(tmp_path: Path) -> None:
    script = tmp_path / "adapter.py"
    _write_adapter(script)

    result = asyncio.run(
        arun_external_refine(
            ExternalRefineInput(
                config=_config(script),
                output_dir=tmp_path / "run",
                user_prompt="goal",
            )
        )
    )

    assert result.status == "awaiting_approval"
