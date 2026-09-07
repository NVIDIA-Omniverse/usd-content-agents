# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protocol tests for the import-safe IsaacLab BYOR reference adapter."""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pxr import Usd


def _load_adapter() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "byor_isaaclab"
        / "isaaclab_bounce_trial.py"
    )
    spec = importlib.util.spec_from_file_location("byor_isaaclab_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(artifacts_dir: Path) -> dict[str, Any]:
    return {
        "purpose": "optimization",
        "artifacts_dir": str(artifacts_dir),
        "trial": {
            "dt_s": 1.0 / 30.0,
            "target_bounce_height_m": 0.55,
        },
        "recording": {
            "artifact_name": "recording_usd",
            "media_type": "model/vnd.usd",
            "fps": 30.0,
            "camera": {
                "position": [2.4, -2.4, 1.6],
                "target": [0.0, 0.0, 0.45],
            },
        },
    }


def _trial_request(artifacts_dir: Path, *, steps: Any) -> dict[str, Any]:
    return {
        "purpose": "optimization",
        "artifacts_dir": str(artifacts_dir),
        "params": {"restitution": 0.5},
        "seed": 1000,
        "trial": {
            "drop_height_m": 1.0,
            "target_bounce_height_m": 0.55,
            "steps": steps,
            "dt_s": 1.0 / 120.0,
        },
        "objective": {
            "name": "bounce_height_error",
            "unit": "m",
            "direction": "minimize",
        },
    }


@pytest.mark.parametrize("steps", [0, -1, 10_001, True, 360.0, "360", None])
def test_adapter_rejects_unbounded_or_non_integer_steps_before_runtime_import(
    tmp_path: Path,
    steps: Any,
) -> None:
    adapter = _load_adapter()

    with pytest.raises(ValueError, match=r"trial\.steps must be an integer between"):
        adapter._run_trial(
            _trial_request(tmp_path, steps=steps),
            SimpleNamespace(device="cpu"),
        )


@pytest.mark.parametrize("steps", [1, 10_000])
def test_adapter_accepts_bounded_step_count(steps: int) -> None:
    adapter = _load_adapter()

    assert adapter._validated_trial_steps(steps) == steps


def test_main_rejects_invalid_steps_before_isaaclab_app_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _load_adapter()
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(_trial_request(tmp_path, steps=10_001)), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "isaaclab_bounce_trial.py",
            "--request",
            str(request_path),
            "--result",
            str(tmp_path / "result.json"),
        ],
    )
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "isaaclab.app":
            pytest.fail("invalid trial.steps reached the IsaacLab app import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ValueError, match=r"trial\.steps must be an integer between"):
        adapter.main()


def test_adapter_builds_protocol_result_and_time_sampled_recording(
    tmp_path: Path,
) -> None:
    adapter = _load_adapter()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    request = _request(artifacts_dir)
    poses = [
        (0.0, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0)),
        (1.0 / 30.0, (0.0, 0.0, 0.8), (1.0, 0.0, 0.0, 0.0)),
    ]

    result = adapter._build_result(
        request,
        artifacts_dir=artifacts_dir,
        restitution=0.7,
        trajectory=[1.0, 0.8],
        contacted=True,
        bounce_height=0.6,
        captured_frames=0,
        evidence_manifest_path=None,
        recorded_poses=poses,
    )

    assert result["status"] == "ok"
    assert result["success"] is True
    assert result["objective"] == {
        **adapter.EXPECTED_OBJECTIVE,
        "value": pytest.approx(0.05),
    }
    assert result["metadata"]["applied_params"] == {"restitution": 0.7}
    assert result["metadata"]["recording"] == {
        "media_type": "model/vnd.usd",
        "fps": 30.0,
        "frame_count": 2,
    }
    trajectory = json.loads(
        Path(result["artifacts"]["trajectory"]).read_text(encoding="utf-8")
    )
    assert trajectory == {
        "dt_s": pytest.approx(1.0 / 30.0),
        "z_m": [1.0, 0.8],
        "contacted": True,
        "bounce_height_m": 0.6,
    }
    recording_path = Path(result["artifacts"]["recording_usd"])
    stage = Usd.Stage.Open(str(recording_path))
    assert stage is not None
    assert stage.GetTimeCodesPerSecond() == 30.0
    assert stage.GetStartTimeCode() == 0.0
    assert stage.GetEndTimeCode() == 1.0
    translate = stage.GetPrimAtPath("/World/Cube").GetAttribute("xformOp:translate")
    assert translate.GetTimeSamples() == [0.0, 1.0]


def test_adapter_builds_evidence_metadata_and_failed_result(tmp_path: Path) -> None:
    adapter = _load_adapter()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    manifest_path = artifacts_dir / "qualification_frames.json"
    manifest_path.write_text("{}", encoding="utf-8")
    request = _request(artifacts_dir)
    request.pop("recording")
    request["evidence"] = {
        "artifact_name": "frames",
        "renderer": "isaac_sim_kit_rtx",
        "width": 960,
        "height": 720,
        "fps": 30.0,
    }

    result = adapter._build_result(
        request,
        artifacts_dir=artifacts_dir,
        restitution=0.5,
        trajectory=[1.0, 0.9],
        contacted=False,
        bounce_height=0.0,
        captured_frames=24,
        evidence_manifest_path=manifest_path,
        recorded_poses=[],
    )

    assert result["success"] is False
    assert "objective" not in result
    assert result["artifacts"]["frames"] == str(manifest_path)
    assert result["metadata"]["evidence"] == {
        "renderer": "isaac_sim_kit_rtx",
        "width": 960,
        "height": 720,
        "fps": 30.0,
        "frame_count": 24,
    }

    with pytest.raises(RuntimeError, match="did not produce PNG frames"):
        adapter._build_result(
            request,
            artifacts_dir=artifacts_dir,
            restitution=0.5,
            trajectory=[],
            contacted=False,
            bounce_height=0.0,
            captured_frames=0,
            evidence_manifest_path=None,
            recorded_poses=[],
        )


def test_adapter_rejects_recording_without_pose_samples(tmp_path: Path) -> None:
    adapter = _load_adapter()

    with pytest.raises(RuntimeError, match="at least two pose samples"):
        adapter._write_recording(
            _request(tmp_path)["recording"],
            tmp_path,
            [],
        )
