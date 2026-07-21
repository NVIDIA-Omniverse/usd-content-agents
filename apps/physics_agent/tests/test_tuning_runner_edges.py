# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Targeted edge coverage for ``physics_agent.tuning.runner``."""

from __future__ import annotations

import builtins
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import physics_agent.tuning.runner as runner_mod
from physics_agent.tuning.errors import TuningCancelledError, TuningError
from physics_agent.tuning.runner import (
    _anchor_relative_paths_in_scenario_dict,
    _backend_param_keys_for_interpreter,
    _discover_camera_paths,
    _do_run_tune,
    _do_run_tune_inner,
    _evaluate_one,
    _explicit_scenario_param_names,
    _load_scenario_override_dict,
    _prepare_visual_evidence_for_judge,
    _render_best_trial_for_visual_judge,
    _run_with_llm_timeout,
    _validate_engine_supports_scenario,
    _validate_inputs,
    _visual_evidence_fail_closed_error,
)
from physics_agent.tuning.types import Scenario, TrialRecord, TunableParam, TuneInput
from physics_agent.tuning.visual_evidence import JudgeVisualEvidence


def _scenario() -> Scenario:
    return Scenario(
        name="drop_settle",
        params=(TunableParam(name="restitution", min_value=0.0, max_value=1.0),),
        target={"duration_s": 1.0},
        metric="settle_distance",
    )


def _freeform_scenario() -> Scenario:
    return Scenario(
        name="freeform",
        params=(TunableParam(name="restitution", min_value=0.0, max_value=1.0),),
        target={"description": "realistic generated rollout", "duration_s": 1.0},
        metric="judge_score",
    )


def _trial(
    *,
    failed: bool = False,
    score: float = 1.0,
    backend_metrics: dict[str, Any] | None = None,
) -> TrialRecord:
    return TrialRecord(
        trial_index=0,
        params={"restitution": 0.5},
        score=score,
        backend_metrics=backend_metrics or {},
        duration_seconds=0.0,
        failed=failed,
    )


def _params(tmp_path: Path, **overrides: Any) -> TuneInput:
    physics = tmp_path / "physics.usda"
    physics.write_text("#usda 1.0\n", encoding="utf-8")
    values: dict[str, Any] = {
        "scenario": {
            "name": "drop_settle",
            "parameters": [{"name": "restitution", "min": 0.0, "max": 1.0}],
        },
        "physics_usd": physics,
        "output_dir": tmp_path / "out",
        "engine": "fake",
        "optimizer": "random",
        "max_trials": 1,
        "enable_judge": False,
    }
    values.update(overrides)
    return TuneInput(**values)


def test_timeout_helper_direct_cancel_and_no_deadline_paths() -> None:
    assert (
        _run_with_llm_timeout(
            lambda value: value + 1,
            2,
            timeout_seconds=0,
            op_label="direct",
        )
        == 3
    )

    cancel = threading.Event()
    with pytest.raises(TuningCancelledError, match="before setup call"):
        cancel.set()
        _run_with_llm_timeout(
            lambda: None,
            timeout_seconds=10,
            cancel_check=cancel.is_set,
            op_label="setup",
        )

    cancel.clear()

    def slow() -> str:
        time.sleep(0.2)
        return "late"

    checks = {"n": 0}

    def cancel_after_first_poll() -> bool:
        checks["n"] += 1
        if checks["n"] == 1:
            return False
        cancel.set()
        return cancel.is_set()

    with pytest.raises(TuningCancelledError, match="while waiting"):
        _run_with_llm_timeout(
            slow,
            timeout_seconds=0,
            cancel_check=cancel_after_first_poll,
            cancel_poll_seconds=0.01,
            op_label="judge",
        )


def test_scenario_override_and_backend_helper_edges(tmp_path: Path) -> None:
    _validate_engine_supports_scenario("future-engine", "drop_settle")

    data = {
        "asset_usd": "asset.usda",
        "empty_path": "   ",
        "remote_usd": "s3://bucket/object.usda",
        "nested": [{"cache_dir": "cache"}],
        "label": "not/a/path",
    }
    _anchor_relative_paths_in_scenario_dict(data, tmp_path)
    assert data["asset_usd"] == str((tmp_path / "asset.usda").resolve())
    assert data["empty_path"] == "   "
    assert data["remote_usd"] == "s3://bucket/object.usda"
    assert data["nested"][0]["cache_dir"] == str((tmp_path / "cache").resolve())
    assert data["label"] == "not/a/path"

    non_mapping = tmp_path / "scenario.yaml"
    non_mapping.write_text("- not\n- mapping\n", encoding="utf-8")
    with pytest.raises(TuningError, match="did not parse to a mapping"):
        _load_scenario_override_dict(non_mapping)

    assert _explicit_scenario_param_names({"parameters": {}}) == set()
    assert _explicit_scenario_param_names({"parameters": ["skip", {"name": "x"}]}) == {
        "x"
    }
    assert _backend_param_keys_for_interpreter(None) is None
    assert (
        _backend_param_keys_for_interpreter(
            SimpleNamespace(tuning_capabilities=lambda: [SimpleNamespace(param_name=1)])
        )
        is None
    )


def test_validate_inputs_rejects_non_path_physics_usd(tmp_path: Path) -> None:
    params = _params(tmp_path, physics_usd=object())
    with pytest.raises(TypeError, match="physics_usd"):
        _validate_inputs(params)


@pytest.mark.parametrize(
    ("backend_result", "error_fragment"),
    [
        (["bad"], "expected dict"),
        ({}, "missing required key"),
        ({"score": float("nan")}, "non-finite"),
    ],
)
def test_evaluate_one_malformed_backend_results(
    tmp_path: Path,
    backend_result: Any,
    error_fragment: str,
) -> None:
    backend = SimpleNamespace(evaluate=lambda **_kwargs: backend_result)
    trial = _evaluate_one(
        backend,
        _scenario(),
        {"restitution": 0.5},
        tmp_path / "physics.usda",
        seed=1,
        trial_index=3,
    )
    assert trial.failed is True
    assert error_fragment in (trial.error or "")


def test_runner_camera_discovery_and_render_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Camera:
        pass

    class Prim:
        def __init__(self, path: str, is_camera: bool) -> None:
            self._path = path
            self._is_camera = is_camera

        def IsA(self, camera_type: object) -> bool:
            return camera_type is Camera and self._is_camera

        def GetPath(self) -> str:
            return self._path

    class RaisingStage:
        @staticmethod
        def Open(_path: str) -> object:
            raise RuntimeError("bad")

    fake_pxr = SimpleNamespace(
        Usd=SimpleNamespace(Stage=RaisingStage),
        UsdGeom=SimpleNamespace(Camera=Camera),
    )
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    assert _discover_camera_paths(tmp_path / "bad.usda") is None

    class MissingStage:
        @staticmethod
        def Open(_path: str) -> None:
            return None

    fake_pxr.Usd = SimpleNamespace(Stage=MissingStage)
    assert _discover_camera_paths(tmp_path / "missing.usda") is None

    class Stage:
        @staticmethod
        def Open(_path: str) -> object:
            return SimpleNamespace(
                Traverse=lambda: [Prim("/World", False), Prim("/Camera", True)]
            )

    fake_pxr.Usd = SimpleNamespace(Stage=Stage)
    assert _discover_camera_paths(tmp_path / "with_camera.usda") == ["/Camera"]

    assert _render_best_trial_for_visual_judge(
        output_dir=tmp_path,
        history=[_trial(failed=True)],
        scenario=_scenario(),
    ) == ([], "every trial failed; no winning trial to render")
    assert _render_best_trial_for_visual_judge(
        output_dir=tmp_path,
        history=[_trial()],
        scenario=_scenario(),
    ) == ([], "winning trial did not persist recording_usd")

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "world_understanding.functions.graphics":
            raise ImportError("missing renderer")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _render_best_trial_for_visual_judge(
        output_dir=tmp_path,
        history=[_trial(backend_metrics={"recording_usd": tmp_path / "rec.usd"})],
        scenario=_scenario(),
    ) == ([], "render_time_sampled_usd unavailable")

    monkeypatch.setattr(builtins, "__import__", real_import)
    graphics = ModuleType("world_understanding.functions.graphics")
    graphics.render_time_sampled_usd = lambda *_args, **_kwargs: []
    monkeypatch.setitem(sys.modules, "world_understanding.functions.graphics", graphics)
    assert _render_best_trial_for_visual_judge(
        output_dir=tmp_path,
        history=[_trial(backend_metrics={"recording_usd": tmp_path / "rec.usd"})],
        scenario=_scenario(),
    ) == ([], "renderer produced no frames")

    frame = tmp_path / "frame.png"
    graphics.render_time_sampled_usd = lambda *_args, **_kwargs: [frame]
    assert _render_best_trial_for_visual_judge(
        output_dir=tmp_path,
        history=[_trial(backend_metrics={"recording_usda": tmp_path / "rec.usda"})],
        scenario=_scenario(),
    ) == ([frame], None)

    invalid_scenario = replace(
        _scenario(),
        target={"duration_s": 1.0, "video_renderer": ""},
    )
    with pytest.raises(ValueError, match="Unknown rendering backend"):
        _render_best_trial_for_visual_judge(
            output_dir=tmp_path / "invalid",
            history=[_trial(backend_metrics={"recording_usd": tmp_path / "rec.usd"})],
            scenario=invalid_scenario,
        )
    assert not (tmp_path / "invalid" / "judge_render").exists()


def test_prepare_visual_evidence_success_and_fail_closed_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = _params(
        tmp_path,
        reference_images=[tmp_path / "reference.png"],
        judge_reference_frames=5,
        judge_generated_frames=7,
        output_dir=tmp_path / "out",
    )
    reference = JudgeVisualEvidence(
        reference_image_caption_pairs=(("Reference:", tmp_path / "reference.png"),)
    )
    monkeypatch.setattr(
        runner_mod,
        "prepare_reference_media",
        lambda **_kwargs: reference,
    )
    monkeypatch.setattr(
        runner_mod,
        "_render_best_trial_for_visual_judge",
        lambda **_kwargs: ([tmp_path / "generated.png"], None),
    )

    def fake_write_comparison_contact_sheet(
        _evidence: JudgeVisualEvidence,
        path: Path,
        *,
        max_reference_images: int,
        max_generated_images: int,
    ) -> tuple[Path, None]:
        assert max_reference_images == 5
        assert max_generated_images == 7
        return path, None

    monkeypatch.setattr(
        runner_mod,
        "write_comparison_contact_sheet",
        fake_write_comparison_contact_sheet,
    )

    evidence = _prepare_visual_evidence_for_judge(
        params=params,
        output_dir=tmp_path / "out",
        history=[_trial()],
        scenario=_scenario(),
    )
    assert evidence is not None
    assert evidence.generated_image_paths == (tmp_path / "generated.png",)
    assert evidence.comparison_image_path == (
        tmp_path / "out" / runner_mod.ARTIFACT_VISUAL_COMPARISON
    )

    assert (
        _prepare_visual_evidence_for_judge(
            params=_params(tmp_path, reference_images=[]),
            output_dir=tmp_path / "out2",
            history=[],
            scenario=_scenario(),
        )
        is None
    )
    assert (
        _visual_evidence_fail_closed_error(None)
        == "visual evidence preparation returned no evidence"
    )


def test_prepare_visual_evidence_generated_only_for_freeform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = tmp_path / "generated.png"
    frame.write_bytes(b"fake generated image bytes")
    monkeypatch.setattr(
        runner_mod,
        "_render_best_trial_for_visual_judge",
        lambda **_kwargs: ([frame], None),
    )

    evidence = _prepare_visual_evidence_for_judge(
        params=_params(tmp_path, reference_images=[]),
        output_dir=tmp_path / "out",
        history=[_trial()],
        scenario=_freeform_scenario(),
        include_generated_without_reference=True,
    )

    assert evidence is not None
    assert evidence.reference_image_caption_pairs == ()
    assert evidence.generated_image_paths == (frame,)
    assert evidence.comparison_image_path is None


def test_do_run_tune_error_paths_without_backend_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(OSError):
        _do_run_tune(
            _params(
                tmp_path,
                scenario=tmp_path / "missing.yaml",
                engine="fake",
            )
        )

    def zero_trial_runner(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(runner_mod, "get_runner", lambda _name: zero_trial_runner)
    with pytest.raises(TuningError, match="zero trials"):
        _do_run_tune_inner(
            params=_params(tmp_path),
            output_dir=tmp_path / "zero",
            listener=None,
            scenario=_scenario(),
            cancel_check=lambda: False,
            physics_usd=tmp_path / "physics.usda",
            optimizer_used="random",
            backend=SimpleNamespace(),
        )

    def cancelling_runner(
        _scenario: Scenario,
        evaluate: Any,
        **_kwargs: Any,
    ) -> None:
        evaluate({"restitution": 0.5})

    monkeypatch.setattr(runner_mod, "get_runner", lambda _name: cancelling_runner)
    cancelled = _do_run_tune_inner(
        params=_params(tmp_path),
        output_dir=tmp_path / "cancelled",
        listener=None,
        scenario=_scenario(),
        cancel_check=lambda: True,
        physics_usd=tmp_path / "physics.usda",
        optimizer_used="random",
        backend=SimpleNamespace(),
    )
    assert cancelled.cancelled is True
    assert cancelled.n_trials == 0
