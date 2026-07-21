# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge coverage for physics refine helpers.

These tests keep uncommon branches covered without adding more expensive
end-to-end refine loops.
"""

from __future__ import annotations

import builtins
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

import physics_agent.tasks.iterative_physics_refinement as iterative_mod
from physics_agent.tasks.iterative_physics_refinement import (
    IterationRecord,
    IterativePhysicsRefinementTask,
    _compact_refine_summary_history,
    _copy_iteration_to_final,
    _discover_camera_paths,
    _extract_metric_value,
    _finite_or_none,
    _materialize_winning_recording,
    _resolve_winning_recording_source,
    _run_with_llm_timeout,
    _scenario_bounds_from_yaml,
    _scenario_to_yaml_text,
)
from physics_agent.tasks.judge_tune import JudgeResult
from physics_agent.tasks.scenario_refine import (
    RefineError,
    RefineResult,
    _coerce_jsonable_number,
    _extract_json_object,
    _scenario_to_dict,
    _summarise_history,
    _trim,
    run_scenario_refine,
)
from physics_agent.tuning.errors import NewtonUnavailableError
from physics_agent.tuning.types import Scenario, TrialRecord, TunableParam, TuneOutput


def _scenario(*, extra: dict[str, Any] | None = None) -> Scenario:
    return Scenario(
        name="drop_settle",
        params=(
            TunableParam(name="restitution", min_value=0.1, max_value=0.9),
            TunableParam(name="mass_scale", min_value=0.5, max_value=2.0),
        ),
        target={"drop_height_m": 0.5, "duration_s": 2.0},
        metric="settle_distance",
        extra=extra or {},
    )


def _trial(
    *,
    idx: int = 0,
    score: float = 1.0,
    failed: bool = False,
    backend_metrics: dict[str, Any] | None = None,
) -> TrialRecord:
    return TrialRecord(
        trial_index=idx,
        params={"restitution": 0.5},
        score=score,
        backend_metrics=backend_metrics or {},
        duration_seconds=0.0,
        failed=failed,
    )


def _tune_output(output_dir: Path) -> TuneOutput:
    history = [
        _trial(
            idx=0,
            score=0.1,
            backend_metrics={"settle_distance": 0.1},
        )
    ]
    return TuneOutput(
        success=True,
        output_dir=output_dir,
        best_params={"restitution": 0.5},
        best_score=0.1,
        n_trials=1,
        optimizer_used="random",
        engine_used="fake",
        history=history,
        artifacts={},
        cancelled=False,
        needs_refinement=False,
    )


def _record(tmp_path: Path, *, score: float = 0.1) -> IterationRecord:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "name": "drop_settle",
                "metric": "settle_distance",
                "target": {},
                "parameters": [
                    {"name": "restitution", "min": 0.1, "max": 0.9},
                ],
            }
        ),
        encoding="utf-8",
    )
    return IterationRecord(
        iteration=1,
        iteration_dir=tmp_path / "iter_1",
        scenario_yaml_path=scenario_path,
        tune_output_dir=tmp_path / "iter_1",
        best_params={"restitution": 0.5},
        best_score=score,
        n_trials=1,
        judge_decision="continue",
        judge_score=score,
        judge_reasoning="needs work",
        judge_llm_unavailable=False,
        refine_llm_unavailable=False,
        refine_reasoning="",
        metric_name="settle_distance",
        metric_value=score,
    )


def _judge() -> JudgeResult:
    return JudgeResult(
        decision="continue",
        score=0.1,
        programmatic_score=0.1,
        llm_score=0.1,
        reasoning="continue",
        iterations=1,
    )


class _Chat:
    pass


class _SequenceCancel:
    def __init__(self, true_on_call: int) -> None:
        self.true_on_call = true_on_call
        self.calls = 0

    def is_set(self) -> bool:
        self.calls += 1
        return self.calls >= self.true_on_call


def test_iterative_yaml_and_metric_edge_helpers(tmp_path: Path) -> None:
    payload = yaml.safe_load(_scenario_to_yaml_text(_scenario(extra={"judge": {}})))
    assert payload["judge"] == {}

    assert _extract_metric_value([], "settle_distance") is None
    assert _extract_metric_value([_trial(failed=True)], "settle_distance") is None
    assert (
        _extract_metric_value(
            [_trial(backend_metrics={"max_bounce_height": "high"})],
            "max_bounce_height",
        )
        is None
    )
    assert (
        _extract_metric_value(
            [_trial(backend_metrics={"settle_distance": "near"})],
            "settle_distance",
        )
        is None
    )
    assert _extract_metric_value([_trial(score="near")], "settle_distance") is None  # type: ignore[arg-type]

    assert _finite_or_none("not-a-number") is None
    assert _finite_or_none(object()) is None
    assert _finite_or_none(math.inf) is None

    source = tmp_path / "iter"
    source.mkdir()
    (source / "scenario.yaml").write_text("name: drop_settle\n", encoding="utf-8")
    final = tmp_path / "final"
    final.mkdir()
    (final / "stale.txt").write_text("old", encoding="utf-8")
    _copy_iteration_to_final(source, final)
    assert not (final / "stale.txt").exists()
    assert (final / "scenario.yaml").exists()


def test_iterative_constructor_capability_and_default_runner_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        iterative_mod,
        "get_backend",
        lambda _engine: (_ for _ in ()).throw(NewtonUnavailableError("missing")),
    )
    monkeypatch.setattr(
        iterative_mod,
        "capabilities_for_backend",
        lambda _engine: [SimpleNamespace(param_name="restitution")],
    )
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=_scenario(),
        physics_usd=tmp_path / "physics.usda",
        output_dir=tmp_path / "out",
        engine="newton",
        run_tune_callable=lambda _params: _tune_output(tmp_path / "out" / "iter_1"),
    )
    assert task._refine_supported_param_keys == ("restitution",)

    monkeypatch.setattr(
        iterative_mod,
        "get_backend",
        lambda _engine: (_ for _ in ()).throw(RuntimeError("capability boom")),
    )
    with pytest.raises(RuntimeError, match="capability boom"):
        IterativePhysicsRefinementTask(
            user_prompt="goal",
            initial_scenario=_scenario(),
            physics_usd=tmp_path / "physics.usda",
            output_dir=tmp_path / "out2",
            engine="fake",
            run_tune_callable=lambda _params: _tune_output(
                tmp_path / "out2" / "iter_1"
            ),
        )

    monkeypatch.setattr(
        iterative_mod,
        "get_backend",
        lambda _engine: SimpleNamespace(
            tuning_capabilities=lambda: [SimpleNamespace(param_name="restitution")]
        ),
    )
    default_task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=_scenario(),
        physics_usd=tmp_path / "physics.usda",
        output_dir=tmp_path / "out3",
        engine="fake",
    )
    assert callable(default_task._run_tune)


def test_iterative_run_cancellation_checkpoints_and_force_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        iterative_mod,
        "run_tune_judge",
        lambda *_args, **_kwargs: JudgeResult(
            decision="approve",
            score=1.0,
            programmatic_score=1.0,
            llm_score=1.0,
            reasoning="ok",
            iterations=1,
        ),
    )
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=_scenario(),
        physics_usd=tmp_path / "physics.usda",
        output_dir=tmp_path / "pre_cancel",
        engine="fake",
        cancel_event=_SequenceCancel(true_on_call=1),
        run_tune_callable=lambda _params: _tune_output(tmp_path / "pre_cancel"),
    )
    assert task.run({}).termination_reason == "cancelled"

    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=_scenario(),
        physics_usd=tmp_path / "physics.usda",
        output_dir=tmp_path / "post_vlm_cancel",
        engine="fake",
        cancel_event=_SequenceCancel(true_on_call=2),
        run_tune_callable=lambda _params: _tune_output(tmp_path / "post_vlm_cancel"),
    )
    assert task.run({}).termination_reason == "cancelled"

    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=_scenario(),
        physics_usd=tmp_path / "physics.usda",
        output_dir=tmp_path / "loop_cancel",
        engine="fake",
        cancel_event=_SequenceCancel(true_on_call=3),
        run_tune_callable=lambda _params: _tune_output(tmp_path / "loop_cancel"),
    )
    assert task.run({}).termination_reason == "cancelled"

    out_dir = tmp_path / "force_record"
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=_scenario(),
        physics_usd=tmp_path / "physics.usda",
        output_dir=out_dir,
        engine="fake",
        force_record_video="off",
        render_winning_trial=False,
        vlm_model=object(),
        run_tune_callable=lambda params: _tune_output(Path(params.output_dir)),
    )
    assert task.run({}).termination_reason == "approved"
    persisted = yaml.safe_load((out_dir / "iter_1" / "scenario.yaml").read_text())
    assert persisted["target"]["record_video"] == "off"


@pytest.mark.parametrize(
    ("true_on_call", "out_name"),
    [
        (4, "after_tune"),
        (5, "before_judge"),
        (6, "after_judge"),
        (7, "before_refine"),
    ],
)
def test_iterative_run_later_cancellation_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    true_on_call: int,
    out_name: str,
) -> None:
    monkeypatch.setattr(
        IterativePhysicsRefinementTask,
        "_render_best_trial_into_iter_dir",
        lambda *_args, **_kwargs: ([], None),
    )
    monkeypatch.setattr(
        iterative_mod,
        "run_tune_judge",
        lambda *_args, **_kwargs: JudgeResult(
            decision="continue",
            score=0.1,
            programmatic_score=0.1,
            llm_score=0.1,
            reasoning="continue",
            iterations=1,
        ),
    )
    monkeypatch.setattr(
        iterative_mod,
        "run_scenario_refine",
        lambda **kwargs: RefineResult(
            refined_yaml="name: drop_settle\n",
            scenario=kwargs["current_scenario"],
            llm_unavailable=False,
            reasoning="same",
        ),
    )
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=_scenario(),
        physics_usd=tmp_path / "physics.usda",
        output_dir=tmp_path / out_name,
        engine="fake",
        max_iterations=2,
        render_winning_trial=true_on_call >= 5,
        vlm_model=object(),
        cancel_event=_SequenceCancel(true_on_call=true_on_call),
        run_tune_callable=lambda params: _tune_output(Path(params.output_dir)),
    )
    result = task.run({})
    assert result.termination_reason == "cancelled"
    assert result.iterations[0].cancelled is True


def test_iterative_generated_frames_without_reference_and_fallback_final_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"frame")
    judge_kwargs: list[dict[str, Any]] = []

    monkeypatch.setattr(
        IterativePhysicsRefinementTask,
        "_render_best_trial_into_iter_dir",
        lambda *_args, **_kwargs: ([frame], None),
    )

    def judge_continue(*_args: Any, **kwargs: Any) -> JudgeResult:
        judge_kwargs.append(kwargs)
        return JudgeResult(
            decision="continue",
            score=0.1,
            programmatic_score=0.1,
            llm_score=0.1,
            reasoning="continue",
            iterations=1,
        )

    monkeypatch.setattr(iterative_mod, "run_tune_judge", judge_continue)
    monkeypatch.setattr(
        iterative_mod,
        "run_scenario_refine",
        lambda **kwargs: RefineResult(
            refined_yaml="name: drop_settle\n",
            scenario=kwargs["current_scenario"],
            llm_unavailable=False,
            reasoning="same",
        ),
    )
    monkeypatch.setattr(iterative_mod, "range", lambda *_args: [1], raising=False)

    out_dir = tmp_path / "generated_only"
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=_scenario(),
        physics_usd=tmp_path / "physics.usda",
        output_dir=out_dir,
        engine="fake",
        max_iterations=2,
        render_winning_trial=True,
        vlm_model=object(),
        run_tune_callable=lambda params: _tune_output(Path(params.output_dir)),
    )
    result = task.run({})
    assert result.termination_reason == "max_iterations"
    assert result.final_dir == out_dir / "final"
    assert judge_kwargs[0]["visual_evidence"].generated_image_paths == (frame,)


def test_iterative_render_best_trial_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=_scenario(),
        physics_usd=tmp_path / "physics.usda",
        output_dir=tmp_path / "out",
        engine="fake",
        run_tune_callable=lambda _params: _tune_output(tmp_path / "out" / "iter_1"),
    )
    listener = SimpleNamespace(
        warning=lambda *_args, **_kwargs: None, info=lambda *_args, **_kwargs: None
    )
    assert task._render_best_trial_into_iter_dir(
        iter_dir=tmp_path / "iter_failed",
        history=[_trial(failed=True)],
        scenario=_scenario(),
        listener=listener,
    ) == ([], "every trial failed; no winning trial to render")

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "world_understanding.functions.graphics":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert task._render_best_trial_into_iter_dir(
        iter_dir=tmp_path / "iter_import",
        history=[_trial(backend_metrics={"recording_usd": tmp_path / "rec.usd"})],
        scenario=_scenario(),
        listener=listener,
    ) == ([], "render_time_sampled_usd unavailable")

    monkeypatch.setattr(builtins, "__import__", real_import)
    graphics = ModuleType("world_understanding.functions.graphics")
    graphics.render_time_sampled_usd = lambda *_args, **_kwargs: []
    monkeypatch.setitem(sys.modules, "world_understanding.functions.graphics", graphics)
    assert task._render_best_trial_into_iter_dir(
        iter_dir=tmp_path / "iter_empty",
        history=[_trial(backend_metrics={"recording_usda": tmp_path / "rec.usda"})],
        scenario=_scenario(),
        listener=listener,
    ) == ([], "renderer produced no frames")

    invalid_iter_dir = tmp_path / "iter_invalid"
    invalid_scenario = replace(
        _scenario(),
        target={"duration_s": 2.0, "video_renderer": False},
    )
    with pytest.raises(ValueError, match="Unknown rendering backend"):
        task._render_best_trial_into_iter_dir(
            iter_dir=invalid_iter_dir,
            history=[_trial(backend_metrics={"recording_usd": tmp_path / "rec.usd"})],
            scenario=invalid_scenario,
            listener=listener,
        )
    assert not (invalid_iter_dir / "render").exists()


def test_winning_recording_source_is_strict_and_resolves_from_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worse_recording = tmp_path / "worse.usd"
    worse_recording.write_bytes(b"worse")
    strict_history = [
        _trial(idx=0, score=0.1),
        _trial(
            idx=1,
            score=0.2,
            backend_metrics={"recording_usd": worse_recording},
        ),
    ]

    assert _resolve_winning_recording_source(strict_history) == (
        None,
        "winning trial did not persist recording_usd",
    )
    assert _materialize_winning_recording(
        strict_history,
        tmp_path / "must-not-exist.usd",
    ) == (None, "winning trial did not persist recording_usd")
    assert _resolve_winning_recording_source([_trial(failed=True)]) == (
        None,
        "every trial failed; no winning recording",
    )

    invalid_source, invalid_error = _resolve_winning_recording_source(
        [_trial(backend_metrics={"recording_usd": object()})]
    )
    assert invalid_source is None
    assert invalid_error is not None
    assert "invalid recording path" in invalid_error

    missing_source, missing_error = _resolve_winning_recording_source(
        [_trial(backend_metrics={"recording_usd": tmp_path / "missing.usd"})]
    )
    assert missing_source is None
    assert missing_error is not None
    assert "does not exist" in missing_error

    monkeypatch.chdir(tmp_path)
    relative_recording = Path("sessions/sid/input/.tune_scenes/recording.usd")
    relative_recording.parent.mkdir(parents=True)
    relative_recording.write_bytes(b"recording")
    source, error = _resolve_winning_recording_source(
        [_trial(backend_metrics={"recording_usd": str(relative_recording)})]
    )
    assert error is None
    assert source == (tmp_path / relative_recording).resolve()


def test_materialize_winning_recording_success_and_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usd"
    source.write_bytes(b"source")
    history = [_trial(score=0.1, backend_metrics={"recording_usd": source})]
    opened: list[Path] = []

    class FakeLayer:
        export_ok = True

        def Export(self, path: str) -> bool:  # noqa: N802
            if self.export_ok:
                Path(path).write_bytes(b"flattened")
            return self.export_ok

    class FakeStageInstance:
        def Flatten(self) -> FakeLayer:  # noqa: N802
            return FakeLayer()

    class FakeStage:
        open_ok = True

        @classmethod
        def Open(cls, path: str):  # noqa: N802
            opened.append(Path(path))
            return FakeStageInstance() if cls.open_ok else None

    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(Usd=SimpleNamespace(Stage=FakeStage)),
    )

    destination = tmp_path / "final" / "recording.usd"
    recording, error = _materialize_winning_recording(history, destination)
    assert error is None
    assert recording == destination
    assert destination.read_bytes() == b"flattened"
    assert opened == [source]

    FakeStage.open_ok = False
    recording, error = _materialize_winning_recording(
        history,
        tmp_path / "open-failure.usd",
    )
    assert recording is None
    assert error is not None
    assert "Could not open winning recording" in error

    FakeStage.open_ok = True
    FakeLayer.export_ok = False
    export_destination = tmp_path / "export-failure.usd"
    recording, error = _materialize_winning_recording(history, export_destination)
    assert recording is None
    assert error is not None
    assert "Could not export flattened recording" in error
    assert not (tmp_path / ".export-failure.tmp.usd").exists()


def test_result_summary_promotes_final_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iter_dir = tmp_path / "iter_1"
    iter_dir.mkdir()
    recording = iter_dir / "recording.usd"
    recording.write_bytes(b"recording")
    record = _record(tmp_path)
    record.recording_usd = recording
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=_scenario(),
        physics_usd=tmp_path / "physics.usd",
        output_dir=tmp_path / "out",
        engine="fake",
        run_tune_callable=lambda _params: _tune_output(iter_dir),
    )

    result = task._write_result_summary(
        records=[record],
        termination_reason="approved",
        final_iter_dir=iter_dir,
    )
    assert result.final_recording_error is None
    assert result.final_recording_usd == tmp_path / "out" / "final" / "recording.usd"
    assert result.final_recording_usd.read_bytes() == b"recording"

    def omit_recording(_iter_dir: Path, final_dir: Path) -> None:
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "recording.usd").unlink(missing_ok=True)

    monkeypatch.setattr(iterative_mod, "_copy_iteration_to_final", omit_recording)
    result = task._write_result_summary(
        records=[record],
        termination_reason="approved",
        final_iter_dir=iter_dir,
    )
    assert result.final_recording_usd is None
    assert result.final_recording_error is not None
    assert "was not copied" in result.final_recording_error


def test_scenario_bounds_cache_and_malformed_inputs(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    assert _scenario_bounds_from_yaml(missing, {}) == {}

    non_list = tmp_path / "non_list.yaml"
    non_list.write_text("parameters: {}\n", encoding="utf-8")
    cache: dict[Path, dict[str, list[float | None]]] = {}
    assert _scenario_bounds_from_yaml(non_list, cache) == {}

    mixed = tmp_path / "mixed.yaml"
    mixed.write_text(
        yaml.safe_dump(
            {
                "parameters": [
                    "skip-me",
                    {"name": "", "min": 1, "max": 2},
                    {"name": "restitution", "min": "low", "max": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )
    cache = {}
    assert _scenario_bounds_from_yaml(mixed, cache) == {"restitution": [None, 0.9]}
    mixed.write_text("parameters: []\n", encoding="utf-8")
    assert _scenario_bounds_from_yaml(mixed, cache) == {"restitution": [None, 0.9]}


def test_compact_summary_empty_and_timeout_direct_error(tmp_path: Path) -> None:
    assert _compact_refine_summary_history([], history_window=0) == []
    assert _compact_refine_summary_history([_record(tmp_path)], history_window=1)[0][
        "best_so_far"
    ]

    assert (
        _run_with_llm_timeout(
            lambda value: value + 1,
            2,
            timeout_seconds=0,
            op_label="unit",
        )
        == 3
    )

    def raises() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _run_with_llm_timeout(raises, timeout_seconds=1, op_label="unit")


def test_discover_camera_paths_stage_edges(
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
            raise RuntimeError("bad stage")

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
                Traverse=lambda: [
                    Prim("/World", False),
                    Prim("/Cameras/main", True),
                ]
            )

    fake_pxr.Usd = SimpleNamespace(Stage=Stage)
    assert _discover_camera_paths(tmp_path / "with_camera.usda") == ["/Cameras/main"]


def test_scenario_refine_helper_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _trim("abcdef", 5) == "ab..."
    assert _coerce_jsonable_number("raw") == "raw"
    assert _coerce_jsonable_number(float("nan")) == "NaN"
    assert _coerce_jsonable_number(float("inf")) == "Infinity"
    assert _coerce_jsonable_number(float("-inf")) == "-Infinity"
    assert _summarise_history([{"score": None}, {"score": 0.1}])[0]["score"] == 0.1

    assert _extract_json_object('```json\n{"ok": true}\n```') == {"ok": True}
    wrapped = 'prefix {"text": "brace } and quote \\" ok", "nested": {"x": 1}} suffix'
    assert _extract_json_object(wrapped)["nested"] == {"x": 1}
    with pytest.raises(RefineError, match="No JSON object"):
        _extract_json_object("no json here")
    with pytest.raises(RefineError, match="Unbalanced JSON"):
        _extract_json_object('prefix {"x": 1')
    with pytest.raises(RefineError, match="Could not parse JSON"):
        _extract_json_object('prefix {"x": } suffix')

    monkeypatch.setattr(
        "physics_agent.tasks.scenario_refine.generate_chat_response",
        lambda *_args, **_kwargs: {"response": ""},
    )
    empty = run_scenario_refine(
        current_scenario=_scenario(),
        judge_result=_judge(),
        user_goal_text="goal",
        chat_model=_Chat(),
    )
    assert empty.llm_unavailable is True
    assert "empty response" in empty.reasoning

    monkeypatch.setattr(
        "physics_agent.tasks.scenario_refine.generate_chat_response",
        lambda *_args, **_kwargs: {
            "response": json.dumps(
                {
                    "scenario": _scenario_to_dict(_scenario()),
                    "reasoning": {"structured": True},
                }
            )
        },
    )
    refined = run_scenario_refine(
        current_scenario=_scenario(),
        judge_result=_judge(),
        user_goal_text="goal",
        chat_model=_Chat(),
    )
    assert refined.llm_unavailable is False
    assert refined.reasoning == "{'structured': True}"
