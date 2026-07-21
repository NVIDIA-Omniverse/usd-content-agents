# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``physics_agent.api.refine`` — the first-class refine API.

Round 15 (doyubkim blocker #3) added ``RefineInput`` / ``RefineOutput`` /
``run_refine`` / ``arun_refine`` as the public physics-agent refine
surface, mirroring material-agent's API shape. These tests exercise the
public contract:

* Input validation (kw-only, required fields, file-existence checks).
* Lazy export through ``physics_agent.api.__getattr__``.
* End-to-end ``run_refine`` with the orchestrator stubbed so we do not
  spin OvPhysX or hit any LLM provider — we mock the inner task class.
* The output dataclass mirrors the loop result and coerces non-finite
  ``best_score`` / ``judge_score`` / ``metric_value`` to ``None``.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml


def _scenario_yaml_dict() -> dict[str, Any]:
    return {
        "name": "drop_settle",
        "metric": "settle_distance",
        "target": {"drop_height_m": 0.5, "duration_s": 2.0, "gravity": -9.81},
        "parameters": [
            {"name": "restitution", "min": 0.4, "max": 0.95},
            {"name": "mass_scale", "min": 0.5, "max": 2.0},
        ],
    }


def _scenario_yaml_file(tmp_path: Path) -> Path:
    p = tmp_path / "scenario.yaml"
    p.write_text(yaml.safe_dump(_scenario_yaml_dict()), encoding="utf-8")
    return p


def _fake_usd(tmp_path: Path) -> Path:
    p = tmp_path / "physics.usda"
    p.write_text("#usda 1.0\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Lazy export contract
# ---------------------------------------------------------------------------


def test_lazy_import_does_not_load_orchestrator() -> None:
    """Importing ``physics_agent.api`` must NOT load
    ``physics_agent.api.refine`` (which would transitively pull in the
    tune runner). The lazy ``__getattr__`` mirrors the tune-side
    contract enforced by ``test_predict_runtime_import_does_not_pull_tuning``.

    This test deliberately runs in a fresh subprocess so deleting
    modules from ``sys.modules`` does not invalidate already-imported
    references in sibling test files. (An earlier in-process variant
    accidentally invalidated
    ``test_iterative_physics_refinement.py``'s module-level
    ``IterativePhysicsRefinementTask`` binding, breaking unrelated
    monkeypatch-based tests.)
    """
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "import physics_agent.api\n"
        "assert 'physics_agent.api.refine' not in sys.modules, "
        "'refine module leaked on physics_agent.api import'\n"
        "# The orchestrator module would transitively load the tuning runner.\n"
        "assert 'physics_agent.api' in sys.modules\n"
    )
    result = subprocess.run(
        [_sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess failed: stderr={result.stderr!r}, stdout={result.stdout!r}"
    )


def test_lazy_export_resolves_refine_symbols() -> None:
    """``from physics_agent.api import RefineInput, ...`` must work and
    pull in the refine module on demand."""
    from physics_agent.api import (
        RefineInput,
        RefineOutput,
        arun_refine,
        run_refine,
    )

    # Sanity-check that the symbols come from the refine module.
    assert RefineInput.__module__ == "physics_agent.api.refine"
    assert RefineOutput.__module__ == "physics_agent.api.refine"
    assert run_refine.__module__ == "physics_agent.api.refine"
    assert arun_refine.__module__ == "physics_agent.api.refine"


def test_refine_symbols_listed_in_all() -> None:
    """The refine surface is part of the documented public API."""
    import physics_agent.api as papi

    for name in ("RefineInput", "RefineOutput", "run_refine", "arun_refine"):
        assert name in papi.__all__, f"{name} missing from physics_agent.api.__all__"


# ---------------------------------------------------------------------------
# RefineInput validation
# ---------------------------------------------------------------------------


def test_refine_input_rejects_empty_user_prompt(tmp_path: Path) -> None:
    from physics_agent.api.refine import RefineInput

    with pytest.raises(ValueError, match="user_prompt"):
        RefineInput(
            scenario=_scenario_yaml_file(tmp_path),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="   ",
            output_dir=tmp_path / "out",
        )


def test_refine_input_rejects_missing_scenario_file(tmp_path: Path) -> None:
    from physics_agent.api.refine import RefineInput

    with pytest.raises(FileNotFoundError):
        RefineInput(
            scenario=tmp_path / "does_not_exist.yaml",
            physics_usd=_fake_usd(tmp_path),
            user_prompt="bouncy",
            output_dir=tmp_path / "out",
        )


def test_refine_input_rejects_missing_physics_usd(tmp_path: Path) -> None:
    from physics_agent.api.refine import RefineInput

    with pytest.raises(FileNotFoundError):
        RefineInput(
            scenario=_scenario_yaml_file(tmp_path),
            physics_usd=tmp_path / "missing.usda",
            user_prompt="bouncy",
            output_dir=tmp_path / "out",
        )


def test_refine_input_accepts_dict_scenario(tmp_path: Path) -> None:
    from physics_agent.api.refine import RefineInput

    params = RefineInput(
        scenario=_scenario_yaml_dict(),
        physics_usd=_fake_usd(tmp_path),
        user_prompt="bouncy",
        output_dir=tmp_path / "out",
        max_iterations=1,
    )
    assert isinstance(params.scenario, dict)
    assert params.scenario["name"] == "drop_settle"
    assert params.history_window == 20


def test_refine_input_rejects_zero_iterations(tmp_path: Path) -> None:
    from physics_agent.api.refine import RefineInput

    with pytest.raises(ValueError, match="max_iterations"):
        RefineInput(
            scenario=_scenario_yaml_dict(),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="bouncy",
            output_dir=tmp_path / "out",
            max_iterations=0,
        )


def test_refine_input_rejects_rest_incompatible_bounds(tmp_path: Path) -> None:
    from physics_agent.api.refine import RefineInput

    base = {
        "scenario": _scenario_yaml_dict(),
        "physics_usd": _fake_usd(tmp_path),
        "user_prompt": "bouncy",
        "output_dir": tmp_path / "out",
    }

    assert RefineInput(**base, max_iterations=12).max_iterations == 12

    with pytest.raises(ValueError, match="max_iterations"):
        RefineInput(**base, max_iterations=13)

    with pytest.raises(ValueError, match="max_trials"):
        RefineInput(**base, max_trials=1001)

    with pytest.raises(ValueError, match="score_threshold"):
        RefineInput(**base, score_threshold=1.1)

    with pytest.raises(ValueError, match="score_threshold"):
        RefineInput(**base, score_threshold=float("nan"))

    with pytest.raises(ValueError, match="llm_timeout_seconds"):
        RefineInput(**base, llm_timeout_seconds=float("inf"))

    with pytest.raises(ValueError, match="reference_video_frames"):
        RefineInput(**base, reference_video_frames=0)

    with pytest.raises(ValueError, match="judge_reference_frames"):
        RefineInput(**base, judge_reference_frames=65)

    with pytest.raises(ValueError, match="judge_generated_frames"):
        RefineInput(**base, judge_generated_frames=0)


def test_refine_input_rejects_negative_history_window(tmp_path: Path) -> None:
    from physics_agent.api.refine import RefineInput

    with pytest.raises(ValueError, match="history_window"):
        RefineInput(
            scenario=_scenario_yaml_dict(),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="bouncy",
            output_dir=tmp_path / "out",
            history_window=-1,
        )


def test_refine_input_rejects_invalid_judge_knobs(tmp_path: Path) -> None:
    from physics_agent.api.refine import RefineInput

    with pytest.raises(ValueError, match="judge_max_tokens"):
        RefineInput(
            scenario=_scenario_yaml_dict(),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="bouncy",
            output_dir=tmp_path / "out",
            judge_max_tokens=0,
        )

    with pytest.raises(ValueError, match="judge_temperature"):
        RefineInput(
            scenario=_scenario_yaml_dict(),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="bouncy",
            output_dir=tmp_path / "out",
            judge_temperature=-0.1,
        )


def test_refine_input_rejects_invalid_force_record_video(tmp_path: Path) -> None:
    from physics_agent.api.refine import RefineInput

    with pytest.raises(ValueError, match="force_record_video"):
        RefineInput(
            scenario=_scenario_yaml_dict(),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="bouncy",
            output_dir=tmp_path / "out",
            force_record_video="bogus",
        )


def test_refine_input_rejects_invalid_cancel_event(tmp_path: Path) -> None:
    from physics_agent.api.refine import RefineInput

    with pytest.raises(TypeError, match="cancel_event"):
        RefineInput(
            scenario=_scenario_yaml_dict(),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="bouncy",
            output_dir=tmp_path / "out",
            cancel_event=object(),
        )


def test_refine_input_validates_reference_media_paths_and_descriptions(
    tmp_path: Path,
) -> None:
    from physics_agent.api.refine import RefineInput

    reference = tmp_path / "reference.png"
    reference.write_bytes(b"fake image bytes")
    video = tmp_path / "reference.mp4"
    video.write_bytes(b"fake video bytes")

    params = RefineInput(
        scenario=_scenario_yaml_dict(),
        physics_usd=_fake_usd(tmp_path),
        user_prompt="match the target motion",
        output_dir=tmp_path / "out",
        reference_images=[reference],
        reference_videos=[video],
        reference_descriptions=["target pose"],
        reference_video_descriptions=["target motion"],
    )
    assert params.reference_images == [reference]
    assert params.reference_videos == [video]

    with pytest.raises(FileNotFoundError, match="reference image"):
        RefineInput(
            scenario=_scenario_yaml_dict(),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="match the target motion",
            output_dir=tmp_path / "out",
            reference_images=[tmp_path / "missing.png"],
        )

    reference_dir = tmp_path / "reference_dir"
    reference_dir.mkdir()
    with pytest.raises(ValueError, match="reference image must be a file"):
        RefineInput(
            scenario=_scenario_yaml_dict(),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="match the target motion",
            output_dir=tmp_path / "out",
            reference_images=[reference_dir],
        )

    with pytest.raises(FileNotFoundError, match="reference video"):
        RefineInput(
            scenario=_scenario_yaml_dict(),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="match the target motion",
            output_dir=tmp_path / "out",
            reference_videos=[tmp_path / "missing.mp4"],
        )

    with pytest.raises(ValueError, match="reference video must be a file"):
        RefineInput(
            scenario=_scenario_yaml_dict(),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="match the target motion",
            output_dir=tmp_path / "out",
            reference_videos=[reference_dir],
        )

    with pytest.raises(ValueError, match="reference_descriptions"):
        RefineInput(
            scenario=_scenario_yaml_dict(),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="match the target motion",
            output_dir=tmp_path / "out",
            reference_images=[reference],
            reference_descriptions=[],
        )

    with pytest.raises(ValueError, match="reference_descriptions"):
        RefineInput(
            scenario=_scenario_yaml_dict(),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="match the target motion",
            output_dir=tmp_path / "out",
            reference_images=[reference, reference],
            reference_descriptions=["only one"],
        )

    with pytest.raises(ValueError, match="reference_video_descriptions"):
        RefineInput(
            scenario=_scenario_yaml_dict(),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="match the target motion",
            output_dir=tmp_path / "out",
            reference_videos=[video],
            reference_video_descriptions=[],
        )

    with pytest.raises(ValueError, match="reference_video_descriptions"):
        RefineInput(
            scenario=_scenario_yaml_dict(),
            physics_usd=_fake_usd(tmp_path),
            user_prompt="match the target motion",
            output_dir=tmp_path / "out",
            reference_videos=[video, video],
            reference_video_descriptions=["only one"],
        )


# ---------------------------------------------------------------------------
# run_refine end-to-end (orchestrator stubbed)
# ---------------------------------------------------------------------------


class _FakeIterativeResult:
    """Minimal stand-in for IterativePhysicsRefinementResult — only the
    attributes ``arun_refine`` reads."""

    def __init__(
        self,
        output_dir: Path,
        *,
        termination_reason: str = "max_iterations",
        iterations: list[Any] | None = None,
        final_iteration: int = 1,
        final_dir: Path | None = None,
        final_recording_usd: Path | None = None,
        final_recording_error: str | None = None,
        user_prompt: str = "bouncy",
    ) -> None:
        self.output_dir = output_dir
        self.termination_reason = termination_reason
        self.iterations = iterations or []
        self.final_iteration = final_iteration
        self.final_dir = final_dir
        self.final_recording_usd = final_recording_usd
        self.final_recording_error = final_recording_error
        self.user_prompt = user_prompt


class _FakeIterationRecord:
    def __init__(
        self,
        iteration: int,
        *,
        judge_decision: str = "approve",
        judge_score: float = 0.9,
        best_score: float = 0.1,
        metric_value: float | None = 0.05,
        error: str | None = None,
        cancelled: bool = False,
    ) -> None:
        self.iteration = iteration
        self.iteration_dir = Path(f"/tmp/iter_{iteration}")
        self.judge_decision = judge_decision
        self.judge_score = judge_score
        self.judge_reasoning = "stubbed"
        self.best_score = best_score
        self.n_trials = 3
        self.metric_name = "settle_distance"
        self.metric_value = metric_value
        self.cancelled = cancelled
        self.error = error


def test_run_refine_happy_path(tmp_path: Path, monkeypatch) -> None:
    """When the orchestrator returns approved, run_refine returns a
    ``RefineOutput`` with ``success=True`` and the iteration summary."""
    from physics_agent.api.refine import RefineInput, run_refine

    final_dir = tmp_path / "final"
    final_dir.mkdir()
    final_recording = final_dir / "recording.usd"
    final_recording.write_bytes(b"recording")

    class _FakeTask:
        last_kwargs: dict[str, Any] | None = None

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            type(self).last_kwargs = kwargs

        def run(self, ctx: dict[str, Any]) -> _FakeIterativeResult:
            return _FakeIterativeResult(
                output_dir=tmp_path / "out",
                termination_reason="approved",
                iterations=[_FakeIterationRecord(1, judge_decision="approve")],
                final_iteration=1,
                final_dir=final_dir,
                final_recording_usd=final_recording,
                user_prompt="bouncy",
            )

    monkeypatch.setattr(
        "physics_agent.tasks.iterative_physics_refinement.IterativePhysicsRefinementTask",
        _FakeTask,
    )

    cancel_event = threading.Event()
    params = RefineInput(
        scenario=_scenario_yaml_dict(),
        physics_usd=_fake_usd(tmp_path),
        user_prompt="bouncy",
        output_dir=tmp_path / "out",
        max_iterations=1,
        history_window=7,
        judge_max_tokens=777,
        judge_temperature=0.25,
        reference_video_frames=12,
        judge_reference_frames=11,
        judge_generated_frames=22,
        cancel_event=cancel_event,
    )

    result = run_refine(params)
    assert result.success is True
    assert result.termination_reason == "approved"
    assert result.iteration_count == 1
    assert result.final_iteration == 1
    assert result.final_dir == final_dir
    assert result.final_recording_usd == final_recording
    assert result.final_recording_error is None
    assert result.iterations[0].judge_decision == "approve"
    assert result.iterations[0].judge_score == pytest.approx(0.9)
    assert result.final_judge_score == pytest.approx(0.9)
    assert _FakeTask.last_kwargs is not None
    assert _FakeTask.last_kwargs["history_window"] == 7
    assert _FakeTask.last_kwargs["judge_max_tokens"] == 777
    assert _FakeTask.last_kwargs["judge_temperature"] == 0.25
    assert _FakeTask.last_kwargs["reference_video_frames"] == 12
    assert _FakeTask.last_kwargs["judge_reference_frames"] == 11
    assert _FakeTask.last_kwargs["judge_generated_frames"] == 22
    assert _FakeTask.last_kwargs["visual_evidence_enabled"] is True
    assert _FakeTask.last_kwargs["cancel_event"] is cancel_event


def test_run_refine_passes_visual_evidence_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    from physics_agent.api.refine import RefineInput, run_refine

    final_dir = tmp_path / "final"
    final_dir.mkdir()

    class _FakeTask:
        last_kwargs: dict[str, Any] | None = None

        def __init__(self, **kwargs: Any) -> None:
            type(self).last_kwargs = kwargs

        def run(self, ctx: dict[str, Any]) -> _FakeIterativeResult:
            return _FakeIterativeResult(
                output_dir=tmp_path / "out",
                termination_reason="approved",
                iterations=[_FakeIterationRecord(1, judge_decision="approve")],
                final_iteration=1,
                final_dir=final_dir,
                user_prompt="bouncy",
            )

    monkeypatch.setattr(
        "physics_agent.tasks.iterative_physics_refinement.IterativePhysicsRefinementTask",
        _FakeTask,
    )

    params = RefineInput(
        scenario=_scenario_yaml_dict(),
        physics_usd=_fake_usd(tmp_path),
        user_prompt="bouncy",
        output_dir=tmp_path / "out",
        max_iterations=1,
        visual_evidence_enabled=False,
    )

    result = run_refine(params)
    assert result.success is True
    assert _FakeTask.last_kwargs is not None
    assert _FakeTask.last_kwargs["visual_evidence_enabled"] is False


def test_run_refine_coerces_non_finite_scores(tmp_path: Path, monkeypatch) -> None:
    """When an iteration records ``best_score=inf`` (e.g. all trials
    failed), the public ``IterationSummary`` must coerce it to ``None``
    so the dataclass round-trips through JSON serializers that reject
    non-finite floats. Same for ``judge_score`` / ``metric_value``."""
    from physics_agent.api.refine import RefineInput, run_refine

    class _FakeTask:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def run(self, ctx: dict[str, Any]) -> _FakeIterativeResult:
            return _FakeIterativeResult(
                output_dir=tmp_path / "out",
                termination_reason="max_iterations",
                iterations=[
                    _FakeIterationRecord(
                        1,
                        judge_decision="continue",
                        judge_score=float("inf"),
                        best_score=float("inf"),
                        metric_value=float("nan"),
                    )
                ],
            )

    monkeypatch.setattr(
        "physics_agent.tasks.iterative_physics_refinement.IterativePhysicsRefinementTask",
        _FakeTask,
    )

    params = RefineInput(
        scenario=_scenario_yaml_dict(),
        physics_usd=_fake_usd(tmp_path),
        user_prompt="settle",
        output_dir=tmp_path / "out",
        max_iterations=1,
    )

    result = run_refine(params)
    assert result.iterations[0].judge_score is None
    assert result.iterations[0].best_score is None
    assert result.iterations[0].metric_value is None
    # The terminal judge score is also surfaced — should be None.
    assert result.final_judge_score is None


def test_run_refine_surfaces_error_termination(tmp_path: Path, monkeypatch) -> None:
    """``termination_reason="error"`` must map to ``success=False`` and
    publish only fixed, value-free failure diagnostics."""
    from physics_agent.api.refine import RefineInput, run_refine

    sentinel = "refine-iteration-provider-secret-713"

    class _FakeTask:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def run(self, ctx: dict[str, Any]) -> _FakeIterativeResult:
            return _FakeIterativeResult(
                output_dir=tmp_path / "out",
                termination_reason="error",
                iterations=[
                    _FakeIterationRecord(1, judge_decision="skipped", error=sentinel)
                ],
            )

    monkeypatch.setattr(
        "physics_agent.tasks.iterative_physics_refinement.IterativePhysicsRefinementTask",
        _FakeTask,
    )

    params = RefineInput(
        scenario=_scenario_yaml_dict(),
        physics_usd=_fake_usd(tmp_path),
        user_prompt="settle",
        output_dir=tmp_path / "out",
        max_iterations=1,
    )

    result = run_refine(params)
    assert result.success is False
    assert result.termination_reason == "error"
    assert result.error == "Physics refinement failed"
    assert result.iterations[0].error == "Physics refinement iteration failed"
    assert sentinel not in repr(result)


def test_arun_refine_is_awaitable(tmp_path: Path, monkeypatch) -> None:
    """``arun_refine`` must return a coroutine that can be awaited inside
    an existing event loop. (Sync ``run_refine`` would explode inside
    a loop with ``RuntimeError: asyncio.run() cannot be called from a
    running event loop``; this test pins the async contract.)"""
    from physics_agent.api.refine import RefineInput, arun_refine

    class _FakeTask:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def run(self, ctx: dict[str, Any]) -> _FakeIterativeResult:
            return _FakeIterativeResult(
                output_dir=tmp_path / "out",
                termination_reason="approved",
                iterations=[_FakeIterationRecord(1)],
                final_iteration=1,
            )

    monkeypatch.setattr(
        "physics_agent.tasks.iterative_physics_refinement.IterativePhysicsRefinementTask",
        _FakeTask,
    )

    params = RefineInput(
        scenario=_scenario_yaml_dict(),
        physics_usd=_fake_usd(tmp_path),
        user_prompt="bouncy",
        output_dir=tmp_path / "out",
        max_iterations=1,
    )

    async def _runner() -> Any:
        return await arun_refine(params)

    result = asyncio.run(_runner())
    assert result.success is True
    assert result.termination_reason == "approved"
