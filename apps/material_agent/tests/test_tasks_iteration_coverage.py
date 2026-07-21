# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional coverage for iterative sub-workflow execution."""

from __future__ import annotations

from pathlib import Path

from material_agent.tasks.iteration import IterationTask


class _FakeWorkflow:
    def __init__(self, *results: dict | Exception) -> None:
        self.name = "fake iteration workflow"
        self.results = list(results)
        self.contexts: list[dict] = []

    def run(self, initial_context: dict) -> dict:
        self.contexts.append(initial_context)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_iteration_task_runs_until_approved_and_propagates_feedback(
    tmp_path: Path,
) -> None:
    workflow = _FakeWorkflow(
        {
            "continue_iteration": True,
            "output_usd_path": str(tmp_path / "iter1.usd"),
            "predictions_path": tmp_path / "iter1_predictions.jsonl",
            "total_predictions": 2,
            "materials_applied": {"Steel": "/World/Looks/Steel"},
            "assignment_stats": {"total_prims": 3},
            "rendered_image_paths": ["render.png"],
            "judge_score": 2.5,
            "judge_reasoning": "needs work",
            "judge_critique": "make it less shiny",
            "previous_prim_feedback": {"/A": "wrong material"},
            "resolved_assignments": {"/B": "Rubber"},
        },
        {
            "continue_iteration": False,
            "output_usd_path": str(tmp_path / "iter2.usd"),
            "total_predictions": 1,
            "materials_applied": {},
            "assignment_stats": {"total_prims": 1},
            "judge_score": 4.5,
            "judge_reasoning": "approved",
        },
    )

    context = {
        "input_usd_path": str(tmp_path / "input.usd"),
        "config_path": "config.yaml",
        "dataset": [{"id": "/A"}],
        "render_enabled": True,
        "max_iterations": 3,
        "save_intermediate": True,
        "intermediate_output_dir": tmp_path / "iterations",
    }
    result = IterationTask(workflow).run(context)

    assert result["termination_reason"] == "approved"
    assert result["iteration_count"] == 2
    assert result["final_iteration"]["judge_score"] == 4.5
    assert result["all_iteration_outputs"] == [
        str(tmp_path / "iter1.usd"),
        str(tmp_path / "iter2.usd"),
    ]
    first_context, second_context = workflow.contexts
    assert first_context["is_first_iteration"] is True
    assert first_context["render_output_dir"].name == "renders"
    assert first_context["input_usd_path_original"] == str(tmp_path / "input.usd")
    assert second_context["previous_judge_critique"] == "make it less shiny"
    assert second_context["previous_prim_feedback"] == {"/A": "wrong material"}
    assert second_context["previous_predictions_path"].endswith(
        "iter1_predictions.jsonl"
    )
    assert second_context["resolved_assignments"] == {"/B": "Rubber"}


def test_iteration_task_stops_at_max_iterations_without_intermediate_outputs(
    tmp_path: Path,
) -> None:
    workflow = _FakeWorkflow({"keep_going": True, "judge_score": 3.0})

    result = IterationTask(
        workflow,
        max_iterations=1,
        save_intermediate=False,
        continue_iteration_key="keep_going",
    ).run({"intermediate_output_dir": tmp_path / "unused"})

    assert result["termination_reason"] == "max_iterations"
    assert result["iteration_count"] == 1
    assert "iteration_output_dir" not in workflow.contexts[0]


def test_iteration_task_records_errors() -> None:
    result = IterationTask(_FakeWorkflow(RuntimeError("boom"))).run(
        {"max_iterations": 2}
    )

    assert result["termination_reason"] == "error"
    assert result["iteration_error"] == "boom"
    assert result["final_iteration"] is None


def test_iteration_prepare_context_uses_previous_iteration_output(
    tmp_path: Path,
) -> None:
    task = IterationTask(_FakeWorkflow())
    prepared = task._prepare_iteration_context(
        context={
            "iteration_results": [{"output_usd_path": str(tmp_path / "previous.usd")}],
        },
        original_context={},
        iteration_num=2,
        intermediate_base_dir=tmp_path / "iterations",
        save_intermediate=False,
    )

    assert prepared["input_usd_path"] == str(tmp_path / "previous.usd")
