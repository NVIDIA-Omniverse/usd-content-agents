# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for defensive branches retained by the release forward-port."""

from __future__ import annotations

import importlib
from unittest.mock import Mock

import pytest

import material_agent.workflows as workflows
import material_agent.workflows.factory as workflow_factory
from material_agent.api.scene_pipeline import ScenePipelineInput
from material_agent.scene.manifest import SceneManifest
from material_agent.tasks.render import _blank_final_render_error

build_dataset_module = importlib.import_module("material_agent.api.build_dataset")
evaluate_module = importlib.import_module("material_agent.api.evaluate")
pipeline_module = importlib.import_module("material_agent.api.pipeline")
refine_module = importlib.import_module("material_agent.api.refine")
scene_pipeline_module = importlib.import_module("material_agent.api.scene_pipeline")


class _AsyncWorkflow:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result

    async def arun(self, *args: object, **kwargs: object) -> dict[str, object]:
        return self.result


@pytest.mark.asyncio
async def test_public_api_projection_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert build_dataset_module._projected_mapping_list({}) == []
    assert build_dataset_module._projected_string_list({}) == []
    assert refine_module._projected_float({"score": object()}, "score") is None

    listener = Mock()
    params = pipeline_module.PipelineInput(
        config={"project": {}}, event_listener=listener
    )
    object.__setattr__(params, "skip_steps", {"unexpected": "mapping"})
    object.__setattr__(params, "only_steps", {"unexpected": "mapping"})
    monkeypatch.setattr(
        workflows,
        "create_unified_pipeline_workflow",
        lambda: _AsyncWorkflow({"error": "failed", "pipeline_results": {}}),
    )
    monkeypatch.setattr(
        pipeline_module,
        "project_result_metadata",
        lambda result: {"pipeline_results": []},
    )
    failed = await pipeline_module.arun_pipeline(params)
    assert not failed.success and failed.step_results == {}

    monkeypatch.setattr(
        workflows,
        "create_unified_pipeline_workflow",
        lambda: _AsyncWorkflow({"pipeline_results": {}}),
    )
    succeeded = await pipeline_module.arun_pipeline(params)
    assert succeeded.success and succeeded.step_results == {}

    monkeypatch.setattr(
        workflows,
        "create_evaluation_workflow_from_config",
        lambda: _AsyncWorkflow({"evaluation_complete": True}),
    )
    monkeypatch.setattr(
        evaluate_module,
        "project_result_metadata",
        lambda result: {"metrics": []},
    )
    evaluated = await evaluate_module.arun_evaluate(
        evaluate_module.EvaluateInput(config={"evaluate": {}})
    )
    assert evaluated.success and evaluated.metrics is not None

    monkeypatch.setattr(
        workflow_factory,
        "create_iterative_apply_workflow_from_config",
        lambda: _AsyncWorkflow({"iteration_count": 1}),
    )
    monkeypatch.setattr(
        refine_module,
        "project_result_metadata",
        lambda result: {
            "iteration_count": 1,
            "iteration_results": {},
            "final_iteration": [],
        },
    )
    refined = await refine_module.arun_refine(
        refine_module.RefineInput(config={"refine": {}})
    )
    assert refined.success
    assert refined.iteration_results == []
    assert refined.final_judge_score is None


def test_scene_projection_fallbacks(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    assert scene_pipeline_module._project_public_path(object()) == ""
    monkeypatch.setattr(
        scene_pipeline_module,
        "project_result_metadata",
        lambda value: {"items": {}},
    )
    assert scene_pipeline_module._project_public_strings(["warning"]) == []

    monkeypatch.setattr(
        scene_pipeline_module,
        "project_result_metadata",
        lambda value: {"validation_report": [], "raw_result": []},
    )
    projected = scene_pipeline_module._build_output(
        success=True,
        error=None,
        working_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        output_path=tmp_path / "output.usd",
        rendered_images=[],
        manifest=SceneManifest(),
        validation_passed=True,
        validation_report={},
        scene_harness_summary_path="",
        warnings=[],
    )
    assert projected.validation_report is None
    assert projected.raw_result == {}


def test_scene_pipeline_emits_active_stage_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    material_path = tmp_path / "materials.usda"
    material_path.write_text("#usda 1.0\n", encoding="utf-8")
    listener = Mock()
    config = {
        "project": {"name": "scene", "working_dir": str(tmp_path / "work")},
        "input": {"usd_path": str(usd_path)},
        "materials": {
            "library_path": str(material_path),
            "entries": [{"name": "Steel", "prim_path": "/Looks/Steel"}],
        },
        "scene": {},
    }
    monkeypatch.setattr(
        scene_pipeline_module,
        "_validate_large_scene_stage_file",
        lambda path: "/Root",
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        Mock(side_effect=RuntimeError("analysis failed")),
    )

    result = scene_pipeline_module.run_scene_pipeline(
        ScenePipelineInput(
            config=config, config_base_dir=tmp_path, event_listener=listener
        )
    )

    assert not result.success
    listener.event.assert_any_call(
        "step.failed",
        {
            "step_name": "scene_analyze",
            "workflow_type": "scene_pipeline",
            "error": "Scene pipeline failed",
        },
    )


def test_unknown_render_backend_keeps_actionable_diagnostic(tmp_path) -> None:
    message = _blank_final_render_error(
        tmp_path / "blank.png",
        {"reason": "blank"},
        "custom",
    )
    assert "custom rendering backend configuration" in message
