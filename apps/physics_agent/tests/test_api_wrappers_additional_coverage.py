# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional coverage for physics-agent public API wrapper branches."""

from __future__ import annotations

import asyncio
import importlib
import threading
from pathlib import Path
from typing import Any

import pytest

from physics_agent.api.build_dataset import (
    BuildDatasetPrepareDatasetInput,
    BuildDatasetUsdInput,
    abuild_dataset_prepare_dataset,
    abuild_dataset_usd,
    build_dataset_prepare_dataset,
    build_dataset_usd,
)
from physics_agent.api.pipeline import (
    PipelineInput,
    apipeline,
    arun_pipeline,
    pipeline,
    run_pipeline,
)
from physics_agent.api.predict import PredictInput, arun_predict, run_predict


class _Listener:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.infos: list[str] = []
        self.errors: list[str] = []

    def event(self, name: str, payload: dict[str, Any]) -> None:
        self.events.append((name, payload))

    def info(self, message: str) -> None:
        self.infos.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)


class _SyncWorkflow:
    def __init__(self, result: dict[str, Any] | Exception):
        self.result = result
        self.contexts: list[dict[str, Any]] = []

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        self.contexts.append(context)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _AsyncWorkflow:
    def __init__(self, result: dict[str, Any] | None | Exception):
        self.result = result
        self.contexts: list[dict[str, Any]] = []

    async def arun(self, context: dict[str, Any]) -> dict[str, Any] | None:
        self.contexts.append(context)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_pipeline_input_validates_empty_dict_and_missing_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        PipelineInput(config={})
    with pytest.raises(FileNotFoundError):
        PipelineInput(config=tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="cancel_event"):
        PipelineInput(config={"project": {}}, cancel_event=object())


@pytest.mark.asyncio
async def test_arun_pipeline_success_error_none_and_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import physics_agent.workflows as workflows

    listener = _Listener()
    cancel_event = threading.Event()
    success_workflow = _AsyncWorkflow(
        {
            "pipeline_results": {"predict": {"count": 1}},
            "session_id": "sid",
            "working_dir": str(tmp_path / "work"),
        }
    )
    monkeypatch.setattr(
        workflows,
        "create_unified_pipeline_workflow",
        lambda: success_workflow,
    )

    result = await arun_pipeline(
        PipelineInput(
            config={"project": {"name": "demo"}},
            skip_steps=["build_dataset_usd"],
            only_steps=["predict"],
            session_id="sid",
            resume=True,
            clean=True,
            verbose=True,
            event_listener=listener,
            cancel_event=cancel_event,
        )
    )

    assert result.success is True
    assert result.completed_steps == ["predict"]
    assert result.session_id == "sid"
    assert result.working_dir == tmp_path / "work"
    context = success_workflow.contexts[-1]
    assert context["config_dict"] == {"project": {"name": "demo"}}
    assert context["session_id"] == "sid"
    assert context["skip_steps"] == ["build_dataset_usd"]
    assert context["cancel_event"] is cancel_event
    assert (
        "workflow.completed",
        {"workflow_type": "pipeline", "completed_steps": ["predict"]},
    ) in listener.events

    none_workflow = _AsyncWorkflow(None)
    monkeypatch.setattr(
        workflows, "create_unified_pipeline_workflow", lambda: none_workflow
    )
    none_result = await arun_pipeline(
        PipelineInput(config={"project": {"name": "demo"}}, event_listener=_Listener())
    )
    assert none_result.success is False
    assert none_result.raw_result is None

    cancelled_workflow = _AsyncWorkflow(
        {
            "workflow_terminated": True,
            "pipeline_cancelled": True,
            "pipeline_results": {"predict": {"count": 1}},
        }
    )
    monkeypatch.setattr(
        workflows,
        "create_unified_pipeline_workflow",
        lambda: cancelled_workflow,
    )
    cancelled_listener = _Listener()
    cancelled_result = await arun_pipeline(
        PipelineInput(
            config={"project": {"name": "demo"}},
            event_listener=cancelled_listener,
        )
    )
    assert cancelled_result.success is False
    assert cancelled_result.cancelled is True
    assert cancelled_result.error == "Pipeline cancelled"
    assert cancelled_result.completed_steps == ["predict"]
    assert any(name == "workflow.cancelled" for name, _ in cancelled_listener.events)
    assert all(name != "workflow.failed" for name, _ in cancelled_listener.events)

    sentinel = "physics-api-pipeline-credential-713"
    runtime_marker = object()
    error_workflow = _AsyncWorkflow(
        {
            "workflow_terminated": True,
            "failed_task": f"https://user:{sentinel}@task.example.test/predict",
            "error": f"backend failed with api_key={sentinel}",
            "pipeline_results": {
                "prepare": {
                    "output_path": f"https://user:{sentinel}@result.example.test/out",
                    "runtime_marker": runtime_marker,
                }
            },
            "session_id": f"https://user:{sentinel}@session.example.test/id",
            "working_dir": str(tmp_path / "partial"),
        }
    )
    monkeypatch.setattr(
        workflows, "create_unified_pipeline_workflow", lambda: error_workflow
    )
    error_listener = _Listener()
    error_result = await arun_pipeline(
        PipelineInput(
            config={"project": {"name": "demo"}},
            skip_steps=[f"https://user:{sentinel}@skip.example.test/step"],
            event_listener=error_listener,
        )
    )
    assert error_result.success is False
    assert error_result.error == "Pipeline execution failed"
    assert error_result.completed_steps == ["prepare"]
    assert error_result.session_id is None
    assert error_result.working_dir == tmp_path / "partial"
    assert error_result.raw_result is None
    assert error_result.skipped_steps == ["<redacted>"]
    assert sentinel not in repr(
        (
            error_result,
            error_listener.events,
            error_listener.infos,
            error_listener.errors,
        )
    )
    assert "<redacted>" in repr(error_result.step_results)
    assert "runtime_marker" not in error_result.step_results["prepare"]

    boom_workflow = _AsyncWorkflow(
        RuntimeError(f"workflow failed with api_key={sentinel}")
    )
    monkeypatch.setattr(
        workflows, "create_unified_pipeline_workflow", lambda: boom_workflow
    )
    boom_listener = _Listener()
    boom_result = await arun_pipeline(
        PipelineInput(
            config={"project": {"name": "demo"}},
            event_listener=boom_listener,
        )
    )
    assert boom_result.success is False
    assert boom_result.error == "Pipeline execution failed"
    assert sentinel not in repr(
        (boom_result, boom_listener.events, boom_listener.errors)
    )


@pytest.mark.asyncio
async def test_arun_pipeline_projects_success_result_without_mutating_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import physics_agent.workflows as workflows

    sentinel = "physics-success-result-credential-713"
    config = {
        "project": {"name": "demo"},
        "steps": {"predict": {"vlm": {"api_key": sentinel}}},
    }
    listener = _Listener()
    runtime_result: dict[str, Any] = {
        "config_dict": config,
        "event_listener": listener,
        "pipeline_results": {
            "predict": {
                "api_key": sentinel,
                "predictions_count": 1,
            }
        },
        "session_id": "session-1",
        "working_dir": str(tmp_path / "work"),
    }
    workflow = _AsyncWorkflow(runtime_result)
    monkeypatch.setattr(
        workflows,
        "create_unified_pipeline_workflow",
        lambda: workflow,
    )

    output = await arun_pipeline(
        PipelineInput(
            config=config,
            skip_steps=[f"https://user:{sentinel}@skip.example.test/predict"],
            event_listener=listener,
        )
    )

    assert workflow.contexts[0]["config_dict"] is config
    assert config["steps"]["predict"]["vlm"]["api_key"] == sentinel
    assert runtime_result["pipeline_results"]["predict"]["api_key"] == sentinel
    assert output.success is True
    assert output.step_results["predict"]["predictions_count"] == 1
    assert output.skipped_steps == ["<redacted>"]
    assert output.session_id == "session-1"
    assert output.working_dir == tmp_path / "work"
    assert output.raw_result is not runtime_result
    assert output.raw_result is not None
    assert (
        output.raw_result["pipeline_results"] is not runtime_result["pipeline_results"]
    )
    assert (
        output.step_results["predict"]
        is not runtime_result["pipeline_results"]["predict"]
    )
    output.step_results["predict"]["predictions_count"] = 2
    assert runtime_result["pipeline_results"]["predict"]["predictions_count"] == 1
    assert "config_dict" not in output.raw_result
    assert "event_listener" not in output.raw_result
    assert sentinel not in repr(output)


@pytest.mark.asyncio
async def test_arun_pipeline_default_listener_and_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import world_understanding.agentic.events as events

    listener = _Listener()
    monkeypatch.setattr(
        events, "create_default_listener", lambda verbose=False: listener
    )

    config = tmp_path / "pipeline.yaml"
    config.write_text(
        """
steps:
  predict:
    enabled:
    model: qwen
  apply_physics:
    enabled: false
  build_dataset_usd: true
""",
        encoding="utf-8",
    )

    result = await arun_pipeline(
        PipelineInput(
            config=config,
            skip_steps=["build_dataset_usd"],
            only_steps=["predict"],
            dry_run=True,
            verbose=True,
        )
    )

    assert result.success is True
    assert result.completed_steps == ["predict"]
    assert "build_dataset_usd" in result.skipped_steps
    assert listener.infos[0] == "Starting pipeline via API"

    bad = tmp_path / "bad.yaml"
    bad.write_text("[", encoding="utf-8")
    bad_result = await arun_pipeline(PipelineInput(config=bad, dry_run=True))
    assert bad_result.success is False
    assert bad_result.error


def test_pipeline_sync_and_convenience_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_mod = importlib.import_module("physics_agent.api.pipeline")

    async def fake_arun(params: PipelineInput):
        return type("Output", (), {"success": True, "raw": params})()

    async def fake_apipeline(*args: Any, **kwargs: Any):
        return type("Output", (), {"success": True, "args": args, "kwargs": kwargs})()

    monkeypatch.setattr(pipeline_mod, "arun_pipeline", fake_arun)
    result = run_pipeline(PipelineInput(config={"project": {"name": "demo"}}))
    assert result.success is True

    apipeline_result = asyncio.run(
        apipeline(
            {"project": {"name": "demo"}},
            skip_steps=None,
            only_steps=None,
            session_id="sid",
            resume=True,
            dry_run=True,
            clean=True,
            event_listener=_Listener(),
            verbose=True,
        )
    )
    assert apipeline_result.success is True

    monkeypatch.setattr(pipeline_mod, "apipeline", fake_apipeline)
    sync_result = pipeline({"project": {"name": "demo"}}, skip_steps=["x"])
    assert sync_result.success is True


@pytest.mark.asyncio
async def test_build_dataset_usd_success_error_defaults_and_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import physics_agent.workflows as workflows

    workflow = _SyncWorkflow(
        {
            "output_dir": str(tmp_path / "dataset"),
            "num_prims": 4,
            "num_images": 12,
            "batch_results": {"a.usd": {"status": "success"}},
        }
    )
    monkeypatch.setattr(
        workflows,
        "create_usd_data_preparation_workflow_from_config",
        lambda: workflow,
    )

    result = await abuild_dataset_usd(
        BuildDatasetUsdInput(
            config={"usd_path": "asset.usd"},
            source_override=tmp_path / "asset.usd",
            output_dir_override=tmp_path / "out",
            extract_metadata=True,
            verbose=True,
        )
    )

    assert result.success is True
    assert result.dataset_path == tmp_path / "dataset"
    assert result.num_prims == 4
    assert workflow.contexts[-1]["config_dict"] == {"usd_path": "asset.usd"}
    assert workflow.contexts[-1]["source_override"] == str(tmp_path / "asset.usd")

    sentinel = "usd-build-provider-secret-713"
    workflow.result = {"workflow_terminated": True, "error": sentinel}
    failed = await abuild_dataset_usd(
        BuildDatasetUsdInput(config=tmp_path / "cfg.yaml")
    )
    assert failed.success is False
    assert failed.error == "USD dataset building failed"

    workflow.result = RuntimeError(sentinel)
    exploded = await abuild_dataset_usd(BuildDatasetUsdInput(config={}))
    assert exploded.success is False
    assert exploded.error == "USD dataset building failed"
    assert sentinel not in repr((failed, exploded, caplog.records))


@pytest.mark.asyncio
async def test_build_dataset_prepare_success_error_and_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import physics_agent.workflows as workflows

    workflow = _SyncWorkflow(
        {
            "dataset_entries": [{"id": "a"}],
            "dataset_jsonl_path": str(tmp_path / "dataset.jsonl"),
            "failed_models": ["bad"],
        }
    )
    monkeypatch.setattr(
        workflows,
        "create_prepare_dataset_workflow_from_config",
        lambda: workflow,
    )

    result = await abuild_dataset_prepare_dataset(
        BuildDatasetPrepareDatasetInput(
            config={"dataset": "raw"},
            dataset_override=tmp_path / "raw",
            verbose=True,
        )
    )
    assert result.success is True
    assert result.dataset_jsonl_path == tmp_path / "dataset.jsonl"
    assert workflow.contexts[-1]["dataset_override"] == str(tmp_path / "raw")

    sentinel = "prepare-dataset-provider-secret-713"
    workflow.result = {"error": sentinel}
    failed = await abuild_dataset_prepare_dataset(
        BuildDatasetPrepareDatasetInput(config=tmp_path / "cfg.yaml")
    )
    assert failed.success is False
    assert failed.error == "Dataset preparation failed"

    workflow.result = RuntimeError(sentinel)
    exploded = await abuild_dataset_prepare_dataset(
        BuildDatasetPrepareDatasetInput(config={})
    )
    assert exploded.success is False
    assert exploded.error == "Dataset preparation failed"
    assert sentinel not in repr((failed, exploded, caplog.records))


def test_build_dataset_sync_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    import physics_agent.api.build_dataset as build_mod

    async def fake_usd(_params: BuildDatasetUsdInput):
        return type("Output", (), {"success": True})()

    async def fake_prepare(_params: BuildDatasetPrepareDatasetInput):
        return type("Output", (), {"success": True})()

    monkeypatch.setattr(build_mod, "abuild_dataset_usd", fake_usd)
    monkeypatch.setattr(build_mod, "abuild_dataset_prepare_dataset", fake_prepare)

    assert build_dataset_usd(BuildDatasetUsdInput(config={})).success is True
    assert (
        build_dataset_prepare_dataset(
            BuildDatasetPrepareDatasetInput(config={})
        ).success
        is True
    )


@pytest.mark.asyncio
async def test_predict_success_error_and_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import physics_agent.workflows as workflows

    workflow = _AsyncWorkflow(
        {
            "predictions_path": str(tmp_path / "predictions.jsonl"),
            "predictions_count": 5,
            "failed_count": 1,
            "token_stats": {"total_tokens": 42},
        }
    )
    monkeypatch.setattr(
        workflows,
        "create_prediction_workflow_from_config",
        lambda: workflow,
    )

    result = await arun_predict(
        PredictInput(
            config={"predict": {}},
            dataset_override=tmp_path / "dataset.jsonl",
            output_dir_override=tmp_path / "out",
            resume=True,
            stream_predictions=False,
            verbose=True,
        )
    )
    assert result.success is True
    assert result.predictions_path == tmp_path / "predictions.jsonl"
    assert workflow.contexts[-1]["dataset_override"] == str(tmp_path / "dataset.jsonl")
    assert workflow.contexts[-1]["stream_predictions"] is False

    sentinel = "predict-provider-secret-713"
    workflow.result = {"workflow_terminated": True, "error": sentinel}
    failed = await arun_predict(PredictInput(config=tmp_path / "cfg.yaml"))
    assert failed.success is False
    assert failed.error == "Prediction failed"

    workflow.result = RuntimeError(sentinel)
    exploded = await arun_predict(PredictInput(config={}))
    assert exploded.success is False
    assert exploded.error == "Prediction failed"
    assert sentinel not in repr((failed, exploded, caplog.records))


def test_predict_sync_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    import physics_agent.api.predict as predict_mod

    async def fake_arun(_params: PredictInput):
        return type("Output", (), {"success": True})()

    monkeypatch.setattr(predict_mod, "arun_predict", fake_arun)
    assert run_predict(PredictInput(config={})).success is True
