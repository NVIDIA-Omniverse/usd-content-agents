# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Targeted edge coverage for the unified pipeline executor."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from physics_agent.tasks.unified_pipeline_executor import UnifiedPipelineExecutorTask


class _Listener:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def event(self, name: str, payload: dict[str, Any]) -> None:
        self.events.append((name, payload))


class _Workflow:
    def __init__(self, result: dict[str, Any] | None) -> None:
        self.result = result
        self.contexts: list[dict[str, Any]] = []

    def run(self, context: dict[str, Any]) -> dict[str, Any] | None:
        self.contexts.append(context)
        return self.result


class _Mode(Enum):
    FAST = "fast"


class _UnknownRuntimeObject:
    def __init__(self, rendered: str = "runtime-object") -> None:
        self.rendered = rendered
        self.render_count = 0
        self.repr_count = 0
        self.dump_count = 0

    def __str__(self) -> str:
        self.render_count += 1
        return self.rendered

    def __repr__(self) -> str:
        self.repr_count += 1
        return self.rendered

    def model_dump(self) -> dict[str, str]:
        self.dump_count += 1
        return {"value": self.rendered}


def test_executor_required_keys_state_file_and_clean_guards(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    assert executor._get_step_list_key() == "steps_to_run"
    assert executor._get_required_context_keys() == ["steps_to_run", "step_configs"]
    assert executor._get_state_file({"working_dir": tmp_path}) == (
        tmp_path / ".pipeline_state.json"
    )

    with pytest.raises(ValueError, match="Refusing to delete"):
        executor.run(
            {
                "steps_to_run": ["predict"],
                "step_configs": {"predict": {}},
                "working_dir": Path("/"),
                "clean": True,
            }
        )

    with pytest.raises(ValueError, match="path too shallow"):
        executor.run(
            {
                "steps_to_run": ["predict"],
                "step_configs": {"predict": {}},
                "working_dir": Path("x"),
                "clean": True,
            }
        )


def test_executor_run_clean_resume_success_and_failure_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "stale.txt").write_text("old", encoding="utf-8")
    listener = _Listener()

    monkeypatch.setattr(
        executor,
        "_execute_step",
        lambda *_args, **_kwargs: {"predictions_path": "preds.jsonl"},
    )
    context = executor.run(
        {
            "steps_to_run": ["predict"],
            "step_configs": {"predict": {}},
            "working_dir": workdir,
            "working_dir_base": tmp_path,
            "clean": True,
            "event_listener": listener,
            "session_id": "s",
            "project_name": "p",
        }
    )
    assert not (workdir / "stale.txt").exists()
    assert context["pipeline_state"] == "completed"
    assert (
        "step.completed",
        {"step_name": "predict", "outputs": {"predictions_path": "preds.jsonl"}},
    ) in listener.events

    state_file = workdir / ".pipeline_state.json"
    state_file.write_text(
        json.dumps(
            {
                "completed_steps": ["predict"],
                "failed_steps": [],
                "step_outputs": {"predict": {"predictions_path": "old.jsonl"}},
                "current_step": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        executor,
        "_execute_step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("resume should skip")
        ),
    )
    resumed = executor.run(
        {
            "steps_to_run": ["predict"],
            "step_configs": {"predict": {}},
            "working_dir": workdir,
            "resume": True,
        }
    )
    assert resumed["pipeline_results"]["predict"]["predictions_path"] == "old.jsonl"

    secret = "physics-provider-failure-api-key-713"

    def fail_step(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise ValueError(f"provider failed with api_key={secret}")

    listener = _Listener()
    monkeypatch.setattr(executor, "_execute_step", fail_step)
    failure_dir = tmp_path / "fail"
    with pytest.raises(
        RuntimeError,
        match=("Pipeline failed at step 'predict': ValueError during step execution"),
    ) as exc_info:
        executor.run(
            {
                "steps_to_run": ["predict"],
                "step_configs": {"predict": {}},
                "working_dir": failure_dir,
                "event_listener": listener,
            }
        )
    assert listener.events[-1] == (
        "step.failed",
        {
            "step_name": "predict",
            "error": "ValueError during step execution",
        },
    )
    checkpoint = (failure_dir / ".pipeline_state.json").read_text(encoding="utf-8")
    observable = caplog.text + str(exc_info.value) + repr(listener.events) + checkpoint
    assert secret not in observable
    assert exc_info.value.__cause__ is None
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
async def test_executor_cooperatively_stops_after_active_threaded_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    cancel_event = threading.Event()
    step_started = threading.Event()
    release_step = threading.Event()
    executed_steps: list[str] = []

    def execute_step(step_name: str, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        executed_steps.append(step_name)
        step_started.set()
        assert release_step.wait(timeout=2)
        return {}

    monkeypatch.setattr(executor, "_execute_step", execute_step)
    task = asyncio.create_task(
        executor.arun(
            {
                "steps_to_run": ["predict", "apply_physics"],
                "step_configs": {"predict": {}, "apply_physics": {}},
                "working_dir": tmp_path / "cancel-work",
                "cancel_event": cancel_event,
            }
        )
    )
    assert await asyncio.to_thread(step_started.wait, 1)

    cancel_event.set()
    release_step.set()

    result = await task
    assert executed_steps == ["predict"]
    assert result["pipeline_cancelled"] is True
    assert result["pipeline_state"] == "cancelled"
    assert result["pipeline_results"] == {"predict": {}}


def test_executor_preserves_completed_outputs_when_later_step_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()

    def execute_step(step_name: str, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        if step_name == "predict":
            return {"predictions_path": "predictions.jsonl"}
        raise ValueError("later step failed")

    monkeypatch.setattr(executor, "_execute_step", execute_step)
    context: dict[str, Any] = {
        "steps_to_run": ["predict", "apply_physics"],
        "step_configs": {"predict": {}, "apply_physics": {}},
        "working_dir": tmp_path / "partial-failure",
    }

    with pytest.raises(RuntimeError, match="apply_physics"):
        executor.run(context)

    assert context["pipeline_state"] == "failed"
    assert context["pipeline_results"] == {
        "predict": {"predictions_path": "predictions.jsonl"}
    }


def test_execute_step_autowire_and_workflow_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    context = {"working_dir": tmp_path}

    with pytest.raises(ValueError, match="optimized_usd_path"):
        executor._execute_step(
            "apply_physics",
            {},
            context,
            None,
            {"step_outputs": {"optimize_usd": {}}},
        )

    assert executor._execute_step(
        "restore_usd",
        {},
        context,
        None,
        {"step_outputs": {}},
    ) == {"restore_skipped": True, "reason": "no optimization metadata"}

    with pytest.raises(ValueError, match="Unknown step"):
        executor._execute_step("unknown", {}, context, None, {"step_outputs": {}})

    captured_workflow = _Workflow(None)

    def workflow_factory() -> _Workflow:
        return captured_workflow

    import physics_agent.workflows as workflows

    monkeypatch.setattr(
        workflows,
        "create_prediction_workflow_from_config",
        workflow_factory,
    )
    with pytest.raises(RuntimeError, match="empty result"):
        executor._execute_step("predict", {}, context, None, {"step_outputs": {}})

    captured_workflow.result = {
        "error": "bad",
        "failed_task": "predict_task",
    }
    with pytest.raises(RuntimeError, match="predict_task"):
        executor._execute_step("predict", {}, context, None, {"step_outputs": {}})

    captured_workflow.result = {
        "predictions_path": "preds.jsonl",
        "predictions_count": 1,
        "failed_count": 0,
        "output_key": "classification",
    }
    outputs = executor._execute_step(
        "predict",
        {
            "report": {
                "image_max_size": 512,
                "image_format": "webp",
                "image_quality": 80,
            }
        },
        {
            "working_dir": tmp_path,
            "event_listener": _Listener(),
            "config_path": tmp_path / "pipeline.yaml",
        },
        None,
        {"step_outputs": {}},
    )
    assert outputs["predictions_path"] == "preds.jsonl"
    assert captured_workflow.contexts[-1]["report_image_max_size"] == 512
    assert captured_workflow.contexts[-1]["report_image_format"] == "webp"
    assert captured_workflow.contexts[-1]["report_image_quality"] == 80
    assert captured_workflow.contexts[-1]["config_path"] == str(
        tmp_path / "pipeline.yaml"
    )

    apply_workflow = _Workflow({"output_usd_path": "out.usda"})
    monkeypatch.setattr(
        workflows,
        "create_apply_physics_workflow_from_config",
        lambda: apply_workflow,
    )
    assert executor._execute_step(
        "apply_physics",
        {},
        context,
        None,
        {
            "step_outputs": {
                "restore_usd": {"restored_predictions_path": "restored.jsonl"},
                "predict": {"predictions_path": "preds.jsonl"},
            }
        },
    ) == {"output_usd_path": "out.usda"}
    restored_config = apply_workflow.contexts[-1]["config_dict"]
    assert restored_config["predictions_path"] == "restored.jsonl"

    assert executor._execute_step(
        "apply_physics",
        {},
        context,
        None,
        {"step_outputs": {"predict": {"predictions_path": "preds.jsonl"}}},
    ) == {"output_usd_path": "out.usda"}
    predict_config = apply_workflow.contexts[-1]["config_dict"]
    assert predict_config["predictions_path"] == "preds.jsonl"

    assert executor._execute_step(
        "apply_physics",
        {},
        context,
        None,
        {"step_outputs": {}},
    ) == {"output_usd_path": "out.usda"}


def test_make_yaml_safe_and_runtime_config_edges(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    safe = executor._make_yaml_safe(
        {
            Path("key"): {"path": Path("/tmp/a"), "mode": _Mode.FAST},
            "items": {Path("/tmp/b"), _Mode.FAST},
        }
    )
    assert safe["key"] == {"path": "/tmp/a", "mode": "fast"}
    assert sorted(safe["items"]) == ["/tmp/b", "fast"]

    source: dict[str, Any] = {
        "renderer": {"_runtime": object(), "name": "rt"},
        "path": Path("/tmp/data"),
        "vlm": {
            "api_key": "xy",
            "nested": [{"token": "${RUNTIME_TOKEN}"}],
        },
    }
    runtime = executor._prepare_runtime_config(source)
    assert runtime == {
        "renderer": {"name": "rt"},
        "path": "/tmp/data",
        "vlm": {
            "api_key": "xy",
            "nested": [{"token": "${RUNTIME_TOKEN}"}],
        },
    }
    assert "_runtime" in source["renderer"]
    assert not (tmp_path / ".pipeline_temp").exists()


@pytest.mark.parametrize("nested_in_set", [False, True])
def test_make_yaml_safe_rejects_opaque_objects_without_rendering(
    nested_in_set: bool,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    sentinel = "physics-runtime-object-secret-713"
    opaque = _UnknownRuntimeObject(sentinel)
    value: object = {opaque, "safe"} if nested_in_set else opaque

    with pytest.raises(
        TypeError,
        match="^Unsupported YAML-equivalent configuration value$",
    ) as exc_info:
        executor._make_yaml_safe({"opaque": value})

    assert opaque.render_count == 0
    assert opaque.repr_count == 0
    assert opaque.dump_count == 0
    assert sentinel not in str(exc_info.value)


def test_runtime_config_isolated_across_concurrent_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import physics_agent.workflows as workflows

    captured: list[_Workflow] = []

    def workflow_factory() -> _Workflow:
        workflow = _Workflow(
            {
                "predictions_path": "preds.jsonl",
                "predictions_count": 1,
                "output_key": "classification",
            }
        )
        captured.append(workflow)
        return workflow

    monkeypatch.setattr(
        workflows, "create_prediction_workflow_from_config", workflow_factory
    )
    executor = UnifiedPipelineExecutorTask()
    configs = [
        {"vlm": {"api_key": "a", "nested": [{"secret": "short-a"}]}},
        {"vlm": {"api_key": "b", "nested": [{"secret": "short-b"}]}},
    ]

    def execute(config: dict[str, Any]) -> dict[str, Any]:
        return executor._execute_step(
            "predict",
            config,
            {"working_dir": tmp_path},
            None,
            {"step_outputs": {}},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(execute, configs))

    observed = {
        workflow.contexts[0]["config_dict"]["vlm"]["api_key"] for workflow in captured
    }
    assert observed == {"a", "b"}
    assert configs[0]["vlm"]["nested"][0]["secret"] == "short-a"
    assert configs[1]["vlm"]["nested"][0]["secret"] == "short-b"
    assert not (tmp_path / ".pipeline_temp").exists()


def test_failure_and_resume_never_persist_runtime_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import physics_agent.workflows as workflows

    workflow = _Workflow({"error": "transient", "failed_task": "predict"})
    monkeypatch.setattr(
        workflows, "create_prediction_workflow_from_config", lambda: workflow
    )
    executor = UnifiedPipelineExecutorTask()
    working_dir = tmp_path / "work"
    context = {
        "steps_to_run": ["predict"],
        "step_configs": {
            "predict": {
                "vlm": {
                    "api_key": "retry-secret",
                    "nested": [{"token": "xy"}],
                }
            }
        },
        "working_dir": working_dir,
    }

    with pytest.raises(RuntimeError, match="Pipeline failed at step 'predict'"):
        executor.run(context)
    assert "retry-secret" not in "".join(
        path.read_text(encoding="utf-8")
        for path in working_dir.rglob("*")
        if path.is_file()
    )

    legacy_temp = working_dir / ".pipeline_temp"
    legacy_temp.mkdir()
    (legacy_temp / "predict.yaml").write_text(
        "api_key: pre-fix-secret\n", encoding="utf-8"
    )
    workflow.result = {
        "predictions_path": "predictions.jsonl",
        "predictions_count": 1,
        "output_key": "classification",
    }
    context["resume"] = True
    result = executor.run(context)
    assert result["pipeline_state"] == "completed"
    assert not (working_dir / ".pipeline_temp").exists()
    artifact_text = "".join(
        path.read_text(encoding="utf-8")
        for path in working_dir.rglob("*")
        if path.is_file()
    )
    assert "retry-secret" not in artifact_text
    assert "pre-fix-secret" not in artifact_text


def test_legacy_temp_cleanup_failure_stops_before_step_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = UnifiedPipelineExecutorTask()
    executed = False

    def fail_cleanup(_working_dir: Path) -> bool:
        raise PermissionError("cleanup denied")

    def execute(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal executed
        executed = True
        return {}

    monkeypatch.setattr(
        "physics_agent.tasks.unified_pipeline_executor."
        "remove_legacy_pipeline_temp_with_safe_diagnostics",
        fail_cleanup,
    )
    monkeypatch.setattr(executor, "_execute_step", execute)
    with pytest.raises(PermissionError, match="cleanup denied"):
        executor.run(
            {
                "steps_to_run": ["predict"],
                "step_configs": {"predict": {}},
                "working_dir": tmp_path / "work",
            }
        )
    assert executed is False
