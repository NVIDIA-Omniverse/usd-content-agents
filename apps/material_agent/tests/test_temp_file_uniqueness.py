# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for secret-safe in-memory child workflow configuration."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import material_agent.workflows as workflows
from material_agent.tasks.unified_pipeline_executor import UnifiedPipelineExecutorTask


class _CaptureWorkflow:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.contexts: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        config = context["config_dict"]
        request_id = config["request_id"]
        time.sleep(0.005)
        with self.lock:
            self.contexts[request_id] = context
        config["credentials"]["nested"][0]["token"] = "child-mutated"
        if self.fail:
            raise RuntimeError("expected child failure")
        return {"predictions_path": f"{request_id}.jsonl"}


def _execute_predict(
    workflow: _CaptureWorkflow,
    root: Path,
    request_id: str,
    secret: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = {
        "request_id": request_id,
        "credentials": {
            "api_key": secret,
            "nested": [{"token": f"nested-{secret}"}],
        },
    }
    context = {
        "working_dir": str(root / request_id),
        "config_path": str(root / request_id / "pipeline.yaml"),
    }
    result = UnifiedPipelineExecutorTask()._execute_step(
        "predict",
        original,
        context,
        object_store=None,
        pipeline_state={"step_outputs": {}},
    )
    return original, result


def test_in_memory_config_preserves_nested_and_short_secrets_without_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = _CaptureWorkflow()
    monkeypatch.setattr(
        workflows, "create_prediction_workflow_from_config", lambda: workflow
    )

    original, result = _execute_predict(workflow, tmp_path, "one", "x")
    received = workflow.contexts["one"]

    assert result["predictions_path"] == "one.jsonl"
    assert received["config_path"] == str(tmp_path / "one" / "pipeline.yaml")
    assert received["config_dict"]["credentials"]["api_key"] == "x"
    assert original["credentials"]["nested"][0]["token"] == "nested-x"
    assert not list(tmp_path.rglob(".pipeline_temp"))
    assert not list(tmp_path.rglob("*_config_*.yaml"))


def test_concurrent_in_memory_configs_do_not_cross_talk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = _CaptureWorkflow()
    monkeypatch.setattr(
        workflows, "create_prediction_workflow_from_config", lambda: workflow
    )
    requests = [(f"request-{index}", f"key-{index}") for index in range(20)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda item: _execute_predict(workflow, tmp_path, *item), requests)
        )

    assert len(workflow.contexts) == len(requests)
    for (request_id, secret), (original, result) in zip(requests, results, strict=True):
        received = workflow.contexts[request_id]["config_dict"]
        assert received["credentials"]["api_key"] == secret
        assert original["credentials"]["nested"][0]["token"] == f"nested-{secret}"
        assert result["predictions_path"] == f"{request_id}.jsonl"
    assert not list(tmp_path.rglob(".pipeline_temp"))


def test_child_failure_leaves_no_config_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = _CaptureWorkflow(fail=True)
    monkeypatch.setattr(
        workflows, "create_prediction_workflow_from_config", lambda: workflow
    )

    with pytest.raises(RuntimeError, match="expected child failure"):
        _execute_predict(workflow, tmp_path, "failure", "tiny")

    assert not list(tmp_path.rglob(".pipeline_temp"))
    assert not list(tmp_path.rglob("*_config_*.yaml"))
