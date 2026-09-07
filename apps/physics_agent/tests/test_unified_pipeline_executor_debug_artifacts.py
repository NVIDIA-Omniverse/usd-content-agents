# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-failure debug artifact coverage for the unified pipeline executor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from world_understanding.utils.debug_traceback import (
    STEP_FAILURE_DEBUG_RELATIVE_PATH,
)

from physics_agent.tasks.unified_pipeline_executor import UnifiedPipelineExecutorTask


class _Listener:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def event(self, name: str, payload: dict[str, Any]) -> None:
        self.events.append((name, payload))


def test_step_failure_writes_scrubbed_debug_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    secret = "nvapi-unified-executor-debug-secret-713"

    def fail_step(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise ValueError(f"provider failed with api_key={secret}")

    listener = _Listener()
    monkeypatch.setattr(executor, "_execute_step", fail_step)
    working_dir = tmp_path / "fail"

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(
            RuntimeError,
            match=(
                "Pipeline failed at step 'predict': ValueError during step execution"
            ),
        ) as exc_info:
            executor.run(
                {
                    "steps_to_run": ["predict"],
                    "step_configs": {"predict": {}},
                    "working_dir": working_dir,
                    "event_listener": listener,
                }
            )

    # Every public failure surface stays value-free exactly as before.
    assert listener.events[-1] == (
        "step.failed",
        {"step_name": "predict", "error": "ValueError during step execution"},
    )
    assert exc_info.value.__cause__ is None
    checkpoint = (working_dir / ".pipeline_state.json").read_text(encoding="utf-8")
    observable = caplog.text + str(exc_info.value) + repr(listener.events) + checkpoint
    assert secret not in observable

    # The session-local debug artifact records the real cause, scrubbed.
    debug_file = working_dir / STEP_FAILURE_DEBUG_RELATIVE_PATH
    assert debug_file.exists()
    content = debug_file.read_text(encoding="utf-8")
    assert "step failure: predict" in content
    assert "Traceback (most recent call last)" in content
    assert "ValueError" in content
    assert "provider failed" in content
    assert secret not in content
    assert "api_key=[REDACTED]" in content


def test_debug_artifact_is_reset_between_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()

    def fail_step(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("transient failure")

    monkeypatch.setattr(executor, "_execute_step", fail_step)
    working_dir = tmp_path / "retries"
    context: dict[str, Any] = {
        "steps_to_run": ["predict"],
        "step_configs": {"predict": {}},
        "working_dir": working_dir,
    }

    # The debug log lives under .pipeline_temp, which pipeline startup
    # removes: each run's failure entry persists only until the next run
    # starts (bounded local retention, excluded from session sync).
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Pipeline failed at step 'predict'"):
            executor.run(dict(context))
        content = (working_dir / STEP_FAILURE_DEBUG_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
        assert content.count("] step failure: predict") == 1


def test_successful_run_writes_no_debug_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    monkeypatch.setattr(
        executor,
        "_execute_step",
        lambda *_args, **_kwargs: {"predictions_path": "preds.jsonl"},
    )
    working_dir = tmp_path / "ok"

    result = executor.run(
        {
            "steps_to_run": ["predict"],
            "step_configs": {"predict": {}},
            "working_dir": working_dir,
        }
    )
    assert result["pipeline_state"] == "completed"
    assert not (working_dir / STEP_FAILURE_DEBUG_RELATIVE_PATH).exists()
