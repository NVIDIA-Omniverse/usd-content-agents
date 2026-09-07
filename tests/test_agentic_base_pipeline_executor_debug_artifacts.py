# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-failure debug artifact coverage for the base pipeline executor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from world_understanding.agentic.base_pipeline_executor import BasePipelineExecutor
from world_understanding.utils.debug_traceback import (
    STEP_FAILURE_DEBUG_RELATIVE_PATH,
)
from world_understanding.utils.object_store import ObjectStore

# Local-only sentinel that must never appear in any persisted failure
# artifact: it exists solely as a frame local of the failing step, so a
# formatter that started serializing frame locals would leak it.
_FRAME_LOCAL_SENTINEL = "frame-local-only-sentinel-947-do-not-persist"


class _FailingExecutor(BasePipelineExecutor):
    """Minimal executor whose only step raises a credential-bearing error."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def _execute_step(
        self,
        step_name: str,
        context: dict[str, Any],
        object_store: ObjectStore | None,
    ) -> dict[str, Any]:
        local_credential = _FRAME_LOCAL_SENTINEL  # noqa: F841 - locals leak canary
        if self.error is not None:
            raise self.error
        return {"status": "completed"}

    def _get_step_list_key(self) -> str:
        return "steps_to_run"

    def _get_required_context_keys(self) -> list[str]:
        return ["steps_to_run"]

    def _get_state_file(self, context: dict[str, Any]) -> Path:
        return Path(context["working_dir"]) / ".pipeline_state.json"


def test_step_failure_writes_scrubbed_debug_artifact(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "nvapi-base-executor-debug-secret-713"
    executor = _FailingExecutor(ValueError(f"provider failed with api_key={secret}"))
    working_dir = tmp_path / "work"
    working_dir.mkdir()

    failure_match = "ValueError during step execution"
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError, match=failure_match) as exc_info:
            executor.run(
                {
                    "steps_to_run": ["predict"],
                    "working_dir": working_dir,
                }
            )

    # The public failure surfaces stay value-free exactly as before.
    assert str(exc_info.value) == (
        "Pipeline failed at step 'predict': ValueError during step execution"
    )
    assert exc_info.value.__cause__ is None
    checkpoint = (working_dir / ".pipeline_state.json").read_text(encoding="utf-8")
    assert secret not in caplog.text + str(exc_info.value) + checkpoint

    # The session-local debug artifact records the structured cause: the
    # exception category and frame locations, never the message text.
    debug_file = working_dir / STEP_FAILURE_DEBUG_RELATIVE_PATH
    assert debug_file.exists()
    content = debug_file.read_text(encoding="utf-8")
    assert "step failure: predict" in content
    assert "Traceback (most recent call last)" in content
    assert "ValueError" in content
    assert "_execute_step" in content
    assert "exception message withheld" in content
    assert secret not in content
    assert "provider failed" not in content
    # Frame locals are never serialized into any persisted failure surface.
    assert _FRAME_LOCAL_SENTINEL not in content
    assert _FRAME_LOCAL_SENTINEL not in caplog.text + str(exc_info.value) + checkpoint


def test_successful_step_writes_no_debug_artifact(tmp_path: Path) -> None:
    executor = _FailingExecutor(None)
    working_dir = tmp_path / "work"
    working_dir.mkdir()

    executor.run({"steps_to_run": ["predict"], "working_dir": working_dir})
    assert not (working_dir / STEP_FAILURE_DEBUG_RELATIVE_PATH).exists()
