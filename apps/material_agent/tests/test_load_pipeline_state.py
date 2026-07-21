# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for _load_pipeline_state helper in unified_pipeline_executor."""

import json
import logging
from pathlib import Path

import pytest

from material_agent.tasks.unified_pipeline_executor import _load_pipeline_state


class TestLoadPipelineStateNoFile:
    """Tests when no state file exists on disk."""

    def test_returns_fresh_state(self, tmp_path: Path):
        """No .pipeline_state.json → fresh state with given IDs."""
        state = _load_pipeline_state(
            working_dir=str(tmp_path),
            session_id="sess-1",
            project_name="proj-1",
            resume=False,
        )

        assert state["session_id"] == "sess-1"
        assert state["project_name"] == "proj-1"
        assert state["completed_steps"] == []
        assert state["failed_steps"] == []
        assert state["step_outputs"] == {}
        assert state["current_step"] is None

    def test_fresh_state_with_resume_flag(self, tmp_path: Path):
        """resume=True but no file → still returns fresh state."""
        state = _load_pipeline_state(
            working_dir=str(tmp_path),
            session_id="sess-1",
            project_name="proj-1",
            resume=True,
        )

        assert state["completed_steps"] == []
        assert state["step_outputs"] == {}


class TestLoadPipelineStateNoResume:
    """Tests when state file exists but resume=False."""

    def test_fresh_state_carries_over_step_outputs(self, tmp_path: Path):
        """resume=False with existing state → fresh state, step_outputs carried over."""
        saved = {
            "session_id": "old-sess",
            "project_name": "old-proj",
            "completed_steps": ["render", "predict"],
            "failed_steps": ["apply"],
            "step_outputs": {
                "optimize_usd": {"optimized_usd_path": "/tmp/opt.usd"},
                "predict": {"predictions_path": "/tmp/pred.json"},
            },
            "current_step": "apply",
        }
        state_file = tmp_path / ".pipeline_state.json"
        state_file.write_text(json.dumps(saved), encoding="utf-8")

        state = _load_pipeline_state(
            working_dir=str(tmp_path),
            session_id="new-sess",
            project_name="new-proj",
            resume=False,
        )

        # Fresh metadata
        assert state["session_id"] == "new-sess"
        assert state["project_name"] == "new-proj"
        assert state["completed_steps"] == []
        assert state["failed_steps"] == []
        assert state["current_step"] is None

        # step_outputs carried over for auto-wiring
        assert state["step_outputs"] == saved["step_outputs"]


class TestLoadPipelineStateResume:
    """Tests when state file exists and resume=True."""

    def test_returns_saved_state(self, tmp_path: Path):
        """resume=True → returns full saved state."""
        saved = {
            "session_id": "sess-1",
            "project_name": "proj-1",
            "completed_steps": ["render", "predict"],
            "failed_steps": [],
            "step_outputs": {"render": {"image": "/tmp/img.png"}},
            "current_step": "apply",
        }
        state_file = tmp_path / ".pipeline_state.json"
        state_file.write_text(json.dumps(saved), encoding="utf-8")

        state = _load_pipeline_state(
            working_dir=str(tmp_path),
            session_id="sess-1",
            project_name="proj-1",
            resume=True,
        )

        assert state == saved

    def test_session_id_mismatch_warns_and_overrides(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """resume=True with mismatched session IDs → warns, uses current ID."""
        saved = {
            "session_id": "old-sess",
            "project_name": "proj-1",
            "completed_steps": ["render"],
            "failed_steps": [],
            "step_outputs": {},
            "current_step": None,
        }
        state_file = tmp_path / ".pipeline_state.json"
        state_file.write_text(json.dumps(saved), encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            state = _load_pipeline_state(
                working_dir=str(tmp_path),
                session_id="new-sess",
                project_name="proj-1",
                resume=True,
            )

        assert state["session_id"] == "new-sess"
        assert "Session ID mismatch" in caplog.text
        assert "old-sess" in caplog.text
        assert "new-sess" in caplog.text
        # Other fields preserved from saved state
        assert state["completed_steps"] == ["render"]

    def test_session_id_mismatch_ignored_when_saved_is_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """No warning when saved session_id is None."""
        saved = {
            "session_id": None,
            "project_name": "proj-1",
            "completed_steps": ["render"],
            "failed_steps": [],
            "step_outputs": {},
            "current_step": None,
        }
        state_file = tmp_path / ".pipeline_state.json"
        state_file.write_text(json.dumps(saved), encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            state = _load_pipeline_state(
                working_dir=str(tmp_path),
                session_id="new-sess",
                project_name="proj-1",
                resume=True,
            )

        assert "Session ID mismatch" not in caplog.text
        assert state["completed_steps"] == ["render"]


class TestLoadPipelineStateCorrupted:
    """Tests when the state file contains invalid JSON."""

    def test_corrupted_json_returns_fresh_state(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """Corrupted JSON → logs warning, returns fresh state."""
        state_file = tmp_path / ".pipeline_state.json"
        state_file.write_text("{bad json!!", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            state = _load_pipeline_state(
                working_dir=str(tmp_path),
                session_id="sess-1",
                project_name="proj-1",
                resume=True,
            )

        assert state["session_id"] == "sess-1"
        assert state["completed_steps"] == []
        assert state["step_outputs"] == {}
        assert "Could not read pipeline state file" in caplog.text

    def test_empty_file_returns_fresh_state(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """Empty file → same as corrupted JSON."""
        state_file = tmp_path / ".pipeline_state.json"
        state_file.write_text("", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            state = _load_pipeline_state(
                working_dir=str(tmp_path),
                session_id="sess-1",
                project_name="proj-1",
                resume=False,
            )

        assert state["completed_steps"] == []
        assert "Could not read pipeline state file" in caplog.text

    @pytest.mark.parametrize(
        "malformed_state",
        [
            {"completed_steps": None},
            {
                "completed_steps": [],
                "failed_steps": [],
                "step_outputs": [],
                "current_step": None,
            },
            {
                "completed_steps": [7],
                "failed_steps": [],
                "step_outputs": {},
                "current_step": None,
            },
            {
                "completed_steps": [],
                "failed_steps": [],
                "step_outputs": {"predict": "not-a-mapping"},
                "current_step": None,
            },
            {
                "completed_steps": [],
                "failed_steps": [],
                "step_outputs": {},
                "step_errors": {"predict": ["not", "text"]},
                "current_step": None,
            },
        ],
    )
    def test_malformed_structure_fails_before_resume_access(
        self,
        tmp_path: Path,
        malformed_state: dict[str, object],
    ) -> None:
        state_file = tmp_path / ".pipeline_state.json"
        state_file.write_text(json.dumps(malformed_state), encoding="utf-8")

        with pytest.raises(
            ValueError,
            match="^Invalid pipeline checkpoint structure:",
        ):
            _load_pipeline_state(
                working_dir=str(tmp_path),
                session_id="sess-1",
                project_name="proj-1",
                resume=True,
            )
