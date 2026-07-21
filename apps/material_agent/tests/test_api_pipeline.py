# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Pipeline API."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from material_agent.api.pipeline import (
    PipelineInput,
    PipelineOutput,
    _dry_run_pipeline,
    arun_pipeline,
    run_pipeline,
)


class TestPipelineInput:
    """Tests for PipelineInput validation."""

    def test_pipeline_input_valid(self, tmp_path):
        """Test creating valid PipelineInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = PipelineInput(
            config=config_file,
            skip_steps=["build_dataset_usd"],
            only_steps=[],
            resume=True,
            dry_run=False,
            clean=True,
            verbose=True,
        )

        assert params.config == config_file
        assert params.skip_steps == ["build_dataset_usd"]
        assert params.resume is True
        assert params.clean is True

    def test_pipeline_input_missing_config(self, tmp_path):
        """Test PipelineInput raises error for missing config."""
        config_file = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            PipelineInput(config=config_file)

    def test_pipeline_input_defaults(self, tmp_path):
        """Test PipelineInput with default values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = PipelineInput(config=config_file)

        assert params.skip_steps == []
        assert params.only_steps == []
        assert params.resume is False
        assert params.dry_run is False
        assert params.clean is False

    def test_pipeline_input_empty_dict(self):
        """Test PipelineInput rejects empty config dictionaries."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PipelineInput(config={})


class TestRunPipeline:
    """Tests for run_pipeline function."""

    @patch("material_agent.workflows.create_unified_pipeline_workflow")
    def test_run_pipeline_success(self, mock_create_workflow, tmp_path):
        """Test successful pipeline execution."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "pipeline_results": {
                    "predict": {"predictions_path": "/path/to/predictions.jsonl"},
                    "apply": {"output_usd_path": "/path/to/output.usd"},
                }
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = PipelineInput(config=config_file)
        result = run_pipeline(params)

        # Verify
        assert result.success is True
        assert "predict" in result.step_results
        assert "apply" in result.step_results
        assert result.completed_steps == ["predict", "apply"]

    @pytest.mark.asyncio
    async def test_arun_pipeline_projects_success_result_without_mutating_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Successful API results detach diagnostics from the raw workflow context."""
        sentinel = "material-success-result-credential-713"
        listener = Mock()
        config = {
            "project": {"name": "demo"},
            "steps": {"predict": {"vlm": {"api_key": sentinel}}},
        }
        runtime_results: list[dict[str, Any]] = []

        async def run_with_raw_context(
            context: dict[str, Any],
        ) -> dict[str, Any]:
            context["pipeline_results"] = {
                "predict": {
                    "api_key": sentinel,
                    "num_predictions": 1,
                }
            }
            runtime_results.append(context)
            return context

        workflow = Mock()
        workflow.arun = AsyncMock(side_effect=run_with_raw_context)
        monkeypatch.setattr(
            "material_agent.workflows.create_unified_pipeline_workflow",
            lambda: workflow,
        )

        output = await arun_pipeline(
            PipelineInput(
                config=config,
                skip_steps=[f"https://user:{sentinel}@skip.example.test/predict"],
                event_listener=listener,
            )
        )

        runtime_result = runtime_results[0]
        assert runtime_result["config_dict"] is config
        assert (
            runtime_result["config_dict"]["steps"]["predict"]["vlm"]["api_key"]
            == sentinel
        )
        assert runtime_result["pipeline_results"]["predict"]["api_key"] == sentinel
        assert runtime_result["event_listener"] is listener
        assert output.success is True
        assert output.step_results["predict"]["num_predictions"] == 1
        assert output.skipped_steps == ["<redacted>"]
        assert output.raw_result is not runtime_result
        assert output.raw_result is not None
        assert "config_dict" not in output.raw_result
        assert "event_listener" not in output.raw_result
        assert sentinel not in repr(output)

    @patch("material_agent.workflows.create_unified_pipeline_workflow")
    def test_run_pipeline_with_skip_steps(self, mock_create_workflow, tmp_path):
        """Test pipeline with skip steps."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={"pipeline_results": {"predict": {}, "apply": {}}}
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = PipelineInput(
            config=config_file,
            skip_steps=["build_dataset_usd", "build_dataset_pdf_vectorstore"],
        )
        result = run_pipeline(params)

        # Verify context passed to workflow
        call_args = mock_workflow.arun.call_args[0][0]
        assert call_args["skip_steps"] == [
            "build_dataset_usd",
            "build_dataset_pdf_vectorstore",
        ]
        assert result.success is True

    @patch("material_agent.workflows.create_unified_pipeline_workflow")
    def test_run_pipeline_passes_cancel_checker(self, mock_create_workflow, tmp_path):
        """Test cancellation callback is forwarded to the workflow context."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        cancel_checker = Mock(return_value=False)

        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={"pipeline_results": {"predict": {}}}
        )
        mock_create_workflow.return_value = mock_workflow

        result = run_pipeline(
            PipelineInput(config=config_file, cancel_checker=cancel_checker)
        )

        call_args = mock_workflow.arun.call_args[0][0]
        assert result.success is True
        assert call_args["cancel_checker"] is cancel_checker

    @patch("material_agent.workflows.create_unified_pipeline_workflow")
    def test_run_pipeline_cancel_checker_stops_before_workflow(
        self, mock_create_workflow, tmp_path
    ):
        """Test cancellation is not converted to a failed PipelineOutput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        with pytest.raises(asyncio.CancelledError):
            run_pipeline(PipelineInput(config=config_file, cancel_checker=lambda: True))

        mock_create_workflow.assert_not_called()

    @patch("material_agent.workflows.create_unified_pipeline_workflow")
    def test_run_pipeline_with_only_steps(self, mock_create_workflow, tmp_path):
        """Test pipeline with only specific steps."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={"pipeline_results": {"predict": {}}}
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = PipelineInput(
            config=config_file,
            only_steps=["predict"],
        )
        result = run_pipeline(params)

        # Verify context
        call_args = mock_workflow.arun.call_args[0][0]
        assert call_args["only_steps"] == ["predict"]
        assert result.success is True

    @patch("material_agent.api.pipeline._dry_run_pipeline")
    def test_run_pipeline_dry_run(self, mock_dry_run, tmp_path):
        """Test pipeline dry run."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "project:\n  name: test\nsteps:\n  predict:\n    enabled: true"
        )

        # Mock dry run
        mock_dry_run.return_value = PipelineOutput(
            success=True,
            completed_steps=["predict", "apply"],
            skipped_steps=["build_dataset_usd"],
        )

        # Execute
        params = PipelineInput(config=config_file, dry_run=True)
        result = run_pipeline(params)

        # Verify
        assert result.success is True
        assert result.completed_steps == ["predict", "apply"]
        assert result.skipped_steps == ["build_dataset_usd"]

    @patch("material_agent.workflows.create_unified_pipeline_workflow")
    def test_run_pipeline_no_results(self, mock_create_workflow, tmp_path):
        """Test pipeline when workflow returns None."""
        # Setup
        sentinel = "material-api-skip-credential-713"
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that returns None
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(return_value=None)
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = PipelineInput(
            config=config_file,
            skip_steps=[f"https://user:{sentinel}@skip.example.test/step"],
        )
        result = run_pipeline(params)

        # Verify
        assert result.success is False
        assert "did not complete" in result.error.lower()
        assert result.skipped_steps == ["<redacted>"]
        assert sentinel not in repr(result)

    @patch("material_agent.workflows.create_unified_pipeline_workflow")
    def test_run_pipeline_exception(self, mock_create_workflow, tmp_path, caplog):
        """Test pipeline when exception occurs."""
        # Setup
        sentinel = "never-replay-pipeline-exception-713"
        config_dir = tmp_path / f"api_key={sentinel}"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that raises exception
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            side_effect=RuntimeError(f"backend reflected {sentinel}")
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = PipelineInput(config=config_file)
        result = run_pipeline(params)

        # Verify
        assert result.success is False
        assert result.error == "Pipeline execution failed"
        assert sentinel not in caplog.text

    @pytest.mark.asyncio
    async def test_arun_pipeline_with_dict_config_and_default_listener(
        self, monkeypatch
    ):
        """Test async pipeline uses default listener and dict config."""
        listener = Mock()
        workflow = Mock()
        workflow.arun = AsyncMock(return_value={"pipeline_results": {"predict": {}}})

        monkeypatch.setattr(
            "world_understanding.agentic.events.create_default_listener",
            lambda verbose=False: listener,
        )
        monkeypatch.setattr(
            "material_agent.workflows.create_unified_pipeline_workflow",
            lambda: workflow,
        )

        params = PipelineInput(
            config={"project": {"name": "demo"}},
            skip_steps=["build_dataset_usd"],
            only_steps=["predict"],
            resume=True,
            clean=True,
            verbose=True,
            session_id="session-1",
        )
        result = await arun_pipeline(params)

        assert result.success is True
        call_args = workflow.arun.call_args[0][0]
        assert call_args["config_dict"] == {"project": {"name": "demo"}}
        assert call_args["event_listener"] is listener
        assert call_args["session_id"] == "session-1"
        listener.info.assert_any_call("Using in-memory config dictionary")
        listener.info.assert_any_call("Resume mode enabled")
        listener.info.assert_any_call(
            "Clean mode enabled (will delete working dir and output files)"
        )
        listener.info.assert_any_call("Using provided session ID: session-1")

    @pytest.mark.asyncio
    async def test_arun_pipeline_simulate_mode_from_file_and_partial_failure(
        self, monkeypatch, tmp_path
    ):
        """Test simulate-mode patching and workflow partial failure handling."""
        sentinel = "never-replay-partial-pipeline-error-713"
        config_dir = tmp_path / f"access_token={sentinel}"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("project:\n  name: demo\n", encoding="utf-8")

        listener = Mock()
        workflow = Mock()
        runtime_marker = object()
        workflow.arun = AsyncMock(
            return_value={
                "error": f"backend reflected {sentinel}",
                "failed_task": f"api_key={sentinel}",
                "pipeline_results": {
                    "build_dataset_usd": {
                        "num_prims": 3,
                        "runtime_marker": runtime_marker,
                    }
                },
            }
        )

        monkeypatch.setattr(
            "material_agent.api.simulate_config.patch_config_for_simulate",
            lambda config: {"patched": True, **config},
        )
        monkeypatch.setattr(
            "material_agent.workflows.create_unified_pipeline_workflow",
            lambda: workflow,
        )

        params = PipelineInput(
            config=config_file, simulate=True, event_listener=listener
        )
        result = await arun_pipeline(params)

        assert result.success is False
        assert result.error == "Pipeline execution failed"
        assert result.completed_steps == ["build_dataset_usd"]
        assert result.step_results == {"build_dataset_usd": {"num_prims": 3}}
        assert result.raw_result == {
            "error": "Pipeline execution failed",
            "failed_task": "<redacted>",
            "pipeline_results": {"build_dataset_usd": {"num_prims": 3}},
        }
        call_args = workflow.arun.call_args[0][0]
        assert call_args["config_dict"]["patched"] is True
        assert call_args["config_path"] == str(config_file)
        listener.info.assert_any_call("Simulate mode: all backends patched to 'mock'")
        listener.info.assert_any_call("Configuration file: <redacted>")
        listener.event.assert_any_call(
            "workflow.failed",
            {
                "workflow_type": "pipeline",
                "error": "Pipeline execution failed",
                "failed_task": "<redacted>",
            },
        )
        assert sentinel not in repr(result)
        assert sentinel not in repr(listener.method_calls)

    @pytest.mark.asyncio
    async def test_arun_pipeline_simulate_mode_from_config_dict(
        self, monkeypatch, tmp_path
    ):
        """Test simulate mode patches in-memory config dictionaries."""
        listener = Mock()
        workflow = Mock()
        workflow.arun = AsyncMock(return_value={"pipeline_results": {"predict": {}}})

        monkeypatch.setattr(
            "material_agent.api.simulate_config.patch_config_for_simulate",
            lambda config: {"patched": True, **config},
        )
        monkeypatch.setattr(
            "material_agent.workflows.create_unified_pipeline_workflow",
            lambda: workflow,
        )

        config_anchor = tmp_path / "source" / "config.yaml"
        result = await arun_pipeline(
            PipelineInput(
                config={"project": {"name": "demo"}},
                config_path=config_anchor,
                simulate=True,
                event_listener=listener,
            )
        )

        assert result.success is True
        call_args = workflow.arun.call_args[0][0]
        assert call_args["config_dict"]["patched"] is True
        assert call_args["config_path"] == str(config_anchor)

    @pytest.mark.asyncio
    async def test_arun_pipeline_preserves_dict_config_path_anchor(
        self, monkeypatch, tmp_path
    ):
        """A dict config retains its source path for relative-path resolution."""
        listener = Mock()
        workflow = Mock()
        workflow.arun = AsyncMock(return_value={"pipeline_results": {"predict": {}}})
        monkeypatch.setattr(
            "material_agent.workflows.create_unified_pipeline_workflow",
            lambda: workflow,
        )

        config_anchor = tmp_path / "source" / "config.yaml"
        result = await arun_pipeline(
            PipelineInput(
                config={"project": {"name": "demo"}},
                config_path=config_anchor,
                event_listener=listener,
            )
        )

        assert result.success is True
        call_args = workflow.arun.call_args[0][0]
        assert call_args["config_dict"] == {"project": {"name": "demo"}}
        assert call_args["config_path"] == str(config_anchor)

    def test_dry_run_pipeline_filters_steps_for_unified_config(self):
        """Test actual dry-run helper for unified config rules."""
        params = PipelineInput(
            config={
                "project": {"name": "demo"},
                "steps": {
                    "build_dataset_usd": {"enabled": True},
                    "predict": {"temperature": 0.0},
                    "evaluate": {"enabled": True},
                    "apply": {"enabled": False},
                },
            },
            skip_steps=["build_dataset_usd"],
            only_steps=["predict"],
            dry_run=True,
        )

        result = _dry_run_pipeline(params)

        assert result.success is True
        assert result.completed_steps == ["predict"]
        assert result.skipped_steps == ["build_dataset_usd", "evaluate"]

    def test_dry_run_pipeline_treats_null_steps_as_empty(self):
        result = _dry_run_pipeline(
            PipelineInput(
                config={"project": {"name": "demo"}, "steps": None},
                dry_run=True,
            )
        )

        assert result.success is True
        assert result.completed_steps == []
        assert result.skipped_steps == []

    def test_dry_run_pipeline_returns_error_for_invalid_yaml(self, tmp_path):
        """Test dry-run helper wraps YAML parsing errors."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("[", encoding="utf-8")

        result = _dry_run_pipeline(PipelineInput(config=config_file))

        assert result.success is False
        assert result.error


class TestPipelineOutput:
    """Tests for PipelineOutput dataclass."""

    def test_pipeline_output_success(self):
        """Test creating successful PipelineOutput."""
        output = PipelineOutput(
            success=True,
            step_results={
                "predict": {"predictions_path": "/path/to/pred.jsonl"},
                "apply": {"output_usd_path": "/path/to/output.usd"},
            },
            completed_steps=["predict", "apply"],
            skipped_steps=["build_dataset_usd"],
        )

        assert output.success is True
        assert len(output.step_results) == 2
        assert output.completed_steps == ["predict", "apply"]
        assert output.skipped_steps == ["build_dataset_usd"]

    def test_pipeline_output_error(self):
        """Test creating error PipelineOutput."""
        output = PipelineOutput(
            success=False,
            error="Test error",
        )

        assert output.success is False
        assert output.error == "Test error"
        assert output.step_results == {}
