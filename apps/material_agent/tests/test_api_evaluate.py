# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Evaluate API."""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from material_agent.api.evaluate import (
    EvaluateInput,
    EvaluateOutput,
    run_evaluate,
)


class TestEvaluateInput:
    """Tests for EvaluateInput validation."""

    def test_evaluate_input_valid(self, tmp_path):
        """Test creating valid EvaluateInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = EvaluateInput(config=config_file, verbose=True)

        assert params.config == config_file
        assert params.verbose is True
        assert params.predictions_override is None

    def test_evaluate_input_missing_config(self, tmp_path):
        """Test EvaluateInput raises error for missing config."""
        sentinel = "api_key=missing-evaluate-config-713"
        config_file = tmp_path / sentinel / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="^Config file not found$") as exc:
            EvaluateInput(config=config_file)

        assert sentinel not in str(exc.value)

    def test_evaluate_input_projects_real_name_too_long_config_error(self):
        sentinel = "api_key=evaluate-config-name-too-long-713"
        config_file = Path("/tmp") / (sentinel + "x" * 5000)

        with pytest.raises(OSError, match="^Unable to inspect config file$") as exc:
            EvaluateInput(config=config_file)

        assert sentinel not in repr(exc.value)
        assert exc.value.__cause__ is None
        assert exc.value.__context__ is None

    def test_evaluate_input_rejects_empty_config_dict(self):
        """Test EvaluateInput raises error for empty config dictionaries."""
        with pytest.raises(ValueError, match="Config dictionary cannot be empty"):
            EvaluateInput(config={})

    def test_evaluate_input_missing_predictions(self, tmp_path):
        """Test EvaluateInput raises error for missing predictions file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        sentinel = "api_key=missing-predictions-713"
        predictions_file = tmp_path / sentinel / "missing.jsonl"

        with pytest.raises(
            FileNotFoundError,
            match="^Predictions file not found$",
        ) as exc:
            EvaluateInput(
                config=config_file,
                predictions_override=predictions_file,
            )

        assert sentinel not in str(exc.value)

    def test_evaluate_input_projects_real_name_too_long_predictions_error(
        self, tmp_path
    ):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        sentinel = "api_key=predictions-name-too-long-713"
        predictions_file = Path("/tmp") / (sentinel + "x" * 5000)

        with pytest.raises(OSError) as exc:
            EvaluateInput(
                config=config_file,
                predictions_override=predictions_file,
            )

        assert sentinel not in repr(exc.value)
        assert exc.value.filename == "<redacted>"
        assert exc.value.__cause__ is None
        assert exc.value.__context__ is None

    def test_evaluate_input_with_predictions(self, tmp_path):
        """Test EvaluateInput with predictions override."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        predictions_file = tmp_path / "predictions.jsonl"
        predictions_file.write_text("{}")

        params = EvaluateInput(
            config=config_file,
            predictions_override=predictions_file,
        )

        assert params.predictions_override == predictions_file


class TestRunEvaluate:
    """Tests for run_evaluate function."""

    @patch("material_agent.workflows.create_evaluation_workflow_from_config")
    def test_run_evaluate_success(self, mock_create_workflow, tmp_path):
        """Test successful evaluation execution."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "evaluation_complete": True,
                "metrics": {
                    "functional_correctness_score": 4.2,
                    "success_rate": 85.0,
                    "total_cases": 50,
                },
                "evaluation_path": str(tmp_path / "evaluation.jsonl"),
                "html_report_path": str(tmp_path / "report.html"),
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = EvaluateInput(config=config_file)
        result = run_evaluate(params)

        # Verify
        assert result.success is True
        assert result.metrics is not None
        assert result.metrics.functional_correctness_score == 4.2
        assert result.evaluation_path == tmp_path / "evaluation.jsonl"
        assert result.html_report_path == tmp_path / "report.html"

    @patch("material_agent.workflows.create_evaluation_workflow_from_config")
    def test_run_evaluate_with_predictions_override(
        self, mock_create_workflow, tmp_path
    ):
        """Test evaluation with predictions override."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        predictions_file = tmp_path / "predictions.jsonl"
        predictions_file.write_text("{}")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "evaluation_complete": True,
                "metrics": {"functional_correctness_score": 3.5},
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = EvaluateInput(
            config=config_file,
            predictions_override=predictions_file,
        )
        result = run_evaluate(params)

        # Verify context passed to workflow
        call_args = mock_workflow.arun.call_args[1]["initial_context"]
        assert call_args["predictions_path"] == str(predictions_file)
        assert result.success is True

    @patch("material_agent.workflows.create_evaluation_workflow_from_config")
    def test_run_evaluate_with_config_dict(self, mock_create_workflow, tmp_path):
        """Test evaluation with an in-memory config dictionary."""
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "evaluation_complete": True,
                "metrics": {"functional_correctness_score": 3.5},
            }
        )
        mock_create_workflow.return_value = mock_workflow

        config_anchor = tmp_path / "source" / "config.yaml"
        result = run_evaluate(
            EvaluateInput(
                config={"evaluation": {"enabled": True}},
                config_path=config_anchor,
            )
        )

        call_args = mock_workflow.arun.call_args.kwargs["initial_context"]
        assert call_args["config_dict"] == {"evaluation": {"enabled": True}}
        assert call_args["config_path"] == str(config_anchor)
        assert result.success is True

    @patch("material_agent.workflows.create_evaluation_workflow_from_config")
    def test_success_result_projects_context_and_metrics_without_mutating_runtime(
        self,
        mock_create_workflow: Mock,
        tmp_path: Path,
    ) -> None:
        sentinel = "material-evaluate-result-credential-713"
        config = {"evaluation": {"vlm": {"api_key": sentinel}}}
        runtime_result: dict[str, Any] = {
            "evaluation_complete": True,
            "metrics": {
                "functional_correctness_score": 4.0,
                "score_distribution": {
                    "valid": 1,
                    "api_key": sentinel,
                },
            },
            "evaluation_path": str(tmp_path / "evaluation.jsonl"),
            "html_report_path": str(tmp_path / "report.html"),
            "config_dict": config,
            "path_resolver": object(),
        }
        workflow = Mock()
        workflow.arun = AsyncMock(return_value=runtime_result)
        mock_create_workflow.return_value = workflow

        result = run_evaluate(EvaluateInput(config=config))

        runtime_context = workflow.arun.await_args.kwargs["initial_context"]
        assert runtime_context["config_dict"] is config
        assert config["evaluation"]["vlm"]["api_key"] == sentinel
        assert runtime_result["metrics"]["score_distribution"]["api_key"] == sentinel
        assert result.success is True
        assert result.metrics is not None
        assert result.metrics.functional_correctness_score == 4.0
        assert result.evaluation_path == tmp_path / "evaluation.jsonl"
        assert result.html_report_path == tmp_path / "report.html"
        assert result.raw_result is not runtime_result
        assert result.raw_result is not None
        assert "config_dict" not in result.raw_result
        assert "path_resolver" not in result.raw_result
        assert sentinel not in repr(result)

    @patch("material_agent.workflows.create_evaluation_workflow_from_config")
    def test_projected_metrics_remain_numeric(
        self,
        mock_create_workflow: Mock,
        tmp_path: Path,
    ) -> None:
        sentinel = "api_key=material-evaluate-metric-713"
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "evaluation_complete": True,
                "metrics": {
                    "functional_correctness_score": sentinel,
                    "success_rate": sentinel,
                    "total_cases": sentinel,
                    "score_distribution": {"safe": sentinel},
                },
            }
        )
        mock_create_workflow.return_value = mock_workflow

        result = run_evaluate(EvaluateInput(config=config_file))

        assert result.success is True
        assert result.metrics is not None
        assert result.metrics.functional_correctness_score == 0.0
        assert type(result.metrics.functional_correctness_score) is float
        assert result.metrics.success_rate == 0.0
        assert type(result.metrics.success_rate) is float
        assert result.metrics.total_cases == 0
        assert type(result.metrics.total_cases) is int
        assert result.metrics.score_distribution == {}
        assert sentinel not in repr(result)

    @patch("material_agent.workflows.create_evaluation_workflow_from_config")
    def test_run_evaluate_not_complete(self, mock_create_workflow, tmp_path):
        """Test evaluation when workflow doesn't complete."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that doesn't complete
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(return_value={"evaluation_complete": False})
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = EvaluateInput(config=config_file)
        result = run_evaluate(params)

        # Verify
        assert result.success is False
        assert result.error == "Evaluation failed"

    @patch("material_agent.workflows.create_evaluation_workflow_from_config")
    def test_run_evaluate_exception_diagnostics_are_secret_safe(
        self, mock_create_workflow, tmp_path, caplog
    ):
        """Raw paths reach the workflow but not logs or returned failures."""
        # Setup
        config_sentinel = "api_key=evaluate-config-path-713"
        predictions_sentinel = "api_key=evaluate-predictions-path-713"
        exception_sentinel = "api_key=reflected-evaluate-exception-713"
        config_dir = tmp_path / config_sentinel
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("# test config")
        predictions_dir = tmp_path / predictions_sentinel
        predictions_dir.mkdir()
        predictions_file = predictions_dir / "predictions.jsonl"
        predictions_file.write_text("{}", encoding="utf-8")

        # Mock workflow that raises exception
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(side_effect=ValueError(exception_sentinel))
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = EvaluateInput(
            config=config_file,
            predictions_override=predictions_file,
        )
        with caplog.at_level(logging.INFO):
            result = run_evaluate(params)

        # Verify
        assert result.success is False
        assert result.error == "Evaluation failed"
        mock_workflow.arun.assert_awaited_once_with(
            initial_context={
                "verbose": False,
                "config_path": str(config_file),
                "predictions_path": str(predictions_file),
            }
        )
        observable = f"{caplog.text}\n{result.error}"
        for sentinel in (
            config_sentinel,
            predictions_sentinel,
            exception_sentinel,
        ):
            assert sentinel not in observable
        assert all(
            record.exc_info is None
            for record in caplog.records
            if record.name == "material_agent.api.evaluate"
        )


class TestEvaluateOutput:
    """Tests for EvaluateOutput dataclass."""

    def test_evaluate_output_success(self, tmp_path):
        """Test creating successful EvaluateOutput."""
        from material_agent.api.types import MetricsResult

        metrics = MetricsResult(functional_correctness_score=4.0)

        output = EvaluateOutput(
            success=True,
            metrics=metrics,
            evaluation_path=tmp_path / "eval.jsonl",
            html_report_path=tmp_path / "report.html",
        )

        assert output.success is True
        assert output.metrics == metrics
        assert output.evaluation_path == tmp_path / "eval.jsonl"

    def test_evaluate_output_error(self):
        """Test creating error EvaluateOutput."""
        output = EvaluateOutput(
            success=False,
            error="Test error",
        )

        assert output.success is False
        assert output.error == "Test error"
        assert output.metrics is None
