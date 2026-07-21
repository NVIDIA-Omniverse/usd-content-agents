# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Benchmark API."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from world_understanding.agentic.events import CollectingEventListener

from material_agent.api.benchmark import (
    BenchmarkInput,
    BenchmarkOutput,
    run_benchmark,
)
from material_agent.api.defaults import DEFAULT_VLM_BACKEND, DEFAULT_VLM_MAX_WORKERS


class TestBenchmarkInput:
    """Tests for BenchmarkInput validation."""

    def test_benchmark_input_valid(self, tmp_path: Path) -> None:
        """Test creating valid BenchmarkInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = BenchmarkInput(
            config=config_file,
            verbose=True,
        )

        assert params.config == config_file
        assert params.verbose is True
        assert params.resume is False

    def test_benchmark_input_missing_config(self, tmp_path: Path) -> None:
        """Test BenchmarkInput raises error for missing config."""
        sentinel = "api_key=missing-benchmark-config-713"
        config_file = tmp_path / sentinel / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="^Config file not found$") as exc:
            BenchmarkInput(config=config_file)

        assert sentinel not in str(exc.value)

    def test_benchmark_input_projects_real_name_too_long_error(self) -> None:
        sentinel = "api_key=benchmark-name-too-long-713"
        config_file = Path("/tmp") / (sentinel + "x" * 5000)

        with pytest.raises(OSError, match="^Unable to inspect config file$") as exc:
            BenchmarkInput(config=config_file)

        assert sentinel not in repr(exc.value)
        assert exc.value.__cause__ is None
        assert exc.value.__context__ is None

    def test_benchmark_input_with_overrides(self, tmp_path: Path) -> None:
        """Test BenchmarkInput with path overrides."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        dataset_file = tmp_path / "dataset.jsonl"
        output_dir = tmp_path / "output"

        params = BenchmarkInput(
            config=config_file,
            dataset_override=dataset_file,
            output_dir_override=output_dir,
            resume=True,
            stream_predictions=False,
        )

        assert params.dataset_override == dataset_file
        assert params.output_dir_override == output_dir
        assert params.resume is True
        assert params.stream_predictions is False

    def test_benchmark_input_with_dict_config(self, tmp_path: Path) -> None:
        """Test BenchmarkInput with dictionary config."""
        config_dict = {
            "model": {"service": "azure", "name": "gpt-4"},
            "dataset_path": "/path/to/dataset.jsonl",
        }

        config_anchor = tmp_path / "source" / "config.yaml"
        params = BenchmarkInput(
            config=config_dict,
            config_path=config_anchor,
            verbose=True,
        )

        assert params.config == config_dict
        assert params.config_path == config_anchor
        assert params.verbose is True

    def test_benchmark_input_empty_dict(self) -> None:
        """Test BenchmarkInput raises error for empty dict config."""
        with pytest.raises(ValueError, match="Config dictionary cannot be empty"):
            BenchmarkInput(config={})

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_run_benchmark_with_dict_config(
        self, mock_create_workflow: MagicMock, tmp_path: Path
    ) -> None:
        """Test running benchmark with dictionary config."""
        # Setup - in-memory config
        config_dict = {
            "model": {
                "service": "azure",
                "name": "gpt-4",
                "deployment": "test-deployment",
            },
            "dataset_path": str(tmp_path / "dataset.jsonl"),
            "output_dir": str(tmp_path / "output"),
        }

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "metrics": {
                    "functional_correctness_score": 4.0,
                    "success_rate": 80.0,
                    "total_cases": 50,
                },
                "evaluation_path": str(tmp_path / "evaluation.jsonl"),
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        config_anchor = tmp_path / "source" / "config.yaml"
        params = BenchmarkInput(
            config=config_dict,
            config_path=config_anchor,
            verbose=True,
        )
        result = run_benchmark(params)

        # Verify
        assert result.success is True
        assert result.metrics.functional_correctness_score == 4.0

        # Verify config_dict was passed to workflow with defaults applied
        call_args = mock_workflow.arun.call_args[0][0]
        assert "config_dict" in call_args
        assert call_args["config_path"] == str(config_anchor)

        # Verify defaults were applied
        passed_config = call_args["config_dict"]
        # Original user values preserved
        assert passed_config["model"] == config_dict["model"]
        assert passed_config["dataset_path"] == config_dict["dataset_path"]
        # Defaults added
        assert "vlm" in passed_config
        assert passed_config["vlm"]["backend"] == DEFAULT_VLM_BACKEND
        assert "llm" in passed_config
        assert "judge" in passed_config
        assert passed_config["max_workers"] == DEFAULT_VLM_MAX_WORKERS


class TestRunBenchmark:
    """Tests for run_benchmark function."""

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_run_benchmark_success(
        self, mock_create_workflow: MagicMock, tmp_path: Path
    ) -> None:
        """Test successful benchmark execution."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "metrics": {
                    "functional_correctness_score": 4.5,
                    "success_rate": 90.0,
                    "exact_match_rate": 75.0,
                    "total_cases": 100,
                    "valid_cases": 95,
                    "successful_cases": 90,
                    "exact_matches": 75,
                    "failure_count": 5,
                    "score_distribution": {"5": 50, "4": 40},
                },
                "evaluation_path": str(tmp_path / "evaluation.jsonl"),
                "predictions_path": str(tmp_path / "predictions.jsonl"),
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BenchmarkInput(config=config_file)
        result = run_benchmark(params)

        # Verify
        assert result.success is True
        assert result.metrics is not None
        assert result.metrics.functional_correctness_score == 4.5
        assert result.metrics.success_rate == 90.0
        assert result.evaluation_path == tmp_path / "evaluation.jsonl"
        assert result.predictions_path == tmp_path / "predictions.jsonl"

        # Verify workflow was called correctly
        mock_create_workflow.assert_called_once()
        mock_workflow.arun.assert_called_once()

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_run_benchmark_with_overrides(
        self, mock_create_workflow: MagicMock, tmp_path: Path
    ) -> None:
        """Test benchmark execution with overrides."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")
        dataset_file = tmp_path / "dataset.jsonl"
        output_dir = tmp_path / "output"

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "metrics": {
                    "functional_correctness_score": 3.5,
                    "success_rate": 70.0,
                    "total_cases": 50,
                },
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Execute
        params = BenchmarkInput(
            config=config_file,
            dataset_override=dataset_file,
            output_dir_override=output_dir,
            resume=True,
        )
        result = run_benchmark(params)

        # Verify context passed to workflow
        call_args = mock_workflow.arun.call_args[0][0]
        assert call_args["config_path"] == str(config_file)
        assert call_args["dataset_override"] == str(dataset_file)
        assert call_args["output_dir_override"] == str(output_dir)
        assert call_args["resume"] is True
        assert result.success is True

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_run_benchmark_projects_diagnostics_but_preserves_raw_data_plane(
        self, mock_create_workflow: MagicMock, tmp_path: Path
    ) -> None:
        sentinels = {
            "config": "api_key=benchmark-config-path-713",
            "dataset": "api_key=benchmark-dataset-path-713",
            "output": "api_key=benchmark-output-path-713",
            "evaluation": "api_key=benchmark-evaluation-path-713",
            "predictions": "api_key=benchmark-predictions-path-713",
            "metric": "api_key=benchmark-metric-result-713",
        }
        config_dir = tmp_path / sentinels["config"]
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("# test config", encoding="utf-8")
        dataset_file = tmp_path / sentinels["dataset"] / "dataset.jsonl"
        output_dir = tmp_path / sentinels["output"]
        evaluation_path = tmp_path / sentinels["evaluation"] / "evaluation.jsonl"
        predictions_path = tmp_path / sentinels["predictions"] / "predictions.jsonl"
        raw_result = {
            "metrics": {
                "functional_correctness_score": 4.0,
                "success_rate": 80.0,
                "score_distribution": {sentinels["metric"]: 1},
            },
            "evaluation_path": str(evaluation_path),
            "predictions_path": str(predictions_path),
            "config_dict": {"vlm": {"api_key": sentinels["metric"]}},
            "runtime_collaborator": object(),
        }
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(return_value=raw_result)
        mock_create_workflow.return_value = mock_workflow
        listener = CollectingEventListener()

        result = run_benchmark(
            BenchmarkInput(
                config=config_file,
                dataset_override=dataset_file,
                output_dir_override=output_dir,
                event_listener=listener,
            )
        )

        assert result.success is True
        assert result.evaluation_path is None
        assert result.predictions_path is None
        assert result.raw_result is not raw_result
        assert result.raw_result is not None
        assert "config_dict" not in result.raw_result
        assert "runtime_collaborator" not in result.raw_result
        assert result.metrics is not None
        assert result.metrics.score_distribution == {}
        assert raw_result["metrics"]["score_distribution"] == {sentinels["metric"]: 1}
        assert raw_result["config_dict"]["vlm"]["api_key"] == sentinels["metric"]
        mock_workflow.arun.assert_awaited_once()
        context = mock_workflow.arun.await_args.args[0]
        assert context["config_path"] == str(config_file)
        assert context["dataset_override"] == str(dataset_file)
        assert context["output_dir_override"] == str(output_dir)

        completion = listener.get_events("workflow.completed")[0]["data"]
        assert completion["evaluation_path"] is None
        assert completion["predictions_path"] is None
        assert completion["metrics"]["score_distribution"] == {}
        observable = f"{listener.logs!r}\n{listener.events!r}\n{result!r}"
        for sentinel in sentinels.values():
            assert sentinel not in observable

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_run_benchmark_no_metrics(
        self, mock_create_workflow: MagicMock, tmp_path: Path
    ) -> None:
        """Test benchmark execution when workflow returns no metrics."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that returns no metrics
        mock_workflow = Mock()
        sentinel = "api_key=benchmark-result-error-713"
        mock_workflow.arun = AsyncMock(
            return_value={
                "error": sentinel,
                "config_dict": {"vlm": {"api_key": sentinel}},
                "runtime_collaborator": object(),
            }
        )
        mock_create_workflow.return_value = mock_workflow
        listener = CollectingEventListener()

        # Execute
        params = BenchmarkInput(config=config_file, event_listener=listener)
        result = run_benchmark(params)

        # Verify
        assert result.success is False
        assert result.error == "Benchmark failed"
        assert result.raw_result == {"error": "Benchmark failed"}
        assert sentinel not in repr(listener.logs)
        assert sentinel not in repr(listener.events)
        assert sentinel not in repr(result)

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_run_benchmark_workflow_exception(
        self, mock_create_workflow: MagicMock, tmp_path: Path
    ) -> None:
        """Test benchmark execution when workflow raises exception."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock workflow that raises exception
        mock_workflow = Mock()
        sentinel = "api_key=reflected-benchmark-exception-713"
        mock_workflow.arun = AsyncMock(side_effect=RuntimeError(sentinel))
        mock_create_workflow.return_value = mock_workflow
        listener = CollectingEventListener()

        # Execute
        params = BenchmarkInput(config=config_file, event_listener=listener)
        result = run_benchmark(params)

        # Verify
        assert result.success is False
        assert result.error == "Benchmark failed"
        assert sentinel not in repr(listener.logs)
        assert sentinel not in repr(listener.events)


class TestBenchmarkOutput:
    """Tests for BenchmarkOutput dataclass."""

    def test_benchmark_output_success(self, tmp_path: Path) -> None:
        """Test creating successful BenchmarkOutput."""
        from material_agent.api.types import MetricsResult

        metrics = MetricsResult(
            functional_correctness_score=4.5,
            success_rate=90.0,
        )

        output = BenchmarkOutput(
            success=True,
            metrics=metrics,
            evaluation_path=tmp_path / "eval.jsonl",
            predictions_path=tmp_path / "pred.jsonl",
        )

        assert output.success is True
        assert output.metrics == metrics
        assert output.evaluation_path == tmp_path / "eval.jsonl"
        assert output.error is None

    def test_benchmark_output_error(self) -> None:
        """Test creating error BenchmarkOutput."""
        output = BenchmarkOutput(
            success=False,
            error="Test error",
        )

        assert output.success is False
        assert output.error == "Test error"
        assert output.metrics is None
