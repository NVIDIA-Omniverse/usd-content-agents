# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Predict API."""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from material_agent.api.predict import PredictInput, PredictOutput, run_predict


class TestPredictInput:
    """Tests for PredictInput validation."""

    def test_predict_input_valid(self, tmp_path):
        """Test creating valid PredictInput."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        params = PredictInput(config=config_file, resume=True, verbose=True)

        assert params.config == config_file
        assert params.resume is True
        assert params.verbose is True

    def test_predict_input_missing_config(self, tmp_path):
        """Test PredictInput raises error for missing config."""
        sentinel = "api_key=missing-predict-config-713"
        config_file = tmp_path / sentinel / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="^Config file not found$") as exc:
            PredictInput(config=config_file)

        assert sentinel not in str(exc.value)

    def test_predict_input_projects_real_name_too_long_error(self):
        """Config inspection errors never retain the credential-bearing path."""
        sentinel = "api_key=predict-name-too-long-713"
        config_file = Path("/tmp") / (sentinel + "x" * 5000)

        with pytest.raises(OSError, match="^Unable to inspect config file$") as exc:
            PredictInput(config=config_file)

        assert sentinel not in repr(exc.value)
        assert exc.value.__cause__ is None
        assert exc.value.__context__ is None


class TestRunPredict:
    """Tests for run_predict function."""

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_run_predict_success(self, mock_arun_pipeline, tmp_path):
        """Test successful predict execution."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock pipeline result
        from material_agent.api.pipeline import PipelineOutput

        mock_arun_pipeline.return_value = PipelineOutput(
            success=True,
            step_results={
                "predict": {
                    "predictions_path": str(tmp_path / "predictions.jsonl"),
                    "report_path": str(tmp_path / "report.html"),
                    "num_predictions": 25,
                }
            },
        )

        # Execute
        params = PredictInput(config=config_file)
        result = run_predict(params)

        # Verify
        assert result.success is True
        assert result.predictions_path == tmp_path / "predictions.jsonl"
        assert result.report_path == tmp_path / "report.html"
        assert result.num_predictions == 25

        # Verify pipeline was called with only=predict
        call_args = mock_arun_pipeline.call_args[0][0]
        assert call_args.only_steps == ["predict"]

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_success_result_projects_metadata_without_mutating_runtime(
        self, mock_arun_pipeline, tmp_path
    ):
        sentinel = "material-predict-result-credential-713"
        config = {"vlm": {"api_key": sentinel}}
        predict_result = {
            "predictions_path": str(tmp_path / "predictions.jsonl"),
            "report_path": str(tmp_path / "report.html"),
            "num_predictions": 1,
            "api_key": sentinel,
            "nested": {
                "config_dict": {"api_key": sentinel},
                "listener": object(),
            },
        }

        from material_agent.api.pipeline import PipelineOutput

        pipeline_output = PipelineOutput(
            success=True,
            step_results={"predict": predict_result},
        )
        mock_arun_pipeline.return_value = pipeline_output

        result = run_predict(PredictInput(config=config))

        pipeline_params = mock_arun_pipeline.await_args.args[0]
        assert pipeline_params.config["vlm"]["api_key"] == sentinel
        assert config["vlm"]["api_key"] == sentinel
        assert predict_result["api_key"] == sentinel
        assert result.success is True
        assert result.predictions_path == tmp_path / "predictions.jsonl"
        assert result.report_path == tmp_path / "report.html"
        assert result.num_predictions == 1
        assert result.raw_result is not predict_result
        assert result.raw_result is not None
        assert "config_dict" not in result.raw_result["nested"]
        assert "listener" not in result.raw_result["nested"]
        assert sentinel not in repr(result)

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_projected_prediction_count_remains_an_integer(self, mock_arun_pipeline):
        sentinel = "api_key=material-predict-count-713"

        from material_agent.api.pipeline import PipelineOutput

        mock_arun_pipeline.return_value = PipelineOutput(
            success=True,
            step_results={"predict": {"num_predictions": sentinel}},
        )

        result = run_predict(PredictInput(config={"predict": {}}))

        assert result.success is True
        assert result.num_predictions == 0
        assert type(result.num_predictions) is int
        assert sentinel not in repr(result)

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_run_predict_pipeline_failure(self, mock_arun_pipeline, tmp_path):
        """Test predict when pipeline fails."""
        # Setup
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test config")

        # Mock pipeline failure
        from material_agent.api.pipeline import PipelineOutput

        mock_arun_pipeline.return_value = PipelineOutput(
            success=False,
            error="api_key=reflected-pipeline-result-713",
        )

        # Execute
        params = PredictInput(config=config_file)
        result = run_predict(params)

        # Verify
        assert result.success is False
        assert result.error == "Predict failed"

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_run_predict_exception_diagnostics_are_secret_safe(
        self, mock_arun_pipeline, tmp_path, caplog
    ):
        """Raw config reaches the pipeline without entering failure diagnostics."""
        # Setup
        path_sentinel = "api_key=predict-config-path-713"
        exception_sentinel = "api_key=reflected-predict-exception-713"
        config_dir = tmp_path / path_sentinel
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("# test config")

        # Mock exception
        mock_arun_pipeline.side_effect = RuntimeError(exception_sentinel)

        # Execute
        params = PredictInput(config=config_file)
        with caplog.at_level(logging.INFO):
            result = run_predict(params)

        # Verify
        assert result.success is False
        assert result.error == "Predict failed"
        pipeline_params = mock_arun_pipeline.await_args.args[0]
        assert pipeline_params.config == config_file
        observable = f"{caplog.text}\n{result.error}"
        assert path_sentinel not in observable
        assert exception_sentinel not in observable
        assert all(
            record.exc_info is None
            for record in caplog.records
            if record.name == "material_agent.api.predict"
        )


class TestPredictOutput:
    """Tests for PredictOutput dataclass."""

    def test_predict_output_success(self, tmp_path):
        """Test creating successful PredictOutput."""
        output = PredictOutput(
            success=True,
            predictions_path=tmp_path / "predictions.jsonl",
            report_path=tmp_path / "report.html",
            num_predictions=50,
        )

        assert output.success is True
        assert output.predictions_path == tmp_path / "predictions.jsonl"
        assert output.num_predictions == 50

    def test_predict_output_error(self):
        """Test creating error PredictOutput."""
        output = PredictOutput(
            success=False,
            error="Test error",
        )

        assert output.success is False
        assert output.error == "Test error"
        assert output.predictions_path is None
