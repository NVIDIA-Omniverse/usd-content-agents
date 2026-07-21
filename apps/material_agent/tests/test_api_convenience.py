# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent API convenience functions."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from material_agent.api import (
    apply,
    benchmark,
    configure,
    evaluate,
    pipeline,
    predict,
    refine,
)


class TestConvenienceFunctions:
    """Tests for convenience wrapper functions."""

    @patch("material_agent.api.benchmark.arun_benchmark", new_callable=AsyncMock)
    def test_benchmark_convenience(self, mock_arun, tmp_path):
        """Test benchmark convenience function."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test")

        from material_agent.api.benchmark import BenchmarkOutput
        from material_agent.api.types import MetricsResult

        mock_arun.return_value = BenchmarkOutput(
            success=True,
            metrics=MetricsResult(functional_correctness_score=4.0),
        )

        # Test minimal usage
        result = benchmark(config_file)
        assert result.success is True
        mock_arun.assert_called_once()

        # Verify BenchmarkInput was created correctly
        call_args = mock_arun.call_args[0][0]
        assert call_args.config == config_file

    @patch("material_agent.api.benchmark.arun_benchmark", new_callable=AsyncMock)
    def test_benchmark_convenience_with_kwargs(self, mock_arun, tmp_path):
        """Test benchmark convenience function with kwargs."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test")

        from material_agent.api.benchmark import BenchmarkOutput

        mock_arun.return_value = BenchmarkOutput(success=True)

        # Test with optional parameters
        result = benchmark(config_file, verbose=True, resume=True)
        assert result.success is True

        # Verify kwargs were passed
        call_args = mock_arun.call_args[0][0]
        assert call_args.verbose is True
        assert call_args.resume is True

    @patch("material_agent.api.predict.arun_predict", new_callable=AsyncMock)
    def test_predict_convenience(self, mock_arun, tmp_path):
        """Test predict convenience function."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test")

        from material_agent.api.predict import PredictOutput

        mock_arun.return_value = PredictOutput(success=True)

        result = predict(config_file)
        assert result.success is True

    @patch("material_agent.api.evaluate.arun_evaluate", new_callable=AsyncMock)
    def test_evaluate_convenience(self, mock_arun, tmp_path):
        """Test evaluate convenience function."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test")

        from material_agent.api.evaluate import EvaluateOutput

        mock_arun.return_value = EvaluateOutput(success=True)

        result = evaluate(config_file)
        assert result.success is True

    @patch("material_agent.api.apply.arun_apply", new_callable=AsyncMock)
    def test_apply_convenience(self, mock_arun, tmp_path):
        """Test apply convenience function."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test")

        from material_agent.api.apply import ApplyOutput

        mock_arun.return_value = ApplyOutput(success=True)

        result = apply(config_file, render_enabled=True)
        assert result.success is True

        # Verify kwargs passed
        call_args = mock_arun.call_args[0][0]
        assert call_args.render_enabled is True

    @patch("material_agent.api.pipeline.arun_pipeline", new_callable=AsyncMock)
    def test_pipeline_convenience(self, mock_arun, tmp_path):
        """Test pipeline convenience function."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test")

        from material_agent.api.pipeline import PipelineOutput

        mock_arun.return_value = PipelineOutput(success=True)

        result = pipeline(config_file, only_steps=["predict"])
        assert result.success is True

        # Verify kwargs passed
        call_args = mock_arun.call_args[0][0]
        assert call_args.only_steps == ["predict"]

    @patch("material_agent.api.refine.arun_refine", new_callable=AsyncMock)
    def test_refine_convenience(self, mock_arun, tmp_path):
        """Test refine convenience function."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test")

        from material_agent.api.refine import RefineOutput

        mock_arun.return_value = RefineOutput(success=True)

        result = refine(config_file, max_iterations_override=5)
        assert result.success is True

        # Verify kwargs passed
        call_args = mock_arun.call_args[0][0]
        assert call_args.max_iterations_override == 5

    @patch("material_agent.api.configure.arun_configure", new_callable=AsyncMock)
    def test_configure_convenience(self, mock_arun, tmp_path):
        """Test configure convenience function."""
        output_file = tmp_path / "new_config.yaml"

        from material_agent.api.configure import ConfigureOutput

        mock_arun.return_value = ConfigureOutput(success=True)

        result = configure(output_file, force=True)
        assert result.success is True

        # Verify kwargs passed
        call_args = mock_arun.call_args[0][0]
        assert call_args.force is True

    def test_convenience_with_dict_config(self):
        """Test convenience functions with dict config."""
        config_dict = {
            "model": {"service": "azure", "name": "example-vlm-model"},
            "dataset_path": "data.jsonl",
        }

        # Should accept dict configs
        from material_agent.api import BenchmarkInput

        with patch(
            "material_agent.api.benchmark.arun_benchmark", new_callable=AsyncMock
        ) as mock_arun:
            from material_agent.api.benchmark import BenchmarkOutput

            mock_arun.return_value = BenchmarkOutput(success=True)

            result = benchmark(config_dict)
            assert result.success is True

            # Verify dict config was used
            call_args = mock_arun.call_args[0][0]
            assert call_args.config == config_dict
