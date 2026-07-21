# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent API defaults system."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from material_agent.api.defaults import (
    BENCHMARK_DEFAULTS,
    DEFAULT_JUDGE_BACKEND,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_MODEL,
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_MAX_WORKERS,
    DEFAULT_VLM_MODEL,
    PREDICT_DEFAULTS,
    apply_defaults,
    get_apply_config_with_defaults,
    get_benchmark_config_with_defaults,
    get_minimal_required_fields,
    get_predict_config_with_defaults,
)


class TestApplyDefaults:
    """Tests for apply_defaults function."""

    def test_apply_defaults_simple(self):
        """Test applying defaults to simple dict."""
        user_config = {"dataset": "data.jsonl"}
        defaults = {"dataset": "default.jsonl", "max_workers": 16}

        result = apply_defaults(user_config, defaults)

        # User value preserved
        assert result["dataset"] == "data.jsonl"
        # Default added
        assert result["max_workers"] == 16

    def test_apply_defaults_nested(self):
        """Test applying defaults to nested dict."""
        user_config = {"vlm": {"model": "example-vlm-model"}}
        defaults = {
            "vlm": {"model": "default", "backend": "nim"},
            "max_workers": 16,
        }

        result = apply_defaults(user_config, defaults)

        # User nested value preserved
        assert result["vlm"]["model"] == "example-vlm-model"
        # Default nested value added
        assert result["vlm"]["backend"] == "nim"
        # Default top-level added
        assert result["max_workers"] == 16

    def test_apply_defaults_no_override(self):
        """Test that user values are never overridden."""
        user_config = {
            "vlm": {"backend": "custom", "model": "custom-model"},
            "max_workers": 32,
        }
        defaults = {
            "vlm": {"backend": "default", "model": "default"},
            "max_workers": 16,
        }

        result = apply_defaults(user_config, defaults)

        # All user values preserved
        assert result["vlm"]["backend"] == "custom"
        assert result["vlm"]["model"] == "custom-model"
        assert result["max_workers"] == 32


class TestPredictDefaults:
    """Tests for predict config defaults."""

    def test_predict_defaults_values(self):
        """Test predict defaults contain expected values."""
        assert "vlm" in PREDICT_DEFAULTS
        assert "llm" in PREDICT_DEFAULTS
        assert PREDICT_DEFAULTS["vlm"]["backend"] == DEFAULT_VLM_BACKEND
        assert PREDICT_DEFAULTS["vlm"]["model"] == DEFAULT_VLM_MODEL

    def test_get_predict_config_with_defaults_minimal(self):
        """Test minimal predict config gets defaults."""
        minimal = {"dataset": "data.jsonl"}

        full = get_predict_config_with_defaults(minimal)

        # User value preserved
        assert full["dataset"] == "data.jsonl"
        # Defaults added
        assert "vlm" in full
        assert full["vlm"]["backend"] == DEFAULT_VLM_BACKEND
        assert full["vlm"]["model"] == DEFAULT_VLM_MODEL
        assert full["max_workers"] == DEFAULT_VLM_MAX_WORKERS

    def test_get_predict_config_with_partial_vlm(self):
        """Test predict config with partial VLM gets defaults."""
        partial = {"dataset": "data.jsonl", "vlm": {"model": "custom-model"}}

        full = get_predict_config_with_defaults(partial)

        # User VLM model preserved
        assert full["vlm"]["model"] == "custom-model"
        # VLM backend default added
        assert full["vlm"]["backend"] == DEFAULT_VLM_BACKEND


class TestBenchmarkDefaults:
    """Tests for benchmark config defaults."""

    def test_benchmark_defaults_values(self):
        """Test benchmark defaults contain expected values."""
        assert "vlm" in BENCHMARK_DEFAULTS
        assert "llm" in BENCHMARK_DEFAULTS
        assert "judge" in BENCHMARK_DEFAULTS
        assert BENCHMARK_DEFAULTS["allow_empty_predictions"] is False

    def test_get_benchmark_config_with_defaults_minimal(self):
        """Test minimal benchmark config gets defaults."""
        minimal = {"dataset": "data.jsonl"}

        full = get_benchmark_config_with_defaults(minimal)

        # User value preserved
        assert full["dataset"] == "data.jsonl"
        # All model defaults added
        assert full["vlm"]["backend"] == DEFAULT_VLM_BACKEND
        assert full["vlm"]["model"] == DEFAULT_VLM_MODEL
        assert full["llm"]["backend"] == DEFAULT_LLM_BACKEND
        assert full["llm"]["model"] == DEFAULT_LLM_MODEL
        assert full["judge"]["backend"] == DEFAULT_JUDGE_BACKEND
        assert full["judge"]["model"] == DEFAULT_JUDGE_MODEL
        assert full["allow_empty_predictions"] is False


class TestApplyConfigDefaults:
    """Tests for apply config defaults."""

    def test_get_apply_config_with_defaults_preserves_user_values(self):
        minimal = {"input_usd": "scene.usd", "output_usd": "scene_out.usd"}

        full = get_apply_config_with_defaults(minimal)

        assert full["input_usd"] == "scene.usd"
        assert full["output_usd"] == "scene_out.usd"
        assert full["layer_only"] is False
        assert full["flatten"] is True
        assert full["render"]["enabled"] is False


class TestMinimalRequiredFields:
    """Tests for minimal required fields."""

    def test_get_minimal_required_fields(self):
        """Test getting minimal required fields."""
        fields = get_minimal_required_fields()

        # Predict should only require dataset
        assert "predict" in fields
        assert "dataset" in fields["predict"]

        # Benchmark should only require dataset
        assert "benchmark" in fields
        assert "dataset" in fields["benchmark"]


class TestPredictAPIWithDefaults:
    """Test predict API with defaults applied."""

    @patch(
        "material_agent.api.pipeline.run_pipeline".replace(
            "run_pipeline", "arun_pipeline"
        ),
        new_callable=AsyncMock,
    )
    def test_predict_with_minimal_dict_config(self, mock_arun_pipeline):
        """Test that minimal dict config works with defaults."""
        from material_agent.api import predict
        from material_agent.api.pipeline import PipelineOutput

        # Mock pipeline success
        mock_arun_pipeline.return_value = PipelineOutput(
            success=True,
            step_results={"predict": {"predictions_path": "output/pred.jsonl"}},
        )

        # Minimal config - only dataset!
        minimal_config = {"dataset": "data/test.jsonl"}

        result = predict(minimal_config)

        # Should succeed with defaults applied
        assert result.success is True

        # Verify pipeline was called with config that has defaults
        call_args = mock_arun_pipeline.call_args[0][0]
        passed_config = call_args.config

        # Check defaults were applied
        assert "vlm" in passed_config
        assert passed_config["vlm"]["backend"] == DEFAULT_VLM_BACKEND
        assert passed_config["vlm"]["model"] == DEFAULT_VLM_MODEL


class TestBenchmarkAPIWithDefaults:
    """Test benchmark API with defaults applied."""

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_benchmark_with_minimal_dict_config(self, mock_create_workflow):
        """Test that minimal dict config works with defaults."""
        from material_agent.api import benchmark

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "metrics": {
                    "functional_correctness_score": 4.0,
                    "success_rate": 80.0,
                    "total_cases": 10,
                }
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Minimal config - only dataset!
        minimal_config = {"dataset": "data/test.jsonl"}

        result = benchmark(minimal_config)

        # Should succeed with defaults applied
        assert result.success is True

        # Verify workflow received config with defaults
        call_args = mock_workflow.arun.call_args[0][0]
        passed_config = call_args["config_dict"]

        # Check defaults were applied
        assert "vlm" in passed_config
        assert passed_config["vlm"]["backend"] == DEFAULT_VLM_BACKEND
        assert "llm" in passed_config
        assert "judge" in passed_config
