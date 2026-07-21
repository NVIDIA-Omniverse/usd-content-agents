# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for auto-wiring logic in UnifiedPipelineExecutorTask._execute_step.

Verifies that pipeline_state["step_outputs"] from earlier steps are correctly
injected into downstream step_config dicts before workflow dispatch.

Pattern: import _execute_step indirectly by instantiating the executor and calling
_execute_step with a mocked workflow factory so the actual workflow never runs.
"""

import copy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from material_agent.tasks.unified_pipeline_executor import UnifiedPipelineExecutorTask


def _make_executor() -> UnifiedPipelineExecutorTask:
    return UnifiedPipelineExecutorTask()


def _base_context(tmp_path: Path) -> dict[str, Any]:
    return {
        "working_dir": str(tmp_path),
        "config_path": str(tmp_path / "pipeline.yaml"),
        "steps_to_run": [],
        "step_configs": {},
    }


def _make_mock_workflow(outputs: dict[str, Any] | None = None):
    """Return a mock workflow whose .run() returns *outputs*."""
    wf = MagicMock()
    wf.run.return_value = outputs or {"status": "ok"}
    return wf


def _call_execute_step(
    executor: UnifiedPipelineExecutorTask,
    step_name: str,
    step_config: dict[str, Any],
    context: dict[str, Any],
    pipeline_state: dict[str, Any],
    workflow_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call _execute_step with a patched workflow factory so no real work runs.

    Returns the step_config dict (mutated in-place by auto-wiring).
    """
    mock_wf = _make_mock_workflow(workflow_outputs)

    with (
        patch(
            "material_agent.workflows.create_prediction_workflow_from_config",
            return_value=mock_wf,
        ),
        patch(
            "material_agent.workflows.create_apply_workflow_from_config",
            return_value=mock_wf,
        ),
        patch(
            "material_agent.workflows.create_usd_data_preparation_workflow_from_config",
            return_value=mock_wf,
        ),
        patch(
            "material_agent.workflows.create_cluster_prims_workflow_from_config",
            return_value=mock_wf,
        ),
        patch(
            "material_agent.workflows.create_expand_cluster_predictions_workflow_from_config",
            return_value=mock_wf,
        ),
        patch(
            "material_agent.workflows.create_restore_usd_workflow_from_config",
            return_value=mock_wf,
        ),
        patch(
            "material_agent.workflows.create_render_workflow_from_config",
            return_value=mock_wf,
        ),
        patch(
            "material_agent.workflows.create_validate_predictions_workflow_from_config",
            return_value=mock_wf,
        ),
        patch(
            "material_agent.workflows.create_harmonize_predictions_workflow_from_config",
            return_value=mock_wf,
        ),
        patch(
            "material_agent.workflows.create_evaluation_workflow_from_config",
            return_value=mock_wf,
        ),
        patch(
            "material_agent.workflows.create_optimize_usd_workflow_from_config",
            return_value=mock_wf,
        ),
        patch(
            "material_agent.workflows.create_iterative_apply_workflow_from_config",
            return_value=mock_wf,
        ),
    ):
        config_copy = copy.deepcopy(step_config)
        executor._execute_step(step_name, config_copy, context, None, pipeline_state)
        return config_copy


# ---------------------------------------------------------------------------
# cluster_prims → predict chain
# ---------------------------------------------------------------------------


class TestClusterPrimsToPredict:
    """When cluster_prims produced a representative dataset, predict should use it."""

    def test_predict_gets_representative_dataset(self, tmp_path: Path) -> None:
        executor = _make_executor()
        ctx = _base_context(tmp_path)

        pipeline_state = {
            "step_outputs": {
                "cluster_prims": {
                    "dataset_representatives_path": "/work/clusters/reps.jsonl",
                    "cluster_map_path": "/work/clusters/cluster_map.jsonl",
                    "cluster_prims_ran": True,
                },
            },
            "completed_steps": ["cluster_prims"],
            "failed_steps": [],
            "current_step": "predict",
        }

        cfg = _call_execute_step(
            executor,
            "predict",
            {"enabled": True},
            ctx,
            pipeline_state,
            workflow_outputs={
                "predictions_path": "/work/predictions/predictions.jsonl"
            },
        )

        assert cfg["dataset"] == "/work/clusters/reps.jsonl"


# ---------------------------------------------------------------------------
# cluster_prims → expand_cluster_predictions chain
# ---------------------------------------------------------------------------


class TestClusterPrimsToExpand:
    """expand_cluster_predictions gets predictions_path, cluster_map_path, cluster_prims_ran."""

    def test_expand_gets_required_wiring(self, tmp_path: Path) -> None:
        executor = _make_executor()
        ctx = _base_context(tmp_path)

        pipeline_state = {
            "step_outputs": {
                "cluster_prims": {
                    "dataset_representatives_path": "/work/clusters/reps.jsonl",
                    "cluster_map_path": "/work/clusters/cluster_map.jsonl",
                    "cluster_prims_ran": True,
                },
                "predict": {
                    "predictions_path": "/work/predictions/predictions.jsonl",
                },
            },
            "completed_steps": ["cluster_prims", "predict"],
            "failed_steps": [],
            "current_step": "expand_cluster_predictions",
        }

        cfg = _call_execute_step(
            executor,
            "expand_cluster_predictions",
            {"enabled": True},
            ctx,
            pipeline_state,
            workflow_outputs={"predictions_path": "/work/predictions/expanded.jsonl"},
        )

        assert cfg["predictions_path"] == "/work/predictions/predictions.jsonl"
        assert cfg["cluster_map_path"] == "/work/clusters/cluster_map.jsonl"
        assert cfg["cluster_prims_ran"] is True


# ---------------------------------------------------------------------------
# optimize_usd → downstream steps
# ---------------------------------------------------------------------------


class TestOptimizeUsdAutoWiring:
    """When optimize_usd produced optimized_usd_path, downstream steps use it."""

    def test_build_dataset_usd_gets_optimized_path(self, tmp_path: Path) -> None:
        executor = _make_executor()
        ctx = _base_context(tmp_path)

        pipeline_state = {
            "step_outputs": {
                "optimize_usd": {
                    "optimized_usd_path": "/work/optimized/scene.usdc",
                    "original_usd_path": "/input/scene.usd",
                    "original_prim_count": 1000,
                },
            },
            "completed_steps": ["optimize_usd"],
            "failed_steps": [],
            "current_step": "build_dataset_usd",
        }

        cfg = _call_execute_step(
            executor,
            "build_dataset_usd",
            {"enabled": True},
            ctx,
            pipeline_state,
            workflow_outputs={"num_prims": 500, "num_images": 1000},
        )

        assert cfg["usd_path"] == "/work/optimized/scene.usdc"

    def test_apply_gets_optimized_path_without_restore(self, tmp_path: Path) -> None:
        executor = _make_executor()
        ctx = _base_context(tmp_path)

        pipeline_state = {
            "step_outputs": {
                "optimize_usd": {
                    "optimized_usd_path": "/work/optimized/scene.usdc",
                    "original_usd_path": "/input/scene.usd",
                },
            },
            "completed_steps": ["optimize_usd"],
            "failed_steps": [],
            "current_step": "apply",
        }

        cfg = _call_execute_step(
            executor,
            "apply",
            {"enabled": True},
            ctx,
            pipeline_state,
            workflow_outputs={"output_usd_path": "/work/output/output.usd"},
        )

        assert cfg["input_usd_path"] == "/work/optimized/scene.usdc"


# ---------------------------------------------------------------------------
# restore_usd → apply chain
# ---------------------------------------------------------------------------


class TestRestoreUsdToApply:
    """When restore_usd ran, apply gets original USD + restored predictions."""

    def test_apply_gets_original_usd_and_restored_predictions(
        self, tmp_path: Path
    ) -> None:
        executor = _make_executor()
        ctx = _base_context(tmp_path)

        pipeline_state = {
            "step_outputs": {
                "optimize_usd": {
                    "optimized_usd_path": "/work/optimized/scene.usdc",
                    "original_usd_path": "/input/scene.usd",
                },
                "restore_usd": {
                    "restored_predictions_path": "/work/restored/restored_predictions.jsonl",
                },
            },
            "completed_steps": ["optimize_usd", "restore_usd"],
            "failed_steps": [],
            "current_step": "apply",
        }

        cfg = _call_execute_step(
            executor,
            "apply",
            {"enabled": True},
            ctx,
            pipeline_state,
            workflow_outputs={"output_usd_path": "/work/output/output.usd"},
        )

        assert cfg["input_usd_path"] == "/input/scene.usd"
        assert cfg["predictions_path"] == "/work/restored/restored_predictions.jsonl"

    def test_render_uses_apply_output_when_restore_only_has_predictions(
        self, tmp_path: Path
    ) -> None:
        executor = _make_executor()
        ctx = _base_context(tmp_path)

        pipeline_state = {
            "step_outputs": {
                "restore_usd": {
                    "restored_predictions_path": "/work/restored/restored_predictions.jsonl",
                },
                "apply": {
                    "output_usd_path": "/work/output/output.usd",
                },
            },
            "completed_steps": ["restore_usd", "apply"],
            "failed_steps": [],
            "current_step": "render",
        }

        cfg = _call_execute_step(
            executor,
            "render",
            {"enabled": True},
            ctx,
            pipeline_state,
            workflow_outputs={"rendered_image_paths": ["/work/renders/final.png"]},
        )

        assert cfg["input_usd_path"] == "/work/output/output.usd"

    def test_render_prefers_restore_usd_path_when_present(self, tmp_path: Path) -> None:
        executor = _make_executor()
        ctx = _base_context(tmp_path)

        pipeline_state = {
            "step_outputs": {
                "restore_usd": {
                    "restored_usd_path": "/work/restored/restored.usd",
                    "restored_predictions_path": "/work/restored/restored_predictions.jsonl",
                },
                "apply": {
                    "output_usd_path": "/work/output/output.usd",
                },
            },
            "completed_steps": ["restore_usd", "apply"],
            "failed_steps": [],
            "current_step": "render",
        }

        cfg = _call_execute_step(
            executor,
            "render",
            {"enabled": True},
            ctx,
            pipeline_state,
            workflow_outputs={"rendered_image_paths": ["/work/renders/final.png"]},
        )

        assert cfg["input_usd_path"] == "/work/restored/restored.usd"


# ---------------------------------------------------------------------------
# restore_usd skip when optimize_usd didn't run
# ---------------------------------------------------------------------------


class TestRestoreUsdSkip:
    """restore_usd is skipped at the run() loop level when optimize_usd didn't run."""

    def test_restore_usd_skipped_without_optimize(self, tmp_path: Path) -> None:
        """Verify the skip logic in run() — restore_usd not in completed_steps."""
        executor = _make_executor()
        ctx = _base_context(tmp_path)
        ctx["steps_to_run"] = ["restore_usd"]
        ctx["step_configs"] = {"restore_usd": {"enabled": True}}

        # Mock run to check that restore_usd is skipped
        pipeline_state = {
            "step_outputs": {},  # optimize_usd did NOT run
            "completed_steps": [],
            "failed_steps": [],
            "current_step": None,
            "session_id": None,
            "project_name": None,
        }

        # Patch _load_pipeline_state so run() uses our empty state
        with patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ):
            executor.run(ctx)

        # restore_usd should NOT be in completed_steps (it was skipped)
        assert "restore_usd" not in pipeline_state["completed_steps"]


# ---------------------------------------------------------------------------
# validate_predictions / harmonize_predictions auto-wiring
# ---------------------------------------------------------------------------


class TestValidateAndHarmonizeAutoWiring:
    """validate_predictions and harmonize_predictions get predictions_path."""

    def test_harmonize_gets_predictions_from_predict(self, tmp_path: Path) -> None:
        executor = _make_executor()
        ctx = _base_context(tmp_path)

        pipeline_state = {
            "step_outputs": {
                "predict": {
                    "predictions_path": "/work/predictions/predictions.jsonl",
                },
            },
            "completed_steps": ["predict"],
            "failed_steps": [],
            "current_step": "harmonize_predictions",
        }

        cfg = _call_execute_step(
            executor,
            "harmonize_predictions",
            {"enabled": True},
            ctx,
            pipeline_state,
            workflow_outputs={"predictions_path": "/work/predictions/harmonized.jsonl"},
        )

        assert cfg["predictions_path"] == "/work/predictions/predictions.jsonl"

    def test_validate_prefers_harmonized_output(self, tmp_path: Path) -> None:
        executor = _make_executor()
        ctx = _base_context(tmp_path)

        pipeline_state = {
            "step_outputs": {
                "predict": {
                    "predictions_path": "/work/predictions/predictions.jsonl",
                },
                "harmonize_predictions": {
                    "predictions_path": "/work/predictions/harmonized.jsonl",
                },
            },
            "completed_steps": ["predict", "harmonize_predictions"],
            "failed_steps": [],
            "current_step": "validate_predictions",
        }

        cfg = _call_execute_step(
            executor,
            "validate_predictions",
            {"enabled": True},
            ctx,
            pipeline_state,
            workflow_outputs={"predictions_path": "/work/predictions/validated.jsonl"},
        )

        assert cfg["predictions_path"] == "/work/predictions/harmonized.jsonl"

    def test_restore_prefers_harmonized_over_raw(self, tmp_path: Path) -> None:
        """restore_usd should prefer harmonized > validated > raw predictions."""
        executor = _make_executor()
        ctx = _base_context(tmp_path)

        # Create optimization metadata file so restore_usd doesn't warn
        opt_dir = tmp_path / "optimized"
        opt_dir.mkdir()
        meta_path = opt_dir / "optimized_input.metadata.json"
        meta_path.write_text('{"prim_map": {}}')

        pipeline_state = {
            "step_outputs": {
                "optimize_usd": {
                    "optimized_usd_path": str(tmp_path / "optimized/scene.usdc"),
                    "original_usd_path": "/input/scene.usd",
                },
                "predict": {
                    "predictions_path": "/work/predictions/predictions.jsonl",
                },
                "harmonize_predictions": {
                    "predictions_path": "/work/predictions/harmonized.jsonl",
                },
            },
            "completed_steps": ["optimize_usd", "predict", "harmonize_predictions"],
            "failed_steps": [],
            "current_step": "restore_usd",
        }

        cfg = _call_execute_step(
            executor,
            "restore_usd",
            {"enabled": True},
            ctx,
            pipeline_state,
            workflow_outputs={
                "restored_predictions_path": "/work/restored/restored.jsonl",
            },
        )

        assert cfg["predictions_path"] == "/work/predictions/harmonized.jsonl"
