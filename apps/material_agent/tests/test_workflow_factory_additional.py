# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional tests for material_agent.workflows.factory."""

from __future__ import annotations

from unittest.mock import Mock

from world_understanding.agentic.tasks import ToolTask
from world_understanding.agentic.workflows import Workflow
from world_understanding.utils.object_store import (
    InMemoryObjectStore,
    TempDirObjectStore,
)

import material_agent.workflows.factory as factory


def test_creates_validation_and_apply_family_workflows() -> None:
    optimize = factory.create_optimize_usd_workflow_from_config()
    validate_predictions = factory.create_validate_predictions_workflow_from_config()
    harmonize = factory.create_harmonize_predictions_workflow_from_config()
    restore = factory.create_restore_usd_workflow_from_config()
    prepare_dataset = factory.create_prepare_dataset_workflow_from_config()
    apply = factory.create_apply_workflow_from_config()
    render = factory.create_render_workflow_from_config()
    configure = factory.create_configure_workflow()
    cluster = factory.create_cluster_prims_workflow_from_config()
    expand = factory.create_expand_cluster_predictions_workflow_from_config()

    assert isinstance(optimize, Workflow)
    assert optimize.name == "Config-Driven USD Optimization"
    assert [type(task).__name__ for task in optimize.tasks] == [
        "OptimizeUSDConfigTask",
        "OptimizeUSDTask",
    ]

    assert [type(task).__name__ for task in validate_predictions.tasks] == [
        "ValidatePredictionsConfigTask",
        "ValidatePredictionsTask",
    ]
    assert [type(task).__name__ for task in harmonize.tasks] == [
        "ValidatePredictionsConfigTask",
        "HarmonizePredictionsTask",
    ]
    assert [type(task).__name__ for task in restore.tasks] == [
        "RestoreUSDConfigTask",
        "RestoreUSDTask",
    ]
    assert [task.name for task in prepare_dataset.tasks] == [
        "PrepareDatasetConfigLoading",
        "ModelProvisioning",
        "PrepareDataset",
    ]
    assert [task.name for task in apply.tasks] == [
        "ApplyConfigLoading",
        "IdentifyUniqueMaterials",
        "MaterialRetrieval",
        "ResolveMaterialFiles",
        "ApplyMaterialsToUSD",
        "Render",
        "ApplyCompletion",
    ]
    assert [task.name for task in render.tasks] == [
        "RenderConfig",
        "Render",
    ]
    assert [task.name for task in configure.tasks] == ["GenerateConfig"]
    assert [type(task).__name__ for task in cluster.tasks] == [
        "ClusterPrimsConfigTask",
        "ClusterPrimsTask",
    ]
    assert [type(task).__name__ for task in expand.tasks] == [
        "ExpandClusterPredictionsConfigTask",
        "ExpandClusterPredictionsTask",
    ]

    assert isinstance(validate_predictions.object_store, InMemoryObjectStore)
    assert isinstance(harmonize.object_store, InMemoryObjectStore)
    assert isinstance(apply.object_store, InMemoryObjectStore)


def test_creates_scene_preview_and_reference_generation_workflows() -> None:
    generate_reference = factory.create_generate_reference_image_workflow_from_config()
    generate_material_library = (
        factory.create_generate_material_library_workflow_from_config()
    )
    render_preview = factory.create_render_preview_workflow_from_config()
    validate_input = factory.create_validate_input_workflow_from_config()
    validate_output = factory.create_validate_output_workflow_from_config()
    identify = factory.create_identify_asset_workflow_from_config()

    assert [task.name for task in generate_reference.tasks] == [
        "GenerateRefImageConfig",
        "GenerateReferenceImage",
    ]
    assert [task.name for task in generate_material_library.tasks] == [
        "GenerateMaterialLibraryConfig",
        "GenerateMaterialLibrary",
    ]
    assert [task.name for task in render_preview.tasks] == [
        "RenderPreviewConfig",
        "RenderScenePreview",
    ]
    assert [type(task).__name__ for task in validate_input.tasks] == [
        "ValidateUSDConfigTask",
        "ValidateUSDTask",
    ]
    assert [type(task).__name__ for task in validate_output.tasks] == [
        "ValidateUSDConfigTask",
        "ValidateOutputUSDTask",
    ]
    assert [task.name for task in identify.tasks] == ["IdentifyAsset"]

    assert isinstance(generate_reference.object_store, InMemoryObjectStore)
    assert isinstance(generate_material_library.object_store, InMemoryObjectStore)
    assert isinstance(render_preview.object_store, InMemoryObjectStore)


def test_create_usd_data_preparation_delegates_to_shared_factory(monkeypatch) -> None:
    delegated = Mock()
    monkeypatch.setattr(factory, "create_usd_dataset_workflow", delegated)
    delegated.return_value = "workflow"

    result = factory.create_usd_data_preparation_workflow_from_config()

    assert result == "workflow"
    delegated.assert_called_once_with(
        workflow_name="USD → Dataset Preparation",
        workflow_description="Prepare prim→rendered views dataset from USD",
    )


def test_iterative_apply_workflow_contains_iteration_subworkflow() -> None:
    workflow = factory.create_iterative_apply_workflow_from_config()

    assert workflow.name == "Iterative Material Refinement"
    assert isinstance(workflow.object_store, InMemoryObjectStore)
    assert [task.name for task in workflow.tasks] == [
        "IterativeApplyConfigLoading",
        "DatasetLoading",
        "ModelProvisioning",
        "Iteration",
        "IterativeApplyCompletion",
    ]

    iteration_task = workflow.tasks[3]
    assert iteration_task.max_iterations == 5
    assert isinstance(iteration_task.sub_workflow, Workflow)
    assert isinstance(iteration_task.sub_workflow.object_store, TempDirObjectStore)
    assert [task.name for task in iteration_task.sub_workflow.tasks] == [
        "VLMInference",
        "GeneratePredictionReport",
        "SavePredictions",
        "IdentifyUniqueMaterials",
        "MaterialRetrieval",
        "ResolveMaterialFiles",
        "ApplyMaterialsToUSD",
        "Render",
        "Judge",
    ]


def test_pdf_vectorstore_tool_tasks_keep_expected_mappings() -> None:
    workflow = factory.create_pdf_vectorstore_workflow_from_config()

    extract_task = workflow.tasks[1]
    split_task = workflow.tasks[2]
    build_task = workflow.tasks[3]

    assert isinstance(extract_task, ToolTask)
    assert extract_task.input_mapping["source"] == "source_path"
    assert split_task.input_mapping["input_file_path"] == "extracted_content_path"
    assert build_task.input_mapping["save_path"] == "vectorstore_save_path"
