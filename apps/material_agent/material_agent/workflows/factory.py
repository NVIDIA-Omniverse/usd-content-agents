# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Factory functions for creating material agent workflows."""

import logging

from world_understanding.agentic import create_usd_dataset_workflow
from world_understanding.agentic.tasks import ToolTask
from world_understanding.agentic.workflows import Workflow
from world_understanding.utils.object_store import (
    InMemoryObjectStore,
    TempDirObjectStore,
)

from material_agent.tasks import (
    BenchmarkConfigTask,
    DatasetLoadingTask,
    EvaluateConfigTask,
    EvaluationTask,
    GenerateConfigTask,
    GenerateEvaluationReportTask,
    GeneratePredictionReportTask,
    ModelProvisioningTask,
    PDFVectorstoreConfigTask,
    PredictConfigTask,
    PrepareDatasetConfigTask,
    PrepareDatasetTask,
    SavePredictionsTask,
    VLMInferenceTask,
)

logger = logging.getLogger(__name__)


def create_optimize_usd_workflow_from_config() -> Workflow:
    """Create a config-driven USD optimization workflow.

    This workflow loads configuration and calls a REST API to optimize a USD file.
    The optimized USD is then used by subsequent pipeline steps.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file

    Returns:
        Configured Workflow instance for USD optimization

    Example:
        workflow = create_optimize_usd_workflow_from_config()
        result = workflow.run({
            "config_path": "configs/optimize_usd.yaml"
        })
    """
    from world_understanding.agentic.usd_tasks import (
        OptimizeUSDConfigTask,
        OptimizeUSDTask,
    )

    tasks = [
        # Load configuration from YAML
        OptimizeUSDConfigTask(),
        # Call optimization API
        OptimizeUSDTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Config-Driven USD Optimization",
        description="Optimize USD file via REST API before dataset building",
    )


def create_validate_predictions_workflow_from_config() -> Workflow:
    """Create a config-driven prediction validation workflow.

    Validates VLM predictions against the material library and auto-corrects
    invalid names using fuzzy matching and optional LLM repair.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file

    Returns:
        Configured Workflow instance for prediction validation
    """
    from material_agent.tasks.config_validate_predictions import (
        ValidatePredictionsConfigTask,
    )
    from material_agent.tasks.validate_predictions import ValidatePredictionsTask

    tasks = [
        ValidatePredictionsConfigTask(),
        ValidatePredictionsTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Config-Driven Prediction Validation",
        description="Validate and repair VLM predictions against material library",
    )


def create_harmonize_predictions_workflow_from_config() -> Workflow:
    """Create a config-driven prediction harmonization workflow.

    Uses prim-path signature grouping to detect instanced meshes with
    conflicting predictions and resolves via majority vote / LLM pick.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file

    Returns:
        Configured Workflow instance for prediction harmonization
    """
    # Reuse the validate config loader — it provides predictions_path and llm
    from material_agent.tasks.config_validate_predictions import (
        ValidatePredictionsConfigTask,
    )
    from material_agent.tasks.harmonize_predictions import (
        HarmonizePredictionsTask,
    )

    tasks = [
        ValidatePredictionsConfigTask(),
        HarmonizePredictionsTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Config-Driven Prediction Harmonization",
        description="Harmonize conflicting predictions for instanced parts",
    )


def create_restore_usd_workflow_from_config() -> Workflow:
    """Create a config-driven USD restoration workflow.

    This workflow loads configuration and calls a REST API to restore the original
    USD structure while preserving applied materials. It uses optimization metadata
    from the optimize_usd step.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file
        - optimization_metadata: Metadata from optimize_usd (injected by executor)

    Returns:
        Configured Workflow instance for USD restoration

    Example:
        workflow = create_restore_usd_workflow_from_config()
        result = workflow.run({
            "config_path": "configs/restore_usd.yaml"
        })
    """
    from world_understanding.agentic.usd_tasks import (
        RestoreUSDConfigTask,
        RestoreUSDTask,
    )

    tasks = [
        # Load configuration from YAML
        RestoreUSDConfigTask(),
        # Call restoration API
        RestoreUSDTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Config-Driven USD Restoration",
        description="Restore USD structure via REST API after material application",
    )


def create_prediction_workflow_from_config() -> Workflow:
    """Create a config-driven prediction workflow.

    This workflow loads configuration from a YAML file, provisions models,
    and runs predictions without evaluation. All tasks use smart parameter resolution.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file
        - dataset_override: Optional dataset path override
        - output_dir_override: Optional output directory override

    Returns:
        Configured Workflow instance for config-driven prediction

    Example:
        workflow = create_prediction_workflow_from_config()
        result = workflow.run({
            "config_path": "configs/prediction.yaml"
        })
    """
    tasks = [
        # Load configuration from YAML
        PredictConfigTask(),
        # Create model instances from config
        ModelProvisioningTask(),
        # Run prediction tasks
        DatasetLoadingTask(),  # Gets dataset_path from context
        VLMInferenceTask(),  # Gets vlm, temperature, etc. from context
        GeneratePredictionReportTask(),  # Generate HTML report
        SavePredictionsTask(include_ground_truth=False),  # Gets output_dir from context
    ]

    return Workflow(
        tasks=tasks,
        object_store=TempDirObjectStore(),
        name="Config-Driven Prediction",
        description="Config-driven prediction workflow without evaluation",
    )


def create_evaluation_workflow_from_config() -> Workflow:
    """Create a config-driven evaluation workflow.

    This workflow loads configuration from a YAML file, provisions the LLM judge,
    and evaluates existing predictions. All tasks use smart parameter resolution.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file
        - predictions_path: Path to the predictions JSONL file to evaluate

    Returns:
        Configured Workflow instance for config-driven evaluation

    Example:
        workflow = create_evaluation_workflow_from_config()
        result = workflow.run({
            "config_path": "configs/evaluation.yaml",
            "predictions_path": "output/predictions.jsonl"
        })
    """
    tasks = [
        # Load configuration from YAML
        EvaluateConfigTask(),
        # Create LLM judge instance from config
        ModelProvisioningTask(),
        # Evaluate the predictions using dataset for ground truth if available
        EvaluationTask(),  # Gets llm_judge, dataset_path from context
        GenerateEvaluationReportTask(),  # Generate evaluation HTML report
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Config-Driven Evaluation",
        description="Config-driven evaluation workflow for existing predictions",
    )


def create_benchmark_workflow_from_config() -> Workflow:
    """Create a config-driven benchmark workflow.

    This workflow loads configuration from a YAML file, provisions models,
    and runs the benchmark. All tasks use smart parameter resolution - they
    can get their parameters from the context if not provided in the constructor.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file
        - dataset_override: Optional dataset path override
        - output_dir_override: Optional output directory override

    Returns:
        Configured Workflow instance with flat task composition

    Example:
        workflow = create_benchmark_workflow_from_config()
        result = workflow.run({
            "config_path": "configs/benchmark.yaml"
        })
    """
    tasks = [
        # Load configuration from YAML
        BenchmarkConfigTask(),
        # Create model instances from config
        ModelProvisioningTask(),
        # Now the actual benchmark tasks - they'll get params from context
        DatasetLoadingTask(),  # Gets dataset_path from context
        VLMInferenceTask(),  # Gets vlm, temperature, etc. from context
        GeneratePredictionReportTask(),  # Generate prediction HTML report
        SavePredictionsTask(include_ground_truth=True),  # Gets output_dir from context
        EvaluationTask(),  # Gets llm_judge from context
        GenerateEvaluationReportTask(),  # Generate evaluation HTML report
    ]

    return Workflow(
        tasks=tasks,
        object_store=TempDirObjectStore(),
        name="Config-Driven Benchmark",
        description="Flat configuration-driven benchmark workflow",
    )


def create_pdf_vectorstore_workflow_from_config() -> Workflow:
    """Create a config-driven PDF to vectorstore workflow.

    This workflow loads configuration from a YAML file, processes PDF documents,
    and builds a multimodal vector store. It uses ToolTask to leverage existing
    tools for document processing.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file
        - source_override: Optional source path override (PDF file or directory)
        - output_dir_override: Optional output directory override

    Returns:
        Configured Workflow instance for PDF to vectorstore conversion

    Example:
        workflow = create_pdf_vectorstore_workflow_from_config()
        result = workflow.run({
            "config_path": "configs/pdf_vectorstore.yaml",
            "source_override": "docs/pdfs/"
        })
    """

    tasks = [
        # Load and validate configuration
        PDFVectorstoreConfigTask(),
        # Extract content from PDFs using ToolTask
        # The config task will set up the context with proper paths
        ToolTask(
            tool_name="extract_document_content",
            inputs={},  # Will be populated from context
            input_mapping={
                "source": "source_path",
                "output_dir": "extraction_output_dir",
                "save_content_only": "save_content_only",
            },
            output_key="extraction_result",
            name="Extract PDF Content",
        ),
        # Split content by type
        ToolTask(
            tool_name="split_document_content",
            inputs={},  # Will be populated from context
            input_mapping={
                "input_file_path": "extracted_content_path",
                "output_dir": "split_output_dir",
            },
            output_key="split_result",
            name="Split Content by Type",
        ),
        # Build multimodal vector store
        ToolTask(
            tool_name="build_multimodal_vector_store",
            inputs={},  # Will be populated from context
            input_mapping={
                "text_source": "split_output_dir",
                "image_source": "split_output_dir",
                "save_path": "vectorstore_save_path",
                "image_embedding_type": "image_embedding_type",
                "embedding_model": "embedding_model",  # Optional, from context
                "include_filename_metadata": "include_filename_metadata",  # Include filename in metadata
            },
            output_key="vectorstore_result",
            name="Build Vector Store",
        ),
    ]

    return Workflow(
        tasks=tasks,
        object_store=TempDirObjectStore(),  # Use file-based storage for large PDFs
        name="Config-Driven PDF to VectorStore",
        description="Convert PDF documents to a searchable multimodal vector store",
    )


def create_prepare_dataset_workflow_from_config() -> Workflow:
    """Create a config-driven prepare dataset workflow.

    This workflow loads configuration from a YAML file, optionally provisions the LLM,
    and prepares dataset by extracting CMF specifications for model numbers.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file
        - vector_store_override: Optional vector store path override
        - dataset_override: Optional dataset path override

    Returns:
        Configured Workflow instance for dataset preparation

    Example:
        workflow = create_prepare_dataset_workflow_from_config()
        result = workflow.run({
            "config_path": "configs/prepare_dataset.yaml",
            "models_override": ["115-3769-000_1", "074-0388-000_1"]
        })
    """
    tasks = [
        # Load configuration from YAML
        PrepareDatasetConfigTask(),
        # Create LLM instance from config
        ModelProvisioningTask(),
        # Prepare dataset
        PrepareDatasetTask(),  # Gets vector_store_path, dataset_path, models, llm from context
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Config-Driven Prepare Dataset",
        description="Prepare dataset with CMF specifications using config-driven workflow",
    )


def create_usd_data_preparation_workflow_from_config() -> Workflow:
    """Create a config-driven USD data preparation workflow.

    This workflow loads configuration from a YAML file, opens a USD stage,
    renders views of each prim, and builds a dataset manifest.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file
        - source_override: Optional USD path override
        - output_dir_override: Optional output directory override
        - resume: Optional flag to resume from previous run
        - max_workers: Optional number of parallel workers
        - prim_filters: Optional filters for prim selection
        - extract_prim_metadata: Optional flag to extract prim metadata

    Returns:
        Configured Workflow instance for data preparation

    Example:
        workflow = create_usd_data_preparation_workflow_from_config()
        result = workflow.run({
            "config_path": "configs/data_prep.yaml",
            "source_override": "assets/vehicle.usd"
        })

    Note:
        This function now delegates to the shared USD workflow factory
        (world_understanding.agents.create_usd_dataset_workflow) to ensure
        consistency across all agents.
    """
    return create_usd_dataset_workflow(
        workflow_name="USD → Dataset Preparation",
        workflow_description="Prepare prim→rendered views dataset from USD",
    )


def create_apply_workflow_from_config() -> Workflow:
    """Create a config-driven material application workflow.

    This workflow loads configuration from a YAML file, identifies unique materials
    from predictions, uses USD Search to find matching materials, and applies them
    to the USD file.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file
        - input_usd_override: Optional input USD file override
        - predictions_override: Optional predictions file override
        - output_usd_override: Optional output USD file override

    Returns:
        Configured Workflow instance for material application

    Example:
        workflow = create_apply_workflow_from_config()
        result = workflow.run({
            "config_path": "configs/apply.yaml",
            "input_usd_override": "input.usd"
        })
    """
    from material_agent.tasks import (
        ApplyCompletionTask,
        ApplyConfigTask,
        ApplyMaterialsToUSDTask,
        IdentifyUniqueMaterialsTask,
        MaterialRetrievalTask,
        RenderTask,
        ResolveMaterialFilesTask,
    )

    tasks = [
        # Load configuration from YAML
        ApplyConfigTask(),
        # Identify unique materials from predictions
        IdentifyUniqueMaterialsTask(),
        # Retrieve materials using USD Search
        MaterialRetrievalTask(),
        # Resolve material files locally or download from S3
        ResolveMaterialFilesTask(),
        # Apply resolved materials to USD prims
        ApplyMaterialsToUSDTask(),
        # Optional: Render the completed USD stage
        RenderTask(),
        # Mark workflow as complete and display results
        ApplyCompletionTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Config-Driven Material Application",
        description="Apply predicted materials to USD file using USD Search",
    )


def create_generate_reference_image_workflow_from_config() -> Workflow:
    """Create a config-driven generate-reference-image workflow.

    This workflow generates photorealistic reference images from scene
    preview renders and a user text prompt.  The generated images are
    automatically injected as reference images for the downstream
    ``predict`` step.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file

    Returns:
        Configured Workflow instance for reference image generation

    Example:
        workflow = create_generate_reference_image_workflow_from_config()
        result = workflow.run({"config_path": "configs/gen_ref.yaml"})
    """
    from world_understanding.agentic.usd_tasks import GenerateReferenceImageTask

    from material_agent.tasks import GenerateRefImageConfigTask

    tasks = [
        GenerateRefImageConfigTask(),
        GenerateReferenceImageTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Generate Reference Image",
        description="Generate photorealistic reference images from preview renders",
    )


def create_generate_material_library_workflow_from_config() -> Workflow:
    """Create a config-driven generated-material-library workflow.

    This workflow reads a material generation plan, directly synthesizes texture
    maps, authors a USD material library, and writes canonical ``materials.yaml``
    for downstream Material Agent steps.
    """
    from material_agent.tasks import (
        GenerateMaterialLibraryConfigTask,
        GenerateMaterialLibraryTask,
    )

    tasks = [
        GenerateMaterialLibraryConfigTask(),
        GenerateMaterialLibraryTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Generate Material Library",
        description="Generate an asset-specific material library package",
    )


def create_render_preview_workflow_from_config() -> Workflow:
    """Create a config-driven render-preview workflow.

    This workflow renders lightweight whole-scene preview images before the
    full dataset-building step.  It uses the shared ``RenderScenePreviewTask``
    so that both the material agent and the asset agent share the same
    rendering logic.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file

    Returns:
        Configured Workflow instance for scene preview rendering

    Example:
        workflow = create_render_preview_workflow_from_config()
        result = workflow.run({"config_path": "configs/render_preview.yaml"})
    """
    from world_understanding.agentic.usd_tasks import RenderScenePreviewTask

    from material_agent.tasks import RenderPreviewConfigTask

    tasks = [
        RenderPreviewConfigTask(),
        RenderScenePreviewTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Scene Preview Rendering",
        description="Render whole-scene preview images",
    )


def create_identify_asset_workflow_from_config() -> Workflow:
    """Create a workflow for identifying asset type/description from previews.

    This workflow uses VLM to identify the asset from preview images and
    USD metadata, producing a structured description that can auto-generate
    reference image prompts.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file
        - rendered_preview_paths: Preview images (from render_preview step)

    Returns:
        Configured Workflow instance for asset identification
    """
    from world_understanding.agentic.usd_tasks import IdentifyAssetTask

    tasks = [
        IdentifyAssetTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Asset Identification",
        description="Identify asset type and description from preview images",
    )


def create_validate_input_workflow_from_config() -> Workflow:
    """Create a config-driven pre-validation workflow.

    Validates the input USD asset before any processing to establish
    a baseline of existing issues. This baseline is used by
    validate_output to detect regressions.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file

    Returns:
        Configured Workflow instance for input USD validation
    """
    from world_understanding.agentic.usd_tasks import (
        ValidateUSDConfigTask,
        ValidateUSDTask,
    )

    tasks = [
        ValidateUSDConfigTask(),
        ValidateUSDTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Pre-Validation (Input USD)",
        description="Validate input USD asset to establish baseline",
    )


def create_validate_output_workflow_from_config() -> Workflow:
    """Create a config-driven post-validation workflow.

    Validates both the original input USD and the output USD (after material
    assignment), then compares them to detect regressions. Self-contained:
    does not require validate_input to have run, but will reuse its results
    if available.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file
        - original_usd_path: Path to original input USD (for baseline)
        - baseline_validation: (optional) Cached baseline from validate_input

    Returns:
        Configured Workflow instance for output USD validation
    """
    from world_understanding.agentic.usd_tasks import (
        ValidateOutputUSDTask,
        ValidateUSDConfigTask,
    )

    tasks = [
        ValidateUSDConfigTask(),
        ValidateOutputUSDTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Post-Validation (Output USD)",
        description="Validate output USD and compare against input baseline",
    )


def create_render_workflow_from_config() -> Workflow:
    """Create a config-driven render workflow.

    This workflow loads configuration, takes a USD file (typically from apply step),
    optionally flattens it, and renders it from specified viewpoints.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file
        - input_usd_override: Optional input USD file override
        - output_path_override: Optional output path override

    Returns:
        Configured Workflow instance for rendering

    Example:
        workflow = create_render_workflow_from_config()
        result = workflow.run({
            "config_path": "configs/unified_ladder.yaml",
            "input_usd_override": "output/ladder_with_materials.usd"
        })
    """
    from material_agent.tasks import RenderConfigTask, RenderTask

    tasks = [
        # Load render configuration from YAML
        RenderConfigTask(),
        # Flatten and render USD
        RenderTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Config-Driven Render",
        description="Flatten and render USD file with materials",
    )


def create_iterative_apply_workflow_from_config() -> Workflow:
    """Create an iterative material refinement workflow with judge feedback.

    This workflow performs iterative material prediction and application with
    a judge that evaluates quality and decides whether to continue refining.

    The workflow structure:
        [Config] → [Dataset] → [Provisioning] → [Iteration Loop] → [Completion]
                                                       │
                                        ┌──────────────┴──────────────┐
                                        │ [Predict → Apply → Judge]  │ ← Repeated
                                        └─────────────────────────────┘

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file
        - max_iterations_override: Optional override for max iterations

    Returns:
        Configured Workflow instance for iterative material refinement

    Example:
        workflow = create_iterative_apply_workflow_from_config()
        result = workflow.run({
            "config_path": "configs/iterative_apply.yaml",
            "max_iterations_override": 3
        })
    """
    # Import tasks needed for iteration sub-workflow
    from material_agent.tasks import (
        ApplyMaterialsToUSDTask,
        DatasetLoadingTask,
        GeneratePredictionReportTask,
        IdentifyUniqueMaterialsTask,
        IterationTask,
        IterativeApplyCompletionTask,
        IterativeApplyConfigTask,
        JudgeTask,
        MaterialRetrievalTask,
        ModelProvisioningTask,
        RenderTask,
        ResolveMaterialFilesTask,
        SavePredictionsTask,
        VLMInferenceTask,
    )

    # Define the iteration sub-workflow (executed repeatedly)
    iteration_sub_workflow = Workflow(
        tasks=[
            # Predict materials using VLM
            VLMInferenceTask(),
            # Generate HTML report of predictions
            GeneratePredictionReportTask(),
            # Save predictions for this iteration
            SavePredictionsTask(),
            # Identify unique materials from predictions
            IdentifyUniqueMaterialsTask(),
            # Retrieve and resolve material files
            MaterialRetrievalTask(),
            ResolveMaterialFilesTask(),
            # Apply materials to USD
            ApplyMaterialsToUSDTask(),
            # Render for judge inspection
            RenderTask(),
            # Judge: Evaluate and decide whether to continue
            JudgeTask(),
        ],
        object_store=TempDirObjectStore(),
        name="Material Refinement Iteration",
        description="Single iteration of predict-apply-judge loop",
    )

    # Main workflow with iteration wrapper
    tasks = [
        # Load configuration for iterative apply
        IterativeApplyConfigTask(),
        # Load dataset
        DatasetLoadingTask(),
        # Provision VLM, LLM, and Judge models
        ModelProvisioningTask(),
        # Execute iteration loop
        IterationTask(
            sub_workflow=iteration_sub_workflow,
            max_iterations=5,  # Default, can be overridden in config
        ),
        # Finalize and copy final output
        IterativeApplyCompletionTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Iterative Material Refinement",
        description="Iteratively refine materials with judge-based feedback",
    )


def create_configure_workflow() -> Workflow:
    """Create a workflow for generating pipeline configuration files.

    This workflow interactively prompts the user for essential configuration
    parameters and generates a complete pipeline configuration file with
    sensible defaults.

    The workflow expects the following initial context:
        - output_config_path: Path where the configuration file will be written
        - force: Whether to overwrite existing configuration file

    Returns:
        Configured Workflow instance for configuration generation

    Example:
        workflow = create_configure_workflow()
        result = workflow.run({
            "output_config_path": "my_pipeline.yaml",
            "force": False
        })
    """
    tasks = [
        # Generate configuration file interactively
        GenerateConfigTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Configuration Generator",
        description="Generate pipeline configuration file interactively",
    )


def create_cluster_prims_workflow_from_config() -> Workflow:
    """Create a workflow that clusters prim images before prediction.

    Reads dataset.jsonl, embeds prim_only images, clusters by
    complexity-aware cosine similarity, and writes a representative-only
    dataset for the predict step.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file

    Returns:
        Configured Workflow instance for prim clustering
    """
    from material_agent.tasks.cluster_prims import ClusterPrimsTask
    from material_agent.tasks.config_cluster_prims import ClusterPrimsConfigTask

    tasks = [
        ClusterPrimsConfigTask(),
        ClusterPrimsTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Prim Clustering",
        description="Cluster visually similar prims to reduce VLM calls",
    )


def create_expand_cluster_predictions_workflow_from_config() -> Workflow:
    """Create a workflow that expands representative predictions to all cluster members.

    After predict runs on representatives only, this step propagates each
    prediction to every member of its cluster.

    The workflow expects the following initial context:
        - config_path: Path to the YAML configuration file

    Returns:
        Configured Workflow instance for cluster prediction expansion
    """
    from material_agent.tasks.cluster_prims import ExpandClusterPredictionsTask
    from material_agent.tasks.config_cluster_prims import (
        ExpandClusterPredictionsConfigTask,
    )

    tasks = [
        ExpandClusterPredictionsConfigTask(),
        ExpandClusterPredictionsTask(),
    ]

    return Workflow(
        tasks=tasks,
        object_store=InMemoryObjectStore(),
        name="Expand Cluster Predictions",
        description="Propagate representative predictions to all cluster members",
    )
