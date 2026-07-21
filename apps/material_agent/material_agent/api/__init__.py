# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material Agent Python API.

This module provides programmatic access to Material Agent functionality.
All commands available in the CLI are also available as Python functions.

The API offers two usage patterns:

1. **Convenience functions** - Minimal, simple usage:
    ```python
    from material_agent.api import benchmark
    from pathlib import Path

    # Run with just config
    result = benchmark(Path("config.yaml"))

    # Or with optional overrides
    result = benchmark(Path("config.yaml"), verbose=True, resume=True)
    ```

2. **Full Input classes** - Maximum control and type safety:
    ```python
    from material_agent.api import run_benchmark, BenchmarkInput
    from pathlib import Path

    params = BenchmarkInput(
        config=Path("config.yaml"),
        dataset_override=Path("data.jsonl"),
        verbose=True
    )

    result = run_benchmark(params)
    if result.success:
        print(f"FCS: {result.metrics.functional_correctness_score}")
    ```
"""

from pathlib import Path
from typing import Any

# Import event system
from world_understanding.agentic.events import (
    CLIEventListener,
    CollectingEventListener,
    EventListener,
    LoggerAsListener,
    NoOpEventListener,
    create_default_listener,
    get_listener,
)

# Import shared types
# Import apply API
from material_agent.api.apply import (
    ApplyInput,
    ApplyOutput,
    aapply,
    apply,
    arun_apply,
    run_apply,
)

# Import benchmark API
from material_agent.api.benchmark import (
    BenchmarkInput,
    BenchmarkOutput,
    abenchmark,
    arun_benchmark,
    benchmark,
    run_benchmark,
)

# Import build_dataset APIs
from material_agent.api.build_dataset import (
    BuildDatasetPdfVectorstoreInput,
    BuildDatasetPdfVectorstoreOutput,
    BuildDatasetPrepareDatasetInput,
    BuildDatasetPrepareDatasetOutput,
    BuildDatasetUsdInput,
    BuildDatasetUsdOutput,
    abuild_dataset_pdf_vectorstore,
    abuild_dataset_prepare_dataset,
    abuild_dataset_usd,
    build_dataset_pdf_vectorstore,
    build_dataset_prepare_dataset,
    build_dataset_usd,
)

# Import config builders
from material_agent.api.builders import (
    build_apply_config,
    build_benchmark_config,
    build_cluster_prims_config,
    build_predict_config,
    build_unified_pipeline_config,
    build_vlm_config,
    get_required_fields,
)

# Import configure API
from material_agent.api.configure import (
    ConfigureInput,
    ConfigureOutput,
    aconfigure,
    arun_configure,
    configure,
    run_configure,
)

# Import refine API
from material_agent.api.refine import (
    IterationResult,
    RefineInput,
    RefineOutput,
    arefine,
    arun_refine,
    refine,
    run_refine,
)
from material_agent.utils import get_version

__version__ = get_version()

# Import defaults and constants
from material_agent.api.defaults import (
    APPLY_DEFAULTS,
    BENCHMARK_DEFAULTS,
    DEFAULT_CAMERA_DIRECTIONS,
    DEFAULT_DATASET_IMAGE_SIZE,
    DEFAULT_FINAL_IMAGE_SIZE,
    DEFAULT_JUDGE_BACKEND,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_MODEL,
    DEFAULT_RENDER_BACKEND,
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_MODEL,
    ITERATION_DEFAULTS,
    MUTUALLY_EXCLUSIVE_STEPS,
    PIPELINE_DEFAULTS,
    PIPELINE_STEP_NAMES,
    PREDICT_DEFAULTS,
    STEP_APPLY,
    STEP_BENCHMARK,
    STEP_BUILD_DATASET_PDF_VECTORSTORE,
    STEP_BUILD_DATASET_PREPARE_DATASET,
    STEP_BUILD_DATASET_USD,
    STEP_CLUSTER_PRIMS,
    STEP_EXPAND_CLUSTER_PREDICTIONS,
    STEP_GENERATE_MATERIAL_LIBRARY,
    STEP_PREDICT,
    STEP_REFINE,
    STEP_RENDER,
    USD_DATASET_DEFAULTS,
    apply_defaults,
    get_apply_config_with_defaults,
    get_benchmark_config_with_defaults,
    get_minimal_required_fields,
    get_predict_config_with_defaults,
)

# Import evaluate API
from material_agent.api.evaluate import (
    EvaluateInput,
    EvaluateOutput,
    aevaluate,
    arun_evaluate,
    evaluate,
    run_evaluate,
)

# Import pipeline API
from material_agent.api.pipeline import (
    PipelineInput,
    PipelineOutput,
    apipeline,
    arun_pipeline,
    pipeline,
    run_pipeline,
)

# Import predict API
from material_agent.api.predict import (
    PredictInput,
    PredictOutput,
    apredict,
    arun_predict,
    predict,
    run_predict,
)

# Import scene pipeline API
from material_agent.api.scene_pipeline import (
    ScenePipelineInput,
    ScenePipelineOutput,
    arun_scene_pipeline,
    ascene_pipeline,
    run_scene_pipeline,
    scene_pipeline,
)
from material_agent.api.types import (
    APIResult,
    AssignmentStats,
    DownloadStats,
    MaterialSearchResult,
    MetricsResult,
)

__all__ = [
    # Event System
    "EventListener",
    "CLIEventListener",
    "CollectingEventListener",
    "NoOpEventListener",
    "LoggerAsListener",
    "create_default_listener",
    "get_listener",
    # Constants
    "PIPELINE_STEP_NAMES",
    "STEP_BUILD_DATASET_USD",
    "STEP_BUILD_DATASET_PDF_VECTORSTORE",
    "STEP_BUILD_DATASET_PREPARE_DATASET",
    "STEP_CLUSTER_PRIMS",
    "STEP_EXPAND_CLUSTER_PREDICTIONS",
    "STEP_GENERATE_MATERIAL_LIBRARY",
    "STEP_PREDICT",
    "STEP_BENCHMARK",
    "STEP_APPLY",
    "STEP_REFINE",
    "STEP_RENDER",
    "MUTUALLY_EXCLUSIVE_STEPS",
    # Config Builders
    "build_vlm_config",
    "build_predict_config",
    "build_benchmark_config",
    "build_apply_config",
    "build_cluster_prims_config",
    "build_unified_pipeline_config",
    "get_required_fields",
    # Defaults & Utilities
    "DEFAULT_VLM_BACKEND",
    "DEFAULT_VLM_MODEL",
    "DEFAULT_LLM_BACKEND",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_JUDGE_BACKEND",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_RENDER_BACKEND",
    "DEFAULT_CAMERA_DIRECTIONS",
    "DEFAULT_DATASET_IMAGE_SIZE",
    "DEFAULT_FINAL_IMAGE_SIZE",
    "PREDICT_DEFAULTS",
    "BENCHMARK_DEFAULTS",
    "APPLY_DEFAULTS",
    "PIPELINE_DEFAULTS",
    "ITERATION_DEFAULTS",
    "USD_DATASET_DEFAULTS",
    "apply_defaults",
    "get_predict_config_with_defaults",
    "get_benchmark_config_with_defaults",
    "get_apply_config_with_defaults",
    "get_minimal_required_fields",
    # Shared types
    "APIResult",
    "MetricsResult",
    "MaterialSearchResult",
    "AssignmentStats",
    "DownloadStats",
    # Benchmark
    "BenchmarkInput",
    "BenchmarkOutput",
    "run_benchmark",
    "arun_benchmark",
    "benchmark",
    "abenchmark",
    # Predict
    "PredictInput",
    "PredictOutput",
    "run_predict",
    "arun_predict",
    "predict",
    "apredict",
    # Evaluate
    "EvaluateInput",
    "EvaluateOutput",
    "run_evaluate",
    "arun_evaluate",
    "evaluate",
    "aevaluate",
    # Apply
    "ApplyInput",
    "ApplyOutput",
    "run_apply",
    "arun_apply",
    "apply",
    "aapply",
    # Pipeline
    "PipelineInput",
    "PipelineOutput",
    "run_pipeline",
    "arun_pipeline",
    "pipeline",
    "apipeline",
    "ScenePipelineInput",
    "ScenePipelineOutput",
    "run_scene_pipeline",
    "arun_scene_pipeline",
    "scene_pipeline",
    "ascene_pipeline",
    # Build Dataset - USD
    "BuildDatasetUsdInput",
    "BuildDatasetUsdOutput",
    "build_dataset_usd",
    "abuild_dataset_usd",
    # Build Dataset - PDF VectorStore
    "BuildDatasetPdfVectorstoreInput",
    "BuildDatasetPdfVectorstoreOutput",
    "build_dataset_pdf_vectorstore",
    "abuild_dataset_pdf_vectorstore",
    # Build Dataset - Prepare Dataset
    "BuildDatasetPrepareDatasetInput",
    "BuildDatasetPrepareDatasetOutput",
    "build_dataset_prepare_dataset",
    "abuild_dataset_prepare_dataset",
    # Refine
    "RefineInput",
    "RefineOutput",
    "IterationResult",
    "run_refine",
    "arun_refine",
    "refine",
    "arefine",
    # Configure
    "ConfigureInput",
    "ConfigureOutput",
    "run_configure",
    "arun_configure",
    "configure",
    "aconfigure",
]


# ============================================================================
# Convenience Functions
# ============================================================================


# All convenience functions are now imported from their respective modules
# (predict, evaluate, apply, pipeline, refine, configure)
