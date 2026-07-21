# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint Agent API - High-level programmatic interface.

This module provides programmatic access to Joint Agent functionality.
All commands available in the CLI are also available as Python functions.

The API offers two usage patterns:

1. **Convenience functions** - Minimal, simple usage:
    ```python
    from joint_agent.api import pipeline
    from pathlib import Path

    result = pipeline(Path("config.yaml"))
    ```

2. **Full Input classes** - Maximum control and type safety:
    ```python
    from joint_agent.api import run_pipeline, PipelineInput
    from pathlib import Path

    params = PipelineInput(
        config=Path("config.yaml"),
        only_steps=["build_dataset_usd"],
        verbose=True,
    )

    result = run_pipeline(params)
    if result.success:
        print(f"Completed: {result.completed_steps}")
    ```
"""

# Import event system from shared module
from world_understanding.agentic.events import (
    CLIEventListener,
    CollectingEventListener,
    EventListener,
    LoggerAsListener,
    NoOpEventListener,
    create_default_listener,
    get_listener,
)

# Import build dataset APIs
from joint_agent.api.build_dataset import (
    BuildDatasetPrepareDatasetInput,
    BuildDatasetPrepareDatasetOutput,
    BuildDatasetUsdInput,
    BuildDatasetUsdOutput,
    abuild_dataset_prepare_dataset,
    abuild_dataset_usd,
    build_dataset_prepare_dataset,
    build_dataset_usd,
)

# Import defaults
from joint_agent.api.defaults import (
    DEFAULT_CAMERA_DIRECTIONS,
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_MODEL,
    PIPELINE_STEP_NAMES,
    PREDICT_DEFAULTS,
    apply_defaults,
    build_default_pipeline_config,
)

# Import pipeline API
from joint_agent.api.pipeline import (
    PipelineInput,
    PipelineOutput,
    apipeline,
    arun_pipeline,
    pipeline,
    run_pipeline,
)

# Import predict API
from joint_agent.api.predict import (
    PredictInput,
    PredictOutput,
    arun_predict,
    run_predict,
)

# Import shared types
from joint_agent.api.types import APIResult

__all__ = [
    # Event System
    "EventListener",
    "CLIEventListener",
    "CollectingEventListener",
    "NoOpEventListener",
    "LoggerAsListener",
    "create_default_listener",
    "get_listener",
    # Shared types
    "APIResult",
    # Pipeline
    "PipelineInput",
    "PipelineOutput",
    "run_pipeline",
    "arun_pipeline",
    "pipeline",
    "apipeline",
    # Predict
    "PredictInput",
    "PredictOutput",
    "run_predict",
    "arun_predict",
    # Build Dataset - USD
    "BuildDatasetUsdInput",
    "BuildDatasetUsdOutput",
    "build_dataset_usd",
    "abuild_dataset_usd",
    # Build Dataset - Prepare Dataset
    "BuildDatasetPrepareDatasetInput",
    "BuildDatasetPrepareDatasetOutput",
    "build_dataset_prepare_dataset",
    "abuild_dataset_prepare_dataset",
    # Defaults
    "DEFAULT_CAMERA_DIRECTIONS",
    "DEFAULT_VLM_BACKEND",
    "DEFAULT_VLM_MODEL",
    "PIPELINE_STEP_NAMES",
    "PREDICT_DEFAULTS",
    "apply_defaults",
    "build_default_pipeline_config",
]
