# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics Agent - VLM-based physics property classification for 3D assets.

Physics Agent classifies 3D asset components (from USD files) by material type,
component function, and physical properties (density, friction, restitution, mass)
using VLM-based analysis.

Example usage:
    # Run full pipeline (convenience function)
    from physics_agent.api import pipeline

    result = pipeline(Path("config.yaml"))

    # Run full pipeline (full control)
    from physics_agent.api import PipelineInput, run_pipeline

    result = run_pipeline(PipelineInput(
        config=Path("pipeline_config.yaml"),
    ))
"""

from dotenv import load_dotenv

# Load .env before API modules cache environment-derived settings at import time.
load_dotenv()

# Keep version lookup after .env initialization and before API module imports.
from .utils import get_version  # noqa: E402

__version__ = get_version()
__package__ = "physics_agent"

# Core API exports
from physics_agent.api import (  # noqa: E402
    BuildDatasetPrepareDatasetInput,
    BuildDatasetUsdInput,
    PipelineInput,
    PipelineOutput,
    PredictInput,
    PredictOutput,
    apipeline,
    arun_pipeline,
    build_dataset_prepare_dataset,
    build_dataset_usd,
    pipeline,
    run_pipeline,
    run_predict,
)

# Function exports
from physics_agent.functions import batch_classify_assets, classify_asset  # noqa: E402

__all__ = [
    # Version
    "__version__",
    # API - Pipeline
    "PipelineInput",
    "PipelineOutput",
    "run_pipeline",
    "arun_pipeline",
    "pipeline",
    "apipeline",
    # API - Predict
    "PredictInput",
    "PredictOutput",
    "run_predict",
    # API - Build Dataset
    "BuildDatasetUsdInput",
    "BuildDatasetPrepareDatasetInput",
    "build_dataset_usd",
    "build_dataset_prepare_dataset",
    # Functions
    "classify_asset",
    "batch_classify_assets",
]
