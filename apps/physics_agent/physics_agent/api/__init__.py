# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics Agent API - High-level programmatic interface.

This module provides programmatic access to Physics Agent functionality.
All commands available in the CLI are also available as Python functions.

The API offers two usage patterns:

1. **Convenience functions** - Minimal, simple usage:
    ```python
    from physics_agent.api import pipeline
    from pathlib import Path

    result = pipeline(Path("config.yaml"))
    ```

2. **Full Input classes** - Maximum control and type safety:
    ```python
    from physics_agent.api import run_pipeline, PipelineInput
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
from typing import TYPE_CHECKING

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
from physics_agent.api.build_dataset import (
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
from physics_agent.api.defaults import (
    DEFAULT_CAMERA_DIRECTIONS,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_TEMPERATURE,
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_MODEL,
    PIPELINE_STEP_NAMES,
    PREDICT_DEFAULTS,
    apply_defaults,
    build_default_pipeline_config,
)

# Import pipeline API
from physics_agent.api.pipeline import (
    PipelineInput,
    PipelineOutput,
    apipeline,
    arun_pipeline,
    pipeline,
    run_pipeline,
)

# Import predict API
from physics_agent.api.predict import (
    PredictInput,
    PredictOutput,
    arun_predict,
    run_predict,
)

# Import shared types
from physics_agent.api.types import APIResult

# Tuning API re-export. The full ``physics_agent.tuning`` package — even
# its leaf submodules ``physics_agent.tuning.types`` and
# ``physics_agent.tuning.errors`` — is exposed via :pep:`562`
# ``__getattr__`` so consumers of this module who only touch
# ``predict`` / build-dataset / pipeline never load any
# ``physics_agent.tuning.*`` submodule. ``run_tune`` / ``arun_tune`` would
# otherwise transitively pull in ``physics_agent.tuning.optimizers``
# (botorch / cma-es / torch); the test
# ``test_predict_runtime_import_does_not_pull_tuning`` enforces that
# importing ``physics_agent.api`` leaves ``sys.modules`` free of every
# ``physics_agent.tuning`` submodule.
#
# Round 15 (doyubkim blocker #3): refine is now a first-class API surface
# mirroring material-agent's ``RefineInput``/``RefineOutput``/``run_refine``/
# ``arun_refine`` shape. Loaded lazily through the same ``__getattr__``
# pathway so importing ``physics_agent.api`` still does NOT pull the
# orchestrator module (which would transitively load
# ``physics_agent.tuning.runner`` and its optimizer/sim dependencies).
if TYPE_CHECKING:  # pragma: no cover - static typing only
    from physics_agent.api.refine import RefineInput as RefineInput
    from physics_agent.api.refine import RefineOutput as RefineOutput
    from physics_agent.api.refine import arun_refine as arun_refine
    from physics_agent.api.refine import run_refine as run_refine
    from physics_agent.tuning import TuneInput as TuneInput
    from physics_agent.tuning import TuneOutput as TuneOutput
    from physics_agent.tuning import arun_tune as arun_tune
    from physics_agent.tuning import run_tune as run_tune


_LAZY_TUNING_NAMES = frozenset({"TuneInput", "TuneOutput", "run_tune", "arun_tune"})
_LAZY_REFINE_NAMES = frozenset(
    {"RefineInput", "RefineOutput", "run_refine", "arun_refine"}
)


def __getattr__(name: str) -> object:
    if name in _LAZY_TUNING_NAMES:
        import physics_agent.tuning as _tuning

        return getattr(_tuning, name)
    if name in _LAZY_REFINE_NAMES:
        import physics_agent.api.refine as _refine

        return getattr(_refine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    # Tune
    "TuneInput",
    "TuneOutput",
    "run_tune",
    "arun_tune",
    # Refine — iterative tune → judge → scenario_refine loop. Mirrors the
    # material-agent refine surface so cross-domain callers get a
    # consistent contract.
    "RefineInput",
    "RefineOutput",
    "run_refine",
    "arun_refine",
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
    "DEFAULT_JUDGE_MAX_TOKENS",
    "DEFAULT_JUDGE_TEMPERATURE",
    "DEFAULT_VLM_BACKEND",
    "DEFAULT_VLM_MODEL",
    "PIPELINE_STEP_NAMES",
    "PREDICT_DEFAULTS",
    "apply_defaults",
    "build_default_pipeline_config",
]
