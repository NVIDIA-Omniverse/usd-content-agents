# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable bounded tuning primitives and trial orchestration."""

from .contracts import (
    DEFAULT_JUDGE_PROGRAMMATIC_WEIGHT,
    DEFAULT_JUDGE_VLM_WEIGHT,
    OPTIMIZER_AUTO,
    OPTIMIZER_BOTORCH,
    OPTIMIZER_CMA_ES,
    OPTIMIZER_RANDOM,
    SUPPORTED_OPTIMIZERS,
    BoundedParameter,
    BoundedSearchSpace,
    OptimizerSettings,
    ReplicaRecord,
    TrialRecord,
    TunableParam,
    TuningObjective,
    combine_judge_scores,
)
from .errors import (
    OptimizationCancelledError,
    OptimizationError,
    OptimizerUnavailableError,
)
from .refinement import (
    RefinementDecision,
    RefinementIteration,
    RefinementLoop,
    RefinementRecord,
    RefinementRun,
    run_refinement,
)
from .runner import TrialRun, TuneRun, TuneWorkflow, run_trials, run_tuning

_LAZY_OPTIMIZER_EXPORTS = frozenset(
    {
        "finite_failed_trial_penalty",
        "get_runner",
        "get_supported_optimizer_names",
        "is_botorch_available",
        "params_from_vector",
        "resolve_optimizer",
        "run_botorch_optimizer",
        "run_cma_es_optimizer",
        "run_random_optimizer",
        "vector_from_params",
    }
)


def __getattr__(name: str) -> object:
    if name in _LAZY_OPTIMIZER_EXPORTS:
        import importlib

        optimizer_module = importlib.import_module(f"{__name__}.optimizers")
        value = getattr(optimizer_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_OPTIMIZER_EXPORTS})


__all__ = [
    "BoundedParameter",
    "BoundedSearchSpace",
    "DEFAULT_JUDGE_PROGRAMMATIC_WEIGHT",
    "DEFAULT_JUDGE_VLM_WEIGHT",
    "OPTIMIZER_AUTO",
    "OPTIMIZER_BOTORCH",
    "OPTIMIZER_CMA_ES",
    "OPTIMIZER_RANDOM",
    "OptimizationCancelledError",
    "OptimizationError",
    "OptimizerSettings",
    "OptimizerUnavailableError",
    "ReplicaRecord",
    "RefinementDecision",
    "RefinementIteration",
    "RefinementLoop",
    "RefinementRecord",
    "RefinementRun",
    "SUPPORTED_OPTIMIZERS",
    "TrialRecord",
    "TrialRun",
    "TuneRun",
    "TuneWorkflow",
    "TunableParam",
    "TuningObjective",
    "combine_judge_scores",
    "finite_failed_trial_penalty",
    "get_runner",
    "get_supported_optimizer_names",
    "is_botorch_available",
    "params_from_vector",
    "resolve_optimizer",
    "run_botorch_optimizer",
    "run_cma_es_optimizer",
    "run_random_optimizer",
    "run_refinement",
    "run_tuning",
    "run_trials",
    "vector_from_params",
]
