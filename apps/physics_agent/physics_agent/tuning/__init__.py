# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics Agent tuning package.

Tunes authored physics parameters on a simulation-ready USD by running a
black-box optimizer (BoTorch / random / CMA-ES) against a deterministic
simulator backend (OvPhysX or a fake backend for tests).

Public API entry points:

    from physics_agent.tuning import TuneInput, TuneOutput, run_tune, arun_tune

The CLI is wired in :mod:`physics_agent.cli` (``physics-agent tune``) and the
REST surface in :mod:`physics_agent_service.service.routers.tune_router`.

**Import-graph invariant.** Eagerly loading :mod:`.runner` here would
chain into :mod:`.optimizers` (botorch / cma-es) for any consumer of
:mod:`physics_agent` — including the prompt interpreter and the
VLM-as-judge, which deliberately doesn't need the optimizer stack. The
``test_tasks_no_optimizer_imports`` subprocess test enforces that
invariant. We therefore expose ``run_tune`` / ``arun_tune`` via
:pep:`562` ``__getattr__`` so the runner module loads only on first
attribute access. Static type checkers still see the symbols via the
``TYPE_CHECKING`` re-export.
"""

from typing import TYPE_CHECKING, Any

from .errors import (
    BoTorchUnavailableError,
    OvPhysXUnavailableError,
    TuningCancelledError,
    TuningError,
)
from .types import (
    SUPPORTED_PARAM_KEYS,
    SUPPORTED_SCENARIOS,
    BackendArtifacts,
    Scenario,
    TrialRecord,
    TunableParam,
    TuneInput,
    TuneOutput,
)

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from .runner import arun_tune as arun_tune
    from .runner import run_tune as run_tune


_LAZY_RUNNER_NAMES = frozenset({"run_tune", "arun_tune"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_RUNNER_NAMES:
        from . import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TuneInput",
    "TuneOutput",
    "Scenario",
    "TunableParam",
    "TrialRecord",
    "BackendArtifacts",
    "SUPPORTED_PARAM_KEYS",
    "SUPPORTED_SCENARIOS",
    "run_tune",
    "arun_tune",
    "TuningError",
    "BoTorchUnavailableError",
    "OvPhysXUnavailableError",
    "TuningCancelledError",
]
