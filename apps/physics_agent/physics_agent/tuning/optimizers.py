# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optimizer dispatch + lazy BoTorch loader.

Three optimizers are exposed:

* ``botorch`` — first-class production optimizer (Bayesian Optimization via
  GP + qEI). Implemented with a lazy import; missing deps raise
  :class:`BoTorchUnavailableError` carrying the install hint mandated by the
  issue body.
* ``random`` — uniform random search baseline. Always available.
* ``cma-es`` — Covariance Matrix Adaptation ES. Always available — it lives in
  :mod:`world_understanding.functions.optimization.cma_es`.

``--optimizer auto`` resolves to ``botorch`` when BoTorch is importable, and
otherwise raises :class:`BoTorchUnavailableError`. **There is no silent
fallback** to random — that contract is part of the issue Acceptance Criteria.
"""

from __future__ import annotations

import logging
import random as _random_module
from collections.abc import Callable

import numpy as np

from .errors import BoTorchUnavailableError
from .types import Scenario, TunableParam

logger = logging.getLogger(__name__)

OPTIMIZER_AUTO = "auto"
OPTIMIZER_BOTORCH = "botorch"
OPTIMIZER_RANDOM = "random"
OPTIMIZER_CMA_ES = "cma-es"

# Order matters for `--help` rendering and error messages.
SUPPORTED_OPTIMIZERS: tuple[str, ...] = (
    OPTIMIZER_AUTO,
    OPTIMIZER_BOTORCH,
    OPTIMIZER_RANDOM,
    OPTIMIZER_CMA_ES,
)


# Type alias for the per-trial evaluation callback the optimizer drives.
EvaluateFn = Callable[[dict[str, float]], float]


def _params_from_vector(scenario: Scenario, x: np.ndarray) -> dict[str, float]:
    """Convert a unit-cube vector ``x`` ∈ [0, 1]^d into named parameters."""
    out: dict[str, float] = {}
    for i, tp in enumerate(scenario.params):
        v = float(np.clip(x[i], 0.0, 1.0))
        out[tp.name] = tp.min_value + v * (tp.max_value - tp.min_value)

    friction_pair = _friction_pair(scenario)
    if friction_pair is not None:
        static_param, dynamic_param = friction_pair
        static_index = next(
            i
            for i, param in enumerate(scenario.params)
            if param.name == static_param.name
        )
        dynamic_index = next(
            i
            for i, param in enumerate(scenario.params)
            if param.name == dynamic_param.name
        )
        static_unit = float(np.clip(x[static_index], 0.0, 1.0))
        dynamic_unit = float(np.clip(x[dynamic_index], 0.0, 1.0))
        static_min = max(static_param.min_value, dynamic_param.min_value)
        static_value = static_min + static_unit * (static_param.max_value - static_min)
        dynamic_max = min(dynamic_param.max_value, static_value)
        dynamic_value = dynamic_param.min_value + dynamic_unit * (
            dynamic_max - dynamic_param.min_value
        )
        out[static_param.name] = static_value
        out[dynamic_param.name] = dynamic_value
    return out


def _vector_from_params(scenario: Scenario, params: dict[str, float]) -> np.ndarray:
    """Inverse of :func:`_params_from_vector`."""
    out = np.zeros(len(scenario.params), dtype=float)
    for i, tp in enumerate(scenario.params):
        denom = max(tp.max_value - tp.min_value, 1e-12)
        out[i] = (float(params[tp.name]) - tp.min_value) / denom

    friction_pair = _friction_pair(scenario)
    if friction_pair is not None:
        static_param, dynamic_param = friction_pair
        static_index = next(
            i
            for i, param in enumerate(scenario.params)
            if param.name == static_param.name
        )
        dynamic_index = next(
            i
            for i, param in enumerate(scenario.params)
            if param.name == dynamic_param.name
        )
        static_value = float(params[static_param.name])
        dynamic_value = float(params[dynamic_param.name])
        static_min = max(static_param.min_value, dynamic_param.min_value)
        static_denom = max(static_param.max_value - static_min, 1e-12)
        dynamic_max = min(dynamic_param.max_value, static_value)
        dynamic_denom = max(dynamic_max - dynamic_param.min_value, 1e-12)
        out[static_index] = (static_value - static_min) / static_denom
        out[dynamic_index] = (dynamic_value - dynamic_param.min_value) / dynamic_denom
    return np.clip(out, 0.0, 1.0)


def _friction_pair(
    scenario: Scenario,
) -> tuple[TunableParam, TunableParam] | None:
    """Return coupled static/dynamic friction parameters when both are tuned."""
    params = scenario.param_dict()
    static_param = params.get("static_friction")
    dynamic_param = params.get("dynamic_friction")
    if static_param is None or dynamic_param is None:
        return None
    if dynamic_param.min_value > static_param.max_value:
        raise ValueError(
            "dynamic_friction minimum must not exceed static_friction maximum"
        )
    return static_param, dynamic_param


def is_botorch_available() -> bool:
    """Return True if BoTorch can be imported in this process."""
    try:
        import botorch  # type: ignore[import-not-found]  # noqa: F401
        import torch  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_optimizer(name: str) -> str:
    """Resolve ``auto`` to a concrete optimizer name.

    The issue Acceptance Criteria pins behaviour:
      * ``auto`` → ``botorch`` when installed, else
        :class:`BoTorchUnavailableError`.
      * ``botorch`` requested but missing → also
        :class:`BoTorchUnavailableError`.
      * ``random`` and ``cma-es`` are always available.
    """
    if name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer {name!r}. Supported: {sorted(SUPPORTED_OPTIMIZERS)}"
        )
    if name == OPTIMIZER_AUTO:
        if not is_botorch_available():
            raise BoTorchUnavailableError()
        return OPTIMIZER_BOTORCH
    if name == OPTIMIZER_BOTORCH and not is_botorch_available():
        raise BoTorchUnavailableError()
    return name


# ---------------------------------------------------------------------------
# Optimizer implementations
#
# Each optimizer accepts a callback `evaluate(params: dict) -> float` and
# returns nothing; it is responsible for calling `evaluate` `max_trials`
# times. The runner records each call into the trial history.
# ---------------------------------------------------------------------------


def _validate_max_trials(max_trials: int) -> None:
    if not isinstance(max_trials, int) or max_trials <= 0:
        raise ValueError(f"max_trials must be a positive integer, got {max_trials!r}")


def run_random_optimizer(
    scenario: Scenario,
    evaluate: EvaluateFn,
    *,
    max_trials: int,
    seed: int,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Uniform random search over scenario parameter ranges (developer baseline)."""
    _validate_max_trials(max_trials)
    rng = np.random.default_rng(seed)
    for _ in range(max_trials):
        if cancel_check is not None and cancel_check():
            return
        x = rng.random(len(scenario.params))
        params = _params_from_vector(scenario, x)
        evaluate(params)


def run_cma_es_optimizer(
    scenario: Scenario,
    evaluate: EvaluateFn,
    *,
    max_trials: int,
    seed: int,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """CMA-ES wrapper.

    We re-use the project's stock CMA-ES implementation but adapt it to a
    fixed trial budget instead of a wall-clock budget. The wrapper raises a
    StopIteration internally when the budget is exhausted.
    """
    _validate_max_trials(max_trials)
    from world_understanding.functions.optimization.cma_es import cma_es

    counter = {"n": 0}

    def evaluator(*, x: np.ndarray) -> float:
        if counter["n"] >= max_trials:
            # CMA-ES wraps `evaluate` calls in a loop; raise to break out.
            raise _BudgetExhausted()
        if cancel_check is not None and cancel_check():
            raise _BudgetExhausted()
        counter["n"] += 1
        # CMA-ES samples in unit-cube here (n_dims=D, bounds=(0, 1)) so we
        # can map to scenario params directly.
        params = _params_from_vector(scenario, np.asarray(x, dtype=float))
        return float(evaluate(params))

    try:
        cma_es(
            evaluate=evaluator,
            bounds=(0.0, 1.0),
            n_dims=len(scenario.params),
            # Set a generous time_budget — the trial budget is the real
            # stopping condition via _BudgetExhausted.
            time_budget=max(60.0, float(max_trials) * 5.0),
            seed=seed,
        )
    except _BudgetExhausted:
        return


def run_botorch_optimizer(
    scenario: Scenario,
    evaluate: EvaluateFn,
    *,
    max_trials: int,
    seed: int,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """BoTorch single-objective Bayesian optimization (qEI on GP).

    Lazy-imports torch / botorch so this module remains importable when the
    optional ``tuning`` extra is missing. Caller is responsible for calling
    :func:`resolve_optimizer` first if they want a clean install-hint error.
    """
    _validate_max_trials(max_trials)
    try:
        import torch  # type: ignore[import-not-found]
        from botorch.acquisition import (
            qExpectedImprovement,  # type: ignore[import-not-found]
        )
        from botorch.fit import fit_gpytorch_mll  # type: ignore[import-not-found]
        from botorch.models import SingleTaskGP  # type: ignore[import-not-found]
        from botorch.optim import optimize_acqf  # type: ignore[import-not-found]
        from gpytorch.mlls import (
            ExactMarginalLogLikelihood,  # type: ignore[import-not-found]
        )
    except ImportError as e:
        raise BoTorchUnavailableError() from e

    torch.manual_seed(seed)
    _random_module.seed(seed)
    np.random.seed(seed)

    d = len(scenario.params)

    # 1) Sobol-style initial design (fall back to uniform random when scipy
    #    is unavailable). ``min(8, max_trials)`` gives BO enough data to fit a
    #    GP without consuming the entire budget on the first iteration.
    n_init = min(max(2, d * 2), max_trials)
    rng = np.random.default_rng(seed)
    init_x = rng.random((n_init, d))

    history_x: list[list[float]] = []
    history_y: list[float] = []

    for row in init_x:
        if cancel_check is not None and cancel_check():
            return
        params = _params_from_vector(scenario, row)
        score = float(evaluate(params))
        history_x.append(row.tolist())
        history_y.append(score)

    # 2) Sequential model-based loop using qEI on a SingleTaskGP.
    n_remaining = max_trials - n_init
    bounds = torch.stack([torch.zeros(d), torch.ones(d)]).double()

    for _ in range(n_remaining):
        if cancel_check is not None and cancel_check():
            return
        x_train = torch.tensor(history_x, dtype=torch.double)
        # Negate because BoTorch maximizes by default and we minimize score.
        y_train = -torch.tensor(history_y, dtype=torch.double).unsqueeze(-1)

        try:
            gp = SingleTaskGP(x_train, y_train)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            best_f = y_train.max().item()
            acq = qExpectedImprovement(model=gp, best_f=best_f)
            candidate, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=64,
            )
            x_next = candidate.detach().cpu().numpy().reshape(-1)
        except Exception as e:
            # If GP fitting fails (small d, ill-conditioned), draw a uniform
            # sample so the run still progresses and the failure is logged.
            logger.warning(
                "BoTorch GP step failed (%s); falling back to random sample "
                "for this iteration",
                e,
            )
            x_next = rng.random(d)

        params = _params_from_vector(scenario, np.asarray(x_next, dtype=float))
        score = float(evaluate(params))
        history_x.append(np.asarray(x_next, dtype=float).tolist())
        history_y.append(score)


class _BudgetExhausted(RuntimeError):
    """Internal control-flow exception for trial-budget exits."""


def get_runner(name: str) -> Callable[..., None]:
    """Return the optimizer entry-point keyed by ``name``.

    Caller must pass a *resolved* optimizer name (i.e. ``auto`` already mapped).
    """
    if name == OPTIMIZER_BOTORCH:
        return run_botorch_optimizer
    if name == OPTIMIZER_RANDOM:
        return run_random_optimizer
    if name == OPTIMIZER_CMA_ES:
        return run_cma_es_optimizer
    raise ValueError(f"No runner for optimizer {name!r}")


__all__ = [
    "OPTIMIZER_AUTO",
    "OPTIMIZER_BOTORCH",
    "OPTIMIZER_RANDOM",
    "OPTIMIZER_CMA_ES",
    "SUPPORTED_OPTIMIZERS",
    "EvaluateFn",
    "resolve_optimizer",
    "is_botorch_available",
    "get_runner",
    "run_random_optimizer",
    "run_cma_es_optimizer",
    "run_botorch_optimizer",
]
