# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared optimizer dispatch for bounded numeric search spaces."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .contracts import (
    OPTIMIZER_AUTO,
    OPTIMIZER_BOTORCH,
    OPTIMIZER_CMA_ES,
    OPTIMIZER_RANDOM,
    SUPPORTED_OPTIMIZERS,
    BoundedSearchSpace,
)
from .errors import OptimizerUnavailableError
from .registry import get_registered_optimizer, list_registered_optimizer_names

logger = logging.getLogger(__name__)

EvaluateFn = Callable[[dict[str, float]], float]
CandidateDecoder = Callable[[BoundedSearchSpace, np.ndarray], dict[str, float]]
CandidateSampler = Callable[
    [BoundedSearchSpace, np.random.Generator],
    NDArray[np.float64],
]
CandidateFeasibility = Callable[[BoundedSearchSpace, np.ndarray], bool]
CandidateRepair = Callable[
    [BoundedSearchSpace, np.ndarray],
    NDArray[np.float64] | None,
]
LinearInequality = tuple[NDArray[np.int64], NDArray[np.float64], float]

_CMA_ES_PROPOSAL_MULTIPLIER = 10
_DEFAULT_FAILED_TRIAL_PENALTY = 1.0e12
_INFEASIBLE_PROPOSAL_PENALTY = float(np.finfo(np.float64).max)


def params_from_vector(
    search_space: BoundedSearchSpace,
    vector: np.ndarray,
) -> dict[str, float]:
    """Convert a unit-cube vector into named bounded parameters."""

    params: dict[str, float] = {}
    for index, parameter in enumerate(search_space.params):
        unit_value = float(np.clip(vector[index], 0.0, 1.0))
        value = parameter.min_value + unit_value * (
            parameter.max_value - parameter.min_value
        )
        if bool(getattr(parameter, "integer", False)):
            value = float(round(value))
        params[parameter.name] = float(
            np.clip(value, parameter.min_value, parameter.max_value)
        )
    return params


def vector_from_params(
    search_space: BoundedSearchSpace,
    params: dict[str, float],
) -> NDArray[np.float64]:
    """Convert named bounded parameters into a clipped unit-cube vector."""

    vector = np.zeros(len(search_space.params), dtype=np.float64)
    for index, parameter in enumerate(search_space.params):
        denominator = max(parameter.max_value - parameter.min_value, 1e-12)
        vector[index] = (
            float(params[parameter.name]) - parameter.min_value
        ) / denominator
    return np.clip(vector, 0.0, 1.0)


def _sample_unit_vector(
    search_space: BoundedSearchSpace,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    return np.asarray(rng.random(len(search_space.params)), dtype=np.float64)


def _always_feasible(
    _search_space: BoundedSearchSpace,
    _vector: np.ndarray,
) -> bool:
    return True


def _clip_candidate(
    _search_space: BoundedSearchSpace,
    vector: np.ndarray,
) -> NDArray[np.float64]:
    return np.clip(np.asarray(vector, dtype=np.float64), 0.0, 1.0)


def _next_finite_penalty(value: float, scale: float) -> float:
    """Return a finite value strictly worse than ``value`` for minimization."""

    limit = np.finfo(np.float64).max / 4.0
    increment = max(1.0, abs(value), scale)
    return float(limit if value >= limit - increment else value + increment)


def finite_failed_trial_penalty(observed_scores: list[float]) -> float:
    """Return a finite penalty for a trial with a non-finite score."""

    finite_scores = [score for score in observed_scores if np.isfinite(score)]
    if not finite_scores:
        return _DEFAULT_FAILED_TRIAL_PENALTY
    worst = max(finite_scores)
    scale = max(1.0, *(abs(score) for score in finite_scores))
    return _next_finite_penalty(worst, scale)


def _finite_infeasible_penalty() -> float:
    """Return the maximum finite minimization penalty."""

    return _INFEASIBLE_PROPOSAL_PENALTY


def _resolved_fixed_features(
    search_space: BoundedSearchSpace,
    fixed_features: Mapping[int, float] | None,
) -> dict[int, float]:
    """Combine zero-width dimensions with domain-provided unit coordinates."""

    resolved = {
        index: 0.0
        for index, parameter in enumerate(search_space.params)
        if parameter.min_value == parameter.max_value
    }
    if fixed_features:
        for index, value in fixed_features.items():
            if index < 0 or index >= len(search_space.params):
                raise ValueError(f"fixed feature index out of range: {index}")
            resolved[index] = float(np.clip(value, 0.0, 1.0))
    return resolved


def _apply_fixed_features(
    vector: np.ndarray,
    fixed_features: Mapping[int, float],
) -> NDArray[np.float64]:
    """Return a unit vector with fixed coordinates applied."""

    candidate = np.asarray(vector, dtype=np.float64).copy()
    for index, value in fixed_features.items():
        candidate[index] = value
    return candidate


def is_botorch_available() -> bool:
    """Return whether BoTorch and Torch can be imported."""

    try:
        import botorch  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def get_supported_optimizer_names() -> tuple[str, ...]:
    """Return built-in and installed extension optimizer names."""

    return tuple(sorted((*SUPPORTED_OPTIMIZERS, *list_registered_optimizer_names())))


def resolve_optimizer(name: str) -> str:
    """Resolve an optimizer name without silently changing algorithms."""

    if name == OPTIMIZER_AUTO:
        if not is_botorch_available():
            raise OptimizerUnavailableError()
        return OPTIMIZER_BOTORCH
    if name == OPTIMIZER_BOTORCH and not is_botorch_available():
        raise OptimizerUnavailableError()
    if name in SUPPORTED_OPTIMIZERS:
        return name

    plugin = get_registered_optimizer(name)
    if plugin is None:
        raise ValueError(
            f"Unknown optimizer {name!r}. "
            f"Supported: {list(get_supported_optimizer_names())}"
        )
    if not plugin.is_available():
        raise OptimizerUnavailableError(plugin.unavailable_message)
    return name


def _validate_max_trials(max_trials: int) -> None:
    if not isinstance(max_trials, int) or max_trials <= 0:
        raise ValueError(f"max_trials must be a positive integer, got {max_trials!r}")


def run_random_optimizer(
    search_space: BoundedSearchSpace,
    evaluate: EvaluateFn,
    *,
    max_trials: int,
    seed: int,
    cancel_check: Callable[[], bool] | None = None,
    candidate_decoder: CandidateDecoder = params_from_vector,
    candidate_sampler: CandidateSampler = _sample_unit_vector,
    candidate_feasibility: CandidateFeasibility = _always_feasible,
    fixed_features: Mapping[int, float] | None = None,
) -> None:
    """Run uniform random search over the supplied bounds."""

    _validate_max_trials(max_trials)
    rng = np.random.default_rng(seed)
    resolved_fixed_features = _resolved_fixed_features(search_space, fixed_features)
    for _ in range(max_trials):
        if cancel_check is not None and cancel_check():
            return
        vector = _apply_fixed_features(
            candidate_sampler(search_space, rng),
            resolved_fixed_features,
        )
        if not candidate_feasibility(search_space, vector):
            raise RuntimeError("Candidate sampler returned an infeasible vector")
        evaluate(candidate_decoder(search_space, vector))


def run_cma_es_optimizer(
    search_space: BoundedSearchSpace,
    evaluate: EvaluateFn,
    *,
    max_trials: int,
    seed: int,
    cancel_check: Callable[[], bool] | None = None,
    candidate_decoder: CandidateDecoder = params_from_vector,
    candidate_sampler: CandidateSampler = _sample_unit_vector,
    candidate_feasibility: CandidateFeasibility = _always_feasible,
    fixed_features: Mapping[int, float] | None = None,
) -> None:
    """Run CMA-ES with a fixed evaluation budget."""

    _validate_max_trials(max_trials)
    from world_understanding.functions.optimization.cma_es import cma_es

    evaluations = 0
    proposals = 0
    infeasible = 0
    failed = 0
    observed_scores: list[float] = []
    max_proposals = max_trials * _CMA_ES_PROPOSAL_MULTIPLIER
    resolved_fixed_features = _resolved_fixed_features(search_space, fixed_features)

    def evaluator(*, x: np.ndarray) -> float:
        nonlocal evaluations, failed, infeasible, proposals
        if evaluations >= max_trials:
            raise _BudgetExhausted("trial_budget")
        if cancel_check is not None and cancel_check():
            raise _OptimizerCancelled
        if proposals >= max_proposals:
            raise _BudgetExhausted("proposal_cap")
        proposals += 1
        vector = _apply_fixed_features(
            np.asarray(x, dtype=np.float64),
            resolved_fixed_features,
        )
        if not candidate_feasibility(search_space, vector):
            infeasible += 1
            return _finite_infeasible_penalty()
        evaluations += 1
        candidate = candidate_decoder(search_space, vector)
        score = float(evaluate(candidate))
        if np.isfinite(score):
            observed_scores.append(score)
            return score
        failed += 1
        return finite_failed_trial_penalty(observed_scores)

    stop_reason = "time_budget"
    try:
        cma_es(
            evaluate=evaluator,
            bounds=(0.0, 1.0),
            n_dims=len(search_space.params),
            time_budget=float("inf"),
            seed=seed,
        )
    except _OptimizerCancelled:
        return
    except _BudgetExhausted as error:
        stop_reason = error.reason

    if cancel_check is not None and cancel_check():
        return
    if evaluations < max_trials:
        if stop_reason == "proposal_cap":
            logger.warning(
                "CMA-ES reached its %d-proposal cap with %d/%d feasible "
                "evaluations (%d infeasible, %d failed); filling the remaining "
                "trial budget with direct feasible samples",
                max_proposals,
                evaluations,
                max_trials,
                infeasible,
                failed,
            )
        else:
            logger.warning(
                "CMA-ES stopped before the trial budget with %d/%d evaluations "
                "after %d proposals (%d infeasible, %d failed); filling the "
                "remaining trial budget with direct feasible samples",
                evaluations,
                max_trials,
                proposals,
                infeasible,
                failed,
            )
        rng = np.random.default_rng(seed)
        while evaluations < max_trials:
            if cancel_check is not None and cancel_check():
                return
            vector = _apply_fixed_features(
                candidate_sampler(search_space, rng),
                resolved_fixed_features,
            )
            if not candidate_feasibility(search_space, vector):
                raise RuntimeError("Candidate sampler returned an infeasible vector")
            evaluations += 1
            evaluate(candidate_decoder(search_space, vector))


def run_botorch_optimizer(
    search_space: BoundedSearchSpace,
    evaluate: EvaluateFn,
    *,
    max_trials: int,
    seed: int,
    cancel_check: Callable[[], bool] | None = None,
    candidate_decoder: CandidateDecoder = params_from_vector,
    candidate_sampler: CandidateSampler = _sample_unit_vector,
    candidate_feasibility: CandidateFeasibility = _always_feasible,
    candidate_repair: CandidateRepair = _clip_candidate,
    inequality_constraints: Sequence[LinearInequality] | None = None,
    fixed_features: Mapping[int, float] | None = None,
) -> None:
    """Run sequential single-objective Bayesian optimization."""

    _validate_max_trials(max_trials)
    try:
        import torch
        from botorch.acquisition import LogExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as error:
        raise OptimizerUnavailableError() from error

    torch.manual_seed(seed)

    dimensions = len(search_space.params)
    resolved_fixed_features = _resolved_fixed_features(search_space, fixed_features)
    initial_count = min(max(2, dimensions * 2), max_trials)
    rng = np.random.default_rng(seed)
    initial_vectors = np.asarray(
        [candidate_sampler(search_space, rng) for _ in range(initial_count)],
        dtype=np.float64,
    )
    for index, value in resolved_fixed_features.items():
        initial_vectors[:, index] = value

    history_x: list[list[float]] = []
    history_y: list[float] = []
    observed_scores: list[float] = []

    for vector in initial_vectors:
        if cancel_check is not None and cancel_check():
            return
        score = float(evaluate(candidate_decoder(search_space, vector)))
        history_x.append(vector.tolist())
        if np.isfinite(score):
            observed_scores.append(score)
            history_y.append(score)
        else:
            history_y.append(finite_failed_trial_penalty(observed_scores))

    remaining = max_trials - initial_count
    if len(resolved_fixed_features) == dimensions:
        fixed_vector = _apply_fixed_features(
            candidate_sampler(search_space, rng),
            resolved_fixed_features,
        )
        if not candidate_feasibility(search_space, fixed_vector):
            raise RuntimeError("Fixed features produced an infeasible vector")
        for _ in range(remaining):
            if cancel_check is not None and cancel_check():
                return
            evaluate(candidate_decoder(search_space, fixed_vector))
        return

    bounds = torch.stack([torch.zeros(dimensions), torch.ones(dimensions)]).double()
    botorch_constraints = None
    if inequality_constraints:
        botorch_constraints = [
            (
                torch.tensor(indices, dtype=torch.long),
                torch.tensor(coefficients, dtype=torch.double),
                float(rhs),
            )
            for indices, coefficients, rhs in inequality_constraints
        ]

    for _ in range(remaining):
        if cancel_check is not None and cancel_check():
            return
        x_train = torch.tensor(history_x, dtype=torch.double)
        y_train = -torch.tensor(history_y, dtype=torch.double).unsqueeze(-1)

        try:
            gp = SingleTaskGP(
                x_train,
                y_train,
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            acquisition = LogExpectedImprovement(
                model=gp,
                best_f=y_train.max().item(),
            )
            candidate, _ = optimize_acqf(
                acq_function=acquisition,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=64,
                inequality_constraints=botorch_constraints,
                fixed_features=resolved_fixed_features or None,
            )
            raw_candidate = candidate.detach().cpu().numpy().reshape(-1)
        except Exception as error:
            logger.warning(
                "BoTorch GP step failed (%s); falling back to random sample "
                "for this iteration",
                error,
            )
            next_vector = _apply_fixed_features(
                candidate_sampler(search_space, rng),
                resolved_fixed_features,
            )
        else:
            repaired = candidate_repair(search_space, raw_candidate)
            if repaired is None or not candidate_feasibility(search_space, repaired):
                logger.warning(
                    "BoTorch returned an infeasible candidate; falling back "
                    "to a direct feasible sample"
                )
                next_vector = _apply_fixed_features(
                    candidate_sampler(search_space, rng),
                    resolved_fixed_features,
                )
            else:
                next_vector = _apply_fixed_features(
                    repaired,
                    resolved_fixed_features,
                )

        if not candidate_feasibility(search_space, next_vector):
            raise RuntimeError(
                "Optimizer candidate remained infeasible after sampling or repair"
            )

        score = float(
            evaluate(
                candidate_decoder(
                    search_space,
                    np.asarray(next_vector, dtype=float),
                )
            )
        )
        history_x.append(np.asarray(next_vector, dtype=float).tolist())
        if np.isfinite(score):
            observed_scores.append(score)
            history_y.append(score)
        else:
            history_y.append(finite_failed_trial_penalty(observed_scores))


class _BudgetExhausted(RuntimeError):
    """Internal control-flow exception for CMA-ES budget exits."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _OptimizerCancelled(Exception):
    """Internal control-flow exception for cooperative cancellation."""


def get_runner(name: str) -> Callable[..., None]:
    """Return the optimizer runner for a resolved name."""

    if name == OPTIMIZER_BOTORCH:
        return run_botorch_optimizer
    if name == OPTIMIZER_RANDOM:
        return run_random_optimizer
    if name == OPTIMIZER_CMA_ES:
        return run_cma_es_optimizer
    plugin = get_registered_optimizer(name)
    if plugin is not None:
        return plugin.runner
    raise ValueError(f"No runner for optimizer {name!r}")


__all__ = [
    "CandidateDecoder",
    "CandidateFeasibility",
    "CandidateRepair",
    "CandidateSampler",
    "EvaluateFn",
    "LinearInequality",
    "OPTIMIZER_AUTO",
    "OPTIMIZER_BOTORCH",
    "OPTIMIZER_CMA_ES",
    "OPTIMIZER_RANDOM",
    "SUPPORTED_OPTIMIZERS",
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
]
