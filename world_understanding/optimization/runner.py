# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared trial-loop orchestration around an optimizer runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .contracts import BoundedSearchSpace, OptimizerSettings, TrialRecord
from .errors import OptimizationCancelledError

EvaluateTrial = Callable[[dict[str, float], int, int], TrialRecord]
OptimizerRunner = Callable[..., None]
OptimizerNameResolver = Callable[[str], str]
OptimizerRunnerResolver = Callable[[str], OptimizerRunner]
TrialCallback = Callable[[TrialRecord], None]


class _BudgetExhausted(Exception):
    """Stop an optimizer that calls the evaluator beyond its trial budget."""


@dataclass
class TrialRun:
    """Trials completed by one optimizer invocation."""

    history: list[TrialRecord] = field(default_factory=list)
    cancelled: bool = False

    def best(self, *, allow_failed: bool = False) -> TrialRecord | None:
        """Return the best successful trial, optionally falling back to failures."""

        successful = [trial for trial in self.history if trial.success]
        candidates = successful or (self.history if allow_failed else [])
        return min(candidates, key=lambda trial: trial.optimizer_score, default=None)


@dataclass
class TuneRun:
    """Result of a domain-neutral tuning workflow."""

    optimizer_used: str
    trials: TrialRun

    @property
    def history(self) -> list[TrialRecord]:
        """Trials in evaluation order."""

        return self.trials.history

    @property
    def cancelled(self) -> bool:
        """Whether the workflow stopped due to cancellation."""

        return self.trials.cancelled

    def best(self, *, allow_failed: bool = False) -> TrialRecord | None:
        """Return the best trial selected by the shared trial policy."""

        return self.trials.best(allow_failed=allow_failed)


class TuneWorkflow:
    """Resolve an optimizer once, then run domain-supplied evaluations.

    Constructing the workflow performs optimizer resolution without touching a
    domain backend. Agents can therefore fail fast before starting simulators,
    loading models, or preparing artifacts. The domain remains responsible for
    translating one candidate into a :class:`TrialRecord`.
    """

    def __init__(
        self,
        settings: OptimizerSettings,
        *,
        resolve_optimizer: OptimizerNameResolver | None = None,
        get_optimizer_runner: OptimizerRunnerResolver | None = None,
    ) -> None:
        if resolve_optimizer is None or get_optimizer_runner is None:
            from .optimizers import get_runner as default_get_runner
            from .optimizers import resolve_optimizer as default_resolve_optimizer

            resolve_optimizer = resolve_optimizer or default_resolve_optimizer
            get_optimizer_runner = get_optimizer_runner or default_get_runner

        if not callable(resolve_optimizer):
            raise TypeError("resolve_optimizer must be callable")
        if not callable(get_optimizer_runner):
            raise TypeError("get_optimizer_runner must be callable")

        optimizer_used = resolve_optimizer(settings.name)
        if not isinstance(optimizer_used, str) or not optimizer_used.strip():
            raise TypeError("resolve_optimizer must return a non-empty string")
        optimizer_runner = get_optimizer_runner(optimizer_used)
        if not callable(optimizer_runner):
            raise TypeError("resolved optimizer runner must be callable")

        self.settings = settings
        self.optimizer_used = optimizer_used
        self._optimizer_runner = optimizer_runner

    def run(
        self,
        *,
        search_space: BoundedSearchSpace,
        evaluate_trial: EvaluateTrial,
        cancel_check: Callable[[], bool] | None = None,
        on_trial: TrialCallback | None = None,
        on_complete: Callable[[TuneRun], None] | None = None,
        cancellation_exceptions: tuple[type[Exception], ...] = (),
    ) -> TuneRun:
        """Execute the resolved optimizer against a domain evaluation callback.

        ``evaluate_trial`` receives one per-trial seed. The domain callback owns
        execution across ``settings.replica_seeds`` and must populate
        :attr:`TrialRecord.replicas` with those replica results.
        """

        trials = run_trials(
            search_space=search_space,
            optimizer_runner=self._optimizer_runner,
            evaluate_trial=evaluate_trial,
            max_trials=self.settings.max_trials,
            seed=self.settings.seed,
            cancel_check=cancel_check,
            on_trial=on_trial,
            cancellation_exceptions=cancellation_exceptions,
        )
        result = TuneRun(optimizer_used=self.optimizer_used, trials=trials)
        if on_complete is not None:
            on_complete(result)
        return result


def _clip_candidate(
    search_space: BoundedSearchSpace,
    candidate: dict[str, float],
) -> dict[str, float]:
    clipped: dict[str, float] = {}
    for parameter in search_space.params:
        value = candidate[parameter.name]
        clip = getattr(parameter, "clip", None)
        if callable(clip):
            clipped[parameter.name] = float(clip(value))
            continue
        bounded = max(parameter.min_value, min(parameter.max_value, float(value)))
        clipped[parameter.name] = (
            float(round(bounded)) if parameter.integer else float(bounded)
        )
    return clipped


def run_trials(
    *,
    search_space: BoundedSearchSpace,
    optimizer_runner: OptimizerRunner,
    evaluate_trial: EvaluateTrial,
    max_trials: int,
    seed: int,
    cancel_check: Callable[[], bool] | None = None,
    on_trial: TrialCallback | None = None,
    cancellation_exceptions: tuple[type[Exception], ...] = (),
) -> TrialRun:
    """Run bounded trials while owning common history and cancellation behavior."""

    history: list[TrialRecord] = []
    cancelled = False

    def is_cancelled() -> bool:
        nonlocal cancelled
        if cancel_check is not None and cancel_check():
            cancelled = True
            return True
        return False

    def evaluate(candidate: dict[str, float]) -> float:
        if is_cancelled():
            raise OptimizationCancelledError("optimization cancelled by caller")
        if len(history) >= max_trials:
            raise _BudgetExhausted
        trial_index = len(history)
        clipped = _clip_candidate(search_space, candidate)
        trial = evaluate_trial(clipped, trial_index, seed + trial_index)
        if not isinstance(trial, TrialRecord):
            raise TypeError("evaluate_trial must return TrialRecord")
        history.append(trial)
        if on_trial is not None:
            on_trial(trial)
        return float(trial.optimizer_score)

    try:
        optimizer_runner(
            search_space,
            evaluate,
            max_trials=max_trials,
            seed=seed,
            cancel_check=is_cancelled,
        )
    except _BudgetExhausted:
        pass
    except Exception as error:
        if isinstance(error, OptimizationCancelledError) or isinstance(
            error,
            cancellation_exceptions,
        ):
            cancelled = True
        else:
            raise

    return TrialRun(history=history, cancelled=cancelled)


def run_tuning(
    *,
    settings: OptimizerSettings,
    search_space: BoundedSearchSpace,
    evaluate_trial: EvaluateTrial,
    resolve_optimizer: OptimizerNameResolver | None = None,
    get_optimizer_runner: OptimizerRunnerResolver | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_trial: TrialCallback | None = None,
    on_complete: Callable[[TuneRun], None] | None = None,
    cancellation_exceptions: tuple[type[Exception], ...] = (),
) -> TuneRun:
    """Construct and execute a domain-neutral tuning workflow."""

    workflow = TuneWorkflow(
        settings,
        resolve_optimizer=resolve_optimizer,
        get_optimizer_runner=get_optimizer_runner,
    )
    return workflow.run(
        search_space=search_space,
        evaluate_trial=evaluate_trial,
        cancel_check=cancel_check,
        on_trial=on_trial,
        on_complete=on_complete,
        cancellation_exceptions=cancellation_exceptions,
    )


__all__ = [
    "EvaluateTrial",
    "OptimizerNameResolver",
    "OptimizerRunner",
    "OptimizerRunnerResolver",
    "TrialCallback",
    "TrialRun",
    "TuneRun",
    "TuneWorkflow",
    "run_tuning",
    "run_trials",
]
