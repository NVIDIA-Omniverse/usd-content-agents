# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage and behavior tests for the shared optimization runtime."""

from __future__ import annotations

import builtins
import importlib
import math
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from world_understanding import optimization
from world_understanding.optimization import optimizers
from world_understanding.optimization.contracts import (
    OptimizerSettings,
    ReplicaRecord,
    TrialRecord,
    TunableParam,
    TuningObjective,
    combine_judge_scores,
)
from world_understanding.optimization.errors import (
    OptimizationCancelledError,
    OptimizationError,
    OptimizerUnavailableError,
)
from world_understanding.optimization.runner import TrialRun, run_trials


@dataclass(frozen=True)
class _BareParameter:
    name: str
    min_value: float
    max_value: float
    integer: bool = False


@dataclass(frozen=True)
class _SearchSpace:
    params: tuple[Any, ...]


def _space() -> _SearchSpace:
    return _SearchSpace(
        params=(
            TunableParam("gain", 0.0, 2.0),
            TunableParam("count", 1.0, 5.0, integer=True),
        )
    )


def test_package_root_lazily_exports_optimizer_helpers() -> None:
    assert "run_random_optimizer" in dir(optimization)
    assert optimization.run_random_optimizer is optimizers.run_random_optimizer
    assert optimization.run_random_optimizer is optimizers.run_random_optimizer

    with pytest.raises(AttributeError, match="missing_optimizer_helper"):
        optimization.__getattr__("missing_optimizer_helper")


def test_shared_judge_score_combiner_uses_normalized_sixty_forty_policy() -> None:
    assert combine_judge_scores(0.9, 0.6) == pytest.approx(0.78)
    assert optimization.combine_judge_scores(0.9, 0.6) == pytest.approx(0.78)
    assert combine_judge_scores(
        0.9,
        0.6,
        programmatic_weight=6.0,
        vlm_weight=4.0,
    ) == pytest.approx(0.78)
    assert combine_judge_scores(
        0.9,
        0.6,
        programmatic_weight=1.2e308,
        vlm_weight=8.0e307,
    ) == pytest.approx(0.78)


@pytest.mark.parametrize(
    ("programmatic", "vlm", "programmatic_weight", "vlm_weight"),
    [
        (math.nan, 0.5, 0.6, 0.4),
        (0.5, 1.1, 0.6, 0.4),
        (0.5, 0.5, -0.1, 1.1),
        (0.5, 0.5, 0.0, 0.0),
    ],
)
def test_shared_judge_score_combiner_rejects_invalid_inputs(
    programmatic: float,
    vlm: float,
    programmatic_weight: float,
    vlm_weight: float,
) -> None:
    with pytest.raises(ValueError):
        combine_judge_scores(
            programmatic,
            vlm,
            programmatic_weight=programmatic_weight,
            vlm_weight=vlm_weight,
        )


def test_refinement_loop_promotes_state_and_approves() -> None:
    loop = optimization.RefinementLoop(initial_state={"limit": 1}, max_iterations=3)

    first = loop.begin_iteration()
    assert first is not None
    assert first.iteration == 1
    assert first.max_iterations == 3
    assert first.state == {"limit": 1}
    assert first.is_last is False
    assert loop.state == {"limit": 1}
    assert loop.is_complete is False
    assert loop.termination_reason is None

    loop.continue_with({"limit": 2})
    second = loop.begin_iteration()
    assert second is not None
    assert second.iteration == 2
    assert second.state == {"limit": 2}

    loop.approve()
    assert loop.is_complete is True
    assert loop.termination_reason == "approved"
    assert loop.begin_iteration() is None


def test_refinement_loop_enforces_iteration_cap() -> None:
    loop = optimization.RefinementLoop(initial_state="initial", max_iterations=1)
    iteration = loop.begin_iteration()
    assert iteration is not None
    assert iteration.is_last is True

    loop.continue_with("unused")
    assert loop.termination_reason == "max_iterations"
    assert loop.state == "initial"
    assert loop.begin_iteration() is None


def test_refinement_loop_validates_transitions() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        optimization.RefinementLoop(initial_state=None, max_iterations=0)
    with pytest.raises(ValueError, match="max_iterations"):
        optimization.RefinementLoop(initial_state=None, max_iterations=True)

    loop = optimization.RefinementLoop(initial_state=1, max_iterations=2)
    with pytest.raises(RuntimeError, match="no active"):
        loop.continue_with(2)
    with pytest.raises(ValueError, match="reason"):
        loop.stop("")

    assert loop.begin_iteration() is not None
    with pytest.raises(RuntimeError, match="explicit transition"):
        loop.begin_iteration()
    loop.stop("cancelled")
    assert loop.termination_reason == "cancelled"

    with pytest.raises(RuntimeError, match="already complete"):
        loop.stop("error")
    with pytest.raises(RuntimeError, match="already complete"):
        loop.continue_with(2)


def test_run_refinement_promotes_state_and_records_callbacks() -> None:
    evaluated: list[tuple[int, int]] = []
    observed_records: list[int] = []
    completed: list[optimization.RefinementRun[int, int, bool]] = []

    def evaluate(iteration: optimization.RefinementIteration[int]) -> int:
        evaluated.append((iteration.iteration, iteration.state))
        return iteration.state * 2

    result = optimization.run_refinement(
        initial_state=1,
        max_iterations=3,
        evaluate=evaluate,
        judge=lambda _iteration, value: value >= 4,
        decide=lambda _iteration, _value, approved: (
            optimization.RefinementDecision.approve()
            if approved
            else optimization.RefinementDecision.continue_()
        ),
        revise=lambda iteration, _value, _approved: iteration.state + 1,
        on_iteration=lambda record: observed_records.append(record.iteration.iteration),
        on_complete=completed.append,
    )

    assert evaluated == [(1, 1), (2, 2)]
    assert observed_records == [1, 2]
    assert result.approved is True
    assert result.termination_reason == "approved"
    assert result.final_state == 2
    assert [record.next_state for record in result.records] == [2, None]
    assert completed == [result]


def test_run_refinement_handles_cap_cancellation_and_opt_in_errors() -> None:
    maxed = optimization.run_refinement(
        initial_state="initial",
        max_iterations=1,
        evaluate=lambda _iteration: "evidence",
        judge=lambda _iteration, _evidence: False,
        decide=lambda _iteration, _evidence, _judgment: (
            optimization.RefinementDecision.continue_()
        ),
        revise=lambda *_args: pytest.fail("last iteration must not revise"),
    )
    assert maxed.termination_reason == "max_iterations"
    assert maxed.final_state == "initial"
    assert maxed.records[0].next_state is None

    cancelled = optimization.run_refinement(
        initial_state=1,
        max_iterations=2,
        evaluate=lambda _iteration: pytest.fail("cancelled workflow evaluated"),
        judge=lambda *_args: pytest.fail("cancelled workflow judged"),
        decide=lambda *_args: pytest.fail("cancelled workflow decided"),
        revise=lambda *_args: pytest.fail("cancelled workflow revised"),
        cancel_check=lambda: True,
    )
    assert cancelled.cancelled is True
    assert cancelled.termination_reason == "cancelled"
    assert cancelled.records == []

    class DomainError(RuntimeError):
        pass

    failed = optimization.run_refinement(
        initial_state=1,
        max_iterations=2,
        evaluate=lambda _iteration: (_ for _ in ()).throw(DomainError("failed")),
        judge=lambda *_args: False,
        decide=lambda *_args: optimization.RefinementDecision.stop("unused"),
        revise=lambda *_args: 2,
        caught_exceptions=(DomainError,),
    )
    assert failed.termination_reason == "error"
    assert isinstance(failed.error, DomainError)

    with pytest.raises(DomainError, match="propagates"):
        optimization.run_refinement(
            initial_state=1,
            max_iterations=2,
            evaluate=lambda _iteration: (_ for _ in ()).throw(
                DomainError("propagates")
            ),
            judge=lambda *_args: False,
            decide=lambda *_args: optimization.RefinementDecision.stop("unused"),
            revise=lambda *_args: 2,
        )


def test_run_refinement_checks_cancellation_between_callbacks() -> None:
    cancelled_after_evaluation = iter((False, True))
    evaluated: list[int] = []

    after_evaluation = optimization.run_refinement(
        initial_state=1,
        max_iterations=2,
        evaluate=lambda iteration: evaluated.append(iteration.iteration),
        judge=lambda *_args: pytest.fail("cancelled workflow judged"),
        decide=lambda *_args: pytest.fail("cancelled workflow decided"),
        revise=lambda *_args: pytest.fail("cancelled workflow revised"),
        cancel_check=lambda: next(cancelled_after_evaluation),
    )

    assert evaluated == [1]
    assert after_evaluation.cancelled is True
    assert after_evaluation.records == []

    cancelled_after_judgment = iter((False, False, True))
    decided: list[int] = []

    def decide_after_judgment(
        iteration: optimization.RefinementIteration[int],
        _evaluation: str,
        _judgment: str,
    ) -> optimization.RefinementDecision:
        decided.append(iteration.iteration)
        return optimization.RefinementDecision.continue_()

    after_judgment = optimization.run_refinement(
        initial_state=1,
        max_iterations=2,
        evaluate=lambda _iteration: "evidence",
        judge=lambda _iteration, _evidence: "judgment",
        decide=decide_after_judgment,
        revise=lambda *_args: pytest.fail("cancelled workflow revised"),
        cancel_check=lambda: next(cancelled_after_judgment),
    )

    assert after_judgment.cancelled is True
    assert after_judgment.records == []
    assert decided == []

    cancelled_after_decision = iter((False, False, False, True))
    revised: list[int] = []
    after_decision = optimization.run_refinement(
        initial_state=1,
        max_iterations=2,
        evaluate=lambda _iteration: "evidence",
        judge=lambda _iteration, _evidence: "judgment",
        decide=lambda *_args: optimization.RefinementDecision.continue_(),
        revise=lambda iteration, *_args: revised.append(iteration.iteration),
        cancel_check=lambda: next(cancelled_after_decision),
    )

    assert after_decision.cancelled is True
    assert after_decision.records[0].decision == optimization.RefinementDecision.stop(
        "cancelled"
    )
    assert after_decision.records[0].next_state is None
    assert revised == []


def test_run_refinement_honors_explicit_stop_decision() -> None:
    result = optimization.run_refinement(
        initial_state=1,
        max_iterations=2,
        evaluate=lambda _iteration: "evidence",
        judge=lambda _iteration, _evidence: "judgment",
        decide=lambda *_args: optimization.RefinementDecision.stop("domain_complete"),
        revise=lambda *_args: pytest.fail("stopped workflow revised"),
    )

    assert result.termination_reason == "domain_complete"
    assert result.cancelled is False
    assert result.records[0].decision.reason == "domain_complete"


def test_run_refinement_promotes_none_as_a_valid_revised_state() -> None:
    observed_states: list[str | None] = []

    result = optimization.run_refinement(
        initial_state="initial",
        max_iterations=2,
        evaluate=lambda iteration: observed_states.append(iteration.state),
        judge=lambda *_args: None,
        decide=lambda iteration, *_args: (
            optimization.RefinementDecision.approve()
            if iteration.is_last
            else optimization.RefinementDecision.continue_()
        ),
        revise=lambda *_args: None,
    )

    assert observed_states == ["initial", None]
    assert result.final_state is None
    assert result.approved is True


def test_refinement_decisions_and_callbacks_are_validated() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        optimization.RefinementDecision("retry")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty reason"):
        optimization.RefinementDecision.stop("")
    with pytest.raises(ValueError, match="must not provide"):
        optimization.RefinementDecision("approve", "extra")

    with pytest.raises(TypeError, match="decide must return"):
        optimization.run_refinement(
            initial_state=1,
            max_iterations=1,
            evaluate=lambda _iteration: None,
            judge=lambda *_args: None,
            decide=lambda *_args: "approve",  # type: ignore[arg-type]
            revise=lambda *_args: 2,
        )

    with pytest.raises(TypeError, match="Exception classes"):
        optimization.run_refinement(
            initial_state=1,
            max_iterations=1,
            evaluate=lambda _iteration: None,
            judge=lambda *_args: None,
            decide=lambda *_args: optimization.RefinementDecision.approve(),
            revise=lambda *_args: 2,
            caught_exceptions=(RuntimeError("not a class"),),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "callback_name",
    (
        "evaluate",
        "judge",
        "decide",
        "revise",
        "cancel_check",
        "on_iteration",
        "on_complete",
    ),
)
def test_run_refinement_rejects_non_callable_callbacks(callback_name: str) -> None:
    kwargs: dict[str, Any] = {
        "initial_state": 1,
        "max_iterations": 1,
        "evaluate": lambda _iteration: None,
        "judge": lambda *_args: None,
        "decide": lambda *_args: optimization.RefinementDecision.approve(),
        "revise": lambda *_args: 2,
    }
    kwargs[callback_name] = object()

    with pytest.raises(TypeError, match=rf"^{callback_name} must be callable$"):
        optimization.run_refinement(**kwargs)


def test_run_refinement_validates_exception_and_reason_contracts() -> None:
    kwargs: dict[str, Any] = {
        "initial_state": 1,
        "max_iterations": 1,
        "evaluate": lambda _iteration: None,
        "judge": lambda *_args: None,
        "decide": lambda *_args: optimization.RefinementDecision.approve(),
        "revise": lambda *_args: 2,
    }

    with pytest.raises(TypeError, match="Exception classes"):
        optimization.run_refinement(
            **kwargs,
            caught_exceptions=[RuntimeError],  # type: ignore[arg-type]
        )

    for reason_name in ("cancellation_reason", "error_reason"):
        with pytest.raises(ValueError, match=rf"^{reason_name} must not be empty$"):
            optimization.run_refinement(**kwargs, **{reason_name: "   "})


class _FakeTensor:
    def __init__(self, data: Any) -> None:
        self.data = np.asarray(data, dtype=float)

    def __neg__(self) -> _FakeTensor:
        return _FakeTensor(-self.data)

    def double(self) -> _FakeTensor:
        return self

    def unsqueeze(self, dimension: int) -> _FakeTensor:
        return _FakeTensor(np.expand_dims(self.data, axis=dimension))

    def max(self) -> _FakeTensor:
        return _FakeTensor(self.data.max())

    def item(self) -> float:
        return float(self.data.item())

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.data


def _package_module(name: str) -> ModuleType:
    module = ModuleType(name)
    vars(module)["__path__"] = []
    return module


def _install_fake_botorch(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    torch = ModuleType("torch")
    manual_seeds: list[int] = []
    vars(torch).update(
        {
            "double": object(),
            "long": object(),
            "manual_seed": manual_seeds.append,
            "zeros": lambda dimensions: _FakeTensor(np.zeros(dimensions)),
            "ones": lambda dimensions: _FakeTensor(np.ones(dimensions)),
            "stack": lambda tensors: _FakeTensor([item.data for item in tensors]),
            "tensor": lambda data, dtype=None: _FakeTensor(data),
        }
    )

    botorch = _package_module("botorch")
    acquisition = ModuleType("botorch.acquisition")
    fit = ModuleType("botorch.fit")
    models = _package_module("botorch.models")
    transforms = _package_module("botorch.models.transforms")
    outcome = ModuleType("botorch.models.transforms.outcome")
    botorch_optim = ModuleType("botorch.optim")
    gp_calls: list[dict[str, Any]] = []

    class SingleTaskGP:
        def __init__(
            self,
            x_train: Any,
            y_train: Any,
            *,
            outcome_transform: Any,
        ) -> None:
            self.likelihood = object()
            gp_calls.append(
                {
                    "x_train": x_train,
                    "y_train": y_train,
                    "outcome_transform": outcome_transform,
                }
            )

    class LogExpectedImprovement:
        def __init__(self, *, model: Any, best_f: float) -> None:
            self.model = model
            self.best_f = best_f

    class Standardize:
        def __init__(self, *, m: int) -> None:
            self.m = m

    optimize_calls: list[dict[str, Any]] = []

    def optimize_acqf(*, bounds: _FakeTensor, **kwargs: Any) -> tuple[Any, None]:
        optimize_calls.append({"bounds": bounds, **kwargs})
        dimensions = bounds.data.shape[1]
        return _FakeTensor([[0.5] * dimensions]), None

    vars(acquisition)["LogExpectedImprovement"] = LogExpectedImprovement
    vars(fit)["fit_gpytorch_mll"] = lambda _mll: None
    vars(models)["SingleTaskGP"] = SingleTaskGP
    vars(outcome)["Standardize"] = Standardize
    vars(botorch_optim)["optimize_acqf"] = optimize_acqf

    gpytorch = _package_module("gpytorch")
    mlls = ModuleType("gpytorch.mlls")

    class ExactMarginalLogLikelihood:
        def __init__(self, likelihood: Any, model: Any) -> None:
            self.likelihood = likelihood
            self.model = model

    vars(mlls)["ExactMarginalLogLikelihood"] = ExactMarginalLogLikelihood

    modules = {
        "torch": torch,
        "botorch": botorch,
        "botorch.acquisition": acquisition,
        "botorch.fit": fit,
        "botorch.models": models,
        "botorch.models.transforms": transforms,
        "botorch.models.transforms.outcome": outcome,
        "botorch.optim": botorch_optim,
        "gpytorch": gpytorch,
        "gpytorch.mlls": mlls,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        models=models,
        gp_calls=gp_calls,
        optimize_calls=optimize_calls,
        manual_seeds=manual_seeds,
    )


def test_contract_validation_and_helpers() -> None:
    parameter = TunableParam("count", 1.0, 5.0, integer=True)
    assert parameter.clip(3.6) == 4.0
    assert parameter.clip(-1.0) == 1.0

    with pytest.raises(ValueError, match="must not be empty"):
        TunableParam("", 0.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        TunableParam("bad", 0.0, math.inf)
    with pytest.raises(ValueError, match="min_value"):
        TunableParam("bad", 2.0, 1.0)
    with pytest.raises(ValueError, match="integer bounds"):
        TunableParam("bad", 0.5, 2.0, integer=True)

    minimize = TuningObjective("error", "m")
    maximize = TuningObjective("quality", "normalized", direction="maximize")
    assert minimize.optimizer_score(2.0) == 2.0
    assert maximize.optimizer_score(2.0) == -2.0
    with pytest.raises(ValueError, match="name"):
        TuningObjective("", "m")
    with pytest.raises(ValueError, match="unit"):
        TuningObjective("error", "")
    with pytest.raises(ValueError, match="direction"):
        TuningObjective("error", "m", direction="sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="failure_penalty"):
        TuningObjective("error", "m", failure_penalty=0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "unknown"}, "unsupported optimizer"),
        ({"max_trials": True}, "max_trials"),
        ({"max_trials": 0}, "max_trials"),
        ({"seed": True}, "seed"),
        ({"seed": -1}, "seed"),
        ({"replicas": True}, "replicas"),
        ({"replicas": 0}, "replicas"),
        ({"replica_seed": True}, "replica_seed"),
        ({"replica_seed": -1}, "replica_seed"),
    ],
)
def test_optimizer_settings_validation(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        OptimizerSettings(**kwargs)


def test_optimizer_settings_seed_schedule_and_records() -> None:
    settings = OptimizerSettings(seed=7, replicas=3)
    assert settings.replica_seeds == (7, 8, 9)
    assert OptimizerSettings(seed=7, replicas=2, replica_seed=20).replica_seeds == (
        20,
        21,
    )

    replica = ReplicaRecord(
        seed=4,
        objective_value=0.25,
        success=True,
        metadata={"worker": "test"},
        artifact_metadata={"preview": {"media_type": "image/png"}},
    )
    payload = replica.to_dict()
    assert payload["metadata"] == {"worker": "test"}
    assert payload["artifact_metadata"]["preview"]["media_type"] == "image/png"
    assert "metadata" not in ReplicaRecord(1, None, False).to_dict()

    trial = TrialRecord(
        trial_index=0,
        params={"gain": 1.0},
        score=0.25,
        replicas=[replica],
    )
    assert trial.success is True
    assert trial.optimizer_score == 0.25
    assert trial.to_dict()["replicas"] == [payload]
    trial.failed = True
    assert trial.success is False


def test_vector_decoding_and_optimizer_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert optimizers.params_from_vector(
        _space(),
        np.asarray([-1.0, 0.61]),
    ) == {"gain": 0.0, "count": 3.0}
    assert optimizers.resolve_optimizer("random") == "random"
    assert optimizers.resolve_optimizer("cma-es") == "cma-es"
    with pytest.raises(ValueError, match="Unknown optimizer"):
        optimizers.resolve_optimizer("missing")

    monkeypatch.setattr(optimizers, "is_botorch_available", lambda: True)
    assert optimizers.resolve_optimizer("auto") == "botorch"
    assert optimizers.resolve_optimizer("botorch") == "botorch"
    monkeypatch.setattr(optimizers, "is_botorch_available", lambda: False)
    with pytest.raises(OptimizerUnavailableError):
        optimizers.resolve_optimizer("auto")
    with pytest.raises(OptimizerUnavailableError):
        optimizers.resolve_optimizer("botorch")

    assert optimizers.get_runner("random") is optimizers.run_random_optimizer
    assert optimizers.get_runner("cma-es") is optimizers.run_cma_es_optimizer
    assert optimizers.get_runner("botorch") is optimizers.run_botorch_optimizer
    with pytest.raises(ValueError, match="No runner"):
        optimizers.get_runner("auto")


def test_random_and_cma_es_optimizers() -> None:
    random_candidates: list[dict[str, float]] = []
    optimizers.run_random_optimizer(
        _space(),
        lambda candidate: random_candidates.append(candidate) or candidate["gain"],
        max_trials=3,
        seed=2,
    )
    assert len(random_candidates) == 3

    cancelled_calls = 0

    def cancel_random() -> bool:
        nonlocal cancelled_calls
        cancelled_calls += 1
        return True

    optimizers.run_random_optimizer(
        _space(),
        lambda _candidate: pytest.fail("cancelled random search evaluated a trial"),
        max_trials=2,
        seed=2,
        cancel_check=cancel_random,
    )
    assert cancelled_calls == 1

    fixed_candidates: list[dict[str, float]] = []
    optimizers.run_random_optimizer(
        _space(),
        lambda candidate: fixed_candidates.append(candidate) or candidate["gain"],
        max_trials=1,
        seed=2,
        candidate_sampler=lambda _space, _rng: np.asarray([0.9, 0.5]),
        candidate_feasibility=lambda _space, vector: vector[0] == 0.25,
        fixed_features={0: 0.25},
    )
    assert fixed_candidates == [{"gain": 0.5, "count": 3.0}]

    with pytest.raises(RuntimeError, match="infeasible"):
        optimizers.run_random_optimizer(
            _space(),
            lambda _candidate: pytest.fail("infeasible candidate was evaluated"),
            max_trials=1,
            seed=2,
            candidate_feasibility=lambda _space, _vector: False,
        )

    cma_candidates: list[dict[str, float]] = []
    optimizers.run_cma_es_optimizer(
        _space(),
        lambda candidate: cma_candidates.append(candidate) or candidate["gain"],
        max_trials=3,
        seed=3,
    )
    assert len(cma_candidates) == 3

    optimizers.run_cma_es_optimizer(
        _space(),
        lambda _candidate: pytest.fail("cancelled CMA-ES evaluated a trial"),
        max_trials=2,
        seed=3,
        cancel_check=lambda: True,
    )

    for runner in (
        optimizers.run_random_optimizer,
        optimizers.run_cma_es_optimizer,
    ):
        with pytest.raises(ValueError, match="positive integer"):
            runner(_space(), lambda _candidate: 0.0, max_trials=0, seed=0)


def test_cma_es_honors_cancellation_after_optimizer_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def short_cma_es(*, evaluate: Any, **_kwargs: Any) -> None:
        evaluate(x=np.asarray([0.5, 0.5], dtype=np.float64))

    cma_module = importlib.import_module(
        "world_understanding.functions.optimization.cma_es"
    )
    monkeypatch.setattr(cma_module, "cma_es", short_cma_es)
    checks = iter((False, True))
    candidates: list[dict[str, float]] = []

    optimizers.run_cma_es_optimizer(
        _space(),
        lambda candidate: candidates.append(candidate) or candidate["gain"],
        max_trials=2,
        seed=3,
        cancel_check=lambda: next(checks),
    )

    assert candidates == [{"gain": 1.0, "count": 3.0}]


def test_cma_es_proposal_cap_fills_with_a_feasible_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def infeasible_cma_es(*, evaluate: Any, **_kwargs: Any) -> None:
        for _ in range(11):
            evaluate(x=np.ones(2, dtype=np.float64))

    cma_module = importlib.import_module(
        "world_understanding.functions.optimization.cma_es"
    )
    monkeypatch.setattr(cma_module, "cma_es", infeasible_cma_es)
    candidates: list[dict[str, float]] = []

    optimizers.run_cma_es_optimizer(
        _space(),
        lambda candidate: candidates.append(candidate) or candidate["gain"],
        max_trials=1,
        seed=3,
        candidate_sampler=lambda _space, _rng: np.zeros(2, dtype=np.float64),
        candidate_feasibility=lambda _space, vector: bool(vector[0] < 0.5),
    )

    assert candidates == [{"gain": 0.0, "count": 1.0}]


def test_cma_es_converts_nonfinite_trial_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer_scores: list[float] = []

    def failed_cma_es(*, evaluate: Any, **_kwargs: Any) -> None:
        optimizer_scores.append(evaluate(x=np.zeros(2, dtype=np.float64)))
        evaluate(x=np.zeros(2, dtype=np.float64))

    cma_module = importlib.import_module(
        "world_understanding.functions.optimization.cma_es"
    )
    monkeypatch.setattr(cma_module, "cma_es", failed_cma_es)

    optimizers.run_cma_es_optimizer(
        _space(),
        lambda _candidate: float("nan"),
        max_trials=1,
        seed=3,
    )

    assert optimizer_scores == [1.0e12]


def test_botorch_availability_and_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_botorch(monkeypatch)
    assert optimizers.is_botorch_available() is True

    real_import = builtins.__import__

    def missing_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"torch", "botorch"} or name.startswith(
            ("torch.", "botorch.", "gpytorch")
        ):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_import)
    assert optimizers.is_botorch_available() is False
    with pytest.raises(OptimizerUnavailableError):
        optimizers.run_botorch_optimizer(
            _space(),
            lambda _candidate: 0.0,
            max_trials=2,
            seed=0,
        )


def test_botorch_success_fallback_and_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_botorch(monkeypatch)
    candidates: list[dict[str, float]] = []
    optimizers.run_botorch_optimizer(
        _space(),
        lambda candidate: candidates.append(candidate) or candidate["gain"],
        max_trials=5,
        seed=5,
    )
    assert len(candidates) == 5

    class BrokenGP:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("singular")

    fake.models.SingleTaskGP = BrokenGP
    fallback_candidates: list[dict[str, float]] = []
    optimizers.run_botorch_optimizer(
        _space(),
        lambda candidate: fallback_candidates.append(candidate) or candidate["gain"],
        max_trials=5,
        seed=5,
    )
    assert len(fallback_candidates) == 5

    initial_checks = 0

    def cancel_initial() -> bool:
        nonlocal initial_checks
        initial_checks += 1
        return True

    optimizers.run_botorch_optimizer(
        _space(),
        lambda _candidate: pytest.fail("cancelled BoTorch evaluated a trial"),
        max_trials=2,
        seed=5,
        cancel_check=cancel_initial,
    )
    assert initial_checks == 1

    completed = 0

    def evaluate(candidate: dict[str, float]) -> float:
        nonlocal completed
        completed += 1
        return candidate["gain"]

    optimizers.run_botorch_optimizer(
        _SearchSpace((TunableParam("gain", 0.0, 2.0),)),
        evaluate,
        max_trials=3,
        seed=5,
        cancel_check=lambda: completed >= 2,
    )
    assert completed == 2

    with pytest.raises(ValueError, match="positive integer"):
        optimizers.run_botorch_optimizer(
            _space(),
            lambda _candidate: 0.0,
            max_trials=0,
            seed=0,
        )


def test_botorch_standardizes_outcome_and_uses_log_ei(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_botorch(monkeypatch)

    optimizers.run_botorch_optimizer(
        _space(),
        lambda candidate: candidate["gain"],
        max_trials=5,
        seed=5,
    )

    assert len(fake.gp_calls) == 1
    assert fake.gp_calls[0]["outcome_transform"].m == 1
    assert len(fake.optimize_calls) == 1
    assert (
        type(fake.optimize_calls[0]["acq_function"]).__name__
        == "LogExpectedImprovement"
    )


def test_botorch_repeats_candidate_sequence_for_same_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_botorch(monkeypatch)

    def run() -> list[dict[str, float]]:
        candidates: list[dict[str, float]] = []
        optimizers.run_botorch_optimizer(
            _space(),
            lambda candidate: candidates.append(candidate.copy()) or candidate["gain"],
            max_trials=6,
            seed=37,
        )
        return candidates

    assert run() == run()
    assert fake.manual_seeds == [37, 37]


def test_botorch_shared_constraints_and_repair_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_botorch(monkeypatch)
    candidates: list[dict[str, float]] = []
    repair_calls: list[np.ndarray] = []
    constraints = (
        (
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([1.0, -1.0], dtype=np.float64),
            0.0,
        ),
    )

    optimizers.run_botorch_optimizer(
        _space(),
        lambda candidate: candidates.append(candidate) or candidate["gain"],
        max_trials=5,
        seed=5,
        candidate_sampler=lambda _space, _rng: np.asarray([0.25, 0.25]),
        candidate_repair=lambda _space, vector: repair_calls.append(vector) or None,
        inequality_constraints=constraints,
    )

    assert len(candidates) == 5
    assert len(repair_calls) == 1
    recorded = fake.optimize_calls[0]["inequality_constraints"][0]
    assert recorded[0].data.tolist() == [0.0, 1.0]
    assert recorded[1].data.tolist() == [1.0, -1.0]
    assert recorded[2] == 0.0


def test_botorch_all_dimensions_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_botorch(monkeypatch)
    space = _SearchSpace(
        (
            TunableParam("gain", 0.5, 0.5),
            TunableParam("count", 3.0, 3.0, integer=True),
        )
    )
    candidates: list[dict[str, float]] = []

    optimizers.run_botorch_optimizer(
        space,
        lambda candidate: candidates.append(candidate) or 0.0,
        max_trials=5,
        seed=5,
    )

    assert candidates == [{"gain": 0.5, "count": 3.0}] * 5
    assert fake.optimize_calls == []


def test_trial_runner_clips_records_and_selects() -> None:
    space = _SearchSpace(
        (
            _BareParameter("gain", 0.0, 1.0),
            _BareParameter("count", 1.0, 5.0, integer=True),
        )
    )
    observed: list[tuple[dict[str, float], int, int]] = []
    callbacks: list[int] = []

    def runner(_space: Any, evaluate: Any, **_kwargs: Any) -> None:
        evaluate({"gain": -2.0, "count": 2.6})
        evaluate({"gain": 0.2, "count": 4.0})

    def evaluate(
        candidate: dict[str, float],
        trial_index: int,
        seed: int,
    ) -> TrialRecord:
        observed.append((candidate, trial_index, seed))
        return TrialRecord(
            trial_index=trial_index,
            params=candidate,
            score=1.0 - candidate["gain"],
            failed=trial_index == 0,
        )

    result = run_trials(
        search_space=space,
        optimizer_runner=runner,
        evaluate_trial=evaluate,
        max_trials=2,
        seed=10,
        on_trial=lambda trial: callbacks.append(trial.trial_index),
    )
    assert observed == [
        ({"gain": 0.0, "count": 3.0}, 0, 10),
        ({"gain": 0.2, "count": 4.0}, 1, 11),
    ]
    assert callbacks == [0, 1]
    assert result.cancelled is False
    assert result.best() is result.history[1]
    assert result.best(allow_failed=True) is result.history[1]

    failed_only = TrialRun(history=[TrialRecord(0, {"gain": 0.0}, 3.0, failed=True)])
    assert failed_only.best() is None
    assert failed_only.best(allow_failed=True) is failed_only.history[0]
    assert TrialRun().best() is None


def test_trial_runner_cancellation_and_errors() -> None:
    space = _SearchSpace((TunableParam("gain", 0.0, 1.0),))

    def eager_runner(_space: Any, evaluate: Any, **_kwargs: Any) -> None:
        evaluate({"gain": 0.5})

    cancelled = run_trials(
        search_space=space,
        optimizer_runner=eager_runner,
        evaluate_trial=lambda *_args: pytest.fail("cancelled trial executed"),
        max_trials=1,
        seed=0,
        cancel_check=lambda: True,
    )
    assert cancelled.cancelled is True

    def polite_runner(
        _space: Any,
        _evaluate: Any,
        *,
        cancel_check: Any,
        **_kwargs: Any,
    ) -> None:
        assert cancel_check() is True

    polite = run_trials(
        search_space=space,
        optimizer_runner=polite_runner,
        evaluate_trial=lambda *_args: pytest.fail("cancelled trial executed"),
        max_trials=1,
        seed=0,
        cancel_check=lambda: True,
    )
    assert polite.cancelled is True

    class DomainCancelled(RuntimeError):
        pass

    custom = run_trials(
        search_space=space,
        optimizer_runner=eager_runner,
        evaluate_trial=lambda *_args: (_ for _ in ()).throw(DomainCancelled()),
        max_trials=1,
        seed=0,
        cancellation_exceptions=(DomainCancelled,),
    )
    assert custom.cancelled is True

    with pytest.raises(RuntimeError, match="task failed"):
        run_trials(
            search_space=space,
            optimizer_runner=eager_runner,
            evaluate_trial=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("task failed")
            ),
            max_trials=1,
            seed=0,
        )

    def excessive_runner(_space: Any, evaluate: Any, **_kwargs: Any) -> None:
        evaluate({"gain": 0.1})
        evaluate({"gain": 0.2})

    exhausted = run_trials(
        search_space=space,
        optimizer_runner=excessive_runner,
        evaluate_trial=lambda candidate, index, _seed: TrialRecord(
            index,
            candidate,
            candidate["gain"],
        ),
        max_trials=1,
        seed=0,
    )
    assert exhausted.cancelled is False
    assert [trial.score for trial in exhausted.history] == [0.1]


def test_tune_workflow_resolves_once_and_runs_domain_callbacks() -> None:
    events: list[tuple[str, object]] = []
    completed: list[optimization.TuneRun] = []

    def resolve(name: str) -> str:
        events.append(("resolve", name))
        return "resolved"

    def runner(
        _space: Any,
        evaluate: Any,
        *,
        max_trials: int,
        seed: int,
        **_kwargs: Any,
    ) -> None:
        events.append(("run", (max_trials, seed)))
        evaluate({"gain": -1.0, "count": 2.6})
        evaluate({"gain": 0.5, "count": 4.0})

    def get_runner(name: str) -> Any:
        events.append(("get_runner", name))
        return runner

    workflow = optimization.TuneWorkflow(
        OptimizerSettings(name="random", max_trials=2, seed=7),
        resolve_optimizer=resolve,
        get_optimizer_runner=get_runner,
    )
    assert events == [("resolve", "random"), ("get_runner", "resolved")]

    result = workflow.run(
        search_space=_space(),
        evaluate_trial=lambda candidate, index, _seed: TrialRecord(
            trial_index=index,
            params=candidate,
            score=2.0 - candidate["gain"],
            failed=index == 0,
        ),
        on_complete=completed.append,
    )

    assert events[-1] == ("run", (2, 7))
    assert result.optimizer_used == "resolved"
    assert result.cancelled is False
    assert result.best() is result.history[1]
    assert result.history[0].params == {"gain": 0.0, "count": 3.0}
    assert completed == [result]


def test_tune_workflow_uses_default_optimizer_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []

    def resolve(name: str) -> str:
        events.append(("resolve", name))
        return "resolved"

    def runner(_space: Any, _evaluate: Any, **_kwargs: Any) -> None:
        return None

    def get_runner(name: str) -> Any:
        events.append(("get_runner", name))
        return runner

    monkeypatch.setattr(optimizers, "resolve_optimizer", resolve)
    monkeypatch.setattr(optimizers, "get_runner", get_runner)

    workflow = optimization.TuneWorkflow(OptimizerSettings(name="random"))

    assert workflow.optimizer_used == "resolved"
    assert events == [("resolve", "random"), ("get_runner", "resolved")]


def test_run_tuning_function_and_workflow_boundaries() -> None:
    def runner(_space: Any, evaluate: Any, **_kwargs: Any) -> None:
        evaluate({"gain": 0.25, "count": 2.0})

    result = optimization.run_tuning(
        settings=OptimizerSettings(name="random", max_trials=1, seed=3),
        search_space=_space(),
        evaluate_trial=lambda candidate, index, _seed: TrialRecord(
            index,
            candidate,
            candidate["gain"],
        ),
        resolve_optimizer=lambda _name: "custom-adapter",
        get_optimizer_runner=lambda _name: runner,
    )
    assert result.optimizer_used == "custom-adapter"
    assert result.best() is result.history[0]

    with pytest.raises(TypeError, match="non-empty string"):
        optimization.TuneWorkflow(
            OptimizerSettings(name="random"),
            resolve_optimizer=lambda _name: "",
            get_optimizer_runner=lambda _name: runner,
        )
    with pytest.raises(TypeError, match="runner must be callable"):
        optimization.TuneWorkflow(
            OptimizerSettings(name="random"),
            resolve_optimizer=lambda name: name,
            get_optimizer_runner=lambda _name: object(),  # type: ignore[arg-type]
        )

    workflow = optimization.TuneWorkflow(
        OptimizerSettings(name="random", max_trials=1),
        resolve_optimizer=lambda name: name,
        get_optimizer_runner=lambda _name: runner,
    )
    with pytest.raises(TypeError, match="must return TrialRecord"):
        workflow.run(
            search_space=_space(),
            evaluate_trial=lambda *_args: object(),  # type: ignore[arg-type]
        )


def test_public_exports_and_error_messages() -> None:
    assert optimization.run_trials is run_trials
    assert callable(optimization.run_tuning)
    assert callable(optimization.run_refinement)
    assert optimization.finite_failed_trial_penalty([]) == 1.0e12
    assert issubclass(OptimizationCancelledError, OptimizationError)
    assert "Requested optimizer is unavailable" in str(OptimizerUnavailableError())
    assert str(OptimizerUnavailableError("custom")) == "custom"
