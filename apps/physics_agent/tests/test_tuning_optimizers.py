# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for optimizer dispatch + BoTorch availability handling."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
from world_understanding.optimization import registry

from physics_agent.tuning import optimizers
from physics_agent.tuning.errors import BoTorchUnavailableError, TuningError
from physics_agent.tuning.optimizers import (
    OPTIMIZER_AUTO,
    OPTIMIZER_BOTORCH,
    OPTIMIZER_CMA_ES,
    OPTIMIZER_RANDOM,
    SUPPORTED_OPTIMIZERS,
    _params_from_vector,
    _vector_from_params,
    get_runner,
    get_supported_optimizer_names,
    is_botorch_available,
    resolve_optimizer,
    run_botorch_optimizer,
    run_cma_es_optimizer,
    run_random_optimizer,
)
from physics_agent.tuning.scenario import parse_scenario
from physics_agent.tuning.types import Scenario


def _scenario_2d():
    return parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "mass_scale", "min": 0.5, "max": 2.0},
                {"name": "static_friction", "min": 0.0, "max": 1.0},
            ],
        }
    )


def _friction_scenario() -> Scenario:
    return parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "static_friction", "min": 0.05, "max": 1.5},
                {"name": "dynamic_friction", "min": 0.05, "max": 1.5},
            ],
        }
    )


def _unequal_friction_scenario() -> Scenario:
    return parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "dynamic_friction", "min": 0.2, "max": 0.9},
                {"name": "mass_scale", "min": 0.5, "max": 2.0},
                {"name": "static_friction", "min": 0.1, "max": 0.6},
            ],
        }
    )


def _touching_friction_scenario() -> Scenario:
    return parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "static_friction", "min": 0.05, "max": 0.5},
                {"name": "dynamic_friction", "min": 0.5, "max": 1.5},
            ],
        }
    )


def _touching_friction_with_mass_scenario() -> Scenario:
    return parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "static_friction", "min": 0.05, "max": 0.5},
                {"name": "dynamic_friction", "min": 0.5, "max": 1.5},
                {"name": "mass_scale", "min": 0.5, "max": 2.0},
            ],
        }
    )


class _FakeTensor:
    def __init__(self, data):
        self.data = optimizers.np.asarray(data, dtype=float)

    def __neg__(self):
        return _FakeTensor(-self.data)

    def double(self):
        return self

    def unsqueeze(self, dim: int):
        return _FakeTensor(optimizers.np.expand_dims(self.data, axis=dim))

    def max(self):
        return _FakeTensor(self.data.max())

    def item(self) -> float:
        return float(self.data.item())

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


def _package_module(name: str) -> ModuleType:
    module = ModuleType(name)
    vars(module)["__path__"] = []
    return module


def _install_fake_botorch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate: list[float] | None = None,
) -> SimpleNamespace:
    torch = ModuleType("torch")
    torch.double = object()
    torch.long = object()
    torch.manual_seed = lambda _seed: None
    torch.zeros = lambda d: _FakeTensor(optimizers.np.zeros(d))
    torch.ones = lambda d: _FakeTensor(optimizers.np.ones(d))
    torch.stack = lambda tensors: _FakeTensor([t.data for t in tensors])
    torch.tensor = lambda data, dtype=None: _FakeTensor(data)

    botorch = _package_module("botorch")
    acquisition = ModuleType("botorch.acquisition")
    fit = ModuleType("botorch.fit")
    models = _package_module("botorch.models")
    transforms = _package_module("botorch.models.transforms")
    outcome = ModuleType("botorch.models.transforms.outcome")
    optim = ModuleType("botorch.optim")
    optimize_calls: list[dict[str, object]] = []
    gp_training_sets: list[tuple[_FakeTensor, _FakeTensor]] = []

    class SingleTaskGP:
        def __init__(
            self,
            x_train: _FakeTensor,
            y_train: _FakeTensor,
            *,
            outcome_transform: object,
        ) -> None:
            self.likelihood = object()
            gp_training_sets.append((x_train, y_train))

    class LogExpectedImprovement:
        def __init__(self, *, model: object, best_f: float) -> None:
            self.model = model
            self.best_f = best_f

    class Standardize:
        def __init__(self, *, m: int) -> None:
            self.m = m

    def optimize_acqf(*, bounds, **kwargs):
        optimize_calls.append(dict(kwargs))
        d = bounds.data.shape[1]
        values = list(candidate if candidate is not None else [0.5] * d)
        for index, value in (kwargs.get("fixed_features") or {}).items():
            values[index] = value
        return _FakeTensor([values]), None

    acquisition.LogExpectedImprovement = LogExpectedImprovement
    fit.fit_gpytorch_mll = lambda _mll: None
    models.SingleTaskGP = SingleTaskGP
    outcome.Standardize = Standardize
    optim.optimize_acqf = optimize_acqf
    botorch.acquisition = acquisition
    botorch.fit = fit
    botorch.models = models
    botorch.optim = optim

    gpytorch = _package_module("gpytorch")
    mlls = ModuleType("gpytorch.mlls")

    class ExactMarginalLogLikelihood:
        def __init__(self, likelihood, model):
            self.likelihood = likelihood
            self.model = model

    mlls.ExactMarginalLogLikelihood = ExactMarginalLogLikelihood
    gpytorch.mlls = mlls

    modules = {
        "torch": torch,
        "botorch": botorch,
        "botorch.acquisition": acquisition,
        "botorch.fit": fit,
        "botorch.models": models,
        "botorch.models.transforms": transforms,
        "botorch.models.transforms.outcome": outcome,
        "botorch.optim": optim,
        "gpytorch": gpytorch,
        "gpytorch.mlls": mlls,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return SimpleNamespace(
        torch=torch,
        botorch=botorch,
        acquisition=acquisition,
        fit=fit,
        models=models,
        optim=optim,
        optimize_calls=optimize_calls,
        gp_training_sets=gp_training_sets,
        gpytorch=gpytorch,
        mlls=mlls,
    )


def test_supported_optimizers_canonical_set() -> None:
    assert OPTIMIZER_AUTO in SUPPORTED_OPTIMIZERS
    assert OPTIMIZER_BOTORCH in SUPPORTED_OPTIMIZERS
    assert OPTIMIZER_RANDOM in SUPPORTED_OPTIMIZERS
    assert OPTIMIZER_CMA_ES in SUPPORTED_OPTIMIZERS


def test_installed_optimizer_extension_is_dispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "_optimizer_plugins", {})
    monkeypatch.setattr(registry, "_optimizer_plugins_scanned", False)
    monkeypatch.setattr(registry, "_loaded_optimizer_plugins", set())
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **_kwargs: ())

    def runner(*_args, **_kwargs) -> None:
        return None

    registry.register_optimizer("test-remote", runner)

    assert "test-remote" in get_supported_optimizer_names()
    assert resolve_optimizer("test-remote") == "test-remote"
    assert get_runner("test-remote") is runner


def test_unavailable_optimizer_extension_uses_tuning_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "_optimizer_plugins", {})
    monkeypatch.setattr(registry, "_optimizer_plugins_scanned", False)
    monkeypatch.setattr(registry, "_loaded_optimizer_plugins", set())
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **_kwargs: ())
    registry.register_optimizer(
        "test-remote",
        lambda *_args, **_kwargs: None,
        is_available=lambda: False,
        unavailable_message="test remote runtime is unavailable",
    )

    with pytest.raises(TuningError, match="test remote runtime is unavailable"):
        resolve_optimizer("test-remote")


def test_resolve_random_passthrough() -> None:
    assert resolve_optimizer(OPTIMIZER_RANDOM) == OPTIMIZER_RANDOM


def test_resolve_cma_es_passthrough() -> None:
    assert resolve_optimizer(OPTIMIZER_CMA_ES) == OPTIMIZER_CMA_ES


def test_resolve_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown optimizer"):
        resolve_optimizer("annealing")


def test_resolve_auto_when_botorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto must hard-error to BoTorchUnavailableError — no silent random fallback."""
    monkeypatch.setattr(optimizers, "is_botorch_available", lambda: False)
    with pytest.raises(BoTorchUnavailableError) as ei:
        resolve_optimizer(OPTIMIZER_AUTO)
    msg = str(ei.value)
    # Exact install hint must be surfaced — part of the issue Acceptance Criteria.
    assert "BoTorch optimizer requires the tuning extra" in msg
    assert 'uv pip install -e "apps/physics_agent[tuning]"' in msg


def test_resolve_botorch_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optimizers, "is_botorch_available", lambda: False)
    with pytest.raises(BoTorchUnavailableError):
        resolve_optimizer(OPTIMIZER_BOTORCH)


def test_resolve_auto_when_botorch_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optimizers, "is_botorch_available", lambda: True)
    assert resolve_optimizer(OPTIMIZER_AUTO) == OPTIMIZER_BOTORCH


def test_vector_param_helpers_clip_and_invert() -> None:
    sc = _scenario_2d()

    params = _params_from_vector(sc, optimizers.np.asarray([-1.0, 2.0]))
    assert params == {"mass_scale": 0.5, "static_friction": 1.0}

    vector = _vector_from_params(
        sc,
        {"mass_scale": 2.5, "static_friction": -1.0},
    )
    assert vector.tolist() == [1.0, 0.0]


def test_vector_mapping_keeps_friction_dimensions_independent() -> None:
    scenario = _friction_scenario()
    vector = optimizers.np.asarray([0.0, 1.0])

    params = _params_from_vector(scenario, vector)

    assert params == {"static_friction": 0.05, "dynamic_friction": 1.5}
    assert _vector_from_params(scenario, params) == pytest.approx(vector)


def test_friction_inequality_uses_physical_ranges_and_parameter_order() -> None:
    constraint = optimizers._friction_inequality_spec(_unequal_friction_scenario())

    assert constraint is not None
    indices, coefficients, rhs = constraint
    assert indices.tolist() == [2, 0]
    assert coefficients.tolist() == pytest.approx([0.5, -0.7])
    assert rhs == pytest.approx(0.1)


def test_friction_constraint_does_not_apply_to_generic_search_space() -> None:
    search_space = SimpleNamespace(
        params=[
            SimpleNamespace(
                name="static_friction",
                min_value=1.0,
                max_value=3.0,
                integer=False,
            ),
            SimpleNamespace(
                name="dynamic_friction",
                min_value=4.0,
                max_value=6.0,
                integer=False,
            ),
        ]
    )
    samples: list[dict[str, float]] = []

    run_random_optimizer(
        search_space,
        lambda params: samples.append(params) or 0.0,
        max_trials=25,
        seed=17,
    )

    assert len(samples) == 25
    assert all(
        sample["dynamic_friction"] > sample["static_friction"] for sample in samples
    )


def test_friction_feasibility_rejects_numerical_constraint_overshoot() -> None:
    vector = optimizers.np.asarray(
        [0.5, optimizers.np.nextafter(0.5, 1.0)],
        dtype=float,
    )

    assert not optimizers._is_friction_feasible(_friction_scenario(), vector)


def test_numerical_friction_overshoot_is_repaired_before_evaluation() -> None:
    scenario = _friction_scenario()
    vector = optimizers.np.asarray(
        [0.5, optimizers.np.nextafter(0.5, 1.0)],
        dtype=float,
    )

    repaired = optimizers._repair_numerical_friction_overshoot(scenario, vector)

    assert repaired is not None
    assert optimizers._is_friction_feasible(scenario, repaired)
    pair = optimizers._friction_pair(scenario)
    assert pair is not None
    assert repaired == pytest.approx(
        vector,
        abs=optimizers._friction_tolerance(pair),
    )


def test_material_friction_violation_is_not_repaired() -> None:
    scenario = _friction_scenario()

    repaired = optimizers._repair_numerical_friction_overshoot(
        scenario,
        optimizers.np.asarray([0.25, 0.75]),
    )

    assert repaired is None


def test_independent_round_trip_preserves_mixed_parameters() -> None:
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "mass_scale", "min": 0.5, "max": 2.0},
                {"name": "dynamic_friction", "min": 0.05, "max": 1.5},
                {"name": "restitution", "min": 0.0, "max": 1.0},
                {"name": "static_friction", "min": 0.05, "max": 1.5},
            ],
        }
    )
    vector = optimizers.np.asarray([0.25, 0.75, 0.5, 0.6])

    params = _params_from_vector(scenario, vector)
    reconstructed = _vector_from_params(scenario, params)

    assert reconstructed == pytest.approx(vector)


def test_fixed_valid_friction_needs_no_inequality() -> None:
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "static_friction", "min": 0.4, "max": 0.4},
                {"name": "dynamic_friction", "min": 0.4, "max": 0.4},
            ],
        }
    )

    assert optimizers._friction_inequality_spec(scenario) is None


def test_random_optimizer_never_samples_dynamic_above_static() -> None:
    samples: list[dict[str, float]] = []

    run_random_optimizer(
        _friction_scenario(),
        lambda params: samples.append(params) or 0.0,
        max_trials=100,
        seed=17,
    )

    assert all(
        sample["dynamic_friction"] <= sample["static_friction"] for sample in samples
    )


def test_random_optimizer_samples_feasible_values_without_remapping_decode() -> None:
    samples: list[dict[str, float]] = []

    run_random_optimizer(
        _friction_scenario(),
        lambda params: samples.append(params) or 0.0,
        max_trials=1,
        seed=1,
    )

    rng = optimizers.np.random.default_rng(1)
    unit = rng.random(2)
    static = 0.05 + unit[0] * 1.45
    dynamic = 0.05 + unit[1] * (static - 0.05)
    assert samples == [
        {
            "static_friction": pytest.approx(static),
            "dynamic_friction": pytest.approx(dynamic),
        }
    ]


def test_random_optimizer_supports_single_point_friction_overlap() -> None:
    samples: list[dict[str, float]] = []

    run_random_optimizer(
        _touching_friction_scenario(),
        lambda params: samples.append(params) or 0.0,
        max_trials=3,
        seed=17,
    )

    assert samples == [
        {"static_friction": 0.5, "dynamic_friction": 0.5},
        {"static_friction": 0.5, "dynamic_friction": 0.5},
        {"static_friction": 0.5, "dynamic_friction": 0.5},
    ]


def test_is_botorch_available_true_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_botorch(monkeypatch)
    assert is_botorch_available() is True


def test_is_botorch_available_false_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "botorch":
            raise ImportError("missing botorch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert is_botorch_available() is False


def test_random_optimizer_runs_max_trials() -> None:
    sc = _scenario_2d()
    calls: list[dict[str, float]] = []

    def evaluate(params: dict[str, float]) -> float:
        calls.append(dict(params))
        return 0.0

    run_random_optimizer(sc, evaluate, max_trials=7, seed=1)
    assert len(calls) == 7
    for params in calls:
        assert 0.5 <= params["mass_scale"] <= 2.0
        assert 0.0 <= params["static_friction"] <= 1.0


def test_random_optimizer_reproducible_for_same_seed() -> None:
    sc = _scenario_2d()
    a: list[float] = []
    b: list[float] = []
    run_random_optimizer(
        sc, lambda p: (a.append(p["mass_scale"]), 0.0)[1], max_trials=3, seed=42
    )
    run_random_optimizer(
        sc, lambda p: (b.append(p["mass_scale"]), 0.0)[1], max_trials=3, seed=42
    )
    assert a == b


def test_random_optimizer_respects_cancel_check() -> None:
    sc = _scenario_2d()
    calls: list[dict[str, float]] = []

    def evaluate(params: dict[str, float]) -> float:
        calls.append(dict(params))
        return 0.0

    cancelled = {"v": False}

    def cancel_check() -> bool:
        return cancelled["v"]

    # Cancel after the first call.
    def evaluate_then_cancel(params: dict[str, float]) -> float:
        calls.append(dict(params))
        cancelled["v"] = True
        return 0.0

    run_random_optimizer(
        sc, evaluate_then_cancel, max_trials=20, seed=7, cancel_check=cancel_check
    )
    # Exactly one trial completed before cancel was observed at the top of
    # the next iteration.
    assert len(calls) == 1


def test_cma_es_optimizer_respects_max_trials_budget() -> None:
    sc = _scenario_2d()
    calls: list[dict[str, float]] = []

    def evaluate(params: dict[str, float]) -> float:
        calls.append(dict(params))
        # Decreasing function so CMA-ES has signal to converge.
        return params["mass_scale"] ** 2 + params["static_friction"] ** 2

    run_cma_es_optimizer(sc, evaluate, max_trials=8, seed=5)
    assert len(calls) == 8


def test_cma_es_optimizer_never_evaluates_invalid_friction() -> None:
    calls: list[dict[str, float]] = []

    run_cma_es_optimizer(
        _unequal_friction_scenario(),
        lambda params: calls.append(dict(params)) or params["mass_scale"],
        max_trials=12,
        seed=5,
    )

    assert len(calls) == 12
    assert all(
        params["dynamic_friction"] <= params["static_friction"] for params in calls
    )


def test_cma_es_uses_finite_penalties_and_fills_trial_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cma_es_module = importlib.import_module(
        "world_understanding.functions.optimization.cma_es"
    )
    optimizer_outputs: list[float] = []

    def fake_cma_es(*, evaluate, **_kwargs):
        optimizer_outputs.append(evaluate(x=optimizers.np.asarray([0.0, 0.5, 1.0])))
        while True:
            optimizer_outputs.append(evaluate(x=optimizers.np.asarray([1.0, 0.5, 0.0])))

    monkeypatch.setattr(cma_es_module, "cma_es", fake_cma_es)
    calls: list[dict[str, float]] = []

    run_cma_es_optimizer(
        _unequal_friction_scenario(),
        lambda params: calls.append(dict(params)) or params["mass_scale"],
        max_trials=4,
        seed=5,
    )

    assert len(calls) == 4
    assert optimizers.np.isfinite(optimizer_outputs).all()
    assert optimizer_outputs[1] > optimizer_outputs[0]
    assert all(
        params["dynamic_friction"] <= params["static_friction"] for params in calls
    )


def test_cma_es_ranks_infeasible_proposals_after_failed_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cma_es_module = importlib.import_module(
        "world_understanding.functions.optimization.cma_es"
    )
    optimizer_outputs: list[float] = []

    def fake_cma_es(*, evaluate, **_kwargs):
        feasible = optimizers.np.asarray([0.0, 0.5, 1.0])
        infeasible = optimizers.np.asarray([1.0, 0.5, 0.0])
        optimizer_outputs.append(evaluate(x=feasible))
        optimizer_outputs.append(evaluate(x=infeasible))
        while True:
            optimizer_outputs.append(evaluate(x=feasible))

    monkeypatch.setattr(cma_es_module, "cma_es", fake_cma_es)
    calls: list[dict[str, float]] = []

    run_cma_es_optimizer(
        _unequal_friction_scenario(),
        lambda params: calls.append(dict(params)) or float("inf"),
        max_trials=2,
        seed=5,
    )

    assert len(calls) == 2
    assert optimizers.np.isfinite(optimizer_outputs).all()
    assert optimizer_outputs[1] > optimizer_outputs[0]


def test_failed_trial_penalty_is_finite_and_worse_than_observed_scores() -> None:
    penalty = optimizers._finite_failed_trial_penalty([float("nan"), -2.0, 1.0])

    assert optimizers.np.isfinite(penalty)
    assert penalty > 1.0
    limit = optimizers.np.finfo(optimizers.np.float64).max / 4.0
    assert optimizers._finite_failed_trial_penalty([limit]) == limit


def test_cma_es_ranks_infeasible_proposals_after_later_feasible_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cma_es_module = importlib.import_module(
        "world_understanding.functions.optimization.cma_es"
    )
    optimizer_outputs: list[float] = []

    def fake_cma_es(*, evaluate, **_kwargs):
        feasible = optimizers.np.asarray([0.0, 0.5, 1.0])
        infeasible = optimizers.np.asarray([1.0, 0.5, 0.0])
        optimizer_outputs.append(evaluate(x=feasible))
        optimizer_outputs.append(evaluate(x=infeasible))
        optimizer_outputs.append(evaluate(x=feasible))
        while True:
            optimizer_outputs.append(evaluate(x=feasible))

    monkeypatch.setattr(cma_es_module, "cma_es", fake_cma_es)
    scores = iter([1.0, 1.0e15])

    run_cma_es_optimizer(
        _unequal_friction_scenario(),
        lambda _params: next(scores),
        max_trials=2,
        seed=5,
    )

    assert optimizers.np.isfinite(optimizer_outputs).all()
    assert optimizer_outputs[1] > optimizer_outputs[2]


def test_cma_es_distinguishes_proposal_cap_fill(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cma_es_module = importlib.import_module(
        "world_understanding.functions.optimization.cma_es"
    )

    def fake_cma_es(*, evaluate, **_kwargs):
        while True:
            evaluate(x=optimizers.np.asarray([1.0, 0.5, 0.0]))

    monkeypatch.setattr(cma_es_module, "cma_es", fake_cma_es)
    calls: list[dict[str, float]] = []

    run_cma_es_optimizer(
        _unequal_friction_scenario(),
        lambda params: calls.append(dict(params)) or 0.0,
        max_trials=2,
        seed=5,
    )

    assert len(calls) == 2
    assert "reached its 20-proposal cap" in caplog.text


def test_cma_es_distinguishes_early_optimizer_return(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cma_es_module = importlib.import_module(
        "world_understanding.functions.optimization.cma_es"
    )
    monkeypatch.setattr(cma_es_module, "cma_es", lambda **_kwargs: {})
    calls: list[dict[str, float]] = []

    run_cma_es_optimizer(
        _unequal_friction_scenario(),
        lambda params: calls.append(dict(params)) or 0.0,
        max_trials=2,
        seed=5,
    )

    assert len(calls) == 2
    assert "stopped before the trial budget" in caplog.text
    assert "proposal cap" not in caplog.text


def test_cma_es_cancel_check_stops_feasible_budget_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cma_es_module = importlib.import_module(
        "world_understanding.functions.optimization.cma_es"
    )
    monkeypatch.setattr(cma_es_module, "cma_es", lambda **_kwargs: {})
    calls: list[dict[str, float]] = []

    run_cma_es_optimizer(
        _touching_friction_scenario(),
        lambda params: calls.append(dict(params)) or 0.0,
        max_trials=3,
        seed=5,
        cancel_check=lambda: len(calls) >= 1,
    )

    assert calls == [{"static_friction": 0.5, "dynamic_friction": 0.5}]


def test_cma_es_supports_single_point_friction_overlap() -> None:
    calls: list[dict[str, float]] = []

    run_cma_es_optimizer(
        _touching_friction_scenario(),
        lambda params: calls.append(dict(params)) or 0.0,
        max_trials=3,
        seed=5,
    )

    assert calls == [
        {"static_friction": 0.5, "dynamic_friction": 0.5},
        {"static_friction": 0.5, "dynamic_friction": 0.5},
        {"static_friction": 0.5, "dynamic_friction": 0.5},
    ]


def test_cma_es_optimizer_respects_cancel_inside_evaluator() -> None:
    sc = _scenario_2d()
    calls = {"n": 0}

    def evaluate(params: dict[str, float]) -> float:
        calls["n"] += 1
        return params["mass_scale"]

    run_cma_es_optimizer(
        sc,
        evaluate,
        max_trials=8,
        seed=5,
        cancel_check=lambda: calls["n"] >= 1,
    )

    assert calls["n"] == 1


def test_get_runner_returns_correct_callable() -> None:
    assert get_runner(OPTIMIZER_RANDOM).__name__ == "run_random_optimizer"
    assert get_runner(OPTIMIZER_CMA_ES).__name__ == "run_cma_es_optimizer"
    assert get_runner(OPTIMIZER_BOTORCH).__name__ == "run_botorch_optimizer"


def test_get_runner_rejects_auto() -> None:
    # Caller should resolve `auto` first; passing it through is a programming
    # error.
    with pytest.raises(ValueError, match="No runner"):
        get_runner(OPTIMIZER_AUTO)


def test_run_botorch_when_missing_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When BoTorch is missing, run_botorch_optimizer raises the install hint.

    This is the authoritative test that proves the Acceptance Criteria:
    ``--optimizer botorch`` must NEVER silently fall back to random.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name in ("torch", "botorch") or name.startswith(
            ("torch.", "botorch.", "gpytorch")
        ):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    sc = _scenario_2d()
    with pytest.raises(BoTorchUnavailableError) as ei:
        optimizers.run_botorch_optimizer(sc, lambda p: 0.0, max_trials=3, seed=0)
    assert "BoTorch optimizer requires the tuning extra" in str(ei.value)


def test_botorch_optimizer_runs_lazy_import_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_botorch(monkeypatch)
    sc = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "mass_scale", "min": 0.5, "max": 2.0}],
        }
    )
    calls: list[dict[str, float]] = []

    run_botorch_optimizer(
        sc,
        lambda params: calls.append(dict(params)) or params["mass_scale"],
        max_trials=3,
        seed=7,
    )

    assert len(calls) == 3
    assert all(0.5 <= item["mass_scale"] <= 2.0 for item in calls)


def test_botorch_replaces_failed_trial_scores_before_gp_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_botorch(monkeypatch)
    scores = iter([float("inf"), 1.0, float("inf"), 0.5])

    run_botorch_optimizer(
        parse_scenario(
            {
                "name": "drop_settle",
                "parameters": [{"name": "mass_scale", "min": 0.5, "max": 2.0}],
            }
        ),
        lambda _params: next(scores),
        max_trials=4,
        seed=7,
    )

    assert fake.gp_training_sets
    assert all(
        optimizers.np.isfinite(y_train.data).all()
        for _, y_train in fake.gp_training_sets
    )


def test_botorch_uses_explicit_friction_inequality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_botorch(
        monkeypatch,
        candidate=[0.0, 0.5, 1.0],
    )
    calls: list[dict[str, float]] = []

    run_botorch_optimizer(
        _unequal_friction_scenario(),
        lambda params: calls.append(dict(params)) or params["mass_scale"],
        max_trials=7,
        seed=42,
    )

    assert len(calls) == 7
    assert all(
        params["dynamic_friction"] <= params["static_friction"] for params in calls
    )
    constraint = fake.optimize_calls[0]["inequality_constraints"][0]
    indices, coefficients, rhs = constraint
    assert indices.data.tolist() == [2, 0]
    assert coefficients.data.tolist() == pytest.approx([0.5, -0.7])
    assert rhs == pytest.approx(0.1)
    assert calls[-1] == {
        "dynamic_friction": pytest.approx(0.2),
        "mass_scale": pytest.approx(1.25),
        "static_friction": pytest.approx(0.6),
    }


def test_botorch_rejects_constraint_violating_acquisition_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_botorch(
        monkeypatch,
        candidate=[1.0, 0.5, 0.0],
    )
    calls: list[dict[str, float]] = []

    run_botorch_optimizer(
        _unequal_friction_scenario(),
        lambda params: calls.append(dict(params)) or params["mass_scale"],
        max_trials=7,
        seed=42,
    )

    assert len(calls) == 7
    assert all(
        params["dynamic_friction"] <= params["static_friction"] for params in calls
    )


def test_botorch_repairs_solver_scale_boundary_overshoot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_botorch(
        monkeypatch,
        candidate=[0.5, optimizers.np.nextafter(0.5, 1.0)],
    )
    calls: list[dict[str, float]] = []

    run_botorch_optimizer(
        _friction_scenario(),
        lambda params: calls.append(dict(params)) or 0.0,
        max_trials=5,
        seed=42,
    )

    assert len(calls) == 5
    assert calls[-1]["dynamic_friction"] <= calls[-1]["static_friction"]
    assert calls[-1]["dynamic_friction"] == pytest.approx(calls[-1]["static_friction"])


def test_botorch_supports_single_point_friction_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_botorch(monkeypatch, candidate=[1.0, 0.0])
    calls: list[dict[str, float]] = []

    run_botorch_optimizer(
        _touching_friction_scenario(),
        lambda params: calls.append(dict(params)) or 0.0,
        max_trials=5,
        seed=42,
    )

    assert calls == [
        {"static_friction": 0.5, "dynamic_friction": 0.5},
        {"static_friction": 0.5, "dynamic_friction": 0.5},
        {"static_friction": 0.5, "dynamic_friction": 0.5},
        {"static_friction": 0.5, "dynamic_friction": 0.5},
        {"static_friction": 0.5, "dynamic_friction": 0.5},
    ]
    assert fake.optimize_calls == []


def test_botorch_pins_single_friction_pair_while_optimizing_other_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_botorch(monkeypatch)
    calls: list[dict[str, float]] = []

    run_botorch_optimizer(
        _touching_friction_with_mass_scenario(),
        lambda params: calls.append(dict(params)) or params["mass_scale"],
        max_trials=7,
        seed=42,
    )

    assert len(calls) == 7
    assert all(
        call["static_friction"] == call["dynamic_friction"] == 0.5 for call in calls
    )
    assert fake.optimize_calls
    assert fake.optimize_calls[0]["fixed_features"] == {0: 1.0, 1: 0.0}
    assert fake.optimize_calls[0]["inequality_constraints"] is None


def test_botorch_fixed_valid_friction_passes_no_inequality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_botorch(monkeypatch)
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "static_friction", "min": 0.5, "max": 0.5},
                {"name": "dynamic_friction", "min": 0.4, "max": 0.4},
            ],
        }
    )

    calls: list[dict[str, float]] = []
    run_botorch_optimizer(
        scenario,
        lambda params: calls.append(dict(params)) or 0.0,
        max_trials=5,
        seed=7,
    )

    assert calls == [
        {"static_friction": 0.5, "dynamic_friction": 0.4},
        {"static_friction": 0.5, "dynamic_friction": 0.4},
        {"static_friction": 0.5, "dynamic_friction": 0.4},
        {"static_friction": 0.5, "dynamic_friction": 0.4},
        {"static_friction": 0.5, "dynamic_friction": 0.4},
    ]
    assert fake.optimize_calls == []


def test_botorch_fixed_parameters_respect_cancel_after_initial_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_botorch(monkeypatch)
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "static_friction", "min": 0.5, "max": 0.5},
                {"name": "dynamic_friction", "min": 0.4, "max": 0.4},
            ],
        }
    )
    calls: list[dict[str, float]] = []

    run_botorch_optimizer(
        scenario,
        lambda params: calls.append(dict(params)) or 0.0,
        max_trials=5,
        seed=7,
        cancel_check=lambda: len(calls) >= 4,
    )

    assert len(calls) == 4
    assert fake.optimize_calls == []


def test_botorch_optimizer_gp_failure_falls_back_to_random(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_botorch(monkeypatch)

    sc = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "mass_scale", "min": 0.5, "max": 2.0}],
        }
    )
    calls: list[dict[str, float]] = []

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic gp failure")

    monkeypatch.setattr(fake.models, "SingleTaskGP", boom)

    run_botorch_optimizer(
        sc,
        lambda params: calls.append(dict(params)) or params["mass_scale"],
        max_trials=3,
        seed=9,
    )

    assert len(calls) == 3


def test_botorch_gp_failure_fallback_preserves_friction_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_botorch(monkeypatch)
    calls: list[dict[str, float]] = []

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic gp failure")

    monkeypatch.setattr(fake.models, "SingleTaskGP", boom)

    run_botorch_optimizer(
        _unequal_friction_scenario(),
        lambda params: calls.append(dict(params)) or params["mass_scale"],
        max_trials=7,
        seed=9,
    )

    assert len(calls) == 7
    assert all(
        params["dynamic_friction"] <= params["static_friction"] for params in calls
    )


def test_botorch_optimizer_respects_cancel_during_initial_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_botorch(monkeypatch)
    sc = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "mass_scale", "min": 0.5, "max": 2.0}],
        }
    )
    calls: list[dict[str, float]] = []

    run_botorch_optimizer(
        sc,
        lambda params: calls.append(dict(params)) or params["mass_scale"],
        max_trials=3,
        seed=0,
        cancel_check=lambda: True,
    )

    assert calls == []


def test_botorch_optimizer_respects_cancel_during_gp_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_botorch(monkeypatch)
    sc = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "mass_scale", "min": 0.5, "max": 2.0}],
        }
    )
    calls: list[dict[str, float]] = []

    def evaluate(params: dict[str, float]) -> float:
        calls.append(dict(params))
        return params["mass_scale"]

    run_botorch_optimizer(
        sc,
        evaluate,
        max_trials=3,
        seed=0,
        cancel_check=lambda: len(calls) >= 2,
    )

    assert len(calls) == 2


def test_real_botorch_respects_explicit_friction_inequality() -> None:
    pytest.importorskip("botorch")
    calls: list[dict[str, float]] = []

    run_botorch_optimizer(
        _unequal_friction_scenario(),
        lambda params: calls.append(dict(params))
        or (params["static_friction"] - 0.45) ** 2
        + (params["dynamic_friction"] - 0.4) ** 2,
        max_trials=7,
        seed=42,
    )

    assert len(calls) == 7
    assert all(
        params["dynamic_friction"] <= params["static_friction"] for params in calls
    )


def test_real_botorch_combines_fixed_features_and_friction_inequality() -> None:
    pytest.importorskip("botorch")
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "static_friction", "min": 0.1, "max": 0.8},
                {"name": "mass_scale", "min": 1.0, "max": 1.0},
                {"name": "dynamic_friction", "min": 0.2, "max": 0.9},
            ],
        }
    )
    calls: list[dict[str, float]] = []

    run_botorch_optimizer(
        scenario,
        lambda params: calls.append(dict(params))
        or (params["static_friction"] - 0.6) ** 2
        + (params["dynamic_friction"] - 0.4) ** 2,
        max_trials=7,
        seed=42,
    )

    assert len(calls) == 7
    assert all(params["mass_scale"] == 1.0 for params in calls)
    assert all(
        params["dynamic_friction"] <= params["static_friction"] for params in calls
    )


def test_real_botorch_supports_single_point_friction_overlap() -> None:
    pytest.importorskip("botorch")
    calls: list[dict[str, float]] = []

    run_botorch_optimizer(
        _touching_friction_with_mass_scenario(),
        lambda params: calls.append(dict(params)) or (params["mass_scale"] - 1.25) ** 2,
        max_trials=7,
        seed=42,
    )

    assert len(calls) == 7
    assert all(
        call["static_friction"] == call["dynamic_friction"] == 0.5 for call in calls
    )
