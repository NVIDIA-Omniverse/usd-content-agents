# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for optimizer dispatch + BoTorch availability handling."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from physics_agent.tuning import optimizers
from physics_agent.tuning.errors import BoTorchUnavailableError
from physics_agent.tuning.optimizers import (
    OPTIMIZER_AUTO,
    OPTIMIZER_BOTORCH,
    OPTIMIZER_CMA_ES,
    OPTIMIZER_RANDOM,
    SUPPORTED_OPTIMIZERS,
    _params_from_vector,
    _vector_from_params,
    get_runner,
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


def _install_fake_botorch(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    torch = ModuleType("torch")
    torch.double = object()
    torch.manual_seed = lambda _seed: None
    torch.zeros = lambda d: _FakeTensor(optimizers.np.zeros(d))
    torch.ones = lambda d: _FakeTensor(optimizers.np.ones(d))
    torch.stack = lambda tensors: _FakeTensor([t.data for t in tensors])
    torch.tensor = lambda data, dtype=None: _FakeTensor(data)

    botorch = ModuleType("botorch")
    botorch.__path__ = []  # type: ignore[attr-defined]
    acquisition = ModuleType("botorch.acquisition")
    fit = ModuleType("botorch.fit")
    models = ModuleType("botorch.models")
    optim = ModuleType("botorch.optim")

    class SingleTaskGP:
        def __init__(self, _x_train, _y_train):
            self.likelihood = object()

    class qExpectedImprovement:
        def __init__(self, *, model, best_f):
            self.model = model
            self.best_f = best_f

    def optimize_acqf(*, bounds, **_kwargs):
        d = bounds.data.shape[1]
        return _FakeTensor([[0.5] * d]), None

    acquisition.qExpectedImprovement = qExpectedImprovement
    fit.fit_gpytorch_mll = lambda _mll: None
    models.SingleTaskGP = SingleTaskGP
    optim.optimize_acqf = optimize_acqf
    botorch.acquisition = acquisition
    botorch.fit = fit
    botorch.models = models
    botorch.optim = optim

    gpytorch = ModuleType("gpytorch")
    gpytorch.__path__ = []  # type: ignore[attr-defined]
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
        gpytorch=gpytorch,
        mlls=mlls,
    )


def test_supported_optimizers_canonical_set() -> None:
    assert OPTIMIZER_AUTO in SUPPORTED_OPTIMIZERS
    assert OPTIMIZER_BOTORCH in SUPPORTED_OPTIMIZERS
    assert OPTIMIZER_RANDOM in SUPPORTED_OPTIMIZERS
    assert OPTIMIZER_CMA_ES in SUPPORTED_OPTIMIZERS


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


@pytest.mark.parametrize(
    "vector",
    ([0.0, 0.0], [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]),
)
def test_vector_mapping_enforces_physical_friction_order(
    vector: list[float],
) -> None:
    scenario = _friction_scenario()

    params = _params_from_vector(scenario, optimizers.np.asarray(vector))

    assert params["dynamic_friction"] <= params["static_friction"]
    reconstructed = _params_from_vector(scenario, _vector_from_params(scenario, params))
    assert reconstructed == pytest.approx(params)


def test_coupled_friction_round_trip_preserves_mixed_parameters() -> None:
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
    reconstructed = _params_from_vector(
        scenario,
        _vector_from_params(scenario, params),
    )

    assert params["dynamic_friction"] <= params["static_friction"]
    assert reconstructed == pytest.approx(params)


def test_coupled_friction_handles_zero_width_boundary() -> None:
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "static_friction", "min": 0.4, "max": 0.4},
                {"name": "dynamic_friction", "min": 0.4, "max": 0.4},
            ],
        }
    )

    params = _params_from_vector(scenario, optimizers.np.asarray([1.0, 1.0]))
    reconstructed = _params_from_vector(
        scenario,
        _vector_from_params(scenario, params),
    )

    assert params == {"static_friction": 0.4, "dynamic_friction": 0.4}
    assert reconstructed == pytest.approx(params)


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
    # Allow CMA-ES to perform an initial-mean evaluation and then up to
    # max_trials total, but never more.
    assert 1 <= len(calls) <= 8


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
