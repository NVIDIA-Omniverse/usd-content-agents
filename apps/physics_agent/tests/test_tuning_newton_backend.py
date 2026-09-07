# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``NewtonBackend`` dispatch tests with a mocked simulator.

Proves the backend hands a ``Simulator`` to per-scenario evaluators, injects
``judge_callback`` for freeform, ``final_state_judge`` for drop_settle, and
lazy-instantiates the simulator on first ``evaluate`` (not construction).

Newton itself is not required to run these tests — the simulator is mocked
out via monkeypatch. The companion ``test_tuning_newton_simulator.py``
exercises the real Newton install when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from physics_agent.tuning.backend import (
    NEWTON_UNSUPPORTED_PARAM_REASONS,
    validate_engine_supports_param_names,
)
from physics_agent.tuning.capabilities import (
    newton_mujoco_capabilities,
    usd_physics_capabilities,
)
from physics_agent.tuning.errors import NewtonUnavailableError, TuningError
from physics_agent.tuning.newton_backend import NewtonBackend
from physics_agent.tuning.simulator import Simulator
from physics_agent.tuning.types import Scenario, TunableParam


class _FakeSimulator:
    """Stand-in for :class:`NewtonSimulator`.

    Returns a deterministic two-sample trajectory in the engine-agnostic
    ``Simulator`` contract shape: ``(t, [px,py,pz,qx,qy,qz,qw], [vx,vy,vz,wx,wy,wz])``.
    """

    name = "newton"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.warmup_calls = 0
        self.shutdown_called = False

    def warmup(self) -> None:
        self.warmup_calls += 1

    def shutdown(self) -> None:
        self.shutdown_called = True

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "trajectory": [
                (
                    0.0,
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                (
                    1.0,
                    [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
                ),
            ],
            "final_pose": [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0],
            "final_velocity": [0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
            "n_bodies": 1,
            "duration_s": 1.0,
            "n_steps": 240,
        }


@pytest.fixture
def fake_simulator(monkeypatch: pytest.MonkeyPatch) -> _FakeSimulator:
    fs = _FakeSimulator()
    monkeypatch.setattr(
        "physics_agent.tuning.newton_simulator.NewtonSimulator",
        lambda *a, **kw: fs,
    )
    return fs


def _drop_settle_scenario() -> Scenario:
    return Scenario(
        name="drop_settle",
        metric="settle_distance",
        target={"drop_height_m": 0.5, "duration_s": 1.0, "gravity": -9.81},
        params=(TunableParam(name="mass_scale", min_value=0.5, max_value=2.0),),
    )


def _freeform_scenario() -> Scenario:
    return Scenario(
        name="freeform",
        metric="judge_score",
        target={
            "description": "test",
            "duration_s": 1.0,
            "initial_pose": {"position": [0.0, 1.0, 0.0]},
            "observations": ["object stayed upright"],
        },
        params=(TunableParam(name="dynamic_friction", min_value=0.05, max_value=0.4),),
    )


def _restitution_scenario() -> Scenario:
    return Scenario(
        name="drop_settle",
        metric="max_bounce_height",
        target={"drop_height_m": 0.5, "duration_s": 1.0, "gravity": -9.81},
        params=(TunableParam(name="restitution", min_value=0.0, max_value=1.0),),
    )


def _contact_scenario() -> Scenario:
    return Scenario(
        name="drop_settle",
        metric="settle_distance",
        target={"drop_height_m": 0.5, "duration_s": 1.0, "gravity": -9.81},
        params=(
            TunableParam(name="contact_ke", min_value=100.0, max_value=100000.0),
            TunableParam(name="contact_kd", min_value=0.0, max_value=5000.0),
        ),
    )


def _static_friction_scenario() -> Scenario:
    return Scenario(
        name="drop_settle",
        metric="settle_distance",
        target={"drop_height_m": 0.5, "duration_s": 1.0, "gravity": -9.81},
        params=(TunableParam(name="static_friction", min_value=0.05, max_value=1.0),),
    )


def _static_and_restitution_scenario() -> Scenario:
    return Scenario(
        name="drop_settle",
        metric="max_bounce_height",
        target={"drop_height_m": 0.5, "duration_s": 1.0, "gravity": -9.81},
        params=(
            TunableParam(name="static_friction", min_value=0.05, max_value=1.0),
            TunableParam(name="restitution", min_value=0.0, max_value=1.0),
        ),
    )


def _physics_usd(tmp_path: Path) -> Path:
    from pxr import (  # type: ignore[import-untyped]
        Gf,
        Usd,
        UsdGeom,
        UsdPhysics,
        UsdShade,
    )

    p = tmp_path / "physics.usda"
    stage = Usd.Stage.CreateNew(str(p))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    body = UsdGeom.Sphere.Define(stage, "/World/Body")
    body.CreateRadiusAttr(0.25)
    body.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr().Set(1.0)
    mat = UsdShade.Material.Define(stage, "/World/Mat")
    mat_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mat_api.CreateDynamicFrictionAttr(0.3)
    mat_api.CreateStaticFrictionAttr(0.4)
    stage.GetRootLayer().Save()
    return p


def test_fake_simulator_satisfies_simulator_protocol() -> None:
    # Structural Protocol check — guards against drift between the fake and
    # the contract.
    assert isinstance(_FakeSimulator(), Simulator)


def test_newton_unsupported_reason_keys_cover_usd_only_params() -> None:
    usd_names = {capability.param_name for capability in usd_physics_capabilities()}
    newton_names = {
        capability.param_name for capability in newton_mujoco_capabilities()
    }
    assert set(NEWTON_UNSUPPORTED_PARAM_REASONS) == usd_names - newton_names


def test_fake_contact_param_hint_does_not_suggest_newton() -> None:
    with pytest.raises(TuningError) as exc_info:
        validate_engine_supports_param_names("fake", ["contact_ke"])
    assert "use --engine newton" not in str(exc_info.value)


def test_lazy_simulator_init(fake_simulator: _FakeSimulator, tmp_path: Path) -> None:
    backend = NewtonBackend()
    assert backend._simulator is None
    backend.evaluate(
        params={"mass_scale": 1.0},
        scenario=_drop_settle_scenario(),
        physics_usd=_physics_usd(tmp_path),
        seed=1,
    )
    assert backend._simulator is fake_simulator
    assert len(fake_simulator.calls) == 1


def test_warmup_starts_simulator_eagerly(
    fake_simulator: _FakeSimulator,
) -> None:
    """``warmup`` must construct + probe the simulator synchronously so a
    missing newton install fails fast BEFORE the runner spends an LLM call
    on the user_prompt path."""
    backend = NewtonBackend()
    assert backend._simulator is None
    backend.warmup()
    assert backend._simulator is fake_simulator
    assert fake_simulator.warmup_calls == 1
    backend.warmup()
    assert fake_simulator.warmup_calls == 2


def test_warmup_propagates_newton_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing newton install (or missing CUDA) → ``NewtonUnavailableError``;
    warmup must let it bubble, not swallow it."""

    class _Unavailable:
        def warmup(self) -> None:
            raise NewtonUnavailableError("test: newton missing")

    monkeypatch.setattr(
        "physics_agent.tuning.newton_simulator.NewtonSimulator",
        lambda *a, **kw: _Unavailable(),
    )
    backend = NewtonBackend()
    with pytest.raises(NewtonUnavailableError, match="newton missing"):
        backend.warmup()


def test_restitution_scenarios_rejected_before_simulator_init(
    fake_simulator: _FakeSimulator,
    tmp_path: Path,
) -> None:
    backend = NewtonBackend()

    with pytest.raises(TuningError, match="does not support tuning restitution"):
        backend.evaluate(
            params={"restitution": 0.6},
            scenario=_restitution_scenario(),
            physics_usd=tmp_path / "unused.usda",
            seed=1,
        )

    assert backend._simulator is None
    assert fake_simulator.calls == []


def test_static_friction_scenarios_rejected_before_simulator_init(
    fake_simulator: _FakeSimulator,
    tmp_path: Path,
) -> None:
    backend = NewtonBackend()

    with pytest.raises(TuningError, match="static_friction-only trials"):
        backend.evaluate(
            params={"static_friction": 0.6},
            scenario=_static_friction_scenario(),
            physics_usd=tmp_path / "unused.usda",
            seed=1,
        )

    assert backend._simulator is None
    assert fake_simulator.calls == []


def test_multiple_unsupported_params_report_both_reasons(
    fake_simulator: _FakeSimulator,
    tmp_path: Path,
) -> None:
    backend = NewtonBackend()

    with pytest.raises(TuningError) as exc_info:
        backend.evaluate(
            params={"static_friction": 0.6, "restitution": 0.4},
            scenario=_static_and_restitution_scenario(),
            physics_usd=tmp_path / "unused.usda",
            seed=1,
        )

    message = str(exc_info.value)
    assert "static_friction" in message
    assert "restitution" in message
    assert "static_friction-only trials" in message
    assert "bouncy/max_bounce_height" in message
    assert backend._simulator is None
    assert fake_simulator.calls == []


def test_drop_settle_dispatch_does_not_inject_judge_callback(
    fake_simulator: _FakeSimulator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """drop_settle must NOT see ``judge_callback`` — that's freeform-only."""
    captured: dict[str, Any] = {}

    from physics_agent.tuning.scenarios import drop_settle as drop_settle_mod

    real = drop_settle_mod.evaluate

    def _spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(drop_settle_mod, "evaluate", _spy)

    backend = NewtonBackend()
    backend.judge_callback = lambda *a, **kw: {"score": 1.0, "reasoning": "x"}
    result = backend.evaluate(
        params={"mass_scale": 1.0},
        scenario=_drop_settle_scenario(),
        physics_usd=_physics_usd(tmp_path),
        seed=2,
    )
    assert "judge_callback" not in captured
    assert "simulator" in captured
    assert captured["simulator"] is fake_simulator
    assert "score" in result
    assert "settle_distance" in result


def test_drop_settle_trials_share_ground_clearance_support_cache(
    fake_simulator: _FakeSimulator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchanged source geometry must reuse one backend-owned cache entry."""
    from physics_agent.tuning.scenarios import drop_settle as drop_settle_mod

    calls: list[dict[str, Any]] = []

    def _capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        cache = kwargs["ground_clearance_support_cache"]
        key = kwargs["ground_clearance_support_cache_key"]
        cache.setdefault(key, {"selected_mode": "mesh_vertices"})
        return {"score": 0.0}

    monkeypatch.setattr(drop_settle_mod, "evaluate", _capture)

    physics_usd = _physics_usd(tmp_path)
    backend = NewtonBackend()
    for seed in (10, 11):
        backend.evaluate(
            params={"mass_scale": 1.0},
            scenario=_drop_settle_scenario(),
            physics_usd=physics_usd,
            seed=seed,
        )

    assert len(calls) == 2
    first_cache = calls[0]["ground_clearance_support_cache"]
    first_key = calls[0]["ground_clearance_support_cache_key"]
    assert first_cache is backend._ground_clearance_support_cache
    assert calls[1]["ground_clearance_support_cache"] is first_cache
    assert calls[1]["ground_clearance_support_cache_key"] == first_key
    assert first_key == f"physics_usd:{physics_usd.resolve()}"
    assert first_cache[first_key] == {"selected_mode": "mesh_vertices"}


def test_direct_evaluate_resolves_newton_contact_bindings(
    fake_simulator: _FakeSimulator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import physics_agent.tuning.usd_patch as usd_patch_mod

    captured: list[list[dict[str, Any]] | None] = []
    real_patch = usd_patch_mod.patch_physics_usd

    def _spy_patch(*args: Any, **kwargs: Any) -> Path:
        captured.append(kwargs.get("bindings"))
        return real_patch(*args, **kwargs)

    monkeypatch.setattr(usd_patch_mod, "patch_physics_usd", _spy_patch)

    backend = NewtonBackend()
    result = backend.evaluate(
        params={"contact_ke": 12345.0, "contact_kd": 321.0},
        scenario=_contact_scenario(),
        physics_usd=_physics_usd(tmp_path),
        seed=4,
    )

    assert fake_simulator.calls
    assert "score" in result
    assert captured and captured[0] is not None
    assert {binding["param"] for binding in captured[0] or []} == {
        "contact_ke",
        "contact_kd",
    }


def test_freeform_dispatch_injects_judge_callback(
    fake_simulator: _FakeSimulator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """freeform MUST receive ``judge_callback`` from the backend."""
    captured: dict[str, Any] = {}

    def judge(
        frames: list[Path], user_prompt: str | None, observations: list[str]
    ) -> dict[str, Any]:
        return {"score": 0.7, "reasoning": "ok"}

    from physics_agent.tuning.scenarios import freeform as freeform_mod

    real = freeform_mod.evaluate

    def _spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(freeform_mod, "evaluate", _spy)

    backend = NewtonBackend()
    backend.judge_callback = judge
    result = backend.evaluate(
        params={"dynamic_friction": 0.3},
        scenario=_freeform_scenario(),
        physics_usd=_physics_usd(tmp_path),
        seed=3,
    )
    assert captured.get("judge_callback") is judge
    assert captured.get("simulator") is fake_simulator
    assert "score" in result


def test_shutdown_releases_simulator(fake_simulator: _FakeSimulator) -> None:
    backend = NewtonBackend()
    backend.warmup()
    assert backend._simulator is fake_simulator
    backend.shutdown()
    assert fake_simulator.shutdown_called
    assert backend._simulator is None
    # Idempotent: a second shutdown when no simulator is held is a no-op.
    backend.shutdown()
