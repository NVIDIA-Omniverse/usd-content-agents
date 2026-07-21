# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``OvPhysXBackend`` dispatch tests with a mocked daemon.

Proves the backend hands the daemon to per-scenario evaluators, injects
``judge_callback`` for freeform, ``final_state_judge`` for drop_settle,
and lazy-instantiates the daemon on first ``evaluate`` (not at
construction time)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from physics_agent.tuning.errors import TuningError
from physics_agent.tuning.ovphysx_backend import OvPhysXBackend
from physics_agent.tuning.scenario_resolution import get_resolved_bindings
from physics_agent.tuning.types import Scenario, TunableParam


class _FakeDaemon:
    """Stand-in for ``_OvPhysXDaemon`` — never spawns subprocess."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.shutdown_called = False
        self.ensure_running_calls = 0

    def ensure_running(self) -> None:
        self.ensure_running_calls += 1

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        # Trajectory shape from the daemon is (t_s, pose7, vel6); both
        # pose and velocity come straight from the simulator's tensor
        # bindings — no finite-differencing on the consumer side.
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

    def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture
def fake_daemon(monkeypatch: pytest.MonkeyPatch) -> _FakeDaemon:
    fd = _FakeDaemon()
    # Patch the lazy daemon constructor so OvPhysXBackend uses our stub.
    monkeypatch.setattr(
        "world_understanding.functions.physics.ovphysx_daemon._OvPhysXDaemon",
        lambda *a, **kw: fd,
    )
    return fd


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


def _physics_usd(tmp_path: Path) -> Path:
    """Minimal physics-authored USD; enough for the scene builders to
    open + traverse."""
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
    mat = UsdShade.Material.Define(stage, "/Mat")
    mat_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mat_api.CreateStaticFrictionAttr().Set(0.4)
    mat_api.CreateDynamicFrictionAttr().Set(0.3)
    mat_api.CreateRestitutionAttr().Set(0.2)
    stage.GetRootLayer().Save()
    return p


def test_lazy_daemon_init(fake_daemon: _FakeDaemon, tmp_path: Path) -> None:
    backend = OvPhysXBackend()
    # Daemon should not exist yet.
    assert backend._daemon is None
    backend.evaluate(
        params={"mass_scale": 1.0},
        scenario=_drop_settle_scenario(),
        physics_usd=_physics_usd(tmp_path),
        seed=1,
    )
    # Now it should be the fake.
    assert backend._daemon is fake_daemon
    assert len(fake_daemon.calls) == 1


def test_warmup_starts_daemon_eagerly(fake_daemon: _FakeDaemon) -> None:
    """Round 14 (Codex CX P2#3): ``warmup`` must construct + ensure the
    daemon synchronously so a missing ovphysx venv fails fast BEFORE
    the runner spends an LLM call on the user_prompt path."""
    backend = OvPhysXBackend()
    assert backend._daemon is None
    backend.warmup()
    assert backend._daemon is fake_daemon
    assert fake_daemon.ensure_running_calls == 1
    # Idempotent: a second call reuses the same daemon.
    backend.warmup()
    assert backend._daemon is fake_daemon
    assert fake_daemon.ensure_running_calls == 2


def test_warmup_propagates_daemon_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing venv → OvPhysXDaemonUnavailableError; warmup must let it
    bubble, not swallow it."""
    from world_understanding.functions.physics.ovphysx_daemon import (
        OvPhysXDaemonUnavailableError,
    )

    class _UnavailableDaemon:
        def ensure_running(self) -> None:
            raise OvPhysXDaemonUnavailableError("test: venv missing")

    monkeypatch.setattr(
        "world_understanding.functions.physics.ovphysx_daemon._OvPhysXDaemon",
        lambda *a, **kw: _UnavailableDaemon(),
    )
    backend = OvPhysXBackend()
    with pytest.raises(OvPhysXDaemonUnavailableError, match="venv missing"):
        backend.warmup()


def test_drop_settle_dispatch_does_not_inject_judge_callback(
    fake_daemon: _FakeDaemon,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """drop_settle's evaluator must NOT see ``judge_callback`` —
    backend.judge_callback is freeform-only. Capture the kwargs the
    backend forwards to the evaluator and assert the absence directly,
    not just the shape of the returned result.
    """
    captured_kwargs: dict[str, Any] = {}

    from physics_agent.tuning.scenarios import drop_settle as drop_settle_mod

    real_evaluate = drop_settle_mod.evaluate

    def _spy_evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured_kwargs.update(kwargs)
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(drop_settle_mod, "evaluate", _spy_evaluate)

    backend = OvPhysXBackend()
    backend.judge_callback = lambda *a, **kw: {"score": 1.0, "reasoning": "x"}
    result = backend.evaluate(
        params={"mass_scale": 1.0},
        scenario=_drop_settle_scenario(),
        physics_usd=_physics_usd(tmp_path),
        seed=2,
    )
    # Wiring assertion: drop_settle evaluator was called WITHOUT
    # judge_callback, even though backend.judge_callback is set.
    assert "judge_callback" not in captured_kwargs, (
        "drop_settle dispatch must not forward judge_callback; saw "
        f"kwargs={sorted(captured_kwargs)}"
    )
    assert get_resolved_bindings(captured_kwargs["scenario"]) is not None
    assert "score" in result
    assert "settle_distance" in result


def test_freeform_dispatch_injects_judge_callback(
    fake_daemon: _FakeDaemon,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """freeform's evaluator MUST receive the backend's ``judge_callback``
    via kwargs. Capture the forwarded kwargs and assert the callable is
    present and identical to the one set on the backend — guards against
    a refactor that drops the wiring without breaking result shapes.
    """
    captured_kwargs: dict[str, Any] = {}

    def judge(
        frames: list[Path], user_prompt: str | None, observations: list[str]
    ) -> dict[str, Any]:
        return {"score": 0.7, "reasoning": "ok"}

    from physics_agent.tuning.scenarios import freeform as freeform_mod

    real_evaluate = freeform_mod.evaluate

    def _spy_evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured_kwargs.update(kwargs)
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(freeform_mod, "evaluate", _spy_evaluate)

    backend = OvPhysXBackend()
    backend.judge_callback = judge
    result = backend.evaluate(
        params={"dynamic_friction": 0.2},
        scenario=_freeform_scenario(),
        physics_usd=_physics_usd(tmp_path),
        seed=3,
    )
    # Wiring assertion: freeform evaluator received judge_callback and
    # it's the same callable the backend was configured with.
    assert "judge_callback" in captured_kwargs, (
        "freeform dispatch must forward judge_callback; saw "
        f"kwargs={sorted(captured_kwargs)}"
    )
    assert captured_kwargs["judge_callback"] is judge
    assert "score" in result
    assert "programmatic_score" in result


def test_unknown_scenario_raises(fake_daemon: _FakeDaemon, tmp_path: Path) -> None:
    # Bypass Scenario __post_init__ via direct dataclass-like fake
    class _S:
        name = "rolling_ball"
        metric = "x"
        target: dict[str, Any] = {}
        params: tuple = ()

    backend = OvPhysXBackend()
    with pytest.raises(RuntimeError, match="unknown scenario kind"):
        backend.evaluate(
            params={},
            scenario=_S(),  # type: ignore[arg-type]
            physics_usd=_physics_usd(tmp_path),
            seed=4,
        )


def test_ovphysx_rejects_newton_contact_params_before_daemon(
    fake_daemon: _FakeDaemon,
    tmp_path: Path,
) -> None:
    backend = OvPhysXBackend()
    scenario = Scenario(
        name="drop_settle",
        metric="max_bounce_height",
        target={"drop_height_m": 0.5},
        params=(TunableParam(name="contact_ke", min_value=1.0, max_value=2.0),),
    )

    with pytest.raises(TuningError, match="use --engine newton"):
        backend.evaluate(
            params={"contact_ke": 1.5},
            scenario=scenario,
            physics_usd=_physics_usd(tmp_path),
            seed=5,
        )

    assert backend._daemon is None
    assert fake_daemon.calls == []


def test_daemon_passthrough_to_evaluator(
    fake_daemon: _FakeDaemon, tmp_path: Path
) -> None:
    backend = OvPhysXBackend()
    result = backend.evaluate(
        params={"mass_scale": 1.0},
        scenario=_drop_settle_scenario(),
        physics_usd=_physics_usd(tmp_path),
        seed=5,
    )
    # The daemon got called with the expected kwargs from the
    # drop_settle evaluator.
    call = fake_daemon.calls[0]
    scene_usd = Path(call["scene_usd"])
    assert scene_usd.name == "scene.usd"
    assert Path(result["patched_usd"]).name == "patched_physics.usd"
    assert Path(result["recording_usd"]).name == "recording.usd"
    assert result["recording_usda"] == result["recording_usd"]
    assert scene_usd.exists()
    assert call["body_pattern"] == "/World/Body"
    assert call["duration_s"] == 1.0
