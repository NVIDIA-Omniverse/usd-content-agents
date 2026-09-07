# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the tuning runner + artifact shapes."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from physics_agent.tuning import (
    BoTorchUnavailableError,
    OvPhysXUnavailableError,
    TuneInput,
    TuningError,
    arun_tune,
    run_tune,
)
from physics_agent.tuning.artifacts import (
    ARTIFACT_BEST_PARAMS,
    ARTIFACT_HISTORY,
    ARTIFACT_REPORT,
    ARTIFACT_RESULTS,
    ARTIFACT_TUNED_USD,
)
from physics_agent.tuning.capabilities import newton_mujoco_capabilities
from physics_agent.tuning.errors import NewtonUnavailableError
from physics_agent.tuning.types import (
    SUPPORTED_PARAM_KEYS,
    Scenario,
    TunableParam,
)

# ---------- helpers ---------------------------------------------------------


def _scenario_dict() -> dict[str, Any]:
    return {
        "name": "drop_settle",
        "parameters": [
            {"name": "mass_scale", "min": 0.5, "max": 2.0},
            {"name": "static_friction", "min": 0.05, "max": 1.0},
        ],
    }


def _physics_usd(tmp_path: Path) -> Path:
    """Create a minimal physics-authored USD that mirrors the real
    ``apply_physics`` output contract:

    - RigidBodyAPI + MassAPI(mass) on the asset's default prim (``/Body``).
    - UsdPhysics.MaterialAPI on a sibling ``UsdShade.Material`` prim
      (``/Mat``) — that's what
      :func:`apply_physics._create_physics_material` actually authors.
      ``usd_patch`` walks every ``MaterialAPI``-tagged prim, so the
      friction/restitution patch needs a ``Material`` prim in the
      fixture to exercise the production code path.

    Round 6's earlier rewrite moved MaterialAPI onto ``/Body`` to match
    a CodeRabbit thread that misread the contract; reverted in round 8
    after re-reading ``apply_physics._create_physics_material`` (which
    applies MaterialAPI to a ``UsdShade.Material`` prim, not the body).
    """
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    p = tmp_path / "physics.usda"
    stage = Usd.Stage.CreateNew(str(p))
    body = UsdGeom.Xform.Define(stage, "/Body")
    body_prim = body.GetPrim()
    mass_api = UsdPhysics.MassAPI.Apply(body_prim)
    mass_api.CreateMassAttr(2.0)
    UsdPhysics.RigidBodyAPI.Apply(body_prim)

    mat = UsdShade.Material.Define(stage, "/Mat")
    mat_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mat_api.CreateStaticFrictionAttr(0.4)
    mat_api.CreateDynamicFrictionAttr(0.3)
    mat_api.CreateRestitutionAttr(0.2)

    stage.SetDefaultPrim(body_prim)
    stage.GetRootLayer().Save()
    return p


# ---------- happy path ------------------------------------------------------


def test_run_tune_random_emits_all_artifacts(tmp_path: Path) -> None:
    """A full random-optimizer run writes the canonical 5 artifacts."""
    out = tmp_path / "out"
    physics = _physics_usd(tmp_path)
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=physics,
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=5,
            seed=42,
        )
    )
    assert result.success
    assert result.n_trials == 5
    assert result.optimizer_used == "random"
    assert result.engine_used == "fake"
    assert set(result.best_params.keys()) == {"mass_scale", "static_friction"}
    assert result.best_objective == result.best_score

    # Five canonical artifacts on disk.
    for name in (
        ARTIFACT_BEST_PARAMS,
        ARTIFACT_RESULTS,
        ARTIFACT_HISTORY,
        ARTIFACT_REPORT,
        ARTIFACT_TUNED_USD,
    ):
        assert (out / name).exists(), f"Missing artifact: {name}"

    # best_params.json has stable schema.
    bp = json.loads((out / ARTIFACT_BEST_PARAMS).read_text())
    assert sorted(bp.keys()) == ["best_score", "params"]
    assert sorted(bp["params"].keys()) == ["mass_scale", "static_friction"]

    # history.jsonl has one record per trial, each line valid JSON.
    history_lines = (out / ARTIFACT_HISTORY).read_text().strip().splitlines()
    assert len(history_lines) == 5
    for line in history_lines:
        rec = json.loads(line)
        assert {"trial_index", "params", "score", "objective_value"}.issubset(
            rec.keys()
        )
        assert rec["objective_value"] == rec["score"]

    # tune_results.json has reproducibility-relevant fields.
    tr = json.loads((out / ARTIFACT_RESULTS).read_text())
    assert tr["scenario"]["name"] == "drop_settle"
    assert tr["config"]["seed"] == 42
    assert tr["config"]["optimizer"] == "random"
    assert tr["config"]["engine"] == "fake"
    assert tr["n_trials"] == 5
    assert tr["objective"] == {
        "name": "synthetic_parameter_error",
        "unit": "unitless",
        "direction": "minimize",
        "best_value": result.best_objective,
    }
    assert tr["best"]["objective_value"] == result.best_objective


def test_run_tune_random_minimises_fake_score(tmp_path: Path) -> None:
    """The fake backend has a known optimum; random search across 50 trials
    should find a substantially better score than the worst trial."""
    physics = _physics_usd(tmp_path)
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=physics,
            output_dir=tmp_path / "out",
            engine="fake",
            optimizer="random",
            max_trials=50,
            seed=0,
        )
    )
    scores = [t.score for t in result.history]
    assert min(scores) < max(scores)
    assert result.best_score == min(scores)


def test_run_tune_cma_es_finds_better_than_random_on_average(
    tmp_path: Path,
) -> None:
    """CMA-ES should converge — best score < worst trial score."""
    physics = _physics_usd(tmp_path)
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=physics,
            output_dir=tmp_path / "out",
            engine="fake",
            optimizer="cma-es",
            max_trials=15,
            seed=1,
        )
    )
    assert result.success
    assert result.optimizer_used == "cma-es"
    assert result.n_trials >= 1
    # Optimizer-quality assertions (CodeRabbit Round 11 thread #15): the
    # original test only checked success/trial count, so a regression that
    # made CMA-ES return the worst-trial score as the best would slip
    # through. The fake backend is deterministic-with-spread, so any sane
    # optimizer must pick the minimum trial as best (lower is better) and
    # that minimum must beat the maximum sampled score by at least
    # something for a non-degenerate run.
    scores = [t.score for t in result.history]
    assert result.best_score == min(scores)
    assert result.best_score < max(scores)


# ---------- patching tuned USD ---------------------------------------------


def test_run_tune_patches_tuned_usd_with_best_params(tmp_path: Path) -> None:
    physics = _physics_usd(tmp_path)
    out = tmp_path / "out"
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=physics,
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=3,
            seed=0,
        )
    )
    tuned = out / ARTIFACT_TUNED_USD
    assert tuned.exists()

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(tuned))
    # Mass attribute should now reflect the original 2.0 * mass_scale.
    body = stage.GetPrimAtPath("/Body")
    assert body.IsValid()
    mass = UsdPhysics.MassAPI(body).GetMassAttr().Get()
    expected = 2.0 * result.best_params["mass_scale"]
    assert mass == pytest.approx(expected, rel=1e-6)


def test_run_tune_persists_explicit_scenario_gravity_in_customer_units(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    physics = _physics_usd(tmp_path)
    stage = Usd.Stage.Open(str(physics))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    scene = UsdPhysics.Scene.Define(stage, "/Body/PhysicsScene")
    scene.CreateGravityMagnitudeAttr(9.81)
    stage.GetRootLayer().Save()
    scenario = _scenario_dict()
    scenario["target"] = {"gravity": -1.62}
    output = tmp_path / "out"

    result = run_tune(
        TuneInput(
            scenario=scenario,
            physics_usd=physics,
            output_dir=output,
            engine="fake",
            optimizer="random",
            max_trials=2,
            seed=0,
        )
    )

    assert result.success
    tuned = Usd.Stage.Open(str(output / ARTIFACT_TUNED_USD))
    tuned_scene = UsdPhysics.Scene(tuned.GetPrimAtPath("/Body/PhysicsScene"))
    assert UsdGeom.GetStageMetersPerUnit(tuned) == pytest.approx(0.01)
    assert tuned_scene.GetGravityMagnitudeAttr().Get() == pytest.approx(162.0)
    assert tuple(tuned_scene.GetGravityDirectionAttr().Get()) == (0.0, 0.0, -1.0)


def test_run_tune_keeps_artifact_when_gravity_scene_is_ambiguous(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from pxr import Usd, UsdPhysics

    physics = _physics_usd(tmp_path)
    stage = Usd.Stage.Open(str(physics))
    UsdPhysics.Scene.Define(stage, "/PhysicsSceneA")
    UsdPhysics.Scene.Define(stage, "/PhysicsSceneB")
    stage.GetRootLayer().Save()
    scenario = _scenario_dict()
    scenario["target"] = {"gravity": -9.81}
    output = tmp_path / "out"

    result = run_tune(
        TuneInput(
            scenario=scenario,
            physics_usd=physics,
            output_dir=output,
            engine="fake",
            optimizer="random",
            max_trials=2,
            seed=0,
        )
    )

    assert result.success
    assert (output / ARTIFACT_TUNED_USD).is_file()
    assert "multiple active PhysicsScene prims" in caplog.text


# ---------- async API mirror -----------------------------------------------


@pytest.mark.asyncio
async def test_arun_tune_is_async(tmp_path: Path) -> None:
    physics = _physics_usd(tmp_path)
    result = await arun_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=physics,
            output_dir=tmp_path / "out",
            engine="fake",
            optimizer="random",
            max_trials=3,
            seed=0,
        )
    )
    assert result.success
    assert result.n_trials == 3


# ---------- error paths -----------------------------------------------------


def test_run_tune_botorch_missing_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning import optimizers

    monkeypatch.setattr(optimizers, "is_botorch_available", lambda: False)
    physics = _physics_usd(tmp_path)
    with pytest.raises(BoTorchUnavailableError) as ei:
        run_tune(
            TuneInput(
                scenario=_scenario_dict(),
                physics_usd=physics,
                output_dir=tmp_path / "out",
                engine="fake",
                optimizer="auto",
                max_trials=3,
                seed=0,
            )
        )
    assert "BoTorch optimizer requires the tuning extra" in str(ei.value)


def test_run_tune_explicit_botorch_missing_raises_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--optimizer botorch` must NEVER fall back silently."""
    from physics_agent.tuning import optimizers

    monkeypatch.setattr(optimizers, "is_botorch_available", lambda: False)
    physics = _physics_usd(tmp_path)
    with pytest.raises(BoTorchUnavailableError):
        run_tune(
            TuneInput(
                scenario=_scenario_dict(),
                physics_usd=physics,
                output_dir=tmp_path / "out",
                engine="fake",
                optimizer="botorch",
                max_trials=3,
                seed=0,
            )
        )


def test_run_tune_ovphysx_missing_raises_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning import backend as backend_mod

    monkeypatch.setattr(
        backend_mod,
        "load_ovphysx_backend",
        lambda: (_ for _ in ()).throw(OvPhysXUnavailableError()),
    )
    physics = _physics_usd(tmp_path)
    with pytest.raises(OvPhysXUnavailableError) as ei:
        run_tune(
            TuneInput(
                scenario=_scenario_dict(),
                physics_usd=physics,
                output_dir=tmp_path / "out",
                engine="ovphysx",
                optimizer="random",
                max_trials=3,
                seed=0,
            )
        )
    assert "OvPhysX backend requires the tuning extra" in str(ei.value)


def test_run_tune_newton_setup_failure_is_not_per_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning import backend as backend_mod

    evaluate_calls = 0

    class _BrokenNewtonBackend:
        name = "newton"

        def warmup(self) -> None:
            pass

        def tuning_capabilities(self):
            return newton_mujoco_capabilities()

        def evaluate(self, **kwargs: Any) -> dict[str, Any]:
            nonlocal evaluate_calls
            evaluate_calls += 1
            raise NewtonUnavailableError("missing Newton importer")

    monkeypatch.setattr(
        backend_mod,
        "load_newton_backend",
        lambda: _BrokenNewtonBackend(),
    )
    physics = _physics_usd(tmp_path)

    with pytest.raises(NewtonUnavailableError, match="missing Newton importer"):
        run_tune(
            TuneInput(
                scenario={
                    "name": "drop_settle",
                    "parameters": [
                        {"name": "mass_scale", "min": 0.5, "max": 2.0},
                    ],
                },
                physics_usd=physics,
                output_dir=tmp_path / "out",
                engine="newton",
                optimizer="random",
                max_trials=3,
                seed=0,
            )
        )
    assert evaluate_calls == 1
    assert (tmp_path / "out" / ARTIFACT_HISTORY).read_text() == ""


def test_run_tune_reraises_typed_ovphysx_daemon_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Typed daemon failures cannot become a string-only failed TuneOutput."""
    from world_understanding.functions.physics.ovphysx_daemon import (
        OvPhysXDaemonUnavailableError,
    )

    import physics_agent.tuning.runner as runner_mod

    daemon_error = OvPhysXDaemonUnavailableError(
        "runtime-controlled daemon detail that must not become TuneOutput.error"
    )
    evaluate_calls = 0

    class _DaemonFailingBackend:
        name = "fake"

        def evaluate(
            self,
            params: dict[str, float],
            scenario: Scenario,
            physics_usd: Path,
            *,
            seed: int,
        ) -> dict[str, Any]:
            nonlocal evaluate_calls
            evaluate_calls += 1
            raise daemon_error

    monkeypatch.setattr(
        runner_mod,
        "get_backend",
        lambda _engine: _DaemonFailingBackend(),
    )

    with pytest.raises(OvPhysXDaemonUnavailableError) as exc_info:
        run_tune(
            TuneInput(
                scenario=_scenario_dict(),
                physics_usd=_physics_usd(tmp_path),
                output_dir=tmp_path / "out",
                engine="fake",
                optimizer="random",
                max_trials=3,
                seed=0,
            )
        )

    assert exc_info.value is daemon_error
    assert evaluate_calls == 1
    assert (tmp_path / "out" / ARTIFACT_HISTORY).read_text() == ""


def test_run_tune_ovphysx_missing_precedes_invalid_scenario_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning import backend as backend_mod

    monkeypatch.setattr(
        backend_mod,
        "load_ovphysx_backend",
        lambda: (_ for _ in ()).throw(OvPhysXUnavailableError()),
    )
    physics = _physics_usd(tmp_path)

    with pytest.raises(OvPhysXUnavailableError):
        run_tune(
            TuneInput(
                scenario={
                    "name": "not_a_real_kind",
                    "parameters": [
                        {"name": "mass_scale", "min": 0.5, "max": 2.0},
                    ],
                },
                physics_usd=physics,
                output_dir=tmp_path / "out",
                engine="ovphysx",
                optimizer="random",
                max_trials=3,
                seed=0,
            )
        )


def test_run_tune_rejects_invalid_max_trials(tmp_path: Path) -> None:
    physics = _physics_usd(tmp_path)
    with pytest.raises(ValueError, match="max_trials must be > 0"):
        run_tune(
            TuneInput(
                scenario=_scenario_dict(),
                physics_usd=physics,
                output_dir=tmp_path / "out",
                engine="fake",
                optimizer="random",
                max_trials=0,
                seed=0,
            )
        )


def test_run_tune_ovphysx_missing_precedes_malformed_scenario_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning import backend as backend_mod

    monkeypatch.setattr(
        backend_mod,
        "load_ovphysx_backend",
        lambda: (_ for _ in ()).throw(OvPhysXUnavailableError()),
    )
    physics = _physics_usd(tmp_path)
    scenario_path = tmp_path / "broken.yaml"
    scenario_path.write_text("name: [unterminated\n", encoding="utf-8")

    with pytest.raises(OvPhysXUnavailableError):
        run_tune(
            TuneInput(
                scenario=scenario_path,
                physics_usd=physics,
                output_dir=tmp_path / "out",
                engine="ovphysx",
                optimizer="random",
                max_trials=3,
                seed=0,
            )
        )


def test_run_tune_ovphysx_missing_precedes_non_mapping_scenario_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physics_agent.tuning import backend as backend_mod

    monkeypatch.setattr(
        backend_mod,
        "load_ovphysx_backend",
        lambda: (_ for _ in ()).throw(OvPhysXUnavailableError()),
    )
    physics = _physics_usd(tmp_path)
    scenario_path = tmp_path / "list.yaml"
    scenario_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(OvPhysXUnavailableError):
        run_tune(
            TuneInput(
                scenario=scenario_path,
                physics_usd=physics,
                output_dir=tmp_path / "out",
                engine="ovphysx",
                optimizer="random",
                max_trials=3,
                seed=0,
            )
        )


def test_run_tune_rejects_unknown_engine(tmp_path: Path) -> None:
    physics = _physics_usd(tmp_path)
    with pytest.raises(ValueError, match="Unknown engine"):
        run_tune(
            TuneInput(
                scenario=_scenario_dict(),
                physics_usd=physics,
                output_dir=tmp_path / "out",
                engine="mujoco",
                optimizer="random",
                max_trials=3,
                seed=0,
            )
        )


def test_run_tune_rejects_newton_restitution_before_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import physics_agent.tuning.runner as runner_mod

    class _NoTrialBackend:
        name = "newton"

        def __init__(self) -> None:
            self.warmup_calls = 0
            self.evaluate_calls = 0

        def warmup(self) -> None:
            self.warmup_calls += 1

        def evaluate(
            self,
            params: dict[str, float],
            scenario: Scenario,
            physics_usd: Path,
            *,
            seed: int,
        ) -> dict[str, Any]:
            self.evaluate_calls += 1
            return {"score": 0.0}

    backend = _NoTrialBackend()
    get_backend_calls = 0

    def _get_backend(_engine: str) -> _NoTrialBackend:
        nonlocal get_backend_calls
        get_backend_calls += 1
        return backend

    monkeypatch.setattr(runner_mod, "get_backend", _get_backend)
    physics = _physics_usd(tmp_path)

    with pytest.raises(TuningError, match="bouncy/max_bounce_height"):
        run_tune(
            TuneInput(
                scenario={
                    "name": "drop_settle",
                    "metric": "max_bounce_height",
                    "target": {"drop_height_m": 0.5, "duration_s": 1.0},
                    "parameters": [
                        {"name": "restitution", "min": 0.0, "max": 1.0},
                    ],
                },
                physics_usd=physics,
                output_dir=tmp_path / "out",
                engine="newton",
                optimizer="random",
                max_trials=3,
                seed=0,
            )
        )

    assert get_backend_calls == 0
    assert backend.warmup_calls == 0
    assert backend.evaluate_calls == 0


def test_run_tune_rejects_newton_static_friction_before_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import physics_agent.tuning.runner as runner_mod

    class _NoTrialBackend:
        name = "newton"

        def __init__(self) -> None:
            self.warmup_calls = 0
            self.evaluate_calls = 0

        def warmup(self) -> None:
            self.warmup_calls += 1

        def evaluate(
            self,
            params: dict[str, float],
            scenario: Scenario,
            physics_usd: Path,
            *,
            seed: int,
        ) -> dict[str, Any]:
            self.evaluate_calls += 1
            return {"score": 0.0}

    backend = _NoTrialBackend()
    get_backend_calls = 0

    def _get_backend(_engine: str) -> _NoTrialBackend:
        nonlocal get_backend_calls
        get_backend_calls += 1
        return backend

    monkeypatch.setattr(runner_mod, "get_backend", _get_backend)
    physics = _physics_usd(tmp_path)

    with pytest.raises(TuningError, match="static_friction-only trials"):
        run_tune(
            TuneInput(
                scenario={
                    "name": "drop_settle",
                    "metric": "settle_distance",
                    "target": {"drop_height_m": 0.5, "duration_s": 1.0},
                    "parameters": [
                        {"name": "static_friction", "min": 0.05, "max": 1.0},
                    ],
                },
                physics_usd=physics,
                output_dir=tmp_path / "out",
                engine="newton",
                optimizer="random",
                max_trials=3,
                seed=0,
            )
        )

    assert get_backend_calls == 0
    assert backend.warmup_calls == 0
    assert backend.evaluate_calls == 0


def test_run_tune_rejects_ovphysx_newton_contact_params_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import physics_agent.tuning.runner as runner_mod

    get_backend_calls = 0

    def _unexpected_get_backend(engine: str) -> Any:
        nonlocal get_backend_calls
        get_backend_calls += 1
        raise AssertionError("backend should not be constructed")

    monkeypatch.setattr(runner_mod, "get_backend", _unexpected_get_backend)
    physics = _physics_usd(tmp_path)

    with pytest.raises(TuningError, match="contact_ke"):
        run_tune(
            TuneInput(
                scenario={
                    "name": "drop_settle",
                    "metric": "settle_distance",
                    "target": {"drop_height_m": 0.5, "duration_s": 1.0},
                    "parameters": [
                        {"name": "contact_ke", "min": 100.0, "max": 100000.0},
                    ],
                },
                physics_usd=physics,
                output_dir=tmp_path / "out",
                engine="ovphysx",
                optimizer="random",
                max_trials=3,
                seed=0,
            )
        )

    assert get_backend_calls == 0


# ---------- cancellation ----------------------------------------------------


def test_run_tune_pre_set_cancel_event_returns_cleanly(tmp_path: Path) -> None:
    """A cancel event already set at start should produce a cancelled
    TuneOutput with 0 trials, NOT raise TuningError."""
    physics = _physics_usd(tmp_path)
    cancel = threading.Event()
    cancel.set()

    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=physics,
            output_dir=tmp_path / "out",
            engine="fake",
            optimizer="random",
            max_trials=10,
            seed=0,
            cancel_event=cancel,
        )
    )
    assert result.cancelled is True
    assert result.n_trials == 0
    assert result.success is False
    # history.jsonl is opened+closed even on pre-cancel (a 0-byte file is fine).
    assert (tmp_path / "out" / "history.jsonl").exists()


def test_run_tune_honours_cancel_event(tmp_path: Path) -> None:
    """The runner must honour an Event-like cancel signal between trials."""
    physics = _physics_usd(tmp_path)
    cancel = threading.Event()

    # Set the cancel after construction; the runner must check before each
    # trial so a pre-set event terminates after exactly 0 trials. To keep
    # the test deterministic we set it after the first trial starts via
    # an EventListener.
    class CancelAfterFirst:
        def __init__(self) -> None:
            self.first_seen = False

        def info(self, *args, **kwargs) -> None: ...
        def debug(self, *args, **kwargs) -> None: ...
        def warning(self, *args, **kwargs) -> None: ...
        def error(self, *args, **kwargs) -> None: ...

        def event(self, event_type: str, data: dict[str, Any], **kwargs: Any) -> None:
            if event_type == "tune.trial.completed" and not self.first_seen:
                self.first_seen = True
                cancel.set()

    listener = CancelAfterFirst()
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=physics,
            output_dir=tmp_path / "out",
            engine="fake",
            optimizer="random",
            max_trials=20,
            seed=0,
            cancel_event=cancel,
            event_listener=listener,
        )
    )
    # We did at least 1 trial but stopped early.
    assert result.cancelled is True
    assert 1 <= result.n_trials < 20
    # Artifacts still emitted on cancel.
    assert (tmp_path / "out" / ARTIFACT_BEST_PARAMS).exists()
    assert (tmp_path / "out" / ARTIFACT_HISTORY).exists()


# ---------- artifact correctness -------------------------------------------


def test_history_jsonl_is_line_flushed(tmp_path: Path) -> None:
    """Each history line is JSON; line-flushed for SSE consumers."""
    physics = _physics_usd(tmp_path)
    out = tmp_path / "out"
    run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=physics,
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=4,
            seed=0,
        )
    )
    raw = (out / ARTIFACT_HISTORY).read_text()
    # Trailing newline is fine, but no extra blank lines mid-file.
    lines = [line for line in raw.split("\n") if line]
    assert len(lines) == 4
    for line in lines:
        assert json.loads(line)["trial_index"] in {0, 1, 2, 3}


def test_supported_param_keys_match_backend_contracts() -> None:
    # Legacy portable material params plus Newton MuJoCo contact params.
    assert set(SUPPORTED_PARAM_KEYS) == {
        "mass_scale",
        "static_friction",
        "dynamic_friction",
        "restitution",
        "contact_ke",
        "contact_kd",
    }


def test_scenario_dataclass_validates_param_name() -> None:
    with pytest.raises(ValueError, match="Unsupported tunable parameter"):
        Scenario(
            name="drop_settle",
            params=(TunableParam(name="elasticity", min_value=0.0, max_value=1.0),),
            target={},
            metric="x",
        )


def test_scenario_dataclass_clip_clamps_to_bounds() -> None:
    p = TunableParam(name="mass_scale", min_value=0.5, max_value=2.0)
    assert p.clip(-1.0) == 0.5
    assert p.clip(0.5) == 0.5
    assert p.clip(1.5) == 1.5
    assert p.clip(5.0) == 2.0


def test_scenario_rejects_empty_params() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Scenario(name="drop_settle", params=(), target={}, metric="x")


def test_runner_records_failed_trial_on_non_numeric_score(tmp_path: Path) -> None:
    """A backend returning a non-numeric score is recorded as failed, not raised."""
    from physics_agent.tuning import backend as backend_mod
    from physics_agent.tuning.scenario import parse_scenario

    class BrokenBackend:
        name = "fake"

        def evaluate(
            self,
            params: dict[str, float],
            scenario: Any,
            physics_usd: Path,
            *,
            seed: int,
        ) -> dict[str, Any]:
            return {"score": "not a number"}

    physics = _physics_usd(tmp_path)
    out = tmp_path / "out"
    # Smoke-check the scenario dict round-trips through ``parse_scenario``
    # before we use it below — catches schema regressions in
    # ``_scenario_dict()`` that would otherwise show up as a confusing
    # tune-side error.
    parsed = parse_scenario(_scenario_dict())
    assert parsed.name == "drop_settle"

    # Patch get_backend to return our broken backend.
    original_get_backend = backend_mod.get_backend

    def fake_get_backend(engine: str):
        if engine == "fake":
            return BrokenBackend()
        return original_get_backend(engine)

    import physics_agent.tuning.runner as runner_mod

    monkey = pytest.MonkeyPatch()
    monkey.setattr(runner_mod, "get_backend", fake_get_backend)
    try:
        result = run_tune(
            TuneInput(
                scenario=_scenario_dict(),
                physics_usd=physics,
                output_dir=out,
                engine="fake",
                optimizer="random",
                max_trials=2,
                seed=0,
            )
        )
    finally:
        monkey.undo()
    assert result.n_trials == 2
    assert all(t.failed for t in result.history)
    # Best param fall-back returns the lowest-score (still inf) trial; the
    # run is marked unsuccessful via the cancel/best-failed path.
    assert result.best_score == float("inf")
    # Artifacts should still be emitted even when every trial fails.
    assert (out / "best_params.json").exists()
    # Round 13 (CodeRabbit thread #5): TuneOutput.error must surface a
    # readable failure reason when every trial fails, not None.
    assert result.success is False
    assert result.error is not None
    # Either propagated from best.error (preferred) or the generic
    # fallback message — both are acceptable, but must not be None.
    assert "fail" in result.error.lower() or "not a number" in result.error.lower()


def test_run_tune_full_determinism_under_same_seed(tmp_path: Path) -> None:
    """Same scenario + seed must produce identical history + best params."""
    physics = _physics_usd(tmp_path)

    a = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=physics,
            output_dir=tmp_path / "a",
            engine="fake",
            optimizer="random",
            max_trials=8,
            seed=12345,
        )
    )
    b = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=physics,
            output_dir=tmp_path / "b",
            engine="fake",
            optimizer="random",
            max_trials=8,
            seed=12345,
        )
    )
    assert [t.params for t in a.history] == [t.params for t in b.history]
    assert [t.score for t in a.history] == [t.score for t in b.history]
    assert a.best_params == b.best_params


def test_run_tune_results_artifact_full_schema(tmp_path: Path) -> None:
    """tune_results.json contains every reproducibility-critical field."""
    physics = _physics_usd(tmp_path)
    out = tmp_path / "out"
    run_tune(
        TuneInput(
            scenario={
                "name": "drop_settle",
                "metric": "settle_distance",
                "target": {"drop_height_m": 0.5, "duration_s": 2.0},
                "parameters": [
                    {"name": "mass_scale", "min": 0.5, "max": 2.0},
                ],
                "scenario_specific_knob": "value",
            },
            physics_usd=physics,
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=2,
            seed=7,
        )
    )
    tr = json.loads((out / ARTIFACT_RESULTS).read_text())
    # Reproducibility fields.
    assert tr["scenario"]["name"] == "drop_settle"
    assert tr["scenario"]["metric"] == "settle_distance"
    assert tr["scenario"]["target"]["drop_height_m"] == 0.5
    assert tr["scenario"]["extra"]["scenario_specific_knob"] == "value"
    assert tr["scenario"]["parameters"][0]["name"] == "mass_scale"
    # Config + portability.
    assert tr["config"]["engine"] == "fake"
    assert tr["config"]["optimizer"] == "random"
    assert tr["config"]["max_trials"] == 2
    assert tr["config"]["seed"] == 7
    assert tr["config"]["physics_usd_basename"] == "physics.usda"
    # Run stats.
    assert tr["n_trials"] == 2
    assert "started_at" in tr
    assert "completed_at" in tr
    assert tr["cancelled"] is False
    assert "history_summary" in tr
    assert len(tr["history_summary"]) == 2
    assert "best" in tr
    assert "score" in tr["best"]
    assert tr["best"]["objective_value"] is not None
    assert tr["objective"]["best_value"] == tr["best"]["objective_value"]
    assert all(item["objective_value"] is not None for item in tr["history_summary"])


def test_run_tune_history_jsonl_is_strict_json(tmp_path: Path) -> None:
    """Every history line is parseable by strict JSON consumers (no Infinity/NaN).

    Failed trials carry score=inf in memory; the writer must coerce this to
    null so browsers / jq / SSE clients can parse every event.
    """
    from physics_agent.tuning import backend as backend_mod

    class FailingBackend:
        name = "fake"

        def evaluate(
            self,
            params: dict[str, float],
            scenario: Any,
            physics_usd: Path,
            *,
            seed: int,
        ) -> dict[str, Any]:
            raise RuntimeError("synthetic failure")

    physics = _physics_usd(tmp_path)
    monkey = pytest.MonkeyPatch()
    import physics_agent.tuning.runner as runner_mod

    monkey.setattr(runner_mod, "get_backend", lambda eng: FailingBackend())
    try:
        run_tune(
            TuneInput(
                scenario=_scenario_dict(),
                physics_usd=physics,
                output_dir=tmp_path / "out",
                engine="fake",
                optimizer="random",
                max_trials=2,
                seed=0,
            )
        )
    finally:
        monkey.undo()
    raw = (tmp_path / "out" / ARTIFACT_HISTORY).read_text()
    # Strict JSON parser — would raise on Infinity / NaN.
    for line in raw.strip().splitlines():
        rec = json.loads(line)
        # score should be None (JSON null), not the Python Infinity literal.
        assert rec["score"] is None
        assert rec["failed"] is True


def test_run_tune_pre_cancel_emits_full_artifact_set(tmp_path: Path) -> None:
    """Pre-first-trial cancel still emits all 4 JSON artifacts."""
    physics = _physics_usd(tmp_path)
    cancel = threading.Event()
    cancel.set()
    out = tmp_path / "out"
    result = run_tune(
        TuneInput(
            scenario=_scenario_dict(),
            physics_usd=physics,
            output_dir=out,
            engine="fake",
            optimizer="random",
            max_trials=10,
            seed=0,
            cancel_event=cancel,
        )
    )
    assert result.cancelled is True
    for name in (
        ARTIFACT_BEST_PARAMS,
        ARTIFACT_HISTORY,
        ARTIFACT_RESULTS,
        ARTIFACT_REPORT,
    ):
        assert (out / name).exists()


@pytest.mark.parametrize(
    "bad_score",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
    ids=["nan", "+inf", "-inf"],
)
def test_runner_records_failed_trial_on_non_finite_score(
    tmp_path: Path, bad_score: float
) -> None:
    """Any non-finite backend score (NaN, +inf, -inf) is a failed trial.

    The runner used to only reject NaN; +/-inf would slip through and a
    backend overflow returning ``-inf`` could win the sweep (it compares
    less than every finite candidate). All non-finite scores must now be
    recorded as failures with an ``inf`` placeholder so the optimizer
    never picks them.
    """

    class BadBackend:
        name = "fake"

        def evaluate(
            self,
            params: dict[str, float],
            scenario: Any,
            physics_usd: Path,
            *,
            seed: int,
        ) -> dict[str, Any]:
            return {"score": bad_score}

    physics = _physics_usd(tmp_path)

    monkey = pytest.MonkeyPatch()
    # The runner imported ``get_backend`` into its own namespace, so we
    # only need to patch ``runner_mod.get_backend``; patching
    # ``backend_mod.get_backend`` would have no effect on the runner.
    import physics_agent.tuning.runner as runner_mod

    monkey.setattr(runner_mod, "get_backend", lambda eng: BadBackend())
    try:
        result = run_tune(
            TuneInput(
                scenario=_scenario_dict(),
                physics_usd=physics,
                output_dir=tmp_path / "out",
                engine="fake",
                optimizer="random",
                max_trials=2,
                seed=0,
            )
        )
    finally:
        monkey.undo()
    assert all(t.failed for t in result.history)
    assert any("non-finite" in (t.error or "") for t in result.history)
    # The recorded sentinel is exactly ``+inf`` per the runner contract.
    # (A previous version also asserted ``not math.isfinite(score)`` but
    # since every trial is failed and ``score == inf``, that check is
    # redundant — CR thread.)
    for t in result.history:
        assert t.score == float("inf"), t.score


def test_run_tune_is_callable_from_inside_event_loop(tmp_path: Path) -> None:
    """``run_tune`` must work even when an event loop is already running
    in the calling thread.

    The previous ``asyncio.run(arun_tune(params))`` wrapper raised
    ``RuntimeError: asyncio.run() cannot be called from a running event
    loop`` whenever a notebook or async test harness invoked the
    blocking entry point. The synchronous body now runs directly so the
    sync API stays usable everywhere.
    """
    import asyncio

    physics = _physics_usd(tmp_path)
    out = tmp_path / "out"

    def _call_blocking() -> None:
        run_tune(
            TuneInput(
                scenario=_scenario_dict(),
                physics_usd=physics,
                output_dir=out,
                engine="fake",
                optimizer="random",
                max_trials=1,
                seed=0,
            )
        )

    async def _driver() -> None:
        # Doesn't await — the point is that we're *inside* a running loop
        # when ``run_tune`` is called.
        _call_blocking()

    asyncio.run(_driver())
    # If we got here, run_tune did not raise the legacy "cannot be called
    # from a running event loop" RuntimeError.
    assert (out / ARTIFACT_RESULTS).exists()
