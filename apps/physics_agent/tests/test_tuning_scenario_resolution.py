# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for backend-aware tuning parameter resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from physics_agent.tuning.backend import FakeBackend
from physics_agent.tuning.capabilities import capabilities_for_backend
from physics_agent.tuning.errors import TuningError
from physics_agent.tuning.newton_backend import NewtonBackend
from physics_agent.tuning.ovphysx_backend import OvPhysXBackend
from physics_agent.tuning.scenario import load_scenario, parse_scenario
from physics_agent.tuning.scenario_resolution import (
    RESOLVED_BINDINGS_EXTRA_KEY,
    resolve_scenario_bindings,
)
from physics_agent.tuning.types import Scenario


def _author_physics_usd(
    path: Path,
    *,
    include_collision: bool = True,
    include_material: bool = True,
    include_authored_mass: bool = True,
) -> Path:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    body = UsdGeom.Cube.Define(stage, "/Body")
    body.CreateSizeAttr(1.0)
    body_prim = body.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    if include_collision:
        UsdPhysics.CollisionAPI.Apply(body_prim)
    mass_api = UsdPhysics.MassAPI.Apply(body_prim)
    if include_authored_mass:
        mass_api.CreateMassAttr(2.0)

    if include_material:
        mat = UsdShade.Material.Define(stage, "/Mat")
        mat_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        mat_api.CreateStaticFrictionAttr(0.4)
        mat_api.CreateDynamicFrictionAttr(0.3)
        mat_api.CreateRestitutionAttr(0.2)

    stage.SetDefaultPrim(body_prim)
    stage.GetRootLayer().Save()
    return path


def _legacy_scenario() -> Scenario:
    return parse_scenario(
        {
            "name": "drop_settle",
            "metric": "settle_distance",
            "parameters": [
                {"name": "mass_scale", "min": 0.5, "max": 2.0},
                {"name": "static_friction", "min": 0.05, "max": 1.5},
                {"name": "dynamic_friction", "min": 0.05, "max": 1.5},
                {"name": "restitution", "min": 0.0, "max": 1.0},
            ],
        }
    )


def _bindings_by_param(scenario: Scenario) -> dict[str, dict[str, Any]]:
    raw = scenario.extra[RESOLVED_BINDINGS_EXTRA_KEY]
    assert isinstance(raw, list)
    return {str(b["param"]): b for b in raw}


def test_legacy_params_resolve_to_ovphysx_usd_bindings(tmp_path: Path) -> None:
    physics_usd = _author_physics_usd(tmp_path / "physics.usda")
    resolved = resolve_scenario_bindings(
        _legacy_scenario(),
        physics_usd=physics_usd,
        backend=OvPhysXBackend(),
    )

    bindings = _bindings_by_param(resolved)
    assert set(bindings) == {
        "mass_scale",
        "static_friction",
        "dynamic_friction",
        "restitution",
    }
    assert bindings["restitution"]["concept"] == "bounce_response"
    assert bindings["restitution"]["kind"] == "usd_attribute"
    assert bindings["restitution"]["attribute"] == "physics:restitution"
    assert bindings["restitution"]["prim_paths"] == ["/Mat"]
    assert bindings["mass_scale"]["kind"] == "usd_mass_scale"
    assert bindings["mass_scale"]["prim_paths"] == ["/Body"]


def test_tire_bounce_yaml_resolves_for_ovphysx() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    scenario = load_scenario(
        repo_root / "apps/physics_agent/configs/tuning/tire_b01_drop_settle.yaml"
    )
    tire_usd = repo_root / "apps/physics_agent/data/examples/Tire_B01/tire.usdc"

    resolved = resolve_scenario_bindings(
        scenario,
        physics_usd=tire_usd,
        backend=OvPhysXBackend(),
    )

    bindings = _bindings_by_param(resolved)
    assert set(bindings) == {
        "mass_scale",
        "static_friction",
        "dynamic_friction",
        "restitution",
    }
    assert bindings["restitution"]["attribute"] == "physics:restitution"


def test_internal_physics_tire_yaml_resolves_for_ovphysx() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    scenario_path = (
        repo_root
        / "apps/physics_agent/configs/internal/tire_b01_internal_physics_drop_settle.yaml"
    )
    if not scenario_path.exists():
        pytest.skip("internal-only Tire_B01 scenario is excluded from public staging")
    scenario = load_scenario(scenario_path)
    tire_usd = repo_root / "apps/physics_agent/data/examples/Tire_B01/tire.usdc"

    resolved = resolve_scenario_bindings(
        scenario,
        physics_usd=tire_usd,
        backend=OvPhysXBackend(),
    )

    assert scenario.metric == "settle_distance"
    bindings = _bindings_by_param(resolved)
    assert set(bindings) == {
        "mass_scale",
        "static_friction",
        "dynamic_friction",
        "restitution",
    }
    assert bindings["restitution"]["attribute"] == "physics:restitution"


def test_fake_backend_uses_same_resolution_for_runner_smoke(
    tmp_path: Path,
) -> None:
    physics_usd = _author_physics_usd(tmp_path / "physics.usda")
    resolved = resolve_scenario_bindings(
        _legacy_scenario(),
        physics_usd=physics_usd,
        backend=FakeBackend(),
    )

    bindings = _bindings_by_param(resolved)
    assert bindings["restitution"]["backend"] == "fake"
    assert bindings["dynamic_friction"]["attribute"] == "physics:dynamicFriction"


def test_missing_usd_material_fails_before_trials(tmp_path: Path) -> None:
    physics_usd = _author_physics_usd(
        tmp_path / "physics.usda",
        include_material=False,
    )
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "restitution", "min": 0.0, "max": 1.0}],
        }
    )

    with pytest.raises(TuningError, match="Could not resolve tunable parameter"):
        resolve_scenario_bindings(
            scenario,
            physics_usd=physics_usd,
            backend=OvPhysXBackend(),
        )


def test_missing_authored_mass_fails_mass_scale_resolution(
    tmp_path: Path,
) -> None:
    physics_usd = _author_physics_usd(
        tmp_path / "physics.usda",
        include_authored_mass=False,
    )
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "mass_scale", "min": 0.5, "max": 2.0}],
        }
    )

    with pytest.raises(TuningError, match="Could not resolve tunable parameter"):
        resolve_scenario_bindings(
            scenario,
            physics_usd=physics_usd,
            backend=OvPhysXBackend(),
        )


def test_newton_declares_mujoco_consumed_capabilities() -> None:
    capabilities = NewtonBackend().tuning_capabilities()
    assert capabilities
    assert {c.param_name for c in capabilities} == {
        "mass_scale",
        "dynamic_friction",
        "contact_ke",
        "contact_kd",
    }
    by_name = {c.param_name: c for c in capabilities}
    assert by_name["contact_ke"].attribute == "newton:contact_ke"
    assert by_name["contact_kd"].attribute == "newton:contact_kd"
    assert by_name["dynamic_friction"].attribute == "physics:dynamicFriction"


def test_unknown_capability_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        capabilities_for_backend("ovphsyx")


def test_backend_without_capability_provider_must_be_known(
    tmp_path: Path,
) -> None:
    class MysteryBackend:
        name = "mystery"

    physics_usd = _author_physics_usd(tmp_path / "physics.usda")

    with pytest.raises(TuningError, match="does not declare tuning capabilities"):
        resolve_scenario_bindings(
            _legacy_scenario(),
            physics_usd=physics_usd,
            backend=MysteryBackend(),
        )


def test_newton_rejects_legacy_restitution_for_mujoco(tmp_path: Path) -> None:
    physics_usd = _author_physics_usd(tmp_path / "physics.usda")
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "restitution", "min": 0.0, "max": 1.0}],
        }
    )

    with pytest.raises(TuningError, match="does not support tunable parameter"):
        resolve_scenario_bindings(
            scenario,
            physics_usd=physics_usd,
            backend=NewtonBackend(),
        )


def test_newton_rejects_static_friction_for_mujoco(tmp_path: Path) -> None:
    physics_usd = _author_physics_usd(tmp_path / "physics.usda")
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "static_friction", "min": 0.05, "max": 1.5}],
        }
    )

    with pytest.raises(TuningError, match="does not support tunable parameter"):
        resolve_scenario_bindings(
            scenario,
            physics_usd=physics_usd,
            backend=NewtonBackend(),
        )


def test_ovphysx_rejects_newton_contact_params(tmp_path: Path) -> None:
    physics_usd = _author_physics_usd(tmp_path / "physics.usda")
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "contact_ke", "min": 100.0, "max": 100000.0}],
        }
    )

    with pytest.raises(TuningError, match="does not support tunable parameter"):
        resolve_scenario_bindings(
            scenario,
            physics_usd=physics_usd,
            backend=OvPhysXBackend(),
        )


def test_newton_dynamic_friction_resolves_to_material_attrs(tmp_path: Path) -> None:
    physics_usd = _author_physics_usd(tmp_path / "physics.usda")
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "dynamic_friction", "min": 0.05, "max": 1.5},
            ],
        }
    )

    resolved = resolve_scenario_bindings(
        scenario,
        physics_usd=physics_usd,
        backend=NewtonBackend(),
    )

    bindings = _bindings_by_param(resolved)
    assert set(bindings) == {"dynamic_friction"}
    assert bindings["dynamic_friction"]["schema"] == "UsdPhysics.MaterialAPI"
    assert bindings["dynamic_friction"]["attribute"] == "physics:dynamicFriction"
    assert bindings["dynamic_friction"]["prim_paths"] == ["/Mat"]


def test_newton_contact_params_resolve_to_collision_attrs(tmp_path: Path) -> None:
    physics_usd = _author_physics_usd(tmp_path / "physics.usda")
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "contact_ke", "min": 100.0, "max": 100000.0},
                {"name": "contact_kd", "min": 0.0, "max": 5000.0},
            ],
        }
    )

    resolved = resolve_scenario_bindings(
        scenario,
        physics_usd=physics_usd,
        backend=NewtonBackend(),
    )

    bindings = _bindings_by_param(resolved)
    assert set(bindings) == {"contact_ke", "contact_kd"}
    assert bindings["contact_ke"]["schema"] == "UsdPhysics.CollisionAPI"
    assert bindings["contact_ke"]["attribute"] == "newton:contact_ke"
    assert bindings["contact_ke"]["prim_paths"] == ["/Body"]
    assert bindings["contact_kd"]["attribute"] == "newton:contact_kd"


def test_newton_contact_params_resolve_all_collision_prims(tmp_path: Path) -> None:
    physics_usd = _author_physics_usd(tmp_path / "physics.usda")

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(physics_usd))
    other = UsdGeom.Cube.Define(stage, "/OtherCollider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(other)
    stage.GetRootLayer().Save()

    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "contact_ke", "min": 100.0, "max": 100000.0},
            ],
        }
    )

    resolved = resolve_scenario_bindings(
        scenario,
        physics_usd=physics_usd,
        backend=NewtonBackend(),
    )

    bindings = _bindings_by_param(resolved)
    assert set(bindings["contact_ke"]["prim_paths"]) == {"/Body", "/OtherCollider"}


def test_newton_contact_params_require_collision_api(tmp_path: Path) -> None:
    physics_usd = _author_physics_usd(
        tmp_path / "physics.usda",
        include_collision=False,
    )
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "contact_ke", "min": 100.0, "max": 100000.0}],
        }
    )

    with pytest.raises(TuningError, match="Could not resolve tunable parameter"):
        resolve_scenario_bindings(
            scenario,
            physics_usd=physics_usd,
            backend=NewtonBackend(),
        )


def test_tire_newton_contact_params_resolve_to_collision_mesh() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    tire_usd = repo_root / "apps/physics_agent/data/examples/Tire_B01/tire.usdc"
    scenario = load_scenario(
        repo_root / "apps/physics_agent/configs/tuning/tire_b01_drop_settle_newton.yaml"
    )

    resolved = resolve_scenario_bindings(
        scenario,
        physics_usd=tire_usd,
        backend=NewtonBackend(),
    )

    bindings = _bindings_by_param(resolved)
    mesh_path = (
        "/RootNode/Geometry/wheelAssemblytire_b01_obj_00/wheelAssemblytire_b01_mesh_00"
    )
    assert set(bindings) == {"mass_scale", "contact_ke", "contact_kd"}
    assert bindings["contact_ke"]["prim_paths"] == [mesh_path]
    assert bindings["contact_kd"]["prim_paths"] == [mesh_path]
