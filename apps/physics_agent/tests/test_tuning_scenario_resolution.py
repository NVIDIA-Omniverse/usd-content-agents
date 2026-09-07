# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for backend-aware tuning parameter resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from physics_agent.tuning.backend import FakeBackend
from physics_agent.tuning.capabilities import (
    BINDING_KIND_SIMULATOR_PARAMETER,
    BINDING_KIND_USD_ATTRIBUTE,
    BindingCapability,
    capabilities_for_backend,
)
from physics_agent.tuning.errors import TuningError
from physics_agent.tuning.newton_backend import NewtonBackend
from physics_agent.tuning.ovphysx_backend import OvPhysXBackend
from physics_agent.tuning.scenario import load_scenario, parse_scenario
from physics_agent.tuning.scenario_resolution import (
    AUTHORED_BOUND_MULTIPLIER,
    RESOLVED_BINDINGS_EXTRA_KEY,
    resolve_scenario_bindings,
    resolve_scenario_parameter_bounds,
)
from physics_agent.tuning.types import Scenario


def _author_physics_usd(
    path: Path,
    *,
    include_collision: bool = True,
    include_material: bool = True,
    include_authored_mass: bool = True,
    static_friction: float = 0.4,
    dynamic_friction: float = 0.3,
    restitution: float = 0.2,
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
        mat_api.CreateStaticFrictionAttr(static_friction)
        mat_api.CreateDynamicFrictionAttr(dynamic_friction)
        mat_api.CreateRestitutionAttr(restitution)

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


def test_omitted_bounds_resolve_from_authored_usd_values(tmp_path: Path) -> None:
    physics_usd = _author_physics_usd(tmp_path / "physics.usda")
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "mass_scale"},
                {"name": "static_friction"},
                {"name": "dynamic_friction"},
                {"name": "restitution"},
            ],
        }
    )

    resolved = resolve_scenario_bindings(
        scenario,
        physics_usd=physics_usd,
        backend=OvPhysXBackend(),
    )

    params = resolved.param_dict()
    assert params["mass_scale"].min_value == pytest.approx(
        1.0 / AUTHORED_BOUND_MULTIPLIER
    )
    assert params["mass_scale"].max_value == pytest.approx(AUTHORED_BOUND_MULTIPLIER)
    for name, authored_value in (
        ("static_friction", 0.4),
        ("dynamic_friction", 0.3),
        ("restitution", 0.2),
    ):
        assert params[name].min_value == pytest.approx(
            authored_value / AUTHORED_BOUND_MULTIPLIER
        )
        assert params[name].max_value == pytest.approx(
            authored_value * AUTHORED_BOUND_MULTIPLIER
        )
    assert resolved.auto_bound_fields == {}


def test_missing_authored_values_use_capability_defaults(tmp_path: Path) -> None:
    physics_usd = _author_physics_usd(tmp_path / "physics.usda")
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "contact_ke"}, {"name": "contact_kd"}],
        }
    )

    resolved = resolve_scenario_parameter_bounds(
        scenario,
        physics_usd=physics_usd,
        backend=NewtonBackend(),
    )

    assert resolved.param_dict()["contact_ke"].min_value == 100.0
    assert resolved.param_dict()["contact_ke"].max_value == 100000.0
    assert resolved.param_dict()["contact_kd"].min_value == 0.0
    assert resolved.param_dict()["contact_kd"].max_value == 5000.0


def test_authored_zero_uses_capability_default_range(tmp_path: Path) -> None:
    physics_usd = _author_physics_usd(
        tmp_path / "physics.usda",
        restitution=0.0,
    )
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "restitution"}],
        }
    )

    resolved = resolve_scenario_parameter_bounds(
        scenario,
        physics_usd=physics_usd,
        backend=OvPhysXBackend(),
    )

    assert resolved.param_dict()["restitution"].min_value == 0.0
    assert resolved.param_dict()["restitution"].max_value == 1.0


def test_authored_restitution_bounds_are_clamped_to_physical_range(
    tmp_path: Path,
) -> None:
    physics_usd = _author_physics_usd(
        tmp_path / "physics.usda",
        restitution=1.0,
    )
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "restitution"}],
        }
    )

    resolved = resolve_scenario_parameter_bounds(
        scenario,
        physics_usd=physics_usd,
        backend=OvPhysXBackend(),
    )

    restitution = resolved.param_dict()["restitution"]
    assert restitution.min_value == pytest.approx(1.0 / AUTHORED_BOUND_MULTIPLIER)
    assert restitution.max_value == 1.0


def test_conflicting_authored_friction_uses_capability_defaults(
    tmp_path: Path,
) -> None:
    physics_usd = _author_physics_usd(
        tmp_path / "physics.usda",
        static_friction=0.3,
        dynamic_friction=0.5,
    )
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "static_friction"},
                {"name": "dynamic_friction"},
            ],
        }
    )

    resolved = resolve_scenario_parameter_bounds(
        scenario,
        physics_usd=physics_usd,
        backend=OvPhysXBackend(),
    )

    static_friction = resolved.param_dict()["static_friction"]
    dynamic_friction = resolved.param_dict()["dynamic_friction"]
    assert (static_friction.min_value, static_friction.max_value) == (0.05, 1.5)
    assert (dynamic_friction.min_value, dynamic_friction.max_value) == (0.05, 1.5)


def test_missing_sibling_value_does_not_hide_authored_value(tmp_path: Path) -> None:
    physics_usd = _author_physics_usd(tmp_path / "physics.usda")

    from pxr import Usd, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(physics_usd))
    missing_material = UsdShade.Material.Define(stage, "/MissingMaterial")
    UsdPhysics.MaterialAPI.Apply(missing_material.GetPrim())
    stage.GetRootLayer().Save()

    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "restitution"}],
        }
    )

    resolved = resolve_scenario_parameter_bounds(
        scenario,
        physics_usd=physics_usd,
        backend=OvPhysXBackend(),
    )

    restitution = resolved.param_dict()["restitution"]
    assert restitution.min_value == pytest.approx(0.2 / AUTHORED_BOUND_MULTIPLIER)
    assert restitution.max_value == pytest.approx(0.2 * AUTHORED_BOUND_MULTIPLIER)


def test_simulator_only_parameter_requires_explicit_bounds(tmp_path: Path) -> None:
    class SimulatorOnlyBackend:
        name = "simulator-only"

        @staticmethod
        def tuning_capabilities() -> tuple[BindingCapability, ...]:
            return (
                BindingCapability(
                    param_name="restitution",
                    concept="bounce_response",
                    binding_kind=BINDING_KIND_SIMULATOR_PARAMETER,
                    simulator_parameter="solver.restitution",
                    default_range=(0.0, 1.0),
                ),
            )

    physics_usd = _author_physics_usd(tmp_path / "physics.usda")
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [{"name": "restitution"}],
        }
    )

    with pytest.raises(
        TuningError,
        match=r"simulator-only parameter 'restitution'.*Specify min and max",
    ):
        resolve_scenario_parameter_bounds(
            scenario,
            physics_usd=physics_usd,
            backend=SimulatorOnlyBackend(),
        )


def test_explicit_bounds_remain_authoritative(tmp_path: Path) -> None:
    physics_usd = _author_physics_usd(tmp_path / "physics.usda")
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "restitution", "min": 0.6, "max": 0.9},
            ],
        }
    )

    resolved = resolve_scenario_parameter_bounds(
        scenario,
        physics_usd=physics_usd,
        backend=OvPhysXBackend(),
    )

    assert resolved is scenario
    assert resolved.param_dict()["restitution"].min_value == 0.6
    assert resolved.param_dict()["restitution"].max_value == 0.9


def test_disjoint_automatic_friction_fallbacks_fail_resolution(
    tmp_path: Path,
) -> None:
    class DisjointFallbackBackend:
        name = "disjoint-fallback"

        @staticmethod
        def tuning_capabilities() -> tuple[BindingCapability, ...]:
            return (
                BindingCapability(
                    param_name="static_friction",
                    concept="surface_grip",
                    binding_kind=BINDING_KIND_USD_ATTRIBUTE,
                    schema="UsdPhysics.MaterialAPI",
                    attribute="physics:staticFriction",
                    default_range=(0.01, 0.02),
                ),
                BindingCapability(
                    param_name="dynamic_friction",
                    concept="surface_grip",
                    binding_kind=BINDING_KIND_USD_ATTRIBUTE,
                    schema="UsdPhysics.MaterialAPI",
                    attribute="physics:dynamicFriction",
                    default_range=(0.05, 0.1),
                ),
            )

    physics_usd = _author_physics_usd(
        tmp_path / "physics.usda",
        static_friction=0.01,
        dynamic_friction=0.1,
    )
    scenario = parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "static_friction"},
                {"name": "dynamic_friction"},
            ],
        }
    )

    with pytest.raises(TuningError, match="remain infeasible after capability"):
        resolve_scenario_parameter_bounds(
            scenario,
            physics_usd=physics_usd,
            backend=DisjointFallbackBackend(),
        )


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
