# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for explicit low-level physics schema authoring."""

from __future__ import annotations

import pytest

from pxr import Usd, UsdGeom


def test_validate_density_zero_ok_with_authored_mass():
    """density 0 means 'derive from density' and is only invalid without a mass to override
    it; with an explicit mass>0 USD ignores density, so validate must not flag it."""
    from usd_core import physics

    s = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(s, "/World/Body")
    physics.define_physics_scene(s, "/World/PhysicsScene")
    physics.apply_rigid_body(s, "/World/Body", mass=10.0, density=0.0)
    assert not any("density" in i for i in physics.validate_schema(s)["issues"])

    s2 = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(s2, "/World/B")
    physics.define_physics_scene(s2, "/World/PhysicsScene")
    physics.apply_rigid_body(s2, "/World/B", density=0.0)  # no mass
    assert any("density 0" in i for i in physics.validate_schema(s2)["issues"])


def test_apply_physics_material_updates_the_explicit_path_in_place():
    """The low-level helper updates the exact path supplied by its caller."""
    from usd_core import physics

    s = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(s, "/World/M")
    p1 = physics.apply_physics_material(
        s,
        path="/World/PhysicsMaterial",
        static_friction=0.6,
        restitution=0.5,
    )
    p2 = physics.apply_physics_material(
        s,
        path="/World/PhysicsMaterial",
        static_friction=0.7,
        restitution=0.4,
    )
    assert p1 == p2
    mats = [pr.GetPath().pathString for pr in s.Traverse() if pr.GetTypeName() == "Material"]
    assert mats == [p1]


def test_apply_physics_material_does_not_retype_existing_prim():
    from usd_core import physics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Looks/Physics_component")

    with pytest.raises(ValueError, match="existing prim has type 'Xform'"):
        physics.apply_physics_material(
            stage,
            path="/World/Looks/Physics_component",
            static_friction=0.6,
        )

    assert stage.GetPrimAtPath("/World/Looks/Physics_component").GetTypeName() == "Xform"


def test_apply_physics_material_does_not_overlay_visual_material():
    from pxr import UsdShade, UsdPhysics
    from usd_core import physics

    stage = Usd.Stage.CreateInMemory()
    visual = UsdShade.Material.Define(stage, "/World/Looks/Physics_component")

    with pytest.raises(ValueError, match="not already a physics material"):
        physics.apply_physics_material(
            stage,
            path="/World/Looks/Physics_component",
            static_friction=0.6,
        )

    assert not visual.GetPrim().HasAPI(UsdPhysics.MaterialAPI)
