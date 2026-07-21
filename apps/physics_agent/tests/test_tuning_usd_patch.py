# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tuned-USD patch semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_agent.tuning.usd_patch import make_tuned_usd_path, patch_physics_usd


def _author_physics_usd(path: Path, *, mass: float = 2.0) -> Path:
    """Mirror the real ``apply_physics`` output contract:

    - ``RigidBodyAPI`` + ``MassAPI(mass)`` on the asset's default prim
      (``/Body``).
    - ``UsdPhysics.MaterialAPI`` on a sibling ``UsdShade.Material`` prim
      (``/Mat``) — that's what
      :func:`apply_physics._create_physics_material` actually authors.

    See the matching ``_physics_usd`` in ``test_tuning_runner.py``.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    body = UsdGeom.Cube.Define(stage, "/Body")
    body.CreateSizeAttr(1.0)
    body_prim = body.GetPrim()
    UsdPhysics.MassAPI.Apply(body_prim).CreateMassAttr(mass)
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    UsdPhysics.CollisionAPI.Apply(body_prim)

    mat = UsdShade.Material.Define(stage, "/Mat")
    mat_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mat_api.CreateStaticFrictionAttr(0.4)
    mat_api.CreateDynamicFrictionAttr(0.3)
    mat_api.CreateRestitutionAttr(0.2)
    stage.SetDefaultPrim(body_prim)
    stage.GetRootLayer().Save()
    return path


def test_patch_scales_mass(tmp_path: Path) -> None:
    in_path = _author_physics_usd(tmp_path / "in.usda", mass=2.0)
    out_path = tmp_path / "out.usda"
    patch_physics_usd(in_path, out_path, {"mass_scale": 1.5})

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(out_path))
    body = stage.GetPrimAtPath("/Body")
    mass_val = UsdPhysics.MassAPI(body).GetMassAttr().Get()
    assert mass_val == pytest.approx(3.0, rel=1e-6)


def test_patch_overrides_friction_and_restitution(tmp_path: Path) -> None:
    in_path = _author_physics_usd(tmp_path / "in.usda")
    out_path = tmp_path / "out.usda"
    patch_physics_usd(
        in_path,
        out_path,
        {
            "static_friction": 0.85,
            "dynamic_friction": 0.7,
            "restitution": 0.6,
        },
    )

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(out_path))
    mat = stage.GetPrimAtPath("/Mat")
    mat_api = UsdPhysics.MaterialAPI(mat)
    assert mat_api.GetStaticFrictionAttr().Get() == pytest.approx(0.85)
    assert mat_api.GetDynamicFrictionAttr().Get() == pytest.approx(0.7)
    assert mat_api.GetRestitutionAttr().Get() == pytest.approx(0.6)


def test_patch_ignores_unknown_keys(tmp_path: Path) -> None:
    in_path = _author_physics_usd(tmp_path / "in.usda")
    out_path = tmp_path / "out.usda"
    # Should not raise.
    patch_physics_usd(
        in_path,
        out_path,
        {"mass_scale": 1.0, "viscosity": 5.0, "damping": 0.1},
    )
    assert out_path.exists()


def test_patch_rejects_negative_mass_scale(tmp_path: Path) -> None:
    in_path = _author_physics_usd(tmp_path / "in.usda")
    out_path = tmp_path / "out.usda"
    with pytest.raises(ValueError, match="mass_scale must be non-negative"):
        patch_physics_usd(in_path, out_path, {"mass_scale": -0.5})


def test_patch_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        patch_physics_usd(
            tmp_path / "missing.usda", tmp_path / "out.usda", {"mass_scale": 1.0}
        )


def test_make_tuned_usd_path_canonical_name(tmp_path: Path) -> None:
    p = make_tuned_usd_path(tmp_path)
    assert p.name == "tuned_physics.usd"


def test_patch_with_no_recognised_keys_is_idempotent_copy(tmp_path: Path) -> None:
    """When tuned_params has no known keys, the output is essentially a flatten
    of the input — no values changed but the file is still emitted."""
    in_path = _author_physics_usd(tmp_path / "in.usda", mass=2.5)
    out_path = tmp_path / "out.usda"
    patch_physics_usd(in_path, out_path, {})
    assert out_path.exists()

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(out_path))
    body = stage.GetPrimAtPath("/Body")
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == pytest.approx(2.5)


def test_patch_applies_resolved_usd_attribute_bindings(tmp_path: Path) -> None:
    in_path = _author_physics_usd(tmp_path / "in.usda")
    out_path = tmp_path / "out.usda"
    patch_physics_usd(
        in_path,
        out_path,
        {"bounce_response": 0.72},
        bindings=[
            {
                "param": "bounce_response",
                "kind": "usd_attribute",
                "schema": "UsdPhysics.MaterialAPI",
                "attribute": "physics:restitution",
                "prim_paths": ["/Mat"],
            }
        ],
    )

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(out_path))
    mat = stage.GetPrimAtPath("/Mat")
    assert UsdPhysics.MaterialAPI(mat).GetRestitutionAttr().Get() == pytest.approx(0.72)


def test_patch_applies_resolved_mass_scale_bindings(tmp_path: Path) -> None:
    in_path = _author_physics_usd(tmp_path / "in.usda", mass=3.0)
    out_path = tmp_path / "out.usda"
    patch_physics_usd(
        in_path,
        out_path,
        {"mass_response": 1.25},
        bindings=[
            {
                "param": "mass_response",
                "kind": "usd_mass_scale",
                "schema": "UsdPhysics.MassAPI",
                "attribute": "physics:mass",
                "prim_paths": ["/Body"],
            }
        ],
    )

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(out_path))
    body = stage.GetPrimAtPath("/Body")
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == pytest.approx(3.75)


def test_patch_disables_regular_instances_before_writing_mass(
    tmp_path: Path,
) -> None:
    in_path = _author_physics_usd(tmp_path / "in.usda", mass=3.0)

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(in_path))
    stage.GetPrimAtPath("/Body").SetInstanceable(True)
    stage.GetRootLayer().Save()

    out_path = tmp_path / "out.usda"
    patch_physics_usd(
        in_path,
        out_path,
        {"mass_response": 1.25},
        bindings=[
            {
                "param": "mass_response",
                "kind": "usd_mass_scale",
                "schema": "UsdPhysics.MassAPI",
                "attribute": "physics:mass",
                "prim_paths": ["/Body"],
            }
        ],
    )

    stage = Usd.Stage.Open(str(out_path))
    body = stage.GetPrimAtPath("/Body")
    assert body.IsInstanceable() is False
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == pytest.approx(3.75)


def test_patch_skips_instance_proxy_bindings(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    in_path = tmp_path / "in.usda"
    stage = Usd.Stage.CreateNew(str(in_path))
    proto = UsdGeom.Xform.Define(stage, "/Proto").GetPrim()
    child = UsdGeom.Cube.Define(stage, "/Proto/Child").GetPrim()
    UsdPhysics.CollisionAPI.Apply(child)
    inst = UsdGeom.Xform.Define(stage, "/Inst").GetPrim()
    inst.GetReferences().AddInternalReference(str(proto.GetPath()))
    inst.SetInstanceable(True)
    stage.SetDefaultPrim(inst)
    stage.GetRootLayer().Save()

    stage = Usd.Stage.Open(str(in_path))
    assert stage.GetPrimAtPath("/Inst/Child").IsInstanceProxy()

    with pytest.raises(ValueError, match="did not update any USD prim"):
        patch_physics_usd(
            in_path,
            tmp_path / "out.usda",
            {"contact_ke": 12345.0},
            bindings=[
                {
                    "param": "contact_ke",
                    "kind": "usd_attribute",
                    "schema": "UsdPhysics.CollisionAPI",
                    "attribute": "newton:contact_ke",
                    "prim_paths": ["/Inst/Child"],
                }
            ],
        )


def test_patch_resolved_binding_requires_updated_prim(tmp_path: Path) -> None:
    in_path = _author_physics_usd(tmp_path / "in.usda")
    out_path = tmp_path / "out.usda"

    with pytest.raises(ValueError, match="did not update any USD prim"):
        patch_physics_usd(
            in_path,
            out_path,
            {"bounce_response": 0.72},
            bindings=[
                {
                    "param": "bounce_response",
                    "kind": "usd_attribute",
                    "schema": "UsdPhysics.MaterialAPI",
                    "attribute": "physics:restitution",
                    "prim_paths": ["/MissingMaterial"],
                }
            ],
        )


def test_patch_resolved_mass_binding_requires_updated_prim(tmp_path: Path) -> None:
    in_path = _author_physics_usd(tmp_path / "in.usda")
    out_path = tmp_path / "out.usda"

    with pytest.raises(ValueError, match="did not update any USD prim"):
        patch_physics_usd(
            in_path,
            out_path,
            {"mass_response": 1.25},
            bindings=[
                {
                    "param": "mass_response",
                    "kind": "usd_mass_scale",
                    "schema": "UsdPhysics.MassAPI",
                    "attribute": "physics:mass",
                    "prim_paths": ["/MissingBody"],
                }
            ],
        )


def test_patch_applies_newton_contact_bindings(tmp_path: Path) -> None:
    in_path = _author_physics_usd(tmp_path / "in.usda")
    out_path = tmp_path / "out.usda"
    patch_physics_usd(
        in_path,
        out_path,
        {"contact_ke": 12345.0, "contact_kd": 321.0},
        bindings=[
            {
                "param": "contact_ke",
                "kind": "usd_attribute",
                "schema": "UsdPhysics.CollisionAPI",
                "attribute": "newton:contact_ke",
                "prim_paths": ["/Body"],
            },
            {
                "param": "contact_kd",
                "kind": "usd_attribute",
                "schema": "UsdPhysics.CollisionAPI",
                "attribute": "newton:contact_kd",
                "prim_paths": ["/Body"],
            },
        ],
    )

    from pxr import Usd

    stage = Usd.Stage.Open(str(out_path))
    body = stage.GetPrimAtPath("/Body")
    assert body.GetAttribute("newton:contact_ke").Get() == pytest.approx(12345.0)
    assert body.GetAttribute("newton:contact_kd").Get() == pytest.approx(321.0)


def test_newton_import_consumes_contact_bindings(tmp_path: Path) -> None:
    newton = pytest.importorskip("newton")
    from importlib import metadata

    if tuple(map(int, metadata.version("newton").split(".")[:2])) < (1, 2):
        pytest.skip("Newton contact binding consumption requires newton>=1.2")
    if not hasattr(newton.ModelBuilder, "add_usd"):
        pytest.skip("Newton USD importer is unavailable")

    in_path = _author_physics_usd(tmp_path / "in.usda")
    out_path = tmp_path / "out.usda"
    patch_physics_usd(
        in_path,
        out_path,
        {"contact_ke": 4567.0, "contact_kd": 89.0},
        bindings=[
            {
                "param": "contact_ke",
                "kind": "usd_attribute",
                "schema": "UsdPhysics.CollisionAPI",
                "attribute": "newton:contact_ke",
                "prim_paths": ["/Body"],
            },
            {
                "param": "contact_kd",
                "kind": "usd_attribute",
                "schema": "UsdPhysics.CollisionAPI",
                "attribute": "newton:contact_kd",
                "prim_paths": ["/Body"],
            },
        ],
    )

    builder = newton.ModelBuilder()
    result = builder.add_usd(str(out_path), collapse_fixed_joints=True)
    shape_idx = result["path_shape_map"]["/Body"]
    assert builder.shape_material_ke[shape_idx] == pytest.approx(4567.0)
    assert builder.shape_material_kd[shape_idx] == pytest.approx(89.0)
