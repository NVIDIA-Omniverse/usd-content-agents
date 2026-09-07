# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the 2026-07-09 world-understanding agent benchmark round:

  * task-12 — `material <material-ref> --color …` must edit the referenced material
    in place, never author a NEW material and bind it onto the Material prim,
  * task-02 — `material --name <N>` must author the material AT /Looks/<N> so
    pre-existing (dangling) binding relationships targeting that exact path resolve;
    an incompatible occupant is an ERROR, never a silent `<N>_2` suffix,
  * task-03 — bulk `--where 'name~=T*'` must not over-match `tn__…` prims
    (strict fnmatchcase semantics; the fix lives in usd_core/selector.py — xfail),
  * task-14 — `material-binding` on a path with no prim must fail with a clean
    error, not pxr's "Accessed schema on invalid prim".

GPU-free: in-memory stages + direct Session calls, no daemon, no rendering.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pxr")


def _world_stage():
    """A minimal stage with a /World defaultPrim and one mesh (Looks lands at
    /World/Looks, matching the benchmark scenes)."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    UsdGeom.Mesh.Define(stage, "/World/Cube")
    return stage


@pytest.fixture()
def session():
    from pxr import UsdGeom

    from usd_core.config import Config
    from usd_core.session import Session

    s = Session(Config())
    s.new()
    UsdGeom.Mesh.Define(s._stage, "/World/Cube")
    s._index_prims()
    return s


# -- task-12: material-ref targets are edited in place, never bound onto ----------


def test_update_material_edits_preview_shader_in_place():
    from pxr import UsdShade
    from usd_core import materials

    stage = _world_stage()
    mat = materials.create_preview_material(stage, name="Steel_Painted_White",
                                            color=(1, 1, 1), roughness=0.6)
    out = materials.update_material(stage, mat, color=(1, 0, 0), roughness=0.2)
    assert out == {"material_path": mat, "kind": "UsdPreviewSurface",
                   "updated": ["diffuseColor", "roughness"]}

    d = materials.describe_material(stage, mat)
    assert d["inputs"]["diffuseColor"] == pytest.approx([1, 0, 0])
    assert d["inputs"]["roughness"] == pytest.approx(0.2)
    assert d["inputs"]["metallic"] == pytest.approx(0.0)  # untouched create-time value
    # in place: still exactly one material under Looks, and it stays unbound
    looks = stage.GetPrimAtPath("/World/Looks")
    mats = [p for p in looks.GetChildren() if UsdShade.Material(p)]
    assert [p.GetPath().pathString for p in mats] == [mat]
    assert materials.bound_material(stage, mat)["binding_type"] == "none"


def test_update_material_only_touches_supplied_params():
    from usd_core import materials

    stage = _world_stage()
    mat = materials.create_preview_material(stage, name="Paint", color=(0, 0, 1),
                                            roughness=0.9)
    materials.update_material(stage, mat, metallic=1.0)
    d = materials.describe_material(stage, mat)
    assert d["inputs"]["diffuseColor"] == pytest.approx([0, 0, 1])
    assert d["inputs"]["roughness"] == pytest.approx(0.9)
    assert d["inputs"]["metallic"] == pytest.approx(1.0)


def test_update_material_maps_mdl_constants():
    from usd_core import materials

    stage = _world_stage()
    mat = materials.create_mdl_material(stage, name="Coating", color=(1, 1, 1))
    materials.update_material(stage, mat, color=(0, 1, 0), opacity=0.5,
                              inputs={"glass_ior": 1.4})
    d = materials.describe_material(stage, mat)
    assert d["kind"] == "mdl"
    assert d["inputs"]["diffuse_color_constant"] == pytest.approx([0, 1, 0])
    assert d["inputs"]["opacity_constant"] == pytest.approx(0.5)
    assert d["inputs"]["enable_opacity"] is True
    assert d["inputs"]["glass_ior"] == pytest.approx(1.4)
    # preview-only params have no OmniPBR constant — refuse rather than guess
    with pytest.raises(ValueError, match="no OmniPBR mapping"):
        materials.update_material(stage, mat, ior=1.5)


def test_update_material_constant_beats_stale_texture_connection():
    from usd_core import materials

    stage = _world_stage()
    mat = materials.create_preview_material(stage, name="Painted",
                                            diffuse_texture="albedo.png")
    materials.update_material(stage, mat, color=(0, 0, 1))
    d = materials.describe_material(stage, mat)
    # were the texture still connected, describe would report "<- diffuseTex (…)"
    assert d["inputs"]["diffuseColor"] == pytest.approx([0, 0, 1])


def test_update_material_rejects_non_materials():
    from usd_core import materials

    stage = _world_stage()
    with pytest.raises(ValueError, match="not a UsdShade.Material"):
        materials.update_material(stage, "/World/Cube", color=(1, 0, 0))


def test_bind_material_refuses_a_material_target():
    from usd_core import materials

    stage = _world_stage()
    m1 = materials.create_preview_material(stage, name="A", color=(1, 0, 0))
    m2 = materials.create_preview_material(stage, name="B", color=(0, 1, 0))
    with pytest.raises(ValueError, match="onto a material"):
        materials.bind_material(stage, m2, m1)
    assert materials.bound_material(stage, m1)["binding_type"] == "none"


def test_material_verb_on_a_material_ref_updates_in_place(session):
    """task-12: `material <material-ref> --color …` silently created a new material
    and bound it ONTO the Material prim (`@m1 ← material @m6`). session.material()
    now routes Material-prim targets to update_material: the shader is edited in
    place, nothing new is created or bound."""
    from usd_core.materials import bound_material, describe_material

    r = session.material(ref="/World/Cube", name="Steel_Painted_White", color=[1, 1, 1])
    assert r.ok, r.issues
    mat_path = "/World/Looks/Steel_Painted_White"

    r2 = session.material(ref=mat_path, color=[1, 0, 0])
    assert r2.ok, r2.issues
    assert "diffuseColor" in r2.summary["params"]
    # the referenced material was edited, not re-bound or clobbered
    assert bound_material(session._stage, mat_path)["binding_type"] == "none"
    d = describe_material(session._stage, mat_path)
    assert d["inputs"]["diffuseColor"] == pytest.approx([1, 0, 0])
    # no orphan material appeared next to it
    looks = session._stage.GetPrimAtPath("/World/Looks")
    mats = [c.GetName() for c in looks.GetChildren()]
    assert mats == ["Steel_Painted_White"]


# -- task-02: author AT the requested path; never silently suffix -----------------


def test_material_authored_at_exact_path_repairs_dangling_bindings():
    from pxr import Sdf, UsdShade
    from usd_core import materials

    stage = _world_stage()
    mesh = stage.GetPrimAtPath("/World/Cube")
    # the benchmark scene: a binding rel dangling at a missing /World/Looks/* path,
    # with only an empty over prim (attrs=0, no type) at the target
    UsdShade.MaterialBindingAPI.Apply(mesh)
    mesh.CreateRelationship("material:binding", custom=False).SetTargets(
        [Sdf.Path("/World/Looks/Plastic_Orange")])
    stage.OverridePrim("/World/Looks/Plastic_Orange")

    mat = materials.create_preview_material(stage, name="Plastic_Orange",
                                            color=(1.0, 0.4, 0.0))
    assert mat == "/World/Looks/Plastic_Orange"  # exactly there, no _2
    prim = stage.GetPrimAtPath(mat)
    assert prim.IsDefined() and UsdShade.Material(prim)
    assert not stage.GetPrimAtPath("/World/Looks/Plastic_Orange_2").IsValid()
    # the pre-existing binding now resolves
    bound, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
    assert bound and bound.GetPath().pathString == mat


def test_material_name_collision_with_typed_prim_errors_instead_of_suffixing():
    from pxr import UsdGeom
    from usd_core import materials

    stage = _world_stage()
    UsdGeom.Mesh.Define(stage, "/World/Looks/Plastic_Orange")
    with pytest.raises(ValueError, match="already exists as a Mesh"):
        materials.create_preview_material(stage, name="Plastic_Orange",
                                          color=(1.0, 0.4, 0.0))
    assert not stage.GetPrimAtPath("/World/Looks/Plastic_Orange_2").IsValid()


def test_recreating_a_material_name_updates_in_place_not_suffixed():
    from usd_core import materials

    stage = _world_stage()
    m1 = materials.create_preview_material(stage, name="Steel", color=(1, 1, 1))
    m2 = materials.create_preview_material(stage, name="Steel", color=(1, 0, 0))
    assert m1 == m2 == "/World/Looks/Steel"
    assert not stage.GetPrimAtPath("/World/Looks/Steel_2").IsValid()
    d = materials.describe_material(stage, m1)
    assert d["inputs"]["diffuseColor"] == pytest.approx([1, 0, 0])


def test_invalid_material_name_is_a_clean_error():
    from usd_core import materials

    stage = _world_stage()
    with pytest.raises(ValueError, match="not a valid material name"):
        materials.create_preview_material(stage, name="Plastic Orange!",
                                          color=(1.0, 0.4, 0.0))


def test_import_library_material_lands_on_dangling_target_path(tmp_path):
    from pxr import Usd, UsdShade
    from usd_core import materials

    lib_file = str(tmp_path / "lib.usda")
    lib = Usd.Stage.CreateNew(lib_file)
    UsdShade.Material.Define(lib, "/Looks/Steel")
    UsdShade.Shader.Define(lib, "/Looks/Steel/Shader")
    lib.Save()

    stage = _world_stage()
    stage.OverridePrim("/World/Looks/Steel")  # empty over at the requested path
    local = materials.import_library_material(stage, library_path=lib_file,
                                              material_name="Steel")
    assert local == "/World/Looks/Steel"
    assert UsdShade.Material(stage.GetPrimAtPath(local))
    assert not stage.GetPrimAtPath("/World/Looks/Steel_2").IsValid()


# -- task-03: strict --where glob semantics (fixed in selector.py) ----------------


def test_where_name_glob_is_case_sensitive_and_anchored():
    from pxr import Usd, UsdGeom
    from usd_core import selector

    stage = Usd.Stage.CreateInMemory()
    for nm in ("Table_01", "Tray", "tn__Board_01", "tn__Board_02", "Chair"):
        UsdGeom.Mesh.Define(stage, f"/World/{nm}")
    hits = {p.GetName() for p in selector.select_prims(stage, where=["name~=T*"])}
    assert hits == {"Table_01", "Tray"}


# -- task-14: missing prim → clean error, not "Accessed schema on invalid prim" ---


def test_bound_material_on_missing_prim_raises_clean_valueerror():
    from usd_core import materials

    stage = _world_stage()
    with pytest.raises(ValueError, match="no prim at /some/raw/path"):
        materials.bound_material(stage, "/some/raw/path")


def test_material_binding_query_on_missing_prim_returns_clean_error(session):
    r = session.material_binding("/some/raw/path")
    assert not r.ok
    assert "no prim at /some/raw/path" in r.issues[0].message
    assert "Accessed schema on invalid prim" not in r.issues[0].message
