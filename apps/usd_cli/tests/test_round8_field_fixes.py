# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-8 field fixes from the task-03/task-12 trace analyses.

Covers:
  1  `properties @ref ATTR...` — multiple attribute names in one call (agents
     batched shader-param reads and hit ~50 usage errors); unknown names get
     close-match hints without failing the good ones
  2  `verify` surfaces external dependencies (reference arcs / asset paths that
     resolve outside the file) — it PASSed a deliverable with 10 live refs into
     ../shared/ because dead_sublayers was its only portability check. External
     deps stay a note, not a FAIL (shared libraries are legitimate in place).
  3  a flattened deliverable reports NO external deps

All GPU-free: direct Session calls.
"""
from __future__ import annotations

from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils

from usd_core.config import Config
from usd_core.session import Session


def _define_bound_material(stage, mat_path, target):
    """A UsdPreviewSurface material bound to `target` — keeps verify's
    unbound/shader checks green so the tests exercise ONLY the new behavior."""
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(target.GetPrim()).Bind(mat)
    return mat


def _scene_with_colored_cube(tmp_path, name="scene.usda"):
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/World/cube")
    cube.CreateDisplayColorAttr([(0.9, 0.1, 0.1)])
    cube.CreateSizeAttr(2.0)
    _define_bound_material(stage, "/World/Looks/Red", cube)
    path = tmp_path / name
    stage.GetRootLayer().Export(str(path))
    return path


def _session_on(tmp_path, path) -> Session:
    s = Session(Config(project_dir=tmp_path), name="default")
    s.open_stage(str(path))
    return s


def test_properties_multiple_attrs_one_call(tmp_path):
    s = _session_on(tmp_path, _scene_with_colored_cube(tmp_path))
    ref = s.refs.ref_for_path("/World/cube")
    r = s.properties(ref, attr=["size", "primvars:displayColor"])
    assert r.ok
    assert r.summary["attrs"] == 2
    attrs = r.data["attrs"]
    assert attrs["size"]["value"] == 2.0
    assert attrs["primvars:displayColor"]["value"] == [[0.9, 0.1, 0.1]]
    assert f"{ref}.size (double) = 2.0" in r.data["text"]


def test_properties_bad_attr_among_good_does_not_fail(tmp_path):
    s = _session_on(tmp_path, _scene_with_colored_cube(tmp_path))
    ref = s.refs.ref_for_path("/World/cube")
    r = s.properties(ref, attr=["size", "displaycolor"])
    assert r.ok  # the good attr still comes back
    assert r.data["attrs"]["size"]["value"] == 2.0
    assert "error" in r.data["attrs"]["displaycolor"]
    assert "close matches" in r.data["text"]


def test_verify_reports_external_reference_deps(tmp_path):
    # a "library" file in a sibling dir + a deliverable that references it
    lib_dir = tmp_path / "shared"
    lib_dir.mkdir()
    lib = Usd.Stage.CreateInMemory()
    looks = UsdGeom.Scope.Define(lib, "/World")
    lib.SetDefaultPrim(looks.GetPrim())
    green = UsdShade.Material.Define(lib, "/World/Looks/Green")
    shader = UsdShade.Shader.Define(lib, "/World/Looks/Green/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    green.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    lib.GetRootLayer().Export(str(lib_dir / "materials.usda"))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    deliv = Usd.Stage.CreateNew(str(out_dir / "deliverable.usda"))
    world = UsdGeom.Xform.Define(deliv, "/World")
    deliv.SetDefaultPrim(world.GetPrim())
    mat_prim = deliv.DefinePrim("/World/Looks/Green")
    mat_prim.GetReferences().AddReference("../shared/materials.usda",
                                          "/World/Looks/Green")
    cube = UsdGeom.Cube.Define(deliv, "/World/cube")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
        UsdShade.Material(mat_prim))
    deliv.GetRootLayer().Save()
    del deliv

    s = Session(Config(project_dir=tmp_path), name="default")
    r = s.verify(str(out_dir / "deliverable.usda"))
    assert r.ok  # external deps are a note, not a FAIL
    assert r.summary["external_deps"] == 1
    assert r.data["external_deps"] == ["../shared/materials.usda"]
    assert "NOT self-contained" in r.data["text"]
    assert "save --flatten" in r.data["text"]


def test_verify_flattened_file_has_no_external_deps(tmp_path):
    path = _scene_with_colored_cube(tmp_path)
    s = Session(Config(project_dir=tmp_path), name="default")
    r = s.verify(str(path))
    assert r.ok
    assert "external_deps" not in r.summary
    assert "NOT self-contained" not in r.data["text"]


def test_verify_self_contained_usdz_has_no_external_deps(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    texture = source / "albedo.png"
    texture.write_bytes(b"fixture texture")

    nested_path = source / "nested.usda"
    nested = Usd.Stage.CreateNew(str(nested_path))
    cube = UsdGeom.Cube.Define(nested, "/World/cube")
    material = UsdShade.Material.Define(nested, "/World/Looks/Textured")
    preview = UsdShade.Shader.Define(nested, "/World/Looks/Textured/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    texture_shader = UsdShade.Shader.Define(
        nested, "/World/Looks/Textured/Texture"
    )
    texture_shader.CreateIdAttr("UsdUVTexture")
    texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("albedo.png")
    )
    texture_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture_shader.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(
        preview.ConnectableAPI(), "surface"
    )
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    nested.GetRootLayer().Save()

    root_path = source / "root.usda"
    root = Usd.Stage.CreateNew(str(root_path))
    world = UsdGeom.Xform.Define(root, "/World")
    root.SetDefaultPrim(world.GetPrim())
    root.GetRootLayer().subLayerPaths = ["nested.usda"]
    root.GetRootLayer().Save()

    package = tmp_path / "deliverable.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(root_path), str(package))

    session = Session(Config(project_dir=tmp_path), name="default")
    result = session.verify(str(package), strict=True)
    assert result.ok, result.data["text"]
    assert "external_deps" not in result.summary
    assert "NOT self-contained" not in result.data["text"]


def _scene_two_cubes(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    a = UsdGeom.Cube.Define(stage, "/World/a")
    b = UsdGeom.Cube.Define(stage, "/World/b")
    _define_bound_material(stage, "/World/Looks/M", a)
    UsdShade.MaterialBindingAPI.Apply(b.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath("/World/Looks/M")))
    # /World/b carries a PRIOR authored opinion isolate must put back
    UsdGeom.Imageable(b.GetPrim()).GetVisibilityAttr().Set("inherited")
    path = tmp_path / "two.usda"
    stage.GetRootLayer().Export(str(path))
    return path


def test_isolate_restore_removes_authored_opinions(tmp_path):
    s = _session_on(tmp_path, _scene_two_cubes(tmp_path))
    ra = s.refs.ref_for_path("/World/a")
    r = s.isolate([ra])
    assert r.ok and r.summary["isolated"] == 1
    layer = s._stage.GetRootLayer()
    from pxr import Sdf
    assert layer.GetObjectAtPath(Sdf.Path("/World/b.visibility")) is not None

    r = s.isolate(restore=True)
    assert r.ok and r.summary["restored"] >= 1
    # /World/b's PRIOR authored token is back (not deleted, not 'invisible')
    spec = layer.GetObjectAtPath(Sdf.Path("/World/b.visibility"))
    assert spec is not None and str(spec.default) == "inherited"
    from pxr import UsdGeom as _ug
    assert (_ug.Imageable(s._stage.GetPrimAtPath("/World/b")).ComputeVisibility()
            != _ug.Tokens.invisible)
    # nothing left to restore
    assert s.isolate(restore=True).summary["restored"] == 0


def test_isolate_restore_deletes_specs_it_created(tmp_path):
    # /World/a had NO authored visibility — restore must DELETE the spec,
    # not author fresh 'inherited' litter
    s = _session_on(tmp_path, _scene_two_cubes(tmp_path))
    rb = s.refs.ref_for_path("/World/b")
    s.isolate([rb])
    from pxr import Sdf
    layer = s._stage.GetRootLayer()
    assert layer.GetObjectAtPath(Sdf.Path("/World/a.visibility")) is not None
    s.isolate(restore=True)
    assert layer.GetObjectAtPath(Sdf.Path("/World/a.visibility")) is None


def test_save_warns_while_isolation_live(tmp_path):
    path = _scene_two_cubes(tmp_path)
    s = _session_on(tmp_path, path)
    ra = s.refs.ref_for_path("/World/a")
    s.isolate([ra])
    r = s.save()
    assert r.ok
    assert any("isolate --restore" in i.message for i in r.issues), \
        [i.message for i in r.issues]
    s.isolate(restore=True)
    r = s.save()
    assert not any("isolate" in i.message for i in r.issues)


def test_verify_flags_hidden_renderables(tmp_path):
    path = _scene_two_cubes(tmp_path)
    s = _session_on(tmp_path, path)
    ra = s.refs.ref_for_path("/World/a")
    s.isolate([ra])
    s.save()  # bakes the leak, like round 8 did
    r = s.verify(str(path))
    assert r.summary["hidden_renderables"] == 1
    assert "hidden by authored visibility" in r.data["text"]
    assert "isolate --restore" in r.data["text"]
