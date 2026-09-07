# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for material authoring/binding edge cases that don't need the daemon —
notably binding onto instanced geometry, which USD forbids authoring on directly.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pxr")


def _instanced_stage():
    """A stage with two instanceable references to one prototype (so `/World/A/Mesh` and
    `/World/B/Mesh` are instance proxies — not directly editable)."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    UsdGeom.Xform.Define(stage, "/World/Proto")
    UsdGeom.Mesh.Define(stage, "/World/Proto/Mesh")
    for nm in ("A", "B"):
        inst = stage.DefinePrim(f"/World/{nm}")
        inst.GetReferences().AddInternalReference("/World/Proto")
        inst.SetInstanceable(True)
    return stage


def test_bind_to_instance_proxy_redirects_to_editable_root():
    from usd_core import materials

    stage = _instanced_stage()
    assert stage.GetPrimAtPath("/World/A/Mesh").IsInstanceProxy()  # precondition

    mat = materials.create_mdl_material(stage, name="Red", color=(1, 0, 0))
    materials.bind_material(stage, mat, "/World/A/Mesh")  # would raise without the redirect

    a = materials.bound_material(stage, "/World/A/Mesh")
    assert a["bound_material_path"] == mat
    assert a["binding_type"] == "inherited"  # authored on the instance root, inherited by proxy
    # the sibling instance must stay untouched
    assert materials.bound_material(stage, "/World/B/Mesh")["bound_material_path"] is None


def test_unbind_instance_proxy_clears_via_root():
    from usd_core import materials

    stage = _instanced_stage()
    mat = materials.create_preview_material(stage, name="Blue", color=(0, 0, 1))
    materials.bind_material(stage, mat, "/World/A/Mesh")
    materials.unbind_material(stage, "/World/A/Mesh")
    assert materials.bound_material(stage, "/World/A/Mesh")["bound_material_path"] is None


def test_describe_distinguishes_mdl_from_preview():
    from usd_core import materials

    stage = _instanced_stage()
    mdl = materials.create_mdl_material(stage, name="Glass", module="OmniGlass.mdl",
                                        inputs={"glass_ior": 1.5})
    preview = materials.create_preview_material(stage, name="Plain", roughness=0.4)

    d = materials.describe_material(stage, mdl)
    assert d["kind"] == "mdl" and d["mdl_module"] == "OmniGlass.mdl"
    assert d["mdl_subidentifier"] == "OmniGlass" and d["inputs"]["glass_ior"] == 1.5
    assert materials.describe_material(stage, preview)["kind"] == "UsdPreviewSurface"


def test_prototype_bind_authors_once_for_all_instances():
    from usd_core import materials

    stage = _instanced_stage()
    src, root = materials.prototype_source_path(stage, "/World/A/Mesh")
    assert (src, root) == ("/World/Proto/Mesh", "/World/A")

    mat = materials.create_preview_material(stage, name="Copper", color=(0.9, 0.4, 0.1))
    authored = materials.bind_material(stage, mat, "/World/A/Mesh", on_prototype=True)
    assert authored == "/World/Proto/Mesh"  # once, at the prototype source
    for nm in ("A", "B"):  # every instance sharing the prototype picks it up
        assert materials.bound_material(stage, f"/World/{nm}/Mesh")["bound_material_path"] == mat


def test_prototype_source_requires_an_instanced_prim():
    from usd_core import materials

    stage = _instanced_stage()
    with pytest.raises(ValueError, match="not inside a native instance"):
        materials.prototype_source_path(stage, "/World/Proto/Mesh")


def test_subset_bind_authors_family_metadata():
    from pxr import UsdGeom
    from usd_core import materials
    from usd_core.subsets import list_subsets, validate_family

    stage = _instanced_stage()
    mesh = stage.GetPrimAtPath("/World/Proto/Mesh")
    UsdGeom.Mesh(mesh).GetFaceVertexCountsAttr().Set([3, 3, 3])
    UsdGeom.Subset.CreateGeomSubset(UsdGeom.Imageable(mesh), "Pads",
                                    UsdGeom.Tokens.face, [0, 2])

    mat = materials.create_preview_material(stage, name="Gold", color=(0.9, 0.8, 0.2))
    authored = materials.bind_material(stage, mat, "/World/Proto/Mesh/Pads")
    assert authored == "/World/Proto/Mesh/Pads"

    (info,) = list_subsets(stage, "/World/Proto/Mesh")
    assert info["family"] == "materialBind"  # authored automatically on bind
    assert info["bound_material_path"] == mat and not info["problems"]
    fam = validate_family(stage, "/World/Proto/Mesh")
    assert fam["valid"] and fam["faces_covered"] == 2 and fam["faces_uncovered"] == 1
    assert fam["family_type"] == "nonOverlapping"


def test_rich_preview_material_authors_pbr_and_texture_network():
    from pxr import UsdShade
    from usd_core import materials

    stage = _instanced_stage()
    mat = materials.create_preview_material(
        stage, name="Coated", color=(1, 0, 0), clearcoat=0.7, clearcoat_roughness=0.1,
        ior=1.45, emissive=(0, 0.2, 0), opacity=0.9,
        diffuse_texture="albedo.png", normal_texture="normal.png",
        roughness_texture="rough.png", metallic_texture="metal.png",
        uv_set="uv1", tex_scale=(2, 2), tex_rotate=90, tex_translate=(0.5, 0),
        inputs={"opacityThreshold": 0.5})

    d = materials.describe_material(stage, mat)
    ins = d["inputs"]
    assert ins["clearcoat"] == pytest.approx(0.7) and ins["ior"] == pytest.approx(1.45)
    assert ins["emissiveColor"] == pytest.approx([0, 0.2, 0])
    assert ins["opacityThreshold"] == 0.5
    # connected texture inputs report their source node + file
    assert "diffuseTex" in str(ins["diffuseColor"]) and "albedo.png" in str(ins["diffuseColor"])
    assert "normal.png" in str(ins["normal"])

    reader = UsdShade.Shader(stage.GetPrimAtPath(f"{mat}/stReader"))
    assert reader.GetInput("varname").Get() == "uv1"
    xform = UsdShade.Shader(stage.GetPrimAtPath(f"{mat}/uvTransform"))
    assert xform.GetInput("rotation").Get() == 90.0
    normal_tex = UsdShade.Shader(stage.GetPrimAtPath(f"{mat}/normalTex"))
    assert normal_tex.GetInput("sourceColorSpace").Get() == "raw"
    assert list(normal_tex.GetInput("bias").Get()) == [-1, -1, -1, -1]


def test_effective_materials_under_sees_into_instances():
    from usd_core import materials

    stage = _instanced_stage()
    mat = materials.create_preview_material(stage, name="Red", color=(1, 0, 0))
    materials.bind_material(stage, mat, "/World/A/Mesh", on_prototype=True)

    agg = materials.effective_materials_under(stage, "/World")
    assert agg["gprims"] == 3  # proto mesh + two instance proxies
    assert agg["materials"] == {mat: 3} and agg["unbound"] == 0


def test_reference_library_material_binds_from_external_lib(tmp_path):
    from pxr import Usd, UsdGeom, UsdShade
    from usd_core import materials

    lib_file = str(tmp_path / "lib.usda")
    lib = Usd.Stage.CreateNew(lib_file)
    mat = UsdShade.Material.Define(lib, "/Looks/Aluminum_Brushed")
    UsdShade.Shader.Define(lib, "/Looks/Aluminum_Brushed/Shader")
    lib.Save()

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, "/World/Mesh")
    local = materials.reference_library_material(
        stage, library_path=lib_file, material_name="Aluminum_Brushed",
        prim_path="/World/Mesh")
    assert UsdShade.Material(stage.GetPrimAtPath(local))
    bound = materials.bound_material(stage, "/World/Mesh")
    assert bound["bound_material_path"] == local
    assert bound["binding_type"] == "direct"


def test_reference_library_material_normalizes_legacy_terminal_prim(tmp_path):
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    from usd_core import materials

    lib_file = str(tmp_path / "legacy-lib.usda")
    lib = Usd.Stage.CreateNew(lib_file)
    legacy = lib.DefinePrim("/World/Looks/Cardboard", "Scope")
    shader = UsdShade.Shader.Define(lib, "/World/Looks/Cardboard/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    terminal = legacy.CreateAttribute("outputs:surface", Sdf.ValueTypeNames.Token)
    terminal.SetConnections([shader_output.GetAttr().GetPath()])
    lib.Save()

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, "/World/Mesh")
    local = materials.reference_library_material(
        stage,
        library_path=lib_file,
        material_name="Cardboard",
        prim_path="/World/Mesh",
    )

    assert lib.GetPrimAtPath("/World/Looks/Cardboard").GetTypeName() == "Scope"
    imported = UsdShade.Material(stage.GetPrimAtPath(local))
    assert imported
    assert imported.GetSurfaceOutput().GetAttr().HasAuthoredConnections()
    assert materials.bound_material(stage, "/World/Mesh")["bound_material_path"] == local


def test_reference_library_material_rejects_untyped_scope_without_terminals(tmp_path):
    from pxr import Usd
    from usd_core import materials

    lib_file = str(tmp_path / "not-a-material.usda")
    lib = Usd.Stage.CreateNew(lib_file)
    lib.DefinePrim("/World/Looks/Cardboard", "Scope")
    lib.Save()

    stage = Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError, match="authored material terminal"):
        materials.import_library_material(
            stage,
            library_path=lib_file,
            material_name="Cardboard",
        )


def test_reference_library_material_uses_exact_source_prim_path(tmp_path):
    from pxr import Usd, UsdGeom, UsdShade
    from usd_core import materials

    lib_file = str(tmp_path / "exact-lib.usda")
    lib = Usd.Stage.CreateNew(lib_file)
    UsdShade.Material.Define(lib, "/World/Looks/Red")
    UsdShade.Material.Define(lib, "/Other/Looks/Paint_Red")
    lib.Save()

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, "/World/Mesh")
    local = materials.reference_library_material(
        stage,
        library_path=lib_file,
        material_name="Paint Red",
        source_prim_path="/World/Looks/Red",
        prim_path="/World/Mesh",
    )

    assert local.startswith("/Looks/Red_")
    assert materials.bound_material(stage, "/World/Mesh")["bound_material_path"] == local


def test_exact_library_material_paths_do_not_alias_same_leaf_or_scene_material(
    tmp_path,
):
    from pxr import Usd, UsdShade
    from usd_core import materials

    lib_file = str(tmp_path / "same-leaf-lib.usda")
    lib = Usd.Stage.CreateNew(lib_file)
    UsdShade.Material.Define(lib, "/A/Looks/Metal")
    UsdShade.Material.Define(lib, "/B/Looks/Metal")
    lib.Save()

    stage = Usd.Stage.CreateInMemory()
    scene_material = UsdShade.Material.Define(stage, "/Looks/Metal").GetPath().pathString
    first = materials.import_library_material(
        stage,
        library_path=lib_file,
        material_name="Metal A",
        source_prim_path="/A/Looks/Metal",
    )
    second = materials.import_library_material(
        stage,
        library_path=lib_file,
        material_name="Metal B",
        source_prim_path="/B/Looks/Metal",
    )

    assert len({scene_material, first, second}) == 3
    assert first.startswith("/Looks/Metal_")
    assert second.startswith("/Looks/Metal_")
    assert materials.import_library_material(
        stage,
        library_path=lib_file,
        material_name="ignored display label",
        source_prim_path="/A/Looks/Metal",
    ) == first


def test_exact_library_material_anchors_reference_to_active_session_layer(tmp_path):
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    from usd_core import materials

    lib_file = str(tmp_path / "library.usda")
    lib = Usd.Stage.CreateNew(lib_file)
    UsdShade.Material.Define(lib, "/World/Looks/Paint")
    lib.Save()

    scene_file = str(tmp_path / "scene.usda")
    scene = Usd.Stage.CreateNew(scene_file)
    world = UsdGeom.Xform.Define(scene, "/World").GetPrim()
    scene.SetDefaultPrim(world)
    UsdGeom.Mesh.Define(scene, "/World/Mesh")
    scene.Save()

    root_layer = Sdf.Layer.FindOrOpen(scene_file)
    session_layer = Sdf.Layer.CreateAnonymous("material-session.usda")
    stage = Usd.Stage.Open(root_layer, session_layer)
    stage.SetEditTarget(session_layer)

    local = materials.import_library_material(
        stage,
        library_path=lib_file,
        material_name="Paint",
        source_prim_path="/World/Looks/Paint",
    )

    imported = stage.GetPrimAtPath(local)
    assert UsdShade.Material(imported)
    assert imported.GetCustomDataByKey("usdCliLibraryAsset") == lib_file


def test_exact_library_material_reuses_canonical_identity_across_edit_targets(
    tmp_path,
):
    import os

    from pxr import Sdf, Usd, UsdGeom, UsdShade
    from usd_core import materials

    library_path = tmp_path / "materials" / "library.usda"
    library_path.parent.mkdir()
    library = Usd.Stage.CreateNew(str(library_path))
    UsdShade.Material.Define(library, "/World/Looks/Paint")
    library.Save()

    scene_path = tmp_path / "scene.usda"
    scene = Usd.Stage.CreateNew(str(scene_path))
    world = UsdGeom.Xform.Define(scene, "/World").GetPrim()
    scene.SetDefaultPrim(world)
    first = materials.import_library_material(
        scene,
        library_path=str(library_path),
        material_name="Paint",
        source_prim_path="/World/Looks/Paint",
    )
    assert (
        scene.GetPrimAtPath(first).GetCustomDataByKey("usdCliLibraryAsset")
        == os.path.realpath(library_path)
    )
    scene.Save()

    root_layer = Sdf.Layer.FindOrOpen(str(scene_path))
    session_layer = Sdf.Layer.CreateAnonymous("material-session.usda")
    reopened = Usd.Stage.Open(root_layer, session_layer)
    reopened.SetEditTarget(session_layer)
    second = materials.import_library_material(
        reopened,
        library_path=str(library_path),
        material_name="Paint",
        source_prim_path="/World/Looks/Paint",
    )

    assert second == first
    imported = reopened.GetPrimAtPath(second)
    assert (
        imported.GetCustomDataByKey("usdCliLibraryAsset")
        == os.path.realpath(library_path)
    )


def test_library_material_lookup_by_display_name_with_spaces(tmp_path, monkeypatch):
    """A display name with spaces ("Stainless Steel") resolves to the underscored prim, and
    the raw spaced name is NEVER handed to GetPrimAtPath (that probe emitted dozens of
    "Ill-formed SdfPath" warnings per bind). We assert the invalid path is filtered out."""
    from pxr import Usd, UsdShade
    from usd_core import materials

    lib = Usd.Stage.CreateNew(str(tmp_path / "lib.usda"))
    UsdShade.Material.Define(lib, "/World/Looks/Stainless_Steel")
    lib.Save()

    probed = []
    real = Usd.Stage.GetPrimAtPath
    monkeypatch.setattr(Usd.Stage, "GetPrimAtPath",
                        lambda self, p: probed.append(str(p)) or real(self, p))

    got = materials._find_library_material(lib, "Stainless Steel")
    assert got == "/World/Looks/Stainless_Steel"  # resolved via the normalized fallback
    # no invalid (space-containing) path string was ever probed
    assert all(materials._is_valid_sdf_path_string(p) for p in probed), probed


def test_reference_library_material_relative_path_resolves_from_saved_stage(tmp_path, monkeypatch):
    """A CWD-relative --library path must still resolve when the stage's root layer lives
    in a different directory (the reference is authored absolute, not layer-relative)."""
    from pxr import Usd, UsdGeom, UsdShade
    from usd_core import materials

    (tmp_path / "shared").mkdir()
    (tmp_path / "task" / "assets").mkdir(parents=True)
    lib = Usd.Stage.CreateNew(str(tmp_path / "shared" / "lib.usda"))
    UsdShade.Material.Define(lib, "/Looks/Steel")
    lib.Save()
    scene_file = str(tmp_path / "task" / "assets" / "scene.usda")
    scene = Usd.Stage.CreateNew(scene_file)
    UsdGeom.Mesh.Define(scene, "/World/Mesh")
    scene.Save()

    monkeypatch.chdir(tmp_path / "task")  # caller-relative ≠ layer-relative
    stage = Usd.Stage.Open(scene_file)
    local = materials.reference_library_material(
        stage, library_path="../shared/lib.usda", material_name="Steel",
        prim_path="/World/Mesh")
    # composes as a real Material only if the authored reference actually resolves
    assert UsdShade.Material(stage.GetPrimAtPath(local))


def test_reference_library_material_display_name_and_missing_name(tmp_path):
    from pxr import Usd, UsdGeom, UsdShade
    import pytest as _pytest
    from usd_core import materials

    lib_file = str(tmp_path / "lib.usda")
    lib = Usd.Stage.CreateNew(lib_file)
    UsdShade.Material.Define(lib, "/Looks/Aluminum_Brushed")
    lib.Save()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, "/World/Mesh")

    # display name with a space normalizes to the prim name (no leaked pxr error)
    local = materials.reference_library_material(
        stage, library_path=lib_file, material_name="Aluminum Brushed",
        prim_path="/World/Mesh")
    assert UsdShade.Material(stage.GetPrimAtPath(local))

    # unknown name: clean error listing what exists, no arbitrary fallback
    with _pytest.raises(ValueError, match="Aluminum_Brushed"):
        materials.reference_library_material(
            stage, library_path=lib_file, material_name="Chrome_Polished",
            prim_path="/World/Mesh")


def test_selector_rules():
    from pxr import UsdGeom
    from usd_core import selector

    stage = _instanced_stage()
    UsdGeom.Mesh.Define(stage, "/World/Conductor_01")
    UsdGeom.Mesh.Define(stage, "/World/Conductor_02")
    UsdGeom.Sphere.Define(stage, "/World/Ball")

    paths = lambda prims: {p.GetPath().pathString for p in prims}  # noqa: E731

    hits = selector.select_prims(stage, where=["name~=Conductor*"])
    assert paths(hits) == {"/World/Conductor_01", "/World/Conductor_02"}

    hits = selector.select_prims(stage, types=["Sphere"])
    assert paths(hits) == {"/World/Ball"}

    # gprims only by default, instance proxies included, scoped by --under
    hits = selector.select_prims(stage, under="/World/A")
    assert paths(hits) == {"/World/A/Mesh"}

    hits = selector.select_prims(stage, where=["name!=Ball", "type==Mesh"])
    assert "/World/Ball" not in paths(hits) and "/World/Conductor_01" in paths(hits)

    with pytest.raises(ValueError, match="bad --where expression"):
        selector.select_prims(stage, where=["name"])
    with pytest.raises(ValueError, match="unknown --where key"):
        selector.select_prims(stage, where=["nope==1"])

    # Long whitespace/value tails stay a single linear capture rather than
    # backtracking across overlapping repetitions.
    long_value = "x" * 100_000
    assert selector.parse_rule(f"  name ~=  {long_value}  ") == (
        "name",
        "~=",
        long_value,
    )


def test_bulk_where_skips_instance_internal_matches():
    """Regression (task-03 benchmark): a --where name glob matching prims INSIDE an
    instance redirected the bind onto every instance root sharing the prototype."""
    from usd_core import selector

    stage = _instanced_stage()  # /World/A, /World/B instances of /World/Proto (Mesh inside)
    # 'mesh' matches the proxy /World/A/Mesh but NOT the instance roots A/B themselves
    prims = selector.select_prims(stage, where=["name~=Mesh*"])
    proxies = [p for p in prims if p.IsInstanceProxy()]
    assert proxies, "precondition: the glob matches instance internals"
    for p in proxies:
        root = selector.instance_root_of(p)
        assert not selector.matches_rules(stage, root, ["name~=Mesh*"])
