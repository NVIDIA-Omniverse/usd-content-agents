# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage tests for USD prim utility helpers."""

import itertools

import pytest

pxr = pytest.importorskip("pxr")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdSkel, Vt  # noqa: E402

import world_understanding.utils.usd.prim as prim_module  # noqa: E402
from world_understanding.utils.usd.prim import (  # noqa: E402
    _copy_layer_metadata,
    _copy_prim_spec_recursive,
    _copy_property_spec,
    _is_valid_bbox_range,
    assign_color_to_meshes,
    assign_random_colors_to_meshes,
    collect_mesh_geometry_stats,
    convert_abstract_prototypes_to_def,
    disable_visibility_except_for_selected_mesh_prim,
    disable_visibility_except_for_selected_mesh_prims,
    disable_visibility_for_all_gprims,
    disable_visibility_for_all_mesh_prims,
    enable_visibility_except_for_selected_mesh_prims,
    enable_visibility_for_all_mesh_prims,
    flatten_prototype_references,
    get_all_mesh_prim_paths,
    get_bbox_from_prim,
    get_subtree_geometry_stats,
    nullify_material,
    nullify_materials,
    print_prim_hierarchy,
    remove_all_lights,
    remove_scope_and_prims_under_it,
    set_gprim_display_color,
    set_mesh_display_color,
    traverse_prims,
)


def _make_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    mesh_a = UsdGeom.Mesh.Define(stage, "/World/MeshA")
    mesh_a.GetPointsAttr().Set(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (0, 1, 0)]))
    mesh_a.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]))
    mesh_a.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2]))
    UsdGeom.Mesh.Define(stage, "/World/MeshB")
    stage.DefinePrim("/World/Typeless")
    return stage


def test_traversal_mesh_paths_subtree_stats_and_hierarchy(capsys):
    stage = _make_stage()

    for method in ("traverse", "traverse_all", "traverse_instanced_proxies"):
        paths = [str(prim.GetPath()) for prim in traverse_prims(stage, method)]
        assert "/World" in paths
        assert "/" not in paths

    assert [str(path) for path in get_all_mesh_prim_paths(stage)] == [
        "/World/MeshA",
        "/World/MeshB",
    ]

    stats = collect_mesh_geometry_stats(stage, top_n=1)
    assert stats["top_meshes_by_vertices"][0]["path"] == "/World/MeshA"

    subtree = get_subtree_geometry_stats(stage, "/World")
    assert subtree["mesh_count"] == 2
    assert subtree["vertex_count"] == 3
    assert subtree["face_count"] == 1
    assert subtree["prim_type_breakdown"]["<no type>"] == 1

    skipped = get_subtree_geometry_stats(stage, "/World", skip_geometry=True)
    assert skipped["mesh_count"] == 2
    assert skipped["vertex_count"] == 0

    missing = get_subtree_geometry_stats(stage, "/Missing")
    assert missing == {
        "mesh_count": 0,
        "vertex_count": 0,
        "face_count": 0,
        "prim_type_breakdown": {},
    }

    print_prim_hierarchy(stage)
    output = capsys.readouterr().out
    assert "World" in output
    assert "MeshA" in output


def test_nullify_material_real_prim_and_targeted_fake_modes(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    material = UsdShade.Material.Define(stage, "/World/Looks/Blue")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh_prim = mesh.GetPrim()
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World")).Bind(material)
    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(material)
    mesh_prim.CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
    ).Set(Vt.Vec3fArray([Gf.Vec3f(1, 0, 0)]))
    mesh_prim.CreateAttribute(
        "primvars:displayColor:indices", Sdf.ValueTypeNames.IntArray
    ).Set(Vt.IntArray([0]))
    mesh_prim.CreateAttribute(
        "primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray
    ).Set(Vt.FloatArray([1.0]))

    nullify_material(mesh_prim, set_triangle_winding_order=1)

    assert (
        UsdShade.MaterialBindingAPI(mesh_prim).GetDirectBindingRel().GetTargets() == []
    )
    assert (
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/World"))
        .GetDirectBindingRel()
        .GetTargets()
        == []
    )
    assert mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.rightHanded

    other = UsdGeom.Mesh.Define(stage, "/World/Excluded")
    nullify_material(
        other.GetPrim(),
        set_triangle_winding_order=2,
        exclude_list=["Excluded"],
        clear_ancestor_bindings=False,
    )
    assert other.GetOrientationAttr().Get() != UsdGeom.Tokens.leftHanded

    left = UsdGeom.Mesh.Define(stage, "/World/Left")
    nullify_material(left.GetPrim(), set_triangle_winding_order=2)
    assert left.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded

    calls = []

    class _FakePrim:
        def __init__(self, name, *, proxy=False, instance=False):
            self.name = name
            self.proxy = proxy
            self.instance = instance
            self.instanceable = True

        def IsInstanceProxy(self):
            return self.proxy

        def IsInstance(self):
            return self.instance

        def SetInstanceable(self, value):
            self.instanceable = value

    fake_proxy = _FakePrim("proxy", proxy=True)
    fake_instance = _FakePrim("instance", instance=True)
    fake_plain = _FakePrim("plain")

    class _FakeStage:
        def GetPrimAtPath(self, path):
            return {
                "/proxy": fake_proxy,
                "/instance": fake_instance,
                "/plain": fake_plain,
            }[path]

    monkeypatch.setattr(
        prim_module,
        "nullify_material",
        lambda prim, winding, exclude: calls.append((prim.name, winding, exclude)),
    )

    nullify_materials(
        _FakeStage(),
        prim_paths=["/proxy", "/instance", "/plain"],
        set_triangle_winding_order=2,
        exclude_list=["skip"],
    )

    assert calls == [("instance", 2, ["skip"]), ("plain", 2, ["skip"])]
    assert fake_instance.instanceable is False

    calls.clear()
    monkeypatch.setattr(
        prim_module,
        "traverse_prims",
        lambda stage, traversal_method: iter([fake_plain]),
    )
    nullify_materials(_FakeStage())
    assert calls == [("plain", 0, [])]


class _FakeAttr:
    def __init__(self):
        self.values = []

    def Set(self, value, time=None):
        self.values.append((value, time))


class _FakeMesh:
    def __init__(self, prim):
        self.prim = prim
        self.display_color = _FakeAttr()
        self.visibility = _FakeAttr()

    def GetDisplayColorAttr(self):
        return self.display_color

    def GetVisibilityAttr(self):
        return self.visibility


class _FakePrim:
    def __init__(self, path, *, mesh=True, proxy=False, instance=False):
        self.path = path
        self.mesh = mesh
        self.proxy = proxy
        self.instance = instance
        self.instanceable = True

    def IsA(self, _type):
        return self.mesh

    def IsInstanceProxy(self):
        return self.proxy

    def IsInstance(self):
        return self.instance

    def SetInstanceable(self, value):
        self.instanceable = value

    def GetPath(self):
        return Sdf.Path(self.path)


def test_color_and_visibility_helpers_with_fake_meshes(monkeypatch):
    prims = [
        _FakePrim("/Mesh"),
        _FakePrim("/Proxy", proxy=True),
        _FakePrim("/Instance", instance=True),
        _FakePrim("/NonMesh", mesh=False),
    ]
    fake_meshes = {}

    def fake_mesh_factory(prim):
        mesh = fake_meshes.setdefault(prim.path, _FakeMesh(prim))
        return mesh

    monkeypatch.setattr(
        prim_module, "traverse_prims", lambda *args, **kwargs: iter(prims)
    )
    monkeypatch.setattr(prim_module.UsdGeom, "Mesh", fake_mesh_factory)

    assign_color_to_meshes(object(), (0.1, 0.2, 0.3))
    assert fake_meshes["/Mesh"].display_color.values
    assert prims[2].instanceable is False
    assert "/Proxy" not in fake_meshes

    colors = itertools.cycle([0.4, 0.5, 0.6])
    monkeypatch.setattr(prim_module.random, "uniform", lambda low, high: next(colors))
    assign_random_colors_to_meshes(object(), range_min=0.4, range_max=0.6)
    random_color = fake_meshes["/Mesh"].display_color.values[-1][0]
    assert tuple(random_color[0]) == pytest.approx((0.4, 0.5, 0.6))

    enable_visibility_for_all_mesh_prims(object())
    assert fake_meshes["/Mesh"].visibility.values[-1][0] == UsdGeom.Tokens.inherited

    enable_visibility_except_for_selected_mesh_prims(object(), ["/Mesh"])
    assert fake_meshes["/Instance"].visibility.values[-1][0] == UsdGeom.Tokens.inherited

    disable_visibility_for_all_mesh_prims(object())
    assert fake_meshes["/Mesh"].visibility.values[-1][0] == UsdGeom.Tokens.invisible

    disable_visibility_except_for_selected_mesh_prim(object(), "/Mesh")
    assert fake_meshes["/Instance"].visibility.values[-1][0] == UsdGeom.Tokens.invisible

    disable_visibility_except_for_selected_mesh_prims(object(), ["/Mesh"])
    assert fake_meshes["/Instance"].visibility.values[-1][0] == UsdGeom.Tokens.invisible


def test_set_display_color_remove_scope_and_light_fallback(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    set_mesh_display_color(mesh, (0.2, 0.3, 0.4))
    assert tuple(mesh.GetDisplayColorAttr().Get()[0]) == pytest.approx((0.2, 0.3, 0.4))

    UsdGeom.Xform.Define(stage, "/World/Scope")
    UsdGeom.Mesh.Define(stage, "/World/Scope/Child")
    remove_scope_and_prims_under_it(stage, "/World/Scope")
    assert not stage.GetPrimAtPath("/World/Scope").IsValid()
    remove_scope_and_prims_under_it(stage, "/World/Missing")

    class _LightPrim:
        def __init__(self):
            self.active = True

        def HasAPI(self, _api):
            return True

        def IsInstanceProxy(self):
            return False

        def GetPath(self):
            return Sdf.Path("/Light")

        def IsActive(self):
            return self.active

        def SetActive(self, value):
            self.active = value

    light = _LightPrim()

    class _Stage:
        def GetPrimAtPath(self, path):
            return light

        def RemovePrim(self, path):
            return False

    monkeypatch.setattr(prim_module, "traverse_prims", lambda stage: iter([light]))
    remove_all_lights(_Stage())
    assert light.active is False


def test_gprim_color_and_visibility_cover_native_geometry() -> None:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    sphere = UsdGeom.Sphere.Define(stage, "/World/Sphere")

    cube_gprim = UsdGeom.Gprim(cube.GetPrim())
    set_gprim_display_color(cube_gprim, (0.1, 0.2, 0.3))
    assert tuple(cube_gprim.GetDisplayColorAttr().Get()[0]) == pytest.approx(
        (0.1, 0.2, 0.3)
    )

    time = Usd.TimeCode(3)
    disable_visibility_for_all_gprims(stage, time=time)
    for schema in (mesh, cube, sphere):
        imageable = UsdGeom.Imageable(schema.GetPrim())
        assert imageable.GetVisibilityAttr().Get(time) == UsdGeom.Tokens.invisible

    world_visibility = UsdGeom.Imageable(world.GetPrim()).GetVisibilityAttr()
    assert world_visibility.GetTimeSamples() == []


def test_gprim_visibility_masks_instance_proxies_at_editable_root() -> None:
    source_layer = Sdf.Layer.CreateAnonymous("instance-source.usda")
    source_stage = Usd.Stage.Open(source_layer)
    prototype = UsdGeom.Xform.Define(source_stage, "/Prototype").GetPrim()
    UsdGeom.Cube.Define(source_stage, "/Prototype/Cube")
    source_stage.SetDefaultPrim(prototype)

    stage = Usd.Stage.CreateInMemory()
    instance = UsdGeom.Xform.Define(stage, "/Instance").GetPrim()
    instance.GetReferences().AddReference(source_layer.identifier, "/Prototype")
    instance.SetInstanceable(True)
    proxy = stage.GetPrimAtPath("/Instance/Cube")
    assert proxy.IsInstanceProxy()

    time = Usd.TimeCode(2)
    disable_visibility_for_all_gprims(stage, time=time)

    assert instance.IsInstance()
    assert (
        UsdGeom.Imageable(instance).GetVisibilityAttr().Get(time)
        == UsdGeom.Tokens.invisible
    )
    assert UsdGeom.Imageable(proxy).ComputeVisibility(time) == UsdGeom.Tokens.invisible


def test_gprim_visibility_deinstances_non_imageable_instance_root() -> None:
    source_layer = Sdf.Layer.CreateAnonymous("instance-source.usda")
    source_stage = Usd.Stage.Open(source_layer)
    prototype = UsdGeom.Xform.Define(source_stage, "/Prototype").GetPrim()
    UsdGeom.Cube.Define(source_stage, "/Prototype/Cube")
    source_stage.SetDefaultPrim(prototype)

    stage = Usd.Stage.CreateInMemory()
    instance = UsdShade.Material.Define(stage, "/Instance").GetPrim()
    instance.GetReferences().AddReference(source_layer.identifier, "/Prototype")
    instance.SetInstanceable(True)
    proxy_path = "/Instance/Cube"
    assert stage.GetPrimAtPath(proxy_path).IsInstanceProxy()
    assert not UsdGeom.Imageable(instance)

    time = Usd.TimeCode(2)
    disable_visibility_for_all_gprims(stage, time=time)

    resolved = stage.GetPrimAtPath(proxy_path)
    assert not instance.IsInstance()
    assert not resolved.IsInstanceProxy()
    assert (
        UsdGeom.Imageable(resolved).ComputeVisibility(time) == UsdGeom.Tokens.invisible
    )


def test_gprim_visibility_deinstances_gprim_instance_root() -> None:
    source_layer = Sdf.Layer.CreateAnonymous("gprim-instance-source.usda")
    source_stage = Usd.Stage.Open(source_layer)
    prototype = UsdGeom.Cube.Define(source_stage, "/Prototype").GetPrim()
    source_stage.SetDefaultPrim(prototype)

    stage = Usd.Stage.CreateInMemory()
    instance = UsdGeom.Cube.Define(stage, "/Instance").GetPrim()
    instance.GetReferences().AddReference(source_layer.identifier, "/Prototype")
    instance.SetInstanceable(True)
    assert instance.IsInstance()

    time = Usd.TimeCode(2)
    disable_visibility_for_all_gprims(stage, time=time)

    assert not instance.IsInstance()
    assert (
        UsdGeom.Imageable(instance).ComputeVisibility(time) == UsdGeom.Tokens.invisible
    )


def test_gprim_visibility_skips_stale_collected_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CollectedGprim:
        def IsA(self, schema: object) -> bool:
            return schema is UsdGeom.Gprim

        def GetPath(self) -> str:
            return "/Stale"

    class _Stage:
        def GetPrimAtPath(self, path: str) -> None:
            assert path == "/Stale"
            return None

    monkeypatch.setattr(
        prim_module,
        "traverse_prims",
        lambda stage, traversal_method: iter([_CollectedGprim()]),
    )

    disable_visibility_for_all_gprims(_Stage())  # type: ignore[arg-type]


def test_bbox_validation_and_direct_valid_bounds():
    class _InvertedRange:
        def IsEmpty(self):
            return False

        def GetMin(self):
            return (2.0, 0.0, 0.0)

        def GetMax(self):
            return (1.0, 1.0, 1.0)

    assert not _is_valid_bbox_range(
        Gf.Range3d(Gf.Vec3d(float("nan"), 0, 0), Gf.Vec3d(1, 1, 1))
    )
    assert not _is_valid_bbox_range(_InvertedRange())

    stage = Usd.Stage.CreateInMemory()
    cube = UsdGeom.Cube.Define(stage, "/Cube")
    bbox = get_bbox_from_prim(cube.GetPrim()).ComputeAlignedRange()
    assert not bbox.IsEmpty()

    render_cube = UsdGeom.Cube.Define(stage, "/RenderCube")
    render_cube.CreatePurposeAttr(UsdGeom.Tokens.render)
    render_cube.AddTranslateOp().Set((100.0, 0.0, 0.0))
    default_render_bbox = get_bbox_from_prim(
        render_cube.GetPrim()
    ).ComputeAlignedRange()
    assert tuple(default_render_bbox.GetMin()) == pytest.approx((0.0, 0.0, 0.0))
    assert tuple(default_render_bbox.GetMax()) == pytest.approx((0.0, 0.0, 0.0))
    render_bbox = get_bbox_from_prim(
        render_cube.GetPrim(),
        included_purposes=(UsdGeom.Tokens.default_, UsdGeom.Tokens.render),
    ).ComputeAlignedRange()
    assert not render_bbox.IsEmpty()
    assert render_bbox.GetMax()[0] == pytest.approx(101.0)

    root = UsdGeom.Xform.Define(stage, "/Root")
    UsdSkel.Root.Define(stage, "/Root/SkelRoot")
    mesh_a = UsdGeom.Mesh.Define(stage, "/Root/SkelRoot/A")
    mesh_a.GetPointsAttr().Set(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (0, 1, 0)]))
    mesh_a.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]))
    mesh_a.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2]))
    mesh_b = UsdGeom.Mesh.Define(stage, "/Root/SkelRoot/B")
    mesh_b.GetPointsAttr().Set(Vt.Vec3fArray([(2, 2, 0), (3, 2, 0), (2, 3, 0)]))
    mesh_b.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]))
    mesh_b.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2]))

    union = get_bbox_from_prim(root.GetPrim()).ComputeAlignedRange()
    assert tuple(union.GetMax()) == pytest.approx((3, 3, 0))


def test_convert_abstract_prototypes_to_def_variants():
    layer = Sdf.Layer.CreateAnonymous("prototypes.usda")
    prototype = Sdf.PrimSpec(layer, "Flattened_Prototype_1", Sdf.SpecifierClass)
    child = Sdf.PrimSpec(prototype, "ChildPrototype", Sdf.SpecifierOver)
    child.typeName = "Scope"
    other = Sdf.PrimSpec(layer, "Other", Sdf.SpecifierOver)
    other.typeName = "Xform"
    stage = Usd.Stage.Open(layer)

    assert convert_abstract_prototypes_to_def(stage) == 2
    assert prototype.specifier == Sdf.SpecifierDef
    assert prototype.typeName == "Xform"
    assert child.specifier == Sdf.SpecifierDef
    assert child.typeName == "Scope"
    assert other.specifier == Sdf.SpecifierOver

    assert convert_abstract_prototypes_to_def(stage, prototype_names=["Other"]) == 1
    assert other.specifier == Sdf.SpecifierDef
    assert convert_abstract_prototypes_to_def(stage, prototype_names=["Missing"]) == 0

    class _BadLayer:
        @property
        def rootPrims(self):
            raise RuntimeError("bad layer")

    class _BadStage:
        def GetRootLayer(self):
            return _BadLayer()

    with pytest.raises(RuntimeError, match="bad layer"):
        convert_abstract_prototypes_to_def(_BadStage())


def test_flatten_prototype_references_and_property_copy_helpers():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().documentation = "doc"
    stage.GetRootLayer().pseudoRoot.customData["rootData"] = "copied"
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    mesh = UsdGeom.Mesh.Define(stage, "/World/Nested/Mesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (0, 1, 0)]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]))
    mesh.GetPrim().SetMetadata("kind", "component")
    mesh.GetPrim().SetCustomData({"validKey": "value"})
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    attr = mesh.GetPrim().CreateAttribute("inputs:surface", Sdf.ValueTypeNames.Token)
    attr.AddConnection(Sdf.Path("/World/Shader.outputs:surface"))
    attr.AddConnection(Sdf.Path("/Flattened_Prototype_1/Shader.outputs:surface"))
    rel = mesh.GetPrim().CreateRelationship("targets")
    rel.AddTarget(Sdf.Path("/World/Shader"))
    rel.AddTarget(Sdf.Path("/Prototypes/Proto"))
    UsdGeom.Mesh.Define(stage, "/Flattened_Prototype_1/Mesh")
    UsdGeom.Mesh.Define(stage, "/Prototypes/Proto/Mesh")

    flattened = flatten_prototype_references(stage)
    assert flattened.defaultPrim == "World"
    assert flattened.documentation == "doc"
    assert flattened.pseudoRoot.customData["rootData"] == "copied"
    assert flattened.GetPrimAtPath("/World/Nested/Mesh")
    assert not flattened.GetPrimAtPath("/Flattened_Prototype_1")
    assert not flattened.GetPrimAtPath("/Prototypes")
    flattened_stage = Usd.Stage.Open(flattened)
    flattened_mesh = flattened_stage.GetPrimAtPath("/World/Nested/Mesh")
    assert flattened_mesh.GetAttribute("inputs:surface").GetConnections() == [
        Sdf.Path("/World/Shader.outputs:surface")
    ]
    assert flattened_mesh.GetRelationship("targets").GetTargets() == [
        Sdf.Path("/World/Shader")
    ]

    kept = flatten_prototype_references(stage, remove_prototypes=False)
    assert kept.GetPrimAtPath("/Flattened_Prototype_1/Mesh")

    source_layer = Sdf.Layer.CreateAnonymous("source.usda")
    source = Sdf.PrimSpec(source_layer, "PrototypeThing", Sdf.SpecifierClass)
    source.typeName = "Scope"
    source.SetInfo("documentation", "source-doc")
    child_source = Sdf.PrimSpec(source, "Child", Sdf.SpecifierDef)
    child_source.typeName = "Xform"
    attr_spec = Sdf.AttributeSpec(
        source, "size", Sdf.ValueTypeNames.Float, Sdf.VariabilityVarying
    )
    attr_spec.default = 1.5
    source_layer.SetTimeSample(attr_spec.path, 1.0, 2.5)
    rel_spec = Sdf.RelationshipSpec(source, "target", False)
    rel_spec.targetPathList.Append(Sdf.Path("/World/Nested/Mesh"))

    target_layer = Sdf.Layer.CreateAnonymous("target.usda")
    assert _copy_prim_spec_recursive(source, target_layer, None, None) == 1
    target = target_layer.GetPrimAtPath("/PrototypeThing")
    assert target.typeName == "Scope"
    assert target.GetInfo("documentation") == "source-doc"
    assert target.attributes["size"].default == 1.5
    assert target_layer.QueryTimeSample(target.attributes["size"].path, 1.0) == 2.5
    assert list(
        target.relationships["target"].targetPathList.GetAddedOrExplicitItems()
    ) == [Sdf.Path("/World/Nested/Mesh")]
    assert target.nameChildren["Child"].typeName == "Xform"

    child_target_layer = Sdf.Layer.CreateAnonymous("child-target.usda")
    parent = Sdf.PrimSpec(child_target_layer, "Parent", Sdf.SpecifierDef)
    _copy_property_spec(attr_spec, parent)
    _copy_property_spec(rel_spec, parent)
    assert "size" in parent.attributes
    assert "target" in parent.relationships

    no_type_layer = Sdf.Layer.CreateAnonymous("no-type.usda")
    no_type = Sdf.PrimSpec(no_type_layer, "Specific", Sdf.SpecifierOver)
    target_no_type = Sdf.Layer.CreateAnonymous("target-no-type.usda")
    assert _copy_prim_spec_recursive(no_type, target_no_type, None, ["Specific"]) == 1
    assert target_no_type.GetPrimAtPath("/Specific").typeName == "Xform"

    metadata_source = Sdf.Layer.CreateAnonymous("metadata-source.usda")
    metadata_source.pseudoRoot.SetInfo("documentation", "pseudo-doc")
    metadata_target = Sdf.Layer.CreateAnonymous("metadata-target.usda")
    _copy_layer_metadata(metadata_source, metadata_target)
    assert metadata_target.pseudoRoot.GetInfo("documentation") == "pseudo-doc"


@pytest.mark.parametrize("indexed", [False, True], ids=["unindexed", "indexed"])
def test_flatten_prototype_references_preserves_face_varying_primvars(indexed):
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    points = Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    face_counts = Vt.IntArray([3, 3])
    face_indices = Vt.IntArray([0, 1, 2, 0, 2, 3])
    flattened_uvs = Vt.Vec2fArray([(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)])
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set(face_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_indices)

    primvars = UsdGeom.PrimvarsAPI(mesh)
    st = primvars.CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.SetElementSize(1)
    st.GetAttr().SetDocumentation("authored UV set")
    if indexed:
        st.Set(Vt.Vec2fArray([(0, 0), (1, 0), (1, 1), (0, 1)]))
        st.SetIndices(face_indices)
    else:
        st.Set(flattened_uvs)
        st.BlockIndices()

    weights = primvars.CreatePrimvar(
        "weights", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
    )
    weights.Set(Vt.FloatArray([0.1, 0.2, 0.3, 0.4]))
    marker = mesh.GetPrim().CreateAttribute(
        "uniformMarker",
        Sdf.ValueTypeNames.Float,
        custom=True,
        variability=Sdf.VariabilityUniform,
    )
    marker.Set(1.0)

    flattened_stage = Usd.Stage.Open(flatten_prototype_references(stage))
    flattened_mesh = UsdGeom.Mesh(flattened_stage.GetPrimAtPath("/World/Mesh"))
    flattened_st = UsdGeom.PrimvarsAPI(flattened_mesh).GetPrimvar("st")
    flattened_weights = UsdGeom.PrimvarsAPI(flattened_mesh).GetPrimvar("weights")
    flattened_marker = flattened_mesh.GetPrim().GetAttribute("uniformMarker")

    assert flattened_mesh.GetPointsAttr().Get() == points
    assert flattened_mesh.GetFaceVertexCountsAttr().Get() == face_counts
    assert flattened_mesh.GetFaceVertexIndicesAttr().Get() == face_indices
    assert flattened_st.GetInterpolation() == UsdGeom.Tokens.faceVarying
    assert flattened_st.HasAuthoredElementSize()
    assert flattened_st.GetElementSize() == 1
    assert flattened_st.GetAttr().GetDocumentation() == "authored UV set"
    assert flattened_st.ComputeFlattened() == flattened_uvs
    assert len(flattened_st.ComputeFlattened()) == sum(face_counts)
    assert flattened_weights.GetInterpolation() == UsdGeom.Tokens.vertex
    assert flattened_weights.GetAttr().Get() == Vt.FloatArray([0.1, 0.2, 0.3, 0.4])
    assert flattened_marker.IsCustom()
    assert flattened_marker.GetVariability() == Sdf.VariabilityUniform
    assert flattened_marker.Get() == 1.0

    if indexed:
        assert flattened_st.IsIndexed()
        assert flattened_st.GetIndices() == face_indices
    else:
        assert not flattened_st.IsIndexed()
        assert flattened_st.GetIndicesAttr().GetResolveInfo().ValueIsBlocked()


def test_flatten_with_fake_stage_for_parent_chain_and_invalid_custom_data(monkeypatch):
    source_layer = Sdf.Layer.CreateAnonymous("fake-source.usda")

    class _FakeComposedPrim:
        def __init__(self, path):
            self._path = Sdf.Path(path)

        def GetPath(self):
            return self._path

        def GetTypeName(self):
            return "Mesh"

        def GetMetadata(self, key):
            return None

        def GetCustomData(self):
            return {"3dsmax": "skip", "validKey": "copy"}

        def GetAttributes(self):
            return []

        def GetRelationships(self):
            return []

    fake_prim = _FakeComposedPrim("/Root/Child/Mesh")

    class _FakeStage:
        def GetRootLayer(self):
            return source_layer

        def Traverse(self, predicate):
            return iter([fake_prim, fake_prim])

    monkeypatch.setattr(prim_module.UsdGeom, "GetStageUpAxis", lambda stage: None)
    monkeypatch.setattr(prim_module.UsdGeom, "GetStageMetersPerUnit", lambda stage: 1.0)

    flattened = flatten_prototype_references(_FakeStage())
    mesh = flattened.GetPrimAtPath("/Root/Child/Mesh")
    assert mesh
    assert mesh.customData["validKey"] == "copy"
    assert "3dsmax" not in mesh.customData


def test_flatten_parent_missing_defensive_branch(monkeypatch):
    source_layer = Sdf.Layer.CreateAnonymous("missing-parent-source.usda")

    class _FakeComposedPrim:
        def GetPath(self):
            return Sdf.Path("/Missing/Child")

        def GetTypeName(self):
            return "Xform"

        def GetMetadata(self, key):
            return None

        def GetCustomData(self):
            return {}

        def GetAttributes(self):
            return []

        def GetRelationships(self):
            return []

    class _FakeStage:
        def GetRootLayer(self):
            return source_layer

        def Traverse(self, predicate):
            return iter([_FakeComposedPrim()])

    monkeypatch.setattr(prim_module.UsdGeom, "GetStageUpAxis", lambda stage: None)
    monkeypatch.setattr(prim_module.UsdGeom, "GetStageMetersPerUnit", lambda stage: 1.0)

    class _NoOpPrimSpec:
        def __new__(cls, *args, **kwargs):
            return None

    monkeypatch.setattr(prim_module.Sdf, "PrimSpec", _NoOpPrimSpec)

    flattened = flatten_prototype_references(_FakeStage())
    assert not flattened.GetPrimAtPath("/Missing/Child")
