# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for USD spatial query helpers."""

from __future__ import annotations

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from world_understanding.functions.graphics import usd_spatial


def _define_mesh(stage: Usd.Stage, path: str, translate: Gf.Vec3d) -> Usd.Prim:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0, 0, 0),
                Gf.Vec3f(1, 0, 0),
                Gf.Vec3f(0, 1, 0),
                Gf.Vec3f(0, 0, 1),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 1, 1)]))
    UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(translate)
    return mesh.GetPrim()


def _make_scene() -> tuple[Usd.Stage, Usd.Prim, Usd.Prim]:
    stage = Usd.Stage.CreateInMemory("scene.usda")
    stage.SetStartTimeCode(1)
    stage.SetEndTimeCode(12)
    stage.SetTimeCodesPerSecond(24)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/World")
    mesh_a = _define_mesh(stage, "/World/ChairA", Gf.Vec3d(0, 0, 0))
    mesh_b = _define_mesh(stage, "/World/TableB", Gf.Vec3d(3, 0, 0))
    UsdGeom.Xform.Define(stage, "/World/Empty")

    material = UsdShade.Material.Define(stage, "/World/Looks/Red")
    UsdShade.MaterialBindingAPI.Apply(mesh_a).Bind(material)

    variant_set = mesh_a.GetVariantSets().AddVariantSet("lod")
    variant_set.AddVariant("high")
    variant_set.SetVariantSelection("high")
    mesh_a.CreateAttribute("custom:int", Sdf.ValueTypeNames.Int).Set(7)
    mesh_a.CreateAttribute("custom:vec", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3f(1, 2, 3)
    )
    mesh_a.CreateAttribute("custom:matrix", Sdf.ValueTypeNames.Matrix4d).Set(
        Gf.Matrix4d(1)
    )
    return stage, mesh_a, mesh_b


def test_bbox_distance_material_geometry_and_transform_helpers() -> None:
    stage, mesh_a, mesh_b = _make_scene()

    assert usd_spatial.get_world_bbox(stage, "/Missing") is None
    bbox_a = usd_spatial.get_world_bbox(stage, "/World/ChairA")
    bbox_b = usd_spatial.get_world_bbox(stage, "/World/TableB")
    assert bbox_a is not None
    assert bbox_b is not None
    assert bbox_a["volume"] > 0
    assert usd_spatial.bbox_overlaps(
        bbox_a["min"], bbox_a["max"], bbox_a["min"], bbox_a["max"]
    )
    assert not usd_spatial.bbox_overlaps(
        bbox_a["min"], bbox_a["max"], bbox_b["min"], bbox_b["max"]
    )
    assert (
        usd_spatial.bbox_distance(
            bbox_a["min"], bbox_a["max"], bbox_b["min"], bbox_b["max"]
        )
        > 0
    )
    assert (
        usd_spatial.bbox_distance(
            bbox_b["min"], bbox_b["max"], bbox_a["min"], bbox_a["max"]
        )
        > 0
    )
    assert (
        usd_spatial.point_to_bbox_distance(
            [0.5, 0.5, 0.5], bbox_a["min"], bbox_a["max"]
        )
        == 0
    )
    assert (
        usd_spatial.point_to_bbox_distance([-1, 0, 0], bbox_a["min"], bbox_a["max"]) > 0
    )

    assert usd_spatial.get_bound_material_path(mesh_a) == "/World/Looks/Red"
    assert usd_spatial.get_bound_material_path(mesh_b) is None
    assert usd_spatial.get_geometry_stats(mesh_a)["vertex_count"] == 4
    assert usd_spatial.get_geometry_stats(stage.GetPrimAtPath("/World/Empty")) is None
    assert usd_spatial.get_world_transform(stage, "/Missing") is None
    assert usd_spatial.get_world_transform(stage, "/World/TableB")[3][0] == 3

    mat_map = usd_spatial.get_material_binding_map(stage)
    assert mat_map["/World/Looks/Red"] == ["/World/ChairA"]
    assert "/World/TableB" in mat_map["(unassigned)"]
    scoped_map = usd_spatial.get_material_binding_map(
        stage, start_prim=stage.GetPrimAtPath("/World/Looks")
    )
    assert scoped_map == {}


def test_query_prims_filters_sorting_and_limits() -> None:
    stage, _mesh_a, _mesh_b = _make_scene()

    assert [r["path"] for r in usd_spatial.query_prims(stage, prim_type="Mesh")] == [
        "/World/ChairA",
        "/World/TableB",
    ]
    assert usd_spatial.query_prims(stage, name_pattern="Chair*")[0]["material"] == (
        "/World/Looks/Red"
    )
    assert usd_spatial.query_prims(stage, path_pattern="/World/Table*")[0]["path"] == (
        "/World/TableB"
    )
    assert [r["path"] for r in usd_spatial.query_prims(stage, has_material=True)] == [
        "/World/ChairA"
    ]
    assert all(
        "material" not in r for r in usd_spatial.query_prims(stage, has_material=False)
    )
    assert usd_spatial.query_prims(stage, min_size=0.1, max_size=2.0)
    assert usd_spatial.query_prims(stage, min_size=999) == []
    assert usd_spatial.query_prims(stage, prim_type="Mesh", max_size=0.0001) == []

    near_point = usd_spatial.query_prims(
        stage, prim_type="Mesh", near=[0, 0, 0], radius=0.1, sort_by="distance"
    )
    assert near_point[0]["path"] == "/World/ChairA"
    near_prim = usd_spatial.query_prims(
        stage, prim_type="Mesh", near="/World/ChairA", sort_by="distance"
    )
    assert near_prim[0]["path"] == "/World/ChairA"
    assert (
        usd_spatial.query_prims(
            stage, prim_type="Mesh", near=[100, 100, 100], radius=0.1
        )
        == []
    )

    overlapping = usd_spatial.query_prims(stage, overlaps="/World/ChairA")
    assert all(result["path"] != "/World/ChairA" for result in overlapping)
    assert usd_spatial.query_prims(stage, overlaps="/Missing")

    by_size = usd_spatial.query_prims(stage, sort_by="size", limit=1)
    assert len(by_size) == 1
    by_type = usd_spatial.query_prims(stage, sort_by="type", start_prim="/World")
    assert by_type[0]["type"] <= by_type[-1]["type"]
    assert usd_spatial.query_prims(stage, start_prim="/Other") == []
    assert usd_spatial.query_prims(stage, active_only=True)


def test_scene_summary_and_inspect_prim() -> None:
    stage, _mesh_a, _mesh_b = _make_scene()

    summary = usd_spatial.scene_summary(stage, start_prim="/World", top_n=1)
    assert summary["stage_info"]["up_axis"] == "Y"
    assert summary["composition"]["total_prims"] >= 4
    assert summary["composition"]["type_counts"]["Mesh"] == 2
    assert summary["largest_prims"]
    assert summary["materials"][0]["bound_prim_count"] >= 1
    assert (
        usd_spatial.scene_summary(stage, start_prim="/Missing")["composition"][
            "total_prims"
        ]
        == 0
    )

    assert usd_spatial.inspect_prim(stage, "/Missing") is None
    inspected = usd_spatial.inspect_prim(
        stage,
        "/World/ChairA",
        include_world_transform=True,
        include_geometry=True,
        include_properties=True,
    )
    assert inspected["path"] == "/World/ChairA"
    assert inspected["child_count"] == 0
    assert inspected["descendant_count"] == 0
    assert inspected["material"] == "/World/Looks/Red"
    assert inspected["variants"] == {"lod": "high"}
    assert inspected["geometry"]["face_count"] == 1
    assert inspected["world_transform"]
    assert inspected["local_transform"]
    assert inspected["properties"]["custom:int"] == 7
    assert inspected["properties"]["custom:vec"] == [1.0, 2.0, 3.0]
    assert inspected["properties"]["custom:matrix"][0][0] == 1.0


def test_edge_branches_with_fakes(monkeypatch) -> None:
    stage, _mesh_a, _mesh_b = _make_scene()

    class _EmptyRange:
        def IsEmpty(self) -> bool:
            return True

    class _EmptyBBox:
        def ComputeAlignedRange(self):
            return _EmptyRange()

    monkeypatch.setattr(usd_spatial, "get_bbox_from_prim", lambda prim: _EmptyBBox())
    assert usd_spatial.get_world_bbox(stage, "/World/ChairA") is None

    class _FakePrim:
        def __init__(self, path: str, *, active: bool = True):
            self._path = path
            self._active = active

        def GetPath(self):
            return self._path

        def GetName(self):
            return self._path.rsplit("/", 1)[-1]

        def GetTypeName(self):
            return "Mesh"

        def IsActive(self):
            return self._active

    monkeypatch.setattr(
        usd_spatial,
        "traverse_prims",
        lambda stage: iter([_FakePrim("/World/Hidden", active=False)]),
    )
    assert usd_spatial.query_prims(stage, active_only=True) == []

    ref_bbox = {
        "min": [0, 0, 0],
        "max": [1, 1, 1],
        "size": [1, 1, 1],
        "center": [0.5, 0.5, 0.5],
        "volume": 1,
    }

    def fake_world_bbox(_stage, path):
        return ref_bbox if path == "/Target" else None

    monkeypatch.setattr(usd_spatial, "get_world_bbox", fake_world_bbox)
    monkeypatch.setattr(
        usd_spatial, "traverse_prims", lambda stage: iter([_FakePrim("/World/NoBBox")])
    )
    assert usd_spatial.query_prims(stage, overlaps="/Target") == []
    assert usd_spatial.query_prims(stage, near=[0, 0, 0], radius=0.1) == []

    class _SummaryPrim:
        def GetPath(self):
            return "/World/Instance"

        def GetTypeName(self):
            return "Mesh"

        def IsInstance(self):
            return True

        def IsA(self, _schema):
            return True

    class _RaisingBBoxCache:
        def __init__(self, *args, **kwargs):
            pass

        def ComputeWorldBound(self, _prim):
            raise RuntimeError("bbox bad")

    monkeypatch.setattr(
        usd_spatial, "traverse_prims", lambda stage: iter([_SummaryPrim()])
    )
    monkeypatch.setattr(usd_spatial.UsdGeom, "BBoxCache", _RaisingBBoxCache)
    monkeypatch.setattr(usd_spatial, "get_material_binding_map", lambda stage: {})
    summary = usd_spatial.scene_summary(stage)
    assert summary["composition"]["instance_count"] == 1
    assert summary["spatial_extents"] is None

    class _BadProperty:
        def GetName(self):
            return "bad"

    class _FakeVariants:
        def GetNames(self):
            return []

    class _FakeParent:
        def GetPath(self):
            return "/World"

    class _InspectablePrim:
        def IsValid(self):
            return True

        def GetPath(self):
            return "/World/Broken"

        def GetTypeName(self):
            return "Xform"

        def IsActive(self):
            return True

        def GetParent(self):
            return _FakeParent()

        def GetChildren(self):
            return []

        def GetVariantSets(self):
            return _FakeVariants()

        def IsA(self, _schema):
            return False

        def GetAuthoredProperties(self):
            return [_BadProperty()]

        def GetAttribute(self, _name):
            raise RuntimeError("bad attr")

    class _FakeStage:
        def GetPrimAtPath(self, _path):
            return _InspectablePrim()

    monkeypatch.setattr(usd_spatial, "get_world_bbox", lambda stage, path: None)
    monkeypatch.setattr(usd_spatial, "get_bound_material_path", lambda prim: None)
    monkeypatch.setattr(usd_spatial.Usd, "PrimRange", lambda prim: [prim])
    assert "properties" not in usd_spatial.inspect_prim(
        _FakeStage(), "/World/Broken", include_properties=True
    )
