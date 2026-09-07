# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for the Scene Optimizer UV subprocess worker."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pxr import Sdf, Usd, UsdGeom

from world_understanding.functions.graphics import so_uv_worker


class FakeMeshSchema:
    pass


class FakeAttr:
    def __init__(self, value: Any = None, *, interpolation: str | None = None) -> None:
        self._value = value
        self._interpolation = interpolation

    def HasAuthoredValue(self) -> bool:
        return self._value is not None

    def Get(self) -> Any:
        return self._value

    def GetMetadata(self, name: str) -> Any:
        if name == "interpolation":
            return self._interpolation
        return None


class FakePrim:
    def __init__(
        self,
        *,
        path: str = "",
        pseudo: bool = False,
        mesh: bool = False,
        uv: bool = False,
        uv_values: list[Any] | None = None,
        uv_interpolation: str = "constant",
        uv_indices: list[int] | None = None,
        points: list[tuple[float, float, float]] | None = None,
        face_counts: list[int] | None = None,
        face_indices: list[int] | None = None,
    ):
        self._path = path
        self._pseudo = pseudo
        self._mesh = mesh
        self._uv = uv
        self._uv_values = uv_values if uv_values is not None else [(0.0, 0.0)]
        self._uv_interpolation = uv_interpolation
        self._uv_indices = uv_indices
        self._points = (
            points
            if points is not None
            else [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        )
        self._face_counts = face_counts if face_counts is not None else [3]
        self._face_indices = face_indices if face_indices is not None else [0, 1, 2]

    def IsPseudoRoot(self) -> bool:
        return self._pseudo

    def IsA(self, schema: Any) -> bool:
        return schema is FakeMeshSchema and self._mesh

    def IsValid(self) -> bool:
        return True

    def IsInstanceProxy(self) -> bool:
        return False

    def GetPrimStack(self) -> list[Any]:
        return []

    def GetPath(self) -> str:
        return self._path

    def GetAttribute(self, name: str) -> FakeAttr | None:
        if name == "primvars:st" and self._uv:
            return FakeAttr(
                self._uv_values,
                interpolation=self._uv_interpolation,
            )
        if name == "primvars:st:indices" and self._uv_indices is not None:
            return FakeAttr(self._uv_indices)
        if name == "points" and self._mesh:
            return FakeAttr(self._points)
        if name == "faceVertexCounts" and self._mesh:
            return FakeAttr(self._face_counts)
        if name == "faceVertexIndices" and self._mesh:
            return FakeAttr(self._face_indices)
        return None

    def RemoveProperty(self, name: str) -> bool:
        if name == "primvars:st":
            self._uv = False
        return True

    def author_uv(
        self,
        values: list[tuple[float, float]] | None = None,
    ) -> None:
        self._uv = True
        if values is not None:
            self._uv_values = values

    def mutate_topology(self) -> None:
        self._face_indices = [0, 2, 1]


class FakeRootLayer:
    rootPrims: list[Any] = []

    def Export(self, output_path: str) -> None:
        Path(output_path).write_text("#usda 1.0\n", encoding="utf-8")

    def GetPrimAtPath(self, _path: Any) -> None:
        return None


class FakeStage:
    def __init__(self) -> None:
        self._prims_by_path: dict[str, FakePrim] = {}

    def set_prims(self, prims: list[FakePrim]) -> None:
        self._prims_by_path = {
            str(prim.GetPath()): prim for prim in prims if not prim.IsPseudoRoot()
        }

    def GetPseudoRoot(self) -> str:
        return "root"

    def GetRootLayer(self) -> FakeRootLayer:
        return FakeRootLayer()

    def GetPrimAtPath(self, path: str) -> FakePrim | None:
        return self._prims_by_path.get(str(path))


class FakeSdfPath:
    def __init__(self, path: str) -> None:
        self._path = path

    def __str__(self) -> str:
        return self._path

    def IsAbsolutePath(self) -> bool:
        return self._path.startswith("/")

    def IsAbsoluteRootPath(self) -> bool:
        return self._path == "/"

    def IsPrimPath(self) -> bool:
        return self.IsAbsolutePath() and self._path != "/"

    def GetPrefixes(self) -> list[FakeSdfPath]:
        return [self]


class FakeExecutionContext:
    instances: list[FakeExecutionContext] = []

    def __init__(self) -> None:
        self.stage = None
        self.removed = False
        type(self).instances.append(self)

    def set_stage(self, stage: Any) -> None:
        self.stage = stage

    def remove_stage(self) -> None:
        self.removed = True


class FakeSceneOptimizerCore:
    calls: list[tuple[str, Any, dict[str, Any]]] = []
    on_execute: Any = None

    @classmethod
    def getInstance(cls) -> FakeSceneOptimizerCore:
        return cls()

    def executeOperation(
        self, operation: str, ctx: FakeExecutionContext, op_params: dict[str, Any]
    ) -> None:
        self.calls.append((operation, ctx.stage, op_params))
        if self.on_execute is not None:
            type(self).on_execute(operation, ctx, op_params)


def _install_fake_scene_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    stage: Any,
    *,
    prims: list[FakePrim] | None = None,
) -> None:
    FakeExecutionContext.instances.clear()
    FakeSceneOptimizerCore.on_execute = None
    omni = types.ModuleType("omni")
    scene = types.ModuleType("omni.scene")
    optimizer = types.ModuleType("omni.scene.optimizer")
    core = types.ModuleType("omni.scene.optimizer.core")
    core.ExecutionContext = FakeExecutionContext
    core.SceneOptimizerCore = FakeSceneOptimizerCore

    fake_prims = prims or [
        FakePrim(pseudo=True),
        FakePrim(path="/MeshA", mesh=True, uv=True),
        FakePrim(path="/MeshB", mesh=True, uv=True),
        FakePrim(path="/Scope", mesh=False),
    ]
    if isinstance(stage, FakeStage):
        stage.set_prims(fake_prims)
    pxr = types.ModuleType("pxr")
    sdf = types.SimpleNamespace(
        Path=FakeSdfPath,
        SpecifierDef=object(),
        SpecifierOver=object(),
    )
    usd = types.SimpleNamespace(
        Stage=types.SimpleNamespace(Open=lambda _path: stage),
        PrimRange=lambda _root, *_args: fake_prims,
        TraverseInstanceProxies=lambda: object(),
    )
    usd_geom = types.SimpleNamespace(Mesh=FakeMeshSchema)
    pxr.Sdf = sdf
    pxr.Usd = usd
    pxr.UsdGeom = usd_geom

    for name, module in {
        "omni": omni,
        "omni.scene": scene,
        "omni.scene.optimizer": optimizer,
        "omni.scene.optimizer.core": core,
        "pxr": pxr,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        so_uv_worker,
        "export_stage_portably",
        lambda current_stage, path, **_kwargs: (
            current_stage.GetRootLayer().Export(path) is None
        ),
    )


def test_so_uv_worker_success_and_error_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_scene_optimizer(monkeypatch, FakeStage())
    exports: list[tuple[Any, str, list[str]]] = []
    export_context_removed: list[bool] = []

    def fake_portable_export(
        stage: Any,
        output_path: str,
        *,
        approved_dependency_roots: list[str],
    ) -> bool:
        export_context_removed.append(FakeExecutionContext.instances[-1].removed)
        exports.append((stage, output_path, approved_dependency_roots))
        stage.GetRootLayer().Export(output_path)
        return True

    monkeypatch.setattr(so_uv_worker, "export_stage_portably", fake_portable_export)
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "out.usda"
    params = {
        "input_usd_path": str(tmp_path / "in.usda"),
        "output_usd_path": str(output),
        "approved_dependency_roots": [str(tmp_path)],
        "operation": "generateAtlasUVs",
        "op_params": {"resolution": 128},
        "manifest_path": str(manifest),
    }
    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", json.dumps(params)])
    so_uv_worker.main()

    payload = json.loads(manifest.read_text())
    assert payload["status"] == "success"
    assert payload["mesh_count"] == 2
    assert payload["meshes_with_uvs"] == 2
    assert payload["scoped_mesh_paths"] == ["/MeshA", "/MeshB"]
    assert payload["uv_authored_mesh_paths"] == []
    assert payload["uv_validated_mesh_paths"] == ["/MeshA", "/MeshB"]
    assert payload["validated_preexisting_uv_paths"] == ["/MeshA", "/MeshB"]
    assert payload["stage_size_bytes"] > 0
    assert FakeSceneOptimizerCore.calls[-1][0] == "generateAtlasUVs"
    assert exports == [
        (FakeSceneOptimizerCore.calls[-1][1], str(output), [str(tmp_path)])
    ]
    assert export_context_removed == [False]
    assert FakeExecutionContext.instances[-1].removed is True

    _install_fake_scene_optimizer(monkeypatch, None)
    manifest = tmp_path / "error-manifest.json"
    params["manifest_path"] = str(manifest)
    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", json.dumps(params)])
    so_uv_worker.main()

    payload = json.loads(manifest.read_text())
    assert payload["status"] == "error"
    assert payload["failure_phase"] == "operation"
    assert payload["error_type"] == "RuntimeError"
    assert "Failed to open USD stage" in payload["error"]


def test_so_uv_worker_removes_stage_after_portable_export_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_scene_optimizer(monkeypatch, FakeStage())
    removed_during_export: list[bool] = []

    def fail_portable_export(
        _stage: Any,
        _output_path: str,
        *,
        approved_dependency_roots: list[str],
    ) -> bool:
        assert approved_dependency_roots == [str(tmp_path)]
        removed_during_export.append(FakeExecutionContext.instances[-1].removed)
        raise RuntimeError("portable export failed")

    monkeypatch.setattr(
        so_uv_worker,
        "export_stage_portably",
        fail_portable_export,
    )
    manifest = tmp_path / "manifest.json"
    params = {
        "input_usd_path": str(tmp_path / "in.usda"),
        "output_usd_path": str(tmp_path / "out.usda"),
        "approved_dependency_roots": [str(tmp_path)],
        "operation": "generateProjectionUVs",
        "op_params": {"projectionType": 4},
        "manifest_path": str(manifest),
    }
    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", json.dumps(params)])

    so_uv_worker.main()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["failure_phase"] == "export"
    assert payload["error_type"] == "RuntimeError"
    assert "portable export failed" in payload["error"]
    assert removed_during_export == [False]
    assert FakeExecutionContext.instances[-1].removed is True


def test_materializes_only_requested_hidden_backing_root_and_restores_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "instance.usda"
    source_stage = Usd.Stage.CreateNew(str(source))
    source_stage.CreateClassPrim("/World/Prototypes")
    backing = UsdGeom.Xform.Define(
        source_stage,
        "/World/Prototypes/Body",
    ).GetPrim()
    mesh = UsdGeom.Mesh.Define(source_stage, "/World/Prototypes/Body/Part/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    instance = UsdGeom.Xform.Define(source_stage, "/World/Instance").GetPrim()
    instance.GetReferences().AddInternalReference(backing.GetPath())
    instance.SetInstanceable(True)

    stage = Usd.Stage.Open(source_stage.Flatten(addSourceFileComment=False))
    hidden_roots = [
        spec
        for spec in stage.GetRootLayer().rootPrims
        if spec.specifier == Sdf.SpecifierOver
    ]
    assert len(hidden_roots) == 1
    hidden_root_path = str(hidden_roots[0].path)
    target_path = f"{hidden_root_path}/Part/Mesh"
    assert not stage.GetPrimAtPath(target_path).IsDefined()
    assert so_uv_worker._scoped_meshes(stage, Usd, UsdGeom, [target_path]) == []

    changed = so_uv_worker._materialize_hidden_mesh_ancestors(
        stage,
        Sdf,
        [target_path],
    )
    scoped = so_uv_worker._scoped_meshes(stage, Usd, UsdGeom, [target_path])
    assert [str(prim.GetPath()) for prim in scoped] == [target_path]
    assert stage.GetPrimAtPath("/World/Instance").IsInstanceable()

    so_uv_worker._restore_materialized_specs(changed)
    assert not stage.GetPrimAtPath(target_path).IsDefined()
    assert stage.GetRootLayer().GetPrimAtPath(hidden_root_path).specifier == (
        Sdf.SpecifierOver
    )
    assert stage.GetPrimAtPath("/World/Instance").IsInstanceable()

    full_stage_changes = so_uv_worker._materialize_hidden_mesh_ancestors(
        stage,
        Sdf,
        [],
    )
    assert [
        str(prim.GetPath())
        for prim in Usd.PrimRange(
            stage.GetPseudoRoot(),
            Usd.TraverseInstanceProxies(),
        )
        if prim.IsInstanceProxy() and prim.IsA(UsdGeom.Mesh)
    ] == ["/World/Instance/Part/Mesh"]
    assert [
        str(prim.GetPath())
        for prim in so_uv_worker._scoped_meshes(
            stage,
            Usd,
            UsdGeom,
            [],
        )
    ] == [target_path]
    so_uv_worker._restore_materialized_specs(full_stage_changes)
    assert not stage.GetPrimAtPath(target_path).IsDefined()


def _external_instance_stage(*, with_uv: bool = False) -> Usd.Stage:
    source_layer = Sdf.Layer.CreateAnonymous("external-instance-source.usda")
    source_stage = Usd.Stage.Open(source_layer)
    source_root = UsdGeom.Xform.Define(source_stage, "/Model").GetPrim()
    source_stage.SetDefaultPrim(source_root)
    mesh = UsdGeom.Mesh.Define(source_stage, "/Model/Part/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    if with_uv:
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        ).Set([(0, 0), (1, 0), (0, 1)])

    stage = Usd.Stage.CreateInMemory()
    instance = UsdGeom.Xform.Define(stage, "/World/Instance").GetPrim()
    instance.GetReferences().AddReference(source_layer.identifier, "/Model")
    instance.SetInstanceable(True)
    return stage


def test_scoped_meshes_descends_into_requested_instance_root() -> None:
    stage = _external_instance_stage()

    assert [
        str(prim.GetPath())
        for prim in so_uv_worker._scoped_meshes(
            stage,
            Usd,
            UsdGeom,
            ["/World/Instance"],
        )
    ] == ["/World/Instance/Part/Mesh"]


def test_scoped_meshes_finds_full_stage_instance_only_mesh() -> None:
    stage = _external_instance_stage()
    assert not any(
        prim.IsA(UsdGeom.Mesh) for prim in Usd.PrimRange(stage.GetPseudoRoot())
    )

    assert [
        str(prim.GetPath())
        for prim in so_uv_worker._scoped_meshes(stage, Usd, UsdGeom, [])
    ] == ["/World/Instance/Part/Mesh"]


def test_overwrite_proxy_scope_only_fails_when_existing_uv_cannot_be_cleared() -> None:
    clean_stage = _external_instance_stage()
    clean_proxy = clean_stage.GetPrimAtPath("/World/Instance/Part/Mesh")
    assert clean_proxy.IsInstanceProxy()
    so_uv_worker._strip_scoped_uvs([clean_proxy])

    authored_stage = _external_instance_stage(with_uv=True)
    authored_proxy = authored_stage.GetPrimAtPath("/World/Instance/Part/Mesh")
    assert authored_proxy.IsInstanceProxy()
    with pytest.raises(
        RuntimeError,
        match=r"instance proxy.*fresh Scene Optimizer authorship cannot be proven",
    ):
        so_uv_worker._strip_scoped_uvs([authored_proxy])


def test_full_stage_materializes_nested_over_without_touching_visible_mesh() -> None:
    layer = Sdf.Layer.CreateAnonymous("nested-over.usda")
    assert layer.ImportFromString(
        """#usda 1.0
def Xform "Container"
{
    over "Hidden"
    {
        def Mesh "Mesh"
        {
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
        }
        def Mesh "MeshB"
        {
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
        }
    }
}
def Mesh "Visible"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
}
"""
    )
    stage = Usd.Stage.Open(layer)
    hidden_path = "/Container/Hidden/Mesh"
    hidden_b_path = "/Container/Hidden/MeshB"
    assert not stage.GetPrimAtPath(hidden_path).IsDefined()
    assert not stage.GetPrimAtPath(hidden_b_path).IsDefined()
    assert stage.GetPrimAtPath("/Visible").IsDefined()

    changed = so_uv_worker._materialize_hidden_mesh_ancestors(stage, Sdf, [])

    assert {
        str(prim.GetPath())
        for prim in so_uv_worker._scoped_meshes(
            stage,
            Usd,
            UsdGeom,
            [],
        )
    } == {hidden_path, hidden_b_path, "/Visible"}
    assert [str(spec.path) for spec, _specifier, _type_name in changed] == [
        "/Container/Hidden"
    ]
    assert stage.GetRootLayer().GetPrimAtPath("/Visible").specifier == Sdf.SpecifierDef

    so_uv_worker._restore_materialized_specs(changed)
    assert not stage.GetPrimAtPath(hidden_path).IsDefined()
    assert not stage.GetPrimAtPath(hidden_b_path).IsDefined()
    assert stage.GetPrimAtPath("/Visible").IsDefined()


def test_scoped_meshes_deduplicates_overlapping_requested_roots() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Root")
    UsdGeom.Mesh.Define(stage, "/Root/Mesh")

    assert [
        str(prim.GetPath())
        for prim in so_uv_worker._scoped_meshes(
            stage,
            Usd,
            UsdGeom,
            ["/Root", "/Root/Mesh"],
        )
    ] == ["/Root/Mesh"]


def test_each_requested_path_must_resolve_a_mesh() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(stage, "/Valid")

    with pytest.raises(RuntimeError, match=r"/Missing"):
        so_uv_worker._resolve_requested_meshes(
            stage,
            Usd,
            UsdGeom,
            ["/Valid", "/Missing"],
            "generateProjectionUVs",
        )

    assert so_uv_worker._resolve_requested_meshes(
        stage,
        Usd,
        UsdGeom,
        ["/Valid"],
        "generateProjectionUVs",
    ) == [
        {
            "requested_path": "/Valid",
            "mesh_count": 1,
            "mesh_paths": ["/Valid"],
        }
    ]


@pytest.mark.parametrize(
    ("prims", "error_text", "operation_called"),
    [
        ([FakePrim(pseudo=True)], "resolved zero mesh prims", False),
        (
            [FakePrim(pseudo=True), FakePrim(path="/Mesh", mesh=True, uv=False)],
            "did not author topology-valid UVs",
            True,
        ),
        (
            [
                FakePrim(pseudo=True),
                FakePrim(path="/Good", mesh=True, uv=True),
                FakePrim(path="/Missing", mesh=True, uv=False),
            ],
            "/Missing: missing authored primvars:st",
            True,
        ),
    ],
)
def test_so_uv_worker_fails_closed_on_empty_effective_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prims: list[FakePrim],
    error_text: str,
    operation_called: bool,
) -> None:
    _install_fake_scene_optimizer(monkeypatch, FakeStage(), prims=prims)
    FakeSceneOptimizerCore.calls.clear()
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "out.usda"
    params = {
        "input_usd_path": str(tmp_path / "in.usda"),
        "output_usd_path": str(output),
        "approved_dependency_roots": [str(tmp_path)],
        "operation": "generateProjectionUVs",
        "op_params": {"projectionType": 4},
        "manifest_path": str(manifest),
    }
    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", json.dumps(params)])

    so_uv_worker.main()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert error_text in payload["error"]
    assert bool(FakeSceneOptimizerCore.calls) is operation_called
    assert not output.exists()


def test_so_uv_worker_overwrite_proves_fresh_exact_scope_authorship(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = FakePrim(path="/Mesh", mesh=True, uv=True)
    _install_fake_scene_optimizer(
        monkeypatch,
        FakeStage(),
        prims=[FakePrim(pseudo=True), target],
    )
    FakeSceneOptimizerCore.calls.clear()

    def reauthor_uv(_operation: str, _ctx: Any, _params: dict[str, Any]) -> None:
        assert target.GetAttribute("primvars:st") is None
        target.author_uv()

    FakeSceneOptimizerCore.on_execute = reauthor_uv
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "out.usda"
    params = {
        "input_usd_path": str(tmp_path / "in.usda"),
        "output_usd_path": str(output),
        "approved_dependency_roots": [str(tmp_path)],
        "operation": "generateProjectionUVs",
        "op_params": {
            "projectionType": 4,
            "overwriteExisting": True,
            "paths": ["/Mesh"],
        },
        "manifest_path": str(manifest),
    }
    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", json.dumps(params)])

    so_uv_worker.main()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["overwrite_existing"] is True
    assert payload["mesh_count"] == 1
    assert payload["meshes_with_uvs"] == 1
    assert payload["scoped_mesh_paths"] == ["/Mesh"]
    assert payload["uv_authored_mesh_paths"] == ["/Mesh"]
    assert payload["uv_validated_mesh_paths"] == ["/Mesh"]
    assert payload["validated_preexisting_uv_paths"] == []
    assert payload["requested_path_meshes"] == [
        {"requested_path": "/Mesh", "mesh_count": 1, "mesh_paths": ["/Mesh"]}
    ]
    assert output.is_file()


def test_so_uv_worker_rejects_malformed_original_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = FakePrim(
        path="/Mesh",
        mesh=True,
        face_counts=[3],
        face_indices=[0, 1],
    )
    _install_fake_scene_optimizer(
        monkeypatch,
        FakeStage(),
        prims=[FakePrim(pseudo=True), malformed],
    )
    FakeSceneOptimizerCore.calls.clear()
    manifest = tmp_path / "manifest.json"
    params = {
        "input_usd_path": str(tmp_path / "in.usda"),
        "output_usd_path": str(tmp_path / "out.usda"),
        "approved_dependency_roots": [str(tmp_path)],
        "operation": "generateProjectionUVs",
        "op_params": {"projectionType": 4},
        "manifest_path": str(manifest),
    }
    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", json.dumps(params)])

    so_uv_worker.main()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert (
        "faceVertexCounts sum 3 does not match faceVertexIndices count 2"
        in (payload["error"])
    )
    assert FakeSceneOptimizerCore.calls == []


def test_so_uv_worker_rejects_topology_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = FakePrim(path="/Mesh", mesh=True, uv=True)
    _install_fake_scene_optimizer(
        monkeypatch,
        FakeStage(),
        prims=[FakePrim(pseudo=True), target],
    )
    FakeSceneOptimizerCore.calls.clear()
    FakeSceneOptimizerCore.on_execute = (
        lambda _operation, _ctx, _params: target.mutate_topology()
    )
    manifest = tmp_path / "manifest.json"
    params = {
        "input_usd_path": str(tmp_path / "in.usda"),
        "output_usd_path": str(tmp_path / "out.usda"),
        "approved_dependency_roots": [str(tmp_path)],
        "operation": "generateProjectionUVs",
        "op_params": {"projectionType": 4},
        "manifest_path": str(manifest),
    }
    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", json.dumps(params)])

    so_uv_worker.main()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert "changed mesh topology outside the UV contract" in payload["error"]
    assert not Path(params["output_usd_path"]).exists()


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_so_uv_worker_rejects_non_finite_uvs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    non_finite: float,
) -> None:
    target = FakePrim(path="/Mesh", mesh=True, uv=False)
    _install_fake_scene_optimizer(
        monkeypatch,
        FakeStage(),
        prims=[FakePrim(pseudo=True), target],
    )
    FakeSceneOptimizerCore.calls.clear()
    FakeSceneOptimizerCore.on_execute = (
        lambda _operation, _ctx, _params: target.author_uv([(non_finite, 0.0)])
    )
    manifest = tmp_path / "manifest.json"
    params = {
        "input_usd_path": str(tmp_path / "in.usda"),
        "output_usd_path": str(tmp_path / "out.usda"),
        "approved_dependency_roots": [str(tmp_path)],
        "operation": "generateProjectionUVs",
        "op_params": {"projectionType": 4},
        "manifest_path": str(manifest),
    }
    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", json.dumps(params)])

    so_uv_worker.main()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert "primvars:st value 0 is non-finite" in payload["error"]
    assert not Path(params["output_usd_path"]).exists()


@pytest.mark.parametrize(
    ("prim", "topology", "expected_error"),
    [
        (
            FakePrim(path="/Mesh", mesh=True, uv=True, uv_values=[]),
            (((0.0, 0.0, 0.0),), (3,), (0, 0, 0)),
            "primvars:st is empty",
        ),
        (
            FakePrim(path="/Mesh", mesh=True, uv=True, uv_values=[(0.0,)]),
            (((0.0, 0.0, 0.0),), (3,), (0, 0, 0)),
            "primvars:st value 0 is not a 2D coordinate",
        ),
        (
            FakePrim(
                path="/Mesh",
                mesh=True,
                uv=True,
                uv_interpolation="uniform",
            ),
            (((0.0, 0.0, 0.0),), (3,), (0, 0, 0)),
            None,
        ),
        (
            FakePrim(
                path="/Mesh",
                mesh=True,
                uv=True,
                uv_values=[(0.0, 0.0)] * 3,
                uv_interpolation="vertex",
            ),
            (
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                ),
                (3,),
                (0, 1, 2),
            ),
            None,
        ),
        (
            FakePrim(
                path="/Mesh",
                mesh=True,
                uv=True,
                uv_values=[(0.0, 0.0)] * 3,
                uv_interpolation="varying",
            ),
            (
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                ),
                (3,),
                (0, 1, 2),
            ),
            None,
        ),
        (
            FakePrim(
                path="/Mesh",
                mesh=True,
                uv=True,
                uv_values=[(0.0, 0.0)] * 3,
                uv_interpolation="faceVarying",
            ),
            (((0.0, 0.0, 0.0),), (3,), (0, 0, 0)),
            None,
        ),
        (
            FakePrim(
                path="/Mesh",
                mesh=True,
                uv=True,
                uv_interpolation="edge",
            ),
            (((0.0, 0.0, 0.0),), (3,), (0, 0, 0)),
            "unsupported interpolation 'edge'",
        ),
        (
            FakePrim(
                path="/Mesh",
                mesh=True,
                uv=True,
                uv_interpolation="uniform",
            ),
            (((0.0, 0.0, 0.0),), (), ()),
            "mesh topology has no uniform elements",
        ),
        (
            FakePrim(
                path="/Mesh",
                mesh=True,
                uv=True,
                uv_interpolation="vertex",
            ),
            (
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                ),
                (3,),
                (0, 1, 2),
            ),
            "vertex UV count 1 does not match topology count 3",
        ),
        (
            FakePrim(
                path="/Mesh",
                mesh=True,
                uv=True,
                uv_values=[(0.0, 0.0), (1.0, 0.0)],
                uv_interpolation="faceVarying",
                uv_indices=[0, 1, 2],
            ),
            (((0.0, 0.0, 0.0),), (3,), (0, 0, 0)),
            "primvars:st:indices references values outside primvars:st",
        ),
    ],
    ids=[
        "empty",
        "not-2d",
        "uniform",
        "vertex",
        "varying",
        "face-varying",
        "unsupported",
        "empty-topology-domain",
        "count-mismatch",
        "bad-index",
    ],
)
def test_uv_topology_validation_covers_interpolation_contracts(
    prim: FakePrim,
    topology: tuple[tuple[Any, ...], tuple[int, ...], tuple[int, ...]],
    expected_error: str | None,
) -> None:
    assert so_uv_worker._uv_topology_error(prim, topology) == expected_error


def test_so_uv_worker_rejects_changed_mesh_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = FakePrim(path="/Original", mesh=True, uv=True)
    replacement = FakePrim(path="/Replacement", mesh=True, uv=True)
    _install_fake_scene_optimizer(
        monkeypatch,
        FakeStage(),
        prims=[FakePrim(pseudo=True), original],
    )
    scopes = iter(([original], [replacement]))
    monkeypatch.setattr(
        so_uv_worker,
        "_scoped_meshes",
        lambda *_args: next(scopes),
    )
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "out.usda"
    params = {
        "input_usd_path": str(tmp_path / "in.usda"),
        "output_usd_path": str(output),
        "approved_dependency_roots": [str(tmp_path)],
        "operation": "generateProjectionUVs",
        "op_params": {"projectionType": 4},
        "manifest_path": str(manifest),
    }
    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", json.dumps(params)])

    so_uv_worker.main()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert "missing=['/Original'], added=['/Replacement']" in payload["error"]
    assert not output.exists()


def test_so_uv_worker_argument_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py"])
    with pytest.raises(SystemExit) as missing_args:
        so_uv_worker.main()
    assert missing_args.value.code == 1
    assert "Usage:" in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", "{not-json"])
    with pytest.raises(SystemExit) as bad_json:
        so_uv_worker.main()
    assert bad_json.value.code == 1
    assert "Invalid JSON" in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", "[]"])
    with pytest.raises(SystemExit) as non_object:
        so_uv_worker.main()
    assert non_object.value.code == 1
    assert "must be an object" in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", "{}"])
    with pytest.raises(SystemExit) as missing_manifest:
        so_uv_worker.main()
    assert missing_manifest.value.code == 1
    assert "manifest_path" in capsys.readouterr().err

    for invalid_manifest in (123, "  "):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "so_uv_worker.py",
                json.dumps({"manifest_path": invalid_manifest}),
            ],
        )
        with pytest.raises(SystemExit) as invalid_manifest_error:
            so_uv_worker.main()
        assert invalid_manifest_error.value.code == 1
        assert "non-empty string" in capsys.readouterr().err

    manifest = tmp_path / "missing-key-manifest.json"
    params = {
        "input_usd_path": str(tmp_path / "in.usda"),
        "output_usd_path": str(tmp_path / "out.usda"),
        "manifest_path": str(manifest),
    }
    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", json.dumps(params)])
    so_uv_worker.main()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert "approved_dependency_roots" in payload["error"]


def test_so_uv_worker_rejects_invalid_dependency_roots_before_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_scene_optimizer(monkeypatch, FakeStage())
    FakeSceneOptimizerCore.calls.clear()
    manifest = tmp_path / "manifest.json"
    params = {
        "input_usd_path": str(tmp_path / "in.usda"),
        "output_usd_path": str(tmp_path / "out.usda"),
        "approved_dependency_roots": [str(Path(tmp_path.anchor))],
        "operation": "generateProjectionUVs",
        "op_params": {},
        "manifest_path": str(manifest),
    }
    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", json.dumps(params)])

    so_uv_worker.main()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert "must not contain filesystem roots" in payload["error"]
    assert FakeSceneOptimizerCore.calls == []
