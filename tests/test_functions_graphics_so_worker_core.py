# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from world_understanding.functions.graphics import so_worker as worker


class _Prim:
    def __init__(
        self,
        path: str,
        *,
        is_mesh: bool = True,
        pseudo: bool = False,
        parent: Any = None,
        name: str | None = None,
    ) -> None:
        self._path = path
        self._is_mesh = is_mesh
        self._pseudo = pseudo
        self._parent = parent
        self._name = name or path.rsplit("/", 1)[-1]

    def IsPseudoRoot(self) -> bool:
        return self._pseudo

    def IsA(self, _schema: object) -> bool:
        return self._is_mesh

    def GetPath(self) -> str:
        return self._path

    def GetParent(self) -> Any:
        return self._parent

    def GetName(self) -> str:
        return self._name


class _Stage:
    def __init__(
        self, prims: list[_Prim] | None = None, *, export_ok: bool = True
    ) -> None:
        self.prims = prims or []
        self.export_ok = export_ok
        self.removed = False

    def GetPseudoRoot(self) -> _Stage:
        return self

    def GetRootLayer(self) -> _Stage:
        return self

    def Export(self, path: str) -> bool:
        Path(path).write_text("usd", encoding="utf-8")
        return self.export_ok


def _install_pxr(monkeypatch: pytest.MonkeyPatch, stage: _Stage | None = None) -> None:
    pxr_mod = types.ModuleType("pxr")
    usd_mod = types.ModuleType("pxr.Usd")
    usd_geom_mod = types.ModuleType("pxr.UsdGeom")

    class Mesh:
        pass

    class StageApi:
        @staticmethod
        def Open(_path: str) -> _Stage | None:
            return stage

    usd_geom_mod.Mesh = Mesh
    usd_mod.PrimDefaultPredicate = object()
    usd_mod.TraverseInstanceProxies = lambda: object()
    usd_mod.PrimRange = lambda root, predicate: iter(root.prims)
    usd_mod.Stage = StageApi
    pxr_mod.Usd = usd_mod
    pxr_mod.UsdGeom = usd_geom_mod

    monkeypatch.setitem(sys.modules, "pxr", pxr_mod)
    monkeypatch.setitem(sys.modules, "pxr.Usd", usd_mod)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", usd_geom_mod)


def _install_scene_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_operation: str | None = None,
) -> list[str]:
    calls: list[str] = []
    omni_mod = types.ModuleType("omni")
    scene_mod = types.ModuleType("omni.scene")
    optimizer_mod = types.ModuleType("omni.scene.optimizer")
    core_mod = types.ModuleType("omni.scene.optimizer.core")

    class ExecutionContext:
        def __init__(self) -> None:
            self.stage = None
            self.removed = False

        def set_stage(self, stage: _Stage) -> None:
            self.stage = stage

        def remove_stage(self) -> None:
            self.removed = True

    class SceneOptimizerCore:
        @classmethod
        def getInstance(cls) -> SceneOptimizerCore:
            return cls()

        def executeOperation(
            self, op_name: str, _ctx: ExecutionContext, _params: dict[str, Any]
        ) -> None:
            calls.append(op_name)
            if op_name == fail_operation:
                raise RuntimeError("operation failed")

    core_mod.ExecutionContext = ExecutionContext
    core_mod.SceneOptimizerCore = SceneOptimizerCore
    monkeypatch.setitem(sys.modules, "omni", omni_mod)
    monkeypatch.setitem(sys.modules, "omni.scene", scene_mod)
    monkeypatch.setitem(sys.modules, "omni.scene.optimizer", optimizer_mod)
    monkeypatch.setitem(sys.modules, "omni.scene.optimizer.core", core_mod)
    return calls


def test_capture_mesh_paths_uses_natural_sort_and_skips_non_meshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _Stage(
        [
            _Prim("/", pseudo=True),
            _Prim("/World/Mesh_10"),
            _Prim("/World/Light", is_mesh=False),
            _Prim("/World/Mesh_2"),
        ]
    )
    _install_pxr(monkeypatch)

    assert worker.capture_mesh_paths(stage) == ["/World/Mesh_2", "/World/Mesh_10"]
    assert worker.capture_mesh_paths(stage, include_instance_proxies=True) == [
        "/World/Mesh_2",
        "/World/Mesh_10",
    ]


def test_merge_split_mappings_handles_empty_and_independent_new_mappings() -> None:
    assert worker._merge_split_mappings({}, {"/World/A": ["/World/A_part"]}) == {
        "/World/A": ["/World/A_part"]
    }
    assert worker._merge_split_mappings({"/World/A": ["/World/A_part"]}, {}) == {
        "/World/A": ["/World/A_part"]
    }

    result = worker._merge_split_mappings(
        {"/World/A": ["/World/A_part"]},
        {"/World/B": ["/World/B_part"]},
    )

    assert result == {
        "/World/A": ["/World/A_part"],
        "/World/B": ["/World/B_part"],
    }


def test_track_deduplicate_geometry_reads_internal_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Reference:
        def __init__(self, *, asset_path: str = "", prim_path: str = "") -> None:
            self.assetPath = asset_path
            self.primPath = prim_path

    class _ReferenceList:
        explicitItems = [_Reference(prim_path="/World/Prototype")]
        prependedItems = [_Reference(asset_path="external.usd", prim_path="/External")]
        appendedItems = [_Reference()]

    class _Spec:
        hasReferences = True
        referenceList = _ReferenceList()

    class _NoRefsSpec:
        hasReferences = False

    class _Parent:
        def __init__(self, *, instance: bool) -> None:
            self._instance = instance

        def IsInstance(self) -> bool:
            return self._instance

        def GetPrimStack(self) -> list[object]:
            return [_NoRefsSpec(), _Spec()]

    instance_parent = _Parent(instance=True)
    prototype_parent = _Parent(instance=True)
    stage = _Stage(
        [
            _Prim("/World/NonMesh", is_mesh=False),
            _Prim("/World/NoParent", parent=None),
            _Prim("/World/NotInstance/Geometry", parent=_Parent(instance=False)),
            _Prim("/World/Instance/Geometry", parent=instance_parent, name="Geometry"),
            _Prim(
                "/World/Prototype/Geometry", parent=prototype_parent, name="Geometry"
            ),
        ]
    )
    _install_pxr(monkeypatch)

    assert worker.track_deduplicate_geometry(stage) == {
        "/World/Instance/Geometry": "/World/Prototype/Geometry"
    }


def test_build_correspondence_map_resolves_geometry_children_and_prototypes() -> None:
    result = worker.build_correspondence_map(
        ["/World/A", "/World/B", "/World/Prototype"],
        {"/World/A": ["/World/A_part"]},
        {
            "/World/A_part/Geometry": "/World/Prototype/Geometry",
            "/World/B/Geometry": "/World/Prototype/Geometry",
        },
        True,
        True,
    )

    assert result["full_mapping"]["original_to_prototype"] == {
        "/World/A": ["/World/Prototype/Geometry"],
        "/World/B": ["/World/Prototype/Geometry"],
        "/World/Prototype": ["/World/Prototype/Geometry"],
    }


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["so_worker.py"], "Usage: so_worker.py"),
        (["so_worker.py", "{"], "Invalid JSON"),
        (["so_worker.py", "[]"], "JSON arguments must be an object"),
        (["so_worker.py", "{}"], "Missing required JSON parameter: manifest_path"),
    ],
)
def test_main_argument_errors(
    argv: list[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        worker.main()

    assert message in capsys.readouterr().err


def test_main_writes_error_manifest_for_missing_required_params(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["so_worker.py", json.dumps({"manifest_path": str(manifest_path)})],
    )

    worker.main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert "input_usd_path" in manifest["error"]


def test_main_writes_success_manifest_for_all_operation_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "out.usd"
    stage = _Stage()
    _install_pxr(monkeypatch, stage)
    calls = _install_scene_optimizer(monkeypatch)

    mesh_snapshots = iter(
        [
            ["/World/A"],
            ["/World/A_part"],
            ["/World/A_part"],
            ["/World/A_part"],
        ]
    )
    monkeypatch.setattr(
        worker, "capture_mesh_paths", lambda *args, **kwargs: next(mesh_snapshots)
    )
    monkeypatch.setattr(
        worker,
        "track_deduplicate_geometry",
        lambda _stage: {"/World/A_part/Geometry": "/World/Prototype/Geometry"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "so_worker.py",
            json.dumps(
                {
                    "manifest_path": str(manifest_path),
                    "input_usd_path": "in.usd",
                    "output_usd_path": str(output_path),
                    "operations": [
                        ["splitMeshes", {}],
                        ["deduplicateGeometry", {}],
                        ["cleanup", {}],
                    ],
                }
            ),
        ],
    )

    worker.main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert calls == ["splitMeshes", "deduplicateGeometry", "cleanup"]
    assert manifest["stage_size_bytes"] == output_path.stat().st_size
    assert manifest["correspondence_map"]["summary"]["operations_run"] == {
        "deinstance": False,
        "split": True,
        "deduplicate": True,
    }


def test_main_writes_error_manifest_when_stage_cannot_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _install_pxr(monkeypatch, None)
    _install_scene_optimizer(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "so_worker.py",
            json.dumps(
                {
                    "manifest_path": str(manifest_path),
                    "input_usd_path": "missing.usd",
                    "output_usd_path": str(tmp_path / "out.usd"),
                    "operations": [],
                }
            ),
        ],
    )

    worker.main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert "Failed to open USD stage" in manifest["error"]


def test_main_stops_after_operation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _install_pxr(monkeypatch, _Stage())
    _install_scene_optimizer(monkeypatch, fail_operation="badOp")
    monkeypatch.setattr(
        worker, "capture_mesh_paths", lambda *args, **kwargs: ["/World/A"]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "so_worker.py",
            json.dumps(
                {
                    "manifest_path": str(manifest_path),
                    "input_usd_path": "in.usd",
                    "output_usd_path": str(tmp_path / "out.usd"),
                    "operations": [["badOp", {}], ["never", {}]],
                }
            ),
        ],
    )

    worker.main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert manifest["operations_executed"][0]["success"] is False
    assert manifest["error"] == "Operation(s) failed: badOp"


def test_main_records_export_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _install_pxr(monkeypatch, _Stage(export_ok=False))
    _install_scene_optimizer(monkeypatch)
    monkeypatch.setattr(
        worker, "capture_mesh_paths", lambda *args, **kwargs: ["/World/A"]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "so_worker.py",
            json.dumps(
                {
                    "manifest_path": str(manifest_path),
                    "input_usd_path": "in.usd",
                    "output_usd_path": str(tmp_path / "out.usd"),
                    "operations": [],
                }
            ),
        ],
    )

    worker.main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert "Failed to export USD stage" in manifest["error"]
