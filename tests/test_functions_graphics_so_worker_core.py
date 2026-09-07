# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import runpy
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
        self,
        prims: list[_Prim] | None = None,
        *,
        export_ok: bool = True,
        layered: bool = False,
    ) -> None:
        self.prims = prims or []
        self.export_ok = export_ok
        self.removed = False
        self.layered = layered
        self.flattened = False
        self._session = object()
        self._extra = object()

    def GetPseudoRoot(self) -> _Stage:
        return self

    def GetRootLayer(self) -> _Stage:
        return self

    def GetSessionLayer(self) -> object:
        return self._session

    def GetUsedLayers(self) -> list[object]:
        # A single-layer stage: the worker exports the root layer directly and
        # keeps authored structure. Include the session layer exactly as USD
        # does so the production filter is exercised. ``layered=True`` adds a
        # second composed layer and takes the flatten branch.
        layers = [self, self._session]
        if self.layered:
            layers.append(self._extra)
        return layers

    def Flatten(self) -> _Stage:
        self.flattened = True
        return self

    def Export(self, path: str) -> bool:
        Path(path).write_text("usd", encoding="utf-8")
        return self.export_ok


@pytest.mark.parametrize(
    ("worker_filename", "expected_exports"),
    [
        (
            "so_worker.py",
            {
                "_export_layer_for",
                "_normalize_dependency_roots",
                "export_stage_portably",
            },
        ),
        (
            "so_uv_worker.py",
            {"_normalize_dependency_roots", "export_stage_portably"},
        ),
    ],
)
def test_worker_script_imports_shared_export_without_package(
    monkeypatch: pytest.MonkeyPatch,
    worker_filename: str,
    expected_exports: set[str],
) -> None:
    """Copied ABI-isolated workers must import their sibling helper as scripts."""
    standalone_export = types.ModuleType("so_export")
    export_layer_for = object()
    export_stage_portably = object()
    normalize_dependency_roots = object()
    standalone_export.export_layer_for = export_layer_for
    standalone_export.export_stage_portably = export_stage_portably
    standalone_export._normalize_dependency_roots = normalize_dependency_roots
    monkeypatch.setitem(sys.modules, "so_export", standalone_export)

    namespace = runpy.run_path(
        str(Path(worker.__file__).with_name(worker_filename)),
        run_name="abi_isolated_worker",
    )

    if "_export_layer_for" in expected_exports:
        assert namespace["_export_layer_for"] is export_layer_for
    assert namespace["_normalize_dependency_roots"] is normalize_dependency_roots
    assert namespace["export_stage_portably"] is export_stage_portably


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
    monkeypatch.setattr(
        worker,
        "export_stage_portably",
        lambda stage, path, **_kwargs: worker.export_layer_for(stage).Export(path),
    )


def _install_scene_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_operation: str | None = None,
    operation_results: dict[str, object] | None = None,
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
        ) -> object:
            calls.append(op_name)
            if op_name == fail_operation:
                raise RuntimeError("operation failed")
            return (operation_results or {}).get(op_name, (True, None, None))

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


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            (),
            (
                False,
                "Scene Optimizer executeOperation returned an invalid tuple of length 0; "
                "expected 3",
                None,
            ),
        ),
        (
            (1, None, {"output": True}),
            (
                False,
                "Scene Optimizer executeOperation returned a non-boolean success value: 1",
                {"output": True},
            ),
        ),
        (
            (True, 7, None),
            (
                False,
                "Scene Optimizer executeOperation returned a non-string error value: 7",
                None,
            ),
        ),
        (
            (False, None, None),
            (False, "Scene Optimizer operation reported failure", None),
        ),
        (True, (True, None, None)),
        (False, (False, "Scene Optimizer operation reported failure", None)),
        (
            "unsupported",
            (
                False,
                "Scene Optimizer executeOperation returned an unsupported result of type str",
                None,
            ),
        ),
    ],
)
def test_parse_operation_result_contracts(
    result: object,
    expected: tuple[bool, str | None, object | None],
) -> None:
    assert worker._parse_operation_result(result) == expected


def test_operation_output_accepts_none() -> None:
    assert worker._json_safe_operation_output(None) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_operation_output_rejects_non_finite_json_numbers(value: float) -> None:
    assert worker._json_safe_operation_output(value) == {
        "serializable": False,
        "type": "float",
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
        ["/World/A", "/World/B", "/World/C", "/World/Prototype"],
        {"/World/A": ["/World/A_part"]},
        {
            "/World/A_part/Geometry": "/World/Prototype/Geometry",
            "/World/B/Geometry": "/World/Prototype/Geometry",
            "/World/C": "/World/Prototype/Geometry",
        },
        True,
        True,
    )

    assert result["full_mapping"]["original_to_prototype"] == {
        "/World/A": ["/World/Prototype/Geometry"],
        "/World/B": ["/World/Prototype/Geometry"],
        "/World/C": ["/World/Prototype/Geometry"],
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


def test_main_rejects_invalid_dependency_roots_before_scene_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    calls = _install_scene_optimizer(monkeypatch)
    _install_pxr(monkeypatch, _Stage())
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
                    "approved_dependency_roots": [str(Path(tmp_path.anchor))],
                    "operations": [["cleanup", {}]],
                }
            ),
        ],
    )

    worker.main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert "must not contain filesystem roots" in manifest["error"]
    assert calls == []


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
                    "approved_dependency_roots": [str(tmp_path)],
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
                    "approved_dependency_roots": [str(tmp_path)],
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
                    "approved_dependency_roots": [str(tmp_path)],
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


def test_main_stops_on_native_operation_failure_and_records_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "out.usd"
    _install_pxr(monkeypatch, _Stage())
    calls = _install_scene_optimizer(
        monkeypatch,
        operation_results={
            "badOp": (
                False,
                "native operation failed",
                {"failed_prim": "/World/A"},
            )
        },
    )
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
                    "output_usd_path": str(output_path),
                    "approved_dependency_roots": [str(tmp_path)],
                    "operations": [["badOp", {}], ["never", {}]],
                }
            ),
        ],
    )

    worker.main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operation = manifest["operations_executed"][0]
    assert manifest["status"] == "error"
    assert calls == ["badOp"]
    assert operation["success"] is False
    assert operation["error"] == "native operation failed"
    assert operation["output"] == {"failed_prim": "/World/A"}
    assert not output_path.exists()


def test_main_records_native_operation_success_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "out.usd"
    _install_pxr(monkeypatch, _Stage())
    calls = _install_scene_optimizer(
        monkeypatch,
        operation_results={"cleanup": (True, None, {"processed_meshes": 1})},
    )
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
                    "output_usd_path": str(output_path),
                    "approved_dependency_roots": [str(tmp_path)],
                    "operations": [["cleanup", {}]],
                }
            ),
        ],
    )

    worker.main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operation = manifest["operations_executed"][0]
    assert manifest["status"] == "success"
    assert calls == ["cleanup"]
    assert operation["success"] is True
    assert operation["output"] == {"processed_meshes": 1}
    assert output_path.exists()


def test_main_accepts_legacy_void_scene_optimizer_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "out.usd"
    _install_pxr(monkeypatch, _Stage())
    _install_scene_optimizer(
        monkeypatch,
        operation_results={"cleanup": None},
    )
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
                    "output_usd_path": str(output_path),
                    "approved_dependency_roots": [str(tmp_path)],
                    "operations": [["cleanup", {}]],
                }
            ),
        ],
    )

    worker.main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operation = manifest["operations_executed"][0]
    assert manifest["status"] == "success"
    assert operation["success"] is True
    assert operation["result_contract"] == "legacy_void"
    assert output_path.exists()


def test_main_records_opaque_native_output_without_breaking_failure_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "out.usd"
    _install_pxr(monkeypatch, _Stage())
    _install_scene_optimizer(
        monkeypatch,
        operation_results={"badOp": (False, "native failure", object())},
    )
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
                    "output_usd_path": str(output_path),
                    "approved_dependency_roots": [str(tmp_path)],
                    "operations": [["badOp", {}]],
                }
            ),
        ],
    )

    worker.main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operation = manifest["operations_executed"][0]
    assert manifest["status"] == "error"
    assert operation["success"] is False
    assert operation["output"] == {
        "serializable": False,
        "type": "object",
    }
    assert not output_path.exists()


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
                    "approved_dependency_roots": [str(tmp_path)],
                    "operations": [],
                }
            ),
        ],
    )

    worker.main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert "Failed to export USD stage" in manifest["error"]


def _run_worker_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: _Stage, tag: str
) -> dict[str, object]:
    """Drive ``worker.main`` through a single cleanup op against ``stage``."""
    manifest_path = tmp_path / f"manifest_{tag}.json"
    output_path = tmp_path / f"out_{tag}.usd"
    _install_pxr(monkeypatch, stage)
    _install_scene_optimizer(monkeypatch)
    monkeypatch.setattr(worker, "capture_mesh_paths", lambda *a, **k: ["/World/A"])
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
                    "approved_dependency_roots": [str(tmp_path)],
                    "operations": [["cleanup", {}]],
                }
            ),
        ],
    )
    worker.main()
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_main_flattens_only_a_layered_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flatten a layered stage on export, but leave a single-layer stage alone.

    Exporting only the root layer drops relative sublayer arcs when the output
    lands outside the source directory, silently emptying a layered asset.
    Flattening collapses variant sets to their active selections, so it must not
    apply to a stage that does not need it. See issue #963.
    """
    single = _Stage([_Prim("/World/A")])
    assert _run_worker_main(tmp_path, monkeypatch, single, "single")["status"] == (
        "success"
    )
    assert single.flattened is False, "a single-layer stage must not be flattened"

    layered = _Stage([_Prim("/World/A")], layered=True)
    assert _run_worker_main(tmp_path, monkeypatch, layered, "layered")["status"] == (
        "success"
    )
    assert layered.flattened is True, "a layered stage must be flattened"
