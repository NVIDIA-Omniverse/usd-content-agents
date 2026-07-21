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

from world_understanding.functions.graphics import so_uv_worker


class FakeMeshSchema:
    pass


class FakeAttr:
    def __init__(self, authored: bool) -> None:
        self._authored = authored

    def HasAuthoredValue(self) -> bool:
        return self._authored


class FakePrim:
    def __init__(self, *, pseudo: bool = False, mesh: bool = False, uv: bool = False):
        self._pseudo = pseudo
        self._mesh = mesh
        self._uv = uv

    def IsPseudoRoot(self) -> bool:
        return self._pseudo

    def IsA(self, schema: Any) -> bool:
        return schema is FakeMeshSchema and self._mesh

    def GetAttribute(self, name: str) -> FakeAttr | None:
        if name == "primvars:st" and self._uv:
            return FakeAttr(True)
        return None


class FakeRootLayer:
    def Export(self, output_path: str) -> None:
        Path(output_path).write_text("#usda 1.0\n", encoding="utf-8")


class FakeStage:
    def GetPseudoRoot(self) -> str:
        return "root"

    def GetRootLayer(self) -> FakeRootLayer:
        return FakeRootLayer()


class FakeExecutionContext:
    def __init__(self) -> None:
        self.stage = None
        self.removed = False

    def set_stage(self, stage: Any) -> None:
        self.stage = stage

    def remove_stage(self) -> None:
        self.removed = True


class FakeSceneOptimizerCore:
    calls: list[tuple[str, Any, dict[str, Any]]] = []

    @classmethod
    def getInstance(cls) -> FakeSceneOptimizerCore:
        return cls()

    def executeOperation(
        self, operation: str, ctx: FakeExecutionContext, op_params: dict[str, Any]
    ) -> None:
        self.calls.append((operation, ctx.stage, op_params))


def _install_fake_scene_optimizer(monkeypatch: pytest.MonkeyPatch, stage: Any) -> None:
    omni = types.ModuleType("omni")
    scene = types.ModuleType("omni.scene")
    optimizer = types.ModuleType("omni.scene.optimizer")
    core = types.ModuleType("omni.scene.optimizer.core")
    core.ExecutionContext = FakeExecutionContext
    core.SceneOptimizerCore = FakeSceneOptimizerCore

    pxr = types.ModuleType("pxr")
    usd = types.SimpleNamespace(
        Stage=types.SimpleNamespace(Open=lambda _path: stage),
        PrimRange=lambda _root: [
            FakePrim(pseudo=True),
            FakePrim(mesh=True, uv=False),
            FakePrim(mesh=True, uv=True),
            FakePrim(mesh=False),
        ],
    )
    usd_geom = types.SimpleNamespace(Mesh=FakeMeshSchema)
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


def test_so_uv_worker_success_and_error_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_scene_optimizer(monkeypatch, FakeStage())
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "out.usda"
    params = {
        "input_usd_path": str(tmp_path / "in.usda"),
        "output_usd_path": str(output),
        "operation": "generateAtlasUVs",
        "op_params": {"resolution": 128},
        "manifest_path": str(manifest),
    }
    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", json.dumps(params)])
    so_uv_worker.main()

    payload = json.loads(manifest.read_text())
    assert payload["status"] == "success"
    assert payload["mesh_count"] == 2
    assert payload["meshes_with_uvs"] == 1
    assert payload["stage_size_bytes"] > 0
    assert FakeSceneOptimizerCore.calls[-1][0] == "generateAtlasUVs"

    _install_fake_scene_optimizer(monkeypatch, None)
    manifest = tmp_path / "error-manifest.json"
    params["manifest_path"] = str(manifest)
    monkeypatch.setattr(sys, "argv", ["so_uv_worker.py", json.dumps(params)])
    so_uv_worker.main()

    payload = json.loads(manifest.read_text())
    assert payload["status"] == "error"
    assert "Failed to open USD stage" in payload["error"]


def test_so_uv_worker_argument_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
