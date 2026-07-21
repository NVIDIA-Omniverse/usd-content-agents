# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import traceback
from pathlib import Path

import pytest

from world_understanding.agentic.events import CollectingEventListener
from world_understanding.agentic.usd_tasks import usd_loader


class _Store:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def set(self, key: str, value: object) -> None:
        self.values[key] = value


class _Layer:
    identifier = "root.usd"


class _Stage:
    def GetPseudoRoot(self) -> _Stage:
        return self

    def GetRootLayer(self) -> _Layer:
        return _Layer()


class _Prim:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def IsA(self, schema: object) -> bool:
        return self.kind == schema


class _USDModel:
    def __init__(self, path: str) -> None:
        self.path = path
        self.prims = {"root": object(), "mesh": object(), "xform": object()}
        self.collections = {"collection": object()}

    def get_all_meshes(self) -> list[str]:
        return ["mesh"]

    def get_all_xforms(self) -> list[str]:
        return ["xform"]


def _patch_stage_metadata(monkeypatch: pytest.MonkeyPatch, prims: list[_Prim]) -> None:
    monkeypatch.setattr(usd_loader.UsdGeom, "Mesh", "mesh")
    monkeypatch.setattr(usd_loader.UsdGeom, "Xform", "xform")
    monkeypatch.setattr(usd_loader.Usd, "PrimRange", lambda root: prims)
    monkeypatch.setattr(usd_loader.UsdGeom, "GetStageUpAxis", lambda stage: "Y")
    monkeypatch.setattr(usd_loader.UsdGeom, "GetStageMetersPerUnit", lambda stage: 0.01)


def test_usd_loading_task_requires_existing_path(tmp_path: Path) -> None:
    task = usd_loader.USDLoadingTask()

    with pytest.raises(ValueError, match="usd_path not found"):
        task.run({}, _Store())

    sentinel = "never-log-missing-usd-path-713"
    context = {"usd_path": tmp_path / sentinel / "missing.usd"}
    with pytest.raises(FileNotFoundError, match="^USD file not found$") as exc:
        task.run(context, _Store())

    assert sentinel not in str(exc.value)
    assert context["error"] == "USD file not found"


def test_usd_loading_task_builds_usd_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("#usda", encoding="utf-8")
    stage = _Stage()
    store = _Store()
    monkeypatch.setattr(usd_loader, "load_stage", lambda path: stage)
    monkeypatch.setattr(usd_loader, "USDModel", _USDModel)
    _patch_stage_metadata(monkeypatch, [])

    context = usd_loader.USDLoadingTask().run({"usd_path": usd_path}, store)

    assert context["stage_loaded"] is True
    assert context["usd_model_built"] is True
    assert context["prototypes_converted"] == 0
    assert context["num_prims"] == 3
    assert context["stage_info"] == {
        "total_prims": 3,
        "mesh_prims": 1,
        "xform_prims": 1,
        "root_layer": "root.usd",
        "up_axis": "Y",
        "meters_per_unit": 0.01,
    }
    assert store.values["usd_stage"] is stage
    assert isinstance(store.values["usd_model"], _USDModel)


def test_usd_loading_task_projects_root_layer_result_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "never-return-root-layer-credential-713"
    sensitive_dir = tmp_path / f"api_key={sentinel}"
    sensitive_dir.mkdir()
    usd_path = sensitive_dir / "scene.usd"
    usd_path.write_text("#usda", encoding="utf-8")

    class _CredentialLayer:
        identifier = str(usd_path)

    class _CredentialStage(_Stage):
        def GetRootLayer(self) -> _CredentialLayer:
            return _CredentialLayer()

    monkeypatch.setattr(usd_loader, "load_stage", lambda path: _CredentialStage())
    _patch_stage_metadata(monkeypatch, [])
    initial_context = {"usd_path": usd_path, "build_usd_model": False}

    context = usd_loader.USDLoadingTask().run(initial_context, _Store())

    assert context["usd_path"] == usd_path
    assert context["stage_info"]["root_layer"] == "<redacted>"
    assert sentinel not in repr(context["stage_info"])


def test_usd_loading_task_can_skip_model_and_convert_prototypes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("#usda", encoding="utf-8")
    stage = _Stage()
    store = _Store()
    monkeypatch.setattr(usd_loader, "load_stage", lambda path: stage)
    _patch_stage_metadata(monkeypatch, [_Prim("mesh"), _Prim("xform"), _Prim("xform")])

    from world_understanding.utils.usd import prim as prim_utils

    monkeypatch.setattr(
        prim_utils,
        "convert_abstract_prototypes_to_def",
        lambda stage, prototype_names=None: 2,
    )

    context = usd_loader.USDLoadingTask().run(
        {
            "usd_path": usd_path,
            "build_usd_model": False,
            "convert_prototypes_to_xforms": True,
            "prototype_names": ["ClassA"],
        },
        store,
    )

    assert context["stage_loaded"] is True
    assert context["usd_model_built"] is False
    assert context["prototypes_converted"] == 2
    assert context["num_prims"] == 3
    assert context["stage_info"]["mesh_prims"] == 1
    assert context["stage_info"]["xform_prims"] == 2
    assert "usd_model" not in store.values


def test_usd_loading_task_logs_when_no_prototypes_convert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("#usda", encoding="utf-8")
    monkeypatch.setattr(usd_loader, "load_stage", lambda path: _Stage())
    _patch_stage_metadata(monkeypatch, [])

    from world_understanding.utils.usd import prim as prim_utils

    monkeypatch.setattr(
        prim_utils,
        "convert_abstract_prototypes_to_def",
        lambda stage, prototype_names=None: 0,
    )

    context = usd_loader.USDLoadingTask().run(
        {
            "usd_path": usd_path,
            "build_usd_model": False,
            "convert_prototypes_to_xforms": True,
        },
        _Store(),
    )

    assert context["prototypes_converted"] == 0


def test_usd_loading_task_records_error_when_load_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinels = (
        "never-log-loader-config-713",
        "never-log-loader-source-713",
        "never-log-loader-output-713",
    )
    usd_dir = tmp_path / sentinels[1]
    usd_dir.mkdir()
    usd_path = usd_dir / "scene.usd"
    usd_path.write_text("#usda", encoding="utf-8")
    listener = CollectingEventListener()
    context = {
        "usd_path": usd_path,
        "config_path": tmp_path / sentinels[0] / "config.yaml",
        "output_dir_override": tmp_path / sentinels[2],
        "event_listener": listener,
    }
    loaded_paths: list[str] = []

    def fail_with_reflected_diagnostics(path: str) -> None:
        loaded_paths.append(path)
        raise RuntimeError("backend reflected " + " ".join(sentinels))

    monkeypatch.setattr(usd_loader, "load_stage", fail_with_reflected_diagnostics)

    with pytest.raises(RuntimeError, match="^USD stage loading failed$") as exc:
        usd_loader.USDLoadingTask().run(context, _Store())

    assert loaded_paths == [str(usd_path)]
    assert context["usd_path"] == usd_path
    assert context["stage_loaded"] is False
    assert context["error"] == "USD stage loading failed"
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    observable = "\n".join(
        (
            "".join(traceback.format_exception(exc.value)),
            repr(listener.logs),
            context["error"],
        )
    )
    for sentinel in sentinels:
        assert sentinel not in observable
