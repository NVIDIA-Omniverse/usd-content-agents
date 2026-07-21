# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from material_agent.scene.extract import (
    _collect_instanceable_prims,
    _find_instances_recursive,
    _remove_prim_recursive,
    _sanitize_name,
    _strip_instance_children,
    _unique_safe_names,
    extract_all,
    extract_sub_asset,
)
from material_agent.scene.manifest import PayloadGroup, SceneManifest, SubAsset


class _FakeReferenceList:
    def __init__(self, prim_path: str | None = None) -> None:
        item = SimpleNamespace(primPath=prim_path) if prim_path else None
        self.prependedItems = [item] if item else []


class _FakeSpec:
    def __init__(
        self,
        path: str,
        *,
        instanceable: bool = False,
        prim_path: str | None = None,
        children: dict[str, _FakeSpec] | None = None,
    ) -> None:
        self.path = path
        self._instanceable = instanceable
        self.referenceList = _FakeReferenceList(prim_path)
        self.nameChildren = children or {}
        self.removed: list[_FakeSpec] = []

    def HasInfo(self, key: str) -> bool:
        return key == "instanceable"

    def GetInfo(self, key: str):
        return self._instanceable if key == "instanceable" else None

    def RemoveNameChild(self, spec: _FakeSpec) -> None:
        self.removed.append(spec)


class _FakeLayer:
    def __init__(self, specs: dict[str, _FakeSpec]) -> None:
        self._specs = specs
        self.pseudoRoot = _FakeSpec("/")

    def GetPrimAtPath(self, path):
        return self._specs.get(str(path))


class _FakePrim:
    def __init__(
        self, path: str, is_instance: bool, children: list[_FakePrim] | None = None
    ) -> None:
        self._path = path
        self._is_instance = is_instance
        self._children = children or []

    def IsInstance(self) -> bool:
        return self._is_instance

    def GetPath(self):
        return self._path

    def GetAllChildren(self):
        return list(self._children)


def test_sanitize_names_and_extract_all_dispatch(monkeypatch, tmp_path: Path) -> None:
    first = SubAsset(id="obj_1", name="Chair / Large", prim_path="/World/ChairA")
    second = SubAsset(id="obj_2", name="Chair / Large", prim_path="/World/ChairB")
    third = SubAsset(id="obj_3", name="Lamp", prim_path="/World/Lamp", status="skipped")
    manifest = SceneManifest(
        sub_assets=[first, second, third],
        payload_groups=[
            PayloadGroup(
                id="payload_a",
                group_name="payload_a",
                payload_file="/tmp/payload_a.usd",
                instance_count=1,
                instance_paths=["/World/ChairA"],
            ),
            PayloadGroup(
                id="payload_b",
                group_name="payload_b",
                payload_file="/tmp/payload_b.usd",
                status="skipped",
            ),
        ],
    )

    calls: list[dict[str, object]] = []

    def fake_extract_sub_asset(**kwargs):
        calls.append(kwargs)
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_text("usd", encoding="utf-8")
        return kwargs["output_path"]

    monkeypatch.setattr(
        "material_agent.scene.extract.extract_sub_asset", fake_extract_sub_asset
    )

    updated = extract_all(
        tmp_path / "scene.usda",
        manifest,
        tmp_path / "out",
        names_filter=["Chair / Large"],
        max_workers=1,
    )

    safe_names = _unique_safe_names([first, second])
    assert _sanitize_name("Chair / Large") == "chair_large"
    assert safe_names == {
        "obj_1": "chair_large_obj_1",
        "obj_2": "chair_large_obj_2",
    }
    assert len(calls) == 2
    assert calls[0]["skip_instance_subtrees"] is True
    assert calls[1]["skip_instance_subtrees"] is False
    assert updated.sub_assets[0].status == "extracted"
    assert updated.sub_assets[1].status == "extracted"
    assert updated.sub_assets[2].status == "skipped"


def test_collect_instanceable_prims_and_recursive_helpers() -> None:
    inst_spec = _FakeSpec(
        "/World/Root/Inst",
        instanceable=True,
        prim_path="/Prototype",
    )
    root_spec = _FakeSpec(
        "/World/Root",
        children={
            "Inst": inst_spec,
            "Ignored": _FakeSpec("/World/Root/Ignored"),
        },
    )
    layer = _FakeLayer(
        {
            "/World/Root": root_spec,
            "/World/Root/Inst": inst_spec,
            "/World/Root/Ignored": root_spec.nameChildren["Ignored"],
        }
    )

    instance_paths: list[str] = []
    mask_paths = ["/World/Root"]
    _collect_instanceable_prims(layer, "/World/Root", instance_paths, mask_paths)

    assert instance_paths == ["/World/Root/Inst"]
    assert mask_paths == ["/World/Root", "/Prototype"]

    _collect_instanceable_prims(layer, "/World/Missing", instance_paths, mask_paths)
    assert instance_paths == ["/World/Root/Inst"]

    nested_prim = _FakePrim(
        "/World/Root/Nested", False, [_FakePrim("/World/Root/Nested/Leaf", True)]
    )
    root_prim = _FakePrim("/World/Root", False, [nested_prim])
    found: list[str] = []
    _find_instances_recursive(root_prim, found)
    assert found == ["/World/Root/Nested/Leaf"]


def test_remove_prim_recursive_removes_children_before_parent() -> None:
    class FakePath:
        def __init__(self, value: str) -> None:
            self.value = value

        def GetParentPath(self) -> str:
            return "/World/Root"

        def AppendChild(self, name: str) -> FakePath:
            return FakePath(f"{self.value}/{name}")

        def __str__(self) -> str:
            return self.value

    child_spec = _FakeSpec("/World/Root/Child")
    root_spec = _FakeSpec("/World/Root", children={"Child": child_spec})
    layer = _FakeLayer(
        {
            "/World/Root": root_spec,
            "/World/Root/Child": child_spec,
        }
    )

    _remove_prim_recursive(layer, FakePath("/World/Root/Child"))

    assert root_spec.removed == [child_spec]


def test_remove_prim_recursive_handles_missing_and_pseudo_root_parent() -> None:
    class FakePath:
        def __init__(self, value: str, parent: str) -> None:
            self.value = value
            self._parent = parent

        def GetParentPath(self) -> str:
            return self._parent

        def AppendChild(self, name: str) -> FakePath:
            return FakePath(f"{self.value}/{name}", self.value)

        def __str__(self) -> str:
            return self.value

    root_child = _FakeSpec("/RootChild")
    layer = _FakeLayer({"/RootChild": root_child})

    _remove_prim_recursive(layer, FakePath("/Missing", "/"))
    _remove_prim_recursive(layer, FakePath("/RootChild", "/"))

    assert layer.pseudoRoot.removed == [root_child]


def test_strip_instance_children_edge_paths(monkeypatch) -> None:
    class FakePath:
        def __init__(self, value: str) -> None:
            self.value = value

        def AppendChild(self, name: str) -> FakePath:
            return FakePath(f"{self.value}/{name}")

        def GetParentPath(self) -> str:
            return self.value.rsplit("/", 1)[0] or "/"

        def __str__(self) -> str:
            return self.value

    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(Sdf=SimpleNamespace(Path=FakePath)),
    )

    class FakeStage:
        def __init__(self, root) -> None:
            self.root = root

        def GetPrimAtPath(self, path: str):
            return self.root

    layer = _FakeLayer({})
    _strip_instance_children(layer, FakeStage(None), "/World/Root")

    no_instance_root = _FakePrim(
        "/World/Root", False, [_FakePrim("/World/Root/A", False)]
    )
    _strip_instance_children(layer, FakeStage(no_instance_root), "/World/Root")

    inst_root = _FakePrim("/World/Root", False, [_FakePrim("/World/Root/Inst", True)])
    _strip_instance_children(layer, FakeStage(inst_root), "/World/Root")

    grand_spec = _FakeSpec("/World/Root/Inst/Child/Grand")
    child_spec = _FakeSpec(
        "/World/Root/Inst/Child",
        children={"Grand": grand_spec},
    )
    inst_spec = _FakeSpec(
        "/World/Root/Inst",
        children={"Child": child_spec},
    )
    parent_spec = _FakeSpec("/World/Root/Inst", children={"Child": child_spec})

    class FakeLayer(_FakeLayer):
        def __init__(self) -> None:
            super().__init__(
                {
                    "/World/Root/Inst": inst_spec,
                    "/World/Root/Inst/Child": child_spec,
                    "/World/Root/Inst/Child/Grand": grand_spec,
                }
            )
            self.removed_if_inert: list[str] = []

        def RemovePrimIfInert(self, path) -> None:
            self.removed_if_inert.append(str(path))

        def GetPrimAtPath(self, path):
            if str(path) == "/World/Root/Inst/Child":
                return child_spec
            if str(path) == "/World/Root/Inst/Child/Grand":
                return grand_spec
            if str(path) == "/World/Root/Inst":
                return parent_spec
            return None

    flat_layer = FakeLayer()
    _strip_instance_children(flat_layer, FakeStage(inst_root), "/World/Root")

    assert flat_layer.removed_if_inert == ["/World/Root/Inst/Child"]
    assert parent_spec.removed == [child_spec]


def test_extract_sub_asset_flatten_and_nonflatten_paths(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeOver:
        def __init__(self) -> None:
            self.info: dict[str, object] = {}

        def SetInfo(self, key: str, value: object) -> None:
            self.info[key] = value

    class FakeSessionLayer:
        def __init__(self) -> None:
            self.cleared = False

        def Clear(self) -> None:
            self.cleared = True

    class FakeLayer:
        def __init__(self) -> None:
            self.exported: list[str] = []

        def Export(self, path: str) -> None:
            self.exported.append(path)

    class FakeMaskedPrim:
        def __init__(self) -> None:
            self.instanceable = True

        def IsInstance(self) -> bool:
            return self.instanceable

        def SetInstanceable(self, value: bool) -> None:
            self.instanceable = value

    class FakeStage:
        def __init__(self) -> None:
            self.session = FakeSessionLayer()
            self.root = FakeLayer()
            self.flat = FakeLayer()
            self.prim = FakeMaskedPrim()

        def GetSessionLayer(self):
            return self.session

        def Flatten(self):
            return self.flat

        def GetRootLayer(self):
            return self.root

        def GetPrimAtPath(self, prim_path: str):
            return self.prim

    stage = FakeStage()
    created_overs: dict[str, FakeOver] = {}
    strip_calls: list[tuple[object, object, str]] = []
    open_masked_calls: list[tuple[object, object, object]] = []

    fake_usd = SimpleNamespace()
    fake_sdf = SimpleNamespace()

    class FakeUsdStage:
        LoadAll = object()

        @staticmethod
        def OpenMasked(root_or_path, mask, load=None):
            open_masked_calls.append((root_or_path, mask, load))
            return stage

    fake_usd.Stage = FakeUsdStage
    fake_usd.StagePopulationMask = lambda paths: tuple(paths)
    fake_sdf.Layer = SimpleNamespace(FindOrOpen=lambda path: object())
    fake_sdf.CreatePrimInLayer = lambda session, prim_path: created_overs.setdefault(
        prim_path, FakeOver()
    )

    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(Usd=fake_usd, Sdf=fake_sdf))
    monkeypatch.setattr(
        "material_agent.scene.extract._collect_instanceable_prims",
        lambda layer, root_path, instance_paths, mask_paths: (
            instance_paths.append("/World/Asset/Instance"),
            mask_paths.append("/Prototype"),
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.extract._strip_instance_children",
        lambda flat_layer, original_stage, prim_path: strip_calls.append(
            (flat_layer, original_stage, prim_path)
        ),
    )

    out_path = tmp_path / "out" / "asset.usd"
    extracted = extract_sub_asset(
        scene_usd_path=tmp_path / "scene.usda",
        prim_path="/World/Asset",
        output_path=out_path,
        flatten=True,
        skip_instance_subtrees=True,
    )

    assert extracted == out_path
    assert open_masked_calls[0][1] == ("/World/Asset", "/Prototype")
    assert created_overs["/World/Asset/Instance"].info["instanceable"] is False
    assert strip_calls == [(stage.flat, stage, "/World/Asset")]
    assert stage.flat.exported == [str(out_path)]
    assert stage.session.cleared is True

    stage.root.exported.clear()
    stage.flat.exported.clear()
    strip_calls.clear()
    created_overs.clear()
    monkeypatch.setattr(
        "material_agent.scene.extract._collect_instanceable_prims",
        lambda layer, root_path, instance_paths, mask_paths: None,
    )

    second_out = tmp_path / "out" / "asset_unflattened.usd"
    extract_sub_asset(
        scene_usd_path=tmp_path / "scene.usda",
        prim_path="/World/Asset",
        output_path=second_out,
        flatten=False,
    )

    assert stage.root.exported == [str(second_out)]
    assert stage.flat.exported == []
    assert strip_calls == []


def test_extract_all_marks_failures_and_uses_thread_pool(
    monkeypatch, tmp_path: Path
) -> None:
    failed = SubAsset(id="fail", name="Fail", prim_path="/World/Fail")
    manifest = SceneManifest(sub_assets=[failed])
    monkeypatch.setattr(
        "material_agent.scene.extract.extract_sub_asset",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    extract_all(tmp_path / "scene.usda", manifest, tmp_path / "out")

    assert failed.status == "failed"

    first = SubAsset(id="a", name="A", prim_path="/World/A")
    second = SubAsset(id="b", name="B", prim_path="/World/B")
    threaded_manifest = SceneManifest(sub_assets=[first, second])
    calls: list[str] = []

    def fake_extract_sub_asset(**kwargs):
        calls.append(kwargs["prim_path"])
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_text("usd", encoding="utf-8")
        return kwargs["output_path"]

    monkeypatch.setattr(
        "material_agent.scene.extract.extract_sub_asset",
        fake_extract_sub_asset,
    )

    extract_all(
        tmp_path / "scene.usda",
        threaded_manifest,
        tmp_path / "threaded",
        max_workers=2,
    )

    assert sorted(calls) == ["/World/A", "/World/B"]
    assert [sa.status for sa in threaded_manifest.sub_assets] == [
        "extracted",
        "extracted",
    ]
