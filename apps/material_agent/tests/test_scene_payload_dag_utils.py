# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom

from material_agent.scene.payload_dag_utils import (
    build_dag,
    collect_arcs_from_file,
    collect_arcs_from_layer,
    compute_depths,
    rewrite_arcs_in_layer,
    topological_sort_leaves_first,
)


def _make_stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def test_collect_arcs_from_layer_and_file_include_sublayers(tmp_path: Path) -> None:
    ref_path = tmp_path / "child_ref.usda"
    payload_path = tmp_path / "child_payload.usda"
    nested_path = tmp_path / "nested_from_sublayer.usda"
    for path in (ref_path, payload_path, nested_path):
        stage = _make_stage(path)
        UsdGeom.Mesh.Define(stage, "/World/Mesh")
        stage.GetRootLayer().Save()

    sublayer_path = tmp_path / "sub.usda"
    sub_stage = _make_stage(sublayer_path)
    sub_stage.DefinePrim("/World/SubThing").GetReferences().AddReference(
        "./nested_from_sublayer.usda"
    )
    sub_stage.GetRootLayer().Save()

    parent_path = tmp_path / "parent.usda"
    stage = _make_stage(parent_path)
    prim = stage.DefinePrim("/World/Thing")
    prim.GetReferences().AddReference("./child_ref.usda")
    prim.GetPayloads().AddPayload("./child_payload.usda")
    stage.GetRootLayer().subLayerPaths.append("./sub.usda")
    stage.GetRootLayer().Save()

    layer_targets = collect_arcs_from_layer(stage.GetRootLayer())
    file_targets = collect_arcs_from_file(str(parent_path))

    assert layer_targets == {str(ref_path.resolve()), str(payload_path.resolve())}
    assert file_targets == {
        str(ref_path.resolve()),
        str(payload_path.resolve()),
        str(nested_path.resolve()),
    }


def test_build_dag_compute_depths_and_topological_sort(
    monkeypatch,
) -> None:
    graph = {
        "/root.usd": {"/child_a.usd", "/child_b.usd"},
        "/child_a.usd": {"/leaf.usd"},
        "/child_b.usd": {"/leaf.usd"},
        "/leaf.usd": set(),
    }

    monkeypatch.setattr(
        "material_agent.scene.payload_dag_utils.collect_arcs_from_file",
        lambda file_path: graph[file_path],
    )

    adj = build_dag({"/root.usd"})
    depths = compute_depths(adj)
    order = topological_sort_leaves_first(adj)

    assert adj == graph
    assert depths["/leaf.usd"] == 0
    assert depths["/child_a.usd"] == 1
    assert depths["/child_b.usd"] == 1
    assert depths["/root.usd"] == 2
    assert order[0] == "/leaf.usd"
    assert order[-1] == "/root.usd"

    cycle_order = topological_sort_leaves_first(
        {"/a.usd": {"/b.usd"}, "/b.usd": {"/a.usd"}}
    )
    assert set(cycle_order) == {"/a.usd", "/b.usd"}


def test_rewrite_arcs_in_layer_handles_child_map_and_moved_layer(
    tmp_path: Path,
) -> None:
    original_dir = tmp_path / "original"
    copied_dir = tmp_path / "copied"
    original_dir.mkdir()
    copied_dir.mkdir()

    child_path = original_dir / "child.usda"
    other_path = original_dir / "other.usda"
    new_child_path = copied_dir / "new_child.usda"
    for path in (child_path, other_path, new_child_path):
        stage = _make_stage(path)
        UsdGeom.Mesh.Define(stage, "/World/Mesh")
        stage.GetRootLayer().Save()

    parent_path = original_dir / "parent.usda"
    stage = _make_stage(parent_path)
    prim = stage.DefinePrim("/World/Thing")
    prim.GetReferences().AddReference("./child.usda")
    prim.GetPayloads().AddPayload("./other.usda")
    stage.GetRootLayer().Save()

    copied_parent = copied_dir / "parent.usda"
    shutil.copy(parent_path, copied_parent)
    layer = stage.GetRootLayer().FindOrOpen(str(copied_parent))
    assert layer is not None

    rewritten = rewrite_arcs_in_layer(
        layer,
        {str(child_path.resolve()): str(new_child_path.resolve())},
        resolve_from=parent_path,
    )

    spec = layer.GetPrimAtPath("/World/Thing")
    ref_items = (
        list(spec.referenceList.prependedItems)
        + list(spec.referenceList.appendedItems)
        + list(spec.referenceList.explicitItems)
    )
    payload_items = (
        list(spec.payloadList.prependedItems)
        + list(spec.payloadList.appendedItems)
        + list(spec.payloadList.explicitItems)
    )

    assert rewritten == 1
    assert {item.assetPath for item in ref_items} == {"new_child.usda"}
    assert payload_items
    assert payload_items[0].assetPath.endswith("../original/other.usda")


def test_collect_arcs_from_file_handles_missing_and_unopenable_layers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    assert collect_arcs_from_file(str(tmp_path / "missing.usda")) == set()

    existing = tmp_path / "existing.usda"
    existing.write_text("#usda 1.0\n")
    monkeypatch.setattr(Sdf.Layer, "FindOrOpen", staticmethod(lambda _path: None))
    assert collect_arcs_from_file(str(existing)) == set()

    def raise_open(_path: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(Sdf.Layer, "FindOrOpen", staticmethod(raise_open))
    assert collect_arcs_from_file(str(existing)) == set()


def test_collect_arcs_from_file_skips_bad_sublayers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sublayer = tmp_path / "sub.usda"
    sub_stage = _make_stage(sublayer)
    sub_stage.GetRootLayer().Save()
    parent = tmp_path / "parent.usda"
    stage = _make_stage(parent)
    stage.GetRootLayer().subLayerPaths.append("./sub.usda")
    stage.GetRootLayer().Save()

    original_find_or_open = Sdf.Layer.FindOrOpen

    def flaky_open(path: str):
        if Path(path).resolve() == sublayer.resolve():
            raise RuntimeError("bad sublayer")
        return original_find_or_open(path)

    monkeypatch.setattr(Sdf.Layer, "FindOrOpen", staticmethod(flaky_open))

    assert collect_arcs_from_file(str(parent)) == set()


def test_compute_depths_cycle_guard() -> None:
    assert compute_depths({"/a.usd": {"/a.usd"}}) == {"/a.usd": 1}


def test_rewrite_arcs_in_layer_handles_unopenable_resolve_source_and_empty_arcs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    layer_path = tmp_path / "layer.usda"
    stage = _make_stage(layer_path)
    prim = stage.DefinePrim("/World/Thing")
    prim.GetReferences().AddReference("")
    stage.GetRootLayer().Save()
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    assert layer is not None

    original_find_or_open = Sdf.Layer.FindOrOpen

    def fake_open(path: str):
        if path == str(tmp_path / "resolve-source.usda"):
            return None
        return original_find_or_open(path)

    monkeypatch.setattr(Sdf.Layer, "FindOrOpen", staticmethod(fake_open))

    assert rewrite_arcs_in_layer(layer, {}) == 0
    assert (
        rewrite_arcs_in_layer(
            layer,
            {},
            resolve_from=tmp_path / "resolve-source.usda",
        )
        == 0
    )

    spec = layer.GetPrimAtPath("/World/Thing")
    assert spec is not None
    ref_items = (
        list(spec.referenceList.prependedItems)
        + list(spec.referenceList.appendedItems)
        + list(spec.referenceList.explicitItems)
        + list(spec.referenceList.deletedItems)
        + list(spec.referenceList.orderedItems)
    )
    assert any(item.assetPath == "" for item in ref_items)
