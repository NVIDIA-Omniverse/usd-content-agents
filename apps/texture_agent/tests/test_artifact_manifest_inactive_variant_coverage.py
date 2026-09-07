# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from texture_agent.functions import artifact_manifest


class _PrimPath:
    def StripAllVariantSelections(self) -> str:
        return "/Root/Looks/Paint/VariantShader"


def test_inactive_variant_texture_references_filters_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packaged_texture = "materials.usdz[textures/paint.png]"
    shader_spec = SimpleNamespace(
        typeName="Shader",
        path=_PrimPath(),
        properties=[
            SimpleNamespace(name="inputs:empty_texture", default=None),
            SimpleNamespace(name="inputs:metadata_texture", default="notes.txt"),
            SimpleNamespace(
                name="inputs:packaged_texture",
                default=packaged_texture,
            ),
            SimpleNamespace(
                name="inputs:packaged_texture",
                default=packaged_texture,
            ),
            SimpleNamespace(name="inputs:broken_texture", default="broken.png"),
        ],
    )
    layer = SimpleNamespace(
        identifier="variants.usda",
        pseudoRoot=SimpleNamespace(nameChildren=[object()]),
    )
    stage = SimpleNamespace(GetUsedLayers=lambda: [layer])

    monkeypatch.setattr(
        artifact_manifest,
        "_variant_prim_specs",
        lambda _root_prim: iter([shader_spec]),
    )

    def _resolve(_layer: object, value: str) -> str:
        if value == "broken.png":
            raise RuntimeError("unresolvable variant texture")
        return f"/resolved/{value}"

    monkeypatch.setattr(
        artifact_manifest,
        "Sdf",
        SimpleNamespace(ComputeAssetPathRelativeToLayer=_resolve),
    )

    assert artifact_manifest._inactive_variant_texture_references(stage, []) == [
        {
            "prim_path": "/Root/Looks/Paint/VariantShader",
            "attribute": "inputs:packaged_texture",
            "value_type": "string",
            "path": packaged_texture,
            "resolved_path": f"/resolved/{packaged_texture}",
        },
        {
            "prim_path": "/Root/Looks/Paint/VariantShader",
            "attribute": "inputs:broken_texture",
            "value_type": "string",
            "path": "broken.png",
            "resolved_path": "",
        },
    ]


def test_inactive_variant_texture_references_inspects_shader_over(
    tmp_path: Path,
) -> None:
    """An inactive over inherits its Shader type outside the variant."""
    from pxr import Sdf, Usd, UsdShade

    stage_path = tmp_path / "variant_shader_over.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    stage.DefinePrim("/Root", "Xform")
    material = UsdShade.Material.Define(stage, "/Root/Looks/Paint")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Paint/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    variants = material.GetPrim().GetVariantSets().AddVariantSet("finish")
    variants.AddVariant("clean")
    variants.AddVariant("weathered")
    variants.SetVariantSelection("weathered")
    with variants.GetVariantEditContext():
        shader_over = stage.OverridePrim("/Root/Looks/Paint/Surface")
        shader_over.CreateAttribute(
            "inputs:base_color_texture",
            Sdf.ValueTypeNames.String,
        ).Set("missing-weathered.png")
    variants.SetVariantSelection("clean")
    stage.GetRootLayer().Save()

    refs = artifact_manifest._inactive_variant_texture_references(stage, [])

    # A missing asset resolves to its authored spelling, so the reference is
    # anchored to the authoring layer rather than the process directory.
    assert refs == [
        {
            "prim_path": "/Root/Looks/Paint/Surface",
            "attribute": "inputs:base_color_texture",
            "value_type": "string",
            "path": "missing-weathered.png",
            "resolved_path": str(tmp_path / "missing-weathered.png"),
        }
    ]


def test_all_prim_specs_descends_into_variant_branches(tmp_path: Path) -> None:
    """Specs nested inside a variant branch are yielded."""
    from pxr import Sdf, Usd, UsdShade

    stage_path = tmp_path / "variants.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    root = stage.DefinePrim("/Root", "Xform")
    stage.SetDefaultPrim(root)
    variants = root.GetVariantSets().AddVariantSet("look")
    variants.AddVariant("only")
    variants.SetVariantSelection("only")
    with variants.GetVariantEditContext():
        UsdShade.Shader.Define(stage, "/Root/Nested/Shader").CreateInput(
            "diffuse_texture", Sdf.ValueTypeNames.String
        ).Set("nested.png")
    stage.GetRootLayer().Save()

    layer = Sdf.Layer.FindOrOpen(str(stage_path))
    root_spec = layer.GetPrimAtPath("/Root")
    paths = {str(spec.path) for spec in artifact_manifest._all_prim_specs(root_spec)}
    assert any("{look=only}" in p for p in paths)


def test_dormant_composition_layers_handles_layer_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate, unresolvable, and raising dependencies are all tolerated."""
    from types import SimpleNamespace

    from pxr import Sdf

    root = SimpleNamespace(
        identifier="root",
        GetCompositionAssetDependencies=lambda: ["dup", "", "boom"],
    )
    dup = SimpleNamespace(identifier="root", GetCompositionAssetDependencies=list)
    stage = SimpleNamespace(GetRootLayer=lambda: root)

    def _compute(_layer, path):
        if path == "boom":
            raise RuntimeError("unresolvable")
        return path

    monkeypatch.setattr(
        artifact_manifest.Sdf, "ComputeAssetPathRelativeToLayer", _compute
    )
    monkeypatch.setattr(
        artifact_manifest.Sdf.Layer,
        "FindOrOpen",
        staticmethod(lambda path: dup if path == "dup" else None),
    )

    result = artifact_manifest._dormant_composition_layers(stage, set())

    # "dup" resolves back to the already-seen identifier, "" is skipped before
    # resolution, and "boom" raises; none of them may escape.
    assert [layer.identifier for layer in result] == ["root"]
    assert Sdf is not None


def test_dormant_composition_layers_tolerates_raising_dependency_listing() -> None:
    """A layer whose dependency listing raises is still recorded."""
    from types import SimpleNamespace

    def _raise():
        raise RuntimeError("no dependencies available")

    root = SimpleNamespace(identifier="root", GetCompositionAssetDependencies=_raise)
    stage = SimpleNamespace(GetRootLayer=lambda: root)

    result = artifact_manifest._dormant_composition_layers(stage, set())

    assert [layer.identifier for layer in result] == ["root"]


def test_dormant_composition_layers_requires_root_layer_accessor() -> None:
    """A stage stub without GetRootLayer yields no dormant layers."""
    from types import SimpleNamespace

    assert artifact_manifest._dormant_composition_layers(SimpleNamespace(), set()) == []


def test_variant_opinion_masked_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid prims, missing attributes, and empty stacks are not masked."""
    from types import SimpleNamespace

    invalid = SimpleNamespace(IsValid=lambda: False)
    assert (
        artifact_manifest._variant_opinion_is_masked(
            SimpleNamespace(GetPrimAtPath=lambda _p: invalid), "/A", "inputs:x_texture"
        )
        is False
    )

    no_attr = SimpleNamespace(IsValid=lambda: True, GetAttribute=lambda _n: None)
    assert (
        artifact_manifest._variant_opinion_is_masked(
            SimpleNamespace(GetPrimAtPath=lambda _p: no_attr), "/A", "inputs:x_texture"
        )
        is False
    )

    empty_stack = SimpleNamespace(
        IsValid=lambda: True,
        GetAttribute=lambda _n: SimpleNamespace(GetPropertyStack=list),
    )
    assert (
        artifact_manifest._variant_opinion_is_masked(
            SimpleNamespace(GetPrimAtPath=lambda _p: empty_stack),
            "/A",
            "inputs:x_texture",
        )
        is False
    )

    def _raise(_path):
        raise RuntimeError("stage unavailable")

    assert (
        artifact_manifest._variant_opinion_is_masked(
            SimpleNamespace(GetPrimAtPath=_raise), "/A", "inputs:x_texture"
        )
        is False
    )
