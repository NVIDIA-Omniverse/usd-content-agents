# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for USD composition arc analysis utilities."""

from pxr import Sdf, Usd, UsdGeom

from world_understanding.utils.usd import composition as composition_utils
from world_understanding.utils.usd.composition import (
    collect_composition_arcs,
    iter_prim_spec_paths,
)


class TestIterPrimSpecPaths:
    """Tests for iter_prim_spec_paths."""

    def test_empty_layer(self):
        """Empty layer returns no paths."""
        layer = Sdf.Layer.CreateAnonymous()
        paths = iter_prim_spec_paths(layer)
        assert paths == []

    def test_single_prim(self):
        """Layer with one prim returns its path."""
        layer = Sdf.Layer.CreateAnonymous()
        Sdf.CreatePrimInLayer(layer, "/Root")
        paths = iter_prim_spec_paths(layer)
        assert len(paths) == 1
        assert str(paths[0]) == "/Root"

    def test_nested_prims(self):
        """Layer with nested prims returns all paths."""
        layer = Sdf.Layer.CreateAnonymous()
        Sdf.CreatePrimInLayer(layer, "/Root")
        Sdf.CreatePrimInLayer(layer, "/Root/Child")
        Sdf.CreatePrimInLayer(layer, "/Root/Child/Grandchild")
        paths = iter_prim_spec_paths(layer)
        path_strs = [str(p) for p in paths]
        assert "/Root" in path_strs
        assert "/Root/Child" in path_strs
        assert "/Root/Child/Grandchild" in path_strs


class TestCollectCompositionArcs:
    """Tests for collect_composition_arcs."""

    def test_simple_stage_no_arcs(self):
        """Stage with no composition arcs returns zeros."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        result = collect_composition_arcs(stage)

        assert result["sublayer_count"] == 0
        assert result["reference_count"] == 0
        assert result["payload_count"] == 0
        assert result["variant_set_count"] == 0
        assert result["unique_sub_usd_count"] == 0
        assert result["sub_usd_files"] == []

    def test_stage_with_sublayers(self, tmp_path):
        """Stage with sublayers reports sublayer count."""
        sub_layer = Sdf.Layer.CreateNew(str(tmp_path / "sub.usda"))
        sub_layer.Save()

        root_layer = Sdf.Layer.CreateNew(str(tmp_path / "root.usda"))
        root_layer.subLayerPaths.append(str(tmp_path / "sub.usda"))
        root_layer.Save()

        stage = Usd.Stage.Open(root_layer)
        result = collect_composition_arcs(stage)
        assert result["sublayer_count"] == 1

    def test_stage_with_reference(self, tmp_path):
        """Stage with a reference reports reference count."""
        # Create a referenced file
        ref_layer = Sdf.Layer.CreateNew(str(tmp_path / "ref.usda"))
        Sdf.CreatePrimInLayer(ref_layer, "/RefRoot")
        ref_layer.Save()

        # Create main stage with a reference
        root_layer = Sdf.Layer.CreateNew(str(tmp_path / "main.usda"))
        prim_spec = Sdf.CreatePrimInLayer(root_layer, "/World")
        prim_spec.referenceList.Prepend(
            Sdf.Reference(str(tmp_path / "ref.usda"), "/RefRoot")
        )
        root_layer.Save()

        stage = Usd.Stage.Open(root_layer)
        result = collect_composition_arcs(stage)
        assert result["reference_count"] == 1
        assert result["unique_sub_usd_count"] == 1
        assert len(result["sub_usd_files"]) == 1

    def test_stage_with_all_list_editor_arcs_and_empty_assets(self, tmp_path):
        """All supported Sdf list editor operations are counted."""
        root_layer = Sdf.Layer.CreateNew(str(tmp_path / "main.usda"))

        prim_spec = Sdf.CreatePrimInLayer(root_layer, "/World")
        prim_spec.referenceList.Prepend(Sdf.Reference())
        prim_spec.referenceList.Prepend(Sdf.Reference("prepend.usda"))
        prim_spec.referenceList.Append(Sdf.Reference("append.usda"))
        prim_spec.referenceList.Add(Sdf.Reference("add.usda"))
        prim_spec.payloadList.Prepend(Sdf.Payload("prepend_payload.usda"))
        prim_spec.payloadList.Append(Sdf.Payload("append_payload.usda"))
        prim_spec.payloadList.Add(Sdf.Payload("add_payload.usda"))

        explicit_spec = Sdf.CreatePrimInLayer(root_layer, "/World/Explicit")
        explicit_spec.referenceList.ClearEditsAndMakeExplicit()
        explicit_spec.referenceList.Add(Sdf.Reference("explicit.usda"))
        explicit_spec.payloadList.ClearEditsAndMakeExplicit()
        explicit_spec.payloadList.Add(Sdf.Payload("explicit_payload.usda"))

        root_layer.Save()
        stage = Usd.Stage.Open(root_layer)

        result = collect_composition_arcs(stage)

        assert result["reference_count"] == 4
        assert result["payload_count"] == 4
        assert result["unique_sub_usd_count"] == 4
        assert {
            item["asset_path"]: item["referencing_prims"]
            for item in result["sub_usd_files"]
        } == {
            "prepend.usda": ["/World"],
            "append.usda": ["/World"],
            "add.usda": ["/World"],
            "explicit.usda": ["/World/Explicit"],
        }

    def test_collect_composition_arcs_skips_missing_prim_specs(self, monkeypatch):
        """Defensively skip paths that disappear before lookup."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        monkeypatch.setattr(
            composition_utils,
            "iter_prim_spec_paths",
            lambda _layer: [Sdf.Path("/Missing")],
        )

        result = collect_composition_arcs(stage)

        assert result["reference_count"] == 0
        assert result["payload_count"] == 0

    def test_stage_with_variant_sets(self):
        """Stage with variant sets reports variant set count."""
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/World")
        vset = prim.GetVariantSets().AddVariantSet("color")
        vset.AddVariant("red")
        vset.AddVariant("blue")

        result = collect_composition_arcs(stage)
        assert result["variant_set_count"] == 1
