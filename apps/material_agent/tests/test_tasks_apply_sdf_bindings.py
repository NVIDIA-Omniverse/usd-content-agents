# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Sdf-level binding and instance-aware path remapping in ApplyMaterialsToUSDTask."""

import json
from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdShade

from material_agent.tasks.apply_materials_to_usd import ApplyMaterialsToUSDTask
from material_agent.tasks.config_apply import ApplyConfigTask


class TestApplySkipInstanceCheck:
    """Tests for skip_instance_check flag."""

    def test_task_instantiation(self) -> None:
        """ApplyMaterialsToUSDTask can be instantiated."""
        task = ApplyMaterialsToUSDTask()
        assert task.name == "ApplyMaterialsToUSD"

    def test_empty_predictions_file_requires_opt_in(self, tmp_path: Path) -> None:
        """Empty predictions fail closed unless explicitly allowed."""
        usd_path = str(tmp_path / "test.usda")
        stage = Usd.Stage.CreateNew(usd_path)
        UsdGeom.Mesh.Define(stage, "/World/Mesh")
        stage.Save()

        pred_path = tmp_path / "predictions.jsonl"
        pred_path.write_text("")

        task = ApplyMaterialsToUSDTask()
        context = {
            "input_usd_path": usd_path,
            "resolved_materials": {},
            "prim_to_material": {},
            "predictions_path": str(pred_path),
            "output_usd_path": str(tmp_path / "output.usda"),
        }

        with pytest.raises(ValueError, match="No material predictions"):
            task.run(context, None)

        context["allow_empty_predictions"] = True
        result = task.run(context, None)
        assert "output_usd_path" in result
        assert result["resolved_material_profile"] == "auto"
        assert result["material_profile_warnings"] == []
        assert result["material_profile_errors"] == []


class TestSdfLevelBindings:
    """Tests for Sdf-level material binding in the material layer."""

    def test_sdf_binding_writes_relationship(self, tmp_path):
        """Sdf-level binding writes material:binding relationship on layer."""
        # Create a simple stage
        usd_path = str(tmp_path / "test.usda")
        stage = Usd.Stage.CreateNew(usd_path)
        UsdGeom.Mesh.Define(stage, "/World/Mesh")
        stage.Save()

        # Create an output layer with an over prim and Sdf-level binding
        output_path = str(tmp_path / "output.usda")
        out_layer = Sdf.Layer.CreateNew(output_path)

        # Create material
        mat_spec = Sdf.CreatePrimInLayer(out_layer, "/Materials/Steel")
        mat_spec.specifier = Sdf.SpecifierDef
        mat_spec.typeName = "Material"

        # Create over prim and write binding at Sdf level
        prim_spec = Sdf.CreatePrimInLayer(out_layer, "/World/Mesh")
        prim_spec.specifier = Sdf.SpecifierOver
        prim_spec.SetInfo(
            "apiSchemas",
            Sdf.TokenListOp.Create(prependedItems=["MaterialBindingAPI"]),
        )
        binding_rel = Sdf.RelationshipSpec(prim_spec, "material:binding")
        binding_rel.targetPathList.explicitItems = [Sdf.Path("/Materials/Steel")]

        out_layer.Save()

        # Verify the binding exists at the Sdf level
        reloaded = Sdf.Layer.FindOrOpen(output_path)
        ps = reloaded.GetPrimAtPath("/World/Mesh")
        assert ps is not None
        rel = ps.relationships.get("material:binding")
        assert rel is not None
        assert Sdf.Path("/Materials/Steel") in rel.targetPathList.explicitItems

    def test_config_apply_has_skip_instance_check(self, tmp_path: Path) -> None:
        """ApplyConfigTask passes skip_instance_check from config."""
        import yaml

        config = {
            "input_usd_path": "/dummy.usd",
            "predictions_path": "/dummy.jsonl",
            "output_usd_path": "/dummy_out.usd",
            "skip_instance_check": True,
            "allow_empty_predictions": True,
            "fail_on_unknown_material": True,
            "shader_target": "openpbr",
        }
        config_path = tmp_path / "apply_config.yaml"
        config_path.write_text(yaml.dump(config))

        task = ApplyConfigTask()
        context = {"config_path": str(config_path)}
        result = task.run(context, None)
        assert result["skip_instance_check"] is True
        assert result["allow_empty_predictions"] is True
        assert result["fail_on_unknown_material"] is True
        assert result["material_profile"] == "openpbr"

    def test_config_apply_validates_allow_empty_predictions(
        self, tmp_path: Path
    ) -> None:
        """ApplyConfigTask rejects non-boolean empty-prediction opt-ins."""
        import yaml

        config_path = tmp_path / "apply_config.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "input_usd_path": "/dummy.usd",
                    "predictions_path": "/dummy.jsonl",
                    "output_usd_path": "/dummy_out.usd",
                    "allow_empty_predictions": "yes",
                }
            )
        )

        with pytest.raises(ValueError, match="apply.allow_empty_predictions"):
            ApplyConfigTask().run({"config_path": str(config_path)}, None)

    def test_config_apply_validates_material_profile_string(
        self, tmp_path: Path
    ) -> None:
        """ApplyConfigTask rejects non-string material profile aliases."""
        import yaml

        config_path = tmp_path / "apply_config.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "input_usd_path": "/dummy.usd",
                    "predictions_path": "/dummy.jsonl",
                    "output_usd_path": "/dummy_out.usd",
                    "material_profile": 123,
                }
            )
        )

        with pytest.raises(ValueError, match="apply.material_profile"):
            ApplyConfigTask().run({"config_path": str(config_path)}, None)

    def test_config_apply_validates_fail_on_unknown_material(
        self, tmp_path: Path
    ) -> None:
        """ApplyConfigTask rejects non-boolean strict unknown-material settings."""
        import yaml

        config_path = tmp_path / "apply_config.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "input_usd_path": "/dummy.usd",
                    "predictions_path": "/dummy.jsonl",
                    "output_usd_path": "/dummy_out.usd",
                    "fail_on_unknown_material": "yes",
                }
            )
        )

        with pytest.raises(ValueError, match="apply.fail_on_unknown_material"):
            ApplyConfigTask().run({"config_path": str(config_path)}, None)


def _write_instanced_geomsubset_stage(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())

    proto = UsdGeom.Xform.Define(stage, "/Root/Prototypes/Proto")
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Prototypes/Proto/Mesh")
    mesh.CreatePointsAttr([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])

    for name, indices in (("SubsetA", [0]), ("SubsetB", [1])):
        subset = UsdGeom.Subset.Define(stage, f"/Root/Prototypes/Proto/Mesh/{name}")
        subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr(indices)

    instance = UsdGeom.Xform.Define(stage, "/Root/Instance")
    instance.GetPrim().GetReferences().AddInternalReference(str(proto.GetPath()))
    instance.GetPrim().SetInstanceable(True)

    stage.Save()


def test_full_stage_remaps_instance_proxy_geomsubset_predictions(
    tmp_path: Path,
) -> None:
    """Predictions on instance-proxy GeomSubsets bind through the prototype source."""
    input_usd_path = tmp_path / "input.usda"
    _write_instanced_geomsubset_stage(input_usd_path)

    predictions_path = tmp_path / "predictions.jsonl"
    predictions = [
        {
            "id": "/Root/Instance/Mesh/SubsetA",
            "material": "Smoke_Material",
        },
        {
            "id": "/Root/Instance/Mesh/SubsetB",
            "material": "Smoke_Material",
        },
    ]
    predictions_path.write_text(
        "".join(json.dumps(prediction) + "\n" for prediction in predictions),
        encoding="utf-8",
    )

    output_usd_path = tmp_path / "output.usda"
    context = {
        "input_usd_path": str(input_usd_path),
        "output_usd_path": str(output_usd_path),
        "predictions_path": str(predictions_path),
        "resolved_materials": {"Smoke_Material": "fallback-material.usd"},
        "flatten_output": False,
        "skip_instance_check": True,
    }

    result = ApplyMaterialsToUSDTask().run(context)

    assert result["assignment_stats"]["failed"] == 0
    assert result["assignment_stats"]["total_prims"] == 2

    output_stage = Usd.Stage.Open(str(output_usd_path))
    for prim_path in (
        "/Root/Instance/Mesh/SubsetA",
        "/Root/Instance/Mesh/SubsetB",
    ):
        prim = output_stage.GetPrimAtPath(prim_path)
        assert prim.IsInstanceProxy()
        material, _relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial()
        assert material
        assert str(material.GetPath()) == "/Materials/Smoke_Material"

    for prim_path in (
        "/Root/Prototypes/Proto/Mesh/SubsetA",
        "/Root/Prototypes/Proto/Mesh/SubsetB",
    ):
        prim = output_stage.GetPrimAtPath(prim_path)
        assert not prim.IsInstanceProxy()
        material, _relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial()
        assert material
        assert str(material.GetPath()) == "/Materials/Smoke_Material"
