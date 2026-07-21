# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for restore_usd plus apply preserving original topology."""

import json
from pathlib import Path

from pxr import Usd, UsdGeom, UsdShade
from world_understanding.agentic.usd_tasks.restore_usd import RestoreUSDTask

from material_agent.tasks.apply_materials_to_usd import ApplyMaterialsToUSDTask


def _write_original_stage(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/Root/Body")
    mesh.CreatePointsAttr([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])

    for name, indices in (("SubsetA", [0]), ("SubsetB", [1])):
        subset = UsdGeom.Subset.Define(stage, f"/Root/Body/{name}")
        subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr(indices)

    stage.GetRootLayer().Export(str(path))


def test_restore_usd_predictions_apply_to_original_topology(tmp_path: Path) -> None:
    """Split optimized predictions should apply to original USD GeomSubsets."""
    original_usd_path = tmp_path / "original.usda"
    _write_original_stage(original_usd_path)

    optimized_predictions_path = tmp_path / "optimized_predictions.jsonl"
    optimized_predictions_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"id": "/Optimized/Body_part_0", "material": "Smoke_Material"}
                ),
                json.dumps(
                    {"id": "/Optimized/Body_part_1", "material": "Smoke_Material"}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    restored_predictions_path = tmp_path / "restored_predictions.jsonl"
    optimization_metadata = {
        "correspondence_map": {
            "full_mapping": {
                "original_to_prototype": {
                    "/Root/Body": [
                        "/Optimized/Body_part_0",
                        "/Optimized/Body_part_1",
                    ]
                }
            },
            "split_mapping": {"/Root/Body": {}},
            "summary": {"operations_run": {"split": True}},
        }
    }

    restore_context = {
        "original_usd_path": str(original_usd_path),
        "predictions_path": str(optimized_predictions_path),
        "output_predictions_path": str(restored_predictions_path),
        "optimization_metadata": optimization_metadata,
    }
    RestoreUSDTask().run(restore_context)

    restored_predictions = [
        json.loads(line)
        for line in restored_predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [prediction["id"] for prediction in restored_predictions] == [
        "/Root/Body/SubsetA",
        "/Root/Body/SubsetB",
    ]

    output_usd_path = tmp_path / "output.usda"
    apply_context = {
        "input_usd_path": str(original_usd_path),
        "output_usd_path": str(output_usd_path),
        "predictions_path": str(restored_predictions_path),
        "resolved_materials": {"Smoke_Material": "fallback-material.usd"},
        "flatten_output": False,
        "skip_instance_check": True,
    }
    ApplyMaterialsToUSDTask().run(apply_context)

    assert apply_context["assignment_stats"]["failed"] == 0
    assert apply_context["assignment_stats"]["total_prims"] == 2
    assert apply_context["assignment_stats"]["bound_prim_ids"] == [
        "/Root/Body/SubsetA",
        "/Root/Body/SubsetB",
    ]
    assert apply_context["assignment_stats"]["unbound_prim_ids"] == []

    output_stage = Usd.Stage.Open(str(output_usd_path))
    assert output_stage.GetPrimAtPath("/Root/Body").IsValid()
    assert not output_stage.GetPrimAtPath("/Optimized/Body_part_0").IsValid()
    assert not output_stage.GetPrimAtPath("/Optimized/Body_part_1").IsValid()

    for prim_path in ("/Root/Body/SubsetA", "/Root/Body/SubsetB"):
        material, _relationship = UsdShade.MaterialBindingAPI(
            output_stage.GetPrimAtPath(prim_path)
        ).ComputeBoundMaterial()
        assert material
        assert str(material.GetPath()) == "/Materials/Smoke_Material"
