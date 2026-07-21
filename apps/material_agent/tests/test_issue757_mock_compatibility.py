# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for Material Agent mock prompt compatibility."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from pxr import Usd, UsdGeom
from world_understanding.functions.models.backends.public.mock import (
    MockChatModel,
    MockVLM,
)

from material_agent.prompt_security import format_material_names_for_prompt
from material_agent.tasks.apply_materials_to_usd import ApplyMaterialsToUSDTask
from material_agent.tasks.inference import VLMInferenceTask
from material_agent.tasks.prepare_dataset import (
    _VLM_SYSTEM_PROMPT_TEMPLATE,
    render_vlm_system_prompt_template,
)


def _create_input_usd(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/RootNode")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/RootNode/Geometry")
    UsdGeom.Mesh.Define(stage, "/RootNode/Geometry/Part1")
    stage.GetRootLayer().Save()


def test_mock_predict_to_apply_succeeds_without_prediction_validation(
    tmp_path: Path,
) -> None:
    """Exercise the simulate predict/apply path with validation intentionally off."""
    material_name = "TestMaterial"
    system_prompt = render_vlm_system_prompt_template(
        _VLM_SYSTEM_PROMPT_TEMPLATE,
        materials_list=format_material_names_for_prompt([{"name": material_name}]),
    )
    image_path = tmp_path / "part.png"
    Image.new("RGB", (8, 8), color="gray").save(image_path)
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "/RootNode/Geometry/Part1",
                "text": "Choose a material for this simulated part.",
                "images": [image_path.name],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    prediction_context = VLMInferenceTask(
        vlm=MockVLM(),
        llm=MockChatModel(),
    ).run(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "output_dir": str(tmp_path / "predictions"),
            "config": {
                "system_prompt": system_prompt,
                "steps": {"validate_predictions": {"enabled": False}},
            },
            "vlm_config": {"max_retries": 1},
            "max_workers": 1,
            "prediction_batch_size": 1,
        }
    )
    predictions_path = Path(prediction_context["predictions_path"])
    prediction = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert prediction["materials"]["material"] == material_name

    input_usd = tmp_path / "input.usda"
    output_usd = tmp_path / "output.usda"
    _create_input_usd(input_usd)
    apply_result = ApplyMaterialsToUSDTask().run(
        {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(output_usd),
            "predictions_path": str(predictions_path),
            "resolved_materials": {material_name: str(tmp_path / "test.mdl")},
            "flatten_output": False,
        }
    )

    assert output_usd.exists()
    assert material_name in apply_result["materials_applied"]
