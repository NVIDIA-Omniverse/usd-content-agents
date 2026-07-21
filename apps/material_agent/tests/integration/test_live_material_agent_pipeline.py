# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live end-to-end tests for Material Agent pipeline (config -> tasks -> backend).

These tests are gated and require RUN_LIVE_INFERENCE=1 and provider API keys.
They create a tiny dataset (one black image), load YAML config via PredictConfigTask,
provision models via ModelProvisioningTask, and run inference via VLMInferenceTask.
"""

import json
import os
from pathlib import Path

import pytest
import yaml
from PIL import Image

from material_agent.tasks import (
    ModelProvisioningTask,
    PredictConfigTask,
    VLMInferenceTask,
)

pytestmark = pytest.mark.live_inference


RUN_LIVE = os.getenv("RUN_LIVE_INFERENCE") == "1"


@pytest.mark.skipif(
    not RUN_LIVE or not os.getenv("NVIDIA_API_KEY"),
    reason="Live NIM test requires RUN_LIVE_INFERENCE=1 and NVIDIA_API_KEY",
)
def test_live_material_agent_nim_vlm(tmp_path: Path) -> None:
    """End-to-end NIM VLM test through Material Agent tasks.

    Verifies a simple black image classification flows through config -> tasks -> backend
    and writes predictions.jsonl.
    """
    # Create a tiny black image and dataset
    img_path = tmp_path / "black.png"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(img_path)

    dataset_path = tmp_path / "dataset.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        rec = {
            "id": "nim_black_001",
            "text": "What material is the closest to the color in the image?",
            "images": [img_path.name],  # relative to image_base_dir
        }
        f.write(json.dumps(rec) + "\n")

    output_dir = tmp_path / "out"

    # Minimal config (to-be-converted to YAML) for PredictConfigTask
    config = {
        "dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "system_prompt": (
            "You are an expert at selecting materials. Choose the single best material "
            "from this list and return ONLY JSON inside <answer> tags.\n\n"
            "Materials list:\n"
            "- black matte plastic\n"
            "- white ceramic\n"
            "- red fabric\n\n"
            'Output:\n<answer>\n{"material": "<exact name from the list>"}\n</answer>'
        ),
        "vlm": {
            "backend": "nim",
            "model": "nvdev/meta/llama-4-scout-17b-16e-instruct",
            "temperature": 0.2,
            "max_tokens": 16,
        },
        "llm": {
            "backend": "nim",
        },
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    # Run tasks: load config -> provision models -> inference
    context: dict = {"config_path": str(config_path)}

    context = PredictConfigTask().run(context)
    # image_base_dir is required by VLMInferenceTask to resolve relative image paths
    context["image_base_dir"] = str(tmp_path)

    context = ModelProvisioningTask().run(context)
    context = VLMInferenceTask().run(context)

    # Assertions
    assert context.get("inference_complete") is True
    predictions_path = Path(context.get("predictions_path", ""))
    assert predictions_path.exists(), "predictions.jsonl should be created"

    # Must contain at least one success line
    with open(predictions_path, encoding="utf-8") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]
    assert len(lines) >= 1

    first = json.loads(lines[0])
    assert "materials" in first
    materials = first["materials"]
    if isinstance(materials, dict) and "material" in materials:
        mat_text = str(materials["material"]).lower().strip()
    else:
        mat_text = str(materials).lower().strip()
    assert mat_text == "black matte plastic"
