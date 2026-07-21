# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for mock rendering and the ComfyUI image-editing wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from world_understanding.functions.graphics import comfyui_workflows, image_editing
from world_understanding.functions.graphics.mock_rendering import MockRenderingBackend


def test_mock_rendering_backend_outputs_and_frame_parsing() -> None:
    backend = MockRenderingBackend()
    assert backend.supports_sensors() is False

    result = backend.render(
        stage=object(),
        cameras=["cam_a", "cam_b"],
        image_width=16,
        frames="0:2",
        sensors=["ignored"],
    )
    assert result["total_cameras"] == 2
    assert result["successful_cameras"] == 2
    assert result["failed_cameras"] == 0
    assert result["results"][0]["frame_count"] == 3
    assert [image.size for image in result["results"][0]["images"]] == [
        (16, 16),
        (16, 16),
        (16, 16),
    ]
    assert result["results"][0]["images"][0].getpixel((8, 8)) != (180, 180, 180, 255)

    default_result = backend.render(stage=object(), image_width=12, frames="4")
    assert default_result["results"][0]["camera"] == "default"
    assert default_result["results"][0]["images"][0].size == (12, 12)

    assert backend._parse_frame_count("1-3") == 3
    assert backend._parse_frame_count("bad-range") == 1
    assert backend._parse_frame_count("1:bad") == 1
    assert backend._parse_frame_count("0,2,4") == 3
    assert len(backend._camera_color("cam_a")) == 3


class FakeComfyClient:
    instances: list[FakeComfyClient] = []

    def __init__(self, server_url: str | None = None) -> None:
        self.server_url = server_url
        self.uploaded_paths: list[str] = []
        self.executions: list[
            tuple[dict[str, Any], dict[str, Any], list[str], int]
        ] = []
        FakeComfyClient.instances.append(self)

    def upload_image(self, image_path: str) -> tuple[str, str, str]:
        self.uploaded_paths.append(image_path)
        return ("uploaded.png", "input", "input")

    def execute_workflow(
        self,
        workflow: dict[str, Any],
        inputs: dict[str, Any],
        output_nodes: list[str],
        timeout: int,
    ) -> dict[str, Image.Image]:
        self.executions.append((workflow, inputs, output_nodes, timeout))
        outputs = {"60": Image.new("RGB", (4, 3), "green")}
        if "104" in output_nodes:
            outputs["104"] = Image.new("RGB", (2, 2), "blue")
        return outputs


def test_edit_image_with_comfyui_path_and_pil_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeComfyClient.instances.clear()
    monkeypatch.setattr(image_editing, "ComfyUIClient", FakeComfyClient)

    input_path = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "red").save(input_path)
    result = image_editing.edit_image_with_comfyui(
        input_path,
        prompt="make it green",
        negative_prompt="blur",
        server_url="http://server",
        timeout=7,
    )

    assert result["edited_image"].size == (4, 3)
    assert result["image_size"] == (4, 3)
    assert result["execution_time"] >= 0.0
    client = FakeComfyClient.instances[-1]
    workflow, inputs, output_nodes, timeout = client.executions[-1]
    assert client.server_url == "http://server"
    assert client.uploaded_paths == [str(input_path)]
    assert inputs == {}
    assert output_nodes == ["60"]
    assert timeout == 7
    assert workflow["78"]["inputs"]["image"] == "uploaded.png"
    assert workflow["76"]["inputs"]["prompt"] == "make it green"
    assert workflow["77"]["inputs"]["prompt"] == "blur"

    pil_result = image_editing.edit_image_with_comfyui(
        Image.new("RGB", (2, 2), "white"),
        prompt="edit",
        return_rescaled_input=True,
    )
    pil_client = FakeComfyClient.instances[-1]
    temp_path = Path(pil_client.uploaded_paths[0])
    assert pil_result["rescaled_input"].size == (2, 2)
    assert pil_client.executions[-1][2] == ["60", "104"]
    assert temp_path.exists() is False


def test_execute_comfyui_workflow_success_and_missing_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeComfyClient.instances.clear()
    monkeypatch.setattr(comfyui_workflows, "ComfyUIClient", FakeComfyClient)

    import time

    times = iter((10.0, 12.5))
    monkeypatch.setattr(time, "time", lambda: next(times))

    result = comfyui_workflows.execute_comfyui_workflow(
        "qwen_image_edit",
        {"prompt": "make it green"},
        output_nodes=["60"],
        server_url="http://comfy",
        timeout=11,
    )

    assert result["images"]["60"].size == (4, 3)
    assert result["execution_time"] == 2.5
    client = FakeComfyClient.instances[-1]
    workflow, inputs, output_nodes, timeout = client.executions[-1]
    assert client.server_url == "http://comfy"
    assert workflow["76"]["inputs"]["prompt"] == "add several chairs"
    assert inputs == {"prompt": "make it green"}
    assert output_nodes == ["60"]
    assert timeout == 11

    with pytest.raises(FileNotFoundError, match="Workflow not found"):
        comfyui_workflows.execute_comfyui_workflow("missing_workflow", {})


def test_edit_image_with_comfyui_numpy_and_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeComfyClient.instances.clear()
    monkeypatch.setattr(image_editing, "ComfyUIClient", FakeComfyClient)

    result = image_editing.edit_image_with_comfyui(
        np.zeros((2, 2, 3), dtype=np.uint8),
        prompt="edit array",
    )
    client = FakeComfyClient.instances[-1]
    temp_path = Path(client.uploaded_paths[0])
    assert result["edited_image"].size == (4, 3)
    assert temp_path.exists() is False

    with pytest.raises(ValueError, match="Unsupported image type"):
        image_editing.edit_image_with_comfyui(object(), prompt="bad")
