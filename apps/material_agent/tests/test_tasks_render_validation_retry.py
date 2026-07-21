# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw
from pxr import Usd, UsdGeom
from world_understanding.functions.graphics import rendering_backend_factory
from world_understanding.functions.graphics.render_validation import (
    RENDER_BLANK_IMAGE,
    validate_image_artifact,
)

import material_agent.tasks.render as render_module
from material_agent.tasks.render import RenderTask


def _make_stage(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    cube = UsdGeom.Cube.Define(stage, "/Cube")
    stage.SetDefaultPrim(cube.GetPrim())
    stage.Save()
    return path


def _valid_render_image() -> Image.Image:
    image = Image.new("RGB", (32, 32), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 15, 15], fill=(30, 30, 30))
    draw.rectangle([16, 0, 31, 15], fill=(120, 120, 120))
    draw.rectangle([0, 16, 31, 31], fill=(200, 200, 200))
    return image


class _FakeRemoteBackend:
    images: list[Image.Image] = []
    calls: int = 0

    def __init__(self, **_: Any) -> None:
        pass

    def render(self, **_: Any) -> dict[str, Any]:
        image_index = min(type(self).calls, len(type(self).images) - 1)
        image = type(self).images[image_index].copy()
        type(self).calls += 1
        return {
            "successful_cameras": 1,
            "failed_cameras": 0,
            "results": [
                {
                    "camera": "/RenderCamera",
                    "images": [image],
                    "status": "success",
                }
            ],
        }


@pytest.fixture(autouse=True)
def _patch_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_module.time, "sleep", lambda *_: None)


def test_render_task_retries_blank_remote_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeRemoteBackend.images = [
        Image.new("RGB", (32, 32), (255, 255, 255)),
        _valid_render_image(),
    ]
    _FakeRemoteBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "RemoteRenderingBackend", _FakeRemoteBackend
    )

    result = RenderTask().run(
        {
            "input_usd_path": str(_make_stage(tmp_path / "scene.usda")),
            "output_base_path": str(tmp_path / "renders"),
            "render_config": {
                "backend": "remote",
                "image_width": 32,
                "image_height": 32,
                "camera_corners": ["+x+y+z"],
                "max_attempts": 2,
            },
        }
    )

    assert _FakeRemoteBackend.calls == 2
    output_path = Path(result["rendered_image_path"])
    assert output_path.exists()
    assert validate_image_artifact(output_path).passed
    assert result["rendering_stats"]["validation_retry_count"] == 1
    validations = result["render_validation"]
    assert len(validations) == 2
    assert validations[0]["validation"]["issues"][0]["code"] == RENDER_BLANK_IMAGE
    assert validations[1]["validation"]["passed"] is True
    assert output_path.with_name(f"{output_path.stem}.blank_attempt_1.png").exists()
    assert (tmp_path / "renders" / "_render_debug").exists()


def test_render_task_fails_if_blank_remote_image_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeRemoteBackend.images = [
        Image.new("RGB", (32, 32), (255, 255, 255)),
        Image.new("RGB", (32, 32), (255, 255, 255)),
    ]
    _FakeRemoteBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "RemoteRenderingBackend", _FakeRemoteBackend
    )

    context = {
        "input_usd_path": str(_make_stage(tmp_path / "scene.usda")),
        "output_base_path": str(tmp_path / "renders"),
        "render_config": {
            "backend": "remote",
            "image_width": 32,
            "image_height": 32,
            "camera_corners": ["+x+y+z"],
            "max_attempts": 2,
        },
    }
    with pytest.raises(RuntimeError, match="Rendered image failed validation"):
        RenderTask().run(context)

    assert _FakeRemoteBackend.calls == 2
    assert len(context["render_validation"]) == 2
    assert context["render_validation"][0]["validation"]["issues"][0]["code"] == (
        RENDER_BLANK_IMAGE
    )
    assert context["rendering_stats"]["total_images"] == 0
    assert context["rendering_stats"]["failed_renders"] == 1
    assert context["rendering_stats"]["validation_retry_count"] == 1
