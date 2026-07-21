# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for render task helper behavior."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image, ImageDraw
from pxr import Usd, UsdGeom
from world_understanding.functions.graphics import rendering_backend_factory

import material_agent.tasks.render as render_module
from material_agent.tasks.render import (
    BlankRenderStatsDict,
    RenderTask,
    _blank_final_render_error,
    _preserve_failed_render_stage,
    _scene_render_max_workers,
)


def _blank_stats() -> BlankRenderStatsDict:
    return cast(
        BlankRenderStatsDict,
        {
            "reason": "solid_color",
            "unique_colors": 1,
            "dominant_color_ratio": 1.0,
            "luma_std": 0.0,
        },
    )


def _make_stage(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    cube = UsdGeom.Cube.Define(stage, "/Cube")
    stage.SetDefaultPrim(cube.GetPrim())
    stage.Save()
    return path


def _valid_render_image(mode: str = "RGB") -> Image.Image:
    image = Image.new(mode, (32, 32), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 15, 15], fill=(30, 30, 30))
    draw.rectangle([16, 0, 31, 15], fill=(120, 120, 120))
    draw.rectangle([0, 16, 31, 31], fill=(200, 200, 200))
    return image


def _success_result(image: Any | None = None) -> dict[str, Any]:
    return {
        "successful_cameras": 1,
        "failed_cameras": 0,
        "results": [
            {
                "camera": "/RenderCamera",
                "images": [_valid_render_image() if image is None else image],
                "status": "success",
            }
        ],
    }


class _SequenceBackend:
    results: list[dict[str, Any]] = []
    calls = 0
    init_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).init_kwargs = kwargs

    def render(self, **_: Any) -> dict[str, Any]:
        index = min(type(self).calls, len(type(self).results) - 1)
        type(self).calls += 1
        return type(self).results[index]


class _FakeValidation:
    def __init__(self, *, passed: bool = True, code: str | None = None) -> None:
        self.passed = passed
        self.issues = [] if code is None else [type("Issue", (), {"code": code})()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [{"code": issue.code} for issue in self.issues],
        }


@pytest.fixture(autouse=True)
def _patch_render_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_module.time, "sleep", lambda *_: None)


def test_blank_final_render_error_uses_remote_hint() -> None:
    error = _blank_final_render_error("out.png", _blank_stats(), "remote")

    assert "remote rendering endpoint logs" in error
    assert "WU_OVRTX_DEFAULT_HDRI" not in error


def test_blank_final_render_error_uses_ovrtx_hint() -> None:
    error = _blank_final_render_error("out.png", _blank_stats(), "ovrtx")

    assert "OVRTX rendering endpoint logs" in error
    assert "WU_OVRTX_DEFAULT_HDRI" in error


def test_scene_render_max_workers_preserves_backend_concurrency_policy() -> None:
    assert (
        _scene_render_max_workers(
            render_config={"max_workers": 8},
            backend_type="warp",
            camera_count=4,
        )
        == 1
    )
    assert (
        _scene_render_max_workers(
            render_config={"max_workers": 8},
            backend_type="ovrtx",
            camera_count=4,
        )
        == 1
    )
    assert (
        _scene_render_max_workers(
            render_config={},
            backend_type="local",
            camera_count=4,
        )
        == 2
    )
    assert (
        _scene_render_max_workers(
            render_config={},
            backend_type="remote",
            camera_count=4,
        )
        == 1
    )
    assert (
        _scene_render_max_workers(
            render_config={"max_workers": 3},
            backend_type="remote",
            camera_count=4,
        )
        == 3
    )
    assert (
        _scene_render_max_workers(
            render_config={"max_workers": 8},
            backend_type="mock",
            camera_count=2,
        )
        == 2
    )


def test_preserve_failed_render_stage_returns_none_when_export_fails(
    tmp_path: Path,
) -> None:
    class RootLayer:
        def Export(self, path: str) -> bool:
            return False

    class Stage:
        def GetRootLayer(self) -> RootLayer:
            return RootLayer()

    assert (
        _preserve_failed_render_stage(
            cast(Any, Stage()),
            output_base_path=tmp_path,
            output_path=tmp_path / "render.png",
            attempt=1,
        )
        is None
    )


def test_render_task_skips_when_disabled_or_input_missing(tmp_path: Path) -> None:
    disabled = RenderTask().run({"render_enabled": False})
    assert disabled["rendering_skipped"] is True

    missing = RenderTask().run({"input_usd_path": str(tmp_path / "missing.usda")})
    assert missing["rendering_skipped"] is True


def test_render_task_non_flatten_focused_side_view_with_remote_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SequenceBackend.results = [_success_result()]
    _SequenceBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "RemoteRenderingBackend", _SequenceBackend
    )
    monkeypatch.setattr(
        "world_understanding.utils.usd.prim.nullify_materials",
        lambda stage: None,
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.rendering.hide_prims_outside_subtree",
        lambda stage, prim_path: None,
    )

    result = RenderTask().run(
        {
            "input_usd_path": str(_make_stage(tmp_path / "scene.usda")),
            "render_config": {
                "backend": "remote",
                "camera_corner": "+x",
                "image_width": 16,
                "image_height": 32,
                "clear_materials": True,
                "prim_path": "/Cube",
                "base_url": "http://renderer",
                "timeout": 5,
                "max_attempts": 1,
            },
            "flatten_before_render": False,
        }
    )

    assert result["rendering_skipped"] is False
    assert result["rendered_image_path"].endswith("scene.png")
    assert (tmp_path / "scene_converted.usda").exists()
    assert _SequenceBackend.init_kwargs["base_url"] == "http://renderer"
    assert _SequenceBackend.init_kwargs["timeout"] == 5


def test_render_task_invalid_focus_prim_renders_full_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SequenceBackend.results = [_success_result()]
    _SequenceBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "RemoteRenderingBackend", _SequenceBackend
    )

    result = RenderTask().run(
        {
            "input_usd_path": str(_make_stage(tmp_path / "scene.usda")),
            "output_base_path": str(tmp_path / "renders"),
            "render_config": {
                "backend": "remote",
                "prim_path": "/Missing",
                "image_width": 16,
                "image_height": 16,
                "max_attempts": 1,
            },
        }
    )

    assert result["rendering_skipped"] is False


def test_render_task_ovrtx_and_unknown_backend_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SequenceBackend.results = [_success_result()]
    _SequenceBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "OvRTXRenderingBackend", _SequenceBackend
    )

    ovrtx = RenderTask().run(
        {
            "input_usd_path": str(_make_stage(tmp_path / "ovrtx.usda")),
            "output_base_path": str(tmp_path / "ovrtx-renders"),
            "render_config": {
                "backend": "ovrtx",
                "log_level": "debug",
                "image_width": 16,
                "image_height": 16,
            },
        }
    )
    assert ovrtx["rendering_skipped"] is False
    assert _SequenceBackend.init_kwargs["log_level"] == "debug"
    assert _SequenceBackend.init_kwargs["num_sensor_updates"] == 500
    assert _SequenceBackend.init_kwargs["render_mode"] == "pt"

    with pytest.raises(ValueError, match="Unknown rendering backend: not-a-backend"):
        RenderTask().run(
            {
                "input_usd_path": str(_make_stage(tmp_path / "unknown.usda")),
                "output_base_path": str(tmp_path / "unknown-renders"),
                "render_config": {"backend": "not-a-backend"},
            }
        )


def test_render_task_supports_warp_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SequenceBackend.results = [_success_result()]
    _SequenceBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "WarpRenderingBackend", _SequenceBackend
    )

    result = RenderTask().run(
        {
            "input_usd_path": str(_make_stage(tmp_path / "warp.usda")),
            "output_base_path": str(tmp_path / "warp-renders"),
            "render_config": {
                "backend": "warp",
                "device": "cuda:2",
                "image_width": 16,
                "image_height": 16,
            },
        }
    )

    assert result["rendering_skipped"] is False
    assert _SequenceBackend.init_kwargs["device"] == "cuda:2"


def test_render_task_retries_non_successful_result_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SequenceBackend.results = [
        {"successful_cameras": 0, "results": [{"error": "temporary outage"}]},
        _success_result(),
    ]
    _SequenceBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "RemoteRenderingBackend", _SequenceBackend
    )

    result = RenderTask().run(
        {
            "input_usd_path": str(_make_stage(tmp_path / "scene.usda")),
            "output_base_path": str(tmp_path / "renders"),
            "render_config": {"backend": "remote", "max_attempts": 2},
        }
    )

    assert _SequenceBackend.calls == 2
    assert result["rendered_image_path"]


def test_render_task_retries_missing_results_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SequenceBackend.results = [
        {"successful_cameras": 1},
        _success_result(),
    ]
    _SequenceBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "RemoteRenderingBackend", _SequenceBackend
    )

    result = RenderTask().run(
        {
            "input_usd_path": str(_make_stage(tmp_path / "scene.usda")),
            "output_base_path": str(tmp_path / "renders"),
            "render_config": {"backend": "remote", "max_attempts": 2},
        }
    )

    assert _SequenceBackend.calls == 2
    assert result["rendered_image_path"]


def test_render_task_no_image_data_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SequenceBackend.results = [
        {"successful_cameras": 1, "results": [{"images": []}]},
        _success_result(),
    ]
    _SequenceBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "RemoteRenderingBackend", _SequenceBackend
    )

    result = RenderTask().run(
        {
            "input_usd_path": str(_make_stage(tmp_path / "scene.usda")),
            "output_base_path": str(tmp_path / "renders"),
            "render_config": {"backend": "remote", "max_attempts": 2},
        }
    )

    assert _SequenceBackend.calls == 2
    assert result["rendered_image_path"]


def test_render_task_no_image_data_final_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SequenceBackend.results = [{"successful_cameras": 1, "results": [{"images": []}]}]
    _SequenceBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "RemoteRenderingBackend", _SequenceBackend
    )

    with pytest.raises(RuntimeError, match="No image data in result"):
        RenderTask().run(
            {
                "input_usd_path": str(_make_stage(tmp_path / "scene.usda")),
                "output_base_path": str(tmp_path / "renders"),
                "render_config": {"backend": "remote", "max_attempts": 1},
            }
        )


def test_render_task_remote_stage_reopen_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SequenceBackend.results = [_success_result()]
    _SequenceBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "RemoteRenderingBackend", _SequenceBackend
    )
    original_open = render_module.Usd.Stage.Open
    flat_opens = {"count": 0}

    def open_until_render_attempt(path: str):
        path_text = str(path) if isinstance(path, str) else ""
        if path_text.endswith("_flat.usd"):
            flat_opens["count"] += 1
        if path_text.endswith("_flat.usd") and flat_opens["count"] == 2:
            return None
        return original_open(path)

    monkeypatch.setattr(render_module.Usd.Stage, "Open", open_until_render_attempt)

    with pytest.raises(RuntimeError, match="Failed to reopen serialized USD stage"):
        RenderTask().run(
            {
                "input_usd_path": str(_make_stage(tmp_path / "scene.usda")),
                "output_base_path": str(tmp_path / "renders"),
                "render_config": {"backend": "remote", "max_attempts": 1},
            }
        )


def test_render_task_accepts_bytes_and_base64_image_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = BytesIO()
    _valid_render_image().save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    _SequenceBackend.results = [
        _success_result({"image": png_bytes}),
        _success_result({"image": base64.b64encode(png_bytes).decode("ascii")}),
    ]
    _SequenceBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "RemoteRenderingBackend", _SequenceBackend
    )

    result = RenderTask().run(
        {
            "input_usd_path": str(_make_stage(tmp_path / "scene.usda")),
            "output_base_path": str(tmp_path / "renders"),
            "render_config": {
                "backend": "remote",
                "camera_corners": ["+x+y+z", "-x-y+z"],
                "max_attempts": 1,
            },
        }
    )

    assert len(result["rendered_image_paths"]) == 2
    assert Path(result["rendered_image_paths"][0]).name == "scene_posx_posy_posz.png"


def test_render_task_mock_backend_writes_images(tmp_path: Path) -> None:
    result = RenderTask().run(
        {
            "input_usd_path": str(_make_stage(tmp_path / "scene.usda")),
            "output_base_path": str(tmp_path / "renders"),
            "render_config": {
                "backend": "mock",
                "camera_corners": ["+x+y+z", "-x-y+z"],
                "image_width": 64,
                "image_height": 64,
            },
        }
    )

    rendered_paths = [Path(path) for path in result["rendered_image_paths"]]
    assert len(rendered_paths) == 2
    assert all(path.is_file() for path in rendered_paths)
    assert result["rendering_skipped"] is False
    assert result["rendering_stats"]["backend"] == "mock"


def test_render_task_unexpected_image_format_retry_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SequenceBackend.results = [
        _success_result("bad-image-payload"),
        _success_result(),
    ]
    _SequenceBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "RemoteRenderingBackend", _SequenceBackend
    )

    retried = RenderTask().run(
        {
            "input_usd_path": str(_make_stage(tmp_path / "retry.usda")),
            "output_base_path": str(tmp_path / "retry-renders"),
            "render_config": {"backend": "remote", "max_attempts": 2},
        }
    )
    assert retried["rendered_image_path"]

    _SequenceBackend.results = [_success_result("bad-image-payload")]
    _SequenceBackend.calls = 0
    with pytest.raises(RuntimeError, match="Unexpected image format"):
        RenderTask().run(
            {
                "input_usd_path": str(_make_stage(tmp_path / "fail.usda")),
                "output_base_path": str(tmp_path / "fail-renders"),
                "render_config": {"backend": "remote", "max_attempts": 1},
            }
        )


def test_render_task_warns_for_non_blank_validation_issue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SequenceBackend.results = [_success_result()]
    _SequenceBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "RemoteRenderingBackend", _SequenceBackend
    )
    monkeypatch.setattr(
        render_module,
        "validate_image_artifact",
        lambda *args, **kwargs: _FakeValidation(passed=False, code="low_contrast"),
    )

    result = RenderTask().run(
        {
            "input_usd_path": str(_make_stage(tmp_path / "scene.usda")),
            "output_base_path": str(tmp_path / "renders"),
            "render_config": {"backend": "remote", "max_attempts": 1},
        }
    )

    assert result["render_validation"][0]["validation"]["issues"][0]["code"] == (
        "low_contrast"
    )


def test_render_task_allows_partial_renders_with_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SequenceBackend.results = [
        _success_result(),
        {"successful_cameras": 0, "results": [{"error": "camera failed"}]},
    ]
    _SequenceBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "RemoteRenderingBackend", _SequenceBackend
    )

    result = RenderTask().run(
        {
            "input_usd_path": str(_make_stage(tmp_path / "scene.usda")),
            "output_base_path": str(tmp_path / "renders"),
            "render_config": {
                "backend": "remote",
                "camera_corners": ["+x+y+z", "-x-y+z"],
                "max_attempts": 1,
                "allow_partial_renders": True,
            },
        }
    )

    assert len(result["rendered_image_paths"]) == 1
    assert result["rendering_errors"][0]["error"] == "camera failed"


def test_render_task_zero_attempts_reports_exhausted_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _SequenceBackend.results = [_success_result()]
    _SequenceBackend.calls = 0
    monkeypatch.setattr(
        rendering_backend_factory, "RemoteRenderingBackend", _SequenceBackend
    )

    with pytest.raises(RuntimeError, match="Render attempts exhausted"):
        RenderTask().run(
            {
                "input_usd_path": str(_make_stage(tmp_path / "scene.usda")),
                "output_base_path": str(tmp_path / "renders"),
                "render_config": {"backend": "remote", "max_attempts": 0},
            }
        )
