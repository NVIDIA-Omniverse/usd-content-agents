# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Validation Agent in-run USD rendering helpers."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image as PILImage
from pydantic import BaseModel

from world_understanding.functions.graphics.render_valid_adapter import (
    run_render_valid_adapter,
)
from world_understanding.validation.usd_rendering import (
    _ASSET_KEY_DIGEST_CHARS,
    _ASSET_KEY_STEM_CHARS,
    DEFAULT_RUNTIME_RENDER_VIEWS,
    _add_view_camera,
    _asset_path_component,
    _camera_projection,
    _create_render_backend,
    _create_rendering_backend_from_factory,
    _entry_image_path,
    _entry_images,
    _image_artifact_issue,
    _isolate_render_prims,
    _json_scalar,
    _json_value,
    _optional_bool,
    _optional_float,
    _optional_int,
    _optional_string_sequence,
    _reset_asset_output_dir,
    _same_file,
    _save_entry_images,
    _view_spec,
    expand_runtime_render_views,
    render_usd_visual_evidence,
)


class _ExternalRenderModel(BaseModel):
    path: Path


class _UnsupportedRenderValue:
    def __str__(self) -> str:
        return "unsupported-render-value"


def _write_test_usd(path: Path) -> Path:
    path.write_text(
        "\n".join(
            (
                "#usda 1.0",
                'def Xform "World" {',
                '  def Cube "Cube" {}',
                "}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def test_json_value_preserves_json_shapes_and_stringifies_unsupported_values() -> None:
    external_model = _ExternalRenderModel(path=Path("model.usda"))
    assert _json_value(
        {
            "asset": Path("assets/example.usd"),
            7: [Path("textures/albedo.png"), {"enabled": True}],
            "scalars": [None, "label", 3, 1.5, False],
            "model": external_model,
            "unsupported": _UnsupportedRenderValue(),
        }
    ) == {
        "asset": str(Path("assets/example.usd")),
        "7": [str(Path("textures/albedo.png")), {"enabled": True}],
        "scalars": [None, "label", 3, 1.5, False],
        "model": str(external_model),
        "unsupported": "unsupported-render-value",
    }


def _write_sublayered_test_usd(path: Path) -> Path:
    sublayer = path.with_name("geometry.usda")
    sublayer.write_text(
        "\n".join(
            (
                "#usda 1.0",
                'def Xform "World" {',
                '  def Cube "Cube" {}',
                "}",
                'def Shader "Shader" {',
                "  asset info:mdl:sourceAsset = @./Material/OmniPBR.mdl@",
                "}",
                "",
            )
        ),
        encoding="utf-8",
    )
    path.write_text(
        "\n".join(
            (
                "#usda 1.0",
                "(",
                '  upAxis = "Y"',
                "  metersPerUnit = 0.01",
                "  subLayers = [",
                "    @geometry.usda@",
                "  ]",
                ")",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def _valid_image() -> PILImage.Image:
    image = PILImage.new("RGB", (64, 64), (255, 255, 255))
    for x in range(32):
        for y in range(32):
            image.putpixel((x, y), (255, 0, 0))
            image.putpixel((x + 32, y), (0, 255, 0))
            image.putpixel((x, y + 32), (0, 0, 255))
            image.putpixel((x + 32, y + 32), (255, 255, 0))
    return image


def _install_backend_factory(
    monkeypatch: pytest.MonkeyPatch,
    expected_backend_name: str,
    backend_class: type[Any],
) -> None:
    def create_backend(backend_name: str, config: dict[str, Any]) -> Any:
        assert backend_name == expected_backend_name
        return backend_class(**config)

    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering."
        "_create_rendering_backend_from_factory",
        create_backend,
    )


def test_create_rendering_backend_delegates_to_shared_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()

    def create_backend(backend_name: object, config: object) -> object:
        assert backend_name == "ovrtx"
        assert config == {"render_mode": "rt2"}
        return sentinel

    monkeypatch.setattr(
        "world_understanding.functions.graphics.rendering_backend_factory."
        "create_rendering_backend",
        create_backend,
    )

    assert (
        _create_rendering_backend_from_factory("ovrtx", {"render_mode": "rt2"})
        is sentinel
    )


def test_expand_runtime_render_views_treats_string_as_single_view() -> None:
    assert expand_runtime_render_views("fixed_6") == (
        "+x",
        "-x",
        "+y",
        "-y",
        "+z",
        "-z",
    )
    assert expand_runtime_render_views("review_6") == (
        "+x",
        "-x",
        "+y",
        "-y",
        "+z",
        "+x+y+z",
    )


def test_expand_runtime_render_views_skips_empty_views_and_defaults() -> None:
    assert expand_runtime_render_views(("", "front")) == ("front",)
    assert expand_runtime_render_views(()) == DEFAULT_RUNTIME_RENDER_VIEWS


def test_view_camera_filmback_matches_render_aspect_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_side_camera(stage: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "world_understanding.utils.usd.camera.add_side_view_camera",
        fake_side_camera,
    )

    _add_view_camera(
        object(),
        label="front",
        direction="+y",
        image_width=1800,
        image_height=1200,
    )

    assert captured["horizontal_aperture"] == pytest.approx(36.0)
    assert captured["vertical_aperture"] == pytest.approx(24.0)


def test_view_camera_uses_explicit_focus_prim_and_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    focus_prim = object()

    def fake_focused_side_camera(prim: object, **kwargs: object) -> None:
        captured["prim"] = prim
        captured.update(kwargs)

    monkeypatch.setattr(
        "world_understanding.utils.usd.camera.add_focused_side_view_camera",
        fake_focused_side_camera,
    )
    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering._expand_camera_clipping_to_visible_geometry",
        lambda *_args: None,
    )

    _add_view_camera(
        object(),
        label="interaction",
        direction="+x",
        image_width=1200,
        image_height=1200,
        focus_prim=focus_prim,
        margin=3.5,
    )

    assert captured["prim"] is focus_prim
    assert captured["margin"] == pytest.approx(3.5)


def test_view_camera_orthographic_filmback_preserves_render_aspect_ratio() -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage, "/World/Cube")

    camera_path = _add_view_camera(
        stage,
        label="front",
        direction="+y",
        image_width=1800,
        image_height=900,
        margin=1.2,
        projection="orthographic",
    )

    camera = UsdGeom.Camera.Get(stage, camera_path)
    assert camera.GetProjectionAttr().Get() == UsdGeom.Tokens.orthographic
    horizontal = float(camera.GetHorizontalApertureAttr().Get())
    vertical = float(camera.GetVerticalApertureAttr().Get())
    assert horizontal / vertical == pytest.approx(2.0)
    assert horizontal >= 48.0


def test_camera_projection_policy_rejects_unknown_values() -> None:
    assert _camera_projection({}) == "perspective"
    assert _camera_projection({"render_camera_projection": "ORTHOGRAPHIC"}) == (
        "orthographic"
    )
    with pytest.raises(ValueError, match="render_camera_projection"):
        _camera_projection({"render_camera_projection": "fisheye"})


def test_isolate_render_prims_hides_geometry_outside_selected_subtree() -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    keep = UsdGeom.Cube.Define(stage, "/World/Keep/Cube").GetPrim()
    drop = UsdGeom.Cube.Define(stage, "/World/Drop/Cube").GetPrim()

    _isolate_render_prims(stage, ("/World/Keep",))

    assert UsdGeom.Imageable(keep).ComputeVisibility() != UsdGeom.Tokens.invisible
    assert UsdGeom.Imageable(drop).ComputeVisibility() == UsdGeom.Tokens.invisible


def test_optional_bool_parses_bool_like_policy_values() -> None:
    assert _optional_bool({"value": True}, "value", False) is True
    assert _optional_bool({"value": "TrUe"}, "value", False) is True
    assert _optional_bool({"value": "off"}, "value", True) is False
    assert _optional_bool({"value": None}, "value", False) is False


def test_optional_bool_rejects_invalid_string_policy_values() -> None:
    with pytest.raises(ValueError, match="Invalid boolean policy value value=''"):
        _optional_bool({"value": ""}, "value", True)


def test_optional_bool_rejects_invalid_non_bool_policy_values() -> None:
    with pytest.raises(ValueError, match="Invalid boolean policy value value=1"):
        _optional_bool({"value": 1}, "value", False)


def test_focus_render_policy_values_are_strict() -> None:
    assert _optional_float({"margin": 2}, "margin", 1.2) == pytest.approx(2.0)
    assert _optional_string_sequence({"parts": "/World/A"}, "parts") == ("/World/A",)
    assert _optional_string_sequence({"parts": ["/World/A", "/World/B"]}, "parts") == (
        "/World/A",
        "/World/B",
    )
    with pytest.raises(ValueError, match="must be absolute"):
        _optional_string_sequence({"parts": ["World/A"]}, "parts")
    with pytest.raises(ValueError, match="must be positive"):
        _optional_float({"margin": 0}, "margin", 1.2)
    with pytest.raises(ValueError, match="must be nonnegative"):
        _optional_float({"intensity": -1}, "intensity", 1.0, allow_zero=True)

    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="must be positive"):
            _optional_float({"margin": value}, "margin", 1.2)
        with pytest.raises(ValueError, match="must be nonnegative"):
            _optional_float({"intensity": value}, "intensity", 1.0, allow_zero=True)


def test_render_usd_visual_evidence_requires_explicit_backend_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)
    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering."
        "_create_rendering_backend_from_factory",
        lambda *_args, **_kwargs: pytest.fail("factory must not be called"),
    )
    working_dir = tmp_path / "run"

    result = render_usd_visual_evidence(
        usd_paths=[tmp_path / "missing.usda"],
        working_dir=working_dir,
        policy={},
    )

    assert result["status"] == "failed"
    assert result["backend"] is None
    assert result["issues"][0]["code"] == "render.backend_required"
    assert result["issues"][0]["details"] == {
        "reason": "backend_not_selected",
        "render_backend": None,
        "supported_render_backends": ["remote", "ovrtx"],
    }
    assert result["image_paths"] == []
    assert not working_dir.exists()


def test_render_usd_visual_evidence_reports_unavailable_without_remote_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)
    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering."
        "_create_rendering_backend_from_factory",
        lambda *_args, **_kwargs: pytest.fail("factory must not be called"),
    )

    result = render_usd_visual_evidence(
        usd_paths=[tmp_path / "missing.usda"],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote"},
    )

    assert result["status"] == "unavailable"
    assert result["backend"] == "remote"
    assert result["issues"][0]["code"] == "render.renderer_unavailable"
    assert result["issues"][0]["details"] == {
        "render_backend": "remote",
        "required_env": ["RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID"],
    }
    assert result["image_paths"] == []


def test_render_usd_visual_evidence_reports_renderer_import_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    def fail_factory(_backend_name: str, _config: object) -> object:
        raise ImportError("renderer module is unavailable")

    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering."
        "_create_rendering_backend_from_factory",
        fail_factory,
    )

    result = render_usd_visual_evidence(
        usd_paths=[tmp_path / "asset.usda"],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote"},
    )

    assert result["status"] == "unavailable"
    assert result["issues"][0]["code"] == "render.renderer_unavailable"
    assert (
        result["issues"][0]["message"]
        == "Remote rendering backend dependencies are required for Validation Agent "
        "in-run rendering."
    )


def test_render_usd_visual_evidence_reports_remote_backend_init_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FailingRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            assert base_url == "http://renderer.example"
            raise RuntimeError("renderer setup failed")

    _install_backend_factory(monkeypatch, "remote", FailingRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[tmp_path / "asset.usda"],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote"},
    )

    assert result["status"] == "unavailable"
    assert result["backend"] == "remote"
    assert result["issues"][0]["code"] == "render.renderer_unavailable"
    assert (
        result["issues"][0]["message"]
        == "Remote rendering backend is unavailable: renderer setup failed"
    )
    assert result["issues"][0]["details"] == {"exception_type": "RuntimeError"}


def test_create_remote_render_backend_applies_no_redirect_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeRemoteRenderingBackend:
        def __init__(
            self,
            *,
            base_url: str,
            api_key: str | None = None,
            allow_redirects: bool = True,
            num_sensor_updates: int | None = None,
            render_mode: str | None = None,
        ) -> None:
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["allow_redirects"] = allow_redirects
            captured["num_sensor_updates"] = num_sensor_updates
            captured["render_mode"] = render_mode

    monkeypatch.setattr(
        "world_understanding.functions.graphics.rendering_backend_factory."
        "RemoteRenderingBackend",
        FakeRemoteRenderingBackend,
    )

    backend = _create_render_backend(
        "remote",
        {
            "render_base_url": "http://renderer.example",
            "render_api_key": "",
            "render_allow_redirects": False,
            "render_ovrtx_num_sensor_updates": 64,
            "render_ovrtx_mode": "pt",
        },
    )

    assert isinstance(backend, FakeRemoteRenderingBackend)
    assert captured == {
        "base_url": "http://renderer.example",
        "api_key": "",
        "allow_redirects": False,
        "num_sensor_updates": 64,
        "render_mode": "pt",
    }


def test_render_usd_visual_evidence_reports_ovrtx_renderer_import_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_factory(_backend_name: str, _config: object) -> object:
        raise ImportError("renderer module is unavailable")

    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering."
        "_create_rendering_backend_from_factory",
        fail_factory,
    )

    result = render_usd_visual_evidence(
        usd_paths=[tmp_path / "asset.usda"],
        working_dir=tmp_path / "run",
        policy={"render_backend": "ovrtx"},
    )

    assert result["status"] == "unavailable"
    assert result["backend"] == "ovrtx"
    assert result["issues"][0]["code"] == "render.renderer_unavailable"
    assert (
        result["issues"][0]["message"]
        == "OVRTX rendering backend dependencies are required for Validation Agent "
        "in-run rendering."
    )
    assert "exception_type" in result["issues"][0]["details"]


def test_render_usd_visual_evidence_reports_ovrtx_transitive_import_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_factory(_backend_name: str, _config: object) -> object:
        raise ImportError("missing transitive renderer dependency")

    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering."
        "_create_rendering_backend_from_factory",
        fail_factory,
    )

    result = render_usd_visual_evidence(
        usd_paths=[tmp_path / "asset.usda"],
        working_dir=tmp_path / "run",
        policy={"render_backend": "ovrtx"},
    )

    assert result["status"] == "unavailable"
    assert result["backend"] == "ovrtx"
    assert result["issues"][0]["code"] == "render.renderer_unavailable"
    assert result["issues"][0]["details"] == {"exception_type": "ImportError"}


def test_render_usd_visual_evidence_reports_missing_backend_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_factory(_backend_name: str, _config: object) -> object:
        raise AttributeError("factory entry point is unavailable")

    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering."
        "_create_rendering_backend_from_factory",
        fail_factory,
    )

    result = render_usd_visual_evidence(
        usd_paths=[tmp_path / "asset.usda"],
        working_dir=tmp_path / "run",
        policy={"render_backend": "ovrtx"},
    )

    assert result["status"] == "unavailable"
    assert result["backend"] == "ovrtx"
    assert result["issues"][0]["code"] == "render.renderer_unavailable"
    assert result["issues"][0]["details"] == {"exception_type": "AttributeError"}


def test_render_usd_visual_evidence_reports_ovrtx_backend_init_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingOvRTXRenderingBackend:
        def __init__(
            self,
            *,
            log_level: str,
            num_sensor_updates: int,
            render_mode: str,
        ) -> None:
            assert log_level == "warn"
            assert num_sensor_updates == 32
            assert render_mode == "rt2"
            raise RuntimeError("ovrtx setup failed")

    _install_backend_factory(monkeypatch, "ovrtx", FailingOvRTXRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[tmp_path / "asset.usda"],
        working_dir=tmp_path / "run",
        policy={"render_backend": "ovrtx"},
    )

    assert result["status"] == "unavailable"
    assert result["backend"] == "ovrtx"
    assert result["issues"][0]["code"] == "render.renderer_unavailable"
    assert (
        result["issues"][0]["message"]
        == "OVRTX rendering backend is unavailable: ovrtx setup failed"
    )
    assert result["issues"][0]["details"] == {"exception_type": "RuntimeError"}


def test_render_usd_visual_evidence_writes_stubbed_renderer_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            assert self.base_url == "http://renderer.example"
            assert cameras == ["/ValidationAgentCameras/front"]
            assert image_width == 128
            assert image_height == 96
            assert frames == "0"
            assert Path(str(base_dir)) == tmp_path
            return {
                "results": [
                    {
                        "camera": cameras[0],
                        "images": [_valid_image()],
                        "frame_count": 1,
                        "status": "success",
                        "ovrtx_render_mode": "pt",
                        "ovrtx_num_sensor_updates": 64,
                        "active_aov": "LdrColor",
                    }
                ]
            }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={
            "render_backend": "remote",
            "expected_cameras": ["front"],
            "render_image_width": 128,
            "render_image_height": 96,
        },
    )

    assert result["status"] == "completed"
    assert result["backend"] == "remote"
    assert result["metadata"]["backend"] == "remote"
    image_paths = result["image_paths"]
    assert len(image_paths) == 1
    assert Path(image_paths[0]).is_file()
    assert result["render_response"]["results"][0]["camera"] == "front"
    assert result["render_response"]["results"][0]["images"] == image_paths
    assert result["render_response"]["results"][0]["ovrtx_render_mode"] == "pt"
    assert result["render_response"]["results"][0]["ovrtx_num_sensor_updates"] == 64
    assert result["render_response"]["results"][0]["active_aov"] == "LdrColor"


def test_render_usd_visual_evidence_frames_isolated_context_instead_of_focus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    usd_path = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Cube.Define(stage, "/World/Keep/Cube")
    UsdGeom.Cube.Define(stage, "/World/Drop/Cube")
    stage.GetRootLayer().Save()
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")
    captured_focus: list[object | None] = []

    def add_camera(
        _stage: object,
        *,
        label: str,
        focus_prim: object | None,
        **_kwargs: object,
    ) -> str:
        captured_focus.append(focus_prim)
        return f"/ValidationAgentCameras/{label}"

    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering._add_view_camera",
        add_camera,
    )

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            assert base_url == "http://renderer.example"

        def render(
            self, prepared_stage: object, **_kwargs: object
        ) -> dict[str, object]:
            assert isinstance(prepared_stage, Usd.Stage)
            dropped = prepared_stage.GetPrimAtPath("/World/Drop/Cube")
            assert (
                UsdGeom.Imageable(dropped).ComputeVisibility()
                == UsdGeom.Tokens.invisible
            )
            return {
                "results": [
                    {
                        "camera": "/ValidationAgentCameras/front",
                        "images": [_valid_image()],
                        "frame_count": 1,
                        "status": "success",
                    }
                ]
            }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={
            "render_backend": "remote",
            "expected_cameras": ["front"],
            "render_focus_prim_path": "/World/Keep",
            "render_isolate_prim_paths": "/World/Keep",
            "render_frame_isolated_context": True,
        },
    )

    assert result["status"] == "completed"
    assert captured_focus == [None]
    assert result["metadata"]["stage_preparation"][0] == {
        "usd_path": str(usd_path),
        "usd_sha256": hashlib.sha256(usd_path.read_bytes()).hexdigest(),
        "backend": "remote",
        "flattened": True,
        "material_normalized": True,
        "asset_base_dir": str(tmp_path),
        "up_axis": "Y",
        "meters_per_unit": 0.01,
        "isolate_prim_paths": ["/World/Keep"],
        "focus_prim_path": "/World/Keep",
        "frame_isolated_context": True,
    }


def test_render_usd_visual_evidence_rejects_missing_focus_prim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            assert base_url == "http://renderer.example"

        def render(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("render must not run with a missing focus prim")

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={
            "render_backend": "remote",
            "render_focus_prim_path": "/World/Missing",
        },
    )

    assert result["status"] == "failed"
    assert result["issues"][0]["code"] == "render.runtime_render_failed"
    assert "render focus prim does not exist" in result["issues"][0]["message"]


def test_asset_path_component_stays_compact_for_verbose_render_artifacts(
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "generated_agent_output_with_a_verbose_pipeline_name.usd"
    usd_path.touch()

    asset_key = _asset_path_component(usd_path, 0)
    expected_max = len("000_") + _ASSET_KEY_STEM_CHARS + 1 + _ASSET_KEY_DIGEST_CHARS

    assert asset_key.startswith("000_")
    assert len(asset_key) <= expected_max
    assert "generated_agent_" in asset_key
    assert usd_path.stem not in asset_key


def test_asset_path_component_keeps_large_indices_compact(tmp_path: Path) -> None:
    usd_path = tmp_path / "asset.usd"
    usd_path.touch()

    asset_key = _asset_path_component(usd_path, 1000)
    expected_max = len("1000_") + _ASSET_KEY_STEM_CHARS + 1 + _ASSET_KEY_DIGEST_CHARS

    assert asset_key.startswith("1000_asset_")
    assert len(asset_key) <= expected_max


def test_render_usd_visual_evidence_replaces_stale_asset_render_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    working_dir = tmp_path / "run"
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            return {
                "results": [
                    {
                        "camera": camera,
                        "images": [_valid_image()],
                        "frame_count": 1,
                        "status": "success",
                    }
                    for camera in cameras
                ]
            }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    first = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=working_dir,
        policy={
            "render_backend": "remote",
            "expected_cameras": ["front", "right"],
        },
    )
    assert first["status"] == "completed"
    first_render_root = Path(first["render_output_dir"])
    assert list(first_render_root.rglob("*_right_*.png"))

    second = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=working_dir,
        policy={"render_backend": "remote", "expected_cameras": ["front"]},
    )

    assert second["status"] == "completed"
    second_render_root = Path(second["render_output_dir"])
    assert first_render_root == second_render_root
    assert len(second["image_paths"]) == 1
    assert not list(second_render_root.rglob("*_right_*.png"))


def test_render_usd_visual_evidence_flattens_stage_before_remote_render(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_sublayered_test_usd(tmp_path / "asset.usda")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            from pxr import Sdf, Usd

            assert isinstance(stage, Usd.Stage)
            assert Path(str(base_dir)) == tmp_path
            assert '"Cube"' in stage.GetRootLayer().ExportToString()
            assert stage.GetPrimAtPath("/ValidationAgentCameras/front").IsValid()
            mdl_attr = stage.GetPrimAtPath("/Shader").GetAttribute(
                "info:mdl:sourceAsset"
            )
            assert mdl_attr.Get() == Sdf.AssetPath("OmniPBR.mdl")
            return {
                "results": [
                    {
                        "camera": cameras[0],
                        "images": [_valid_image()],
                        "frame_count": 1,
                        "status": "success",
                    }
                ]
            }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote", "expected_cameras": ["front"]},
    )

    assert result["status"] == "completed"
    assert result["metadata"]["stage_preparation"] == [
        {
            "usd_path": str(usd_path),
            "usd_sha256": hashlib.sha256(usd_path.read_bytes()).hexdigest(),
            "backend": "remote",
            "flattened": True,
            "material_normalized": True,
            "asset_base_dir": str(tmp_path),
            "up_axis": "Y",
            "meters_per_unit": 0.01,
        }
    ]


def test_render_usd_visual_evidence_can_add_neutral_studio_lighting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            from pxr import Usd

            assert isinstance(stage, Usd.Stage)
            assert stage.GetPrimAtPath("/__WUStudioLighting/Dome").IsValid()
            assert stage.GetPrimAtPath("/__WUStudioLighting/Key").IsValid()
            assert stage.GetPrimAtPath("/__WUStudioLighting/Fill").IsValid()
            assert stage.GetPrimAtPath("/__WUStudioLighting/Rim").IsValid()
            return {
                "results": [
                    {
                        "camera": cameras[0],
                        "images": [_valid_image()],
                        "frame_count": 1,
                        "status": "success",
                    }
                ]
            }

    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering._create_rendering_backend_from_factory",
        lambda _backend, config: FakeRemoteRenderingBackend(
            base_url=str(config["base_url"])
        ),
    )

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={
            "render_backend": "remote",
            "expected_cameras": ["front"],
            "render_studio_lighting": True,
        },
    )

    assert result["status"] == "completed"
    assert result["metadata"]["stage_preparation"][0]["studio_lighting"] is True
    assert result["metadata"]["stage_preparation"][0]["studio_dome"] is True
    assert result["metadata"]["stage_preparation"][0]["studio_dome_intensity"] == 350.0


def test_neutral_studio_lighting_can_omit_visible_dome(tmp_path: Path) -> None:
    from pxr import Usd

    from world_understanding.validation.usd_rendering import (
        _add_neutral_studio_lighting,
    )

    stage = Usd.Stage.Open(str(_write_test_usd(tmp_path / "asset.usda")))
    _add_neutral_studio_lighting(stage, include_dome=False)

    assert not stage.GetPrimAtPath("/__WUStudioLighting/Dome").IsValid()
    assert stage.GetPrimAtPath("/__WUStudioLighting/Key").IsValid()
    assert stage.GetPrimAtPath("/__WUStudioLighting/Fill").IsValid()
    assert stage.GetPrimAtPath("/__WUStudioLighting/Rim").IsValid()


def test_neutral_studio_lighting_accepts_bounded_dome_intensity(tmp_path: Path) -> None:
    from pxr import Usd, UsdLux

    from world_understanding.validation.usd_rendering import (
        _add_neutral_studio_lighting,
    )

    stage = Usd.Stage.Open(str(_write_test_usd(tmp_path / "asset.usda")))
    _add_neutral_studio_lighting(stage, dome_intensity=60.0)

    dome = UsdLux.DomeLight.Get(stage, "/__WUStudioLighting/Dome")
    assert dome.GetIntensityAttr().Get() == 60.0


def test_neutral_studio_lighting_replaces_existing_rig(tmp_path: Path) -> None:
    from pxr import Usd, UsdLux

    from world_understanding.validation.usd_rendering import (
        _add_neutral_studio_lighting,
    )

    stage = Usd.Stage.Open(str(_write_test_usd(tmp_path / "asset.usda")))
    _add_neutral_studio_lighting(stage, dome_intensity=60.0)
    _add_neutral_studio_lighting(stage, dome_intensity=25.0)

    dome = UsdLux.DomeLight.Get(stage, "/__WUStudioLighting/Dome")
    assert dome.GetIntensityAttr().Get() == 25.0


def test_stage_preparation_allows_zero_to_disable_reflection_light(
    tmp_path: Path,
) -> None:
    from pxr import Usd

    from world_understanding.validation.usd_rendering import _prepare_stage_for_render

    stage = Usd.Stage.Open(str(_write_test_usd(tmp_path / "asset.usda")))
    prepared, metadata = _prepare_stage_for_render(
        stage,
        backend_name="ovrtx",
        policy={
            "render_studio_lighting": True,
            "render_studio_reflection_safe": True,
            "render_studio_reflection_intensity": 0,
        },
    )

    assert metadata["studio_reflection_intensity"] == 0.0
    assert not prepared.GetPrimAtPath("/__WUStudioLighting/TopReflection").IsValid()


def test_reflection_safe_studio_lighting_hides_reflection_sources(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdLux

    from world_understanding.validation.usd_rendering import (
        _add_neutral_studio_lighting,
    )

    stage = Usd.Stage.Open(str(_write_test_usd(tmp_path / "asset.usda")))
    _add_neutral_studio_lighting(
        stage,
        dome_intensity=20.0,
        reflection_safe=True,
        hdri_intensity=600.0,
        reflection_intensity=100.0,
    )

    dome = UsdLux.DomeLight.Get(stage, "/__WUStudioLighting/Dome")
    environment = UsdLux.DomeLight.Get(
        stage,
        "/__WUStudioLighting/Environment",
    )
    reflection = UsdLux.DistantLight.Get(
        stage,
        "/__WUStudioLighting/TopReflection",
    )
    assert dome.GetIntensityAttr().Get() == 20.0
    assert environment.GetIntensityAttr().Get() == 600.0
    assert environment.GetPrim().GetAttribute("inputs:texture:file").Get()
    assert environment.GetPrim().GetAttribute("visibleInPrimaryRay").Get() is False
    assert reflection.GetIntensityAttr().Get() == 100.0
    assert reflection.GetPrim().GetAttribute("visibleInPrimaryRay").Get() is False


def test_focused_camera_clipping_includes_visible_context_depth() -> None:
    from pxr import Gf, Usd, UsdGeom

    from world_understanding.validation.usd_rendering import _add_view_camera

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    focus = UsdGeom.Cube.Define(stage, "/Scene/Focus")
    context = UsdGeom.Cube.Define(stage, "/Scene/Context")
    context.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -8.0))

    camera_path = _add_view_camera(
        stage,
        label="top",
        direction="+z",
        image_width=800,
        image_height=600,
        focus_prim=focus.GetPrim(),
        margin=1.2,
    )

    camera = UsdGeom.Camera.Get(stage, camera_path)
    transform = UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    camera_position = transform.ExtractTranslation()
    view_direction = transform.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
    context_range = (
        UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_],
        )
        .ComputeWorldBound(context.GetPrim())
        .ComputeAlignedRange()
    )
    maximum_context_depth = max(
        Gf.Dot(
            Gf.Vec3d(x, y, z) - camera_position,
            view_direction,
        )
        for x in (context_range.GetMin()[0], context_range.GetMax()[0])
        for y in (context_range.GetMin()[1], context_range.GetMax()[1])
        for z in (context_range.GetMin()[2], context_range.GetMax()[2])
    )

    assert camera.GetClippingRangeAttr().Get()[1] > maximum_context_depth


def test_camera_framing_helpers_ignore_empty_bounds_and_missing_cameras() -> None:
    from pxr import Usd, UsdGeom

    from world_understanding.validation.usd_rendering import (
        _expand_camera_clipping_to_visible_geometry,
        _set_orthographic_camera_framing,
    )

    empty_stage = Usd.Stage.CreateInMemory()
    _set_orthographic_camera_framing(
        empty_stage,
        "/MissingCamera",
        focus_prim=None,
        image_width=800,
        image_height=600,
        margin=1.2,
    )
    _expand_camera_clipping_to_visible_geometry(empty_stage, "/MissingCamera")

    bounded_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(bounded_stage, "/World/Cube")
    _set_orthographic_camera_framing(
        bounded_stage,
        "/MissingCamera",
        focus_prim=None,
        image_width=800,
        image_height=600,
        margin=1.2,
    )
    _expand_camera_clipping_to_visible_geometry(bounded_stage, "/MissingCamera")


def test_camera_clipping_ignores_geometry_behind_camera() -> None:
    from pxr import Gf, Usd, UsdGeom

    from world_understanding.validation.usd_rendering import (
        _expand_camera_clipping_to_visible_geometry,
    )

    stage = Usd.Stage.CreateInMemory()
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 10.0))
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    before = camera.GetClippingRangeAttr().Get()

    _expand_camera_clipping_to_visible_geometry(stage, "/Camera")

    assert camera.GetClippingRangeAttr().Get() == before


def test_render_usd_visual_evidence_reports_missing_renderer_image_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    missing_image_path = tmp_path / "renderer-cache" / "front.png"
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            assert self.base_url == "http://renderer.example"
            return {
                "results": [
                    {
                        "camera": cameras[0],
                        "images": [missing_image_path],
                        "frame_count": 1,
                        "status": "success",
                    }
                ]
            }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote", "expected_cameras": ["front"]},
    )

    assert result["status"] == "failed"
    assert result["image_paths"] == []
    assert result["render_response"]["results"][0]["images"] == []
    issue_codes = [issue["code"] for issue in result["issues"]]
    assert "render.missing_output" in issue_codes
    assert "render.no_images_generated" in issue_codes
    missing_issue = next(
        issue for issue in result["issues"] if issue["code"] == "render.missing_output"
    )
    assert missing_issue["subject"] == str(missing_image_path)
    assert missing_issue["details"]["source_path"] == str(missing_image_path)


def test_render_usd_visual_evidence_copies_renderer_image_file_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    renderer_image_path = tmp_path / "renderer-cache" / "front.png"
    renderer_image_path.parent.mkdir()
    _valid_image().save(renderer_image_path)
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            assert self.base_url == "http://renderer.example"
            return {
                "results": [
                    {
                        "camera": cameras[0],
                        "image_files": [renderer_image_path],
                        "frame_count": 1,
                        "status": "success",
                    }
                ]
            }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote", "expected_cameras": ["front"]},
    )

    assert result["status"] == "completed"
    image_paths = result["image_paths"]
    assert len(image_paths) == 1
    copied_path = Path(image_paths[0])
    assert copied_path.is_file()
    assert copied_path != renderer_image_path
    assert result["render_response"]["results"][0]["images"] == image_paths


def test_render_usd_visual_evidence_skips_null_image_key_for_file_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    renderer_image_path = tmp_path / "renderer-cache" / "front.png"
    renderer_image_path.parent.mkdir()
    _valid_image().save(renderer_image_path)
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            assert self.base_url == "http://renderer.example"
            return {
                "results": [
                    {
                        "camera": cameras[0],
                        "images": None,
                        "image_files": [renderer_image_path],
                        "frame_count": 1,
                        "status": "success",
                    }
                ]
            }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote", "expected_cameras": ["front"]},
    )

    assert result["status"] == "completed"
    image_paths = result["image_paths"]
    assert len(image_paths) == 1
    assert Path(image_paths[0]).is_file()
    assert result["render_response"]["results"][0]["images"] == image_paths


def test_render_usd_visual_evidence_reports_blank_renderer_image_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            assert self.base_url == "http://renderer.example"
            return {
                "results": [
                    {
                        "camera": cameras[0],
                        "images": ["  "],
                        "frame_count": 1,
                        "status": "success",
                    }
                ]
            }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote", "expected_cameras": ["front"]},
    )

    assert result["status"] == "failed"
    assert result["image_paths"] == []
    missing_issue = next(
        issue for issue in result["issues"] if issue["code"] == "render.missing_output"
    )
    assert missing_issue["message"] == "Renderer reported a blank image artifact path."
    assert missing_issue["subject"].endswith("_front_0000.png")
    assert missing_issue["details"]["reported_source_path"] == "  "
    assert "source_path" not in missing_issue["details"]


def test_render_usd_visual_evidence_empty_images_key_preserves_missing_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    ignored_image_path = tmp_path / "renderer-cache" / "front.png"
    ignored_image_path.parent.mkdir()
    _valid_image().save(ignored_image_path)
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            assert self.base_url == "http://renderer.example"
            return {
                "results": [
                    {
                        "camera": cameras[0],
                        "images": [],
                        "image_files": [ignored_image_path],
                        "frame_count": 0,
                        "status": "success",
                    }
                ]
            }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote", "expected_cameras": ["front"]},
    )

    assert result["status"] == "failed"
    assert result["image_paths"] == []
    assert [issue["code"] for issue in result["issues"]] == [
        "render.missing_view_evidence",
        "render.no_images_generated",
    ]


def test_render_usd_visual_evidence_uses_local_ovrtx_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)

    class FakeOvRTXRenderingBackend:
        def __init__(
            self,
            *,
            log_level: str,
            num_sensor_updates: int,
            render_mode: str,
        ) -> None:
            assert log_level == "error"
            assert num_sensor_updates == 64
            assert render_mode == "pt"

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            assert cameras == ["/ValidationAgentCameras/front"]
            assert image_width == 128
            assert image_height == 96
            assert frames == "0"
            assert Path(str(base_dir)) == tmp_path
            return {
                "results": [
                    {
                        "camera": cameras[0],
                        "images": [_valid_image()],
                        "frame_count": 1,
                        "status": "success",
                    }
                ]
            }

    _install_backend_factory(monkeypatch, "ovrtx", FakeOvRTXRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={
            "render_backend": " OVRTX ",
            "render_ovrtx_log_level": "error",
            "render_ovrtx_num_sensor_updates": "64",
            "render_ovrtx_mode": "pt",
            "expected_cameras": ["front"],
            "render_image_width": 128,
            "render_image_height": 96,
        },
    )

    assert result["status"] == "completed"
    assert result["backend"] == "ovrtx"
    assert result["metadata"]["backend"] == "ovrtx"
    assert result["metadata"]["base_url_configured"] is False
    image_paths = result["image_paths"]
    assert len(image_paths) == 1
    assert Path(image_paths[0]).is_file()


def test_render_usd_visual_evidence_blank_backend_selects_no_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering."
        "_create_rendering_backend_from_factory",
        lambda *_args, **_kwargs: pytest.fail("factory must not be called"),
    )
    working_dir = tmp_path / "run"

    result = render_usd_visual_evidence(
        usd_paths=[tmp_path / "asset.usda"],
        working_dir=working_dir,
        policy={"render_backend": "   ", "expected_cameras": ["front"]},
    )

    assert result["status"] == "failed"
    assert result["backend"] is None
    assert result["issues"][0]["code"] == "render.backend_required"
    assert not working_dir.exists()


def test_render_usd_visual_evidence_expands_fixed_six_view_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    expected_cameras = [
        "/ValidationAgentCameras/plus_x",
        "/ValidationAgentCameras/minus_x",
        "/ValidationAgentCameras/plus_y",
        "/ValidationAgentCameras/minus_y",
        "/ValidationAgentCameras/plus_z",
        "/ValidationAgentCameras/minus_z",
    ]
    side_directions: list[str] = []
    corner_directions: list[str] = []

    def fake_side_camera(
        stage: object, *, camera_path: str, direction: str, **kwargs: object
    ) -> None:
        side_directions.append(direction)

    def fake_corner_camera(
        stage: object, *, camera_path: str, direction: str, **kwargs: object
    ) -> None:
        corner_directions.append(direction)

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            assert cameras == expected_cameras
            return {
                "results": [
                    {
                        "camera": camera,
                        "images": [_valid_image()],
                        "frame_count": 1,
                        "status": "success",
                    }
                    for camera in cameras
                ]
            }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)
    monkeypatch.setattr(
        "world_understanding.utils.usd.camera.add_side_view_camera",
        fake_side_camera,
    )
    monkeypatch.setattr(
        "world_understanding.utils.usd.camera.add_corner_view_camera",
        fake_corner_camera,
    )

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote", "expected_cameras": ["fixed_6"]},
    )

    assert result["status"] == "completed"
    assert result["metadata"]["views"] == ["+x", "-x", "+y", "-y", "+z", "-z"]
    assert result["metadata"]["view_count"] == 6
    assert side_directions == ["+x", "-x", "+y", "-y", "-z", "+z"]
    assert corner_directions == []
    assert len(result["image_paths"]) == 6
    assert result["render_response"]["cameras"] == [
        "+x",
        "-x",
        "+y",
        "-y",
        "+z",
        "-z",
    ]

    expected_cameras[:] = [
        "/ValidationAgentCameras/plus_x",
        "/ValidationAgentCameras/minus_x",
        "/ValidationAgentCameras/plus_y",
        "/ValidationAgentCameras/minus_y",
        "/ValidationAgentCameras/plus_z",
        "/ValidationAgentCameras/plus_xplus_yplus_z",
    ]
    side_directions.clear()
    corner_directions.clear()
    review_result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "review_run",
        policy={"render_backend": "remote", "expected_cameras": ["review_6"]},
    )

    assert review_result["status"] == "completed"
    assert review_result["metadata"]["views"] == [
        "+x",
        "-x",
        "+y",
        "-y",
        "+z",
        "+x+y+z",
    ]
    assert side_directions == ["+x", "-x", "+y", "-y", "-z"]
    assert corner_directions == ["+x+y+z"]


@pytest.mark.parametrize("backend_name", ("warp", "mock"))
def test_render_usd_visual_evidence_reports_unsupported_runtime_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_name: str,
) -> None:
    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering."
        "_create_rendering_backend_from_factory",
        lambda *_args, **_kwargs: pytest.fail("factory must not be called"),
    )
    working_dir = tmp_path / "run"

    result = render_usd_visual_evidence(
        usd_paths=[tmp_path / "asset.usda"],
        working_dir=working_dir,
        policy={"render_backend": backend_name},
    )

    assert result["status"] == "unavailable"
    assert result["backend"] == backend_name
    assert result["issues"][0]["code"] == "render.renderer_unavailable"
    assert result["issues"][0]["severity"] == "warn"
    assert result["issues"][0]["details"] == {
        "reason": "unsupported_by_validation",
        "render_backend": backend_name,
        "canonical_render_backends": ["remote", "warp", "ovrtx", "mock"],
        "supported_render_backends": ["remote", "ovrtx"],
    }
    assert not working_dir.exists()


@pytest.mark.parametrize("backend_value", ("typo", ["remote"]))
def test_render_usd_visual_evidence_fails_unknown_backend_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_value: object,
) -> None:
    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering."
        "_create_rendering_backend_from_factory",
        lambda *_args, **_kwargs: pytest.fail("factory must not be called"),
    )
    working_dir = tmp_path / "run"

    result = render_usd_visual_evidence(
        usd_paths=[tmp_path / "asset.usda"],
        working_dir=working_dir,
        policy={"render_backend": backend_value},
    )

    backend_label = backend_value if isinstance(backend_value, str) else "invalid"
    assert result["status"] == "failed"
    assert result["backend"] == backend_label
    assert result["issues"][0]["code"] == "render.backend_unknown"
    assert result["issues"][0]["severity"] == "fail"
    assert result["issues"][0]["details"] == {
        "reason": "unknown_backend",
        "render_backend": backend_value,
        "render_backend_type": type(backend_value).__name__,
        "canonical_render_backends": ["remote", "warp", "ovrtx", "mock"],
        "supported_render_backends": ["remote", "ovrtx"],
    }
    if not isinstance(backend_value, str):
        assert result["issues"][0]["message"] == (
            "Invalid rendering backend selector; expected a string value."
        )
    assert not working_dir.exists()


def test_render_usd_visual_evidence_fails_when_requested_views_lack_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            assert cameras == [
                "/ValidationAgentCameras/front",
                "/ValidationAgentCameras/back",
                "/ValidationAgentCameras/top",
            ]
            return {
                "results": [
                    {
                        "camera": cameras[0],
                        "images": [_valid_image()],
                        "frame_count": 1,
                        "status": "success",
                    },
                    {
                        "camera": cameras[1],
                        "images": [],
                        "frame_count": 0,
                        "status": "success",
                    },
                ]
            }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={
            "render_backend": "remote",
            "expected_cameras": ["front", "back", "top"],
        },
    )

    assert result["status"] == "failed"
    assert len(result["image_paths"]) == 1
    assert result["metadata"]["view_count"] == 3
    assert result["metadata"]["usd_path_count"] == 1
    assert result["render_response"]["results"][1]["images"] == []
    missing_issues = [
        issue
        for issue in result["issues"]
        if issue["code"] == "render.missing_view_evidence"
    ]
    assert [issue["details"]["view"] for issue in missing_issues] == ["back", "top"]


def test_render_usd_visual_evidence_uses_unique_paths_for_duplicate_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_usd = _write_test_usd(first_dir / "asset.usda")
    second_usd = _write_test_usd(second_dir / "asset.usda")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            assert Path(str(base_dir)) in {first_dir, second_dir}
            return {
                "results": [
                    {
                        "camera": cameras[0],
                        "images": [_valid_image()],
                        "frame_count": 1,
                        "status": "success",
                    }
                ]
            }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[first_usd, second_usd],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote", "expected_cameras": ["front"]},
    )

    assert result["status"] == "completed"
    image_paths = [Path(path) for path in result["image_paths"]]
    response_cameras = [
        entry["camera"] for entry in result["render_response"]["results"]
    ]
    assert len(image_paths) == 2
    assert len(set(image_paths)) == 2
    assert len(set(response_cameras)) == 2
    assert response_cameras == result["metadata"]["response_cameras"]
    assert image_paths[0].parent != image_paths[1].parent
    assert all(path.is_file() for path in image_paths)
    assert result["render_response"]["results"][0]["images"] == [str(image_paths[0])]
    assert result["render_response"]["results"][1]["images"] == [str(image_paths[1])]
    adapter_result = run_render_valid_adapter(
        render_response=result["render_response"],
        expected_cameras=response_cameras,
    )
    assert adapter_result["status"] == "pass"


def test_render_usd_visual_evidence_preserves_signed_view_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            assert cameras == [
                "/ValidationAgentCameras/plus_y",
                "/ValidationAgentCameras/minus_y",
            ]
            return {
                "results": [
                    {
                        "camera": camera,
                        "images": [_valid_image()],
                        "frame_count": 1,
                        "status": "success",
                    }
                    for camera in cameras
                ]
            }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote", "expected_cameras": ["+y", "-y"]},
    )

    assert result["status"] == "completed"
    image_paths = [Path(path) for path in result["image_paths"]]
    assert len(image_paths) == 2
    assert len(set(image_paths)) == 2
    assert "plus_y" in image_paths[0].name
    assert "minus_y" in image_paths[1].name


def test_render_usd_visual_evidence_reports_usd_open_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "missing.usda"
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(self, *args: object, **kwargs: object) -> dict[str, object]:
            raise AssertionError("render should not run when USD open fails")

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)
    from pxr import Usd

    monkeypatch.setattr(Usd.Stage, "Open", lambda _path: None)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote", "expected_cameras": ["front"]},
    )

    assert result["status"] == "failed"
    assert result["render_response"]["status"] == "failed"
    assert result["issues"][0]["code"] == "render.usd_open_failed"
    assert result["issues"][0]["subject"] == str(usd_path)


def test_render_usd_visual_evidence_rejects_source_replaced_during_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    usd_sha256_before_open = hashlib.sha256(usd_path.read_bytes()).hexdigest()
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(self, *args: object, **kwargs: object) -> dict[str, object]:
            raise AssertionError("render should not run with unstable USD input")

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)
    from pxr import Usd

    open_stage = Usd.Stage.Open

    def replace_after_open(path: str) -> object:
        stage = open_stage(path)
        usd_path.write_text('#usda 1.0\ndef Xform "Replacement" {}\n', encoding="utf-8")
        return stage

    monkeypatch.setattr(Usd.Stage, "Open", replace_after_open)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote", "expected_cameras": ["front"]},
    )

    assert result["status"] == "failed"
    assert result["metadata"]["stage_preparation"] == []
    assert result["issues"][0]["code"] == "render.usd_changed_during_open"
    assert result["issues"][0]["details"] == {
        "sha256_before_open": usd_sha256_before_open,
        "sha256_after_open": hashlib.sha256(usd_path.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("change_mode", ("mutate", "replace"))
def test_render_usd_visual_evidence_rejects_source_changed_during_render(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change_mode: str,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    usd_sha256_before_render = hashlib.sha256(usd_path.read_bytes()).hexdigest()
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            replacement_content = '#usda 1.0\ndef Xform "Replacement" {}\n'
            if change_mode == "mutate":
                usd_path.write_text(replacement_content, encoding="utf-8")
            else:
                replacement_path = tmp_path / "replacement.usda"
                replacement_path.write_text(replacement_content, encoding="utf-8")
                replacement_path.replace(usd_path)
            return {
                "results": [
                    {
                        "camera": cameras[0],
                        "images": [_valid_image()],
                        "frame_count": 1,
                        "status": "success",
                    }
                ]
            }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote", "expected_cameras": ["front"]},
    )

    assert result["status"] == "failed"
    assert result["image_paths"] == []
    assert result["metadata"]["render_invocation_count"] == 1
    assert result["render_response"]["cameras"] == []
    assert result["render_response"]["results"] == []
    assert result["issues"][0]["code"] == "render.usd_changed_during_render"
    assert result["issues"][0]["details"] == {
        "sha256_before_render": usd_sha256_before_render,
        "sha256_after_render": hashlib.sha256(usd_path.read_bytes()).hexdigest(),
    }
    assert not list((tmp_path / "run" / "renders").rglob("*.png"))


def test_render_usd_visual_evidence_handles_asset_output_prepare_issue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            return {
                "results": [
                    {
                        "camera": cameras[0],
                        "images": [_valid_image()],
                        "frame_count": 1,
                        "status": "success",
                    }
                ]
            }

    def fake_reset_asset_output_dir(
        output_root: Path,
        *,
        asset_key: str,
        usd_path: Path,
    ) -> tuple[Path, dict[str, object]]:
        return output_root / asset_key, {
            "code": "render.output_dir_prepare_failed",
            "severity": "fail",
            "message": "could not reset output directory",
            "details": {"asset_key": asset_key, "usd_path": str(usd_path)},
        }

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)
    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering._reset_asset_output_dir",
        fake_reset_asset_output_dir,
    )

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote", "expected_cameras": ["front"]},
    )

    assert result["status"] == "failed"
    assert result["image_paths"] == []
    assert result["issues"][0]["code"] == "render.output_dir_prepare_failed"


def test_render_usd_visual_evidence_treats_non_sequence_results_as_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = _write_test_usd(tmp_path / "asset.usda")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.example")

    class FakeRemoteRenderingBackend:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

        def render(
            self,
            stage: object,
            *,
            cameras: Sequence[str],
            image_width: int,
            image_height: int,
            frames: str,
            base_dir: str | Path | None = None,
        ) -> dict[str, object]:
            return {"results": "not-a-result-list"}

    _install_backend_factory(monkeypatch, "remote", FakeRemoteRenderingBackend)

    result = render_usd_visual_evidence(
        usd_paths=[usd_path],
        working_dir=tmp_path / "run",
        policy={"render_backend": "remote", "expected_cameras": ["front"]},
    )

    assert result["status"] == "failed"
    assert result["render_response"]["results"] == []
    assert result["issues"][0]["code"] == "render.missing_view_evidence"


def test_usd_rendering_small_helpers_cover_fallback_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unknown_backend = _create_render_backend("unsupported", {})
    assert unknown_backend["status"] == "failed"
    assert unknown_backend["issues"][0]["code"] == "render.backend_unknown"
    assert expand_runtime_render_views(42) == DEFAULT_RUNTIME_RENDER_VIEWS
    assert _view_spec("") == ("corner", "+x+y+z")
    assert _view_spec("three-quarter") == ("three-quarter", "+x+y+z")
    assert _entry_image_path("relative.png", tmp_path) == (
        tmp_path / "relative.png",
        True,
    )
    assert _entry_images({"images": "relative.png"}) == ("relative.png",)
    assert _entry_images({}) == ()
    assert _json_scalar(Path("render.png")) == "render.png"
    assert _optional_int({"value": "bad"}, "value", 128) == 128
    assert _optional_int({"value": "0"}, "value", 128) == 128

    output_root = tmp_path / "renders"
    output_root.mkdir()
    stale_file = output_root / "asset"
    stale_file.write_text("old", encoding="utf-8")
    reset_dir, issue = _reset_asset_output_dir(
        output_root,
        asset_key="asset",
        usd_path=tmp_path / "asset.usda",
    )
    assert issue is None
    assert reset_dir.is_dir()
    assert stale_file.is_dir()

    external_output = tmp_path / "external-render-output"
    external_asset = external_output / "asset"
    external_asset.mkdir(parents=True)
    sentinel = external_asset / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    linked_output_root = tmp_path / "linked-renders"
    linked_output_root.symlink_to(
        external_output,
        target_is_directory=True,
    )
    reset_dir, issue = _reset_asset_output_dir(
        linked_output_root,
        asset_key="asset",
        usd_path=tmp_path / "asset.usda",
    )
    assert reset_dir == linked_output_root / "asset"
    assert issue is not None
    assert issue["code"] == "render.output_dir_prepare_failed"
    assert sentinel.read_text(encoding="utf-8") == "keep\n"

    output_path = tmp_path / "output.png"
    output_path.write_text("existing", encoding="utf-8")
    assert _same_file(tmp_path / "missing.png", output_path) is False

    issue_with_details = _image_artifact_issue(
        code="render.missing_output",
        message="copy failed",
        output_path=tmp_path / "copied.png",
        view_label="front",
        frame_index=1,
        source_path=tmp_path / "relative.png",
        source_value="relative.png",
        source_was_relative=True,
        source_path_base=tmp_path,
        exception=OSError("disk full"),
    )
    assert issue_with_details["details"]["source_path_was_relative"] is True
    assert issue_with_details["details"]["source_path_base"] == str(tmp_path)
    assert issue_with_details["details"]["exception_type"] == "OSError"

    source_path = tmp_path / "renderer-source.png"
    _valid_image().save(source_path)

    def raise_same_file(source: Path, destination: Path) -> None:
        raise shutil.SameFileError(source, destination, "same")

    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering._same_file",
        lambda _source, _destination: False,
    )
    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering.shutil.copy2",
        raise_same_file,
    )
    copied_paths, copied_issues = _save_entry_images(
        {"images": [source_path]},
        output_dir=tmp_path / "copy",
        asset_key="asset",
        view_label="front",
    )
    assert copied_paths == ()
    assert copied_issues[0]["code"] == "render.missing_output"


def test_save_entry_images_reports_missing_artifact_after_processing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def skip_save(
        self: PILImage.Image, fp: str | Path, *args: object, **kwargs: object
    ) -> None:
        return None

    monkeypatch.setattr(PILImage.Image, "save", skip_save)

    saved_paths, save_issues = _save_entry_images(
        {"images": [_valid_image()]},
        output_dir=tmp_path / "images",
        asset_key="asset",
        view_label="front",
    )
    ignored_paths, ignored_issues = _save_entry_images(
        {"images": [object()]},
        output_dir=tmp_path / "ignored",
        asset_key="asset",
        view_label="front",
    )

    assert saved_paths == ()
    assert save_issues[0]["code"] == "render.missing_output"
    assert "expected artifact is missing" in save_issues[0]["message"]
    assert ignored_paths == ()
    assert ignored_issues == ()
