# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from world_understanding.agentic.usd_tasks import render_scene_preview as rsp


class _Listener:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)


class _Vec(tuple):
    pass


class _Range:
    def GetMin(self) -> _Vec:
        return _Vec((0.0, 0.0, 0.0))

    def GetMax(self) -> _Vec:
        return _Vec((2.0, 3.0, 4.0))


class _BBox:
    def ComputeAlignedRange(self) -> _Range:
        return _Range()


class _Stage:
    def __init__(self, prims: list[_Prim] | None = None) -> None:
        self.prims = prims or []

    def GetPseudoRoot(self) -> _Stage:
        return self

    def TraverseAll(self) -> list[_Prim]:
        return self.prims


class _Prim:
    def __init__(
        self,
        path: str,
        *,
        is_mesh: bool = False,
        is_imageable: bool = True,
        instance: bool = False,
        instance_proxy: bool = False,
        prototype: bool = False,
        active: bool = True,
    ) -> None:
        self.path = path
        self.is_mesh = is_mesh
        self.is_imageable = is_imageable
        self.instance = instance
        self.instance_proxy = instance_proxy
        self.prototype = prototype
        self.active = active

    def GetPath(self) -> str:
        return self.path

    def IsInstance(self) -> bool:
        return self.instance

    def IsInstanceProxy(self) -> bool:
        return self.instance_proxy

    def IsInPrototype(self) -> bool:
        return self.prototype

    def IsA(self, schema: object) -> bool:
        if schema == "mesh":
            return self.is_mesh
        if schema == "imageable":
            return self.is_imageable
        return False

    def IsActive(self) -> bool:
        return self.active

    def SetActive(self, value: bool) -> None:
        self.active = value


def _png_bytes(size: tuple[int, int] = (2, 2)) -> bytes:
    buf = BytesIO()
    Image.new("RGBA", size, color=(255, 0, 0, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _patch_preview_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage: _Stage,
    backend: object | None,
    listener: _Listener | None = None,
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {
        "corner": [],
        "side": [],
        "nullify": [],
        "lights": [],
        "prepare": [],
    }
    listener = listener or _Listener()

    monkeypatch.setattr(rsp, "get_listener", lambda context, logger_name=None: listener)
    monkeypatch.setattr(rsp, "load_stage", lambda path: stage)
    monkeypatch.setattr(rsp, "get_bbox_from_prim", lambda prim: _BBox())
    monkeypatch.setattr(rsp.Usd.TimeCode, "Default", lambda: "default-time")
    monkeypatch.setattr(
        rsp,
        "format_direction_for_filename",
        lambda direction: direction.replace("+", "p").replace("-", "m"),
    )
    monkeypatch.setattr(
        rsp,
        "add_corner_view_camera",
        lambda *args, **kwargs: calls["corner"].append(kwargs),
    )
    monkeypatch.setattr(
        rsp,
        "add_side_view_camera",
        lambda *args, **kwargs: calls["side"].append(kwargs),
    )
    monkeypatch.setattr(
        rsp,
        "prepare_stage_for_render",
        lambda stage, **kwargs: (
            calls["prepare"].append(kwargs) or (stage, {"asset_base_dir": "/prepared"})
        ),
    )
    monkeypatch.setattr(
        rsp, "nullify_materials", lambda stage: calls["nullify"].append(stage)
    )
    monkeypatch.setattr(
        rsp, "remove_all_lights", lambda stage: calls["lights"].append(stage)
    )
    if backend is not None:
        monkeypatch.setattr(
            rsp,
            "create_rendering_backend",
            lambda backend_type, render_config: backend,
        )
    return calls


def _install_fake_usdgeom(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_usd_geom = SimpleNamespace(Mesh="mesh", Imageable="imageable")
    monkeypatch.setattr(sys.modules["pxr"], "UsdGeom", fake_usd_geom, raising=False)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", fake_usd_geom)


def test_render_scene_preview_requires_existing_usd(tmp_path: Path) -> None:
    task = rsp.RenderScenePreviewTask()

    with pytest.raises(ValueError, match="usd_path not found"):
        task.run({})

    with pytest.raises(FileNotFoundError):
        task.run({"usd_path": tmp_path / "missing.usd"})


def test_render_scene_preview_remote_success_with_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("#usda", encoding="utf-8")
    prims = [
        _Prim("/World", is_imageable=False),
        _Prim("/World/Keep", is_mesh=True),
        _Prim("/World/Drop", is_imageable=True),
        _Prim("/PreviewCameras/Preview", is_imageable=True),
    ]
    stage = _Stage(prims)
    listener = _Listener()

    class _Backend:
        def render(self, stage: _Stage, **kwargs: Any) -> dict[str, Any]:
            camera = kwargs["cameras"][0]
            if "px" in camera and "py" in camera:
                return {
                    "results": [
                        {
                            "images": [
                                Image.new("RGB", (4, 5), color=(0, 255, 0)),
                                None,
                            ]
                        }
                    ]
                }
            return {"results": [{"images": [{"image": _png_bytes((3, 3))}]}]}

    calls = _patch_preview_dependencies(
        monkeypatch,
        stage=stage,
        backend=_Backend(),
        listener=listener,
    )
    _install_fake_usdgeom(monkeypatch)
    monkeypatch.setattr(rsp.Usd, "TraverseInstanceProxies", lambda: object())
    monkeypatch.setattr(rsp.Usd, "PrimRange", lambda root, predicate: iter(root.prims))

    context = rsp.RenderScenePreviewTask().run(
        {
            "usd_path": usd_path,
            "output_dir": tmp_path,
            "render_config": {
                "backend": "remote",
                "image_width": 8,
                "image_height": 9,
                "cameras": ["+x+y+z", "+x"],
                "background_color": [2.0, -1.0, 0.5],
            },
            "prim_filters": {"types": ["UsdGeom.Mesh"]},
        }
    )

    assert len(context["rendered_preview_paths"]) == 2
    assert context["composition_images"] == context["rendered_preview_paths"]
    assert calls["corner"][0]["max_scene_size"] == 4.0
    assert calls["side"][0]["direction"] == "+x"
    assert calls["prepare"] == [{"flatten": True, "normalize_materials": False}]
    assert calls["nullify"] == [stage]
    assert calls["lights"] == [stage]
    assert prims[2].active is False
    assert any("Deactivated 1 prims" in message for message in listener.infos)


def test_render_scene_preview_flattened_local_backend_and_render_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("#usda", encoding="utf-8")
    stage = _Stage()

    class _FailingBackend:
        def render(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("render failed")

    calls = _patch_preview_dependencies(
        monkeypatch, stage=stage, backend=_FailingBackend()
    )

    context = rsp.RenderScenePreviewTask().run(
        {
            "usd_path": usd_path,
            "output_dir": tmp_path,
            "render_config": {
                "backend": "warp",
                "flatten_before_render": True,
                "should_reset_materials": False,
                "use_lights": True,
            },
        }
    )

    assert context["rendered_preview_paths"] == []
    assert calls["prepare"] == [{"flatten": True, "normalize_materials": False}]
    assert calls["nullify"] == []
    assert calls["lights"] == []


def test_render_scene_preview_skips_none_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("#usda", encoding="utf-8")

    class _NoneImageBackend:
        def render(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"results": [{"images": [None]}]}

    _patch_preview_dependencies(
        monkeypatch,
        stage=_Stage(),
        backend=_NoneImageBackend(),
    )

    context = rsp.RenderScenePreviewTask().run(
        {
            "usd_path": usd_path,
            "output_dir": tmp_path,
            "render_config": {"backend": "warp"},
        }
    )

    assert context["rendered_preview_paths"] == []


def test_render_scene_preview_mock_backend_renders_without_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("#usda", encoding="utf-8")
    _patch_preview_dependencies(monkeypatch, stage=_Stage(), backend=None)

    context = rsp.RenderScenePreviewTask().run(
        {
            "usd_path": usd_path,
            "output_dir": tmp_path,
            "render_config": {
                "backend": "mock",
                "image_width": 16,
                "image_height": 8,
            },
        }
    )

    assert len(context["rendered_preview_paths"]) == 1
    rendered = Path(context["rendered_preview_paths"][0])
    assert rendered.exists()
    assert Image.open(rendered).size == (16, 8)


def test_render_scene_preview_propagates_unknown_backend_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("#usda", encoding="utf-8")
    _patch_preview_dependencies(monkeypatch, stage=_Stage(), backend=None)

    with pytest.raises(ValueError, match="Unknown rendering backend: typo"):
        rsp.RenderScenePreviewTask().run(
            {
                "usd_path": usd_path,
                "output_dir": tmp_path,
                "render_config": {"backend": "typo"},
            }
        )


def test_apply_prim_filters_unknown_empty_and_deactivate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = rsp.RenderScenePreviewTask()
    listener = _Listener()
    _install_fake_usdgeom(monkeypatch)
    monkeypatch.setattr(rsp.Usd, "TraverseInstanceProxies", lambda: object())

    stage = _Stage([_Prim("/World/Only", is_mesh=False)])
    monkeypatch.setattr(rsp.Usd, "PrimRange", lambda root, predicate: iter(root.prims))
    assert (
        task._apply_prim_filters(stage, {"types": ["UsdGeom.DoesNotExist"]}, listener)
        == 0
    )
    assert any("Unknown prim type" in message for message in listener.warnings)
    assert task._apply_prim_filters(stage, {"types": ["Bad.Type"]}, listener) == 0
    assert task._apply_prim_filters(stage, {"types": ["UsdGeom.Mesh"]}, listener) == 0
    assert any("No prims matched" in message for message in listener.warnings)

    prims = [
        _Prim("/World", is_imageable=False),
        _Prim("/World/Keep", is_mesh=True),
        _Prim("/World/SkipInstance", is_mesh=True, instance=True),
        _Prim("/World/SkipProxy", is_mesh=True, instance_proxy=True),
        _Prim("/World/SkipPrototype", is_mesh=True, prototype=True),
        _Prim("/World/Excluded", is_mesh=True),
        _Prim("/World/Drop", is_imageable=True),
        _Prim("/PreviewCameras/Camera", is_imageable=True),
        _Prim("/World/Inactive", is_imageable=True, active=False),
        _Prim("/Other/Structure", is_imageable=False),
    ]
    stage = _Stage(prims)
    deactivated = task._apply_prim_filters(
        stage,
        {
            "types": ["UsdGeom.Mesh"],
            "skip_prototypes": True,
            "exclude_paths": ["/World/Excluded"],
        },
        listener,
    )

    assert deactivated == 5
    assert prims[6].active is False
    assert prims[2].active is False
    assert prims[3].active is False
    assert prims[4].active is False
    assert prims[5].active is False
    assert prims[7].active is True
    assert prims[8].active is False


def test_to_pil_decodes_supported_formats() -> None:
    task = rsp.RenderScenePreviewTask()
    pil = Image.new("RGB", (1, 1))
    encoded = base64.b64encode(_png_bytes((4, 4))).decode()

    assert task._to_pil(pil) is pil
    assert task._to_pil({"image": _png_bytes((2, 3))}).size == (2, 3)
    assert task._to_pil({"image": encoded}).size == (4, 4)

    with pytest.raises(ValueError, match="Failed to decode"):
        task._to_pil({"image": "not base64"})

    with pytest.raises(ValueError, match="Unexpected image format"):
        task._to_pil(object())
