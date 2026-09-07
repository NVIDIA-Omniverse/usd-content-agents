# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for packaged-stage render input composition."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


def _packaged_textured_stage(tmp_path: Path) -> tuple[Path, bytes]:
    from PIL import Image
    from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils

    source_dir = tmp_path / "source"
    texture_dir = source_dir / "0"
    texture_dir.mkdir(parents=True)
    texture_path = texture_dir / "albedo.png"
    Image.new("RGB", (2, 2), (240, 96, 24)).save(texture_path)
    texture_bytes = texture_path.read_bytes()

    root_path = source_dir / "root.usda"
    stage = Usd.Stage.CreateNew(str(root_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    material = UsdShade.Material.Define(stage, "/World/Looks/Textured")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Textured/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("0/albedo.png")
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    package_path = tmp_path / "textured.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(root_path), str(package_path))
    with zipfile.ZipFile(package_path) as archive:
        assert archive.read("0/albedo.png") == texture_bytes
    return package_path, texture_bytes


def _assert_packaged_texture(stage, package_path: Path, texture_bytes: bytes) -> None:
    from pxr import Ar, Sdf

    value = stage.GetAttributeAtPath("/World/Looks/Textured/Shader.inputs:file").Get()
    assert isinstance(value, Sdf.AssetPath)
    assert value.path == f"{package_path}[0/albedo.png]"
    assert value.resolvedPath == value.path
    asset = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(value.resolvedPath))
    assert asset is not None
    assert bytes(asset.GetBuffer()) == texture_bytes


def test_prepare_render_input_reuses_only_an_untouched_usdz(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom
    from usd_core.render.base import prepare_render_input

    package_path, texture_bytes = _packaged_textured_stage(tmp_path)
    package_sha256 = hashlib.sha256(package_path.read_bytes()).hexdigest()

    untouched = Usd.Stage.Open(str(package_path))
    prepared, is_temp = prepare_render_input(untouched, tmp_path / "untouched")
    assert prepared == package_path
    assert not is_temp

    stage = Usd.Stage.Open(str(package_path))
    root = stage.GetRootLayer()
    assert not root.dirty
    assert stage.GetSessionLayer().empty
    UsdGeom.Camera.Define(stage, "/World/usd_cam")
    assert root.dirty
    assert stage.GetSessionLayer().empty

    prepared, is_temp = prepare_render_input(stage, tmp_path / "dirty")
    assert prepared != package_path
    assert not is_temp
    reopened = Usd.Stage.Open(str(prepared))
    assert reopened.GetPrimAtPath("/World/usd_cam").IsA(UsdGeom.Camera)
    _assert_packaged_texture(reopened, package_path, texture_bytes)
    assert hashlib.sha256(package_path.read_bytes()).hexdigest() == package_sha256


def test_ovrtx_render_composes_dirty_usdz_camera_and_texture(
    tmp_path: Path, monkeypatch
) -> None:
    from PIL import Image
    from pxr import Usd, UsdGeom
    from usd_core.render.ovrtx import OvRTXRenderBackend, _product_path

    package_path, texture_bytes = _packaged_textured_stage(tmp_path)
    package_sha256 = hashlib.sha256(package_path.read_bytes()).hexdigest()
    stage = Usd.Stage.Open(str(package_path))
    UsdGeom.Camera.Define(stage, "/World/usd_cam")
    assert stage.GetRootLayer().dirty

    observed: dict[str, object] = {}

    class InspectingDaemon:
        def render(self, params: dict) -> list[dict]:
            combined = Usd.Stage.Open(params["usd_path"])
            assert combined is not None
            product_path = _product_path("/World/usd_cam")
            product = combined.GetPrimAtPath(product_path)
            camera = combined.GetPrimAtPath("/World/usd_cam")
            assert product.IsValid() and product.GetTypeName() == "RenderProduct"
            assert camera.IsA(UsdGeom.Camera)
            assert product.GetRelationship("camera").GetTargets() == [camera.GetPath()]
            _assert_packaged_texture(combined, package_path, texture_bytes)
            observed["product"] = product_path
            observed["camera"] = camera.GetPath().pathString
            output_path = Path(params["out_paths"][0])
            Image.new("RGB", (4, 4), (240, 96, 24)).save(output_path)
            return [{"camera": "/World/usd_cam", "path": str(output_path)}]

    backend = OvRTXRenderBackend(venv_dir=tmp_path / "unused-venv")
    monkeypatch.setattr(backend, "_ensure_daemon", lambda: InspectingDaemon())
    results = backend.render(
        stage,
        ["/World/usd_cam"],
        4,
        4,
        tmp_path / "render",
        mode="fast",
    )

    assert observed == {
        "product": _product_path("/World/usd_cam"),
        "camera": "/World/usd_cam",
    }
    assert len(results) == 1
    assert Path(results[0].path).read_bytes()
    assert results[0].camera == "/World/usd_cam"
    assert hashlib.sha256(package_path.read_bytes()).hexdigest() == package_sha256
