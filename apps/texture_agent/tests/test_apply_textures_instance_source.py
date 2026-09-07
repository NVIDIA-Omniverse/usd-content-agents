# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolving an instance proxy to the authored prim it references (issue #916)."""

from __future__ import annotations

from pathlib import Path

import pytest

from texture_agent.tasks import apply_textures as at
from texture_agent.tasks.apply_textures import (
    _editable_prim_for_path,
    _instance_source_prim,
)


def _stage_with_internal_instance(tmp_path: Path, instance_count: int = 1):
    """A stage whose instances reference an authored source in the same layer."""
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateNew(str(tmp_path / f"scene{instance_count}.usda"))
    UsdGeom.Xform.Define(stage, "/Body/Prototypes/Body")
    UsdShade.Material.Define(stage, "/Body/Prototypes/Body/Looks/Diffuse")

    for index in range(instance_count):
        name = f"Body{index}" if index else "Body"
        inst = UsdGeom.Xform.Define(stage, f"/Body/{name}").GetPrim()
        inst.GetReferences().AddInternalReference("/Body/Prototypes/Body")
        inst.SetInstanceable(True)

    stage.SetDefaultPrim(stage.GetPrimAtPath("/Body"))
    return stage


def test_proxy_resolves_to_the_authored_source(tmp_path: Path) -> None:
    """The proxy maps to the referenced prim, which is authorable."""
    pytest.importorskip("pxr")
    from pxr import Sdf

    stage = _stage_with_internal_instance(tmp_path)
    proxy = stage.GetPrimAtPath("/Body/Body/Looks/Diffuse")
    assert proxy.IsInstanceProxy()

    source = _instance_source_prim(stage, proxy)

    assert source is not None
    assert str(source.GetPath()) == "/Body/Prototypes/Body/Looks/Diffuse"
    assert not source.IsInstanceProxy()
    # The resolved prim really does accept authored properties.
    source.CreateAttribute("inputs:probe", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("x.png")
    )


def test_resolution_leaves_instancing_intact(tmp_path: Path) -> None:
    """The whole point of #916: the exported stage is not de-instanced."""
    pytest.importorskip("pxr")

    stage = _stage_with_internal_instance(tmp_path)
    assert stage.GetPrimAtPath("/Body/Body").IsInstanceable()

    _editable_prim_for_path(stage, "/Body/Body/Looks/Diffuse")

    assert stage.GetPrimAtPath("/Body/Body").IsInstanceable()
    assert stage.GetPrototypes()


def test_shared_source_is_refused(tmp_path: Path) -> None:
    """Two instances over one source must not be edited through that source.

    Authoring there would apply one instance's textures to every instance.
    """
    pytest.importorskip("pxr")

    stage = _stage_with_internal_instance(tmp_path, instance_count=2)
    proxy = stage.GetPrimAtPath("/Body/Body/Looks/Diffuse")

    assert _instance_source_prim(stage, proxy) is None


def test_editable_prim_falls_back_when_source_is_shared(tmp_path: Path) -> None:
    """With no safe source the caller keeps the pre-existing behaviour."""
    pytest.importorskip("pxr")

    stage = _stage_with_internal_instance(tmp_path, instance_count=2)

    prim = _editable_prim_for_path(stage, "/Body/Body/Looks/Diffuse")

    assert prim.IsValid()
    assert not stage.GetPrimAtPath("/Body/Body").IsInstanceable()


def test_proxy_without_an_enclosing_instance_resolves_to_none() -> None:
    """A proxy whose ancestors run out yields no source rather than looping."""

    class _Path:
        @staticmethod
        def IsAbsoluteRootPath() -> bool:
            return True

    class _Prim:
        @staticmethod
        def IsValid() -> bool:
            return False

        @staticmethod
        def IsInstance() -> bool:
            return False

    class _Proxy:
        @staticmethod
        def IsInstanceProxy() -> bool:
            return True

        @staticmethod
        def IsInstance() -> bool:
            return False

        @staticmethod
        def IsValid() -> bool:
            return True

        @staticmethod
        def GetParent() -> _Prim:
            return _Prim()

        @staticmethod
        def GetPath() -> _Path:
            return _Path()

    assert _instance_source_prim(object(), _Proxy()) is None


def test_external_reference_does_not_resolve_to_a_colliding_path(
    tmp_path: Path,
) -> None:
    """A same-named prim on this stage is not the external reference's source.

    ``arc.GetTargetNode().path`` for an external reference names a prim in the
    referenced layer. Looking that path up on the containing stage can land on an
    unrelated prim, which must not be textured in the real source's place.
    """
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdShade

    model_path = tmp_path / "model.usda"
    model_stage = Usd.Stage.CreateNew(str(model_path))
    UsdGeom.Xform.Define(model_stage, "/Model")
    UsdShade.Material.Define(model_stage, "/Model/Looks/Steel")
    model_stage.SetDefaultPrim(model_stage.GetPrimAtPath("/Model"))
    model_stage.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(str(tmp_path / "collide.usda"))
    # A decoy at exactly the path the external arc reports.
    UsdShade.Material.Define(stage, "/Model/Looks/Steel")
    inst = UsdGeom.Xform.Define(stage, "/World/Instance").GetPrim()
    inst.GetReferences().AddReference(str(model_path), "/Model")
    inst.SetInstanceable(True)

    proxy = stage.GetPrimAtPath("/World/Instance/Looks/Steel")
    assert proxy.IsInstanceProxy()

    assert _instance_source_prim(stage, proxy) is None


def test_apply_pbr_follows_the_resolved_path_for_shader_edits(
    tmp_path: Path,
) -> None:
    """Shader-level overrides must target the resolved prim, not the proxy.

    ``_editable_prim_for_path`` can return a prim whose path differs from the one
    requested. The helpers that rewrite shader inputs look their targets up by
    path, so the caller has to follow the resolved path or every such write lands
    on an unauthorable proxy.
    """
    pytest.importorskip("pxr")
    from PIL import Image

    from texture_agent.tasks.apply_textures import _apply_pbr_textures
    from texture_agent.tasks.blend_textures import BlendedTextures

    stage = _stage_with_internal_instance(tmp_path)
    mat_path = "/Body/Body/Looks/Diffuse"
    assert stage.GetPrimAtPath(mat_path).IsInstanceProxy()

    textures = tmp_path / "textures"
    textures.mkdir(parents=True, exist_ok=True)
    albedo = textures / "Diffuse_albedo.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(albedo)

    seen: list[str] = []
    real = at._override_usd_preview_texture_inputs

    def _record(stage_arg, path_arg, *args, **kwargs):
        seen.append(path_arg)
        return real(stage_arg, path_arg, *args, **kwargs)

    at._override_usd_preview_texture_inputs = _record
    try:
        _apply_pbr_textures(
            stage,
            mat_path,
            BlendedTextures(albedo=str(albedo), normal="", orm=""),
            tmp_path,
            "Diffuse",
            str(tmp_path / "scene1.usda"),
            tmp_path / "output" / "out.usda",
        )
    finally:
        at._override_usd_preview_texture_inputs = real

    assert seen, "the shader-override helper was never called"
    for path_arg in seen:
        assert path_arg != mat_path, "shader edits were routed at the proxy path"
        assert not stage.GetPrimAtPath(path_arg).IsInstanceProxy()


def test_resolution_skips_an_arc_that_lacks_the_relative_path(
    tmp_path: Path,
) -> None:
    """Several references compose one instance; only some carry the child.

    The first arc contributes no prim at the proxy's relative path, so it is
    skipped rather than treated as the source.
    """
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateNew(str(tmp_path / "multi.usda"))
    UsdGeom.Xform.Define(stage, "/Src/Empty")
    UsdGeom.Xform.Define(stage, "/Src/Full")
    UsdShade.Material.Define(stage, "/Src/Full/Looks/Diffuse")
    inst = UsdGeom.Xform.Define(stage, "/World/Inst").GetPrim()
    inst.GetReferences().AddInternalReference("/Src/Empty")
    inst.GetReferences().AddInternalReference("/Src/Full")
    inst.SetInstanceable(True)

    proxy = stage.GetPrimAtPath("/World/Inst/Looks/Diffuse")
    assert proxy.IsInstanceProxy()

    source = _instance_source_prim(stage, proxy)

    assert source is not None
    assert str(source.GetPath()) == "/Src/Full/Looks/Diffuse"
