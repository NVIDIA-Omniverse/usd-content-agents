# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
from texture_agent.functions.material_discovery import MaterialInfo, PrimTextureUnit

from ...service.workers import executor

pytest.importorskip("pxr")
from pxr import Ar, Sdf, Usd, UsdShade  # noqa: E402


def _unit(key: str, material_path: str) -> PrimTextureUnit:
    return PrimTextureUnit(
        prim_path="",
        material_info=MaterialInfo(
            prim_path=material_path,
            name=Path(material_path).name,
        ),
        key=key,
        prompt="paint",
        opacity=1.0,
    )


def _portable_result() -> dict[str, Any]:
    return {
        "portable": True,
        "diagnostics": [],
        "texture_reference_count": 0,
        "non_relative_texture_paths": [],
        "missing_texture_paths": [],
    }


def test_package_reference_channel_fallback_edges(tmp_path: Path) -> None:
    output_dir = tmp_path / "cache" / "output"
    textures_dir = tmp_path / "cache" / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    albedo = textures_dir / "paint_albedo.png"
    orm = textures_dir / "paint_orm.png"
    loose = textures_dir / "loose_albedo.png"
    for path in (albedo, orm, loose):
        path.write_bytes(b"texture")

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    root = stage.DefinePrim("/Root", "Xform")
    stage.SetDefaultPrim(root)
    material = UsdShade.Material.Define(stage, "/Root/Looks/Paint")

    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Paint/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    albedo_texture = UsdShade.Shader.Define(
        stage,
        "/Root/Looks/Paint/AlbedoTexture",
    )
    albedo_texture.CreateIdAttr("UsdUVTexture")
    albedo_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("missing-generic.png")
    )
    albedo_output = albedo_texture.CreateOutput(
        "rgb",
        Sdf.ValueTypeNames.Float3,
    )
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        albedo_output
    )

    packed_texture = UsdShade.Shader.Define(
        stage,
        "/Root/Looks/Paint/PackedTexture",
    )
    packed_texture.CreateIdAttr("UsdUVTexture")
    packed_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("missing-packed.png")
    )
    for preview_name, packed_output in (
        ("occlusion", "r"),
        ("roughness", "g"),
        ("metallic", "b"),
    ):
        output = packed_texture.CreateOutput(
            f"outputs:{packed_output}",
            Sdf.ValueTypeNames.Float,
        )
        preview.CreateInput(preview_name, Sdf.ValueTypeNames.Float).ConnectToSource(
            output
        )

    material.GetPrim().CreateAttribute(
        "inputs:file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("missing-non-shader.png"))

    stage.DefinePrim("/Root/Loose", "Scope")
    orphan_texture = UsdShade.Shader.Define(stage, "/Root/Loose/Texture")
    orphan_texture.CreateIdAttr("UsdUVTexture")
    orphan_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("missing-orphan.png")
    )
    stage.GetRootLayer().Save()

    context = {
        "output_usd_paths": [str(output_usd)],
        "prim_texture_units": [
            _unit("paint", "/Root/Looks/Paint"),
            _unit("loose", "/Root/Loose"),
        ],
        "blended_textures": {
            "paint": {
                "albedo": str(albedo),
                "normal": "",
                "orm": str(orm),
            },
            "loose": {
                "albedo": str(loose),
                "normal": "",
                "orm": "",
            },
        },
    }

    assert executor._package_usdz(context, tmp_path) is None
    rewritten = Usd.Stage.Open(str(output_usd))
    assert (
        Path(
            UsdShade.Shader(rewritten.GetPrimAtPath("/Root/Looks/Paint/AlbedoTexture"))
            .GetInput("file")
            .Get()
            .path
        ).name
        == albedo.name
    )
    assert (
        Path(
            UsdShade.Shader(rewritten.GetPrimAtPath("/Root/Looks/Paint/PackedTexture"))
            .GetInput("file")
            .Get()
            .path
        ).name
        == orm.name
    )


def test_package_rewrite_handles_weak_authored_value_and_package_ref_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "cache" / "output"
    textures_dir = tmp_path / "cache" / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    (output_dir / "weak.png").write_bytes(b"weak")
    generated = textures_dir / "paint_albedo.png"
    generated.write_bytes(b"generated")

    weak_usd = output_dir / "weak.usda"
    weak_stage = Usd.Stage.CreateNew(str(weak_usd))
    UsdShade.Material.Define(weak_stage, "/Root/Looks/Paint")
    weak_shader = UsdShade.Shader.Define(
        weak_stage,
        "/Root/Looks/Paint/Texture",
    )
    weak_shader.CreateIdAttr("UsdUVTexture")
    weak_shader.CreateInput("albedo_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("weak.png")
    )
    weak_stage.GetRootLayer().Save()

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    stage.GetRootLayer().subLayerPaths = ["weak.usda"]
    root = stage.DefinePrim("/Root", "Xform")
    stage.SetDefaultPrim(root)
    UsdShade.Material.Define(stage, "/Root/Looks/Paint")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Paint/Texture")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("albedo_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("legacy.png")
    )
    stage.GetRootLayer().Save()

    expected_ref = "../textures/paint_albedo.png"
    original_is_package_relative = Ar.IsPackageRelativePath
    intercepted = False

    def _one_shot_package_relative(path_value: str) -> bool:
        nonlocal intercepted
        if str(path_value) == expected_ref and not intercepted:
            intercepted = True
            return True
        return bool(original_is_package_relative(path_value))

    monkeypatch.setattr(Ar, "IsPackageRelativePath", _one_shot_package_relative)
    context = {
        "output_usd_paths": [str(output_usd)],
        "prim_texture_units": [_unit("paint", "/Root/Looks/Paint")],
        "blended_textures": {
            "paint": {
                "albedo": str(generated),
                "normal": "",
                "orm": "",
            }
        },
    }

    assert executor._package_usdz(context, tmp_path) is not None
    assert intercepted is True
    root_layer = Sdf.Layer.FindOrOpen(str(output_usd))
    root_spec = root_layer.GetPropertyAtPath(
        "/Root/Looks/Paint/Texture.inputs:albedo_texture"
    )
    assert root_spec.default.path == expected_ref
    weak_layer = Sdf.Layer.FindOrOpen(str(weak_usd))
    weak_spec = weak_layer.GetPropertyAtPath(
        "/Root/Looks/Paint/Texture.inputs:albedo_texture"
    )
    assert weak_spec.default.path == "weak.png"


def test_package_clear_skips_external_and_mismatched_authored_values(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    output_dir = session_dir / "cache" / "output"
    input_dir = session_dir / "input"
    output_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)

    inside_usd = output_dir / "inside.usda"
    inside_stage = Usd.Stage.CreateNew(str(inside_usd))
    inside_shader = UsdShade.Shader.Define(
        inside_stage,
        "/Root/Looks/Paint/Shader",
    )
    inside_shader.CreateIdAttr("mdlMaterial")
    inside_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("Other.mdl"))
    inside_stage.GetRootLayer().Save()

    outside_usd = input_dir / "outside.usda"
    outside_stage = Usd.Stage.CreateNew(str(outside_usd))
    outside_shader = UsdShade.Shader.Define(
        outside_stage,
        "/Root/Looks/Paint/Shader",
    )
    outside_shader.CreateIdAttr("mdlMaterial")
    outside_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("External.mdl"))
    outside_stage.GetRootLayer().Save()

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    stage.GetRootLayer().subLayerPaths = ["inside.usda", str(outside_usd)]
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Paint/Shader")
    shader.CreateIdAttr("mdlMaterial")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("OmniPBR.mdl"))
    stage.GetRootLayer().Save()
    context = {"output_usd_paths": [str(output_usd)]}

    assert executor._package_usdz(context, session_dir) is None
    assert context["usdz_mdl_source_assets_cleared"] == [
        "/Root/Looks/Paint/Shader.info:mdl:sourceAsset"
    ]
    root_layer = Sdf.Layer.FindOrOpen(str(output_usd))
    root_spec = root_layer.GetPropertyAtPath(
        "/Root/Looks/Paint/Shader.info:mdl:sourceAsset"
    )
    assert not root_spec.HasInfo("default")


def test_layered_package_handles_disappearing_existing_texture_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = tmp_path / "session"
    output_dir = session_dir / "cache" / "output"
    extract_root = session_dir / "cache" / "reconstructed"
    output_dir.mkdir(parents=True)
    extract_root.mkdir(parents=True)
    asset_texture = extract_root / "asset.png"
    string_texture = extract_root / "string.png"
    asset_texture.write_bytes(b"asset")
    string_texture.write_bytes(b"string")

    package_usd = extract_root / "root.usda"
    package_stage = Usd.Stage.CreateNew(str(package_usd))
    root = package_stage.DefinePrim("/Root", "Xform")
    package_stage.SetDefaultPrim(root)
    shader = UsdShade.Shader.Define(package_stage, "/Root/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("asset_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("asset.png")
    )
    shader.CreateInput("string_texture", Sdf.ValueTypeNames.String).Set("string.png")
    package_stage.GetRootLayer().Save()

    output_usd = output_dir / "textured_output.usda"
    output_stage = Usd.Stage.CreateNew(str(output_usd))
    output_stage.DefinePrim("/Root", "Xform")
    output_stage.GetRootLayer().Save()

    monkeypatch.setattr(
        executor,
        "_prepare_source_usdz_stage",
        lambda _context, _session_dir: package_usd,
    )
    real_is_file = Path.is_file
    target_paths = {asset_texture.resolve(), string_texture.resolve()}

    def _is_file_with_rewrite_race(path: Path) -> bool:
        resolved = path.resolve()
        if resolved in target_paths:
            for frame_info in inspect.stack():
                if frame_info.function != "_rewritten_source_package_member_ref":
                    continue
                caller = frame_info.frame.f_back
                assert caller is not None
                caller_source = inspect.getframeinfo(caller).code_context or []
                return any("staged_member_ref" in line for line in caller_source)
        return real_is_file(path)

    monkeypatch.setattr(Path, "is_file", _is_file_with_rewrite_race)
    context = {
        "output_usd_paths": [str(output_usd)],
        "source_usdz_extract_root": str(extract_root),
    }

    assert executor._package_usdz(context, session_dir) is not None
    assert (
        Usd.Stage.Open(str(package_usd))
        .GetPrimAtPath("/Root/Shader")
        .GetAttribute("inputs:asset_texture")
        .Get()
        .path
        == "asset.png"
    )
    assert Usd.Stage.Open(str(package_usd)).GetPrimAtPath("/Root/Shader").GetAttribute(
        "inputs:string_texture"
    ).Get() == str(string_texture)
    assert context["render_string_texture_localizations"] == 1


def test_single_layer_package_skips_unusable_collected_string_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from texture_agent.functions import artifact_manifest

    output_dir = tmp_path / "cache" / "output"
    output_dir.mkdir(parents=True)
    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    root = stage.DefinePrim("/Root", "Xform")
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()

    monkeypatch.setattr(
        artifact_manifest,
        "validate_output_texture_portability",
        lambda *_args, **_kwargs: _portable_result(),
    )
    monkeypatch.setattr(
        artifact_manifest,
        "_output_texture_references",
        lambda _path: [
            {"value_type": "string", "resolved_path": ""},
            {
                "value_type": "string",
                "resolved_path": "omniverse://server/texture.png",
            },
            {
                "value_type": "string",
                "resolved_path": str(tmp_path / "missing.png"),
            },
        ],
    )

    context = {"output_usd_paths": [str(output_usd)]}
    assert executor._package_usdz(context, tmp_path) is not None


def test_layered_render_skips_unresolvable_string_texture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from texture_agent.functions import artifact_manifest

    session_dir = tmp_path / "session"
    output_dir = session_dir / "cache" / "output"
    extract_root = session_dir / "cache" / "reconstructed"
    output_dir.mkdir(parents=True)
    extract_root.mkdir(parents=True)

    package_usd = extract_root / "root.usda"
    package_stage = Usd.Stage.CreateNew(str(package_usd))
    root = package_stage.DefinePrim("/Root", "Xform")
    package_stage.SetDefaultPrim(root)
    shader = UsdShade.Shader.Define(package_stage, "/Root/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("missing_texture", Sdf.ValueTypeNames.String).Set("missing.png")
    package_stage.GetRootLayer().Save()

    output_usd = output_dir / "textured_output.usda"
    output_stage = Usd.Stage.CreateNew(str(output_usd))
    output_stage.DefinePrim("/Root", "Xform")
    output_stage.GetRootLayer().Save()
    monkeypatch.setattr(
        executor,
        "_prepare_source_usdz_stage",
        lambda _context, _session_dir: package_usd,
    )
    monkeypatch.setattr(
        artifact_manifest,
        "validate_output_texture_portability",
        lambda *_args, **_kwargs: _portable_result(),
    )
    context = {
        "output_usd_paths": [str(output_usd)],
        "source_usdz_extract_root": str(extract_root),
    }

    assert executor._package_usdz(context, session_dir) is not None
    assert "render_string_texture_localizations" not in context
