# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest
from PIL import Image

from texture_agent.functions.material_discovery import MaterialInfo, PrimTextureUnit
from texture_agent.tasks import apply_textures as at
from texture_agent.tasks.blend_textures import BlendedTextures


def _unit(
    key: str = "Steel",
    *,
    prim_path: str = "",
    material_prim_path: str = "/Root/Looks/Steel",
) -> PrimTextureUnit:
    return PrimTextureUnit(
        prim_path=prim_path,
        material_info=MaterialInfo(
            prim_path=material_prim_path,
            name="Steel",
            bound_prim_paths=[prim_path] if prim_path else ["/Root/Mesh"],
        ),
        key=key,
        prompt="brushed steel",
        opacity=0.8,
    )


def _png(path: Path, color: tuple[int, int, int] = (20, 30, 40)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)
    return str(path)


def _corrupt_png_idat(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    offset = 8
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = bytes(payload[offset + 4 : offset + 8])
        start = offset + 8
        crc_offset = start + length
        if kind == b"IDAT":
            data = bytearray(payload[start:crc_offset])
            corrupt_length = min(6, len(data))
            data[:corrupt_length] = b"\0" * corrupt_length
            payload[start:crc_offset] = data
            payload[crc_offset : crc_offset + 4] = struct.pack(
                ">I",
                zlib.crc32(kind + data) & 0xFFFFFFFF,
            )
            path.write_bytes(payload)
            return
        offset = crc_offset + 4
    raise AssertionError("seed PNG has no IDAT chunk")


def test_apply_textures_small_helper_edges(tmp_path: Path) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateNew(str(tmp_path / "helpers.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    UsdGeom.Xform.Define(stage, "/Root/Looks/Steel/not_shader")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Steel/no_file")
    shader.CreateIdAttr("UsdUVTexture")

    assert at._clone_material(stage, "/Root/Looks/Steel", "Steel_clone") == (
        "/Root/Looks/Steel_clone"
    )
    assert stage.GetPrimAtPath("/Root/Looks/Steel_clone").IsValid()

    at._set_tiledimage_file_input(stage, "/Root/Looks/Steel", "not_shader", "a.png")
    at._set_tiledimage_file_input(stage, "/Root/Looks/Steel", "no_file", "b.png")
    assert shader.GetInput("file").Get().path == "b.png"

    output_usd = tmp_path / "run" / "output" / "out.usda"
    output_usd.parent.mkdir(parents=True)
    assert at._author_texture_reference("", output_usd) == ""
    assert at._author_texture_reference("omniverse://server/a.png", output_usd) == (
        "omniverse://server/a.png"
    )
    assert (
        at._author_texture_reference(r"textures\a.png", output_usd) == "textures/a.png"
    )
    assert at._author_texture_reference(
        str(tmp_path / "outside.png"), output_usd
    ).endswith("outside.png")
    assert at._is_portable_texture_reference("../../outside.png", output_usd) is False

    bad_report = tmp_path / "bad.json"
    bad_report.write_text("{", encoding="utf-8")
    roots = at._allowed_texture_source_roots(
        str(tmp_path / "input.usda"),
        tmp_path / "work",
        {"uv_preparation": {"uv_report_path": str(bad_report)}},
    )
    assert roots
    assert at._is_under_any_root(tmp_path / "a.png", [tmp_path / "other"]) is False

    class UnresolvablePath:
        def resolve(self):
            raise OSError("bad path")

    assert at._is_under_any_root(UnresolvablePath(), [tmp_path]) is False

    tex_dir = tmp_path / "textures"
    tex_dir.mkdir()
    candidate = tmp_path / "source" / "same.png"
    _png(candidate, (1, 2, 3))
    target = tex_dir / "same.png"
    _png(target, (1, 2, 3))
    assert at._localized_texture_copy_path(candidate, tex_dir) == target
    _png(target, (9, 8, 7))
    assert at._localized_texture_copy_path(candidate, tex_dir).name.startswith("same_")

    _png(target, (1, 2, 3))
    assert at._localized_texture_copy_path(target, tex_dir) == target

    original_resolve = Path.resolve
    try:
        Path.resolve = lambda self, *args, **kwargs: (
            (_ for _ in ()).throw(OSError("resolve failed"))
            if self == candidate
            else original_resolve(self, *args, **kwargs)
        )
        assert at._localized_texture_copy_path(candidate, tex_dir) == target
    finally:
        Path.resolve = original_resolve

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            at.filecmp,
            "cmp",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cmp failed")),
        )
        assert at._localized_texture_copy_path(candidate, tex_dir).name.startswith(
            "same_"
        )
    finally:
        monkeypatch.undo()

    assert at._is_mdl_shader(stage.GetPrimAtPath("/Root/Looks/Steel")) is False
    assert at._preview_source_output_name("outputs:r") == "r"
    assert at._shared_preview_texture_uses_packed_orm({}) is False


def test_preview_graph_shader_reserves_after_hundred_name_collisions(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdShade

    stage = Usd.Stage.CreateNew(str(tmp_path / "shader-collisions.usda"))
    mat_path = "/Root/Looks/Steel"
    UsdShade.Material.Define(stage, mat_path)
    for suffix in range(100):
        name = (
            "TextureAgentSTReader" if suffix == 0 else f"TextureAgentSTReader_{suffix}"
        )
        stage.DefinePrim(f"{mat_path}/{name}", "Scope")

    shader = at._preview_graph_shader(
        stage,
        mat_path,
        "TextureAgentSTReader",
        "UsdPrimvarReader_float2",
    )

    assert shader.GetPrim().GetName() == "TextureAgentSTReader_100"
    assert shader.GetIdAttr().Get() == "UsdPrimvarReader_float2"


def test_preview_graph_shader_skips_inactive_same_id_collision(tmp_path: Path) -> None:
    from pxr import Usd, UsdShade

    stage = Usd.Stage.CreateNew(str(tmp_path / "inactive-shader-collision.usda"))
    mat_path = "/Root/Looks/Steel"
    UsdShade.Material.Define(stage, mat_path)
    occupied = UsdShade.Shader.Define(
        stage,
        f"{mat_path}/TextureAgentSTReader",
    )
    occupied.CreateIdAttr("UsdPrimvarReader_float2")
    occupied.GetPrim().SetActive(False)

    shader = at._preview_graph_shader(
        stage,
        mat_path,
        "TextureAgentSTReader",
        "UsdPrimvarReader_float2",
    )

    assert shader.GetPrim().GetName() == "TextureAgentSTReader_1"
    assert shader.GetIdAttr().Get() == "UsdPrimvarReader_float2"


def test_localize_stage_texture_reference_defensive_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    class InstanceProxyPrim:
        def IsInstanceProxy(self):
            return True

        def IsA(self, _schema):
            return False

        def GetAttributes(self):
            raise AssertionError("instance proxies are skipped before attrs")

    class InstanceProxyStage:
        def Traverse(self):
            return [InstanceProxyPrim()]

    assert (
        at._localize_stage_texture_references(
            InstanceProxyStage(),
            usd_path=str(tmp_path / "input.usda"),
            working_dir=tmp_path / "work",
            output_usd_path=tmp_path / "work" / "output" / "out.usda",
            context={},
        )
        == []
    )

    stage = Usd.Stage.CreateNew(str(tmp_path / "stage.usda"))
    shader = UsdShade.Shader.Define(stage, "/Root/Shader")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("remote_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://server/texture.png")
    )
    stage.GetRootLayer().Save()
    assert (
        at._localize_stage_texture_references(
            stage,
            usd_path=str(tmp_path / "stage.usda"),
            working_dir=tmp_path / "work",
            output_usd_path=tmp_path / "work" / "output" / "out.usda",
            context={},
        )
        == []
    )

    original_resolver = at._resolve_layer_anchored_path
    monkeypatch.setattr(at, "_resolve_layer_anchored_path", lambda *args: None)
    shader.GetInput("remote_texture").Set(Sdf.AssetPath("local.png"))
    assert (
        at._localize_stage_texture_references(
            stage,
            usd_path=str(tmp_path / "stage.usda"),
            working_dir=tmp_path / "work",
            output_usd_path=tmp_path / "work" / "output" / "out.usda",
            context={},
        )
        == []
    )

    class BadCandidate:
        def resolve(self):
            raise OSError("cannot resolve")

    monkeypatch.setattr(
        at, "_resolve_layer_anchored_path", lambda *args: BadCandidate()
    )
    assert (
        at._localize_stage_texture_references(
            stage,
            usd_path=str(tmp_path / "stage.usda"),
            working_dir=tmp_path / "work",
            output_usd_path=tmp_path / "work" / "output" / "out.usda",
            context={},
        )
        == []
    )
    monkeypatch.setattr(at, "_resolve_layer_anchored_path", original_resolver)

    outside = tmp_path.parent / f"{tmp_path.name}_outside.png"
    _png(outside)
    shader.GetInput("remote_texture").Set(Sdf.AssetPath(str(outside)))
    assert (
        at._localize_stage_texture_references(
            stage,
            usd_path=str(tmp_path / "stage.usda"),
            working_dir=tmp_path / "work",
            output_usd_path=tmp_path / "work" / "output" / "out.usda",
            context={},
        )
        == []
    )

    bundle_texture = tmp_path / "work" / "textures" / "inside.png"
    _png(bundle_texture)
    string_input = shader.CreateInput("inside_texture", Sdf.ValueTypeNames.String)
    string_input.Set(str(bundle_texture))
    localized = at._localize_stage_texture_references(
        stage,
        usd_path=str(tmp_path / "stage.usda"),
        working_dir=tmp_path / "work",
        output_usd_path=tmp_path / "work" / "output" / "out.usda",
        context={},
    )
    assert localized == ["/Root/Shader:inputs:inside_texture"]
    assert string_input.Get() == "../textures/inside.png"


def test_resolve_layer_anchor_and_localize_asset_error_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf

    class NoneAttr:
        def Get(self):
            return None

    assert at._resolve_layer_anchored_path(NoneAttr(), "a.png", tmp_path) is None

    class ResolvedBadAttr:
        def Get(self):
            return type("Value", (), {"resolvedPath": "\0"})()

    assert at._resolve_layer_anchored_path(ResolvedBadAttr(), "a.png", tmp_path) is None

    class StackRaisesAttr:
        def Get(self):
            return Sdf.AssetPath("a.png")

        def GetPropertyStack(self, *_args):
            raise RuntimeError("stack failed")

    assert (
        at._resolve_layer_anchored_path(StackRaisesAttr(), "a.png", tmp_path)
        == (tmp_path / "a.png").resolve()
    )

    class CandidateRaisesIsFile:
        def is_file(self):
            raise OSError("bad stat")

    assert (
        at._localize_asset(
            CandidateRaisesIsFile(),
            tmp_path,
            tmp_path / "textures",
            "Mat",
            "albedo",
        )
        is None
    )

    source = tmp_path / "upload" / "source.png"
    _png(source)

    class BadUploadRoot:
        def resolve(self):
            raise OSError("bad root")

    assert (
        at._localize_asset(
            source,
            BadUploadRoot(),
            tmp_path / "textures",
            "Mat",
            "albedo",
        )
        is None
    )

    tex_dir = tmp_path / "textures"
    tex_dir.mkdir()
    original_samefile = Path.samefile
    try:
        Path.samefile = lambda self, other: (
            (_ for _ in ()).throw(OSError("samefile failed"))
            if self == source.parent
            else original_samefile(self, other)
        )
        assert at._localize_asset(source, tmp_path / "upload", tex_dir, "Mat", "albedo")
    finally:
        Path.samefile = original_samefile


def test_localize_asset_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upload = tmp_path / "upload"
    tex_dir = upload / "textures"
    upload.mkdir()
    tex_dir.mkdir()

    assert (
        at._localize_asset(upload / "missing.png", upload, tex_dir, "Mat", "opacity")
        is None
    )

    outside = tmp_path / "outside.png"
    _png(outside)
    assert at._localize_asset(outside, upload, tex_dir, "Mat", "opacity") is None

    text = upload / "source.txt"
    text.write_text("not png", encoding="utf-8")
    assert at._localize_asset(text, upload, tex_dir, "Mat", "opacity") is None

    already = tex_dir / "already.png"
    _png(already)
    assert at._localize_asset(already, upload, tex_dir, "Mat", "opacity") == str(
        already
    )

    source = upload / "opacity.png"
    _png(source)
    monkeypatch.setattr(
        at.shutil,
        "copyfile",
        lambda *_args: (_ for _ in ()).throw(OSError("copy failed")),
    )
    assert at._localize_asset(source, upload, tex_dir, "Mat", "opacity") is None


def test_mdl_and_preview_guard_edges(tmp_path: Path) -> None:
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateNew(str(tmp_path / "guards.usda"))
    assert at._override_mdl_texture_inputs(
        stage,
        "/Missing",
        {},
        str(tmp_path / "input.usda"),
        tmp_path,
        tmp_path / "output" / "out.usda",
    ) == (0, [], [])
    assert at._override_usd_preview_texture_inputs(stage, "/Missing", {}) == []

    UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    mdl = UsdShade.Shader.Define(stage, "/Root/Looks/Steel/MDL")
    mdl.GetPrim().CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("Steel.mdl")
    )
    mdl.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("old.png")
    )
    mdl.CreateInput("display_color", Sdf.ValueTypeNames.String).Set("value")
    mdl.CreateInput("opacity_texture", Sdf.ValueTypeNames.String).Set("")

    overridden, cleared, localized = at._override_mdl_texture_inputs(
        stage,
        "/Root/Looks/Steel",
        {},
        str(tmp_path / "input.usda"),
        tmp_path,
        tmp_path / "output" / "out.usda",
    )
    assert (overridden, cleared, localized) == (0, [], [])

    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Steel/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("unknown", Sdf.ValueTypeNames.Float).Set(1.0)
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    assert at._override_usd_preview_texture_inputs(stage, "/Root/Looks/Steel", {}) == []

    texture = UsdShade.Shader.Define(stage, "/Root/Looks/Steel/Albedo")
    texture.CreateIdAttr("UsdUVTexture")
    diffuse = preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    diffuse.ConnectToSource(texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    records = at._override_usd_preview_texture_inputs(
        stage,
        "/Root/Looks/Steel",
        {"albedo": "../textures/albedo.png"},
    )
    assert records == ["/Root/Looks/Steel/Albedo:file"]
    assert texture.GetInput("file").Get().path == "../textures/albedo.png"

    assert (
        at._read_texture_input_string(
            mdl.CreateInput("unset_texture", Sdf.ValueTypeNames.Asset),
            Sdf.ValueTypeNames.Asset,
        )
        is None
    )
    assert (
        at._safe_set_typed_value(
            mdl.CreateInput("bad", Sdf.ValueTypeNames.Int),
            Sdf.ValueTypeNames.Int,
            "x",
        )
        is False
    )

    class SourceMissing:
        def GetPrim(self):
            raise AssertionError("falsy source skips before GetPrim")

        def __bool__(self):
            return False

    class FakeInput:
        def __init__(self, connected):
            self._connected = connected

        def GetConnectedSource(self):
            return self._connected

    assert (
        at._connected_usd_uv_texture_source(FakeInput((SourceMissing(), "rgb", None)))
        is None
    )

    class InvalidSource:
        class Prim:
            def IsValid(self):
                return False

        def GetPrim(self):
            return self.Prim()

        def __bool__(self):
            return True

    assert (
        at._connected_usd_uv_texture_source(FakeInput((InvalidSource(), "rgb", None)))
        is None
    )

    other_texture = UsdShade.Shader.Define(stage, "/Root/Looks/Steel/OtherTexture")
    other_texture.CreateIdAttr("OtherShader")
    other_input = preview.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
    other_input.ConnectToSource(
        other_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    assert at._connected_usd_uv_texture_source(other_input) is None

    normal_texture = UsdShade.Shader.Define(stage, "/Root/Looks/Steel/Normal")
    normal_texture.CreateIdAttr("UsdUVTexture")
    normal_input = preview.CreateInput("normal", Sdf.ValueTypeNames.Normal3f)
    normal_input.ConnectToSource(
        normal_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    assert at._override_usd_preview_texture_inputs(
        stage,
        "/Root/Looks/Steel",
        {"albedo": "../textures/albedo.png"},
    ) == ["/Root/Looks/Steel/Albedo:file"]


def test_empty_source_asset_omnipbr_shader_inputs_are_overridden(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateNew(str(tmp_path / "empty_source_asset.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Bucket")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Bucket/Shader")
    shader.GetPrim().CreateAttribute(
        "info:implementationSource",
        Sdf.ValueTypeNames.Token,
    ).Set("sourceAsset")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    )
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier",
        Sdf.ValueTypeNames.Token,
    ).Set("OmniPBR")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../textures/T_cleaning_bucket_albedo.png")
    )
    shader.CreateInput("detail_normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../textures/T_cleaning_bucket_normal.png")
    )
    shader.CreateInput("ORM_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../textures/T_cleaning_bucket_orm.png")
    )

    assert at._is_mdl_shader(shader.GetPrim()) is True
    overridden, cleared, localized = at._override_mdl_texture_inputs(
        stage,
        "/Root/Looks/Bucket",
        {
            "albedo": "../textures/Bucket_albedo.png",
            "normal": "../textures/Bucket_normal.png",
            "orm": "../textures/Bucket_orm.png",
        },
        str(tmp_path / "input.usda"),
        tmp_path,
        tmp_path / "output" / "out.usda",
    )

    assert overridden == 3
    assert cleared == []
    assert localized == []
    assert shader.GetInput("diffuse_texture").Get().path == (
        "../textures/Bucket_albedo.png"
    )
    assert shader.GetInput("detail_normalmap_texture").Get().path == (
        "../textures/Bucket_normal.png"
    )
    assert shader.GetInput("ORM_texture").Get().path == "../textures/Bucket_orm.png"


def test_apply_pbr_and_run_resume_per_prim_edges(tmp_path: Path) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    parent = UsdGeom.Xform.Define(stage, "/Root")
    parent.GetPrim().SetInstanceable(True)
    stage.DefinePrim("/Root/Looks").SetInstanceable(True)
    UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
    missing_mesh_path = "/Root/MissingMesh"
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "textures"
    blended = BlendedTextures(
        albedo=_png(textures_dir / "Steel_albedo.png", (1, 2, 3)),
        normal="",
        orm="",
    )
    assert at._apply_pbr_textures(
        stage,
        "/Missing",
        blended,
        tmp_path,
        "Steel",
        str(tmp_path / "input.usda"),
        tmp_path / "output" / "out.usda",
    ) == (0, [], [], [])
    assert (
        at._apply_pbr_textures(
            stage,
            "/Root/Looks/Steel",
            blended,
            tmp_path,
            "Steel",
            str(tmp_path / "input.usda"),
            tmp_path / "output" / "out.usda",
        )[0]
        == 0
    )

    task = at.ApplyTexturesTask()
    empty = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {},
            "prim_texture_units": [],
            "working_dir": str(tmp_path / "empty"),
        }
    )
    assert empty["output_usd_paths"] == []

    cached_dir = tmp_path / "cached" / "textures"
    cached = BlendedTextures(
        albedo=_png(cached_dir / "Steel_A_albedo.png", (1, 2, 3)),
        normal=_png(cached_dir / "Steel_A_normal.png", (128, 128, 255)),
        orm=_png(cached_dir / "Steel_A_orm.png", (255, 64, 32)),
    )
    assert Path(cached.albedo).is_file()
    per_prim_units = [
        _unit("Steel_A", prim_path=str(mesh.GetPath())),
        _unit("Steel_B", prim_path=missing_mesh_path),
    ]
    for suffix, color in (
        ("albedo", (10, 20, 30)),
        ("normal", (128, 128, 255)),
        ("orm", (255, 64, 32)),
    ):
        _png(cached_dir / f"Steel_B_{suffix}.png", color)

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {},
            "prim_texture_units": per_prim_units,
            "working_dir": str(tmp_path / "cached"),
            "resume": True,
        }
    )

    assert len(result["output_usd_paths"]) == 1
    assert set(result["blended_textures"]) == {"Steel_A", "Steel_B"}
    assert result["apply_textures_stats"]["applied_count"] == 2


def test_cached_apply_rejects_partial_blended_map_cache(tmp_path: Path) -> None:
    from pxr import Usd, UsdShade

    input_path = tmp_path / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_path))
    UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    stage.GetRootLayer().Save()

    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    units = [_unit("Steel_A"), _unit("Steel_B")]
    for key in ("Steel_A", "Steel_B"):
        _png(textures_dir / f"{key}_albedo.png")
        _png(textures_dir / f"{key}_normal.png")
    _png(textures_dir / "Steel_A_orm.png")

    with pytest.raises(RuntimeError, match="Steel_B:orm"):
        at.ApplyTexturesTask().run(
            {
                "usd_path": str(input_path),
                "prim_texture_units": units,
                "working_dir": str(working_dir),
                "cached_apply_only": True,
                "resume": True,
            }
        )


@pytest.mark.parametrize("corrupt_channel", ["albedo", "normal", "orm"])
def test_cached_apply_rejects_corrupt_png(
    tmp_path: Path,
    corrupt_channel: str,
) -> None:
    from pxr import Usd, UsdShade

    input_path = tmp_path / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_path))
    UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    stage.GetRootLayer().Save()

    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    for channel in ("albedo", "normal", "orm"):
        path = textures_dir / f"Steel_{channel}.png"
        _png(path)
        if channel == corrupt_channel:
            path.write_bytes(b"not a png")

    with pytest.raises(RuntimeError, match=rf"Steel:{corrupt_channel}"):
        at.ApplyTexturesTask().run(
            {
                "usd_path": str(input_path),
                "prim_texture_units": [_unit()],
                "working_dir": str(working_dir),
                "cached_apply_only": True,
                "resume": True,
            }
        )


def test_cached_apply_rejects_png_with_corrupt_idat(tmp_path: Path) -> None:
    from pxr import Usd, UsdShade

    input_path = tmp_path / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_path))
    UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    stage.GetRootLayer().Save()

    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    for channel in ("albedo", "normal", "orm"):
        path = textures_dir / f"Steel_{channel}.png"
        _png(path)
        if channel == "albedo":
            _corrupt_png_idat(path)
            with Image.open(path) as image:
                image.verify()

    with pytest.raises(RuntimeError, match=r"Steel:albedo"):
        at.ApplyTexturesTask().run(
            {
                "usd_path": str(input_path),
                "prim_texture_units": [_unit()],
                "working_dir": str(working_dir),
                "cached_apply_only": True,
                "resume": True,
            }
        )


def test_apply_per_material_writes_alias_material_paths(tmp_path: Path) -> None:
    from pxr import Usd, UsdShade

    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Copper")
    UsdShade.Material.Define(stage, "/Root/Assembly/Part/Looks/Diffuse_44")
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "work" / "textures"
    blended = BlendedTextures(
        albedo=_png(textures_dir / "Copper_albedo.png", (1, 2, 3)),
        normal="",
        orm="",
    )
    unit = PrimTextureUnit(
        prim_path="",
        material_info=MaterialInfo(
            prim_path="/Root/Looks/Copper",
            name="Copper",
            bound_prim_paths=["/Root/Assembly/Part/Mesh"],
            bound_subset_paths=["/Root/Assembly/Part/Mesh/face_0"],
            material_alias_paths=[
                "/Root/Looks/Copper",
                "/Root/Assembly/Part/Looks/Diffuse_44",
            ],
        ),
        key="Copper",
        prompt="brushed copper",
        opacity=0.8,
    )

    result = at.ApplyTexturesTask().run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Copper": blended},
            "prim_texture_units": [unit],
            "working_dir": str(tmp_path / "work"),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    assert output_stage is not None
    for material_path in (
        "/Root/Looks/Copper",
        "/Root/Assembly/Part/Looks/Diffuse_44",
    ):
        attr = output_stage.GetPrimAtPath(material_path).GetAttribute(
            "inputs:base_color_texture_file"
        )
        assert attr.Get().path == "../textures/Copper_albedo.png"
    assert result["apply_textures_stats"]["applied_count"] == 1


def test_apply_pbr_deinstances_regular_instance_material(tmp_path: Path) -> None:
    from pxr import Usd, UsdShade

    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    material_prim = UsdShade.Material.Define(stage, "/Root/Looks/Steel").GetPrim()
    material_prim.SetInstanceable(True)
    stage.GetRootLayer().Save()

    blended = BlendedTextures(
        albedo=_png(tmp_path / "textures" / "Steel_albedo.png", (1, 2, 3)),
        normal="",
        orm="",
    )

    at._apply_pbr_textures(
        stage,
        "/Root/Looks/Steel",
        blended,
        tmp_path,
        "Steel",
        str(tmp_path / "input.usda"),
        tmp_path / "output" / "out.usda",
    )

    assert not material_prim.IsInstanceable()
    attr = material_prim.GetAttribute("inputs:base_color_texture_file")
    assert attr.Get().path == "../textures/Steel_albedo.png"
