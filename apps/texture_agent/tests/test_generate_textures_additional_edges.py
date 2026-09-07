# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from PIL import Image

from texture_agent.functions.external_authoring import (
    ExternalAuthoringCapabilityReceipt,
)
from texture_agent.functions.material_discovery import MaterialInfo, PrimTextureUnit
from texture_agent.functions.texture_generation import (
    Conditioning,
    GeneratedTextures,
    GenerationResult,
    MapArtifact,
)
from texture_agent.tasks import generate_textures as gt
from texture_agent.tasks.generate_textures import GenerateTexturesTask


def _unit(
    key: str = "Body",
    *,
    material_name: str | None = None,
    material_path: str | None = None,
    bound_prim_paths: list[str] | None = None,
    base_color_texture: str | None = None,
) -> PrimTextureUnit:
    name = material_name or key
    return PrimTextureUnit(
        prim_path="",
        material_info=MaterialInfo(
            prim_path=material_path or f"/World/Looks/{name}",
            name=name,
            bound_prim_paths=bound_prim_paths or ["/World/BodyMesh"],
            base_color_texture=base_color_texture,
            specular_roughness=0.5,
            base_metalness=0.25,
        ),
        key=key,
        prompt=f"weathered {name}",
        opacity=0.8,
    )


def _write_rgb(
    path: Path, color: tuple[int, int, int], size: tuple[int, int] = (4, 4)
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return str(path)


def _write_rebake_fixture_usd(
    path: Path,
    *,
    albedo_ref: str,
    normal_ref: str | None = None,
    orm_ref: str | None = None,
    material_path: str = "/World/Looks/Body",
    include_child_shader: bool = False,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    mesh = UsdGeom.Mesh.Define(stage, "/World/BodyMesh")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-1.0, -1.0, 0.0),
            Gf.Vec3f(1.0, -1.0, 0.0),
            Gf.Vec3f(1.0, 1.0, 0.0),
            Gf.Vec3f(-1.0, 1.0, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])

    material = UsdShade.Material.Define(stage, material_path)
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(albedo_ref))
    if normal_ref is not None:
        material.GetPrim().CreateAttribute(
            "inputs:normalmap_texture",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath(normal_ref))
    if orm_ref is not None:
        material.GetPrim().CreateAttribute(
            "inputs:ORM_texture",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath(orm_ref))
    if include_child_shader:
        shader = UsdShade.Shader.Define(stage, f"{material_path}/albedo_texture")
        shader.CreateInput("filename", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(albedo_ref)
        )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


def test_generate_textures_small_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = urlparse("file://C:/textures/albedo.png")
    assert gt._file_uri_path(parsed) == Path("C:/textures/albedo.png")
    parsed = urlparse("file:///C:/textures/albedo.png")
    assert gt._file_uri_path(parsed) == Path("C:/textures/albedo.png")
    parsed = urlparse(r"file://C:\textures\albedo.png")
    assert gt._file_uri_path(parsed) == Path(r"C:\textures\albedo.png")

    diagnostics = [{"code": "A", "stage": "s", "material_name": "m"}]
    gt._append_diagnostic_once(diagnostics, dict(diagnostics[0]))
    assert len(diagnostics) == 1

    assert gt._allows_step1x_overlay_targets({"custom_parameters": "bad"}) is False
    assert gt._step1x_overlay_target_token(_unit("Body")) is None
    supported, errors, records, response_diagnostics = gt._preflight_step1x_targets(
        [_unit("Body")],
        {"custom_parameters": {"variant": "x"}},
    )
    assert len(supported) == 1
    assert errors == []
    assert records == {}
    assert response_diagnostics == []

    caps = gt._capabilities_from_config({"capabilities": "bad"})
    assert caps.image_conditioning is None
    assert gt._as_string_list("file:///ref.png") == ["file:///ref.png"]

    local = tmp_path / "ref image.png"
    local.write_text("ref", encoding="utf-8")
    gt._validate_conditioning_uri(str(local), field_name="reference_image_uris")
    monkeypatch.chdir(tmp_path)
    windows_like = tmp_path / "C:" / "textures" / "ref.png"
    windows_like.parent.mkdir(parents=True)
    windows_like.write_text("ref", encoding="utf-8")
    gt._validate_conditioning_uri(
        "C:/textures/ref.png",
        field_name="reference_image_uris",
    )

    assert gt._path_or_uri_to_uri("") == ""
    assert gt._path_or_uri_to_uri("s3://bucket/key.usd") == "s3://bucket/key.usd"
    assert gt._path_or_uri_to_uri(str(local)).startswith("file:")
    assert gt._service_source_asset_uri({}) == ""
    assert gt._normalized_path_set(None) == set()
    assert gt._normalized_path_set("/A/B/") == {"/A/B"}
    assert gt._normalized_path_set(7) == {"7"}
    assert gt._path_from_local_path_or_uri("") == Path()
    assert gt._path_from_local_path_or_uri("C:/textures/ref.png") == Path(
        "C:/textures/ref.png"
    )
    assert gt._path_from_local_path_or_uri("/tmp/ref%20map.png") == Path(
        "/tmp/ref%20map.png"
    )
    assert gt._positive_int(True) is None
    assert gt._positive_int(3.0) == 3
    assert gt._expected_size_tuple([8, 9]) == (8, 9)


def test_external_authoring_preflight_disabled_and_unavailable_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = {
        "adapter_id": "approved-painter-adapter",
        "workflow": "paint",
        "tool_name": "Approved Painter",
        "tool_version": "1.2.3",
    }
    assert (
        gt._external_authoring_spec_from_config(
            {"external_authoring": {"enabled": False}}
        )
        is None
    )

    unit = _unit()
    for config in (
        {"backend": "simple_image_gen", "external_authoring": external},
        {"backend": "service", "endpoint": "", "external_authoring": external},
    ):
        supported, errors, metadata, diagnostics, spec, receipt, feasibility = (
            gt._preflight_external_authoring([unit], config)
        )
        assert supported == []
        assert errors[0]["type"] == "ExternalAuthoringNoGo"
        assert metadata[unit.key]["metadata"]["external_authoring_preflight"] == (
            feasibility
        )
        assert diagnostics[0]["code"] == "EXTERNAL_AUTHORING_PREFLIGHT_UNAVAILABLE"
        assert spec is not None
        assert receipt is None
        assert feasibility["verdict"] == "no_go"

    receipt = ExternalAuthoringCapabilityReceipt.from_payload(
        {
            "schema_version": "texture-agent-external-authoring-capabilities.v1",
            "ready": True,
            "adapter_id": "approved-painter-adapter",
            "adapter_version": "4.5.6",
            "tool_name": "Approved Painter",
            "tool_version": "1.2.3",
            "headless": True,
            "license_status": "valid",
            "deployment_mode": "remote_headless",
            "environment_digest": "sha256:" + "a" * 64,
            "supported_workflows": ["paint"],
            "supported_map_channels": ["albedo"],
            "supported_auxiliary_artifacts": [],
            "seed_control": True,
            "deterministic_parameters": True,
            "normalized_output": "texture_variation_maps",
            "diagnostics": [],
        }
    )

    class _PreflightClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def preflight_external_authoring(self, _spec: object) -> object:
            return receipt

    from texture_agent.functions import rest_client

    monkeypatch.setattr(rest_client, "RestTextureVariationClient", _PreflightClient)
    result = gt._preflight_external_authoring(
        [unit],
        {
            "backend": "service",
            "endpoint": "http://approved-adapter.invalid",
            "external_authoring": external,
        },
    )
    assert result[-1]["verdict"] == "no_go"
    assert any(
        item["code"] == "EXTERNAL_AUTHORING_REPRODUCIBILITY_UNSUPPORTED"
        for item in result[-1]["diagnostics"]
    )


def test_resolve_texture_path_preserves_percent_sequences_in_bare_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scene.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    texture = tmp_path / "ref%20map.png"
    texture.write_text("png", encoding="utf-8")

    assert (
        gt._resolve_texture_path(
            "ref%20map.png",
            base_usd_path=source,
        )
        == texture.resolve()
    )


def test_uv_report_and_rebake_gate_edges(tmp_path: Path) -> None:
    assert gt._prepared_usd_from_uv_report({}) is None
    assert gt._prepared_usd_from_uv_report({"uv_preparation": {}}) is None
    assert (
        gt._prepared_usd_from_uv_report(
            {"uv_preparation": {"uv_report_path": "s3://bucket/report.json"}}
        )
        is None
    )
    assert (
        gt._prepared_usd_from_uv_report(
            {"uv_preparation": {"uv_report_path": str(tmp_path / "missing.json")}}
        )
        is None
    )
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert (
        gt._prepared_usd_from_uv_report(
            {"uv_preparation": {"uv_report_path": str(bad_json)}}
        )
        is None
    )
    empty_report = tmp_path / "empty.json"
    empty_report.write_text(json.dumps({"prepared_usd": ""}), encoding="utf-8")
    assert (
        gt._prepared_usd_from_uv_report(
            {"uv_preparation": {"uv_report_path": str(empty_report)}}
        )
        is None
    )

    context: dict[str, object] = {}
    assert gt._read_uv_report_payload(context) is None
    assert gt._read_uv_report_payload({"uv_preparation": {}}) is None
    assert (
        gt._read_uv_report_payload(
            {"uv_preparation": {"uv_report_path": str(tmp_path / "missing.json")}}
        )
        is None
    )
    assert (
        gt._read_uv_report_payload(
            {"uv_preparation": {"uv_report_path": str(bad_json)}}
        )
        is None
    )

    unit = _unit(base_color_texture="textures/body.png")
    assert (
        gt._should_rebake_source_textures({}, unit, {"uv_rebake_source_albedo": True})
        is False
    )
    assert (
        gt._should_rebake_source_textures(
            {"uv_preparation": {"uv_scope": "all"}},
            unit,
            {"uv_rebake_source_albedo": True},
        )
        is False
    )
    assert (
        gt._should_rebake_source_textures(
            {"uv_preparation": {"uv_scope": "target_prims"}},
            _unit(base_color_texture=None),
            {"uv_rebake_source_albedo": True},
        )
        is False
    )
    assert (
        gt._should_rebake_source_textures(
            {"uv_preparation": {"uv_scope": "target_prims", "target_prim_paths": []}},
            unit,
            {"uv_rebake_source_albedo": True},
        )
        is False
    )


def test_texture_path_and_rebake_ref_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert gt._resolve_texture_path(None, base_usd_path=tmp_path / "scene.usda") is None
    assert (
        gt._resolve_texture_path(
            "https://example.com/a.png", base_usd_path=tmp_path / "scene.usda"
        )
        is None
    )

    monkeypatch.setattr(
        gt,
        "extract_usdz_member_to_dir",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("extract failed")),
    )
    assert gt._extract_usdz_member(tmp_path / "missing.usdz", "textures/a.png") is None

    class FakePrim:
        def __init__(self, children: list[object] | None = None) -> None:
            self._children = children or []

        def GetChildren(self) -> list[object]:
            return self._children

    child = FakePrim()
    subtree = gt._iter_rebake_material_subtree(FakePrim([child]))
    assert child in subtree

    assert gt._coerce_rebake_texture_ref("textures/albedo.png") == "textures/albedo.png"
    assert gt._coerce_rebake_texture_ref(123) is None
    assert gt._coerce_rebake_texture_ref("") is None
    assert gt._coerce_rebake_texture_ref("@@") is None
    assert gt._coerce_rebake_texture_ref("scene.usdz[textures/albedo.png]") == (
        "scene.usdz[textures/albedo.png]"
    )
    assert gt._coerce_rebake_texture_ref("textures/readme.txt") is None


def test_rebake_discovery_and_material_authoring_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    with monkeypatch.context() as patch:
        patch.setattr(Usd.Stage, "Open", lambda _path: None)
        assert gt._find_unit_source_texture_refs(
            tmp_path / "missing.usda",
            _unit(base_color_texture="textures/body.png"),
        ) == {"albedo": "textures/body.png"}

    source_usd = tmp_path / "source.usda"
    _write_rebake_fixture_usd(
        source_usd,
        albedo_ref="textures/body.png",
        material_path="/World/Looks/Other",
    )
    assert (
        gt._find_unit_source_texture_refs(
            source_usd,
            _unit(material_path="/World/Looks/Missing", base_color_texture=None),
        )
        == {}
    )

    shader_usd = tmp_path / "shader_source.usda"
    _write_rebake_fixture_usd(
        shader_usd,
        albedo_ref="textures/body.png",
        include_child_shader=True,
    )
    refs = gt._find_unit_source_texture_refs(
        shader_usd,
        _unit(base_color_texture=None),
    )
    assert refs["albedo"].endswith("body.png")

    output_usd = tmp_path / "authored.usda"
    _write_rebake_fixture_usd(
        output_usd,
        albedo_ref="textures/body.png",
        normal_ref="textures/normal.png",
        orm_ref="textures/orm.png",
        include_child_shader=True,
    )
    albedo = _write_rgb(tmp_path / "rebaked" / "albedo.png", (1, 2, 3))
    gt._author_unit_source_usd(
        prepared_usd=output_usd,
        output_usd=tmp_path / "rewritten.usda",
        material_path="/World/Looks/Body",
        texture_paths={"albedo": Path(albedo)},
    )

    stage = Usd.Stage.CreateNew(str(tmp_path / "facevarying.usda"))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex,
    )
    primvar.Set([(0, 0), (1, 0), (0, 1)])
    assert len(gt._face_varying_uvs(mesh)) == 3

    indexed = UsdGeom.Mesh.Define(stage, "/World/Indexed")
    indexed.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    indexed.CreateFaceVertexIndicesAttr([0, 1, 2])
    indexed_pv = UsdGeom.PrimvarsAPI(indexed).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    indexed_pv.Set([(0, 0), (1, 0), (0, 1)])
    indexed_pv.SetIndices([0, 1, 2])
    assert len(gt._face_varying_uvs(indexed)) == 3

    non_shader = UsdGeom.Xform.Define(stage, "/World/Looks/Body/NotShader")
    assert not non_shader.GetPrim().IsA(UsdShade.Shader)


def test_materialization_and_localization_edges(tmp_path: Path) -> None:
    out_dir = tmp_path / "generated"
    out_dir.mkdir()
    task = GenerateTexturesTask

    assert task._localize_artifact_uri(
        "https://example.com/a.png", "Mat", "albedo", out_dir
    ) == ("https://example.com/a.png")
    assert task._localize_artifact_uri(
        str(tmp_path / "missing.png"), "Mat", "albedo", out_dir
    ).endswith("missing.png")

    small = _write_rgb(tmp_path / "small.png", (10, 20, 30), (1, 1))
    channel = task._channel_from_map_or_constant(small, (2, 2), 99)
    assert channel.shape == (2, 2)
    assert task._constant_to_byte(None, 123) == 123

    unit = _unit()
    diagnostics: list[dict[str, object]] = []
    blank = tmp_path / "blank.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(blank)
    assert (
        task._drop_blank_optional_map(
            path=str(blank),
            channel="normal",
            unit=unit,
            diagnostics=diagnostics,
        )
        == ""
    )
    assert diagnostics[0]["code"] == "BACKEND_TEXTURE_BLANK"
    assert (
        task._drop_blank_optional_map(
            path="s3://bucket/normal.png",
            channel="normal",
            unit=unit,
            diagnostics=[],
        )
        == ""
    )

    result = SimpleNamespace(
        metadata={"capabilities": {"multiview": False}},
        diagnostics=[],
        auxiliary_artifacts={},
    )
    diagnostics = []
    task._append_response_diagnostics(
        diagnostics=diagnostics,
        result=result,
        unit=unit,
        conditioning=Conditioning(
            turntable_video_uri="file:///turntable.mp4",
            multiview_image_uris=["file:///view.png"],
        ),
    )
    assert diagnostics[0]["details"]["unsupported_fields"] == [
        "turntable_video_uri",
        "multiview_image_uris",
    ]

    generated_albedo = _write_rgb(tmp_path / "fallback_albedo.png", (1, 2, 3))
    weathering_mask = _write_rgb(tmp_path / "weathering_mask.png", (255, 255, 255))
    service_result = GenerationResult(
        variant_asset_uri="file:///asset.usd",
        variant_name="Body",
        generated_textures=GeneratedTextures(
            albedo=generated_albedo, normal="", orm=""
        ),
        maps={},
        metadata={},
        auxiliary_artifacts={
            "masks": {
                "weathering": {
                    "uri": Path(weathering_mask).as_uri(),
                    "sha256": "f" * 64,
                }
            }
        },
        diagnostics=[],
    )
    textures, record, backend_map_paths = task._materialize_service_result(
        service_result,
        unit=unit,
        conditioning=Conditioning(),
        out_dir=out_dir,
        endpoint="http://backend",
    )
    assert Path(textures.albedo).is_file()
    assert Path(textures.normal).is_file()
    assert Path(textures.orm).is_file()
    assert record["endpoint"] == "http://backend"
    localized_weathering_mask = record["auxiliary_artifacts"]["masks"]["weathering"][
        "uri"
    ]
    assert Path(localized_weathering_mask).is_file()
    assert Path(localized_weathering_mask).parent == out_dir
    assert backend_map_paths == {"albedo": textures.albedo, "normal": "", "orm": ""}

    with pytest.raises(gt._BackendResultError):
        task._materialize_service_result(
            GenerationResult(
                variant_asset_uri="file:///asset.usd",
                variant_name="Body",
                generated_textures=GeneratedTextures(albedo="", normal="", orm=""),
                maps={},
                metadata={},
                diagnostics=[],
            ),
            unit=unit,
            conditioning=Conditioning(),
            out_dir=out_dir,
            endpoint="http://backend",
        )

    already_local = _write_rgb(tmp_path / "already.png", (3, 4, 5))
    localized = task._localize_textures(
        GeneratedTextures(albedo=already_local, normal="file:///missing.png", orm=""),
        "Body",
        out_dir,
        "http://backend",
    )
    assert localized.albedo == already_local
    assert localized.normal == "file:///missing.png"


def test_step1x_and_relative_path_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supported, errors, records, diagnostics = gt._preflight_step1x_targets(
        [_unit("Body")],
        {"engine": "step1x"},
    )
    assert [unit.key for unit in supported] == ["Body"]
    assert errors == []
    assert records == {}
    assert diagnostics == []

    report = tmp_path / "report.json"
    report.write_text(json.dumps({"prepared_usd": "prepared.usda"}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert (
        gt._prepared_usd_from_uv_report(
            {"uv_preparation": {"uv_report_path": "report.json"}}
        )
        == "prepared.usda"
    )

    local = tmp_path / "relative.txt"
    local.write_text("x", encoding="utf-8")
    assert gt._path_or_uri_to_uri("relative.txt") == local.resolve().as_uri()


def test_rebake_unit_skips_missing_optional_maps(tmp_path: Path) -> None:
    source_usd = tmp_path / "source.usda"
    prepared_usd = tmp_path / "prepared.usda"
    texture = tmp_path / "textures" / "body.png"
    _write_rgb(texture, (10, 20, 30))
    _write_rebake_fixture_usd(source_usd, albedo_ref="textures/body.png")
    _write_rebake_fixture_usd(prepared_usd, albedo_ref="textures/body.png")

    report = tmp_path / "uv_report.json"
    report.write_text(
        json.dumps(
            {
                "input_usd": str(source_usd),
                "prepared_usd": str(prepared_usd),
            }
        ),
        encoding="utf-8",
    )

    rebaked = gt._rebake_unit_source_textures(
        {"uv_preparation": {"uv_report_path": str(report)}},
        _unit(base_color_texture="textures/body.png"),
        {"mode": "per_material", "uv_rebake_source_albedo": True},
        tmp_path / "out",
    )

    assert rebaked.is_file()


def test_rebake_texture_geometry_guard_edges(tmp_path: Path) -> None:
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom

    source_texture = Path(_write_rgb(tmp_path / "source.png", (20, 40, 60)))

    source_usd = tmp_path / "source_missing.usda"
    prepared_usd = tmp_path / "prepared_missing.usda"
    source_stage = Usd.Stage.CreateNew(str(source_usd))
    source_stage.GetRootLayer().Save()
    prepared_stage = Usd.Stage.CreateNew(str(prepared_usd))
    prepared_stage.GetRootLayer().Save()
    missing_output = tmp_path / "missing_output.png"
    gt._rebake_texture_between_uv_sets(
        source_usd=source_usd,
        prepared_usd=prepared_usd,
        prim_paths=["/Missing"],
        source_texture=source_texture,
        output_path=missing_output,
        output_size=4,
    )
    assert missing_output.is_file()

    def write_line_face(path: Path) -> None:
        stage = Usd.Stage.CreateNew(str(path))
        mesh = UsdGeom.Mesh.Define(stage, "/World/Line")
        mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0)])
        mesh.CreateFaceVertexCountsAttr([2])
        mesh.CreateFaceVertexIndicesAttr([0, 1])
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        )
        primvar.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0)])
        stage.GetRootLayer().Save()

    line_source = tmp_path / "line_source.usda"
    line_prepared = tmp_path / "line_prepared.usda"
    write_line_face(line_source)
    write_line_face(line_prepared)
    line_output = tmp_path / "line_output.png"
    gt._rebake_texture_between_uv_sets(
        source_usd=line_source,
        prepared_usd=line_prepared,
        prim_paths=["/World/Line"],
        source_texture=source_texture,
        output_path=line_output,
        output_size=4,
    )
    assert line_output.is_file()

    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/TwoTri")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(1, 0, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(0, 1, 0),
        ]
    )
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex,
    )
    primvar.Set(
        [
            Gf.Vec2f(0, 0),
            Gf.Vec2f(1, 0),
            Gf.Vec2f(1, 1),
            Gf.Vec2f(0, 1),
        ]
    )
    assert len(gt._face_varying_uvs(mesh)) == 6

    source_image = Image.new("RGB", (4, 4), (1, 2, 3))
    dest_image = Image.new("RGB", (4, 4), (0, 0, 0))
    gt._paste_rebaked_triangle(
        source_image,
        dest_image,
        [(0, 0), (1, 0), (0, 1)],
        [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    )
    gt._paste_rebaked_triangle(
        source_image,
        Image.new("RGB", (1, 1), (0, 0, 0)),
        [(0, 0), (1, 0), (0, 1)],
        [(0, 0), (1, 1), (0.5, 0.5)],
    )
    gt._paste_rebaked_triangle(
        source_image,
        dest_image,
        [(0, 0), (1, 0), (0, 1)],
        [(0, 0), (1, 1), (0.5, 0.5)],
    )


def test_author_unit_source_usd_shader_input_edges(tmp_path: Path) -> None:
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    prepared = tmp_path / "prepared_author.usda"
    _write_rebake_fixture_usd(prepared, albedo_ref="textures/body.png")
    stage = Usd.Stage.Open(str(prepared))
    UsdGeom.Xform.Define(stage, "/World/Looks/Body/NotShader")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Body/albedo_texture")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("albedo_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/old.png")
    )
    shader.CreateInput("roughness_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("")
    )
    no_file = UsdShade.Shader.Define(stage, "/World/Looks/Body/normal_texture")
    no_file.CreateIdAttr("UsdUVTexture")
    no_file.CreateInput("normal_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("")
    )
    stage.GetRootLayer().Save()

    albedo = Path(_write_rgb(tmp_path / "rebaked" / "albedo.png", (1, 2, 3)))
    normal = Path(_write_rgb(tmp_path / "rebaked" / "normal.png", (4, 5, 6)))
    gt._author_unit_source_usd(
        prepared_usd=prepared,
        output_usd=tmp_path / "authored_edges.usda",
        material_path="/World/Looks/Body",
        texture_paths={"albedo": albedo, "normal": normal},
    )

    authored = Usd.Stage.Open(str(tmp_path / "authored_edges.usda"))
    authored_shader = UsdShade.Shader(
        authored.GetPrimAtPath("/World/Looks/Body/albedo_texture")
    )
    authored_no_file = UsdShade.Shader(
        authored.GetPrimAtPath("/World/Looks/Body/normal_texture")
    )
    assert authored_shader.GetInput("albedo_texture").Get().path == str(
        albedo.resolve()
    )
    assert authored_no_file.GetInput("file").Get().path == str(normal.resolve())


def test_author_unit_source_usd_skips_instance_proxy_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd

    class FakeLayer:
        def Export(self, path: str) -> bool:
            # _author_unit_source_usd exports the composed stage and checks the
            # result, so this must report success.
            Path(path).write_text("usd", encoding="utf-8")
            return True

        def Save(self) -> None:
            return None

    class FakeMaterialPrim:
        def __bool__(self):
            return True

        def IsInstanceProxy(self):
            return False

        def IsInstance(self):
            return False

        def IsInstanceable(self):
            return False

    class FakeInstanceProxyChild:
        def GetPath(self):
            return "/World/Looks/Body/Child"

        def IsInstanceProxy(self):
            return True

    class FakeStage:
        def GetRootLayer(self):
            return FakeLayer()

        def Flatten(self, *, addSourceFileComment: bool = True) -> FakeLayer:
            assert addSourceFileComment is False
            return FakeLayer()

        def GetPrimAtPath(self, _path: str):
            return FakeMaterialPrim()

        def TraverseAll(self):
            return [FakeInstanceProxyChild()]

    monkeypatch.setattr(Usd.Stage, "Open", lambda *_args, **_kwargs: FakeStage())

    gt._author_unit_source_usd(
        prepared_usd=tmp_path / "prepared.usda",
        output_usd=tmp_path / "output.usda",
        material_path="/World/Looks/Body",
        texture_paths={},
    )


def test_drop_blank_optional_map_returns_original_on_probe_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path(_write_rgb(tmp_path / "normal.png", (1, 2, 3)))
    monkeypatch.setattr(
        GenerateTexturesTask,
        "_image_size_and_blank",
        classmethod(lambda cls, _path: (_ for _ in ()).throw(RuntimeError("bad png"))),
    )

    assert GenerateTexturesTask._drop_blank_optional_map(
        path=str(path),
        channel="normal",
        unit=_unit(),
        diagnostics=[],
    ) == str(path)


def test_author_unit_source_usd_keeps_geometry_from_a_layered_asset(
    tmp_path: Path,
) -> None:
    """A layered source must not lose its geometry on the way out.

    Exporting only the root layer leaves relative composition arcs dangling once
    the output lands in a different directory. USD drops an unresolved sublayer
    with a warning rather than an error, so the export looks successful and the
    stage composes to nothing. On the acceptance asset that turned 13 meshes into
    0, and surfaced only as Step1X rejecting a target prim path that is valid in
    the source. See issue #957.
    """
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    # Geometry lives in a sublayer reached by a relative path, as in a packaged
    # CAD asset.
    payload_dir = tmp_path / "source"
    payload_dir.mkdir()
    geometry = payload_dir / "0" / "Body.usda"
    geometry.parent.mkdir(parents=True)
    geometry_stage = Usd.Stage.CreateNew(str(geometry))
    mesh = UsdGeom.Mesh.Define(geometry_stage, "/World/Body/Mesh")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 1, 0), Gf.Vec3f(0, 1, 0)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdShadeMaterial = pytest.importorskip("pxr.UsdShade").Material
    UsdShadeMaterial.Define(geometry_stage, "/World/Looks/Paint")
    geometry_stage.GetRootLayer().Save()

    root = payload_dir / "root.usda"
    root_stage = Usd.Stage.CreateNew(str(root))
    root_stage.GetRootLayer().subLayerPaths.append("0/Body.usda")
    root_stage.GetRootLayer().Save()

    # Hold the stage: Usd.Stage.Open(x).Traverse() in one expression releases the
    # stage before traversal and raises on an expired prim.
    check_stage = Usd.Stage.Open(str(root))
    assert len([p for p in check_stage.Traverse() if p.IsA(UsdGeom.Mesh)]) == 1, (
        "fixture must start with reachable geometry"
    )

    # Export into a different directory, as the rebake path does.
    output = tmp_path / "rebaked_source_assets" / "unit_source_asset.usda"
    output.parent.mkdir(parents=True)
    gt._author_unit_source_usd(
        prepared_usd=root,
        output_usd=output,
        material_path="/World/Looks/Paint",
        texture_paths={},
        required_prim_paths=["/World/Body/Mesh"],
    )

    exported = Usd.Stage.Open(str(output))
    assert exported is not None
    meshes = [p for p in exported.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert meshes, "geometry was lost: the relative sublayer did not survive export"
    assert exported.GetPrimAtPath("/World/Body/Mesh").IsValid()

    missing_output = tmp_path / "rebaked_source_assets" / "missing_target.usda"
    with pytest.raises(
        RuntimeError,
        match="lost required target prim path.*World/Missing",
    ):
        gt._author_unit_source_usd(
            prepared_usd=root,
            output_usd=missing_output,
            material_path="/World/Looks/Paint",
            texture_paths={},
            required_prim_paths=["/World/Missing"],
        )
    assert not missing_output.exists()


def test_author_unit_source_usd_raises_when_export_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed export must raise rather than leave an empty file behind.

    Continuing past a failed export is how issue #957 stayed hidden: the stage
    composed to nothing and the first symptom appeared much later, in a backend
    rejecting a prim path that was valid in the source.
    """
    pytest.importorskip("pxr")
    from pxr import Usd

    class FailingLayer:
        def Export(self, path: str) -> bool:
            Path(path).write_text("partial export", encoding="utf-8")
            return False

        def Save(self) -> None:
            return None

    class FailingStage:
        def GetRootLayer(self) -> FailingLayer:
            return FailingLayer()

        def Flatten(self, *, addSourceFileComment: bool = True) -> FailingLayer:
            assert addSourceFileComment is False
            return FailingLayer()

    monkeypatch.setattr(Usd.Stage, "Open", lambda *_a, **_k: FailingStage())

    output = tmp_path / "unit_source_asset.usda"
    with pytest.raises(RuntimeError, match="Failed to export rebaked source USD"):
        gt._author_unit_source_usd(
            prepared_usd=tmp_path / "prepared.usda",
            output_usd=output,
            material_path="/World/Looks/Paint",
            texture_paths={},
        )
    assert not output.exists()
    assert list(tmp_path.glob(".unit_source_asset.*")) == []
