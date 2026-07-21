# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GenerateTexturesTask error-propagation behavior.

The task uses a per-unit thread pool; a previous version logged each
per-unit failure but returned an empty result map without raising,
so the pipeline reported "complete" with zero textures generated.
These tests pin the corrected behavior: total failure raises, partial
failure logs a warning and returns whatever did succeed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from apps.texture_gen_service_common.artifacts import local_path_from_file_uri

from texture_agent.functions.detail_policy import apply_detail_policy_to_prompt
from texture_agent.functions.material_discovery import (
    MaterialInfo,
    PrimTextureUnit,
    expand_to_prim_units,
)
from texture_agent.functions.texture_generation import (
    GeneratedTextures,
    GenerationResult,
    JobStatus,
    MapArtifact,
)
from texture_agent.tasks import generate_textures as generate_textures_module
from texture_agent.tasks.generate_textures import (
    GenerateTexturesTask,
    _cached_texture_set,
    _custom_parameters_for_unit,
    _path_or_uri_to_uri,
    _prepared_usd_from_uv_report,
    _resolve_texture_path,
    _service_job_timeout_sec,
    _validate_conditioning_uri,
    _validate_textures_or_raise,
)


def _unit(key: str) -> PrimTextureUnit:
    return PrimTextureUnit(
        prim_path="",
        material_info=MaterialInfo(prim_path=f"/World/Looks/{key}", name=key),
        key=key,
        prompt=f"weathered {key}",
        opacity=0.85,
    )


def _service_unit(key: str = "Aluminum_Matte") -> PrimTextureUnit:
    bound_prim = (
        "/RootNode/SM_Ladder_A/SM_Ladder_A_Aluminum_0"
        if key == "Aluminum_Matte"
        else f"/RootNode/SM_Ladder_A/SM_Ladder_A_{key}_0"
    )
    material = MaterialInfo(
        prim_path=f"/RootNode/Looks/{key}",
        name=key,
        bound_prim_paths=[bound_prim],
        base_metalness=1.0,
        specular_roughness=0.25,
    )
    return PrimTextureUnit(
        prim_path="",
        material_info=material,
        key=key,
        prompt="brushed aluminum with visible scuffs",
        opacity=0.8,
        seed=11631,
    )


@pytest.mark.parametrize(
    ("include_canonical", "expected_label"),
    [(True, "canonical"), (False, "alias")],
)
def test_material_conditioning_uses_expansion_spec_precedence(
    include_canonical: bool,
    expected_label: str,
) -> None:
    material = MaterialInfo(
        prim_path="/World/Looks/Canonical",
        name="SharedName",
        material_alias_paths=["/World/Alias/Shared"],
    )
    specs = {
        "/World/Alias/Shared": {
            "prompt": "alias prompt",
            "reference_image_uris": ["file:///alias.png"],
        },
        "SharedName": {
            "prompt": "name prompt",
            "reference_image_uris": ["file:///name.png"],
        },
        "tu_runtime": {
            "prompt": "runtime prompt",
            "reference_image_uris": ["file:///runtime.png"],
        },
    }
    if include_canonical:
        specs["/World/Looks/Canonical"] = {
            "prompt": "canonical prompt",
            "reference_image_uris": ["file:///canonical.png"],
        }

    expanded = expand_to_prim_units([material], specs)
    unit = PrimTextureUnit(
        prim_path="",
        material_info=material,
        key="tu_runtime",
        prompt=expanded[0].prompt,
        opacity=0.85,
    )
    conditioning = generate_textures_module._conditioning_for_unit(
        unit,
        {"material_textures": specs},
        {},
        validate_uris=False,
    )

    assert expanded[0].prompt == f"{expected_label} prompt"
    assert conditioning.reference_image_uris == [f"file:///{expected_label}.png"]


def test_surface_only_detail_policy_prompt_is_idempotent() -> None:
    prompt = apply_detail_policy_to_prompt(
        "brushed aluminum with visible copper traces",
        "surface_only",
    )

    assert "brushed aluminum with." not in prompt
    assert "material swatch: brushed aluminum." in prompt
    assert apply_detail_policy_to_prompt(prompt, "surface_only") == prompt


def test_surface_only_detail_policy_does_not_trust_prefix_only_prompt() -> None:
    prompt = apply_detail_policy_to_prompt(
        "Surface-only material texture: green pcb material with labels",
        "surface_only",
    )

    assert "Avoid traces, vias, pads" in prompt
    assert "green material" in prompt
    assert "pcb" not in prompt.lower()
    assert "with labels" not in prompt.lower()


def test_surface_only_detail_policy_uses_plain_surface_fallback() -> None:
    prompt = apply_detail_policy_to_prompt("visible copper traces", "surface_only")

    assert "plain continuous material surface" in prompt


def test_detail_policy_reserved_custom_parameters_follow_typed_policy() -> None:
    unit = _unit("Steel")
    unit.detail_policy = "default"
    params = _custom_parameters_for_unit(
        {
            "custom_parameters": {
                "variant": "kept",
                "detail_policy": "surface_only",
                "forbidden_details": ["custom-only-sentinel"],
            }
        },
        unit,
    )

    assert params == {"variant": "kept"}

    unit.detail_policy = "surface_only"
    params = _custom_parameters_for_unit(
        {
            "custom_parameters": {
                "variant": "kept",
                "detail_policy": "default",
                "forbidden_details": ["custom-only-sentinel"],
            }
        },
        unit,
    )

    assert params["variant"] == "kept"
    assert params["detail_policy"] == "surface_only"
    assert "traces" in params["forbidden_details"]
    assert "custom-only-sentinel" not in params["forbidden_details"]


def test_resolve_texture_path_extracts_usdz_member(tmp_path: Path) -> None:
    import zipfile

    usdz = tmp_path / "scene.usdz"
    with zipfile.ZipFile(usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("textures/albedo.png", b"fake-png-bytes")

    extracted = _resolve_texture_path("./textures/albedo.png", base_usd_path=usdz)

    assert extracted is not None
    assert extracted.is_file()
    assert extracted.read_bytes() == b"fake-png-bytes"
    assert ".texture_agent_usdz_assets" in extracted.parts

    resolved_ref = f"{usdz}[textures/albedo.png]"
    assert _resolve_texture_path(resolved_ref, base_usd_path=usdz) == extracted


def test_resolve_texture_path_extracts_relative_usdz_package_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zipfile

    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    usdz = stage_dir / "asset.usdz"
    with zipfile.ZipFile(usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("textures/albedo.png", b"relative-package-texture")

    unrelated_cwd = tmp_path / "cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    extracted = _resolve_texture_path(
        "asset.usdz[textures/albedo.png]",
        base_usd_path=stage_dir / "scene.usda",
    )

    assert extracted is not None
    assert extracted.is_file()
    assert extracted.read_bytes() == b"relative-package-texture"


def test_resolve_texture_path_skips_oversized_usdz_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zipfile

    usdz = tmp_path / "scene.usdz"
    with zipfile.ZipFile(usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("textures/albedo.png", b"abcd")

    monkeypatch.setattr(generate_textures_module, "_MAX_PACKAGE_TEXTURE_BYTES", 3)

    extracted = _resolve_texture_path("./textures/albedo.png", base_usd_path=usdz)

    assert extracted == tmp_path / "textures" / "albedo.png"
    assert not extracted.exists()
    assert not (
        tmp_path / ".texture_agent_usdz_assets" / "scene" / "textures" / "albedo.png"
    ).exists()


def _write_rebake_fixture_usd(
    path: Path,
    *,
    albedo_ref: str,
    normal_ref: str | None = None,
    orm_ref: str | None = None,
    uvs: list[tuple[float, float]],
) -> None:
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
    st.Set([Gf.Vec2f(u, v) for u, v in uvs])

    material = UsdShade.Material.Define(stage, "/World/Looks/Body")
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
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


def _texture_sibling(albedo: str, suffix: str) -> str:
    if not albedo:
        return ""
    if albedo.endswith("_albedo.png"):
        return albedo.removesuffix("_albedo.png") + f"_{suffix}.png"
    return albedo


def _ok_status(albedo: str, key: str) -> JobStatus:
    textures = GeneratedTextures(
        albedo=albedo,
        normal=_texture_sibling(albedo, "normal"),
        orm=_texture_sibling(albedo, "orm"),
    )
    return JobStatus(
        job_id=f"job-{key}",
        status="completed",
        result=GenerationResult(
            variant_asset_uri=f"/tmp/{key}.usd",
            variant_name=key,
            generated_textures=textures,
        ),
    )


def _make_real_texture_set(directory: Path, key: str) -> str:
    """Materialize a tiny PBR set so the post-gen existence checks pass."""
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    albedo = directory / f"{key}_albedo.png"
    normal = directory / f"{key}_normal.png"
    orm = directory / f"{key}_orm.png"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(albedo)
    Image.new("RGB", (4, 4), (128, 128, 255)).save(normal)
    Image.new("RGB", (4, 4), (255, 200, 0)).save(orm)
    return str(albedo)


def _write_rgb(path: Path, color: tuple[int, int, int]) -> str:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)
    return str(path)


def _write_sized_rgb(
    path: Path,
    color: tuple[int, int, int],
    size: tuple[int, int],
) -> str:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return str(path)


def test_localize_artifact_uri_decodes_file_uri_paths(tmp_path: Path) -> None:
    source = _write_rgb(tmp_path / "backend outputs" / "albedo map.png", (1, 2, 3))
    out_dir = tmp_path / "generated"
    out_dir.mkdir()
    uri = f"file://{source.replace(' ', '%20')}"

    localized = GenerateTexturesTask._localize_artifact_uri(
        uri,
        "Aluminum_Matte",
        "albedo",
        out_dir,
    )

    assert Path(localized) == out_dir / "Aluminum_Matte_albedo.png"
    assert Path(localized).is_file()


def test_validate_textures_accepts_service_upscaled_maps(tmp_path: Path) -> None:
    textures = GeneratedTextures(
        albedo=_write_sized_rgb(tmp_path / "albedo.png", (10, 20, 30), (8, 8)),
        normal=_write_sized_rgb(tmp_path / "normal.png", (128, 128, 255), (8, 8)),
        orm=_write_sized_rgb(tmp_path / "orm.png", (255, 200, 0), (8, 8)),
    )

    _validate_textures_or_raise(
        "Aluminum_Matte",
        textures,
        expected_size=4,
    )


def test_validate_textures_rejects_service_undersized_maps(tmp_path: Path) -> None:
    textures = GeneratedTextures(
        albedo=_write_sized_rgb(tmp_path / "albedo.png", (10, 20, 30), (2, 2)),
        normal=_write_sized_rgb(tmp_path / "normal.png", (128, 128, 255), (4, 4)),
        orm=_write_sized_rgb(tmp_path / "orm.png", (255, 200, 0), (4, 4)),
    )

    with pytest.raises(RuntimeError, match=r"expected at least \(4, 4\)"):
        _validate_textures_or_raise(
            "Aluminum_Matte",
            textures,
            expected_size=4,
        )


def test_localize_artifact_uri_decodes_local_paths(tmp_path: Path) -> None:
    source = _write_rgb(tmp_path / "backend outputs" / "albedo map.png", (1, 2, 3))
    out_dir = tmp_path / "generated"
    out_dir.mkdir()

    localized = GenerateTexturesTask._localize_artifact_uri(
        source.replace(" ", "%20"),
        "Aluminum_Matte",
        "albedo",
        out_dir,
    )

    assert Path(localized) == out_dir / "Aluminum_Matte_albedo.png"
    assert Path(localized).is_file()


def test_windows_drive_source_asset_path_becomes_file_uri() -> None:
    assert _path_or_uri_to_uri(r"C:\refs\asset.usd") == "file:///C:/refs/asset.usd"


def test_windows_drive_conditioning_path_is_validated_as_local_file() -> None:
    with pytest.raises(FileNotFoundError):
        _validate_conditioning_uri(
            r"Z:\definitely-missing\reference.png",
            field_name="reference_image_uris",
        )


def test_windows_drive_uv_report_path_is_read_as_local_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = r"C:\work\uv_report.json"

    def fake_exists(self: Path) -> bool:
        return str(self) == report_path

    def fake_read_text(self: Path, *, encoding: str) -> str:
        assert str(self) == report_path
        assert encoding == "utf-8"
        return json.dumps({"prepared_usd": r"C:\work\prepared_asset.usd"})

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "read_text", fake_read_text)

    assert (
        _prepared_usd_from_uv_report(
            {"uv_preparation": {"uv_report_path": f" {report_path} "}}
        )
        == r"C:\work\prepared_asset.usd"
    )


def test_service_job_timeout_accepts_config_aliases_only() -> None:
    assert _service_job_timeout_sec({"job_timeout_sec": "7200"}) == 7200
    assert (
        _service_job_timeout_sec(
            {
                "custom_parameters": {"timeout_sec": 5400},
            },
        )
        == 3600
    )
    assert _service_job_timeout_sec({"service_timeout_sec": 1800}) == 1800


def _fail_status(key: str, message: str) -> JobStatus:
    return JobStatus(
        job_id=f"job-{key}",
        status="failed",
        error_message=message,
    )


def _projection_status_from_maps(
    key: str,
    *,
    maps: dict[str, str],
    metadata: dict | None = None,
    diagnostics: list[dict] | None = None,
    status: str = "completed",
    error_message: str | None = None,
) -> JobStatus:
    metadata_payload = dict(metadata or {})
    auxiliary_artifacts = metadata_payload.pop("auxiliary_artifacts", {})
    artifacts = {
        channel: MapArtifact(uri=uri, width=4, height=4)
        for channel, uri in maps.items()
    }
    return JobStatus(
        job_id=f"job-{key}",
        status=status,
        error_message=error_message,
        result=GenerationResult(
            variant_asset_uri="/tmp/prepared.usd",
            variant_name=key,
            generated_textures=GeneratedTextures(
                albedo=maps.get("albedo", ""),
                normal=maps.get("normal", ""),
                orm=maps.get("orm", ""),
            ),
            maps=artifacts,
            auxiliary_artifacts=auxiliary_artifacts,
            metadata=metadata_payload,
            diagnostics=diagnostics or [],
        ),
    )


@pytest.fixture
def context_factory(tmp_path):
    """Build a minimal context dict + texture_config for the task."""

    def _make(units: list[PrimTextureUnit]) -> dict:
        return {
            "prim_texture_units": units,
            "working_dir": str(tmp_path),
            "usd_path": "/tmp/asset.usd",
            "texture_config": {
                "backend": "simple_image_gen",
                "image_gen": {"backend": "nim"},
                "skip_existing": False,
                "workers": 2,
            },
        }

    return _make


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_service_backend_sends_projection_contract_and_packs_orm(
    mock_client_cls, tmp_path
):
    """Service backend uses normalized scope, conditioning, and channel maps."""
    albedo = _write_rgb(tmp_path / "backend" / "albedo.png", (10, 20, 30))
    roughness = _write_rgb(tmp_path / "backend" / "roughness.png", (40, 40, 40))
    metalness = _write_rgb(tmp_path / "backend" / "metalness.png", (200, 200, 200))
    global_ref = _write_rgb(tmp_path / "refs" / "global.png", (1, 2, 3))
    material_ref = _write_rgb(tmp_path / "refs" / "material.png", (4, 5, 6))
    global_view = _write_rgb(tmp_path / "refs" / "view_global.png", (9, 8, 7))
    view_ref = _write_rgb(tmp_path / "refs" / "view0.png", (7, 8, 9))
    turntable = tmp_path / "refs" / "turntable.mp4"
    turntable.write_bytes(b"fake-video")
    original_usd = tmp_path / "input.usd"
    original_usd.write_text("#usda 1.0\n", encoding="utf-8")
    prepared_usd = tmp_path / "prepared_input.usd"
    prepared_usd.write_text("#usda 1.0\n", encoding="utf-8")
    uv_report = tmp_path / "prepared" / "uv_report.json"
    uv_report.parent.mkdir(parents=True)
    uv_report.write_text(
        json.dumps(
            {
                "input_usd": str(original_usd),
                "prepared_usd": str(prepared_usd),
            }
        ),
        encoding="utf-8",
    )

    mock_client = mock_client_cls.return_value
    mock_client.generate.return_value = _projection_status_from_maps(
        "Aluminum_Matte",
        maps={
            "albedo": f"file://{albedo}",
            "roughness": f"file://{roughness}",
            "metalness": f"file://{metalness}",
        },
        metadata={
            "backend_name": "fake_projection_backend",
            "model": "fake-projection-v1",
            "capabilities": {"normal_map": False, "orm": False},
            "degraded_channels": ["normal", "orm"],
        },
    )

    unit = _service_unit()
    context = {
        "prim_texture_units": [unit],
        "working_dir": str(tmp_path),
        "usd_path": str(original_usd),
        "uv_preparation": {"uv_report_path": str(uv_report)},
        "texture_config": {
            "backend": "service",
            "endpoint": "http://fake-backend",
            "engine": "fake_projection",
            "size": 4,
            "skip_existing": False,
            "workers": 1,
            "reference_image_uris": [f"file://{global_ref}"],
            "turntable_video_uri": f"file://{turntable}",
            "multiview_image_uris": [f"file://{global_view}"],
            "custom_parameters": {"variant": "success_full_pbr"},
            "capabilities": {
                "image_conditioning": True,
                "normal_map": True,
                "orm": True,
                "geometry_output": "none",
            },
        },
        "material_textures": {
            "Aluminum_Matte": {
                "reference_image_uris": [f"file://{material_ref}"],
                "multiview_image_uris": [f"file://{view_ref}"],
            },
            "Other_Material": {"prompt": "this material is outside selected units"},
        },
    }

    result = GenerateTexturesTask().run(context)

    generated = result["generated_textures"]["Aluminum_Matte"]
    assert Path(generated.albedo).is_file()
    assert Path(generated.normal).is_file()
    assert Path(generated.orm).is_file()

    from PIL import Image

    orm = Image.open(generated.orm).convert("RGB")
    assert orm.getpixel((0, 0)) == (255, 40, 200)

    call = mock_client.generate.call_args.kwargs
    assert mock_client.generate.call_count == 1
    assert call["source_asset_uri"] == prepared_usd.resolve().as_uri()
    assert call["target"].material_name == "Aluminum_Matte"
    assert call["target"].material_path == "/RootNode/Looks/Aluminum_Matte"
    assert call["target"].prim_paths == ["/RootNode/SM_Ladder_A/SM_Ladder_A_Aluminum_0"]
    assert call["target"].mode == "per_material"
    assert call["target"].strict_scope is True
    assert call["conditioning"].text_prompt == "brushed aluminum with visible scuffs"
    assert call["conditioning"].reference_image_uris == [
        f"file://{global_ref}",
        f"file://{material_ref}",
    ]
    assert call["conditioning"].turntable_video_uri == f"file://{turntable}"
    assert call["conditioning"].multiview_image_uris == [
        f"file://{global_view}",
        f"file://{view_ref}",
    ]
    assert call["config"].seed == 11631
    assert call["config"].texture_size == 4
    assert call["config"].engine == "fake_projection"
    assert call["config"].custom_parameters == {"variant": "success_full_pbr"}
    assert call["capabilities"].geometry_output == "none"

    backend_record = result["projection_backend_results"]["Aluminum_Matte"]
    assert backend_record["maps"]["roughness"]["width"] == 4
    assert backend_record["capabilities"]["orm"] is False
    assert backend_record["degraded_channels"] == ["normal", "orm"]
    assert any(
        item["code"] == "BACKEND_MAP_MISSING" for item in backend_record["diagnostics"]
    )


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_service_backend_propagates_surface_only_detail_policy(
    mock_client_cls, tmp_path: Path
) -> None:
    albedo = _write_rgb(tmp_path / "backend" / "albedo.png", (10, 120, 40))
    input_usd = tmp_path / "input.usd"
    input_usd.write_text("#usda 1.0\n", encoding="utf-8")
    mock_client = mock_client_cls.return_value
    mock_client.generate.return_value = _projection_status_from_maps(
        "Plastic_Green",
        maps={"albedo": f"file://{albedo}"},
        metadata={
            "backend_name": "fake_projection_backend",
            "capabilities": {"normal_map": False, "orm": False},
        },
    )
    unit = _service_unit("Plastic_Green")
    unit.prompt = apply_detail_policy_to_prompt(
        (
            "green PCB material texture, realistic printed circuit board "
            "surface, solder mask coating, electronic board material"
        ),
        "surface_only",
    )
    unit.detail_policy = "surface_only"
    context = {
        "prim_texture_units": [unit],
        "working_dir": str(tmp_path),
        "usd_path": str(input_usd),
        "texture_config": {
            "backend": "service",
            "endpoint": "http://fake-backend",
            "size": 4,
            "skip_existing": False,
            "workers": 1,
            "custom_parameters": {"variant": "surface-check"},
            "capabilities": {"normal_map": False, "orm": False},
        },
        "material_textures": {},
    }

    result = GenerateTexturesTask().run(context)

    call = mock_client.generate.call_args.kwargs
    assert "Surface-only material texture:" in call["conditioning"].text_prompt
    assert "Avoid traces, vias, pads" in call["conditioning"].text_prompt
    assert "pcb" not in call["conditioning"].text_prompt.lower()
    assert call["config"].custom_parameters["variant"] == "surface-check"
    assert call["config"].custom_parameters["detail_policy"] == "surface_only"
    assert "traces" in call["config"].custom_parameters["forbidden_details"]
    metadata = result["projection_backend_results"]["Plastic_Green"]["metadata"]
    assert metadata["detail_policy"] == "surface_only"


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_service_backend_rebakes_scoped_uv_source_texture_set(
    mock_client_cls,
    tmp_path: Path,
) -> None:
    source_albedo = Path(
        _write_sized_rgb(
            tmp_path / "textures" / "body_source.png",
            (42, 84, 126),
            (8, 8),
        )
    )
    source_normal = Path(
        _write_sized_rgb(
            tmp_path / "textures" / "body_normal.png",
            (128, 128, 255),
            (8, 8),
        )
    )
    source_orm = Path(
        _write_sized_rgb(
            tmp_path / "textures" / "body_orm.png",
            (255, 96, 0),
            (8, 8),
        )
    )
    source_usd = tmp_path / "source.usda"
    prepared_usd = tmp_path / "prepared.usda"
    _write_rebake_fixture_usd(
        source_usd,
        albedo_ref="textures/body_source.png",
        normal_ref="textures/body_normal.png",
        orm_ref="textures/body_orm.png",
        uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    )
    _write_rebake_fixture_usd(
        prepared_usd,
        albedo_ref="textures/body_source.png",
        normal_ref="textures/body_normal.png",
        orm_ref="textures/body_orm.png",
        uvs=[(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)],
    )
    uv_report = tmp_path / "prepared" / "uv_report.json"
    uv_report.parent.mkdir()
    uv_report.write_text(
        json.dumps(
            {
                "input_usd": str(source_usd),
                "prepared_usd": str(prepared_usd),
            }
        ),
        encoding="utf-8",
    )

    backend_albedo = _write_rgb(tmp_path / "backend" / "Body_albedo.png", (1, 2, 3))
    backend_normal = _write_rgb(
        tmp_path / "backend" / "Body_normal.png",
        (128, 128, 255),
    )
    backend_orm = _write_rgb(tmp_path / "backend" / "Body_orm.png", (255, 64, 8))
    mock_client_cls.return_value.generate.return_value = _projection_status_from_maps(
        "Body",
        maps={
            "albedo": f"file://{backend_albedo}",
            "normal": f"file://{backend_normal}",
            "orm": f"file://{backend_orm}",
        },
    )

    unit = PrimTextureUnit(
        prim_path="",
        material_info=MaterialInfo(
            prim_path="/World/Looks/Body",
            name="Body",
            bound_prim_paths=["/World/BodyMesh"],
            base_color_texture="textures/body_source.png",
            has_existing_texture=True,
        ),
        key="Body",
        prompt="cobalt blue painted toolbox body",
        opacity=0.9,
    )
    context = {
        "prim_texture_units": [unit],
        "working_dir": str(tmp_path),
        "usd_path": str(source_usd),
        "uv_preparation": {
            "uv_scope": "target_prims",
            "target_prim_paths": ["/World/BodyMesh"],
            "uv_report_path": str(uv_report),
        },
        "texture_config": {
            "backend": "service",
            "endpoint": "http://fake-backend",
            "size": 4,
            "skip_existing": False,
            "workers": 1,
            "uv_rebake_source_albedo": True,
            "uv_rebake_size": 8,
        },
    }

    result = GenerateTexturesTask().run(context)

    assert result["generated_textures"]["Body"].albedo
    call = mock_client_cls.return_value.generate.call_args.kwargs
    assert call["source_asset_uri"] != prepared_usd.resolve().as_uri()
    source_asset_path = local_path_from_file_uri(call["source_asset_uri"])
    assert source_asset_path is not None
    assert source_asset_path.name == "Body_source_asset.usda"
    assert source_asset_path.is_file()

    rebaked_albedo = (
        tmp_path
        / "generated"
        / "rebaked_source_textures"
        / "Body_source_albedo_rebaked.png"
    )
    rebaked_normal = (
        tmp_path
        / "generated"
        / "rebaked_source_textures"
        / "Body_source_normal_rebaked.png"
    )
    rebaked_orm = (
        tmp_path
        / "generated"
        / "rebaked_source_textures"
        / "Body_source_orm_rebaked.png"
    )
    assert rebaked_albedo.is_file()
    assert rebaked_normal.is_file()
    assert rebaked_orm.is_file()
    assert source_albedo.is_file()
    assert source_normal.is_file()
    assert source_orm.is_file()

    from PIL import Image
    from pxr import Usd

    assert Image.open(rebaked_albedo).size == (8, 8)
    assert Image.open(rebaked_normal).size == (8, 8)
    assert Image.open(rebaked_orm).size == (8, 8)
    stage = Usd.Stage.Open(str(source_asset_path))
    material = stage.GetPrimAtPath("/World/Looks/Body")
    authored = material.GetAttribute("inputs:base_color_texture_file").Get()
    assert Path(authored.path) == rebaked_albedo.resolve()
    authored_normal = material.GetAttribute("inputs:normalmap_texture").Get()
    assert Path(authored_normal.path) == rebaked_normal.resolve()
    authored_orm = material.GetAttribute("inputs:ORM_texture").Get()
    assert Path(authored_orm.path) == rebaked_orm.resolve()


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_service_backend_materializes_albedo_only_degraded_response(
    mock_client_cls, tmp_path
):
    """Albedo-only service responses stay compatible via fallback PBR maps."""
    albedo = _write_rgb(tmp_path / "backend" / "albedo.png", (10, 20, 30))
    reference = _write_rgb(tmp_path / "refs" / "reference.png", (1, 2, 3))
    prepared_usd = tmp_path / "prepared_input.usd"
    prepared_usd.write_text("#usda 1.0\n", encoding="utf-8")

    mock_client = mock_client_cls.return_value
    mock_client.generate.return_value = _projection_status_from_maps(
        "Aluminum_Matte",
        maps={"albedo": f"file://{albedo}"},
        metadata={
            "backend_name": "fake_projection_backend",
            "capabilities": {
                "image_conditioning": False,
                "multiview": False,
                "normal_map": False,
                "orm": False,
                "geometry_output": "replacement",
            },
            "coverage": {"target_coverage": 0.41},
            "degraded_channels": ["normal", "orm"],
            "auxiliary_artifacts": {
                "geometry": [
                    {
                        "label": "replacement",
                        "uri": "file:///backend/replacement.usd",
                    }
                ]
            },
        },
    )

    context = {
        "prim_texture_units": [_service_unit()],
        "working_dir": str(tmp_path),
        "usd_path": str(prepared_usd),
        "texture_config": {
            "backend": "service",
            "endpoint": "http://fake-backend",
            "size": 4,
            "skip_existing": False,
            "workers": 1,
            "reference_image_uris": [f"file://{reference}"],
        },
    }

    result = GenerateTexturesTask().run(context)
    generated = result["generated_textures"]["Aluminum_Matte"]

    from PIL import Image

    assert Image.open(generated.normal).convert("RGB").getpixel((0, 0)) == (
        128,
        128,
        255,
    )
    assert Image.open(generated.orm).convert("RGB").getpixel((0, 0)) == (
        255,
        64,
        255,
    )

    diagnostics = result["projection_backend_results"]["Aluminum_Matte"]["diagnostics"]
    codes = {item["code"] for item in diagnostics}
    assert "BACKEND_CONDITIONING_UNSUPPORTED" in codes
    assert "BACKEND_MAP_MISSING" in codes
    assert "BACKEND_LOW_COVERAGE" in codes
    assert "BACKEND_GEOMETRY_IGNORED" in codes


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_service_backend_records_partial_failure_below_threshold(
    mock_client_cls, tmp_path
):
    good_albedo = _write_rgb(tmp_path / "backend" / "good_albedo.png", (1, 2, 3))
    good_normal = _write_rgb(tmp_path / "backend" / "good_normal.png", (128, 128, 255))
    good_orm = _write_rgb(tmp_path / "backend" / "good_orm.png", (255, 20, 10))
    bad_normal = _write_rgb(tmp_path / "backend" / "bad_normal.png", (128, 128, 255))
    prepared_usd = tmp_path / "prepared_input.usd"
    prepared_usd.write_text("#usda 1.0\n", encoding="utf-8")

    def side_effect(*args, **kwargs):
        key = kwargs["config"].variant_name
        if key == "Good":
            return _projection_status_from_maps(
                key,
                maps={
                    "albedo": f"file://{good_albedo}",
                    "normal": f"file://{good_normal}",
                    "orm": f"file://{good_orm}",
                },
            )
        return _projection_status_from_maps(
            key,
            maps={"normal": f"file://{bad_normal}"},
            status="failed",
            error_message="Backend did not return required albedo map.",
            diagnostics=[
                {
                    "schema_version": "texture-agent-diagnostic.v1",
                    "code": "BACKEND_MAP_MISSING",
                    "severity": "error",
                    "stage": "generate_textures",
                    "prim_path": "/RootNode/SM_Ladder_A/SM_Ladder_A_Bad_0",
                    "material_name": "Bad",
                    "message": "Backend did not return required albedo map.",
                    "recommended_action": "Retry.",
                    "details": {"missing_maps": ["albedo"]},
                }
            ],
        )

    mock_client_cls.return_value.generate.side_effect = side_effect
    context = {
        "prim_texture_units": [_service_unit("Good"), _service_unit("Bad")],
        "working_dir": str(tmp_path),
        "usd_path": str(prepared_usd),
        "texture_config": {
            "backend": "service",
            "endpoint": "http://fake-backend",
            "size": 4,
            "skip_existing": False,
            "workers": 1,
        },
    }

    result = GenerateTexturesTask().run(context)

    assert set(result["generated_textures"]) == {"Good"}
    assert result["generate_textures_failed_count"] == 1
    codes = {item["code"] for item in result["generate_textures_diagnostics"]}
    assert "BACKEND_MAP_MISSING" in codes
    assert "BACKEND_PARTIAL_FAILURE" not in codes


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_service_backend_failed_result_preserves_backend_error_diagnostic(
    mock_client_cls,
    tmp_path,
):
    prepared_usd = tmp_path / "prepared_input.usd"
    prepared_usd.write_text("#usda 1.0\n", encoding="utf-8")

    mock_client_cls.return_value.generate.return_value = _projection_status_from_maps(
        "Aluminum_Matte",
        maps={},
        status="failed",
        error_message="STEP1X_COMMAND_FAILED: CUDA assert while rendering selected mesh.",
        diagnostics=[
            {
                "code": "STEP1X_COMMAND_FAILED",
                "severity": "error",
                "message": "CUDA assert while rendering selected mesh.",
            }
        ],
    )
    context = {
        "prim_texture_units": [_service_unit()],
        "working_dir": str(tmp_path),
        "usd_path": str(prepared_usd),
        "texture_config": {
            "backend": "service",
            "endpoint": "http://fake-backend",
            "size": 4,
            "skip_existing": False,
            "workers": 1,
        },
    }

    with pytest.raises(
        RuntimeError,
        match=r"Service failed for Aluminum_Matte: STEP1X_COMMAND_FAILED",
    ):
        GenerateTexturesTask().run(context)

    backend_record = context["projection_backend_results"]["Aluminum_Matte"]
    codes = {item["code"] for item in backend_record["diagnostics"]}
    assert "STEP1X_COMMAND_FAILED" in codes
    assert "BACKEND_MAP_MISSING" not in codes


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_step1x_overlay_target_rejects_before_backend_dispatch(
    mock_client_cls,
    tmp_path,
):
    decal_material = MaterialInfo(
        prim_path="/RootNode/Materials/M_SteelRollingScaffold_A_Decals_01",
        name="M_SteelRollingScaffold_A_Decals_01",
        bound_prim_paths=["/RootNode/Geometry/SM_SteelRollingScaffold_A01_Decals_01"],
        has_existing_texture=True,
    )
    unit = PrimTextureUnit(
        prim_path="",
        material_info=decal_material,
        key=decal_material.name,
        prompt="worn scaffold decals",
        opacity=0.9,
        seed=331003,
    )
    context = {
        "prim_texture_units": [unit],
        "working_dir": str(tmp_path),
        "usd_path": str(tmp_path / "prepared_input.usd"),
        "texture_config": {
            "backend": "service",
            "endpoint": "http://texture-gen-step1x:8000",
            "engine": "step1x",
            "skip_existing": False,
            "workers": 1,
            "failure_threshold": 0.0,
        },
    }

    with pytest.raises(RuntimeError, match="UnsupportedStep1XTarget"):
        GenerateTexturesTask().run(context)

    mock_client_cls.assert_not_called()
    assert context["generate_textures_failed_count"] == 1
    assert context["generate_textures_attempted_count"] == 1
    assert context["generate_textures_errors"][0]["material"] == decal_material.name
    backend_record = context["projection_backend_results"][decal_material.name]
    assert backend_record["maps"] == {}
    assert backend_record["skipped_before_backend_launch"] is True
    assert backend_record["metadata"]["skipped_before_backend_launch"] is True
    assert backend_record["target"]["prim_paths"] == [
        "/RootNode/Geometry/SM_SteelRollingScaffold_A01_Decals_01"
    ]
    codes = {item["code"] for item in context["generate_textures_diagnostics"]}
    assert "STEP1X_UNSUPPORTED_OVERLAY_TARGET" in codes


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_service_backend_status_failure_without_specific_diagnostic_falls_back(
    mock_client_cls,
    tmp_path,
):
    albedo = _write_rgb(tmp_path / "backend" / "albedo.png", (1, 2, 3))
    normal = _write_rgb(tmp_path / "backend" / "normal.png", (128, 128, 255))
    orm = _write_rgb(tmp_path / "backend" / "orm.png", (255, 20, 10))
    prepared_usd = tmp_path / "prepared_input.usd"
    prepared_usd.write_text("#usda 1.0\n", encoding="utf-8")

    mock_client_cls.return_value.generate.return_value = _projection_status_from_maps(
        "Aluminum_Matte",
        maps={
            "albedo": f"file://{albedo}",
            "normal": f"file://{normal}",
            "orm": f"file://{orm}",
        },
        status="failed",
        error_message="Backend finalization failed.",
    )
    context = {
        "prim_texture_units": [_service_unit()],
        "working_dir": str(tmp_path),
        "usd_path": str(prepared_usd),
        "texture_config": {
            "backend": "service",
            "endpoint": "http://fake-backend",
            "size": 4,
            "skip_existing": False,
            "workers": 1,
        },
    }

    with pytest.raises(RuntimeError, match=r"1/1 texture generation requests failed"):
        GenerateTexturesTask().run(context)

    assert context["projection_backend_results"]["Aluminum_Matte"]["diagnostics"] == []
    codes = {item["code"] for item in context["generate_textures_diagnostics"]}
    assert "BACKEND_PARTIAL_FAILURE" in codes


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_service_backend_status_failure_with_warning_diagnostic_falls_back(
    mock_client_cls,
    tmp_path,
):
    albedo = _write_rgb(tmp_path / "backend" / "albedo.png", (1, 2, 3))
    normal = _write_rgb(tmp_path / "backend" / "normal.png", (128, 128, 255))
    orm = _write_rgb(tmp_path / "backend" / "orm.png", (255, 20, 10))
    prepared_usd = tmp_path / "prepared_input.usd"
    prepared_usd.write_text("#usda 1.0\n", encoding="utf-8")

    mock_client_cls.return_value.generate.return_value = _projection_status_from_maps(
        "Aluminum_Matte",
        maps={
            "albedo": f"file://{albedo}",
            "normal": f"file://{normal}",
            "orm": f"file://{orm}",
        },
        status="failed",
        error_message="Backend finalization failed.",
        diagnostics=[
            {
                "schema_version": "texture-agent-diagnostic.v1",
                "code": "BACKEND_LOW_COVERAGE",
                "severity": "warning",
                "stage": "generate_textures",
                "prim_path": "/RootNode/SM_Ladder_A/SM_Ladder_A_Aluminum_0",
                "material_name": "Aluminum_Matte",
                "message": "Backend reported low target coverage.",
                "recommended_action": "Inspect coverage masks.",
                "details": {"target_coverage": 0.41},
            }
        ],
    )
    context = {
        "prim_texture_units": [_service_unit()],
        "working_dir": str(tmp_path),
        "usd_path": str(prepared_usd),
        "texture_config": {
            "backend": "service",
            "endpoint": "http://fake-backend",
            "size": 4,
            "skip_existing": False,
            "workers": 1,
        },
    }

    with pytest.raises(RuntimeError, match=r"1/1 texture generation requests failed"):
        GenerateTexturesTask().run(context)

    codes = [item["code"] for item in context["generate_textures_diagnostics"]]
    assert "BACKEND_LOW_COVERAGE" in codes
    assert "BACKEND_PARTIAL_FAILURE" in codes


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_service_backend_failure_threshold_raises_for_partial_failure(
    mock_client_cls, tmp_path
):
    albedo = _write_rgb(tmp_path / "backend" / "albedo.png", (1, 2, 3))
    normal = _write_rgb(tmp_path / "backend" / "normal.png", (128, 128, 255))
    orm = _write_rgb(tmp_path / "backend" / "orm.png", (255, 20, 10))
    prepared_usd = tmp_path / "prepared_input.usd"
    prepared_usd.write_text("#usda 1.0\n", encoding="utf-8")

    def side_effect(*args, **kwargs):
        key = kwargs["config"].variant_name
        if key == "Good":
            return _projection_status_from_maps(
                key,
                maps={
                    "albedo": f"file://{albedo}",
                    "normal": f"file://{normal}",
                    "orm": f"file://{orm}",
                },
            )
        return JobStatus(job_id="bad", status="failed", error_message="HTTP 500")

    mock_client_cls.return_value.generate.side_effect = side_effect
    context = {
        "prim_texture_units": [_service_unit("Good"), _service_unit("Bad")],
        "working_dir": str(tmp_path),
        "usd_path": str(prepared_usd),
        "texture_config": {
            "backend": "service",
            "endpoint": "http://fake-backend",
            "size": 4,
            "skip_existing": False,
            "workers": 1,
            "failure_threshold": 0.5,
        },
    }

    with pytest.raises(RuntimeError, match=r"1/2 texture generation requests failed"):
        GenerateTexturesTask().run(context)

    assert context["generate_textures_failed_count"] == 1
    assert any(
        item["code"] == "BACKEND_PARTIAL_FAILURE"
        for item in context["generate_textures_diagnostics"]
    )


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_service_backend_rejects_blank_required_albedo(mock_client_cls, tmp_path):
    from PIL import Image

    blank_albedo = tmp_path / "backend" / "blank_albedo.png"
    blank_albedo.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), (0, 0, 0)).save(blank_albedo)
    prepared_usd = tmp_path / "prepared_input.usd"
    prepared_usd.write_text("#usda 1.0\n", encoding="utf-8")

    mock_client_cls.return_value.generate.return_value = _projection_status_from_maps(
        "Aluminum_Matte",
        maps={"albedo": f"file://{blank_albedo}"},
    )
    context = {
        "prim_texture_units": [_service_unit()],
        "working_dir": str(tmp_path),
        "usd_path": str(prepared_usd),
        "texture_config": {
            "backend": "service",
            "endpoint": "http://fake-backend",
            "size": 4,
            "skip_existing": False,
            "workers": 1,
        },
    }

    with pytest.raises(RuntimeError, match=r"1/1 texture generation requests failed"):
        GenerateTexturesTask().run(context)

    assert any(
        item["code"] == "BACKEND_TEXTURE_BLANK"
        for item in context["generate_textures_diagnostics"]
    )


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_service_backend_rejects_unreadable_png_map(mock_client_cls, tmp_path):
    bad_albedo = tmp_path / "backend" / "bad_albedo.png"
    bad_albedo.parent.mkdir(parents=True)
    bad_albedo.write_bytes(b"not a png")
    prepared_usd = tmp_path / "prepared_input.usd"
    prepared_usd.write_text("#usda 1.0\n", encoding="utf-8")

    mock_client_cls.return_value.generate.return_value = _projection_status_from_maps(
        "Aluminum_Matte",
        maps={"albedo": f"file://{bad_albedo}"},
    )
    context = {
        "prim_texture_units": [_service_unit()],
        "working_dir": str(tmp_path),
        "usd_path": str(prepared_usd),
        "texture_config": {
            "backend": "service",
            "endpoint": "http://fake-backend",
            "size": 4,
            "skip_existing": False,
            "workers": 1,
        },
    }

    with pytest.raises(RuntimeError, match=r"1/1 texture generation requests failed"):
        GenerateTexturesTask().run(context)

    diagnostic = context["generate_textures_diagnostics"][0]
    assert diagnostic["code"] == "BACKEND_TEXTURE_BLANK"
    assert diagnostic["details"]["reason"] == "UnidentifiedImageError"


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_service_backend_rejects_missing_local_reference(mock_client_cls, tmp_path):
    context = {
        "prim_texture_units": [_service_unit()],
        "working_dir": str(tmp_path),
        "usd_path": str(tmp_path / "prepared_input.usd"),
        "texture_config": {
            "backend": "service",
            "endpoint": "http://fake-backend",
            "skip_existing": False,
            "workers": 1,
            "reference_image_uris": [str(tmp_path / "missing.png")],
        },
    }

    with pytest.raises(FileNotFoundError, match="reference_image_uris"):
        GenerateTexturesTask().run(context)

    mock_client_cls.assert_not_called()


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_service_backend_rejects_unsupported_reference_scheme_early(
    mock_client_cls, tmp_path
):
    context = {
        "prim_texture_units": [_service_unit()],
        "working_dir": str(tmp_path),
        "usd_path": str(tmp_path / "prepared_input.usd"),
        "texture_config": {
            "backend": "service",
            "endpoint": "http://fake-backend",
            "skip_existing": False,
            "workers": 1,
            "reference_image_uris": ["ftp://example.test/reference.png"],
        },
    }

    with pytest.raises(ValueError, match="Unsupported URI scheme 'ftp'"):
        GenerateTexturesTask().run(context)

    mock_client_cls.assert_not_called()


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_simple_service_route_rejects_reference_conditioning_before_launch(
    mock_client_cls: MagicMock,
    tmp_path: Path,
) -> None:
    reference = _write_rgb(tmp_path / "reference.png", (10, 20, 30))
    source = tmp_path / "prepared_input.usd"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    context = {
        "prim_texture_units": [_service_unit()],
        "working_dir": str(tmp_path),
        "usd_path": str(source),
        "material_textures": {
            "Aluminum_Matte": {
                "reference_image_uris": [f"file://{reference}"],
            }
        },
        "texture_config": {
            "backend": "service",
            "endpoint": "http://simple-backend",
            "engine": "simple_image_gen",
            "skip_existing": False,
            "workers": 1,
            "failure_threshold": 0.0,
        },
    }

    with pytest.raises(RuntimeError, match="text-only backend"):
        GenerateTexturesTask().run(context)

    mock_client_cls.assert_not_called()
    diagnostic = context["generate_textures_diagnostics"][0]
    assert diagnostic["code"] == "BACKEND_CONDITIONING_UNSUPPORTED"
    assert diagnostic["severity"] == "error"
    assert diagnostic["details"]["unsupported_fields"] == ["reference_image_uris"]
    record = context["projection_backend_results"]["Aluminum_Matte"]
    assert record["skipped_before_backend_launch"] is True
    assert record["capabilities"]["image_conditioning"] is False


@pytest.mark.parametrize(
    "image_gen_config",
    [
        pytest.param(None, id="default-nim"),
        pytest.param({"backend": "nim"}, id="explicit-nim"),
    ],
)
def test_direct_simple_nim_rejects_missing_per_material_reference_before_launch(
    image_gen_config: dict[str, str] | None,
    context_factory: Callable[[list[PrimTextureUnit]], dict[str, Any]],
    tmp_path: Path,
) -> None:
    unit = _unit("Steel_Carbon")
    context = context_factory([unit])
    if image_gen_config is None:
        context["texture_config"].pop("image_gen")
    else:
        context["texture_config"]["image_gen"] = image_gen_config
    missing_reference = tmp_path / "missing-per-material-reference.png"
    assert not missing_reference.exists()
    context["material_textures"] = {
        unit.material_info.name: {
            "reference_image_uris": [missing_reference.as_uri()],
        }
    }

    with (
        patch.object(GenerateTexturesTask, "_run_simple_image_gen") as mock_run,
        patch(
            "texture_agent.tasks.generate_textures.ImageGenEngine"
        ) as mock_engine_cls,
        pytest.raises(RuntimeError, match="text-only backend"),
    ):
        GenerateTexturesTask().run(context)

    mock_run.assert_not_called()
    mock_engine_cls.assert_not_called()
    diagnostic = context["generate_textures_diagnostics"][0]
    assert diagnostic["code"] == "BACKEND_CONDITIONING_UNSUPPORTED"
    assert diagnostic["details"] == {
        "unsupported_fields": ["reference_image_uris"],
        "backend": "simple_image_gen",
    }
    record = context["projection_backend_results"][unit.key]
    assert record["skipped_before_backend_launch"] is True
    assert context["generate_textures_errors"][0]["type"] == (
        "UnsupportedBackendConditioning"
    )


def test_direct_non_nim_provider_is_not_rejected_by_text_only_preflight(
    tmp_path: Path,
) -> None:
    unit = _unit("Steel_Carbon")
    missing_reference = tmp_path / "missing-openai-reference.png"
    context = {
        "material_textures": {
            unit.material_info.name: {
                "reference_image_uris": [missing_reference.as_uri()],
            }
        }
    }
    texture_config = {
        "backend": "simple_image_gen",
        "image_gen": {"backend": "openai"},
    }

    supported, errors, metadata, diagnostics = (
        generate_textures_module._preflight_simple_image_gen_conditioning(
            [unit],
            context,
            texture_config,
        )
    )

    assert supported == [unit]
    assert errors == []
    assert metadata == {}
    assert diagnostics == []


@pytest.mark.parametrize(
    ("reference_uri", "expected_scheme"),
    [
        ("http://example.test/reference.png", "http"),
        ("https://example.test/reference.png", "https"),
        ("s3://example-bucket/reference.png", "s3"),
        ("omni://example.test/reference.png", "omni"),
        ("omniverse://example.test/reference.png", "omniverse"),
    ],
)
@pytest.mark.parametrize("explicit_backend", [True, False])
def test_direct_non_nim_rejects_remote_reference_before_launch(
    reference_uri: str,
    expected_scheme: str,
    explicit_backend: bool,
    context_factory: Callable[[list[PrimTextureUnit]], dict[str, Any]],
) -> None:
    unit = _unit("Steel_Carbon")
    context = context_factory([unit])
    context["texture_config"]["image_gen"] = {"backend": "openai"}
    if not explicit_backend:
        context["texture_config"].pop("backend")
    context["material_textures"] = {
        unit.material_info.name: {
            "reference_image_uris": [reference_uri],
        }
    }

    with (
        patch.object(GenerateTexturesTask, "_run_simple_image_gen") as mock_run,
        patch(
            "texture_agent.tasks.generate_textures.ImageGenEngine"
        ) as mock_engine_cls,
        pytest.raises(RuntimeError, match="local paths or file URIs"),
    ):
        GenerateTexturesTask().run(context)

    mock_run.assert_not_called()
    mock_engine_cls.assert_not_called()
    diagnostic = context["generate_textures_diagnostics"][0]
    assert diagnostic["code"] == "BACKEND_CONDITIONING_UNSUPPORTED"
    assert diagnostic["details"] == {
        "unsupported_fields": ["reference_image_uris"],
        "backend": "simple_image_gen",
        "unsupported_reference_uri_schemes": [expected_scheme],
    }
    record = context["projection_backend_results"][unit.key]
    assert record["skipped_before_backend_launch"] is True
    assert context["generate_textures_errors"][0]["type"] == (
        "UnsupportedBackendConditioning"
    )


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_direct_non_nim_simple_image_gen_passes_per_material_reference_conditioning(
    mock_engine_cls: MagicMock,
    mock_client_cls: MagicMock,
    context_factory: Callable[[list[PrimTextureUnit]], dict[str, Any]],
    tmp_path: Path,
) -> None:
    unit = _unit("Steel_Carbon")
    reference = _write_rgb(tmp_path / "references" / "steel.png", (10, 20, 30))
    albedo = _make_real_texture_set(tmp_path / "backend", unit.key)
    mock_client_cls.return_value.generate.return_value = _ok_status(albedo, unit.key)
    context = context_factory([unit])
    context["texture_config"]["workers"] = 1
    context["texture_config"]["image_gen"] = {"backend": "openai"}
    context["material_textures"] = {
        unit.material_info.name: {
            "reference_image_uris": [Path(reference).as_uri()],
        }
    }

    result = GenerateTexturesTask().run(context)

    assert set(result["generated_textures"]) == {unit.key}
    mock_engine_cls.assert_called_once()
    mock_client_cls.assert_called_once()
    call = mock_client_cls.return_value.generate.call_args.kwargs
    assert call["conditioning"].text_prompt == unit.prompt
    assert call["conditioning"].reference_image_uris == [Path(reference).as_uri()]
    assert call["conditioning"].turntable_video_uri is None
    assert call["conditioning"].multiview_image_uris == []


@pytest.mark.parametrize(
    ("conditioning_field", "unsupported_field"),
    [
        pytest.param(
            "turntable_video_uri",
            "turntable_video_uri",
            id="turntable",
        ),
        pytest.param(
            "multiview_image_uris",
            "multiview_image_uris",
            id="multiview",
        ),
    ],
)
@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_direct_non_nim_simple_image_gen_rejects_multiview_before_launch(
    mock_engine_cls: MagicMock,
    mock_client_cls: MagicMock,
    conditioning_field: str,
    unsupported_field: str,
    context_factory: Callable[[list[PrimTextureUnit]], dict[str, Any]],
    tmp_path: Path,
) -> None:
    unit = _unit("Steel_Carbon")
    if conditioning_field == "turntable_video_uri":
        media = tmp_path / "references" / "turntable.mp4"
        media.parent.mkdir(parents=True)
        media.write_bytes(b"fake-video")
        conditioning_value: str | list[str] = media.as_uri()
    else:
        media_path = _write_rgb(
            tmp_path / "references" / "view.png",
            (10, 20, 30),
        )
        conditioning_value = [Path(media_path).as_uri()]

    context = context_factory([unit])
    context["texture_config"]["image_gen"] = {"backend": "openai"}
    context["material_textures"] = {
        unit.material_info.name: {conditioning_field: conditioning_value}
    }

    with pytest.raises(RuntimeError, match="turntable or multiview conditioning"):
        GenerateTexturesTask().run(context)

    mock_engine_cls.assert_not_called()
    mock_client_cls.assert_not_called()
    diagnostic = context["generate_textures_diagnostics"][0]
    assert diagnostic["code"] == "BACKEND_CONDITIONING_UNSUPPORTED"
    assert diagnostic["details"] == {
        "unsupported_fields": [unsupported_field],
        "backend": "simple_image_gen",
    }


@pytest.mark.parametrize(
    "engine",
    [
        "simple_image_gen",
        "simple-image-gen",
        "simple",
        "image_gen",
        "image-gen",
    ],
)
def test_simple_service_engine_aliases_select_text_only_preflight(engine: str) -> None:
    assert generate_textures_module._simple_image_gen_conditioning_capabilities(
        {"backend": "service", "engine": engine}
    ) == {"image_conditioning": False, "multiview": False}


@patch("texture_agent.functions.rest_client.RestTextureVariationClient")
def test_simple_service_route_dispatches_text_only_per_material(
    mock_client_cls: MagicMock,
    tmp_path: Path,
) -> None:
    albedo = _write_rgb(tmp_path / "backend" / "albedo.png", (10, 20, 30))
    source = tmp_path / "prepared_input.usd"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    mock_client_cls.return_value.generate.return_value = _projection_status_from_maps(
        "Aluminum_Matte",
        maps={"albedo": f"file://{albedo}"},
        metadata={
            "backend_name": "texture_gen_simple_service:nim",
            "capabilities": {
                "image_conditioning": False,
                "normal_map": True,
                "orm": True,
            },
        },
    )
    context = {
        "prim_texture_units": [_service_unit()],
        "working_dir": str(tmp_path),
        "usd_path": str(source),
        "texture_config": {
            "backend": "service",
            "endpoint": "http://simple-backend",
            "engine": "simple_image_gen",
            "size": 4,
            "skip_existing": False,
            "workers": 1,
        },
    }

    result = GenerateTexturesTask().run(context)

    assert set(result["generated_textures"]) == {"Aluminum_Matte"}
    call = mock_client_cls.return_value.generate.call_args.kwargs
    assert call["conditioning"].reference_image_uris == []
    assert call["target"].mode == "per_material"
    assert call["config"].engine == "simple_image_gen"


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_passes_endpoint_api_key(
    mock_engine_cls, mock_client_cls, context_factory, tmp_path
):
    """Local OpenAI-compatible image endpoints need an explicit placeholder key."""
    mock_client = mock_client_cls.return_value
    albedo = _make_real_texture_set(tmp_path, "Steel_Carbon")
    mock_client.generate.return_value = _ok_status(albedo, "Steel_Carbon")

    units = [_unit("Steel_Carbon")]
    context = context_factory(units)
    context["texture_config"]["workers"] = 1
    context["texture_config"]["image_gen"] = {
        "backend": "openai",
        "model": "black-forest-labs/flux.2-klein-4b",
        "base_url": "http://localhost:8005/v1",
        "api_key": "not-used",
    }

    GenerateTexturesTask().run(context)

    mock_engine_cls.assert_called_once_with(
        backend="openai",
        model="black-forest-labs/flux.2-klein-4b",
        base_url="http://localhost:8005/v1",
        api_key="not-used",
        api_key_env=None,
    )


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_passes_endpoint_api_key_env(
    mock_engine_cls, mock_client_cls, context_factory, tmp_path
):
    mock_client = mock_client_cls.return_value
    albedo = _make_real_texture_set(tmp_path, "Steel_Carbon")
    mock_client.generate.return_value = _ok_status(albedo, "Steel_Carbon")

    units = [_unit("Steel_Carbon")]
    context = context_factory(units)
    context["texture_config"]["workers"] = 1
    context["texture_config"]["image_gen"] = {
        "backend": "openai",
        "model": "black-forest-labs/flux.2-klein-4b",
        "base_url": "https://api.openai-compatible.example/v1",
        "api_key_env": "IMAGE_GEN_API_KEY",
    }

    GenerateTexturesTask().run(context)

    mock_engine_cls.assert_called_once_with(
        backend="openai",
        model="black-forest-labs/flux.2-klein-4b",
        base_url="https://api.openai-compatible.example/v1",
        api_key=None,
        api_key_env="IMAGE_GEN_API_KEY",
    )


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_raises_when_every_unit_fails(
    mock_engine_cls, mock_client_cls, context_factory
):
    """All units failing must raise so the pipeline doesn't silently exit 0."""
    mock_client = mock_client_cls.return_value
    mock_client.generate.return_value = _fail_status("any", "HTTP 403 Forbidden")

    units = [_unit("Steel_Carbon"), _unit("Copper_Polished")]
    task = GenerateTexturesTask()

    with pytest.raises(RuntimeError, match=r"2/2 texture generation requests failed"):
        task.run(context_factory(units))


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_continues_on_partial_failure(
    mock_engine_cls, mock_client_cls, context_factory, tmp_path, caplog
):
    """One success + one failure is allowed: returns partial result + warns."""
    import logging

    mock_client = mock_client_cls.return_value

    def side_effect(*args, **kwargs):
        cfg = kwargs["config"]
        if cfg.variant_name == "Steel_Carbon":
            albedo = _make_real_texture_set(tmp_path, "Steel_Carbon")
            return _ok_status(albedo, "Steel_Carbon")
        return _fail_status(cfg.variant_name, "HTTP 403 Forbidden")

    mock_client.generate.side_effect = side_effect

    units = [_unit("Steel_Carbon"), _unit("Copper_Polished")]
    task = GenerateTexturesTask()

    with caplog.at_level(logging.WARNING):
        result = task.run(context_factory(units))

    generated = result["generated_textures"]
    assert "Steel_Carbon" in generated
    assert "Copper_Polished" not in generated
    assert any("1/2 failures" in rec.message for rec in caplog.records)


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_succeeds_when_all_units_succeed(
    mock_engine_cls, mock_client_cls, context_factory, tmp_path
):
    """All-success path is unchanged."""
    mock_client = mock_client_cls.return_value

    def side_effect(*args, **kwargs):
        cfg = kwargs["config"]
        albedo = _make_real_texture_set(tmp_path, cfg.variant_name)
        return _ok_status(albedo, cfg.variant_name)

    mock_client.generate.side_effect = side_effect

    units = [_unit("Steel_Carbon"), _unit("Copper_Polished")]
    task = GenerateTexturesTask()

    result = task.run(context_factory(units))
    generated = result["generated_textures"]
    assert set(generated.keys()) == {"Steel_Carbon", "Copper_Polished"}


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_raises_when_completed_but_albedo_empty(
    mock_engine_cls, mock_client_cls, context_factory
):
    """Schema-drift guard: status='completed' with empty albedo must raise.

    Without this guard a degraded service that returns a parseable
    ``status="completed"`` but empty ``GeneratedTextures(albedo="", ...)``
    would slip past _raise_if_all_failed (because each unit is "successful")
    and silently produce no output.
    """
    mock_client = mock_client_cls.return_value

    def side_effect(*args, **kwargs):
        cfg = kwargs["config"]
        return _ok_status("", cfg.variant_name)  # empty albedo path

    mock_client.generate.side_effect = side_effect

    units = [_unit("Steel_Carbon"), _unit("Copper_Polished")]
    task = GenerateTexturesTask()

    with pytest.raises(RuntimeError, match=r"2/2 texture generation requests failed"):
        task.run(context_factory(units))


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_raises_when_completed_but_albedo_missing_on_disk(
    mock_engine_cls, mock_client_cls, context_factory, tmp_path
):
    """Schema-drift guard: albedo path set but file does not exist."""
    mock_client = mock_client_cls.return_value

    def side_effect(*args, **kwargs):
        cfg = kwargs["config"]
        # Path looks plausible but no file on disk (mimics failed localization).
        missing_path = str(tmp_path / f"missing_{cfg.variant_name}.png")
        return _ok_status(missing_path, cfg.variant_name)

    mock_client.generate.side_effect = side_effect

    units = [_unit("Steel_Carbon"), _unit("Copper_Polished")]
    task = GenerateTexturesTask()

    with pytest.raises(RuntimeError, match=r"2/2 texture generation requests failed"):
        task.run(context_factory(units))


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_raises_when_completed_map_size_mismatches_config(
    mock_engine_cls, mock_client_cls, context_factory, tmp_path
):
    mock_client = mock_client_cls.return_value
    key = "Steel_Carbon"
    albedo = _make_real_texture_set(tmp_path, key)
    mock_client.generate.return_value = _ok_status(albedo, key)
    context = context_factory([_unit(key)])
    context["texture_config"]["size"] = 8

    with pytest.raises(RuntimeError, match=r"1/1 texture generation requests failed"):
        GenerateTexturesTask().run(context)


def test_no_units_is_noop(context_factory):
    """Empty unit list short-circuits without invoking the backend."""
    task = GenerateTexturesTask()
    result = task.run(context_factory([]))
    assert result["generated_textures"] == {}


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_simple_image_gen_rejects_unsupported_scheme_albedo(
    mock_engine_cls, mock_client_cls, context_factory
):
    """Remote URIs (s3://, http://, omni://) are rejected as per-unit failures.

    Although the texture-variation API contract permits storage-agnostic
    URIs, this pipeline's BlendTexturesTask currently only opens local
    file paths -- so trusting an s3:// albedo would resurrect the
    silent-success bug (blend skips with a warning, apply sees nothing,
    CLI prints "Pipeline complete!" with no output). Until downstream
    learns to fetch remote schemes, validate them as failures here.
    """
    mock_client = mock_client_cls.return_value

    def side_effect(*args, **kwargs):
        cfg = kwargs["config"]
        return _ok_status(f"s3://bucket/{cfg.variant_name}.png", cfg.variant_name)

    mock_client.generate.side_effect = side_effect

    units = [_unit("Steel_Carbon"), _unit("Copper_Polished")]
    task = GenerateTexturesTask()

    with pytest.raises(RuntimeError, match=r"2/2 texture generation requests failed"):
        task.run(context_factory(units))


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_cache_does_not_mask_total_fresh_failure(
    mock_engine_cls, mock_client_cls, tmp_path
):
    """Cached entries must not rescue the all-failed signal.

    If every fresh request fails (e.g. expired NIM key returning HTTP
    403), the customer's environment is broken regardless of what
    cache had from a prior run. Surfacing the failure is more important
    than reporting "Pipeline complete!" with stale cache as the only
    output.
    """
    # Pre-seed cached textures for Steel_Carbon so skip_existing keeps it.
    out_dir = tmp_path / "generated"
    _make_real_texture_set(out_dir, "Steel_Carbon")

    mock_client = mock_client_cls.return_value
    mock_client.generate.return_value = _fail_status("any", "HTTP 403 Forbidden")

    context = {
        "prim_texture_units": [_unit("Steel_Carbon"), _unit("Copper_Polished")],
        "working_dir": str(tmp_path),
        "usd_path": "/tmp/asset.usd",
        "texture_config": {
            "backend": "simple_image_gen",
            "image_gen": {"backend": "nim"},
            "skip_existing": True,
            "workers": 2,
        },
    }
    task = GenerateTexturesTask()

    with pytest.raises(RuntimeError, match=r"1/1 texture generation requests failed"):
        task.run(context)


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_skip_existing_with_cache_and_fresh_success_does_not_raise(
    mock_engine_cls, mock_client_cls, tmp_path, caplog
):
    """Resumed run with cache + at least one fresh success completes cleanly."""
    import logging

    # Pre-seed cached textures for Steel_Carbon.
    out_dir = tmp_path / "generated"
    _make_real_texture_set(out_dir, "Steel_Carbon")

    mock_client = mock_client_cls.return_value

    def side_effect(*args, **kwargs):
        cfg = kwargs["config"]
        albedo = _make_real_texture_set(tmp_path, cfg.variant_name)
        return _ok_status(albedo, cfg.variant_name)

    mock_client.generate.side_effect = side_effect

    context = {
        "prim_texture_units": [_unit("Steel_Carbon"), _unit("Copper_Polished")],
        "working_dir": str(tmp_path),
        "usd_path": "/tmp/asset.usd",
        "texture_config": {
            "backend": "simple_image_gen",
            "image_gen": {"backend": "nim"},
            "skip_existing": True,
            "workers": 2,
        },
    }
    task = GenerateTexturesTask()

    with caplog.at_level(logging.WARNING):
        result = task.run(context)

    generated = result["generated_textures"]
    assert set(generated.keys()) == {"Steel_Carbon", "Copper_Polished"}
    # No failure warnings -- everything succeeded.
    assert not any("failures" in rec.message for rec in caplog.records)


@patch("texture_agent.tasks.generate_textures.TextureVariationClient")
@patch("texture_agent.tasks.generate_textures.ImageGenEngine")
def test_resume_reuses_cache_even_when_skip_existing_is_disabled(
    mock_engine_cls, mock_client_cls, tmp_path
):
    """--resume must avoid regenerating complete cached texture sets."""
    out_dir = tmp_path / "generated"
    _make_real_texture_set(out_dir, "Steel_Carbon")

    context = {
        "prim_texture_units": [_unit("Steel_Carbon")],
        "working_dir": str(tmp_path),
        "usd_path": "/tmp/asset.usd",
        "resume": True,
        "texture_config": {
            "backend": "simple_image_gen",
            "image_gen": {"backend": "nim"},
            "skip_existing": False,
            "workers": 2,
        },
    }

    result = GenerateTexturesTask().run(context)

    assert set(result["generated_textures"]) == {"Steel_Carbon"}
    mock_client_cls.return_value.generate.assert_not_called()


def test_cached_texture_set_skips_invalid_candidate(tmp_path, monkeypatch, caplog):
    """Cache lookup validates candidates before returning them.

    This simulates a stale flat-layout cache entry disappearing between the
    initial existence check and validation. The helper should skip it and fall
    through to the nested-layout candidate instead.
    """
    import logging

    out_dir = tmp_path / "generated"
    nested_dir = out_dir / "Steel_Carbon"
    nested_dir.mkdir(parents=True)

    flat_albedo = out_dir / "Steel_Carbon_albedo.png"
    flat_normal = out_dir / "Steel_Carbon_normal.png"
    flat_orm = out_dir / "Steel_Carbon_orm.png"
    nested_albedo = nested_dir / "Steel_Carbon_albedo.png"
    nested_normal = nested_dir / "Steel_Carbon_normal.png"
    nested_orm = nested_dir / "Steel_Carbon_orm.png"
    for path in (flat_albedo, nested_albedo):
        _write_rgb(path, (10, 20, 30))
    for path in (flat_normal, nested_normal):
        _write_rgb(path, (128, 128, 255))
    for path in (flat_orm, nested_orm):
        _write_rgb(path, (255, 200, 0))

    original_exists = Path.exists
    flat_exists_calls = 0

    # fake_exists lets flat_albedo pass _cached_texture_set's initial check,
    # then fail the validation-time exists() call so fallback can be verified.
    def fake_exists(path: Path) -> bool:
        nonlocal flat_exists_calls
        if path == flat_albedo:
            flat_exists_calls += 1
            return flat_exists_calls == 1
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)

    with caplog.at_level(logging.WARNING):
        textures = _cached_texture_set(out_dir, "Steel_Carbon")

    assert textures is not None
    assert textures.albedo == str(nested_albedo)
    assert textures.normal == str(nested_normal)
    assert textures.orm == str(nested_orm)
    assert any(
        "Skipping invalid cached textures" in rec.message for rec in caplog.records
    )


def test_cached_texture_set_rejects_partial_sets(tmp_path, caplog):
    """Albedo-only cache entries are rejected as incomplete PBR sets."""
    import logging

    out_dir = tmp_path / "generated"
    out_dir.mkdir(parents=True)
    _write_rgb(out_dir / "Steel_Carbon_albedo.png", (10, 20, 30))

    with caplog.at_level(logging.WARNING):
        textures = _cached_texture_set(out_dir, "Steel_Carbon")

    assert textures is None
    assert any(
        "Skipping invalid cached textures" in rec.message for rec in caplog.records
    )
