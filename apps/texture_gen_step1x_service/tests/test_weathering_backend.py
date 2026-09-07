# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
from pathlib import Path

import apps.texture_gen_step1x_service.backend as step1x_backend_module
import numpy as np
import pytest
from apps.texture_gen_service_common import (
    Conditioning,
    CreateJobRequest,
    TextureGenerationBackendError,
    TextureTarget,
)
from apps.texture_gen_step1x_service.backend import (
    Step1XBackend,
    Step1XBackendConfig,
    Step1XRunRequest,
    Step1XRunResult,
    Step1XScopeInfo,
    _inspect_step1x_scope,
    _material_scalar,
    _measure_weathering_uv_seam_continuity,
    _merge_auxiliary_artifacts,
    _sample_texture_edge,
)
from PIL import Image


def _write_source_stage(tmp_path: Path) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    albedo = tmp_path / "source_albedo.png"
    Image.new("RGB", (64, 64), (72, 72, 72)).save(albedo)
    stage_path = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-1, -1, 0),
            Gf.Vec3f(1, -1, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(-1, 1, 0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    st.Set(
        [
            Gf.Vec2f(0, 0),
            Gf.Vec2f(1, 0),
            Gf.Vec2f(1, 1),
            Gf.Vec2f(0, 1),
        ]
    )
    material = UsdShade.Material.Define(stage, "/World/Looks/Steel")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(albedo)))
    material.GetPrim().CreateAttribute(
        "inputs:roughness_constant",
        Sdf.ValueTypeNames.Float,
    ).Set(0.25)
    material.GetPrim().CreateAttribute(
        "inputs:metalness_constant",
        Sdf.ValueTypeNames.Float,
    ).Set(1.0)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.Save()
    return stage_path


class _WeatheringRunner:
    def run(
        self,
        request: Step1XRunRequest,
        *,
        cancel_event: threading.Event,
    ) -> Step1XRunResult:
        assert not cancel_event.is_set()
        assert request.custom_parameters["skip_material_anything"] is False
        size = 64
        variation = np.random.default_rng(1045).integers(
            0,
            180,
            size=(size, size),
            dtype=np.uint8,
        )
        albedo = np.stack(
            [np.clip(90 + variation, 0, 255), 45 + variation // 4, 25 + variation // 8],
            axis=2,
        )
        orm = np.empty_like(albedo)
        orm[:, :, 0] = 255
        orm[:, :, 1] = 225
        orm[:, :, 2] = 255
        albedo_path = request.output_dir / "candidate.png"
        orm_path = request.output_dir / "candidate_orm.png"
        request.output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(albedo, mode="RGB").save(albedo_path)
        Image.fromarray(orm, mode="RGB").save(orm_path)
        return Step1XRunResult(
            albedo_uri=albedo_path.as_uri(),
            orm_uri=orm_path.as_uri(),
            width=size,
            height=size,
            auxiliary_artifacts={
                "masks": {"runner": {"uri": "runner://conditioning-mask"}}
            },
        )


def test_weathering_artifact_merge_adds_new_top_level_entries() -> None:
    assert _merge_auxiliary_artifacts(
        None,
        {"masks": {"weathering": {"uri": "file:///mask.png"}}},
    ) == {"masks": {"weathering": {"uri": "file:///mask.png"}}}


def test_backend_publishes_passing_weathering_receipt_and_mask(tmp_path: Path) -> None:
    source = _write_source_stage(tmp_path)
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            command_template="provider {source_asset} {output_dir}",
            validate_assets=True,
        ),
        runner=_WeatheringRunner(),
    )
    request = CreateJobRequest(
        source_asset_uri=source.as_uri(),
        target={
            "material_path": "/World/Looks/Steel",
            "prim_paths": ["/World/Mesh"],
            "mode": "per_material",
        },
        conditioning=Conditioning(text_prompt="localized orange rust around joints"),
        configuration={
            "texture_size": 64,
            "strength": 0.8,
        },
    )

    result = backend.generate(
        request,
        job_id="rust-1",
        output_dir=tmp_path / "out",
        cancel_event=threading.Event(),
    )

    evidence = result.metadata["weathering"]
    assert evidence["status"] == "pass", evidence["failures"]
    assert evidence["metrics"]["uv_seam_sample_count"] == 0
    assert evidence["artifacts"]["final_albedo_sha256"]
    assert evidence["artifacts"]["source_usd_sha256"]
    assert evidence["artifacts"]["request_sha256"]
    assert result.auxiliary_artifacts["masks"]["weathering"]["sha256"]
    assert result.auxiliary_artifacts["masks"]["runner"] == {
        "uri": "runner://conditioning-mask"
    }
    assert result.metadata["capabilities"]["weathering"] is True
    assert result.maps["orm"].packing == "occlusion_roughness_metallic"


def test_backend_rejects_weathering_material_anything_bypass(tmp_path: Path) -> None:
    source = _write_source_stage(tmp_path)
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            command_template="provider {source_asset} {output_dir}",
            validate_assets=True,
        ),
        runner=_WeatheringRunner(),
    )
    request = CreateJobRequest(
        source_asset_uri=source.as_uri(),
        target={
            "material_path": "/World/Looks/Steel",
            "prim_paths": ["/World/Mesh"],
            "mode": "per_material",
        },
        conditioning=Conditioning(text_prompt="localized orange rust around joints"),
        configuration={
            "texture_size": 64,
            "custom_parameters": {"skip_material_anything": True},
        },
    )

    with pytest.raises(
        TextureGenerationBackendError,
        match="STEP1X_WEATHERING_REQUIRES_MATERIAL_ANYTHING",
    ) as exc_info:
        backend.generate(
            request,
            job_id="rust-skip-ma",
            output_dir=tmp_path / "out",
            cancel_event=threading.Event(),
        )

    assert exc_info.value.result is not None
    assert exc_info.value.result.diagnostics[0]["code"] == (
        "STEP1X_WEATHERING_REQUIRES_MATERIAL_ANYTHING"
    )


def test_backend_rejects_weathering_when_scope_validation_is_disabled(
    tmp_path: Path,
) -> None:
    source = _write_source_stage(tmp_path)
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            command_template="provider {source_asset} {output_dir}",
            validate_assets=False,
        ),
        runner=_WeatheringRunner(),
    )
    request = CreateJobRequest(
        source_asset_uri=source.as_uri(),
        conditioning=Conditioning(text_prompt="localized rust"),
    )

    with pytest.raises(
        TextureGenerationBackendError,
        match=(
            "prompt-requested weathering cannot run when source material "
            "validation is disabled"
        ),
    ) as exc_info:
        backend.generate(
            request,
            job_id="rust-no-validation",
            output_dir=tmp_path / "out",
            cancel_event=threading.Event(),
        )

    assert exc_info.value.result is not None
    assert exc_info.value.result.diagnostics[0]["code"] == (
        "STEP1X_WEATHERING_SCOPE_REQUIRED"
    )


def test_backend_rejects_unexpected_scope_without_source_albedo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _write_source_stage(tmp_path)
    monkeypatch.setattr(
        step1x_backend_module,
        "_inspect_step1x_scope",
        lambda *args, **kwargs: Step1XScopeInfo(
            source_asset_path=source,
            material_path="/World/Looks/Steel",
            prim_paths=("/World/Mesh",),
            source_albedo_path=None,
            source_metalness=1.0,
        ),
    )
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            command_template="provider {source_asset} {output_dir}",
            validate_assets=True,
        ),
        runner=_WeatheringRunner(),
    )
    request = CreateJobRequest(
        source_asset_uri=source.as_uri(),
        conditioning=Conditioning(text_prompt="localized rust"),
    )

    with pytest.raises(
        TextureGenerationBackendError,
        match="prompt-requested weathering requires validated material scope",
    ) as exc_info:
        backend.generate(
            request,
            job_id="rust-missing-albedo",
            output_dir=tmp_path / "out",
            cancel_event=threading.Event(),
        )

    assert exc_info.value.result is not None
    assert exc_info.value.result.diagnostics[0]["code"] == (
        "STEP1X_WEATHERING_SCOPE_REQUIRED"
    )


def test_backend_fails_closed_on_weathering_uv_seam_quality(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _write_source_stage(tmp_path)
    monkeypatch.setattr(
        step1x_backend_module,
        "_measure_weathering_uv_seam_continuity",
        lambda *args, **kwargs: {
            "uv_seam_sample_count": 3,
            "uv_seam_rgb_delta_mean": 1.0,
            "uv_seam_rgb_delta_p95": 1.0,
            "uv_seam_rgb_delta_max": 1.0,
        },
    )
    backend = Step1XBackend(
        config=Step1XBackendConfig(
            command_template="provider {source_asset} {output_dir}",
            validate_assets=True,
        ),
        runner=_WeatheringRunner(),
    )
    request = CreateJobRequest(
        source_asset_uri=source.as_uri(),
        target={
            "material_path": "/World/Looks/Steel",
            "prim_paths": ["/World/Mesh"],
            "mode": "per_material",
        },
        conditioning=Conditioning(text_prompt="localized rust"),
        configuration={"texture_size": 64},
    )

    with pytest.raises(
        TextureGenerationBackendError,
        match="STEP1X_WEATHERING_QUALITY_FAILED",
    ) as exc_info:
        backend.generate(
            request,
            job_id="rust-seam-fail",
            output_dir=tmp_path / "out",
            cancel_event=threading.Event(),
        )

    assert exc_info.value.result is not None
    assert exc_info.value.result.metadata["weathering"]["status"] == "fail"
    assert exc_info.value.result.diagnostics[-1]["code"] == (
        "STEP1X_WEATHERING_QUALITY_FAILED"
    )
    assert exc_info.value.result.variant_asset_uri == ""
    assert exc_info.value.result.generated_textures.albedo is None
    assert exc_info.value.result.generated_textures.orm is None
    assert exc_info.value.result.maps == {}
    assert exc_info.value.result.auxiliary_artifacts == {}


def test_uv_seam_metric_compares_both_texture_representations(tmp_path: Path) -> None:
    from pxr import Gf, Usd, UsdGeom

    source = _write_source_stage(tmp_path)
    stage = Usd.Stage.Open(str(source))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Mesh"))
    mesh.GetFaceVertexCountsAttr().Set([3, 3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3])
    st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    st.Set(
        [
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(0.0, 1.0),
            Gf.Vec2f(0.1, 1.0),
            Gf.Vec2f(0.9, 0.0),
            Gf.Vec2f(1.0, 1.0),
            Gf.Vec2f(1.0, 0.0),
        ]
    )
    stage.Save()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:, :] = 255
    albedo = tmp_path / "seam.png"
    Image.fromarray(image, mode="RGB").save(albedo)

    metrics = _measure_weathering_uv_seam_continuity(
        Step1XScopeInfo(
            source_asset_path=source,
            material_path="/World/Looks/Steel",
            prim_paths=("/World/Mesh",),
        ),
        tmp_path / "source_albedo.png",
        albedo.as_uri(),
    )

    assert metrics["uv_seam_sample_count"] == 3
    assert metrics["uv_seam_rgb_delta_p95"] == 1.0


def test_uv_seam_metric_returns_empty_for_unresolvable_inputs(tmp_path: Path) -> None:
    source = _write_source_stage(tmp_path)
    metrics = _measure_weathering_uv_seam_continuity(
        Step1XScopeInfo(source_asset_path=source),
        tmp_path / "source_albedo.png",
        "https://example.invalid/weathered.png",
    )

    assert metrics == {
        "uv_seam_sample_count": 0,
        "uv_seam_rgb_delta_mean": None,
        "uv_seam_rgb_delta_p95": None,
        "uv_seam_rgb_delta_max": None,
    }


def test_uv_seam_metric_skips_mesh_without_uvs(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    source = _write_source_stage(tmp_path)
    stage = Usd.Stage.Open(str(source))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Mesh"))
    UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").GetAttr().Clear()
    stage.Save()
    albedo = tmp_path / "weathered.png"
    Image.new("RGB", (64, 64), (72, 72, 72)).save(albedo)

    metrics = _measure_weathering_uv_seam_continuity(
        Step1XScopeInfo(
            source_asset_path=source,
            material_path="/World/Looks/Steel",
            prim_paths=("/World/Mesh",),
        ),
        tmp_path / "source_albedo.png",
        albedo.as_uri(),
    )

    assert metrics["uv_seam_sample_count"] == 0


def test_uv_seam_metric_ignores_matching_shared_edge_uvs(tmp_path: Path) -> None:
    from pxr import Gf, Usd, UsdGeom

    source = _write_source_stage(tmp_path)
    stage = Usd.Stage.Open(str(source))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Mesh"))
    mesh.GetFaceVertexCountsAttr().Set([3, 3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3])
    UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Set(
        [
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(1.0, 1.0),
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(1.0, 1.0),
            Gf.Vec2f(0.0, 1.0),
        ]
    )
    stage.Save()
    albedo = tmp_path / "weathered.png"
    Image.new("RGB", (64, 64), (72, 72, 72)).save(albedo)

    metrics = _measure_weathering_uv_seam_continuity(
        Step1XScopeInfo(
            source_asset_path=source,
            material_path="/World/Looks/Steel",
            prim_paths=("/World/Mesh",),
        ),
        tmp_path / "source_albedo.png",
        albedo.as_uri(),
    )

    assert metrics["uv_seam_sample_count"] == 0


def test_texture_edge_sampling_wraps_out_of_range_uvs() -> None:
    image = Image.new("RGB", (4, 4), (0, 0, 0))
    image.putpixel((2, 2), (10, 20, 30))

    assert _sample_texture_edge(
        image,
        ((1.5, -0.5), (1.5, -0.5)),
        0.5,
    ) == (10, 20, 30)


def test_material_scalar_ignores_boolean_and_non_numeric_values() -> None:
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Material")
    material.GetPrim().CreateAttribute(
        "inputs:roughness",
        Sdf.ValueTypeNames.Bool,
    ).Set(True)
    material.GetPrim().CreateAttribute(
        "inputs:specular_roughness",
        Sdf.ValueTypeNames.String,
    ).Set("not-a-number")

    assert (
        _material_scalar(
            material.GetPrim(),
            ("roughness", "specular_roughness"),
            default=0.35,
        )
        == 0.35
    )


def test_checked_in_ladder_resolves_metal_and_dielectric_pbr_scalars(
    tmp_path: Path,
) -> None:
    ladder = Path(
        "apps/texture_agent/data/examples/ladder/sources/usd/ladder.usd"
    ).resolve()
    aluminum = _inspect_step1x_scope(
        ladder.as_uri(),
        TextureTarget(
            material_path="/RootNode/Looks/Aluminum_Matte",
            prim_paths=["/RootNode/Geometry/M_AluminumStepLadder_B01_Aluminum"],
        ),
        output_dir=tmp_path / "aluminum",
        texture_size=8,
    )
    rubber = _inspect_step1x_scope(
        ladder.as_uri(),
        TextureTarget(
            material_path="/RootNode/Looks/Rubber_Black_Matte",
            prim_paths=["/RootNode/Geometry/M_AluminumStepLadder_B01_Rubber"],
        ),
        output_dir=tmp_path / "rubber",
        texture_size=8,
    )

    assert aluminum.source_metalness == 1.0
    assert aluminum.source_roughness == pytest.approx(0.4)
    assert rubber.source_metalness == 0.0
    assert rubber.source_roughness == pytest.approx(0.8)
