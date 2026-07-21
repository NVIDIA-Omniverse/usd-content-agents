# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import struct
import subprocess
import sys
import threading
import tomllib
import warnings
import zlib
from dataclasses import replace
from pathlib import Path

import pytest
from apps.texture_gen_step1x_service.backend import (
    Step1XBackendConfig,
    Step1XRunRequest,
    Step1XRunResult,
)
from PIL import Image

from material_agent.material_library_generation.creation_contract import (
    CreateMaterialRequest,
    MaterialChannel,
    MaterialChannelComponent,
    MaterialChannelSource,
    MaterialColorSpace,
    MaterialConditioningArtifact,
    MaterialConditioningKind,
    MaterialCreationError,
    MaterialCreationErrorCode,
    MaterialCreationMode,
    MaterialDegradationCode,
    ORMPacking,
    PreparedMaterialConditioning,
)
from material_agent.material_library_generation.schema import (
    IntendedPart,
    MaterialRecipe,
    PBRHints,
)
from material_agent.material_library_generation.step1x_backend import (
    Step1XMaterialCreationBackend,
    Step1XMaterialCreationConfig,
    result_fingerprint,
)

SOURCE_MATERIAL_PATH = "/World/Looks/SourceMaterial"


class _RecordingRunner:
    def __init__(
        self,
        *,
        normal: bool = True,
        orm: bool = True,
        error: str | None = None,
        albedo_uri: str | None = None,
        orm_uri: str | None = None,
        orm_pixels: tuple[tuple[int, int, int], ...] | None = None,
        orm_size: tuple[int, int] | None = None,
        profile_temp_directory: bool = False,
    ) -> None:
        self.normal = normal
        self.orm = orm
        self.error = error
        self.albedo_uri = albedo_uri
        self.orm_uri = orm_uri
        self.orm_pixels = orm_pixels
        self.orm_size = orm_size
        self.profile_temp_directory = profile_temp_directory
        self.requests: list[Step1XRunRequest] = []
        self.cancel_events: list[threading.Event] = []

    def run(
        self,
        request: Step1XRunRequest,
        *,
        cancel_event: threading.Event,
    ) -> Step1XRunResult:
        self.cancel_events.append(cancel_event)
        self.requests.append(request)
        if cancel_event.is_set():
            raise RuntimeError("Step1X job was cancelled")
        if self.error is not None:
            raise RuntimeError(self.error)
        albedo = request.output_dir / "final_albedo.png"
        normal = request.output_dir / "final_normal.png"
        orm = request.output_dir / "final_orm.png"
        _write_png(albedo, (20, 80, 200))
        normal_uri = None
        orm_uri = None
        if self.normal:
            _write_png(normal, (128, 128, 255))
            normal_uri = normal.as_uri()
        if self.orm:
            if self.orm_uri is None:
                if self.orm_size is not None:
                    image = Image.new("RGB", self.orm_size, (255, 120, 10))
                    image.save(orm)
                elif self.orm_pixels is None:
                    _write_png(orm, (255, 120, 10))
                else:
                    _write_row_png(orm, self.orm_pixels)
                orm_uri = orm.as_uri()
                if self.profile_temp_directory:
                    (request.output_dir / ".orm.png.pbr-profile.tmp").mkdir()
            else:
                orm_uri = self.orm_uri
        return Step1XRunResult(
            albedo_uri=self.albedo_uri or albedo.as_uri(),
            normal_uri=normal_uri,
            orm_uri=orm_uri,
            width=request.texture_size,
            height=request.texture_size,
            metadata={"runner": "fake-step1x"},
        )


def test_step1x_material_creation_normalizes_full_runner_output(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner()
    backend = _backend(tmp_path, runner)
    request = _request(tmp_path)
    conditioning = _conditioning(tmp_path, request)

    result = backend.create(
        request,
        output_dir=tmp_path / "out",
        conditioning=conditioning,
    )

    assert len(runner.requests) == 1
    run_request = runner.requests[0]
    assert run_request.prompt == request.recipe.appearance_prompt
    assert run_request.seed == request.effective_seed
    assert run_request.texture_size == request.texture_size
    assert run_request.source_asset_path == tmp_path / "scoped.usda"
    assert run_request.source_albedo_path == tmp_path / "source_albedo.png"
    assert run_request.reference_image_uris == request.effective_reference_image_uris
    assert run_request.multiview_image_uris == ((tmp_path / "render.png").as_posix(),)
    assert run_request.target is not None
    assert run_request.target.material_path == SOURCE_MATERIAL_PATH
    assert run_request.target.prim_paths == ["/World/Asset/Body"]
    assert run_request.scope is not None
    assert run_request.scope.material_path == SOURCE_MATERIAL_PATH
    assert run_request.scope.prim_paths == ("/World/Asset/Body",)
    assert run_request.custom_parameters["skip_material_anything"] is False
    assert run_request.custom_parameters["roughness"] == 0.72
    assert run_request.custom_parameters["metallic"] == 0.0

    albedo = result.artifact(MaterialChannel.ALBEDO)
    normal = result.artifact(MaterialChannel.NORMAL)
    orm = result.artifact(MaterialChannel.ORM)
    assert albedo is not None and albedo.color_space is MaterialColorSpace.SRGB
    assert normal is not None and normal.normal_convention is not None
    assert orm is not None and orm.packing is ORMPacking.OCCLUSION_ROUGHNESS_METALLIC
    assert albedo.path.name == "albedo.png"
    assert normal.path.name == "normal.png"
    assert orm.path.name == "orm.png"
    assert _read_png(orm.path) == (1, 1, (255, round(0.72 * 255), 0))
    assert [item.code for item in result.degradations] == [
        MaterialDegradationCode.NEUTRAL_AO
    ]
    assert [item.code for item in result.diagnostics] == [
        "STEP1X_RECIPE_PBR_PROFILE_APPLIED"
    ]
    profile = result.diagnostics[0].details
    assert profile["packing"] == ORMPacking.OCCLUSION_ROUGHNESS_METALLIC.value
    assert profile["roughness"]["recipe_hint"] == pytest.approx(0.72)
    assert profile["metallic"]["class"] == "dielectric"
    assert result.provenance.backend == "step1x_material_anything"
    assert result.provenance.backend_revision.startswith(
        "step1x-material-creation-adapter.v2"
    )
    assert result.provenance.conditioning_fingerprint is not None
    assert len(result_fingerprint(result)) == 64
    assert runner.cancel_events and not runner.cancel_events[0].is_set()
    assert (
        normal.component_provenance[0].source is MaterialChannelSource.MODEL_GENERATED
    )
    orm_sources = {item.component: item.source for item in orm.component_provenance}
    assert orm_sources == {
        MaterialChannelComponent.OCCLUSION: MaterialChannelSource.NEUTRAL_FALLBACK,
        MaterialChannelComponent.ROUGHNESS: MaterialChannelSource.DERIVED,
        MaterialChannelComponent.METALLIC: MaterialChannelSource.RECIPE_HINT,
    }
    later_result = replace(
        result,
        provenance=replace(
            result.provenance,
            duration_seconds=result.provenance.duration_seconds + 10.0,
        ),
    )
    assert result_fingerprint(result) == result_fingerprint(later_result)


def test_step1x_direct_backend_revision_tracks_generation_configuration(
    tmp_path: Path,
) -> None:
    runtime = Step1XBackendConfig(
        runtime_dir=tmp_path / "runtime",
        model_dir=tmp_path / "models",
        validate_assets=False,
    )
    base_config = Step1XMaterialCreationConfig(
        step1x=runtime,
        strength=0.4,
        custom_parameters={"ma_steps": 4, "guidance": 7.5},
    )
    base_revision = Step1XMaterialCreationBackend(config=base_config).revision
    reordered_revision = Step1XMaterialCreationBackend(
        config=replace(
            base_config,
            custom_parameters={"guidance": 7.5, "ma_steps": 4},
        )
    ).revision
    changed_strength_revision = Step1XMaterialCreationBackend(
        config=replace(base_config, strength=0.5)
    ).revision
    changed_custom_revision = Step1XMaterialCreationBackend(
        config=replace(base_config, custom_parameters={"ma_steps": 8})
    ).revision
    changed_runtime_revision = Step1XMaterialCreationBackend(
        config=replace(
            base_config,
            step1x=replace(runtime, extra_args=("--debug",)),
        )
    ).revision
    relocated_io_revision = Step1XMaterialCreationBackend(
        config=replace(
            base_config,
            step1x=replace(
                runtime,
                cache_dir=tmp_path / "other-cache",
                output_dir=tmp_path / "other-output",
            ),
        )
    ).revision
    already_resolved_revision = Step1XMaterialCreationBackend(
        config=replace(base_config, backend_revision=base_revision)
    ).revision

    assert base_revision.startswith("step1x-material-creation-adapter.v2+cfg.")
    assert (
        base_revision
        == reordered_revision
        == relocated_io_revision
        == already_resolved_revision
    )
    assert base_revision != changed_strength_revision
    assert base_revision != changed_custom_revision
    assert base_revision != changed_runtime_revision


def test_step1x_recipe_profile_constrains_dielectric_and_preserves_roughness_detail(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner(
        normal=False,
        orm_pixels=(
            (11, 0, 255),
            (22, 64, 192),
            (33, 128, 128),
            (44, 192, 64),
            (55, 255, 0),
        ),
    )
    backend = _backend(tmp_path, runner)
    request = _request(
        tmp_path,
        pbr_hints=PBRHints(roughness=0.72, metallic=0.0),
    )

    result = backend.create(
        request,
        output_dir=tmp_path / "out",
        conditioning=_conditioning(tmp_path, request),
    )

    orm = result.artifact(MaterialChannel.ORM)
    assert orm is not None
    pixels = _read_pixels(orm.path)
    assert {pixel[0] for pixel in pixels} == {255}
    assert {pixel[2] for pixel in pixels} == {0}
    roughness = [pixel[1] for pixel in pixels]
    assert roughness == sorted(roughness)
    assert len(set(roughness)) == len(roughness)
    assert min(roughness) >= math.ceil((0.72 - 0.15) * 255)
    assert max(roughness) <= math.floor((0.72 + 0.15) * 255)

    profile = result.diagnostics[0].details
    assert profile["detail_policy"] == "mean_centered_compress_only"
    assert profile["roughness"]["source"] == {
        "min": 0.0,
        "max": 1.0,
        "mean": pytest.approx((0 + 64 + 128 + 192 + 255) / (5 * 255)),
    }
    assert profile["metallic"]["class"] == "dielectric"
    assert profile["metallic"]["output"] == {
        "min": 0.0,
        "max": 0.0,
        "mean": 0.0,
    }


def test_step1x_recipe_profile_enforces_metal_class_and_bounds_roughness(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner(
        orm_pixels=(
            (0, 0, 0),
            (64, 128, 64),
            (128, 255, 128),
        ),
    )
    backend = _backend(tmp_path, runner)
    request = _request(
        tmp_path,
        pbr_hints=PBRHints(roughness=0.24, metallic=1.0),
    )

    result = backend.create(
        request,
        output_dir=tmp_path / "out",
        conditioning=_conditioning(tmp_path, request),
    )

    orm = result.artifact(MaterialChannel.ORM)
    assert orm is not None
    pixels = _read_pixels(orm.path)
    assert {pixel[0] for pixel in pixels} == {255}
    assert {pixel[2] for pixel in pixels} == {255}
    roughness = [pixel[1] for pixel in pixels]
    assert len(set(roughness)) > 1
    assert min(roughness) >= math.ceil((0.24 - 0.15) * 255)
    assert max(roughness) <= math.floor((0.24 + 0.15) * 255)

    profile = result.diagnostics[0].details
    assert profile["metallic"]["class"] == "metal"
    orm_sources = {item.component: item.source for item in orm.component_provenance}
    assert orm_sources[MaterialChannelComponent.METALLIC] is (
        MaterialChannelSource.RECIPE_HINT
    )


def test_step1x_recipe_profile_preserves_bounded_mixed_metalness(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner(
        orm_pixels=(
            (0, 128, 0),
            (64, 128, 128),
            (128, 128, 255),
        ),
    )
    backend = _backend(tmp_path, runner)
    request = _request(
        tmp_path,
        pbr_hints=PBRHints(roughness=0.5, metallic=0.4),
    )

    result = backend.create(
        request,
        output_dir=tmp_path / "out",
        conditioning=_conditioning(tmp_path, request),
    )

    orm = result.artifact(MaterialChannel.ORM)
    assert orm is not None
    metallic = [pixel[2] for pixel in _read_pixels(orm.path)]
    assert len(set(metallic)) > 1
    assert min(metallic) >= math.ceil((0.4 - 0.05) * 255)
    assert max(metallic) <= math.floor((0.4 + 0.05) * 255)
    profile = result.diagnostics[0].details
    assert profile["metallic"]["class"] == "mixed"
    assert profile["metallic"]["declared_recipe_hint"] == pytest.approx(0.4)
    orm_sources = {item.component: item.source for item in orm.component_provenance}
    assert orm_sources[MaterialChannelComponent.METALLIC] is (
        MaterialChannelSource.DERIVED
    )


def test_step1x_claimed_corrupt_orm_fails_closed(tmp_path: Path) -> None:
    corrupt_orm = tmp_path / "corrupt.png"
    corrupt_orm.write_bytes(b"not a PNG")
    backend = _backend(
        tmp_path,
        _RecordingRunner(orm_uri=corrupt_orm.as_uri()),
    )
    request = _request(tmp_path)

    with pytest.raises(MaterialCreationError) as exc_info:
        backend.create(
            request,
            output_dir=tmp_path / "out",
            conditioning=_conditioning(tmp_path, request),
        )

    assert exc_info.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert exc_info.value.backend == backend.name
    assert [item.code for item in exc_info.value.diagnostics] == ["STEP1X_ORM_INVALID"]


def test_step1x_material_creation_synthesizes_orm_and_degrades_missing_normal(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path, _RecordingRunner(normal=False, orm=False))
    request = _request(tmp_path)
    conditioning = _conditioning(tmp_path, request)

    result = backend.create(
        request,
        output_dir=tmp_path / "out",
        conditioning=conditioning,
    )

    assert result.artifact(MaterialChannel.NORMAL) is None
    orm = result.artifact(MaterialChannel.ORM)
    assert orm is not None and orm.path.exists()
    assert _read_png(orm.path) == (512, 512, (255, round(0.72 * 255), 0))
    degradation_codes = {degradation.code for degradation in result.degradations}
    assert degradation_codes == {
        MaterialDegradationCode.MISSING_NORMAL,
        MaterialDegradationCode.NEUTRAL_AO,
        MaterialDegradationCode.RECIPE_HINT_FALLBACK,
    }
    orm_sources = {item.component: item.source for item in orm.component_provenance}
    assert MaterialChannelSource.NEUTRAL_FALLBACK in orm_sources.values()
    assert MaterialChannelSource.RECIPE_HINT in orm_sources.values()


def test_step1x_material_creation_requires_source_albedo_before_runner(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner()
    backend = _backend(tmp_path, runner)
    request = _request(tmp_path)

    with pytest.raises(MaterialCreationError) as exc_info:
        backend.create(
            request,
            output_dir=tmp_path / "out",
            conditioning=_conditioning(
                tmp_path,
                request,
                include_source_albedo=False,
            ),
        )

    assert exc_info.value.code is MaterialCreationErrorCode.INVALID_REQUEST
    assert exc_info.value.backend == backend.name
    assert runner.requests == []


def test_step1x_material_creation_requires_source_material_binding(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner()
    backend = _backend(tmp_path, runner)
    request = _request(tmp_path)

    with pytest.raises(MaterialCreationError) as exc_info:
        backend.create(
            request,
            output_dir=tmp_path / "out",
            conditioning=_conditioning(
                tmp_path,
                request,
                include_source_material_binding=False,
            ),
        )

    assert exc_info.value.code is MaterialCreationErrorCode.INVALID_REQUEST
    assert exc_info.value.backend == backend.name
    assert "source material binding" in str(exc_info.value)
    assert runner.requests == []


def test_step1x_material_creation_rejects_invalid_requests(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path, _RecordingRunner())
    request = _request(tmp_path)

    with pytest.raises(MaterialCreationError) as missing_conditioning:
        backend.create(request, output_dir=tmp_path / "out", conditioning=None)
    assert missing_conditioning.value.code is MaterialCreationErrorCode.INVALID_REQUEST

    tileable_request = _request(tmp_path, creation_mode=MaterialCreationMode.TILEABLE)
    with pytest.raises(MaterialCreationError) as tileable:
        backend.create(
            tileable_request,
            output_dir=tmp_path / "out",
            conditioning=_conditioning(tmp_path, tileable_request),
        )
    assert tileable.value.code is MaterialCreationErrorCode.UNSUPPORTED_MATERIAL

    with pytest.raises(MaterialCreationError) as transparent:
        transparent_request = _request(
            tmp_path,
            pbr_hints=PBRHints(opacity=0.5, transmission=0.1),
        )
        backend.create(
            transparent_request,
            output_dir=tmp_path / "out",
            conditioning=_conditioning(tmp_path, transparent_request),
        )
    assert transparent.value.code is MaterialCreationErrorCode.UNSUPPORTED_MATERIAL

    with pytest.raises(MaterialCreationError) as thin_walled:
        thin_walled_request = _request(
            tmp_path,
            pbr_hints=PBRHints(thin_walled=True),
        )
        backend.create(
            thin_walled_request,
            output_dir=tmp_path / "out",
            conditioning=_conditioning(tmp_path, thin_walled_request),
        )
    assert thin_walled.value.code is MaterialCreationErrorCode.UNSUPPORTED_MATERIAL

    with pytest.raises(ValueError, match="strength"):
        Step1XMaterialCreationConfig(strength=2.0)
    with pytest.raises(ValueError, match="model_revisions"):
        Step1XMaterialCreationConfig(model_revisions=())


def test_step1x_conditioning_mismatch_is_invalid_request(tmp_path: Path) -> None:
    backend = _backend(tmp_path, _RecordingRunner())
    request = _request(tmp_path)
    conditioning = _conditioning(tmp_path, request)
    mismatched_request = replace(
        request,
        target_prim_paths=("/World/Asset/Other",),
    )

    with pytest.raises(MaterialCreationError) as exc_info:
        backend.create(
            mismatched_request,
            output_dir=tmp_path / "out",
            conditioning=conditioning,
        )

    assert exc_info.value.code is MaterialCreationErrorCode.INVALID_REQUEST
    assert exc_info.value.backend == backend.name


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("Step1X job was cancelled", MaterialCreationErrorCode.CANCELLED),
        ("STEP1X_TIMEOUT: timed out", MaterialCreationErrorCode.TIMEOUT),
        ("CUDA_OUT_OF_MEMORY", MaterialCreationErrorCode.CUDA_OUT_OF_MEMORY),
        ("CUDA out of memory", MaterialCreationErrorCode.CUDA_OUT_OF_MEMORY),
        ("MODEL CHECKPOINT missing", MaterialCreationErrorCode.MISSING_CHECKPOINT),
        ("MODEL missing", MaterialCreationErrorCode.MISSING_CHECKPOINT),
        ("CHECKPOINT corrupted", MaterialCreationErrorCode.BACKEND_FAILURE),
        ("STEP1X_ASSET_UNREACHABLE", MaterialCreationErrorCode.BACKEND_UNAVAILABLE),
        ("STEP1X_OUTPUT_MISSING", MaterialCreationErrorCode.PARTIAL_OUTPUT),
        ("unexpected failure", MaterialCreationErrorCode.BACKEND_FAILURE),
    ],
)
def test_step1x_runner_errors_map_to_material_creation_errors(
    tmp_path: Path,
    message: str,
    code: MaterialCreationErrorCode,
) -> None:
    backend = _backend(tmp_path, _RecordingRunner(error=message))
    request = _request(tmp_path)

    with pytest.raises(MaterialCreationError) as exc_info:
        backend.create(
            request,
            output_dir=tmp_path / "out",
            conditioning=_conditioning(tmp_path, request),
        )

    assert exc_info.value.code is code
    assert exc_info.value.backend == backend.name


def test_step1x_non_local_conditioning_uri_is_invalid_request(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner()
    backend = _backend(tmp_path, runner)
    request = _request(tmp_path)
    conditioning = _conditioning(tmp_path, request)
    artifacts = tuple(
        replace(artifact, uri="https://example.invalid/source.usda")
        if artifact.kind is MaterialConditioningKind.SCOPED_USD
        else artifact
        for artifact in conditioning.artifacts
    )
    conditioning = PreparedMaterialConditioning.for_request(
        request,
        artifacts=artifacts,
    )

    with pytest.raises(MaterialCreationError) as exc_info:
        backend.create(
            request,
            output_dir=tmp_path / "out",
            conditioning=conditioning,
        )

    assert exc_info.value.code is MaterialCreationErrorCode.INVALID_REQUEST
    assert exc_info.value.backend == backend.name
    assert runner.requests == []


@pytest.mark.parametrize("albedo_uri", ["missing.png", "https://example.invalid/a.png"])
def test_step1x_invalid_albedo_output_is_partial_output(
    tmp_path: Path,
    albedo_uri: str,
) -> None:
    backend = _backend(tmp_path, _RecordingRunner(albedo_uri=albedo_uri))
    request = _request(tmp_path)

    with pytest.raises(MaterialCreationError) as exc_info:
        backend.create(
            request,
            output_dir=tmp_path / "out",
            conditioning=_conditioning(tmp_path, request),
        )

    assert exc_info.value.code is MaterialCreationErrorCode.PARTIAL_OUTPUT
    assert exc_info.value.backend == backend.name


@pytest.mark.parametrize("claimed_path_kind", ["missing", "directory"])
def test_step1x_claimed_non_file_orm_fails_closed(
    tmp_path: Path,
    claimed_path_kind: str,
) -> None:
    claimed_orm = tmp_path / "claimed_orm.png"
    if claimed_path_kind == "directory":
        claimed_orm.mkdir()
    backend = _backend(
        tmp_path,
        _RecordingRunner(orm_uri=claimed_orm.as_uri()),
    )
    request = _request(tmp_path)

    with pytest.raises(MaterialCreationError) as exc_info:
        backend.create(
            request,
            output_dir=tmp_path / "out",
            conditioning=_conditioning(tmp_path, request),
        )

    assert exc_info.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert [item.code for item in exc_info.value.diagnostics] == ["STEP1X_ORM_INVALID"]


def test_step1x_orm_cleanup_failure_preserves_structured_error(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner(profile_temp_directory=True)
    backend = _backend(tmp_path, runner)
    request = _request(tmp_path)

    with pytest.raises(MaterialCreationError) as exc_info:
        backend.create(
            request,
            output_dir=tmp_path / "out",
            conditioning=_conditioning(tmp_path, request),
        )

    assert exc_info.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert [item.code for item in exc_info.value.diagnostics] == ["STEP1X_ORM_INVALID"]


@pytest.mark.parametrize("pixel_count", [2, 3])
def test_step1x_oversized_orm_fails_closed_without_warning_filter_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pixel_count: int,
) -> None:
    runner = _RecordingRunner(
        orm_pixels=((255, 128, 0),) * pixel_count,
    )
    backend = _backend(tmp_path, runner)
    request = _request(tmp_path)
    monkeypatch.setattr(
        "material_agent.material_library_generation.step1x_backend._MAX_ORM_PIXELS",
        1,
    )

    def unexpected_warning_filter(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("ORM validation must not mutate global warning filters")

    monkeypatch.setattr(warnings, "simplefilter", unexpected_warning_filter)

    with pytest.raises(MaterialCreationError) as exc_info:
        backend.create(
            request,
            output_dir=tmp_path / "out",
            conditioning=_conditioning(tmp_path, request),
        )

    assert exc_info.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert [item.code for item in exc_info.value.diagnostics] == ["STEP1X_ORM_INVALID"]


@pytest.mark.parametrize("orm_size", [(4097, 1), (1, 4097)])
def test_step1x_rejects_orm_dimensions_above_service_limit(
    tmp_path: Path,
    orm_size: tuple[int, int],
) -> None:
    runner = _RecordingRunner(orm_size=orm_size)
    backend = _backend(tmp_path, runner)
    request = _request(tmp_path)

    with pytest.raises(MaterialCreationError) as exc_info:
        backend.create(
            request,
            output_dir=tmp_path / "out",
            conditioning=_conditioning(tmp_path, request),
        )

    assert exc_info.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert [item.code for item in exc_info.value.diagnostics] == ["STEP1X_ORM_INVALID"]


def test_step1x_rejects_unsupported_texture_size_before_runner(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner()
    backend = _backend(tmp_path, runner)
    request = replace(_request(tmp_path), texture_size=4097)

    with pytest.raises(MaterialCreationError) as exc_info:
        backend.create(
            request,
            output_dir=tmp_path / "out",
            conditioning=_conditioning(tmp_path, request),
        )

    assert exc_info.value.code is MaterialCreationErrorCode.INVALID_REQUEST
    assert "at most 4096" in str(exc_info.value)
    assert runner.requests == []


def test_step1x_passes_caller_cancel_event_to_runner(tmp_path: Path) -> None:
    runner = _RecordingRunner()
    backend = _backend(tmp_path, runner)
    request = _request(tmp_path)
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(MaterialCreationError) as exc_info:
        backend.create(
            request,
            output_dir=tmp_path / "out",
            conditioning=_conditioning(tmp_path, request),
            cancel_event=cancel_event,
        )

    assert exc_info.value.code is MaterialCreationErrorCode.CANCELLED
    assert runner.cancel_events == [cancel_event]


def test_material_library_generation_lazy_step1x_exports() -> None:
    import material_agent.material_library_generation as generation

    generation.__dict__.pop("Step1XMaterialCreationBackend", None)
    generation.__dict__.pop("Step1XMaterialCreationConfig", None)

    assert "Step1XMaterialCreationBackend" not in generation.__all__
    assert "Step1XMaterialCreationConfig" not in generation.__all__
    assert generation.Step1XMaterialCreationBackend is Step1XMaterialCreationBackend
    assert generation.Step1XMaterialCreationConfig is Step1XMaterialCreationConfig
    with pytest.raises(AttributeError):
        generation.__getattr__("NotAStep1XExport")


def test_material_library_generation_import_does_not_require_step1x_service() -> None:
    code = """
import builtins

real_import = builtins.__import__


def blocked_import(name, *args, **kwargs):
    if name.startswith("apps.texture_gen_step1x_service"):
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)


builtins.__import__ = blocked_import
import material_agent.material_library_generation as generation

assert "Step1XMaterialCreationBackend" not in generation.__all__
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_material_agent_declares_step1x_optional_dependency_path() -> None:
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    optional_dependencies = pyproject["project"]["optional-dependencies"]
    dev_dependencies = optional_dependencies["dev"]
    step1x_dependencies = optional_dependencies["step1x"]
    assert any(
        dependency.startswith("texture-gen-service-common>=")
        for dependency in dev_dependencies
    )
    assert any(
        dependency.startswith("texture-gen-step1x-service>=")
        for dependency in dev_dependencies
    )
    assert any(
        dependency.startswith("texture-gen-service-common>=")
        for dependency in step1x_dependencies
    )
    assert any(
        dependency.startswith("texture-gen-step1x-service>=")
        for dependency in step1x_dependencies
    )
    assert "material-agent[dev,step1x]" in optional_dependencies["all"]

    uv_sources = pyproject["tool"]["uv"]["sources"]
    assert uv_sources["texture-gen-service-common"]["path"] == (
        "../texture_gen_service_common"
    )
    assert uv_sources["texture-gen-step1x-service"]["path"] == (
        "../texture_gen_step1x_service"
    )


def _backend(
    tmp_path: Path,
    runner: _RecordingRunner,
) -> Step1XMaterialCreationBackend:
    return Step1XMaterialCreationBackend(
        config=Step1XMaterialCreationConfig(
            step1x=Step1XBackendConfig(
                runtime_dir=tmp_path / "runtime",
                model_dir=tmp_path / "models",
                cache_dir=tmp_path / "cache",
                validate_assets=False,
            ),
            model_revisions=("step1x-test", "material-anything-test"),
            custom_parameters={"guidance": 7.5},
        ),
        runner=runner,
    )


def _request(
    tmp_path: Path,
    *,
    pbr_hints: PBRHints | None = None,
    creation_mode: MaterialCreationMode = MaterialCreationMode.ASSET_UV,
) -> CreateMaterialRequest:
    source_usd = tmp_path / "source.usda"
    source_usd.write_text("#usda 1.0\n", encoding="utf-8")
    recipe = MaterialRecipe(
        name="Matte blue plastic",
        description="Opaque matte blue plastic.",
        appearance_prompt="matte blue plastic with fine molded texture",
        color="blue",
        material="plastic",
        finish="matte",
        base_color_hint=(0.05, 0.18, 0.62),
        pbr_hints=pbr_hints or PBRHints(roughness=0.72, metallic=0.0),
        reference_image_uris=((tmp_path / "recipe_ref.png").as_posix(),),
        intended_parts=(
            IntendedPart(
                semantic_label="body",
                evidence="unit test target",
                prim_path_hints=("/World/Asset/Body",),
            ),
        ),
    )
    return CreateMaterialRequest(
        source_usd=source_usd.resolve(),
        target_prim_paths=("/World/Asset/Body",),
        recipe=recipe,
        reference_image_uris=((tmp_path / "request_ref.png").as_posix(),),
        creation_mode=creation_mode,
        texture_size=512,
        backend="step1x_material_anything",
        seed=482,
    )


def _conditioning(
    tmp_path: Path,
    request: CreateMaterialRequest,
    *,
    include_source_albedo: bool = True,
    include_source_material_binding: bool = True,
) -> PreparedMaterialConditioning:
    scoped_usd = tmp_path / "scoped.usda"
    render = tmp_path / "render.png"
    _write_scoped_usd(
        scoped_usd,
        include_source_material_binding=include_source_material_binding,
    )
    _write_png(render, (64, 128, 255))
    artifacts = [
        MaterialConditioningArtifact(
            kind=MaterialConditioningKind.SCOPED_USD,
            uri=scoped_usd.as_posix(),
        ),
        MaterialConditioningArtifact(
            kind=MaterialConditioningKind.RENDER,
            uri=render.as_posix(),
            color_space=MaterialColorSpace.SRGB,
            view="oblique",
        ),
    ]
    if include_source_albedo:
        source_albedo = tmp_path / "source_albedo.png"
        _write_png(source_albedo, (20, 80, 200))
        artifacts.append(
            MaterialConditioningArtifact(
                kind=MaterialConditioningKind.SOURCE_ALBEDO,
                uri=source_albedo.as_posix(),
                color_space=MaterialColorSpace.SRGB,
                view="source_albedo",
            )
        )
    artifacts.extend(
        MaterialConditioningArtifact(
            kind=MaterialConditioningKind.REFERENCE_IMAGE,
            uri=uri,
            color_space=MaterialColorSpace.SRGB,
        )
        for uri in request.effective_reference_image_uris
    )
    return PreparedMaterialConditioning.for_request(
        request,
        artifacts=tuple(artifacts),
    )


def _write_scoped_usd(
    path: Path,
    *,
    include_source_material_binding: bool,
) -> None:
    binding_api = ""
    binding_rel = ""
    if include_source_material_binding:
        binding_api = '(\n                prepend apiSchemas = ["MaterialBindingAPI"]\n            )'
        binding_rel = f"            rel material:binding = <{SOURCE_MATERIAL_PATH}>\n"
    path.write_text(
        f"""#usda 1.0

def "World"
{{
    def "Asset"
    {{
        def Mesh "Body" {binding_api}
        {{
{binding_rel}        }}
    }}

    def Scope "Looks"
    {{
        def Material "SourceMaterial"
        {{
        }}
    }}
}}
""",
        encoding="utf-8",
    )


def _write_png(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    scanline = b"\x00" + bytes(rgb)
    path.write_bytes(
        signature
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanline, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _write_row_png(
    path: Path,
    pixels: tuple[tuple[int, int, int], ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (len(pixels), 1))
    image.putdata(pixels)
    image.save(path)


def _read_png(path: Path) -> tuple[int, int, tuple[int, int, int]]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return rgb.width, rgb.height, rgb.getpixel((0, 0))


def _read_pixels(path: Path) -> list[tuple[int, int, int]]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return [
            rgb.getpixel((x, y)) for y in range(rgb.height) for x in range(rgb.width)
        ]


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)
    )
