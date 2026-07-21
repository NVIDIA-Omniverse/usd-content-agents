# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step1X + Material Anything adapter for the WP0 material-creation protocol."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import threading
import time
import zlib
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from apps.texture_gen_service_common import TextureTarget
from apps.texture_gen_step1x_service.backend import (
    ExternalStep1XRunner,
    Step1XBackendConfig,
    Step1XRunner,
    Step1XRunRequest,
    Step1XRunResult,
    Step1XScopeInfo,
)
from PIL import Image

from material_agent.material_library_generation.creation_contract import (
    BackendMaterialResult,
    CreateMaterialRequest,
    MaterialChannel,
    MaterialChannelArtifact,
    MaterialChannelComponent,
    MaterialChannelSource,
    MaterialColorSpace,
    MaterialComponentProvenance,
    MaterialConditioningArtifact,
    MaterialConditioningKind,
    MaterialCreationBackend,
    MaterialCreationDiagnostic,
    MaterialCreationError,
    MaterialCreationErrorCode,
    MaterialCreationMode,
    MaterialCreationProvenance,
    MaterialDegradation,
    MaterialDegradationCode,
    MaterialDiagnosticSeverity,
    NormalConvention,
    ORMPacking,
    PreparedMaterialConditioning,
)

STEP1X_MATERIAL_CREATION_ADAPTER_REVISION = "step1x-material-creation-adapter.v2"
_PBR_PROFILE_POLICY = "recipe_pbr_profile.v1"
_ROUGHNESS_DETAIL_HALF_RANGE = 0.15
_METALLIC_DETAIL_HALF_RANGE = 0.05
_PROFILE_ENDPOINT_TOLERANCE = 1e-6
_STEP1X_MAX_TEXTURE_SIZE = 4096
_MAX_ORM_PIXELS = int(Image.MAX_IMAGE_PIXELS or 89_478_485)
_STEP1X_REVISION_RUNTIME_FIELDS = (
    "runtime_dir",
    "model_dir",
    "python_executable",
    "edit_script",
    "command_template",
    "timeout_sec",
    "validate_assets",
    "skip_material_anything",
    "require_upscaler",
    "extra_args",
    "required_executables",
)


@dataclass(frozen=True)
class Step1XMaterialCreationConfig:
    """Configuration for the material-creation Step1X adapter."""

    step1x: Step1XBackendConfig = field(default_factory=Step1XBackendConfig.from_env)
    backend_name: str = "step1x_material_anything"
    backend_revision: str = STEP1X_MATERIAL_CREATION_ADAPTER_REVISION
    model_revisions: tuple[str, ...] = ("step1x-runtime", "material-anything-runtime")
    strength: float = 0.8
    custom_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        if not self.model_revisions:
            raise ValueError("model_revisions must contain at least one revision")
        object.__setattr__(self, "custom_parameters", dict(self.custom_parameters))


class Step1XMaterialCreationBackend(MaterialCreationBackend):
    """Material creation backend backed by the existing Step1X runner seam."""

    def __init__(
        self,
        *,
        config: Step1XMaterialCreationConfig | None = None,
        runner: Step1XRunner | None = None,
    ) -> None:
        self.config = config or Step1XMaterialCreationConfig()
        self.runner = runner or ExternalStep1XRunner(self.config.step1x)

    @property
    def name(self) -> str:
        return self.config.backend_name

    @property
    def revision(self) -> str:
        return resolve_step1x_material_creation_revision(
            self.config.backend_revision,
            step1x=self.config.step1x,
            model_revisions=self.config.model_revisions,
            strength=self.config.strength,
            custom_parameters=self.config.custom_parameters,
        )

    def create(
        self,
        request: CreateMaterialRequest,
        *,
        output_dir: Path,
        conditioning: PreparedMaterialConditioning | None = None,
        cancel_event: threading.Event | None = None,
    ) -> BackendMaterialResult:
        """Run Step1X and normalize its textures into the WP0 result contract."""

        _validate_step1x_request(request, self.name)
        if conditioning is None:
            raise MaterialCreationError(
                MaterialCreationErrorCode.INVALID_REQUEST,
                "Step1X material creation requires prepared geometry conditioning",
                backend=self.name,
            )
        try:
            conditioning.validate_request(request)
        except ValueError as exc:
            raise MaterialCreationError(
                MaterialCreationErrorCode.INVALID_REQUEST,
                f"Step1X conditioning does not match request: {exc}",
                backend=self.name,
            ) from exc

        run_dir = Path(output_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        scoped_usd = _required_conditioning_path(
            conditioning,
            MaterialConditioningKind.SCOPED_USD,
            backend=self.name,
        )
        source_albedo = _required_conditioning_path(
            conditioning,
            MaterialConditioningKind.SOURCE_ALBEDO,
            backend=self.name,
        )
        source_material_path = _source_material_path_from_scope(
            scoped_usd,
            request.target_prim_paths,
            backend=self.name,
        )
        render_uris = tuple(
            artifact.uri
            for artifact in conditioning.artifacts
            if artifact.kind is MaterialConditioningKind.RENDER
        )
        run_request = Step1XRunRequest(
            prompt=request.recipe.appearance_prompt,
            seed=request.effective_seed,
            strength=self.config.strength,
            texture_size=request.texture_size,
            source_asset_uri=scoped_usd.as_uri(),
            source_asset_path=scoped_usd,
            source_albedo_path=source_albedo,
            reference_image_uris=request.effective_reference_image_uris,
            turntable_video_uri=None,
            multiview_image_uris=render_uris,
            target=TextureTarget(
                material_name=request.recipe.name,
                material_path=source_material_path,
                prim_paths=list(request.target_prim_paths),
                mode="per_prim",
                strict_scope=True,
            ),
            scope=Step1XScopeInfo(
                source_asset_path=scoped_usd,
                material_path=source_material_path,
                material_name=request.recipe.name,
                prim_paths=request.target_prim_paths,
                source_albedo_path=source_albedo,
            ),
            job_id=request.request_id,
            output_dir=run_dir,
            runtime_dir=self.config.step1x.runtime_dir,
            model_dir=self.config.step1x.model_dir,
            cache_dir=self.config.step1x.cache_dir,
            custom_parameters=_custom_parameters(request, self.config),
        )

        started = time.perf_counter()
        try:
            result = self.runner.run(
                run_request,
                cancel_event=cancel_event or threading.Event(),
            )
        except RuntimeError as exc:
            raise _creation_error_from_step1x(exc, backend=self.name) from exc
        duration = time.perf_counter() - started
        return _normalize_step1x_result(
            request,
            conditioning,
            result,
            output_dir=run_dir,
            backend=self,
            duration_seconds=duration,
            model_revisions=self.config.model_revisions,
        )


def _validate_step1x_request(request: CreateMaterialRequest, backend: str) -> None:
    if request.creation_mode is not MaterialCreationMode.ASSET_UV:
        raise MaterialCreationError(
            MaterialCreationErrorCode.UNSUPPORTED_MATERIAL,
            "Step1X material creation currently supports only asset_uv mode",
            backend=backend,
        )
    if request.texture_size > _STEP1X_MAX_TEXTURE_SIZE:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_REQUEST,
            "Step1X material creation texture_size must be at most "
            f"{_STEP1X_MAX_TEXTURE_SIZE}, got {request.texture_size}",
            backend=backend,
        )
    hints = request.recipe.pbr_hints
    if hints.opacity != 1.0 or hints.transmission != 0.0 or hints.thin_walled:
        raise MaterialCreationError(
            MaterialCreationErrorCode.UNSUPPORTED_MATERIAL,
            "Step1X baseline supports opaque non-transmissive materials only",
            backend=backend,
        )


def _custom_parameters(
    request: CreateMaterialRequest,
    config: Step1XMaterialCreationConfig,
) -> dict[str, Any]:
    hints = request.recipe.pbr_hints
    custom = {
        "skip_material_anything": False,
        "material": request.recipe.material,
        "finish": request.recipe.finish,
        "roughness": hints.roughness,
        "metallic": hints.metallic,
    }
    custom.update(config.custom_parameters)
    return {key: value for key, value in custom.items() if value is not None}


def _normalize_step1x_result(
    request: CreateMaterialRequest,
    conditioning: PreparedMaterialConditioning,
    result: Step1XRunResult,
    *,
    output_dir: Path,
    backend: Step1XMaterialCreationBackend,
    duration_seconds: float,
    model_revisions: tuple[str, ...],
) -> BackendMaterialResult:
    albedo = _copy_required_texture(
        result.albedo_uri,
        output_dir / "albedo.png",
        backend=backend.name,
        channel="albedo",
    )
    normal = _copy_optional_texture(
        result.normal_uri,
        output_dir / "normal.png",
        backend=backend.name,
    )
    orm = _copy_claimed_orm(
        result.orm_uri,
        output_dir / "orm.png",
        backend=backend.name,
    )
    orm_generated = orm is not None
    degradations: list[MaterialDegradation] = []
    diagnostics: list[MaterialCreationDiagnostic] = []
    profile_details: dict[str, Any] | None = None
    if orm is None:
        orm = output_dir / "orm.png"
        _write_orm_from_recipe(orm, request.recipe.pbr_hints, request.texture_size)
        degradations.extend(
            [
                MaterialDegradation(
                    code=MaterialDegradationCode.NEUTRAL_AO,
                    channels=(MaterialChannel.ORM,),
                    message="Material Anything did not provide ambient occlusion.",
                    fallback="Packed neutral white AO into the ORM red channel.",
                ),
                MaterialDegradation(
                    code=MaterialDegradationCode.RECIPE_HINT_FALLBACK,
                    channels=(MaterialChannel.ORM,),
                    message=("Material Anything roughness/metallic maps were missing."),
                    fallback=(
                        "Packed scalar recipe roughness and metallic hints into ORM."
                    ),
                ),
            ]
        )
    else:
        profile_details = _apply_recipe_pbr_profile(
            orm,
            request.recipe.pbr_hints,
            backend=backend.name,
        )
        diagnostics.append(
            MaterialCreationDiagnostic(
                code="STEP1X_RECIPE_PBR_PROFILE_APPLIED",
                message=(
                    "Material Anything ORM detail was normalized to the recipe "
                    "roughness and metallic profile."
                ),
                severity=MaterialDiagnosticSeverity.INFO,
                phase="normalize_output",
                channels=(MaterialChannel.ORM,),
                retryable=False,
                details=profile_details,
            )
        )
        degradations.append(
            MaterialDegradation(
                code=MaterialDegradationCode.NEUTRAL_AO,
                channels=(MaterialChannel.ORM,),
                message="Material Anything does not provide ambient occlusion.",
                fallback=(
                    "Packed neutral white AO into the ORM red channel during "
                    "recipe-profile normalization."
                ),
            )
        )
    if normal is None:
        degradations.append(
            MaterialDegradation(
                code=MaterialDegradationCode.MISSING_NORMAL,
                channels=(MaterialChannel.NORMAL,),
                message="Step1X/Material Anything did not provide a normal map.",
                fallback="Normal channel omitted with explicit degradation.",
            )
        )

    artifacts = [
        MaterialChannelArtifact(
            channel=MaterialChannel.ALBEDO,
            path=albedo,
            color_space=MaterialColorSpace.SRGB,
            component_provenance=(
                MaterialComponentProvenance(
                    component=MaterialChannelComponent.BASE_COLOR,
                    source=MaterialChannelSource.MODEL_GENERATED,
                    source_detail="Step1X geometry-aware albedo",
                ),
            ),
        ),
        MaterialChannelArtifact(
            channel=MaterialChannel.ORM,
            path=orm,
            color_space=MaterialColorSpace.RAW,
            packing=ORMPacking.OCCLUSION_ROUGHNESS_METALLIC,
            component_provenance=_orm_component_provenance(
                orm_generated,
                profile_details=profile_details,
            ),
        ),
    ]
    if normal is not None:
        artifacts.append(
            MaterialChannelArtifact(
                channel=MaterialChannel.NORMAL,
                path=normal,
                color_space=MaterialColorSpace.RAW,
                normal_convention=NormalConvention.TANGENT_OPENGL,
                component_provenance=(
                    MaterialComponentProvenance(
                        component=MaterialChannelComponent.TANGENT_NORMAL,
                        source=MaterialChannelSource.MODEL_GENERATED,
                        source_detail=(
                            "Step1X/Material Anything tangent-space normal output"
                        ),
                    ),
                ),
            )
        )

    provenance = MaterialCreationProvenance.for_request(
        request,
        backend=backend.name,
        backend_revision=backend.revision,
        model_revisions=model_revisions,
        duration_seconds=duration_seconds,
        conditioning=conditioning,
    )
    return BackendMaterialResult(
        artifacts=tuple(artifacts),
        provenance=provenance,
        degradations=tuple(degradations),
        diagnostics=tuple(diagnostics),
    )


def _apply_recipe_pbr_profile(
    path: Path,
    hints: Any,
    *,
    backend: str,
) -> dict[str, Any]:
    """Constrain a model ORM to one recipe while retaining spatial detail."""

    try:
        with Image.open(path) as source:
            if source.width < 1 or source.height < 1:
                raise ValueError("ORM image has invalid dimensions")
            pixel_count = source.width * source.height
            if (
                source.width > _STEP1X_MAX_TEXTURE_SIZE
                or source.height > _STEP1X_MAX_TEXTURE_SIZE
            ):
                raise ValueError(
                    "ORM image dimensions exceed Step1X limit: "
                    f"{source.width}x{source.height}; "
                    f"limit is {_STEP1X_MAX_TEXTURE_SIZE}"
                )
            if pixel_count > _MAX_ORM_PIXELS:
                raise ValueError(
                    f"ORM image has {pixel_count} pixels; limit is {_MAX_ORM_PIXELS}"
                )
            source.load()
            orm = source.convert("RGB")
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        ValueError,
    ) as exc:
        raise _invalid_orm_error(
            backend,
            f"Step1X returned an unreadable ORM image: {exc}",
        ) from exc

    try:
        _, roughness, metallic = orm.split()
        constrained_roughness, roughness_details = _constrain_profile_channel(
            roughness,
            target=float(hints.roughness),
            half_range=_ROUGHNESS_DETAIL_HALF_RANGE,
        )
        metallic_hint = float(hints.metallic)
        metallic_class = _metallic_class(metallic_hint)
        metallic_target = {
            "dielectric": 0.0,
            "metal": 1.0,
        }.get(metallic_class, metallic_hint)
        metallic_half_range = (
            0.0
            if metallic_class in {"dielectric", "metal"}
            else _METALLIC_DETAIL_HALF_RANGE
        )
        constrained_metallic, metallic_details = _constrain_profile_channel(
            metallic,
            target=metallic_target,
            half_range=metallic_half_range,
        )
    except (OSError, ValueError) as exc:
        raise _invalid_orm_error(
            backend,
            f"Step1X ORM profile normalization failed: {exc}",
        ) from exc
    metallic_details["declared_recipe_hint"] = _rounded(metallic_hint)
    metallic_details["class"] = metallic_class

    constrained = Image.merge(
        "RGB",
        (
            Image.new("L", orm.size, 255),
            constrained_roughness,
            constrained_metallic,
        ),
    )
    temporary_path = path.with_name(f".{path.name}.pbr-profile.tmp")
    try:
        constrained.save(temporary_path, format="PNG")
        temporary_path.replace(path)
    except OSError as exc:
        raise _invalid_orm_error(
            backend,
            f"Failed to write the recipe-constrained ORM image: {exc}",
        ) from exc
    finally:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)

    return {
        "policy": _PBR_PROFILE_POLICY,
        "packing": ORMPacking.OCCLUSION_ROUGHNESS_METALLIC.value,
        "detail_policy": "mean_centered_compress_only",
        "occlusion": {
            "source": MaterialChannelSource.NEUTRAL_FALLBACK.value,
            "value": 1.0,
        },
        "roughness": roughness_details,
        "metallic": metallic_details,
    }


def _constrain_profile_channel(
    channel: Image.Image,
    *,
    target: float,
    half_range: float,
) -> tuple[Image.Image, dict[str, Any]]:
    source_stats = _channel_stats(channel)
    lower = max(0.0, target - half_range)
    upper = min(1.0, target + half_range)
    lower_byte = math.ceil(lower * 255.0)
    upper_byte = math.floor(upper * 255.0)
    target_byte = min(max(round(target * 255.0), lower_byte), upper_byte)

    source_center = source_stats["mean"] * 255.0
    source_max_delta = max(
        source_center - source_stats["min"] * 255.0,
        source_stats["max"] * 255.0 - source_center,
    )
    target_max_delta = min(
        target_byte - lower_byte,
        upper_byte - target_byte,
    )
    if source_max_delta <= 0.0 or target_max_delta <= 0.0:
        detail_scale = 0.0
    else:
        detail_scale = min(1.0, target_max_delta / source_max_delta)

    lookup = [
        min(
            max(
                round(target_byte + (value - source_center) * detail_scale), lower_byte
            ),
            upper_byte,
        )
        for value in range(256)
    ]
    constrained = channel.point(lookup)
    return constrained, {
        "recipe_hint": _rounded(target),
        "target_range": [_rounded(lower), _rounded(upper)],
        "source": source_stats,
        "output": _channel_stats(constrained),
        "detail_scale": _rounded(detail_scale),
        "transform": (
            "recipe_class_constant"
            if lower_byte == upper_byte
            else "mean_centered_detail_clamp"
        ),
    }


def _channel_stats(channel: Image.Image) -> dict[str, float]:
    histogram = channel.histogram()
    count = sum(histogram)
    if count <= 0:
        raise ValueError("ORM channel has no pixels")
    populated = [value for value, frequency in enumerate(histogram) if frequency]
    mean = sum(value * frequency for value, frequency in enumerate(histogram)) / count
    return {
        "min": _rounded(populated[0] / 255.0),
        "max": _rounded(populated[-1] / 255.0),
        "mean": _rounded(mean / 255.0),
    }


def _metallic_class(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=_PROFILE_ENDPOINT_TOLERANCE):
        return "dielectric"
    if math.isclose(value, 1.0, abs_tol=_PROFILE_ENDPOINT_TOLERANCE):
        return "metal"
    return "mixed"


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _invalid_orm_error(backend: str, message: str) -> MaterialCreationError:
    diagnostic = MaterialCreationDiagnostic(
        code="STEP1X_ORM_INVALID",
        message=message,
        severity=MaterialDiagnosticSeverity.ERROR,
        phase="normalize_output",
        channels=(MaterialChannel.ORM,),
        retryable=False,
        details={"policy": _PBR_PROFILE_POLICY},
    )
    return MaterialCreationError(
        MaterialCreationErrorCode.INVALID_OUTPUT,
        message,
        backend=backend,
        retryable=False,
        diagnostics=(diagnostic,),
    )


def _orm_component_provenance(
    generated_orm: bool,
    *,
    profile_details: dict[str, Any] | None = None,
) -> tuple[MaterialComponentProvenance, ...]:
    if generated_orm:
        if profile_details is None:
            raise ValueError("generated ORM requires recipe-profile details")
        metallic_class = profile_details["metallic"]["class"]
        metallic_source = (
            MaterialChannelSource.RECIPE_HINT
            if metallic_class in {"dielectric", "metal"}
            else MaterialChannelSource.DERIVED
        )
        return (
            MaterialComponentProvenance(
                MaterialChannelComponent.OCCLUSION,
                MaterialChannelSource.NEUTRAL_FALLBACK,
                "Neutral white AO packed during Step1X recipe-profile normalization",
            ),
            MaterialComponentProvenance(
                MaterialChannelComponent.ROUGHNESS,
                MaterialChannelSource.DERIVED,
                "Material Anything roughness detail bounded around the recipe hint",
            ),
            MaterialComponentProvenance(
                MaterialChannelComponent.METALLIC,
                metallic_source,
                (
                    "Recipe metallic class enforced over the Material Anything "
                    "prediction"
                    if metallic_class in {"dielectric", "metal"}
                    else "Material Anything metallic detail bounded around the recipe hint"
                ),
            ),
        )
    return (
        MaterialComponentProvenance(
            MaterialChannelComponent.OCCLUSION,
            MaterialChannelSource.NEUTRAL_FALLBACK,
            "Neutral AO fallback",
        ),
        MaterialComponentProvenance(
            MaterialChannelComponent.ROUGHNESS,
            MaterialChannelSource.RECIPE_HINT,
            "Recipe roughness hint",
        ),
        MaterialComponentProvenance(
            MaterialChannelComponent.METALLIC,
            MaterialChannelSource.RECIPE_HINT,
            "Recipe metallic hint",
        ),
    )


def _required_conditioning_path(
    conditioning: PreparedMaterialConditioning,
    kind: MaterialConditioningKind,
    *,
    backend: str,
) -> Path:
    path = _optional_conditioning_path(conditioning, kind, backend=backend)
    if path is None:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_REQUEST,
            f"prepared conditioning is missing required {kind.value} artifact",
            backend=backend,
        )
    return path


def _optional_conditioning_path(
    conditioning: PreparedMaterialConditioning,
    kind: MaterialConditioningKind,
    *,
    backend: str,
) -> Path | None:
    artifact = _first_artifact(conditioning.artifacts, kind)
    if artifact is None:
        return None
    return _local_path_from_uri(
        artifact.uri,
        backend=backend,
        error_code=MaterialCreationErrorCode.INVALID_REQUEST,
        uri_kind=f"{kind.value} conditioning",
    ).resolve()


def _first_artifact(
    artifacts: tuple[MaterialConditioningArtifact, ...],
    kind: MaterialConditioningKind,
) -> MaterialConditioningArtifact | None:
    return next((artifact for artifact in artifacts if artifact.kind is kind), None)


def _copy_required_texture(
    uri: str,
    destination: Path,
    *,
    backend: str,
    channel: str,
) -> Path:
    path = _copy_optional_texture(uri, destination, backend=backend)
    if path is None:
        raise MaterialCreationError(
            MaterialCreationErrorCode.PARTIAL_OUTPUT,
            f"Step1X did not produce a usable {channel} texture",
            backend=backend,
        )
    return path


def _copy_optional_texture(
    uri: str | None,
    destination: Path,
    *,
    backend: str,
) -> Path | None:
    if uri is None:
        return None
    source = _local_path_from_uri(
        uri,
        backend=backend,
        error_code=MaterialCreationErrorCode.PARTIAL_OUTPUT,
        uri_kind="Step1X output",
    )
    if not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination.resolve()


def _copy_claimed_orm(
    uri: str | None,
    destination: Path,
    *,
    backend: str,
) -> Path | None:
    if uri is None:
        return None
    try:
        path = _copy_optional_texture(uri, destination, backend=backend)
    except (MaterialCreationError, OSError) as exc:
        raise _invalid_orm_error(
            backend,
            f"Step1X claimed an unusable ORM output: {exc}",
        ) from exc
    if path is None:
        raise _invalid_orm_error(
            backend,
            "Step1X claimed an ORM output, but its path is not a readable file.",
        )
    return path


def _local_path_from_uri(
    uri: str,
    *,
    backend: str,
    error_code: MaterialCreationErrorCode,
    uri_kind: str,
) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(uri)
    raise MaterialCreationError(
        error_code,
        f"{uri_kind} URI is not local: {uri}",
        backend=backend,
    )


def _source_material_path_from_scope(
    scoped_usd: Path,
    target_prim_paths: tuple[str, ...],
    *,
    backend: str,
) -> str:
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(scoped_usd))
    if stage is None:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_REQUEST,
            f"failed to open scoped USD conditioning: {scoped_usd}",
            backend=backend,
        )

    material_paths: set[str] = set()
    for prim_path in target_prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim:
            for candidate in Usd.PrimRange(prim):
                if candidate.IsA(UsdGeom.Mesh):
                    material, _ = UsdShade.MaterialBindingAPI(
                        candidate
                    ).ComputeBoundMaterial()
                    if material and material.GetPrim():
                        material_paths.add(str(material.GetPath()))

    if len(material_paths) != 1:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_REQUEST,
            "Step1X scoped USD conditioning must contain exactly one source "
            f"material binding for target prims; found {sorted(material_paths)}",
            backend=backend,
        )
    return next(iter(material_paths))


def _creation_error_from_step1x(
    exc: RuntimeError,
    *,
    backend: str,
) -> MaterialCreationError:
    message = str(exc)
    upper = message.upper()
    if "CANCEL" in upper:
        code = MaterialCreationErrorCode.CANCELLED
    elif "TIMEOUT" in upper:
        code = MaterialCreationErrorCode.TIMEOUT
    elif "CUDA" in upper and (
        "OOM" in upper or "OUT_OF_MEMORY" in upper or "OUT OF MEMORY" in upper
    ):
        code = MaterialCreationErrorCode.CUDA_OUT_OF_MEMORY
    elif ("CHECKPOINT" in upper or "MODEL" in upper) and "MISSING" in upper:
        code = MaterialCreationErrorCode.MISSING_CHECKPOINT
    elif "UNAVAILABLE" in upper or "UNREACHABLE" in upper:
        code = MaterialCreationErrorCode.BACKEND_UNAVAILABLE
    elif "OUTPUT_MISSING" in upper or "PARTIAL" in upper:
        code = MaterialCreationErrorCode.PARTIAL_OUTPUT
    else:
        code = MaterialCreationErrorCode.BACKEND_FAILURE
    return MaterialCreationError(code, message, backend=backend)


def _write_orm_from_recipe(path: Path, hints: Any, texture_size: int) -> None:
    ao = 255
    roughness = round(max(0.0, min(1.0, hints.roughness)) * 255)
    metallic = round(max(0.0, min(1.0, hints.metallic)) * 255)
    _write_rgb_png(path, (ao, roughness, metallic), size=texture_size)


def _write_rgb_png(path: Path, rgb: tuple[int, int, int], *, size: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    scanline = b"\x00" + (bytes(rgb) * size)
    path.write_bytes(
        signature
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanline * size, level=9))
        + _png_chunk(b"IEND", b"")
    )


def resolve_step1x_material_creation_revision(
    base_revision: str,
    *,
    step1x: Step1XBackendConfig,
    model_revisions: tuple[str, ...],
    strength: float,
    custom_parameters: dict[str, Any],
) -> str:
    """Return an idempotent revision for the current generation configuration."""

    generation_settings = {
        "model_revisions": model_revisions,
        "strength": strength,
        "custom_parameters": custom_parameters,
        "runtime": {
            field_name: getattr(step1x, field_name, None)
            for field_name in _STEP1X_REVISION_RUNTIME_FIELDS
        },
    }
    digest = hashlib.sha256(
        json.dumps(
            generation_settings,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:12]
    suffix = f"+cfg.{digest}"
    if base_revision.endswith(suffix):
        return base_revision
    return f"{base_revision}{suffix}"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)
    )


def result_fingerprint(result: BackendMaterialResult) -> str:
    """Return a stable fingerprint for Step1X adapter test/evidence records."""

    payload = result.to_dict()
    payload["provenance"].pop("duration_seconds", None)
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
