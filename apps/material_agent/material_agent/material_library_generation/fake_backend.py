# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic fake backend shared by material-creation work packages."""

from __future__ import annotations

import hashlib
import struct
import threading
import zlib
from enum import StrEnum
from pathlib import Path

from material_agent.material_library_generation.creation_contract import (
    BackendMaterialResult,
    CreateMaterialRequest,
    MaterialArtifactLayout,
    MaterialChannel,
    MaterialChannelArtifact,
    MaterialChannelComponent,
    MaterialChannelSource,
    MaterialColorSpace,
    MaterialComponentProvenance,
    MaterialCreationDiagnostic,
    MaterialCreationError,
    MaterialCreationErrorCode,
    MaterialCreationProvenance,
    MaterialDegradation,
    MaterialDegradationCode,
    MaterialDiagnosticSeverity,
    NormalConvention,
    ORMPacking,
    PreparedMaterialConditioning,
)


class FakeMaterialBackendBehavior(StrEnum):
    """Deterministic outcomes used by contract and downstream workflow tests."""

    SUCCESS = "success"
    DEGRADED_NORMAL = "degraded_normal"
    FAILURE = "failure"
    UNSUPPORTED = "unsupported"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)
    )


def _write_rgb_png(path: Path, rgb: tuple[int, int, int]) -> None:
    """Write a valid deterministic one-pixel RGB PNG using only the stdlib."""

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


class FakeMaterialCreationBackend:
    """A model-free backend with stable artifacts and injectable outcomes."""

    def __init__(
        self,
        behavior: FakeMaterialBackendBehavior
        | str = FakeMaterialBackendBehavior.SUCCESS,
    ) -> None:
        self.behavior = FakeMaterialBackendBehavior(behavior)
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def revision(self) -> str:
        return "fake-material-backend.v1"

    def create(
        self,
        request: CreateMaterialRequest,
        *,
        output_dir: Path,
        conditioning: PreparedMaterialConditioning | None = None,
        cancel_event: threading.Event | None = None,
    ) -> BackendMaterialResult:
        if conditioning is not None:
            conditioning.validate_request(request)
        if cancel_event is not None and cancel_event.is_set():
            raise MaterialCreationError(
                MaterialCreationErrorCode.CANCELLED,
                "Fake material creation was cancelled before execution.",
                backend=self.name,
                retryable=False,
            )

        self.calls.append(request.request_id)
        if self.behavior is FakeMaterialBackendBehavior.FAILURE:
            diagnostic = MaterialCreationDiagnostic(
                code="FAKE_BACKEND_FAILURE",
                message="Injected fake backend failure.",
                severity=MaterialDiagnosticSeverity.ERROR,
                phase="generation",
                retryable=True,
            )
            raise MaterialCreationError(
                MaterialCreationErrorCode.BACKEND_FAILURE,
                diagnostic.message,
                backend=self.name,
                retryable=True,
                diagnostics=(diagnostic,),
            )
        if self.behavior is FakeMaterialBackendBehavior.UNSUPPORTED or _is_unsupported(
            request
        ):
            diagnostic = MaterialCreationDiagnostic(
                code="FAKE_UNSUPPORTED_MATERIAL",
                message="Fake backend supports only opaque, non-transmissive recipes.",
                severity=MaterialDiagnosticSeverity.ERROR,
                phase="capability_check",
                retryable=False,
            )
            raise MaterialCreationError(
                MaterialCreationErrorCode.UNSUPPORTED_MATERIAL,
                diagnostic.message,
                backend=self.name,
                diagnostics=(diagnostic,),
            )

        layout = MaterialArtifactLayout(output_dir, request.recipe.material_id)
        digest = hashlib.sha256(request.request_id.encode("utf-8")).digest()
        albedo_path = layout.texture_path(MaterialChannel.ALBEDO)
        normal_path = layout.texture_path(MaterialChannel.NORMAL)
        orm_path = layout.texture_path(MaterialChannel.ORM)
        preview_path = layout.previews_dir / "preview.png"

        _write_rgb_png(albedo_path, (digest[0], digest[1], digest[2]))
        hints = request.recipe.pbr_hints
        roughness = round(getattr(hints, "roughness", 0.5) * 255)
        metallic = round(getattr(hints, "metallic", 0.0) * 255)
        _write_rgb_png(orm_path, (255, roughness, metallic))
        _write_rgb_png(preview_path, (digest[0], digest[1], digest[2]))

        artifacts = [
            MaterialChannelArtifact(
                channel=MaterialChannel.ALBEDO,
                path=albedo_path,
                color_space=MaterialColorSpace.SRGB,
                component_provenance=(
                    MaterialComponentProvenance(
                        component=MaterialChannelComponent.BASE_COLOR,
                        source=MaterialChannelSource.MODEL_GENERATED,
                        source_detail="deterministic fake generation",
                    ),
                ),
            ),
            MaterialChannelArtifact(
                channel=MaterialChannel.ORM,
                path=orm_path,
                color_space=MaterialColorSpace.RAW,
                component_provenance=(
                    MaterialComponentProvenance(
                        component=MaterialChannelComponent.OCCLUSION,
                        source=MaterialChannelSource.NEUTRAL_FALLBACK,
                        source_detail="neutral AO value 1.0 written to red",
                    ),
                    MaterialComponentProvenance(
                        component=MaterialChannelComponent.ROUGHNESS,
                        source=MaterialChannelSource.RECIPE_HINT,
                        source_detail="recipe roughness written to green",
                    ),
                    MaterialComponentProvenance(
                        component=MaterialChannelComponent.METALLIC,
                        source=MaterialChannelSource.RECIPE_HINT,
                        source_detail="recipe metallic written to blue",
                    ),
                ),
                packing=ORMPacking.OCCLUSION_ROUGHNESS_METALLIC,
            ),
        ]
        degradations = [
            MaterialDegradation(
                code=MaterialDegradationCode.NEUTRAL_AO,
                channels=(MaterialChannel.ORM,),
                message="The fake backend does not generate ambient occlusion.",
                fallback="ORM red contains the documented neutral AO value 1.0.",
            )
        ]
        if self.behavior is FakeMaterialBackendBehavior.DEGRADED_NORMAL:
            degradations.append(
                MaterialDegradation(
                    code=MaterialDegradationCode.MISSING_NORMAL,
                    channels=(MaterialChannel.NORMAL,),
                    message="The fake backend omitted the tangent-space normal map.",
                    fallback="Package may proceed only with explicit degraded-normal status.",
                )
            )
        else:
            _write_rgb_png(normal_path, (128, 128, 255))
            artifacts.append(
                MaterialChannelArtifact(
                    channel=MaterialChannel.NORMAL,
                    path=normal_path,
                    color_space=MaterialColorSpace.RAW,
                    component_provenance=(
                        MaterialComponentProvenance(
                            component=MaterialChannelComponent.TANGENT_NORMAL,
                            source=MaterialChannelSource.MODEL_GENERATED,
                            source_detail="deterministic flat fake normal",
                        ),
                    ),
                    normal_convention=NormalConvention.TANGENT_OPENGL,
                )
            )

        provenance = MaterialCreationProvenance.for_request(
            request,
            backend=self.name,
            backend_revision=self.revision,
            model_revisions=("fake-material-model@v1",),
            duration_seconds=0.0,
            conditioning=conditioning,
        )
        return BackendMaterialResult(
            artifacts=tuple(artifacts),
            provenance=provenance,
            degradations=tuple(degradations),
            preview_paths=(preview_path,),
        )


def _is_unsupported(request: CreateMaterialRequest) -> bool:
    hints = request.recipe.pbr_hints
    if hints is None:
        return False
    return bool(hints.opacity < 1.0 or hints.transmission > 0.0 or hints.thin_walled)
