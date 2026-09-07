# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP and persistence models for the Geometry Agent service."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from geometry_authoring_contracts import (
    GEOMETRY_AUTHORING_IDENTIFIER_PATTERN,
    GeometryCoordinateSystem,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SERVICE_SCHEMA_VERSION: Literal["geometry-agent-service.v1"] = (
    "geometry-agent-service.v1"
)
SOURCE_SCHEMA_VERSION: Literal["geometry.source-record.v1"] = (
    "geometry.source-record.v1"
)

ArtifactRole = Literal["geometry_source", "reference_image"]
JobKind = Literal["generation", "revision", "family", "export", "run"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]
RuntimeEngine = Literal["ovphysx", "fake", "none"]
GeometryFormat = Literal[
    "step",
    "stl",
    "obj",
    "ply",
    "glb",
    "gltf",
    "3mf",
    "usd",
    "usda",
    "usdc",
]
_SOURCE_ID_PATTERN = r"^src_[0-9a-f]{64}$"
_MAX_PARAMETER_COUNT = 256
_MAX_PARAMETER_STRING_CHARACTERS = 4_096
SourceId = Annotated[str, Field(pattern=_SOURCE_ID_PATTERN)]


def _default_step_formats() -> list[GeometryFormat]:
    return ["step"]


class StrictModel(BaseModel):
    """Base model that rejects unrecognized public input fields."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _validate_parameter_mapping(
    values: dict[str, GeometryParameterRequestValue],
) -> dict[str, GeometryParameterRequestValue]:
    for name, item in values.items():
        if re.fullmatch(GEOMETRY_AUTHORING_IDENTIFIER_PATTERN, name) is None:
            raise ValueError("parameter names must be safe identifiers")
        scalar = item.value if isinstance(item, GeometryParameterInput) else item
        if isinstance(scalar, str) and len(scalar) > _MAX_PARAMETER_STRING_CHARACTERS:
            raise ValueError("parameter string values must not exceed 4,096 characters")
    return values


class ArtifactRecord(StrictModel):
    artifact_id: str
    sha256: str
    filename: str
    bundle_member: str | None = None
    media_type: str | None = None
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bundle_member(self) -> ArtifactRecord:
        if self.bundle_member is None:
            return self
        path = PurePosixPath(self.bundle_member)
        if (
            path.is_absolute()
            or self.bundle_member != path.as_posix()
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in self.bundle_member
            or "\x00" in self.bundle_member
            or any(ord(character) < 32 for character in self.bundle_member)
            or len(self.bundle_member) > 2_048
            or len(path.parts) > 64
            or path.name != self.filename
        ):
            raise ValueError("bundle_member must be a bounded safe POSIX artifact name")
        return self


class SourceRecord(StrictModel):
    schema_version: Literal["geometry.source-record.v1"] = SOURCE_SCHEMA_VERSION
    source_id: str
    role: ArtifactRole
    artifact: ArtifactRecord
    archive_entrypoint: str | None = None
    source_bundle_manifest: ArtifactRecord | None = None
    source_bundle_artifacts: list[ArtifactRecord] = Field(default_factory=list)
    source_representation_id: str | None = None
    created_at: datetime


class GeometryParameterInput(StrictModel):
    value: bool | int | float | str
    unit: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("value")
    @classmethod
    def _validate_value(
        cls,
        value: bool | int | float | str,
    ) -> bool | int | float | str:
        if isinstance(value, str) and len(value) > _MAX_PARAMETER_STRING_CHARACTERS:
            raise ValueError("parameter string values must not exceed 4,096 characters")
        return value

    @model_validator(mode="after")
    def _validate_unit(self) -> GeometryParameterInput:
        if self.unit is not None and (
            isinstance(self.value, bool) or not isinstance(self.value, int | float)
        ):
            raise ValueError("parameter units require numeric values")
        return self


GeometryParameterRequestValue = bool | int | float | str | GeometryParameterInput


class GeometryGenerationRequest(StrictModel):
    provider_id: str = Field(pattern=GEOMETRY_AUTHORING_IDENTIFIER_PATTERN)
    prompt: str | None = Field(default=None, min_length=1, max_length=32_768)
    image_source_ids: list[SourceId] = Field(default_factory=list, max_length=8)
    parameters: dict[str, GeometryParameterRequestValue] = Field(
        default_factory=dict,
        max_length=_MAX_PARAMETER_COUNT,
    )
    requested_formats: list[GeometryFormat] = Field(
        default_factory=_default_step_formats,
        min_length=1,
        max_length=8,
    )
    target_profile: str = Field(
        default="geometry-agent.insertion-or-fixture-asset.v1",
        min_length=1,
        max_length=256,
    )

    _validate_parameters = field_validator("parameters")(_validate_parameter_mapping)

    @model_validator(mode="after")
    def _require_design_intent(self) -> GeometryGenerationRequest:
        if self.prompt is None and not self.image_source_ids:
            raise ValueError("prompt or at least one image_source_id is required")
        if len(self.requested_formats) != len(set(self.requested_formats)):
            raise ValueError("requested_formats must be unique")
        if len(self.image_source_ids) != len(set(self.image_source_ids)):
            raise ValueError("image_source_ids must be unique")
        return self


class GeometryRevisionRequest(StrictModel):
    provider_id: str = Field(pattern=GEOMETRY_AUTHORING_IDENTIFIER_PATTERN)
    source_id: SourceId
    instructions: str | None = Field(default=None, min_length=1, max_length=32_768)
    image_source_ids: list[SourceId] = Field(default_factory=list, max_length=8)
    parameter_overrides: dict[str, GeometryParameterRequestValue] = Field(
        default_factory=dict,
        max_length=_MAX_PARAMETER_COUNT,
    )
    requested_formats: list[GeometryFormat] = Field(
        default_factory=_default_step_formats,
        min_length=1,
        max_length=8,
    )

    _validate_parameter_overrides = field_validator("parameter_overrides")(
        _validate_parameter_mapping
    )

    @model_validator(mode="after")
    def _require_revision_intent(self) -> GeometryRevisionRequest:
        if (
            self.instructions is None
            and not self.image_source_ids
            and not self.parameter_overrides
        ):
            raise ValueError(
                "instructions, an image_source_id, or a parameter override is required"
            )
        if len(self.requested_formats) != len(set(self.requested_formats)):
            raise ValueError("requested_formats must be unique")
        if len(self.image_source_ids) != len(set(self.image_source_ids)):
            raise ValueError("image_source_ids must be unique")
        return self


class GeometryFamilyVariant(StrictModel):
    variant_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    parameter_overrides: dict[str, GeometryParameterRequestValue] = Field(
        min_length=1,
        max_length=256,
    )
    instructions: str | None = Field(default=None, min_length=1, max_length=32_768)

    _validate_parameter_overrides = field_validator("parameter_overrides")(
        _validate_parameter_mapping
    )


class GeometryFamilyRequest(StrictModel):
    provider_id: str = Field(pattern=GEOMETRY_AUTHORING_IDENTIFIER_PATTERN)
    source_id: SourceId
    variants: list[GeometryFamilyVariant] = Field(min_length=1, max_length=64)
    requested_formats: list[GeometryFormat] = Field(
        default_factory=_default_step_formats,
        min_length=1,
        max_length=8,
    )

    @model_validator(mode="after")
    def _validate_family(self) -> GeometryFamilyRequest:
        variant_ids = [item.variant_id for item in self.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("variant_id values must be unique")
        if len(self.requested_formats) != len(set(self.requested_formats)):
            raise ValueError("requested_formats must be unique")
        return self


class GeometryExportRequest(StrictModel):
    provider_id: str = Field(pattern=GEOMETRY_AUTHORING_IDENTIFIER_PATTERN)
    source_id: SourceId
    requested_formats: list[GeometryFormat] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _validate_formats(self) -> GeometryExportRequest:
        if len(self.requested_formats) != len(set(self.requested_formats)):
            raise ValueError("requested_formats must be unique")
        return self


class GeometryImmutableExportRequest(StrictModel):
    """Export a provider-owned immutable revision not yet registered locally."""

    provider_id: str = Field(pattern=GEOMETRY_AUTHORING_IDENTIFIER_PATTERN)
    source_revision: str = Field(min_length=1, max_length=1_024)
    coordinate_system: GeometryCoordinateSystem
    rights_assertion: str = Field(min_length=1, max_length=4_096)
    license_identifier: str | None = Field(default=None, min_length=1, max_length=256)
    upstream_edit_uri: str | None = Field(default=None, min_length=1, max_length=4_096)
    requested_formats: list[GeometryFormat] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _validate_formats(self) -> GeometryImmutableExportRequest:
        if len(self.requested_formats) != len(set(self.requested_formats)):
            raise ValueError("requested_formats must be unique")
        return self


class GeometryRunRequest(StrictModel):
    source_id: SourceId
    target_profile: str = Field(
        default="geometry-agent.insertion-or-fixture-asset.v1",
        min_length=1,
        max_length=256,
    )
    target_runtime: str = Field(default="isaac-lab", min_length=1, max_length=128)
    run_runtime_validation: bool = False
    runtime_engine: RuntimeEngine | None = None
    render_evidence: bool = False
    render_preset: Literal["hero", "4view", "six_view", "vertical4", "turntable"] = (
        "six_view"
    )
    fail_on_validation_error: bool = False
    allow_lossy_recovery: bool = False
    param_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Must be empty. To apply parameter changes, create a new immutable "
            "provider revision with POST /api/geometry/revisions and run the returned "
            "source_id."
        ),
    )
    usd_tessellation_tolerance: float = Field(default=0.5, gt=0.0, le=100.0)


class ServiceError(StrictModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class JobRecord(StrictModel):
    schema_version: Literal["geometry-agent-service.v1"] = SERVICE_SCHEMA_VERSION
    job_id: str
    kind: JobKind
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    provider_id: str | None = None
    source_id: str | None = None
    result: dict[str, Any] | None = None
    error: ServiceError | None = None


class ProviderInfo(StrictModel):
    provider_id: str
    capabilities: dict[str, Any] = Field(default_factory=dict)


class ProviderListResponse(StrictModel):
    providers: list[ProviderInfo]


class HealthResponse(StrictModel):
    ok: bool
    service: Literal["geometry-agent-service"] = "geometry-agent-service"


class InfoResponse(StrictModel):
    service: Literal["geometry-agent-service"] = "geometry-agent-service"
    schema_version: Literal["geometry-agent-service.v1"] = SERVICE_SCHEMA_VERSION
    source_schema_version: Literal["geometry.source-record.v1"] = SOURCE_SCHEMA_VERSION
    authoring_bundle_schema_version: Literal["geometry.source.v1"] = (
        "geometry.source.v1"
    )
    routes: list[str]
