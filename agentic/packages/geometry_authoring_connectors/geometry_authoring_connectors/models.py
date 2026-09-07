# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable wire and materialized-artifact models shared by connectors."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import math
import re
from pathlib import Path
from typing import Any, Literal, Self, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._formats import validate_geometry_bytes, validate_supporting_asset_bytes

AUTHORING_REQUEST_SCHEMA_VERSION: Literal["geometry-authoring-connectors.authoring-request.v1"] = (
    "geometry-authoring-connectors.authoring-request.v1"
)
GEOMETRY_SOURCE_SCHEMA_VERSION: Literal["geometry.source.v1"] = "geometry.source.v1"

SHA256_PATTERN = r"^[0-9a-f]{64}$"
SAFE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
SAFE_FILENAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,254}$"
MAX_INPUT_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_SOURCE_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_SOURCE_BUNDLE_ARTIFACTS = 32

ArtifactRole = Literal[
    "native_source",
    "cad_geometry",
    "render_geometry",
    "mesh_geometry",
    "collision_geometry",
    "supporting_asset",
    "manifest",
    "reference_image",
]
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

_IDENTITY_TRANSFORM = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _validate_json_value(value: Any, *, depth: int = 0) -> None:
    if depth > 16:
        raise ValueError("JSON metadata exceeds the maximum nesting depth")
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON metadata numbers must be finite")
        return
    if isinstance(value, list):
        if len(value) > 1024:
            raise ValueError("JSON metadata arrays are bounded to 1024 entries")
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 1024:
            raise ValueError("JSON metadata objects are bounded to 1024 entries")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 256:
                raise ValueError("JSON metadata keys must be bounded strings")
            _validate_json_value(item, depth=depth + 1)
        return
    raise ValueError("metadata must contain only JSON-compatible values")


def canonical_json_digest(value: Any) -> str:
    """Digest one JSON-compatible value with stable separators and key order."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    _validate_json_value(value)
    document = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest()


def _validate_upstream_edit_uri(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().rstrip("/")
    try:
        parsed = urlparse(normalized)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("upstream edit URI must be a credential-free HTTP URL") from exc
    if (
        not normalized
        or any(character.isspace() for character in normalized)
        or parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("upstream edit URI must be a credential-free HTTP URL")
    loopback = parsed.hostname.casefold() == "localhost"
    try:
        loopback = loopback or ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        pass
    if parsed.scheme != "https" and not loopback:
        raise ValueError("non-loopback upstream edit URIs require HTTPS")
    return normalized


def decode_bound_content(
    *, content_base64: str, expected_size: int, expected_sha256: str, max_bytes: int
) -> bytes:
    """Strictly decode and verify one bounded inline artifact."""

    if expected_size > max_bytes:
        raise ValueError(f"artifact exceeds the {max_bytes}-byte limit")
    try:
        content = base64.b64decode(content_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("artifact content is not canonical base64") from exc
    if len(content) != expected_size:
        raise ValueError("artifact content differs from its declared size")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("artifact content differs from its SHA-256 binding")
    return content


class InlineInputArtifact(_FrozenModel):
    """A bounded, immutable request artifact with no fetchable URL."""

    filename: str = Field(pattern=SAFE_FILENAME_PATTERN)
    media_type: Literal["image/png", "image/jpeg", "image/webp"]
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=0, le=MAX_INPUT_ARTIFACT_BYTES)
    content_base64: str = Field(max_length=(MAX_INPUT_ARTIFACT_BYTES * 4 // 3) + 8)

    @model_validator(mode="after")
    def verify_content(self) -> Self:
        content = decode_bound_content(
            content_base64=self.content_base64,
            expected_size=self.size_bytes,
            expected_sha256=self.sha256,
            max_bytes=MAX_INPUT_ARTIFACT_BYTES,
        )
        suffix = Path(self.filename).suffix.lower()
        expected_suffixes = {
            "image/png": {".png"},
            "image/jpeg": {".jpg", ".jpeg"},
            "image/webp": {".webp"},
        }[self.media_type]
        valid_signature = {
            "image/png": content.startswith(b"\x89PNG\r\n\x1a\n"),
            "image/jpeg": content.startswith(b"\xff\xd8\xff"),
            "image/webp": (
                len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WEBP"
            ),
        }[self.media_type]
        if suffix not in expected_suffixes or not valid_signature:
            raise ValueError("reference image suffix or signature differs from its media type")
        return self

    @classmethod
    def from_bytes(cls, *, filename: str, media_type: str, content: bytes) -> Self:
        return cls(
            filename=filename,
            media_type=cast(Literal["image/png", "image/jpeg", "image/webp"], media_type),
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            content_base64=base64.b64encode(content).decode("ascii"),
        )


class AuthoringRequest(_FrozenModel):
    """Provider-neutral, bounded authoring request sent to a worker."""

    schema_version: Literal["geometry-authoring-connectors.authoring-request.v1"] = (
        AUTHORING_REQUEST_SCHEMA_VERSION
    )
    request_id: str = Field(pattern=SAFE_ID_PATTERN)
    operation: Literal["generate", "revise", "export"] = "generate"
    prompt: str | None = Field(default=None, min_length=1, max_length=65_536)
    images: tuple[InlineInputArtifact, ...] = Field(default=(), max_length=8)
    target_profile: str | None = Field(default=None, min_length=1, max_length=256)
    parameters: dict[str, bool | int | float | str] = Field(
        default_factory=dict,
        max_length=256,
    )
    parameter_units: dict[str, str] = Field(default_factory=dict, max_length=256)
    target_formats: tuple[GeometryFormat, ...] = Field(min_length=1, max_length=8)
    prior_source_revision: str | None = Field(default=None, pattern=SAFE_ID_PATTERN)

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls,
        value: dict[str, bool | int | float | str],
    ) -> dict[str, bool | int | float | str]:
        _validate_json_value(value)
        if any(re.fullmatch(SAFE_ID_PATTERN, key) is None for key in value):
            raise ValueError("authoring parameter names must be safe identifiers")
        if any(isinstance(item, str) and len(item) > 4_096 for item in value.values()):
            raise ValueError("authoring string parameters exceed the 4,096-character limit")
        if len(json.dumps(value, ensure_ascii=True, allow_nan=False)) > 64 * 1024:
            raise ValueError("authoring parameters exceed the 64-KiB JSON limit")
        return value

    @field_validator("parameter_units")
    @classmethod
    def validate_parameter_units(cls, value: dict[str, str]) -> dict[str, str]:
        if any(re.fullmatch(SAFE_ID_PATTERN, key) is None for key in value):
            raise ValueError("authoring parameter unit names must be safe identifiers")
        if any(not unit or len(unit) > 64 for unit in value.values()):
            raise ValueError("authoring parameter units must be bounded values")
        return value

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if len(self.target_formats) != len(set(self.target_formats)):
            raise ValueError("target formats must be unique")
        if not set(self.parameter_units) <= set(self.parameters):
            raise ValueError("authoring parameter units reference unknown parameters")
        has_change = self.prompt is not None or bool(self.images) or bool(self.parameters)
        if self.operation in {"generate", "revise"} and not has_change:
            raise ValueError(
                "generation and revision require a prompt, image, or semantic parameter change"
            )
        if self.operation in {"revise", "export"} and self.prior_source_revision is None:
            raise ValueError("revision and export requests require prior_source_revision")
        if self.operation == "generate" and self.prior_source_revision is not None:
            raise ValueError("generation requests cannot bind a prior source revision")
        if self.operation == "export" and (
            has_change or self.target_profile is not None or self.parameter_units
        ):
            raise ValueError("export requests may select formats only")
        return self


class AuthoringCapabilities(_FrozenModel):
    provider_id: str = Field(pattern=SAFE_ID_PATTERN)
    text: bool
    image: bool
    revision: bool
    export: bool
    native_source: bool
    formats: tuple[GeometryFormat, ...]
    parameter_definitions: bool = False
    semantic_parts: bool = False
    provider_assertions: bool = False
    max_family_variants: int = Field(default=0, ge=0, le=64)

    @model_validator(mode="after")
    def validate_capabilities(self) -> Self:
        if len(self.formats) != len(set(self.formats)) or not self.formats:
            raise ValueError("authoring capability formats must be non-empty and unique")
        if self.max_family_variants and not (self.revision and self.parameter_definitions):
            raise ValueError("parameter families require revision and parameter definitions")
        return self


class WireArtifact(_FrozenModel):
    """One digest-bound provider output transported as bounded base64."""

    filename: str = Field(pattern=SAFE_FILENAME_PATTERN)
    role: ArtifactRole
    media_type: str = Field(min_length=1, max_length=128)
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=0, le=MAX_OUTPUT_ARTIFACT_BYTES)
    content_base64: str = Field(max_length=(MAX_OUTPUT_ARTIFACT_BYTES * 4 // 3) + 8)

    @model_validator(mode="after")
    def verify_content(self) -> Self:
        decode_bound_content(
            content_base64=self.content_base64,
            expected_size=self.size_bytes,
            expected_sha256=self.sha256,
            max_bytes=MAX_OUTPUT_ARTIFACT_BYTES,
        )
        return self

    @classmethod
    def from_bytes(
        cls,
        *,
        filename: str,
        role: ArtifactRole,
        media_type: str,
        content: bytes,
    ) -> Self:
        return cls(
            filename=filename,
            role=role,
            media_type=media_type,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            content_base64=base64.b64encode(content).decode("ascii"),
        )

    def content(self) -> bytes:
        return decode_bound_content(
            content_base64=self.content_base64,
            expected_size=self.size_bytes,
            expected_sha256=self.sha256,
            max_bytes=MAX_OUTPUT_ARTIFACT_BYTES,
        )


class WirePart(_FrozenModel):
    """Provider semantic part bound to one or more wire artifacts."""

    part_id: str = Field(pattern=SAFE_ID_PATTERN)
    name: str = Field(min_length=1, max_length=256)
    parent_part_id: str | None = Field(default=None, pattern=SAFE_ID_PATTERN)
    artifact_filenames: tuple[str, ...] = Field(default=(), max_length=32)
    transform: tuple[float, ...] = Field(
        default=_IDENTITY_TRANSFORM,
        min_length=16,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_part(self) -> Self:
        if self.parent_part_id == self.part_id:
            raise ValueError("a wire part cannot parent itself")
        if len(self.artifact_filenames) != len(set(self.artifact_filenames)):
            raise ValueError("wire part artifact filenames must be unique")
        return self


class WireSemanticParameter(_FrozenModel):
    """Semantic parameter definition returned by an authoring worker."""

    name: str = Field(pattern=SAFE_ID_PATTERN)
    value: bool | int | float | str
    value_type: Literal["boolean", "integer", "number", "string", "choice"] = "number"
    unit: str | None = Field(default=None, min_length=1, max_length=64)
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[bool | int | float | str, ...] = Field(default=(), max_length=256)
    group: str | None = Field(default=None, min_length=1, max_length=256)
    semantic_role: str | None = Field(default=None, pattern=SAFE_ID_PATTERN)
    effects: tuple[
        Literal[
            "exact_geometry",
            "topology",
            "placement",
            "appearance",
            "behavior",
            "metadata",
            "packaging",
        ],
        ...,
    ] = Field(default=(), max_length=16)
    affects: tuple[str, ...] = Field(default=(), max_length=4096)
    visible: bool = True
    description: str | None = Field(default=None, min_length=1, max_length=1024)

    @model_validator(mode="before")
    @classmethod
    def infer_value_type(cls, value: Any) -> Any:
        if not isinstance(value, dict) or (
            "value_type" in value and value.get("value_type") is not None
        ):
            return value
        inferred = dict(value)
        scalar = value.get("value")
        if value.get("choices"):
            inferred["value_type"] = "choice"
        elif isinstance(scalar, bool):
            inferred["value_type"] = "boolean"
        elif isinstance(scalar, int):
            inferred["value_type"] = "integer"
        elif isinstance(scalar, float):
            inferred["value_type"] = "number"
        elif isinstance(scalar, str):
            inferred["value_type"] = "string"
        return inferred

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        from geometry_authoring_contracts import GeometrySemanticParameter

        GeometrySemanticParameter.model_validate(self.model_dump(mode="python"))
        return self


class WireVerificationAssertion(_FrozenModel):
    """Provider assertion transported with, but independent from, Geometry checks."""

    assertion_id: str = Field(pattern=SAFE_ID_PATTERN)
    status: Literal["passed", "warning", "failed", "skipped"]
    summary: str = Field(min_length=1, max_length=4096)
    metrics: dict[str, Any] = Field(default_factory=dict, max_length=256)

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_json_value(value)
        return value


class WireSourceBundle(_FrozenModel):
    """Provider result before its artifacts are materialized locally."""

    schema_version: Literal["geometry.source.v1"] = GEOMETRY_SOURCE_SCHEMA_VERSION
    provider_id: str = Field(pattern=SAFE_ID_PATTERN)
    provider_version: str = Field(min_length=1, max_length=128)
    source_revision: str = Field(pattern=SAFE_ID_PATTERN)
    units: Literal["millimeter", "centimeter", "meter", "inch"]
    up_axis: Literal["X", "Y", "Z"]
    forward_axis: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"] = "+Y"
    handedness: Literal["left", "right"] = "right"
    upstream_edit_uri: str | None = Field(default=None, max_length=4096)
    artifacts: tuple[WireArtifact, ...] = Field(
        min_length=1, max_length=MAX_SOURCE_BUNDLE_ARTIFACTS
    )
    parts: tuple[WirePart, ...] = Field(default=(), max_length=4096)
    parameters: tuple[WireSemanticParameter, ...] = Field(default=(), max_length=1024)
    verification_assertions: tuple[WireVerificationAssertion, ...] = Field(
        default=(), max_length=4096
    )
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=256)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_json_value(value)
        if len(json.dumps(value, ensure_ascii=True, allow_nan=False)) > 256 * 1024:
            raise ValueError("source metadata exceeds the 256-KiB JSON limit")
        return value

    @field_validator("upstream_edit_uri")
    @classmethod
    def validate_upstream_edit_uri(cls, value: str | None) -> str | None:
        return _validate_upstream_edit_uri(value)

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        names = tuple(item.filename for item in self.artifacts)
        if len(names) != len(set(names)):
            raise ValueError("source bundle artifact filenames must be unique")
        if sum(item.size_bytes for item in self.artifacts) > MAX_SOURCE_BUNDLE_BYTES:
            raise ValueError("source bundle exceeds the aggregate byte limit")
        if not any(
            item.role in {"cad_geometry", "render_geometry", "mesh_geometry", "collision_geometry"}
            for item in self.artifacts
        ):
            raise ValueError("source bundle requires at least one geometry artifact")
        known_names = set(names)
        part_ids = tuple(item.part_id for item in self.parts)
        if len(part_ids) != len(set(part_ids)):
            raise ValueError("wire source part IDs must be unique")
        parents = {
            item.part_id: item.parent_part_id
            for item in self.parts
            if item.parent_part_id is not None
        }
        for part in self.parts:
            if part.parent_part_id is not None and part.parent_part_id not in set(part_ids):
                raise ValueError("wire source part references an unknown parent")
            if not set(part.artifact_filenames) <= known_names:
                raise ValueError("wire source part references an unknown artifact")
        for part_id in part_ids:
            seen: set[str] = set()
            current = part_id
            while current in parents:
                if current in seen:
                    raise ValueError("wire source part hierarchy contains a cycle")
                seen.add(current)
                parent = parents[current]
                assert parent is not None
                current = parent
        parameter_names = tuple(item.name for item in self.parameters)
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("wire source parameter names must be unique")
        assertion_ids = tuple(item.assertion_id for item in self.verification_assertions)
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("wire source verification assertion IDs must be unique")
        for item in self.artifacts:
            if item.role == "supporting_asset":
                validate_supporting_asset_bytes(
                    item.filename,
                    item.media_type,
                    item.content(),
                )
                continue
            if item.role not in {
                "cad_geometry",
                "render_geometry",
                "mesh_geometry",
                "collision_geometry",
            }:
                continue
            expected_role, expected_media_type = validate_geometry_bytes(
                item.filename,
                item.content(),
            )
            role_matches = item.role == expected_role or item.role == "collision_geometry"
            if not role_matches or item.media_type != expected_media_type:
                raise ValueError(
                    f"geometry artifact role or media type differs for {item.filename}"
                )
        return self


class MaterializedArtifact(_FrozenModel):
    filename: str = Field(pattern=SAFE_FILENAME_PATTERN)
    role: ArtifactRole
    media_type: str = Field(min_length=1, max_length=128)
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=0, le=MAX_OUTPUT_ARTIFACT_BYTES)
    path: Path


class MaterializedWireSourceBundle(_FrozenModel):
    """Materialized result of the connector-private wire protocol.

    Public callers receive the canonical ``GeometrySourceBundle`` from
    ``geometry_authoring_contracts`` instead.
    """

    schema_version: Literal["geometry.source.v1"] = GEOMETRY_SOURCE_SCHEMA_VERSION
    provider_id: str = Field(pattern=SAFE_ID_PATTERN)
    provider_version: str = Field(min_length=1, max_length=128)
    source_revision: str = Field(pattern=SAFE_ID_PATTERN)
    units: Literal["millimeter", "centimeter", "meter", "inch"]
    up_axis: Literal["X", "Y", "Z"]
    forward_axis: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"] = "+Y"
    handedness: Literal["left", "right"] = "right"
    upstream_edit_uri: str | None = Field(default=None, max_length=4096)
    artifacts: tuple[MaterializedArtifact, ...] = Field(
        min_length=1, max_length=MAX_SOURCE_BUNDLE_ARTIFACTS
    )
    parts: tuple[WirePart, ...] = Field(default=(), max_length=4096)
    parameters: tuple[WireSemanticParameter, ...] = Field(default=(), max_length=1024)
    verification_assertions: tuple[WireVerificationAssertion, ...] = Field(
        default=(), max_length=4096
    )
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=256)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_json_value(value)
        return value
