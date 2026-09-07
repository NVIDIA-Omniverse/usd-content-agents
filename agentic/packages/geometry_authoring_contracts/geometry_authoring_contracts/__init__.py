# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral contracts for external geometry authoring systems.

Adapters in this module exchange JSON and verify artifact identities. They never
import, evaluate, or execute a provider's native source representation.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self, runtime_checkable
from urllib.parse import urlparse

import requests
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

GEOMETRY_SOURCE_SCHEMA_VERSION: Literal["geometry.source.v1"] = "geometry.source.v1"
GEOMETRY_AUTHORING_REQUEST_SCHEMA_VERSION: Literal[
    "content-agent-workflows.geometry-authoring-request.v1"
] = "content-agent-workflows.geometry-authoring-request.v1"
GEOMETRY_AUTHORING_REVISION_REQUEST_SCHEMA_VERSION: Literal[
    "content-agent-workflows.geometry-authoring-revision-request.v1"
] = "content-agent-workflows.geometry-authoring-revision-request.v1"
GEOMETRY_AUTHORING_EXPORT_REQUEST_SCHEMA_VERSION: Literal[
    "content-agent-workflows.geometry-authoring-export-request.v1"
] = "content-agent-workflows.geometry-authoring-export-request.v1"
GEOMETRY_AUTHORING_FAMILY_REQUEST_SCHEMA_VERSION: Literal[
    "content-agent-workflows.geometry-authoring-family-request.v1"
] = "content-agent-workflows.geometry-authoring-family-request.v1"
GEOMETRY_AUTHORING_CAPABILITY_SCHEMA_VERSION: Literal[
    "content-agent-workflows.geometry-authoring-capabilities.v1"
] = "content-agent-workflows.geometry-authoring-capabilities.v1"
GEOMETRY_AUTHORING_FAILURE_SCHEMA_VERSION: Literal[
    "content-agent-workflows.geometry-authoring-failure.v1"
] = "content-agent-workflows.geometry-authoring-failure.v1"
GEOMETRY_AUTHORING_RECEIPT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.geometry-authoring-receipt.v1"
] = "content-agent-workflows.geometry-authoring-receipt.v1"

_MAX_PROVIDER_JSON_BYTES = 16 * 1024 * 1024
_HTTP_CHUNK_SIZE = 64 * 1024
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
GEOMETRY_AUTHORING_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_IDENTIFIER_PATTERN = GEOMETRY_AUTHORING_IDENTIFIER_PATTERN
_FORMAT_PATTERN = r"^[a-z0-9][a-z0-9.+_-]{0,63}$"
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

GeometryAuthoringOperation = Literal["capabilities", "generate", "revise", "export"]
GeometryRepresentationRole = Literal[
    "native_source",
    "design_exchange",
    "render_geometry",
    "collision_candidate",
    "supporting_asset",
    "reference",
    "metadata",
]
_DELIVERABLE_REPRESENTATION_ROLES = frozenset(
    {"design_exchange", "render_geometry", "collision_candidate"}
)
GeometryInputModality = Literal["text", "image", "text_image", "existing_source"]
GeometryParameterScalar = bool | int | float | str
GeometryParameterType = Literal["boolean", "integer", "number", "string", "choice"]
GeometryParameterEffect = Literal[
    "exact_geometry",
    "topology",
    "placement",
    "appearance",
    "behavior",
    "metadata",
    "packaging",
]
GeometryAuthoringFeature = Literal[
    "image_conditioning",
    "immutable_revisions",
    "semantic_parameters",
    "parameter_definitions",
    "parameter_families",
    "semantic_parts",
    "provider_assertions",
    "native_source",
    "multi_format_output",
]
GeometryFailureCode = Literal[
    "provider_unavailable",
    "unsupported_operation",
    "transport_error",
    "http_error",
    "invalid_response",
    "provider_failure",
    "input_drift",
    "artifact_mismatch",
]

_FORMAT_FAMILIES: dict[str, frozenset[str]] = {
    "usd": frozenset({"usd", "usda", "usdc"}),
}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class GeometryArtifactBinding(_FrozenModel):
    """Portable immutable identity for one provider input or output artifact."""

    path: str = Field(min_length=1, max_length=8_192)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int = Field(ge=0)


@dataclass(frozen=True)
class _CapturedArtifact:
    path: Path
    sha256: str
    size_bytes: int
    data: bytes | None


def _read_regular_artifact(
    root: str | Path,
    candidate: str | Path,
    *,
    max_bytes: int | None = None,
    capture_bytes: bool = False,
) -> _CapturedArtifact:
    """Read a stable, contained regular file without following symlinks."""

    if (
        not getattr(os, "O_NOFOLLOW", 0)
        or not getattr(os, "O_DIRECTORY", 0)
        or os.open not in os.supports_dir_fd
    ):
        raise RuntimeError(
            "secure provider artifact reads require descriptor-relative POSIX file access"
        )
    trusted_root = Path(os.path.abspath(Path(root).expanduser()))
    resolved_root = trusted_root.resolve(strict=True)
    if resolved_root != trusted_root or not trusted_root.is_dir():
        raise ValueError(f"artifact root is not a direct directory: {trusted_root}")
    expected_root = os.stat(trusted_root, follow_symlinks=False)
    raw = Path(candidate).expanduser()
    if any(part == ".." for part in raw.parts):
        raise ValueError("artifact path must not contain parent traversal")
    absolute = Path(os.path.abspath(raw if raw.is_absolute() else trusted_root / raw))
    if not absolute.is_relative_to(trusted_root):
        raise ValueError(f"artifact escapes its trusted root: {absolute}")
    relative = absolute.relative_to(trusted_root)
    if not relative.parts:
        raise ValueError("artifact path must identify a file below its trusted root")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    )
    root_descriptor = current_descriptor = descriptor = -1
    try:
        root_descriptor = os.open(trusted_root, directory_flags)
        opened_root = os.fstat(root_descriptor)
        if (expected_root.st_dev, expected_root.st_ino) != (
            opened_root.st_dev,
            opened_root.st_ino,
        ):
            raise ValueError("artifact root changed while being opened")
        current_descriptor = root_descriptor
        for component in relative.parts[:-1]:
            child_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=current_descriptor,
            )
            opened_directory = os.fstat(child_descriptor)
            if not stat.S_ISDIR(opened_directory.st_mode):
                os.close(child_descriptor)
                raise ValueError("artifact parent is not a directory")
            if current_descriptor != root_descriptor:
                os.close(current_descriptor)
            current_descriptor = child_descriptor
        descriptor = os.open(
            relative.parts[-1],
            file_flags,
            dir_fd=current_descriptor,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"artifact is not a single-link regular file: {absolute}")
        if max_bytes is not None and before.st_size > max_bytes:
            raise ValueError(f"artifact exceeds the {max_bytes}-byte limit")

        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if capture_bytes else None
        size_bytes = 0
        while True:
            chunk = os.read(descriptor, _HTTP_CHUNK_SIZE)
            if not chunk:
                break
            size_bytes += len(chunk)
            if max_bytes is not None and size_bytes > max_bytes:
                raise ValueError(f"artifact exceeds the {max_bytes}-byte limit")
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)

        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or size_bytes != after.st_size:
            raise ValueError(f"artifact changed while being read: {absolute}")
    except OSError as exc:
        raise ValueError("artifact path changed or traverses a symbolic link") from exc
    finally:
        for opened_descriptor in {descriptor, current_descriptor, root_descriptor}:
            if opened_descriptor >= 0:
                os.close(opened_descriptor)

    return _CapturedArtifact(
        path=absolute,
        sha256=digest.hexdigest(),
        size_bytes=size_bytes,
        data=b"".join(chunks) if chunks is not None else None,
    )


class GeometryAuthoringProviderIdentity(_FrozenModel):
    """Stable identity reported by an external authoring provider."""

    provider_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    provider_version: str = Field(min_length=1, max_length=256)


class GeometryCoordinateSystem(_FrozenModel):
    """Units and axes needed to interpret every representation in a bundle."""

    meters_per_unit: float = Field(gt=0)
    up_axis: Literal["X", "Y", "Z"] = "Z"
    forward_axis: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"] = "+Y"
    handedness: Literal["left", "right"] = "right"

    @model_validator(mode="after")
    def validate_axes(self) -> Self:
        if self.forward_axis[-1] == self.up_axis:
            raise ValueError("forward and up axes must be different")
        return self


class GeometryRepresentationBinding(_FrozenModel):
    """One immutable representation produced by an authoring provider."""

    representation_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    role: GeometryRepresentationRole
    format: str = Field(pattern=_FORMAT_PATTERN)
    media_type: str = Field(min_length=3, max_length=256)
    artifact: GeometryArtifactBinding
    root_prim_path: str | None = Field(default=None, min_length=1, max_length=2_048)


class GeometryPartBinding(_FrozenModel):
    """Provider-neutral semantic part and its source-local transform."""

    part_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    name: str = Field(min_length=1, max_length=256)
    parent_part_id: str | None = Field(
        default=None,
        pattern=_IDENTIFIER_PATTERN,
    )
    representation_ids: tuple[str, ...] = Field(default=(), max_length=256)
    transform: tuple[float, ...] = Field(
        default=_IDENTITY_TRANSFORM,
        min_length=16,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_part(self) -> Self:
        if self.parent_part_id == self.part_id:
            raise ValueError("a geometry part cannot parent itself")
        if len(self.representation_ids) != len(set(self.representation_ids)):
            raise ValueError("part representation IDs must be unique")
        return self


def _parameter_value_type(value: Any) -> GeometryParameterType:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    raise ValueError("parameter values must be JSON scalars")


def _validate_parameter_scalar(
    value: GeometryParameterScalar,
    value_type: GeometryParameterType,
    *,
    label: str,
) -> None:
    if isinstance(value, str) and len(value) > 4_096:
        raise ValueError(f"{label} exceeds the 4,096-character limit")
    valid = {
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, int | float) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "choice": isinstance(value, bool | int | float | str),
    }[value_type]
    if not valid:
        raise ValueError(f"{label} does not match declared type {value_type}")


def _typed_scalar_key(value: GeometryParameterScalar) -> tuple[str, GeometryParameterScalar]:
    return (_parameter_value_type(value), value)


def _choice_scalar_key(value: GeometryParameterScalar) -> tuple[str, GeometryParameterScalar]:
    value_type = _parameter_value_type(value)
    if value_type in {"integer", "number"}:
        return ("number", float(value))
    return (value_type, value)


def _contains_typed_scalar(
    values: tuple[GeometryParameterScalar, ...],
    candidate: GeometryParameterScalar,
) -> bool:
    candidate_key = _typed_scalar_key(candidate)
    if candidate_key[0] in {"integer", "number"}:
        return any(
            _parameter_value_type(item) in {"integer", "number"} and float(item) == float(candidate)
            for item in values
        )
    return any(_typed_scalar_key(item) == candidate_key for item in values)


class GeometrySemanticParameter(_FrozenModel):
    """Semantic parameter retained with the provider-native source revision."""

    name: str = Field(pattern=_IDENTIFIER_PATTERN)
    value: GeometryParameterScalar
    value_type: GeometryParameterType = "number"
    unit: str | None = Field(default=None, min_length=1, max_length=64)
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[GeometryParameterScalar, ...] = Field(default=(), max_length=256)
    group: str | None = Field(default=None, min_length=1, max_length=256)
    semantic_role: str | None = Field(default=None, pattern=_IDENTIFIER_PATTERN)
    effects: tuple[GeometryParameterEffect, ...] = Field(default=(), max_length=16)
    affects: tuple[str, ...] = Field(default=(), max_length=4_096)
    visible: bool = True
    description: str | None = Field(default=None, min_length=1, max_length=1_024)

    @model_validator(mode="before")
    @classmethod
    def infer_value_type(cls, value: Any) -> Any:
        if not isinstance(value, Mapping) or (
            "value_type" in value and value.get("value_type") is not None
        ):
            return value
        inferred = dict(value)
        if value.get("choices"):
            inferred["value_type"] = "choice"
        else:
            inferred["value_type"] = _parameter_value_type(value.get("value"))
        return inferred

    @model_validator(mode="after")
    def validate_parameter(self) -> Self:
        _validate_parameter_scalar(self.value, self.value_type, label="parameter value")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("parameter minimum must not exceed maximum")
        if self.step is not None and self.step <= 0:
            raise ValueError("parameter step must be positive")
        if len(self.effects) != len(set(self.effects)):
            raise ValueError("parameter effects must be unique")
        if len(self.affects) != len(set(self.affects)):
            raise ValueError("parameter affects entries must be unique")
        if any(not item or len(item) > 256 for item in self.affects):
            raise ValueError("parameter affects entries must be bounded values")
        if self.value_type == "choice":
            if not self.choices or not _contains_typed_scalar(self.choices, self.value):
                raise ValueError("choice parameter requires choices containing its value")
            if len({_choice_scalar_key(item) for item in self.choices}) != len(self.choices):
                raise ValueError("parameter choices must be unique")
            choice_types = {_parameter_value_type(item) for item in self.choices}
            if choice_types <= {"integer", "number"}:
                pass
            elif len(choice_types) != 1:
                raise ValueError("parameter choices must have one scalar type")
        elif self.choices:
            raise ValueError("only choice parameters may declare choices")
        numeric_value = (
            float(self.value)
            if isinstance(self.value, int | float) and not isinstance(self.value, bool)
            else None
        )
        numeric_parameter = self.value_type in {"integer", "number"} or (
            self.value_type == "choice" and numeric_value is not None
        )
        if not numeric_parameter and any(
            item is not None for item in (self.minimum, self.maximum, self.step)
        ):
            raise ValueError("non-numeric parameters cannot declare bounds or step")
        if self.unit is not None and not numeric_parameter:
            raise ValueError("parameter unit requires a numeric value")
        if numeric_value is not None:
            if self.minimum is not None and numeric_value < self.minimum:
                raise ValueError("parameter value is below its minimum")
            if self.maximum is not None and numeric_value > self.maximum:
                raise ValueError("parameter value is above its maximum")
            if self.step is not None:
                anchor = self.minimum if self.minimum is not None else 0.0
                steps = (numeric_value - anchor) / self.step
                if not math.isclose(steps, round(steps), rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError("parameter value does not align to its step")
        for choice in self.choices:
            if not isinstance(choice, int | float) or isinstance(choice, bool):
                continue
            numeric_choice = float(choice)
            if self.minimum is not None and numeric_choice < self.minimum:
                raise ValueError("parameter choice is below its minimum")
            if self.maximum is not None and numeric_choice > self.maximum:
                raise ValueError("parameter choice is above its maximum")
            if self.step is not None:
                anchor = self.minimum if self.minimum is not None else 0.0
                steps = (numeric_choice - anchor) / self.step
                if not math.isclose(steps, round(steps), rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError("parameter choice does not align to its step")
        return self


class GeometryVerificationAssertion(_FrozenModel):
    """Provider-authored assertion retained as evidence, never trusted as Geometry truth."""

    assertion_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    status: Literal["passed", "warning", "failed", "skipped"]
    summary: str = Field(min_length=1, max_length=4_096)
    metrics: dict[str, Any] = Field(default_factory=dict, max_length=256)


class GeometryRightsAssertion(_FrozenModel):
    """Provider assertion that the returned source may enter this workflow."""

    authorized_for_requested_use: Literal[True] = True
    assertion: str = Field(min_length=1, max_length=4_096)
    license_identifier: str | None = Field(default=None, min_length=1, max_length=256)


class GeometrySourceProvenance(_FrozenModel):
    """Immutable authoring lineage without embedded credentials."""

    request_digest: str = Field(pattern=_SHA256_PATTERN)
    parent_bundle_id: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    input_artifacts: tuple[GeometryArtifactBinding, ...] = Field(
        default=(),
        max_length=256,
    )
    upstream_edit_uri: str | None = Field(default=None, max_length=4_096)

    @field_validator("upstream_edit_uri")
    @classmethod
    def validate_upstream_edit_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_credential_free_url(
            value,
            label="upstream edit URI",
            allow_loopback_http=True,
        )


class GeometrySourceBundle(_FrozenModel):
    """Content-addressed geometry handoff produced by an authoring provider."""

    schema_version: Literal["geometry.source.v1"] = GEOMETRY_SOURCE_SCHEMA_VERSION
    bundle_id: str = Field(pattern=_SHA256_PATTERN)
    producer: GeometryAuthoringProviderIdentity
    source_revision: str = Field(min_length=1, max_length=1_024)
    coordinate_system: GeometryCoordinateSystem
    representations: tuple[GeometryRepresentationBinding, ...] = Field(
        min_length=1,
        max_length=256,
    )
    parts: tuple[GeometryPartBinding, ...] = Field(default=(), max_length=4_096)
    parameters: tuple[GeometrySemanticParameter, ...] = Field(
        default=(),
        max_length=1_024,
    )
    verification_assertions: tuple[GeometryVerificationAssertion, ...] = Field(
        default=(),
        max_length=4_096,
    )
    provenance: GeometrySourceProvenance
    rights: GeometryRightsAssertion

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        representation_ids = tuple(item.representation_id for item in self.representations)
        if len(representation_ids) != len(set(representation_ids)):
            raise ValueError("source representation IDs must be unique")
        representation_paths = tuple(item.artifact.path for item in self.representations)
        if len(representation_paths) != len(set(representation_paths)):
            raise ValueError("source representation paths must be unique")
        parameter_names = tuple(item.name for item in self.parameters)
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("source parameter names must be unique")
        assertion_ids = tuple(item.assertion_id for item in self.verification_assertions)
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("source verification assertion IDs must be unique")

        part_ids = tuple(item.part_id for item in self.parts)
        if len(part_ids) != len(set(part_ids)):
            raise ValueError("source part IDs must be unique")
        known_parts = set(part_ids)
        known_representations = set(representation_ids)
        parents: dict[str, str] = {}
        for part in self.parts:
            if part.parent_part_id is not None:
                if part.parent_part_id not in known_parts:
                    raise ValueError("source part references an unknown parent")
                parents[part.part_id] = part.parent_part_id
            if not set(part.representation_ids) <= known_representations:
                raise ValueError("source part references an unknown representation")
        for part_id in part_ids:
            seen: set[str] = set()
            current = part_id
            while current in parents:
                if current in seen:
                    raise ValueError("source part hierarchy contains a cycle")
                seen.add(current)
                current = parents[current]
        return self


class GeometryAuthoringReference(_FrozenModel):
    """Digest-bound image or document supplied to an authoring provider."""

    reference_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    kind: Literal["image", "sketch", "drawing", "document"]
    media_type: str = Field(min_length=3, max_length=256)
    artifact: GeometryArtifactBinding


class GeometryAuthoringParameterValue(_FrozenModel):
    """One requested semantic value or revision override."""

    name: str = Field(pattern=_IDENTIFIER_PATTERN)
    value: GeometryParameterScalar
    unit: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        value_type = _parameter_value_type(self.value)
        _validate_parameter_scalar(self.value, value_type, label="authoring parameter value")
        if self.unit is not None and value_type not in {"integer", "number"}:
            raise ValueError("authoring parameter unit requires a numeric value")
        return self


def resolve_semantic_parameter_overrides(
    parameters: tuple[GeometrySemanticParameter, ...],
    overrides: tuple[GeometryAuthoringParameterValue, ...],
    *,
    allow_undeclared: bool = False,
) -> tuple[GeometrySemanticParameter, ...]:
    """Validate one parameter row and return the complete resulting definitions."""

    parameter_by_name = {item.name: item for item in parameters}
    if len(parameter_by_name) != len(parameters):
        raise ValueError("semantic parameter names must be unique")
    override_by_name = {item.name: item for item in overrides}
    if len(override_by_name) != len(overrides):
        raise ValueError("authoring parameter names must be unique")

    unknown = set(override_by_name).difference(parameter_by_name)
    if unknown and not allow_undeclared:
        raise ValueError(
            "parameter overrides contain undeclared names: " + ", ".join(sorted(unknown))
        )

    resolved: list[GeometrySemanticParameter] = []
    for parameter in parameters:
        override = override_by_name.get(parameter.name)
        if override is None:
            resolved.append(parameter)
            continue
        if override.unit is not None and parameter.unit != override.unit:
            raise ValueError(
                f"parameter {parameter.name!r} override unit differs from its definition"
            )
        payload = parameter.model_dump(mode="python")
        payload["value"] = override.value
        resolved.append(GeometrySemanticParameter.model_validate(payload))

    if allow_undeclared:
        for name in sorted(unknown):
            override = override_by_name[name]
            resolved.append(
                GeometrySemanticParameter(
                    name=override.name,
                    value=override.value,
                    unit=override.unit,
                )
            )
    return tuple(resolved)


def validate_returned_parameter_values(
    parameters: tuple[GeometrySemanticParameter, ...],
    requested: tuple[GeometryAuthoringParameterValue, ...],
) -> None:
    """Require a provider result to report every exact requested semantic value."""

    parameter_by_name = {item.name: item for item in parameters}
    for item in requested:
        returned = parameter_by_name.get(item.name)
        if returned is None:
            raise ValueError(f"provider result omitted requested parameter {item.name!r}")
        if not _contains_typed_scalar((returned.value,), item.value):
            raise ValueError(f"provider result changed requested parameter {item.name!r}")
        if item.unit is not None and returned.unit != item.unit:
            raise ValueError(f"provider result changed requested unit for {item.name!r}")


def validate_returned_parameter_state(
    parameters: tuple[GeometrySemanticParameter, ...],
    expected: tuple[GeometrySemanticParameter, ...],
) -> None:
    """Require a provider result to preserve one complete parameter-family row."""

    returned_by_name = {item.name: item for item in parameters}
    expected_by_name = {item.name: item for item in expected}
    if len(returned_by_name) != len(parameters):
        raise ValueError("provider result contains duplicate parameter names")
    if len(expected_by_name) != len(expected):
        raise ValueError("expected parameter state contains duplicate names")
    missing = set(expected_by_name).difference(returned_by_name)
    extra = set(returned_by_name).difference(expected_by_name)
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected " + ", ".join(sorted(extra)))
        raise ValueError("provider result changed parameter definitions: " + "; ".join(details))
    for name, expected_parameter in expected_by_name.items():
        returned_parameter = returned_by_name[name]
        if returned_parameter.model_dump(mode="python") != expected_parameter.model_dump(
            mode="python"
        ):
            raise ValueError(f"provider result changed parameter definition or value for {name!r}")


def geometry_requested_formats_satisfied(
    requested: tuple[str, ...],
    available: set[str] | frozenset[str],
) -> bool:
    """Return whether available representations cover each requested format family."""

    return all(
        bool(available.intersection(_FORMAT_FAMILIES.get(item, frozenset({item}))))
        for item in requested
    )


def geometry_deliverable_representation_formats(
    representations: Iterable[GeometryRepresentationBinding],
) -> frozenset[str]:
    """Return formats that consumers may use as requested geometry exports."""

    return frozenset(
        item.format for item in representations if item.role in _DELIVERABLE_REPRESENTATION_ROLES
    )


class GeometryAuthoringRequest(_FrozenModel):
    """Exact text/image request sent to one explicitly selected provider."""

    schema_version: Literal["content-agent-workflows.geometry-authoring-request.v1"] = (
        GEOMETRY_AUTHORING_REQUEST_SCHEMA_VERSION
    )
    request_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    prompt: str | None = Field(default=None, min_length=1, max_length=32_768)
    references: tuple[GeometryAuthoringReference, ...] = Field(
        default=(),
        max_length=32,
    )
    target_profile: str = Field(min_length=1, max_length=256)
    requested_formats: tuple[str, ...] = Field(min_length=1, max_length=32)
    parameters: tuple[GeometryAuthoringParameterValue, ...] = Field(
        default=(),
        max_length=1_024,
    )

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.prompt is None and not self.references:
            raise ValueError("geometry authoring requires text or reference artifacts")
        _validate_unique_request_values(
            references=self.references,
            formats=self.requested_formats,
            parameters=self.parameters,
        )
        return self


class GeometryAuthoringRevisionRequest(_FrozenModel):
    """Exact correction request bound to one immutable source bundle."""

    schema_version: Literal["content-agent-workflows.geometry-authoring-revision-request.v1"] = (
        GEOMETRY_AUTHORING_REVISION_REQUEST_SCHEMA_VERSION
    )
    request_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    source_bundle: GeometrySourceBundle
    instructions: str | None = Field(default=None, min_length=1, max_length=32_768)
    references: tuple[GeometryAuthoringReference, ...] = Field(
        default=(),
        max_length=32,
    )
    parameter_overrides: tuple[GeometryAuthoringParameterValue, ...] = Field(
        default=(),
        max_length=1_024,
    )
    requested_formats: tuple[str, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.instructions is None and not self.references and not self.parameter_overrides:
            raise ValueError("geometry revision requires a requested change")
        _validate_unique_request_values(
            references=self.references,
            formats=self.requested_formats,
            parameters=self.parameter_overrides,
        )
        return self


class GeometryAuthoringVariant(_FrozenModel):
    """One named semantic row materialized from a shared immutable source."""

    variant_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    parameter_overrides: tuple[GeometryAuthoringParameterValue, ...] = Field(
        min_length=1,
        max_length=1_024,
    )
    instructions: str | None = Field(default=None, min_length=1, max_length=32_768)

    @model_validator(mode="after")
    def validate_variant(self) -> Self:
        names = tuple(item.name for item in self.parameter_overrides)
        if len(names) != len(set(names)):
            raise ValueError("variant parameter names must be unique")
        return self


class GeometryAuthoringFamilyRequest(_FrozenModel):
    """Bounded set of independent variants from one provider-owned revision."""

    schema_version: Literal["content-agent-workflows.geometry-authoring-family-request.v1"] = (
        GEOMETRY_AUTHORING_FAMILY_REQUEST_SCHEMA_VERSION
    )
    request_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    source_bundle: GeometrySourceBundle
    variants: tuple[GeometryAuthoringVariant, ...] = Field(min_length=1, max_length=64)
    requested_formats: tuple[str, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        variant_ids = tuple(item.variant_id for item in self.variants)
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("family variant IDs must be unique")
        if not self.source_bundle.parameters:
            raise ValueError("parameter families require declared source parameters")
        for variant in self.variants:
            resolve_semantic_parameter_overrides(
                self.source_bundle.parameters,
                variant.parameter_overrides,
            )
        _validate_unique_formats(self.requested_formats)
        return self


class GeometryAuthoringExportRequest(_FrozenModel):
    """Exact export request bound to one immutable source bundle."""

    schema_version: Literal["content-agent-workflows.geometry-authoring-export-request.v1"] = (
        GEOMETRY_AUTHORING_EXPORT_REQUEST_SCHEMA_VERSION
    )
    request_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    source_bundle: GeometrySourceBundle
    requested_formats: tuple[str, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _validate_unique_formats(self.requested_formats)
        return self


class GeometryAuthoringCapabilityReport(_FrozenModel):
    """Capabilities of one provider; never an implicit provider-selection hint."""

    schema_version: Literal["content-agent-workflows.geometry-authoring-capabilities.v1"] = (
        GEOMETRY_AUTHORING_CAPABILITY_SCHEMA_VERSION
    )
    provider: GeometryAuthoringProviderIdentity
    operations: tuple[Literal["generate", "revise", "export"], ...] = Field(
        min_length=1,
        max_length=3,
    )
    input_modalities: tuple[GeometryInputModality, ...] = Field(
        min_length=1,
        max_length=4,
    )
    output_formats: tuple[str, ...] = Field(min_length=1, max_length=64)
    supports_semantic_parameters: bool
    returns_native_source: bool
    features: tuple[GeometryAuthoringFeature, ...] = Field(default=(), max_length=16)
    max_family_variants: int = Field(default=0, ge=0, le=64)
    max_reference_artifacts: int = Field(ge=0, le=1_024)
    max_prompt_characters: int = Field(ge=0, le=1_000_000)

    @model_validator(mode="after")
    def validate_capabilities(self) -> Self:
        for label, values in (
            ("operations", self.operations),
            ("input modalities", self.input_modalities),
            ("output formats", self.output_formats),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"provider {label} must be unique")
        _validate_unique_formats(self.output_formats)
        if len(self.features) != len(set(self.features)):
            raise ValueError("provider authoring features must be unique")
        feature_set = set(self.features)
        if "semantic_parameters" in feature_set and not self.supports_semantic_parameters:
            raise ValueError("semantic_parameters feature requires semantic parameter support")
        if "parameter_definitions" in feature_set and "semantic_parameters" not in feature_set:
            raise ValueError("parameter_definitions feature requires semantic_parameters")
        if "parameter_families" in feature_set:
            if "revise" not in self.operations or "parameter_definitions" not in feature_set:
                raise ValueError("parameter_families requires revision and parameter definitions")
            if self.max_family_variants < 1:
                raise ValueError("parameter_families requires a positive family limit")
        elif self.max_family_variants:
            raise ValueError("max_family_variants requires parameter_families")
        if "native_source" in feature_set and not self.returns_native_source:
            raise ValueError("native_source feature requires returns_native_source")
        return self


class GeometryAuthoringFailure(_FrozenModel):
    """Typed, provider-neutral failure for one explicit operation."""

    schema_version: Literal["content-agent-workflows.geometry-authoring-failure.v1"] = (
        GEOMETRY_AUTHORING_FAILURE_SCHEMA_VERSION
    )
    provider_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    operation: GeometryAuthoringOperation
    code: GeometryFailureCode
    summary: str = Field(min_length=1, max_length=4_096)
    retryable: bool = False
    drifted_artifacts: tuple[str, ...] = Field(default=(), max_length=1_024)

    @model_validator(mode="after")
    def validate_failure(self) -> Self:
        if self.code == "input_drift" and not self.drifted_artifacts:
            raise ValueError("input drift must identify changed artifacts")
        if self.code != "input_drift" and self.drifted_artifacts:
            raise ValueError("only input drift may identify changed artifacts")
        if len(self.drifted_artifacts) != len(set(self.drifted_artifacts)):
            raise ValueError("drifted artifact paths must be unique")
        return self


class GeometryAuthoringProviderReceipt(_FrozenModel):
    """Terminal response from one explicitly selected authoring operation."""

    schema_version: Literal["content-agent-workflows.geometry-authoring-receipt.v1"] = (
        GEOMETRY_AUTHORING_RECEIPT_SCHEMA_VERSION
    )
    provider: GeometryAuthoringProviderIdentity
    operation: Literal["generate", "revise", "export"]
    request_digest: str = Field(pattern=_SHA256_PATTERN)
    disposition: Literal["succeeded", "failed"]
    source_bundle: GeometrySourceBundle | None = None
    failure: GeometryAuthoringFailure | None = None
    fallback_used: Literal[False] = False
    source_executed_by_geometry_agent: Literal[False] = False

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if self.disposition == "succeeded":
            if self.source_bundle is None or self.failure is not None:
                raise ValueError("successful authoring receipts require only a bundle")
            if self.source_bundle.producer != self.provider:
                raise ValueError("source bundle producer differs from receipt provider")
        else:
            if self.failure is None or self.source_bundle is not None:
                raise ValueError("failed authoring receipts require only a failure")
            if (
                self.failure.provider_id != self.provider.provider_id
                or self.failure.operation != self.operation
            ):
                raise ValueError("authoring failure identity differs from receipt")
        return self


class GeometryAuthoringProviderError(RuntimeError):
    """One selected provider failed; callers must select any replacement."""

    def __init__(
        self,
        failure: GeometryAuthoringFailure,
        *,
        receipt: GeometryAuthoringProviderReceipt | None = None,
    ) -> None:
        super().__init__(failure.summary)
        self.failure = failure
        self.receipt = receipt


class GeometryAuthoringInputDrift(GeometryAuthoringProviderError):
    """A digest-bound provider input changed around an invocation."""


class GeometryAuthoringInvalidResponse(GeometryAuthoringProviderError):
    """Provider bytes or output artifacts did not satisfy the public contract."""


@runtime_checkable
class GeometryAuthoringProvider(Protocol):
    """One explicitly selected provider; this contract never chooses a fallback."""

    @property
    def provider_id(self) -> str: ...

    def capabilities(self) -> GeometryAuthoringCapabilityReport: ...

    def generate(
        self,
        request: GeometryAuthoringRequest,
    ) -> GeometryAuthoringProviderReceipt: ...

    def revise(
        self,
        request: GeometryAuthoringRevisionRequest,
    ) -> GeometryAuthoringProviderReceipt: ...

    def export(
        self,
        request: GeometryAuthoringExportRequest,
    ) -> GeometryAuthoringProviderReceipt: ...


type GeometryAuthoringProviderRequest = (
    GeometryAuthoringRequest | GeometryAuthoringRevisionRequest | GeometryAuthoringExportRequest
)
type GeometryAuthoringRequestContract = (
    GeometryAuthoringProviderRequest | GeometryAuthoringFamilyRequest
)


def _validate_unique_formats(formats: tuple[str, ...]) -> None:
    if len(formats) != len(set(formats)):
        raise ValueError("requested formats must be unique")
    for value in formats:
        if not value or len(value) > 64:
            raise ValueError("requested formats must contain bounded values")
        if value.lower() != value:
            raise ValueError("requested formats must be lowercase")
        if re.fullmatch(_FORMAT_PATTERN, value) is None:
            raise ValueError("requested format is invalid")


def _validate_unique_request_values(
    *,
    references: tuple[GeometryAuthoringReference, ...],
    formats: tuple[str, ...],
    parameters: tuple[GeometryAuthoringParameterValue, ...],
) -> None:
    reference_ids = tuple(item.reference_id for item in references)
    reference_paths = tuple(item.artifact.path for item in references)
    parameter_names = tuple(item.name for item in parameters)
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("authoring reference IDs must be unique")
    if len(reference_paths) != len(set(reference_paths)):
        raise ValueError("authoring reference paths must be unique")
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError("authoring parameter names must be unique")
    _validate_unique_formats(formats)


def _validate_credential_free_url(
    value: str,
    *,
    label: str,
    allow_loopback_http: bool,
) -> str:
    normalized = value.strip().rstrip("/")
    if not normalized or any(character.isspace() for character in normalized):
        raise ValueError(f"{label} must be a credential-free HTTP URL")
    try:
        parsed = urlparse(normalized)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} must be a credential-free HTTP URL") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{label} must be a credential-free HTTP URL")
    if parsed.scheme == "http" and (
        not allow_loopback_http or not _is_loopback_host(parsed.hostname)
    ):
        raise ValueError(f"non-loopback {label}s require HTTPS transport")
    return normalized


def _is_loopback_host(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _model_digest(model: BaseModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def geometry_authoring_request_digest(
    request: GeometryAuthoringRequestContract,
) -> str:
    """Return the canonical digest a provider must bind into its receipt."""

    return _model_digest(request)


def geometry_source_bundle_id(
    *,
    provider: GeometryAuthoringProviderIdentity,
    source_revision: str,
    coordinate_system: GeometryCoordinateSystem,
    representations: tuple[GeometryRepresentationBinding, ...],
    parts: tuple[GeometryPartBinding, ...] = (),
    parameters: tuple[GeometrySemanticParameter, ...] = (),
    verification_assertions: tuple[GeometryVerificationAssertion, ...] = (),
    provenance: GeometrySourceProvenance,
    rights: GeometryRightsAssertion,
) -> str:
    """Return the canonical identity for one provider source revision.

    Artifact paths are intentionally excluded. A provider may materialize the
    same immutable bytes under a different trusted root without changing the
    source identity.
    """

    payload = {
        "coordinate_system": coordinate_system.model_dump(mode="json"),
        "parameters": [_semantic_parameter_identity(item) for item in parameters],
        "parts": [item.model_dump(mode="json") for item in parts],
        "producer": provider.model_dump(mode="json"),
        "provenance": {
            **provenance.model_dump(mode="json", exclude={"input_artifacts"}),
            "input_artifacts": [
                {
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in provenance.input_artifacts
            ],
        },
        "representations": [
            {
                "format": item.format,
                "media_type": item.media_type,
                "representation_id": item.representation_id,
                "role": item.role,
                "root_prim_path": item.root_prim_path,
                "sha256": item.artifact.sha256,
                "size_bytes": item.artifact.size_bytes,
            }
            for item in representations
        ],
        "rights": rights.model_dump(mode="json"),
        "source_revision": source_revision,
        "verification_assertions": [
            item.model_dump(mode="json") for item in verification_assertions
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _semantic_parameter_identity(
    parameter: GeometrySemanticParameter,
) -> dict[str, Any]:
    """Bind rich semantics while preserving legacy v1 identities by default."""

    payload: dict[str, Any] = {
        "name": parameter.name,
        "value": parameter.value,
        "unit": parameter.unit,
        "minimum": parameter.minimum,
        "maximum": parameter.maximum,
        "description": parameter.description,
    }
    inferred_type = _parameter_value_type(parameter.value)
    if parameter.value_type != inferred_type or parameter.choices:
        payload["value_type"] = parameter.value_type
    if parameter.step is not None:
        payload["step"] = parameter.step
    if parameter.choices:
        payload["choices"] = parameter.choices
    if parameter.group is not None:
        payload["group"] = parameter.group
    if parameter.semantic_role is not None:
        payload["semantic_role"] = parameter.semantic_role
    if parameter.effects:
        payload["effects"] = parameter.effects
    if parameter.affects:
        payload["affects"] = parameter.affects
    if not parameter.visible:
        payload["visible"] = False
    return payload


def validate_geometry_source_bundle_identity(
    bundle: GeometrySourceBundle,
) -> GeometrySourceBundle:
    """Reject a source bundle whose declared identity is not canonical."""

    expected = geometry_source_bundle_id(
        provider=bundle.producer,
        source_revision=bundle.source_revision,
        coordinate_system=bundle.coordinate_system,
        representations=bundle.representations,
        parts=bundle.parts,
        parameters=bundle.parameters,
        verification_assertions=bundle.verification_assertions,
        provenance=bundle.provenance,
        rights=bundle.rights,
    )
    if bundle.bundle_id != expected:
        raise ValueError("geometry source bundle identity is not canonical")
    return bundle


def _capture_binding(path: str | Path, *, label: str) -> GeometryArtifactBinding:
    expanded = Path(path).expanduser()
    absolute = Path(os.path.abspath(expanded))
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is not a safe regular file: {absolute}") from exc
    if resolved != absolute:
        raise ValueError(f"{label} traverses a symlink: {absolute}")
    try:
        captured = _read_regular_artifact(resolved.parent, resolved.name)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is not a safe regular file: {resolved}") from exc
    return GeometryArtifactBinding(
        path=str(captured.path),
        sha256=captured.sha256,
        size_bytes=captured.size_bytes,
    )


def _trusted_artifact_root(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    absolute = Path(os.path.abspath(expanded))
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"trusted provider artifact root does not exist: {absolute}") from exc
    if resolved != absolute:
        raise ValueError(f"trusted provider artifact root traverses a symlink: {absolute}")
    if not resolved.is_dir():
        raise ValueError(f"trusted provider artifact root is not a directory: {resolved}")
    return resolved


def _binding_is_within_roots(
    binding: GeometryArtifactBinding,
    roots: tuple[Path, ...],
) -> bool:
    absolute = Path(os.path.abspath(Path(binding.path).expanduser()))
    return any(absolute.is_relative_to(root) for root in roots)


def _drifted_paths(bindings: tuple[GeometryArtifactBinding, ...]) -> tuple[str, ...]:
    drifted: list[str] = []
    seen: set[str] = set()
    for binding in bindings:
        if binding.path in seen:
            continue
        seen.add(binding.path)
        try:
            observed = _capture_binding(binding.path, label="bound provider artifact")
        except ValueError:
            drifted.append(binding.path)
            continue
        if observed != binding:
            drifted.append(binding.path)
    return tuple(drifted)


def _failure(
    *,
    provider_id: str,
    operation: GeometryAuthoringOperation,
    code: GeometryFailureCode,
    summary: str,
    retryable: bool = False,
    drifted_artifacts: tuple[str, ...] = (),
) -> GeometryAuthoringFailure:
    return GeometryAuthoringFailure(
        provider_id=provider_id,
        operation=operation,
        code=code,
        summary=summary,
        retryable=retryable,
        drifted_artifacts=drifted_artifacts,
    )


def _raise_input_drift(
    *,
    provider_id: str,
    operation: GeometryAuthoringOperation,
    bindings: tuple[GeometryArtifactBinding, ...],
) -> None:
    drifted = _drifted_paths(bindings)
    if drifted:
        raise GeometryAuthoringInputDrift(
            _failure(
                provider_id=provider_id,
                operation=operation,
                code="input_drift",
                summary="Digest-bound geometry authoring inputs changed.",
                drifted_artifacts=drifted,
            )
        )


def _read_bound_model[ModelT: BaseModel](
    binding: GeometryArtifactBinding,
    model_type: type[ModelT],
    *,
    provider_id: str,
    operation: GeometryAuthoringOperation,
) -> ModelT:
    _raise_input_drift(
        provider_id=provider_id,
        operation=operation,
        bindings=(binding,),
    )
    if binding.size_bytes > _MAX_PROVIDER_JSON_BYTES:
        raise GeometryAuthoringInvalidResponse(
            _failure(
                provider_id=provider_id,
                operation=operation,
                code="invalid_response",
                summary="Selected geometry authoring provider returned oversized JSON.",
            )
        )
    try:
        captured = _read_regular_artifact(
            Path(binding.path).parent,
            Path(binding.path).name,
            max_bytes=_MAX_PROVIDER_JSON_BYTES,
            capture_bytes=True,
        )
    except (OSError, ValueError) as exc:
        raise GeometryAuthoringInputDrift(
            _failure(
                provider_id=provider_id,
                operation=operation,
                code="input_drift",
                summary="Digest-bound provider JSON changed while it was read.",
                drifted_artifacts=(binding.path,),
            )
        ) from exc
    observed = GeometryArtifactBinding(
        path=str(captured.path),
        sha256=captured.sha256,
        size_bytes=captured.size_bytes,
    )
    if observed != binding or captured.data is None:
        raise GeometryAuthoringInputDrift(
            _failure(
                provider_id=provider_id,
                operation=operation,
                code="input_drift",
                summary="Digest-bound provider JSON changed while it was read.",
                drifted_artifacts=(binding.path,),
            )
        )
    try:
        return model_type.model_validate_json(captured.data)
    except ValidationError as exc:
        raise GeometryAuthoringInvalidResponse(
            _failure(
                provider_id=provider_id,
                operation=operation,
                code="invalid_response",
                summary="Selected geometry authoring provider returned invalid typed JSON.",
            )
        ) from exc


def _bundle_bindings(
    bundle: GeometrySourceBundle,
) -> tuple[GeometryArtifactBinding, ...]:
    return (
        *(item.artifact for item in bundle.representations),
        *bundle.provenance.input_artifacts,
    )


def _request_bindings(
    request: (
        GeometryAuthoringRequest | GeometryAuthoringRevisionRequest | GeometryAuthoringExportRequest
    ),
) -> tuple[GeometryArtifactBinding, ...]:
    if isinstance(request, GeometryAuthoringRequest):
        return tuple(item.artifact for item in request.references)
    if isinstance(request, GeometryAuthoringRevisionRequest):
        return (
            *_bundle_bindings(request.source_bundle),
            *(item.artifact for item in request.references),
        )
    return _bundle_bindings(request.source_bundle)


class _ExplicitGeometryAuthoringProviderAdapter:
    adapter_id: Literal["artifact-json", "http-json"]

    def __init__(
        self,
        *,
        provider_id: str,
        artifact_roots: tuple[str | Path, ...],
    ) -> None:
        provider_id = provider_id.strip()
        if not provider_id:
            raise ValueError("geometry authoring provider ID must not be empty")
        GeometryAuthoringProviderIdentity(
            provider_id=provider_id,
            provider_version="configuration-validation",
        )
        self._provider_id = provider_id
        self._artifact_roots = tuple(
            dict.fromkeys(_trusted_artifact_root(path) for path in artifact_roots)
        )

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def _validate_capabilities(
        self,
        capabilities: GeometryAuthoringCapabilityReport,
    ) -> GeometryAuthoringCapabilityReport:
        if capabilities.provider.provider_id != self.provider_id:
            raise GeometryAuthoringInvalidResponse(
                _failure(
                    provider_id=self.provider_id,
                    operation="capabilities",
                    code="invalid_response",
                    summary="Capability provider identity differs from configuration.",
                )
            )
        return capabilities

    def _validate_receipt(
        self,
        receipt: GeometryAuthoringProviderReceipt,
        *,
        operation: Literal["generate", "revise", "export"],
        request: GeometryAuthoringProviderRequest,
    ) -> GeometryAuthoringProviderReceipt:
        if (
            receipt.provider.provider_id != self.provider_id
            or receipt.operation != operation
            or receipt.request_digest != geometry_authoring_request_digest(request)
        ):
            raise GeometryAuthoringInvalidResponse(
                _failure(
                    provider_id=self.provider_id,
                    operation=operation,
                    code="invalid_response",
                    summary="Provider receipt differs from the exact selected request.",
                )
            )
        if receipt.disposition == "failed":
            assert receipt.failure is not None
            raise GeometryAuthoringProviderError(receipt.failure, receipt=receipt)
        assert receipt.source_bundle is not None
        source = receipt.source_bundle
        try:
            validate_geometry_source_bundle_identity(source)
        except ValueError as exc:
            raise GeometryAuthoringInvalidResponse(
                _failure(
                    provider_id=self.provider_id,
                    operation=operation,
                    code="invalid_response",
                    summary="Provider bundle identity is not canonical.",
                )
            ) from exc
        request_digest = geometry_authoring_request_digest(request)
        requested_formats = tuple(getattr(request, "requested_formats", ()))
        available_formats = geometry_deliverable_representation_formats(source.representations)
        lineage_is_valid = source.provenance.request_digest == request_digest
        if isinstance(request, GeometryAuthoringRequest):
            lineage_is_valid = lineage_is_valid and source.provenance.parent_bundle_id is None
        else:
            lineage_is_valid = (
                lineage_is_valid
                and source.provenance.parent_bundle_id == request.source_bundle.bundle_id
            )
        if not lineage_is_valid or not geometry_requested_formats_satisfied(
            requested_formats,
            available_formats,
        ):
            raise GeometryAuthoringInvalidResponse(
                _failure(
                    provider_id=self.provider_id,
                    operation=operation,
                    code="invalid_response",
                    summary="Provider bundle lineage or requested exports are invalid.",
                )
            )
        try:
            if isinstance(request, GeometryAuthoringRequest):
                validate_returned_parameter_values(source.parameters, request.parameters)
            elif isinstance(request, GeometryAuthoringRevisionRequest):
                if source.source_revision == request.source_bundle.source_revision:
                    raise ValueError("revision reused the immutable source revision")
                expected_parameters = resolve_semantic_parameter_overrides(
                    request.source_bundle.parameters,
                    request.parameter_overrides,
                    allow_undeclared=not request.source_bundle.parameters,
                )
                validate_returned_parameter_state(source.parameters, expected_parameters)
            else:
                if source.source_revision != request.source_bundle.source_revision:
                    raise ValueError("export changed the immutable source revision")
                validate_returned_parameter_state(
                    source.parameters,
                    request.source_bundle.parameters,
                )
        except ValueError as exc:
            raise GeometryAuthoringInvalidResponse(
                _failure(
                    provider_id=self.provider_id,
                    operation=operation,
                    code="invalid_response",
                    summary="Provider violated immutable revision or parameter state.",
                )
            ) from exc
        output_bindings = tuple(item.artifact for item in source.representations)
        if not self._artifact_roots or not all(
            _binding_is_within_roots(binding, self._artifact_roots) for binding in output_bindings
        ):
            raise GeometryAuthoringInvalidResponse(
                _failure(
                    provider_id=self.provider_id,
                    operation=operation,
                    code="artifact_mismatch",
                    summary="Provider output escaped its trusted artifact roots.",
                )
            )
        request_artifacts = set(_request_bindings(request))
        if not set(source.provenance.input_artifacts) <= request_artifacts:
            raise GeometryAuthoringInvalidResponse(
                _failure(
                    provider_id=self.provider_id,
                    operation=operation,
                    code="artifact_mismatch",
                    summary="Provider provenance introduced an unbound input artifact.",
                )
            )
        output_drift = _drifted_paths(_bundle_bindings(receipt.source_bundle))
        if output_drift:
            raise GeometryAuthoringInvalidResponse(
                _failure(
                    provider_id=self.provider_id,
                    operation=operation,
                    code="artifact_mismatch",
                    summary="Provider output artifacts do not match their bindings.",
                )
            )
        return receipt


class ArtifactJsonGeometryAuthoringProvider(_ExplicitGeometryAuthoringProviderAdapter):
    """Read exact, pre-produced JSON responses for one explicit provider."""

    adapter_id: Literal["artifact-json"] = "artifact-json"

    def __init__(
        self,
        *,
        provider_id: str,
        capabilities_path: str | Path,
        generate_receipt_path: str | Path | None = None,
        revise_receipt_path: str | Path | None = None,
        export_receipt_path: str | Path | None = None,
        artifact_roots: tuple[str | Path, ...] = (),
    ) -> None:
        self._capabilities = _capture_binding(
            capabilities_path,
            label="geometry authoring capability payload",
        )
        paths = {
            "generate": generate_receipt_path,
            "revise": revise_receipt_path,
            "export": export_receipt_path,
        }
        self._receipts = {
            operation: _capture_binding(
                path,
                label=f"geometry authoring {operation} receipt",
            )
            for operation, path in paths.items()
            if path is not None
        }
        derived_roots = tuple(
            dict.fromkeys(
                (
                    *(Path(binding.path).parent for binding in self._receipts.values()),
                    *artifact_roots,
                )
            )
        )
        super().__init__(provider_id=provider_id, artifact_roots=derived_roots)

    def capabilities(self) -> GeometryAuthoringCapabilityReport:
        result = _read_bound_model(
            self._capabilities,
            GeometryAuthoringCapabilityReport,
            provider_id=self.provider_id,
            operation="capabilities",
        )
        _raise_input_drift(
            provider_id=self.provider_id,
            operation="capabilities",
            bindings=(self._capabilities,),
        )
        return self._validate_capabilities(result)

    def generate(
        self,
        request: GeometryAuthoringRequest,
    ) -> GeometryAuthoringProviderReceipt:
        return self._invoke("generate", request)

    def revise(
        self,
        request: GeometryAuthoringRevisionRequest,
    ) -> GeometryAuthoringProviderReceipt:
        return self._invoke("revise", request)

    def export(
        self,
        request: GeometryAuthoringExportRequest,
    ) -> GeometryAuthoringProviderReceipt:
        return self._invoke("export", request)

    def _invoke(
        self,
        operation: Literal["generate", "revise", "export"],
        request: GeometryAuthoringProviderRequest,
    ) -> GeometryAuthoringProviderReceipt:
        response_binding = self._receipts.get(operation)
        if response_binding is None:
            raise GeometryAuthoringProviderError(
                _failure(
                    provider_id=self.provider_id,
                    operation=operation,
                    code="unsupported_operation",
                    summary=(
                        f"Selected artifact provider has no {operation} receipt; "
                        "no fallback was attempted."
                    ),
                )
            )
        inputs = (*_request_bindings(request), response_binding)
        _raise_input_drift(
            provider_id=self.provider_id,
            operation=operation,
            bindings=inputs,
        )
        receipt = _read_bound_model(
            response_binding,
            GeometryAuthoringProviderReceipt,
            provider_id=self.provider_id,
            operation=operation,
        )
        _raise_input_drift(
            provider_id=self.provider_id,
            operation=operation,
            bindings=inputs,
        )
        result = self._validate_receipt(
            receipt,
            operation=operation,
            request=request,
        )
        _raise_input_drift(
            provider_id=self.provider_id,
            operation=operation,
            bindings=inputs,
        )
        return result


class HttpJsonGeometryAuthoringProvider(_ExplicitGeometryAuthoringProviderAdapter):
    """Call one explicit remote provider through bounded JSON endpoints."""

    adapter_id: Literal["http-json"] = "http-json"

    def __init__(
        self,
        *,
        provider_id: str,
        endpoint_alias: str,
        endpoint_url: str,
        bearer_token: str | None = None,
        timeout_seconds: float = 120.0,
        artifact_roots: tuple[str | Path, ...] = (),
        session: Any | None = None,
    ) -> None:
        super().__init__(provider_id=provider_id, artifact_roots=artifact_roots)
        alias = endpoint_alias.strip()
        if not alias:
            raise ValueError("geometry authoring endpoint alias must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("geometry authoring provider timeout must be positive")
        if bearer_token is not None and (
            not bearer_token.strip()
            or not bearer_token.isascii()
            or any(character.isspace() for character in bearer_token)
            or any(ord(character) < 32 or ord(character) == 127 for character in bearer_token)
        ):
            raise ValueError("geometry authoring bearer token must be a non-empty single value")
        self._endpoint_alias = alias
        self._endpoint_url = _validate_credential_free_url(
            endpoint_url,
            label="geometry authoring provider URL",
            allow_loopback_http=True,
        )
        self._bearer_token = bearer_token
        self._timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        if session is None:
            self._session.trust_env = False

    def capabilities(self) -> GeometryAuthoringCapabilityReport:
        result = self._request_model(
            operation="capabilities",
            method="get",
            model_type=GeometryAuthoringCapabilityReport,
        )
        return self._validate_capabilities(result)

    def generate(
        self,
        request: GeometryAuthoringRequest,
    ) -> GeometryAuthoringProviderReceipt:
        return self._invoke("generate", request)

    def revise(
        self,
        request: GeometryAuthoringRevisionRequest,
    ) -> GeometryAuthoringProviderReceipt:
        return self._invoke("revise", request)

    def export(
        self,
        request: GeometryAuthoringExportRequest,
    ) -> GeometryAuthoringProviderReceipt:
        return self._invoke("export", request)

    def _invoke(
        self,
        operation: Literal["generate", "revise", "export"],
        request: GeometryAuthoringProviderRequest,
    ) -> GeometryAuthoringProviderReceipt:
        inputs = _request_bindings(request)
        _raise_input_drift(
            provider_id=self.provider_id,
            operation=operation,
            bindings=inputs,
        )
        pending_error: GeometryAuthoringProviderError | None = None
        receipt: GeometryAuthoringProviderReceipt | None = None
        try:
            receipt = self._request_model(
                operation=operation,
                method="post",
                model_type=GeometryAuthoringProviderReceipt,
                payload=request,
            )
        except GeometryAuthoringProviderError as exc:
            pending_error = exc
        try:
            _raise_input_drift(
                provider_id=self.provider_id,
                operation=operation,
                bindings=inputs,
            )
        except GeometryAuthoringInputDrift as drift:
            raise drift from pending_error
        if pending_error is not None:
            raise pending_error
        assert receipt is not None
        result = self._validate_receipt(
            receipt,
            operation=operation,
            request=request,
        )
        _raise_input_drift(
            provider_id=self.provider_id,
            operation=operation,
            bindings=inputs,
        )
        return result

    def _request_model[ModelT: BaseModel](
        self,
        *,
        operation: GeometryAuthoringOperation,
        method: Literal["get", "post"],
        model_type: type[ModelT],
        payload: BaseModel | None = None,
    ) -> ModelT:
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self._bearer_token is not None:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        kwargs: dict[str, Any] = {
            "headers": headers,
            "timeout": self._timeout_seconds,
            "allow_redirects": False,
            "stream": True,
        }
        if payload is not None:
            kwargs["json"] = payload.model_dump(mode="json")
        try:
            call = getattr(self._session, method)
            response = call(f"{self._endpoint_url}/{operation}", **kwargs)
        except Exception as exc:
            raise GeometryAuthoringProviderError(
                _failure(
                    provider_id=self.provider_id,
                    operation=operation,
                    code="transport_error",
                    summary=(
                        "Selected geometry authoring provider failed at transport; "
                        "no fallback was attempted."
                    ),
                    retryable=True,
                )
            ) from exc
        try:
            try:
                status_code = getattr(response, "status_code", None)
                if not isinstance(status_code, int):
                    raise GeometryAuthoringInvalidResponse(
                        _failure(
                            provider_id=self.provider_id,
                            operation=operation,
                            code="invalid_response",
                            summary="Provider returned an invalid HTTP status.",
                        )
                    )
                if status_code != 200:
                    raise GeometryAuthoringProviderError(
                        _failure(
                            provider_id=self.provider_id,
                            operation=operation,
                            code="http_error",
                            summary=(
                                "Selected geometry authoring provider failed without "
                                f"fallback: {self._endpoint_alias} returned HTTP "
                                f"{status_code}."
                            ),
                            retryable=(status_code in {408, 425, 429} or status_code >= 500),
                        )
                    )
                headers_value = getattr(response, "headers", {})
                if not isinstance(headers_value, Mapping):
                    raise GeometryAuthoringInvalidResponse(
                        _failure(
                            provider_id=self.provider_id,
                            operation=operation,
                            code="invalid_response",
                            summary="Provider returned invalid HTTP headers.",
                        )
                    )
                content_length = headers_value.get("Content-Length")
                try:
                    parsed_length = int(content_length) if content_length is not None else None
                except (TypeError, ValueError) as exc:
                    raise GeometryAuthoringInvalidResponse(
                        _failure(
                            provider_id=self.provider_id,
                            operation=operation,
                            code="invalid_response",
                            summary="Provider returned an invalid Content-Length.",
                        )
                    ) from exc
                if parsed_length is not None and (
                    parsed_length < 0 or parsed_length > _MAX_PROVIDER_JSON_BYTES
                ):
                    raise GeometryAuthoringInvalidResponse(
                        _failure(
                            provider_id=self.provider_id,
                            operation=operation,
                            code="invalid_response",
                            summary="Provider returned oversized JSON.",
                        )
                    )
                document = bytearray()
                try:
                    for chunk in response.iter_content(chunk_size=_HTTP_CHUNK_SIZE):
                        if not chunk:
                            continue
                        document.extend(chunk)
                        if len(document) > _MAX_PROVIDER_JSON_BYTES:
                            raise GeometryAuthoringInvalidResponse(
                                _failure(
                                    provider_id=self.provider_id,
                                    operation=operation,
                                    code="invalid_response",
                                    summary="Provider returned oversized JSON.",
                                )
                            )
                except GeometryAuthoringProviderError:
                    raise
                except Exception as exc:
                    raise GeometryAuthoringProviderError(
                        _failure(
                            provider_id=self.provider_id,
                            operation=operation,
                            code="transport_error",
                            summary=(
                                "Selected geometry authoring provider response failed "
                                "while streaming; no fallback was attempted."
                            ),
                            retryable=True,
                        )
                    ) from exc
                try:
                    return model_type.model_validate_json(document)
                except ValidationError as exc:
                    raise GeometryAuthoringInvalidResponse(
                        _failure(
                            provider_id=self.provider_id,
                            operation=operation,
                            code="invalid_response",
                            summary="Provider returned invalid typed JSON.",
                        )
                    ) from exc
            except GeometryAuthoringProviderError:
                raise
            except Exception as exc:
                raise GeometryAuthoringInvalidResponse(
                    _failure(
                        provider_id=self.provider_id,
                        operation=operation,
                        code="invalid_response",
                        summary="Provider returned a malformed HTTP response.",
                    )
                ) from exc
        finally:
            try:
                response.close()
            except Exception:
                pass


__all__ = [
    "GEOMETRY_AUTHORING_IDENTIFIER_PATTERN",
    "GEOMETRY_AUTHORING_CAPABILITY_SCHEMA_VERSION",
    "GEOMETRY_AUTHORING_EXPORT_REQUEST_SCHEMA_VERSION",
    "GEOMETRY_AUTHORING_FAMILY_REQUEST_SCHEMA_VERSION",
    "GEOMETRY_AUTHORING_FAILURE_SCHEMA_VERSION",
    "GEOMETRY_AUTHORING_RECEIPT_SCHEMA_VERSION",
    "GEOMETRY_AUTHORING_REQUEST_SCHEMA_VERSION",
    "GEOMETRY_AUTHORING_REVISION_REQUEST_SCHEMA_VERSION",
    "GEOMETRY_SOURCE_SCHEMA_VERSION",
    "ArtifactJsonGeometryAuthoringProvider",
    "GeometryArtifactBinding",
    "GeometryAuthoringCapabilityReport",
    "GeometryAuthoringExportRequest",
    "GeometryAuthoringFamilyRequest",
    "GeometryAuthoringFeature",
    "GeometryAuthoringFailure",
    "GeometryAuthoringInputDrift",
    "GeometryAuthoringInvalidResponse",
    "GeometryAuthoringOperation",
    "GeometryAuthoringParameterValue",
    "GeometryAuthoringProvider",
    "GeometryAuthoringProviderError",
    "GeometryAuthoringProviderIdentity",
    "GeometryAuthoringProviderRequest",
    "GeometryAuthoringProviderReceipt",
    "GeometryAuthoringReference",
    "GeometryAuthoringRequest",
    "GeometryAuthoringRequestContract",
    "GeometryAuthoringRevisionRequest",
    "GeometryAuthoringVariant",
    "GeometryCoordinateSystem",
    "GeometryFailureCode",
    "GeometryInputModality",
    "GeometryParameterEffect",
    "GeometryParameterType",
    "GeometryPartBinding",
    "GeometryRepresentationBinding",
    "GeometryRepresentationRole",
    "GeometryRightsAssertion",
    "GeometrySemanticParameter",
    "GeometrySourceBundle",
    "GeometrySourceProvenance",
    "GeometryVerificationAssertion",
    "HttpJsonGeometryAuthoringProvider",
    "geometry_authoring_request_digest",
    "geometry_deliverable_representation_formats",
    "geometry_requested_formats_satisfied",
    "geometry_source_bundle_id",
    "resolve_semantic_parameter_overrides",
    "validate_returned_parameter_values",
    "validate_returned_parameter_state",
    "validate_geometry_source_bundle_identity",
]
