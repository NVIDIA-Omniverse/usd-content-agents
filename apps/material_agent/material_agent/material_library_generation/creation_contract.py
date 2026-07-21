# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared contracts for geometry-conditioned material creation.

This module is deliberately orchestration-free.  It freezes the boundary shared by
agent policy, material-creation backends, packaging, validation, and workflow
integration without selecting a backend or changing assignment behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from material_agent.material_library_generation.schema import (
    MaterialRecipe,
    make_material_id,
)

MATERIAL_CREATION_SCHEMA_VERSION = "material-agent-create.v1"
MATERIAL_CREATION_MANIFEST_NAME = "material_creation_manifest.json"
MATERIAL_LIBRARY_NAME = "material_library.usda"
MATERIAL_LIST_MANIFEST_NAME = "materials.yaml"
_REQUEST_ID_RE = re.compile(r"^mc_[0-9a-f]{24}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_USD_SUFFIXES = frozenset((".usd", ".usda", ".usdc", ".usdz"))


class MaterialAction(StrEnum):
    """Actions available to Material Agent after material-list ranking."""

    ASSIGN_EXISTING = "assign_existing"
    CREATE_NEW = "create_new"
    MODIFY_EXISTING = "modify_existing"


class MaterialCreationMode(StrEnum):
    """Texture-space ownership of a created material."""

    ASSET_UV = "asset_uv"
    TILEABLE = "tileable"


class MaterialChannel(StrEnum):
    """Canonical texture outputs consumed by Material Agent packaging."""

    ALBEDO = "albedo"
    NORMAL = "normal"
    ORM = "orm"


class MaterialColorSpace(StrEnum):
    """Canonical texture colorspaces."""

    SRGB = "sRGB"
    RAW = "raw"


class MaterialChannelSource(StrEnum):
    """How a channel was produced."""

    MODEL_GENERATED = "model_generated"
    BUMP_TO_NORMAL = "generated_bump_conversion"
    DERIVED = "derived"
    SYNTHESIZED = "synthesized"
    PRESERVED_INPUT = "preserved_input"
    RECIPE_HINT = "recipe_hint"
    NEUTRAL_FALLBACK = "neutral_fallback"


class NormalConvention(StrEnum):
    """Normal-map convention accepted by the initial OpenPBR authoring path."""

    TANGENT_OPENGL = "tangent_opengl_positive_y"


class ORMPacking(StrEnum):
    """Canonical channel packing for occlusion, roughness, and metallic."""

    OCCLUSION_ROUGHNESS_METALLIC = "r_occlusion_g_roughness_b_metallic"


class MaterialChannelComponent(StrEnum):
    """Semantic components whose provenance must be tracked independently."""

    BASE_COLOR = "base_color"
    TANGENT_NORMAL = "tangent_normal"
    OCCLUSION = "occlusion"
    ROUGHNESS = "roughness"
    METALLIC = "metallic"


class MaterialConditioningKind(StrEnum):
    """Canonical geometry/reference inputs prepared once for all backends."""

    SCOPED_USD = "scoped_usd"
    UV_LAYOUT = "uv_layout"
    UV_MASK = "uv_mask"
    NORMAL = "normal"
    DEPTH = "depth"
    SEGMENTATION = "segmentation"
    RENDER = "render"
    RENDER_REQUEST = "render_request"
    MESH_ST = "mesh_st"
    SEED_MATERIAL = "seed_material"
    SOURCE_ALBEDO = "source_albedo"
    REFERENCE_IMAGE = "reference_image"


class MaterialConditioningArtifactSource(StrEnum):
    """Origin category for one prepared conditioning artifact."""

    PLACEHOLDER = "placeholder"
    SOURCE_DERIVED = "source_derived"
    RENDERER_DERIVED = "renderer_derived"
    RECIPE_REFERENCE = "recipe_reference"
    REQUEST_REFERENCE = "request_reference"


class MaterialDiagnosticSeverity(StrEnum):
    """Severity of a structured backend or packaging diagnostic."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class MaterialCreationErrorCode(StrEnum):
    """Stable error categories surfaced by every creation backend."""

    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_MATERIAL = "unsupported_material"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    MISSING_CHECKPOINT = "missing_checkpoint"
    BACKEND_FAILURE = "backend_failure"
    CUDA_OUT_OF_MEMORY = "cuda_out_of_memory"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PARTIAL_OUTPUT = "partial_output"
    INVALID_OUTPUT = "invalid_output"


class MaterialDegradationCode(StrEnum):
    """Stable categories for accepted, explicitly degraded output."""

    MISSING_ALBEDO = "missing_albedo"
    MISSING_NORMAL = "missing_normal"
    MISSING_ORM = "missing_orm"
    NEUTRAL_AO = "neutral_ao"
    BUMP_TO_NORMAL_FAILED = "bump_to_normal_failed"
    RECIPE_HINT_FALLBACK = "recipe_hint_fallback"
    REFERENCE_IGNORED = "reference_ignored"


@dataclass(frozen=True)
class MaterialArtifactLayout:
    """Canonical paths for one run-local created-material package.

    A backend writes normalized textures beneath ``textures/<material_id>``.
    WP2 owns authoring the USD library, list manifest, and creation manifest.
    """

    package_dir: Path
    material_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "package_dir", Path(self.package_dir))
        object.__setattr__(self, "material_id", make_material_id(self.material_id))

    @property
    def textures_dir(self) -> Path:
        return self.package_dir / "textures" / self.material_id

    @property
    def previews_dir(self) -> Path:
        return self.package_dir / "previews" / self.material_id

    @property
    def material_usd_path(self) -> Path:
        return self.package_dir / MATERIAL_LIBRARY_NAME

    @property
    def materials_manifest_path(self) -> Path:
        return self.package_dir / MATERIAL_LIST_MANIFEST_NAME

    @property
    def creation_manifest_path(self) -> Path:
        return self.package_dir / MATERIAL_CREATION_MANIFEST_NAME

    def texture_path(self, channel: MaterialChannel | str) -> Path:
        normalized = MaterialChannel(channel)
        return self.textures_dir / f"{normalized.value}.png"


def _non_empty(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _canonical_prim_path(value: str, *, field_name: str) -> str:
    path = _non_empty(value, field_name=field_name)
    if not path.startswith("/"):
        raise ValueError(f"{field_name} must be an absolute USD prim path: {path!r}")
    if path == "/":
        raise ValueError(f"{field_name} must identify a prim below the pseudo-root")
    if path.endswith("/") or "//" in path:
        raise ValueError(f"{field_name} is not a canonical USD prim path: {path!r}")
    return path


def _canonical_prim_paths(
    values: tuple[str, ...],
    *,
    field_name: str,
    require_non_empty: bool,
) -> tuple[str, ...]:
    paths = tuple(
        _canonical_prim_path(value, field_name=field_name) for value in values
    )
    if require_non_empty and not paths:
        raise ValueError(f"{field_name} must contain at least one USD prim path")
    if len(paths) != len(set(paths)):
        raise ValueError(f"{field_name} must not contain duplicate USD prim paths")
    return tuple(sorted(paths))


def intended_part_prim_path_hints(recipe: MaterialRecipe) -> tuple[str, ...]:
    """Return canonical planner hints without resolving or expanding USD scope.

    ``IntendedPart.prim_path_hints`` remain hints.  The existing assignment/scope
    resolver must turn them into authoritative target prim paths, which are then
    supplied to :class:`CreateMaterialRequest`.  Creation backends must not perform
    their own competing semantic-to-prim traversal.
    """

    paths = {
        _canonical_prim_path(path, field_name="intended_parts.prim_path_hints")
        for part in recipe.intended_parts
        for path in part.prim_path_hints
    }
    return tuple(sorted(paths))


def _effective_references(
    recipe: MaterialRecipe, explicit: tuple[str, ...]
) -> tuple[str, ...]:
    combined: list[str] = []
    seen: set[str] = set()
    for raw_uri in (*recipe.reference_image_uris, *explicit):
        uri = _non_empty(raw_uri, field_name="reference_image_uris")
        if uri not in seen:
            seen.add(uri)
            combined.append(uri)
    return tuple(combined)


@dataclass(frozen=True)
class CreateMaterialRequest:
    """Normalized, cache-addressable input to material creation."""

    source_usd: Path
    target_prim_paths: tuple[str, ...]
    recipe: MaterialRecipe
    reference_image_uris: tuple[str, ...] = ()
    creation_mode: MaterialCreationMode = MaterialCreationMode.ASSET_UV
    texture_size: int = 1024
    backend: str = "auto"
    seed: int | None = None
    source_usd_sha256: str | None = None
    schema_version: str = MATERIAL_CREATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        source_usd = Path(self.source_usd)
        if not source_usd.is_absolute():
            raise ValueError("source_usd must be an absolute path")
        if source_usd.suffix.lower() not in _SUPPORTED_USD_SUFFIXES:
            raise ValueError("source_usd must be a USD, USDA, USDC, or USDZ asset")
        self.recipe.validate()
        targets = _canonical_prim_paths(
            tuple(self.target_prim_paths),
            field_name="target_prim_paths",
            require_non_empty=True,
        )
        refs = tuple(
            _non_empty(uri, field_name="reference_image_uris")
            for uri in self.reference_image_uris
        )
        if len(refs) != len(set(refs)):
            raise ValueError("reference_image_uris must not contain duplicates")
        # Validate recipe-level references together with request-local additions.
        _effective_references(self.recipe, refs)
        if not 1 <= self.texture_size <= 16384:
            raise ValueError("texture_size must be in [1, 16384]")
        if self.seed is not None and not 0 <= self.seed <= 0xFFFFFFFF:
            raise ValueError("seed must be in [0, 4294967295]")
        digest = self.source_usd_sha256
        if digest is not None and not _SHA256_RE.fullmatch(digest):
            raise ValueError("source_usd_sha256 must be a lowercase SHA-256 digest")
        if self.schema_version != MATERIAL_CREATION_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {MATERIAL_CREATION_SCHEMA_VERSION!r}"
            )

        object.__setattr__(self, "source_usd", source_usd)
        object.__setattr__(self, "target_prim_paths", targets)
        object.__setattr__(self, "reference_image_uris", refs)
        object.__setattr__(
            self, "creation_mode", MaterialCreationMode(self.creation_mode)
        )
        object.__setattr__(
            self, "backend", _non_empty(self.backend, field_name="backend")
        )

    @property
    def effective_reference_image_uris(self) -> tuple[str, ...]:
        """References from the recipe followed by request-local references."""

        return _effective_references(self.recipe, self.reference_image_uris)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_usd": self.source_usd.as_posix(),
            "source_usd_sha256": self.source_usd_sha256,
            "target_prim_paths": self.target_prim_paths,
            "recipe": self.recipe.to_dict(),
            "reference_image_uris": self.effective_reference_image_uris,
            "creation_mode": self.creation_mode.value,
            "texture_size": self.texture_size,
            "backend": self.backend,
            "seed": self.seed,
        }

    @property
    def request_id(self) -> str:
        """Stable identifier for exact-request caching and run-local reuse."""

        canonical = json.dumps(
            self._identity_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"mc_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]}"

    @property
    def effective_seed(self) -> int:
        """Return an explicit seed, deriving a stable one when omitted."""

        if self.seed is not None:
            return self.seed
        return int(self.request_id.removeprefix("mc_")[:8], 16)

    @property
    def reuse_key(self) -> str:
        """Recipe-level run-local lookup key.

        Callers may reuse an entry only when its recorded creation cache key is
        compatible with the new request.  A duplicate recipe key with a conflicting
        fingerprint is a conflict, never permission to overwrite the first entry.
        """

        return str(self.recipe.material_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible manifest representation."""

        return {
            **self._identity_payload(),
            "request_id": self.request_id,
            "effective_seed": self.effective_seed,
        }


@dataclass(frozen=True)
class MaterialConditioningArtifact:
    """One geometry, render, or reference input prepared for a backend."""

    kind: MaterialConditioningKind
    uri: str
    color_space: MaterialColorSpace | None = None
    view: str | None = None
    sha256: str | None = None
    evidence_source: MaterialConditioningArtifactSource | None = None
    evidence_source_detail: str | None = None

    def __post_init__(self) -> None:
        kind = MaterialConditioningKind(self.kind)
        color_space = (
            MaterialColorSpace(self.color_space)
            if self.color_space is not None
            else None
        )
        evidence_source = (
            MaterialConditioningArtifactSource(self.evidence_source)
            if self.evidence_source is not None
            else None
        )
        image_kinds = {
            MaterialConditioningKind.UV_LAYOUT,
            MaterialConditioningKind.UV_MASK,
            MaterialConditioningKind.NORMAL,
            MaterialConditioningKind.DEPTH,
            MaterialConditioningKind.SEGMENTATION,
            MaterialConditioningKind.RENDER,
            MaterialConditioningKind.SOURCE_ALBEDO,
            MaterialConditioningKind.REFERENCE_IMAGE,
        }
        if kind in image_kinds and color_space is None:
            raise ValueError(f"{kind.value} conditioning must declare a color space")
        if kind is MaterialConditioningKind.SCOPED_USD and color_space is not None:
            raise ValueError("scoped_usd conditioning must not declare a color space")
        if self.sha256 is not None and not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("conditioning sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "uri", _non_empty(self.uri, field_name="uri"))
        object.__setattr__(self, "color_space", color_space)
        object.__setattr__(self, "evidence_source", evidence_source)
        if self.view is not None:
            object.__setattr__(self, "view", _non_empty(self.view, field_name="view"))
        if self.evidence_source_detail is not None:
            object.__setattr__(
                self,
                "evidence_source_detail",
                _non_empty(
                    self.evidence_source_detail,
                    field_name="evidence_source_detail",
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind.value, "uri": self.uri}
        if self.color_space is not None:
            data["color_space"] = self.color_space.value
        if self.view is not None:
            data["view"] = self.view
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        if self.evidence_source is not None:
            data["evidence_source"] = self.evidence_source.value
        if self.evidence_source_detail is not None:
            data["evidence_source_detail"] = self.evidence_source_detail
        return data


@dataclass(frozen=True)
class PreparedMaterialConditioning:
    """WP4 output consumed unchanged by backend adapters."""

    request_id: str
    target_prim_paths: tuple[str, ...]
    artifacts: tuple[MaterialConditioningArtifact, ...]
    reference_image_uris: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _REQUEST_ID_RE.fullmatch(self.request_id):
            raise ValueError("request_id must be a canonical material-creation ID")
        targets = _canonical_prim_paths(
            tuple(self.target_prim_paths),
            field_name="target_prim_paths",
            require_non_empty=True,
        )
        scoped_assets = tuple(
            artifact
            for artifact in self.artifacts
            if artifact.kind is MaterialConditioningKind.SCOPED_USD
        )
        if len(scoped_assets) != 1:
            raise ValueError(
                "prepared conditioning requires exactly one scoped_usd artifact"
            )
        references = tuple(
            _non_empty(uri, field_name="reference_image_uris")
            for uri in self.reference_image_uris
        )
        reference_artifacts = tuple(
            artifact.uri
            for artifact in self.artifacts
            if artifact.kind is MaterialConditioningKind.REFERENCE_IMAGE
        )
        if reference_artifacts != references:
            raise ValueError(
                "reference-image conditioning artifacts must preserve request reference order"
            )
        object.__setattr__(self, "target_prim_paths", targets)
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "reference_image_uris", references)

    @classmethod
    def for_request(
        cls,
        request: CreateMaterialRequest,
        *,
        artifacts: tuple[MaterialConditioningArtifact, ...],
    ) -> PreparedMaterialConditioning:
        return cls(
            request_id=request.request_id,
            target_prim_paths=request.target_prim_paths,
            artifacts=artifacts,
            reference_image_uris=request.effective_reference_image_uris,
        )

    def validate_request(self, request: CreateMaterialRequest) -> None:
        """Reject conditioning prepared for another source/scope request."""

        if self.request_id != request.request_id:
            raise ValueError("conditioning request_id does not match creation request")
        if self.target_prim_paths != request.target_prim_paths:
            raise ValueError(
                "conditioning target scope does not match creation request"
            )
        if self.reference_image_uris != request.effective_reference_image_uris:
            raise ValueError("conditioning references do not match creation request")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "target_prim_paths": list(self.target_prim_paths),
            "reference_image_uris": list(self.reference_image_uris),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


@dataclass(frozen=True)
class MaterialComponentProvenance:
    """Origin of one semantic component, including components packed in ORM."""

    component: MaterialChannelComponent
    source: MaterialChannelSource
    source_detail: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component",
            MaterialChannelComponent(self.component),
        )
        object.__setattr__(self, "source", MaterialChannelSource(self.source))
        object.__setattr__(
            self,
            "source_detail",
            _non_empty(self.source_detail, field_name="source_detail"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "component": self.component.value,
            "source": self.source.value,
            "source_detail": self.source_detail,
        }


@dataclass(frozen=True)
class MaterialChannelArtifact:
    """One normalized texture channel returned by a backend."""

    channel: MaterialChannel
    path: Path
    color_space: MaterialColorSpace
    component_provenance: tuple[MaterialComponentProvenance, ...]
    packing: ORMPacking | None = None
    normal_convention: NormalConvention | None = None

    def __post_init__(self) -> None:
        channel = MaterialChannel(self.channel)
        color_space = MaterialColorSpace(self.color_space)
        packing = ORMPacking(self.packing) if self.packing is not None else None
        normal_convention = (
            NormalConvention(self.normal_convention)
            if self.normal_convention is not None
            else None
        )
        if (
            channel is MaterialChannel.ALBEDO
            and color_space is not MaterialColorSpace.SRGB
        ):
            raise ValueError("albedo must use the sRGB color space")
        if (
            channel is not MaterialChannel.ALBEDO
            and color_space is not MaterialColorSpace.RAW
        ):
            raise ValueError(f"{channel.value} must use the raw color space")
        if channel is MaterialChannel.ORM:
            if packing is not ORMPacking.OCCLUSION_ROUGHNESS_METALLIC:
                raise ValueError(
                    "orm must declare canonical R=AO, G=roughness, B=metallic packing"
                )
        elif packing is not None:
            raise ValueError("packing is valid only for the orm channel")
        if channel is MaterialChannel.NORMAL:
            if normal_convention is not NormalConvention.TANGENT_OPENGL:
                raise ValueError(
                    "normal must be a tangent-space OpenGL (+Y) normal map"
                )
        elif normal_convention is not None:
            raise ValueError("normal_convention is valid only for the normal channel")

        expected_components = {
            MaterialChannel.ALBEDO: {MaterialChannelComponent.BASE_COLOR},
            MaterialChannel.NORMAL: {MaterialChannelComponent.TANGENT_NORMAL},
            MaterialChannel.ORM: {
                MaterialChannelComponent.OCCLUSION,
                MaterialChannelComponent.ROUGHNESS,
                MaterialChannelComponent.METALLIC,
            },
        }[channel]
        components = tuple(
            component.component for component in self.component_provenance
        )
        actual_components = set(components)
        if len(components) != len(actual_components):
            raise ValueError(
                f"{channel.value} component provenance must not duplicate components"
            )
        if actual_components != expected_components:
            raise ValueError(
                f"{channel.value} component provenance must cover exactly "
                f"{sorted(value.value for value in expected_components)}"
            )

        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "color_space", color_space)
        object.__setattr__(
            self, "component_provenance", tuple(self.component_provenance)
        )
        object.__setattr__(self, "packing", packing)
        object.__setattr__(self, "normal_convention", normal_convention)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "channel": self.channel.value,
            "path": self.path.as_posix(),
            "color_space": self.color_space.value,
            "component_provenance": [
                component.to_dict() for component in self.component_provenance
            ],
        }
        if self.packing is not None:
            data["packing"] = self.packing.value
        if self.normal_convention is not None:
            data["normal_convention"] = self.normal_convention.value
        return data


@dataclass(frozen=True)
class MaterialDegradation:
    """An accepted channel limitation with an explicit fallback or omission."""

    code: MaterialDegradationCode
    channels: tuple[MaterialChannel, ...]
    message: str
    fallback: str

    def __post_init__(self) -> None:
        channels = tuple(
            sorted({MaterialChannel(value) for value in self.channels}, key=str)
        )
        if not channels:
            raise ValueError("degradation channels must not be empty")
        object.__setattr__(self, "code", MaterialDegradationCode(self.code))
        object.__setattr__(self, "channels", channels)
        object.__setattr__(
            self, "message", _non_empty(self.message, field_name="message")
        )
        object.__setattr__(
            self, "fallback", _non_empty(self.fallback, field_name="fallback")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "channels": [channel.value for channel in self.channels],
            "message": self.message,
            "fallback": self.fallback,
        }


@dataclass(frozen=True)
class MaterialCreationDiagnostic:
    """Machine-readable diagnostic suitable for progress and failure reporting."""

    code: str
    message: str
    severity: MaterialDiagnosticSeverity
    phase: str
    channels: tuple[MaterialChannel, ...] = ()
    retryable: bool = False
    details: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _non_empty(self.code, field_name="code"))
        object.__setattr__(
            self, "message", _non_empty(self.message, field_name="message")
        )
        object.__setattr__(self, "severity", MaterialDiagnosticSeverity(self.severity))
        object.__setattr__(self, "phase", _non_empty(self.phase, field_name="phase"))
        object.__setattr__(
            self,
            "channels",
            tuple(sorted({MaterialChannel(value) for value in self.channels}, key=str)),
        )
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "phase": self.phase,
            "channels": [channel.value for channel in self.channels],
            "retryable": self.retryable,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class MaterialCreationProvenance:
    """Backend, model, conditioning, and timing provenance for one result."""

    request_id: str
    cache_key: str
    backend: str
    backend_revision: str
    model_revisions: tuple[str, ...]
    recipe_id: str
    prompt: str
    seed: int
    target_prim_paths: tuple[str, ...]
    reference_image_uris: tuple[str, ...]
    duration_seconds: float
    source_usd: str
    source_usd_sha256: str | None = None
    conditioning_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not _REQUEST_ID_RE.fullmatch(self.request_id):
            raise ValueError("request_id must be a canonical material-creation ID")
        if not _SHA256_RE.fullmatch(self.cache_key):
            raise ValueError("cache_key must be a lowercase SHA-256 digest")
        if not self.model_revisions:
            raise ValueError("model_revisions must contain at least one revision")
        if self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be non-negative")
        if self.source_usd_sha256 is not None and not _SHA256_RE.fullmatch(
            self.source_usd_sha256
        ):
            raise ValueError("source_usd_sha256 must be a lowercase SHA-256 digest")
        if self.conditioning_fingerprint is not None and not _SHA256_RE.fullmatch(
            self.conditioning_fingerprint
        ):
            raise ValueError(
                "conditioning_fingerprint must be a lowercase SHA-256 digest"
            )
        object.__setattr__(
            self, "backend", _non_empty(self.backend, field_name="backend")
        )
        object.__setattr__(
            self,
            "backend_revision",
            _non_empty(self.backend_revision, field_name="backend_revision"),
        )
        object.__setattr__(
            self,
            "model_revisions",
            tuple(
                _non_empty(revision, field_name="model_revisions")
                for revision in self.model_revisions
            ),
        )
        object.__setattr__(self, "recipe_id", make_material_id(self.recipe_id))
        object.__setattr__(self, "prompt", _non_empty(self.prompt, field_name="prompt"))
        object.__setattr__(
            self,
            "source_usd",
            _non_empty(self.source_usd, field_name="source_usd"),
        )
        object.__setattr__(
            self,
            "target_prim_paths",
            _canonical_prim_paths(
                tuple(self.target_prim_paths),
                field_name="target_prim_paths",
                require_non_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "reference_image_uris",
            tuple(
                _non_empty(uri, field_name="reference_image_uris")
                for uri in self.reference_image_uris
            ),
        )

    @classmethod
    def for_request(
        cls,
        request: CreateMaterialRequest,
        *,
        backend: str,
        backend_revision: str,
        model_revisions: tuple[str, ...],
        duration_seconds: float,
        conditioning: PreparedMaterialConditioning | None = None,
    ) -> MaterialCreationProvenance:
        conditioning_payload = None
        if conditioning is not None:
            conditioning.validate_request(request)
            conditioning_payload = conditioning.to_dict()
        conditioning_fingerprint = None
        if conditioning_payload is not None:
            serialized = json.dumps(
                conditioning_payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            conditioning_fingerprint = hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest()
        cache_payload = {
            "request": request.to_dict(),
            "backend": backend,
            "backend_revision": backend_revision,
            "model_revisions": model_revisions,
            "conditioning_fingerprint": conditioning_fingerprint,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return cls(
            request_id=request.request_id,
            cache_key=cache_key,
            backend=backend,
            backend_revision=backend_revision,
            model_revisions=model_revisions,
            recipe_id=request.recipe.material_id,
            prompt=request.recipe.appearance_prompt,
            seed=request.effective_seed,
            target_prim_paths=request.target_prim_paths,
            reference_image_uris=request.effective_reference_image_uris,
            duration_seconds=duration_seconds,
            source_usd=request.source_usd.as_posix(),
            source_usd_sha256=request.source_usd_sha256,
            conditioning_fingerprint=conditioning_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "cache_key": self.cache_key,
            "backend": self.backend,
            "backend_revision": self.backend_revision,
            "model_revisions": list(self.model_revisions),
            "recipe_id": self.recipe_id,
            "prompt": self.prompt,
            "seed": self.seed,
            "target_prim_paths": list(self.target_prim_paths),
            "reference_image_uris": list(self.reference_image_uris),
            "duration_seconds": self.duration_seconds,
            "source_usd": self.source_usd,
            "source_usd_sha256": self.source_usd_sha256,
            "conditioning_fingerprint": self.conditioning_fingerprint,
        }


_MISSING_DEGRADATION_CODES_BY_CHANNEL = {
    MaterialChannel.ALBEDO: (MaterialDegradationCode.MISSING_ALBEDO,),
    MaterialChannel.NORMAL: (
        MaterialDegradationCode.MISSING_NORMAL,
        MaterialDegradationCode.BUMP_TO_NORMAL_FAILED,
    ),
    MaterialChannel.ORM: (MaterialDegradationCode.MISSING_ORM,),
}


def _has_channel_degradation(
    degradations: tuple[MaterialDegradation, ...],
    *,
    channel: MaterialChannel,
    codes: tuple[MaterialDegradationCode, ...],
) -> bool:
    return any(
        degradation.code in codes and channel in degradation.channels
        for degradation in degradations
    )


@dataclass(frozen=True)
class BackendMaterialResult:
    """Normalized successful or explicitly degraded backend output."""

    artifacts: tuple[MaterialChannelArtifact, ...]
    provenance: MaterialCreationProvenance
    degradations: tuple[MaterialDegradation, ...] = ()
    diagnostics: tuple[MaterialCreationDiagnostic, ...] = ()
    preview_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        channels = tuple(artifact.channel for artifact in self.artifacts)
        if len(channels) != len(set(channels)):
            raise ValueError(
                "backend result must contain at most one artifact per channel"
            )
        for channel, missing_codes in _MISSING_DEGRADATION_CODES_BY_CHANNEL.items():
            if channel not in channels and not _has_channel_degradation(
                self.degradations,
                channel=channel,
                codes=missing_codes,
            ):
                valid_codes = ", ".join(code.value for code in missing_codes)
                raise ValueError(
                    f"missing {channel.value} requires an explicit "
                    f"degradation ({valid_codes})"
                )
        object.__setattr__(
            self,
            "preview_paths",
            tuple(Path(path) for path in self.preview_paths),
        )

    def artifact(
        self, channel: MaterialChannel | str
    ) -> MaterialChannelArtifact | None:
        normalized = MaterialChannel(channel)
        return next(
            (artifact for artifact in self.artifacts if artifact.channel is normalized),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "provenance": self.provenance.to_dict(),
            "degradations": [item.to_dict() for item in self.degradations],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "preview_paths": [path.as_posix() for path in self.preview_paths],
        }


class MaterialCreationError(RuntimeError):
    """Structured backend/orchestration failure; no package may be registered."""

    def __init__(
        self,
        code: MaterialCreationErrorCode | str,
        message: str,
        *,
        backend: str | None = None,
        retryable: bool = False,
        diagnostics: tuple[MaterialCreationDiagnostic, ...] = (),
    ) -> None:
        super().__init__(_non_empty(message, field_name="message"))
        self.code = MaterialCreationErrorCode(code)
        self.backend = backend
        self.retryable = retryable
        self.diagnostics = diagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": str(self),
            "backend": self.backend,
            "retryable": self.retryable,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@runtime_checkable
class MaterialCreationBackend(Protocol):
    """Backend boundary used by the material-creation orchestrator."""

    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def create(
        self,
        request: CreateMaterialRequest,
        *,
        output_dir: Path,
        conditioning: PreparedMaterialConditioning | None = None,
        cancel_event: threading.Event | None = None,
    ) -> BackendMaterialResult: ...


@dataclass(frozen=True)
class CreatedMaterialListEntry:
    """Normal Material Agent list entry augmented with run-local provenance."""

    name: str
    description: str
    binding: str
    generation_id: str
    creation_request_id: str
    creation_cache_key: str
    reuse_key: str
    target_prim_paths: tuple[str, ...]
    creation_manifest: str
    intended_parts: tuple[str, ...] = ()
    source: str = "generated"

    def __post_init__(self) -> None:
        if not _REQUEST_ID_RE.fullmatch(self.creation_request_id):
            raise ValueError(
                "creation_request_id must be a canonical material-creation ID"
            )
        if not _SHA256_RE.fullmatch(self.creation_cache_key):
            raise ValueError("creation_cache_key must be a lowercase SHA-256 digest")
        if self.source != "generated":
            raise ValueError(
                "created material-list entries must use source='generated'"
            )
        object.__setattr__(self, "name", _non_empty(self.name, field_name="name"))
        object.__setattr__(
            self,
            "description",
            _non_empty(self.description, field_name="description"),
        )
        object.__setattr__(
            self, "binding", _canonical_prim_path(self.binding, field_name="binding")
        )
        object.__setattr__(self, "generation_id", make_material_id(self.generation_id))
        object.__setattr__(self, "reuse_key", make_material_id(self.reuse_key))
        if self.reuse_key != self.generation_id:
            raise ValueError("reuse_key must match the recipe generation_id")
        object.__setattr__(
            self,
            "target_prim_paths",
            _canonical_prim_paths(
                tuple(self.target_prim_paths),
                field_name="target_prim_paths",
                require_non_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "creation_manifest",
            _non_empty(self.creation_manifest, field_name="creation_manifest"),
        )
        object.__setattr__(
            self,
            "intended_parts",
            tuple(
                _non_empty(part, field_name="intended_parts")
                for part in self.intended_parts
            ),
        )

    @classmethod
    def for_request(
        cls,
        request: CreateMaterialRequest,
        *,
        creation_manifest: str | Path,
        provenance: MaterialCreationProvenance,
    ) -> CreatedMaterialListEntry:
        if provenance.request_id != request.request_id:
            raise ValueError("provenance does not belong to the creation request")
        recipe = request.recipe
        return cls(
            name=recipe.name,
            description=recipe.description,
            binding=recipe.binding,
            generation_id=recipe.material_id,
            creation_request_id=request.request_id,
            creation_cache_key=provenance.cache_key,
            reuse_key=recipe.material_id,
            target_prim_paths=request.target_prim_paths,
            creation_manifest=str(creation_manifest),
            intended_parts=tuple(part.semantic_label for part in recipe.intended_parts),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "binding": self.binding,
            "source": self.source,
            "generation_id": self.generation_id,
            "creation_request_id": self.creation_request_id,
            "creation_cache_key": self.creation_cache_key,
            "reuse_key": self.reuse_key,
            "target_prim_paths": list(self.target_prim_paths),
            "creation_manifest": self.creation_manifest,
        }
        if self.intended_parts:
            data["intended_parts"] = list(self.intended_parts)
        return data


@dataclass(frozen=True)
class CreatedMaterial:
    """Packaged material ready for run-local list insertion and assignment."""

    material_id: str
    material_prim_path: str
    material_usd_path: Path
    creation_manifest_path: Path
    texture_artifacts: tuple[MaterialChannelArtifact, ...]
    material_list_entry: CreatedMaterialListEntry
    preview_paths: tuple[Path, ...]
    validation: dict[str, Any] = field(compare=False, hash=False)
    provenance: MaterialCreationProvenance
    degradations: tuple[MaterialDegradation, ...] = ()

    def __post_init__(self) -> None:
        material_id = make_material_id(self.material_id)
        if self.material_list_entry.generation_id != material_id:
            raise ValueError("material_list_entry generation_id must match material_id")
        if self.material_list_entry.creation_request_id != self.provenance.request_id:
            raise ValueError(
                "material-list entry and provenance request IDs must match"
            )
        if self.material_list_entry.creation_cache_key != self.provenance.cache_key:
            raise ValueError("material-list entry and provenance cache keys must match")
        if (
            self.material_list_entry.target_prim_paths
            != self.provenance.target_prim_paths
        ):
            raise ValueError(
                "material-list entry and provenance target scopes must match"
            )
        if self.material_prim_path != self.material_list_entry.binding:
            raise ValueError(
                "material_prim_path must match material-list entry binding"
            )
        artifact_channels = tuple(
            artifact.channel for artifact in self.texture_artifacts
        )
        if len(artifact_channels) != len(set(artifact_channels)):
            raise ValueError(
                "packaged material texture_artifacts must not duplicate channels"
            )
        artifacts = {artifact.channel: artifact for artifact in self.texture_artifacts}
        for required in (MaterialChannel.ALBEDO, MaterialChannel.ORM):
            if required not in artifacts:
                raise ValueError(f"packaged material requires {required.value}")
        if MaterialChannel.NORMAL not in artifacts and not _has_channel_degradation(
            self.degradations,
            channel=MaterialChannel.NORMAL,
            codes=(
                MaterialDegradationCode.MISSING_NORMAL,
                MaterialDegradationCode.BUMP_TO_NORMAL_FAILED,
            ),
        ):
            raise ValueError("missing packaged normal requires an explicit degradation")
        object.__setattr__(self, "material_id", material_id)
        object.__setattr__(
            self,
            "material_prim_path",
            _canonical_prim_path(
                self.material_prim_path, field_name="material_prim_path"
            ),
        )
        object.__setattr__(self, "material_usd_path", Path(self.material_usd_path))
        object.__setattr__(
            self,
            "creation_manifest_path",
            Path(self.creation_manifest_path),
        )
        object.__setattr__(
            self, "preview_paths", tuple(Path(path) for path in self.preview_paths)
        )
        object.__setattr__(self, "validation", dict(self.validation))

    @property
    def texture_paths(self) -> dict[str, Path]:
        return {
            artifact.channel.value: artifact.path for artifact in self.texture_artifacts
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id,
            "material_prim_path": self.material_prim_path,
            "material_usd_path": self.material_usd_path.as_posix(),
            "creation_manifest_path": self.creation_manifest_path.as_posix(),
            "texture_paths": {
                channel: path.as_posix() for channel, path in self.texture_paths.items()
            },
            "texture_artifacts": [
                artifact.to_dict() for artifact in self.texture_artifacts
            ],
            "material_list_entry": self.material_list_entry.to_dict(),
            "preview_paths": [path.as_posix() for path in self.preview_paths],
            "validation": dict(self.validation),
            "provenance": self.provenance.to_dict(),
            "degradations": [item.to_dict() for item in self.degradations],
        }
