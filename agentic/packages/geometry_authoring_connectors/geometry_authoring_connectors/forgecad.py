# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact-only ForgeCAD adapter.

The adapter never imports, installs, invokes, or evaluates ForgeCAD. It accepts
native source only as inert provenance accompanying an already exported geometry
artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self, cast

from geometry_authoring_contracts import GeometrySourceBundle
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ._artifacts import (
    materialize_wire_bundle,
    read_regular_file,
    validate_artifact_filename,
)
from ._formats import GEOMETRY_MEDIA_TYPES, validate_geometry_bytes
from .errors import (
    ForgeCadExecutionUnavailableError,
    InvalidProviderResponseError,
    UnsupportedCapabilityError,
)
from .models import (
    SAFE_FILENAME_PATTERN,
    SHA256_PATTERN,
    ArtifactRole,
    AuthoringCapabilities,
    MaterializedWireSourceBundle,
    WireArtifact,
    WireSourceBundle,
    canonical_json_digest,
)

FORGECAD_PROVIDER_ID = "forgecad-artifact-adapter"
FORGECAD_MANIFEST_SCHEMA_VERSION: Literal["geometry-authoring-connectors.forgecad-artifacts.v1"] = (
    "geometry-authoring-connectors.forgecad-artifacts.v1"
)
_MAX_FORGE_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024


class ForgeCadManifestArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(pattern=SAFE_FILENAME_PATTERN)
    role: ArtifactRole
    media_type: str = Field(min_length=1, max_length=128)
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=0, le=256 * 1024 * 1024)


class ForgeCadArtifactManifest(BaseModel):
    """Exact digest inventory accepted by the public artifact adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["geometry-authoring-connectors.forgecad-artifacts.v1"] = (
        FORGECAD_MANIFEST_SCHEMA_VERSION
    )
    provider_version: str = Field(min_length=1, max_length=128)
    source_revision: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    units: Literal["millimeter", "centimeter", "meter", "inch"]
    up_axis: Literal["X", "Y", "Z"]
    rights_assertion: str = Field(min_length=1, max_length=2048)
    artifacts: tuple[ForgeCadManifestArtifact, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_unique_artifacts(self) -> Self:
        names = tuple(item.filename for item in self.artifacts)
        if len(names) != len(set(names)):
            raise ValueError("ForgeCAD manifest artifact filenames must be unique")
        return self


def _validate_forge_source(content: bytes) -> None:
    if len(content) > _MAX_FORGE_SOURCE_BYTES:
        raise InvalidProviderResponseError("ForgeCAD source exceeds the bounded source limit")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidProviderResponseError("ForgeCAD source must be UTF-8 text") from exc
    if "\x00" in text or not text.strip():
        raise InvalidProviderResponseError("ForgeCAD source is empty or contains NUL bytes")


class ForgeCadArtifactAdapter:
    """Validate and materialize inert ForgeCAD source plus existing exports."""

    def __init__(self, *, provider_id: str = FORGECAD_PROVIDER_ID) -> None:
        self._provider_id = provider_id

    def capabilities(self) -> AuthoringCapabilities:
        return AuthoringCapabilities(
            provider_id=self._provider_id,
            text=False,
            image=False,
            revision=False,
            export=False,
            native_source=True,
            formats=("step", "stl", "obj", "ply", "glb", "gltf", "3mf", "usd", "usda", "usdc"),
        )

    def generate(self, *_args: Any, **_kwargs: Any) -> None:
        raise ForgeCadExecutionUnavailableError(
            "public ForgeCAD support is artifact-only; generation requires an independently licensed provider",
            provider_id=self._provider_id,
        )

    def revise(self, *_args: Any, **_kwargs: Any) -> None:
        raise ForgeCadExecutionUnavailableError(
            "public ForgeCAD support cannot execute or revise .forge.js source",
            provider_id=self._provider_id,
        )

    def export(self, *_args: Any, **_kwargs: Any) -> None:
        raise ForgeCadExecutionUnavailableError(
            "public ForgeCAD support imports existing exports and never invokes ForgeCAD",
            provider_id=self._provider_id,
        )

    def import_artifacts(
        self,
        *,
        exports: tuple[str | Path, ...],
        output_dir: str | Path,
        forge_source: str | Path | None = None,
        manifest_path: str | Path | None = None,
        units: Literal["millimeter", "centimeter", "meter", "inch"] = "millimeter",
        up_axis: Literal["X", "Y", "Z"] = "Z",
    ) -> MaterializedWireSourceBundle:
        if forge_source is not None and not str(forge_source).lower().endswith(".forge.js"):
            raise UnsupportedCapabilityError(
                "ForgeCAD native source must use the .forge.js suffix",
                provider_id=self._provider_id,
            )
        if not exports:
            raise ForgeCadExecutionUnavailableError(
                "a .forge.js source requires an existing STEP, mesh, or USD export",
                provider_id=self._provider_id,
            )

        wire_artifacts: list[WireArtifact] = []
        if forge_source is not None:
            source_path = Path(forge_source)
            validate_artifact_filename(source_path.name)
            source_content = read_regular_file(
                source_path,
                label="ForgeCAD native source",
                max_bytes=_MAX_FORGE_SOURCE_BYTES,
            )
            _validate_forge_source(source_content)
            wire_artifacts.append(
                WireArtifact.from_bytes(
                    filename=source_path.name,
                    role="native_source",
                    media_type="text/javascript",
                    content=source_content,
                )
            )

        for export_path_value in exports:
            export_path = Path(export_path_value)
            validate_artifact_filename(export_path.name)
            extension = export_path.suffix.lower()
            definition = GEOMETRY_MEDIA_TYPES.get(extension)
            if definition is None:
                raise UnsupportedCapabilityError(
                    f"unsupported ForgeCAD export format: {extension or '<none>'}",
                    provider_id=self._provider_id,
                )
            role, media_type = definition
            content = read_regular_file(export_path, label="ForgeCAD geometry export")
            validate_geometry_bytes(export_path.name, content)
            wire_artifacts.append(
                WireArtifact.from_bytes(
                    filename=export_path.name,
                    role=cast(ArtifactRole, role),
                    media_type=media_type,
                    content=content,
                )
            )

        names = tuple(item.filename for item in wire_artifacts)
        if len(names) != len(set(names)):
            raise InvalidProviderResponseError(
                "ForgeCAD input artifact filenames must be unique",
                provider_id=self._provider_id,
            )

        manifest: ForgeCadArtifactManifest | None = None
        if manifest_path is not None:
            manifest_content = read_regular_file(
                manifest_path,
                label="ForgeCAD artifact manifest",
                max_bytes=_MAX_MANIFEST_BYTES,
            )
            try:
                manifest = ForgeCadArtifactManifest.model_validate_json(manifest_content)
            except ValidationError as exc:
                raise InvalidProviderResponseError(
                    "ForgeCAD artifact manifest is invalid",
                    provider_id=self._provider_id,
                ) from exc
            self._verify_manifest(manifest, wire_artifacts)
            manifest_name = Path(manifest_path).name
            validate_artifact_filename(manifest_name)
            if manifest_name in names:
                raise InvalidProviderResponseError(
                    "ForgeCAD manifest filename collides with an artifact",
                    provider_id=self._provider_id,
                )
            wire_artifacts.append(
                WireArtifact.from_bytes(
                    filename=manifest_name,
                    role="manifest",
                    media_type="application/json",
                    content=manifest_content,
                )
            )

        source_revision = (
            manifest.source_revision
            if manifest is not None
            else canonical_json_digest(
                [{"filename": item.filename, "sha256": item.sha256} for item in wire_artifacts]
            )
        )
        bundle = WireSourceBundle(
            provider_id=self._provider_id,
            provider_version=(
                manifest.provider_version if manifest is not None else "artifact-only-v1"
            ),
            source_revision=source_revision,
            units=manifest.units if manifest is not None else units,
            up_axis=manifest.up_axis if manifest is not None else up_axis,
            artifacts=tuple(wire_artifacts),
            metadata={
                "source_system": "forgecad",
                "execution_invoked": False,
                "artifact_only": True,
                "manifest_verified": manifest is not None,
                "rights_assertion": manifest.rights_assertion
                if manifest is not None
                else "not_provided",
            },
        )
        return materialize_wire_bundle(bundle, output_dir)

    def import_source_bundle(
        self,
        *,
        exports: tuple[str | Path, ...],
        output_dir: str | Path,
        rights_assertion: str | None = None,
        forge_source: str | Path | None = None,
        manifest_path: str | Path | None = None,
        units: Literal["millimeter", "centimeter", "meter", "inch"] = "millimeter",
        up_axis: Literal["X", "Y", "Z"] = "Z",
    ) -> GeometrySourceBundle:
        """Import existing exports into the canonical public source contract.

        ForgeCAD is never invoked. A rights assertion must be supplied directly or
        by the optional digest manifest before any artifact enters the workflow.
        """

        materialized = self.import_artifacts(
            exports=exports,
            output_dir=output_dir,
            forge_source=forge_source,
            manifest_path=manifest_path,
            units=units,
            up_axis=up_axis,
        )
        asserted_rights = (
            str(materialized.metadata.get("rights_assertion", "")).strip()
            if manifest_path is not None
            else (rights_assertion or "").strip()
        )
        if not asserted_rights or asserted_rights == "not_provided":
            raise InvalidProviderResponseError(
                "ForgeCAD artifact import requires an explicit rights assertion",
                provider_id=self._provider_id,
            )
        request_digest = canonical_json_digest(
            {
                "operation": "artifact_import",
                "provider_id": self._provider_id,
                "source_revision": materialized.source_revision,
                "artifacts": [
                    {
                        "filename": artifact.filename,
                        "sha256": artifact.sha256,
                        "size_bytes": artifact.size_bytes,
                    }
                    for artifact in materialized.artifacts
                ],
            }
        )
        from .providers import canonicalize_materialized_bundle

        return canonicalize_materialized_bundle(
            materialized,
            request_digest=request_digest,
            rights_assertion=asserted_rights,
        )

    def _verify_manifest(
        self,
        manifest: ForgeCadArtifactManifest,
        artifacts: list[WireArtifact],
    ) -> None:
        expected = {item.filename: item for item in manifest.artifacts}
        observed = {item.filename: item for item in artifacts}
        if expected.keys() != observed.keys():
            raise InvalidProviderResponseError(
                "ForgeCAD manifest inventory differs from supplied artifacts",
                provider_id=self._provider_id,
            )
        for filename, declared in expected.items():
            actual = observed[filename]
            if (
                declared.sha256 != actual.sha256
                or declared.size_bytes != actual.size_bytes
                or declared.role != actual.role
                or declared.media_type != actual.media_type
            ):
                raise InvalidProviderResponseError(
                    f"ForgeCAD manifest binding differs for {filename}",
                    provider_id=self._provider_id,
                )
