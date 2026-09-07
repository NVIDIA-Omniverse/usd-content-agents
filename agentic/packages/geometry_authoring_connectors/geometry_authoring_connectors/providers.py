# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical provider adapters for the connector-private transports."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from pathlib import Path
from typing import Literal, cast

from geometry_authoring_contracts import (
    GeometryArtifactBinding,
    GeometryAuthoringCapabilityReport,
    GeometryAuthoringExportRequest,
    GeometryAuthoringFailure,
    GeometryAuthoringFeature,
    GeometryAuthoringInputDrift,
    GeometryAuthoringInvalidResponse,
    GeometryAuthoringParameterValue,
    GeometryAuthoringProviderIdentity,
    GeometryAuthoringProviderReceipt,
    GeometryAuthoringReference,
    GeometryAuthoringRequest,
    GeometryAuthoringRevisionRequest,
    GeometryCoordinateSystem,
    GeometryPartBinding,
    GeometryRepresentationBinding,
    GeometryRightsAssertion,
    GeometrySemanticParameter,
    GeometrySourceBundle,
    GeometrySourceProvenance,
    GeometryVerificationAssertion,
    geometry_authoring_request_digest,
    geometry_source_bundle_id,
    resolve_semantic_parameter_overrides,
    validate_returned_parameter_values,
)

from ._artifacts import read_regular_file
from .build123d import Build123dHttpConnector
from .delegated import DelegatedAuthoringHttpConnector
from .errors import (
    ArtifactIntegrityError,
    GeometryAuthoringConnectorError,
    InvalidProviderResponseError,
    ProviderTransportError,
    ProviderUnavailableError,
    UnsafeArtifactError,
    UnsupportedCapabilityError,
)
from .models import (
    AuthoringRequest,
    GeometryFormat,
    InlineInputArtifact,
    MaterializedWireSourceBundle,
)

_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".3mf": "3mf",
    ".brep": "brep",
    ".glb": "glb",
    ".gltf": "gltf",
    ".iges": "iges",
    ".igs": "iges",
    ".obj": "obj",
    ".ply": "ply",
    ".step": "step",
    ".stl": "stl",
    ".stp": "step",
    ".usd": "usd",
    ".usda": "usda",
    ".usdc": "usdc",
}
_ROLE_MAP = {
    "native_source": "native_source",
    "cad_geometry": "design_exchange",
    "render_geometry": "render_geometry",
    "mesh_geometry": "render_geometry",
    "collision_geometry": "collision_candidate",
    "supporting_asset": "supporting_asset",
    "manifest": "metadata",
    "reference_image": "reference",
}
_METERS_PER_UNIT = {
    "millimeter": 0.001,
    "centimeter": 0.01,
    "meter": 1.0,
    "inch": 0.0254,
}


def _discard_owned_output_directory(root: Path, candidate: Path) -> None:
    """Best-effort rollback for one flat, provider-owned request directory."""

    lexical_root = Path(os.path.abspath(root))
    lexical_candidate = Path(os.path.abspath(candidate))
    if lexical_candidate.parent != lexical_root:
        return
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_fd = candidate_fd = quarantine_fd = -1
    quarantine_name: str | None = None
    quarantine_entry: str | None = None
    try:
        root_fd = os.open(lexical_root, directory_flags)
        candidate_fd = os.open(lexical_candidate.name, directory_flags, dir_fd=root_fd)
        opened = os.fstat(candidate_fd)
        current = os.stat(
            lexical_candidate.name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            return
        entries = os.listdir(candidate_fd)
        if any(
            stat.S_ISDIR(os.stat(name, dir_fd=candidate_fd, follow_symlinks=False).st_mode)
            for name in entries
        ):
            return

        quarantine_name = f".geometry-authoring-cleanup-{uuid.uuid4().hex}"
        quarantine_entry = uuid.uuid4().hex
        os.mkdir(quarantine_name, mode=0o700, dir_fd=root_fd)
        quarantine_fd = os.open(quarantine_name, directory_flags, dir_fd=root_fd)
        os.rename(
            lexical_candidate.name,
            quarantine_entry,
            src_dir_fd=root_fd,
            dst_dir_fd=quarantine_fd,
        )
        quarantined = os.stat(
            quarantine_entry,
            dir_fd=quarantine_fd,
            follow_symlinks=False,
        )
        if (opened.st_dev, opened.st_ino) != (
            quarantined.st_dev,
            quarantined.st_ino,
        ):
            return

        for name in entries:
            os.unlink(name, dir_fd=candidate_fd)
        os.fsync(candidate_fd)
        quarantined = os.stat(
            quarantine_entry,
            dir_fd=quarantine_fd,
            follow_symlinks=False,
        )
        if (opened.st_dev, opened.st_ino) != (
            quarantined.st_dev,
            quarantined.st_ino,
        ):
            return
        os.close(candidate_fd)
        candidate_fd = -1
        os.rmdir(quarantine_entry, dir_fd=quarantine_fd)
        quarantine_entry = None
        os.fsync(quarantine_fd)
        os.close(quarantine_fd)
        quarantine_fd = -1
        os.rmdir(quarantine_name, dir_fd=root_fd)
        quarantine_name = None
        os.fsync(root_fd)
    except OSError:
        pass
    finally:
        for descriptor in (candidate_fd, quarantine_fd):
            if descriptor >= 0:
                os.close(descriptor)
        if quarantine_name is not None and root_fd >= 0:
            try:
                os.rmdir(quarantine_name, dir_fd=root_fd)
            except OSError:
                pass
        if root_fd >= 0:
            os.close(root_fd)


def _artifact_binding(path: Path, *, sha256: str, size_bytes: int) -> GeometryArtifactBinding:
    absolute = path.resolve(strict=True)
    content = read_regular_file(absolute, label="provider output")
    if len(content) != size_bytes or hashlib.sha256(content).hexdigest() != sha256:
        raise InvalidProviderResponseError(
            "Materialized provider artifact differs from its declared identity"
        )
    return GeometryArtifactBinding(
        path=str(absolute),
        sha256=sha256,
        size_bytes=size_bytes,
    )


def canonicalize_materialized_bundle(
    bundle: MaterializedWireSourceBundle,
    *,
    request_digest: str,
    rights_assertion: str,
    license_identifier: str | None = None,
    parent_bundle_id: str | None = None,
    input_artifacts: tuple[GeometryArtifactBinding, ...] = (),
    parameters: tuple[GeometryAuthoringParameterValue, ...] = (),
    parameter_definitions: tuple[GeometrySemanticParameter, ...] = (),
    parts: tuple[GeometryPartBinding, ...] = (),
    upstream_edit_uri: str | None = None,
) -> GeometrySourceBundle:
    """Translate a verified connector result into the only public source schema."""

    producer = GeometryAuthoringProviderIdentity(
        provider_id=bundle.provider_id,
        provider_version=bundle.provider_version,
    )
    representations: list[GeometryRepresentationBinding] = []
    for index, artifact in enumerate(bundle.artifacts):
        suffix = artifact.path.suffix.lower()
        format_name = _FORMAT_BY_SUFFIX.get(suffix)
        if format_name is None:
            if artifact.role == "manifest":
                format_name = "json"
            elif artifact.role in {"native_source", "supporting_asset"}:
                format_name = suffix.removeprefix(".").replace(".", "+") or "source"
            else:
                raise InvalidProviderResponseError(
                    f"Provider returned an unsupported artifact suffix: {suffix or '<none>'}",
                    provider_id=bundle.provider_id,
                )
        representations.append(
            GeometryRepresentationBinding(
                representation_id=f"representation-{index + 1:03d}",
                role=cast(
                    Literal[
                        "native_source",
                        "design_exchange",
                        "render_geometry",
                        "collision_candidate",
                        "supporting_asset",
                        "reference",
                        "metadata",
                    ],
                    _ROLE_MAP[artifact.role],
                ),
                format=format_name,
                media_type=artifact.media_type,
                artifact=_artifact_binding(
                    artifact.path,
                    sha256=artifact.sha256,
                    size_bytes=artifact.size_bytes,
                ),
            )
        )
    typed_representations = tuple(representations)
    representation_by_filename = {
        artifact.filename: representation.representation_id
        for artifact, representation in zip(bundle.artifacts, typed_representations, strict=True)
    }
    wire_parts = tuple(
        GeometryPartBinding(
            part_id=item.part_id,
            name=item.name,
            parent_part_id=item.parent_part_id,
            representation_ids=tuple(
                representation_by_filename[name] for name in item.artifact_filenames
            ),
            transform=item.transform,
        )
        for item in bundle.parts
    )
    canonical_parts = wire_parts or parts
    if bundle.parameters:
        returned_parameters = tuple(
            GeometrySemanticParameter(
                name=item.name,
                value=item.value,
                unit=item.unit,
                minimum=item.minimum,
                maximum=item.maximum,
                step=item.step,
                choices=item.choices,
                group=item.group,
                semantic_role=item.semantic_role,
                effects=item.effects,
                affects=item.affects,
                visible=item.visible,
                value_type=item.value_type,
                description=item.description,
            )
            for item in bundle.parameters
        )
        returned_names = {item.name for item in returned_parameters}
        semantic_parameters = (
            *returned_parameters,
            *(item for item in parameter_definitions if item.name not in returned_names),
        )
    else:
        semantic_parameters = resolve_semantic_parameter_overrides(
            parameter_definitions,
            parameters,
            allow_undeclared=not parameter_definitions,
        )
    validate_returned_parameter_values(semantic_parameters, parameters)
    verification_assertions = tuple(
        GeometryVerificationAssertion(
            assertion_id=item.assertion_id,
            status=item.status,
            summary=item.summary,
            metrics=item.metrics,
        )
        for item in bundle.verification_assertions
    )
    coordinate_system = GeometryCoordinateSystem(
        meters_per_unit=_METERS_PER_UNIT[bundle.units],
        up_axis=bundle.up_axis,
        forward_axis=bundle.forward_axis,
        handedness=bundle.handedness,
    )
    provenance = GeometrySourceProvenance(
        request_digest=request_digest,
        parent_bundle_id=parent_bundle_id,
        input_artifacts=input_artifacts,
        upstream_edit_uri=bundle.upstream_edit_uri or upstream_edit_uri,
    )
    rights = GeometryRightsAssertion(
        assertion=rights_assertion,
        license_identifier=license_identifier,
    )
    return GeometrySourceBundle(
        bundle_id=geometry_source_bundle_id(
            provider=producer,
            source_revision=bundle.source_revision,
            coordinate_system=coordinate_system,
            representations=typed_representations,
            parts=canonical_parts,
            parameters=semantic_parameters,
            verification_assertions=verification_assertions,
            provenance=provenance,
            rights=rights,
        ),
        producer=producer,
        source_revision=bundle.source_revision,
        coordinate_system=coordinate_system,
        representations=typed_representations,
        parts=canonical_parts,
        parameters=semantic_parameters,
        verification_assertions=verification_assertions,
        provenance=provenance,
        rights=rights,
    )


def _failed_receipt(
    *,
    provider: GeometryAuthoringProviderIdentity,
    operation: Literal["generate", "revise", "export"],
    request_digest: str,
    exc: Exception,
) -> GeometryAuthoringProviderReceipt:
    code: Literal[
        "provider_unavailable",
        "unsupported_operation",
        "transport_error",
        "http_error",
        "invalid_response",
        "provider_failure",
        "input_drift",
        "artifact_mismatch",
    ] = "provider_failure"
    retryable = False
    if isinstance(exc, UnsupportedCapabilityError):
        code = "unsupported_operation"
    elif isinstance(exc, ProviderUnavailableError):
        code = "provider_unavailable"
        retryable = True
    elif isinstance(exc, ProviderTransportError):
        code = "transport_error"
        retryable = True
    elif isinstance(exc, ArtifactIntegrityError | UnsafeArtifactError):
        code = "artifact_mismatch"
    elif isinstance(exc, InvalidProviderResponseError | GeometryAuthoringInvalidResponse):
        code = "invalid_response"
    elif isinstance(exc, GeometryAuthoringInputDrift):
        code = "input_drift"
    summary = (
        str(exc)
        if isinstance(exc, GeometryAuthoringConnectorError)
        else ("The selected geometry authoring provider failed.")
    )
    failure = GeometryAuthoringFailure(
        provider_id=provider.provider_id,
        operation=operation,
        code=code,
        summary=summary,
        retryable=retryable,
        drifted_artifacts=(
            exc.failure.drifted_artifacts if isinstance(exc, GeometryAuthoringInputDrift) else ()
        ),
    )
    return GeometryAuthoringProviderReceipt(
        provider=provider,
        operation=operation,
        request_digest=request_digest,
        disposition="failed",
        failure=failure,
    )


class DelegatedGeometryAuthoringProvider:
    """Canonical adapter for one explicitly configured external authoring worker."""

    def __init__(
        self,
        *,
        connector: DelegatedAuthoringHttpConnector,
        artifact_root: str | Path,
        rights_assertion: str,
    ) -> None:
        if not rights_assertion.strip():
            raise ValueError("Delegated authoring provider requires a rights assertion")
        validated_rights = GeometryRightsAssertion(assertion=rights_assertion)
        self._connector = connector
        self._provider_label = connector.provider_label
        raw_artifact_root = Path(os.path.abspath(Path(artifact_root).expanduser()))
        resolved_artifact_root = raw_artifact_root.resolve()
        if raw_artifact_root != resolved_artifact_root:
            raise ValueError("Delegated authoring artifact root must not be a symlink")
        self._artifact_root = resolved_artifact_root
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        self._rights_assertion = validated_rights.assertion
        low_capabilities = connector.capabilities()
        self._identity = GeometryAuthoringProviderIdentity(
            provider_id=low_capabilities.provider_id,
            provider_version="connector-v1",
        )
        self._formats = low_capabilities.formats

    @property
    def provider_id(self) -> str:
        return self._identity.provider_id

    def capabilities(self) -> GeometryAuthoringCapabilityReport:
        low_capabilities = self._connector.capabilities()
        modalities: list[Literal["text", "image", "text_image", "existing_source"]] = []
        if low_capabilities.text:
            modalities.append("text")
        if low_capabilities.image:
            modalities.append("image")
        if low_capabilities.text and low_capabilities.image:
            modalities.append("text_image")
        if low_capabilities.revision or low_capabilities.export:
            modalities.append("existing_source")
        feature_flags: tuple[tuple[GeometryAuthoringFeature, bool], ...] = (
            ("image_conditioning", low_capabilities.image),
            ("immutable_revisions", low_capabilities.revision),
            ("semantic_parameters", self._connector.supports_semantic_parameters),
            ("parameter_definitions", low_capabilities.parameter_definitions),
            ("parameter_families", low_capabilities.max_family_variants > 0),
            ("semantic_parts", low_capabilities.semantic_parts),
            ("provider_assertions", low_capabilities.provider_assertions),
            ("native_source", low_capabilities.native_source),
            ("multi_format_output", len(self._formats) > 1),
        )
        return GeometryAuthoringCapabilityReport(
            provider=self._identity,
            operations=(
                "generate",
                *(("revise",) if low_capabilities.revision else ()),
                *(("export",) if low_capabilities.export else ()),
            ),
            input_modalities=tuple(modalities),
            output_formats=self._formats,
            supports_semantic_parameters=self._connector.supports_semantic_parameters,
            returns_native_source=low_capabilities.native_source,
            features=tuple(feature for feature, enabled in feature_flags if enabled),
            max_family_variants=low_capabilities.max_family_variants,
            max_reference_artifacts=8 if low_capabilities.image else 0,
            max_prompt_characters=32_768 if low_capabilities.text else 0,
        )

    def generate(
        self,
        request: GeometryAuthoringRequest,
    ) -> GeometryAuthoringProviderReceipt:
        return self._author_generate(request)

    def revise(
        self,
        request: GeometryAuthoringRevisionRequest,
    ) -> GeometryAuthoringProviderReceipt:
        request_digest = geometry_authoring_request_digest(request)
        output_dir: Path | None = None
        try:
            if request.source_bundle.producer.provider_id != self.provider_id:
                raise UnsupportedCapabilityError(
                    f"{self._provider_label} revisions require a source revision produced "
                    "by the same worker",
                    provider_id=self.provider_id,
                )
            resolve_semantic_parameter_overrides(
                request.source_bundle.parameters,
                request.parameter_overrides,
                allow_undeclared=not request.source_bundle.parameters,
            )
            low_request = self._low_request(
                request_id=request.request_id,
                operation="revise",
                prompt=request.instructions,
                references=request.references,
                parameters=request.parameter_overrides,
                requested_formats=request.requested_formats,
                target_profile=None,
                prior_source_revision=request.source_bundle.source_revision,
            )
            output_dir = self._output_dir(request.request_id, request_digest)
            materialized = self._connector.revise(
                low_request,
                output_dir=output_dir,
            )
            bundle = canonicalize_materialized_bundle(
                materialized,
                request_digest=request_digest,
                rights_assertion=self._rights_assertion,
                license_identifier=request.source_bundle.rights.license_identifier,
                parent_bundle_id=request.source_bundle.bundle_id,
                input_artifacts=(
                    *(item.artifact for item in request.source_bundle.representations),
                    *(item.artifact for item in request.references),
                ),
                parameters=request.parameter_overrides,
                parameter_definitions=request.source_bundle.parameters,
                parts=request.source_bundle.parts,
                upstream_edit_uri=request.source_bundle.provenance.upstream_edit_uri,
            )
            return GeometryAuthoringProviderReceipt(
                provider=bundle.producer,
                operation="revise",
                request_digest=request_digest,
                disposition="succeeded",
                source_bundle=bundle,
            )
        except Exception as exc:
            if output_dir is not None:
                _discard_owned_output_directory(self._artifact_root, output_dir)
            return _failed_receipt(
                provider=self._identity,
                operation="revise",
                request_digest=request_digest,
                exc=exc,
            )

    def export(
        self,
        request: GeometryAuthoringExportRequest,
    ) -> GeometryAuthoringProviderReceipt:
        request_digest = geometry_authoring_request_digest(request)
        output_dir: Path | None = None
        try:
            if request.source_bundle.producer.provider_id != self.provider_id:
                raise UnsupportedCapabilityError(
                    f"{self._provider_label} exports require a source revision produced "
                    "by the same worker",
                    provider_id=self.provider_id,
                )
            low_request = self._low_request(
                request_id=request.request_id,
                operation="export",
                prompt=None,
                references=(),
                parameters=(),
                requested_formats=request.requested_formats,
                target_profile=None,
                prior_source_revision=request.source_bundle.source_revision,
            )
            output_dir = self._output_dir(request.request_id, request_digest)
            materialized = self._connector.export(
                low_request,
                output_dir=output_dir,
            )
            if materialized.source_revision != request.source_bundle.source_revision:
                raise InvalidProviderResponseError(
                    f"{self._provider_label} export changed the immutable source revision",
                    provider_id=self.provider_id,
                )
            bundle = canonicalize_materialized_bundle(
                materialized,
                request_digest=request_digest,
                rights_assertion=self._rights_assertion,
                license_identifier=request.source_bundle.rights.license_identifier,
                parent_bundle_id=request.source_bundle.bundle_id,
                input_artifacts=tuple(
                    item.artifact for item in request.source_bundle.representations
                ),
                parameter_definitions=request.source_bundle.parameters,
                parts=request.source_bundle.parts,
                upstream_edit_uri=request.source_bundle.provenance.upstream_edit_uri,
            )
            return GeometryAuthoringProviderReceipt(
                provider=bundle.producer,
                operation="export",
                request_digest=request_digest,
                disposition="succeeded",
                source_bundle=bundle,
            )
        except Exception as exc:
            if output_dir is not None:
                _discard_owned_output_directory(self._artifact_root, output_dir)
            return _failed_receipt(
                provider=self._identity,
                operation="export",
                request_digest=request_digest,
                exc=exc,
            )

    def _author_generate(
        self,
        request: GeometryAuthoringRequest,
    ) -> GeometryAuthoringProviderReceipt:
        request_digest = geometry_authoring_request_digest(request)
        output_dir: Path | None = None
        try:
            low_request = self._low_request(
                request_id=request.request_id,
                operation="generate",
                prompt=request.prompt,
                references=request.references,
                parameters=request.parameters,
                requested_formats=request.requested_formats,
                target_profile=request.target_profile,
                prior_source_revision=None,
            )
            output_dir = self._output_dir(request.request_id, request_digest)
            materialized = self._connector.generate(
                low_request,
                output_dir=output_dir,
            )
            bundle = canonicalize_materialized_bundle(
                materialized,
                request_digest=request_digest,
                rights_assertion=self._rights_assertion,
                input_artifacts=tuple(item.artifact for item in request.references),
                parameters=request.parameters,
            )
            return GeometryAuthoringProviderReceipt(
                provider=bundle.producer,
                operation="generate",
                request_digest=request_digest,
                disposition="succeeded",
                source_bundle=bundle,
            )
        except Exception as exc:
            if output_dir is not None:
                _discard_owned_output_directory(self._artifact_root, output_dir)
            return _failed_receipt(
                provider=self._identity,
                operation="generate",
                request_digest=request_digest,
                exc=exc,
            )

    def _low_request(
        self,
        *,
        request_id: str,
        operation: Literal["generate", "revise", "export"],
        prompt: str | None,
        references: tuple[GeometryAuthoringReference, ...],
        parameters: tuple[GeometryAuthoringParameterValue, ...],
        requested_formats: tuple[str, ...],
        target_profile: str | None,
        prior_source_revision: str | None,
    ) -> AuthoringRequest:
        images: list[InlineInputArtifact] = []
        low_capabilities = self._connector.capabilities()
        if prompt is not None and not low_capabilities.text:
            raise UnsupportedCapabilityError(
                f"{self._provider_label} does not advertise text input",
                provider_id=self.provider_id,
            )
        if parameters and not self._connector.supports_semantic_parameters:
            raise UnsupportedCapabilityError(
                f"{self._provider_label} does not advertise semantic parameters",
                provider_id=self.provider_id,
            )
        for reference in references:
            if reference.kind != "image":
                raise UnsupportedCapabilityError(
                    f"{self._provider_label} accepts image references only",
                    provider_id=self.provider_id,
                )
            if not low_capabilities.image:
                raise UnsupportedCapabilityError(
                    f"{self._provider_label} does not advertise image input",
                    provider_id=self.provider_id,
                )
            path = Path(reference.artifact.path)
            content = read_regular_file(path, label="authoring reference image")
            if (
                len(content) != reference.artifact.size_bytes
                or hashlib.sha256(content).hexdigest() != reference.artifact.sha256
            ):
                raise ArtifactIntegrityError(
                    f"{self._provider_label} reference image changed after request admission",
                    provider_id=self.provider_id,
                )
            images.append(
                InlineInputArtifact.from_bytes(
                    filename=f"{reference.reference_id}{path.suffix.lower()}",
                    media_type=reference.media_type,
                    content=content,
                )
            )
        requested = tuple(cast(GeometryFormat, value) for value in requested_formats)
        return AuthoringRequest(
            request_id=request_id,
            operation=operation,
            prompt=prompt,
            images=tuple(images),
            target_profile=target_profile,
            parameters={item.name: item.value for item in parameters},
            parameter_units={item.name: item.unit for item in parameters if item.unit is not None},
            target_formats=requested,
            prior_source_revision=prior_source_revision,
        )

    def _output_dir(self, request_id: str, request_digest: str) -> Path:
        safe_request_id = request_id.replace(":", "_")
        return self._artifact_root / (f"{safe_request_id}-{request_digest[:12]}-{uuid.uuid4().hex}")


class Build123dGeometryAuthoringProvider(DelegatedGeometryAuthoringProvider):
    """Compatibility adapter for an administrator-isolated Build123d worker."""

    def __init__(
        self,
        *,
        connector: Build123dHttpConnector,
        artifact_root: str | Path,
        rights_assertion: str,
    ) -> None:
        super().__init__(
            connector=connector,
            artifact_root=artifact_root,
            rights_assertion=rights_assertion,
        )


__all__ = [
    "Build123dGeometryAuthoringProvider",
    "DelegatedGeometryAuthoringProvider",
    "canonicalize_materialized_bundle",
]
