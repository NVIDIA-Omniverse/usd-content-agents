# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free HTTP delegation for externally operated authoring systems."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from ._artifacts import materialize_wire_bundle
from ._http import BoundedHttpClient, HttpTransport
from .errors import (
    ConnectorConfigurationError,
    InvalidProviderResponseError,
    UnsupportedCapabilityError,
)
from .models import (
    AuthoringCapabilities,
    AuthoringRequest,
    GeometryFormat,
    MaterializedWireSourceBundle,
    WireSourceBundle,
)

FORGECAD_AUTHORING_PROVIDER_ID = "forgecad-http"

_FORMAT_SUFFIXES: dict[GeometryFormat, frozenset[str]] = {
    "step": frozenset({".step", ".stp"}),
    "stl": frozenset({".stl"}),
    "obj": frozenset({".obj"}),
    "ply": frozenset({".ply"}),
    "glb": frozenset({".glb"}),
    "gltf": frozenset({".gltf"}),
    "3mf": frozenset({".3mf"}),
    "usd": frozenset({".usd", ".usda", ".usdc"}),
    "usda": frozenset({".usda"}),
    "usdc": frozenset({".usdc"}),
}


def validate_delegated_outputs(
    request: AuthoringRequest,
    bundle: WireSourceBundle,
    *,
    provider_label: str,
    returns_native_source: bool,
) -> None:
    """Require every advertised output and any promised native source."""

    geometry_names = {
        Path(item.filename).suffix.lower()
        for item in bundle.artifacts
        if item.role in {"cad_geometry", "render_geometry", "mesh_geometry", "collision_geometry"}
    }
    missing = [
        format_name
        for format_name in request.target_formats
        if not geometry_names.intersection(_FORMAT_SUFFIXES[format_name])
    ]
    if missing:
        raise InvalidProviderResponseError(
            f"{provider_label} omitted requested geometry formats: " + ", ".join(sorted(missing)),
            provider_id=bundle.provider_id,
        )
    if returns_native_source and not any(item.role == "native_source" for item in bundle.artifacts):
        raise InvalidProviderResponseError(
            f"{provider_label} omitted the advertised native source",
            provider_id=bundle.provider_id,
        )
    if request.operation == "revise" and bundle.source_revision == request.prior_source_revision:
        raise InvalidProviderResponseError(
            f"{provider_label} revision did not create a new immutable revision",
            provider_id=bundle.provider_id,
        )
    if request.operation == "export" and bundle.source_revision != request.prior_source_revision:
        raise InvalidProviderResponseError(
            f"{provider_label} export changed the immutable source revision",
            provider_id=bundle.provider_id,
        )


class DelegatedAuthoringHttpConnector:
    """Call one explicitly configured authoring worker without its SDK or runtime."""

    def __init__(
        self,
        *,
        provider_id: str,
        endpoint_url: str,
        endpoint_alias: str,
        provider_label: str,
        bearer_token: str | None = None,
        supported_formats: tuple[GeometryFormat, ...] = ("step",),
        supports_text: bool = True,
        supports_image: bool = False,
        supports_revision: bool = True,
        supports_export: bool = False,
        supports_semantic_parameters: bool = False,
        supports_parameter_definitions: bool = False,
        supports_semantic_parts: bool = False,
        supports_provider_assertions: bool = False,
        max_family_variants: int = 0,
        returns_native_source: bool = False,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 300.0,
        transport: HttpTransport | None = None,
    ) -> None:
        label = provider_label.strip()
        if (
            not label
            or len(label) > 128
            or not label.isascii()
            or any(ord(character) < 32 or ord(character) == 127 for character in label)
        ):
            raise ConnectorConfigurationError(
                "Delegated provider label must be a bounded printable ASCII value"
            )
        if not supported_formats or len(supported_formats) != len(set(supported_formats)):
            raise ConnectorConfigurationError(
                f"{label} supported formats must be a non-empty unique tuple"
            )
        if not (supports_text or supports_image):
            raise ConnectorConfigurationError(f"{label} must advertise text or image input")
        if supports_parameter_definitions and not supports_semantic_parameters:
            raise ConnectorConfigurationError(
                f"{label} parameter definitions require semantic parameter support"
            )
        if isinstance(max_family_variants, bool) or not 0 <= max_family_variants <= 64:
            raise ConnectorConfigurationError(
                f"{label} maximum family variants must be an integer in [0, 64]"
            )
        if max_family_variants and not (supports_revision and supports_parameter_definitions):
            raise ConnectorConfigurationError(
                f"{label} parameter families require revision and parameter definitions"
            )
        self._provider_id = provider_id
        self._provider_label = label
        self._supported_formats = supported_formats
        self._supports_text = supports_text
        self._supports_image = supports_image
        self._supports_revision = supports_revision
        self._supports_export = supports_export
        self._supports_semantic_parameters = supports_semantic_parameters
        self._supports_parameter_definitions = supports_parameter_definitions
        self._supports_semantic_parts = supports_semantic_parts
        self._supports_provider_assertions = supports_provider_assertions
        self._max_family_variants = max_family_variants
        self._returns_native_source = returns_native_source
        self._http = BoundedHttpClient(
            provider_id=provider_id,
            endpoint_alias=endpoint_alias,
            base_url=endpoint_url,
            bearer_token=bearer_token,
            connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds,
            transport=transport,
        )

    @property
    def provider_label(self) -> str:
        return self._provider_label

    @property
    def supports_semantic_parameters(self) -> bool:
        return self._supports_semantic_parameters

    def capabilities(self) -> AuthoringCapabilities:
        return AuthoringCapabilities(
            provider_id=self._provider_id,
            text=self._supports_text,
            image=self._supports_image,
            revision=self._supports_revision,
            export=self._supports_export,
            native_source=self._returns_native_source,
            formats=self._supported_formats,
            parameter_definitions=self._supports_parameter_definitions,
            semantic_parts=self._supports_semantic_parts,
            provider_assertions=self._supports_provider_assertions,
            max_family_variants=self._max_family_variants,
        )

    def generate(
        self,
        request: AuthoringRequest,
        *,
        output_dir: str | Path,
    ) -> MaterializedWireSourceBundle:
        if request.operation != "generate":
            raise UnsupportedCapabilityError(
                f"{self._provider_label} generate requires a generation request",
                provider_id=self._provider_id,
            )
        return self._author(request, output_dir=output_dir)

    def revise(
        self,
        request: AuthoringRequest,
        *,
        output_dir: str | Path,
    ) -> MaterializedWireSourceBundle:
        if not self._supports_revision:
            raise UnsupportedCapabilityError(
                f"{self._provider_label} does not advertise revision",
                provider_id=self._provider_id,
            )
        if request.operation != "revise":
            raise UnsupportedCapabilityError(
                f"{self._provider_label} revise requires a revision request",
                provider_id=self._provider_id,
            )
        return self._author(request, output_dir=output_dir)

    def export(
        self,
        request: AuthoringRequest,
        *,
        output_dir: str | Path,
    ) -> MaterializedWireSourceBundle:
        if not self._supports_export:
            raise UnsupportedCapabilityError(
                f"{self._provider_label} does not advertise separate export",
                provider_id=self._provider_id,
            )
        if request.operation != "export":
            raise UnsupportedCapabilityError(
                f"{self._provider_label} export requires an export request",
                provider_id=self._provider_id,
            )
        return self._author(request, output_dir=output_dir)

    def _author(
        self,
        request: AuthoringRequest,
        *,
        output_dir: str | Path,
    ) -> MaterializedWireSourceBundle:
        if request.prompt is not None and not self._supports_text:
            raise UnsupportedCapabilityError(
                f"{self._provider_label} does not advertise text input",
                provider_id=self._provider_id,
            )
        if request.images and not self._supports_image:
            raise UnsupportedCapabilityError(
                f"{self._provider_label} does not advertise image input",
                provider_id=self._provider_id,
            )
        unsupported = set(request.target_formats).difference(self._supported_formats)
        if unsupported:
            raise UnsupportedCapabilityError(
                f"{self._provider_label} does not advertise requested formats: "
                + ", ".join(sorted(unsupported)),
                provider_id=self._provider_id,
            )
        payload = self._http.request_json(
            "POST",
            payload=request.model_dump(mode="json"),
            # Native sources are base64-inlined in JSON, so this bounded ceiling
            # includes encoding overhead above the binary artifact limit.
            max_bytes=768 * 1024 * 1024,
        )
        try:
            bundle = WireSourceBundle.model_validate(payload)
        except ValidationError as exc:
            raise InvalidProviderResponseError(
                f"{self._provider_label} returned an invalid geometry.source.v1 bundle",
                provider_id=self._provider_id,
            ) from exc
        if bundle.provider_id != self._provider_id:
            raise InvalidProviderResponseError(
                f"{self._provider_label} response identifies another provider",
                provider_id=self._provider_id,
            )
        validate_delegated_outputs(
            request,
            bundle,
            provider_label=self._provider_label,
            returns_native_source=self._returns_native_source,
        )
        return materialize_wire_bundle(bundle, output_dir)


__all__ = [
    "FORGECAD_AUTHORING_PROVIDER_ID",
    "DelegatedAuthoringHttpConnector",
    "validate_delegated_outputs",
]
