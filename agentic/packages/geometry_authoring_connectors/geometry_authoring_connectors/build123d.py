# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote-only Build123d authoring connector for the Geometry Agent."""

from __future__ import annotations

from ._http import HttpTransport
from .delegated import DelegatedAuthoringHttpConnector, validate_delegated_outputs
from .models import (
    AuthoringRequest,
    GeometryFormat,
    WireSourceBundle,
)

BUILD123D_PROVIDER_ID = "build123d-http"


def validate_requested_outputs(
    request: AuthoringRequest,
    bundle: WireSourceBundle,
) -> None:
    validate_delegated_outputs(
        request,
        bundle,
        provider_label="Build123d worker",
        returns_native_source=True,
    )


class Build123dHttpConnector(DelegatedAuthoringHttpConnector):
    """Call one administrator-configured isolated Build123d worker endpoint."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        bearer_token: str | None = None,
        endpoint_alias: str = "build123d-worker",
        provider_id: str = BUILD123D_PROVIDER_ID,
        supported_formats: tuple[GeometryFormat, ...] = ("step",),
        supports_text: bool = True,
        supports_image: bool = False,
        supports_revision: bool = False,
        supports_export: bool = False,
        supports_semantic_parameters: bool = False,
        supports_parameter_definitions: bool = False,
        supports_semantic_parts: bool = False,
        supports_provider_assertions: bool = False,
        max_family_variants: int = 0,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 300.0,
        transport: HttpTransport | None = None,
    ) -> None:
        super().__init__(
            provider_id=provider_id,
            provider_label="Build123d worker",
            endpoint_alias=endpoint_alias,
            endpoint_url=endpoint_url,
            bearer_token=bearer_token,
            supported_formats=supported_formats,
            supports_text=supports_text,
            supports_image=supports_image,
            supports_revision=supports_revision,
            supports_export=supports_export,
            supports_semantic_parameters=supports_semantic_parameters,
            supports_parameter_definitions=supports_parameter_definitions,
            supports_semantic_parts=supports_semantic_parts,
            supports_provider_assertions=supports_provider_assertions,
            max_family_variants=max_family_variants,
            returns_native_source=True,
            connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds,
            transport=transport,
        )
