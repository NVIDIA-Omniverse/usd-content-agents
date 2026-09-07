# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit registry for the public geometry authoring provider contract."""

from __future__ import annotations

import asyncio
import inspect

from geometry_authoring_contracts import (
    GeometryAuthoringExportRequest,
    GeometryAuthoringFailure,
    GeometryAuthoringInvalidResponse,
    GeometryAuthoringOperation,
    GeometryAuthoringProvider,
    GeometryAuthoringProviderReceipt,
    GeometryAuthoringRequest,
    GeometryAuthoringRevisionRequest,
    GeometrySourceBundle,
    geometry_authoring_request_digest,
    geometry_deliverable_representation_formats,
    geometry_requested_formats_satisfied,
    resolve_semantic_parameter_overrides,
    validate_geometry_source_bundle_identity,
    validate_returned_parameter_state,
    validate_returned_parameter_values,
)
from pydantic import ValidationError

from .models import ProviderInfo


def _invalid_provider_response(
    *,
    provider_id: str,
    operation: GeometryAuthoringOperation,
    summary: str,
    receipt: GeometryAuthoringProviderReceipt | None = None,
) -> GeometryAuthoringInvalidResponse:
    return GeometryAuthoringInvalidResponse(
        GeometryAuthoringFailure(
            provider_id=provider_id,
            operation=operation,
            code="invalid_response",
            summary=summary,
            retryable=False,
        ),
        receipt=receipt,
    )


def _coerce_provider_receipt(
    raw: object,
    *,
    provider_id: str,
    operation: GeometryAuthoringOperation,
) -> GeometryAuthoringProviderReceipt:
    if isinstance(raw, GeometryAuthoringProviderReceipt):
        return raw
    try:
        return GeometryAuthoringProviderReceipt.model_validate(raw)
    except ValidationError as exc:
        raise _invalid_provider_response(
            provider_id=provider_id,
            operation=operation,
            summary="Provider returned an invalid typed authoring receipt.",
        ) from exc


def _validate_provider_bundle(
    bundle: GeometrySourceBundle,
    *,
    provider_id: str,
    operation: GeometryAuthoringOperation,
    request_digest: str,
    parent_bundle_id: str | None,
    requested_formats: tuple[str, ...],
    receipt: GeometryAuthoringProviderReceipt,
) -> None:
    try:
        validate_geometry_source_bundle_identity(bundle)
    except ValueError as exc:
        raise _invalid_provider_response(
            provider_id=provider_id,
            operation=operation,
            summary="Provider returned a non-canonical source bundle.",
            receipt=receipt,
        ) from exc
    available_formats = geometry_deliverable_representation_formats(
        bundle.representations
    )
    if (
        bundle.provenance.request_digest != request_digest
        or bundle.provenance.parent_bundle_id != parent_bundle_id
        or not geometry_requested_formats_satisfied(
            requested_formats, available_formats
        )
    ):
        raise _invalid_provider_response(
            provider_id=provider_id,
            operation=operation,
            summary="Provider source lineage or requested exports are invalid.",
            receipt=receipt,
        )


class ProviderRegistry:
    """Register explicitly configured providers without fallback selection."""

    def __init__(self) -> None:
        self._providers: dict[str, GeometryAuthoringProvider] = {}

    def register(self, provider: GeometryAuthoringProvider) -> None:
        capabilities = provider.capabilities()
        provider_id = provider.provider_id
        if capabilities.provider.provider_id != provider_id:
            raise ValueError("Provider identity differs from its capability report")
        if provider_id in self._providers:
            raise ValueError(
                f"Geometry authoring provider already registered: {provider_id}"
            )
        self._providers[provider_id] = provider

    def clear(self) -> None:
        self._providers.clear()

    def get(self, provider_id: str) -> GeometryAuthoringProvider | None:
        return self._providers.get(provider_id)

    def list(self) -> list[ProviderInfo]:
        return [
            ProviderInfo(
                provider_id=provider_id,
                capabilities=self._providers[provider_id]
                .capabilities()
                .model_dump(mode="json"),
            )
            for provider_id in sorted(self._providers)
        ]

    async def generate(
        self,
        provider_id: str,
        request: GeometryAuthoringRequest,
    ) -> GeometryAuthoringProviderReceipt:
        provider = self.get(provider_id)
        if provider is None:
            raise LookupError(provider_id)
        method = provider.generate
        if inspect.iscoroutinefunction(method):
            raw = await method(request)
        else:
            raw = await asyncio.to_thread(method, request)
        if inspect.isawaitable(raw):
            raw = await raw
        raw = _coerce_provider_receipt(
            raw,
            provider_id=provider_id,
            operation="generate",
        )
        request_digest = geometry_authoring_request_digest(request)
        if (
            raw.provider.provider_id != provider_id
            or raw.operation != "generate"
            or raw.request_digest != request_digest
        ):
            raise _invalid_provider_response(
                provider_id=provider_id,
                operation="generate",
                summary="Provider receipt differs from the selected generation request.",
                receipt=raw,
            )
        if raw.source_bundle is not None:
            _validate_provider_bundle(
                raw.source_bundle,
                provider_id=provider_id,
                operation="generate",
                request_digest=request_digest,
                parent_bundle_id=None,
                requested_formats=request.requested_formats,
                receipt=raw,
            )
            try:
                validate_returned_parameter_values(
                    raw.source_bundle.parameters,
                    request.parameters,
                )
            except ValueError as exc:
                raise _invalid_provider_response(
                    provider_id=provider_id,
                    operation="generate",
                    summary="Provider generation changed requested semantic parameters.",
                    receipt=raw,
                ) from exc
        return raw

    async def revise(
        self,
        provider_id: str,
        request: GeometryAuthoringRevisionRequest,
    ) -> GeometryAuthoringProviderReceipt:
        provider = self.get(provider_id)
        if provider is None:
            raise LookupError(provider_id)
        method = provider.revise
        if inspect.iscoroutinefunction(method):
            raw = await method(request)
        else:
            raw = await asyncio.to_thread(method, request)
        if inspect.isawaitable(raw):
            raw = await raw
        raw = _coerce_provider_receipt(
            raw,
            provider_id=provider_id,
            operation="revise",
        )
        request_digest = geometry_authoring_request_digest(request)
        if (
            raw.provider.provider_id != provider_id
            or raw.operation != "revise"
            or raw.request_digest != request_digest
        ):
            raise _invalid_provider_response(
                provider_id=provider_id,
                operation="revise",
                summary="Provider receipt differs from the selected revision request.",
                receipt=raw,
            )
        if raw.source_bundle is not None:
            _validate_provider_bundle(
                raw.source_bundle,
                provider_id=provider_id,
                operation="revise",
                request_digest=request_digest,
                parent_bundle_id=request.source_bundle.bundle_id,
                requested_formats=request.requested_formats,
                receipt=raw,
            )
            if (
                raw.source_bundle.source_revision
                == request.source_bundle.source_revision
            ):
                raise _invalid_provider_response(
                    provider_id=provider_id,
                    operation="revise",
                    summary="Provider revision did not create a new immutable revision.",
                    receipt=raw,
                )
            try:
                expected_parameters = resolve_semantic_parameter_overrides(
                    request.source_bundle.parameters,
                    request.parameter_overrides,
                    allow_undeclared=not request.source_bundle.parameters,
                )
                validate_returned_parameter_state(
                    raw.source_bundle.parameters,
                    expected_parameters,
                )
            except ValueError as exc:
                raise _invalid_provider_response(
                    provider_id=provider_id,
                    operation="revise",
                    summary="Provider revision changed the declared semantic parameter family.",
                    receipt=raw,
                ) from exc
        return raw

    async def export(
        self,
        provider_id: str,
        request: GeometryAuthoringExportRequest,
    ) -> GeometryAuthoringProviderReceipt:
        provider = self.get(provider_id)
        if provider is None:
            raise LookupError(provider_id)
        method = provider.export
        if inspect.iscoroutinefunction(method):
            raw = await method(request)
        else:
            raw = await asyncio.to_thread(method, request)
        if inspect.isawaitable(raw):
            raw = await raw
        raw = _coerce_provider_receipt(
            raw,
            provider_id=provider_id,
            operation="export",
        )
        request_digest = geometry_authoring_request_digest(request)
        if (
            raw.provider.provider_id != provider_id
            or raw.operation != "export"
            or raw.request_digest != request_digest
        ):
            raise _invalid_provider_response(
                provider_id=provider_id,
                operation="export",
                summary="Provider receipt differs from the selected export request.",
                receipt=raw,
            )
        if raw.source_bundle is not None:
            _validate_provider_bundle(
                raw.source_bundle,
                provider_id=provider_id,
                operation="export",
                request_digest=request_digest,
                parent_bundle_id=request.source_bundle.bundle_id,
                requested_formats=request.requested_formats,
                receipt=raw,
            )
            if (
                raw.source_bundle.source_revision
                != request.source_bundle.source_revision
            ):
                raise _invalid_provider_response(
                    provider_id=provider_id,
                    operation="export",
                    summary="Provider export changed the immutable source revision.",
                    receipt=raw,
                )
            try:
                validate_returned_parameter_state(
                    raw.source_bundle.parameters,
                    request.source_bundle.parameters,
                )
            except ValueError as exc:
                raise _invalid_provider_response(
                    provider_id=provider_id,
                    operation="export",
                    summary="Provider export changed immutable semantic parameters.",
                    receipt=raw,
                ) from exc
        return raw


__all__ = ["ProviderRegistry"]
