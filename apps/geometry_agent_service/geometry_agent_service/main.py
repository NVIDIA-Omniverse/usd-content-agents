# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated public Geometry Agent service.

The API accepts only uploaded, content-addressed artifacts. It never accepts
server-local paths, fetches caller-supplied URLs, or executes uploaded source.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import secrets
import stat
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Any, NamedTuple

import uvicorn
from content_agent_workflows.common.artifacts import read_contained_artifact
from content_agent_workflows.geometry import (
    GeometryWorkflowInput,
    run_geometry_workflow,
)
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from geometry_authoring_contracts import (
    GeometryArtifactBinding,
    GeometryAuthoringExportRequest,
    GeometryAuthoringFailure,
    GeometryAuthoringFamilyRequest,
    GeometryAuthoringInvalidResponse,
    GeometryAuthoringParameterValue,
    GeometryAuthoringProviderError,
    GeometryAuthoringProviderReceipt,
    GeometryAuthoringReference,
    GeometryAuthoringRequest,
    GeometryAuthoringRevisionRequest,
    GeometryAuthoringVariant,
    GeometryCoordinateSystem,
    GeometryRepresentationBinding,
    GeometryRightsAssertion,
    GeometrySourceBundle,
    GeometrySourceProvenance,
    geometry_source_bundle_id,
    resolve_semantic_parameter_overrides,
    validate_geometry_source_bundle_identity,
    validate_returned_parameter_state,
)
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import __version__
from .config import AuthoringFormat, settings
from .models import (
    ArtifactRecord,
    ArtifactRole,
    GeometryExportRequest,
    GeometryFamilyRequest,
    GeometryGenerationRequest,
    GeometryImmutableExportRequest,
    GeometryParameterInput,
    GeometryRevisionRequest,
    GeometryRunRequest,
    HealthResponse,
    InfoResponse,
    JobRecord,
    ProviderListResponse,
    ServiceError,
    SourceRecord,
)
from .providers import ProviderRegistry
from .storage import JobStore, StorageError, WorkspaceStorage

logger = logging.getLogger(__name__)

_PUBLIC_ROUTES = [
    "/api/geometry/sources",
    "/api/geometry/generations",
    "/api/geometry/revisions",
    "/api/geometry/families",
    "/api/geometry/provider-exports",
    "/api/geometry/exports",
    "/api/geometry/runs",
    "/api/geometry/jobs/{job_id}",
    "/api/geometry/providers",
    "/api/geometry/artifacts/{artifact_id}",
]

_api_key_header = APIKeyHeader(
    name="X-Geometry-Agent-Service-Key",
    auto_error=False,
)
_bearer_header = HTTPBearer(auto_error=False)

authoring_providers = ProviderRegistry()
_runtime_storage: WorkspaceStorage | None = None
_runtime_jobs: JobStore | None = None
_runtime_root: str | None = None
_runtime_instance_id: str | None = None


class RequestBodyTooLargeError(Exception):
    """Raised by the receive wrapper when a streamed body crosses its limit."""


class RequestBodyLimitMiddleware:
    """Enforce the outer request limit for Content-Length and chunked requests.

    ``Settings._validate_limits`` keeps this above the role-specific streaming
    upload limits enforced by ``WorkspaceStorage.store_upload``.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except ValueError:
                await _body_limit_response("Content-Length must be an integer.")(
                    scope, receive, send
                )
                return
            if content_length < 0:
                await _body_limit_response("Content-Length must not be negative.")(
                    scope, receive, send
                )
                return
            if content_length > settings.max_request_bytes:
                await _body_limit_response(
                    "Request body exceeds the configured byte limit."
                )(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > settings.max_request_bytes:
                    raise RequestBodyTooLargeError
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLargeError:
            await _body_limit_response(
                "Request body exceeds the configured byte limit."
            )(scope, receive, send)


def _body_limit_response(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={
            "detail": ServiceError(
                code="request_body_too_large",
                message=message,
            ).model_dump(mode="json")
        },
    )


def _validate_service_security(_host: str) -> None:
    api_key = settings.api_key
    if api_key is not None and not api_key.strip():
        raise RuntimeError("GEOMETRY_AGENT_SERVICE_API_KEY must not be blank")
    if api_key:
        return
    raise RuntimeError(
        "Geometry Agent service requires GEOMETRY_AGENT_SERVICE_API_KEY."
    )


def _require_access(
    supplied_key: Annotated[str | None, Depends(_api_key_header)] = None,
    bearer: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_bearer_header),
    ] = None,
) -> None:
    expected = settings.api_key
    if expected:
        token = supplied_key or (bearer.credentials if bearer else None)
        if token is None or not secrets.compare_digest(token, expected):
            raise HTTPException(
                status_code=401,
                detail=ServiceError(
                    code="authentication_required",
                    message="A valid Geometry Agent service API key is required.",
                ).model_dump(mode="json"),
                headers={"WWW-Authenticate": "Bearer"},
            )
        return
    raise HTTPException(
        status_code=503,
        detail=ServiceError(
            code="service_authentication_not_configured",
            message="Geometry Agent service authentication is not configured.",
            retryable=True,
        ).model_dump(mode="json"),
    )


def _runtime() -> tuple[WorkspaceStorage, JobStore]:
    global _runtime_instance_id, _runtime_jobs, _runtime_root, _runtime_storage
    root = str(Path(settings.workspace_root).expanduser().resolve())
    if (
        _runtime_storage is None
        or _runtime_jobs is None
        or root != _runtime_root
        or settings.instance_id != _runtime_instance_id
    ):
        _runtime_storage = WorkspaceStorage(settings)
        _runtime_storage.initialize()
        _runtime_jobs = JobStore(
            _runtime_storage,
            owner_instance_id=settings.instance_id,
            owner_lease_timeout_seconds=settings.job_owner_lease_timeout_seconds,
        )
        _runtime_root = root
        _runtime_instance_id = settings.instance_id
    return _runtime_storage, _runtime_jobs


def reset_runtime_state() -> None:
    """Clear cached storage handles; intended for process reconfiguration and tests."""

    global _runtime_instance_id, _runtime_jobs, _runtime_root, _runtime_storage
    _runtime_storage = None
    _runtime_jobs = None
    _runtime_root = None
    _runtime_instance_id = None


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    _validate_service_security(settings.host)
    storage, jobs = _runtime()
    _register_configured_providers(storage)
    recovered = jobs.recover_interrupted()
    if recovered:
        logger.warning("Marked %d interrupted Geometry Agent jobs as failed", recovered)
    lease_task = asyncio.create_task(_maintain_job_owner_lease(jobs))
    try:
        yield
    finally:
        lease_task.cancel()
        with suppress(asyncio.CancelledError):
            await lease_task


async def _maintain_job_owner_lease(jobs: JobStore) -> None:
    while True:
        await asyncio.sleep(settings.job_owner_heartbeat_seconds)
        try:
            jobs.heartbeat()
            recovered = jobs.recover_interrupted(include_current_owner=False)
            if recovered:
                logger.warning(
                    "Marked %d stale Geometry Agent jobs as failed", recovered
                )
        except Exception:
            logger.exception("Could not renew the Geometry Agent job-owner lease")


def _register_configured_providers(storage: WorkspaceStorage) -> None:
    """Register only administrator-configured reference providers."""

    if settings.delegated_authoring_endpoint_url:
        _register_delegated_authoring_provider(
            storage=storage,
            provider_id=settings.delegated_authoring_provider_id or "",
            provider_label=settings.delegated_authoring_provider_label,
            endpoint_url=settings.delegated_authoring_endpoint_url,
            bearer_token=(
                settings.delegated_authoring_bearer_token.get_secret_value()
                if settings.delegated_authoring_bearer_token is not None
                else None
            ),
            rights_assertion=settings.delegated_authoring_rights_assertion or "",
            supported_formats=settings.delegated_authoring_supported_formats,
            supports_text=True,
            supports_image=settings.delegated_authoring_supports_image,
            supports_revision=settings.delegated_authoring_supports_revision,
            supports_export=settings.delegated_authoring_supports_export,
            supports_semantic_parameters=False,
            supports_parameter_definitions=False,
            supports_semantic_parts=False,
            supports_provider_assertions=False,
            max_family_variants=0,
            returns_native_source=settings.delegated_authoring_returns_native_source,
            connect_timeout_seconds=10.0,
            read_timeout_seconds=300.0,
            artifact_directory="delegated-authoring",
        )

    if (
        settings.build123d_endpoint_url
        and authoring_providers.get("build123d-http") is None
    ):
        from geometry_authoring_connectors import (
            Build123dGeometryAuthoringProvider,
            Build123dHttpConnector,
        )

        authoring_providers.register(
            Build123dGeometryAuthoringProvider(
                connector=Build123dHttpConnector(
                    endpoint_url=settings.build123d_endpoint_url,
                    bearer_token=(
                        settings.build123d_bearer_token.get_secret_value()
                        if settings.build123d_bearer_token is not None
                        else None
                    ),
                    supported_formats=settings.build123d_supported_formats,
                    supports_text=settings.build123d_supports_text,
                    supports_image=settings.build123d_supports_image,
                    supports_revision=settings.build123d_supports_revision,
                    supports_export=settings.build123d_supports_export,
                    supports_semantic_parameters=(
                        settings.build123d_supports_semantic_parameters
                    ),
                    supports_parameter_definitions=(
                        settings.build123d_supports_parameter_definitions
                    ),
                    supports_semantic_parts=settings.build123d_supports_semantic_parts,
                    supports_provider_assertions=(
                        settings.build123d_supports_provider_assertions
                    ),
                    max_family_variants=settings.build123d_max_family_variants,
                ),
                artifact_root=storage.root / "provider-artifacts" / "build123d",
                rights_assertion=settings.build123d_rights_assertion or "",
            )
        )
    if settings.forgecad_authoring_endpoint_url:
        if not settings.forgecad_automated_use_authorized:
            raise RuntimeError(
                "ForgeCAD delegated authoring requires explicit automated-use authorization"
            )
        _register_delegated_authoring_provider(
            storage=storage,
            provider_id="forgecad-http",
            provider_label="ForgeCAD authoring worker",
            endpoint_url=settings.forgecad_authoring_endpoint_url,
            bearer_token=(
                settings.forgecad_authoring_bearer_token.get_secret_value()
                if settings.forgecad_authoring_bearer_token is not None
                else None
            ),
            rights_assertion=settings.forgecad_authoring_rights_assertion or "",
            supported_formats=settings.forgecad_authoring_supported_formats,
            supports_text=settings.forgecad_authoring_supports_text,
            supports_image=settings.forgecad_authoring_supports_image,
            supports_revision=settings.forgecad_authoring_supports_revision,
            supports_export=settings.forgecad_authoring_supports_export,
            supports_semantic_parameters=(
                settings.forgecad_authoring_supports_semantic_parameters
            ),
            supports_parameter_definitions=(
                settings.forgecad_authoring_supports_parameter_definitions
            ),
            supports_semantic_parts=settings.forgecad_authoring_supports_semantic_parts,
            supports_provider_assertions=(
                settings.forgecad_authoring_supports_provider_assertions
            ),
            max_family_variants=settings.forgecad_authoring_max_family_variants,
            returns_native_source=settings.forgecad_authoring_returns_native_source,
            connect_timeout_seconds=(
                settings.forgecad_authoring_connect_timeout_seconds
            ),
            read_timeout_seconds=settings.forgecad_authoring_read_timeout_seconds,
            artifact_directory="forgecad-authoring",
        )


def _register_delegated_authoring_provider(
    *,
    storage: WorkspaceStorage,
    provider_id: str,
    provider_label: str,
    endpoint_url: str,
    bearer_token: str | None,
    rights_assertion: str,
    supported_formats: tuple[AuthoringFormat, ...],
    supports_text: bool,
    supports_image: bool,
    supports_revision: bool,
    supports_export: bool,
    supports_semantic_parameters: bool,
    supports_parameter_definitions: bool,
    supports_semantic_parts: bool,
    supports_provider_assertions: bool,
    max_family_variants: int,
    returns_native_source: bool,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    artifact_directory: str,
) -> None:
    """Register one dependency-free remote worker under an exact provider ID."""

    if authoring_providers.get(provider_id) is not None:
        return
    from geometry_authoring_connectors import (
        DelegatedAuthoringHttpConnector,
        DelegatedGeometryAuthoringProvider,
    )

    authoring_providers.register(
        DelegatedGeometryAuthoringProvider(
            connector=DelegatedAuthoringHttpConnector(
                provider_id=provider_id,
                provider_label=provider_label,
                endpoint_alias=provider_id,
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
                returns_native_source=returns_native_source,
                connect_timeout_seconds=connect_timeout_seconds,
                read_timeout_seconds=read_timeout_seconds,
            ),
            artifact_root=storage.root / "provider-artifacts" / artifact_directory,
            rights_assertion=rights_assertion,
        )
    )


app = FastAPI(
    title="Geometry Agent Service",
    description=(
        "Public geometry source intake, provider orchestration, validation, repair, "
        "and evidence handoff over content-addressed artifacts."
    ),
    version=__version__,
    lifespan=_lifespan,
)
app.add_middleware(RequestBodyLimitMiddleware)


@app.get("/api/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    return HealthResponse(ok=True)


@app.get("/api/info", response_model=InfoResponse, tags=["service"])
def info() -> InfoResponse:
    return InfoResponse(routes=_PUBLIC_ROUTES)


@app.post(
    "/api/geometry/sources",
    response_model=SourceRecord,
    status_code=201,
    dependencies=[Depends(_require_access)],
    tags=["geometry"],
)
async def create_source(
    file: Annotated[UploadFile, File(description="Geometry source or reference image")],
    role: Annotated[ArtifactRole, Form()] = "geometry_source",
    archive_entrypoint: Annotated[
        str | None,
        Form(
            max_length=1024,
            description="Optional geometry member in a ZIP-based source container.",
        ),
    ] = None,
    expected_sha256: Annotated[
        str | None,
        Header(alias="X-Content-SHA256"),
    ] = None,
) -> SourceRecord:
    storage, _ = _runtime()
    try:
        return await storage.store_upload(
            file,
            role=role,
            expected_sha256=expected_sha256,
            archive_entrypoint=archive_entrypoint,
        )
    except StorageError as exc:
        raise _storage_http_error(exc) from exc
    finally:
        await file.close()


def _reference_image_sources(
    storage: WorkspaceStorage,
    source_ids: list[str],
) -> list[SourceRecord]:
    sources: list[SourceRecord] = []
    for source_id in source_ids:
        source = storage.get_source(source_id)
        if source.role != "reference_image":
            raise StorageError(
                "invalid_reference_image",
                "Authoring image_source_ids must reference uploaded images.",
                status_code=422,
            )
        sources.append(source)
    return sources


def _materialize_authoring_references(
    storage: WorkspaceStorage,
    sources: list[SourceRecord],
    destination: Path,
) -> list[GeometryAuthoringReference]:
    references: list[GeometryAuthoringReference] = []
    for index, source in enumerate(sources):
        reference_id = f"reference-{index + 1:02d}"
        reference_path = storage.materialize_source(
            source,
            destination / reference_id,
        )
        references.append(
            GeometryAuthoringReference(
                reference_id=reference_id,
                kind="image",
                media_type=source.artifact.media_type or "application/octet-stream",
                artifact=GeometryArtifactBinding(
                    path=str(reference_path.resolve()),
                    sha256=source.artifact.sha256,
                    size_bytes=source.artifact.size_bytes,
                ),
            )
        )
    return references


def _authoring_parameter_values(
    values: Mapping[str, bool | int | float | str | GeometryParameterInput],
) -> tuple[GeometryAuthoringParameterValue, ...]:
    return tuple(
        GeometryAuthoringParameterValue(
            name=name,
            value=item.value if isinstance(item, GeometryParameterInput) else item,
            unit=item.unit if isinstance(item, GeometryParameterInput) else None,
        )
        for name, item in sorted(values.items())
    )


def _family_variant_request_id(job_id: str, variant_id: str, index: int) -> str:
    variant_digest = hashlib.sha256(variant_id.encode("utf-8")).hexdigest()[:16]
    return f"{job_id}:variant-{index:02d}-{variant_digest}"


def _invalid_provider_variant(
    provider_id: str,
    summary: str,
    *,
    receipt: GeometryAuthoringProviderReceipt,
) -> GeometryAuthoringInvalidResponse:
    return GeometryAuthoringInvalidResponse(
        GeometryAuthoringFailure(
            provider_id=provider_id,
            operation="revise",
            code="invalid_response",
            summary=summary,
            retryable=False,
        ),
        receipt=receipt,
    )


def _materialize_source_bundle(
    storage: WorkspaceStorage,
    source: SourceRecord,
    destination: Path,
) -> GeometrySourceBundle:
    _source_path, manifest_path = storage.materialize_source_handoff(
        source,
        destination,
    )
    if manifest_path is None:
        raise StorageError(
            "source_bundle_required",
            "Geometry revisions require a provider-produced immutable source bundle.",
            status_code=422,
        )
    try:
        captured_manifest = read_contained_artifact(
            destination,
            manifest_path.name,
            max_bytes=16 * 1024 * 1024,
            capture_bytes=True,
        )
        assert captured_manifest.data is not None
        bundle = GeometrySourceBundle.model_validate_json(captured_manifest.data)
        representations: list[GeometryRepresentationBinding] = []
        for representation in bundle.representations:
            captured = read_contained_artifact(
                destination,
                representation.artifact.path,
                max_bytes=storage.settings.max_generated_artifact_bytes,
            )
            if (
                captured.sha256 != representation.artifact.sha256
                or captured.size_bytes != representation.artifact.size_bytes
            ):
                raise ValueError(
                    "Source-bundle representation differs from its manifest binding"
                )
            representations.append(
                representation.model_copy(
                    update={
                        "artifact": GeometryArtifactBinding(
                            path=str(captured.path),
                            sha256=captured.sha256,
                            size_bytes=captured.size_bytes,
                        )
                    }
                )
            )
        provenance_inputs: list[GeometryArtifactBinding] = []
        for input_artifact in bundle.provenance.input_artifacts:
            captured = read_contained_artifact(
                destination,
                input_artifact.path,
                max_bytes=storage.settings.max_generated_artifact_bytes,
            )
            if (
                captured.sha256 != input_artifact.sha256
                or captured.size_bytes != input_artifact.size_bytes
            ):
                raise ValueError(
                    "Source-bundle provenance input differs from its manifest binding"
                )
            provenance_inputs.append(
                GeometryArtifactBinding(
                    path=str(captured.path),
                    sha256=captured.sha256,
                    size_bytes=captured.size_bytes,
                )
            )
        provenance = bundle.provenance.model_copy(
            update={"input_artifacts": tuple(provenance_inputs)}
        )
        return bundle.model_copy(
            update={
                "representations": tuple(representations),
                "provenance": provenance,
            }
        )
    except (OSError, ValueError) as exc:
        raise StorageError(
            "source_bundle_invalid",
            "The immutable geometry source bundle is invalid or has drifted.",
            status_code=422,
        ) from exc


def _immutable_provider_source_bundle(
    request: GeometryImmutableExportRequest,
    *,
    provider_identity: Any,
    descriptor_path: Path,
) -> GeometrySourceBundle:
    """Bind an external immutable revision into the provider-neutral contract."""

    descriptor = request.model_dump_json(indent=2).encode("utf-8") + b"\n"
    descriptor_path.parent.mkdir(parents=True, exist_ok=False)
    descriptor_path.write_bytes(descriptor)
    digest = hashlib.sha256(descriptor).hexdigest()
    representation = GeometryRepresentationBinding(
        representation_id="provider-source",
        role="metadata",
        format="json",
        media_type="application/json",
        artifact=GeometryArtifactBinding(
            path=str(descriptor_path.resolve()),
            sha256=digest,
            size_bytes=len(descriptor),
        ),
    )
    coordinate_system = GeometryCoordinateSystem.model_validate(
        request.coordinate_system
    )
    provenance = GeometrySourceProvenance(
        request_digest=digest,
        upstream_edit_uri=request.upstream_edit_uri,
    )
    rights = GeometryRightsAssertion(
        assertion=request.rights_assertion,
        license_identifier=request.license_identifier,
    )
    bundle_id = geometry_source_bundle_id(
        provider=provider_identity,
        source_revision=request.source_revision,
        coordinate_system=coordinate_system,
        representations=(representation,),
        provenance=provenance,
        rights=rights,
    )
    return GeometrySourceBundle(
        bundle_id=bundle_id,
        producer=provider_identity,
        source_revision=request.source_revision,
        coordinate_system=coordinate_system,
        representations=(representation,),
        provenance=provenance,
        rights=rights,
    )


@app.post(
    "/api/geometry/generations",
    response_model=JobRecord,
    status_code=202,
    dependencies=[Depends(_require_access)],
    tags=["geometry"],
)
async def create_generation(request: GeometryGenerationRequest) -> JobRecord:
    storage, jobs = _runtime()
    job = jobs.create("generation", provider_id=request.provider_id)
    provider = authoring_providers.get(request.provider_id)
    if provider is None:
        return jobs.fail(
            job,
            code="authoring_provider_unavailable",
            message=f"Geometry authoring provider {request.provider_id!r} is unavailable.",
            retryable=True,
            details={"provider_id": request.provider_id},
        )

    try:
        image_sources = _reference_image_sources(storage, request.image_source_ids)
    except StorageError as exc:
        return jobs.fail(job, code=exc.code, message=exc.message)

    job = jobs.running(job)
    try:
        execution_dir = storage.execution_dir(job.job_id)
        references = _materialize_authoring_references(
            storage,
            image_sources,
            execution_dir / "references",
        )
        provider_request = GeometryAuthoringRequest(
            request_id=job.job_id,
            prompt=request.prompt,
            references=tuple(references),
            target_profile=request.target_profile,
            requested_formats=tuple(request.requested_formats),
            parameters=_authoring_parameter_values(request.parameters),
        )
        receipt = await authoring_providers.generate(
            request.provider_id,
            provider_request,
        )
        if receipt.disposition != "succeeded" or receipt.source_bundle is None:
            assert receipt.failure is not None
            raise GeometryAuthoringProviderError(receipt.failure, receipt=receipt)
        published = _publish_provider_bundle(
            receipt.source_bundle,
            storage=storage,
            provider_output_dir=execution_dir / "provider",
            provider_request_id=provider_request.request_id,
        )
    except Exception as exc:
        _discard_rejected_provider_staging(
            exc,
            storage=storage,
            provider_request_id=job.job_id,
        )
        logger.exception("Geometry authoring provider %s failed", request.provider_id)
        failure = _typed_provider_failure(exc)
        return jobs.fail(
            job,
            code=failure["code"],
            message=failure["message"],
            retryable=failure["retryable"],
            details=failure["details"],
        )
    return jobs.succeed(job, {"source": published})


@app.post(
    "/api/geometry/revisions",
    response_model=JobRecord,
    status_code=202,
    dependencies=[Depends(_require_access)],
    tags=["geometry"],
)
async def create_revision(request: GeometryRevisionRequest) -> JobRecord:
    storage, jobs = _runtime()
    job = jobs.create(
        "revision",
        provider_id=request.provider_id,
        source_id=request.source_id,
    )
    provider = authoring_providers.get(request.provider_id)
    if provider is None:
        return jobs.fail(
            job,
            code="authoring_provider_unavailable",
            message=f"Geometry authoring provider {request.provider_id!r} is unavailable.",
            retryable=True,
            details={"provider_id": request.provider_id},
        )

    try:
        source = storage.get_source(request.source_id)
        if source.source_bundle_manifest is None:
            raise StorageError(
                "source_bundle_required",
                "Geometry revisions require a provider-produced immutable source bundle.",
                status_code=422,
            )
        image_sources = _reference_image_sources(
            storage,
            request.image_source_ids,
        )
    except StorageError as exc:
        return jobs.fail(job, code=exc.code, message=exc.message)

    job = jobs.running(job)
    try:
        execution_dir = storage.execution_dir(job.job_id)
        source_bundle = _materialize_source_bundle(
            storage,
            source,
            execution_dir / "prior-source",
        )
        if source_bundle.producer.provider_id != request.provider_id:
            raise StorageError(
                "source_provider_mismatch",
                "Revisions require the provider that owns the source revision.",
                status_code=422,
            )
        resolve_semantic_parameter_overrides(
            source_bundle.parameters,
            _authoring_parameter_values(request.parameter_overrides),
            allow_undeclared=not source_bundle.parameters,
        )
        references = _materialize_authoring_references(
            storage,
            image_sources,
            execution_dir / "references",
        )
        provider_request = GeometryAuthoringRevisionRequest(
            request_id=job.job_id,
            source_bundle=source_bundle,
            instructions=request.instructions,
            references=tuple(references),
            parameter_overrides=_authoring_parameter_values(
                request.parameter_overrides
            ),
            requested_formats=tuple(request.requested_formats),
        )
        receipt = await authoring_providers.revise(
            request.provider_id,
            provider_request,
        )
        if receipt.disposition != "succeeded" or receipt.source_bundle is None:
            assert receipt.failure is not None
            raise GeometryAuthoringProviderError(receipt.failure, receipt=receipt)
        published = _publish_provider_bundle(
            receipt.source_bundle,
            storage=storage,
            provider_output_dir=execution_dir / "provider",
            provider_request_id=provider_request.request_id,
        )
    except StorageError as exc:
        return jobs.fail(job, code=exc.code, message=exc.message)
    except Exception as exc:
        _discard_rejected_provider_staging(
            exc,
            storage=storage,
            provider_request_id=job.job_id,
        )
        logger.exception("Geometry authoring provider %s failed", request.provider_id)
        failure = _typed_provider_failure(exc)
        return jobs.fail(
            job,
            code=failure["code"],
            message=failure["message"],
            retryable=failure["retryable"],
            details=failure["details"],
        )
    return jobs.succeed(job, {"source": published})


@app.post(
    "/api/geometry/families",
    response_model=JobRecord,
    status_code=202,
    dependencies=[Depends(_require_access)],
    tags=["geometry"],
)
async def create_family(request: GeometryFamilyRequest) -> JobRecord:
    """Materialize independent semantic variants from one immutable base."""

    storage, jobs = _runtime()
    job = jobs.create(
        "family",
        provider_id=request.provider_id,
        source_id=request.source_id,
    )
    provider = authoring_providers.get(request.provider_id)
    if provider is None:
        return jobs.fail(
            job,
            code="authoring_provider_unavailable",
            message=f"Geometry authoring provider {request.provider_id!r} is unavailable.",
            retryable=True,
            details={"provider_id": request.provider_id},
        )
    capabilities = provider.capabilities()
    if "parameter_families" not in capabilities.features:
        return jobs.fail(
            job,
            code="parameter_families_unsupported",
            message="The selected provider does not advertise semantic parameter families.",
        )
    if len(request.variants) > capabilities.max_family_variants:
        return jobs.fail(
            job,
            code="parameter_family_limit_exceeded",
            message="The requested family exceeds the selected provider's variant limit.",
            details={"max_family_variants": capabilities.max_family_variants},
        )

    try:
        source = storage.get_source(request.source_id)
        execution_dir = storage.execution_dir(job.job_id)
        source_bundle = _materialize_source_bundle(
            storage,
            source,
            execution_dir / "prior-source",
        )
        if source_bundle.producer.provider_id != request.provider_id:
            raise ValueError(
                "parameter families require the provider that owns the base revision"
            )
        family_request = GeometryAuthoringFamilyRequest(
            request_id=job.job_id,
            source_bundle=source_bundle,
            variants=tuple(
                GeometryAuthoringVariant(
                    variant_id=item.variant_id,
                    instructions=item.instructions,
                    parameter_overrides=_authoring_parameter_values(
                        item.parameter_overrides
                    ),
                )
                for item in request.variants
            ),
            requested_formats=tuple(request.requested_formats),
        )
    except (StorageError, ValueError) as exc:
        message = exc.message if isinstance(exc, StorageError) else str(exc)
        return jobs.fail(job, code="parameter_family_invalid", message=message)

    job = jobs.running(job)
    semaphore = asyncio.Semaphore(min(8, len(family_request.variants)))

    async def author_variant(
        index: int,
        variant: GeometryAuthoringVariant,
    ) -> GeometrySourceBundle:
        async with semaphore:
            revision_request = GeometryAuthoringRevisionRequest(
                request_id=_family_variant_request_id(
                    job.job_id,
                    variant.variant_id,
                    index,
                ),
                source_bundle=source_bundle,
                instructions=variant.instructions,
                parameter_overrides=variant.parameter_overrides,
                requested_formats=family_request.requested_formats,
            )
            receipt = await authoring_providers.revise(
                request.provider_id,
                revision_request,
            )
            if receipt.disposition != "succeeded" or receipt.source_bundle is None:
                assert receipt.failure is not None
                raise GeometryAuthoringProviderError(receipt.failure, receipt=receipt)
            result = receipt.source_bundle
            if result.provenance.parent_bundle_id != source_bundle.bundle_id:
                raise _invalid_provider_variant(
                    request.provider_id,
                    "Provider variant does not identify the immutable family base.",
                    receipt=receipt,
                )
            expected_parameters = resolve_semantic_parameter_overrides(
                source_bundle.parameters,
                variant.parameter_overrides,
            )
            try:
                validate_returned_parameter_state(
                    result.parameters, expected_parameters
                )
            except ValueError as exc:
                raise _invalid_provider_variant(
                    request.provider_id,
                    "Provider variant changed the declared semantic parameter family.",
                    receipt=receipt,
                ) from exc
            return result

    outcomes = await asyncio.gather(
        *(
            author_variant(index, item)
            for index, item in enumerate(family_request.variants, start=1)
        ),
        return_exceptions=True,
    )
    variant_results: list[dict[str, Any]] = []
    failure_count = 0
    for index, (variant, outcome) in enumerate(
        zip(family_request.variants, outcomes, strict=True),
        start=1,
    ):
        provider_request_id = _family_variant_request_id(
            job.job_id,
            variant.variant_id,
            index,
        )
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome
        if isinstance(outcome, Exception):
            failure_count += 1
            _discard_rejected_provider_staging(
                outcome,
                storage=storage,
                provider_request_id=provider_request_id,
            )
            failure = _typed_provider_failure(outcome)
            variant_results.append(
                {
                    "variant_id": variant.variant_id,
                    "status": "failed",
                    "parameter_overrides": {
                        item.name: item.value for item in variant.parameter_overrides
                    },
                    "parameter_units": {
                        item.name: item.unit
                        for item in variant.parameter_overrides
                        if item.unit is not None
                    },
                    "error": failure,
                }
            )
            continue
        if isinstance(outcome, BaseException):
            raise outcome
        try:
            published = _publish_provider_bundle(
                outcome,
                storage=storage,
                provider_output_dir=(
                    execution_dir / "variants" / variant.variant_id / "provider"
                ),
                provider_request_id=provider_request_id,
            )
            variant_results.append(
                {
                    "variant_id": variant.variant_id,
                    "status": "succeeded",
                    "parameter_overrides": {
                        item.name: item.value for item in variant.parameter_overrides
                    },
                    "parameter_units": {
                        item.name: item.unit
                        for item in variant.parameter_overrides
                        if item.unit is not None
                    },
                    "source": published,
                }
            )
        except Exception as exc:
            failure_count += 1
            _discard_rejected_provider_staging(
                exc,
                storage=storage,
                provider_request_id=provider_request_id,
            )
            variant_results.append(
                {
                    "variant_id": variant.variant_id,
                    "status": "failed",
                    "parameter_overrides": {
                        item.name: item.value for item in variant.parameter_overrides
                    },
                    "parameter_units": {
                        item.name: item.unit
                        for item in variant.parameter_overrides
                        if item.unit is not None
                    },
                    "error": _typed_provider_failure(exc),
                }
            )
    result = {
        "schema_version": "geometry.parameter-family-result.v1",
        "base_source_id": request.source_id,
        "base_bundle_id": source_bundle.bundle_id,
        "provider_id": request.provider_id,
        "variant_count": len(variant_results),
        "succeeded_count": len(variant_results) - failure_count,
        "failed_count": failure_count,
        "variants": variant_results,
    }
    if failure_count:
        return jobs.fail(
            job,
            code="parameter_family_incomplete",
            message="One or more parameter variants failed; no fallback was attempted.",
            result=result,
        )
    return jobs.succeed(job, result)


@app.post(
    "/api/geometry/provider-exports",
    response_model=JobRecord,
    status_code=202,
    dependencies=[Depends(_require_access)],
    tags=["geometry"],
)
async def create_provider_export(
    request: GeometryImmutableExportRequest,
) -> JobRecord:
    """Export a provider-owned immutable revision into a registered source."""

    storage, jobs = _runtime()
    job = jobs.create("export", provider_id=request.provider_id)
    provider = authoring_providers.get(request.provider_id)
    if provider is None:
        return jobs.fail(
            job,
            code="authoring_provider_unavailable",
            message=f"Geometry authoring provider {request.provider_id!r} is unavailable.",
            retryable=True,
            details={"provider_id": request.provider_id},
        )
    capabilities = provider.capabilities()
    if "export" not in capabilities.operations:
        return jobs.fail(
            job,
            code="authoring_export_unsupported",
            message="The selected provider does not advertise an export operation.",
        )

    job = jobs.running(job)
    try:
        execution_dir = storage.execution_dir(job.job_id)
        source_bundle = _immutable_provider_source_bundle(
            request,
            provider_identity=capabilities.provider,
            descriptor_path=execution_dir / "provider-source" / "source.json",
        )
        provider_request = GeometryAuthoringExportRequest(
            request_id=job.job_id,
            source_bundle=source_bundle,
            requested_formats=tuple(request.requested_formats),
        )
        receipt = await authoring_providers.export(
            request.provider_id,
            provider_request,
        )
        if receipt.disposition != "succeeded" or receipt.source_bundle is None:
            assert receipt.failure is not None
            raise GeometryAuthoringProviderError(receipt.failure, receipt=receipt)
        published = _publish_provider_bundle(
            receipt.source_bundle,
            storage=storage,
            provider_output_dir=execution_dir / "provider",
            provider_request_id=provider_request.request_id,
        )
    except Exception as exc:
        _discard_rejected_provider_staging(
            exc,
            storage=storage,
            provider_request_id=job.job_id,
        )
        failure = _typed_provider_failure(exc)
        return jobs.fail(
            job,
            code=failure["code"],
            message=failure["message"],
            retryable=failure["retryable"],
            details=failure["details"],
        )
    return jobs.succeed(job, {"source": published})


@app.post(
    "/api/geometry/exports",
    response_model=JobRecord,
    status_code=202,
    dependencies=[Depends(_require_access)],
    tags=["geometry"],
)
async def create_export(request: GeometryExportRequest) -> JobRecord:
    storage, jobs = _runtime()
    job = jobs.create(
        "export",
        provider_id=request.provider_id,
        source_id=request.source_id,
    )
    provider = authoring_providers.get(request.provider_id)
    if provider is None:
        return jobs.fail(
            job,
            code="authoring_provider_unavailable",
            message=f"Geometry authoring provider {request.provider_id!r} is unavailable.",
            retryable=True,
            details={"provider_id": request.provider_id},
        )
    if "export" not in provider.capabilities().operations:
        return jobs.fail(
            job,
            code="authoring_export_unsupported",
            message="The selected provider does not advertise a separate export operation.",
        )
    job = jobs.running(job)
    try:
        source = storage.get_source(request.source_id)
        execution_dir = storage.execution_dir(job.job_id)
        source_bundle = _materialize_source_bundle(
            storage,
            source,
            execution_dir / "prior-source",
        )
        if source_bundle.producer.provider_id != request.provider_id:
            raise ValueError(
                "exports require the provider that owns the source revision"
            )
        provider_request = GeometryAuthoringExportRequest(
            request_id=job.job_id,
            source_bundle=source_bundle,
            requested_formats=tuple(request.requested_formats),
        )
        receipt = await authoring_providers.export(
            request.provider_id,
            provider_request,
        )
        if receipt.disposition != "succeeded" or receipt.source_bundle is None:
            assert receipt.failure is not None
            raise GeometryAuthoringProviderError(receipt.failure, receipt=receipt)
        if (
            receipt.source_bundle.provenance.parent_bundle_id != source_bundle.bundle_id
            or receipt.source_bundle.source_revision != source_bundle.source_revision
        ):
            raise ValueError(
                "provider export does not preserve the requested immutable source revision"
            )
        published = _publish_provider_bundle(
            receipt.source_bundle,
            storage=storage,
            provider_output_dir=execution_dir / "provider",
            provider_request_id=provider_request.request_id,
        )
    except Exception as exc:
        _discard_rejected_provider_staging(
            exc,
            storage=storage,
            provider_request_id=job.job_id,
        )
        failure = _typed_provider_failure(exc)
        return jobs.fail(
            job,
            code=failure["code"],
            message=failure["message"],
            retryable=failure["retryable"],
            details=failure["details"],
        )
    return jobs.succeed(job, {"source": published})


@app.post(
    "/api/geometry/runs",
    response_model=JobRecord,
    status_code=202,
    dependencies=[Depends(_require_access)],
    tags=["geometry"],
)
async def create_run(request: GeometryRunRequest) -> JobRecord:
    if request.param_overrides:
        raise HTTPException(
            status_code=422,
            detail=ServiceError(
                code="run_parameter_overrides_require_revision",
                message=(
                    "Geometry runs consume an immutable source revision and cannot apply "
                    "parameter overrides. Create a provider revision with "
                    "POST /api/geometry/revisions, then run the returned source_id."
                ),
            ).model_dump(mode="json"),
        )
    runtime_engine = request.runtime_engine or settings.default_runtime_engine
    if request.run_runtime_validation and runtime_engine == "none":
        raise HTTPException(
            status_code=422,
            detail=ServiceError(
                code="runtime_validation_engine_unavailable",
                message=(
                    "Runtime validation was requested, but no runtime engine is "
                    "configured. Set runtime_engine explicitly or configure "
                    "GEOMETRY_AGENT_SERVICE_DEFAULT_RUNTIME_ENGINE."
                ),
            ).model_dump(mode="json"),
        )
    storage, jobs = _runtime()
    try:
        source = storage.get_source(request.source_id)
        if source.role != "geometry_source":
            raise StorageError(
                "invalid_geometry_source",
                "Geometry runs require a geometry_source artifact.",
                status_code=422,
            )
    except StorageError as exc:
        raise _storage_http_error(exc) from exc

    job = jobs.create("run", source_id=source.source_id)
    job = jobs.running(job)
    try:
        execution_dir = storage.execution_dir(job.job_id)
        source_path, source_manifest_path = storage.materialize_source_handoff(
            source,
            execution_dir / "input",
        )
        output_dir = execution_dir / "output"
        params = GeometryWorkflowInput(
            source_path=source_path,
            source_manifest_path=source_manifest_path,
            source_representation_id=source.source_representation_id,
            expected_source_sha256=storage.materialized_source_sha256(source_path),
            expected_source_manifest_sha256=(
                source.source_bundle_manifest.sha256
                if source.source_bundle_manifest is not None
                else None
            ),
            output_dir=output_dir,
            target_profile=request.target_profile,
            target_runtime=request.target_runtime,
            run_runtime_validation=request.run_runtime_validation,
            runtime_engine=runtime_engine,
            render_evidence=request.render_evidence,
            render_preset=request.render_preset,
            render_backend=settings.render_backend,
            render_remote_base_url=settings.render_remote_base_url,
            render_remote_api_key=settings.render_remote_api_key,
            render_remote_allow_unauthenticated_identity=(
                settings.render_remote_allow_unauthenticated_identity
            ),
            fail_on_validation_error=request.fail_on_validation_error,
            allow_lossy_recovery=request.allow_lossy_recovery,
            usd_tessellation_tolerance=request.usd_tessellation_tolerance,
            audit_context={"geometry_agent_service_job_id": job.job_id},
        )
        workflow_result = await asyncio.to_thread(run_geometry_workflow, params)
        published = _publish_workflow_result(
            workflow_result,
            storage=storage,
            execution_dir=execution_dir,
        )
        if not bool(published["workflow"].get("success")):
            return jobs.fail(
                job,
                code="geometry_workflow_failed",
                message="Geometry workflow completed without a valid handoff.",
                result=published,
            )
        return jobs.succeed(job, published)
    except StorageError as exc:
        logger.warning("Geometry job %s failed storage validation: %s", job.job_id, exc)
        return jobs.fail(job, code=exc.code, message=exc.message)
    except Exception as exc:
        logger.exception("Geometry workflow job %s failed", job.job_id)
        return jobs.fail(
            job,
            code="geometry_workflow_error",
            message="Geometry workflow execution failed.",
            details={"exception_type": type(exc).__name__},
        )


@app.get(
    "/api/geometry/jobs/{job_id}",
    response_model=JobRecord,
    dependencies=[Depends(_require_access)],
    tags=["geometry"],
)
def get_job(job_id: str) -> JobRecord:
    _, jobs = _runtime()
    try:
        return jobs.get(job_id)
    except StorageError as exc:
        raise _storage_http_error(exc) from exc


@app.get(
    "/api/geometry/artifacts/{artifact_id}",
    dependencies=[Depends(_require_access)],
    tags=["geometry"],
)
def get_artifact(artifact_id: str) -> FileResponse:
    storage, _ = _runtime()
    try:
        record, path = storage.get_artifact(artifact_id)
    except StorageError as exc:
        raise _storage_http_error(exc) from exc
    return FileResponse(
        path,
        media_type=record.media_type or "application/octet-stream",
        filename=record.filename,
        headers={
            "Digest": f"sha-256={record.sha256}",
            "X-Content-SHA256": record.sha256,
        },
    )


@app.get(
    "/api/geometry/providers",
    response_model=ProviderListResponse,
    dependencies=[Depends(_require_access)],
    tags=["geometry"],
)
def list_providers() -> ProviderListResponse:
    return ProviderListResponse(providers=authoring_providers.list())


def _publish_workflow_result(
    result: Any,
    *,
    storage: WorkspaceStorage,
    execution_dir: Path,
) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        payload = result.model_dump(mode="json")
    elif isinstance(result, Mapping):
        payload = dict(result)
    else:
        raise TypeError("Geometry workflow result must be a mapping or Pydantic model")

    root = execution_dir.resolve()
    preserved_json_paths: set[Path] = set()
    for key in ("source_bundle_manifest_path",):
        if payload.get(key) is None:
            continue
        with suppress(OSError, RuntimeError, ValueError):
            candidate = Path(str(payload[key])).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            preserved_json_paths.add(candidate.resolve(strict=False))
    publisher = _WorkflowArtifactPublisher(
        storage=storage,
        execution_root=root,
        preserved_json_paths=preserved_json_paths,
    )
    workflow: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "output_dir":
            continue
        if key.endswith("_path"):
            if value is None:
                workflow[key] = None
                continue
            workflow[key] = publisher.publish_reference(
                value,
                label=key,
                base_dir=root,
                required=True,
            )
            continue
        workflow[key] = publisher.rewrite_references(
            value,
            label=key,
            base_dir=root,
        )
    return {
        "workflow": workflow,
        "artifacts": publisher.artifacts,
        "omitted_path_fields": publisher.omitted_path_fields,
    }


class _ResolvedWorkflowReference(NamedTuple):
    path: Path
    traverses_symlink: bool


class _WorkflowArtifactPublisher:
    """Publish workflow files and rewrite nested local references for API clients."""

    def __init__(
        self,
        *,
        storage: WorkspaceStorage,
        execution_root: Path,
        preserved_json_paths: set[Path],
    ) -> None:
        self.storage = storage
        self.execution_root = execution_root
        self.preserved_json_paths = preserved_json_paths
        self.artifacts: dict[str, Any] = {}
        self.omitted_path_fields: list[str] = []
        self._published_json_paths: dict[Path, Path] = {}
        self._json_paths_in_progress: set[Path] = set()

    def publish_reference(
        self,
        value: Any,
        *,
        label: str,
        base_dir: Path,
        required: bool = False,
    ) -> str | None:
        if not isinstance(value, str):
            if required:
                self.omitted_path_fields.append(label)
            return None
        if not value:
            if required:
                self.omitted_path_fields.append(label)
                return None
            return value
        reference = self._resolve_reference(value, base_dir=base_dir)
        if (
            reference is not None
            and not reference.traverses_symlink
            and reference.path.is_file()
            and reference.path.is_relative_to(self.execution_root)
        ):
            return self._publish_file(reference.path, label=label)
        if required or (
            reference is not None and reference.path.is_relative_to(self.execution_root)
        ):
            self.omitted_path_fields.append(label)
            return None
        redacted = _redact_workspace(value, self.storage.root)
        assert isinstance(redacted, str)
        return redacted

    def rewrite_references(
        self,
        value: Any,
        *,
        label: str,
        base_dir: Path,
    ) -> Any:
        if isinstance(value, str):
            reference = self._resolve_reference(value, base_dir=base_dir)
            if (
                reference is not None
                and not reference.traverses_symlink
                and reference.path.is_file()
                and reference.path.is_relative_to(self.execution_root)
            ):
                return self._publish_file(reference.path, label=label)
            if reference is not None and (
                reference.traverses_symlink
                or (
                    reference.path.is_relative_to(self.execution_root)
                    and str(self.storage.root) in value
                )
            ):
                self.omitted_path_fields.append(label)
                return None
            return _redact_workspace(value, self.storage.root)
        if isinstance(value, list):
            return [
                self.rewrite_references(
                    item,
                    label=f"{label}/{index}",
                    base_dir=base_dir,
                )
                for index, item in enumerate(value)
            ]
        if isinstance(value, dict):
            return {
                str(key): self.rewrite_references(
                    item,
                    label=f"{label}/{key}",
                    base_dir=base_dir,
                )
                for key, item in value.items()
            }
        return value

    def _publish_file(self, path: Path, *, label: str) -> str | None:
        publish_path = path
        if path.suffix.lower() == ".json" and path not in self.preserved_json_paths:
            public_copy = self._public_json_copy(path, label=label)
            if public_copy is None:
                return None
            publish_path = public_copy
        artifact = self.storage.store_generated_file(publish_path)
        self.artifacts[label] = artifact.model_dump(mode="json")
        return artifact.artifact_id

    def _public_json_copy(self, path: Path, *, label: str) -> Path | None:
        cached = self._published_json_paths.get(path)
        if cached is not None:
            return cached
        if path in self._json_paths_in_progress:
            self.omitted_path_fields.append(label)
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            self.omitted_path_fields.append(label)
            return None

        self._json_paths_in_progress.add(path)
        try:
            rewritten = self.rewrite_references(
                payload,
                label=label,
                base_dir=path.parent,
            )
            relative = path.relative_to(self.execution_root)
            destination = self.execution_root / ".published" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp")
            temporary.write_text(
                json.dumps(rewritten, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
            self._published_json_paths[path] = destination
            return destination
        finally:
            self._json_paths_in_progress.remove(path)

    def _resolve_reference(
        self,
        value: str,
        *,
        base_dir: Path,
    ) -> _ResolvedWorkflowReference | None:
        try:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            lexical_path = Path(os.path.abspath(candidate))
            resolved_path = lexical_path.resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            return None

        traverses_symlink = False
        try:
            relative = lexical_path.relative_to(self.execution_root)
        except ValueError:
            # An outside lexical path that resolves into the workspace necessarily
            # crossed a symlink and must not become a published workflow artifact.
            traverses_symlink = resolved_path.is_relative_to(self.execution_root)
        else:
            cursor = self.execution_root
            for component in relative.parts:
                cursor /= component
                if cursor.is_symlink():
                    traverses_symlink = True
                    break
        return _ResolvedWorkflowReference(resolved_path, traverses_symlink)


def _publish_provider_bundle(
    bundle: GeometrySourceBundle,
    *,
    storage: WorkspaceStorage,
    provider_output_dir: Path,
    provider_request_id: str,
) -> dict[str, Any]:
    """Ingest one bundle, then discard any service-owned provider staging bytes."""

    try:
        return _ingest_provider_bundle(
            bundle,
            storage=storage,
            provider_output_dir=provider_output_dir,
        )
    finally:
        _discard_provider_staging_bundle(
            bundle,
            storage=storage,
            provider_request_id=provider_request_id,
        )


def _ingest_provider_bundle(
    bundle: GeometrySourceBundle,
    *,
    storage: WorkspaceStorage,
    provider_output_dir: Path,
) -> dict[str, Any]:
    """Ingest one materialized ``geometry.source.v1`` bundle by digest."""

    validate_geometry_source_bundle_identity(bundle)
    provider_output_dir.mkdir(parents=True, exist_ok=False)
    rewritten_representations: list[GeometryRepresentationBinding] = []
    published_by_representation: dict[str, Any] = {}
    validated_representations: list[tuple[GeometryRepresentationBinding, Path]] = []
    for representation in bundle.representations:
        binding = representation.artifact
        raw_path = Path(os.path.abspath(Path(binding.path).expanduser()))
        path = raw_path.resolve(strict=True)
        file_stat = path.stat()
        if (
            raw_path != path
            or path.is_symlink()
            or not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_nlink != 1
        ):
            raise TypeError("Provider artifacts must be single-link regular files")
        validated_representations.append((representation, path))

    common_root = Path(
        os.path.commonpath([str(path.parent) for _, path in validated_representations])
    )
    if common_root == Path(common_root.anchor):
        raise TypeError("Provider artifacts must share a bounded bundle root")
    validated_provenance_inputs: list[tuple[GeometryArtifactBinding, Path]] = []
    for binding in bundle.provenance.input_artifacts:
        raw_path = Path(os.path.abspath(Path(binding.path).expanduser()))
        path = raw_path.resolve(strict=True)
        file_stat = path.stat()
        if (
            raw_path != path
            or path.is_symlink()
            or not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_nlink != 1
            or not path.is_relative_to(storage.executions_dir)
        ):
            raise TypeError(
                "Provider provenance inputs must be service-materialized regular files"
            )
        validated_provenance_inputs.append((binding, path))

    published_paths: set[str] = set()
    for representation, path in validated_representations:
        binding = representation.artifact
        relative_path = path.relative_to(common_root).as_posix()
        if relative_path == "geometry.source.json" or relative_path in published_paths:
            raise TypeError("Provider representations contain duplicate reserved paths")
        published_paths.add(relative_path)
        destination = provider_output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_provider_artifact(
            path,
            destination,
            expected_sha256=binding.sha256,
            expected_size_bytes=binding.size_bytes,
            max_bytes=storage.settings.max_generated_artifact_bytes,
        )
        artifact = storage.store_generated_file(
            destination,
            expected_filename=path.name,
            expected_sha256=binding.sha256,
            expected_size_bytes=binding.size_bytes,
            media_type=representation.media_type,
            bundle_member=relative_path,
        )
        published_by_representation[representation.representation_id] = artifact
        rewritten_representations.append(
            representation.model_copy(
                update={
                    "artifact": GeometryArtifactBinding(
                        path=relative_path,
                        sha256=artifact.sha256,
                        size_bytes=artifact.size_bytes,
                    )
                }
            )
        )

    rewritten_provenance_inputs: list[GeometryArtifactBinding] = []
    published_provenance_artifacts: list[ArtifactRecord] = []
    for index, (binding, path) in enumerate(validated_provenance_inputs, start=1):
        suffix = path.suffix.lower()
        if len(suffix) > 17 or not suffix.removeprefix(".").isalnum():
            suffix = ""
        relative_path = f"provenance/input-{index:03d}-{binding.sha256[:12]}{suffix}"
        if relative_path in published_paths:
            raise TypeError("Provider bundle contains duplicate published paths")
        published_paths.add(relative_path)
        destination = provider_output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_provider_artifact(
            path,
            destination,
            expected_sha256=binding.sha256,
            expected_size_bytes=binding.size_bytes,
            max_bytes=storage.settings.max_generated_artifact_bytes,
        )
        artifact = storage.store_generated_file(
            destination,
            expected_filename=destination.name,
            expected_sha256=binding.sha256,
            expected_size_bytes=binding.size_bytes,
            bundle_member=relative_path,
        )
        published_provenance_artifacts.append(artifact)
        rewritten_provenance_inputs.append(
            binding.model_copy(update={"path": relative_path})
        )

    rewritten_provenance = bundle.provenance.model_copy(
        update={"input_artifacts": tuple(rewritten_provenance_inputs)}
    )
    portable_bundle = bundle.model_copy(
        update={
            "representations": tuple(rewritten_representations),
            "provenance": GeometrySourceProvenance.model_validate(rewritten_provenance),
        }
    )
    manifest_path = provider_output_dir / "geometry.source.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(portable_bundle.model_dump_json(indent=2))
        stream.write("\n")
    manifest_artifact = storage.store_generated_file(
        manifest_path,
        expected_filename=manifest_path.name,
        media_type="application/json",
    )

    runnable_representation = _select_runnable_representation(portable_bundle)
    runnable_artifact = published_by_representation[
        runnable_representation.representation_id
    ]
    source = storage.create_source_from_bundle(
        runnable_artifact,
        manifest=manifest_artifact,
        representation_artifacts=[
            (
                item.role,
                published_by_representation[item.representation_id],
            )
            for item in portable_bundle.representations
        ],
        provenance_artifacts=published_provenance_artifacts,
        representation_id=runnable_representation.representation_id,
    )
    return {
        "schema_version": "geometry.source.v1",
        "source_id": source.source_id,
        "bundle_id": portable_bundle.bundle_id,
        "producer": portable_bundle.producer.model_dump(mode="json"),
        "source_revision": portable_bundle.source_revision,
        "coordinate_system": portable_bundle.coordinate_system.model_dump(mode="json"),
        "selected_representation_id": runnable_representation.representation_id,
        "manifest_artifact": manifest_artifact.model_dump(mode="json"),
        "representations": [
            {
                "representation_id": item.representation_id,
                "role": item.role,
                "format": item.format,
                "media_type": item.media_type,
                "artifact": published_by_representation[
                    item.representation_id
                ].model_dump(mode="json"),
            }
            for item in portable_bundle.representations
        ],
        "parts": [item.model_dump(mode="json") for item in portable_bundle.parts],
        "parameters": [
            item.model_dump(mode="json") for item in portable_bundle.parameters
        ],
        "verification_assertions": [
            item.model_dump(mode="json")
            for item in portable_bundle.verification_assertions
        ],
        "provenance": portable_bundle.provenance.model_dump(mode="json"),
        "rights": portable_bundle.rights.model_dump(mode="json"),
    }


def _discard_provider_staging_bundle(
    bundle: GeometrySourceBundle,
    *,
    storage: WorkspaceStorage,
    provider_request_id: str,
) -> bool:
    """Remove one flat request directory owned by this service's provider staging."""

    staging_root = Path(os.path.abspath(storage.root / "provider-artifacts"))
    lexical_paths = tuple(
        Path(os.path.abspath(Path(item.artifact.path).expanduser()))
        for item in bundle.representations
    )
    parent_paths = {path.parent for path in lexical_paths}
    if len(parent_paths) != 1:
        return False
    request_dir = next(iter(parent_paths))
    try:
        relative = request_dir.relative_to(staging_root)
    except ValueError:
        return False
    if len(relative.parts) != 2:
        return False
    safe_request_id = provider_request_id.replace(":", "_")
    expected_prefix = f"{safe_request_id}-"
    if not request_dir.name.startswith(expected_prefix):
        return False
    suffix = request_dir.name.removeprefix(expected_prefix)
    request_digest, separator, nonce = suffix.partition("-")
    lowercase_hex = frozenset("0123456789abcdef")
    if (
        separator != "-"
        or len(request_digest) != 12
        or len(nonce) != 32
        or not set(request_digest).issubset(lowercase_hex)
        or not set(nonce).issubset(lowercase_hex)
    ):
        return False
    if any(
        path.parent != request_dir or path.name in {"", ".", ".."}
        for path in lexical_paths
    ):
        return False

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    staging_fd = provider_fd = request_fd = quarantine_fd = -1
    quarantine_name: str | None = None
    quarantine_entry: str | None = None
    try:
        staging_fd = os.open(staging_root, directory_flags)
        provider_fd = os.open(relative.parts[0], directory_flags, dir_fd=staging_fd)
        request_fd = os.open(relative.parts[1], directory_flags, dir_fd=provider_fd)
        opened = os.fstat(request_fd)
        current = os.stat(
            relative.parts[1],
            dir_fd=provider_fd,
            follow_symlinks=False,
        )
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            return False

        entries = os.listdir(request_fd)
        for name in entries:
            metadata = os.stat(name, dir_fd=request_fd, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                logger.warning(
                    "Refusing to remove nested provider staging directory: %s/%s",
                    request_dir,
                    name,
                )
                return False

        quarantine_name = f".geometry-agent-cleanup-{secrets.token_hex(16)}"
        quarantine_entry = secrets.token_hex(16)
        os.mkdir(quarantine_name, mode=0o700, dir_fd=provider_fd)
        quarantine_fd = os.open(quarantine_name, directory_flags, dir_fd=provider_fd)
        os.rename(
            relative.parts[1],
            quarantine_entry,
            src_dir_fd=provider_fd,
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
            return False

        for name in entries:
            os.unlink(name, dir_fd=request_fd)
        os.fsync(request_fd)

        quarantined = os.stat(
            quarantine_entry,
            dir_fd=quarantine_fd,
            follow_symlinks=False,
        )
        if (opened.st_dev, opened.st_ino) != (
            quarantined.st_dev,
            quarantined.st_ino,
        ):
            return False
        os.close(request_fd)
        request_fd = -1
        os.rmdir(quarantine_entry, dir_fd=quarantine_fd)
        quarantine_entry = None
        os.fsync(quarantine_fd)
        os.close(quarantine_fd)
        quarantine_fd = -1
        os.rmdir(quarantine_name, dir_fd=provider_fd)
        quarantine_name = None
        os.fsync(provider_fd)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning(
            "Unable to remove service-owned provider staging directory %s",
            request_dir,
            exc_info=True,
        )
        return False
    finally:
        for descriptor in (request_fd, quarantine_fd):
            if descriptor >= 0:
                os.close(descriptor)
        if quarantine_name is not None and provider_fd >= 0:
            try:
                os.rmdir(quarantine_name, dir_fd=provider_fd)
            except OSError:
                pass
        for descriptor in (provider_fd, staging_fd):
            if descriptor >= 0:
                os.close(descriptor)


def _discard_rejected_provider_staging(
    exc: Exception,
    *,
    storage: WorkspaceStorage,
    provider_request_id: str,
) -> None:
    """Discard service-owned bytes retained on a rejected successful receipt."""

    if not isinstance(exc, GeometryAuthoringProviderError):
        return
    receipt = exc.receipt
    if receipt is None or receipt.source_bundle is None:
        return
    _discard_provider_staging_bundle(
        receipt.source_bundle,
        storage=storage,
        provider_request_id=provider_request_id,
    )


def _select_runnable_representation(
    bundle: GeometrySourceBundle,
) -> GeometryRepresentationBinding:
    """Choose the highest-fidelity root representation for shared processing."""

    root_representation_ids = {
        representation_id
        for part in bundle.parts
        if part.parent_part_id is None
        for representation_id in part.representation_ids
    }
    for role in (
        "design_exchange",
        "render_geometry",
        "collision_candidate",
        "reference",
    ):
        candidates = [item for item in bundle.representations if item.role == role]
        root_candidates = [
            item
            for item in candidates
            if item.representation_id in root_representation_ids
        ]
        if len(root_candidates) == 1:
            return root_candidates[0]
        if candidates:
            # Provider order is the deterministic fallback for bundles that do
            # not declare one unique root semantic-part binding.
            return candidates[0]
    raise TypeError("Provider source bundle contains no runnable representation")


def _copy_provider_artifact(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    max_bytes: int,
) -> None:
    """Copy exact provider bytes through held no-follow descriptors."""

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    destination_fd = os.open(destination, flags, 0o600)
    try:
        captured = read_contained_artifact(
            source.parent,
            source.name,
            max_bytes=max_bytes,
            copy_fd=destination_fd,
        )
        os.fsync(destination_fd)
        if (
            captured.sha256 != expected_sha256
            or captured.size_bytes != expected_size_bytes
        ):
            raise TypeError("Provider artifact differs from its declared identity")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(destination_fd)


def _redact_workspace(value: Any, workspace_root: Path) -> Any:
    marker = "<geometry-agent-workspace>"
    if isinstance(value, str):
        return value.replace(str(workspace_root), marker)
    if isinstance(value, list):
        return [_redact_workspace(item, workspace_root) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _redact_workspace(item, workspace_root)
            for key, item in value.items()
        }
    return value


def _typed_provider_failure(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, GeometryAuthoringProviderError):
        return {
            "code": exc.failure.code,
            "message": exc.failure.summary,
            "retryable": exc.failure.retryable,
            "details": {"provider_id": exc.failure.provider_id},
        }
    failure_method = getattr(exc, "as_failure", None)
    if callable(failure_method):
        raw = failure_method()
        if isinstance(raw, Mapping):
            code = raw.get("code")
            summary = raw.get("summary")
            if (
                isinstance(code, str)
                and code.replace("_", "").isalnum()
                and code[0].isalpha()
                and len(code) <= 128
                and isinstance(summary, str)
                and 0 < len(summary) <= 2048
            ):
                details = {}
                provider_id = raw.get("provider_id")
                if isinstance(provider_id, str) and len(provider_id) <= 128:
                    details["provider_id"] = provider_id
                return {
                    "code": code,
                    "message": summary,
                    "retryable": bool(raw.get("retryable", False)),
                    "details": details,
                }
    return {
        "code": "authoring_provider_failed",
        "message": "The selected geometry authoring provider failed.",
        "retryable": True,
        "details": {"exception_type": type(exc).__name__},
    }


def _storage_http_error(exc: StorageError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail=ServiceError(code=exc.code, message=exc.message).model_dump(mode="json"),
    )


def serve(host: str, port: int) -> None:
    _validate_service_security(host)
    uvicorn.run("geometry_agent_service.main:app", host=host, port=port, reload=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Geometry Agent service")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
