# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit replacement-provider leaf for advisory Articulation proposals."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Literal, Protocol, Self
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from content_agent_workflows.common.artifacts import (
    file_sha256,
    prepare_writable_directory,
    read_contained_artifact,
)
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_decision import (
    DomainProposalPayload,
    ProducerIdentity,
    canonical_json_digest,
)

from .embedded_decision import (
    EmbeddedArticulationError,
    EmbeddedArticulationPreparation,
    EmbeddedArticulationProposalCapability,
    EmbeddedArticulationProviderProposal,
)

ARTICULATION_PROPOSAL_PROVIDER_REQUEST_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-proposal-provider-request.v1"
] = "content-agent-workflows.articulation-proposal-provider-request.v1"
ARTICULATION_PROPOSAL_PROVIDER_REQUEST_V2_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-proposal-provider-request.v2"
] = "content-agent-workflows.articulation-proposal-provider-request.v2"
ARTICULATION_PROPOSAL_PROVIDER_PUBLICATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-proposal-provider-publication.v1"
] = "content-agent-workflows.articulation-proposal-provider-publication.v1"
ARTICULATION_PROPOSAL_ATTEMPT_RECEIPT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-proposal-attempt-receipt.v1"
] = "content-agent-workflows.articulation-proposal-attempt-receipt.v1"
ARTICULATION_PROPOSAL_ATTEMPT_PUBLICATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-proposal-attempt-publication.v1"
] = "content-agent-workflows.articulation-proposal-attempt-publication.v1"
_PROPOSAL_SCHEMA_VERSION: Literal[
    "content-agent-workflows.embedded-articulation-provider-proposal.v2"
] = "content-agent-workflows.embedded-articulation-provider-proposal.v2"
_WRAPPED_PAYLOAD_SCHEMA_VERSION = (
    "content-agent-workflows.articulation-provider-proposal-payload.v2"
)
_MAX_ARTICULATION_INPUT_JSON_BYTES = 16 * 1024 * 1024
_MAX_ARTICULATION_PREPARATION_JSON_BYTES = 64 * 1024 * 1024
_MAX_ARTICULATION_GENERATED_JSON_BYTES = 128 * 1024 * 1024
_MAX_REPLACEMENT_LINEAGE_DEPTH = 128
_MAX_ARTICULATION_PARTIAL_OUTPUTS = 128
_PARTIAL_OUTPUT_OVERFLOW_SUMMARY = (
    "The provider emitted more partial outputs than the bounded terminal inventory "
    "can retain; excess outputs were discarded."
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _binding(path: str | Path, *, label: str) -> ExecutionArtifactBinding:
    expanded = Path(path).expanduser()
    absolute = Path(os.path.abspath(expanded))
    try:
        resolved = absolute.resolve(strict=True)
        metadata = resolved.stat()
        digest = file_sha256(resolved)
    except OSError as exc:
        raise EmbeddedArticulationError(
            f"{label} is not a regular file: {absolute}"
        ) from exc
    if resolved != absolute:
        raise EmbeddedArticulationError(f"{label} traverses a symlink: {absolute}")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise EmbeddedArticulationError(f"{label} is not a regular file: {resolved}")
    return ExecutionArtifactBinding(
        path=str(resolved),
        sha256=digest,
        size_bytes=metadata.st_size,
    )


def _verify_binding(binding: ExecutionArtifactBinding, *, label: str) -> None:
    observed = _binding(binding.path, label=label)
    if observed != binding:
        raise EmbeddedArticulationError(f"{label} changed after capture")


def _read_bound_bytes(
    binding: ExecutionArtifactBinding,
    *,
    label: str,
    max_bytes: int = _MAX_ARTICULATION_INPUT_JSON_BYTES,
) -> bytes:
    path = Path(binding.path)
    try:
        captured = read_contained_artifact(
            path.parent,
            path.name,
            max_bytes=max_bytes,
            capture_bytes=True,
        )
    except (OSError, ValueError) as exc:
        raise EmbeddedArticulationError(f"{label} is unsafe: {exc}") from exc
    observed = ExecutionArtifactBinding(
        path=str(captured.path),
        sha256=captured.sha256,
        size_bytes=captured.size_bytes,
    )
    if observed != binding or captured.data is None:
        raise EmbeddedArticulationError(f"{label} changed after capture")
    return captured.data


def _evidence_digest(preparation: EmbeddedArticulationPreparation) -> str:
    return canonical_json_digest(
        {
            "records": [
                item.model_dump(mode="json") for item in preparation.evidence_records
            ]
        }
    )


class ArticulationProposalProviderRequest(_FrozenModel):
    """Frozen legacy v1 provider input."""

    schema_version: Literal[
        "content-agent-workflows.articulation-proposal-provider-request.v1"
    ] = ARTICULATION_PROPOSAL_PROVIDER_REQUEST_SCHEMA_VERSION
    preparation: ExecutionArtifactBinding
    preparation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    intent: str = Field(min_length=1)
    capability: EmbeddedArticulationProposalCapability
    preparation_payload: EmbeddedArticulationPreparation


class ArticulationProposalProviderRequestV2(_FrozenModel):
    """Current exact request that also binds explicit provider input artifacts."""

    schema_version: Literal[
        "content-agent-workflows.articulation-proposal-provider-request.v2"
    ] = ARTICULATION_PROPOSAL_PROVIDER_REQUEST_V2_SCHEMA_VERSION
    preparation: ExecutionArtifactBinding
    preparation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    intent: str = Field(min_length=1, max_length=16_384)
    capability: EmbeddedArticulationProposalCapability
    preparation_payload: EmbeddedArticulationPreparation
    provider_inputs: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)


type ArticulationProposalProviderRequestContract = (
    ArticulationProposalProviderRequest | ArticulationProposalProviderRequestV2
)


def _request_provider_inputs(
    request: ArticulationProposalProviderRequestContract,
) -> tuple[ExecutionArtifactBinding, ...]:
    if isinstance(request, ArticulationProposalProviderRequestV2):
        return request.provider_inputs
    return ()


class ArticulationProposalProviderPublication(_FrozenModel):
    """Exact output of one explicitly selected replacement-provider call."""

    schema_version: Literal[
        "content-agent-workflows.articulation-proposal-provider-publication.v1"
    ] = ARTICULATION_PROPOSAL_PROVIDER_PUBLICATION_SCHEMA_VERSION
    request: ExecutionArtifactBinding
    native_payload: ExecutionArtifactBinding
    proposal: ExecutionArtifactBinding
    bound_preparation: ExecutionArtifactBinding
    result: EmbeddedArticulationProviderProposal
    nested_agent_launched: Literal[False] = False
    fallback_used: Literal[False] = False
    classic_controller_invoked: Literal[False] = False
    joint_agent_local_client_invoked: Literal[False] = False


class ArticulationProposalAttemptReplacement(_FrozenModel):
    """Exact prior terminal superseded by this independently rooted attempt."""

    prior_terminal_receipt: ExecutionArtifactBinding
    prior_terminal_lineage: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    prior_attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_provider: ProducerIdentity
    reason: str = Field(min_length=1, max_length=4_096)

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        if self.prior_terminal_lineage[0] != self.prior_terminal_receipt:
            raise ValueError("replacement lineage must begin with the prior terminal")
        paths = tuple(item.path for item in self.prior_terminal_lineage)
        if len(paths) != len(set(paths)):
            raise ValueError("replacement lineage must not repeat terminal paths")
        return self


class ArticulationProposalAttemptTerminalReceipt(_FrozenModel):
    """Immutable terminal disposition for one invoked provider attempt."""

    schema_version: Literal[
        "content-agent-workflows.articulation-proposal-attempt-receipt.v1"
    ] = ARTICULATION_PROPOSAL_ATTEMPT_RECEIPT_SCHEMA_VERSION
    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: Literal[
        "succeeded",
        "provider_failure",
        "invalid_response",
        "input_drift",
        "publication_failure",
    ]
    request: ExecutionArtifactBinding
    request_payload: ArticulationProposalProviderRequestContract
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation: ExecutionArtifactBinding
    preparation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: ProducerIdentity
    capability: EmbeddedArticulationProposalCapability
    provider_inputs: tuple[ExecutionArtifactBinding, ...] = ()
    replacement: ArticulationProposalAttemptReplacement | None = None
    failure_code: str | None = Field(default=None, min_length=1)
    failure_summary: str | None = Field(default=None, min_length=1)
    drifted_inputs: tuple[str, ...] = ()
    native_payload: ExecutionArtifactBinding | None = None
    proposal: ExecutionArtifactBinding | None = None
    bound_preparation: ExecutionArtifactBinding | None = None
    partial_outputs: tuple[ExecutionArtifactBinding, ...] = Field(
        default=(), max_length=_MAX_ARTICULATION_PARTIAL_OUTPUTS
    )
    partial_outputs_discarded: bool = False
    nested_agent_launched: Literal[False] = False
    fallback_used: Literal[False] = False
    classic_controller_invoked: Literal[False] = False
    joint_agent_local_client_invoked: Literal[False] = False

    @model_validator(mode="after")
    def validate_terminal_shape(self) -> Self:
        if self.provider_inputs != _request_provider_inputs(self.request_payload):
            raise ValueError("terminal provider inputs differ from the exact request")
        success_bindings = (
            self.native_payload,
            self.proposal,
            self.bound_preparation,
        )
        if self.disposition == "succeeded":
            if any(item is None for item in success_bindings):
                raise ValueError("successful proposal attempts require every output")
            if self.failure_code is not None or self.failure_summary is not None:
                raise ValueError("successful proposal attempts cannot report failure")
            if self.partial_outputs:
                raise ValueError("successful proposal attempts cannot be partial")
            if self.partial_outputs_discarded:
                raise ValueError("successful proposal attempts cannot discard outputs")
        elif self.failure_code is None or self.failure_summary is None:
            raise ValueError("failed proposal attempts require a typed failure")
        elif self.proposal is not None or self.bound_preparation is not None:
            raise ValueError("failed proposal attempts cannot publish a proposal")
        if self.disposition == "input_drift" and not self.drifted_inputs:
            raise ValueError("input_drift attempts require exact drifted inputs")
        if self.disposition != "input_drift" and self.drifted_inputs:
            raise ValueError("only input_drift attempts may name drifted inputs")
        partial_paths = tuple(item.path for item in self.partial_outputs)
        if len(partial_paths) != len(set(partial_paths)):
            raise ValueError("partial proposal outputs must be unique")
        return self


class ArticulationProposalProviderAttemptPublication(_FrozenModel):
    """Successful legacy publication plus its mandatory terminal receipt."""

    schema_version: Literal[
        "content-agent-workflows.articulation-proposal-attempt-publication.v1"
    ] = ARTICULATION_PROPOSAL_ATTEMPT_PUBLICATION_SCHEMA_VERSION
    request: ExecutionArtifactBinding
    native_payload: ExecutionArtifactBinding
    proposal: ExecutionArtifactBinding
    bound_preparation: ExecutionArtifactBinding
    result: EmbeddedArticulationProviderProposal
    nested_agent_launched: Literal[False] = False
    fallback_used: Literal[False] = False
    classic_controller_invoked: Literal[False] = False
    joint_agent_local_client_invoked: Literal[False] = False
    terminal_receipt: ExecutionArtifactBinding


class ArticulationProposalAttemptFailed(EmbeddedArticulationError):
    """Provider attempt failed after publishing an immutable terminal receipt."""

    def __init__(
        self,
        message: str,
        *,
        receipt: ArticulationProposalAttemptTerminalReceipt,
        receipt_binding: ExecutionArtifactBinding,
    ) -> None:
        super().__init__(message)
        self.receipt = receipt
        self.receipt_binding = receipt_binding


class ArticulationProposalProviderFailure(EmbeddedArticulationError):
    """Selected provider failed at transport, HTTP, or runtime level."""


class ArticulationProposalInvalidResponse(EmbeddedArticulationError):
    """Selected provider returned bytes that do not satisfy the typed schema."""


class _ArticulationProposalInputDrift(EmbeddedArticulationError):
    """Request-bound inputs changed after the provider invocation began."""

    def __init__(self, drifted_inputs: tuple[str, ...]) -> None:
        super().__init__("request-bound proposal inputs changed after capture")
        self.drifted_inputs = drifted_inputs


class _ArticulationProposalPublicationFailure(EmbeddedArticulationError):
    """Local create-only attempt publication failed after invocation."""


class ArticulationProposalProvider(Protocol):
    """One explicitly selected provider; the caller never selects a fallback."""

    @property
    def producer(self) -> ProducerIdentity: ...

    @property
    def capability(self) -> EmbeddedArticulationProposalCapability: ...

    @property
    def input_artifacts(self) -> tuple[ExecutionArtifactBinding, ...]: ...

    def propose(
        self,
        request: ArticulationProposalProviderRequestContract,
    ) -> DomainProposalPayload: ...


class _ExplicitProviderAdapter:
    adapter_id: Literal["artifact-json", "http-json"]

    def __init__(
        self,
        *,
        provider_id: str,
        capability_id: str,
        provider_alias: str,
        configuration_sha256: str,
    ) -> None:
        provider_id = provider_id.strip()
        capability_id = capability_id.strip()
        provider_alias = provider_alias.strip()
        if not provider_id or not capability_id or not provider_alias:
            raise ValueError(
                "proposal provider, capability, and non-secret alias must not be empty"
            )
        self._provider_id = provider_id
        self._capability_id = capability_id
        self._provider_alias = provider_alias
        self._configuration_sha256 = configuration_sha256
        self._implementation = (
            f"content_agent_workflows.articulation.{type(self).__name__}"
        )
        self._implementation_sha256 = file_sha256(Path(__file__).resolve())

    @property
    def producer(self) -> ProducerIdentity:
        return ProducerIdentity(
            producer_id=self._provider_id,
            role="proposal_provider",
            implementation=self._implementation,
            implementation_digest=self._implementation_sha256,
        )

    @property
    def capability(self) -> EmbeddedArticulationProposalCapability:
        return EmbeddedArticulationProposalCapability(
            provider_id=self._provider_id,
            capability_id=self._capability_id,
            provider_alias=self._provider_alias,
            adapter_id=self.adapter_id,
            adapter_implementation=self._implementation,
            adapter_implementation_sha256=self._implementation_sha256,
            provider_configuration_sha256=self._configuration_sha256,
        )

    @property
    def input_artifacts(self) -> tuple[ExecutionArtifactBinding, ...]:
        return ()


class ArtifactJsonArticulationProposalProvider(_ExplicitProviderAdapter):
    """Adapt an exact external-provider JSON payload into the public leaf."""

    adapter_id: Literal["artifact-json"] = "artifact-json"

    def __init__(
        self,
        *,
        provider_id: str,
        capability_id: str,
        payload_path: str | Path,
        artifact_alias: str = "artifact-json-payload",
        expected_payload: ExecutionArtifactBinding | None = None,
    ) -> None:
        self._payload = _binding(payload_path, label="proposal provider payload")
        if expected_payload is not None and self._payload != expected_payload:
            raise EmbeddedArticulationError(
                "proposal provider payload differs from the selected-leaf invocation"
            )
        super().__init__(
            provider_id=provider_id,
            capability_id=capability_id,
            provider_alias=artifact_alias,
            configuration_sha256=canonical_json_digest(
                {
                    "adapter_id": self.adapter_id,
                    "payload": self._payload.model_dump(mode="json"),
                }
            ),
        )

    def propose(
        self,
        request: ArticulationProposalProviderRequestContract,
    ) -> DomainProposalPayload:
        del request
        if self._payload.size_bytes > _MAX_ARTICULATION_INPUT_JSON_BYTES:
            raise ArticulationProposalInvalidResponse(
                "selected proposal provider returned an oversized typed payload"
            )
        try:
            return DomainProposalPayload.model_validate_json(
                _read_bound_bytes(
                    self._payload,
                    label="proposal provider payload",
                )
            )
        except (EmbeddedArticulationError, ValidationError) as exc:
            if isinstance(exc, EmbeddedArticulationError):
                raise _ArticulationProposalInputDrift((self._payload.path,)) from exc
            raise ArticulationProposalInvalidResponse(
                "selected proposal provider returned an invalid typed payload"
            ) from exc

    @property
    def input_artifacts(self) -> tuple[ExecutionArtifactBinding, ...]:
        return (self._payload,)


class HttpJsonArticulationProposalProvider(_ExplicitProviderAdapter):
    """POST the typed request to one explicit JSON provider endpoint."""

    adapter_id: Literal["http-json"] = "http-json"

    def __init__(
        self,
        *,
        provider_id: str,
        capability_id: str,
        endpoint_alias: str,
        endpoint_url: str,
        bearer_token: str | None = None,
        timeout_seconds: float = 120.0,
        session: Any | None = None,
    ) -> None:
        alias = endpoint_alias.strip()
        url = endpoint_url.strip().rstrip("/")
        parsed = urlparse(url)
        if not alias:
            raise ValueError("proposal provider endpoint alias must not be empty")
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("proposal provider URL must be a credential-free HTTP URL")
        if parsed.scheme == "http" and parsed.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise ValueError("non-loopback proposal providers require HTTPS transport")
        if timeout_seconds <= 0:
            raise ValueError("proposal provider timeout must be positive")
        self._endpoint_alias = alias
        self._endpoint_url = url
        self._bearer_token = bearer_token
        self._timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        super().__init__(
            provider_id=provider_id,
            capability_id=capability_id,
            provider_alias=alias,
            configuration_sha256=canonical_json_digest(
                {
                    "adapter_id": self.adapter_id,
                    "endpoint_alias": alias,
                    "endpoint_url": url,
                    "timeout_seconds": timeout_seconds,
                }
            ),
        )

    def propose(
        self,
        request: ArticulationProposalProviderRequestContract,
    ) -> DomainProposalPayload:
        headers = {"Content-Type": "application/json"}
        if self._bearer_token is not None:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        try:
            response = self._session.post(
                self._endpoint_url,
                json=request.model_dump(mode="json"),
                headers=headers,
                timeout=self._timeout_seconds,
                allow_redirects=False,
                stream=True,
            )
        except Exception as exc:
            raise ArticulationProposalProviderFailure(
                "selected proposal provider failed without fallback at runtime"
            ) from exc
        try:
            if response.status_code != 200:
                raise ArticulationProposalProviderFailure(
                    "selected proposal provider failed without fallback: "
                    f"{self._endpoint_alias} returned HTTP {response.status_code}"
                )
            content_length = getattr(response, "headers", {}).get("Content-Length")
            try:
                if (
                    content_length is not None
                    and int(content_length) > _MAX_ARTICULATION_INPUT_JSON_BYTES
                ):
                    raise ArticulationProposalInvalidResponse(
                        "selected proposal provider returned an oversized payload"
                    )
            except (TypeError, ValueError) as exc:
                raise ArticulationProposalInvalidResponse(
                    "selected proposal provider returned an invalid content length"
                ) from exc
            document = bytearray()
            try:
                for chunk in response.iter_content(chunk_size=64 * 1_024):
                    if not chunk:
                        continue
                    document.extend(chunk)
                    if len(document) > _MAX_ARTICULATION_INPUT_JSON_BYTES:
                        raise ArticulationProposalInvalidResponse(
                            "selected proposal provider returned an oversized payload"
                        )
            except ArticulationProposalInvalidResponse:
                raise
            except Exception as exc:
                raise ArticulationProposalProviderFailure(
                    "selected proposal provider failed without fallback at runtime"
                ) from exc
            try:
                return DomainProposalPayload.model_validate_json(document)
            except ValidationError as exc:
                raise ArticulationProposalInvalidResponse(
                    "selected proposal provider returned an invalid typed payload"
                ) from exc
        finally:
            try:
                response.close()
            except Exception:
                pass


def bind_embedded_articulation_provider_proposal(
    preparation: EmbeddedArticulationPreparation,
    proposal: EmbeddedArticulationProviderProposal,
) -> EmbeddedArticulationPreparation:
    """Bind a v2 proposal only to an explicitly selected pending leaf."""

    if preparation.proposal_status != "not_evaluated" or preparation.proposal:
        raise EmbeddedArticulationError(
            "proposal binding requires an explicitly selected not_evaluated leaf"
        )
    if proposal.schema_version != _PROPOSAL_SCHEMA_VERSION:
        raise EmbeddedArticulationError("replacement providers require proposal v2")
    if proposal.preparation_digest != canonical_json_digest(preparation):
        raise EmbeddedArticulationError("proposal binds another preparation")
    if proposal.evidence_digest != _evidence_digest(preparation):
        raise EmbeddedArticulationError("proposal binds another evidence set")
    return preparation.model_copy(
        update={"proposal_status": "available", "proposal": proposal}
    )


def _fresh_attempt_root(path: str | Path) -> Path:
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    try:
        resolved = absolute.resolve(strict=False)
    except OSError as exc:
        raise EmbeddedArticulationError(
            f"proposal publication root is unsafe: {absolute}"
        ) from exc
    if resolved != absolute:
        raise EmbeddedArticulationError(
            f"proposal publication root traverses a symlink: {absolute}"
        )
    try:
        prepare_writable_directory(absolute.parent)
        absolute.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise EmbeddedArticulationError(
            "proposal publication output already exists; each provider attempt "
            "requires a fresh output directory"
        ) from exc
    except (OSError, ValueError) as exc:
        raise EmbeddedArticulationError(
            f"proposal publication root is unsafe: {absolute}"
        ) from exc
    try:
        metadata = absolute.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or absolute.resolve(strict=True) != absolute
        ):
            raise ValueError("created attempt root is not the requested directory")
    except (OSError, ValueError) as exc:
        raise EmbeddedArticulationError(
            f"proposal publication root is unsafe after creation: {absolute}"
        ) from exc
    return absolute


def _json_document(payload: BaseModel) -> bytes:
    document = (
        json.dumps(
            payload.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
        + b"\n"
    )
    if len(document) > _MAX_ARTICULATION_GENERATED_JSON_BYTES:
        raise EmbeddedArticulationError(
            "generated proposal-attempt JSON exceeds the bounded limit"
        )
    return document


def _write_json_create_only(root: Path, name: str, payload: BaseModel) -> Path:
    document = _json_document(payload)
    root_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    file_fd = -1
    try:
        file_fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
            dir_fd=root_fd,
        )
        os.fchmod(file_fd, 0o444)
        view = memoryview(document)
        while view:
            written = os.write(file_fd, view)
            if written <= 0:  # pragma: no cover - defensive OS contract guard
                raise OSError("create-only provider-attempt write made no progress")
            view = view[written:]
        os.fsync(file_fd)
        os.fsync(root_fd)
    except BaseException:
        if file_fd >= 0:
            os.close(file_fd)
            file_fd = -1
            try:
                os.unlink(name, dir_fd=root_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(root_fd)
    return root / name


def _published_binding(
    root: Path,
    name: str,
    *,
    label: str,
    max_bytes: int | None = None,
) -> ExecutionArtifactBinding:
    try:
        read = read_contained_artifact(root, name, max_bytes=max_bytes)
    except (OSError, ValueError) as exc:  # pragma: no cover - write invariant
        raise EmbeddedArticulationError(f"{label} is unsafe: {exc}") from exc
    return ExecutionArtifactBinding(
        path=str(read.path),
        sha256=read.sha256,
        size_bytes=read.size_bytes,
    )


def _publish_attempt_artifact(
    root: Path,
    name: str,
    payload: BaseModel,
    *,
    label: str,
) -> ExecutionArtifactBinding:
    try:
        _write_json_create_only(root, name, payload)
        return _published_binding(root, name, label=label)
    except (EmbeddedArticulationError, OSError, ValueError) as exc:
        raise _ArticulationProposalPublicationFailure(
            f"{label} could not be published"
        ) from exc


def _reset_unfinalized_terminal(root: Path) -> None:
    """Rollback only an unsealed terminal created by a failed finalization."""

    terminal_name = "articulation_proposal_attempt_terminal.json"
    root_fd = -1
    try:
        root_fd = os.open(
            root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        os.fchmod(root_fd, 0o700)
        try:
            metadata = os.stat(terminal_name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("unfinalized terminal path is unsafe")
        os.unlink(terminal_name, dir_fd=root_fd)
        os.fsync(root_fd)
    except (OSError, ValueError) as exc:
        raise EmbeddedArticulationError(
            "cannot reset failed proposal-attempt finalization"
        ) from exc
    finally:
        if root_fd >= 0:
            os.close(root_fd)


def _discard_unpublishable_partial(root: Path, name: str) -> None:
    """Remove one unsafe or unreadable output from the fresh attempt root."""

    root_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        try:
            os.unlink(name, dir_fd=root_fd)
        except OSError as unlink_error:
            try:
                metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except OSError:
                raise unlink_error
            if not stat.S_ISDIR(metadata.st_mode):
                raise unlink_error
            shutil.rmtree(root / name)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)


def _partial_output_is_unsafe(root: Path, name: str) -> bool:
    root_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        return not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
    finally:
        os.close(root_fd)


def _seal_partial_output(root: Path, name: str) -> None:
    root_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    file_fd = -1
    try:
        file_fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OSError("partial proposal output is not a private regular file")
        os.fchmod(file_fd, 0o444)
        os.fsync(file_fd)
        os.fsync(root_fd)
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(root_fd)


def _capture_partial_outputs(
    root: Path,
    *,
    include_native_payload_as_partial: bool,
) -> tuple[tuple[ExecutionArtifactBinding, ...], bool, bool]:
    partial: list[ExecutionArtifactBinding] = []
    excluded = {
        "articulation_proposal_provider_request.json",
        "articulation_proposal_attempt_terminal.json",
    }
    if not include_native_payload_as_partial:
        excluded.add("articulation_proposal_provider_payload.json")
    preferred_order = [
        "articulation_proposal_provider_payload.json",
        "embedded_articulation_provider_proposal.json",
        "embedded_articulation_preparation.json",
    ]
    inventory: list[str]
    try:
        inventory = sorted(os.listdir(root))
    except OSError as exc:
        raise EmbeddedArticulationError(
            "cannot inventory partial proposal attempt outputs"
        ) from exc
    preferred_present = (set(inventory) - excluded) & set(preferred_order)
    selected = [
        name
        for name in inventory
        if name not in excluded and name not in preferred_present
    ]
    names = [item for item in preferred_order if item in preferred_present][
        :_MAX_ARTICULATION_PARTIAL_OUTPUTS
    ]
    names.extend(selected[: _MAX_ARTICULATION_PARTIAL_OUTPUTS - len(names)])
    retained_names = set(names)
    excess_names = [
        name
        for name in inventory
        if name not in excluded and name not in retained_names
    ]
    overflow = bool(excess_names)
    discarded = False
    for name in excess_names:
        try:
            _discard_unpublishable_partial(root, name)
            discarded = True
        except OSError as cleanup_exc:
            raise EmbeddedArticulationError(
                f"cannot discard excess partial proposal output: {name}"
            ) from cleanup_exc
    for name in names:
        try:
            binding = _published_binding(
                root,
                name,
                label="partial proposal attempt output",
                max_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
            )
            _seal_partial_output(root, name)
            partial.append(binding)
        except (EmbeddedArticulationError, OSError):
            try:
                unsafe = _partial_output_is_unsafe(root, name)
            except FileNotFoundError:
                discarded = True
                continue
            except OSError:
                unsafe = True
            if not unsafe:
                # Preserve a regular single-link forensic artifact and retry
                # one transient read instead of deleting exact attempt output.
                try:
                    binding = _published_binding(
                        root,
                        name,
                        label="partial proposal attempt output",
                        max_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
                    )
                    _seal_partial_output(root, name)
                    partial.append(binding)
                    continue
                except (EmbeddedArticulationError, OSError):
                    pass
            try:
                _discard_unpublishable_partial(root, name)
                discarded = True
            except OSError as cleanup_exc:
                raise EmbeddedArticulationError(
                    f"cannot discard unsafe partial proposal output: {name}"
                ) from cleanup_exc
    return tuple(partial), overflow, discarded


def _seal_attempt_root(root: Path) -> None:
    root_fd = -1
    try:
        root_fd = os.open(
            root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        os.fchmod(root_fd, 0o500)
        os.fsync(root_fd)
    except OSError as exc:
        raise EmbeddedArticulationError(
            f"cannot seal proposal attempt root: {root}"
        ) from exc
    finally:
        if root_fd >= 0:
            os.close(root_fd)


def _validate_sealed_attempt_root(root: Path, names: set[str]) -> None:
    try:
        root_metadata = root.lstat()
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or stat.S_IMODE(root_metadata.st_mode) != 0o500
        ):
            raise ValueError("attempt root is not sealed read-only")
        for name in names:
            metadata = (root / name).lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o444
            ):
                raise ValueError(f"attempt artifact is mutable or unsafe: {name}")
    except (OSError, ValueError) as exc:
        raise EmbeddedArticulationError(
            f"proposal attempt root is not sealed: {exc}"
        ) from exc


def _attempt_entry_names(root: Path) -> set[str]:
    try:
        return {item.name for item in root.iterdir()}
    except OSError as exc:
        raise EmbeddedArticulationError(
            "proposal attempt entries are unavailable"
        ) from exc


def _replacement_binding(
    terminal_receipt_path: str | Path | None,
    *,
    reason: str | None,
    expected_terminal: ExecutionArtifactBinding | None = None,
) -> ArticulationProposalAttemptReplacement | None:
    if terminal_receipt_path is None:
        if reason is not None or expected_terminal is not None:
            raise ValueError("replacement reason requires a prior terminal receipt")
        return None
    replacement_reason = (reason or "").strip()
    if not replacement_reason:
        raise ValueError("replacement attempts require an explicit reason")
    prior_binding = _binding(
        terminal_receipt_path,
        label="superseded proposal terminal receipt",
    )
    if expected_terminal is not None and prior_binding != expected_terminal:
        raise EmbeddedArticulationError(
            "replacement terminal differs from the selected-leaf invocation"
        )
    prior = validate_articulation_proposal_attempt_receipt(prior_binding.path)
    observed_prior_binding = _binding(
        prior_binding.path,
        label="superseded proposal terminal receipt",
    )
    if observed_prior_binding != prior_binding or (
        expected_terminal is not None and observed_prior_binding != expected_terminal
    ):
        raise EmbeddedArticulationError(
            "replacement terminal changed during selected-leaf validation"
        )
    prior_lineage = (
        prior_binding,
        *(
            prior.replacement.prior_terminal_lineage
            if prior.replacement is not None
            else ()
        ),
    )
    if len(prior_lineage) >= _MAX_REPLACEMENT_LINEAGE_DEPTH:
        raise EmbeddedArticulationError(
            "proposal attempt replacement lineage cannot accept another attempt"
        )
    return ArticulationProposalAttemptReplacement(
        prior_terminal_receipt=prior_binding,
        prior_terminal_lineage=prior_lineage,
        prior_attempt_id=prior.attempt_id,
        prior_request_digest=prior.request_digest,
        prior_provider=prior.provider,
        reason=replacement_reason,
    )


def _attempt_id(
    request_binding: ExecutionArtifactBinding,
    replacement: ArticulationProposalAttemptReplacement | None,
) -> str:
    return canonical_json_digest(
        {
            "schema_version": (
                "content-agent-workflows.articulation-proposal-attempt-identity.v1"
            ),
            "request": request_binding.model_dump(mode="json"),
            "replacement": (
                replacement.model_dump(mode="json") if replacement is not None else None
            ),
        }
    )


def _failure_disposition(
    error: BaseException,
) -> tuple[
    Literal["provider_failure", "invalid_response", "publication_failure"],
    str,
    str,
]:
    if not isinstance(error, Exception):
        return (
            "publication_failure",
            "attempt_interrupted",
            "Proposal-attempt execution was interrupted locally.",
        )
    if isinstance(error, _ArticulationProposalPublicationFailure | MemoryError):
        return (
            "publication_failure",
            "attempt_publication_failure",
            "Local create-only proposal-attempt publication failed.",
        )
    if isinstance(error, ArticulationProposalInvalidResponse | ValidationError):
        return (
            "invalid_response",
            "invalid_typed_response",
            "Selected provider returned an invalid typed response.",
        )
    if isinstance(error, ArticulationProposalProviderFailure):
        return (
            "provider_failure",
            "provider_runtime_failure",
            "Selected provider failed at HTTP, transport, or runtime level.",
        )
    if isinstance(error, EmbeddedArticulationError):
        return (
            "publication_failure",
            "attempt_publication_failure",
            "Local create-only proposal-attempt publication failed.",
        )
    return (
        "provider_failure",
        "provider_runtime_failure",
        "Selected provider failed at HTTP, transport, or runtime level.",
    )


def _terminal_receipt(
    *,
    disposition: Literal[
        "succeeded",
        "provider_failure",
        "invalid_response",
        "input_drift",
        "publication_failure",
    ],
    request: ArticulationProposalProviderRequestContract,
    request_binding: ExecutionArtifactBinding,
    provider: ProducerIdentity,
    replacement: ArticulationProposalAttemptReplacement | None,
    failure_code: str | None = None,
    failure_summary: str | None = None,
    drifted_inputs: tuple[str, ...] = (),
    native_payload: ExecutionArtifactBinding | None = None,
    proposal: ExecutionArtifactBinding | None = None,
    bound_preparation: ExecutionArtifactBinding | None = None,
    partial_outputs: tuple[ExecutionArtifactBinding, ...] = (),
    partial_outputs_discarded: bool = False,
) -> ArticulationProposalAttemptTerminalReceipt:
    return ArticulationProposalAttemptTerminalReceipt(
        attempt_id=_attempt_id(request_binding, replacement),
        disposition=disposition,
        request=request_binding,
        request_payload=request,
        request_digest=canonical_json_digest(request),
        preparation=request.preparation,
        preparation_digest=request.preparation_digest,
        evidence_digest=request.evidence_digest,
        source_sha256=request.source_sha256,
        source_dependency_bundle_sha256=(request.source_dependency_bundle_sha256),
        configuration_sha256=request.configuration_sha256,
        provider=provider,
        capability=request.capability,
        provider_inputs=_request_provider_inputs(request),
        replacement=replacement,
        failure_code=failure_code,
        failure_summary=failure_summary,
        drifted_inputs=drifted_inputs,
        native_payload=native_payload,
        proposal=proposal,
        bound_preparation=bound_preparation,
        partial_outputs=partial_outputs,
        partial_outputs_discarded=partial_outputs_discarded,
    )


def _prospective_output_binding(root: Path, name: str) -> ExecutionArtifactBinding:
    return ExecutionArtifactBinding(
        path=str(root / name),
        sha256="f" * 64,
        size_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
    )


def _preflight_terminal_capacity(
    *,
    root: Path,
    request: ArticulationProposalProviderRequestContract,
    request_binding: ExecutionArtifactBinding,
    provider: ProducerIdentity,
    replacement: ArticulationProposalAttemptReplacement | None,
) -> None:
    """Prove every bounded terminal shape fits before invoking a provider."""

    native_payload = _prospective_output_binding(
        root,
        "articulation_proposal_provider_payload.json",
    )
    drifted_inputs = tuple(
        dict.fromkeys(
            (
                request_binding.path,
                request.preparation.path,
                *(item.path for item in _request_provider_inputs(request)),
                *(
                    artifact.path
                    for record in request.preparation_payload.evidence_records
                    for artifact in record.artifacts
                ),
                *(
                    item.path
                    for item in (
                        replacement.prior_terminal_lineage
                        if replacement is not None
                        else ()
                    )
                ),
                "provider-identity",
                "provider-input-identity",
            )
        )
    )
    six_character_escape_controls = tuple(
        chr(code) for code in range(1, 32) if code not in {8, 9, 10, 12, 13}
    )
    partial_names = tuple(
        "\x01" * 253
        + six_character_escape_controls[index // len(six_character_escape_controls)]
        + six_character_escape_controls[index % len(six_character_escape_controls)]
        for index in range(_MAX_ARTICULATION_PARTIAL_OUTPUTS)
    )
    partial_outputs = tuple(
        _prospective_output_binding(root, name) for name in partial_names
    )
    # Budget every failure disposition with every request-bound input path,
    # 128 worst-case legal 255-byte filenames whose one-byte units each need
    # six JSON escape characters, and the overflow disclosure.
    failure_shapes: tuple[
        tuple[
            Literal[
                "provider_failure",
                "invalid_response",
                "input_drift",
                "publication_failure",
            ],
            str,
            str,
        ],
        ...,
    ] = (
        (
            "input_drift",
            "bound_input_drift",
            "A request, preparation, or evidence artifact changed.",
        ),
        (
            "provider_failure",
            "provider_runtime_failure",
            "Selected provider failed at HTTP, transport, or runtime level.",
        ),
        (
            "invalid_response",
            "invalid_typed_response",
            "Selected provider returned an invalid typed response.",
        ),
        (
            "publication_failure",
            "attempt_publication_failure",
            "Local create-only proposal-attempt publication failed.",
        ),
        (
            "publication_failure",
            "attempt_interrupted",
            "Proposal-attempt execution was interrupted locally.",
        ),
    )
    for disposition, failure_code, failure_summary in failure_shapes:
        terminal = _terminal_receipt(
            disposition=disposition,
            request=request,
            request_binding=request_binding,
            provider=provider,
            replacement=replacement,
            failure_code=failure_code,
            failure_summary=(f"{failure_summary} {_PARTIAL_OUTPUT_OVERFLOW_SUMMARY}"),
            drifted_inputs=(drifted_inputs if disposition == "input_drift" else ()),
            native_payload=native_payload,
            partial_outputs=partial_outputs,
        )
        _json_document(terminal)


def _drifted_inputs(
    request_binding: ExecutionArtifactBinding,
    preparation_binding: ExecutionArtifactBinding,
    preparation: EmbeddedArticulationPreparation,
    *,
    provider: ArticulationProposalProvider,
    provider_inputs: tuple[ExecutionArtifactBinding, ...],
    producer: ProducerIdentity,
    capability: EmbeddedArticulationProposalCapability,
    replacement: ArticulationProposalAttemptReplacement | None,
) -> tuple[str, ...]:
    drifted: list[str] = []
    try:
        observed_provider_inputs = provider.input_artifacts
        if observed_provider_inputs != provider_inputs:
            drifted.extend(item.path for item in provider_inputs)
            drifted.append("provider-input-identity")
    except BaseException:
        drifted.extend(item.path for item in provider_inputs)
        drifted.append("provider-input-identity")
    for binding in (
        request_binding,
        preparation_binding,
        *provider_inputs,
        *(
            artifact
            for record in preparation.evidence_records
            for artifact in record.artifacts
        ),
        *(replacement.prior_terminal_lineage if replacement is not None else ()),
    ):
        try:
            observed = _binding(binding.path, label="proposal attempt input")
            if binding.path == request_binding.path:
                metadata = Path(binding.path).lstat()
                if stat.S_IMODE(metadata.st_mode) != 0o444:
                    drifted.append(binding.path)
                    continue
        except (EmbeddedArticulationError, OSError, ValueError):
            drifted.append(binding.path)
            continue
        if observed != binding:
            drifted.append(binding.path)
    try:
        if provider.producer != producer or provider.capability != capability:
            drifted.append("provider-identity")
    except BaseException:
        drifted.append("provider-identity")
    return tuple(dict.fromkeys(drifted))


def request_embedded_articulation_provider_proposal(
    preparation_path: str | Path,
    *,
    output_dir: str | Path,
    intent: str,
    provider: ArticulationProposalProvider,
    replaces_terminal_receipt: str | Path | None = None,
    replacement_reason: str | None = None,
    expected_preparation: ExecutionArtifactBinding | None = None,
    expected_replacement_terminal: ExecutionArtifactBinding | None = None,
) -> ArticulationProposalProviderAttemptPublication:
    """Invoke one provider and publish exact proposal/preparation bindings."""

    preparation_binding = _binding(
        preparation_path,
        label="provider-neutral Articulation preparation",
    )
    if expected_preparation is not None and preparation_binding != expected_preparation:
        raise EmbeddedArticulationError(
            "Articulation preparation differs from the selected-leaf invocation"
        )
    preparation = EmbeddedArticulationPreparation.model_validate_json(
        _read_bound_bytes(
            preparation_binding,
            label="provider-neutral Articulation preparation",
            max_bytes=_MAX_ARTICULATION_PREPARATION_JSON_BYTES,
        )
    )
    if preparation.proposal_status != "not_evaluated" or preparation.proposal:
        raise EmbeddedArticulationError(
            "proposal provider requires an explicitly selected not_evaluated leaf"
        )
    for record in preparation.evidence_records:
        for artifact in record.artifacts:
            _verify_binding(
                artifact,
                label=f"Articulation evidence {record.evidence_id}",
            )
    capability = provider.capability
    producer = provider.producer
    provider_inputs = provider.input_artifacts
    if (
        producer.role != "proposal_provider"
        or capability.provider_id != producer.producer_id
    ):
        raise EmbeddedArticulationError("selected provider provenance is inconsistent")
    replacement = _replacement_binding(
        replaces_terminal_receipt,
        reason=replacement_reason,
        expected_terminal=expected_replacement_terminal,
    )
    intent = intent.strip()
    if not intent:
        raise ValueError("proposal provider intent must not be empty")
    preparation_digest = canonical_json_digest(preparation)
    evidence_digest = _evidence_digest(preparation)
    request: ArticulationProposalProviderRequestContract
    if provider_inputs:
        request = ArticulationProposalProviderRequestV2(
            preparation=preparation_binding,
            preparation_digest=preparation_digest,
            evidence_digest=evidence_digest,
            source_sha256=preparation.source_sha256,
            source_dependency_bundle_sha256=(
                preparation.source_dependency_bundle_sha256
            ),
            configuration_sha256=preparation.configuration_sha256,
            intent=intent,
            capability=capability,
            preparation_payload=preparation,
            provider_inputs=provider_inputs,
        )
    else:
        request = ArticulationProposalProviderRequest(
            preparation=preparation_binding,
            preparation_digest=preparation_digest,
            evidence_digest=evidence_digest,
            source_sha256=preparation.source_sha256,
            source_dependency_bundle_sha256=(
                preparation.source_dependency_bundle_sha256
            ),
            configuration_sha256=preparation.configuration_sha256,
            intent=intent,
            capability=capability,
            preparation_payload=preparation,
        )
    # Refuse a request whose largest terminal shape cannot fit before creating
    # a root or invoking a provider.
    request_document = _json_document(request)
    predicted_root = Path(os.path.abspath(Path(output_dir).expanduser()))
    predicted_request_binding = ExecutionArtifactBinding(
        path=str(predicted_root / "articulation_proposal_provider_request.json"),
        sha256=hashlib.sha256(request_document).hexdigest(),
        size_bytes=len(request_document),
    )
    _preflight_terminal_capacity(
        root=predicted_root,
        request=request,
        request_binding=predicted_request_binding,
        provider=producer,
        replacement=replacement,
    )
    root = _fresh_attempt_root(output_dir)
    request_path = root / "articulation_proposal_provider_request.json"
    _write_json_create_only(root, request_path.name, request)
    request_binding = _published_binding(
        root,
        request_path.name,
        label="proposal provider request",
    )
    if request_binding != predicted_request_binding:
        raise EmbeddedArticulationError(
            "proposal provider request differed from its preflight identity"
        )
    native_payload_binding: ExecutionArtifactBinding | None = None
    try:
        changed_inputs = _drifted_inputs(
            request_binding,
            preparation_binding,
            preparation,
            provider=provider,
            provider_inputs=provider_inputs,
            producer=producer,
            capability=capability,
            replacement=replacement,
        )
        if changed_inputs:
            raise _ArticulationProposalInputDrift(changed_inputs)
        native_payload = DomainProposalPayload.model_validate(provider.propose(request))
        changed_inputs = _drifted_inputs(
            request_binding,
            preparation_binding,
            preparation,
            provider=provider,
            provider_inputs=provider_inputs,
            producer=producer,
            capability=capability,
            replacement=replacement,
        )
        if changed_inputs:
            raise _ArticulationProposalInputDrift(changed_inputs)
        native_payload_path = root / "articulation_proposal_provider_payload.json"
        native_payload_binding = _publish_attempt_artifact(
            root,
            native_payload_path.name,
            native_payload,
            label="proposal provider native payload",
        )
        request_digest = canonical_json_digest(request)
        payload = DomainProposalPayload(
            schema_version=_WRAPPED_PAYLOAD_SCHEMA_VERSION,
            values={
                "provider_request_digest": request_digest,
                "preparation_digest": request.preparation_digest,
                "evidence_digest": request.evidence_digest,
                "provider_capability": capability.model_dump(mode="json"),
                "native_payload": native_payload.model_dump(mode="json"),
                "native_payload_binding": native_payload_binding.model_dump(
                    mode="json"
                ),
            },
        )
        proposal = EmbeddedArticulationProviderProposal(
            schema_version=_PROPOSAL_SCHEMA_VERSION,
            producer=producer,
            payload=payload,
            preparation_digest=request.preparation_digest,
            evidence_digest=request.evidence_digest,
            provider_request_digest=request_digest,
            capability=capability,
            provider_request=request_binding,
            native_payload=native_payload_binding,
        )
        bound_preparation = bind_embedded_articulation_provider_proposal(
            preparation,
            proposal,
        )
        proposal_path = root / "embedded_articulation_provider_proposal.json"
        bound_preparation_path = root / "embedded_articulation_preparation.json"
        proposal_binding = _publish_attempt_artifact(
            root,
            proposal_path.name,
            proposal,
            label="Articulation proposal",
        )
        bound_preparation_binding = _publish_attempt_artifact(
            root,
            bound_preparation_path.name,
            bound_preparation,
            label="proposal-bound Articulation preparation",
        )
        if (
            EmbeddedArticulationProviderProposal.model_validate_json(
                _read_bound_bytes(
                    proposal_binding,
                    label="Articulation proposal",
                    max_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
                )
            )
            != proposal
            or EmbeddedArticulationPreparation.model_validate_json(
                _read_bound_bytes(
                    bound_preparation_binding,
                    label="proposal-bound Articulation preparation",
                    max_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
                )
            )
            != bound_preparation
        ):
            raise _ArticulationProposalPublicationFailure(
                "proposal publication exact readback failed"
            )
        changed_inputs = _drifted_inputs(
            request_binding,
            preparation_binding,
            preparation,
            provider=provider,
            provider_inputs=provider_inputs,
            producer=producer,
            capability=capability,
            replacement=replacement,
        )
        if changed_inputs:
            raise _ArticulationProposalInputDrift(changed_inputs)
        receipt = _terminal_receipt(
            disposition="succeeded",
            request=request,
            request_binding=request_binding,
            provider=producer,
            replacement=replacement,
            native_payload=native_payload_binding,
            proposal=proposal_binding,
            bound_preparation=bound_preparation_binding,
        )
        try:
            receipt_path = _write_json_create_only(
                root,
                "articulation_proposal_attempt_terminal.json",
                receipt,
            )
            receipt_binding = _published_binding(
                root,
                receipt_path.name,
                label="proposal attempt terminal receipt",
            )
            _seal_attempt_root(root)
            if validate_articulation_proposal_attempt_receipt(receipt_path) != receipt:
                raise EmbeddedArticulationError(
                    "proposal attempt terminal self-validation differed"
                )
        except (EmbeddedArticulationError, OSError, ValueError) as exc:
            raise _ArticulationProposalPublicationFailure(
                "successful proposal-attempt finalization failed"
            ) from exc
        return ArticulationProposalProviderAttemptPublication(
            request=request_binding,
            native_payload=native_payload_binding,
            proposal=proposal_binding,
            bound_preparation=bound_preparation_binding,
            result=proposal,
            terminal_receipt=receipt_binding,
        )
    except BaseException as exc:
        _reset_unfinalized_terminal(root)
        observed_drift = _drifted_inputs(
            request_binding,
            preparation_binding,
            preparation,
            provider=provider,
            provider_inputs=provider_inputs,
            producer=producer,
            capability=capability,
            replacement=replacement,
        )
        drifted_inputs = tuple(
            dict.fromkeys(
                (
                    *(
                        exc.drifted_inputs
                        if isinstance(exc, _ArticulationProposalInputDrift)
                        else ()
                    ),
                    *observed_drift,
                )
            )
        )
        disposition: Literal[
            "provider_failure",
            "invalid_response",
            "input_drift",
            "publication_failure",
        ]
        if drifted_inputs:
            disposition = "input_drift"
            failure_code = "bound_input_drift"
            failure_summary = "A request, preparation, or evidence artifact changed."
        else:
            disposition, failure_code, failure_summary = _failure_disposition(exc)
        if request_binding.path in drifted_inputs:
            try:
                _discard_unpublishable_partial(
                    root,
                    "articulation_proposal_provider_request.json",
                )
            except FileNotFoundError:
                pass
            except OSError as cleanup_exc:
                raise EmbeddedArticulationError(
                    "cannot discard drifted proposal request"
                ) from cleanup_exc
        (
            partial_outputs,
            partial_output_overflow,
            partial_outputs_discarded,
        ) = _capture_partial_outputs(
            root,
            include_native_payload_as_partial=native_payload_binding is None,
        )
        if partial_output_overflow:
            failure_summary = f"{failure_summary} {_PARTIAL_OUTPUT_OVERFLOW_SUMMARY}"
        receipt = _terminal_receipt(
            disposition=disposition,
            request=request,
            request_binding=request_binding,
            provider=producer,
            replacement=replacement,
            failure_code=failure_code,
            failure_summary=failure_summary,
            drifted_inputs=drifted_inputs,
            native_payload=native_payload_binding,
            partial_outputs=partial_outputs,
            partial_outputs_discarded=partial_outputs_discarded,
        )
        receipt_path = _write_json_create_only(
            root,
            "articulation_proposal_attempt_terminal.json",
            receipt,
        )
        receipt_binding = _published_binding(
            root,
            receipt_path.name,
            label="proposal attempt terminal receipt",
        )
        _seal_attempt_root(root)
        if validate_articulation_proposal_attempt_receipt(receipt_path) != receipt:
            raise EmbeddedArticulationError(
                "proposal attempt terminal self-validation differed"
            )
        if isinstance(exc, Exception):
            public_message = (
                str(exc)
                if isinstance(
                    exc,
                    ArticulationProposalProviderFailure
                    | ArticulationProposalInvalidResponse
                    | EmbeddedArticulationError,
                )
                else failure_summary
            )
            raise ArticulationProposalAttemptFailed(
                public_message,
                receipt=receipt,
                receipt_binding=receipt_binding,
            ) from exc
        raise


def _validate_terminal_receipt(
    terminal_receipt_path: str | Path,
    *,
    seen: set[Path],
    known_drifted_lineage: set[str] | None = None,
) -> ArticulationProposalAttemptTerminalReceipt:
    receipt_binding = _binding(
        terminal_receipt_path,
        label="proposal attempt terminal receipt",
    )
    receipt_path = Path(receipt_binding.path)
    if receipt_path.name != "articulation_proposal_attempt_terminal.json":
        raise EmbeddedArticulationError(
            "proposal attempt terminal receipt has a non-canonical filename"
        )
    if receipt_path in seen:
        raise EmbeddedArticulationError(
            "proposal attempt replacement receipts contain a cycle"
        )
    if len(seen) >= _MAX_REPLACEMENT_LINEAGE_DEPTH:
        raise EmbeddedArticulationError(
            "proposal attempt replacement lineage exceeds the bounded depth"
        )
    seen.add(receipt_path)
    try:
        receipt = ArticulationProposalAttemptTerminalReceipt.model_validate_json(
            _read_bound_bytes(
                receipt_binding,
                label="proposal attempt terminal receipt",
                max_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
            )
        )
    except (EmbeddedArticulationError, ValidationError) as exc:
        raise EmbeddedArticulationError(
            "proposal attempt terminal receipt is invalid"
        ) from exc
    root = receipt_path.parent
    drifted_inputs = (
        set(receipt.drifted_inputs) if receipt.disposition == "input_drift" else set()
    )
    expected_names = {receipt_path.name}
    if receipt.request.path not in drifted_inputs:
        expected_names.add("articulation_proposal_provider_request.json")
    output_bindings = (
        receipt.native_payload,
        receipt.proposal,
        receipt.bound_preparation,
        *receipt.partial_outputs,
    )
    for binding in output_bindings:
        if binding is not None:
            if Path(binding.path).parent != root:
                raise EmbeddedArticulationError(
                    "proposal attempt output escaped its create-only root"
                )
            expected_names.add(Path(binding.path).name)
            _verify_binding(binding, label="proposal attempt output")
    if _attempt_entry_names(root) != expected_names:
        raise EmbeddedArticulationError(
            "proposal attempt root contains unbound or mutable entries"
        )
    _validate_sealed_attempt_root(root, expected_names)
    if Path(receipt.request.path).parent != root:
        raise EmbeddedArticulationError(
            "proposal attempt request escaped its create-only root"
        )
    if receipt.request.path in drifted_inputs:
        request = receipt.request_payload
    else:
        _verify_binding(receipt.request, label="proposal provider request")
        try:
            request_bytes = _read_bound_bytes(
                receipt.request,
                label="proposal provider request",
                max_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
            )
            request_model = (
                ArticulationProposalProviderRequestV2
                if isinstance(
                    receipt.request_payload,
                    ArticulationProposalProviderRequestV2,
                )
                else ArticulationProposalProviderRequest
            )
            request = request_model.model_validate_json(request_bytes)
        except (EmbeddedArticulationError, ValidationError) as exc:
            raise EmbeddedArticulationError(
                "proposal attempt request is invalid"
            ) from exc
    if (
        request != receipt.request_payload
        or canonical_json_digest(request) != receipt.request_digest
        or request.preparation != receipt.preparation
        or request.preparation_digest != receipt.preparation_digest
        or request.evidence_digest != receipt.evidence_digest
        or request.source_sha256 != receipt.source_sha256
        or request.source_dependency_bundle_sha256
        != receipt.source_dependency_bundle_sha256
        or request.configuration_sha256 != receipt.configuration_sha256
        or request.capability != receipt.capability
        or receipt.provider.role != "proposal_provider"
        or request.capability.provider_id != receipt.provider.producer_id
        or request.capability.adapter_implementation_sha256
        != receipt.provider.implementation_digest
        or _attempt_id(receipt.request, receipt.replacement) != receipt.attempt_id
    ):
        raise EmbeddedArticulationError(
            "proposal attempt terminal identity differs from its exact request"
        )
    if receipt.disposition == "input_drift":
        preparation = request.preparation_payload
        allowed_drifted_inputs = {
            receipt.request.path,
            receipt.preparation.path,
            *(item.path for item in receipt.provider_inputs),
            *(
                artifact.path
                for record in preparation.evidence_records
                for artifact in record.artifacts
            ),
            *(
                tuple(item.path for item in receipt.replacement.prior_terminal_lineage)
                if receipt.replacement is not None
                else ()
            ),
            "provider-identity",
            "provider-input-identity",
        }
        if not drifted_inputs <= allowed_drifted_inputs:
            raise EmbeddedArticulationError(
                "proposal attempt terminal names an unbound drifted input"
            )
        if receipt.preparation.path not in drifted_inputs:
            _verify_binding(receipt.preparation, label="provider-neutral preparation")
            try:
                preparation = EmbeddedArticulationPreparation.model_validate_json(
                    _read_bound_bytes(
                        receipt.preparation,
                        label="provider-neutral preparation",
                        max_bytes=_MAX_ARTICULATION_PREPARATION_JSON_BYTES,
                    )
                )
            except (EmbeddedArticulationError, ValidationError) as exc:
                raise EmbeddedArticulationError(
                    "proposal attempt preparation is invalid"
                ) from exc
    else:
        _verify_binding(receipt.preparation, label="provider-neutral preparation")
        try:
            preparation = EmbeddedArticulationPreparation.model_validate_json(
                _read_bound_bytes(
                    receipt.preparation,
                    label="provider-neutral preparation",
                    max_bytes=_MAX_ARTICULATION_PREPARATION_JSON_BYTES,
                )
            )
        except (EmbeddedArticulationError, ValidationError) as exc:
            raise EmbeddedArticulationError(
                "proposal attempt preparation is invalid"
            ) from exc
    if (
        preparation != request.preparation_payload
        or canonical_json_digest(preparation) != receipt.preparation_digest
        or _evidence_digest(preparation) != receipt.evidence_digest
    ):
        raise EmbeddedArticulationError(
            "proposal attempt preparation or evidence identity changed"
        )
    for record in preparation.evidence_records:
        for artifact in record.artifacts:
            if artifact.path not in drifted_inputs:
                _verify_binding(
                    artifact,
                    label=f"Articulation evidence {record.evidence_id}",
                )
    for provider_input in receipt.provider_inputs:
        if provider_input.path not in drifted_inputs:
            _verify_binding(provider_input, label="proposal provider input")
    native_payload: DomainProposalPayload | None = None
    if receipt.native_payload is not None:
        try:
            native_payload = DomainProposalPayload.model_validate_json(
                _read_bound_bytes(
                    receipt.native_payload,
                    label="proposal attempt native payload",
                    max_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
                )
            )
        except (EmbeddedArticulationError, ValidationError) as exc:
            raise EmbeddedArticulationError(
                "proposal attempt native payload is invalid"
            ) from exc
    if receipt.disposition == "succeeded":
        if (
            native_payload is None
            or receipt.native_payload is None
            or receipt.proposal is None
            or receipt.bound_preparation is None
        ):  # pragma: no cover - model validator contract guard
            raise EmbeddedArticulationError(
                "successful proposal attempt omitted a required output"
            )
        try:
            proposal = EmbeddedArticulationProviderProposal.model_validate_json(
                _read_bound_bytes(
                    receipt.proposal,
                    label="proposal attempt proposal",
                    max_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
                )
            )
            bound = EmbeddedArticulationPreparation.model_validate_json(
                _read_bound_bytes(
                    receipt.bound_preparation,
                    label="proposal-bound Articulation preparation",
                    max_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
                )
            )
        except (EmbeddedArticulationError, ValidationError) as exc:
            raise EmbeddedArticulationError(
                "successful proposal attempt output is invalid"
            ) from exc
        expected_bound = bind_embedded_articulation_provider_proposal(
            preparation,
            proposal,
        )
        expected_payload = DomainProposalPayload(
            schema_version=_WRAPPED_PAYLOAD_SCHEMA_VERSION,
            values={
                "provider_request_digest": receipt.request_digest,
                "preparation_digest": receipt.preparation_digest,
                "evidence_digest": receipt.evidence_digest,
                "provider_capability": receipt.capability.model_dump(mode="json"),
                "native_payload": native_payload.model_dump(mode="json"),
                "native_payload_binding": receipt.native_payload.model_dump(
                    mode="json"
                ),
            },
        )
        if (
            bound != expected_bound
            or proposal.payload != expected_payload
            or proposal.producer != receipt.provider
            or proposal.capability != receipt.capability
            or proposal.preparation_digest != receipt.preparation_digest
            or proposal.evidence_digest != receipt.evidence_digest
            or proposal.provider_request != receipt.request
            or proposal.native_payload != receipt.native_payload
            or proposal.provider_request_digest != receipt.request_digest
        ):
            raise EmbeddedArticulationError(
                "successful proposal attempt output changed after publication"
            )
    if receipt.replacement is not None:
        replacement = receipt.replacement
        if len(replacement.prior_terminal_lineage) > _MAX_REPLACEMENT_LINEAGE_DEPTH:
            raise EmbeddedArticulationError(
                "proposal attempt replacement lineage exceeds its bounded depth"
            )
        try:
            lineage_cycles = any(
                Path(item.path) in seen or Path(item.path).resolve(strict=False) in seen
                for item in replacement.prior_terminal_lineage
            )
        except (OSError, RuntimeError) as exc:
            raise EmbeddedArticulationError(
                "proposal attempt replacement lineage path is invalid"
            ) from exc
        if lineage_cycles:
            raise EmbeddedArticulationError(
                "proposal attempt replacement receipts contain a cycle"
            )
        lineage_paths = {item.path for item in replacement.prior_terminal_lineage}
        known_drift = set(known_drifted_lineage or ()) | (
            lineage_paths & drifted_inputs
        )
        if replacement.prior_terminal_receipt.path in known_drift:
            for ancestor_index, ancestor in enumerate(
                replacement.prior_terminal_lineage[1:],
                start=1,
            ):
                if ancestor.path not in known_drift:
                    _verify_binding(
                        ancestor,
                        label="non-drifted proposal terminal ancestor",
                    )
                    validated_ancestor = _validate_terminal_receipt(
                        ancestor.path,
                        seen=set(seen),
                        known_drifted_lineage=known_drift,
                    )
                    expected_tail = (
                        ancestor,
                        *(
                            validated_ancestor.replacement.prior_terminal_lineage
                            if validated_ancestor.replacement is not None
                            else ()
                        ),
                    )
                    if (
                        replacement.prior_terminal_lineage[ancestor_index:]
                        != expected_tail
                    ):
                        raise EmbeddedArticulationError(
                            "proposal replacement lineage differs from its exact "
                            "non-drifted ancestor"
                        )
                    break
        else:
            _verify_binding(
                replacement.prior_terminal_receipt,
                label="superseded proposal terminal receipt",
            )
            prior = _validate_terminal_receipt(
                replacement.prior_terminal_receipt.path,
                seen=seen,
                known_drifted_lineage=known_drift,
            )
            if (
                prior.attempt_id != replacement.prior_attempt_id
                or prior.request_digest != replacement.prior_request_digest
                or prior.provider != replacement.prior_provider
                or prior.attempt_id == receipt.attempt_id
                or replacement.prior_terminal_lineage
                != (
                    replacement.prior_terminal_receipt,
                    *(
                        prior.replacement.prior_terminal_lineage
                        if prior.replacement is not None
                        else ()
                    ),
                )
            ):
                raise EmbeddedArticulationError(
                    "proposal replacement does not bind the exact prior terminal"
                )
    final_drifted_paths = set(known_drifted_lineage or ()) | drifted_inputs
    _verify_binding(receipt_binding, label="proposal attempt terminal receipt")
    for binding in output_bindings:
        if binding is not None:
            _verify_binding(binding, label="proposal attempt output")
    for binding in (
        receipt.request,
        receipt.preparation,
        *receipt.provider_inputs,
        *(
            artifact
            for record in preparation.evidence_records
            for artifact in record.artifacts
        ),
        *(
            receipt.replacement.prior_terminal_lineage
            if receipt.replacement is not None
            else ()
        ),
    ):
        if binding.path not in final_drifted_paths:
            _verify_binding(binding, label="proposal attempt final input recheck")
    if _attempt_entry_names(root) != expected_names:
        raise EmbeddedArticulationError(
            "proposal attempt root changed during final validation"
        )
    _validate_sealed_attempt_root(root, expected_names)
    return receipt


def validate_articulation_proposal_attempt_receipt(
    terminal_receipt_path: str | Path,
) -> ArticulationProposalAttemptTerminalReceipt:
    """Verify one terminal attempt and its complete replacement lineage."""

    return _validate_terminal_receipt(terminal_receipt_path, seen=set())


__all__ = [
    "ARTICULATION_PROPOSAL_ATTEMPT_PUBLICATION_SCHEMA_VERSION",
    "ARTICULATION_PROPOSAL_ATTEMPT_RECEIPT_SCHEMA_VERSION",
    "ARTICULATION_PROPOSAL_PROVIDER_PUBLICATION_SCHEMA_VERSION",
    "ARTICULATION_PROPOSAL_PROVIDER_REQUEST_SCHEMA_VERSION",
    "ARTICULATION_PROPOSAL_PROVIDER_REQUEST_V2_SCHEMA_VERSION",
    "ArtifactJsonArticulationProposalProvider",
    "ArticulationProposalAttemptFailed",
    "ArticulationProposalAttemptReplacement",
    "ArticulationProposalAttemptTerminalReceipt",
    "ArticulationProposalInvalidResponse",
    "ArticulationProposalProvider",
    "ArticulationProposalProviderAttemptPublication",
    "ArticulationProposalProviderFailure",
    "ArticulationProposalProviderPublication",
    "ArticulationProposalProviderRequest",
    "ArticulationProposalProviderRequestContract",
    "ArticulationProposalProviderRequestV2",
    "HttpJsonArticulationProposalProvider",
    "bind_embedded_articulation_provider_proposal",
    "request_embedded_articulation_provider_proposal",
    "validate_articulation_proposal_attempt_receipt",
]
