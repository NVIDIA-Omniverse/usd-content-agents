# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated-worker boundary shared by external geometry authoring providers.

The runner never imports a provider runtime and never evaluates returned native
source. A deployment supplies a backend whose isolation is independently
enforced, then exposes ``handle`` through its authenticated HTTP framework.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ._artifacts import read_worker_output
from .build123d import BUILD123D_PROVIDER_ID
from .delegated import (
    FORGECAD_AUTHORING_PROVIDER_ID,
    validate_delegated_outputs,
)
from .errors import (
    GeometryAuthoringConnectorError,
    InvalidProviderResponseError,
    ProviderUnavailableError,
    WorkerIsolationError,
)
from .models import (
    MAX_SOURCE_BUNDLE_ARTIFACTS,
    SAFE_FILENAME_PATTERN,
    ArtifactRole,
    AuthoringRequest,
    WireArtifact,
    WirePart,
    WireSemanticParameter,
    WireSourceBundle,
    WireVerificationAssertion,
    _validate_json_value,
)

WorkerIsolationKind = Literal["container", "sandboxed-process", "trusted-test-fixture"]


class WorkerArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    path: Path
    filename: str = Field(pattern=SAFE_FILENAME_PATTERN)
    role: ArtifactRole
    media_type: str = Field(min_length=1, max_length=128)


class ExternalAuthoringWorkerResult(BaseModel):
    """Typed provider result before runner-owned artifact serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    provider_version: str = Field(min_length=1, max_length=128)
    source_revision: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    units: Literal["millimeter", "centimeter", "meter", "inch"]
    up_axis: Literal["X", "Y", "Z"]
    forward_axis: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"] = "+Y"
    handedness: Literal["left", "right"] = "right"
    upstream_edit_uri: str | None = Field(default=None, max_length=4096)
    artifacts: tuple[WorkerArtifact, ...] = Field(
        min_length=1,
        max_length=MAX_SOURCE_BUNDLE_ARTIFACTS,
    )
    parts: tuple[WirePart, ...] = Field(default=(), max_length=4096)
    parameters: tuple[WireSemanticParameter, ...] = Field(default=(), max_length=1024)
    verification_assertions: tuple[WireVerificationAssertion, ...] = Field(
        default=(),
        max_length=4096,
    )
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=256)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_json_value(value)
        return value


class ExternalAuthoringExecutionBackend(Protocol):
    """Deployment-owned executor with an externally enforced isolation mode."""

    @property
    def isolation_kind(self) -> WorkerIsolationKind: ...

    def execute(
        self,
        request: AuthoringRequest,
        *,
        workspace: Path,
    ) -> ExternalAuthoringWorkerResult: ...


class ExternalAuthoringWorkerRunner:
    """Validate one backend invocation and serialize only bounded artifacts."""

    def __init__(
        self,
        backend: ExternalAuthoringExecutionBackend,
        *,
        provider_id: str,
        provider_label: str,
        returns_native_source: bool,
        workspace_parent: str | Path | None = None,
        allow_trusted_test_fixture: bool = False,
    ) -> None:
        isolation_kind = backend.isolation_kind
        if isolation_kind not in {"container", "sandboxed-process"} and not (
            isolation_kind == "trusted-test-fixture" and allow_trusted_test_fixture
        ):
            raise WorkerIsolationError(
                f"{provider_label} backends require container or sandboxed-process isolation",
                provider_id=provider_id,
            )
        self._backend = backend
        self._provider_id = provider_id
        self._provider_label = provider_label
        self._returns_native_source = returns_native_source
        self._workspace_parent = (
            Path(workspace_parent).expanduser() if workspace_parent is not None else None
        )
        self._isolation_kind = isolation_kind

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Worker HTTP handler core; authentication and routing remain external."""

        try:
            request = AuthoringRequest.model_validate(payload)
        except ValidationError as exc:
            raise InvalidProviderResponseError(
                f"{self._provider_label} received an invalid bounded authoring request",
                provider_id=self._provider_id,
            ) from exc
        try:
            with tempfile.TemporaryDirectory(
                prefix="geometry-authoring-worker-",
                dir=self._workspace_parent,
            ) as temporary:
                workspace = Path(temporary)
                workspace.chmod(0o700)
                result = self._backend.execute(request, workspace=workspace)
                if not isinstance(result, ExternalAuthoringWorkerResult):
                    result = ExternalAuthoringWorkerResult.model_validate(result)
                wire_artifacts: list[WireArtifact] = []
                for artifact in result.artifacts:
                    content = read_worker_output(workspace, artifact.path)
                    wire_artifacts.append(
                        WireArtifact.from_bytes(
                            filename=artifact.filename,
                            role=artifact.role,
                            media_type=artifact.media_type,
                            content=content,
                        )
                    )
                metadata = dict(result.metadata)
                metadata["worker_isolation"] = self._isolation_kind
                bundle = WireSourceBundle(
                    provider_id=self._provider_id,
                    provider_version=result.provider_version,
                    source_revision=result.source_revision,
                    units=result.units,
                    up_axis=result.up_axis,
                    forward_axis=result.forward_axis,
                    handedness=result.handedness,
                    upstream_edit_uri=result.upstream_edit_uri,
                    artifacts=tuple(wire_artifacts),
                    parts=result.parts,
                    parameters=result.parameters,
                    verification_assertions=result.verification_assertions,
                    metadata=metadata,
                )
                validate_delegated_outputs(
                    request,
                    bundle,
                    provider_label=self._provider_label,
                    returns_native_source=self._returns_native_source,
                )
                return bundle.model_dump(mode="json")
        except GeometryAuthoringConnectorError:
            raise
        except ValidationError as exc:
            raise InvalidProviderResponseError(
                f"isolated {self._provider_label} backend returned an invalid typed result",
                provider_id=self._provider_id,
            ) from exc
        except Exception as exc:
            raise ProviderUnavailableError(
                f"isolated {self._provider_label} backend failed without fallback",
                provider_id=self._provider_id,
            ) from exc


class Build123dWorkerRunner(ExternalAuthoringWorkerRunner):
    """Build123d-compatible specialization of the common external runner."""

    def __init__(
        self,
        backend: ExternalAuthoringExecutionBackend,
        *,
        provider_id: str = BUILD123D_PROVIDER_ID,
        workspace_parent: str | Path | None = None,
        allow_trusted_test_fixture: bool = False,
    ) -> None:
        super().__init__(
            backend,
            provider_id=provider_id,
            provider_label="Build123d worker",
            returns_native_source=True,
            workspace_parent=workspace_parent,
            allow_trusted_test_fixture=allow_trusted_test_fixture,
        )


class ForgeCadWorkerRunner(ExternalAuthoringWorkerRunner):
    """Licensed ForgeCAD specialization with explicit automated-use admission."""

    def __init__(
        self,
        backend: ExternalAuthoringExecutionBackend,
        *,
        automated_use_authorized: bool,
        workspace_parent: str | Path | None = None,
    ) -> None:
        if not automated_use_authorized:
            raise WorkerIsolationError(
                "ForgeCAD worker requires explicit automated-use authorization",
                provider_id=FORGECAD_AUTHORING_PROVIDER_ID,
            )
        super().__init__(
            backend,
            provider_id=FORGECAD_AUTHORING_PROVIDER_ID,
            provider_label="ForgeCAD authoring worker",
            returns_native_source=True,
            workspace_parent=workspace_parent,
        )


# Compatibility names retained for existing Build123d worker deployments.
Build123dWorkerResult = ExternalAuthoringWorkerResult
Build123dExecutionBackend = ExternalAuthoringExecutionBackend


__all__ = [
    "Build123dExecutionBackend",
    "Build123dWorkerResult",
    "Build123dWorkerRunner",
    "ExternalAuthoringExecutionBackend",
    "ExternalAuthoringWorkerResult",
    "ExternalAuthoringWorkerRunner",
    "ForgeCadWorkerRunner",
    "WorkerArtifact",
    "WorkerIsolationKind",
]
