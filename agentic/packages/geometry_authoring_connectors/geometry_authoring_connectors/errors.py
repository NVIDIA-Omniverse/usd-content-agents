# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, secret-free failures for geometry authoring connectors."""

from __future__ import annotations

from typing import Any, ClassVar


class GeometryAuthoringConnectorError(RuntimeError):
    """Base connector failure suitable for a typed workflow terminal."""

    code: ClassVar[str] = "geometry_authoring_connector_error"
    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, provider_id: str | None = None) -> None:
        super().__init__(message)
        self.provider_id = provider_id

    def as_failure(self) -> dict[str, Any]:
        """Return a bounded public failure record without exception internals."""

        failure: dict[str, Any] = {
            "code": self.code,
            "summary": str(self),
            "retryable": self.retryable,
        }
        if self.provider_id is not None:
            failure["provider_id"] = self.provider_id
        return failure


class ConnectorConfigurationError(GeometryAuthoringConnectorError, ValueError):
    """Connector configuration violates a trust-boundary requirement."""

    code = "connector_configuration_invalid"


class ProviderUnavailableError(GeometryAuthoringConnectorError):
    """The explicitly selected provider cannot service the request."""

    code = "authoring_provider_unavailable"
    retryable = True


class UnsupportedCapabilityError(GeometryAuthoringConnectorError):
    """The selected connector intentionally does not implement an operation."""

    code = "authoring_capability_unsupported"


class ProviderTransportError(GeometryAuthoringConnectorError):
    """A bounded provider request failed without selecting a fallback."""

    code = "authoring_provider_transport_failure"
    retryable = True


class InvalidProviderResponseError(GeometryAuthoringConnectorError):
    """Provider bytes do not satisfy the selected connector contract."""

    code = "authoring_provider_response_invalid"


class ArtifactIntegrityError(GeometryAuthoringConnectorError):
    """Artifact bytes differ from their immutable digest or size binding."""

    code = "authoring_artifact_integrity_failure"


class ArtifactLimitError(GeometryAuthoringConnectorError):
    """A request or artifact exceeded an explicit connector bound."""

    code = "authoring_artifact_limit_exceeded"


class UnsafeArtifactError(GeometryAuthoringConnectorError):
    """An artifact path or file identity is unsafe to consume."""

    code = "authoring_artifact_unsafe"


class WorkerIsolationError(GeometryAuthoringConnectorError):
    """A Build123d worker does not declare a production isolation boundary."""

    code = "build123d_worker_isolation_required"


class ForgeCadExecutionUnavailableError(ProviderUnavailableError):
    """ForgeCAD execution is deliberately absent from the public adapter."""

    code = "forgecad_execution_unavailable"
