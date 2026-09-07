# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral failures raised by SDF tools."""

from __future__ import annotations

from collections.abc import Iterable


class SdfToolsError(RuntimeError):
    """Base class for SDF backend, policy, and operation failures."""


class BackendNotRegisteredError(SdfToolsError):
    """Raised when a caller selects a backend outside the reviewed registry."""


class BackendUnavailableError(SdfToolsError):
    """Raised when no admitted backend can satisfy an operation request."""


class CapabilityUnavailableError(SdfToolsError):
    """Raised when a selected backend lacks required semantic capabilities."""

    def __init__(
        self,
        backend_id: str,
        operations: Iterable[str] = (),
        *,
        formats: Iterable[str] = (),
        read_formats: Iterable[str] = (),
        write_formats: Iterable[str] = (),
    ) -> None:
        missing = tuple(sorted(set(operations)))
        generic_formats = tuple(sorted(set(formats)))
        missing_read_formats = tuple(sorted(set(read_formats)))
        missing_write_formats = tuple(sorted(set(write_formats)))
        self.backend_id = backend_id
        self.operations = missing
        self.read_formats = missing_read_formats
        self.write_formats = missing_write_formats
        self.formats = tuple(
            sorted(set(generic_formats) | set(missing_read_formats) | set(missing_write_formats))
        )
        details = []
        if missing:
            details.append(f"required operations: {', '.join(missing)}")
        if generic_formats:
            details.append(f"required formats: {', '.join(generic_formats)}")
        if missing_read_formats:
            details.append(f"required read formats: {', '.join(missing_read_formats)}")
        if missing_write_formats:
            details.append(f"required write formats: {', '.join(missing_write_formats)}")
        if not details:
            details.append("an unspecified required capability")
        super().__init__(f"SDF backend {backend_id!r} is missing " + "; ".join(details))


class BackendMismatchError(SdfToolsError):
    """Raised when one operation receives fields from different backends."""


class BackendOperationError(SdfToolsError):
    """Raised when a validated operation fails inside a backend driver."""


class InvalidGeometryError(SdfToolsError, ValueError):
    """Raised when mesh or field input violates the SDF contract."""


class ResourceLimitError(SdfToolsError):
    """Raised before or after an operation exceeds a declared resource bound."""


class LicensePolicyError(SdfToolsError):
    """Raised when a backend dependency attestation violates admission policy."""
