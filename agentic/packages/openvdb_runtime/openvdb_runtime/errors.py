# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Errors raised by the OpenVDB runtime facade."""

from __future__ import annotations

from collections.abc import Iterable


class OpenVDBRuntimeError(RuntimeError):
    """Base class for runtime, capability, and geometry failures."""


class RuntimeUnavailableError(OpenVDBRuntimeError):
    """Raised when the native OpenVDB module cannot be imported."""


class RuntimeVersionError(OpenVDBRuntimeError):
    """Raised when the loaded OpenVDB library does not match runtime policy."""


class NativeOperationError(OpenVDBRuntimeError):
    """Raised when a validated call fails inside the OpenVDB binding."""


class CapabilityUnavailableError(OpenVDBRuntimeError):
    """Raised when the loaded module lacks a required operation."""

    def __init__(self, capabilities: Iterable[str]) -> None:
        missing = tuple(sorted(set(capabilities)))
        self.capabilities = missing
        super().__init__(f"OpenVDB runtime is missing required capabilities: {', '.join(missing)}")


class InvalidGeometryError(OpenVDBRuntimeError, ValueError):
    """Raised when mesh or level-set input violates the facade contract."""


class ResourceLimitError(OpenVDBRuntimeError):
    """Raised when an operation would exceed a configured resource limit."""
