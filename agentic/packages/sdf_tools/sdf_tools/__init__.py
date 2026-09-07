# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral signed-distance-field tools for content agents."""

from .backend import BackendDescriptor, SdfBackend, SdfBackendExtension
from .errors import (
    BackendMismatchError,
    BackendNotRegisteredError,
    BackendOperationError,
    BackendUnavailableError,
    CapabilityUnavailableError,
    InvalidGeometryError,
    LicensePolicyError,
    ResourceLimitError,
    SdfToolsError,
)
from .licensing import (
    BackendLicenseManifest,
    DependencyScope,
    LicenseComponent,
    validate_license_manifest,
)
from .registry import BackendRejection, SdfBackendRegistry
from .session import SdfSession, SdfToolkit, create_session
from .types import (
    DEFAULT_LIMITS,
    BackendInfo,
    ExecutionLimits,
    Field,
    FieldContents,
    FieldKind,
    Interpolation,
    Mesh,
    Operation,
)

__all__ = [
    "DEFAULT_LIMITS",
    "BackendDescriptor",
    "BackendInfo",
    "BackendLicenseManifest",
    "BackendMismatchError",
    "BackendNotRegisteredError",
    "BackendOperationError",
    "BackendRejection",
    "BackendUnavailableError",
    "CapabilityUnavailableError",
    "DependencyScope",
    "ExecutionLimits",
    "Field",
    "FieldContents",
    "FieldKind",
    "Interpolation",
    "InvalidGeometryError",
    "LicenseComponent",
    "LicensePolicyError",
    "Mesh",
    "Operation",
    "ResourceLimitError",
    "SdfBackend",
    "SdfBackendExtension",
    "SdfBackendRegistry",
    "SdfSession",
    "SdfToolkit",
    "SdfToolsError",
    "create_session",
    "validate_license_manifest",
]
