# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend extension contracts for SDF tools."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .licensing import BackendLicenseManifest
from .types import BackendInfo, Operation

_BACKEND_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    """Static backend metadata safe to inspect before loading native code."""

    backend_id: str
    implementation_version: str
    operations: frozenset[Operation]
    priority: int
    execution_mode: str
    read_formats: frozenset[str]
    write_formats: frozenset[str]
    license_manifest: BackendLicenseManifest

    def __post_init__(self) -> None:
        if not _BACKEND_ID.fullmatch(self.backend_id):
            raise ValueError("backend_id must be a lowercase hyphenated identifier")
        if not self.implementation_version:
            raise ValueError("implementation_version must be nonempty")
        if not self.operations:
            raise ValueError("backend must declare at least one operation")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("backend priority must be an integer")
        if self.execution_mode != "in_process":
            raise ValueError("SDF backend drivers must execute in_process")
        for name in ("read_formats", "write_formats"):
            formats = getattr(self, name)
            if not isinstance(formats, frozenset):
                raise TypeError(f"{name} must be a frozenset")
            if any(not isinstance(value, str) or not value for value in formats):
                raise ValueError(f"{name} must contain nonempty string identifiers")
        if self.read_formats and Operation.READ_FIELDS not in self.operations:
            raise ValueError("read_formats require the read_fields operation")
        if self.write_formats and Operation.WRITE_FIELDS not in self.operations:
            raise ValueError("write_formats require the write_fields operation")

    @property
    def supported_formats(self) -> frozenset[str]:
        """Compatibility view of every readable or writable artifact format."""

        return self.read_formats | self.write_formats


class SdfBackend(Protocol):
    """Runtime interface implemented by admitted backend drivers."""

    @property
    def descriptor(self) -> BackendDescriptor: ...

    @property
    def field_owner(self) -> object: ...

    def inspect(self) -> BackendInfo: ...

    def execute(self, operation: Operation, /, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class SdfBackendExtension:
    """Lazy backend factory exported by an installed driver distribution."""

    descriptor: BackendDescriptor
    factory: Callable[[], SdfBackend]

    def create(self) -> SdfBackend:
        backend = self.factory()
        if backend.descriptor != self.descriptor:
            raise ValueError("backend factory descriptor differs from registered descriptor")
        return backend
