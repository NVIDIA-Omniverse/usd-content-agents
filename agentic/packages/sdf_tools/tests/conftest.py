# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import sdf_tools


@dataclass(frozen=True)
class SyntheticSamples:
    shape: tuple[int, ...]
    values: tuple[Any, ...]

    def __iter__(self):
        return iter(self.values)


def license_manifest(name: str = "fake") -> sdf_tools.BackendLicenseManifest:
    return sdf_tools.BackendLicenseManifest(
        schema="world-understanding.sdf-backend-license.v1",
        claim="test backend introduces no LGPL component",
        components=(
            sdf_tools.LicenseComponent(
                name,
                "1.0",
                "Apache-2.0",
                sdf_tools.DependencyScope.RUNTIME,
                True,
            ),
        ),
    )


@dataclass
class FakeBackend:
    descriptor: sdf_tools.BackendDescriptor
    available: bool = True

    def __post_init__(self) -> None:
        self.calls: list[tuple[sdf_tools.Operation, tuple[Any, ...], dict[str, Any]]] = []
        self._field_owner = object()
        self._files: dict[
            Path,
            tuple[tuple[sdf_tools.FieldKind, ...], dict[str, Any]],
        ] = {}

    @property
    def field_owner(self) -> object:
        return self._field_owner

    def inspect(self) -> sdf_tools.BackendInfo:
        if not self.available:
            raise sdf_tools.BackendUnavailableError("disabled for test")
        return sdf_tools.BackendInfo(
            backend_id=self.descriptor.backend_id,
            implementation_version=self.descriptor.implementation_version,
            operations=self.descriptor.operations,
            execution_mode="in_process",
            read_formats=self.descriptor.read_formats,
            write_formats=self.descriptor.write_formats,
            provenance={"test": True},
        )

    def execute(
        self,
        operation: sdf_tools.Operation,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.calls.append((operation, args, kwargs))
        if operation in {sdf_tools.Operation.MESH_TO_SDF, sdf_tools.Operation.MESH_TO_UDF}:
            kind = (
                sdf_tools.FieldKind.SIGNED_DISTANCE
                if operation is sdf_tools.Operation.MESH_TO_SDF
                else sdf_tools.FieldKind.UNSIGNED_DISTANCE
            )
            return sdf_tools.Field(
                self.descriptor.backend_id,
                kind,
                object(),
                self.field_owner,
            )
        if operation is sdf_tools.Operation.READ_FIELDS:
            kinds, metadata = self._files.get(
                Path(args[0]),
                (
                    (sdf_tools.FieldKind.SCALAR,),
                    {"format": kwargs["format"], "synthetic": True},
                ),
            )
            fields = tuple(
                sdf_tools.Field(
                    self.descriptor.backend_id,
                    kind,
                    object(),
                    self.field_owner,
                )
                for kind in kinds
            )
            if "field_name" in kwargs:
                return fields[0]
            return sdf_tools.FieldContents(
                fields=fields,
                metadata=dict(metadata),
            )
        if operation is sdf_tools.Operation.WRITE_FIELDS:
            path = Path(args[0])
            self._files[path] = (
                tuple(field.kind for field in args[1]),
                dict(kwargs["metadata"]),
            )
            return path
        if operation is sdf_tools.Operation.FIELD_TO_MESH:
            return sdf_tools.Mesh(
                vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                triangles=[(0, 1, 2)],
            )
        if operation is sdf_tools.Operation.SAMPLE_VALUES:
            return SyntheticSamples(
                (len(args[1]),),
                tuple(0.0 for _ in args[1]),
            )
        if operation is sdf_tools.Operation.SAMPLE_GRADIENTS:
            return SyntheticSamples(
                (len(args[1]), 3),
                tuple((0.0, 0.0, 0.0) for _ in args[1]),
            )
        if operation in {
            sdf_tools.Operation.UNION,
            sdf_tools.Operation.INTERSECTION,
            sdf_tools.Operation.DIFFERENCE,
            sdf_tools.Operation.NORMALIZE_SDF,
            sdf_tools.Operation.REBUILD_SDF,
            sdf_tools.Operation.TOPOLOGY_TO_SDF,
        }:
            return sdf_tools.Field(
                self.descriptor.backend_id,
                sdf_tools.FieldKind.SIGNED_DISTANCE,
                object(),
                self.field_owner,
            )
        if operation in {
            sdf_tools.Operation.OFFSET,
            sdf_tools.Operation.SMOOTH_SDF,
            sdf_tools.Operation.SMOOTH_SCALAR,
            sdf_tools.Operation.RESAMPLE_TO_MATCH,
        }:
            return sdf_tools.Field(
                self.descriptor.backend_id,
                args[0].kind,
                object(),
                self.field_owner,
            )
        if operation in {
            sdf_tools.Operation.ACTIVE_VALUE_MASK,
            sdf_tools.Operation.EXTRACT_ENCLOSED_REGION,
        }:
            return sdf_tools.Field(
                self.descriptor.backend_id,
                sdf_tools.FieldKind.MASK,
                object(),
                self.field_owner,
            )
        raise AssertionError(f"unexpected fake operation: {operation}")


def extension(
    backend_id: str,
    *,
    priority: int = 0,
    operations: frozenset[sdf_tools.Operation] | None = None,
    read_formats: frozenset[str] | None = None,
    write_formats: frozenset[str] | None = None,
    available: bool = True,
) -> tuple[sdf_tools.SdfBackendExtension, FakeBackend]:
    resolved_operations = frozenset(sdf_tools.Operation) if operations is None else operations
    descriptor = sdf_tools.BackendDescriptor(
        backend_id=backend_id,
        implementation_version="1.0",
        operations=resolved_operations,
        priority=priority,
        execution_mode="in_process",
        read_formats=(
            frozenset({"sdf-test"})
            if read_formats is None and sdf_tools.Operation.READ_FIELDS in resolved_operations
            else (read_formats if read_formats is not None else frozenset())
        ),
        write_formats=(
            frozenset({"sdf-test"})
            if write_formats is None and sdf_tools.Operation.WRITE_FIELDS in resolved_operations
            else (write_formats if write_formats is not None else frozenset())
        ),
        license_manifest=license_manifest(backend_id),
    )
    backend = FakeBackend(descriptor, available=available)
    return sdf_tools.SdfBackendExtension(descriptor, lambda: backend), backend


def _testing_registry(
    *extensions: sdf_tools.SdfBackendExtension,
) -> sdf_tools.SdfBackendRegistry:
    """Inject synthetic drivers from test code, outside the shipped API surface."""

    registry = sdf_tools.SdfBackendRegistry()
    identifiers: set[str] = set()
    for extension_value in extensions:
        if not isinstance(extension_value, sdf_tools.SdfBackendExtension):
            raise TypeError("extension must be an sdf_tools.SdfBackendExtension")
        sdf_tools.validate_license_manifest(extension_value.descriptor.license_manifest)
        backend_id = extension_value.descriptor.backend_id
        if backend_id in identifiers:
            raise ValueError(f"duplicate synthetic backend identifier: {backend_id}")
        identifiers.add(backend_id)
        registry._extensions[backend_id] = extension_value  # type: ignore[attr-defined]
    return registry


@pytest.fixture
def triangle_mesh() -> sdf_tools.Mesh:
    return sdf_tools.Mesh(
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        triangles=[(0, 1, 2)],
    )


@pytest.fixture
def sdf_backend_factory():
    return extension


@pytest.fixture
def sdf_license_manifest_factory():
    return license_manifest


@pytest.fixture
def sdf_testing_registry_factory():
    return _testing_registry


@pytest.fixture
def sdf_testing_toolkit_factory():
    def factory(*extensions: sdf_tools.SdfBackendExtension) -> sdf_tools.SdfToolkit:
        return sdf_tools.SdfToolkit(
            registry=_testing_registry(*extensions),
            discover_installed=False,
        )

    return factory
