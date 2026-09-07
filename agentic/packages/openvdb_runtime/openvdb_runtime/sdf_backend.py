# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenVDB 13 driver for the backend-neutral :mod:`sdf_tools` contract."""

from __future__ import annotations

from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from sdf_tools import (
    BackendDescriptor,
    BackendInfo,
    BackendLicenseManifest,
    BackendOperationError,
    BackendUnavailableError,
    CapabilityUnavailableError,
    DependencyScope,
    Field,
    FieldKind,
    InvalidGeometryError,
    LicenseComponent,
    Mesh,
    Operation,
    ResourceLimitError,
    SdfBackendExtension,
)

_OPERATIONS = frozenset(Operation) - {Operation.READ_FIELDS}
_FIELD_KIND_METADATA = "world_understanding:sdf_field_kind"
_CAPABILITY_OPERATIONS = {
    "vdb_io": {Operation.WRITE_FIELDS},
    "mesh_to_level_set": {Operation.MESH_TO_SDF},
    "mesh_to_unsigned_distance_field": {Operation.MESH_TO_UDF},
    "volume_to_mesh": {Operation.FIELD_TO_MESH},
    "csg": {Operation.UNION, Operation.INTERSECTION, Operation.DIFFERENCE},
    "level_set_offset": {Operation.OFFSET},
    "level_set_filter": {Operation.SMOOTH_SDF},
    "scalar_mean_filter": {Operation.SMOOTH_SCALAR},
    "level_set_normalize": {Operation.NORMALIZE_SDF},
    "level_set_rebuild": {Operation.REBUILD_SDF},
    "resample_to_match": {Operation.RESAMPLE_TO_MATCH},
    "sample_values": {Operation.SAMPLE_VALUES},
    "sample_gradients": {Operation.SAMPLE_GRADIENTS},
    "active_value_mask": {Operation.ACTIVE_VALUE_MASK},
    "topology_to_level_set": {Operation.TOPOLOGY_TO_SDF},
    "extract_enclosed_region": {Operation.EXTRACT_ENCLOSED_REGION},
}


class _LazyRuntimeFacade:
    """Defer the numerical/native facade until the backend is inspected or used."""

    def __getattr__(self, name: str) -> Any:
        return getattr(import_module("openvdb_runtime"), name)


vdb = _LazyRuntimeFacade()

_LICENSE_MANIFEST = BackendLicenseManifest(
    schema="world-understanding.sdf-backend-license.v1",
    claim=(
        "No LGPL component is introduced or bundled by the OpenVDB SDF driver; "
        "application-owned NumPy and glibc as an external Linux platform ABI dependency "
        "are recorded outside this delta-scoped claim."
    ),
    native_closure_attestation=(
        "sha256:8e0cb662ca1a9865a195d8fc6a7c64224fd8edeebfed9c266417a3afc7d3a59f"
    ),
    components=(
        LicenseComponent("sdf-tools", "0.6.0", "Apache-2.0", DependencyScope.RUNTIME, True),
        LicenseComponent("openvdb-runtime", "0.6.0", "Apache-2.0", DependencyScope.RUNTIME, True),
        LicenseComponent("OpenVDB", "13.0.0", "Apache-2.0", DependencyScope.BUNDLED, True),
        LicenseComponent(
            "OpenEXR Half",
            "openvdb-13.0.0-embedded",
            "BSD-3-Clause",
            DependencyScope.BUNDLED,
            True,
        ),
        LicenseComponent("nanobind", "2.13.0", "BSD-3-Clause", DependencyScope.BUNDLED, True),
        LicenseComponent("robin-map", "1.4.0", "MIT", DependencyScope.BUNDLED, True),
        LicenseComponent("oneTBB", "2022.2.0", "Apache-2.0", DependencyScope.BUNDLED, True),
        LicenseComponent("c-blosc", "1.21.6", "BSD-3-Clause", DependencyScope.BUNDLED, True),
        LicenseComponent("LZ4", "1.9.4", "BSD-2-Clause", DependencyScope.BUNDLED, True),
        LicenseComponent("Zstandard", "1.5.6", "BSD-3-Clause", DependencyScope.BUNDLED, True),
        LicenseComponent(
            "libdivsufsort-lite",
            "zstd-1.5.6-vendored",
            "MIT",
            DependencyScope.BUNDLED,
            True,
        ),
        LicenseComponent(
            "Bitshuffle",
            "c-blosc-1.21.6-adapted",
            "MIT",
            DependencyScope.BUNDLED,
            True,
        ),
        LicenseComponent(
            "FastLZ-derived",
            "c-blosc-1.21.6-derived",
            "MIT",
            DependencyScope.BUNDLED,
            True,
        ),
        LicenseComponent(
            "zlib-ng-derived",
            "c-blosc-1.21.6-derived",
            "Zlib",
            DependencyScope.BUNDLED,
            True,
        ),
        LicenseComponent("zlib", "1.3.1", "Zlib", DependencyScope.BUNDLED, True),
        LicenseComponent(
            "NumPy",
            "application-provided compatible runtime",
            "BSD-3-Clause",
            DependencyScope.PREEXISTING_APPLICATION,
            False,
        ),
        LicenseComponent(
            "glibc",
            "manylinux platform ABI",
            "LGPL-2.1-or-later",
            DependencyScope.PLATFORM_ABI,
            False,
        ),
        LicenseComponent(
            "GCC runtime",
            "manylinux platform ABI",
            "GPL-3.0-or-later WITH GCC-exception-3.1",
            DependencyScope.PLATFORM_ABI,
            False,
        ),
    ),
)

OPENVDB_SDF_BACKEND = BackendDescriptor(
    backend_id="openvdb",
    implementation_version="openvdb-runtime-13.0.0+wu.3",
    operations=_OPERATIONS,
    priority=100,
    execution_mode="in_process",
    read_formats=frozenset(),
    write_formats=frozenset({"vdb"}),
    license_manifest=_LICENSE_MANIFEST,
)


def _runtime_operations(capabilities: frozenset[object]) -> frozenset[Operation]:
    operations: set[Operation] = set()
    for capability, provided in _CAPABILITY_OPERATIONS.items():
        if any(getattr(value, "value", value) == capability for value in capabilities):
            operations.update(provided)
    return frozenset(operations)


def _limits(value: object) -> vdb.ExecutionLimits:
    from sdf_tools import ExecutionLimits

    if not isinstance(value, ExecutionLimits):
        raise TypeError("limits must be an sdf_tools.ExecutionLimits value")
    return vdb.ExecutionLimits(**asdict(value))


def _mesh(value: object, limits: vdb.ExecutionLimits) -> vdb.Mesh:
    if not isinstance(value, Mesh):
        raise TypeError("mesh must be an sdf_tools.Mesh")
    return vdb.Mesh._from_bounded_buffers(
        vertices=value.vertices,
        triangles=value.triangles,
        quads=value.quads,
        max_vertices=limits.max_vertices,
        max_faces=limits.max_faces,
    )


def _field(value: object, owner: object) -> Field:
    if not isinstance(value, Field):
        raise TypeError("field must be an sdf_tools.Field")
    value._payload_for("openvdb", owner)
    return value


def _wrap(payload: object, kind: FieldKind, owner: object) -> Field:
    return Field(backend_id="openvdb", kind=kind, _payload=payload, _owner=owner)


def _payload_memory_bytes(payload: object) -> int:
    evaluator = getattr(payload, "memUsage", None)
    if not callable(evaluator):
        raise BackendOperationError("OpenVDB field does not report its in-memory size")
    try:
        value = evaluator()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise BackendOperationError("OpenVDB field memory usage could not be inspected") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackendOperationError("OpenVDB field returned malformed memory usage")
    return value


def _preflight_persistable_payloads(
    fields: tuple[Field, ...], owner: object, limits: vdb.ExecutionLimits
) -> tuple[object, ...]:
    payloads: list[object] = []
    total_memory_bytes = 0
    for field in fields:
        payload = field._payload_for("openvdb", owner)
        memory_bytes = _payload_memory_bytes(payload)
        if memory_bytes > limits.max_field_memory_bytes:
            raise ResourceLimitError(
                f"VDB field requires {memory_bytes} bytes; limit is {limits.max_field_memory_bytes}"
            )
        total_memory_bytes += memory_bytes
        if total_memory_bytes > limits.max_total_field_memory_bytes:
            raise ResourceLimitError(
                f"VDB fields require {total_memory_bytes} aggregate bytes; "
                f"limit is {limits.max_total_field_memory_bytes}"
            )
        payloads.append(payload)
    return tuple(payloads)


def _persistable_payload(field: Field, payload: object) -> object:
    copier = getattr(payload, "deepCopy", None)
    if not callable(copier):
        raise BackendOperationError("OpenVDB field does not support an owned copy for VDB output")
    result = copier()
    try:
        cast(Any, result)[_FIELD_KIND_METADATA] = field.kind.value
    except (TypeError, ValueError, RuntimeError) as exc:
        raise BackendOperationError("OpenVDB field kind metadata could not be recorded") from exc
    return result


class OpenVdbSdfBackend:
    """Translate semantic SDF operations to the source-locked v13 facade."""

    def __init__(self) -> None:
        self._field_owner = object()

    @property
    def descriptor(self) -> BackendDescriptor:
        return OPENVDB_SDF_BACKEND

    @property
    def field_owner(self) -> object:
        return self._field_owner

    def _field(self, value: object) -> Field:
        return _field(value, self.field_owner)

    def _payload(self, value: object) -> object:
        return self._field(value)._payload_for("openvdb", self.field_owner)

    def _wrap(self, payload: object, kind: FieldKind) -> Field:
        return _wrap(payload, kind, self.field_owner)

    def inspect(self) -> BackendInfo:
        try:
            module = vdb.require_runtime(require_distribution_identity=True)
            runtime = vdb.inspect_runtime(module)
        except (vdb.RuntimeUnavailableError, vdb.RuntimeVersionError) as exc:
            raise BackendUnavailableError(str(exc)) from exc
        operations = _runtime_operations(runtime.capabilities)
        return BackendInfo(
            backend_id="openvdb",
            implementation_version=self.descriptor.implementation_version,
            operations=operations,
            execution_mode="in_process",
            read_formats=self.descriptor.read_formats,
            write_formats=(
                self.descriptor.write_formats
                if Operation.WRITE_FIELDS in operations
                else frozenset()
            ),
            provenance=runtime.as_dict(),
        )

    def execute(self, operation: Operation, /, *args: Any, **kwargs: Any) -> Any:
        operation = Operation(operation)
        limits = _limits(kwargs.pop("limits"))
        try:
            return self._execute(operation, args, kwargs, limits)
        except vdb.InvalidGeometryError as exc:
            raise InvalidGeometryError(str(exc)) from exc
        except vdb.ResourceLimitError as exc:
            raise ResourceLimitError(str(exc)) from exc
        except vdb.CapabilityUnavailableError as exc:
            raise CapabilityUnavailableError("openvdb", (operation.value,)) from exc
        except (vdb.RuntimeUnavailableError, vdb.RuntimeVersionError) as exc:
            raise BackendUnavailableError(str(exc)) from exc
        except vdb.OpenVDBRuntimeError as exc:
            raise BackendOperationError(f"SDF backend 'openvdb' failed: {exc}") from exc

    def _execute(
        self,
        operation: Operation,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        limits: vdb.ExecutionLimits,
    ) -> Any:
        if operation is Operation.MESH_TO_SDF:
            return self._wrap(
                vdb.mesh_to_level_set(_mesh(args[0], limits), limits=limits, **kwargs),
                FieldKind.SIGNED_DISTANCE,
            )
        if operation is Operation.MESH_TO_UDF:
            return self._wrap(
                vdb.mesh_to_unsigned_distance_field(
                    _mesh(args[0], limits), limits=limits, **kwargs
                ),
                FieldKind.UNSIGNED_DISTANCE,
            )
        if operation is Operation.FIELD_TO_MESH:
            repair_orientation = kwargs.pop("repair_orientation", True)
            result = vdb.volume_to_mesh(
                self._payload(args[0]),
                relax_disoriented_triangles=repair_orientation,
                limits=limits,
                **kwargs,
            )
            return Mesh(
                vertices=result.vertices,
                triangles=result.triangles,
                quads=result.quads,
            )
        if operation in {Operation.UNION, Operation.INTERSECTION, Operation.DIFFERENCE}:
            function = {
                Operation.UNION: vdb.union,
                Operation.INTERSECTION: vdb.intersection,
                Operation.DIFFERENCE: vdb.difference,
            }[operation]
            result = function(
                self._payload(args[0]),
                self._payload(args[1]),
                limits=limits,
                **kwargs,
            )
            return self._wrap(result, FieldKind.SIGNED_DISTANCE)
        if operation is Operation.OFFSET:
            source = self._field(args[0])
            return self._wrap(
                vdb.offset(
                    source._payload_for("openvdb", self.field_owner),
                    args[1],
                    limits=limits,
                    **kwargs,
                ),
                source.kind,
            )
        if operation in {Operation.SMOOTH_SDF, Operation.SMOOTH_SCALAR}:
            source = self._field(args[0])
            function = (
                vdb.mean_filter if operation is Operation.SMOOTH_SDF else vdb.scalar_mean_filter
            )
            return self._wrap(
                function(
                    source._payload_for("openvdb", self.field_owner),
                    limits=limits,
                    **kwargs,
                ),
                source.kind,
            )
        if operation is Operation.NORMALIZE_SDF:
            return self._wrap(
                vdb.normalize(self._payload(args[0]), limits=limits, **kwargs),
                FieldKind.SIGNED_DISTANCE,
            )
        if operation is Operation.REBUILD_SDF:
            return self._wrap(
                vdb.rebuild(self._payload(args[0]), limits=limits, **kwargs),
                FieldKind.SIGNED_DISTANCE,
            )
        if operation is Operation.RESAMPLE_TO_MATCH:
            source = self._field(args[0])
            reference = self._field(args[1])
            interpolation = kwargs.pop("interpolation")
            return self._wrap(
                vdb.resample_to_match(
                    source._payload_for("openvdb", self.field_owner),
                    reference._payload_for("openvdb", self.field_owner),
                    interpolation=interpolation.value,
                    limits=limits,
                    **kwargs,
                ),
                source.kind,
            )
        if operation in {Operation.SAMPLE_VALUES, Operation.SAMPLE_GRADIENTS}:
            function = (
                vdb.sample_values if operation is Operation.SAMPLE_VALUES else vdb.sample_gradients
            )
            interpolation = kwargs.pop("interpolation")
            try:
                point_count = len(args[1])
            except TypeError:
                point_count = None
            if point_count is not None and point_count > limits.max_sample_points:
                raise ResourceLimitError(
                    f"sample batch has {point_count} points; limit is {limits.max_sample_points}"
                )
            return function(
                self._payload(args[0]),
                args[1],
                interpolation=interpolation.value,
                limits=limits,
                **kwargs,
            )
        if operation is Operation.ACTIVE_VALUE_MASK:
            return self._wrap(
                vdb.active_value_mask(self._payload(args[0]), limits=limits, **kwargs),
                FieldKind.MASK,
            )
        if operation is Operation.TOPOLOGY_TO_SDF:
            return self._wrap(
                vdb.topology_to_level_set(self._payload(args[0]), limits=limits, **kwargs),
                FieldKind.SIGNED_DISTANCE,
            )
        if operation is Operation.EXTRACT_ENCLOSED_REGION:
            return self._wrap(
                vdb.extract_enclosed_region(self._payload(args[0]), limits=limits, **kwargs),
                FieldKind.MASK,
            )
        if operation is Operation.WRITE_FIELDS:
            format_name = kwargs.pop("format")
            if format_name != "vdb":
                raise ValueError("OpenVDB backend supports only format='vdb'")
            input_fields = args[1]
            field_count = len(input_fields)
            if field_count > limits.max_fields:
                raise ResourceLimitError(
                    f"VDB output has {field_count} fields; limit is {limits.max_fields}"
                )
            admitted_fields = tuple(self._field(field) for field in input_fields)
            payloads = _preflight_persistable_payloads(admitted_fields, self.field_owner, limits)
            fields = tuple(
                _persistable_payload(field, payload)
                for field, payload in zip(admitted_fields, payloads, strict=True)
            )
            return Path(vdb.write(args[0], fields, limits=limits, **kwargs))
        raise CapabilityUnavailableError("openvdb", (operation.value,))


def sdf_backend_extension() -> SdfBackendExtension:
    """Return the lazy extension registered through package metadata."""

    return SdfBackendExtension(descriptor=OPENVDB_SDF_BACKEND, factory=OpenVdbSdfBackend)


__all__ = [
    "OPENVDB_SDF_BACKEND",
    "OpenVdbSdfBackend",
    "sdf_backend_extension",
]
