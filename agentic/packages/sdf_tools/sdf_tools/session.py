# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-bound SDF tool sessions and the default toolkit."""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from threading import RLock
from typing import Any

from .backend import SdfBackend
from .errors import (
    BackendMismatchError,
    BackendOperationError,
    BackendUnavailableError,
    CapabilityUnavailableError,
    InvalidGeometryError,
    ResourceLimitError,
)
from .registry import (
    BackendRejection,
    SdfBackendRegistry,
    discover_installed_backends,
)
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

_MAX_METADATA_DEPTH = 16

_DRIVER_OPERATION_ERRORS = (
    BackendMismatchError,
    BackendOperationError,
    BackendUnavailableError,
    CapabilityUnavailableError,
    InvalidGeometryError,
    ResourceLimitError,
)


def _exact_values(value: object, *, count: int, label: str) -> tuple[object, ...]:
    try:
        iterator = iter(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{label} is not iterable") from exc
    result: list[object] = []
    for _ in range(count):
        try:
            result.append(next(iterator))
        except StopIteration as exc:
            raise ValueError(f"{label} has fewer than {count} values") from exc
    try:
        next(iterator)
    except StopIteration:
        return tuple(result)
    raise ValueError(f"{label} has more than {count} values")


def _exact_row(value: object, *, width: int, label: str, index: int) -> tuple[object, ...]:
    return _exact_values(value, count=width, label=f"{label} row {index}")


def _matrix_rows(
    value: object,
    *,
    count: int,
    width: int,
    label: str,
) -> Iterable[tuple[object, ...]]:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is not None:
        shape = _exact_values(raw_shape, count=2, label=f"{label} shape")
        if any(
            isinstance(part, bool) or not isinstance(part, numbers.Integral) or part < 0
            for part in shape
        ):
            raise TypeError(f"{label} shape must contain nonnegative integers")
        if tuple(int(part) for part in shape) != (count, width):
            raise ValueError(f"{label} must have shape ({count}, {width})")
    try:
        iterator = iter(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{label} must be an iterable row buffer") from exc
    for index in range(count):
        try:
            row = next(iterator)
        except StopIteration as exc:
            raise ValueError(f"{label} contains fewer rows than reported") from exc
        yield _exact_row(row, width=width, label=label, index=index)
    try:
        next(iterator)
    except StopIteration:
        return
    raise ValueError(f"{label} contains more rows than reported")


def _finite_real(value: object, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{label} must be a real number")
    try:
        finite = math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be a finite real number") from exc
    if not finite:
        raise ValueError(f"{label} must be a finite real number")


def _validate_mesh_buffers(
    mesh: Mesh,
    *,
    vertex_count: int,
    triangle_count: int,
    quad_count: int,
) -> None:
    for row_index, vertex in enumerate(
        _matrix_rows(mesh.vertices, count=vertex_count, width=3, label="vertices")
    ):
        for component in vertex:
            _finite_real(component, label=f"vertex row {row_index} component")

    for values, count, width, label in (
        (mesh.triangles, triangle_count, 3, "triangles"),
        (mesh.quads, quad_count, 4, "quads"),
    ):
        for row_index, face in enumerate(
            _matrix_rows(values, count=count, width=width, label=label)
        ):
            indices = []
            for component in face:
                if isinstance(component, bool) or not isinstance(component, numbers.Integral):
                    raise TypeError(f"{label} row {row_index} must contain integer indices")
                index = int(component)
                if index < 0 or index >= vertex_count:
                    raise ValueError(
                        f"{label} row {row_index} references a vertex outside the vertex buffer"
                    )
                indices.append(index)
            if len(set(indices)) != width:
                raise ValueError(f"{label} row {row_index} repeats a vertex")


def _bounded_metadata(
    metadata: Mapping[str, Any] | None, limits: ExecutionLimits
) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")

    remaining_entries = limits.max_metadata_entries
    byte_limit = min(limits.max_metadata_bytes, limits.max_file_bytes)
    remaining_bytes = byte_limit
    active_containers: set[int] = set()

    def consume_entry() -> None:
        nonlocal remaining_entries
        if remaining_entries == 0:
            raise ResourceLimitError(
                f"metadata exceeds the configured entry limit ({limits.max_metadata_entries})"
            )
        remaining_entries -= 1

    def consume_bytes(count: int) -> None:
        nonlocal remaining_bytes
        if count > remaining_bytes:
            raise ResourceLimitError(f"metadata exceeds the configured byte limit ({byte_limit})")
        remaining_bytes -= count

    def consume_scalar(value: Any) -> None:
        if isinstance(value, str) and len(value) > byte_limit:
            raise ResourceLimitError(f"metadata exceeds the configured byte limit ({byte_limit})")
        consume_bytes(
            len(
                json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":")).encode(
                    "ascii"
                )
            )
        )

    def copy_value(value: Any, depth: int) -> Any:
        if depth > _MAX_METADATA_DEPTH:
            raise ResourceLimitError(
                f"metadata nesting exceeds the supported depth ({_MAX_METADATA_DEPTH})"
            )
        if value is None or isinstance(value, bool | str):
            consume_scalar(value)
            return value
        if isinstance(value, numbers.Integral):
            integer = int(value)
            if integer.bit_length() > 4_096:
                raise ResourceLimitError("metadata integer exceeds the supported size")
            consume_scalar(integer)
            return integer
        if isinstance(value, numbers.Real):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("metadata numbers must be finite")
            consume_scalar(result)
            return result
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active_containers:
                raise ValueError("metadata must not contain cycles")
            active_containers.add(identity)
            try:
                result: dict[str, Any] = {}
                consume_bytes(1)
                first = True
                for key in value:
                    consume_entry()
                    if not isinstance(key, str):
                        raise TypeError("metadata keys must be strings")
                    if not first:
                        consume_bytes(1)
                    first = False
                    consume_scalar(key)
                    consume_bytes(1)
                    result[key] = copy_value(value[key], depth + 1)
                consume_bytes(1)
                return result
            finally:
                active_containers.remove(identity)
        if type(value) in (list, tuple):
            identity = id(value)
            if identity in active_containers:
                raise ValueError("metadata must not contain cycles")
            if len(value) > remaining_entries:
                raise ResourceLimitError(
                    f"metadata exceeds the configured entry limit ({limits.max_metadata_entries})"
                )
            active_containers.add(identity)
            try:
                result = []
                consume_bytes(1)
                first = True
                for item in value:
                    consume_entry()
                    if not first:
                        consume_bytes(1)
                    first = False
                    result.append(copy_value(item, depth + 1))
                consume_bytes(1)
                return result
            finally:
                active_containers.remove(identity)
        raise TypeError("metadata values must be finite JSON-compatible data")

    canonical = copy_value(metadata, 0)
    assert isinstance(canonical, dict)
    encoded = json.dumps(
        canonical,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    if len(encoded) > byte_limit:
        raise ResourceLimitError(f"metadata is {len(encoded)} bytes; limit is {byte_limit}")
    return canonical


class SdfToolkit:
    """Own backend discovery, policy admission, and deterministic selection."""

    def __init__(
        self,
        *,
        registry: SdfBackendRegistry | None = None,
        discover_installed: bool = True,
    ) -> None:
        if registry is not None and type(registry) is not SdfBackendRegistry:
            raise TypeError("registry must be an exact sdf_tools.SdfBackendRegistry instance")
        self.registry = registry if registry is not None else SdfBackendRegistry()
        self._discover_installed = discover_installed
        self._discovery_complete = not discover_installed
        self._attempted_backends: set[str] = set()
        self._discovery_lock = RLock()

    def load_installed_backends(self, backend: str | None = None) -> tuple[str, ...]:
        """Load qualified automatic drivers or one explicitly selected driver."""

        with self._discovery_lock:
            if (backend is None and self._discovery_complete) or (
                backend is not None and backend in self._attempted_backends
            ):
                return tuple(self.registry.snapshot())
            loaded = discover_installed_backends(self.registry, backend=backend)
            if backend is None:
                self._discovery_complete = True
            else:
                self._attempted_backends.add(backend)
            return loaded

    def create_session(
        self,
        *,
        backend: str = "auto",
        require: Iterable[Operation] = (),
        require_formats: Iterable[str] | Mapping[Operation | str, Iterable[str]] = (),
        limits: ExecutionLimits = DEFAULT_LIMITS,
    ) -> SdfSession:
        """Select and bind one backend for a sequence of SDF operations."""

        if not isinstance(backend, str) or not backend:
            raise TypeError("backend must be a nonempty registered identifier or 'auto'")
        if not isinstance(limits, ExecutionLimits):
            raise TypeError("limits must be an sdf_tools.ExecutionLimits value")
        if self._discover_installed:
            self.load_installed_backends(None if backend == "auto" else backend)
        driver, info, rejections = self.registry.resolve(
            backend=backend,
            require=require,
            require_formats=require_formats,
        )
        return SdfSession(driver=driver, info=info, limits=limits, rejections=rejections)


class SdfSession:
    """One immutable backend selection with shared execution limits."""

    def __init__(
        self,
        *,
        driver: SdfBackend,
        info: BackendInfo,
        limits: ExecutionLimits,
        rejections: tuple[BackendRejection, ...],
    ) -> None:
        self._driver = driver
        self._field_owner = driver.field_owner
        self._info = info
        self._limits = limits
        self._rejections = rejections

    @property
    def backend_id(self) -> str:
        return self._info.backend_id

    @property
    def backend_info(self) -> BackendInfo:
        return self._info

    @property
    def execution_limits(self) -> ExecutionLimits:
        return self._limits

    @property
    def selection_rejections(self) -> tuple[BackendRejection, ...]:
        return self._rejections

    def _field(self, value: Field) -> Field:
        if not isinstance(value, Field):
            raise TypeError("field must be an sdf_tools.Field")
        if value.backend_id != self.backend_id:
            raise BackendMismatchError(
                f"field belongs to backend {value.backend_id!r}; session uses {self.backend_id!r}"
            )
        if not value._is_owned_by(self._field_owner):
            raise BackendMismatchError(
                f"field does not belong to the selected {self.backend_id!r} implementation"
            )
        return value

    def _result_field(
        self,
        operation: Operation,
        value: object,
        *,
        expected_kind: FieldKind | None = None,
    ) -> Field:
        if not isinstance(value, Field):
            raise BackendOperationError(
                f"SDF backend {self.backend_id!r} returned a non-field result for {operation.value}"
            )
        if value.backend_id != self.backend_id:
            raise BackendOperationError(
                f"SDF backend {self.backend_id!r} returned a field owned by "
                f"{value.backend_id!r} for {operation.value}"
            )
        if not value._is_owned_by(self._field_owner):
            raise BackendOperationError(
                f"SDF backend {self.backend_id!r} returned a field owned by another "
                f"implementation for {operation.value}"
            )
        if not isinstance(value.kind, FieldKind):
            raise BackendOperationError(
                f"SDF backend {self.backend_id!r} returned an invalid field kind for "
                f"{operation.value}"
            )
        if expected_kind is not None and value.kind is not expected_kind:
            raise BackendOperationError(
                f"SDF backend {self.backend_id!r} returned {value.kind.value!r} for "
                f"{operation.value}; expected {expected_kind.value!r}"
            )
        return value

    def _validate_result(
        self,
        operation: Operation,
        value: object,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        fixed_field_kinds = {
            Operation.MESH_TO_SDF: FieldKind.SIGNED_DISTANCE,
            Operation.MESH_TO_UDF: FieldKind.UNSIGNED_DISTANCE,
            Operation.UNION: FieldKind.SIGNED_DISTANCE,
            Operation.INTERSECTION: FieldKind.SIGNED_DISTANCE,
            Operation.DIFFERENCE: FieldKind.SIGNED_DISTANCE,
            Operation.NORMALIZE_SDF: FieldKind.SIGNED_DISTANCE,
            Operation.REBUILD_SDF: FieldKind.SIGNED_DISTANCE,
            Operation.ACTIVE_VALUE_MASK: FieldKind.MASK,
            Operation.TOPOLOGY_TO_SDF: FieldKind.SIGNED_DISTANCE,
            Operation.EXTRACT_ENCLOSED_REGION: FieldKind.MASK,
        }
        if operation in fixed_field_kinds:
            return self._result_field(
                operation,
                value,
                expected_kind=fixed_field_kinds[operation],
            )
        if operation in {
            Operation.OFFSET,
            Operation.SMOOTH_SDF,
            Operation.SMOOTH_SCALAR,
            Operation.RESAMPLE_TO_MATCH,
        }:
            source = args[0]
            if not isinstance(source, Field):
                raise BackendOperationError("internal SDF field dispatch contract was violated")
            return self._result_field(operation, value, expected_kind=source.kind)
        if operation is Operation.FIELD_TO_MESH:
            if not isinstance(value, Mesh):
                raise BackendOperationError(
                    f"SDF backend {self.backend_id!r} returned a non-mesh result for "
                    f"{operation.value}"
                )
            try:
                vertex_count = len(value.vertices)
                triangle_count = len(value.triangles)
                quad_count = len(value.quads)
            except Exception as exc:
                raise BackendOperationError(
                    f"SDF backend {self.backend_id!r} returned a malformed mesh for "
                    f"{operation.value}"
                ) from exc
            face_count = triangle_count + quad_count
            if vertex_count > self._limits.max_vertices:
                raise ResourceLimitError(
                    f"backend mesh has {vertex_count} vertices; "
                    f"limit is {self._limits.max_vertices}"
                )
            if face_count > self._limits.max_faces:
                raise ResourceLimitError(
                    f"backend mesh has {face_count} faces; limit is {self._limits.max_faces}"
                )
            if vertex_count == 0 or face_count == 0:
                raise BackendOperationError(
                    f"SDF backend {self.backend_id!r} returned a malformed mesh for "
                    f"{operation.value}"
                )
            try:
                _validate_mesh_buffers(
                    value,
                    vertex_count=vertex_count,
                    triangle_count=triangle_count,
                    quad_count=quad_count,
                )
            except Exception as exc:
                raise BackendOperationError(
                    f"SDF backend {self.backend_id!r} returned a malformed mesh for "
                    f"{operation.value}"
                ) from exc
            return value
        if operation is Operation.READ_FIELDS:
            if "field_name" in kwargs:
                return self._result_field(operation, value)
            if not isinstance(value, FieldContents):
                raise BackendOperationError(
                    f"SDF backend {self.backend_id!r} returned invalid field contents"
                )
            if not isinstance(value.fields, tuple) or not isinstance(value.metadata, dict):
                raise BackendOperationError(
                    f"SDF backend {self.backend_id!r} returned malformed field contents"
                )
            if len(value.fields) > self._limits.max_fields:
                raise ResourceLimitError(
                    f"backend returned {len(value.fields)} fields; "
                    f"limit is {self._limits.max_fields}"
                )
            for field in value.fields:
                self._result_field(operation, field)
            metadata = _bounded_metadata(value.metadata, self._limits)
            return FieldContents(fields=value.fields, metadata=metadata)
        if operation is Operation.WRITE_FIELDS:
            if not isinstance(value, Path):
                raise BackendOperationError(
                    f"SDF backend {self.backend_id!r} returned a non-path write result"
                )
            if value != Path(args[0]):
                raise BackendOperationError(
                    f"SDF backend {self.backend_id!r} wrote to {str(value)!r} "
                    "instead of the requested path"
                )
            return value
        if operation in {Operation.SAMPLE_VALUES, Operation.SAMPLE_GRADIENTS}:
            expected = (
                (len(args[1]),) if operation is Operation.SAMPLE_VALUES else (len(args[1]), 3)
            )
            try:
                raw_shape = _exact_values(
                    value.shape,  # type: ignore[attr-defined]
                    count=len(expected),
                    label="sample shape",
                )
                if any(
                    isinstance(part, bool) or not isinstance(part, numbers.Integral) or part < 0
                    for part in raw_shape
                ):
                    raise TypeError("sample dimensions must be nonnegative integers")
                shape = tuple(int(part) for part in raw_shape)
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise BackendOperationError(
                    f"SDF backend {self.backend_id!r} returned malformed samples for "
                    f"{operation.value}"
                ) from exc
            if shape != expected:
                raise BackendOperationError(
                    f"SDF backend {self.backend_id!r} returned sample shape {shape}; "
                    f"expected {expected}"
                )
            try:
                if operation is Operation.SAMPLE_VALUES:
                    samples = _exact_values(value, count=expected[0], label="sample values")
                    for index, sample in enumerate(samples):
                        _finite_real(sample, label=f"sample value {index}")
                else:
                    for row_index, gradient in enumerate(
                        _matrix_rows(
                            value,
                            count=expected[0],
                            width=3,
                            label="sample gradients",
                        )
                    ):
                        for component in gradient:
                            _finite_real(
                                component,
                                label=f"sample gradient row {row_index} component",
                            )
            except Exception as exc:
                raise BackendOperationError(
                    f"SDF backend {self.backend_id!r} returned malformed samples for "
                    f"{operation.value}"
                ) from exc
            return value
        raise BackendOperationError(
            f"SDF backend {self.backend_id!r} has no result contract for {operation.value}"
        )

    def _execute(self, operation: Operation, /, *args: Any, **kwargs: Any) -> Any:
        if operation not in self._info.operations:
            raise CapabilityUnavailableError(self.backend_id, (operation.value,))
        try:
            result = self._driver.execute(operation, *args, limits=self._limits, **kwargs)
            return self._validate_result(operation, result, args, kwargs)
        except _DRIVER_OPERATION_ERRORS:
            raise
        except Exception as exc:
            raise BackendOperationError(
                f"SDF backend {self.backend_id!r} failed while executing {operation.value}"
            ) from exc

    def _require_format(self, operation: Operation, value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("format must be a nonempty backend-supported identifier")
        if operation is Operation.READ_FIELDS:
            if value not in self._info.read_formats:
                raise CapabilityUnavailableError(self.backend_id, read_formats=(value,))
        elif operation is Operation.WRITE_FIELDS:
            if value not in self._info.write_formats:
                raise CapabilityUnavailableError(self.backend_id, write_formats=(value,))
        else:
            raise ValueError("format preflight requires a field I/O operation")
        return value

    def mesh_to_sdf(
        self,
        mesh: Mesh,
        *,
        voxel_size: float,
        half_width: float = 3.0,
        name: str | None = None,
        thread_count: int = 1,
    ) -> Field:
        """Rasterize a closed world-space mesh to a negative-inside SDF."""

        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be an sdf_tools.Mesh")
        return self._execute(
            Operation.MESH_TO_SDF,
            mesh,
            voxel_size=voxel_size,
            half_width=half_width,
            name=name,
            thread_count=thread_count,
        )

    def mesh_to_udf(
        self,
        mesh: Mesh,
        *,
        voxel_size: float,
        half_width: float = 3.0,
        name: str | None = None,
        thread_count: int = 1,
    ) -> Field:
        """Rasterize a world-space mesh to an unsigned distance field."""

        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be an sdf_tools.Mesh")
        return self._execute(
            Operation.MESH_TO_UDF,
            mesh,
            voxel_size=voxel_size,
            half_width=half_width,
            name=name,
            thread_count=thread_count,
        )

    def field_to_mesh(
        self,
        field: Field,
        *,
        isovalue: float = 0.0,
        adaptivity: float = 0.0,
        repair_orientation: bool = True,
        thread_count: int = 1,
    ) -> Mesh:
        return self._execute(
            Operation.FIELD_TO_MESH,
            self._field(field),
            isovalue=isovalue,
            adaptivity=adaptivity,
            repair_orientation=repair_orientation,
            thread_count=thread_count,
        )

    def union(self, left: Field, right: Field, *, thread_count: int = 1) -> Field:
        return self._binary(Operation.UNION, left, right, thread_count=thread_count)

    def intersection(self, left: Field, right: Field, *, thread_count: int = 1) -> Field:
        return self._binary(Operation.INTERSECTION, left, right, thread_count=thread_count)

    def difference(self, left: Field, right: Field, *, thread_count: int = 1) -> Field:
        return self._binary(Operation.DIFFERENCE, left, right, thread_count=thread_count)

    def _binary(
        self,
        operation: Operation,
        left: Field,
        right: Field,
        *,
        thread_count: int,
    ) -> Field:
        left = self._field(left)
        right = self._field(right)
        if (
            left.kind is not FieldKind.SIGNED_DISTANCE
            or right.kind is not FieldKind.SIGNED_DISTANCE
        ):
            raise InvalidGeometryError("SDF boolean operations require signed-distance fields")
        return self._execute(operation, left, right, thread_count=thread_count)

    def offset(self, field: Field, distance: float, *, thread_count: int = 1) -> Field:
        field = self._field(field)
        if field.kind is not FieldKind.SIGNED_DISTANCE:
            raise InvalidGeometryError("offset requires a signed-distance field")
        return self._execute(Operation.OFFSET, field, distance, thread_count=thread_count)

    def smooth(
        self,
        field: Field,
        *,
        width: int = 1,
        iterations: int = 1,
        thread_count: int = 1,
    ) -> Field:
        field = self._field(field)
        if field.kind is FieldKind.SIGNED_DISTANCE:
            operation = Operation.SMOOTH_SDF
        elif field.kind in {FieldKind.UNSIGNED_DISTANCE, FieldKind.SCALAR}:
            operation = Operation.SMOOTH_SCALAR
        else:
            raise InvalidGeometryError("smooth requires a distance or scalar field")
        return self._execute(
            operation,
            field,
            width=width,
            iterations=iterations,
            thread_count=thread_count,
        )

    def normalize(self, field: Field, *, thread_count: int = 1) -> Field:
        field = self._field(field)
        if field.kind is not FieldKind.SIGNED_DISTANCE:
            raise InvalidGeometryError("normalize requires a signed-distance field")
        return self._execute(Operation.NORMALIZE_SDF, field, thread_count=thread_count)

    def rebuild(
        self,
        field: Field,
        *,
        isovalue: float = 0.0,
        exterior_width: float = 3.0,
        interior_width: float = 3.0,
        thread_count: int = 1,
    ) -> Field:
        field = self._field(field)
        if field.kind is not FieldKind.SIGNED_DISTANCE:
            raise InvalidGeometryError("rebuild requires a signed-distance field")
        return self._execute(
            Operation.REBUILD_SDF,
            field,
            isovalue=isovalue,
            exterior_width=exterior_width,
            interior_width=interior_width,
            thread_count=thread_count,
        )

    def resample_to_match(
        self,
        field: Field,
        reference: Field,
        *,
        interpolation: Interpolation | str = Interpolation.QUADRATIC,
        thread_count: int = 1,
    ) -> Field:
        return self._execute(
            Operation.RESAMPLE_TO_MATCH,
            self._field(field),
            self._field(reference),
            interpolation=Interpolation(interpolation),
            thread_count=thread_count,
        )

    def sample_values(
        self,
        field: Field,
        points: object,
        *,
        interpolation: Interpolation | str = Interpolation.QUADRATIC,
        thread_count: int = 1,
    ) -> object:
        return self._execute(
            Operation.SAMPLE_VALUES,
            self._field(field),
            points,
            interpolation=Interpolation(interpolation),
            thread_count=thread_count,
        )

    def sample_gradients(
        self,
        field: Field,
        points: object,
        *,
        interpolation: Interpolation | str = Interpolation.QUADRATIC,
        thread_count: int = 1,
    ) -> object:
        return self._execute(
            Operation.SAMPLE_GRADIENTS,
            self._field(field),
            points,
            interpolation=Interpolation(interpolation),
            thread_count=thread_count,
        )

    def active_value_mask(
        self,
        field: Field,
        *,
        min_value: float | None = None,
        max_value: float | None = None,
        thread_count: int = 1,
    ) -> Field:
        return self._execute(
            Operation.ACTIVE_VALUE_MASK,
            self._field(field),
            min_value=min_value,
            max_value=max_value,
            thread_count=thread_count,
        )

    def topology_to_sdf(
        self,
        field: Field,
        *,
        half_width: int = 3,
        closing_steps: int = 0,
        dilation: int = 0,
        smoothing_steps: int = 0,
        thread_count: int = 1,
    ) -> Field:
        return self._execute(
            Operation.TOPOLOGY_TO_SDF,
            self._field(field),
            half_width=half_width,
            closing_steps=closing_steps,
            dilation=dilation,
            smoothing_steps=smoothing_steps,
            thread_count=thread_count,
        )

    def extract_enclosed_region(self, field: Field, *, thread_count: int = 1) -> Field:
        return self._execute(
            Operation.EXTRACT_ENCLOSED_REGION,
            self._field(field),
            thread_count=thread_count,
        )

    def read_fields(self, path: PathLike[str] | str, *, format: str) -> FieldContents:
        return self._execute(
            Operation.READ_FIELDS,
            path,
            format=self._require_format(Operation.READ_FIELDS, format),
        )

    def read_field(
        self,
        path: PathLike[str] | str,
        field_name: str,
        *,
        format: str,
    ) -> Field:
        if not isinstance(field_name, str) or not field_name:
            raise ValueError("field_name must be nonempty")
        return self._execute(
            Operation.READ_FIELDS,
            path,
            field_name=field_name,
            format=self._require_format(Operation.READ_FIELDS, format),
        )

    def write_fields(
        self,
        path: PathLike[str] | str,
        fields: Field | Sequence[Field],
        *,
        format: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        format = self._require_format(Operation.WRITE_FIELDS, format)
        if isinstance(fields, Field):
            values = (fields,)
        else:
            if not isinstance(fields, Sequence):
                raise TypeError("fields must be a Field or finite sequence of fields")
            field_count = len(fields)
            if field_count > self._limits.max_fields:
                raise ResourceLimitError(
                    f"field sequence has {field_count} fields; limit is {self._limits.max_fields}"
                )
            values = tuple(fields)
        if not values:
            raise ValueError("at least one field is required")
        return self._execute(
            Operation.WRITE_FIELDS,
            path,
            tuple(self._field(field) for field in values),
            format=format,
            metadata=_bounded_metadata(metadata, self._limits),
        )


_DEFAULT_TOOLKIT = SdfToolkit()


def create_session(
    *,
    backend: str = "auto",
    require: Iterable[Operation] = (),
    require_formats: Iterable[str] | Mapping[Operation | str, Iterable[str]] = (),
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> SdfSession:
    """Create a session from the process-wide policy-bound toolkit."""

    return _DEFAULT_TOOLKIT.create_session(
        backend=backend,
        require=require,
        require_formats=require_formats,
        limits=limits,
    )
