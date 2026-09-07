# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted-artifact VDB I/O through the official in-process Python module.

Native VDB parsing may allocate before Python regains control. The checks in
this module do not make an untrusted VDB file safe to parse.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .errors import InvalidGeometryError, ResourceLimitError
from .mesh import _check_output_limit, _inspect_output_limits
from .runtime import _call_native, require_runtime
from .types import DEFAULT_LIMITS, Capability, ExecutionLimits, VDBContents

_MAX_METADATA_DEPTH = 16


def _path(value: os.PathLike[str] | str) -> str:
    try:
        result = os.fspath(value)
    except TypeError as exc:
        raise TypeError("VDB path must be a string or path-like value") from exc
    if not isinstance(result, str):
        raise TypeError("VDB path must resolve to text, not bytes")
    return result


def _check_input_file_limit(path: str, limits: ExecutionLimits) -> None:
    try:
        status = os.stat(path)
    except OSError as exc:
        raise InvalidGeometryError(f"VDB input could not be inspected: {path}") from exc
    if not stat.S_ISREG(status.st_mode):
        raise InvalidGeometryError("VDB input must be a regular file")
    size = status.st_size
    if size > limits.max_file_bytes:
        raise ResourceLimitError(f"VDB input is {size} bytes; limit is {limits.max_file_bytes}")


def _require_trusted_artifact(value: bool) -> None:
    if value is not True:
        raise ValueError("native VDB reads require trusted_artifact=True")


def _metadata_integer(grid: Any, key: str) -> int:
    try:
        metadata = dict(grid.metadata)
        value = metadata[key]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise InvalidGeometryError(f"VDB field metadata is missing a valid {key!r}") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidGeometryError(f"VDB field metadata has a malformed {key!r}")
    return value


def _preflight_grid_metadata(grids: Sequence[Any], limits: ExecutionLimits) -> None:
    field_count = len(grids)
    if field_count > limits.max_fields:
        raise ResourceLimitError(
            f"VDB input has {field_count} fields; limit is {limits.max_fields}"
        )
    total_active_voxels = 0
    total_memory_bytes = 0
    for grid in grids:
        active_voxels = _metadata_integer(grid, "file_voxel_count")
        memory_bytes = _metadata_integer(grid, "file_mem_bytes")
        if active_voxels > limits.max_active_voxels:
            raise ResourceLimitError(
                f"VDB field has {active_voxels} active voxels; limit is {limits.max_active_voxels}"
            )
        if memory_bytes > limits.max_field_memory_bytes:
            raise ResourceLimitError(
                f"VDB field requires {memory_bytes} bytes; limit is {limits.max_field_memory_bytes}"
            )
        total_active_voxels += active_voxels
        total_memory_bytes += memory_bytes
        if total_active_voxels > limits.max_active_voxels:
            raise ResourceLimitError(
                f"VDB fields have {total_active_voxels} aggregate active voxels; "
                f"limit is {limits.max_active_voxels}"
            )
        if total_memory_bytes > limits.max_total_field_memory_bytes:
            raise ResourceLimitError(
                f"VDB fields require {total_memory_bytes} aggregate bytes; "
                f"limit is {limits.max_total_field_memory_bytes}"
            )


def _preflight_write_grids(grids: Sequence[Any], limits: ExecutionLimits) -> None:
    total_active_voxels = 0
    total_memory_bytes = 0
    for grid in grids:
        active_voxels, memory_bytes = _inspect_output_limits(grid, limits)
        if active_voxels is not None:
            total_active_voxels += active_voxels
        if total_active_voxels > limits.max_active_voxels:
            raise ResourceLimitError(
                f"VDB fields have {total_active_voxels} aggregate active voxels; "
                f"limit is {limits.max_active_voxels}"
            )
        total_memory_bytes += memory_bytes
        if total_memory_bytes > limits.max_total_field_memory_bytes:
            raise ResourceLimitError(
                f"VDB fields require {total_memory_bytes} aggregate bytes; "
                f"limit is {limits.max_total_field_memory_bytes}"
            )


def _bounded_metadata(
    metadata: Mapping[str, Any] | None, limits: ExecutionLimits
) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    remaining = limits.max_metadata_entries
    byte_limit = min(limits.max_metadata_bytes, limits.max_file_bytes)
    remaining_bytes = byte_limit
    active: set[int] = set()

    def consume() -> None:
        nonlocal remaining
        if remaining == 0:
            raise ResourceLimitError(
                f"metadata exceeds the configured entry limit ({limits.max_metadata_entries})"
            )
        remaining -= 1

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

    def copy(value: Any, depth: int) -> Any:
        if depth > _MAX_METADATA_DEPTH:
            raise ResourceLimitError(
                f"metadata nesting exceeds the supported depth ({_MAX_METADATA_DEPTH})"
            )
        if value is None or isinstance(value, bool | str):
            consume_scalar(value)
            return value
        if isinstance(value, numbers.Integral):
            result = int(value)
            if result.bit_length() > 4_096:
                raise ResourceLimitError("metadata integer exceeds the supported size")
            consume_scalar(result)
            return result
        if isinstance(value, numbers.Real):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("metadata numbers must be finite")
            consume_scalar(result)
            return result
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active:
                raise ValueError("metadata must not contain cycles")
            active.add(identity)
            try:
                result: dict[str, Any] = {}
                consume_bytes(1)
                first = True
                for key in value:
                    consume()
                    if not isinstance(key, str):
                        raise TypeError("metadata keys must be strings")
                    if not first:
                        consume_bytes(1)
                    first = False
                    consume_scalar(key)
                    consume_bytes(1)
                    result[key] = copy(value[key], depth + 1)
                consume_bytes(1)
                return result
            finally:
                active.remove(identity)
        if type(value) in (list, tuple):
            if len(value) > remaining:
                raise ResourceLimitError(
                    f"metadata exceeds the configured entry limit ({limits.max_metadata_entries})"
                )
            identity = id(value)
            if identity in active:
                raise ValueError("metadata must not contain cycles")
            active.add(identity)
            try:
                items: list[Any] = []
                consume_bytes(1)
                first = True
                for item in value:
                    consume()
                    if not first:
                        consume_bytes(1)
                    first = False
                    items.append(copy(item, depth + 1))
                consume_bytes(1)
                return items
            finally:
                active.remove(identity)
        raise TypeError("metadata values must be finite JSON-compatible data")

    result = copy(metadata, 0)
    assert isinstance(result, dict)
    encoded = json.dumps(
        result,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    if len(encoded) > byte_limit:
        raise ResourceLimitError(f"metadata is {len(encoded)} bytes; limit is {byte_limit}")
    return result


def read_grid(
    path: os.PathLike[str] | str,
    grid_name: str,
    *,
    trusted_artifact: bool,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Read one named grid from a caller-authenticated VDB artifact.

    File-size and metadata checks run before the full grid read, and output
    checks run afterwards. They are not an allocation sandbox for native VDB
    parsing and must not be used to admit model- or user-supplied files.
    """

    _require_trusted_artifact(trusted_artifact)
    if not isinstance(grid_name, str) or not grid_name:
        raise ValueError("grid_name must be a nonempty string")
    source = _path(path)
    _check_input_file_limit(source, limits)
    module = require_runtime((Capability.VDB_IO,))
    grid_metadata = _call_native("readGridMetadata", module.readGridMetadata, source, grid_name)
    _preflight_grid_metadata((grid_metadata,), limits)
    grid = _call_native("read", module.read, source, grid_name)
    _check_output_limit(grid, limits)
    return grid


def read_all(
    path: os.PathLike[str] | str,
    *,
    trusted_artifact: bool,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> VDBContents:
    """Read every grid from a caller-authenticated VDB artifact.

    File-size and metadata checks run before full-grid reads, and output checks
    run afterwards. They cannot bound allocations performed while native
    OpenVDB parses an artifact.
    """

    _require_trusted_artifact(trusted_artifact)
    source = _path(path)
    _check_input_file_limit(source, limits)
    module = require_runtime((Capability.VDB_IO,))
    metadata_grids = tuple(_call_native("readAllGridMetadata", module.readAllGridMetadata, source))
    _preflight_grid_metadata(metadata_grids, limits)
    grids, metadata = _call_native("readAll", module.readAll, source)
    owned_grids = tuple(grids)
    if len(owned_grids) != len(metadata_grids):
        raise InvalidGeometryError("VDB field inventory changed during the read")
    try:
        bounded_metadata = _bounded_metadata(metadata, limits)
    except ResourceLimitError:
        raise
    except (TypeError, ValueError) as exc:
        raise InvalidGeometryError("VDB file metadata is malformed") from exc
    total_active_voxels = 0
    total_memory_bytes = 0
    for grid in owned_grids:
        active_voxels, memory_bytes = _inspect_output_limits(grid, limits)
        total_memory_bytes += memory_bytes
        if total_memory_bytes > limits.max_total_field_memory_bytes:
            raise ResourceLimitError(
                f"VDB fields require {total_memory_bytes} aggregate bytes; "
                f"limit is {limits.max_total_field_memory_bytes}"
            )
        if active_voxels is not None:
            total_active_voxels += active_voxels
        if total_active_voxels > limits.max_active_voxels:
            raise ResourceLimitError(
                f"VDB fields have {total_active_voxels} aggregate active voxels; "
                f"limit is {limits.max_active_voxels}"
            )
    return VDBContents(grids=owned_grids, metadata=bounded_metadata)


def write(
    path: os.PathLike[str] | str,
    grids: Any | Sequence[Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Path:
    """Write one grid or a sequence of grids and return the output path."""

    output = Path(_path(path))
    if not output.name:
        raise ValueError("output path must name a file")
    if isinstance(grids, list | tuple):
        field_count = len(grids)
        if field_count > limits.max_fields:
            raise ResourceLimitError(
                f"VDB output has {field_count} fields; limit is {limits.max_fields}"
            )
        payload: Any = list(grids)
        values = tuple(payload)
    else:
        payload = grids
        values = (payload,)
    if len(values) > limits.max_fields:
        raise ResourceLimitError(
            f"VDB output has {len(values)} fields; limit is {limits.max_fields}"
        )
    bounded_metadata = _bounded_metadata(metadata, limits)
    _preflight_write_grids(values, limits)
    module = require_runtime((Capability.VDB_IO,))

    output_parent = output.parent
    try:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix=f".{output.name}.", dir=output_parent
        )
    except OSError as exc:
        raise InvalidGeometryError(f"VDB output could not be created: {output}") from exc

    with temporary_directory as private_directory:
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix="field.", suffix=".vdb", dir=private_directory
            )
        except OSError as exc:
            raise InvalidGeometryError(f"VDB output could not be created: {output}") from exc
        temporary = Path(temporary_name)
        try:
            _call_native("write", module.write, str(temporary), payload, metadata=bounded_metadata)
            try:
                descriptor_status = os.fstat(descriptor)
                path_status = temporary.lstat()
            except OSError as exc:
                raise InvalidGeometryError(
                    "native VDB write did not produce an inspectable output file"
                ) from exc
            if (
                descriptor_status.st_dev,
                descriptor_status.st_ino,
            ) != (path_status.st_dev, path_status.st_ino):
                raise InvalidGeometryError("native VDB write replaced its secure staging file")
            if not stat.S_ISREG(descriptor_status.st_mode) or not stat.S_ISREG(path_status.st_mode):
                raise InvalidGeometryError("native VDB write did not produce a regular file")
            if descriptor_status.st_size > limits.max_file_bytes:
                raise ResourceLimitError(
                    f"VDB output is {descriptor_status.st_size} bytes; "
                    f"limit is {limits.max_file_bytes}"
                )
            os.replace(temporary, output)
        finally:
            os.close(descriptor)
    return output
