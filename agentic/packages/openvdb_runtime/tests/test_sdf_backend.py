# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import sdf_tools

import openvdb_runtime
from openvdb_runtime import sdf_backend


def _runtime_info() -> openvdb_runtime.RuntimeInfo:
    return openvdb_runtime.RuntimeInfo(
        library_version=(13, 0, 0),
        distribution_version="13.0.0+wu.3",
        file_format_version=230,
        module_path="/runtime/openvdb.so",
        module_sha256="1" * 64,
        source_lock_path="/runtime/_source_lock.json",
        source_lock_sha256="2" * 64,
        source_distribution_version="13.0.0+wu.3",
        policy_schema="world-understanding.openvdb-runtime-policy.v1",
        source_commit="7c03e1f084873cd1b3422c7ff7aec6ee681b3b38",
        capabilities=frozenset(openvdb_runtime.Capability),
    )


def test_extension_is_in_process_and_license_admitted() -> None:
    extension = sdf_backend.sdf_backend_extension()

    assert extension.descriptor.backend_id == "openvdb"
    assert extension.descriptor.execution_mode == "in_process"
    assert extension.descriptor.operations == frozenset(sdf_tools.Operation) - {
        sdf_tools.Operation.READ_FIELDS
    }
    sdf_tools.validate_license_manifest(extension.descriptor.license_manifest)
    introduced = {
        component.name
        for component in extension.descriptor.license_manifest.components
        if component.introduced_by_backend
    }
    assert introduced == {
        "Bitshuffle",
        "FastLZ-derived",
        "LZ4",
        "OpenEXR Half",
        "OpenVDB",
        "Zstandard",
        "c-blosc",
        "libdivsufsort-lite",
        "nanobind",
        "oneTBB",
        "openvdb-runtime",
        "robin-map",
        "sdf-tools",
        "zlib",
        "zlib-ng-derived",
    }
    components = {
        component.name: component for component in extension.descriptor.license_manifest.components
    }
    assert components["nanobind"].scope is sdf_tools.DependencyScope.BUNDLED
    assert components["robin-map"].scope is sdf_tools.DependencyScope.BUNDLED
    assert components["NumPy"].scope is sdf_tools.DependencyScope.PREEXISTING_APPLICATION
    assert components["NumPy"].version == "application-provided compatible runtime"
    assert not components["NumPy"].introduced_by_backend


def test_inspect_preserves_source_locked_provenance(monkeypatch) -> None:
    module = SimpleNamespace()
    monkeypatch.setattr(sdf_backend.vdb, "require_runtime", lambda **_kwargs: module)
    monkeypatch.setattr(sdf_backend.vdb, "inspect_runtime", lambda value: _runtime_info())

    info = sdf_backend.OpenVdbSdfBackend().inspect()

    assert info.backend_id == "openvdb"
    assert info.execution_mode == "in_process"
    assert sdf_tools.Operation.READ_FIELDS not in info.operations
    assert sdf_tools.Operation.WRITE_FIELDS in info.operations
    assert info.provenance["library_version"] == (13, 0, 0)
    assert info.provenance["source_lock_sha256"] == "2" * 64


def test_mesh_and_field_translation_stays_backend_neutral(monkeypatch) -> None:
    calls: list[tuple[str, object, dict[str, object]]] = []

    def mesh_to_level_set(mesh, **kwargs):
        calls.append(("mesh_to_sdf", mesh, kwargs))
        return "native-field"

    def volume_to_mesh(field, **kwargs):
        calls.append(("field_to_mesh", field, kwargs))
        return openvdb_runtime.Mesh(
            vertices=np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=np.float32),
            triangles=np.array([(0, 1, 2)], dtype=np.int32),
        )

    monkeypatch.setattr(sdf_backend.vdb, "mesh_to_level_set", mesh_to_level_set)
    monkeypatch.setattr(sdf_backend.vdb, "volume_to_mesh", volume_to_mesh)
    driver = sdf_backend.OpenVdbSdfBackend()
    limits = sdf_tools.ExecutionLimits()
    mesh = sdf_tools.Mesh(
        vertices=np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=np.float32),
        triangles=np.array([(0, 1, 2)], dtype=np.int32),
    )

    field = driver.execute(
        sdf_tools.Operation.MESH_TO_SDF,
        mesh,
        voxel_size=0.1,
        limits=limits,
    )
    result = driver.execute(
        sdf_tools.Operation.FIELD_TO_MESH,
        field,
        limits=limits,
    )

    assert isinstance(field, sdf_tools.Field)
    assert field.backend_id == "openvdb"
    assert field.kind is sdf_tools.FieldKind.SIGNED_DISTANCE
    assert isinstance(result, sdf_tools.Mesh)
    assert calls[0][0] == "mesh_to_sdf"
    assert isinstance(calls[0][1], openvdb_runtime.Mesh)
    assert calls[1][1] == "native-field"


def test_driver_source_has_no_process_transport() -> None:
    source = sdf_backend.Path(__file__).parent.parent / "openvdb_runtime/sdf_backend.py"
    text = source.read_text(encoding="utf-8")
    assert "import subprocess" not in text
    assert "from subprocess" not in text


def test_read_fields_fails_static_capability_admission() -> None:
    extension = sdf_backend.sdf_backend_extension()
    sdf_tools.validate_license_manifest(extension.descriptor.license_manifest)
    registry = sdf_tools.SdfBackendRegistry()
    registry._extensions[extension.descriptor.backend_id] = extension  # type: ignore[attr-defined]
    toolkit = sdf_tools.SdfToolkit(
        registry=registry,
        discover_installed=False,
    )

    with pytest.raises(sdf_tools.CapabilityUnavailableError, match="read_fields"):
        toolkit.create_session(
            backend="openvdb",
            require={sdf_tools.Operation.READ_FIELDS},
        )


def test_read_fields_is_not_available_through_direct_driver_execution() -> None:
    with pytest.raises(sdf_tools.CapabilityUnavailableError, match="read_fields"):
        sdf_backend.OpenVdbSdfBackend().execute(
            "read_fields",  # type: ignore[arg-type]
            "untrusted.vdb",
            format="vdb",
            limits=sdf_tools.ExecutionLimits(),
        )


def test_mesh_cardinality_is_checked_before_native_array_conversion() -> None:
    class OversizedVertices:
        shape = (2, 3)

        def __len__(self) -> int:
            return 2

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("array conversion must not run before cardinality admission")

    mesh = sdf_tools.Mesh(
        vertices=OversizedVertices(),
        triangles=[(0, 0, 0)],
    )
    limits = sdf_tools.ExecutionLimits(max_vertices=1)

    with pytest.raises(sdf_tools.ResourceLimitError, match="2 vertices"):
        sdf_backend.OpenVdbSdfBackend().execute(
            sdf_tools.Operation.MESH_TO_SDF,
            mesh,
            voxel_size=0.1,
            limits=limits,
        )


def test_mesh_rejects_deceptive_array_like_without_materializing() -> None:
    class DeceptiveVertices:
        shape = (1, 3)

        def __len__(self) -> int:
            return 1

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("untrusted __array__ must not be invoked")

    mesh = sdf_tools.Mesh(vertices=DeceptiveVertices(), triangles=[(0, 1, 2)])

    with pytest.raises(sdf_tools.InvalidGeometryError, match="NumPy array"):
        sdf_backend.OpenVdbSdfBackend().execute(
            sdf_tools.Operation.MESH_TO_SDF,
            mesh,
            voxel_size=0.1,
            limits=sdf_tools.ExecutionLimits(),
        )


def test_mesh_rejects_oversized_ragged_rows_before_inspection() -> None:
    class UntouchedRow:
        def __len__(self) -> int:
            raise AssertionError("oversized rows must not be inspected")

    mesh = sdf_tools.Mesh(
        vertices=[UntouchedRow(), UntouchedRow()],
        triangles=[(0, 1, 2)],
    )

    with pytest.raises(sdf_tools.ResourceLimitError, match="2 vertices"):
        sdf_backend.OpenVdbSdfBackend().execute(
            sdf_tools.Operation.MESH_TO_SDF,
            mesh,
            voxel_size=0.1,
            limits=sdf_tools.ExecutionLimits(max_vertices=1),
        )


@pytest.mark.parametrize(
    ("memory_bytes", "limits", "field_count", "message"),
    [
        (9, sdf_tools.ExecutionLimits(max_field_memory_bytes=8), 1, "9 bytes"),
        (
            5,
            sdf_tools.ExecutionLimits(max_total_field_memory_bytes=9),
            2,
            "aggregate bytes",
        ),
    ],
)
def test_write_fields_preflights_memory_before_any_deep_copy(
    monkeypatch, memory_bytes, limits, field_count, message
) -> None:
    driver = sdf_backend.OpenVdbSdfBackend()
    copies: list[str] = []

    class Payload:
        def memUsage(self) -> int:
            return memory_bytes

        def deepCopy(self):
            copies.append("copied")
            raise AssertionError("deepCopy must not run before aggregate admission")

    fields = tuple(
        sdf_tools.Field(
            backend_id="openvdb",
            kind=sdf_tools.FieldKind.SIGNED_DISTANCE,
            _payload=Payload(),
            _owner=driver.field_owner,
        )
        for _ in range(field_count)
    )

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("native write must not run")

    monkeypatch.setattr(sdf_backend.vdb, "write", unexpected_write)

    with pytest.raises(sdf_tools.ResourceLimitError, match=message):
        driver.execute(
            sdf_tools.Operation.WRITE_FIELDS,
            "asset.vdb",
            fields,
            format="vdb",
            metadata={},
            limits=limits,
        )

    assert copies == []
