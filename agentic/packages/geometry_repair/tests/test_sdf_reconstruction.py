# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import trimesh

from geometry_repair.workers import sdf_reconstruction as backend

_TETRA_VERTICES = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float32,
)
_TETRA_TRIANGLES = np.asarray(
    (
        (0, 2, 1),
        (0, 1, 3),
        (0, 3, 2),
        (1, 2, 3),
    ),
    dtype=np.int32,
)
_OCTAHEDRON_VERTICES = np.asarray(
    (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ),
    dtype=np.float32,
)
_OPEN_OCTAHEDRON_TRIANGLES = np.asarray(
    (
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
    ),
    dtype=np.int32,
)


def _native_mesh(
    vertices: np.ndarray = _TETRA_VERTICES,
    triangles: np.ndarray = _TETRA_TRIANGLES,
    quads: np.ndarray | None = None,
) -> SimpleNamespace:
    if quads is None:
        quads = np.empty((0, 4), dtype=np.int32)
    return SimpleNamespace(vertices=vertices, triangles=triangles, quads=quads)


def _backend_info(
    version: tuple[int, int, int] = (13, 0, 0),
) -> SimpleNamespace:
    implementation_version = "openvdb-runtime-13.0.0+wu.3"
    provenance = {
        "library_version": list(version),
        "distribution_version": "13.0.0+wu.3",
        "file_format_version": 224,
        "module_path": "/runtime/openvdb.so",
        "module_sha256": "1" * 64,
        "source_lock_path": "/runtime/source-lock.json",
        "source_lock_sha256": "2" * 64,
        "source_distribution_version": "13.0.0+wu.3",
        "policy_schema": "world-understanding.openvdb-runtime-policy.v1",
        "source_commit": "7c03e1f084873cd1b3422c7ff7aec6ee681b3b38",
        "capabilities": ["mesh_to_sdf", "field_to_mesh"],
    }
    identity = {
        "backend_id": "openvdb",
        "implementation_version": implementation_version,
        "operations": sorted(
            {operation.value for operation in backend._RECONSTRUCTION_OPERATIONS} | {"write_fields"}
        ),
        "execution_mode": "in_process",
        "read_formats": [],
        "write_formats": ["vdb"],
        "supported_formats": ["vdb"],
        "provenance": provenance,
    }
    return SimpleNamespace(as_dict=lambda: dict(identity))


def _install_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    outputs: list[object] | None = None,
    runtime_version: tuple[int, int, int] = (13, 0, 0),
    failures: dict[str, Exception] | None = None,
) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    remaining_outputs = list(outputs or [_native_mesh()])
    failures = failures or {}

    def create_session(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(("create_session", args, kwargs))
        if error := failures.get("create_session"):
            raise error
        if runtime_version != (13, 0, 0):
            raise backend.sdf_tools.BackendUnavailableError("pinned runtime identity mismatch")

    def operation(name: str) -> Callable[..., object]:
        def invoke(*args: Any, **kwargs: Any) -> object:
            calls.append((name, args, kwargs))
            if error := failures.get(name):
                raise error
            if name == "field_to_mesh":
                return remaining_outputs.pop(0)
            return SimpleNamespace(label=name)

        return invoke

    session = SimpleNamespace(backend_info=_backend_info(runtime_version))
    for name in {
        "mesh_to_sdf",
        "mesh_to_udf",
        "active_value_mask",
        "topology_to_sdf",
        "extract_enclosed_region",
        "smooth",
        "field_to_mesh",
    }:
        setattr(session, name, operation(name))

    def finish_session(*args: Any, **kwargs: Any) -> SimpleNamespace:
        create_session(*args, **kwargs)
        return session

    monkeypatch.setattr(backend.sdf_tools, "create_session", finish_session)
    return calls


class _AlternateSdfDriver:
    def __init__(
        self,
        descriptor: backend.sdf_tools.BackendDescriptor,
        output: backend.sdf_tools.Mesh,
    ) -> None:
        self.descriptor = descriptor
        self.field_owner = object()
        self.output = output
        self.calls: list[tuple[backend.sdf_tools.Operation, tuple[Any, ...], dict[str, Any]]] = []
        self.consumed_payloads: list[tuple[str, ...]] = []

    def inspect(self) -> backend.sdf_tools.BackendInfo:
        return backend.sdf_tools.BackendInfo(
            backend_id=self.descriptor.backend_id,
            implementation_version=self.descriptor.implementation_version,
            operations=self.descriptor.operations,
            execution_mode=self.descriptor.execution_mode,
            read_formats=self.descriptor.read_formats,
            write_formats=self.descriptor.write_formats,
            provenance={
                "kernel": "alternate-test-kernel",
                "kernel_version": "1.0",
                "capabilities": ["mesh_to_sdf", "field_to_mesh"],
            },
        )

    def _derived_field(
        self,
        operation: backend.sdf_tools.Operation,
        source: backend.sdf_tools.Field | None,
        kind: backend.sdf_tools.FieldKind,
    ) -> backend.sdf_tools.Field:
        if source is None:
            payload = (operation.value,)
        else:
            payload = source._payload_for(self.descriptor.backend_id, self.field_owner)
            self.consumed_payloads.append(payload)
            payload = (*payload, operation.value)
        return backend.sdf_tools.Field(
            self.descriptor.backend_id,
            kind,
            payload,
            self.field_owner,
        )

    def execute(
        self,
        operation: backend.sdf_tools.Operation,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        self.calls.append((operation, args, kwargs))
        if operation is backend.sdf_tools.Operation.MESH_TO_SDF:
            return self._derived_field(
                operation,
                None,
                backend.sdf_tools.FieldKind.SIGNED_DISTANCE,
            )
        if operation is backend.sdf_tools.Operation.MESH_TO_UDF:
            return self._derived_field(
                operation,
                None,
                backend.sdf_tools.FieldKind.UNSIGNED_DISTANCE,
            )

        source = args[0]
        assert isinstance(source, backend.sdf_tools.Field)
        if operation is backend.sdf_tools.Operation.ACTIVE_VALUE_MASK:
            return self._derived_field(operation, source, backend.sdf_tools.FieldKind.MASK)
        if operation is backend.sdf_tools.Operation.TOPOLOGY_TO_SDF:
            return self._derived_field(
                operation,
                source,
                backend.sdf_tools.FieldKind.SIGNED_DISTANCE,
            )
        if operation is backend.sdf_tools.Operation.EXTRACT_ENCLOSED_REGION:
            return self._derived_field(operation, source, backend.sdf_tools.FieldKind.MASK)
        if operation in {
            backend.sdf_tools.Operation.SMOOTH_SDF,
            backend.sdf_tools.Operation.SMOOTH_SCALAR,
        }:
            return self._derived_field(operation, source, source.kind)
        if operation is backend.sdf_tools.Operation.FIELD_TO_MESH:
            payload = source._payload_for(self.descriptor.backend_id, self.field_owner)
            self.consumed_payloads.append(payload)
            return self.output
        raise AssertionError(f"unexpected alternate SDF operation: {operation.value}")


def _install_real_alternate_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    list[_AlternateSdfDriver],
    list[backend.sdf_tools.SdfSession],
    list[dict[str, Any]],
]:
    manifest = backend.sdf_tools.BackendLicenseManifest(
        schema="world-understanding.sdf-backend-license.v1",
        claim="test backend introduces no LGPL component",
        components=(
            backend.sdf_tools.LicenseComponent(
                "alternate-test-kernel",
                "1.0",
                "Apache-2.0",
                backend.sdf_tools.DependencyScope.RUNTIME,
                True,
            ),
        ),
    )
    descriptor = backend.sdf_tools.BackendDescriptor(
        backend_id="alternate",
        implementation_version="alternate-sdf-kernel-1.0",
        operations=backend._RECONSTRUCTION_OPERATIONS,
        priority=1,
        execution_mode="in_process",
        read_formats=frozenset(),
        write_formats=frozenset(),
        license_manifest=manifest,
    )
    output = backend.sdf_tools.Mesh(
        vertices=_TETRA_VERTICES,
        triangles=_TETRA_TRIANGLES,
        quads=np.empty((0, 4), dtype=np.int32),
    )
    drivers: list[_AlternateSdfDriver] = []

    def create_driver() -> _AlternateSdfDriver:
        driver = _AlternateSdfDriver(descriptor, output)
        drivers.append(driver)
        return driver

    extension = backend.sdf_tools.SdfBackendExtension(descriptor, create_driver)
    backend.sdf_tools.validate_license_manifest(extension.descriptor.license_manifest)
    registry = backend.sdf_tools.SdfBackendRegistry()
    registry._extensions[descriptor.backend_id] = extension  # type: ignore[attr-defined]
    toolkit = backend.sdf_tools.SdfToolkit(registry=registry, discover_installed=False)
    sessions: list[backend.sdf_tools.SdfSession] = []
    requests: list[dict[str, Any]] = []

    def create_session(**kwargs: Any) -> backend.sdf_tools.SdfSession:
        requests.append(kwargs)
        session = toolkit.create_session(**kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(backend.sdf_tools, "create_session", create_session)
    monkeypatch.setattr(
        backend,
        "get_sdf_backend_qualification",
        lambda backend_id: SimpleNamespace(
            backend_id=backend_id,
            qualification_id="geometry-repair.alternate-test.v1",
            required_operations=backend._RECONSTRUCTION_OPERATIONS,
            validate=lambda identity: (
                None
                if identity["backend_id"] == backend_id
                else pytest.fail("alternate qualification received the wrong identity")
            ),
        ),
    )
    return drivers, sessions, requests


def _call(
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]], name: str, occurrence: int = 0
) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    return [call for call in calls if call[0] == name][occurrence]


def _assert_single_thread_limits(calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> None:
    session_call = _call(calls, "create_session")
    limits = session_call[2]["limits"]
    assert limits.max_threads == 1
    operation_calls = [call for call in calls if call[0] != "create_session"]
    for _name, _args, kwargs in operation_calls:
        assert kwargs["thread_count"] == 1


def test_closed_signed_mesh_uses_level_set_route(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_backend(monkeypatch)

    result = backend.reconstruct_sdf_mesh(
        _TETRA_VERTICES,
        _TETRA_TRIANGLES,
        backend.SdfReconstructionControls(voxel_size=0.1),
    )

    assert [name for name, _args, _kwargs in calls] == [
        "create_session",
        "mesh_to_sdf",
        "smooth",
        "field_to_mesh",
    ]
    session_request = _call(calls, "create_session")[2]
    assert session_request["backend"] == "openvdb"
    assert session_request["require"] == backend._RECONSTRUCTION_OPERATIONS
    assert _call(calls, "field_to_mesh")[2]["isovalue"] == 0.0
    assert _call(calls, "field_to_mesh")[2]["repair_orientation"] is True
    assert result.evidence["algorithm"]["route"] == "signed_level_set"
    assert result.evidence["resource_usage"]["input_boundary_edges"] == 0
    assert result.evidence["requested_backend_id"] == "openvdb"
    assert result.evidence["selected_backend_identity"]["backend_id"] == "openvdb"
    assert result.evidence["selected_backend_identity"]["provenance"]["library_version"] == [
        13,
        0,
        0,
    ]
    assert isinstance(result.evidence, backend.SdfExecutionEvidence)
    assert result.evidence.schema_version == backend.SDF_EXECUTION_EVIDENCE_SCHEMA_VERSION
    assert result.evidence.backend_qualification_id == "geometry-repair.openvdb13.v1"
    assert result.evidence.required_operations == tuple(
        sorted(operation.value for operation in backend._RECONSTRUCTION_OPERATIONS)
    )
    assert result.evidence.limits.execution_limits.max_threads == 1
    assert result.evidence.backend_call_status == "succeeded"
    assert result.evidence.candidate_validation_status == "accepted"
    assert result.evidence.output_digests is not None
    assert result.evidence.selection_rejections == ()
    _assert_single_thread_limits(calls)


@pytest.mark.parametrize(
    ("vertices", "triangles", "control_kwargs", "route", "operations"),
    [
        (
            _TETRA_VERTICES,
            _TETRA_TRIANGLES,
            {},
            "signed_level_set",
            ("mesh_to_sdf", "smooth_sdf", "field_to_mesh"),
        ),
        (
            _OCTAHEDRON_VERTICES,
            _OPEN_OCTAHEDRON_TRIANGLES,
            {},
            "signed_topology_closing",
            (
                "mesh_to_udf",
                "active_value_mask",
                "topology_to_sdf",
                "extract_enclosed_region",
                "topology_to_sdf",
                "smooth_sdf",
                "field_to_mesh",
            ),
        ),
        (
            _TETRA_VERTICES,
            _TETRA_TRIANGLES,
            {"mode": "unsigned_offset"},
            "unsigned_offset",
            ("mesh_to_udf", "smooth_scalar", "field_to_mesh"),
        ),
    ],
)
def test_alternate_backend_completes_geometry_routes_through_real_session(
    monkeypatch: pytest.MonkeyPatch,
    vertices: np.ndarray,
    triangles: np.ndarray,
    control_kwargs: dict[str, Any],
    route: str,
    operations: tuple[str, ...],
) -> None:
    drivers, sessions, requests = _install_real_alternate_backend(monkeypatch)

    result = backend.reconstruct_sdf_mesh(
        vertices,
        triangles,
        backend.SdfReconstructionControls(
            backend_id="alternate",
            voxel_size=0.1,
            **control_kwargs,
        ),
    )

    assert len(drivers) == 1
    assert len(sessions) == 1
    assert isinstance(sessions[0], backend.sdf_tools.SdfSession)
    assert requests[0]["backend"] == "alternate"
    assert requests[0]["require"] == backend._RECONSTRUCTION_OPERATIONS
    driver = drivers[0]
    assert tuple(operation.value for operation, _args, _kwargs in driver.calls) == operations
    assert all(
        call_kwargs["limits"] is sessions[0].execution_limits for _, _, call_kwargs in driver.calls
    )
    assert all(call_kwargs["thread_count"] == 1 for _, _, call_kwargs in driver.calls)
    assert len(driver.consumed_payloads) == len(driver.calls) - 1
    assert result.evidence["selected_backend_identity"] == {
        "backend_id": "alternate",
        "implementation_version": "alternate-sdf-kernel-1.0",
        "operations": sorted(operation.value for operation in backend._RECONSTRUCTION_OPERATIONS),
        "execution_mode": "in_process",
        "read_formats": [],
        "write_formats": [],
        "supported_formats": [],
        "provenance": {
            "kernel": "alternate-test-kernel",
            "kernel_version": "1.0",
            "capabilities": ["mesh_to_sdf", "field_to_mesh"],
        },
    }
    assert result.evidence["algorithm"]["route"] == route


def test_open_signed_mesh_uses_v10_topology_closing_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_backend(monkeypatch)
    controls = backend.SdfReconstructionControls(
        voxel_size=0.1,
        half_width=3.2,
        offset_voxels=1.5,
        closing_steps=2,
    )

    result = backend.reconstruct_sdf_mesh(
        _OCTAHEDRON_VERTICES,
        _OPEN_OCTAHEDRON_TRIANGLES,
        controls,
    )

    assert [name for name, _args, _kwargs in calls] == [
        "create_session",
        "mesh_to_udf",
        "active_value_mask",
        "topology_to_sdf",
        "extract_enclosed_region",
        "topology_to_sdf",
        "smooth",
        "field_to_mesh",
    ]
    assert _call(calls, "create_session")[2]["require"] == backend._RECONSTRUCTION_OPERATIONS
    assert _call(calls, "active_value_mask")[2]["max_value"] == pytest.approx(0.15)
    first_topology = _call(calls, "topology_to_sdf")[2]
    assert first_topology["half_width"] == 4
    assert first_topology["closing_steps"] == 2
    assert first_topology["dilation"] == 0
    assert first_topology["smoothing_steps"] == 0
    second_topology = _call(calls, "topology_to_sdf", 1)[2]
    assert second_topology["closing_steps"] == 0
    assert second_topology["dilation"] == 0
    assert second_topology["smoothing_steps"] == 0
    assert _call(calls, "field_to_mesh")[2]["isovalue"] == pytest.approx(-0.15)
    assert result.evidence["algorithm"]["route"] == "signed_topology_closing"
    assert result.evidence["algorithm"]["explicit_closing_operator_applied"] is True
    assert result.evidence["resource_usage"]["input_boundary_edges"] > 0
    _assert_single_thread_limits(calls)


def test_closed_non_manifold_mesh_uses_topology_closing_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_backend(monkeypatch)
    triangles = np.concatenate((_TETRA_TRIANGLES, _TETRA_TRIANGLES[:1]), axis=0)

    result = backend.reconstruct_sdf_mesh(
        _TETRA_VERTICES,
        triangles,
        backend.SdfReconstructionControls(voxel_size=0.1),
    )

    assert _call(calls, "create_session")[2]["require"] == backend._RECONSTRUCTION_OPERATIONS
    assert _call(calls, "mesh_to_udf")
    assert not [call for call in calls if call[0] == "mesh_to_sdf"]
    usage = result.evidence["resource_usage"]
    assert usage["input_boundary_edges"] == 0
    assert usage["input_non_manifold_edges"] == 3
    assert usage["input_duplicate_faces"] == 1
    assert usage["input_topology_defects"] == 4
    assert result.evidence["algorithm"]["route"] == "signed_topology_closing"


def test_explicit_unsigned_mode_uses_scalar_filter_and_positive_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_backend(monkeypatch)
    controls = backend.SdfReconstructionControls(
        mode="unsigned_offset",
        voxel_size=0.2,
        offset_voxels=1.25,
    )

    result = backend.reconstruct_sdf_mesh(
        _TETRA_VERTICES,
        _TETRA_TRIANGLES,
        controls,
    )

    assert [name for name, _args, _kwargs in calls] == [
        "create_session",
        "mesh_to_udf",
        "smooth",
        "field_to_mesh",
    ]
    assert _call(calls, "create_session")[2]["require"] == backend._RECONSTRUCTION_OPERATIONS
    assert _call(calls, "field_to_mesh")[2]["isovalue"] == pytest.approx(0.25)
    assert result.evidence["algorithm"]["route"] == "unsigned_offset"
    assert result.evidence["algorithm"]["filter_operator"] == "smooth_scalar"
    _assert_single_thread_limits(calls)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voxel_size": True},
        {"voxel_size": "0.1"},
        {"voxel_size": 0.1, "closing_steps": True},
        {"voxel_size": 0.1, "smoothing_steps": True},
        {"voxel_size": 0.1, "deterministic_seed": False},
        {"voxel_size": 0.1, "max_grid_dimension": "256"},
    ],
)
def test_controls_reject_booleans_and_coercion(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        backend.SdfReconstructionControls(**kwargs)


def test_geometry_repair_controls_reject_exploratory_auto_selection() -> None:
    with pytest.raises(ValueError, match="explicit qualified SDF backend"):
        backend.SdfReconstructionControls(backend_id="auto", voxel_size=0.1)


def test_output_vertex_budget_is_capped_at_the_portable_driver_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_backend(monkeypatch)
    controls = backend.SdfReconstructionControls(
        voxel_size=0.1,
        max_output_faces=5_000_000,
    )

    result = backend.reconstruct_sdf_mesh(_TETRA_VERTICES, _TETRA_TRIANGLES, controls)

    limits = _call(calls, "create_session")[2]["limits"]
    assert 3 * controls.max_output_faces > backend.sdf_tools.DEFAULT_LIMITS.max_vertices
    assert limits.max_vertices == backend.sdf_tools.DEFAULT_LIMITS.max_vertices
    assert (
        result.evidence["limits"]["max_output_vertices"]
        == backend.sdf_tools.DEFAULT_LIMITS.max_vertices
    )


def test_caller_owned_session_is_reused_without_backend_rediscovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drivers, sessions, requests = _install_real_alternate_backend(monkeypatch)
    controls = backend.SdfReconstructionControls(
        backend_id="alternate",
        voxel_size=0.1,
    )
    session = backend.sdf_tools.create_session(
        backend="alternate",
        require=backend._RECONSTRUCTION_OPERATIONS,
        limits=backend._execution_limits(controls),
    )

    first = backend.reconstruct_sdf_mesh(
        _TETRA_VERTICES,
        _TETRA_TRIANGLES,
        controls,
        session=session,
    )
    second = backend.reconstruct_sdf_mesh(
        _TETRA_VERTICES,
        _TETRA_TRIANGLES,
        controls,
        session=session,
    )

    assert len(requests) == len(sessions) == len(drivers) == 1
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.triangles, second.triangles)


def test_backend_owns_input_output_and_records_canonical_array_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_vertices = _TETRA_VERTICES.copy()
    native_triangles = _TETRA_TRIANGLES.copy()
    calls = _install_fake_backend(
        monkeypatch,
        outputs=[_native_mesh(native_vertices, native_triangles)],
    )
    input_vertices = _TETRA_VERTICES.astype(np.float64)
    input_triangles = _TETRA_TRIANGLES.astype(np.int64)

    result = backend.reconstruct_sdf_mesh(
        input_vertices,
        input_triangles,
        backend.SdfReconstructionControls(voxel_size=0.1),
    )

    backend_mesh = _call(calls, "mesh_to_sdf")[1][0]
    assert backend_mesh.vertices.dtype == np.float32
    assert backend_mesh.triangles.dtype == np.int32
    assert not np.shares_memory(backend_mesh.vertices, input_vertices)
    assert not np.shares_memory(backend_mesh.triangles, input_triangles)
    assert not np.shares_memory(result.vertices, native_vertices)
    assert not np.shares_memory(result.triangles, native_triangles)
    assert result.vertices.flags.owndata
    assert result.triangles.flags.owndata

    input_vertices[:] = 9.0
    input_triangles[:] = 0
    native_vertices[:] = 8.0
    native_triangles[:] = 0
    assert np.array_equal(backend_mesh.vertices, _TETRA_VERTICES)
    assert np.array_equal(backend_mesh.triangles, _TETRA_TRIANGLES)
    assert not np.all(result.vertices == 8.0)
    assert not np.all(result.triangles == 0)

    canonical_input = result.evidence["source_digests"]
    assert (
        canonical_input["vertices"]["sha256"]
        == hashlib.sha256(_TETRA_VERTICES.tobytes(order="C")).hexdigest()
    )
    assert (
        canonical_input["triangles"]["sha256"]
        == hashlib.sha256(_TETRA_TRIANGLES.tobytes(order="C")).hexdigest()
    )
    json.dumps(result.evidence.model_dump(mode="json"), sort_keys=True, allow_nan=False)


def test_canonicalization_sorts_points_faces_and_flips_global_winding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    permutation = np.asarray((2, 0, 3, 1), dtype=np.int64)
    inverse = np.empty(4, dtype=np.int64)
    inverse[permutation] = np.arange(4)
    native_vertices = _TETRA_VERTICES[permutation].copy()
    native_vertices[1, 0] = np.float32(-0.0)
    native_triangles = inverse[_TETRA_TRIANGLES[:, (0, 2, 1)]]
    native_triangles = np.roll(native_triangles[[2, 0, 3, 1]], 1, axis=1).astype(np.int32)
    _install_fake_backend(
        monkeypatch,
        outputs=[_native_mesh(native_vertices, native_triangles)],
    )

    result = backend.reconstruct_sdf_mesh(
        _TETRA_VERTICES,
        _TETRA_TRIANGLES,
        backend.SdfReconstructionControls(voxel_size=0.1),
    )

    expected_vertices = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        dtype=np.float32,
    )
    expected_triangles = np.asarray(
        ((0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)),
        dtype=np.int32,
    )
    assert np.array_equal(result.vertices, expected_vertices)
    assert np.array_equal(result.triangles, expected_triangles)
    assert not np.signbit(result.vertices[result.vertices == 0.0]).any()
    assert result.evidence["geometry"]["output_signed_volume"] == pytest.approx(1.0 / 6.0)
    assert result.evidence["geometry"]["output_surface_area"] > 0.0


def test_quad_splitting_uses_shorter_diagonal_and_ties_choose_zero_two() -> None:
    empty = np.empty((0, 3), dtype=np.int64)
    tie_vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float32,
    )
    quad = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    assert np.array_equal(
        backend._triangulate_faces(tie_vertices, empty, quad),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
    )

    shorter_13 = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (4.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
        dtype=np.float32,
    )
    assert np.array_equal(
        backend._triangulate_faces(shorter_13, empty, quad),
        np.asarray(((0, 1, 3), (1, 2, 3)), dtype=np.int64),
    )


def _cpp_float32_cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(
        (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ),
        dtype=np.float32,
    )


def test_volume_uses_local_float64_coordinates_and_area_matches_cpp_float32() -> None:
    vertices = np.asarray(
        (
            (0.125, -0.25, 0.375),
            (1.125, 0.5, -0.125),
            (-0.75, 1.25, 0.625),
            (0.5, -0.875, 1.5),
        ),
        dtype=np.float32,
    )
    triangles = _TETRA_TRIANGLES.astype(np.int64)
    expected_area = 0.0
    for face in triangles:
        a, b, c = vertices[face]
        left = np.asarray(b - a, dtype=np.float32)
        right = np.asarray(c - a, dtype=np.float32)
        area_cross = _cpp_float32_cross(left, right)
        squared_length = np.float32(
            np.float32(area_cross[0] * area_cross[0] + area_cross[1] * area_cross[1])
            + area_cross[2] * area_cross[2]
        )
        length = np.float32(math.sqrt(float(squared_length)))
        expected_area += 0.5 * float(length)

    local = vertices.astype(np.float64) - vertices[0].astype(np.float64)
    expected_volume = sum(
        float(np.dot(local[a], np.cross(local[b], local[c]))) / 6.0 for a, b, c in triangles
    )
    assert backend._signed_volume(vertices, triangles) == pytest.approx(expected_volume)
    assert backend._surface_area(vertices, triangles) == expected_area


def test_signed_volume_is_stable_for_far_translated_float32_mesh() -> None:
    triangles = _TETRA_TRIANGLES.astype(np.int64)
    translation = np.asarray((10_000.0, -20_000.0, 30_000.0), dtype=np.float32)
    translated = np.asarray(_TETRA_VERTICES + translation, dtype=np.float32)

    assert backend._signed_volume(_TETRA_VERTICES, triangles) == pytest.approx(1.0 / 6.0)
    assert backend._signed_volume(translated, triangles) == pytest.approx(1.0 / 6.0)


def test_signed_volume_ignores_far_unreferenced_vertex() -> None:
    vertices = np.concatenate(
        (
            np.asarray(((-3.0e38, -3.0e38, -3.0e38),), dtype=np.float32),
            _TETRA_VERTICES,
        ),
        axis=0,
    )
    triangles = _TETRA_TRIANGLES.astype(np.int64) + 1

    assert backend._signed_volume(vertices, triangles) == pytest.approx(1.0 / 6.0)


def test_signed_volume_is_exactly_stable_across_face_orderings() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=123.0)
    vertices = np.asarray(
        mesh.vertices + np.asarray((100_000.0, -200_000.0, 300_000.0)),
        dtype=np.float32,
    )
    triangles = np.asarray(mesh.faces, dtype=np.int64)
    shuffled = triangles[np.random.default_rng(6174).permutation(len(triangles))]

    assert backend._signed_volume(vertices, triangles) == backend._signed_volume(vertices, shuffled)


def test_different_native_orderings_have_identical_canonical_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    permutation = np.asarray((3, 1, 0, 2), dtype=np.int64)
    inverse = np.empty(4, dtype=np.int64)
    inverse[permutation] = np.arange(4)
    variant_vertices = _TETRA_VERTICES[permutation].copy()
    variant_triangles = inverse[_TETRA_TRIANGLES]
    variant_triangles = np.roll(variant_triangles[[3, 1, 0, 2]], -1, axis=1).astype(np.int32)
    _install_fake_backend(
        monkeypatch,
        outputs=[
            _native_mesh(_TETRA_VERTICES, _TETRA_TRIANGLES),
            _native_mesh(variant_vertices, variant_triangles),
        ],
    )
    controls = backend.SdfReconstructionControls(voxel_size=0.1)

    first = backend.reconstruct_sdf_mesh(_TETRA_VERTICES, _TETRA_TRIANGLES, controls)
    second = backend.reconstruct_sdf_mesh(_TETRA_VERTICES, _TETRA_TRIANGLES, controls)

    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.triangles, second.triangles)
    assert first.evidence == second.evidence


@pytest.mark.parametrize(
    ("vertices", "triangles", "message"),
    [
        (_TETRA_VERTICES[:3], _TETRA_TRIANGLES, "at least four vertices"),
        (
            np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, np.inf))),
            _TETRA_TRIANGLES,
            "finite float32",
        ),
        (_TETRA_VERTICES, _TETRA_TRIANGLES.astype(np.float32), "integer indices"),
        (
            _TETRA_VERTICES,
            np.asarray(((0, 0, 1), (0, 1, 2), (0, 2, 3), (0, 3, 1))),
            "repeats a vertex",
        ),
    ],
)
def test_invalid_input_fails_before_runtime_inspection(
    monkeypatch: pytest.MonkeyPatch,
    vertices: np.ndarray,
    triangles: np.ndarray,
    message: str,
) -> None:
    calls = _install_fake_backend(monkeypatch)

    with pytest.raises(backend.sdf_tools.InvalidGeometryError, match=message):
        backend.reconstruct_sdf_mesh(
            vertices,
            triangles,
            backend.SdfReconstructionControls(voxel_size=0.1),
        )

    assert calls == []


def test_explicit_backend_unavailability_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_backend(
        monkeypatch,
        failures={
            "create_session": backend.sdf_tools.BackendUnavailableError(
                "distribution identity mismatch"
            )
        },
    )

    with pytest.raises(
        backend.sdf_tools.BackendUnavailableError, match="distribution identity mismatch"
    ) as caught:
        backend.reconstruct_sdf_mesh(
            _TETRA_VERTICES,
            _TETRA_TRIANGLES,
            backend.SdfReconstructionControls(voxel_size=0.1),
        )

    assert [name for name, _args, _kwargs in calls] == ["create_session"]
    assert calls[0][2]["backend"] == "openvdb"
    assert calls[0][2]["require"] == backend._RECONSTRUCTION_OPERATIONS
    evidence = backend.sdf_execution_evidence_from_exception(caught.value)
    assert evidence is not None
    assert evidence.requested_backend_id == "openvdb"
    assert evidence.selected_backend_identity is None
    assert evidence.backend_qualification_id == "geometry-repair.openvdb13.v1"
    assert evidence.backend_call_status == "unavailable"
    assert evidence.candidate_validation_status == "not_evaluated"


def test_backend_failure_propagates_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = RuntimeError("native conversion failed")
    calls = _install_fake_backend(
        monkeypatch,
        failures={"mesh_to_sdf": failure},
    )

    with pytest.raises(RuntimeError, match="native conversion failed") as caught:
        backend.reconstruct_sdf_mesh(
            _TETRA_VERTICES,
            _TETRA_TRIANGLES,
            backend.SdfReconstructionControls(voxel_size=0.1),
        )

    assert caught.value is failure
    evidence = backend.sdf_execution_evidence_from_exception(caught.value)
    assert evidence is not None
    assert evidence.backend_call_status == "failed"
    assert evidence.candidate_validation_status == "not_evaluated"
    assert evidence.output_digests is None
    assert [name for name, _args, _kwargs in calls] == [
        "create_session",
        "mesh_to_sdf",
    ]


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (
            _native_mesh(
                triangles=np.concatenate((_TETRA_TRIANGLES, _TETRA_TRIANGLES[:1]), axis=0)
            ),
            "duplicate oriented faces",
        ),
        (
            _native_mesh(
                vertices=np.concatenate(
                    (_TETRA_VERTICES, np.asarray(((2.0, 0.0, 0.0),), dtype=np.float32)),
                    axis=0,
                ),
                triangles=np.concatenate(
                    (_TETRA_TRIANGLES, np.asarray(((0, 1, 4),), dtype=np.int32)), axis=0
                ),
            ),
            "degenerate triangle",
        ),
        (
            _native_mesh(
                vertices=np.asarray(
                    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, np.nan, 0.0), (0.0, 0.0, 1.0)),
                    dtype=np.float32,
                )
            ),
            "non-finite vertex",
        ),
        (
            _native_mesh(triangles=np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 9)))),
            "out of range",
        ),
    ],
)
def test_invalid_native_surface_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    output: object,
    message: str,
) -> None:
    _install_fake_backend(monkeypatch, outputs=[output])

    with pytest.raises(backend.sdf_tools.InvalidGeometryError, match=message) as caught:
        backend.reconstruct_sdf_mesh(
            _TETRA_VERTICES,
            _TETRA_TRIANGLES,
            backend.SdfReconstructionControls(voxel_size=0.1),
        )

    evidence = backend.sdf_execution_evidence_from_exception(caught.value)
    assert evidence is not None
    assert evidence.backend_call_status == "succeeded"
    assert evidence.candidate_validation_status == "rejected"
    assert evidence.output_digests is None
