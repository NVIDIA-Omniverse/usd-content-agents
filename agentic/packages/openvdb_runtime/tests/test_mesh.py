# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

import openvdb_runtime.mesh as mesh_module
from openvdb_runtime import (
    CapabilityUnavailableError,
    ExecutionLimits,
    InvalidGeometryError,
    Mesh,
    ResourceLimitError,
    mesh_to_level_set,
    mesh_to_unsigned_distance_field,
    volume_to_mesh,
)


def _call(module, name):
    return next(call for call in module.calls if call[0] == name)


def test_mesh_to_level_set_prefers_bounded_native_tool(fake_openvdb, tetrahedron_mesh):
    grid = mesh_to_level_set(
        tetrahedron_mesh,
        voxel_size=0.05,
        half_width=4.0,
        name="body",
        thread_count=3,
    )

    assert grid.label == "mesh"
    assert grid.name == "body"
    _, args, kwargs = _call(fake_openvdb, "mesh_to_level_set")
    assert args[0].dtype == np.float32
    assert args[1].dtype == np.int32
    assert kwargs == {
        "voxel_size": 0.05,
        "half_width": 4.0,
        "thread_count": 3,
        "max_vertices": 10_000_000,
        "max_faces": 20_000_000,
        "max_active_voxels": 50_000_000,
    }


def test_mesh_to_level_set_threads_caller_limits_into_native_tool(fake_openvdb, tetrahedron_mesh):
    limits = ExecutionLimits(max_vertices=4, max_faces=4, max_active_voxels=50_000)

    mesh_to_level_set(tetrahedron_mesh, voxel_size=0.1, limits=limits)

    _, _, kwargs = _call(fake_openvdb, "mesh_to_level_set")
    assert kwargs["max_vertices"] == 4
    assert kwargs["max_faces"] == 4
    assert kwargs["max_active_voxels"] == 50_000
    assert not any(call[0] == "create_transform" for call in fake_openvdb.calls)


def test_mesh_to_level_set_supports_stock_openvdb_fallback(stock_openvdb, tetrahedron_mesh):
    grid = mesh_to_level_set(tetrahedron_mesh, voxel_size=0.1)

    assert grid.label == "official-mesh"
    _call(stock_openvdb, "create_transform")
    _, _, kwargs = _call(stock_openvdb, "official_mesh_to_level_set")
    assert kwargs["halfWidth"] == 3.0
    assert kwargs["triangles"].dtype == np.int32


def test_mesh_to_level_set_revalidates_mutated_mesh_before_stock_fallback(
    stock_openvdb, tetrahedron_mesh
):
    tetrahedron_mesh.triangles[0, 2] = len(tetrahedron_mesh.vertices)

    with pytest.raises(InvalidGeometryError, match="outside the vertex array"):
        mesh_to_level_set(tetrahedron_mesh, voxel_size=0.1)

    assert not any(call[0] == "official_mesh_to_level_set" for call in stock_openvdb.calls)


def test_mesh_snapshot_uses_one_bounded_copy_boundary(tetrahedron_mesh):
    snapshot = mesh_module._snapshot_mesh(tetrahedron_mesh, ExecutionLimits())

    assert not np.shares_memory(snapshot.vertices, tetrahedron_mesh.vertices)
    assert not np.shares_memory(snapshot.triangles, tetrahedron_mesh.triangles)
    assert not np.shares_memory(snapshot.quads, tetrahedron_mesh.quads)


def test_volume_to_mesh_returns_owned_validated_mesh(fake_openvdb):
    mesh = volume_to_mesh(fake_openvdb.FloatGrid(), isovalue=0.25, adaptivity=0.5, thread_count=2)

    assert mesh.vertices.dtype == np.float32
    assert mesh.face_count == 4
    _, _, kwargs = _call(fake_openvdb, "volume_to_mesh")
    assert kwargs == {
        "isovalue": 0.25,
        "adaptivity": 0.5,
        "relax_disoriented_triangles": True,
        "thread_count": 2,
        "max_vertices": 10_000_000,
        "max_faces": 20_000_000,
        "max_active_voxels": 50_000_000,
    }


def test_volume_to_mesh_threads_caller_limits_into_native_tool(fake_openvdb):
    limits = ExecutionLimits(max_vertices=4, max_faces=4, max_active_voxels=8)

    volume_to_mesh(fake_openvdb.FloatGrid(active_voxels=8), limits=limits)

    _, _, kwargs = _call(fake_openvdb, "volume_to_mesh")
    assert kwargs["max_vertices"] == 4
    assert kwargs["max_faces"] == 4
    assert kwargs["max_active_voxels"] == 8


@pytest.mark.parametrize("active_voxels", [True, 1.5, "4", object()])
def test_output_limit_rejects_a_malformed_active_voxel_count(active_voxels: object) -> None:
    grid = type("MalformedGrid", (), {"activeVoxelCount": lambda _self: active_voxels})()

    with pytest.raises(InvalidGeometryError, match="malformed active-voxel count"):
        mesh_module._check_output_limit(grid, ExecutionLimits())


def test_mesh_to_level_set_enforces_caller_field_memory_limit(
    fake_openvdb, tetrahedron_mesh
) -> None:
    fake_openvdb.tools.mesh_to_level_set = lambda *_args, **_kwargs: fake_openvdb.FloatGrid(
        memory_bytes=9
    )

    with pytest.raises(ResourceLimitError, match="VDB field requires 9 bytes; limit is 8"):
        mesh_to_level_set(
            tetrahedron_mesh,
            voxel_size=0.1,
            limits=ExecutionLimits(max_field_memory_bytes=8),
        )


@pytest.mark.parametrize("memory_usage", [True, -1, object()])
def test_output_limit_rejects_malformed_field_memory_usage(memory_usage: object) -> None:
    class MalformedMemoryGrid:
        def activeVoxelCount(self) -> int:
            return 0

        def evalActiveVoxelDim(self) -> tuple[int, int, int]:
            return (0, 0, 0)

        def memUsage(self) -> object:
            return memory_usage

    with pytest.raises(InvalidGeometryError, match="malformed memory usage"):
        mesh_module._check_output_limit(MalformedMemoryGrid(), ExecutionLimits())


def test_volume_to_mesh_supports_stock_grid_method(stock_openvdb):
    grid = stock_openvdb.FloatGrid()

    mesh = volume_to_mesh(grid)

    assert mesh.face_count == 4


def test_volume_to_mesh_exposes_disoriented_triangle_policy(fake_openvdb):
    volume_to_mesh(fake_openvdb.FloatGrid(), relax_disoriented_triangles=False)

    _, _, kwargs = _call(fake_openvdb, "volume_to_mesh")
    assert kwargs["relax_disoriented_triangles"] is False


def test_volume_to_mesh_keeps_tools_v1_call_compatible(fake_openvdb):
    del fake_openvdb.tools.API_VERSION

    mesh = volume_to_mesh(fake_openvdb.FloatGrid())

    assert mesh.face_count == 4
    _, _, kwargs = _call(fake_openvdb, "volume_to_mesh")
    assert kwargs == {"isovalue": 0.0, "adaptivity": 0.0, "thread_count": 1}


def test_tools_v1_volume_mesher_rejects_unavailable_extended_control(fake_openvdb):
    del fake_openvdb.tools.API_VERSION

    with pytest.raises(CapabilityUnavailableError, match="extended_volume_to_mesh"):
        volume_to_mesh(fake_openvdb.FloatGrid(), relax_disoriented_triangles=False)


def test_stock_volume_mesher_rejects_unavailable_extended_control(stock_openvdb):
    with pytest.raises(CapabilityUnavailableError, match="extended_volume_to_mesh"):
        volume_to_mesh(
            stock_openvdb.FloatGrid(),
            relax_disoriented_triangles=False,
        )


def test_mesh_to_unsigned_distance_field_is_bounded(fake_openvdb, tetrahedron_mesh):
    grid = mesh_to_unsigned_distance_field(
        tetrahedron_mesh,
        voxel_size=0.05,
        half_width=5.0,
        name="collision-distance",
        thread_count=3,
    )

    assert grid.label == "unsigned"
    assert grid.name == "collision-distance"
    _, args, kwargs = _call(fake_openvdb, "mesh_to_unsigned_distance_field")
    assert args[0].dtype == np.float32
    assert args[1].dtype == np.int32
    assert kwargs == {
        "voxel_size": 0.05,
        "half_width": 5.0,
        "thread_count": 3,
        "max_vertices": 10_000_000,
        "max_faces": 20_000_000,
        "max_active_voxels": 50_000_000,
    }


def test_unsigned_distance_requires_repository_tool(stock_openvdb, tetrahedron_mesh):
    with pytest.raises(CapabilityUnavailableError, match="mesh_to_unsigned_distance_field"):
        mesh_to_unsigned_distance_field(tetrahedron_mesh, voxel_size=0.1)


@pytest.mark.parametrize("thread_count", [True, 0, -1, 1.5])
def test_thread_count_is_a_positive_integer(fake_openvdb, tetrahedron_mesh, thread_count):
    with pytest.raises(ValueError, match="positive integer"):
        mesh_to_level_set(tetrahedron_mesh, voxel_size=0.1, thread_count=thread_count)


@pytest.mark.parametrize("voxel_size", [0.0, -1.0, float("nan"), float("inf")])
def test_voxel_size_must_be_finite_and_positive(fake_openvdb, tetrahedron_mesh, voxel_size):
    with pytest.raises(InvalidGeometryError, match="voxel_size"):
        mesh_to_level_set(tetrahedron_mesh, voxel_size=voxel_size)


@pytest.mark.parametrize("voxel_size", [True, np.bool_(False), "0.1"])
def test_voxel_size_must_be_real_not_boolean(fake_openvdb, tetrahedron_mesh, voxel_size):
    with pytest.raises(TypeError, match="voxel_size must be a real number"):
        mesh_to_level_set(tetrahedron_mesh, voxel_size=voxel_size)


@pytest.mark.parametrize(
    ("kwargs", "label"),
    [
        ({"half_width": True}, "half_width"),
        ({"isovalue": False}, "isovalue"),
        ({"adaptivity": np.bool_(True)}, "adaptivity"),
    ],
)
def test_mesh_numeric_controls_reject_booleans(fake_openvdb, tetrahedron_mesh, kwargs, label):
    operation = mesh_to_level_set if "half_width" in kwargs else volume_to_mesh
    args = (tetrahedron_mesh,) if operation is mesh_to_level_set else (object(),)
    call_kwargs = {"voxel_size": 0.1, **kwargs} if operation is mesh_to_level_set else kwargs
    with pytest.raises(TypeError, match=rf"{label} must be a real number"):
        operation(*args, **call_kwargs)


def test_mesh_limits_are_checked_before_snapshot_and_native_call(
    fake_openvdb, tetrahedron_mesh, monkeypatch
):
    monkeypatch.setattr(
        mesh_module,
        "_snapshot_mesh",
        lambda *_args, **_kwargs: pytest.fail("oversized mesh must not be copied"),
    )
    with pytest.raises(ResourceLimitError, match="vertices"):
        mesh_to_level_set(
            tetrahedron_mesh,
            voxel_size=0.1,
            limits=ExecutionLimits(max_vertices=3),
        )

    assert not any(call[0] == "mesh_to_level_set" for call in fake_openvdb.calls)


def test_mesh_absolute_voxel_coordinates_are_checked_before_native_call(
    fake_openvdb, tetrahedron_mesh
):
    far_mesh = Mesh(
        vertices=tetrahedron_mesh.vertices * np.float32(2048) + np.float32(1e10),
        triangles=tetrahedron_mesh.triangles,
    )

    with pytest.raises(ResourceLimitError, match="int32 voxel range"):
        mesh_to_level_set(far_mesh, voxel_size=1.0)

    assert not any(call[0] == "mesh_to_level_set" for call in fake_openvdb.calls)


def test_mesh_dense_voxel_budget_is_checked_before_native_call(fake_openvdb, tetrahedron_mesh):
    with pytest.raises(ResourceLimitError, match="active voxels limit"):
        mesh_to_level_set(
            tetrahedron_mesh,
            voxel_size=0.1,
            limits=ExecutionLimits(max_active_voxels=1_000),
        )

    assert not any(call[0] == "mesh_to_level_set" for call in fake_openvdb.calls)


def test_mesh_band_float32_range_is_checked_before_native_call(fake_openvdb, tetrahedron_mesh):
    with pytest.raises(ResourceLimitError, match="float32 background range"):
        mesh_to_level_set(
            tetrahedron_mesh,
            voxel_size=1e38,
            half_width=4.0,
        )

    assert not any(call[0] == "mesh_to_level_set" for call in fake_openvdb.calls)


def test_thread_limit_is_checked_before_native_call(fake_openvdb, tetrahedron_mesh):
    with pytest.raises(ResourceLimitError, match="thread_count"):
        mesh_to_level_set(
            tetrahedron_mesh,
            voxel_size=0.1,
            thread_count=5,
            limits=ExecutionLimits(max_threads=4),
        )

    assert not any(call[0] == "mesh_to_level_set" for call in fake_openvdb.calls)


def test_active_voxel_limit_is_checked_after_native_call(fake_openvdb, tetrahedron_mesh):
    with pytest.raises(ResourceLimitError, match="active voxels"):
        mesh_to_level_set(
            tetrahedron_mesh,
            voxel_size=0.1,
            limits=ExecutionLimits(max_active_voxels=11),
        )


def test_band_width_limit_is_checked_before_native_call(fake_openvdb, tetrahedron_mesh):
    with pytest.raises(ResourceLimitError, match="half_width"):
        mesh_to_unsigned_distance_field(
            tetrahedron_mesh,
            voxel_size=0.1,
            half_width=9.0,
            limits=ExecutionLimits(max_band_width_voxels=8.0),
        )

    assert not any(call[0] == "mesh_to_unsigned_distance_field" for call in fake_openvdb.calls)
