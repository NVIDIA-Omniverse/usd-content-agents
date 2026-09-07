# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

import openvdb_runtime.types as runtime_types
from openvdb_runtime import ExecutionLimits, InvalidGeometryError, Mesh


def test_mesh_owns_canonical_contiguous_arrays():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = Mesh(
        vertices=vertices,
        triangles=triangles,
    )

    assert mesh.vertices.dtype == np.float32
    assert mesh.triangles.dtype == np.int32
    assert mesh.vertices.flags.c_contiguous
    assert mesh.triangles.flags.c_contiguous
    assert mesh.vertices.flags.aligned and mesh.vertices.flags.owndata
    assert mesh.triangles.flags.aligned and mesh.triangles.flags.owndata
    assert not np.shares_memory(mesh.vertices, vertices)
    assert not np.shares_memory(mesh.triangles, triangles)

    vertices[:] = 9
    triangles[:] = 0
    np.testing.assert_array_equal(mesh.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(mesh.triangles, [[0, 1, 2]])


def test_mesh_copies_unaligned_input_buffers():
    vertex_storage = np.zeros(3 * 3 * 4 + 1, dtype=np.uint8)
    vertices = np.ndarray((3, 3), dtype=np.float32, buffer=vertex_storage, offset=1)
    vertices[:] = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
    triangle_storage = np.zeros(3 * 4 + 1, dtype=np.uint8)
    triangles = np.ndarray((1, 3), dtype=np.int32, buffer=triangle_storage, offset=1)
    triangles[:] = ((0, 1, 2),)
    assert not vertices.flags.aligned and not triangles.flags.aligned

    mesh = Mesh(vertices=vertices, triangles=triangles)

    assert mesh.vertices.flags.aligned and mesh.vertices.flags.owndata
    assert mesh.triangles.flags.aligned and mesh.triangles.flags.owndata
    assert not np.shares_memory(mesh.vertices, vertices)
    assert not np.shares_memory(mesh.triangles, triangles)


def test_mesh_constructors_share_validation_helper(monkeypatch):
    calls: list[tuple[int, int]] = []
    original_validate = runtime_types._validated_mesh_arrays

    def validate(vertices, triangles, quads, *, max_vertices, max_faces):
        calls.append((max_vertices, max_faces))
        return original_validate(
            vertices,
            triangles,
            quads,
            max_vertices=max_vertices,
            max_faces=max_faces,
        )

    monkeypatch.setattr(runtime_types, "_validated_mesh_arrays", validate)
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    triangles = [[0, 1, 2]]

    Mesh(vertices=vertices, triangles=triangles)
    Mesh._from_bounded_buffers(
        vertices=vertices,
        triangles=triangles,
        max_vertices=3,
        max_faces=1,
    )

    assert calls == [
        (runtime_types.DEFAULT_LIMITS.max_vertices, runtime_types.DEFAULT_LIMITS.max_faces),
        (3, 1),
    ]


def test_mesh_triangulates_quads_deterministically():
    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32),
        quads=np.array([[0, 1, 2, 3]], dtype=np.int32),
    )

    np.testing.assert_array_equal(mesh.triangulated_faces(), [[0, 1, 2], [0, 2, 3]])


@pytest.mark.parametrize(
    ("vertices", "triangles", "message"),
    [
        ([[0, 0]], [[0, 0, 0]], "vertices"),
        ([[0, 0, float("nan")]], [[0, 0, 0]], "finite"),
        (
            np.asarray([[0.0 + 1.0j, 0.0, 0.0]]),
            [[0, 0, 0]],
            "real numeric",
        ),
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 3]], "outside"),
        ([[0, 0, 0]], [[0.0, 0.0, 0.0]], "integer"),
        ([[0, 0, 0]], [[-1, 0, 0]], "nonnegative"),
        ([[0, 0, 0]], [[0, 0, 0]], "repeat a vertex"),
    ],
)
def test_mesh_rejects_malformed_arrays(vertices, triangles, message):
    with pytest.raises(InvalidGeometryError, match=message):
        Mesh(vertices=np.asarray(vertices), triangles=np.asarray(triangles))


def test_mesh_rejects_repeated_quad_indices():
    with pytest.raises(InvalidGeometryError, match="repeat a vertex"):
        Mesh(
            vertices=np.asarray(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                dtype=np.float32,
            ),
            quads=np.asarray([[0, 1, 2, 0]], dtype=np.int32),
        )


def test_execution_limits_must_be_positive():
    with pytest.raises(ValueError, match="max_faces"):
        ExecutionLimits(max_faces=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_faces": True},
        {"max_threads": 1.5},
        {"max_filter_work": False},
        {"max_metadata_entries": 1.5},
        {"max_metadata_bytes": True},
        {"max_band_width_voxels": False},
        {"max_band_width_voxels": "3"},
    ],
)
def test_execution_limits_reject_wrong_numeric_types(kwargs):
    with pytest.raises(TypeError):
        ExecutionLimits(**kwargs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0])
def test_execution_limits_require_finite_positive_band_width(value):
    with pytest.raises(ValueError, match="finite and positive"):
        ExecutionLimits(max_band_width_voxels=value)
