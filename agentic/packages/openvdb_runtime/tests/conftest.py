# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


class FakeGrid:
    def __init__(
        self,
        label: str = "grid",
        active_voxels: int = 8,
        active_dimensions: tuple[int, int, int] = (4, 4, 4),
        memory_bytes: int = 4_096,
    ) -> None:
        self.label = label
        self.name = ""
        self._active_voxels = active_voxels
        self._active_dimensions = active_dimensions
        self._memory_bytes = memory_bytes
        self.metadata = {
            "file_voxel_count": active_voxels,
            "file_mem_bytes": memory_bytes,
        }

    def activeVoxelCount(self) -> int:
        return self._active_voxels

    def evalActiveVoxelDim(self) -> tuple[int, int, int]:
        return self._active_dimensions

    def memUsage(self) -> int:
        return self._memory_bytes

    def convertToPolygons(self, *, isovalue: float, adaptivity: float):
        return _tetrahedron_arrays()


def _tetrahedron_arrays():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    triangles = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32)
    quads = np.empty((0, 4), dtype=np.int32)
    return vertices, triangles, quads


@pytest.fixture
def fake_openvdb(monkeypatch):
    from openvdb_runtime import runtime

    policy = runtime._load_policy().copy()
    policy["required_core_capabilities"] = []
    policy["required_repository_capabilities"] = []
    monkeypatch.setattr(runtime, "_load_policy", lambda: policy)
    for name in tuple(sys.modules):
        if name == "openvdb" or name.startswith("openvdb."):
            monkeypatch.delitem(sys.modules, name)
    calls: list[tuple[str, tuple, dict]] = []
    module = ModuleType("openvdb")
    module.LIBRARY_VERSION = (13, 0, 0)
    module.FILE_FORMAT_VERSION = 224
    module.calls = calls

    class FloatGrid(FakeGrid):
        @staticmethod
        def copyFromArray(*args, **kwargs):
            return None

        @staticmethod
        def copyToArray(*args, **kwargs):
            return None

        @classmethod
        def createLevelSetFromPolygons(cls, points, **kwargs):
            calls.append(("official_mesh_to_level_set", (points,), kwargs))
            return cls("official-mesh")

    module.FloatGrid = FloatGrid

    def record(name, result):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result() if callable(result) else result

        return call

    module.createLinearTransform = record("create_transform", lambda: object())
    module.read = record("read", lambda: FakeGrid("read"))
    module.readAll = record("read_all", lambda: ([FakeGrid("all")], {"creator": "test"}))
    module.readGridMetadata = record("read_grid_metadata", lambda: FakeGrid("metadata"))
    module.readAllGridMetadata = record(
        "read_all_grid_metadata", lambda: [FakeGrid("all-metadata")]
    )

    def write(path, *args, **kwargs):
        calls.append(("write", (path, *args), kwargs))
        Path(path).write_bytes(b"vdb")

    module.write = write
    module.tools = SimpleNamespace(
        API_VERSION=3,
        mesh_to_level_set=record("mesh_to_level_set", lambda: FakeGrid("mesh", 12)),
        volume_to_mesh=record("volume_to_mesh", _tetrahedron_arrays),
        csg_union=record("csg_union", lambda: FakeGrid("union")),
        csg_intersection=record("csg_intersection", lambda: FakeGrid("intersection")),
        csg_difference=record("csg_difference", lambda: FakeGrid("difference")),
        level_set_offset=record("level_set_offset", lambda: FakeGrid("offset")),
        level_set_mean=record("level_set_mean", lambda: FakeGrid("mean")),
        scalar_mean=record("scalar_mean", lambda: FakeGrid("scalar-mean")),
        level_set_normalize=record("level_set_normalize", lambda: FakeGrid("normalized")),
        level_set_rebuild=record("level_set_rebuild", lambda: FakeGrid("rebuilt")),
        resample_to_match=record("resample_to_match", lambda: FakeGrid("resampled")),
        sample_values=record("sample_values", lambda: np.array([-0.5, 0.25], dtype=np.float32)),
        sample_gradients=record(
            "sample_gradients",
            lambda: np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        ),
        mesh_to_unsigned_distance_field=record(
            "mesh_to_unsigned_distance_field", lambda: FakeGrid("unsigned", 10)
        ),
        active_value_mask=record("active_value_mask", lambda: FakeGrid("mask", 6)),
        topology_to_level_set=record(
            "topology_to_level_set", lambda: FakeGrid("topology-level-set", 9)
        ),
        extract_enclosed_region=record("extract_enclosed_region", lambda: FakeGrid("enclosed", 5)),
    )
    monkeypatch.setitem(sys.modules, "openvdb", module)
    # Production code has no unverified-module switch. Tests install their fake
    # directly into the private, process-local successful-admission cache.
    monkeypatch.setattr(runtime, "_ADMITTED_MODULE", module)
    monkeypatch.setattr(runtime, "_ADMITTED_SYS_MODULES", {"openvdb": module})
    monkeypatch.setattr(runtime, "_ADMITTED_DISTRIBUTION_VERSION", None)
    monkeypatch.setattr(runtime, "_revalidate_cached_admission", lambda: None)
    monkeypatch.setattr(runtime, "_deep_revalidate_admission", lambda: None)
    return module


@pytest.fixture
def stock_openvdb(fake_openvdb):
    del fake_openvdb.tools
    return fake_openvdb


@pytest.fixture
def tetrahedron_mesh():
    from openvdb_runtime import Mesh

    vertices, triangles, quads = _tetrahedron_arrays()
    return Mesh(vertices=vertices, triangles=triangles, quads=quads)
