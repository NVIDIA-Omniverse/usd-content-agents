# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from typing import NoReturn

import numpy as np
import sdf_tools

from openvdb_runtime import mean_filter, mesh_to_level_set, sample_values, sdf_backend


def _reject_process(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("in-process SDF execution attempted to create a process")


def test_geometry_operations_do_not_create_a_process(fake_openvdb, tetrahedron_mesh, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _reject_process)
    monkeypatch.setattr(subprocess, "Popen", _reject_process)
    grid = mesh_to_level_set(tetrahedron_mesh, voxel_size=0.1)
    mean_filter(grid)
    sample_values(grid, np.zeros((2, 3), dtype=np.float32))


def test_sdf_tools_openvdb_operations_do_not_create_a_process(fake_openvdb, monkeypatch):
    def inspect(_self: sdf_backend.OpenVdbSdfBackend) -> sdf_tools.BackendInfo:
        descriptor = sdf_backend.OPENVDB_SDF_BACKEND
        return sdf_tools.BackendInfo(
            backend_id=descriptor.backend_id,
            implementation_version=descriptor.implementation_version,
            operations=descriptor.operations,
            execution_mode=descriptor.execution_mode,
            read_formats=descriptor.read_formats,
            write_formats=descriptor.write_formats,
            provenance={"runtime": "fake-openvdb-13"},
        )

    monkeypatch.setattr(sdf_backend.OpenVdbSdfBackend, "inspect", inspect)
    monkeypatch.setattr(subprocess, "run", _reject_process)
    monkeypatch.setattr(subprocess, "Popen", _reject_process)

    extension = sdf_backend.sdf_backend_extension()
    sdf_tools.validate_license_manifest(extension.descriptor.license_manifest)
    registry = sdf_tools.SdfBackendRegistry()
    registry._extensions[extension.descriptor.backend_id] = extension  # type: ignore[attr-defined]
    toolkit = sdf_tools.SdfToolkit(
        registry=registry,
        discover_installed=False,
    )
    session = toolkit.create_session(
        backend="openvdb",
        require={
            sdf_tools.Operation.FIELD_TO_MESH,
            sdf_tools.Operation.MESH_TO_SDF,
            sdf_tools.Operation.SAMPLE_VALUES,
            sdf_tools.Operation.SMOOTH_SDF,
        },
    )
    mesh = sdf_tools.Mesh(
        vertices=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float32,
        ),
        triangles=np.array(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
            dtype=np.int32,
        ),
    )

    field = session.mesh_to_sdf(mesh, voxel_size=0.1, thread_count=1)
    smoothed = session.smooth(field, width=1, iterations=1, thread_count=1)
    samples = session.sample_values(
        smoothed,
        np.zeros((2, 3), dtype=np.float32),
        thread_count=1,
    )
    result = session.field_to_mesh(smoothed, adaptivity=0.0, thread_count=1)

    assert np.asarray(samples).shape == (2,)
    assert result.face_count == 4
