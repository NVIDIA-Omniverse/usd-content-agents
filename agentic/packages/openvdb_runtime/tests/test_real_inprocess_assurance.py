# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys
import textwrap

import pytest

_SESSION_PRELUDE = textwrap.dedent(
    """
    import importlib.metadata
    import re

    import sdf_tools

    _EXPECTED_VERSION = importlib.metadata.version("openvdb")
    _SHA256 = re.compile(r"[0-9a-f]{64}\\Z")
    _ROUNDTRIP_OPERATIONS = frozenset(
        {
            sdf_tools.Operation.FIELD_TO_MESH,
            sdf_tools.Operation.MESH_TO_SDF,
            sdf_tools.Operation.SAMPLE_VALUES,
        }
    )
    _BOUNDED_LIMITS = sdf_tools.ExecutionLimits(
        max_vertices=100_000,
        max_faces=200_000,
        max_active_voxels=1_000_000,
        max_voxel_extent=128,
        max_threads=1,
        max_sample_points=32,
        max_topology_steps=8,
        max_band_width_voxels=8.0,
        max_filter_width=4,
        max_filter_iterations=4,
        max_filter_work=4_000_000,
        max_file_bytes=1_048_576,
        max_fields=8,
        max_field_memory_bytes=64 * 1_048_576,
        max_total_field_memory_bytes=128 * 1_048_576,
    )

    def _create_session(*, require=_ROUNDTRIP_OPERATIONS):
        session = sdf_tools.SdfToolkit().create_session(
            backend="openvdb",
            require=require,
            limits=_BOUNDED_LIMITS,
        )
        info = session.backend_info
        assert info.backend_id == "openvdb"
        assert info.execution_mode == "in_process"
        assert info.implementation_version == f"openvdb-runtime-{_EXPECTED_VERSION}"
        assert info.provenance["library_version"] == (13, 0, 0)
        assert info.provenance["distribution_version"] == _EXPECTED_VERSION
        assert info.provenance["source_distribution_version"] == _EXPECTED_VERSION
        for key in ("module_sha256", "source_lock_sha256"):
            digest = info.provenance[key]
            assert isinstance(digest, str) and _SHA256.fullmatch(digest)
        return session
    """
)

_GEOMETRY_HELPERS = textwrap.dedent(
    """
    import numpy as np

    _SAMPLE_POINTS = np.array(
        [[0.1, 0.1, 0.1], [2.0, 2.0, 2.0]],
        dtype=np.float32,
    )

    def _tetrahedron():
        return sdf_tools.Mesh(
            vertices=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                dtype=np.float32,
            ),
            triangles=np.array(
                [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
                dtype=np.int32,
            ),
        )

    def _roundtrip(session):
        field = session.mesh_to_sdf(
            _tetrahedron(),
            voxel_size=0.1,
            half_width=3.0,
            thread_count=1,
        )
        samples = np.asarray(
            session.sample_values(field, _SAMPLE_POINTS, thread_count=1),
            dtype=np.float64,
        )
        surface = session.field_to_mesh(field, adaptivity=0.0, thread_count=1)
        vertices = np.asarray(surface.vertices)
        triangles = np.asarray(surface.triangles)
        quads = np.asarray(surface.quads)
        assert samples.shape == (2,)
        assert np.isfinite(samples).all()
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert triangles.ndim == 2 and triangles.shape[1] == 3
        assert quads.ndim == 2 and quads.shape[1] == 4
        assert len(vertices) > 0
        assert surface.face_count > 0
        assert np.isfinite(vertices).all()
        return len(vertices), surface.face_count, tuple(np.round(samples, decimals=6))
    """
)


def _require_real_distribution() -> None:
    try:
        importlib.metadata.version("openvdb")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("the repository-built OpenVDB native distribution is not installed")


def _run_clean_assurance(body: str, *, timeout: int = 120) -> None:
    _require_real_distribution()
    completed = subprocess.run(
        [sys.executable, "-I", "-c", f"{_SESSION_PRELUDE}\n{textwrap.dedent(body)}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, (
        f"isolated real-runtime assurance failed\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_real_sdf_tools_openvdb_operations_do_not_create_a_process() -> None:
    _run_clean_assurance(
        textwrap.dedent(
            """
            import os
            import subprocess

            def _reject_process(*_args, **_kwargs):
                raise AssertionError("SDF execution attempted to create a process")

            for name in ("Popen", "run", "call", "check_call", "check_output"):
                setattr(subprocess, name, _reject_process)
            for name in (
                "fork",
                "forkpty",
                "popen",
                "posix_spawn",
                "posix_spawnp",
                "spawnl",
                "spawnle",
                "spawnlp",
                "spawnlpe",
                "spawnv",
                "spawnve",
                "spawnvp",
                "spawnvpe",
                "system",
            ):
                if hasattr(os, name):
                    setattr(os, name, _reject_process)

            session = _create_session(
                require=_ROUNDTRIP_OPERATIONS | {sdf_tools.Operation.OFFSET},
            )
            """
        )
        + _GEOMETRY_HELPERS
        + textwrap.dedent(
            """
            field = session.mesh_to_sdf(
                _tetrahedron(),
                voxel_size=0.1,
                half_width=3.0,
                thread_count=1,
            )
            offset = session.offset(field, 0.02, thread_count=1)
            samples = np.asarray(session.sample_values(offset, _SAMPLE_POINTS, thread_count=1))
            surface = session.field_to_mesh(offset, adaptivity=0.0, thread_count=1)
            assert samples.shape == (2,)
            assert np.isfinite(samples).all()
            assert surface.face_count > 0
            """
        )
    )


def test_real_openvdb_repeated_create_destroy_roundtrips() -> None:
    _run_clean_assurance(
        textwrap.dedent(
            """
            first_session = _create_session()

            import gc
            """
        )
        + _GEOMETRY_HELPERS
        + textwrap.dedent(
            """
            observations = [_roundtrip(first_session)]
            del first_session
            gc.collect()
            for _ in range(7):
                session = _create_session()
                observations.append(_roundtrip(session))
                del session
                gc.collect()
            assert len(set(observations)) == 1
            """
        )
    )


def test_real_openvdb_concurrent_bounded_roundtrips() -> None:
    _run_clean_assurance(
        textwrap.dedent(
            """
            session = _create_session()

            import threading
            from concurrent.futures import ThreadPoolExecutor
            """
        )
        + _GEOMETRY_HELPERS
        + textwrap.dedent(
            """
            worker_count = 4
            barrier = threading.Barrier(worker_count)

            def execute_roundtrip(_index):
                barrier.wait(timeout=30)
                return _roundtrip(session)

            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="openvdb-sdf-assurance",
            ) as executor:
                observations = list(executor.map(execute_roundtrip, range(worker_count)))
            assert len(set(observations)) == 1
            """
        )
    )


_USD_IMPORT_ORDER_SMOKE = f"""
__USD_BEFORE__
session = _create_session()
__USD_AFTER__

{_GEOMETRY_HELPERS}

stage = Usd.Stage.CreateInMemory()
assert stage.DefinePrim("/World", "Xform")
_roundtrip(session)
assert stage.GetPrimAtPath("/World").IsValid()
"""


@pytest.mark.parametrize(
    ("usd_before", "usd_after"),
    (
        pytest.param(
            "from pxr import Usd",
            "",
            id="usd-then-authenticated-sdf",
        ),
        pytest.param(
            "",
            "from pxr import Usd",
            id="authenticated-sdf-then-usd",
        ),
    ),
)
def test_real_openvdb_usd_import_orders(usd_before: str, usd_after: str) -> None:
    try:
        usd_spec = importlib.util.find_spec("pxr.Usd")
    except ImportError:
        usd_spec = None
    if usd_spec is None:
        pytest.skip("USD Python bindings are not installed")
    smoke = _USD_IMPORT_ORDER_SMOKE.replace("__USD_BEFORE__", usd_before).replace(
        "__USD_AFTER__", usd_after
    )
    _run_clean_assurance(smoke)
