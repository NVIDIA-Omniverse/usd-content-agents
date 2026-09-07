# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tomllib
import typing
from pathlib import Path

import openvdb_runtime

PACKAGE_ROOT = Path(openvdb_runtime.__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parents[2]
NATIVE_TOOLS_SOURCE = PROJECT_ROOT / "native/overlay/openvdb/openvdb/python/pyTools.cc"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_package_metadata_matches_runtime_contract():
    configuration = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = configuration["project"]
    build = configuration["tool"]["hatch"]["build"]["targets"]

    assert project["name"] == "openvdb-runtime"
    assert project["requires-python"] == ">=3.12,<3.13"
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert project["dependencies"] == ["sdf-tools"]
    assert build["wheel"]["force-include"] == {
        "LICENSE": "openvdb_runtime/LICENSE",
        "native/release-lock.json": "openvdb_runtime/_release_lock.json",
    }
    assert build["sdist"]["include"] == [
        "/LICENSE",
        "/README.md",
        "/native",
        "/openvdb_runtime",
        "/pyproject.toml",
        "/tests",
        "/uv.lock",
    ]
    assert (PROJECT_ROOT / "LICENSE").read_bytes() == (REPO_ROOT / "LICENSE").read_bytes()


def test_package_and_sdf_entry_point_defer_numerical_runtime_imports():
    package_tree = ast.parse((PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8"))
    eager_relative_imports = [
        node for node in package_tree.body if isinstance(node, ast.ImportFrom) and node.level
    ]
    assert eager_relative_imports == []

    backend_tree = ast.parse((PACKAGE_ROOT / "sdf_backend.py").read_text(encoding="utf-8"))
    eager_modules = {
        alias.name
        for node in backend_tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "numpy" not in eager_modules
    assert "openvdb_runtime" not in eager_modules


def test_clean_entry_point_import_does_not_load_numpy():
    script = """
import sys
import openvdb_runtime
assert "numpy" not in sys.modules
import openvdb_runtime.sdf_backend
assert "numpy" not in sys.modules
import openvdb_runtime.types
import openvdb_runtime.mesh
import openvdb_runtime.sampling
assert "numpy" not in sys.modules
import typing
typing.get_type_hints(openvdb_runtime.Mesh)
typing.get_type_hints(openvdb_runtime.sample_values)
assert "numpy" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_missing_application_numpy_is_reported_at_the_numerical_boundary():
    script = """
import sys

class BlockNumpy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("NumPy blocked for isolation test", name="numpy")
        return None

sys.meta_path.insert(0, BlockNumpy())
from openvdb_runtime import Mesh, RuntimeUnavailableError
assert "numpy" not in sys.modules
try:
    Mesh(vertices=[(0, 0, 0)], triangles=[(0, 0, 0)])
except RuntimeUnavailableError as exc:
    assert "NumPy supplied by the application" in str(exc)
else:
    raise AssertionError("a numerical operation unexpectedly succeeded without NumPy")
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_deferred_array_annotations_support_runtime_type_hint_resolution():
    mesh_hints = typing.get_type_hints(openvdb_runtime.Mesh)
    sample_hints = typing.get_type_hints(openvdb_runtime.sample_values)

    assert set(mesh_hints) == {"vertices", "triangles", "quads"}
    assert sample_hints["return"] is typing.Any


def test_packaged_policy_pins_openvdb_13_source():
    policy = json.loads((PACKAGE_ROOT / "_build_manifest.json").read_text(encoding="utf-8"))

    assert policy == {
        "schema": "world-understanding.openvdb-runtime-policy.v1",
        "openvdb_distribution": "openvdb",
        "openvdb_distribution_version": "13.0.0+wu.3",
        "openvdb_library_version": [13, 0, 0],
        "openvdb_source_commit": "7c03e1f084873cd1b3422c7ff7aec6ee681b3b38",
        "openvdb_source_lock_sha256": (
            "de73f6d8c34b7dc838e2754e08276021b67a101ce47bdd24d2f60a133ea312b6"
        ),
        "openvdb_release_lock_sha256": (
            "8e0cb662ca1a9865a195d8fc6a7c64224fd8edeebfed9c266417a3afc7d3a59f"
        ),
        "required_core_capabilities": [
            "transforms",
            "vdb_io",
            "mesh_to_level_set",
            "volume_to_mesh",
        ],
        "required_repository_capabilities": [
            "active_value_mask",
            "csg",
            "extended_volume_to_mesh",
            "extract_enclosed_region",
            "level_set_filter",
            "level_set_normalize",
            "level_set_offset",
            "level_set_rebuild",
            "mesh_to_unsigned_distance_field",
            "resample_to_match",
            "sample_gradients",
            "sample_values",
            "scalar_mean_filter",
            "topology_to_level_set",
        ],
    }
    assert (PACKAGE_ROOT / "py.typed").is_file()


def test_packaged_policy_binds_exact_repository_native_locks() -> None:
    policy = json.loads((PACKAGE_ROOT / "_build_manifest.json").read_text(encoding="utf-8"))

    assert policy["openvdb_source_lock_sha256"] == _sha256(
        PROJECT_ROOT / "native" / "source-lock.json"
    )
    assert policy["openvdb_release_lock_sha256"] == _sha256(
        PROJECT_ROOT / "native" / "release-lock.json"
    )


def test_python_facade_contains_no_process_transport():
    for path in PACKAGE_ROOT.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import subprocess" not in source
        assert "from subprocess" not in source


def test_native_gradient_sampling_bounds_the_derived_grid_memory() -> None:
    source = NATIVE_TOOLS_SOURCE.read_text(encoding="utf-8")
    preflight = source.index("validateGradientGridInputBudget(*snapshot)")
    build = source.index("tools::gradient(grid, true)")
    actual_grid_check = source.index(
        'validateGridDomain(*gradientGrid, 0, "derived gradient grid")', build
    )

    assert "grid.memUsage() > MAX_GRID_MEMORY_BYTES / 3" in source
    assert preflight < source.index("nb::gil_scoped_release release", preflight)
    assert build < actual_grid_check
