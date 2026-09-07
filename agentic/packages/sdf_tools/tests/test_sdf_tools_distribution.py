# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import sdf_tools

PACKAGE_ROOT = Path(sdf_tools.__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DRIVER_PROJECT_ROOT = PROJECT_ROOT.parent / "openvdb_runtime"
DRIVER_PACKAGE_ROOT = DRIVER_PROJECT_ROOT / "openvdb_runtime"
REPO_ROOT = PROJECT_ROOT.parents[2]
REPOSITORY_VERSION = (REPO_ROOT / "VERSION.md").read_text(encoding="utf-8").strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _driver_license_manifest():
    driver_root = str(DRIVER_PROJECT_ROOT)
    preloaded = frozenset(sys.modules)
    sys.path.insert(0, driver_root)
    try:
        return importlib.import_module("openvdb_runtime.sdf_backend")._LICENSE_MANIFEST
    finally:
        sys.path.remove(driver_root)
        for name in tuple(sys.modules):
            if name not in preloaded and (
                name == "openvdb_runtime" or name.startswith("openvdb_runtime.")
            ):
                sys.modules.pop(name, None)


def test_core_import_does_not_load_a_concrete_backend() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; import sdf_tools; "
                "assert 'openvdb_runtime' not in sys.modules; "
                "assert 'openvdb' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_driver_manifest_helper_does_not_leak_runtime_modules() -> None:
    preloaded = {
        name: module
        for name, module in sys.modules.items()
        if name == "openvdb_runtime" or name.startswith("openvdb_runtime.")
    }

    _driver_license_manifest()

    assert {
        name: module
        for name, module in sys.modules.items()
        if name == "openvdb_runtime" or name.startswith("openvdb_runtime.")
    } == preloaded


def test_distribution_is_dependency_light_and_typed() -> None:
    configuration = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = configuration["project"]
    build = configuration["tool"]["hatch"]["build"]["targets"]
    assert project["name"] == "sdf-tools"
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert project["dependencies"] == []
    assert build["wheel"]["force-include"] == {"LICENSE": "sdf_tools/LICENSE"}
    assert build["sdist"]["include"] == [
        "/LICENSE",
        "/README.md",
        "/pyproject.toml",
        "/sdf_tools",
        "/tests",
        "/uv.lock",
    ]
    assert (PROJECT_ROOT / "LICENSE").read_bytes() == (REPO_ROOT / "LICENSE").read_bytes()
    assert (PACKAGE_ROOT / "py.typed").is_file()


def test_backend_policy_admits_exact_driver_entry_point() -> None:
    policy = json.loads((PACKAGE_ROOT / "backend_policy.json").read_text(encoding="utf-8"))
    assert policy == {
        "schema": "world-understanding.sdf-backend-policy.v1",
        "entry_point_group": "sdf_tools.backends",
        "backends": {
            "openvdb": {
                "authenticated_data_files": {
                    "openvdb_runtime/_build_manifest.json": (
                        "4bb75e47c1dfd5504191a6a5afa549f8b44774c66e85d01cf79a94d2c9cd07ba"
                    ),
                },
                "authenticated_files": {
                    "openvdb_runtime/__init__.py": (
                        "e43e05f7a297e54e33c1c5719da3af78bbf4663c7d90a65a1513b0f1fc698ef7"
                    ),
                    "openvdb_runtime/errors.py": (
                        "ac1f6e4aa58b1389e3cd6a26681a8b93390c997309807e4a3c8635858701c605"
                    ),
                    "openvdb_runtime/io.py": (
                        "cea9aae1375ae7c52824041b9e752151a096aac36685633e08e759a3819fbf17"
                    ),
                    "openvdb_runtime/level_set.py": (
                        "b1a781df0935eddc265260176a6cb5d7b0b933f31d1a4959227007d0c93b5e2d"
                    ),
                    "openvdb_runtime/mesh.py": (
                        "e0fc2f8b56871802b63dca00e01594866941239573add65ce619a9853e2a98e5"
                    ),
                    "openvdb_runtime/runtime.py": (
                        "57af6344bc29b0b6da2d7465ce663d8262049a779722077155ca589644de7f0e"
                    ),
                    "openvdb_runtime/sampling.py": (
                        "269d161c20805ece17bc1bdfd6530ccb63c66383d1dadfe73b3533aa45a703b9"
                    ),
                    "openvdb_runtime/sdf_backend.py": (
                        "888ecd21ac9abdff43cf02baa718daf2bbf9acdf8f2c29c3178e7958fa2f6db8"
                    ),
                    "openvdb_runtime/topology.py": (
                        "ba99181ebbaf8f22b4df6e410936fbac5725187488555f84ddb18406e30df69f"
                    ),
                    "openvdb_runtime/types.py": (
                        "a31c7394ec36b8a97977c732a810c809afe645f66b5c84ecaff42cc3361fbaeb"
                    ),
                },
                "distribution": "openvdb-runtime",
                "distribution_version": REPOSITORY_VERSION,
                "entry_point": "openvdb_runtime.sdf_backend:sdf_backend_extension",
                "entry_point_module_path": "openvdb_runtime/sdf_backend.py",
                "entry_point_module_sha256": (
                    "888ecd21ac9abdff43cf02baa718daf2bbf9acdf8f2c29c3178e7958fa2f6db8"
                ),
                "implementation_version": "openvdb-runtime-13.0.0+wu.3",
                "license_manifest_sha256": (
                    "1b3402e56da598b491f249c6a06179a1d7747da5362ce3a482ed4c146b93a4d2"
                ),
                "native_closure_attestation": (
                    "sha256:8e0cb662ca1a9865a195d8fc6a7c64224fd8edeebfed9c266417a3afc7d3a59f"
                ),
                "production_qualified": True,
            }
        },
    }


def test_backend_policy_binds_exact_driver_files() -> None:
    policy = json.loads((PACKAGE_ROOT / "backend_policy.json").read_text(encoding="utf-8"))
    admission = policy["backends"]["openvdb"]

    actual_python_files = {
        f"openvdb_runtime/{path.relative_to(DRIVER_PACKAGE_ROOT).as_posix()}"
        for path in DRIVER_PACKAGE_ROOT.rglob("*.py")
    }
    assert set(admission["authenticated_files"]) == actual_python_files
    for relative_path, expected_sha256 in {
        **admission["authenticated_files"],
        **admission["authenticated_data_files"],
    }.items():
        assert expected_sha256 == _sha256(DRIVER_PROJECT_ROOT / relative_path)

    entry_point_path = admission["entry_point_module_path"]
    assert (
        admission["entry_point_module_sha256"] == admission["authenticated_files"][entry_point_path]
    )


def test_backend_policy_binds_promoted_native_closure() -> None:
    policy = json.loads((PACKAGE_ROOT / "backend_policy.json").read_text(encoding="utf-8"))
    admission = policy["backends"]["openvdb"]
    runtime_policy = json.loads(
        (DRIVER_PACKAGE_ROOT / "_build_manifest.json").read_text(encoding="utf-8")
    )
    release_lock_sha256 = _sha256(DRIVER_PROJECT_ROOT / "native" / "release-lock.json")
    expected_attestation = f"sha256:{release_lock_sha256}"
    assert runtime_policy["openvdb_release_lock_sha256"] == release_lock_sha256
    assert admission["native_closure_attestation"] == expected_attestation

    license_manifest = _driver_license_manifest()
    assert license_manifest.native_closure_attestation == expected_attestation
    assert admission["license_manifest_sha256"] == license_manifest.sha256


def test_backend_license_manifest_tracks_repository_version() -> None:
    manifest = _driver_license_manifest()
    components = {component.name: component for component in manifest.components}

    assert components["sdf-tools"].version == REPOSITORY_VERSION
    assert components["openvdb-runtime"].version == REPOSITORY_VERSION


def test_public_python_surface_does_not_import_or_type_against_openvdb() -> None:
    for path in PACKAGE_ROOT.glob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        assert "import openvdb" not in source
        assert "openvdb_runtime" not in source
        assert "subprocess" not in source
