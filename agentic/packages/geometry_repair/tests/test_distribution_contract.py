# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distribution metadata and required notice-file coverage."""

from __future__ import annotations

import tomllib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[2]
REQUIRED_WHEEL_FILES = {
    "LICENSE": "geometry_repair/LICENSE",
    "geometry_repair/DEPENDENCY_AUDIT.md": "geometry_repair/DEPENDENCY_AUDIT.md",
    "geometry_repair/GEOGRAM_THIRD_PARTY_NOTICES.md": (
        "geometry_repair/GEOGRAM_THIRD_PARTY_NOTICES.md"
    ),
    "geometry_repair/OPEN_SOURCE_POLICY.md": "geometry_repair/OPEN_SOURCE_POLICY.md",
    "geometry_repair/sdf_backend_qualifications.json": (
        "geometry_repair/sdf_backend_qualifications.json"
    ),
}


def test_wheel_forces_license_policy_and_third_party_notices() -> None:
    configuration = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = configuration["project"]
    build = configuration["tool"]["hatch"]["build"]["targets"]
    wheel = build["wheel"]

    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert (PACKAGE_ROOT / "LICENSE").read_bytes() == (REPO_ROOT / "LICENSE").read_bytes()
    assert wheel["force-include"] == REQUIRED_WHEEL_FILES
    assert build["sdist"]["include"] == [
        "/LICENSE",
        "/README.md",
        "/pyproject.toml",
        "/geometry_repair",
        "/native",
        "/scripts",
        "/tests",
        "/uv.lock",
    ]
    for source in REQUIRED_WHEEL_FILES:
        path = (PACKAGE_ROOT / source).resolve()
        assert path.is_relative_to(REPO_ROOT)
        assert path.is_file()
        assert path.stat().st_size > 0
    assert not (PACKAGE_ROOT / "geometry_repair/OPENVDB_MPL_SOURCE.md").exists()


def test_docker_context_includes_current_geometry_dependency_audit() -> None:
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert (
        "!agentic/packages/geometry_repair/geometry_repair/DEPENDENCY_AUDIT.md"
        in dockerignore.splitlines()
    )
    assert "PERMISSIVE_NATIVE_NOTICES.md" not in dockerignore


def test_native_builder_distinguishes_fresh_clones_from_dirty_reuse() -> None:
    script = (PACKAGE_ROOT / "scripts" / "build_pmp_patch.sh").read_text(encoding="utf-8")

    assert 'local freshly_cloned="false"' in script
    assert 'freshly_cloned="true"' in script
    assert script.count('if [[ "${freshly_cloned}" == "false" ]] && \\') == 2
    assert "status --porcelain --untracked-files=all" in script
    assert 'rev-parse HEAD)" == "${commit}"' in script
    assert 'cat-file -e "${commit}^{commit}"' in script
    assert "GEOMETRY_REPAIR_JSON_REPOSITORY" in script
    assert "GEOMETRY_REPAIR_JSON_SOURCE_DIR" in script


def test_pmp_builder_accepts_only_explicit_mirror_and_source_overrides() -> None:
    script = (PACKAGE_ROOT / "scripts" / "build_pmp_patch.sh").read_text(encoding="utf-8")

    assert "GEOMETRY_REPAIR_PMP_REPOSITORY" in script
    assert "GEOMETRY_REPAIR_PMP_SOURCE_DIR" in script
    assert 'remote get-url origin)" != "${repository}"' in script
    assert 'verify_tree_inventory "${PMP_SOURCE}" "${PMP_TREE_INVENTORY_SHA256}"' in script


def test_pmp_native_writer_uses_exclusive_random_temporary_files() -> None:
    source = (PACKAGE_ROOT / "native" / "pmp_patch.cpp").read_text(encoding="utf-8")

    assert "mkstemp" in source
    assert 'path.filename().string() + ".tmp.XXXXXX"' in source
    assert "fsync" in source
    assert 'path.string() + ".tmp"' not in source
    assert "std::filesystem::remove(path" not in source


def test_pmp_native_writer_processes_one_digest_bound_source_capture() -> None:
    source = (PACKAGE_ROOT / "native" / "pmp_patch.cpp").read_text(encoding="utf-8")

    assert "output path must not overwrite the immutable source" in source
    assert source.count("std::filesystem::weakly_canonical(") >= 2
    assert "const auto source_text = read_bounded_text(source" in source
    assert "const auto captured_source_sha256 = sha256_text(source_text)" in source
    assert "read_source_mesh(source_text)" in source
    assert "sha256_file(source_path)" not in source
    assert source.count('request.at("source_sha256")') == 1


def test_native_mesh_adapters_reject_invalid_triangle_soups_before_libraries() -> None:
    pmp_source = (PACKAGE_ROOT / "native" / "pmp_patch.cpp").read_text(encoding="utf-8")
    cgal_source = (PACKAGE_ROOT / "native" / "cgal_exact_audit.cpp").read_text(encoding="utf-8")

    pmp_validation = pmp_source.index("neutral OBJ contains a repeated triangle vertex")
    assert pmp_validation < pmp_source.index("source.mesh.add_triangle(")
    assert "neutral OBJ contains a degenerate triangle" in pmp_source

    cgal_validation = cgal_source.index("polygon soup must contain triangles only")
    assert cgal_validation < cgal_source.index("triangle_soup_self_intersections(")
    assert "polygon soup contains an invalid triangle" in cgal_source


def test_production_pmp_builder_cannot_enable_gpl_evaluators() -> None:
    builder = (PACKAGE_ROOT / "scripts" / "build_pmp_patch.sh").read_text(encoding="utf-8")
    cmake = (PACKAGE_ROOT / "native" / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "-DGEOMETRY_REPAIR_BUILD_GPL_EVALUATORS=OFF" in builder
    assert (
        "option(\n  GEOMETRY_REPAIR_BUILD_GPL_EVALUATORS\n"
        '  "Build GPL evaluation-only binaries that are never shipped in production images"\n'
        "  OFF\n)" in cmake
    )
