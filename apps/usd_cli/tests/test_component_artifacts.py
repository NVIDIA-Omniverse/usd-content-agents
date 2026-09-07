# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the bounded usd-cli component package surface."""

from __future__ import annotations

import importlib.util
import io
import re
import stat
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = ROOT / "scripts/check_component_artifacts.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "usd_cli_component_artifact_checker", CHECKER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_component_secret_scan_excludes_only_shipped_runtime_locks() -> None:
    spec = importlib.util.spec_from_file_location(
        "usd_cli_component_secret_scan", ROOT / "scripts/scan_component_secrets.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exclusion = module.DETECT_SECRETS_EXCLUSION

    for path in (
        "src/usd_core/render/pylock.ovrtx-runtime.toml",
        "src/usd_core/pylock.ovphysx-runtime.toml",
        "src/usd_core/pylock.ovphysx-runtime.aarch64.toml",
        "src/usd_core/pylock.ovphysx-runtime.py311.toml",
        "src/usd_core/pylock.ovphysx-runtime.py311.aarch64.toml",
        "src/usd_core/pylock.ovphysx-runtime-windows.toml",
    ):
        assert re.fullmatch(exclusion, path)
        assert re.fullmatch(exclusion, f"usd-cli/{path}")
    for path in (
        "src/usd_core/pylock.other.toml",
        "src/usd_core/pylock.ovphysx-runtime.py311-windows.toml",
        "src/usd_core/pylock.ovphysx-runtime.toml.orig",
    ):
        assert re.search(exclusion, path) is None


def test_packaged_ovphysx_locks_match_canonical_physics_agent_locks() -> None:
    """The standalone usd-cli image must attest the same reviewed profiles."""

    repo_root = ROOT.parents[1]
    runtime_dir = repo_root / "apps/physics_agent/runtime"
    for name in (
        "pylock.ovphysx-runtime.toml",
        "pylock.ovphysx-runtime.aarch64.toml",
        "pylock.ovphysx-runtime-windows.toml",
    ):
        assert (ROOT / "src/usd_core" / name).read_bytes() == (
            runtime_dir / name
        ).read_bytes()


def _metadata() -> bytes:
    return (
        b"Metadata-Version: 2.4\n"
        b"License-Expression: Apache-2.0\n"
        b"License-File: LICENSE\n\n"
    )


def _write_sdist(
    path: Path,
    extra_members: list[tuple[tarfile.TarInfo, bytes]] | None = None,
) -> None:
    entries = {
        "demo/LICENSE": b"Apache License\nVersion 2.0\n",
        "demo/PKG-INFO": _metadata(),
        "demo/src/module.py": b"VALUE = 1\n",
    }
    with tarfile.open(path, "w:gz") as archive:
        root = tarfile.TarInfo("demo/")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for name, payload in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        for member, payload in extra_members or []:
            archive.addfile(member, io.BytesIO(payload) if member.isfile() else None)


def _write_wheel(
    path: Path,
    extra_members: list[tuple[zipfile.ZipInfo, bytes]] | None = None,
) -> None:
    entries = {
        "demo/__init__.py": b"VALUE = 1\n",
        "demo-1.0.dist-info/METADATA": _metadata(),
        "demo-1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
        "demo-1.0.dist-info/licenses/LICENSE": b"Apache License\nVersion 2.0\n",
        "demo-1.0.dist-info/RECORD": b"",
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            member = zipfile.ZipInfo(name)
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(member, payload)
        for member, payload in extra_members or []:
            archive.writestr(member, payload)


def test_build_metadata_and_manifests_are_explicit_and_apache_licensed() -> None:
    root = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    service = tomllib.loads(
        (ROOT / "apps/ovrtx_rendering_api/pyproject.toml").read_text(encoding="utf-8")
    )

    assert root["project"]["license"] == "Apache-2.0"
    assert service["project"]["license"] == "Apache-2.0"
    assert "authors" not in root["project"]
    assert root["project"]["description"].startswith("In-tree Content Agents component")
    assert root["tool"]["hatch"]["build"]["targets"]["sdist"]["include"] == [
        "/LICENSE",
        "/README.md",
        "/pyproject.toml",
        "/requirements/constraints.txt",
        "/src",
    ]
    assert service["tool"]["hatch"]["build"]["targets"]["sdist"]["include"] == [
        "/LICENSE",
        "/README.md",
        "/pyproject.toml",
        "/service",
    ]
    for project in (root, service):
        targets = project["tool"]["hatch"]["build"]["targets"]
        assert targets["wheel"]["ignore-vcs"] is True
        assert targets["sdist"]["ignore-vcs"] is True
    assert (ROOT / "LICENSE").read_bytes() == (
        ROOT / "apps/ovrtx_rendering_api/LICENSE"
    ).read_bytes()


def test_linux_runtime_constraints_cover_service_dependency_closure() -> None:
    constraint_lines = {
        line
        for raw_line in (ROOT / "requirements/constraints.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    }
    constraints = {line.split("==", maxsplit=1)[0] for line in constraint_lines}
    assert {
        "annotated-doc",
        "annotated-types",
        "anyio",
        "certifi",
        "click",
        "fastapi",
        "h11",
        "httpcore",
        "httptools",
        "idna",
        "markdown-it-py",
        "mdurl",
        "numpy",
        "pillow",
        "pydantic",
        "pydantic-core",
        "pygments",
        "python-dotenv",
        "python-multipart",
        "pyyaml",
        "rich",
        "shellingham",
        "sniffio",
        "starlette",
        "typer",
        "typing-extensions",
        "typing-inspection",
        "usd-core",
        "uvicorn",
        "uvloop",
        "watchfiles",
        "websockets",
        "zstandard",
    } <= constraints
    assert "usd-core==26.8" in constraint_lines
    assert "zstandard==0.25.0" in constraint_lines
    service = tomllib.loads(
        (ROOT / "apps/ovrtx_rendering_api/pyproject.toml").read_text(encoding="utf-8")
    )
    assert "zstandard>=0.22,<0.26" in service["project"]["dependencies"]


def test_docker_context_and_python_contract_fail_closed() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    rules = [
        line
        for raw_line in dockerignore.splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]
    assert rules[0] == "**"
    assert set(rules[1:]) == {
        "**/*.dylib",
        "**/*.py[cod]",
        "**/*.pyd",
        "**/*.so",
        "**/__pycache__/",
        "!.dockerignore",
        "!LICENSE",
        "!README.md",
        "!apps/",
        "!apps/ovrtx_rendering_api/",
        "!apps/ovrtx_rendering_api/LICENSE",
        "!apps/ovrtx_rendering_api/README.md",
        "!apps/ovrtx_rendering_api/docker-entrypoint.sh",
        "!apps/ovrtx_rendering_api/pyproject.toml",
        "!apps/ovrtx_rendering_api/service/",
        "!apps/ovrtx_rendering_api/service/**",
        "!pyproject.toml",
        "!requirements/",
        "!requirements/constraints.txt",
        "!src/",
        "!src/**",
    }

    dockerfile = (ROOT / "apps/ovrtx_rendering_api/Dockerfile").read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "apps/ovrtx_rendering_api/README.md").read_text(encoding="utf-8")
    remote = (ROOT / "src/usd_cli/remote.py").read_text(encoding="utf-8")
    assert "COPY pyproject.toml README.md LICENSE /app/repo/" in dockerfile
    assert "(3, 11) <= sys.version_info[:2] < (3, 13)" in dockerfile
    assert "'/app/repo[cli,server]'" in dockerfile
    assert "-c /app/repo/requirements/constraints.txt" in dockerfile
    assert dockerfile.count("python3 -m pip wheel") == 1
    assert "'/app/repo[cli,server]' /app/service_app" in dockerfile
    assert dockerfile.index("COPY apps/ovrtx_rendering_api /app/service_app") < (
        dockerfile.index("RUN python3 -m pip wheel")
    )
    # The app install must keep resolving usd-cli and the service from the
    # locally built wheels. The isolated ovrtx venv below it legitimately uses
    # --no-deps, because that install is pinned to exact hashed artifacts from
    # the runtime lock and must not resolve anything from an index.
    app_install = dockerfile.split("&& python3 -m venv /opt/ovrtx_venv")[0]
    assert "--no-deps" not in app_install
    assert "--no-index --find-links /tmp/wheels" in dockerfile
    assert "'usd-cli[cli,server]==0.0.1'" in dockerfile
    assert "'ovrtx-rendering-api==0.1.0'" in dockerfile
    assert "/tmp/wheels/*.whl" not in dockerfile
    assert "ubuntu22.04" not in dockerfile + readme + remote
    assert "ubuntu24.04@sha256:<approved-digest>" in readme
    assert "USD_CLI_CUDA_BASE_IMAGE" in readme
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized_root_readme = " ".join(root_readme.split())
    assert "low-level component of NVIDIA Content Agents" in root_readme
    assert "not a standalone repository, product, workflow" in normalized_root_readme


def test_lfs_determinism_check_uses_isolated_source_copies() -> None:
    check = (ROOT / "scripts/check_component_build_determinism.py").read_text(
        encoding="utf-8"
    )

    assert "_copy_source(pointer_source)" in check
    assert "_copy_source(hydrated_source)" in check
    assert "(pointer_source / LFS_FIXTURE_RELATIVE).write_bytes(pointer)" in check
    assert "(hydrated_source / LFS_FIXTURE_RELATIVE).write_bytes(materialized)" in check
    assert "LFS_FIXTURE.write_bytes" not in check
    assert "assert _git_status() == original_status" in check
    assert "final_stat.st_mtime_ns" in check


def test_valid_minimal_archives_pass_inventory_and_license_checks(
    tmp_path: Path,
) -> None:
    sdist = tmp_path / "demo.tar.gz"
    wheel = tmp_path / "demo.whl"
    _write_sdist(sdist)
    _write_wheel(wheel)

    sdist_summary = checker.validate_sdist(
        sdist,
        label="demo sdist",
        allowed_top_level=frozenset({"LICENSE", "PKG-INFO", "src"}),
        required_files=frozenset({"LICENSE", "PKG-INFO", "src/module.py"}),
        max_bytes=1024,
    )
    wheel_summary = checker.validate_wheel(
        wheel,
        label="demo wheel",
        package_roots=frozenset({"demo"}),
        required_files=frozenset({"demo/__init__.py"}),
        dist_info_prefix="demo-",
        max_bytes=1024,
    )

    assert sdist_summary.files == 3
    assert wheel_summary.files == 5


@pytest.mark.parametrize("archive_kind", ["sdist", "wheel"])
def test_archives_reject_git_lfs_pointer_payloads(
    tmp_path: Path, archive_kind: str
) -> None:
    pointer = checker.LFS_POINTER_PREFIX + b"oid sha256:" + b"0" * 64 + b"\nsize 1\n"
    if archive_kind == "sdist":
        archive = tmp_path / "pointer.tar.gz"
        member = tarfile.TarInfo("demo/src/pointer.usd")
        member.size = len(pointer)
        _write_sdist(archive, [(member, pointer)])

        def validate() -> None:
            checker.validate_sdist(
                archive,
                label="pointer sdist",
                allowed_top_level=frozenset({"LICENSE", "PKG-INFO", "src"}),
                required_files=frozenset(),
                max_bytes=1024,
            )

    else:
        archive = tmp_path / "pointer.whl"
        member = zipfile.ZipInfo("demo/pointer.usd")
        member.external_attr = (stat.S_IFREG | 0o644) << 16
        _write_wheel(archive, [(member, pointer)])

        def validate() -> None:
            checker.validate_wheel(
                archive,
                label="pointer wheel",
                package_roots=frozenset({"demo"}),
                required_files=frozenset(),
                dist_info_prefix="demo-",
                max_bytes=1024,
            )

    with pytest.raises(checker.ArtifactValidationError, match="Git LFS pointer"):
        validate()


def test_archives_reject_traversal_special_types_and_oversize(
    tmp_path: Path,
) -> None:
    traversal = tarfile.TarInfo("demo/../escape")
    traversal.size = 1
    traversal_archive = tmp_path / "traversal.tar.gz"
    _write_sdist(traversal_archive, [(traversal, b"x")])
    with pytest.raises(checker.ArtifactValidationError, match="unsafe archive"):
        checker.validate_sdist(
            traversal_archive,
            label="traversal",
            allowed_top_level=frozenset({"LICENSE", "PKG-INFO", "src"}),
            required_files=frozenset(),
            max_bytes=1024,
        )

    symlink = zipfile.ZipInfo("demo/link")
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    symlink_archive = tmp_path / "symlink.whl"
    _write_wheel(symlink_archive, [(symlink, b"target")])
    with pytest.raises(checker.ArtifactValidationError, match="non-regular"):
        checker.validate_wheel(
            symlink_archive,
            label="symlink",
            package_roots=frozenset({"demo"}),
            required_files=frozenset(),
            dist_info_prefix="demo-",
            max_bytes=1024,
        )

    oversized_archive = tmp_path / "oversized.tar.gz"
    oversized = tarfile.TarInfo("demo/src/large.bin")
    oversized.size = 32
    _write_sdist(oversized_archive, [(oversized, b"x" * 32)])
    with pytest.raises(checker.ArtifactValidationError, match="exceeds 16"):
        checker.validate_sdist(
            oversized_archive,
            label="oversized",
            allowed_top_level=frozenset({"LICENSE", "PKG-INFO", "src"}),
            required_files=frozenset(),
            max_bytes=16,
        )


def test_checker_bounds_raw_archive_size_and_member_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdist = tmp_path / "bounded.tar.gz"
    wheel = tmp_path / "bounded.whl"
    _write_sdist(sdist)
    _write_wheel(wheel)

    monkeypatch.setattr(checker, "MAX_ARCHIVE_FILE_BYTES", sdist.stat().st_size - 1)
    with pytest.raises(checker.ArtifactValidationError, match="archive file exceeds"):
        checker.validate_sdist(
            sdist,
            label="raw-size",
            allowed_top_level=frozenset({"LICENSE", "PKG-INFO", "src"}),
            required_files=frozenset(),
            max_bytes=1024,
        )

    monkeypatch.setattr(checker, "MAX_ARCHIVE_FILE_BYTES", 1024 * 1024)
    monkeypatch.setattr(checker, "MAX_ARCHIVE_MEMBERS", 3)
    with pytest.raises(checker.ArtifactValidationError, match="archive members"):
        checker.validate_sdist(
            sdist,
            label="sdist-members",
            allowed_top_level=frozenset({"LICENSE", "PKG-INFO", "src"}),
            required_files=frozenset(),
            max_bytes=1024,
        )

    with pytest.raises(checker.ArtifactValidationError, match="archive members"):
        checker.validate_wheel(
            wheel,
            label="wheel-members",
            package_roots=frozenset({"demo"}),
            required_files=frozenset(),
            dist_info_prefix="demo-",
            max_bytes=1024,
        )
