# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that Dockerfiles contain required security packages and CVE documentation."""

import re
import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
OVRTX_RUNTIME_LOCK = (
    REPO_ROOT
    / "world_understanding"
    / "functions"
    / "graphics"
    / "pylock.ovrtx-runtime.toml"
)
OVRTX_PROVISION_COMMAND = (
    "python -m world_understanding.functions.graphics.render_ovrtx --provision-only"
)
OVRTX_TEMP_UV_CACHE = "/tmp/wu-ovrtx-uv-cache"
ROOT_UV_CACHE = "/root/.cache/uv"

# Dockerfiles for each service
PHYSICS_AGENT_DOCKERFILES = [
    REPO_ROOT / "apps" / "physics_agent_service" / "Dockerfile",
    REPO_ROOT / "apps" / "physics_agent_service" / "Dockerfile.ci",
]

JOINT_AGENT_DOCKERFILES = [
    REPO_ROOT / "apps" / "joint_agent_service" / "Dockerfile",
    REPO_ROOT / "apps" / "joint_agent_service" / "Dockerfile.ci",
]

MATERIAL_AGENT_DOCKERFILES = [
    REPO_ROOT / "apps" / "material_agent_service" / "Dockerfile",
    REPO_ROOT / "apps" / "material_agent_service" / "Dockerfile.ci",
]

OVRTX_RENDERING_API_CI_DOCKERFILES = [
    REPO_ROOT / "apps" / "ovrtx_rendering_api" / "Dockerfile.ci",
]

OVRTX_RENDERING_API_PUBLIC_DOCKERFILES = [
    REPO_ROOT / "apps" / "ovrtx_rendering_api" / "Dockerfile",
]

TEXTURE_AGENT_DOCKERFILES = [
    REPO_ROOT / "apps" / "texture_agent_service" / "Dockerfile",
    REPO_ROOT / "apps" / "texture_agent_service" / "Dockerfile.ci",
]

TEXTURE_STEP1X_SERVICE_DOCKERFILES = [
    REPO_ROOT / "apps" / "texture_gen_step1x_service" / "Dockerfile",
]

INTERNAL_SERVICE_CI_DOCKERFILES = [
    REPO_ROOT / "apps" / "material_agent_service" / "Dockerfile.ci",
    REPO_ROOT / "apps" / "physics_agent_service" / "Dockerfile.ci",
    REPO_ROOT / "apps" / "joint_agent_service" / "Dockerfile.ci",
    REPO_ROOT / "apps" / "texture_agent_service" / "Dockerfile.ci",
]

IMAGE_SCAN_SENSITIVE_INTERNAL_CI_DOCKERFILES = [
    REPO_ROOT / "apps" / "material_agent_service" / "Dockerfile.ci",
    REPO_ROOT / "apps" / "physics_agent_service" / "Dockerfile.ci",
    REPO_ROOT / "apps" / "texture_agent_service" / "Dockerfile.ci",
    REPO_ROOT / "apps" / "ovrtx_rendering_api" / "Dockerfile.ci",
]

IMAGE_SCAN_SENSITIVE_PUBLIC_DOCKERFILES = [
    REPO_ROOT / "apps" / "material_agent_service" / "Dockerfile",
    REPO_ROOT / "apps" / "physics_agent_service" / "Dockerfile",
    REPO_ROOT / "apps" / "texture_agent_service" / "Dockerfile",
    *TEXTURE_STEP1X_SERVICE_DOCKERFILES,
    *OVRTX_RENDERING_API_PUBLIC_DOCKERFILES,
]

ROOT_UV_CACHE_CLEANUP_DOCKERFILES = [
    REPO_ROOT / "apps" / "material_agent_service" / "Dockerfile",
    REPO_ROOT / "apps" / "physics_agent_service" / "Dockerfile",
    REPO_ROOT / "apps" / "texture_agent_service" / "Dockerfile",
]

UV_BUILD_CACHE_DISABLED_PUBLIC_DOCKERFILES = [
    *ROOT_UV_CACHE_CLEANUP_DOCKERFILES,
    *OVRTX_RENDERING_API_PUBLIC_DOCKERFILES,
]

ALL_DOCKERFILES = (
    PHYSICS_AGENT_DOCKERFILES + JOINT_AGENT_DOCKERFILES + MATERIAL_AGENT_DOCKERFILES
)

CVE_2026_4519_DOCKERFILES = (
    ALL_DOCKERFILES + OVRTX_RENDERING_API_CI_DOCKERFILES + TEXTURE_AGENT_DOCKERFILES
)

DISCOVERED_DOCKERFILES = sorted((REPO_ROOT / "apps").glob("*/Dockerfile*"))


def _read_discovered_dockerfiles() -> tuple[dict[Path, str], dict[Path, OSError]]:
    """Read discovered Dockerfiles without failing pytest collection."""
    contents: dict[Path, str] = {}
    read_errors: dict[Path, OSError] = {}
    for path in DISCOVERED_DOCKERFILES:
        if not path.is_file():
            continue
        try:
            contents[path] = path.read_text()
        except OSError as exc:
            read_errors[path] = exc
    return contents, read_errors


def _rel(path: Path) -> str:
    """Return a stable repository-relative path for parametrized test ids."""
    return path.relative_to(REPO_ROOT).as_posix()


def _docker_instructions(content: str) -> list[str]:
    """Normalize Dockerfile instructions across line continuations."""
    instructions: list[str] = []
    current: list[str] = []
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not current and (not stripped or stripped.startswith("#")):
            continue
        current.append(stripped)
        if not stripped.endswith("\\"):
            instructions.append(
                re.sub(r"\s+", " ", " ".join(part.rstrip("\\") for part in current))
            )
            current = []
    return instructions


def _base_images(content: str) -> list[str]:
    """Return Dockerfile base image references in declaration order."""
    return [
        line.split()[1]
        for line in content.splitlines()
        if line.strip().startswith("FROM ")
    ]


def _assert_pip_and_uv_scan_floor(dockerfile: Path, content: str) -> None:
    """Assert the venv bootstrap installs the scanned pip floor and uv."""
    upgrade_instructions = [
        instruction
        for instruction in _docker_instructions(content)
        if "python -m pip install --no-cache-dir --upgrade" in instruction
    ]
    assert any(
        '"pip>=26.1"' in instruction
        and re.search(r'(?:^|\s)"?uv(?:[<>=][^"\s]+)?"?(?:\s|$)', instruction)
        for instruction in upgrade_instructions
    ), f"{_rel(dockerfile)} does not install pip>=26.1 and uv in its venv"


def _expected_gpgme_package(content: str) -> str:
    """Choose the libgpgme package name for the Dockerfile base OS."""
    base_images = _base_images(content)
    if any(base == "python:3.12-slim" for base in base_images):
        return "libgpgme11"
    return "libgpgme11t64"


def _matches_scene_optimizer_chmod_pattern(instruction: str) -> bool:
    """Check for the chmod-based Scene Optimizer bundle permission repair."""
    return (
        "BUILD_RESOURCES=/app/.build-resources" in instruction
        and "SO=$BUILD_RESOURCES/scene_optimizer_core" in instruction
        and 'chmod a+X "$BUILD_RESOURCES"' in instruction
        and 'chmod -R a+rX "$SO"' in instruction
    )


DOCKERFILE_CONTENT, DOCKERFILE_READ_ERRORS = _read_discovered_dockerfiles()
CUDA_UBUNTU_2404_DOCKERFILES = [
    path
    for path, content in DOCKERFILE_CONTENT.items()
    if "nvcr.io/nvidia/cuda:" in content and "ubuntu24.04" in content
]
SCENE_OPTIMIZER_BUNDLE_DOCKERFILES = [
    path
    for path, content in DOCKERFILE_CONTENT.items()
    if "COPY .build-resources /app/.build-resources" in content
    and "WU_SO_PACKAGE_DIR=/app/.build-resources/scene_optimizer_core" in content
]
SOURCE_PERMISSION_DOCKERFILES = {
    REPO_ROOT / "apps" / "material_agent_service" / "Dockerfile": (
        "/app/world_understanding",
        "/app/apps/material_agent",
        "/app/apps/material_agent_service",
    ),
    REPO_ROOT / "apps" / "physics_agent_service" / "Dockerfile": (
        "/app/world_understanding",
        "/app/apps/physics_agent",
        "/app/apps/physics_agent_service",
    ),
    REPO_ROOT / "apps" / "ovrtx_rendering_api" / "Dockerfile": (
        "/app/world_understanding",
        "/app/apps/ovrtx_rendering_api",
    ),
    REPO_ROOT / "apps" / "texture_agent_service" / "Dockerfile": (
        "/app/world_understanding",
        "/app/apps/texture_agent",
        "/app/apps/texture_agent_service",
    ),
}

# gnupg packages required by CVE-2025-68973 fix
GNUPG_COMMON_PACKAGES = [
    "gnupg",
    "gpg",
    "gpg-agent",
    "gpgconf",
    "gpgsm",
    "gpg-wks-client",
    "dirmngr",
]

_APT_COMMAND_PREFIX = (
    r"\b(?:apt-get|apt)(?:\s+--?[A-Za-z0-9][A-Za-z0-9-]*(?:=\S+)?)*\s+"
)

# Matches apt-get install blocks (handles line continuations with \)
_APT_INSTALL_RE = re.compile(
    rf"{_APT_COMMAND_PREFIX}install\b.*?(?=(?:^[A-Z]|\Z))",
    re.DOTALL | re.MULTILINE,
)
_DOCKERFILE_INSTRUCTION_RE = (
    r"ADD|ARG|CMD|COPY|ENTRYPOINT|ENV|EXPOSE|FROM|HEALTHCHECK|LABEL|MAINTAINER|"
    r"ONBUILD|RUN|SHELL|STOPSIGNAL|USER|VOLUME|WORKDIR"
)
_RUN_RE = re.compile(
    rf"^RUN\s+(.*?)(?=^(?:{_DOCKERFILE_INSTRUCTION_RE})\b|\Z)",
    re.MULTILINE | re.DOTALL,
)
_APT_UPGRADE_RE = re.compile(rf"{_APT_COMMAND_PREFIX}(?:dist-)?upgrade\b")
_APT_CLEAN_RE = re.compile(rf"{_APT_COMMAND_PREFIX}clean\b")
_PILLOW_REQ_PIN_RE = re.compile(
    r"^\s*pillow\s*==\s*([A-Za-z0-9][A-Za-z0-9.!+_-]*)\s*(?:#.*)?$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_apt_install_sections(content: str) -> str:
    """Extract all apt-get install command sections from Dockerfile content."""
    return " ".join(_APT_INSTALL_RE.findall(content))


def _strip_comment_lines(content: str) -> str:
    """Remove comment-only lines before command regex checks."""
    return "\n".join(
        line for line in content.splitlines() if not line.lstrip().startswith("#")
    )


def _extract_cuda_ubuntu24_sections(content: str) -> str:
    """Extract stages that inherit from CUDA Ubuntu 24.04 runtime images."""
    sections: list[str] = []
    current: list[str] = []
    in_cuda_ubuntu24_stage = False

    for line in content.splitlines():
        if line.startswith("FROM "):
            if in_cuda_ubuntu24_stage:
                sections.append("\n".join(current))
            in_cuda_ubuntu24_stage = (
                "nvcr.io/nvidia/cuda:" in line and "ubuntu24.04" in line
            )
            current = [line] if in_cuda_ubuntu24_stage else []
        elif in_cuda_ubuntu24_stage:
            current.append(line)

    if in_cuda_ubuntu24_stage:
        sections.append("\n".join(current))
    return "\n".join(sections)


def _extract_ubuntu24_sections(content: str) -> str:
    """Extract stages that inherit from Ubuntu 24.04 runtime images."""
    sections: list[str] = []
    current: list[str] = []
    in_ubuntu24_stage = False

    for line in content.splitlines():
        if line.startswith("FROM "):
            if in_ubuntu24_stage:
                sections.append("\n".join(current))
            in_ubuntu24_stage = (
                line.startswith("FROM ubuntu:24.04") or "ubuntu24.04" in line
            )
            current = [line] if in_ubuntu24_stage else []
        elif in_ubuntu24_stage:
            current.append(line)

    if in_ubuntu24_stage:
        sections.append("\n".join(current))
    return "\n".join(sections)


def _extract_run_sections(content: str) -> list[str]:
    """Extract Dockerfile RUN instructions so comments cannot satisfy checks."""
    return _RUN_RE.findall(_strip_comment_lines(content))


def _assert_refreshes_cuda_base_os_packages(dockerfile: Path, content: str) -> None:
    """Assert CUDA Ubuntu Dockerfiles refresh and clean inherited OS packages."""
    cuda_sections = _extract_cuda_ubuntu24_sections(content)
    run_sections = _extract_run_sections(cuda_sections)
    upgrade_runs = [
        section for section in run_sections if _APT_UPGRADE_RE.search(section)
    ]
    assert upgrade_runs, (
        f"{dockerfile.name} uses an NGC CUDA Ubuntu 24.04 base image but does "
        "not run apt-get upgrade before installing packages in that stage. CUDA "
        "runtime tags can lag Ubuntu security pockets, leaving stale "
        "OpenSSL/PAM/GnuTLS packages that fail image scans."
    )
    for run_section in upgrade_runs:
        upgrade_match = _APT_UPGRADE_RE.search(run_section)
        clean_match = _APT_CLEAN_RE.search(run_section)
        archive_index = run_section.find("/var/cache/apt/archives")
        assert upgrade_match is not None
        assert (
            clean_match is not None and clean_match.start() > upgrade_match.start()
        ), (
            f"{dockerfile.name} runs apt-get upgrade but does not clean apt caches "
            "later in the same RUN layer."
        )
        assert archive_index > upgrade_match.start(), (
            f"{dockerfile.name} runs apt-get upgrade but does not remove downloaded "
            "apt package archives later in the same RUN layer."
        )


def _assert_refreshes_ubuntu_base_os_packages(dockerfile: Path, content: str) -> None:
    """Assert Ubuntu 24.04 Dockerfiles refresh and clean inherited packages."""
    ubuntu_sections = _extract_ubuntu24_sections(content)
    run_sections = _extract_run_sections(ubuntu_sections)
    upgrade_runs = [
        section for section in run_sections if _APT_UPGRADE_RE.search(section)
    ]
    assert upgrade_runs, (
        f"{dockerfile.name} uses an Ubuntu 24.04 base image but does not run "
        "apt-get upgrade before installing packages in that stage."
    )
    for run_section in upgrade_runs:
        upgrade_match = _APT_UPGRADE_RE.search(run_section)
        clean_match = _APT_CLEAN_RE.search(run_section)
        archive_index = run_section.find("/var/cache/apt/archives")
        assert upgrade_match is not None
        assert (
            clean_match is not None and clean_match.start() > upgrade_match.start()
        ), (
            f"{dockerfile.name} runs apt-get upgrade but does not clean apt caches "
            "later in the same RUN layer."
        )
        assert archive_index > upgrade_match.start(), (
            f"{dockerfile.name} runs apt-get upgrade but does not remove downloaded "
            "apt package archives later in the same RUN layer."
        )


def _assert_secure_pillow_pin(path: Path, content: str) -> None:
    """Assert a requirements file explicitly pins a non-vulnerable Pillow."""
    versions = [
        Version(match.group(1)) for match in _PILLOW_REQ_PIN_RE.finditer(content)
    ]
    assert versions, (
        f"{path.name} does not explicitly pin Pillow. Image scans require "
        "Pillow >= 12.3.0 for the isolated OVRTX runtime."
    )
    for version in versions:
        assert version >= Version("12.3.0"), (
            f"{path.name} pins Pillow {version}; image scans require Pillow "
            ">= 12.3.0 for the isolated OVRTX runtime."
        )


def _assert_ovrtx_provision_cleans_uv_cache(dockerfile: Path, content: str) -> None:
    """Assert OVRTX provisioning cannot leave scanner-visible uv archives."""
    provision_runs = [
        section
        for section in _extract_run_sections(content)
        if OVRTX_PROVISION_COMMAND in section
    ]
    assert provision_runs, f"{dockerfile.name} does not run OVRTX provisioning"

    for run_section in provision_runs:
        provision_index = run_section.index(OVRTX_PROVISION_COMMAND)
        uv_cache_index = run_section.find(f"UV_CACHE_DIR={OVRTX_TEMP_UV_CACHE}")
        assert 0 <= uv_cache_index < provision_index, (
            f"{dockerfile.name} provisions OVRTX without a temporary UV_CACHE_DIR; "
            "uv can otherwise persist scanner-visible OVRTX wheel archives."
        )
        for cache_path in (OVRTX_TEMP_UV_CACHE, ROOT_UV_CACHE):
            cleanup_index = run_section.find(cache_path, provision_index)
            assert cleanup_index > provision_index, (
                f"{dockerfile.name} provisions OVRTX but does not remove "
                f"{cache_path} later in the same RUN layer."
            )


def test_apt_command_patterns_accept_flags_before_subcommands() -> None:
    """Apt guardrails should accept common flags before the subcommand."""
    assert _APT_UPGRADE_RE.search("apt-get -y upgrade")
    assert _APT_UPGRADE_RE.search("apt-get --yes upgrade")
    assert _APT_CLEAN_RE.search("apt-get --quiet clean")

    install_section = _extract_apt_install_sections(
        "RUN apt-get --no-install-recommends install \\\n    gnupg \\\n    gpg\n"
    )
    assert "gnupg" in install_section
    assert "gpg" in install_section


def test_secure_pillow_pin_accepts_inline_comments() -> None:
    """Requirements comments should not hide a secure Pillow pin."""
    _assert_secure_pillow_pin(
        Path("requirements.txt"),
        "# Historical vulnerable pin: pillow==12.1.1\n"
        "pillow==12.3.0  # image scan floor\n",
    )


@pytest.mark.parametrize(
    "dockerfile",
    DISCOVERED_DOCKERFILES,
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_discovered_dockerfiles_are_readable(dockerfile: Path) -> None:
    """Unreadable Dockerfiles should fail as tests, not during collection."""
    error = DOCKERFILE_READ_ERRORS.get(dockerfile)
    if error is not None:
        pytest.fail(f"Could not read discovered Dockerfile {dockerfile}: {error}")


@pytest.mark.parametrize(
    "dockerfile",
    DISCOVERED_DOCKERFILES,
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_dockerfiles_do_not_append_existing_pythonpath(dockerfile: Path) -> None:
    """Docker images must not add an empty current-directory entry to sys.path."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")

    content = dockerfile.read_text()
    assert ":${PYTHONPATH}" not in content
    assert ":$PYTHONPATH" not in content


@pytest.mark.parametrize(
    "dockerfile",
    PHYSICS_AGENT_DOCKERFILES,
    ids=lambda p: p.name,
)
def test_physics_agent_has_gnupg_packages(dockerfile: Path) -> None:
    """CVE-2025-68973: physics-agent-service Dockerfiles must install gnupg packages."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")
    content = dockerfile.read_text()
    apt_sections = _extract_apt_install_sections(content)
    for pkg in GNUPG_COMMON_PACKAGES + [_expected_gpgme_package(content)]:
        assert re.search(rf"\b{re.escape(pkg)}\b", apt_sections), (
            f"{dockerfile.name} is missing gnupg package '{pkg}' "
            f"in apt install commands (required for CVE-2025-68973)"
        )


@pytest.mark.parametrize(
    "dockerfile",
    INTERNAL_SERVICE_CI_DOCKERFILES,
    ids=_rel,
)
def test_internal_service_ci_images_copy_logging_config(dockerfile: Path) -> None:
    """Internal service CI images should preserve structured logging config."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")

    content = dockerfile.read_text()
    assert "COPY logging.yaml /logging.yaml" in content


@pytest.mark.parametrize(
    "dockerfile",
    IMAGE_SCAN_SENSITIVE_INTERNAL_CI_DOCKERFILES,
    ids=_rel,
)
def test_image_scan_sensitive_internal_ci_images_use_distro_python_runtime(
    dockerfile: Path,
) -> None:
    """Release-scanned CI images must not ship official Python /usr/local CPEs."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")

    content = dockerfile.read_text()
    apt_sections = _extract_apt_install_sections(content)

    assert "FROM python:" not in content
    assert "FROM ubuntu:24.04" in content
    assert "python3.12" in apt_sections
    assert "python3.12-venv" in apt_sections
    assert "RUN python3.12 -m venv /opt/venv" in content
    assert 'ENV PATH="/opt/venv/bin:$PATH"' in content
    _assert_pip_and_uv_scan_floor(dockerfile, content)
    assert "ENV UV_NO_CACHE=1" in content
    assert "uv pip install --system" not in content
    assert "/usr/local/bin/python3.12" not in content
    assert "/usr/local/lib/libpython3.12" not in content
    _assert_refreshes_ubuntu_base_os_packages(dockerfile, content)


@pytest.mark.parametrize(
    "dockerfile",
    UV_BUILD_CACHE_DISABLED_PUBLIC_DOCKERFILES,
    ids=_rel,
)
def test_public_service_builds_disable_uv_cache(dockerfile: Path) -> None:
    """Public images that run uv while building must disable its cache."""
    content = dockerfile.read_text()
    assert "ENV UV_NO_CACHE=1" in content, (
        f"{_rel(dockerfile)} does not disable uv's build cache"
    )


@pytest.mark.parametrize(
    "dockerfile",
    IMAGE_SCAN_SENSITIVE_PUBLIC_DOCKERFILES,
    ids=_rel,
)
def test_image_scan_sensitive_public_images_use_distro_python_runtime(
    dockerfile: Path,
) -> None:
    """Release-scanned public images must not ship official Python /usr/local CPEs."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")

    content = dockerfile.read_text()
    apt_sections = _extract_apt_install_sections(content)

    assert "FROM python:" not in content
    assert "FROM ubuntu:24.04" in content
    assert "python3.12" in apt_sections
    assert "python3.12-venv" in apt_sections
    assert "RUN python3.12 -m venv /opt/venv" in content
    assert 'ENV PATH="/opt/venv/bin:$PATH"' in content
    _assert_pip_and_uv_scan_floor(dockerfile, content)
    assert "uv pip install --system" not in content
    assert "/usr/local/bin/python3.12" not in content
    assert "/usr/local/lib/libpython3.12" not in content
    _assert_refreshes_ubuntu_base_os_packages(dockerfile, content)


@pytest.mark.parametrize(
    "dockerfile",
    ROOT_UV_CACHE_CLEANUP_DOCKERFILES,
    ids=_rel,
)
def test_public_service_images_remove_root_uv_cache(dockerfile: Path) -> None:
    """Public service builds must remove uv's root-owned bootstrap cache."""
    content = dockerfile.read_text()
    assert "RUN rm -rf /root/.cache/uv" in content, (
        f"{_rel(dockerfile)} does not remove uv's root-owned bootstrap cache"
    )


def test_ovrtx_ci_healthcheck_is_http_only_for_gpu_less_hosts() -> None:
    """The OVRTX CI image healthcheck must not require a GPU-ready renderer."""
    dockerfile = REPO_ROOT / "apps" / "ovrtx_rendering_api" / "Dockerfile.ci"
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")

    instructions = _docker_instructions(dockerfile.read_text())
    healthchecks = [
        instruction
        for instruction in instructions
        if instruction.startswith("HEALTHCHECK ")
    ]
    assert healthchecks, f"{_rel(dockerfile)} has no HEALTHCHECK instruction"
    assert "curl -fsS http://localhost:8000/health" in healthchecks[-1]
    assert "gpu_initialized" not in healthchecks[-1]


@pytest.mark.parametrize(
    "dockerfile",
    CVE_2026_4519_DOCKERFILES,
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_dockerfile_has_cve_2026_4519_comment(dockerfile: Path) -> None:
    """CVE-2026-4519: all Dockerfiles must document the webbrowser.open() CVE status."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")
    content = dockerfile.read_text()
    comment_lines = [
        line for line in content.splitlines() if line.strip().startswith("#")
    ]
    assert any("CVE-2026-4519" in line for line in comment_lines), (
        f"{dockerfile.name} is missing CVE-2026-4519 documentation comment"
    )


@pytest.mark.parametrize(
    "dockerfile",
    SCENE_OPTIMIZER_BUNDLE_DOCKERFILES,
    ids=_rel,
)
def test_scene_optimizer_bundle_permissions_before_runtime_user(
    dockerfile: Path,
) -> None:
    """Scene Optimizer build resources must be readable by non-root services."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")

    instructions = _docker_instructions(dockerfile.read_text())
    copy_index = instructions.index("COPY .build-resources /app/.build-resources")
    user_index = next(
        (
            index
            for index, instruction in enumerate(instructions)
            if instruction.startswith("USER ")
        ),
        None,
    )
    if user_index is None:
        pytest.fail(f"{_rel(dockerfile)} has no USER instruction")

    permission_repairs = [
        instruction
        for instruction in instructions[copy_index + 1 : user_index]
        if instruction.startswith("RUN ")
        and _matches_scene_optimizer_chmod_pattern(instruction)
    ]
    assert permission_repairs, (
        f"{_rel(dockerfile)} must make the staged Scene Optimizer bundle "
        "readable and traversable before switching to the non-root service user"
    )


@pytest.mark.parametrize(
    ("dockerfile", "expected_paths"),
    SOURCE_PERMISSION_DOCKERFILES.items(),
    ids=[_rel(path) for path in SOURCE_PERMISSION_DOCKERFILES],
)
def test_runtime_source_permissions_before_non_root_user(
    dockerfile: Path,
    expected_paths: tuple[str, ...],
) -> None:
    """Copied source trees must be importable by non-root runtime users."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")

    instructions = _docker_instructions(dockerfile.read_text())
    user_index = next(
        index
        for index, instruction in enumerate(instructions)
        if instruction.startswith("USER ")
    )
    permission_repairs = [
        instruction
        for instruction in instructions[:user_index]
        if instruction.startswith("RUN ")
        and "chmod -R a+rX" in instruction
        and all(path in instruction for path in expected_paths)
    ]
    assert permission_repairs, (
        f"{_rel(dockerfile)} must make copied application source readable and "
        "traversable before switching to the non-root service user"
    )


def test_joint_ci_installs_app_before_ovrtx_provisioning() -> None:
    """The source-style CI image must install world_understanding before OVRTX."""
    dockerfile = REPO_ROOT / "apps" / "joint_agent_service" / "Dockerfile.ci"
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")

    instructions = _docker_instructions(dockerfile.read_text())
    install_idx = next(
        (
            index
            for index, instruction in enumerate(instructions)
            if instruction.startswith("RUN ")
            and 'uv pip install -e ".[warp]"' in instruction
        ),
        None,
    )
    provision_idx = next(
        (
            index
            for index, instruction in enumerate(instructions)
            if instruction.startswith("RUN ") and OVRTX_PROVISION_COMMAND in instruction
        ),
        None,
    )
    if install_idx is None or provision_idx is None:
        pytest.fail(
            f"{_rel(dockerfile)} is missing the warp install or OVRTX "
            "provisioning instruction"
        )

    assert install_idx < provision_idx, (
        f"{_rel(dockerfile)} must install world_understanding before OVRTX "
        f"provisioning; found install at instruction {install_idx} and "
        f"provisioning at instruction {provision_idx}"
    )


@pytest.mark.parametrize(
    "dockerfile",
    JOINT_AGENT_DOCKERFILES,
    ids=lambda p: p.name,
)
def test_joint_agent_ovrtx_uses_shared_provisioner(dockerfile: Path) -> None:
    """Joint images must use the shared OVRTX provisioner and its sanitizer."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")

    content = _strip_comment_lines(dockerfile.read_text())
    assert OVRTX_PROVISION_COMMAND in content
    assert "ovrtx==" not in content


@pytest.mark.parametrize(
    "dockerfile",
    JOINT_AGENT_DOCKERFILES,
    ids=lambda p: p.name,
)
def test_joint_agent_ovrtx_provisioning_cleans_uv_cache(
    dockerfile: Path,
) -> None:
    """Joint image scans must not see OVRTX libpython copies in uv caches."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")

    _assert_ovrtx_provision_cleans_uv_cache(dockerfile, dockerfile.read_text())


@pytest.mark.parametrize(
    "dockerfile",
    OVRTX_RENDERING_API_CI_DOCKERFILES,
    ids=lambda p: p.name,
)
def test_ngc_ovrtx_api_provisioning_cleans_uv_cache(
    dockerfile: Path,
) -> None:
    """NGC OVRTX image scans must not see uv wheel archives in final layers."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")

    _assert_ovrtx_provision_cleans_uv_cache(dockerfile, dockerfile.read_text())


@pytest.mark.parametrize(
    "dockerfile",
    JOINT_AGENT_DOCKERFILES,
    ids=lambda p: p.name,
)
def test_joint_agent_dockerfiles_are_covered_by_scan_guardrails(
    dockerfile: Path,
) -> None:
    """Joint service images are the image-scan regression surface."""
    if not dockerfile.exists():
        pytest.skip(f"{dockerfile} not present in this checkout (e.g. public mirror)")
    assert dockerfile in CUDA_UBUNTU_2404_DOCKERFILES
    content = dockerfile.read_text()
    _assert_refreshes_cuda_base_os_packages(dockerfile, content)


@pytest.mark.parametrize(
    "dockerfile",
    CUDA_UBUNTU_2404_DOCKERFILES,
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_cuda_ubuntu24_dockerfiles_refresh_base_os_packages(
    dockerfile: Path,
) -> None:
    """CUDA Ubuntu 24.04 runtime images must refresh stale base OS packages."""
    content = dockerfile.read_text()
    _assert_refreshes_cuda_base_os_packages(dockerfile, content)


def test_ovrtx_runtime_lock_pillow_pin_is_not_vulnerable() -> None:
    """The shared OVRTX runtime dependency pin must not use vulnerable Pillow."""
    if not OVRTX_RUNTIME_LOCK.exists():
        pytest.skip(
            f"{OVRTX_RUNTIME_LOCK} not present in this checkout (e.g. public mirror)"
        )

    with OVRTX_RUNTIME_LOCK.open("rb") as stream:
        lock = tomllib.load(stream)
    pillow_versions = [
        Version(package["version"])
        for package in lock["packages"]
        if package["name"] == "pillow"
    ]
    assert pillow_versions
    assert all(version >= Version("12.3.0") for version in pillow_versions)
