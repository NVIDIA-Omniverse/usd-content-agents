# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for service package declarations.

`*_agent_service` apps used to declare `[tool.hatch.build.targets.wheel].only-include`
which ships files but does not register a Python package — so `pip install -e
apps/<svc>_agent_service` produced an empty wheel with no `_editable_impl_*.pth`,
and `from client.client import ...` from any cwd outside the service directory
raised `ModuleNotFoundError`.

The fix switched to `packages = ["service", "client"]`. This test pins that
declaration so a future edit cannot revert to the broken `only-include` form.
"""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent

AGENT_SERVICE_PYPROJECTS = sorted(
    (REPO_ROOT / "apps").glob("*_agent_service/pyproject.toml")
) + sorted((REPO_ROOT / "apps").glob("*_simple_service/pyproject.toml"))

TEXTURE_GEN_SERVICE_PYPROJECTS = sorted(
    (REPO_ROOT / "apps").glob("texture_gen_*_service/pyproject.toml")
)

REQUIRED_PACKAGES = {"service", "client"}
OVRTX_RUNTIME_LOCK_PACKAGE_PATH = (
    "world_understanding/functions/graphics/pylock.ovrtx-runtime.toml"
)
OVRTX_PYPROJECT_PATH = REPO_ROOT / "apps/ovrtx_rendering_api/pyproject.toml"


def test_world_understanding_wheel_force_includes_ovrtx_runtime_lock() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = data["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert force_include[OVRTX_RUNTIME_LOCK_PACKAGE_PATH] == (
        OVRTX_RUNTIME_LOCK_PACKAGE_PATH
    )
    assert (REPO_ROOT / OVRTX_RUNTIME_LOCK_PACKAGE_PATH).is_file()


def test_ovrtx_service_pydantic_floor_excludes_cve_2024_3772() -> None:
    data = tomllib.loads(OVRTX_PYPROJECT_PATH.read_text(encoding="utf-8"))
    pydantic = next(
        Requirement(dependency)
        for dependency in data["project"]["dependencies"]
        if Requirement(dependency).name == "pydantic"
    )

    assert Version("2.3.99") not in pydantic.specifier
    assert Version("2.11") in pydantic.specifier
    assert Version("3.0") not in pydantic.specifier


@pytest.mark.parametrize(
    "pyproject_path",
    AGENT_SERVICE_PYPROJECTS,
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_service_pyproject_declares_packages(pyproject_path: Path) -> None:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    wheel_target = (
        data.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel")
    )
    if wheel_target is None:
        pytest.skip(
            f"{pyproject_path.relative_to(REPO_ROOT)} has no hatch wheel target"
        )

    packages = wheel_target.get("packages")
    assert packages is not None, (
        f"{pyproject_path.relative_to(REPO_ROOT)} must declare "
        "`[tool.hatch.build.targets.wheel].packages`. The previous "
        "`only-include` form ships files without registering them as a "
        "Python package, so editable installs produced an empty wheel and "
        "`from client.client import ...` failed."
    )
    missing = REQUIRED_PACKAGES - set(packages)
    assert not missing, (
        f"{pyproject_path.relative_to(REPO_ROOT)} packages={packages!r} is "
        f"missing {sorted(missing)}. Both `service` and `client` must be "
        "registered so the documented `from client.client import ...` import "
        "works for editable installs."
    )
    for package in packages:
        package_init = pyproject_path.parent / package.replace(".", "/") / "__init__.py"
        assert package_init.is_file(), (
            f"{pyproject_path.relative_to(REPO_ROOT)} declares package "
            f"{package!r}, but {package_init.relative_to(REPO_ROOT)} is missing."
        )

    scripts = data.get("project", {}).get("scripts", {})
    for script_name, target in scripts.items():
        module_name = target.split(":", maxsplit=1)[0]
        module_path = pyproject_path.parent / f"{module_name.replace('.', '/')}.py"
        package_path = (
            pyproject_path.parent / module_name.replace(".", "/") / "__init__.py"
        )
        assert module_path.is_file() or package_path.is_file(), (
            f"{pyproject_path.relative_to(REPO_ROOT)} script {script_name!r} "
            f"targets {target!r}, but neither "
            f"{module_path.relative_to(REPO_ROOT)} nor "
            f"{package_path.relative_to(REPO_ROOT)} exists."
        )


def test_texture_generation_common_wheel_ships_shared_package() -> None:
    data = tomllib.loads(
        (REPO_ROOT / "apps/texture_gen_service_common/pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    force_include = (
        data["tool"]["hatch"]["build"]["targets"]["wheel"].get("force-include") or {}
    )
    expected = {
        "apps/__init__.py",
        "apps/texture_gen_service_common/__init__.py",
        "apps/texture_gen_service_common/artifacts.py",
        "apps/texture_gen_service_common/backend.py",
        "apps/texture_gen_service_common/models.py",
        "apps/texture_gen_service_common/service.py",
    }
    assert expected <= set(force_include.values())


def test_texture_gen_simple_wheel_ships_apps_compatibility_modules() -> None:
    data = tomllib.loads(
        (REPO_ROOT / "apps/texture_gen_simple_service/pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    force_include = (
        data["tool"]["hatch"]["build"]["targets"]["wheel"].get("force-include") or {}
    )
    expected = {
        "apps/__init__.py",
        "apps/texture_gen_simple_service/__init__.py",
        "apps/texture_gen_simple_service/app.py",
        "apps/texture_gen_simple_service/client/__init__.py",
        "apps/texture_gen_simple_service/client/client.py",
        "apps/texture_gen_simple_service/service/__init__.py",
        "apps/texture_gen_simple_service/service/main.py",
    }

    assert expected <= set(force_include.values())


@pytest.mark.parametrize(
    "pyproject_path",
    (
        REPO_ROOT / "apps/texture_gen_service_common/pyproject.toml",
        REPO_ROOT / "apps/texture_gen_simple_service/pyproject.toml",
    ),
    ids=lambda path: str(path.relative_to(REPO_ROOT)),
)
def test_texture_generation_force_includes_are_sdist_local(
    pyproject_path: Path,
) -> None:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    force_include = data["tool"]["hatch"]["build"]["targets"]["wheel"].get(
        "force-include", {}
    )

    assert force_include
    for source in force_include:
        source_path = Path(source)
        assert not source_path.is_absolute()
        assert ".." not in source_path.parts
        assert (pyproject_path.parent / source_path).is_file()


def test_texture_agent_declares_texture_gen_common_dependency() -> None:
    data = tomllib.loads(
        (REPO_ROOT / "apps/texture_agent/pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = data.get("project", {}).get("dependencies", [])
    common_requirements = [
        Requirement(dependency)
        for dependency in dependencies
        if Requirement(dependency).name == "texture-gen-service-common"
    ]

    assert common_requirements, (
        "apps/texture_agent/pyproject.toml must depend on "
        "texture-gen-service-common because texture_agent imports "
        "apps.texture_gen_service_common at runtime."
    )
    assert any(
        spec.operator == ">=" and Version(spec.version) >= Version("0.4.2")
        for requirement in common_requirements
        for spec in requirement.specifier
    )

    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    assert sources.get("texture-gen-service-common", {}).get("path") == (
        "../texture_gen_service_common"
    )


@pytest.mark.parametrize(
    "pyproject_path",
    TEXTURE_GEN_SERVICE_PYPROJECTS,
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_texture_generation_services_do_not_vendor_common_package(
    pyproject_path: Path,
) -> None:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    wheel_target = (
        data.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
    )
    force_include = wheel_target.get("force-include", {})
    vendored_common = [
        destination
        for destination in force_include.values()
        if str(destination).startswith("apps/texture_gen_service_common/")
    ]
    assert not vendored_common, (
        f"{pyproject_path.relative_to(REPO_ROOT)} vendors shared common files "
        f"that are already shipped by the texture-gen-service-common wheel: "
        f"{vendored_common}"
    )

    dependencies = data.get("project", {}).get("dependencies", [])
    world_understanding_requirements = [
        Requirement(dependency)
        for dependency in dependencies
        if Requirement(dependency).name == "world-understanding"
    ]
    assert world_understanding_requirements, (
        f"{pyproject_path.relative_to(REPO_ROOT)} must depend on world-understanding."
    )
    assert any(
        spec.operator == ">=" and Version(spec.version) >= Version("0.4.2")
        for requirement in world_understanding_requirements
        for spec in requirement.specifier
    ), (
        f"{pyproject_path.relative_to(REPO_ROOT)} must require "
        "world-understanding>=0.4.2 for compatibility with this service release."
    )

    common_requirements = [
        Requirement(dependency)
        for dependency in dependencies
        if Requirement(dependency).name == "texture-gen-service-common"
    ]
    assert common_requirements, (
        f"{pyproject_path.relative_to(REPO_ROOT)} must depend on "
        "texture-gen-service-common."
    )
    assert any(
        spec.operator == ">=" and Version(spec.version) >= Version("0.4.2")
        for requirement in common_requirements
        for spec in requirement.specifier
    ), (
        f"{pyproject_path.relative_to(REPO_ROOT)} must require "
        "texture-gen-service-common>=0.4.2 because the shared "
        "apps.texture_gen_service_common package is shipped by that wheel."
    )
