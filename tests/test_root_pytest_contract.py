# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep root pytest collection and its locked environment in agreement."""

from __future__ import annotations

import tomllib
from pathlib import Path
from xml.etree import ElementTree

import pytest

from scripts import run_root_tests
from tests.public_artifact_utils import public_doc_path

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT_TEST_EXCLUSIONS = (
    "geometry-authoring-app-" + "internal",
    "geometry-authoring-" + "internal",
    "geometry-authoring-service-" + "internal",
    "nvidia-" + "ocp-direct",
    "world-understanding-" + "internal",
)
NON_PACKAGED_PUBLIC_TEST_ROOTS = {"apps/content_agents_dashboard/tests"}


def _project_name(directory: str) -> str:
    project = tomllib.loads((REPO_ROOT / directory / "pyproject.toml").read_text())
    return project["project"]["name"]


def _requirement_name(requirement: str) -> str:
    return requirement.split("@", 1)[0].split("[", 1)[0].strip()


def test_root_test_packages_are_in_the_locked_aggregate_group() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    root_tests = project["tool"]["pytest"]["ini_options"]["testpaths"]
    aggregate = {
        _requirement_name(requirement)
        for requirement in project["dependency-groups"]["root-tests"]
    }
    sources = project["tool"]["uv"]["sources"]
    portable_local_constraints = {
        _requirement_name(requirement)
        for requirement in project["tool"]["uv"]["constraint-dependencies"]
        if "@ file://${PWD}/" in requirement
    }

    packaged_test_roots = [
        path.removesuffix("/tests").removesuffix("/tests/unit")
        for path in root_tests
        if path != "tests"
        and (REPO_ROOT / path).is_dir()
        and (
            REPO_ROOT
            / path.removesuffix("/tests").removesuffix("/tests/unit")
            / "pyproject.toml"
        ).is_file()
    ]
    package_names = {_project_name(directory) for directory in packaged_test_roots}

    assert package_names <= aggregate
    assert package_names <= set(sources) | portable_local_constraints
    unpackaged_test_roots = {
        path
        for path in root_tests
        if path != "tests"
        and (REPO_ROOT / path).is_dir()
        and not (
            REPO_ROOT
            / path.removesuffix("/tests").removesuffix("/tests/unit")
            / "pyproject.toml"
        ).is_file()
    }
    assert unpackaged_test_roots == NON_PACKAGED_PUBLIC_TEST_ROOTS


def test_root_aggregate_environment_excludes_internal_packages() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    aggregate = project["dependency-groups"]["root-tests"]
    sources = project["tool"]["uv"]["sources"]
    testpaths = project["tool"]["pytest"]["ini_options"]["testpaths"]

    assert not {_requirement_name(requirement) for requirement in aggregate} & set(
        PUBLIC_ROOT_TEST_EXCLUSIONS
    )
    assert not set(sources) & set(PUBLIC_ROOT_TEST_EXCLUSIONS)
    assert not any("/internal/" in path for path in testpaths)


def test_root_pytest_config_does_not_hide_nested_service_suites() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    pytest_options = project["tool"]["pytest"]["ini_options"]

    assert "norecursedirs" not in pytest_options
    assert "--ignore=tests/internal" in pytest_options["addopts"]


def test_documented_root_pytest_command_is_locked_and_aggregate() -> None:
    if not (REPO_ROOT / "README_PUBLIC.md").is_file():
        pytest.skip("public staging replaces the internal README")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "UV_PROJECT_ENVIRONMENT=.venv-root-tests" in readme
    assert "uv run --locked --group root-tests --extra dev" in readme
    assert "python scripts/run_root_tests.py" in readme


def test_public_documented_root_pytest_command_regenerates_its_lock() -> None:
    readme = public_doc_path(REPO_ROOT, "README_PUBLIC.md").read_text(encoding="utf-8")

    assert "UV_PROJECT_ENVIRONMENT=.venv-root-tests" in readme
    assert (
        "uv run --group root-tests --extra dev python scripts/run_root_tests.py"
        in readme
    )
    assert "uv run --locked --group root-tests" not in readme


def test_root_runner_rejects_malformed_testpaths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = 'tests'\n", encoding="utf-8"
    )
    monkeypatch.setattr(run_root_tests, "REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="list of strings"):
        run_root_tests.configured_testpaths()


@pytest.mark.parametrize(
    ("option", "equals_delimited"),
    (
        ("--junitxml", False),
        ("--junitxml", True),
        ("--junit-xml", False),
        ("--junit-xml", True),
    ),
)
def test_root_runner_merges_isolated_junit_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    equals_delimited: bool,
) -> None:
    monkeypatch.setattr(run_root_tests, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_root_tests, "configured_testpaths", lambda: ["one", "two"])
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> object:
        commands.append(command)
        report = Path(
            next(item for item in command if item.startswith("--junitxml=")).split(
                "=", 1
            )[1]
        )
        ElementTree.ElementTree(
            ElementTree.Element("testsuite", name=command[-1])
        ).write(report, encoding="utf-8", xml_declaration=True)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(run_root_tests.subprocess, "run", fake_run)
    destination = tmp_path / "reports" / "root.xml"

    arguments = (
        [f"{option}={destination}"] if equals_delimited else [option, str(destination)]
    )
    assert run_root_tests.main(arguments) == 0

    report_names = [
        suite.get("name") for suite in ElementTree.parse(destination).getroot()
    ]
    assert report_names == ["one", "two"]
    assert (
        len(
            {
                item
                for command in commands
                for item in command
                if item.startswith("--junitxml=")
            }
        )
        == 2
    )


def test_root_runner_keeps_available_junit_reports_after_a_child_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_root_tests, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_root_tests, "configured_testpaths", lambda: ["one", "two"])

    def fake_run(command: list[str], **_: object) -> object:
        if command[-1] == "one":
            report = Path(
                next(item for item in command if item.startswith("--junitxml=")).split(
                    "=", 1
                )[1]
            )
            ElementTree.ElementTree(ElementTree.Element("testsuite", name="one")).write(
                report, encoding="utf-8", xml_declaration=True
            )
            return type("Result", (), {"returncode": 0})()
        return type("Result", (), {"returncode": 1})()

    monkeypatch.setattr(run_root_tests.subprocess, "run", fake_run)
    destination = tmp_path / "reports" / "root.xml"

    assert run_root_tests.main([f"--junit-xml={destination}"]) == 1
    assert [
        suite.get("name") for suite in ElementTree.parse(destination).getroot()
    ] == ["one"]


@pytest.mark.parametrize(
    ("return_codes", "expected"),
    (
        ([5, 0], 0),
        ([5, 1], 1),
        ([5, 5], 5),
    ),
)
def test_root_runner_treats_empty_filtered_children_as_neutral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    return_codes: list[int],
    expected: int,
) -> None:
    monkeypatch.setattr(run_root_tests, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        run_root_tests,
        "configured_testpaths",
        lambda: [f"root-{index}" for index in range(len(return_codes))],
    )
    results = iter(return_codes)

    def fake_run(*_: object, **__: object) -> object:
        return type("Result", (), {"returncode": next(results)})()

    monkeypatch.setattr(run_root_tests.subprocess, "run", fake_run)

    assert run_root_tests.main(["-k", "selected_test"]) == expected


@pytest.mark.parametrize(
    ("arguments", "expected_path"),
    (
        (["-o", "junitxml=report.xml"], "report.xml"),
        (["-o", "junit_xml=report.xml"], "report.xml"),
        (["-ojunitxml=report.xml"], "report.xml"),
        (["--override-ini=junitxml=report.xml"], "report.xml"),
        (["--junit-xml", "-"], "-"),
    ),
)
def test_root_runner_accepts_pytest_junit_option_spellings(
    arguments: list[str], expected_path: str
) -> None:
    pytest_arguments, destination = run_root_tests.junitxml_destination(arguments)

    assert pytest_arguments == []
    assert destination == expected_path


@pytest.mark.parametrize("arguments", (["--junitxml="], ["-o", "junitxml="]))
def test_root_runner_rejects_empty_junit_report_paths(arguments: list[str]) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        run_root_tests.junitxml_destination(arguments)


def test_root_runner_reserves_stdout_for_a_stdout_junit_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_root_tests, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_root_tests, "configured_testpaths", lambda: ["tests"])
    observed_stdout: list[object] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        observed_stdout.append(kwargs.get("stdout"))
        report = Path(
            next(item for item in command if item.startswith("--junitxml=")).split(
                "=", 1
            )[1]
        )
        ElementTree.ElementTree(ElementTree.Element("testsuite", name="tests")).write(
            report, encoding="utf-8", xml_declaration=True
        )
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(run_root_tests.subprocess, "run", fake_run)
    monkeypatch.setattr(run_root_tests, "merge_junitxml", lambda *_: True)

    assert run_root_tests.main(["--junitxml=-"]) == 0
    assert observed_stdout == [run_root_tests.sys.stderr]
