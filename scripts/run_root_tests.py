#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run each public root pytest path in an isolated process.

Several applications deliberately expose generic test helpers and service
packages. A process per configured test root keeps those helpers local to their
owner while still using the shared, locked aggregate environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ElementTree
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTEST_EXIT_NO_TESTS_COLLECTED = 5
SHARED_PUBLIC_TEST_HELPERS = {
    "apps/texture_agent_service/tests": ("apps/texture_agent/tests",),
}


def configured_testpaths() -> list[str]:
    """Read the public root test paths from the authoritative pytest config."""
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    testpaths = project["tool"]["pytest"]["ini_options"]["testpaths"]
    if not isinstance(testpaths, list) or not all(
        isinstance(testpath, str) for testpath in testpaths
    ):
        raise ValueError(
            "[tool.pytest.ini_options].testpaths must be a list of strings"
        )
    return testpaths


def junitxml_destination(arguments: list[str]) -> tuple[list[str], str | None]:
    """Remove one JUnit report option and return its requested destination."""
    pytest_arguments: list[str] = []
    destination: str | None = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in {"--junitxml", "--junit-xml"}:
            index += 1
            if index == len(arguments):
                raise ValueError(f"{argument} requires a report path")
            candidate = arguments[index]
        elif argument.startswith("--junitxml=") or argument.startswith("--junit-xml="):
            candidate = argument.split("=", 1)[1]
        elif argument in {"-o", "--override-ini"}:
            index += 1
            if index == len(arguments):
                raise ValueError(f"{argument} requires an option")
            option = arguments[index]
            if option.startswith(("junitxml=", "junit_xml=")):
                candidate = option.split("=", 1)[1]
            else:
                pytest_arguments.extend((argument, option))
                index += 1
                continue
        elif argument.startswith("-o"):
            option = argument.removeprefix("-o")
            if option.startswith(("junitxml=", "junit_xml=")):
                candidate = option.split("=", 1)[1]
            else:
                pytest_arguments.append(argument)
                index += 1
                continue
        elif argument.startswith("--override-ini="):
            option = argument.split("=", 1)[1]
            if option.startswith(("junitxml=", "junit_xml=")):
                candidate = option.split("=", 1)[1]
            else:
                pytest_arguments.append(argument)
                index += 1
                continue
        else:
            pytest_arguments.append(argument)
            index += 1
            continue
        if not candidate:
            raise ValueError("JUnit report path must not be empty")
        if destination is not None:
            raise ValueError("Specify --junitxml only once")
        destination = candidate
        index += 1
    return pytest_arguments, destination


def merge_junitxml(reports: list[Path], destination: str) -> bool:
    """Merge suite nodes from isolated pytest reports into one JUnit document."""
    suites = ElementTree.Element("testsuites")
    complete = True
    for report in reports:
        try:
            root = ElementTree.parse(report).getroot()
        except (OSError, ElementTree.ParseError) as error:
            print(f"Unable to merge JUnit report {report}: {error}", file=sys.stderr)
            complete = False
            continue
        if root.tag == "testsuite":
            suites.append(root)
        elif root.tag == "testsuites":
            suites.extend(root.findall("testsuite"))
        else:
            print(
                f"Unable to merge JUnit report {report}: unexpected root {root.tag}",
                file=sys.stderr,
            )
            complete = False
    ElementTree.indent(suites)
    document = ElementTree.ElementTree(suites)
    if destination == "-":
        document.write(sys.stdout.buffer, encoding="utf-8", xml_declaration=True)
    else:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        document.write(path, encoding="utf-8", xml_declaration=True)
    return complete


def main(arguments: list[str]) -> int:
    """Run the caller's pytest options against every configured test root."""
    pytest_arguments, junitxml = junitxml_destination(arguments)
    with tempfile.TemporaryDirectory(prefix="root-pytest-") as temporary_directory:
        reports: list[Path] = []
        result_code = 0
        selected_tests = False
        for index, testpath in enumerate(configured_testpaths()):
            test_root = REPO_ROOT / testpath
            environment = os.environ.copy()
            local_imports = os.pathsep.join(
                (
                    str(test_root),
                    str(test_root.parent),
                    *(
                        str(REPO_ROOT / path)
                        for path in SHARED_PUBLIC_TEST_HELPERS.get(testpath, ())
                    ),
                    environment.get("PYTHONPATH", ""),
                )
            )
            environment["PYTHONPATH"] = local_imports.rstrip(os.pathsep)
            # The aggregate coverage runner owns coverage. Each isolated
            # child here would otherwise overwrite the prior child's reports.
            command = [sys.executable, "-m", "pytest", *pytest_arguments, "--no-cov"]
            if junitxml is not None:
                report = Path(temporary_directory) / f"{index}.xml"
                reports.append(report)
                command.append(f"--junitxml={report}")
            result = subprocess.run(
                [*command, testpath],
                cwd=REPO_ROOT,
                check=False,
                env=environment,
                stdout=sys.stderr if junitxml == "-" else None,
            )
            # A filter such as ``-k`` or ``-m`` normally selects tests from
            # only a subset of roots. Pytest's code 5 is neutral for one child,
            # but the aggregate must still report code 5 when no root matched.
            if result.returncode != PYTEST_EXIT_NO_TESTS_COLLECTED:
                selected_tests = True
                result_code = result_code or result.returncode
        if junitxml is not None:
            if not merge_junitxml(reports, junitxml):
                result_code = result_code or 1
        if not selected_tests and result_code == 0:
            result_code = PYTEST_EXIT_NO_TESTS_COLLECTED
    return result_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
