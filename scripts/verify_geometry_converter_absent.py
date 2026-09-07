# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail closed if the deferred Geometry CAD converter enters an environment."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
import sysconfig
from pathlib import Path

CONVERTER_DISTRIBUTION = "usd-convert-cad"
CONVERTER_MODULE = "usd_convert_cad"
CONVERTER_ENTRY_POINTS = ("usd-convert-cad", "usd-convert-cad.exe")
EMBEDDED_PYTHON_GLOB = "omni/converter/hoops/libpython3.12.so*"


def main() -> int:
    failures: list[str] = []
    try:
        installed_version = importlib.metadata.version(CONVERTER_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
    if installed_version is not None:
        failures.append(
            f"forbidden {CONVERTER_DISTRIBUTION} {installed_version} distribution found"
        )

    if importlib.util.find_spec(CONVERTER_MODULE) is not None:
        failures.append(f"forbidden {CONVERTER_MODULE} import found")

    scripts_directory = Path(sys.executable).expanduser().absolute().parent.resolve()
    for entry_point in CONVERTER_ENTRY_POINTS:
        candidate = scripts_directory / entry_point
        if candidate.exists():
            failures.append(f"forbidden converter entry point found: {candidate}")

    site_packages = {
        Path(value).resolve()
        for key, value in sysconfig.get_paths().items()
        if key in {"purelib", "platlib"} and value
    }
    for directory in sorted(site_packages):
        for candidate in directory.glob(EMBEDDED_PYTHON_GLOB):
            failures.append(f"forbidden embedded Python runtime found: {candidate}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print("Geometry CAD converter absence verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
