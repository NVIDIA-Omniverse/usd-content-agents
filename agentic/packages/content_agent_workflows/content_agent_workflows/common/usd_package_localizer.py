# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolate OpenUSD dependency localization from the caller's layer registry."""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from world_understanding.utils.usd.package import safe_usdz_member_name

_LOCALIZED_MEMBER_WARNING = re.compile(
    r"^Warning: in _EnqueueDependency at line \d+ of .*[/\\]usdUtils[/\\]"
    r"assetLocalization\.cpp -- Failed to resolve reference "
    r"@(?P<member>[0-9]+/[^@]+)@ with computed asset path "
    r"@(?P=member)@ found in layer @[^@]+@\.$"
)


def create_localized_usdz_package(
    source_path: str | Path,
    output_path: str | Path,
    first_layer_name: str,
) -> None:
    """Localize one USD closure without leaking known USD cache diagnostics."""

    source_input = Path(source_path).expanduser()
    source_metadata = source_input.lstat()
    if not stat.S_ISREG(source_metadata.st_mode):
        raise ValueError(
            f"USD package localization source is not regular: {source_input}"
        )
    source = source_input.resolve(strict=True)
    output = Path(output_path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    normalized_root = safe_usdz_member_name(first_layer_name)
    if normalized_root != first_layer_name or Path(
        first_layer_name
    ).suffix.lower() not in {
        ".usd",
        ".usda",
        ".usdc",
    }:
        raise ValueError("USD package localization root member is invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    localization_error: BaseException | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix=".content-agent-usd-localizer-",
            dir=output.parent,
        ) as temporary_value:
            try:
                temporary_root = Path(temporary_value)
                if os.name != "nt":
                    temporary_root.chmod(0o700)
                temporary_metadata = temporary_root.lstat()
                if (
                    not stat.S_ISDIR(temporary_metadata.st_mode)
                    or temporary_root.is_symlink()
                    or (
                        os.name != "nt"
                        and (
                            stat.S_IMODE(temporary_metadata.st_mode) != 0o700
                            or temporary_metadata.st_uid != os.geteuid()
                        )
                    )
                ):
                    raise RuntimeError(
                        "USD package localization temporary root is not "
                        "owner-controlled"
                    )
                child_environment = os.environ.copy()
                child_environment.update(
                    {
                        "TMPDIR": str(temporary_root),
                        "TMP": str(temporary_root),
                        "TEMP": str(temporary_root),
                    }
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--localize-child",
                        str(source),
                        str(output),
                        first_layer_name,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=child_environment,
                )
                stderr_lines = tuple(
                    line.strip()
                    for line in completed.stderr.splitlines()
                    if line.strip()
                )
                unexpected_diagnostics = tuple(
                    line
                    for line in stderr_lines
                    if _LOCALIZED_MEMBER_WARNING.fullmatch(line) is None
                )
                if (
                    completed.returncode != 0
                    or completed.stdout.strip()
                    or unexpected_diagnostics
                    or not output.is_file()
                    or output.is_symlink()
                ):
                    detail = (
                        unexpected_diagnostics[0]
                        if unexpected_diagnostics
                        else completed.stdout.strip()
                        or f"localizer exited with code {completed.returncode}"
                    )
                    raise RuntimeError(
                        f"USD package localization failed closed: {detail}"
                    )
            except BaseException as error:
                localization_error = error
                raise
    except BaseException as exit_error:
        primary_error = (
            localization_error if localization_error is not None else exit_error
        )
        cleanup_replaced_localization_error = (
            localization_error is not None and exit_error is not localization_error
        )
        if cleanup_replaced_localization_error:
            add_note = getattr(localization_error, "add_note", None)
            if callable(add_note):
                add_note(
                    "USD package localization temporary cleanup also failed: "
                    f"{type(exit_error).__name__}: {exit_error}"
                )
        try:
            output.unlink(missing_ok=True)
        except BaseException as removal_error:
            add_note = getattr(primary_error, "add_note", None)
            if callable(add_note):
                add_note(
                    "USD package localization output removal also failed: "
                    f"{type(removal_error).__name__}: {removal_error}"
                )
            raise primary_error from removal_error
        if cleanup_replaced_localization_error:
            raise primary_error from exit_error
        raise


def _localize_child(argv: list[str]) -> int:
    if len(argv) != 5 or argv[1] != "--localize-child":
        raise ValueError(
            "usage: usd_package_localizer.py --localize-child SOURCE OUTPUT ROOT"
        )
    from pxr import UsdUtils

    source, output, first_layer_name = argv[2:]
    return (
        0
        if UsdUtils.CreateNewUsdzPackage(
            source,
            output,
            first_layer_name,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(_localize_child(sys.argv))
