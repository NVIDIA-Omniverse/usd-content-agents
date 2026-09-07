#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the bounded detect-secrets scan used by the usd-cli component gate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DETECT_SECRETS_EXCLUSION = (
    r"(.*/)?src/usd_core/(?:render/pylock\.ovrtx-runtime|"
    r"pylock\.ovphysx-runtime(?:\.aarch64|\.py311(?:\.aarch64)?|-windows)?)\.toml$"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.source.resolve(strict=True)
    if not source.is_dir():
        raise SystemExit(f"component scan source is not a directory: {source}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "detect_secrets",
                "scan",
                "--all-files",
                "--exclude-files",
                DETECT_SECRETS_EXCLUSION,
            ],
            cwd=source,
            stdout=output,
            check=False,
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
