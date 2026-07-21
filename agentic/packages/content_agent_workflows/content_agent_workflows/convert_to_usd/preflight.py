# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency preflight entrypoint for the agentic convert-to-USD workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import preflight_convert_to_usd_dependencies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install/check the converter dependency implied by a source asset."
    )
    parser.add_argument("source_asset", type=Path)
    parser.add_argument(
        "--install-missing",
        dest="install_missing",
        action="store_true",
        default=True,
        help="Install the converter package implied by the source extension. This is the default.",
    )
    parser.add_argument(
        "--no-install-missing",
        dest="install_missing",
        action="store_false",
        help="Check the implied converter dependency without installing it.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path to write the normalized preflight report JSON.",
    )
    args = parser.parse_args(argv)

    report = preflight_convert_to_usd_dependencies(
        args.source_asset,
        install_missing=args.install_missing,
    )
    report_json = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_json + "\n", encoding="utf-8")
    print(report_json)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
