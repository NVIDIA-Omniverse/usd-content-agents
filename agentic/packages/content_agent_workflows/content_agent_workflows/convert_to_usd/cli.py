# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for the agentic convert-to-USD workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .workflow import OUTPUT_USD_FORMATS, convert_source_to_usd_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a source asset to a requested OpenUSD file."
    )
    parser.add_argument("source_asset", type=Path)
    parser.add_argument(
        "output_usd",
        type=Path,
        nargs="?",
        help="Output USD file. Defaults to ./<source-stem>.usda in the current working directory.",
    )
    parser.add_argument(
        "--install-missing",
        dest="install_missing",
        action="store_true",
        default=True,
        help="Install the converter package implied by the source extension before converting. This is the default.",
    )
    parser.add_argument(
        "--no-install-missing",
        dest="install_missing",
        action="store_false",
        help="Do not install missing converter dependencies before converting.",
    )
    parser.add_argument(
        "--output-format",
        choices=OUTPUT_USD_FORMATS,
        default=None,
        help=(
            "Output USD format to use when OUTPUT_USD is omitted. If OUTPUT_USD "
            "is provided, its extension must match this format."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path to write the normalized conversion report JSON.",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        help="Optional path to write the normalized conversion report Markdown.",
    )
    args = parser.parse_args(argv)

    try:
        report, _probe_artifact = convert_source_to_usd_file(
            args.source_asset,
            args.output_usd,
            output_format=args.output_format,
            install_missing=args.install_missing,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report_json = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_json + "\n", encoding="utf-8")
    if args.markdown_report is not None:
        args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_report.write_text(report.to_markdown(), encoding="utf-8")
    print(report_json)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
