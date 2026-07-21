# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight command for SimReady Foundation workflow dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .foundation_runtime import resolve_simready_runtime
from .models import SimReadyPreflightReport


def preflight_simready_foundation(
    *,
    foundation_root: Path | str | None = None,
    foundation_spec_root: Path | str | None = None,
    venv_path: Path | str | None = None,
    install_missing: bool = True,
    update_foundation: bool = False,
) -> SimReadyPreflightReport:
    """Resolve SimReady Foundation specs/runtime and return a report."""

    runtime = resolve_simready_runtime(
        foundation_root=foundation_root,
        foundation_spec_root=foundation_spec_root,
        venv_path=venv_path,
        install_missing=install_missing,
        update_foundation=update_foundation,
    )
    return runtime.to_preflight_report()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare/check SimReady Foundation workflow dependencies."
    )
    parser.add_argument("--foundation-root", type=Path)
    parser.add_argument("--foundation-spec-root", type=Path)
    parser.add_argument("--venv", dest="venv_path", type=Path)
    parser.add_argument(
        "--install-missing",
        dest="install_missing",
        action="store_true",
        default=True,
        help="Install missing SimReady validation dependencies. This is the default.",
    )
    parser.add_argument(
        "--no-install-missing",
        dest="install_missing",
        action="store_false",
        help="Check only; do not clone or install missing dependencies.",
    )
    parser.add_argument(
        "--update-foundation",
        action="store_true",
        help="Fetch/update a managed Foundation checkout before checking it.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    report = preflight_simready_foundation(
        foundation_root=args.foundation_root,
        foundation_spec_root=args.foundation_spec_root,
        venv_path=args.venv_path,
        install_missing=args.install_missing,
        update_foundation=args.update_foundation,
    )
    report_json = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_json + "\n", encoding="utf-8")
    print(report_json)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
