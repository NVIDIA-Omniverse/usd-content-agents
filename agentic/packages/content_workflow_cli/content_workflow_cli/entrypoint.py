# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-safe console entry point for host-gated workflows."""

from __future__ import annotations

import sys

UNSUPPORTED_CAD_TO_SIMREADY_EXECUTION_HOST_MESSAGE = (
    "CAD-to-SimReady execution supports Linux, WSL2, and native Windows; this "
    "host is not supported. Run the workflow on a supported host. When that host "
    "cannot run OVRTX locally, set OVRTX_API_KEY and run `usd-cli remote configure "
    "https://gpu-host.example.com`, then verify it with `usd-cli render-probe "
    "--require-engine ovrtx`. The URL must expose the usd-cli OVRTX protocol "
    "service: /live identifies engine=ovrtx and protocol_version, while "
    "authenticated /ready and /health return JSON readiness responses; a "
    "plain-text metrics /health endpoint is incompatible."
)


def main(argv: list[str] | None = None) -> int:
    """Apply host capability gates before importing workflow implementations."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if sys.platform == "win32" and arguments[:2] == ["artifact", "write-json"]:
        # Managed Windows child commands have a short host-execution budget.
        # Keep the confined writer independent from the full workflow import
        # graph so a successful write is not misreported as a timeout.
        from .controlled_artifact_cli import main as artifact_main

        return artifact_main(arguments[2:])
    if (
        sys.platform not in {"linux", "win32"}
        and arguments[:1] == ["cad-to-simready"]
        and "--dry-run" not in arguments
        and not any(argument in {"-h", "--help"} for argument in arguments)
    ):
        print(
            f"cad-to-simready failed: "
            f"{UNSUPPORTED_CAD_TO_SIMREADY_EXECUTION_HOST_MESSAGE}",
            file=sys.stderr,
        )
        return 2
    from .cli import main as cli_main

    return cli_main(arguments)
