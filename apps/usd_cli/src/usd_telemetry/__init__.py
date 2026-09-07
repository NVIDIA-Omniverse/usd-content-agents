# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""usd-cli-tel — transparent telemetry wrapper around usd-cli.

A separate binary that execs the real ``usd-cli`` and records one
OTel-shaped span per invocation on the side. It imports nothing from
usd_cli/usd_core/usd_server, consumes no argv (all configuration is via
USD_CLI_TEL_* environment variables), and mirrors the child's exit code —
so it survives any usd-cli change short of renaming the binary.
"""

__version__ = "0.1.0"

# Bumped when the JSONL record shape changes incompatibly.
SCHEMA_VERSION = 1
