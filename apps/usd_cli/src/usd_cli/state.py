# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Global CLI options, captured once in the Typer callback and read by every command."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GlobalOpts:
    json: bool = False
    quiet: bool = False
    server: str | None = None
    session: str | None = None
    timeout: float = 30.0
    # True when the user passed --timeout explicitly. An explicit value is a HARD ceiling on
    # total wall time even for slow commands (render/physics); otherwise slow commands get a
    # generous default floor so a snappy 30s default doesn't abort a legitimate GPU render.
    timeout_explicit: bool = False


# Single process-wide instance; set in main.callback().
G = GlobalOpts()
