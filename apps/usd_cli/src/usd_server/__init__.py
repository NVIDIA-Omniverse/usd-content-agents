# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""usd_server — thin HTTP daemon over usd_core (PR-9).

Holds the live USD stage in memory, exposes a single /cmd endpoint plus /batch, writes
`.usd-cli/server.json` for CLI discovery, and auto-shuts down when idle. Requires the
`server` extra (fastapi + uvicorn). The heavy deps (USD, render backends) live here, not
in the CLI process — that is the PR-9.7 win.
"""

from usd_server.app import build_app, serve

__all__ = ["build_app", "serve"]
