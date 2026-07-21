# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for the historical Physics Agent client v2.

The v2-only polling fallback behavior now lives in ``client.py``. Keep this
module so older internal scripts that execute ``client_v2.py`` directly continue
to work while sharing the canonical client implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from client import PhysicsAgentClient, SSEMessage, build_arg_parser, main
else:  # pragma: no cover - exercised by package imports
    from .client import PhysicsAgentClient, SSEMessage, build_arg_parser, main

__all__ = ["PhysicsAgentClient", "SSEMessage", "build_arg_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
