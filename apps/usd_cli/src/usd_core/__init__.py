# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""usd_core — the pure usd-cli engine.

This package has ZERO CLI/web dependencies. It is the thing you `import` to drive a
scene as a library; `usd_server` wraps it over HTTP and `usd_cli` is a thin client.

    from usd_core import Session
    s = Session.open("village.usd")
    snap = s.snapshot(visible=True, depth=2)
    s.transform(s.ref("@n5"), tx="+3", ry="90")
    delta = s.snapshot(diff=True)
"""

from usd_core.config import Config, load_config
from usd_core.history import History, Op
from usd_core.models import Artifact, Issue, Response, SCHEMA_VERSION
from usd_core.refs import RefTable, StaleRefError
from usd_core.session import Session

__all__ = [
    "Config",
    "load_config",
    "Artifact",
    "Issue",
    "Response",
    "SCHEMA_VERSION",
    "RefTable",
    "StaleRefError",
    "History",
    "Op",
    "Session",
]
