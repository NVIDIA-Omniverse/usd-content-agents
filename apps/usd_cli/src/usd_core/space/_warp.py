# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp bootstrap — the single place `warp` gets imported and initialized.

Two reasons this exists instead of the vendored modules' original module-scope
`import warp as wp; wp.init()`:

1. **stdout is a contract.** `wp.init()` prints a ten-line banner (version, CUDA
   toolkit, device table, kernel-cache path) to *stdout*. usd-cli's stdout carries
   the `--json` Response envelope and nothing else, so an un-suppressed init
   corrupts every machine-readable result the moment a space command runs.
   `wp.config.quiet` is set before the import completes.

2. **Import cost.** Warp initialization enumerates CUDA devices and warms a JIT
   cache — hundreds of ms. `usd_core.session` imports must stay cheap, so nothing
   under `usd_core.space` may be imported at `usd_core` import time; the Session
   reaches for these modules lazily, inside the space commands only.

Import failure is re-raised as `SpaceUnavailable` carrying the install hint, so a
host without the optional extra gets the standard usd-cli error envelope rather
than a bare `ModuleNotFoundError` traceback.
"""

from __future__ import annotations

import threading

from .errors import SpaceError

_INIT_LOCK = threading.Lock()
_initialized = False


class SpaceUnavailable(SpaceError):
    """The `space` extra (warp-lang) is not installed."""


_INSTALL_HINT = (
    "empty-space detection needs the 'space' extra (NVIDIA Warp). Install it with:\n"
    "  pip install -e '.[cli,server,space]'   (or: pip install 'warp-lang>=1.14')"
)


def warp():
    """Return the initialized `warp` module. Raises `SpaceUnavailable` if absent.

    Idempotent and thread-safe: the daemon runs commands on a worker pool, and two
    concurrent space queries must not race the one-time init.
    """
    global _initialized
    try:
        import warp as wp
    except ImportError as exc:
        raise SpaceUnavailable(_INSTALL_HINT) from exc

    if not _initialized:
        with _INIT_LOCK:
            if not _initialized:
                # Must be set before init() to suppress the stdout banner.
                wp.config.quiet = True
                wp.init()
                _initialized = True
    return wp
