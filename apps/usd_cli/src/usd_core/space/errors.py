# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA EmptySpaceDetectionDemo (esd/runtime.py), Apache-2.0.
# Only the exception hierarchy is carried over; upstream's ``runtime.py`` also binds
# the C ABI shared library (``esd.Instance`` / ``esd.version``), which this vendoring
# deliberately drops along with the whole C++ tree.

"""Detection-layer exceptions.

These stay ``RuntimeError`` subclasses so ``usd_server.app.dispatch`` turns them
into the ordinary ``ok: false`` Response envelope — a bad ``--size`` is a user
error with a clean message, not an HTTP 500 and not a traceback.
"""

from __future__ import annotations


class SpaceError(RuntimeError):
    """Base class for empty-space detection failures."""


class SpaceInvalidArgument(SpaceError):
    """A scope, size, or cell argument the detector cannot act on."""


class SpaceOutOfRange(SpaceError):
    """Config/scope would exceed a resource guard (DoS-by-input protection)."""
