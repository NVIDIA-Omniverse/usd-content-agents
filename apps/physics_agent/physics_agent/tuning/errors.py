# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exception types for the tuning package.

The error message strings here are part of the public contract: they are
asserted by the unit tests and documented in the issue Acceptance Criteria.
"""

from __future__ import annotations


class TuningError(RuntimeError):
    """Base class for physics-agent tuning failures."""


class TuningConfigError(ValueError):
    """Base class for invalid tuning configuration."""


class BoTorchUnavailableError(TuningError):
    """Raised when the BoTorch optimizer is requested but not installed.

    The issue body specifies the exact install hint that must be surfaced to
    the user; keep this message in lockstep.
    """

    DEFAULT_MESSAGE = (
        "BoTorch optimizer requires the tuning extra:\n"
        'uv pip install -e "apps/physics_agent[tuning]"'
    )

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE)


class OvPhysXUnavailableError(TuningError):
    """Raised when the OvPhysX backend is requested but not installed."""

    DEFAULT_MESSAGE = (
        "OvPhysX backend requires the tuning extra:\n"
        'uv pip install -e "apps/physics_agent[tuning]"'
    )

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE)


class NewtonUnavailableError(TuningError):
    """Raised when the Newton backend is requested but not installed."""

    DEFAULT_MESSAGE = (
        "Newton backend requires the newton extra:\n"
        'uv pip install -e "apps/physics_agent[newton]"'
    )

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE)


class TuningCancelledError(TuningError):
    """Raised when a tuning run is cancelled cooperatively.

    The CLI maps this to a soft-exit (status="cancelled" in the artifacts);
    the REST runner converts it into a session status update.
    """
