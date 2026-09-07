# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Errors raised by the domain-neutral optimization runtime."""

from __future__ import annotations


class OptimizationError(RuntimeError):
    """Base class for shared optimization failures."""


class OptimizationCancelledError(OptimizationError):
    """Raised internally when a caller cooperatively cancels a trial run."""


class OptimizerUnavailableError(OptimizationError):
    """Raised when an optional optimizer dependency is unavailable."""

    DEFAULT_MESSAGE = (
        "Requested optimizer is unavailable; install the application's "
        "required optimization dependencies."
    )

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE)


__all__ = [
    "OptimizationCancelledError",
    "OptimizationError",
    "OptimizerUnavailableError",
]
