# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safe, visible parsing for numeric environment overrides."""

from __future__ import annotations

import logging
import math
import os


def _range_description(
    type_name: str,
    minimum: int | float | None,
    maximum: int | float | None,
) -> str:
    if minimum is not None and maximum is not None:
        return f"{type_name} in the inclusive range {minimum}..{maximum}"
    if minimum is not None:
        return f"{type_name} greater than or equal to {minimum}"
    if maximum is not None:
        return f"{type_name} less than or equal to {maximum}"
    return type_name


def _warn_invalid(
    logger: logging.Logger,
    name: str,
    expected: str,
    default: int | float,
) -> None:
    # Never render the rejected environment value. Environment variables can
    # be populated from secret-bearing deployment configuration.
    logger.warning(
        "Invalid %s; expected %s. Using default %s.",
        name,
        expected,
        default,
    )


def parse_float_env(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    logger: logging.Logger,
) -> float:
    """Parse a finite float override, warning and falling back when invalid."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = float(raw_value)
        valid = math.isfinite(value)
    except (TypeError, ValueError):
        value = default
        valid = False

    if minimum is not None and value < minimum:
        valid = False
    if maximum is not None and value > maximum:
        valid = False
    if valid:
        return value

    _warn_invalid(
        logger,
        name,
        _range_description("a finite number", minimum, maximum),
        default,
    )
    return default


def parse_int_env(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    logger: logging.Logger,
) -> int:
    """Parse an integer override, warning and falling back when invalid."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
        valid = True
    except (TypeError, ValueError):
        value = default
        valid = False

    if minimum is not None and value < minimum:
        valid = False
    if maximum is not None and value > maximum:
        valid = False
    if valid:
        return value

    _warn_invalid(
        logger,
        name,
        _range_description("an integer", minimum, maximum),
        default,
    )
    return default
