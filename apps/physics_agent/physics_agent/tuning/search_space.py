# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the shared optimization search-space contracts."""

from __future__ import annotations

from world_understanding.optimization.contracts import (
    BoundedParameter,
    BoundedSearchSpace,
)

__all__ = ["BoundedParameter", "BoundedSearchSpace"]
