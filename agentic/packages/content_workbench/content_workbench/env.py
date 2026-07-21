# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment parsing helpers for the content workbench."""

from __future__ import annotations

import os


def first_nonempty_env(names: tuple[str, ...], default: str) -> str:
    """Return the first non-empty environment value from names."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default
