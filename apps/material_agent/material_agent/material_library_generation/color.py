# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared color-transfer helpers for textures and linear shader controls."""

from __future__ import annotations

import numpy as np


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """Decode normalized IEC 61966-2-1 sRGB values into linear light."""

    encoded = np.asarray(values, dtype=np.float64)
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )


__all__ = ["srgb_to_linear"]
