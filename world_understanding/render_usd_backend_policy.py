# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capability policy shared by ``wu render-usd`` and strict passthroughs."""

from world_understanding.rendering_backend_contract import rendering_backend_subset

RENDER_USD_BACKEND_NAMES: tuple[str, ...] = rendering_backend_subset(
    "remote",
    "ovrtx",
)
