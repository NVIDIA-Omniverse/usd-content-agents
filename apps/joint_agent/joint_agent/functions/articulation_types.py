# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared wire-level articulation type aliases."""

from typing import Literal

type ArticulationReviewStatus = Literal[
    "ready_for_rigger_input",
    "review_required",
]

__all__ = ["ArticulationReviewStatus"]
