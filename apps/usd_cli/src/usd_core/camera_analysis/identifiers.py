# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable identifiers shared by camera analysis and verification export."""

from __future__ import annotations

import hashlib


def camera_stable_identity(prim, camera_path: str) -> tuple[str, str]:
    """Return the authored camera ID or its deterministic prim-path fallback."""

    authored = prim.GetCustomDataByKey("usdCameraStableId")
    if authored:
        return str(authored), "authored_custom_data"
    digest = hashlib.sha256(camera_path.encode("utf-8")).hexdigest()
    return f"camera-path-sha256:{digest}", "derived_prim_path"
