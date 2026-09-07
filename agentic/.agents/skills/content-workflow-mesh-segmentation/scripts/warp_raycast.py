#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public Warp closest-hit ray query shared by mesh-segmentation tools."""

from __future__ import annotations

import warp as wp


def resolve_warp_device(name: str) -> wp.context.Device:
    """Resolve ``auto`` to CUDA when available and otherwise use Warp's CPU."""

    wp.init()
    if name != "auto":
        return wp.get_device(name)
    return wp.get_device("cuda:0" if wp.is_cuda_available() else "cpu")


@wp.func
def ray_intersect_mesh(
    mesh_id: wp.uint64,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    enable_backface_culling: bool,
    max_t: float,
) -> tuple[float, wp.vec3, float, float, int]:
    """Return distance, normal, barycentrics, and source face for one ray."""

    if mesh_id == wp.uint64(0):
        return -1.0, wp.vec3(0.0), 0.0, 0.0, -1

    query = wp.mesh_query_ray(mesh_id, ray_origin, ray_direction, max_t)
    if query.result:
        if not enable_backface_culling or wp.dot(ray_direction, query.normal) < 0.0:
            return query.t, wp.normalize(query.normal), query.u, query.v, query.face

    return -1.0, wp.vec3(0.0), 0.0, 0.0, -1
