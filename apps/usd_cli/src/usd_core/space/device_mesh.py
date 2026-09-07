# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident triangle soup — the geometry every kernel here reads.

Holds a scene's vertices, indices and face->object table on one device, so the
heightfield rasteriser uploads once per query instead of once per pass. That is its
whole job.

There is deliberately **no BVH**. One was built here on every construction, left over
from the ray sampler below, and nothing read it after that sampler was removed — a
full acceleration structure over 2.3M triangles, per query, for a code path that never
traces a ray. Re-add it with the traversal that needs it, not before.

**Renamed from ESD's `raycast.Raycaster`.** The old name described the original
single-ray-per-cell sampler, which cast one down-ray per grid cell. The detector moved
to a rasterised multi-layer heightfield (`heightfield.build_spans_raster`) that reads
the resident arrays directly and casts nothing, so the sampler — and with it this
class's `cast()` and the `Raycaster` alias — has been removed.

Keeping the old name was actively misleading: it made "the ray fell through the slat
gap onto the floor below" read as a live description of the pipeline, when in fact the
rasteriser reports *no span at all* for a column it finds no geometry in. That
difference is the whole reason `gapfill` exists (see its module docstring), and the
stale name hid it.
"""

from __future__ import annotations

from ._warp import warp as _warp

wp = _warp()

from .geometry import SceneGeometry


class DeviceMesh:
    """A scene's triangles resident on one device."""

    def __init__(self, scene: SceneGeometry, device: str = "cpu"):
        self.device = device
        self._points = wp.array(scene.vertices, dtype=wp.vec3, device=device)
        self._indices = wp.array(scene.indices.reshape(-1), dtype=wp.int32, device=device)
        self._face_object = wp.array(scene.tri_object_id, dtype=wp.int32, device=device)
