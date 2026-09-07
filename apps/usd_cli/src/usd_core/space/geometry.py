# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA EmptySpaceDetectionDemo (esd/usd_scene.py), Apache-2.0.
# Only SceneGeometry + mask_out are carried over; upstream's ``load_scene`` is
# deliberately NOT vendored. It traverses with a plain ``stage.Traverse()`` and
# handles ``UsdGeom.Mesh`` only, which silently drops native-instanced geometry
# (near-empty on usd-cli's instance-heavy benchmark scenes), analytic Gprims
# (Cube/Sphere/Cylinder/Cone/Capsule/Plane — everything ``usd-cli create`` authors),
# deactivated prims, and invisible prims. ``usd_core.space.adapter.scene_geometry``
# replaces it by driving ``usd_core.raster._iter_geometry``, which already handles
# all four.

"""World-space triangle soup — the only input the detection pipeline consumes.

Pure numpy on purpose: no ``pxr`` import at module scope, so the compute layer
(heightfield → smooth → extract / support / contain) stays testable against
synthetic geometry and never drags USD into a Warp kernel path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class SceneGeometry:
    """Triangulated, world-space scene geometry (meters, Z-up)."""

    vertices: np.ndarray  # (V, 3) float32
    indices: np.ndarray  # (T, 3) int32 into vertices
    tri_object_id: np.ndarray  # (T,) int32 -> object slot
    object_paths: List[str] = field(default_factory=list)  # object slot -> prim path
    meters_per_unit: float = 1.0

    @property
    def num_triangles(self) -> int:
        return int(self.indices.shape[0])



def mask_out(scene: SceneGeometry, exclude_paths) -> SceneGeometry:
    """Return a copy of ``scene`` with every triangle belonging to ``exclude_paths`` dropped.

    Because each triangle carries its source prim path (``tri_object_id`` → ``object_paths``),
    "hiding the contents of a container" is a pure triangle mask — no stage mutation, no
    reload. Consumers (support-region / free-box detection) take an explicit scope window, so
    leaving the excluded objects' vertices in the pool is harmless; only their triangles are
    removed. Returns ``scene`` unchanged if nothing matches.
    """
    exclude = {str(p) for p in (exclude_paths or [])}
    if not exclude or scene.num_triangles == 0:
        return scene
    ex_oids = [oid for oid, path in enumerate(scene.object_paths) if path in exclude]
    if not ex_oids:
        return scene
    keep = ~np.isin(scene.tri_object_id, np.asarray(ex_oids, dtype=np.int32))
    return SceneGeometry(
        vertices=scene.vertices,
        indices=scene.indices[keep],
        tri_object_id=scene.tri_object_id[keep],
        object_paths=list(scene.object_paths),
        meters_per_unit=scene.meters_per_unit,
    )


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _under_any(path: str, roots) -> bool:
    """`path` is at or below any of `roots` — the container is the union of its parts.

    A container is rarely one prim. A cabinet ships as a left panel, a right panel, a
    back and a base; a rack is its uprights plus every shelf. Those have no useful
    common ancestor to name (or one that also sweeps in half the stage), so the
    container is defined by listing its components and taking them together.
    """
    return any(_under(path, r) for r in roots)
