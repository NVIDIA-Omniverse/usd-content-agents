# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic geometry repair workers."""

from .collision_geometry import CollisionGeometryWorker
from .geogram_local_repair import GeogramLocalRepairWorker
from .manifold_seam import ManifoldSeamWorker
from .ocp_shape_heal import OcpShapeHealWorker
from .pmp_patch import PmpPatchWorker
from .scene_optimizer_deinstance import SceneOptimizerDeinstanceWorker
from .sdf_rebuild import SdfRebuildWorker
from .trimesh_cleanup import TrimeshCleanupWorker
from .trimesh_hole_fill import TrimeshBoundedHoleFillWorker
from .usd_structure import UsdStructureRepairWorker

# Preserve explicit imports for one compatibility window without advertising the
# library-qualified name as a production worker ID.
OpenVdbRebuildWorker = SdfRebuildWorker

__all__ = [
    "CollisionGeometryWorker",
    "OcpShapeHealWorker",
    "SdfRebuildWorker",
    "PmpPatchWorker",
    "GeogramLocalRepairWorker",
    "ManifoldSeamWorker",
    "SceneOptimizerDeinstanceWorker",
    "TrimeshCleanupWorker",
    "TrimeshBoundedHoleFillWorker",
    "UsdStructureRepairWorker",
]
