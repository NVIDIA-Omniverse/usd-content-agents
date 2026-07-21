# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for Workbench-owned physics operations."""

from __future__ import annotations

from content_workbench.physics_ops import (
    PhysicsSimulationEngine,
    _write_trajectory_response,
    apply_schema,
    apply_topology_plan,
    inspect_authored_physics,
    inspect_components,
    inspect_mesh_candidates,
    inspect_topology,
    validate_runtime,
)

__all__ = [
    "PhysicsSimulationEngine",
    "_write_trajectory_response",
    "apply_schema",
    "apply_topology_plan",
    "inspect_authored_physics",
    "inspect_components",
    "inspect_mesh_candidates",
    "inspect_topology",
    "validate_runtime",
]
