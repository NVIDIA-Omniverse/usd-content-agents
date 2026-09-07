# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-neutral capability constants for the opt-in Joint 0.6 adapter."""

SOURCE_BACKED_STAGE2_SOURCES = frozenset(
    {
        "accepted_manifest",
        "authored_metadata",
        "authored_reference",
        "source_metadata",
    }
)
PUBLIC_OWNED_CORE_STAGE2_CAPABILITY_IDS = frozenset(
    {
        "prismatic.cardinal_topology.0_5",
        "revolute.cardinal_topology.0_5",
    }
)
PUBLIC_OWNED_CORE_STAGE2_TYPES = frozenset({"prismatic", "revolute"})
INTERNAL_V1_BREADTH_CAPABILITY_IDS = frozenset(
    {
        "prismatic.arbitrary_axis",
        "revolute.arbitrary_axis",
        "revolute.continuous_normalization",
        "spherical.passive_topology",
    }
)
INTERNAL_V1_BREADTH_STAGE2_TYPES = frozenset(
    {*PUBLIC_OWNED_CORE_STAGE2_TYPES, "spherical"}
)
SOURCE_BACKED_DERIVATION = "source_backed_stage2_adapter_to_articulation_contract_v1"
CONTINUOUS_DERIVATION = "continuous_to_unbounded_revolute_v1"

__all__ = [
    "CONTINUOUS_DERIVATION",
    "INTERNAL_V1_BREADTH_CAPABILITY_IDS",
    "INTERNAL_V1_BREADTH_STAGE2_TYPES",
    "PUBLIC_OWNED_CORE_STAGE2_CAPABILITY_IDS",
    "PUBLIC_OWNED_CORE_STAGE2_TYPES",
    "SOURCE_BACKED_STAGE2_SOURCES",
    "SOURCE_BACKED_DERIVATION",
]
