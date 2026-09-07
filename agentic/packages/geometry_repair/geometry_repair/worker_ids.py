# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical repair-worker identifiers and bounded legacy aliases."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

SDF_REBUILD_WORKER = "sdf_rebuild"
SDF_COLLISION_REBUILD_WORKER = "sdf_collision_rebuild"

LEGACY_WORKER_ALIASES = MappingProxyType(
    {
        "openvdb_rebuild": SDF_REBUILD_WORKER,
        "openvdb_collision_rebuild": SDF_COLLISION_REBUILD_WORKER,
    }
)


def canonical_worker_name(name: str) -> str:
    """Return the canonical worker ID for one exact supported legacy alias."""

    if not isinstance(name, str):
        raise TypeError("worker name must be a string")
    return LEGACY_WORKER_ALIASES.get(name, name)


def canonical_worker_names(names: Iterable[str]) -> list[str]:
    """Normalize worker IDs and reject aliases that collapse to a duplicate."""

    canonical: list[str] = []
    originals_by_canonical: dict[str, str] = {}
    for name in names:
        normalized = canonical_worker_name(name)
        previous = originals_by_canonical.get(normalized)
        if previous is not None:
            raise ValueError(
                f"worker names {previous!r} and {name!r} resolve to duplicate canonical "
                f"worker {normalized!r}"
            )
        originals_by_canonical[normalized] = name
        canonical.append(normalized)
    return canonical


__all__ = [
    "LEGACY_WORKER_ALIASES",
    "SDF_COLLISION_REBUILD_WORKER",
    "SDF_REBUILD_WORKER",
    "canonical_worker_name",
    "canonical_worker_names",
]
