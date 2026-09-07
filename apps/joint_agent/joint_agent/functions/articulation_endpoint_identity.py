# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-backed endpoint identity for Joint topology inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ArticulationEndpointResolution(BaseModel):
    """Deterministic resolution of one asserted endpoint alias family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: Literal["resolved", "ambiguous", "unresolved"]
    requested_paths: tuple[str, ...]
    canonical_path: str | None = None
    candidate_paths: tuple[str, ...] = ()
    reason: Literal[
        "canonical_identity",
        "conflicting_canonical_identities",
        "ambiguous_source_owner",
        "unknown_endpoint",
        "missing_endpoint",
    ]


class ArticulationEndpointIdentityIndex(BaseModel):
    """One authoritative endpoint identity boundary for a Joint asset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    structure_mode: Literal["legacy", "hierarchy", "rigid_body"]
    vocabulary: tuple[str, ...]
    canonical_by_path: dict[str, str]
    ambiguous_candidates_by_path: dict[str, tuple[str, ...]]
    owner_by_prediction_path: dict[str, str]

    def resolve(self, paths: Iterable[str]) -> ArticulationEndpointResolution:
        """Resolve exact aliases without path-name or traversal-order guessing."""
        requested_paths = tuple(
            sorted(
                {
                    path.strip()
                    for path in paths
                    if isinstance(path, str) and path.strip().startswith("/")
                }
            )
        )
        if not requested_paths:
            return ArticulationEndpointResolution(
                outcome="unresolved",
                requested_paths=(),
                reason="missing_endpoint",
            )

        ambiguous_candidates = {
            candidate
            for path in requested_paths
            for candidate in self.ambiguous_candidates_by_path.get(path, ())
        }
        if ambiguous_candidates:
            return ArticulationEndpointResolution(
                outcome="ambiguous",
                requested_paths=requested_paths,
                candidate_paths=tuple(sorted(ambiguous_candidates)),
                reason="ambiguous_source_owner",
            )

        unknown_paths = [
            path for path in requested_paths if path not in self.canonical_by_path
        ]
        if unknown_paths:
            return ArticulationEndpointResolution(
                outcome="unresolved",
                requested_paths=requested_paths,
                candidate_paths=tuple(unknown_paths),
                reason="unknown_endpoint",
            )

        canonical_paths = {self.canonical_by_path[path] for path in requested_paths}
        if len(canonical_paths) != 1:
            return ArticulationEndpointResolution(
                outcome="ambiguous",
                requested_paths=requested_paths,
                candidate_paths=tuple(sorted(canonical_paths)),
                reason="conflicting_canonical_identities",
            )
        return ArticulationEndpointResolution(
            outcome="resolved",
            requested_paths=requested_paths,
            canonical_path=next(iter(canonical_paths)),
            candidate_paths=tuple(sorted(canonical_paths)),
            reason="canonical_identity",
        )

    def unambiguous_aliases(self) -> dict[str, str]:
        """Return only aliases whose source structure proves one owner."""
        return {
            path: canonical
            for path, canonical in self.canonical_by_path.items()
            if path != canonical
        }


def build_articulation_endpoint_identity_index(
    prediction_prim_paths: Sequence[str],
    source_metadata: Mapping[str, Mapping[str, Any]],
) -> ArticulationEndpointIdentityIndex:
    """Build canonical endpoint identities from exported source structure only."""
    source_ids = tuple(sorted(set(prediction_prim_paths)))
    if not source_ids:
        return ArticulationEndpointIdentityIndex(
            structure_mode="legacy",
            vocabulary=(),
            canonical_by_path={},
            ambiguous_candidates_by_path={},
            owner_by_prediction_path={},
        )

    modes = {
        _structure_mode(source_metadata.get(prim_path, {})) for prim_path in source_ids
    }
    modes.discard("legacy")
    if len(modes) > 1:
        # The strict callers separately reject conflicting source modes. Keeping
        # only exact prediction identities here prevents this helper from
        # creating cross-mode aliases before that rejection.
        return _legacy_identity_index(source_ids)
    structure_mode = next(iter(modes), "legacy")
    if structure_mode == "rigid_body":
        return _rigid_body_identity_index(source_ids, source_metadata)
    if structure_mode == "hierarchy":
        return _hierarchy_identity_index(source_ids, source_metadata)
    return _legacy_identity_index(source_ids)


def _legacy_identity_index(
    source_ids: Sequence[str],
) -> ArticulationEndpointIdentityIndex:
    canonical_by_path = {prim_path: prim_path for prim_path in source_ids}
    return ArticulationEndpointIdentityIndex(
        structure_mode="legacy",
        vocabulary=tuple(sorted(canonical_by_path)),
        canonical_by_path=canonical_by_path,
        ambiguous_candidates_by_path={},
        owner_by_prediction_path=canonical_by_path,
    )


def _hierarchy_identity_index(
    source_ids: Sequence[str],
    source_metadata: Mapping[str, Mapping[str, Any]],
) -> ArticulationEndpointIdentityIndex:
    canonical_by_path = {prim_path: prim_path for prim_path in source_ids}
    vocabulary = set(source_ids)
    all_descendants_by_xform: dict[str, set[str]] = {}
    nearest_descendants_by_xform: dict[str, set[str]] = {}
    for prim_path in source_ids:
        structure = _structure_payload(source_metadata.get(prim_path, {}))
        hierarchy_paths = set(_absolute_paths(structure.get("hierarchy_xform_paths")))
        vocabulary.update(hierarchy_paths)
        exported_ancestors = tuple(
            ancestor
            for ancestor in _absolute_paths(
                structure.get("hierarchy_ancestor_xform_paths")
            )
            if ancestor in hierarchy_paths
        )
        for ancestor in exported_ancestors:
            all_descendants_by_xform.setdefault(ancestor, set()).add(prim_path)
        nearest_ancestor = (
            max(exported_ancestors, key=lambda path: (path.count("/"), path))
            if exported_ancestors
            else None
        )
        if nearest_ancestor is not None:
            nearest_descendants_by_xform.setdefault(nearest_ancestor, set()).add(
                prim_path
            )

    ambiguous = {
        xform_path: tuple(sorted(descendants))
        for xform_path, descendants in sorted(all_descendants_by_xform.items())
        if len(descendants) > 1
    }
    for xform_path, descendants in sorted(nearest_descendants_by_xform.items()):
        if xform_path not in ambiguous and len(descendants) == 1:
            canonical_by_path[xform_path] = next(iter(descendants))

    return ArticulationEndpointIdentityIndex(
        structure_mode="hierarchy",
        vocabulary=tuple(sorted(vocabulary)),
        canonical_by_path=canonical_by_path,
        ambiguous_candidates_by_path=ambiguous,
        owner_by_prediction_path={prim_path: prim_path for prim_path in source_ids},
    )


def _rigid_body_identity_index(
    source_ids: Sequence[str],
    source_metadata: Mapping[str, Mapping[str, Any]],
) -> ArticulationEndpointIdentityIndex:
    vocabulary = set(source_ids)
    canonical_by_path: dict[str, str] = {}
    owner_by_prediction_path: dict[str, str] = {}
    for prim_path in source_ids:
        structure = _structure_payload(source_metadata.get(prim_path, {}))
        endpoints = _absolute_paths(structure.get("rigid_body_endpoint_paths"))
        vocabulary.update(endpoints)
        owner = structure.get("rigid_body_owner_path")
        canonical_owner = (
            owner.strip()
            if isinstance(owner, str) and owner.strip().startswith("/")
            else prim_path
        )
        prior_prim_identity = canonical_by_path.get(prim_path)
        prior_owner_identity = canonical_by_path.get(canonical_owner)
        if (
            prior_prim_identity is not None and prior_prim_identity != canonical_owner
        ) or (
            prior_owner_identity is not None and prior_owner_identity != canonical_owner
        ):
            raise ValueError("inconsistent rigid-body owner chain")
        owner_by_prediction_path[prim_path] = canonical_owner
        canonical_by_path[prim_path] = canonical_owner
        canonical_by_path[canonical_owner] = canonical_owner
    for endpoint in vocabulary:
        canonical_by_path.setdefault(endpoint, endpoint)
    return ArticulationEndpointIdentityIndex(
        structure_mode="rigid_body",
        vocabulary=tuple(sorted(vocabulary)),
        canonical_by_path=canonical_by_path,
        ambiguous_candidates_by_path={},
        owner_by_prediction_path=owner_by_prediction_path,
    )


def _structure_payload(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    usd_metadata = metadata.get("usd_metadata")
    return usd_metadata if isinstance(usd_metadata, Mapping) else metadata


def _structure_mode(
    metadata: Mapping[str, Any],
) -> Literal["legacy", "hierarchy", "rigid_body"]:
    provenance = (
        str(_structure_payload(metadata).get("structure_provenance", ""))
        .strip()
        .lower()
    )
    if provenance == "source_hierarchy":
        return "hierarchy"
    if provenance == "source_metadata":
        return "rigid_body"
    return "legacy"


def _absolute_paths(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(
        path
        for path in (item.strip() for item in value if isinstance(item, str))
        if path.startswith("/")
    )
