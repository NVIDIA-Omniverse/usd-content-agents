# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from content_agent_workflows.material_assignment.decisions import (
    validate_material_decision,
)
from content_agent_workflows.material_assignment.manifest import (
    MaterialManifestEntry,
    ResolvedMaterialManifest,
)


def _manifest(tmp_path: Path) -> ResolvedMaterialManifest:
    return ResolvedMaterialManifest(
        manifest_path=tmp_path / "materials.yaml",
        library_path=tmp_path / "materials.usda",
        entries=(
            MaterialManifestEntry(
                name="Red",
                description="Red paint",
                binding_path="/Looks/Red",
                tags=("red", "paint"),
            ),
        ),
    )


def _candidates() -> dict[str, object]:
    return {
        "schema_version": "content-agents.visible-candidate-prims.v1",
        "path_space": "source",
        "candidate_visible_prim_count": 2,
        "candidates": [
            {"source_path": "/World/A"},
            {"source_path": "/World/B"},
        ],
    }


def test_decision_requires_exact_authoritative_candidate_coverage(
    tmp_path: Path,
) -> None:
    result = validate_material_decision(
        {
            "schema_version": "content-agents.material-decision-patch.v1",
            "material_assignments": [
                {
                    "material_name": "Red",
                    "material_path": "/Looks/Red",
                    "prim_paths": ["/World/A", "/World/B"],
                }
            ],
            "reviewed_no_override": [],
        },
        candidates=_candidates(),
        manifest=_manifest(tmp_path),
        clear_materials=True,
    )

    assert result.valid
    assert result.claimed_candidate_paths == ("/World/A", "/World/B")


def test_decision_rejects_unknown_material_duplicate_and_missing_candidate(
    tmp_path: Path,
) -> None:
    result = validate_material_decision(
        {
            "schema_version": "content-agents.material-decision-patch.v1",
            "material_assignments": [
                {
                    "material_name": "Unknown",
                    "material_path": "/Looks/Wrong",
                    "prim_paths": ["/World/A", "/World/A"],
                }
            ],
            "reviewed_no_override": [],
        },
        candidates=_candidates(),
        manifest=_manifest(tmp_path),
        clear_materials=True,
    )

    assert not result.valid
    assert {error.code for error in result.errors} >= {
        "unknown_material",
        "duplicate_decision_path",
        "missing_candidate_decision",
    }


def test_clean_slate_rejects_reviewed_no_override(tmp_path: Path) -> None:
    result = validate_material_decision(
        {
            "schema_version": "content-agents.material-decision-patch.v1",
            "material_assignments": [
                {"material_name": "Red", "prim_paths": ["/World/A"]}
            ],
            "reviewed_no_override": [{"prim_paths": ["/World/B"]}],
        },
        candidates=_candidates(),
        manifest=_manifest(tmp_path),
        clear_materials=True,
    )

    assert not result.valid
    assert "clean_slate_reviewed_no_override" in {error.code for error in result.errors}


def test_zero_candidates_must_be_explicit(tmp_path: Path) -> None:
    candidates = {
        "schema_version": "content-agents.visible-candidate-prims.v1",
        "path_space": "source",
        "candidate_visible_prim_count": 0,
        "candidates": [],
    }
    implicit = validate_material_decision(
        {
            "schema_version": "content-agents.material-decision-patch.v1",
            "material_assignments": [],
            "reviewed_no_override": [],
        },
        candidates=candidates,
        manifest=_manifest(tmp_path),
        clear_materials=True,
    )
    explicit = validate_material_decision(
        {
            "schema_version": "content-agents.material-decision-patch.v1",
            "candidate_count": 0,
            "material_assignments": [],
            "reviewed_no_override": [],
        },
        candidates=candidates,
        manifest=_manifest(tmp_path),
        clear_materials=True,
    )

    assert not implicit.valid
    assert explicit.valid


def test_decision_rejects_unversioned_or_unknown_candidate_space(
    tmp_path: Path,
) -> None:
    candidates = _candidates()
    candidates.pop("schema_version")
    candidates["path_space"] = "guessed"

    result = validate_material_decision(
        {
            "schema_version": "content-agents.material-decision-patch.v1",
            "material_assignments": [
                {
                    "material_name": "Red",
                    "material_path": "/Looks/Red",
                    "prim_paths": ["/World/A", "/World/B"],
                }
            ],
            "reviewed_no_override": [],
        },
        candidates=candidates,
        manifest=_manifest(tmp_path),
        clear_materials=True,
    )

    assert not result.valid
    assert {error.code for error in result.errors} >= {
        "invalid_candidate_schema_version",
        "invalid_candidate_path_space",
    }
