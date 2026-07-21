# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for validation evidence workflow schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from content_agent_workflows.common import (
    VALIDATION_CHECK_TAXONOMY,
    EvidenceArtifact,
    ValidationEvidence,
    material_assignment_validation_evidence,
)


def test_validation_evidence_accepts_tiered_runtime_result() -> None:
    evidence = ValidationEvidence(
        workflow="runtime_validation",
        asset="/assets/robot.usd",
        target_runtime="isaac-sim",
        validation_tier="T1_basic_stability",
        sim_ready_status="conditional",
        warnings=["Physics checks were not evaluated."],
    )

    payload = evidence.model_dump()
    assert payload["schema_version"] == "content-agent-workflows.validation-evidence.v1"
    assert payload["target_runtime"] == "isaac-sim"
    assert "visual_materials" in VALIDATION_CHECK_TAXONOMY


def test_validation_evidence_rejects_unknown_tier() -> None:
    with pytest.raises(ValidationError):
        ValidationEvidence(
            workflow="runtime_validation",
            asset="/assets/robot.usd",
            target_runtime="isaac-sim",
            validation_tier="T4_unknown",
        )


def test_material_assignment_validation_evidence_maps_unresolved_status() -> None:
    evidence = material_assignment_validation_evidence(
        asset="/assets/agv.usd",
        target_runtime="isaac-sim",
        visual_materials_status="warning",
        evidence_artifacts=[
            EvidenceArtifact(
                kind="render",
                path="final_renders/final_oblique.png",
                description="Final material verification render.",
            )
        ],
        unresolved_issues=["Wheel rubber finish is approximate."],
    )

    assert evidence.workflow == "material_assignment"
    assert evidence.sim_ready_status == "conditional"
    assert evidence.checks[0].name == "visual_materials"
    assert evidence.evidence_artifacts[0].path == "final_renders/final_oblique.png"


@pytest.mark.parametrize(
    ("visual_materials_status", "expected_sim_ready_status"),
    [
        ("pass", "pass"),
        ("warning", "conditional"),
        ("fail", "fail"),
        ("not_evaluated", "not_evaluated"),
    ],
)
def test_material_assignment_validation_evidence_maps_visual_status(
    visual_materials_status: str,
    expected_sim_ready_status: str,
) -> None:
    evidence = material_assignment_validation_evidence(
        asset="/assets/agv.usd",
        target_runtime="isaac-sim",
        visual_materials_status=visual_materials_status,
    )

    assert evidence.sim_ready_status == expected_sim_ready_status
    assert evidence.checks[0].status == visual_materials_status
