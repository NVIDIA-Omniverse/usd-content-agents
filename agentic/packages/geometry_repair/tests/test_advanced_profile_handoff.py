# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused contract tests for geometry-owned advanced-profile handoffs."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from geometry_repair.advanced_profiles import (
    AdvancedProfileEvidenceReport,
    AdvancedProfileRequest,
    ContactRichProbeInput,
    JointSweepInput,
    LinkGeometryMapping,
    SemanticLinkHypothesis,
    SourceEvidence,
    collision_unavailable_advanced_profile_report,
)


def _evidence(source_ref: str) -> list[SourceEvidence]:
    return [
        SourceEvidence(
            fact_kind="source_fact",
            source_ref=source_ref,
            summary="authoritative test input",
            confidence=1.0,
        )
    ]


def _articulated_request() -> AdvancedProfileRequest:
    link_ids = ["base", "arm"]
    return AdvancedProfileRequest(
        profile="articulated_rigid",
        semantic_links=[
            SemanticLinkHypothesis(
                link_id=link_id,
                semantic_label=link_id,
                source_part_paths=[f"/Source/{link_id}"],
                confidence=1.0,
                evidence=_evidence(f"source.usda#/Source/{link_id}"),
            )
            for link_id in link_ids
        ],
        link_mappings=[
            LinkGeometryMapping(
                link_id=link_id,
                render_paths=[f"/Asset/{link_id}_render"],
                collision_paths=[f"/Asset/{link_id}_collision"],
                evidence=_evidence(f"mapping.json#/{link_id}"),
            )
            for link_id in link_ids
        ],
        joints=[
            JointSweepInput(
                joint_id="source_hinge",
                parent_link_id="base",
                child_link_id="arm",
                moving_link_ids=["arm"],
                joint_type="revolute",
                sample_count=3,
                evidence=_evidence("source.usda#/Source/source_hinge"),
            )
        ],
    )


def _contact_probe(probe_id: str) -> ContactRichProbeInput:
    return ContactRichProbeInput(
        probe_id=probe_id,
        moving_part_id=f"{probe_id}_moving",
        receiver_part_id=f"{probe_id}_receiver",
        receiver_collision_paths=[
            f"/Asset/{probe_id}_receiver_b",
            f"/Asset/{probe_id}_receiver_a",
        ],
        approach_direction="along_axis",
        axis_tolerance_m=0.001,
        angular_tolerance_deg=1.0,
        moving_envelope_radius_m=0.01,
        required_radial_clearance_m=0.001,
        seated_stop_tolerance_m=0.001,
        minimum_protected_feature_m=0.003,
        collision_representation="sdf",
        sample_step_m=0.001,
        evidence=_evidence(f"task.json#/probes/{probe_id}"),
    )


def test_articulated_handoff_has_per_link_artifacts_and_strict_downstream_routes(
    tmp_path: Path,
) -> None:
    report = collision_unavailable_advanced_profile_report(
        _articulated_request(),
        reason="accepted collision evidence is missing",
        collision_path=tmp_path / "missing-collision.usda",
    )

    assert report.claim_scope == "geometry_repair.articulated_rigid.geometry_only"
    assert report.handoff is not None
    handoff = report.handoff
    assert handoff.geometry_status == "not_evaluated"
    assert handoff.geometry_disposition == "conditional"
    assert handoff.downstream_status == "not_evaluated"
    assert handoff.downstream_disposition == "conditional"
    assert [item.link_id for item in handoff.link_artifacts] == ["arm", "base"]
    assert [route.owner for route in handoff.downstream_routes] == [
        "articulation",
        "physics",
        "runtime_validation",
        "simready",
    ]

    artifact_ids = []
    for item in handoff.link_artifacts:
        assert item.mapping_status == "not_evaluated"
        assert item.render_artifact.prim_paths == [f"/Asset/{item.link_id}_render"]
        assert item.collision_artifact.prim_paths == [f"/Asset/{item.link_id}_collision"]
        assert item.render_artifact.availability == "not_evaluated"
        assert item.collision_artifact.availability == "not_evaluated"
        artifact_ids.extend(
            [item.render_artifact.reference_id, item.collision_artifact.reference_id]
        )

    for route in handoff.downstream_routes:
        assert route.status == "not_evaluated"
        assert route.readiness == "conditional"
        assert route.result_claimed is False
        assert route.upstream_claim_scope == handoff.claim_scope
        assert route.artifact_reference_ids == artifact_ids
        assert route.blockers

    route_payload = handoff.downstream_routes[0].model_dump(mode="json")
    with pytest.raises(ValidationError, match="not_evaluated"):
        handoff.downstream_routes[0].__class__.model_validate({**route_payload, "status": "pass"})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        handoff.downstream_routes[0].__class__.model_validate(
            {**route_payload, "final_status": "pass"}
        )
    incomplete_handoff = handoff.model_dump(mode="json")
    incomplete_handoff["downstream_routes"][0]["artifact_reference_ids"] = artifact_ids[:-1]
    with pytest.raises(ValidationError, match="every handoff artifact"):
        handoff.__class__.model_validate(incomplete_handoff)


def test_contact_handoff_has_deterministic_per_probe_references_without_defaults() -> None:
    request = AdvancedProfileRequest(
        profile="contact_rich",
        contact_probes=[_contact_probe("probe_z"), _contact_probe("probe_a")],
    )

    report = collision_unavailable_advanced_profile_report(
        request,
        reason="accepted collision evidence is missing",
    )

    assert report.status == "not_evaluated"
    assert report.disposition == "conditional"
    assert report.handoff is not None
    handoff = report.handoff
    assert handoff.claim_scope == "geometry_repair.contact_rich.geometry_only"
    assert [item.probe_id for item in handoff.probe_artifacts] == ["probe_a", "probe_z"]
    for item in handoff.probe_artifacts:
        assert item.receiver_selection_status == "not_evaluated"
        assert item.receiver_collision_artifact.availability == "not_evaluated"
        assert item.receiver_collision_artifact.prim_paths == [
            f"/Asset/{item.probe_id}_receiver_a",
            f"/Asset/{item.probe_id}_receiver_b",
        ]
        assert item.geometry_check_ids == [
            f"receiver_collision_selection:{item.probe_id}",
            f"contact_axis:{item.probe_id}",
            f"contact_approach:{item.probe_id}",
            f"contact_path:{item.probe_id}",
            f"contact_clearance:{item.probe_id}",
            f"seated_stop:{item.probe_id}",
            f"sdf_resolution:{item.probe_id}",
        ]
    assert all(route.status == "not_evaluated" for route in handoff.downstream_routes)
    assert all(route.readiness == "conditional" for route in handoff.downstream_routes)


def test_semantic_link_without_mapping_remains_explicitly_not_evaluated() -> None:
    request = AdvancedProfileRequest(
        profile="articulated_rigid",
        semantic_links=[
            SemanticLinkHypothesis(
                link_id="unmapped_link",
                semantic_label="unmapped link",
                source_part_paths=["/Source/unmapped_link"],
                confidence=1.0,
                evidence=_evidence("source.usda#/Source/unmapped_link"),
            )
        ],
    )

    report = collision_unavailable_advanced_profile_report(
        request,
        reason="accepted collision evidence is missing",
    )

    assert report.handoff is not None
    assert len(report.handoff.link_artifacts) == 1
    link = report.handoff.link_artifacts[0]
    assert link.link_id == "unmapped_link"
    assert link.mapping_status == "not_evaluated"
    assert link.render_artifact.prim_paths == []
    assert link.collision_artifact.prim_paths == []
    assert link.render_artifact.availability == "not_evaluated"
    assert link.collision_artifact.availability == "not_evaluated"


def test_legacy_report_without_additive_handoff_fields_still_validates() -> None:
    report = collision_unavailable_advanced_profile_report(
        AdvancedProfileRequest(profile="contact_rich", contact_probes=[_contact_probe("probe")]),
        reason="accepted collision evidence is missing",
    )
    legacy_payload = report.model_dump(mode="json")
    legacy_payload.pop("claim_scope")
    legacy_payload.pop("handoff")

    restored = AdvancedProfileEvidenceReport.model_validate(legacy_payload)

    assert restored.claim_scope == "geometry_repair.contact_rich.geometry_only"
    assert restored.handoff is None
    assert restored.status == "not_evaluated"
    assert restored.disposition == "conditional"
