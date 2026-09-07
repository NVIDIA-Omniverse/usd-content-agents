# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for articulated and contact-rich geometry-only evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from geometry_repair.advanced_profiles import (
    AdjacentLinkExclusion,
    AdvancedProfileEvidenceReport,
    AdvancedProfileRequest,
    ContactRichProbeInput,
    JointSweepInput,
    LinkGeometryMapping,
    SemanticLinkHypothesis,
    SourceEvidence,
    _mesh_pair_intersection,
    collision_unavailable_advanced_profile_report,
    evaluate_advanced_profile,
    evaluate_articulated_geometry,
    evaluate_contact_rich_geometry,
    write_advanced_profile_report,
    write_collision_unavailable_advanced_profile_report,
)
from geometry_repair.models import RepairRequest
from geometry_repair.orchestrator import run_geometry_repair


def _write_box_stage(
    path: Path,
    boxes: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]],
    *,
    open_meshes: set[str] | None = None,
    collision_meshes: set[str] | None = None,
) -> Path:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    for name, (center, size) in boxes.items():
        center_values = np.asarray(center, dtype=np.float64)
        half = np.asarray(size, dtype=np.float64) * 0.5
        minimum = center_values - half
        maximum = center_values + half
        points = [
            (minimum[0], minimum[1], minimum[2]),
            (maximum[0], minimum[1], minimum[2]),
            (maximum[0], maximum[1], minimum[2]),
            (minimum[0], maximum[1], minimum[2]),
            (minimum[0], minimum[1], maximum[2]),
            (maximum[0], minimum[1], maximum[2]),
            (maximum[0], maximum[1], maximum[2]),
            (minimum[0], maximum[1], maximum[2]),
        ]
        mesh_faces = faces[:-2] if name in (open_meshes or set()) else faces
        mesh = UsdGeom.Mesh.Define(stage, f"/Asset/{name}")
        mesh.CreatePointsAttr(
            Vt.Vec3fArray([Gf.Vec3f(*[float(value) for value in point]) for point in points])
        )
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh_faces)))
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray([index for face in mesh_faces for index in face])
        )
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(*[float(value) for value in minimum]),
                    Gf.Vec3f(*[float(value) for value in maximum]),
                ]
            )
        )
        if name in (collision_meshes or set()):
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(
                "convexHull"
            )
    stage.GetRootLayer().Save()
    return path


def _source_evidence(*, inferred: bool = False) -> list[SourceEvidence]:
    return [
        SourceEvidence(
            fact_kind="inferred_hypothesis" if inferred else "source_fact",
            source_ref="source.usd#/Asset",
            summary="source hierarchy and supplied task manifest",
            confidence=0.8 if inferred else 1.0,
        )
    ]


def _link_inputs(
    link_ids: list[str],
    *,
    inferred_link: str | None = None,
) -> tuple[list[SemanticLinkHypothesis], list[LinkGeometryMapping]]:
    hypotheses = [
        SemanticLinkHypothesis(
            link_id=link_id,
            semantic_label=link_id,
            source_part_paths=[f"/Asset/{link_id}_render"],
            confidence=0.8 if link_id == inferred_link else 1.0,
            evidence=_source_evidence(inferred=link_id == inferred_link),
        )
        for link_id in link_ids
    ]
    mappings = [
        LinkGeometryMapping(
            link_id=link_id,
            render_paths=[f"/Asset/{link_id}_render"],
            collision_paths=[f"/Asset/{link_id}_collision"],
            evidence=_source_evidence(),
        )
        for link_id in link_ids
    ]
    return hypotheses, mappings


def _joint() -> JointSweepInput:
    return JointSweepInput(
        joint_id="hinge",
        parent_link_id="base",
        child_link_id="arm",
        moving_link_ids=["arm"],
        joint_type="revolute",
        axis_world=(0.0, 0.0, 1.0),
        origin_world_m=(0.0, 0.0, 0.0),
        lower_limit=0.0,
        upper_limit=math.pi / 2.0,
        reference_value=0.0,
        units="radians",
        sample_count=5,
        evidence=_source_evidence(),
    )


def _exclusion() -> AdjacentLinkExclusion:
    return AdjacentLinkExclusion(
        link_a="base",
        link_b="arm",
        reason="source joint metadata disables adjacent-link self collision",
        evidence=_source_evidence(),
    )


def test_articulated_geometry_passes_with_explicit_authoritative_inputs(tmp_path: Path) -> None:
    render = _write_box_stage(
        tmp_path / "render.usda",
        {
            "base_render": ((0.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
            "arm_render": ((1.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
        },
    )
    collision = _write_box_stage(
        tmp_path / "collision.usda",
        {
            "base_collision": ((0.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
            "arm_collision": ((1.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
        },
    )
    hypotheses, mappings = _link_inputs(["base", "arm"])

    result = evaluate_articulated_geometry(
        render,
        collision,
        semantic_links=hypotheses,
        link_mappings=mappings,
        joints=[_joint()],
        adjacent_link_exclusions=[_exclusion()],
    )

    assert result.status == "pass"
    assert result.certification_eligible is True
    assert all(check.status == "pass" for check in result.link_collision_checks)
    assert result.swept_motion_checks[0].status == "pass"
    assert result.swept_motion_checks[0].evaluated_sample_count == 5
    assert result.swept_motion_checks[0].excluded_link_pairs[0]["links"] == ["arm", "base"]
    assert "runtime-validation" in " ".join(result.downstream_owners)


def test_orchestrator_emits_geometry_only_articulated_evidence(tmp_path: Path) -> None:
    source = _write_box_stage(
        tmp_path / "source.usda",
        {
            "base_render": ((0.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
            "arm_render": ((1.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
            "base_collision": ((0.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
            "arm_collision": ((1.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
        },
        collision_meshes={"base_collision", "arm_collision"},
    )
    hypotheses, mappings = _link_inputs(["base", "arm"])
    advanced = AdvancedProfileRequest(
        profile="articulated_rigid",
        semantic_links=hypotheses,
        link_mappings=mappings,
        joints=[_joint()],
        adjacent_link_exclusions=[_exclusion()],
    )

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="articulated_rigid",
            mode="auto",
            advanced_profile=advanced.model_dump(mode="json"),
        )
    )

    assert result.outcome == "conditional"
    report = AdvancedProfileEvidenceReport.model_validate_json(
        Path(result.advanced_profile_evidence_path or "").read_text(encoding="utf-8")
    )
    assert report.status == "pass"
    assert report.disposition == "evidence_complete"
    assert report.claim_scope == "geometry_repair.articulated_rigid.geometry_only"
    assert report.handoff is not None
    handoff = report.handoff
    assert handoff.downstream_status == "not_evaluated"
    assert handoff.downstream_disposition == "conditional"
    assert [item.link_id for item in handoff.link_artifacts] == ["arm", "base"]
    assert [route.owner for route in handoff.downstream_routes] == [
        "articulation",
        "physics",
        "runtime_validation",
        "simready",
    ]
    assert [route.readiness for route in handoff.downstream_routes] == [
        "ready",
        "ready",
        "conditional",
        "conditional",
    ]
    assert all(route.status == "not_evaluated" for route in handoff.downstream_routes)
    assert all(route.result_claimed is False for route in handoff.downstream_routes)
    for link in handoff.link_artifacts:
        assert link.render_artifact.availability == "available"
        assert link.collision_artifact.availability == "available"
        assert link.render_artifact.sha256
        assert link.collision_artifact.sha256
        assert link.render_artifact.prim_paths == [f"/Asset/{link.link_id}_render"]
        assert link.collision_artifact.prim_paths == [f"/Asset/{link.link_id}_collision"]
    certificate = json.loads(Path(result.certificate_path).read_text(encoding="utf-8"))
    assert certificate["claim_scope"] == "geometry_repair.articulated_rigid.geometry_only"
    assert certificate["validation_results"]["advanced_profile_geometry"] == "pass"


def test_articulated_geometry_detects_non_adjacent_swept_collision(tmp_path: Path) -> None:
    boxes = {
        "base_render": ((0.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
        "arm_render": ((1.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
        "obstacle_render": ((0.0, 1.0, 0.0), (0.4, 0.4, 0.4)),
    }
    render = _write_box_stage(tmp_path / "render.usda", boxes)
    collision = _write_box_stage(
        tmp_path / "collision.usda",
        {name.replace("render", "collision"): value for name, value in boxes.items()},
    )
    hypotheses, mappings = _link_inputs(["base", "arm", "obstacle"])

    result = evaluate_articulated_geometry(
        render,
        collision,
        semantic_links=hypotheses,
        link_mappings=mappings,
        joints=[_joint()],
        adjacent_link_exclusions=[_exclusion()],
    )

    assert result.status == "fail"
    assert result.certification_eligible is False
    assert result.swept_motion_checks[0].status == "fail"
    assert any(
        event["stationary_link"] == "obstacle"
        for event in result.swept_motion_checks[0].collision_events
    )


def test_articulated_geometry_rejects_invalid_per_link_collision(tmp_path: Path) -> None:
    render = _write_box_stage(
        tmp_path / "render.usda",
        {
            "base_render": ((0.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
            "arm_render": ((1.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
        },
    )
    collision = _write_box_stage(
        tmp_path / "collision.usda",
        {
            "base_collision": ((0.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
            "arm_collision": ((1.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
        },
        open_meshes={"arm_collision"},
    )
    hypotheses, mappings = _link_inputs(["base", "arm"])

    result = evaluate_articulated_geometry(
        render,
        collision,
        semantic_links=hypotheses,
        link_mappings=mappings,
        joints=[_joint()],
        adjacent_link_exclusions=[_exclusion()],
    )

    assert result.status == "fail"
    assert any(
        "not watertight" in failure
        for failure in result.link_collision_checks[0].failures
        + result.link_collision_checks[1].failures
    )
    assert result.swept_motion_checks[0].status == "not_evaluated"


def test_inferred_link_hypothesis_cannot_make_articulated_evidence_certifiable(
    tmp_path: Path,
) -> None:
    render = _write_box_stage(
        tmp_path / "render.usda",
        {
            "base_render": ((0.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
            "arm_render": ((1.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
        },
    )
    collision = _write_box_stage(
        tmp_path / "collision.usda",
        {
            "base_collision": ((0.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
            "arm_collision": ((1.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
        },
    )
    hypotheses, mappings = _link_inputs(["base", "arm"], inferred_link="arm")

    result = evaluate_articulated_geometry(
        render,
        collision,
        semantic_links=hypotheses,
        link_mappings=mappings,
        joints=[_joint()],
        adjacent_link_exclusions=[_exclusion()],
    )

    assert result.status == "pass"
    assert result.certification_eligible is False
    assert "inferred hypotheses" in result.blockers[0]


def test_joint_geometry_is_never_inferred_or_silently_defaulted(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SemanticLinkHypothesis.model_validate(
            {
                "link_id": "door",
                "semantic_label": "door",
                "source_part_paths": ["/Door"],
                "confidence": 1.0,
                "evidence": [item.model_dump() for item in _source_evidence()],
                "joint_axis": [0.0, 0.0, 1.0],
            }
        )

    render = _write_box_stage(
        tmp_path / "render.usda",
        {"base_render": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))},
    )
    collision = _write_box_stage(
        tmp_path / "collision.usda",
        {"base_collision": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))},
    )
    hypotheses, mappings = _link_inputs(["base"])
    result = evaluate_articulated_geometry(
        render,
        collision,
        semantic_links=hypotheses,
        link_mappings=mappings,
        joints=[],
    )

    assert result.status == "not_evaluated"
    assert result.certification_eligible is False
    assert "caller-supplied joint geometry" in " ".join(result.blockers)


def test_missing_joint_geometry_yields_serializable_conditional_report(tmp_path: Path) -> None:
    render = _write_box_stage(
        tmp_path / "render.usda",
        {
            "base_render": ((0.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
            "arm_render": ((1.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
        },
    )
    collision = _write_box_stage(
        tmp_path / "collision.usda",
        {
            "base_collision": ((0.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
            "arm_collision": ((1.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
        },
    )
    hypotheses, mappings = _link_inputs(["base", "arm"])
    incomplete_joint = JointSweepInput(
        joint_id="hinge",
        parent_link_id="base",
        child_link_id="arm",
        moving_link_ids=["arm"],
        joint_type="revolute",
        sample_count=5,
        evidence=_source_evidence(),
    )
    request = AdvancedProfileRequest(
        profile="articulated_rigid",
        semantic_links=hypotheses,
        link_mappings=mappings,
        joints=[incomplete_joint],
        adjacent_link_exclusions=[_exclusion()],
    )

    report = evaluate_advanced_profile(render, collision, request)
    round_trip = AdvancedProfileEvidenceReport.model_validate_json(report.model_dump_json())
    report_path = write_advanced_profile_report(report, tmp_path / "advanced_profile_report.json")
    persisted = AdvancedProfileEvidenceReport.model_validate(
        json.loads(report_path.read_text(encoding="utf-8"))
    )

    assert report.status == "not_evaluated"
    assert report.disposition == "conditional"
    assert report.articulated is not None
    assert report.articulated.swept_motion_checks[0].status == "not_evaluated"
    assert "axis_world" in " ".join(report.articulated.swept_motion_checks[0].warnings)
    assert round_trip == report
    assert persisted == report


def test_swept_intersection_reports_resource_indeterminate() -> None:
    left_vertices = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    left_triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
    right_vertices = np.asarray([[2.0, 2.0, 0.0], [2.0, 0.6, 0.0], [0.6, 2.0, 0.0]])
    right_triangles = np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64)

    status, intersects, tested, reason = _mesh_pair_intersection(
        left_vertices,
        left_triangles,
        right_vertices,
        right_triangles,
        candidate_budget=1,
    )

    assert status == "indeterminate"
    assert intersects is False
    assert tested == 1
    assert reason and "exceeds limit" in reason


def _receiver_stage(path: Path, *, open_side: bool = False) -> Path:
    return _write_box_stage(
        path,
        {
            "left_collision": ((-0.75, 0.0, 0.5), (0.5, 1.0, 3.0)),
            "right_collision": ((0.75, 0.0, 0.5), (0.5, 1.0, 3.0)),
            "stop_collision": ((0.0, 0.0, -0.5), (1.0, 1.0, 0.5)),
        },
        open_meshes={"left_collision"} if open_side else None,
    )


def _contact_probe(
    *,
    representation: str = "sdf",
    voxel_size_m: float | None = 0.0005,
    inferred: bool = False,
    path_points: list[tuple[float, float, float]] | None = None,
) -> ContactRichProbeInput:
    return ContactRichProbeInput(
        probe_id="plug_to_socket",
        moving_part_id="plug",
        receiver_part_id="socket",
        receiver_collision_paths=[
            "/Asset/left_collision",
            "/Asset/right_collision",
            "/Asset/stop_collision",
        ],
        axis_origin_world_m=(0.0, 0.0, 0.0),
        axis_world=(0.0, 0.0, 1.0),
        approach_direction="against_axis",
        path_points_world_m=path_points or [(0.0, 0.0, 1.0), (0.0, 0.0, 0.0)],
        axis_tolerance_m=0.01,
        angular_tolerance_deg=2.0,
        moving_envelope_radius_m=0.2,
        required_radial_clearance_m=0.04,
        seated_stop_point_world_m=(0.0, 0.0, -0.25),
        seated_stop_tolerance_m=1e-5,
        minimum_protected_feature_m=0.003,
        collision_representation=representation,
        sdf_voxel_size_m=voxel_size_m,
        required_voxels_across_feature=3,
        sample_step_m=0.05,
        evidence=_source_evidence(inferred=inferred),
    )


def test_contact_rich_geometry_passes_axis_path_clearance_stop_and_sdf(tmp_path: Path) -> None:
    receiver = _receiver_stage(tmp_path / "receiver.usda")
    probe = _contact_probe()

    result = evaluate_contact_rich_geometry(receiver, probes=[probe])
    report = evaluate_advanced_profile(
        receiver,
        receiver,
        AdvancedProfileRequest(profile="contact_rich", contact_probes=[probe]),
    )

    assert result.status == "pass"
    assert result.certification_eligible is True
    evidence = result.probe_evidence[0]
    assert evidence.status == "pass"
    assert all(check.status == "pass" for check in evidence.checks)
    sdf = next(check for check in evidence.checks if check.check_id.startswith("sdf_resolution"))
    assert sdf.metrics["maximum_voxel_size_m"] == pytest.approx(0.001)
    assert report.handoff is not None
    handoff = report.handoff
    assert handoff.claim_scope == "geometry_repair.contact_rich.geometry_only"
    assert len(handoff.probe_artifacts) == 1
    artifact = handoff.probe_artifacts[0]
    assert artifact.probe_id == "plug_to_socket"
    assert artifact.receiver_selection_status == "pass"
    assert artifact.receiver_collision_artifact.availability == "available"
    assert artifact.receiver_collision_artifact.prim_paths == sorted(
        probe.receiver_collision_paths or []
    )
    assert [route.readiness for route in handoff.downstream_routes] == [
        "ready",
        "ready",
        "conditional",
        "conditional",
    ]


def test_contact_rich_static_mesh_applies_bounded_clearance_erosion(tmp_path: Path) -> None:
    receiver = _receiver_stage(tmp_path / "receiver.usda")
    probe = _contact_probe(
        representation="static_triangle_mesh",
        voxel_size_m=None,
    ).model_copy(update={"maximum_clearance_erosion_m": 0.01})

    result = evaluate_contact_rich_geometry(receiver, probes=[probe])

    assert result.status == "pass"
    clearance = next(
        check
        for check in result.probe_evidence[0].checks
        if check.check_id.startswith("contact_clearance")
    )
    assert clearance.metrics["required_radial_clearance_m"] == pytest.approx(0.04)
    assert clearance.metrics["maximum_clearance_erosion_m"] == pytest.approx(0.01)
    assert clearance.metrics["minimum_allowed_clearance_m"] == pytest.approx(0.03)
    assert clearance.metrics["realized_radial_clearance_m"] >= 0.03


def test_contact_rich_geometry_fails_misalignment_and_coarse_sdf(tmp_path: Path) -> None:
    receiver = _receiver_stage(tmp_path / "receiver.usda")
    probe = _contact_probe(
        voxel_size_m=0.002,
        path_points=[(0.0, 0.0, 1.0), (0.2, 0.0, 0.0)],
    )

    result = evaluate_contact_rich_geometry(receiver, probes=[probe])

    assert result.status == "fail"
    checks = result.probe_evidence[0].checks
    assert (
        next(check for check in checks if check.check_id.startswith("contact_axis")).status
        == "fail"
    )
    assert (
        next(check for check in checks if check.check_id.startswith("sdf_resolution")).status
        == "fail"
    )
    assert result.certification_eligible is False


def test_contact_path_is_indeterminate_when_receiver_occupancy_is_unavailable(
    tmp_path: Path,
) -> None:
    receiver = _receiver_stage(tmp_path / "receiver.usda", open_side=True)

    result = evaluate_contact_rich_geometry(receiver, probes=[_contact_probe()])

    assert result.status == "indeterminate"
    path = next(
        check
        for check in result.probe_evidence[0].checks
        if check.check_id.startswith("contact_path")
    )
    assert path.status == "indeterminate"
    assert "watertight" in " ".join(path.warnings)


def test_convex_contact_probe_does_not_require_sdf_and_inference_cannot_certify(
    tmp_path: Path,
) -> None:
    receiver = _receiver_stage(tmp_path / "receiver.usda")
    probe = _contact_probe(representation="convex", voxel_size_m=None, inferred=True)

    result = evaluate_contact_rich_geometry(receiver, probes=[probe])

    assert result.status == "pass"
    sdf = next(
        check
        for check in result.probe_evidence[0].checks
        if check.check_id.startswith("sdf_resolution")
    )
    assert sdf.required is False
    assert sdf.status == "not_evaluated"
    assert result.certification_eligible is False
    assert "inferred hypotheses" in result.blockers[0]


def test_missing_sdf_resolution_and_missing_probes_are_not_evaluated(tmp_path: Path) -> None:
    receiver = _receiver_stage(tmp_path / "receiver.usda")

    missing_resolution = evaluate_contact_rich_geometry(
        receiver,
        probes=[_contact_probe(voxel_size_m=None)],
    )
    no_probes = evaluate_contact_rich_geometry(receiver, probes=[])

    assert missing_resolution.status == "not_evaluated"
    assert missing_resolution.certification_eligible is False
    assert no_probes.status == "not_evaluated"
    assert no_probes.certification_eligible is False


def test_missing_contact_path_yields_conditional_report_not_validation_failure(
    tmp_path: Path,
) -> None:
    receiver = _receiver_stage(tmp_path / "receiver.usda")
    probe = _contact_probe().model_copy(update={"path_points_world_m": None})
    request = AdvancedProfileRequest(profile="contact_rich", contact_probes=[probe])

    report = evaluate_advanced_profile(receiver, receiver, request)

    assert report.status == "not_evaluated"
    assert report.disposition == "conditional"
    assert report.contact_rich is not None
    assert report.contact_rich.probe_evidence[0].sampled_path_points == 0
    checks = report.contact_rich.probe_evidence[0].checks
    assert checks[0].check_id.startswith("receiver_collision_selection")
    assert checks[0].status == "pass"
    assert all(check.status == "not_evaluated" for check in checks[1:])
    assert "path_points_world_m" in report.model_dump_json()


def test_contact_queries_only_explicit_receiver_collision_prims(tmp_path: Path) -> None:
    receiver_with_moving = _write_box_stage(
        tmp_path / "combined_collision.usda",
        {
            "left_collision": ((-0.75, 0.0, 0.5), (0.5, 1.0, 3.0)),
            "right_collision": ((0.75, 0.0, 0.5), (0.5, 1.0, 3.0)),
            "stop_collision": ((0.0, 0.0, -0.5), (1.0, 1.0, 0.5)),
            # Querying this moving part as receiver geometry would block the path.
            "moving_collision": ((0.0, 0.0, 0.5), (0.3, 0.3, 0.3)),
        },
    )

    result = evaluate_contact_rich_geometry(
        receiver_with_moving,
        probes=[_contact_probe()],
    )

    assert result.status == "pass"
    selection = result.probe_evidence[0].checks[0]
    assert selection.status == "pass"
    assert selection.metrics["selected_paths"] == [
        "/Asset/left_collision",
        "/Asset/right_collision",
        "/Asset/stop_collision",
    ]
    assert "/Asset/moving_collision" not in selection.metrics["selected_paths"]
    assert [item.name for item in tmp_path.iterdir()] == ["combined_collision.usda"]


def test_receiver_collision_selection_missing_or_ambiguous_is_fail_closed(
    tmp_path: Path,
) -> None:
    collision = _receiver_stage(tmp_path / "receiver.usda")
    absent = _contact_probe().model_copy(update={"receiver_collision_paths": None})
    duplicated = _contact_probe().model_copy(
        update={
            "receiver_collision_paths": [
                "/Asset/left_collision",
                "/Asset/left_collision",
            ]
        }
    )
    unresolved = _contact_probe().model_copy(
        update={"receiver_collision_paths": ["/Asset/missing_collision"]}
    )

    absent_result = evaluate_contact_rich_geometry(collision, probes=[absent])
    duplicate_result = evaluate_contact_rich_geometry(collision, probes=[duplicated])
    unresolved_result = evaluate_contact_rich_geometry(collision, probes=[unresolved])

    assert absent_result.status == "not_evaluated"
    assert absent_result.probe_evidence[0].checks[0].status == "not_evaluated"
    assert duplicate_result.status == "fail"
    assert "ambiguous" in " ".join(duplicate_result.probe_evidence[0].checks[0].failures)
    assert unresolved_result.status == "fail"
    assert "was not found" in " ".join(unresolved_result.probe_evidence[0].checks[0].failures)


def test_collision_unavailable_helper_serializes_conditional_phase3_evidence(
    tmp_path: Path,
) -> None:
    request = AdvancedProfileRequest(
        profile="contact_rich",
        contact_probes=[_contact_probe()],
    )
    report = collision_unavailable_advanced_profile_report(
        request,
        reason="accepted candidate has no collision artifact",
    )
    output = write_collision_unavailable_advanced_profile_report(
        request,
        tmp_path / "advanced_profile_unavailable.json",
        reason="accepted candidate has no collision artifact",
    )
    persisted = AdvancedProfileEvidenceReport.model_validate_json(
        output.read_text(encoding="utf-8")
    )
    hypotheses, mappings = _link_inputs(["base", "arm"])
    articulated_report = collision_unavailable_advanced_profile_report(
        AdvancedProfileRequest(
            profile="articulated_rigid",
            semantic_links=hypotheses,
            link_mappings=mappings,
            joints=[_joint()],
            adjacent_link_exclusions=[_exclusion()],
        ),
        reason="accepted candidate has no collision artifact",
    )

    assert report.status == "not_evaluated"
    assert report.disposition == "conditional"
    assert report.contact_rich is not None
    assert report.contact_rich.receiver_collision_path is None
    assert all(
        check.status == "not_evaluated" for check in report.contact_rich.probe_evidence[0].checks
    )
    assert persisted == report
    assert articulated_report.status == "not_evaluated"
    assert articulated_report.disposition == "conditional"
    assert articulated_report.articulated is not None
    assert articulated_report.articulated.swept_motion_checks[0].status == "not_evaluated"
