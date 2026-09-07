# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Role-scoped diagnosis, repair-intent, and certificate contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from geometry_repair.diagnosis import diagnose_asset
from geometry_repair.models import (
    DiagnosisIssue,
    RepairBudgets,
    RepairIntent,
    RepairOperation,
    RepairRequest,
)
from geometry_repair.orchestrator import _plan_repair
from geometry_repair.repair_intent import (
    rank_operations_from_proposed_intents,
    validate_proposed_repair_intents,
)
from geometry_repair.roles import build_final_role_validation, required_roles_for_profile


def _write_cube(path: Path) -> Path:
    from pxr import Gf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/Cube")
    mesh.GetPointsAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0, 0, 0),
                Gf.Vec3f(1, 0, 0),
                Gf.Vec3f(1, 1, 0),
                Gf.Vec3f(0, 1, 0),
                Gf.Vec3f(0, 0, 1),
                Gf.Vec3f(1, 0, 1),
                Gf.Vec3f(1, 1, 1),
                Gf.Vec3f(0, 1, 1),
            ]
        )
    )
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * 12))
    mesh.GetFaceVertexIndicesAttr().Set(
        Vt.IntArray(
            [
                0,
                2,
                1,
                0,
                3,
                2,
                4,
                5,
                6,
                4,
                6,
                7,
                0,
                1,
                5,
                0,
                5,
                4,
                1,
                2,
                6,
                1,
                6,
                5,
                2,
                3,
                7,
                2,
                7,
                6,
                3,
                0,
                4,
                3,
                4,
                7,
            ]
        )
    )
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()
    return path


def _write_open_cube(path: Path) -> Path:
    from pxr import Usd, UsdGeom, Vt

    source = _write_cube(path)
    stage = Usd.Stage.Open(str(source))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Cube"))
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())[:-6]
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * (len(indices) // 3)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
    stage.GetRootLayer().Save()
    return source


def test_required_roles_are_profile_specific() -> None:
    assert required_roles_for_profile("visual_only", source_format="usd") == {"render"}
    assert required_roles_for_profile("rigid_pick_place", source_format="usd") == {
        "render",
        "collision",
    }
    assert required_roles_for_profile("visual_only", source_format="step") == {
        "render",
        "brep_source",
    }


def test_diagnosis_exposes_role_scoped_states(tmp_path: Path) -> None:
    source = _write_cube(tmp_path / "cube.usda")
    visual = diagnose_asset(source, profile="visual_only", run_scalable_audit=False)
    rigid = diagnose_asset(source, profile="rigid_pick_place", run_scalable_audit=False)

    assert visual.role_validation["render"].required is True
    assert visual.role_validation["render"].status == "pass"
    assert visual.role_validation["collision"].status == "not_applicable"
    assert rigid.role_validation["collision"].required is True
    assert rigid.role_validation["collision"].status == "not_evaluated"
    assert rigid.role_validation["brep_source"].status == "not_applicable"


def test_plan_binds_every_operation_to_deterministic_intent(tmp_path: Path) -> None:
    source = _write_cube(tmp_path / "cube.usda")
    diagnosis = diagnose_asset(source, profile="visual_only", run_scalable_audit=False)
    plan = _plan_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="visual_only",
            mode="diagnose",
            budgets=RepairBudgets(max_attempts=2),
            enabled_workers=["trimesh_conservative_cleanup"],
        ),
        diagnosis,
        source,
        tmp_path / "repair_plan.json",
    )

    assert len(plan.operations) == len(plan.intents) == 1
    assert plan.operations[0].intent_id == plan.intents[0].intent_id
    assert plan.operations[0].target_role == "render"
    assert plan.intents[0].expected_topology_effects == ["none"]


def test_agent_intent_is_strict_and_cannot_cross_role_boundary(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RepairIntent.model_validate(
            {
                "intent_id": "unsafe-threshold",
                "authority": "agent_proposal",
                "target_role": "render",
                "issue_ids": ["mesh:physics_incompatible_transforms"],
                "defect_class": "transform",
                "expected_topology_effects": ["none"],
                "candidate_workers": ["trimesh_conservative_cleanup"],
                "evidence_paths": [str(evidence)],
                "max_drift_override": 1.0,
            }
        )

    source = _write_cube(tmp_path / "cube.usda")
    diagnosis = diagnose_asset(source, profile="rigid_pick_place", run_scalable_audit=False)
    collision_issue = DiagnosisIssue(
        issue_id="task:collision_proxy_required",
        category="task",
        scope="asset",
        severity="warning",
        fact_kind="measured_fact",
        summary="Collision representation requires bounded reconstruction.",
        candidate_repairs=["sdf_collision_rebuild"],
        affected_roles=["collision"],
        affected_prim_paths=["/World/Cube"],
    )
    diagnosis = diagnosis.model_copy(update={"issues": [*diagnosis.issues, collision_issue]})
    proposal = RepairIntent(
        intent_id="wrong-role",
        authority="agent_proposal",
        target_role="render",
        target_prim_paths=["/World/Cube"],
        issue_ids=[collision_issue.issue_id],
        defect_class="collision_proxy",
        expected_topology_effects=["replace_target_surface"],
        candidate_workers=["trimesh_conservative_cleanup"],
        evidence_paths=[str(evidence)],
    )
    with pytest.raises(ValueError, match="does not match issue role evidence"):
        validate_proposed_repair_intents(
            [proposal],
            diagnosis,
            enabled_workers={"trimesh_conservative_cleanup"},
            protected_features=[],
        )
    valid_collision_proposal = RepairIntent(
        intent_id="collision-only",
        authority="agent_proposal",
        target_role="collision",
        target_prim_paths=["/World/Cube"],
        issue_ids=[collision_issue.issue_id],
        defect_class="collision_proxy",
        expected_topology_effects=["generate_collision_representation"],
        candidate_workers=["sdf_collision_rebuild"],
        evidence_paths=[str(evidence)],
    )
    assert validate_proposed_repair_intents(
        [valid_collision_proposal],
        diagnosis,
        enabled_workers={"sdf_collision_rebuild"},
        protected_features=[],
    ) == [valid_collision_proposal]

    forged_deterministic_proposal = valid_collision_proposal.model_copy(
        update={"intent_id": "forged-deterministic", "authority": "deterministic"}
    )
    with pytest.raises(ValueError, match="cannot claim deterministic authority"):
        validate_proposed_repair_intents(
            [forged_deterministic_proposal],
            diagnosis,
            enabled_workers={"sdf_collision_rebuild"},
            protected_features=[],
        )


def test_final_role_status_does_not_hide_render_failure(tmp_path: Path) -> None:
    source = _write_cube(tmp_path / "cube.usda")
    diagnosis = diagnose_asset(source, profile="rigid_pick_place", run_scalable_audit=False)
    result = build_final_role_validation(
        profile="rigid_pick_place",
        diagnosis_role_validation=diagnosis.role_validation,
        source_format="usd",
        render_path=str(source),
        collision_path=str(source),
        source_format_status="pass",
        fidelity_status="fail",
        visual_review_required=False,
        collision_status="pass",
        collision_runtime_status="pass",
        usd_package_status="pass",
        advanced_profile_status="not_evaluated",
        evidence_paths_by_role={
            "render": [str(source)],
            "collision": [str(source)],
            "brep_source": [],
            "helper": [],
        },
    )
    assert result["render"].status == "fail"
    assert result["collision"].status == "pass"
    assert "fidelity" in result["render"].blockers


def test_validated_collision_replacement_clears_only_collision_source_blocker(
    tmp_path: Path,
) -> None:
    source = _write_open_cube(tmp_path / "open_cube.usda")
    diagnosis = diagnose_asset(source, profile="rigid_pick_place", run_scalable_audit=False)
    boundary = next(issue for issue in diagnosis.issues if issue.issue_id == "mesh:boundary_edges")

    assert boundary.affected_roles == ["render", "collision"]
    assert diagnosis.role_validation["render"].status == "fail"
    assert diagnosis.role_validation["collision"].status == "fail"

    result = build_final_role_validation(
        profile="rigid_pick_place",
        diagnosis_role_validation=diagnosis.role_validation,
        source_format="usd",
        render_path=str(source),
        collision_path=str(source),
        source_format_status="pass",
        fidelity_status="fail",
        visual_review_required=False,
        collision_status="conditional",
        collision_runtime_status="not_evaluated",
        usd_package_status="pass",
        advanced_profile_status="not_evaluated",
        evidence_paths_by_role={
            "render": [str(source)],
            "collision": [str(source)],
            "brep_source": [],
            "helper": [],
        },
    )

    assert result["render"].status == "fail"
    assert "mesh:boundary_edges" in result["render"].blockers
    assert result["collision"].status == "conditional"
    assert result["collision"].blockers == []


def test_agent_ranking_reorders_only_existing_candidate_objects() -> None:
    operations = [
        RepairOperation(
            operation_id="noop",
            worker="noop",
            implementation="test",
            issue_ids=[],
            drift_band="identity",
            source_checkpoint="source.usda",
            target_role="render",
        ),
        RepairOperation(
            operation_id="cleanup",
            worker="trimesh_conservative_cleanup",
            implementation="test",
            parameters={"epsilon": 1e-9},
            issue_ids=["mesh:boundary_edges"],
            drift_band="conservative",
            source_checkpoint="source.usda",
            target_role="render",
            target_prim_paths=["/World/Cube"],
        ),
        RepairOperation(
            operation_id="patch",
            worker="pmp_patch",
            implementation="test",
            parameters={"operation": "pmp_fill_classified_hole"},
            issue_ids=["mesh:boundary_edges"],
            drift_band="conservative",
            source_checkpoint="source.usda",
            target_role="render",
            target_prim_paths=["/World/Cube"],
        ),
    ]
    source_dumps = {item.operation_id: item.model_dump() for item in operations}
    proposal = RepairIntent(
        intent_id="rank-patch-first",
        authority="agent_proposal",
        target_role="render",
        target_prim_paths=["/World/Cube"],
        issue_ids=["mesh:boundary_edges"],
        defect_class="classified_hole",
        expected_topology_effects=["generate_local_faces"],
        candidate_workers=["pmp_patch"],
        evidence_paths=["evidence.json"],
        confidence=0.9,
    )

    ranked, evidence = rank_operations_from_proposed_intents(operations, [proposal])

    assert [item.operation_id for item in ranked] == ["noop", "patch", "cleanup"]
    assert {item.operation_id: item.model_dump() for item in ranked} == source_dumps
    assert evidence[0]["matching_intent_ids"] == ["rank-patch-first"]
