# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the agentic physics authoring workflow."""

from __future__ import annotations

import json
from pathlib import Path

import content_workbench_agent_client
import pytest

from content_agent_workflows.common import physics_validation_evidence
from content_agent_workflows.physics import (
    PhysicsApplyWorkflowInput,
    PhysicsBehaviorAssessment,
    PhysicsComponentDecision,
    PhysicsDecision,
    infer_material_profile,
    infer_physics_decisions,
    inspect_mesh_prims,
    merge_physics_behavior_assessment,
    run_physics_apply_workflow,
)
from content_agent_workflows.physics import workbench_ops as physics_workbench_ops
from content_agent_workflows.physics import workflow as physics_workflow


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _simple_cube() -> Path:
    return (
        _repo_root()
        / "apps"
        / "physics_agent_service"
        / "tests"
        / "test_data"
        / "simple_cube.usda"
    )


def _component_inspection(usd_path: Path | str) -> dict[str, object]:
    return {
        "asset": str(usd_path),
        "source_digest": "sha256:test",
        "component_count": 1,
        "components": [
            {
                "component_id": "component_001",
                "path_space": "source",
                "body_root_path": "/World",
                "visual_evidence_paths": ["/World/Cube"],
                "collider_paths": [],
                "helper_paths": [],
                "rigid_body_paths": [],
                "joint_paths": [],
                "material_evidence": [
                    {
                        "prim_path": "/World/Cube",
                        "material_path": "/World/Looks/Test_Metal",
                        "material_name": "Test_Metal",
                    }
                ],
                "bounds_m": {
                    "min_m": [0.0, 0.0, 0.0],
                    "max_m": [1.0, 1.0, 1.0],
                    "size_m": [1.0, 1.0, 1.0],
                    "volume_m3": 1.0,
                },
                "topology_findings": [],
            }
        ],
    }


def test_write_json_normalizes_non_finite_numbers(tmp_path: Path) -> None:
    path = physics_workflow._write_json(
        tmp_path / "payload.json",
        {"nan": float("nan"), "inf": float("inf"), "nested": [-float("inf")]},
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "inf": None,
        "nan": None,
        "nested": [None],
    }


def test_runtime_acceptance_uses_enabled_body_count_and_zero_drop() -> None:
    assert (
        physics_workflow._runtime_acceptance_from_authored_report(
            {"rigid_body_count": 2, "enabled_rigid_body_count": 2}
        )
        is None
    )
    assert physics_workflow._runtime_acceptance_from_authored_report(
        {"rigid_body_count": 2, "enabled_rigid_body_count": 1},
        drop_height_m=0.0,
    ) == {"expected_body_count": 1, "require_gravity_response": False}


def test_physics_validation_evidence_maps_runtime_pass() -> None:
    evidence = physics_validation_evidence(
        asset="/tmp/asset.usda",
        target_runtime="ovphysx",
        physics_properties_status="pass",
        runtime_loadability_status="pass",
        no_explosions_status="pass",
    )

    assert evidence.workflow == "physics_authoring"
    assert evidence.sim_ready_status == "pass"
    assert [check.name for check in evidence.checks] == [
        "physics_properties",
        "runtime_loadability",
        "no_explosions",
    ]


def test_physics_visual_behavior_pass_preserves_runtime_pass() -> None:
    evidence = physics_validation_evidence(
        asset="/tmp/asset.usda",
        target_runtime="ovphysx",
        physics_properties_status="pass",
        runtime_loadability_status="pass",
        no_explosions_status="pass",
    )
    merged = merge_physics_behavior_assessment(
        evidence,
        PhysicsBehaviorAssessment(
            status="pass",
            checked_views=["/tmp/frame_0000.png"],
            rendered_frames=["/tmp/frame_0000.png"],
            runtime_report="/tmp/runtime_validation_report.json",
            assessment_notes="Behavior is plausible.",
        ),
    )

    assert merged.sim_ready_status == "pass"
    assert [check.name for check in merged.checks][-1] == "simulation_visual_review"
    assert merged.checks[-1].status == "pass"


def test_physics_visual_behavior_pass_preserves_runtime_conditional() -> None:
    evidence = physics_validation_evidence(
        asset="/tmp/asset.usda",
        target_runtime="ovphysx",
        physics_properties_status="pass",
        runtime_loadability_status="pass",
        no_explosions_status="pass",
        warnings=["Body did not reach the default settle threshold."],
    )
    merged = merge_physics_behavior_assessment(
        evidence,
        PhysicsBehaviorAssessment(
            status="pass",
            checked_views=["/tmp/frame_0000.png"],
            rendered_frames=["/tmp/frame_0000.png"],
            runtime_report="/tmp/runtime_validation_report.json",
            assessment_notes="Behavior is visually plausible.",
        ),
    )

    assert merged.sim_ready_status == "conditional"
    assert "settle threshold" in merged.warnings[0]
    assert merged.checks[-1].status == "pass"


def test_physics_visual_behavior_unresolved_makes_runtime_pass_conditional() -> None:
    evidence = physics_validation_evidence(
        asset="/tmp/asset.usda",
        target_runtime="ovphysx",
        physics_properties_status="pass",
        runtime_loadability_status="pass",
        no_explosions_status="pass",
    )
    merged = merge_physics_behavior_assessment(
        evidence,
        PhysicsBehaviorAssessment(
            status="unresolved_issues",
            checked_views=["/tmp/frame_0000.png"],
            rendered_frames=["/tmp/frame_0000.png"],
            runtime_report="/tmp/runtime_validation_report.json",
            unresolved_issues=["The bulb visually separates from the screw cap."],
            assessment_notes="One behavior issue remains.",
        ),
    )

    assert merged.sim_ready_status == "conditional"
    assert merged.checks[-1].name == "simulation_visual_review"
    assert merged.checks[-1].status == "warning"
    assert "bulb visually separates" in merged.unresolved_issues[0]


def test_physics_visual_behavior_unresolved_without_details_is_conditional() -> None:
    evidence = physics_validation_evidence(
        asset="/tmp/asset.usda",
        target_runtime="ovphysx",
        physics_properties_status="pass",
        runtime_loadability_status="pass",
        no_explosions_status="pass",
    )
    merged = merge_physics_behavior_assessment(
        evidence,
        PhysicsBehaviorAssessment(
            status="unresolved_issues",
            checked_views=["/tmp/frame_0000.png"],
            rendered_frames=["/tmp/frame_0000.png"],
            runtime_report="/tmp/runtime_validation_report.json",
            assessment_notes="Issues remain.",
        ),
    )

    assert merged.sim_ready_status == "conditional"
    assert merged.checks[-1].name == "simulation_visual_review"
    assert merged.checks[-1].status == "warning"
    assert "without details" in merged.unresolved_issues[0]


def test_physics_visual_behavior_cannot_override_runtime_failure() -> None:
    evidence = physics_validation_evidence(
        asset="/tmp/asset.usda",
        target_runtime="ovphysx",
        physics_properties_status="pass",
        runtime_loadability_status="fail",
        no_explosions_status="not_evaluated",
        failures=["Simulation did not load any rigid bodies."],
    )
    merged = merge_physics_behavior_assessment(
        evidence,
        PhysicsBehaviorAssessment(
            status="pass",
            checked_views=["/tmp/frame_0000.png"],
            rendered_frames=["/tmp/frame_0000.png"],
            runtime_report="/tmp/runtime_validation_report.json",
            assessment_notes="Rendered frame looked plausible.",
        ),
    )

    assert merged.sim_ready_status == "fail"
    assert merged.failures == ["Simulation did not load any rigid bodies."]


def test_material_profile_prefers_specific_binding_over_asset_name() -> None:
    assert (
        infer_material_profile(
            "Bulb_Screw_Cap", "Cap_Plastic", "/light_bulb_01/..."
        ).family
        == "plastic"
    )
    assert (
        infer_material_profile("Bulb_Screw", "Screw_Metal", "/light_bulb_01/...").family
        == "metal"
    )


def test_inspect_and_infer_physics_decisions_from_mesh_fixture() -> None:
    candidates = inspect_mesh_prims(_simple_cube())
    decisions = infer_physics_decisions(candidates)

    assert [candidate.prim_path for candidate in candidates] == ["/World/Cube"]
    assert decisions[0].prim_paths == ["/World/Cube"]
    assert decisions[0].physical_properties["density"] > 0.0
    assert decisions[0].physical_properties["estimated_mass_kg"] > 0.0


def test_run_physics_apply_workflow_writes_canonical_artifacts(tmp_path: Path) -> None:
    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=_simple_cube(),
            output_dir=tmp_path,
            simulation_engine="fake",
            simulation_duration_s=0.2,
            simulation_sample_fps=10,
            drop_height_m=0.1,
        )
    )

    assert result.success
    assert result.physics_usd_path is not None
    assert Path(result.physics_usd_path).exists()
    assert result.assignments_path is not None
    assert result.validation_evidence_path is not None
    assert result.simulation_report_path is not None

    assignments = json.loads(Path(result.assignments_path).read_text())
    evidence = json.loads(Path(result.validation_evidence_path).read_text())
    report = json.loads(Path(result.simulation_report_path).read_text())

    assert assignments["candidate_count"] == 1
    assert assignments["decision_count"] == 1
    assert assignments["apply_report"]["collision_count"] == 1
    assert evidence["workflow"] == "physics_authoring"
    assert report["engine"] == "fake"
    assert Path(report["trajectory_jsonl"]).exists()
    assert Path(report["recording_usda"]).exists()


def test_trajectory_response_omits_full_trajectory(tmp_path: Path) -> None:
    response_path = physics_workbench_ops._write_trajectory_response(
        tmp_path / "simulation_response.json",
        {
            "status": "ok",
            "trajectory": [
                (0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0] * 6),
                (0.1, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0] * 6),
            ],
        },
    )

    payload = json.loads(response_path.read_text(encoding="utf-8"))

    assert payload["trajectory_sample_count"] == 2
    assert "trajectory" not in payload


def test_predictions_jsonl_expands_grouped_decisions(tmp_path: Path) -> None:
    predictions_path = physics_workflow._write_predictions_jsonl(
        tmp_path / "physics_predictions.jsonl",
        [
            PhysicsDecision(
                decision_id="fixture",
                prim_paths=["/World/A", "/World/B"],
                component_label="grouped component",
                inferred_material_family="metal",
                inferred_material_name=None,
                collision_approximation="convexHull",
                physical_properties={
                    "density": 2700.0,
                    "estimated_mass_kg": 1.0,
                    "static_friction": 0.6,
                    "dynamic_friction": 0.5,
                    "restitution": 0.1,
                },
                confidence=0.8,
                rationale="Grouped decision fixture.",
            )
        ],
    )

    records = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["id"] for record in records] == ["/World/A", "/World/B"]
    assert all(record["classification"]["material"] == "metal" for record in records)
    assert [
        record["classification"]["physical_properties"]["estimated_mass_kg"]
        for record in records
    ] == [0.5, 0.5]


def test_predictions_jsonl_preserves_v2_component_mass_metadata(
    tmp_path: Path,
) -> None:
    predictions_path = physics_workflow._write_predictions_jsonl(
        tmp_path / "physics_predictions.jsonl",
        [
            PhysicsComponentDecision(
                decision_id="component_001",
                component_id="component_001",
                body_root_path="/World",
                visual_evidence_paths=["/World/Visual"],
                collider_paths=["/World/ColliderA", "/World/ColliderB"],
                collision_mode="preserve_existing",
                mass_authoring_path="/World",
                inferred_material_family="metal",
                inferred_material_name=None,
                collision_approximation="convexHull",
                physical_properties={
                    "density": 2700.0,
                    "estimated_mass_kg": 1.0,
                    "static_friction": 0.6,
                    "dynamic_friction": 0.5,
                    "restitution": 0.1,
                },
                confidence=0.8,
                rationale="Grouped component fixture.",
            )
        ],
    )

    records = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["id"] for record in records] == [
        "/World/ColliderA",
        "/World/ColliderB",
    ]
    assert [
        record["classification"]["physical_properties"]["estimated_mass_kg"]
        for record in records
    ] == [0.5, 0.5]
    assert [
        record["classification"]["component_estimated_mass_kg"] for record in records
    ] == [1.0, 1.0]
    assert [record["classification"]["collision_mode"] for record in records] == [
        "preserve_existing",
        "preserve_existing",
    ]
    assert all(
        record["classification"]["component_id"] == "component_001"
        and record["classification"]["mass_authoring_path"] == "/World"
        for record in records
    )


def test_validate_component_decisions_rejects_authoring_over_existing_colliders() -> (
    None
):
    component = physics_workflow.PhysicsComponent.model_validate(
        {
            "component_id": "component_001",
            "body_root_path": "/World",
            "visual_evidence_paths": ["/World/Visual"],
            "collider_paths": ["/World/Collider"],
            "helper_paths": [],
            "rigid_body_paths": ["/World"],
            "joint_paths": [],
            "material_evidence": [],
            "bounds_m": {},
            "topology_findings": [],
        }
    )
    decision = PhysicsComponentDecision(
        decision_id="component_001",
        component_id="component_001",
        body_root_path="/World",
        visual_evidence_paths=["/World/Visual"],
        collider_paths=["/World/Visual"],
        collision_mode="author_on_targets",
        mass_authoring_path="/World",
        inferred_material_family="metal",
        inferred_material_name=None,
        collision_approximation="convexHull",
        physical_properties={"density": 2700.0, "estimated_mass_kg": 1.0},
        confidence=0.8,
        rationale="Fixture.",
    )

    with pytest.raises(RuntimeError, match="preserve existing colliders"):
        physics_workflow._validate_component_decisions([component], [decision], [])


def test_run_physics_apply_workflow_uses_workbench_ops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_inspect_components(_usd_path: Path) -> dict[str, object]:
        calls.append("inspect")
        return _component_inspection(_usd_path)

    def fake_apply_schema(**kwargs: object) -> dict[str, object]:
        calls.append("apply")
        predictions = Path(str(kwargs["predictions_jsonl_path"]))
        assert predictions.exists()
        physics_usd = tmp_path / "physics.usda"
        physics_usd.write_text("#usda 1.0\n", encoding="utf-8")
        return {
            "physics_usd": str(physics_usd),
            "default_prim": "/World",
            "physics_scene_paths": ["/World/PhysicsScene"],
            "rigid_body_paths": ["/World", "/World/Part"],
            "collision_paths": ["/World/Cube"],
            "physics_material_paths": ["/World/Looks/PhysMat"],
            "rigid_body_count": 2,
            "collision_count": 1,
            "physics_material_count": 1,
        }

    validate_kwargs: list[dict[str, object]] = []

    def fake_validate_runtime(**kwargs: object) -> dict[str, object]:
        calls.append("validate")
        validate_kwargs.append(dict(kwargs))
        report = tmp_path / "runtime" / "runtime_validation_report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text('{"engine":"fake"}\n', encoding="utf-8")
        return {
            "engine": "fake",
            "runtime_report": str(report),
            "failures": [],
            "warnings": [],
            "settle_distance": 0.0,
            "summary": {"settle_time_s": 0.1},
            "evidence_artifacts": [
                {
                    "kind": "runtime_report",
                    "path": str(report),
                    "description": "Runtime validation metrics.",
                }
            ],
        }

    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "inspect_components",
        fake_inspect_components,
    )
    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "apply_schema",
        fake_apply_schema,
    )
    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "validate_runtime",
        fake_validate_runtime,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            simulation_engine="fake",
        )
    )

    assert result.success
    assert calls == ["inspect", "apply"]
    assert validate_kwargs == []
    evidence = json.loads(Path(result.validation_evidence_path or "").read_text())
    assert evidence["sim_ready_status"] == "conditional"
    assert any(
        "single bound rigid body" in item for item in evidence["unresolved_issues"]
    )
    runtime_report = json.loads(Path(result.simulation_report_path or "").read_text())
    assert runtime_report["not_evaluated"] is True
    assert runtime_report["enabled_rigid_body_count"] == 2


def test_run_physics_apply_workflow_preserves_static_topology_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_usd = tmp_path / "prepared.usda"
    prepared_usd.write_text("#usda 1.0\n", encoding="utf-8")
    topology_plan = tmp_path / "topology_plan.json"
    topology_plan.write_text(
        json.dumps(
            {
                "schema_version": "content-workflows.physics-topology-plan.v1",
                "expected_source_digest": "sha256:source",
                "mobility_intent": "static",
                "operations": [],
                "invariants": {
                    "enabled_collider_count": 1,
                    "reject_articulation_changes": True,
                },
            }
        ),
        encoding="utf-8",
    )
    apply_kwargs: list[dict[str, object]] = []

    def fake_apply_topology_plan(**kwargs: object) -> dict[str, object]:
        return {
            "operation": "physics.apply_topology_plan",
            "output_usd_path": str(prepared_usd),
            "before": {},
            "after": {},
            "mobility_intent": kwargs["mobility_intent"],
        }

    def fake_inspect_components(_usd_path: Path) -> dict[str, object]:
        return _component_inspection(_usd_path)

    def fake_apply_schema(**kwargs: object) -> dict[str, object]:
        apply_kwargs.append(dict(kwargs))
        physics_usd = tmp_path / "physics.usda"
        physics_usd.write_text("#usda 1.0\n", encoding="utf-8")
        return {
            "physics_usd": str(physics_usd),
            "default_prim": "/World",
            "physics_scene_paths": ["/World/PhysicsScene"],
            "rigid_body_paths": ["/World"],
            "enabled_rigid_body_paths": [],
            "disabled_rigid_body_paths": ["/World"],
            "collision_paths": ["/World/Cube"],
            "physics_material_paths": ["/World/Looks/PhysMat"],
            "rigid_body_count": 1,
            "enabled_rigid_body_count": 0,
            "disabled_rigid_body_count": 1,
            "collision_count": 1,
            "physics_material_count": 1,
        }

    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "apply_topology_plan",
        fake_apply_topology_plan,
    )
    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "inspect_components",
        fake_inspect_components,
    )
    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "apply_schema",
        fake_apply_schema,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            simulation_engine="fake",
            topology_plan_path=topology_plan,
        )
    )

    assert result.success
    assert apply_kwargs[0]["author_rigid_body"] is False
    runtime_report = json.loads(Path(result.simulation_report_path or "").read_text())
    assert runtime_report["not_evaluated"] is True
    assert runtime_report["mobility_intent"] == "static"


def test_run_physics_apply_workflow_rejects_static_topology_with_enabled_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_usd = tmp_path / "prepared.usda"
    prepared_usd.write_text("#usda 1.0\n", encoding="utf-8")
    topology_plan = tmp_path / "topology_plan.json"
    topology_plan.write_text(
        json.dumps(
            {
                "schema_version": "content-workflows.physics-topology-plan.v1",
                "expected_source_digest": "sha256:source",
                "mobility_intent": "static",
                "operations": [],
                "invariants": {
                    "enabled_collider_count": 1,
                    "reject_articulation_changes": True,
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_apply_topology_plan(**kwargs: object) -> dict[str, object]:
        return {
            "operation": "physics.apply_topology_plan",
            "output_usd_path": str(prepared_usd),
            "before": {},
            "after": {},
            "mobility_intent": kwargs["mobility_intent"],
        }

    def fake_apply_schema(**_kwargs: object) -> dict[str, object]:
        physics_usd = tmp_path / "physics.usda"
        physics_usd.write_text("#usda 1.0\n", encoding="utf-8")
        return {
            "physics_usd": str(physics_usd),
            "default_prim": "/World",
            "physics_scene_paths": ["/World/PhysicsScene"],
            "rigid_body_paths": ["/World"],
            "collision_paths": ["/World/Cube"],
            "physics_material_paths": ["/World/Looks/PhysMat"],
            "rigid_body_count": 1,
            "collision_count": 1,
            "physics_material_count": 1,
        }

    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "apply_topology_plan",
        fake_apply_topology_plan,
    )
    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "inspect_components",
        _component_inspection,
    )
    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "apply_schema",
        fake_apply_schema,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            topology_plan_path=topology_plan,
            simulation_engine="fake",
        )
    )

    assert not result.success
    evidence = json.loads(Path(result.validation_evidence_path or "").read_text())
    assert any(
        "zero enabled rigid bodies" in failure for failure in evidence["failures"]
    )


def test_run_physics_apply_workflow_rebases_supplied_patch_after_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_usd = tmp_path / "prepared.usda"
    prepared_usd.write_text("#usda 1.0\n", encoding="utf-8")
    topology_plan = tmp_path / "topology_plan.json"
    topology_plan.write_text(
        json.dumps(
            {
                "schema_version": "content-workflows.physics-topology-plan.v1",
                "expected_source_digest": "sha256:source",
                "mobility_intent": "movable",
                "operations": [],
                "invariants": {
                    "enabled_collider_count": 1,
                    "reject_articulation_changes": True,
                },
            }
        ),
        encoding="utf-8",
    )
    patch_path = tmp_path / "physics_decision_patch.json"
    patch_path.write_text(
        json.dumps(
            {
                "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
                "asset": str(tmp_path / "input.usda"),
                "source_digest": "sha256:source",
                "decisions": [
                    {
                        "decision_id": "component_001",
                        "component_id": "component_001",
                        "body_root_path": "/World",
                        "visual_evidence_paths": ["/World/Cube"],
                        "collider_paths": ["/World/Cube"],
                        "collision_mode": "author_on_targets",
                        "mass_authoring_path": "/World",
                        "inferred_material_family": "metal",
                        "collision_approximation": "convexHull",
                        "physical_properties": {
                            "density": 2700.0,
                            "estimated_mass_kg": 1.0,
                        },
                        "confidence": 0.8,
                        "rationale": "Fixture.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    observed_patch_payloads: list[dict[str, object]] = []
    observed_patch_paths: list[Path] = []

    def fake_apply_topology_plan(**kwargs: object) -> dict[str, object]:
        return {
            "operation": "physics.apply_topology_plan",
            "output_usd_path": str(prepared_usd),
            "before": {},
            "after": {},
            "mobility_intent": kwargs["mobility_intent"],
        }

    def fake_inspect_components(usd_path: Path) -> dict[str, object]:
        result = _component_inspection(usd_path)
        result["source_digest"] = (
            "sha256:prepared" if Path(usd_path) == prepared_usd else "sha256:source"
        )
        return result

    def fake_apply_schema(**kwargs: object) -> dict[str, object]:
        copied_patch = Path(str(kwargs["decision_patch_path"]))
        observed_patch_paths.append(copied_patch)
        observed_patch_payloads.append(json.loads(copied_patch.read_text()))
        physics_usd = tmp_path / "physics.usda"
        physics_usd.write_text("#usda 1.0\n", encoding="utf-8")
        return {
            "physics_usd": str(physics_usd),
            "default_prim": "/World",
            "physics_scene_paths": ["/World/PhysicsScene"],
            "rigid_body_paths": ["/World"],
            "collision_paths": ["/World/Cube"],
            "physics_material_paths": ["/World/Looks/PhysMat"],
            "rigid_body_count": 1,
            "collision_count": 1,
            "physics_material_count": 1,
        }

    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "apply_topology_plan",
        fake_apply_topology_plan,
    )
    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "inspect_components",
        fake_inspect_components,
    )
    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "apply_schema",
        fake_apply_schema,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            decision_patch_path=patch_path,
            topology_plan_path=topology_plan,
            run_simulation=False,
        )
    )

    assert result.success
    canonical_payload = json.loads(
        Path(result.decision_patch_path or "").read_text(encoding="utf-8")
    )
    assert canonical_payload["asset"] == str(tmp_path / "input.usda")
    assert canonical_payload["source_digest"] == "sha256:source"
    assert observed_patch_paths[0].name == "physics_decision_patch_apply.json"
    assert observed_patch_payloads[0]["asset"] == str(prepared_usd.resolve())
    assert observed_patch_payloads[0]["source_digest"] == "sha256:prepared"


def test_rebase_supplied_patch_rejects_unresolved_components_on_heuristic_path(
    tmp_path: Path,
) -> None:
    component = physics_workflow.PhysicsComponent.model_validate(
        {
            "component_id": "component_new",
            "body_root_path": "/World",
            "visual_evidence_paths": ["/World/Cube"],
            "collider_paths": [],
            "helper_paths": [],
            "rigid_body_paths": [],
            "joint_paths": [],
            "material_evidence": [],
            "bounds_m": {},
            "topology_findings": [],
        }
    )
    payload = {
        "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
        "asset": str(tmp_path / "input.usda"),
        "source_digest": "sha256:source",
        "decisions": [
            {
                "decision_id": "component_old",
                "component_id": "component_old",
                "body_root_path": "/World",
                "visual_evidence_paths": ["/World/Cube"],
                "collider_paths": ["/World/Cube"],
                "collision_mode": "author_on_targets",
                "mass_authoring_path": "/World",
                "inferred_material_family": "metal",
                "collision_approximation": "convexHull",
                "physical_properties": {
                    "density": 2700.0,
                    "estimated_mass_kg": 1.0,
                },
                "confidence": 0.8,
                "rationale": "Fixture.",
            }
        ],
        "unresolved_components": [
            {"component_id": "missing_component", "reason": "No material evidence."}
        ],
    }

    with pytest.raises(RuntimeError, match="unresolved_components"):
        physics_workflow._rebase_v2_patch_payload_to_components(
            payload,
            components=[component],
            source_digest="sha256:prepared",
            asset=tmp_path / "prepared.usda",
        )


def test_rebase_supplied_patch_rejects_ambiguous_heuristic_match(
    tmp_path: Path,
) -> None:
    components = [
        physics_workflow.PhysicsComponent.model_validate(
            {
                "component_id": component_id,
                "body_root_path": body_root,
                "visual_evidence_paths": ["/World/SharedMesh"],
                "collider_paths": [],
                "helper_paths": [],
                "rigid_body_paths": [],
                "joint_paths": [],
                "material_evidence": [],
                "bounds_m": {},
                "topology_findings": [],
            }
        )
        for component_id, body_root in [
            ("component_new_a", "/World/A"),
            ("component_new_b", "/World/B"),
        ]
    ]
    payload = {
        "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
        "asset": str(tmp_path / "input.usda"),
        "source_digest": "sha256:source",
        "decisions": [
            {
                "decision_id": decision_id,
                "component_id": decision_id,
                "body_root_path": "/World",
                "visual_evidence_paths": ["/World/SharedMesh"],
                "collider_paths": ["/World/SharedMesh"],
                "collision_mode": "author_on_targets",
                "mass_authoring_path": "/World",
                "inferred_material_family": "metal",
                "collision_approximation": "convexHull",
                "physical_properties": {
                    "density": 2700.0,
                    "estimated_mass_kg": 1.0,
                },
                "confidence": 0.8,
                "rationale": "Fixture.",
            }
            for decision_id in ["component_old_a", "component_old_b"]
        ],
    }

    with pytest.raises(RuntimeError, match="changed physics component identity"):
        physics_workflow._rebase_v2_patch_payload_to_components(
            payload,
            components=components,
            source_digest="sha256:prepared",
            asset=tmp_path / "prepared.usda",
        )


def test_rebase_supplied_patch_coalesces_compatible_decisions(
    tmp_path: Path,
) -> None:
    component = physics_workflow.PhysicsComponent.model_validate(
        {
            "component_id": "component_merged",
            "body_root_path": "/World",
            "visual_evidence_paths": ["/World/A", "/World/B"],
            "collider_paths": [],
            "helper_paths": [],
            "rigid_body_paths": ["/World"],
            "joint_paths": [],
            "material_evidence": [],
            "bounds_m": {},
            "topology_findings": [],
        }
    )
    payload = {
        "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
        "asset": str(tmp_path / "input.usda"),
        "source_digest": "sha256:source",
        "decisions": [
            {
                "decision_id": decision_id,
                "component_id": decision_id,
                "body_root_path": body_root,
                "visual_evidence_paths": [path],
                "collider_paths": [path],
                "collision_mode": "author_on_targets",
                "mass_authoring_path": body_root,
                "inferred_material_family": "metal",
                "collision_approximation": "convexHull",
                "physical_properties": {
                    "density": 2700.0,
                    "estimated_mass_kg": mass,
                },
                "confidence": confidence,
                "rationale": "Fixture.",
            }
            for decision_id, body_root, path, mass, confidence in [
                ("component_old_a", "/World/A", "/World/A", 1.25, 0.8),
                ("component_old_b", "/World/B", "/World/B", 2.75, 0.6),
            ]
        ],
    }

    rebased_payload, decisions, unresolved = (
        physics_workflow._rebase_v2_patch_payload_to_components(
            payload,
            components=[component],
            source_digest="sha256:prepared",
            asset=tmp_path / "prepared.usda",
        )
    )

    assert unresolved == []
    assert rebased_payload["source_digest"] == "sha256:prepared"
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.component_id == "component_merged"
    assert decision.body_root_path == "/World"
    assert decision.mass_authoring_path == "/World"
    assert decision.collider_paths == ["/World/A", "/World/B"]
    assert decision.physical_properties["estimated_mass_kg"] == pytest.approx(4.0)
    assert decision.confidence == pytest.approx(0.6)


def test_rebase_supplied_patch_rejects_role_changing_heuristic_match(
    tmp_path: Path,
) -> None:
    component = physics_workflow.PhysicsComponent.model_validate(
        {
            "component_id": "component_static",
            "component_role": "unowned_static",
            "body_root_path": "/World/StaticCollider",
            "visual_evidence_paths": ["/World/StaticCollider"],
            "collider_paths": ["/World/StaticCollider"],
            "helper_paths": [],
            "rigid_body_paths": [],
            "joint_paths": [],
            "material_evidence": [],
            "bounds_m": {},
            "topology_findings": [],
        }
    )
    payload = {
        "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
        "asset": str(tmp_path / "input.usda"),
        "source_digest": "sha256:source",
        "decisions": [
            {
                "decision_id": "component_body",
                "component_id": "component_body",
                "body_root_path": "/World/DynamicBody",
                "visual_evidence_paths": ["/World/StaticCollider"],
                "collider_paths": ["/World/StaticCollider"],
                "collision_mode": "preserve_existing",
                "mass_authoring_path": "/World/DynamicBody",
                "inferred_material_family": "metal",
                "collision_approximation": "convexHull",
                "physical_properties": {
                    "density": 2700.0,
                    "estimated_mass_kg": 1.0,
                },
                "confidence": 0.8,
                "rationale": "Fixture.",
            }
        ],
    }

    with pytest.raises(RuntimeError, match="changed physics component identity"):
        physics_workflow._rebase_v2_patch_payload_to_components(
            payload,
            components=[component],
            source_digest="sha256:prepared",
            asset=tmp_path / "prepared.usda",
        )


def test_run_physics_apply_workflow_allows_all_unresolved_v2_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_usd = tmp_path / "input.usda"
    input_usd.write_text("#usda 1.0\n", encoding="utf-8")
    patch_path = tmp_path / "physics_decision_patch.json"
    patch_path.write_text(
        json.dumps(
            {
                "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
                "asset": str(input_usd),
                "source_digest": "sha256:test",
                "decisions": [],
                "unresolved_components": [
                    {
                        "component_id": "component_001",
                        "reason": "No reliable material or geometry evidence.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "inspect_components",
        _component_inspection,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=input_usd,
            output_dir=tmp_path / "out",
            decision_patch_path=patch_path,
            run_simulation=False,
        )
    )

    assert result.success
    assert result.validation_status == "conditional"
    assert result.physics_usd_path is not None
    assert Path(result.physics_usd_path) == input_usd.resolve()
    assignments = json.loads(Path(result.assignments_path or "").read_text())
    assert assignments["decision_count"] == 0
    assert assignments["apply_report"]["authoring_skipped"] is True
    evidence = json.loads(Path(result.validation_evidence_path or "").read_text())
    assert evidence["sim_ready_status"] == "conditional"
    assert evidence["unresolved_issues"] == [
        "component_001: No reliable material or geometry evidence."
    ]


def test_run_physics_apply_workflow_rejects_stale_all_unresolved_v2_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_usd = tmp_path / "input.usda"
    input_usd.write_text("#usda 1.0\n", encoding="utf-8")
    patch_path = tmp_path / "physics_decision_patch.json"
    patch_path.write_text(
        json.dumps(
            {
                "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
                "asset": str(input_usd),
                "source_digest": "sha256:stale",
                "decisions": [],
                "unresolved_components": [
                    {"component_id": "component_001", "reason": "No evidence."}
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "inspect_components",
        _component_inspection,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=input_usd,
            output_dir=tmp_path / "out",
            decision_patch_path=patch_path,
            run_simulation=False,
        )
    )

    assert not result.success
    assert "source_digest" in (result.error or "")


def test_run_physics_apply_workflow_rejects_static_all_unresolved_enabled_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    input_usd = tmp_path / "input.usda"
    input_usd.write_text("#usda 1.0\n", encoding="utf-8")
    prepared_usd = tmp_path / "prepared.usda"
    stage = Usd.Stage.CreateNew(str(prepared_usd))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdPhysics.RigidBodyAPI.Apply(world).CreateRigidBodyEnabledAttr(True)
    stage.GetRootLayer().Save()
    topology_plan = tmp_path / "topology_plan.json"
    topology_plan.write_text(
        json.dumps(
            {
                "schema_version": "content-workflows.physics-topology-plan.v1",
                "expected_source_digest": "sha256:source",
                "mobility_intent": "static",
                "operations": [],
                "invariants": {
                    "enabled_collider_count": 0,
                    "reject_articulation_changes": True,
                },
            }
        ),
        encoding="utf-8",
    )
    patch_path = tmp_path / "physics_decision_patch.json"
    patch_path.write_text(
        json.dumps(
            {
                "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
                "asset": str(input_usd),
                "source_digest": "sha256:source",
                "decisions": [],
                "unresolved_components": [
                    {"component_id": "component_001", "reason": "No evidence."}
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_apply_topology_plan(**kwargs: object) -> dict[str, object]:
        return {
            "operation": "physics.apply_topology_plan",
            "output_usd_path": str(prepared_usd),
            "before": {},
            "after": {},
            "mobility_intent": kwargs["mobility_intent"],
        }

    def fake_inspect_components(usd_path: Path) -> dict[str, object]:
        result = _component_inspection(usd_path)
        result["source_digest"] = (
            "sha256:prepared" if Path(usd_path) == prepared_usd else "sha256:source"
        )
        return result

    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "apply_topology_plan",
        fake_apply_topology_plan,
    )
    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "inspect_components",
        fake_inspect_components,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=input_usd,
            output_dir=tmp_path / "out",
            decision_patch_path=patch_path,
            topology_plan_path=topology_plan,
            run_simulation=False,
        )
    )

    assert not result.success
    evidence = json.loads(Path(result.validation_evidence_path or "").read_text())
    assert evidence["sim_ready_status"] == "fail"
    assert any("zero enabled rigid bodies" in item for item in evidence["failures"])


def test_run_physics_apply_workflow_rejects_stale_patch_before_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology_plan = tmp_path / "topology_plan.json"
    topology_plan.write_text(
        json.dumps(
            {
                "schema_version": "content-workflows.physics-topology-plan.v1",
                "expected_source_digest": "sha256:source",
                "mobility_intent": "movable",
                "operations": [],
                "invariants": {
                    "enabled_collider_count": 1,
                    "reject_articulation_changes": True,
                },
            }
        ),
        encoding="utf-8",
    )
    patch_path = tmp_path / "physics_decision_patch.json"
    patch_path.write_text(
        json.dumps(
            {
                "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
                "asset": str(tmp_path / "input.usda"),
                "source_digest": "sha256:stale",
                "decisions": [
                    {
                        "decision_id": "component_001",
                        "component_id": "component_001",
                        "body_root_path": "/World",
                        "visual_evidence_paths": ["/World/Cube"],
                        "collider_paths": ["/World/Cube"],
                        "collision_mode": "author_on_targets",
                        "mass_authoring_path": "/World",
                        "inferred_material_family": "metal",
                        "collision_approximation": "convexHull",
                        "physical_properties": {
                            "density": 2700.0,
                            "estimated_mass_kg": 1.0,
                        },
                        "confidence": 0.8,
                        "rationale": "Fixture.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_inspect_components(usd_path: Path) -> dict[str, object]:
        result = _component_inspection(usd_path)
        result["source_digest"] = "sha256:source"
        return result

    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "inspect_components",
        fake_inspect_components,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            decision_patch_path=patch_path,
            topology_plan_path=topology_plan,
            run_simulation=False,
        )
    )

    assert not result.success
    assert result.error is not None
    assert "source_digest" in result.error


def test_run_physics_apply_workflow_rejects_legacy_patch_with_topology(
    tmp_path: Path,
) -> None:
    topology_plan = tmp_path / "topology_plan.json"
    topology_plan.write_text(
        json.dumps(
            {
                "schema_version": "content-workflows.physics-topology-plan.v1",
                "expected_source_digest": "sha256:source",
                "mobility_intent": "movable",
                "operations": [],
                "invariants": {
                    "enabled_collider_count": 1,
                    "reject_articulation_changes": True,
                },
            }
        ),
        encoding="utf-8",
    )
    patch_path = tmp_path / "legacy_patch.json"
    patch_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    physics_workflow.LEGACY_PHYSICS_DECISION_PATCH_SCHEMA_VERSION
                ),
                "decisions": [
                    {
                        "decision_id": "legacy",
                        "prim_paths": ["/World/Cube"],
                        "component_label": "cube",
                        "inferred_material_family": "metal",
                        "inferred_material_name": None,
                        "collision_approximation": "convexHull",
                        "physical_properties": {
                            "density": 2700.0,
                            "estimated_mass_kg": 1.0,
                        },
                        "confidence": 0.8,
                        "rationale": "Legacy fixture.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            decision_patch_path=patch_path,
            topology_plan_path=topology_plan,
            run_simulation=False,
        )
    )

    assert not result.success
    assert result.error is not None
    assert "Legacy physics decision patches" in result.error


def test_run_physics_apply_workflow_preserves_supplied_patch_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_path = tmp_path / "physics_decision_patch.json"
    patch_path.write_text(
        json.dumps(
            {
                "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
                "asset": str(tmp_path / "older.usda"),
                "source_digest": "sha256:test",
                "decisions": [
                    {
                        "decision_id": "component_001",
                        "component_id": "component_001",
                        "body_root_path": "/World",
                        "visual_evidence_paths": ["/World/Cube"],
                        "collider_paths": ["/World/Cube"],
                        "collision_mode": "author_on_targets",
                        "mass_authoring_path": "/World",
                        "inferred_material_family": "metal",
                        "collision_approximation": "convexHull",
                        "physical_properties": {
                            "density": 2700.0,
                            "estimated_mass_kg": 1.0,
                        },
                        "confidence": 0.8,
                        "rationale": "Fixture.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    observed_patch_payloads: list[dict[str, object]] = []
    observed_patch_paths: list[Path] = []

    def fake_inspect_components(_usd_path: Path) -> dict[str, object]:
        return _component_inspection(_usd_path)

    def fake_apply_schema(**kwargs: object) -> dict[str, object]:
        copied_patch = Path(str(kwargs["decision_patch_path"]))
        observed_patch_paths.append(copied_patch)
        observed_patch_payloads.append(json.loads(copied_patch.read_text()))
        physics_usd = tmp_path / "physics.usda"
        physics_usd.write_text("#usda 1.0\n", encoding="utf-8")
        return {
            "physics_usd": str(physics_usd),
            "default_prim": "/World",
            "physics_scene_paths": ["/World/PhysicsScene"],
            "rigid_body_paths": ["/World"],
            "collision_paths": ["/World/Cube"],
            "physics_material_paths": ["/World/Looks/PhysMat"],
            "rigid_body_count": 1,
            "collision_count": 1,
            "physics_material_count": 1,
        }

    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "inspect_components",
        fake_inspect_components,
    )
    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "apply_schema",
        fake_apply_schema,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            decision_patch_path=patch_path,
            run_simulation=False,
        )
    )

    assert result.success
    canonical_payload = json.loads(
        Path(result.decision_patch_path or "").read_text(encoding="utf-8")
    )
    assert canonical_payload["asset"].endswith("older.usda")
    assert canonical_payload["source_digest"] == "sha256:test"
    assert observed_patch_paths[0].name == "physics_decision_patch_apply.json"
    assert observed_patch_payloads[0]["asset"].endswith("input.usda")
    assert observed_patch_payloads[0]["source_digest"] == "sha256:test"


def test_fail_on_validation_error_preserves_artifact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_inspect_components(_usd_path: Path) -> dict[str, object]:
        return _component_inspection(_usd_path)

    def fake_apply_schema(**_kwargs: object) -> dict[str, object]:
        physics_usd = tmp_path / "physics.usda"
        physics_usd.write_text("#usda 1.0\n", encoding="utf-8")
        return {
            "physics_usd": str(physics_usd),
            "default_prim": "/World",
            "physics_scene_paths": [],
            "rigid_body_paths": [],
            "collision_paths": [],
            "physics_material_paths": [],
            "rigid_body_count": 0,
            "collision_count": 0,
            "physics_material_count": 0,
        }

    def fake_validate_runtime(**_kwargs: object) -> dict[str, object]:
        report = tmp_path / "runtime" / "runtime_validation_report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text('{"engine":"fake"}\n', encoding="utf-8")
        return {
            "engine": "fake",
            "runtime_report": str(report),
            "failures": [],
            "warnings": [],
            "evidence_artifacts": [],
        }

    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "inspect_components",
        fake_inspect_components,
    )
    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "apply_schema",
        fake_apply_schema,
    )
    monkeypatch.setattr(
        physics_workflow.workbench_ops,
        "validate_runtime",
        fake_validate_runtime,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            simulation_engine="fake",
            fail_on_validation_error=True,
        )
    )

    assert not result.success
    assert result.validation_status == "fail"
    assert result.error is not None
    assert result.assignments_path is not None
    assert Path(result.assignments_path).exists()
    assert result.validation_evidence_path is not None
    evidence = json.loads(Path(result.validation_evidence_path).read_text())
    assert evidence["sim_ready_status"] == "fail"
    assert evidence["failures"]
    physics_properties_check = next(
        check for check in evidence["checks"] if check["name"] == "physics_properties"
    )
    assert physics_properties_check["status"] == "fail"


def test_run_physics_apply_workflow_can_route_to_remote_workbench(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_inspect(
        workbench_url: str,
        session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        calls.append(("inspect", workbench_url, session_id))
        assert payload["usd_path"].endswith("input.usda")
        assert timeout == 42.0
        return {
            **_component_inspection(str(payload["usd_path"])),
        }

    def fake_apply(
        workbench_url: str,
        session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        calls.append(("apply", workbench_url, session_id))
        assert Path(str(payload["predictions_jsonl_path"])).exists()
        assert "output_usd_path" not in payload
        assert timeout == 42.0
        physics_usd = tmp_path / "remote_physics.usda"
        physics_usd.write_text("#usda 1.0\n", encoding="utf-8")
        return {
            "physics_usd": str(physics_usd),
            "default_prim": "/World",
            "physics_scene_paths": ["/World/PhysicsScene"],
            "rigid_body_paths": ["/World"],
            "collision_paths": ["/World/Cube"],
            "physics_material_paths": ["/World/Looks/PhysMat"],
            "rigid_body_count": 1,
            "collision_count": 1,
            "physics_material_count": 1,
        }

    def fake_validate(
        workbench_url: str,
        session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        calls.append(("validate", workbench_url, session_id))
        assert payload["engine"] == "fake"
        assert payload["acceptance"] == {"expected_body_count": 1}
        assert "output_dir" not in payload
        assert timeout == 42.0
        report = tmp_path / "remote_runtime_report.json"
        report.write_text('{"engine":"fake"}\n', encoding="utf-8")
        return {
            "engine": "fake",
            "runtime_report": str(report),
            "failures": [],
            "warnings": [],
            "settle_distance": 0.0,
            "summary": {"settle_time_s": 0.1},
            "evidence_artifacts": [
                {
                    "kind": "runtime_report",
                    "path": str(report),
                    "description": "Runtime validation metrics.",
                }
            ],
        }

    monkeypatch.setattr(
        content_workbench_agent_client,
        "inspect_physics_components",
        fake_inspect,
    )
    monkeypatch.setattr(
        content_workbench_agent_client,
        "apply_physics_schema",
        fake_apply,
    )
    monkeypatch.setattr(
        content_workbench_agent_client,
        "validate_physics_runtime",
        fake_validate,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            simulation_engine="fake",
            workbench_url="http://127.0.0.1:8088",
            workbench_session_id="session-one",
            workbench_timeout_s=42.0,
        )
    )

    assert result.success
    assert calls == [
        ("inspect", "http://127.0.0.1:8088", "session-one"),
        ("apply", "http://127.0.0.1:8088", "session-one"),
        ("validate", "http://127.0.0.1:8088", "session-one"),
    ]


def test_run_physics_apply_workflow_remote_topology_strips_plan_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_usd = tmp_path / "prepared.usda"
    prepared_usd.write_text("#usda 1.0\n", encoding="utf-8")
    topology_plan = tmp_path / "topology_plan.json"
    topology_plan.write_text(
        json.dumps(
            {
                "schema_version": "content-workflows.physics-topology-plan.v1",
                "expected_source_digest": "sha256:source",
                "mobility_intent": "movable",
                "operations": [],
                "invariants": {
                    "enabled_collider_count": 1,
                    "reject_articulation_changes": True,
                },
                "metadata": {"local_only": True},
            }
        ),
        encoding="utf-8",
    )
    observed_topology_payloads: list[dict[str, object]] = []

    def fake_apply_topology(
        _workbench_url: str,
        _session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        assert timeout == 42.0
        observed_topology_payloads.append(dict(payload))
        return {
            "operation": "physics.apply_topology_plan",
            "output_usd_path": str(prepared_usd),
            "before": {},
            "after": {},
            "mobility_intent": payload["mobility_intent"],
        }

    def fake_inspect(
        _workbench_url: str,
        _session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        assert timeout == 42.0
        return _component_inspection(str(payload["usd_path"]))

    def fake_apply_schema(
        _workbench_url: str,
        _session_id: str,
        _payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        assert timeout == 42.0
        physics_usd = tmp_path / "remote_physics.usda"
        physics_usd.write_text("#usda 1.0\n", encoding="utf-8")
        return {
            "physics_usd": str(physics_usd),
            "default_prim": "/World",
            "physics_scene_paths": ["/World/PhysicsScene"],
            "rigid_body_paths": ["/World"],
            "collision_paths": ["/World/Cube"],
            "physics_material_paths": ["/World/Looks/PhysMat"],
            "rigid_body_count": 1,
            "collision_count": 1,
            "physics_material_count": 1,
        }

    monkeypatch.setattr(
        content_workbench_agent_client,
        "apply_physics_topology_plan",
        fake_apply_topology,
    )
    monkeypatch.setattr(
        content_workbench_agent_client,
        "inspect_physics_components",
        fake_inspect,
    )
    monkeypatch.setattr(
        content_workbench_agent_client,
        "apply_physics_schema",
        fake_apply_schema,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            topology_plan_path=topology_plan,
            run_simulation=False,
            workbench_url="http://127.0.0.1:8088",
            workbench_session_id="session-one",
            workbench_timeout_s=42.0,
        )
    )

    assert result.success
    assert observed_topology_payloads == [
        {
            "schema_version": "content-workflows.physics-topology-plan.v1",
            "input_usd_path": str((tmp_path / "input.usda").resolve()),
            "output_usd_path": None,
            "expected_source_digest": "sha256:source",
            "mobility_intent": "movable",
            "operations": [],
            "invariants": {
                "enabled_collider_count": 1,
                "reject_articulation_changes": True,
            },
        }
    ]
