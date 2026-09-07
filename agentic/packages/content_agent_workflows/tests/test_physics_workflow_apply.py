# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the agentic physics authoring workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from content_agent_workflows.common import physics_validation_evidence
from content_agent_workflows.physics import (
    PhysicsApplyWorkflowInput,
    PhysicsBehaviorAssessment,
    PhysicsComponentDecision,
    PhysicsComponentTargetDecision,
    PhysicsDecision,
    PhysicsVompMassConfig,
    PhysicsVompMassResult,
    infer_material_profile,
    infer_physics_decisions,
    inspect_mesh_prims,
    merge_physics_behavior_assessment,
    run_physics_apply_workflow,
)
from content_agent_workflows.physics import scene_ops as physics_scene_ops
from content_agent_workflows.physics import workflow as physics_workflow


def test_apply_input_preserves_parent_owned_support_cache_identity(
    tmp_path: Path,
) -> None:
    cache: dict[str, dict[str, object]] = {}

    params = PhysicsApplyWorkflowInput(
        usd_path=tmp_path / "source.usda",
        output_dir=tmp_path / "run",
        ground_clearance_support_cache=cache,
    )

    assert params.ground_clearance_support_cache is cache
    assert "ground_clearance_support_cache" not in params.model_dump(mode="json")


@pytest.mark.parametrize(
    ("reason_code", "expected_count", "unexpected_count"),
    [
        ("raw_point_estimate_exceeded", 400_001, 300_001),
        ("unique_point_bound_exceeded", 300_001, 400_001),
    ],
)
def test_conservative_penetration_warning_reports_breaching_count(
    reason_code: str,
    expected_count: int,
    unexpected_count: int,
) -> None:
    """Bound diagnostics identify the count that triggered each fallback."""

    warning = physics_scene_ops._conservative_penetration_warning(
        {
            "reason_code": reason_code,
            "raw_point_count_estimate": 400_001,
            "observed_unique_point_count": 300_001,
            "processing_bound": 300_000,
        }
    )

    assert f"estimated points {expected_count}" in warning
    assert f"estimated points {unexpected_count}" not in warning


def test_physics_workflow_rejects_symlinked_output_before_authoring(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    victim = tmp_path / "victim.usda"
    victim.write_text("unchanged\n", encoding="utf-8")
    output_path = output_dir / "physics.usda"
    output_path.symlink_to(victim)

    with pytest.raises(ValueError, match="without traversing symlinks"):
        run_physics_apply_workflow(
            PhysicsApplyWorkflowInput(
                usd_path=tmp_path / "source.usda",
                output_dir=output_dir,
                output_usd_path=output_path,
            )
        )

    assert victim.read_text(encoding="utf-8") == "unchanged\n"


@pytest.mark.parametrize("link_kind", ["final", "intermediate", "dangling"])
def test_physics_workflow_rejects_symlinked_output_directory_before_writing(
    tmp_path: Path,
    link_kind: str,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    if link_kind == "final":
        output_dir = tmp_path / "run"
        output_dir.symlink_to(outside, target_is_directory=True)
    elif link_kind == "intermediate":
        link = tmp_path / "linked-parent"
        link.symlink_to(outside, target_is_directory=True)
        output_dir = link / "run"
    else:
        link = tmp_path / "dangling-parent"
        link.symlink_to(tmp_path / "missing-target", target_is_directory=True)
        output_dir = link / "run"

    with pytest.raises(ValueError, match="without traversing symlinks"):
        run_physics_apply_workflow(
            PhysicsApplyWorkflowInput(
                usd_path=tmp_path / "source.usda",
                output_dir=output_dir,
            )
        )

    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert sorted(path.name for path in outside.iterdir()) == ["sentinel.txt"]


def test_physics_workflow_rejects_symlinked_external_output_parent_before_writing(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    linked_parent = tmp_path / "linked-output"
    linked_parent.symlink_to(outside, target_is_directory=True)
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="without traversing symlinks"):
        run_physics_apply_workflow(
            PhysicsApplyWorkflowInput(
                usd_path=tmp_path / "source.usda",
                output_dir=run_dir,
                output_usd_path=linked_parent / "physics.usda",
            )
        )

    assert not run_dir.exists()
    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert sorted(path.name for path in outside.iterdir()) == ["sentinel.txt"]


def test_physics_workflow_rejects_source_as_output(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not overwrite the source"):
        run_physics_apply_workflow(
            PhysicsApplyWorkflowInput(
                usd_path=source,
                output_dir=tmp_path / "run",
                output_usd_path=source,
            )
        )

    assert source.read_text(encoding="utf-8") == "#usda 1.0\n"


def test_physics_workflow_rejects_hardlinked_source_output(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "output.usda"
    output.hardlink_to(source)

    with pytest.raises(ValueError, match="must not overwrite the source"):
        run_physics_apply_workflow(
            PhysicsApplyWorkflowInput(
                usd_path=source,
                output_dir=tmp_path / "run",
                output_usd_path=output,
            )
        )

    assert source.read_text(encoding="utf-8") == "#usda 1.0\n"


def test_physics_finalizer_json_write_rejects_child_symlink(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    victim = tmp_path / "victim.json"
    victim.write_text('{"preserve":true}\n', encoding="utf-8")
    artifact = raw_dir / "physics_components.json"
    artifact.symlink_to(victim)

    with pytest.raises(ValueError, match="regular file or absent"):
        physics_workflow._write_json(
            artifact,
            {"overwritten": True},
            within=run_dir,
        )

    assert victim.read_text(encoding="utf-8") == '{"preserve":true}\n'


def test_physics_child_json_inputs_reject_symlinks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    external = tmp_path / "external.json"
    external.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.physics-behavior-assessment.v1"
                ),
                "status": "pass",
            }
        ),
        encoding="utf-8",
    )

    decision_link = raw_dir / "physics_decision_patch.json"
    decision_link.symlink_to(external)
    with pytest.raises(ValueError, match="symlinks"):
        physics_workflow._load_decision_patch_payload(
            decision_link,
            within=run_dir,
        )

    assessment_link = run_dir / "physics_behavior_assessment.json"
    assessment_link.symlink_to(external)
    with pytest.raises(ValueError, match="symlinks"):
        physics_workflow.load_physics_behavior_assessment(
            assessment_link,
            within=run_dir,
        )


def test_physics_child_json_input_preserves_cwd_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    patch_path = raw_dir / "physics_decision_patch.json"
    patch_path.write_text('{"schema_version":"test"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    relative_path = Path("run/raw/physics_decision_patch.json")
    assert physics_workflow._contained_input_root(relative_path, run_dir) == run_dir
    read_path, payload = physics_workflow._read_json_object(
        relative_path,
        within=run_dir,
    )

    assert read_path == patch_path
    assert payload == {"schema_version": "test"}


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


def _component_inspection(
    usd_path: Path | str,
    *,
    path_space: str = "source",
) -> dict[str, object]:
    return {
        "asset": str(usd_path),
        "source_digest": "sha256:test",
        "component_count": 1,
        "components": [
            {
                "component_id": "component_001",
                "path_space": path_space,
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


def _authoring_target_id(component_id: str, role: str, prim_path: str) -> str:
    digest = hashlib.sha256(prim_path.encode("utf-8")).hexdigest()
    return f"{component_id}:{role}:{digest}"


def _fake_usd_cli_apply_success(**kwargs: object) -> dict[str, object]:
    """Return a raw usd-cli receipt without invoking a sidecar in unit tests."""

    output = Path(str(kwargs["output_usd"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("#usda 1.0\n", encoding="utf-8")
    raw_dir = Path(str(kwargs["raw_dir"]))
    command_record = raw_dir / "usd_cli_physics_commands.json"
    command_record.write_text('{"commands": []}\n', encoding="utf-8")
    author_rigid_body = bool(kwargs.get("author_rigid_body", True))
    return {
        "scene_backend": "usd-cli",
        "scene_tool_transport": "usd-cli-tel",
        "physics_usd": str(output),
        "command_record_path": str(command_record),
        "structural_validation": {
            "ok": True,
            "data": {
                "checks": {
                    "scenes": 1,
                    "rigid_bodies": int(author_rigid_body),
                    "enabled_rigid_bodies": int(author_rigid_body),
                    "colliders": 1,
                },
                "issues": [],
            },
        },
    }


@pytest.fixture(autouse=True)
def _stub_usd_cli_apply_for_workflow_tests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Keep workflow-policy tests independent of the authenticated sidecar."""

    if request.node.name.startswith("test_usd_cli_physics_"):
        return
    (tmp_path / "input.usda").write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        _fake_usd_cli_apply_success,
    )
    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "select_physics_scene_path",
        lambda _path: "/World/PhysicsScene",
    )


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


def test_physics_visual_behavior_includes_digest_bound_render_receipt(
    tmp_path: Path,
) -> None:
    evidence = physics_validation_evidence(
        asset="/tmp/asset.usda",
        target_runtime="ovphysx",
        physics_properties_status="pass",
        runtime_loadability_status="pass",
        no_explosions_status="pass",
    )
    receipt = tmp_path / "physics_render_frame_receipt_1.json"
    receipt.write_bytes(b'{"schema_version":"test"}\n')

    merged = merge_physics_behavior_assessment(
        evidence,
        PhysicsBehaviorAssessment(
            status="pass",
            rendered_frames=["/tmp/frame_0000.png"],
            runtime_report="/tmp/runtime_validation_report.json",
            assessment_notes="Behavior is plausible.",
        ),
        render_receipt_path=receipt,
    )

    artifact = next(
        item
        for item in merged.checks[-1].evidence_artifacts
        if item.kind == "simulation_frame_receipt"
    )
    assert artifact.path == str(receipt.resolve())
    assert (
        artifact.metadata["sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    )
    assert artifact in merged.evidence_artifacts


def test_physics_visual_behavior_does_not_treat_view_summary_as_artifact_path() -> None:
    evidence = physics_validation_evidence(
        asset="/tmp/asset.usda",
        target_runtime="ovphysx",
        physics_properties_status="pass",
        runtime_loadability_status="pass",
        no_explosions_status="pass",
    )
    summary = '{"view":"three_quarter","finding":"stable"}'

    merged = merge_physics_behavior_assessment(
        evidence,
        PhysicsBehaviorAssessment(
            status="pass",
            checked_views=[summary],
            rendered_frames=[],
            runtime_report="/tmp/runtime_validation_report.json",
            assessment_notes="Behavior is plausible.",
        ),
    )

    artifacts = merged.checks[-1].evidence_artifacts
    assert all(artifact.path != summary for artifact in artifacts)
    assert all(artifact.kind != "simulation_frame" for artifact in artifacts)


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
            run_simulation=False,
            simulation_engine="none",
        )
    )

    assert result.success
    assert result.physics_usd_path is not None
    assert Path(result.physics_usd_path).exists()
    assert Path(result.physics_usd_path).name == "physics.usdc"
    assert result.assignments_path is not None
    assert result.validation_evidence_path is not None

    assignments = json.loads(Path(result.assignments_path).read_text())
    evidence = json.loads(Path(result.validation_evidence_path).read_text())
    components = json.loads(Path(result.components_path or "").read_text())

    assert assignments["candidate_count"] == 1
    assert assignments["decision_count"] == 1
    assert assignments["path_space"] == "source"
    assert (
        assignments["source_asset_sha256"]
        == hashlib.sha256(_simple_cube().read_bytes()).hexdigest()
    )
    assert assignments["prepared_asset_sha256"] == assignments["source_asset_sha256"]
    assert assignments["apply_report"]["collision_count"] == 1
    assert evidence["workflow"] == "physics_authoring"
    assert components["components"][0]["authoring_targets"] == [
        {
            "prim_path": "/World/Cube",
            "role": "visual",
            "target_id": _authoring_target_id("component_001", "visual", "/World/Cube"),
        }
    ]


def test_embedded_physics_workflow_preserves_outer_run_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    outer_request = output_dir / "request.json"
    outer_manifest = output_dir / "workflow_run_manifest.json"
    outer_request.write_text('{"workflow":"outer"}\n', encoding="utf-8")
    outer_manifest.write_text('{"workflow":"outer"}\n', encoding="utf-8")
    component = physics_workflow.PhysicsComponent.model_validate(
        _component_inspection(source)["components"][0]
    )

    monkeypatch.setattr(
        physics_workflow,
        "_inspect_workflow_components",
        lambda _params, _path: ([component], "sha256:test"),
    )

    def fake_apply_schema(**kwargs):
        output = Path(kwargs["output_usd_path"])
        output.write_text("#usda 1.0\n", encoding="utf-8")
        return {
            "physics_usd": str(output),
            "rigid_body_count": 1,
            "enabled_rigid_body_count": 1,
            "collision_count": 1,
            "physics_scene_paths": ["/World/PhysicsScene"],
        }

    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        _fake_usd_cli_apply_success,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=source,
            output_dir=output_dir,
            run_simulation=False,
            simulation_engine="none",
            workflow_run_record_subdir=(
                Path("raw") / "physics_workflow" / "iteration-0001"
            ),
        )
    )

    assert result.success
    assert outer_request.read_text(encoding="utf-8") == '{"workflow":"outer"}\n'
    assert outer_manifest.read_text(encoding="utf-8") == '{"workflow":"outer"}\n'
    nested_manifest = Path(result.workflow_run_manifest_path or "")
    assert nested_manifest == (
        output_dir
        / "raw"
        / "physics_workflow"
        / "iteration-0001"
        / "workflow_run_manifest.json"
    )
    assert nested_manifest.is_file()


def test_physics_workflow_records_manifest_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    component = physics_workflow.PhysicsComponent.model_validate(
        _component_inspection(source)["components"][0]
    )

    monkeypatch.setattr(
        physics_workflow,
        "_inspect_workflow_components",
        lambda _params, _path: ([component], "sha256:test"),
    )

    def fake_apply_schema(**kwargs):
        output = Path(kwargs["output_usd_path"])
        output.write_text("#usda 1.0\n", encoding="utf-8")
        return {
            "physics_usd": str(output),
            "rigid_body_count": 1,
            "enabled_rigid_body_count": 1,
            "collision_count": 1,
            "physics_scene_paths": ["/World/PhysicsScene"],
        }

    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        _fake_usd_cli_apply_success,
    )

    params = PhysicsApplyWorkflowInput(
        usd_path=source,
        output_dir=tmp_path / "run",
        run_simulation=False,
        simulation_engine="none",
    )
    result = run_physics_apply_workflow(params)

    assert result.success
    run_manifest = json.loads(
        Path(result.workflow_run_manifest_path or "").read_text(encoding="utf-8")
    )
    assert run_manifest["workflow"] == "physics_authoring"
    assert run_manifest["status"] == "pass"
    assert run_manifest["source_path"] == str(source.resolve())
    assert run_manifest["source_sha256"]
    assert run_manifest["backend"] == {
        "low_level_operations": [
            "physics.apply",
            "physics.validate",
            "save",
        ],
        "renderer": {"required": "ovrtx", "probe_timing": "before_scene_mutation"},
        "scene_tool": "usd-cli",
        "simulation_engine": "none",
        "transport": "usd-cli-tel",
        "workflow_owned_helpers": [
            "component_grouping",
            "topology_plan_guarded_apply",
            "authored_physics_summary",
            "runtime_acceptance_and_evidence",
        ],
    }
    assert run_manifest["policy"]["run_simulation"] is False
    assert set(run_manifest["required_artifacts"]) == {
        "assignments",
        "decision_patch",
        "components",
        "predictions",
        "apply_report",
        "validation_evidence",
    }
    assert run_manifest["checkpoints"][-1]["phase"] == "validated"

    resumed = run_physics_apply_workflow(params.model_copy(update={"resume": True}))
    assert resumed.success
    resumed_manifest = json.loads(
        Path(resumed.workflow_run_manifest_path or "").read_text(encoding="utf-8")
    )
    assert [item["phase"] for item in resumed_manifest["checkpoints"]] == [
        "validated",
        "validated",
    ]


def test_physics_workflow_usd_cli_applies_only_validated_low_level_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    component = physics_workflow.PhysicsComponent.model_validate(
        _component_inspection(source)["components"][0]
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        physics_workflow,
        "_inspect_workflow_components",
        lambda _params, _path: ([component], "sha256:test"),
    )

    def fake_usd_cli_apply(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        output = Path(str(kwargs["output_usd"]))
        output.write_text("#usda 1.0\n", encoding="utf-8")
        command_record = tmp_path / "run" / "raw" / "usd_cli_physics_commands.json"
        command_record.parent.mkdir(parents=True, exist_ok=True)
        command_record.write_text('{"commands": []}\n', encoding="utf-8")
        return {
            "scene_backend": "usd-cli",
            "scene_tool_transport": "usd-cli-tel",
            "physics_usd": str(output),
            "command_record_path": str(command_record),
            "structural_validation": {
                "ok": True,
                "data": {
                    "checks": {
                        "scenes": 1,
                        "rigid_bodies": 1,
                        "enabled_rigid_bodies": 1,
                        "colliders": 1,
                    },
                    "issues": [],
                },
            },
        }

    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        fake_usd_cli_apply,
    )
    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=source,
            output_dir=tmp_path / "run",
            scene_backend="usd-cli",
            run_simulation=False,
            simulation_engine="none",
        )
    )

    assert result.success
    assert captured["source_usd"] == source.resolve()
    decisions = captured["workflow_decisions"]
    assert isinstance(decisions, list)
    assert decisions[0]["component_id"] == component.component_id
    assert decisions[0]["collider_paths"] == component.visual_evidence_paths
    assert result.scene_operation_record_path
    assignments = json.loads(Path(result.assignments_path or "").read_text())
    assert assignments["scene_backend"] == "usd-cli"
    assert assignments["apply_report"]["scene_operation_backend"] == "usd-cli"
    assert assignments["apply_report"]["inspection_backend"] == "usd-cli"
    assert assignments["apply_report"]["inspection_command"] == "physics.validate"
    manifest = json.loads(
        Path(result.workflow_run_manifest_path or "").read_text(encoding="utf-8")
    )
    assert manifest["backend"]["scene_tool"] == "usd-cli"
    assert manifest["backend"]["transport"] == "usd-cli-tel"
    assert manifest["backend"]["renderer"] == {
        "required": "ovrtx",
        "probe_timing": "before_scene_mutation",
    }
    assert manifest["backend"]["low_level_operations"] == [
        "physics.apply",
        "physics.validate",
        "save",
    ]
    assert "component_grouping" in manifest["backend"]["workflow_owned_helpers"]


def test_physics_workflow_usd_cli_rejects_legacy_patch_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    source_before = source.read_bytes()
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

    def fail_scene_operation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unsupported legacy patch must not mutate a scene")

    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        fail_scene_operation,
    )
    monkeypatch.setattr(
        physics_workflow,
        "_inspect_workflow_components",
        fail_scene_operation,
    )

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    stale_output = output_dir / "physics.usdc"
    stale_output.write_bytes(b"stale output from a prior attempt\n")
    stale_output_before = stale_output.read_bytes()
    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=source,
            output_dir=output_dir,
            decision_patch_path=patch_path,
            scene_backend="usd-cli",
            run_simulation=False,
            simulation_engine="none",
        )
    )

    assert not result.success
    assert result.physics_usd_path is None
    assert result.error is not None
    assert "Legacy V1 physics decision patches are not supported" in result.error
    assert "Regenerate a V2 component decision patch" in result.error
    assert "no scene mutation was attempted" in result.error
    assert source.read_bytes() == source_before
    assert stale_output.read_bytes() == stale_output_before
    manifest = json.loads(
        Path(result.workflow_run_manifest_path or "").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "fail"
    assert manifest["backend"]["scene_tool"] == "usd-cli"
    assert manifest["backend"]["transport"] == "usd-cli-tel"
    assert manifest["failure"]["error"] == result.error
    assert manifest["artifacts"] == []
    assert manifest["checkpoints"] == []


def test_physics_workflow_usd_cli_fails_on_raw_structural_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    component = physics_workflow.PhysicsComponent.model_validate(
        _component_inspection(source)["components"][0]
    )
    monkeypatch.setattr(
        physics_workflow,
        "_inspect_workflow_components",
        lambda _params, _path: ([component], "sha256:test"),
    )

    def fake_usd_cli_apply(**kwargs: object) -> dict[str, object]:
        output = Path(str(kwargs["output_usd"]))
        output.write_text("#usda 1.0\n", encoding="utf-8")
        command_record = tmp_path / "run" / "raw" / "usd_cli_physics_commands.json"
        command_record.parent.mkdir(parents=True, exist_ok=True)
        command_record.write_text('{"commands": []}\n', encoding="utf-8")
        return {
            "scene_backend": "usd-cli",
            "physics_usd": str(output),
            "command_record_path": str(command_record),
            "structural_validation": {
                "ok": False,
                "data": {
                    "checks": {
                        "scenes": 1,
                        "rigid_bodies": 1,
                        "enabled_rigid_bodies": 1,
                        "colliders": 1,
                    },
                    "issues": ["nested rigid body"],
                },
            },
        }

    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        fake_usd_cli_apply,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=source,
            output_dir=tmp_path / "run",
            scene_backend="usd-cli",
            run_simulation=False,
            simulation_engine="none",
        )
    )

    assert not result.success
    evidence = json.loads(Path(result.validation_evidence_path or "").read_text())
    assert "nested rigid body" in evidence["failures"]


def test_usd_cli_physics_patch_translation_preserves_workflow_policy() -> None:
    patch = physics_workflow.usd_cli_ops.physics_patch_from_workflow_decisions(
        [
            {
                "decision_id": "component-1",
                "component_id": "component-1",
                "mass_authoring_path": "/World/Body",
                "collider_paths": ["/World/Body/Collider"],
                "collision_approximation": "convexHull",
                "physical_properties": {
                    "density": 900.0,
                    "estimated_mass_kg": 2.5,
                    "static_friction": 0.6,
                    "dynamic_friction": 0.5,
                    "restitution": 0.1,
                },
            }
        ],
        author_rigid_body=True,
        physics_scene_path="/World/PhysicsScene",
        physics_material_scope_path="/World/Looks",
    )

    assert patch == {
        "scene_paths": ["/World/PhysicsScene"],
        "rigid_bodies": [{"path": "/World/Body", "density": 900.0, "mass": 2.5}],
        "colliders": [{"path": "/World/Body/Collider", "approximation": "convexHull"}],
        "materials": [
            {
                "path": "/World/Looks/Physics_component_1",
                "static_friction": 0.6,
                "dynamic_friction": 0.5,
                "restitution": 0.1,
            }
        ],
        "bindings": [
            {
                "target_path": "/World/Body",
                "material_path": "/World/Looks/Physics_component_1",
            }
        ],
    }


def test_usd_cli_physics_deinstances_an_explicit_scene_target(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd

    source = tmp_path / "instanced_scene.usda"
    source.write_text(
        "#usda 1.0\n"
        'def Xform "Template"\n{\n'
        '    def PhysicsScene "Scene" {}\n'
        "}\n"
        'def Xform "Copy" (\n'
        "    instanceable = true\n"
        "    prepend references = </Template>\n"
        ") {}\n",
        encoding="utf-8",
    )
    root_layer = Sdf.Layer.FindOrOpen(str(source))
    assert root_layer is not None
    source_text = root_layer.ExportToString()
    assert not root_layer.dirty

    roots = physics_workflow.usd_cli_ops._deinstance_roots_for_patch(
        source,
        {"scene_paths": ["/Copy/Scene"]},
    )

    assert roots == ["/Copy"]
    assert not root_layer.dirty
    assert root_layer.ExportToString() == source_text
    reopened = Usd.Stage.Open(root_layer)
    assert reopened.GetPrimAtPath("/Copy").IsInstance()
    assert (
        physics_workflow.usd_cli_ops._deinstance_roots_for_patch(
            source,
            {"scene_paths": ["/World/NewPhysicsScene"]},
        )
        == []
    )


def test_usd_cli_physics_deinstances_instance_root_for_targets_and_new_children(
    tmp_path: Path,
) -> None:
    source = tmp_path / "instanced_body.usda"
    source.write_text(
        "#usda 1.0\n"
        'def Xform "Template"\n{\n'
        '    def Mesh "Collider" {}\n'
        "}\n"
        'def Xform "Copy" (\n'
        "    instanceable = true\n"
        "    prepend references = </Template>\n"
        ") {}\n",
        encoding="utf-8",
    )

    roots = physics_workflow.usd_cli_ops._deinstance_roots_for_patch(
        source,
        {
            "scene_paths": ["/Copy/PhysicsScene"],
            "rigid_bodies": [{"path": "/Copy"}],
            "colliders": [{"path": "/Copy/Collider"}],
            "materials": [{"path": "/Copy/Looks/Physics_body"}],
            "bindings": [{"target_path": "/Copy"}],
        },
    )

    assert roots == ["/Copy"]


def test_usd_cli_physics_material_scope_uses_stage_default_prim(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    source = tmp_path / "default_prim.usda"
    stage = Usd.Stage.CreateNew(str(source))
    asset_root = UsdGeom.Xform.Define(stage, "/AssetRoot")
    stage.SetDefaultPrim(asset_root.GetPrim())
    stage.GetRootLayer().Save()

    assert (
        physics_workflow.usd_cli_ops._physics_material_scope_path(source)
        == "/AssetRoot/Looks"
    )


def test_usd_cli_physics_material_names_disambiguate_sanitization_collisions() -> None:
    patch = physics_workflow.usd_cli_ops.physics_patch_from_workflow_decisions(
        [
            {
                "component_id": "component-1",
                "mass_authoring_path": "/World/BodyA",
                "collider_paths": ["/World/BodyA/Collider"],
                "collision_approximation": "convexHull",
                "physical_properties": {
                    "static_friction": 0.2,
                    "dynamic_friction": 0.1,
                    "restitution": 0.0,
                },
            },
            {
                "component_id": "component_1",
                "mass_authoring_path": "/World/BodyB",
                "collider_paths": ["/World/BodyB/Collider"],
                "collision_approximation": "convexHull",
                "physical_properties": {
                    "static_friction": 0.8,
                    "dynamic_friction": 0.7,
                    "restitution": 0.5,
                },
            },
        ],
        author_rigid_body=True,
        physics_scene_path="/World/PhysicsScene",
        physics_material_scope_path="/World/Looks",
    )

    assert [material["path"] for material in patch["materials"]] == [
        "/World/Looks/Physics_component_1",
        "/World/Looks/Physics_component_1_2",
    ]
    assert patch["materials"][0]["static_friction"] == 0.2
    assert patch["materials"][1]["static_friction"] == 0.8
    assert patch["bindings"] == [
        {
            "target_path": "/World/BodyA",
            "material_path": "/World/Looks/Physics_component_1",
        },
        {
            "target_path": "/World/BodyB",
            "material_path": "/World/Looks/Physics_component_1_2",
        },
    ]


def test_usd_cli_physics_material_names_avoid_existing_visual_material(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom, UsdShade

    source = tmp_path / "visual_material_collision.usda"
    stage = Usd.Stage.CreateNew(str(source))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdShade.Material.Define(stage, "/World/Looks/Physics_component_1")
    stage.GetRootLayer().Save()

    scope_path = physics_workflow.usd_cli_ops._physics_material_scope_path(source)
    reserved = physics_workflow.usd_cli_ops._reserved_physics_material_paths(
        source, scope_path
    )
    patch = physics_workflow.usd_cli_ops.physics_patch_from_workflow_decisions(
        [
            {
                "component_id": "component-1",
                "mass_authoring_path": "/World/Body",
                "collider_paths": ["/World/Body/Collider"],
                "physical_properties": {"static_friction": 0.6},
            }
        ],
        author_rigid_body=True,
        physics_scene_path="/World/PhysicsScene",
        physics_material_scope_path=scope_path,
        reserved_material_paths=reserved,
    )

    assert reserved == {"/World/Looks/Physics_component_1"}
    assert patch["materials"][0]["path"] == "/World/Looks/Physics_component_1_2"
    assert patch["bindings"][0]["material_path"] == (
        "/World/Looks/Physics_component_1_2"
    )


def test_usd_cli_physics_scene_policy_reuses_existing_scene_for_preserved_colliders(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    source = tmp_path / "existing_scene.usda"
    stage = Usd.Stage.CreateNew(str(source))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Simulation")
    UsdPhysics.Scene.Define(stage, "/World/Simulation/ExistingScene")
    stage.GetRootLayer().Save()

    scene_path = physics_workflow.scene_ops.select_physics_scene_path(source)
    patch = physics_workflow.usd_cli_ops.physics_patch_from_workflow_decisions(
        [
            {
                "decision_id": "component-1",
                "component_id": "component-1",
                "collision_mode": "preserve_existing",
                "mass_authoring_path": "/World/Body",
                "collider_paths": ["/World/Body/Collider"],
                "collision_approximation": "convexHull",
                "physical_properties": {"density": 900.0},
            }
        ],
        author_rigid_body=True,
        physics_scene_path=scene_path,
        physics_material_scope_path="/World/Looks",
    )

    assert scene_path == "/World/Simulation/ExistingScene"
    assert patch["scene_paths"] == ["/World/Simulation/ExistingScene"]
    assert "/PhysicsScene" not in patch["scene_paths"]


def test_usd_cli_physics_scene_policy_places_packaged_asset_scene_under_default_prim(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom, UsdUtils

    source = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(source))
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    stage.GetRootLayer().Save()
    package = tmp_path / "asset.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(source), str(package))

    scene_path = physics_workflow.scene_ops.select_physics_scene_path(package)

    assert scene_path == "/Asset/PhysicsScene"
    assert scene_path != "/PhysicsScene"


def test_usd_cli_physics_apply_rejects_unready_ovrtx_before_scene_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    calls: list[list[str]] = []

    def fail_probe(**kwargs: object) -> dict[str, object]:
        arguments = kwargs["arguments"]
        assert isinstance(arguments, list)
        calls.append(arguments)
        raise physics_workflow.usd_cli_ops.UsdCliPhysicsError("OVRTX unavailable")

    monkeypatch.setattr(physics_workflow.usd_cli_ops, "_run_json", fail_probe)
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "_create_owned_session",
        lambda *, run_root, **_kwargs: SimpleNamespace(
            project_dir=run_root,
            session_id="physics-owned",
            close=lambda: None,
        ),
    )

    with pytest.raises(
        physics_workflow.usd_cli_ops.UsdCliPhysicsError,
        match="OVRTX unavailable",
    ):
        physics_workflow.usd_cli_ops.apply_physics_patch(
            source_usd=source,
            output_usd=run_dir / "physics.usda",
            raw_dir=raw_dir,
            workflow_decisions=[
                {
                    "component_id": "component-1",
                    "mass_authoring_path": "/World/Body",
                    "collider_paths": ["/World/Collider"],
                    "collision_approximation": "convexHull",
                    "physical_properties": {"density": 1000.0},
                }
            ],
            author_rigid_body=True,
            physics_scene_path="/World/PhysicsScene",
        )

    assert calls == [
        [
            "render-probe",
            "--require-engine",
            "ovrtx",
            "--output-dir",
            str(run_dir / "ovrtx_probe"),
        ]
    ]
    assert not (run_dir / "physics.usda").exists()


def test_usd_cli_physics_apply_force_reloads_shared_session_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finalizer authors from the staged source as it exists on disk.

    When the shared workflow session still holds unsaved exploration edits on
    that file (the agentic child inspects without saving), a plain ``open`` is
    refused by usd-cli's same-session reload guard and every finalize pass
    fails. The re-open must carry ``--force-reload`` to discard those scratch
    edits deliberately.
    """
    source = tmp_path / "source.usda"
    source.write_text(
        "#usda 1.0\n"
        'def Xform "World"\n{\n'
        '    def Xform "Body" {}\n'
        '    def Mesh "Collider" {}\n'
        "}\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    calls: list[list[str]] = []

    class _StopAfterOpen(RuntimeError):
        pass

    def record_run_json(**kwargs: object) -> dict[str, object]:
        arguments = kwargs["arguments"]
        assert isinstance(arguments, list)
        calls.append(arguments)
        if arguments[0] == "checkpoint":
            raise _StopAfterOpen()
        return {"ok": True}

    monkeypatch.setattr(physics_workflow.usd_cli_ops, "_run_json", record_run_json)
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "_require_ovrtx_probe",
        lambda *_args, **_kwargs: None,
    )
    shared_session = SimpleNamespace(
        project_dir=run_dir,
        session_id="workflow-shared",
    )

    with pytest.raises(_StopAfterOpen):
        physics_workflow.usd_cli_ops.apply_physics_patch(
            source_usd=source,
            output_usd=run_dir / "physics.usda",
            raw_dir=raw_dir,
            workflow_decisions=[
                {
                    "component_id": "component-1",
                    "mass_authoring_path": "/World/Body",
                    "collider_paths": ["/World/Collider"],
                    "collision_approximation": "convexHull",
                    "physical_properties": {"density": 1000.0},
                }
            ],
            author_rigid_body=True,
            physics_scene_path="/World/PhysicsScene",
            usd_cli_session=shared_session,  # type: ignore[arg-type]
        )

    open_calls = [call for call in calls if call and call[0] == "open"]
    assert open_calls == [["open", str(source), "--force-reload"]]


def test_usd_cli_physics_apply_rejects_symlink_output_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    external = tmp_path / "external.usda"
    external.write_text("external\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    output = run_dir / "physics.usda"
    output.symlink_to(external)

    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "_run_json",
        lambda **_kwargs: pytest.fail("no usd-cli command may run"),
    )

    with pytest.raises(
        physics_workflow.usd_cli_ops.UsdCliPhysicsError,
        match="output target is unsafe",
    ):
        physics_workflow.usd_cli_ops.apply_physics_patch(
            source_usd=source,
            output_usd=output,
            raw_dir=raw_dir,
            workflow_decisions=[
                {
                    "component_id": "component-1",
                    "mass_authoring_path": "/World/Body",
                    "collider_paths": ["/World/Collider"],
                    "collision_approximation": "convexHull",
                    "physical_properties": {"density": 1000.0},
                }
            ],
            author_rigid_body=True,
            physics_scene_path="/World/PhysicsScene",
        )

    assert external.read_text(encoding="utf-8") == "external\n"


def test_usd_cli_physics_simulation_uses_explicit_workflow_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scene = run_dir / "drop_scenario.usda"
    scene.write_text("#usda 1.0\n", encoding="utf-8")
    output = run_dir / "runtime" / "usd_cli_simulation"
    calls: list[list[str]] = []

    def fake_run_json(**kwargs: object) -> dict[str, object]:
        arguments = kwargs["arguments"]
        assert isinstance(arguments, list)
        calls.append(arguments)
        if arguments[0] == "open":
            return {"ok": True}
        return {
            "ok": True,
            "data": {
                "engine": "ovphysx",
                "trajectory_jsonl": str(output / "trajectory.jsonl"),
                "recording_usda": str(output / "recording.usda"),
                "report_path": str(output / "runtime_validation_report.json"),
                "simulation_facts": {"trajectory_sample_count": 3},
            },
        }

    monkeypatch.setattr(physics_workflow.usd_cli_ops, "_run_json", fake_run_json)
    session = SimpleNamespace(project_dir=run_dir, session_id="physics-owned")

    result = physics_workflow.usd_cli_ops.simulate_physics_scene(
        scene_usd=scene,
        output_dir=output,
        body_path="/World/Body",
        body_pattern="/World/Body*",
        rest_position=[0.0, 0.0, 1.0],
        world_up=[0.0, 0.0, 1.0],
        duration_s=2.0,
        dt=1.0 / 240.0,
        sample_fps=30,
        usd_cli_session=session,
    )

    assert result["engine"] == "ovphysx"
    assert calls[0] == ["open", str(scene)]
    assert calls[1][:4] == ["physics", "simulate", "--scene", str(scene)]
    assert ["--body", "/World/Body"] == calls[1][4:6]
    assert ["--body-pattern", "/World/Body*"] == calls[1][6:8]
    assert "--rest-position" in calls[1]
    assert "--world-up" in calls[1]
    assert "--output" in calls[1]


def test_runtime_policy_authors_scenario_above_usd_cli_simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent import recording
    from physics_agent.tuning.scenarios import _scene_builder
    from world_understanding.functions.physics import trajectory as trajectory_module

    physics_usd = tmp_path / "physics.usda"
    physics_usd.write_text("#usda 1.0\n", encoding="utf-8")
    output_dir = tmp_path / "runtime"
    session = SimpleNamespace(project_dir=tmp_path, session_id="physics-owned")
    calls: dict[str, object] = {}

    def fake_build_drop_settle_scene(
        source: Path,
        destination: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        calls["scenario_source"] = source
        calls["scenario_options"] = kwargs
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("#usda 1.0\n", encoding="utf-8")
        return {
            "body_prim_path": "/World/Body",
            "body_pattern": "/World/Body*",
            "rest_position": [0.0, 0.0, 0.0],
            "world_up": [0.0, 0.0, 1.0],
            "drop_height_m_resolved": 0.05,
        }

    def fake_simulate_physics_scene(**kwargs: object) -> dict[str, object]:
        calls["simulation"] = kwargs
        simulation_dir = Path(str(kwargs["output_dir"]))
        simulation_dir.mkdir(parents=True, exist_ok=True)
        trajectory = simulation_dir / "trajectory.jsonl"
        trajectory.write_text(
            json.dumps(
                {
                    "t": 0.0,
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        recording_path = simulation_dir / "recording.usda"
        recording_path.write_text("#usda 1.0\n", encoding="utf-8")
        report_path = simulation_dir / "runtime_validation_report.json"
        report_path.write_text("{}\n", encoding="utf-8")
        return {
            "engine": "ovphysx",
            "n_bodies": 1,
            "trajectory_jsonl": str(trajectory),
            "recording_usda": str(recording_path),
            "report_path": str(report_path),
            "simulation_facts": {
                "reported_body_count": 1,
                "reported_step_count": 1,
            },
        }

    monkeypatch.setattr(
        _scene_builder,
        "build_drop_settle_scene",
        fake_build_drop_settle_scene,
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "simulate_physics_scene",
        fake_simulate_physics_scene,
    )
    monkeypatch.setattr(
        trajectory_module,
        "trajectory_summary",
        lambda *_args, **_kwargs: {"settle_time_s": 0.0},
    )
    monkeypatch.setattr(
        trajectory_module,
        "settle_distance",
        lambda *_args, **_kwargs: 0.0,
    )
    monkeypatch.setattr(
        recording,
        "author_trajectory_jsonl",
        lambda *_args, **_kwargs: pytest.fail(
            "workflow must preserve the usd-cli trajectory artifact"
        ),
    )
    monkeypatch.setattr(
        recording,
        "author_trajectory_usda",
        lambda *_args, **_kwargs: pytest.fail(
            "workflow must preserve the usd-cli recording artifact"
        ),
    )

    result = physics_workflow.scene_ops.validate_runtime(
        physics_usd=physics_usd,
        output_dir=output_dir,
        engine="ovphysx",
        drop_height_m=0.05,
        acceptance={
            "expected_body_count": 1,
            "require_gravity_response": False,
        },
        usd_cli_session=session,
    )

    simulation = calls["simulation"]
    assert isinstance(simulation, dict)
    assert calls["scenario_source"] == physics_usd
    assert simulation["scene_usd"] == output_dir / "drop_settle_scene.usda"
    assert simulation["body_path"] == "/World/Body"
    assert simulation["body_pattern"] == "/World/Body*"
    assert simulation["rest_position"] == [0.0, 0.0, 0.0]
    assert simulation["world_up"] == [0.0, 0.0, 1.0]
    assert simulation["usd_cli_session"] is session
    assert result["failures"] == []
    assert result["summary"]["loaded_body_count"] == 1
    assert result["acceptance"]["require_settle"] is True


def test_runtime_validation_rejects_disabled_settle_gate(tmp_path: Path) -> None:
    physics_usd = tmp_path / "physics.usda"
    physics_usd.write_text("#usda 1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="require_settle cannot be disabled"):
        physics_workflow.scene_ops.validate_runtime(
            physics_usd=physics_usd,
            output_dir=tmp_path / "runtime",
            engine="fake",
            acceptance={"require_settle": False},
        )


def test_common_prim_ancestor_returns_deepest_shared_prefix() -> None:
    assert (
        physics_workflow._common_prim_ancestor(
            ["/World/SimBodies/BodyA", "/World/SimBodies/BodyB"]
        )
        == "/World/SimBodies"
    )
    assert (
        physics_workflow._common_prim_ancestor(["/World/Body", "/World/Body/Child"])
        == "/World/Body"
    )
    assert physics_workflow._common_prim_ancestor(["/A/Body", "/B/Body"]) is None
    assert physics_workflow._common_prim_ancestor([]) is None


def _fake_usd_cli_apply_multi_body(
    body_paths: list[str],
) -> object:
    def fake_apply(**kwargs: object) -> dict[str, object]:
        receipt = _fake_usd_cli_apply_success(**kwargs)
        checks = receipt["structural_validation"]["data"]["checks"]  # type: ignore[index]
        checks["rigid_bodies"] = len(body_paths)  # type: ignore[index]
        checks["enabled_rigid_bodies"] = len(body_paths)  # type: ignore[index]
        receipt["enabled_rigid_body_paths"] = list(body_paths)
        # Author a real stage shaped like the body paths: /World is an
        # Xform, /World/SimBodies a TYPELESS grouping prim (as
        # UsdStage.DefinePrim creates ancestors), so the placement root
        # must resolve UP to /World — xform ops on the typeless common
        # ancestor would be silently ignored by USD.
        Path(str(receipt["physics_usd"])).write_text(
            "#usda 1.0\n"
            '(\n    defaultPrim = "World"\n)\n\n'
            'def Xform "World"\n{\n'
            '    def Xform "SimBodies"\n    {\n'
            '        def Xform "BodyA"\n        {\n        }\n'
            '        def Xform "BodyB" (\n'
            '            prepend apiSchemas = ["PhysicsRigidBodyAPI"]\n'
            "        )\n        {\n"
            "            bool physics:kinematicEnabled = 1\n"
            "        }\n"
            "    }\n}\n",
            encoding="utf-8",
        )
        return receipt

    return fake_apply


def _fake_per_body_validate_runtime(
    calls: list[dict[str, object]],
    *,
    failures_by_body: dict[str, list[str]] | None = None,
) -> object:
    def fake_validate_runtime(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        body_dir = Path(str(kwargs["output_dir"]))
        body_dir.mkdir(parents=True, exist_ok=True)
        body_path = str(kwargs["body_prim_path_hint"])
        report = body_dir / "runtime_validation_report.json"
        report.write_text('{"engine":"fake"}\n', encoding="utf-8")
        trajectory = body_dir / "trajectory.jsonl"
        trajectory.write_text(
            json.dumps(
                {
                    "t": 0.0,
                    "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        recording = body_dir / "recording.usda"
        # A real minimal stage with the body prims: the aggregate multi-body
        # recording sublayers this file and authors pose samples on the
        # OTHER bodies, so they must exist here.
        recording.write_text(
            "#usda 1.0\n"
            '(\n    defaultPrim = "World"\n)\n\n'
            'def Xform "World"\n{\n'
            '    def "SimBodies"\n    {\n'
            '        def Xform "BodyA"\n        {\n        }\n'
            '        def Xform "BodyB"\n        {\n        }\n'
            "    }\n}\n",
            encoding="utf-8",
        )
        body_failures = list((failures_by_body or {}).get(body_path, []))
        support_decision = {
            "schema_version": "physics-agent.ground-clearance-support-decision.v1",
            "policy_version": "raw-hull-input-bound.v1",
            "decision_id": f"support:{body_path}",
            "body_prim_path": body_path,
            "selected_support_type": "conservative_bbox_corners",
            "reason_code": "raw_point_estimate_exceeded",
            "raw_point_count_estimate": 400_001,
            "processing_bound": 400_000,
            "validation_outcome": "fail" if body_failures else "pass",
            "fallback_accepted": not body_failures,
        }
        return {
            "engine": "fake",
            "physics_usd": str(kwargs["physics_usd"]),
            "scene_usd": str(body_dir / "drop_settle_scene.usda"),
            "runtime_report": str(report),
            "trajectory_jsonl": str(trajectory),
            "recording_usda": str(recording),
            "settle_distance": 0.01,
            "summary": {"settle_time_s": 0.1},
            "failures": body_failures,
            "warnings": [],
            "diagnostics": [],
            "scene_info": {
                "ground_clearance_support_decision": support_decision,
            },
            "phase_timings_seconds": {
                "support_selection": 0.01,
                "scene_build": 0.02,
                "simulation": 0.03,
                "acceptance": 0.04,
            },
            "acceptance": dict(kwargs.get("acceptance") or {}),  # type: ignore[call-overload]
            "evidence_artifacts": [
                {
                    "kind": "runtime_report",
                    "path": str(report),
                    "description": "Runtime validation metrics.",
                },
                {
                    "kind": "recording_usda",
                    "path": str(recording),
                    "description": "Time-sampled USD recording.",
                },
            ],
        }

    return fake_validate_runtime


def test_multi_body_runtime_validation_runs_per_enabled_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body_paths = ["/World/SimBodies/BodyA", "/World/SimBodies/BodyB"]
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "inspect_components",
        lambda usd_path, *, path_space="source": _component_inspection(
            usd_path, path_space=path_space
        ),
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        _fake_usd_cli_apply_multi_body(body_paths),
    )
    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "validate_runtime",
        _fake_per_body_validate_runtime(calls),
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            # The fake apply writes an ASCII stage; a .usda output path keeps
            # it readable for the Xformable placement-root resolution.
            output_usd_path=tmp_path / "out" / "physics.usda",
            simulation_engine="fake",
            drop_height_m=0.0,
            max_ground_penetration_m=0.01,
        )
    )

    assert result.success
    assert [call["body_prim_path_hint"] for call in calls] == body_paths
    assert all(call["placement_prim_path_hint"] == "/World/SimBodies" for call in calls)
    assert all(
        call["acceptance"]
        == {"require_gravity_response": False, "max_ground_penetration_m": 0.01}
        for call in calls
    )
    report = json.loads(Path(result.simulation_report_path or "").read_text())
    assert report["mode"] == "multi_body"
    assert report.get("not_evaluated") is not True
    assert report["failures"] == []
    assert report["enabled_rigid_body_count"] == 2
    # BodyB is authored kinematic in the fixture: an anchored body correctly
    # does not fall, and the report records the exemption.
    assert [item["kinematic"] for item in report["per_body_results"]] == [
        False,
        True,
    ]
    assert [item["body_prim_path"] for item in report["per_body_results"]] == (
        body_paths
    )
    assert [item["status"] for item in report["per_body_results"]] == [
        "pass",
        "pass",
    ]
    rows = [
        json.loads(line)
        for line in Path(report["trajectory_jsonl"]).read_text().splitlines()
        if line.strip()
    ]
    assert [row["body_prim_path"] for row in rows] == body_paths
    evidence = json.loads(Path(result.validation_evidence_path or "").read_text())
    assert evidence["sim_ready_status"] == "pass"
    checks = {check["name"]: check["status"] for check in evidence["checks"]}
    assert checks["runtime_loadability"] == "pass"
    assert checks["no_explosions"] == "pass"
    assert evidence["metadata"]["runtime_validation_mode"] == "multi_body"
    assert [
        decision["body_prim_path"]
        for decision in report["ground_clearance_support_decisions"]
    ] == body_paths
    assert report["phase_timings_seconds"] == {
        "acceptance": pytest.approx(0.08),
        "scene_build": pytest.approx(0.04),
        "simulation": pytest.approx(0.06),
        "support_selection": pytest.approx(0.02),
    }
    assert (
        evidence["metadata"]["ground_clearance_support_decisions"]
        == report["ground_clearance_support_decisions"]
    )


def test_runtime_evidence_preserves_support_decision_diagnostics_and_timing(
    tmp_path: Path,
) -> None:
    report = tmp_path / "runtime_validation_report.json"
    report.write_text("{}\n", encoding="utf-8")
    decision = {
        "schema_version": "physics-agent.ground-clearance-support-decision.v1",
        "decision_id": "support-1",
        "selected_support_type": "conservative_bbox_corners",
        "reason_code": "raw_point_estimate_exceeded",
        "validation_outcome": "pass",
        "fallback_accepted": True,
    }

    evidence, returned_report = physics_workflow._runtime_result_to_evidence(
        result={
            "runtime_report": str(report),
            "failures": [],
            "warnings": [],
            "diagnostics": [
                "The authored collider was unchanged; this fallback is accepted."
            ],
            "summary": {"settle_time_s": 0.2},
            "settle_distance": 0.0,
            "scene_info": {"ground_clearance_support_decision": decision},
            "phase_timings_seconds": {
                "support_selection": 0.01,
                "simulation": 0.2,
            },
            "evidence_artifacts": [],
        },
        physics_usd_path=tmp_path / "physics.usda",
        engine="fake",
        duration_s=1.0,
        sample_fps=30,
    )

    assert returned_report == report
    assert evidence.sim_ready_status == "pass"
    assert evidence.warnings == []
    assert evidence.metadata["ground_clearance_support_decision"] == decision
    assert evidence.metadata["phase_timings_seconds"]["support_selection"] == 0.01
    assert "authored collider was unchanged" in evidence.metadata["diagnostics"][0]


def test_multi_body_runtime_validation_fails_when_any_body_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body_paths = ["/World/SimBodies/BodyA", "/World/SimBodies/BodyB"]
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "inspect_components",
        lambda usd_path, *, path_space="source": _component_inspection(
            usd_path, path_space=path_space
        ),
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        _fake_usd_cli_apply_multi_body(body_paths),
    )
    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "validate_runtime",
        _fake_per_body_validate_runtime(
            calls,
            failures_by_body={
                "/World/SimBodies/BodyB": [
                    "Simulation did not exhibit the expected gravity response."
                ]
            },
        ),
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            output_usd_path=tmp_path / "out" / "physics.usda",
            simulation_engine="fake",
        )
    )

    assert not result.success
    assert result.validation_status == "fail"
    assert len(calls) == 2
    report = json.loads(Path(result.simulation_report_path or "").read_text())
    assert report["failures"] == [
        "/World/SimBodies/BodyB: Simulation did not exhibit the expected "
        "gravity response."
    ]
    assert [item["status"] for item in report["per_body_results"]] == [
        "pass",
        "fail",
    ]
    evidence = json.loads(Path(result.validation_evidence_path or "").read_text())
    assert evidence["sim_ready_status"] == "fail"
    checks = {check["name"]: check["status"] for check in evidence["checks"]}
    assert checks["runtime_loadability"] == "fail"
    assert checks["no_explosions"] == "fail"


def test_multi_body_runtime_validation_caps_body_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body_paths = [
        f"/World/SimBodies/Body{index:02d}"
        for index in range(physics_workflow.MAX_MULTI_BODY_RUNTIME_BODIES + 1)
    ]
    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "inspect_components",
        lambda usd_path, *, path_space="source": _component_inspection(
            usd_path, path_space=path_space
        ),
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        _fake_usd_cli_apply_multi_body(body_paths),
    )
    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "validate_runtime",
        lambda **_kwargs: pytest.fail(
            "per-body validation must not run above the body-count cap"
        ),
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            simulation_engine="fake",
        )
    )

    assert result.success
    assert result.validation_status == "conditional"
    report = json.loads(Path(result.simulation_report_path or "").read_text())
    assert report["not_evaluated"] is True
    assert any("capped at" in warning for warning in report["warnings"])
    evidence = json.loads(Path(result.validation_evidence_path or "").read_text())
    checks = {check["name"]: check["status"] for check in evidence["checks"]}
    assert checks["runtime_loadability"] == "not_evaluated"


def test_multi_body_runtime_validation_skips_without_body_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_apply_without_paths(**kwargs: object) -> dict[str, object]:
        receipt = _fake_usd_cli_apply_success(**kwargs)
        checks = receipt["structural_validation"]["data"]["checks"]  # type: ignore[index]
        checks["rigid_bodies"] = 2  # type: ignore[index]
        checks["enabled_rigid_bodies"] = 2  # type: ignore[index]
        return receipt

    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "inspect_components",
        lambda usd_path, *, path_space="source": _component_inspection(
            usd_path, path_space=path_space
        ),
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        fake_apply_without_paths,
    )
    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "validate_runtime",
        lambda **_kwargs: pytest.fail(
            "per-body validation must not run without body paths"
        ),
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            simulation_engine="fake",
        )
    )

    assert result.success
    assert result.validation_status == "conditional"
    report = json.loads(Path(result.simulation_report_path or "").read_text())
    assert report["not_evaluated"] is True
    assert any("could not be determined" in warning for warning in report["warnings"])


def test_validate_physics_runtime_multi_body_fake_engine_end_to_end(
    tmp_path: Path,
) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    source = tmp_path / "multibody.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    # The grouping prim is an explicit Xform so it IS the placement root;
    # the typeless-ancestor fallback is covered by the scene-builder guard
    # test and _xformable_placement_root's walk-up.
    UsdGeom.Xform.Define(stage, "/Asset/SimBodies")
    for name, z_value in (("BodyA", 1.0), ("BodyB", -0.5)):
        body = UsdGeom.Xform.Define(stage, f"/Asset/SimBodies/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_value))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.0)
        collider = UsdGeom.Cube.Define(stage, f"/Asset/SimBodies/{name}/Collider")
        collider.CreateSizeAttr(0.2)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(
            True
        )
    stage.GetRootLayer().Save()

    body_paths = ["/Asset/SimBodies/BodyA", "/Asset/SimBodies/BodyB"]
    evidence, report_path = physics_workflow.validate_physics_runtime_multi_body(
        physics_usd=source,
        output_dir=tmp_path / "runtime",
        body_prim_paths=body_paths,
        engine="fake",
        duration_s=1.0,
        sample_fps=10,
        drop_height_m=0.05,
        acceptance={},
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "multi_body"
    # Visual review renders frames from the top-level recording; without it
    # the review would pass with zero rendered frames — and the recording
    # must animate EVERY tracked body, or the frames would show the other
    # bodies frozen mid-air.
    assert report["recording_usda"]
    recording_stage = Usd.Stage.Open(str(report["recording_usda"]))
    for body_path in body_paths:
        xformable = UsdGeom.Xformable(recording_stage.GetPrimAtPath(body_path))
        sampled = [
            op
            for op in xformable.GetOrderedXformOps()
            if op.GetOpName() == "xformOp:translate" and op.GetAttr().GetTimeSamples()
        ]
        assert sampled, f"{body_path} has no animated translate samples"
    assert report["placement_prim_path"] == "/Asset/SimBodies"
    assert report["failures"] == []
    assert [item["status"] for item in report["per_body_results"]] == [
        "pass",
        "pass",
    ]
    for item in report["per_body_results"]:
        assert Path(str(item["runtime_report"])).is_file()
        assert Path(str(item["trajectory_jsonl"])).is_file()
    rows = [
        json.loads(line)
        for line in Path(report["trajectory_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert {row["body_prim_path"] for row in rows} == set(body_paths)
    assert evidence.sim_ready_status == "pass"


def test_usd_cli_physics_render_rejects_non_ovrtx_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "run" / "runtime" / "recording.usda"
    recording.parent.mkdir(parents=True)
    recording.write_text("#usda 1.0\n", encoding="utf-8")
    raw_dir = tmp_path / "run" / "raw"
    raw_dir.mkdir()
    responses = iter(
        [
            {"ok": True},
            {
                "schema_version": "usd-cli.render-probe.v1",
                "ready": True,
                "engine": "ovrtx",
                "resolved_renderer": "remote",
                "transport": "remote",
            },
            {
                "ok": True,
                "summary": {"backend": "usdrecord"},
                "data": {"frame_paths": []},
            },
        ]
    )

    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "_run_json",
        lambda **_kwargs: next(responses),
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "_require_ovrtx_probe",
        lambda _payload, _project_dir: None,
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "_create_owned_session",
        lambda *, run_root, **_kwargs: SimpleNamespace(
            project_dir=run_root,
            session_id="physics-owned",
            close=lambda: None,
        ),
    )

    with pytest.raises(
        physics_workflow.usd_cli_ops.UsdCliPhysicsError,
        match="did not preserve its probed OVRTX-backed renderer identity",
    ):
        physics_workflow.usd_cli_ops.render_physics_frames(
            recording_usd=recording,
            output_dir=tmp_path / "run" / "runtime" / "frames",
            raw_dir=raw_dir,
        )


def test_usd_cli_physics_render_binds_verified_remote_frame_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    recording = tmp_path / "run" / "runtime" / "recording.usda"
    recording.parent.mkdir(parents=True)
    recording.write_text("#usda 1.0\n", encoding="utf-8")
    raw_dir = tmp_path / "run" / "raw"
    raw_dir.mkdir()
    frames_dir = tmp_path / "run" / "runtime" / "frames"
    endpoint = "https://ovrtx.example.test"

    def fake_run_json(**kwargs: object) -> dict[str, object]:
        arguments = kwargs["arguments"]
        assert isinstance(arguments, list)
        if arguments[0] == "open":
            return {"ok": True}
        if arguments[0] == "render-probe":
            return {
                "schema_version": "usd-cli.render-probe.v1",
                "ready": True,
                "engine": "ovrtx",
                "resolved_renderer": "remote",
                "transport": "remote",
                "backends": [
                    {
                        "url": f"{endpoint}/",
                        "engine": "ovrtx",
                        "protocol_version": 2,
                        "status": "ready",
                    }
                ],
            }
        frame = frames_dir / "frame.png"
        Image.new("RGB", (8, 8), color=(64, 96, 128)).save(frame)
        return {
            "ok": True,
            "summary": {"backend": "remote"},
            "data": {
                "frame_paths": [str(frame)],
                "renderer_identities": [
                    {
                        "endpoint": endpoint,
                        "engine": "ovrtx",
                        "protocol_version": 2,
                        "status": "ready",
                    }
                ],
            },
        }

    monkeypatch.setattr(physics_workflow.usd_cli_ops, "_run_json", fake_run_json)
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "_require_ovrtx_probe",
        lambda _payload, _project_dir: None,
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "_create_owned_session",
        lambda *, run_root, **_kwargs: SimpleNamespace(
            project_dir=run_root,
            session_id="physics-owned",
            close=lambda: None,
        ),
    )

    frames, record = physics_workflow.usd_cli_ops.render_physics_frames(
        recording_usd=recording,
        output_dir=frames_dir,
        raw_dir=raw_dir,
    )

    assert len(frames) == 1
    assert record["renderer"] == "ovrtx"
    assert record["resolved_renderer"] == "remote"
    assert record["transport"] == "remote"
    assert record["frame_renderer_identities"] == [
        {
            "endpoint": endpoint,
            "engine": "ovrtx",
            "protocol_version": 2,
            "status": "ready",
        }
    ]


def test_usd_cli_physics_render_synthesizes_local_frame_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    recording = tmp_path / "run" / "runtime" / "recording.usda"
    recording.parent.mkdir(parents=True)
    recording.write_text("#usda 1.0\n", encoding="utf-8")
    raw_dir = tmp_path / "run" / "raw"
    raw_dir.mkdir()
    frames_dir = tmp_path / "run" / "runtime" / "frames"

    def fake_run_json(**kwargs: object) -> dict[str, object]:
        arguments = kwargs["arguments"]
        assert isinstance(arguments, list)
        if arguments[0] == "open":
            return {"ok": True}
        if arguments[0] == "render-probe":
            return {
                "schema_version": "usd-cli.render-probe.v1",
                "ready": True,
                "engine": "ovrtx",
                "resolved_renderer": "ovrtx",
                "transport": "local",
            }
        frame = frames_dir / "frame.png"
        Image.new("RGB", (8, 8), color=(64, 96, 128)).save(frame)
        return {
            "ok": True,
            "summary": {"backend": "ovrtx"},
            "data": {
                "frame_paths": [str(frame)],
                "renderer_identities": [None],
            },
        }

    monkeypatch.setattr(physics_workflow.usd_cli_ops, "_run_json", fake_run_json)
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "_require_ovrtx_probe",
        lambda _payload, _project_dir: None,
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "_create_owned_session",
        lambda *, run_root, **_kwargs: SimpleNamespace(
            project_dir=run_root,
            session_id="physics-owned",
            close=lambda: None,
        ),
    )

    frames, record = physics_workflow.usd_cli_ops.render_physics_frames(
        recording_usd=recording,
        output_dir=frames_dir,
        raw_dir=raw_dir,
    )

    assert len(frames) == 1
    assert record["resolved_renderer"] == "ovrtx"
    assert record["transport"] == "local"
    assert record["frame_renderer_identities"] == [
        {
            "endpoint": "local",
            "engine": "ovrtx",
            "protocol_version": None,
            "status": "ready",
        }
    ]


def test_trajectory_response_omits_full_trajectory(tmp_path: Path) -> None:
    response_path = physics_workflow.scene_ops._write_trajectory_response(
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
                rigid_body_grouping="single_rigid_body_at_root",
                quality_warnings=[{"code": "mass_scale_suspicious"}],
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
        and record["classification"]["rigid_body_grouping"]
        == "single_rigid_body_at_root"
        and record["classification"]["quality_warnings"]
        == [{"code": "mass_scale_suspicious"}]
        for record in records
    )
    assert all(
        record["quality_warnings"] == [{"code": "mass_scale_suspicious"}]
        for record in records
    )


@pytest.mark.parametrize(
    "quality_warnings",
    [
        ["mass may be implausible"],
        [{"severity": "warning", "message": "missing code"}],
        [{"code": "mass_scale_suspicious", "severity": "info"}],
    ],
)
def test_component_decision_rejects_unevaluable_quality_warnings(
    quality_warnings: list[object],
) -> None:
    with pytest.raises(ValueError, match="quality_warnings"):
        PhysicsComponentDecision.model_validate(
            {
                "decision_id": "component_001",
                "component_id": "component_001",
                "body_root_path": "/World",
                "visual_evidence_paths": ["/World/Visual"],
                "collider_paths": ["/World/Visual"],
                "collision_mode": "author_on_targets",
                "mass_authoring_path": "/World",
                "inferred_material_family": "metal",
                "collision_approximation": "convexHull",
                "physical_properties": {"estimated_mass_kg": 1.0},
                "confidence": 0.8,
                "rationale": "Fixture.",
                "quality_warnings": quality_warnings,
            }
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


def test_run_physics_apply_workflow_keeps_policy_above_usd_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_inspect_components(
        _usd_path: Path, *, path_space: str = "source"
    ) -> dict[str, object]:
        calls.append("inspect")
        return _component_inspection(_usd_path, path_space=path_space)

    def fake_usd_cli_apply(**kwargs: object) -> dict[str, object]:
        calls.append("apply")
        return _fake_usd_cli_apply_success(**kwargs)

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
        physics_workflow.scene_ops,
        "inspect_components",
        fake_inspect_components,
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        fake_usd_cli_apply,
    )
    monkeypatch.setattr(
        physics_workflow.scene_ops,
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
    assert calls == ["inspect", "apply", "validate"]
    assert validate_kwargs
    evidence = json.loads(Path(result.validation_evidence_path or "").read_text())
    assert evidence["sim_ready_status"] == "pass"


def test_agentic_vomp_runs_after_schema_authoring_and_before_runtime_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_usd = tmp_path / "input.usda"
    input_usd.write_text("#usda 1.0\n", encoding="utf-8")
    runtime_root = tmp_path / "VoMP"
    runtime_root.mkdir()
    calls: list[str] = []

    def fake_inspect_components(
        usd_path: Path, *, path_space: str = "source"
    ) -> dict[str, object]:
        calls.append("inspect")
        return _component_inspection(usd_path, path_space=path_space)

    def fake_usd_cli_apply(**kwargs: object) -> dict[str, object]:
        calls.append("apply")
        schema_output = Path(str(kwargs["output_usd"]))
        assert schema_output.name == "physics_pre_vomp.usdc"
        schema_output.parent.mkdir(parents=True, exist_ok=True)
        schema_output.write_bytes(b"schema-authored")
        receipt = _fake_usd_cli_apply_success(**kwargs)
        schema_output.write_bytes(b"schema-authored")
        return receipt

    def fake_vomp_authoring(**kwargs: object) -> PhysicsVompMassResult:
        calls.append("vomp")
        source = Path(str(kwargs["input_usd_path"]))
        output = Path(str(kwargs["output_usd_path"]))
        assert source.name == "physics_pre_vomp.usdc"
        assert source.read_bytes() == b"schema-authored"
        assert output == (tmp_path / "out" / "physics.usdc").resolve()
        assert kwargs["target_prim_path"] == "/World"
        output.write_bytes(b"vomp-authored")
        provenance = tmp_path / "out" / "raw" / "physics_vomp_mass_properties.json"
        provenance.write_text("{}\n", encoding="utf-8")
        evidence_dir = tmp_path / "out" / "vomp" / "evidence"
        evidence_dir.mkdir(parents=True)
        manifest = evidence_dir / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        return PhysicsVompMassResult(
            target_prim_path="/World",
            input_usd_path=str(source),
            output_usd_path=str(output),
            output_usd_sha256="a" * 64,
            provenance_path=str(provenance),
            provenance_sha256="b" * 64,
            evidence_dir=str(evidence_dir),
            evidence_manifest_path=str(manifest),
            vomp_npz_path=str(evidence_dir / "vomp.npz"),
            worker_manifest_path=str(evidence_dir / "worker.json"),
            worker_log_path=str(evidence_dir / "worker.log"),
            sample_count=512,
            mass_kg=2.5,
            center_of_mass_local_m=(0.1, 0.2, 0.3),
            diagonal_inertia_kg_m2=(0.4, 0.5, 0.6),
            principal_axes_wxyz=(1.0, 0.0, 0.0, 0.0),
        )

    def fake_inspect_authored(physics_usd: Path) -> dict[str, object]:
        calls.append("inspect_vomp")
        assert physics_usd.read_bytes() == b"vomp-authored"
        return {
            "physics_usd": str(physics_usd),
            "default_prim": "/World",
            "physics_scene_paths": ["/World/PhysicsScene"],
            "rigid_body_paths": ["/World"],
            "enabled_rigid_body_paths": ["/World"],
            "disabled_rigid_body_paths": [],
            "collision_paths": ["/World/Cube"],
            "physics_material_paths": ["/World/Looks/PhysMat"],
            "rigid_body_count": 1,
            "enabled_rigid_body_count": 1,
            "disabled_rigid_body_count": 0,
            "collision_count": 1,
            "physics_material_count": 1,
        }

    def fake_validate_runtime(**kwargs: object) -> tuple[object, Path]:
        calls.append("validate")
        physics_usd = Path(str(kwargs["physics_usd"]))
        assert physics_usd == (tmp_path / "out" / "physics.usdc").resolve()
        assert physics_usd.read_bytes() == b"vomp-authored"
        assert kwargs["ground_clearance_support_cache"] is None
        report = tmp_path / "out" / "runtime" / "runtime_validation_report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text('{"engine":"fake"}\n', encoding="utf-8")
        return (
            physics_validation_evidence(
                asset=str(physics_usd),
                target_runtime="fake",
                physics_properties_status="pass",
                runtime_loadability_status="pass",
                no_explosions_status="pass",
            ),
            report,
        )

    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "inspect_components",
        fake_inspect_components,
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        fake_usd_cli_apply,
    )
    monkeypatch.setattr(
        physics_workflow,
        "run_agentic_vomp_mass_authoring",
        fake_vomp_authoring,
    )
    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "inspect_authored_physics",
        fake_inspect_authored,
    )
    monkeypatch.setattr(
        physics_workflow,
        "validate_physics_runtime",
        fake_validate_runtime,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=input_usd,
            output_dir=tmp_path / "out",
            simulation_engine="fake",
            vomp_mass=PhysicsVompMassConfig(runtime_root=runtime_root),
        )
    )

    assert result.success
    assert calls == ["inspect", "apply", "vomp", "inspect_vomp", "validate"]
    assert result.physics_usd_path == str((tmp_path / "out" / "physics.usdc").resolve())
    assert result.vomp_result_path == str(
        (tmp_path / "out" / "raw" / "physics_vomp_result.json").resolve()
    )
    assignments = json.loads(Path(result.assignments_path or "").read_text())
    assert assignments["vomp_mass"]["target_prim_path"] == "/World"
    assert assignments["apply_report"]["rigid_body_count"] == 1
    evidence = json.loads(Path(result.validation_evidence_path or "").read_text())
    assert evidence["metadata"]["mass_property_provider"] == "vomp"
    assert {item["kind"] for item in evidence["evidence_artifacts"]} >= {
        "vomp_mass_result",
        "vomp_mass_provenance",
        "vomp_evidence_manifest",
    }


def test_agentic_vomp_preserves_each_finalization_artifact_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "out"
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    source = raw_dir / "physics_pre_vomp.usda"
    source.write_text(
        '#usda 1.0\n(defaultPrim = "World")\ndef Xform "World" {}\n',
        encoding="utf-8",
    )
    output = output_dir / "physics.usda"

    def fake_vomp_authoring(**kwargs: object) -> PhysicsVompMassResult:
        namespace = str(kwargs["artifact_namespace"])
        input_path = Path(str(kwargs["input_usd_path"]))
        output_path = Path(str(kwargs["output_usd_path"]))
        assert input_path.parent == raw_dir / "vomp" / namespace
        output_path.write_text(
            '#usda 1.0\n(defaultPrim = "World")\n'
            f'def Xform "World"\n{{\n    custom string marker = "{namespace}"\n}}\n',
            encoding="utf-8",
        )
        provenance = raw_dir / "vomp" / namespace / "physics_vomp_mass_properties.json"
        provenance.write_text(
            json.dumps({"finalization": namespace}),
            encoding="utf-8",
        )
        evidence_dir = output_dir / "vomp" / namespace / "evidence"
        evidence_dir.mkdir(parents=True)
        manifest = evidence_dir / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        return PhysicsVompMassResult(
            target_prim_path="/World",
            input_usd_path=str(input_path),
            output_usd_path=str(output_path),
            output_usd_sha256=physics_workflow.file_sha256(output_path),
            provenance_path=str(provenance),
            provenance_sha256=physics_workflow.file_sha256(provenance),
            evidence_dir=str(evidence_dir),
            evidence_manifest_path=str(manifest),
            vomp_npz_path=str(evidence_dir / "vomp.npz"),
            worker_manifest_path=str(evidence_dir / "worker.json"),
            worker_log_path=str(evidence_dir / "worker.log"),
            sample_count=8,
            mass_kg=1.0,
            center_of_mass_local_m=(0.0, 0.0, 0.0),
            diagonal_inertia_kg_m2=(1.0, 1.0, 1.0),
            principal_axes_wxyz=(1.0, 0.0, 0.0, 0.0),
        )

    monkeypatch.setattr(
        physics_workflow,
        "run_agentic_vomp_mass_authoring",
        fake_vomp_authoring,
    )
    monkeypatch.setattr(
        physics_workflow,
        "verify_vomp_mass_properties",
        lambda *_args, **_kwargs: {"verified": True},
    )
    first_params = PhysicsApplyWorkflowInput(
        usd_path=source,
        output_dir=output_dir,
        vomp_mass=PhysicsVompMassConfig(runtime_root=tmp_path / "VoMP"),
        vomp_artifact_namespace="finalize-0001",
    )
    _, first_result, first_result_path = physics_workflow._apply_vomp_mass_if_requested(
        params=first_params,
        decisions=[{"mass_authoring_path": "/World"}],
        mobility_intent="dynamic",
        authored_usd=source,
        output_usd=output,
        output_dir=output_dir,
        raw_dir=raw_dir,
    )
    assert first_result is not None
    assert first_result_path is not None
    first_result_bytes = first_result_path.read_bytes()
    first_output = Path(first_result.output_usd_path)
    first_output_bytes = first_output.read_bytes()
    first_provenance = Path(first_result.provenance_path)
    first_provenance_bytes = first_provenance.read_bytes()

    second_params = first_params.model_copy(
        update={"vomp_artifact_namespace": "finalize-0002"}
    )
    _, second_result, second_result_path = (
        physics_workflow._apply_vomp_mass_if_requested(
            params=second_params,
            decisions=[{"mass_authoring_path": "/World"}],
            mobility_intent="dynamic",
            authored_usd=source,
            output_usd=output,
            output_dir=output_dir,
            raw_dir=raw_dir,
        )
    )

    assert second_result is not None
    assert second_result_path is not None
    assert first_result_path != second_result_path
    assert first_result_path.read_bytes() == first_result_bytes
    assert first_output.read_bytes() == first_output_bytes
    assert first_provenance.read_bytes() == first_provenance_bytes
    assert Path(second_result.output_usd_path).read_bytes() != first_output_bytes
    canonical = PhysicsVompMassResult.model_validate_json(
        (raw_dir / "physics_vomp_result.json").read_bytes()
    )
    assert canonical.output_usd_path == str(output.resolve())


def test_agentic_vomp_rejects_unsupported_output_before_schema_authoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_usd = tmp_path / "input.usda"
    input_usd.write_text("#usda 1.0\n", encoding="utf-8")
    apply_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "inspect_components",
        lambda usd_path: _component_inspection(usd_path),
    )

    def unexpected_apply(**kwargs: object) -> dict[str, object]:
        apply_calls.append(dict(kwargs))
        raise AssertionError("schema authoring must not run")

    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "apply_schema",
        unexpected_apply,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=input_usd,
            output_dir=tmp_path / "out",
            output_usd_path=tmp_path / "out" / "physics.usdz",
            simulation_engine="none",
            vomp_mass=PhysicsVompMassConfig(runtime_root=tmp_path / "VoMP"),
        )
    )

    assert result.success is False
    assert result.error == "Agentic VoMP output must be .usd, .usda, or .usdc"
    assert apply_calls == []


def test_agentic_vomp_rejects_original_input_as_output_before_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_usd = tmp_path / "input.usda"
    original_bytes = b"#usda 1.0\n"
    input_usd.write_bytes(original_bytes)
    inspect_calls: list[Path] = []

    def unexpected_inspect(usd_path: Path) -> dict[str, object]:
        inspect_calls.append(usd_path)
        raise AssertionError("inspection must not run for an in-place VoMP output")

    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "inspect_components",
        unexpected_inspect,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=input_usd,
            output_dir=tmp_path / "out",
            output_usd_path=input_usd,
            simulation_engine="none",
            vomp_mass=PhysicsVompMassConfig(runtime_root=tmp_path / "VoMP"),
        )
    )

    assert result.success is False
    assert result.error == (
        "VoMP canonical output USD must differ from the original input USD"
    )
    assert inspect_calls == []
    assert input_usd.read_bytes() == original_bytes


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

    def fake_inspect_components(
        _usd_path: Path, *, path_space: str = "source"
    ) -> dict[str, object]:
        return _component_inspection(_usd_path, path_space=path_space)

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
        physics_workflow.scene_ops,
        "apply_topology_plan",
        fake_apply_topology_plan,
    )
    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "inspect_components",
        fake_inspect_components,
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        lambda **kwargs: (
            apply_kwargs.append(dict(kwargs)) or _fake_usd_cli_apply_success(**kwargs)
        ),
    )
    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            run_simulation=False,
            simulation_engine="none",
            topology_plan_path=topology_plan,
        )
    )

    assert result.success
    assert apply_kwargs[0]["author_rigid_body"] is False
    assert result.validation_status == "conditional"


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
        physics_workflow.scene_ops,
        "apply_topology_plan",
        fake_apply_topology_plan,
    )
    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "inspect_components",
        _component_inspection,
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        lambda **kwargs: {
            **_fake_usd_cli_apply_success(**kwargs),
            "structural_validation": {
                "ok": True,
                "data": {
                    "checks": {
                        "scenes": 1,
                        "rigid_bodies": 1,
                        "enabled_rigid_bodies": 1,
                        "colliders": 1,
                    },
                    "issues": [],
                },
            },
        },
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


@pytest.mark.parametrize(
    "patch_source_digest",
    ["sha256:source", "sha256:prepared"],
)
def test_run_physics_apply_workflow_accepts_pre_or_post_topology_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_source_digest: str,
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
    patch_asset = (
        prepared_usd
        if patch_source_digest == "sha256:prepared"
        else tmp_path / "input.usda"
    )
    patch_path.write_text(
        json.dumps(
            {
                "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
                "asset": str(patch_asset),
                "source_digest": patch_source_digest,
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

    def fake_inspect_components(
        usd_path: Path, *, path_space: str = "source"
    ) -> dict[str, object]:
        result = _component_inspection(usd_path, path_space=path_space)
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
        physics_workflow.scene_ops,
        "apply_topology_plan",
        fake_apply_topology_plan,
    )
    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "inspect_components",
        fake_inspect_components,
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        _fake_usd_cli_apply_success,
    )
    output_dir = tmp_path / "out"
    if patch_source_digest == "sha256:prepared":
        cached_catalog = physics_workflow.add_physics_component_target_catalog(
            _component_inspection(prepared_usd)
        )
        cached_catalog["source_digest"] = "sha256:prepared"
        cached_catalog_path = output_dir / "raw" / "physics_components.json"
        cached_catalog_path.parent.mkdir(parents=True)
        cached_catalog_path.write_text(
            json.dumps(cached_catalog),
            encoding="utf-8",
        )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=output_dir,
            decision_patch_path=patch_path,
            topology_plan_path=topology_plan,
            run_simulation=False,
        )
    )

    assert result.success
    canonical_payload = json.loads(
        Path(result.decision_patch_path or "").read_text(encoding="utf-8")
    )
    assert canonical_payload["asset"] == str(patch_asset)
    assert canonical_payload["source_digest"] == patch_source_digest
    applied_payload = json.loads(
        (output_dir / "raw" / "physics_decision_patch_apply.json").read_text()
    )
    assert applied_payload["asset"] == str(prepared_usd.resolve())
    assert applied_payload["source_digest"] == "sha256:prepared"
    assert applied_payload["decisions"][0]["rigid_body_grouping"] is None
    assert applied_payload["decisions"][0]["quality_warnings"] == []


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
        physics_workflow.rebase_physics_v2_patch_to_components(
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
        physics_workflow.rebase_physics_v2_patch_to_components(
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
                "rigid_body_grouping": f"grouping note for {decision_id}",
                "quality_warnings": [{"code": f"warning_{decision_id}"}],
            }
            for decision_id, body_root, path, mass, confidence in [
                ("component_old_a", "/World/A", "/World/A", 1.25, 0.8),
                ("component_old_b", "/World/B", "/World/B", 2.75, 0.6),
            ]
        ],
    }

    rebased_payload, decisions, unresolved = (
        physics_workflow.rebase_physics_v2_patch_to_components(
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
    assert decision.rigid_body_grouping == (
        "Merged pre-topology grouping notes: "
        "grouping note for component_old_a | grouping note for component_old_b"
    )
    assert decision.quality_warnings == [
        {"code": "warning_component_old_a"},
        {"code": "warning_component_old_b"},
    ]
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
        physics_workflow.rebase_physics_v2_patch_to_components(
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
        physics_workflow.scene_ops,
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


def test_run_physics_apply_workflow_refuses_vomp_without_accepted_decision(
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
        physics_workflow.scene_ops,
        "inspect_components",
        _component_inspection,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=input_usd,
            output_dir=tmp_path / "out",
            decision_patch_path=patch_path,
            run_simulation=False,
            vomp_mass=PhysicsVompMassConfig(runtime_root=tmp_path / "VoMP"),
        )
    )

    assert result.success is False
    assert result.error == (
        "VoMP mass authoring requires at least one accepted body decision."
    )
    assert result.vomp_result_path is None


def test_vomp_target_selection_excludes_unowned_static_decisions() -> None:
    component = physics_workflow.PhysicsComponent.model_validate(
        {
            "component_id": "component_static",
            "component_role": "unowned_static",
            "body_root_path": "/World/Fixture",
            "visual_evidence_paths": ["/World/Fixture"],
            "collider_paths": ["/World/Fixture"],
            "helper_paths": [],
            "rigid_body_paths": [],
            "joint_paths": [],
            "material_evidence": [],
            "bounds_m": {},
            "topology_findings": [],
        }
    )
    decision = physics_workflow.infer_component_decisions([component])[0]

    assert physics_workflow._accepted_vomp_body_decisions([decision], [component]) == []


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
        physics_workflow.scene_ops,
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

    def fake_inspect_components(
        usd_path: Path, *, path_space: str = "source"
    ) -> dict[str, object]:
        result = _component_inspection(usd_path, path_space=path_space)
        result["source_digest"] = (
            "sha256:prepared" if Path(usd_path) == prepared_usd else "sha256:source"
        )
        return result

    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "apply_topology_plan",
        fake_apply_topology_plan,
    )
    monkeypatch.setattr(
        physics_workflow.scene_ops,
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

    def fake_inspect_components(
        usd_path: Path, *, path_space: str = "source"
    ) -> dict[str, object]:
        result = _component_inspection(usd_path, path_space=path_space)
        result["source_digest"] = "sha256:source"
        return result

    monkeypatch.setattr(
        physics_workflow.scene_ops,
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
    assert "Legacy V1 physics decision patches" in result.error


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

    def fake_inspect_components(
        _usd_path: Path, *, path_space: str = "source"
    ) -> dict[str, object]:
        return _component_inspection(_usd_path, path_space=path_space)

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
        physics_workflow.scene_ops,
        "inspect_components",
        fake_inspect_components,
    )
    monkeypatch.setattr(
        physics_workflow.scene_ops,
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
    applied_payload = json.loads(
        (tmp_path / "out" / "raw" / "physics_decision_patch_apply.json").read_text()
    )
    assert applied_payload["asset"].endswith("input.usda")
    assert applied_payload["source_digest"] == "sha256:test"


def test_fail_on_validation_error_preserves_artifact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_inspect_components(
        _usd_path: Path, *, path_space: str = "source"
    ) -> dict[str, object]:
        return _component_inspection(_usd_path, path_space=path_space)

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
        physics_workflow.scene_ops,
        "inspect_components",
        fake_inspect_components,
    )
    monkeypatch.setattr(
        physics_workflow.usd_cli_ops,
        "apply_physics_patch",
        lambda **kwargs: {
            **_fake_usd_cli_apply_success(**kwargs),
            "structural_validation": {
                "ok": False,
                "data": {
                    "checks": {
                        "scenes": 0,
                        "rigid_bodies": 0,
                        "enabled_rigid_bodies": 0,
                        "colliders": 0,
                    },
                    "issues": ["no authored physics"],
                },
            },
        },
    )
    monkeypatch.setattr(
        physics_workflow.scene_ops,
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


def test_load_behavior_assessment_coerces_llm_shapes(tmp_path: Path) -> None:
    """The visual-review child emits richer shapes than the strict schema.

    Regression for the tune-gating bug: runtime_report/rendered_frames arriving as
    objects and assessment_notes as a list raised a ValidationError that aborted
    finalization and skipped the downstream tune phase. The loader must coerce
    these instead of raising.
    """

    payload = {
        "schema_version": physics_workflow.PHYSICS_BEHAVIOR_ASSESSMENT_SCHEMA_VERSION,
        "status": "fixed",
        "checked_views": ["/tmp/frame_0000.png"],
        "runtime_report": {
            "path": "/tmp/runtime_validation_report.json",
            "engine": "ovphysx",
            "validation_status": "conditional",
        },
        "rendered_frames": {"count": 31, "range": "frame_0000-frame_0030"},
        "issues_found": [{"id": "settle_timeout", "severity": "minor"}],
        "issues_fixed": [],
        "unresolved_issues": [{"id": "settle_timeout"}],
        "assessment_notes": ["Falls under gravity.", "Barely bounces."],
        "some_unknown_key": "should be dropped",
    }
    assessment_path = tmp_path / "physics_behavior_assessment.json"
    assessment_path.write_text(json.dumps(payload), encoding="utf-8")

    assessment = physics_workflow.load_physics_behavior_assessment(assessment_path)

    assert assessment.status == "fixed"
    assert assessment.runtime_report == "/tmp/runtime_validation_report.json"
    assert assessment.rendered_frames == []
    assert assessment.assessment_notes == "Falls under gravity.\nBarely bounces."
    assert len(assessment.unresolved_issues) == 1


def test_load_behavior_assessment_coerces_object_view_lists(tmp_path: Path) -> None:
    """checked_views / rendered_frames arriving as a LIST OF OBJECTS must coerce.

    Regression for a batch failure: the review LLM emitted
    ``checked_views: [{"frame_range": ..., "note": ...}]`` (list of dicts), which
    the ``list[str]`` field rejected with a ValidationError — aborting finalize.
    Each object element is stringified rather than raising.
    """

    payload = {
        "status": "pass",
        "checked_views": [
            {"frame_range": "0000-0010", "note": "body integrity intact"},
            {"frame_range": "0011-0030"},
        ],
        "rendered_frames": [
            {"path": "/tmp/f0.png"},
            {"count": 2, "note": "summary only"},
            "/tmp/f1.png",
        ],
        "assessment_notes": "ok",
    }
    assessment_path = tmp_path / "physics_behavior_assessment.json"
    assessment_path.write_text(json.dumps(payload), encoding="utf-8")

    assessment = physics_workflow.load_physics_behavior_assessment(assessment_path)

    assert assessment.status == "pass"
    assert len(assessment.checked_views) == 2
    assert all(isinstance(v, str) for v in assessment.checked_views)
    assert "frame_range" in assessment.checked_views[0]
    assert assessment.rendered_frames == ["/tmp/f0.png", "/tmp/f1.png"]


def test_load_behavior_assessment_extracts_paths_from_captured_review_shape(
    tmp_path: Path,
) -> None:
    payload = {
        "status": "fixed",
        "checked_views": [
            {
                "view": "reference_three_quarter",
                "path": "/tmp/reference.png",
                "finding": "One continuous mug body.",
            },
            {
                "view": "runtime_frame_sequence_0000_0030",
                "finding": "The body leaves the frame.",
            },
        ],
        "runtime_report": {
            "path": "/tmp/runtime_validation_report.json",
            "engine": "ovphysx",
            "status": "fail",
        },
        "rendered_frames": ["/tmp/frame_0000.png"],
        "assessment_notes": "Runtime metrics remain authoritative.",
    }
    assessment_path = tmp_path / "physics_behavior_assessment.json"
    assessment_path.write_text(json.dumps(payload), encoding="utf-8")

    assessment = physics_workflow.load_physics_behavior_assessment(assessment_path)

    assert "/tmp/reference.png" in assessment.checked_views[0]
    assert "One continuous mug body." in assessment.checked_views[0]
    assert "runtime_frame_sequence_0000_0030" in assessment.checked_views[1]
    assert assessment.runtime_report == "/tmp/runtime_validation_report.json"


def test_component_target_catalog_assigns_stable_role_ids() -> None:
    inspection = _component_inspection(Path("/tmp/asset.usda"))

    result = physics_workflow.add_physics_component_target_catalog(inspection)

    assert "authoring_targets" not in inspection["components"][0]
    assert result["components"][0]["authoring_targets"] == [
        {
            "target_id": _authoring_target_id("component_001", "visual", "/World/Cube"),
            "role": "visual",
            "prim_path": "/World/Cube",
        }
    ]


def test_component_target_catalog_accepts_documented_additive_fields() -> None:
    inspection = _component_inspection(Path("/tmp/asset.usda"))
    inspection["schema_version"] = "content-agent-workflows.physics-components.v2"
    component = inspection["components"][0]
    assert isinstance(component, dict)
    component["authoring_targets"] = [{"target_id": "stale"}]
    component["source_path_expansions"] = {"/World/Cube": ["/Source/Cube"]}

    result = physics_workflow.add_physics_component_target_catalog(
        inspection,
        source_path_expansions={"/World/Cube": ["/Source/Cube"]},
    )

    assert result["components"][0]["source_path_expansions"] == {
        "/World/Cube": ["/Source/Cube"]
    }
    assert result["components"][0]["authoring_targets"][0]["target_id"] == (
        _authoring_target_id("component_001", "visual", "/World/Cube")
    )


def test_component_target_catalog_rejects_unsupported_additive_fields() -> None:
    inspection = _component_inspection(Path("/tmp/asset.usda"))
    component = inspection["components"][0]
    assert isinstance(component, dict)
    component["future_field"] = True

    with pytest.raises(RuntimeError, match="unsupported fields"):
        physics_workflow.add_physics_component_target_catalog(inspection)


def test_raw_v2_coverage_rejects_missing_component_id() -> None:
    component = physics_workflow.PhysicsComponent.model_validate(
        _component_inspection(Path("/tmp/asset.usda"))["components"][0]
    )

    with pytest.raises(RuntimeError, match="requires a non-empty component_id"):
        physics_workflow._validate_raw_v2_component_coverage(
            {"decisions": [{"decision_id": "missing-component"}]},
            [component],
        )


def test_inspection_assignment_records_unmapped_runtime_paths() -> None:
    decision = physics_workflow.PhysicsComponentDecision(
        decision_id="component_001",
        component_id="component_001",
        body_root_path="/World",
        visual_evidence_paths=["/World/Cube"],
        collider_paths=["/World/Cube"],
        collision_mode="author_on_targets",
        mass_authoring_path="/World",
        inferred_material_family="generic",
        collision_approximation="convexHull",
        physical_properties={"density": 1000.0, "estimated_mass_kg": 1.0},
        confidence=0.8,
        rationale="Fixture.",
    )

    payload = physics_workflow.physics_decision_assignment_payload(
        decision,
        path_space="inspection",
        source_path_expansions={"/World": ["/Source"]},
    )

    assert payload["source_prim_paths"] == ["/Source"]
    assert payload["unmapped_runtime_prim_paths"] == ["/World/Cube"]


def test_workflow_rejects_changed_inspection_digest(tmp_path: Path) -> None:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=source,
            inspection_asset_sha256="0" * 64,
            output_dir=tmp_path / "out",
            run_simulation=False,
        )
    )

    assert not result.success
    assert result.error is not None
    assert "Inspection USD changed after physics preflight" in result.error


def test_component_target_ids_do_not_shift_when_path_sets_change() -> None:
    original = physics_workflow.PhysicsComponent(
        component_id="component_001",
        body_root_path="/World",
        visual_evidence_paths=["/World/B", "/World/C"],
    )
    expanded = original.model_copy(
        update={"visual_evidence_paths": ["/World/A", "/World/B", "/World/C"]}
    )

    original_targets = {
        target["prim_path"]: target["target_id"]
        for target in physics_workflow._component_authoring_targets(original)
    }
    expanded_targets = {
        target["prim_path"]: target["target_id"]
        for target in physics_workflow._component_authoring_targets(expanded)
    }

    assert original_targets["/World/B"] == expanded_targets["/World/B"]
    assert original_targets["/World/C"] == expanded_targets["/World/C"]


def test_v2_patch_resolves_target_ids_and_documented_metadata() -> None:
    component = physics_workflow.PhysicsComponent.model_validate(
        _component_inspection(Path("/tmp/asset.usda"))["components"][0]
    )
    payload = {
        "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
        "asset": "/tmp/asset.usda",
        "source_digest": "sha256:test",
        "decisions": [
            {
                "decision_id": "component_001",
                "component_id": "component_001",
                "collider_target_ids": [
                    _authoring_target_id("component_001", "visual", "/World/Cube")
                ],
                "collision_mode": "author_on_targets",
                "inferred_material_family": "ceramic",
                "inferred_material_name": None,
                "collision_approximation": "convexDecomposition",
                "physical_properties": {
                    "density": 2400.0,
                    "estimated_mass_kg": 0.3,
                },
                "rigid_body_grouping": "single_rigid_body_at_root",
                "quality_warnings": [
                    {"code": "mass_scale_suspicious", "severity": "warning"}
                ],
                "confidence": 0.8,
                "rationale": "The mug is one rigid ceramic object.",
            }
        ],
    }

    decisions, unresolved = (
        physics_workflow._validate_v2_patch_payload_against_components(
            payload,
            components=[component],
            source_digest="sha256:test",
        )
    )

    assert unresolved == []
    assert decisions[0].collider_paths == ["/World/Cube"]
    assert decisions[0].body_root_path == "/World"
    assert decisions[0].mass_authoring_path == "/World"
    assert decisions[0].rigid_body_grouping == "single_rigid_body_at_root"
    assert decisions[0].quality_warnings[0]["code"] == "mass_scale_suspicious"


def test_v2_patch_rejects_unknown_target_id() -> None:
    component = physics_workflow.PhysicsComponent.model_validate(
        _component_inspection(Path("/tmp/asset.usda"))["components"][0]
    )
    payload = {
        "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
        "source_digest": "sha256:test",
        "decisions": [
            {
                "decision_id": "component_001",
                "component_id": "component_001",
                "collider_target_ids": ["component_001:visual:missing"],
                "collision_mode": "author_on_targets",
                "inferred_material_family": "ceramic",
                "collision_approximation": "convexHull",
                "physical_properties": {"density": 2400.0},
                "confidence": 0.8,
                "rationale": "Fixture.",
            }
        ],
    }

    with pytest.raises(RuntimeError, match="unknown collider target IDs"):
        physics_workflow._validate_v2_patch_payload_against_components(
            payload,
            components=[component],
            source_digest="sha256:test",
        )


def test_v2_patch_rejects_role_incompatible_target_id() -> None:
    component = physics_workflow.PhysicsComponent(
        component_id="component_001",
        body_root_path="/World",
        visual_evidence_paths=["/World/Visual"],
        collider_paths=["/World/Collider"],
    )
    payload = {
        "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
        "source_digest": "sha256:test",
        "decisions": [
            {
                "decision_id": "component_001",
                "component_id": "component_001",
                "collider_target_ids": [
                    _authoring_target_id("component_001", "collider", "/World/Collider")
                ],
                "collision_mode": "author_on_targets",
                "inferred_material_family": "metal",
                "collision_approximation": "convexHull",
                "physical_properties": {"density": 2700.0},
                "confidence": 0.8,
                "rationale": "Fixture.",
            }
        ],
    }

    with pytest.raises(RuntimeError, match="the wrong role"):
        physics_workflow._validate_v2_patch_payload_against_components(
            payload,
            components=[component],
            source_digest="sha256:test",
        )


def test_v2_patch_rejects_mixed_target_and_resolved_decisions_before_binding() -> None:
    components = [
        physics_workflow.PhysicsComponent(
            component_id="component_001",
            body_root_path="/World/A",
            visual_evidence_paths=["/World/A/Mesh"],
        ),
        physics_workflow.PhysicsComponent(
            component_id="component_002",
            body_root_path="/World/B",
            visual_evidence_paths=["/World/B/Mesh"],
        ),
    ]
    payload = {
        "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
        "source_digest": "sha256:test",
        "decisions": [
            {
                "decision_id": "component_001",
                "component_id": "component_001",
                "collider_target_ids": [
                    _authoring_target_id("component_001", "visual", "/World/A/Mesh")
                ],
                "collision_mode": "author_on_targets",
                "inferred_material_family": "metal",
                "collision_approximation": "convexHull",
                "physical_properties": {"density": 2700.0},
                "confidence": 0.8,
                "rationale": "Use the inspected target.",
            },
            {
                "decision_id": "component_002",
                "component_id": "component_002",
                "body_root_path": "/World/B",
                "visual_evidence_paths": ["/World/B/Mesh"],
                "collider_paths": ["/World/B/Mesh"],
                "collision_mode": "author_on_targets",
                "mass_authoring_path": "/World/B",
                "inferred_material_family": "metal",
                "collision_approximation": "convexHull",
                "physical_properties": {"density": 2700.0},
                "confidence": 0.8,
                "rationale": "Legacy resolved decision.",
            },
        ],
    }
    original = json.loads(json.dumps(payload))

    with pytest.raises(RuntimeError, match="cannot mix target-ID and resolved-path"):
        physics_workflow._validate_v2_patch_payload_against_components(
            payload,
            components=components,
            source_digest="sha256:test",
        )

    assert payload == original


def test_v2_patch_reports_unknown_component_before_target_binding() -> None:
    component = physics_workflow.PhysicsComponent.model_validate(
        _component_inspection(Path("/tmp/asset.usda"))["components"][0]
    )
    payload = {
        "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
        "source_digest": "sha256:test",
        "decisions": [
            {
                "decision_id": "component_missing",
                "component_id": "component_missing",
                "collider_target_ids": ["component_missing:visual:missing"],
                "collision_mode": "author_on_targets",
                "inferred_material_family": "generic",
                "collision_approximation": "convexHull",
                "physical_properties": {"density": 1000.0},
                "confidence": 0.5,
                "rationale": "Fixture.",
            }
        ],
    }

    with pytest.raises(
        RuntimeError,
        match=r"coverage mismatch: missing=\['component_001'\], unknown=\['component_missing'\]",
    ):
        physics_workflow._validate_v2_patch_payload_against_components(
            payload,
            components=[component],
            source_digest="sha256:test",
        )


_PLACEHOLDER_PROPERTIES = {
    "density": 0.0,
    "estimated_mass_kg": 0.0,
    "static_friction": 0.0,
    "dynamic_friction": 0.0,
    "restitution": 0.0,
}

_TARGET_DECISION_KWARGS: dict[str, object] = {
    "decision_id": "component_001",
    "component_id": "component_001",
    "collider_target_ids": ["component_001:visual:stable"],
    "collision_mode": "author_on_targets",
    "inferred_material_family": "plastic",
    "collision_approximation": "convexHull",
    "confidence": 0.8,
    "rationale": "Copied the schema example.",
}

_RESOLVED_DECISION_KWARGS: dict[str, object] = {
    "decision_id": "component_001",
    "component_id": "component_001",
    "body_root_path": "/World",
    "visual_evidence_paths": ["/World/Visual"],
    "collider_paths": ["/World/Visual"],
    "collision_mode": "author_on_targets",
    "mass_authoring_path": "/World",
    "inferred_material_family": "plastic",
    "collision_approximation": "convexHull",
    "confidence": 0.8,
    "rationale": "Copied the schema example.",
}


def test_both_decision_models_reject_copied_placeholder_properties() -> None:
    """The prompt's patch-schema example fixes JSON types with 0.0
    placeholders. Copied verbatim they are type-valid, but the apply path
    would skip the nonpositive density and mass while authoring a
    frictionless, rebound-free material -- so validation must fail the
    all-placeholder vector loudly on BOTH V2 decision models. The CLI apply
    path parses raw patch JSON through ``PhysicsComponentDecision``
    (``_v2_decisions_from_payload``), never the target-ID model, so a guard
    on ``PhysicsComponentTargetDecision`` alone cannot protect it."""

    with pytest.raises(ValidationError, match="copied schema placeholder"):
        PhysicsComponentTargetDecision(
            **_TARGET_DECISION_KWARGS,
            physical_properties=dict(_PLACEHOLDER_PROPERTIES),
        )

    with pytest.raises(ValidationError, match="copied schema placeholder"):
        PhysicsComponentDecision(
            **_RESOLVED_DECISION_KWARGS,
            physical_properties=dict(_PLACEHOLDER_PROPERTIES),
        )

    # The exact CLI parse site: raw v2 payload decisions -> resolved model.
    with pytest.raises(ValidationError, match="copied schema placeholder"):
        physics_workflow._v2_decisions_from_payload(
            {
                "decisions": [
                    {
                        **_RESOLVED_DECISION_KWARGS,
                        "physical_properties": dict(_PLACEHOLDER_PROPERTIES),
                    }
                ]
            }
        )

    # A nonzero auxiliary key beside the copied zeros must not smuggle the
    # placeholder past the guard: downstream authoring ignores auxiliary keys
    # and would still act on the zeroed physical values.
    with pytest.raises(ValidationError, match="copied schema placeholder"):
        PhysicsComponentDecision(
            **_RESOLVED_DECISION_KWARGS,
            physical_properties={**_PLACEHOLDER_PROPERTIES, "fill_fraction": 0.4},
        )


def test_decision_placeholder_guard_spares_legitimate_zeros() -> None:
    """A genuine zero among real values (an inelastic part's restitution) and
    an unowned_static fixture's zero density/mass must still validate."""

    decision = PhysicsComponentTargetDecision(
        **_TARGET_DECISION_KWARGS,
        physical_properties={
            "density": 150.0,
            "estimated_mass_kg": 0.2,
            "static_friction": 0.8,
            "dynamic_friction": 0.65,
            "restitution": 0.0,
        },
    )
    assert decision.physical_properties["density"] == 150.0

    fixture = PhysicsComponentDecision(
        **{**_RESOLVED_DECISION_KWARGS, "collision_mode": "preserve_existing"},
        physical_properties={"density": 0.0, "estimated_mass_kg": 0.0},
    )
    assert fixture.physical_properties == {"density": 0.0, "estimated_mass_kg": 0.0}


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_both_decision_models_reject_non_finite_properties(bad_value: float) -> None:
    """NaN and infinities parse as floats but poison every downstream mass and
    material computation; both decision models must reject them."""

    properties = {
        "density": 1050.0,
        "estimated_mass_kg": bad_value,
        "static_friction": 0.6,
        "dynamic_friction": 0.45,
        "restitution": 0.55,
    }

    with pytest.raises(ValidationError, match="finite"):
        PhysicsComponentTargetDecision(
            **_TARGET_DECISION_KWARGS,
            physical_properties=dict(properties),
        )

    with pytest.raises(ValidationError, match="finite"):
        PhysicsComponentDecision(
            **_RESOLVED_DECISION_KWARGS,
            physical_properties=dict(properties),
        )


def test_load_physics_decision_patch_accepts_target_id_schema(tmp_path: Path) -> None:
    patch_path = tmp_path / "decision.json"
    patch_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION
                ),
                "decisions": [
                    {
                        "decision_id": "component_001",
                        "component_id": "component_001",
                        "collider_target_ids": ["component_001:visual:stable"],
                        "collision_mode": "author_on_targets",
                        "inferred_material_family": "generic",
                        "collision_approximation": "convexHull",
                        "physical_properties": {"density": 1000.0},
                        "confidence": 0.8,
                        "rationale": "Use the inspected target ID.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    decisions = physics_workflow.load_physics_decision_patch(patch_path)

    assert isinstance(decisions[0], PhysicsComponentTargetDecision)
    assert decisions[0].collider_target_ids == ["component_001:visual:stable"]


def test_workflow_preserves_target_id_patch_and_writes_resolved_apply_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "asset.usda"
    original_source = tmp_path / "asset.original.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    original_source.write_text("#usda 1.0\n# original\n", encoding="utf-8")
    patch_path = tmp_path / "decision.json"
    patch_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION
                ),
                "asset": str(source),
                "source_digest": "sha256:test",
                "decisions": [
                    {
                        "decision_id": "component_001",
                        "component_id": "component_001",
                        "collider_target_ids": [
                            _authoring_target_id(
                                "component_001", "visual", "/World/Cube"
                            )
                        ],
                        "collision_mode": "author_on_targets",
                        "inferred_material_family": "generic",
                        "inferred_material_name": None,
                        "collision_approximation": "convexHull",
                        "physical_properties": {
                            "density": 1000.0,
                            "estimated_mass_kg": 1.0,
                        },
                        "confidence": 0.8,
                        "rationale": "Use the inspected visual target.",
                    }
                ],
                "unresolved_components": [],
            }
        ),
        encoding="utf-8",
    )
    applied_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "inspect_components",
        lambda usd_path, *, path_space="source": _component_inspection(
            usd_path, path_space=path_space
        ),
    )

    def fake_apply_schema(**kwargs: object) -> dict[str, object]:
        applied_payloads.append(
            json.loads(Path(str(kwargs["decision_patch_path"])).read_text())
        )
        physics_usd = tmp_path / "out" / "physics.usdc"
        physics_usd.write_bytes(b"PXR-USDC")
        return {
            "physics_usd": str(physics_usd),
            "physics_scene_paths": ["/World/PhysicsScene"],
            "rigid_body_paths": ["/World"],
            "collision_paths": ["/World/Cube"],
            "rigid_body_count": 1,
            "collision_count": 1,
            "physics_material_count": 1,
        }

    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "apply_schema",
        fake_apply_schema,
    )

    result = run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=source,
            source_usd_path=original_source,
            source_asset_sha256=hashlib.sha256(
                original_source.read_bytes()
            ).hexdigest(),
            path_space="inspection",
            source_path_expansions={
                "/World": ["/Source"],
                "/World/Cube": ["/Source/CubeA", "/Source/CubeB"],
            },
            output_dir=tmp_path / "out",
            decision_patch_path=patch_path,
            run_simulation=False,
        )
    )

    assert result.success
    canonical = json.loads(Path(result.decision_patch_path or "").read_text())
    assert canonical["decisions"][0]["collider_target_ids"] == [
        _authoring_target_id("component_001", "visual", "/World/Cube")
    ]
    assert "collider_paths" not in canonical["decisions"][0]
    applied_payload = json.loads(
        (tmp_path / "out" / "raw" / "physics_decision_patch_apply.json").read_text()
    )
    assert applied_payload["decisions"][0]["collider_paths"] == ["/World/Cube"]
    assert "collider_target_ids" not in applied_payload["decisions"][0]
    assert applied_payload["decisions"][0]["rigid_body_grouping"] is None
    assert applied_payload["decisions"][0]["quality_warnings"] == []
    assignments = json.loads(Path(result.assignments_path or "").read_text())
    assert assignments["asset"] == str(original_source)
    assert assignments["prepared_asset"] == str(source)
    assert assignments["path_space"] == "inspection"
    assert (
        assignments["source_asset_sha256"]
        == hashlib.sha256(original_source.read_bytes()).hexdigest()
    )
    assert assignments["decisions"][0]["runtime_prim_paths"] == [
        "/World",
        "/World/Cube",
    ]
    assert assignments["decisions"][0]["source_prim_paths"] == [
        "/Source",
        "/Source/CubeA",
        "/Source/CubeB",
    ]


def test_v2_patch_rejects_captured_full_target_path_transcription() -> None:
    real_paths = [
        "/box/sm_box_corrugated_b04_obj_00/mesh_00/Mesh",
        "/box/sm_box_corrugated_b04_obj_00/mesh_01/Mesh",
    ]
    invented_paths = [
        path.replace("corrugated_b04", "corrugated_brown_b04") for path in real_paths
    ]
    component = physics_workflow.PhysicsComponent(
        component_id="component_001",
        body_root_path="/box",
        visual_evidence_paths=real_paths,
    )
    payload = {
        "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
        "asset": "/tmp/box.usda",
        "source_digest": "sha256:box",
        "decisions": [
            {
                "decision_id": "component_001-box",
                "component_id": "component_001",
                "body_root_path": "/box",
                "visual_evidence_paths": invented_paths,
                "collider_paths": invented_paths,
                "collision_mode": "author_on_targets",
                "mass_authoring_path": "/box",
                "inferred_material_family": "cardboard",
                "inferred_material_name": None,
                "collision_approximation": "convexHull",
                "physical_properties": {
                    "density": 700.0,
                    "estimated_mass_kg": 0.5,
                },
                "confidence": 0.8,
                "rationale": "Use every visible box mesh.",
            }
        ],
    }

    with pytest.raises(RuntimeError, match="not present in the inspected component"):
        physics_workflow._validate_v2_patch_payload_against_components(
            payload,
            components=[component],
            source_digest="sha256:box",
        )


def test_v2_patch_rejects_equal_cardinality_mixed_target_selection() -> None:
    component = physics_workflow.PhysicsComponent(
        component_id="component_001",
        body_root_path="/box",
        visual_evidence_paths=["/box/mesh_00", "/box/mesh_01"],
        helper_paths=["/box/helper/Guide"],
    )
    payload = {
        "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
        "asset": "/tmp/box.usda",
        "source_digest": "sha256:box",
        "decisions": [
            {
                "decision_id": "component_001-box",
                "component_id": "component_001",
                "body_root_path": "/box",
                "visual_evidence_paths": [
                    "/box/mesh_00",
                    "/box/helper/Guide",
                ],
                "collider_paths": ["/box/mesh_00", "/box/helper/Guide"],
                "collision_mode": "author_on_targets",
                "mass_authoring_path": "/box",
                "inferred_material_family": "cardboard",
                "collision_approximation": "convexHull",
                "physical_properties": {"density": 700.0},
                "confidence": 0.8,
                "rationale": "Fixture.",
            }
        ],
    }

    with pytest.raises(RuntimeError, match="targets helper geometry"):
        physics_workflow._validate_v2_patch_payload_against_components(
            payload,
            components=[component],
            source_digest="sha256:box",
        )


def test_v2_patch_reports_unknown_partial_target_paths() -> None:
    component = physics_workflow.PhysicsComponent(
        component_id="component_001",
        body_root_path="/box",
        visual_evidence_paths=["/box/mesh_00", "/box/mesh_01"],
    )
    payload = {
        "schema_version": physics_workflow.PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
        "asset": "/tmp/box.usda",
        "source_digest": "sha256:box",
        "decisions": [
            {
                "decision_id": "component_001-box",
                "component_id": "component_001",
                "body_root_path": "/box",
                "visual_evidence_paths": ["/box/invented"],
                "collider_paths": ["/box/invented"],
                "collision_mode": "author_on_targets",
                "mass_authoring_path": "/box",
                "inferred_material_family": "cardboard",
                "collision_approximation": "convexHull",
                "physical_properties": {"density": 700.0},
                "confidence": 0.8,
                "rationale": "Fixture.",
            }
        ],
    }

    with pytest.raises(RuntimeError, match="not present in the inspected component"):
        physics_workflow._validate_v2_patch_payload_against_components(
            payload,
            components=[component],
            source_digest="sha256:box",
        )


def test_snap_patch_body_roots_to_inspected_component() -> None:
    """A decision that nominates a path inside the component (the mesh) instead of
    the inspected body root is snapped, not rejected.

    Regression for a batch failure: the child agent authored
    ``body_root_path=/World/Object`` while the inspected component's body root was
    ``/World``, which raised "changes body_root_path" and aborted the run. The
    inspected topology is authoritative, so the in-subtree path is snapped; a
    genuinely foreign path is left for the validator to reject.
    """
    from content_agent_workflows.physics.workflow import (
        _snap_patch_body_roots_in_place,
    )

    component = physics_workflow.PhysicsComponent(
        component_id="component_001",
        body_root_path="/World",
        visual_evidence_paths=["/World/Object"],
    )

    # in-subtree path (the mesh) -> snapped to the component body root
    inside = {
        "decisions": [
            {
                "component_id": "component_001",
                "body_root_path": "/World/Object",
                "mass_authoring_path": "/World/Object",
            }
        ]
    }
    _snap_patch_body_roots_in_place(inside, [component])
    assert inside["decisions"][0]["body_root_path"] == "/World"
    assert inside["decisions"][0]["mass_authoring_path"] == "/World"

    # already-correct -> untouched; foreign path -> untouched (validator rejects)
    for path in ("/World", "/Foreign"):
        payload = {
            "decisions": [{"component_id": "component_001", "body_root_path": path}]
        }
        _snap_patch_body_roots_in_place(payload, [component])
        assert payload["decisions"][0]["body_root_path"] == path


def test_apply_workflow_forwards_the_penetration_override_to_the_runtime_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without this the apply-phase gate always enforces the runtime default
    (scale-relative on exact collider geometry, 0.005 m on the bbox fallback),
    so a run configured for a deeper legitimate rest records a hard
    ground-clearance failure that the refinement child is simultaneously
    instructed not to fix."""

    def fake_apply_schema(**kwargs: object) -> dict[str, object]:
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

    validate_kwargs: list[dict[str, object]] = []

    def fake_validate_runtime(**kwargs: object) -> dict[str, object]:
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
            "evidence_artifacts": [],
        }

    monkeypatch.setattr(
        physics_workflow.scene_ops,
        "inspect_components",
        lambda _usd_path, *, path_space="source": _component_inspection(
            _usd_path, path_space=path_space
        ),
    )
    monkeypatch.setattr(physics_workflow.scene_ops, "apply_schema", fake_apply_schema)
    monkeypatch.setattr(
        physics_workflow.scene_ops, "validate_runtime", fake_validate_runtime
    )

    run_physics_apply_workflow(
        PhysicsApplyWorkflowInput(
            usd_path=tmp_path / "input.usda",
            output_dir=tmp_path / "out",
            simulation_engine="fake",
            max_ground_penetration_m=0.05,
        )
    )

    assert validate_kwargs, "runtime validation did not run"
    acceptance = validate_kwargs[0].get("acceptance") or {}
    assert acceptance.get("max_ground_penetration_m") == 0.05


def test_validate_physics_runtime_multi_body_engine_none_not_evaluated(
    tmp_path: Path,
) -> None:
    """A disabled engine must yield fail-safe not_evaluated evidence, matching
    validate_physics_runtime — not a fabricated per-body runtime failure."""
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")

    evidence, report_path = physics_workflow.validate_physics_runtime_multi_body(
        physics_usd=source,
        output_dir=tmp_path / "runtime",
        body_prim_paths=["/Asset/BodyA", "/Asset/BodyB"],
        engine="none",
    )

    assert report_path is not None
    skip_report = json.loads(Path(str(report_path)).read_text(encoding="utf-8"))
    assert skip_report["not_evaluated"] is True
    assert skip_report["skip_reason"]
    assert skip_report["body_prim_paths"]
    statuses = {check.name: check.status for check in evidence.checks}
    assert statuses["runtime_loadability"] == "not_evaluated"
    assert statuses["no_explosions"] == "not_evaluated"
    assert "Runtime simulation was disabled." in evidence.unresolved_issues


def test_validate_physics_runtime_multi_body_no_common_ancestor_skips(
    tmp_path: Path,
) -> None:
    """Bodies without a shared placement root must fail safe: per-body drop
    placement would move only the tracked body and break relative poses."""
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")

    evidence, report_path = physics_workflow.validate_physics_runtime_multi_body(
        physics_usd=source,
        output_dir=tmp_path / "runtime",
        body_prim_paths=["/A/Body", "/B/Body"],
        engine="fake",
    )

    assert report_path is not None
    skip_report = json.loads(Path(str(report_path)).read_text(encoding="utf-8"))
    assert skip_report["not_evaluated"] is True
    assert skip_report["skip_reason"]
    assert skip_report["body_prim_paths"]
    statuses = {check.name: check.status for check in evidence.checks}
    assert statuses["runtime_loadability"] == "not_evaluated"
    assert any(
        "no common Xformable ancestor" in issue for issue in evidence.unresolved_issues
    )


@pytest.mark.parametrize(
    "variant",
    ["missing", "empty", "invalid", "nondict", "nonfinite", "none", "unreadable"],
)
def test_multi_body_trajectory_guards_fail_the_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    """Each trajectory guard (missing/empty/invalid/non-dict/non-finite
    artifact) must fail the body — deleting any of the fail-closed returns
    would record a pass with no usable trajectory evidence."""
    from pxr import Usd, UsdGeom

    source = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(source))
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdGeom.Xform.Define(stage, "/Asset/Body")
    stage.GetRootLayer().Save()

    def fake_validate_runtime(**kwargs: object) -> dict[str, object]:
        body_dir = Path(str(kwargs["output_dir"]))
        body_dir.mkdir(parents=True, exist_ok=True)
        trajectory = body_dir / "trajectory.jsonl"
        if variant == "empty":
            trajectory.write_text("\n\n", encoding="utf-8")
        elif variant == "invalid":
            trajectory.write_text("{broken\n", encoding="utf-8")
        elif variant == "nondict":
            trajectory.write_text("[1, 2, 3]\n", encoding="utf-8")
        elif variant == "nonfinite":
            trajectory.write_text(
                json.dumps(
                    {
                        "t": 0.0,
                        "pose": [float("inf")] + [0.0] * 5 + [1.0],
                        "vel": [0.0] * 6,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif variant == "unreadable":
            trajectory.write_text(
                json.dumps({"t": 0.0, "pose": [0.0] * 6 + [1.0], "vel": [0.0] * 6})
                + "\n",
                encoding="utf-8",
            )
            trajectory.chmod(0)
        # "missing": never written; "none": no artifact path recorded at all.
        return {
            "engine": "fake",
            "trajectory_jsonl": None if variant == "none" else str(trajectory),
            "failures": [],
            "warnings": [],
            "diagnostics": [],
            "acceptance": {},
        }

    monkeypatch.setattr(
        physics_workflow.scene_ops, "validate_runtime", fake_validate_runtime
    )

    evidence, report_path = physics_workflow.validate_physics_runtime_multi_body(
        physics_usd=source,
        output_dir=tmp_path / "runtime",
        body_prim_paths=["/Asset/Body"],
        engine="fake",
    )

    report = json.loads(Path(str(report_path)).read_text(encoding="utf-8"))
    assert report["per_body_results"][0]["status"] == "fail"
    assert report["failures"]
    assert evidence.sim_ready_status == "fail"


def test_aggregate_recording_converts_world_poses_to_local(tmp_path: Path) -> None:
    """Simulator poses are world-space; under a translated placement ancestor
    the authored LOCAL samples must subtract the ancestor transform, or the
    rendered rollout re-applies it and the asset floats above the ground."""
    from pxr import Gf, Usd, UsdGeom

    base = tmp_path / "recording.usda"
    stage = Usd.Stage.CreateNew(str(base))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    group = UsdGeom.Xform.Define(stage, "/World/Grp")
    # The drop placement translated the ancestor by +5 in Z.
    group.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))
    UsdGeom.Xform.Define(stage, "/World/Grp/Body")
    stage.GetRootLayer().Save()

    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "t": 0.0,
                # World pose: the body rests at z=1.
                "pose": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "vel": [0.0] * 6,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    aggregate = physics_workflow._author_multi_body_recording(
        [
            {
                "body_prim_path": "/World/Grp/Body",
                "recording_usda": str(base),
                "trajectory_jsonl": str(trajectory),
            }
        ]
    )

    assert aggregate is not None
    out = Usd.Stage.Open(aggregate)
    body = out.GetPrimAtPath("/World/Grp/Body")
    xf = UsdGeom.Xformable(body)
    t_op = next(
        op for op in xf.GetOrderedXformOps() if op.GetOpName() == "xformOp:translate"
    )
    local = t_op.Get(Usd.TimeCode(0.0))
    # Local = world (z=1) minus the ancestor's +5 → -4.
    assert local[2] == pytest.approx(-4.0)
    # And the COMPOSED world position matches the simulator pose.
    cache = UsdGeom.XformCache(Usd.TimeCode(0.0))
    composed = cache.GetLocalToWorldTransform(body).ExtractTranslation()
    assert composed[2] == pytest.approx(1.0)
