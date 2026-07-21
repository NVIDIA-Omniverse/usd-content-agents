# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from content_workbench import physics_ops


def test_decision_patch_predictions_expand_grouped_prims(tmp_path: Path) -> None:
    patch_path = tmp_path / "physics_decision_patch.json"
    patch_path.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "prim_paths": ["/World/A", "/World/B"],
                        "component_label": "grouped component",
                        "inferred_material_family": "metal",
                        "collision_approximation": "convexHull",
                        "physical_properties": {
                            "density": 2700.0,
                            "estimated_mass_kg": 1.0,
                        },
                        "confidence": 0.8,
                        "rationale": "Grouped decision fixture.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = physics_ops._prediction_records_from_decision_patch(patch_path)

    assert [record["id"] for record in records] == ["/World/A", "/World/B"]
    assert [
        record["classification"]["physical_properties"]["estimated_mass_kg"]
        for record in records
    ] == [0.5, 0.5]
    assert records[0]["source"] == "content_workbench.physics_ops.decision_patch"


def test_v2_decision_records_carry_explicit_body_mass_target(tmp_path: Path) -> None:
    patch_path = tmp_path / "physics_decision_patch.json"
    patch_path.write_text(
        json.dumps(
            {
                "schema_version": "content-agent-workflows.physics-decision-patch.v2",
                "decisions": [
                    {
                        "component_id": "component_001",
                        "body_root_path": "/World",
                        "collider_paths": ["/World/A", "/World/B"],
                        "collision_mode": "preserve_existing",
                        "mass_authoring_path": "/World",
                        "inferred_material_family": "metal",
                        "physical_properties": {
                            "density": 2700.0,
                            "estimated_mass_kg": 1.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    records = physics_ops._prediction_records_from_decision_patch(patch_path)

    assert [record["id"] for record in records] == ["/World/A", "/World/B"]
    assert all(
        record["classification"]["mass_authoring_path"] == "/World"
        for record in records
    )
    assert all(
        record["classification"]["collision_mode"] == "preserve_existing"
        for record in records
    )
    assert [
        record["classification"]["physical_properties"]["estimated_mass_kg"]
        for record in records
    ] == [0.5, 0.5]
    assert [
        record["classification"]["component_estimated_mass_kg"] for record in records
    ] == [1.0, 1.0]


def test_validate_v2_decision_patch_rejects_stale_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        physics_ops,
        "inspect_components",
        lambda _usd_path: {
            "source_digest": "sha256:current",
            "components": [
                {
                    "component_id": "component_001",
                    "body_root_path": "/World",
                    "helper_paths": [],
                    "collider_paths": ["/World/Collision"],
                    "visual_evidence_paths": ["/World/Visual"],
                }
            ],
        },
    )
    patch_path = tmp_path / "stale.json"
    patch_path.write_text(
        json.dumps(
            {
                "schema_version": "content-agent-workflows.physics-decision-patch.v2",
                "source_digest": "sha256:older",
                "decisions": [
                    {
                        "component_id": "component_001",
                        "body_root_path": "/World",
                        "mass_authoring_path": "/World",
                        "collision_mode": "preserve_existing",
                        "collider_paths": ["/World/Collision"],
                        "physical_properties": {"density": 100.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_digest"):
        physics_ops._validate_v2_decision_patch(tmp_path / "asset.usda", patch_path)


def test_validate_v2_decision_patch_rejects_malformed_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        physics_ops,
        "inspect_components",
        lambda _usd_path: {
            "source_digest": "sha256:test",
            "components": [
                {
                    "component_id": "component_001",
                    "body_root_path": "/World",
                    "helper_paths": [],
                    "collider_paths": ["/World/Collision"],
                    "visual_evidence_paths": ["/World/Visual"],
                }
            ],
        },
    )

    invalid_payloads = [
        [],
        {
            "schema_version": "content-agent-workflows.physics-decision-patch.v2",
            "source_digest": "sha256:test",
            "decisions": {},
        },
        {
            "schema_version": "content-agent-workflows.physics-decision-patch.v2",
            "source_digest": "sha256:test",
            "decisions": ["/World/Collision"],
        },
        {
            "schema_version": "content-agent-workflows.physics-decision-patch.v2",
            "source_digest": "sha256:test",
            "decisions": [{"component_id": "component_001", "collider_paths": []}],
            "unresolved_components": {},
        },
        {
            "schema_version": "content-agent-workflows.physics-decision-patch.v2",
            "source_digest": "sha256:test",
            "decisions": [
                {
                    "component_id": "component_001",
                    "body_root_path": "/World",
                    "mass_authoring_path": "/World",
                    "collision_mode": "preserve_existing",
                    "collider_paths": "/World/Collision",
                }
            ],
        },
        {
            "schema_version": "content-agent-workflows.physics-decision-patch.v2",
            "source_digest": "sha256:test",
            "decisions": [
                {
                    "component_id": "component_001",
                    "body_root_path": "/World",
                    "mass_authoring_path": "/World",
                    "collision_mode": "author_on_targets",
                    "collider_paths": ["/World/Visual"],
                }
            ],
        },
    ]

    for index, payload in enumerate(invalid_payloads):
        patch_path = tmp_path / f"invalid_{index}.json"
        patch_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError):
            physics_ops._validate_v2_decision_patch(tmp_path / "asset.usda", patch_path)


def test_apply_schema_prefers_decision_patch_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.functions import apply_physics as apply_physics_module

    patch_path = tmp_path / "physics_decision_patch.json"
    patch_path.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "prim_paths": ["/World/A", "/World/B"],
                        "component_label": "grouped component",
                        "inferred_material_family": "metal",
                        "collision_approximation": "convexHull",
                        "physical_properties": {
                            "density": 2700.0,
                            "estimated_mass_kg": 1.0,
                        },
                        "confidence": 0.8,
                        "rationale": "Grouped decision fixture.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    stale_predictions_path = tmp_path / "stale_predictions.jsonl"
    stale_predictions_path.write_text(
        json.dumps({"id": "/World/Stale", "classification": {}}) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "physics.usda"
    called: dict[str, str] = {}

    def fake_apply_physics(
        usd_path: str,
        predictions_path: str,
        authored_output_path: str,
        **_kwargs: Any,
    ) -> str:
        called["usd_path"] = usd_path
        called["predictions_path"] = predictions_path
        Path(authored_output_path).write_text("#usda 1.0\n", encoding="utf-8")
        return authored_output_path

    monkeypatch.setattr(
        apply_physics_module,
        "apply_physics",
        fake_apply_physics,
    )
    monkeypatch.setattr(
        physics_ops,
        "inspect_authored_physics",
        lambda authored: {"physics_usd": str(authored), "collision_count": 2},
    )

    report = physics_ops.apply_schema(
        usd_path=tmp_path / "asset.usda",
        decision_patch_path=patch_path,
        predictions_jsonl_path=stale_predictions_path,
        output_usd_path=output_path,
    )

    authored_predictions_path = Path(called["predictions_path"])
    records = [
        json.loads(line)
        for line in authored_predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert authored_predictions_path != stale_predictions_path
    assert [record["id"] for record in records] == ["/World/A", "/World/B"]
    assert report["predictions_jsonl"] == str(authored_predictions_path)
    assert report["source_predictions_jsonl"] == str(stale_predictions_path.resolve())


def test_write_json_sanitizes_non_finite_values(tmp_path: Path) -> None:
    path = physics_ops._write_json(
        tmp_path / "report.json",
        {"distance": float("inf"), "nested": [float("nan"), 1.0]},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {"distance": None, "nested": [None, 1.0]}


def test_inspect_authored_physics_reports_enabled_rigid_bodies(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    asset = tmp_path / "authored.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    enabled = UsdGeom.Xform.Define(stage, "/World/Enabled").GetPrim()
    disabled = UsdGeom.Xform.Define(stage, "/World/Disabled").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(enabled).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.RigidBodyAPI.Apply(disabled).CreateRigidBodyEnabledAttr(False)
    stage.GetRootLayer().Save()

    report = physics_ops.inspect_authored_physics(asset)

    assert report["rigid_body_count"] == 2
    assert report["enabled_rigid_body_count"] == 1
    assert report["disabled_rigid_body_count"] == 1
    assert report["enabled_rigid_body_paths"] == ["/World/Enabled"]
    assert report["disabled_rigid_body_paths"] == ["/World/Disabled"]


def test_validate_runtime_rejects_invalid_engine_before_side_effects(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "runtime"

    with pytest.raises(ValueError, match="engine must be one of"):
        physics_ops.validate_runtime(
            physics_usd=tmp_path / "missing.usda",
            output_dir=output_dir,
            engine="typo",  # type: ignore[arg-type]
        )

    assert not output_dir.exists()


def test_validate_runtime_reports_empty_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tuning.scenarios import _scene_builder

    def fake_build_drop_settle_scene(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "rest_position": [0.0, 0.0, 0.0],
            "world_up": [0.0, 0.0, 1.0],
            "drop_height_m_resolved": 0.05,
            "body_pattern": "/World",
            "body_prim_path": "/World",
        }

    monkeypatch.setattr(
        _scene_builder,
        "build_drop_settle_scene",
        fake_build_drop_settle_scene,
    )
    monkeypatch.setattr(physics_ops, "_fake_trajectory", lambda **_kwargs: [])

    result = physics_ops.validate_runtime(
        physics_usd=tmp_path / "physics.usda",
        output_dir=tmp_path / "runtime",
        engine="fake",
    )

    assert result["recording_usda"] is None
    assert result["failures"] == ["Simulation produced no trajectory samples."]
    assert Path(result["trajectory_jsonl"]).read_text(encoding="utf-8") == ""
    report = json.loads(Path(result["runtime_report"]).read_text(encoding="utf-8"))
    assert report["failures"] == ["Simulation produced no trajectory samples."]
    assert report["acceptance"] is None
    assert result["acceptance"] is None


def test_apply_topology_plan_report_contains_self_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "prepared.usda"

    def fake_apply_physics_topology_plan(**kwargs: object) -> dict[str, object]:
        return {
            "operation": "physics.apply_topology_plan",
            "output_usd_path": str(kwargs["output_usd_path"]),
        }

    monkeypatch.setattr(
        "world_understanding.functions.physics.physics_topology."
        "apply_physics_topology_plan",
        fake_apply_physics_topology_plan,
    )

    result = physics_ops.apply_topology_plan(
        input_usd_path=tmp_path / "input.usda",
        output_usd_path=output,
        expected_source_digest="sha256:source",
        mobility_intent="movable",
        operations=[],
        invariants={"enabled_collider_count": 1, "reject_articulation_changes": True},
    )

    report = json.loads(Path(result["topology_report"]).read_text(encoding="utf-8"))
    assert report["topology_report"] == result["topology_report"]


def test_validate_runtime_skips_acceptance_metrics_when_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent import recording
    from physics_agent.tuning.scenarios import _scene_builder

    def fake_build_drop_settle_scene(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "rest_position": [0.0, 0.0, 0.0],
            "world_up": [0.0, 0.0, 1.0],
            "drop_height_m_resolved": 0.05,
            "body_pattern": "/World",
            "body_prim_path": "/World",
        }

    def fake_author_trajectory_jsonl(
        _trajectory: object,
        path: Path,
        **_kwargs: object,
    ) -> Path:
        path.write_text("{}\n", encoding="utf-8")
        return path

    def fake_author_trajectory_usda(
        _scene_path: Path,
        _trajectory: object,
        _body_prim_path: str,
        path: Path,
        **_kwargs: object,
    ) -> Path:
        path.write_text("#usda 1.0\n", encoding="utf-8")
        return path

    monkeypatch.setattr(
        _scene_builder,
        "build_drop_settle_scene",
        fake_build_drop_settle_scene,
    )
    monkeypatch.setattr(
        recording,
        "author_trajectory_jsonl",
        fake_author_trajectory_jsonl,
    )
    monkeypatch.setattr(
        recording,
        "author_trajectory_usda",
        fake_author_trajectory_usda,
    )

    result = physics_ops.validate_runtime(
        physics_usd=tmp_path / "physics.usda",
        output_dir=tmp_path / "runtime",
        engine="fake",
        duration_s=0.1,
        sample_fps=10,
    )

    assert result["acceptance"] is None
    assert result["failures"] == []
    assert "initial_pose_discontinuity" not in result["summary"]
    report = json.loads(Path(result["runtime_report"]).read_text(encoding="utf-8"))
    assert report["acceptance"] is None

    accepted = physics_ops.validate_runtime(
        physics_usd=tmp_path / "physics.usda",
        output_dir=tmp_path / "runtime-accepted",
        engine="fake",
        duration_s=0.1,
        sample_fps=10,
        acceptance={
            "detect_initial_pose_discontinuity": False,
            "require_gravity_response": False,
            "expected_body_count": 1,
        },
    )

    assert accepted["acceptance"]["expected_body_count"] == 1
    assert accepted["acceptance"]["detect_initial_pose_discontinuity"] is False
    assert accepted["acceptance"]["max_ground_penetration_m"] is None
    assert "initial_pose_discontinuity" in accepted["summary"]

    zero_drop = physics_ops.validate_runtime(
        physics_usd=tmp_path / "physics.usda",
        output_dir=tmp_path / "runtime-zero-drop",
        engine="fake",
        duration_s=0.1,
        sample_fps=10,
        drop_height_m=0.0,
        acceptance={"expected_body_count": 1},
    )

    assert zero_drop["acceptance"]["require_gravity_response"] is False
    assert zero_drop["acceptance"]["detect_initial_pose_discontinuity"] is False
    assert not any("gravity response" in failure for failure in zero_drop["failures"])

    aliased = physics_ops.validate_runtime(
        physics_usd=tmp_path / "physics.usda",
        output_dir=tmp_path / "runtime-aliased",
        engine="fake",
        duration_s=0.1,
        sample_fps=10,
        acceptance={
            "detect_discontinuity": False,
            "require_gravity_response": False,
            "expected_body_count": 1,
        },
    )

    assert "detect_discontinuity" not in aliased["acceptance"]
    assert aliased["acceptance"]["detect_initial_pose_discontinuity"] is False


def test_runtime_acceptance_rejects_pose_snap_penetration_and_body_mismatch() -> None:
    metrics, failures = physics_ops._runtime_acceptance_metrics(
        trajectory=[
            (0.0, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0] * 6),
            (
                1.0 / 30.0,
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            ),
        ],
        scene_info={
            "world_up": [0.0, 0.0, 1.0],
            "bbox_size_m": [1.0, 1.0, 1.0],
            "bbox_min_local_stage": [-0.5, -0.5, -0.5],
            "bbox_max_local_stage": [0.5, 0.5, 0.5],
        },
        loaded_body_count=2,
        acceptance={"expected_body_count": 1},
    )

    assert metrics["initial_pose_discontinuity"] is True
    assert metrics["maximum_ground_penetration_m"] == pytest.approx(0.5)
    assert metrics["gravity_response_observed"] is True
    assert len(failures) == 3
    assert any("discontinuous" in failure for failure in failures)
    assert any("penetrated" in failure for failure in failures)
    assert any("body count" in failure for failure in failures)


def test_runtime_acceptance_allows_continuous_drop_to_ground() -> None:
    metrics, failures = physics_ops._runtime_acceptance_metrics(
        trajectory=[
            (0.0, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0] * 6),
            (
                1.0 / 30.0,
                [0.0, 0.0, 0.995, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -0.3, 0.0, 0.0, 0.0],
            ),
            (1.0, [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], [0.0] * 6),
        ],
        scene_info={
            "world_up": [0.0, 0.0, 1.0],
            "bbox_size_m": [1.0, 1.0, 1.0],
            "bbox_min_local_stage": [-0.5, -0.5, -0.5],
            "bbox_max_local_stage": [0.5, 0.5, 0.5],
        },
        loaded_body_count=1,
        acceptance={"expected_body_count": 1},
    )

    assert failures == []
    assert metrics["initial_pose_discontinuity"] is False
    assert metrics["maximum_ground_penetration_m"] == pytest.approx(0.0)
    assert metrics["gravity_response_observed"] is True


def test_runtime_acceptance_uses_scene_gravity_for_pose_snap_threshold() -> None:
    metrics, failures = physics_ops._runtime_acceptance_metrics(
        trajectory=[
            (0.0, [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0], [0.0] * 6),
            (
                1.0,
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            ),
        ],
        scene_info={
            "world_up": [0.0, 0.0, 1.0],
            "gravity_magnitude_m_per_s2": 1.0,
        },
        loaded_body_count=1,
        acceptance={
            "expected_body_count": 1,
            "require_gravity_response": False,
        },
    )

    assert metrics["gravity_magnitude_m_per_s2"] == pytest.approx(1.0)
    assert metrics["expected_ballistic_displacement_m"] == pytest.approx(0.5)
    assert metrics["initial_pose_discontinuity"] is True
    assert len(failures) == 1
    assert "discontinuous" in failures[0]


def test_runtime_acceptance_detects_mid_sim_ground_penetration_after_rebound() -> None:
    metrics, failures = physics_ops._runtime_acceptance_metrics(
        trajectory=[
            (0.0, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0] * 6),
            (
                0.1,
                [0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            ),
            (
                0.2,
                [0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ),
        ],
        scene_info={
            "world_up": [0.0, 0.0, 1.0],
            "bbox_size_m": [1.0, 1.0, 1.0],
            "bbox_min_local_stage": [-0.5, -0.5, -0.5],
            "bbox_max_local_stage": [0.5, 0.5, 0.5],
        },
        loaded_body_count=1,
        acceptance={
            "detect_initial_pose_discontinuity": False,
            "require_gravity_response": False,
            "max_ground_penetration_m": 0.005,
        },
    )

    assert metrics["maximum_ground_penetration_m"] == pytest.approx(0.6)
    assert len(failures) == 1
    assert "penetrated" in failures[0]


def test_runtime_acceptance_allows_disabled_ground_penetration_limit() -> None:
    metrics, failures = physics_ops._runtime_acceptance_metrics(
        trajectory=[
            (0.0, [0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0], [0.0] * 6),
        ],
        scene_info={
            "world_up": [0.0, 0.0, 1.0],
            "bbox_min_local_stage": [-0.5, -0.5, -0.5],
            "bbox_max_local_stage": [0.5, 0.5, 0.5],
        },
        loaded_body_count=1,
        acceptance={
            "detect_initial_pose_discontinuity": False,
            "require_gravity_response": False,
            "max_ground_penetration_m": None,
            "expected_body_count": 1,
        },
    )

    assert metrics["maximum_ground_penetration_m"] == pytest.approx(1.0)
    assert failures == []


def test_runtime_acceptance_rotates_pose_local_bounds_by_pose() -> None:
    quarter_turn_y = math.sin(math.pi / 4.0)
    quarter_turn_w = math.cos(math.pi / 4.0)

    metrics, failures = physics_ops._runtime_acceptance_metrics(
        trajectory=[
            (
                0.0,
                [0.0, 0.0, 0.05, 0.0, quarter_turn_y, 0.0, quarter_turn_w],
                [0.0] * 6,
            ),
            (
                0.1,
                [0.0, 0.0, 0.05, 0.0, quarter_turn_y, 0.0, quarter_turn_w],
                [0.0, 0.0, -0.1, 0.0, 0.0, 0.0],
            ),
        ],
        scene_info={
            "world_up": [0.0, 0.0, 1.0],
            "bbox_min_local_stage": [-0.05, -0.05, -1.0],
            "bbox_max_local_stage": [0.05, 0.05, 1.0],
            "bbox_local_stage_space": "pose_local",
        },
        loaded_body_count=1,
        acceptance={
            "detect_initial_pose_discontinuity": False,
            "require_gravity_response": False,
            "max_ground_penetration_m": 0.005,
            "expected_body_count": 1,
        },
    )

    assert metrics["maximum_ground_penetration_m"] == pytest.approx(0.0)
    assert failures == []


def test_runtime_acceptance_scales_pose_local_bounds_before_rotation() -> None:
    metrics, failures = physics_ops._runtime_acceptance_metrics(
        trajectory=[
            (
                0.0,
                [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
                [0.0] * 6,
            ),
            (
                0.1,
                [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -0.1, 0.0, 0.0, 0.0],
            ),
        ],
        scene_info={
            "world_up": [0.0, 0.0, 1.0],
            "meters_per_unit": 1.0,
            "bbox_min_local_stage": [-1.0, -1.0, -1.0],
            "bbox_max_local_stage": [1.0, 1.0, 1.0],
            "bbox_local_stage_scale": [1.0, 1.0, 0.1],
            "bbox_local_stage_space": "pose_local",
        },
        loaded_body_count=1,
        acceptance={
            "detect_initial_pose_discontinuity": False,
            "require_gravity_response": False,
            "max_ground_penetration_m": 0.005,
            "expected_body_count": 1,
        },
    )

    assert metrics["maximum_ground_penetration_m"] == pytest.approx(0.0)
    assert failures == []


def test_runtime_acceptance_does_not_rerotate_world_aligned_bounds() -> None:
    quarter_turn_y = math.sin(math.pi / 4.0)
    quarter_turn_w = math.cos(math.pi / 4.0)

    metrics, failures = physics_ops._runtime_acceptance_metrics(
        trajectory=[
            (
                0.0,
                [0.0, 0.0, 0.05, 0.0, quarter_turn_y, 0.0, quarter_turn_w],
                [0.0] * 6,
            ),
            (
                0.1,
                [0.0, 0.0, 0.05, 0.0, quarter_turn_y, 0.0, quarter_turn_w],
                [0.0, 0.0, -0.1, 0.0, 0.0, 0.0],
            ),
        ],
        scene_info={
            "world_up": [0.0, 0.0, 1.0],
            "bbox_min_local_stage": [-1.0, -0.05, -0.05],
            "bbox_max_local_stage": [1.0, 0.05, 0.05],
            "bbox_local_stage_space": "world_aligned_translation_removed",
        },
        loaded_body_count=1,
        acceptance={
            "detect_initial_pose_discontinuity": False,
            "require_gravity_response": False,
            "max_ground_penetration_m": 0.005,
            "expected_body_count": 1,
        },
    )

    assert metrics["maximum_ground_penetration_m"] == pytest.approx(0.0)
    assert failures == []


def test_meters_per_stage_unit_prefers_scene_metadata() -> None:
    assert physics_ops._meters_per_stage_unit(
        {
            "meters_per_unit": 0.01,
            "bbox_size_m": [100.0, 100.0, 100.0],
            "bbox_min_local_stage": [-0.5, -0.5, -0.5],
            "bbox_max_local_stage": [0.5, 0.5, 0.5],
        }
    ) == pytest.approx(0.01)
