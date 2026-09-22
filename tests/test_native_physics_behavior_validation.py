# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic contract tests: no solver, renderer, model, or provider is invoked."""
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from world_understanding.agentic.validation_scaffold import (
    create_draft_validation_request,
    run_validation_scaffold,
)
from world_understanding.functions.physics import native_behavior_validation as adapter


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def native(tmp_path):
    asset = tmp_path / "physics.usda"
    asset.write_text('#usda 1.0\ndef Xform "Asset" {}\n')
    scene = tmp_path / "scene.usda"
    scene.write_bytes(asset.read_bytes())
    recording = tmp_path / "recording.usda"
    recording.write_bytes(asset.read_bytes())
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text(
        "\n".join(
            json.dumps(
                {
                    "frame": i,
                    "t": float(i),
                    "pose": [0, 0, 0, 0, 0, 0, 1],
                    "vel": [0] * 6,
                }
            )
            for i in range(3)
        )
    )
    raw_sim = write(
        tmp_path / "raw_runtime.json",
        {
            "engine": "ovphysx",
            "scene_usd": str(scene),
            "recording_usda": str(recording),
            "trajectory_jsonl": str(trajectory),
            "simulation_facts": {
                "trajectory_sample_count": 3,
                "trajectory_finite": True,
                "reported_step_count": 480,
                "reported_body_count": 1,
            },
        },
    )
    response = write(
        tmp_path / "response.json",
        {
            "engine": "ovphysx",
            "status": "ok",
            "n_steps": 480,
            "n_bodies": 1,
            "trajectory_sample_count": 3,
            "usd_cli_report_path": str(raw_sim),
        },
    )
    report = write(
        tmp_path / "runtime.json",
        {
            "engine": "ovphysx",
            "failures": [],
            "warnings": [],
            "physics_usd": str(asset),
            "scene_usd": str(scene),
            "recording_usda": str(recording),
            "trajectory_jsonl": str(trajectory),
            "response_path": str(response),
            "acceptance": {"expected_body_count": 1},
            "summary": {
                "n_samples": 3,
                "loaded_body_count": 1,
                "initial_pose_discontinuity": False,
                "duration_s": 2.0,
            },
        },
    )
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32), "red").save(frame)
    camera = write(tmp_path / "frame.camera.json", {"camera_path": "/Camera"})
    render_response = write(
        tmp_path / "raw/physics_render_frames_1.json",
        {"renderer": "ovrtx", "status": "completed"},
    )
    render_inputs = {
        k: adapter._binding(p)
        for k, p in {
            "physics_usd": asset,
            "scene_usd": scene,
            "recording_usda": recording,
        }.items()
    }
    bundle_files = list(render_inputs.values())
    receipt = write(
        tmp_path / "frame_receipt.json",
        {
            "schema_version": "content-workflow-cli.physics-render-frame-receipt.v1",
            "iteration": 1,
            "simulation_report": adapter._binding(report),
            "render_inputs": render_inputs,
            "render_input_bundle": {
                "files": bundle_files,
                "sha256": hashlib.sha256(
                    json.dumps(
                        bundle_files,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
            },
            "render_response": {"renderer": "ovrtx"},
            "response_artifact": adapter._binding(render_response),
            "frames": [
                {
                    "local_frame_path": str(frame),
                    "frame_sha256": adapter._binding(frame)["sha256"],
                    "frame_size_bytes": frame.stat().st_size,
                    "local_camera_path": str(camera),
                    "camera_sha256": adapter._binding(camera)["sha256"],
                    "camera_size_bytes": camera.stat().st_size,
                }
            ],
        },
    )
    assessment = write(
        tmp_path / "assessment.json",
        {
            "schema_version": adapter.ASSESSMENT_SCHEMA,
            "status": "pass",
            "checked_views": [str(frame)],
            "runtime_report": str(report),
            "rendered_frames": [str(frame)],
            "issues_found": [],
            "issues_fixed": [],
            "unresolved_issues": [],
            "assessment_notes": "Synthetic contract fixture, not physical evidence.",
        },
    )
    validation = write(
        tmp_path / "validation.json",
        {
            "schema_version": adapter.EVIDENCE_SCHEMA,
            "workflow": "physics_authoring",
            "asset": str(asset),
            "target_runtime": "ovphysx",
            "validation_tier": "T2_simulation_match",
            "sim_ready_status": "pass",
            "checks": [
                {"name": name, "status": "pass", "summary": "Synthetic fixture"}
                for name in (
                    "physics_properties",
                    "runtime_loadability",
                    "no_explosions",
                    "simulation_visual_review",
                )
            ],
            "evidence_artifacts": [
                {
                    "kind": "simulation_frame_receipt",
                    "path": str(receipt),
                    "metadata": {"sha256": adapter._binding(receipt)["sha256"]},
                }
            ],
            "failures": [],
            "warnings": [],
            "unresolved_issues": [],
            "metadata": {
                "asset_sha256": adapter._binding(asset)["sha256"],
                "duration_s": 2.0,
            },
        },
    )
    evidence = json.loads(validation.read_text())
    evidence["checks"][-1]["metadata"] = {"assessment_status": "pass"}
    evidence["evidence_artifacts"] += [
        {"kind": "physics_behavior_assessment", "path": str(assessment)},
        {"kind": "runtime_report", "path": str(report)},
    ]
    write(validation, evidence)
    data = {
        "asset": asset,
        "assessment": assessment,
        "validation_evidence": validation,
        "run_dir": tmp_path,
    }
    bundle = write(
        tmp_path / "bundle.json", adapter.prepare_native_physics_bundle(**data)
    )
    return {
        **data,
        "bundle": bundle,
        "report": report,
        "receipt": receipt,
        "frame": frame,
        "trajectory": trajectory,
        "response": response,
    }


def validate(native):
    return adapter.validate_native_physics_bundle(
        native["bundle"], usd_paths=[native["asset"]]
    )


def refresh_outer_bundle(native):
    """Rebind mutated bytes to exercise semantic checks beyond stale-hash checks."""
    bundle = json.loads(native["bundle"].read_text())
    bundle["artifacts"] = [
        adapter._binding(Path(a["path"])) for a in bundle["artifacts"]
    ]
    write(native["bundle"], bundle)


def test_actual_native_verifier_clean_contract(native):
    result = validate(native)
    assert result["status"] == "passed"
    assert result["native_assessment_status"] == "pass"
    assert "decision" not in result


@pytest.mark.parametrize(
    "target",
    [
        "asset",
        "assessment",
        "validation_evidence",
        "report",
        "receipt",
        "frame",
        "trajectory",
        "response",
    ],
)
def test_stale_evidence_never_passes(native, target):
    with native[target].open("ab") as stream:
        stream.write(b" ")
    assert validate(native)["status"] == "failed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "unresolved_issues"),
        ("status", "warn"),
        ("unresolved_issues", ["unresolved"]),
        ("checked_views", []),
        ("rendered_frames", []),
        ("schema_version", "unknown"),
    ],
)
def test_rebound_bad_assessment_rejected(native, field, value):
    raw = json.loads(native["assessment"].read_text())
    raw[field] = value
    write(native["assessment"], raw)
    refresh_outer_bundle(native)
    assert validate(native)["status"] == "failed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("sim_ready_status", "fail"),
        ("sim_ready_status", "conditional"),
        ("warnings", ["warning"]),
        ("failures", ["failure"]),
        ("checks", []),
        ("metadata", {"duration_s": 2.0}),
    ],
)
def test_rebound_native_failure_rejected(native, field, value):
    raw = json.loads(native["validation_evidence"].read_text())
    raw[field] = value
    write(native["validation_evidence"], raw)
    refresh_outer_bundle(native)
    assert validate(native)["status"] == "failed"


def test_trajectory_incomplete_even_with_fresh_manifest(native):
    native["trajectory"].write_text(native["trajectory"].read_text().splitlines()[0])
    refresh_outer_bundle(native)
    assert validate(native)["status"] == "failed"


def test_source_mismatch_and_omitted_dependency(native):
    other = native["run_dir"] / "other.usda"
    other.write_bytes(native["asset"].read_bytes())
    assert (
        adapter.validate_native_physics_bundle(native["bundle"], usd_paths=[other])[
            "status"
        ]
        == "failed"
    )
    raw = json.loads(native["bundle"].read_text())
    raw["artifacts"] = [
        a for a in raw["artifacts"] if a["path"] != str(native["trajectory"])
    ]
    write(native["bundle"], raw)
    assert validate(native)["status"] == "failed"


def scaffold(native, evidence):
    request = create_draft_validation_request(
        task_description="Check recorded single-body runtime stability and visual review.",
        inputs=(native["asset"],),
        working_dir=native["run_dir"] / "validation_run",
        requested_templates=("physical_behavior",),
        policy={
            "behavior_evidence_required": True,
            "physical_behavior_evidence": evidence,
        },
    )
    return run_validation_scaffold(request)


def spec(path, role):
    return {
        "path": str(path),
        "kind": "simulation_json",
        "role": role,
        "required": True,
    }


def test_native_scaffold_and_legacy_approval_are_distinct(native):
    result = scaffold(native, [spec(native["bundle"], "native_physics_bundle")])
    assert result.verdict == "pass"
    assert (
        result.template_results[0].evidence["behavior_summary"]["kind"]
        == "native_physics_behavior"
    )
    legacy = write(
        native["run_dir"] / "legacy.json",
        {"status": "completed", "decision": "approve"},
    )
    assert scaffold(native, [spec(legacy, "judge_result")]).verdict == "pass"
    assert (
        scaffold(
            native,
            [
                spec(legacy, "judge_result"),
                spec(native["assessment"], "physics_behavior_assessment"),
            ],
        ).verdict
        == "fail"
    )


def test_legacy_approve_cannot_hide_failed_native_bundle(native):
    legacy = write(
        native["run_dir"] / "legacy.json",
        {"status": "completed", "decision": "approve"},
    )
    native["trajectory"].write_text("")
    assert (
        scaffold(
            native,
            [
                spec(legacy, "judge_result"),
                spec(native["bundle"], "native_physics_bundle"),
            ],
        ).verdict
        == "fail"
    )


def test_native_pass_cannot_hide_legacy_failure_or_missing_required(native):
    legacy = write(
        native["run_dir"] / "legacy.json", {"status": "failed", "decision": "approve"}
    )
    assert (
        scaffold(
            native,
            [
                spec(legacy, "judge_result"),
                spec(native["bundle"], "native_physics_bundle"),
            ],
        ).verdict
        == "fail"
    )
    assert (
        scaffold(
            native,
            [
                spec(native["run_dir"] / "missing.json", "simulation_json"),
                spec(native["bundle"], "native_physics_bundle"),
            ],
        ).verdict
        == "fail"
    )


def test_missing_native_bundle_fails_without_legacy_fallback(native):
    assert (
        scaffold(
            native, [spec(native["run_dir"] / "missing.json", "native_physics_bundle")]
        ).verdict
        == "fail"
    )
