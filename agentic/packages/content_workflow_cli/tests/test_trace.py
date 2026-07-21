# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace builder tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from content_workflow_cli import trace as trace_module
from content_workflow_cli.trace import (
    UnsafeRunArtifactError,
    append_run_text,
    build_trace,
)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_append_run_text_normalizes_directory_creation_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trace_path = run_dir / "trace" / "events.jsonl"
    original_mkdir = trace_module.os.mkdir

    def race_mkdir(
        path: str | bytes,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        original_mkdir(path, mode=mode, dir_fd=dir_fd)
        raise FileExistsError("injected concurrent directory creation")

    monkeypatch.setattr(trace_module.os, "mkdir", race_mkdir)

    with pytest.raises(
        UnsafeRunArtifactError,
        match="directory changed while being created",
    ) as error:
        append_run_text(run_dir, trace_path, "event\n")

    assert isinstance(error.value.__cause__, FileExistsError)
    assert (run_dir / "trace").is_dir()
    assert not trace_path.exists()


def test_build_trace_creates_replay_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)

    _write_json(
        run_dir / "request.json",
        {
            "workflow": "materials.assign",
            "dry_run": False,
            "workbench_url": "http://127.0.0.1:8088",
            "runner": "codex",
            "inputs": {
                "usd": "/assets/agv.usdc",
                "reference_images": ["/assets/ref.png"],
                "materials_yaml": "/materials/materials.yaml",
                "materials_usd": "/materials/materials.usd",
            },
        },
    )
    _write_json(
        run_dir / "assignments.json",
        {
            "source_usd": "/assets/agv.usdc",
            "assignments": [
                {
                    "family": "lift_frame",
                    "material_name": "Steel Painted Orange",
                    "prim_paths": ["/World/Lift"],
                }
            ],
            "per_prim_material_assignment_count": 1,
        },
    )
    _write_json(
        run_dir / "api_operation_counts.json",
        {
            "api_operation_count_total": 12,
            "render_count_total": 2,
            "pick_calls": 1,
            "material_override_commands": 1,
        },
    )
    _write_json(
        raw / "tree_summary.json",
        {"mesh_count": 1, "prim_count": 3},
    )
    _write_json(
        raw / "initial_render_records.json",
        [
            {
                "name": "initial_iso",
                "direction": "+x-y+z",
                "response": {"copied_image_path": str(run_dir / "initial.png")},
            }
        ],
    )
    _write_json(
        raw / "pick_records.json",
        [
            {
                "kind": "render",
                "name": "pick_top",
                "response": {"copied_image_path": str(run_dir / "pick.png")},
            },
            {
                "kind": "pick",
                "label": "lift_sidewall",
                "payload": {"x": 100, "y": 200},
                "response": {"prim_paths": ["/World/Lift"]},
            },
        ],
    )
    _write_json(
        raw / "isolation_render_records.json",
        [
            {
                "name": "isolate_lift",
                "paths": ["/World/Lift"],
                "render": {"copied_image_path": str(run_dir / "isolate.png")},
            }
        ],
    )
    _write_json(
        raw / "final_render_records.json",
        [
            {
                "name": "final_iso",
                "direction": "+x-y+z",
                "response": {"copied_image_path": str(run_dir / "final.png")},
            }
        ],
    )
    events_path = run_dir / "trace" / "events.jsonl"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {
                    "schema_version": "content-agents.trace.v1",
                    "event_type": "render",
                    "phase": "final",
                    "summary": "Rendered final contact sheet.",
                    "artifacts": [str(run_dir / "final_contact_sheet.jpg")],
                    "data": {"api_calls": ["POST /render"]},
                },
                {
                    "schema_version": "content-agents.trace.v1",
                    "event_type": "child_agent_finished",
                    "phase": "runner",
                    "summary": "Child agent process exited.",
                    "artifacts": [str(run_dir / "child-output.log")],
                    "data": {"returncode": 1},
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "child-output.log").write_text(
        "\n".join(
            [
                "$ node codex_sdk_bridge.mjs request.json",
                "Traceback (most recent call last): scene endpoint returned Not Found",
            ]
        ),
        encoding="utf-8",
    )
    _write_json(
        raw / "codex_items.json",
        [
            {
                "id": "item_1",
                "type": "command_execution",
                "command": "/bin/bash -lc \"python - <<'PY'\nprint('glue')\nPY\"",
                "aggregated_output": "temporary script wrote final render glue\n",
                "exit_code": 0,
                "status": "completed",
            },
            {
                "id": "item_2",
                "type": "command_execution",
                "command": "apply_patch <<'PATCH'\n*** Begin Patch\nPATCH",
                "aggregated_output": "",
                "exit_code": 1,
                "status": "failed",
            },
        ],
    )
    _write_json(
        raw / "vqa_refinement_1_items.json",
        {
            "items": [
                {
                    "id": "item_refine_1",
                    "type": "command_execution",
                    "command": "python inspect_refinement.py",
                    "aggregated_output": "refinement inspected the wrong roller bars\n",
                    "exit_code": 0,
                    "status": "completed",
                }
            ]
        },
    )

    result = build_trace(run_dir)

    trace = json.loads(Path(result["operation_trace_json"]).read_text(encoding="utf-8"))
    assert trace["schema_version"] == "content-agents.trace.v1"
    assert any(event["kind"] == "pick" for event in trace["timeline"])
    assert any(event["kind"] == "assignment" for event in trace["timeline"])
    assert any(event["kind"] == "agent_render" for event in trace["timeline"])
    retrospective = trace["run_retrospective"]
    assert len(trace["child_commands"]) == 3
    assert any(event["kind"] == "child_commands" for event in trace["timeline"])
    assert any(
        command["source_artifact"].endswith("vqa_refinement_1_items.json")
        for command in trace["child_commands"]
    )
    assert retrospective["patches_or_code_changes"]["detected"] is True
    assert retrospective["generated_glue_code"]["detected"] is True
    assert any(
        "child command execution(s) exited non-zero" in item
        for item in retrospective["what_did_not_go_well"]
    )
    assert any(
        "return code 1" in item for item in retrospective["what_did_not_go_well"]
    )
    assert trace["material_coverage"]["present"] is False
    assert trace["assignments_summary"][0]["coverage_status"] == "unknown"
    assert any(
        "does not include material coverage accounting" in item
        for item in retrospective["what_did_not_go_well"]
    )
    assert any(
        "does not include final_review" in item
        for item in retrospective["what_did_not_go_well"]
    )

    manifest = json.loads(
        Path(result["replay_manifest_json"]).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "content-agents.replay.v1"
    assert any(frame["markers"] for frame in manifest["frames"])
    assert any(
        frame["image_path"] == "final_contact_sheet.jpg" for frame in manifest["frames"]
    )
    assert Path(result["run_retrospective_json"]).exists()
    trace_md = Path(result["operation_trace_md"]).read_text(encoding="utf-8")
    assert "## Run Retrospective" in trace_md
    assert "Patches or code changes" in trace_md


def test_build_trace_reports_complete_material_coverage(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)

    _write_json(
        run_dir / "request.json",
        {
            "workflow": "materials.assign",
            "dry_run": True,
            "workbench_url": "http://127.0.0.1:8088",
            "runner": "codex",
            "inputs": {
                "usd": "/assets/g1.usdc",
                "reference_images": ["/assets/ref.png"],
                "materials_yaml": "/materials/materials.yaml",
                "materials_usd": "/materials/materials.usd",
            },
        },
    )
    _write_json(
        run_dir / "assignments.json",
        {
            "source_usd": "/assets/g1.usdc",
            "coverage": {
                "candidate_visible_prim_count": 4,
                "material_decision_prim_count": 4,
                "material_assignment_prim_count": 2,
                "preserved_existing_prim_count": 2,
                "ambiguous_unassigned_prim_count": 0,
                "coverage_notes": "All visible families accounted for.",
            },
            "assignments": [
                {
                    "family": "black_head_shell",
                    "coverage_status": "material_assignment",
                    "material_name": "Plastic Black",
                    "material_path": "/World/Materials/PlasticBlack",
                    "prim_paths": ["/World/Head", "/World/Hands"],
                },
                {
                    "family": "silver_body_shell",
                    "coverage_status": "preserved_existing",
                    "material_name": "Observed Silver Paint",
                    "material_path": None,
                    "prim_paths": ["/World/Torso", "/World/Legs"],
                },
            ],
            "per_prim_material_assignment_count": 2,
            "final_review": {
                "issues_found": ["Missing preserved silver body coverage."],
                "issues_fixed": ["Added preserved silver body coverage."],
                "unresolved_issues": [],
                "review_notes": "Added preserved silver body coverage.",
            },
            "visual_quality_assessment": {
                "status": "fixed",
                "checked_views": ["/assets/final_front.png", "/assets/final_side.png"],
                "reference_images": ["/assets/ref.png"],
                "issues_found": [
                    {
                        "severity": "medium",
                        "description": "Body shell was underrepresented.",
                        "affected_prim_paths": ["/World/Torso", "/World/Legs"],
                        "evidence_artifacts": ["/assets/final_front.png"],
                        "expected_appearance": "silver shell",
                        "actual_appearance": "unaccounted shell",
                        "status": "fixed",
                    }
                ],
                "issues_fixed": ["Preserved silver shell family explicitly."],
                "unresolved_issues": [],
                "assessment_notes": "Final renders match material families.",
            },
        },
    )
    _write_json(
        run_dir / "api_operation_counts.json",
        {
            "api_operation_count_total": 20,
            "render_count_total": 3,
            "pick_calls": 2,
            "material_override_commands": 2,
            "coverage_candidate_visible_prims": 4,
            "coverage_material_decision_prims": 4,
            "final_review_issues_found": 1,
            "final_review_issues_fixed": 1,
        },
    )

    result = build_trace(run_dir)

    trace = json.loads(Path(result["operation_trace_json"]).read_text(encoding="utf-8"))
    retrospective = trace["run_retrospective"]
    assert trace["material_coverage"]["present"] is True
    assert trace["material_coverage"]["candidate_visible_prim_count"] == 4
    assert trace["assignments_summary"][0]["coverage_status"] == "material_assignment"
    assert any(
        "accounts for 4/4 canonical material candidate" in item
        for item in retrospective["what_went_well"]
    )
    assert any(
        "final material review found 1 issue(s) and fixed 1" in item.lower()
        for item in retrospective["what_went_well"]
    )
    assert any(
        "visual quality assessment checked 2 final render" in item.lower()
        for item in retrospective["what_went_well"]
    )
    assert any(
        "visual quality assessment found 1 issue(s) and fixed 1" in item.lower()
        for item in retrospective["what_went_well"]
    )
    assert not any(
        "does not include material coverage accounting" in item
        for item in retrospective["what_did_not_go_well"]
    )


def test_build_trace_extracts_claude_tool_commands(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)

    _write_json(
        run_dir / "request.json",
        {
            "workflow": "materials.assign",
            "dry_run": False,
            "workbench_url": "http://127.0.0.1:8088",
            "runner": "claude",
            "inputs": {
                "usd": "/assets/g1.usdc",
                "reference_images": ["/assets/ref.png"],
                "materials_yaml": "/materials/materials.yaml",
                "materials_usd": "/materials/materials.usd",
            },
        },
    )
    _write_json(
        run_dir / "assignments.json",
        {
            "source_usd": "/assets/g1.usdc",
            "assignments": [],
        },
    )
    _write_json(run_dir / "api_operation_counts.json", {})
    _write_json(
        raw / "claude_items.json",
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Bash",
                            "input": {"command": "python - <<'PY'\nprint('glue')\nPY"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "glue\n"}],
                        }
                    ]
                },
            },
        ],
    )

    result = build_trace(run_dir)

    trace = json.loads(Path(result["operation_trace_json"]).read_text(encoding="utf-8"))
    assert len(trace["child_commands"]) == 1
    assert trace["child_commands"][0]["command"].startswith("python - <<")
    assert trace["child_commands"][0]["output_excerpt"] == "glue"
    assert trace["child_commands"][0]["source_artifact"].endswith("claude_items.json")
    assert trace["run_retrospective"]["generated_glue_code"]["detected"] is True
