# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace builder tests."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from content_workflow_cli import trace as trace_module
from content_workflow_cli.trace import (
    TraceWriter,
    UnsafeRunArtifactError,
    append_run_text,
    build_trace,
)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.skipif(
    os.name != "posix",
    reason="exercises descriptor-relative directory creation",
)
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


@pytest.mark.skipif(
    os.name != "posix",
    reason="injects a descriptor-relative file-open failure",
)
def test_append_run_text_preserves_transient_file_open_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    trace_dir = run_dir / "trace"
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "events.jsonl"
    original_open = trace_module.os.open

    def fail_trace_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == trace_path.name and flags & os.O_WRONLY:
            raise OSError(errno.ENOSPC, "injected disk full")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(trace_module.os, "open", fail_trace_open)

    with pytest.raises(OSError) as error:
        append_run_text(run_dir, trace_path, "event\n")

    assert error.value.errno == errno.ENOSPC
    assert not isinstance(error.value, UnsafeRunArtifactError)


@pytest.mark.skipif(
    os.name != "posix",
    reason="injects POSIX O_NOFOLLOW rejection",
)
def test_append_run_text_treats_no_follow_eloop_as_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    trace_dir = run_dir / "trace"
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "events.jsonl"
    original_open = trace_module.os.open

    def fail_trace_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == trace_path.name and flags & os.O_WRONLY:
            raise OSError(errno.ELOOP, "injected no-follow rejection")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(trace_module.os, "open", fail_trace_open)

    with pytest.raises(
        UnsafeRunArtifactError, match="changed to an unsafe file"
    ) as error:
        append_run_text(run_dir, trace_path, "event\n")

    assert isinstance(error.value.__cause__, OSError)
    assert error.value.__cause__.errno == errno.ELOOP


def test_trace_writer_restores_jsonl_boundary_after_partial_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original_append = trace_module.append_run_text
    append_attempts = 0

    def fail_after_partial_append(run_dir: Path, path: Path, text: str) -> None:
        nonlocal append_attempts
        append_attempts += 1
        if append_attempts == 1:
            original_append(run_dir, path, text[:20])
            raise trace_module._RunArtifactWriteError(
                errno.ENOSPC,
                "injected disk full after partial write",
            )
        original_append(run_dir, path, text)

    monkeypatch.setattr(trace_module, "append_run_text", fail_after_partial_append)
    writer = TraceWriter(run_dir)

    writer.write("partial", phase="runner", summary="Partially written event.")
    writer.write(
        "terminal_validation",
        phase="finalize",
        summary="Terminal validation completed.",
    )

    lines = (
        (run_dir / "trace" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(lines) == 2
    with pytest.raises(json.JSONDecodeError):
        json.loads(lines[0])
    assert json.loads(lines[1])["event_type"] == "terminal_validation"


def test_timeline_projects_typed_memory_observation(tmp_path: Path) -> None:
    timeline = trace_module._timeline_from_agent_events(
        tmp_path,
        [
            {
                "schema_version": "content-agent-memory.event.v1",
                "kind": "observation",
                "phase": "material-refinement",
                "observation": {
                    "observation_id": "obs-01",
                    "interaction": {"operation": "render_material_preview"},
                    "artifacts": [{"role": "preview.rgb"}],
                    "outcome": {
                        "classification": "contradicted",
                        "summary": "Preview remained opaque.",
                    },
                },
            }
        ],
        0,
    )

    assert timeline[0]["kind"] == "agent_memory_observation"
    assert timeline[0]["summary"] == "Preview remained opaque."
    assert timeline[0]["data"] == {
        "observation_id": "obs-01",
        "operation": "render_material_preview",
        "outcome": "contradicted",
        "artifact_roles": ["preview.rgb"],
    }


def test_timeline_projects_memory_pin_without_none_summary(tmp_path: Path) -> None:
    timeline = trace_module._timeline_from_agent_events(
        tmp_path,
        [
            {
                "schema_version": "content-agent-memory.event.v1",
                "kind": "pin",
                "phase": "memory",
                "observation": {"observation_id": "obs-01"},
            }
        ],
        0,
    )

    assert timeline[0]["kind"] == "agent_memory_pin"
    assert timeline[0]["summary"] == "Memory pin event for observation obs-01."
    assert timeline[0]["data"] == {"observation_id": "obs-01"}


def test_build_trace_creates_replay_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    for name in (
        "initial.png",
        "pick.png",
        "isolate.png",
        "final.png",
        "final_contact_sheet.jpg",
    ):
        (run_dir / name).write_bytes(b"image")

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


def test_build_trace_treats_array_operation_counts_as_invalid_object(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "request.json",
        {
            "workflow": "materials.assign",
            "scene_backend": "usd-cli",
            "inputs": {"usd": "/assets/robot.usda"},
        },
    )
    _write_json(run_dir / "assignments.json", {})
    _write_json(run_dir / "api_operation_counts.json", [])

    result = build_trace(run_dir)

    assert result["trace"]["stats"] == {}
    assert result["trace"]["timeline"][-1]["kind"] == "finish"
    assert result["trace"]["timeline"][-1]["data"] == {
        "usd_cli_calls": None,
        "render_count": None,
    }


def test_build_trace_uses_usd_cli_backend_timeline_without_workbench_claims(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "final_renders").mkdir()
    (run_dir / "output").mkdir()
    (run_dir / "final_renders" / "hero.png").write_bytes(b"png")
    (run_dir / "output" / "materialized.usda").write_text(
        "#usda 1.0\n",
        encoding="utf-8",
    )
    _write_json(
        run_dir / "request.json",
        {
            "workflow": "materials.assign",
            "scene_backend": "usd-cli",
            "runner": "codex",
            "inputs": {
                "usd": "/assets/robot.usda",
                "reference_images": ["/assets/reference.png"],
            },
            "telemetry": {
                "trace_id": "ab" * 16,
                "root_span_id": "12" * 8,
            },
        },
    )
    _write_json(
        run_dir / "assignments.json",
        {
            "source_usd": "/assets/robot.usda",
            "assignments": [
                {
                    "family": "body",
                    "coverage_status": "material_assignment",
                    "material_name": "Paint White",
                    "prim_paths": ["/World/Body"],
                }
            ],
        },
    )
    _write_json(
        run_dir / "api_operation_counts.json",
        {"usd_cli_calls_total": 3, "render_calls_total": 1},
    )
    events_path = run_dir / "trace" / "events.jsonl"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": "content-agents.trace.v1",
                        "time": "2026-07-24T10:00:00+00:00",
                        "event_type": "usd_cli_invocation",
                        "phase": "usd-cli",
                        "summary": "usd-cli render completed.",
                        "artifacts": [str(run_dir / "final_renders" / "hero.png")],
                        "data": {
                            "schema_version": (
                                "content-agents.usd-cli-telemetry-import.v1"
                            ),
                            "trace_id": "ab" * 16,
                            "command": "render",
                            "start_time_unix_nano": 1,
                            "exit_code": 0,
                            "trace_matches_run": True,
                            "parent_matches_run_root": True,
                        },
                    }
                ),
                json.dumps(
                    {
                        "schema_version": "content-agents.trace.v1",
                        "time": "2026-07-24T10:00:01+00:00",
                        "event_type": "usd_cli_telemetry_ingested",
                        "phase": "usd-cli telemetry",
                        "summary": "Imported one span.",
                        "artifacts": [],
                        "data": {
                            "schema_version": (
                                "content-agents.usd-cli-telemetry-import.v1"
                            ),
                            "expected_trace_id": "ab" * 16,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_trace(run_dir)

    trace = json.loads(Path(result["operation_trace_json"]).read_text(encoding="utf-8"))
    serialized_timeline = json.dumps(trace["timeline"])
    assert "Workbench API discovered" not in serialized_timeline
    assert "Content Workbench" not in serialized_timeline
    assert any(
        event["kind"] == "agent_usd_cli_invocation" for event in trace["timeline"]
    )
    assert any(
        event["kind"] == "agent_usd_cli_telemetry_ingested"
        for event in trace["timeline"]
    )
    assert any(event["kind"] == "persistence" for event in trace["timeline"])
    kinds = [event["kind"] for event in trace["timeline"]]
    assert kinds.index("agent_usd_cli_invocation") < kinds.index("assignment")
    assert kinds.index("agent_usd_cli_invocation") < kinds.index("persistence")
    assert kinds[-1] == "finish"
    invocation = next(
        event
        for event in trace["timeline"]
        if event["kind"] == "agent_usd_cli_invocation"
    )
    assert invocation["time"] == "2026-07-24T10:00:00+00:00"
    markdown = Path(result["operation_trace_md"]).read_text(encoding="utf-8")
    assert "usd-cli telemetry" in markdown
    assert "Content Workbench artifacts" not in markdown


def test_usd_cli_trace_does_not_claim_missing_telemetry(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    _write_json(
        run_dir / "request.json",
        {
            "workflow": "materials.assign",
            "scene_backend": "usd-cli",
            "runner": "codex",
            "inputs": {"usd": "/assets/robot.usda"},
            "telemetry": {
                "trace_id": "ab" * 16,
                "root_span_id": "12" * 8,
                "routing": {
                    "status": "wrapper_unavailable",
                    "enabled": False,
                },
            },
        },
    )

    result = build_trace(run_dir)

    trace = json.loads(Path(result["operation_trace_json"]).read_text(encoding="utf-8"))
    assert trace["timeline"][-1]["kind"] == "finish"
    assert "usd-cli telemetry" not in trace["timeline"][-1]["summary"]
    markdown = Path(result["operation_trace_md"]).read_text(encoding="utf-8")
    assert "reconstructed from usd-cli telemetry" not in markdown
    assert (
        "reconstructed from child-agent artifacts and wrapper verification metadata"
        in markdown
    )


def test_usd_cli_trace_filters_events_from_reused_output_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    _write_json(
        run_dir / "request.json",
        {
            "workflow": "materials.assign",
            "scene_backend": "usd-cli",
            "created_at": "2026-07-24T10:00:00+00:00",
            "telemetry": {
                "trace_id": "ab" * 16,
                "root_span_id": "12" * 8,
            },
        },
    )
    events_path = run_dir / "trace" / "events.jsonl"
    events_path.parent.mkdir()
    events_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "time": "2026-07-23T10:00:00+00:00",
                        "event_type": "usd_cli_invocation",
                        "phase": "usd-cli",
                        "summary": "stale",
                        "data": {
                            "schema_version": (
                                "content-agents.usd-cli-telemetry-import.v1"
                            ),
                            "trace_id": "cd" * 16,
                            "trace_matches_run": True,
                            "parent_matches_run_root": True,
                        },
                    }
                ),
                json.dumps(
                    {
                        "time": "2026-07-24T10:00:01+00:00",
                        "event_type": "usd_cli_ready",
                        "phase": "setup",
                        "summary": "current",
                        "data": {},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_trace(run_dir)

    trace = json.loads(Path(result["operation_trace_json"]).read_text(encoding="utf-8"))
    assert [event["summary"] for event in trace["agent_events"]] == ["current"]
    markdown = Path(result["operation_trace_md"]).read_text(encoding="utf-8")
    assert "reconstructed from usd-cli telemetry" not in markdown


def test_trace_and_replay_reject_escaping_final_render_symlink(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    final_dir = run_dir / "final_renders"
    final_dir.mkdir(parents=True)
    (final_dir / "valid.png").write_bytes(b"png")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"secret")
    (final_dir / "escaping.png").symlink_to(outside)
    _write_json(
        run_dir / "request.json",
        {
            "workflow": "materials.assign",
            "scene_backend": "usd-cli",
            "inputs": {"usd": "/assets/robot.usda"},
        },
    )
    _write_json(run_dir / "assignments.json", {})
    _write_json(run_dir / "api_operation_counts.json", {})

    result = build_trace(run_dir)
    rendered = json.dumps(
        {
            "trace": result["trace"],
            "replay_manifest": result["replay_manifest"],
        }
    )

    assert "final_renders/valid.png" in rendered
    assert "escaping.png" not in rendered
    assert str(outside) not in rendered


def test_trace_writer_refuses_symlinked_event_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    trace_dir = run_dir / "trace"
    trace_dir.mkdir(parents=True)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("unchanged\n", encoding="utf-8")
    (trace_dir / "events.jsonl").symlink_to(outside)

    with pytest.raises(UnsafeRunArtifactError):
        TraceWriter(run_dir).write(
            "warning",
            phase="test",
            summary="must not escape",
        )

    assert outside.read_text(encoding="utf-8") == "unchanged\n"


def test_trace_writer_keeps_resolved_root_when_alias_is_retargeted(
    tmp_path: Path,
) -> None:
    real_run = tmp_path / "real-run"
    real_run.mkdir()
    run_alias = tmp_path / "run"
    run_alias.symlink_to(real_run, target_is_directory=True)
    writer = TraceWriter(run_alias)

    outside = tmp_path / "outside"
    outside.mkdir()
    run_alias.unlink()
    run_alias.symlink_to(outside, target_is_directory=True)

    writer.write(
        "warning",
        phase="test",
        summary="must stay in the original run",
    )

    events_path = real_run / "trace" / "events.jsonl"
    assert "must stay in the original run" in events_path.read_text(encoding="utf-8")
    assert not (outside / "trace").exists()


def test_build_trace_bounds_child_authored_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trace_module, "_MAX_JSON_ARTIFACT_BYTES", 256)
    monkeypatch.setattr(trace_module, "_MAX_JSONL_ARTIFACT_BYTES", 256)
    monkeypatch.setattr(trace_module, "_MAX_TEXT_ARTIFACT_BYTES", 64)

    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "request.json",
        {
            "workflow": "materials.assign",
            "scene_backend": "usd-cli",
        },
    )
    _write_json(
        run_dir / "assignments.json",
        {"SECRET_OVERSIZED_ASSIGNMENT": "x" * 512},
    )
    _write_json(run_dir / "api_operation_counts.json", {})
    events_path = run_dir / "trace" / "events.jsonl"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        json.dumps(
            {
                "event_type": "warning",
                "summary": "SECRET_OVERSIZED_EVENT",
                "padding": "x" * 512,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "child-output.log").write_text(
        "apply_patch SECRET_OVERSIZED_CHILD_OUTPUT " + "x" * 512,
        encoding="utf-8",
    )

    result = build_trace(run_dir)
    serialized = json.dumps(result)

    assert result["trace"]["request"]["workflow"] == "materials.assign"
    assert result["trace"]["assignments_summary"] == []
    assert result["trace"]["agent_events"] == []
    assert (
        result["trace"]["run_retrospective"]["patches_or_code_changes"]["detected"]
        is False
    )
    assert "SECRET_OVERSIZED_ASSIGNMENT" not in serialized
    assert "SECRET_OVERSIZED_EVENT" not in serialized
    assert "SECRET_OVERSIZED_CHILD_OUTPUT" not in serialized


def test_build_trace_enforces_aggregate_child_input_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "request.json",
        {
            "workflow": "materials.assign",
            "scene_backend": "usd-cli",
        },
    )
    _write_json(
        run_dir / "assignments.json",
        {
            "assignments": [
                {
                    "family": "SECRET_OVER_BUDGET_ASSIGNMENT",
                    "material_name": "paint",
                    "prim_paths": ["/World/Body"],
                }
            ]
        },
    )
    _write_json(run_dir / "api_operation_counts.json", {})
    request_size = (run_dir / "request.json").stat().st_size
    assignments_size = (run_dir / "assignments.json").stat().st_size
    monkeypatch.setattr(
        trace_module,
        "_MAX_TOTAL_TRACE_INPUT_BYTES",
        request_size + assignments_size - 1,
    )

    result = build_trace(run_dir)
    serialized = json.dumps(result)

    assert result["trace"]["request"]["workflow"] == "materials.assign"
    assert result["trace"]["assignments_summary"] == []
    assert "SECRET_OVER_BUDGET_ASSIGNMENT" not in serialized


def test_build_trace_ignores_child_symlinked_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside_request = tmp_path / "outside-request.json"
    outside_request.write_text(
        json.dumps(
            {
                "workflow": "SECRET_EXTERNAL_WORKFLOW",
                "scene_backend": "usd-cli",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "request.json").symlink_to(outside_request)

    outside_raw = tmp_path / "outside-raw"
    outside_raw.mkdir()
    _write_json(
        outside_raw / "codex_items.json",
        [
            {
                "type": "command_execution",
                "command": "SECRET_EXTERNAL_COMMAND",
            }
        ],
    )
    (run_dir / "raw").symlink_to(outside_raw, target_is_directory=True)
    _write_json(run_dir / "assignments.json", {})
    _write_json(run_dir / "api_operation_counts.json", {})

    result = build_trace(run_dir)

    serialized = json.dumps(result)
    assert result["trace"]["request"] == {}
    assert result["trace"]["child_commands"] == []
    assert "SECRET_EXTERNAL_WORKFLOW" not in serialized
    assert "SECRET_EXTERNAL_COMMAND" not in serialized


def test_build_trace_refuses_symlinked_fixed_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "request.json",
        {
            "workflow": "materials.assign",
            "scene_backend": "usd-cli",
        },
    )
    _write_json(run_dir / "assignments.json", {})
    _write_json(run_dir / "api_operation_counts.json", {})
    trace_dir = run_dir / "trace"
    trace_dir.mkdir()
    outside = tmp_path / "outside-trace.json"
    outside.write_text('{"sentinel":"unchanged"}\n', encoding="utf-8")
    (trace_dir / "operation_trace.json").symlink_to(outside)

    with pytest.raises(ValueError, match="regular file"):
        build_trace(run_dir)

    assert outside.read_text(encoding="utf-8") == '{"sentinel":"unchanged"}\n'
