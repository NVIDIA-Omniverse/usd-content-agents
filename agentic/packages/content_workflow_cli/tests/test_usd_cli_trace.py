# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for importing usd-cli-tel spans into TraceWriter events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from content_workflow_cli import usd_cli_trace
from content_workflow_cli.trace import TraceWriter
from content_workflow_cli.usd_cli_trace import import_usd_cli_telemetry

TRACE_ID = "ab" * 16
ROOT_SPAN_ID = "12" * 8


def _prepare_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run-001"
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "request.json").write_text(
        json.dumps(
            {
                "scene_backend": "usd-cli",
                "telemetry": {
                    "trace_id": TRACE_ID,
                    "root_span_id": ROOT_SPAN_ID,
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _span(
    run_dir: Path,
    *,
    span_id: str,
    argv: list[str],
    start: int,
    exit_code: int = 0,
    trace_id: str = TRACE_ID,
    parent_span_id: str = ROOT_SPAN_ID,
    run_name: str | None = None,
    status_code: str | None = None,
    extra_attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command = next((token for token in argv if not token.startswith("-")), "<none>")
    attributes: dict[str, Any] = {
        "process.command_args": argv,
        "process.exit_code": exit_code,
        "process.executable.path": "/opt/usd-cli/bin/usd-cli",
        "usd.command": command,
        "usd.session": "material-run",
        "telemetry.sdk.name": "usd-cli-tel",
        "telemetry.sdk.version": "0.1.0",
        "telemetry.schema_version": 1,
        "wu.run_id": run_name or TRACE_ID,
    }
    attributes.update(extra_attributes or {})
    end = start + 25_000_000
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": f"usd-cli.{command}",
        "kind": "SPAN_KIND_CLIENT",
        "start_time_unix_nano": start,
        "end_time_unix_nano": end,
        "duration_ms": 25.0,
        "status": {
            "code": status_code or ("OK" if exit_code == 0 else "ERROR"),
            "message": "sensitive stderr must not be copied",
        },
        "attributes": attributes,
    }


def _events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "trace" / "events.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_imports_rotated_spans_in_timestamp_order_and_cross_checks(
    tmp_path: Path,
) -> None:
    run_dir = _prepare_run(tmp_path)
    render = run_dir / "final_renders" / "hero.png"
    output = run_dir / "output" / "materialized.usda"
    render.parent.mkdir()
    output.parent.mkdir()
    render.write_bytes(b"png")
    output.write_text("#usda 1.0\n", encoding="utf-8")
    (run_dir / "api_operation_counts.json").write_text(
        json.dumps({"usd_cli_calls_total": 2, "render_calls_total": 1}),
        encoding="utf-8",
    )
    (run_dir / "run_cost_metrics.json").write_text(
        json.dumps({"shell_commands_total": 4, "failed_commands_total": 0}),
        encoding="utf-8",
    )

    telemetry = run_dir / "raw" / "usd_cli_telemetry.jsonl"
    rotated = telemetry.with_name(telemetry.name + ".1")
    later = _span(
        run_dir,
        span_id="22" * 8,
        argv=["save", str(output)],
        start=2_000_000_000,
        extra_attributes={
            "usd.scene.path": str(output),
            "usd.scene.size_bytes": output.stat().st_size,
        },
    )
    earlier = _span(
        run_dir,
        span_id="11" * 8,
        argv=[
            "--session",
            "material-run",
            "--json",
            "render",
            "--output",
            str(render),
        ],
        start=1_000_000_000,
    )
    rotated.write_text(
        json.dumps(later) + "\n" + "{truncated\n",
        encoding="utf-8",
    )
    telemetry.write_text(
        json.dumps(earlier) + "\n" + json.dumps(later) + "\n",
        encoding="utf-8",
    )

    result = import_usd_cli_telemetry(
        run_dir,
        TraceWriter(run_dir),
        usd_cli_source_revision="f" * 40,
    )

    assert result["valid_spans"] == 2
    assert result["imported_spans"] == 2
    assert result["invalid_records"] == 1
    assert result["duplicate_spans"] == 1
    assert result["cross_checks"]["api_operation_counts"]["call_count_matches"] is True
    assert (
        result["cross_checks"]["api_operation_counts"]["render_count_matches"] is True
    )
    assert result["cross_checks"]["api_operation_counts"]["load_status"] == "parsed"
    assert result["cross_checks"]["api_operation_counts"]["load_reason"] is None
    assert result["cross_checks"]["run_cost_metrics"]["load_status"] == "parsed"
    assert result["cross_checks"]["run_cost_metrics"]["load_reason"] is None

    events = _events(run_dir)
    invocations = [
        event for event in events if event["event_type"] == "usd_cli_invocation"
    ]
    assert [event["data"]["command"] for event in invocations] == ["render", "save"]
    assert invocations[0]["time"] == "1970-01-01T00:00:01+00:00"
    assert invocations[0]["artifacts"] == [str(render)]
    assert invocations[1]["artifacts"] == [str(output)]
    assert invocations[1]["data"]["scene"] == {
        "path": str(output),
        "size_bytes": output.stat().st_size,
    }
    assert invocations[0]["data"]["argv_mapping"] == {
        "schema_version": "usd-cli.argv-semantics.v2",
        "source_revision": "f" * 40,
    }
    assert "status_message" not in invocations[0]["data"]
    summary = events[-1]
    assert summary["event_type"] == "warning"
    assert summary["data"]["warning_code"] == "usd_cli_telemetry_records_rejected"
    assert summary["data"]["command_counts"] == {"render": 1, "save": 1}


def test_redacts_url_secrets_from_argv_scene_and_artifact_arguments(
    tmp_path: Path,
) -> None:
    run_dir = _prepare_run(tmp_path)
    scene_url = (
        "https://scene-user:scene-pass@example.test/scene.usd"
        "?X-Amz-Credential=SCENE-SECRET&X-Amz-Signature=SCENE-SIGNATURE"
    )
    output_url = (
        "https://output-user:output-pass@example.test/out.png"
        "?token=OUTPUT-SECRET#fragment"
    )
    telemetry = run_dir / "raw" / "usd_cli_telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            _span(
                run_dir,
                span_id="13" * 8,
                argv=["usd-cli", "render", f"--output={output_url}"],
                start=1,
                extra_attributes={
                    "usd.scene.path": scene_url,
                    "usd.scene.paths": [
                        scene_url,
                        "file:///tmp/scene.usda?token=LOCAL",
                    ],
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    import_usd_cli_telemetry(run_dir, TraceWriter(run_dir))

    invocation = next(
        event
        for event in _events(run_dir)
        if event["event_type"] == "usd_cli_invocation"
    )
    assert invocation["data"]["argv"] == [
        "usd-cli",
        "render",
        "--output=https://example.test/out.png",
    ]
    assert invocation["data"]["scene"] == {
        "path": "https://example.test/scene.usd",
        "paths": [
            "https://example.test/scene.usd",
            "file:///tmp/scene.usda",
        ],
    }
    assert invocation["data"]["artifact_checks"][0]["argument"] == (
        "https://example.test/out.png"
    )
    serialized = json.dumps(invocation)
    for secret in (
        "scene-user",
        "scene-pass",
        "SCENE-SECRET",
        "SCENE-SIGNATURE",
        "output-user",
        "output-pass",
        "OUTPUT-SECRET",
        "fragment",
        "LOCAL",
    ):
        assert secret not in serialized


def test_telemetry_parser_uses_bytes_captured_from_same_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _prepare_run(tmp_path)
    telemetry = run_dir / "raw" / "usd_cli_telemetry.jsonl"
    original_span = _span(
        run_dir,
        span_id="14" * 8,
        argv=["snapshot"],
        start=1,
    )
    telemetry.write_text(json.dumps(original_span) + "\n", encoding="utf-8")
    original_loader = usd_cli_trace._load_span_records

    def replace_path_after_capture(
        artifacts: list[Any],
    ) -> tuple[list[dict[str, Any]], int]:
        telemetry.write_text("{malformed replacement\n", encoding="utf-8")
        return original_loader(artifacts)

    monkeypatch.setattr(
        usd_cli_trace,
        "_load_span_records",
        replace_path_after_capture,
    )

    result = import_usd_cli_telemetry(run_dir, TraceWriter(run_dir))

    assert result["valid_spans"] == 1
    assert result["invalid_records"] == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["usd-cli", "--session", "material-run", "render"],
        ["/opt/usd-cli/bin/usd-cli", "--session", "material-run", "render"],
        ["usd-cli-tel", "--session", "material-run", "render"],
        ["/usr/bin/python3", "-m", "usd_cli", "--session", "material-run", "render"],
    ],
)
def test_import_accepts_executable_prefixed_process_command_args(
    tmp_path: Path,
    argv: list[str],
) -> None:
    run_dir = _prepare_run(tmp_path)
    output = run_dir / "final_renders" / "prefixed.png"
    output.parent.mkdir()
    output.write_bytes(b"png")
    telemetry = run_dir / "raw" / "usd_cli_telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            _span(
                run_dir,
                span_id="19" * 8,
                argv=[*argv, "--output", str(output)],
                start=1,
                extra_attributes={"usd.command": "render"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    import_usd_cli_telemetry(run_dir, TraceWriter(run_dir))

    invocation = next(
        event
        for event in _events(run_dir)
        if event["event_type"] == "usd_cli_invocation"
    )
    assert invocation["data"]["command"] == "render"
    assert invocation["data"]["operation"] == "render"
    assert invocation["artifacts"] == [str(output)]


def test_missing_file_is_a_clean_noop(tmp_path: Path) -> None:
    run_dir = _prepare_run(tmp_path)

    result = import_usd_cli_telemetry(run_dir, TraceWriter(run_dir))

    assert result["source_files"] == []
    assert result["imported_spans"] == 0
    assert _events(run_dir) == []


def test_rejects_unrelated_or_unsupported_spans_and_external_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = _prepare_run(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")
    telemetry = run_dir / "raw" / "usd_cli_telemetry.jsonl"
    wrong_trace = _span(
        run_dir,
        span_id="21" * 8,
        argv=["render", "-o", str(outside)],
        start=1,
        trace_id="cd" * 16,
    )
    wrong_parent = _span(
        run_dir,
        span_id="22" * 8,
        argv=["snapshot"],
        start=2,
        parent_span_id="34" * 8,
    )
    wrong_run = _span(
        run_dir,
        span_id="23" * 8,
        argv=["snapshot"],
        start=3,
        run_name="another-run",
    )
    wrong_schema = _span(
        run_dir,
        span_id="24" * 8,
        argv=["snapshot"],
        start=4,
        extra_attributes={"telemetry.schema_version": 2},
    )
    accepted = _span(
        run_dir,
        span_id="25" * 8,
        argv=["render", "-o", str(outside)],
        start=5,
        exit_code=7,
        status_code="OK",
    )
    telemetry.write_text(
        "\n".join(
            json.dumps(span)
            for span in [
                wrong_trace,
                wrong_parent,
                wrong_run,
                wrong_schema,
                accepted,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = import_usd_cli_telemetry(run_dir, TraceWriter(run_dir))

    assert result["valid_spans"] == 1
    assert result["wrong_trace_spans"] == 1
    assert result["wrong_parent_spans"] == 1
    assert result["wrong_run_spans"] == 1
    assert result["unsupported_schema_spans"] == 1
    summary = _events(run_dir)[-1]
    assert summary["event_type"] == "warning"
    assert summary["data"]["warning_code"] == "usd_cli_telemetry_records_rejected"
    assert "wrong_trace_spans=1" in summary["summary"]
    invocation = next(
        event
        for event in _events(run_dir)
        if event["event_type"] == "usd_cli_invocation"
    )
    assert invocation["artifacts"] == []
    assert invocation["data"]["exit_code"] == 7
    assert invocation["data"]["status_exit_code_consistent"] is False
    assert invocation["data"]["artifact_checks"] == [
        {
            "argument": str(outside),
            "resolved_path": None,
            "within_run_dir": False,
            "exists": False,
            "accepted": False,
        }
    ]


def test_import_is_idempotent_and_understands_nested_commands(tmp_path: Path) -> None:
    run_dir = _prepare_run(tmp_path)
    telemetry = run_dir / "raw" / "usd_cli_telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            _span(
                run_dir,
                span_id="31" * 8,
                argv=["--timeout", "60", "checkpoint", "save", "before-bind"],
                start=10,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    writer = TraceWriter(run_dir)

    first = import_usd_cli_telemetry(run_dir, writer)
    second = import_usd_cli_telemetry(run_dir, writer)

    assert first["imported_spans"] == 1
    assert second["imported_spans"] == 0
    assert second["already_imported_spans"] == 1
    events = _events(run_dir)
    assert sum(event["event_type"] == "usd_cli_invocation" for event in events) == 1
    assert (
        sum(event["event_type"] == "usd_cli_telemetry_ingested" for event in events)
        == 1
    )
    invocation = next(
        event for event in events if event["event_type"] == "usd_cli_invocation"
    )
    assert invocation["data"]["command"] == "checkpoint.save"
    assert invocation["data"]["operation"] == "history"

    (run_dir / "api_operation_counts.json").write_text(
        json.dumps({"usd_cli_calls_total": 7}),
        encoding="utf-8",
    )
    third = import_usd_cli_telemetry(run_dir, writer)
    assert third["imported_spans"] == 0
    events = _events(run_dir)
    assert sum(event["event_type"] == "usd_cli_invocation" for event in events) == 1
    summaries = [
        event
        for event in events
        if event["data"].get("schema_version")
        == "content-agents.usd-cli-telemetry-import.v1"
        and event["event_type"] != "usd_cli_invocation"
    ]
    assert len(summaries) == 2
    assert summaries[-1]["data"]["warning_code"] == "usd_cli_telemetry_count_mismatch"


def test_rejects_run_relative_traversal_and_escaping_symlink(tmp_path: Path) -> None:
    run_dir = _prepare_run(tmp_path)
    outside = tmp_path / "outside.usda"
    outside.write_text("#usda 1.0\n", encoding="utf-8")
    escaping_link = run_dir / "escaping.usda"
    escaping_link.symlink_to(outside)
    telemetry = run_dir / "raw" / "usd_cli_telemetry.jsonl"
    telemetry.write_text(
        "\n".join(
            [
                json.dumps(
                    _span(
                        run_dir,
                        span_id="41" * 8,
                        argv=["save", "../outside.usda"],
                        start=1,
                    )
                ),
                json.dumps(
                    _span(
                        run_dir,
                        span_id="42" * 8,
                        argv=["save", "escaping.usda"],
                        start=2,
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    import_usd_cli_telemetry(run_dir, TraceWriter(run_dir))

    invocations = [
        event
        for event in _events(run_dir)
        if event["event_type"] == "usd_cli_invocation"
    ]
    assert len(invocations) == 2
    assert all(event["artifacts"] == [] for event in invocations)


def test_rejects_telemetry_through_symlinked_raw_directory(tmp_path: Path) -> None:
    run_dir = _prepare_run(tmp_path)
    raw_dir = run_dir / "raw"
    raw_dir.rmdir()
    outside_raw = tmp_path / "outside-raw"
    outside_raw.mkdir()
    (outside_raw / "usd_cli_telemetry.jsonl").write_text(
        json.dumps(
            _span(
                run_dir,
                span_id="43" * 8,
                argv=["snapshot"],
                start=1,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    raw_dir.symlink_to(outside_raw, target_is_directory=True)

    result = import_usd_cli_telemetry(run_dir, TraceWriter(run_dir))

    assert result["source_files"] == []
    assert result["records_seen"] == 0
    assert result["imported_spans"] == 0
    assert _events(run_dir) == []


def test_missing_request_trace_context_refuses_import(tmp_path: Path) -> None:
    run_dir = _prepare_run(tmp_path)
    (run_dir / "request.json").write_text("{}", encoding="utf-8")
    telemetry = run_dir / "raw" / "usd_cli_telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            _span(
                run_dir,
                span_id="51" * 8,
                argv=["snapshot"],
                start=1,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = import_usd_cli_telemetry(run_dir, TraceWriter(run_dir))

    assert result["imported_spans"] == 0
    assert result["correlation_error"]
    events = _events(run_dir)
    assert [event["event_type"] for event in events] == ["warning"]
    assert (
        events[0]["data"]["warning_code"] == "usd_cli_telemetry_correlation_unavailable"
    )


def test_count_mismatch_is_an_observable_warning(tmp_path: Path) -> None:
    run_dir = _prepare_run(tmp_path)
    (run_dir / "api_operation_counts.json").write_text(
        json.dumps({"usd_cli_calls_total": 9, "render_calls_total": 4}),
        encoding="utf-8",
    )
    telemetry = run_dir / "raw" / "usd_cli_telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            _span(
                run_dir,
                span_id="52" * 8,
                argv=["snapshot"],
                start=1,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = import_usd_cli_telemetry(run_dir, TraceWriter(run_dir))

    assert result["imported_spans"] == 1
    summary = _events(run_dir)[-1]
    assert summary["event_type"] == "warning"
    assert summary["data"]["warning_code"] == "usd_cli_telemetry_count_mismatch"
    assert summary["data"]["cross_check_mismatches"] == [
        "api_operation_counts.call_count_matches",
        "api_operation_counts.render_count_matches",
    ]
    assert "Count cross-check mismatch" in summary["summary"]


def test_missing_cross_check_inputs_do_not_warn(tmp_path: Path) -> None:
    run_dir = _prepare_run(tmp_path)
    telemetry = run_dir / "raw" / "usd_cli_telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            _span(
                run_dir,
                span_id="54" * 8,
                argv=["snapshot"],
                start=1,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = import_usd_cli_telemetry(run_dir, TraceWriter(run_dir))

    assert result["cross_check_load_failures"] == []
    for source_name in ("api_operation_counts", "run_cost_metrics"):
        cross_check = result["cross_checks"][source_name]
        assert cross_check["present"] is False
        assert cross_check["load_status"] == "missing"
        assert cross_check["load_reason"] is None
    summary = _events(run_dir)[-1]
    assert summary["event_type"] == "usd_cli_telemetry_ingested"
    assert summary["data"]["cross_check_load_failures"] == []


@pytest.mark.parametrize(
    ("source_name", "filename"),
    [
        ("api_operation_counts", "api_operation_counts.json"),
        ("run_cost_metrics", "run_cost_metrics.json"),
    ],
)
@pytest.mark.parametrize(
    "load_status",
    ["malformed", "symlink", "oversize", "non_object"],
)
def test_unusable_cross_check_input_is_an_observable_warning(
    tmp_path: Path,
    source_name: str,
    filename: str,
    load_status: str,
) -> None:
    run_dir = _prepare_run(tmp_path)
    input_path = run_dir / filename
    if load_status == "malformed":
        input_path.write_text("{not-json", encoding="utf-8")
    elif load_status == "symlink":
        target = tmp_path / f"{source_name}-target.json"
        target.write_text("{}", encoding="utf-8")
        input_path.symlink_to(target)
    elif load_status == "oversize":
        with input_path.open("wb") as stream:
            stream.truncate(16 * 1024 * 1024 + 1)
    else:
        input_path.write_text("[]", encoding="utf-8")
    telemetry = run_dir / "raw" / "usd_cli_telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            _span(
                run_dir,
                span_id="55" * 8,
                argv=["snapshot"],
                start=1,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = import_usd_cli_telemetry(run_dir, TraceWriter(run_dir))

    cross_check = result["cross_checks"][source_name]
    assert cross_check["present"] is True
    assert cross_check["load_status"] == load_status
    assert isinstance(cross_check["load_reason"], str)
    failure = f"{source_name}.{load_status}"
    assert result["cross_check_load_failures"] == [failure]
    summary = _events(run_dir)[-1]
    assert summary["event_type"] == "warning"
    assert (
        summary["data"]["warning_code"]
        == "usd_cli_telemetry_cross_check_input_unusable"
    )
    assert summary["data"]["cross_check_load_failures"] == [failure]
    assert failure in summary["summary"]


def test_rejects_relative_output_without_recorded_process_cwd(tmp_path: Path) -> None:
    run_dir = _prepare_run(tmp_path)
    output = run_dir / "final_renders" / "relative.png"
    output.parent.mkdir()
    output.write_bytes(b"png")
    telemetry = run_dir / "raw" / "usd_cli_telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            _span(
                run_dir,
                span_id="53" * 8,
                argv=["render", "--output", "final_renders/relative.png"],
                start=1,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    import_usd_cli_telemetry(run_dir, TraceWriter(run_dir))

    invocation = next(
        event
        for event in _events(run_dir)
        if event["event_type"] == "usd_cli_invocation"
    )
    assert invocation["artifacts"] == []
    assert invocation["data"]["artifact_checks"][0]["accepted"] is False


def test_argv_mapping_handles_hoisted_flags_and_backend_specific_reads(
    tmp_path: Path,
) -> None:
    run_dir = _prepare_run(tmp_path)
    telemetry = run_dir / "raw" / "usd_cli_telemetry.jsonl"
    telemetry.write_text(
        "\n".join(
            json.dumps(span)
            for span in (
                _span(
                    run_dir,
                    span_id="61" * 8,
                    argv=["material", "--under", "/World", "--json", "audit"],
                    start=1,
                ),
                _span(
                    run_dir,
                    span_id="62" * 8,
                    argv=["camera", "-q", "list"],
                    start=2,
                ),
                _span(
                    run_dir,
                    span_id="63" * 8,
                    argv=["sublayers", "--drop-dead"],
                    start=3,
                ),
                _span(
                    run_dir,
                    span_id="64" * 8,
                    argv=["raycast", "--origin", "0,0,0"],
                    start=4,
                ),
                _span(
                    run_dir,
                    span_id="65" * 8,
                    argv=["appearance", "audit"],
                    start=5,
                ),
                _span(
                    run_dir,
                    span_id="66" * 8,
                    argv=["appearance", "clear"],
                    start=6,
                ),
                _span(
                    run_dir,
                    span_id="67" * 8,
                    argv=["render-probe", "--require-engine", "ovrtx"],
                    start=7,
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    import_usd_cli_telemetry(run_dir, TraceWriter(run_dir))

    invocations = [
        event
        for event in _events(run_dir)
        if event["event_type"] == "usd_cli_invocation"
    ]
    assert [
        (event["data"]["command"], event["data"]["operation"]) for event in invocations
    ] == [
        ("material.audit", "scene_read"),
        ("camera.list", "scene_read"),
        ("sublayers", "scene_edit"),
        ("raycast", "scene_read"),
        ("appearance.audit", "scene_read"),
        ("appearance.clear", "scene_edit"),
        ("render-probe", "validation"),
    ]
