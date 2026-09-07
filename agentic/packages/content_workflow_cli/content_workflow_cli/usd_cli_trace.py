# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import ``usd-cli-tel`` JSONL spans into the workflow trace.

The telemetry wrapper deliberately knows nothing about World Understanding
workflows.  This module owns the versioned argv interpretation at that boundary.
Because the child can write the run-local stream and knows its correlation IDs,
imported spans are explicitly child-reported, non-authoritative observations.
Only independently validated workflow artifacts can affect acceptance.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit, urlunsplit

from content_agent_workflows.common.artifacts import (
    ContainedArtifactRead,
    contained_regular_file,
    read_contained_artifact,
)

from .trace import TraceWriter

USD_CLI_TELEMETRY_IMPORT_SCHEMA_VERSION = "content-agents.usd-cli-telemetry-import.v1"
USD_CLI_ARGV_MAPPING_VERSION = "usd-cli.argv-semantics.v2"
MAX_TELEMETRY_RECORDS = 100_000
MAX_TELEMETRY_LINE_CHARS = 1_048_576
MAX_TELEMETRY_FILE_BYTES = 64 * 1024 * 1024
MAX_TELEMETRY_TOTAL_BYTES = 128 * 1024 * 1024
MAX_AUXILIARY_JSON_BYTES = 16 * 1024 * 1024

_GLOBAL_VALUE_OPTIONS = {"--server", "--session", "--timeout"}
_SENSITIVE_OPTION_NAMES = {
    "--api-key",
    "--authorization",
    "--credential",
    "--header",
    "--password",
    "--secret",
    "--token",
}
_INLINE_SECRET_RE = re.compile(
    r"(?i)^(?P<name>[^=]*(?:api[_-]?key|authorization|credential|password|secret|token)[^=]*)=(?P<value>.*)$"
)
_USD_CLI_EXECUTABLE_NAMES = {
    "3dsc",
    "3dsc.exe",
    "ov",
    "ov.exe",
    "usd-cli",
    "usd-cli.exe",
    "usd-cli-tel",
    "usd-cli-tel.exe",
}
_USD_CLI_PYTHON_MODULES = {
    "usd_cli",
    "usd_cli.__main__",
    "usd_cli.main",
    "usd_telemetry",
    "usd_telemetry.__main__",
    "usd_telemetry.main",
}
_MATERIAL_VALUE_OPTIONS = {
    "--bind",
    "--clearcoat",
    "--clearcoat-roughness",
    "--color",
    "--diffuse-tex",
    "--emissive",
    "--input",
    "--ior",
    "--library",
    "--mdl",
    "--mdl-id",
    "--metallic",
    "--metallic-tex",
    "--name",
    "--normal-tex",
    "--opacity",
    "--orm-tex",
    "--roughness",
    "--roughness-tex",
    "--subset",
    "--tex-rotate",
    "--tex-scale",
    "--tex-translate",
    "--type",
    "--under",
    "--uv-set",
    "--where",
    "-t",
}
_READ_COMMANDS = {
    "bounds",
    "cheatsheet",
    "describe",
    "distance",
    "find",
    "history",
    "info",
    "jobs",
    "material-binding",
    "nearest",
    "overlapping",
    "properties",
    "raycast",
    "resolve",
    "selection",
    "snapshot",
    "stats",
    "sublayers",
    "subsets",
    "visibility",
    "wait",
    "within",
}
_RENDER_COMMANDS = {"render", "render-frames"}
_VALIDATION_COMMANDS = {"render-probe", "validate", "verify"}
_SCENE_IO_COMMANDS = {"convert", "export", "open", "save"}
_HISTORY_COMMANDS = {"checkpoint", "redo", "undo"}
_EDIT_COMMANDS = {
    "align",
    "appearance",
    "create",
    "delete",
    "deselect",
    "duplicate",
    "group",
    "hide",
    "import",
    "isolate",
    "material",
    "material-apply",
    "new",
    "physics",
    "remove-api",
    "rename",
    "reparent",
    "scatter",
    "select",
    "set",
    "show",
    "transform",
}


def import_usd_cli_telemetry(
    run_dir: Path,
    trace_writer: TraceWriter,
    *,
    usd_cli_source_revision: str | None = None,
) -> dict[str, Any]:
    """Append one ``TraceWriter`` event per valid, not-yet-imported span.

    Missing telemetry is a supported state: the function returns an empty
    result and leaves the existing child-artifact trace untouched.  Malformed
    or truncated lines are counted and skipped.  A rotated ``.jsonl.1`` file is
    read before the live file, then spans are sorted by their source timestamp
    and deduplicated by ``(trace_id, span_id)``.
    """

    resolved_run_dir = run_dir.resolve()
    source_path = resolved_run_dir / "raw" / "usd_cli_telemetry.jsonl"
    source_artifacts: list[ContainedArtifactRead] = []
    for candidate in (
        source_path.with_name(source_path.name + ".1"),
        source_path,
    ):
        try:
            source_artifacts.append(
                read_contained_artifact(
                    resolved_run_dir,
                    candidate,
                    max_bytes=MAX_TELEMETRY_FILE_BYTES,
                    capture_bytes=True,
                )
            )
        except (OSError, ValueError):
            # Child-writable directories are untrusted. In particular, do not
            # follow a replaced ``raw`` directory to telemetry outside the run.
            continue
    source_paths = [artifact.path for artifact in source_artifacts]
    empty_result = {
        "schema_version": USD_CLI_TELEMETRY_IMPORT_SCHEMA_VERSION,
        "source_files": [str(path) for path in source_paths],
        "records_seen": 0,
        "valid_spans": 0,
        "imported_spans": 0,
        "invalid_records": 0,
        "duplicate_spans": 0,
        "already_imported_spans": 0,
        "wrong_trace_spans": 0,
        "wrong_parent_spans": 0,
        "wrong_run_spans": 0,
        "unsupported_schema_spans": 0,
        "correlation_error": None,
        "cross_checks": {},
        "cross_check_mismatches": [],
        "cross_check_load_failures": [],
    }
    if not source_paths:
        return empty_result

    loaded, invalid_records = _load_span_records(source_artifacts)
    empty_result["records_seen"] = len(loaded) + invalid_records
    empty_result["invalid_records"] = invalid_records

    request = _load_json(
        resolved_run_dir / "request.json",
        run_dir=resolved_run_dir,
    )
    expected_trace_id, expected_parent_span_id = _expected_trace_context(request)
    if expected_trace_id is None or expected_parent_span_id is None:
        correlation_error = (
            "request.json does not contain a valid telemetry trace_id and "
            "root_span_id; telemetry spans were not imported."
        )
        correlation_signature = _summary_signature(
            {
                "correlation_error": correlation_error,
                "source_files": [str(path) for path in source_paths],
                "records_seen": empty_result["records_seen"],
                "invalid_records": invalid_records,
                "request_created_at": request.get("created_at"),
                "request_telemetry": request.get("telemetry"),
            }
        )
        result = {
            **empty_result,
            "correlation_error": correlation_error,
        }
        if not _summary_already_written(
            trace_writer.path,
            signature=correlation_signature,
        ):
            trace_writer.write(
                "warning",
                phase="usd-cli telemetry",
                summary=correlation_error,
                artifacts=[str(path) for path in source_paths],
                data={
                    "schema_version": USD_CLI_TELEMETRY_IMPORT_SCHEMA_VERSION,
                    "warning_code": "usd_cli_telemetry_correlation_unavailable",
                    "records_seen": result["records_seen"],
                    "invalid_records": invalid_records,
                    "source_state_sha256": correlation_signature,
                },
            )
        return result

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_spans = 0
    wrong_trace_spans = 0
    wrong_parent_spans = 0
    wrong_run_spans = 0
    unsupported_schema_spans = 0
    for record in loaded:
        parsed = _parse_span_record(record)
        if parsed is None:
            invalid_records += 1
            continue
        if (
            parsed["telemetry_sdk"]["name"] != "usd-cli-tel"
            or parsed["telemetry_sdk"]["schema_version"] != 1
        ):
            unsupported_schema_spans += 1
            continue
        if parsed["trace_id"] != expected_trace_id:
            wrong_trace_spans += 1
            continue
        if parsed.get("parent_span_id") != expected_parent_span_id:
            wrong_parent_spans += 1
            continue
        if not _run_attribute_matches(
            parsed,
            expected_trace_id=expected_trace_id,
            run_dir=resolved_run_dir,
        ):
            wrong_run_spans += 1
            continue
        key = (parsed["trace_id"], parsed["span_id"])
        if key in unique:
            duplicate_spans += 1
            continue
        unique[key] = parsed

    spans = sorted(
        unique.values(),
        key=lambda span: (
            span["start_time_unix_nano"],
            span["end_time_unix_nano"],
            span["trace_id"],
            span["span_id"],
        ),
    )
    already_imported = _imported_span_keys(
        trace_writer.path,
        trace_id=expected_trace_id,
    )
    mapping = {
        "schema_version": USD_CLI_ARGV_MAPPING_VERSION,
        "source_revision": usd_cli_source_revision,
    }

    imported = 0
    already_imported_count = 0
    command_counts: Counter[str] = Counter()
    failed_count = 0
    render_count = 0
    accepted_artifacts: set[str] = set()
    for span in spans:
        key = (span["trace_id"], span["span_id"])
        command = _command_from_argv(span["argv"])
        command_counts[command] += 1
        if command in _RENDER_COMMANDS:
            render_count += 1
        if span["exit_code"] != 0:
            failed_count += 1
        artifact_checks = _artifact_checks(
            resolved_run_dir,
            command=command,
            argv=span["argv"],
        )
        artifacts = [
            str(check["resolved_path"])
            for check in artifact_checks
            if check["accepted"]
        ]
        accepted_artifacts.update(artifacts)
        if key in already_imported:
            already_imported_count += 1
            continue

        status_word = "completed" if span["exit_code"] == 0 else "failed"
        duration_ms = _duration_ms(span)
        trace_writer.write(
            "usd_cli_invocation",
            phase="usd-cli",
            summary=(
                f"usd-cli {command} {status_word} with exit code "
                f"{span['exit_code']} in {duration_ms:.3f} ms."
            ),
            artifacts=artifacts,
            time=_iso_time_from_unix_nano(span["start_time_unix_nano"]),
            data={
                "schema_version": USD_CLI_TELEMETRY_IMPORT_SCHEMA_VERSION,
                "evidence_trust": "child_reported_non_authoritative",
                "acceptance_use": False,
                "argv_mapping": mapping,
                "source_file": span["source_file"],
                "source_line": span["source_line"],
                "trace_id": span["trace_id"],
                "span_id": span["span_id"],
                "parent_span_id": span.get("parent_span_id"),
                "trace_matches_run": (span["trace_id"] == expected_trace_id),
                "parent_matches_run_root": (
                    span.get("parent_span_id") == expected_parent_span_id
                ),
                "name": span["name"],
                "kind": span["kind"],
                "start_time_unix_nano": span["start_time_unix_nano"],
                "end_time_unix_nano": span["end_time_unix_nano"],
                "duration_ms": duration_ms,
                "reported_duration_ms": span["duration_ms"],
                "duration_matches_timestamps": (
                    abs(span["duration_ms"] - duration_ms) <= 1.0
                ),
                "status_code": span["status_code"],
                "exit_code": span["exit_code"],
                "status_exit_code_consistent": (
                    (span["status_code"] == "OK") == (span["exit_code"] == 0)
                ),
                "command": command,
                "wrapper_command_hint": span.get("wrapper_command_hint"),
                "command_hint_matches_argv": (
                    span.get("wrapper_command_hint") == command.partition(".")[0]
                    if span.get("wrapper_command_hint")
                    else None
                ),
                "operation": _operation_for_command(command, span["argv"]),
                "argv": _redacted_argv(span["argv"]),
                "executable": span.get("executable"),
                "session": span.get("session"),
                "server": _redacted_url(span.get("server")),
                "scene": _redacted_scene(span.get("scene")),
                "telemetry_sdk": span.get("telemetry_sdk"),
                "artifact_checks": artifact_checks,
            },
        )
        imported += 1

    cross_checks = _cross_checks(
        resolved_run_dir,
        span_count=len(spans),
        render_count=render_count,
        failed_count=failed_count,
    )
    cross_check_mismatches = _cross_check_mismatches(cross_checks)
    cross_check_load_failures = _cross_check_load_failures(cross_checks)
    summary_signature = _summary_signature(
        {
            "source_files": [str(path) for path in source_paths],
            "expected_trace_id": expected_trace_id,
            "expected_root_span_id": expected_parent_span_id,
            "records_seen": len(loaded) + empty_result["invalid_records"],
            "valid_spans": len(spans),
            "invalid_records": invalid_records,
            "duplicate_spans": duplicate_spans,
            "wrong_trace_spans": wrong_trace_spans,
            "wrong_parent_spans": wrong_parent_spans,
            "wrong_run_spans": wrong_run_spans,
            "unsupported_schema_spans": unsupported_schema_spans,
            "command_counts": dict(sorted(command_counts.items())),
            "failed_spans": failed_count,
            "render_spans": render_count,
            "cross_checks": cross_checks,
            "accepted_artifacts": sorted(accepted_artifacts),
            "argv_mapping": mapping,
        }
    )
    summary_already_written = _summary_already_written(
        trace_writer.path,
        signature=summary_signature,
    )
    if imported or not summary_already_written:
        rejection_counts = {
            "invalid_records": invalid_records,
            "wrong_trace_spans": wrong_trace_spans,
            "wrong_parent_spans": wrong_parent_spans,
            "wrong_run_spans": wrong_run_spans,
            "unsupported_schema_spans": unsupported_schema_spans,
        }
        rejected_count = sum(rejection_counts.values())
        rejection_details = [
            f"{name}={count}" for name, count in rejection_counts.items() if count
        ]
        summary_event_type = (
            "warning"
            if rejected_count or cross_check_mismatches or cross_check_load_failures
            else "usd_cli_telemetry_ingested"
        )
        summary = (
            f"Imported {imported} child-reported usd-cli invocation span(s)"
            f" from {len(source_paths)} telemetry file(s)."
        )
        if invalid_records:
            summary += f" Skipped {invalid_records} invalid record(s)."
        if rejection_details:
            summary += " Rejected telemetry: " + ", ".join(rejection_details) + "."
        if cross_check_mismatches:
            summary += (
                " Count cross-check mismatch(es): "
                + ", ".join(cross_check_mismatches)
                + "."
            )
        if cross_check_load_failures:
            summary += (
                " Auxiliary cross-check input(s) unusable: "
                + ", ".join(cross_check_load_failures)
                + "."
            )
        trace_writer.write(
            summary_event_type,
            phase="usd-cli telemetry",
            summary=summary,
            artifacts=[
                *[str(path) for path in source_paths],
                *sorted(accepted_artifacts),
            ],
            data={
                "schema_version": USD_CLI_TELEMETRY_IMPORT_SCHEMA_VERSION,
                "argv_mapping": mapping,
                "expected_trace_id": expected_trace_id,
                "expected_root_span_id": expected_parent_span_id,
                "records_seen": len(loaded) + empty_result["invalid_records"],
                "valid_spans": len(spans),
                "imported_spans": imported,
                "invalid_records": invalid_records,
                "duplicate_spans": duplicate_spans,
                "already_imported_spans": already_imported_count,
                "wrong_trace_spans": wrong_trace_spans,
                "wrong_parent_spans": wrong_parent_spans,
                "wrong_run_spans": wrong_run_spans,
                "unsupported_schema_spans": unsupported_schema_spans,
                "command_counts": dict(sorted(command_counts.items())),
                "failed_spans": failed_count,
                "render_spans": render_count,
                "cross_checks": cross_checks,
                "cross_check_mismatches": cross_check_mismatches,
                "cross_check_load_failures": cross_check_load_failures,
                "rejection_details": rejection_details,
                "warning_code": (
                    "usd_cli_telemetry_count_mismatch"
                    if cross_check_mismatches
                    else (
                        "usd_cli_telemetry_cross_check_input_unusable"
                        if cross_check_load_failures
                        else (
                            "usd_cli_telemetry_records_rejected"
                            if rejected_count
                            else None
                        )
                    )
                ),
                "warning_codes": [
                    *(["usd_cli_telemetry_records_rejected"] if rejected_count else []),
                    *(
                        ["usd_cli_telemetry_count_mismatch"]
                        if cross_check_mismatches
                        else []
                    ),
                    *(
                        ["usd_cli_telemetry_cross_check_input_unusable"]
                        if cross_check_load_failures
                        else []
                    ),
                ],
                "source_state_sha256": summary_signature,
            },
        )

    return {
        **empty_result,
        "valid_spans": len(spans),
        "imported_spans": imported,
        "invalid_records": invalid_records,
        "duplicate_spans": duplicate_spans,
        "already_imported_spans": already_imported_count,
        "wrong_trace_spans": wrong_trace_spans,
        "wrong_parent_spans": wrong_parent_spans,
        "wrong_run_spans": wrong_run_spans,
        "unsupported_schema_spans": unsupported_schema_spans,
        "correlation_error": None,
        "cross_checks": cross_checks,
        "cross_check_mismatches": cross_check_mismatches,
        "cross_check_load_failures": cross_check_load_failures,
    }


def _load_span_records(
    artifacts: list[ContainedArtifactRead],
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    invalid = 0
    total_bytes = 0
    for artifact in artifacts:
        path = artifact.path
        if (
            artifact.size_bytes > MAX_TELEMETRY_FILE_BYTES
            or total_bytes + artifact.size_bytes > MAX_TELEMETRY_TOTAL_BYTES
        ):
            invalid += 1
            continue
        total_bytes += artifact.size_bytes
        if artifact.data is None:
            invalid += 1
            continue
        stream = io.TextIOWrapper(
            io.BytesIO(artifact.data),
            encoding="utf-8",
            errors="replace",
        )
        with stream:
            for line_number, line in _bounded_lines(stream):
                if len(records) >= MAX_TELEMETRY_RECORDS:
                    invalid += 1
                    break
                if line is None:
                    invalid += 1
                    continue
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                if not isinstance(value, dict):
                    invalid += 1
                    continue
                records.append(
                    {
                        "record": value,
                        "source_file": str(path),
                        "source_line": line_number,
                    }
                )
    return records, invalid


def _bounded_lines(stream: TextIO) -> Iterator[tuple[int, str | None]]:
    """Yield lines without ever allocating an unbounded child-authored line."""

    line_number = 0
    while True:
        line = stream.readline(MAX_TELEMETRY_LINE_CHARS + 1)
        if not line:
            return
        line_number += 1
        oversized = len(line) > MAX_TELEMETRY_LINE_CHARS
        if oversized and not line.endswith("\n"):
            while True:
                remainder = stream.readline(MAX_TELEMETRY_LINE_CHARS + 1)
                if not remainder or remainder.endswith("\n"):
                    break
        yield line_number, None if oversized else line


def _parse_span_record(raw: dict[str, Any]) -> dict[str, Any] | None:
    record = raw["record"]
    trace_id = record.get("trace_id")
    span_id = record.get("span_id")
    parent_span_id = record.get("parent_span_id")
    attributes = record.get("attributes")
    status = record.get("status")
    argv = (
        attributes.get("process.command_args") if isinstance(attributes, dict) else None
    )
    exit_code = (
        attributes.get("process.exit_code") if isinstance(attributes, dict) else None
    )
    start = record.get("start_time_unix_nano")
    end = record.get("end_time_unix_nano")
    duration = record.get("duration_ms")
    name = record.get("name")
    kind = record.get("kind")
    if not (
        _is_lower_hex(trace_id, 32)
        and _is_lower_hex(span_id, 16)
        and (parent_span_id is None or _is_lower_hex(parent_span_id, 16))
        and isinstance(name, str)
        and (name == "usd-cli" or name.startswith("usd-cli."))
        and kind == "SPAN_KIND_CLIENT"
        and isinstance(attributes, dict)
        and isinstance(status, dict)
        and status.get("code") in {"OK", "ERROR"}
        and isinstance(argv, list)
        and all(isinstance(token, str) for token in argv)
        and isinstance(exit_code, int)
        and not isinstance(exit_code, bool)
        and isinstance(start, int)
        and not isinstance(start, bool)
        and start >= 0
        and _iso_time_from_unix_nano(start) is not None
        and isinstance(end, int)
        and not isinstance(end, bool)
        and end >= start
        and _iso_time_from_unix_nano(end) is not None
        and _is_finite_nonnegative_number(duration)
    ):
        return None

    scene: dict[str, Any] | None = None
    scene_path = attributes.get("usd.scene.path")
    scene_paths = attributes.get("usd.scene.paths")
    scene_size = attributes.get("usd.scene.size_bytes")
    if isinstance(scene_path, str):
        scene = {"path": scene_path}
        if isinstance(scene_size, int) and not isinstance(scene_size, bool):
            scene["size_bytes"] = scene_size
    if isinstance(scene_paths, list) and all(
        isinstance(path, str) for path in scene_paths
    ):
        scene = scene or {}
        scene["paths"] = list(scene_paths)

    telemetry_sdk = {
        "name": attributes.get("telemetry.sdk.name"),
        "version": attributes.get("telemetry.sdk.version"),
        "schema_version": attributes.get("telemetry.schema_version"),
    }
    return {
        "source_file": raw["source_file"],
        "source_line": raw["source_line"],
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "kind": kind,
        "start_time_unix_nano": start,
        "end_time_unix_nano": end,
        "duration_ms": float(duration),
        "status_code": status["code"],
        "argv": list(argv),
        "exit_code": exit_code,
        "wrapper_command_hint": (
            attributes.get("usd.command")
            if isinstance(attributes.get("usd.command"), str)
            else None
        ),
        "executable": (
            attributes.get("process.executable.path")
            if isinstance(attributes.get("process.executable.path"), str)
            else None
        ),
        "session": (
            attributes.get("usd.session")
            if isinstance(attributes.get("usd.session"), str)
            else None
        ),
        "server": (
            attributes.get("usd.server")
            if isinstance(attributes.get("usd.server"), str)
            else None
        ),
        "scene": scene,
        "telemetry_sdk": telemetry_sdk,
        "run_dir_attribute": (
            attributes.get("wu.run_dir")
            if isinstance(attributes.get("wu.run_dir"), str)
            else None
        ),
        "run_id_attribute": (
            attributes.get("wu.run_id")
            if isinstance(attributes.get("wu.run_id"), str)
            else None
        ),
    }


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value != "0" * length
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_finite_nonnegative_number(value: object) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return value >= 0 and math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _command_from_argv(argv: list[str]) -> str:
    argv = _normalized_command_argv(argv)
    command_index: int | None = None
    skip_next = False
    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--":
            continue
        if token.startswith("-"):
            flag = token.partition("=")[0]
            if flag in _GLOBAL_VALUE_OPTIONS and "=" not in token:
                skip_next = True
            continue
        command_index = index
        break
    if command_index is None:
        return "<none>"
    command = argv[command_index]
    remaining = argv[command_index + 1 :]
    material_ref = _first_non_option_token(
        remaining,
        value_options=_MATERIAL_VALUE_OPTIONS,
    )
    if command == "material" and material_ref == "audit":
        return "material.audit"
    subcommand = _first_non_option_token(remaining)
    if (
        command in {"appearance", "camera", "checkpoint", "physics", "remote", "server"}
        and subcommand is not None
    ):
        return f"{command}.{subcommand}"
    return command


def _normalized_command_argv(argv: list[str]) -> list[str]:
    """Return argv beginning at usd-cli's first global option or command.

    ``usd-cli-tel`` currently records only the wrapped arguments, while the
    OpenTelemetry process semantic convention permits ``argv[0]`` to be
    present. Importers must accept both shapes without classifying the
    executable itself as the command.
    """

    if not argv:
        return []
    first_name = Path(argv[0]).name.lower()
    if first_name in _USD_CLI_EXECUTABLE_NAMES:
        return argv[1:]
    if first_name.startswith(("python", "pypy")):
        if len(argv) >= 3 and argv[1] == "-m" and argv[2] in _USD_CLI_PYTHON_MODULES:
            return argv[3:]
        if len(argv) >= 2:
            script_name = Path(argv[1]).name.lower()
            if script_name in _USD_CLI_EXECUTABLE_NAMES | {
                "__main__.py",
                "main.py",
            }:
                return argv[2:]
    return argv


def _redacted_url(value: object) -> str | None:
    """Remove URL userinfo, query, and fragment from trace-visible evidence."""

    if not isinstance(value, str):
        return None
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value) is None:
        return value
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<redacted-invalid-url>"
    netloc = ""
    if parsed.netloc:
        try:
            hostname = parsed.hostname or ""
            port_value = parsed.port
        except ValueError:
            return "<redacted-invalid-url>"
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        port = f":{port_value}" if port_value is not None else ""
        netloc = f"{hostname}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _redacted_token(token: str) -> str:
    """Redact URL credentials/query data and inline credentials in one token."""

    sanitized_url = _redacted_url(token)
    if sanitized_url != token:
        return sanitized_url or "<redacted>"

    flag, separator, value = token.partition("=")
    if separator:
        sanitized_value = _redacted_url(value)
        if sanitized_value != value:
            return f"{flag}={sanitized_value or '<redacted>'}"

    inline = _INLINE_SECRET_RE.match(token)
    if inline is not None:
        return f"{inline.group('name')}=<redacted>"
    if token.lower().startswith("bearer "):
        return "Bearer <redacted>"
    return token


def _redacted_scene(value: object) -> dict[str, Any] | None:
    """Redact URL-like scene attributes while preserving non-sensitive shape."""

    if not isinstance(value, dict):
        return None
    redacted: dict[str, Any] = {}
    path = value.get("path")
    if isinstance(path, str):
        redacted["path"] = _redacted_token(path)
    paths = value.get("paths")
    if isinstance(paths, list):
        redacted["paths"] = [
            _redacted_token(path) for path in paths if isinstance(path, str)
        ]
    size_bytes = value.get("size_bytes")
    if isinstance(size_bytes, int) and not isinstance(size_bytes, bool):
        redacted["size_bytes"] = size_bytes
    return redacted or None


def _redacted_argv(argv: list[str]) -> list[str]:
    """Defensively redact credentials even if the child wrapper did not."""

    redacted: list[str] = []
    redact_next = False
    sanitize_url_next = False
    for token in argv:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if sanitize_url_next:
            redacted.append(_redacted_url(token) or "<redacted>")
            sanitize_url_next = False
            continue
        flag, separator, value = token.partition("=")
        normalized_flag = flag.lower()
        if normalized_flag in _SENSITIVE_OPTION_NAMES:
            if separator:
                redacted.append(f"{flag}=<redacted>")
            else:
                redacted.append(token)
                redact_next = True
            continue
        if normalized_flag == "--server":
            if separator:
                redacted.append(f"{flag}={_redacted_url(value) or '<redacted>'}")
            else:
                redacted.append(token)
                sanitize_url_next = True
            continue
        redacted.append(_redacted_token(token))
    return redacted


def _first_non_option_token(
    tokens: list[str],
    *,
    value_options: set[str] | None = None,
) -> str | None:
    options_with_values = _GLOBAL_VALUE_OPTIONS | (value_options or set())
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token == "--":
            continue
        if token.startswith("-"):
            flag = token.partition("=")[0]
            if flag in options_with_values and "=" not in token:
                skip_next = True
            continue
        return token
    return None


def _operation_for_command(command: str, argv: list[str]) -> str:
    root_command = command.partition(".")[0]
    if root_command in _RENDER_COMMANDS:
        return "render"
    if root_command in _VALIDATION_COMMANDS or command.startswith("physics.validate"):
        return "validation"
    if command == "camera.list":
        return "scene_read"
    if command.startswith("camera."):
        return "scene_edit"
    if root_command == "sublayers" and "--drop-dead" in argv:
        return "scene_edit"
    if root_command in _READ_COMMANDS or command in {
        "appearance.audit",
        "material.audit",
        "physics.inspect",
        "server.status",
    }:
        return "scene_read"
    if root_command in _SCENE_IO_COMMANDS:
        return "scene_io"
    if root_command in _HISTORY_COMMANDS:
        return "history"
    if root_command in _EDIT_COMMANDS:
        return "scene_edit"
    return "usd_cli"


def _artifact_checks(
    run_dir: Path,
    *,
    command: str,
    argv: list[str],
) -> list[dict[str, Any]]:
    candidates = _artifact_candidates(command, argv)
    checks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        resolved = _resolve_run_artifact_candidate(run_dir, candidate)
        within_run_dir = resolved is not None
        exists = bool(
            resolved is not None and (resolved.is_file() or resolved.is_dir())
        )
        checks.append(
            {
                "argument": _redacted_token(candidate),
                "resolved_path": str(resolved) if resolved is not None else None,
                "within_run_dir": within_run_dir,
                "exists": exists,
                "accepted": within_run_dir and exists,
            }
        )
    return checks


def _artifact_candidates(command: str, argv: list[str]) -> list[str]:
    argv = _normalized_command_argv(argv)
    root_command = command.partition(".")[0]
    command_index = _command_index(argv, root_command)
    if command_index is None:
        return []
    command_args = argv[command_index + 1 :]
    if root_command in _RENDER_COMMANDS or command == "physics.simulate":
        return _option_values(command_args, {"-o", "--output"})
    positional = _positional_args(command_args)
    if root_command == "save":
        return positional[:1]
    if root_command == "export":
        return positional[1:2]
    if root_command == "convert":
        return positional[1:2]
    return []


def _command_index(argv: list[str], command: str) -> int | None:
    skip_next = False
    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-"):
            flag = token.partition("=")[0]
            if flag in _GLOBAL_VALUE_OPTIONS and "=" not in token:
                skip_next = True
            continue
        return index if token == command else None
    return None


def _option_values(argv: list[str], names: set[str]) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(argv):
        flag, separator, value = token.partition("=")
        if flag not in names:
            continue
        if separator:
            if value:
                values.append(value)
        elif index + 1 < len(argv):
            values.append(argv[index + 1])
    return values


def _positional_args(argv: list[str]) -> list[str]:
    """Return conservative positional args for the mapped scene-I/O commands."""

    positional: list[str] = []
    skip_next = False
    value_options = {
        "--output-format",
    }
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--":
            continue
        if token.startswith("-"):
            flag = token.partition("=")[0]
            if flag in value_options and "=" not in token:
                skip_next = True
            continue
        positional.append(token)
    return positional


def _resolve_run_artifact_candidate(run_dir: Path, value: str) -> Path | None:
    if not value or value == "-" or "://" in value or "[" in value:
        return None
    try:
        candidate = Path(os.path.expanduser(value))
        if not candidate.is_absolute():
            # The wrapper schema does not record process.cwd, so a relative
            # argv value cannot be tied to the invocation's actual output.
            return None
        resolved = contained_regular_file(run_dir, candidate)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _run_attribute_matches(
    span: dict[str, Any],
    *,
    expected_trace_id: str,
    run_dir: Path,
) -> bool:
    run_id = span.get("run_id_attribute")
    if run_id is not None:
        return run_id == expected_trace_id
    # Compatibility with telemetry produced before the grammar-safe run id was
    # introduced. Trace id and parent id remain mandatory primary correlation.
    return span.get("run_dir_attribute") == run_dir.name


def _expected_trace_context(
    request: dict[str, Any],
) -> tuple[str | None, str | None]:
    telemetry = request.get("telemetry")
    if not isinstance(telemetry, dict):
        return None, None
    trace_id = telemetry.get("trace_id")
    parent_span_id = telemetry.get("root_span_id")
    return (
        trace_id if _is_lower_hex(trace_id, 32) else None,
        parent_span_id if _is_lower_hex(parent_span_id, 16) else None,
    )


def _duration_ms(span: dict[str, Any]) -> float:
    return round(
        (span["end_time_unix_nano"] - span["start_time_unix_nano"]) / 1e6,
        3,
    )


def _iso_time_from_unix_nano(value: int) -> str | None:
    try:
        seconds, nanoseconds = divmod(value, 1_000_000_000)
        moment = datetime.fromtimestamp(seconds, tz=UTC).replace(
            microsecond=nanoseconds // 1_000
        )
    except (OSError, OverflowError, ValueError):
        return None
    return moment.isoformat()


def _cross_checks(
    run_dir: Path,
    *,
    span_count: int,
    render_count: int,
    failed_count: int,
) -> dict[str, Any]:
    counts_path = run_dir / "api_operation_counts.json"
    metrics_path = run_dir / "run_cost_metrics.json"
    counts, counts_load = _load_json_with_status(counts_path, run_dir=run_dir)
    metrics, metrics_load = _load_json_with_status(metrics_path, run_dir=run_dir)
    reported_call_count = _first_int(
        counts,
        "usd_cli_calls_total",
        "api_operation_count_total",
    )
    reported_render_count = _first_int(
        counts,
        "render_calls_total",
        "render_count_total",
    )
    reported_failed_count = _first_int(
        counts,
        "failed_usd_cli_calls",
        "failed_api_calls",
    )
    shell_commands_total = _first_int(metrics, "shell_commands_total")
    failed_commands_total = _first_int(metrics, "failed_commands_total")
    metrics_call_count = _first_int(metrics, "usd_cli_calls_total")
    metrics_render_count = _first_int(metrics, "render_calls_total")
    metrics_failed_count = _first_int(metrics, "failed_usd_cli_calls_total")
    return {
        "api_operation_counts": {
            "path": str(counts_path),
            **counts_load,
            "reported_usd_cli_calls_total": reported_call_count,
            "telemetry_usd_cli_calls_total": span_count,
            "call_count_matches": (
                reported_call_count == span_count
                if reported_call_count is not None
                else None
            ),
            "reported_render_calls_total": reported_render_count,
            "telemetry_render_calls_total": render_count,
            "render_count_matches": (
                reported_render_count == render_count
                if reported_render_count is not None
                else None
            ),
            "reported_failed_calls": reported_failed_count,
            "telemetry_failed_calls": failed_count,
            "failed_count_matches": (
                reported_failed_count == failed_count
                if reported_failed_count is not None
                else None
            ),
        },
        "run_cost_metrics": {
            "path": str(metrics_path),
            **metrics_load,
            "shell_commands_total": shell_commands_total,
            "failed_commands_total": failed_commands_total,
            "reported_usd_cli_calls_total": metrics_call_count,
            "reported_render_calls_total": metrics_render_count,
            "reported_failed_usd_cli_calls_total": metrics_failed_count,
            "telemetry_usd_cli_calls_total": span_count,
            "telemetry_render_calls_total": render_count,
            "telemetry_failed_usd_cli_calls": failed_count,
            "call_count_matches": (
                metrics_call_count == span_count
                if metrics_call_count is not None
                else None
            ),
            "render_count_matches": (
                metrics_render_count == render_count
                if metrics_render_count is not None
                else None
            ),
            "failed_count_matches": (
                metrics_failed_count == failed_count
                if metrics_failed_count is not None
                else None
            ),
            "comparison_scope": (
                "Shell metrics count child shell executions; telemetry counts "
                "usd-cli invocations, so values are recorded for audit but are "
                "not required to be equal."
            ),
        },
    }


def _cross_check_mismatches(cross_checks: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for source_name in ("api_operation_counts", "run_cost_metrics"):
        source = cross_checks.get(source_name)
        if not isinstance(source, dict):
            continue
        for check_name in (
            "call_count_matches",
            "render_count_matches",
            "failed_count_matches",
        ):
            if source.get(check_name) is False:
                mismatches.append(f"{source_name}.{check_name}")
    return mismatches


def _cross_check_load_failures(cross_checks: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for source_name in ("api_operation_counts", "run_cost_metrics"):
        source = cross_checks.get(source_name)
        if not isinstance(source, dict):
            continue
        load_status = source.get("load_status")
        if source.get("present") is True and load_status != "parsed":
            failures.append(f"{source_name}.{load_status}")
    return failures


def _first_int(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _load_json(
    path: Path,
    *,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    value, _load = _load_json_with_status(path, run_dir=run_dir)
    return value


def _load_json_with_status(
    path: Path,
    *,
    run_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = run_dir or path.parent
    try:
        artifact = read_contained_artifact(
            root,
            path,
            max_bytes=MAX_AUXILIARY_JSON_BYTES,
            capture_bytes=True,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        reason = str(exc)
        if "does not exist" in reason or isinstance(exc, FileNotFoundError):
            return {}, {
                "present": False,
                "load_status": "missing",
                "load_reason": None,
            }
        if "exceeds" in reason:
            return {}, {
                "present": True,
                "load_status": "oversize",
                "load_reason": (
                    f"Input exceeds the {MAX_AUXILIARY_JSON_BYTES}-byte limit."
                ),
            }
        return {}, {
            "present": True,
            "load_status": "symlink" if "symlink" in reason.lower() else "malformed",
            "load_reason": reason,
        }
    if artifact.data is None:
        return {}, {
            "present": True,
            "load_status": "malformed",
            "load_reason": "Input bytes were unavailable.",
        }
    try:
        value = json.loads(artifact.data.decode("utf-8"))
    except UnicodeDecodeError:
        return {}, {
            "present": True,
            "load_status": "malformed",
            "load_reason": "Input is not valid UTF-8.",
        }
    except json.JSONDecodeError as exc:
        return {}, {
            "present": True,
            "load_status": "malformed",
            "load_reason": (f"Invalid JSON at line {exc.lineno}, column {exc.colno}."),
        }
    if not isinstance(value, dict):
        return {}, {
            "present": True,
            "load_status": "non_object",
            "load_reason": "Top-level JSON value must be an object.",
        }
    return value, {
        "present": True,
        "load_status": "parsed",
        "load_reason": None,
    }


def _imported_span_keys(
    path: Path,
    *,
    trace_id: str | None = None,
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for event in _load_trace_events(path):
        if event.get("event_type") != "usd_cli_invocation":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        event_trace_id = data.get("trace_id")
        span_id = data.get("span_id")
        if (
            isinstance(event_trace_id, str)
            and isinstance(span_id, str)
            and (trace_id is None or event_trace_id == trace_id)
        ):
            keys.add((event_trace_id, span_id))
    return keys


def _summary_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _summary_already_written(
    path: Path,
    *,
    signature: str | None = None,
) -> bool:
    for event in _load_trace_events(path):
        data = event.get("data")
        is_summary = event.get("event_type") in {
            "usd_cli_telemetry_imported",
            "usd_cli_telemetry_ingested",
            "usd_cli_telemetry_parse_warning",
        } or (
            event.get("event_type") == "warning"
            and isinstance(data, dict)
            and data.get("schema_version") == USD_CLI_TELEMETRY_IMPORT_SCHEMA_VERSION
        )
        if not is_summary:
            continue
        if signature is None:
            return True
        if isinstance(data, dict) and data.get("source_state_sha256") == signature:
            return True
    return False


def _load_trace_events(path: Path) -> list[dict[str, Any]]:
    try:
        run_dir = path.parent.parent
        artifact = read_contained_artifact(
            run_dir,
            path,
            max_bytes=MAX_TELEMETRY_TOTAL_BYTES,
            capture_bytes=True,
        )
    except ValueError as exc:
        if "does not exist" in str(exc):
            return []
        raise ValueError(f"Refusing to audit unsafe trace file {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to audit trace event file {path}: {exc}") from exc
    if artifact.data is None:
        raise ValueError(f"Unable to audit trace event file {path}: bytes unavailable")
    stream = io.TextIOWrapper(
        io.BytesIO(artifact.data),
        encoding="utf-8",
        errors="replace",
    )
    events: list[dict[str, Any]] = []
    with stream:
        for _, line in _bounded_lines(stream):
            if len(events) >= MAX_TELEMETRY_RECORDS:
                raise ValueError(f"Trace event count exceeds the audit limit: {path}")
            if line is None:
                raise ValueError(f"Trace event line exceeds the audit limit: {path}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
    return events
