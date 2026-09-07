# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observable trace support for content-workflow-cli runs."""

from __future__ import annotations

import errno
import json
import logging
import os
import stat
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from content_agent_workflows.common.artifacts import (
    atomic_write_text,
    contained_regular_file,
    read_contained_artifact,
)
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    append_bytes_to_confined,
    open_confined_directory,
)

TRACE_SCHEMA_VERSION = "content-agents.trace.v1"
MEMORY_EVENT_SCHEMA_VERSION = "content-agent-memory.event.v1"
REPLAY_SCHEMA_VERSION = "content-agents.replay.v1"
RETROSPECTIVE_SCHEMA_VERSION = "content-agents.retrospective.v1"
_MAX_JSON_ARTIFACT_BYTES = 8 * 1024 * 1024
_MAX_JSONL_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_TEXT_ARTIFACT_BYTES = 4 * 1024 * 1024
_MAX_TOTAL_TRACE_INPUT_BYTES = 32 * 1024 * 1024
_MAX_JSONL_RECORDS = 50_000
_MAX_CHILD_ITEM_FILES = 128


@dataclass(slots=True)
class _TraceInputBudget:
    """Per-build cap for all child-writable inputs consumed by the trace."""

    max_bytes: int
    consumed_bytes: int = 0

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.max_bytes - self.consumed_bytes)

    def consume(self, size: int) -> None:
        if size < 0 or size > self.remaining_bytes:
            raise ValueError("Trace input aggregate byte limit exceeded.")
        self.consumed_bytes += size


_ACTIVE_TRACE_INPUT_BUDGET: ContextVar[_TraceInputBudget | None] = ContextVar(
    "content_workflow_cli_trace_input_budget",
    default=None,
)

logger = logging.getLogger(__name__)

_UNSAFE_DIRECTORY_OPEN_ERRNOS = frozenset({errno.ELOOP, errno.ENOTDIR})
_UNSAFE_FILE_OPEN_ERRNOS = frozenset(
    {errno.EEXIST, errno.EISDIR, errno.ELOOP, errno.ENOTDIR, errno.ENXIO}
)


class UnsafeRunArtifactError(RuntimeError):
    """Raised when an untrusted run artifact could redirect a parent write."""


class _RunArtifactWriteError(OSError):
    """Operational write error that may have left a partial append."""


def _open_run_artifact(
    path: str | os.PathLike[str],
    flags: int,
    mode: int = 0o777,
    *,
    dir_fd: int | None = None,
    unsafe_errnos: frozenset[int],
    unsafe_message: str,
) -> int:
    """Open a validated artifact component and preserve operational I/O errors."""

    try:
        return os.open(path, flags, mode, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in unsafe_errnos:
            raise UnsafeRunArtifactError(unsafe_message) from exc
        raise


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _contained_relative_path(root: Path, candidate: Path) -> Path:
    value = candidate.expanduser()
    absolute = Path(os.path.abspath(value if value.is_absolute() else root / value))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Trace path is outside the workflow run: {absolute}") from exc
    if not relative.parts or relative.name in {"", ".", ".."}:
        raise ValueError(f"Unsafe trace path: {absolute}")
    return relative


def _open_contained_trace_directory(
    root: Path,
    relative: Path,
    *,
    create: bool,
) -> int:
    """Open a trace directory from a trusted root without following symlinks."""

    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(root, directory_flags)
    try:
        for component in relative.parts:
            if component in {"", ".", ".."}:
                raise ValueError(f"Unsafe trace directory component: {component!r}")
            try:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o700, dir_fd=directory_fd)
                except FileExistsError:
                    pass
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except Exception:
        os.close(directory_fd)
        raise


def append_jsonl(
    path: Path,
    event: dict[str, Any],
    *,
    within: Path | None = None,
) -> None:
    value = path.expanduser()
    if within is None:
        absolute = Path(os.path.abspath(value))
        absolute.parent.mkdir(parents=True, exist_ok=True)
        root = absolute.parent.resolve(strict=True)
        relative = Path(absolute.name)
    else:
        root = within.expanduser().resolve(strict=True)
        relative = _contained_relative_path(root, value)

    serialized_event = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
    if os.name != "posix":
        with open_confined_directory(root) as root_descriptor:
            append_bytes_to_confined(
                root_descriptor,
                relative.as_posix(),
                serialized_event,
                file_mode=0o600,
            )
        return

    directory_fd = _open_contained_trace_directory(
        root,
        relative.parent,
        create=True,
    )
    file_flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_fd = os.open(relative.name, file_flags, 0o600, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    try:
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise OSError(f"Refusing to append trace event to non-file: {path}")
        with os.fdopen(file_fd, "a", encoding="utf-8") as stream:
            file_fd = -1
            stream.write(serialized_event.decode("utf-8"))
    finally:
        if file_fd >= 0:
            os.close(file_fd)


def append_run_text(run_dir: Path, path: Path, text: str) -> None:
    """Append beneath ``run_dir`` without following child-controlled links."""

    lexical_run_dir = Path(os.path.abspath(run_dir.expanduser()))
    lexical_path = Path(os.path.abspath(path.expanduser()))
    try:
        relative_path = lexical_path.relative_to(lexical_run_dir)
    except ValueError as exc:
        raise UnsafeRunArtifactError(
            f"Run artifact path escapes the run directory: {path}"
        ) from exc
    if not relative_path.parts:
        raise UnsafeRunArtifactError(f"Run artifact path is not a file: {path}")

    if os.name != "posix":
        try:
            with open_confined_directory(lexical_run_dir) as root_descriptor:
                append_bytes_to_confined(
                    root_descriptor,
                    relative_path.as_posix(),
                    text.encode("utf-8"),
                    file_mode=0o600,
                )
        except ArtifactPathError as exc:
            raise UnsafeRunArtifactError(
                f"Run artifact changed to an unsafe path: {lexical_path}"
            ) from exc
        except OSError as exc:
            raise _RunArtifactWriteError(exc.errno, exc.strerror) from exc
        return

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = (
        os.O_WRONLY
        | os.O_APPEND
        | os.O_CREAT
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
        | os.O_NONBLOCK
    )
    directory_fds: list[int] = []
    file_fd: int | None = None
    try:
        directory_fds.append(
            _open_run_artifact(
                lexical_run_dir,
                directory_flags,
                unsafe_errnos=_UNSAFE_DIRECTORY_OPEN_ERRNOS,
                unsafe_message=(
                    "Run directory must not be a symlink or non-directory: "
                    f"{lexical_run_dir}"
                ),
            )
        )
        current_fd = directory_fds[-1]
        for component in relative_path.parts[:-1]:
            try:
                next_fd = _open_run_artifact(
                    component,
                    directory_flags,
                    dir_fd=current_fd,
                    unsafe_errnos=_UNSAFE_DIRECTORY_OPEN_ERRNOS,
                    unsafe_message=(
                        "Run artifact directory must not be a symlink or "
                        f"non-directory: {lexical_path.parent}"
                    ),
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                except FileExistsError as exc:
                    raise UnsafeRunArtifactError(
                        "Run artifact directory changed while being created: "
                        f"{lexical_path.parent}"
                    ) from exc
                next_fd = _open_run_artifact(
                    component,
                    directory_flags,
                    dir_fd=current_fd,
                    unsafe_errnos=_UNSAFE_DIRECTORY_OPEN_ERRNOS,
                    unsafe_message=(
                        "Run artifact directory changed to a symlink or "
                        f"non-directory: {lexical_path.parent}"
                    ),
                )
            directory_fds.append(next_fd)
            current_fd = next_fd

        existing_metadata: os.stat_result | None
        try:
            existing_metadata = os.stat(
                relative_path.name,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing_metadata = None
        if existing_metadata is not None and (
            not stat.S_ISREG(existing_metadata.st_mode)
            or existing_metadata.st_nlink != 1
        ):
            raise UnsafeRunArtifactError(
                f"Run artifact must be a singly linked regular file: {lexical_path}"
            )

        open_flags = file_flags
        if existing_metadata is None:
            open_flags |= os.O_EXCL
        file_fd = _open_run_artifact(
            relative_path.name,
            open_flags,
            0o600,
            dir_fd=current_fd,
            unsafe_errnos=_UNSAFE_FILE_OPEN_ERRNOS,
            unsafe_message=f"Run artifact changed to an unsafe file: {lexical_path}",
        )
        metadata = os.fstat(file_fd)
        replaced_existing_file = existing_metadata is not None and (
            metadata.st_dev != existing_metadata.st_dev
            or metadata.st_ino != existing_metadata.st_ino
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or replaced_existing_file
        ):
            raise UnsafeRunArtifactError(
                f"Run artifact must be a singly linked regular file: {lexical_path}"
            )
        stream = os.fdopen(file_fd, "a", encoding="utf-8")
        file_fd = None
        try:
            with stream:
                written = stream.write(text)
                if written != len(text):
                    raise OSError(
                        errno.EIO,
                        f"Short run artifact write: {written}/{len(text)} characters",
                    )
        except OSError as exc:
            raise _RunArtifactWriteError(exc.errno, exc.strerror) from exc
    except UnsafeRunArtifactError:
        raise
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


class TraceWriter:
    """Append-only writer for observable run events.

    Operational append failures are logged and ignored so the run can continue;
    unsafe artifact-path errors still propagate and remain fatal.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.expanduser().resolve(strict=True)
        if not self.run_dir.is_dir():
            raise ValueError(f"Workflow run root is not a directory: {self.run_dir}")
        self.path = self.run_dir / "trace" / "events.jsonl"
        self._needs_line_boundary = False

    def write(
        self,
        event_type: str,
        *,
        phase: str,
        summary: str,
        artifacts: list[str] | None = None,
        data: dict[str, Any] | None = None,
        time: str | None = None,
    ) -> None:
        """Write an event best-effort while preserving fatal path-safety errors."""

        event = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "time": time or utc_now(),
            "event_type": event_type,
            "phase": phase,
            "summary": summary,
            "artifacts": artifacts or [],
            "data": data or {},
        }
        serialized_event = json.dumps(event, sort_keys=True) + "\n"
        if self._needs_line_boundary:
            serialized_event = "\n" + serialized_event
        try:
            append_run_text(
                self.run_dir,
                self.path,
                serialized_event,
            )
        except OSError as exc:
            # A failed append may have written a JSON prefix. Ensure the next
            # successful event starts on a fresh line so it remains readable.
            self._needs_line_boundary = isinstance(exc, _RunArtifactWriteError)
            logger.warning(
                "Unable to append trace event %s to %s; continuing without this "
                "observability event: %s",
                event_type,
                self.path,
                exc,
            )
        else:
            self._needs_line_boundary = False


def build_trace(run_dir: Path) -> dict[str, Any]:
    """Build operation trace and replay manifest files for a run directory."""

    budget = _TraceInputBudget(max_bytes=_MAX_TOTAL_TRACE_INPUT_BYTES)
    token = _ACTIVE_TRACE_INPUT_BUDGET.set(budget)
    try:
        return _build_trace(run_dir)
    finally:
        _ACTIVE_TRACE_INPUT_BUDGET.reset(token)


def _build_trace(run_dir: Path) -> dict[str, Any]:
    """Build trace outputs with one aggregate child-input read budget."""

    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    trace_dir = run_dir / "trace"

    request = _load_json_object(run_dir / "request.json", run_dir=run_dir)
    assignments = _load_json_object(run_dir / "assignments.json", run_dir=run_dir)
    counts = _load_json_object(
        run_dir / "api_operation_counts.json",
        run_dir=run_dir,
    )
    agent_events = _current_run_agent_events(
        request,
        _load_jsonl(trace_dir / "events.jsonl", run_dir=run_dir),
    )
    child_commands = _load_child_command_records(run_dir)
    timeline = _build_timeline(
        run_dir,
        request,
        assignments,
        counts,
        agent_events,
    )
    event_timeline = _timeline_from_agent_events(run_dir, agent_events, 0)
    child_timeline = _timeline_from_child_commands(run_dir, child_commands, 0)
    if request.get("scene_backend") == "usd-cli":
        timeline = _merge_usd_cli_timeline(
            timeline,
            event_timeline,
            child_timeline,
        )
    else:
        timeline.extend(event_timeline)
        timeline.extend(child_timeline)
    _renumber_timeline(timeline)
    replay_manifest = _build_replay_manifest(run_dir, timeline)
    run_retrospective = _build_run_retrospective(
        run_dir=run_dir,
        request=request,
        assignments=assignments,
        counts=counts,
        agent_events=agent_events,
        child_commands=child_commands,
        timeline=timeline,
    )

    trace = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "workflow": request.get("workflow", "materials.assign"),
        "request": request,
        "stats": counts,
        "material_coverage": _material_coverage_summary(assignments, counts),
        "assignments_summary": _assignments_summary(assignments),
        "agent_events": agent_events,
        "child_commands": child_commands,
        "run_retrospective": run_retrospective,
        "timeline": timeline,
    }

    operation_trace_json = trace_dir / "operation_trace.json"
    operation_trace_md = trace_dir / "operation_trace.md"
    run_retrospective_json = trace_dir / "run_retrospective.json"
    replay_manifest_json = trace_dir / "replay_manifest.json"

    atomic_write_text(
        operation_trace_json,
        json.dumps(trace, indent=2),
        within=run_dir,
    )
    atomic_write_text(
        operation_trace_md,
        _render_trace_markdown(trace),
        within=run_dir,
    )
    atomic_write_text(
        run_retrospective_json,
        json.dumps(run_retrospective, indent=2),
        within=run_dir,
    )
    atomic_write_text(
        replay_manifest_json,
        json.dumps(replay_manifest, indent=2),
        within=run_dir,
    )

    return {
        "trace": trace,
        "replay_manifest": replay_manifest,
        "operation_trace_json": str(operation_trace_json),
        "operation_trace_md": str(operation_trace_md),
        "run_retrospective_json": str(run_retrospective_json),
        "replay_manifest_json": str(replay_manifest_json),
    }


def _build_timeline(
    run_dir: Path,
    request: dict[str, Any],
    assignments: dict[str, Any],
    counts: dict[str, Any],
    agent_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if request.get("scene_backend") == "usd-cli":
        return _build_usd_cli_timeline(
            run_dir,
            request,
            assignments,
            counts,
            agent_events,
        )

    raw_dir = run_dir / "raw"
    timeline: list[dict[str, Any]] = []

    def add(
        kind: str,
        title: str,
        summary: str,
        *,
        artifacts: list[str] | None = None,
        api_calls: list[str] | None = None,
        decisions: list[str] | None = None,
        data: dict[str, Any] | None = None,
        markers: list[dict[str, Any]] | None = None,
        duration_seconds_hint: float = 1.25,
    ) -> None:
        timeline.append(
            {
                "index": len(timeline) + 1,
                "kind": kind,
                "title": title,
                "summary": summary,
                "artifacts": artifacts or [],
                "api_calls": api_calls or [],
                "decisions": decisions or [],
                "data": data or {},
                "markers": markers or [],
                "duration_seconds_hint": duration_seconds_hint,
            }
        )

    source_usd = _request_input(request, "usd") or assignments.get("source_usd")
    reference_images = _request_input(request, "reference_images") or []
    if isinstance(reference_images, str):
        reference_images = [reference_images]
    reference_files = _request_input(request, "reference_files") or []
    if isinstance(reference_files, str):
        reference_files = [reference_files]

    if request:
        start_summary = "The wrapper launched a child agent against usd-cli with formal asset, reference, and material-library inputs."
    else:
        start_summary = (
            "This trace was reconstructed from an existing usd-cli run directory."
        )

    add(
        "start",
        "Material assignment run started",
        start_summary,
        artifacts=[_rel(run_dir, source_usd)] if source_usd else [],
        decisions=[
            "Use usd-cli as the scene interaction surface.",
            "Use non-destructive material overrides for the assignment.",
        ],
        data={
            "source_usd": source_usd,
            "usd_cli_session_id": request.get("usd_cli_session_id"),
            "runner": request.get("runner"),
        },
    )

    if reference_images:
        add(
            "reference",
            "Reference images loaded",
            "Reference images provide the visual target for material color, finish, and region matching.",
            artifacts=[_rel(run_dir, str(path)) for path in reference_images],
            decisions=[
                "Use reference images as external visual evidence, not as geometry.",
                "Compare them against usd-cli renders from multiple camera views.",
            ],
        )

    if reference_files:
        add(
            "reference",
            "Reference files loaded",
            "Generic reference files provide supplemental visual or material target evidence.",
            artifacts=[_rel(run_dir, str(path)) for path in reference_files],
            decisions=[
                "Use non-image reference files as external evidence, not as geometry.",
                "Compare their material guidance against usd-cli renders from multiple camera views.",
            ],
        )

    add(
        "api_discovery",
        "usd-cli API discovered",
        "The agent fetched the canonical usd-cli API docs before operating on the scene.",
        artifacts=[
            _rel(run_dir, raw_dir / "agent-api.md"),
            _rel(run_dir, raw_dir / "agent-api.json"),
            _rel(run_dir, raw_dir / "openapi.json"),
        ],
        api_calls=["GET /agent-api", "GET /agent-api.json", "GET /openapi.json"],
    )

    tree_summary = _load_json(
        raw_dir / "tree_summary.json",
        default={},
        run_dir=run_dir,
    )
    add(
        "scene_query",
        "Scene hierarchy and bindings queried",
        "The agent enumerated scene prims and queried properties/material bindings before editing.",
        artifacts=[
            _rel(run_dir, raw_dir / "tree_summary.json"),
            _rel(run_dir, raw_dir / "properties_all_candidates.json"),
            _rel(run_dir, raw_dir / "material_bindings_all_candidates.json"),
        ],
        api_calls=[
            "GET /sessions/{session_id}/tree",
            "GET /sessions/{session_id}/properties",
            "GET /sessions/{session_id}/material-binding",
        ],
        data=tree_summary,
    )

    for record in _load_json(
        raw_dir / "initial_render_records.json",
        default=[],
        run_dir=run_dir,
    ):
        image = _record_image(record)
        add(
            "render",
            f"Initial render: {record.get('name', 'view')}",
            f"Rendered direction {record.get('direction', 'session-camera')} to establish the asset's visible material regions.",
            artifacts=[_rel(run_dir, image)] if image else [],
            api_calls=["POST /sessions/{session_id}/render"],
            decisions=[
                "Use fixed initial views to locate large material groups.",
                "Reserve pixel picking for ambiguous regions.",
            ],
            data={"request": record.get("request", {})},
        )

    current_pick_image: str | None = None
    for record in _load_json(
        raw_dir / "pick_records.json",
        default=[],
        run_dir=run_dir,
    ):
        kind = record.get("kind")
        if kind == "command":
            command = record.get("command", "command")
            payload = record.get("payload", {})
            add(
                "camera_command",
                f"Camera command: {command}",
                "The agent moved the usd-cli camera to inspect a target region before picking.",
                api_calls=["POST /sessions/{session_id}/commands"],
                data={"command": command, "payload": payload},
                duration_seconds_hint=0.75,
            )
        elif kind == "render":
            image = _record_image(record)
            current_pick_image = image
            add(
                "render",
                f"Pick camera render: {record.get('name', 'view')}",
                "Rendered the current usd-cli camera state before pixel picking.",
                artifacts=[_rel(run_dir, image)] if image else [],
                api_calls=["POST /sessions/{session_id}/render"],
            )
        elif kind == "pick":
            payload = record.get("payload", {})
            response = record.get("response", {})
            prim_paths = response.get("prim_paths") or []
            label = record.get("label", "pick")
            add(
                "pick",
                f"Pixel pick: {label}",
                f"Picked screen pixel ({payload.get('x')}, {payload.get('y')}) and resolved it to {prim_paths[0] if prim_paths else 'no prim hit'}.",
                artifacts=[_rel(run_dir, current_pick_image)]
                if current_pick_image
                else [],
                api_calls=["POST /sessions/{session_id}/pick"],
                decisions=[
                    "Use picked prim identity to connect visible shape to USD path and material binding."
                ],
                data={"payload": payload, "prim_paths": prim_paths},
                markers=[
                    {
                        "x": payload.get("x"),
                        "y": payload.get("y"),
                        "label": label,
                    }
                ],
                duration_seconds_hint=0.9,
            )

    for record in _load_json(
        raw_dir / "isolation_render_records.json",
        default=[],
        run_dir=run_dir,
    ):
        image = _record_image(record)
        name = record.get("name", "isolation")
        add(
            "isolation",
            f"Isolation render: {name}",
            "The agent isolated an ambiguous prim family and rendered it separately for material evidence.",
            artifacts=[_rel(run_dir, image)] if image else [],
            api_calls=[
                "POST /sessions/{session_id}/commands isolate",
                "POST /sessions/{session_id}/render",
            ],
            decisions=[
                "Use isolation to avoid confusing overlapping geometry with the selected material family."
            ],
            data={"prim_count": len(record.get("paths") or [])},
        )

    test_record = _load_json(
        raw_dir / "test_orange_lift_frame_record.json",
        default={},
        run_dir=run_dir,
    )
    test_image = _record_image(test_record)
    test_orange_render = _contained_run_artifact_reference(
        run_dir,
        run_dir / "evidence_renders" / "test_orange_lift_frame.png",
    )
    if test_record or test_orange_render is not None:
        add(
            "material_test",
            "Material-library binding smoke test",
            "The agent applied a single material override to verify that library-backed material binding worked before bulk assignment.",
            artifacts=[
                _rel(
                    run_dir,
                    test_image or test_orange_render,
                )
            ],
            api_calls=["POST /sessions/{session_id}/commands material_override"],
            decisions=[
                "Proceed with library-backed material overrides after the test render matched the expected orange finish."
            ],
        )

    assignment_summary = _assignments_summary(assignments)
    if assignment_summary:
        add(
            "assignment",
            "Material coverage decisions recorded",
            "The agent recorded visible material coverage decisions, including overrides, preserved materials, and ambiguous selections.",
            artifacts=[
                _rel(run_dir, run_dir / "assignments.json"),
                _rel(run_dir, raw_dir / "override_command_records.json"),
            ],
            api_calls=["POST /sessions/{session_id}/commands material_override"],
            decisions=[
                f"{item['family']}: {item['coverage_status']} -> {item['material_name']} ({item['count']} prims)"
                for item in assignment_summary[:12]
            ],
            data={
                "assignment_count": len(assignments.get("assignments") or []),
                "per_prim_material_assignment_count": assignments.get(
                    "per_prim_material_assignment_count"
                ),
                "coverage": _material_coverage_summary(assignments, counts),
            },
            duration_seconds_hint=2.0,
        )

    for record in _load_json(
        raw_dir / "final_render_records.json",
        default=[],
        run_dir=run_dir,
    ):
        image = _record_image(record)
        add(
            "final_render",
            f"Final render: {record.get('name', 'view')}",
            f"Rendered direction {record.get('direction', 'session-camera')} after all material overrides.",
            artifacts=[_rel(run_dir, image)] if image else [],
            api_calls=["POST /sessions/{session_id}/render"],
            decisions=[
                "Use final views to check for missed prims or visually inconsistent material assignments."
            ],
            data={"request": record.get("request", {})},
        )

    add(
        "finish",
        "Run completed",
        "The observable trace was compiled from usd-cli artifacts, child-agent outputs, and wrapper metadata.",
        artifacts=[
            _rel(run_dir, run_dir / "final_summary.md"),
            _rel(run_dir, run_dir / "api_operation_counts.json"),
        ],
        data={
            "render_count": counts.get("render_count_total"),
            "pick_count": counts.get("pick_calls"),
            "override_count": counts.get("material_override_commands"),
        },
    )
    return timeline


def _build_usd_cli_timeline(
    run_dir: Path,
    request: dict[str, Any],
    assignments: dict[str, Any],
    counts: dict[str, Any],
    agent_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build only backend-neutral and usd-cli-backed timeline entries."""

    timeline: list[dict[str, Any]] = []

    def add(
        kind: str,
        title: str,
        summary: str,
        *,
        artifacts: list[str] | None = None,
        decisions: list[str] | None = None,
        data: dict[str, Any] | None = None,
        duration_seconds_hint: float = 1.0,
    ) -> None:
        timeline.append(
            {
                "index": len(timeline) + 1,
                "kind": kind,
                "title": title,
                "summary": summary,
                "artifacts": artifacts or [],
                "api_calls": [],
                "decisions": decisions or [],
                "data": data or {},
                "markers": [],
                "duration_seconds_hint": duration_seconds_hint,
            }
        )

    source_usd = _request_input(request, "usd") or assignments.get("source_usd")
    route_enabled = _usd_cli_route_enabled(request)
    telemetry_evidence = _has_usd_cli_telemetry_evidence(agent_events)
    if request.get("dry_run"):
        start_summary = (
            "The wrapper prepared a dry-run child prompt with usd-cli as the "
            "scene interaction surface."
        )
    elif route_enabled:
        start_summary = (
            "The wrapper launched a child agent with usd-cli as the scene "
            "interaction surface and an active run-local telemetry route."
        )
    else:
        start_summary = (
            "The wrapper launched a child agent with usd-cli as the scene "
            "interaction surface; telemetry routing was unavailable or "
            "unconfirmed, so execution continued fail-open."
        )
    add(
        "start",
        "Material assignment run started",
        start_summary,
        artifacts=[_rel(run_dir, source_usd)] if source_usd else [],
        decisions=[
            "Use usd-cli for scene inspection, authoring, rendering, and verification.",
            "Keep source USD immutable and save the materialized deliverable under the run directory.",
        ],
        data={
            "source_usd": source_usd,
            "runner": request.get("runner"),
            "telemetry": request.get("telemetry"),
        },
    )

    references = [
        *_as_string_list(_request_input(request, "reference_images")),
        *_as_string_list(_request_input(request, "reference_files")),
    ]
    if references:
        add(
            "reference",
            "Reference evidence loaded",
            "Reference inputs provided the visual and material target.",
            artifacts=[_rel(run_dir, path) for path in references],
            decisions=["Use references as external evidence, not as scene geometry."],
        )

    assignment_summary = _assignments_summary(assignments)
    if assignment_summary:
        add(
            "assignment",
            "Material coverage decisions recorded",
            "The child recorded material decisions and the wrapper verified the canonical artifact contract.",
            artifacts=[_rel(run_dir, run_dir / "assignments.json")],
            decisions=[
                f"{item['family']}: {item['coverage_status']} -> "
                f"{item['material_name']} ({item['count']} prims)"
                for item in assignment_summary[:12]
            ],
            data={
                "assignment_count": len(assignments.get("assignments") or []),
                "coverage": _material_coverage_summary(assignments, counts),
            },
            duration_seconds_hint=2.0,
        )

    final_render_dir = run_dir / "final_renders"
    try:
        final_render_dir_metadata = final_render_dir.lstat()
    except FileNotFoundError:
        final_renders: list[Path] = []
    else:
        final_renders = (
            sorted(final_render_dir.glob("*"))
            if stat.S_ISDIR(final_render_dir_metadata.st_mode)
            and not stat.S_ISLNK(final_render_dir_metadata.st_mode)
            else []
        )
    final_render_artifacts = [
        reference
        for path in final_renders
        if path.suffix.lower() in {".gif", ".jpeg", ".jpg", ".png"}
        and (
            reference := _contained_run_artifact_reference(
                run_dir,
                path,
            )
        )
        is not None
    ]
    if final_render_artifacts:
        add(
            "final_render",
            "Final verification renders recorded",
            "The usd-cli run produced final visual evidence for material verification.",
            artifacts=final_render_artifacts,
            decisions=["Use final renders to verify coverage and visual consistency."],
            duration_seconds_hint=1.5,
        )

    output_usd = run_dir / "output" / "materialized.usda"
    output_reference = _contained_run_artifact_reference(run_dir, output_usd)
    if output_reference is not None:
        add(
            "persistence",
            "Materialized USD saved",
            "usd-cli saved the verified materialized stage under the run directory.",
            artifacts=[output_reference],
        )

    finish_evidence = (
        "usd-cli telemetry, child-agent artifacts, and wrapper verification metadata"
        if telemetry_evidence
        else "child-agent artifacts and wrapper verification metadata"
    )
    add(
        "finish",
        "Run completed",
        f"The observable trace was compiled from {finish_evidence}.",
        artifacts=[
            reference
            for path in (
                run_dir / "final_summary.md",
                run_dir / "api_operation_counts.json",
                run_dir / "run_cost_metrics.json",
            )
            if (
                reference := _contained_run_artifact_reference(
                    run_dir,
                    path,
                )
            )
            is not None
        ],
        data={
            "usd_cli_calls": counts.get(
                "usd_cli_calls_total",
                counts.get("api_operation_count_total"),
            ),
            "render_count": counts.get(
                "render_calls_total",
                counts.get("render_count_total"),
            ),
        },
    )
    return timeline


def _usd_cli_route_enabled(request: dict[str, Any]) -> bool:
    telemetry = request.get("telemetry")
    if not isinstance(telemetry, dict):
        return False
    routing = telemetry.get("routing")
    return isinstance(routing, dict) and routing.get("enabled") is True


def _has_usd_cli_telemetry_evidence(agent_events: list[dict[str, Any]]) -> bool:
    for event in agent_events:
        if (
            not isinstance(event, dict)
            or event.get("event_type") != "usd_cli_invocation"
        ):
            continue
        data = event.get("data")
        if (
            isinstance(data, dict)
            and data.get("schema_version")
            == "content-agents.usd-cli-telemetry-import.v1"
            and data.get("trace_matches_run") is True
            and data.get("parent_matches_run_root") is True
        ):
            return True
    return False


def _current_run_agent_events(
    request: dict[str, Any],
    agent_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Exclude only imported telemetry events from earlier reused-dir runs."""

    if request.get("scene_backend") != "usd-cli":
        return agent_events
    telemetry = request.get("telemetry")
    if not isinstance(telemetry, dict):
        return agent_events
    expected_trace_id = telemetry.get("trace_id")
    if not (
        isinstance(expected_trace_id, str)
        and len(expected_trace_id) == 32
        and expected_trace_id != "0" * 32
        and all(character in "0123456789abcdef" for character in expected_trace_id)
    ):
        return agent_events
    current: list[dict[str, Any]] = []
    for event in agent_events:
        data = event.get("data")
        if not (
            isinstance(data, dict)
            and data.get("schema_version")
            == "content-agents.usd-cli-telemetry-import.v1"
        ):
            current.append(event)
            continue
        event_trace_id = data.get("trace_id") or data.get("expected_trace_id")
        if event_trace_id == expected_trace_id:
            current.append(event)
    return current


def _merge_usd_cli_timeline(
    synthetic: list[dict[str, Any]],
    agent_events: list[dict[str, Any]],
    child_commands: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Place source-timed invocations before derived artifacts and completion."""

    finish = [event for event in synthetic if event.get("kind") == "finish"]
    body = [event for event in synthetic if event.get("kind") != "finish"]
    split_at = next(
        (
            index
            for index, event in enumerate(body)
            if event.get("kind") in {"assignment", "final_render", "persistence"}
        ),
        len(body),
    )
    early_agent_events: list[dict[str, Any]] = []
    ingestion_events: list[dict[str, Any]] = []
    for event in agent_events:
        data = event.get("data")
        is_ingestion_summary = (
            isinstance(data, dict)
            and data.get("schema_version")
            == "content-agents.usd-cli-telemetry-import.v1"
            and event.get("kind") != "agent_usd_cli_invocation"
        )
        (ingestion_events if is_ingestion_summary else early_agent_events).append(event)
    return [
        *body[:split_at],
        *early_agent_events,
        *body[split_at:],
        *child_commands,
        *ingestion_events,
        *finish,
    ]


def _timeline_from_agent_events(
    run_dir: Path,
    agent_events: list[dict[str, Any]],
    start_index: int,
) -> list[dict[str, Any]]:
    wrapper_event_types = {
        "run_created",
        "prompt_written",
        "usd_cli_started",
        "usd_cli_ready",
        "child_agent_finished",
        "child_agent_failed",
        "usd_cli_stopped",
    }
    timeline = []
    index = start_index
    ordered_events = sorted(
        enumerate(agent_events),
        key=lambda item: (
            str(item[1].get("time") or "9999-12-31T23:59:59+00:00"),
            item[0],
        ),
    )
    for _, event in ordered_events:
        if event.get("schema_version") == MEMORY_EVENT_SCHEMA_VERSION:
            observation = (
                event.get("observation")
                if isinstance(event.get("observation"), dict)
                else {}
            )
            memory_kind = str(event.get("kind") or "event")
            event_type = f"memory_{memory_kind}"
            if memory_kind == "observation":
                outcome = (
                    observation.get("outcome")
                    if isinstance(observation.get("outcome"), dict)
                    else {}
                )
                summary = str(outcome.get("summary") or "Observation recorded.")
                data = {
                    "observation_id": observation.get("observation_id"),
                    "operation": (
                        observation.get("interaction", {}).get("operation")
                        if isinstance(observation.get("interaction"), dict)
                        else None
                    ),
                    "outcome": outcome.get("classification"),
                    "artifact_roles": [
                        artifact.get("role")
                        for artifact in observation.get("artifacts") or []
                        if isinstance(artifact, dict) and artifact.get("role")
                    ],
                }
            else:
                observation_id = observation.get("observation_id")
                summary = (
                    f"Memory {memory_kind} event for observation {observation_id}."
                    if observation_id
                    else f"Memory {memory_kind} event recorded."
                )
                data = {"observation_id": observation_id}
            index += 1
            timeline.append(
                {
                    "index": index,
                    "kind": f"agent_{event_type}",
                    "title": _title_from_event(
                        event_type, str(event.get("phase") or "memory")
                    ),
                    "summary": summary,
                    "artifacts": [],
                    "api_calls": [],
                    "decisions": [],
                    "data": data,
                    "markers": [],
                    "duration_seconds_hint": 0.1,
                }
            )
            continue
        event_type = str(event.get("event_type") or "event")
        if event_type in wrapper_event_types:
            continue
        artifacts = [
            reference
            for artifact in event.get("artifacts") or []
            if artifact
            and (reference := _contained_run_artifact_reference(run_dir, artifact))
            is not None
        ]
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        api_calls = data.get("api_calls") if isinstance(data, dict) else []
        material_names = data.get("material_names") if isinstance(data, dict) else []
        prim_paths = data.get("prim_paths") if isinstance(data, dict) else []
        decisions = []
        if material_names:
            decisions.append(f"Material(s): {', '.join(map(str, material_names))}")
        if prim_paths:
            decisions.append(f"Prim paths referenced: {len(prim_paths)}")
        uncertainty = data.get("uncertainty") if isinstance(data, dict) else None
        if uncertainty:
            decisions.append(f"Uncertainty: {uncertainty}")
        index += 1
        timeline.append(
            {
                "index": index,
                "kind": f"agent_{event_type}",
                "title": _title_from_event(event_type, str(event.get("phase") or "")),
                "summary": str(event.get("summary") or ""),
                "artifacts": artifacts,
                "api_calls": api_calls if isinstance(api_calls, list) else [],
                "decisions": decisions,
                "data": data,
                "time": event.get("time"),
                "markers": [],
                "duration_seconds_hint": _duration_hint_for_event(
                    event_type, artifacts
                ),
            }
        )
    return timeline


def _timeline_from_child_commands(
    run_dir: Path,
    child_commands: list[dict[str, Any]],
    start_index: int,
) -> list[dict[str, Any]]:
    if not child_commands:
        return []

    failed = [
        command
        for command in child_commands
        if command.get("exit_code") not in {None, 0}
    ]
    glue = _matching_child_activity(
        child_commands,
        [
            "python - <<",
            "python3 - <<",
            "node - <<",
            "cat <<",
            "temporary script",
            "glue code",
            "one-off script",
        ],
    )
    patches = _matching_child_activity(
        child_commands,
        [
            "apply_patch",
            "*** begin patch",
            "diff --git",
            "git apply",
            "patching file",
            "applied patch",
        ],
    )
    source_artifacts = sorted(
        {
            str(command.get("source_artifact"))
            for command in child_commands
            if command.get("source_artifact")
        }
    )
    return [
        {
            "index": start_index + 1,
            "kind": "child_commands",
            "title": "Child SDK command executions captured",
            "summary": (
                f"The SDK item log recorded {len(child_commands)} child command "
                f"execution(s), including {len(failed)} failed command(s), "
                f"{len(glue)} glue-code indicator(s), and {len(patches)} "
                "patch indicator(s)."
            ),
            "artifacts": [
                reference
                for artifact in source_artifacts
                if (reference := _contained_run_artifact_reference(run_dir, artifact))
                is not None
            ],
            "api_calls": [],
            "decisions": [
                _clip(str(command.get("command") or ""))
                for command in child_commands[:6]
                if command.get("command")
            ],
            "data": {
                "command_count": len(child_commands),
                "failed_command_count": len(failed),
                "glue_indicator_count": len(glue),
                "patch_indicator_count": len(patches),
            },
            "markers": [],
            "duration_seconds_hint": 1.0,
        }
    ]


def _renumber_timeline(timeline: list[dict[str, Any]]) -> None:
    for index, event in enumerate(timeline, 1):
        event["index"] = index


def _title_from_event(event_type: str, phase: str) -> str:
    readable_type = event_type.replace("_", " ").title()
    readable_phase = phase.replace("_", " ").title()
    return f"{readable_type}: {readable_phase}" if readable_phase else readable_type


def _duration_hint_for_event(event_type: str, artifacts: list[str]) -> float:
    if _first_image(artifacts):
        return 1.25
    if event_type in {"decision", "assignment", "verification"}:
        return 1.0
    return 0.75


def _build_run_retrospective(
    *,
    run_dir: Path,
    request: dict[str, Any],
    assignments: dict[str, Any],
    counts: dict[str, Any],
    agent_events: list[dict[str, Any]],
    child_commands: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compile an evidence-based review of the run."""

    went_well: list[str] = []
    did_not_go_well: list[str] = []
    followups: list[str] = []

    if request:
        went_well.append(
            "The wrapper captured workflow inputs, constraints, and runner metadata in request.json."
        )
    else:
        followups.append(
            "No request.json was found; this trace was reconstructed from partial artifacts."
        )

    render_count = _int_or_none(
        counts.get("render_calls_total", counts.get("render_count_total"))
    )
    if render_count:
        went_well.append(f"The run produced {render_count} render operation(s).")

    pick_count = _int_or_none(counts.get("pick_calls"))
    if pick_count:
        went_well.append(f"The run used {pick_count} pixel pick operation(s).")

    override_count = _int_or_none(counts.get("material_override_commands"))
    if override_count:
        went_well.append(
            f"The run applied {override_count} material override command(s)."
        )

    assignments_summary = _assignments_summary(assignments)
    if assignments_summary:
        assignment_count = len(assignments.get("assignments") or [])
        per_prim_count = assignments.get(
            "per_prim_material_assignment_count",
            assignment_count,
        )
        went_well.append(
            f"The run recorded {assignment_count} material family assignment(s) covering {per_prim_count} prim override(s)."
        )
        missing_status_count = sum(
            1
            for item in assignments_summary
            if item.get("coverage_status") == "unknown"
        )
        if missing_status_count:
            did_not_go_well.append(
                f"{missing_status_count} material assignment(s) did not record a coverage_status."
            )

    coverage = _material_coverage_summary(assignments, counts)
    if coverage["present"]:
        candidate_count = coverage.get("candidate_visible_prim_count")
        decision_count = coverage.get("material_decision_prim_count")
        if candidate_count is None or decision_count is None:
            did_not_go_well.append(
                "The material coverage report is present but missing candidate or decision counts."
            )
            followups.append(
                "Record coverage.candidate_visible_prim_count and coverage.material_decision_prim_count in assignments.json."
            )
        elif candidate_count <= 0:
            did_not_go_well.append(
                "The material coverage report did not identify any canonical material candidate prims."
            )
        elif decision_count >= candidate_count:
            went_well.append(
                f"The material coverage report accounts for {decision_count}/{candidate_count} canonical material candidate prim(s)."
            )
        else:
            did_not_go_well.append(
                f"The material coverage report accounts for only {decision_count}/{candidate_count} canonical material candidate prim(s)."
            )
            followups.append(
                "Improve material coverage before accepting this run; every canonical material candidate should be assigned, preserved with rationale, or marked ambiguous."
            )

        assigned_count = coverage.get("material_assignment_prim_count") or 0
        preserved_count = coverage.get("preserved_existing_prim_count") or 0
        ambiguous_count = coverage.get("ambiguous_unassigned_prim_count") or 0
        if assigned_count or preserved_count or ambiguous_count:
            went_well.append(
                f"Coverage split: {assigned_count} assigned, {preserved_count} preserved, {ambiguous_count} ambiguous prim(s)."
            )
        if ambiguous_count:
            did_not_go_well.append(
                f"The material coverage report leaves {ambiguous_count} prim(s) ambiguous/unassigned."
            )
    elif assignments_summary:
        did_not_go_well.append(
            "assignments.json does not include material coverage accounting; it may only report changed override groups."
        )
        followups.append(
            "Require future material-assignment runs to include assignments.coverage and per-family coverage_status."
        )

    final_review = assignments.get("final_review")
    if isinstance(final_review, dict):
        issues_found = _count_items(final_review.get("issues_found"))
        issues_fixed = _count_items(final_review.get("issues_fixed"))
        unresolved = final_review.get("unresolved_issues")
        unresolved_count = _count_items(unresolved) or 0
        if issues_found is None:
            did_not_go_well.append(
                "The final material review did not record issues_found."
            )
        elif issues_found == 0:
            went_well.append(
                "The final material review reported no remaining coverage or visual issues."
            )
        else:
            went_well.append(
                f"The final material review found {issues_found} issue(s) and fixed {issues_fixed or 0}."
            )
        if unresolved_count:
            did_not_go_well.append(
                f"The final material review left {unresolved_count} unresolved issue(s)."
            )
    elif assignments_summary:
        did_not_go_well.append(
            "assignments.json does not include final_review results from a remediation pass."
        )
        followups.append(
            "Run a final review/remediation pass before accepting material assignment artifacts."
        )

    visual_quality = assignments.get("visual_quality_assessment")
    if not isinstance(visual_quality, dict):
        if assignments_summary:
            did_not_go_well.append(
                "assignments.json does not include visual_quality_assessment results."
            )
            followups.append(
                "Run a visual quality assessment against final renders before accepting material assignment artifacts."
            )
    else:
        status = str(visual_quality.get("status") or "unknown")
        issues_found = _count_items(visual_quality.get("issues_found"))
        issues_fixed = _count_items(visual_quality.get("issues_fixed")) or 0
        unresolved_count = _count_items(visual_quality.get("unresolved_issues")) or 0
        checked_views = _count_items(visual_quality.get("checked_views")) or 0
        if checked_views:
            went_well.append(
                f"The visual quality assessment checked {checked_views} final render view(s)."
            )
        if issues_found is None:
            did_not_go_well.append(
                "The visual quality assessment did not record issues_found."
            )
        elif issues_found == 0 and unresolved_count == 0:
            went_well.append(
                f"The visual quality assessment passed with status `{status}`."
            )
        else:
            went_well.append(
                f"The visual quality assessment found {issues_found} issue(s) and fixed {issues_fixed}."
            )
        if unresolved_count:
            did_not_go_well.append(
                f"The visual quality assessment left {unresolved_count} unresolved issue(s)."
            )
        if status == "unresolved_issues" and unresolved_count == 0:
            did_not_go_well.append(
                "The visual quality assessment status is unresolved_issues but no unresolved issues were listed."
            )

    if any(event.get("kind") == "final_render" for event in timeline):
        went_well.append("Final verification render artifacts were recorded.")

    if child_commands:
        went_well.append(
            f"The SDK item log recorded {len(child_commands)} child command execution(s) for audit."
        )
        failed_commands = [
            command
            for command in child_commands
            if command.get("exit_code") not in {None, 0}
        ]
        if failed_commands:
            did_not_go_well.append(
                f"{len(failed_commands)} child command execution(s) exited non-zero."
            )
            for command in failed_commands[:4]:
                did_not_go_well.append(
                    f"Failed child command: {_clip(str(command.get('command') or ''))}"
                )
    elif request.get("runner") in {"codex", "claude"} and not request.get("dry_run"):
        did_not_go_well.append(
            "No SDK command execution items were captured; trace cannot audit child shell/tool activity."
        )

    returncode = _child_returncode(agent_events)
    if returncode is not None:
        if returncode == 0:
            went_well.append("The child agent runner completed successfully.")
        else:
            did_not_go_well.append(
                f"The child agent runner exited with return code {returncode}."
            )
    else:
        if request.get("dry_run"):
            followups.append(
                "No child-agent completion event was recorded; this is expected for dry runs."
            )
        else:
            did_not_go_well.append(
                "No child-agent completion event was recorded for this non-dry run."
            )

    if any(event.get("event_type") == "usd_cli_started" for event in agent_events):
        if any(event.get("event_type") == "usd_cli_stopped" for event in agent_events):
            went_well.append("The wrapper-started usd-cli sidecar was stopped.")
        else:
            did_not_go_well.append(
                "The wrapper started usd-cli but no usd_cli_stopped event was recorded."
            )

    material_cap_warning = counts.get("material_assignment_cap_warning")
    if material_cap_warning:
        did_not_go_well.append(str(material_cap_warning))

    for event in agent_events:
        event_type = str(event.get("event_type") or "")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if event_type in {"warning", "error", "child_agent_failed"}:
            summary = str(event.get("summary") or event_type)
            if data.get("error"):
                summary = f"{summary}: {data['error']}"
            did_not_go_well.append(summary)

    child_output = _load_text(
        run_dir / "child-output.log",
        run_dir=run_dir,
    )
    activity_text = _child_activity_text(child_output, child_commands)
    patches = _build_detected_activity_report(
        activity_text,
        label="repository patch activity",
        summary="The child process appears to have patched repository code during the run.",
        patterns=[
            "apply_patch",
            "*** begin patch",
            "diff --git",
            "git apply",
            "patching file",
            "applied patch",
        ],
    )
    glue = _build_detected_activity_report(
        activity_text,
        label="ad hoc glue-code activity",
        summary="The child process appears to have generated one-off shell or script glue during the run.",
        patterns=[
            "python - <<",
            "python3 - <<",
            "node - <<",
            "cat <<",
            "temporary script",
            "glue code",
            "one-off script",
        ],
    )

    error_lines = _matching_lines(
        _child_activity_text(child_output, child_commands, include_commands=False),
        [
            "traceback",
            "http error",
            '"detail":"',
            '"detail": "',
            "connectionrefusederror",
            "not found",
            "timeout",
            "runner failed",
        ],
    )
    for line in error_lines[:4]:
        did_not_go_well.append(f"Child output reported: {line}")

    if patches["detected"]:
        followups.append(
            "Review code patches made during the run and decide whether they should become product changes or stay as experiment notes."
        )
    if glue["detected"]:
        if request.get("scene_backend") == "usd-cli":
            followups.append(
                "Convert useful one-off glue code into usd-cli or "
                "content-workflow-cli commands so future runs do not need to "
                "improvise it."
            )
        else:
            followups.append(
                "Convert useful one-off glue code into usd-cli or "
                "content-workflow-cli APIs so future runs do not need to "
                "improvise it."
            )
    if not went_well:
        went_well.append("Trace files were generated for reviewer inspection.")
    if not did_not_go_well:
        did_not_go_well.append(
            "No runner failures, warning events, or error patterns were detected in the recorded artifacts."
        )

    return {
        "schema_version": RETROSPECTIVE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "summary": {
            "timeline_event_count": len(timeline),
            "runner": request.get("runner"),
            "returncode": returncode,
        },
        "what_went_well": _dedupe_preserve_order(went_well),
        "what_did_not_go_well": _dedupe_preserve_order(did_not_go_well),
        "patches_or_code_changes": patches,
        "generated_glue_code": glue,
        "recommended_followups": _dedupe_preserve_order(followups),
    }


def _build_detected_activity_report(
    text: str,
    *,
    label: str,
    summary: str,
    patterns: list[str],
) -> dict[str, Any]:
    evidence = _matching_lines(text, patterns)
    return {
        "detected": bool(evidence),
        "summary": summary
        if evidence
        else f"No {label} detected in child-output.log or SDK item logs.",
        "evidence": evidence[:8],
    }


def _load_child_command_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(_load_codex_child_command_records(run_dir))
    records.extend(_load_claude_child_command_records(run_dir))
    return records


def _safe_child_item_paths(run_dir: Path) -> list[Path]:
    """List direct child item logs without traversing a replaced ``raw`` dir."""

    raw_dir = run_dir / "raw"
    try:
        metadata = raw_dir.lstat()
    except OSError:
        return []
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        return []
    paths: list[Path] = []
    try:
        for path in raw_dir.iterdir():
            if not path.name.endswith("_items.json"):
                continue
            paths.append(path)
            if len(paths) > _MAX_CHILD_ITEM_FILES:
                return []
    except OSError:
        return []
    return sorted(paths)


def _load_codex_child_command_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _safe_child_item_paths(run_dir):
        items = _load_json(path, default=[], run_dir=run_dir)
        if isinstance(items, dict):
            items = items.get("items") or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "command_execution" and "command" not in item:
                continue
            command = item.get("command")
            if not isinstance(command, str) or not command.strip():
                continue
            output = item.get("aggregated_output")
            if not isinstance(output, str):
                output = (
                    item.get("output") if isinstance(item.get("output"), str) else ""
                )
            records.append(
                {
                    "id": item.get("id"),
                    "type": item.get("type"),
                    "status": item.get("status"),
                    "command": command,
                    "exit_code": _int_or_none(item.get("exit_code")),
                    "output_excerpt": _clip_multiline(output),
                    "source_artifact": str(path),
                }
            )
    return records


def _load_claude_child_command_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    by_tool_use_id: dict[str, dict[str, Any]] = {}
    for path in _safe_child_item_paths(run_dir):
        items = _load_json(path, default=[], run_dir=run_dir)
        if isinstance(items, dict):
            items = items.get("items") or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "assistant":
                for block in _claude_content_blocks(item):
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    command = _claude_tool_command(block)
                    if not command:
                        continue
                    record = {
                        "id": block.get("id"),
                        "type": block.get("name") or "tool_use",
                        "status": item.get("type"),
                        "command": command,
                        "exit_code": None,
                        "output_excerpt": "",
                        "source_artifact": str(path),
                    }
                    records.append(record)
                    tool_use_id = block.get("id")
                    if isinstance(tool_use_id, str):
                        by_tool_use_id[tool_use_id] = record
            elif item.get("type") == "user":
                for block in _claude_content_blocks(item):
                    if (
                        not isinstance(block, dict)
                        or block.get("type") != "tool_result"
                    ):
                        continue
                    tool_use_id = block.get("tool_use_id")
                    if not isinstance(tool_use_id, str):
                        continue
                    record = by_tool_use_id.get(tool_use_id)
                    if record is not None:
                        record["output_excerpt"] = _clip_multiline(
                            _claude_tool_result_text(block.get("content"))
                        )
    return records


def _claude_content_blocks(item: dict[str, Any]) -> list[object]:
    message = item.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, list):
        return content
    return []


def _claude_tool_command(block: dict[str, Any]) -> str | None:
    name = str(block.get("name") or "")
    tool_input = block.get("input")
    if not isinstance(tool_input, dict):
        return None
    if name == "Bash":
        command = tool_input.get("command") or tool_input.get("cmd")
        return str(command).strip() if command else None
    if name in {"Write", "Edit", "MultiEdit"}:
        file_path = tool_input.get("file_path") or tool_input.get("path")
        return f"{name} {file_path}".strip() if file_path else name
    return None


def _claude_tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            chunks.append(block["text"])
    return "\n".join(chunks)


def _child_activity_text(
    child_output: str,
    child_commands: list[dict[str, Any]],
    *,
    include_commands: bool = True,
) -> str:
    lines: list[str] = []
    if child_output:
        for line in child_output.splitlines():
            lines.append(f"child-output.log: {line}")
    for index, command in enumerate(child_commands, 1):
        source = Path(str(command.get("source_artifact") or "sdk-items")).name
        command_text = str(command.get("command") or "")
        if include_commands and command_text:
            lines.append(f"{source} command {index}: {command_text}")
        output = str(command.get("output_excerpt") or "")
        for line in output.splitlines():
            lines.append(f"{source} output {index}: {line}")
    return "\n".join(lines)


def _matching_child_activity(
    child_commands: list[dict[str, Any]],
    patterns: list[str],
) -> list[str]:
    return _matching_lines(_child_activity_text("", child_commands), patterns)


def _child_returncode(agent_events: list[dict[str, Any]]) -> int | None:
    for event in reversed(agent_events):
        if event.get("event_type") not in {
            "child_agent_finished",
            "child_agent_failed",
        }:
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        return _int_or_none(data.get("returncode"))
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _count_items(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return 1 if value else 0
    if isinstance(value, list | tuple | set):
        return len(value)
    return None


def _matching_lines(text: str, patterns: list[str], *, limit: int = 12) -> list[str]:
    lowered_patterns = [pattern.lower() for pattern in patterns]
    matches: list[str] = []
    for line in text.splitlines():
        lowered_line = line.lower()
        if not any(pattern in lowered_line for pattern in lowered_patterns):
            continue
        clipped = _clip(line.strip())
        if clipped and clipped not in matches:
            matches.append(clipped)
        if len(matches) >= limit:
            break
    return matches


def _clip(value: str, *, limit: int = 240) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _clip_multiline(value: str, *, limit: int = 1600) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _build_replay_manifest(
    run_dir: Path, timeline: list[dict[str, Any]]
) -> dict[str, Any]:
    frames = []
    for event in timeline:
        image_artifact = _first_safe_image(run_dir, event.get("artifacts") or [])
        if image_artifact is None and event["kind"] not in {
            "start",
            "reference",
            "assignment",
            "finish",
        }:
            continue
        frames.append(
            {
                "index": event["index"],
                "kind": event["kind"],
                "title": event["title"],
                "caption": event["summary"],
                "image_path": image_artifact,
                "markers": event.get("markers", []),
                "duration_seconds": event.get("duration_seconds_hint", 1.25),
                "decisions": event.get("decisions", []),
            }
        )
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "frames": frames,
    }


def _render_trace_markdown(trace: dict[str, Any]) -> str:
    request = trace.get("request") or {}
    usd_cli_backend = request.get("scene_backend") == "usd-cli"
    telemetry_evidence = _has_usd_cli_telemetry_evidence(
        trace.get("agent_events") or []
    )
    if usd_cli_backend:
        evidence_source = (
            "usd-cli telemetry, child-agent artifacts, and wrapper verification metadata"
            if telemetry_evidence
            else "child-agent artifacts and wrapper verification metadata"
        )
    else:
        evidence_source = "usd-cli artifacts"
    lines = [
        "# Content Agent Operation Trace",
        "",
        (
            "This trace contains observable evidence and decision summaries "
            f"reconstructed from {evidence_source}. It does not include private "
            "model chain-of-thought."
        ),
        "",
        f"- Workflow: `{trace.get('workflow')}`",
        f"- Run directory: `{trace.get('run_dir')}`",
    ]
    stats = trace.get("stats") or {}
    if stats:
        operation_count = stats.get(
            "usd_cli_calls_total",
            stats.get("api_operation_count_total", "unknown"),
        )
        render_count = stats.get(
            "render_calls_total",
            stats.get("render_count_total", "unknown"),
        )
        lines.extend(
            [
                (
                    f"- {'usd-cli invocations' if usd_cli_backend else 'API operations tracked'}: "
                    f"`{operation_count}`"
                ),
                f"- Render operations: `{render_count}`",
            ]
        )
        if not usd_cli_backend:
            lines.extend(
                [
                    f"- Pixel picks: `{stats.get('pick_calls', 'unknown')}`",
                    f"- Material override commands: `{stats.get('material_override_commands', 'unknown')}`",
                ]
            )
    retrospective = trace.get("run_retrospective") or {}
    if retrospective:
        lines.extend(["", "## Run Retrospective"])
        _append_markdown_list(
            lines,
            "What went well",
            retrospective.get("what_went_well") or [],
        )
        _append_markdown_list(
            lines,
            "What did not go well",
            retrospective.get("what_did_not_go_well") or [],
        )
        _append_activity_markdown(
            lines,
            "Patches or code changes",
            retrospective.get("patches_or_code_changes") or {},
        )
        _append_activity_markdown(
            lines,
            "Generated glue code",
            retrospective.get("generated_glue_code") or {},
        )
        _append_markdown_list(
            lines,
            "Recommended followups",
            retrospective.get("recommended_followups") or [],
        )
    lines.extend(["", "## Timeline"])
    for event in trace.get("timeline") or []:
        lines.append(f"{event['index']}. **{event['title']}**")
        lines.append(f"   - Kind: `{event['kind']}`")
        lines.append(f"   - Evidence: {event['summary']}")
        for artifact in event.get("artifacts") or []:
            lines.append(f"   - Artifact: `{artifact}`")
        for api_call in event.get("api_calls") or []:
            lines.append(f"   - API: `{api_call}`")
        for decision in (event.get("decisions") or [])[:6]:
            lines.append(f"   - Decision: {decision}")
        for marker in event.get("markers") or []:
            lines.append(
                f"   - Pick marker: `{marker.get('label')}` at ({marker.get('x')}, {marker.get('y')})"
            )
        lines.append("")
    return "\n".join(lines)


def _append_markdown_list(lines: list[str], title: str, items: list[str]) -> None:
    lines.append(f"### {title}")
    if not items:
        lines.append("- None recorded.")
        return
    for item in items:
        lines.append(f"- {item}")


def _append_activity_markdown(
    lines: list[str], title: str, activity: dict[str, Any]
) -> None:
    lines.append(f"### {title}")
    detected = bool(activity.get("detected"))
    lines.append(f"- Detected: `{str(detected).lower()}`")
    summary = activity.get("summary")
    if summary:
        lines.append(f"- Summary: {summary}")
    for evidence in activity.get("evidence") or []:
        lines.append(f"- Evidence: `{evidence}`")


def _assignments_summary(assignments: dict[str, Any]) -> list[dict[str, Any]]:
    summary = []
    for item in assignments.get("assignments") or []:
        summary.append(
            {
                "family": item.get("family"),
                "coverage_status": item.get("coverage_status") or "unknown",
                "material_name": item.get("material_name") or "unassigned",
                "count": len(item.get("prim_paths") or []),
            }
        )
    return summary


def _material_coverage_summary(
    assignments: dict[str, Any], counts: dict[str, Any]
) -> dict[str, Any]:
    coverage = assignments.get("coverage")
    present = isinstance(coverage, dict)
    source = coverage if present else {}
    candidate_count = _int_or_none(source.get("candidate_visible_prim_count"))
    decision_count = _int_or_none(source.get("material_decision_prim_count"))
    assigned_count = _int_or_none(source.get("material_assignment_prim_count"))
    preserved_count = _int_or_none(source.get("preserved_existing_prim_count"))
    ambiguous_count = _int_or_none(source.get("ambiguous_unassigned_prim_count"))

    if candidate_count is None:
        candidate_count = _int_or_none(counts.get("coverage_candidate_visible_prims"))
    if decision_count is None:
        decision_count = _int_or_none(counts.get("coverage_material_decision_prims"))
    if decision_count is None:
        parts = [
            value
            for value in [assigned_count, preserved_count, ambiguous_count]
            if value is not None
        ]
        if parts:
            decision_count = sum(parts)

    return {
        "present": present,
        "candidate_visible_prim_count": candidate_count,
        "material_decision_prim_count": decision_count,
        "material_assignment_prim_count": assigned_count,
        "preserved_existing_prim_count": preserved_count,
        "ambiguous_unassigned_prim_count": ambiguous_count,
        "coverage_notes": source.get("coverage_notes") if present else None,
    }


def _request_input(request: dict[str, Any], key: str) -> Any:
    inputs = request.get("inputs")
    if isinstance(inputs, dict):
        return inputs.get(key)
    if key == "usd":
        return request.get("source_scene")
    references = request.get("references")
    if isinstance(references, dict):
        if key == "reference_images":
            return references.get("images")
        if key == "reference_files":
            return references.get("files")
    if key in {"materials_yaml", "materials_usd"}:
        tasks = request.get("tasks")
        if isinstance(tasks, list):
            for task in tasks:
                if not isinstance(task, dict) or task.get("domain") != "material":
                    continue
                task_inputs = task.get("inputs")
                if isinstance(task_inputs, dict):
                    return task_inputs.get(key)
    return None


def _read_bounded_trace_artifact(
    path: Path,
    *,
    run_dir: Path,
    max_bytes: int,
) -> bytes:
    """Read one run-local regular file through a no-follow descriptor chain."""

    if max_bytes < 0:
        raise ValueError("Trace input byte limit must be non-negative.")
    root = run_dir.expanduser().resolve(strict=True)
    relative = _contained_relative_path(root, path)
    budget = _ACTIVE_TRACE_INPUT_BUDGET.get()
    effective_limit = max_bytes
    if budget is not None:
        effective_limit = min(effective_limit, budget.remaining_bytes)
    if os.name != "posix":
        opened = read_contained_artifact(
            root,
            root / relative,
            max_bytes=effective_limit,
            capture_bytes=True,
        )
        assert opened.data is not None
        if budget is not None:
            budget.consume(len(opened.data))
        return opened.data

    parent_fd = _open_contained_trace_directory(
        root,
        relative.parent,
        create=False,
    )
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_fd = os.open(relative.name, file_flags, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Trace input is not a regular file: {path}")
        if metadata.st_size > effective_limit:
            raise ValueError(
                f"Trace input exceeds its byte limit: {path} "
                f"({metadata.st_size} > {effective_limit})"
            )
        with os.fdopen(file_fd, "rb") as stream:
            file_fd = -1
            payload = stream.read(effective_limit + 1)
        if len(payload) > effective_limit:
            raise ValueError(f"Trace input grew beyond its byte limit: {path}")
        if budget is not None:
            budget.consume(len(payload))
        return payload
    finally:
        if file_fd >= 0:
            os.close(file_fd)


def _load_json(path: Path, *, default: Any, run_dir: Path) -> Any:
    try:
        payload = _read_bounded_trace_artifact(
            path,
            run_dir=run_dir,
            max_bytes=_MAX_JSON_ARTIFACT_BYTES,
        )
        return json.loads(payload)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return default


def _load_json_object(path: Path, *, run_dir: Path) -> dict[str, Any]:
    loaded = _load_json(path, default={}, run_dir=run_dir)
    return loaded if isinstance(loaded, dict) else {}


def _load_jsonl(path: Path, *, run_dir: Path) -> list[dict[str, Any]]:
    try:
        payload = _read_bounded_trace_artifact(
            path,
            run_dir=run_dir,
            max_bytes=_MAX_JSONL_ARTIFACT_BYTES,
        )
        text = payload.decode("utf-8")
    except (OSError, UnicodeError, ValueError):
        return []
    events = []
    records_seen = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        records_seen += 1
        if records_seen > _MAX_JSONL_RECORDS:
            return []
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            events.append(loaded)
    return events


def _load_text(path: Path, *, run_dir: Path) -> str:
    try:
        payload = _read_bounded_trace_artifact(
            path,
            run_dir=run_dir,
            max_bytes=_MAX_TEXT_ARTIFACT_BYTES,
        )
        return payload.decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def _record_image(record: dict[str, Any]) -> str | None:
    response = record.get("response") or record.get("render") or record
    if not isinstance(response, dict):
        return None
    copied = response.get("copied_image_path")
    if copied:
        return str(copied)
    image = response.get("image_path")
    return str(image) if image else None


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def _first_image(artifacts: list[str]) -> str | None:
    for artifact in artifacts:
        suffix = Path(artifact).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return artifact
    return None


def _contained_run_artifact_reference(
    run_dir: Path,
    path: str | Path,
    *,
    image: bool = False,
) -> str | None:
    """Return one verified run-relative artifact reference, or reject it."""

    try:
        resolved = contained_regular_file(
            run_dir,
            path,
            max_bytes=128 * 1024 * 1024 if image else None,
            image=image,
        )
    except (OSError, ValueError):
        return None
    return resolved.relative_to(run_dir.resolve()).as_posix()


def _first_safe_image(run_dir: Path, artifacts: list[str]) -> str | None:
    """Select a replay image without following a child-authored run symlink."""

    root = run_dir.resolve()
    for artifact in artifacts:
        path = Path(artifact)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        candidate = path if path.is_absolute() else root / path
        try:
            candidate.relative_to(root)
        except ValueError:
            # External reference inputs are allowed in reference frames, but they
            # must themselves be regular non-symlink files.
            try:
                if candidate.is_symlink() or not candidate.is_file():
                    continue
            except OSError:
                continue
            return str(candidate)
        reference = _contained_run_artifact_reference(root, candidate)
        if reference is not None:
            return reference
    return None


def _rel(run_dir: Path, path: str | Path | None) -> str:
    if path is None:
        return ""
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    try:
        return str(path_obj.relative_to(run_dir))
    except ValueError:
        return str(path_obj)
