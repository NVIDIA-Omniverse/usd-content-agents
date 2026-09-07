# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The usd-cli daemon.

One persistent process holds a registry of named `usd_core.Session`s and dispatches
commands to them. Requests carry an optional session name (the CLI's global `--session`
flag); requests without one share the default session, which preserves the original
single-session contract. The CLI discovers the daemon via `.usd-cli/server.json`. This file
is a thin transport shell: it does no scene logic itself — it forwards to a Session.
"""

from __future__ import annotations

import asyncio
import copy
import ipaddress
import json
import logging
import os
import re
import secrets
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from usd_core.config import Config, load_config
from usd_core.models import Issue, Response
from usd_core.session import Session

# `from __future__ import annotations` makes endpoint annotations strings that
# FastAPI resolves against MODULE globals — `Request` must live here, not inside
# build_app, or the `/cmd` request parameter degrades into a required query field
# (every command then 422s). Guarded: fastapi is a server-extra dependency and
# importing this module must stay possible without it (build_app fails cleanly).
try:
    from fastapi import Request
except ImportError:  # pragma: no cover - CLI-only installs never call build_app
    Request = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

MAX_SERVER_STATE_BYTES = 64 * 1024

# Wall-clock of the last command IN ANY SESSION; the idle watcher reads it to
# auto-shutdown (PR-9.5).
_LAST_ACTIVITY = [time.monotonic()]
# Commands currently executing (or queued on a session lock), totaled across ALL
# sessions; the idle watcher must never shut the daemon down mid-render in any session,
# and /health reports it so a slow-but-alive daemon is distinguishable from a dead one.
_IN_FLIGHT = [0]
# Guards _IN_FLIGHT + _LAST_ACTIVITY (+ the per-app busy_sessions map) as ONE unit.
# The idle watcher must never observe _IN_FLIGHT == 0 paired with a stale
# _LAST_ACTIVITY: that window let a command longer than idle_timeout finish and have
# the daemon exit before its response was returned (see _exit_busy/_should_idle_exit).
_BUSY_LOCK = threading.Lock()


def _enter_busy(
    busy_sessions: dict[str, dict], session_name: str, command: str
) -> None:
    """Record an accepted command: stamp activity and bump the in-flight counters,
    atomically w.r.t. the idle watcher."""
    with _BUSY_LOCK:
        _LAST_ACTIVITY[0] = time.monotonic()  # reset only for an accepted command
        _IN_FLIGHT[0] += 1
        entry = busy_sessions.setdefault(
            session_name, {"count": 0, "command": None, "since": 0.0, "queued": 0}
        )
        entry["count"] += 1
        if entry["count"] == 1:
            entry["command"], entry["since"] = command, time.monotonic()


def _bump_queued(busy_sessions: dict[str, dict], session_name: str, delta: int) -> None:
    """Track commands WAITING on a session lock (accepted but not yet executing).

    /health used to fold them into the in-flight count, so a caller could not tell
    "one long command running" from "one running and five stacked up behind it" —
    which is exactly what an agent deciding between waiting and using another
    --session needs (v4 benchmark, busy-daemon pile-ups). Caller bumps +1 before a
    blocking acquire and -1 once the lock is held; the entry is guaranteed to exist
    (_enter_busy ran first) and is popped only when its count drains in _exit_busy."""
    with _BUSY_LOCK:
        entry = busy_sessions.get(session_name)
        if entry is not None:
            entry["queued"] = max(0, entry.get("queued", 0) + delta)


def _mark_executing(
    busy_sessions: dict[str, dict], session_name: str, command: str
) -> None:
    """Stamp the session's active metadata when a command ACTUALLY starts executing
    (session-lock handoff), not when it was accepted. Without this, /health kept
    reporting the completed predecessor's command/since after a queued waiter took
    over — the wrong command and a wildly wrong duration."""
    with _BUSY_LOCK:
        entry = busy_sessions.get(session_name)
        if entry is not None:
            entry["command"], entry["since"] = command, time.monotonic()


@contextmanager
def _held_session_lock(
    busy_sessions: dict[str, dict], session_name: str, lock, command: str
):
    """Acquire a session lock, counting the wait as "queued" for /health, and stamp
    the active command/since at handoff (see _mark_executing). The non-blocking fast
    path keeps an uncontended command from ever flickering the queued counter; the
    stamp still runs there — a predecessor may have released the lock but not yet
    drained its busy entry, which would otherwise leave its stale metadata showing."""
    if not lock.acquire(blocking=False):
        _bump_queued(busy_sessions, session_name, +1)
        try:
            lock.acquire()
        finally:
            _bump_queued(busy_sessions, session_name, -1)
    _mark_executing(busy_sessions, session_name, command)
    try:
        yield
    finally:
        lock.release()


def _exit_busy(busy_sessions: dict[str, dict], session_name: str) -> None:
    """Record a finished command. The decrement and the activity refresh happen under
    ONE lock hold: a long render must not count toward idle time (the countdown
    restarts when it finishes, not when it was accepted), and the idle watcher must
    never see the decrement without the refreshed timestamp."""
    with _BUSY_LOCK:
        _IN_FLIGHT[0] = max(0, _IN_FLIGHT[0] - 1)
        entry = busy_sessions.get(session_name)
        if entry is not None:
            entry["count"] -= 1
            if entry["count"] <= 0:
                busy_sessions.pop(session_name, None)
        _LAST_ACTIVITY[0] = time.monotonic()


def _should_idle_exit(idle_s: float) -> bool:
    """The idle watcher's decision, read under the same lock _enter/_exit_busy hold
    so it can never fire between a command finishing and its response returning."""
    with _BUSY_LOCK:
        return _IN_FLIGHT[0] == 0 and time.monotonic() - _LAST_ACTIVITY[0] > idle_s


# ── per-canonical-root reader/writer safety ────────────────────────────────────────
# `open --read-only` sessions deliberately share the writer session's LIVE SdfLayer
# (USD caches layers process-wide), but every session has its own serialization lock —
# so a reader could traverse the shared stage WHILE the writer runs a multi-step edit
# or Reload: mixed state at best, unsafe concurrent USD access at worst. Dispatch
# therefore also takes a per-canonical-root RW lock: SHARED for read-only sessions,
# EXCLUSIVE for writers. Sessions on different roots are unaffected.
class _RWLock:
    """Minimal reader-writer lock: writers exclusive, readers shared.

    A waiting writer blocks NEW readers (writer preference), so a stream of reader
    polls cannot starve the writer's edit indefinitely. Not reentrant — each daemon
    request acquires it exactly once, after its session lock (a strict session-lock →
    root-lock order, so no lock-order cycles are possible)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer_active = False
        self._writers_waiting = 0

    def acquire_shared(self) -> None:
        with self._cond:
            while self._writer_active or self._writers_waiting:
                self._cond.wait()
            self._readers += 1

    def release_shared(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_exclusive(self) -> None:
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers:
                    self._cond.wait()
            finally:
                self._writers_waiting -= 1
            self._writer_active = True

    def release_exclusive(self) -> None:
        with self._cond:
            self._writer_active = False
            self._cond.notify_all()


def _session_root_key(session) -> str | None:  # noqa: ANN001 — Session or a test fake
    """The canonical identity of the USD root layer a session currently holds.

    Prefers the session's own `root_key` (landing separately in usd_core.session) and
    falls back to reading its state non-invasively: the ownership/readers registry
    keys (both are `os.path.realpath(stage path)`), then the stage path, then the
    root layer's resolved path. None means no shared on-disk root (no stage open, or
    an anonymous in-memory stage) — the per-session lock alone suffices there."""
    for attr in ("root_key", "_layer_key", "_reader_key"):
        value = getattr(session, attr, None)
        if value:
            return str(value)
    path = getattr(session, "_stage_path", None)
    if path:
        return os.path.realpath(str(path))
    stage = getattr(session, "_stage", None)
    if stage is not None:
        try:
            real = getattr(stage.GetRootLayer(), "realPath", "")
        except Exception:  # noqa: BLE001 — a fake/anonymous stage has no shared root
            return None
        if real:
            return os.path.realpath(str(real))
    return None


def _root_lock_plan(
    session,
    command: str,  # noqa: ANN001 — Session or a test fake
    payload: dict,
) -> tuple[str | None, bool]:
    """(canonical root key, shared?) for the RW lock guarding one dispatch.

    `open` targets a DIFFERENT root than the session currently holds: a writer open
    force-reloads the target's cached layer (unsafe under readers mid-traversal), so
    it takes the TARGET's lock — exclusively for a writer open, shared for
    `open --read-only` (a pure read of the possibly-live shared layer)."""
    if command == "open":
        raw = payload.get("file")
        if isinstance(raw, str) and raw:
            return os.path.realpath(raw), bool(payload.get("read_only"))
        return None, True  # malformed payload — dispatch will reject it cleanly
    return _session_root_key(session), bool(getattr(session, "read_only", False))


# Session registry bounds. Requests without a session name share DEFAULT_SESSION; each
# distinct name gets its own isolated Session (multi-agent runs proved a single global
# Session lets one sub-agent's `--session foo open X` clobber another agent's stage).
DEFAULT_SESSION = "default"
MAX_SESSIONS = 16
_SESSION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _session_key(name: str | None) -> str:
    """Normalize a request's session field: missing/blank means the default session."""
    return (name or "").strip() or DEFAULT_SESSION


try:
    PACKAGE_VERSION = version("usd-cli")
except PackageNotFoundError:  # source tree without an install
    PACKAGE_VERSION = "0+unknown"


class CommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str = Field(min_length=1, max_length=64)
    payload: dict = Field(default_factory=dict)
    session: str | None = Field(default=None, max_length=128)


def _project_id(config: Config) -> str:
    """Stable, non-secret identity for the project owning this daemon."""
    from usd_cli.daemon import _project_id as daemon_project_id

    return daemon_project_id(config)


def _write_all(file_descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(file_descriptor, view)
        if written <= 0:
            raise OSError("server-state write made no forward progress")
        view = view[written:]


def _publish_server_state(
    config: Config,
    payload: bytes,
    *,
    state_directory_descriptor: int | None = None,
    state_path: Path | None = None,
) -> Path:
    """Install bounded state without following caller-controlled filesystem nodes."""

    if len(payload) > MAX_SERVER_STATE_BYTES:
        raise RuntimeError(
            f"daemon server state exceeds {MAX_SERVER_STATE_BYTES} bytes"
        )
    if os.name == "nt":
        from usd_core.windows_files import (
            confined_directory_path,
            open_confined_directory,
            replace_confined_regular_file_at,
        )

        if state_directory_descriptor is None:
            config.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            publication_path = Path(os.path.abspath(config.state_dir))
            directory_context = open_confined_directory(publication_path)
        else:
            if state_path is None:
                raise RuntimeError(
                    "anchored daemon state publication requires its path"
                )
            publication_path = Path(os.path.abspath(state_path))

            @contextmanager
            def duplicated_directory() -> Iterator[int]:
                duplicate = os.dup(state_directory_descriptor)
                try:
                    yield duplicate
                finally:
                    os.close(duplicate)

            directory_context = duplicated_directory()
        with directory_context as directory_descriptor:
            held_path = confined_directory_path(directory_descriptor)
            expected = os.path.normcase(os.path.normpath(str(publication_path)))
            observed = os.path.normcase(os.path.normpath(str(held_path)))
            if observed != expected:
                raise RuntimeError("daemon server-state directory identity changed")
            replace_confined_regular_file_at(
                directory_descriptor,
                "server.json",
                payload,
                max_bytes=MAX_SERVER_STATE_BYTES,
            )
        return publication_path / "server.json"
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("safe daemon state publication requires POSIX flags")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    if state_directory_descriptor is None:
        config.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        publication_path = config.state_dir
        directory_descriptor = os.open(publication_path, directory_flags)
    else:
        if state_path is None:
            raise RuntimeError("anchored daemon state publication requires its path")
        publication_path = state_path
        directory_descriptor = os.dup(state_directory_descriptor)
    temporary_descriptor = -1
    temporary_name = f".server.json.{secrets.token_hex(8)}.tmp"
    temporary_created = False
    try:
        held_metadata = os.fstat(directory_descriptor)
        path_metadata = os.stat(publication_path, follow_symlinks=False)
        if (
            not stat.S_ISDIR(held_metadata.st_mode)
            or not stat.S_ISDIR(path_metadata.st_mode)
            or (held_metadata.st_dev, held_metadata.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise RuntimeError("daemon server-state directory identity changed")
        os.fchmod(directory_descriptor, 0o700)
        temporary_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        temporary_flags |= getattr(os, "O_CLOEXEC", 0)
        temporary_descriptor = os.open(
            temporary_name,
            temporary_flags,
            0o600,
            dir_fd=directory_descriptor,
        )
        temporary_created = True
        _write_all(temporary_descriptor, payload)
        os.fchmod(temporary_descriptor, 0o600)
        os.fsync(temporary_descriptor)
        descriptor_metadata = os.fstat(temporary_descriptor)
        path_metadata = os.stat(
            temporary_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        temporary_identity = (
            descriptor_metadata.st_dev,
            descriptor_metadata.st_ino,
        )
        if (
            not stat.S_ISREG(descriptor_metadata.st_mode)
            or descriptor_metadata.st_nlink != 1
            or not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_nlink != 1
            or (path_metadata.st_dev, path_metadata.st_ino) != temporary_identity
        ):
            raise RuntimeError("daemon server-state temporary identity changed")
        os.replace(
            temporary_name,
            "server.json",
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        temporary_name = ""
        installed_metadata = os.stat(
            "server.json",
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(installed_metadata.st_mode)
            or installed_metadata.st_nlink != 1
            or (installed_metadata.st_dev, installed_metadata.st_ino)
            != temporary_identity
        ):
            raise RuntimeError("daemon server-state installed identity changed")
        final_path_metadata = os.stat(publication_path, follow_symlinks=False)
        if not stat.S_ISDIR(final_path_metadata.st_mode) or (
            held_metadata.st_dev,
            held_metadata.st_ino,
        ) != (final_path_metadata.st_dev, final_path_metadata.st_ino):
            raise RuntimeError("daemon server-state directory identity changed")
        os.fsync(directory_descriptor)
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_created and temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
        os.close(directory_descriptor)
    return publication_path / "server.json"


def write_server_state(
    config: Config,
    host: str,
    port: int,
    token: str,
    instance_id: str,
    *,
    daemon_pid: int,
    process_start_token: str,
    state_directory_descriptor: int | None = None,
    state_path: Path | None = None,
) -> Path:
    """Write `.usd-cli/server.json` for CLI discovery (PR-9.6), atomically (temp+rename)."""
    from usd_core.render.factory import resolved_renderer

    if (
        daemon_pid != os.getpid()
        or re.fullmatch(r"[^\s:]{1,256}", process_start_token) is None
    ):
        raise RuntimeError("refusing to publish an unregistered daemon identity")
    state = {
        "pid": daemon_pid,
        "process_start_token": process_start_token,
        "host": host,
        "port": port,
        "token": token,
        "instance_id": instance_id,
        "project_id": _project_id(config),
        "lifecycle_owner": config.server.get("lifecycle_owner"),
        "stage": None,
        "engine": config.backend.get("engine"),
        "renderer": resolved_renderer(config),  # the concrete renderer, not "auto"
        "mode": "headless",
        "started": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }
    payload = json.dumps(state, indent=2).encode("utf-8")
    return _publish_server_state(
        config,
        payload,
        state_directory_descriptor=state_directory_descriptor,
        state_path=state_path,
    )


def _parse_duration(s: str | int | None, default_s: int = 1800) -> int:
    """'30m' -> 1800, '45s' -> 45, '2h' -> 7200, 90 -> 90 seconds."""
    if s is None:
        return default_s
    s = str(s).strip().lower()
    try:
        if s.endswith("h"):
            return int(float(s[:-1]) * 3600)
        if s.endswith("m"):
            return int(float(s[:-1]) * 60)
        return int(float(s[:-1] if s.endswith("s") else s))
    except ValueError:
        return default_s


# Public wire command -> Session method. Keep this deliberately explicit: adding a
# Session helper must never make it remotely callable by accident.
COMMANDS = {
    "open": "open_stage",
    "resolve": "resolve",
    "snapshot": "snapshot",
    "render": "render",
    "render-probe": "render_probe",
    "render-frames": "render_frames",
    "camera.list": "camera_list",
    "camera.use": "camera_use",
    "camera.look-at": "camera_look_at",
    "camera.orbit": "camera_orbit",
    "camera.fit": "camera_fit",
    "camera.create": "camera_create",
    "camera.coverage": "camera_coverage",
    "camera.place": "camera_place",
    "camera.rig-export": "camera_rig_export",
    "camera.pan": "camera_pan",
    "camera.zoom": "camera_zoom",
    "describe": "describe",
    "find": "find",
    "properties": "properties",
    "material-binding": "material_binding",
    "material-audit": "material_audit",
    "appearance.clear": "appearance_clear",
    "appearance.audit": "appearance_audit",
    "subsets": "subsets",
    "validate": "validate_stage",
    "transform": "transform",
    "create": "create",
    "delete": "delete",
    "duplicate": "duplicate",
    "reparent": "reparent",
    "rename": "rename",
    "group": "group",
    "material": "material",
    "material.apply": "material_apply",
    "show": "show",
    "hide": "hide",
    "select": "select",
    "deselect": "deselect",
    "selection": "selection",
    "isolate": "isolate",
    "set": "set",
    "remove-api": "remove_api",
    "import": "import_",
    "align": "align",
    "scatter": "scatter",
    "stats": "stats",
    "verify": "verify",
    "bounds": "bounds",
    "visibility": "visibility",
    "distance": "distance",
    "nearest": "nearest",
    "within": "within",
    "overlapping": "overlapping",
    "above": "above",
    "below": "below",
    "left": "left",
    "right": "right",
    "front": "front",
    "behind": "behind",
    "raycast": "raycast",
    "space.free": "space_free",
    "space.support": "space_support",
    "physics.inspect": "physics_inspect",
    "physics.topology": "physics_topology",
    "physics.apply": "physics_apply",
    "physics.validate": "physics_validate",
    "physics.simulate": "physics_simulate",
    "convert": "convert",
    "undo": "undo",
    "redo": "redo",
    "history": "op_history",
    "checkpoint.save": "checkpoint_save",
    "checkpoint.load": "checkpoint_load",
    "checkpoint.list": "checkpoint_list",
    "checkpoint.delete": "checkpoint_delete",
    "save": "save",
    "viewer.snapshot": "viewer_snapshot",
    "export": "export",
    "new": "new",
    "info": "info",
    "sublayers": "sublayers",
}


_REQUEST_READ_PATH_FIELDS: dict[str, tuple[str, ...]] = {
    "open": ("file",),
    "import": ("source",),
    "convert": ("source",),
    "verify": ("file",),
    "render": ("against",),
    "render-frames": ("scene",),
    "physics.simulate": ("scene",),
    "material": (
        "library",
        "mdl",
        "diffuse_texture",
        "normal_texture",
        "orm_texture",
        "roughness_texture",
        "metallic_texture",
    ),
    "material.apply": ("library",),
}

_REQUEST_WRITE_PATH_FIELDS: dict[str, tuple[str, ...]] = {
    "save": ("path",),
    "viewer.snapshot": ("path", "line_geometry"),
    "export": ("path",),
    "convert": ("output",),
    "render": ("output",),
    "render-probe": ("output_dir",),
    "render-frames": ("output",),
    "physics.simulate": ("output",),
    "camera.coverage": ("output",),
    "camera.place": ("output",),
    "camera.rig-export": ("output",),
}


def _configured_roots(config: Config, key: str) -> tuple[Path, ...]:
    raw_roots = config.server.get(key) or [str(config.project_dir or Path.cwd())]
    if isinstance(raw_roots, str):
        raw_roots = [part for part in raw_roots.split(os.pathsep) if part]
    if not isinstance(raw_roots, (list, tuple)) or not raw_roots:
        raise ValueError(f"server.{key} must contain at least one path")
    roots: list[Path] = []
    for raw_root in raw_roots:
        if not isinstance(raw_root, str) or not raw_root:
            raise ValueError(f"server.{key} entries must be non-empty paths")
        roots.append(Path(raw_root).expanduser().resolve())
    return tuple(roots)


def _configured_allowed_roots(config: Config) -> tuple[Path, ...]:
    return _configured_roots(config, "allowed_roots")


def _configured_allowed_write_roots(config: Config) -> tuple[Path, ...]:
    """Return the daemon's filesystem write capability.

    Read roots may deliberately include immutable source libraries outside the
    project.  They never imply permission to publish beside, overwrite, or render
    into those sources; writes default to the project root unless configured
    separately.
    """

    return _configured_roots(config, "allowed_write_roots")


def _material_input_asset_values(payload: dict) -> Iterator[tuple[str, object]]:
    """Yield raw string shader inputs, which usd_core authors as asset paths."""

    inputs = payload.get("inputs")
    if isinstance(inputs, dict):
        for name, value in inputs.items():
            if isinstance(value, str):
                yield f"inputs.{name}", value
        return
    if not isinstance(inputs, list):
        return
    for index, item in enumerate(inputs):
        if not isinstance(item, str):
            continue
        _name, separator, raw_value = item.partition("=")
        if not separator:
            continue
        value = raw_value.strip()
        if value.lower() in {"true", "false"}:
            continue
        try:
            if "," in value:
                [float(part) for part in value.split(",")]
            else:
                float(value)
        except ValueError:
            yield f"inputs[{index}]", value


def _request_path_values(command: str, payload: dict) -> Iterator[tuple[str, object]]:
    for field in _REQUEST_READ_PATH_FIELDS.get(command, ()):
        yield field, payload.get(field)
    if command == "material":
        yield from _material_input_asset_values(payload)


def _request_write_path_values(
    command: str,
    payload: dict,
    session: Session | None,
) -> Iterator[tuple[str, object]]:
    for field in _REQUEST_WRITE_PATH_FIELDS.get(command, ()):
        raw = payload.get(field)
        if raw not in (None, ""):
            yield field, raw

    # Several commands derive a destination when the caller omits it. Validate
    # that effective destination at execution time, once the live session is
    # available, so opening a read-only source root can never turn into an
    # implicit write capability.
    stage_path = getattr(session, "_stage_path", None) if session is not None else None
    if command == "save" and payload.get("path") in (None, "") and stage_path:
        yield "path", stage_path
    elif command == "export" and payload.get("path") in (None, ""):
        fmt = str(payload.get("format") or "")
        if stage_path:
            yield "path", str(Path(stage_path).with_suffix(f".{fmt}"))
        elif fmt:
            yield "path", f"export.{fmt}"
    elif command == "convert" and payload.get("output") in (None, ""):
        source = payload.get("source")
        if isinstance(source, str) and source:
            suffix = Path(source).suffix.lower()
            output_format = payload.get("output_format")
            if suffix not in {".usd", ".usda", ".usdc", ".usdz"} or output_format:
                yield (
                    "output",
                    str(
                        Path(source).with_suffix(
                            "." + str(output_format or "usd").lstrip(".")
                        )
                    ),
                )
    elif command == "camera.rig-export" and payload.get("output") in (None, ""):
        # Session.camera_rig_export publishes to this relative default.  Policy
        # validation must cover the effective destination even for non-CLI callers
        # that omit ``output`` from the wire payload.
        yield "output", "rig.json"


def _normalize_camera_output_payload(
    config: Config,
    command: str,
    payload: dict,
) -> None:
    """Carry the daemon's project-based path decision into Session dispatch."""

    if command not in {"camera.coverage", "camera.place", "camera.rig-export"}:
        return
    raw = payload.get("output")
    if raw in (None, ""):
        if command != "camera.rig-export":
            return
        raw = "rig.json"
    if not isinstance(raw, str):
        return  # request policy reports the typed error before this helper runs
    path = Path(raw).expanduser()
    payload["output"] = str(
        path.resolve()
        if path.is_absolute()
        else ((config.project_dir or Path.cwd()) / path).resolve()
    )


def _set_asset_values(session: Session, payload: dict) -> Iterator[tuple[str, object]]:
    """Yield values for ``set`` only when the existing USD attribute is asset-typed."""

    ref = payload.get("ref")
    attr_name = payload.get("attr")
    if not isinstance(ref, str) or not isinstance(attr_name, str):
        return
    stage = getattr(session, "_stage", None)
    if stage is None:
        return
    try:
        path = session._edit_path_of(ref)
        prim = stage.GetPrimAtPath(path)
        attribute = prim.GetAttribute(attr_name) if prim and prim.IsValid() else None
        type_name = str(attribute.GetTypeName()) if attribute else ""
    except Exception:  # noqa: BLE001 - dispatch owns ordinary ref/type errors
        return
    raw_value = payload.get("value")
    if type_name == "asset":
        yield "value", raw_value
    elif type_name == "asset[]":
        if isinstance(raw_value, (list, tuple)):
            for index, value in enumerate(raw_value):
                yield f"value[{index}]", value
        elif isinstance(raw_value, str):
            for index, value in enumerate(raw_value.strip("[] ").split(",")):
                yield f"value[{index}]", value.strip()


def _effective_camera_count(session: Session | None, rig: object = None) -> int | None:
    """Count stage cameras when a request selects them implicitly.

    The wire-level list cap is insufficient for ``camera coverage`` with no
    ``cameras`` field and for ``camera rig-export``: both commands discover cameras
    from the live stage.  Do this cheap traversal during execution-time policy
    validation so a stage cannot bypass the camera/ray budget by omitting a list.
    Invalid refs and unopened stages are left to ordinary command validation.
    """

    if session is None or getattr(session, "_stage", None) is None:
        return None
    try:
        if rig is None:
            from pxr import UsdGeom
            from usd_core.camera_analysis.scene import imageable_analysis_policy

            # Implicit coverage evaluates only perspective cameras. Keep this cheap
            # policy preflight aligned so unrelated orthographic, hidden, or
            # visualization-purpose cameras cannot make an otherwise bounded request
            # fail before Session applies the same composed-imageable filter.
            count = 0
            for path in session._camera_paths():
                prim = session._stage.GetPrimAtPath(path)
                allowed, _ = imageable_analysis_policy(prim)
                projection = str(
                    UsdGeom.Camera(prim).GetProjectionAttr().Get()
                    or UsdGeom.Tokens.perspective
                )
                if allowed and projection == str(UsdGeom.Tokens.perspective):
                    count += 1
            return count
        from usd_core.camera_analysis.rig_export import rig_camera_paths

        path = session._path_of(rig)
        return len(rig_camera_paths(session._stage, path))
    except Exception:  # noqa: BLE001 - dispatch owns ref/stage diagnostics
        return None


def _validate_request_path(
    config: Config,
    roots: tuple[Path, ...],
    field: str,
    raw: object,
    *,
    capability: str = "allowed_roots",
) -> None:
    if raw in (None, ""):
        return
    if not isinstance(raw, str):
        raise ValueError(f"{field} must be a filesystem path string")
    if "\x00" in raw:
        raise ValueError(f"{field} contains a NUL byte")
    # allowed_roots is a local-filesystem capability. Treat every RFC-style URI
    # scheme explicitly instead of letting identifiers such as ``file:/...`` or
    # ``omniverse:asset.usd`` appear to be harmless relative paths below the
    # project directory.
    windows_drive_path = os.name == "nt" and re.match(r"^[A-Za-z]:[\\/]", raw)
    if not windows_drive_path and re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw):
        raise ValueError(
            f"{field} URI is not permitted by filesystem-only server.{capability}"
        )
    # A package member (asset.usdz[layer.usda]) is opened through the package on
    # disk. Validate the outer package; the member itself is not a host path.
    filesystem_value = raw.partition("[")[0] if "[" in raw else raw
    path = Path(filesystem_value).expanduser()
    resolved = (
        path.resolve()
        if path.is_absolute()
        else ((config.project_dir or Path.cwd()) / path).resolve()
    )
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError(f"{field} path is outside server.{capability}")


def _validate_request_policy(
    config: Config,
    command: str,
    payload: dict,
    *,
    shared: bool,
    session: Session | None = None,
) -> None:
    """Apply resource bounds and the filesystem sandbox to every daemon request.

    ``shared`` remains in the private call signature for compatibility with tests
    and older in-tree callers.  Loopback and token authentication do not confer
    filesystem authority, so it intentionally cannot disable path validation.
    """
    encoded = json.dumps(payload)
    if len(encoded.encode()) > 1_048_576:
        raise ValueError("command payload exceeds 1 MiB")
    if len(payload) > 64:
        raise ValueError("command payload has too many fields")
    for value in payload.values():
        if isinstance(value, list) and len(value) > 256:
            raise ValueError("command list argument exceeds 256 items")
    if command in {"camera.coverage", "camera.place", "camera.rig-export"}:
        raw_grid = payload.get("grid")
        grid = int(32 if raw_grid is None else raw_grid)
        if grid < 2 or grid > 512:
            raise ValueError("camera analysis grid must be between 2 and 512")
        cameras = payload.get("cameras")
        if isinstance(cameras, list) and len(cameras) > 64:
            raise ValueError("camera analysis accepts at most 64 cameras")
        if command == "camera.place":
            raw_candidates = payload.get("candidates")
            candidates = int(64 if raw_candidates is None else raw_candidates)
            if candidates < 4 or candidates > 512:
                raise ValueError(
                    "camera placement candidates must be between 4 and 512"
                )
            for field in ("max_cameras", "cameras"):
                raw_value = payload.get(field)
                value = int(1 if raw_value is None else raw_value)
                if value < 1 or value > 64:
                    raise ValueError(
                        f"camera placement {field.replace('_', '-')} must be 1-64"
                    )
        elif command == "camera.coverage":
            camera_count = (
                len(cameras)
                if isinstance(cameras, list) and cameras
                else _effective_camera_count(session)
            )
            # The first admission pass has no live session.  When cameras are
            # selected implicitly, defer the camera-count check to the existing
            # execution-time pass, which can inspect the stage. Explicit lists can
            # still be capped during admission.
            if camera_count is not None:
                if camera_count > 64:
                    raise ValueError("camera analysis accepts at most 64 cameras")
        elif command == "camera.rig-export":
            camera_count = _effective_camera_count(session, payload.get("rig"))
            if camera_count is not None and camera_count > 64:
                raise ValueError("camera rig export accepts at most 64 cameras")
        # Do not approximate aspect-sensitive workloads as ``grid * grid`` here.
        # ``grid`` is the long-axis resolution; Session derives the other axis from
        # the admitted scope bounds and applies ``validate_grid_workload`` before any
        # Newton allocation.  A square daemon estimate would reject valid narrow
        # scopes, while repeating scene ingestion in transport policy would be both
        # expensive and liable to diverge from the command's exact analysis policy.
    res = payload.get("res")
    if isinstance(res, (list, tuple)) and len(res) == 2:
        width, height = int(res[0]), int(res[1])
        if (
            width < 1
            or height < 1
            or width > 4096
            or height > 4096
            or width * height > 16_777_216
        ):
            raise ValueError("render resolution exceeds 4096x4096 / 16,777,216 pixels")
        if command == "camera.rig-export" and payload.get("verify"):
            pixels_per_camera = width * height
            if pixels_per_camera > 4_194_304:
                raise ValueError(
                    "camera rig verification exceeds 4,194,304 rays per camera"
                )
            if (
                camera_count is not None
                and camera_count * pixels_per_camera > 67_108_864
            ):
                raise ValueError(
                    "camera rig verification exceeds the 67,108,864-ray total budget"
                )
    if int(payload.get("orbit") or 0) > 64:
        raise ValueError("orbit render exceeds 64 cameras")
    if float(payload.get("duration") or 0) > 1800:
        raise ValueError("operation duration exceeds 1800 seconds")
    if os.environ.get("USD_CLI_LOCK_RENDER_CONFIG") == "1" and command in {
        "render",
        "render-frames",
    }:
        if payload.get("renderer") not in (None, ""):
            raise ValueError("render-config-locked daemon rejects renderer overrides")
        if payload.get("max_upload_mb") is not None:
            raise ValueError("render-config-locked daemon rejects upload-cap overrides")

    del shared
    roots = _configured_allowed_roots(config)
    for field, raw in _request_path_values(command, payload):
        _validate_request_path(config, roots, field, raw)
    if command == "set" and session is not None:
        for field, raw in _set_asset_values(session, payload):
            _validate_request_path(config, roots, field, raw)
    write_roots = _configured_allowed_write_roots(config)
    for field, raw in _request_write_path_values(command, payload, session):
        _validate_request_path(
            config,
            write_roots,
            field,
            raw,
            capability="allowed_write_roots",
        )


# Wire commands whose Session method is guarded by @_mutating: the daemon rejects
# these on a read-only session up front, before they queue on the session lock —
# queue-then-reject made a read-only `camera fit` hang behind a long render.
MUTATING_COMMANDS = frozenset(
    wire
    for wire, meth in COMMANDS.items()
    if getattr(getattr(Session, meth, None), "_is_mutating", False)
)

# Commands with side effects beyond @_mutating stage edits (files written, the
# session's open stage swapped, checkpoints created/removed): a queued one whose
# client disconnected must also be skipped — a ghost `open --force-reload` firing
# after its client died discards edits just as surely as a ghost `save`.
SIDE_EFFECT_COMMANDS = MUTATING_COMMANDS | frozenset(
    (
        "open",
        "new",
        "export",
        "convert",
        "render",
        "render-probe",
        "render-frames",
        "checkpoint.save",
        "checkpoint.delete",
        "physics.simulate",
        "camera.coverage",
        "camera.place",
        "camera.rig-export",
        "viewer.snapshot",
    )
)

CAMERA_ANALYSIS_COMMANDS = frozenset(
    ("camera.coverage", "camera.place", "camera.rig-export")
)


def _skippable_when_client_gone(command: str, payload: dict) -> bool:
    """Should a queued command whose client vanished be skipped instead of run?

    Everything with side effects (stage edits, files written, the open stage
    swapped) plus renders (GPU-minutes for nobody) — and the read-mostly
    commands whose FLAGS make them mutate (`validate --fix`,
    `sublayers --drop-dead`)."""
    if command in SIDE_EFFECT_COMMANDS:
        return True
    if command == "validate" and payload.get("fix"):
        return True
    return command == "sublayers" and bool(payload.get("drop_dead"))


def _request_is_mutating(command: str, payload: dict | None = None) -> bool:
    if command in MUTATING_COMMANDS:
        return True
    return command == "camera.place" and bool((payload or {}).get("author_under"))


def _read_only_reject(
    session: Session, command: str, payload: dict | None = None
) -> Response | None:
    """The fast-fail envelope for a mutating command on a read-only session, or
    None when the command may proceed. Mirrors the @_mutating in-method guard
    (which still runs at execution — this is the no-queue fast path)."""
    if _request_is_mutating(command, payload) and getattr(session, "read_only", False):
        return Response(
            command=command,
            ok=False,
            issues=[
                Issue(
                    "error",
                    f"session '{getattr(session, 'name', '?')}' is read-only (opened with "
                    "--read-only) — mutating commands are blocked; re-open without "
                    "--read-only to edit",
                )
            ],
        )
    return None


def dispatch(session: Session, command: str, payload: dict) -> Response:
    """Route one parsed command to the Session (e.g. 'checkpoint.save' -> checkpoint_save)."""
    name = COMMANDS.get(command)
    if name is None:
        return Response(
            command=command,
            ok=False,
            issues=[Issue("error", f"unknown command '{command}'")],
        )
    cancelled_type: type[Exception] | None = None
    if command in CAMERA_ANALYSIS_COMMANDS:
        from usd_core.camera_analysis.cancellation import (  # noqa: PLC0415
            CameraAnalysisCancelled,
        )

        cancelled_type = CameraAnalysisCancelled
    try:
        import inspect

        if not isinstance(payload, dict):
            raise TypeError("payload must be an object")
        method = getattr(session, name)
        accepted = {
            parameter
            for parameter, spec in inspect.signature(method).parameters.items()
            if spec.kind
            not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        }
        unknown = sorted(set(payload) - accepted)
        if unknown:
            raise TypeError(
                f"unknown payload field(s) for '{command}': {', '.join(unknown)}"
            )
        result = method(**payload)
        if not isinstance(result, Response):
            raise TypeError(f"command '{command}' returned an invalid response")
        return result
    except NotImplementedError as exc:
        return Response(
            command=command,
            ok=False,
            issues=[Issue("error", f"not implemented: {exc}")],
        )
    except Exception as exc:  # noqa: BLE001 — any engine error becomes a clean
        # structured response (stale refs, no-stage-open, bad args, render failures, …)
        # rather than an opaque HTTP 500.
        if cancelled_type is not None and isinstance(exc, cancelled_type):
            raise
        return Response(
            command=command,
            ok=False,
            issues=[Issue("error", str(exc) or type(exc).__name__)],
        )


def _dispatch_with_camera_cancellation(
    session: Session,
    command: str,
    payload: dict,
    cancel_event: threading.Event,
) -> Response:
    """Bind one request's event and normalize a cooperative stop as a response."""

    if command not in CAMERA_ANALYSIS_COMMANDS:
        return dispatch(session, command, payload)
    # Lazy by design: ordinary daemon commands do not import camera-analysis,
    # NumPy, Newton, or Warp merely to pass through the transport shell.
    from usd_core.camera_analysis.cancellation import (  # noqa: PLC0415
        CameraAnalysisCancelled,
        cancellation_scope,
        check_cancelled,
    )

    try:
        with cancellation_scope(cancel_event):
            check_cancelled()  # queued jobs must stop before Session dispatch
            return dispatch(session, command, payload)
    except CameraAnalysisCancelled as exc:
        return Response(
            command=command,
            ok=False,
            summary={"error_type": "cancelled", "state": "cancelled"},
            issues=[Issue("error", str(exc))],
        )


def build_app(
    config: Config | None = None,
    *,
    token: str | None = None,
    instance_id: str | None = None,
    daemon_pid: int | None = None,
    process_start_token: str | None = None,
):
    """Construct the FastAPI app. Imported lazily so the CLI never pulls fastapi."""
    from fastapi import Depends, FastAPI, Header, HTTPException  # noqa: PLC0415
    from fastapi.responses import JSONResponse  # noqa: PLC0415

    cfg = config or load_config()
    session_config = copy.deepcopy(cfg)
    session_config.server["allowed_write_roots"] = [
        str(root) for root in _configured_allowed_write_roots(cfg)
    ]
    expected_token = token or secrets.token_hex(32)
    daemon_instance = instance_id or secrets.token_hex(16)
    reported_pid = daemon_pid if daemon_pid is not None else os.getpid()
    reported_start_token = process_start_token
    # Session registry (RR-04, extended for multi-agent runs): one Session and one
    # serialization RLock PER named session, so `--session foo` truly isolates stages
    # between sub-agents. The default session exists from startup, preserving the
    # original single-session behavior for un-sessioned requests. Work WITHIN a session
    # stays strictly serialized; different sessions may run concurrently (the GIL still
    # bounds real parallelism for C++ USD calls — correctness, not throughput). All
    # sessions live and die with this process, so `server stop` tears down every one.
    registry_lock = threading.Lock()

    def _new_session(name: str) -> Session:
        # Session(name=...) is landing separately; pass the registry key through when
        # the constructor accepts it (per-session artifact namespacing), and fall back
        # for a Session (or test fake) that predates the parameter.
        try:
            return Session(session_config, name=name)
        except TypeError:
            return Session(session_config)

    sessions: dict[str, Session] = {DEFAULT_SESSION: _new_session(DEFAULT_SESSION)}
    session_locks: dict[str, threading.RLock] = {DEFAULT_SESSION: threading.RLock()}

    def _validated_session_key(name: str | None) -> str:
        key = _session_key(name)
        if not _SESSION_NAME_RE.match(key):
            raise ValueError(
                f"invalid session name '{key}' (use letters, digits, "
                "'.', '_', '-'; max 64 chars)"
            )
        return key

    def _resolve_session_locked(
        key: str,
    ) -> tuple[str, Session, threading.RLock]:
        if key not in sessions:
            if len(sessions) >= MAX_SESSIONS:
                raise ValueError(
                    f"session limit reached ({MAX_SESSIONS}); reuse one of: "
                    f"{', '.join(sorted(sessions))} — or release a named session"
                )
            sessions[key] = _new_session(key)
            session_locks[key] = threading.RLock()
        return key, sessions[key], session_locks[key]

    def _resolve_session(name: str | None) -> tuple[str, Session, threading.RLock]:
        """Return (key, session, lock), creating the session on first use (capped)."""

        key = _validated_session_key(name)
        with registry_lock:
            return _resolve_session_locked(key)

    # Busy bookkeeping for /health and the idle watcher, per session (see the
    # module-level _enter_busy/_exit_busy, which take _BUSY_LOCK). The session locks
    # serialize the actual work; this only records what is in flight, so it must never
    # be read while HOLDING a session lock (that would make /health block behind a
    # render). _IN_FLIGHT stays the all-sessions total the idle watcher checks.
    busy_sessions: dict[str, dict] = {}  # session -> {"count", "command", "since"}

    def _accept_session(
        name: str | None,
        command: str,
    ) -> tuple[str, Session, threading.RLock]:
        """Resolve a session and atomically reserve it for one accepted command."""

        key = _validated_session_key(name)
        # Registry -> busy is the sole two-lock order. Session release takes the
        # same order, so it cannot remove a session between request admission and
        # the worker publishing its busy state.
        with registry_lock:
            resolved = _resolve_session_locked(key)
            _enter_busy(busy_sessions, key, command)
            return resolved

    def _release_session(name: object) -> Response:
        """Release one idle named session without changing daemon lifecycle."""

        if not isinstance(name, str) or not name.strip():
            return Response(
                command="server.release-session",
                ok=False,
                issues=[Issue("error", "an explicit session name is required")],
            )
        try:
            key = _validated_session_key(name)
        except ValueError as exc:
            return Response(
                command="server.release-session",
                ok=False,
                issues=[Issue("error", str(exc))],
            )
        if key == DEFAULT_SESSION:
            return Response(
                command="server.release-session",
                ok=False,
                issues=[Issue("error", "the default daemon session cannot be released")],
            )
        with registry_lock:
            with _BUSY_LOCK:
                entry = busy_sessions.get(key)
                if entry is not None and entry.get("count", 0) > 0:
                    return Response(
                        command="server.release-session",
                        ok=False,
                        summary={
                            "session": key,
                            "busy": True,
                            "queued": int(entry.get("queued", 0)),
                        },
                        issues=[
                            Issue(
                                "error",
                                f"session '{key}' is busy; wait for its commands "
                                "to finish before releasing it",
                            )
                        ],
                    )
                released = sessions.pop(key, None) is not None
                session_locks.pop(key, None)
        return Response(
            command="server.release-session",
            summary={"session": key, "released": released},
        )

    # R13 async jobs: `render --detach` (and friends) run through the SAME locked
    # pipeline in a daemon-side thread and return a job id immediately; `usd-cli wait`
    # blocks on the result. Kills the poll-turn pattern that cost ~15-25M input
    # tokens across the round-7 traces.
    jobs: dict[str, dict] = {}
    jobs_lock = threading.Lock()
    job_seq = [0]
    # The command daemon is loopback-only, but loopback callers still receive no
    # ambient filesystem authority. The compatibility flag passed to the policy
    # helper is therefore always false and never bypasses allowed_roots.
    shared = False
    # /health must never block or hop to the worker threadpool (see the async routes
    # below); the renderer resolution probes the filesystem, so do it once up front.
    from usd_core.render.factory import resolved_renderer  # noqa: PLC0415

    renderer_name = resolved_renderer(cfg)

    # Per-canonical-root RW locks (module docs at _RWLock): read-only sessions take
    # SHARED, writers EXCLUSIVE, so a reader can never traverse the shared SdfLayer
    # mid-edit. Guarded by registry_lock; entries are tiny and live for the process.
    root_locks: dict[str, _RWLock] = {}

    @contextmanager
    def _held_root_lock(session: Session, command: str, payload: dict):
        """Hold the per-root RW lock around one dispatch.

        Yields a zero-arg `release_early` callable: renders call it (via
        usd_core.render.remote.STAGE_RELEASE_HOOK) the moment the stage is fully
        packaged, so a multi-minute upload/render no longer holds the root
        exclusively against read-only audit sessions. Idempotent — the exit path
        releases only if nothing released earlier."""
        key, shared_mode = _root_lock_plan(session, command, payload)
        if key is None:
            yield lambda: None
            return
        with registry_lock:
            rw = root_locks.setdefault(key, _RWLock())
        acquire = rw.acquire_shared if shared_mode else rw.acquire_exclusive
        release = rw.release_shared if shared_mode else rw.release_exclusive
        acquire()
        # one-shot gate: whichever of {early hook, exit path} comes first releases;
        # the gate is acquired exactly once and never released, so a race between
        # the two can never double-release the RW lock
        release_gate = threading.Lock()

        def release_early() -> None:
            if release_gate.acquire(blocking=False):
                release()

        try:
            yield release_early
        finally:
            release_early()

    from contextlib import asynccontextmanager  # noqa: PLC0415

    @asynccontextmanager
    async def _lifespan(_app):
        try:
            yield
        finally:
            # Release every Session at app shutdown. The daemon process case doesn't
            # need this (sessions die with the process), but embedded apps (tests,
            # in-process servers) must not keep Sessions — and therefore root-layer
            # ownership (usd_core.session's cross-session open guard) — alive after
            # the app is torn down.
            with registry_lock:
                sessions.clear()
                session_locks.clear()
            # Local OVRTX is a child process shared across per-command backend
            # objects. Stop it explicitly on graceful shutdown; its parent-death
            # signal handles SIGKILL and other abrupt daemon loss.
            from usd_core.render.ovrtx import close_shared_daemons
            close_shared_daemons()

    app = FastAPI(title="usd-cli daemon", version=PACKAGE_VERSION, lifespan=_lifespan)

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        max_body_bytes = 2 * 1024 * 1024
        raw_length = request.headers.get("content-length")
        if raw_length:
            try:
                parsed_length = int(raw_length)
                if parsed_length < 0:
                    raise ValueError
                if parsed_length > max_body_bytes:
                    return JSONResponse(
                        status_code=413, content={"detail": "request body too large"}
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400, content={"detail": "invalid content-length"}
                )
        # Content-Length is optional for chunked HTTP. Bound the bytes consumed by
        # the framework before FastAPI/Pydantic parse them, then cache the accepted
        # body so call_next can replay it to the downstream request.
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > max_body_bytes:
                return JSONResponse(
                    status_code=413, content={"detail": "request body too large"}
                )
            body.extend(chunk)
        request._body = bytes(body)  # noqa: SLF001 - Starlette's replay contract
        return await call_next(request)

    async def authenticate(
        x_usd_cli_token: str | None = Header(default=None, alias="x-usd-cli-token"),
        x_ov_token: str | None = Header(default=None, alias="x-ov-token"),
        x_3dsc_token: str | None = Header(default=None, alias="x-3dsc-token"),
    ) -> None:
        # Async on purpose: a SYNC dependency runs on the AnyIO worker threadpool,
        # and /health depends on this — same-session waiters saturating that pool
        # must never be able to starve the health probe's auth check.
        # x-ov-token / x-3dsc-token: pre-rename clients still authenticate
        # during deprecation.
        supplied = next(
            (v for v in (x_usd_cli_token, x_ov_token, x_3dsc_token) if v is not None),
            None,
        )
        if supplied is None or not secrets.compare_digest(supplied, expected_token):
            raise HTTPException(status_code=401, detail="invalid daemon token")

    from usd_core.render import remote as _remote_render  # noqa: PLC0415

    async def _watch_disconnect(request: Request, gone: threading.Event) -> None:
        try:
            while not gone.is_set():
                if await request.is_disconnected():
                    gone.set()
                    return
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a broken watcher must not affect the command
            pass

    def _job_state(job: dict) -> tuple[str, str | None]:
        """Return the outcome-honest public state and optional failure detail."""

        future = job["future"]
        if future.cancelled():
            return "cancelled", "camera analysis cancelled before execution"
        if not future.done():
            return (
                "cancelling" if job["cancel_event"].is_set() else "running",
                None,
            )
        try:
            result = future.result(timeout=0)
        except Exception as exc:  # noqa: BLE001 — job died outside dispatch
            return "failed", str(exc)
        if result.get("summary", {}).get("error_type") == "cancelled":
            return "cancelled", next(
                (
                    issue.get("message")
                    for issue in result.get("issues", [])
                    if issue.get("severity") == "error"
                ),
                "camera analysis cancelled",
            )
        if result.get("ok", True):
            return "done", None
        return "failed", next(
            (
                issue.get("message")
                for issue in result.get("issues", [])
                if issue.get("severity") == "error"
            ),
            None,
        )

    @app.post("/cmd", dependencies=[Depends(authenticate)])
    async def cmd(body: CommandRequest, request: Request):  # noqa: ANN202
        try:
            _validate_request_policy(cfg, body.command, body.payload, shared=shared)
            _normalize_camera_output_payload(cfg, body.command, body.payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if body.command == "server.release-session":
            return _release_session(body.payload.get("name")).to_dict()
        try:
            key, session, session_lock = _resolve_session(body.session)
        except (
            ValueError
        ) as exc:  # bad name / session cap — a clean envelope, not a 500
            return Response(
                command=body.command, ok=False, issues=[Issue("error", str(exc))]
            ).to_dict()
        if body.command == "jobs":
            with jobs_lock:
                rows = []
                for jid, j in jobs.items():
                    # "done" must mean SUCCEEDED — a stalled render listed as
                    # `j2 done 188.8s` sent an agent shipping unverified work
                    # (round 9, task-03). Peek at the finished result's envelope.
                    state, error = _job_state(j)
                    row = {
                        "job": jid,
                        "command": j["command"],
                        "session": j["session"],
                        "state": state,
                        "elapsed_s": round(
                            (j.get("finished", time.monotonic())) - j["started"], 1
                        ),
                    }
                    if error:
                        row["error"] = error[:200]
                    if j.get("non_preemptible_after_dispatch"):
                        row["cancellation_boundary"] = (
                            "ovrtx_verification_non_preemptible_after_dispatch"
                        )
                    rows.append(row)
            text = (
                "\n".join(
                    f"{r['job']:6} {r['state']:8} {r['elapsed_s']:8}s "
                    f"{r['command']} ({r['session']})"
                    + (f" — {r['error']}" if r.get("error") else "")
                    for r in rows
                )
                or "no jobs"
            )
            return Response(
                command="jobs",
                summary={"jobs": len(rows)},
                data={"text": text, "jobs": rows},
            ).to_dict()
        if body.command == "cancel":
            if set(body.payload) != {"job"}:
                return Response(
                    command="cancel",
                    ok=False,
                    issues=[Issue("error", "cancel requires exactly one job id")],
                ).to_dict()
            jid = str(body.payload.get("job") or "").strip()
            with jobs_lock:
                job = jobs.get(jid)
                if job is None:
                    return Response(
                        command="cancel",
                        ok=False,
                        issues=[
                            Issue(
                                "error",
                                f"unknown job '{jid}' — see `usd-cli jobs`",
                            )
                        ],
                    ).to_dict()
                if job["command"] not in CAMERA_ANALYSIS_COMMANDS:
                    return Response(
                        command="cancel",
                        ok=False,
                        issues=[
                            Issue(
                                "error",
                                "cooperative cancellation is supported only for "
                                "detached camera-analysis jobs",
                            )
                        ],
                    ).to_dict()
                state, _error = _job_state(job)
                if state in {"done", "failed", "cancelled"}:
                    return Response(
                        command="cancel",
                        summary={
                            "job": jid,
                            "state": state,
                            "cancel_requested": False,
                            "late_cancel_ignored": True,
                        },
                        data={
                            "text": f"job {jid} already {state}; cancellation ignored"
                        },
                    ).to_dict()
                job["cancel_event"].set()
                queued = job["future"].cancel()
                if queued:
                    job["finished"] = time.monotonic()
                    state = "cancelled"
                else:
                    state = "cancelling"
                non_preemptible = bool(job.get("non_preemptible_after_dispatch"))
            if queued:
                # Admission reserves the session before submitting the detached
                # worker. A Future cancelled while still queued never enters
                # _job_work(), so its finally block cannot release that reservation.
                # Do it here, outside jobs_lock, to preserve the busy -> jobs lock
                # order used by a completing worker.
                _exit_busy(busy_sessions, job["session"])
            return Response(
                command="cancel",
                summary={
                    "job": jid,
                    "state": state,
                    "cancel_requested": True,
                    "queued": queued,
                    **(
                        {
                            "cancellation_boundary": (
                                "ovrtx_verification_non_preemptible_after_dispatch"
                            )
                        }
                        if non_preemptible
                        else {}
                    ),
                },
                data={
                    "text": (
                        f"job {jid} cancelled before execution"
                        if queued
                        else (
                            f"job {jid} will stop before OVRTX verification if it has "
                            "not started; once dispatched, verification evidence and "
                            "rig publication complete atomically"
                            if non_preemptible
                            else f"job {jid} is cancelling at the next safe boundary"
                        )
                    )
                },
            ).to_dict()
        if body.command == "wait":
            jid = str(body.payload.get("job") or "")
            timeout_s = float(body.payload.get("timeout", 1500))
            with jobs_lock:
                if not jid and jobs:
                    # Bare `usd-cli wait` = newest job. A `cmd && render --detach` chain
                    # can lose the printed job id (round 8: the agent re-issued an
                    # identical render because j3's id vanished into an exec yield).
                    jid = max(jobs, key=lambda k: jobs[k]["started"])
                job = jobs.get(jid)
            if job is None:
                msg = (
                    f"unknown job '{jid}' — see `usd-cli jobs`"
                    if jid
                    else "no detached jobs — start one with `render --detach`"
                )
                return Response(
                    command="wait", ok=False, issues=[Issue("error", msg)]
                ).to_dict()

            def _wait_job() -> dict:
                from concurrent.futures import (  # noqa: PLC0415
                    CancelledError as _FutCancelled,
                    TimeoutError as _FutTimeout,
                )

                try:
                    result = job["future"].result(timeout=timeout_s)
                except _FutCancelled:
                    return Response(
                        command=job["command"],
                        ok=False,
                        summary={
                            "error_type": "cancelled",
                            "job": jid,
                            "state": "cancelled",
                        },
                        issues=[
                            Issue(
                                "error",
                                "camera analysis cancelled before execution",
                            )
                        ],
                    ).to_dict()
                except _FutTimeout:
                    return Response(
                        command="wait",
                        ok=False,
                        summary={
                            "error_type": "busy",
                            "job": jid,
                            "state": "running",
                            "elapsed_s": round(time.monotonic() - job["started"], 1),
                        },
                        issues=[
                            Issue(
                                "error",
                                f"job {jid} still running after {timeout_s:.0f}s "
                                "— `usd-cli wait` again or check `usd-cli jobs`",
                            )
                        ],
                    ).to_dict()
                if result.get("summary", {}).get("error_type") == "cancelled":
                    result = dict(result)
                    result["summary"] = {**result["summary"], "job": jid}
                return result

            return await asyncio.to_thread(_wait_job)
        try:
            key, session, session_lock = _accept_session(
                body.session,
                body.command,
            )
        except ValueError as exc:
            return Response(
                command=body.command, ok=False, issues=[Issue("error", str(exc))]
            ).to_dict()
        detach = bool(body.payload.pop("detach", False))
        # Read-only sessions fast-fail mutating commands HERE — before queueing on
        # the session lock. Queue-then-reject made a read-only `camera fit` wait
        # out a long render just to be told no (round 5). The in-method guard
        # still runs at execution for anything that races an open.
        rejected = _read_only_reject(session, body.command, body.payload)
        if rejected is not None:
            _exit_busy(busy_sessions, key)
            return rejected.to_dict()
        # A mutating command whose client has already given up (timeout/Ctrl-C
        # while queued) must not execute after the fact: those ghost mutations
        # (round 5: a queued `save` fired 5 minutes after its client died)
        # surprise the agent's next read of the stage. Read-only commands still
        # run — executing them for a vanished client is merely wasted work.
        client_gone = threading.Event()
        watcher = asyncio.get_running_loop().create_task(
            _watch_disconnect(request, client_gone)
        )

        def _work() -> dict:
            try:
                with _held_session_lock(busy_sessions, key, session_lock, body.command):
                    try:
                        # re-check path policy AT EXECUTION: the pre-queue check
                        # validated a pathname that a symlink swap during a long
                        # queue wait could have redirected out of allowed_roots
                        _validate_request_policy(
                            cfg,
                            body.command,
                            body.payload,
                            shared=shared,
                            session=session,
                        )
                    except (TypeError, ValueError) as exc:
                        return Response(
                            command=body.command,
                            ok=False,
                            issues=[Issue("error", str(exc))],
                        ).to_dict()
                    if client_gone.is_set() and _skippable_when_client_gone(
                        body.command, body.payload
                    ):
                        logger.warning(
                            "session %s: skipping queued '%s' — its client "
                            "disconnected before execution started",
                            key,
                            body.command,
                        )
                        return Response(
                            command=body.command,
                            ok=False,
                            issues=[
                                Issue(
                                    "error",
                                    "skipped: the requesting client disconnected while this "
                                    "command was queued",
                                )
                            ],
                        ).to_dict()
                    with _held_root_lock(
                        session, body.command, body.payload
                    ) as release_root:
                        # final checks INSIDE every lock: the root-lock wait can
                        # be long too — a client that vanished during it must
                        # not fire side effects, and the path policy must hold
                        # at the moment of use, not just at queue admission
                        if client_gone.is_set() and _skippable_when_client_gone(
                            body.command, body.payload
                        ):
                            logger.warning(
                                "session %s: skipping '%s' — client disconnected "
                                "during the root-lock wait",
                                key,
                                body.command,
                            )
                            return Response(
                                command=body.command,
                                ok=False,
                                issues=[
                                    Issue(
                                        "error",
                                        "skipped: the requesting client disconnected "
                                        "while this command was queued",
                                    )
                                ],
                            ).to_dict()
                        try:
                            _validate_request_policy(
                                cfg,
                                body.command,
                                body.payload,
                                shared=shared,
                                session=session,
                            )
                        except (TypeError, ValueError) as exc:
                            return Response(
                                command=body.command,
                                ok=False,
                                issues=[Issue("error", str(exc))],
                            ).to_dict()
                        if body.command in ("render", "render-frames"):
                            token = _remote_render.STAGE_RELEASE_HOOK.set(release_root)
                            try:
                                return _dispatch_with_camera_cancellation(
                                    session,
                                    body.command,
                                    body.payload,
                                    client_gone,
                                ).to_dict()
                            finally:
                                _remote_render.STAGE_RELEASE_HOOK.reset(token)
                        return _dispatch_with_camera_cancellation(
                            session,
                            body.command,
                            body.payload,
                            client_gone,
                        ).to_dict()
            finally:
                _exit_busy(busy_sessions, key)

        if detach:
            watcher.cancel()  # a detached job's client is EXPECTED to go away
            job_cancel_event = threading.Event()

            def _job_work() -> dict:
                try:
                    with (
                        _held_session_lock(
                            busy_sessions, key, session_lock, body.command
                        ),
                        _held_root_lock(
                            session, body.command, body.payload
                        ) as release_root,
                    ):
                        try:
                            # Detached work may wait behind a long session/root
                            # lock. Re-resolve every path at the actual execution
                            # boundary just like the synchronous path.
                            _validate_request_policy(
                                cfg,
                                body.command,
                                body.payload,
                                shared=shared,
                                session=session,
                            )
                        except (TypeError, ValueError) as exc:
                            return Response(
                                command=body.command,
                                ok=False,
                                issues=[Issue("error", str(exc))],
                            ).to_dict()
                        if body.command in ("render", "render-frames"):
                            token = _remote_render.STAGE_RELEASE_HOOK.set(release_root)
                            try:
                                return _dispatch_with_camera_cancellation(
                                    session,
                                    body.command,
                                    body.payload,
                                    job_cancel_event,
                                ).to_dict()
                            finally:
                                _remote_render.STAGE_RELEASE_HOOK.reset(token)
                        return _dispatch_with_camera_cancellation(
                            session,
                            body.command,
                            body.payload,
                            job_cancel_event,
                        ).to_dict()
                finally:
                    _exit_busy(busy_sessions, key)
                    # freeze elapsed_s at completion — a long-finished job must not
                    # keep "aging" in the jobs listing (the submitter holds jobs_lock
                    # until the entry exists, so this lookup cannot miss)
                    with jobs_lock:
                        if jid in jobs:
                            jobs[jid]["finished"] = time.monotonic()

            import concurrent.futures as _fut

            with jobs_lock:
                job_seq[0] += 1
                jid = f"j{job_seq[0]}"
                pool = getattr(app.state, "job_pool", None)
                if pool is None:
                    pool = _fut.ThreadPoolExecutor(
                        max_workers=4, thread_name_prefix="ov_job"
                    )
                    app.state.job_pool = pool
                jobs[jid] = {
                    "future": pool.submit(_job_work),
                    "command": body.command,
                    "session": key,
                    "started": time.monotonic(),
                    "cancel_event": job_cancel_event,
                    "non_preemptible_after_dispatch": (
                        body.command == "camera.rig-export"
                        and bool(body.payload.get("verify"))
                    ),
                }
            return Response(
                command=body.command,
                summary={
                    "job": jid,
                    "state": "running",
                    "hint": f"usd-cli wait {jid} [--timeout S] — or usd-cli jobs",
                },
            ).to_dict()
        try:
            return await asyncio.to_thread(_work)
        finally:
            client_gone.set()  # stop the watcher loop
            watcher.cancel()

    @app.post("/batch", dependencies=[Depends(authenticate)])
    def batch(cmds: list[CommandRequest]):  # noqa: ANN202
        if len(cmds) > 100:
            raise HTTPException(status_code=413, detail="batch exceeds 100 commands")
        if len({_session_key(c.session) for c in cmds}) > 1:
            raise HTTPException(
                status_code=400, detail="batch commands must target a single session"
            )
        try:
            for item in cmds:
                _validate_request_policy(cfg, item.command, item.payload, shared=shared)
                _normalize_camera_output_payload(cfg, item.command, item.payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            key, session, session_lock = _resolve_session(
                cmds[0].session if cmds else None
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            key, session, session_lock = _accept_session(
                cmds[0].session if cmds else None,
                f"batch[{len(cmds)}]",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def _run(c: CommandRequest) -> dict:
            # per-item root lock: an `open` mid-batch can move the session to a
            # different root, so the key is re-planned for every command
            with _held_root_lock(session, c.command, c.payload):
                try:
                    # re-validate at execution: batch items can sit behind long
                    # predecessors, leaving the same symlink-swap window /cmd had
                    _validate_request_policy(
                        cfg,
                        c.command,
                        c.payload,
                        shared=shared,
                        session=session,
                    )
                except (TypeError, ValueError) as exc:
                    return Response(
                        command=c.command, ok=False, issues=[Issue("error", str(exc))]
                    ).to_dict()
                return dispatch(session, c.command, c.payload).to_dict()

        try:
            # the whole batch stays atomic w.r.t. its session (RR-04)
            with _held_session_lock(
                busy_sessions, key, session_lock, f"batch[{len(cmds)}]"
            ):
                return [_run(c) for c in cmds]
        finally:
            _exit_busy(busy_sessions, key)

    @app.get("/health", dependencies=[Depends(authenticate)])
    async def health():  # noqa: ANN202
        """Lightweight status. Deliberately takes NO session lock, so a daemon busy
        with a long render/sim still answers and callers can tell busy from dead.
        ASYNC on purpose: a sync endpoint would hop to the AnyIO worker threadpool,
        where same-session waiters (each parked on a session lock) can occupy every
        thread and starve the probe past the client's 2s window. This path must stay
        free of blocking work — it only snapshots bookkeeping dicts under two
        micro-held locks (_BUSY_LOCK / registry_lock) and reads plain attributes."""
        now = time.monotonic()
        with _BUSY_LOCK:
            busy = _IN_FLIGHT[0] > 0
            in_flight = {
                name: (
                    entry["command"],
                    round(now - entry["since"], 1),
                    entry.get("queued", 0),
                )
                for name, entry in busy_sessions.items()
            }
        with registry_lock:
            stages = {name: s._stage_path for name, s in sessions.items()}
        # "queued" counts commands accepted but still WAITING on the session lock —
        # in-flight alone hid a pile-up behind one long packaging/upload command.
        report = {
            name: {
                "stage": stage,
                "busy": name in in_flight,
                "current_command": in_flight.get(name, (None, 0.0, 0))[0],
                "busy_seconds": in_flight.get(name, (None, 0.0, 0))[1],
                "queued": in_flight.get(name, (None, 0.0, 0))[2],
            }
            for name, stage in stages.items()
        }
        # The legacy top-level fields mirror the DEFAULT session's stage and the
        # longest-running in-flight command, so existing single-session callers
        # (CLI status, discovery) keep working unchanged.
        current, busy_s, _ = max(
            in_flight.values(), key=lambda item: item[1], default=(None, 0.0, 0)
        )
        return {
            "ok": True,
            "engine": cfg.backend.get("engine"),
            "renderer": renderer_name,
            "stage": stages.get(DEFAULT_SESSION),
            "pid": reported_pid,
            "process_start_token": reported_start_token,
            "instance_id": daemon_instance,
            "project_id": _project_id(cfg),
            "lifecycle_owner": cfg.server.get("lifecycle_owner"),
            "busy": busy,
            "current_command": current,
            "busy_seconds": busy_s,
            "queued": sum(item[2] for item in in_flight.values()),
            "sessions": report,
        }

    @app.get("/live")
    async def live():  # noqa: ANN202
        """Public liveness only; contains no project, scene, or renderer details.
        Async for the same reason as /health: it must answer even when the worker
        threadpool is fully occupied by blocked same-session commands."""
        return {"ok": True}

    return app


def _normalized_bind_host(host: object) -> str:
    """Return the numeric loopback bind host supported by daemon discovery."""

    if not isinstance(host, str):
        raise RuntimeError("daemon host must be a numeric IP address")
    normalized = "127.0.0.1" if host == "localhost" else host
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError as exc:
        raise RuntimeError("daemon host must be a numeric IP address") from exc
    if not address.is_loopback:
        raise RuntimeError(
            "usd-cli daemon binding is loopback-only; configure remote rendering "
            "through the authenticated OVRTX service"
        )
    return normalized


def _free_port(host: str = "127.0.0.1") -> int:
    import socket

    family = (
        socket.AF_INET6 if ipaddress.ip_address(host).version == 6 else socket.AF_INET
    )
    s = socket.socket(family)
    s.bind((host, 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _remove_server_state_if_owned(
    config: Config,
    *,
    daemon_pid: int,
    process_start_token: str,
    instance_id: str,
    state_directory_descriptor: int | None = None,
) -> None:
    """Remove only this daemon's state while excluding cooperating replacements."""

    from usd_cli.daemon import (
        _locked_start_lifecycle,
        _read_regular_file_at,
        _read_state,
    )

    expected = {
        "pid": daemon_pid,
        "process_start_token": process_start_token,
        "project_id": _project_id(config),
        "instance_id": instance_id,
    }

    if state_directory_descriptor is not None and os.name != "nt":
        try:
            directory_descriptor = os.dup(state_directory_descriptor)
            try:
                payload = _read_regular_file_at(
                    directory_descriptor,
                    "server.json",
                    max_bytes=MAX_SERVER_STATE_BYTES,
                )
                state = json.loads(payload)
                if isinstance(state, dict) and all(
                    state.get(field) == value for field, value in expected.items()
                ):
                    os.unlink("server.json", dir_fd=directory_descriptor)
                    os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError):
            pass
        return

    try:
        with _locked_start_lifecycle(config):
            state = _read_state(config)
            if not isinstance(state, dict):
                return
            if all(state.get(field) == value for field, value in expected.items()):
                config.server_state_path.unlink(missing_ok=True)
    except (OSError, RuntimeError):
        # Cleanup must neither erase unverified state nor mask process shutdown.
        return


def serve(
    config: Config | None = None,
    *,
    startup_identity: object | None = None,
) -> None:  # pragma: no cover - entrypoint
    import atexit
    import logging
    import uvicorn  # noqa: PLC0415

    # The daemon's stdout/stderr land in .usd-cli/daemon.log (usd_cli.daemon._spawn). Give
    # usd_core's INFO progress lines (remote-render packaging/upload stages, backend
    # lifecycle) real handlers + timestamps there — without this only WARNING+ reached
    # the log and an 8-minute packaging stall looked like a hang (benchmark task-14).
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.WARNING
    )
    logging.getLogger("usd_core").setLevel(
        logging.DEBUG
        if (
            os.environ.get("USD_CLI_DEBUG")
            or os.environ.get("OV_DEBUG")
            or os.environ.get("3DSC_DEBUG")
        )
        else logging.INFO
    )

    cfg = config or load_config()
    host = _normalized_bind_host(cfg.server.get("host", "127.0.0.1"))
    # port: explicit env (set by the CLI auto-start) > config > a free port
    port = int(
        os.environ.get("USD_CLI_SERVER_PORT") or cfg.server.get("port") or 0
    ) or _free_port(host)
    token = secrets.token_hex(32)
    instance_id = secrets.token_hex(16)
    from usd_cli.daemon import (
        DaemonStartupIdentity,
        _proc_start_token,
        child_startup_handshake,
        validate_child_startup_identity,
    )

    if startup_identity is None:
        startup_identity = child_startup_handshake(cfg)
    state_directory_descriptor: int | None = None
    state_path: Path | None = None
    if isinstance(startup_identity, DaemonStartupIdentity):
        validate_child_startup_identity(cfg, startup_identity)
        daemon_pid = startup_identity.pid
        process_start_token = startup_identity.process_start_token
        state_directory_descriptor = startup_identity.state_descriptor
        state_path = startup_identity.state_path
    else:
        try:
            daemon_pid, process_start_token = startup_identity  # type: ignore[misc]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("daemon startup identity is malformed") from exc
        if (
            daemon_pid != os.getpid()
            or _proc_start_token(daemon_pid) != process_start_token
        ):
            raise RuntimeError("daemon startup identity changed before server import")
    cleanup_state = lambda: _remove_server_state_if_owned(  # noqa: E731
        cfg,
        daemon_pid=daemon_pid,
        process_start_token=process_start_token,
        instance_id=instance_id,
        state_directory_descriptor=state_directory_descriptor,
    )
    cleanup_registered = False
    try:
        write_server_state(
            cfg,
            host,
            port,
            token,
            instance_id,
            daemon_pid=daemon_pid,
            process_start_token=process_start_token,
            state_directory_descriptor=state_directory_descriptor,
            state_path=state_path,
        )
        atexit.register(cleanup_state)
        cleanup_registered = True

        # Idle auto-shutdown (PR-9.5): exit once no command has arrived for
        # idle_timeout.
        idle_s = _parse_duration(cfg.server.get("idle_timeout"), default_s=1800)

        def _idle_watch() -> None:
            while True:
                time.sleep(min(60, max(5, idle_s // 4)))
                # A command in flight — in ANY session — is never idle: _IN_FLIGHT
                # totals across the session registry and _LAST_ACTIVITY is stamped on
                # accept, so a render outlasting idle_timeout would otherwise get the
                # daemon killed mid-render. Both are read under _BUSY_LOCK
                # (_should_idle_exit): reading them unlocked raced _exit_busy's
                # decrement-then-stamp and could exit the daemon between a long
                # command finishing and its response returning.
                if _should_idle_exit(idle_s):
                    cleanup_state()
                    os._exit(0)

        threading.Thread(target=_idle_watch, daemon=True).start()
        uvicorn.run(
            build_app(
                cfg,
                token=token,
                instance_id=instance_id,
                daemon_pid=daemon_pid,
                process_start_token=process_start_token,
            ),
            host=host,
            port=port,
            log_level="warning",
        )
    finally:
        cleanup_state()
        if cleanup_registered:
            atexit.unregister(cleanup_state)
        if isinstance(startup_identity, DaemonStartupIdentity):
            for descriptor in (
                startup_identity.state_descriptor,
                startup_identity.project_descriptor,
            ):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
