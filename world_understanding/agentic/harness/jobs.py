# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic long-running job manager for harness recipe execution."""

from __future__ import annotations

import asyncio
import builtins
import inspect
import json
import logging
import re
import threading
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"succeeded", "failed", "canceled", "blocked"}
_KIND_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_RESERVED_EVENT_KEYS = {
    "index",
    "event",
    "job_id",
    "kind",
    "status",
    "stage",
    "message",
    "time",
    "data",
}


class LongRunCancelled(RuntimeError):
    """Raised when a long-running job observes cooperative cancellation."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_compatible(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_json_compatible(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value))
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _normalize_result(result: Any) -> dict[str, Any]:
    """Return a JSON-compatible mapping for persisted snapshots."""
    if result is None:
        return {}
    normalized = _json_compatible(result)
    if isinstance(normalized, Mapping):
        return dict(normalized)
    return {"value": normalized}


def _status_from_result(result: Mapping[str, Any]) -> str:
    status = str(result.get("status") or "").lower()
    status_aliases = {
        "success": "succeeded",
        "successful": "succeeded",
        "done": "succeeded",
        "cancelled": "canceled",
        "error": "failed",
    }
    status = status_aliases.get(status, status)
    has_ok = "ok" in result
    if result.get("ok") is False and status not in {"blocked", "canceled"}:
        return "failed"
    if status in TERMINAL_STATUSES:
        return status
    if status and not has_ok:
        return "failed"
    if has_ok:
        return "succeeded" if bool(result["ok"]) else "failed"
    return "succeeded"


async def _await_any(awaitable: Awaitable[Any]) -> Any:
    """Await any Awaitable accepted by LongRunCallable."""
    return await awaitable


@dataclass
class JobRuntime:
    """Runtime controls passed to a long-running job runner."""

    job_id: str
    kind: str
    job_dir: Path
    output_dir: Path
    cancel_event: threading.Event
    emit_callback: Callable[[str, str, str, dict[str, Any] | None], None]

    def emit(
        self,
        stage: str,
        message: str,
        *,
        event: str = "log",
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a structured event for status consumers."""
        self.emit_callback(event, stage, message, data)

    def cancel_requested(self) -> bool:
        """Return whether cooperative cancellation has been requested."""
        return self.cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise when cooperative cancellation has been requested."""
        if self.cancel_requested():
            raise LongRunCancelled("job cancellation requested")


type LongRunCallable = Callable[
    [dict[str, Any], JobRuntime],
    Any | Awaitable[Any],
]


@dataclass
class LongRunningJob:
    """Persistent state for one in-process long-running job."""

    job_id: str
    kind: str
    request: dict[str, Any]
    job_dir: Path
    output_dir: Path
    status: str = "queued"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    completed_at: str | None = None
    stage: str = "queued"
    message: str = "queued"
    progress_count: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    resumed_from: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class LongRunningJobManager:
    """In-process background runner with persistent status and event replay."""

    def __init__(
        self,
        root: str | Path,
        runner: LongRunCallable,
        *,
        kind: str = "harness",
        output_dir_name: str = "output",
        max_loaded_jobs: int = 200,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.kind = self._validate_kind(kind)
        self.output_dir_name = output_dir_name
        self.max_loaded_jobs = max(1, max_loaded_jobs)
        self._jobs: dict[str, LongRunningJob] = {}
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)

    def start(
        self,
        request: Mapping[str, Any] | None = None,
        *,
        resumed_from: str | None = None,
        kind: str | None = None,
    ) -> dict[str, Any]:
        """Start a background job and return its initial snapshot."""
        job_kind = self._validate_kind(kind or self.kind)
        job_id = (
            f"{job_kind}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        job_dir = self._job_dir_for_id(job_id)
        request_copy = dict(request or {})
        output_dir = self._resolve_output_dir(job_dir, request_copy.get("output_dir"))
        job_dir.mkdir(parents=True, exist_ok=True)
        request_copy["output_dir"] = str(output_dir)
        job = LongRunningJob(
            job_id=job_id,
            kind=job_kind,
            request=request_copy,
            job_dir=job_dir,
            output_dir=output_dir,
            resumed_from=resumed_from,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._emit_locked(job, "queued", stage="queued", message="queued")
            thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
            job.thread = thread
            self._persist_locked(job)
            thread.start()
            return self.snapshot(job_id)

    def snapshot(self, job_id: str) -> dict[str, Any]:
        """Return current job state."""
        with self._lock:
            job = self._get_job_locked(job_id)
            return self._snapshot_locked(job)

    def list(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent job snapshots."""
        with self._lock:
            self._load_recent_locked()
            jobs = sorted(
                self._jobs.values(), key=lambda job: job.created_at, reverse=True
            )
            return [self._snapshot_locked(job) for job in jobs[: max(1, limit)]]

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Request cooperative cancellation for a running job."""
        with self._lock:
            job = self._get_job_locked(job_id)
            if job.status in TERMINAL_STATUSES:
                return self._snapshot_locked(job)
            job.cancel_event.set()
            job.status = "cancel_requested"
            self._emit_locked(
                job,
                "cancel_requested",
                stage="cancel",
                message="cancellation requested",
            )
            self._persist_locked(job)
            return self._snapshot_locked(job)

    def resume(self, job_id: str) -> dict[str, Any]:
        """Start a replacement job from a terminal job's saved request."""
        with self._lock:
            job = self._get_job_locked(job_id)
            if job.status not in TERMINAL_STATUSES:
                raise RuntimeError(f"job {job_id} is still {job.status}")
            request = dict(job.request)
            request.pop("output_dir", None)
        return self.start(request, resumed_from=job_id, kind=job.kind)

    def event_batches(
        self,
        job_id: str,
        *,
        since: int = 0,
        follow: bool = True,
        keepalive_s: float = 15.0,
    ) -> Iterator[builtins.list[dict[str, Any]]]:
        """Yield event batches, optionally following with keepalive pings."""
        next_index = max(0, int(since))
        while True:
            with self._cond:
                job = self._get_job_locked(job_id)
                if next_index < len(job.events):
                    batch = job.events[next_index:]
                    next_index = len(job.events)
                    done = job.status in TERMINAL_STATUSES and next_index >= len(
                        job.events
                    )
                elif job.status in TERMINAL_STATUSES or not follow:
                    return
                else:
                    self._cond.wait(timeout=keepalive_s)
                    job = self._get_job_locked(job_id)
                    if next_index >= len(job.events):
                        batch = [
                            {
                                "index": next_index,
                                "event": "ping",
                                "job_id": job_id,
                                "kind": job.kind,
                                "status": job.status,
                                "time": _now_iso(),
                                "keepalive": True,
                            }
                        ]
                    else:
                        continue
                    done = False
            yield batch
            if done:
                return

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._get_job_locked(job_id)
            job.status = "running"
            job.started_at = _now_iso()
            self._emit_locked(job, "started", stage="start", message="job started")
            self._persist_locked(job)

        def emit_callback(
            event: str,
            stage: str,
            message: str,
            data: dict[str, Any] | None = None,
        ) -> None:
            with self._lock:
                current = self._get_job_locked(job_id)
                self._emit_locked(
                    current,
                    event,
                    stage=stage,
                    message=message,
                    data=data,
                )
                self._persist_locked(current)

        try:
            with self._lock:
                job = self._get_job_locked(job_id)
                request = dict(job.request)
                runtime = JobRuntime(
                    job_id=job.job_id,
                    kind=job.kind,
                    job_dir=job.job_dir,
                    output_dir=job.output_dir,
                    cancel_event=job.cancel_event,
                    emit_callback=emit_callback,
                )

            result = self.runner(request, runtime)
            if inspect.isawaitable(result):
                # Each async runner executes inside this worker thread's short-lived loop.
                result = asyncio.run(_await_any(result))

            with self._lock:
                job = self._get_job_locked(job_id)
                normalized_result = _normalize_result(result)
                job.completed_at = _now_iso()
                if job.cancel_event.is_set():
                    job.result = {
                        "ok": False,
                        "status": "canceled",
                        "canceled_after_result": True,
                        "runner_result": normalized_result,
                    }
                    job.status = "canceled"
                    self._emit_locked(
                        job, "canceled", stage="cancel", message="job canceled"
                    )
                else:
                    job.result = normalized_result
                    job.status = _status_from_result(job.result)
                    event = {
                        "succeeded": "completed",
                        "failed": "failed",
                        "canceled": "canceled",
                        "blocked": "blocked",
                    }[job.status]
                    stage = {
                        "succeeded": "complete",
                        "failed": "error",
                        "canceled": "cancel",
                        "blocked": "blocked",
                    }[job.status]
                    self._emit_locked(
                        job,
                        event,
                        stage=stage,
                        message=f"job {job.status}",
                        data={"ok": job.result.get("ok"), "result_status": job.status},
                    )
                self._persist_locked(job)
                self._evict_loaded_jobs_locked()
        except (LongRunCancelled, asyncio.CancelledError) as exc:
            with self._lock:
                job = self._get_job_locked(job_id)
                job.completed_at = _now_iso()
                job.error = str(exc) or "job cancellation requested"
                job.status = "canceled"
                self._emit_locked(
                    job, "canceled", stage="cancel", message="job canceled"
                )
                self._persist_locked(job)
                self._evict_loaded_jobs_locked()
        except Exception as exc:
            with self._lock:
                job = self._get_job_locked(job_id)
                job.completed_at = _now_iso()
                job.error = str(exc)
                if job.cancel_event.is_set():
                    job.status = "canceled"
                    self._emit_locked(
                        job, "canceled", stage="cancel", message="job canceled"
                    )
                else:
                    job.status = "failed"
                    self._emit_locked(
                        job,
                        "failed",
                        stage="error",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                self._persist_locked(job)
                self._evict_loaded_jobs_locked()

    def _emit_locked(
        self,
        job: LongRunningJob,
        event: str,
        *,
        stage: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        job.stage = stage
        job.message = message
        job.updated_at = _now_iso()
        if event in {"log", "started", "completed", "failed", "canceled", "blocked"}:
            job.progress_count += 1
        row = {
            "index": len(job.events),
            "event": event,
            "job_id": job.job_id,
            "kind": job.kind,
            "status": job.status,
            "stage": stage,
            "message": message,
            "time": job.updated_at,
        }
        if data:
            event_data = _json_compatible(data)
            row["data"] = event_data
            for key, value in event_data.items():
                if key not in _RESERVED_EVENT_KEYS:
                    row[key] = value
        job.events.append(row)
        # TODO: Buffer high-frequency event/state writes outside this manager
        # lock if callers begin emitting many events per second.
        events_path = job.job_dir / "events.jsonl"
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        self._cond.notify_all()

    def _snapshot_locked(self, job: LongRunningJob) -> dict[str, Any]:
        return {
            "ok": job.status == "succeeded",
            "job_id": job.job_id,
            "kind": job.kind,
            "status": job.status,
            "stage": job.stage,
            "message": job.message,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "progress_count": job.progress_count,
            "event_count": len(job.events),
            "output_dir": str(job.output_dir),
            "job_dir": str(job.job_dir),
            "resumed_from": job.resumed_from,
            "cancel_requested": job.cancel_event.is_set(),
            "result": job.result,
            "error": job.error,
            "log_tail": job.events[-50:],
        }

    def _persist_locked(self, job: LongRunningJob) -> None:
        state = self._snapshot_locked(job)
        state["request"] = job.request
        path = job.job_dir / "job_state.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)

    def _validate_kind(self, kind: str) -> str:
        if not isinstance(kind, str) or _KIND_PATTERN.fullmatch(kind) is None:
            raise ValueError(f"Invalid job kind: {kind!r}; expected [A-Za-z0-9_.-]+")
        return kind

    def _job_dir_for_id(self, job_id: str) -> Path:
        job_id_path = Path(job_id)
        if (
            not job_id
            or job_id in {".", ".."}
            or job_id_path.is_absolute()
            or len(job_id_path.parts) != 1
            or "\\" in job_id
        ):
            raise ValueError(f"Invalid job id: {job_id!r}")
        job_dir = (self.root / job_id).resolve()
        if job_dir.parent != self.root:
            raise ValueError(f"Invalid job id: {job_id!r}")
        return job_dir

    def _resolve_output_dir(self, job_dir: Path, output_dir: Any) -> Path:
        job_dir = job_dir.resolve()
        if output_dir is None:
            output_path = Path(self.output_dir_name)
        else:
            output_path = Path(str(output_dir))
        if output_path.is_absolute():
            candidate = output_path.resolve()
        else:
            candidate = (job_dir / output_path).resolve()
        if candidate == job_dir or not candidate.is_relative_to(job_dir):
            raise ValueError("output_dir must be a subdirectory of the job directory")
        return candidate

    def _evict_loaded_jobs_locked(self) -> None:
        if len(self._jobs) <= self.max_loaded_jobs:
            return
        evictable = sorted(
            (
                job
                for job in self._jobs.values()
                if job.status in TERMINAL_STATUSES
                and not (job.thread is not None and job.thread.is_alive())
            ),
            key=lambda job: job.updated_at,
        )
        for job in evictable:
            if len(self._jobs) <= self.max_loaded_jobs:
                break
            del self._jobs[job.job_id]

    def _get_job_locked(self, job_id: str) -> LongRunningJob:
        self._job_dir_for_id(job_id)
        if job_id in self._jobs:
            return self._jobs[job_id]
        job = self._load_job_locked(job_id)
        if job is None:
            raise KeyError(job_id)
        self._jobs[job_id] = job
        self._evict_loaded_jobs_locked()
        return job

    def _load_recent_locked(self) -> None:
        for state_path in sorted(
            self.root.glob("*/job_state.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:50]:
            job_id = state_path.parent.name
            if job_id not in self._jobs:
                job = self._load_job_locked(job_id)
                if job is not None:
                    self._jobs[job_id] = job
        self._evict_loaded_jobs_locked()

    def _load_job_locked(self, job_id: str) -> LongRunningJob | None:
        state_path = self._job_dir_for_id(job_id) / "job_state.json"
        if not state_path.exists():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.warning("Failed to read job state for %s: %s", job_id, exc)
            return None
        except json.JSONDecodeError as exc:
            logger.warning("Invalid job state JSON for %s: %s", job_id, exc)
            return None
        request = dict(state.get("request") or {})
        raw_output_dir = state.get("output_dir") or request.get("output_dir")
        try:
            output_dir = self._resolve_output_dir(state_path.parent, raw_output_dir)
        except ValueError:
            logger.warning(
                "Invalid persisted output_dir for %s: %r", job_id, raw_output_dir
            )
            output_dir = self._resolve_output_dir(state_path.parent, None)
        kind = str(state.get("kind") or self.kind)
        try:
            kind = self._validate_kind(kind)
        except ValueError:
            logger.warning("Invalid persisted job kind for %s: %r", job_id, kind)
            kind = self.kind
        job = LongRunningJob(
            job_id=job_id,
            kind=kind,
            request=request,
            job_dir=state_path.parent,
            output_dir=output_dir,
            status=str(state.get("status") or "failed"),
            created_at=str(state.get("created_at") or _now_iso()),
            updated_at=str(state.get("updated_at") or _now_iso()),
            started_at=state.get("started_at"),
            completed_at=state.get("completed_at"),
            stage=str(state.get("stage") or "loaded"),
            message=str(state.get("message") or "loaded from disk"),
            progress_count=int(state.get("progress_count") or 0),
            result=state.get("result"),
            error=state.get("error"),
            resumed_from=state.get("resumed_from"),
        )
        # Cancellation is process-local: a reloaded job receives a fresh event
        # object and non-terminal loaded states become orphaned failures below.
        # Consumers should not treat cancel_requested as durable across restarts.
        events_path = state_path.parent / "events.jsonl"
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    job.events.append(event)
        if job.status not in TERMINAL_STATUSES:
            job.status = "failed"
            job.error = job.error or "job was loaded without an active worker"
            job.completed_at = job.completed_at or _now_iso()
            job.stage = "orphaned"
            job.message = "worker is no longer active; resume to restart"
            self._persist_locked(job)
        return job
