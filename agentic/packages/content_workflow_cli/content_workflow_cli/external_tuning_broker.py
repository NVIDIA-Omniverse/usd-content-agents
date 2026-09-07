# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper-owned external-runtime (BYOR) refinement sweep broker.

The broker is the enforcement point for the agent-owned external refinement
loop: the coding agent decides *what* active parameter search to run and
*whether* the rendered result matches the user's behavior goal, but every
sweep must be reserved here first. This is the BYOR counterpart of
``tuning_broker.PhysicsTuningBroker``, with the engine call swapped from the
built-in ``run_tune`` to ``physics_agent.tuning.external.run_external_tune``.

Structural differences from the in-house broker, all deliberate:

- **Sweeps are sequential.** The engine's refine semantics pin every
  parameter omitted from the next active search to the *previous* sweep's
  winning value. Pinned parameters are broker-owned state that advances only
  on a succeeded sweep, so concurrent sweeps would make "previous" ambiguous.
- **Evidence is published to the child-readable iteration directory at sweep
  completion**, not after the child exits: the agent must visually review the
  rendered winner frames to judge the iteration. Every published file is
  digest-bound in the ledger, created exclusively, and re-verified at
  conclusion, so a post-publication edit fails the run closed.
- **No model call can occur inside the engine.** ``run_external_tune`` is
  fixed tuning: qualification evidence checks are programmatic and no VLM
  judge or LLM refiner exists on this path. The agent session's own review
  replaces both.
- **No candidate materialization.** There is no USD to patch: the deliverable
  is the accepted sweep's best parameters plus its exact recorded rollout,
  published by the wrapper from the broker's private engine run directory.

Qualification approval is wrapper-owned and validated before the broker is
constructed; every sweep re-validates it inside ``run_external_tune`` against
the runtime fingerprint, so a runtime edited mid-session fails closed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import secrets as _secrets
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO

from content_agent_workflows.physics.tuning_contract import sha256_file
from world_understanding.utils.artifacts import (
    copy_open_file_to_confined,
    open_confined_directory,
    open_confined_directory_at,
    open_confined_regular_file,
    write_bytes_to_confined,
)

from content_workflow_cli.tuning_broker import (
    BrokerError,
    _reject_child_output_links,
    canonical_json,
)

logger = logging.getLogger(__name__)

EXTERNAL_BROKER_SCHEMA_VERSION = "content-agents.physics-external-tuning-sweep.v1"
# BYOR trials run a customer simulator process per trial (Kit startup alone
# is minutes), so the per-sweep ceiling is far above the in-house default.
DEFAULT_EXTERNAL_SWEEP_DEADLINE_SECONDS = 7200.0
# Matches ExternalRefineInput.max_iterations — the engine loop this replaces.
DEFAULT_EXTERNAL_REFINE_MAX_ITERATIONS = 5
_ABORTED_SWEEP_STATUSES = frozenset({"cancelled", "deadline_exceeded"})
# Wrapper-owned run-directory entries a sweep's output_dir may not target.
_RESERVED_RUN_ENTRIES = frozenset({"final", "raw", "qualification", "reference_media"})
TERMINAL_SWEEP_STATUSES = frozenset(
    {"succeeded", "failed", "cancelled", "deadline_exceeded"}
)


@dataclass
class ExternalSweepBudget:
    """Hard sweep budget plus deadlines, owned by the broker."""

    max_sweeps: int
    max_trials_per_sweep: int
    sweep_deadline_seconds: float = DEFAULT_EXTERNAL_SWEEP_DEADLINE_SECONDS
    phase_deadline_seconds: float | None = None
    sweeps_reserved: int = 0

    def snapshot(self, *, phase_started_monotonic: float) -> dict[str, Any]:
        remaining_phase_seconds: float | None = None
        if self.phase_deadline_seconds is not None:
            remaining_phase_seconds = max(
                0.0,
                self.phase_deadline_seconds
                - (time.monotonic() - phase_started_monotonic),
            )
        return {
            "max_sweeps": self.max_sweeps,
            "max_trials_per_sweep": self.max_trials_per_sweep,
            "sweeps_reserved": self.sweeps_reserved,
            "sweeps_remaining": max(0, self.max_sweeps - self.sweeps_reserved),
            "sweep_deadline_seconds": self.sweep_deadline_seconds,
            "phase_deadline_seconds": self.phase_deadline_seconds,
            "phase_seconds_remaining": remaining_phase_seconds,
        }


# Engine-authored core tune artifacts pinned at sweep success and later
# republished digest-verified into an accepted run's final bundle
# (external_tune_results.json is republished as a portable rewrite after its
# raw bytes verify against the pinned digest).
CORE_TUNE_ARTIFACT_NAMES = (
    "run_spec.json",
    "best_params.json",
    "external_tune_results.json",
)


@dataclass
class ExternalSweepRecord:
    """Authoritative broker-side record of one reserved external sweep."""

    sweep_id: str
    iteration: int
    iter_dir: Path
    work_dir: Path
    active_search: dict[str, dict[str, float]]
    pinned_params: dict[str, float]
    max_trials: int
    status: str = "running"
    error: str | None = None
    engine_status: str | None = None
    evidence_path: Path | None = None
    evidence_sha256: str | None = None
    best_params: dict[str, float] = field(default_factory=dict)
    best_objective: float | None = None
    best_score: float | None = None
    n_trials: int = 0
    frames: list[dict[str, str]] = field(default_factory=list)
    distinct_frames: list[str] = field(default_factory=list)
    recording_path: Path | None = None
    recording_sha256: str | None = None
    response_metadata_path: Path | None = None
    response_metadata_sha256: str | None = None
    tune_artifacts: dict[str, str] = field(default_factory=dict)
    published_outputs: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, Any] | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def public_view(self) -> dict[str, Any]:
        return {
            "schema_version": EXTERNAL_BROKER_SCHEMA_VERSION,
            "sweep_id": self.sweep_id,
            "iteration": self.iteration,
            "status": self.status,
            "error": self.error,
            "engine_status": self.engine_status,
            "iter_dir": str(self.iter_dir),
            "active_search": {
                name: dict(bounds) for name, bounds in self.active_search.items()
            },
            "pinned_params": dict(self.pinned_params),
            "max_trials": self.max_trials,
            "evidence_path": str(self.evidence_path) if self.evidence_path else None,
            "evidence_sha256": self.evidence_sha256,
            "best_params": dict(self.best_params),
            "best_objective": self.best_objective,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "frames": [dict(frame) for frame in self.frames],
            "distinct_frames": list(self.distinct_frames),
            "recording_path": (
                str(self.recording_path) if self.recording_path else None
            ),
            "recording_sha256": self.recording_sha256,
            "response_metadata_path": (
                str(self.response_metadata_path)
                if self.response_metadata_path
                else None
            ),
            "response_metadata_sha256": self.response_metadata_sha256,
            "tune_artifacts": dict(self.tune_artifacts),
            "published_outputs": dict(self.published_outputs),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _default_external_tune_runner(tune_input: Any) -> Any:
    from physics_agent.tuning.external import run_external_tune

    return run_external_tune(tune_input)


def _open_dir_nofollow(name: str, *, dir_fd: int | None) -> int:
    """Open one directory component, refusing symlinks in that position."""

    try:
        return os.open(
            name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd
        )
    except OSError as exc:
        raise BrokerError(
            400, f"evidence path component is not a plain directory: {name}"
        ) from exc


def _copy_exclusive_at(source: Path, name: str, *, dir_fd: int) -> str:
    """Copy a regular file into a pinned directory fd; return its sha256.

    ``O_CREAT | O_EXCL | O_NOFOLLOW`` relative to a held directory descriptor
    means a child re-pointing any path component between validation and write
    cannot redirect the write; the digest is computed from the bytes written
    through the descriptor, so it cannot diverge from the published file.
    """

    if source.is_symlink() or not source.is_file():
        raise BrokerError(409, f"evidence source is not a regular file: {source}")
    digest = hashlib.sha256()
    dest_fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=dir_fd,
    )
    with os.fdopen(dest_fd, "wb") as dest_stream, source.open("rb") as source_stream:
        while chunk := source_stream.read(1 << 20):
            digest.update(chunk)
            dest_stream.write(chunk)
        dest_stream.flush()
        os.fsync(dest_stream.fileno())
    return digest.hexdigest()


def _write_exclusive_at(data: bytes, name: str, *, dir_fd: int) -> str:
    """Write bytes into a pinned directory fd exclusively; return sha256."""

    dest_fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=dir_fd,
    )
    with os.fdopen(dest_fd, "wb") as dest_stream:
        dest_stream.write(data)
        dest_stream.flush()
        os.fsync(dest_stream.fileno())
    return hashlib.sha256(data).hexdigest()


class _DigestingReader:
    """Hash the exact source byte stream delivered to a confined copy."""

    def __init__(self, source: BinaryIO) -> None:
        self._source = source
        self._digest = hashlib.sha256()

    def fileno(self) -> int:
        return self._source.fileno()

    def read(self, size: int = -1) -> bytes:
        chunk = self._source.read(size)
        self._digest.update(chunk)
        return chunk

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _copy_exclusive_confined(source: Path, name: str, *, root: Any) -> str:
    """Copy through the portable confined backend and hash copied bytes."""

    if source.is_symlink() or not source.is_file():
        raise BrokerError(409, f"evidence source is not a regular file: {source}")
    with open_confined_directory(source.parent) as source_root:
        with open_confined_regular_file(
            source_root,
            source.name,
        ) as (source_stream, source_metadata):
            digesting_source = _DigestingReader(source_stream)
            published = copy_open_file_to_confined(
                root,
                name,
                digesting_source,  # type: ignore[arg-type]
                source_metadata,
                overwrite=False,
            )
    if not published:
        raise BrokerError(409, f"evidence destination already exists: {name}")
    return digesting_source.hexdigest()


def _write_exclusive_confined(data: bytes, name: str, *, root: Any) -> str:
    """Publish known bytes through the portable confined backend exclusively."""

    published = write_bytes_to_confined(
        root,
        name,
        data,
        overwrite=False,
        file_mode=0o600,
    )
    if not published:
        raise BrokerError(409, f"evidence destination already exists: {name}")
    return hashlib.sha256(data).hexdigest()


def _copy_evidence_file(source: Path, name: str, *, root: Any) -> str:
    if os.name == "nt":
        return _copy_exclusive_confined(source, name, root=root)
    return _copy_exclusive_at(source, name, dir_fd=root)


def _write_evidence_file(data: bytes, name: str, *, root: Any) -> str:
    if os.name == "nt":
        return _write_exclusive_confined(data, name, root=root)
    return _write_exclusive_at(data, name, dir_fd=root)


class ExternalTuningBroker:
    """Localhost BYOR sweep broker owned by the wrapper process.

    ``tune_runner`` is injectable for tests; production uses
    ``physics_agent.tuning.external.run_external_tune``.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        spec: Any,
        qualification_dir: Path,
        approval_digest: str,
        max_sweeps: int = DEFAULT_EXTERNAL_REFINE_MAX_ITERATIONS,
        max_trials_per_sweep: int,
        sweep_deadline_seconds: float = DEFAULT_EXTERNAL_SWEEP_DEADLINE_SECONDS,
        phase_deadline_seconds: float | None = None,
        private_dir: Path | None = None,
        tune_runner: Callable[[Any], Any] | None = None,
        iteration_spec_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self._spec = spec
        self._qualification_dir = Path(qualification_dir).resolve()
        self._approval_digest = approval_digest
        self.budget = ExternalSweepBudget(
            max_sweeps=max_sweeps,
            max_trials_per_sweep=max_trials_per_sweep,
            sweep_deadline_seconds=sweep_deadline_seconds,
            phase_deadline_seconds=phase_deadline_seconds,
        )
        self._tune_runner = tune_runner or _default_external_tune_runner
        self._iteration_spec_factory = (
            iteration_spec_factory or self._default_iteration_spec_factory
        )
        # Pinned-parameter carryover is broker-owned state, mirroring the
        # engine refine loop: parameters omitted from an iteration's active
        # search keep the previous succeeded sweep's winning values, seeded
        # from the qualified nominal parameters.
        self._pinned_params: dict[str, float] = {
            name: float(value)
            for name, value in spec.qualification.nominal_params.items()
        }
        self._phase_started_monotonic = time.monotonic()
        self._lock = threading.Lock()
        self._records: dict[str, ExternalSweepRecord] = {}
        self._workers: list[threading.Thread] = []
        self._cancel_events: dict[str, threading.Event] = {}
        self._deadline_timers: list[threading.Timer] = []
        self._closing = False
        self._private_dir_removed = False
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        if private_dir is not None:
            self._private_dir = Path(private_dir).resolve()
            self._private_dir_owned = False
        else:
            self._private_dir = Path(
                tempfile.mkdtemp(prefix="physics-external-tuning-broker-")
            )
            self._private_dir_owned = True
        if (
            self._private_dir == self.run_dir
            or self.run_dir in self._private_dir.parents
        ):
            raise ValueError(
                "external tuning broker private_dir must be outside the "
                "child-writable run directory"
            )
        self._private_dir.mkdir(parents=True, exist_ok=True)
        # The engine writes an approval audit record into qualification_dir on
        # every sweep with plain symlink-following opens. The run-dir copy sits
        # inside the child's writable sandbox, where a pre-planted
        # approval.json.tmp symlink would redirect that unsandboxed engine
        # write to any host path — so sweeps always run against a
        # broker-private copy taken before the child session starts (the run
        # tree is link-checked at invocation entry, before this copy).
        if self._qualification_dir.is_dir():
            private_qualification = self._private_dir / "qualification"
            shutil.copytree(
                self._qualification_dir, private_qualification, symlinks=False
            )
            self._qualification_dir = private_qualification
        self._secret = _secrets.token_bytes(32)
        self._ledger_path = self._private_dir / "external_sweep_ledger.jsonl"

    @staticmethod
    def _default_iteration_spec_factory(
        spec: Any, *, active_search: Mapping[str, Mapping[str, float]], iteration: int
    ) -> Any:
        from physics_agent.tuning.external import derive_iteration_spec

        return derive_iteration_spec(
            spec, active_search=active_search, iteration=iteration
        )

    @property
    def ledger_path(self) -> Path:
        """Path of the append-only HMAC-signed ledger (wrapper-only)."""

        return self._ledger_path

    @property
    def private_dir(self) -> Path:
        """The broker-private workspace root (wrapper-only)."""

        return self._private_dir

    # ------------------------------------------------------------------ http

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("Broker is not started.")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        broker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
                logger.debug("external-tuning-broker: " + fmt, *args)

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, indent=2).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise BrokerError(400, f"invalid JSON body: {exc}") from exc
                if not isinstance(payload, dict):
                    raise BrokerError(400, "JSON body must be an object")
                return payload

            def do_GET(self) -> None:  # noqa: N802
                try:
                    parts = [p for p in self.path.split("/") if p]
                    if parts == ["budget"]:
                        self._send(200, broker.budget_view())
                    elif parts == ["sweeps"]:
                        # Recovery surface: a client that lost its blocking
                        # run call (timeout, crash) can list reserved sweeps
                        # to recover the sweep_id it must cite.
                        self._send(200, {"sweeps": list(broker.ledger().values())})
                    elif len(parts) == 2 and parts[0] == "sweeps":
                        self._send(200, broker.sweep_view(parts[1]))
                    else:
                        self._send(404, {"error": f"unknown path {self.path}"})
                except BrokerError as exc:
                    self._send(exc.status, {"error": str(exc)})
                except Exception as exc:  # noqa: BLE001 - report, don't crash
                    self._send(500, {"error": str(exc)})

            def do_POST(self) -> None:  # noqa: N802
                try:
                    parts = [p for p in self.path.split("/") if p]
                    payload = self._read_json()
                    if parts == ["sweeps"]:
                        self._send(200, broker.request_sweep(payload))
                    else:
                        self._send(404, {"error": f"unknown path {self.path}"})
                except BrokerError as exc:
                    self._send(exc.status, {"error": str(exc)})
                except Exception as exc:  # noqa: BLE001 - report, don't crash
                    self._send(500, {"error": str(exc)})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="physics-external-tuning-broker",
            daemon=True,
        )
        self._server_thread.start()

    def close(self, *, wait_for_workers: bool = True) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        with self._lock:
            self._closing = True
            events = [
                event
                for sweep_id, event in self._cancel_events.items()
                if self._records[sweep_id].status == "running"
            ]
            finished_at = time.time()
            for sweep_id in self._cancel_events:
                record = self._records[sweep_id]
                if record.status == "running":
                    record.status = "cancelled"
                    record.error = "broker closed before sweep completed"
                    record.finished_at = finished_at
            timers = list(self._deadline_timers)
        for timer in timers:
            timer.cancel()
        for event in events:
            event.set()
        if wait_for_workers:
            for worker in self._workers:
                worker.join(timeout=10.0)

    def release_private_dir(self) -> bool:
        """Remove a broker-owned private directory once the wrapper is done.

        Unlike the in-house broker this is explicit, not automatic on close:
        the accepted sweep's engine run directory lives beneath the private
        directory, and the wrapper publishes the final bundle from it *after*
        the child session has exited and the decision chain has verified.

        Returns ``True`` when no broker-owned private directory remains on
        disk (removed by this call, removed earlier, or never owned by the
        broker). ``False`` means the release was skipped because a worker
        thread is still alive — e.g. a BYOR runtime that outlived the
        ``close()`` join — and the directory is still on disk; callers must
        report that leaked path rather than assume the workspace is gone.
        """

        with self._lock:
            if not self._private_dir_owned or self._private_dir_removed:
                return True
            if any(worker.is_alive() for worker in self._workers):
                return False
        # Success is confirmed by the directory actually being gone, not by
        # having attempted removal: rmtree can fail partway (e.g. a
        # non-traversable directory left by the external runtime), and a
        # leaked workspace must be reported, not recorded as released.
        shutil.rmtree(self._private_dir, ignore_errors=True)
        removed = not self._private_dir.exists()
        if removed:
            with self._lock:
                self._private_dir_removed = True
        else:
            logger.warning(
                "broker private workspace could not be fully removed: %s",
                self._private_dir,
            )
        return removed

    # --------------------------------------------------------------- actions

    def budget_view(self) -> dict[str, Any]:
        with self._lock:
            return self.budget.snapshot(
                phase_started_monotonic=self._phase_started_monotonic
            )

    def sweep_view(self, sweep_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(sweep_id)
            if record is None:
                raise BrokerError(404, f"unknown sweep_id {sweep_id!r}")
            return record.public_view()

    def sweep_run_dir(self, sweep_id: str) -> Path:
        """Return a succeeded sweep's engine run directory (wrapper-only).

        This path lives in the broker's private workspace; the wrapper uses
        it to publish the accepted result. Never expose it to the child.
        """

        with self._lock:
            record = self._records.get(sweep_id)
            if record is None:
                raise BrokerError(404, f"unknown sweep_id {sweep_id!r}")
            if record.status != "succeeded":
                raise BrokerError(
                    409, f"sweep {sweep_id} is {record.status}, not succeeded"
                )
            return record.work_dir / "tune"

    def _parse_active_search(
        self, payload: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        raw = payload.get("active_search")
        if raw is None:
            # Default to the runtime config's declared search so the agent's
            # first sweep needs no search authoring.
            return {
                param.name: {
                    "min": float(param.min_value),
                    "max": float(param.max_value),
                }
                for param in self._spec.params
            }
        if not isinstance(raw, dict) or not raw:
            raise BrokerError(400, "active_search must be a non-empty object")
        parsed: dict[str, dict[str, float]] = {}
        for name, bounds in raw.items():
            if not isinstance(bounds, dict) or {"min", "max"} - set(bounds):
                raise BrokerError(
                    400,
                    f"active_search parameter {name!r} must provide 'min' and 'max'",
                )
            try:
                parsed[str(name)] = {
                    "min": float(bounds["min"]),
                    "max": float(bounds["max"]),
                }
            except (TypeError, ValueError) as exc:
                raise BrokerError(
                    400, f"active_search parameter {name!r} bounds must be numbers"
                ) from exc
        # Catalog membership and bound ordering are validated synchronously,
        # before any budget is reserved: deferring them to the worker's spec
        # derivation would permanently consume a sweep on a typo'd parameter
        # name or inverted bounds without any simulator work done.
        catalog = {param.name: param for param in self._spec.parameter_catalog}
        for name, bounds in parsed.items():
            parameter = catalog.get(name)
            if parameter is None:
                raise BrokerError(
                    400,
                    f"active_search parameter {name!r} is not in the "
                    "qualified parameter catalog",
                )
            if not all(math.isfinite(value) for value in bounds.values()):
                raise BrokerError(
                    400,
                    f"active_search parameter {name!r} bounds must be finite",
                )
            if not bounds["min"] < bounds["max"]:
                raise BrokerError(
                    400,
                    f"active_search parameter {name!r} requires min < max",
                )
            if getattr(parameter, "integer", False) and not (
                bounds["min"].is_integer() and bounds["max"].is_integer()
            ):
                raise BrokerError(
                    400,
                    f"active_search parameter {name!r} is integer-valued; "
                    "bounds must be whole numbers",
                )
        return parsed

    def request_sweep(self, payload: dict[str, Any]) -> dict[str, Any]:
        iter_dir_raw = payload.get("output_dir")
        if not iter_dir_raw:
            raise BrokerError(400, "output_dir is required")
        iter_dir = _reject_child_output_links(self.run_dir, Path(str(iter_dir_raw)))
        # Reserved run-directory entries are wrapper-owned: a sweep published
        # under final/ would later be destroyed by final-bundle publication,
        # and raw/ or qualification/ would collide with wrapper artifacts.
        relative_parts = iter_dir.relative_to(self.run_dir).parts
        if relative_parts and relative_parts[0] in _RESERVED_RUN_ENTRIES:
            raise BrokerError(
                400,
                "output_dir may not target the reserved run entry "
                f"{relative_parts[0]!r}",
            )
        active_search = self._parse_active_search(payload)
        requested_trials = payload.get("max_trials")
        # The runtime config's declared optimizer.max_trials is a ceiling the
        # broker caps against, never replaces: a BYOR spec declaring few
        # trials (each launches a customer simulator) must not silently run
        # more under the wrapper's default budget. Mirrors the engine's own
        # refine-external, which honours the spec value per iteration.
        max_trials = min(
            self.budget.max_trials_per_sweep,
            int(self._spec.optimizer.max_trials),
        )
        if requested_trials is not None:
            try:
                max_trials = min(max_trials, int(requested_trials))
            except (TypeError, ValueError) as exc:
                raise BrokerError(
                    400, f"invalid max_trials: {requested_trials!r}"
                ) from exc
            if max_trials <= 0:
                raise BrokerError(400, "max_trials must be positive")

        # Let a just-finished worker unwind before gating below, so a child
        # that polled a terminal status is not spuriously refused while the
        # thread returns from the engine call.
        for worker in list(self._workers):
            if worker.is_alive():
                worker.join(timeout=5.0)
        with self._lock:
            if self._closing:
                raise BrokerError(409, "external tuning broker is closed")
            if any(
                record.status == "running" for record in self._records.values()
            ) or any(worker.is_alive() for worker in self._workers):
                # Sequential by design: pinned-parameter carryover makes the
                # "previous sweep's winner" ambiguous under concurrency. The
                # worker-liveness gate matters after a deadline expiry: the
                # timer exposes deadline_exceeded before the external runtime
                # has actually terminated, and a new sweep must not start
                # while the old runtime is still unwinding. 503 (not 409):
                # this refusal is transient and must not be conflated with
                # terminal budget exhaustion — the client maps it to a
                # retry-later exit code so the deadline-recovery path stays
                # reachable.
                running = [
                    record.sweep_id
                    for record in self._records.values()
                    if record.status == "running"
                ]
                # The running sweep_id is included so a client that lost its
                # blocking call can recover the id it must cite.
                raise BrokerError(
                    503,
                    "an external sweep is already running or its runtime is "
                    f"still terminating ({', '.join(running) or 'unwinding'}); "
                    "external sweeps are sequential — wait and retry, or "
                    "GET /sweeps to recover its record",
                )
            now = time.monotonic()
            phase_deadline = (
                self._phase_started_monotonic + self.budget.phase_deadline_seconds
                if self.budget.phase_deadline_seconds is not None
                else None
            )
            if phase_deadline is not None and phase_deadline <= now:
                raise BrokerError(
                    409,
                    "external tuning phase deadline exceeded; no further "
                    "sweeps allowed",
                )
            if self.budget.sweeps_reserved >= self.budget.max_sweeps:
                raise BrokerError(
                    409,
                    "sweep budget exhausted "
                    f"({self.budget.sweeps_reserved}/{self.budget.max_sweeps})",
                )
            self.budget.sweeps_reserved += 1
            iteration = self.budget.sweeps_reserved
            sweep_id = f"external-sweep-{iteration:03d}-{_secrets.token_hex(4)}"
            work_dir = Path(
                tempfile.mkdtemp(prefix=f"{sweep_id}-", dir=self._private_dir)
            )
            record = ExternalSweepRecord(
                sweep_id=sweep_id,
                iteration=iteration,
                iter_dir=iter_dir,
                work_dir=work_dir,
                active_search=active_search,
                pinned_params=dict(self._pinned_params),
                max_trials=max_trials,
            )
            self._records[sweep_id] = record
            cancel_event = threading.Event()
            self._cancel_events[sweep_id] = cancel_event
            deadline_monotonic = now + self.budget.sweep_deadline_seconds
            if phase_deadline is not None:
                deadline_monotonic = min(deadline_monotonic, phase_deadline)
            worker = threading.Thread(
                target=self._run_sweep,
                args=(record, deadline_monotonic, cancel_event),
                name=f"physics-external-tuning-{record.sweep_id}",
                daemon=True,
            )
            self._workers.append(worker)
            worker.start()
        return {
            **record.public_view(),
            "budget": self.budget_view(),
        }

    def _run_sweep(
        self,
        record: ExternalSweepRecord,
        deadline_monotonic: float,
        cancel_event: threading.Event,
    ) -> None:
        def expire_deadline() -> None:
            with self._lock:
                if record.status == "running":
                    record.status = "deadline_exceeded"
                    record.error = "sweep deadline exceeded"
                    record.finished_at = time.time()
            cancel_event.set()

        deadline_timer = threading.Timer(
            max(0.0, deadline_monotonic - time.monotonic()), expire_deadline
        )
        deadline_timer.daemon = True
        with self._lock:
            self._deadline_timers.append(deadline_timer)
        deadline_timer.start()
        try:
            with self._lock:
                if (
                    record.status == "running"
                    and time.monotonic() >= deadline_monotonic
                ):
                    record.status = "deadline_exceeded"
                    record.error = "sweep deadline exceeded"
                    record.finished_at = time.time()
                aborted_before_start = record.status in _ABORTED_SWEEP_STATUSES
            if aborted_before_start:
                self._append_ledger(record)
                return
            iteration_spec = self._iteration_spec_factory(
                self._spec,
                active_search=record.active_search,
                iteration=record.iteration,
            )
            if record.max_trials != iteration_spec.optimizer.max_trials:
                iteration_spec = replace(
                    iteration_spec,
                    optimizer=replace(
                        iteration_spec.optimizer, max_trials=record.max_trials
                    ),
                )
            from physics_agent.tuning.external import ExternalTuneInput

            result = self._tune_runner(
                ExternalTuneInput(
                    config=iteration_spec,
                    output_dir=record.work_dir / "tune",
                    approval_digest=self._approval_digest,
                    qualification_dir=self._qualification_dir,
                    fixed_params=dict(record.pinned_params),
                    render_winning_trial=True,
                    cancel_event=cancel_event,
                )
            )
            self._finish_sweep(record, result, deadline_monotonic)
        except Exception as exc:  # noqa: BLE001 - record failure durably
            with self._lock:
                if record.status not in _ABORTED_SWEEP_STATUSES:
                    if time.monotonic() >= deadline_monotonic:
                        record.status = "deadline_exceeded"
                        record.error = "sweep deadline exceeded"
                    else:
                        record.status = "failed"
                        record.error = f"{type(exc).__name__}: {exc}"
                    record.finished_at = time.time()
            self._append_ledger(record)
        finally:
            deadline_timer.cancel()

    def _finish_sweep(
        self,
        record: ExternalSweepRecord,
        result: Any,
        deadline_monotonic: float,
    ) -> None:
        engine_status = str(getattr(result, "status", "") or "")
        succeeded = bool(getattr(result, "success", False)) and (
            engine_status == "completed"
        )
        cancelled = bool(getattr(result, "cancelled", False))
        publish_error: str | None = None
        evidence_payload: dict[str, Any] | None = None
        with self._lock:
            # The aborted state is authoritative: if the deadline timer or
            # close() marked the sweep terminal while the engine was
            # returning, a client may already have authored a valid
            # failed-sweep decision against the digest-less record, and a
            # late publication would retroactively invalidate it (and write
            # into the run directory after shutdown).
            aborted_before_publish = record.status in _ABORTED_SWEEP_STATUSES
        if succeeded and not aborted_before_publish:
            try:
                evidence_payload = self._publish_evidence(record, result)
            except Exception as exc:  # noqa: BLE001 - evidence must be complete
                succeeded = False
                publish_error = f"evidence publication failed: {exc}"

        with self._lock:
            record.engine_status = engine_status or None
            best_params = dict(getattr(result, "best_params", {}) or {})
            record.best_params = {
                name: float(value) for name, value in best_params.items()
            }
            best_objective = getattr(result, "best_objective", None)
            record.best_objective = (
                float(best_objective) if best_objective is not None else None
            )
            best_score = getattr(result, "best_score", None)
            record.best_score = float(best_score) if best_score is not None else None
            record.n_trials = int(getattr(result, "n_trials", 0))
            record.evidence = evidence_payload
            if record.status in _ABORTED_SWEEP_STATUSES:
                pass
            elif time.monotonic() >= deadline_monotonic:
                record.status = "deadline_exceeded"
                record.error = "sweep deadline exceeded"
            elif cancelled:
                record.status = "cancelled"
            elif succeeded:
                record.status = "succeeded"
                # Engine refine semantics: parameters omitted from the next
                # active search retain this sweep's winning values.
                self._pinned_params.update(record.best_params)
            else:
                record.status = "failed"
                record.error = (
                    publish_error
                    or getattr(result, "error", None)
                    or f"external tune ended with status {engine_status or 'failed'}"
                )
            if record.finished_at is None:
                record.finished_at = time.time()
        self._append_ledger(record)

    def _publish_evidence(
        self, record: ExternalSweepRecord, result: Any
    ) -> dict[str, Any]:
        """Copy reviewable evidence into the child-readable iteration dir.

        Published at completion (not after child exit) because the agent must
        review the frames to judge the sweep. Every file is digest-bound in
        the evidence packet and the ledger; the wrapper rehashes at
        conclusion, so post-publication edits fail closed.

        All writes go through directory descriptors opened ``O_NOFOLLOW``
        from the run directory down: the child knows ``sweep_id`` while a
        sweep runs and could otherwise pre-plant any component (for example
        ``frames``) as a symlink between validation and write, redirecting
        these unsandboxed broker writes outside the run directory.
        """

        _reject_child_output_links(self.run_dir, record.iter_dir)
        evidence_dir = _reject_child_output_links(
            self.run_dir, record.iter_dir / f"evidence-{record.sweep_id}"
        )
        relative = evidence_dir.relative_to(self.run_dir)

        tune_dir = record.work_dir / "tune"
        work_root = record.work_dir.resolve()
        rendered: list[Path] = []
        for frame in getattr(result, "rendered_frames", []) or []:
            frame_path = Path(frame)
            if not frame_path.is_file():
                # The engine returned this frame as part of the winner
                # rollout; publishing the remainder would present a
                # truncated motion sequence as complete review evidence
                # while the render metadata still describes the full range.
                raise BrokerError(
                    409,
                    f"engine-rendered frame is missing at publication: "
                    f"{frame_path}; a truncated rollout must not become "
                    "review evidence",
                )
            # The engine is trusted, but a traversal path here would make the
            # broker publish arbitrary readable bytes as evidence.
            if not frame_path.resolve().is_relative_to(work_root):
                raise BrokerError(
                    409,
                    f"rendered frame escapes the engine work directory: {frame_path}",
                )
            rendered.append(frame_path)
        if not rendered:
            raise BrokerError(
                409,
                "sweep succeeded but produced no rendered winner frames; "
                "visual review evidence is mandatory",
            )
        recording_source: Path | None = None
        artifacts = getattr(result, "artifacts", {}) or {}
        recording_raw = artifacts.get("best_recording")
        if recording_raw:
            source = Path(recording_raw)
            if not source.is_absolute():
                source = tune_dir / source
            if source.is_file():
                recording_source = source
        if recording_source is None:
            raise BrokerError(
                409,
                "sweep succeeded but published no winner recording; frames "
                "without their source recording are not reproducible evidence",
            )
        evidence_settings = getattr(self._spec, "evidence", None)
        playback_renderer = getattr(evidence_settings, "playback_renderer", None)
        if not playback_renderer:
            raise BrokerError(
                409,
                "engine spec names no evidence playback renderer; render "
                "provenance is mandatory for published evidence",
            )
        if str(playback_renderer) != "ovrtx":
            # AGENTS.md requires final visual evidence to come from the
            # shared OVRTX render path. "remote" is any REST render service
            # and its response carries no verifiable OVRTX identity
            # attestation, so accepting it here would let a generic or
            # misconfigured endpoint mint final review evidence. Fail closed
            # until remote responses carry provable OVRTX provenance.
            raise BrokerError(
                409,
                f"evidence playback renderer {playback_renderer!r} does not "
                "attest OVRTX provenance; final review evidence must render "
                "through the OVRTX backend",
            )
        # The actual backend render response, written by
        # render_time_sampled_usd beside the frames it produced. Final
        # evidence must carry the real response metadata, not a
        # reconstruction from requested settings.
        response_metadata_source = rendered[0].parent / "render_response_metadata.json"
        if not response_metadata_source.is_file():
            raise BrokerError(
                409,
                "winner frames carry no render response metadata; "
                "reproducible render provenance is mandatory",
            )
        try:
            response_metadata = json.loads(
                response_metadata_source.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise BrokerError(
                409,
                f"winner render response metadata is unreadable: {exc}",
            ) from exc
        # The metadata records the renderer that actually produced the
        # frames; requiring it to match the spec catches an engine that
        # rendered through a different backend than the one configured.
        metadata_renderer = (
            response_metadata.get("renderer")
            if isinstance(response_metadata, dict)
            else None
        )
        if metadata_renderer != "ovrtx":
            raise BrokerError(
                409,
                "winner render response metadata names renderer "
                f"{metadata_renderer!r}, not the required OVRTX backend",
            )

        frames: list[dict[str, str]] = []
        seen_digests: dict[str, str] = {}
        distinct_frames: list[str] = []
        recording_path = evidence_dir / "best_recording.usd"
        with ExitStack() as directory_stack:
            if os.name == "nt":
                run_root = directory_stack.enter_context(
                    open_confined_directory(self.run_dir)
                )
                try:
                    evidence_fd = directory_stack.enter_context(
                        open_confined_directory_at(
                            run_root,
                            relative.as_posix(),
                            create=True,
                            mode=0o700,
                            exclusive_create=True,
                        )
                    )
                except FileExistsError as exc:
                    raise BrokerError(
                        409,
                        f"evidence destination already exists: {evidence_dir}",
                    ) from exc
                frames_fd = directory_stack.enter_context(
                    open_confined_directory_at(
                        evidence_fd,
                        "frames",
                        create=True,
                        mode=0o700,
                        exclusive_create=True,
                    )
                )
            else:
                fd = _open_dir_nofollow(str(self.run_dir), dir_fd=None)
                directory_stack.callback(os.close, fd)
                for part in relative.parts[:-1]:
                    try:
                        os.mkdir(part, 0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                    fd = _open_dir_nofollow(part, dir_fd=fd)
                    directory_stack.callback(os.close, fd)
                try:
                    os.mkdir(relative.parts[-1], 0o700, dir_fd=fd)
                except FileExistsError as exc:
                    raise BrokerError(
                        409, f"evidence destination already exists: {evidence_dir}"
                    ) from exc
                evidence_fd = _open_dir_nofollow(relative.parts[-1], dir_fd=fd)
                directory_stack.callback(os.close, evidence_fd)
                os.mkdir("frames", 0o700, dir_fd=evidence_fd)
                frames_fd = _open_dir_nofollow("frames", dir_fd=evidence_fd)
                directory_stack.callback(os.close, frames_fd)

            for index, source in enumerate(rendered):
                name = f"frame_{index:04d}.png"
                digest = _copy_evidence_file(source, name, root=frames_fd)
                destination = evidence_dir / "frames" / name
                frames.append({"path": str(destination), "sha256": digest})
                # Fully settled assets render byte-identical frames; expose a
                # distinct subset so agent runners with duplicate-image limits
                # can review evidence without submitting redundant bytes.
                if digest not in seen_digests:
                    seen_digests[digest] = str(destination)
                    distinct_frames.append(str(destination))

            recording_sha256 = _copy_evidence_file(
                recording_source, "best_recording.usd", root=evidence_fd
            )
            # The engine pinned the winning recording's digest when it
            # validated and published best_recording.usd, and the frames
            # were rendered from those bytes. Hashing only the copy would
            # let a lingering BYOR descendant rewrite the file between
            # engine return and this copy and mint a broker digest for a
            # recording that never produced the reviewed frames.
            engine_recording = (getattr(result, "selected_evidence", None) or {}).get(
                "published_recording"
            ) or {}
            engine_recording_sha256 = str(
                engine_recording.get("sha256") or ""
            ).removeprefix("sha256:")
            if recording_sha256 != engine_recording_sha256:
                raise BrokerError(
                    409,
                    "winner recording does not match the engine-pinned "
                    f"digest: copied {recording_sha256}, engine recorded "
                    f"{engine_recording_sha256 or '<missing>'}",
                )
            response_metadata_sha256 = _copy_evidence_file(
                response_metadata_source,
                "render_response_metadata.json",
                root=evidence_fd,
            )
            payload, evidence_sha256 = self._write_evidence_payload(
                record=record,
                result=result,
                evidence_fd=evidence_fd,
                frames=frames,
                distinct_frames=distinct_frames,
                recording_path=recording_path,
                recording_sha256=recording_sha256,
                response_metadata_path=evidence_dir / "render_response_metadata.json",
                response_metadata_sha256=response_metadata_sha256,
                playback_renderer=str(playback_renderer),
                evidence_settings=evidence_settings,
            )
            # Durability: the ledger digests reference these bytes, so the
            # directory entries must survive a host crash too.
            if os.name != "nt":
                os.fsync(frames_fd)
                os.fsync(evidence_fd)

        # Core tune artifacts are pinned now, while the sweep's success is
        # being established: an accepted run's publication rehashes them
        # against these digests, so a post-sweep mutation of the engine's
        # tune directory fails closed instead of silently shipping changed
        # or missing bytes. Every core artifact is mandatory — a sweep that
        # cannot pin one must not become succeeded, or final publication
        # would silently omit it from an accepted bundle.
        tune_artifacts: dict[str, str] = {}
        for name in CORE_TUNE_ARTIFACT_NAMES:
            source_file = tune_dir / name
            if not source_file.is_file():
                raise BrokerError(
                    409,
                    f"succeeded sweep is missing core tune artifact {name!r}; "
                    "the deliverable would be incomplete",
                )
            tune_artifacts[name] = sha256_file(source_file)

        # Declared winner outputs (spec.publish_artifacts) were digest-pinned
        # by the engine when it promoted them into tune/outputs; snapshot
        # those digests so final publication republishes exactly the
        # engine-validated bytes and requires every declared output.
        published_outputs: dict[str, str] = {}
        for name, descriptor in (
            getattr(result, "published_outputs", None) or {}
        ).items():
            relative = str((descriptor or {}).get("path") or "")
            output_digest = str((descriptor or {}).get("sha256") or "").removeprefix(
                "sha256:"
            )
            if not relative or len(output_digest) != 64:
                raise BrokerError(
                    409,
                    f"declared winner output {name!r} carries no digest-bound "
                    "descriptor",
                )
            published_outputs[relative] = output_digest

        evidence_path = evidence_dir / "evidence.json"
        with self._lock:
            if record.status in _ABORTED_SWEEP_STATUSES:
                # The deadline timer or close() fired mid-publication. A
                # client may already have authored a valid failed-sweep
                # decision against the digest-less terminal record, so the
                # record must not gain digests now; the staged files are
                # inert without record references.
                raise BrokerError(
                    409,
                    "sweep was aborted during evidence publication; "
                    "discarding the staged evidence",
                )
            record.evidence_path = evidence_path
            record.evidence_sha256 = evidence_sha256
            record.frames = frames
            record.distinct_frames = distinct_frames
            record.recording_path = recording_path
            record.recording_sha256 = recording_sha256
            record.response_metadata_path = (
                evidence_dir / "render_response_metadata.json"
            )
            record.response_metadata_sha256 = response_metadata_sha256
            record.tune_artifacts = tune_artifacts
            record.published_outputs = published_outputs
        return payload

    def _write_evidence_payload(
        self,
        *,
        record: ExternalSweepRecord,
        result: Any,
        evidence_fd: Any,
        frames: list[dict[str, str]],
        distinct_frames: list[str],
        recording_path: Path,
        recording_sha256: str,
        response_metadata_path: Path,
        response_metadata_sha256: str,
        playback_renderer: str,
        evidence_settings: Any,
    ) -> tuple[dict[str, Any], str]:
        """Build the evidence packet and write it through the pinned fd."""

        from physics_agent.tuning.external.artifacts import external_trial_payload

        history = [
            external_trial_payload(trial, include_internal_artifacts=False)
            for trial in (getattr(result, "history", []) or [])
        ]
        payload: dict[str, Any] = {
            "schema_version": EXTERNAL_BROKER_SCHEMA_VERSION,
            "sweep_id": record.sweep_id,
            "iteration": record.iteration,
            "task": self._spec.task,
            "objective": {
                "name": self._spec.objective.name,
                "unit": self._spec.objective.unit,
                "direction": self._spec.objective.direction,
            },
            "active_search": record.active_search,
            "pinned_params": record.pinned_params,
            "success": True,
            "n_trials": int(getattr(result, "n_trials", 0)),
            "best_params": dict(getattr(result, "best_params", {}) or {}),
            "best_objective": getattr(result, "best_objective", None),
            "best_score": getattr(result, "best_score", None),
            "selected_evidence": dict(getattr(result, "selected_evidence", {}) or {}),
            "history": history,
            "frames": frames,
            "distinct_frames": distinct_frames,
            "recording_path": str(recording_path),
            "recording_sha256": recording_sha256,
            # Reproducibility provenance for the published render evidence:
            # the renderer identity and settings that produced the frames,
            # bound to the exact source recording digest (AGENTS.md requires
            # final visual evidence to carry its render metadata).
            "render_provenance": {
                "renderer": playback_renderer,
                "render_mode": getattr(evidence_settings, "render_mode", None),
                "width": getattr(evidence_settings, "width", None),
                "height": getattr(evidence_settings, "height", None),
                "num_sensor_updates": getattr(
                    evidence_settings, "num_sensor_updates", None
                ),
                "source_recording_sha256": recording_sha256,
                # The actual backend render response captured beside the
                # frames, digest-bound like every other evidence file.
                "response_metadata_path": str(response_metadata_path),
                "response_metadata_sha256": response_metadata_sha256,
            },
            "budget": self.budget_view(),
        }
        evidence_bytes = json.dumps(payload, indent=2, default=str).encode("utf-8")
        evidence_sha256 = _write_evidence_file(
            evidence_bytes,
            "evidence.json",
            root=evidence_fd,
        )
        return payload, evidence_sha256

    # ---------------------------------------------------------------- ledger

    def _append_ledger(self, record: ExternalSweepRecord) -> None:
        with self._lock:
            payload = record.public_view()
            signature = hmac.new(
                self._secret,
                canonical_json(payload).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            line = json.dumps({"record": payload, "hmac_sha256": signature})
            with self._ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def ledger(self) -> dict[str, dict[str, Any]]:
        """Authoritative in-memory sweep records, keyed by sweep_id."""

        with self._lock:
            return {
                sweep_id: record.public_view()
                for sweep_id, record in self._records.items()
            }

    def verify_claim(
        self, claim: dict[str, Any], *, allow_missing_evidence: bool = False
    ) -> tuple[bool, str]:
        """Check an agent-claimed sweep reference against broker state.

        Returns ``(ok, reason)``. Broker records are the source of truth: a
        claim of an unknown sweep or with mismatched digests is rejected, and
        the referenced evidence file is rehashed on disk so a post-sweep edit
        fails closed. A FAILED sweep is still a valid reference — a
        revise/stop decision legitimately cites the failed sweep it learned
        from. ``allow_missing_evidence`` tolerates a claim that names a
        sweep with published evidence but omits the digest — appropriate for
        ``stop`` decisions, whose contract keeps the digest optional; a
        *present* digest is always verified.
        """

        sweep_id = str(claim.get("sweep_id") or "")
        with self._lock:
            record = self._records.get(sweep_id)
        if record is None:
            return False, f"sweep_id {sweep_id!r} has no broker record"
        claimed = claim.get("evidence_sha256")
        if record.evidence_sha256 and not claimed and not allow_missing_evidence:
            return False, (
                f"evidence_sha256 missing from the decision for sweep "
                f"{sweep_id}; digest bindings are mandatory"
            )
        if claimed and not record.evidence_sha256:
            return False, (
                f"evidence_sha256 claimed for sweep {sweep_id} but the broker "
                "recorded none"
            )
        if claimed and record.evidence_sha256 and claimed != record.evidence_sha256:
            return False, (
                f"evidence_sha256 mismatch for sweep {sweep_id}: claimed "
                f"{claimed}, broker recorded {record.evidence_sha256}"
            )
        # Every digest-addressed published file is rehashed, not just
        # evidence.json: the recording and render-response metadata are the
        # reproducibility anchors of the frames, so a post-sweep edit to
        # either must fail the claim closed.
        digest_bound = (
            ("evidence file", record.evidence_path, record.evidence_sha256),
            ("winner recording", record.recording_path, record.recording_sha256),
            (
                "render response metadata",
                record.response_metadata_path,
                record.response_metadata_sha256,
            ),
        )
        for label, path, recorded_digest in digest_bound:
            if path is None or not recorded_digest:
                continue
            if not Path(path).is_file():
                return False, f"{label} for sweep {sweep_id} is missing: {path}"
            current = sha256_file(path)
            if current != recorded_digest:
                return False, (
                    f"{label} for sweep {sweep_id} was modified after "
                    f"the sweep: recorded {recorded_digest}, on disk "
                    f"{current}"
                )
        return True, "ok"

    def verify_reviewed_frames(
        self, sweep_id: str, reviewed_frames: list[dict[str, str]]
    ) -> tuple[bool, str]:
        """Verify agent-claimed reviewed frames against broker records.

        Every claimed frame must be one this sweep published, its digest must
        match the broker's record, and the bytes on disk must still hash to
        it. At least one frame is required — an accept without reviewed
        evidence is not verifiable.
        """

        with self._lock:
            record = self._records.get(sweep_id)
            published = (
                {frame["path"]: frame["sha256"] for frame in record.frames}
                if record is not None
                else {}
            )
        if record is None:
            return False, f"sweep_id {sweep_id!r} has no broker record"
        if not reviewed_frames:
            return False, "reviewed_frames is empty; visual review is mandatory"
        for claim in reviewed_frames:
            path = str(claim.get("path") or "")
            digest = str(claim.get("sha256") or "")
            recorded = published.get(path)
            if recorded is None:
                return False, (
                    f"reviewed frame {path!r} was not published by sweep {sweep_id}"
                )
            if digest != recorded:
                return False, (
                    f"reviewed frame {path!r} digest mismatch: claimed "
                    f"{digest}, broker recorded {recorded}"
                )
            if not Path(path).is_file():
                return False, f"reviewed frame is missing on disk: {path}"
            current = sha256_file(path)
            if current != recorded:
                return False, (
                    f"reviewed frame {path!r} was modified after the sweep: "
                    f"recorded {recorded}, on disk {current}"
                )
        return True, "ok"
