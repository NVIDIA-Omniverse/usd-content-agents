# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper-owned physics tuning sweep broker.

The broker is the enforcement point for the agent-owned physics tuning loop:
the coding agent decides *what* to sweep and *whether* the result is good, but
every sweep must be reserved here first. Budget state lives in this process's
memory plus a wrapper-private directory outside the run directory, and every
completed sweep is recorded with an HMAC signature whose secret never leaves
the wrapper. An agent that edits run-directory files or invokes ``run_tune``
directly cannot mint a verifiable sweep record, so wrapper verification fails
closed on any bypass.

The budget is protocol-enforced, not unforgeable: the ledger guarantees that
every *promotable* candidate came from a brokered, budgeted sweep, but it
cannot prevent a child session with engine access from running ``run_tune``
directly and using those unbudgeted runs as private search evidence. The
session prompt prohibits direct engine invocation; the hard guarantee is
narrower — nothing outside the ledger can ever be promoted.

Sweeps are sanitized to be pure inner-loop compute: the VLM judge is forced
off, no natural-language prompt reaches the engine's scenario interpreter,
and the scenario's ``target.vlm_check`` is rewritten to ``"off"`` so no model
call can occur inside the engine.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import logging
import os
import secrets as _secrets
import shutil
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml
from content_agent_workflows.physics.tuning_contract import sha256_file

logger = logging.getLogger(__name__)

BROKER_SCHEMA_VERSION = "content-agents.physics-tuning-sweep.v1"
DEFAULT_TOP_K = 5
# Per-sweep wall-clock ceiling for the broker's cooperative cancellation.
DEFAULT_SWEEP_DEADLINE_SECONDS = 3600.0
# Hard sweep budget for the agent-owned tuning outer loop. Matches
# physics_agent's RefineInput.max_iterations / ``physics-agent refine`` default.
DEFAULT_AGENTIC_TUNE_MAX_ITERATIONS = 5
# Formats candidates can be materialized in. ``SdfLayer.Export`` resolves the
# file format from the extension, so materializing directly in the promotion
# target's format keeps the digest chain byte-identical from materialization
# through revalidation to promotion. ``.usdz`` is excluded: it is a package
# format, and a plain layer export does not localize external references.
SUPPORTED_CANDIDATE_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
BROKER_TUNE_SEED = 42
_ABORTED_SWEEP_STATUSES = frozenset({"cancelled", "deadline_exceeded"})


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def check_tune_optimizer_available(optimizer: str) -> None:
    """Fail closed unless ``optimizer`` is built in or a registered, available
    plugin. Raises ``ValueError`` with an actionable message otherwise."""
    from world_understanding.optimization.contracts import SUPPORTED_OPTIMIZERS
    from world_understanding.optimization.registry import (
        get_registered_optimizer,
        load_optimizer_plugins,
    )

    if optimizer in SUPPORTED_OPTIMIZERS:
        return
    load_optimizer_plugins()
    plugin = get_registered_optimizer(optimizer)
    if plugin is None:
        raise ValueError(
            f"Unknown optimizer {optimizer!r}. No plugin is registered for this "
            "name. Verify that the required optional package is installed and "
            "exposes the 'world_understanding.optimizers' entry point."
        )
    if not plugin.is_available():
        raise ValueError(plugin.unavailable_message)


@dataclass
class SweepBudget:
    """Hard sweep/trial budget plus deadlines, owned by the broker."""

    max_sweeps: int
    max_trials_per_sweep: int
    sweep_deadline_seconds: float = DEFAULT_SWEEP_DEADLINE_SECONDS
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


@dataclass
class SweepRecord:
    """Authoritative broker-side record of one reserved sweep."""

    sweep_id: str
    iter_dir: Path
    work_dir: Path
    scenario_path: Path
    scenario_sha256: str
    physics_usd: Path
    physics_usd_sha256: str
    engine: str
    optimizer: str
    max_trials: int
    seed: int = BROKER_TUNE_SEED
    status: str = "running"
    error: str | None = None
    evidence_path: Path | None = None
    evidence_sha256: str | None = None
    best_score: float | None = None
    n_trials: int = 0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] | None = None
    materialized: dict[int, dict[str, str]] = field(default_factory=dict)
    published_dir: Path | None = None
    # Resolved scenario parameter bindings recovered from the sweep's
    # tune_results.json. ``bindings_recovered`` distinguishes "recovered as
    # None (legacy param-name patching)" from "artifact missing/unreadable";
    # only recovered candidates may be materialized, so the promoted USD is
    # patched exactly the way the engine patched the evaluated trials.
    resolved_bindings: list[dict[str, Any]] | None = None
    bindings_recovered: bool = False
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def public_view(self) -> dict[str, Any]:
        return {
            "schema_version": BROKER_SCHEMA_VERSION,
            "sweep_id": self.sweep_id,
            "status": self.status,
            "error": self.error,
            "iter_dir": str(self.iter_dir),
            "scenario_path": str(self.scenario_path),
            "scenario_sha256": self.scenario_sha256,
            "physics_usd": str(self.physics_usd),
            "physics_usd_sha256": self.physics_usd_sha256,
            "engine": self.engine,
            "optimizer": self.optimizer,
            "max_trials": self.max_trials,
            "seed": self.seed,
            "evidence_path": str(self.evidence_path) if self.evidence_path else None,
            "evidence_sha256": self.evidence_sha256,
            "n_trials": self.n_trials,
            "best_score": self.best_score,
            "evidence": copy.deepcopy(self.evidence),
            "published_dir": (
                str(self.published_dir) if self.published_dir is not None else None
            ),
            "published_evidence_path": (
                str(self.published_dir / "evidence.json")
                if self.published_dir is not None and self.evidence_path is not None
                else None
            ),
            "bindings_recovered": self.bindings_recovered,
            "resolved_binding_count": (
                len(self.resolved_bindings)
                if self.resolved_bindings is not None
                else None
            ),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "materialized": {
                str(trial): dict(entry)
                for trial, entry in sorted(self.materialized.items())
            },
        }


class BrokerError(RuntimeError):
    """Broker request error with an HTTP status code."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _sanitize_scenario(scenario_path: Path, output_path: Path) -> Path:
    """Copy the agent-authored scenario, forcing model calls off.

    ``target.vlm_check`` is rewritten to ``"off"`` — both engine backends
    invoke a VLM verifier for any other value, which would put a model call
    back inside wrapper-invoked code.
    """

    raw = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise BrokerError(400, f"scenario is not a mapping: {scenario_path}")
    sanitized = copy.deepcopy(raw)
    target = sanitized.get("target")
    if not isinstance(target, dict):
        target = {}
        sanitized["target"] = target
    target["vlm_check"] = "off"
    # The judge block only matters when judging is enabled; drop it so a
    # stale judge config cannot influence anything.
    sanitized.pop("judge", None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(sanitized, sort_keys=False), encoding="utf-8")
    return output_path


def _scenario_parameter_names(scenario_path: Path) -> set[str]:
    """Read parameter names for wrapper-owned tuning policy checks."""

    try:
        raw = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BrokerError(
            400, f"scenario could not be read: {scenario_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        return set()
    parameters = raw.get("parameters")
    if not isinstance(parameters, list):
        return set()
    return {
        str(entry["name"])
        for entry in parameters
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }


def _localize_candidate_dependencies(
    candidate: Path, *, copy_roots: tuple[Path, ...]
) -> None:
    """Rewrite and copy the candidate's file-backed deps into its sidecar."""

    PhysicsTuningBroker._localize_candidate_dependencies_impl(candidate, copy_roots)


# Snapshot/copy budgets - kept in sync with the runner's closure budgets:
# sidecar members are child-authored, so a cheap sparse file or huge entry
# tree must overflow the budget instead of exhausting broker disk or time.
MAX_SNAPSHOT_MEMBER_BYTES = 512 * 1024 * 1024
MAX_SNAPSHOT_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
MAX_SNAPSHOT_MEMBER_COUNT = 4096


def _bounded_sidecar_entries(sidecar: Path, *, max_entries: int) -> list[Path] | None:
    """Regular files under ``sidecar`` via a bounded no-follow walk, or None.

    Every directory entry counts against the budget DURING the walk;
    symlinks and non-regular entries fail closed (mirrors the runner's
    closure walk).
    """

    files: list[Path] = []
    entries = 0
    stack = [sidecar]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as iterator:
                for entry in iterator:
                    entries += 1
                    if entries > max_entries:
                        return None
                    if entry.is_symlink():
                        return None
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        files.append(Path(entry.path))
                    else:
                        return None
        except OSError:
            return None
    return sorted(files)


def _copy_regular_file_nofollow(
    source_root: Path,
    source_relative: Path | str,
    dest: Path,
    *,
    max_bytes: int | None = None,
) -> tuple[str, int]:
    """Copy one regular file, refusing links at EVERY path component.

    ``O_NOFOLLOW`` on a single ``os.open`` protects only the final
    component: a child can replace an already-enumerated ancestor directory
    with a symlink and route the copy anywhere the broker can read.
    Traverse from the trusted ``source_root`` descriptor instead, opening
    each component with ``O_NOFOLLOW`` (directories) and finally the leaf
    with ``O_NOFOLLOW|O_NONBLOCK``, verifying the OPENED descriptor is a
    single-link regular file; a pre-planted FIFO fails fast instead of
    parking the worker.
    """

    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        # The snapshot trust chain REQUIRES descriptor-anchored no-follow
        # traversal (O_DIRECTORY/O_NOFOLLOW + dir_fd), which native Windows
        # does not provide. Official runtime targets are Linux, Linux
        # containers, and WSL2 (root AGENTS.md - native Windows is a
        # convenience, not a supported broker target); fail fast with a
        # clear error instead of an AttributeError at first use.
        raise BrokerError(
            500,
            "physics tuning sweeps require POSIX no-follow filesystem "
            "semantics (O_DIRECTORY/O_NOFOLLOW); run the broker on Linux, "
            "a Linux container, or WSL2",
        )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    component_flags = directory_flags | getattr(os, "O_NOFOLLOW", 0)
    leaf_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    parts = Path(source_relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise BrokerError(400, f"unsafe sweep input path: {source_relative}")
    try:
        fd = os.open(source_root, directory_flags)
    except OSError as exc:
        raise BrokerError(400, f"unreadable sweep input root: {source_root}") from exc
    descriptor: int | None = None
    try:
        try:
            for part in parts[:-1]:
                next_fd = os.open(part, component_flags, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            descriptor = os.open(parts[-1], leaf_flags, dir_fd=fd)
        except OSError as exc:
            raise BrokerError(
                400,
                "unreadable or link-substituted sweep input file: "
                f"{source_root / Path(source_relative)}",
            ) from exc
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            # st_nlink != 1 is rejected at THIS boundary: the copy sources are
            # child-writable sweep inputs, and the child shares the broker's
            # UID, so fs.protected_hardlinks permits aliasing same-UID files
            # from outside the approved roots into the sidecar. The
            # dedupe-by-hardlink layouts raised previously live in
            # parent-owned artifact stores, which never feed this path - the
            # sweep-input contract is plain regular files.
            raise BrokerError(
                400,
                "sweep input file is not a single-link plain regular file: "
                f"{source_root / Path(source_relative)}",
            )
        if max_bytes is not None and info.st_size > max_bytes:
            raise BrokerError(
                400,
                "sweep input file exceeds the snapshot budget: "
                f"{source_root / Path(source_relative)}",
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        total = 0
        with open(dest, "wb") as sink:
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise BrokerError(
                        400,
                        "sweep input file exceeds the snapshot budget: "
                        f"{source_root / Path(source_relative)}",
                    )
                digest.update(chunk)
                sink.write(chunk)
        return digest.hexdigest(), total
    finally:
        os.close(fd)
        if descriptor is not None:
            os.close(descriptor)


def _assert_self_contained_usd(physics_usd: Path) -> None:
    """Reject a physics USD that would not compose from the private workspace.

    Sweeps and ``materialize`` open a byte-identical snapshot of the finalized
    USD from the broker's private workspace, so any composition arc resolved
    relative to the original directory (sublayer, reference, payload) would
    silently drop there — USD composition errors are non-fatal, and the sweep
    would then score, and the wrapper could promote, a partially composed
    stage. The physics workflow normally exports a flattened output, but the
    same-suffix/same-directory branch of ``apply_physics`` preserves
    references, and the zero-decision branch returns the input asset as-is.

    Texture and other non-layer asset paths are not checked: they do not
    contribute prims to the drop/settle scene the sweep evaluates.
    """

    from pxr import UsdUtils

    try:
        layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(str(physics_usd))
    except Exception as exc:  # noqa: BLE001 - treat an unreadable USD as unusable
        raise BrokerError(
            400, f"could not inspect physics_usd composition: {exc}"
        ) from exc
    from world_understanding.functions.graphics.so_export import (
        is_runtime_resolved_asset_path,
    )

    external = [
        layer.identifier
        for layer in layers
        if not is_runtime_resolved_asset_path(layer.identifier)
        and Path(layer.identifier).resolve() != physics_usd.resolve()
    ]
    blocking_unresolved = [
        str(item) for item in unresolved if not is_runtime_resolved_asset_path(item)
    ]
    if external or blocking_unresolved:
        detail = ", ".join(sorted(external + blocking_unresolved)[:5])
        raise BrokerError(
            400,
            "physics_usd must be self-contained for tuning: sweeps run from a "
            "wrapper-private workspace, where composition arcs resolved "
            "relative to the original directory would silently drop. "
            f"External layer dependencies: {detail}",
        )


def _absolute_without_symlink_resolution(path: Path) -> Path:
    """Return a normalized absolute path without following existing links."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_child_output_links(run_dir: Path, output_dir: Path) -> Path:
    """Reject links already present in a child-selected logical output tree.

    Broker and engine writes never use this tree; they go to a wrapper-private
    workspace. This check still rejects an already-armed child path so callers
    get an explicit security error. A link created after this check is harmless
    because no unsandboxed broker write targets ``output_dir``.
    """

    logical = _absolute_without_symlink_resolution(output_dir)
    try:
        relative = logical.relative_to(run_dir)
    except ValueError as exc:
        raise BrokerError(
            400, f"output_dir must live under the run directory: {logical}"
        ) from exc

    current = run_dir
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise BrokerError(400, f"output_dir contains a symbolic link: {current}")
        if current.exists() and not current.is_dir():
            raise BrokerError(
                400, f"output_dir component is not a directory: {current}"
            )
        if not current.exists():
            break

    if logical.is_dir():
        for root, directories, files in os.walk(logical, followlinks=False):
            root_path = Path(root)
            for name in [*directories, *files]:
                candidate = root_path / name
                if candidate.is_symlink():
                    raise BrokerError(
                        400,
                        f"output_dir contains a symbolic link: {candidate}",
                    )
    return logical


def _default_tune_runner(tune_input: Any) -> Any:
    from physics_agent.tuning import run_tune

    return run_tune(tune_input)


class PhysicsTuningBroker:
    """Localhost sweep broker owned by the wrapper process.

    ``tune_runner`` is injectable for tests; production uses
    ``physics_agent.tuning.run_tune``.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        engine: str,
        optimizer: str = "auto",
        max_sweeps: int,
        max_trials_per_sweep: int,
        sweep_deadline_seconds: float = DEFAULT_SWEEP_DEADLINE_SECONDS,
        phase_deadline_seconds: float | None = None,
        top_k: int = DEFAULT_TOP_K,
        initial_physics_usd: Path | str | None = None,
        private_dir: Path | None = None,
        tune_runner: Callable[[Any], Any] | None = None,
        candidate_suffix: str = ".usda",
        forbidden_parameters: Iterable[str] = (),
        allow_rebuilt_inputs: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.engine = engine
        self.optimizer = optimizer
        self.top_k = max(1, top_k)
        self.forbidden_parameters = frozenset(
            str(value) for value in forbidden_parameters
        )
        self.allow_rebuilt_inputs = allow_rebuilt_inputs
        # Candidates are materialized in the promotion target's format so the
        # digest-verified bytes the agent accepts are exactly the bytes that
        # land at the user's requested output path — no post-verification
        # format conversion.
        normalized_suffix = candidate_suffix.lower()
        if normalized_suffix not in SUPPORTED_CANDIDATE_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_CANDIDATE_SUFFIXES))
            raise ValueError(
                f"Unsupported candidate suffix {candidate_suffix!r}; expected "
                f"one of: {supported}"
            )
        self._candidate_suffix = normalized_suffix
        # Digest of the finalized physics USD the tuning phase started from.
        # Sweep inputs must either match it or be digest-bound to a prior
        # ``revise_patch`` decision — anything else is an untrusted stand-in
        # whose evidence would not describe the asset being promoted.
        self._initial_physics_usd_sha256 = (
            sha256_file(initial_physics_usd)
            if initial_physics_usd is not None
            else None
        )
        self.budget = SweepBudget(
            max_sweeps=max_sweeps,
            max_trials_per_sweep=max_trials_per_sweep,
            sweep_deadline_seconds=sweep_deadline_seconds,
            phase_deadline_seconds=phase_deadline_seconds,
        )
        self._tune_runner = tune_runner or _default_tune_runner
        self._phase_started_monotonic = time.monotonic()
        self._lock = threading.Lock()
        self._records: dict[str, SweepRecord] = {}
        self._workers: list[threading.Thread] = []
        self._cancel_events: dict[str, threading.Event] = {}
        self._deadline_timers: list[threading.Timer] = []
        self._closing = False
        self._private_dir_removed = False
        self._private_dir_cleanup_scheduled = False
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        # Wrapper-private state: outside the run directory so the child agent
        # has no path to it from the prompt; the HMAC secret exists only here
        # and in this process.
        if private_dir is not None:
            self._private_dir = Path(private_dir).resolve()
            self._private_dir_owned = False
        else:
            self._private_dir = Path(tempfile.mkdtemp(prefix="physics-tuning-broker-"))
            self._private_dir_owned = True
        if (
            self._private_dir == self.run_dir
            or self.run_dir in self._private_dir.parents
        ):
            raise ValueError(
                "physics tuning broker private_dir must be outside the "
                "child-writable run directory"
            )
        self._private_dir.mkdir(parents=True, exist_ok=True)
        self._secret = _secrets.token_bytes(32)
        self._ledger_path = self._private_dir / "sweep_ledger.jsonl"

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
                logger.debug("tuning-broker: " + fmt, *args)

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, indent=2).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_file(self, path: Path, *, sha256: str) -> None:
                metadata = path.stat()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(metadata.st_size))
                self.send_header("X-Content-SHA256", sha256)
                self.send_header("X-Content-Suffix", path.suffix.lower())
                self.end_headers()
                with path.open("rb") as stream:
                    shutil.copyfileobj(stream, self.wfile)

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
                    elif len(parts) == 2 and parts[0] == "sweeps":
                        self._send(200, broker.sweep_view(parts[1]))
                    elif (
                        len(parts) == 4
                        and parts[0] == "sweeps"
                        and parts[2] == "materialized"
                    ):
                        path, digest = broker.materialized_candidate_file(
                            parts[1], parts[3]
                        )
                        self._send_file(path, sha256=digest)
                    elif (
                        len(parts) == 5
                        and parts[0] == "sweeps"
                        and parts[2] == "materialized"
                        and parts[4] == "closure"
                    ):
                        self._send(
                            200,
                            {
                                "members": broker.materialized_candidate_closure(
                                    parts[1], parts[3]
                                )
                            },
                        )
                    elif (
                        len(parts) == 6
                        and parts[0] == "sweeps"
                        and parts[2] == "materialized"
                        and parts[4] == "closure"
                    ):
                        path, digest = broker.materialized_candidate_closure_file(
                            parts[1], parts[3], parts[5]
                        )
                        self._send_file(path, sha256=digest)
                    elif (
                        len(parts) == 4
                        and parts[0] == "sweeps"
                        and parts[2] == "recordings"
                    ):
                        path, digest = broker.candidate_recording_file(
                            parts[1], parts[3]
                        )
                        self._send_file(path, sha256=digest)
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
                    elif (
                        len(parts) == 3
                        and parts[0] == "sweeps"
                        and parts[2] == "materialize"
                    ):
                        self._send(200, broker.materialize_request(parts[1], payload))
                    else:
                        self._send(404, {"error": f"unknown path {self.path}"})
                except BrokerError as exc:
                    self._send(exc.status, {"error": str(exc)})
                except Exception as exc:  # noqa: BLE001 - report, don't crash
                    self._send(500, {"error": str(exc)})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, name="physics-tuning-broker", daemon=True
        )
        self._server_thread.start()

    def close(self, *, wait_for_workers: bool = True) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        # Make cancellation authoritative before asking backends to stop. A
        # backend may ignore its event and return success later; that result
        # must never make a closed sweep materializable.
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
                worker.join(timeout=5.0)
        self._cleanup_owned_private_dir_if_ready()
        self._schedule_owned_private_dir_cleanup()

    def _cleanup_owned_private_dir_if_ready(
        self, *, completed_worker: threading.Thread | None = None
    ) -> None:
        # Only remove a broker-owned temporary directory, and only after all
        # workers have actually exited — a non-cooperative sweep that outlived
        # the join timeout is still writing its ledger, so removing the
        # directory under it would cause a FileNotFoundError crash. Callers
        # that supplied their own private_dir own its lifetime.
        with self._lock:
            ready = (
                self._closing
                and self._private_dir_owned
                and not self._private_dir_removed
                and all(
                    worker is completed_worker or not worker.is_alive()
                    for worker in self._workers
                )
            )
            if ready:
                self._private_dir_removed = True
        if ready:
            shutil.rmtree(self._private_dir, ignore_errors=True)

    def _schedule_owned_private_dir_cleanup(self) -> None:
        """Reap a private directory after non-cooperative workers eventually exit."""

        with self._lock:
            needs_reaper = (
                self._closing
                and self._private_dir_owned
                and not self._private_dir_removed
                and not self._private_dir_cleanup_scheduled
                and any(worker.is_alive() for worker in self._workers)
            )
            if needs_reaper:
                self._private_dir_cleanup_scheduled = True
                workers = list(self._workers)
            else:
                workers = []
        if not workers:
            return

        def wait_and_cleanup() -> None:
            for worker in workers:
                worker.join()
            self._cleanup_owned_private_dir_if_ready()

        threading.Thread(
            target=wait_and_cleanup,
            name="physics-tuning-private-dir-cleanup",
            daemon=True,
        ).start()

    def publish_artifacts(self) -> None:
        """Publish vetted broker artifacts after the child process has exited.

        Production calls this only after the child supervisor has terminated
        and the run-tree link gate has passed. This method independently
        rejects destination links, refuses source links/non-regular files, and
        creates every destination exclusively so a broker write never follows
        a child-controlled path.
        """

        with self._lock:
            if any(record.status == "running" for record in self._records.values()):
                raise BrokerError(409, "cannot publish while sweeps are still running")
            records = list(self._records.values())

        for record in records:
            _reject_child_output_links(self.run_dir, record.iter_dir)
            published_dir = record.iter_dir / "broker_artifacts" / record.sweep_id
            _reject_child_output_links(self.run_dir, published_dir)
            try:
                published_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
            except FileExistsError as exc:
                raise BrokerError(
                    409,
                    f"broker artifact destination already exists: {published_dir}",
                ) from exc

            try:
                for source in sorted(record.work_dir.rglob("*")):
                    if source.is_symlink():
                        raise BrokerError(
                            409,
                            f"broker artifact source is a symbolic link: {source}",
                        )
                    relative = source.relative_to(record.work_dir)
                    destination = published_dir / relative
                    if source.is_dir():
                        destination.mkdir(mode=0o700)
                    elif source.is_file():
                        with (
                            source.open("rb") as source_stream,
                            destination.open("xb") as destination_stream,
                        ):
                            shutil.copyfileobj(source_stream, destination_stream)
                    else:
                        raise BrokerError(
                            409,
                            f"broker artifact source is not a regular file: {source}",
                        )
            except Exception:
                shutil.rmtree(published_dir, ignore_errors=True)
                raise

            with self._lock:
                record.published_dir = published_dir
            self._append_ledger(record)

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

    def _phase_seconds_remaining(self) -> float | None:
        deadline = self.budget.phase_deadline_seconds
        if deadline is None:
            return None
        return deadline - (time.monotonic() - self._phase_started_monotonic)

    def _phase_deadline_exceeded(self) -> bool:
        remaining = self._phase_seconds_remaining()
        return remaining is not None and remaining <= 0.0

    def _validate_sweep_physics_usd(
        self, physics_usd: Path, physics_usd_sha256: str, payload: dict[str, Any]
    ) -> None:
        """Enforce the sweep-input trust boundary.

        The input must be the finalized physics USD the phase started from,
        or a rebuilt artifact digest-bound to a prior ``revise_patch``
        decision. Anything else could attach real broker evidence to an
        asset that is not the one being tuned/promoted.
        """

        if self._initial_physics_usd_sha256 is None:
            return
        if physics_usd_sha256 == self._initial_physics_usd_sha256:
            return
        if not self.allow_rebuilt_inputs:
            raise BrokerError(
                400,
                "rebuilt physics USD inputs are disabled for this tuning run; "
                "continue from the finalized physics USD",
            )
        decision_raw = payload.get("rebuilt_decision_path")
        if not decision_raw:
            raise BrokerError(
                400,
                "physics_usd is not the finalized input; sweeps over a rebuilt "
                "USD must cite the revise_patch decision that produced it via "
                "rebuilt_decision_path",
            )
        decision_path = Path(str(decision_raw)).resolve()
        if self.run_dir not in decision_path.parents:
            raise BrokerError(
                400,
                f"rebuilt_decision_path must live under the run directory: "
                f"{decision_path}",
            )
        if not decision_path.is_file():
            raise BrokerError(
                400, f"rebuilt_decision_path does not exist: {decision_path}"
            )
        from content_agent_workflows.physics import verify_decision_chain

        try:
            raw_dir = (self.run_dir / "raw").resolve()
            iteration = int(decision_path.stem.rsplit("_", 1)[1])
            if iteration < 1:
                raise ValueError("rebuilt decision iteration must be at least 1")
            expected_path = (
                raw_dir / f"physics_tuning_decision_{iteration}.json"
            ).resolve()
            if decision_path != expected_path:
                raise ValueError(
                    "rebuilt decision must use the canonical "
                    "raw/physics_tuning_decision_<iteration>.json path"
                )
            decision_paths = [
                raw_dir / f"physics_tuning_decision_{index}.json"
                for index in range(1, iteration + 1)
            ]
            missing = [str(path) for path in decision_paths if not path.is_file()]
            if missing:
                raise ValueError(
                    "rebuilt decision chain is incomplete; missing "
                    + ", ".join(missing)
                )
            decisions = verify_decision_chain(
                decision_paths,
                max_iterations=self.budget.max_sweeps,
            )
        except Exception as exc:  # noqa: BLE001 - agent-authored file
            raise BrokerError(
                400,
                f"rebuilt_decision_path does not name a valid decision chain: {exc}",
            ) from exc
        decision = decisions[-1]
        if decision.decision != "revise_patch":
            raise BrokerError(
                400,
                "rebuilt_decision_path must reference a revise_patch decision, "
                f"got {decision.decision!r}",
            )
        rebuilt_raw = decision.rebuilt_physics_usd
        if not rebuilt_raw or Path(rebuilt_raw).resolve() != physics_usd:
            raise BrokerError(
                400,
                "revise_patch decision does not name this physics_usd as its "
                "rebuilt artifact",
            )
        if decision.rebuilt_physics_usd_sha256 != physics_usd_sha256:
            raise BrokerError(
                400,
                "physics_usd digest does not match the revise_patch decision's "
                "rebuilt_physics_usd_sha256",
            )
        # A syntactically valid decision is not enough to authorize a new USD.
        # Its complete prior chain must cite broker records that still bind the
        # evidence it claims; otherwise a child could fabricate a standalone
        # revise_patch file and attach a budgeted sweep to arbitrary physics.
        for chain_decision in decisions:
            if chain_decision.sweep_id is None:
                continue
            ok, reason = self.verify_claim(
                {
                    "sweep_id": chain_decision.sweep_id,
                    "scenario_sha256": chain_decision.scenario_sha256,
                    "evidence_sha256": chain_decision.evidence_sha256,
                }
            )
            if not ok:
                raise BrokerError(
                    400,
                    "rebuilt_decision_path does not belong to a broker-backed "
                    f"decision chain: {reason}",
                )

    def request_sweep(self, payload: dict[str, Any]) -> dict[str, Any]:
        scenario_raw = payload.get("scenario_path")
        physics_raw = payload.get("physics_usd")
        iter_dir_raw = payload.get("output_dir")
        if not scenario_raw or not physics_raw or not iter_dir_raw:
            raise BrokerError(
                400, "scenario_path, physics_usd, and output_dir are required"
            )
        scenario_path = Path(str(scenario_raw)).resolve()
        physics_usd = Path(str(physics_raw)).resolve()
        iter_dir = _reject_child_output_links(self.run_dir, Path(str(iter_dir_raw)))
        if not scenario_path.is_file():
            raise BrokerError(400, f"scenario does not exist: {scenario_path}")
        if not physics_usd.is_file():
            raise BrokerError(400, f"physics USD does not exist: {physics_usd}")
        forbidden = sorted(
            self.forbidden_parameters & _scenario_parameter_names(scenario_path)
        )
        if forbidden:
            raise BrokerError(
                400,
                "scenario tunes wrapper-protected parameter(s): "
                + ", ".join(forbidden),
            )
        # The engine is broker-enforced: an agent-selected engine (notably
        # "fake") could mint a valid ledger record whose scores are synthetic
        # while generic revalidation still passes. Reject any mismatch rather
        # than silently overriding, so the agent gets an actionable error.
        requested_engine = payload.get("engine")
        if requested_engine is not None and str(requested_engine) != self.engine:
            raise BrokerError(
                400,
                f"engine is broker-enforced as {self.engine!r}; "
                f"requested {requested_engine!r}",
            )
        # The optimizer, unlike the engine, may be overridden per sweep, so it
        # must be validated here as well: budget is consumed at reservation,
        # and an unknown or unavailable optimizer would otherwise burn a sweep
        # by dying inside the worker before producing any evidence.
        optimizer = str(payload.get("optimizer") or self.optimizer)
        try:
            check_tune_optimizer_available(optimizer)
        except ValueError as exc:
            raise BrokerError(400, str(exc)) from exc
        physics_usd_sha256 = sha256_file(physics_usd)
        self._validate_sweep_physics_usd(physics_usd, physics_usd_sha256, payload)
        requested_trials = payload.get("max_trials")
        max_trials = self.budget.max_trials_per_sweep
        if requested_trials is not None:
            try:
                max_trials = min(max_trials, int(requested_trials))
            except (TypeError, ValueError) as exc:
                raise BrokerError(
                    400, f"invalid max_trials: {requested_trials!r}"
                ) from exc
            if max_trials <= 0:
                raise BrokerError(400, "max_trials must be positive")

        # Atomic reservation BEFORE any work starts: budget is consumed by
        # reservation, not by completion, so a crashed sweep still counts.
        with self._lock:
            if self._closing:
                raise BrokerError(409, "tuning broker is closed")
            now = time.monotonic()
            phase_deadline = (
                self._phase_started_monotonic + self.budget.phase_deadline_seconds
                if self.budget.phase_deadline_seconds is not None
                else None
            )
            if phase_deadline is not None and phase_deadline <= now:
                raise BrokerError(
                    409,
                    "tuning phase deadline exceeded; no further sweeps allowed",
                )
            if self.budget.sweeps_reserved >= self.budget.max_sweeps:
                raise BrokerError(
                    409,
                    "sweep budget exhausted "
                    f"({self.budget.sweeps_reserved}/{self.budget.max_sweeps})",
                )
            self.budget.sweeps_reserved += 1
            sweep_id = (
                f"sweep-{self.budget.sweeps_reserved:03d}-{_secrets.token_hex(4)}"
            )
            work_dir = Path(
                tempfile.mkdtemp(prefix=f"{sweep_id}-", dir=self._private_dir)
            )
            record = SweepRecord(
                sweep_id=sweep_id,
                iter_dir=iter_dir,
                work_dir=work_dir,
                scenario_path=scenario_path,
                scenario_sha256=sha256_file(scenario_path),
                physics_usd=physics_usd,
                physics_usd_sha256=physics_usd_sha256,
                engine=self.engine,
                optimizer=optimizer,
                max_trials=max_trials,
            )
            self._records[sweep_id] = record
            # Register the cancellation handle at reservation time so a
            # close() racing the worker's startup can still cancel it.
            cancel_event = threading.Event()
            self._cancel_events[sweep_id] = cancel_event
            # Anchor the deadline at reservation, not worker startup. A
            # queued worker must not get an additional full sweep window.
            deadline_monotonic = now + self.budget.sweep_deadline_seconds
            if phase_deadline is not None:
                deadline_monotonic = min(deadline_monotonic, phase_deadline)
            worker = threading.Thread(
                target=self._run_sweep_worker,
                args=(record, deadline_monotonic, cancel_event),
                name=f"physics-tuning-{record.sweep_id}",
                daemon=True,
            )
            self._workers.append(worker)
            worker.start()
        return {
            **record.public_view(),
            "budget": self.budget_view(),
        }

    def _run_sweep_worker(
        self,
        record: SweepRecord,
        deadline_monotonic: float,
        cancel_event: threading.Event,
    ) -> None:
        try:
            self._run_sweep(record, deadline_monotonic, cancel_event)
        finally:
            self._cleanup_owned_private_dir_if_ready(
                completed_worker=threading.current_thread()
            )

    def _run_sweep(
        self,
        record: SweepRecord,
        deadline_monotonic: float,
        cancel_event: threading.Event,
    ) -> None:
        # Keep the deadline distinct from ordinary cancellation.  A backend may
        # ignore ``cancel_event`` and return a late success, which must never
        # become eligible for materialization or promotion.
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
            physics_suffix = record.physics_usd.suffix
            if not physics_suffix or len(physics_suffix) > 16:
                physics_suffix = ".usd"
            # The trusted snapshot stays in BROKER-PRIVATE storage: the
            # original run directory remains child-writable while the sweep
            # runs, so a snapshot placed there (or at any deterministic leaf
            # a child can pre-create as a symlink) could be swapped after the
            # digest check. Texture/material references the decision patch
            # author left pointing at the original run directory (e.g.
            # inputs/source/*.png) are admitted by passing that SOURCE
            # directory to the tune runner as an explicit approved dependency
            # root instead of relocating the trusted bytes into it.
            trusted_physics_usd = (
                record.work_dir / f"physics_input{physics_suffix.lower()}"
            )
            _assert_self_contained_usd(record.physics_usd)
            shutil.copyfile(record.physics_usd, trusted_physics_usd)
            if sha256_file(trusted_physics_usd) != record.physics_usd_sha256:
                raise BrokerError(
                    400,
                    "physics_usd changed while the broker was snapshotting it",
                )
            # Snapshot the localized dependency sidecars alongside, under
            # their ORIGINAL names: a portable physics output references its
            # sibling "<original-name>_assets" directory relatively, so the
            # digest-checked snapshot (and every candidate materialize()
            # patches from it) composes only when those files exist in the
            # private workspace too. File-by-file so a child-planted symlink
            # is never dereferenced into broker storage.
            # ONE budget across every recognized sidecar: per-directory
            # resets would let a child plant the full allowance in each of
            # the current- and legacy-named directories.
            snapshot_budget = MAX_SNAPSHOT_TOTAL_BYTES
            snapshot_entries_left = MAX_SNAPSHOT_MEMBER_COUNT
            for sidecar in {
                record.physics_usd.with_name(record.physics_usd.name + "_assets"),
                record.physics_usd.with_name(record.physics_usd.stem + "_assets"),
            }:
                if sidecar.is_symlink() or not sidecar.is_dir():
                    continue
                members = _bounded_sidecar_entries(
                    sidecar, max_entries=snapshot_entries_left
                )
                if members is None:
                    raise BrokerError(
                        400,
                        "sidecar snapshot exceeds the entry budget or "
                        f"contains unsafe entries: {sidecar}",
                    )
                snapshot_entries_left -= len(members)
                for member in members:
                    # Component-safe no-follow copy anchored at the declared
                    # source directory: a child racing this loop can swap
                    # neither the member nor any enumerated ancestor for a
                    # symlink, and a FIFO fails fast. The budget is debited
                    # with the bytes the copy ACTUALLY read from the opened
                    # descriptor - an lstat-time size could be resized by
                    # the child between stat and open, repeatedly exceeding
                    # the advertised total.
                    _, copied_bytes = _copy_regular_file_nofollow(
                        record.physics_usd.parent,
                        member.relative_to(record.physics_usd.parent),
                        record.work_dir / sidecar.name / member.relative_to(sidecar),
                        max_bytes=min(MAX_SNAPSHOT_MEMBER_BYTES, snapshot_budget),
                    )
                    snapshot_budget -= copied_bytes
                    if snapshot_budget < 0:
                        raise BrokerError(
                            400,
                            "sidecar snapshot exceeds the total byte budget: "
                            f"{sidecar}",
                        )
            trusted_scenario = record.work_dir / "scenario_input.yaml"
            shutil.copyfile(record.scenario_path, trusted_scenario)
            if sha256_file(trusted_scenario) != record.scenario_sha256:
                raise BrokerError(
                    400,
                    "scenario changed while the broker was snapshotting it",
                )
            forbidden = sorted(
                self.forbidden_parameters & _scenario_parameter_names(trusted_scenario)
            )
            if forbidden:
                raise BrokerError(
                    400,
                    "snapshotted scenario tunes wrapper-protected parameter(s): "
                    + ", ".join(forbidden),
                )
            sanitized = _sanitize_scenario(
                trusted_scenario, record.work_dir / "scenario_sanitized.yaml"
            )
            from physics_agent.tuning import TuneInput

            result = self._tune_runner(
                TuneInput(
                    scenario=sanitized,
                    user_prompt=None,
                    physics_usd=trusted_physics_usd,
                    # The digest-checked snapshot lives in broker-private
                    # storage; references the decision-patch author left
                    # pointing at the ORIGINAL run directory (e.g.
                    # inputs/source/*.png) are admitted by naming that
                    # source directory as an approved dependency root.
                    approved_dependency_roots=[record.physics_usd.parent],
                    output_dir=record.work_dir,
                    engine=record.engine,
                    optimizer=record.optimizer,
                    max_trials=record.max_trials,
                    seed=record.seed,
                    enable_judge=False,
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
        self, record: SweepRecord, result: Any, deadline_monotonic: float
    ) -> None:
        history = list(getattr(result, "history", []) or [])
        scored = [trial for trial in history if not getattr(trial, "failed", False)]
        scored.sort(key=lambda trial: trial.score)
        candidates = []
        for rank, trial in enumerate(scored[: self.top_k], start=1):
            metrics = dict(getattr(trial, "backend_metrics", {}) or {})
            recording = metrics.get("recording_usd") or metrics.get("recording_usda")
            candidates.append(
                {
                    "rank": rank,
                    "trial_index": trial.trial_index,
                    "params": dict(trial.params),
                    "score": trial.score,
                    "recording": str(recording) if recording else None,
                    "backend_metrics": metrics,
                }
            )
        artifacts = {
            str(name): str(path)
            for name, path in (getattr(result, "artifacts", {}) or {}).items()
        }
        resolved_bindings, bindings_recovered = self._recover_resolved_bindings(
            record, artifacts
        )
        evidence = {
            "schema_version": BROKER_SCHEMA_VERSION,
            "sweep_id": record.sweep_id,
            "scenario_path": str(record.scenario_path),
            "scenario_sha256": record.scenario_sha256,
            "physics_usd": str(record.physics_usd),
            "physics_usd_sha256": record.physics_usd_sha256,
            "engine": getattr(result, "engine_used", "") or record.engine,
            "optimizer": getattr(result, "optimizer_used", "") or record.optimizer,
            "success": bool(getattr(result, "success", False)),
            "cancelled": bool(getattr(result, "cancelled", False)),
            "error": getattr(result, "error", None),
            "n_trials": int(getattr(result, "n_trials", 0)),
            "best_params": dict(getattr(result, "best_params", {}) or {}),
            "best_score": getattr(result, "best_score", None),
            "candidates": candidates,
            "artifacts": artifacts,
            "budget": self.budget_view(),
        }
        evidence_path = record.work_dir / "evidence.json"
        evidence_path.write_text(
            json.dumps(evidence, indent=2, default=str), encoding="utf-8"
        )
        with self._lock:
            record.evidence_path = evidence_path
            record.evidence_sha256 = sha256_file(evidence_path)
            best_score = getattr(result, "best_score", None)
            record.best_score = float(best_score) if best_score is not None else None
            record.n_trials = int(getattr(result, "n_trials", 0))
            record.candidates = candidates
            record.evidence = evidence
            record.resolved_bindings = resolved_bindings
            record.bindings_recovered = bindings_recovered
            # The monotonic deadline is authoritative even if the timer
            # thread has not run yet. Likewise, close() may already have
            # terminally cancelled this record while the backend kept running.
            if record.status in _ABORTED_SWEEP_STATUSES:
                pass
            elif time.monotonic() >= deadline_monotonic:
                record.status = "deadline_exceeded"
                record.error = "sweep deadline exceeded"
            elif getattr(result, "cancelled", False):
                record.status = "cancelled"
            elif getattr(result, "success", False):
                record.status = "succeeded"
            else:
                record.status = "failed"
                record.error = getattr(result, "error", None) or "tune failed"
            if record.finished_at is None:
                record.finished_at = time.time()
        self._append_ledger(record)

    def _recover_resolved_bindings(
        self, record: SweepRecord, artifacts: dict[str, str]
    ) -> tuple[list[dict[str, Any]] | None, bool]:
        """Recover the resolved scenario parameter bindings a sweep ran with.

        ``run_tune`` writes its own tuned USD via
        ``patch_physics_usd(..., bindings=get_resolved_bindings(scenario))``;
        the resolved bindings are persisted in ``tune_results.json`` under
        ``scenario.extra["resolved_parameter_bindings"]``. Materialization
        must apply candidate params through the same bindings, otherwise
        bound backend parameters (e.g. ``contact_ke``/``contact_kd``) are
        silently dropped and the promoted USD does not reproduce the
        evaluated physics.

        Returns ``(bindings, recovered)`` where ``bindings`` mirrors exactly
        what ``get_resolved_bindings`` would have returned (``None`` when the
        scenario carried no resolved bindings — legacy param-name patching)
        and ``recovered`` is False when the artifact is missing or unreadable.
        """

        results_raw = artifacts.get("tune_results.json")
        results_path = (
            Path(results_raw) if results_raw else record.work_dir / "tune_results.json"
        )
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "tuning-broker: could not recover resolved bindings for %s: %s",
                record.sweep_id,
                exc,
            )
            return None, False
        scenario = payload.get("scenario")
        extra = scenario.get("extra") if isinstance(scenario, dict) else None
        if not isinstance(extra, dict):
            return None, False
        if "resolved_parameter_bindings" not in extra:
            return None, True
        bindings = extra["resolved_parameter_bindings"]
        if not isinstance(bindings, list) or not all(
            isinstance(item, dict) for item in bindings
        ):
            return None, False
        return [dict(item) for item in bindings], True

    # ----------------------------------------------------------- materialize

    # ------------------------------------------------------- localization

    @staticmethod
    def _localize_candidate_dependencies_impl(
        candidate: Path, copy_roots: tuple[Path, ...]
    ) -> None:
        from pxr import Sdf, UsdUtils

        try:
            layer = Sdf.Layer.FindOrOpen(str(candidate))
        except Exception as exc:  # noqa: BLE001 - test doubles are not USD
            layer = None
            logger.warning(
                "skipping candidate dependency localization for %s: %s",
                candidate,
                exc,
            )
        if layer is None:
            # patch_physics_usd (trusted pxr code) just wrote this file, so
            # an unreadable candidate is a test double, not a production
            # artifact; production candidates always take the full
            # localize-and-verify path below.
            return
        # Portable naming contract (world_understanding so_export
        # portable_sidecar_name): downstream bundling discovers
        # "<output-name>_assets".
        sidecar = candidate.parent / f"{candidate.name}_assets"
        candidate_dir = candidate.parent.resolve()
        resolved_roots = tuple(root.resolve() for root in copy_roots)
        # ModifyAssetPaths invokes the callback once PER OCCURRENCE; a large
        # shared texture must be copied and hashed once, not per reference.
        localized: dict[Path, str] = {}
        # Aggregate localization budgets: many individually-valid members
        # must not copy an unbounded total into broker-private storage.
        localization_state = {
            "bytes_left": MAX_SNAPSHOT_TOTAL_BYTES,
            "members_left": MAX_SNAPSHOT_MEMBER_COUNT,
        }

        from world_understanding.functions.graphics.so_export import (
            is_runtime_resolved_asset_path,
        )

        def _rebase(asset_path: str) -> str:
            if not asset_path:
                return asset_path
            if is_runtime_resolved_asset_path(asset_path):
                # Resolver-owned assets (bare MDL tokens, omniverse:// and
                # other non-file URIs) intentionally stay runtime-resolved;
                # they are not filesystem dependencies to localize.
                return asset_path
            # Package-relative assets (library.usdz[textures/albedo.png]):
            # localize the OUTER package and preserve the inner selector -
            # treating the whole identifier as a filesystem path would leave
            # it unlocalized and fail self-containment for a valid shape.
            from pxr import Ar

            outer_path, inner_selector = Ar.SplitPackageRelativePathOuter(asset_path)
            if inner_selector:
                rewritten_outer = _rebase(outer_path)
                if rewritten_outer == outer_path:
                    return asset_path
                return Ar.JoinPackageRelativePath(rewritten_outer, inner_selector)
            source = Path(asset_path)
            if not source.is_absolute():
                source = candidate_dir / asset_path
            source = Path(os.path.abspath(os.fspath(source)))
            try:
                inside = source.relative_to(candidate_dir)
            except ValueError:
                pass
            else:
                # Already inside the candidate directory: keep authored
                # relative references verbatim, but rewrite ABSOLUTE
                # identifiers to the candidate-relative form - the candidate
                # outlives broker.close() under a different path, and an
                # absolute reference into the broker work directory would
                # dangle after promotion even though the local
                # ComputeAllDependencies verification passes here.
                if Path(asset_path).is_absolute():
                    return f"./{inside.as_posix()}"
                return asset_path
            if not source.is_file():
                # Not a resolvable file path (search-path token, shader id):
                # the post-localization ComputeAllDependencies verification
                # owns the unresolved-dependency decision.
                return asset_path
            containing_root = next(
                (
                    root
                    for root in resolved_roots
                    if source == root or root in source.parents
                ),
                None,
            )
            if containing_root is None:
                # Only the broker workspace and the declared source directory
                # may feed candidate bytes; anything else stays unrewritten
                # and the self-containment verification below rejects it.
                return asset_path
            memoized = localized.get(source)
            if memoized is not None:
                return memoized
            # Copy FIRST through the component-safe no-follow descriptor and
            # derive the digest from those exact bytes: is_file()/sha256_file
            # on the child-writable source path would follow a racing
            # symlink swap into files outside the declared roots.
            if localization_state["members_left"] <= 0:
                raise BrokerError(
                    400,
                    "candidate dependency closure exceeds the member budget "
                    f"while localizing: {candidate}",
                )
            staged = sidecar / f".staging_{source.name}"
            copied_digest, copied_bytes = _copy_regular_file_nofollow(
                containing_root,
                source.relative_to(containing_root),
                staged,
                max_bytes=min(
                    MAX_SNAPSHOT_MEMBER_BYTES, localization_state["bytes_left"]
                ),
            )
            localization_state["members_left"] -= 1
            localization_state["bytes_left"] -= copied_bytes
            if localization_state["bytes_left"] < 0:
                staged.unlink(missing_ok=True)
                raise BrokerError(
                    400,
                    "candidate dependency closure exceeds the total byte "
                    f"budget while localizing: {candidate}",
                )
            dest = sidecar / f"{copied_digest[:12]}_{source.name}"
            if dest.exists():
                staged.unlink(missing_ok=True)
            else:
                staged.replace(dest)
            rewritten = f"./{sidecar.name}/{dest.name}"
            localized[source] = rewritten
            return rewritten

        UsdUtils.ModifyAssetPaths(layer, _rebase)
        layer.Save()
        # Verify: the finished candidate must compose entirely from its own
        # directory - promotion outlives broker.close(), which deletes
        # everything else.
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(candidate))
        escaped: list[str] = []
        for identifier in [item.identifier for item in layers] + [
            str(item) for item in assets
        ]:
            if is_runtime_resolved_asset_path(identifier):
                continue
            try:
                Path(identifier).resolve().relative_to(candidate_dir)
            except (ValueError, OSError):
                escaped.append(identifier)
        blocking_unresolved = [
            str(item) for item in unresolved if not is_runtime_resolved_asset_path(item)
        ]
        if escaped or blocking_unresolved:
            detail = ", ".join(sorted(escaped + blocking_unresolved)[:5])
            raise BrokerError(
                500,
                f"materialized candidate is not self-contained: {detail}",
            )

    def materialize_request(
        self, sweep_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        trial_raw = payload.get("trial_index")
        try:
            trial_index = int(trial_raw)
        except (TypeError, ValueError) as exc:
            raise BrokerError(400, f"invalid trial_index: {trial_raw!r}") from exc
        return self.materialize(sweep_id, trial_index)

    def materialize(self, sweep_id: str, trial_index: int) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(sweep_id)
            if record is None:
                raise BrokerError(404, f"unknown sweep_id {sweep_id!r}")
            if record.status != "succeeded":
                raise BrokerError(
                    409,
                    f"sweep {sweep_id} is {record.status}; only succeeded "
                    "sweeps can be materialized",
                )
            candidate = next(
                (
                    entry
                    for entry in record.candidates
                    if entry["trial_index"] == trial_index
                ),
                None,
            )
            if candidate is None:
                raise BrokerError(
                    404,
                    f"trial {trial_index} is not a top-{self.top_k} candidate "
                    f"of sweep {sweep_id}",
                )
            if not record.bindings_recovered:
                raise BrokerError(
                    409,
                    f"sweep {sweep_id} candidates cannot be reproduced exactly: "
                    "resolved parameter bindings were not recovered from the "
                    "sweep artifacts",
                )
            resolved_bindings = (
                [dict(item) for item in record.resolved_bindings]
                if record.resolved_bindings is not None
                else None
            )
            cached = record.materialized.get(trial_index)
        if cached is not None:
            return {"sweep_id": sweep_id, "trial_index": trial_index, **cached}

        from physics_agent.tuning.usd_patch import patch_physics_usd

        output_usd = (
            record.work_dir
            / "candidates"
            / f"trial_{trial_index:04d}{self._candidate_suffix}"
        )
        patch_physics_usd(
            record.work_dir
            / (
                f"physics_input{record.physics_usd.suffix.lower()}"
                if record.physics_usd.suffix and len(record.physics_usd.suffix) <= 16
                else "physics_input.usd"
            ),
            output_usd,
            dict(candidate["params"]),
            bindings=resolved_bindings,
        )
        # Make the candidate SELF-CONTAINED and PORTABLE: Stage.Flatten()
        # inside patch_physics_usd can anchor file-backed asset paths to the
        # absolute snapshot or source location, so copying directories beside
        # the candidate rewrites nothing. Localize the candidate's actual
        # dependency closure instead: every file-backed asset is copied (via
        # no-follow descriptor reads) into the candidate's own sidecar and
        # the asset path is rewritten to that relative copy, then the result
        # is verified to compose entirely from the candidate's directory.
        _localize_candidate_dependencies(
            output_usd,
            copy_roots=(record.work_dir, record.physics_usd.parent),
        )
        entry = {
            "usd_path": str(output_usd),
            "usd_sha256": sha256_file(output_usd),
        }
        with self._lock:
            record.materialized[trial_index] = entry
        self._append_ledger(record)
        return {"sweep_id": sweep_id, "trial_index": trial_index, **entry}

    def materialized_candidate_file(
        self, sweep_id: str, trial_index_raw: str
    ) -> tuple[Path, str]:
        """Return one digest-verified candidate for agent-side export.

        The broker only reads its private file. The sandboxed client writes the
        response into the run directory, so an agent-controlled destination can
        never make the privileged broker follow a run-tree link.
        """

        try:
            trial_index = int(trial_index_raw)
        except (TypeError, ValueError) as exc:
            raise BrokerError(400, f"invalid trial_index: {trial_index_raw!r}") from exc
        materialized = self.materialize(sweep_id, trial_index)
        path = Path(str(materialized["usd_path"]))
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise BrokerError(
                404, f"materialized candidate is missing: {path}"
            ) from exc
        if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1:
            raise BrokerError(
                409, f"materialized candidate is not a private regular file: {path}"
            )
        digest = sha256_file(path)
        expected = str(materialized["usd_sha256"])
        if digest != expected:
            raise BrokerError(
                409, f"materialized candidate digest changed on disk: {path}"
            )
        return path, digest

    def materialized_candidate_closure(
        self, sweep_id: str, trial_index_raw: str
    ) -> list[dict[str, Any]]:
        """Enumerate the localized sidecar members of one materialized root.

        A localized candidate references its broker-private "*_assets"
        sidecar relatively; exporting only the root would leave a bundle
        that cannot compose outside the broker workspace. The listing is
        deterministic (sorted by relative path) so member indexes are
        stable across the listing and file endpoints."""

        try:
            trial_index = int(trial_index_raw)
        except (TypeError, ValueError) as exc:
            raise BrokerError(400, f"invalid trial_index: {trial_index_raw!r}") from exc
        materialized = self.materialize(sweep_id, trial_index)
        root = Path(str(materialized["usd_path"])).resolve(strict=True)
        candidate_dir = root.parent
        from pxr import UsdUtils

        try:
            dep_layers, dep_assets, dep_unresolved = UsdUtils.ComputeAllDependencies(
                str(root)
            )
        except Exception as exc:  # noqa: BLE001 - inspection failure blocks export
            raise BrokerError(
                500,
                f"materialized candidate closure inspection failed: {root}: {exc}",
            ) from exc
        from world_understanding.functions.graphics.so_export import (
            is_runtime_resolved_asset_path,
        )

        blocking_unresolved = [
            str(item)
            for item in dep_unresolved
            if not is_runtime_resolved_asset_path(item)
        ]
        if blocking_unresolved:
            detail = ", ".join(sorted(blocking_unresolved)[:5])
            raise BrokerError(
                500, f"materialized candidate has unresolved references: {detail}"
            )
        relative_members: set[str] = set()
        for identifier in [layer.identifier for layer in dep_layers] + [
            str(asset) for asset in dep_assets
        ]:
            if is_runtime_resolved_asset_path(identifier):
                # Resolver-owned (bare MDL / non-file URIs): intentionally
                # runtime-resolved, never a downloadable closure member.
                continue
            member = Path(identifier).resolve()
            if member == root:
                continue
            try:
                relative = member.relative_to(candidate_dir)
            except ValueError as exc:
                raise BrokerError(
                    500,
                    "materialized candidate dependency escapes the candidate "
                    f"directory: {identifier}",
                ) from exc
            relative_members.add(relative.as_posix())
        members: list[dict[str, Any]] = []
        for index, relative_posix in enumerate(sorted(relative_members)):
            member_path = candidate_dir / Path(relative_posix)
            metadata = member_path.lstat()
            if (
                member_path.is_symlink()
                or not member_path.is_file()
                or metadata.st_nlink != 1
            ):
                raise BrokerError(
                    409,
                    "materialized closure member is not a private regular "
                    f"file: {member_path}",
                )
            members.append(
                {
                    "index": index,
                    "relative_path": relative_posix,
                    "sha256": sha256_file(member_path),
                }
            )
        return members

    def materialized_candidate_closure_file(
        self, sweep_id: str, trial_index_raw: str, member_index_raw: str
    ) -> tuple[Path, str]:
        """Return one digest-verified closure member for agent-side export."""

        try:
            member_index = int(member_index_raw)
        except (TypeError, ValueError) as exc:
            raise BrokerError(
                400, f"invalid closure member index: {member_index_raw!r}"
            ) from exc
        members = self.materialized_candidate_closure(sweep_id, trial_index_raw)
        member = next(
            (entry for entry in members if entry["index"] == member_index), None
        )
        if member is None:
            raise BrokerError(404, f"closure member {member_index} does not exist")
        materialized = self.materialize(sweep_id, int(trial_index_raw))
        root = Path(str(materialized["usd_path"])).resolve(strict=True)
        path = root.parent / Path(str(member["relative_path"]))
        digest = sha256_file(path)
        if digest != str(member["sha256"]):
            raise BrokerError(
                409, f"materialized closure member digest changed on disk: {path}"
            )
        return path, digest

    def candidate_recording_file(
        self, sweep_id: str, trial_index_raw: str
    ) -> tuple[Path, str]:
        """Return the exact scenario rollout recorded for one top-K trial."""

        try:
            trial_index = int(trial_index_raw)
        except (TypeError, ValueError) as exc:
            raise BrokerError(400, f"invalid trial_index: {trial_index_raw!r}") from exc
        with self._lock:
            record = self._records.get(sweep_id)
            if record is None:
                raise BrokerError(404, f"unknown sweep_id {sweep_id!r}")
            if record.status != "succeeded":
                raise BrokerError(
                    409,
                    f"sweep {sweep_id} is {record.status}; only succeeded "
                    "sweep recordings can be exported",
                )
            candidate = next(
                (
                    copy.deepcopy(entry)
                    for entry in record.candidates
                    if entry["trial_index"] == trial_index
                ),
                None,
            )
            work_dir = record.work_dir.resolve(strict=True)
        if candidate is None:
            raise BrokerError(
                404,
                f"trial {trial_index} is not a top-{self.top_k} candidate "
                f"of sweep {sweep_id}",
            )
        recording_raw = candidate.get("recording")
        if not isinstance(recording_raw, str) or not recording_raw:
            raise BrokerError(
                404,
                f"trial {trial_index} of sweep {sweep_id} has no recording",
            )
        recording = Path(recording_raw)
        if not recording.is_absolute():
            recording = work_dir / recording
        recording = _absolute_without_symlink_resolution(recording)
        try:
            relative = recording.relative_to(work_dir)
        except ValueError as exc:
            raise BrokerError(
                409,
                f"trial recording escapes the broker workspace: {recording}",
            ) from exc
        current = work_dir
        for component in relative.parts:
            current /= component
            if current.is_symlink():
                raise BrokerError(
                    409,
                    f"trial recording contains a symbolic link: {current}",
                )
        try:
            metadata = recording.lstat()
        except FileNotFoundError as exc:
            raise BrokerError(404, f"trial recording is missing: {recording}") from exc
        if not recording.is_file() or metadata.st_nlink != 1:
            raise BrokerError(
                409, f"trial recording is not a private regular file: {recording}"
            )
        resolved = recording.resolve(strict=True)
        if not resolved.is_relative_to(work_dir):
            raise BrokerError(
                409, f"trial recording escapes the broker workspace: {recording}"
            )
        return resolved, sha256_file(resolved)

    def selected_candidate_replay_context(
        self,
        sweep_id: str,
        trial_index: int,
        *,
        expected_usd_sha256: str,
    ) -> dict[str, Any]:
        """Stage digest-bound, run-owned inputs for replaying an accepted trial."""

        materialized = self.materialize(sweep_id, trial_index)
        if str(materialized["usd_sha256"]) != expected_usd_sha256:
            raise BrokerError(
                409,
                "selected candidate digest does not match the materialized trial",
            )
        with self._lock:
            record = self._records.get(sweep_id)
            if record is None:
                raise BrokerError(404, f"unknown sweep_id {sweep_id!r}")
            candidate = next(
                (
                    copy.deepcopy(entry)
                    for entry in record.candidates
                    if entry["trial_index"] == trial_index
                ),
                None,
            )
            work_dir = record.work_dir
            scenario_sha256 = record.scenario_sha256
            seed = record.seed
        if candidate is None:
            raise BrokerError(
                404,
                f"trial {trial_index} is not a selected top candidate of {sweep_id}",
            )
        replay_root = self.run_dir / "tuning" / "final_visual" / "broker_replay_inputs"
        _reject_child_output_links(self.run_dir, replay_root)
        replay_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        _reject_child_output_links(self.run_dir, replay_root)
        staged_dir = Path(tempfile.mkdtemp(prefix=f"{sweep_id}-", dir=replay_root))
        os.chmod(staged_dir, 0o700)

        def staged_artifact(
            relative: Path, *, expected: str | None = None
        ) -> tuple[Path, str]:
            source = work_dir / relative
            destination = staged_dir / relative
            try:
                source_metadata = source.lstat()
            except FileNotFoundError as exc:
                raise BrokerError(
                    409,
                    f"selected replay artifact is missing: {relative}",
                ) from exc
            if (
                source.is_symlink()
                or not source.is_file()
                or source_metadata.st_nlink != 1
                or not source.resolve(strict=True).is_relative_to(
                    work_dir.resolve(strict=True)
                )
            ):
                raise BrokerError(
                    409,
                    f"selected replay artifact is not a regular file: {relative}",
                )
            source_digest_before = sha256_file(source)
            try:
                with (
                    source.open("rb") as source_stream,
                    destination.open("xb") as destination_stream,
                ):
                    shutil.copyfileobj(source_stream, destination_stream)
                    destination_stream.flush()
                    os.fsync(destination_stream.fileno())
            except FileExistsError as exc:
                raise BrokerError(
                    409,
                    f"selected replay destination already exists: {relative}",
                ) from exc
            os.chmod(destination, 0o600)
            source_metadata_after = source.lstat()
            destination_metadata = destination.lstat()
            source_digest_after = sha256_file(source)
            destination_digest = sha256_file(destination)
            source_identity = (
                source_metadata.st_dev,
                source_metadata.st_ino,
                source_metadata.st_size,
                source_metadata.st_mtime_ns,
                source_metadata.st_ctime_ns,
            )
            if (
                not stat.S_ISREG(source_metadata_after.st_mode)
                or source_metadata_after.st_nlink != 1
                or (
                    source_metadata_after.st_dev,
                    source_metadata_after.st_ino,
                    source_metadata_after.st_size,
                    source_metadata_after.st_mtime_ns,
                    source_metadata_after.st_ctime_ns,
                )
                != source_identity
                or not stat.S_ISREG(destination_metadata.st_mode)
                or destination_metadata.st_nlink != 1
                or source_digest_before != source_digest_after
                or source_digest_after != destination_digest
                or (expected is not None and destination_digest != expected)
            ):
                raise BrokerError(
                    409,
                    f"selected replay artifact digest mismatch: {relative}",
                )
            return destination.resolve(strict=True), destination_digest

        try:
            scenario_path, observed_scenario_sha256 = staged_artifact(
                Path("scenario_input.yaml"),
                expected=scenario_sha256,
            )
            sanitized_scenario_path, sanitized_scenario_sha256 = staged_artifact(
                Path("scenario_sanitized.yaml")
            )
        except Exception:
            shutil.rmtree(staged_dir, ignore_errors=True)
            raise
        return {
            "sweep_id": sweep_id,
            "trial_index": trial_index,
            "trial_seed": seed + trial_index,
            "params": dict(candidate.get("params") or {}),
            "score": candidate.get("score"),
            "scenario_path": str(scenario_path),
            "scenario_sha256": observed_scenario_sha256,
            "sanitized_scenario_path": str(sanitized_scenario_path),
            "sanitized_scenario_sha256": sanitized_scenario_sha256,
            "materialized_usd_path": str(materialized["usd_path"]),
            "materialized_usd_sha256": str(materialized["usd_sha256"]),
        }

    # ---------------------------------------------------------------- ledger

    def _append_ledger(self, record: SweepRecord) -> None:
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

    def verify_claim(self, claim: dict[str, Any]) -> tuple[bool, str]:
        """Check an agent-claimed sweep reference against broker state.

        Returns ``(ok, reason)``. The broker's in-memory records are the
        source of truth — a claim of an unknown sweep or with mismatched
        digests is rejected. A FAILED sweep is still a valid reference: a
        revise/stop decision legitimately cites the failed sweep it learned
        from. Promotion safety does not rely on this check — ``materialize``
        refuses non-succeeded sweeps, so an accepted candidate can only come
        from a successful one.
        """

        sweep_id = str(claim.get("sweep_id") or "")
        with self._lock:
            record = self._records.get(sweep_id)
        if record is None:
            return False, f"sweep_id {sweep_id!r} has no broker record"
        expected = {
            "scenario_sha256": record.scenario_sha256,
            "evidence_sha256": record.evidence_sha256,
        }
        for key, value in expected.items():
            claimed = claim.get(key)
            if value and not claimed:
                return False, (
                    f"{key} missing from the decision for sweep {sweep_id}; "
                    "digest bindings are mandatory"
                )
            if claimed and not value:
                return False, (
                    f"{key} claimed for sweep {sweep_id} but the broker recorded none"
                )
            if claimed and value and claimed != value:
                return False, (
                    f"{key} mismatch for sweep {sweep_id}: "
                    f"claimed {claimed}, broker recorded {value}"
                )
        # Rehash the referenced artifacts on disk: they live under the run
        # directory, so a post-sweep edit would otherwise leave the recorded
        # digests intact while the bytes the agent's decision cites changed.
        rehash_targets = [
            ("scenario", record.scenario_path, record.scenario_sha256),
            ("evidence", record.evidence_path, record.evidence_sha256),
        ]
        for label, path, recorded in rehash_targets:
            if path is None or not recorded:
                continue
            if not Path(path).is_file():
                return False, (f"{label} file for sweep {sweep_id} is missing: {path}")
            current = sha256_file(path)
            if current != recorded:
                return False, (
                    f"{label} file for sweep {sweep_id} was modified after "
                    f"the sweep: recorded {recorded}, on disk {current}"
                )
        return True, "ok"
