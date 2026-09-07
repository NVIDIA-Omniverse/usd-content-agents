# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Formal SimReady Foundation profile validation adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import stat
import subprocess
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    open_held_confined_artifact,
)
from world_understanding.utils.usd.package import (
    DEFAULT_MAX_USDZ_EXTRACTED_BYTES,
    DEFAULT_MAX_USDZ_MEMBERS,
    extract_usdz_members_to_dir,
    safe_usdz_member_parts,
)
from world_understanding.utils.windows_process import WindowsKillOnCloseJob

from content_agent_workflows.common.run_record import WorkflowRunRecorder

from .asset_identity import (
    AssetDependencyIdentityError,
    asset_dependency_identity_errors,
    asset_dependency_root_sha256,
    build_asset_dependency_manifest,
)
from .foundation_runtime import (
    build_simready_subprocess_environment,
    build_validation_command,
    resolve_simready_runtime,
    verify_simready_runtime_identity,
)
from .models import (
    DEFAULT_SIMREADY_PROFILE,
    DEFAULT_SIMREADY_PROFILE_VERSION,
    SimReadyValidationInput,
    SimReadyValidationReport,
    default_simready_report_name,
)

NONBLOCKING_SINGLE_COMPONENT_REQUIREMENT = "RB.MB.001"
ERROR_SEVERITIES = {"ERROR", "FAIL", "FAILED", "FAILURE"}
TOOLCHAIN_ERROR_KEY = "_content_workflow_toolchain_error"
SIMREADY_VALIDATOR_SUFFIXES = frozenset({".usd", ".usda"})
USDZ_VALIDATION_WORKSPACE_SUFFIX = ".simready-usdz-validation"
SIMREADY_SUPPORT_BUDGET_ERROR_PREFIX = "simready_support_budget_exceeded"


class _UsdzValidationStagingError(RuntimeError):
    """Raised when a USDZ cannot be staged safely for Foundation validation."""


@dataclass(frozen=True)
class _ValidationTarget:
    """Actual validator input and optional USDZ staging evidence."""

    asset_path: Path
    workspace_path: Path | None = None
    package_root: str | None = None

    @property
    def staged_usdz(self) -> bool:
        return self.workspace_path is not None


def _validation_run_record_subdir(report_path: Path) -> Path:
    """Return a deterministic recorder namespace for one validation report."""

    return Path(f".{report_path.name}.workflow-run")


def run_simready_profile_validation(
    params: SimReadyValidationInput,
) -> SimReadyValidationReport:
    """Run profile validation and durably record unexpected failures."""

    recorders: list[WorkflowRunRecorder] = []
    try:
        return _run_simready_profile_validation(params, recorders)
    except Exception as exc:
        if recorders:
            recorders[-1].finalize_unexpected_failure(exc)
        raise


def _run_simready_profile_validation(
    params: SimReadyValidationInput,
    recorders: list[WorkflowRunRecorder],
) -> SimReadyValidationReport:
    """Run Foundation profile validation and write a normalized report."""

    asset_path = Path(params.asset_path).expanduser().resolve()
    report_path = (
        Path(params.report_path).expanduser().resolve()
        if params.report_path is not None
        else asset_path.parent
        / default_simready_report_name(asset_path, kind="profile")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    raw_report_path = report_path.with_suffix(".raw.json")
    stdout_log_path = (
        Path(params.stdout_log_path).expanduser().resolve()
        if params.stdout_log_path is not None
        else report_path.with_suffix(".stdout.log")
    )
    stderr_log_path = (
        Path(params.stderr_log_path).expanduser().resolve()
        if params.stderr_log_path is not None
        else report_path.with_suffix(".stderr.log")
    )
    request = params.model_dump(mode="json")
    request.pop("resume", None)
    recorder = WorkflowRunRecorder.start(
        report_path.parent,
        workflow="simready_profile_validation",
        request=request,
        source_path=asset_path if asset_path.is_file() else None,
        backend={
            "kind": "simready_foundation",
            "validator": "simready-validate",
        },
        policy={
            "profile": params.profile,
            "profile_version": params.profile_version,
            "install_missing": params.install_missing,
            "update_foundation": params.update_foundation,
        },
        required_artifacts=["validation_report"],
        resume=params.resume,
        record_subdir=_validation_run_record_subdir(report_path),
    )
    recorders.append(recorder)

    def finish(report: SimReadyValidationReport) -> SimReadyValidationReport:
        report.workflow_run_manifest_path = str(recorder.path)
        written = _write_report(
            report_path,
            report,
            max_bytes=params.max_support_file_bytes,
        )
        recorded = [
            recorder.record_artifact(
                "validation_report",
                report_path,
                kind="simready_validation",
            ).logical_name
        ]
        for logical_name, path, kind in (
            ("raw_validation_report", raw_report_path, "validator_report"),
            ("validator_stdout", stdout_log_path, "log"),
            ("validator_stderr", stderr_log_path, "log"),
        ):
            if not path.is_file():
                continue
            try:
                artifact = recorder.record_artifact(
                    logical_name,
                    path,
                    kind=kind,
                    required=False,
                )
            except ValueError:
                continue
            recorded.append(artifact.logical_name)
        recorder.checkpoint("validated", recorded)
        status = (
            "pass"
            if report.passed
            else "blocked"
            if report.status == "BLOCKED"
            else "fail"
        )
        recorder.finalize(
            status,
            failure={"errors": report.errors, "next_step": report.next_step}
            if not report.passed
            else None,
        )
        return written

    if not asset_path.exists():
        report = _blocked_report(
            asset_path=asset_path,
            params=params,
            report_path=raw_report_path,
            errors=[f"Asset path does not exist: {asset_path}"],
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
        )
        return finish(report)

    try:
        asset_dependency_manifest = build_asset_dependency_manifest(asset_path)
        asset_sha256 = asset_dependency_root_sha256(asset_dependency_manifest)
    except AssetDependencyIdentityError as exc:
        report = _blocked_report(
            asset_path=asset_path,
            params=params,
            report_path=raw_report_path,
            errors=[f"Could not bind asset dependency identity: {exc}"],
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            next_step="resolve-usd-dependencies",
        )
        return finish(report)

    runtime = resolve_simready_runtime(
        foundation_root=params.foundation_root,
        foundation_spec_root=params.foundation_spec_root,
        venv_path=params.venv_path,
        install_missing=params.install_missing,
        update_foundation=params.update_foundation,
    )
    if not runtime.passed:
        report = _blocked_report(
            asset_path=asset_path,
            params=params,
            report_path=raw_report_path,
            errors=runtime.errors,
            warnings=runtime.warnings,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            runtime=runtime.model_dump(mode="json"),
        )
        return finish(report)

    _initialize_validation_logs(stdout_log_path, stderr_log_path)
    raw_invocation_path, raw_preparation_error = _prepare_raw_report_invocation(
        raw_report_path
    )
    if raw_preparation_error is not None or raw_invocation_path is None:
        report = _blocked_report(
            asset_path=asset_path,
            params=params,
            report_path=raw_report_path,
            errors=[
                raw_preparation_error
                or "Could not prepare a fresh validation raw-report workspace."
            ],
            warnings=runtime.warnings,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            runtime=runtime.model_dump(mode="json"),
            next_step="fix-simready-validator-runtime",
        )
        return finish(report)

    try:
        validation_target = _prepare_validation_target(
            asset_path,
            report_path,
            max_file_bytes=params.max_support_file_bytes,
            max_total_bytes=params.max_support_total_bytes,
            max_file_count=params.max_support_file_count,
            max_directory_count=params.max_support_directory_count,
        )
    except _UsdzValidationStagingError as exc:
        _cleanup_raw_report_invocation(raw_invocation_path)
        report = _blocked_report(
            asset_path=asset_path,
            params=params,
            report_path=raw_report_path,
            errors=[str(exc)],
            warnings=runtime.warnings,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            runtime=runtime.model_dump(mode="json"),
            next_step="fix-simready-usdz-package",
            validation_policy={
                "blocking": False,
                "blocked_runtime_is_not_profile_failure": False,
                "usdz_validation_staging": {
                    "original_asset_path": str(asset_path),
                    "workspace_path": str(_validation_workspace_path(report_path)),
                    "status": "rejected",
                },
            },
        )
        return finish(report)

    runtime_identity_errors = verify_simready_runtime_identity(runtime)
    runtime_identity_errors.extend(
        asset_dependency_identity_errors(asset_path, asset_dependency_manifest)
    )
    if runtime_identity_errors:
        _cleanup_raw_report_invocation(raw_invocation_path)
        _cleanup_validation_target(validation_target)
        report = _blocked_report(
            asset_path=asset_path,
            params=params,
            report_path=raw_report_path,
            errors=runtime_identity_errors,
            warnings=runtime.warnings,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            runtime=runtime.model_dump(mode="json"),
            next_step="prepare-simready-foundation-runtime",
        )
        return finish(report)

    cleanup_error: str | None = None
    try:
        invocation_command = build_validation_command(
            runtime=runtime,
            asset_path=validation_target.asset_path,
            profile=params.profile,
            profile_version=params.profile_version,
            raw_report_path=raw_invocation_path,
        )
        command = _command_for_report(
            invocation_command,
            validator_asset_path=validation_target.asset_path,
            original_asset_path=asset_path,
            invocation_raw_report_path=raw_invocation_path,
            reported_raw_report_path=raw_report_path,
        )
        validator_environment = _validator_subprocess_environment(invocation_command[0])
        if (
            params.max_stdout_log_bytes is not None
            and params.max_stderr_log_bytes is not None
        ):
            returncode, subprocess_error = _run_bounded_validator_subprocess(
                invocation_command,
                cwd=raw_invocation_path.parent,
                env=validator_environment,
                stdout_log_path=stdout_log_path,
                stderr_log_path=stderr_log_path,
                timeout_s=params.timeout_s,
                stdout_limit_bytes=params.max_stdout_log_bytes,
                stderr_limit_bytes=params.max_stderr_log_bytes,
            )
        else:
            try:
                with stdout_log_path.open("w", encoding="utf-8") as stdout_file:
                    with stderr_log_path.open("w", encoding="utf-8") as stderr_file:
                        completed = subprocess.run(
                            invocation_command,
                            check=False,
                            cwd=raw_invocation_path.parent,
                            env=validator_environment,
                            stdout=stdout_file,
                            stderr=stderr_file,
                            text=True,
                            timeout=params.timeout_s,
                        )
                returncode = completed.returncode
                subprocess_error = None
            except (OSError, subprocess.TimeoutExpired) as exc:
                returncode = 1
                subprocess_error = str(exc)
        sigxfsz = getattr(signal, "SIGXFSZ", None)
        if (
            sigxfsz is not None
            and returncode == -sigxfsz
            and params.max_support_file_bytes is not None
        ):
            _cleanup_raw_report_invocation(raw_invocation_path)
            raise RuntimeError(
                f"{SIMREADY_SUPPORT_BUDGET_ERROR_PREFIX}: file_bytes "
                f"limit={params.max_support_file_bytes}"
            )
    finally:
        cleanup_error = _cleanup_validation_target(validation_target)

    runtime_identity_errors = verify_simready_runtime_identity(runtime)
    runtime_identity_errors.extend(
        asset_dependency_identity_errors(asset_path, asset_dependency_manifest)
    )
    raw_payload = _publish_fresh_raw_report(
        invocation_path=raw_invocation_path,
        raw_report_path=raw_report_path,
        max_bytes=params.max_support_file_bytes,
    )
    staging_evidence = _validation_staging_evidence(
        validation_target,
        original_asset_path=asset_path,
        cleanup_error=cleanup_error,
    )
    report = _normalize_validation_report(
        asset_path=asset_path,
        asset_sha256=asset_sha256,
        asset_dependency_manifest=asset_dependency_manifest,
        validator_asset_path=validation_target.asset_path,
        params=params,
        runtime=runtime.model_dump(mode="json"),
        command=command,
        returncode=returncode,
        raw_payload=raw_payload,
        raw_report_path=raw_report_path,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        subprocess_error=subprocess_error,
        adapter_warnings=[_usdz_staging_warning(staging_evidence)]
        if staging_evidence is not None
        else [],
        adapter_errors=[
            *runtime_identity_errors,
            *([cleanup_error] if cleanup_error is not None else []),
        ],
        staging_evidence=staging_evidence,
    )
    return finish(report)


def _run_bounded_validator_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_log_path: Path,
    stderr_log_path: Path,
    timeout_s: float,
    stdout_limit_bytes: int | None,
    stderr_limit_bytes: int | None,
) -> tuple[int, str | None]:
    """Run one validator while retaining at most each configured log prefix."""

    if os.name == "nt":
        return _run_bounded_validator_subprocess_windows(
            command,
            cwd=cwd,
            env=env,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            timeout_s=timeout_s,
            stdout_limit_bytes=stdout_limit_bytes,
            stderr_limit_bytes=stderr_limit_bytes,
        )

    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    output_limit_exceeded = False
    timed_out = False
    with stdout_log_path.open("wb") as stdout_file:
        with stderr_log_path.open("wb") as stderr_file:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=os.name != "nt",
                )
            except OSError as exc:
                return 1, str(exc)

            assert process.stdout is not None
            assert process.stderr is not None
            stream_state = (
                (process.stdout, stdout_file, stdout_limit_bytes),
                (process.stderr, stderr_file, stderr_limit_bytes),
            )
            written: dict[int, int] = {}
            with selectors.DefaultSelector() as selector:
                for stream, destination, limit in stream_state:
                    os.set_blocking(stream.fileno(), False)
                    selector.register(
                        stream,
                        selectors.EVENT_READ,
                        data=(destination, limit),
                    )
                    written[stream.fileno()] = 0

                while True:
                    remaining_timeout = timeout_s - (time.monotonic() - started)
                    if remaining_timeout <= 0:
                        timed_out = True
                        break

                    if _validator_leader_has_exited(process):
                        # A successful leader may leave same-group children behind.
                        _kill_validator_process_group(process)
                        output_limit_exceeded |= _drain_ready_validator_output(
                            selector,
                            written=written,
                        )
                        break

                    if not selector.get_map():
                        # The validator may deliberately close both output pipes and
                        # continue producing its report. Keep supervising the leader
                        # without spinning on an empty selector.
                        time.sleep(min(remaining_timeout, 0.1))
                        continue

                    ready = selector.select(min(remaining_timeout, 0.1))
                    for key, _events in ready:
                        try:
                            chunk = os.read(key.fd, 64 * 1024)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        destination, limit = key.data
                        retained = chunk
                        if limit is not None:
                            available = max(0, limit - written[key.fd])
                            retained = chunk[:available]
                            if len(chunk) > available:
                                output_limit_exceeded = True
                        if retained:
                            destination.write(retained)
                            written[key.fd] += len(retained)
                    if output_limit_exceeded:
                        break

            # This is mandatory on every path, including a zero leader exit.
            _kill_validator_process_group(process)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)

    if output_limit_exceeded:
        return (
            1,
            "simready_validator_output_limit_exceeded: validator output "
            "exceeded a configured per-stream limit",
        )
    if timed_out:
        return 1, f"simready_validator_timeout_exceeded: exceeded {timeout_s} seconds"
    return process.returncode, None


def _run_bounded_validator_subprocess_windows(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_log_path: Path,
    stderr_log_path: Path,
    timeout_s: float,
    stdout_limit_bytes: int | None,
    stderr_limit_bytes: int | None,
) -> tuple[int, str | None]:
    """Pump pipes while a Job Object contains the Windows validator tree."""

    for path in (stdout_log_path, stderr_log_path):
        path.write_bytes(b"")
    try:
        job = WindowsKillOnCloseJob()
    except (OSError, RuntimeError) as exc:
        return 1, str(exc)
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=job.creation_flags,
        )
    except OSError as exc:
        try:
            job.close()
        except OSError as cleanup_exc:
            return (
                1,
                f"simready_validator_process_cleanup_failed: {cleanup_exc}",
            )
        return 1, str(exc)

    try:
        job.assign_process(process)
        job.resume_process(process)
    except (OSError, RuntimeError) as exc:
        cleanup_errors: list[str] = []
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired) as cleanup_exc:
            cleanup_errors.append(str(cleanup_exc))
        try:
            job.terminate(exit_code=1)
            job.wait_for_empty(timeout_s=5.0)
        except (OSError, RuntimeError, TimeoutError) as cleanup_exc:
            cleanup_errors.append(str(cleanup_exc))
        try:
            job.close()
        except OSError as cleanup_exc:
            cleanup_errors.append(str(cleanup_exc))
        if cleanup_errors:
            return (
                1,
                "simready_validator_process_cleanup_failed: "
                + "; ".join(cleanup_errors),
            )
        return 1, str(exc)

    assert process.stdout is not None
    assert process.stderr is not None
    output_limit_exceeded = threading.Event()
    pump_changed = threading.Event()
    pump_errors: list[str] = []
    pump_error_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_drain_bounded_validator_stream_windows,
            args=(
                stream,
                path,
                limit,
                output_limit_exceeded,
                pump_changed,
                pump_errors,
                pump_error_lock,
            ),
            daemon=True,
        )
        for stream, path, limit in (
            (process.stdout, stdout_log_path, stdout_limit_bytes),
            (process.stderr, stderr_log_path, stderr_limit_bytes),
        )
    ]
    for thread in threads:
        thread.start()

    started = time.monotonic()
    timed_out = False
    leader_returncode: int | None = None
    while True:
        current_returncode = process.poll()
        if current_returncode is not None:
            leader_returncode = int(current_returncode)
            break
        if output_limit_exceeded.is_set() or pump_errors:
            break
        remaining_timeout = timeout_s - (time.monotonic() - started)
        if remaining_timeout <= 0:
            timed_out = True
            break
        pump_changed.wait(min(remaining_timeout, 0.05))
        pump_changed.clear()

    cleanup_errors = []
    job_closed = False
    try:
        job.terminate(exit_code=1)
        job.wait_for_empty(timeout_s=5.0)
    except (OSError, RuntimeError, TimeoutError) as exc:
        cleanup_errors.append(str(exc))
        try:
            job.close()
            job_closed = True
        except OSError as close_exc:
            cleanup_errors.append(str(close_exc))
    try:
        process.wait(timeout=5.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        cleanup_errors.append(str(exc))
        try:
            process.kill()
            process.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired) as kill_exc:
            cleanup_errors.append(str(kill_exc))
    for thread in threads:
        thread.join(timeout=5.0)
    if any(thread.is_alive() for thread in threads):
        cleanup_errors.append("validator output pump remained live after tree cleanup")
    if not job_closed:
        try:
            job.close()
        except OSError as exc:
            cleanup_errors.append(str(exc))

    if cleanup_errors:
        return (
            1,
            "simready_validator_process_cleanup_failed: " + "; ".join(cleanup_errors),
        )

    if output_limit_exceeded.is_set():
        return (
            1,
            "simready_validator_output_limit_exceeded: validator output "
            "exceeded a configured per-stream limit",
        )
    if timed_out:
        return 1, f"simready_validator_timeout_exceeded: exceeded {timeout_s} seconds"
    if pump_errors:
        return 1, f"simready_validator_output_capture_failed: {pump_errors[0]}"
    return (
        leader_returncode
        if leader_returncode is not None
        else int(process.returncode or 0),
        None,
    )


def _drain_bounded_validator_stream_windows(
    stream: Any,
    path: Path,
    limit: int | None,
    output_limit_exceeded: threading.Event,
    pump_changed: threading.Event,
    pump_errors: list[str],
    pump_error_lock: threading.Lock,
) -> None:
    retained = 0
    try:
        with path.open("wb") as destination:
            while chunk := stream.read1(64 * 1024):
                available = len(chunk) if limit is None else max(0, limit - retained)
                if available:
                    selected = chunk[:available]
                    destination.write(selected)
                    retained += len(selected)
                if limit is not None and len(chunk) > available:
                    output_limit_exceeded.set()
                    pump_changed.set()
                    return
    except OSError as exc:
        with pump_error_lock:
            pump_errors.append(str(exc))
        pump_changed.set()


def _drain_ready_validator_output(
    selector: selectors.BaseSelector,
    *,
    written: dict[int, int],
) -> bool:
    """Drain bytes already available after group cleanup without waiting on orphans."""

    deadline = time.monotonic() + 0.25
    exceeded = False
    while selector.get_map() and time.monotonic() < deadline:
        ready = selector.select(0)
        if not ready:
            time.sleep(0.005)
            continue
        for key, _events in ready:
            try:
                chunk = os.read(key.fd, 64 * 1024)
            except BlockingIOError:
                continue
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            destination, limit = key.data
            retained = chunk
            if limit is not None:
                available = max(0, limit - written[key.fd])
                retained = chunk[:available]
                if len(chunk) > available:
                    exceeded = True
            if retained:
                destination.write(retained)
                written[key.fd] += len(retained)
    return exceeded


def _kill_validator_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        if process.poll() is None:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _validator_leader_has_exited(process: subprocess.Popen[bytes]) -> bool:
    """Observe a Linux validator exit without reaping and reusing its PGID."""

    if os.name == "nt":
        return process.poll() is not None
    result = os.waitid(
        os.P_PID,
        process.pid,
        os.WEXITED | os.WNOHANG | os.WNOWAIT,
    )
    return result is not None and result.si_pid == process.pid


def _validator_subprocess_environment(
    validator_executable: str,
) -> dict[str, str]:
    """Keep the dedicated validator venv isolated from caller runtimes."""

    return build_simready_subprocess_environment(
        executable_dir=Path(validator_executable).resolve().parent,
    )


def _initialize_validation_logs(stdout_log_path: Path, stderr_log_path: Path) -> None:
    for path in (stdout_log_path, stderr_log_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def _prepare_raw_report_invocation(
    raw_report_path: Path,
) -> tuple[Path | None, str | None]:
    """Remove stale evidence and allocate a private path for this invocation."""

    try:
        raw_report_path.unlink(missing_ok=True)
    except OSError as exc:
        return None, f"Could not remove prior validation raw report: {exc}"

    for _ in range(16):
        workspace = raw_report_path.parent / (
            f".{raw_report_path.name}.validation-{uuid4().hex}"
        )
        try:
            workspace.mkdir(mode=0o700)
        except FileExistsError:
            continue
        except OSError as exc:
            return None, f"Could not create validation raw-report workspace: {exc}"
        return workspace / raw_report_path.name, None
    return None, "Could not allocate a unique validation raw-report workspace."


def _publish_fresh_raw_report(
    *,
    invocation_path: Path,
    raw_report_path: Path,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish output written for this invocation only."""

    payload = _load_json_if_present(invocation_path, max_bytes=max_bytes)
    if not payload.get(TOOLCHAIN_ERROR_KEY):
        try:
            invocation_path.replace(raw_report_path)
        except OSError as exc:
            payload = {
                TOOLCHAIN_ERROR_KEY: True,
                "errors": [f"Validation raw report could not be published: {exc}"],
            }

    cleanup_error = _cleanup_raw_report_invocation(invocation_path)
    if cleanup_error is not None:
        errors = _string_list(payload.get("errors"))
        errors.append(cleanup_error)
        payload["errors"] = errors
        payload[TOOLCHAIN_ERROR_KEY] = True
    return payload


def _cleanup_raw_report_invocation(invocation_path: Path) -> str | None:
    """Remove the invocation output and its private workspace when still present."""

    try:
        invocation_path.unlink(missing_ok=True)
        invocation_path.parent.rmdir()
    except OSError as exc:
        return f"Could not clean validation raw-report workspace: {exc}"
    return None


def _prepare_validation_target(
    asset_path: Path,
    report_path: Path,
    *,
    max_file_bytes: int | None = None,
    max_total_bytes: int | None = None,
    max_file_count: int | None = None,
    max_directory_count: int | None = None,
) -> _ValidationTarget:
    if asset_path.suffix.lower() != ".usdz":
        return _ValidationTarget(asset_path=asset_path)

    source_identity = _archive_source_identity(asset_path)
    package_root = _usdz_package_root(
        asset_path,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        max_file_count=max_file_count,
        max_directory_count=max_directory_count,
    )
    workspace_path = _validation_workspace_path(report_path)
    try:
        workspace_path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise _UsdzValidationStagingError(
            "Deterministic USDZ validation workspace already exists; refusing "
            f"to reuse or overwrite it: {workspace_path}"
        ) from exc
    except OSError as exc:
        raise _UsdzValidationStagingError(
            f"Could not create USDZ validation workspace {workspace_path}: {exc}"
        ) from exc

    try:
        extract_usdz_members_to_dir(
            asset_path,
            workspace_path,
            allowed_suffixes=None,
            max_members=(
                max_file_count + max_directory_count
                if max_file_count is not None and max_directory_count is not None
                else DEFAULT_MAX_USDZ_MEMBERS
            ),
            max_total_bytes=(
                max_total_bytes
                if max_total_bytes is not None
                else DEFAULT_MAX_USDZ_EXTRACTED_BYTES
            ),
            fail_on_filtered_member=True,
        )
        staged_root = workspace_path / package_root
        if not staged_root.is_file():
            raise _UsdzValidationStagingError(
                "USDZ package root was not extracted as a regular file: "
                f"{package_root.as_posix()}"
            )
        if _archive_source_identity(asset_path) != source_identity:
            raise _UsdzValidationStagingError(
                "USDZ source identity changed while preparing Foundation validation: "
                f"{asset_path}"
            )
    except BaseException as exc:
        cleanup_error = _remove_validation_workspace(workspace_path)
        if isinstance(exc, _UsdzValidationStagingError) and cleanup_error is None:
            raise
        if isinstance(exc, Exception):
            detail = str(exc)
            if cleanup_error is not None:
                detail = f"{detail}; {cleanup_error}"
            raise _UsdzValidationStagingError(
                f"Could not stage USDZ package {asset_path}: {detail}"
            ) from exc
        if cleanup_error is not None:
            exc.add_note(cleanup_error)
        raise

    return _ValidationTarget(
        asset_path=staged_root,
        workspace_path=workspace_path,
        package_root=package_root.as_posix(),
    )


def _archive_source_identity(path: Path) -> tuple[int, int, int, int]:
    try:
        metadata = path.stat()
    except OSError as exc:
        raise _UsdzValidationStagingError(
            f"Could not inspect USDZ source identity {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise _UsdzValidationStagingError(
            f"USDZ validation source is not a regular file: {path}"
        )
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _usdz_package_root(
    asset_path: Path,
    *,
    max_file_bytes: int | None = None,
    max_total_bytes: int | None = None,
    max_file_count: int | None = None,
    max_directory_count: int | None = None,
) -> Path:
    limits = (
        max_file_bytes,
        max_total_bytes,
        max_file_count,
        max_directory_count,
    )
    if any(limit is not None for limit in limits) and not all(
        limit is not None for limit in limits
    ):
        raise ValueError("all USDZ staging support limits must be provided together")
    try:
        with zipfile.ZipFile(asset_path) as archive:
            infos = archive.infolist()
            if (
                max_file_count is not None
                and max_directory_count is not None
                and len(infos) > max_file_count + max_directory_count
            ):
                raise _UsdzValidationStagingError(
                    f"{SIMREADY_SUPPORT_BUDGET_ERROR_PREFIX}: archive_member_count "
                    f"limit={max_file_count + max_directory_count}"
                )
            normalized_members: set[tuple[str, ...]] = set()
            staging_directories: set[tuple[str, ...]] = set()
            root_parts: tuple[str, ...] | None = None
            file_count = 0
            total_bytes = 0
            for info in infos:
                parts = _validated_usdz_member_parts(info)
                if parts in normalized_members:
                    if not info.is_dir() and parts == root_parts:
                        raise _UsdzValidationStagingError(
                            "USDZ package root is ambiguous because its normalized "
                            "member path appears more than once: "
                            f"{'/'.join(parts)}"
                        )
                    raise _UsdzValidationStagingError(
                        "USDZ package contains colliding normalized member paths: "
                        f"{'/'.join(parts)}"
                    )
                normalized_members.add(parts)
                directory_depth = len(parts) if info.is_dir() else len(parts) - 1
                for depth in range(1, directory_depth + 1):
                    staging_directories.add(parts[:depth])
                    if (
                        max_directory_count is not None
                        and len(staging_directories) + 1 > max_directory_count
                    ):
                        raise _UsdzValidationStagingError(
                            f"{SIMREADY_SUPPORT_BUDGET_ERROR_PREFIX}: "
                            "directory_count "
                            f"limit={max_directory_count}"
                        )
                if info.is_dir():
                    continue
                if root_parts is None:
                    root_parts = parts
                file_count += 1
                if max_file_count is not None and file_count > max_file_count:
                    raise _UsdzValidationStagingError(
                        f"{SIMREADY_SUPPORT_BUDGET_ERROR_PREFIX}: file_count "
                        f"limit={max_file_count}"
                    )
                if max_file_bytes is not None and info.file_size > max_file_bytes:
                    raise _UsdzValidationStagingError(
                        f"{SIMREADY_SUPPORT_BUDGET_ERROR_PREFIX}: file_bytes "
                        f"limit={max_file_bytes}"
                    )
                total_bytes += info.file_size
                if max_total_bytes is not None and total_bytes > max_total_bytes:
                    raise _UsdzValidationStagingError(
                        f"{SIMREADY_SUPPORT_BUDGET_ERROR_PREFIX}: aggregate_bytes "
                        f"limit={max_total_bytes}"
                    )
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise _UsdzValidationStagingError(
            f"Malformed USDZ archive {asset_path}: {exc}"
        ) from exc

    if root_parts is None:
        raise _UsdzValidationStagingError(
            f"USDZ package has no package root: {asset_path}"
        )

    # USDZ package order defines the first non-directory member as the root layer.
    root = Path(*root_parts)
    if root.suffix.lower() not in SIMREADY_VALIDATOR_SUFFIXES:
        raise _UsdzValidationStagingError(
            "USDZ package root uses an unsupported SimReady validator suffix; "
            f"expected .usd or .usda, found {root.as_posix()}"
        )
    return root


def _validated_usdz_member_parts(info: zipfile.ZipInfo) -> tuple[str, ...]:
    parts = safe_usdz_member_parts(info.filename)
    if parts is None or PureWindowsPath("/".join(parts)).drive:
        raise _UsdzValidationStagingError(
            f"USDZ package contains an unsafe archive member: {info.filename}"
        )

    member_type = stat.S_IFMT(info.external_attr >> 16)
    expected_types = {0, stat.S_IFDIR} if info.is_dir() else {0, stat.S_IFREG}
    if member_type not in expected_types:
        raise _UsdzValidationStagingError(
            "USDZ package contains an unsafe non-regular archive member: "
            f"{info.filename}"
        )
    return parts


def _validation_workspace_path(report_path: Path) -> Path:
    return report_path.with_name(
        f".{report_path.name}{USDZ_VALIDATION_WORKSPACE_SUFFIX}"
    )


def _cleanup_validation_target(target: _ValidationTarget) -> str | None:
    if target.workspace_path is None:
        return None
    return _remove_validation_workspace(target.workspace_path)


def _remove_validation_workspace(workspace_path: Path) -> str | None:
    try:
        shutil.rmtree(workspace_path)
    except OSError as exc:
        return f"Could not remove USDZ validation workspace {workspace_path}: {exc}"
    return None


def _command_for_report(
    invocation_command: list[str],
    *,
    validator_asset_path: Path,
    original_asset_path: Path,
    invocation_raw_report_path: Path,
    reported_raw_report_path: Path,
) -> list[str]:
    command = list(invocation_command)
    if not command or command[-1] != str(validator_asset_path):
        raise RuntimeError(
            "SimReady validation command did not end with the staged validator target."
        )
    if validator_asset_path != original_asset_path:
        command[-1] = str(original_asset_path)
    try:
        output_index = command.index("--output") + 1
    except (ValueError, IndexError) as exc:
        raise RuntimeError("SimReady validation command has no output path.") from exc
    if command[output_index] != str(invocation_raw_report_path):
        raise RuntimeError("SimReady validation command output identity differs.")
    command[output_index] = str(reported_raw_report_path)
    return command


def _validation_staging_evidence(
    target: _ValidationTarget,
    *,
    original_asset_path: Path,
    cleanup_error: str | None,
) -> dict[str, Any] | None:
    if not target.staged_usdz:
        return None
    return {
        "original_asset_path": str(original_asset_path),
        "package_root": target.package_root,
        "validator_target": str(target.asset_path),
        "workspace_path": str(target.workspace_path),
        "workspace_cleanup": "failed" if cleanup_error is not None else "removed",
    }


def _usdz_staging_warning(staging_evidence: dict[str, Any]) -> str:
    return (
        "SimReady Foundation received extracted USDZ package root "
        f"{staging_evidence['package_root']!r} at "
        f"{staging_evidence['validator_target']}; normalized evidence remains "
        f"bound to the original asset {staging_evidence['original_asset_path']}."
    )


def _blocked_report(
    *,
    asset_path: Path,
    params: SimReadyValidationInput,
    report_path: Path,
    errors: list[str],
    stdout_log_path: Path,
    stderr_log_path: Path,
    warnings: list[str] | None = None,
    runtime: dict[str, Any] | None = None,
    next_step: str = "prepare-simready-foundation-runtime",
    validation_policy: dict[str, Any] | None = None,
) -> SimReadyValidationReport:
    runtime = runtime or {}
    return SimReadyValidationReport(
        asset_path=str(asset_path),
        asset_sha256=_regular_file_sha256(asset_path),
        passed=False,
        status="BLOCKED",
        profile_name=params.profile,
        profile_version=params.profile_version,
        profile_target=f"{params.profile}@{params.profile_version}",
        foundation_root=runtime.get("foundation_root"),
        foundation_commit=runtime.get("foundation_commit"),
        foundation_checkout_verified=bool(runtime.get("foundation_checkout_verified")),
        foundation_requirements_sha256=runtime.get("foundation_requirements_sha256"),
        foundation_spec_tree_sha256=runtime.get("foundation_spec_tree_sha256"),
        foundation_spec_root=runtime.get("foundation_spec_root"),
        validator_executable=runtime.get("validator_executable"),
        validator_executable_sha256=runtime.get("validator_executable_sha256"),
        validator_distributions_sha256=runtime.get("validator_distributions_sha256"),
        validator_runtime_verified=bool(runtime.get("validator_runtime_verified")),
        runtime_contract_sha256=runtime.get("runtime_contract_sha256"),
        available_profiles=list(runtime.get("available_profiles") or []),
        warnings=warnings or [],
        errors=errors,
        needs_rerun=False,
        rerun_reasons=[],
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        raw_report_path=str(report_path),
        next_step=next_step,
        validation_policy=validation_policy
        or {
            "blocking": False,
            "blocked_runtime_is_not_profile_failure": True,
        },
    )


def _normalize_validation_report(
    *,
    asset_path: Path,
    asset_sha256: str,
    asset_dependency_manifest: dict[str, Any],
    validator_asset_path: Path,
    params: SimReadyValidationInput,
    runtime: dict[str, Any],
    command: list[str],
    returncode: int,
    raw_payload: dict[str, Any],
    raw_report_path: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
    subprocess_error: str | None,
    adapter_warnings: list[str],
    adapter_errors: list[str],
    staging_evidence: dict[str, Any] | None,
) -> SimReadyValidationReport:
    issues = _extract_issues(raw_payload)
    asset_payload = _asset_payload(raw_payload, validator_asset_path)
    feature_results = raw_payload.get("feature_results", raw_payload.get("features"))
    if feature_results is None:
        feature_results = _feature_summary_results(asset_payload)
    profile_results = raw_payload.get(
        "profile_results", raw_payload.get("profile", asset_payload)
    )
    requirement_counts = _coerce_int_dict(raw_payload.get("requirement_counts"))
    warnings = _string_list(raw_payload.get("warnings"))
    errors = _string_list(raw_payload.get("errors"))
    toolchain_error = bool(raw_payload.get(TOOLCHAIN_ERROR_KEY))
    raw_passed = _optional_bool(raw_payload.get("passed"))
    raw_status = str(raw_payload.get("status") or "").upper()
    warnings.extend(adapter_warnings)
    errors.extend(adapter_errors)
    if adapter_errors:
        toolchain_error = True
    if subprocess_error:
        errors.append(subprocess_error)
        toolchain_error = True
    if returncode not in {0, 1}:
        errors.append(
            "simready-validate exited with unexpected status "
            f"{returncode}; only 0 (success) and 1 (validation findings) are "
            "part of the validator contract."
        )
        toolchain_error = True
    if returncode != 0 and (raw_passed is True or raw_status == "PASS"):
        errors.append(
            "simready-validate reported a passing payload but exited non-zero "
            f"with status {returncode}."
        )
        toolchain_error = True

    topology = _inspect_asset_topology(asset_path)
    ignored_issues: list[dict[str, Any]] = []
    if topology.get("single_prim_or_geomsubset") or topology.get("single_rigid_body"):
        remaining: list[dict[str, Any]] = []
        for issue in issues:
            requirement = _issue_requirement(issue)
            if requirement == NONBLOCKING_SINGLE_COMPONENT_REQUIREMENT:
                ignored = dict(issue)
                ignored["ignored_reason"] = (
                    "Asset is authored as a single component or one rigid body; "
                    "the optional multi-body requirement is not applicable."
                )
                ignored_issues.append(ignored)
            else:
                remaining.append(issue)
        issues = remaining
        ignored_issues.extend(
            _ignored_feature_requirements_for_single_component(feature_results)
        )

    failed_requirements = _active_failed_requirements(
        issues,
        feature_results,
        ignored_requirements={NONBLOCKING_SINGLE_COMPONENT_REQUIREMENT}
        if ignored_issues
        else set(),
    )
    issue_counts = _count_issues(issues)

    passed = (
        bool(raw_passed)
        if raw_passed is not None
        else returncode == 0 and not _has_error_issues(issues, failed_requirements)
    )
    if errors or _has_error_issues(issues, failed_requirements):
        passed = False
    elif ignored_issues and not failed_requirements:
        passed = True
    ignored_only_nonzero_exit = (
        returncode == 1
        and bool(ignored_issues)
        and not failed_requirements
        and not errors
    )
    if toolchain_error or (returncode != 0 and not ignored_only_nonzero_exit):
        passed = False

    failed_requirements = sorted(set(failed_requirements))
    needs_rerun = bool(failed_requirements) and not passed
    status = (
        "ERROR"
        if toolchain_error
        else "PASS"
        if passed
        else str(raw_payload.get("status") or "FAIL")
    )
    if status == "PASS" and not passed:
        status = "FAIL"

    if ignored_issues:
        for ignored in ignored_issues:
            ignored.setdefault(
                "ignored_reason",
                "Requirement is not applicable to this single-component asset.",
            )

    validation_policy: dict[str, Any] = {
        "blocking": False,
        "strict_available": True,
        "single_component_requirement_ignored": bool(ignored_issues),
    }
    if staging_evidence is not None:
        validation_policy["usdz_validation_staging"] = staging_evidence

    return SimReadyValidationReport(
        asset_path=str(asset_path),
        asset_sha256=asset_sha256,
        asset_dependency_manifest=asset_dependency_manifest,
        passed=passed,
        status=status,
        profile_name=params.profile,
        profile_version=params.profile_version,
        profile_target=f"{params.profile}@{params.profile_version}",
        command=command,
        foundation_root=runtime.get("foundation_root"),
        foundation_commit=runtime.get("foundation_commit"),
        foundation_checkout_verified=bool(runtime.get("foundation_checkout_verified")),
        foundation_requirements_sha256=runtime.get("foundation_requirements_sha256"),
        foundation_spec_tree_sha256=runtime.get("foundation_spec_tree_sha256"),
        foundation_spec_root=runtime.get("foundation_spec_root"),
        validator_executable=runtime.get("validator_executable"),
        validator_executable_sha256=runtime.get("validator_executable_sha256"),
        validator_distributions_sha256=runtime.get("validator_distributions_sha256"),
        validator_runtime_verified=bool(runtime.get("validator_runtime_verified")),
        runtime_contract_sha256=runtime.get("runtime_contract_sha256"),
        available_profiles=list(runtime.get("available_profiles") or []),
        profile_results=profile_results,
        feature_results=feature_results,
        requirement_counts=requirement_counts,
        issue_counts=issue_counts,
        issues=issues,
        ignored_issues=ignored_issues,
        asset_topology=topology,
        validation_policy=validation_policy,
        warnings=_dedupe(warnings),
        errors=_dedupe(errors),
        needs_rerun=needs_rerun,
        rerun_reasons=failed_requirements,
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        raw_report_path=str(raw_report_path),
        next_step="fix-simready-validator-runtime"
        if toolchain_error
        else "simready-conform-profile"
        if needs_rerun
        else "complete",
    )


def _inspect_asset_topology(asset_path: Path) -> dict[str, Any]:
    topology: dict[str, Any] = {
        "mesh_count": None,
        "geom_subset_count": None,
        "mesh_with_geom_subset_count": None,
        "component_count": None,
        "rigid_body_count": None,
        "single_prim_or_geomsubset": False,
        "single_rigid_body": False,
        "inspected": False,
    }
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        topology["warning"] = "OpenUSD Python APIs are unavailable."
        return topology
    try:
        stage = Usd.Stage.Open(str(asset_path))
    except Exception as exc:  # pragma: no cover - OpenUSD exception types vary
        topology["warning"] = (
            "OpenUSD could not open asset for topology inspection: " + str(exc)
        )
        return topology
    if stage is None:
        topology["warning"] = "OpenUSD could not open asset for topology inspection."
        return topology
    subset_type = getattr(UsdGeom, "GeomSubset", None) or getattr(
        UsdGeom, "Subset", None
    )
    mesh_count = 0
    geom_subset_count = 0
    mesh_with_geom_subset_count = 0
    rigid_body_count = 0
    try:
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_count += 1
            if prim.IsA(UsdGeom.Mesh):
                mesh_count += 1
                child_subset_count = (
                    sum(1 for child in prim.GetChildren() if child.IsA(subset_type))
                    if subset_type is not None
                    else 0
                )
                if child_subset_count:
                    mesh_with_geom_subset_count += 1
            elif subset_type is not None and prim.IsA(subset_type):
                geom_subset_count += 1
    except Exception as exc:  # pragma: no cover - OpenUSD exception types vary
        topology["warning"] = "OpenUSD could not inspect asset topology: " + str(exc)
        return topology
    component_count = (mesh_count - mesh_with_geom_subset_count) + geom_subset_count
    topology.update(
        {
            "mesh_count": mesh_count,
            "geom_subset_count": geom_subset_count,
            "mesh_with_geom_subset_count": mesh_with_geom_subset_count,
            "component_count": component_count,
            "rigid_body_count": rigid_body_count,
            "single_prim_or_geomsubset": component_count == 1,
            "single_rigid_body": rigid_body_count == 1,
            "inspected": True,
        }
    )
    return topology


def _extract_issues(raw_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_issues = raw_payload.get("issues")
    if isinstance(raw_issues, list):
        return [dict(item) for item in raw_issues if isinstance(item, dict)]
    raw_failures = raw_payload.get("failures")
    if isinstance(raw_failures, list):
        issues: list[dict[str, Any]] = []
        for item in raw_failures:
            if isinstance(item, dict):
                issues.append(dict(item))
            else:
                issues.append({"message": str(item), "severity": "FAILURE"})
        return issues
    return []


def _has_error_issues(
    issues: list[dict[str, Any]], failed_requirements: list[str]
) -> bool:
    if any(
        str(issue.get("severity", "")).upper() in ERROR_SEVERITIES for issue in issues
    ):
        return True
    return bool(failed_requirements)


def _asset_payload(raw_payload: dict[str, Any], asset_path: Path) -> dict[str, Any]:
    """Return the per-asset payload used by Foundation's compact JSON shape."""

    direct = raw_payload.get(str(asset_path))
    if isinstance(direct, dict):
        return direct
    resolved = str(asset_path.resolve())
    direct = raw_payload.get(resolved)
    if isinstance(direct, dict):
        return direct
    if "features_summary" in raw_payload:
        return raw_payload
    values = list(raw_payload.values())
    if len(values) == 1 and isinstance(values[0], dict):
        candidate = values[0]
        if "features_summary" in candidate:
            return candidate
    return {}


def _feature_summary_results(asset_payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary = asset_payload.get("features_summary")
    if not isinstance(summary, dict):
        return []
    results: list[dict[str, Any]] = []
    for feature_id, value in summary.items():
        if not isinstance(value, dict):
            continue
        result = {"id": str(feature_id), **value}
        if "failing requirements" in result and "failing_requirements" not in result:
            result["failing_requirements"] = result["failing requirements"]
        results.append(result)
    return results


def _active_failed_requirements(
    issues: list[dict[str, Any]],
    feature_results: Any,
    *,
    ignored_requirements: set[str],
) -> list[str]:
    failures = [
        requirement
        for issue in issues
        if (requirement := _issue_requirement(issue)) is not None
    ]
    failures.extend(_feature_failing_requirements(feature_results))
    return sorted(
        {
            requirement
            for requirement in failures
            if requirement not in ignored_requirements
        }
    )


def _ignored_feature_requirements_for_single_component(
    feature_results: Any,
) -> list[dict[str, Any]]:
    ignored: list[dict[str, Any]] = []
    items = (
        feature_results.values()
        if isinstance(feature_results, dict)
        else feature_results
    )
    if not isinstance(items, list) and not hasattr(items, "__iter__"):
        return ignored
    for feature in items:
        if not isinstance(feature, dict):
            continue
        failing = _feature_requirement_list(feature)
        if NONBLOCKING_SINGLE_COMPONENT_REQUIREMENT not in failing:
            continue
        feature_id = str(feature.get("feature_id") or feature.get("id") or "")
        ignored.append(
            {
                "requirement_id": NONBLOCKING_SINGLE_COMPONENT_REQUIREMENT,
                "feature_id": feature_id,
                "severity": "IGNORED",
                "message": (
                    f"{NONBLOCKING_SINGLE_COMPONENT_REQUIREMENT} is not applicable "
                    "to a single-component or explicitly single-rigid-body prop."
                ),
                "ignored_reason": (
                    "Physical AI Skill Hub SimReady conformance policy treats "
                    "RB.MB.001 as non-blocking/not applicable for single-body props "
                    "and forbids inventing rigid bodies to satisfy it."
                ),
            }
        )
    return ignored


def _feature_failing_requirements(feature_results: Any) -> list[str]:
    failures: list[str] = []
    items = (
        feature_results.values()
        if isinstance(feature_results, dict)
        else feature_results
    )
    if not isinstance(items, list) and not hasattr(items, "__iter__"):
        return failures
    for feature in items:
        if not isinstance(feature, dict):
            continue
        failing = _feature_requirement_list(feature)
        failures.extend(failing)
        if feature.get("passed") is False and not failing:
            feature_id = (
                feature.get("feature_id") or feature.get("id") or "unknown_feature"
            )
            failures.append(str(feature_id))
    return failures


def _feature_requirement_list(feature: dict[str, Any]) -> list[str]:
    failing = feature.get("failing_requirements")
    if failing is None:
        failing = feature.get("failing requirements")
    if isinstance(failing, list):
        return [
            requirement
            for item in failing
            if (requirement := _parse_requirement(item)) is not None
        ]
    if isinstance(failing, str):
        return _parse_requirements(failing)
    return []


def _issue_requirement(issue: dict[str, Any]) -> str | None:
    for key in ("requirement_id", "requirement", "code"):
        if requirement := _parse_requirement(issue.get(key)):
            return requirement
    return _parse_requirement(issue.get("message"))


def _parse_requirement(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"\b[A-Z]+(?:\.[A-Z]+)*\.\d+\b", str(value))
    return match.group(0) if match else None


def _parse_requirements(value: str) -> list[str]:
    return re.findall(r"\b[A-Z]+(?:\.[A-Z]+)*\.\d+\b", value)


def _coerce_int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        try:
            result[str(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return result


def _count_issues(issues: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        severity = str(issue.get("severity") or "UNKNOWN").upper()
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _load_json_if_present(
    path: Path,
    *,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    try:
        artifact = open_held_confined_artifact(path.parent, path.name)
        descriptor = artifact.stream.fileno()
    except FileNotFoundError:
        return {
            TOOLCHAIN_ERROR_KEY: True,
            "errors": [f"Validation raw report was not written: {path}"],
        }
    except (ArtifactPathError, OSError, RuntimeError, ValueError) as exc:
        return {
            TOOLCHAIN_ERROR_KEY: True,
            "errors": [f"Validation raw report could not be opened: {exc}"],
        }
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("raw report is not a regular file")
        if max_bytes is not None and before.st_size > max_bytes:
            raise ValueError(
                f"{SIMREADY_SUPPORT_BUDGET_ERROR_PREFIX}: file_bytes limit={max_bytes}"
            )
        # Even without an explicit policy limit, never chase a concurrently
        # growing file to EOF.  The opened snapshot size is the maximum payload
        # this read can legitimately consume; one extra byte is enough to prove
        # that the report changed after ``fstat``.
        read_limit = before.st_size if max_bytes is None else max_bytes
        remaining = read_limit
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(
            descriptor,
            min(1024 * 1024, remaining + 1),
        ):
            size += len(chunk)
            if size > read_limit:
                if max_bytes is not None:
                    raise ValueError(
                        f"{SIMREADY_SUPPORT_BUDGET_ERROR_PREFIX}: file_bytes "
                        f"limit={max_bytes}"
                    )
                raise ValueError("raw report changed while it was read")
            chunks.append(chunk)
            remaining = max(0, remaining - len(chunk))
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or size != after.st_size:
            raise ValueError("raw report changed while it was read")
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {
            TOOLCHAIN_ERROR_KEY: True,
            "errors": [f"Validation raw report could not be parsed: {exc}"],
        }
    finally:
        artifact.stream.close()
    if not isinstance(payload, dict):
        return {
            TOOLCHAIN_ERROR_KEY: True,
            "errors": [f"Validation raw report is not a JSON object: {path}"],
        }
    return payload


def _write_report(
    report_path: Path,
    report: SimReadyValidationReport,
    *,
    max_bytes: int | None = None,
) -> SimReadyValidationReport:
    report.report_path = str(report_path)
    payload = (
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if max_bytes is not None and len(payload) > max_bytes:
        raise RuntimeError(
            f"{SIMREADY_SUPPORT_BUDGET_ERROR_PREFIX}: file_bytes limit={max_bytes}"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(payload)
    return report


def _regular_file_sha256(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _as_json(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a USD asset against a SimReady Foundation profile."
    )
    parser.add_argument("asset_path", type=Path)
    parser.add_argument("--profile", default=DEFAULT_SIMREADY_PROFILE)
    parser.add_argument("--profile-version", default=DEFAULT_SIMREADY_PROFILE_VERSION)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--foundation-root", type=Path)
    parser.add_argument("--foundation-spec-root", type=Path)
    parser.add_argument("--venv", dest="venv_path", type=Path)
    parser.add_argument(
        "--install-missing",
        dest="install_missing",
        action="store_true",
        default=True,
        help="Install missing SimReady validation dependencies. This is the default.",
    )
    parser.add_argument(
        "--no-install-missing",
        dest="install_missing",
        action="store_false",
        help="Check only; do not clone or install missing dependencies.",
    )
    parser.add_argument(
        "--update-foundation",
        action="store_true",
        help=(
            "Prepare a missing checkout for the selected immutable Foundation ref. "
            "Existing managed checkouts are never updated in place."
        ),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when the recorded request and source identity match.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when profile validation fails.",
    )
    args = parser.parse_args(argv)

    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(args.asset_path),
            profile=args.profile,
            profile_version=args.profile_version,
            report_path=str(args.report) if args.report is not None else None,
            foundation_root=str(args.foundation_root)
            if args.foundation_root is not None
            else None,
            foundation_spec_root=str(args.foundation_spec_root)
            if args.foundation_spec_root is not None
            else None,
            venv_path=str(args.venv_path) if args.venv_path is not None else None,
            install_missing=args.install_missing,
            update_foundation=args.update_foundation,
            timeout_s=args.timeout,
            stdout_log_path=str(args.stdout_log)
            if args.stdout_log is not None
            else None,
            stderr_log_path=str(args.stderr_log)
            if args.stderr_log is not None
            else None,
            resume=args.resume,
        )
    )
    print(json.dumps(_as_json(report), indent=2, sort_keys=True))
    if report.passed:
        return 0
    if args.strict:
        return 1
    if (
        report.status == "BLOCKED"
        or report.errors
        or report.next_step == "fix-simready-validator-runtime"
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
