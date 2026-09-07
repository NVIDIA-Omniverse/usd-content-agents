# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal usd-cli distribution and OVRTX workflow preflight."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    read_contained_artifact,
)
from content_agent_workflows.common.usd_cli import (
    USD_CLI_REQUIRED_SOURCE,
    UsdCliSubprocessOutputError,
    resolve_package_owned_usd_cli_route,
    run_bounded_usd_cli_subprocess,
    sanitized_usd_cli_execution_env,
    usd_cli_source_distributed,
    validate_ovrtx_probe,
)
from content_agent_workflows.common.usd_cli_session import WorkflowUsdCliSession

OVRTX_PROBE_ARTIFACT_SCHEMA_VERSION = "content-agents.ovrtx-probe.v1"
_OVRTX_PROVISIONING_POLL_SECONDS = 5.0
_OVRTX_PROVISIONING_MESSAGES = (
    "ovrtx auto-install STARTED in the background",
    "ovrtx auto-install in progress",
)

__all__ = [
    "USD_CLI_REQUIRED_SOURCE",
    "UsdCliReadiness",
    "ensure_usd_cli_ovrtx_ready",
    "usd_cli_source_distributed",
]


@dataclass(frozen=True, slots=True)
class UsdCliReadiness:
    version: str
    source_revision: str
    probe: dict[str, Any]
    artifact_path: Path | None


def _wait_for_ovrtx_provisioning(
    probe_once: Any,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Retry only the daemon's explicit, non-terminal provisioning states."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            probe = probe_once(max(0.0, deadline - time.monotonic()))
        except RuntimeError as exc:
            detail = str(exc)
            if not any(message in detail for message in _OVRTX_PROVISIONING_MESSAGES):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "OVRTX provisioning did not finish before the workflow readiness "
                    f"timeout ({timeout_seconds:g} seconds). See the usd-cli daemon "
                    "log, then retry the workflow or configure a remote OVRTX service."
                ) from exc
            time.sleep(min(_OVRTX_PROVISIONING_POLL_SECONDS, remaining))
            continue
        if not isinstance(probe, dict):
            raise RuntimeError("usd-cli OVRTX probe returned a non-object JSON value.")
        return probe


def ensure_usd_cli_ovrtx_ready(
    repo_root: Path,
    *,
    run_dir: Path | None = None,
    executable: Path | None = None,
    timeout_seconds: float = 900.0,
    required_capabilities: tuple[str, ...] = ("appearance.clear.v1",),
    session: WorkflowUsdCliSession | None = None,
    artifact_stem: str = "ovrtx_probe",
) -> UsdCliReadiness:
    """Fail closed unless the exact in-tree package can render via OVRTX."""

    if re.fullmatch(r"[a-z][a-z0-9_-]{0,127}", artifact_stem) is None:
        raise ValueError("usd-cli readiness artifact stem is invalid")
    try:
        route = resolve_package_owned_usd_cli_route(repo_root)
    except RuntimeError as exc:
        raise RuntimeError(
            "usd-cli is unavailable as an exact package-owned workflow backend: "
            f"{exc}. Install or repair the in-tree usd-cli package."
        ) from exc
    if executable is not None:
        try:
            requested_executable = executable.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                f"The requested usd-cli executable is unavailable: {executable}"
            ) from exc
        if requested_executable != route.target:
            raise RuntimeError(
                "The requested usd-cli executable is not the package-owned "
                f"launcher: {requested_executable}"
            )
    if session is not None and session.route != route:
        raise RuntimeError(
            "The supplied workflow session does not use the package-owned usd-cli route."
        )
    executable_value = str(route.target)
    version_process = _run(
        [executable_value, "--version"],
        timeout_seconds=min(timeout_seconds, 60.0),
        description="usd-cli version check",
    )
    version = version_process.stdout.strip()
    if not version:
        raise RuntimeError("`usd-cli --version` returned an empty version banner.")

    probe_dir: Path | None = None
    session_raw_dir: Path | None = None
    if session is not None:
        # Pin the workflow-owned 0700 raw directory before the probe packet's
        # atomic writer can create it using the ambient process umask.
        session_raw_dir = session.prepare_raw_directory()
    if run_dir is not None:
        run_root = run_dir.resolve(strict=True)
        if session_raw_dir is not None and session_raw_dir != run_root / "raw":
            raise RuntimeError(
                "The supplied workflow session belongs to a different run directory."
            )
        probe_dir = run_root / "raw" / artifact_stem
        atomic_write_json(
            probe_dir / ".workflow-owned.json",
            {"owner": "content-workflow-cli", "purpose": "ovrtx-probe"},
            within=run_root,
        )
    if session is not None:
        assert session_raw_dir is not None

        def probe_once(_remaining: float) -> dict[str, Any]:
            return session.require_ovrtx(probe_dir or session_raw_dir / "ovrtx_probe")

    else:
        command = [
            executable_value,
            "--json",
            "render-probe",
            "--require-engine",
            "ovrtx",
        ]
        if probe_dir is not None:
            command.extend(["--output-dir", str(probe_dir)])

        def probe_once(remaining: float) -> dict[str, Any]:
            probe_process = _run(
                command,
                timeout_seconds=remaining,
                description="OVRTX readiness/render probe",
            )
            try:
                probe = json.loads(probe_process.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "usd-cli OVRTX probe did not return a JSON object: "
                    f"{probe_process.stdout[-1000:]!r}"
                ) from exc
            if not isinstance(probe, dict):
                raise RuntimeError(
                    "usd-cli OVRTX probe returned a non-object JSON value."
                )
            return probe

    probe = _wait_for_ovrtx_provisioning(
        probe_once,
        timeout_seconds=timeout_seconds,
    )
    validate_ovrtx_probe(
        probe,
        required_capabilities=required_capabilities,
    )
    if run_dir is not None:
        render = probe["render"]
        render_path = render.get("path")
        if not isinstance(render_path, str) or not render_path:
            raise RuntimeError("OVRTX probe omitted its run-local render path.")
        render_artifact = read_contained_artifact(
            run_dir,
            render_path,
            max_bytes=16 * 1024 * 1024,
            image=True,
        )
        if render.get("size_bytes") != render_artifact.size_bytes:
            raise RuntimeError(
                "OVRTX probe render size does not match the captured artifact."
            )
        probe = {
            **probe,
            "render": {
                **render,
                "size_bytes": render_artifact.size_bytes,
                "sha256": render_artifact.sha256,
            },
        }

    artifact_path: Path | None = None
    if run_dir is not None:
        artifact_path = run_dir.resolve() / "raw" / f"{artifact_stem}.json"
        atomic_write_json(
            artifact_path,
            {
                "schema_version": OVRTX_PROBE_ARTIFACT_SCHEMA_VERSION,
                "usd_cli_version": version,
                "usd_cli_source_revision": route.source_revision,
                "probe": probe,
            },
            within=run_dir.resolve(strict=True),
        )
    return UsdCliReadiness(
        version=version,
        source_revision=route.source_revision,
        probe=probe,
        artifact_path=artifact_path,
    )


def _run(
    command: list[str],
    *,
    timeout_seconds: float,
    description: str,
) -> subprocess.CompletedProcess[str]:
    environment = sanitized_usd_cli_execution_env(
        executable_dir=Path(command[0]).resolve().parent,
    )
    # The sanitizer intentionally removes user-local PATH entries. Preserve the
    # operator-resolved uv executable as an absolute capability so the shared
    # Texture/Articulation readiness path can provision the hash-pinned OVRTX
    # runtime on first use. usd-cli revalidates this path before executing it.
    if not environment.get("USD_CLI_UV_EXECUTABLE"):
        uv_executable = shutil.which("uv", path=os.environ.get("PATH", os.defpath))
        if uv_executable:
            environment["USD_CLI_UV_EXECUTABLE"] = str(
                Path(uv_executable).resolve(strict=True)
            )
    try:
        return run_bounded_usd_cli_subprocess(
            command,
            check=True,
            # Every backend command starts with the package-verified absolute
            # route target; pin the child's search path to that launcher's
            # directory plus the fixed system path.
            env=environment,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{description} timed out after {timeout_seconds:g} seconds."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"{description} failed closed: {detail}") from exc
    except UsdCliSubprocessOutputError as exc:
        raise RuntimeError(f"{description} failed closed: {exc}") from exc
