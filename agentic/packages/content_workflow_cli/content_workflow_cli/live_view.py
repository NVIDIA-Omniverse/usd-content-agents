# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch and supervise the optional browser live-view sidecar."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

LIVE_VIEW_SCHEMA_VERSION = "content-workflow-cli.live-view.v1"
VIEWER_RESULT_SCHEMA_VERSION = "content-workflow-cli.viewer-result.v1"
LIVE_VIEW_PROGRESS_ENV = "CONTENT_WORKFLOW_LIVE_VIEWER_PROGRESS"
LIVE_VIEW_CHILD_LAUNCHED_ENV = "CONTENT_WORKFLOW_LIVE_VIEWER_CHILD_LAUNCHED"
USD_CLI_EXTERNAL_LIFECYCLE_BOOTSTRAP_ENV = "USD_CLI_EXTERNAL_LIFECYCLE_BOOTSTRAP"
USD_CLI_SESSION_ENV = "USD_CLI_SESSION"
USD_CLI_DAEMON_TOKEN_ENVS = (
    "USD_CLI_TOKEN",
    "USD_CLI_SERVER_TOKEN",
    "OV_TOKEN",
    "OV_SERVER_TOKEN",
    "3DSC_TOKEN",
    "3DSC_SERVER_TOKEN",
)
VIEWER_COMMAND_ENV = "CONTENT_WORKFLOW_VIEWER_COMMAND"
VIEWER_RUN_ROOT_ENV = "CONTENT_WORKFLOW_LIVE_VIEW_RUN_ROOT"
logger = logging.getLogger(__name__)


class LiveViewError(RuntimeError):
    """The requested live viewer could not be started safely."""


def add_live_view_args(
    parser: argparse.ArgumentParser,
    *,
    workflow: str,
    asset_argument: str = "usd",
    output_argument: str = "output_dir",
) -> None:
    """Add the common live-view contract to one fresh-run command."""

    if not _viewer_package_available():
        return

    group = parser.add_argument_group("live view")
    group.add_argument(
        "--live-view",
        action="store_true",
        help=(
            "Start the OVRTX/WebRTC workflow viewer and print its URL before "
            "the workflow begins."
        ),
    )
    group.add_argument(
        "--live-view-bind-host",
        default=os.getenv("CONTENT_WORKFLOW_LIVE_VIEW_BIND_HOST", "127.0.0.1"),
        help="HTTP bind address. Defaults to loopback; use 0.0.0.0 deliberately.",
    )
    group.add_argument(
        "--live-view-public-host",
        default=os.getenv("CONTENT_WORKFLOW_LIVE_VIEW_PUBLIC_HOST"),
        help=(
            "Browser-reachable host advertised in the viewer URL. Non-loopback "
            "values are also advertised in WebRTC SDP."
        ),
    )
    group.add_argument(
        "--live-view-port",
        type=_port_number,
        default=_env_port("CONTENT_WORKFLOW_LIVE_VIEW_PORT"),
        help="Viewer HTTP port. Defaults to an available local port.",
    )
    group.add_argument(
        "--live-view-signaling-port",
        type=_port_number,
        default=_env_port("CONTENT_WORKFLOW_LIVE_VIEW_SIGNALING_PORT"),
        help="OvStream WebRTC signaling port. Defaults to an available local port.",
    )
    group.add_argument(
        "--live-view-media-port",
        type=_port_number,
        default=_env_port("CONTENT_WORKFLOW_LIVE_VIEW_MEDIA_PORT"),
        help=(
            "OvStream WebRTC UDP media port. Defaults to an available UDP port; "
            "allow this port through the firewall for direct browser viewing."
        ),
    )
    group.add_argument(
        "--live-view-retain-seconds",
        type=_non_negative_float,
        default=_env_float("CONTENT_WORKFLOW_LIVE_VIEW_RETAIN_SECONDS", 900.0),
        help="Seconds to keep the final result available after workflow completion.",
    )
    group.add_argument(
        "--live-view-startup-timeout",
        type=_positive_float,
        default=_env_float("CONTENT_WORKFLOW_LIVE_VIEW_STARTUP_TIMEOUT", 45.0),
        help="Seconds to wait for the viewer HTTP and OvStream processes to start.",
    )
    parser.set_defaults(
        live_view_workflow=workflow,
        live_view_asset_argument=asset_argument,
        live_view_output_argument=output_argument,
    )


def _viewer_package_available() -> bool:
    """Return whether a complete optional viewer runtime can be launched."""

    module_path = Path(__file__).resolve()
    source_package_root = module_path.parents[1]
    if (source_package_root / "pyproject.toml").is_file():
        source_viewer_root = source_package_root.parent / "content_workflow_viewer"
        if not source_viewer_root.is_dir():
            return False

    try:
        _command, viewer_root = _viewer_command()
    except LiveViewError:
        return False
    if viewer_root is None:
        return True
    return (viewer_root / "frontend" / "dist" / "index.html").is_file()


def run_with_live_view(args: argparse.Namespace) -> int:
    """Run the selected CLI handler with a detached live-view sidecar."""

    if getattr(args, "deterministic_workflow", False):
        raise LiveViewError(
            "--live-view cannot be combined with physics --direct-executor; "
            "use the agent-authored physics workflow."
        )
    if getattr(args, "dry_run", False):
        raise LiveViewError("--live-view cannot be combined with --dry-run.")
    run_dir = _resolve_run_dir(args)
    if run_dir.exists() or run_dir.is_symlink():
        raise LiveViewError(
            "--live-view requires a fresh output directory; refusing to write "
            f"viewer artifacts into existing path: {run_dir}"
        )
    workflow = str(args.live_view_workflow)
    asset = _asset_label(args)
    from .runner import _workflow_usd_cli_session_id

    usd_cli_session = _workflow_usd_cli_session_id(
        workflow=_usd_cli_workflow_scope(workflow),
        project_dir=run_dir,
    )
    session = LiveViewSession(
        run_dir=run_dir,
        workflow=workflow,
        asset=asset,
        bind_host=str(args.live_view_bind_host),
        public_host=(
            str(args.live_view_public_host).strip()
            if args.live_view_public_host
            else None
        ),
        http_port=args.live_view_port,
        signaling_port=args.live_view_signaling_port,
        media_port=getattr(args, "live_view_media_port", None),
        retain_seconds=float(args.live_view_retain_seconds),
        startup_timeout=float(args.live_view_startup_timeout),
        usd_cli_session=usd_cli_session,
        usd_cli_executable=_usd_cli_executable(args),
    )
    session.start()
    print(f"Live view: {session.url}", file=sys.stderr, flush=True)
    direct_url = getattr(session, "webrtc_url", f"{session.url}?transport=webrtc")
    print(f"Live view WebRTC: {direct_url}", file=sys.stderr, flush=True)
    print(f"Live view metadata: {session.metadata_path}", file=sys.stderr, flush=True)

    previous_progress = os.environ.get(LIVE_VIEW_PROGRESS_ENV)
    previous_child_launched = os.environ.get(LIVE_VIEW_CHILD_LAUNCHED_ENV)
    previous_usd_cli_session = os.environ.get(USD_CLI_SESSION_ENV)
    os.environ[LIVE_VIEW_PROGRESS_ENV] = "1"
    os.environ[LIVE_VIEW_CHILD_LAUNCHED_ENV] = "0"
    os.environ[USD_CLI_SESSION_ENV] = usd_cli_session
    try:
        returncode = int(args.handler(args))
    except KeyboardInterrupt:
        _finish_without_masking(
            session,
            130,
            status="interrupted",
            retain=_child_was_launched(),
        )
        raise
    except BaseException:
        _finish_without_masking(
            session,
            2,
            status="failed",
            retain=_child_was_launched(),
        )
        raise
    else:
        retain = returncode == 0 or _child_was_launched()
        retained = session.finish(
            returncode,
            status="completed" if returncode == 0 else "failed",
            retain=retain,
        )
        if retained is not False:
            print(
                "Live view final result retained for "
                f"{_duration_label(session.retain_seconds)}: {session.url}",
                file=sys.stderr,
                flush=True,
            )
        elif not _report_unverified_cleanup(session):
            print(
                (
                    "Live view stopped after preflight failed before child launch; "
                    "no viewer URL was retained."
                    if returncode != 0 and not retain
                    else "Live view sidecar exited before retention; durable "
                    "workflow artifacts were published without a retained URL."
                ),
                file=sys.stderr,
                flush=True,
            )
        return returncode
    finally:
        if previous_progress is None:
            os.environ.pop(LIVE_VIEW_PROGRESS_ENV, None)
        else:
            os.environ[LIVE_VIEW_PROGRESS_ENV] = previous_progress
        if previous_child_launched is None:
            os.environ.pop(LIVE_VIEW_CHILD_LAUNCHED_ENV, None)
        else:
            os.environ[LIVE_VIEW_CHILD_LAUNCHED_ENV] = previous_child_launched
        if previous_usd_cli_session is None:
            os.environ.pop(USD_CLI_SESSION_ENV, None)
        else:
            os.environ[USD_CLI_SESSION_ENV] = previous_usd_cli_session


def _child_was_launched() -> bool:
    return os.environ.get(LIVE_VIEW_CHILD_LAUNCHED_ENV) == "1"


def _finish_without_masking(
    session: LiveViewSession,
    returncode: int,
    *,
    status: str,
    retain: bool,
) -> None:
    try:
        session.finish(returncode, status=status, retain=retain)
    except Exception:
        logger.exception(
            "Unable to finish the live viewer while preserving workflow failure"
        )
    else:
        _report_unverified_cleanup(session)


def _report_unverified_cleanup(session: LiveViewSession) -> bool:
    cleanup_error = getattr(session, "cleanup_error", None)
    if cleanup_error is None:
        return False
    print(
        "Live view cleanup could not be verified; the viewer process group may "
        "still be running and no URL was retained. "
        f"See {session.log_path}. Error: {cleanup_error}",
        file=sys.stderr,
        flush=True,
    )
    return True


@dataclass
class LiveViewSession:
    """One workflow viewer process plus its durable public handoff metadata."""

    run_dir: Path
    workflow: str
    asset: str
    bind_host: str = "127.0.0.1"
    public_host: str | None = None
    http_port: int | None = None
    signaling_port: int | None = None
    media_port: int | None = None
    retain_seconds: float = 900.0
    startup_timeout: float = 45.0
    usd_cli_session: str | None = None
    usd_cli_executable: Path | None = None
    process: subprocess.Popen[bytes] | None = field(default=None, init=False)
    url: str = field(default="", init=False)
    webrtc_url: str = field(default="", init=False)
    started_at: str = field(default="", init=False)
    webrtc_public_ip: str = field(default="", init=False)
    log_path: Path = field(init=False)
    metadata_path: Path = field(init=False)
    control_path: Path = field(init=False)
    cleanup_attempted: bool = field(default=False, init=False)
    cleanup_error: str | None = field(default=None, init=False)
    _metadata_stop: threading.Event = field(
        default_factory=threading.Event,
        init=False,
    )
    _metadata_thread: threading.Thread | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.run_dir = self.run_dir.expanduser().absolute()
        if self.usd_cli_executable is not None:
            self.usd_cli_executable = self.usd_cli_executable.expanduser().resolve()
        self.log_path = self.run_dir.with_name(f"{self.run_dir.name}.viewer.log")
        self.metadata_path = self.run_dir / "live_view.json"
        self.control_path = self.run_dir.with_name(
            f".{self.run_dir.name}.live-view-control.json"
        )

    def start(self) -> None:
        self._validate_hosts()
        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        explicit_http_port = self.http_port is not None
        explicit_signaling_port = self.signaling_port is not None
        self.http_port = self.http_port or _available_port(self.bind_host)
        self.signaling_port = self.signaling_port or _available_port(self.bind_host)
        self.media_port = self.media_port or _available_udp_port(self.bind_host)
        if self.http_port == self.signaling_port:
            if explicit_http_port and explicit_signaling_port:
                raise LiveViewError(
                    "Viewer HTTP and WebRTC signaling ports must be different."
                )
            for _attempt in range(8):
                self.signaling_port = _available_port(self.bind_host)
                if self.http_port != self.signaling_port:
                    break
            else:
                raise LiveViewError("Unable to allocate distinct live-view ports.")
        public_host = self.public_host or _default_public_host(self.bind_host)
        self.public_host = public_host
        self.webrtc_public_ip = _webrtc_public_ip(public_host)
        browser_host = _url_host(public_host)
        # The HTTP shell proxies /sign_in to the private OvStream signaling
        # listener. Keeping the browser connection same-origin lets embedded
        # and remotely proxied browsers follow the stream without direct
        # network access to this host's separate signaling port.
        self.url = f"http://{browser_host}:{self.http_port}/"
        self.webrtc_url = f"{self.url}?transport=webrtc"
        self.started_at = _utc_now()
        self._write_control(status="starting")

        command, viewer_root = _viewer_command()
        command.extend(
            [
                "--run-dir",
                str(self.run_dir),
                "--host",
                self.bind_host,
                "--port",
                str(self.http_port),
                "--signaling-port",
                str(self.signaling_port),
                "--stream-port",
                str(self.media_port),
                "--public-ip",
                self.webrtc_public_ip,
                "--workflow",
                self.workflow,
                "--asset",
                self.asset,
                "--terminal-state-file",
                str(self.control_path),
                "--launcher-pid",
                str(os.getpid()),
                "--stop-after-terminal-seconds",
                str(self.retain_seconds),
            ]
        )
        if self.usd_cli_session is not None and self.usd_cli_executable is not None:
            command.extend(
                [
                    "--usd-cli-session",
                    self.usd_cli_session,
                    "--usd-cli-executable",
                    str(self.usd_cli_executable),
                    "--usd-cli-project-dir",
                    str(self.run_dir),
                ]
            )
        environment = os.environ.copy()
        environment.pop(LIVE_VIEW_PROGRESS_ENV, None)
        environment.pop(LIVE_VIEW_CHILD_LAUNCHED_ENV, None)
        environment.pop(USD_CLI_EXTERNAL_LIFECYCLE_BOOTSTRAP_ENV, None)
        for name in USD_CLI_DAEMON_TOKEN_ENVS:
            environment.pop(name, None)
        environment.setdefault("OVRTX_SKIP_USD_CHECK", "1")
        # The workflow launcher exclusively owns the run-scoped usd-cli
        # daemon. A viewer may attach to that session, but it must never
        # bootstrap a replacement that could outlive a preflight failure or
        # the launcher's authenticated teardown lease.
        environment["USD_CLI_NO_DAEMON"] = "1"
        if viewer_root is not None:
            environment.setdefault("XDG_CACHE_HOME", str(viewer_root / ".cache"))
            environment.setdefault(
                "CUDA_CACHE_PATH", str(viewer_root / ".cache" / "cuda")
            )
            environment.setdefault(
                "__GL_SHADER_DISK_CACHE_PATH", str(viewer_root / ".cache" / "gl")
            )
        with self.log_path.open("ab", buffering=0) as log_stream:
            self.process = subprocess.Popen(
                command,
                cwd=str(viewer_root) if viewer_root is not None else None,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            _wait_for_livez(
                host=_probe_host(self.bind_host),
                port=self.http_port,
                timeout=self.startup_timeout,
                process=self.process,
                log_path=self.log_path,
            )
        except BaseException:
            self._terminate_process_without_masking()
            try:
                self._write_control(status="failed", error="viewer startup failed")
            except Exception:
                logger.exception(
                    "Unable to publish failed live-view startup state while "
                    "preserving the startup failure"
                )
            try:
                self.control_path.unlink(missing_ok=True)
            except OSError:
                logger.exception(
                    "Unable to remove failed live-view startup control while "
                    "preserving the startup failure"
                )
            _report_unverified_cleanup(self)
            raise
        self._write_control(status="running")
        self._metadata_thread = threading.Thread(
            target=self._publish_metadata_when_ready,
            name="live-view-metadata",
            daemon=True,
        )
        self._metadata_thread.start()

    def finish(self, returncode: int, *, status: str, retain: bool = True) -> bool:
        completed_at = _utc_now()
        retain_until = None
        if retain:
            retain_until = (
                datetime.now(tz=UTC) + timedelta(seconds=self.retain_seconds)
            ).isoformat()
        self._metadata_stop.set()
        if self._metadata_thread is not None:
            self._metadata_thread.join(timeout=1.0)
            self._metadata_thread = None
        if not retain:
            self._terminate_process_without_masking()
        viewer_available = retain and self._viewer_is_alive()
        if viewer_available:
            assert retain_until is not None
            self._write_control(
                status=status,
                completed_at=completed_at,
                exit_code=returncode,
                retain_until=retain_until,
            )
        if self._safe_run_dir_exists():
            terminal = {
                "schema_version": VIEWER_RESULT_SCHEMA_VERSION,
                "workflow": self.workflow,
                "exit_code": returncode,
                "status": status,
                "completed_at": completed_at,
                **self._cleanup_metadata(),
            }
            _atomic_write_json(
                self.run_dir / "viewer_workflow_result.json",
                terminal,
            )
            self._write_metadata(
                status=status,
                completed_at=completed_at,
                exit_code=returncode,
                viewer_available=viewer_available,
                **self._cleanup_metadata(),
                **(
                    {"retain_until": retain_until}
                    if viewer_available and retain_until is not None
                    else {}
                ),
            )
        else:
            # There is nothing useful to retain when workflow validation failed
            # before it could create its run directory.
            if self.process is not None and self.process.poll() is None:
                self._terminate_process_without_masking()
            self.control_path.unlink(missing_ok=True)
            return False
        viewer_available = viewer_available and self._viewer_is_alive()
        if not viewer_available:
            self.control_path.unlink(missing_ok=True)
            self._write_metadata(
                status=status,
                completed_at=completed_at,
                exit_code=returncode,
                viewer_available=False,
                **self._cleanup_metadata(),
            )
        return viewer_available

    def _viewer_is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _terminate_process_without_masking(self) -> None:
        """Log viewer cleanup failures without replacing workflow results."""

        self.cleanup_attempted = True
        self.cleanup_error = None
        try:
            self._terminate_process()
        except LiveViewError as exc:
            self.cleanup_error = str(exc)
            logger.exception(
                "Unable to stop the live viewer process group during finish"
            )

    def _cleanup_metadata(self) -> dict[str, object]:
        if not self.cleanup_attempted:
            return {}
        return {
            "viewer_cleanup_verified": self.cleanup_error is None,
            **(
                {"viewer_cleanup_error": self.cleanup_error}
                if self.cleanup_error is not None
                else {}
            ),
        }

    def _publish_metadata_when_ready(self) -> None:
        while not self._metadata_stop.wait(0.1):
            if self._safe_run_dir_exists():
                self._write_metadata(status="running")
                return
            if self.process is not None and self.process.poll() is not None:
                return

    def _safe_run_dir_exists(self) -> bool:
        try:
            return self.run_dir.is_dir() and not self.run_dir.is_symlink()
        except OSError:
            return False

    def _write_metadata(self, *, status: str, **extra: Any) -> None:
        payload = self._payload(status=status, **extra)
        _atomic_write_json(self.metadata_path, payload)

    def _write_control(self, *, status: str, **extra: Any) -> None:
        _atomic_write_json(self.control_path, self._payload(status=status, **extra))

    def _payload(self, *, status: str, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": LIVE_VIEW_SCHEMA_VERSION,
            "workflow": self.workflow,
            "scene_backend": "usd-cli",
            "asset": self.asset,
            "run_dir": str(self.run_dir),
            "status": status,
            "url": self.url,
            "http": {
                "bind_host": self.bind_host,
                "port": self.http_port,
            },
            "webrtc": {
                "public_host": self.public_host,
                "public_ip": self.webrtc_public_ip,
                "signaling_port": self.signaling_port,
                "media_port": self.media_port,
                "direct_url": self.webrtc_url,
                "browser_signaling": "same-origin-proxy",
                "codex_desktop_fallback": "same-origin-ovrtx-frame-mirror",
            },
            "viewer_pid": self.process.pid if self.process is not None else None,
            "viewer_log": str(self.log_path),
            "started_at": self.started_at,
            "retain_seconds": self.retain_seconds,
        }
        payload.update(extra)
        return payload

    def _validate_hosts(self) -> None:
        self.bind_host = self.bind_host.strip()
        if not self.bind_host:
            raise LiveViewError("--live-view-bind-host must not be empty.")
        if self.public_host is not None and not self.public_host.strip():
            raise LiveViewError("--live-view-public-host must not be empty.")

    def _terminate_process(self) -> None:
        process = self.process
        if process is None:
            return
        # A Popen that has already reaped its leader no longer proves ownership
        # of this numeric PID. Avoid signaling a process group that the OS may
        # since have assigned to an unrelated process.
        if process.poll() is not None:
            return
        try:
            process_group_id = os.getpgid(process.pid)
        except ProcessLookupError:
            process.poll()
            return
        except OSError as exc:
            raise LiveViewError(
                f"Unable to verify live viewer process group for {process.pid}."
            ) from exc
        if process_group_id != process.pid:
            raise LiveViewError(
                f"Live viewer leader {process.pid} no longer owns its process group."
            )
        try:
            os.killpg(process_group_id, signal.SIGTERM)
            process.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        kill_error: OSError | None = None
        if _process_group_exists(process_group_id):
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except OSError as exc:
                kill_error = exc
        if process.poll() is None:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
        deadline = time.monotonic() + 5.0
        while _process_group_exists(process_group_id):
            if time.monotonic() >= deadline:
                raise LiveViewError(
                    f"Live viewer process group {process_group_id} did not stop."
                ) from kill_error
            time.sleep(0.1)
        if process.poll() is None:
            raise LiveViewError(
                f"Live viewer leader {process.pid} remained live after group cleanup."
            )


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _usd_cli_executable(args: argparse.Namespace) -> Path:
    """Resolve the trusted in-repository usd-cli used by the workflow child."""

    from .runner import find_repo_root

    repo_root_value = getattr(args, "repo_root", None)
    repo_root = (
        Path(repo_root_value).expanduser().resolve()
        if repo_root_value is not None
        else find_repo_root()
    )
    candidate = repo_root / ".venv" / "bin" / "usd-cli"
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise LiveViewError(
            "Interactive USD live view requires the repository usd-cli launcher at "
            f"{candidate}. Run the repository setup first."
        )
    return candidate.resolve()


def _usd_cli_workflow_scope(workflow: str) -> str:
    """Map the public CLI command name to its workflow session scope."""

    return workflow.replace(".", "-")


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    output_argument = str(args.live_view_output_argument)
    current = getattr(args, output_argument, None)
    if current is None:
        root_value = os.getenv(VIEWER_RUN_ROOT_ENV)
        root = Path(root_value).expanduser() if root_value else Path.cwd() / "runs"
        stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S-%f")
        slug = _slug(_asset_label(args))
        workflow_slug = _slug(str(args.live_view_workflow).replace(".", "-"))
        current = root / f"{workflow_slug}-{slug}-{stamp}"
        setattr(args, output_argument, current)
    if not isinstance(current, Path):
        current = Path(current)
    resolved = current.expanduser().absolute()
    setattr(args, output_argument, resolved)
    return resolved


def _asset_label(args: argparse.Namespace) -> str:
    value = getattr(args, str(args.live_view_asset_argument), None)
    if isinstance(value, Path):
        return value.stem or value.name
    if value is not None:
        path = Path(str(value))
        return path.stem or str(value)
    return "USD asset"


def _viewer_command() -> tuple[list[str], Path | None]:
    configured = os.getenv(VIEWER_COMMAND_ENV)
    if configured:
        command = shlex.split(configured)
        if not command:
            raise LiveViewError(f"{VIEWER_COMMAND_ENV} is empty.")
        return command, None

    # This checkout-relative probe is intentionally only an internal fast path.
    # Installed and staged/public layouts use the configured command above or
    # the console-script lookup below when the sibling viewer package is absent.
    packages_root = Path(__file__).resolve().parents[2]
    viewer_root = packages_root / "content_workflow_viewer"
    viewer_python = viewer_root / ".venv" / "bin" / "python"
    if viewer_python.is_file():
        return [str(viewer_python), "-m", "content_workflow_viewer"], viewer_root

    executable = shutil.which("content-workflow-viewer")
    if executable:
        return [executable], None
    if not viewer_root.is_dir():
        raise LiveViewError(
            "The optional live viewer is not included in this checkout. "
            "Run without --live-view or install an internal viewer command and "
            f"set {VIEWER_COMMAND_ENV}."
        )
    raise LiveViewError(
        "The live viewer runtime is not installed. Run "
        f"{viewer_root / 'scripts' / 'setup.sh'} first, or set "
        f"{VIEWER_COMMAND_ENV}."
    )


def _wait_for_livez(
    *,
    host: str,
    port: int,
    timeout: float,
    process: subprocess.Popen[bytes],
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://{_url_host(host)}:{port}/livez"
    last_error = "viewer did not answer"
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise LiveViewError(
                f"Live viewer exited with code {returncode}; see {log_path}."
            )
        try:
            with urlopen(url, timeout=0.5) as response:  # noqa: S310 - local endpoint
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise LiveViewError(
        f"Live viewer did not become live within {timeout:g}s ({last_error}); "
        f"see {log_path}."
    )


def _available_port(bind_host: str) -> int:
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    probe_host = bind_host
    if bind_host in {"0.0.0.0", ""}:
        probe_host = "0.0.0.0"
    with socket.socket(family, socket.SOCK_STREAM) as candidate:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        candidate.bind((probe_host, 0))
        return int(candidate.getsockname()[1])


def _available_udp_port(bind_host: str) -> int:
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    probe_host = bind_host
    if bind_host in {"0.0.0.0", ""}:
        probe_host = "0.0.0.0"
    with socket.socket(family, socket.SOCK_DGRAM) as candidate:
        candidate.bind((probe_host, 0))
        return int(candidate.getsockname()[1])


def _default_public_host(bind_host: str) -> str:
    if bind_host in {"0.0.0.0", "::"}:
        return _discovered_public_ip()
    return bind_host


def _webrtc_public_ip(public_host: str) -> str:
    """Return a usable ICE candidate without widening the HTTP listener.

    OvStream's WebRTC backend does not establish a local connection when its
    advertised candidate is loopback. The signaling page can still remain on
    loopback; only the SDP candidate needs the host's routable interface.
    """

    normalized = public_host.strip("[]").lower()
    if (
        normalized == "localhost"
        or normalized == "::1"
        or normalized.startswith("127.")
    ):
        return _discovered_public_ip()
    return public_host


def _discovered_public_ip() -> str:
    override = os.getenv("OVSTREAM_PUBLIC_IP")
    if override:
        return override
    try:
        candidate = socket.gethostbyname(socket.gethostname())
    except OSError as exc:
        raise LiveViewError(
            "Unable to discover the OvStream WebRTC address; set "
            "OVSTREAM_PUBLIC_IP to a reachable interface address."
        ) from exc
    if not candidate or candidate.startswith("127.") or candidate == "0.0.0.0":
        raise LiveViewError(
            "Automatic OvStream WebRTC address discovery returned a loopback or "
            "unspecified address; set OVSTREAM_PUBLIC_IP to a reachable interface "
            "address."
        )
    return candidate


def _probe_host(bind_host: str) -> str:
    if bind_host == "0.0.0.0":
        return "127.0.0.1"
    if bind_host == "::":
        return "::1"
    return bind_host


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    directory_flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory_fd = os.open(path.parent, directory_flags)
    temporary_name: str | None = None
    temporary_fd: int | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        for _attempt in range(16):
            candidate = f".{path.name}.{secrets.token_hex(12)}.tmp"
            try:
                temporary_fd = os.open(
                    candidate,
                    file_flags,
                    0o600,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if temporary_fd is None or temporary_name is None:
            raise LiveViewError(f"Unable to create a secure temporary file for {path}")

        descriptor_stat = os.fstat(temporary_fd)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise LiveViewError("Live-view temporary artifact is not a regular file")
        temporary_identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        pending = memoryview(data)
        while pending:
            written = os.write(temporary_fd, pending)
            if written <= 0:
                raise OSError("live-view artifact write made no progress")
            pending = pending[written:]
        os.fsync(temporary_fd)
        path_stat = os.stat(
            temporary_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if temporary_identity != (path_stat.st_dev, path_stat.st_ino):
            raise LiveViewError("Live-view temporary artifact path was replaced")
        os.close(temporary_fd)
        temporary_fd = None
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = None
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_name is not None and temporary_identity is not None:
            try:
                current = os.stat(
                    temporary_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if temporary_identity == (current.st_dev, current.st_ino):
                    os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return normalized or "workflow"


def _duration_label(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:g} seconds"
    minutes = seconds / 60
    return f"{minutes:g} minutes"


def _port_number(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _env_port(name: str) -> int | None:
    value = os.getenv(name)
    return _port_number(value) if value else None


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()
