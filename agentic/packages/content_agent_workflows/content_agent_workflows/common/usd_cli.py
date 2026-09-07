# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorize and resolve the in-tree usd-cli package.

The normal tracked ``apps/usd_cli/`` source is the component boundary. The public
staging copy ships the reviewed component surface but omits internal engineering
material, and a similarly named executable on ``PATH`` is never enough to authorize a
workflow backend.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import sysconfig
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

from world_understanding.utils.windows_process import WindowsKillOnCloseJob

from .artifacts import contained_regular_file, read_contained_artifact

USD_CLI_SOURCE_PATH = "apps/usd_cli"
# Canonical editable-install guidance (PR #819 review item 1): the override is
# mandatory so `usd-exchange` stays the sole owner of the native `pxr`
# modules. The override ships with usd-cli so this guidance remains valid in
# both the development repository and its public staging copy.
USD_CLI_EDITABLE_INSTALL_COMMAND = (
    'uv pip install -e "apps/usd_cli[cli,server]" '
    "--overrides apps/usd_cli/requirements/usd-exchange-override.txt"
)
USD_CLI_REQUIRED_SOURCE = (
    Path("apps/usd_cli/pyproject.toml"),
    Path("apps/usd_cli/src/usd_cli/__init__.py"),
    Path("apps/usd_cli/src/usd_cli/main.py"),
    Path("apps/usd_cli/src/usd_core/__init__.py"),
    Path("apps/usd_cli/src/usd_telemetry/main.py"),
    Path("apps/usd_cli/src/usd_core/appearance.py"),
    Path("apps/usd_cli/src/usd_core/render/probe.py"),
    Path("apps/usd_cli/src/usd_server/__init__.py"),
)
USD_CLI_EXPECTED_ENTRY_POINTS = {
    "usd-cli": "usd_cli.main:run",
    "usd-cli-tel": "usd_telemetry.main:run",
}
USD_CLI_EXPECTED_MODULES = (
    "usd_cli",
    "usd_core",
    "usd_server",
    "usd_telemetry",
)
OVRTX_PROBE_SCHEMA_VERSION = "usd-cli.render-probe.v1"
MAX_CONSOLE_SCRIPT_BYTES = 1024 * 1024
MAX_LFS_POINTER_BYTES = 1024
MAX_COMMITTED_METADATA_BYTES = 64 * 1024
MAX_COMMITTED_STRUCTURE_BYTES = 16 * 1024 * 1024
MAX_COMMITTED_TREE_ENTRIES = 1_000_000
MAX_COMMITTED_PATH_BYTES = 4096
MAX_GIT_STDERR_BYTES = 16 * 1024
GIT_VERIFICATION_TIMEOUT_SECONDS = 30.0
DEFAULT_USD_CLI_STDOUT_BYTES = 16 * 1024 * 1024
DEFAULT_USD_CLI_STDERR_BYTES = 2 * 1024 * 1024
_GIT_VERIFICATION_OPTIONS = ("-c", "core.fsmonitor=false")
_REVIEWED_LFS_ATTRIBUTES = (
    "filter=lfs",
    "diff=lfs",
    "merge=lfs",
    "-text",
)
_REVIEWED_LFS_PATTERNS = frozenset(
    {
        "*.gif",
        "*.jpeg",
        "*.jpg",
        "*.mov",
        "*.png",
        "*.usd",
        "*.usdc",
        "*.usdz",
    }
)
_REVIEWED_PRUNABLE_RUNTIME_ROOTS = frozenset(
    {
        ".gstack",
        ".idea",
        ".ipynb_checkpoints",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".uv",
        ".venv",
        ".vscode",
        "ENV",
        "archive",
        "artifacts",
        "build",
        "cache",
        "dist",
        "env",
        "htmlcov",
        "node_modules",
        "output",
        "venv",
    }
)
_REVIEWED_STATE_DIRECTORIES = frozenset({".3dsc", ".ov", ".usd-cli"})
_REVIEWED_STATE_RUNTIME_ROOTS = frozenset({"checkpoints", "package_probe", "renders"})


@dataclass(frozen=True, slots=True)
class UsdCliPackageRoute:
    """Absolute package-owned launchers and their exact WU subtree revision."""

    wrapper: Path
    target: Path
    source_root: Path
    source_revision: str


@dataclass(frozen=True, slots=True)
class _AuthenticatedTreeEntry:
    """One strictly parsed entry from an authenticated raw Git tree."""

    mode: str
    name: str
    object_id: str

    @property
    def is_tree(self) -> bool:
        return self.mode == "40000"


class UsdCliSubprocessOutputError(RuntimeError):
    """Raised when a usd-cli subprocess exceeds a bounded output stream."""


_LINUX_PARENT_GUARD = """
import ctypes
import json
import os
import signal
import subprocess
import sys

parent_pid = int(sys.argv[1])
status_descriptor = int(sys.argv[2])
command = sys.argv[4:]
if os.getppid() != parent_pid:
    raise SystemExit(127)
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "could not set parent-death signal")
if os.getppid() != parent_pid:
    raise SystemExit(127)

def terminate_group(_signum, _frame):
    os.killpg(os.getpgrp(), signal.SIGKILL)

signal.signal(signal.SIGTERM, terminate_group)
try:
    child = subprocess.Popen(command)
except (IndexError, OSError, TypeError, ValueError) as exc:
    details = {
        "type": "OSError" if isinstance(exc, OSError) else type(exc).__name__,
        "args": list(exc.args),
    }
    if isinstance(exc, OSError):
        details.update(
            {
                "errno": exc.errno,
                "filename": exc.filename,
                "strerror": exc.strerror,
            }
        )
    os.write(
        status_descriptor,
        json.dumps(details).encode("utf-8"),
    )
    os.close(status_descriptor)
    raise SystemExit(127)
returncode = child.wait()
os.write(status_descriptor, json.dumps({"returncode": returncode}).encode("utf-8"))
os.close(status_descriptor)
# Issue group cleanup while this session leader still owns its PGID, so a
# recycled PID can never target an unrelated process group in the parent.
os.killpg(os.getpgrp(), signal.SIGKILL)
"""


def _bounded_usd_cli_command(
    command: list[str], *, status_descriptor: int
) -> list[str]:
    """Guard a Linux subprocess group against unexpected parent termination."""

    if sys.platform != "linux":
        return command
    return [
        sys.executable,
        "-c",
        _LINUX_PARENT_GUARD,
        str(os.getpid()),
        str(status_descriptor),
        "--",
        *command,
    ]


def _guarded_subprocess_result(
    descriptor: int,
) -> tuple[BaseException | None, int | None]:
    """Read a Linux parent-guard launch failure or child return code."""

    payload = bytearray()
    while len(payload) <= 64 * 1024:
        chunk = os.read(descriptor, 4096)
        if not chunk:
            break
        payload.extend(chunk)
    if len(payload) > 64 * 1024:
        raise RuntimeError("bounded usd-cli launch-error status is too large")
    if not payload:
        return None, None
    try:
        details = json.loads(payload)
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError("invalid bounded usd-cli launch-error status") from exc
    if not isinstance(details, dict):
        raise RuntimeError("invalid bounded usd-cli launch-error status")
    returncode = details.get("returncode")
    if returncode is not None:
        if not isinstance(returncode, int):
            raise RuntimeError("invalid bounded usd-cli launch-error status")
        return None, returncode
    error_type = details.get("type")
    arguments = details.get("args")
    if not isinstance(error_type, str) or not isinstance(arguments, list):
        raise RuntimeError("invalid bounded usd-cli launch-error status")
    if error_type == "OSError":
        error_number = details.get("errno")
        filename = details.get("filename")
        error_text = details.get("strerror")
        if (
            not isinstance(error_number, int)
            or not isinstance(filename, str)
            or not isinstance(error_text, str)
        ):
            raise RuntimeError("invalid bounded usd-cli launch-error status")
        return OSError(error_number, error_text, filename), None
    exception_types = {
        "IndexError": IndexError,
        "TypeError": TypeError,
        "ValueError": ValueError,
    }
    exception_type = exception_types.get(error_type)
    if exception_type is None or not all(isinstance(arg, str) for arg in arguments):
        raise RuntimeError("invalid bounded usd-cli launch-error status")
    return exception_type(*arguments), None


def _kill_bounded_usd_cli_process(
    process: subprocess.Popen[bytes], process_group_id: int | None
) -> None:
    """Kill a bounded invocation and every descendant in its dedicated session."""

    if process_group_id is not None:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
            return
        except OSError:
            # The process group may already be gone. Fall back to Popen's
            # process-specific kill when its leader is still available.
            pass
    if process.poll() is not None:
        return
    try:
        process.kill()
    except OSError:
        pass


def run_bounded_usd_cli_subprocess(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float,
    check: bool = False,
    stdout_limit: int = DEFAULT_USD_CLI_STDOUT_BYTES,
    stderr_limit: int = DEFAULT_USD_CLI_STDERR_BYTES,
) -> subprocess.CompletedProcess[str]:
    """Run a CLI while draining both output pipes into bounded memory."""

    if stdout_limit < 0 or stderr_limit < 0:
        raise ValueError("subprocess output limits must be non-negative")
    windows_job: WindowsKillOnCloseJob | None = None
    status_descriptor = -1
    status_writer = -1
    popen_kwargs: dict[str, object] = {
        "cwd": cwd,
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if os.name == "nt":
        # This runner launches only usd-cli, whose persistent daemon is the
        # explicitly trusted breakaway child; every other job caller defaults
        # to strict kill-on-close containment.
        windows_job = WindowsKillOnCloseJob(allow_breakaway=True)
        popen_kwargs["creationflags"] = windows_job.creation_flags
    else:
        # The bounded invocation can launch the telemetry wrapper, which may in
        # turn exec or spawn usd-cli.  Isolate the entire tree so the timeout
        # and output-limit paths can terminate all of it.
        popen_kwargs["start_new_session"] = True
        if sys.platform == "linux":
            status_descriptor, status_writer = os.pipe()
            popen_kwargs["pass_fds"] = (status_writer,)
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            _bounded_usd_cli_command(command, status_descriptor=status_writer),
            **popen_kwargs,
        )
        if status_writer >= 0:
            os.close(status_writer)
            status_writer = -1
        if windows_job is not None:
            windows_job.assign_process(process)
            windows_job.resume_process(process)
    except BaseException:
        if process is not None:
            # On Windows this process is still suspended when assignment to
            # the Job Object fails, so it is not covered by kill-on-close.
            # Terminate and reap it directly before releasing the job handle.
            try:
                if windows_job is not None:
                    windows_job.resume_process(process)
                process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if status_writer >= 0:
            os.close(status_writer)
        if status_descriptor >= 0:
            os.close(status_descriptor)
        if windows_job is not None:
            windows_job.close()
        raise
    assert process is not None
    process_group_id = process.pid if os.name == "posix" else None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    exceeded: set[str] = set()
    cancellation_requested = threading.Event()
    cleanup_lock = threading.Lock()
    cleanup_done = False

    def kill_process_tree() -> None:
        """Stop the leader and promptly release Windows job descendants."""

        nonlocal cleanup_done
        with cleanup_lock:
            if cleanup_done:
                return
            if process.poll() is not None:
                cleanup_done = True
                if windows_job is not None:
                    windows_job.terminate()
                    windows_job.close()
                return
            cleanup_done = True
            _kill_bounded_usd_cli_process(process, process_group_id)
            if windows_job is not None:
                # A descendant can retain an output pipe after its leader exits.
                # Releasing the kill-on-close Job Object before draining prevents
                # that pipe from holding the bounded operation open indefinitely.
                windows_job.terminate()
                windows_job.close()

    def drain(name: str, stream: object, limit: int) -> None:
        assert hasattr(stream, "read")
        while True:
            # BufferedReader.read() may wait to fill the requested size.  Use
            # read1 when available so an overflowing producer is cancelled as
            # soon as its already-piped output is observed.
            read = getattr(stream, "read1", stream.read)
            chunk = read(64 * 1024)
            if not chunk:
                return
            assert isinstance(chunk, bytes)
            remaining = max(0, limit - len(buffers[name]))
            if remaining:
                buffers[name].extend(chunk[:remaining])
            if len(chunk) > remaining:
                exceeded.add(name)
                cancellation_requested.set()
                kill_process_tree()

    assert process.stdout is not None
    assert process.stderr is not None
    threads = [
        threading.Thread(
            target=drain,
            args=("stdout", process.stdout, stdout_limit),
            daemon=True,
        ),
        threading.Thread(
            target=drain,
            args=("stderr", process.stderr, stderr_limit),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    timeout_error: subprocess.TimeoutExpired | None = None
    try:
        # Do not hold ``cleanup_lock`` across the full caller timeout.  A drain
        # thread needs it to stop a producer that has exceeded an output limit;
        # polling in small bounded intervals lets that cancellation win promptly.
        deadline = time.monotonic() + timeout
        while True:
            try:
                # Reaping and overflow cancellation share this lock.  In
                # particular, do not reap the leader after a drain thread has
                # observed it alive but before that thread can kill its process
                # group: a later killpg could otherwise target a recycled PGID.
                # The short wait keeps output-limit cancellation responsive.
                with cleanup_lock:
                    returncode = process.wait(
                        timeout=max(0.0, min(0.1, deadline - time.monotonic()))
                    )
                    needs_success_cleanup = not cleanup_done
                    cleanup_done = True
                    if windows_job is not None and needs_success_cleanup:
                        # The persistent daemon breaks away from this job.  Tear down
                        # the bounded runner job before pipe joins so a contained
                        # descendant cannot delay successful cleanup.
                        windows_job.terminate()
                        windows_job.close()
                break
            except subprocess.TimeoutExpired:
                # The drain thread publishes overflow before it contends for
                # cleanup_lock.  This prevents the short polling waiter from
                # repeatedly reacquiring the lock and starving cancellation.
                if cancellation_requested.is_set():
                    kill_process_tree()
                if time.monotonic() >= deadline:
                    raise
    except subprocess.TimeoutExpired as exc:
        kill_process_tree()
        process.wait()
        timeout_error = exc
    except BaseException:
        kill_process_tree()
        process.wait(timeout=5)
        raise
    finally:
        for thread in threads:
            thread.join(timeout=5)
        if not any(thread.is_alive() for thread in threads):
            process.stdout.close()
            process.stderr.close()
        if windows_job is not None:
            windows_job.close()
        if status_descriptor >= 0:
            try:
                launch_error, guarded_returncode = _guarded_subprocess_result(
                    status_descriptor
                )
            finally:
                os.close(status_descriptor)
            if launch_error is not None:
                raise launch_error
            if guarded_returncode is not None:
                returncode = guarded_returncode

    stdout = bytes(buffers["stdout"]).decode("utf-8", errors="replace")
    stderr = bytes(buffers["stderr"]).decode("utf-8", errors="replace")
    if timeout_error is not None:
        raise subprocess.TimeoutExpired(
            command, timeout, output=stdout, stderr=stderr
        ) from timeout_error
    if exceeded:
        limits = ", ".join(
            f"{name}={stdout_limit if name == 'stdout' else stderr_limit} bytes"
            for name in sorted(exceeded)
        )
        raise UsdCliSubprocessOutputError(
            f"usd-cli subprocess output exceeded its limit ({limits})"
        )
    completed = subprocess.CompletedProcess(command, returncode, stdout, stderr)
    if check and returncode:
        raise subprocess.CalledProcessError(
            returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return completed


def usd_cli_source_distributed(repo_root: Path) -> bool:
    """Return whether this checkout carries workflow-capable in-tree source.

    This deliberately checks the source boundary, not ``PATH``.  Git/index and
    cleanliness verification is performed by :func:`usd_cli_source_revision`
    before a route is authorized.
    """

    root = repo_root.expanduser().resolve()
    try:
        if os.path.lexists(root / ".gitmodules"):
            return False
        source_root = root / USD_CLI_SOURCE_PATH
        if os.path.lexists(source_root / ".git"):
            return False
        for relative in USD_CLI_REQUIRED_SOURCE:
            contained_regular_file(root, root / relative)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _authenticated_git_object(
    *,
    root: Path,
    git_environment: dict[str, str],
    object_id: str,
    object_format: str,
    expected_type: str,
    capture_limit: int,
) -> tuple[int, str, bytes | None]:
    """Read and authenticate one raw Git object with bounded pipe handling."""

    if expected_type not in {"blob", "commit", "tree"}:
        raise RuntimeError(f"unsupported Git object type: {expected_type!r}")
    if capture_limit < 0:
        raise RuntimeError("Git object capture limit must be non-negative")
    if object_format == "sha1":
        git_digest = hashlib.sha1(usedforsecurity=False)
    elif object_format == "sha256":
        git_digest = hashlib.sha256()
    else:
        raise RuntimeError(f"unsupported Git object format: {object_format!r}")
    if re.fullmatch(rf"[0-9a-f]{{{git_digest.digest_size * 2}}}", object_id) is None:
        raise RuntimeError(f"invalid committed Git object identity: {object_id!r}")

    command = git_verification_command("cat-file", "--batch")
    deadline = time.monotonic() + GIT_VERIFICATION_TIMEOUT_SECONDS
    process: subprocess.Popen[bytes] | None = None

    def stop_started_process() -> None:
        if process is None:
            return
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=git_environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        active_process = process
        assert active_process.stdin is not None
        assert active_process.stdout is not None
        assert active_process.stderr is not None
        active_process.stdin.write(f"{object_id}\n".encode("ascii"))
        active_process.stdin.close()

        result: dict[str, int | str | bytes | None] = {}
        stderr_buffer = bytearray()
        stream_errors: list[Exception] = []

        def record_stream_error(exc: Exception) -> None:
            stream_errors.append(exc)
            try:
                active_process.kill()
            except OSError:
                pass

        def drain_stdout() -> None:
            try:
                header = active_process.stdout.readline(1025)
                if len(header) > 1024 or not header.endswith(b"\n"):
                    raise RuntimeError("Git returned an invalid cat-file header")
                try:
                    header_oid, object_type, size_text = (
                        header[:-1].decode("ascii").split()
                    )
                except (UnicodeDecodeError, ValueError) as exc:
                    raise RuntimeError(
                        "Git returned an invalid cat-file header"
                    ) from exc
                if (
                    header_oid.lower() != object_id
                    or object_type != expected_type
                    or re.fullmatch(r"0|[1-9][0-9]*", size_text) is None
                ):
                    raise RuntimeError("Git returned an invalid cat-file object header")
                expected_size = int(size_text)
                git_digest.update(f"{expected_type} {expected_size}\0".encode("ascii"))
                content_digest = hashlib.sha256()
                content_size = 0
                retained_content: bytearray | None = bytearray()
                while content_size < expected_size:
                    chunk = active_process.stdout.read(
                        min(64 * 1024, expected_size - content_size)
                    )
                    if not chunk:
                        raise RuntimeError("Git returned a truncated cat-file blob")
                    content_size += len(chunk)
                    git_digest.update(chunk)
                    content_digest.update(chunk)
                    if retained_content is not None:
                        if len(retained_content) + len(chunk) <= capture_limit:
                            retained_content.extend(chunk)
                        else:
                            retained_content = None
                if active_process.stdout.read(1) != b"\n":
                    raise RuntimeError(
                        "Git returned an invalid cat-file object trailer"
                    )
                trailing = active_process.stdout.read(1)
                if trailing:
                    raise RuntimeError("Git returned trailing cat-file output")
                result.update(
                    {
                        "size": content_size,
                        "sha256": content_digest.hexdigest(),
                        "content": (
                            bytes(retained_content)
                            if retained_content is not None
                            else None
                        ),
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive pipe guard
                record_stream_error(exc)

        def drain_stderr() -> None:
            try:
                while True:
                    chunk = active_process.stderr.read(64 * 1024)
                    if not chunk:
                        return
                    remaining = max(0, MAX_GIT_STDERR_BYTES - len(stderr_buffer))
                    if remaining:
                        stderr_buffer.extend(chunk[:remaining])
            except Exception as exc:  # pragma: no cover - defensive pipe guard
                record_stream_error(exc)

        threads = [
            threading.Thread(target=drain_stdout, daemon=True),
            threading.Thread(target=drain_stderr, daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    command, GIT_VERIFICATION_TIMEOUT_SECONDS
                )
            returncode = active_process.wait(timeout=remaining)
            for thread in threads:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(
                        command, GIT_VERIFICATION_TIMEOUT_SECONDS
                    )
                thread.join(timeout=remaining)
            if any(thread.is_alive() for thread in threads):
                raise subprocess.TimeoutExpired(
                    command,
                    GIT_VERIFICATION_TIMEOUT_SECONDS,
                )
        except Exception:
            if active_process.poll() is None:
                active_process.kill()
            active_process.wait()
            for thread in threads:
                thread.join(timeout=1.0)
            raise
        finally:
            active_process.stdout.close()
            active_process.stderr.close()

        stderr = bytes(stderr_buffer).decode("utf-8", errors="replace").strip()
        if stream_errors:
            raise RuntimeError(
                f"could not drain committed usd-cli {expected_type} {object_id}"
            ) from stream_errors[0]
        if returncode != 0:
            raise RuntimeError(
                f"could not read committed usd-cli {expected_type} "
                f"{object_id}: {stderr}"
            )
        if git_digest.hexdigest() != object_id:
            raise RuntimeError(
                f"committed usd-cli {expected_type} {object_id} does not match "
                "its Git identity"
            )
    except subprocess.TimeoutExpired as exc:
        stop_started_process()
        raise RuntimeError(
            f"timed out reading committed usd-cli {expected_type} {object_id}"
        ) from exc
    except (OSError, ValueError) as exc:
        stop_started_process()
        raise RuntimeError(
            f"could not read committed usd-cli {expected_type} {object_id}"
        ) from exc
    size = result.get("size")
    sha256 = result.get("sha256")
    content = result.get("content")
    if (
        not isinstance(size, int)
        or not isinstance(sha256, str)
        or (content is not None and not isinstance(content, bytes))
    ):
        raise RuntimeError(
            f"could not read committed usd-cli {expected_type} {object_id}"
        )
    return (
        size,
        sha256,
        content,
    )


def _committed_blob_identity(
    *,
    root: Path,
    git_environment: dict[str, str],
    object_id: str,
    object_format: str,
) -> tuple[int, str, bytes | None]:
    """Read and authenticate one raw Git blob."""

    return _authenticated_git_object(
        root=root,
        git_environment=git_environment,
        object_id=object_id,
        object_format=object_format,
        expected_type="blob",
        capture_limit=MAX_COMMITTED_METADATA_BYTES,
    )


def _git_object_id_length(object_format: str) -> int:
    """Return the hexadecimal object-id width for one repository format."""

    if object_format == "sha1":
        return 40
    if object_format == "sha256":
        return 64
    raise RuntimeError(f"unsupported Git object format: {object_format!r}")


def _authenticated_structure_content(
    *,
    root: Path,
    git_environment: dict[str, str],
    object_id: str,
    object_format: str,
    expected_type: str,
) -> bytes:
    """Return one authenticated bounded commit/tree payload."""

    size, _sha256, content = _authenticated_git_object(
        root=root,
        git_environment=git_environment,
        object_id=object_id,
        object_format=object_format,
        expected_type=expected_type,
        capture_limit=MAX_COMMITTED_STRUCTURE_BYTES,
    )
    if content is None or len(content) != size:
        raise RuntimeError(
            f"committed usd-cli {expected_type} {object_id} exceeds the "
            "authenticated structure size limit"
        )
    return content


def _parse_authenticated_commit_tree(
    content: bytes,
    *,
    object_format: str,
) -> str:
    """Extract the unique root tree from authenticated raw commit bytes."""

    header_end = content.find(b"\n\n")
    if header_end <= 0:
        raise RuntimeError("the captured HEAD commit has malformed headers")
    header = content[:header_end]
    if b"\0" in header or b"\r" in header:
        raise RuntimeError("the captured HEAD commit has unsafe headers")
    lines = header.split(b"\n")
    tree_ids: list[str] = []
    previous_key: bytes | None = None
    expected_oid_length = _git_object_id_length(object_format)
    for index, line in enumerate(lines):
        if not line:
            raise RuntimeError("the captured HEAD commit has malformed headers")
        if line.startswith(b" "):
            if previous_key is None or previous_key == b"tree":
                raise RuntimeError("the captured HEAD commit has malformed headers")
            continue
        try:
            key, value = line.split(b" ", maxsplit=1)
        except ValueError as exc:
            raise RuntimeError(
                "the captured HEAD commit has malformed headers"
            ) from exc
        if re.fullmatch(rb"[a-z][a-z0-9-]*", key) is None or not value:
            raise RuntimeError("the captured HEAD commit has malformed headers")
        if index == 0 and key != b"tree":
            raise RuntimeError("the captured HEAD commit does not begin with its tree")
        if key == b"tree":
            if (
                re.fullmatch(
                    rb"[0-9a-f]{" + str(expected_oid_length).encode("ascii") + rb"}",
                    value,
                )
                is None
            ):
                raise RuntimeError(
                    "the captured HEAD commit has an invalid tree identity"
                )
            tree_ids.append(value.decode("ascii"))
        previous_key = key
    if len(tree_ids) != 1:
        raise RuntimeError(
            "the captured HEAD commit must contain exactly one root tree"
        )
    return tree_ids[0]


def _parse_authenticated_tree(
    content: bytes,
    *,
    object_format: str,
) -> tuple[_AuthenticatedTreeEntry, ...]:
    """Parse one authenticated raw Git tree with canonical ordering."""

    raw_oid_bytes = _git_object_id_length(object_format) // 2
    entries: list[_AuthenticatedTreeEntry] = []
    seen_names: set[bytes] = set()
    previous_sort_key: bytes | None = None
    offset = 0
    canonical_modes = {b"100644", b"100755", b"120000", b"160000", b"40000"}
    while offset < len(content):
        mode_end = content.find(b" ", offset)
        if mode_end <= offset:
            raise RuntimeError("a committed Git tree has a malformed entry mode")
        mode = content[offset:mode_end]
        if mode not in canonical_modes:
            raise RuntimeError("a committed Git tree has a non-canonical entry mode")
        name_end = content.find(b"\0", mode_end + 1)
        if name_end < 0:
            raise RuntimeError("a committed Git tree has an unterminated entry name")
        raw_name = content[mode_end + 1 : name_end]
        object_end = name_end + 1 + raw_oid_bytes
        if object_end > len(content):
            raise RuntimeError("a committed Git tree has a truncated object identity")
        raw_object_id = content[name_end + 1 : object_end]
        offset = object_end
        if (
            not raw_name
            or raw_name in {b".", b".."}
            or b"/" in raw_name
            or b"\\" in raw_name
            or any(value < 0x20 or value == 0x7F for value in raw_name)
        ):
            raise RuntimeError("a committed Git tree has an unsafe entry name")
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                "a committed Git tree entry name is not valid UTF-8"
            ) from exc
        if name.casefold() == ".git":
            raise RuntimeError("a committed Git tree has an unsafe .git entry")
        if raw_name in seen_names:
            raise RuntimeError("a committed Git tree has duplicate entry names")
        seen_names.add(raw_name)
        sort_key = raw_name + (b"/" if mode == b"40000" else b"\0")
        if previous_sort_key is not None and sort_key <= previous_sort_key:
            raise RuntimeError("a committed Git tree has incorrectly ordered entries")
        previous_sort_key = sort_key
        if not any(raw_object_id):
            raise RuntimeError("a committed Git tree has a null object identity")
        entries.append(
            _AuthenticatedTreeEntry(
                mode=mode.decode("ascii"),
                name=name,
                object_id=raw_object_id.hex(),
            )
        )
    return tuple(entries)


def _authenticated_usd_cli_snapshot(
    *,
    root: Path,
    git_environment: dict[str, str],
    head_object_id: str,
    object_format: str,
) -> tuple[str, dict[str, tuple[str, str]]]:
    """Derive the committed usd-cli map only from one authenticated HEAD."""

    commit_content = _authenticated_structure_content(
        root=root,
        git_environment=git_environment,
        object_id=head_object_id,
        object_format=object_format,
        expected_type="commit",
    )
    root_tree_id = _parse_authenticated_commit_tree(
        commit_content,
        object_format=object_format,
    )
    parsed_trees: dict[str, tuple[_AuthenticatedTreeEntry, ...]] = {}

    def read_tree(tree_object_id: str) -> tuple[_AuthenticatedTreeEntry, ...]:
        cached = parsed_trees.get(tree_object_id)
        if cached is not None:
            return cached
        tree_content = _authenticated_structure_content(
            root=root,
            git_environment=git_environment,
            object_id=tree_object_id,
            object_format=object_format,
            expected_type="tree",
        )
        parsed = _parse_authenticated_tree(
            tree_content,
            object_format=object_format,
        )
        parsed_trees[tree_object_id] = parsed
        return parsed

    root_entries = read_tree(root_tree_id)
    if any(entry.name == ".gitmodules" for entry in root_entries):
        raise RuntimeError("normal in-tree usd-cli source forbids .gitmodules")

    source_tree_id = root_tree_id
    source_ancestors = [root_tree_id]
    for component in Path(USD_CLI_SOURCE_PATH).parts:
        source_entry = next(
            (entry for entry in read_tree(source_tree_id) if entry.name == component),
            None,
        )
        if source_entry is None or not source_entry.is_tree:
            raise RuntimeError("the captured HEAD has no ordinary usd-cli source tree")
        source_tree_id = source_entry.object_id
        source_ancestors.append(source_tree_id)
    source_revision = source_tree_id
    committed_entries: dict[str, tuple[str, str]] = {}
    traversed_entries = 0
    stack: list[tuple[str, str, tuple[str, ...]]] = [
        (USD_CLI_SOURCE_PATH, source_revision, tuple(source_ancestors[:-1]))
    ]
    while stack:
        prefix, tree_object_id, ancestors = stack.pop()
        if tree_object_id in ancestors:
            raise RuntimeError("the committed usd-cli tree contains a cycle")
        next_ancestors = (*ancestors, tree_object_id)
        for entry in reversed(read_tree(tree_object_id)):
            traversed_entries += 1
            if traversed_entries > MAX_COMMITTED_TREE_ENTRIES:
                raise RuntimeError("the committed usd-cli tree is too large to verify")
            relative = f"{prefix}/{entry.name}"
            if len(relative.encode("utf-8")) > MAX_COMMITTED_PATH_BYTES:
                raise RuntimeError("the committed usd-cli tree has an unsafe long path")
            if entry.is_tree:
                stack.append((relative, entry.object_id, next_ancestors))
                continue
            if entry.mode not in {"100644", "100755"}:
                raise RuntimeError("usd-cli must contain only ordinary committed files")
            if relative in committed_entries:
                raise RuntimeError("the committed usd-cli tree has duplicate paths")
            committed_entries[relative] = (entry.mode, entry.object_id)
    return source_revision, committed_entries


def _lfs_payload_identity(pointer: bytes | None) -> tuple[str, int] | None:
    """Parse the canonical three-line Git LFS pointer used by the import."""

    if pointer is None:
        return None
    match = re.fullmatch(
        rb"version https://git-lfs\.github\.com/spec/v1\n"
        rb"oid sha256:([0-9a-f]{64})\n"
        rb"size (0|[1-9][0-9]*)\n",
        pointer,
    )
    if match is None:
        return None
    return match.group(1).decode("ascii"), int(match.group(2))


def _reviewed_lfs_patterns(
    *,
    root: Path,
    tracked_entries: dict[str, tuple[str, str]],
    git_environment: dict[str, str],
    object_format: str,
    committed_blob_identities: dict[str, tuple[int, str, bytes | None]],
) -> frozenset[str]:
    """Return exact LFS patterns from the committed package attributes blob."""

    attributes_path = f"{USD_CLI_SOURCE_PATH}/.gitattributes"
    nested_attributes = sorted(
        path
        for path in tracked_entries
        if path != attributes_path and path.endswith("/.gitattributes")
    )
    if nested_attributes:
        raise RuntimeError(
            "usd-cli contains an unreviewed nested attribute program: "
            + ", ".join(nested_attributes[:5])
        )
    attributes_entry = tracked_entries.get(attributes_path)
    if attributes_entry is None:
        return frozenset()
    attributes_mode, attributes_oid = attributes_entry
    if attributes_mode != "100644":
        raise RuntimeError("apps/usd_cli/.gitattributes must be a non-executable file")
    committed_identity = committed_blob_identities.get(attributes_oid)
    if committed_identity is None:
        committed_identity = _committed_blob_identity(
            root=root,
            git_environment=git_environment,
            object_id=attributes_oid,
            object_format=object_format,
        )
        committed_blob_identities[attributes_oid] = committed_identity
    attributes_size, _attributes_sha256, attributes_content = committed_identity
    if attributes_content is None or len(attributes_content) != attributes_size:
        raise RuntimeError("apps/usd_cli/.gitattributes exceeds the review size limit")
    try:
        lines = attributes_content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError("apps/usd_cli/.gitattributes is not valid UTF-8") from exc

    patterns: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = tuple(stripped.split())
        if (
            len(fields) != 1 + len(_REVIEWED_LFS_ATTRIBUTES)
            or fields[0] not in _REVIEWED_LFS_PATTERNS
            or fields[1:] != _REVIEWED_LFS_ATTRIBUTES
            or fields[0] in patterns
        ):
            raise RuntimeError(
                "apps/usd_cli/.gitattributes contains an unreviewed attribute pattern"
            )
        patterns.add(fields[0])
    return frozenset(patterns)


def _lfs_pattern_for_path(relative: str, patterns: frozenset[str]) -> str | None:
    """Return the exact reviewed basename pattern for one package path."""

    name = Path(relative).name
    for pattern in patterns:
        if name.endswith(pattern.removeprefix("*")):
            return pattern
    return None


def _reviewed_prunable_runtime_root(relative: str) -> bool:
    """Return whether a directory is one exact reviewed runtime-only root."""

    try:
        source_parts = Path(relative).relative_to(USD_CLI_SOURCE_PATH).parts
    except ValueError:
        return False
    if len(source_parts) == 1:
        return source_parts[0] in _REVIEWED_PRUNABLE_RUNTIME_ROOTS
    if len(source_parts) == 2:
        if (
            source_parts[0] in _REVIEWED_STATE_DIRECTORIES
            and source_parts[1] in _REVIEWED_STATE_RUNTIME_ROOTS
        ):
            return True
        return source_parts[0] == "example_tasks" and source_parts[1].startswith(
            "runs-"
        )
    if len(source_parts) == 3:
        return source_parts[:2] == ("internal", "example_tasks") and source_parts[
            2
        ].startswith("runs-")
    return False


def _reviewed_runtime_file(relative: str) -> bool:
    """Return whether one file has a reviewed, non-import runtime location."""

    try:
        source_parts = Path(relative).relative_to(USD_CLI_SOURCE_PATH).parts
    except ValueError:
        return False
    if not source_parts:
        return False
    name = source_parts[-1]
    if "__pycache__" in source_parts:
        # Workflow launches redirect every Python cache read and disable
        # bytecode writes. Only a direct PEP 3147 cache child is safe; this
        # covers the package, its OVRTX service app, and tests run before a
        # workflow. Every other ignored file remains unauthorized.
        return (
            len(source_parts) >= 3
            and source_parts.count("__pycache__") == 1
            and source_parts[-2] == "__pycache__"
            and name.endswith(".pyc")
        )
    if (
        len(source_parts) >= 3
        and source_parts[0] == "example_tasks"
        and source_parts[1] == "runs"
    ):
        return True
    if len(source_parts) >= 4 and source_parts[:3] == (
        "internal",
        "example_tasks",
        "runs",
    ):
        return True
    if len(source_parts) == 2 and source_parts[0] in _REVIEWED_STATE_DIRECTORIES:
        return name in {
            "audit.jsonl",
            "config.toml",
            "daemon.log",
            "daemon.pids",
            "server.json",
            "server.json.lock",
        } or name.endswith((".usd", ".usda", ".usdc"))
    return (
        (len(source_parts) == 1 and name in {".DS_Store", "horde_ovrtx_instances.json"})
        or (len(source_parts) == 1 and name.startswith(".coverage"))
        or (
            name.startswith((".3dsc_render_", ".ov_render_", ".usd-cli_render_"))
            and name.endswith((".usd", ".usda", ".usdc"))
        )
    )


def _git_path_is_ignored(
    *,
    root: Path,
    git_environment: dict[str, str],
    relative: str,
) -> bool:
    """Check a reviewed runtime path against the checkout's ignore rules."""

    try:
        result = subprocess.run(
            git_verification_command("check-ignore", "--quiet", "--", relative),
            cwd=root,
            env=git_environment,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=GIT_VERIFICATION_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not verify the ignore status of {relative}") from exc
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise RuntimeError(f"could not verify the ignore status of {relative}")


def _verify_source_files_match_index(
    *,
    root: Path,
    source_root: Path,
    tracked_entries: dict[str, tuple[str, str]],
    git_environment: dict[str, str],
    object_format: str,
) -> None:
    """Compare raw files to index blobs without running Git clean filters."""

    filesystem_paths: set[str] = set()
    committed_blob_identities: dict[str, tuple[int, str, bytes | None]] = {}
    lfs_patterns = _reviewed_lfs_patterns(
        root=root,
        tracked_entries=tracked_entries,
        git_environment=git_environment,
        object_format=object_format,
        committed_blob_identities=committed_blob_identities,
    )
    try:
        for directory, directory_names, file_names in os.walk(
            source_root,
            topdown=True,
            followlinks=False,
        ):
            directory_path = Path(directory)
            retained_directories: list[str] = []
            for name in directory_names:
                candidate = directory_path / name
                relative = candidate.relative_to(root).as_posix()
                metadata = candidate.lstat()
                if not stat.S_ISDIR(metadata.st_mode):
                    raise RuntimeError(
                        "the in-tree usd-cli source contains a symlink or special "
                        f"directory entry: {candidate}"
                    )
                tracked_below = any(
                    tracked.startswith(f"{relative}/") for tracked in tracked_entries
                )
                if _reviewed_prunable_runtime_root(relative) and not tracked_below:
                    if not _git_path_is_ignored(
                        root=root,
                        git_environment=git_environment,
                        relative=relative,
                    ):
                        raise RuntimeError(
                            "the in-tree usd-cli runtime directory is not ignored: "
                            f"{relative}"
                        )
                    # The reviewed root itself is a real directory. Its internal
                    # virtual-environment and tool-cache symlinks cannot affect
                    # package imports and need not be followed or rejected.
                    continue
                retained_directories.append(name)
            directory_names[:] = retained_directories
            for name in file_names:
                candidate = directory_path / name
                relative = candidate.relative_to(root).as_posix()
                expected = tracked_entries.get(relative)
                metadata = candidate.lstat()
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise RuntimeError(
                        "the in-tree usd-cli source must contain only single-link "
                        f"regular files: {relative}"
                    )
                if expected is None:
                    if _reviewed_runtime_file(relative) and _git_path_is_ignored(
                        root=root,
                        git_environment=git_environment,
                        relative=relative,
                    ):
                        continue
                    raise RuntimeError(
                        "the in-tree usd-cli source contains unrecorded changes "
                        "(ignored or untracked artifacts): "
                        f"{relative}"
                    )
                filesystem_paths.add(relative)
                expected_mode, expected_oid = expected
                if os.name == "posix" and bool(metadata.st_mode & 0o111) != (
                    expected_mode == "100755"
                ):
                    raise RuntimeError(
                        "the in-tree usd-cli source contains an unrecorded mode "
                        f"change: {relative}"
                    )
                committed_identity = committed_blob_identities.get(expected_oid)
                if committed_identity is None:
                    committed_identity = _committed_blob_identity(
                        root=root,
                        git_environment=git_environment,
                        object_id=expected_oid,
                        object_format=object_format,
                    )
                    committed_blob_identities[expected_oid] = committed_identity
                expected_size, expected_sha256, committed_content = committed_identity
                pointer = (
                    committed_content
                    if committed_content is not None
                    and len(committed_content) <= MAX_LFS_POINTER_BYTES
                    else None
                )
                lfs_identity = _lfs_payload_identity(pointer)
                lfs_pattern = _lfs_pattern_for_path(relative, lfs_patterns)
                authorized_lfs_identity: tuple[str, int] | None = None
                if lfs_identity is not None:
                    if expected_mode == "100755":
                        raise RuntimeError(
                            "the in-tree usd-cli source contains an executable "
                            f"LFS artifact: {relative}"
                        )
                    if relative.startswith(f"{USD_CLI_SOURCE_PATH}/src/"):
                        raise RuntimeError(
                            "the in-tree usd-cli import source contains an LFS "
                            f"artifact: {relative}"
                        )
                    if lfs_pattern is not None:
                        authorized_lfs_identity = lfs_identity
                max_worktree_bytes = max(
                    expected_size,
                    (
                        authorized_lfs_identity[1]
                        if authorized_lfs_identity is not None
                        else 0
                    ),
                )
                if os.name == "nt" and authorized_lfs_identity is None:
                    # Git for Windows may materialize an ordinary LF blob with
                    # CRLF line endings under core.autocrlf.  Bound that one
                    # reversible transformation without trusting Git filters.
                    max_worktree_bytes = max(
                        max_worktree_bytes,
                        expected_size * 2,
                    )
                allowed_worktree_sizes = {expected_size}
                if authorized_lfs_identity is not None:
                    allowed_worktree_sizes.add(authorized_lfs_identity[1])
                if metadata.st_size not in allowed_worktree_sizes and not (
                    os.name == "nt"
                    and authorized_lfs_identity is None
                    and metadata.st_size <= max_worktree_bytes
                ):
                    # The descriptor-safe reader's byte ceiling is a resource
                    # bound, not the source-integrity diagnostic. Detect a
                    # size change before applying that ceiling so a grown
                    # tracked file reports the same actionable dirty-source
                    # error as an equal-size content change.
                    raise RuntimeError(
                        "the in-tree usd-cli source contains unrecorded changes: "
                        f"{relative}"
                    )
                artifact = read_contained_artifact(
                    source_root,
                    candidate,
                    max_bytes=max_worktree_bytes,
                )
                exact_match = (
                    artifact.size_bytes == expected_size
                    and artifact.sha256 == expected_sha256
                )
                smudged_lfs_match = authorized_lfs_identity == (
                    artifact.sha256,
                    artifact.size_bytes,
                )
                crlf_match = False
                if (
                    os.name == "nt"
                    and not exact_match
                    and authorized_lfs_identity is None
                ):
                    captured = read_contained_artifact(
                        source_root,
                        candidate,
                        max_bytes=max_worktree_bytes,
                        capture_bytes=True,
                    )
                    assert captured.data is not None
                    normalized = captured.data.replace(b"\r\n", b"\n")
                    crlf_match = (
                        b"\r" not in normalized
                        and len(normalized) == expected_size
                        and hashlib.sha256(normalized).hexdigest() == expected_sha256
                    )
                if not exact_match and not smudged_lfs_match and not crlf_match:
                    raise RuntimeError(
                        "the in-tree usd-cli source contains unrecorded changes: "
                        f"{relative}"
                    )
    except (OSError, ValueError) as exc:
        raise RuntimeError("could not inspect the in-tree usd-cli source") from exc

    missing = sorted(set(tracked_entries) - filesystem_paths)
    if missing:
        raise RuntimeError(
            "the in-tree usd-cli source is missing tracked files: "
            + ", ".join(missing[:5])
        )


def usd_cli_source_revision(repo_root: Path, *, require_clean: bool = True) -> str:
    """Return the committed WU ``usd-cli`` subtree identity or fail closed.

    Every capability file must be ordinary source tracked by the parent WU
    repository.  Runtime execution rejects index tricks and source changes by
    default so evidence never claims a reproducible subtree while running
    unrecorded code. Internal import provenance is not part of the distributed
    usd-cli runtime boundary.
    """

    root = repo_root.expanduser().resolve(strict=True)
    source_candidate = root / USD_CLI_SOURCE_PATH
    if source_candidate.is_symlink():
        raise RuntimeError("the in-tree usd-cli source root must not be a symlink")
    source_root = source_candidate.resolve(strict=True)
    try:
        source_root.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("the usd-cli source root escapes the repository") from exc
    if os.path.lexists(source_root / ".git"):
        raise RuntimeError("the in-tree usd-cli source contains nested Git metadata")
    git_environment = sanitized_git_verification_env(trusted_repository=root)
    try:
        top_level = subprocess.run(
            git_verification_command("rev-parse", "--show-toplevel"),
            cwd=root,
            env=git_environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
        ).stdout.strip()
        if Path(top_level).resolve() != root:
            raise RuntimeError("repo_root is not the WU Git worktree root")
        object_format = subprocess.run(
            git_verification_command("rev-parse", "--show-object-format"),
            cwd=root,
            env=git_environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=GIT_VERIFICATION_TIMEOUT_SECONDS,
        ).stdout.strip()
        if object_format not in {"sha1", "sha256"}:
            raise RuntimeError(f"unsupported Git object format: {object_format!r}")
        head_object_id = (
            subprocess.run(
                git_verification_command("rev-parse", "--verify", "HEAD"),
                cwd=root,
                env=git_environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=GIT_VERIFICATION_TIMEOUT_SECONDS,
            )
            .stdout.strip()
            .lower()
        )
        if (
            re.fullmatch(
                rf"[0-9a-f]{{{_git_object_id_length(object_format)}}}",
                head_object_id,
            )
            is None
        ):
            raise RuntimeError("could not capture the committed WU HEAD identity")
        source_revision, committed_entries = _authenticated_usd_cli_snapshot(
            root=root,
            git_environment=git_environment,
            head_object_id=head_object_id,
            object_format=object_format,
        )

        tracked_gitmodules = subprocess.run(
            git_verification_command(
                "ls-files",
                "--stage",
                "--",
                ".gitmodules",
            ),
            cwd=root,
            env=git_environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
        ).stdout.strip()
        if tracked_gitmodules or os.path.lexists(root / ".gitmodules"):
            raise RuntimeError("normal in-tree usd-cli source forbids .gitmodules")

        staged = subprocess.run(
            git_verification_command(
                "ls-files",
                "--stage",
                "-z",
                "--",
                USD_CLI_SOURCE_PATH,
            ),
            cwd=root,
            env=git_environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
        ).stdout
        tracked_entries: dict[str, tuple[str, str]] = {}
        entries = [entry for entry in staged.split("\0") if entry]
        if not entries:
            raise RuntimeError("usd-cli source is not tracked by world-understanding")
        for entry in entries:
            try:
                metadata, path = entry.split("\t", maxsplit=1)
                mode, object_id, stage = metadata.split()
            except ValueError as exc:
                raise RuntimeError("usd-cli has an invalid parent index entry") from exc
            if mode not in {"100644", "100755"} or stage != "0":
                raise RuntimeError(
                    "usd-cli must contain only ordinary stage-0 tracked files"
                )
            normalized_object_id = object_id.lower()
            if (
                path in tracked_entries
                or re.fullmatch(
                    rf"[0-9a-f]{{{_git_object_id_length(object_format)}}}",
                    normalized_object_id,
                )
                is None
            ):
                raise RuntimeError("usd-cli has an invalid parent index entry")
            tracked_entries[path] = (mode, normalized_object_id)
        missing = [
            relative.as_posix()
            for relative in USD_CLI_REQUIRED_SOURCE
            if relative.as_posix() not in tracked_entries
        ]
        if missing:
            raise RuntimeError(
                "required usd-cli source is absent from the parent index: "
                + ", ".join(missing)
            )

        if require_clean:
            index_states = subprocess.run(
                git_verification_command(
                    "ls-files",
                    "-v",
                    "-z",
                    "--",
                    USD_CLI_SOURCE_PATH,
                ),
                cwd=root,
                env=git_environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=30.0,
            ).stdout.split("\0")
            nonordinary = [
                entry for entry in index_states if entry and not entry.startswith("H ")
            ]
            if nonordinary:
                raise RuntimeError(
                    "the in-tree usd-cli source uses non-ordinary index "
                    "flags (for example assume-unchanged or skip-worktree)"
                )
            if committed_entries != tracked_entries:
                raise RuntimeError(
                    "the in-tree usd-cli index contains unrecorded changes"
                )
            _verify_source_files_match_index(
                root=root,
                source_root=source_root,
                tracked_entries=tracked_entries,
                git_environment=git_environment,
                object_format=object_format,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "could not verify the committed in-tree usd-cli source"
        ) from exc
    return source_revision


def pinned_usd_cli_executable_path(executable_dir: Path | None = None) -> str:
    """Return a ``PATH`` limited to the launcher's directory and system dirs.

    usd-cli processes resolve helper executables (OVRTX launchers, converters)
    through ``PATH``, so a caller-controlled entry could substitute a hostile
    binary even when the launcher itself is invoked by absolute path. Trust
    only the verified launcher's own directory plus the platform's fixed
    system path; empty, relative, and dot components are dropped.
    """

    trusted_entries: list[str] = []
    if executable_dir is not None:
        trusted_entries.append(str(Path(executable_dir).resolve()))
    trusted_entries.extend(
        entry
        for entry in os.defpath.split(os.pathsep)
        if entry and Path(entry).is_absolute()
    )
    return os.pathsep.join(dict.fromkeys(trusted_entries))


def sanitized_usd_cli_execution_env(
    base: dict[str, str] | None = None,
    *,
    executable_dir: Path | None = None,
) -> dict[str, str]:
    """Return a launch environment without caller-controlled code injection.

    ``PATH`` is always replaced with :func:`pinned_usd_cli_executable_path`;
    the caller's search path never reaches a usd-cli subprocess. Pass
    ``executable_dir`` so the verified launcher's sibling tools stay
    resolvable.
    """

    environment = dict(os.environ if base is None else base)
    for key in tuple(environment):
        if (
            key.startswith("PYTHON")
            or key.startswith("DYLD_")
            or key
            in {
                "LD_AUDIT",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "PXR_PLUGINPATH_NAME",
                "3DSC_NO_DAEMON",
                "OV_NO_DAEMON",
                "USD_CLI_NO_DAEMON",
                "USD_PLUGIN_PATH",
            }
        ):
            environment.pop(key, None)
    # A local __pycache__ can execute bytes that are not represented by the
    # committed source identity. Point cache reads at an impossible child of
    # the platform null device and disable writes for every usd-cli process.
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPYCACHEPREFIX"] = str(
        Path(os.devnull) / "content-agents-usd-cli"
    )
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    environment["PATH"] = pinned_usd_cli_executable_path(executable_dir)
    return environment


def git_verification_command(*arguments: str) -> list[str]:
    """Build a Git verification command independent of caller ``PATH``."""

    executable = (
        _windows_system_git_executable()
        if os.name == "nt"
        else shutil.which("git", path=os.defpath)
    )
    if executable is None:
        raise RuntimeError("could not resolve Git from the system executable path")
    return [executable, *_GIT_VERIFICATION_OPTIONS, *arguments]


def _windows_program_files_directories() -> tuple[Path, ...]:
    """Return machine-scoped Program Files roots from the protected registry."""

    try:
        import winreg
    except ImportError:  # pragma: no cover - imported only on Windows
        return ()

    directories: list[Path] = []
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion",
        ) as key:
            for value_name in ("ProgramFilesDir", "ProgramFilesDir (x86)"):
                try:
                    raw_value, _value_type = winreg.QueryValueEx(key, value_name)
                except FileNotFoundError:
                    continue
                if isinstance(raw_value, str) and raw_value:
                    directories.append(Path(raw_value))
    except OSError:
        return ()
    return tuple(dict.fromkeys(directories))


def _windows_system_git_executable() -> str | None:
    """Resolve machine-installed Git without consulting caller-controlled PATH."""

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    for program_files in _windows_program_files_directories():
        for relative in (Path("Git/cmd/git.exe"), Path("Git/bin/git.exe")):
            candidate = program_files / relative
            current = Path(candidate.anchor)
            try:
                for component in candidate.parts[1:]:
                    current /= component
                    metadata = current.lstat()
                    if getattr(metadata, "st_file_attributes", 0) & reparse_flag:
                        raise OSError("reparse point in system Git path")
                if stat.S_ISREG(metadata.st_mode):
                    return str(candidate.resolve(strict=True))
            except OSError:
                continue
    return None


def sanitized_git_verification_env(
    base: dict[str, str] | None = None,
    *,
    trusted_repository: Path | None = None,
) -> dict[str, str]:
    """Return a Git environment that cannot redirect repository verification.

    A Windows sandbox token can differ from the account that owns a worktree.
    Git rejects that checkout before any authenticated read unless its exact
    path is named as a protected ``safe.directory``. Ambient Git configuration
    remains disabled; only the caller-resolved repository is authorized.
    """

    environment = dict(os.environ if base is None else base)
    for key in tuple(environment):
        if (
            key.startswith("GIT_")
            or key.startswith("DYLD_")
            or key
            in {
                "LD_AUDIT",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "PXR_PLUGINPATH_NAME",
                "USD_PLUGIN_PATH",
            }
        ):
            environment.pop(key, None)
    environment.update(
        {
            "GIT_ALLOW_PROTOCOL": "",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "PATH": os.defpath,
        }
    )
    if os.name == "nt" and trusted_repository is not None:
        repository = trusted_repository.expanduser().resolve(strict=True)
        environment.update(
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "safe.directory",
                "GIT_CONFIG_VALUE_0": os.fspath(repository),
            }
        )
    return environment


def _recorded_distribution_file(
    distribution: importlib.metadata.Distribution,
    candidate: Path,
) -> importlib.metadata.PackagePath:
    """Return the exact RECORD entry for a console script or fail closed."""

    candidate_absolute = Path(os.path.abspath(candidate))
    for record in distribution.files or ():
        recorded_path = Path(os.path.abspath(distribution.locate_file(record)))
        if recorded_path == candidate_absolute:
            return record
    raise RuntimeError(
        f"the package-owned launcher is absent from the distribution RECORD: "
        f"{candidate_absolute}"
    )


def _verify_recorded_console_script(
    distribution: importlib.metadata.Distribution,
    candidate: Path,
    *,
    name: str,
) -> Path:
    """Verify one executable against its required sha256/size RECORD entry."""

    record = _recorded_distribution_file(distribution, candidate)
    record_hash = record.hash
    record_size = record.size
    if (
        record_hash is None
        or record_hash.mode.lower() != "sha256"
        or not record_hash.value
        or not isinstance(record_size, int)
        or isinstance(record_size, bool)
        or record_size < 0
        or record_size > MAX_CONSOLE_SCRIPT_BYTES
    ):
        raise RuntimeError(
            f"the package-owned {name} launcher lacks a bounded sha256/size "
            "distribution RECORD entry"
        )

    try:
        scripts_root = candidate.parent.resolve(strict=True)
        launcher = scripts_root / candidate.name
        verified = read_contained_artifact(
            scripts_root,
            launcher,
            max_bytes=MAX_CONSOLE_SCRIPT_BYTES,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"the package-owned {name} launcher is unavailable or unsafe: {candidate}"
        ) from exc

    actual_record_hash = (
        base64.urlsafe_b64encode(bytes.fromhex(verified.sha256))
        .rstrip(b"=")
        .decode("ascii")
    )
    if verified.size_bytes != record_size or actual_record_hash != record_hash.value:
        raise RuntimeError(
            f"the package-owned {name} launcher does not match its distribution "
            "RECORD entry"
        )
    if not os.access(verified.path, os.X_OK):
        raise RuntimeError(
            f"the package-owned {name} launcher is not executable: {verified.path}"
        )
    return verified.path


def _editable_distribution_source(
    distribution: importlib.metadata.Distribution,
) -> Path:
    """Return the exact editable source recorded by PEP 610 metadata."""

    try:
        raw_direct_url = distribution.read_text("direct_url.json")
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(
            "the installed usd-cli distribution cannot read direct_url.json"
        ) from exc
    if raw_direct_url is None:
        raise RuntimeError("the installed usd-cli distribution has no direct_url.json")
    try:
        direct_url = json.loads(raw_direct_url)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "the installed usd-cli distribution has invalid direct_url.json"
        ) from exc
    if not isinstance(direct_url, dict):
        raise RuntimeError("the installed usd-cli direct_url.json is not an object")
    url = direct_url.get("url")
    dir_info = direct_url.get("dir_info")
    if (
        not isinstance(url, str)
        or not isinstance(dir_info, dict)
        or dir_info.get("editable") is not True
    ):
        raise RuntimeError(
            "the installed usd-cli distribution is not an editable local source"
        )
    try:
        parsed = urlsplit(url)
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError(
            "the installed usd-cli distribution has an invalid local source URL"
        ) from exc
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            "the installed usd-cli distribution does not identify a local source"
        )
    try:
        decoded_path = url2pathname(parsed.path)
        if "\0" in decoded_path:
            raise ValueError("local source URL contains a null byte")
        return Path(decoded_path).resolve(strict=True)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(
            "the installed usd-cli editable source path is unavailable"
        ) from exc


def resolve_package_owned_usd_cli_route(repo_root: Path) -> UsdCliPackageRoute:
    """Resolve only launchers from this checkout's exact editable package."""

    root = repo_root.expanduser().resolve()
    if not usd_cli_source_distributed(root):
        raise RuntimeError(
            "usd-cli workflow capabilities are not distributed by this checkout"
        )
    revision = usd_cli_source_revision(root)
    source_root = (root / USD_CLI_SOURCE_PATH / "src").resolve(strict=True)
    try:
        distribution = importlib.metadata.distribution("usd-cli")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "the authorized in-tree usd-cli package is not installed in this environment"
        ) from exc
    distribution_source = _editable_distribution_source(distribution)
    if distribution_source != (root / USD_CLI_SOURCE_PATH).resolve(strict=True):
        raise RuntimeError(
            "the installed usd-cli distribution is editable from a foreign source: "
            f"{distribution_source}"
        )
    declared = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    for name, expected in USD_CLI_EXPECTED_ENTRY_POINTS.items():
        if declared.get(name) != expected:
            raise RuntimeError(
                f"the installed usd-cli package declares {name!r} as "
                f"{declared.get(name)!r}, expected {expected!r}"
            )

    for module_name in USD_CLI_EXPECTED_MODULES:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(
                f"the installed usd-cli package cannot resolve {module_name}"
            )
        try:
            Path(spec.origin).resolve(strict=True).relative_to(source_root)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"{module_name} resolves outside the authorized usd-cli source: "
                f"{spec.origin}"
            ) from exc

    scripts_value = sysconfig.get_path("scripts")
    if not scripts_value:
        raise RuntimeError(
            "the active Python environment has no console-script directory"
        )
    scripts_dir = Path(scripts_value)
    resolved: dict[str, Path] = {}
    for name in USD_CLI_EXPECTED_ENTRY_POINTS:
        launcher_name = f"{name}.exe" if os.name == "nt" else name
        resolved[name] = _verify_recorded_console_script(
            distribution,
            scripts_dir / launcher_name,
            name=name,
        )
    return UsdCliPackageRoute(
        wrapper=resolved["usd-cli-tel"],
        target=resolved["usd-cli"],
        source_root=source_root,
        source_revision=revision,
    )


def controlled_usd_cli_telemetry_env(
    *,
    route: UsdCliPackageRoute,
    telemetry_file: Path,
    attrs: str,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment whose telemetry controls cannot be inherited."""

    environment = sanitized_usd_cli_execution_env(
        base,
        executable_dir=route.wrapper.parent,
    )
    for key in tuple(environment):
        if key.startswith("USD_CLI_TEL_"):
            environment.pop(key, None)
    environment.update(
        {
            "USD_CLI_AGENT": "1",
            "USD_CLI_TEL_BACKENDS": "file",
            "USD_CLI_TEL_DISABLED": "0",
            "USD_CLI_TEL_FILE": str(telemetry_file),
            "USD_CLI_TEL_FILE_MAX_BYTES": str(50 * 1024 * 1024),
            "USD_CLI_TEL_ATTRS": attrs,
            "USD_CLI_TEL_TARGET": str(route.target),
        }
    )
    return environment


def find_usd_cli_repository_root(start: Path) -> Path:
    """Find the internal repository root that authorizes in-tree usd-cli."""

    resolved = start.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if usd_cli_source_distributed(candidate):
            return candidate
    raise RuntimeError(
        "the usd-cli workflow backend requires workflow-capable internal in-tree source"
    )


def validate_ovrtx_probe(
    probe: dict[str, object],
    *,
    required_capabilities: tuple[str, ...] = (),
) -> None:
    """Validate the workflow-wide OVRTX identity and render contract."""

    if probe.get("schema_version") != OVRTX_PROBE_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported usd-cli render probe schema: {probe.get('schema_version')!r}"
        )
    if probe.get("engine") != "ovrtx":
        raise RuntimeError(
            "content workflows require OVRTX rendering; usd-cli resolved "
            f"{probe.get('resolved_renderer') or probe.get('engine')!r}"
        )
    transport = probe.get("transport")
    if transport not in {"local", "remote"}:
        raise RuntimeError(f"OVRTX probe returned invalid transport: {transport!r}")
    if transport == "remote":
        backends = probe.get("backends")
        if not isinstance(backends, list) or not backends:
            raise RuntimeError(
                "remote OVRTX probe did not report verified endpoint profiles"
            )
        for backend in backends:
            if (
                not isinstance(backend, dict)
                or backend.get("engine") != "ovrtx"
                or not isinstance(backend.get("protocol_version"), int)
                or isinstance(backend.get("protocol_version"), bool)
                or backend.get("status") not in {"alive", "ready"}
            ):
                raise RuntimeError(
                    "remote OVRTX probe reported an unverified engine, protocol, "
                    "or readiness identity"
                )
    if probe.get("ready") is not True:
        raise RuntimeError(
            "OVRTX readiness/render probe failed: "
            f"{probe.get('error') or 'backend was not ready'}"
        )
    render = probe.get("render")
    if not isinstance(render, dict):
        raise RuntimeError("OVRTX probe did not report its render evidence")
    if render.get("width") != 64 or render.get("height") != 64:
        raise RuntimeError("OVRTX probe returned unexpected render dimensions")
    size_bytes = render.get("size_bytes")
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes <= 0
    ):
        raise RuntimeError("OVRTX probe did not produce a non-empty render")
    expected_backends = {"remote"} if transport == "remote" else {"ovrtx"}
    if render.get("backend") not in expected_backends:
        raise RuntimeError(
            "OVRTX probe render evidence reported an unexpected backend: "
            f"{render.get('backend')!r}"
        )
    capabilities = probe.get("capabilities")
    capability_values = (
        capabilities
        if isinstance(capabilities, list)
        and all(isinstance(value, str) for value in capabilities)
        else []
    )
    missing = sorted(set(required_capabilities).difference(capability_values))
    if missing:
        raise RuntimeError(
            "installed usd-cli is missing required workflow capability/capabilities: "
            + ", ".join(missing)
        )
