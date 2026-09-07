# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from content_agent_workflows.common import usd_cli as usd_cli_common
from PIL import Image

from content_workflow_cli import usd_cli_backend

_REVISION = "a" * 40


def test_backend_run_pins_uv_outside_sanitized_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = tmp_path / "verified" / "usd-cli"
    launcher.parent.mkdir()
    launcher.write_text("launcher", encoding="utf-8")
    uv_executable = tmp_path / "user-local" / "uv.exe"
    uv_executable.parent.mkdir()
    uv_executable.write_text("uv", encoding="utf-8")
    monkeypatch.setenv("PATH", str(uv_executable.parent))
    monkeypatch.delenv("USD_CLI_UV_EXECUTABLE", raising=False)
    monkeypatch.setattr(
        usd_cli_backend.shutil,
        "which",
        lambda name, *, path: str(uv_executable) if name == "uv" else None,
    )
    captured: dict[str, object] = {}

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(usd_cli_backend, "run_bounded_usd_cli_subprocess", fake_run)

    usd_cli_backend._run(
        [str(launcher), "--version"],
        timeout_seconds=1.0,
        description="test",
    )

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["USD_CLI_UV_EXECUTABLE"] == str(uv_executable.resolve())
    assert str(uv_executable.parent) not in environment["PATH"].split(os.pathsep)


def test_bounded_usd_cli_subprocess_fails_closed_on_output_overflow() -> None:
    started = time.monotonic()
    with pytest.raises(
        usd_cli_common.UsdCliSubprocessOutputError,
        match="stdout=1024 bytes",
    ):
        usd_cli_common.run_bounded_usd_cli_subprocess(
            [
                sys.executable,
                "-c",
                (
                    "import sys,time;"
                    "sys.stderr.write('e' * 512);"
                    "sys.stdout.write('x' * 4096);"
                    "sys.stdout.flush();"
                    "time.sleep(60)"
                ),
            ],
            timeout=10,
            stdout_limit=1024,
            stderr_limit=1024,
        )
    assert time.monotonic() - started < 2


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX process-group guard")
def test_bounded_usd_cli_subprocess_does_not_cancel_after_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An overflow observed during wait cannot kill a recycled POSIX PGID."""

    wait_entered = threading.Event()

    class OverflowStream:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def read(self, size: int) -> bytes:
            assert wait_entered.wait(timeout=1.0)
            payload, self.payload = self.payload, b""
            return payload

        def close(self) -> None:
            pass

    class FakeProcess:
        pid = 12345
        stdout = OverflowStream(b"x" * 2)
        stderr = OverflowStream(b"")
        killed = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            wait_entered.set()
            # Give the drain thread a chance to observe its overflow.  A waiter
            # that does not own cleanup_lock lets that thread kill this leader
            # after it is reaped; the shared lock prevents that stale kill.
            time.sleep(0.05)
            return 0

    process = FakeProcess()
    monkeypatch.setattr(
        usd_cli_common.subprocess, "Popen", lambda *args, **kwargs: process
    )

    with pytest.raises(usd_cli_common.UsdCliSubprocessOutputError):
        usd_cli_common.run_bounded_usd_cli_subprocess(
            ["usd-cli"], timeout=1.0, stdout_limit=1
        )

    assert not process.killed


def test_bounded_usd_cli_subprocess_preserves_launch_errors() -> None:
    executable = "__missing_usd_cli_executable__"
    with pytest.raises(FileNotFoundError) as error:
        usd_cli_common.run_bounded_usd_cli_subprocess(
            [executable],
            timeout=1.0,
        )
    assert error.value.filename == executable


@pytest.mark.parametrize(
    ("command", "error_type"),
    [([], IndexError), ([None], TypeError)],
)
def test_bounded_usd_cli_subprocess_preserves_launch_validation_errors(
    command: list[str | None], error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        usd_cli_common.run_bounded_usd_cli_subprocess(command, timeout=1.0)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux parent guard only")
def test_bounded_usd_cli_subprocess_preserves_signal_returncode() -> None:
    completed = usd_cli_common.run_bounded_usd_cli_subprocess(
        [
            sys.executable,
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        ],
        timeout=1.0,
    )
    assert completed.returncode == -signal.SIGTERM


def test_bounded_usd_cli_subprocess_closes_windows_job_before_pipe_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    job_closed = threading.Event()

    class FakeJob:
        creation_flags = 0

        def __init__(self, *, allow_breakaway: bool = False) -> None:
            assert allow_breakaway

        def assign_process(self, process: object) -> None:
            events.append("job.assign")

        def resume_process(self, process: object) -> None:
            events.append("job.resume")

        def close(self) -> None:
            events.append("job.close")
            job_closed.set()

        def terminate(self) -> None:
            events.append("job.terminate")

    class BlockingStream:
        def read(self, size: int) -> bytes:
            assert job_closed.wait(timeout=1.0)
            return b""

        def close(self) -> None:
            events.append("stream.close")

    class FakeProcess:
        pid = 1
        stdout = BlockingStream()
        stderr = BlockingStream()
        wait_calls = 0

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            events.append("process.kill")

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            events.append(f"process.wait.{self.wait_calls}")
            if timeout is not None:
                time.sleep(timeout)
                raise subprocess.TimeoutExpired(["usd-cli"], timeout)
            return 1

    process = FakeProcess()
    monkeypatch.setattr(usd_cli_common.os, "name", "nt")
    monkeypatch.setattr(usd_cli_common.sys, "platform", "win32")
    monkeypatch.setattr(usd_cli_common, "WindowsKillOnCloseJob", FakeJob)
    monkeypatch.setattr(
        usd_cli_common.subprocess, "Popen", lambda *args, **kwargs: process
    )

    with pytest.raises(subprocess.TimeoutExpired):
        usd_cli_common.run_bounded_usd_cli_subprocess(["usd-cli"], timeout=0.1)

    assert events.index("job.terminate") < events.index("process.wait.2")


def test_bounded_usd_cli_subprocess_closes_successful_windows_job_before_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    job_closed = threading.Event()

    class FakeJob:
        creation_flags = 0

        def __init__(self, *, allow_breakaway: bool = False) -> None:
            assert allow_breakaway

        def assign_process(self, process: object) -> None:
            pass

        def resume_process(self, process: object) -> None:
            pass

        def terminate(self) -> None:
            events.append("job.terminate")

        def close(self) -> None:
            events.append("job.close")
            job_closed.set()

    class BlockingStream:
        def read(self, size: int) -> bytes:
            assert job_closed.wait(timeout=1.0)
            events.append("stream.read")
            return b""

        def close(self) -> None:
            pass

    class FakeProcess:
        pid = 1
        stdout = BlockingStream()
        stderr = BlockingStream()

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr(usd_cli_common.os, "name", "nt")
    monkeypatch.setattr(usd_cli_common.sys, "platform", "win32")
    monkeypatch.setattr(usd_cli_common, "WindowsKillOnCloseJob", FakeJob)
    monkeypatch.setattr(
        usd_cli_common.subprocess, "Popen", lambda *args, **kwargs: FakeProcess()
    )

    usd_cli_common.run_bounded_usd_cli_subprocess(["usd-cli"], timeout=0.1)

    assert events.index("job.terminate") < events.index("stream.read")
    assert events.index("job.close") < events.index("stream.read")


def test_bounded_usd_cli_subprocess_reaps_windows_process_on_job_assignment_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeJob:
        creation_flags = 0

        def __init__(self, *, allow_breakaway: bool = False) -> None:
            assert allow_breakaway

        def assign_process(self, process: object) -> None:
            raise OSError("assignment failed")

        def resume_process(self, process: object) -> None:
            events.append("job.resume")

        def close(self) -> None:
            events.append("job.close")

    class FakeProcess:
        stdout = object()
        stderr = object()

        def kill(self) -> None:
            events.append("process.kill")

        def wait(self, timeout: float | None = None) -> int:
            events.append(f"process.wait.{timeout}")
            return 1

    process = FakeProcess()
    monkeypatch.setattr(usd_cli_common.os, "name", "nt")
    monkeypatch.setattr(usd_cli_common.sys, "platform", "win32")
    monkeypatch.setattr(usd_cli_common, "WindowsKillOnCloseJob", FakeJob)
    monkeypatch.setattr(
        usd_cli_common.subprocess, "Popen", lambda *args, **kwargs: process
    )

    with pytest.raises(OSError, match="assignment failed"):
        usd_cli_common.run_bounded_usd_cli_subprocess(["usd-cli"], timeout=0.1)

    assert events == ["job.resume", "process.kill", "process.wait.5", "job.close"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process states are required")
def test_bounded_usd_cli_subprocess_kills_successful_daemon_descendant(
    tmp_path: Path,
) -> None:
    ready_path = tmp_path / "daemon-ready"
    script = "\n".join(
        (
            "import os, subprocess, sys, time",
            f"ready_path = {str(ready_path)!r}",
            "child = subprocess.Popen(",
            "    [sys.executable, '-c', 'import pathlib, sys, time; "
            "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)', ready_path],",
            "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,",
            ")",
            "deadline = time.monotonic() + 2",
            "while not os.path.exists(ready_path) and time.monotonic() < deadline:",
            "    time.sleep(0.01)",
            "assert os.path.exists(ready_path)",
            "print(child.pid, flush=True)",
        )
    )
    child_pid: int | None = None
    try:
        completed = usd_cli_common.run_bounded_usd_cli_subprocess(
            [sys.executable, "-c", script], timeout=3.0
        )
        assert ready_path.exists()
        child_pid = int(completed.stdout.strip())

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                state = (
                    (Path("/proc") / str(child_pid) / "stat")
                    .read_text(encoding="utf-8")
                    .rsplit(")", 1)[1]
                    .split()[0]
                )
            except FileNotFoundError:
                break
            if state == "Z":
                break
            time.sleep(0.05)
        else:
            pytest.fail("successful usd-cli wrapper left a daemon descendant")
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, 9)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process states are required")
def test_bounded_usd_cli_subprocess_timeout_kills_descendant(tmp_path: Path) -> None:
    ready_path = tmp_path / "child-ready"
    script = "\n".join(
        (
            "import os, subprocess, sys, time",
            f"ready_path = {str(ready_path)!r}",
            "child = subprocess.Popen([",
            "    sys.executable, '-c',",
            "    'import pathlib, sys, time; "
            "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)',",
            "    ready_path,",
            "], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
            "deadline = time.monotonic() + 2",
            "while not os.path.exists(ready_path) and time.monotonic() < deadline:",
            "    time.sleep(0.01)",
            "assert os.path.exists(ready_path)",
            "print(child.pid, flush=True)",
            "time.sleep(60)",
        )
    )
    command = [
        sys.executable,
        "-c",
        script,
    ]
    child_pid: int | None = None
    try:
        with pytest.raises(subprocess.TimeoutExpired) as error:
            usd_cli_common.run_bounded_usd_cli_subprocess(command, timeout=3.0)
        assert ready_path.exists()
        assert error.value.output.strip()
        child_pid = int(error.value.output.strip())

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                state = (
                    (Path("/proc") / str(child_pid) / "stat")
                    .read_text(encoding="utf-8")
                    .rsplit(")", 1)[1]
                    .split()[0]
                )
            except FileNotFoundError:
                break
            if state == "Z":
                break
            time.sleep(0.05)
        else:
            pytest.fail("timed-out usd-cli descendant survived process-group cleanup")
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, 9)
            except ProcessLookupError:
                pass


def _write_required_source(repo: Path) -> None:
    for relative in usd_cli_backend.USD_CLI_REQUIRED_SOURCE:
        source = repo / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"{relative.as_posix()}\n", encoding="utf-8")


def _authorized_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_required_source(repo)
    return repo


def _initialized_source_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = _authorized_repo(tmp_path)
    source_root = repo / usd_cli_common.USD_CLI_SOURCE_PATH
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Content Workflow Tests",
            "-c",
            "user.email=content-workflow-tests@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", f"HEAD:{usd_cli_common.USD_CLI_SOURCE_PATH}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, source_root, revision


def _commit_fixture(repo: Path, message: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Content Workflow Tests",
            "-c",
            "user.email=content-workflow-tests@example.invalid",
            "commit",
            "--quiet",
            "-m",
            message,
        ],
        cwd=repo,
        check=True,
    )


def _source_revision(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"HEAD:{usd_cli_common.USD_CLI_SOURCE_PATH}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _revision(repo: Path, value: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "rev-parse", value],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _raw_git_object(repo: Path, object_type: str, object_id: str) -> bytes:
    return subprocess.run(
        ["git", "cat-file", object_type, object_id],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def _replace_loose_git_object(
    repo: Path,
    *,
    victim_object_id: str,
    object_type: str,
    foreign_content: bytes,
) -> None:
    object_location = subprocess.run(
        [
            "git",
            "rev-parse",
            "--git-path",
            f"objects/{victim_object_id[:2]}/{victim_object_id[2:]}",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    object_path = Path(object_location)
    if not object_path.is_absolute():
        object_path = repo / object_path
    object_path.chmod(0o644)
    object_path.write_bytes(
        zlib.compress(
            f"{object_type} {len(foreign_content)}\0".encode("ascii") + foreign_content
        )
    )


def _alternate_source_commit(
    repo: Path,
    source_root: Path,
) -> tuple[str, str]:
    old_head = _revision(repo)
    (source_root / "src/usd_cli/main.py").write_text(
        "alternate committed source\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "apps/usd_cli/src/usd_cli/main.py"],
        cwd=repo,
        check=True,
    )
    _commit_fixture(repo, "alternate source tree")
    new_head = _revision(repo)
    subprocess.run(["git", "reset", "--hard", old_head], cwd=repo, check=True)
    return old_head, new_head


def _raw_tree_entry(mode: str, name: bytes, object_byte: int) -> bytes:
    return mode.encode("ascii") + b" " + name + b"\0" + bytes([object_byte]) * 20


def _commit_usd_cli_ignore(repo: Path, *patterns: str) -> str:
    gitignore = repo / "apps/usd_cli/.gitignore"
    gitignore.write_text(
        "".join(f"{pattern}\n" for pattern in patterns), encoding="utf-8"
    )
    subprocess.run(["git", "add", "apps/usd_cli/.gitignore"], cwd=repo, check=True)
    _commit_fixture(repo, "configure usd-cli runtime ignores")
    return _source_revision(repo)


def _package_route(repo: Path) -> usd_cli_common.UsdCliPackageRoute:
    scripts = repo / "venv" / "bin"
    scripts.mkdir(parents=True)
    wrapper = scripts / "usd-cli-tel"
    target = scripts / "usd-cli"
    for executable in (wrapper, target):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(executable.stat().st_mode | os.X_OK)
    return usd_cli_common.UsdCliPackageRoute(
        wrapper=wrapper.resolve(),
        target=target.resolve(),
        source_root=(repo / usd_cli_common.USD_CLI_SOURCE_PATH / "src").resolve(),
        source_revision=_REVISION,
    )


def _record_hash(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


class _FakeDistribution:
    def __init__(
        self,
        scripts: dict[str, tuple[Path, bytes]],
        *,
        source_root: Path,
    ) -> None:
        self._source_root = source_root
        self.entry_points = [
            SimpleNamespace(group="console_scripts", name=name, value=value)
            for name, value in usd_cli_common.USD_CLI_EXPECTED_ENTRY_POINTS.items()
        ]
        self.files = [
            SimpleNamespace(
                location=path,
                hash=SimpleNamespace(mode="sha256", value=_record_hash(data)),
                size=len(data),
            )
            for path, data in scripts.values()
        ]

    @staticmethod
    def locate_file(record: SimpleNamespace) -> Path:
        return record.location

    def read_text(self, filename: str) -> str | None:
        if filename != "direct_url.json":
            return None
        return json.dumps(
            {
                "url": self._source_root.resolve().as_uri(),
                "dir_info": {"editable": True},
            }
        )


def _probe(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "usd-cli.render-probe.v1",
        "capabilities": ["appearance.clear.v1"],
        "required_engine": "ovrtx",
        "resolved_renderer": "ovrtx",
        "engine": "ovrtx",
        "transport": "local",
        "ready": True,
        "render": {
            "path": "/run/raw/ovrtx_probe/probe.png",
            "width": 64,
            "height": 64,
            "size_bytes": 128,
            "backend": "ovrtx",
        },
    }
    value.update(updates)
    return value


def test_distribution_requires_normal_source_and_no_submodule_metadata(
    tmp_path: Path,
) -> None:
    repo = _authorized_repo(tmp_path)

    assert usd_cli_backend.usd_cli_source_distributed(repo) is True

    (repo / ".gitmodules").write_text(
        '[submodule "usd-cli"]\n\tpath = apps/usd_cli\n',
        encoding="utf-8",
    )
    assert usd_cli_backend.usd_cli_source_distributed(repo) is False

    (repo / ".gitmodules").unlink()
    (repo / usd_cli_common.USD_CLI_SOURCE_PATH / ".git").mkdir()
    assert usd_cli_backend.usd_cli_source_distributed(repo) is False

    (repo / usd_cli_common.USD_CLI_SOURCE_PATH / ".git").rmdir()
    required = repo / usd_cli_backend.USD_CLI_REQUIRED_SOURCE[-1]
    required.unlink()
    assert usd_cli_backend.usd_cli_source_distributed(repo) is False


def test_symlinked_required_source_cannot_authorize_distribution(
    tmp_path: Path,
) -> None:
    repo = _authorized_repo(tmp_path)
    required = repo / usd_cli_backend.USD_CLI_REQUIRED_SOURCE[-1]
    outside = tmp_path / "outside-probe.py"
    outside.write_text("source", encoding="utf-8")
    required.unlink()
    required.symlink_to(outside)

    assert usd_cli_backend.usd_cli_source_distributed(repo) is False


def test_distribution_source_is_not_reopened_after_descriptor_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _authorized_repo(tmp_path)

    def forbidden_read_text(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("authorization source must not be reopened by path")

    monkeypatch.setattr(Path, "read_text", forbidden_read_text)

    assert usd_cli_backend.usd_cli_source_distributed(repo) is True


def test_global_executable_cannot_bypass_missing_source_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_bin = tmp_path / "global-bin"
    tool_bin.mkdir()
    execution_marker = tmp_path / "global-usd-cli-executed"
    executable = tool_bin / "usd-cli"
    executable.write_text(
        f"#!/bin/sh\ntouch {execution_marker!s}\nexit 0\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | os.X_OK)
    monkeypatch.setenv("PATH", str(tool_bin))

    with pytest.raises(RuntimeError, match="not distributed"):
        usd_cli_backend.ensure_usd_cli_ovrtx_ready(tmp_path)

    assert not execution_marker.exists()


def test_readiness_artifact_stem_rejects_path_syntax(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="artifact stem is invalid"):
        usd_cli_backend.ensure_usd_cli_ovrtx_ready(
            tmp_path,
            artifact_stem="../outside",
        )


def test_source_revision_uses_committed_parent_subtree(tmp_path: Path) -> None:
    repo, _source_root, revision = _initialized_source_repo(tmp_path)

    assert usd_cli_common.usd_cli_source_revision(repo) == revision
    assert (
        revision
        == subprocess.run(
            ["git", "rev-parse", f"HEAD:{usd_cli_common.USD_CLI_SOURCE_PATH}"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def test_source_revision_authenticates_sha256_commit_and_trees(
    tmp_path: Path,
) -> None:
    repo = _authorized_repo(tmp_path)
    initialized = subprocess.run(
        ["git", "init", "--quiet", "--object-format=sha256"],
        cwd=repo,
        check=False,
        capture_output=True,
    )
    if initialized.returncode != 0:
        pytest.skip("installed Git does not support sha256 object repositories")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    _commit_fixture(repo, "sha256 fixture")
    expected = _source_revision(repo)

    assert len(_revision(repo)) == 64
    assert len(expected) == 64
    assert usd_cli_common.usd_cli_source_revision(repo) == expected


def test_source_revision_ignores_inherited_git_repository_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _source_root, revision = _initialized_source_repo(tmp_path)
    poison = tmp_path / "poison.git"
    subprocess.run(["git", "init", "--bare", "--quiet", str(poison)], check=True)
    monkeypatch.setenv("GIT_DIR", str(poison))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "foreign-worktree"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "foreign-index"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.bare")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")

    assert usd_cli_common.usd_cli_source_revision(repo) == revision


def test_git_verification_environment_disables_replace_objects() -> None:
    environment = usd_cli_common.sanitized_git_verification_env(
        {
            "DYLD_INSERT_LIBRARIES": "/tmp/injected.dylib",
            "GIT_ALLOW_PROTOCOL": "file:ssh:https",
            "GIT_DIR": "/tmp/foreign.git",
            "GIT_NO_LAZY_FETCH": "0",
            "GIT_NO_REPLACE_OBJECTS": "0",
            "KEEP_ME": "yes",
            "LD_AUDIT": "/tmp/audit.so",
            "LD_LIBRARY_PATH": "/tmp/foreign-libs",
            "LD_PRELOAD": "/tmp/injected.so",
            "PXR_PLUGINPATH_NAME": "/tmp/foreign-plugins",
            "3DSC_NO_DAEMON": "1",
            "OV_NO_DAEMON": "1",
            "USD_PLUGIN_PATH": "/tmp/foreign-usd-plugins",
        }
    )

    assert environment["KEEP_ME"] == "yes"
    assert environment["GIT_ALLOW_PROTOCOL"] == ""
    assert environment["GIT_NO_LAZY_FETCH"] == "1"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["PATH"] == os.defpath
    assert "GIT_DIR" not in environment
    for key in (
        "DYLD_INSERT_LIBRARIES",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PXR_PLUGINPATH_NAME",
        "USD_PLUGIN_PATH",
    ):
        assert key not in environment


@pytest.mark.skipif(os.name != "nt", reason="Windows safe.directory handling")
def test_git_verification_environment_authorizes_only_exact_repository(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    environment = usd_cli_common.sanitized_git_verification_env(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": "*",
        },
        trusted_repository=repo,
    )

    assert environment["GIT_CONFIG_COUNT"] == "1"
    assert environment["GIT_CONFIG_KEY_0"] == "safe.directory"
    assert environment["GIT_CONFIG_VALUE_0"] == os.fspath(repo.resolve())


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_source_revision_rejects_nonordinary_index_flags(
    tmp_path: Path,
    index_flag: str,
) -> None:
    repo, _source_root, _revision = _initialized_source_repo(tmp_path)
    relative = Path("apps/usd_cli/src/usd_cli/main.py")
    subprocess.run(
        ["git", "update-index", index_flag, relative.as_posix()],
        cwd=repo,
        check=True,
    )
    (repo / relative).write_text("modified but hidden\n", encoding="utf-8")
    assert (
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )

    with pytest.raises(RuntimeError, match="non-ordinary index flags"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_source_revision_disables_repository_fsmonitor_hook(
    tmp_path: Path,
) -> None:
    repo, _source_root, revision = _initialized_source_repo(tmp_path)
    execution_marker = tmp_path / "fsmonitor-executed"
    hook = tmp_path / "fsmonitor-hook"
    hook.write_text(
        f"#!/bin/sh\ntouch {shlex.quote(str(execution_marker))}\nprintf '\\n'\n",
        encoding="utf-8",
    )
    hook.chmod(hook.stat().st_mode | os.X_OK)
    subprocess.run(
        ["git", "config", "core.fsmonitor", str(hook)],
        cwd=repo,
        check=True,
    )

    assert usd_cli_common.usd_cli_source_revision(repo) == revision
    assert not execution_marker.exists()


def test_source_revision_does_not_execute_repository_clean_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _source_root, revision = _initialized_source_repo(tmp_path)
    attributes = repo / ".gitattributes"
    attributes.write_text("*.py filter=evil\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitattributes"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Content Workflow Tests",
            "-c",
            "user.email=content-workflow-tests@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "configure attributes",
        ],
        cwd=repo,
        check=True,
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    execution_marker = tmp_path / "clean-filter-executed"
    clean_filter = fake_bin / "evil-filter"
    clean_filter.write_text(
        f"#!/bin/sh\ntouch {shlex.quote(str(execution_marker))}\ncat\n",
        encoding="utf-8",
    )
    clean_filter.chmod(clean_filter.stat().st_mode | os.X_OK)
    subprocess.run(
        ["git", "config", "filter.evil.clean", "evil-filter"],
        cwd=repo,
        check=True,
    )
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")

    assert usd_cli_common.usd_cli_source_revision(repo) == revision
    assert not execution_marker.exists()


def test_dirty_source_revision_fails_closed(tmp_path: Path) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    (source_root / "src/usd_cli/main.py").write_text(
        "modified but unrecorded\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="unrecorded changes"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_source_revision_authenticates_captured_head_commit(
    tmp_path: Path,
) -> None:
    repo, source_root, _revision_value = _initialized_source_repo(tmp_path)
    old_head, new_head = _alternate_source_commit(repo, source_root)
    foreign_commit = _raw_git_object(repo, "commit", new_head)
    _replace_loose_git_object(
        repo,
        victim_object_id=old_head,
        object_type="commit",
        foreign_content=foreign_commit,
    )

    with pytest.raises(RuntimeError, match="commit.*does not match its Git identity"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_source_revision_rejects_corrupted_root_tree_remapping(
    tmp_path: Path,
) -> None:
    repo, source_root, _revision_value = _initialized_source_repo(tmp_path)
    old_head, new_head = _alternate_source_commit(repo, source_root)
    old_root_tree = _revision(repo, f"{old_head}^{{tree}}")
    new_root_tree = _revision(repo, f"{new_head}^{{tree}}")
    assert old_root_tree != new_root_tree
    _replace_loose_git_object(
        repo,
        victim_object_id=old_root_tree,
        object_type="tree",
        foreign_content=_raw_git_object(repo, "tree", new_root_tree),
    )

    with pytest.raises(RuntimeError, match="tree.*does not match its Git identity"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_source_revision_rejects_corrupted_nested_tree_remapping(
    tmp_path: Path,
) -> None:
    repo, source_root, _revision_value = _initialized_source_repo(tmp_path)
    old_head, new_head = _alternate_source_commit(repo, source_root)
    nested_path = "apps/usd_cli/src/usd_cli"
    old_nested_tree = _revision(repo, f"{old_head}:{nested_path}")
    new_nested_tree = _revision(repo, f"{new_head}:{nested_path}")
    assert old_nested_tree != new_nested_tree
    _replace_loose_git_object(
        repo,
        victim_object_id=old_nested_tree,
        object_type="tree",
        foreign_content=_raw_git_object(repo, "tree", new_nested_tree),
    )

    with pytest.raises(RuntimeError, match="tree.*does not match its Git identity"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_source_revision_uses_one_captured_head_when_ref_moves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source_root, old_source_revision = _initialized_source_repo(tmp_path)
    old_head, new_head = _alternate_source_commit(repo, source_root)
    assert _revision(repo) == old_head
    real_run = subprocess.run
    head_references: list[tuple[str, ...]] = []

    def moving_run(
        command: list[str],
        *args: object,
        **kwargs: object,
    ):
        result = real_run(command, *args, **kwargs)
        arguments = tuple(str(value) for value in command)
        if any(value == "HEAD" or value.startswith("HEAD:") for value in arguments):
            head_references.append(arguments)
        if arguments[-3:] == ("rev-parse", "--verify", "HEAD"):
            real_run(
                ["git", "update-ref", "HEAD", new_head],
                cwd=repo,
                check=True,
            )
        return result

    monkeypatch.setattr(usd_cli_common.subprocess, "run", moving_run)

    assert usd_cli_common.usd_cli_source_revision(repo) == old_source_revision
    assert len(head_references) == 1
    assert head_references[0][-3:] == ("rev-parse", "--verify", "HEAD")
    assert (
        real_run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == new_head
    )


@pytest.mark.parametrize(
    ("content", "message"),
    (
        (
            _raw_tree_entry("100644", b"z", 1) + _raw_tree_entry("100644", b"a", 2),
            "incorrectly ordered",
        ),
        (
            _raw_tree_entry("100644", b"same", 1)
            + _raw_tree_entry("100755", b"same", 2),
            "duplicate",
        ),
        (_raw_tree_entry("100644", b"..", 1), "unsafe"),
        (_raw_tree_entry("100644", b"dir\\escape", 1), "unsafe"),
        (_raw_tree_entry("100644", b"truncated", 1)[:-1], "truncated"),
        (_raw_tree_entry("100664", b"mode", 1), "non-canonical"),
    ),
)
def test_authenticated_tree_parser_rejects_noncanonical_entries(
    content: bytes,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        usd_cli_common._parse_authenticated_tree(content, object_format="sha1")


def test_authenticated_tree_parser_uses_git_directory_ordering() -> None:
    incorrectly_ordered = _raw_tree_entry("40000", b"folder", 1) + _raw_tree_entry(
        "100644", b"folder.txt", 2
    )

    with pytest.raises(RuntimeError, match="incorrectly ordered"):
        usd_cli_common._parse_authenticated_tree(
            incorrectly_ordered,
            object_format="sha1",
        )


def test_source_revision_rejects_corrupted_loose_blob_identity(
    tmp_path: Path,
) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    relative = "apps/usd_cli/src/usd_cli/main.py"
    object_id = subprocess.run(
        ["git", "rev-parse", f"HEAD:{relative}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    foreign_payload = b"foreign but internally valid loose blob\n"
    object_location = subprocess.run(
        [
            "git",
            "rev-parse",
            "--git-path",
            f"objects/{object_id[:2]}/{object_id[2:]}",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    object_path = Path(object_location)
    if not object_path.is_absolute():
        object_path = repo / object_path
    object_path.chmod(0o644)
    object_path.write_bytes(
        zlib.compress(
            f"blob {len(foreign_payload)}\0".encode("ascii") + foreign_payload
        )
    )
    (source_root / "src/usd_cli/main.py").write_bytes(foreign_payload)

    with pytest.raises(RuntimeError, match="does not match its Git identity"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_committed_blob_reader_drains_stderr_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"x"
    object_id = hashlib.sha1(b"blob 1\0" + payload, usedforsecurity=False).hexdigest()
    script = (
        "import sys\n"
        "request = sys.stdin.buffer.readline().strip()\n"
        "sys.stderr.buffer.write(b'e' * (1024 * 1024))\n"
        "sys.stderr.buffer.flush()\n"
        "sys.stdout.buffer.write(request + b' blob 1\\n' + b'x\\n')\n"
        "sys.stdout.buffer.flush()\n"
    )
    monkeypatch.setattr(
        usd_cli_common,
        "git_verification_command",
        lambda *_arguments: [sys.executable, "-c", script],
    )

    assert usd_cli_common._committed_blob_identity(
        root=tmp_path,
        git_environment=dict(os.environ),
        object_id=object_id,
        object_format="sha1",
    ) == (1, hashlib.sha256(payload).hexdigest(), payload)


def test_committed_blob_reader_uses_sha256_repository_object_format(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "sha256-repo"
    initialized = subprocess.run(
        ["git", "init", "--quiet", "--object-format=sha256", str(repo)],
        check=False,
        capture_output=True,
    )
    if initialized.returncode != 0:
        pytest.skip("installed Git does not support sha256 object repositories")
    payload = b"sha256 repository blob\n"
    object_id = (
        subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            cwd=repo,
            input=payload,
            check=True,
            capture_output=True,
        )
        .stdout.decode("ascii")
        .strip()
    )

    assert usd_cli_common._committed_blob_identity(
        root=repo,
        git_environment=usd_cli_common.sanitized_git_verification_env(),
        object_id=object_id,
        object_format="sha256",
    ) == (len(payload), hashlib.sha256(payload).hexdigest(), payload)


def test_committed_blob_reader_normalizes_whole_operation_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_id = hashlib.sha1(b"blob 1\0x", usedforsecurity=False).hexdigest()
    script = (
        "import sys, time\n"
        "sys.stdin.buffer.readline()\n"
        "sys.stderr.buffer.write(b'e' * (1024 * 1024))\n"
        "sys.stderr.buffer.flush()\n"
        "time.sleep(5)\n"
    )
    monkeypatch.setattr(
        usd_cli_common,
        "git_verification_command",
        lambda *_arguments: [sys.executable, "-c", script],
    )
    monkeypatch.setattr(
        usd_cli_common,
        "GIT_VERIFICATION_TIMEOUT_SECONDS",
        0.2,
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        usd_cli_common._committed_blob_identity(
            root=tmp_path,
            git_environment=dict(os.environ),
            object_id=object_id,
            object_format="sha1",
        )
    assert time.monotonic() - started < 2.0


def test_source_revision_reports_grown_tracked_file_as_unrecorded_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    target = source_root / "src/usd_cli/main.py"
    committed_size = target.stat().st_size
    target.write_bytes(b"x" * (1024 * 1024))
    real_read = usd_cli_common.read_contained_artifact
    target_was_read = False

    def recording_read(
        run_dir: str | Path,
        candidate: str | Path,
        **kwargs: object,
    ):
        nonlocal target_was_read
        if Path(candidate) == target:
            target_was_read = True
        return real_read(run_dir, candidate, **kwargs)

    monkeypatch.setattr(usd_cli_common, "read_contained_artifact", recording_read)

    with pytest.raises(
        RuntimeError,
        match=r"unrecorded changes: apps/usd_cli/src/usd_cli/main\.py",
    ):
        usd_cli_common.usd_cli_source_revision(repo)
    assert committed_size < target.stat().st_size
    assert target_was_read is False


def test_source_revision_accepts_matching_smudged_lfs_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    payload = (b"content-agents-lfs-payload\n" * 128) + b"complete\n"
    subprocess.run(
        ["git", "lfs", "install", "--local"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "lfs", "track", "*.usd"],
        cwd=source_root,
        check=True,
        capture_output=True,
    )
    asset = source_root / "sample.usd"
    asset.write_bytes(payload)
    subprocess.run(
        ["git", "add", "apps/usd_cli/.gitattributes", "apps/usd_cli/sample.usd"],
        cwd=repo,
        check=True,
    )
    _commit_fixture(repo, "add locally configured LFS payload")
    revision = _source_revision(repo)
    pointer = subprocess.run(
        ["git", "show", "HEAD:apps/usd_cli/sample.usd"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    assert usd_cli_common._lfs_payload_identity(pointer) == (
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )
    assert (
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )

    real_read = usd_cli_common.read_contained_artifact
    observed_limits: dict[str, int | None] = {}

    def recording_read(
        run_dir: str | Path,
        candidate: str | Path,
        **kwargs: object,
    ):
        observed_limits[Path(candidate).relative_to(Path(run_dir)).as_posix()] = (
            kwargs.get("max_bytes")
            if isinstance(kwargs.get("max_bytes"), int)
            else None
        )
        return real_read(run_dir, candidate, **kwargs)

    monkeypatch.setattr(usd_cli_common, "read_contained_artifact", recording_read)

    assert usd_cli_common.usd_cli_source_revision(repo) == revision
    assert observed_limits["sample.usd"] == len(payload)


def test_source_revision_rejects_unattributed_smudged_pointer(
    tmp_path: Path,
) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    payload = b"unattributed payload\n" * 64
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{hashlib.sha256(payload).hexdigest()}\n"
        f"size {len(payload)}\n"
    ).encode("ascii")
    asset = source_root / "sample.usd"
    asset.write_bytes(pointer)
    subprocess.run(["git", "add", "apps/usd_cli/sample.usd"], cwd=repo, check=True)
    _commit_fixture(repo, "add unattributed pointer")
    asset.write_bytes(payload)

    with pytest.raises(RuntimeError, match="inspect|unrecorded changes"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_source_revision_rejects_pointer_shaped_python_smudge(
    tmp_path: Path,
) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    payload = b"AUTHORIZED = False\n"
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{hashlib.sha256(payload).hexdigest()}\n"
        f"size {len(payload)}\n"
    ).encode("ascii")
    source = source_root / "src/usd_cli/plugin.py"
    source.write_bytes(pointer)
    subprocess.run(
        ["git", "add", "apps/usd_cli/src/usd_cli/plugin.py"], cwd=repo, check=True
    )
    _commit_fixture(repo, "add pointer-shaped Python source")
    source.write_bytes(payload)

    with pytest.raises(RuntimeError, match="import source"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_source_revision_rejects_attributed_import_source(
    tmp_path: Path,
) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    subprocess.run(
        ["git", "lfs", "install", "--local"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "lfs", "track", "src/usd_cli/plugin.py"],
        cwd=source_root,
        check=True,
        capture_output=True,
    )
    source = source_root / "src/usd_cli/plugin.py"
    source.write_text("AUTHORIZED = False\n", encoding="utf-8")
    subprocess.run(
        [
            "git",
            "add",
            "apps/usd_cli/.gitattributes",
            "apps/usd_cli/src/usd_cli/plugin.py",
        ],
        cwd=repo,
        check=True,
    )
    _commit_fixture(repo, "add attributed Python source")

    with pytest.raises(RuntimeError, match="unreviewed attribute pattern"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_source_revision_rejects_executable_lfs_payload(
    tmp_path: Path,
) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    subprocess.run(
        ["git", "lfs", "install", "--local"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "lfs", "track", "*.usd"],
        cwd=source_root,
        check=True,
        capture_output=True,
    )
    asset = source_root / "executable.usd"
    asset.write_bytes(b"executable LFS payload\n" * 64)
    asset.chmod(0o755)
    subprocess.run(
        ["git", "add", "apps/usd_cli/.gitattributes", "apps/usd_cli/executable.usd"],
        cwd=repo,
        check=True,
    )
    _commit_fixture(repo, "add executable LFS payload")

    with pytest.raises(RuntimeError, match="executable LFS artifact"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_source_revision_allows_reviewed_non_source_cache(
    tmp_path: Path,
) -> None:
    repo, _source_root, _revision = _initialized_source_repo(tmp_path)
    revision = _commit_usd_cli_ignore(repo, ".pytest_cache/")
    cache = repo / "apps/usd_cli/.pytest_cache/CACHEDIR.TAG"
    cache.parent.mkdir()
    cache.write_text("Signature: pytest cache directory\n", encoding="utf-8")

    assert usd_cli_common.usd_cli_source_revision(repo) == revision


def test_source_revision_rejects_nonignored_runtime_cache(
    tmp_path: Path,
) -> None:
    repo, _source_root, _revision = _initialized_source_repo(tmp_path)
    cache = repo / "apps/usd_cli/.pytest_cache/CACHEDIR.TAG"
    cache.parent.mkdir()
    cache.write_text("not ignored\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="runtime directory is not ignored"):
        usd_cli_common.usd_cli_source_revision(repo)


@pytest.mark.parametrize(
    "relative_cache",
    [
        "src/usd_cli/__pycache__/main.cpython-312.pyc",
        "apps/ovrtx_rendering_api/service/__pycache__/physics.cpython-312.pyc",
        "tests/__pycache__/test_render.cpython-312-pytest-8.4.1.pyc",
    ],
)
def test_source_revision_allows_redirected_bytecode_cache(
    tmp_path: Path,
    relative_cache: str,
) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    revision = _commit_usd_cli_ignore(repo, "__pycache__/")
    bytecode = source_root / relative_cache
    bytecode.parent.mkdir(parents=True)
    bytecode.write_bytes(b"ignored bytecode generated by direct CLI use")

    assert usd_cli_common.usd_cli_source_revision(repo) == revision


def test_source_revision_rejects_non_bytecode_file_in_source_cache(
    tmp_path: Path,
) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    _commit_usd_cli_ignore(repo, "__pycache__/")
    shadow = source_root / "src/usd_cli/__pycache__/main.py"
    shadow.parent.mkdir()
    shadow.write_text("unrecorded source\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="ignored or untracked artifacts"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_source_revision_rejects_ignored_cache_below_docs(
    tmp_path: Path,
) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    _commit_usd_cli_ignore(repo, "cache/")
    untracked = source_root / "docs/cache/untracked.py"
    untracked.parent.mkdir(parents=True)
    untracked.write_text("unrecorded source\n", encoding="utf-8")
    assert (
        subprocess.run(
            [
                "git",
                "check-ignore",
                "--quiet",
                "--",
                "apps/usd_cli/docs/cache/untracked.py",
            ],
            cwd=repo,
            check=False,
        ).returncode
        == 0
    )

    with pytest.raises(RuntimeError, match="ignored or untracked artifacts"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_source_revision_prunes_real_ignored_venv_with_internal_symlink(
    tmp_path: Path,
) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    revision = _commit_usd_cli_ignore(repo, ".venv/")
    executable = source_root / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to("/usr/bin/python3")

    assert usd_cli_common.usd_cli_source_revision(repo) == revision


def test_source_revision_rejects_symlinked_runtime_root(
    tmp_path: Path,
) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    _commit_usd_cli_ignore(repo, ".venv/")
    foreign_venv = tmp_path / "foreign-venv"
    foreign_venv.mkdir()
    (source_root / ".venv").symlink_to(foreign_venv, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink or special directory"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_source_revision_rejects_ignored_import_surface_artifact(
    tmp_path: Path,
) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    gitignore = repo / ".gitignore"
    gitignore.write_text("*.so\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Content Workflow Tests",
            "-c",
            "user.email=content-workflow-tests@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "ignore native extensions",
        ],
        cwd=repo,
        check=True,
    )
    shadow = source_root / "src/usd_cli/main.so"
    shadow.write_bytes(b"unrecorded native module")
    assert (
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )

    with pytest.raises(RuntimeError, match="ignored or untracked artifacts"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_source_revision_rejects_gitlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{_REVISION},{usd_cli_common.USD_CLI_SOURCE_PATH}",
        ],
        cwd=repo,
        check=True,
    )
    (repo / usd_cli_common.USD_CLI_SOURCE_PATH).mkdir(parents=True)
    _commit_fixture(repo, "gitlink fixture")

    with pytest.raises(RuntimeError, match="ordinary usd-cli source tree"):
        usd_cli_common.usd_cli_source_revision(repo, require_clean=False)


def test_source_revision_rejects_tracked_symlink(tmp_path: Path) -> None:
    repo, source_root, _revision = _initialized_source_repo(tmp_path)
    required = source_root / "src/usd_core/render/probe.py"
    required.unlink()
    required.symlink_to(source_root / "src/usd_cli/main.py")
    subprocess.run(["git", "add", required], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Content Workflow Tests",
            "-c",
            "user.email=content-workflow-tests@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "symlink",
        ],
        cwd=repo,
        check=True,
    )

    with pytest.raises(RuntimeError, match="ordinary committed files"):
        usd_cli_common.usd_cli_source_revision(repo)


def test_package_route_requires_console_scripts_to_match_distribution_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _authorized_repo(tmp_path)
    scripts_dir = tmp_path / "venv" / "bin"
    scripts_dir.mkdir(parents=True)
    script_records: dict[str, tuple[Path, bytes]] = {}
    for name in usd_cli_common.USD_CLI_EXPECTED_ENTRY_POINTS:
        content = f"#!/bin/sh\n# {name}\nexit 0\n".encode()
        path = scripts_dir / name
        path.write_bytes(content)
        path.chmod(0o755)
        script_records[name] = (path, content)

    distribution = _FakeDistribution(
        script_records,
        source_root=repo / usd_cli_common.USD_CLI_SOURCE_PATH,
    )
    monkeypatch.setattr(
        usd_cli_common,
        "usd_cli_source_revision",
        lambda _repo_root: _REVISION,
    )
    monkeypatch.setattr(
        usd_cli_common.importlib.metadata,
        "distribution",
        lambda _name: distribution,
    )
    module_origins = {
        "usd_cli": repo / "apps/usd_cli/src/usd_cli/__init__.py",
        "usd_core": repo / "apps/usd_cli/src/usd_core/__init__.py",
        "usd_server": repo / "apps/usd_cli/src/usd_server/__init__.py",
        "usd_telemetry": repo / "apps/usd_cli/src/usd_telemetry/main.py",
    }
    monkeypatch.setattr(
        usd_cli_common.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(module_origins[name])),
    )
    monkeypatch.setattr(
        usd_cli_common.sysconfig,
        "get_path",
        lambda name: str(scripts_dir) if name == "scripts" else None,
    )

    foreign_distribution = _FakeDistribution(
        script_records,
        source_root=tmp_path / "foreign-usd-cli",
    )
    (tmp_path / "foreign-usd-cli").mkdir()
    monkeypatch.setattr(
        usd_cli_common.importlib.metadata,
        "distribution",
        lambda _name: foreign_distribution,
    )
    with pytest.raises(RuntimeError, match="editable from a foreign source"):
        usd_cli_common.resolve_package_owned_usd_cli_route(repo)
    monkeypatch.setattr(
        usd_cli_common.importlib.metadata,
        "distribution",
        lambda _name: distribution,
    )

    route = usd_cli_common.resolve_package_owned_usd_cli_route(repo)
    assert route.wrapper == scripts_dir / "usd-cli-tel"
    assert route.target == scripts_dir / "usd-cli"

    foreign_module = tmp_path / "foreign-package/__init__.py"
    foreign_module.parent.mkdir()
    foreign_module.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        usd_cli_common.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(
            origin=str(foreign_module if name == "usd_core" else module_origins[name])
        ),
    )
    with pytest.raises(RuntimeError, match="usd_core resolves outside"):
        usd_cli_common.resolve_package_owned_usd_cli_route(repo)
    monkeypatch.setattr(
        usd_cli_common.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(module_origins[name])),
    )

    route.target.write_text("#!/bin/sh\nexit 42\n", encoding="utf-8")
    route.target.chmod(0o755)
    with pytest.raises(RuntimeError, match="does not match.*RECORD"):
        usd_cli_common.resolve_package_owned_usd_cli_route(repo)


def test_editable_distribution_source_decodes_file_url_once(tmp_path: Path) -> None:
    source_root = tmp_path / "literal%20source"
    source_root.mkdir()
    distribution = _FakeDistribution({}, source_root=source_root)

    assert usd_cli_common._editable_distribution_source(distribution) == source_root


@pytest.mark.parametrize(
    "source_url",
    (
        "file://[malformed",
        "file:///tmp/source%00path",
    ),
)
def test_editable_distribution_source_wraps_malformed_file_url(
    source_url: str,
) -> None:
    distribution = SimpleNamespace(
        read_text=lambda _filename: json.dumps(
            {"url": source_url, "dir_info": {"editable": True}}
        )
    )

    with pytest.raises(RuntimeError, match="usd-cli distribution|editable source"):
        usd_cli_common._editable_distribution_source(distribution)


def test_source_revision_ignores_path_shadowed_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _source_root, revision = _initialized_source_repo(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    execution_marker = tmp_path / "fake-git-executed"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"#!/bin/sh\ntouch {shlex.quote(str(execution_marker))}\nexit 99\n",
        encoding="utf-8",
    )
    fake_git.chmod(fake_git.stat().st_mode | os.X_OK)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")

    assert usd_cli_common.usd_cli_source_revision(repo) == revision
    assert not execution_marker.exists()


def test_usd_cli_execution_environment_drops_python_import_overrides() -> None:
    environment = usd_cli_common.sanitized_usd_cli_execution_env(
        {
            "KEEP_ME": "yes",
            "PATH": "/tmp/hostile-bin",
            "DYLD_FALLBACK_LIBRARY_PATH": "/tmp/foreign-dylibs",
            "DYLD_INSERT_LIBRARIES": "/tmp/injected.dylib",
            "LD_AUDIT": "/tmp/audit.so",
            "LD_LIBRARY_PATH": "/tmp/foreign-libs",
            "LD_PRELOAD": "/tmp/injected.so",
            "PXR_PLUGINPATH_NAME": "/tmp/foreign-plugins",
            "PYTHONEVIL": "import evil_module",
            "PYTHONHOME": "/tmp/foreign-home",
            "PYTHONINSPECT": "1",
            "PYTHONDONTWRITEBYTECODE": "0",
            "PYTHONPYCACHEPREFIX": "/tmp/foreign-pycache",
            "PYTHONNOUSERSITE": "0",
            "PYTHONPATH": "/tmp/foreign-modules",
            "PYTHONSAFEPATH": "0",
            "PYTHONSTARTUP": "/tmp/startup.py",
            "PYTHONUSERBASE": "/tmp/foreign-userbase",
            "PYTHONWARNINGS": "error::Warning:evil_module",
            "USD_CLI_NO_DAEMON": "1",
            "USD_PLUGIN_PATH": "/tmp/foreign-usd-plugins",
        }
    )

    assert environment["KEEP_ME"] == "yes"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONPYCACHEPREFIX"] == str(
        Path(os.devnull) / "content-agents-usd-cli"
    )
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONSAFEPATH"] == "1"
    for key in (
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONWARNINGS",
        "PYTHONEVIL",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PXR_PLUGINPATH_NAME",
        "3DSC_NO_DAEMON",
        "OV_NO_DAEMON",
        "USD_CLI_NO_DAEMON",
        "USD_PLUGIN_PATH",
    ):
        assert key not in environment
    assert environment["PYTHONPYCACHEPREFIX"] != "/tmp/foreign-pycache"
    # The caller-controlled search path never reaches a usd-cli subprocess.
    launch_path_entries = environment["PATH"].split(os.pathsep)
    assert "/tmp/hostile-bin" not in launch_path_entries
    assert launch_path_entries == [
        entry
        for entry in dict.fromkeys(os.defpath.split(os.pathsep))
        if entry and Path(entry).is_absolute()
    ]


def test_usd_cli_execution_environment_pins_path_to_launcher_directory(
    tmp_path: Path,
) -> None:
    launcher_dir = tmp_path / "venv" / "bin"
    launcher_dir.mkdir(parents=True)
    environment = usd_cli_common.sanitized_usd_cli_execution_env(
        {"PATH": "/tmp/hostile-bin:."},
        executable_dir=launcher_dir,
    )

    launch_path_entries = environment["PATH"].split(os.pathsep)
    assert launch_path_entries[0] == str(launcher_dir.resolve())
    assert "/tmp/hostile-bin" not in launch_path_entries
    assert "." not in launch_path_entries
    assert "" not in launch_path_entries
    assert all(Path(entry).is_absolute() for entry in launch_path_entries)


def test_successful_probe_uses_exact_package_route_and_persists_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _authorized_repo(tmp_path)
    route = _package_route(repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(
        usd_cli_backend,
        "resolve_package_owned_usd_cli_route",
        lambda _repo_root: route,
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object):
        calls.append(command)
        if "--version" in command:
            stdout = "usd-cli 1.0 (engine: usd, renderer: ovrtx)\n"
        else:
            probe_dir = Path(command[command.index("--output-dir") + 1])
            probe_dir.mkdir(parents=True, exist_ok=True)
            probe_path = probe_dir / "probe.png"
            Image.new("RGB", (64, 64), "red").save(probe_path)
            stdout = json.dumps(
                _probe(
                    render={
                        "path": str(probe_path),
                        "width": 64,
                        "height": 64,
                        "size_bytes": probe_path.stat().st_size,
                        "backend": "ovrtx",
                    }
                )
            )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        usd_cli_backend,
        "run_bounded_usd_cli_subprocess",
        fake_run,
    )

    readiness = usd_cli_backend.ensure_usd_cli_ovrtx_ready(
        repo,
        run_dir=run_dir,
        executable=route.target,
    )

    assert readiness.source_revision == _REVISION
    assert readiness.probe["ready"] is True
    assert readiness.artifact_path == run_dir / "raw" / "ovrtx_probe.json"
    evidence = json.loads(readiness.artifact_path.read_text(encoding="utf-8"))
    assert evidence["usd_cli_source_revision"] == _REVISION
    assert evidence["probe"]["engine"] == "ovrtx"
    assert (
        evidence["probe"]["render"]["size_bytes"]
        == Path(evidence["probe"]["render"]["path"]).stat().st_size
    )
    assert (
        evidence["probe"]["render"]["sha256"]
        == hashlib.sha256(
            Path(evidence["probe"]["render"]["path"]).read_bytes()
        ).hexdigest()
    )
    assert calls[0] == [str(route.target), "--version"]
    assert calls[1][:5] == [
        str(route.target),
        "--json",
        "render-probe",
        "--require-engine",
        "ovrtx",
    ]


def test_probe_rejects_executable_outside_exact_package_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _authorized_repo(tmp_path)
    route = _package_route(repo)
    foreign = tmp_path / "foreign-usd-cli"
    foreign.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    foreign.chmod(foreign.stat().st_mode | os.X_OK)
    monkeypatch.setattr(
        usd_cli_backend,
        "resolve_package_owned_usd_cli_route",
        lambda _repo_root: route,
    )

    with pytest.raises(RuntimeError, match="not the package-owned launcher"):
        usd_cli_backend.ensure_usd_cli_ovrtx_ready(repo, executable=foreign)


def test_readiness_retries_background_ovrtx_provisioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _authorized_repo(tmp_path)
    route = _package_route(repo)
    monkeypatch.setattr(
        usd_cli_backend,
        "resolve_package_owned_usd_cli_route",
        lambda _repo_root: route,
    )
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def fake_run(command: list[str], **_kwargs: object):
        calls.append(command)
        if "--version" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout="usd-cli 1.0\n", stderr=""
            )
        probe_attempt = len([call for call in calls if "render-probe" in call])
        if probe_attempt == 1:
            raise subprocess.CalledProcessError(
                1,
                command,
                stderr=(
                    "ovrtx auto-install STARTED in the background (~2.5 GB one-time "
                    "download)"
                ),
            )
        if probe_attempt == 2:
            raise subprocess.CalledProcessError(
                1,
                command,
                stderr="ovrtx auto-install in progress (5s elapsed)",
            )
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(_probe()), stderr=""
        )

    monkeypatch.setattr(
        usd_cli_backend,
        "run_bounded_usd_cli_subprocess",
        fake_run,
    )
    monkeypatch.setattr(usd_cli_backend.time, "sleep", sleeps.append)

    readiness = usd_cli_backend.ensure_usd_cli_ovrtx_ready(
        repo,
        executable=route.target,
        timeout_seconds=10,
    )

    assert readiness.probe["ready"] is True
    assert len([call for call in calls if "render-probe" in call]) == 3
    assert sleeps == [
        usd_cli_backend._OVRTX_PROVISIONING_POLL_SECONDS,
        usd_cli_backend._OVRTX_PROVISIONING_POLL_SECONDS,
    ]


def test_session_probe_prepares_private_raw_directory_before_packet_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _authorized_repo(tmp_path)
    route = _package_route(repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    session = usd_cli_backend.WorkflowUsdCliSession(
        project_dir=run_dir,
        session_id="workflow-test",
        route=route,
        workflow="materials-assign",
    )
    monkeypatch.setattr(
        usd_cli_backend,
        "resolve_package_owned_usd_cli_route",
        lambda _repo_root: route,
    )

    def fake_run(command: list[str], **_kwargs: object):
        assert command == [str(route.target), "--version"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="usd-cli 1.0 (engine: usd, renderer: ovrtx)\n",
            stderr="",
        )

    def fake_require_ovrtx(
        _session: usd_cli_backend.WorkflowUsdCliSession,
        output_dir: Path,
    ) -> dict[str, object]:
        assert stat.S_IMODE((run_dir / "raw").stat().st_mode) == 0o700
        assert (output_dir / ".workflow-owned.json").is_file()
        probe_path = output_dir / "probe.png"
        Image.new("RGB", (64, 64), "red").save(probe_path)
        return _probe(
            render={
                "path": str(probe_path),
                "width": 64,
                "height": 64,
                "size_bytes": probe_path.stat().st_size,
                "backend": "ovrtx",
            }
        )

    monkeypatch.setattr(usd_cli_backend, "run_bounded_usd_cli_subprocess", fake_run)
    monkeypatch.setattr(
        usd_cli_backend.WorkflowUsdCliSession,
        "require_ovrtx",
        fake_require_ovrtx,
    )

    readiness = usd_cli_backend.ensure_usd_cli_ovrtx_ready(
        repo,
        run_dir=run_dir,
        executable=route.target,
        session=session,
    )

    assert readiness.probe["ready"] is True
    assert readiness.artifact_path == run_dir / "raw" / "ovrtx_probe.json"


@pytest.mark.parametrize(
    "probe",
    [
        _probe(
            resolved_renderer="usdrecord",
            engine="usdrecord",
            ready=False,
            render=None,
        ),
        _probe(
            resolved_renderer="software",
            engine="software",
            ready=False,
            render=None,
        ),
        _probe(ready=False, render=None, error="GPU initialization failed"),
        _probe(transport="unknown"),
        _probe(capabilities=[]),
        _probe(
            transport="remote",
            render={
                "path": "/run/raw/ovrtx_probe/probe.png",
                "width": 64,
                "height": 64,
                "size_bytes": 128,
                "backend": "remote",
            },
        ),
        _probe(
            transport="remote",
            backends=[
                {
                    "url": "https://renderer.example.test",
                    "engine": "unknown",
                    "protocol_version": 2,
                    "status": "alive",
                }
            ],
            render={
                "path": "/run/raw/ovrtx_probe/probe.png",
                "width": 64,
                "height": 64,
                "size_bytes": 128,
                "backend": "remote",
            },
        ),
        _probe(
            render={
                "path": "/run/raw/ovrtx_probe/probe.png",
                "width": 64,
                "height": 64,
                "size_bytes": 128,
                "backend": "usdrecord",
            }
        ),
    ],
)
def test_unsupported_or_unready_renderer_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe: dict[str, object],
) -> None:
    repo = _authorized_repo(tmp_path)
    route = _package_route(repo)
    monkeypatch.setattr(
        usd_cli_backend,
        "resolve_package_owned_usd_cli_route",
        lambda _repo_root: route,
    )

    def fake_run(command: list[str], **_kwargs: object):
        stdout = "usd-cli 1.0\n" if "--version" in command else json.dumps(probe)
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        usd_cli_backend,
        "run_bounded_usd_cli_subprocess",
        fake_run,
    )

    with pytest.raises(RuntimeError):
        usd_cli_backend.ensure_usd_cli_ovrtx_ready(repo)
