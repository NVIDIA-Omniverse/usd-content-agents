# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the OVRTX protocol-service container entrypoint."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "apps/ovrtx_rendering_api/docker-entrypoint.sh"
DOCKERFILE = ROOT / "apps/ovrtx_rendering_api/Dockerfile"

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX shell process test")


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _entrypoint_env(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    return {
        "HOME": str(tmp_path),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PORT": "8765",
        "DISPLAY": ":9876",
    }


def test_dockerfile_creates_world_writable_x11_directory_before_user() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")

    assert "install -d -m 1777 /tmp/.X11-unix" in dockerfile
    assert dockerfile.index("install -d -m 1777 /tmp/.X11-unix") < dockerfile.index(
        "USER 10001:10001"
    )
    assert not any(
        line.strip().startswith("exec uvicorn") for line in entrypoint.splitlines()
    )


def test_entrypoint_keeps_uvicorn_as_child_and_forwards_shutdown(
    tmp_path: Path,
) -> None:
    env = _entrypoint_env(tmp_path)
    service_parent = tmp_path / "service-parent"
    service_terminated = tmp_path / "service-terminated"
    env["SERVICE_PARENT"] = str(service_parent)
    env["SERVICE_TERMINATED"] = str(service_terminated)

    _write_executable(
        tmp_path / "bin" / "Xvfb",
        "#!/usr/bin/env bash\n"
        'socket="/tmp/.X11-unix/X${DISPLAY#:}"\n'
        "mkdir -p /tmp/.X11-unix\n"
        'touch "$socket"\n'
        "trap 'rm -f \"$socket\"; exit 0' TERM INT\n"
        "while :; do sleep 1; done\n",
    )
    _write_executable(
        tmp_path / "bin" / "uvicorn",
        "#!/usr/bin/env bash\n"
        'echo "$PPID" > "$SERVICE_PARENT"\n'
        "trap 'touch \"$SERVICE_TERMINATED\"; exit 0' TERM INT\n"
        "while :; do sleep 1; done\n",
    )

    process = subprocess.Popen(["bash", str(ENTRYPOINT)], env=env)
    try:
        deadline = time.monotonic() + 5
        while not service_parent.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert service_parent.exists(), "uvicorn was not started"
        assert int(service_parent.read_text(encoding="utf-8")) == process.pid

        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 143
        assert service_terminated.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_entrypoint_returns_uvicorn_exit_status(tmp_path: Path) -> None:
    env = _entrypoint_env(tmp_path)
    _write_executable(
        tmp_path / "bin" / "Xvfb",
        "#!/usr/bin/env bash\n"
        "mkdir -p /tmp/.X11-unix\n"
        'touch "/tmp/.X11-unix/X${DISPLAY#:}"\n',
    )
    _write_executable(tmp_path / "bin" / "uvicorn", "#!/usr/bin/env bash\nexit 23\n")

    try:
        result = subprocess.run(
            ["bash", str(ENTRYPOINT)], env=env, check=False, timeout=5
        )
    finally:
        Path("/tmp/.X11-unix/X9876").unlink(missing_ok=True)

    assert result.returncode == 23
