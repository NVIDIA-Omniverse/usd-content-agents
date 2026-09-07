# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-Windows process identity coverage."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from world_understanding.utils.windows_process import (
    WindowsKillOnCloseJob,
    windows_process_is_live,
    windows_process_start_token,
)

pytestmark = pytest.mark.skipif(
    os.name != "nt",
    reason="requires native Windows process handles",
)


def test_start_token_rejects_exited_process_with_retained_handle() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    try:
        token = windows_process_start_token(process.pid)
        assert token is not None
        assert windows_process_is_live(process.pid) is True
        process.terminate()
        process.wait(timeout=5)

        assert windows_process_is_live(process.pid) is False
        assert windows_process_start_token(process.pid) is None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_kill_on_close_job_owns_suspended_process_lifecycle() -> None:
    ping = os.path.join(os.environ["SystemRoot"], "System32", "ping.exe")
    process: subprocess.Popen[bytes] | None = None
    job = WindowsKillOnCloseJob()
    try:
        process = subprocess.Popen(
            [ping, "-n", "30", "127.0.0.1"],
            creationflags=job.creation_flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        job.assign_process(process)
        assert job.active_process_count() == 1
        job.resume_process(process)

        job.terminate(exit_code=7)
        job.wait_for_empty(timeout_s=5.0)
        assert job.active_process_count() == 0
        job.terminate(exit_code=7)
    finally:
        job.close()
        job.close()
        if process is not None:
            process.wait(timeout=5)


def test_kill_on_close_job_rejects_queries_after_close() -> None:
    job = WindowsKillOnCloseJob()
    job.close()

    with pytest.raises(RuntimeError, match="Job Object is closed"):
        job.active_process_count()
