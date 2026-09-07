# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the in-tree usd-cli server inside a parent-owned Windows Job Object."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from world_understanding.utils.artifacts import (
    delete_confined_file,
    open_confined_directory,
    open_confined_regular_file,
    write_bytes_to_confined,
)
from world_understanding.utils.windows_process import windows_process_start_token


def _validate_windows_request_path(
    config: object,
    roots: tuple[Path, ...],
    field: str,
    raw: object,
    *,
    capability: str = "allowed_roots",
) -> None:
    """Apply the server path capability without mistaking ``C:\\`` for a URI."""

    if raw in (None, ""):
        return
    if not isinstance(raw, str):
        raise ValueError(f"{field} must be a filesystem path string")
    if "\x00" in raw:
        raise ValueError(f"{field} contains a NUL byte")
    filesystem_value = raw.partition("[")[0] if "[" in raw else raw
    is_absolute_drive_path = bool(
        re.match(r"^[A-Za-z]:[\\/]", filesystem_value)
        and Path(filesystem_value).is_absolute()
    )
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw) and not is_absolute_drive_path:
        raise ValueError(
            f"{field} URI is not permitted by filesystem-only server.{capability}"
        )
    path = Path(filesystem_value).expanduser()
    project_dir = getattr(config, "project_dir", None)
    resolved = (
        path.resolve()
        if path.is_absolute()
        else (
            (Path(project_dir) if project_dir is not None else Path.cwd()) / path
        ).resolve()
    )
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError(f"{field} path is outside server.{capability}")


def _publish_server_state(
    config: object,
    payload: bytes,
    *,
    state_directory_descriptor: int | None = None,
    state_path: Path | None = None,
) -> Path:
    del state_directory_descriptor
    configured_state = Path(getattr(config, "state_dir"))
    publication_path = state_path or configured_state
    with open_confined_directory(publication_path, create=True) as root:
        write_bytes_to_confined(
            root,
            "server.json",
            payload,
            overwrite=True,
            file_mode=0o600,
        )
    return publication_path / "server.json"


def _remove_server_state_if_owned(
    config: object,
    *,
    daemon_pid: int,
    process_start_token: str,
    instance_id: str,
    state_directory_descriptor: int | None = None,
) -> None:
    del state_directory_descriptor
    state_dir = Path(getattr(config, "state_dir"))
    expected = {
        "pid": daemon_pid,
        "process_start_token": process_start_token,
        "instance_id": instance_id,
    }
    try:
        from usd_server.app import MAX_SERVER_STATE_BYTES

        with open_confined_directory(state_dir) as root:
            with open_confined_regular_file(root, "server.json") as (stream, _metadata):
                payload = stream.read(MAX_SERVER_STATE_BYTES + 1)
            if len(payload) > MAX_SERVER_STATE_BYTES:
                return
            state = json.loads(payload)
            if isinstance(state, dict) and all(
                state.get(field) == value for field, value in expected.items()
            ):
                delete_confined_file(root, "server.json")
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return


def main() -> int:
    if os.name != "nt":
        print("windows-usd-cli-daemon: Windows is required", file=sys.stderr)
        return 125

    from usd_cli import daemon
    from usd_core.config import load_config
    from usd_server import app

    config = load_config()
    pid = os.getpid()
    process_start_token = windows_process_start_token(pid)
    if process_start_token is None:
        raise RuntimeError("could not capture the usd-cli server process identity")

    # The upstream server's POSIX publication and /proc identity hooks are
    # replaced only in this isolated Windows daemon process. The package source
    # itself remains authenticated byte-for-byte against Git.
    daemon._proc_start_token = windows_process_start_token
    app._publish_server_state = _publish_server_state
    app._remove_server_state_if_owned = _remove_server_state_if_owned
    app._validate_request_path = _validate_windows_request_path

    with open_confined_directory(config.state_dir, create=True) as root:
        write_bytes_to_confined(
            root,
            "daemon.pids",
            f"{pid}:{process_start_token}\n".encode("ascii"),
            overwrite=True,
            file_mode=0o600,
        )
    app.serve(config, startup_identity=(pid, process_start_token))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
