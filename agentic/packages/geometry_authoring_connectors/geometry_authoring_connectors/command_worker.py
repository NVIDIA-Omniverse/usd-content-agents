# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-command adapter for independently isolated authoring workers."""

from __future__ import annotations

import json
import os
import re
import signal
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import ValidationError

from .build123d_worker import (
    ExternalAuthoringWorkerResult,
    WorkerIsolationKind,
)
from .errors import (
    InvalidProviderResponseError,
    ProviderUnavailableError,
    WorkerIsolationError,
)
from .models import AuthoringRequest

_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_RESERVED_ENVIRONMENT_NAMES = frozenset({"HOME", "LANG", "LC_ALL", "TMPDIR"})
_MAX_RESULT_MANIFEST_BYTES = 4 * 1024 * 1024


class CommandAuthoringExecutionBackend:
    """Invoke one operator-fixed provider command without a shell.

    The command runs only inside the worker deployment. Geometry Agent requests
    cannot select or modify its argv, environment, working directory, or result
    path. Container or sandbox enforcement remains a deployment prerequisite.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        isolation_kind: WorkerIsolationKind,
        timeout_seconds: float = 600.0,
        environment: Mapping[str, str] | None = None,
        provider_id: str = "external-authoring-worker",
    ) -> None:
        if os.name != "posix":
            raise ValueError("command authoring workers require POSIX process isolation")
        argv = tuple(command)
        if not argv or len(argv) > 64:
            raise ValueError("authoring worker command must contain 1 to 64 arguments")
        executable = Path(argv[0]).expanduser()
        if not executable.is_absolute():
            raise ValueError("authoring worker executable must use an absolute path")
        if (
            not executable.is_file()
            or executable.is_symlink()
            or not os.access(executable, os.X_OK)
        ):
            raise ValueError("authoring worker executable must be an executable regular file")
        if any(
            not item
            or len(item) > 4_096
            or "\x00" in item
            or any(ord(character) < 32 for character in item)
            for item in argv
        ):
            raise ValueError("authoring worker argv contains an invalid value")
        if isolation_kind not in {"container", "sandboxed-process"}:
            raise ValueError("command workers require container or sandboxed-process isolation")
        if not 1.0 <= timeout_seconds <= 7_200.0:
            raise ValueError("authoring worker timeout must be between 1 and 7,200 seconds")
        configured_environment = dict(environment or {})
        for name, value in configured_environment.items():
            if _ENVIRONMENT_NAME.fullmatch(name) is None:
                raise ValueError("authoring worker environment names must be bounded uppercase IDs")
            if len(value) > 16_384 or "\x00" in value:
                raise ValueError("authoring worker environment values must be bounded")
        if _RESERVED_ENVIRONMENT_NAMES.intersection(configured_environment):
            raise ValueError("authoring worker environment cannot override isolation-owned values")

        self._command = (str(executable.resolve(strict=True)), *argv[1:])
        self._isolation_kind = isolation_kind
        self._timeout_seconds = timeout_seconds
        self._environment = configured_environment
        self._provider_id = provider_id

    @property
    def isolation_kind(self) -> WorkerIsolationKind:
        return self._isolation_kind

    def execute(
        self,
        request: AuthoringRequest,
        *,
        workspace: Path,
    ) -> ExternalAuthoringWorkerResult:
        raw_workspace = Path(os.path.abspath(workspace.expanduser()))
        workspace = raw_workspace.resolve(strict=True)
        if raw_workspace != workspace or workspace.is_symlink() or not workspace.is_dir():
            raise ValueError("authoring worker workspace must be a direct regular directory")
        output_dir = workspace / "output"
        try:
            output_dir.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise WorkerIsolationError("authoring worker output path must not pre-exist") from exc
        request_path = workspace / "authoring-request.json"
        result_path = workspace / "authoring-result.json"
        with request_path.open("x", encoding="utf-8") as stream:
            stream.write(request.model_dump_json(indent=2))
            stream.write("\n")
        request_path.chmod(0o600)

        environment = {
            "HOME": str(workspace),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TMPDIR": str(workspace),
            **self._environment,
        }
        argv = (
            *self._command,
            "--request",
            str(request_path),
            "--output-dir",
            str(output_dir),
            "--result",
            str(result_path),
        )
        process = subprocess.Popen(  # noqa: S603 - fixed operator-owned absolute executable
            argv,
            cwd=workspace,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=self._timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            raise ProviderUnavailableError(
                "isolated authoring command exceeded its configured timeout",
                provider_id=self._provider_id,
            ) from exc
        _terminate_process_group(process)
        if return_code != 0:
            raise ProviderUnavailableError(
                "isolated authoring command failed without exposing process output",
                provider_id=self._provider_id,
            )
        try:
            descriptor = os.open(
                result_path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
        except OSError as exc:
            raise InvalidProviderResponseError(
                "isolated authoring command omitted its typed result manifest",
                provider_id=self._provider_id,
            ) from exc
        try:
            file_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_nlink != 1
                or file_stat.st_size > _MAX_RESULT_MANIFEST_BYTES
            ):
                raise InvalidProviderResponseError(
                    "isolated authoring result manifest is not a bounded regular file",
                    provider_id=self._provider_id,
                )
            payload = os.read(descriptor, _MAX_RESULT_MANIFEST_BYTES + 1)
            after_stat = os.fstat(descriptor)
            before_identity = (
                file_stat.st_dev,
                file_stat.st_ino,
                file_stat.st_mode,
                file_stat.st_nlink,
                file_stat.st_size,
                file_stat.st_mtime_ns,
                file_stat.st_ctime_ns,
            )
            after_identity = (
                after_stat.st_dev,
                after_stat.st_ino,
                after_stat.st_mode,
                after_stat.st_nlink,
                after_stat.st_size,
                after_stat.st_mtime_ns,
                after_stat.st_ctime_ns,
            )
            if len(payload) != file_stat.st_size or before_identity != after_identity:
                raise InvalidProviderResponseError(
                    "isolated authoring result manifest changed while being read",
                    provider_id=self._provider_id,
                )
        finally:
            os.close(descriptor)
        try:
            document = json.loads(payload)
            return ExternalAuthoringWorkerResult.model_validate(document)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
            raise InvalidProviderResponseError(
                "isolated authoring command returned an invalid typed result manifest",
                provider_id=self._provider_id,
            ) from exc


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


__all__ = ["CommandAuthoringExecutionBackend"]
