# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wrapper-owned loopback broker for one child agent's observation memory."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

from content_agent_workflows.common.memory import (
    MAX_INSPECT_BYTES,
    AgentMemory,
    MemoryArtifactInput,
    MemorySearchQuery,
    RememberRequest,
)
from content_agent_workflows.common.memory import (
    MemoryError as AgentMemoryError,
)
from pydantic import ValidationError
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    open_confined_directory,
    open_confined_regular_file,
)

_MAX_REQUEST_BYTES = 4 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_SAFE_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,8}")
MEMORY_ORIGIN_AGENT_TAG = "memory-origin:agent"
MEMORY_ORIGIN_LAUNCHER_TAG = "memory-origin:launcher"
_KNOWN_COMMANDS = frozenset(
    {"init", "record", "context", "search", "inspect", "pin", "unpin"}
)


class MemoryBrokerError(RuntimeError):
    """Protocol or trust-boundary failure with an HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


class AgentMemoryBroker:
    """Expose bounded memory operations without exposing the managed store.

    The child can name artifacts only beneath ``run_dir``. The broker pins the
    run directory once, traverses every artifact with ``openat`` and
    ``O_NOFOLLOW``, and snapshots it into wrapper-private storage before
    calling :class:`AgentMemory`.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        memory: AgentMemory,
        private_dir: Path | None = None,
        search_ready_path: Path | None = None,
    ) -> None:
        self.run_dir = Path(os.path.abspath(run_dir))
        self.memory = memory
        self._search_ready_path = (
            Path(os.path.abspath(search_ready_path))
            if search_ready_path is not None
            else None
        )
        if (
            self._search_ready_path is not None
            and not self._search_ready_path.is_relative_to(self.run_dir)
        ):
            raise ValueError(
                "search_ready_path must be beneath the child run directory"
            )
        memory_run_dir = Path(os.path.abspath(memory.run_dir))
        if memory_run_dir == self.run_dir or memory_run_dir.is_relative_to(
            self.run_dir
        ):
            raise ValueError(
                "agent memory store must be outside the child-writable run directory"
            )
        self._run_dir_context = open_confined_directory(self.run_dir)
        self._run_dir_root = self._run_dir_context.__enter__()
        if private_dir is None:
            self._private_dir = Path(tempfile.mkdtemp(prefix="agent-memory-broker-"))
            self._private_dir_owned = True
        else:
            self._private_dir = Path(private_dir).resolve()
            self._private_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self._private_dir, 0o700)
            self._private_dir_owned = False
        if self._private_dir == self.run_dir or self._private_dir.is_relative_to(
            self.run_dir
        ):
            self._run_dir_context.__exit__(None, None, None)
            if self._private_dir_owned:
                shutil.rmtree(self._private_dir)
            raise ValueError(
                "memory broker private_dir must be outside the child-writable run "
                "directory"
            )
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._operation_counts: dict[str, int] = {}
        self._operation_counts_lock = threading.Lock()
        self._closed = False

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("Memory broker is not started.")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def operation_counts(self) -> dict[str, int]:
        """Return a stable snapshot of child broker operations."""

        with self._operation_counts_lock:
            return dict(self._operation_counts)

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Memory broker is closed.")
        if self._server is not None:
            raise RuntimeError("Memory broker is already started.")
        broker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *args: Any) -> None:
                del args

            def _send(self, status: int, payload: Any) -> None:
                body = json.dumps(_json_value(payload), sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError as exc:
                    raise MemoryBrokerError(400, "invalid Content-Length") from exc
                if length <= 0 or length > _MAX_REQUEST_BYTES:
                    raise MemoryBrokerError(
                        413,
                        f"request body must be between 1 and {_MAX_REQUEST_BYTES} bytes",
                    )
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise MemoryBrokerError(400, f"invalid JSON body: {exc}") from exc
                if not isinstance(payload, dict):
                    raise MemoryBrokerError(400, "JSON body must be an object")
                return payload

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    self._send(200, {"status": "ok", "run_id": broker.memory.run_id})
                    return
                self._send(404, {"error": f"unknown path {self.path}"})

            def do_POST(self) -> None:  # noqa: N802
                try:
                    parts = [part for part in self.path.split("/") if part]
                    if len(parts) != 2 or parts[0] != "v1":
                        raise MemoryBrokerError(404, f"unknown path {self.path}")
                    payload = self._read_json()
                    self._send(200, broker.dispatch(parts[1], payload))
                except MemoryBrokerError as exc:
                    self._send(exc.status, {"error": str(exc)})
                except (ValidationError, KeyError, ValueError) as exc:
                    self._send(400, {"error": str(exc)[:1000]})
                except AgentMemoryError as exc:
                    self._send(400, {"error": str(exc)[:1000]})
                except Exception as exc:  # noqa: BLE001 - report, do not crash
                    self._send(
                        500,
                        {"error": f"{type(exc).__name__}: internal broker failure"},
                    )

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="agent-memory-broker",
            daemon=True,
        )
        self._server_thread.start()
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if response.status != 200 or payload.get("status") != "ok":
                    raise RuntimeError("Memory broker returned an unhealthy response.")
        except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
            self.close()
            raise RuntimeError(
                "Memory broker failed its startup health check."
            ) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
            self._server_thread = None
        self._run_dir_context.__exit__(None, None, None)
        if self._private_dir_owned:
            shutil.rmtree(self._private_dir, ignore_errors=True)

    def dispatch(self, command: str, payload: dict[str, Any]) -> Any:
        if payload.get("run_id") != self.memory.run_id:
            raise MemoryBrokerError(403, "run_id does not match this memory broker")
        counted_command = command if command in _KNOWN_COMMANDS else "unknown"
        with self._operation_counts_lock:
            self._operation_counts[counted_command] = (
                self._operation_counts.get(counted_command, 0) + 1
            )
        if command == "init":
            return {
                "run_id": self.memory.run_id,
                "access": "launcher_broker",
            }
        if command == "record":
            return self._record(payload)
        if command == "context":
            return self.memory.context(limit=int(payload.get("limit", 12)))
        if command == "search":
            if (
                self._search_ready_path is not None
                and not self._search_ready_path.is_file()
            ):
                raise MemoryBrokerError(
                    409,
                    "memory search is unavailable until the initial part plan exists",
                )
            raw_query = payload.get("query")
            if not isinstance(raw_query, dict):
                raise MemoryBrokerError(400, "search requires a query object")
            return self.memory.search(MemorySearchQuery.model_validate(raw_query))
        if command == "inspect":
            return self._inspect(payload)
        if command in {"pin", "unpin"}:
            observation_id = payload.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id:
                raise MemoryBrokerError(400, f"{command} requires observation_id")
            operation = self.memory.pin if command == "pin" else self.memory.unpin
            return operation(observation_id)
        raise MemoryBrokerError(404, f"unknown memory operation: {command}")

    def _record(self, payload: dict[str, Any]) -> Any:
        raw_request = payload.get("request")
        if not isinstance(raw_request, dict):
            raise MemoryBrokerError(400, "record requires a request object")
        request = RememberRequest.model_validate(raw_request)
        return self.remember(request, origin="agent")

    def remember(
        self,
        request: RememberRequest,
        *,
        origin: Literal["agent", "launcher"] = "launcher",
    ) -> Any:
        """Snapshot run artifacts safely, then record a trusted observation."""

        snapshots: list[Path] = []
        try:
            artifacts: list[MemoryArtifactInput] = []
            for artifact in request.artifacts:
                snapshot = self._snapshot_run_artifact(artifact.path)
                snapshots.append(snapshot)
                artifacts.append(artifact.model_copy(update={"path": snapshot}))
            origin_tag = (
                MEMORY_ORIGIN_AGENT_TAG
                if origin == "agent"
                else MEMORY_ORIGIN_LAUNCHER_TAG
            )
            reserved_origin_tags = {
                MEMORY_ORIGIN_AGENT_TAG,
                MEMORY_ORIGIN_LAUNCHER_TAG,
            }
            caller_tags = tuple(
                tag for tag in request.tags if tag not in reserved_origin_tags
            )
            safe_request = request.model_copy(
                update={
                    "artifacts": tuple(artifacts),
                    "tags": tuple(dict.fromkeys((*caller_tags, origin_tag))),
                }
            )
            return self.memory.remember(safe_request)
        finally:
            for snapshot in snapshots:
                snapshot.unlink(missing_ok=True)

    def _inspect(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation_ids = payload.get("observation_ids")
        artifact_roles = payload.get("artifact_roles", [])
        if not isinstance(observation_ids, list) or not all(
            isinstance(item, str) for item in observation_ids
        ):
            raise MemoryBrokerError(400, "inspect requires observation_ids")
        if not isinstance(artifact_roles, list) or not all(
            isinstance(item, str) for item in artifact_roles
        ):
            raise MemoryBrokerError(400, "artifact_roles must be a string list")
        requested_max_bytes = int(payload.get("max_bytes", MAX_INSPECT_BYTES))
        if requested_max_bytes <= 0:
            raise MemoryBrokerError(400, "inspect max_bytes must be positive")
        result = self.memory.inspect(
            observation_ids,
            artifact_roles=artifact_roles,
            max_bytes=min(requested_max_bytes, MAX_INSPECT_BYTES),
        )
        try:
            response = result.model_dump(mode="json")
            encoded_artifacts: list[dict[str, Any]] = []
            for artifact in result.artifacts:
                content = artifact.path.read_bytes()
                item = artifact.model_dump(mode="json")
                item.pop("path", None)
                item["filename"] = artifact.path.name
                item["content_base64"] = base64.b64encode(content).decode("ascii")
                encoded_artifacts.append(item)
            response["artifacts"] = encoded_artifacts
            return response
        finally:
            if result.lease_id is not None:
                shutil.rmtree(
                    self.memory.cache_root / result.lease_id, ignore_errors=True
                )

    def _snapshot_run_artifact(self, source: Path) -> Path:
        lexical = Path(os.path.abspath(source))
        try:
            relative = lexical.relative_to(self.run_dir)
        except ValueError as exc:
            raise MemoryBrokerError(
                400, f"memory artifact is outside the child run directory: {source}"
            ) from exc
        if not relative.parts:
            raise MemoryBrokerError(400, "memory artifact must name a file")
        temporary_fd = -1
        temporary_path: Path | None = None
        snapshot_complete = False
        try:
            with open_confined_regular_file(
                self._run_dir_root,
                relative.as_posix(),
            ) as (source_stream, before):
                source_fd = source_stream.fileno()
                if not stat.S_ISREG(before.st_mode):
                    raise MemoryBrokerError(
                        400, "memory artifact must be a regular file"
                    )
                if before.st_size > self.memory.max_artifact_bytes:
                    raise MemoryBrokerError(
                        413, "memory artifact exceeds the size limit"
                    )
                suffix = (
                    lexical.suffix if _SAFE_SUFFIX.fullmatch(lexical.suffix) else ".bin"
                )
                temporary_fd, temporary_name = tempfile.mkstemp(
                    prefix="artifact-",
                    suffix=suffix.lower(),
                    dir=self._private_dir,
                )
                temporary_path = Path(temporary_name)
                copied = 0
                copied_digest = hashlib.sha256()
                while chunk := os.read(source_fd, _COPY_CHUNK_BYTES):
                    copied += len(chunk)
                    if copied > self.memory.max_artifact_bytes:
                        raise MemoryBrokerError(
                            413, "memory artifact exceeds the size limit"
                        )
                    copied_digest.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        view = view[os.write(temporary_fd, view) :]
                os.fsync(temporary_fd)
                after = os.fstat(source_fd)
                if (
                    before.st_size != after.st_size
                    or before.st_mtime_ns != after.st_mtime_ns
                    or before.st_ctime_ns != after.st_ctime_ns
                ):
                    raise MemoryBrokerError(
                        409, "memory artifact changed while reading"
                    )
                os.lseek(source_fd, 0, os.SEEK_SET)
                verified_size = 0
                verified_digest = hashlib.sha256()
                while chunk := os.read(source_fd, _COPY_CHUNK_BYTES):
                    verified_size += len(chunk)
                    if verified_size > self.memory.max_artifact_bytes:
                        raise MemoryBrokerError(
                            409, "memory artifact changed while reading"
                        )
                    verified_digest.update(chunk)
                verified_after = os.fstat(source_fd)
                if (
                    verified_size != copied
                    or verified_digest.digest() != copied_digest.digest()
                    or before.st_size != verified_after.st_size
                    or before.st_mtime_ns != verified_after.st_mtime_ns
                    or before.st_ctime_ns != verified_after.st_ctime_ns
                ):
                    raise MemoryBrokerError(
                        409, "memory artifact changed while reading"
                    )
            os.close(temporary_fd)
            temporary_fd = -1
            snapshot_complete = True
            return temporary_path
        except (ArtifactPathError, OSError) as exc:
            raise MemoryBrokerError(
                400, f"could not safely open memory artifact: {source}"
            ) from exc
        finally:
            if temporary_fd >= 0:
                os.close(temporary_fd)
            if temporary_path is not None and not snapshot_complete:
                temporary_path.unlink(missing_ok=True)


__all__ = ["AgentMemoryBroker", "MemoryBrokerError"]
