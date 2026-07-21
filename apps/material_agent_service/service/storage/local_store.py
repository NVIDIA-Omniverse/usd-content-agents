# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, TypeVar

from world_understanding.utils.artifacts import (
    ArtifactPathError,
    append_bytes_to_confined,
    confined_artifact_exists,
    copy_open_file_to_confined,
    delete_confined_file,
    is_pipeline_temp_path,
    iter_open_regular_files,
    list_confined_artifact_keys,
    open_confined_directory,
    open_confined_directory_at,
    open_confined_lock_file,
    open_confined_regular_file,
    open_held_confined_artifact,
    open_regular_file_no_follow,
    read_confined_artifact_bytes,
    remove_confined_tree,
    write_bytes_to_confined,
)
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)
from world_understanding.utils.session_paths import (
    confined_session_path,
    confined_storage_child_path,
    is_safe_session_id,
    validated_storage_child_name,
)

from ..json_utils import to_json_safe
from .base import (
    JsonPreconditionError,
    SessionMetadataContentionError,
    SessionStore,
    VersionedJson,
)

logger = logging.getLogger(__name__)

_JSON_LOCK_TIMEOUT_SECONDS = 10.0
_T = TypeVar("_T")


class LocalSessionStore(SessionStore):
    """Single-instance session storage backed by one local artifact tree.

    The JSON compare-and-swap operations are cross-process safe, but pipeline
    workers still write mutable canonical files in the same session directory.
    Therefore a shared LocalSessionStore/PVC is not a supported multi-instance
    deployment or expired-lease takeover topology.  Multi-instance deployments
    require S3 metadata/artifacts and an isolated pod-local working directory.
    """

    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)

    @property
    def kind(self) -> str:
        return "local"

    def _session_dir(self, session_id: str) -> Path:
        if is_safe_session_id(session_id):
            return confined_session_path(self.root, session_id)
        return confined_storage_child_path(self.root, session_id)

    def _read_session_dir(self, session_id: str) -> Path | None:
        """Resolve a read root, treating an escaped UUID alias as absent."""
        try:
            return self._session_dir(session_id)
        except ValueError:
            return None

    @contextmanager
    def _json_lock(self, session_id: str, key: str) -> Iterator[int]:
        """Hold a descriptor-confined cross-process metadata lock and root."""

        validated_storage_child_name(session_id)
        key_digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        lock_key = f".locks/{session_id}/{key_digest}.lock"
        with open_confined_directory(self.root, create=True) as root_descriptor:
            with open_confined_lock_file(root_descriptor, lock_key) as lock_descriptor:
                deadline = time.monotonic() + _JSON_LOCK_TIMEOUT_SECONDS
                while True:
                    try:
                        fcntl.flock(
                            lock_descriptor,
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                        break
                    except BlockingIOError as exc:
                        if time.monotonic() >= deadline:
                            raise SessionMetadataContentionError(
                                "Session metadata is temporarily busy"
                            ) from exc
                        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
                try:
                    yield root_descriptor
                finally:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)

    @contextmanager
    def _locked_json(self, session_id: str, key: str) -> Iterator[int]:
        """Acquire a bounded metadata lock and expose retryable contention."""
        with self._json_lock(session_id, key) as root_descriptor:
            yield root_descriptor

    async def _run_locked_io(
        self,
        operation: Callable[..., _T],
        *args: Any,
    ) -> _T:
        """Run one locked critical section off-loop and drain it on cancellation."""
        worker = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # The thread cannot be cancelled. Preserve the old uninterruptible
            # critical-section contract by waiting until it commits or fails;
            # cancellation remains authoritative for the caller. Keep the
            # worker shielded while draining so repeated cancellation requests
            # cannot orphan the in-flight filesystem mutation.
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            with suppress(Exception, asyncio.CancelledError):
                worker.result()
            raise

    @staticmethod
    def _json_version(data: bytes) -> str:
        return f"sha256:{hashlib.sha256(data).hexdigest()}"

    @staticmethod
    def _encode_json(obj: dict) -> bytes:
        return json.dumps(to_json_safe(obj)).encode("utf-8")

    async def init_session(self, session_id: str) -> None:
        self._session_dir(session_id)
        with open_confined_directory(self.root, create=True) as root_descriptor:
            with open_confined_directory_at(
                root_descriptor,
                session_id,
                create=True,
            ):
                pass

    async def delete_session(self, session_id: str) -> None:
        for attempt in range(3):
            try:
                remove_confined_tree(self._session_dir(session_id), self.root)
                return
            except OSError:
                if attempt == 2:
                    raise
                log_durable_failure(
                    logger,
                    "session_local_delete_retry_failed",
                    phase=FailurePhase.ROLLBACK,
                    retryable=True,
                )
                await asyncio.sleep(0.5 * (attempt + 1))

    async def list_sessions(self, use_cache: bool = True) -> list[str]:
        """List all session IDs in the local store.

        Args:
            use_cache: Ignored for local storage (filesystem is fast,
                       no caching needed)

        Returns:
            List of session IDs (directory names that contain session.json)
        """
        sessions: list[str] = []
        if not self.root.exists():
            return sessions

        for listed_path in self.root.iterdir():
            try:
                session_dir = self._session_dir(listed_path.name)
            except ValueError:
                continue
            if confined_artifact_exists(session_dir, "session.json"):
                sessions.append(listed_path.name)

        return sessions

    def invalidate_sessions_cache(self) -> None:
        """No-op for local storage - no caching needed."""
        pass

    async def put_bytes(
        self, session_id: str, key: str, data: bytes, content_type: str | None = None
    ) -> None:
        del content_type
        with open_confined_directory(
            self._session_dir(session_id),
            create=True,
        ) as destination_descriptor:
            write_bytes_to_confined(destination_descriptor, key, data)

    async def put_file(
        self, session_id: str, key: str, file_path: str, content_type: str | None = None
    ) -> None:
        del content_type
        with open_regular_file_no_follow(file_path) as (source, metadata):
            with open_confined_directory(
                self._session_dir(session_id),
                create=True,
            ) as destination_descriptor:
                copy_open_file_to_confined(
                    destination_descriptor,
                    key,
                    source,
                    metadata,
                    overwrite=True,
                )

    async def delete_file(self, session_id: str, key: str) -> None:
        try:
            with open_confined_directory(
                self._session_dir(session_id),
            ) as session_descriptor:
                delete_confined_file(session_descriptor, key)
        except FileNotFoundError:
            pass

    async def open_read(self, session_id: str, key: str):
        base = self._read_session_dir(session_id)
        if base is None:
            raise FileNotFoundError(key)
        try:
            return open_held_confined_artifact(base, key).stream
        except (
            ArtifactPathError,
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
        ):
            raise FileNotFoundError(key) from None

    async def iter_read(
        self,
        session_id: str,
        key: str,
        *,
        chunk_size: int,
    ) -> AsyncIterator[bytes]:
        """Yield bounded chunks from a held local artifact."""
        stream = await self.open_read(session_id, key)
        try:
            while chunk := await asyncio.to_thread(stream.read, chunk_size):
                yield chunk
        finally:
            stream.close()

    async def exists(self, session_id: str, key: str) -> bool:
        base = self._read_session_dir(session_id)
        return base is not None and confined_artifact_exists(base, key)

    async def list_keys(self, session_id: str, prefix: str = "") -> list[str]:
        base = self._read_session_dir(session_id)
        if base is None or is_pipeline_temp_path(prefix):
            return []
        return list_confined_artifact_keys(base, prefix=prefix)

    async def put_json(self, session_id: str, key: str, obj: dict) -> None:
        data = self._encode_json(obj)
        await self._run_locked_io(self._put_json_locked, session_id, key, data)

    def _put_json_locked(self, session_id: str, key: str, data: bytes) -> None:
        """Write JSON while holding the complete critical section off-loop."""
        with self._locked_json(session_id, key) as root_descriptor:
            with open_confined_directory_at(
                root_descriptor,
                session_id,
                create=True,
            ) as session_descriptor:
                write_bytes_to_confined(session_descriptor, key, data)

    async def get_json(self, session_id: str, key: str) -> dict | None:
        return (await self.get_json_versioned(session_id, key)).value

    async def get_json_versioned(self, session_id: str, key: str) -> VersionedJson:
        if is_pipeline_temp_path(key):
            return VersionedJson(value=None, version=None)
        return await self._run_locked_io(
            self._get_json_versioned_locked,
            session_id,
            key,
        )

    def _get_json_versioned_locked(
        self,
        session_id: str,
        key: str,
    ) -> VersionedJson:
        """Read and version JSON in one off-loop locked critical section."""
        with self._locked_json(session_id, key) as root_descriptor:
            try:
                with open_confined_directory_at(
                    root_descriptor,
                    session_id,
                ) as session_descriptor:
                    with open_confined_regular_file(
                        session_descriptor,
                        key,
                    ) as (stream, _metadata):
                        data = stream.read()
            except (ArtifactPathError, FileNotFoundError):
                return VersionedJson(value=None, version=None)
            return VersionedJson(
                value=json.loads(data),
                version=self._json_version(data),
            )

    async def replace_json_if_version(
        self,
        session_id: str,
        key: str,
        obj: dict,
        expected_version: str | None,
    ) -> str:
        data = self._encode_json(obj)
        return await self._run_locked_io(
            self._replace_json_if_version_locked,
            session_id,
            key,
            data,
            expected_version,
        )

    def _replace_json_if_version_locked(
        self,
        session_id: str,
        key: str,
        data: bytes,
        expected_version: str | None,
    ) -> str:
        """Compare and replace JSON in one off-loop locked critical section."""
        with self._locked_json(session_id, key) as root_descriptor:
            with open_confined_directory_at(
                root_descriptor,
                session_id,
                create=True,
            ) as session_descriptor:
                try:
                    with open_confined_regular_file(
                        session_descriptor,
                        key,
                    ) as (stream, _metadata):
                        current_version = self._json_version(stream.read())
                except FileNotFoundError:
                    current_version = None
                if current_version != expected_version:
                    raise JsonPreconditionError("JSON version changed")
                write_bytes_to_confined(session_descriptor, key, data)
        return self._json_version(data)

    async def get_json_batch(
        self, session_ids: list[str], key: str
    ) -> list[dict | None]:
        return [await self.get_json(sid, key) for sid in session_ids]

    async def append_event(self, session_id: str, event: dict) -> None:
        line = (json.dumps(to_json_safe(event)) + "\n").encode("utf-8")
        with open_confined_directory(
            self._session_dir(session_id),
            create=True,
        ) as session_descriptor:
            append_bytes_to_confined(session_descriptor, "events.jsonl", line)

    async def get_event_log(self, session_id: str) -> list[dict]:
        base = self._read_session_dir(session_id)
        if base is None:
            return []
        try:
            text = read_confined_artifact_bytes(base, "events.jsonl").decode("utf-8")
            return [json.loads(line) for line in text.splitlines()]
        except (
            ArtifactPathError,
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
        ):
            return []

    async def make_public_url(
        self, session_id: str, key: str, expires_seconds: int = 3600
    ) -> str | None:
        return None

    async def sync_from_local(
        self, session_id: str, local_session_dir: str, prefix: str = ""
    ) -> int:
        """No-op for local storage - files are already local.

        Args:
            session_id: Session identifier
            local_session_dir: Path to local session directory
            prefix: Optional prefix to filter files

        Returns:
            0 (no files synced, already local)
        """
        return 0

    async def sync_to_local(
        self, session_id: str, local_session_dir: str, prefix: str = ""
    ) -> int:
        """Copy files from store to local dir (no-op if they are the same path).

        Args:
            session_id: Session identifier
            local_session_dir: Path to local session directory
            prefix: Optional prefix to filter files (e.g., "input/")

        Returns:
            Number of files copied
        """
        store_dir = self._read_session_dir(session_id)
        if store_dir is None:
            return 0
        try:
            with open_confined_directory(store_dir) as source_descriptor:
                with open_confined_directory(
                    local_session_dir,
                    create=True,
                ) as destination_descriptor:
                    source_identity = os.fstat(source_descriptor)
                    destination_identity = os.fstat(destination_descriptor)
                    if (
                        source_identity.st_dev == destination_identity.st_dev
                        and source_identity.st_ino == destination_identity.st_ino
                    ):
                        return 0
                    count = 0
                    for artifact in iter_open_regular_files(
                        source_descriptor,
                        prefix=prefix,
                    ):
                        if copy_open_file_to_confined(
                            destination_descriptor,
                            artifact.relative_key,
                            artifact.stream,
                            artifact.metadata,
                            overwrite=False,
                        ):
                            count += 1
                    return count
        except FileNotFoundError:
            return 0

    async def cleanup_stale_local_sessions(
        self, local_storage_path: str, max_age_hours: float = 24.0
    ) -> int:
        """No-op for local storage - no remote sync needed.

        Args:
            local_storage_path: Root path where local sessions are stored
            max_age_hours: Maximum age in hours (ignored)

        Returns:
            0 (no cleanup needed for local storage)
        """
        return 0
