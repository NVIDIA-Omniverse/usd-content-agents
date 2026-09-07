# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP responses backed by descriptor-confined regular files."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from secrets import token_hex
from typing import BinaryIO

import anyio
from starlette.background import BackgroundTask
from starlette.datastructures import MutableHeaders
from starlette.responses import FileResponse
from starlette.types import Receive, Scope, Send

from world_understanding.utils.artifacts import (
    OpenArtifactFile,
    open_held_confined_artifact,
)


def open_held_artifact_file(
    storage_root: str | Path,
    relative_key: str,
) -> OpenArtifactFile:
    """Open one confined file and return an independently held descriptor."""
    return open_held_confined_artifact(storage_root, relative_key)


class HeldFileResponse(FileResponse):
    """Serve an already-open regular file without reopening its pathname."""

    _snapshot_memory_limit = 8 * 1024 * 1024

    def __init__(
        self,
        artifact: OpenArtifactFile,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        media_type: str | None = None,
        background: BackgroundTask | None = None,
        filename: str | None = None,
        content_disposition_type: str = "attachment",
    ) -> None:
        self._stream: BinaryIO = artifact.stream
        self._descriptor = artifact.stream.fileno()
        self._metadata = artifact.metadata
        self._snapshot: BinaryIO | None = None
        super().__init__(
            artifact.relative_key,
            status_code=status_code,
            headers=headers,
            media_type=media_type,
            background=background,
            filename=filename,
            stat_result=artifact.metadata,
            content_disposition_type=content_disposition_type,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await self._ensure_snapshot()
            await super().__call__(scope, receive, send)
        finally:
            if self._snapshot is not None:
                self._snapshot.close()
            self._stream.close()

    async def _read_at(self, size: int, offset: int) -> bytes:
        def read() -> bytes:
            self._stream.seek(offset, os.SEEK_SET)
            return self._stream.read(size)

        return await anyio.to_thread.run_sync(read)

    @staticmethod
    def _metadata_signature(metadata: os.stat_result) -> tuple[int, int, int, int]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )

    async def _ensure_snapshot(self) -> None:
        if self._snapshot is not None:
            return
        snapshot = tempfile.SpooledTemporaryFile(
            max_size=self._snapshot_memory_limit,
            mode="w+b",
        )
        digest = hashlib.sha256()
        offset = 0
        file_size = self._metadata.st_size
        try:
            before = await anyio.to_thread.run_sync(os.fstat, self._descriptor)
            if self._metadata_signature(before) != self._metadata_signature(
                self._metadata
            ):
                raise OSError("Artifact metadata changed before download")
            while offset < file_size:
                requested = min(self.chunk_size, file_size - offset)
                chunk = await self._read_at(requested, offset)
                if not chunk:
                    raise OSError(
                        f"Artifact download ended after {offset} of {file_size} bytes"
                    )
                if len(chunk) > requested:
                    raise OSError("Artifact reader returned more bytes than requested")
                await anyio.to_thread.run_sync(snapshot.write, chunk)
                digest.update(chunk)
                offset += len(chunk)
            if await self._read_at(1, file_size):
                raise OSError("Artifact grew while preparing its download")
            after = await anyio.to_thread.run_sync(os.fstat, self._descriptor)
            if self._metadata_signature(after) != self._metadata_signature(before):
                raise OSError("Artifact metadata changed while preparing its download")
            await anyio.to_thread.run_sync(snapshot.seek, 0)
        except BaseException:
            snapshot.close()
            raise
        self._snapshot = snapshot
        content_digest = digest.hexdigest()
        self.headers["etag"] = f'"sha256:{content_digest}"'
        self.headers["x-content-sha256"] = content_digest
        self._stream.close()

    async def _read_snapshot_at(self, size: int, offset: int) -> bytes:
        await self._ensure_snapshot()
        assert self._snapshot is not None

        def read() -> bytes:
            assert self._snapshot is not None
            self._snapshot.seek(offset, os.SEEK_SET)
            return self._snapshot.read(size)

        return await anyio.to_thread.run_sync(read)

    async def _handle_simple(
        self,
        send: Send,
        send_header_only: bool,
        _send_pathsend: bool,
    ) -> None:
        await self._ensure_snapshot()
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        if send_header_only:
            await send(
                {
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                }
            )
            return

        offset = 0
        file_size = self._metadata.st_size
        if file_size == 0:
            await send(
                {
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                }
            )
            return
        while offset < file_size:
            chunk = await self._read_snapshot_at(
                min(self.chunk_size, file_size - offset),
                offset,
            )
            if not chunk:
                raise OSError("Prepared artifact snapshot ended unexpectedly")
            offset += len(chunk)
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": offset < file_size,
                }
            )

    async def _handle_single_range(
        self,
        send: Send,
        start: int,
        end: int,
        file_size: int,
        send_header_only: bool,
    ) -> None:
        await self._ensure_snapshot()
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-range"] = f"bytes {start}-{end - 1}/{file_size}"
        headers["content-length"] = str(end - start)
        await send(
            {
                "type": "http.response.start",
                "status": 206,
                "headers": headers.raw,
            }
        )
        if send_header_only:
            await send(
                {
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                }
            )
            return

        offset = start
        while offset < end:
            chunk = await self._read_snapshot_at(
                min(self.chunk_size, end - offset), offset
            )
            if not chunk:
                raise OSError("Prepared artifact snapshot ended unexpectedly")
            offset += len(chunk)
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": offset < end,
                }
            )

    async def _handle_multiple_ranges(
        self,
        send: Send,
        ranges: list[tuple[int, int]],
        file_size: int,
        send_header_only: bool,
    ) -> None:
        await self._ensure_snapshot()
        boundary = token_hex(13)
        content_length, header_generator = self.generate_multipart(
            ranges,
            boundary,
            file_size,
            self.headers["content-type"],
        )
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-type"] = f"multipart/byteranges; boundary={boundary}"
        headers["content-length"] = str(content_length)
        await send(
            {
                "type": "http.response.start",
                "status": 206,
                "headers": headers.raw,
            }
        )
        if send_header_only:
            await send(
                {
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                }
            )
            return

        for start, end in ranges:
            await send(
                {
                    "type": "http.response.body",
                    "body": header_generator(start, end),
                    "more_body": True,
                }
            )
            offset = start
            while offset < end:
                chunk = await self._read_snapshot_at(
                    min(self.chunk_size, end - offset),
                    offset,
                )
                if not chunk:
                    raise OSError("Prepared artifact snapshot ended unexpectedly")
                offset += len(chunk)
                await send(
                    {
                        "type": "http.response.body",
                        "body": chunk,
                        "more_body": True,
                    }
                )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"\r\n",
                    "more_body": True,
                }
            )
        await send(
            {
                "type": "http.response.body",
                "body": f"--{boundary}--".encode("latin-1"),
                "more_body": False,
            }
        )
