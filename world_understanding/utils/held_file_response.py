# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP responses backed by descriptor-confined regular files."""

from __future__ import annotations

import os
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
            await super().__call__(scope, receive, send)
        finally:
            self._stream.close()

    async def _read_at(self, size: int, offset: int) -> bytes:
        return await anyio.to_thread.run_sync(
            os.pread,
            self._descriptor,
            size,
            offset,
        )

    async def _handle_simple(
        self,
        send: Send,
        send_header_only: bool,
        _send_pathsend: bool,
    ) -> None:
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
        while offset < file_size:
            chunk = await self._read_at(
                min(self.chunk_size, file_size - offset),
                offset,
            )
            if not chunk:
                break
            offset += len(chunk)
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": offset < file_size,
                }
            )
        if offset == 0 or offset < file_size:
            await send(
                {
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
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
            chunk = await self._read_at(min(self.chunk_size, end - offset), offset)
            if not chunk:
                break
            offset += len(chunk)
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": offset < end,
                }
            )
        if offset == start or offset < end:
            await send(
                {
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                }
            )

    async def _handle_multiple_ranges(
        self,
        send: Send,
        ranges: list[tuple[int, int]],
        file_size: int,
        send_header_only: bool,
    ) -> None:
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
                chunk = await self._read_at(
                    min(self.chunk_size, end - offset),
                    offset,
                )
                if not chunk:
                    break
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
