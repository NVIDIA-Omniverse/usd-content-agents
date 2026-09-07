# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage and behavior tests for descriptor-backed file responses."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from starlette.responses import FileResponse

from world_understanding.utils import held_file_response
from world_understanding.utils.artifacts import OpenArtifactFile
from world_understanding.utils.held_file_response import (
    HeldFileResponse,
    open_held_artifact_file,
)


def _artifact(path: Path, payload: bytes = b"abcdef") -> OpenArtifactFile:
    path.write_bytes(payload)
    stream = path.open("rb")
    return OpenArtifactFile(
        relative_key=path.name,
        stream=stream,
        metadata=os.fstat(stream.fileno()),
    )


@pytest.mark.asyncio
async def test_held_response_reads_the_held_descriptor_and_closes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "artifact.bin").write_bytes(b"abcdef")
    artifact = open_held_artifact_file(tmp_path, "artifact.bin")
    response = HeldFileResponse(artifact, media_type="application/octet-stream")

    async def run_sync(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(held_file_response.anyio.to_thread, "run_sync", run_sync)

    assert await response._read_at(3, 2) == b"cde"

    called = False

    async def fake_call(
        _response: FileResponse,
        _scope: Any,
        _receive: Any,
        _send: Any,
    ) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(FileResponse, "__call__", fake_call)
    await response({}, None, None)  # type: ignore[arg-type]

    assert called is True
    assert artifact.stream.closed is True


@pytest.mark.asyncio
async def test_simple_response_handles_head_full_and_short_reads(
    tmp_path: Path,
) -> None:
    messages: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    head_artifact = _artifact(tmp_path / "head.bin")
    head = HeldFileResponse(head_artifact)
    await head._handle_simple(send, True, False)
    assert messages[-1] == {
        "type": "http.response.body",
        "body": b"",
        "more_body": False,
    }
    head_artifact.stream.close()

    messages.clear()
    empty_artifact = _artifact(tmp_path / "empty.bin", b"")
    empty = HeldFileResponse(empty_artifact)
    await empty._handle_simple(send, False, False)
    assert messages[-1] == {
        "type": "http.response.body",
        "body": b"",
        "more_body": False,
    }
    assert empty.headers["x-content-sha256"] == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert empty_artifact.stream.closed is True

    messages.clear()
    body_artifact = _artifact(tmp_path / "body.bin")
    body = HeldFileResponse(body_artifact)
    body.chunk_size = 2
    body._snapshot_memory_limit = 2
    original_read = body._read_at

    async def body_read(size: int, offset: int) -> bytes:
        return await original_read(min(size, 1), offset)

    body._read_at = body_read  # type: ignore[method-assign]
    await body._handle_simple(send, False, False)
    assert b"".join(message.get("body", b"") for message in messages) == b"abcdef"
    assert body.headers["x-content-sha256"] == (
        "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721"
    )
    body_artifact.stream.close()

    messages.clear()
    short_artifact = _artifact(tmp_path / "short.bin")
    short = HeldFileResponse(short_artifact)

    async def empty_read(_size: int, _offset: int) -> bytes:
        return b""

    short._read_at = empty_read  # type: ignore[method-assign]
    with pytest.raises(OSError, match="ended after 0 of 6 bytes"):
        await short._handle_simple(send, False, False)
    assert messages == []
    short_artifact.stream.close()


@pytest.mark.asyncio
async def test_response_stages_large_artifact_before_success(
    tmp_path: Path,
) -> None:
    payload = bytes(range(256)) * 1025
    artifact = _artifact(tmp_path / "large.bin", payload)
    response = HeldFileResponse(artifact, media_type="application/octet-stream")
    response.chunk_size = 1024
    response._snapshot_memory_limit = 2048
    messages: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    await response(
        {"type": "http", "method": "GET", "headers": []},
        None,
        send,
    )  # type: ignore[arg-type]

    assert messages[0]["type"] == "http.response.start"
    assert b"".join(message.get("body", b"") for message in messages) == payload
    assert response.headers["content-length"] == str(len(payload))
    assert response.headers["x-content-sha256"] == (
        "85e1298a87a2077b5de87c6ea60e77be9ceba06f955c51b73676ee5a71f1187f"
    )
    assert response.headers["etag"] == (
        '"sha256:85e1298a87a2077b5de87c6ea60e77be9ceba06f955c51b73676ee5a71f1187f"'
    )
    assert response._snapshot is not None
    assert response._snapshot.closed is True
    assert artifact.stream.closed is True


@pytest.mark.asyncio
async def test_response_detects_read_failure_before_success(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path / "failure.bin")
    response = HeldFileResponse(artifact)
    messages: list[dict[str, Any]] = []

    async def fail_read(_size: int, _offset: int) -> bytes:
        raise PermissionError("injected read failure")

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    response._read_at = fail_read  # type: ignore[method-assign]
    with pytest.raises(PermissionError, match="injected read failure"):
        await response(
            {"type": "http", "method": "GET", "headers": []},
            None,
            send,
        )  # type: ignore[arg-type]
    assert messages == []
    assert artifact.stream.closed is True


@pytest.mark.asyncio
async def test_single_range_handles_head_full_and_short_reads(tmp_path: Path) -> None:
    messages: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    head_artifact = _artifact(tmp_path / "range-head.bin")
    head = HeldFileResponse(head_artifact)
    await head._handle_single_range(send, 1, 4, 6, True)
    assert messages[0]["status"] == 206
    assert messages[-1]["body"] == b""
    head_artifact.stream.close()

    messages.clear()
    body_artifact = _artifact(tmp_path / "range-body.bin")
    body = HeldFileResponse(body_artifact)
    body.chunk_size = 2

    await body._handle_single_range(send, 1, 5, 6, False)
    assert b"".join(message.get("body", b"") for message in messages) == b"bcde"
    body_artifact.stream.close()

    messages.clear()
    short_artifact = _artifact(tmp_path / "range-short.bin")
    short = HeldFileResponse(short_artifact)

    async def empty_read(_size: int, _offset: int) -> bytes:
        return b""

    short._read_at = empty_read  # type: ignore[method-assign]
    with pytest.raises(OSError, match="ended after 0 of 6 bytes"):
        await short._handle_single_range(send, 1, 5, 6, False)
    assert messages == []
    short_artifact.stream.close()


@pytest.mark.asyncio
async def test_multiple_ranges_handles_head_full_and_short_reads(
    tmp_path: Path,
) -> None:
    messages: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    head_artifact = _artifact(tmp_path / "multi-head.bin")
    head = HeldFileResponse(head_artifact, media_type="application/octet-stream")
    await head._handle_multiple_ranges(send, [(0, 2), (4, 6)], 6, True)
    assert messages[0]["status"] == 206
    assert messages[-1]["body"] == b""
    head_artifact.stream.close()

    messages.clear()
    body_artifact = _artifact(tmp_path / "multi-body.bin")
    body = HeldFileResponse(body_artifact, media_type="application/octet-stream")
    body.chunk_size = 1

    await body._handle_multiple_ranges(send, [(0, 2), (4, 6)], 6, False)
    response_body = b"".join(message.get("body", b"") for message in messages)
    assert b"ab" in response_body
    assert b"ef" in response_body
    assert messages[-1]["more_body"] is False
    body_artifact.stream.close()

    messages.clear()
    short_artifact = _artifact(tmp_path / "multi-short.bin")
    short = HeldFileResponse(short_artifact, media_type="application/octet-stream")

    async def empty_read(_size: int, _offset: int) -> bytes:
        return b""

    short._read_at = empty_read  # type: ignore[method-assign]
    with pytest.raises(OSError, match="ended after 0 of 6 bytes"):
        await short._handle_multiple_ranges(send, [(0, 2)], 6, False)
    assert messages == []
    short_artifact.stream.close()
