# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Archive extraction helpers shared by agent services."""

from __future__ import annotations

from typing import BinaryIO

DEFAULT_ARCHIVE_COPY_CHUNK_BYTES = 1024 * 1024


class ArchiveSizeLimitExceeded(ValueError):
    """Raised when an archive member exceeds an actual streamed-byte limit."""

    def __init__(self, *, max_bytes: int, attempted_bytes: int) -> None:
        super().__init__(f"Archive member exceeded {max_bytes} bytes while extracting.")
        self.max_bytes = max_bytes
        self.attempted_bytes = attempted_bytes


def copy_stream_limited(
    src: BinaryIO,
    dst: BinaryIO,
    *,
    max_bytes: int,
    chunk_size: int = DEFAULT_ARCHIVE_COPY_CHUNK_BYTES,
) -> int:
    """Copy a binary stream while enforcing an actual-byte limit.

    ``zipfile.ZipInfo.file_size`` comes from archive metadata and is not a
    sufficient guard against malicious or malformed inputs. This helper counts
    decompressed bytes as they are read and raises before writing a chunk that
    would exceed ``max_bytes``.

    Returns the number of bytes written.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    copied = 0
    while True:
        chunk = src.read(chunk_size)
        if not chunk:
            return copied
        next_copied = copied + len(chunk)
        if next_copied > max_bytes:
            raise ArchiveSizeLimitExceeded(
                max_bytes=max_bytes,
                attempted_bytes=next_copied,
            )
        dst.write(chunk)
        copied = next_copied
