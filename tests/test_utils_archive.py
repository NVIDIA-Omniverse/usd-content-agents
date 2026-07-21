# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared archive extraction helpers."""

from __future__ import annotations

from io import BytesIO

import pytest

from world_understanding.utils.archive import (
    ArchiveSizeLimitExceeded,
    copy_stream_limited,
)


def test_copy_stream_limited_returns_bytes_written() -> None:
    src = BytesIO(b"abcdef")
    dst = BytesIO()

    copied = copy_stream_limited(src, dst, max_bytes=6, chunk_size=2)

    assert copied == 6
    assert dst.getvalue() == b"abcdef"


def test_copy_stream_limited_raises_before_writing_excess_chunk() -> None:
    src = BytesIO(b"abcdef")
    dst = BytesIO()

    with pytest.raises(ArchiveSizeLimitExceeded) as exc_info:
        copy_stream_limited(src, dst, max_bytes=3, chunk_size=2)

    assert exc_info.value.max_bytes == 3
    assert exc_info.value.attempted_bytes == 4
    assert dst.getvalue() == b"ab"
