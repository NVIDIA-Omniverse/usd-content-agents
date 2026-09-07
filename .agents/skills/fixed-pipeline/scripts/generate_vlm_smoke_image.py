#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print a deterministic blue PNG data URL for public VLM smoke tests."""

from __future__ import annotations

import base64
import binascii
import struct
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)
    )


def generate_png(*, size: int = 16) -> bytes:
    """Return a small RGB PNG whose dominant color is unambiguously blue."""

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    blue_pixel = bytes((32, 96, 224))
    scanlines = b"".join(b"\x00" + blue_pixel * size for _ in range(size))
    return (
        PNG_SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _chunk(b"IEND", b"")
    )


def main() -> None:
    encoded = base64.b64encode(generate_png()).decode("ascii")
    print(f"data:image/png;base64,{encoded}")


if __name__ == "__main__":
    main()
