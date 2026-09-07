#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite a wheel with a deterministic ZIP envelope."""

from __future__ import annotations

import argparse
import os
import pathlib
import stat
import time
import zipfile

EXPECTED_ZLIB_VERSION = "1.3.1"
_COMPRESSION_LEVEL = 9


def _archive_timestamp(source_date_epoch: int) -> tuple[int, int, int, int, int, int]:
    if isinstance(source_date_epoch, bool) or not isinstance(source_date_epoch, int):
        raise TypeError("source_date_epoch must be an integer")
    timestamp = time.gmtime(source_date_epoch)[:6]
    if not 1980 <= timestamp[0] <= 2107:
        raise ValueError("source_date_epoch is outside the ZIP timestamp range")
    return timestamp


def _validate_member_name(name: str) -> None:
    path = pathlib.PurePosixPath(name)
    if not name or "\x00" in name or "\\" in name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"wheel contains an unsafe member name: {name!r}")


def canonicalize_wheel(wheel: pathlib.Path, *, source_date_epoch: int) -> None:
    """Canonicalize member order, metadata, and compression in place."""

    wheel = pathlib.Path(wheel)
    if wheel.suffix != ".whl" or not wheel.is_file():
        raise ValueError(f"wheel must be an existing .whl file: {wheel}")
    if (
        zipfile.zlib is None
        or zipfile.zlib.ZLIB_VERSION != EXPECTED_ZLIB_VERSION
        or zipfile.zlib.ZLIB_RUNTIME_VERSION != EXPECTED_ZLIB_VERSION
    ):
        raise RuntimeError(f"wheel canonicalization requires exact zlib {EXPECTED_ZLIB_VERSION}")

    timestamp = _archive_timestamp(source_date_epoch)
    members: dict[str, tuple[bytes, bool]] = {}
    with zipfile.ZipFile(wheel) as source:
        for item in source.infolist():
            _validate_member_name(item.filename)
            if item.filename in members:
                raise ValueError(f"wheel contains a duplicate member: {item.filename}")
            members[item.filename] = (source.read(item), item.is_dir())

    temporary = wheel.with_name(f".{wheel.name}.canonical.{os.getpid()}")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=_COMPRESSION_LEVEL,
            strict_timestamps=True,
        ) as target:
            target.comment = b""
            for name in sorted(members):
                data, is_directory = members[name]
                info = zipfile.ZipInfo(name, date_time=timestamp)
                info.create_system = 3
                info.internal_attr = 0
                info.external_attr = (
                    (stat.S_IFDIR | 0o755) << 16 | 0x10
                    if is_directory
                    else (stat.S_IFREG | 0o644) << 16
                )
                info.compress_type = zipfile.ZIP_STORED if is_directory else zipfile.ZIP_DEFLATED
                target.writestr(
                    info,
                    data,
                    compress_type=info.compress_type,
                    compresslevel=None if is_directory else _COMPRESSION_LEVEL,
                )
        os.replace(temporary, wheel)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=pathlib.Path)
    parser.add_argument("source_date_epoch", type=int)
    args = parser.parse_args()
    canonicalize_wheel(args.wheel, source_date_epoch=args.source_date_epoch)


if __name__ == "__main__":
    main()
