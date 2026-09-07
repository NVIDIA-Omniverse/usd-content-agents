# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic ZIP-envelope tests for the native wheel build."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

NATIVE_ROOT = Path(__file__).resolve().parents[1] / "native"
CANONICALIZER_PATH = NATIVE_ROOT / "canonicalize_wheel.py"
SPEC = importlib.util.spec_from_file_location("canonicalize_wheel", CANONICALIZER_PATH)
assert SPEC is not None and SPEC.loader is not None
canonicalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(canonicalizer)

_MEMBERS = {
    "demo/": b"",
    "demo/__init__.py": b"VALUE = 1\n",
    "demo-1.0.dist-info/METADATA": b"Name: demo\nVersion: 1.0\n",
    "demo-1.0.dist-info/RECORD": b"demo/__init__.py,,\n",
}

_HAS_LOCKED_ZLIB = (
    canonicalizer.zipfile.zlib is not None
    and canonicalizer.zipfile.zlib.ZLIB_VERSION == canonicalizer.EXPECTED_ZLIB_VERSION
    and canonicalizer.zipfile.zlib.ZLIB_RUNTIME_VERSION == canonicalizer.EXPECTED_ZLIB_VERSION
)


def _write_wheel(
    path: Path,
    *,
    member_order: list[str],
    compression: int,
    timestamp: tuple[int, int, int, int, int, int],
    mode: int,
) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        archive.comment = b"host-specific comment"
        for name in member_order:
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.create_system = 3
            info.external_attr = mode << 16
            info.comment = b"member comment"
            info.compress_type = compression
            archive.writestr(info, _MEMBERS[name])


@pytest.mark.skipif(
    not _HAS_LOCKED_ZLIB,
    reason="byte-level wheel canonicalization requires the locked zlib toolchain",
)
def test_canonicalizer_collapses_different_zip_envelopes_to_identical_bytes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    names = list(_MEMBERS)
    _write_wheel(
        first,
        member_order=names,
        compression=zipfile.ZIP_STORED,
        timestamp=(2024, 1, 2, 3, 4, 6),
        mode=stat.S_IFREG | 0o600,
    )
    _write_wheel(
        second,
        member_order=list(reversed(names)),
        compression=zipfile.ZIP_DEFLATED,
        timestamp=(2026, 7, 8, 9, 10, 12),
        mode=stat.S_IFREG | 0o664,
    )

    for wheel in (first, second):
        canonicalizer.canonicalize_wheel(wheel, source_date_epoch=1_762_211_065)

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == sorted(_MEMBERS)
        assert archive.comment == b""
        for info in archive.infolist():
            assert info.date_time == (2025, 11, 3, 23, 4, 24)
            assert info.comment == b""
            assert info.extra == b""
            assert info.create_system == 3
            if info.is_dir():
                assert info.compress_type == zipfile.ZIP_STORED
                assert info.external_attr >> 16 == stat.S_IFDIR | 0o755
            else:
                assert info.compress_type == zipfile.ZIP_DEFLATED
                assert info.external_attr >> 16 == stat.S_IFREG | 0o644
            assert archive.read(info) == _MEMBERS[info.filename]


@pytest.mark.parametrize(
    ("compile_version", "runtime_version"),
    [
        ("1.2.13", canonicalizer.EXPECTED_ZLIB_VERSION),
        (canonicalizer.EXPECTED_ZLIB_VERSION, "1.2.13"),
    ],
)
def test_canonicalizer_rejects_a_nonlocked_zlib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compile_version: str,
    runtime_version: str,
) -> None:
    wheel = tmp_path / "input.whl"
    _write_wheel(
        wheel,
        member_order=list(_MEMBERS),
        compression=zipfile.ZIP_STORED,
        timestamp=(2024, 1, 2, 3, 4, 6),
        mode=stat.S_IFREG | 0o600,
    )
    monkeypatch.setattr(
        canonicalizer.zipfile,
        "zlib",
        SimpleNamespace(
            ZLIB_VERSION=compile_version,
            ZLIB_RUNTIME_VERSION=runtime_version,
        ),
    )

    with pytest.raises(RuntimeError, match="requires exact zlib 1[.]3[.]1"):
        canonicalizer.canonicalize_wheel(wheel, source_date_epoch=1_762_211_065)


def test_native_build_digest_binds_and_invokes_the_canonicalizer() -> None:
    build_path = NATIVE_ROOT / "build_openvdb_wheel.sh"
    source_lock = json.loads((NATIVE_ROOT / "source-lock.json").read_text(encoding="utf-8"))
    build = build_path.read_text(encoding="utf-8")
    canonicalizer_sha256 = hashlib.sha256(CANONICALIZER_PATH.read_bytes()).hexdigest()

    assert f'WHEEL_CANONICALIZER_SHA256="{canonicalizer_sha256}"' in build
    invocation = '"${TOOL_PYTHON}" "${WHEEL_CANONICALIZER}" "${FINAL_WHEEL}" "${SOURCE_DATE_EPOCH}"'
    assert invocation in build
    assert build.index('"${TOOL_ENV}/bin/auditwheel" repair') < build.index(invocation)
    assert build.index(invocation) < build.index('"${TOOL_PYTHON}" "${LICENSE_VERIFIER}"')
    assert source_lock["build_recipe"] == {
        "path": build_path.name,
        "sha256": hashlib.sha256(build_path.read_bytes()).hexdigest(),
    }
