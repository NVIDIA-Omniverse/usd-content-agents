# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for remote-upload compression and measurement:

  1  zstd wire compression — chosen only when the local lib exists AND every
     known backend advertises the feature; gzip stays the floor; the service
     round-trips both codecs
  2  backend limit advertisement — a DEFAULT client cap adopts the backend's
     advertised max_body_bytes (round 8: the fail-fast idled the fleet's own
     L40 on a scene the service would have accepted); an explicit cap wins
  3  measurement counters — re-uploading identical content logs the re-shipped
     byte total (the number Phase 2's CAS transport will delete)
  4  bundle cache accepts either compressed sibling (.gz / .zst)
"""
from __future__ import annotations

import gzip
import logging

import pytest

zstandard = pytest.importorskip("zstandard")

from usd_core.render import remote as R
from usd_core.render.remote import RemoteRenderBackend


def _backend(**kw) -> RemoteRenderBackend:
    kw.setdefault("verify_version", False)
    kw.setdefault("bundle_cache", False)
    b = RemoteRenderBackend("http://a:1", **kw)
    # isolate pool state from other tests (module-level pool registry)
    for s in b._pool:
        s.verified = False
        s.info = {}
    b._backend_info = {}
    return b


# ── 2: effective cap ─────────────────────────────────────────────────────────────


def test_default_cap_adopts_backend_advertised_limit():
    b = _backend(max_upload_mb=R._DEFAULT_MAX_UPLOAD_MB)
    b._backend_info = {"max_body_bytes": 2 * 1024**3}
    cap, source = b._effective_cap_mb()
    assert cap == 2048.0
    assert "advertised" in source


def test_explicit_cap_beats_advertised_limit():
    b = _backend(max_upload_mb=100)
    b._backend_info = {"max_body_bytes": 2 * 1024**3}
    assert b._effective_cap_mb() == (100.0, "render.remote_max_upload_mb")


def test_pool_adopts_smallest_advertised_limit():
    b = _backend(max_upload_mb=R._DEFAULT_MAX_UPLOAD_MB)
    b._pool[0].verified = True
    b._pool[0].info = {"max_body_bytes": 1024**3}
    b._backend_info = {"max_body_bytes": 2 * 1024**3}
    cap, _ = b._effective_cap_mb()
    assert cap == 1024.0  # min across the fleet — every node must accept it


def test_smaller_advertised_limit_never_shrinks_the_default():
    b = _backend(max_upload_mb=R._DEFAULT_MAX_UPLOAD_MB)
    b._backend_info = {"max_body_bytes": 64 * 1024**2}
    assert b._effective_cap_mb()[0] == R._DEFAULT_MAX_UPLOAD_MB


# ── 1: compression choice ────────────────────────────────────────────────────────


def _compressible_usdz(tmp_path):
    p = tmp_path / "scene_bundle.usdz"
    p.write_bytes(b"#usda 1.0\n" + b"def Xform \"a\" {}\n" * 20000)
    return p


def test_prepare_upload_uses_zstd_when_fleet_advertises_it(tmp_path):
    b = _backend()
    b._backend_info = {"features": ["zstd"]}
    send, fmt = b._prepare_upload(_compressible_usdz(tmp_path))
    assert fmt == "zstd" and send.name.endswith(".usdz.zst")
    raw = zstandard.ZstdDecompressor().decompress(
        send.read_bytes(), max_output_size=1 << 26)
    assert raw.startswith(b"#usda 1.0")


def test_prepare_upload_falls_back_to_gzip_without_advertisement(tmp_path):
    b = _backend()  # no backend info at all -> unknown fleet -> gzip
    send, fmt = b._prepare_upload(_compressible_usdz(tmp_path))
    assert fmt == "gzip" and send.name.endswith(".usdz.gz")


def test_mixed_fleet_disables_zstd(tmp_path):
    b = _backend()
    b._pool[0].verified = True
    b._pool[0].info = {"features": []}  # one node without zstd
    b._backend_info = {"features": ["zstd"]}
    send, fmt = b._prepare_upload(_compressible_usdz(tmp_path))
    assert fmt == "gzip"


def test_gzip_from_zstd_transcode(tmp_path):
    b = _backend()
    b._backend_info = {"features": ["zstd"]}
    send, fmt = b._prepare_upload(_compressible_usdz(tmp_path))
    assert fmt == "zstd"
    gz = b._gzip_from_zstd(send)
    assert gz.name.endswith(".usdz.gz")
    assert gzip.decompress(gz.read_bytes()).startswith(b"#usda 1.0")
    assert b._gzip_from_zstd(send) == gz  # idempotent


# ── 3: measurement counters ─────────────────────────────────────────────────────


def test_upload_ledger_counts_reshipped_bytes(tmp_path, caplog):
    payload = tmp_path / "scene_bundle.usdz.gz"
    payload.write_bytes(b"x" * (2 * 1024 * 1024))
    R._UPLOAD_LEDGER.clear()
    R._UPLOAD_RESHIPPED[0] = 0
    with caplog.at_level(logging.INFO, logger=R.logger.name):
        RemoteRenderBackend._log_upload_ledger(payload, payload.stat().st_size, 1.0)
        RemoteRenderBackend._log_upload_ledger(payload, payload.stat().st_size, 2.0)
    first, second = [r.message for r in caplog.records if r.message.startswith("upload:")]
    assert "re-shipped" not in first
    assert "SAME content uploaded 2x" in second
    assert "2.0 MB re-shipped" in second


# ── 4: cache sibling formats ────────────────────────────────────────────────────


def test_bundle_cache_roundtrips_zstd_sibling(tmp_path):
    entry = tmp_path / "cache" / "scene--abc"
    usdz = tmp_path / "scene_bundle.usdz"
    usdz.write_bytes(b"#usda 1.0\n" + b"z" * 4096)
    zst = tmp_path / "scene_bundle.usdz.zst"
    zst.write_bytes(zstandard.ZstdCompressor().compress(usdz.read_bytes()))
    out = R._bundle_cache_put(entry, usdz, zst)
    assert out is not None
    got = R._bundle_cache_get(entry)
    assert got is not None
    _, send, fmt = got
    assert fmt == "zstd" and send.name == "scene_bundle.usdz.zst"


# ── service round-trip of both codecs ───────────────────────────────────────────


def test_service_decompress_body_roundtrip():
    import asyncio
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                           / "apps" / "ovrtx_rendering_api"))
    from service.main import _decompress_body

    raw = b"#usda 1.0\n" + b"payload" * 1000
    assert asyncio.run(_decompress_body(raw, "none")) == raw
    assert asyncio.run(_decompress_body(gzip.compress(raw), "gzip")) == raw
    z = zstandard.ZstdCompressor().compress(raw)
    assert asyncio.run(_decompress_body(z, "zstd")) == raw
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        asyncio.run(_decompress_body(b"garbage", "zstd"))
