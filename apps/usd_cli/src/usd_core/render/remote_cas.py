# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Client half of the CAS manifest transport (upload-dedup plan Phase 2, §3.C).

The manifest is the packaged USDZ's zip entries, hashed per file. A USDZ is an
*uncompressed* zip whose internal references were rewritten package-relative at
packaging time (usdUtils asset localization), so the entry set is a self-contained
relative tree the render service can materialize on disk and open directly — no
staging-tree path-anchoring problem, and every packaging correctness measure
(instance-binding bakes, camera stripping, broken-ref fallback) is inherited as-is.

Warm-path consequence: the heavy immutable entries (textures, payload layers) keep
stable hashes across renders, so an edit re-ships only the entries it changed —
typically just the root layer. See `RemoteRenderBackend._dispatch_cas` for the wire
flow (negotiate → PUT missing blobs → render-by-manifest).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import stat as stat_mod
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
#: sidecar format version — bump when the manifest shape changes
MANIFEST_VERSION = 1
#: entry suffixes whose content is already compressed — per-blob wire compression
#: is skipped for them instead of measured (it never pays)
INCOMPRESSIBLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".jxl",
                           ".ktx2", ".mp4", ".gz", ".zst", ".usdz"}
#: entries smaller than this upload raw — compression setup costs more than it saves
_COMPRESS_MIN_BYTES = 4096


def _hash_entries(usdz_path: Path) -> dict:
    """{"root", "files": [{path, sha256, size}...]} for the bundle's zip entries.

    The root layer is the package's default layer — by the USDZ spec, the first
    file in the archive (CreateNewUsdzPackage writes it first)."""
    files: list[dict] = []
    with zipfile.ZipFile(usdz_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            digest = hashlib.sha256()
            with zf.open(info) as fh:
                while chunk := fh.read(1 << 20):
                    digest.update(chunk)
            files.append({"path": info.filename, "sha256": digest.hexdigest(),
                          "size": info.file_size})
    if not files:
        raise RuntimeError(f"bundle {usdz_path} contains no files")
    return {"root": files[0]["path"], "files": files}


def _trusted_owned_file(path: Path) -> bool:
    """True only for a regular file owned by us, checked via O_NOFOLLOW — the
    sidecar lives next to cached bundles and gets the same planted-symlink
    paranoia as the bundle cache itself."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError:
        return False
    try:
        st = os.fstat(fd)
    finally:
        os.close(fd)
    if not stat_mod.S_ISREG(st.st_mode):
        return False
    uid = os.getuid() if hasattr(os, "getuid") else None
    return uid is None or st.st_uid == uid


def manifest_for_usdz(usdz_path: Path) -> dict:
    """The bundle's manifest, reusing a stat-validated sidecar when present.

    Hashing a multi-GB bundle takes seconds; cached bundles are rendered many
    times, so the manifest is persisted as `<bundle>.manifest.json` next to the
    usdz (which, for cache hits, is the long-lived bundle-cache entry). The
    sidecar records the bundle's (size, mtime_ns) and is discarded on any
    mismatch — a rewritten bundle must never serve a stale manifest. Sidecar
    read/write is best-effort; the hash walk is the fallback, never an error."""
    usdz_path = Path(usdz_path)
    st = os.stat(usdz_path)
    bundle_key = [st.st_size, st.st_mtime_ns]
    sidecar = usdz_path.with_name(usdz_path.name + ".manifest.json")
    try:
        if _trusted_owned_file(sidecar):
            raw = json.loads(sidecar.read_text())
            if (isinstance(raw, dict) and raw.get("version") == MANIFEST_VERSION
                    and raw.get("bundle") == bundle_key
                    and isinstance(raw.get("files"), list) and raw.get("root")):
                return {"root": raw["root"], "files": raw["files"]}
    except (OSError, ValueError):
        logger.debug("CAS: unreadable manifest sidecar %s — rehashing", sidecar,
                     exc_info=True)
    manifest = _hash_entries(usdz_path)
    try:
        tmp = sidecar.with_name(f"{sidecar.name}.tmp{os.getpid()}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump({"version": MANIFEST_VERSION, "bundle": bundle_key,
                       **manifest}, fh)
        os.replace(tmp, sidecar)
    except OSError:
        logger.debug("CAS: could not persist manifest sidecar %s", sidecar,
                     exc_info=True)
    return manifest


def stage_blob(zf: zipfile.ZipFile, entry_path: str, tmpdir: Path,
               codec: str | None) -> tuple[Path, int, str]:
    """Extract one entry to `tmpdir` and pick its wire form.

    Returns (file_to_send, its byte size, compression name). `codec` ("zstd" /
    "gzip" / None) is attempted only for plausibly-compressible entries and kept
    only when it saves at least 5% — the same policy the bundle transport applies.
    Files are reused per call (callers upload sequentially from one tmpdir)."""
    raw = tmpdir / "blob.raw"
    with zf.open(entry_path) as src, raw.open("wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    raw_size = raw.stat().st_size
    suffix = Path(entry_path).suffix.lower()
    if (not codec or suffix in INCOMPRESSIBLE_SUFFIXES
            or raw_size < _COMPRESS_MIN_BYTES):
        return raw, raw_size, "none"
    comp = tmpdir / "blob.comp"
    with raw.open("rb") as src:
        if codec == "zstd":
            import zstandard

            with comp.open("wb") as dst:
                zstandard.ZstdCompressor(level=10).copy_stream(src, dst)
        else:
            with gzip.open(comp, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    comp_size = comp.stat().st_size
    if comp_size < raw_size * 0.95:
        return comp, comp_size, codec
    comp.unlink(missing_ok=True)
    return raw, raw_size, "none"
