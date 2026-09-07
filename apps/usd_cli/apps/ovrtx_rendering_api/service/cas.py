# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed blob store for the manifest render transport.

Clients negotiate content-addressed uploads by manifest:
a per-file sha256 manifest of the packaged bundle's zip entries, upload only the
blobs this node is missing, and render by manifest. The store is a plain directory
of sha-named files — deliberately NOT durable: blob loss costs one 409 + re-upload
(the client re-negotiates), never a wrong render. Single-tenant by construction
(the service has one API key), so there is no per-tenant namespacing.

Layout: `<root>/objects/<sha[:2]>/<sha>` (0400 once published) + `<root>/staging/`
for per-render materialized trees (same filesystem, so materialization is pure
hardlinks — metadata-only, and eviction mid-render is harmless because the staged
links keep the inodes alive).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
#: blobs untouched for this long are eviction candidates regardless of quota
#: (mirrors the client bundle cache's 24h bound)
BLOB_TTL_S = 24 * 3600


def _ensure_private_directory(path: Path) -> None:
    """Create or normalize one service-owned directory to owner-only access."""
    if path.is_symlink():
        raise RuntimeError(f"CAS directory may not be a symlink: {path}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    get_euid = getattr(os, "geteuid", None)
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"CAS path is not a directory: {path}")
        os.chmod(path, 0o700)
        return
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, directory_flags)
    try:
        metadata = os.fstat(descriptor)
        named = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise RuntimeError(f"CAS path is not a stable directory: {path}")
        if get_euid is not None and metadata.st_uid != get_euid():
            raise RuntimeError(
                f"CAS directory is owned by uid {metadata.st_uid}, not the "
                f"service: {path}"
            )
        os.fchmod(descriptor, 0o700)
        secured = os.fstat(descriptor)
        secured_named = path.stat(follow_symlinks=False)
        if (
            (secured.st_dev, secured.st_ino)
            != (secured_named.st_dev, secured_named.st_ino)
            or stat.S_IMODE(secured.st_mode) & 0o077
        ):
            raise RuntimeError(f"CAS directory is not owner-only: {path}")
    finally:
        os.close(descriptor)


class MissingBlobsError(Exception):
    """A materialization referenced blobs the store does not (or no longer) hold.

    Carries the missing sha256 list so the endpoint can answer 409 with it and the
    client can re-upload exactly those — never compose a partial tree (plan goal 4).
    """

    def __init__(self, missing: list[str]):
        self.missing = sorted(set(missing))
        super().__init__(f"{len(self.missing)} blob(s) missing from the store")


def validate_rel_path(path: str) -> str:
    """A manifest path must be a plain, forward-slash relative path with no traversal.

    Manifest paths come from the network and are joined under a staging directory —
    reject anything that could escape it (absolute paths, `..`, backslashes,
    control characters) instead of normalizing. Returns the path unchanged."""
    if not path or len(path) > 1024:
        raise ValueError("manifest path must be 1..1024 characters")
    if "\\" in path:
        raise ValueError(f"manifest path may not contain backslashes: {path!r}")
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in path):
        raise ValueError("manifest path may not contain control characters")
    if path.startswith("/"):
        raise ValueError(f"manifest path must be relative: {path!r}")
    for segment in path.split("/"):
        if segment in ("", ".", ".."):
            raise ValueError(f"manifest path may not contain '{segment or '//'}' "
                             f"segments: {path!r}")
    return path


class BlobStore:
    """sha256-addressed blobs on disk with verify-on-write and LRU/TTL eviction.

    Interface is kept deliberately narrow (`missing` / `put` / `materialize` /
    `prefetch`) so a Phase-3 fleet-shared backend (S3/EFS/peer-fetch — plan §7 Q3)
    can replace the on-disk implementation without touching the protocol layer.
    """

    def __init__(self, root: str | os.PathLike, max_bytes: int = 20 * 1024 ** 3):
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.staging = self.root / "staging"
        self.max_bytes = int(max_bytes)
        _ensure_private_directory(self.root)
        _ensure_private_directory(self.objects)
        _ensure_private_directory(self.staging)

    def blob_path(self, sha: str) -> Path:
        return self.objects / sha[:2] / sha

    def missing(self, shas: list[str]) -> list[str]:
        """The subset of `shas` not in the store (sorted, deduped). Present blobs
        get their mtime bumped so LRU eviction sees the reuse."""
        now = time.time()
        out: list[str] = []
        for sha in sorted(set(shas)):
            try:
                os.utime(self.blob_path(sha), (now, now))
            except OSError:
                out.append(sha)
        return out

    def prefetch(self, shas: list[str]) -> None:
        """Hint present blobs into the page cache (posix_fadvise WILLNEED) so disk
        warm-up overlaps the client's upload of the missing ones. Advisory only."""
        if not hasattr(os, "posix_fadvise"):
            return
        for sha in set(shas):
            try:
                fd = os.open(self.blob_path(sha), os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
                finally:
                    os.close(fd)
            except OSError:
                continue

    def put(self, sha: str, data: bytes) -> bool:
        """Store `data` as blob `sha` after verifying the digest; returns False when
        the blob already existed. A digest mismatch raises ValueError and stores
        nothing — a lying/corrupted upload must never poison the store."""
        digest = hashlib.sha256(data).hexdigest()
        if digest != sha:
            raise ValueError(
                f"blob digest mismatch: the uploaded bytes hash to {digest[:16]}…, "
                f"not the declared {sha[:16]}… — refusing to store")
        path = self.blob_path(sha)
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"CAS object is not a regular file: {path}")
            os.chmod(path, 0o400)
            now = time.time()
            os.utime(path, (now, now))
            return False
        _ensure_private_directory(path.parent)
        tmp = path.with_name(f".{sha}.tmp{os.getpid()}.{time.monotonic_ns()}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(tmp, flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.chmod(tmp, 0o400)  # published blobs are immutable and service-private
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        self._evict()
        return True

    def materialize(self, files: list[dict], root_rel: str, dst: str | os.PathLike,
                    max_total_bytes: int | None = None) -> Path:
        """Hardlink the manifest's blobs into `dst` under their manifest paths and
        return the absolute path of the root layer. All-or-nothing: any absent blob
        raises MissingBlobsError before a single link is made (never render a
        partially-materialized scene). Falls back to copying when the destination
        is on a different filesystem."""
        dst = Path(dst)
        paths: set[str] = set()
        for f in files:
            p = validate_rel_path(str(f["path"]))
            if p in paths:
                raise ValueError(f"duplicate manifest path: {p!r}")
            paths.add(p)
        if root_rel not in paths:
            raise ValueError(f"manifest root {root_rel!r} is not among its file paths")
        absent = [str(f["sha256"]) for f in files
                  if not self.blob_path(str(f["sha256"])).exists()]
        if absent:
            raise MissingBlobsError(absent)
        total = 0
        for f in files:
            blob = self.blob_path(str(f["sha256"]))
            target = dst / str(f["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(blob, target)
            except OSError:  # cross-device or FS without hardlinks — copy instead
                shutil.copy2(blob, target)
            try:
                total += target.stat().st_size
            except OSError:
                pass
            if max_total_bytes and total > max_total_bytes:
                raise ValueError(
                    f"manifest scene exceeds the staged-scene limit "
                    f"({total / _MB:.1f}+ MB > {max_total_bytes / _MB:.1f} MB) — "
                    "raise OVRTX_MAX_BODY_BYTES on the service or reduce the scene")
        return dst / root_rel

    def _evict(self) -> None:
        """Bound the store: drop blobs older than BLOB_TTL_S, then oldest-mtime-first
        until under `max_bytes`. Best-effort — eviction is hygiene, never a gate, and
        an in-flight render is immune (its staged hardlinks keep the bytes alive)."""
        try:
            entries: list[tuple[float, int, Path]] = []
            now = time.time()
            for prefix_dir in self.objects.iterdir():
                if not prefix_dir.is_dir():
                    continue
                for blob in prefix_dir.iterdir():
                    try:
                        st = blob.stat()
                    except OSError:
                        continue
                    if blob.name.startswith("."):  # crashed-put temp files
                        if now - st.st_mtime > BLOB_TTL_S:
                            blob.unlink(missing_ok=True)
                        continue
                    entries.append((st.st_mtime, st.st_size, blob))
            total = sum(size for _mtime, size, _p in entries)
            entries.sort()  # oldest first
            evicted = 0
            for mtime, size, blob in entries:
                over_ttl = now - mtime > BLOB_TTL_S
                if not over_ttl and total <= self.max_bytes:
                    break
                blob.unlink(missing_ok=True)
                total -= size
                evicted += 1
            if evicted:
                logger.info("CAS: evicted %d blob(s) (TTL/quota); store now %.1f MB",
                            evicted, total / _MB)
        except OSError:
            logger.debug("CAS: eviction sweep failed", exc_info=True)
