# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable, descriptor-backed captures for opaque artifact bytes.

The public capture paths always detach bytes from their source before yielding
them. Small artifacts use Linux sealed memory files; larger artifacts use
anonymous disk-backed snapshots. Callers receive neither a mutable pathname nor
an eager byte buffer. They can only stream bounded chunks from independently
opened, read-only descriptors.

Remote storage is intentionally outside this module. Providers implement
``CapturedOpaqueArtifactResolver`` and return the same context-managed capture
contract after opening their own bounded byte stream.

``WU_CAPTURED_ARTIFACT_SNAPSHOT_DIR`` can select an operator-managed spill
directory for anonymous snapshots. It must be writable and non-memory-backed.
Without it, capture tries bounded system temporary locations and, for local
files, the source directory.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib
import os
import re
import stat
import sys
import tempfile
import threading
from collections.abc import Callable, Generator, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Literal, Protocol, runtime_checkable


def _load_fcntl() -> Any:
    """Return the POSIX descriptor module without breaking Windows imports."""

    if os.name != "posix":  # pragma: no cover - selected on native Windows
        return None
    try:
        return importlib.import_module("fcntl")
    except ImportError:  # pragma: no cover - supported Linux provides fcntl
        return None


fcntl: Any = _load_fcntl()

CapturedStorageKind = Literal["sealed_memfd", "anonymous_snapshot"]

_MFD_CLOEXEC = 1
_MFD_ALLOW_SEALING = 2
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_CAPTURE_MEMFD_SEALS = 1 | 2 | 4 | 8
_DEFAULT_MEMFD_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_CHUNK_BYTES = 1024 * 1024
_MAX_CHUNK_BYTES = 8 * 1024 * 1024
_SNAPSHOT_DIRECTORY_ENV = "WU_CAPTURED_ARTIFACT_SNAPSHOT_DIR"
_MEMORY_BACKED_FILESYSTEM_TYPES = frozenset({"hugetlbfs", "ramfs", "tmpfs"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_USE_DEFAULT_MEMFD_CREATE = object()
_RLOCK_TYPE = type(threading.RLock())
# These errors mean the memfd facility is unavailable to this process before
# any source bytes have been consumed.  Resource exhaustion (for example
# EMFILE, ENFILE, or ENOMEM) and ABI/programming errors remain hard failures.
_MEMFD_AVAILABILITY_ERRNOS = frozenset({errno.EACCES, errno.ENOSYS, errno.EPERM})
_CAPTURE_PLATFORM_SUPPORTED = sys.platform.startswith("linux")


def _load_process_libc() -> Any:
    """Return the POSIX process handle without breaking non-POSIX imports."""

    if os.name != "posix":  # pragma: no cover - selected on native Windows
        return None
    try:
        return ctypes.CDLL(None, use_errno=True)
    except (OSError, TypeError):  # pragma: no cover - supported Linux has libc
        return None


_LIBC = _load_process_libc()
_MEMFD_CREATE: Any
try:
    _MEMFD_CREATE = _LIBC.memfd_create
except AttributeError:  # pragma: no cover - official runtimes are Linux
    _MEMFD_CREATE = None
else:
    _MEMFD_CREATE.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    _MEMFD_CREATE.restype = ctypes.c_int


class CapturedArtifactError(RuntimeError):
    """Fail-closed opaque-artifact capture or integrity error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CapturedArtifactCleanupError(CapturedArtifactError):
    """A retained descriptor could not be cleaned up safely."""


def _require_capture_platform() -> None:
    """Fail before source access when descriptor-backed capture is unavailable."""

    if _CAPTURE_PLATFORM_SUPPORTED:
        return
    raise CapturedArtifactError(
        "unsupported_platform",
        "Descriptor-backed opaque artifact capture requires Linux, a Linux "
        "container, or WSL2.",
    )


@dataclass(frozen=True, slots=True)
class OpaqueArtifactRequest:
    """Exact identity and admission budget for one opaque artifact."""

    uri: str
    sha256: str
    size_bytes: int
    max_bytes: int

    def __init_subclass__(cls, **_kwargs: object) -> None:
        """Keep request scalar validation closed to overridden behavior."""

        raise TypeError("OpaqueArtifactRequest cannot be subclassed")

    def __post_init__(self) -> None:
        if type(self.uri) is not str:
            raise TypeError("opaque artifact uri must be a string")
        if not self.uri.strip() or "\x00" in self.uri:
            raise ValueError("opaque artifact uri must be nonblank")
        if type(self.sha256) is not str:
            raise TypeError("opaque artifact sha256 must be a string")
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError(
                "opaque artifact sha256 must be 64 lowercase hexadecimal characters"
            )
        _require_nonnegative_integer(self.size_bytes, "opaque artifact size_bytes")
        _require_nonnegative_integer(self.max_bytes, "opaque artifact max_bytes")
        if self.size_bytes > self.max_bytes:
            raise ValueError(
                "opaque artifact size_bytes exceeds its declared max_bytes"
            )


@runtime_checkable
class BoundedBinaryReader(Protocol):
    """Minimal caller-owned stream contract used by opaque-artifact capture."""

    def read(self, size: int, /) -> bytes: ...


@runtime_checkable
class CapturedOpaqueArtifactResolver(Protocol):
    """Provider boundary for resolving remote/object-store artifact bytes.

    Implementations own transport and authentication. They must return a
    context manager that yields an already captured immutable file matching the
    supplied request; this module deliberately provides no network resolver.
    """

    def capture(
        self,
        request: OpaqueArtifactRequest,
    ) -> AbstractContextManager[CapturedOpaqueFile]: ...


class CapturedOpaqueFile:
    """One immutable artifact snapshot with bounded duplicate-reader access."""

    __slots__ = (
        "_closed",
        "_descriptor",
        "_descriptor_identity",
        "_descriptor_state",
        "_lock",
        "_readers",
        "_request_snapshot",
        "_storage_kind",
    )

    def __init_subclass__(cls, **_kwargs: object) -> None:
        """Keep capture integrity and cleanup behavior closed to overrides."""

        raise TypeError("CapturedOpaqueFile cannot be subclassed")

    def __init__(
        self,
        *,
        request: OpaqueArtifactRequest,
        descriptor: int,
        storage_kind: CapturedStorageKind,
        descriptor_state: tuple[int, ...],
    ) -> None:
        # Retain only exact admitted scalars. The frozen request object remains
        # externally mutable through object.__setattr__, so it cannot be an
        # integrity authority after this construction boundary.
        self._request_snapshot = _snapshot_exact_opaque_request(
            request,
            label="capture request",
        )
        self._descriptor = descriptor
        self._storage_kind = storage_kind
        self._descriptor_state = descriptor_state
        metadata = os.fstat(descriptor)
        self._descriptor_identity = (metadata.st_dev, metadata.st_ino)
        self._closed = False
        self._readers: dict[int, tuple[int, int]] = {}
        self._lock = threading.RLock()

    @property
    def request(self) -> OpaqueArtifactRequest:
        """Return a defensive identity and byte-budget projection."""

        return OpaqueArtifactRequest(
            uri=self._request_snapshot[0],
            sha256=self._request_snapshot[1],
            size_bytes=self._request_snapshot[2],
            max_bytes=self._request_snapshot[3],
        )

    @property
    def uri(self) -> str:
        return self._request_snapshot[0]

    @property
    def sha256(self) -> str:
        return self._request_snapshot[1]

    @property
    def size_bytes(self) -> int:
        return self._request_snapshot[2]

    @property
    def max_bytes(self) -> int:
        return self._request_snapshot[3]

    @property
    def storage_kind(self) -> CapturedStorageKind:
        return self._storage_kind

    @property
    def closed(self) -> bool:
        return self._closed

    def require_intact(self) -> None:
        """Revalidate descriptor identity, immutability, size, and exact bytes."""

        with self._lock:
            self._require_intact_locked(verify_digest=True)

    def _require_intact_locked(self, *, verify_digest: bool) -> None:
        """Validate capture state while the exact ownership lock is held."""

        if self._closed or self._descriptor < 0:
            raise CapturedArtifactError(
                "capture_closed",
                "captured opaque artifact is already closed",
            )
        try:
            metadata = os.fstat(self._descriptor)
        except OSError as exc:
            raise CapturedArtifactError(
                "descriptor_invalid",
                "captured opaque artifact descriptor is no longer valid",
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != self._descriptor_identity
        ):
            raise CapturedArtifactError(
                "descriptor_reused",
                "captured opaque artifact descriptor no longer identifies its snapshot",
            )
        if _descriptor_state(metadata) != self._descriptor_state:
            raise CapturedArtifactError(
                "snapshot_changed",
                "captured opaque artifact snapshot state changed",
            )
        if self._storage_kind == "sealed_memfd":
            try:
                observed_seals = fcntl.fcntl(self._descriptor, _F_GET_SEALS)
            except OSError as exc:
                raise CapturedArtifactError(
                    "snapshot_unsealed",
                    "captured opaque artifact seals could not be verified",
                ) from exc
            if observed_seals & _CAPTURE_MEMFD_SEALS != _CAPTURE_MEMFD_SEALS:
                raise CapturedArtifactError(
                    "snapshot_unsealed",
                    "captured opaque artifact lost required memory-file seals",
                )
        elif self._storage_kind == "anonymous_snapshot":
            access_mode = fcntl.fcntl(self._descriptor, fcntl.F_GETFL) & os.O_ACCMODE
            if access_mode != os.O_RDONLY or metadata.st_nlink != 0:
                raise CapturedArtifactError(
                    "snapshot_mutable",
                    "captured opaque artifact disk snapshot is not anonymous "
                    "and read-only",
                )
        else:  # pragma: no cover - construction is closed over two literals
            raise CapturedArtifactError(
                "storage_kind_invalid",
                "captured opaque artifact has an unknown storage kind",
            )
        if metadata.st_size != self._request_snapshot[2]:
            raise CapturedArtifactError(
                "size_mismatch",
                "captured opaque artifact snapshot size changed",
            )
        if metadata.st_size > self._request_snapshot[3]:
            raise CapturedArtifactError(
                "max_bytes_exceeded",
                "captured opaque artifact snapshot exceeds max_bytes",
            )
        if (
            verify_digest
            and _stable_descriptor_sha256(self._descriptor) != self._request_snapshot[1]
        ):
            raise CapturedArtifactError(
                "snapshot_changed",
                "captured opaque artifact snapshot bytes changed",
            )

    def iter_chunks(
        self,
        *,
        chunk_size: int = _DEFAULT_CHUNK_BYTES,
    ) -> Generator[bytes, None, None]:
        """Yield exact bytes from a distinct read-only descriptor.

        ``chunk_size`` is capped so this API cannot accidentally turn an opaque
        artifact into an eager in-memory payload. Each bounded read owns a
        separate read-only descriptor and closes it before yielding. The
        iterator verifies the full capture once before streaming, maintains its
        own digest, and performs the full post-use integrity check.
        """

        _require_chunk_size(chunk_size)
        primary_error: BaseException | None = None
        primary_traceback: TracebackType | None = None
        additional_failures: list[tuple[str, BaseException]] = []
        completed = False
        closing = False
        try:
            self.require_intact()
            digest = hashlib.sha256()
            offset = 0
            while offset < self._request_snapshot[2]:
                chunk = self._read_ephemeral_chunk(
                    min(chunk_size, self._request_snapshot[2] - offset),
                    offset,
                )
                if not chunk:
                    raise CapturedArtifactError(
                        "reader_truncated",
                        "captured opaque artifact reader ended before size_bytes",
                    )
                digest.update(chunk)
                offset += len(chunk)
                yield chunk
            if self._read_ephemeral_chunk(1, offset):
                raise CapturedArtifactError(
                    "reader_grew",
                    "captured opaque artifact reader exceeded size_bytes",
                )
            if digest.hexdigest() != self._request_snapshot[1]:
                raise CapturedArtifactError(
                    "reader_digest_mismatch",
                    "captured opaque artifact reader disagrees with sha256",
                )
            completed = True
        except GeneratorExit:
            closing = True
        except BaseException as exc:
            primary_error = exc
            primary_traceback = exc.__traceback__
        finally:
            if not self._closed:
                try:
                    self.require_intact()
                except BaseException as integrity_error:
                    if primary_error is None:
                        primary_error = integrity_error
                        primary_traceback = integrity_error.__traceback__
                    else:
                        additional_failures.append(
                            ("post-read integrity check also failed", integrity_error)
                        )
        if primary_error is not None:
            _attach_failures(primary_error, additional_failures)
            raise primary_error.with_traceback(primary_traceback)
        if closing:
            return
        if not completed:  # pragma: no cover - generator protocol invariant
            raise CapturedArtifactError(
                "reader_incomplete",
                "captured opaque artifact reader did not complete",
            )

    def _read_ephemeral_chunk(self, size: int, offset: int) -> bytes:
        """Read one bounded chunk without suspending with descriptor ownership."""

        reader_descriptor = -1
        chunk: bytes | None = None
        primary_error: BaseException | None = None
        primary_traceback: TracebackType | None = None
        cleanup_failures: list[BaseException] = []
        try:
            reader_descriptor = self._open_reader()
            chunk = self._pread_reader(reader_descriptor, size, offset)
        except BaseException as exc:
            primary_error = exc
            primary_traceback = exc.__traceback__
        finally:
            if reader_descriptor >= 0:
                cleanup_failures.extend(self._close_reader_errors(reader_descriptor))
        if primary_error is not None:
            _attach_failures(
                primary_error,
                [
                    ("duplicate-reader cleanup also failed", cleanup_error)
                    for cleanup_error in cleanup_failures
                ],
            )
            raise primary_error.with_traceback(primary_traceback)
        if cleanup_failures:
            _raise_cleanup_failures(
                "captured opaque artifact reader cleanup failed",
                cleanup_failures,
            )
        assert chunk is not None
        return chunk

    def _open_reader(self) -> int:
        with self._lock:
            self._require_intact_locked(verify_digest=False)
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NONBLOCK", 0)
            try:
                if self._storage_kind == "anonymous_snapshot":
                    descriptor = os.dup(self._descriptor)
                else:
                    descriptor = os.open(f"/proc/self/fd/{self._descriptor}", flags)
            except OSError as exc:
                raise CapturedArtifactError(
                    "reader_open_failed",
                    "captured opaque artifact duplicate reader could not be opened",
                ) from exc
            try:
                metadata = os.fstat(descriptor)
                identity = (metadata.st_dev, metadata.st_ino)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or identity != self._descriptor_identity
                    or _descriptor_state(metadata) != self._descriptor_state
                ):
                    raise CapturedArtifactError(
                        "reader_identity_mismatch",
                        "captured opaque artifact duplicate reader resolved to a "
                        "different snapshot",
                    )
                access_mode = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
                if access_mode != os.O_RDONLY:
                    raise CapturedArtifactError(
                        "reader_not_read_only",
                        "captured opaque artifact duplicate reader is not read-only",
                    )
                self._readers[descriptor] = identity
                return descriptor
            except BaseException as primary_error:
                try:
                    os.close(descriptor)
                except BaseException as cleanup_error:
                    primary_error.add_note(
                        "Duplicate-reader open cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise

    def _pread_reader(self, descriptor: int, size: int, offset: int) -> bytes:
        """Read and revalidate while cleanup is excluded by the ownership lock."""

        with self._lock:
            if self._closed:
                raise CapturedArtifactError(
                    "capture_closed",
                    "captured opaque artifact is already closed",
                )
            self._require_reader_intact_locked(descriptor)
            chunk = os.pread(descriptor, size, offset)
            self._require_reader_intact_locked(descriptor)
            return chunk

    def _require_reader_intact_locked(self, descriptor: int) -> None:
        expected_identity = self._readers.get(descriptor)
        if expected_identity is None:
            raise CapturedArtifactError(
                "reader_closed",
                "captured opaque artifact duplicate reader is no longer owned",
            )
        try:
            metadata = os.fstat(descriptor)
        except OSError as exc:
            raise CapturedArtifactError(
                "reader_invalid",
                "captured opaque artifact duplicate reader is no longer valid",
            ) from exc
        identity = (metadata.st_dev, metadata.st_ino)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or identity != expected_identity
            or identity != self._descriptor_identity
            or _descriptor_state(metadata) != self._descriptor_state
        ):
            raise CapturedArtifactError(
                "reader_identity_mismatch",
                "captured opaque artifact duplicate reader no longer identifies "
                "its snapshot",
            )
        access_mode = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        if access_mode != os.O_RDONLY:
            raise CapturedArtifactError(
                "reader_not_read_only",
                "captured opaque artifact duplicate reader is not read-only",
            )

    def _close_reader_errors(self, descriptor: int) -> list[BaseException]:
        with self._lock:
            expected_identity = self._readers.get(descriptor)
            if expected_identity is None:
                return []
            errors, ownership_released = _close_owned_descriptor(
                descriptor,
                expected_identity=expected_identity,
                label="captured opaque artifact duplicate reader",
            )
            if ownership_released:
                self._readers.pop(descriptor, None)
            return errors

    def _close_errors(self) -> list[BaseException]:
        """Close attributable descriptors under trusted single-owner discipline.

        A different file identity is rejected.  Portable Python cannot prove
        that a same-inode descriptor number still denotes the original open file
        description, so private descriptor access must not violate ownership.
        """

        with self._lock:
            if self._closed and self._descriptor < 0 and not self._readers:
                return []
            self._closed = True
            errors: list[BaseException] = []
            for descriptor in sorted(self._readers):
                reader_errors, ownership_released = _close_owned_descriptor(
                    descriptor,
                    expected_identity=self._readers[descriptor],
                    label="captured opaque artifact duplicate reader",
                )
                errors.extend(reader_errors)
                if ownership_released:
                    self._readers.pop(descriptor, None)
            descriptor = self._descriptor
            if descriptor >= 0:
                snapshot_errors, ownership_released = _close_owned_descriptor(
                    descriptor,
                    expected_identity=self._descriptor_identity,
                    label="captured opaque artifact snapshot",
                )
                errors.extend(snapshot_errors)
                if ownership_released:
                    self._descriptor = -1
            return errors

    def _take_descriptor(self) -> int:
        """Transfer snapshot ownership to a compatibility adapter.

        This is deliberately private and only for a trusted, single-owner
        legacy adapter.  The returned raw integer has no portable generation
        identity, so the adapter must close it exactly once and must not expose
        it to competing owners. Public callers must use the managed capture and
        bounded-reader APIs, where this wrapper remains the descriptor owner.
        If transfer cannot complete, this wrapper closes every descriptor it
        still owns before propagating the primary failure.
        """

        try:
            with self._lock:
                if self._closed or self._descriptor < 0:
                    raise CapturedArtifactError(
                        "capture_closed",
                        "captured opaque artifact is already closed",
                    )
                if self._readers:
                    raise CapturedArtifactError(
                        "capture_busy",
                        "captured opaque artifact has active duplicate readers",
                    )
                self.require_intact()
                descriptor = self._descriptor
                self._descriptor = -1
                self._closed = True
                return descriptor
        except BaseException as primary_error:
            cleanup_failures = self._close_errors()
            _attach_failures(
                primary_error,
                [
                    (
                        "failed descriptor-transfer cleanup also failed",
                        cleanup_error,
                    )
                    for cleanup_error in cleanup_failures
                ],
            )
            raise


_OpaqueArtifactRequestSnapshot = tuple[str, str, int, int]


@contextmanager
def capture_resolved_opaque_file(
    resolver: CapturedOpaqueArtifactResolver,
    request: OpaqueArtifactRequest,
) -> Iterator[CapturedOpaqueFile]:
    """Detach one resolver capture before publishing it to a caller.

    The provider capture is fully validated and detached onto a helper-owned
    read-only capture. Provider teardown completes before the caller can
    observe bytes, and trusted code never closes a provider-visible descriptor
    after untrusted teardown. Retention, transfer, reuse, or poisoned provider
    bookkeeping therefore fails before publication.
    """

    _require_capture_platform()
    if type(request) is not OpaqueArtifactRequest:
        raise TypeError("request must be an exact OpaqueArtifactRequest")

    request_snapshot = _snapshot_exact_opaque_request(
        request,
        label="caller request",
    )
    resolver_request = OpaqueArtifactRequest(
        uri=request_snapshot[0],
        sha256=request_snapshot[1],
        size_bytes=request_snapshot[2],
        max_bytes=request_snapshot[3],
    )

    capture_context: AbstractContextManager[CapturedOpaqueFile] | None = None
    provider_capture: object | None = None
    detached_capture: CapturedOpaqueFile | None = None
    detached_request: OpaqueArtifactRequest | None = None
    entered = False
    provider_descriptor = -1
    provider_identity: tuple[int, int] | None = None
    primary_error: BaseException | None = None
    primary_traceback: TracebackType | None = None
    cleanup_failures: list[tuple[str, BaseException]] = []
    try:
        capture_context = resolver.capture(resolver_request)
        provider_capture = capture_context.__enter__()
        entered = True
        _require_exact_resolved_capture(
            provider_capture,
            caller_request=request,
            resolver_request=resolver_request,
            request_snapshot=request_snapshot,
        )
        provider_descriptor = provider_capture._descriptor
        provider_identity = provider_capture._descriptor_identity
        detached_capture = _detach_resolved_capture(
            provider_capture,
            request_snapshot,
        )
        detached_request = OpaqueArtifactRequest(
            uri=request_snapshot[0],
            sha256=request_snapshot[1],
            size_bytes=request_snapshot[2],
            max_bytes=request_snapshot[3],
        )
        _require_exact_resolved_capture(
            provider_capture,
            caller_request=request,
            resolver_request=resolver_request,
            request_snapshot=request_snapshot,
        )
        _require_exact_resolved_capture(
            detached_capture,
            caller_request=request,
            resolver_request=detached_request,
            request_snapshot=request_snapshot,
        )
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__
    finally:
        if entered:
            assert capture_context is not None
            try:
                # A provider may observe the active exception, but its return
                # value never decides whether that exception is suppressed.
                capture_context.__exit__(
                    None if primary_error is None else type(primary_error),
                    primary_error,
                    primary_traceback,
                )
            except BaseException as cleanup_error:
                if cleanup_error is not primary_error:
                    cleanup_failures.append(
                        ("resolver capture context cleanup also failed", cleanup_error)
                    )

        if type(provider_capture) is CapturedOpaqueFile:
            try:
                _require_request_matches_snapshot(
                    request,
                    request_snapshot,
                    label="caller request",
                )
                _require_request_matches_snapshot(
                    resolver_request,
                    request_snapshot,
                    label="resolver request",
                )
                _require_request_snapshot_matches_snapshot(
                    provider_capture._request_snapshot,
                    request_snapshot,
                    label="captured request",
                )
            except BaseException as cleanup_error:
                cleanup_failures.append(
                    (
                        "post-teardown request verification also failed",
                        cleanup_error,
                    )
                )

            if not _capture_reports_released_ownership(provider_capture):
                cleanup_failures.append(
                    (
                        "resolver capture context retained ownership",
                        CapturedArtifactCleanupError(
                            "resolver_capture_ownership_retained",
                            "resolver capture context did not release all captured "
                            "artifact descriptors",
                        ),
                    )
                )
            if provider_identity is None:
                cleanup_failures.append(
                    (
                        "captured-artifact ownership verification also failed",
                        CapturedArtifactCleanupError(
                            "capture_representation_unverifiable",
                            "provider capture ownership could not be snapshotted "
                            "before resolver teardown",
                        ),
                    )
                )
            else:
                cleanup_failures.extend(
                    (
                        "original captured descriptor release also failed",
                        cleanup_error,
                    )
                    for cleanup_error in _verify_resolver_descriptor_release(
                        provider_descriptor,
                        provider_identity,
                    )
                )

    if detached_capture is not None and detached_request is not None:
        try:
            # Provider teardown is untrusted. Revalidate the helper-owned
            # representation and bytes after it completes, before publishing
            # the detached capture to the caller.
            _require_exact_resolved_capture(
                detached_capture,
                caller_request=request,
                resolver_request=detached_request,
                request_snapshot=request_snapshot,
            )
        except BaseException as detached_error:
            if primary_error is None:
                primary_error = detached_error
                primary_traceback = detached_error.__traceback__
            else:
                cleanup_failures.append(
                    (
                        "post-teardown detached capture validation also failed",
                        detached_error,
                    )
                )

    if (primary_error is not None or cleanup_failures) and detached_capture is not None:
        cleanup_failures.extend(
            ("detached captured-artifact cleanup also failed", cleanup_error)
            for cleanup_error in CapturedOpaqueFile._close_errors(detached_capture)
        )
    if primary_error is not None:
        _attach_failures(primary_error, cleanup_failures)
        raise primary_error.with_traceback(primary_traceback)
    if cleanup_failures:
        cleanup_errors = [failure for _, failure in cleanup_failures]
        if not any(
            _contains_captured_artifact_cleanup_error(failure)
            for failure in cleanup_errors
        ):
            cleanup_errors.insert(
                0,
                CapturedArtifactCleanupError(
                    "resolver_capture_cleanup_failed",
                    "resolver capture teardown or post-teardown verification failed",
                ),
            )
        _raise_cleanup_failures(
            "resolved opaque artifact cleanup failed",
            cleanup_errors,
        )
    if detached_capture is None or detached_request is None:
        raise CapturedArtifactError(
            "resolver_capture_not_detached",
            "opaque artifact resolver did not produce a detached capture",
        )

    caller_error: BaseException | None = None
    caller_traceback: TracebackType | None = None
    caller_failures: list[tuple[str, BaseException]] = []
    try:
        try:
            yield detached_capture
        except BaseException as exc:
            caller_error = exc
            caller_traceback = exc.__traceback__
        try:
            _require_exact_resolved_capture(
                detached_capture,
                caller_request=request,
                resolver_request=detached_request,
                request_snapshot=request_snapshot,
            )
        except BaseException as integrity_error:
            if caller_error is None:
                caller_error = integrity_error
                caller_traceback = integrity_error.__traceback__
            else:
                caller_failures.append(
                    (
                        "post-use detached capture validation also failed",
                        integrity_error,
                    )
                )
    finally:
        caller_failures.extend(
            ("detached captured-artifact cleanup also failed", cleanup_error)
            for cleanup_error in CapturedOpaqueFile._close_errors(detached_capture)
        )
    if caller_error is not None:
        _attach_failures(caller_error, caller_failures)
        raise caller_error.with_traceback(caller_traceback)
    detached_cleanup_errors = [failure for _, failure in caller_failures]
    if detached_cleanup_errors:
        _raise_cleanup_failures(
            "detached resolved capture cleanup failed",
            detached_cleanup_errors,
        )


def _detach_resolved_capture(
    provider_capture: CapturedOpaqueFile,
    request_snapshot: _OpaqueArtifactRequestSnapshot,
) -> CapturedOpaqueFile:
    """Create a helper-owned capture while provider ownership is still valid."""

    try:
        source_metadata = os.fstat(provider_capture._descriptor)
    except OSError as exc:
        raise CapturedArtifactError(
            "capture_detach_failed",
            "provider capture descriptor could not be inspected for detachment",
        ) from exc
    source_identity = (source_metadata.st_dev, source_metadata.st_ino)
    if (
        not stat.S_ISREG(source_metadata.st_mode)
        or source_identity != provider_capture._descriptor_identity
        or _descriptor_state(source_metadata) != provider_capture._descriptor_state
    ):
        raise CapturedArtifactError(
            "capture_detach_failed",
            "provider capture changed before it could be detached",
        )

    if provider_capture._storage_kind == "anonymous_snapshot":
        # A read-only anonymous descriptor does not prove that the provider
        # lacks another pre-opened writer for the same unlinked inode. Always
        # copy its exact bytes into an independent helper-owned snapshot.
        return _capture_open_file_snapshot(
            provider_capture._descriptor,
            uri=request_snapshot[0],
            expected_sha256=request_snapshot[1],
            size_bytes=request_snapshot[2],
            max_bytes=request_snapshot[3],
            source_state=source_metadata,
            prefer_disk_snapshot=True,
            memfd_max_bytes=0,
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    detached_descriptor = -1
    try:
        detached_descriptor = os.open(
            f"/proc/self/fd/{provider_capture._descriptor}",
            flags,
        )
    except OSError as exc:
        raise CapturedArtifactError(
            "capture_detach_failed",
            "sealed provider capture could not be reopened as a helper-owned reader",
        ) from exc

    primary_error: BaseException | None = None
    primary_traceback: TracebackType | None = None
    try:
        detached_metadata = os.fstat(detached_descriptor)
        detached_identity = (
            detached_metadata.st_dev,
            detached_metadata.st_ino,
        )
        access_mode = fcntl.fcntl(detached_descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        if (
            not stat.S_ISREG(detached_metadata.st_mode)
            or detached_identity != source_identity
            or _descriptor_state(detached_metadata)
            != provider_capture._descriptor_state
            or access_mode != os.O_RDONLY
        ):
            raise CapturedArtifactError(
                "capture_detach_failed",
                "detached provider capture does not preserve exact read-only identity",
            )
        detached = CapturedOpaqueFile(
            request=OpaqueArtifactRequest(
                uri=request_snapshot[0],
                sha256=request_snapshot[1],
                size_bytes=request_snapshot[2],
                max_bytes=request_snapshot[3],
            ),
            descriptor=detached_descriptor,
            storage_kind=provider_capture._storage_kind,
            descriptor_state=_descriptor_state(detached_metadata),
        )
        detached.require_intact()
        detached_descriptor = -1
        return detached
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__
    finally:
        if detached_descriptor >= 0:
            try:
                os.close(detached_descriptor)
            except BaseException as cleanup_error:
                assert primary_error is not None
                primary_error.add_note(
                    "Detached descriptor cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
    assert primary_error is not None
    raise primary_error.with_traceback(primary_traceback)


def _require_exact_resolved_capture(
    captured: object,
    *,
    caller_request: OpaqueArtifactRequest,
    resolver_request: OpaqueArtifactRequest,
    request_snapshot: _OpaqueArtifactRequestSnapshot,
) -> None:
    if type(captured) is not CapturedOpaqueFile:
        raise CapturedArtifactError(
            "capture_type_mismatch",
            "resolver did not yield an exact CapturedOpaqueFile",
        )
    try:
        representation_valid = (
            type(captured._lock) is _RLOCK_TYPE
            and type(captured._readers) is dict
            and not captured._readers
            and captured._closed is False
            and type(captured._descriptor) is int
            and captured._descriptor >= 0
            and _is_descriptor_identity(captured._descriptor_identity)
            and _is_descriptor_state(captured._descriptor_state)
            and captured._descriptor_state[:2] == captured._descriptor_identity
            and type(captured._storage_kind) is str
            and captured._storage_kind in {"sealed_memfd", "anonymous_snapshot"}
            and _is_exact_opaque_request_snapshot(captured._request_snapshot)
        )
    except BaseException as exc:
        raise CapturedArtifactError(
            "capture_representation_invalid",
            "resolver yielded an incomplete concrete capture representation",
        ) from exc
    if representation_valid:
        _require_request_matches_snapshot(
            caller_request,
            request_snapshot,
            label="caller request",
        )
        _require_request_matches_snapshot(
            resolver_request,
            request_snapshot,
            label="resolver request",
        )
        _require_request_snapshot_matches_snapshot(
            captured._request_snapshot,
            request_snapshot,
            label="captured request",
        )
        CapturedOpaqueFile.require_intact(captured)
        _require_request_matches_snapshot(
            caller_request,
            request_snapshot,
            label="caller request",
        )
        _require_request_matches_snapshot(
            resolver_request,
            request_snapshot,
            label="resolver request",
        )
        _require_request_snapshot_matches_snapshot(
            captured._request_snapshot,
            request_snapshot,
            label="captured request",
        )
        _require_captured_bytes_match_snapshot(captured, request_snapshot)
        return
    raise CapturedArtifactError(
        "capture_representation_invalid",
        "resolver yielded an invalid concrete capture representation or request "
        "binding",
    )


def _snapshot_exact_opaque_request(
    request: OpaqueArtifactRequest,
    *,
    label: str,
) -> _OpaqueArtifactRequestSnapshot:
    try:
        snapshot = (
            request.uri,
            request.sha256,
            request.size_bytes,
            request.max_bytes,
        )
    except BaseException as exc:
        raise CapturedArtifactError(
            "capture_representation_invalid",
            f"{label} has an incomplete concrete representation",
        ) from exc
    if (
        type(snapshot[0]) is not str
        or type(snapshot[1]) is not str
        or type(snapshot[2]) is not int
        or type(snapshot[3]) is not int
    ):
        raise CapturedArtifactError(
            "capture_representation_invalid",
            f"{label} does not contain exact immutable scalar fields",
        )
    try:
        OpaqueArtifactRequest(
            uri=snapshot[0],
            sha256=snapshot[1],
            size_bytes=snapshot[2],
            max_bytes=snapshot[3],
        )
    except (TypeError, ValueError) as exc:
        raise CapturedArtifactError(
            "capture_representation_invalid",
            f"{label} contains invalid scalar fields",
        ) from exc
    return snapshot


def _require_request_matches_snapshot(
    request: object,
    request_snapshot: _OpaqueArtifactRequestSnapshot,
    *,
    label: str,
) -> None:
    if type(request) is not OpaqueArtifactRequest:
        raise CapturedArtifactError(
            "capture_representation_invalid",
            f"{label} is not an exact OpaqueArtifactRequest",
        )
    if _snapshot_exact_opaque_request(request, label=label) != request_snapshot:
        raise CapturedArtifactError(
            "capture_representation_invalid",
            f"{label} changed after the capture boundary snapshotted it",
        )


def _is_exact_opaque_request_snapshot(value: object) -> bool:
    return (
        type(value) is tuple
        and len(value) == 4
        and type(value[0]) is str
        and bool(value[0].strip())
        and "\x00" not in value[0]
        and type(value[1]) is str
        and _SHA256_RE.fullmatch(value[1]) is not None
        and type(value[2]) is int
        and value[2] >= 0
        and type(value[3]) is int
        and value[3] >= value[2]
    )


def _require_request_snapshot_matches_snapshot(
    value: object,
    request_snapshot: _OpaqueArtifactRequestSnapshot,
    *,
    label: str,
) -> None:
    if not _is_exact_opaque_request_snapshot(value) or value != request_snapshot:
        raise CapturedArtifactError(
            "capture_representation_invalid",
            f"{label} changed after the capture boundary snapshotted it",
        )


def _require_captured_bytes_match_snapshot(
    captured: CapturedOpaqueFile,
    request_snapshot: _OpaqueArtifactRequestSnapshot,
) -> None:
    try:
        metadata = os.fstat(captured._descriptor)
    except OSError as exc:
        raise CapturedArtifactError(
            "descriptor_invalid",
            "captured opaque artifact descriptor is no longer valid",
        ) from exc
    if metadata.st_size != request_snapshot[2]:
        raise CapturedArtifactError(
            "size_mismatch",
            "captured opaque artifact bytes disagree with the pre-call size snapshot",
        )
    if metadata.st_size > request_snapshot[3]:
        raise CapturedArtifactError(
            "max_bytes_exceeded",
            "captured opaque artifact exceeds the pre-call max_bytes snapshot",
        )
    if _stable_descriptor_sha256(captured._descriptor) != request_snapshot[1]:
        raise CapturedArtifactError(
            "snapshot_changed",
            "captured opaque artifact bytes disagree with the pre-call digest snapshot",
        )


def _is_descriptor_identity(value: object) -> bool:
    return (
        type(value) is tuple
        and len(value) == 2
        and all(type(item) is int for item in value)
    )


def _is_descriptor_state(value: object) -> bool:
    return (
        type(value) is tuple
        and len(value) == 7
        and all(type(item) is int for item in value)
    )


def _capture_reports_released_ownership(captured: CapturedOpaqueFile) -> bool:
    try:
        return (
            captured._closed is True
            and type(captured._descriptor) is int
            and captured._descriptor == -1
            and type(captured._readers) is dict
            and not captured._readers
        )
    except BaseException:
        return False


def _verify_resolver_descriptor_release(
    descriptor: int,
    expected_identity: tuple[int, int],
) -> list[BaseException]:
    """Fail closed if the original descriptor number remains open.

    Once the exact capture has released ownership, a descriptor number cannot
    safely be closed by number: it may already refer to an unrelated open-file
    description, including one for the same inode. This check therefore reports
    an unverifiable release but never closes the observed descriptor.
    """

    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        if exc.errno == errno.EBADF:
            return []
        return [
            CapturedArtifactCleanupError(
                "descriptor_release_unverifiable",
                "original captured artifact descriptor release could not be verified",
            )
        ]
    if (metadata.st_dev, metadata.st_ino) != expected_identity:
        message = (
            "original captured artifact descriptor number was reused during "
            "resolver teardown"
        )
    else:
        message = (
            "original captured artifact descriptor remains open or its number was "
            "reused for the same snapshot; release cannot be proven"
        )
    return [
        CapturedArtifactCleanupError(
            "descriptor_release_unverifiable",
            message,
        )
    ]


@contextmanager
def capture_local_opaque_file(
    path: str | Path,
    request: OpaqueArtifactRequest,
    *,
    memfd_max_bytes: int = _DEFAULT_MEMFD_MAX_BYTES,
    chunk_size: int = _DEFAULT_CHUNK_BYTES,
    snapshot_directory: str | Path | None = None,
) -> Iterator[CapturedOpaqueFile]:
    """Capture one local regular file without retaining its mutable pathname."""

    _require_capture_options(memfd_max_bytes, chunk_size)
    if type(request) is not OpaqueArtifactRequest:
        raise TypeError("request must be an exact OpaqueArtifactRequest")
    request_snapshot = _snapshot_exact_opaque_request(
        request,
        label="local capture request",
    )
    input_path = Path(path).expanduser()
    resolved = input_path.resolve(strict=True)
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    source_descriptor = os.open(resolved, flags)
    captured: CapturedOpaqueFile | None = None
    primary_error: BaseException | None = None
    primary_traceback: TracebackType | None = None
    cleanup_failures: list[tuple[str, BaseException]] = []
    try:
        opened = os.fstat(source_descriptor)
        observed = os.stat(resolved, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            observed.st_dev,
            observed.st_ino,
        ):
            raise CapturedArtifactError(
                "source_not_regular",
                "local opaque artifact must resolve to one stable regular file",
            )
        if opened.st_size != request_snapshot[2]:
            raise CapturedArtifactError(
                "size_mismatch",
                "local opaque artifact size does not match size_bytes",
            )
        captured = _capture_open_file_snapshot(
            source_descriptor,
            uri=request_snapshot[0],
            expected_sha256=request_snapshot[1],
            size_bytes=request_snapshot[2],
            max_bytes=request_snapshot[3],
            source_state=opened,
            prefer_disk_snapshot=False,
            memfd_max_bytes=memfd_max_bytes,
            chunk_size=chunk_size,
            snapshot_factory=lambda: _create_anonymous_snapshot_file(
                source_path=resolved,
                snapshot_directory=snapshot_directory,
            ),
        )
        try:
            resolved_after = input_path.resolve(strict=True)
            named_after = os.stat(resolved, follow_symlinks=False)
        except (OSError, RuntimeError) as exc:
            raise CapturedArtifactError(
                "source_changed",
                "local opaque artifact path changed while it was captured",
            ) from exc
        if resolved_after != resolved or (named_after.st_dev, named_after.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise CapturedArtifactError(
                "source_changed",
                "local opaque artifact path changed while it was captured",
            )
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__
    finally:
        try:
            os.close(source_descriptor)
        except BaseException as close_error:
            cleanup_failures.append(
                ("local source descriptor cleanup also failed", close_error)
            )

    if primary_error is not None:
        if captured is not None:
            cleanup_failures.extend(
                ("captured snapshot cleanup also failed", cleanup_error)
                for cleanup_error in captured._close_errors()
            )
        _attach_failures(primary_error, cleanup_failures)
        raise primary_error.with_traceback(primary_traceback)
    if cleanup_failures:
        assert captured is not None
        cleanup_failures.extend(
            ("captured snapshot cleanup also failed", cleanup_error)
            for cleanup_error in captured._close_errors()
        )
        _raise_cleanup_failures(
            "local opaque artifact capture cleanup failed",
            [failure for _, failure in cleanup_failures],
        )
    assert captured is not None
    with _managed_capture(captured) as managed:
        yield managed


@contextmanager
def capture_streamed_opaque_file(
    stream: BoundedBinaryReader,
    request: OpaqueArtifactRequest,
    *,
    memfd_max_bytes: int = _DEFAULT_MEMFD_MAX_BYTES,
    chunk_size: int = _DEFAULT_CHUNK_BYTES,
    snapshot_directory: str | Path | None = None,
) -> Iterator[CapturedOpaqueFile]:
    """Capture one caller-owned bounded stream into an immutable snapshot.

    The stream itself remains caller-owned. A remote resolver should own its
    transport context outside this function and delegate only the byte capture.
    """

    _require_capture_options(memfd_max_bytes, chunk_size)
    if type(request) is not OpaqueArtifactRequest:
        raise TypeError("request must be an exact OpaqueArtifactRequest")
    request_snapshot = _snapshot_exact_opaque_request(
        request,
        label="stream capture request",
    )
    if not callable(getattr(stream, "read", None)):
        raise TypeError("stream must provide a bounded read(size) method")

    def read_chunk(size: int, _offset: int) -> bytes:
        chunk = stream.read(size)
        if type(chunk) is not bytes:
            raise CapturedArtifactError(
                "stream_protocol_invalid",
                "opaque artifact stream read(size) must return exact bytes",
            )
        if len(chunk) > size:
            raise CapturedArtifactError(
                "stream_protocol_invalid",
                "opaque artifact stream returned more bytes than requested",
            )
        return chunk

    captured = _capture_reader_snapshot(
        read_chunk,
        uri=request_snapshot[0],
        expected_sha256=request_snapshot[1],
        size_bytes=request_snapshot[2],
        max_bytes=request_snapshot[3],
        prefer_disk_snapshot=False,
        memfd_max_bytes=memfd_max_bytes,
        chunk_size=chunk_size,
        snapshot_factory=lambda: _create_anonymous_snapshot_file(
            source_path=None,
            snapshot_directory=snapshot_directory,
        ),
        source_stability_check=None,
    )
    with _managed_capture(captured) as managed:
        yield managed


@contextmanager
def _managed_capture(
    captured: CapturedOpaqueFile,
) -> Iterator[CapturedOpaqueFile]:
    primary_error: BaseException | None = None
    primary_traceback: TracebackType | None = None
    additional_failures: list[tuple[str, BaseException]] = []
    try:
        try:
            captured.require_intact()
        except BaseException as exc:
            primary_error = exc
            primary_traceback = exc.__traceback__
        else:
            try:
                yield captured
            except BaseException as exc:
                primary_error = exc
                primary_traceback = exc.__traceback__
            try:
                captured.require_intact()
            except BaseException as integrity_error:
                if primary_error is None:
                    primary_error = integrity_error
                    primary_traceback = integrity_error.__traceback__
                else:
                    additional_failures.append(
                        ("post-use integrity check also failed", integrity_error)
                    )
    finally:
        additional_failures.extend(
            ("captured snapshot cleanup also failed", cleanup_error)
            for cleanup_error in captured._close_errors()
        )
    if primary_error is not None:
        _attach_failures(primary_error, additional_failures)
        raise primary_error.with_traceback(primary_traceback)
    cleanup_errors = [failure for _, failure in additional_failures]
    if cleanup_errors:
        _raise_cleanup_failures(
            "captured opaque artifact cleanup failed",
            cleanup_errors,
        )


def _capture_open_file_snapshot(
    descriptor: int,
    *,
    uri: str,
    expected_sha256: str | None,
    size_bytes: int,
    max_bytes: int,
    source_state: os.stat_result | None = None,
    prefer_disk_snapshot: bool = False,
    memfd_max_bytes: int = _DEFAULT_MEMFD_MAX_BYTES,
    chunk_size: int = _DEFAULT_CHUNK_BYTES,
    snapshot_factory: Callable[[], BinaryIO] | None = None,
    memfd_create: Any = _USE_DEFAULT_MEMFD_CREATE,
) -> CapturedOpaqueFile:
    """Capture one already-open local descriptor for compatibility adapters."""

    _require_capture_fields(
        uri=uri,
        expected_sha256=expected_sha256,
        size_bytes=size_bytes,
        max_bytes=max_bytes,
    )
    _require_capture_options(memfd_max_bytes, chunk_size)
    before = os.fstat(descriptor) if source_state is None else source_state
    if not stat.S_ISREG(before.st_mode):
        raise CapturedArtifactError(
            "source_not_regular",
            "opaque artifact source descriptor must identify a regular file",
        )
    if before.st_size != size_bytes:
        raise CapturedArtifactError(
            "size_mismatch",
            "opaque artifact source size does not match size_bytes",
        )

    def read_chunk(size: int, offset: int) -> bytes:
        return os.pread(descriptor, size, offset)

    def require_source_stable(captured_sha256: str) -> None:
        try:
            live_sha256 = _stable_descriptor_sha256(descriptor)
        except CapturedArtifactError as exc:
            raise CapturedArtifactError(
                "source_changed",
                "opaque artifact source changed while it was captured",
            ) from exc
        after = os.fstat(descriptor)
        if (
            _descriptor_state(after) != _descriptor_state(before)
            or live_sha256 != captured_sha256
        ):
            raise CapturedArtifactError(
                "source_changed",
                "opaque artifact source changed while it was captured",
            )

    if snapshot_factory is None:

        def snapshot_factory() -> BinaryIO:
            return _create_anonymous_snapshot_file(
                source_path=None,
                snapshot_directory=None,
            )

    return _capture_reader_snapshot(
        read_chunk,
        uri=uri,
        expected_sha256=expected_sha256,
        size_bytes=size_bytes,
        max_bytes=max_bytes,
        prefer_disk_snapshot=prefer_disk_snapshot,
        memfd_max_bytes=memfd_max_bytes,
        chunk_size=chunk_size,
        snapshot_factory=snapshot_factory,
        source_stability_check=require_source_stable,
        memfd_create=memfd_create,
    )


def _capture_reader_snapshot(
    read_chunk: Callable[[int, int], bytes],
    *,
    uri: str,
    expected_sha256: str | None,
    size_bytes: int,
    max_bytes: int,
    prefer_disk_snapshot: bool,
    memfd_max_bytes: int,
    chunk_size: int,
    snapshot_factory: Callable[[], BinaryIO],
    source_stability_check: Callable[[str], None] | None,
    memfd_create: Any = _USE_DEFAULT_MEMFD_CREATE,
) -> CapturedOpaqueFile:
    _require_capture_fields(
        uri=uri,
        expected_sha256=expected_sha256,
        size_bytes=size_bytes,
        max_bytes=max_bytes,
    )
    selected_memfd_create = (
        _MEMFD_CREATE if memfd_create is _USE_DEFAULT_MEMFD_CREATE else memfd_create
    )
    use_memfd = (
        not prefer_disk_snapshot
        and selected_memfd_create is not None
        and size_bytes <= memfd_max_bytes
    )
    storage_kind: CapturedStorageKind = "anonymous_snapshot"
    binding_descriptor = -1
    writer_descriptor = -1
    snapshot_file: BinaryIO | None = None
    primary_error: BaseException | None = None
    primary_traceback: TracebackType | None = None
    cleanup_failures: list[tuple[str, BaseException]] = []
    try:
        if use_memfd:
            try:
                writer_descriptor = int(
                    selected_memfd_create(
                        b"wu-captured-artifact",
                        _MFD_CLOEXEC | _MFD_ALLOW_SEALING,
                    )
                )
                if writer_descriptor < 0:
                    error_number = ctypes.get_errno()
                    raise OSError(error_number, os.strerror(error_number))
            except OSError as exc:
                writer_descriptor = -1
                if exc.errno not in _MEMFD_AVAILABILITY_ERRNOS:
                    raise
                use_memfd = False
            else:
                storage_kind = "sealed_memfd"
        if not use_memfd:
            snapshot_file = snapshot_factory()
            writer_descriptor = snapshot_file.fileno()

        digest = hashlib.sha256()
        offset = 0
        while offset < size_bytes:
            requested = min(chunk_size, size_bytes - offset)
            chunk = read_chunk(requested, offset)
            if type(chunk) is not bytes:
                raise CapturedArtifactError(
                    "stream_protocol_invalid",
                    "opaque artifact reader must return exact bytes",
                )
            if len(chunk) > requested:
                raise CapturedArtifactError(
                    "stream_protocol_invalid",
                    "opaque artifact reader returned more bytes than requested",
                )
            if not chunk:
                raise CapturedArtifactError(
                    "size_mismatch",
                    "opaque artifact ended before its declared size_bytes",
                )
            if offset + len(chunk) > max_bytes:
                raise CapturedArtifactError(
                    "max_bytes_exceeded",
                    "opaque artifact exceeded its declared max_bytes",
                )
            digest.update(chunk)
            _write_all(writer_descriptor, chunk)
            offset += len(chunk)
        extra = read_chunk(1, offset)
        if type(extra) is not bytes or len(extra) > 1:
            raise CapturedArtifactError(
                "stream_protocol_invalid",
                "opaque artifact reader violated bounded read semantics",
            )
        if extra:
            code = "max_bytes_exceeded" if offset >= max_bytes else "size_mismatch"
            message = (
                "opaque artifact exceeded its declared max_bytes"
                if code == "max_bytes_exceeded"
                else "opaque artifact exceeded its declared size_bytes"
            )
            raise CapturedArtifactError(code, message)
        actual_sha256 = digest.hexdigest()
        if source_stability_check is not None:
            source_stability_check(actual_sha256)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise CapturedArtifactError(
                "digest_mismatch",
                "opaque artifact bytes do not match the declared sha256",
            )
        request = OpaqueArtifactRequest(
            uri=uri,
            sha256=actual_sha256,
            size_bytes=size_bytes,
            max_bytes=max_bytes,
        )
        os.fsync(writer_descriptor)
        if use_memfd:
            fcntl.fcntl(writer_descriptor, _F_ADD_SEALS, _CAPTURE_MEMFD_SEALS)
            binding_descriptor = writer_descriptor
            writer_descriptor = -1
        else:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NONBLOCK", 0)
            binding_descriptor = os.open(
                f"/proc/self/fd/{writer_descriptor}",
                flags,
            )
            os.fchmod(writer_descriptor, 0)
            if _stable_descriptor_sha256(binding_descriptor) != actual_sha256:
                raise CapturedArtifactError(
                    "snapshot_changed",
                    "anonymous opaque artifact snapshot changed while finalized",
                )
            assert snapshot_file is not None
            snapshot_file.close()
            snapshot_file = None
            writer_descriptor = -1
        descriptor_state = _descriptor_state(os.fstat(binding_descriptor))
        captured = CapturedOpaqueFile(
            request=request,
            descriptor=binding_descriptor,
            storage_kind=storage_kind,
            descriptor_state=descriptor_state,
        )
        captured.require_intact()
        binding_descriptor = -1
        return captured
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__
    finally:
        if binding_descriptor >= 0:
            cleanup_failures.extend(
                (
                    "partial snapshot descriptor cleanup also failed",
                    cleanup_error,
                )
                for cleanup_error in _close_owned_descriptor_without_identity(
                    binding_descriptor
                )
            )
        if snapshot_file is not None:
            try:
                snapshot_file.close()
            except BaseException as cleanup_error:
                cleanup_failures.append(
                    ("snapshot writer cleanup also failed", cleanup_error)
                )
        elif writer_descriptor >= 0:
            cleanup_failures.extend(
                (
                    "snapshot writer cleanup also failed",
                    cleanup_error,
                )
                for cleanup_error in _close_owned_descriptor_without_identity(
                    writer_descriptor
                )
            )
    assert primary_error is not None
    _attach_failures(primary_error, cleanup_failures)
    raise primary_error.with_traceback(primary_traceback)


def _create_anonymous_snapshot_file(
    *,
    source_path: Path | None,
    snapshot_directory: str | Path | None,
) -> BinaryIO:
    failures: list[str] = []
    for directory in _snapshot_candidate_directories(
        source_path=source_path,
        snapshot_directory=snapshot_directory,
    ):
        try:
            candidate = tempfile.TemporaryFile(
                prefix="wu-captured-artifact-",
                dir=directory,
            )
        except OSError as exc:
            failures.append(f"{directory}: {exc}")
            continue
        try:
            filesystem_types = _descriptor_filesystem_types(candidate.fileno())
        except OSError as exc:
            try:
                candidate.close()
            except BaseException as cleanup_error:
                exc.add_note(
                    "Rejected snapshot candidate cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                raise exc from cleanup_error
            failures.append(f"{directory}: {exc}")
            continue
        memory_backed = sorted(
            filesystem_types.intersection(_MEMORY_BACKED_FILESYSTEM_TYPES)
        )
        if memory_backed:
            try:
                candidate.close()
            except BaseException as cleanup_error:
                raise CapturedArtifactCleanupError(
                    "snapshot_candidate_cleanup_failed",
                    "rejected memory-backed snapshot candidate could not be "
                    f"closed: {cleanup_error}",
                ) from cleanup_error
            failures.append(
                f"{directory}: memory-backed filesystem " + ", ".join(memory_backed)
            )
            continue
        return candidate
    detail = "; ".join(failures) or "no candidate directories"
    raise CapturedArtifactError(
        "snapshot_storage_unavailable",
        "captured opaque artifacts require writable non-memory-backed anonymous "
        f"snapshot storage; tried: {detail}",
    )


def _snapshot_candidate_directories(
    *,
    source_path: Path | None,
    snapshot_directory: str | Path | None,
) -> tuple[Path, ...]:
    configured: str | Path | None = snapshot_directory
    if configured is None:
        environment_value = os.environ.get(_SNAPSHOT_DIRECTORY_ENV)
        if environment_value is not None:
            if not environment_value.strip():
                raise CapturedArtifactError(
                    "snapshot_directory_invalid",
                    f"{_SNAPSHOT_DIRECTORY_ENV} must name a non-empty directory",
                )
            configured = environment_value
    if configured is not None:
        return (Path(os.path.abspath(Path(configured).expanduser())),)
    candidates = [Path(tempfile.gettempdir()), Path("/var/tmp")]
    if source_path is not None:
        candidates.append(source_path.parent)
    normalized: list[Path] = []
    for candidate in candidates:
        absolute = Path(os.path.abspath(candidate.expanduser()))
        if absolute not in normalized:
            normalized.append(absolute)
    return tuple(normalized)


def _descriptor_filesystem_types(descriptor: int) -> frozenset[str]:
    state = os.fstat(descriptor)
    device = f"{os.major(state.st_dev)}:{os.minor(state.st_dev)}"
    filesystem_types: set[str] = set()
    with Path("/proc/self/mountinfo").open(encoding="utf-8") as mount_info:
        for line in mount_info:
            fields = line.split()
            if len(fields) < 7 or fields[2] != device:
                continue
            try:
                separator = fields.index("-", 6)
            except ValueError:
                continue
            if separator + 1 < len(fields):
                filesystem_types.add(fields[separator + 1])
    if not filesystem_types:
        raise OSError(
            f"could not identify filesystem type for descriptor device {device}"
        )
    return frozenset(filesystem_types)


def _stable_descriptor_sha256(descriptor: int) -> str:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise CapturedArtifactError(
            "snapshot_not_regular",
            "captured opaque artifact descriptor is not a regular file",
        )
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(
            descriptor,
            min(_DEFAULT_CHUNK_BYTES, before.st_size - offset),
            offset,
        )
        if not chunk:
            raise CapturedArtifactError(
                "snapshot_changed",
                "captured opaque artifact changed while hashing",
            )
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, offset):
        raise CapturedArtifactError(
            "snapshot_changed",
            "captured opaque artifact grew while hashing",
        )
    after = os.fstat(descriptor)
    if _descriptor_state(before) != _descriptor_state(after):
        raise CapturedArtifactError(
            "snapshot_changed",
            "captured opaque artifact changed while hashing",
        )
    return digest.hexdigest()


def _descriptor_state(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:  # pragma: no cover - regular-file OS invariant
            raise OSError("short write while capturing opaque artifact")
        remaining = remaining[written:]


def _close_owned_descriptor(
    descriptor: int,
    *,
    expected_identity: tuple[int, int],
    label: str,
) -> tuple[list[BaseException], bool]:
    """Return cleanup failures and whether this wrapper released ownership."""

    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        if exc.errno != errno.EBADF:
            return (
                [
                    CapturedArtifactCleanupError(
                        "descriptor_state_unverifiable",
                        f"{label} descriptor state could not be inspected; "
                        f"ownership remains retained: {exc}",
                    )
                ],
                False,
            )
        return (
            [
                CapturedArtifactCleanupError(
                    "descriptor_already_closed",
                    f"{label} descriptor was already closed: {exc}",
                )
            ],
            True,
        )
    if (metadata.st_dev, metadata.st_ino) != expected_identity:
        return (
            [
                CapturedArtifactCleanupError(
                    "descriptor_reused",
                    f"{label} descriptor now identifies a different file; "
                    "refusing to close it",
                )
            ],
            True,
        )
    try:
        os.close(descriptor)
    except BaseException as exc:
        try:
            after_failure = os.fstat(descriptor)
        except OSError as state_error:
            if state_error.errno == errno.EBADF:
                return [exc], True
            return (
                [
                    exc,
                    CapturedArtifactCleanupError(
                        "descriptor_state_unverifiable",
                        f"{label} descriptor state could not be inspected after "
                        f"close failed; ownership remains retained: {state_error}",
                    ),
                ],
                False,
            )
        if (after_failure.st_dev, after_failure.st_ino) != expected_identity:
            return [exc], True
        return [exc], False
    return [], True


def _close_owned_descriptor_without_identity(
    descriptor: int,
) -> list[BaseException]:
    try:
        os.close(descriptor)
    except BaseException as exc:
        return [exc]
    return []


def _require_capture_fields(
    *,
    uri: str,
    expected_sha256: str | None,
    size_bytes: int,
    max_bytes: int,
) -> None:
    if type(uri) is not str:
        raise TypeError("opaque artifact uri must be a string")
    if not uri.strip() or "\x00" in uri:
        raise ValueError("opaque artifact uri must be nonblank")
    if expected_sha256 is not None:
        if type(expected_sha256) is not str:
            raise TypeError("opaque artifact sha256 must be a string")
        if _SHA256_RE.fullmatch(expected_sha256) is None:
            raise ValueError(
                "opaque artifact sha256 must be 64 lowercase hexadecimal characters"
            )
    _require_nonnegative_integer(size_bytes, "opaque artifact size_bytes")
    _require_nonnegative_integer(max_bytes, "opaque artifact max_bytes")
    if size_bytes > max_bytes:
        raise ValueError("opaque artifact size_bytes exceeds its declared max_bytes")


def _require_capture_options(memfd_max_bytes: int, chunk_size: int) -> None:
    _require_capture_platform()
    _require_nonnegative_integer(memfd_max_bytes, "memfd_max_bytes")
    _require_chunk_size(chunk_size)


def _require_chunk_size(chunk_size: int) -> None:
    if type(chunk_size) is not int:
        raise TypeError("chunk_size must be an integer")
    if not 1 <= chunk_size <= _MAX_CHUNK_BYTES:
        raise ValueError(f"chunk_size must be between 1 and {_MAX_CHUNK_BYTES} bytes")


def _require_nonnegative_integer(value: int, label: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")


def _attach_failures(
    primary_error: BaseException,
    failures: list[tuple[str, BaseException]],
) -> None:
    for context, failure in failures:
        primary_error.add_note(f"{context}: {type(failure).__name__}: {failure}")


def _raise_cleanup_failures(
    label: str,
    failures: list[BaseException],
) -> None:
    if len(failures) == 1:
        raise failures[0]
    raise BaseExceptionGroup(label, failures)


def _contains_captured_artifact_cleanup_error(failure: BaseException) -> bool:
    pending = [failure]
    while pending:
        current = pending.pop()
        if isinstance(current, CapturedArtifactCleanupError):
            return True
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return False


__all__ = [
    "BoundedBinaryReader",
    "CapturedArtifactCleanupError",
    "CapturedArtifactError",
    "CapturedOpaqueArtifactResolver",
    "CapturedOpaqueFile",
    "CapturedStorageKind",
    "OpaqueArtifactRequest",
    "capture_local_opaque_file",
    "capture_resolved_opaque_file",
    "capture_streamed_opaque_file",
]
