# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private staging and atomic no-overwrite publication for static evidence.

The transaction is intentionally ignorant of authoring, readback, and Gate 3
meaning.  It accepts only the already-admitted evidence allowlist, writes each
non-terminal file once below a random private sibling, and links the completed
tree into its final name with Linux ``renameat2(RENAME_NOREPLACE)``.  The final
``completion.json`` is caller-owned evidence and is the last file written.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import shutil
import stat
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from world_understanding.utils.artifacts import (
    ArtifactPathError,
    open_confined_directory,
    validated_artifact_relative_key,
)
from world_understanding.utils.captured_artifacts import CapturedOpaqueFile

from joint_agent.articulation_v2_static_evidence_contracts import (
    STATIC_EVIDENCE_GATE3_FAILURE_RECEIPT_PATH,
    ArticulationV2StaticEvidencePathAllowlistV1,
    articulation_v2_static_gate3a_failure_paths,
    articulation_v2_static_gate3b_failure_paths,
)

_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_RENAME_NOREPLACE = 1
_LIBC = ctypes.CDLL(None, use_errno=True)
_RENAMEAT2 = getattr(_LIBC, "renameat2", None)
if _RENAMEAT2 is not None:
    _RENAMEAT2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    _RENAMEAT2.restype = ctypes.c_int


class ArticulationV2StaticEvidencePublicationError(RuntimeError):
    """Fail-closed evidence staging or publication error."""


class ArticulationV2StaticEvidenceDurabilityUnknown(
    ArticulationV2StaticEvidencePublicationError
):
    """The final root was renamed, but its parent directory could not be synced."""

    committed: Literal[True] = True


@dataclass(frozen=True, slots=True)
class ArticulationV2StaticPublishedFileV1:
    """One private-write identity captured before the root commit point."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ArticulationV2StaticEvidencePublicationResultV1:
    """The exact file inventory historically committed under one final root."""

    destination: Path
    files: tuple[ArticulationV2StaticPublishedFileV1, ...]


@dataclass(frozen=True, slots=True)
class _EvidencePathAuthority:
    """Private immutable snapshot of the validated evidence-path contract."""

    schema_version: Literal[
        "joint-agent-articulation-v2-static-evidence-path-allowlist-v1"
    ]
    constraint_kind: Literal["fixed", "distance"]
    generated_output: str
    paths: tuple[str, ...]
    retained_paths: tuple[str, ...]
    terminal_receipt: Literal["completion.json"]

    @classmethod
    def from_allowlist(
        cls,
        allowlist: ArticulationV2StaticEvidencePathAllowlistV1,
    ) -> _EvidencePathAuthority:
        return cls(
            schema_version=allowlist.schema_version,
            constraint_kind=allowlist.constraint_kind,
            generated_output=allowlist.generated_output,
            paths=tuple(allowlist.paths),
            retained_paths=tuple(allowlist.retained_paths),
            terminal_receipt=allowlist.terminal_receipt,
        )

    def to_allowlist(self) -> ArticulationV2StaticEvidencePathAllowlistV1:
        """Return a defensive, strictly revalidated model for introspection."""

        return ArticulationV2StaticEvidencePathAllowlistV1(
            schema_version=self.schema_version,
            constraint_kind=self.constraint_kind,
            generated_output=self.generated_output,
            paths=self.paths,
            terminal_receipt=self.terminal_receipt,
        )


class ArticulationV2StaticEvidenceTransaction:
    """One uncommitted private evidence root, valid only inside its context."""

    __slots__ = (
        "_authority",
        "_closed",
        "_committed",
        "_file_states",
        "_parent_context",
        "_parent_descriptor",
        "_parent_identity",
        "_parent_path",
        "_staging_descriptor",
        "_staging_identity",
        "_staging_name",
        "_target_name",
        "_written",
    )

    def __init__(
        self,
        *,
        allowlist: ArticulationV2StaticEvidencePathAllowlistV1,
        parent_context: AbstractContextManager[int],
        parent_descriptor: int,
        parent_path: str,
        target_name: str,
        staging_name: str,
        staging_descriptor: int,
    ) -> None:
        self._authority = _EvidencePathAuthority.from_allowlist(allowlist)
        self._parent_context: AbstractContextManager[int] | None = parent_context
        self._parent_descriptor = parent_descriptor
        parent_metadata = os.fstat(parent_descriptor)
        self._parent_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
        self._parent_path = parent_path
        self._target_name = target_name
        self._staging_name = staging_name
        self._staging_descriptor = staging_descriptor
        metadata = os.fstat(staging_descriptor)
        self._staging_identity = (metadata.st_dev, metadata.st_ino)
        # Keep only primitive values under transaction control.  Public frozen
        # Pydantic/dataclass-like records are still mutable through
        # ``object.__setattr__`` and must never become commit authority.
        self._written: dict[str, tuple[int, str]] = {}
        self._file_states: dict[str, tuple[int, int, int, int, int, int, int]] = {}
        self._closed = False
        self._committed = False

    @property
    def allowlist(self) -> ArticulationV2StaticEvidencePathAllowlistV1:
        return self._authority.to_allowlist()

    @property
    def committed(self) -> bool:
        return self._committed

    def write_bytes(
        self, path: str, payload: bytes
    ) -> ArticulationV2StaticPublishedFileV1:
        """Write one allowed non-terminal file once through held descriptors."""

        if type(payload) is not bytes or not payload:
            raise ArticulationV2StaticEvidencePublicationError(
                "evidence payloads must be nonempty exact bytes"
            )
        return self._write(path, (payload,), expected_sha256=None, expected_size=None)

    def write_capture(
        self,
        path: str,
        capture: CapturedOpaqueFile,
    ) -> ArticulationV2StaticPublishedFileV1:
        """Copy a retained opaque snapshot without reopening a mutable path."""

        if type(capture) is not CapturedOpaqueFile:
            raise TypeError("capture must be an exact CapturedOpaqueFile")
        capture.require_intact()
        if capture.size_bytes == 0:
            raise ArticulationV2StaticEvidencePublicationError(
                "evidence captures must be nonempty"
            )
        record = self._write(
            path,
            capture.iter_chunks(),
            expected_sha256=capture.sha256,
            expected_size=capture.size_bytes,
        )
        capture.require_intact()
        return record

    def commit(
        self,
        completion_payload: bytes,
    ) -> ArticulationV2StaticEvidencePublicationResultV1:
        """Write the terminal receipt and atomically publish the completed root."""

        self._require_open()
        if self._committed:
            raise ArticulationV2StaticEvidencePublicationError(
                "evidence transaction is already committed"
            )
        expected = set(self._authority.retained_paths)
        actual = set(self._written)
        if actual != expected:
            raise ArticulationV2StaticEvidencePublicationError(
                "evidence root is incomplete or contains an unadmitted file: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        self._write_terminal(completion_payload)
        if set(self._written) != set(self._authority.paths):  # defensive invariant
            raise ArticulationV2StaticEvidencePublicationError(
                "terminal receipt did not complete the admitted evidence inventory"
            )
        return self._commit_written_root(self._authority.paths)

    def commit_gate3a_failure(
        self,
        failure_payload: bytes,
    ) -> ArticulationV2StaticEvidencePublicationResultV1:
        """Atomically publish the exact non-qualifying Gate 3A failure root."""

        self._require_open()
        if self._committed:
            raise ArticulationV2StaticEvidencePublicationError(
                "evidence transaction is already committed"
            )
        expected_paths = articulation_v2_static_gate3a_failure_paths(
            constraint_kind=self._authority.constraint_kind,
            generated_output=self._authority.generated_output,
        )
        return self._commit_gate3_failure(
            failure_payload,
            expected_paths=expected_paths,
            label="Gate 3A",
        )

    def commit_gate3b_failure(
        self,
        failure_payload: bytes,
    ) -> ArticulationV2StaticEvidencePublicationResultV1:
        """Atomically publish the exact non-qualifying Gate 3B failure root."""

        expected_paths = articulation_v2_static_gate3b_failure_paths(
            constraint_kind=self._authority.constraint_kind,
            generated_output=self._authority.generated_output,
        )
        return self._commit_gate3_failure(
            failure_payload,
            expected_paths=expected_paths,
            label="Gate 3B",
        )

    def _commit_gate3_failure(
        self,
        failure_payload: bytes,
        *,
        expected_paths: tuple[str, ...],
        label: Literal["Gate 3A", "Gate 3B"],
    ) -> ArticulationV2StaticEvidencePublicationResultV1:
        """Atomically publish one exact non-qualifying Gate 3 failure root."""

        self._require_open()
        if self._committed:
            raise ArticulationV2StaticEvidencePublicationError(
                "evidence transaction is already committed"
            )
        expected_retained = set(expected_paths) - {
            STATIC_EVIDENCE_GATE3_FAILURE_RECEIPT_PATH
        }
        actual = set(self._written)
        if actual != expected_retained:
            raise ArticulationV2StaticEvidencePublicationError(
                f"{label} failure root is incomplete or contains an unadmitted file: "
                f"missing={sorted(expected_retained - actual)}, "
                f"extra={sorted(actual - expected_retained)}"
            )
        self._write_failure_terminal(failure_payload)
        if set(self._written) != set(expected_paths):  # defensive invariant
            raise ArticulationV2StaticEvidencePublicationError(
                f"terminal {label} failure receipt did not complete its inventory"
            )
        return self._commit_written_root(expected_paths)

    def _commit_written_root(
        self,
        publication_paths: tuple[str, ...],
    ) -> ArticulationV2StaticEvidencePublicationResultV1:
        """Commit one already-complete success or typed-failure inventory."""

        self._verify_destination_parent_path()
        self._verify_staging_for_commit()
        os.fsync(self._staging_descriptor)
        try:
            _rename_no_replace(
                self._parent_descriptor,
                self._staging_name,
                self._target_name,
            )
        except BaseException as exc:
            try:
                renamed = self._reconcile_rename_outcome()
            except (ArticulationV2StaticEvidencePublicationError, OSError) as probe:
                self._committed = True
                raise ArticulationV2StaticEvidenceDurabilityUnknown(
                    "evidence_rename_outcome_unknown: atomic rename was interrupted "
                    "and its outcome could not be probed; do not retry blindly"
                ) from probe
            if not renamed:
                self._raise_rename_failure(exc)
        # The no-replace rename is the irrevocable commit point.  Every later
        # observation can fail, but it must not be reported as an uncommitted
        # transaction that a caller may safely retry.
        self._committed = True
        return self._finish_committed_publication(publication_paths)

    def _finish_committed_publication(
        self,
        publication_paths: tuple[str, ...],
    ) -> ArticulationV2StaticEvidencePublicationResultV1:
        try:
            os.fsync(self._parent_descriptor)
        except OSError as exc:
            raise ArticulationV2StaticEvidenceDurabilityUnknown(
                "evidence_committed_durability_unknown: final root was renamed but "
                "the parent directory could not be synced; do not retry blindly"
            ) from exc
        try:
            self._verify_destination_parent_path()
            self._verify_published_target_identity()
            self._verify_private_staging_contents()
        except (ArticulationV2StaticEvidencePublicationError, OSError) as exc:
            raise ArticulationV2StaticEvidenceDurabilityUnknown(
                "evidence_committed_verification_unknown: final evidence root was "
                "renamed but its destination or recorded contents could not be "
                "verified; do not retry blindly"
            ) from exc
        return ArticulationV2StaticEvidencePublicationResultV1(
            destination=Path(self._parent_path) / self._target_name,
            files=tuple(self._published_file(path) for path in publication_paths),
        )

    def _write(
        self,
        path: str,
        chunks: Iterator[bytes] | tuple[bytes, ...],
        *,
        expected_sha256: str | None,
        expected_size: int | None,
    ) -> ArticulationV2StaticPublishedFileV1:
        self._require_open()
        if self._committed:
            raise ArticulationV2StaticEvidencePublicationError(
                "cannot write after evidence publication"
            )
        canonical = _require_allowed_path(path, self._authority, terminal_allowed=False)
        if canonical in self._written:
            raise ArticulationV2StaticEvidencePublicationError(
                "evidence files may be written only once"
            )
        parts = tuple(canonical.split("/"))
        digest = hashlib.sha256()
        size = 0
        with _open_private_directory(
            self._staging_descriptor,
            parts[:-1],
            create=True,
        ) as parent:
            descriptor = os.open(
                parts[-1],
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent,
            )
            try:
                for chunk in chunks:
                    if type(chunk) is not bytes or not chunk:
                        raise ArticulationV2StaticEvidencePublicationError(
                            "evidence streams must yield nonempty exact bytes"
                        )
                    _write_all(descriptor, chunk)
                    digest.update(chunk)
                    size += len(chunk)
                if size == 0:
                    raise ArticulationV2StaticEvidencePublicationError(
                        "evidence payloads must be nonempty"
                    )
                if expected_size is not None and size != expected_size:
                    raise ArticulationV2StaticEvidencePublicationError(
                        "retained evidence capture size changed during publication"
                    )
                if (
                    expected_sha256 is not None
                    and digest.hexdigest() != expected_sha256
                ):
                    raise ArticulationV2StaticEvidencePublicationError(
                        "retained evidence capture digest changed during publication"
                    )
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
                file_state = _file_state(os.fstat(descriptor))
            finally:
                os.close(descriptor)
            os.fsync(parent)
        snapshot = (size, digest.hexdigest())
        self._written[canonical] = snapshot
        self._file_states[canonical] = file_state
        return self._published_file(canonical)

    def _write_terminal(self, payload: bytes) -> None:
        # Kept separate from public `write_bytes` so completion is always last.
        terminal = self._authority.terminal_receipt
        if type(payload) is not bytes or not payload:
            raise ArticulationV2StaticEvidencePublicationError(
                "terminal completion receipt must be nonempty exact bytes"
            )
        if terminal in self._written:
            raise ArticulationV2StaticEvidencePublicationError(
                "terminal completion receipt was already written"
            )
        self._write_allowed_terminal(terminal, payload)

    def _write_failure_terminal(self, payload: bytes) -> None:
        terminal = STATIC_EVIDENCE_GATE3_FAILURE_RECEIPT_PATH
        if type(payload) is not bytes or not payload:
            raise ArticulationV2StaticEvidencePublicationError(
                "terminal Gate 3A failure receipt must be nonempty exact bytes"
            )
        if terminal in self._written:
            raise ArticulationV2StaticEvidencePublicationError(
                "terminal Gate 3A failure receipt was already written"
            )
        self._write_allowed_terminal(terminal, payload)

    def _write_allowed_terminal(self, path: str, payload: bytes) -> None:
        parts = tuple(path.split("/"))
        digest = hashlib.sha256(payload).hexdigest()
        with _open_private_directory(
            self._staging_descriptor,
            parts[:-1],
            create=True,
        ) as parent:
            descriptor = os.open(
                parts[-1],
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent,
            )
            try:
                _write_all(descriptor, payload)
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
                file_state = _file_state(os.fstat(descriptor))
            finally:
                os.close(descriptor)
            os.fsync(parent)
        self._written[path] = (len(payload), digest)
        self._file_states[path] = file_state

    def _published_file(self, path: str) -> ArticulationV2StaticPublishedFileV1:
        """Build a fresh public record from the private scalar snapshot."""

        size_bytes, sha256 = self._written[path]
        return ArticulationV2StaticPublishedFileV1(
            path=path,
            size_bytes=size_bytes,
            sha256=sha256,
        )

    def _verify_destination_parent_path(self) -> None:
        """Ensure the lexical destination parent still names the held directory."""

        try:
            with open_confined_directory(self._parent_path) as current:
                metadata = os.fstat(current)
        except (ArtifactPathError, OSError, ValueError) as exc:
            raise ArticulationV2StaticEvidencePublicationError(
                "evidence destination parent changed during publication"
            ) from exc
        if (metadata.st_dev, metadata.st_ino) != self._parent_identity:
            raise ArticulationV2StaticEvidencePublicationError(
                "evidence destination parent changed during publication"
            )

    def _verify_published_target_identity(self) -> None:
        """Require the final name to retain the private root just verified."""

        target = _stat_child_or_none(self._parent_descriptor, self._target_name)
        if (
            target is None
            or not stat.S_ISDIR(target.st_mode)
            or (target.st_dev, target.st_ino) != self._staging_identity
        ):
            raise ArticulationV2StaticEvidencePublicationError(
                "evidence commit target did not retain the verified private staging "
                "identity"
            )

    def _reconcile_rename_outcome(self) -> bool:
        """Return whether a raised rename call already moved the private root.

        A signal or injected exception can arrive after the kernel completed the
        rename but before Python records it.  The retained directory descriptor
        is authoritative for that outcome; returning ``True`` lets commit
        finish the durability and lexical-parent checks instead of inviting a
        retry of an already-visible evidence root.
        """

        source = _stat_child_or_none(self._parent_descriptor, self._staging_name)
        target = _stat_child_or_none(self._parent_descriptor, self._target_name)
        source_matches = (
            source is not None
            and (
                source.st_dev,
                source.st_ino,
            )
            == self._staging_identity
        )
        target_matches = (
            target is not None
            and (
                target.st_dev,
                target.st_ino,
            )
            == self._staging_identity
        )
        if source is None and target_matches:
            return True
        if source_matches:
            return False
        raise ArticulationV2StaticEvidencePublicationError(
            "evidence publication rename outcome is ambiguous; do not retry blindly"
        )

    @staticmethod
    def _raise_rename_failure(exc: BaseException) -> None:
        if isinstance(exc, FileExistsError):
            raise ArticulationV2StaticEvidencePublicationError(
                "evidence_destination_exists: final evidence root already exists"
            ) from exc
        if isinstance(exc, OSError):
            raise ArticulationV2StaticEvidencePublicationError(
                f"evidence_publication_failed: atomic no-overwrite rename failed: {exc}"
            ) from exc
        raise exc

    def _verify_staging_for_commit(self) -> None:
        """Verify the named private tree still matches the held-file inventory."""

        named = os.stat(
            self._staging_name,
            dir_fd=self._parent_descriptor,
            follow_symlinks=False,
        )
        held = os.fstat(self._staging_descriptor)
        if (
            not stat.S_ISDIR(named.st_mode)
            or not stat.S_ISDIR(held.st_mode)
            or (named.st_dev, named.st_ino) != self._staging_identity
            or (held.st_dev, held.st_ino) != self._staging_identity
        ):
            raise ArticulationV2StaticEvidencePublicationError(
                "private staging directory ownership changed before publication"
            )
        self._verify_private_staging_contents()

    def _verify_private_staging_contents(self) -> None:
        """Verify the held tree against the recorded file state and digests."""

        files, directories, special_paths = _private_directory_inventory(
            self._staging_descriptor
        )
        expected_directories = _private_parent_directories(self._written)
        if (
            set(files) != set(self._written)
            or directories != expected_directories
            or special_paths
        ):
            raise ArticulationV2StaticEvidencePublicationError(
                "private staging inventory changed before publication"
            )
        for path, (size_bytes, sha256) in self._written.items():
            metadata, digest = _digest_private_file(self._staging_descriptor, path)
            if (
                _file_state(metadata) != self._file_states[path]
                or metadata.st_size != size_bytes
                or digest != sha256
            ):
                raise ArticulationV2StaticEvidencePublicationError(
                    "private staged evidence changed before publication"
                )

    def close(self) -> None:
        """Discard an uncommitted private root and close all held descriptors."""

        if self._closed:
            return
        self._closed = True
        cleanup_error: BaseException | None = None
        try:
            if not self._committed:
                self._discard_staging()
        except BaseException as exc:  # preserve an explicit cleanup failure
            cleanup_error = exc
        finally:
            try:
                os.close(self._staging_descriptor)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
            finally:
                self._staging_descriptor = -1
                context = self._parent_context
                self._parent_context = None
                if context is not None:
                    try:
                        context.__exit__(None, None, None)
                    except BaseException as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
        if cleanup_error is not None:
            if self._committed:
                raise ArticulationV2StaticEvidenceDurabilityUnknown(
                    "evidence_committed_cleanup_unknown: final evidence root was "
                    "renamed but a held descriptor could not be closed; do not "
                    "retry blindly"
                ) from cleanup_error
            raise cleanup_error

    def _require_open(self) -> None:
        if self._closed:
            raise ArticulationV2StaticEvidencePublicationError(
                "evidence transaction is already closed"
            )

    def _discard_staging(self) -> None:
        metadata = os.stat(
            self._staging_name,
            dir_fd=self._parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != self._staging_identity
        ):
            raise ArticulationV2StaticEvidencePublicationError(
                "private staging directory ownership changed; refusing cleanup"
            )
        if not shutil.rmtree.avoids_symlink_attacks:
            raise ArticulationV2StaticEvidencePublicationError(
                "descriptor-safe private staging cleanup is unavailable"
            )
        shutil.rmtree(self._staging_name, dir_fd=self._parent_descriptor)
        os.fsync(self._parent_descriptor)


@contextmanager
def stage_articulation_v2_static_evidence(
    destination: str | os.PathLike[str],
    allowlist: ArticulationV2StaticEvidencePathAllowlistV1,
) -> Iterator[ArticulationV2StaticEvidenceTransaction]:
    """Open one private root for a later atomic no-overwrite commit."""

    if type(allowlist) is not ArticulationV2StaticEvidencePathAllowlistV1:
        raise TypeError("allowlist must be an exact static evidence allowlist")
    try:
        allowlist = ArticulationV2StaticEvidencePathAllowlistV1.model_validate(
            allowlist.model_dump(mode="python"),
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArticulationV2StaticEvidencePublicationError(
            "evidence allowlist fails strict revalidation"
        ) from exc
    parent, target = _destination_parent_and_leaf(destination)
    try:
        parent_context = open_confined_directory(parent)
        parent_descriptor = parent_context.__enter__()
    except (ArtifactPathError, OSError, ValueError) as exc:
        raise ArticulationV2StaticEvidencePublicationError(
            "evidence destination parent is unsafe or unavailable"
        ) from exc
    transaction: ArticulationV2StaticEvidenceTransaction | None = None
    staging_descriptor: int | None = None
    staging_name: str | None = None
    staging_identity: tuple[int, int] | None = None
    try:
        _require_absent_destination(parent_descriptor, target)
        staging_name, staging_identity = _create_private_staging_directory(
            parent_descriptor,
            target,
        )
        os.fsync(parent_descriptor)
        staging_descriptor = os.open(
            staging_name,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(staging_descriptor)
        if (metadata.st_dev, metadata.st_ino) != staging_identity:
            raise ArticulationV2StaticEvidencePublicationError(
                "private staging directory ownership changed during setup"
            )
        transaction = ArticulationV2StaticEvidenceTransaction(
            allowlist=allowlist,
            parent_context=parent_context,
            parent_descriptor=parent_descriptor,
            parent_path=parent,
            target_name=target,
            staging_name=staging_name,
            staging_descriptor=staging_descriptor,
        )
        staging_descriptor = None  # transaction now owns the descriptor
        yield transaction
    except BaseException:
        if transaction is not None:
            try:
                transaction.close()
            except BaseException:
                pass
        else:
            if staging_descriptor is not None:
                os.close(staging_descriptor)
            if staging_name is not None and staging_identity is not None:
                try:
                    _discard_uninitialized_staging_directory(
                        parent_descriptor,
                        staging_name,
                        staging_identity,
                    )
                except BaseException:
                    pass
            parent_context.__exit__(None, None, None)
        raise
    else:
        assert transaction is not None
        transaction.close()


def _destination_parent_and_leaf(
    destination: str | os.PathLike[str],
) -> tuple[str, str]:
    raw = os.fspath(destination)
    if not raw or "\x00" in raw:
        raise ArticulationV2StaticEvidencePublicationError(
            "evidence destination must be a nonblank path"
        )
    parent, leaf = os.path.split(os.path.abspath(raw))
    if not leaf or leaf in {".", ".."} or "/" in leaf or "\\" in leaf:
        raise ArticulationV2StaticEvidencePublicationError(
            "evidence destination must name one directory beneath its parent"
        )
    return parent, leaf


def _require_absent_destination(parent_descriptor: int, target: str) -> None:
    try:
        os.stat(target, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise ArticulationV2StaticEvidencePublicationError(
        "evidence_destination_exists: final evidence root already exists"
    )


def _stat_child_or_none(parent_descriptor: int, name: str) -> os.stat_result | None:
    """Return one no-follow child stat, preserving all errors except absence."""

    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _create_private_staging_directory(
    parent_descriptor: int,
    target: str,
) -> tuple[str, tuple[int, int]]:
    for _attempt in range(8):
        name = f".{target}.staging-{uuid4().hex}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:  # cryptographic random collision: try a new name
            continue
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):  # defensive post-mkdir invariant
            raise ArticulationV2StaticEvidencePublicationError(
                "private staging allocation did not create a directory"
            )
        return name, (metadata.st_dev, metadata.st_ino)
    raise ArticulationV2StaticEvidencePublicationError(
        "could not allocate a private evidence staging directory"
    )


def _discard_uninitialized_staging_directory(
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
) -> None:
    """Remove a setup-aborted private staging root without following paths."""

    metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != expected_identity
    ):
        raise ArticulationV2StaticEvidencePublicationError(
            "private staging directory ownership changed; refusing cleanup"
        )
    os.rmdir(name, dir_fd=parent_descriptor)
    try:
        os.fsync(parent_descriptor)
    except OSError:
        pass


def _private_directory_inventory(
    root_descriptor: int,
) -> tuple[dict[str, os.stat_result], set[str], set[str]]:
    """List every private child through held descriptors without following links."""

    files: dict[str, os.stat_result] = {}
    directories: set[str] = set()
    special_paths: set[str] = set()

    def _visit(descriptor: int, prefix: str) -> None:
        for name in os.listdir(descriptor):
            path = f"{prefix}/{name}" if prefix else name
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(path)
                with _open_private_directory(
                    descriptor,
                    (name,),
                    create=False,
                ) as child:
                    _visit(child, path)
            elif stat.S_ISREG(metadata.st_mode):
                files[path] = metadata
            else:
                special_paths.add(path)

    _visit(root_descriptor, "")
    return files, directories, special_paths


def _private_parent_directories(paths: Iterable[str]) -> set[str]:
    """Return the exact non-root directory inventory implied by file paths."""

    directories: set[str] = set()
    for path in paths:
        components = path.split("/")
        for index in range(1, len(components)):
            directories.add("/".join(components[:index]))
    return directories


def _digest_private_file(
    root_descriptor: int,
    path: str,
) -> tuple[os.stat_result, str]:
    """Hash one existing regular private file through no-follow descriptors."""

    components = tuple(path.split("/"))
    try:
        with _open_private_directory(
            root_descriptor,
            components[:-1],
            create=False,
        ) as parent:
            descriptor = os.open(
                components[-1],
                _FILE_READ_FLAGS,
                dir_fd=parent,
            )
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise ArticulationV2StaticEvidencePublicationError(
                        "private staged evidence is not a regular file"
                    )
                digest = hashlib.sha256()
                while chunk := os.read(descriptor, 1024 * 1024):
                    digest.update(chunk)
                after = os.fstat(descriptor)
                if _file_state(before) != _file_state(after):
                    raise ArticulationV2StaticEvidencePublicationError(
                        "private staged evidence changed while it was verified"
                    )
                return after, digest.hexdigest()
            finally:
                os.close(descriptor)
    except OSError as exc:
        raise ArticulationV2StaticEvidencePublicationError(
            "could not verify private staged evidence"
        ) from exc


def _file_state(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


@contextmanager
def _open_private_directory(
    root_descriptor: int,
    components: tuple[str, ...],
    *,
    create: bool,
) -> Iterator[int]:
    descriptor = os.dup(root_descriptor)
    try:
        for component in components:
            try:
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ArticulationV2StaticEvidencePublicationError(
                        "evidence traversal encountered a symlink or non-directory"
                    ) from exc
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        yield descriptor
    finally:
        os.close(descriptor)


def _require_allowed_path(
    path: str,
    authority: _EvidencePathAuthority,
    *,
    terminal_allowed: bool,
) -> str:
    try:
        canonical = validated_artifact_relative_key(path)
    except ValueError as exc:
        raise ArticulationV2StaticEvidencePublicationError(
            "evidence path is not canonical and traversal-safe"
        ) from exc
    if canonical not in authority.paths or (
        not terminal_allowed and canonical == authority.terminal_receipt
    ):
        raise ArticulationV2StaticEvidencePublicationError(
            "evidence path is not admitted for this static run"
        )
    return canonical


def _rename_no_replace(parent_descriptor: int, source: str, destination: str) -> None:
    if _RENAMEAT2 is None:
        raise OSError(errno.ENOSYS, "renameat2(RENAME_NOREPLACE) is unavailable")
    result = _RENAMEAT2(
        parent_descriptor,
        os.fsencode(source),
        parent_descriptor,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), destination)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - regular-file invariant
            raise OSError("could not write evidence bytes")
        view = view[written:]


__all__ = [
    "ArticulationV2StaticEvidenceDurabilityUnknown",
    "ArticulationV2StaticEvidencePublicationError",
    "ArticulationV2StaticEvidencePublicationResultV1",
    "ArticulationV2StaticEvidenceTransaction",
    "ArticulationV2StaticPublishedFileV1",
    "stage_articulation_v2_static_evidence",
]
