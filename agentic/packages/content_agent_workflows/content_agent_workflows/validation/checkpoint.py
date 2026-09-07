# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic local checkpoint storage for resumable validation workflows."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from filelock import FileLock
from pydantic import ValidationError
from world_understanding.utils.credentials import ensure_no_inline_secrets

from content_agent_workflows.common.artifacts import _create_temporary_file_at

from .models import ValidationWorkflowCheckpoint


class ValidationCheckpointError(RuntimeError):
    """Raised when a validation checkpoint cannot be safely used."""


class _ValidationCheckpointParentMissing(FileNotFoundError):
    """Signal that a read-only checkpoint lookup has no parent directory."""


CheckpointMutation = Callable[
    [ValidationWorkflowCheckpoint], ValidationWorkflowCheckpoint
]


class ValidationCheckpointStore(Protocol):
    """Storage boundary for a single validation workflow checkpoint."""

    @property
    def path(self) -> Path:
        """Return the canonical checkpoint path."""

    def load(self) -> ValidationWorkflowCheckpoint | None:
        """Load the current checkpoint when present."""

    def create(
        self, checkpoint: ValidationWorkflowCheckpoint
    ) -> ValidationWorkflowCheckpoint:
        """Create a new checkpoint without replacing an existing run."""

    def update(self, mutation: CheckpointMutation) -> ValidationWorkflowCheckpoint:
        """Atomically mutate the latest checkpoint revision."""

    def finalize[FinalizationResult](
        self,
        finalization: Callable[
            [ValidationWorkflowCheckpoint],
            FinalizationResult,
        ],
    ) -> FinalizationResult:
        """Finalize artifacts while excluding concurrent checkpoint mutations."""


class FileValidationCheckpointStore:
    """File-backed checkpoint store serialized by a same-run lock."""

    def __init__(self, path: str | Path) -> None:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        self._path = candidate.parent.resolve() / candidate.name
        self._lock_path = self._path.with_suffix(f"{self._path.suffix}.lock")
        self._captured_parent_chain = self._capture_existing_parent_chain()
        self._parent_identity = (
            self._captured_parent_chain[-1][1]
            if len(self._captured_parent_chain) == len(self._path.parent.parts)
            else None
        )
        self._active_parent_fd: int | None = None
        self._loaded_checkpoint_identity: tuple[int, int] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def _capture_existing_parent_chain(
        self,
    ) -> tuple[tuple[str, tuple[int, int]], ...]:
        if os.name != "posix":  # pragma: win32 cover
            return ()
        captured: list[tuple[str, tuple[int, int]]] = []
        current = Path(self._path.parent.anchor)
        for index, component in enumerate(self._path.parent.parts):
            if index:
                current /= component
            try:
                component_stat = current.lstat()
            except FileNotFoundError:
                break
            except OSError as exc:
                raise ValidationCheckpointError(
                    "Validation checkpoint parent directory could not be inspected "
                    f"safely: {current}"
                ) from exc
            if not stat.S_ISDIR(component_stat.st_mode):
                raise ValidationCheckpointError(
                    "Validation checkpoint parent path components must be "
                    f"directories, not symlinks or files: {current}"
                )
            captured.append(
                (
                    component,
                    (component_stat.st_dev, component_stat.st_ino),
                )
            )
        return tuple(captured)

    def _open_parent_dir(self, *, create_missing: bool = True) -> int | None:
        if os.name != "posix":  # pragma: win32 cover
            return None
        flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        fd = -1
        opened_chain: list[tuple[str, tuple[int, int]]] = []
        try:
            fd = os.open(self._path.parent.anchor, flags)
            root_stat = os.fstat(fd)
            opened_chain.append(
                (
                    self._path.parent.anchor,
                    (root_stat.st_dev, root_stat.st_ino),
                )
            )
            if self._captured_parent_chain and (
                opened_chain[0] != self._captured_parent_chain[0]
            ):
                raise ValidationCheckpointError(
                    "Validation checkpoint parent directory identity changed: "
                    f"{self._path.parent}"
                )
            for index, component in enumerate(self._path.parent.parts[1:], start=1):
                try:
                    next_fd = os.open(component, flags, dir_fd=fd)
                except FileNotFoundError:
                    if not create_missing:
                        raise _ValidationCheckpointParentMissing(
                            self._path.parent
                        ) from None
                    try:
                        os.mkdir(component, dir_fd=fd)
                    except FileExistsError:
                        pass
                    next_fd = os.open(component, flags, dir_fd=fd)
                component_stat = os.fstat(next_fd)
                component_identity = (
                    component_stat.st_dev,
                    component_stat.st_ino,
                )
                opened_chain.append((component, component_identity))
                if index < len(self._captured_parent_chain):
                    expected_name, expected_identity = self._captured_parent_chain[
                        index
                    ]
                    if component != expected_name or (
                        component_identity != expected_identity
                    ):
                        os.close(next_fd)
                        raise ValidationCheckpointError(
                            "Validation checkpoint parent directory identity "
                            f"changed: {self._path.parent}"
                        )
                os.close(fd)
                fd = next_fd
        except (ValidationCheckpointError, _ValidationCheckpointParentMissing):
            if fd >= 0:
                os.close(fd)
            raise
        except OSError as exc:
            if fd >= 0:
                os.close(fd)
            raise ValidationCheckpointError(
                "Validation checkpoint parent directory could not be opened "
                f"safely: {self._path.parent}"
            ) from exc
        parent_stat = os.fstat(fd)
        identity = (parent_stat.st_dev, parent_stat.st_ino)
        if len(self._captured_parent_chain) != len(opened_chain):
            self._captured_parent_chain = tuple(opened_chain)
        if self._parent_identity is None:
            self._parent_identity = identity
        elif identity != self._parent_identity:
            os.close(fd)
            raise ValidationCheckpointError(
                "Validation checkpoint parent directory identity changed: "
                f"{self._path.parent}"
            )
        return fd

    def _require_active_parent_fd(self) -> int:
        if self._active_parent_fd is None:
            raise ValidationCheckpointError(
                "Validation checkpoint file operation requires the checkpoint lock."
            )
        return self._active_parent_fd

    def _parent_path_matches(self, fd: int) -> bool:
        try:
            fd_stat = os.fstat(fd)
        except OSError:
            return False
        return (
            self._parent_chain_matches()
            and (
                fd_stat.st_dev,
                fd_stat.st_ino,
            )
            == self._captured_parent_chain[-1][1]
        )

    def _parent_chain_matches(self) -> bool:
        if len(self._path.parent.parts) != len(self._captured_parent_chain):
            return False
        current = Path(self._path.parent.anchor)
        for index, (expected_name, expected_identity) in enumerate(
            self._captured_parent_chain
        ):
            component = self._path.parent.parts[index]
            if component != expected_name:
                return False
            if index:
                current /= component
            try:
                component_stat = current.lstat()
            except OSError:
                return False
            if (
                not stat.S_ISDIR(component_stat.st_mode)
                or (
                    component_stat.st_dev,
                    component_stat.st_ino,
                )
                != expected_identity
            ):
                return False
        return True

    def _checkpoint_stat_unlocked(self) -> os.stat_result | None:
        if os.name == "posix":
            try:
                return os.stat(
                    self._path.name,
                    dir_fd=self._require_active_parent_fd(),
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return None
        try:  # pragma: win32 cover
            return self._path.lstat()
        except FileNotFoundError:  # pragma: win32 cover
            return None

    def _load_unlocked(self) -> ValidationWorkflowCheckpoint | None:
        try:
            if os.name == "posix":
                fd = os.open(
                    self._path.name,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=self._require_active_parent_fd(),
                )
            else:  # pragma: win32 cover
                fd = os.open(
                    self._path,
                    os.O_RDONLY | getattr(os, "O_NONBLOCK", 0),
                )
        except FileNotFoundError:
            self._loaded_checkpoint_identity = None
            return None
        except OSError as exc:
            raise ValidationCheckpointError(
                f"Invalid validation checkpoint at {self._path}: {exc}"
            ) from exc
        try:
            try:
                fd_stat = os.fstat(fd)
                path_stat = self._checkpoint_stat_unlocked()
                if (
                    path_stat is None
                    or not stat.S_ISREG(fd_stat.st_mode)
                    or fd_stat.st_nlink != 1
                    or path_stat.st_nlink != 1
                    or (path_stat.st_dev, path_stat.st_ino)
                    != (fd_stat.st_dev, fd_stat.st_ino)
                ):
                    raise ValidationCheckpointError(
                        "Validation checkpoint path must be a stable, single-link "
                        f"regular file: {self._path}"
                    )
                self._loaded_checkpoint_identity = (
                    fd_stat.st_dev,
                    fd_stat.st_ino,
                )
                with os.fdopen(fd, encoding="utf-8") as stream:
                    fd = -1
                    payload = json.load(stream)
            finally:
                if fd >= 0:
                    os.close(fd)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected a JSON object in validation checkpoint {self._path}"
                )
            ensure_no_inline_secrets(
                payload,
                context="validation workflow checkpoint",
            )
            return ValidationWorkflowCheckpoint.model_validate(payload)
        except (OSError, ValueError, ValidationError) as exc:
            raise ValidationCheckpointError(
                f"Invalid validation checkpoint at {self._path}: {exc}"
            ) from exc

    def _atomic_write_unlocked(
        self,
        checkpoint: ValidationWorkflowCheckpoint,
        *,
        expected_identity: tuple[int, int] | None,
    ) -> None:
        document = checkpoint.model_dump(mode="json")
        text = (
            json.dumps(
                document,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n"
        )
        parent_fd = self._require_active_parent_fd() if os.name == "posix" else None
        temporary_name: str | None = None
        temporary_path = ""
        temporary_fd = -1
        try:
            if parent_fd is not None:
                temporary_fd, temporary_name = _create_temporary_file_at(
                    parent_fd,
                    destination_name=self._path.name,
                )
                temporary_path = temporary_name
            else:  # pragma: win32 cover
                temporary_fd, temporary_path = tempfile.mkstemp(
                    dir=self._path.parent,
                    prefix=f".{self._path.name}.",
                    suffix=".tmp",
                )
                temporary_name = Path(temporary_path).name
            with os.fdopen(
                os.dup(temporary_fd),
                mode="w",
                encoding="utf-8",
            ) as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            temporary_stat = os.fstat(temporary_fd)
            temporary_identity = (
                temporary_stat.st_dev,
                temporary_stat.st_ino,
            )
            path_stat = self._checkpoint_stat_unlocked()
            if expected_identity is None:
                if path_stat is not None:
                    raise ValidationCheckpointError(
                        f"Validation checkpoint was created concurrently: {self._path}"
                    )
            elif (
                path_stat is None
                or not stat.S_ISREG(path_stat.st_mode)
                or path_stat.st_nlink != 1
                or (path_stat.st_dev, path_stat.st_ino) != expected_identity
            ):
                raise ValidationCheckpointError(
                    "Validation checkpoint identity changed before update: "
                    f"{self._path}"
                )
            if parent_fd is not None:
                staged_stat = os.stat(
                    temporary_name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(staged_stat.st_mode)
                    or staged_stat.st_nlink != 1
                    or (staged_stat.st_dev, staged_stat.st_ino) != temporary_identity
                ):
                    raise ValidationCheckpointError(
                        "Validation checkpoint temporary file identity changed."
                    )
                os.replace(
                    temporary_name,
                    self._path.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                temporary_name = None
                installed_stat = os.stat(
                    self._path.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(installed_stat.st_mode)
                    or installed_stat.st_nlink != 1
                    or (installed_stat.st_dev, installed_stat.st_ino)
                    != temporary_identity
                ):
                    raise ValidationCheckpointError(
                        "Validation checkpoint destination identity changed "
                        "during update."
                    )
                if not self._parent_path_matches(parent_fd):
                    raise ValidationCheckpointError(
                        "Validation checkpoint parent directory identity changed: "
                        f"{self._path.parent}"
                    )
                os.fsync(parent_fd)
                if not self._parent_path_matches(parent_fd):
                    raise ValidationCheckpointError(
                        "Validation checkpoint parent directory identity changed: "
                        f"{self._path.parent}"
                    )
            else:  # pragma: win32 cover
                os.replace(temporary_path, self._path)
                temporary_name = None
        finally:
            if temporary_name is not None:
                with suppress(OSError):
                    if parent_fd is not None:
                        os.unlink(temporary_name, dir_fd=parent_fd)
                    else:  # pragma: win32 cover
                        Path(temporary_path).unlink(missing_ok=True)
            if temporary_fd >= 0:
                os.close(temporary_fd)

    def _validate_lock_path(self, *, parent_fd: int | None = None) -> None:
        try:
            if parent_fd is not None:
                try:
                    lock_stat = os.stat(
                        self._lock_path.name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    return
            else:  # pragma: win32 cover
                if self._lock_path.is_symlink():
                    raise ValidationCheckpointError(
                        "Validation checkpoint lock path cannot use a symlink: "
                        f"{self._lock_path}"
                    )
                if not self._lock_path.exists():
                    return
                lock_stat = self._lock_path.lstat()
            if stat.S_ISLNK(lock_stat.st_mode):
                raise ValidationCheckpointError(
                    "Validation checkpoint lock path cannot use a symlink: "
                    f"{self._lock_path}"
                )
            if not stat.S_ISREG(lock_stat.st_mode):
                raise ValidationCheckpointError(
                    "Validation checkpoint lock path must be a regular file: "
                    f"{self._lock_path}"
                )
            if lock_stat.st_nlink > 1:
                raise ValidationCheckpointError(
                    "Validation checkpoint lock path cannot be a hard link: "
                    f"{self._lock_path}"
                )
        except OSError as exc:
            raise ValidationCheckpointError(
                "Validation checkpoint lock path could not be inspected safely: "
                f"{self._lock_path}"
            ) from exc

    def _lock_path_matches(self, lock_fd: int, parent_fd: int) -> bool:
        try:
            fd_stat = os.fstat(lock_fd)
            path_stat = os.stat(
                self._lock_path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except OSError:
            return False
        return (
            stat.S_ISREG(fd_stat.st_mode)
            and fd_stat.st_nlink == 1
            and stat.S_ISREG(path_stat.st_mode)
            and path_stat.st_nlink == 1
            and (path_stat.st_dev, path_stat.st_ino) == (fd_stat.st_dev, fd_stat.st_ino)
        )

    def _acquire_lock_file(self, parent_fd: int) -> int:
        import fcntl  # noqa: PLC0415

        try:
            lock_fd = os.open(
                self._lock_path.name,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise ValidationCheckpointError(
                "Validation checkpoint lock path could not be opened safely: "
                f"{self._lock_path}"
            ) from exc
        try:
            if not self._lock_path_matches(lock_fd, parent_fd):
                raise ValidationCheckpointError(
                    "Validation checkpoint lock path identity changed during "
                    f"acquisition: {self._lock_path}"
                )
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValidationCheckpointError(
                    "Validation checkpoint lock file is held outside the "
                    f"checkpoint parent lock: {self._lock_path}"
                ) from exc
            if not self._lock_path_matches(lock_fd, parent_fd):
                raise ValidationCheckpointError(
                    "Validation checkpoint lock path identity changed during "
                    f"acquisition: {self._lock_path}"
                )
            return lock_fd
        except BaseException:
            with suppress(OSError):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            raise

    def _release_lock_file(self, lock_fd: int, parent_fd: int) -> None:
        import fcntl  # noqa: PLC0415

        with suppress(OSError):
            if self._lock_path_matches(lock_fd, parent_fd):
                os.unlink(self._lock_path.name, dir_fd=parent_fd)
        with suppress(OSError):
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        with suppress(OSError):
            os.close(lock_fd)

    @contextmanager
    def _locked(self, *, create_parent: bool = True) -> Iterator[None]:
        parent_fd = self._open_parent_dir(create_missing=create_parent)
        if parent_fd is None:  # pragma: win32 cover
            self._validate_lock_path()
            with FileLock(str(self._lock_path)):
                yield
            return
        import fcntl  # noqa: PLC0415

        lock_fd = -1
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_EX)
            self._validate_lock_path(parent_fd=parent_fd)
            lock_fd = self._acquire_lock_file(parent_fd)
            if not self._parent_path_matches(parent_fd):
                raise ValidationCheckpointError(
                    "Validation checkpoint parent directory identity changed: "
                    f"{self._path.parent}"
                )
            self._active_parent_fd = parent_fd
            completed = False
            try:
                yield
                completed = True
            finally:
                self._active_parent_fd = None
            if completed:
                if not self._parent_path_matches(parent_fd):
                    raise ValidationCheckpointError(
                        "Validation checkpoint parent directory identity changed: "
                        f"{self._path.parent}"
                    )
                if not self._lock_path_matches(lock_fd, parent_fd):
                    raise ValidationCheckpointError(
                        "Validation checkpoint lock path identity changed: "
                        f"{self._lock_path}"
                    )
        finally:
            if lock_fd >= 0:
                self._release_lock_file(lock_fd, parent_fd)
            with suppress(OSError):
                fcntl.flock(parent_fd, fcntl.LOCK_UN)
            os.close(parent_fd)

    def load(self) -> ValidationWorkflowCheckpoint | None:
        if os.name != "posix" and not self._path.parent.is_dir():
            return None
        try:
            with self._locked(create_parent=False):
                return self._load_unlocked()
        except _ValidationCheckpointParentMissing:
            return None

    def create(
        self, checkpoint: ValidationWorkflowCheckpoint
    ) -> ValidationWorkflowCheckpoint:
        with self._locked():
            if self._checkpoint_stat_unlocked() is not None:
                raise ValidationCheckpointError(
                    "Validation checkpoint already exists; resume the run or "
                    f"use a new output directory: {self._path}"
                )
            created = checkpoint.model_copy(
                update={
                    "revision": 1,
                    "updated_at": datetime.now(UTC),
                }
            )
            ensure_no_inline_secrets(
                created.model_dump(mode="json"),
                context="validation workflow checkpoint",
            )
            self._atomic_write_unlocked(created, expected_identity=None)
            return created

    def update(self, mutation: CheckpointMutation) -> ValidationWorkflowCheckpoint:
        with self._locked():
            current = self._load_unlocked()
            if current is None:
                raise ValidationCheckpointError(
                    f"Validation checkpoint does not exist: {self._path}"
                )
            current_identity = self._loaded_checkpoint_identity
            if current_identity is None:
                raise ValidationCheckpointError(
                    "Validation checkpoint identity was not captured while loading: "
                    f"{self._path}"
                )
            mutated = mutation(current)
            if (
                mutated.workflow_identity != current.workflow_identity
                or mutated.plan_digest != current.plan_digest
                or mutated.ordered_work_item_ids != current.ordered_work_item_ids
            ):
                raise ValidationCheckpointError(
                    "Checkpoint mutation attempted to change immutable run identity"
                )
            if mutated is current:
                return current
            updated = mutated.model_copy(
                update={
                    "revision": current.revision + 1,
                    "updated_at": datetime.now(UTC),
                }
            )
            ensure_no_inline_secrets(
                updated.model_dump(mode="json"),
                context="validation workflow checkpoint",
            )
            self._atomic_write_unlocked(
                updated,
                expected_identity=current_identity,
            )
            return updated

    def finalize[FinalizationResult](
        self,
        finalization: Callable[
            [ValidationWorkflowCheckpoint],
            FinalizationResult,
        ],
    ) -> FinalizationResult:
        with self._locked():
            current = self._load_unlocked()
            if current is None:
                raise ValidationCheckpointError(
                    f"Validation checkpoint does not exist: {self._path}"
                )
            current_identity = self._loaded_checkpoint_identity
            if current_identity is None:
                raise ValidationCheckpointError(
                    "Validation checkpoint identity was not captured before "
                    f"finalization: {self._path}"
                )
            result = finalization(current)
            verified = self._load_unlocked()
            if (
                verified != current
                or self._loaded_checkpoint_identity != current_identity
            ):
                raise ValidationCheckpointError(
                    "Validation checkpoint changed during final artifact "
                    f"publication: {self._path}"
                )
            return result
