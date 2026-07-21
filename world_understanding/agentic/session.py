# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session management for agent workflows.

This module provides session-based state management for agent workflows,
enabling reproducible runs, debugging, and state persistence.

Key concepts:
- Session ID: Unique identifier for a workflow run
- Session directory: Isolated workspace for all session artifacts
- Path resolution: Automatic path derivation within session structure

Example:
    ```python
    from world_understanding.agentic.session import SessionManager

    # Create new session
    session = SessionManager.create(
        base_dir=Path("outputs"),
        project_name="my_project"
    )

    # Or reuse existing session
    session = SessionManager.from_id(
        session_id="abc-123-def",
        base_dir=Path("outputs")
    )

    # Get session paths
    print(session.session_dir)  # outputs/.abc-123-def
    print(session.get_subdir("dataset"))  # outputs/.abc-123-def/dataset
    print(session.get_subdir("iterations/iteration_1"))  # ...
    ```
"""

import json
import logging
import math
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, NoReturn

from world_understanding.utils.credentials import (
    create_directory_with_safe_diagnostics,
    ensure_no_inline_secrets,
    path_exists_with_safe_diagnostics,
    redact_sensitive_config,
    redact_sensitive_path,
    resolve_path_with_safe_diagnostics,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SessionListFailure:
    error_type: type[Exception]
    args: tuple[Any, ...]
    os_error: tuple[int | None, str | None, str | None] | None = None


def _capture_session_list_failure(error: Exception) -> _SessionListFailure:
    os_error = (
        (error.errno, error.strerror, error.filename)
        if isinstance(error, OSError)
        else None
    )
    return _SessionListFailure(type(error), error.args, os_error)


def _raise_session_list_failure(failure: _SessionListFailure) -> NoReturn:
    if failure.os_error is not None:
        error_number, message, filename = failure.os_error
        if filename is not None:
            raise failure.error_type(error_number, message, filename)
        if error_number is not None:
            raise failure.error_type(error_number, message)
    raise failure.error_type(*failure.args)


def _validate_filename_component(
    value: str,
    *,
    label: str,
    allow_empty: bool = False,
    allow_dot: bool = False,
) -> None:
    """Require a value that cannot select a parent or nested path."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a single filename component")
    is_special = value == ".." or (value == "." and not allow_dot)
    if (
        (not value and not allow_empty)
        or is_special
        or "\x00" in value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{label} must be a single filename component")


def _validate_session_id(session_id: str) -> None:
    _validate_filename_component(session_id, label="Session ID")


def _validate_session_prefix(prefix: str) -> None:
    # Empty prefixes are supported, and the conventional default ``.`` is a
    # safe prefix once concatenated with a validated non-empty session ID.
    _validate_filename_component(
        prefix,
        label="Session prefix",
        allow_empty=True,
        allow_dot=True,
    )


_UNSUPPORTED_METADATA_MESSAGE = "Unsupported session metadata"
_INVALID_SESSION_DIRECTORY_MESSAGE = "Invalid session directory"
_INVALID_SESSION_PATH_MESSAGE = "Invalid session path"


def _raise_safe_path_inspection_error(
    error: OSError | RuntimeError,
    *,
    path: Path,
    label: str,
) -> NoReturn:
    """Raise one path-inspection error without exposing its runtime value."""
    safe_path = redact_sensitive_path(path)
    if isinstance(error, OSError):
        raise type(error)(
            error.errno,
            f"Unable to inspect {label}",
            safe_path,
        ) from None
    raise RuntimeError(f"Unable to inspect {label}: {safe_path}") from None


def _path_is_symlink_with_safe_diagnostics(path: Path, *, label: str) -> bool:
    """Inspect one path without following a link or leaking it on failure."""
    try:
        return path.is_symlink()
    except (OSError, RuntimeError) as error:
        _raise_safe_path_inspection_error(error, path=path, label=label)


def _session_directory_component(*, session_id: str, prefix: str) -> str:
    """Validate and return one portable session-directory component."""
    if not isinstance(session_id, str) or not isinstance(prefix, str):
        raise ValueError(_INVALID_SESSION_DIRECTORY_MESSAGE) from None
    ensure_no_inline_secrets(
        session_id,
        context="session identifier",
        path_context=True,
    )
    ensure_no_inline_secrets(
        prefix,
        context="session directory prefix",
        path_context=True,
    )

    component = f"{prefix}{session_id}"
    posix_component = PurePosixPath(component)
    windows_component = PureWindowsPath(component)
    invalid = (
        session_id in {"", ".", ".."}
        or prefix == ".."
        or not component
        or "\x00" in component
        or "/" in component
        or "\\" in component
        or posix_component.anchor != ""
        or windows_component.anchor != ""
        or len(posix_component.parts) != 1
        or len(windows_component.parts) != 1
        or component in {".", ".."}
    )
    if invalid:
        raise ValueError(_INVALID_SESSION_DIRECTORY_MESSAGE) from None
    return component


def _session_directory_for_id(
    base_dir: Path,
    *,
    session_id: str,
    prefix: str,
) -> Path:
    """Return one non-symlinked immediate child of the configured base."""
    component = _session_directory_component(session_id=session_id, prefix=prefix)
    base_path = Path(base_dir)
    session_path = base_path / component
    if _path_is_symlink_with_safe_diagnostics(
        session_path,
        label="session directory",
    ):
        raise ValueError(_INVALID_SESSION_DIRECTORY_MESSAGE) from None

    resolved_base = resolve_path_with_safe_diagnostics(
        base_path,
        label="session base directory",
    )
    resolved_session = resolve_path_with_safe_diagnostics(
        session_path,
        label="session directory",
    )
    if resolved_session.parent != resolved_base:
        raise ValueError(_INVALID_SESSION_DIRECTORY_MESSAGE) from None
    return session_path


def _confined_session_child(session_dir: Path, value: str, *, label: str) -> Path:
    """Resolve one relative path whose final target remains inside a session."""
    if not isinstance(value, str | os.PathLike):
        raise ValueError(_INVALID_SESSION_PATH_MESSAGE) from None
    ensure_no_inline_secrets(value, context=label, path_context=True)
    path_text = os.fspath(value)
    if type(path_text) is not str:
        raise ValueError(_INVALID_SESSION_PATH_MESSAGE) from None

    posix_path = PurePosixPath(path_text)
    windows_path = PureWindowsPath(path_text)
    invalid = (
        not path_text
        or "\x00" in path_text
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.anchor != ""
        or windows_path.drive != ""
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    )
    if invalid:
        raise ValueError(_INVALID_SESSION_PATH_MESSAGE) from None

    resolved_root = resolve_path_with_safe_diagnostics(
        session_dir,
        label="session directory",
    )
    resolved_child = resolve_path_with_safe_diagnostics(
        session_dir / Path(path_text),
        label=label,
    )
    if resolved_child == resolved_root or not resolved_child.is_relative_to(
        resolved_root
    ):
        raise ValueError(_INVALID_SESSION_PATH_MESSAGE) from None
    return resolved_child


def _metadata_json_sort_key(value: Any) -> tuple[int, str]:
    """Return a deterministic ordering key for a projected set member."""
    type_rank = {
        type(None): 0,
        bool: 1,
        int: 2,
        float: 3,
        str: 4,
        list: 5,
        dict: 6,
    }
    return (
        type_rank[type(value)],
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _project_metadata_json_value(
    value: Any,
    *,
    active_container_ids: set[int] | None = None,
) -> Any:
    """Recursively project benign metadata values to JSON primitives."""
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(_UNSUPPORTED_METADATA_MESSAGE) from None
        return value
    if isinstance(value, os.PathLike):
        path_value = os.fspath(value)
        if type(path_value) is not str:
            raise TypeError(_UNSUPPORTED_METADATA_MESSAGE) from None
        return path_value

    container_type = type(value)
    if container_type not in {dict, list, tuple, set, frozenset}:
        raise TypeError(_UNSUPPORTED_METADATA_MESSAGE) from None

    active_ids = active_container_ids if active_container_ids is not None else set()
    container_id = id(value)
    if container_id in active_ids:
        raise ValueError(_UNSUPPORTED_METADATA_MESSAGE) from None
    active_ids.add(container_id)
    try:
        if container_type is dict:
            projected_mapping: dict[str, Any] = {}
            for key, child in value.items():
                if type(key) is not str:
                    raise TypeError(_UNSUPPORTED_METADATA_MESSAGE) from None
                projected_mapping[key] = _project_metadata_json_value(
                    child,
                    active_container_ids=active_ids,
                )
            return projected_mapping

        projected_items = [
            _project_metadata_json_value(
                child,
                active_container_ids=active_ids,
            )
            for child in value
        ]
        if container_type in {set, frozenset}:
            projected_items.sort(key=_metadata_json_sort_key)
        return projected_items
    finally:
        active_ids.remove(container_id)


def _project_metadata_for_durable_storage(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return a diagnostic metadata copy that is safe to persist or reload."""
    projected = redact_sensitive_config(metadata)
    if not isinstance(projected, dict):
        # A credential-bearing mapping key makes the whole mapping atomic; no
        # safe key/value projection exists in that case.
        return {}
    ensure_no_inline_secrets(projected, context="session metadata")
    durable_metadata = _project_metadata_json_value(projected)
    if type(durable_metadata) is not dict:  # pragma: no cover - input is a dict
        raise TypeError(_UNSUPPORTED_METADATA_MESSAGE) from None
    return durable_metadata


def _write_metadata_atomic(path: Path, payload: str) -> None:
    """Publish one fully serialized metadata document atomically."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class SessionManager:
    """Manages session state and directory structure for agent workflows.

    A session represents a single execution of an agent workflow with its own
    isolated workspace. Sessions can be resumed by providing the same session_id.

    Session directory structure:
        .{session_id}/
            dataset/        # Input dataset files
            iterations/     # Iteration outputs (for iterative workflows)
                iteration_1/
                iteration_2/
            output/         # Final outputs
            logs/           # Session logs
            .metadata.json  # Session metadata

    Attributes:
        session_id: Unique identifier for this session
        session_dir: Root directory for all session files
        project_name: Optional project name for metadata
        metadata: Session metadata (creation time, project info, etc.)
    """

    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        project_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize a session manager.

        Note: Use SessionManager.create() or SessionManager.from_id() instead
        of calling this constructor directly.

        Args:
            session_id: Unique session identifier
            session_dir: Root directory for session files
            project_name: Optional project name
            metadata: Optional session metadata
        """
        _validate_session_id(session_id)
        ensure_no_inline_secrets(
            session_id,
            context="session identifier",
            path_context=True,
        )
        self.session_id = session_id
        self.session_dir = resolve_path_with_safe_diagnostics(
            session_dir,
            label="session directory",
        )
        self.project_name = project_name or "unknown_project"
        self.metadata = metadata or {}

        # Ensure metadata has required fields
        if "session_id" not in self.metadata:
            self.metadata["session_id"] = session_id
        if "project_name" not in self.metadata:
            self.metadata["project_name"] = self.project_name

    @classmethod
    def create(
        cls,
        base_dir: Path,
        project_name: str | None = None,
        session_id: str | None = None,
        prefix: str = ".",
        metadata: dict[str, Any] | None = None,
    ) -> "SessionManager":
        """Create a new session or reuse existing one.

        Args:
            base_dir: Base directory where session directories are created
            project_name: Optional project name for metadata
            session_id: Optional session ID to reuse; if None, generates new UUID
            prefix: Prefix for session directory (default: "." for hidden dirs)
            metadata: Optional additional metadata to store

        Returns:
            SessionManager instance

        Example:
            ```python
            # Create new session
            session = SessionManager.create(
                base_dir=Path("outputs"),
                project_name="material_assignment"
            )

            # Reuse existing session
            session = SessionManager.create(
                base_dir=Path("outputs"),
                session_id="abc-123-def"
            )
            ```
        """
        # Generate or validate session_id
        _validate_session_prefix(prefix)
        generated_session_id = session_id is None
        if session_id is None:
            session_id = str(uuid.uuid4())
        _validate_session_id(session_id)
        session_dir = _session_directory_for_id(
            base_dir,
            session_id=session_id,
            prefix=prefix,
        )
        if generated_session_id:
            logger.info(
                "Generated new session ID: %s",
                redact_sensitive_config(session_id),
            )
        else:
            _validate_session_id(session_id)
            logger.info(
                "Using provided session ID: %s",
                redact_sensitive_path(session_id),
            )

        # Create session directory if it doesn't exist
        create_directory_with_safe_diagnostics(
            session_dir,
            label="session directory",
        )

        # Initialize metadata
        import datetime

        session_metadata = metadata or {}
        session_metadata.update(
            {
                "session_id": session_id,
                "project_name": project_name or "unknown_project",
                "created_at": datetime.datetime.now().isoformat(),
                "base_dir": str(base_dir),
                "session_dir": str(session_dir),
            }
        )

        return cls(
            session_id=session_id,
            session_dir=session_dir,
            project_name=project_name,
            metadata=session_metadata,
        )

    @classmethod
    def from_id(
        cls,
        session_id: str,
        base_dir: Path,
        prefix: str = ".",
        project_name: str | None = None,
    ) -> "SessionManager":
        """Load an existing session by ID.

        Args:
            session_id: Session ID to load
            base_dir: Base directory where session exists
            prefix: Prefix for session directory (default: ".")
            project_name: Optional project name override

        Returns:
            SessionManager instance

        Raises:
            FileNotFoundError: If session directory doesn't exist

        Example:
            ```python
            # Load existing session
            session = SessionManager.from_id(
                session_id="abc-123-def",
                base_dir=Path("outputs")
            )
            ```
        """
        _validate_session_id(session_id)
        _validate_session_prefix(prefix)
        ensure_no_inline_secrets(
            session_id,
            context="session identifier",
            path_context=True,
        )
        session_dir = _session_directory_for_id(
            base_dir,
            session_id=session_id,
            prefix=prefix,
        )
        safe_session_dir = redact_sensitive_path(session_dir)

        session_exists = path_exists_with_safe_diagnostics(
            session_dir,
            label="session directory",
        )
        if not session_exists:
            raise FileNotFoundError(
                f"Session directory not found: {safe_session_dir}. "
                "The requested session ID does not exist in "
                f"{redact_sensitive_path(base_dir)}."
            )

        # Try to load metadata if it exists
        metadata_file = session_dir / ".metadata.json"
        metadata = {}
        metadata_is_symlink = _path_is_symlink_with_safe_diagnostics(
            metadata_file,
            label="session metadata",
        )
        metadata_exists = not metadata_is_symlink and path_exists_with_safe_diagnostics(
            metadata_file,
            label="session metadata",
        )
        if metadata_exists:
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    loaded_metadata = json.load(f)
                if isinstance(loaded_metadata, dict):
                    metadata = _project_metadata_for_durable_storage(loaded_metadata)
                logger.debug(
                    "Loaded session metadata from %s",
                    redact_sensitive_path(metadata_file),
                )
            except Exception:
                logger.warning("Failed to load session metadata")

        # Use project_name from metadata or parameter
        metadata_project_name = metadata.get("project_name")
        if not project_name and isinstance(metadata_project_name, str):
            project_name = metadata_project_name

        logger.info(
            "Loaded existing session: %s",
            redact_sensitive_path(session_id),
        )

        return cls(
            session_id=session_id,
            session_dir=session_dir,
            project_name=project_name,
            metadata=metadata,
        )

    def get_subdir(self, subdir: str, create: bool = True) -> Path:
        """Get a subdirectory within the session.

        Args:
            subdir: Subdirectory path relative to session root
            create: Whether to create the directory if it doesn't exist

        Returns:
            Absolute path to the subdirectory

        Example:
            ```python
            dataset_dir = session.get_subdir("dataset")
            iter1_dir = session.get_subdir("iterations/iteration_1")
            ```
        """
        path = _confined_session_child(
            self.session_dir,
            subdir,
            label="session subdirectory",
        )
        if create:
            create_directory_with_safe_diagnostics(
                path,
                label="session subdirectory",
            )
        return path

    def get_file(self, filepath: str) -> Path:
        """Get a file path within the session.

        Args:
            filepath: File path relative to session root

        Returns:
            Absolute path to the file

        Example:
            ```python
            config_file = session.get_file("config.yaml")
            output_file = session.get_file("output/result.json")
            ```
        """
        return _confined_session_child(
            self.session_dir,
            filepath,
            label="session file",
        )

    def save_metadata(self) -> None:
        """Save session metadata to disk.

        Writes metadata to .metadata.json in the session directory.
        """
        metadata_file = self.session_dir / ".metadata.json"

        try:
            durable_metadata = _project_metadata_for_durable_storage(self.metadata)
            payload = json.dumps(durable_metadata, indent=2, allow_nan=False)
            _write_metadata_atomic(metadata_file, payload)
            logger.debug(
                "Saved session metadata to %s",
                redact_sensitive_path(metadata_file),
            )
        except Exception:
            logger.warning(
                "Failed to save session metadata to %s",
                redact_sensitive_path(metadata_file),
            )

    def update_metadata(self, **kwargs: Any) -> None:
        """Update session metadata with new key-value pairs.

        Args:
            **kwargs: Key-value pairs to add to metadata

        Example:
            ```python
            session.update_metadata(
                status="completed",
                num_predictions=42,
                final_score=0.95
            )
            ```
        """
        self.metadata.update(kwargs)
        self.save_metadata()

    @staticmethod
    def list_sessions(base_dir: Path, prefix: str = ".") -> list[dict[str, Any]]:
        """List sessions without retaining request paths in public tracebacks."""
        failure: _SessionListFailure | None = None
        try:
            return SessionManager._list_sessions_impl(base_dir, prefix)
        except (OSError, ValueError) as error:
            failure = _capture_session_list_failure(error)

        del base_dir, prefix
        assert failure is not None
        _raise_session_list_failure(failure)

    @staticmethod
    def _list_sessions_impl(base_dir: Path, prefix: str = ".") -> list[dict[str, Any]]:
        """List all sessions in a base directory.

        Args:
            base_dir: Directory to search for sessions
            prefix: Session directory prefix (default: ".")

        Returns:
            List of session info dicts with session_id, path, and metadata

        Example:
            ```python
            sessions = SessionManager.list_sessions(Path("outputs"))
            for session in sessions:
                print(f"{session['session_id']}: {session['project_name']}")
            ```
        """
        _validate_session_prefix(prefix)
        base_path = Path(base_dir)
        base_exists = path_exists_with_safe_diagnostics(
            base_path,
            label="session directory",
        )
        if not base_exists:
            return []

        resolved_base = resolve_path_with_safe_diagnostics(
            base_path,
            label="session base directory",
        )
        sessions: list[dict[str, Any]] = []

        # Find all directories matching the session pattern
        list_failure: tuple[type[OSError], int | None] | None = None
        try:
            session_items = list(base_path.iterdir())
        except OSError as error:
            list_failure = (type(error), error.errno)
        if list_failure is not None:
            error_type, error_number = list_failure
            raise error_type(
                error_number,
                "Unable to list session directory",
                redact_sensitive_path(base_path),
            ) from None

        for item in session_items:
            try:
                if _path_is_symlink_with_safe_diagnostics(
                    item,
                    label="session entry",
                ):
                    continue
                is_directory = item.is_dir()
                resolved_item = resolve_path_with_safe_diagnostics(
                    item,
                    label="session entry",
                )
            except (OSError, RuntimeError):
                logger.warning(
                    "Unable to inspect session entry: %s",
                    redact_sensitive_path(item),
                )
                continue
            if not is_directory or resolved_item.parent != resolved_base:
                continue

            # Check if it matches session naming pattern
            if item.name.startswith(prefix):
                # Extract session_id by removing prefix
                potential_session_id = item.name[len(prefix) :]
                try:
                    _validate_session_id(potential_session_id)
                except ValueError:
                    continue

                # Try to load metadata
                metadata_file = item / ".metadata.json"
                metadata = {}
                try:
                    metadata_is_symlink = _path_is_symlink_with_safe_diagnostics(
                        metadata_file,
                        label="session metadata",
                    )
                    metadata_exists = not metadata_is_symlink and metadata_file.exists()
                except (OSError, RuntimeError):
                    logger.warning(
                        "Unable to inspect session metadata: %s",
                        redact_sensitive_path(metadata_file),
                    )
                    metadata_exists = False
                if metadata_exists:
                    try:
                        with open(metadata_file, encoding="utf-8") as f:
                            loaded_metadata = json.load(f)
                        if isinstance(loaded_metadata, dict):
                            metadata = _project_metadata_for_durable_storage(
                                loaded_metadata
                            )
                    except Exception:
                        pass

                project_name = metadata.get("project_name")
                if not isinstance(project_name, str):
                    project_name = "unknown"
                created_at = metadata.get("created_at")
                if not isinstance(created_at, str):
                    created_at = None
                sessions.append(
                    {
                        "session_id": redact_sensitive_path(potential_session_id),
                        "session_dir": redact_sensitive_path(item),
                        "project_name": project_name,
                        "created_at": created_at,
                        "metadata": metadata,
                    }
                )

        # Sort by creation time (newest first)
        sessions.sort(key=lambda session: session["created_at"] or "", reverse=True)

        return sessions

    def __repr__(self) -> str:
        """String representation of session."""
        return (
            "SessionManager("
            f"id={redact_sensitive_path(self.session_id)}, "
            f"dir={redact_sensitive_path(self.session_dir)})"
        )

    def __str__(self) -> str:
        """Human-readable session info."""
        return (
            f"Session {redact_sensitive_path(self.session_id)} "
            f"({redact_sensitive_config(self.project_name)}) @ "
            f"{redact_sensitive_path(self.session_dir)}"
        )
