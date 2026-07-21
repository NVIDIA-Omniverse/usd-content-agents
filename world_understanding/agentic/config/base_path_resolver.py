# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base path resolver for agent applications.

This module provides a base class for path resolution that handles:
- Session management with SessionManager
- Automatic working directory derivation from session_id
- Basic path resolution relative to config file
- Common directory creation patterns

material-agent, physics-agent, and joint-agent inherit from this base class
and add their domain-specific path methods.
"""

import logging
import uuid
from pathlib import Path
from typing import Any

from world_understanding.agentic.session import SessionManager
from world_understanding.utils.credentials import (
    create_directory_with_safe_diagnostics,
    path_exists_with_safe_diagnostics,
    redact_sensitive_config,
    redact_sensitive_path,
    resolve_path_with_safe_diagnostics,
)

logger = logging.getLogger(__name__)


class BasePathResolver:
    """Base class for path resolution across all agent applications.

    This class handles common path resolution patterns:
    - Session management (automatic .{session_id} directories)
    - Path resolution relative to config file
    - Working directory creation
    - Basic output directory structure

    Subclasses should:
    - Add domain-specific path methods (e.g., get_predictions_dir())
    - Override get_path_summary() to include custom paths
    - Keep domain logic separate from base infrastructure

    Session Management:
    - If session_id is provided but not working_dir: auto-generates .{session_id}
    - If working_dir is provided: uses it directly (backward compatibility)
    - If neither: generates new session_id and creates .{session_id}

    Example:
        >>> class MyPathResolver(BasePathResolver):
        ...     def get_predictions_dir(self) -> Path:
        ...         return self.working_dir / "predictions"
        ...
        >>> config = {"project": {"name": "my_project"}}
        >>> resolver = MyPathResolver(config, Path("config.yaml"))
        >>> resolver.get_predictions_dir()
        Path('.abc123/predictions')
    """

    def __init__(
        self,
        config: dict[str, Any],
        config_file_path: Path,
        default_project_name: str = "agent_project",
    ):
        """Initialize the base path resolver.

        Args:
            config: Configuration dictionary (must contain 'project' section)
            config_file_path: Path to the configuration file (for relative path resolution)
            default_project_name: Default project name if not specified in config

        Raises:
            ValueError: If config is missing required sections
        """
        self.config = config
        self.config_dir = resolve_path_with_safe_diagnostics(
            config_file_path.parent,
            label="configuration directory",
        )

        # Extract project settings
        project = config.get("project") or {}
        self.project_name = project.get("name", default_project_name)

        # Handle session management
        self.session_id, self.working_dir = self._resolve_working_dir_and_session(
            config
        )
        self.working_dir_base = self._resolve_working_dir_base()

        logger.info("Project: %s", redact_sensitive_config(self.project_name))
        logger.info("Session ID: %s", redact_sensitive_config(self.session_id))
        logger.info("Working directory: %s", redact_sensitive_path(self.working_dir))

    def _resolve_working_dir_and_session(
        self, config: dict[str, Any]
    ) -> tuple[str, Path]:
        """Resolve working directory and session ID from config.

        Priority order:
        1. Explicit working_dir provided → use it, derive or generate session_id
        2. No working_dir, session_id provided → use SessionManager with session_id
        3. Neither provided → generate new session_id with SessionManager

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (session_id, working_dir)
        """
        project = config.get("project") or {}
        session_id = project.get("session_id")
        working_dir = project.get("working_dir")

        if working_dir:
            # Explicit working_dir provided - use it directly (backward compatibility)
            logger.info(
                "Using explicit working_dir: %s", redact_sensitive_path(working_dir)
            )
            resolved_working_dir = self._resolve_path(working_dir)

            # _resolve_path should not return None for valid working_dir string
            if resolved_working_dir is None:
                raise ValueError("Invalid working_dir configuration")

            # Use provided session_id or generate/extract one
            if session_id:
                resolved_session_id = session_id
            else:
                # Try to extract from working_dir if it starts with '.'
                if working_dir.startswith("."):
                    resolved_session_id = working_dir[1:]
                else:
                    resolved_session_id = str(uuid.uuid4())

            # Update config with session_id
            if "project" not in config:
                config["project"] = {}
            config["project"]["session_id"] = resolved_session_id

            return resolved_session_id, resolved_working_dir
        else:
            # Use SessionManager for automatic session-based directory management
            diagnostic_config_dir = redact_sensitive_path(self.config_dir)
            session = SessionManager.create(
                base_dir=self.config_dir,
                project_name=self.project_name,
                session_id=session_id,
                prefix=".",
                metadata={"config_dir": diagnostic_config_dir},
            )

            # Update config with generated session_id
            if "project" not in config:
                config["project"] = {}
            config["project"]["session_id"] = session.session_id

            # SessionManager.create() records the real base/session directories
            # in memory. Project those fields to diagnostic-safe strings before
            # persistence while leaving self.config_dir and session.session_dir
            # untouched for runtime path resolution.
            session.metadata["base_dir"] = diagnostic_config_dir
            session.metadata["session_dir"] = redact_sensitive_path(session.session_dir)

            # Save session metadata for future use
            session.update_metadata(
                project_name=self.project_name,
                config_dir=diagnostic_config_dir,
            )

            return session.session_id, session.session_dir

    def _resolve_working_dir_base(self) -> Path:
        """Return the ownership root that confines recursive workspace cleanup.

        Session-managed and ordinary config-relative working directories are
        owned by the configuration directory.  An explicitly configured path
        may resolve outside that directory (including an absolute path); in
        that compatibility case its canonical parent is the narrowest root
        that can own the selected workspace.  The executor still requires the
        workspace to be a strict child of this root and performs descriptor-
        based removal, so this propagation does not restore unconfined cleanup.
        """
        try:
            self.working_dir.relative_to(self.config_dir)
        except ValueError:
            return self.working_dir.parent
        return self.config_dir

    def resolve_path(self, path: str | Path | None) -> Path | None:
        """Resolve a path relative to the config file.

        If the path is relative, it's resolved relative to the config directory.
        If the path is absolute, it's returned as-is.

        Args:
            path: Path string or Path object to resolve

        Returns:
            Resolved absolute Path object, or None if path is None
        """
        if path is None:
            return None

        p = Path(path)
        if not p.is_absolute():
            p = self.config_dir / p
        return resolve_path_with_safe_diagnostics(p, label="configuration path")

    def _resolve_path(self, path: str | Path | None) -> Path | None:
        """Backward-compatible alias for :meth:`resolve_path`."""
        return self.resolve_path(path)

    @staticmethod
    def _path_exists_for_validation(path: Path, *, label: str) -> bool:
        """Check a runtime path without exposing it through OS diagnostics."""
        return path_exists_with_safe_diagnostics(path, label=label)

    def _resolve_path_to_working_dir(self, path: str | Path | None) -> Path | None:
        """Resolve a path relative to the working directory.

        This is used for output paths that should go in the working directory.

        Args:
            path: Path string or Path object to resolve

        Returns:
            Resolved absolute Path object, or None if path is None
        """
        if path is None:
            return None

        p = Path(path)
        if not p.is_absolute():
            p = self.working_dir / p
        return resolve_path_with_safe_diagnostics(p, label="working-directory path")

    def get_output_dir(self) -> Path:
        """Get the standard output directory.

        Returns:
            Path to output directory (working_dir/output)
        """
        return self.working_dir / "output"

    def get_temp_dir(self) -> Path:
        """Get the temporary files directory.

        Returns:
            Path to temp directory (working_dir/temp)
        """
        return self.working_dir / "temp"

    def create_working_directories(self) -> None:
        """Create all necessary working directories.

        This creates the working_dir and its standard subdirectories (output, temp).
        Subclasses should override this to create additional domain-specific directories.
        """
        create_directory_with_safe_diagnostics(
            self.working_dir,
            label="working directory",
        )
        logger.debug(
            "Created working directory: %s",
            redact_sensitive_path(self.working_dir),
        )

        # Create standard directories
        create_directory_with_safe_diagnostics(
            self.get_output_dir(),
            label="output directory",
        )
        logger.debug(
            "Created output directory: %s",
            redact_sensitive_path(self.get_output_dir()),
        )

        create_directory_with_safe_diagnostics(
            self.get_temp_dir(),
            label="temporary directory",
        )
        logger.debug(
            "Created temp directory: %s",
            redact_sensitive_path(self.get_temp_dir()),
        )

    def get_path_summary(self) -> dict[str, Any]:
        """Get a summary of all resolved paths.

        Subclasses should override this to include domain-specific paths.

        Returns:
            Dictionary with path information
        """
        return {
            "project_name": redact_sensitive_config(self.project_name),
            "session_id": redact_sensitive_config(self.session_id),
            "config_dir": redact_sensitive_path(self.config_dir),
            "working_dir": redact_sensitive_path(self.working_dir),
            "output_dir": redact_sensitive_path(self.get_output_dir()),
            "temp_dir": redact_sensitive_path(self.get_temp_dir()),
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"project_name={redact_sensitive_config(self.project_name)!r}, "
            f"session_id={redact_sensitive_config(self.session_id)!r}, "
            f"working_dir={redact_sensitive_path(self.working_dir)!r})"
        )
