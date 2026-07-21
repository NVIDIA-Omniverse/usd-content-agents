# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared logging configuration for agents.

This module provides a unified logging setup used across all World Understanding agents,
ensuring consistent log formatting, handlers, and configuration.
"""

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from world_understanding.utils.credentials import redact_sensitive_path
from world_understanding.utils.model_auth import ModelAuthenticationLogFilter

_LOG_FILE_OPEN_FAILURE_MESSAGE = "Unable to open log file"


def setup_logging(
    agent_name: str,
    verbose: bool = False,
    log_file: Path | None = None,
    log_level: str = "INFO",
) -> logging.Logger:
    """Setup logging configuration with Rich handler.

    This function configures logging for both the agent and world_understanding packages,
    ensuring consistent behavior across all agents.

    Args:
        agent_name: Name of the agent package (e.g., "material_agent", "physics_agent")
        verbose: Enable verbose output (sets DEBUG level)
        log_file: Optional path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance for the agent

    Example:
        ```python
        from world_understanding.agentic.cli import setup_logging

        logger = setup_logging("material_agent", verbose=True)
        logger.info("Agent started")
        ```
    """
    # Set log level based on verbose flag
    if verbose:
        log_level = "DEBUG"

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create Rich console for pretty terminal output
    console = Console(stderr=True)

    # Create Rich console handler
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=verbose,
        rich_tracebacks=True,
        # Runtime/config locals can contain resolved credentials. Verbose mode
        # may add source paths, but it must never serialize frame locals.
        tracebacks_show_locals=False,
    )
    console_handler.setLevel(numeric_level)

    # Create formatter
    format_str = "%(message)s"
    console_handler.setFormatter(logging.Formatter(format_str))
    console_handler.addFilter(ModelAuthenticationLogFilter())

    # Configure main logger for the agent
    logger = logging.getLogger(agent_name)
    logger.setLevel(numeric_level)
    logger.propagate = False
    logger.handlers.clear()
    logger.addHandler(console_handler)

    # Configure logger for world_understanding package
    wu_logger = logging.getLogger("world_understanding")
    wu_logger.setLevel(numeric_level)
    wu_logger.propagate = False
    wu_logger.handlers.clear()
    wu_logger.addHandler(console_handler)

    # Configure all child loggers to inherit settings
    for name in logging.root.manager.loggerDict:
        if name.startswith(f"{agent_name}.") or name.startswith("world_understanding."):
            child_logger = logging.getLogger(name)
            child_logger.setLevel(numeric_level)

    # Clean up root logger to prevent duplicate console output.
    # Only remove RichHandlers — avoid removing other handlers like pytest's
    # LogCaptureHandler which also inherits from StreamHandler.
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        if isinstance(h, RichHandler):
            root_logger.removeHandler(h)

    # Add file handler if log file specified
    if log_file:
        file_handler: logging.FileHandler | None = None
        file_open_failed = False
        try:
            # Keep the raw path solely for the requested I/O operation.
            file_handler = logging.FileHandler(log_file)
        except Exception:
            file_open_failed = True
            # Do not retain the credential-bearing path in the frame that will
            # be rendered if the caller does not catch the replacement error.
            log_file = None
        if file_open_failed:
            raise RuntimeError(_LOG_FILE_OPEN_FAILURE_MESSAGE)
        assert file_handler is not None
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(ModelAuthenticationLogFilter())
        logger.addHandler(file_handler)
        wu_logger.addHandler(file_handler)
        logger.info("Logging to file: %s", redact_sensitive_path(log_file))

    return logger
