# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event-based progress reporting for Material Agent.

This module provides an event listener system that allows different clients
to handle workflow progress in different ways. The CLI uses logging, but API
services can use WebSocket, dashboards, etc.
"""

import logging
from datetime import datetime
from typing import Any, Protocol


def _console_symbol(console: Any, symbol: str, ascii_fallback: str) -> str:
    """Return ``symbol`` only when the target console can encode it."""

    encoding = getattr(console, "encoding", None)
    if not isinstance(encoding, str) or not encoding:
        encoding = "utf-8"
    try:
        symbol.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return ascii_fallback
    return symbol


class EventListener(Protocol):
    """Protocol for receiving both logs and structured events from workflows.

    Listeners can implement this interface to receive:
    - Log messages (info, debug, warning, error)
    - Structured events (step progress, task status, etc.)

    This allows different clients to handle progress differently:
    - CLI: Log to console
    - REST API: Send via WebSocket
    - Tests: Collect for assertions
    - Dashboard: Update UI
    """

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message.

        Args:
            message: Log message
            **kwargs: Additional context
        """
        ...

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message.

        Args:
            message: Log message
            **kwargs: Additional context
        """
        ...

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message.

        Args:
            message: Log message
            **kwargs: Additional context
        """
        ...

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message.

        Args:
            message: Log message
            **kwargs: Additional context
        """
        ...

    def event(self, event_type: str, data: dict[str, Any], **kwargs: Any) -> None:
        """Receive a structured event.

        Args:
            event_type: Type of event (e.g., "step.started", "task.progress")
            data: Event data dictionary
            **kwargs: Additional context
        """
        ...


class CLIEventListener:
    """Event listener for CLI - logs messages and shows structured progress.

    This is the default listener used by the CLI. It:
    - Sends log messages to Python logger
    - Shows structured events as formatted console output (optional)
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        console: Any | None = None,
        show_events: bool = False,
    ):
        """Initialize CLI event listener.

        Args:
            logger: Python logger instance (default: material_agent logger)
            console: Rich console instance (optional)
            show_events: Whether to show structured events in console
        """
        self.logger = logger or logging.getLogger("material_agent")
        self.console = console
        self.show_events = show_events

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(message)

    def event(self, event_type: str, data: dict[str, Any], **kwargs: Any) -> None:
        """Handle structured event.

        Args:
            event_type: Event type (e.g., "step.started", "task.progress")
            data: Event data
            **kwargs: Additional context
        """
        if not self.show_events and not self.console:
            return  # Silent for events unless enabled or console available

        # Always show step events in console (if console available)
        if self.console:
            if event_type == "step.started":
                from rich.panel import Panel

                step_name = data.get("step_name", "Unknown")
                self.console.print(
                    Panel(
                        f"[bold cyan]{step_name}[/bold cyan]",
                        title="Step",
                        border_style="cyan",
                    )
                )

            elif event_type == "step.completed":
                step_name = data.get("step_name", "Unknown")
                success = _console_symbol(self.console, "✓", "OK")
                self.console.print(
                    f"[green]{success} Step '{step_name}' completed[/green]"
                )

            elif event_type == "step.failed":
                step_name = data.get("step_name", "Unknown")
                error = data.get("error", "Unknown error")
                failure = _console_symbol(self.console, "✗", "X")
                self.console.print(
                    f"[red]{failure} Step '{step_name}' failed: {error}[/red]"
                )

            elif event_type == "pipeline.overview":
                from rich.table import Table

                steps = data.get("steps", [])
                completed_steps = set(data.get("completed_steps", []))

                table = Table(title="Pipeline Overview", show_header=True)
                table.add_column("Step", style="cyan")
                table.add_column("Status", style="yellow")

                for step in steps:
                    if step in completed_steps:
                        status = f"{_console_symbol(self.console, '✓', 'OK')} Completed"
                        style = "green"
                    else:
                        status = f"{_console_symbol(self.console, '○', '-')} Pending"
                        style = "white"

                    table.add_row(step, f"[{style}]{status}[/{style}]")

                self.console.print(table)
                self.console.print()

            elif event_type == "pipeline.success":
                from rich.panel import Panel

                self.console.print()
                self.console.print(
                    Panel(
                        f"[bold green]{_console_symbol(self.console, '✓', 'OK')} "
                        "Pipeline completed successfully![/bold green]\n\n"
                        "All steps executed successfully.",
                        title="Success",
                        border_style="green",
                    )
                )

            elif event_type == "pipeline.failed":
                from rich.panel import Panel

                failed_step = data.get("failed_step", "Unknown")
                error = data.get("error", "Unknown error")

                self.console.print()
                self.console.print(
                    Panel(
                        f"[bold red]{_console_symbol(self.console, '✗', 'X')} "
                        f"Pipeline failed at step: {failed_step}[/bold red]\n\n"
                        f"Error: {error}",
                        title="Failed",
                        border_style="red",
                    )
                )

            elif event_type == "pipeline.config.display":
                from rich.panel import Panel

                skip_steps_str = ", ".join(data.get("skip_steps", [])) or "None"
                only_steps_str = ", ".join(data.get("only_steps", [])) or "All"
                resume_str = "Yes" if data.get("resume", False) else "No"
                dry_run_str = "Yes" if data.get("dry_run", False) else "No"
                clean_str = (
                    "Yes (working dir + output files)"
                    if data.get("clean", False)
                    else "No"
                )

                self.console.print(
                    Panel.fit(
                        f"[bold]Material Agent Pipeline[/bold]\n\n"
                        f"Configuration: {data.get('config', 'N/A')}\n"
                        f"Skip steps: {skip_steps_str}\n"
                        f"Only steps: {only_steps_str}\n"
                        f"Resume: {resume_str}\n"
                        f"Dry run: {dry_run_str}\n"
                        f"Clean: {clean_str}",
                        title="Pipeline Configuration",
                        border_style="blue",
                    )
                )

        # Show other events only if enabled
        if not self.show_events:
            return

        if self.console and event_type == "task.progress":
            # Show progress updates
            current = data.get("current", 0)
            total = data.get("total", 0)
            pct = data.get("percentage", 0)
            task = data.get("task_name", "Task")
            self.console.print(
                f"[dim]{task}: {current}/{total} ({pct:.1f}%)[/dim]", end="\r"
            )


class CollectingEventListener:
    """Event listener that collects all logs and events for inspection.

    Useful for testing and debugging.
    """

    def __init__(self):
        """Initialize collecting listener."""
        self.logs: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def info(self, message: str, **kwargs: Any) -> None:
        """Collect info log."""
        self.logs.append(
            {"level": "info", "message": message, "timestamp": datetime.now(), **kwargs}
        )

    def debug(self, message: str, **kwargs: Any) -> None:
        """Collect debug log."""
        self.logs.append(
            {
                "level": "debug",
                "message": message,
                "timestamp": datetime.now(),
                **kwargs,
            }
        )

    def warning(self, message: str, **kwargs: Any) -> None:
        """Collect warning log."""
        self.logs.append(
            {
                "level": "warning",
                "message": message,
                "timestamp": datetime.now(),
                **kwargs,
            }
        )

    def error(self, message: str, **kwargs: Any) -> None:
        """Collect error log."""
        self.logs.append(
            {
                "level": "error",
                "message": message,
                "timestamp": datetime.now(),
                **kwargs,
            }
        )

    def event(self, event_type: str, data: dict[str, Any], **kwargs: Any) -> None:
        """Collect structured event."""
        self.events.append(
            {
                "type": event_type,
                "data": data,
                "timestamp": datetime.now(),
                **kwargs,
            }
        )

    def get_logs(self, level: str | None = None) -> list[dict[str, Any]]:
        """Get collected logs, optionally filtered by level.

        Args:
            level: Optional level filter ("info", "debug", "warning", "error")

        Returns:
            List of log dictionaries
        """
        if level:
            return [log for log in self.logs if log["level"] == level]
        return self.logs

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Get collected events, optionally filtered by type.

        Args:
            event_type: Optional event type filter

        Returns:
            List of event dictionaries
        """
        if event_type:
            return [evt for evt in self.events if evt["type"] == event_type]
        return self.events


class NoOpEventListener:
    """Event listener that does nothing.

    Useful for silent execution or when you don't want any output.
    """

    def info(self, message: str, **kwargs: Any) -> None:
        """No-op."""
        pass

    def debug(self, message: str, **kwargs: Any) -> None:
        """No-op."""
        pass

    def warning(self, message: str, **kwargs: Any) -> None:
        """No-op."""
        pass

    def error(self, message: str, **kwargs: Any) -> None:
        """No-op."""
        pass

    def event(self, event_type: str, data: dict[str, Any], **kwargs: Any) -> None:
        """No-op."""
        pass


def create_default_listener(
    verbose: bool = False,
    show_events: bool = False,
    logger: logging.Logger | None = None,
    console: Any | None = None,
) -> EventListener:
    """Create default event listener for CLI usage.

    Args:
        verbose: Enable verbose logging
        show_events: Show structured events in console
        logger: Python logger (default: material_agent logger)
        console: Rich console instance (optional)

    Returns:
        CLIEventListener configured for CLI usage
    """
    if logger is None:
        logger = logging.getLogger("material_agent")

    return CLIEventListener(logger=logger, console=console, show_events=show_events)


class LoggerAsListener:
    """Adapter that wraps a Python logger to act as an EventListener.

    This is used as a fallback when no event_listener is provided in context.
    It logs messages normally and ignores structured events.
    """

    def __init__(self, logger: logging.Logger):
        """Initialize logger adapter.

        Args:
            logger: Python logger instance to wrap
        """
        self.logger = logger

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(message)

    def event(self, event_type: str, data: dict[str, Any], **kwargs: Any) -> None:
        """Ignore structured events (log at debug level)."""
        self.logger.debug(f"Event: {event_type} - {data}")


def get_listener(
    context: dict[str, Any], logger_name: str = "material_agent"
) -> EventListener:
    """Get event listener from context or create logger fallback.

    This is the helper function that tasks should use to get a listener.
    It ensures backward compatibility when event_listener is not in context.

    Args:
        context: Workflow context (may contain event_listener)
        logger_name: Name of logger to use as fallback

    Returns:
        EventListener from context, or LoggerAsListener fallback

    Example:
        >>> # In a task
        >>> listener = get_listener(context)
        >>> listener.info("Processing...")
        >>> listener.event("task.progress", {"current": 10, "total": 100})
    """
    event_listener = context.get("event_listener")

    if event_listener is not None:
        return event_listener

    # Fallback: Create logger adapter
    return LoggerAsListener(logging.getLogger(logger_name))
