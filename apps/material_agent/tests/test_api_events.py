# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent API event system."""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from world_understanding.agentic.events import (
    CLIEventListener,
    CollectingEventListener,
    EventListener,
    NoOpEventListener,
    create_default_listener,
)


class TestEventListenerProtocol:
    """Tests for EventListener protocol."""

    def test_cli_event_listener_implements_protocol(self):
        """Test that CLIEventListener implements EventListener protocol."""
        listener = CLIEventListener()

        # Has all required methods
        assert hasattr(listener, "info")
        assert hasattr(listener, "debug")
        assert hasattr(listener, "warning")
        assert hasattr(listener, "error")
        assert hasattr(listener, "event")

    def test_collecting_event_listener_implements_protocol(self):
        """Test that CollectingEventListener implements EventListener protocol."""
        listener = CollectingEventListener()

        assert hasattr(listener, "info")
        assert hasattr(listener, "debug")
        assert hasattr(listener, "warning")
        assert hasattr(listener, "error")
        assert hasattr(listener, "event")


class TestCLIEventListener:
    """Tests for CLIEventListener."""

    def test_cli_listener_logs_messages(self):
        """Test that CLI listener logs messages."""
        mock_logger = Mock(spec=logging.Logger)
        listener = CLIEventListener(logger=mock_logger)

        listener.info("Info message")
        listener.debug("Debug message")
        listener.warning("Warning message")
        listener.error("Error message")

        mock_logger.info.assert_called_once_with("Info message")
        mock_logger.debug.assert_called_once_with("Debug message")
        mock_logger.warning.assert_called_once_with("Warning message")
        mock_logger.error.assert_called_once_with("Error message")

    def test_cli_listener_ignores_events_by_default(self):
        """Test that CLI listener ignores events by default."""
        mock_logger = Mock(spec=logging.Logger)
        listener = CLIEventListener(logger=mock_logger, show_events=False)

        listener.event("test.event", {"data": "value"})

        # No logging should happen for events
        mock_logger.info.assert_not_called()

    def test_cli_listener_shows_events_when_enabled(self):
        """Test that CLI listener shows events when enabled."""
        mock_console = Mock()
        listener = CLIEventListener(console=mock_console, show_events=True)

        listener.event(
            "task.progress",
            {"current": 50, "total": 100, "percentage": 50.0, "task_name": "Test"},
        )

        # Should print to console
        mock_console.print.assert_called_once()


class TestCollectingEventListener:
    """Tests for CollectingEventListener."""

    def test_collecting_listener_collects_logs(self):
        """Test that collecting listener captures all logs."""
        listener = CollectingEventListener()

        listener.info("Info message")
        listener.debug("Debug message")
        listener.warning("Warning message")
        listener.error("Error message")

        assert len(listener.logs) == 4
        assert listener.logs[0]["level"] == "info"
        assert listener.logs[0]["message"] == "Info message"
        assert listener.logs[3]["level"] == "error"

    def test_collecting_listener_collects_events(self):
        """Test that collecting listener captures events."""
        listener = CollectingEventListener()

        listener.event("test.event1", {"data": "value1"})
        listener.event("test.event2", {"data": "value2"})

        assert len(listener.events) == 2
        assert listener.events[0]["type"] == "test.event1"
        assert listener.events[0]["data"]["data"] == "value1"
        assert listener.events[1]["type"] == "test.event2"

    def test_collecting_listener_get_logs_filtered(self):
        """Test filtering logs by level."""
        listener = CollectingEventListener()

        listener.info("Info 1")
        listener.debug("Debug 1")
        listener.info("Info 2")
        listener.error("Error 1")

        info_logs = listener.get_logs("info")
        assert len(info_logs) == 2
        assert all(log["level"] == "info" for log in info_logs)

    def test_collecting_listener_get_events_filtered(self):
        """Test filtering events by type."""
        listener = CollectingEventListener()

        listener.event("task.started", {"task": "A"})
        listener.event("task.progress", {"progress": 50})
        listener.event("task.started", {"task": "B"})

        started_events = listener.get_events("task.started")
        assert len(started_events) == 2
        assert all(evt["type"] == "task.started" for evt in started_events)


class TestNoOpEventListener:
    """Tests for NoOpEventListener."""

    def test_noop_listener_does_nothing(self):
        """Test that NoOp listener doesn't raise errors."""
        listener = NoOpEventListener()

        # Should not raise
        listener.info("Test")
        listener.debug("Test")
        listener.warning("Test")
        listener.error("Test")
        listener.event("test", {})


class TestCreateDefaultListener:
    """Tests for create_default_listener."""

    def test_create_default_listener_returns_cli_listener(self):
        """Test that default listener is CLIEventListener."""
        listener = create_default_listener()

        assert isinstance(listener, CLIEventListener)

    def test_create_default_listener_with_custom_logger(self):
        """Test creating listener with custom logger."""
        mock_logger = Mock(spec=logging.Logger)

        listener = create_default_listener(logger=mock_logger)

        assert isinstance(listener, CLIEventListener)
        assert listener.logger == mock_logger


class TestBenchmarkAPIWithEventListener:
    """Test benchmark API with event listener."""

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_benchmark_with_custom_listener(self, mock_create_workflow, tmp_path):
        """Test that benchmark API uses custom event listener."""
        from material_agent.api import BenchmarkInput, run_benchmark

        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test")

        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "metrics": {
                    "functional_correctness_score": 4.0,
                    "success_rate": 80.0,
                    "total_cases": 10,
                }
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Create collecting listener
        listener = CollectingEventListener()

        # Run with custom listener
        params = BenchmarkInput(config=config_file, event_listener=listener)
        result = run_benchmark(params)

        # Should succeed
        assert result.success is True

        # Should have collected logs
        assert len(listener.logs) > 0
        info_logs = listener.get_logs("info")
        assert any("Starting benchmark" in log["message"] for log in info_logs)

        # Should have collected events
        assert len(listener.events) > 0
        started_events = listener.get_events("workflow.started")
        assert len(started_events) == 1
        assert started_events[0]["data"]["workflow_type"] == "benchmark"

        completed_events = listener.get_events("workflow.completed")
        assert len(completed_events) == 1
        assert "metrics" in completed_events[0]["data"]

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_benchmark_without_listener_uses_default(
        self, mock_create_workflow, tmp_path
    ):
        """Test that benchmark creates default listener if none provided."""
        from material_agent.api import BenchmarkInput, run_benchmark

        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test")

        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(
            return_value={
                "metrics": {"functional_correctness_score": 4.0, "total_cases": 10}
            }
        )
        mock_create_workflow.return_value = mock_workflow

        # Run WITHOUT event listener (should use default)
        params = BenchmarkInput(config=config_file)
        result = run_benchmark(params)

        # Should succeed (default listener works)
        assert result.success is True

    @patch("material_agent.workflows.factory.create_benchmark_workflow_from_config")
    def test_benchmark_emits_failure_event(self, mock_create_workflow, tmp_path):
        """Test that benchmark emits failure event on error."""
        from material_agent.api import BenchmarkInput, run_benchmark

        config_file = tmp_path / "config.yaml"
        config_file.write_text("# test")

        # Mock workflow that raises error
        mock_workflow = Mock()
        mock_workflow.arun = AsyncMock(side_effect=RuntimeError("Test error"))
        mock_create_workflow.return_value = mock_workflow

        listener = CollectingEventListener()

        params = BenchmarkInput(config=config_file, event_listener=listener)
        result = run_benchmark(params)

        # Should fail
        assert result.success is False

        # Should have failure event
        failed_events = listener.get_events("workflow.failed")
        assert len(failed_events) == 1
        assert failed_events[0]["data"]["error"] == "Benchmark failed"
        assert "Test error" not in repr(failed_events)
