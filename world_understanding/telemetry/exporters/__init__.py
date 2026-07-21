# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Telemetry exporters for various backends.

This module provides factory functions for creating OpenTelemetry span
exporters configured for different observability backends.

Supported exporters:
    - Console: Development/debugging output to stdout
    - Tempo: Grafana Tempo via OTLP/gRPC
    - Langfuse: Langfuse observability platform via OTLP/HTTP
    - HTTP: Generic HTTP/JSON exporter for custom backends
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .file import LocalJsonSpanExporter, create_file_exporter
from .http import HTTPJsonExporter, create_http_exporter
from .langfuse import create_langfuse_exporter
from .tempo import create_tempo_exporter

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from opentelemetry.sdk.trace.export import SpanExporter, SpanProcessor

    from world_understanding.telemetry.config import (
        HTTPConfig,
        LangfuseConfig,
        TempoConfig,
    )

logger = logging.getLogger(__name__)


def create_console_exporter() -> SpanProcessor | None:
    """Create a console exporter for development and debugging.

    The console exporter prints spans to stdout in a human-readable format,
    useful for local development and debugging telemetry instrumentation.

    Returns:
        A configured SimpleSpanProcessor with ConsoleSpanExporter, or None if unavailable.
    """
    try:
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        exporter = ConsoleSpanExporter()
        processor = SimpleSpanProcessor(exporter)
        logger.debug("Console span exporter initialized")
        return processor

    except ImportError as e:
        logger.warning(
            f"Console exporter unavailable - missing dependency: {e}. "
            "Install with: pip install opentelemetry-sdk"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to initialize console exporter: {e}")
        return None


__all__ = [
    "create_console_exporter",
    "create_tempo_exporter",
    "create_langfuse_exporter",
    "create_http_exporter",
    "create_file_exporter",
    "HTTPJsonExporter",
    "LocalJsonSpanExporter",
]
