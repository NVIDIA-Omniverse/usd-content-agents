# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Telemetry module for the MAA (World Understanding) project.

This module provides OpenTelemetry-based distributed tracing capabilities
for the MAA (World Understanding) project. It supports multiple exporter
backends including Grafana Tempo, Langfuse, and custom HTTP endpoints.

Usage:
    >>> from world_understanding.telemetry import (
    ...     initialize_telemetry,
    ...     shutdown_telemetry,
    ...     get_tracer,
    ...     TelemetryConfig,
    ... )
    >>> config = TelemetryConfig(service_name="my-service")
    >>> initialize_telemetry(config)
    >>> tracer = get_tracer(__name__)
    >>> with tracer.start_as_current_span("my-operation") as span:
    ...     span.set_attribute("key", "value")
    ...     # ... your code here ...
    >>> shutdown_telemetry()

Environment Variables:
    OTEL_ENABLED: Enable/disable telemetry (default: true)
    OTEL_SERVICE_NAME: Service name for traces
    OTEL_EXPORTERS: Comma-separated list of exporters
    OTEL_SAMPLE_RATE: Sampling rate (0.0 to 1.0)
    OTEL_TEMPO__ENDPOINT: Tempo gRPC endpoint
    OTEL_LANGFUSE__PUBLIC_KEY: Langfuse public key
    OTEL_LANGFUSE__SECRET_KEY: Langfuse secret key
"""

from __future__ import annotations

import atexit
import logging
from typing import TYPE_CHECKING, Any

# Import configuration classes
from world_understanding.telemetry.attributes import GenAIAttributes, MAAttributes
from world_understanding.telemetry.config import (
    ExporterType,
    FileConfig,
    HTTPConfig,
    LangfuseConfig,
    TelemetryConfig,
    TempoConfig,
)
from world_understanding.telemetry.decorators import (
    add_token_usage_to_span,
    traced,
    traced_llm,
    traced_vlm,
)

if TYPE_CHECKING:  # pragma: no cover - static typing imports.
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Span, Tracer

logger = logging.getLogger(__name__)

# Module-level state for telemetry management
_tracer_provider: TracerProvider | None = None
_initialized: bool = False
_shutdown_registered: bool = False


def initialize_telemetry(config: TelemetryConfig) -> TracerProvider | None:
    """Initialize the OpenTelemetry tracing system.

    Sets up the TracerProvider with configured exporters and resource attributes.
    This function should be called once at application startup.

    Args:
        config: Telemetry configuration specifying exporters and settings.

    Returns:
        The configured TracerProvider, or None if telemetry is disabled or
        initialization fails.

    Raises:
        RuntimeError: If telemetry has already been initialized.

    Example:
        >>> config = TelemetryConfig(
        ...     service_name="my-service",
        ...     exporters=[ExporterType.CONSOLE, ExporterType.TEMPO],
        ...     tempo=TempoConfig(endpoint="tempo:4317"),
        ... )
        >>> provider = initialize_telemetry(config)
    """
    global _tracer_provider, _initialized, _shutdown_registered

    if _initialized:
        logger.warning("Telemetry already initialized, skipping re-initialization")
        return _tracer_provider

    if not config.enabled:
        logger.info("Telemetry is disabled via configuration")
        _initialized = True
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import (
            ParentBasedTraceIdRatio,
            TraceIdRatioBased,
        )
    except ImportError as e:
        logger.error(
            f"OpenTelemetry SDK not installed: {e}. "
            "Install with: pip install opentelemetry-sdk"
        )
        return None

    try:
        # Create resource with service information
        resource = Resource.create(
            {
                "service.name": config.service_name,
                "service.version": config.service_version,
                "deployment.environment": config.environment,
            }
        )

        # Configure sampler based on sample rate
        if config.sample_rate < 1.0:
            sampler = ParentBasedTraceIdRatio(config.sample_rate)
        else:
            sampler = ParentBasedTraceIdRatio(1.0)

        # Create the tracer provider
        _tracer_provider = TracerProvider(resource=resource, sampler=sampler)

        # Add configured exporters
        _add_exporters(_tracer_provider, config)

        # Set as the global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        # Register shutdown handler
        if not _shutdown_registered:
            atexit.register(shutdown_telemetry)
            _shutdown_registered = True

        _initialized = True
        exporters = config.get_exporters()
        logger.info(
            f"Telemetry initialized: service={config.service_name}, "
            f"exporters={[e.value for e in exporters]}, "
            f"sample_rate={config.sample_rate}"
        )

        return _tracer_provider

    except Exception as e:
        logger.error(f"Failed to initialize telemetry: {e}")
        return None


def _add_exporters(provider: TracerProvider, config: TelemetryConfig) -> None:
    """Add configured exporters to the tracer provider.

    Args:
        provider: The TracerProvider to add exporters to.
        config: Telemetry configuration with exporter settings.
    """
    from world_understanding.telemetry.exporters import (
        create_console_exporter,
        create_file_exporter,
        create_http_exporter,
        create_langfuse_exporter,
        create_tempo_exporter,
    )

    for exporter_type in config.get_exporters():
        processor = None

        if exporter_type == ExporterType.NONE:
            logger.debug("Skipping NONE exporter type")
            continue

        elif exporter_type == ExporterType.CONSOLE:
            processor = create_console_exporter()

        elif exporter_type == ExporterType.TEMPO:
            if config.tempo is None:
                logger.warning(
                    "Tempo exporter requested but no tempo config provided, skipping"
                )
                continue
            processor = create_tempo_exporter(
                endpoint=config.tempo.endpoint,
                insecure=config.tempo.insecure,
                compression=config.tempo.compression,
                max_queue_size=config.batch_max_queue_size,
                schedule_delay_ms=config.batch_schedule_delay_ms,
                max_export_batch_size=config.batch_max_export_size,
                export_timeout_ms=config.batch_export_timeout_ms,
            )

        elif exporter_type == ExporterType.LANGFUSE:
            if config.langfuse is None:
                logger.warning(
                    "Langfuse exporter requested but no langfuse config provided, "
                    "skipping"
                )
                continue
            processor = create_langfuse_exporter(
                endpoint=config.langfuse.endpoint,
                public_key=config.langfuse.public_key,
                secret_key=config.langfuse.secret_key,
                environment=config.langfuse.environment,
                max_queue_size=config.batch_max_queue_size,
                schedule_delay_ms=config.batch_schedule_delay_ms,
                max_export_batch_size=config.batch_max_export_size,
                export_timeout_ms=config.batch_export_timeout_ms,
            )

        elif exporter_type == ExporterType.HTTP:
            if config.http is None:
                logger.warning(
                    "HTTP exporter requested but no http config provided, skipping"
                )
                continue
            processor = create_http_exporter(
                endpoint=config.http.endpoint,
                headers=config.http.headers,
                timeout_seconds=config.http.timeout_seconds,
                max_queue_size=config.batch_max_queue_size,
                schedule_delay_ms=config.batch_schedule_delay_ms,
                max_export_batch_size=config.batch_max_export_size,
                export_timeout_ms=config.batch_export_timeout_ms,
            )

        elif exporter_type == ExporterType.FILE:
            if config.file is None:
                logger.warning(
                    "File exporter requested but no file config provided, skipping"
                )
                continue
            processor = create_file_exporter(
                path=config.file.path,
                append=config.file.append,
                max_queue_size=config.batch_max_queue_size,
                schedule_delay_ms=config.batch_schedule_delay_ms,
                max_export_batch_size=config.batch_max_export_size,
                export_timeout_ms=config.batch_export_timeout_ms,
            )

        if processor is not None:
            provider.add_span_processor(processor)
            logger.debug(f"Added {exporter_type.value} exporter")
        else:
            logger.warning(f"Failed to create {exporter_type.value} exporter")


def shutdown_telemetry() -> None:
    """Shutdown the telemetry system gracefully.

    Flushes any pending spans and releases resources. This function is
    automatically registered with atexit but can be called explicitly
    for controlled shutdown.

    This function is idempotent and safe to call multiple times.
    """
    global _tracer_provider, _initialized

    if not _initialized or _tracer_provider is None:
        logger.debug("Telemetry not initialized or already shut down")
        return

    try:
        _tracer_provider.shutdown()
        logger.info("Telemetry shut down successfully")
    except Exception as e:
        logger.error(f"Error during telemetry shutdown: {e}")
    finally:
        _tracer_provider = None
        _initialized = False


def get_tracer(name: str, version: str | None = None) -> Tracer:
    """Get a tracer instance for creating spans.

    Args:
        name: Name for the tracer, typically __name__ of the module.
        version: Optional version string for the tracer.

    Returns:
        A Tracer instance. If telemetry is not initialized or disabled,
        returns a no-op tracer that doesn't produce spans.

    Example:
        >>> tracer = get_tracer(__name__)
        >>> with tracer.start_as_current_span("my-operation") as span:
        ...     span.set_attribute("key", "value")
    """
    from opentelemetry import trace

    return trace.get_tracer(name, version)


def get_current_span() -> Span | None:
    """Get the currently active span from the context.

    Returns:
        The current active Span, or None if there is no active span
        or if the current span is a non-recording span.

    Example:
        >>> span = get_current_span()
        >>> if span is not None:
        ...     span.set_attribute("additional.info", "value")
    """
    from opentelemetry import trace

    span = trace.get_current_span()

    # Check if it's a valid recording span
    if span is None or not span.is_recording():
        return None

    return span


# Public API exports
__all__ = [
    # Initialization functions
    "initialize_telemetry",
    "shutdown_telemetry",
    "get_tracer",
    "get_current_span",
    # Configuration classes
    "TelemetryConfig",
    "TempoConfig",
    "LangfuseConfig",
    "HTTPConfig",
    "FileConfig",
    "ExporterType",
    # Attribute classes
    "MAAttributes",
    "GenAIAttributes",
    # Decorators
    "traced",
    "traced_llm",
    "traced_vlm",
    "add_token_usage_to_span",
]
