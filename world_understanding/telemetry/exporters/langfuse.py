# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Langfuse exporter using OTLP/HTTP protocol."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - static typing import.
    from opentelemetry.sdk.trace.export import SpanProcessor

logger = logging.getLogger(__name__)


def create_langfuse_exporter(
    endpoint: str = "https://cloud.langfuse.com/api/public/otel",
    public_key: str = "",
    secret_key: str = "",
    environment: str = "default",
    max_queue_size: int = 2048,
    schedule_delay_ms: int = 5000,
    max_export_batch_size: int = 512,
    export_timeout_ms: int = 30000,
) -> SpanProcessor | None:
    """
    Create a Langfuse exporter using OTLP/HTTP.

    Langfuse does NOT support gRPC - this uses HTTP/protobuf.

    Args:
        endpoint: Langfuse OTLP base URL (without /v1/traces)
        public_key: Langfuse public API key
        secret_key: Langfuse secret API key
        environment: Environment name for traces
        max_queue_size: Maximum spans in queue
        schedule_delay_ms: Batch export interval
        max_export_batch_size: Maximum spans per batch
        export_timeout_ms: Export timeout

    Returns:
        BatchSpanProcessor configured for Langfuse, or None if:
        - public_key or secret_key is empty
        - Required dependencies are not installed
        - Initialization fails for any other reason
    """
    if not public_key or not secret_key:
        logger.error("Langfuse exporter requires public_key and secret_key")
        return None

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Create Basic Auth header
        credentials = f"{public_key}:{secret_key}"
        auth_header = base64.b64encode(credentials.encode()).decode()

        # Ensure endpoint ends correctly
        traces_endpoint = endpoint.rstrip("/")
        if not traces_endpoint.endswith("/v1/traces"):
            traces_endpoint = f"{traces_endpoint}/v1/traces"

        exporter = OTLPSpanExporter(
            endpoint=traces_endpoint,
            headers={
                "Authorization": f"Basic {auth_header}",
                "X-Langfuse-Environment": environment,
            },
        )

        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=max_queue_size,
            schedule_delay_millis=schedule_delay_ms,
            max_export_batch_size=max_export_batch_size,
            export_timeout_millis=export_timeout_ms,
        )

        logger.info(f"Langfuse exporter initialized: {endpoint} (env={environment})")
        return processor

    except ImportError as e:
        logger.warning(
            f"Langfuse exporter unavailable - missing dependency: {e}. "
            "Install with: pip install opentelemetry-exporter-otlp-proto-http"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse exporter: {e}")
        return None
