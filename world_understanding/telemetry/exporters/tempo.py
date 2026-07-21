# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grafana Tempo exporter using OTLP/gRPC protocol."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - static typing import.
    from opentelemetry.sdk.trace.export import SpanProcessor

logger = logging.getLogger(__name__)


def create_tempo_exporter(
    endpoint: str = "localhost:4317",
    insecure: bool = True,
    compression: str = "gzip",
    max_queue_size: int = 2048,
    schedule_delay_ms: int = 5000,
    max_export_batch_size: int = 512,
    export_timeout_ms: int = 30000,
) -> SpanProcessor | None:
    """
    Create a Tempo exporter using OTLP/gRPC.

    Args:
        endpoint: Tempo gRPC endpoint (host:port)
        insecure: Use insecure connection (no TLS)
        compression: Compression algorithm ('none' or 'gzip')
        max_queue_size: Maximum spans in queue
        schedule_delay_ms: Batch export interval
        max_export_batch_size: Maximum spans per batch
        export_timeout_ms: Export timeout

    Returns:
        BatchSpanProcessor configured for Tempo, or None if dependencies unavailable
    """
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Configure compression
        compression_enum = None
        if compression == "gzip":
            from grpc import Compression

            compression_enum = Compression.Gzip

        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            insecure=insecure,
            compression=compression_enum,
        )

        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=max_queue_size,
            schedule_delay_millis=schedule_delay_ms,
            max_export_batch_size=max_export_batch_size,
            export_timeout_millis=export_timeout_ms,
        )

        logger.info(f"Tempo exporter initialized: {endpoint} (insecure={insecure})")
        return processor

    except ImportError as e:
        logger.warning(
            f"Tempo exporter unavailable - missing dependency: {e}. "
            "Install with: pip install opentelemetry-exporter-otlp-proto-grpc"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Tempo exporter: {e}")
        return None
