# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic HTTP/JSON span exporter for custom backends."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static typing imports.
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanProcessor

logger = logging.getLogger(__name__)


class HTTPJsonExporter:
    """
    Custom span exporter that sends spans as JSON to an HTTP endpoint.

    This exporter is useful for custom observability backends that don't
    support OTLP but can accept JSON payloads.
    """

    def __init__(
        self,
        endpoint: str,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> None:
        """
        Initialize HTTP JSON exporter.

        Args:
            endpoint: HTTP endpoint URL to send spans to
            headers: Optional HTTP headers (auth, content-type, etc.)
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self._session: Any = None  # Lazy-loaded requests.Session

    def _get_session(self) -> Any:
        """Lazy-load requests session."""
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update(self.headers)
        return self._session

    def export(self, spans: Sequence[ReadableSpan]) -> int:
        """
        Export spans to HTTP endpoint.

        Args:
            spans: Sequence of spans to export

        Returns:
            SpanExportResult.SUCCESS (0) or SpanExportResult.FAILURE (1)
        """
        from opentelemetry.sdk.trace.export import SpanExportResult

        if not spans:
            return SpanExportResult.SUCCESS

        try:
            payload = {
                "spans": [self._serialize_span(span) for span in spans],
                "resource": self._serialize_resource(spans[0].resource)
                if spans
                else {},
            }

            response = self._get_session().post(
                self.endpoint,
                data=json.dumps(payload, default=str),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return SpanExportResult.SUCCESS

        except Exception as e:
            logger.warning(f"HTTP export failed: {e}")
            return SpanExportResult.FAILURE

    def _serialize_span(self, span: ReadableSpan) -> dict[str, Any]:
        """Convert a span to JSON-serializable dictionary."""
        parent_span_id = None
        if span.parent is not None:
            parent_span_id = format(span.parent.span_id, "016x")

        return {
            "trace_id": format(span.context.trace_id, "032x"),
            "span_id": format(span.context.span_id, "016x"),
            "parent_span_id": parent_span_id,
            "name": span.name,
            "kind": span.kind.name if span.kind else "INTERNAL",
            "start_time_unix_nano": span.start_time,
            "end_time_unix_nano": span.end_time,
            "attributes": dict(span.attributes) if span.attributes else {},
            "status": {
                "code": span.status.status_code.name,
                "message": span.status.description or "",
            },
            "events": [
                {
                    "name": event.name,
                    "timestamp_unix_nano": event.timestamp,
                    "attributes": dict(event.attributes) if event.attributes else {},
                }
                for event in span.events
            ],
            "links": [
                {
                    "trace_id": format(link.context.trace_id, "032x"),
                    "span_id": format(link.context.span_id, "016x"),
                    "attributes": dict(link.attributes) if link.attributes else {},
                }
                for link in span.links
            ],
        }

    def _serialize_resource(self, resource: Any) -> dict[str, Any]:
        """Convert resource to JSON-serializable dictionary."""
        if resource is None:
            return {}
        return dict(resource.attributes) if resource.attributes else {}

    def shutdown(self) -> None:
        """Shutdown the exporter and close connections."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending exports."""
        return True


def create_http_exporter(
    endpoint: str,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = 30,
    max_queue_size: int = 2048,
    schedule_delay_ms: int = 5000,
    max_export_batch_size: int = 512,
    export_timeout_ms: int = 30000,
) -> SpanProcessor | None:
    """
    Create an HTTP JSON exporter with batch processing.

    Args:
        endpoint: HTTP endpoint URL
        headers: Optional HTTP headers
        timeout_seconds: Request timeout
        max_queue_size: Maximum spans in queue
        schedule_delay_ms: Batch export interval
        max_export_batch_size: Maximum spans per batch
        export_timeout_ms: Export timeout

    Returns:
        BatchSpanProcessor with HTTP exporter, or None on failure
    """
    try:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = HTTPJsonExporter(
            endpoint=endpoint,
            headers=headers,
            timeout=timeout_seconds,
        )

        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=max_queue_size,
            schedule_delay_millis=schedule_delay_ms,
            max_export_batch_size=max_export_batch_size,
            export_timeout_millis=export_timeout_ms,
        )

        logger.info(f"HTTP JSON exporter initialized: {endpoint}")
        return processor

    except ImportError as e:
        logger.warning(f"HTTP exporter unavailable - missing dependency: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize HTTP exporter: {e}")
        return None
