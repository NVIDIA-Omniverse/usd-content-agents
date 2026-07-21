# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local JSONL span exporter."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static typing imports.
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanProcessor

logger = logging.getLogger(__name__)


class LocalJsonSpanExporter:
    """Export spans to a local JSONL file."""

    def __init__(self, path: str, append: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not append:
            self.path.write_text("", encoding="utf-8")

    def export(self, spans: Sequence[ReadableSpan]) -> int:
        """Export spans to file.

        Returns:
            SpanExportResult.SUCCESS (0) or SpanExportResult.FAILURE (1)
        """
        from opentelemetry.sdk.trace.export import SpanExportResult

        if not spans:
            return SpanExportResult.SUCCESS

        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                for span in spans:
                    rec = self._serialize_span(span)
                    f.write(json.dumps(rec, default=str, ensure_ascii=True))
                    f.write("\n")
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.warning(f"File span export failed: {e}")
            return SpanExportResult.FAILURE

    def _serialize_span(self, span: ReadableSpan) -> dict[str, Any]:
        """Convert span to JSON-serializable dict."""
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
            "resource": dict(span.resource.attributes) if span.resource else {},
        }

    def shutdown(self) -> None:
        """No-op shutdown."""
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op flush for file exporter."""
        return True


def create_file_exporter(
    path: str,
    append: bool = True,
    max_queue_size: int = 2048,
    schedule_delay_ms: int = 5000,
    max_export_batch_size: int = 512,
    export_timeout_ms: int = 30000,
) -> SpanProcessor | None:
    """Create a local JSONL file exporter with batch processing."""
    try:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = LocalJsonSpanExporter(path=path, append=append)
        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=max_queue_size,
            schedule_delay_millis=schedule_delay_ms,
            max_export_batch_size=max_export_batch_size,
            export_timeout_millis=export_timeout_ms,
        )
        logger.info(f"File span exporter initialized: {path}")
        return processor
    except ImportError as e:
        logger.warning(f"File exporter unavailable - missing dependency: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize file exporter: {e}")
        return None
