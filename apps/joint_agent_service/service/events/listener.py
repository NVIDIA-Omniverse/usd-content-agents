# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI Event Listener - Bridges Joint Agent API events to FastAPI SSE.

This listener implements the EventListener protocol from the Joint Agent API and
converts API events into ProgressEvent objects for the service's EventBus.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import EventListener
from world_understanding.utils.preview_paths import (
    normalize_render_image_path,
    preview_filename_for_render_path,
    resolve_preview_filename,
)

from ..runtime import get_event_bus
from ..runtime.events import ProgressEvent, StepState

logger = logging.getLogger(__name__)


class FastAPIEventListener(EventListener):
    """Event listener that bridges Joint Agent API events to FastAPI SSE.

    Thread-safe: Can be called from thread pool workers (via asyncio.to_thread).
    Uses loop.call_soon_threadsafe() to emit events on the main event loop.
    """

    def __init__(
        self,
        session_id: str,
        session_dir: "Path | None" = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        """Initialize FastAPI event listener.

        Args:
            session_id: Session identifier for this pipeline run
            session_dir: Session directory for finding thumbnails (optional)
            loop: Event loop (None = get running loop)
        """
        self.session_id = session_id
        self.session_dir = session_dir
        self.event_bus = get_event_bus()
        self.current_step: str | None = None

        # Track thumbnailed images to avoid duplicates
        self.thumbnailed_images: set[str] = set()

        # Cache dataset entries for image lookup (loaded on-demand)
        self.dataset_cache: dict[str, dict] = {}

        try:
            self.loop = loop or asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

    # =================================================================
    # Logging Methods (EventListener protocol)
    # =================================================================

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log info message."""
        logger.info(f"[{self.session_id[:8]}] {message}", *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message."""
        logger.debug(f"[{self.session_id[:8]}] {message}", *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message."""
        logger.warning(f"[{self.session_id[:8]}] {message}", *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log error message."""
        logger.error(f"[{self.session_id[:8]}] {message}", *args, **kwargs)

    # =================================================================
    # Event Handling (EventListener protocol)
    # =================================================================

    def event(self, event_type: str, data: dict[str, Any], **kwargs: Any) -> None:
        """Handle structured event from API.

        Maps API events to ProgressEvent objects and emits to EventBus.
        """
        if event_type in ("step.started", "task.started"):
            self.current_step = data.get("step_name") or data.get("task_name")

        progress_event = self._map_event_to_progress(event_type, data)

        if progress_event:
            self._emit_event_threadsafe(progress_event)

    # =================================================================
    # Internal Methods
    # =================================================================

    def _map_event_to_progress(
        self, event_type: str, data: dict[str, Any]
    ) -> ProgressEvent | None:
        """Map API event to ProgressEvent."""
        task_to_step = {
            "VLMInference": "predict",
            "USDPrimTraversalAndRendering": "build_dataset_usd",
        }

        step_name = self.current_step or data.get("step_name")
        if not step_name:
            task_name = data.get("task_name")
            step_name = (
                task_to_step.get(task_name, task_name) if task_name else "unknown"
            )

        if step_name == "VLMInference":
            step_name = "predict"

        # Step lifecycle events
        if event_type == "step.started" or event_type == "task.started":
            if step_name in ("predict", "build_dataset_usd"):
                return None

            return ProgressEvent(
                session_id=self.session_id,
                step=step_name,
                state=StepState.RUNNING,
                percent=0,
                message=data.get("message", f"Starting {step_name}"),
                extra=data,
            )

        elif (
            event_type == "step.progress"
            or event_type == "task.progress"
            or event_type == "prediction.completed"
        ):
            if step_name == "predict" and not (event_type == "prediction.completed"):
                return None

            if step_name == "build_dataset_usd":
                current = data.get("current")
                total = data.get("total")
                if current is None or total is None or current < 1:
                    return None

            known_fields = {
                "step_name",
                "task_name",
                "current",
                "total",
                "percent",
                "percentage",
                "message",
                "entry_id",
            }
            extra_data = {k: v for k, v in data.items() if k not in known_fields}

            # Enrich with thumbnails for rendering
            if step_name == "build_dataset_usd" and self.session_dir:
                thumbnails = self._scan_for_new_thumbnails(step_name)
                if thumbnails:
                    extra_data["rendered_images"] = thumbnails

            # For predictions: lookup image and extract reasoning
            if "entry_id" in data and self.session_dir and step_name == "predict":
                image_path = self._get_prim_image_from_dataset(data["entry_id"])
                if image_path:
                    thumbnail_filename = self._get_thumbnail_filename(image_path)
                    if thumbnail_filename:
                        extra_data["preview_image"] = thumbnail_filename

            # Extract reasoning from response_snippet
            if "response_snippet" in data and data["response_snippet"]:
                import re

                snippet = data["response_snippet"]
                if "<reasoning>" in snippet:
                    match = re.search(
                        r"<reasoning>(.*?)(?:</reasoning>|$)", snippet, re.DOTALL
                    )
                    if match:
                        reasoning = match.group(1).strip()
                        extra_data["reasoning"] = reasoning

            message = data.get("message")

            percent = data.get("percent") or data.get("percentage")
            if percent is not None and not isinstance(percent, int):
                percent = int(percent)

            return ProgressEvent(
                session_id=self.session_id,
                step=step_name,
                state=StepState.RUNNING,
                current=data.get("current"),
                total=data.get("total"),
                percent=percent,
                message=message,
                extra=extra_data if extra_data else None,
            )

        elif event_type == "step.completed" or event_type == "task.completed":
            return ProgressEvent(
                session_id=self.session_id,
                step=step_name,
                state=StepState.COMPLETED,
                percent=100,
                message=data.get("message", f"Completed {step_name}"),
                extra=data,
            )

        elif event_type == "step.failed" or event_type == "task.failed":
            return ProgressEvent(
                session_id=self.session_id,
                step=step_name,
                state=StepState.FAILED,
                message=data.get("error") or data.get("message", f"Failed {step_name}"),
                extra=data,
            )

        # Workflow events
        elif event_type == "workflow.completed":
            extra_data = dict(data) if data else {}
            extra_data["workflow_completed"] = True
            return ProgressEvent(
                session_id=self.session_id,
                step="predict",
                state=StepState.COMPLETED,
                percent=100,
                message="Pipeline completed successfully",
                extra=extra_data,
            )

        elif event_type == "workflow.failed":
            return ProgressEvent(
                session_id=self.session_id,
                step=self.current_step or "unknown",
                state=StepState.FAILED,
                message=data.get("error") or data.get("message", "Pipeline failed"),
                extra=data,
            )

        elif event_type in ("workflow.started", "workflow.executing"):
            return None

        # Rendering progress events
        elif event_type == "rendering.progress":
            current = data.get("current")
            total = data.get("total")

            if current is None or total is None or current < 1:
                return None

            extra_data = dict(data)
            if self.session_dir:
                thumbnails = self._scan_for_new_thumbnails("build_dataset_usd")
                if thumbnails:
                    extra_data["rendered_images"] = thumbnails

            percent = data.get("percent", 0)

            return ProgressEvent(
                session_id=self.session_id,
                step="build_dataset_usd",
                state=StepState.RUNNING,
                current=current,
                total=total,
                percent=percent,
                message=data.get("message", f"Rendering batch {current}/{total}"),
                extra=extra_data if extra_data else None,
            )

        elif event_type == "rendering.completed":
            if self.session_dir:
                self._scan_for_new_thumbnails("build_dataset_usd")
            return None

        elif event_type == "rendering.all_completed":
            extra_data = dict(data)
            if self.session_dir:
                thumbnails = self._scan_for_new_thumbnails("build_dataset_usd")
                if thumbnails:
                    extra_data["rendered_images"] = thumbnails

            return ProgressEvent(
                session_id=self.session_id,
                step="build_dataset_usd",
                state=StepState.COMPLETED,
                percent=50,
                message=f"Rendered {data.get('total_prims', 0)} prims ({data.get('total_images', 0)} images)",
                extra=extra_data if extra_data else None,
            )

        else:
            logger.debug(f"Unhandled event type: {event_type}")
            return None

    def _emit_event_threadsafe(self, event: ProgressEvent) -> None:
        """Emit event in a thread-safe way."""
        if self.loop is None:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.warning(
                    f"No event loop found for {self.session_id[:8]}, event may not be delivered"
                )
                return

        self.loop.call_soon_threadsafe(self._emit_event_sync, event)

    def _emit_event_sync(self, event: ProgressEvent) -> None:
        """Emit event synchronously (called on main event loop)."""
        asyncio.create_task(self.event_bus.emit(event))

    def _load_dataset_cache(self) -> None:
        """Load dataset.jsonl into cache for image lookup."""
        if self.dataset_cache or not self.session_dir:
            return

        try:
            import json

            dataset_path = self.session_dir / "cache" / "dataset" / "dataset.jsonl"
            if not dataset_path.exists():
                return

            with open(dataset_path) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if "id" in entry:
                            self.dataset_cache[entry["id"]] = entry

            logger.info(f"Loaded {len(self.dataset_cache)} dataset entries into cache")
        except Exception as e:
            logger.warning(f"Failed to load dataset cache: {e}")

    def _get_prim_image_from_dataset(self, entry_id: str) -> str | None:
        """Get prim_only image path for an entry from dataset."""
        if not self.dataset_cache:
            self._load_dataset_cache()

        entry = self.dataset_cache.get(entry_id)
        if not entry or "images" not in entry:
            return None

        for img_path in entry["images"]:
            if "prim_only" in str(img_path):
                return str(img_path)

        for img_path in entry["images"]:
            if "reference" not in str(img_path):
                return str(img_path)

        return None

    def _get_thumbnail_filename(self, image_path: str) -> str | None:
        """Get thumbnail filename from image path."""
        try:
            normalized_path = normalize_render_image_path(image_path)
            if self.session_dir:
                preview_dir = self.session_dir / "cache" / "preview"
                if preview_dir.exists():
                    return resolve_preview_filename(preview_dir, normalized_path)
            return preview_filename_for_render_path(normalized_path)

        except Exception as e:
            logger.warning(
                f"Failed to construct thumbnail filename for {image_path}: {e}"
            )
            return None

    def _scan_for_new_thumbnails(self, step_name: str) -> list[str]:
        """Scan for new rendered images and create thumbnails."""
        if not self.session_dir or step_name != "build_dataset_usd":
            return []

        try:
            renders_dir = self.session_dir / "cache" / "dataset" / "usd" / "renders"
            preview_dir = self.session_dir / "cache" / "preview"

            if not renders_dir.exists():
                return []

            preview_dir.mkdir(parents=True, exist_ok=True)

            all_pngs = list(renders_dir.rglob("*.png"))
            new_thumbnails = []

            for img_path in all_pngs:
                relative_path = str(img_path.relative_to(renders_dir))
                unique_filename = resolve_preview_filename(preview_dir, relative_path)

                if unique_filename in self.thumbnailed_images:
                    continue

                preview_path = preview_dir / unique_filename
                if preview_path.is_file():
                    self.thumbnailed_images.add(unique_filename)
                    continue
                try:
                    from PIL import Image

                    with Image.open(img_path) as img:
                        img.thumbnail((128, 128), Image.Resampling.LANCZOS)
                        img.save(preview_path, "PNG", optimize=True)

                    self.thumbnailed_images.add(unique_filename)
                    new_thumbnails.append(unique_filename)

                except Exception as e:
                    logger.warning(
                        f"Failed to create thumbnail for {img_path.name}: {e}"
                    )

            return new_thumbnails

        except Exception as e:
            logger.warning(f"Failed to scan for thumbnails: {e}")
            return []
