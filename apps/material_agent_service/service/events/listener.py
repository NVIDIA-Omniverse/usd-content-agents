# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI Event Listener - Bridges MAA API events to FastAPI SSE.

This listener implements the EventListener protocol from the MAA API and
converts API events into ProgressEvent objects for the service's EventBus.

It also enriches events with service-specific data (thumbnails, material icons).
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from world_understanding.agentic.events import EventListener
from world_understanding.utils.preview_paths import (
    normalize_render_image_path,
    preview_filename_for_render_path,
    resolve_preview_filename,
)

from ..runtime import get_event_bus
from ..runtime.events import ProgressEvent, StepState
from .sanitization import bounded_step_completion_data

if TYPE_CHECKING:  # pragma: no cover
    from ..session.manager import RegenerationClaim

logger = logging.getLogger(__name__)


class FastAPIEventListener(EventListener):
    """Event listener that bridges MAA API events to FastAPI SSE.

    This listener receives events from the MAA Python API and converts them
    to ProgressEvent objects that flow through the service's EventBus to SSE.

    Thread-safe: Can be called from thread pool workers (via asyncio.to_thread).
    Uses loop.call_soon_threadsafe() to emit events on the main event loop.
    """

    def __init__(
        self,
        session_id: str,
        session_dir: "Path | None" = None,
        loop: asyncio.AbstractEventLoop | None = None,
        session_material_icons: dict[str, str] | None = None,
        regeneration_claim: "RegenerationClaim | None" = None,
    ):
        """Initialize FastAPI event listener.

        Args:
            session_id: Session identifier for this pipeline run
            session_dir: Session directory for finding thumbnails (optional)
            loop: Event loop (None = get running loop)
            session_material_icons: Mapping of material name to icon path for custom materials
            regeneration_claim: Immutable owner for a regeneration listener;
                None explicitly identifies an unbound standard or scene run
        """
        self.session_id = session_id
        self.session_dir = session_dir
        self.event_bus = get_event_bus()
        self.regeneration_claim = regeneration_claim
        self.current_step: str | None = None

        # Track thumbnailed images to avoid duplicates
        self.thumbnailed_images: set[str] = set()

        # Cache dataset entries for image lookup (loaded on-demand)
        self.dataset_cache: dict[str, dict] = {}

        # Session-specific material icons (from custom materials zip)
        self.session_material_icons = session_material_icons or {}

        # Get event loop - try running loop first, fall back to lazy loading
        self.loop: asyncio.AbstractEventLoop | None
        try:
            self.loop = loop or asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - will get it lazily when needed
            self.loop = None

    # =================================================================
    # Logging Methods (EventListener protocol)
    # =================================================================

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        logger.info(f"[{self.session_id[:8]}] {message}")

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        logger.debug(f"[{self.session_id[:8]}] {message}")

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        logger.warning(f"[{self.session_id[:8]}] {message}")

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        logger.error(f"[{self.session_id[:8]}] {message}")

    # =================================================================
    # Event Handling (EventListener protocol)
    # =================================================================

    def event(self, event_type: str, data: dict[str, Any], **kwargs: Any) -> None:
        """Handle structured event from API.

        Maps API events to ProgressEvent objects and emits to EventBus.

        Args:
            event_type: Event type (e.g., "step.started", "step.progress")
            data: Event data dictionary
            **kwargs: Additional context
        """
        # Track current step for progress events
        if event_type in ("step.started", "task.started"):
            self.current_step = data.get("step_name") or data.get("task_name")

        # Map API events to ProgressEvent
        progress_event = self._map_event_to_progress(event_type, data)

        if progress_event:
            self._emit_event_threadsafe(progress_event)

    # =================================================================
    # Internal Methods
    # =================================================================

    def _map_event_to_progress(
        self, event_type: str, data: dict[str, Any]
    ) -> ProgressEvent | None:
        """Map API event to ProgressEvent.

        Args:
            event_type: API event type
            data: Event data

        Returns:
            ProgressEvent or None if event should not be emitted
        """
        # Map task_name to parent step_name for consistent UI display
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

        step_name = task_to_step.get(str(step_name), step_name)

        if event_type in {
            "step.cancelled",
            "task.cancelled",
            "workflow.cancelled",
        }:
            # Core cancellation notifications may be scheduled after the
            # owning worker has returned. The executor emits the sole terminal
            # cancellation event after durable finalization and owner fencing.
            return None

        # Step lifecycle events
        if event_type == "step.started":
            return ProgressEvent(
                session_id=self.session_id,
                step=step_name,
                state=StepState.RUNNING,
                percent=0,
                message=data.get("message", f"Starting {step_name}"),
                extra=data,
            )

        if event_type == "task.started":
            # Skip task.started duplicates for steps that emit richer progress later.
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

            # Skip build_dataset_usd progress until we have valid progress data
            if step_name == "build_dataset_usd":
                current = data.get("current")
                total = data.get("total")
                # Require valid current AND total, and current >= 1
                if current is None or total is None or current < 1:
                    logger.debug(
                        f"[task.progress] Skipping build_dataset_usd - invalid progress: current={current}, total={total}"
                    )
                    return None

            # Extract known progress fields, pass everything else as extra
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

            # Enrich with service-specific data based on step
            # For rendering: scan for new thumbnails
            if step_name == "build_dataset_usd" and self.session_dir:
                thumbnails = self._scan_for_new_thumbnails(step_name)
                if thumbnails:
                    extra_data["rendered_images"] = thumbnails
                    logger.info(f"Added {len(thumbnails)} new thumbnails to event")

            # For prediction: add material icon, prim preview, and extract full reasoning
            if "material" in data:
                extra_data["material"] = data["material"]
                material_icon = self._get_material_icon(data["material"])
                if material_icon:
                    extra_data["material_icon"] = material_icon

            # For predictions: lookup image and extract reasoning
            if "entry_id" in data and self.session_dir and step_name == "predict":
                # Get prim image from dataset
                image_path = self._get_prim_image_from_dataset(data["entry_id"])
                if image_path:
                    # Return just the filename (HTML constructs full URL)
                    thumbnail_filename = self._get_thumbnail_filename(image_path)
                    if thumbnail_filename:
                        extra_data["preview_image"] = thumbnail_filename

            # Extract reasoning from response_snippet (available in event data)
            if "response_snippet" in data and data["response_snippet"]:
                import re

                snippet = data["response_snippet"]
                if "<reasoning>" in snippet:
                    # Try to extract reasoning from snippet
                    match = re.search(
                        r"<reasoning>(.*?)(?:</reasoning>|$)", snippet, re.DOTALL
                    )
                    if match:
                        reasoning = match.group(1).strip()
                        extra_data["reasoning"] = reasoning
                        logger.info(
                            f"[REASONING] Extracted from snippet: {len(reasoning)} chars"
                        )
                else:
                    logger.debug("[REASONING] No <reasoning> tag in snippet")

            # Use material name in message if present
            message = data.get("message")
            if not message and "material" in data:
                message = f"Predicted Event: {data['material']}"

            # Get percent and convert to int if needed
            percent = data.get("percent") or data.get("percentage")
            if percent is not None and not isinstance(percent, int):
                percent = int(percent)

            logger.info(
                f"ProgressEvent: {step_name}, {message}, {percent}, {extra_data}"
            )

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
                extra=bounded_step_completion_data(data),
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
            # The core workflow is complete, but service-side result validation
            # (including strict material coverage) still has to run.  The
            # executor emits the authoritative terminal completion event only
            # after final metadata has been persisted.
            return None

        elif event_type == "workflow.failed":
            # Emit failure event
            return ProgressEvent(
                session_id=self.session_id,
                step=self.current_step or "unknown",
                state=StepState.FAILED,
                message=data.get("error") or data.get("message", "Pipeline failed"),
                extra=data,
            )

        elif event_type in ("workflow.started", "workflow.executing"):
            # These are logged automatically via info/error methods
            return None

        # Rendering progress events - only emit when we have valid progress data
        elif event_type == "rendering.progress":
            # Validate we have real progress data (current >= 1 and total >= 1)
            current = data.get("current")
            total = data.get("total")

            # Skip if no valid progress data
            if current is None or total is None:
                logger.debug(
                    f"[Rendering] Skipping progress - missing current/total: {data}"
                )
                return None

            # Skip if current is 0 (no batches completed yet)
            if current < 1:
                logger.debug(
                    f"[Rendering] Skipping progress - no batches completed: {data}"
                )
                return None

            # Scan for new thumbnails on each rendering event
            extra_data = dict(data)
            if self.session_dir:
                thumbnails = self._scan_for_new_thumbnails("build_dataset_usd")
                if thumbnails:
                    extra_data["rendered_images"] = thumbnails
                    logger.info(f"[Rendering] Added {len(thumbnails)} new thumbnails")

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

        # Per-prim rendering completion - don't update progress, just scan thumbnails
        elif event_type == "rendering.completed":
            # Per-prim completion events don't have progress data
            # Only scan for thumbnails, don't emit progress updates
            if self.session_dir:
                thumbnails = self._scan_for_new_thumbnails("build_dataset_usd")
                if thumbnails:
                    logger.debug(
                        f"[Rendering] Per-prim complete - {len(thumbnails)} new thumbnails"
                    )
            # Don't emit progress event for per-prim completion
            return None

        elif event_type == "rendering.all_completed":
            # Final rendering completion - scan for any remaining thumbnails
            extra_data = dict(data)
            if self.session_dir:
                thumbnails = self._scan_for_new_thumbnails("build_dataset_usd")
                if thumbnails:
                    extra_data["rendered_images"] = thumbnails
                    logger.info(f"[Rendering] Final scan: {len(thumbnails)} thumbnails")

            # Note: This marks build_dataset_usd step as complete (45% of overall pipeline)
            # Don't use percent=100 here as that's reserved for pipeline completion
            return ProgressEvent(
                session_id=self.session_id,
                step="build_dataset_usd",
                state=StepState.COMPLETED,
                percent=45,  # Step completion percentage, not overall
                message=f"Rendered {data.get('total_prims', 0)} prims ({data.get('total_images', 0)} images)",
                extra=extra_data if extra_data else None,
            )

        # Unknown event type - log but don't emit
        else:
            logger.debug(f"Unhandled event type: {event_type}")
            return None

    def _emit_event_threadsafe(self, event: ProgressEvent) -> None:
        """Emit event in a thread-safe way.

        Args:
            event: Progress event to emit
        """
        # Get loop lazily if not set
        if self.loop is None:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                # Called from thread pool - find the main loop
                # Try to get the loop from event bus's context
                logger.warning(
                    f"No event loop found for {self.session_id[:8]}, event may not be delivered"
                )
                return

        # Schedule event emission on the main event loop (thread-safe)
        self.loop.call_soon_threadsafe(self._emit_event_sync, event)

    def _emit_event_sync(self, event: ProgressEvent) -> None:
        """Emit event synchronously (called on main event loop).

        Args:
            event: Progress event to emit
        """
        # Create coroutine and schedule it
        asyncio.create_task(
            self.event_bus.emit_for_owner(
                event,
                regeneration_claim=self.regeneration_claim,
            )
        )

    def _get_material_icon(self, material_name: str) -> str | None:
        """Get material icon path for a material name (service-specific).

        Checks session-specific custom material icons first, then falls back
        to global server material icons.

        Args:
            material_name: Material name from prediction

        Returns:
            Icon URL path or None
        """
        try:
            # First check session-specific icons (from custom materials zip)
            if material_name in self.session_material_icons:
                # Return session-specific URL path (endpoint is in pipeline_router)
                return f"/pipeline/sessions/{self.session_id}/materials/icon/{material_name}"

            # Fall back to global config
            from ..config import config

            icon_path = config.material_icons.get(material_name)
            if icon_path:
                # Return URL path for serving the icon
                return f"/materials/icon/{material_name}"
            return None
        except Exception as e:
            logger.warning(f"Failed to get material icon for '{material_name}': {e}")
            return None

    def _load_dataset_cache(self) -> None:
        """Load dataset.jsonl into cache for image lookup."""
        if self.dataset_cache or not self.session_dir:
            return  # Already loaded or no session_dir

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
        """Get prim_only image path for an entry from dataset.

        Args:
            entry_id: Entry ID (prim path)

        Returns:
            Image path (relative) or None
        """
        # Load dataset on first use
        if not self.dataset_cache:
            self._load_dataset_cache()

        entry = self.dataset_cache.get(entry_id)
        if not entry or "images" not in entry:
            return None

        # Find prim_only image (prefer it for cleaner view)
        for img_path in entry["images"]:
            if "prim_only" in str(img_path):
                return str(img_path)

        # Fallback to first non-reference image
        for img_path in entry["images"]:
            if "reference" not in str(img_path):
                return str(img_path)

        return None

    def _get_reasoning_from_predictions(self, entry_id: str) -> str | None:
        """Get full reasoning for an entry from predictions.jsonl.

        Args:
            entry_id: Entry ID (prim path)

        Returns:
            Full reasoning text or None
        """
        if not self.session_dir:
            logger.debug("[REASONING] No session_dir")
            return None

        try:
            import json
            import re

            predictions_path = (
                self.session_dir / "cache" / "predictions" / "predictions.jsonl"
            )
            if not predictions_path.exists():
                logger.debug(
                    f"[REASONING] predictions.jsonl not found at {predictions_path}"
                )
                return None

            # Read predictions.jsonl and find entry
            with open(predictions_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    pred = json.loads(line)
                    if pred.get("id") == entry_id:
                        # Found the prediction
                        logger.debug(
                            f"[REASONING] Found prediction for {entry_id[:50]}"
                        )
                        vlm_response = pred.get("materials") or pred.get(
                            "vlm_response", {}
                        )
                        logger.debug(
                            f"[REASONING] vlm_response type: {type(vlm_response)}, keys: {vlm_response.keys() if isinstance(vlm_response, dict) else 'N/A'}"
                        )

                        if isinstance(vlm_response, dict):
                            original_response = vlm_response.get(
                                "original_response", ""
                            )
                            logger.debug(
                                f"[REASONING] original_response length: {len(original_response) if original_response else 0}"
                            )

                            if original_response and "<reasoning>" in original_response:
                                match = re.search(
                                    r"<reasoning>(.*?)</reasoning>",
                                    original_response,
                                    re.DOTALL,
                                )
                                if match:
                                    reasoning = match.group(1).strip()
                                    logger.info(
                                        f"[REASONING] Extracted {len(reasoning)} chars"
                                    )
                                    return reasoning
                                else:
                                    logger.debug("[REASONING] No match found in regex")
                            else:
                                logger.debug(
                                    "[REASONING] No <reasoning> tag in original_response"
                                )
                        break

        except Exception as e:
            logger.warning(f"[REASONING] Failed to get reasoning for {entry_id}: {e}")

        return None

    def _get_thumbnail_filename(self, image_path: str) -> str | None:
        """Get thumbnail filename from image path.

        Thumbnails were created during build_dataset_usd with format: {hash}_{filename}
        Returns just the filename - HTML will construct full URL.

        Args:
            image_path: Relative path from dataset (e.g., "usd/renders/.../mesh_I3_prim_only.png")

        Returns:
            Thumbnail filename (e.g., "3bcac4bf_mesh_I3_prim_only.png")
        """
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
        """Scan for new rendered images and create thumbnails (service-specific).

        Only runs for rendering steps, scans the renders directory for new PNGs,
        creates 128x128 thumbnails in preview/ directory.

        Args:
            step_name: Current step name

        Returns:
            List of thumbnail URLs for newly created thumbnails
        """
        if not self.session_dir or step_name != "build_dataset_usd":
            return []

        try:
            renders_dir = self.session_dir / "cache" / "dataset" / "usd" / "renders"
            preview_dir = self.session_dir / "cache" / "preview"

            if not renders_dir.exists():
                return []

            preview_dir.mkdir(parents=True, exist_ok=True)

            # Find all PNG files
            all_pngs = list(renders_dir.rglob("*.png"))
            new_thumbnails = []

            for img_path in all_pngs:
                relative_path = str(img_path.relative_to(renders_dir))
                unique_filename = resolve_preview_filename(preview_dir, relative_path)

                # Skip if already thumbnailed
                if unique_filename in self.thumbnailed_images:
                    continue

                # Create thumbnail
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
                    new_thumbnails.append(
                        f"/assets/{self.session_id}/preview/{unique_filename}"
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to create thumbnail for {img_path.name}: {e}"
                    )

            return new_thumbnails

        except Exception as e:
            logger.warning(f"Failed to scan for thumbnails: {e}")
            return []
