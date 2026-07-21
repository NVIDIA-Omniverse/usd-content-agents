# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLM inference task for material assignment."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from filelock import FileLock, Timeout
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore
from world_understanding.utils.token_tracking import TokenTracker, format_token_stats

from material_agent.functions.inference import (
    assign_materials_multi_prim,
    async_batch_assign_materials,
    batch_assign_materials,
    reconcile_untrusted_spec_evidence,
)
from material_agent.tasks.prepare_dataset import (
    _VLM_MULTI_PRIM_SYSTEM_PROMPT_TEMPLATE,
    _VLM_MULTI_PRIM_USER_PROMPT_TEMPLATE,
    render_vlm_multi_prim_user_prompt_template,
    render_vlm_system_prompt_template,
)

logger = logging.getLogger(__name__)
_TOKEN_USAGE_ARTIFACT_LOCK_TIMEOUT_SECONDS = 30
_TOKEN_USAGE_ARTIFACT_LOCK = Lock()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _merge_count_buckets(
    existing: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    merged = {
        key: dict(value) for key, value in existing.items() if isinstance(value, dict)
    }
    for key, value in current.items():
        if not isinstance(value, dict):
            continue
        bucket = merged.setdefault(key, {})
        bucket["input_tokens"] = _as_int(bucket.get("input_tokens")) + _as_int(
            value.get("input_tokens")
        )
        bucket["output_tokens"] = _as_int(bucket.get("output_tokens")) + _as_int(
            value.get("output_tokens")
        )
        bucket["total_tokens"] = _as_int(bucket.get("total_tokens")) + _as_int(
            value.get("total_tokens")
        )
        bucket["count"] = _as_int(bucket.get("count")) + _as_int(value.get("count"))
    return merged


def _merge_token_usage_stats(
    existing: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Merge previous full-run token stats with usage from a resumed run."""
    existing_usages = existing.get("all_usages")
    current_usages = current.get("all_usages")
    return {
        "total_input_tokens": _as_int(existing.get("total_input_tokens"))
        + _as_int(current.get("total_input_tokens")),
        "total_output_tokens": _as_int(existing.get("total_output_tokens"))
        + _as_int(current.get("total_output_tokens")),
        "total_tokens": _as_int(existing.get("total_tokens"))
        + _as_int(current.get("total_tokens")),
        "invocation_count": _as_int(existing.get("invocation_count"))
        + _as_int(current.get("invocation_count")),
        "by_model": _merge_count_buckets(
            existing.get("by_model")
            if isinstance(existing.get("by_model"), dict)
            else {},
            current.get("by_model")
            if isinstance(current.get("by_model"), dict)
            else {},
        ),
        "by_type": _merge_count_buckets(
            existing.get("by_type")
            if isinstance(existing.get("by_type"), dict)
            else {},
            current.get("by_type") if isinstance(current.get("by_type"), dict) else {},
        ),
        "all_usages": [
            *(existing_usages if isinstance(existing_usages, list) else []),
            *(current_usages if isinstance(current_usages, list) else []),
        ],
    }


def _write_token_usage_artifact(
    predictions_path: Path,
    token_stats: dict[str, Any],
    *,
    merge_existing: bool = False,
) -> Path | None:
    """Persist token usage beside predictions for scene-level aggregation."""
    token_path = predictions_path.parent / "token_usage.json"
    lock_path = token_path.with_name(f".{token_path.name}.lock")

    try:
        token_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "Failed to prepare token usage artifact directory %s: %s",
            token_path.parent,
            exc,
        )
        return None

    try:
        with _TOKEN_USAGE_ARTIFACT_LOCK:
            with FileLock(
                str(lock_path), timeout=_TOKEN_USAGE_ARTIFACT_LOCK_TIMEOUT_SECONDS
            ):
                stats_to_write = token_stats
                if merge_existing and token_path.exists():
                    try:
                        existing_payload = json.loads(
                            token_path.read_text(encoding="utf-8")
                        )
                        existing_stats = existing_payload.get("token_usage")
                        if isinstance(existing_stats, dict):
                            stats_to_write = _merge_token_usage_stats(
                                existing_stats,
                                token_stats,
                            )
                    except (OSError, json.JSONDecodeError) as exc:
                        logger.warning(
                            "Failed to merge existing token usage artifact %s: %s",
                            token_path,
                            exc,
                        )

                payload = {
                    "schema_version": "1.0.0",
                    "generated_at": datetime.now(UTC).isoformat(),
                    "scope": "asset_predict",
                    "predictions_path": str(predictions_path),
                    "token_usage": stats_to_write,
                }
                tmp_path = token_path.with_name(f".{token_path.name}.tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                    f.write("\n")
                tmp_path.replace(token_path)
                return token_path
    except Timeout as exc:
        logger.warning(
            "Timed out acquiring token usage artifact lock %s: %s",
            lock_path,
            exc,
        )
        return None
    except OSError as exc:
        logger.warning(
            "Failed to write token usage artifact %s: %s",
            token_path,
            exc,
        )
        return None


class VLMInferenceTask(Task):
    """Run VLM inference on dataset for material assignment.

    This task supports iterative refinement by incorporating feedback from
    previous iterations. If a previous_judge_critique is present in the context,
    it will be automatically appended to the system prompt to guide improvements.

    Input context keys:
        - dataset or dataset_path: Dataset to process
        - vlm: VLM instance
        - llm: LLM instance (optional, uses VLM if not provided)
        - vlm_config: VLM configuration
        - system_prompt: Base system prompt
        - previous_judge_critique: Critique from previous iteration (optional)
        - iteration_count: Current iteration number (for logging)

    Output context keys:
        - predictions: List of prediction results
        - predictions_path: Path to saved predictions file
    """

    def __init__(
        self,
        vlm: Any = None,
        llm: Any | None = None,
        system_prompt: str | None = None,
    ):
        """Initialize the VLM inference task.

        Args:
            vlm: VLM instance for material assignment (None to use from context)
            llm: Optional LLM for parsing (None to use from context or VLM)
            system_prompt: Optional custom system prompt (None to use from context)
        """
        self.vlm = vlm
        self.llm = llm
        self.system_prompt = system_prompt
        self.name = "VLMInference"
        self.description = "Run VLM inference for material assignment"

    @staticmethod
    def _classify_entries_for_selective_reprediction(
        dataset: list[dict[str, Any]],
        resolved_assignments: dict[str, str],
        prim_feedback: dict[str, str],
        prev_preds: dict[str, dict[str, Any]],
        visual_refinement_context_by_prim: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        """Classify dataset entries for selective re-prediction.

        Separates entries into those that can be carried forward unchanged,
        those that need VLM re-prediction, and those resolved by the analyzer.

        Args:
            dataset: Full list of dataset entries.
            resolved_assignments: Mapping of entry_id -> resolved material
                from the analyzer (deterministic fixes).
            prim_feedback: Mapping of entry_id -> feedback text for entries
                that need re-prediction with guidance.
            prev_preds: Mapping of entry_id -> previous prediction dict.

        Returns:
            Tuple of:
                - carried_forward_predictions: Predictions to keep as-is
                  (includes both resolved and unchanged entries).
                - re_predict_entries: Entries that need VLM re-prediction.
                - resolved_count: Number of entries resolved by the analyzer.
        """
        carried_forward_predictions: list[dict[str, Any]] = []
        re_predict_entries: list[dict[str, Any]] = []
        resolved_count = 0

        for entry in dataset:
            entry_id = entry.get("id", "")
            if entry_id in resolved_assignments:
                # Deterministic fix from analyzer — apply directly
                resolved_mat = resolved_assignments[entry_id]
                resolved_pred = dict(prev_preds.get(entry_id, {}))
                resolved_pred["id"] = entry_id
                resolved_pred["materials"] = {"material": resolved_mat}
                if "images" in entry:
                    resolved_pred["images"] = entry["images"]
                carried_forward_predictions.append(resolved_pred)
                resolved_count += 1
            elif entry_id in prim_feedback:
                # Has feedback but no resolution — re-predict with VLM
                feedback_text = prim_feedback[entry_id]
                visual_context_raw = (
                    visual_refinement_context_by_prim.get(entry_id, {})
                    if visual_refinement_context_by_prim
                    else {}
                )
                visual_context = (
                    visual_context_raw if isinstance(visual_context_raw, dict) else {}
                )
                visual_text = str(visual_context.get("text") or "").strip()
                visual_section = (
                    f"\n\n**TARGETED VISUAL REFINEMENT EVIDENCE:**\n{visual_text}\n"
                    if visual_text
                    else ""
                )
                entry["text"] = (
                    f"\n**FEEDBACK FOR THIS SPECIFIC PART "
                    f"(from previous iteration):**\n"
                    f"{feedback_text}"
                    f"{visual_section}\n\n"
                    f"{entry.get('text', '')}"
                )
                VLMInferenceTask._attach_visual_refinement_images(
                    entry,
                    visual_context,
                )
                re_predict_entries.append(entry)
            elif entry_id in prev_preds:
                # Good prediction — carry forward unchanged
                carried_forward_predictions.append(prev_preds[entry_id])
            else:
                # New or missing entry — must re-predict
                re_predict_entries.append(entry)

        return carried_forward_predictions, re_predict_entries, resolved_count

    @staticmethod
    def _attach_visual_refinement_images(
        entry: dict[str, Any],
        visual_context: dict[str, Any],
    ) -> None:
        """Attach targeted full-render crops to one reprediction entry."""
        raw_images = visual_context.get("images")
        if not isinstance(raw_images, list) or not raw_images:
            return

        media = entry.setdefault("media", {})
        if not isinstance(media, dict):
            media = {}
            entry["media"] = media
        images = media.setdefault("images", [])
        if not isinstance(images, list):
            images = []
            media["images"] = images

        existing_paths = {
            str(image.get("path"))
            for image in images
            if isinstance(image, dict) and image.get("path") is not None
        }
        for raw_image in raw_images:
            if not isinstance(raw_image, dict):
                continue
            raw_path = raw_image.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            if raw_path in existing_paths:
                continue
            caption = str(raw_image.get("caption") or "").strip()
            images.append(
                {
                    "path": raw_path,
                    "type": "render",
                    "metadata": {
                        "render_mode": "visual_refinement",
                        "view": "targeted_full_scene_visual_evidence",
                        "camera": "full_scene",
                        "vlm_prompt": caption
                        or (
                            "Targeted full-scene visual evidence for this "
                            "follow-up material prediction."
                        ),
                    },
                }
            )
            existing_paths.add(raw_path)

    # ------------------------------------------------------------------
    # Multi-prim grouping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_entries(
        entries: list[dict[str, Any]], batch_size: int
    ) -> list[list[dict[str, Any]]]:
        """Split dataset entries into groups of at most *batch_size*.

        Args:
            entries: Flat list of dataset entries.
            batch_size: Maximum entries per group (must be >= 2).

        Returns:
            List of groups, each a list of entries.
        """
        return [entries[i : i + batch_size] for i in range(0, len(entries), batch_size)]

    @staticmethod
    def _build_multi_prim_images_and_prompt(
        group: list[dict[str, Any]],
        image_base_dir: Path,
    ) -> tuple[list[str | Path], list[str], list[str], str]:
        """Merge images and build the multi-prim user prompt for one group.

        Reference images (type == "reference") are included once at the start.
        Per-prim render images follow, labelled by prim.

        Args:
            group: List of dataset entries in this group.
            image_base_dir: Base directory for resolving relative image paths.

        Returns:
            Tuple of:
                - merged_images: Ordered list of image paths/objects
                - merged_image_prompts: Parallel list of per-image captions
                - prim_ids: Ordered prim IDs in this group
                - user_prompt: Formatted multi-prim user prompt text
        """
        from world_understanding.functions.classification.inference import (
            _extract_image_metadata_from_entry,
            _extract_images_from_entry,
            _extract_text_from_entry,
        )

        merged_images: list[str | Path] = []
        merged_image_prompts: list[str] = []
        prim_ids: list[str] = []

        # --- Collect reference images from the first entry (shared) ---
        first_entry = group[0]
        first_images = _extract_images_from_entry(first_entry)
        first_meta = _extract_image_metadata_from_entry(first_entry)

        ref_count = 0
        for img, meta in zip(first_images, first_meta, strict=False):
            if meta.get("render_mode") == "reference_image":
                # Resolve path
                if isinstance(img, str) and not Path(img).is_absolute():
                    img = image_base_dir / img
                merged_images.append(img)
                merged_image_prompts.append(meta.get("vlm_prompt", "Reference image"))
                ref_count += 1

        # --- Per-prim images ---
        image_layout_lines: list[str] = []
        per_part_context_lines: list[str] = []
        running_idx = ref_count  # image index counter

        if ref_count > 0:
            if ref_count == 1:
                image_layout_lines.append("- Image [0]: Reference image (shared)")
            else:
                image_layout_lines.append(
                    f"- Images [0–{ref_count - 1}]: Reference images (shared)"
                )

        for entry in group:
            entry_id = entry.get("id", "unknown")
            prim_ids.append(entry_id)

            images = _extract_images_from_entry(entry)
            metadata = _extract_image_metadata_from_entry(entry)

            # Collect only render images (skip references — already added)
            prim_start_idx = running_idx
            for img, meta in zip(images, metadata, strict=False):
                if meta.get("render_mode") == "reference_image":
                    continue  # skip — already included once
                if isinstance(img, str) and not Path(img).is_absolute():
                    img = image_base_dir / img
                merged_images.append(img)
                caption = meta.get("vlm_prompt", "Rendered view")
                merged_image_prompts.append(f"[Part: {entry_id}] {caption}")
                running_idx += 1

            prim_end_idx = running_idx - 1
            if prim_start_idx <= prim_end_idx:
                if prim_start_idx == prim_end_idx:
                    image_layout_lines.append(
                        f'- Image [{prim_start_idx}]: Part "{entry_id}"'
                    )
                else:
                    image_layout_lines.append(
                        f'- Images [{prim_start_idx}–{prim_end_idx}]: Part "{entry_id}"'
                    )

            # Per-part context from user_prompt / text
            part_text = _extract_text_from_entry(entry)
            per_part_context_lines.append(f"### Part: {entry_id}\n{part_text}")

        image_layout = "\n".join(image_layout_lines)
        per_part_context = "\n\n".join(per_part_context_lines)

        user_prompt = render_vlm_multi_prim_user_prompt_template(
            _VLM_MULTI_PRIM_USER_PROMPT_TEMPLATE,
            image_layout=image_layout,
            per_part_context=per_part_context,
        )

        return merged_images, merged_image_prompts, prim_ids, user_prompt

    @staticmethod
    def _extract_materials_list_from_system_prompt(system_prompt: str) -> str:
        """Extract the materials list section from a formatted system prompt.

        The single-prim system prompt contains a section like:
            Available materials:
            <materials list>

            Please answer ...

        We extract the materials list text so we can re-use it in the
        multi-prim system prompt template.

        Args:
            system_prompt: Formatted single-prim system prompt.

        Returns:
            Materials list string, or empty string if not found.
        """
        import re as _re

        # Look for "Available materials:\n<content>" up to the next double newline
        match = _re.search(
            r"Available materials:\s*\n(.*?)(?:\n\n|$)",
            system_prompt,
            _re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _allow_empty_predictions(
        context: dict[str, Any],
        field_name: str = "inference.allow_empty_predictions",
    ) -> bool:
        allow_empty_predictions = context.get("allow_empty_predictions", False)
        if not isinstance(allow_empty_predictions, bool):
            raise ValueError(
                f"{field_name} must be a boolean, got "
                f"{type(allow_empty_predictions).__name__}"
            )
        return allow_empty_predictions

    @staticmethod
    def _fail_if_predictions_empty(
        dataset_count: int,
        predictions: list[dict[str, Any]],
        failed: list[dict[str, Any]],
        predictions_path: Path,
        allow_empty_predictions: bool,
    ) -> None:
        if dataset_count == 0 or predictions or allow_empty_predictions:
            return
        raise RuntimeError(
            "VLM inference produced zero successful material predictions for "
            f"{dataset_count} dataset entr{'y' if dataset_count == 1 else 'ies'} "
            f"({len(failed)} failed). Refusing to continue; set "
            "allow_empty_predictions=true only for workflows that intentionally "
            f"permit empty prediction output. predictions_path={predictions_path}"
        )

    @staticmethod
    def _carried_forward_predictions_as_results(
        carried_forward_predictions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert carried-forward prediction records into inference results."""
        results: list[dict[str, Any]] = []
        for pred in carried_forward_predictions:
            pred_id = pred.get("id")
            materials = pred.get("materials")
            if not pred_id or not materials:
                logger.warning(
                    "Skipping carried-forward material prediction without "
                    "required id/materials fields"
                )
                continue
            results.append(
                {
                    "id": pred_id,
                    "vlm_response": materials,
                    "status": "success",
                }
            )
        return results

    def _run_multi_prim_inference(
        self,
        dataset: list[dict[str, Any]],
        context: dict[str, Any],
        prediction_batch_size: int,
        vlm: Any,
        llm: Any,
        system_prompt: str | None,
        vlm_invoke_kwargs: dict[str, Any],
        max_retries: int,
        predictions_path: Path,
        stream_predictions: bool,
        listener: Any,
        token_tracker: Any,
    ) -> list[dict[str, Any]]:
        """Run inference with multi-prim grouping.

        Groups entries, calls VLM once per group, splits results back into
        individual prediction records. Entries that fail in a group are
        re-queued as individual (batch_size=1) calls.

        Returns:
            List of result dicts (same schema as batch_assign_materials).
        """
        image_base_dir = Path(context["image_base_dir"])

        # Build multi-prim system prompt from the materials list
        materials_list = ""
        if system_prompt:
            materials_list = self._extract_materials_list_from_system_prompt(
                system_prompt
            )
        if not materials_list:
            # Fallback: try to get from config
            materials_list = context.get("config", {}).get(
                "_materials_formatted", ""
            ) or context.get("config", {}).get("materials_list", "")
            if isinstance(materials_list, list):
                materials_list = ", ".join(materials_list)

        multi_prim_system_prompt = render_vlm_system_prompt_template(
            _VLM_MULTI_PRIM_SYSTEM_PROMPT_TEMPLATE,
            materials_list=materials_list,
        )
        listener.debug(
            f"Multi-prim system prompt built with "
            f"{len(materials_list)} chars of materials list"
        )
        groups = self._group_entries(dataset, prediction_batch_size)
        raw_max_workers = context.get("max_workers")
        try:
            max_workers = max(1, int(raw_max_workers)) if raw_max_workers else 1
        except (TypeError, ValueError):
            max_workers = 1

        effective_workers = min(max_workers, max(1, len(groups)))
        listener.info(
            f"Multi-prim mode: {len(dataset)} entries → {len(groups)} groups "
            f"(batch_size={prediction_batch_size}, workers={effective_workers})"
        )

        all_results: list[dict[str, Any]] = []
        retry_entries: list[dict[str, Any]] = []  # entries to retry individually
        processed_count = 0
        progress_lock = Lock()
        stream_lock = Lock()

        def process_group(
            group_idx: int, group: list[dict[str, Any]]
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            nonlocal processed_count
            group_results: list[dict[str, Any]] = []
            group_retry_entries: list[dict[str, Any]] = []

            prim_ids_in_group = [e.get("id", "unknown") for e in group]
            listener.info(
                f"Processing group {group_idx}/{len(groups)} "
                f"({len(group)} prims: {prim_ids_in_group})"
            )

            try:
                (
                    merged_images,
                    merged_image_prompts,
                    prim_ids,
                    user_prompt,
                ) = self._build_multi_prim_images_and_prompt(group, image_base_dir)

                # Resolve image paths (ensure they exist)
                resolved_images = []
                for img in merged_images:
                    if isinstance(img, str | Path):
                        p = Path(img)
                        if not p.exists():
                            logger.warning(f"Image not found: {p}")
                        resolved_images.append(p)
                    else:
                        resolved_images.append(img)

                multi_results = assign_materials_multi_prim(
                    vlm=vlm,
                    prim_ids=prim_ids,
                    text=user_prompt,
                    images=resolved_images,
                    llm=llm,
                    system_prompt=multi_prim_system_prompt,
                    invoke_kwargs=vlm_invoke_kwargs,
                    image_prompts=merged_image_prompts
                    if merged_image_prompts
                    else None,
                    max_retries=max_retries,
                    token_tracker=token_tracker,
                )

                # Split results back into individual records
                for entry in group:
                    entry_id = entry.get("id", "unknown")

                    if entry_id in multi_results:
                        pred = reconcile_untrusted_spec_evidence(
                            multi_results[entry_id],
                            entry,
                        )
                        result = {
                            "id": entry_id,
                            "vlm_response": pred,
                            "status": "success",
                        }
                        group_results.append(result)

                        # Stream to file
                        if stream_predictions:
                            try:
                                from world_understanding.functions.classification.inference import (
                                    _extract_images_from_entry,
                                )

                                output_entry: dict[str, Any] = {
                                    "id": entry_id,
                                    "materials": pred,
                                }
                                images = _extract_images_from_entry(entry)
                                if images:
                                    output_entry["images"] = images
                                with stream_lock:
                                    with open(
                                        predictions_path, "a", encoding="utf-8"
                                    ) as f:
                                        f.write(json.dumps(output_entry) + "\n")
                            except Exception as e:
                                logger.warning(
                                    f"Failed to stream prediction for {entry_id}: {e}"
                                )

                        with progress_lock:
                            processed_count += 1
                            current = processed_count
                        # Progress event
                        listener.event(
                            "task.progress",
                            {
                                "task_name": "VLMInference",
                                "current": current,
                                "total": len(dataset),
                                "percentage": (current / len(dataset)) * 100,
                                "entry_id": entry_id,
                            },
                        )
                        event_payload: dict[str, Any] = {
                            "entry_id": entry_id,
                            "material": pred.get("material", "unknown"),
                            "confidence": pred.get("confidence"),
                            "response_snippet": pred.get("original_response", ""),
                        }
                        reconciliation = pred.get("evidence_reconciliation")
                        if isinstance(reconciliation, dict):
                            event_payload["evidence_reconciliation"] = reconciliation
                        listener.event("prediction.completed", event_payload)
                    else:
                        # Prim missing from response → queue for individual retry
                        logger.warning(
                            f"Prim {entry_id} missing from multi-prim response, "
                            f"will retry individually"
                        )
                        with progress_lock:
                            processed_count += 1
                        group_retry_entries.append(entry)

                listener.info(
                    f"Group {group_idx} done: "
                    f"{len(multi_results)}/{len(group)} prims parsed"
                )

            except Exception as e:
                logger.error(f"Group {group_idx} failed entirely: {e}", exc_info=True)
                listener.error(f"Group {group_idx} failed: {e}")
                # Queue all entries in this group for individual retry
                group_retry_entries.extend(group)
                with progress_lock:
                    processed_count += len(group)

            return group_results, group_retry_entries

        if effective_workers > 1 and len(groups) > 1:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                future_to_group = {
                    executor.submit(process_group, group_idx, group): (
                        group_idx,
                        group,
                    )
                    for group_idx, group in enumerate(groups, 1)
                }
                for future in as_completed(future_to_group):
                    group_idx, group = future_to_group[future]
                    try:
                        group_results, group_retry_entries = future.result()
                        all_results.extend(group_results)
                        retry_entries.extend(group_retry_entries)
                    except Exception as e:
                        logger.error(
                            f"Unexpected error in group {group_idx}: {e}",
                            exc_info=True,
                        )
                        listener.error(f"Group {group_idx} failed: {e}")
                        retry_entries.extend(group)
                        with progress_lock:
                            processed_count += len(group)
        else:
            for group_idx, group in enumerate(groups, 1):
                group_results, group_retry_entries = process_group(group_idx, group)
                all_results.extend(group_results)
                retry_entries.extend(group_retry_entries)

        # --- Retry failed entries individually (batch_size=1) ---
        if retry_entries:
            deduped_retry_entries = list(
                {
                    entry.get("id", f"unknown_{idx}"): entry
                    for idx, entry in enumerate(retry_entries)
                }.values()
            )
            if len(deduped_retry_entries) != len(retry_entries):
                listener.info(
                    f"Deduplicated retry queue: {len(retry_entries)} -> "
                    f"{len(deduped_retry_entries)} entries"
                )
            listener.info(
                f"Retrying {len(deduped_retry_entries)} entries individually "
                f"(batch_size=1)"
            )

            def on_retry_result(result: dict[str, Any], entry: dict[str, Any]) -> None:
                if not stream_predictions:
                    return
                try:
                    if result.get("status") != "success":
                        return
                    from world_understanding.functions.classification.inference import (
                        _extract_images_from_entry,
                    )

                    output_entry: dict[str, Any] = {
                        "id": result["id"],
                        "materials": result.get("vlm_response"),
                    }
                    images = _extract_images_from_entry(entry)
                    if images:
                        output_entry["images"] = images
                    with stream_lock:
                        with open(predictions_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(output_entry) + "\n")
                except Exception as e:
                    logger.warning(f"Failed to stream retry prediction: {e}")

            retry_results = batch_assign_materials(
                vlm=vlm,
                entries=deduped_retry_entries,
                llm=llm,
                image_base_dir=image_base_dir,
                system_prompt=context.get("config", {}).get("system_prompt"),
                invoke_kwargs=vlm_invoke_kwargs,
                on_result=on_retry_result,
                max_workers=context.get("max_workers"),
                max_retries=max_retries,
                token_tracker=token_tracker,
            )
            all_results.extend(retry_results)

        return all_results

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Run batch inference on dataset.

        Args:
            context: Workflow context with dataset metadata
            object_store: Storage for dataset and predictions

        Returns:
            Updated context with inference results
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Resolve parameters from constructor or context
        vlm = self.vlm if self.vlm is not None else context.get("vlm")
        llm = self.llm if self.llm is not None else context.get("llm")
        if llm is None:
            llm = vlm  # Use VLM as LLM if not provided

        # Get config values from context if not provided in constructor
        vlm_config = context.get("vlm_config", {})
        max_retries = vlm_config.get("max_retries", 3)  # Default to 3 retries
        allow_empty_predictions = self._allow_empty_predictions(context)
        system_prompt = (
            self.system_prompt
            if self.system_prompt is not None
            else context.get("config", {}).get("system_prompt")
        )

        # Include previous judge critique in system prompt (for iterative refinement)
        previous_critique = context.get("previous_judge_critique")
        if previous_critique:
            iteration_count = context.get("iteration_count", 1)
            listener.info(
                f"Including previous judge critique in system prompt "
                f"(iteration {iteration_count})"
            )
            critique_section = (
                "\n\n**FEEDBACK FROM PREVIOUS ITERATION:**\n"
                "In the previous iteration, your material assignments were evaluated. "
                "Here is the critique:\n\n"
                f"{previous_critique}\n\n"
                "**IMPORTANT — BE CONSERVATIVE:**\n"
                "- Only change parts that the critique specifically identifies as wrong.\n"
                "- Keep ALL other assignments exactly as they were.\n"
                "- Do NOT change more than 3-4 parts based on this feedback.\n"
                "- If the critique suggests a wholesale color change, ignore it — "
                "only fix specific inconsistencies or clear errors.\n"
                "- Prefer your previous answer unless the feedback gives a clear, "
                "specific reason to change a particular part."
            )
            if system_prompt:
                system_prompt = system_prompt + critique_section
            else:
                system_prompt = critique_section

        # Save the actual system prompt used (with critique) for reporting
        # Use a separate key to avoid mutating the base config.system_prompt
        context["actual_system_prompt_used"] = system_prompt
        listener.debug("Saved actual system prompt (with feedback) to context")

        # Build per-invoke kwargs from provisioning (if available). Use as-is.
        vlm_invoke_kwargs: dict[str, Any] = dict(context.get("vlm_invoke_kwargs", {}))

        # Validate required values
        if vlm is None:
            raise ValueError("VLM not provided in constructor or context")

        # Get dataset from object store or context
        if object_store and object_store.exists("dataset"):
            dataset = object_store.get("dataset")
        else:
            # Fallback: load from file if not in store
            dataset_path = Path(context["dataset_path"])
            with open(dataset_path, encoding="utf-8") as f:
                dataset = [json.loads(line) for line in f]
        input_dataset_count = len(dataset)

        # Selective re-prediction with resolved assignments:
        # 1. Prims with resolved_assignments -> apply directly (no VLM call)
        # 2. Prims without issues -> carry forward from previous iteration
        # 3. Remaining prims -> re-predict with VLM (only if no resolution)
        resolved_assignments = context.get("resolved_assignments", {})
        prim_feedback = context.get("previous_prim_feedback", {})
        visual_refinement_context_by_prim = context.get(
            "visual_refinement_context_by_prim", {}
        )
        if not isinstance(visual_refinement_context_by_prim, dict):
            visual_refinement_context_by_prim = {}
        previous_predictions_path = context.get("previous_predictions_path")
        carried_forward_predictions: list[dict[str, Any]] = []

        if (resolved_assignments or prim_feedback) and previous_predictions_path:
            prev_path = Path(previous_predictions_path)
            if prev_path.exists():
                # Load previous predictions
                prev_preds: dict[str, dict[str, Any]] = {}
                with open(prev_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            pred = json.loads(line)
                            prev_preds[pred.get("id", "")] = pred

                carried_forward_predictions, re_predict_entries, resolved_count = (
                    self._classify_entries_for_selective_reprediction(
                        dataset,
                        resolved_assignments,
                        prim_feedback,
                        prev_preds,
                        visual_refinement_context_by_prim,
                    )
                )

                listener.info(
                    f"Selective re-prediction: "
                    f"{resolved_count} resolved by analyzer, "
                    f"{len(re_predict_entries)} to re-predict with VLM, "
                    f"{len(carried_forward_predictions) - resolved_count} "
                    f"carried forward (unchanged)"
                )
                dataset = re_predict_entries
            else:
                listener.warning(
                    f"Previous predictions not found at {prev_path}, "
                    f"re-predicting all entries"
                )
        elif prim_feedback:
            # Fallback: feedback exists but no previous predictions path
            feedback_count = 0
            for entry in dataset:
                entry_id = entry.get("id", "")
                if entry_id in prim_feedback:
                    feedback_text = prim_feedback[entry_id]
                    visual_context_raw = (
                        visual_refinement_context_by_prim.get(entry_id, {})
                        if visual_refinement_context_by_prim
                        else {}
                    )
                    visual_context = (
                        visual_context_raw
                        if isinstance(visual_context_raw, dict)
                        else {}
                    )
                    visual_text = str(visual_context.get("text") or "").strip()
                    visual_section = (
                        f"\n\n**TARGETED VISUAL REFINEMENT EVIDENCE:**\n{visual_text}\n"
                        if visual_text
                        else ""
                    )
                    entry["text"] = (
                        f"\n**FEEDBACK FOR THIS SPECIFIC PART "
                        f"(from previous iteration):**\n"
                        f"{feedback_text}"
                        f"{visual_section}\n\n"
                        f"{entry.get('text', '')}"
                    )
                    VLMInferenceTask._attach_visual_refinement_images(
                        entry,
                        visual_context,
                    )
                    feedback_count += 1
            if feedback_count:
                listener.info(
                    f"Injected per-prim feedback for {feedback_count}/{len(dataset)} "
                    f"entries (no previous predictions to carry forward)"
                )

        # Emit task started event
        listener.event(
            "task.started",
            {
                "task_name": "VLMInference",
                "total_entries": len(dataset),
            },
        )
        listener.info(f"Starting VLM inference for {len(dataset)} entries")

        # Progress callback
        processed_count = [0]  # Use list to allow modification in closure

        def on_progress(entry_id: str, response: str) -> None:
            """Log progress after processing each entry."""
            processed_count[0] += 1

            # Emit progress event
            listener.event(
                "task.progress",
                {
                    "task_name": "VLMInference",
                    "current": processed_count[0],
                    "total": len(dataset),
                    "percentage": (processed_count[0] / len(dataset)) * 100,
                    "entry_id": entry_id,
                },
            )

            if processed_count[0] % 10 == 0:
                listener.debug(f"Processed {processed_count[0]}/{len(dataset)} entries")
            logger.debug(f"Completed processing entry: {entry_id}")

        # Error callback
        def on_error(entry_id: str, error: str) -> None:
            """Handle errors during processing."""
            logger.error(f"Error processing {entry_id}: {error}")
            listener.error(f"Error processing {entry_id}: {error}")

        # Prediction callback
        def on_prediction(entry_id: str, material_dict: dict[str, Any]) -> None:
            """Emit event for each material prediction."""
            try:
                # Extract material and confidence from response
                material = material_dict.get("material", "unknown")
                confidence = material_dict.get("confidence")
                response_snippet = material_dict.get("original_response", "")

                # Emit detailed prediction event
                event_payload = {
                    "entry_id": entry_id,
                    "material": material,
                    "confidence": confidence,
                    "response_snippet": response_snippet,
                }
                reconciliation = material_dict.get("evidence_reconciliation")
                if isinstance(reconciliation, dict):
                    event_payload["evidence_reconciliation"] = reconciliation
                listener.event("prediction.completed", event_payload)
            except Exception as e:
                logger.warning(f"Failed to emit prediction event for {entry_id}: {e}")

        # Streaming and resume options
        stream_predictions = context.get("stream_predictions", True)
        resume_enabled = context.get("resume", False)

        # Resolve predictions path
        # Check if predictions_path is already set (e.g., by IterationTask)
        predictions_path = context.get("predictions_path")

        if predictions_path:
            # Use the provided path (e.g., from IterationTask)
            predictions_path = Path(predictions_path)
            output_dir = predictions_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using predictions path from context: {predictions_path}")
        else:
            # Compute from output_dir or dataset_path (standard behavior)
            output_dir = context.get("output_dir")
            if output_dir is None:
                dataset_path_str = context.get("dataset_path")
                if not dataset_path_str:
                    raise ValueError(
                        "dataset_path not found in context and output_dir not specified"
                    )
                dataset_path = Path(dataset_path_str).parent / "output"
                output_dir = dataset_path
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = output_dir / "predictions.jsonl"

        # If not resuming and streaming, clear any existing predictions file to avoid duplicates
        if stream_predictions and not resume_enabled and predictions_path.exists():
            try:
                predictions_path.unlink()
                logger.info(
                    "Cleared existing predictions file since resume is disabled: %s",
                    predictions_path,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to remove existing predictions file {predictions_path}: {e}"
                )

        # Write carried-forward predictions (from selective re-prediction)
        if carried_forward_predictions and stream_predictions:
            try:
                written_count = 0
                with open(predictions_path, "a", encoding="utf-8") as f:
                    for pred in carried_forward_predictions:
                        # Validate required fields before writing
                        if not pred.get("id") or not pred.get("materials"):
                            logger.warning(
                                f"Skipping invalid carried-forward prediction: "
                                f"{pred.get('id', 'unknown')}"
                            )
                            continue
                        f.write(json.dumps(pred) + "\n")
                        written_count += 1
                logger.info(
                    "Wrote %d of %d carried-forward predictions to %s",
                    written_count,
                    len(carried_forward_predictions),
                    predictions_path,
                )
            except Exception as e:
                logger.warning(f"Failed to write carried-forward predictions: {e}")

        # Load processed ids from existing predictions.jsonl when resuming
        processed_ids: set[str] = set()
        if stream_predictions and resume_enabled and predictions_path.exists():
            try:
                with open(predictions_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            if rec and "id" in rec:
                                processed_ids.add(rec["id"])
                        except Exception:
                            # Ignore malformed trailing lines
                            continue
                logger.info(
                    "Resuming: %d entries already in predictions.jsonl",
                    len(processed_ids),
                )
            except Exception as e:
                logger.warning(f"Failed to parse existing predictions.jsonl: {e}")

        # Define result callback to stream successes to predictions.jsonl
        def on_result(result: dict[str, Any], entry: dict[str, Any]) -> None:
            if not stream_predictions:
                return
            try:
                if result.get("status") != "success":
                    return
                output_entry: dict[str, Any] = {
                    "id": result["id"],
                    "materials": result.get("vlm_response"),
                }
                if "images" in entry:
                    output_entry["images"] = entry["images"]
                elif "image_path" in entry:
                    output_entry["image_path"] = entry.get("image_path", "")

                vr = result.get("vlm_response")
                if isinstance(vr, dict) and "confidence" in vr:
                    output_entry["confidence"] = vr["confidence"]

                with open(predictions_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_entry) + "\n")
            except Exception as e:
                logger.warning(f"Failed to append streaming prediction: {e}")

        # Get optional max_workers for parallel processing
        max_workers = context.get("max_workers")
        if max_workers:
            logger.info(f"Using parallel processing with {max_workers} workers")

        # Create token tracker to collect usage statistics
        token_tracker = TokenTracker()
        logger.info("Token usage tracking enabled")

        # Check prediction_batch_size for multi-prim mode
        prediction_batch_size = context.get("prediction_batch_size", 1)

        if prediction_batch_size > 1 and len(dataset) > 0:
            # --- Multi-prim path ---
            listener.info(
                f"Using multi-prim inference (prediction_batch_size="
                f"{prediction_batch_size})"
            )
            results = self._run_multi_prim_inference(
                dataset=dataset,
                context=context,
                prediction_batch_size=prediction_batch_size,
                vlm=vlm,
                llm=llm,
                system_prompt=system_prompt,
                vlm_invoke_kwargs=vlm_invoke_kwargs,
                max_retries=max_retries,
                predictions_path=predictions_path,
                stream_predictions=stream_predictions,
                listener=listener,
                token_tracker=token_tracker,
            )
        else:
            # --- Standard single-prim path (batch_size=1, unchanged) ---
            # Run inference using core function with streaming and resume support
            results = batch_assign_materials(
                vlm=vlm,
                entries=dataset,
                llm=llm,
                image_base_dir=Path(context["image_base_dir"]),
                system_prompt=system_prompt,
                invoke_kwargs=vlm_invoke_kwargs,
                on_progress=on_progress,
                on_error=on_error,
                processed_ids=processed_ids,
                on_result=on_result,
                on_prediction=on_prediction,
                max_workers=max_workers,
                max_retries=max_retries,
                token_tracker=token_tracker,
            )

        # Get and log token usage statistics
        token_stats = token_tracker.get_stats()
        _write_token_usage_artifact(
            predictions_path,
            token_stats,
            merge_existing=stream_predictions
            and resume_enabled
            and bool(processed_ids),
        )
        logger.info(f"\n{format_token_stats(token_stats)}")
        listener.info(
            f"Token usage: {token_stats['total_tokens']:,} total "
            f"({token_stats['total_input_tokens']:,} input, "
            f"{token_stats['total_output_tokens']:,} output) "
            f"across {token_stats['invocation_count']} VLM calls"
        )

        # Filter successful predictions
        predictions = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]
        if carried_forward_predictions and not stream_predictions:
            predictions = (
                self._carried_forward_predictions_as_results(
                    carried_forward_predictions
                )
                + predictions
            )

        listener.info(
            f"✓ Inference complete: {len(predictions)} successful, {len(failed)} failed"
        )

        # Emit task completed event
        listener.event(
            "task.completed",
            {
                "task_name": "VLMInference",
                "successful": len(predictions),
                "failed": len(failed),
                "total": len(dataset),
            },
        )
        listener.info(
            f"VLM inference complete: {len(predictions)}/{len(dataset)} successful"
        )

        # When streaming, reload predictions from file as the authoritative set
        if stream_predictions and predictions_path.exists():
            try:
                with open(predictions_path, encoding="utf-8") as f:
                    file_predictions = [json.loads(line) for line in f if line.strip()]

                # Create a set of IDs that were successfully saved to file
                file_prediction_ids = {p.get("id") for p in file_predictions}

                # Reconcile with original results to maintain accurate failure count
                # Keep failures that weren't in the file (since we only stream successes)
                actual_failed = [
                    r
                    for r in results
                    if r["status"] == "error" or r["id"] not in file_prediction_ids
                ]

                # Update predictions from file (these are the confirmed successes)
                predictions = [
                    {
                        "id": p.get("id"),
                        "vlm_response": p.get("materials"),
                        "status": "success",
                    }
                    for p in file_predictions
                ]

                # Update the failed list with the reconciled failures
                failed = actual_failed

                logger.info(
                    f"Reloaded {len(predictions)} successful predictions from file, "
                    f"reconciled {len(failed)} failures"
                )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from predictions file: {e}")
            except Exception as e:
                logger.warning(f"Failed to reload streamed predictions: {e}")

        self._fail_if_predictions_empty(
            input_dataset_count,
            predictions,
            failed,
            predictions_path,
            allow_empty_predictions,
        )

        # Store predictions in object store
        if object_store:
            object_store.set("predictions", predictions)
            object_store.set("failed_predictions", failed)

        # Update context
        context["predictions_count"] = len(predictions)
        context["failed_count"] = len(failed)
        context["inference_complete"] = True
        context["predictions_path"] = str(predictions_path)
        context["token_stats"] = token_stats  # Save token statistics for reporting

        return context

    async def arun(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Run batch inference on dataset asynchronously.

        Async version of run() that uses async_batch_assign_materials() for
        true async I/O with asyncio.gather() instead of ThreadPoolExecutor.

        Args:
            context: Workflow context with dataset metadata
            object_store: Storage for dataset and predictions

        Returns:
            Updated context with inference results
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Resolve parameters from constructor or context
        vlm = self.vlm if self.vlm is not None else context.get("vlm")
        llm = self.llm if self.llm is not None else context.get("llm")
        if llm is None:
            llm = vlm  # Use VLM as LLM if not provided

        # Get config values from context if not provided in constructor
        vlm_config = context.get("vlm_config", {})
        max_retries = vlm_config.get("max_retries", 3)
        allow_empty_predictions = self._allow_empty_predictions(context)
        system_prompt = (
            self.system_prompt
            if self.system_prompt is not None
            else context.get("config", {}).get("system_prompt")
        )

        # Include previous judge critique in system prompt (for iterative refinement)
        previous_critique = context.get("previous_judge_critique")
        if previous_critique:
            iteration_count = context.get("iteration_count", 1)
            listener.info(
                f"Including previous judge critique in system prompt "
                f"(iteration {iteration_count})"
            )
            critique_section = (
                "\n\n**FEEDBACK FROM PREVIOUS ITERATION:**\n"
                "In the previous iteration, your material assignments were evaluated. "
                "Here is the critique:\n\n"
                f"{previous_critique}\n\n"
                "**IMPORTANT — BE CONSERVATIVE:**\n"
                "- Only change parts that the critique specifically identifies as wrong.\n"
                "- Keep ALL other assignments exactly as they were.\n"
                "- Do NOT change more than 3-4 parts based on this feedback.\n"
                "- If the critique suggests a wholesale color change, ignore it — "
                "only fix specific inconsistencies or clear errors.\n"
                "- Prefer your previous answer unless the feedback gives a clear, "
                "specific reason to change a particular part."
            )
            if system_prompt:
                system_prompt = system_prompt + critique_section
            else:
                system_prompt = critique_section

        context["actual_system_prompt_used"] = system_prompt
        listener.debug("Saved actual system prompt (with feedback) to context")

        vlm_invoke_kwargs: dict[str, Any] = dict(context.get("vlm_invoke_kwargs", {}))

        if vlm is None:
            raise ValueError("VLM not provided in constructor or context")

        # Get dataset from object store or context
        if object_store and object_store.exists("dataset"):
            dataset = object_store.get("dataset")
        else:
            dataset_path = Path(context["dataset_path"])
            with open(dataset_path, encoding="utf-8") as f:
                dataset = [json.loads(line) for line in f]
        input_dataset_count = len(dataset)

        # Selective re-prediction with resolved assignments
        resolved_assignments = context.get("resolved_assignments", {})
        prim_feedback = context.get("previous_prim_feedback", {})
        visual_refinement_context_by_prim = context.get(
            "visual_refinement_context_by_prim", {}
        )
        if not isinstance(visual_refinement_context_by_prim, dict):
            visual_refinement_context_by_prim = {}
        previous_predictions_path = context.get("previous_predictions_path")
        carried_forward_predictions: list[dict[str, Any]] = []

        if (resolved_assignments or prim_feedback) and previous_predictions_path:
            prev_path = Path(previous_predictions_path)
            if prev_path.exists():
                prev_preds: dict[str, dict[str, Any]] = {}
                with open(prev_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            pred = json.loads(line)
                            prev_preds[pred.get("id", "")] = pred

                carried_forward_predictions, re_predict_entries, resolved_count = (
                    self._classify_entries_for_selective_reprediction(
                        dataset,
                        resolved_assignments,
                        prim_feedback,
                        prev_preds,
                        visual_refinement_context_by_prim,
                    )
                )

                listener.info(
                    f"Selective re-prediction: "
                    f"{resolved_count} resolved by analyzer, "
                    f"{len(re_predict_entries)} to re-predict with VLM, "
                    f"{len(carried_forward_predictions) - resolved_count} "
                    f"carried forward (unchanged)"
                )
                dataset = re_predict_entries
            else:
                listener.warning(
                    f"Previous predictions not found at {prev_path}, "
                    f"re-predicting all entries"
                )
        elif prim_feedback:
            feedback_count = 0
            for entry in dataset:
                entry_id = entry.get("id", "")
                if entry_id in prim_feedback:
                    feedback_text = prim_feedback[entry_id]
                    visual_context_raw = (
                        visual_refinement_context_by_prim.get(entry_id, {})
                        if visual_refinement_context_by_prim
                        else {}
                    )
                    visual_context = (
                        visual_context_raw
                        if isinstance(visual_context_raw, dict)
                        else {}
                    )
                    visual_text = str(visual_context.get("text") or "").strip()
                    visual_section = (
                        f"\n\n**TARGETED VISUAL REFINEMENT EVIDENCE:**\n{visual_text}\n"
                        if visual_text
                        else ""
                    )
                    entry["text"] = (
                        f"\n**FEEDBACK FOR THIS SPECIFIC PART "
                        f"(from previous iteration):**\n"
                        f"{feedback_text}"
                        f"{visual_section}\n\n"
                        f"{entry.get('text', '')}"
                    )
                    VLMInferenceTask._attach_visual_refinement_images(
                        entry,
                        visual_context,
                    )
                    feedback_count += 1
            if feedback_count:
                listener.info(
                    f"Injected per-prim feedback for {feedback_count}/{len(dataset)} "
                    f"entries (no previous predictions to carry forward)"
                )

        listener.event(
            "task.started",
            {
                "task_name": "VLMInference",
                "total_entries": len(dataset),
            },
        )
        listener.info(f"Starting async VLM inference for {len(dataset)} entries")

        # Progress callback
        processed_count = [0]

        def on_progress(entry_id: str, response: str) -> None:
            processed_count[0] += 1
            listener.event(
                "task.progress",
                {
                    "task_name": "VLMInference",
                    "current": processed_count[0],
                    "total": len(dataset),
                    "percentage": (processed_count[0] / len(dataset)) * 100,
                    "entry_id": entry_id,
                },
            )
            if processed_count[0] % 10 == 0:
                listener.debug(f"Processed {processed_count[0]}/{len(dataset)} entries")
            logger.debug(f"Completed processing entry: {entry_id}")

        def on_error(entry_id: str, error: str) -> None:
            logger.error(f"Error processing {entry_id}: {error}")
            listener.error(f"Error processing {entry_id}: {error}")

        def on_prediction(entry_id: str, material_dict: dict[str, Any]) -> None:
            try:
                material = material_dict.get("material", "unknown")
                confidence = material_dict.get("confidence")
                response_snippet = material_dict.get("original_response", "")
                event_payload = {
                    "entry_id": entry_id,
                    "material": material,
                    "confidence": confidence,
                    "response_snippet": response_snippet,
                }
                reconciliation = material_dict.get("evidence_reconciliation")
                if isinstance(reconciliation, dict):
                    event_payload["evidence_reconciliation"] = reconciliation
                listener.event("prediction.completed", event_payload)
            except Exception as e:
                logger.warning(f"Failed to emit prediction event for {entry_id}: {e}")

        stream_predictions = context.get("stream_predictions", True)
        resume_enabled = context.get("resume", False)

        predictions_path = context.get("predictions_path")

        if predictions_path:
            predictions_path = Path(predictions_path)
            output_dir = predictions_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using predictions path from context: {predictions_path}")
        else:
            output_dir = context.get("output_dir")
            if output_dir is None:
                dataset_path_str = context.get("dataset_path")
                if not dataset_path_str:
                    raise ValueError(
                        "dataset_path not found in context and output_dir not specified"
                    )
                dataset_path = Path(dataset_path_str).parent / "output"
                output_dir = dataset_path
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = output_dir / "predictions.jsonl"

        if stream_predictions and not resume_enabled and predictions_path.exists():
            try:
                predictions_path.unlink()
                logger.info(
                    "Cleared existing predictions file since resume is disabled: %s",
                    predictions_path,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to remove existing predictions file {predictions_path}: {e}"
                )

        # Write carried-forward predictions (from selective re-prediction)
        if carried_forward_predictions and stream_predictions:
            try:
                written_count = 0
                with open(predictions_path, "a", encoding="utf-8") as f:
                    for pred in carried_forward_predictions:
                        # Validate required fields before writing
                        if not pred.get("id") or not pred.get("materials"):
                            logger.warning(
                                f"Skipping invalid carried-forward prediction: "
                                f"{pred.get('id', 'unknown')}"
                            )
                            continue
                        f.write(json.dumps(pred) + "\n")
                        written_count += 1
                logger.info(
                    "Wrote %d of %d carried-forward predictions to %s",
                    written_count,
                    len(carried_forward_predictions),
                    predictions_path,
                )
            except Exception as e:
                logger.warning(f"Failed to write carried-forward predictions: {e}")

        processed_ids: set[str] = set()
        if stream_predictions and resume_enabled and predictions_path.exists():
            try:
                with open(predictions_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            if rec and "id" in rec:
                                processed_ids.add(rec["id"])
                        except Exception:
                            continue
                logger.info(
                    "Resuming: %d entries already in predictions.jsonl",
                    len(processed_ids),
                )
            except Exception as e:
                logger.warning(f"Failed to parse existing predictions.jsonl: {e}")

        def on_result(result: dict[str, Any], entry: dict[str, Any]) -> None:
            if not stream_predictions:
                return
            try:
                if result.get("status") != "success":
                    return
                output_entry: dict[str, Any] = {
                    "id": result["id"],
                    "materials": result.get("vlm_response"),
                }
                if "images" in entry:
                    output_entry["images"] = entry["images"]
                elif "image_path" in entry:
                    output_entry["image_path"] = entry.get("image_path", "")

                vr = result.get("vlm_response")
                if isinstance(vr, dict) and "confidence" in vr:
                    output_entry["confidence"] = vr["confidence"]

                with open(predictions_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_entry) + "\n")
            except Exception as e:
                logger.warning(f"Failed to append streaming prediction: {e}")

        max_workers = context.get("max_workers")
        if max_workers:
            logger.info(f"Using async processing with {max_workers} concurrency")

        token_tracker = TokenTracker()
        logger.info("Token usage tracking enabled")

        # Check prediction_batch_size for multi-prim mode
        prediction_batch_size = context.get("prediction_batch_size", 1)

        if prediction_batch_size > 1 and len(dataset) > 0:
            # --- Multi-prim path ---
            listener.info(
                f"Using multi-prim inference (prediction_batch_size="
                f"{prediction_batch_size})"
            )
            # _run_multi_prim_inference is synchronous; run in thread
            import asyncio

            results = await asyncio.to_thread(
                self._run_multi_prim_inference,
                dataset=dataset,
                context=context,
                prediction_batch_size=prediction_batch_size,
                vlm=vlm,
                llm=llm,
                system_prompt=system_prompt,
                vlm_invoke_kwargs=vlm_invoke_kwargs,
                max_retries=max_retries,
                predictions_path=predictions_path,
                stream_predictions=stream_predictions,
                listener=listener,
                token_tracker=token_tracker,
            )
        else:
            # --- Standard single-prim path (batch_size=1) ---
            # Run async inference
            results = await async_batch_assign_materials(
                vlm=vlm,
                entries=dataset,
                llm=llm,
                image_base_dir=Path(context["image_base_dir"]),
                system_prompt=system_prompt,
                invoke_kwargs=vlm_invoke_kwargs,
                on_progress=on_progress,
                on_error=on_error,
                processed_ids=processed_ids,
                on_result=on_result,
                on_prediction=on_prediction,
                max_workers=max_workers,
                max_retries=max_retries,
                token_tracker=token_tracker,
            )

        # Get and log token usage statistics
        token_stats = token_tracker.get_stats()
        _write_token_usage_artifact(
            predictions_path,
            token_stats,
            merge_existing=stream_predictions
            and resume_enabled
            and bool(processed_ids),
        )
        logger.info(f"\n{format_token_stats(token_stats)}")
        listener.info(
            f"Token usage: {token_stats['total_tokens']:,} total "
            f"({token_stats['total_input_tokens']:,} input, "
            f"{token_stats['total_output_tokens']:,} output) "
            f"across {token_stats['invocation_count']} VLM calls"
        )

        predictions = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]
        if carried_forward_predictions and not stream_predictions:
            predictions = (
                self._carried_forward_predictions_as_results(
                    carried_forward_predictions
                )
                + predictions
            )

        listener.info(
            f"✓ Inference complete: {len(predictions)} successful, {len(failed)} failed"
        )

        listener.event(
            "task.completed",
            {
                "task_name": "VLMInference",
                "successful": len(predictions),
                "failed": len(failed),
                "total": len(dataset),
            },
        )
        listener.info(
            f"VLM inference complete: {len(predictions)}/{len(dataset)} successful"
        )

        # When streaming, reload predictions from file as the authoritative set
        if stream_predictions and predictions_path.exists():
            try:
                with open(predictions_path, encoding="utf-8") as f:
                    file_predictions = [json.loads(line) for line in f if line.strip()]

                file_prediction_ids = {p.get("id") for p in file_predictions}

                actual_failed = [
                    r
                    for r in results
                    if r["status"] == "error" or r["id"] not in file_prediction_ids
                ]

                predictions = [
                    {
                        "id": p.get("id"),
                        "vlm_response": p.get("materials"),
                        "status": "success",
                    }
                    for p in file_predictions
                ]

                failed = actual_failed

                logger.info(
                    f"Reloaded {len(predictions)} successful predictions from file, "
                    f"reconciled {len(failed)} failures"
                )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from predictions file: {e}")
            except Exception as e:
                logger.warning(f"Failed to reload streamed predictions: {e}")

        self._fail_if_predictions_empty(
            input_dataset_count,
            predictions,
            failed,
            predictions_path,
            allow_empty_predictions,
        )

        if object_store:
            object_store.set("predictions", predictions)
            object_store.set("failed_predictions", failed)

        context["predictions_count"] = len(predictions)
        context["failed_count"] = len(failed)
        context["inference_complete"] = True
        context["predictions_path"] = str(predictions_path)
        context["token_stats"] = token_stats

        return context
