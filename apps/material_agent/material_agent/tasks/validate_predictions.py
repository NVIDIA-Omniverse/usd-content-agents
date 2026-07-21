# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and repair VLM predictions against the material library.

After the predict step, some VLM predictions may contain material names that
don't exactly match any entry in the material library. This task:

1. Loads predictions JSONL and material library names from config.
2. Checks each prediction's material name against the library.
3. For invalid names: fuzzy-match via difflib.SequenceMatcher (>0.7 → auto-correct).
4. For low-confidence matches: batch LLM repair call.
5. Atomically rewrites predictions JSONL with corrected names.
6. Logs a summary of corrections.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

from material_agent.materials import (
    DISALLOWED_UNKNOWN_VALIDATION_STATUS,
    FALLBACK_MATERIAL_NAME,
    PREDICTION_CONTAINER_KEYS,
    PREDICTION_ID_KEYS,
    PREDICTION_MATERIAL_KEYS,
    UNKNOWN_MATERIAL_SENTINEL,
    is_unknown_material_name,
)

logger = logging.getLogger(__name__)

# Fuzzy match thresholds
_AUTO_CORRECT_THRESHOLD = 0.7
_LLM_REPAIR_THRESHOLD = 0.4
_STALE_TEMP_FILE_AGE_SECONDS = 60 * 60


@dataclass
class _PredictionMaterialRecord:
    """Mutable material selection discovered in a prediction payload."""

    index: int
    material: str | None
    set_material: Callable[[str], None]
    mark_disallowed_unknown: Callable[[], None]
    mark_fallback: Callable[[str | None, str], None]
    report_id: str | None = None
    reason: str | None = None


def _best_fuzzy_match(name: str, valid_names: list[str]) -> tuple[str | None, float]:
    """Find the best fuzzy match for a name in the valid names list.

    Uses SequenceMatcher ratio with a token-containment bonus: if the
    candidate contains every word from the query, it gets a +0.1 boost.
    This prevents short-but-unrelated names from beating longer correct ones
    (e.g. "Gold Polished" beating "Stainless Steel Polished" for "Steel Polished").
    """
    best_match = None
    best_score = 0.0
    name_lower = name.lower()
    name_tokens = set(name_lower.split())
    for valid in valid_names:
        valid_lower = valid.lower()
        score = SequenceMatcher(None, name_lower, valid_lower).ratio()
        # Bonus if the candidate contains all tokens from the query
        valid_tokens = set(valid_lower.split())
        if name_tokens and name_tokens.issubset(valid_tokens):
            score += 0.1
        if score > best_score:
            best_score = score
            best_match = valid
    return best_match, best_score


def _extract_material_name(prediction: dict[str, Any]) -> str | None:
    """Extract material name from a prediction entry."""
    has_material, material, _, _, _, _ = _selected_material_location(prediction)
    if not has_material:
        return None
    return material if isinstance(material, str) else None


def _set_material_name(prediction: dict[str, Any], name: str) -> None:
    """Set material name in a prediction entry."""
    if not isinstance(prediction.get("materials"), dict):
        prediction["materials"] = {}
    prediction["materials"]["material"] = name


def _mark_materials_disallowed_unknown(prediction: dict[str, Any]) -> None:
    """Persist that an unknown sentinel was cleared by validation policy."""
    materials = prediction.get("materials")
    if not isinstance(materials, dict):
        prediction["materials"] = {
            "material": materials if isinstance(materials, str) else ""
        }
    prediction["materials"]["validation_status"] = DISALLOWED_UNKNOWN_VALIDATION_STATUS


def _mark_materials_fallback(
    prediction: dict[str, Any], original: str | None, reason: str
) -> None:
    """Annotate a prediction whose material was replaced by the fallback."""
    materials = prediction.get("materials")
    if not isinstance(materials, dict):
        prediction["materials"] = {
            "material": materials
            if isinstance(materials, str)
            else FALLBACK_MATERIAL_NAME
        }
        materials = prediction["materials"]
    materials["fallback_source"] = "validation"
    materials["fallback_reason"] = reason
    if original:
        materials["fallback_original_material"] = original


def _top_level_validation_status_marker(
    prediction: dict[str, Any],
) -> Callable[[], None]:
    def mark_disallowed_unknown() -> None:
        prediction["validation_status"] = DISALLOWED_UNKNOWN_VALIDATION_STATUS

    return mark_disallowed_unknown


def _noop_disallowed_unknown_marker() -> None:
    """Do nothing for synthesized missing-material records."""
    return None


def _noop_fallback_marker(original: str | None, reason: str) -> None:
    """Do nothing for synthesized fallback records."""
    return None


def _prediction_prim_id(prediction: dict[str, Any]) -> str | None:
    """Return the first supported prim identifier from a prediction record."""
    for key in PREDICTION_ID_KEYS:
        value = prediction.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _top_level_material_setter(
    prediction: dict[str, Any], key: str
) -> Callable[[str], None]:
    def set_material(name: str) -> None:
        prediction[key] = name

    return set_material


def _top_level_fallback_marker(
    prediction: dict[str, Any],
) -> Callable[[str | None, str], None]:
    def mark_fallback(original: str | None, reason: str) -> None:
        prediction["fallback_source"] = "validation"
        prediction["fallback_reason"] = reason
        if original:
            prediction["fallback_original_material"] = original

    return mark_fallback


def _selected_material_location(
    prediction: dict[str, Any],
) -> tuple[
    bool,
    Any | None,
    Callable[[str], None],
    Callable[[], None],
    Callable[[str | None, str], None],
    str | None,
]:
    """Return selected material value, setter, and optional reason."""
    materials = prediction.get("materials")
    if isinstance(materials, dict):
        return (
            True,
            materials.get("material"),
            lambda name: materials.__setitem__("material", name),
            lambda: materials.__setitem__(
                "validation_status", DISALLOWED_UNKNOWN_VALIDATION_STATUS
            ),
            lambda original, reason: (
                materials.__setitem__("fallback_source", "validation"),
                materials.__setitem__("fallback_reason", reason),
                materials.__setitem__(
                    "fallback_original_material",
                    original,
                )
                if original
                else None,
            ),
            materials.get("reason")
            if isinstance(materials.get("reason"), str)
            else None,
        )
    if isinstance(materials, str):
        return (
            True,
            materials,
            lambda name: prediction.__setitem__("materials", name),
            lambda: _mark_materials_disallowed_unknown(prediction),
            lambda original, reason: _mark_materials_fallback(
                prediction, original, reason
            ),
            None,
        )
    for key in PREDICTION_MATERIAL_KEYS:
        if key not in prediction:
            continue
        return (
            True,
            prediction.get(key),
            _top_level_material_setter(prediction, key),
            _top_level_validation_status_marker(prediction),
            _top_level_fallback_marker(prediction),
            None,
        )
    return (
        False,
        None,
        lambda name: _set_material_name(prediction, name),
        lambda: _mark_materials_disallowed_unknown(prediction),
        lambda original, reason: _mark_materials_fallback(prediction, original, reason),
        None,
    )


def _unknown_prediction_report_entry(
    record: _PredictionMaterialRecord,
) -> dict[str, Any]:
    """Build a compact report entry for an unknown-material prediction."""
    entry: dict[str, Any] = {"index": record.index}
    if record.report_id:
        entry["id"] = record.report_id
    if isinstance(record.reason, str) and record.reason.strip():
        entry["reason"] = record.reason.strip()
    return entry


def _write_predictions_atomically(
    predictions_path: Path,
    predictions: Any,
    *,
    json_document: bool = False,
) -> None:
    """Write predictions via a same-directory temp file and atomic replace."""
    stale_before = time.time() - _STALE_TEMP_FILE_AGE_SECONDS
    for stale_path in predictions_path.parent.glob(f".{predictions_path.name}.*.tmp"):
        try:
            if stale_path.stat().st_mtime < stale_before:
                stale_path.unlink(missing_ok=True)
        except OSError:
            continue

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=predictions_path.parent,
            prefix=f".{predictions_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            temp_path = Path(tmp.name)
            if json_document:
                json.dump(predictions, tmp, indent=2, ensure_ascii=False)
                tmp.write("\n")
            else:
                for pred in predictions:
                    tmp.write(json.dumps(pred, ensure_ascii=False) + "\n")

        temp_path.replace(predictions_path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _load_predictions_for_validation(
    predictions_path: Path, listener: Any
) -> tuple[Any, bool]:
    """Load a JSON document or JSONL predictions while preserving format."""
    content = predictions_path.read_text(encoding="utf-8").strip()
    if not content:
        listener.warning("Predictions file is empty")
        return [], False

    if predictions_path.suffix.lower() == ".jsonl":
        return _load_jsonl_predictions(content), False

    try:
        return json.loads(content), True
    except json.JSONDecodeError:
        return _load_jsonl_predictions(content), False


def _load_jsonl_predictions(content: str) -> list[Any]:
    """Load one JSON value per non-empty line."""
    predictions = []
    for line_num, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            predictions.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL prediction on line {line_num}: {exc}"
            ) from exc
    return predictions


def _prediction_material_records(payload: Any) -> list[_PredictionMaterialRecord]:
    """Return mutable material records from flexible prediction payloads."""
    records: list[_PredictionMaterialRecord] = []

    def add_record(
        material: Any | None,
        setter: Callable[[str], None],
        report_id: str | None,
        reason: str | None = None,
        marker: Callable[[], None] = _noop_disallowed_unknown_marker,
        fallback_marker: Callable[[str | None, str], None] = _noop_fallback_marker,
    ) -> None:
        records.append(
            _PredictionMaterialRecord(
                index=len(records),
                material=material if isinstance(material, str) else None,
                set_material=setter,
                mark_disallowed_unknown=marker,
                mark_fallback=fallback_marker,
                report_id=report_id,
                reason=reason,
            )
        )

    def parent_accessors(
        parent: dict[str, Any] | list[Any],
        key: str | int,
    ) -> (
        tuple[
            Callable[[str], None],
            Callable[[], None],
            Callable[[str | None, str], None],
        ]
        | None
    ):
        if isinstance(parent, list) and isinstance(key, int):

            def set_list_item(name: str) -> None:
                parent[key] = name

            def mark_list_item() -> None:
                parent[key] = {
                    "material": "",
                    "validation_status": DISALLOWED_UNKNOWN_VALIDATION_STATUS,
                }

            def mark_list_fallback(original: str | None, reason: str) -> None:
                parent[key] = {
                    "material": parent[key]
                    if isinstance(parent[key], str)
                    else FALLBACK_MATERIAL_NAME,
                    "fallback_source": "validation",
                    "fallback_reason": reason,
                }
                if original:
                    parent[key]["fallback_original_material"] = original

            return set_list_item, mark_list_item, mark_list_fallback
        if isinstance(parent, dict) and isinstance(key, str):

            def set_dict_item(name: str) -> None:
                parent[key] = name

            def mark_dict_item() -> None:
                parent[key] = {
                    "material": "",
                    "validation_status": DISALLOWED_UNKNOWN_VALIDATION_STATUS,
                }

            def mark_dict_fallback(original: str | None, reason: str) -> None:
                parent[key] = {
                    "material": parent[key]
                    if isinstance(parent[key], str)
                    else FALLBACK_MATERIAL_NAME,
                    "fallback_source": "validation",
                    "fallback_reason": reason,
                }
                if original:
                    parent[key]["fallback_original_material"] = original

            return set_dict_item, mark_dict_item, mark_dict_fallback
        return None  # pragma: no cover - defensive guard for impossible traversal state

    def visit(
        node: Any,
        *,
        fallback_id: str | None = None,
        parent: dict[str, Any] | list[Any] | None = None,
        key: str | int | None = None,
    ) -> None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                child_fallback = (
                    f"{fallback_id}.{index}" if fallback_id is not None else None
                )
                visit(item, fallback_id=child_fallback, parent=node, key=index)
            return

        if isinstance(node, str):
            if parent is not None and key is not None:
                accessors = parent_accessors(parent, key)
                if accessors is not None:
                    setter, marker, fallback_marker = accessors
                    add_record(
                        node,
                        setter,
                        fallback_id,
                        marker=marker,
                        fallback_marker=fallback_marker,
                    )
            return

        if not isinstance(node, dict):
            return

        report_id = _prediction_prim_id(node) or fallback_id
        (
            has_material,
            material,
            setter,
            marker,
            fallback_marker,
            reason,
        ) = _selected_material_location(node)
        record_count_before = len(records)
        if has_material:
            add_record(
                material,
                setter,
                report_id,
                reason,
                marker=marker,
                fallback_marker=fallback_marker,
            )

        for container_key in PREDICTION_CONTAINER_KEYS:
            container = node.get(container_key)
            if isinstance(container, dict | list):
                visit(container, fallback_id=report_id)

        for item_key, value in node.items():
            if item_key in PREDICTION_CONTAINER_KEYS or item_key in PREDICTION_ID_KEYS:
                continue
            if not (isinstance(item_key, str) and item_key.startswith("/")):
                continue
            visit(value, fallback_id=item_key, parent=node, key=item_key)

        if len(records) == record_count_before and (
            report_id is not None or (parent is not None and key is not None)
        ):
            add_record(
                None,
                setter,
                report_id,
                marker=marker,
                fallback_marker=fallback_marker,
            )

    visit(payload)
    return records


class ValidatePredictionsTask(Task):
    """Validate VLM predictions against the material library.

    Input context keys:
        - predictions_path: Path to predictions JSONL file
        - material_names: List of valid material names from library
        - llm_config: Optional LLM config for repair (backend, model, etc.)

    Output context keys:
        - predictions_path: Same path (rewritten in-place)
        - validation_stats: Dict with correction counts
    """

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        predictions_path = Path(context["predictions_path"])
        material_names: list[str] = context["material_names"]
        llm_config: dict[str, Any] | None = context.get("llm_config")
        allow_unknown_material = context.get("allow_unknown_material", True)
        if not isinstance(allow_unknown_material, bool):
            raise ValueError(
                "allow_unknown_material must be a boolean, got "
                f"{type(allow_unknown_material).__name__}"
            )

        if not predictions_path.exists():
            listener.warning(f"Predictions file not found: {predictions_path}")
            context["validation_stats"] = {"skipped": True}
            return context

        fallback_material_name = context.get("fallback_material_name")
        if not isinstance(fallback_material_name, str) or not fallback_material_name:
            fallback_material_name = FALLBACK_MATERIAL_NAME
        if fallback_material_name not in material_names:
            material_names = [*material_names, fallback_material_name]
        valid_set = set(material_names)
        listener.info(f"Validating predictions against {len(valid_set)} material names")

        # Load predictions
        predictions, json_document = _load_predictions_for_validation(
            predictions_path, listener
        )
        prediction_records = _prediction_material_records(predictions)
        record_by_index = {record.index: record for record in prediction_records}

        # Classify each prediction
        valid_count = 0
        auto_corrected: list[tuple[int, str, str, float]] = []  # idx, old, new, score
        needs_llm: list[tuple[int, str, str, float]] = []  # idx, old, best_match, score
        no_material: list[int] = []
        unknown_materials: list[dict[str, Any]] = []
        unknown_disallowed_materials: list[dict[str, Any]] = []
        fallback_applied: list[dict[str, Any]] = []

        def apply_fallback(
            record: _PredictionMaterialRecord,
            original: str | None,
            reason: str,
        ) -> None:
            record.set_material(fallback_material_name)
            record.material = fallback_material_name
            record.mark_fallback(original, reason)
            entry: dict[str, Any] = {
                "index": record.index,
                "new": fallback_material_name,
                "reason": reason,
            }
            if record.report_id:
                entry["id"] = record.report_id
            if original:
                entry["old"] = original
            fallback_applied.append(entry)

        for record in prediction_records:
            mat_name = record.material
            if mat_name is None or not mat_name.strip():
                no_material.append(record.index)
                apply_fallback(record, mat_name, "missing_material")
                continue
            if is_unknown_material_name(mat_name):
                entry = _unknown_prediction_report_entry(record)
                if allow_unknown_material:
                    unknown_materials.append(entry)
                else:
                    unknown_disallowed_materials.append(entry)
                    apply_fallback(record, mat_name, "unknown_material")
                continue
            if mat_name in valid_set:
                valid_count += 1
                continue

            best_match, score = _best_fuzzy_match(mat_name, material_names)
            if best_match and score >= _AUTO_CORRECT_THRESHOLD:
                auto_corrected.append((record.index, mat_name, best_match, score))
            elif best_match and score >= _LLM_REPAIR_THRESHOLD:
                needs_llm.append((record.index, mat_name, best_match, score))
            else:
                # Very low match - still try LLM
                needs_llm.append((record.index, mat_name, best_match or "", score))

        listener.info(
            f"Validation: {valid_count} valid, "
            f"{len(unknown_materials)} unknown, "
            f"{len(auto_corrected)} auto-correctable, "
            f"{len(needs_llm)} need LLM repair, "
            f"{len(no_material)} missing material field"
        )
        if unknown_materials:
            listener.warning(
                f"{len(unknown_materials)} prediction(s) were classified as "
                f"'{UNKNOWN_MATERIAL_SENTINEL}' and are allowed by policy."
            )
        if unknown_disallowed_materials:
            listener.warning(
                f"{len(unknown_disallowed_materials)} prediction(s) were classified as "
                f"'{UNKNOWN_MATERIAL_SENTINEL}' and will use "
                f"'{fallback_material_name}'."
            )

        # Apply auto-corrections
        for idx, old_name, new_name, score in auto_corrected:
            record = record_by_index[idx]
            record.set_material(new_name)
            record.material = new_name
            listener.info(
                f"  Auto-corrected: '{old_name}' -> '{new_name}' (score={score:.2f})"
            )

        # LLM repair for low-confidence matches
        llm_repaired = 0
        llm_failed = 0
        if needs_llm and llm_config:
            repaired = self._llm_repair(
                [(idx, old, best) for idx, old, best, _ in needs_llm],
                material_names,
                llm_config,
                listener,
            )
            for idx, old_name, repaired_name in repaired:
                record = record_by_index.get(idx)
                if record is None:
                    listener.warning(
                        "  LLM repair returned unexpected prediction index "
                        f"{idx}; skipping."
                    )
                    llm_failed += 1
                    continue
                if (
                    repaired_name
                    and repaired_name in valid_set
                    and repaired_name != fallback_material_name
                ):
                    record.set_material(repaired_name)
                    record.material = repaired_name
                    listener.info(f"  LLM-repaired: '{old_name}' -> '{repaired_name}'")
                    llm_repaired += 1
                else:
                    # LLM couldn't fix it - use fuzzy best match as fallback
                    fallback_match = next(
                        (t for t in needs_llm if t[0] == idx),
                        None,
                    )
                    if fallback_match is None:
                        listener.warning(
                            "  LLM repair returned unexpected prediction index "
                            f"{idx}; skipping fuzzy fallback."
                        )
                        llm_failed += 1
                        continue
                    _, _, best, score = fallback_match
                    if best and score >= _LLM_REPAIR_THRESHOLD:
                        record.set_material(best)
                        record.material = best
                        listener.warning(
                            f"  LLM failed for '{old_name}', "
                            f"fuzzy fallback: '{best}' (score={score:.2f})"
                        )
                        llm_repaired += 1
                    else:
                        listener.warning(
                            f"  Could not repair: '{old_name}' (no good match)"
                        )
                        apply_fallback(record, old_name, "unrepaired_invalid_material")
                        llm_failed += 1
        elif needs_llm:
            # No LLM config - use fuzzy best match for all
            listener.warning("No LLM config for repair, using fuzzy fallback only")
            for idx, old_name, best_match, score in needs_llm:
                if best_match and score >= _LLM_REPAIR_THRESHOLD:
                    record = record_by_index[idx]
                    record.set_material(best_match)
                    record.material = best_match
                    listener.info(
                        f"  Fuzzy fallback: '{old_name}' -> '{best_match}' "
                        f"(score={score:.2f})"
                    )
                    llm_repaired += 1
                else:
                    listener.warning(
                        f"  Could not repair: '{old_name}' (no good match)"
                    )
                    apply_fallback(
                        record_by_index[idx],
                        old_name,
                        "unrepaired_invalid_material",
                    )
                    llm_failed += 1

        _write_predictions_atomically(
            predictions_path,
            predictions,
            json_document=json_document,
        )

        stats = {
            "total": len(prediction_records),
            "valid": valid_count,
            "auto_corrected": len(auto_corrected),
            "llm_repaired": llm_repaired,
            "failed": llm_failed,
            "no_material": len(no_material),
            "unknown": len(unknown_materials),
            "unknown_disallowed": len(unknown_disallowed_materials),
            "fallback_applied": len(fallback_applied),
        }
        context["validation_stats"] = stats
        existing_unknown_count = context.get("unknown_material_predictions", 0)
        if not isinstance(existing_unknown_count, int):
            existing_unknown_count = 0
        context["unknown_material_predictions"] = max(
            existing_unknown_count,
            stats["unknown"],
        )
        listener.info(
            f"Validation complete: {stats['auto_corrected']} auto-corrected, "
            f"{stats['llm_repaired']} LLM-repaired, {stats['failed']} failed, "
            f"{stats['unknown']} unknown, "
            f"{stats['unknown_disallowed']} disallowed unknown, "
            f"{stats['fallback_applied']} fallback-applied"
        )

        # Write validation report
        report: dict[str, Any] = {
            "stats": stats,
            "auto_corrected": [
                {"index": idx, "old": old, "new": new, "score": round(score, 3)}
                for idx, old, new, score in auto_corrected
            ],
            "llm_repaired": [],
            "failed": [],
            "unknown": unknown_materials,
            "unknown_disallowed": unknown_disallowed_materials,
            "fallback_applied": fallback_applied,
        }
        # Collect LLM repair results if available
        if needs_llm and llm_config:
            for idx, old_name, best_match, score in needs_llm:
                final_mat = record_by_index[idx].material
                if (
                    final_mat
                    and final_mat != old_name
                    and final_mat in valid_set
                    and final_mat != fallback_material_name
                ):
                    report["llm_repaired"].append(
                        {"index": idx, "old": old_name, "new": final_mat}
                    )
                else:
                    report["failed"].append(
                        {
                            "index": idx,
                            "name": old_name,
                            "best_fuzzy": best_match,
                            "score": round(score, 3),
                        }
                    )
        report_path = predictions_path.parent / "validate_report.json"
        _write_predictions_atomically(report_path, report, json_document=True)
        listener.info(f"Validation report written to {report_path}")

        return context

    # Maximum items per LLM repair batch to avoid token-limit timeouts
    _LLM_BATCH_SIZE = 30

    def _llm_repair(
        self,
        items: list[tuple[int, str, str]],  # (idx, invalid_name, fuzzy_best)
        valid_names: list[str],
        llm_config: dict[str, Any],
        listener: Any,
    ) -> list[tuple[int, str, str | None]]:
        """Use LLM to repair invalid material names.

        Splits items into batches of _LLM_BATCH_SIZE and runs them in parallel
        using a thread pool. Each batch is a single LLM call.
        Returns list of (idx, old_name, repaired_name_or_None).
        """
        from world_understanding.functions.models.chat_models import (
            create_chat_model_from_config,
        )

        llm = create_chat_model_from_config(llm_config)
        if llm is None:
            listener.warning("No API key for LLM repair — skipping")
            return [(idx, old, None) for idx, old, _ in items]

        # Split into batches
        batches = [
            items[i : i + self._LLM_BATCH_SIZE]
            for i in range(0, len(items), self._LLM_BATCH_SIZE)
        ]

        if len(batches) == 1:
            return self._llm_repair_batch(batches[0], valid_names, llm, listener)

        # Run batches in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = min(len(batches), 8)
        listener.info(
            f"Splitting {len(items)} repairs into {len(batches)} batches "
            f"({max_workers} parallel workers)"
        )

        all_repaired: list[tuple[int, str, str | None]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._llm_repair_batch, batch, valid_names, llm, listener
                ): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    all_repaired.extend(future.result())
                except Exception:
                    logger.exception("LLM repair batch %d failed", batch_idx)
                    batch = batches[batch_idx]
                    all_repaired.extend((idx, old, None) for idx, old, _ in batch)

        return all_repaired

    def _llm_repair_batch(
        self,
        items: list[tuple[int, str, str]],
        valid_names: list[str],
        llm: Any,
        listener: Any,
    ) -> list[tuple[int, str, str | None]]:
        """Run a single LLM repair call for a batch of invalid names."""
        from world_understanding.utils.llm_parsing import (
            extract_json_from_llm_response,
        )

        names_text = "\n".join(
            f'  {i + 1}. "{old}" (closest fuzzy match: "{best}")'
            for i, (_, old, best) in enumerate(items)
        )
        valid_names_text = ", ".join(f'"{n}"' for n in valid_names)

        system_prompt = (
            "You are an expert at matching material names. "
            "Given a list of invalid material names and the valid material library, "
            "map each invalid name to the correct valid name from the library. "
            "Return ONLY a JSON object mapping each invalid name to its corrected name.\n"
            'Example: {"Steel Polished": "Stainless Steel Polished", '
            '"Plastic Pure White": "Plastic White"}'
        )
        user_prompt = (
            f"Valid material names:\n{valid_names_text}\n\n"
            f"Invalid names to fix:\n{names_text}\n\n"
            "Return a JSON object mapping each invalid name to the best matching "
            "valid name from the library above."
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            result = extract_json_from_llm_response(response.content)
        except Exception:
            logger.exception("LLM repair call failed")
            return [(idx, old, None) for idx, old, _ in items]

        if not result or not isinstance(result, dict):
            listener.warning("Failed to parse LLM repair response")
            return [(idx, old, None) for idx, old, _ in items]

        repaired = []
        for idx, old_name, _ in items:
            corrected = result.get(old_name)
            repaired.append((idx, old_name, corrected))
        return repaired
