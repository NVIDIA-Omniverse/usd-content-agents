# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material prediction and binding coverage qualification.

Coverage is deliberately computed from prim-level artifacts and assignment
evidence.  ``materials_applied`` is a count of unique material definitions and
must never be used as a proxy for prim bindings.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, cast

from material_agent.materials import (
    PREDICTION_CONTAINER_KEYS,
    PREDICTION_ID_KEYS,
    PREDICTION_MATERIAL_KEYS,
    PREDICTION_VALIDATION_STATUS_KEYS,
    is_actionable_material_name,
    is_default_library_fallback_name,
    is_disallowed_unknown_validation_status,
    is_fallback_material_name,
    is_unknown_material_name,
)

logger = logging.getLogger(__name__)

CoveragePolicy = Literal["strict", "allow_partial"]
COVERAGE_POLICIES = frozenset({"strict", "allow_partial"})


def normalize_coverage_policy(value: object) -> CoveragePolicy:
    """Validate and normalize a caller-supplied coverage policy."""
    normalized = str(value or "").strip().lower()
    if normalized not in COVERAGE_POLICIES:
        allowed = ", ".join(sorted(COVERAGE_POLICIES))
        raise ValueError(f"coverage_policy must be one of: {allowed}")
    return cast(CoveragePolicy, normalized)


def coverage_is_release_ready(coverage: dict[str, Any]) -> bool:
    """Return whether coverage qualifies for strict final success."""
    return coverage.get("readiness_grade") in {
        "complete",
        "complete_with_fallback",
    }


def build_not_evaluated_material_coverage(
    *,
    policy: CoveragePolicy,
    warning: str,
) -> dict[str, Any]:
    """Return an explicit unqualified contract when prim evidence is unavailable."""
    return {
        "schema_version": "1.0",
        "policy": policy,
        "readiness_grade": "not_evaluated",
        "target_count": 0,
        "prepared_count": 0,
        "predicted_count": 0,
        "usable_prediction_count": 0,
        "unknown_prediction_count": 0,
        "fallback_count": 0,
        "bound_count": 0,
        "unbound_count": 0,
        "prediction_coverage_ratio": 0.0,
        "binding_coverage_ratio": 0.0,
        "missing_prepared_prim_ids": [],
        "missing_prediction_prim_ids": [],
        "unknown_prim_ids": [],
        "fallback_prim_ids": [],
        "unbound_prim_ids": [],
        "extra_prediction_prim_ids": [],
        "warnings": [warning],
    }


LEGACY_COVERAGE_WARNING = (
    "This completed session predates persisted material coverage metadata; "
    "material readiness is not evaluated."
)


def normalize_legacy_completed_coverage(
    metadata: dict[str, Any],
    *,
    pipeline_active: bool,
) -> dict[str, Any]:
    """Add explicit read-only coverage to an inactive legacy completion."""
    if (
        pipeline_active
        or metadata.get("status") != "completed"
        or not metadata.get("completed_at")
        or "results" not in metadata
        or "coverage" in metadata
    ):
        return metadata

    policy: CoveragePolicy = "allow_partial"
    stored_config = metadata.get("config")
    raw_policy = (
        stored_config.get("coverage_policy")
        if isinstance(stored_config, dict)
        else None
    )
    try:
        policy = normalize_coverage_policy(raw_policy)
    except ValueError:
        pass

    return {
        **metadata,
        "coverage": build_not_evaluated_material_coverage(
            policy=policy,
            warning=LEGACY_COVERAGE_WARNING,
        ),
    }


def build_material_coverage(
    result: Any,
    session_dir: Path,
    *,
    policy: CoveragePolicy,
) -> dict[str, Any]:
    """Build the service-facing prim coverage contract for one pipeline run."""
    step_results = result.step_results or {}
    completed_steps = set(result.completed_steps or ()) | set(step_results)
    warnings: list[str] = []

    source_target_ids = _load_target_prim_ids(session_dir)
    source_prepared_ids = _load_prepared_prim_ids(result, session_dir)
    source_target_count_hint = max(
        _target_count_hint(step_results),
        _prepared_count_hint(step_results),
    )
    if not source_target_ids:
        source_target_ids = set(source_prepared_ids)

    (
        restore_ran,
        restore_mapping_complete,
        restored_prim_sources,
        restore_warnings,
    ) = _restore_namespace_evidence(step_results, completed_steps)
    warnings.extend(restore_warnings)

    namespace_qualified = True
    if restore_ran:
        restored_scope = {
            restored_id: source_id
            for restored_id, source_id in restored_prim_sources.items()
            if source_id in source_target_ids
        }
        mapped_source_ids = set(restored_prim_sources.values())
        source_ids_complete = bool(source_target_ids) and (
            len(source_target_ids) == source_target_count_hint
            if source_target_count_hint
            else source_target_ids == mapped_source_ids
        )
        if (
            source_target_count_hint
            and len(source_target_ids) != source_target_count_hint
        ):
            if len(source_target_ids) < source_target_count_hint:
                warnings.append(
                    f"{source_target_count_hint - len(source_target_ids)} optimized "
                    "target prim ID(s) were unavailable. Restored target scope is "
                    "taken from the complete restore correspondence instead."
                )
            else:
                warnings.append(
                    f"{len(source_target_ids) - source_target_count_hint} unexpected "
                    "optimized target prim ID(s) were present. Restored target scope "
                    "is taken from the complete restore correspondence instead."
                )
        if source_target_ids - mapped_source_ids:
            warnings.append(
                f"{len(source_target_ids - mapped_source_ids)} optimized target prim "
                "ID(s) were absent from the restore correspondence map."
            )
        if not source_target_count_hint and mapped_source_ids - source_target_ids:
            warnings.append(
                f"{len(mapped_source_ids - source_target_ids)} restore source prim "
                "ID(s) were absent from the prepared target scope."
            )

        namespace_qualified = bool(
            restore_mapping_complete
            and source_ids_complete
            and restored_scope
            and not (source_target_ids - mapped_source_ids)
        )
        target_ids = set(restored_scope)
        target_count = len(target_ids)
        target_ids_complete = namespace_qualified
        prepared_ids = {
            restored_id
            for restored_id, source_id in restored_scope.items()
            if source_id in source_prepared_ids
        }
        prepared_count = len(prepared_ids)
        missing_prepared_ids = sorted(target_ids - prepared_ids)
    else:
        target_ids = source_target_ids
        prepared_ids = source_prepared_ids
        target_count = max(len(target_ids), source_target_count_hint)
        target_ids_complete = target_count > 0 and len(target_ids) == target_count

        if target_count and not target_ids_complete:
            warnings.append(
                f"{target_count - len(target_ids)} target prim ID(s) were unavailable; "
                "aggregate coverage cannot qualify strict readiness or provide every "
                "missing ID."
            )

        prepared_count = len(prepared_ids)
        if not prepared_ids:
            prepared_count = _prepared_count_hint(step_results)
        if target_ids_complete:
            prepared_count = len(prepared_ids & target_ids)
            missing_prepared_ids = sorted(target_ids - prepared_ids)
        else:
            prepared_count = min(prepared_count, target_count)
            missing_prepared_ids = sorted(target_ids - prepared_ids)

    predictions_path = _prediction_path(result, session_dir)
    prediction_states = _load_prediction_states(predictions_path)
    all_prediction_ids = set(prediction_states)
    if target_ids_complete:
        prediction_ids = all_prediction_ids & target_ids
        extra_prediction_ids = sorted(all_prediction_ids - target_ids)
    else:
        prediction_ids = all_prediction_ids
        extra_prediction_ids = []

    usable_ids = {
        prim_id for prim_id in prediction_ids if prediction_states[prim_id] == "usable"
    }
    fallback_ids = {
        prim_id
        for prim_id in prediction_ids
        if prediction_states[prim_id] == "fallback"
    }
    unknown_ids = {
        prim_id for prim_id in prediction_ids if prediction_states[prim_id] == "unknown"
    }
    missing_prediction_ids = sorted(target_ids - prediction_ids) if target_ids else []

    (
        bound_ids,
        reported_unbound_ids,
        bound_count_hint,
        exact_binding_ids,
    ) = _binding_evidence(step_results.get("apply", {}))
    scoped_binding_ids = bool(
        target_ids_complete
        and exact_binding_ids
        and bound_ids.issubset(target_ids)
        and reported_unbound_ids.issubset(target_ids)
        and not (bound_ids & reported_unbound_ids)
    )
    if scoped_binding_ids:
        bound_count = len(bound_ids)
        unbound_ids = target_ids - bound_ids
    else:
        bound_count = len(bound_ids) or bound_count_hint
        bound_count = min(bound_count, target_count)
        unbound_ids = set(missing_prediction_ids) | reported_unbound_ids
        if target_ids and not bound_ids:
            # Prediction gaps are certainly unbound.  Assignment stats emitted
            # by current Material Agent add the remaining exact IDs.
            unbound_ids |= target_ids - prediction_ids

    unbound_count = max(0, target_count - bound_count)
    if len(unbound_ids) > unbound_count:
        # Do not let stale or out-of-scope assignment IDs inflate the target
        # contract.  Prefer IDs known to be in target scope.
        scoped = unbound_ids & target_ids if target_ids else set()
        unbound_ids = scoped or set(sorted(unbound_ids)[:unbound_count])
    if len(unbound_ids) < unbound_count:
        warnings.append(
            f"{unbound_count - len(unbound_ids)} unbound target prim ID(s) were "
            "not present in assignment evidence."
        )

    prediction_evaluated = predictions_path is not None
    apply_evaluated = "apply" in completed_steps
    qualified_stages = bool(
        namespace_qualified
        and target_ids_complete
        and prediction_evaluated
        and apply_evaluated
        and scoped_binding_ids
    )
    prediction_covered = len(usable_ids) + len(fallback_ids)
    complete = bool(
        qualified_stages
        and prepared_count == target_count
        and not missing_prediction_ids
        and not unknown_ids
        and prediction_covered == target_count
        and bound_count == target_count
        and unbound_count == 0
    )
    if complete:
        readiness_grade = "complete_with_fallback" if fallback_ids else "complete"
    elif qualified_stages:
        readiness_grade = "partial"
    else:
        readiness_grade = "not_evaluated"

    if missing_prepared_ids:
        warnings.append(
            f"{len(missing_prepared_ids)} target prim(s) were not prepared."
        )
    if missing_prediction_ids:
        warnings.append(
            f"{len(missing_prediction_ids)} target prim(s) have no prediction record."
        )
    if unknown_ids:
        warnings.append(
            f"{len(unknown_ids)} predicted prim(s) have no actionable material."
        )
    if fallback_ids:
        warnings.append(
            f"{len(fallback_ids)} prim(s) use the approved fallback material."
        )
    if unbound_count:
        warnings.append(f"{unbound_count} target prim(s) remain unbound.")
    if extra_prediction_ids:
        warnings.append(
            f"{len(extra_prediction_ids)} prediction record(s) are outside target scope."
        )
    if not prediction_evaluated:
        warnings.append(
            "Prediction coverage was not evaluated because no prediction artifact "
            "was available."
        )
    if not apply_evaluated:
        warnings.append("Binding coverage was not evaluated because apply did not run.")
    elif target_count and not scoped_binding_ids:
        warnings.append(
            "Exact binding prim IDs were unavailable or outside target scope; "
            "aggregate binding counts do not qualify strict readiness."
        )
    if target_count == 0:
        warnings.append("No target prims were available for coverage qualification.")

    return {
        "schema_version": "1.0",
        "policy": policy,
        "readiness_grade": readiness_grade,
        "target_count": target_count,
        "prepared_count": prepared_count,
        "predicted_count": len(prediction_ids),
        "usable_prediction_count": len(usable_ids),
        "unknown_prediction_count": len(unknown_ids),
        "fallback_count": len(fallback_ids),
        "bound_count": bound_count,
        "unbound_count": unbound_count,
        "prediction_coverage_ratio": _ratio(prediction_covered, target_count),
        "binding_coverage_ratio": _ratio(bound_count, target_count),
        "missing_prepared_prim_ids": missing_prepared_ids,
        "missing_prediction_prim_ids": missing_prediction_ids,
        "unknown_prim_ids": sorted(unknown_ids),
        "fallback_prim_ids": sorted(fallback_ids),
        "unbound_prim_ids": sorted(unbound_ids),
        "extra_prediction_prim_ids": extra_prediction_ids,
        "warnings": warnings,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(min(numerator, denominator) / denominator, 6) if denominator else 0.0


def _read_jsonl(path: Path) -> Iterator[Any]:
    try:
        with open(path, encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping invalid coverage JSONL record %s:%s: %s",
                        path,
                        line_number,
                        exc,
                    )
    except OSError as exc:
        logger.warning("Failed to read coverage artifact %s: %s", path, exc)


def _record_id(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    for key in PREDICTION_ID_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.startswith("/"):
            return value
    return None


def _ids_from_jsonl(path: Path) -> set[str]:
    return {
        prim_id
        for record in _read_jsonl(path)
        if (prim_id := _record_id(record)) is not None
    }


def _load_target_prim_ids(session_dir: Path) -> set[str]:
    cache_dir = session_dir / "cache"
    target_ids: set[str] = set()
    if cache_dir.exists():
        for path in sorted(cache_dir.rglob("prims.jsonl")):
            target_ids.update(_ids_from_jsonl(path))
    return target_ids


def _load_prepared_prim_ids(result: Any, session_dir: Path) -> set[str]:
    prepare_output = (result.step_results or {}).get(
        "build_dataset_prepare_dataset", {}
    )
    candidates = [
        prepare_output.get("dataset_jsonl_path"),
        session_dir / "cache" / "dataset" / "dataset.jsonl",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return _ids_from_jsonl(path)
    return set()


def _target_count_hint(step_results: dict[str, Any]) -> int:
    build_output = step_results.get("build_dataset_usd", {})
    return _nonnegative_int(build_output.get("num_prims"))


def _prepared_count_hint(step_results: dict[str, Any]) -> int:
    prepare_output = step_results.get("build_dataset_prepare_dataset", {})
    return _nonnegative_int(prepare_output.get("num_entries"))


def _nonnegative_int(value: object) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


def _restore_namespace_evidence(
    step_results: dict[str, Any],
    completed_steps: set[str],
) -> tuple[bool, bool, dict[str, str], list[str]]:
    """Return validated restored-target namespace evidence.

    The restore task is the canonical owner of GeomSubset and split-dedup target
    paths.  Coverage must not independently recreate that positional mapping.
    """
    restore_ran = "restore_usd" in completed_steps or "restore_usd" in step_results
    if not restore_ran:
        return False, True, {}, []

    warnings: list[str] = []
    restore_output = step_results.get("restore_usd")
    if not isinstance(restore_output, dict):
        return (
            True,
            False,
            {},
            ["Restore ran without structured namespace evidence."],
        )
    if restore_output.get("restore_success") is not True:
        warnings.append("Restore did not report successful namespace translation.")

    restore_stats = restore_output.get("restore_stats")
    if not isinstance(restore_stats, dict):
        warnings.append("Restore target correspondence metadata was unavailable.")
        return True, False, {}, warnings

    raw_mapping_warnings = restore_stats.get("mapping_warnings")
    if isinstance(raw_mapping_warnings, list):
        warnings.extend(
            warning for warning in raw_mapping_warnings if isinstance(warning, str)
        )

    raw_sources = restore_stats.get("restored_prim_sources")
    sources: dict[str, str] = {}
    invalid_entries = 0
    if isinstance(raw_sources, dict):
        for restored_id, source_id in raw_sources.items():
            if (
                isinstance(restored_id, str)
                and restored_id.startswith("/")
                and isinstance(source_id, str)
                and source_id.startswith("/")
            ):
                sources[restored_id] = source_id
            else:
                invalid_entries += 1
    else:
        warnings.append("Restore target correspondence map was malformed.")

    if invalid_entries:
        warnings.append(
            f"Restore target correspondence contained {invalid_entries} invalid "
            "entry or entries."
        )

    expected_target_count = _nonnegative_int(restore_stats.get("expected_target_count"))
    count_matches = bool(
        expected_target_count > 0 and len(sources) == expected_target_count
    )
    if not count_matches:
        warnings.append(
            "Restore target correspondence count was incomplete: "
            f"expected {expected_target_count}, received {len(sources)}."
        )

    mapping_complete = bool(
        restore_output.get("restore_success") is True
        and restore_stats.get("mapping_complete") is True
        and isinstance(raw_sources, dict)
        and not invalid_entries
        and count_matches
    )
    if restore_stats.get("mapping_complete") is not True:
        warnings.append("Restore marked its target correspondence as incomplete.")

    return True, mapping_complete, sources, warnings


def _prediction_path(result: Any, session_dir: Path) -> Path | None:
    step_results = result.step_results or {}
    restore_ran = "restore_usd" in (result.completed_steps or ()) or (
        "restore_usd" in step_results
    )
    if restore_ran:
        restore_output = step_results.get("restore_usd", {})
        raw_path = (
            restore_output.get("restored_predictions_path")
            if isinstance(restore_output, dict)
            else None
        )
        if raw_path and Path(raw_path).exists():
            return Path(raw_path)
        # Once restore ran, optimized predictions are in the wrong namespace.
        return None

    for step_name in reversed(result.completed_steps or list(step_results)):
        if step_name not in {
            "predict",
            "benchmark",
            "expand_cluster_predictions",
            "validate_predictions",
            "harmonize_predictions",
            "create_materials",
        }:
            continue
        raw_path = step_results.get(step_name, {}).get("predictions_path")
        if raw_path and Path(raw_path).exists():
            return Path(raw_path)
    fallback = session_dir / "cache" / "predictions" / "predictions.jsonl"
    return fallback if fallback.exists() else None


def _load_prediction_states(path: Path | None) -> dict[str, str]:
    states: dict[str, str] = {}
    if path is None:
        return states
    for payload in _read_jsonl(path):
        for prim_id, material, is_fallback in _iter_prediction_records(payload):
            if is_fallback or is_fallback_material_name(material):
                states[prim_id] = "fallback"
            elif is_unknown_material_name(material) or is_default_library_fallback_name(
                material
            ):
                states[prim_id] = "unknown"
            elif is_actionable_material_name(material):
                states[prim_id] = "usable"
            else:
                states[prim_id] = "unknown"
    return states


def _iter_prediction_records(
    payload: Any,
    fallback_id: str | None = None,
) -> Iterator[tuple[str, Any, bool]]:
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_prediction_records(item, fallback_id)
        return
    if isinstance(payload, str):
        if fallback_id:
            yield fallback_id, payload, False
        return
    if not isinstance(payload, dict):
        return

    prim_id = _record_id(payload) or fallback_id
    materials = payload.get("materials")
    material: Any = None
    has_material = False
    fallback_marker = bool(payload.get("fallback_source"))
    if isinstance(materials, dict):
        has_material = True
        material = materials.get("material")
        fallback_marker = fallback_marker or bool(materials.get("fallback_source"))
        if any(
            is_disallowed_unknown_validation_status(materials.get(key))
            for key in PREDICTION_VALIDATION_STATUS_KEYS
        ) and not is_actionable_material_name(material):
            material = None
    elif isinstance(materials, str):
        has_material = True
        material = materials
    else:
        for key in PREDICTION_MATERIAL_KEYS:
            if key in payload:
                has_material = True
                material = payload.get(key)
                break

    if prim_id and (has_material or _has_prediction_identity(payload)):
        yield prim_id, material, fallback_marker

    for container_key in PREDICTION_CONTAINER_KEYS:
        container = payload.get(container_key)
        if isinstance(container, dict | list):
            yield from _iter_prediction_records(container, prim_id)

    ignored_keys = {
        *PREDICTION_CONTAINER_KEYS,
        *PREDICTION_ID_KEYS,
        *PREDICTION_MATERIAL_KEYS,
        *PREDICTION_VALIDATION_STATUS_KEYS,
        "materials",
    }
    for key, value in payload.items():
        if key in ignored_keys:
            continue
        child_id = key if isinstance(key, str) and key.startswith("/") else None
        if isinstance(value, dict | list):
            yield from _iter_prediction_records(value, child_id)
        elif child_id and isinstance(value, str):
            yield child_id, value, False


def _has_prediction_identity(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in PREDICTION_ID_KEYS)


def _binding_evidence(
    apply_output: Any,
) -> tuple[set[str], set[str], int, bool]:
    if not isinstance(apply_output, dict):
        return set(), set(), 0, False
    assignment_stats = apply_output.get("assignment_stats")
    if not isinstance(assignment_stats, dict):
        assignment_stats = {}
    exact_binding_ids = isinstance(assignment_stats.get("bound_prim_ids"), list)
    bound_ids = _path_set(assignment_stats.get("bound_prim_ids"))
    unbound_ids = _path_set(assignment_stats.get("unbound_prim_ids"))
    bound_count_hint = _nonnegative_int(assignment_stats.get("total_prims"))

    # Older test/service fixtures represented material -> list[prim ID].  Keep
    # that evidence path compatible without treating real material prim paths
    # (string values) as target bindings.
    materials_applied = apply_output.get("materials_applied")
    if not bound_ids and isinstance(materials_applied, dict):
        for value in materials_applied.values():
            if isinstance(value, list):
                bound_ids.update(_path_set(value))
                exact_binding_ids = True

    return bound_ids, unbound_ids, bound_count_hint, exact_binding_ids


def _path_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str) and item.startswith("/")}
