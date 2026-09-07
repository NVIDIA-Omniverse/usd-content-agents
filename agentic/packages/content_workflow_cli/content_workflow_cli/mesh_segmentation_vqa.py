# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation of the agent's own selection evidence for mesh segmentation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _resolve_run_file(run_dir: Path, raw_path: str, *, label: str) -> Path:
    candidate = Path(raw_path)
    path = (candidate if candidate.is_absolute() else run_dir / candidate).resolve()
    if not path.is_relative_to(run_dir):
        raise ValueError(f"{label} escapes the isolated run directory: {raw_path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {raw_path}")
    return path


def _resolve_run_artifact(run_dir: Path, raw_path: str) -> Path | None:
    candidate = Path(raw_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    try:
        direct = _resolve_run_file(
            run_dir,
            raw_path,
            label="Run artifact",
        )
    except (ValueError, FileNotFoundError):
        direct = None
    if direct is not None:
        return direct
    suffix = candidate.as_posix().lstrip("/")
    suffix_matches = [
        path
        for path in run_dir.glob("hypotheses/**/*.json")
        if path.as_posix().endswith(suffix)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise ValueError(
            f"Run artifact path is ambiguous: {raw_path!r} matched "
            f"{len(suffix_matches)} hypothesis artifacts"
        )
    basename_matches = [
        path
        for path in run_dir.glob("hypotheses/**/*.json")
        if path.name == candidate.name
    ]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        raise ValueError(
            f"Run artifact basename is ambiguous: {raw_path!r} matched "
            f"{len(basename_matches)} hypothesis artifacts"
        )
    return None


def _directions_are_independent(events: list[dict[str, Any]]) -> bool:
    directions: list[tuple[float, float, float]] = []
    for event in events:
        raw = event.get("ray_direction")
        if not isinstance(raw, list) or len(raw) != 3:
            continue
        vector = tuple(float(value) for value in raw)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1.0e-12:
            continue
        directions.append(tuple(value / norm for value in vector))
    for first_index, first in enumerate(directions):
        for second in directions[first_index + 1 :]:
            # At least 90 degrees prevents nearby views from masquerading as
            # independent front/back evidence.
            if sum(a * b for a, b in zip(first, second, strict=True)) <= 0.0:
                return True
    return False


def _validate_selection_events(
    *,
    target_name: str,
    events: list[dict[str, Any]],
    expected_instance_count: int | None = None,
) -> list[str]:
    errors: list[str] = []
    locked_patch_ids = {
        str(event.get("target_patch_id", event.get("candidate_component_id")))
        for event in events
        if event.get("operation") == "lock_candidate_component_at_pick"
    }
    locked_patch_ids.discard("None")
    locked_patch_ids.discard("-1")
    if not locked_patch_ids:
        return [f"Target {target_name} has no pixel-picked locked patch identities"]
    if (
        isinstance(expected_instance_count, int)
        and expected_instance_count > 0
        and len(locked_patch_ids) != expected_instance_count
    ):
        errors.append(
            f"Target {target_name} expected {expected_instance_count} instances, "
            f"but pixel picks locked {len(locked_patch_ids)} patches"
        )

    failed_probes = [
        event
        for event in events
        if str(event.get("operation", "")).startswith("verify_")
        and event.get("probe_passed") is not True
    ]
    if failed_probes:
        errors.append(
            f"Target {target_name} has {len(failed_probes)} failed face probes"
        )

    for patch_id in sorted(locked_patch_ids):
        patch_events = [
            event
            for event in events
            if str(
                event.get(
                    "target_patch_id",
                    event.get("candidate_component_id"),
                )
            )
            == patch_id
        ]
        target_probes = [
            event
            for event in patch_events
            if event.get("operation") == "verify_target_at_pick"
            and event.get("probe_passed") is True
        ]
        other_probes = [
            event
            for event in patch_events
            if event.get("operation") == "verify_other_at_pick"
            and event.get("probe_passed") is True
        ]
        if len(target_probes) < 2 or not _directions_are_independent(target_probes):
            errors.append(
                f"Target {target_name} patch {patch_id} lacks opposing "
                "verify_target picks"
            )
        if len(other_probes) < 2 or not _directions_are_independent(other_probes):
            errors.append(
                f"Target {target_name} patch {patch_id} lacks opposing "
                "boundary-adjacent verify_other picks"
            )
    return errors


def validate_part_lock_selection_evidence(
    run_dir: Path,
    *,
    part: dict[str, Any],
    lock: dict[str, Any],
) -> list[str]:
    """Validate picker locks and opposing boundary probes for one queued part."""

    target_name = str(part.get("name", part.get("part_name", "target")))
    raw_paths = lock.get("selection_events")
    if not isinstance(raw_paths, list) or not raw_paths:
        return [f"Target {target_name} lock lacks selection_events"]
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    for raw_path in raw_paths:
        try:
            path = _resolve_run_artifact(run_dir, str(raw_path))
        except ValueError as exc:
            errors.append(f"Target {target_name} selection events are ambiguous: {exc}")
            continue
        if path is None:
            errors.append(
                f"Target {target_name} selection events could not be resolved: "
                f"{raw_path}"
            )
            continue
        try:
            payload = _load_json_object(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Could not parse selection events {path}: {exc}")
            continue
        events.extend(
            event for event in payload.get("events", []) if isinstance(event, dict)
        )
    expected = part.get("expected_instance_count")
    return [
        *errors,
        *_validate_selection_events(
            target_name=target_name,
            events=events,
            expected_instance_count=expected if isinstance(expected, int) else None,
        ),
    ]


def _validate_target_evidence(
    run_dir: Path,
    *,
    target_name: str,
    validation_reference: object,
) -> list[str]:
    """Bind one locked target to its passing falsification validation."""

    # Resolution compares against an absolute run root, so normalize first.
    run_dir = run_dir.resolve()
    reference = str(validation_reference)
    # The contract is run-relative, but agents naturally write the path
    # relative to `final/target_evidence.json`, which yields `../part/...`.
    # Accept that spelling by collapsing it against `final/` first, still
    # refusing anything that escapes the run directory.
    if reference.startswith("../"):
        candidate = (run_dir / "final" / reference).resolve()
        if candidate.is_relative_to(run_dir) and candidate.is_file():
            reference = str(candidate.relative_to(run_dir))
    try:
        path = _resolve_run_artifact(run_dir, reference)
    except ValueError as exc:
        return [f"Target {target_name} falsification validation is ambiguous: {exc}"]
    if path is None:
        return [
            f"Target {target_name} falsification validation could not be resolved: "
            f"{validation_reference}"
        ]
    try:
        payload = _load_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"Could not parse falsification validation for {target_name}: {exc}"]
    errors: list[str] = []
    if payload.get("status") != "passed":
        errors.append(f"Target {target_name} falsification validation did not pass")
    # A missing or non-string field used to bind this validation to any target.
    # Terminal validation already requires `semantic_part == target`, so this
    # only ever read stricter than it was; make it actually say what it means.
    recorded = payload.get("semantic_part")
    if not isinstance(recorded, str):
        errors.append(
            f"Target {target_name} falsification validation has no semantic_part"
        )
    elif recorded != target_name:
        errors.append(
            f"Target {target_name} falsification validation names {recorded!r}"
        )
    return errors


def validate_targeted_selection_evidence(
    run_dir: Path,
    *,
    authoritative_request: dict[str, object] | None = None,
) -> list[str]:
    """Require per-patch opposing target/other face probes before lock."""

    evidence_path = run_dir / "final" / "target_evidence.json"
    if not evidence_path.is_file():
        return ["Targeted segmentation is missing final/target_evidence.json"]
    try:
        evidence = _load_json_object(evidence_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"Could not parse targeted selection evidence: {exc}"]

    raw_targets = evidence.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        return ["Targeted selection evidence must contain at least one target"]

    errors: list[str] = []
    target_names: list[str] = []
    for target in raw_targets:
        if not isinstance(target, dict):
            errors.append("Target evidence contains a non-object target record")
            continue
        raw_target_name = target.get("target_semantic_part")
        if not isinstance(raw_target_name, str) or not raw_target_name:
            errors.append("Target evidence contains an unnamed target record")
            continue
        target_names.append(raw_target_name)
        if target.get("status") != "locked":
            errors.append(f"Target {raw_target_name} is not locked")
            continue
        target_name = raw_target_name
        # This workflow locks whole frozen fragments through immutable revisions and never
        # produces the v4 `lock_candidate_component_at_pick` pixel-patch events.
        # When a target carries falsification evidence instead, validate that
        # evidence rather than demanding a protocol this run does not execute.
        recorded_validation = target.get("falsification_validation")
        if not target.get("selection_events") and recorded_validation is not None:
            errors.extend(
                _validate_target_evidence(
                    run_dir,
                    target_name=target_name,
                    validation_reference=recorded_validation,
                )
            )
            continue
        events: list[dict[str, Any]] = []
        for raw_path in target.get("selection_events", []):
            try:
                path = _resolve_run_artifact(run_dir, str(raw_path))
            except ValueError as exc:
                errors.append(
                    f"Target {target_name} selection events are ambiguous: {exc}"
                )
                continue
            if path is None:
                errors.append(
                    f"Target {target_name} selection events could not be resolved: "
                    f"{raw_path}"
                )
                continue
            try:
                payload = _load_json_object(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"Could not parse selection events {path}: {exc}")
                continue
            events.extend(
                event for event in payload.get("events", []) if isinstance(event, dict)
            )

        errors.extend(
            _validate_selection_events(
                target_name=target_name,
                events=events,
            )
        )
    if len(target_names) != len(set(target_names)):
        errors.append("Targeted selection evidence contains duplicate target names")
    request = authoritative_request
    if request is None:
        request_path = run_dir / "request.json"
    else:
        request_path = None
    if request is not None or (request_path is not None and request_path.is_file()):
        try:
            if request is None:
                assert request_path is not None
                request = _load_json_object(request_path)
            assert request is not None
            inputs = request.get("inputs")
            expected_targets = (
                inputs.get("target_semantic_parts")
                if isinstance(inputs, dict)
                else None
            )
            if isinstance(expected_targets, list) and target_names != expected_targets:
                errors.append(
                    "Targeted selection evidence names do not match request.json"
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Could not parse request.json target vocabulary: {exc}")
    return errors
