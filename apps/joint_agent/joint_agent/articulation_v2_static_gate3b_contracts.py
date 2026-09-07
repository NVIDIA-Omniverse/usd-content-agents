# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fail-closed contract for retained Gate 3B aggregate reports."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any, cast


class ArticulationV2StaticGate3BContractError(ValueError):
    """A Gate 3B aggregate report contradicts its producer contract."""


_PRODUCER_STATUS_TOKENS = frozenset({"BLOCKED", "ERROR", "FAIL", "NOT_RUN", "PASS"})
_PRODUCER_HARD_ISSUE_SEVERITIES = frozenset({"ERROR", "FAIL", "FAILED", "FAILURE"})
_FEATURE_FIELDS = frozenset(
    {
        "feature_results",
        "feature_results_sha256",
        "feature_count",
        "feature_passed_count",
        "feature_failed_count",
    }
)
_DEPENDENCY_FIELDS = (
    "artifact_dependency_bundle_schema_version",
    "artifact_dependency_bundle_sha256",
    "artifact_dependency_bundle_entry_count",
)
_REQUIRED_RESULT_FIELDS = frozenset(
    {
        "asset_id",
        "kind",
        "path",
        "original_path",
        "artifact_identity",
        "profile_target",
        *_DEPENDENCY_FIELDS,
        "status",
        "passed",
        "requirement_counts",
        "issue_counts",
        "warnings",
        "errors",
        "needs_rerun",
        "rerun_reasons",
        "report_path",
        "raw_report_path",
        "stdout_log_path",
        "stderr_log_path",
        "foundation_commit",
        "validator_executable",
        "next_step",
    }
)
_OPTIONAL_RESULT_FIELDS = frozenset(
    {
        "artifact_sha256",
        "exception_type",
        "missing_reason",
        *_FEATURE_FIELDS,
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_sha256(document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ArticulationV2StaticGate3BContractError(f"{label} must be an object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ArticulationV2StaticGate3BContractError(f"{label} must be a JSON array")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ArticulationV2StaticGate3BContractError(
            f"{label} must be a canonical nonblank string without padding"
        )
    return value


def _optional_text(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label=label)


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArticulationV2StaticGate3BContractError(
            f"{label} must be a nonnegative integer"
        )
    return cast(int, value)


def _positive_integer(value: Any, *, label: str) -> int:
    count = _nonnegative_integer(value, label=label)
    if count == 0:
        raise ArticulationV2StaticGate3BContractError(
            f"{label} must be a positive integer"
        )
    return count


def _sha256(value: Any, *, label: str) -> str:
    digest = _text(value, label=label)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ArticulationV2StaticGate3BContractError(
            f"{label} must be a lowercase SHA-256"
        )
    return digest


def _string_list(value: Any, *, label: str) -> list[str]:
    items = _list(value, label=label)
    if not all(isinstance(item, str) for item in items):
        raise ArticulationV2StaticGate3BContractError(
            f"{label} must contain only strings"
        )
    return cast(list[str], items)


def _nonnegative_count_map(value: Any, *, label: str) -> dict[str, int]:
    raw_counts = _mapping(value, label=label)
    counts: dict[str, int] = {}
    for raw_name, raw_count in raw_counts.items():
        name = _text(raw_name, label=f"{label} key")
        counts[name] = _nonnegative_integer(raw_count, label=f"{label} {name!r}")
    return counts


def _status_count_map(value: Any, *, label: str) -> dict[str, int]:
    raw_counts = _mapping(value, label=label)
    counts: dict[str, int] = {}
    for raw_status, raw_count in raw_counts.items():
        status = _text(raw_status, label=f"{label} key")
        if status not in _PRODUCER_STATUS_TOKENS:
            raise ArticulationV2StaticGate3BContractError(
                f"{label} status {status!r} is not a producer status token"
            )
        counts[status] = _positive_integer(raw_count, label=f"{label} {status!r}")
    return counts


def _optional_path(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ArticulationV2StaticGate3BContractError(
            f"{label} must be a string or null"
        )
    return value


def _artifact_identity(path: str) -> str | None:
    if not path:
        return None
    basename = path.rstrip("/").rsplit("/", maxsplit=1)[-1]
    parsed = PurePath(basename)
    if parsed.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}:
        return parsed.stem
    return basename


def _validate_dependency_identity(entry: Mapping[str, Any], *, label: str) -> bool:
    values = tuple(entry.get(field) for field in _DEPENDENCY_FIELDS)
    if all(value is None for value in values):
        return False
    if any(value is None for value in values):
        raise ArticulationV2StaticGate3BContractError(
            f"{label} dependency-bundle identity is incomplete"
        )
    _text(values[0], label=f"{label} {_DEPENDENCY_FIELDS[0]}")
    _sha256(values[1], label=f"{label} {_DEPENDENCY_FIELDS[1]}")
    _positive_integer(values[2], label=f"{label} {_DEPENDENCY_FIELDS[2]}")
    return True


def _validate_feature_projection(
    entry: Mapping[str, Any],
    *,
    label: str,
    expected_feature_ids: tuple[str, ...],
) -> int | None:
    present = _FEATURE_FIELDS.intersection(entry)
    if not present:
        return None
    if present != _FEATURE_FIELDS:
        missing = tuple(sorted(_FEATURE_FIELDS - present))
        raise ArticulationV2StaticGate3BContractError(
            f"{label} feature projection is incomplete; missing {missing}"
        )

    raw_features = _list(
        entry.get("feature_results"),
        label=f"{label} feature_results",
    )
    if not raw_features:
        raise ArticulationV2StaticGate3BContractError(
            f"{label} feature_results must not be empty"
        )
    features: list[dict[str, Any]] = []
    feature_ids: list[str] = []
    for index, raw_feature in enumerate(raw_features):
        feature = _mapping(raw_feature, label=f"{label} feature_results[{index}]")
        expected_keys = {"id", "version", "passed"}
        if set(feature) != expected_keys:
            raise ArticulationV2StaticGate3BContractError(
                f"{label} feature_results[{index}] fields differ from the producer"
            )
        feature_id = _text(
            feature.get("id"),
            label=f"{label} feature_results[{index}].id",
        )
        version = _text(
            feature.get("version"),
            label=f"{label} feature_results[{index}].version",
        )
        passed = feature.get("passed")
        if not isinstance(passed, bool):
            raise ArticulationV2StaticGate3BContractError(
                f"{label} feature_results[{index}].passed must be boolean"
            )
        feature_ids.append(feature_id)
        features.append({"id": feature_id, "version": version, "passed": passed})
    if len(feature_ids) != len(set(feature_ids)):
        raise ArticulationV2StaticGate3BContractError(
            f"{label} feature IDs must be unique"
        )
    if tuple(feature_ids) != expected_feature_ids:
        raise ArticulationV2StaticGate3BContractError(
            f"{label} feature roster differs from the approved ordered profile roster"
        )

    passed_count = sum(1 for feature in features if feature["passed"] is True)
    failed_count = len(features) - passed_count
    expected_counts = {
        "feature_count": len(features),
        "feature_passed_count": passed_count,
        "feature_failed_count": failed_count,
    }
    for field, expected in expected_counts.items():
        observed = _nonnegative_integer(entry.get(field), label=f"{label} {field}")
        if observed != expected:
            raise ArticulationV2StaticGate3BContractError(
                f"{label} {field} does not match feature_results"
            )
    if entry.get("feature_results_sha256") != _canonical_sha256(
        {"feature_results": features}
    ):
        raise ArticulationV2StaticGate3BContractError(
            f"{label} feature_results_sha256 mismatch"
        )
    return failed_count


def _validate_result(
    entry: Mapping[str, Any],
    *,
    label: str,
    expected_feature_ids: tuple[str, ...],
) -> str:
    missing = _REQUIRED_RESULT_FIELDS - entry.keys()
    unexpected = entry.keys() - (_REQUIRED_RESULT_FIELDS | _OPTIONAL_RESULT_FIELDS)
    if missing or unexpected:
        raise ArticulationV2StaticGate3BContractError(
            f"{label} fields differ from the producer; missing={tuple(sorted(missing))}, "
            f"unexpected={tuple(sorted(unexpected))}"
        )

    _text(entry.get("asset_id"), label=f"{label} asset_id")
    kind = _text(entry.get("kind"), label=f"{label} kind")
    if kind not in {"generated", "reference"}:
        raise ArticulationV2StaticGate3BContractError(
            f"{label} kind is not a producer target kind"
        )
    path = entry.get("path")
    original_path = entry.get("original_path")
    if not isinstance(path, str) or not isinstance(original_path, str):
        raise ArticulationV2StaticGate3BContractError(
            f"{label} path and original_path must be strings"
        )
    expected_artifact_identity = _artifact_identity(path)
    if entry.get("artifact_identity") != expected_artifact_identity:
        raise ArticulationV2StaticGate3BContractError(
            f"{label} artifact_identity does not match path"
        )
    _text(entry.get("profile_target"), label=f"{label} profile_target")
    has_dependency_identity = _validate_dependency_identity(entry, label=label)

    if "artifact_sha256" in entry and entry.get("artifact_sha256") is not None:
        _sha256(entry.get("artifact_sha256"), label=f"{label} artifact_sha256")
    status = _text(entry.get("status"), label=f"{label} status")
    if status not in _PRODUCER_STATUS_TOKENS:
        raise ArticulationV2StaticGate3BContractError(
            f"{label} status {status!r} is not a producer status token"
        )
    passed = entry.get("passed")
    if not isinstance(passed, bool):
        raise ArticulationV2StaticGate3BContractError(f"{label} passed must be boolean")
    if passed != (status == "PASS"):
        raise ArticulationV2StaticGate3BContractError(
            f"{label} status/passed relationship is inconsistent"
        )

    _nonnegative_count_map(
        entry.get("requirement_counts"),
        label=f"{label} requirement_counts",
    )
    issue_counts = _nonnegative_count_map(
        entry.get("issue_counts"),
        label=f"{label} issue_counts",
    )
    _string_list(entry.get("warnings"), label=f"{label} warnings")
    errors = _string_list(entry.get("errors"), label=f"{label} errors")
    needs_rerun = entry.get("needs_rerun")
    if not isinstance(needs_rerun, bool):
        raise ArticulationV2StaticGate3BContractError(
            f"{label} needs_rerun must be boolean"
        )
    rerun_reasons = _string_list(
        entry.get("rerun_reasons"),
        label=f"{label} rerun_reasons",
    )
    if rerun_reasons != sorted(set(rerun_reasons)):
        raise ArticulationV2StaticGate3BContractError(
            f"{label} rerun_reasons must be sorted and unique"
        )
    for field in (
        "report_path",
        "raw_report_path",
        "stdout_log_path",
        "stderr_log_path",
    ):
        _optional_path(entry.get(field), label=f"{label} {field}")
    foundation_commit = _optional_text(
        entry.get("foundation_commit"),
        label=f"{label} foundation_commit",
    )
    validator_executable = _optional_text(
        entry.get("validator_executable"),
        label=f"{label} validator_executable",
    )
    _text(entry.get("next_step"), label=f"{label} next_step")
    failed_features = _validate_feature_projection(
        entry,
        label=label,
        expected_feature_ids=expected_feature_ids,
    )

    if failed_features is not None and (
        not has_dependency_identity
        or foundation_commit is None
        or validator_executable is None
    ):
        raise ArticulationV2StaticGate3BContractError(
            f"{label} executed feature evidence lacks dependency/runtime identity"
        )
    if status in {"PASS", "FAIL"} and failed_features is None:
        raise ArticulationV2StaticGate3BContractError(
            f"{label} completed status requires feature evidence"
        )
    if status == "PASS" and failed_features:
        raise ArticulationV2StaticGate3BContractError(
            f"{label} PASS contradicts failed feature evidence"
        )
    if status == "PASS" and any(
        issue_counts.get(severity, 0) for severity in _PRODUCER_HARD_ISSUE_SEVERITIES
    ):
        raise ArticulationV2StaticGate3BContractError(
            f"{label} PASS contradicts hard issue counts"
        )
    if status in {"BLOCKED", "NOT_RUN"} and failed_features is not None:
        raise ArticulationV2StaticGate3BContractError(
            f"{label} {status} must not claim executed feature evidence"
        )
    elif status == "ERROR" and not any(message.strip() for message in errors):
        raise ArticulationV2StaticGate3BContractError(
            f"{label} ERROR has no retained error message"
        )

    if status == "NOT_RUN":
        _text(entry.get("missing_reason"), label=f"{label} missing_reason")
    elif "missing_reason" in entry:
        raise ArticulationV2StaticGate3BContractError(
            f"{label} missing_reason is only valid for NOT_RUN"
        )
    if "exception_type" in entry:
        if status != "ERROR":
            raise ArticulationV2StaticGate3BContractError(
                f"{label} exception_type is only valid for ERROR"
            )
        _text(entry.get("exception_type"), label=f"{label} exception_type")
    return status


def _aggregate_status(status_counts: Mapping[str, int], result_count: int) -> str:
    if result_count == 0:
        return "NOT_RUN"
    for status in ("ERROR", "BLOCKED", "NOT_RUN", "FAIL"):
        if status_counts.get(status, 0):
            return status
    return "PASS"


def validate_articulation_v2_static_gate3b_report_consistency(
    document: Mapping[str, Any],
    *,
    profile_id: str,
    feature_ids: tuple[str, ...],
) -> None:
    """Recompute ordered rows, row projections, digest, and aggregate fields."""

    approved_profile_id = _text(profile_id, label="approved Gate 3B profile_id")
    approved_profile_name, separator, approved_profile_version = (
        approved_profile_id.rpartition("@")
    )
    if not separator or not approved_profile_name or not approved_profile_version:
        raise ArticulationV2StaticGate3BContractError(
            "approved Gate 3B profile_id must use name@version"
        )
    approved_feature_ids = tuple(
        _text(feature_id, label="approved Gate 3B feature ID")
        for feature_id in feature_ids
    )
    if not approved_feature_ids or len(approved_feature_ids) != len(
        set(approved_feature_ids)
    ):
        raise ArticulationV2StaticGate3BContractError(
            "approved Gate 3B feature roster must be nonempty and unique"
        )

    profile = _mapping(document.get("profile"), label="Gate 3B profile")
    expected_profile = {
        "name": approved_profile_name,
        "version": approved_profile_version,
        "target": approved_profile_id,
    }
    if profile != expected_profile:
        raise ArticulationV2StaticGate3BContractError(
            "Gate 3B profile differs from the approved release profile"
        )

    preflight = _mapping(document.get("preflight"), label="Gate 3B preflight")
    preflight_passed = preflight.get("passed")
    if not isinstance(preflight_passed, bool):
        raise ArticulationV2StaticGate3BContractError(
            "Gate 3B preflight passed must be boolean"
        )
    preflight_status = _text(
        preflight.get("status"),
        label="Gate 3B preflight status",
    )
    if preflight_status not in _PRODUCER_STATUS_TOKENS - {"FAIL"}:
        raise ArticulationV2StaticGate3BContractError(
            "Gate 3B preflight status is not a producer preflight token"
        )
    if preflight_passed != (preflight_status == "PASS"):
        raise ArticulationV2StaticGate3BContractError(
            "Gate 3B preflight status/passed relationship is inconsistent"
        )
    available_profiles = tuple(
        _text(item, label="Gate 3B preflight available profile")
        for item in _list(
            preflight.get("available_profiles"),
            label="Gate 3B preflight available_profiles",
        )
    )
    if approved_profile_name not in available_profiles:
        raise ArticulationV2StaticGate3BContractError(
            "Gate 3B preflight available_profiles omits the approved selected profile"
        )

    results = tuple(
        _mapping(item, label=f"Gate 3B result {index}")
        for index, item in enumerate(
            _list(document.get("results"), label="Gate 3B results")
        )
    )
    result_count = _nonnegative_integer(
        document.get("result_count"),
        label="Gate 3B result_count",
    )
    if result_count != len(results):
        raise ArticulationV2StaticGate3BContractError(
            "Gate 3B result_count does not match the ordered results"
        )
    if document.get("results_sha256") != _canonical_sha256({"results": results}):
        raise ArticulationV2StaticGate3BContractError("Gate 3B results_sha256 mismatch")

    status_counts: Counter[str] = Counter()
    foundation_commits: set[str] = set()
    validator_executables: set[str] = set()
    for index, result in enumerate(results):
        status_counts[
            _validate_result(
                result,
                label=f"Gate 3B result {index}",
                expected_feature_ids=approved_feature_ids,
            )
        ] += 1
        foundation_commit = result.get("foundation_commit")
        if foundation_commit:
            foundation_commits.add(str(foundation_commit))
        validator_executable = result.get("validator_executable")
        if validator_executable:
            validator_executables.add(str(validator_executable))

    if not preflight_passed and set(status_counts) - {preflight_status}:
        raise ArticulationV2StaticGate3BContractError(
            "Gate 3B completed or substituted results contradict failed preflight"
        )

    aggregate = _mapping(document.get("aggregate"), label="Gate 3B aggregate")
    expected_aggregate_fields = {
        "status",
        "status_counts",
        "foundation_commits",
        "validator_executables",
    }
    if set(aggregate) != expected_aggregate_fields:
        raise ArticulationV2StaticGate3BContractError(
            "Gate 3B aggregate fields differ from the producer"
        )
    declared_status_counts = _status_count_map(
        aggregate.get("status_counts"),
        label="Gate 3B aggregate status_counts",
    )
    expected_status_counts = dict(sorted(status_counts.items()))
    if declared_status_counts != expected_status_counts:
        raise ArticulationV2StaticGate3BContractError(
            "Gate 3B aggregate status_counts do not match results"
        )
    aggregate_status = _text(
        aggregate.get("status"),
        label="Gate 3B aggregate status",
    )
    if aggregate_status != _aggregate_status(expected_status_counts, result_count):
        raise ArticulationV2StaticGate3BContractError(
            "Gate 3B aggregate status does not match results"
        )

    declared_commits = _string_list(
        aggregate.get("foundation_commits"),
        label="Gate 3B aggregate foundation_commits",
    )
    if declared_commits != sorted(foundation_commits):
        raise ArticulationV2StaticGate3BContractError(
            "Gate 3B aggregate foundation_commits do not match results"
        )
    declared_executables = _string_list(
        aggregate.get("validator_executables"),
        label="Gate 3B aggregate validator_executables",
    )
    if declared_executables != sorted(validator_executables):
        raise ArticulationV2StaticGate3BContractError(
            "Gate 3B aggregate validator_executables do not match results"
        )


__all__ = [
    "ArticulationV2StaticGate3BContractError",
    "validate_articulation_v2_static_gate3b_report_consistency",
]
