# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fail-closed contract for retained Gate 3A aggregate reports."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any, cast


class ArticulationV2StaticGate3AContractError(ValueError):
    """A Gate 3A aggregate report contradicts its producer contract."""


_PRODUCER_SEVERITY_TOKENS = frozenset(
    {"INFO", "WARNING", "FAILURE", "ERROR", "UNKNOWN"}
)
_PRODUCER_STATUS_TOKENS = frozenset(
    {"pass", "warning", "fail", "validator_exception", "not_run"}
)
ARTICULATION_V2_STATIC_GATE3A_COMPLETED_RESULT_FIELDS = frozenset(
    {
        "artifact_dependency_bundle_entry_count",
        "artifact_dependency_bundle_schema_version",
        "artifact_dependency_bundle_sha256",
        "artifact_identity",
        "artifact_sha256",
        "asset_id",
        "category_counts",
        "elapsed_seconds",
        "hard_failure_count",
        "issue_count",
        "issues",
        "issues_truncated",
        "kind",
        "original_path",
        "path",
        "resolver_working_directory",
        "rule_counts",
        "severity_counts",
        "status",
    }
)
_PRODUCER_RESULT_FIELDS = ARTICULATION_V2_STATIC_GATE3A_COMPLETED_RESULT_FIELDS | {
    "exception_message",
    "exception_type",
    "missing_reason",
}


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
        raise ArticulationV2StaticGate3AContractError(f"{label} must be a JSON object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ArticulationV2StaticGate3AContractError(f"{label} must be a JSON array")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ArticulationV2StaticGate3AContractError(
            f"{label} must be a canonical nonblank string without padding"
        )
    return value


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArticulationV2StaticGate3AContractError(
            f"{label} must be a nonnegative integer"
        )
    return cast(int, value)


def _severity_count_map(value: Any, *, label: str) -> dict[str, int]:
    counts = _mapping(value, label=label)
    validated: dict[str, int] = {}
    for raw_name, raw_count in counts.items():
        name = _text(raw_name, label=f"{label} key")
        if name not in _PRODUCER_SEVERITY_TOKENS:
            raise ArticulationV2StaticGate3AContractError(
                f"{label} severity {name!r} is not a producer severity token"
            )
        count = _nonnegative_integer(raw_count, label=f"{label} {name!r}")
        if count == 0:
            raise ArticulationV2StaticGate3AContractError(
                f"{label} must omit zero-valued severity {name!r}"
            )
        validated[name] = count
    return validated


def _status_count_map(value: Any, *, label: str) -> dict[str, int]:
    counts = _mapping(value, label=label)
    validated: dict[str, int] = {}
    for raw_name, raw_count in counts.items():
        name = _text(raw_name, label=f"{label} key")
        if name not in _PRODUCER_STATUS_TOKENS:
            raise ArticulationV2StaticGate3AContractError(
                f"{label} status {name!r} is not a producer status token"
            )
        count = _nonnegative_integer(raw_count, label=f"{label} {name!r}")
        if count == 0:
            raise ArticulationV2StaticGate3AContractError(
                f"{label} must omit zero-valued status {name!r}"
            )
        validated[name] = count
    return validated


def _nested_severity_count_map(
    value: Any,
    *,
    label: str,
) -> dict[str, dict[str, int]]:
    raw_counts = _mapping(value, label=label)
    validated: dict[str, dict[str, int]] = {}
    for raw_name, raw_severity_counts in raw_counts.items():
        name = _text(raw_name, label=f"{label} key")
        severity_counts = _severity_count_map(
            raw_severity_counts,
            label=f"{label} {name!r}",
        )
        if not severity_counts:
            raise ArticulationV2StaticGate3AContractError(
                f"{label} must omit empty count row {name!r}"
            )
        validated[name] = severity_counts
    return validated


def _registered_rule_categories(document: Mapping[str, Any]) -> dict[str, str]:
    registered = _mapping(
        document.get("registered_rules_by_category"),
        label="Gate 3A registered_rules_by_category",
    )
    rule_to_category: dict[str, str] = {}
    for raw_category, raw_rules in registered.items():
        category = _text(
            raw_category,
            label="Gate 3A registered rule category",
        )
        rules = _list(
            raw_rules,
            label=f"Gate 3A registered rules for {category!r}",
        )
        for index, raw_rule in enumerate(rules):
            rule = _text(
                raw_rule,
                label=f"Gate 3A registered rule {category!r}[{index}]",
            )
            if rule in rule_to_category:
                raise ArticulationV2StaticGate3AContractError(
                    f"Gate 3A registered rule {rule!r} appears in more than one "
                    "category"
                )
            rule_to_category[rule] = category
    return rule_to_category


def _findings(
    value: Any,
    *,
    label: str,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        _mapping(item, label=f"{label} entry") for item in _list(value, label=label)
    )


def _retained_issue_counts(
    findings: tuple[Mapping[str, Any], ...],
    *,
    label: str,
) -> tuple[Counter[str], dict[str, Counter[str]]]:
    severity_counts: Counter[str] = Counter()
    rule_counts: dict[str, Counter[str]] = {}
    for index, finding in enumerate(findings):
        raw_severity = finding.get("severity")
        if raw_severity is None:
            severity = "UNKNOWN"
        else:
            severity = _text(
                raw_severity,
                label=f"{label} issue {index} severity",
            )
            if severity not in _PRODUCER_SEVERITY_TOKENS:
                raise ArticulationV2StaticGate3AContractError(
                    f"{label} issue {index} severity {severity!r} is not a "
                    "producer severity token"
                )
        raw_rule = finding.get("rule")
        rule = (
            "UNKNOWN"
            if raw_rule is None
            else _text(raw_rule, label=f"{label} issue {index} rule")
        )
        severity_counts[severity] += 1
        rule_counts.setdefault(rule, Counter())[severity] += 1
    return severity_counts, rule_counts


def _flatten_severity_counts(
    nested: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
    flattened: Counter[str] = Counter()
    for counts in nested.values():
        flattened.update(counts)
    return dict(sorted(flattened.items()))


def _category_counts_from_rule_counts(
    rule_counts: Mapping[str, Mapping[str, int]],
    *,
    rule_to_category: Mapping[str, str],
) -> dict[str, dict[str, int]]:
    category_counts: dict[str, Counter[str]] = {}
    for rule, counts in rule_counts.items():
        category = rule_to_category.get(rule, "UNKNOWN")
        category_counts.setdefault(category, Counter()).update(counts)
    return {
        category: dict(sorted(counts.items()))
        for category, counts in sorted(category_counts.items())
    }


def _require_retained_counts_within_declared(
    observed: Mapping[str, Mapping[str, int]],
    declared: Mapping[str, Mapping[str, int]],
    *,
    label: str,
) -> None:
    exceeded = {
        (name, severity): count
        for name, counts in observed.items()
        for severity, count in counts.items()
        if count > declared.get(name, {}).get(severity, 0)
    }
    if exceeded:
        raise ArticulationV2StaticGate3AContractError(
            f"{label} retained issue counts exceed declared counts: {exceeded}"
        )


def _validate_result(
    entry: Mapping[str, Any],
    *,
    label: str,
    rule_to_category: Mapping[str, str],
) -> tuple[str, dict[str, int]]:
    unexpected = entry.keys() - _PRODUCER_RESULT_FIELDS
    if unexpected:
        raise ArticulationV2StaticGate3AContractError(
            f"{label} has fields outside the producer contract: "
            f"{tuple(sorted(unexpected))}"
        )
    raw_status = _text(entry.get("status"), label=f"{label} status")
    issue_count = _nonnegative_integer(
        entry.get("issue_count"),
        label=f"{label} issue_count",
    )
    hard_failure_count = _nonnegative_integer(
        entry.get("hard_failure_count"),
        label=f"{label} hard_failure_count",
    )
    severity_counts = _severity_count_map(
        entry.get("severity_counts"),
        label=f"{label} severity_counts",
    )
    findings = _findings(entry.get("issues"), label=f"{label} issues")

    if raw_status == "validator_exception":
        if (
            issue_count != 0
            or hard_failure_count != 1
            or severity_counts != {"ERROR": 1}
            or findings
            or _mapping(entry.get("rule_counts"), label=f"{label} rule_counts")
            or _mapping(entry.get("category_counts"), label=f"{label} category_counts")
            or "issues_truncated" in entry
        ):
            raise ArticulationV2StaticGate3AContractError(
                f"{label} validator_exception fields do not match the producer"
            )
        _text(entry.get("exception_type"), label=f"{label} exception_type")
        if not isinstance(entry.get("exception_message"), str):
            raise ArticulationV2StaticGate3AContractError(
                f"{label} exception_message must be a string"
            )
        return raw_status, severity_counts

    if raw_status == "not_run":
        if (
            issue_count != 0
            or hard_failure_count != 0
            or severity_counts
            or findings
            or _mapping(entry.get("rule_counts"), label=f"{label} rule_counts")
            or _mapping(entry.get("category_counts"), label=f"{label} category_counts")
            or "issues_truncated" in entry
        ):
            raise ArticulationV2StaticGate3AContractError(
                f"{label} not_run fields do not match the producer"
            )
        _text(entry.get("missing_reason"), label=f"{label} missing_reason")
        return raw_status, severity_counts

    if raw_status not in {"pass", "warning", "fail"}:
        raise ArticulationV2StaticGate3AContractError(
            f"Gate 3A result status {raw_status!r} is not a completed outcome"
        )

    truncated = 0
    if "issues_truncated" in entry:
        truncated = _nonnegative_integer(
            entry.get("issues_truncated"),
            label=f"{label} issues_truncated",
        )
        if truncated == 0:
            raise ArticulationV2StaticGate3AContractError(
                f"{label} must omit a zero issues_truncated field"
            )
    if issue_count != len(findings) + truncated:
        raise ArticulationV2StaticGate3AContractError(
            f"{label} issue_count/truncation does not match retained issues"
        )
    if issue_count != sum(severity_counts.values()):
        raise ArticulationV2StaticGate3AContractError(
            f"{label} issue_count does not match severity_counts"
        )

    expected_hard = severity_counts.get("FAILURE", 0) + severity_counts.get("ERROR", 0)
    if hard_failure_count != expected_hard:
        raise ArticulationV2StaticGate3AContractError(
            f"{label} hard_failure_count does not match hard severities"
        )
    expected_status = (
        "fail"
        if expected_hard
        else "warning"
        if severity_counts.get("WARNING", 0)
        else "pass"
    )
    if raw_status != expected_status:
        raise ArticulationV2StaticGate3AContractError(
            f"{label} status does not match its severity_counts"
        )

    rule_counts = _nested_severity_count_map(
        entry.get("rule_counts"),
        label=f"{label} rule_counts",
    )
    if _flatten_severity_counts(rule_counts) != severity_counts:
        raise ArticulationV2StaticGate3AContractError(
            f"{label} rule_counts do not flatten to severity_counts"
        )
    category_counts = _nested_severity_count_map(
        entry.get("category_counts"),
        label=f"{label} category_counts",
    )
    if _flatten_severity_counts(category_counts) != severity_counts:
        raise ArticulationV2StaticGate3AContractError(
            f"{label} category_counts do not flatten to severity_counts"
        )
    derived_category_counts = _category_counts_from_rule_counts(
        rule_counts,
        rule_to_category=rule_to_category,
    )
    if category_counts != derived_category_counts:
        raise ArticulationV2StaticGate3AContractError(
            f"{label} category_counts do not match registered rule categories"
        )

    retained_severity_counts, retained_rule_counts = _retained_issue_counts(
        findings,
        label=label,
    )
    exceeded_severities = {
        severity: count
        for severity, count in retained_severity_counts.items()
        if count > severity_counts.get(severity, 0)
    }
    if exceeded_severities:
        raise ArticulationV2StaticGate3AContractError(
            f"{label} retained issue severities exceed declared counts: "
            f"{exceeded_severities}"
        )
    _require_retained_counts_within_declared(
        retained_rule_counts,
        rule_counts,
        label=f"{label} rule_counts",
    )
    retained_category_counts = _category_counts_from_rule_counts(
        retained_rule_counts,
        rule_to_category=rule_to_category,
    )
    _require_retained_counts_within_declared(
        retained_category_counts,
        category_counts,
        label=f"{label} category_counts",
    )
    if truncated == 0:
        normalized_retained_rule_counts = {
            rule: dict(sorted(counts.items()))
            for rule, counts in sorted(retained_rule_counts.items())
        }
        if rule_counts != normalized_retained_rule_counts:
            raise ArticulationV2StaticGate3AContractError(
                f"{label} untruncated rule_counts do not match retained issues"
            )
    # When the producer truncates issue rows, the omitted rows are unrecoverable.
    # The complete rule map still fixes their severity/category allocation: it
    # must flatten to the declared severities, project exactly through the
    # registered rule inventory, and dominate every retained row above.
    return raw_status, severity_counts


def validate_articulation_v2_static_gate3a_report_consistency(
    document: Mapping[str, Any],
) -> None:
    """Recompute ordered rows, producer summaries, digest, and aggregates."""

    rule_to_category = _registered_rule_categories(document)
    results = tuple(
        _mapping(item, label="Gate 3A result")
        for item in _list(document.get("results"), label="Gate 3A results")
    )
    result_count = _nonnegative_integer(
        document.get("result_count"),
        label="Gate 3A result_count",
    )
    if result_count != len(results):
        raise ArticulationV2StaticGate3AContractError(
            "Gate 3A result_count does not match the ordered results"
        )
    if document.get("results_sha256") != _canonical_sha256({"results": results}):
        raise ArticulationV2StaticGate3AContractError("Gate 3A results_sha256 mismatch")

    status_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    for index, result in enumerate(results):
        status, row_severity_counts = _validate_result(
            result,
            label=f"Gate 3A result {index}",
            rule_to_category=rule_to_category,
        )
        status_counts[status] += 1
        severity_counts.update(row_severity_counts)

    aggregate = _mapping(document.get("aggregate"), label="Gate 3A aggregate")
    declared_status_counts = _status_count_map(
        aggregate.get("status_counts"),
        label="Gate 3A aggregate status_counts",
    )
    if declared_status_counts != dict(sorted(status_counts.items())):
        raise ArticulationV2StaticGate3AContractError(
            "Gate 3A aggregate status_counts do not match results"
        )
    declared_severity_counts = _severity_count_map(
        aggregate.get("severity_counts"),
        label="Gate 3A aggregate severity_counts",
    )
    if declared_severity_counts != dict(sorted(severity_counts.items())):
        raise ArticulationV2StaticGate3AContractError(
            "Gate 3A aggregate severity_counts do not match results"
        )


__all__ = [
    "ARTICULATION_V2_STATIC_GATE3A_COMPLETED_RESULT_FIELDS",
    "ArticulationV2StaticGate3AContractError",
    "validate_articulation_v2_static_gate3a_report_consistency",
]
