#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate exact topology-component coverage in the global semantic proposal."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery-manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate(
    gallery: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    raw_components = gallery.get("components")
    if not isinstance(raw_components, list):
        raise ValueError("Gallery manifest lacks components")
    expected = {
        record.get("component_id")
        for record in raw_components
        if isinstance(record, dict)
        and isinstance(record.get("component_id"), int)
        and not isinstance(record.get("component_id"), bool)
    }
    if len(expected) != len(raw_components):
        raise ValueError("Gallery component IDs must be unique integers")

    assignments = review.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("Review lacks assignments")
    assigned: list[int] = []
    invalid_entries: list[str] = []
    for assignment_index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            invalid_entries.append(f"assignment {assignment_index} is not an object")
            continue
        raw_ids = assignment.get("component_ids")
        if not isinstance(raw_ids, list):
            invalid_entries.append(f"assignment {assignment_index} lacks component_ids")
            continue
        for value in raw_ids:
            if not isinstance(value, int) or isinstance(value, bool):
                invalid_entries.append(
                    f"assignment {assignment_index} has a non-integer component ID"
                )
            else:
                assigned.append(value)

    unresolved_raw = review.get("unresolved_components")
    unresolved: list[int] = []
    if unresolved_raw is not None:
        if not isinstance(unresolved_raw, list):
            invalid_entries.append("unresolved_components must be a list or null")
        else:
            for value in unresolved_raw:
                if not isinstance(value, int) or isinstance(value, bool):
                    invalid_entries.append(
                        "unresolved_components has a non-integer component ID"
                    )
                else:
                    unresolved.append(value)

    counts = Counter([*assigned, *unresolved])
    observed = set(counts)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    problems = [*invalid_entries]
    if missing:
        problems.append(f"missing component IDs: {missing}")
    if unexpected:
        problems.append(f"unexpected component IDs: {unexpected}")
    if duplicates:
        problems.append(f"duplicate component IDs: {duplicates}")
    return {
        "schema_version": "mesh-segmentation-component-assignment-validation.v1",
        "status": "passed" if not problems else "failed",
        "expected_component_count": len(expected),
        "assigned_component_count": len(assigned),
        "unresolved_component_count": len(unresolved),
        "missing_component_ids": missing,
        "unexpected_component_ids": unexpected,
        "duplicate_component_ids": duplicates,
        "problems": problems,
    }


def main() -> int:
    args = parse_args()
    result = validate(
        _read_object(args.gallery_manifest.resolve()),
        _read_object(args.review.resolve()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
