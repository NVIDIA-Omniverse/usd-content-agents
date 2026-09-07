#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check one proposed face-label revision against evidence and immutable labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mesh_geometry import (
    fragment_atomic_conflicts,
    fragments_for_faces,
    load_evidence,
    load_fragment_labels,
    load_labels,
    load_usd,
    sha256_file,
    source_metadata,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-usd", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--fragment-labels", type=Path, required=True)
    parser.add_argument("--parent-labels", type=Path, required=True)
    parser.add_argument("--candidate-labels", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--background-segment-id", type=int, default=0)
    parser.add_argument("--active-segment-id", type=int, default=1)
    parser.add_argument("--allow-noop", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_usd(args.source_usd, args.target)
    fragment_labels = load_fragment_labels(args.fragment_labels, data.face_count)
    parent = load_labels(args.parent_labels, data.face_count)
    candidate = load_labels(args.candidate_labels, data.face_count)
    evidence = load_evidence(args.evidence, data.face_count)
    immutable = (parent != args.background_segment_id) & (
        parent != args.active_segment_id
    )
    locked_change_ids = np.flatnonzero(immutable & (candidate != parent))
    added_ids = np.flatnonzero(
        (parent != args.active_segment_id) & (candidate == args.active_segment_id)
    )
    removed_ids = np.flatnonzero(
        (parent == args.active_segment_id) & (candidate != args.active_segment_id)
    )
    positive_ids = np.asarray(
        sorted(
            {
                int(event["face_id"])
                for event in evidence
                if event["polarity"] == "positive"
            }
        ),
        dtype=np.int64,
    )
    negative_ids = np.asarray(
        sorted(
            {
                int(event["face_id"])
                for event in evidence
                if event["polarity"] == "negative"
            }
        ),
        dtype=np.int64,
    )
    unsatisfied_positive = positive_ids[
        candidate[positive_ids] != args.active_segment_id
    ]
    violated_negative = negative_ids[candidate[negative_ids] == args.active_segment_id]
    selected_degenerate_ids = np.flatnonzero(
        (~data.valid_faces) & (candidate == args.active_segment_id)
    )
    parent_fragment_conflicts = fragment_atomic_conflicts(fragment_labels, parent)
    candidate_fragment_conflicts = fragment_atomic_conflicts(
        fragment_labels,
        candidate,
    )
    changed_face_ids = np.flatnonzero(candidate != parent)
    changed_fragment_ids = fragments_for_faces(fragment_labels, changed_face_ids)
    is_noop = not len(added_ids) and not len(removed_ids)
    failures = []
    if len(parent_fragment_conflicts):
        failures.append("parent labels split immutable fragments")
    if len(candidate_fragment_conflicts):
        failures.append("candidate labels split immutable fragments")
    if len(locked_change_ids):
        failures.append("candidate changes immutable labels")
    if len(unsatisfied_positive):
        failures.append("candidate omits positive evidence")
    if len(violated_negative):
        failures.append("candidate includes negative evidence")
    if len(selected_degenerate_ids):
        failures.append("candidate selects degenerate source faces")
    if is_noop and not args.allow_noop:
        failures.append("candidate is a no-op")
    payload = {
        "schema_version": "mesh-segmentation-fragment-label-comparison.v1",
        **source_metadata(data),
        "semantic_decision_unit": "immutable_fragment",
        "fragment_labels": str(args.fragment_labels.resolve()),
        "fragment_labels_sha256": sha256_file(args.fragment_labels.resolve()),
        "fragment_count": int(fragment_labels.max()) + 1,
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "parent_labels": str(args.parent_labels.resolve()),
        "parent_labels_sha256": sha256_file(args.parent_labels.resolve()),
        "candidate_labels": str(args.candidate_labels.resolve()),
        "candidate_labels_sha256": sha256_file(args.candidate_labels.resolve()),
        "active_segment_id": args.active_segment_id,
        "background_segment_id": args.background_segment_id,
        "parent_active_face_count": int(
            np.count_nonzero(parent == args.active_segment_id)
        ),
        "candidate_active_face_count": int(
            np.count_nonzero(candidate == args.active_segment_id)
        ),
        "added_face_count": int(len(added_ids)),
        "removed_face_count": int(len(removed_ids)),
        "changed_fragment_count": int(len(changed_fragment_ids)),
        "changed_fragment_ids": [int(value) for value in changed_fragment_ids],
        "parent_fragment_atomicity_conflict_ids": [
            int(value) for value in parent_fragment_conflicts
        ],
        "candidate_fragment_atomicity_conflict_ids": [
            int(value) for value in candidate_fragment_conflicts
        ],
        "locked_face_change_count": int(len(locked_change_ids)),
        "locked_face_change_ids": [int(value) for value in locked_change_ids],
        "unsatisfied_positive_face_ids": [int(value) for value in unsatisfied_positive],
        "violated_negative_face_ids": [int(value) for value in violated_negative],
        "selected_degenerate_face_ids": [
            int(value) for value in selected_degenerate_ids
        ],
        "is_noop": is_noop,
    }
    write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
