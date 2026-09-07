#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply an auditable batch of whole-fragment semantic corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from mesh_geometry import (
    faces_for_fragments,
    fragment_atomic_conflicts,
    fragments_for_faces,
    load_fragment_labels,
    load_labels,
    load_usd,
    sha256_file,
    source_metadata,
    write_diagnostic_usd,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-usd", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--fragment-labels", type=Path, required=True)
    parser.add_argument("--parent-labels", type=Path, required=True)
    parser.add_argument("--edits", type=Path, required=True)
    parser.add_argument("--background-segment-id", type=int, default=0)
    parser.add_argument("--active-segment-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _fragment_ids(raw: dict[str, Any], fragment_count: int) -> np.ndarray:
    values = raw.get("fragment_ids")
    if not isinstance(values, list) or not values:
        raise ValueError("Every edit must contain a nonempty fragment_ids list")
    fragment_ids = np.asarray(
        sorted({int(value) for value in values}),
        dtype=np.int64,
    )
    if np.any((fragment_ids < 0) | (fragment_ids >= fragment_count)):
        raise ValueError("Edit contains an out-of-range fragment ID")
    return fragment_ids


def main() -> None:
    args = parse_args()
    if args.active_segment_id == args.background_segment_id:
        raise ValueError("Active and background segment IDs must differ")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_usd(args.source_usd, args.target)
    fragment_labels = load_fragment_labels(args.fragment_labels, data.face_count)
    fragment_count = int(fragment_labels.max()) + 1
    parent = load_labels(args.parent_labels, data.face_count)
    parent_conflicts = fragment_atomic_conflicts(fragment_labels, parent)
    if len(parent_conflicts):
        raise ValueError(
            "Parent labels split immutable fragments: "
            f"{parent_conflicts.astype(int).tolist()}"
        )
    labels = parent.copy()
    immutable_faces = (parent != args.background_segment_id) & (
        parent != args.active_segment_id
    )
    immutable_fragments = fragments_for_faces(
        fragment_labels,
        np.flatnonzero(immutable_faces),
    )
    invalid_fragments = fragments_for_faces(
        fragment_labels,
        data.degenerate_face_ids,
    )

    edits_path = args.edits.resolve()
    payload = json.loads(edits_path.read_text(encoding="utf-8"))
    raw_edits = payload.get("edits")
    if not isinstance(raw_edits, list) or not raw_edits:
        raise ValueError("Edit file must contain a nonempty edits list")

    records = []
    touched: set[int] = set()
    for index, raw in enumerate(raw_edits):
        if not isinstance(raw, dict):
            raise ValueError(f"Edit {index} is not an object")
        operation = str(raw.get("operation", "")).lower()
        if operation not in {"include", "exclude"}:
            raise ValueError(f"Edit {index} has invalid operation: {operation}")
        fragment_ids = _fragment_ids(raw, fragment_count)
        overlap = touched.intersection(int(value) for value in fragment_ids)
        if overlap:
            raise ValueError(
                "Fragment IDs appear in more than one edit operation: "
                f"{sorted(overlap)}"
            )
        touched.update(int(value) for value in fragment_ids)
        immutable_overlap = np.intersect1d(
            fragment_ids,
            immutable_fragments,
            assume_unique=True,
        )
        if len(immutable_overlap):
            raise ValueError(
                "Edit overlaps immutable segment fragments: "
                f"{immutable_overlap.astype(int).tolist()}"
            )
        if operation == "include":
            invalid_overlap = np.intersect1d(
                fragment_ids,
                invalid_fragments,
                assume_unique=True,
            )
            if len(invalid_overlap):
                raise ValueError(
                    "Include edit contains fragments with degenerate source faces: "
                    f"{invalid_overlap.astype(int).tolist()}"
                )
        face_mask = faces_for_fragments(fragment_labels, fragment_ids)
        labels[face_mask] = np.uint32(
            args.active_segment_id
            if operation == "include"
            else args.background_segment_id
        )
        records.append(
            {
                "index": index,
                "operation": operation,
                "fragment_ids": fragment_ids.astype(int).tolist(),
                "face_count": int(np.count_nonzero(face_mask)),
                "reason": raw.get("reason"),
                "evidence_render_ids": raw.get("evidence_render_ids", []),
            }
        )

    if np.array_equal(labels, parent):
        raise ValueError("Edit batch is a no-op")
    conflicts = fragment_atomic_conflicts(fragment_labels, labels)
    if len(conflicts):
        raise RuntimeError(
            "Internal error: edit batch split fragments "
            f"{conflicts.astype(int).tolist()}"
        )

    labels_path = output_dir / "face_labels.u32le"
    labels.astype("<u4").tofile(labels_path)
    diagnostic_path = output_dir / "diagnostic.usdc"
    write_diagnostic_usd(
        diagnostic_path,
        data,
        labels,
        active_segment_id=args.active_segment_id,
    )
    touched_array = np.asarray(sorted(touched), dtype=np.int64)
    touched_face_count = int(
        np.count_nonzero(faces_for_fragments(fragment_labels, touched_array))
    )
    manifest = {
        "schema_version": "mesh-segmentation-fragment-edit-batch.v1",
        **source_metadata(data),
        "status": "passed",
        "semantic_decision_unit": "immutable_fragment",
        "fragment_labels": str(args.fragment_labels.resolve()),
        "fragment_labels_sha256": sha256_file(args.fragment_labels.resolve()),
        "fragment_count": fragment_count,
        "parent_labels": str(args.parent_labels.resolve()),
        "parent_labels_sha256": sha256_file(args.parent_labels.resolve()),
        "edits": str(edits_path),
        "edits_sha256": sha256_file(edits_path),
        "active_segment_id": args.active_segment_id,
        "background_segment_id": args.background_segment_id,
        "touched_fragment_count": len(touched),
        "touched_face_count": touched_face_count,
        "added_fragment_count": int(
            len(
                fragments_for_faces(
                    fragment_labels,
                    np.flatnonzero(
                        (labels == args.active_segment_id)
                        & (parent != args.active_segment_id)
                    ),
                )
            )
        ),
        "removed_fragment_count": int(
            len(
                fragments_for_faces(
                    fragment_labels,
                    np.flatnonzero(
                        (labels != args.active_segment_id)
                        & (parent == args.active_segment_id)
                    ),
                )
            )
        ),
        "added_face_count": int(
            np.count_nonzero(
                (labels == args.active_segment_id) & (parent != args.active_segment_id)
            )
        ),
        "removed_face_count": int(
            np.count_nonzero(
                (labels != args.active_segment_id) & (parent == args.active_segment_id)
            )
        ),
        "fragment_atomicity_conflict_ids": [],
        "edit_records": records,
        "face_labels": str(labels_path),
        "face_labels_sha256": sha256_file(labels_path),
        "diagnostic_usd": str(diagnostic_path),
        "diagnostic_usd_sha256": sha256_file(diagnostic_path),
    }
    write_json(output_dir / "edit_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
