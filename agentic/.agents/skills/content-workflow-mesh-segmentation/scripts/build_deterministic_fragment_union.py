#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project registered masks per view into an exact fragment-set union."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from mesh_geometry import schema_matches, sha256_file, write_json
from PIL import Image
from scipy.ndimage import binary_erosion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registration-manifest",
        type=Path,
        action="append",
        required=True,
        help=(
            "Repeat for retry manifests. Exactly one accepted record must remain "
            "for each required view."
        ),
    )
    parser.add_argument("--id-buffer-manifest", type=Path, required=True)
    parser.add_argument("--fragment-labels", type=Path, required=True)
    parser.add_argument("--parent-labels", type=Path, required=True)
    parser.add_argument("--active-segment-id", type=int, required=True)
    parser.add_argument("--background-segment-id", type=int, default=0)
    parser.add_argument("--erosion-pixels", type=int, choices=[2], default=2)
    parser.add_argument("--expected-view-count", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def _load_fragment_labels(path: Path) -> np.ndarray:
    resolved = path.resolve()
    if resolved.suffix == ".npy":
        values = np.load(resolved, allow_pickle=False)
    else:
        values = np.fromfile(resolved, dtype="<u4")
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Fragment labels must be a one-dimensional integer array")
    if len(values) == 0:
        raise ValueError("Fragment labels are empty")
    return values.astype(np.uint32, copy=False)


def _load_parent_labels(path: Path, face_count: int) -> np.ndarray:
    values = np.fromfile(path.resolve(), dtype="<u4")
    if values.ndim != 1 or len(values) != face_count:
        raise ValueError(
            f"Expected {face_count} parent labels, found shape {values.shape}"
        )
    return values.astype(np.uint32, copy=False)


def _accepted_registration_views(
    manifest_paths: list[Path],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    accepted: dict[str, dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    for manifest_arg in manifest_paths:
        manifest_path = manifest_arg.resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not schema_matches(
            payload.get("schema_version"), "mesh-segmentation-semantic-registration.v1"
        ):
            raise ValueError(f"Unsupported registration manifest: {manifest_path}")
        for record in payload.get("views", []):
            view_id = str(record["view_id"])
            status = str(record.get("status", "")).lower()
            attempts.append(
                {
                    "view_id": view_id,
                    "status": status,
                    "manifest": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                }
            )
            if status != "accepted" and record.get("accepted") is not True:
                continue
            if view_id in accepted:
                raise ValueError(
                    f"More than one accepted registration exists for view {view_id}"
                )
            accepted[view_id] = {
                **record,
                "_manifest_path": manifest_path,
            }
    return accepted, attempts


def _id_views(
    manifest_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not schema_matches(
        payload.get("schema_version"), "mesh-segmentation-id-buffers.v1"
    ):
        raise ValueError("Unsupported ID-buffer manifest")
    views: dict[str, dict[str, Any]] = {}
    for record in payload.get("views", []):
        view_id = str(record["name"])
        if view_id in views:
            raise ValueError(f"Duplicate ID-buffer view: {view_id}")
        if record.get("closest_visible_hit_only") is not True:
            raise ValueError(f"ID-buffer view is not closest-visible: {view_id}")
        views[view_id] = record
    return views, payload


def _tint(
    base_rgb: np.ndarray, pixels: np.ndarray, color: tuple[int, int, int]
) -> np.ndarray:
    output = base_rgb.copy()
    if np.any(pixels):
        tint = np.asarray(color, dtype=np.float32)
        output[pixels] = np.rint(
            output[pixels].astype(np.float32) * 0.35 + tint * 0.65
        ).astype(np.uint8)
    return output


def main() -> None:
    args = parse_args()
    if args.active_segment_id == args.background_segment_id:
        raise ValueError("Active and background segment IDs must differ")
    if args.erosion_pixels != 2:
        raise ValueError(
            "The deterministic initializer contract requires exactly 2 px erosion"
        )
    if args.expected_view_count <= 0:
        raise ValueError("--expected-view-count must be positive")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    registration_paths = [path.resolve() for path in args.registration_manifest]
    accepted, registration_attempts = _accepted_registration_views(registration_paths)
    id_manifest_path = args.id_buffer_manifest.resolve()
    id_views, id_manifest = _id_views(id_manifest_path)
    accepted_ids = set(accepted)
    id_view_ids = set(id_views)
    if len(accepted_ids) != args.expected_view_count:
        raise ValueError(
            f"Expected {args.expected_view_count} accepted registered views, "
            f"found {len(accepted_ids)}"
        )
    if len(id_view_ids) != args.expected_view_count:
        raise ValueError(
            f"Expected {args.expected_view_count} ID-buffer views, "
            f"found {len(id_view_ids)}"
        )
    if accepted_ids != id_view_ids:
        raise ValueError(
            "Accepted registration and ID-buffer view sets differ: "
            f"registration_only={sorted(accepted_ids - id_view_ids)}, "
            f"id_only={sorted(id_view_ids - accepted_ids)}"
        )

    fragment_labels_path = args.fragment_labels.resolve()
    fragment_labels = _load_fragment_labels(fragment_labels_path)
    parent_labels_path = args.parent_labels.resolve()
    parent_labels = _load_parent_labels(parent_labels_path, len(fragment_labels))
    fragment_count = int(fragment_labels.max()) + 1

    minimum = np.full(fragment_count, np.iinfo(np.uint32).max, dtype=np.uint32)
    maximum = np.zeros(fragment_count, dtype=np.uint32)
    np.minimum.at(minimum, fragment_labels, parent_labels)
    np.maximum.at(maximum, fragment_labels, parent_labels)
    split_parent_fragments = np.flatnonzero(minimum != maximum)
    if len(split_parent_fragments):
        raise ValueError(
            "Parent labels split immutable fragments: "
            f"{split_parent_fragments.astype(int).tolist()}"
        )

    view_order = [str(view["name"]) for view in id_manifest.get("views", [])]
    per_view_sets: dict[str, np.ndarray] = {}
    view_records: list[dict[str, Any]] = []
    structure = np.ones((3, 3), dtype=bool)
    for view_id in view_order:
        registered = accepted[view_id]
        registration_manifest_path = Path(registered["_manifest_path"])
        mask_value = registered.get(
            "registered_mask",
            registered.get("aligned_mask", registered.get("mask")),
        )
        if not mask_value:
            raise ValueError(f"Accepted registered view has no mask: {view_id}")
        mask_path = _resolve(registration_manifest_path.parent, str(mask_value))
        id_record = id_views[view_id]
        fragment_path = _resolve(
            id_manifest_path.parent,
            str(id_record["channels"]["fragment_ids"]["raw"]),
        )
        source_value = registered.get("source_render")
        if not source_value:
            raise ValueError(
                f"Accepted registered view has no source render: {view_id}"
            )
        source_path = _resolve(registration_manifest_path.parent, str(source_value))

        mask = _load_mask(mask_path)
        visible_fragment_ids = np.load(fragment_path, allow_pickle=False)
        source_rgb = np.asarray(Image.open(source_path).convert("RGB"))
        if mask.shape != visible_fragment_ids.shape:
            raise ValueError(
                f"View {view_id} mask/ID shape mismatch: "
                f"{mask.shape} versus {visible_fragment_ids.shape}"
            )
        if source_rgb.shape[:2] != mask.shape:
            raise ValueError(
                f"View {view_id} source/mask shape mismatch: "
                f"{source_rgb.shape[:2]} versus {mask.shape}"
            )

        eroded = mask
        for _ in range(args.erosion_pixels):
            eroded = binary_erosion(eroded, structure=structure)
        chosen_pixels = eroded & (visible_fragment_ids >= 0)
        selected_ids = np.unique(visible_fragment_ids[chosen_pixels]).astype(np.int64)
        if len(selected_ids) and (
            int(selected_ids.min()) < 0 or int(selected_ids.max()) >= fragment_count
        ):
            raise ValueError(f"View {view_id} contains an out-of-range fragment ID")
        per_view_sets[view_id] = selected_ids

        view_dir = output_dir / "views" / view_id
        view_dir.mkdir(parents=True, exist_ok=True)
        eroded_path = view_dir / "eroded_mask.png"
        Image.fromarray(eroded.astype(np.uint8) * 255, mode="L").save(eroded_path)
        pixel_overlay_path = view_dir / "chosen_pixels.png"
        Image.fromarray(
            _tint(source_rgb, chosen_pixels, (0, 255, 255)),
            mode="RGB",
        ).save(pixel_overlay_path)
        projection_pixels = np.isin(visible_fragment_ids, selected_ids)
        projection_path = view_dir / "selected_fragments.png"
        Image.fromarray(
            _tint(source_rgb, projection_pixels, (255, 0, 255)),
            mode="RGB",
        ).save(projection_path)
        ids_npy_path = view_dir / "selected_fragment_ids.npy"
        np.save(ids_npy_path, selected_ids, allow_pickle=False)
        ids_raw_path = view_dir / "selected_fragment_ids.u32le"
        selected_ids.astype("<u4").tofile(ids_raw_path)

        view_records.append(
            {
                "view_id": view_id,
                "registration_manifest": str(registration_manifest_path),
                "registration_manifest_sha256": sha256_file(registration_manifest_path),
                "source_render": str(source_path),
                "source_render_sha256": sha256_file(source_path),
                "registered_mask": str(mask_path),
                "registered_mask_sha256": sha256_file(mask_path),
                "fragment_id_buffer": str(fragment_path),
                "fragment_id_buffer_sha256": sha256_file(fragment_path),
                "erosion_pixels": args.erosion_pixels,
                "eroded_positive_pixel_count": int(np.count_nonzero(eroded)),
                "chosen_visible_pixel_count": int(np.count_nonzero(chosen_pixels)),
                "selected_fragment_count": int(len(selected_ids)),
                "selected_fragment_ids_npy": str(ids_npy_path),
                "selected_fragment_ids_npy_sha256": sha256_file(ids_npy_path),
                "selected_fragment_ids_raw": str(ids_raw_path),
                "selected_fragment_ids_raw_sha256": sha256_file(ids_raw_path),
                "eroded_mask": str(eroded_path),
                "eroded_mask_sha256": sha256_file(eroded_path),
                "chosen_pixel_overlay": str(pixel_overlay_path),
                "chosen_pixel_overlay_sha256": sha256_file(pixel_overlay_path),
                "selected_fragment_projection": str(projection_path),
                "selected_fragment_projection_sha256": sha256_file(projection_path),
            }
        )

    raw_union = np.unique(
        np.concatenate(list(per_view_sets.values()))
        if per_view_sets
        else np.empty(0, dtype=np.int64)
    ).astype(np.int64)
    if len(raw_union) == 0:
        raise ValueError("Eight-view projection selected no fragments")

    immutable_faces = (parent_labels != args.background_segment_id) & (
        parent_labels != args.active_segment_id
    )
    immutable_fragments = np.unique(fragment_labels[immutable_faces]).astype(np.int64)
    applied_union = np.setdiff1d(
        raw_union,
        immutable_fragments,
        assume_unique=True,
    ).astype(np.int64)
    locked_overlap = np.intersect1d(
        raw_union,
        immutable_fragments,
        assume_unique=True,
    ).astype(np.int64)
    if len(applied_union) == 0:
        raise ValueError("All union fragments are locked by completed parts")

    raw_union_path = output_dir / "raw_union_fragment_ids.npy"
    np.save(raw_union_path, raw_union, allow_pickle=False)
    union_npy_path = output_dir / "union_fragment_ids.npy"
    np.save(union_npy_path, applied_union, allow_pickle=False)
    union_raw_path = output_dir / "union_fragment_ids.u32le"
    applied_union.astype("<u4").tofile(union_raw_path)

    provenance = {
        str(fragment_id): [
            view_id for view_id in view_order if fragment_id in per_view_sets[view_id]
        ]
        for fragment_id in raw_union.astype(int)
    }
    provenance_path = output_dir / "fragment_provenance.json"
    write_json(provenance_path, provenance)

    edits = {
        "schema_version": "mesh-segmentation-deterministic-initial-union-edits.v1",
        "edits": [
            {
                "operation": "include",
                "fragment_ids": applied_union.astype(int).tolist(),
                "reason": (
                    "exact union of independently projected 2px-eroded positive "
                    "masks from eight accepted closest-visible views"
                ),
                "evidence_render_ids": view_order,
            }
        ],
    }
    edits_path = output_dir / "edits.json"
    write_json(edits_path, edits)

    expected_labels = parent_labels.copy()
    expected_labels[np.isin(fragment_labels, applied_union)] = np.uint32(
        args.active_segment_id
    )
    labels_path = output_dir / "expected_face_labels.u32le"
    expected_labels.astype("<u4").tofile(labels_path)

    replayed_union = np.unique(
        np.concatenate(
            [
                np.load(
                    output_dir / "views" / view_id / "selected_fragment_ids.npy",
                    allow_pickle=False,
                )
                for view_id in view_order
            ]
        )
    ).astype(np.int64)
    if not np.array_equal(replayed_union, raw_union):
        raise RuntimeError(
            "Saved per-view fragment sets do not replay to the raw union"
        )
    if not np.array_equal(
        np.setdiff1d(replayed_union, immutable_fragments, assume_unique=True),
        applied_union,
    ):
        raise RuntimeError(
            "Saved per-view fragment sets do not replay to applied union"
        )

    validation = {
        "schema_version": "mesh-segmentation-deterministic-fragment-union-validation.v1",
        "status": "passed",
        "exactly_expected_views_verified": True,
        "closest_visible_buffers_verified": True,
        "per_view_projection_independence_verified": True,
        "two_pixel_erosion_verified": True,
        "set_union_replay_verified": True,
        "negative_pixels_used": False,
        "pixel_ratios_used": False,
        "cross_view_voting_used": False,
        "cross_view_negative_veto_used": False,
        "agent_confirmation_before_rev_000": False,
        "raw_union_fragment_count": int(len(raw_union)),
        "locked_overlap_fragment_count": int(len(locked_overlap)),
        "applied_union_fragment_count": int(len(applied_union)),
        "selected_face_count": int(
            np.count_nonzero(expected_labels == args.active_segment_id)
        ),
    }
    validation_path = output_dir / "validation.json"
    write_json(validation_path, validation)

    manifest = {
        "schema_version": "mesh-segmentation-deterministic-fragment-union.v1",
        "method": "eight_view_two_pixel_eroded_closest_visible_fragment_set_union",
        "view_count": len(view_order),
        "view_order": view_order,
        "erosion_pixels": args.erosion_pixels,
        "per_view_selection_rule": (
            "select every closest-visible fragment touched by at least one "
            "surviving eroded positive-mask pixel"
        ),
        "cross_view_reduction": "exact_set_union",
        "negative_pixels_used": False,
        "pixel_ratios_used": False,
        "minimum_fragment_pixel_count": 1,
        "cross_view_voting_used": False,
        "cross_view_negative_veto_used": False,
        "registration_manifests": [str(path) for path in registration_paths],
        "registration_manifest_sha256": [
            sha256_file(path) for path in registration_paths
        ],
        "registration_attempts": registration_attempts,
        "id_buffer_manifest": str(id_manifest_path),
        "id_buffer_manifest_sha256": sha256_file(id_manifest_path),
        "fragment_labels": str(fragment_labels_path),
        "fragment_labels_sha256": sha256_file(fragment_labels_path),
        "parent_labels": str(parent_labels_path),
        "parent_labels_sha256": sha256_file(parent_labels_path),
        "active_segment_id": args.active_segment_id,
        "background_segment_id": args.background_segment_id,
        "views": view_records,
        "raw_union_fragment_ids": str(raw_union_path),
        "raw_union_fragment_ids_sha256": sha256_file(raw_union_path),
        "raw_union_fragment_count": int(len(raw_union)),
        "locked_overlap_fragment_ids": locked_overlap.astype(int).tolist(),
        "union_fragment_ids_npy": str(union_npy_path),
        "union_fragment_ids_npy_sha256": sha256_file(union_npy_path),
        "union_fragment_ids_raw": str(union_raw_path),
        "union_fragment_ids_raw_sha256": sha256_file(union_raw_path),
        "union_fragment_count": int(len(applied_union)),
        "fragment_provenance": str(provenance_path),
        "fragment_provenance_sha256": sha256_file(provenance_path),
        "edits": str(edits_path),
        "edits_sha256": sha256_file(edits_path),
        "expected_face_labels": str(labels_path),
        "expected_face_labels_sha256": sha256_file(labels_path),
        "validation": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "view_count": len(view_order),
                "raw_union_fragment_count": len(raw_union),
                "locked_overlap_fragment_count": len(locked_overlap),
                "union_fragment_count": len(applied_union),
                "selected_face_count": validation["selected_face_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
