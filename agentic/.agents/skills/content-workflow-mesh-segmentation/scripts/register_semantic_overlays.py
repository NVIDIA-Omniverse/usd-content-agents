#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Affine-register generated semantic overlays deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mesh_geometry import sha256_file, write_json
from PIL import Image
from semantic_overlay_registration import register_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=Path)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-silhouette-iou", type=float, default=0.95)
    parser.add_argument("--minimum-edge-f-score", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.overlay_manifest.resolve()
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != (
        "mesh-segmentation-semantic-overlays.v1"
    ):
        raise ValueError("Unsupported semantic-overlay manifest")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for source_record in source_manifest["records"]:
        view_id = str(source_record["view_id"])
        source_path = Path(str(source_record["source_render"]))
        if args.original_dir is not None:
            source_path = args.original_dir.resolve() / f"{view_id}.png"
        generated_path = Path(str(source_record["raw_overlay"]))
        source_rgb = np.asarray(Image.open(source_path).convert("RGB"))
        generated_rgb = np.asarray(Image.open(generated_path).convert("RGB"))
        try:
            result = register_overlay(source_rgb, generated_rgb)
            failure = None
        except ValueError as exc:
            result = None
            failure = str(exc)

        if result is None:
            records.append(
                {
                    **source_record,
                    "view_id": view_id,
                    "status": "rejected",
                    "accepted": False,
                    "rejection_reason": failure,
                }
            )
            print(f"registered {view_id}: rejected ({failure})", flush=True)
            continue
        aligned_path = output_dir / f"{view_id}_aligned_overlay.png"
        mask_path = output_dir / f"{view_id}_aligned_mask.png"
        blend_path = output_dir / f"{view_id}_alignment_blend.png"
        Image.fromarray(result["aligned_rgb"], mode="RGB").save(aligned_path)
        Image.fromarray(
            result["aligned_semantic_mask"].astype(np.uint8) * 255, mode="L"
        ).save(mask_path)
        blend = np.rint(
            0.5 * source_rgb.astype(np.float32)
            + 0.5 * result["aligned_rgb"].astype(np.float32)
        ).astype(np.uint8)
        Image.fromarray(blend, mode="RGB").save(blend_path)
        iou = float(result["silhouette_iou"])
        edge_score = float(result["edge_f_score_2px"])
        plausibility = dict(result["plausibility"])
        semantic_pixel_count = int(np.count_nonzero(result["aligned_semantic_mask"]))
        accepted = bool(
            iou >= args.minimum_silhouette_iou
            and edge_score >= args.minimum_edge_f_score
            and plausibility["accepted"]
        )
        records.append(
            {
                **source_record,
                "view_id": view_id,
                "status": "accepted" if accepted else "rejected",
                "accepted": accepted,
                "registered_overlay": str(aligned_path),
                "registered_overlay_sha256": sha256_file(aligned_path),
                "registered_mask": str(mask_path),
                "registered_mask_sha256": sha256_file(mask_path),
                "alignment_blend": str(blend_path),
                "alignment_blend_sha256": sha256_file(blend_path),
                "source_to_generated_affine": (
                    result["source_to_generated_affine"].tolist()
                ),
                "ecc_trace": result["ecc_trace"],
                "silhouette_iou": iou,
                "edge_f_score_2px": edge_score,
                "affine_plausibility": plausibility,
                "semantic_pixel_count": semantic_pixel_count,
                "semantic_evidence": (
                    "positive_mask"
                    if semantic_pixel_count > 0
                    else "empty_positive_set"
                ),
                "rejection_reason": (None if accepted else "registration_gate"),
            }
        )
        print(
            f"registered {view_id}: accepted={accepted} "
            f"IoU={iou:.4f} edgeF={edge_score:.4f}",
            flush=True,
        )

    payload = {
        "schema_version": "mesh-segmentation-semantic-registration.v1",
        "target_semantic_part": source_manifest["target_semantic_part"],
        "overlay_manifest": str(manifest_path),
        "overlay_manifest_sha256": sha256_file(manifest_path),
        "gates": {
            "minimum_silhouette_iou": args.minimum_silhouette_iou,
            "minimum_edge_f_score_2px": args.minimum_edge_f_score,
            "require_plausible_affine": True,
            "require_nonempty_semantic_mask": False,
            "empty_semantic_mask_policy": (
                "accepted_when_registration_passes_and_contributes_no_fragments"
            ),
        },
        "accepted_view_count": sum(record["accepted"] for record in records),
        "views": records,
    }
    output_path = output_dir / "manifest.json"
    write_json(output_path, payload)
    print(
        json.dumps(
            {
                "accepted_view_count": payload["accepted_view_count"],
                "manifest": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
