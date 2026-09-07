#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare immutable neutral and all-face candidate artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mesh_geometry import (
    load_usd,
    sha256_file,
    source_metadata,
    write_json,
    write_neutral_usd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-usd", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_usd(args.source_usd, args.target)
    neutral = output_dir / "neutral.usdc"
    write_neutral_usd(neutral, data)
    candidate = output_dir / "all_faces_candidate.u32le"
    candidate.write_bytes(data.valid_faces.astype("<u4").tobytes())
    degenerate_face_ids = output_dir / "degenerate_face_ids.u32le"
    data.degenerate_face_ids.astype("<u4").tofile(degenerate_face_ids)
    topology = output_dir / "topology.json"
    component_face_counts = np.bincount(data.component_ids).astype(int)
    write_json(
        topology,
        {
            "schema_version": "mesh-segmentation-topology.v1",
            **source_metadata(data),
            "topology_component_count": int(data.component_ids.max()) + 1,
            "largest_component_face_counts": sorted(
                component_face_counts.tolist(),
                reverse=True,
            )[:32],
            "neutral_usd": str(neutral),
            "all_faces_candidate": str(candidate),
            "degenerate_face_ids_file": str(degenerate_face_ids),
            "degenerate_face_ids_sha256": sha256_file(degenerate_face_ids),
            "degenerate_face_policy": (
                "preserve_source_face_id_force_background_never_pick_or_select"
            ),
        },
    )
    print(topology.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
