# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated pinned CoACD decomposition entrypoint."""

from __future__ import annotations

import argparse
import os
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from .artifacts import atomic_write_json, file_sha256
from .process_limits import limit_address_space

_PINNED_COACD_VERSION = "1.0.11"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geometry-repair-coacd")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--threshold-m", required=True, type=float)
    parser.add_argument("--max-hulls", required=True, type=int)
    parser.add_argument("--max-vertices", required=True, type=int)
    parser.add_argument("--max-faces", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--memory-mb", required=True, type=int)
    args = parser.parse_args(argv)

    started = time.monotonic()
    report: dict
    try:
        limit_address_space(args.memory_mb)
        if not 1 <= args.threads <= 64:
            raise ValueError("threads must be between 1 and 64")
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        version = metadata.version("coacd")
        if version != _PINNED_COACD_VERSION:
            raise RuntimeError(
                f"CoACD version {version} does not match pinned {_PINNED_COACD_VERSION}"
            )
        import coacd

        with np.load(args.input, allow_pickle=False) as payload:
            vertices = np.asarray(payload["vertices"], dtype=np.float64)
            faces = np.asarray(payload["faces"], dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("CoACD input vertices must be Nx3")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("CoACD input faces must be Mx3")
        if not np.isfinite(vertices).all():
            raise ValueError("CoACD input vertices must be finite")
        parts = coacd.run_coacd(
            coacd.Mesh(vertices, faces),
            threshold=args.threshold_m,
            max_convex_hull=args.max_hulls,
            preprocess_mode="off",
            resolution=1000,
            mcts_nodes=6,
            mcts_iterations=12,
            mcts_max_depth=2,
            merge=True,
            decimate=True,
            max_ch_vertex=min(args.max_vertices, max(4, (args.max_faces + 4) // 2)),
            seed=args.seed,
            real_metric=True,
        )
        if not parts:
            raise RuntimeError("CoACD returned no convex parts")
        if len(parts) > args.max_hulls:
            raise RuntimeError(f"CoACD returned {len(parts)} parts above limit {args.max_hulls}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        part_records = []
        for index, (raw_vertices, raw_faces) in enumerate(parts):
            part_vertices = np.asarray(raw_vertices, dtype=np.float64).reshape((-1, 3))
            part_faces = np.asarray(raw_faces, dtype=np.int64).reshape((-1, 3))
            if len(part_vertices) > args.max_vertices:
                raise RuntimeError(
                    f"CoACD part {index} has {len(part_vertices)} vertices above "
                    f"limit {args.max_vertices}"
                )
            if len(part_faces) > args.max_faces:
                raise RuntimeError(
                    f"CoACD part {index} has {len(part_faces)} faces above limit {args.max_faces}"
                )
            if not np.isfinite(part_vertices).all():
                raise RuntimeError(f"CoACD part {index} contains non-finite vertices")
            if (
                not len(part_faces)
                or np.any(part_faces < 0)
                or np.any(part_faces >= len(part_vertices))
            ):
                raise RuntimeError(f"CoACD part {index} has invalid topology")
            path = args.output_dir / f"part_{index:04d}.npz"
            np.savez_compressed(path, vertices=part_vertices, faces=part_faces)
            part_records.append(
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "vertex_count": len(part_vertices),
                    "triangle_count": len(part_faces),
                }
            )
        report = {
            "schema_version": "geometry-repair.coacd-result.v1",
            "status": "pass",
            "coacd_version": version,
            "input_sha256": file_sha256(args.input),
            "threshold_m": args.threshold_m,
            "max_hulls": args.max_hulls,
            "max_vertices": args.max_vertices,
            "max_faces": args.max_faces,
            "seed": args.seed,
            "thread_limit": args.threads,
            "elapsed_s": time.monotonic() - started,
            "parts": part_records,
            "failures": [],
        }
    except Exception as exc:
        report = {
            "schema_version": "geometry-repair.coacd-result.v1",
            "status": "fail",
            "elapsed_s": time.monotonic() - started,
            "parts": [],
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    atomic_write_json(args.result, report)
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
