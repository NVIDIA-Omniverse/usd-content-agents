#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create an auditable selected-only USD view for one exported mesh segment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mesh_geometry import sha256_file, write_json
from pxr import Usd, UsdGeom

SCHEMA_VERSION = "mesh-segmentation-selected-only-stage.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-usd", type=Path, required=True)
    parser.add_argument("--target-prim", required=True)
    parser.add_argument("--output-usd", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.source_usd.resolve()
    output_path = args.output_usd.resolve()
    manifest_path = args.manifest.resolve()
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite selected-only artifacts")
    stage = Usd.Stage.Open(str(source_path))
    if stage is None:
        raise ValueError(f"Failed to open USD stage: {source_path}")
    target = stage.GetPrimAtPath(args.target_prim)
    if not target.IsValid() or not target.IsA(UsdGeom.Mesh):
        raise ValueError(f"Selected-only target is not a mesh: {args.target_prim}")

    hidden_paths: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or prim.GetPath() == target.GetPath():
            continue
        # OVRTX may still draw a mesh with authored visibility=invisible when
        # loading an exported binary stage. Deactivation removes it from stage
        # traversal entirely, which makes this review artifact truly selected-only.
        prim.SetActive(False)
        hidden_paths.append(str(prim.GetPath()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(output_path)):
        raise RuntimeError(f"Failed to export selected-only stage: {output_path}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "source_usd": str(source_path),
        "source_usd_sha256": sha256_file(source_path),
        "target_prim": str(target.GetPath()),
        "hidden_mesh_prims": hidden_paths,
        "output_usd": str(output_path),
        "output_usd_sha256": sha256_file(output_path),
    }
    write_json(manifest_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
