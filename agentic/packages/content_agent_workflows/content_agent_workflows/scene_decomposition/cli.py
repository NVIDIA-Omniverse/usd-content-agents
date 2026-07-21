# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for generic scene decomposition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .decomposition import run_scene_decomposition
from .manifest import SceneDecompositionRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decompose a USD scene into a generic agentic scene manifest."
    )
    parser.add_argument("usd_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-id", default="default")
    parser.add_argument("--intent", default="generic_processing")
    parser.add_argument("--input-digest")
    parser.add_argument("--root-prim-path")
    parser.add_argument("--include-path", action="append", default=[])
    parser.add_argument("--exclude-path", action="append", default=[])
    parser.add_argument("--asset-filter", action="append", default=[])
    parser.add_argument("--min-mesh-count", type=int, default=0)
    parser.add_argument("--exclude-invisible-assets", action="store_true")
    parser.add_argument("--detect-structural-duplicates", action="store_true")
    parser.add_argument("--no-payload-groups", action="store_true")
    parser.add_argument("--no-native-prototypes", action="store_true")
    parser.add_argument("--extract-large-payload-representatives", action="store_true")
    parser.add_argument("--extract-assets", action="store_true")
    parser.add_argument("--no-flatten-extracts", action="store_true")
    parser.add_argument("--extract-workers", type=int, default=1)
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--building-block-min-reuse", type=int, default=20)
    parser.add_argument("--enable-llm-refinement", action="store_true")
    parser.add_argument("--no-material-agent-manifest", action="store_true")
    args = parser.parse_args(argv)

    try:
        request = SceneDecompositionRequest(
            usd_path=args.usd_path,
            output_dir=args.output_dir,
            manifest_id=args.manifest_id,
            decomposition_intent=args.intent,
            root_prim_path=args.root_prim_path,
            include_paths=args.include_path,
            exclude_paths=args.exclude_path,
            asset_filter=args.asset_filter,
            min_mesh_count=args.min_mesh_count,
            exclude_invisible_assets=args.exclude_invisible_assets,
            detect_structural_duplicates=args.detect_structural_duplicates,
            detect_payload_groups=not args.no_payload_groups,
            detect_native_prototypes=not args.no_native_prototypes,
            extract_large_payload_representatives=(
                args.extract_large_payload_representatives
            ),
            extract_assets=args.extract_assets,
            flatten_extracts=not args.no_flatten_extracts,
            extract_workers=args.extract_workers,
            skip_geometry=args.skip_geometry,
            building_block_min_reuse=args.building_block_min_reuse,
            enable_llm_refinement=args.enable_llm_refinement,
            write_material_agent_manifest=not args.no_material_agent_manifest,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.input_digest:
        result = run_scene_decomposition(request, input_digest=args.input_digest)
    else:
        result = run_scene_decomposition(request)
    print(result.model_dump_json(indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
