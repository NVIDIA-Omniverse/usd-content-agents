# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for deterministic geometry repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import (
    ClassifiedHoleIntent,
    ProtectedFeature,
    RepairBudgets,
    RepairIntent,
    RepairRequest,
)
from .orchestrator import run_geometry_repair
from .worker_ids import canonical_worker_name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geometry-repair")
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=[
            "visual_only",
            "static_environment",
            "rigid_pick_place",
            "articulated_rigid",
            "contact_rich",
            "deformable_or_cae",
        ],
        required=True,
    )
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--profile-inferred", action="store_true")
    parser.add_argument(
        "--production-use",
        action="store_true",
        help="Require source rights/provenance evidence before certification.",
    )
    parser.add_argument("--protected-features-json", type=Path)
    parser.add_argument("--classified-holes-json", type=Path)
    parser.add_argument("--repair-intents-json", type=Path)
    parser.add_argument("--use-proposed-intent-ranking", action="store_true")
    parser.add_argument("--budgets-json", type=Path)
    parser.add_argument(
        "--worker",
        action="append",
        type=canonical_worker_name,
        choices=[
            "trimesh_conservative_cleanup",
            "trimesh_bounded_hole_fill",
            "ocp_shape_heal",
            "usd_structure_repair",
            "geogram_local_repair",
            "scene_optimizer_deinstance",
            "sdf_rebuild",
            "sdf_collision_rebuild",
            "manifold_restore_merge_vectors",
            "pmp_patch",
            "coacd_collision",
        ],
    )
    parser.add_argument("--source-uri")
    parser.add_argument("--source-license")
    parser.add_argument("--source-provenance-json", type=Path)
    parser.add_argument("--source-meters-per-unit", type=float)
    parser.add_argument("--source-up-axis", choices=["X", "Y", "Z"])
    parser.add_argument("--advanced-profile-json", type=Path)
    parser.add_argument(
        "--dependency-root",
        action="append",
        type=Path,
        default=[],
        help="Allow hashing local USD dependencies under this root; remote assets are never fetched.",
    )
    parser.add_argument(
        "--dependency-remap-manifest",
        type=Path,
        help="Exact authored-identifier to local-file remaps; basename or glob matching is forbidden.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--collision-runtime-engine",
        choices=["skip", "fake", "ovphysx"],
        default="skip",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one repair request and print its canonical result."""

    args = _parser().parse_args(argv)
    protected_payload = (
        json.loads(args.protected_features_json.read_text(encoding="utf-8"))
        if args.protected_features_json
        else []
    )
    classified_holes_payload = (
        json.loads(args.classified_holes_json.read_text(encoding="utf-8"))
        if args.classified_holes_json
        else []
    )
    repair_intents_payload = (
        json.loads(args.repair_intents_json.read_text(encoding="utf-8"))
        if args.repair_intents_json
        else []
    )
    budgets_payload = (
        json.loads(args.budgets_json.read_text(encoding="utf-8")) if args.budgets_json else {}
    )
    if not isinstance(protected_payload, list):
        raise ValueError("--protected-features-json must contain a JSON array")
    if not isinstance(classified_holes_payload, list):
        raise ValueError("--classified-holes-json must contain a JSON array")
    if not isinstance(repair_intents_payload, list):
        raise ValueError("--repair-intents-json must contain a JSON array")
    if not isinstance(budgets_payload, dict):
        raise ValueError("--budgets-json must contain a JSON object")
    source_provenance = (
        json.loads(args.source_provenance_json.read_text(encoding="utf-8"))
        if args.source_provenance_json
        else {}
    )
    if not isinstance(source_provenance, dict):
        raise ValueError("--source-provenance-json must contain a JSON object")
    advanced_profile = (
        json.loads(args.advanced_profile_json.read_text(encoding="utf-8"))
        if args.advanced_profile_json
        else None
    )
    if advanced_profile is not None and not isinstance(advanced_profile, dict):
        raise ValueError("--advanced-profile-json must contain a JSON object")
    result = run_geometry_repair(
        RepairRequest(
            source_path=args.source,
            output_dir=args.out,
            profile=args.profile,
            mode="diagnose" if args.diagnose_only else "auto",
            profile_confirmed=not args.profile_inferred,
            production_use=args.production_use,
            protected_features=[
                ProtectedFeature.model_validate(item) for item in protected_payload
            ],
            classified_holes=[
                ClassifiedHoleIntent.model_validate(item) for item in classified_holes_payload
            ],
            proposed_intents=[RepairIntent.model_validate(item) for item in repair_intents_payload],
            use_proposed_intent_ranking=args.use_proposed_intent_ranking,
            budgets=RepairBudgets.model_validate(budgets_payload),
            enabled_workers=args.worker,
            deterministic_seed=args.seed,
            collision_runtime_engine=args.collision_runtime_engine,
            source_uri=args.source_uri,
            source_license=args.source_license,
            source_provenance=source_provenance,
            source_meters_per_unit=args.source_meters_per_unit,
            source_up_axis=args.source_up_axis,
            dependency_roots=args.dependency_root,
            dependency_remap_manifest=args.dependency_remap_manifest,
            advanced_profile=advanced_profile,
        )
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if result.outcome != "rejected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
