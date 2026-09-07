# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene-level post-processing to reconcile material predictions.

After all sub-assets are predicted, this module gathers predictions across
the entire scene, detects inconsistencies (e.g. "Car Paint Orange" vs
"Steel Painted Orange" for visually identical surfaces), and uses an LLM
to produce a unified material mapping.
"""

from __future__ import annotations

import json
import logging
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from world_understanding.utils.artifacts import (
    open_confined_directory,
    open_confined_regular_file,
    write_bytes_to_confined,
)
from world_understanding.utils.llm_parsing import extract_json_from_llm_response
from world_understanding.utils.response_content import extract_text_content

from .manifest import SceneManifest

logger = logging.getLogger(__name__)
_REASONING_REMAP_KEYS = frozenset({"analysis", "explanation", "reasoning", "thought"})


def _find_best_predictions_file(working_dir: Path) -> Path | None:
    """Find the prediction file that downstream steps will read.

    Uses the same preference order as harmonize._find_best_predictions and
    collect._find_predictions_path: restored > raw.  This ensures reconcile
    modifies the file that later steps actually consume.
    """
    restored = working_dir / "restored" / "restored_predictions.jsonl"
    if restored.exists():
        return restored
    raw = working_dir / "predictions" / "predictions.jsonl"
    if raw.exists():
        return raw
    return None


def reconcile_predictions(
    manifest: SceneManifest,
    llm_config: dict[str, Any],
    materials_list: list[str] | None = None,
    token_tracker: Any | None = None,
) -> dict[str, str]:
    """Reconcile inconsistent material predictions across all sub-assets.

    Gathers all predictions, identifies materials that are likely the same
    surface but named differently, and asks an LLM to produce a canonical
    mapping.

    Args:
        manifest: Scene manifest with completed sub-assets.
        llm_config: LLM configuration (backend, model, temperature, etc.).
        materials_list: Optional list of valid material names for context.
        token_tracker: Optional TokenTracker for scene reconciliation usage.

    Returns:
        Remapping dict: {original_material: canonical_material}.
        Only contains entries where a change is needed.
    """
    # Gather all predictions
    all_predictions = _gather_predictions(manifest)
    if not all_predictions:
        logger.info("No predictions to reconcile")
        return {}

    # Build per-asset material distributions
    asset_distributions = _build_asset_distributions(all_predictions)

    # Detect ambiguous material pairs
    ambiguous = _detect_ambiguous_pairs(asset_distributions)
    if not ambiguous:
        logger.info("No ambiguous material pairs detected")
        return {}

    logger.info(f"Detected {len(ambiguous)} ambiguous material groups to reconcile")
    for group, stats in ambiguous.items():
        logger.info(
            f"  {group}: {stats['materials']} across {stats['asset_count']} assets"
        )

    # Ask LLM to reconcile
    remap = _llm_reconcile(ambiguous, llm_config, materials_list, token_tracker)

    if remap:
        logger.info(f"LLM reconciliation produced {len(remap)} remappings:")
        for old, new in remap.items():
            logger.info(f"  {old} -> {new}")

    return remap


def apply_remapping(
    manifest: SceneManifest,
    remap: dict[str, str],
) -> int:
    """Apply material remapping to all prediction files.

    Targets whichever prediction file downstream steps (harmonize, collect)
    will read — ``restored_predictions.jsonl`` when it exists, otherwise
    ``predictions.jsonl``.

    Args:
        manifest: Scene manifest with prediction paths.
        remap: Material remapping dict from reconcile_predictions().

    Returns:
        Number of predictions updated.
    """
    if not remap:
        return 0

    updated = 0
    for sa in manifest.sub_assets:
        if sa.status != "completed" or not sa.working_dir:
            continue
        pred_file = _find_best_predictions_file(Path(sa.working_dir))
        if not pred_file:
            continue
        updated += _remap_predictions_file(pred_file, remap)

    # Also remap payload predictions
    for pg in manifest.payload_groups:
        if pg.status != "completed":
            continue
        config_path = pg.config_path
        if not config_path:
            continue
        # Derive working dir from config path
        working_dir = Path(config_path).parent / f".{Path(config_path).stem}"
        pred_file = _find_best_predictions_file(working_dir)
        if not pred_file:
            continue
        updated += _remap_predictions_file(pred_file, remap)

    logger.info(f"Updated {updated} predictions across scene")
    return updated


def _gather_predictions(
    manifest: SceneManifest,
) -> list[dict[str, Any]]:
    """Gather all predictions from completed sub-assets.

    Reads from the same file that downstream steps will consume (restored
    predictions when available, raw predictions otherwise).
    """
    all_preds: list[dict[str, Any]] = []

    for sa in manifest.sub_assets:
        if sa.status != "completed" or not sa.working_dir:
            continue
        pred_file = _find_best_predictions_file(Path(sa.working_dir))
        if not pred_file:
            continue
        with open(pred_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entry["_asset_name"] = sa.name
                    all_preds.append(entry)
                except json.JSONDecodeError:
                    continue

    logger.info(f"Gathered {len(all_preds)} predictions from scene")
    return all_preds


def _build_asset_distributions(
    predictions: list[dict[str, Any]],
) -> dict[str, Counter]:
    """Build per-asset material distribution."""
    dist: dict[str, Counter] = defaultdict(Counter)
    for entry in predictions:
        asset = entry.get("_asset_name", "unknown")
        mat = entry.get("materials", {}).get("material", "")
        if mat:
            dist[asset][mat] += 1
    return dict(dist)


def _detect_ambiguous_pairs(
    asset_distributions: dict[str, Counter],
) -> dict[str, dict[str, Any]]:
    """Detect groups of materials that appear to be used interchangeably.

    Two materials are ambiguous if they co-occur in multiple assets and
    have similar names or belong to the same color family.
    """
    # Global material counts
    global_counts: Counter = Counter()
    for dist in asset_distributions.values():
        global_counts.update(dist)

    # Find materials that co-occur in the same assets
    # Group by color keywords
    color_keywords = [
        "orange",
        "black",
        "white",
        "yellow",
        "red",
        "blue",
        "green",
        "gray",
        "grey",
        "silver",
        "brown",
        "beige",
        "ivory",
    ]

    # Build groups of materials sharing the same color
    color_groups: dict[str, list[str]] = defaultdict(list)
    for mat in global_counts:
        mat_lower = mat.lower()
        for color in color_keywords:
            if color in mat_lower:
                color_groups[color].append(mat)
                break

    # For each color group, check if multiple materials co-occur in assets
    ambiguous: dict[str, dict[str, Any]] = {}
    for color, mats in color_groups.items():
        if len(mats) < 2:
            continue

        # Check co-occurrence: how many assets use 2+ materials from this group
        cooccur_assets = []
        for asset, dist in asset_distributions.items():
            group_mats = [m for m in mats if m in dist]
            if len(group_mats) >= 2:
                cooccur_assets.append(asset)

        if cooccur_assets:
            group_key = f"{color}_group"
            mat_counts = {m: global_counts[m] for m in mats}
            ambiguous[group_key] = {
                "materials": mat_counts,
                "asset_count": len(cooccur_assets),
                "co_occurring_assets": cooccur_assets[:10],
            }

    return ambiguous


def _llm_reconcile(
    ambiguous: dict[str, dict[str, Any]],
    llm_config: dict[str, Any],
    materials_list: list[str] | None = None,
    token_tracker: Any | None = None,
) -> dict[str, str]:
    """Ask LLM to reconcile ambiguous material groups."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from world_understanding.functions.models.chat_models import (
        create_chat_model_from_config,
    )

    system_prompt = (
        "You are an expert at industrial material identification for 3D scenes. "
        "You will be given groups of materials that are being used interchangeably "
        "in the same scene by a VLM. For each group, decide which material name "
        "should be the canonical one, and map the others to it.\n\n"
        "Rules:\n"
        "- Pick the most specific and accurate material for industrial contexts.\n"
        "- 'Steel Painted X' is preferred over 'Car Paint X' for industrial "
        "machinery (robots, conveyors, frames, enclosures).\n"
        "- 'Car Paint X' is appropriate for automotive surfaces or very high-gloss "
        "finishes only.\n"
        "- Keep both if they genuinely represent different surface finishes.\n"
        "- Only remap materials that are truly interchangeable.\n\n"
        "Return ONLY a JSON object with a `remap` object mapping old material "
        "names to canonical names. Only include entries that need changing. "
        "Example:\n"
        '{"remap": {"Car Paint Orange": "Steel Painted Orange", '
        '"Plastic Orange": "Steel Painted Orange"}}\n\n'
        'If no changes are needed, return {"remap": {}}.'
    )

    user_parts = ["Here are the ambiguous material groups found in the scene:\n"]
    for group_key, stats in ambiguous.items():
        user_parts.append(f"\n## {group_key}")
        user_parts.append(f"Co-occurs in {stats['asset_count']} assets.")
        user_parts.append("Material usage counts:")
        for mat, count in sorted(stats["materials"].items(), key=lambda x: -x[1]):
            user_parts.append(f"  - {mat}: {count}x")

    if materials_list:
        user_parts.append(
            f"\n\nValid materials in the library: {', '.join(materials_list)}"
        )

    user_parts.append(
        "\n\nFor each group, produce entries for one combined remap JSON object."
    )

    user_prompt = "\n".join(user_parts)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    llm = create_chat_model_from_config(llm_config)
    if llm is None:
        logger.warning("No API key for reconciliation — skipping")
        return {}

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    from .stats import record_model_response_usage

    record_model_response_usage(
        token_tracker,
        response,
        llm_config.get("model"),
        "scene_reconcile_llm",
    )

    # Parse JSON from response
    return _parse_remap_json(response.content)


def _parse_remap_json(response: Any) -> dict[str, str]:
    """Extract JSON remapping from LLM response."""
    response = extract_text_content(response)
    result = extract_json_from_llm_response(response, expected_keys=["remap"])
    if isinstance(result, dict):
        return _coerce_remap_dict(result)

    # Backward-compatible fallback for the previous direct mapping format.
    result = extract_json_from_llm_response(response)
    if isinstance(result, dict):
        return _coerce_remap_dict(result)

    logger.warning(f"Failed to parse LLM reconciliation response: {response[:200]}")
    return {}


def _coerce_remap_dict(result: dict[str, Any]) -> dict[str, str]:
    mapping = result.get("remap") if isinstance(result.get("remap"), dict) else result
    remap: dict[str, str] = {}
    for old_name, new_name in mapping.items():
        if str(old_name).lower() in _REASONING_REMAP_KEYS:
            continue
        if not isinstance(new_name, str):
            logger.warning(
                "Dropped non-string material remap for %r from LLM response",
                old_name,
            )
            continue
        if old_name != new_name:
            remap[old_name] = new_name
    return remap


def _remap_predictions_file(
    pred_file: Path,
    remap: dict[str, str],
) -> int:
    """Remap materials in a predictions.jsonl file in-place.

    Returns number of predictions updated.
    """
    with open_confined_directory(pred_file.parent) as parent_descriptor:
        with open_confined_regular_file(
            parent_descriptor,
            pred_file.name,
        ) as (source, source_metadata):
            lines = source.read().decode().strip().split("\n")

        updated = 0
        new_lines = []
        for line in lines:
            if not line.strip():
                new_lines.append(line)
                continue
            try:
                entry = json.loads(line)
                mats = entry.get("materials", {})
                mat = mats.get("material", "")
                if mat in remap:
                    mats["material"] = remap[mat]
                    mats["original_material"] = mat
                    entry["materials"] = mats
                    updated += 1
                new_lines.append(json.dumps(entry))
            except json.JSONDecodeError:
                new_lines.append(line)

        serialized = ("\n".join(new_lines) + "\n").encode()
        write_bytes_to_confined(
            parent_descriptor,
            pred_file.name,
            serialized,
            overwrite=True,
            file_mode=stat.S_IMODE(source_metadata.st_mode),
        )
    return updated
