# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect results — compose material layers onto the original scene.

Provides two composition strategies:

1. ``apply_and_compose()`` (preferred for scene mode):
   Merges predictions from all per-asset runs and applies materials in a
   single pass against the master scene USD. This produces correct bindings
   for instanced prims because the master stage has the real reference /
   instance structure.

2. ``compose_material_layers()`` (legacy):
   Sublayers per-asset material output layers over the original scene.
   Kept for backward compatibility but not used in the default flow.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # pragma: no cover
    from pxr import Sdf

from .manifest import PayloadGroup, SceneManifest, SubAsset

logger = logging.getLogger(__name__)


class _PathIndex:
    """Sorted path index for O(log P + k) prefix queries on prim_to_material."""

    def __init__(self, prim_to_material: dict[str, str]) -> None:
        self._sorted = sorted(prim_to_material)
        self._data = prim_to_material

    def get_paths_under(self, prefix: str) -> dict[str, str]:
        """Return all entries equal to *prefix* or starting with *prefix* + '/'."""
        result: dict[str, str] = {}
        lo = bisect.bisect_left(self._sorted, prefix)
        if lo < len(self._sorted) and self._sorted[lo] == prefix:
            result[prefix] = self._data[prefix]
            lo += 1
        hi = bisect.bisect_right(self._sorted, prefix + "/\U0010ffff")
        for p in self._sorted[lo:hi]:
            result[p] = self._data[p]
        return result

    def is_under_any(self, prim_path: str, prefixes: list[str]) -> bool:
        """Return True if *prim_path* equals or is a child of any prefix.

        *prefixes* must be a sorted list (call sorted() once and reuse).
        Uses binary search: O(log S) instead of O(S).
        """
        if not prefixes:
            return False
        # Candidate: largest prefix <= prim_path
        idx = bisect.bisect_right(prefixes, prim_path) - 1
        if idx >= 0:
            pfx = prefixes[idx]
            if prim_path == pfx or prim_path.startswith(pfx + "/"):
                return True
        return False


def _prune_nested_prefixes(prefixes: set[str]) -> list[str]:
    """Remove prefixes already covered by an earlier ancestor prefix."""
    pruned: list[str] = []
    for prefix in sorted(prefixes):
        if any(prefix == kept or prefix.startswith(kept + "/") for kept in pruned):
            continue
        pruned.append(prefix)
    return pruned


def _relative_path_under_any_member(
    prim_path: str,
    member_paths: list[str],
) -> str | None:
    """Return the relative suffix when *prim_path* is inside an IG member."""
    for member_path in member_paths:
        if prim_path == member_path:
            return ""
        if prim_path.startswith(member_path + "/"):
            return prim_path[len(member_path) :]
    return None


def _copy_layer_stage_metadata(source: Sdf.Layer, target: Sdf.Layer) -> None:
    """Copy stage-level metadata (upAxis, defaultPrim) from source to target layer."""
    if source.defaultPrim:
        target.defaultPrim = source.defaultPrim
    if source.pseudoRoot.HasInfo("upAxis"):
        target.pseudoRoot.SetInfo("upAxis", source.pseudoRoot.GetInfo("upAxis"))


def apply_and_compose(
    scene_usd_path: Path,
    manifest: SceneManifest,
    output_usd_path: Path,
    material_library_yaml: Path,
    names_filter: list[str] | None = None,
) -> Path:
    """Merge all per-asset predictions and apply materials against the master scene.

    This is the preferred composition strategy for scene mode.  Instead of
    relying on per-asset material layers (which fail for instanced prims),
    this function:

    1. Merges predictions from every completed asset into a single mapping.
    2. Loads the material library to identify which materials are used.
    3. Creates an output layer that sublayers the original scene USD.
    4. Copies only the used material definitions from the library.
    5. Writes ``material:binding`` over-specs for every predicted prim.

    Because bindings are authored against the *master* stage, they resolve
    correctly for instanced prims (instances inherit from prototypes via
    USD composition).

    Args:
        scene_usd_path: Path to the original USD scene.
        manifest: Scene manifest with completed assets.
        output_usd_path: Where to write the composed USD.
        material_library_yaml: Path to the materials library YAML file.
        names_filter: Optional name/path filter for assets.

    Returns:
        Path to the composed USD file.
    """
    from pxr import Sdf

    output_usd_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Merge predictions from all completed assets ---
    prim_to_material = _merge_predictions(manifest, names_filter)
    if not prim_to_material:
        logger.warning("No predictions found across completed assets")

    # --- 1c. Fill prediction gaps via ancestor/sibling inheritance ---
    prim_to_material = _fill_prediction_gaps(
        scene_usd_path, prim_to_material, manifest, names_filter
    )

    # --- 2. Load material library ---
    library_usd_path, name_to_prim = _load_material_library(material_library_yaml)
    logger.info(f"Material library: {library_usd_path} ({len(name_to_prim)} materials)")

    # Identify which materials are actually used
    used_materials: dict[str, str] = {}
    unknown_materials: set[str] = set()
    for _prim_path, mat_name in prim_to_material.items():
        if mat_name in name_to_prim:
            used_materials[mat_name] = name_to_prim[mat_name]
        elif mat_name not in unknown_materials:
            unknown_materials.add(mat_name)
            logger.warning(f"Predicted material not in library: '{mat_name}'")

    logger.info(
        f"Merged {len(prim_to_material)} prim assignments, "
        f"{len(used_materials)} unique materials"
    )

    # --- 3. Create output layer ---
    composed_layer = Sdf.Layer.CreateNew(str(output_usd_path))
    if not composed_layer:
        raise RuntimeError(f"Failed to create output layer: {output_usd_path}")

    # Sublayer the original scene (base geometry, weakest opinion)
    composed_layer.subLayerPaths = [str(scene_usd_path.resolve())]

    # Copy stage metadata from the original scene
    source_layer = Sdf.Layer.FindOrOpen(str(scene_usd_path.resolve()))
    if source_layer:
        _copy_layer_stage_metadata(source_layer, composed_layer)

    # --- 4. Copy used material definitions from library ---
    scene_default_prim = source_layer.defaultPrim if source_layer else ""
    if used_materials and library_usd_path:
        path_remap = _copy_materials_from_library(
            composed_layer,
            library_usd_path,
            used_materials,
            output_usd_path,
            scene_default_prim=scene_default_prim,
        )
        # Update name_to_prim so binding targets use remapped paths.
        # Also remove any used material that failed to copy (not in path_remap)
        # to prevent writing bindings that target non-existent paths.
        for mat_name in list(name_to_prim):
            old_path = name_to_prim[mat_name]
            if old_path in path_remap:
                name_to_prim[mat_name] = path_remap[old_path]
            elif mat_name in used_materials:
                # Was attempted but copy failed (prim missing in library USD)
                del name_to_prim[mat_name]
                logger.warning(
                    f"Removing '{mat_name}' from bindings: "
                    f"not found in library USD at {old_path}"
                )

    # --- 5. Write material bindings ---
    # Build a set of instance path prefixes that are instanceable in the scene.
    # Bindings under these paths must NOT be written as direct overs because
    # that would break USD instancing.  They are handled separately by
    # prototype source path remapping.
    instance_skip_prefixes: set[str] = set()
    # From payload groups
    for pg in manifest.payload_groups:
        for ip in pg.instance_paths:
            instance_skip_prefixes.add(ip)
    # From instance groups (sub-assets that are instanceable prims).
    # Skip non-representative members only — the representative's bindings
    # are written directly; propagation copies them to the other members.
    # Groups with no representative are skipped entirely — their child
    # sub-assets have their own predictions that should be written directly.
    ig_representative_paths: set[str] = set()
    for ig in manifest.instance_groups:
        if not ig.representative_id:
            # No representative — don't skip member paths so child
            # sub-asset bindings can be written directly.
            continue
        rep = manifest.get_asset_by_id(ig.representative_id)
        rep_path = rep.prim_path if rep else None
        if rep_path:
            ig_representative_paths.add(rep_path)
        # Find the "source member" — the member whose subtree contains the
        # representative.  When the representative IS a member, source_member
        # equals rep_path.  When the representative is a *descendant* of a
        # member (LLM split), source_member is that ancestor member path.
        # We must NOT skip the source member from direct writes because it
        # contains the predictions that propagation copies to other members.
        source_member: str | None = rep_path
        if rep_path:
            for mp in ig.member_paths:
                if rep_path.startswith(mp + "/"):
                    source_member = mp
                    break
        for mp in ig.member_paths:
            if mp != source_member:
                instance_skip_prefixes.add(mp)

    bindings_written = _write_material_bindings(
        composed_layer, prim_to_material, name_to_prim, instance_skip_prefixes
    )

    # --- 6. Handle payload groups (bottom-up approach) ---
    payload_arcs = 0
    instance_bindings = 0
    proto_bindings = 0
    if manifest.payload_groups:
        # The bottom-up pipeline already produced output.usd for each payload.
        # Rewrite the scene's sublayer arcs to point to the updated versions.
        modified_sublayers = _rewrite_scene_payload_arcs(
            scene_usd_path=scene_usd_path,
            manifest=manifest,
            output_dir=output_usd_path.parent,
        )
        # Replace the scene sublayer (original) with modified copies
        # that have updated payload arcs pointing to materialized versions.
        # Always include the original scene as the weakest sublayer for any
        # prims not covered by modified copies (e.g. non-payload references,
        # or scenes with no sublayers at all like Siemens NX).
        scene_abs = str(scene_usd_path.resolve())
        if modified_sublayers:
            if scene_abs not in modified_sublayers:
                modified_sublayers.append(scene_abs)
            composed_layer.subLayerPaths = modified_sublayers
        else:
            # No sublayers were rewritten — keep the original scene sublayer
            composed_layer.subLayerPaths = [scene_abs]
        payload_arcs = len(modified_sublayers)

        # For scenes with no sublayers (e.g. Siemens NX), payload arc rewriting
        # has no effect.  Fall back to prototype source path remapping: read
        # each payload's bindings and remap them from instance paths to the
        # prototype source paths so all instances inherit materials via USD
        # composition.
        no_arcs_rewritten = payload_arcs == 0 or all(
            sl == scene_abs for sl in modified_sublayers
        )
        if no_arcs_rewritten:
            proto_bindings = _compose_prototype_payloads(
                scene_usd_path=scene_usd_path,
                manifest=manifest,
                prim_to_material=prim_to_material,
                name_to_prim=name_to_prim,
                composed_layer=composed_layer,
            )
    else:
        # No payload groups — propagate bindings via instance groups (legacy)
        instance_bindings = _propagate_instance_bindings(
            manifest,
            prim_to_material,
            name_to_prim,
            composed_layer,
            scene_usd_path=scene_usd_path,
        )

    # --- 7. Remap instance group bindings to prototype source paths ---
    # Instance groups whose members are instanceable prims with local references
    # need their bindings remapped to prototype source paths, regardless of
    # whether payload groups exist.  This covers sub-assets in instance groups
    # that are NOT handled by _compose_prototype_payloads (which only covers
    # payload group instance paths).
    ig_proto_bindings = _remap_instance_group_bindings(
        scene_usd_path=scene_usd_path,
        manifest=manifest,
        prim_to_material=prim_to_material,
        name_to_prim=name_to_prim,
        composed_layer=composed_layer,
        skip_paths={ip for pg in manifest.payload_groups for ip in pg.instance_paths},
    )
    proto_bindings += ig_proto_bindings

    composed_layer.Save()

    logger.info(
        f"Composed USD saved to {output_usd_path} "
        f"({bindings_written} direct bindings, "
        f"{len(used_materials)} materials, "
        f"{payload_arcs} scene sublayers, "
        f"{instance_bindings} instance bindings, "
        f"{proto_bindings} prototype bindings)"
    )
    return output_usd_path


def author_projected_material_layer(
    scene_usd_path: Path,
    output_layer_path: Path,
    material_library_yaml: Path,
    prim_to_material: Mapping[str, str],
) -> dict[str, Any]:
    """Author canonical material bindings without changing the source scene.

    Workflow 3 performs projection, representative propagation, and conflict
    harmonization before calling this helper. This function reuses the fixed
    pipeline's material-library copy and binding authoring implementation to
    produce one independently composable material layer.
    """
    from pxr import Sdf, Tf, Usd, UsdShade

    scene_path = scene_usd_path.expanduser().resolve()
    output_path = output_layer_path.expanduser().resolve()
    library_yaml = material_library_yaml.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_layer = Sdf.Layer.FindOrOpen(str(scene_path))
    if source_layer is None:
        raise RuntimeError(f"Failed to open source scene layer: {scene_path}")
    source_stage = Usd.Stage.Open(str(scene_path), Usd.Stage.LoadNone)
    if source_stage is None:
        raise RuntimeError(f"Failed to open source scene stage: {scene_path}")

    library_usd_path, name_to_prim = _load_material_library(library_yaml)
    if library_usd_path is None or not library_usd_path.is_file():
        raise RuntimeError(f"Material library USD is unavailable: {library_usd_path}")
    unknown_materials = sorted(set(prim_to_material.values()) - set(name_to_prim))
    if unknown_materials:
        raise ValueError(f"Unknown projected materials: {unknown_materials}")

    target_prims: dict[str, Any] = {}
    for path in sorted(prim_to_material):
        prim = source_stage.GetPrimAtPath(path)
        if not prim.IsValid():
            try:
                source_stage.Load(path)
            except Tf.ErrorException:
                pass
            prim = source_stage.GetPrimAtPath(path)
        target_prims[path] = prim

    missing_targets = sorted(
        path for path, prim in target_prims.items() if not prim.IsValid()
    )
    if missing_targets:
        preview = missing_targets[:10]
        raise ValueError(
            "Projected material targets do not exist in the composed source scene: "
            f"{preview}"
        )
    instance_proxy_targets = sorted(
        path
        for path, prim in target_prims.items()
        if prim.IsInstanceProxy() or prim.IsInPrototype()
    )
    if instance_proxy_targets:
        raise ValueError(
            "Projected material targets must be mapped out of instance/prototype "
            f"namespace before authoring: {instance_proxy_targets[:10]}"
        )

    used_materials = {
        name: name_to_prim[name] for name in sorted(set(prim_to_material.values()))
    }
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=output_path.suffix,
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
    temporary_path.unlink()
    try:
        output_layer = Sdf.Layer.CreateNew(str(temporary_path))
        if output_layer is None:
            raise RuntimeError(f"Failed to create material layer: {temporary_path}")
        _copy_layer_stage_metadata(source_layer, output_layer)
        scene_default_prim = source_layer.defaultPrim or ""
        path_remap = _copy_materials_from_library(
            output_layer,
            library_usd_path,
            used_materials,
            output_path,
            scene_default_prim=scene_default_prim,
        )
        material_paths: dict[str, str] = {}
        for name, source_path in used_materials.items():
            target_path = path_remap.get(source_path)
            if target_path is None:
                raise RuntimeError(
                    f"Failed to copy material '{name}' from {source_path}"
                )
            material_paths[name] = target_path

        blocking_bindings: set[tuple[str, str]] = set()
        for prim_path in prim_to_material:
            ancestor = target_prims[prim_path].GetParent()
            while ancestor.IsValid() and not ancestor.IsPseudoRoot():
                for relationship in ancestor.GetRelationships():
                    relationship_name = relationship.GetName()
                    if (
                        relationship_name == "material:binding"
                        or relationship_name.startswith("material:binding:")
                    ) and relationship.GetMetadata("bindMaterialAs") == (
                        UsdShade.Tokens.strongerThanDescendants
                    ):
                        blocking_bindings.add(
                            (str(ancestor.GetPath()), relationship_name)
                        )
                ancestor = ancestor.GetParent()

        for ancestor_path, relationship_name in sorted(blocking_bindings):
            ancestor_spec = Sdf.CreatePrimInLayer(output_layer, ancestor_path)
            if ancestor_spec is None:
                raise RuntimeError(
                    "Failed to author material binding-strength override at "
                    f"{ancestor_path}"
                )
            ancestor_spec.specifier = Sdf.SpecifierOver
            relationship_spec = ancestor_spec.relationships.get(
                relationship_name
            ) or Sdf.RelationshipSpec(ancestor_spec, relationship_name)
            relationship_spec.SetInfo(
                "bindMaterialAs", UsdShade.Tokens.weakerThanDescendants
            )

        bindings_written = 0
        for prim_path, material_name in sorted(prim_to_material.items()):
            bindings_written += _write_binding_over(
                output_layer,
                prim_path,
                material_paths[material_name],
                scene_layer=source_layer,
            )
        output_layer.Save()
        del output_layer
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "output_layer_path": str(output_path),
        "source_scene_path": str(scene_path),
        "material_library_yaml": str(library_yaml),
        "material_library_usd": str(library_usd_path.resolve()),
        "projected_target_count": len(prim_to_material),
        "bindings_written": bindings_written,
        "weakened_ancestor_binding_count": len(blocking_bindings),
        "used_material_count": len(used_materials),
        "material_paths": material_paths,
    }


def _merge_predictions(
    manifest: SceneManifest,
    names_filter: list[str] | None = None,
) -> dict[str, str]:
    """Merge predictions from all completed assets into one prim→material dict.

    For each asset, prefers restored predictions (which have original scene
    paths) over raw predictions (which may use SO-optimized paths).

    Args:
        manifest: Scene manifest.
        names_filter: Optional filter.

    Returns:
        Dict mapping prim path → material name.
    """
    prim_to_material: dict[str, str] = {}
    assets = manifest.get_processable_assets(names_filter)

    for sa in assets:
        if sa.status != "completed":
            logger.debug(f"Skipping '{sa.name}': status={sa.status}")
            continue

        predictions_path = _find_predictions_path(sa)
        if not predictions_path:
            logger.warning(f"No predictions found for completed asset '{sa.name}'")
            continue

        count = 0
        with open(predictions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prediction = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in {predictions_path}: {line[:80]}")
                    continue

                prim_id = prediction.get("id")
                material = _extract_material_name(prediction)
                if prim_id and material:
                    prim_to_material[prim_id] = material
                    count += 1

        logger.info(f"Loaded {count} predictions from '{sa.name}' ({predictions_path})")

    # Also merge predictions from completed payload groups.
    # Payload predictions use scene-level prim paths (after restore_usd)
    # and are needed for prototype source path remapping in collect.
    for pg in manifest.payload_groups:
        if pg.status != "completed":
            continue
        pred_path = _find_predictions_path(pg)
        if not pred_path:
            continue

        count = 0
        with open(pred_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prediction = json.loads(line)
                except json.JSONDecodeError:
                    continue

                prim_id = prediction.get("id")
                material = _extract_material_name(prediction)
                if prim_id and material:
                    prim_to_material[prim_id] = material
                    count += 1

        if count:
            logger.info(f"Loaded {count} predictions from payload '{pg.group_name}'")

    # Fill in parent Mesh prims that only have GeomSubset children predicted.
    # This happens when SO splits a Mesh into per-GeomSubset meshes and
    # restore_usd maps them back to Mesh/Diffuse_N paths but not the
    # parent Mesh itself.
    parents_to_add: dict[str, str] = {}
    for prim_path in list(prim_to_material):
        parent, _, child_name = prim_path.rpartition("/")
        if (
            child_name.startswith("Diffuse")
            and parent
            and parent not in prim_to_material
        ):
            # Use the most common material among siblings
            if parent not in parents_to_add:
                parents_to_add[parent] = prim_to_material[prim_path]

    if parents_to_add:
        # Pick dominant material per parent from all its children.
        # Use path index for O(log P + k) child lookup instead of O(P) scan.
        from collections import Counter

        _idx = _PathIndex(prim_to_material)
        for parent in parents_to_add:
            # Exclude the parent itself (get_paths_under includes the exact match)
            child_mats = [
                m for p, m in _idx.get_paths_under(parent).items() if p != parent
            ]
            if child_mats:
                parents_to_add[parent] = Counter(child_mats).most_common(1)[0][0]

        prim_to_material.update(parents_to_add)
        logger.info(
            f"Inferred {len(parents_to_add)} parent Mesh predictions from "
            f"GeomSubset children"
        )

    return prim_to_material


def _find_predictions_path(sa: Any) -> Path | None:
    """Find the best predictions file for a sub-asset.

    Always prefers restored predictions (which have original scene prim
    paths) over raw predictions (which may use SO-optimized paths).
    Checks the working directory first, then falls back to the manifest's
    ``predictions_path``.
    """
    # Always check for restored predictions first — they have the correct
    # original scene paths, regardless of what the manifest records.
    if sa.working_dir:
        working_dir = Path(sa.working_dir)
        restored = working_dir / "restored" / "restored_predictions.jsonl"
        if restored.exists():
            return restored

    # Fall back to manifest's predictions_path (may be raw or restored
    # depending on when the manifest was written).
    if sa.predictions_path:
        p = Path(sa.predictions_path)
        if p.exists():
            return p

    # Last resort: raw predictions in working_dir
    if sa.working_dir:
        raw = Path(sa.working_dir) / "predictions" / "predictions.jsonl"
        if raw.exists():
            return raw

    return None


def _extract_material_name(prediction: dict) -> str | None:
    """Extract the material name from a prediction dict."""
    if "materials" in prediction:
        mat_data = prediction["materials"]
        if isinstance(mat_data, dict):
            return mat_data.get("material")
        if isinstance(mat_data, str):
            return mat_data
    if "material" in prediction:
        return prediction["material"]
    return None


def _fill_prediction_gaps(
    scene_usd_path: Path,
    prim_to_material: dict[str, str],
    manifest: SceneManifest,
    names_filter: list[str] | None = None,
) -> dict[str, str]:
    """Fill prediction gaps by inheriting materials from nearest predicted sibling.

    When the scene optimizer merges meshes, some original meshes may not
    appear in the optimized dataset and therefore have no predictions.
    For each sub-asset, this function scans the original scene for Mesh
    prims that lack predictions and assigns them the material of the
    nearest sibling (same parent prim) that *does* have a prediction.

    This is a conservative fill — only within the same parent prim, and
    only when there is exactly one material among the predicted siblings
    (unanimous siblings).
    """
    from pxr import Usd

    # Keep large-scene payloads unloaded; gap filling should only inspect
    # authored prims already visible without pulling full factory assets in.
    stage = Usd.Stage.Open(str(scene_usd_path), Usd.Stage.LoadNone)
    if not stage:
        logger.warning(f"Cannot open scene for gap fill: {scene_usd_path}")
        return prim_to_material

    predicted_paths = set(prim_to_material.keys())

    # Collect all sub-asset prim prefixes
    assets = manifest.get_processable_assets(names_filter)
    asset_prefixes = []
    selected_asset_ids: set[str] = set()
    selected_assets_by_id: dict[str, SubAsset] = {}
    for sa in assets:
        if sa.status == "completed" and sa.prim_path:
            asset_prefixes.append(sa.prim_path)
            selected_asset_ids.add(sa.id)
            selected_assets_by_id[sa.id] = sa

    filled = 0
    # For each asset, find meshes under its prim_path that have no prediction
    for prefix in asset_prefixes:
        root_prim = stage.GetPrimAtPath(prefix)
        if not root_prim or not root_prim.IsValid():
            continue

        # Group meshes by parent prim
        parent_groups: dict[str, list[str]] = {}
        for prim in Usd.PrimRange(root_prim):
            if prim.GetTypeName() != "Mesh":
                continue
            path = str(prim.GetPath())
            parent = str(prim.GetParent().GetPath()) if prim.GetParent() else ""
            parent_groups.setdefault(parent, []).append(path)

        # For each parent group, fill gaps from predicted siblings
        for _parent, siblings in parent_groups.items():
            predicted = {
                s: prim_to_material[s] for s in siblings if s in predicted_paths
            }
            unpredicted = [s for s in siblings if s not in predicted_paths]

            if not predicted or not unpredicted:
                continue

            # Use majority vote among predicted siblings
            mat_counts: Counter[str] = Counter(predicted.values())
            winner, winner_count = mat_counts.most_common(1)[0]
            # Require >50% agreement or only 1 material
            if len(mat_counts) > 1 and winner_count / len(predicted) < 0.5:
                continue

            for path in unpredicted:
                prim_to_material[path] = winner
                filled += 1

    # Second pass: for still-unpredicted meshes, inherit from the
    # dominant material in their sub-asset (most common prediction).
    predicted_paths = set(prim_to_material.keys())
    asset_fill = 0
    for prefix in asset_prefixes:
        root_prim = stage.GetPrimAtPath(prefix)
        if not root_prim or not root_prim.IsValid():
            continue

        # Collect predicted materials in this sub-asset
        asset_mats: Counter[str] = Counter()
        unpredicted_in_asset: list[str] = []
        for prim in Usd.PrimRange(root_prim):
            if prim.GetTypeName() != "Mesh":
                continue
            path = str(prim.GetPath())
            if path in predicted_paths:
                asset_mats[prim_to_material[path]] += 1
            else:
                unpredicted_in_asset.append(path)

        if not unpredicted_in_asset or not asset_mats:
            continue

        # Use the most common material in the sub-asset
        dominant_mat = asset_mats.most_common(1)[0][0]
        for path in unpredicted_in_asset:
            prim_to_material[path] = dominant_mat
            asset_fill += 1

    sibling_fill = filled
    filled += asset_fill
    suffix_fill = 0

    # Third pass: fill structurally renamed clones by matching stable leaf
    # suffixes.  Siemens NX scenes often reuse the same referenced product under
    # different container names, so direct representative→member path remapping
    # may fail even though the leaf component path is identical.  Prefer longer
    # suffixes first so generic leaves such as ``bool3/shape/mesh`` do not
    # override a more specific assembly-local match.
    suffix_lengths = (6, 5, 4)
    suffix_materials: dict[int, dict[tuple[str, ...], Counter[str]]] = {
        length: {} for length in suffix_lengths
    }
    for path, material in prim_to_material.items():
        parts = tuple(part for part in path.split("/") if part)
        for length in suffix_lengths:
            if len(parts) < length:
                continue
            suffix = parts[-length:]
            suffix_materials[length].setdefault(suffix, Counter())[material] += 1

    unanimous_suffix_materials: dict[int, dict[tuple[str, ...], str]] = {
        length: {
            suffix: next(iter(counts))
            for suffix, counts in materials.items()
            if len(counts) == 1
        }
        for length, materials in suffix_materials.items()
    }

    suffix_target_prefixes = set(asset_prefixes)
    for ig in manifest.instance_groups:
        if not ig.representative_id or ig.representative_id not in selected_asset_ids:
            continue
        representative = selected_assets_by_id.get(ig.representative_id)
        if not representative or not representative.prim_path:
            continue
        relative = _relative_path_under_any_member(
            representative.prim_path,
            ig.member_paths,
        )
        if relative is None:
            continue
        suffix_target_prefixes.update(
            f"{member_path}{relative}" for member_path in ig.member_paths
        )

    if any(unanimous_suffix_materials.values()) and suffix_target_prefixes:
        predicted_paths = set(prim_to_material.keys())
        visited_paths: set[str] = set()
        for prefix in _prune_nested_prefixes(suffix_target_prefixes):
            root_prim = stage.GetPrimAtPath(prefix)
            if not root_prim or not root_prim.IsValid():
                continue
            for prim in Usd.PrimRange(root_prim):
                if prim.GetTypeName() != "Mesh":
                    continue
                path = str(prim.GetPath())
                if path in predicted_paths or path in visited_paths:
                    continue
                visited_paths.add(path)
                parts = tuple(part for part in path.split("/") if part)
                material = None
                for length in suffix_lengths:
                    if len(parts) < length:
                        continue
                    material = unanimous_suffix_materials[length].get(parts[-length:])
                    if material:
                        break
                if material is None:
                    continue
                prim_to_material[path] = material
                predicted_paths.add(path)
                suffix_fill += 1

    filled += suffix_fill
    if filled:
        logger.info(
            f"Gap fill: {filled} meshes filled "
            f"({sibling_fill} from siblings, "
            f"{asset_fill} from asset dominant material, "
            f"{suffix_fill} from matching clone suffixes)"
        )
    else:
        logger.info("Gap fill: no gaps to fill")

    return prim_to_material


def _load_material_library(
    library_yaml_path: Path,
) -> tuple[Path | None, dict[str, str]]:
    """Load the material library YAML and return (library_usd_path, name→prim_path).

    Args:
        library_yaml_path: Path to the materials YAML file.

    Returns:
        Tuple of (path to library USD, dict mapping material name → prim path).
    """
    with open(library_yaml_path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    if not isinstance(loaded, dict):
        return None, {}

    data = loaded.get("materials", loaded)
    if not isinstance(data, dict):
        return None, {}

    # Resolve library_path relative to the YAML file's directory
    library_usd_path: Path | None = None
    raw_lib = data.get("library_path")
    if raw_lib:
        lib = Path(raw_lib)
        if not lib.is_absolute():
            lib = (library_yaml_path.parent / lib).resolve()
        library_usd_path = lib

    # Build name → binding (prim path) mapping
    name_to_prim: dict[str, str] = {}
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        binding = entry.get("binding") or entry.get("prim_path")
        if name and binding:
            name_to_prim[str(name)] = str(binding)

    return library_usd_path, name_to_prim


def _copy_materials_from_library(
    target_layer: Sdf.Layer,
    library_usd_path: Path,
    used_materials: dict[str, str],
    output_usd_path: Path,
    scene_default_prim: str = "",
) -> dict[str, str]:
    """Copy used material prim specs from the library USD into the target layer.

    Also remaps texture/MDL asset paths so they resolve from the output
    location.  If *scene_default_prim* differs from the library's root
    prim, material paths are remapped (e.g. ``/World/Looks/…`` →
    ``/Root/Looks/…``).

    Args:
        target_layer: The output Sdf.Layer to copy materials into.
        library_usd_path: Path to the material library USD.
        used_materials: Dict mapping material name → prim path in library.
        output_usd_path: Path to the output USD (for asset path remapping).
        scene_default_prim: The scene's defaultPrim name (e.g. ``Root``).

    Returns:
        Dict mapping original library prim path → actual target prim path
        (may differ if root was remapped).
    """
    from pxr import Sdf
    from world_understanding.utils.usd.material import ensure_looks_scope_spec

    library_layer = Sdf.Layer.FindOrOpen(str(library_usd_path))
    if not library_layer:
        logger.error(f"Failed to open material library USD: {library_usd_path}")
        return {}

    library_dir = library_usd_path.resolve().parent
    output_dir = output_usd_path.resolve().parent

    # Determine if we need to remap the library root prim.
    # Library materials live under /World/Looks/… but the scene may use
    # a different defaultPrim (e.g. /Root).
    lib_root = library_layer.defaultPrim or "World"
    remap_root = scene_default_prim and scene_default_prim != lib_root
    if remap_root:
        logger.info(
            f"Remapping library root '/{lib_root}' → '/{scene_default_prim}' "
            f"to match scene defaultPrim"
        )

    def _target_path(lib_path: str) -> str:
        """Remap a library prim path to the scene's namespace."""
        if remap_root and lib_path.startswith(f"/{lib_root}/"):
            return f"/{scene_default_prim}/{lib_path[len(lib_root) + 2 :]}"
        if remap_root and lib_path == f"/{lib_root}":
            return f"/{scene_default_prim}"
        return lib_path

    # Build the path remapping for callers
    path_remap: dict[str, str] = {}

    # Ensure parent prim hierarchy exists (e.g. /Root/Looks)
    parent_paths: set[str] = set()
    for prim_path in used_materials.values():
        target = _target_path(prim_path)
        path = Sdf.Path(target)
        parent = path.GetParentPath()
        while parent != Sdf.Path.absoluteRootPath:
            parent_paths.add(str(parent))
            parent = parent.GetParentPath()

    for parent_path in sorted(parent_paths):
        if not target_layer.GetPrimAtPath(parent_path):
            prim_spec = Sdf.CreatePrimInLayer(target_layer, parent_path)
            if prim_spec:
                prim_spec.specifier = Sdf.SpecifierDef
        ensure_looks_scope_spec(target_layer, parent_path)

    # Copy each used material
    copied = 0
    for material_name, prim_path in used_materials.items():
        source_spec = library_layer.GetPrimAtPath(prim_path)
        if not source_spec:
            logger.warning(
                f"Material '{material_name}' not found at {prim_path} in library"
            )
            continue

        target = _target_path(prim_path)
        success = Sdf.CopySpec(
            library_layer,
            Sdf.Path(prim_path),
            target_layer,
            Sdf.Path(target),
        )
        if success:
            _remap_asset_paths_in_prim(
                target_layer, Sdf.Path(target), library_dir, output_dir
            )
            path_remap[prim_path] = target
            copied += 1
            logger.debug(f"Copied material '{material_name}' ({prim_path} → {target})")
        else:
            logger.error(f"Failed to copy material '{material_name}' ({prim_path})")

    logger.info(f"Copied {copied}/{len(used_materials)} materials from library")
    return path_remap


def _remap_asset_paths_in_prim(
    layer: Sdf.Layer,
    prim_path: Sdf.Path,
    source_dir: Path,
    target_dir: Path,
) -> None:
    """Remap SdfAssetPath values in a prim and its descendants.

    Converts relative paths that were relative to *source_dir* so they
    resolve correctly from *target_dir*.
    """
    from pxr import Sdf

    prim_spec = layer.GetPrimAtPath(prim_path)
    if not prim_spec:
        return

    for attr_name in list(prim_spec.attributes.keys()):
        attr_spec = prim_spec.attributes[attr_name]
        value = attr_spec.default
        if isinstance(value, Sdf.AssetPath):
            new_path = _remap_single_asset_path(value.path, source_dir, target_dir)
            if new_path != value.path:
                attr_spec.default = Sdf.AssetPath(new_path)
        elif isinstance(value, Sdf.AssetPathArray):
            new_arr = Sdf.AssetPathArray(
                [
                    Sdf.AssetPath(
                        _remap_single_asset_path(ap.path, source_dir, target_dir)
                    )
                    for ap in value
                ]
            )
            if new_arr != value:
                attr_spec.default = new_arr

    # Recurse into children
    for child_spec in prim_spec.nameChildren:
        _remap_asset_paths_in_prim(
            layer,
            prim_path.AppendChild(child_spec.name),
            source_dir,
            target_dir,
        )


def _remap_single_asset_path(path_str: str, source_dir: Path, target_dir: Path) -> str:
    """Remap a single asset path from source_dir-relative to target_dir-relative."""
    if not path_str or "://" in path_str or os.path.isabs(path_str):
        return path_str

    abs_path = (source_dir / path_str).resolve()
    try:
        new_rel = os.path.relpath(abs_path, target_dir)
    except ValueError:
        return abs_path.as_posix()

    return new_rel.replace("\\", "/")


def _write_binding_over(
    layer: Sdf.Layer,
    prim_path: str,
    material_prim_path: str,
    scene_layer: Sdf.Layer | None = None,
) -> int:
    """Write a material:binding over for *prim_path* and any GeomSubset children.

    If *scene_layer* is provided, GeomSubset children of the target prim are
    discovered from it and given the same binding.  This prevents the original
    scene's per-subset ``Diffuse_*`` bindings from overriding our parent-level
    material assignment (USD resolves child bindings before parent bindings).

    Returns the number of binding overs written (1 + subsets).
    """
    from pxr import Sdf

    written = 0

    prim_spec = Sdf.CreatePrimInLayer(layer, prim_path)
    if not prim_spec:
        return 0
    prim_spec.specifier = Sdf.SpecifierOver

    api_name = "MaterialBindingAPI"
    api_schemas = prim_spec.GetInfo("apiSchemas")
    if not api_schemas or api_name not in api_schemas.prependedItems:
        prim_spec.SetInfo(
            "apiSchemas",
            Sdf.TokenListOp.Create(prependedItems=[api_name]),
        )

    binding_rel = prim_spec.relationships.get(
        "material:binding"
    ) or Sdf.RelationshipSpec(prim_spec, "material:binding")
    binding_rel.targetPathList.explicitItems = [Sdf.Path(material_prim_path)]
    written += 1

    # Override GeomSubset children so their original bindings don't win
    if scene_layer:
        scene_spec = scene_layer.GetPrimAtPath(prim_path)
        if scene_spec:
            for child_name in scene_spec.nameChildren.keys():
                child_spec = scene_spec.nameChildren[child_name]
                if child_spec.typeName == "GeomSubset":
                    child_path = prim_path + "/" + child_name
                    child_over = Sdf.CreatePrimInLayer(layer, child_path)
                    if not child_over:
                        continue
                    child_over.specifier = Sdf.SpecifierOver

                    child_api = child_over.GetInfo("apiSchemas")
                    if not child_api or api_name not in child_api.prependedItems:
                        child_over.SetInfo(
                            "apiSchemas",
                            Sdf.TokenListOp.Create(prependedItems=[api_name]),
                        )

                    child_rel = child_over.relationships.get(
                        "material:binding"
                    ) or Sdf.RelationshipSpec(child_over, "material:binding")
                    child_rel.targetPathList.explicitItems = [
                        Sdf.Path(material_prim_path)
                    ]
                    written += 1

    return written


def _write_material_bindings(
    layer: Sdf.Layer,
    prim_to_material: dict[str, str],
    name_to_prim: dict[str, str],
    skip_prefixes: set[str] | None = None,
) -> int:
    """Write material:binding over-specs into the layer for all predictions.

    Args:
        layer: The output Sdf.Layer.
        prim_to_material: Dict mapping prim path → material name.
        name_to_prim: Dict mapping material name → material prim path.
        skip_prefixes: Prim path prefixes to skip (e.g. instance paths that
            will be handled by prototype source remapping instead).

    Returns:
        Number of bindings written.
    """
    from pxr import Sdf

    written = 0
    skipped = 0
    # Sort once for O(log S) binary-search skip checks (instead of O(S) per prim).
    sorted_skip = sorted(skip_prefixes) if skip_prefixes else []
    idx = _PathIndex(prim_to_material)
    for prim_path, material_name in prim_to_material.items():
        # Skip bindings under instance paths that will be remapped to
        # prototype source paths (writing overs here breaks instancing).
        if sorted_skip and idx.is_under_any(prim_path, sorted_skip):
            skipped += 1
            continue

        material_prim_path = name_to_prim.get(material_name)
        if not material_prim_path:
            continue

        prim_spec = Sdf.CreatePrimInLayer(layer, prim_path)
        if not prim_spec:
            logger.warning(f"Failed to create over spec for {prim_path}")
            continue
        prim_spec.specifier = Sdf.SpecifierOver

        # Ensure MaterialBindingAPI is applied so ComputeBoundMaterial works.
        api_name = "MaterialBindingAPI"
        api_schemas = prim_spec.GetInfo("apiSchemas")
        if not api_schemas or api_name not in api_schemas.prependedItems:
            prim_spec.SetInfo(
                "apiSchemas",
                Sdf.TokenListOp.Create(prependedItems=[api_name]),
            )

        binding_rel = prim_spec.relationships.get(
            "material:binding"
        ) or Sdf.RelationshipSpec(prim_spec, "material:binding")
        binding_rel.targetPathList.explicitItems = [Sdf.Path(material_prim_path)]
        written += 1

    if skipped:
        logger.info(
            f"Skipped {skipped} bindings under instance paths (handled by prototype remapping)"
        )
    logger.info(f"Wrote {written} material bindings")
    return written


def _target_path_exists_or_payload_backed(
    stage: Any,
    target_path: str,
    member_path: str,
) -> bool:
    """Return True when target exists, or may exist inside an unloaded payload."""
    prim = stage.GetPrimAtPath(target_path)
    if prim and prim.IsValid():
        return True

    current = target_path
    while current and current != "/":
        if current == member_path or current.startswith(member_path + "/"):
            ancestor = stage.GetPrimAtPath(current)
            if ancestor and ancestor.IsValid():
                has_payload = getattr(ancestor, "HasPayload", None)
                if callable(has_payload) and has_payload():
                    return True
                has_authored_payloads = getattr(ancestor, "HasAuthoredPayloads", None)
                if callable(has_authored_payloads) and has_authored_payloads():
                    return True
        if current == member_path:
            break
        current = current.rsplit("/", 1)[0] or "/"

    return False


def _propagate_instance_bindings(
    manifest: SceneManifest,
    prim_to_material: dict[str, str],
    name_to_prim: dict[str, str],
    layer: Sdf.Layer,
    scene_usd_path: Path | None = None,
) -> int:
    """Propagate material bindings from representative to instance group members.

    For each instance group, remaps the representative's prim bindings to
    every non-representative member's prim path namespace.

    Args:
        manifest: Scene manifest with instance groups.
        prim_to_material: Merged predictions dict.
        name_to_prim: Material name → prim path mapping.
        layer: Output layer to write bindings into.

    Returns:
        Number of propagated bindings written.
    """
    from pxr import Usd

    written = 0
    _path_idx = _PathIndex(prim_to_material)
    scene_stage = (
        Usd.Stage.Open(str(scene_usd_path), Usd.Stage.LoadNone)
        if scene_usd_path
        else None
    )
    scene_layer = scene_stage.GetRootLayer() if scene_stage is not None else None

    for ig in manifest.instance_groups:
        if not ig.representative_id:
            continue

        rep = manifest.get_asset_by_id(ig.representative_id)
        if not rep:
            continue

        # Determine the source prefix for binding collection.
        # Normally the representative's prim_path matches a member_path.
        # When the representative is a descendant of a member (LLM split the
        # member into children), use the member path as the source prefix
        # and collect bindings from ALL sub-assets under that member.
        rep_prefix = rep.prim_path
        source_prefix = rep_prefix
        for mp in ig.member_paths:
            if rep_prefix.startswith(mp + "/"):
                source_prefix = mp
                break

        # Collect ALL bindings under the source prefix
        source_bindings = _path_idx.get_paths_under(source_prefix)

        if not source_bindings:
            continue

        # Remap to each non-source member
        for member_path in ig.member_paths:
            if member_path == source_prefix:
                continue

            this_member = 0
            direct_targets: set[str] = set()
            member_bindings = _path_idx.get_paths_under(member_path)
            if member_bindings:
                for target_prim, mat_name in member_bindings.items():
                    material_prim_path = name_to_prim.get(mat_name)
                    if not material_prim_path:
                        continue
                    if (
                        scene_stage is not None
                        and not _target_path_exists_or_payload_backed(
                            scene_stage, target_prim, member_path
                        )
                    ):
                        continue
                    direct_targets.add(target_prim)
                    count = _write_binding_over(
                        layer,
                        target_prim,
                        material_prim_path,
                        scene_layer=scene_layer,
                    )
                    written += count
                    this_member += count

            for src_prim, mat_name in source_bindings.items():
                material_prim_path = name_to_prim.get(mat_name)
                if not material_prim_path:
                    continue

                # Remap: /World/SourceMember/Mesh → /World/OtherMember/Mesh
                relative = src_prim[len(source_prefix) :]
                target_prim = member_path + relative
                if target_prim in direct_targets:
                    continue
                if (
                    scene_stage is not None
                    and not _target_path_exists_or_payload_backed(
                        scene_stage, target_prim, member_path
                    )
                ):
                    continue

                count = _write_binding_over(
                    layer,
                    target_prim,
                    material_prim_path,
                    scene_layer=scene_layer,
                )
                written += count
                this_member += count

            if this_member == 0 and source_bindings and scene_stage is not None:
                source_meshes = _collect_mesh_paths_from_stage(
                    scene_stage, source_prefix
                )
                member_meshes = _collect_mesh_paths_from_stage(scene_stage, member_path)
                this_member = _write_ordered_mesh_bindings(
                    layer=layer,
                    source_meshes=source_meshes,
                    target_meshes=member_meshes,
                    source_bindings=source_bindings,
                    name_to_prim=name_to_prim,
                    scene_layer=scene_layer,
                    scene_stage=scene_stage,
                    skip_targets=direct_targets,
                )
                written += this_member
                if this_member:
                    logger.debug(
                        f"Fallback: propagated {this_member} ordered mesh "
                        f"bindings to renamed member {member_path}"
                    )

        if source_bindings:
            logger.info(
                f"Propagated {len(source_bindings)} bindings to "
                f"{len(ig.member_paths) - 1} members of '{ig.group_name}'"
            )

    if written:
        logger.info(f"Wrote {written} propagated instance bindings")
    return written


def _compose_prototype_payloads(
    scene_usd_path: Path,
    manifest: SceneManifest,
    prim_to_material: dict[str, str],
    name_to_prim: dict[str, str],
    composed_layer: Sdf.Layer,
) -> int:
    """Compose prototype payload materials by remapping bindings to prototype source paths.

    For scenes where instances reference class-specifier prototype source prims
    via local references (e.g. Siemens NX), the sublayer-based payload arc
    rewriting has no effect (no sublayers exist).  Instead, this function:

    1. Opens the master scene stage to resolve instance → prototype source mappings.
    2. For each completed prototype payload group, collects bindings from
       ``prim_to_material`` that fall under the representative instance path.
    3. Remaps those binding paths from the instance namespace to the prototype
       source namespace.
    4. Authors the remapped bindings as overs in the composed layer.

    Because all instances reference the prototype source via local references,
    USD composition automatically propagates these materials to every instance.

    Args:
        scene_usd_path: Path to the original USD scene.
        manifest: Scene manifest with completed payload groups.
        prim_to_material: Merged predictions (prim path → material name).
        name_to_prim: Material name → material prim path mapping.
        composed_layer: The output Sdf.Layer to write bindings into.

    Returns:
        Number of prototype bindings written.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(str(scene_usd_path.resolve()), Usd.Stage.LoadNone)
    if not stage:
        logger.warning(f"Failed to open scene stage: {scene_usd_path}")
        return 0

    root_layer = stage.GetRootLayer()
    written = 0
    _path_idx = _PathIndex(prim_to_material)

    completed_pgs = [
        pg
        for pg in manifest.payload_groups
        if pg.status == "completed" and pg.instance_paths
    ]

    if not completed_pgs:
        return 0

    logger.info(
        f"Composing {len(completed_pgs)} prototype payloads "
        f"via prototype source path remapping"
    )

    for pg in completed_pgs:
        representative_path = pg.instance_paths[0]

        # Look up the local reference target (prototype source path)
        prim_spec = root_layer.GetPrimAtPath(representative_path)
        if not prim_spec:
            logger.warning(
                f"No prim spec for representative '{representative_path}' "
                f"in {pg.group_name}"
            )
            continue

        refs = prim_spec.referenceList.prependedItems
        if not refs or not refs[0].primPath:
            logger.debug(
                f"No local reference on '{representative_path}' — skipping {pg.group_name}"
            )
            continue

        proto_source_path = str(refs[0].primPath)

        # Collect bindings under the representative instance path
        rep_bindings = _path_idx.get_paths_under(representative_path)

        if not rep_bindings:
            logger.debug(f"No bindings for {pg.group_name}")
            continue

        # Remap bindings from instance path → prototype source path
        pg_written = 0
        for inst_prim_path, mat_name in rep_bindings.items():
            material_prim_path = name_to_prim.get(mat_name)
            if not material_prim_path:
                continue

            # Remap: /Scene/.../Instance/Mesh/X → /Scene/Prototypes/Instance/Mesh/X
            relative = inst_prim_path[len(representative_path) :]
            target_prim_path = proto_source_path + relative

            pg_written += _write_binding_over(
                composed_layer,
                target_prim_path,
                material_prim_path,
                scene_layer=root_layer,
            )

        written += pg_written
        logger.info(
            f"Remapped {pg_written} bindings for '{pg.group_name}': "
            f"{representative_path} → {proto_source_path} "
            f"({len(pg.instance_paths)} instances)"
        )

    if written:
        logger.info(f"Wrote {written} prototype source bindings total")
    return written


def _collect_mesh_paths_from_layer(layer: Sdf.Layer, root_path: str) -> list[str]:
    """Recursively collect Mesh prim paths from an Sdf.Layer under *root_path*."""
    result: list[str] = []

    def _walk(path: str) -> None:
        spec = layer.GetPrimAtPath(path)
        if not spec:
            return
        if spec.typeName == "Mesh":
            result.append(path)
        for child_name in spec.nameChildren.keys():
            _walk(path + "/" + child_name)

    _walk(root_path)
    return result


def _collect_mesh_paths_from_stage(stage: Any, root_path: str) -> list[str]:
    """Collect composed Mesh prim paths from a USD stage under *root_path*."""
    from pxr import Usd, UsdGeom

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        return []
    return [
        str(prim.GetPath())
        for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies())
        if prim.IsA(UsdGeom.Mesh)
    ]


def _material_for_source_mesh(
    source_mesh: str,
    source_bindings: dict[str, str],
) -> str | None:
    """Return the material assigned to *source_mesh*, descendants, or ancestors."""
    material = source_bindings.get(source_mesh)
    if material:
        return material

    ancestor = source_mesh.rsplit("/", 1)[0]
    while ancestor:
        material = source_bindings.get(ancestor)
        if material:
            return material
        if "/" not in ancestor.strip("/"):
            break
        ancestor = ancestor.rsplit("/", 1)[0]

    prefix = source_mesh + "/"
    child_materials = [
        mat for path, mat in source_bindings.items() if path.startswith(prefix)
    ]
    if not child_materials:
        return None
    return Counter(child_materials).most_common(1)[0][0]


def _mesh_attr_len(attr: Any) -> int | None:
    """Return the authored value length for a mesh topology attribute."""
    try:
        value = attr.Get()
    except (RuntimeError, TypeError, ValueError):
        return None
    if value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return 1


def _mesh_structural_fingerprint(
    scene_stage: Any | None,
    mesh_path: str,
) -> tuple[int | None, int | None, int | None] | None:
    """Return a cheap mesh-shape fingerprint from composed topology counts."""
    if scene_stage is None:
        return None

    from pxr import UsdGeom

    prim = scene_stage.GetPrimAtPath(mesh_path)
    if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        return None

    mesh = UsdGeom.Mesh(prim)
    fingerprint = (
        _mesh_attr_len(mesh.GetPointsAttr()),
        _mesh_attr_len(mesh.GetFaceVertexCountsAttr()),
        _mesh_attr_len(mesh.GetFaceVertexIndicesAttr()),
    )
    if all(part is None for part in fingerprint):
        return None
    return fingerprint


def _ordered_mesh_fingerprint_mismatches(
    scene_stage: Any | None,
    source_meshes: list[str],
    target_meshes: list[str],
) -> tuple[list[tuple[str, str, tuple[int | None, ...], tuple[int | None, ...]]], int]:
    """Compare same-position mesh fingerprints for ordered fallback mapping."""
    if scene_stage is None:
        return [], 0

    mismatches = []
    checked = 0
    for source_mesh, target_mesh in zip(source_meshes, target_meshes, strict=False):
        source_fp = _mesh_structural_fingerprint(scene_stage, source_mesh)
        target_fp = _mesh_structural_fingerprint(scene_stage, target_mesh)
        if source_fp is None or target_fp is None:
            continue
        checked += 1
        if source_fp != target_fp:
            mismatches.append((source_mesh, target_mesh, source_fp, target_fp))
    return mismatches, checked


def _write_dominant_mesh_bindings(
    *,
    layer: Sdf.Layer,
    target_meshes: list[str],
    source_bindings: dict[str, str],
    name_to_prim: dict[str, str],
    scene_layer: Sdf.Layer | None,
    skip_targets: set[str],
    reason: str,
) -> int:
    """Apply the dominant source material when ordered mesh mapping is unsafe."""
    if not source_bindings:
        return 0
    dominant_material = Counter(source_bindings.values()).most_common(1)[0][0]
    material_prim_path = name_to_prim.get(dominant_material)
    if not material_prim_path:
        return 0

    logger.warning(
        "Fallback: %s; applying dominant material '%s' to %d target mesh(es)",
        reason,
        dominant_material,
        len(target_meshes),
    )

    written = 0
    for target_mesh in target_meshes:
        if target_mesh in skip_targets:
            continue
        existing = layer.GetPrimAtPath(target_mesh)
        if existing and existing.relationships.get("material:binding"):
            continue
        written += _write_binding_over(
            layer,
            target_mesh,
            material_prim_path,
            scene_layer=scene_layer,
        )
    return written


def _write_ordered_mesh_bindings(
    *,
    layer: Sdf.Layer,
    source_meshes: list[str],
    target_meshes: list[str],
    source_bindings: dict[str, str],
    name_to_prim: dict[str, str],
    scene_layer: Sdf.Layer | None = None,
    scene_stage: Any | None = None,
    skip_targets: set[str] | None = None,
) -> int:
    """Propagate renamed duplicate bindings by traversal order.

    Some CAD exports duplicate the same sub-asset with different numbered node
    and mesh names.  Exact relative-path remapping then writes nothing even
    though the source and target contain the same mesh sequence.  In that case,
    preserve the representative's per-mesh material detail by mapping source
    mesh N to target mesh N instead of collapsing the member to one dominant
    material.
    """
    if not source_meshes or not target_meshes:
        return 0

    skip_targets = skip_targets or set()

    if len(source_meshes) != len(target_meshes):
        return _write_dominant_mesh_bindings(
            layer=layer,
            target_meshes=target_meshes,
            source_bindings=source_bindings,
            name_to_prim=name_to_prim,
            scene_layer=scene_layer,
            skip_targets=skip_targets,
            reason=(
                "ordered mesh count mismatch "
                f"({len(source_meshes)} source, {len(target_meshes)} target)"
            ),
        )

    mismatches, checked = _ordered_mesh_fingerprint_mismatches(
        scene_stage,
        source_meshes,
        target_meshes,
    )
    if mismatches:
        source_mesh, target_mesh, source_fp, target_fp = mismatches[0]
        return _write_dominant_mesh_bindings(
            layer=layer,
            target_meshes=target_meshes,
            source_bindings=source_bindings,
            name_to_prim=name_to_prim,
            scene_layer=scene_layer,
            skip_targets=skip_targets,
            reason=(
                "ordered mesh structural fingerprint mismatch "
                f"({len(mismatches)}/{checked} checked; "
                f"{source_mesh} {source_fp} != {target_mesh} {target_fp})"
            ),
        )

    if checked:
        logger.debug(
            "Fallback: propagating %d ordered mesh bindings after %d "
            "matching structural fingerprint check(s)",
            len(source_meshes),
            checked,
        )
    else:
        logger.debug(
            "Fallback: propagating %d ordered mesh bindings without "
            "structural fingerprint checks; topology attributes unavailable",
            len(source_meshes),
        )

    written = 0
    for source_mesh, target_mesh in zip(source_meshes, target_meshes, strict=False):
        if target_mesh in skip_targets:
            continue

        material = _material_for_source_mesh(source_mesh, source_bindings)
        if not material:
            continue
        material_prim_path = name_to_prim.get(material)
        if not material_prim_path:
            continue

        existing = layer.GetPrimAtPath(target_mesh)
        if existing and existing.relationships.get("material:binding"):
            continue

        written += _write_binding_over(
            layer,
            target_mesh,
            material_prim_path,
            scene_layer=scene_layer,
        )

    return written


def _remap_instance_group_bindings(
    scene_usd_path: Path,
    manifest: SceneManifest,
    prim_to_material: dict[str, str],
    name_to_prim: dict[str, str],
    composed_layer: Sdf.Layer,
    skip_paths: set[str] | None = None,
) -> int:
    """Remap instance group bindings to prototype source paths.

    For sub-assets in instance groups whose prims are ``instanceable=true``
    with local references to prototype source prims, direct binding overs
    at the instance path are invisible.  This function remaps bindings from
    each instance group representative's path to the prototype source path
    found via the prim's local reference.

    Args:
        scene_usd_path: Path to the original USD scene.
        manifest: Scene manifest.
        prim_to_material: Merged predictions (prim path → material name).
        name_to_prim: Material name → material prim path mapping.
        composed_layer: Output layer to write bindings into.
        skip_paths: Instance paths already handled by payload group remapping.

    Returns:
        Number of prototype bindings written.
    """
    from pxr import Usd

    skip_paths = skip_paths or set()

    stage = Usd.Stage.Open(str(scene_usd_path.resolve()), Usd.Stage.LoadNone)
    if not stage:
        return 0

    root_layer = stage.GetRootLayer()
    written = 0
    _path_idx = _PathIndex(prim_to_material)

    for ig in manifest.instance_groups:
        if not ig.representative_id:
            continue

        rep = manifest.get_asset_by_id(ig.representative_id)
        if not rep or rep.status != "completed":
            continue

        rep_path = rep.prim_path

        # When the representative is a descendant of a member (LLM split
        # the member into children), use the member path as source prefix
        # and collect bindings from ALL sub-assets under that member.
        source_prefix = rep_path
        for mp in ig.member_paths:
            if rep_path.startswith(mp + "/"):
                source_prefix = mp
                break

        rep_in_skip = source_prefix in skip_paths

        # Check if the source prim has a local reference (prototype source)
        prim_spec = root_layer.GetPrimAtPath(source_prefix)
        if not prim_spec:
            continue

        refs = prim_spec.referenceList.prependedItems
        has_proto_ref = refs and refs[0].primPath
        proto_source_path = str(refs[0].primPath) if has_proto_ref else None

        # Collect bindings under the source prefix
        source_bindings = _path_idx.get_paths_under(source_prefix)

        if not source_bindings:
            continue

        # No local reference → fall back to direct path propagation
        # (e.g. sub-assembly instance groups without prototype sources).
        if not has_proto_ref:
            for member_path in ig.member_paths:
                if member_path == source_prefix:
                    continue

                this_member = 0
                # If gap filling or restoration produced exact predictions
                # under this member path, bind those paths directly.  This
                # handles structural duplicates whose instance container names
                # differ from the representative, where naive relative path
                # remapping would point at non-existent prims.
                direct_targets: set[str] = set()
                member_bindings = _path_idx.get_paths_under(member_path)
                if member_bindings:
                    for target, mat_name in member_bindings.items():
                        material_prim_path = name_to_prim.get(mat_name)
                        if not material_prim_path:
                            continue
                        if not _target_path_exists_or_payload_backed(
                            stage, target, member_path
                        ):
                            continue
                        direct_targets.add(target)
                        existing = composed_layer.GetPrimAtPath(target)
                        if existing and existing.relationships.get("material:binding"):
                            continue
                        count = _write_binding_over(
                            composed_layer,
                            target,
                            material_prim_path,
                            scene_layer=root_layer,
                        )
                        written += count
                        this_member += count

                for src_prim, mat_name in source_bindings.items():
                    material_prim_path = name_to_prim.get(mat_name)
                    if not material_prim_path:
                        continue
                    relative = src_prim[len(source_prefix) :]
                    target = member_path + relative
                    if target in direct_targets:
                        continue
                    if not _target_path_exists_or_payload_backed(
                        stage, target, member_path
                    ):
                        continue
                    count = _write_binding_over(
                        composed_layer,
                        target,
                        material_prim_path,
                        scene_layer=root_layer,
                    )
                    written += count
                    this_member += count

                if this_member == 0 and source_bindings:
                    source_meshes = _collect_mesh_paths_from_stage(stage, source_prefix)
                    member_meshes = _collect_mesh_paths_from_stage(stage, member_path)
                    this_member = _write_ordered_mesh_bindings(
                        layer=composed_layer,
                        source_meshes=source_meshes,
                        target_meshes=member_meshes,
                        source_bindings=source_bindings,
                        name_to_prim=name_to_prim,
                        scene_layer=root_layer,
                        scene_stage=stage,
                    )
                    written += this_member
                    if this_member:
                        logger.debug(
                            f"Fallback: propagated {this_member} ordered mesh "
                            f"bindings to renamed member {member_path}"
                        )

            logger.info(
                f"Direct-propagated {len(source_bindings)} bindings to "
                f"{len(ig.member_paths) - 1} members of '{ig.group_name}' "
                f"(no prototype ref)"
            )
            continue

        # Remap to representative's prototype source path (skip if already
        # handled by _compose_prototype_payloads for payload groups).
        ig_written = 0
        if rep_in_skip:
            logger.debug(
                f"Rep '{rep_path}' in skip_paths — prototype already handled, "
                f"propagating to {len(ig.member_paths)} members only"
            )
        for src_prim_path, mat_name in source_bindings.items():
            material_prim_path = name_to_prim.get(mat_name)
            if not material_prim_path:
                continue

            relative = src_prim_path[len(source_prefix) :]
            target_prim_path = proto_source_path + relative

            # Skip if already written by _compose_prototype_payloads
            existing = composed_layer.GetPrimAtPath(target_prim_path)
            if existing and existing.relationships.get("material:binding"):
                continue

            ig_written += _write_binding_over(
                composed_layer,
                target_prim_path,
                material_prim_path,
                scene_layer=root_layer,
            )

        written += ig_written

        # Propagate the same bindings to each member's prototype source path.
        # Structural duplicates have identical mesh hierarchies but separate
        # prototype source prims — USD does not share them.
        member_written = 0
        for member_path in ig.member_paths:
            if member_path == source_prefix or member_path in skip_paths:
                continue
            member_spec = root_layer.GetPrimAtPath(member_path)
            if not member_spec:
                continue
            member_refs = member_spec.referenceList.prependedItems
            if not member_refs or not member_refs[0].primPath:
                continue
            member_proto = str(member_refs[0].primPath)
            if member_proto == proto_source_path:
                # Same prototype — already handled by USD composition
                continue

            this_member = 0
            for src_prim_path, mat_name in source_bindings.items():
                material_prim_path = name_to_prim.get(mat_name)
                if not material_prim_path:
                    continue
                relative = src_prim_path[len(source_prefix) :]
                target = member_proto + relative

                existing = composed_layer.GetPrimAtPath(target)
                if existing and existing.relationships.get("material:binding"):
                    continue

                # Only write if the target prim actually exists in the scene
                member_proto_spec = root_layer.GetPrimAtPath(target)
                if not member_proto_spec:
                    continue

                this_member += _write_binding_over(
                    composed_layer,
                    target,
                    material_prim_path,
                    scene_layer=root_layer,
                )

            # Fallback: if 1:1 path mapping found nothing (different internal
            # structure), preserve per-mesh detail by mapping the representative
            # mesh traversal order to the member prototype mesh traversal order.
            if this_member == 0 and source_bindings:
                source_meshes = _collect_mesh_paths_from_stage(stage, source_prefix)
                member_meshes = _collect_mesh_paths_from_layer(root_layer, member_proto)
                this_member = _write_ordered_mesh_bindings(
                    layer=composed_layer,
                    source_meshes=source_meshes,
                    target_meshes=member_meshes,
                    source_bindings=source_bindings,
                    name_to_prim=name_to_prim,
                    scene_layer=root_layer,
                    scene_stage=stage,
                )
                if this_member:
                    logger.debug(
                        f"Fallback: propagated {this_member} ordered mesh "
                        f"bindings to renamed member proto {member_proto}"
                    )

            member_written += this_member

        written += member_written
        total_ig = ig_written + member_written
        if total_ig:
            logger.info(
                f"Remapped {ig_written} IG bindings for '{ig.group_name}' "
                f"+ {member_written} member prototype bindings "
                f"({len(ig.member_paths)} members)"
            )

    if written:
        logger.info(f"Wrote {written} instance group prototype bindings")
    return written


def _rewrite_scene_payload_arcs(
    scene_usd_path: Path,
    manifest: SceneManifest,
    output_dir: Path,
) -> list[str]:
    """Create modified copies of scene sublayers with rewritten payload arcs.

    For each sublayer of the scene (e.g., ``Assets_Phase_01.usd``), creates a
    copy where payload arcs point to the updated versions (``output.usd`` from
    the bottom-up pipeline). The composed scene sublayers these modified copies
    instead of the originals, so USD composition naturally flows materials
    through the payload chain to all instance prims.

    Also includes the original scene as the weakest sublayer for any prims
    that don't have updated payloads (e.g., non-payload references).

    Args:
        scene_usd_path: Path to the original USD scene.
        manifest: Scene manifest with completed payload groups.
        output_dir: Directory for modified sublayer copies.

    Returns:
        List of sublayer paths for the composed scene (modified copies +
        original scene as fallback).
    """
    import shutil

    from pxr import Sdf

    from material_agent.scene.payload_dag_utils import rewrite_arcs_in_layer

    scene_layers_dir = output_dir / "scene_layers"
    scene_layers_dir.mkdir(parents=True, exist_ok=True)

    root_layer = Sdf.Layer.FindOrOpen(str(scene_usd_path.resolve()))
    if not root_layer:
        raise RuntimeError(f"Failed to open scene: {scene_usd_path}")

    # Build a cascaded map: original payload file (abs) -> materialized output USD.
    # Processed bottom-up so each parent can reference children's materialized USDs.
    payload_update_map = _build_cascaded_payload_map(
        manifest, output_dir, rewrite_arcs_in_layer, shutil
    )

    if not payload_update_map:
        logger.warning("No updated payloads — using original scene as-is")
        return [str(scene_usd_path.resolve())]

    logger.info(
        f"Rewriting payload arcs in scene sublayers "
        f"({len(payload_update_map)} updated payloads)"
    )

    modified_sublayer_paths: list[str] = []
    total_rewritten = 0

    for sl_path in root_layer.subLayerPaths:
        resolved_sl = root_layer.ComputeAbsolutePath(sl_path)
        if not resolved_sl or not Path(resolved_sl).exists():
            # Keep unresolvable sublayers as-is
            modified_sublayer_paths.append(resolved_sl or sl_path)
            continue

        # Copy the sublayer to our working directory
        sl_name = Path(resolved_sl).stem + "_updated" + Path(resolved_sl).suffix
        modified_sl = scene_layers_dir / sl_name
        shutil.copy2(resolved_sl, str(modified_sl))

        # Rewrite payload arcs in the copy, resolving from the original
        # location (relative paths in the layer are relative to where the
        # file was authored, not where we copied it)
        sl_layer = Sdf.Layer.FindOrOpen(str(modified_sl))
        if sl_layer:
            count = rewrite_arcs_in_layer(
                sl_layer, payload_update_map, resolve_from=resolved_sl
            )
            sl_layer.Save()
            total_rewritten += count
            if count:
                logger.info(f"Rewrote {count} payload arcs in {sl_name}")

        modified_sublayer_paths.append(str(modified_sl.resolve()))

    logger.info(f"Total payload arcs rewritten in scene sublayers: {total_rewritten}")
    return modified_sublayer_paths


def _build_cascaded_payload_map(
    manifest: SceneManifest,
    output_dir: Path,
    rewrite_arcs_in_layer: Any,
    shutil: Any,
) -> dict[str, str]:
    """Build a cascaded map of original payload file (abs) -> materialized output USD.

    Processes payload groups bottom-up (deepest first) so that when a parent
    payload is processed, its children's materialized USDs are already known.
    For each payload with child arcs that need rewriting:
      1. Creates ``payload_copies/{name}_base.usd`` — copy of the original payload
         USD with child payload arcs rewritten to child materialized outputs.
      2. Creates ``payload_copies/{name}.usd`` — copy of the pipeline output.usd
         with its sublayer updated from the original to the base copy.
    Payloads whose original USD has no child arcs requiring rewriting are
    mapped directly to their existing output.usd (no copy needed).

    Returns:
        Dict mapping absolute original payload path -> cascaded output path.
    """
    from pxr import Sdf

    payload_copies_dir = output_dir / "payload_copies"
    payload_copies_dir.mkdir(parents=True, exist_ok=True)

    completed_pgs = [
        pg
        for pg in manifest.payload_groups
        if pg.status == "completed"
        and pg.output_usd_path
        and pg.payload_file
        and Path(pg.output_usd_path).exists()
    ]

    # Process leaves-first (depth=0) so children are in cascaded_map before parents
    sorted_pgs = sorted(completed_pgs, key=lambda pg: pg.depth or 0)

    cascaded_map: dict[str, str] = {}

    for pg in sorted_pgs:
        orig_file = pg.payload_file
        orig_abs = str(Path(orig_file).resolve())
        output_usd = pg.output_usd_path

        # Create a copy of the original payload USD and rewrite child arcs
        base_copy = payload_copies_dir / f"{pg.group_name}_base.usd"
        shutil.copy2(orig_file, str(base_copy))

        base_layer = Sdf.Layer.FindOrOpen(str(base_copy))
        child_arc_count = 0
        if base_layer:
            child_arc_count = rewrite_arcs_in_layer(
                base_layer, cascaded_map, resolve_from=orig_file
            )
            if child_arc_count:
                base_layer.Save()
            else:
                base_copy.unlink(missing_ok=True)

        if child_arc_count == 0:
            # No children to cascade — original output.usd is correct as-is
            cascaded_map[orig_abs] = output_usd
            continue

        # Create a copy of output.usd with its sublayer updated to base_copy
        output_copy = payload_copies_dir / f"{pg.group_name}.usd"
        shutil.copy2(output_usd, str(output_copy))

        out_layer = Sdf.Layer.FindOrOpen(str(output_copy))
        if out_layer:
            new_sublayers = []
            for sl in out_layer.subLayerPaths:
                resolved_sl = str(Path(out_layer.ComputeAbsolutePath(sl)).resolve())
                if resolved_sl == orig_abs:
                    new_sublayers.append(str(base_copy.resolve()))
                else:
                    new_sublayers.append(sl)
            out_layer.subLayerPaths = new_sublayers
            out_layer.Save()

        cascaded_map[orig_abs] = str(output_copy)
        logger.debug(
            f"Cascaded '{pg.group_name}': rewrote {child_arc_count} child payload arcs"
        )

    # Second pass: payload groups with no output_usd_path but with a
    # modified_input_path (SO-optimized USD). These modified USDs already
    # reference materialized child outputs. Map original → modified so the
    # scene uses the materialized version.
    no_output_pgs = [
        pg
        for pg in manifest.payload_groups
        if pg.status == "completed"
        and not pg.output_usd_path
        and pg.payload_file
        and pg.modified_input_path
        and Path(pg.modified_input_path).exists()
    ]
    for pg in no_output_pgs:
        orig_abs = str(Path(pg.payload_file).resolve())
        if orig_abs in cascaded_map:
            continue  # already handled
        cascaded_map[orig_abs] = pg.modified_input_path
        logger.debug(
            f"Mapped (no-output) '{pg.group_name}': original -> modified_input_path"
        )

    logger.info(
        f"Built cascaded payload map: {len(cascaded_map)} entries "
        f"({sum(1 for v in cascaded_map.values() if 'payload_copies' in v)} with cascaded rewrites)"
    )
    return cascaded_map


def _process_payload_groups(
    manifest: SceneManifest,
    composed_layer: Sdf.Layer,
    output_usd_path: Path,
    material_library_yaml: Path,
    name_to_prim: dict[str, str],
) -> int:
    """Create per-payload material layers and inject via ``prepend payloads``.

    Each payload layer is self-contained: material definitions are placed
    inside the ``defaultPrim`` subtree (e.g., ``/<defaultPrim>/Looks/...``)
    so that binding targets are within the payload's namespace scope.  This
    preserves instancing — all instances sharing the same payload file share
    a single prototype that includes the material opinions.

    For each completed payload group:
    1. Load predictions from the payload pipeline run.
    2. Create a self-contained material layer with material defs + bindings,
       all paths scoped within ``defaultPrim``.
    3. Write ``prepend payloads`` arcs on each instance prim.

    Args:
        manifest: Scene manifest with payload groups.
        composed_layer: The output Sdf.Layer being built.
        output_usd_path: Path to the output USD (for path calculations).
        material_library_yaml: Path to the materials library YAML.
        name_to_prim: Material name → prim path mapping.

    Returns:
        Number of payload arcs written.
    """
    from pxr import Sdf

    payload_layers_dir = output_usd_path.parent / "payload_layers"
    payload_layers_dir.mkdir(parents=True, exist_ok=True)

    library_usd_path, lib_name_to_prim = _load_material_library(material_library_yaml)

    total_arcs = 0

    for pg in manifest.payload_groups:
        if pg.status != "completed":
            logger.debug(f"Skipping payload '{pg.group_name}': status={pg.status}")
            continue

        # Load predictions for this payload
        predictions = _load_payload_predictions(pg)
        if not predictions:
            logger.warning(
                f"No predictions found for completed payload '{pg.group_name}'"
            )
            continue

        # Identify used materials
        used_materials: dict[str, str] = {}
        for _prim_path, mat_name in predictions.items():
            if mat_name in lib_name_to_prim:
                used_materials[mat_name] = lib_name_to_prim[mat_name]

        # Resolve defaultPrim from the original payload file
        original_layer = Sdf.Layer.FindOrOpen(pg.payload_file)
        default_prim = ""
        if original_layer and original_layer.defaultPrim:
            default_prim = original_layer.defaultPrim

        # Create per-payload material layer (self-contained, scope-safe)
        payload_layer_path = payload_layers_dir / f"{pg.group_name}.usd"
        _create_payload_material_layer(
            payload_layer_path=payload_layer_path,
            default_prim=default_prim,
            predictions=predictions,
            used_materials=used_materials,
            library_usd_path=library_usd_path,
            name_to_prim=lib_name_to_prim,
        )
        pg.material_layer_path = str(payload_layer_path)

        # Write prepend payloads arcs on each instance prim
        arcs = _write_payload_arcs(
            composed_layer=composed_layer,
            instance_paths=pg.instance_paths,
            payload_layer_path=payload_layer_path,
            output_usd_path=output_usd_path,
        )
        total_arcs += arcs

        logger.info(
            f"Payload '{pg.group_name}': {len(predictions)} bindings, "
            f"{len(used_materials)} materials, {arcs} instance arcs"
        )

    logger.info(f"Wrote {total_arcs} payload arcs across all payload groups")
    return total_arcs


def _load_payload_predictions(pg: PayloadGroup) -> dict[str, str]:
    """Load predictions from a payload group's pipeline output.

    Returns:
        Dict mapping prim path → material name.
    """
    predictions: dict[str, str] = {}

    predictions_path = None
    if pg.predictions_path:
        p = Path(pg.predictions_path)
        if p.exists():
            predictions_path = p

    if not predictions_path and pg.working_dir:
        # Try raw predictions
        raw = Path(pg.working_dir) / "predictions" / "predictions.jsonl"
        if raw.exists():
            predictions_path = raw

    if not predictions_path:
        return predictions

    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                prediction = json.loads(line)
            except json.JSONDecodeError:
                continue

            prim_id = prediction.get("id")
            material = _extract_material_name(prediction)
            if prim_id and material:
                predictions[prim_id] = material

    logger.info(f"Loaded {len(predictions)} predictions for payload '{pg.group_name}'")
    return predictions


def _create_payload_material_layer(
    payload_layer_path: Path,
    default_prim: str,
    predictions: dict[str, str],
    used_materials: dict[str, str],
    library_usd_path: Path | None,
    name_to_prim: dict[str, str],
) -> None:
    """Create a self-contained material layer scoped within ``defaultPrim``.

    All paths — material definitions AND binding targets — are placed under
    the ``defaultPrim`` subtree so they stay within the payload's namespace
    scope when USD composes a ``prepend payloads`` arc.

    For example, if ``defaultPrim`` is ``World``, the library material at
    ``/World/Looks/Steel_Yellow`` is relocated to ``/World/Looks/Steel_Yellow``
    (already under ``/World`` — no change needed for this KION convention).
    The binding target also points there, so both remap consistently when
    USD maps ``/World`` → the instance prim.

    Args:
        payload_layer_path: Where to write the payload material layer.
        default_prim: The defaultPrim name from the original payload file.
        predictions: Dict mapping prim path → material name.
        used_materials: Dict mapping material name → library prim path.
        library_usd_path: Path to the material library USD.
        name_to_prim: Material name → prim path mapping.
    """
    from pxr import Sdf
    from world_understanding.utils.usd.material import ensure_looks_scope_spec

    payload_layer_path.parent.mkdir(parents=True, exist_ok=True)

    layer = Sdf.Layer.CreateNew(str(payload_layer_path))
    if not layer:
        raise RuntimeError(f"Failed to create payload layer: {payload_layer_path}")

    if default_prim:
        layer.defaultPrim = default_prim

    # Determine the prefix for relocating material paths into the
    # defaultPrim subtree.  Library materials live at e.g. /World/Looks/...
    # which is already under /World (the typical defaultPrim).  If the
    # defaultPrim differs, we remap.
    dp_prefix = f"/{default_prim}" if default_prim else ""

    # Build a remapped name→path dict where material paths are under
    # the defaultPrim.  For each library path like /World/Looks/Mat,
    # check if it's already under dp_prefix; if not, relocate it.
    scoped_name_to_prim: dict[str, str] = {}
    library_to_scoped: dict[str, str] = {}  # original lib path → scoped path
    for mat_name, lib_path in name_to_prim.items():
        if mat_name not in used_materials:
            continue
        if dp_prefix and lib_path.startswith(dp_prefix + "/"):
            # Already scoped — e.g. /World/Looks/Mat under /World
            scoped_path = lib_path
        elif dp_prefix:
            # Relocate: /World/Looks/Mat → /<defaultPrim>/Looks/Mat
            # Strip the first path component and prepend defaultPrim
            parts = lib_path.split("/")
            # parts[0] is '', parts[1] is 'World', rest is the material path
            if len(parts) > 2:
                scoped_path = dp_prefix + "/" + "/".join(parts[2:])
            else:
                scoped_path = dp_prefix + lib_path
        else:
            scoped_path = lib_path
        scoped_name_to_prim[mat_name] = scoped_path
        library_to_scoped[lib_path] = scoped_path

    # Copy material definitions from library, then relocate if needed
    if used_materials and library_usd_path:
        _copy_materials_from_library(
            layer, library_usd_path, used_materials, payload_layer_path
        )
        # Relocate material specs that need to move under defaultPrim
        for lib_path, scoped_path in library_to_scoped.items():
            if lib_path != scoped_path and layer.GetPrimAtPath(lib_path):
                # Ensure parent hierarchy exists at the target
                target_parent = Sdf.Path(scoped_path).GetParentPath()
                while target_parent != Sdf.Path.absoluteRootPath:
                    if not layer.GetPrimAtPath(target_parent):
                        parent_spec = Sdf.CreatePrimInLayer(layer, target_parent)
                        if parent_spec:
                            parent_spec.specifier = Sdf.SpecifierOver
                    # Payload overlay parents remain overs; Scope is just the
                    # type opinion for the Looks container.
                    ensure_looks_scope_spec(layer, target_parent, allow_over=True)
                    target_parent = target_parent.GetParentPath()
                # Copy to new path, then remove old
                Sdf.CopySpec(layer, Sdf.Path(lib_path), layer, Sdf.Path(scoped_path))
                _remove_prim_spec(layer, lib_path)

    # Write material bindings using scoped paths
    written = _write_material_bindings(layer, predictions, scoped_name_to_prim)

    layer.Save()
    logger.debug(
        f"Created payload material layer: {payload_layer_path} "
        f"({written} bindings, {len(used_materials)} materials)"
    )


def _remove_prim_spec(layer: Sdf.Layer, prim_path: str) -> None:
    """Remove a prim spec and its descendants from a layer."""
    from pxr import Sdf

    spec = layer.GetPrimAtPath(prim_path)
    if not spec:
        return
    parent_path = Sdf.Path(prim_path).GetParentPath()
    parent_spec = layer.GetPrimAtPath(parent_path)
    if parent_spec:
        del parent_spec.nameChildren[spec.name]
    else:
        del layer.pseudoRoot.nameChildren[spec.name]


def _write_payload_arcs(
    composed_layer: Sdf.Layer,
    instance_paths: list[str],
    payload_layer_path: Path,
    output_usd_path: Path,
) -> int:
    """Write ``prepend payloads`` arcs on instance prims.

    For each instance prim path, creates an ``over`` spec in the composed
    layer and adds the payload material layer as a prepended payload.

    Args:
        composed_layer: The output Sdf.Layer.
        instance_paths: Scene paths of instance prims.
        payload_layer_path: Path to the per-payload material layer.
        output_usd_path: Path to the output USD (for relative path calculation).

    Returns:
        Number of arcs written.
    """
    from pxr import Sdf

    # Compute relative path from output USD to payload layer
    try:
        rel_path = os.path.relpath(
            payload_layer_path.resolve(), output_usd_path.resolve().parent
        )
    except ValueError:
        rel_path = str(payload_layer_path.resolve())
    # Normalize to forward slashes for USD
    rel_path = rel_path.replace("\\", "/")

    written = 0
    for prim_path in instance_paths:
        prim_spec = Sdf.CreatePrimInLayer(composed_layer, prim_path)
        if not prim_spec:
            logger.warning(f"Failed to create over spec for instance {prim_path}")
            continue
        prim_spec.specifier = Sdf.SpecifierOver

        # Add the payload material layer as a prepended payload.
        # No de-instancing needed: the material layer is scoped within
        # defaultPrim, so all instances sharing this payload file get
        # identical prepend-payload lists and can share a prototype.
        payload = Sdf.Payload(rel_path)
        existing = list(prim_spec.payloadList.prependedItems)
        if payload not in existing:
            prim_spec.payloadList.prependedItems = [payload] + existing
            written += 1

    return written


def compose_material_layers(
    scene_usd_path: Path,
    manifest: SceneManifest,
    output_usd_path: Path,
    names_filter: list[str] | None = None,
) -> Path:
    """Create a composed USD with material layers sublayered over the original scene.

    Output structure:
        output.usd
          sublayer[0]: asset_a/output.usd   (material "over" prims)
          sublayer[1]: asset_b/output.usd   (material "over" prims)
          ...
          sublayer[N]: original_scene.usd   (base geometry)

    For instance groups, the representative's material layer is propagated
    to other members by creating remapped "over" layers.

    Args:
        scene_usd_path: Path to the original USD scene.
        manifest: Scene manifest with completed assets.
        output_usd_path: Where to write the composed USD.
        names_filter: Optional name/path filter for assets.

    Returns:
        Path to the composed USD file.
    """
    from pxr import Sdf

    output_usd_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect material layers from completed assets (deduplicate by resolved path).
    # Each output layer may sublayer its extracted USD for standalone use, but
    # for scene composition we only need the material opinions (bindings +
    # material definitions).  We strip sublayers so the extracted geometry
    # doesn't conflict with the base scene.
    assets = manifest.get_processable_assets(names_filter)
    layer_paths: list[str] = []
    seen_paths: set[str] = set()
    stripped_dir = output_usd_path.parent / "material_layers"
    stripped_dir.mkdir(parents=True, exist_ok=True)

    for sa in assets:
        if sa.status != "completed" or not sa.material_layer_path:
            logger.warning(
                f"Skipping '{sa.name}': status={sa.status}, "
                f"layer={sa.material_layer_path}"
            )
            continue

        layer_path = Path(sa.material_layer_path)
        if not layer_path.exists():
            logger.warning(f"Material layer not found: {layer_path}")
            continue

        resolved = str(layer_path.resolve())
        if resolved in seen_paths:
            logger.info(f"Skipping duplicate layer: {sa.name} -> {layer_path}")
            continue
        seen_paths.add(resolved)

        # Strip sublayers from the output layer so only material opinions
        # (bindings + material defs) are included in the composed scene.
        use_path = _strip_sublayers(layer_path, stripped_dir, sa.name)
        layer_paths.append(str(use_path))
        logger.info(f"Including material layer: {sa.name} -> {use_path}")

    # Generate instance propagation layers
    instance_layers = _create_instance_propagation_layers(
        manifest, output_usd_path.parent
    )
    layer_paths.extend(instance_layers)

    if not layer_paths:
        logger.warning("No material layers found to compose")

    # Create the composed stage
    logger.info(f"Composing {len(layer_paths)} material layers over {scene_usd_path}")

    # Create a new layer for the composed output
    composed_layer = Sdf.Layer.CreateNew(str(output_usd_path))
    if not composed_layer:
        raise RuntimeError(f"Failed to create output layer: {output_usd_path}")

    # Add sublayers: material layers first (stronger opinions), then original scene
    sublayer_paths = layer_paths + [str(scene_usd_path.resolve())]
    composed_layer.subLayerPaths = sublayer_paths

    # Copy stage metadata (upAxis, defaultPrim) from the original scene
    source_layer = Sdf.Layer.FindOrOpen(str(scene_usd_path.resolve()))
    if source_layer:
        _copy_layer_stage_metadata(source_layer, composed_layer)

    composed_layer.Save()

    logger.info(
        f"Composed USD saved to {output_usd_path} "
        f"({len(layer_paths)} material layers + original scene)"
    )
    return output_usd_path


def _strip_sublayers(layer_path: Path, output_dir: Path, asset_name: str) -> Path:
    """Create a copy of a material layer with sublayer references removed.

    Output layers sublayer the extracted/original USD for standalone use.
    When composing into a scene, we only need the material opinions (bindings
    and material definitions) — the base scene provides the geometry.
    Keeping the sublayers would introduce duplicate geometry with potentially
    conflicting transforms (e.g. baked pivots from the extract step).
    """
    import shutil

    from pxr import Sdf

    source = Sdf.Layer.FindOrOpen(str(layer_path))
    if not source or not source.subLayerPaths:
        # No sublayers to strip — use the original
        return layer_path.resolve()

    num_sublayers = len(source.subLayerPaths)

    # Copy the file and remove sublayer references from the copy.
    stripped_path = output_dir / f"{asset_name}.usd"
    shutil.copy2(str(layer_path.resolve()), str(stripped_path))
    stripped = Sdf.Layer.FindOrOpen(str(stripped_path))
    del stripped.subLayerPaths[:]
    stripped.Save()

    logger.debug(
        "Stripped %d sublayer(s) from %s -> %s",
        num_sublayers,
        layer_path.name,
        stripped_path.name,
    )
    return stripped_path


def _create_instance_propagation_layers(
    manifest: SceneManifest,
    output_dir: Path,
) -> list[str]:
    """Create layers that propagate material assignments from representative to instances.

    For each instance group, copies the representative's material "over" prims
    with paths remapped to each instance member.

    Args:
        manifest: Scene manifest with instance groups.
        output_dir: Directory to write propagation layers.

    Returns:
        List of paths to generated propagation layers.
    """
    from pxr import Sdf

    generated_layers: list[str] = []

    for ig in manifest.instance_groups:
        if not ig.representative_id:
            continue

        rep = manifest.get_asset_by_id(ig.representative_id)
        if not rep or not rep.material_layer_path:
            continue

        rep_layer_path = Path(rep.material_layer_path)
        if not rep_layer_path.exists():
            continue

        # Load the representative's material layer
        rep_layer = Sdf.Layer.FindOrOpen(str(rep_layer_path))
        if not rep_layer:
            continue

        # For each non-representative member, create a remapped layer
        for member_path in ig.member_paths:
            if member_path == rep.prim_path:
                continue

            # Find the member sub-asset (if it exists)
            member_sa = None
            for sa in manifest.sub_assets:
                if sa.prim_path == member_path:
                    member_sa = sa
                    break

            if member_sa and member_sa.material_layer_path:
                # Member already has its own layer, skip
                continue

            # Create remapped layer
            layer_name = f"instance_propagation_{ig.group_name}_{_path_to_filename(member_path)}.usd"
            layer_path = output_dir / layer_name

            try:
                _remap_layer_prims(
                    source_layer=rep_layer,
                    source_prefix=rep.prim_path,
                    target_prefix=member_path,
                    output_path=layer_path,
                )
                generated_layers.append(str(layer_path.resolve()))
                logger.info(f"Propagated materials: {rep.prim_path} -> {member_path}")
            except Exception:
                logger.exception(f"Failed to propagate materials to {member_path}")

    return generated_layers


def _remap_layer_prims(
    source_layer: Sdf.Layer,
    source_prefix: str,
    target_prefix: str,
    output_path: Path,
) -> None:
    """Create a new layer with prim paths remapped from source to target prefix.

    Walks the source layer's full hierarchy.  For the subtree rooted at
    *source_prefix* the prim paths are remapped to *target_prefix*; all
    other prims (e.g. ``/World/Looks`` material definitions) are copied
    as-is so that material binding targets still resolve.

    Args:
        source_layer: The source Sdf.Layer with material "over" prims.
        source_prefix: Prim path prefix to replace (e.g., "/Root/Robot_A").
        target_prefix: New prim path prefix (e.g., "/Root/Robot_B").
        output_path: Where to write the remapped layer.
    """
    from pxr import Sdf

    new_layer = Sdf.Layer.CreateNew(str(output_path))
    if not new_layer:
        raise RuntimeError(f"Failed to create layer: {output_path}")

    source_sdf = Sdf.Path(source_prefix)

    def _walk_and_copy(spec: Sdf.PrimSpec, path: Sdf.Path) -> None:
        """Recursively walk specs, remapping paths under source_prefix."""
        path_str = str(path)

        # Determine the destination path
        if path_str == source_prefix or path_str.startswith(source_prefix + "/"):
            # Inside source subtree: remap
            relative = path_str[len(source_prefix) :]
            dst_path = Sdf.Path(target_prefix + relative)
        else:
            # Outside source subtree (ancestors, /World, etc.): keep as-is
            dst_path = path

        # Create the prim spec in the new layer
        dst_spec = Sdf.CreatePrimInLayer(new_layer, dst_path)
        if dst_spec:
            dst_spec.specifier = spec.specifier

            # Copy all properties and relationships via CopySpec
            for prop in spec.properties:
                Sdf.CopySpec(
                    source_layer,
                    path.AppendProperty(prop.name),
                    new_layer,
                    dst_path.AppendProperty(prop.name),
                )

        # Recurse into children
        for child_name in spec.nameChildren.keys():
            child_spec = spec.nameChildren[child_name]
            child_path = path.AppendChild(child_name)
            child_str = str(child_path)

            # Skip subtrees that are not ancestors of source_prefix,
            # not under source_prefix, and not shared prims like /World
            is_ancestor = source_prefix.startswith(child_str + "/")
            is_inside = child_str.startswith(source_prefix)
            is_shared = not child_str.startswith(str(source_sdf.GetPrefixes()[0]))

            if is_ancestor or is_inside or is_shared:
                _walk_and_copy(child_spec, child_path)

    for root_spec in source_layer.rootPrims:
        _walk_and_copy(root_spec, root_spec.path)

    new_layer.Save()


def _camera_config_number(
    camera_config: dict[str, Any] | None,
    *keys: str,
) -> float | None:
    if not camera_config:
        return None
    for key in keys:
        value = camera_config.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.warning("Ignoring non-numeric camera config value %s=%r", key, value)
    return None


def _camera_config_int(
    camera_config: dict[str, Any] | None,
    *keys: str,
) -> int | None:
    value = _camera_config_number(camera_config, *keys)
    return int(value) if value is not None else None


def _safe_render_suffix(value: str) -> str:
    suffix = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    return suffix.strip("_") or "configured"


def _apply_render_light_overrides(
    stage: Any,
    *,
    dome_light: float | None,
    distant_light: float | None,
) -> None:
    if dome_light is None and distant_light is None:
        return

    from pxr import Gf, UsdGeom, UsdLux
    from world_understanding.utils.usd.prim import remove_all_lights

    remove_all_lights(stage)
    if dome_light is not None:
        dome = UsdLux.DomeLight.Define(stage, "/RenderLights/DomeLight")
        dome.GetIntensityAttr().Set(dome_light)
    if distant_light is not None:
        distant = UsdLux.DistantLight.Define(stage, "/RenderLights/DistantLight")
        distant.GetIntensityAttr().Set(distant_light)
        xform = UsdGeom.Xformable(distant)
        xform.AddRotateXYZOp().Set(Gf.Vec3f(315, 45, 0))


def render_composed_scene(
    composed_usd_path: Path,
    output_dir: Path,
    camera_corners: list[str] | None = None,
    image_width: int = 1024,
    image_height: int = 1024,
    camera_margin: float = 1.0,
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    clear_materials: bool = False,
    camera_config: dict[str, Any] | None = None,
) -> list[Path]:
    """Render the composed scene from multiple camera angles.

    Args:
        composed_usd_path: Path to the composed USD file.
        output_dir: Directory to save rendered images.
        camera_corners: Camera directions (default: ["+x+y+z", "-x-y-z"]).
        image_width: Render width in pixels.
        image_height: Render height in pixels.
        camera_margin: Camera margin multiplier.
        background_color: Background RGB color (0-1 range).
        clear_materials: If True, strip original material bindings before
            rendering so only the newly-assigned materials are visible.
        camera_config: Optional explicit camera/light config. Supports the
            JSON shape used by legacy scene camera files, e.g.
            ``focal_length_mm``, ``cam_x/y/z``, ``target_x/y/z``,
            ``dome_light``, ``distant_light``, ``image_width``, and
            ``image_height``.

    Returns:
        List of paths to rendered images.
    """
    import base64
    from io import BytesIO

    from PIL import Image
    from pxr import Sdf, Usd
    from world_understanding.functions.graphics.rendering import (
        RemoteRenderingBackend,
        format_direction_for_filename,
    )
    from world_understanding.utils.image_utils import paste_on_background
    from world_understanding.utils.usd.camera import add_corner_view_camera

    if camera_corners is None:
        camera_corners = ["+x+y+z", "-x-y-z"]

    if camera_config:
        configured_width = _camera_config_int(camera_config, "image_width")
        configured_height = _camera_config_int(camera_config, "image_height")
        if configured_width is not None:
            image_width = configured_width
        if configured_height is not None:
            image_height = configured_height
    if image_width <= 0 or image_height <= 0:
        raise ValueError("camera_config image_width and image_height must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Opening composed scene for rendering: {composed_usd_path}")

    if clear_materials:
        # To clear only original bindings while keeping pipeline-assigned ones,
        # we rebuild the composition: first flatten the base scene with all
        # bindings stripped, then layer our material layers on top, then flatten
        # the result for rendering.
        from world_understanding.utils.usd.prim import nullify_materials

        logger.info("Clearing original material bindings (clear_materials=True)")

        # Read the composed layer to get sublayer paths
        composed_layer = Sdf.Layer.FindOrOpen(str(composed_usd_path))
        if not composed_layer:
            raise RuntimeError(f"Failed to open layer: {composed_usd_path}")

        sublayer_paths = list(composed_layer.subLayerPaths)
        # Last sublayer is the original scene; everything before is material layers
        base_scene_path = sublayer_paths[-1]
        material_layer_paths = sublayer_paths[:-1]

        # Flatten the base scene first (resolves all references, instances,
        # and internal SubUSDs), then strip material bindings on the flat
        # result so that de-instancing doesn't break geometry during flatten.
        logger.info("Flattening base scene and clearing materials...")
        base_stage = Usd.Stage.Open(base_scene_path)
        if not base_stage:
            raise RuntimeError(f"Failed to open base scene: {base_scene_path}")
        cleaned_base_path = output_dir / "base_scene_cleared.usd"
        base_stage.Flatten().Export(str(cleaned_base_path))

        # Now nullify materials on the already-flat stage
        flat_base_stage = Usd.Stage.Open(str(cleaned_base_path))
        if not flat_base_stage:
            raise RuntimeError(f"Failed to open flattened base: {cleaned_base_path}")
        nullify_materials(flat_base_stage)
        flat_base_stage.GetRootLayer().Save()

        # Recompose: material layers over the cleaned base
        recomposed_layer = Sdf.Layer.CreateAnonymous()
        recomposed_layer.subLayerPaths = material_layer_paths + [str(cleaned_base_path)]
        _copy_layer_stage_metadata(composed_layer, recomposed_layer)
        recomposed_stage = Usd.Stage.Open(recomposed_layer)
        if not recomposed_stage:
            raise RuntimeError("Failed to open recomposed stage")

        flat_layer = recomposed_stage.Flatten()
    else:
        stage = Usd.Stage.Open(str(composed_usd_path))
        if not stage:
            raise RuntimeError(f"Failed to open stage: {composed_usd_path}")
        flat_layer = stage.Flatten()

    # Flatten for rendering (NVCF needs a self-contained stage)
    logger.info("Flattening composed stage for rendering...")
    flat_path = output_dir / "composed_scene_flat.usd"
    flat_layer.Export(str(flat_path))

    flat_stage = Usd.Stage.Open(str(flat_path))
    if not flat_stage:
        raise RuntimeError(f"Failed to open flattened stage: {flat_path}")

    _apply_render_light_overrides(
        flat_stage,
        dome_light=_camera_config_number(camera_config, "dome_light"),
        distant_light=_camera_config_number(camera_config, "distant_light"),
    )

    camera_specs: list[tuple[str, str, str, dict[str, float | None]]] = []
    if camera_config:
        direction = str(camera_config.get("direction") or "+x+y+z")
        camera_name = str(camera_config.get("name") or "configured")
        camera_specs.append(
            (
                direction,
                "/Cameras/SceneCamera_configured",
                _safe_render_suffix(camera_name),
                {
                    "focal_length": _camera_config_number(
                        camera_config, "focal_length_mm", "focal_length"
                    ),
                    "horizontal_aperture": _camera_config_number(
                        camera_config,
                        "aperture",
                        "horizontal_aperture",
                        "horizontal_aperture_mm",
                    ),
                    "cam_x": _camera_config_number(camera_config, "cam_x"),
                    "cam_y": _camera_config_number(camera_config, "cam_y"),
                    "cam_z": _camera_config_number(camera_config, "cam_z"),
                    "target_x": _camera_config_number(camera_config, "target_x"),
                    "target_y": _camera_config_number(camera_config, "target_y"),
                    "target_z": _camera_config_number(camera_config, "target_z"),
                    "margin": _camera_config_number(
                        camera_config, "margin", "camera_margin"
                    ),
                },
            )
        )
    else:
        for corner in camera_corners:
            camera_path = (
                f"/Cameras/SceneCamera_{corner.replace('+', 'p').replace('-', 'n')}"
            )
            camera_specs.append(
                (corner, camera_path, format_direction_for_filename(corner), {})
            )

    # Add cameras
    for direction, camera_path, _suffix, overrides in camera_specs:
        focal_length = overrides.get("focal_length") or 60.0
        horizontal_aperture = overrides.get("horizontal_aperture") or 36.0
        aspect_ratio = image_width / image_height
        if aspect_ratio >= 1.0:
            vertical_aperture = horizontal_aperture / aspect_ratio
        else:
            vertical_aperture = horizontal_aperture
            horizontal_aperture = vertical_aperture * aspect_ratio

        add_corner_view_camera(
            flat_stage,
            camera_path=camera_path,
            direction=direction,
            margin=overrides.get("margin") or camera_margin,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            cam_x=overrides.get("cam_x"),
            cam_y=overrides.get("cam_y"),
            cam_z=overrides.get("cam_z"),
            target_x=overrides.get("target_x"),
            target_y=overrides.get("target_y"),
            target_z=overrides.get("target_z"),
        )
    flat_stage.Save()

    # Render each camera
    rendering_backend = RemoteRenderingBackend()
    rendered_paths: list[Path] = []

    for direction, camera_path, suffix, _overrides in camera_specs:
        output_path = output_dir / f"composed_scene_{suffix}.png"

        logger.info(f"Rendering {direction} -> {output_path.name}")
        try:
            render_result = rendering_backend.render(
                stage=flat_stage,
                cameras=[camera_path],
                image_width=image_width,
                image_height=image_height,
                frames="0",
            )

            if (
                render_result
                and render_result.get("successful_cameras", 0) > 0
                and render_result.get("results")
            ):
                camera_result = render_result["results"][0]
                images = camera_result.get("images", [])
                if images:
                    image = images[0]
                    if hasattr(image, "save"):
                        pass  # Already a PIL Image
                    elif isinstance(image, dict) and "image" in image:
                        img_bytes = image["image"]
                        if not isinstance(img_bytes, bytes):
                            img_bytes = base64.b64decode(img_bytes)
                        image = Image.open(BytesIO(img_bytes))
                    else:
                        logger.warning(f"Unexpected image format for {direction}")
                        continue

                    # Apply background
                    if image.mode != "RGBA":
                        image = image.convert("RGBA")
                    bg = tuple(int(c * 255) for c in background_color)
                    image = paste_on_background(image, bg)

                    image.save(str(output_path))
                    rendered_paths.append(output_path)
                    logger.info(f"Saved render: {output_path.name}")
                else:
                    logger.warning(f"No image data for {direction}")
            else:
                error = "Unknown error"
                if render_result and render_result.get("results"):
                    for r in render_result["results"]:
                        if "error" in r:
                            error = r["error"]
                            break
                logger.error(f"Render failed for {direction}: {error}")
        except Exception:
            logger.exception(f"Error rendering {direction}")

    logger.info(f"Rendered {len(rendered_paths)}/{len(camera_specs)} camera views")
    return rendered_paths


def _path_to_filename(prim_path: str) -> str:
    """Convert a prim path to a safe filename fragment."""
    return prim_path.strip("/").replace("/", "_").lower()
