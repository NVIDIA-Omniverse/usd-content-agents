# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD composition arc analysis utilities.

Functions for inspecting Sdf layer stacks to extract references,
payloads, sublayers, and variant information.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pxr import Sdf, Usd  # pragma: no cover

logger = logging.getLogger(__name__)


def iter_prim_spec_paths(layer: Sdf.Layer) -> list[Sdf.Path]:
    """Recursively iterate all prim spec paths in a layer.

    Walks the Sdf layer's prim spec tree depth-first, collecting every
    prim spec path found (root prims and their descendants).

    Args:
        layer: The Sdf layer to walk.

    Returns:
        List of all prim spec paths found in the layer.
    """
    from pxr import Sdf

    paths: list[Sdf.Path] = []

    def _recurse(spec: Sdf.PrimSpec) -> None:
        paths.append(spec.path)
        for child_spec in spec.nameChildren:
            _recurse(child_spec)

    root_spec = layer.GetPrimAtPath(Sdf.Path.absoluteRootPath)
    if root_spec:
        for child_spec in root_spec.nameChildren:
            _recurse(child_spec)
    return paths


def collect_composition_arcs(stage: Usd.Stage) -> dict[str, Any]:
    """Walk the Sdf layer stack to collect references, payloads, sublayers, and variants.

    Inspects the root layer's prim specs for composition arcs across all
    list editor operations (prepended, appended, added, explicit).

    Args:
        stage: The USD stage to analyze.

    Returns:
        Dictionary containing:
            - sublayer_count: Number of sublayers
            - sublayer_paths: List of sublayer paths
            - reference_count: Total number of references
            - payload_count: Total number of payloads
            - variant_set_count: Total number of variant sets
            - unique_sub_usd_count: Number of unique referenced sub-USD files
            - sub_usd_files: List of dicts with asset_path, reference_count,
              referencing_prims — sorted by reference count descending
    """
    root_layer = stage.GetRootLayer()

    sublayer_paths = list(root_layer.subLayerPaths)

    ref_count = 0
    payload_count = 0
    variant_set_count = 0
    # asset_path -> {count, referencing_prims}
    sub_usd_map: dict[str, dict[str, Any]] = {}

    def _record_ref(asset: str, prim_path: Sdf.Path) -> None:
        nonlocal ref_count
        if not asset:
            return
        ref_count += 1
        if asset not in sub_usd_map:
            sub_usd_map[asset] = {
                "asset_path": asset,
                "reference_count": 0,
                "referencing_prims": [],
            }
        sub_usd_map[asset]["reference_count"] += 1
        sub_usd_map[asset]["referencing_prims"].append(str(prim_path))

    def _walk_layer(layer: Sdf.Layer) -> None:
        nonlocal ref_count, payload_count, variant_set_count
        for prim_path in iter_prim_spec_paths(layer):
            prim_spec = layer.GetPrimAtPath(prim_path)
            if not prim_spec:
                continue
            # References — check all list editor operations
            if prim_spec.referenceList:
                for ref in prim_spec.referenceList.prependedItems:
                    _record_ref(str(ref.assetPath) if ref.assetPath else "", prim_path)
                for ref in prim_spec.referenceList.appendedItems:
                    _record_ref(str(ref.assetPath) if ref.assetPath else "", prim_path)
                for ref in prim_spec.referenceList.addedItems:
                    _record_ref(str(ref.assetPath) if ref.assetPath else "", prim_path)
                for ref in prim_spec.referenceList.explicitItems:
                    _record_ref(str(ref.assetPath) if ref.assetPath else "", prim_path)
            # Payloads — check all list editor operations
            if prim_spec.payloadList:
                for pl in prim_spec.payloadList.prependedItems:
                    if pl.assetPath:
                        payload_count += 1
                for pl in prim_spec.payloadList.appendedItems:
                    if pl.assetPath:
                        payload_count += 1
                for pl in prim_spec.payloadList.addedItems:
                    if pl.assetPath:
                        payload_count += 1
                for pl in prim_spec.payloadList.explicitItems:
                    if pl.assetPath:
                        payload_count += 1
            # Variant sets
            if prim_spec.variantSets:
                variant_set_count += len(prim_spec.variantSets)

    _walk_layer(root_layer)

    sub_usd_files = sorted(sub_usd_map.values(), key=lambda x: -x["reference_count"])

    return {
        "sublayer_count": len(sublayer_paths),
        "sublayer_paths": sublayer_paths,
        "reference_count": ref_count,
        "payload_count": payload_count,
        "variant_set_count": variant_set_count,
        "unique_sub_usd_count": len(sub_usd_files),
        "sub_usd_files": sub_usd_files,
    }
