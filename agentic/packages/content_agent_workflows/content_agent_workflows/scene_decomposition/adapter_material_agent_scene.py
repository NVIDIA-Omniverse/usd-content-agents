# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter from Material Agent scene manifests to generic decomposition manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .manifest import (
    ArtifactReference,
    DecomposedAsset,
    DecompositionPolicy,
    SceneDecompositionManifest,
    SceneInstanceGroup,
    ScenePayloadGroup,
    ScenePrototypeGroup,
    StageMetadata,
)


def _get(obj: object, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _is_native_prototype_group(payload_group: object) -> bool:
    group_id = str(_get(payload_group, "id", ""))
    return group_id.startswith("proto_")


def _instance_member_asset_ids(
    sub_assets: list[object],
    instance_groups: list[object],
) -> set[str]:
    member_ids: set[str] = set()
    for group in instance_groups:
        representative_id = _get(group, "representative_id")
        member_paths = set(_get(group, "member_paths", []) or [])
        for asset in sub_assets:
            if (
                _get(asset, "prim_path") in member_paths
                and _get(asset, "id") != representative_id
            ):
                member_ids.add(str(_get(asset, "id")))
    return member_ids


def _representative_ids(instance_groups: list[object]) -> set[str]:
    return {
        str(rep_id)
        for rep_id in (_get(group, "representative_id") for group in instance_groups)
        if rep_id
    }


def _representative_by_group(instance_groups: list[object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in instance_groups:
        group_name = _get(group, "group_name")
        representative_id = _get(group, "representative_id")
        if group_name and representative_id:
            result[str(group_name)] = str(representative_id)
    return result


def _group_has_manifest_member(group: object, asset_paths: set[str]) -> bool:
    member_paths = set(_get(group, "member_paths", []) or [])
    return bool(member_paths & asset_paths)


def convert_material_agent_manifest(
    material_manifest: object,
    *,
    policy: DecompositionPolicy | None = None,
    mapping_artifacts: list[ArtifactReference] | None = None,
) -> SceneDecompositionManifest:
    """Convert a Material Agent scene manifest to the generic schema."""

    scene_path = str(_get(material_manifest, "scene_usd_path", ""))
    scene_id = Path(scene_path).stem if scene_path else "scene"
    analysis = dict(_get(material_manifest, "analysis", {}) or {})
    sub_assets = list(_get(material_manifest, "sub_assets", []) or [])
    material_instance_groups = list(
        _get(material_manifest, "instance_groups", []) or []
    )
    material_payload_groups = list(_get(material_manifest, "payload_groups", []) or [])
    asset_paths = {
        str(prim_path)
        for prim_path in (_get(asset, "prim_path") for asset in sub_assets)
        if prim_path
    }

    member_ids = _instance_member_asset_ids(sub_assets, material_instance_groups)
    representative_ids = _representative_ids(material_instance_groups)
    representative_by_group = _representative_by_group(material_instance_groups)

    assets: list[DecomposedAsset] = []
    for asset in sub_assets:
        asset_id = str(_get(asset, "id", ""))
        status = str(_get(asset, "status", "pending"))
        instance_group = _get(asset, "instance_group")
        processable = status != "skipped" and (
            asset_id not in member_ids or asset_id in representative_ids
        )
        skip_reason = None
        if status == "skipped":
            skip_reason = "status_skipped"
        elif not processable:
            skip_reason = "non_representative_instance_member"

        prim_path = str(_get(asset, "prim_path", ""))
        split_context = _get(asset, "split_context") or {}
        context = dict(split_context) if isinstance(split_context, dict) else {}

        assets.append(
            DecomposedAsset(
                asset_id=asset_id,
                label=str(_get(asset, "name", "")) or asset_id,
                original_root_path=prim_path,
                working_usd_path=_get(asset, "extracted_usd"),
                working_root_path=prim_path,
                source_path_prefixes=[prim_path] if prim_path else [],
                parent_group=_get(asset, "parent_group"),
                source_classification=_get(asset, "source_classification"),
                mesh_count=int(_get(asset, "mesh_count", 0) or 0),
                vertex_count=int(_get(asset, "vertex_count", 0) or 0),
                instance_group_id=str(instance_group) if instance_group else None,
                representative_asset_id=representative_by_group.get(str(instance_group))
                if instance_group
                else None,
                processable=processable,
                skip_reason=skip_reason,
                status=status,
                context=context,
            )
        )

    instance_groups = [
        SceneInstanceGroup(
            group_id=str(_get(group, "group_name", "")),
            label=str(_get(group, "group_name", "")),
            source_file=_get(group, "source_file"),
            instance_count=int(_get(group, "instance_count", 0) or 0),
            member_paths=list(_get(group, "member_paths", []) or []),
            representative_asset_id=_get(group, "representative_id"),
        )
        for group in material_instance_groups
        if _group_has_manifest_member(group, asset_paths)
    ]

    payload_groups: list[ScenePayloadGroup] = []
    prototype_groups: list[ScenePrototypeGroup] = []
    for group in material_payload_groups:
        group_id = str(_get(group, "id", ""))
        label = str(_get(group, "group_name", "")) or group_id
        if _is_native_prototype_group(group):
            prototype_groups.append(
                ScenePrototypeGroup(
                    group_id=group_id,
                    label=label,
                    representative_usd_path=_get(group, "payload_file") or None,
                    instance_count=int(_get(group, "instance_count", 0) or 0),
                    instance_paths=list(_get(group, "instance_paths", []) or []),
                    status=str(_get(group, "status", "pending")),
                )
            )
            continue

        payload_groups.append(
            ScenePayloadGroup(
                group_id=group_id,
                label=label,
                payload_file=str(_get(group, "payload_file", "")),
                instance_count=int(_get(group, "instance_count", 0) or 0),
                instance_paths=list(_get(group, "instance_paths", []) or []),
                depth=int(_get(group, "depth", 0) or 0),
                child_payload_files=list(_get(group, "child_payload_files", []) or []),
                parent_payload_files=list(
                    _get(group, "parent_payload_files", []) or []
                ),
                representative_usd_path=_get(group, "representative_path"),
                modified_input_path=_get(group, "modified_input_path"),
                output_usd_path=_get(group, "output_usd_path"),
                status=str(_get(group, "status", "pending")),
            )
        )

    return SceneDecompositionManifest(
        scene_id=scene_id,
        original_usd_path=scene_path,
        generated_at=str(_get(material_manifest, "generated_at", "")),
        stage_metadata=StageMetadata(
            default_prim_path=analysis.get("default_prim_path"),
            up_axis=analysis.get("up_axis"),
            meters_per_unit=analysis.get("meters_per_unit"),
        ),
        decomposition_policy=policy or DecompositionPolicy(),
        analysis=analysis,
        assets=assets,
        instance_groups=instance_groups,
        payload_groups=payload_groups,
        prototype_groups=prototype_groups,
        mapping_artifacts=mapping_artifacts or [],
        diagnostics=[],
    )
