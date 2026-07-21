# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the large-scene manifest model."""

from pathlib import Path

from material_agent.scene.manifest import (
    InstanceGroup,
    PayloadGroup,
    SceneManifest,
    SubAsset,
)


def test_scene_manifest_save_uses_atomic_replacement(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("previous")

    manifest = SceneManifest(
        scene_usd_path="/scene.usda",
        sub_assets=[
            SubAsset(
                id="asset_a",
                name="AssetA",
                prim_path="/Root/AssetA",
            )
        ],
    )

    manifest.save(manifest_path)

    loaded = SceneManifest.load(manifest_path)
    assert loaded.scene_usd_path == "/scene.usda"
    assert loaded.sub_assets[0].id == "asset_a"
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_scene_manifest_filters_and_lookup_helpers(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.usda"
    payload_path.write_text("#usda 1.0\n")
    manifest = SceneManifest(
        sub_assets=[
            SubAsset(id="rep", name="Rep", prim_path="/Root/Rep"),
            SubAsset(id="member", name="Member", prim_path="/Root/Member"),
            SubAsset(id="child", name="Child", prim_path="/Root/Rep/Child"),
            SubAsset(
                id="skipped",
                name="Skipped",
                prim_path="/Root/Skipped",
                status="skipped",
            ),
        ],
        instance_groups=[
            InstanceGroup(
                group_name="dupes",
                representative_id="rep",
                member_paths=["/Root/Rep", "/Root/Member"],
            )
        ],
        payload_groups=[
            PayloadGroup(
                id="payload",
                group_name="Payload",
                payload_file=str(payload_path),
                depth=2,
            ),
            PayloadGroup(
                id="skipped-payload",
                group_name="SkippedPayload",
                payload_file=str(tmp_path / "skipped.usda"),
                status="skipped",
            ),
        ],
    )

    assert [asset.id for asset in manifest.get_processable_assets(["/Root/Rep"])] == [
        "rep",
        "child",
    ]
    assert [asset.id for asset in manifest.get_processable_assets(["REP"])] == ["rep"]
    assert manifest.get_processable_assets(["NoMatch"]) == []
    assert manifest.get_payloads_by_depth() == {2: [manifest.payload_groups[0]]}
    assert manifest.get_payload_by_file(str(payload_path)) is manifest.payload_groups[0]
    assert manifest.get_payload_by_file(str(tmp_path / "missing.usda")) is None
    assert manifest.get_asset_by_id("rep") is manifest.sub_assets[0]
    assert manifest.get_asset_by_id("missing") is None
    assert manifest.get_instance_group("dupes") is manifest.instance_groups[0]
    assert manifest.get_instance_group("missing") is None
    assert SceneManifest.timestamp()
