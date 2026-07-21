# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for generic scene decomposition workflow contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from content_agent_workflows.scene_decomposition import (
    DecomposedAsset,
    DecompositionPolicy,
    ManifestCatalog,
    ManifestCatalogEntry,
    SceneDecompositionManifest,
    SceneDecompositionRequest,
    SceneDecompositionResult,
    SceneInstanceGroup,
    run_scene_decomposition,
)
from content_agent_workflows.scene_decomposition import cli as scene_cli
from content_agent_workflows.scene_decomposition import decomposition as scene_decomp
from content_agent_workflows.scene_decomposition.adapter_material_agent_scene import (
    convert_material_agent_manifest,
)


class FakeMaterialManifest(SimpleNamespace):
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"scene_usd_path": self.scene_usd_path}) + "\n",
            encoding="utf-8",
        )


def _fake_material_manifest() -> FakeMaterialManifest:
    return FakeMaterialManifest(
        scene_usd_path="/tmp/factory.usd",
        generated_at="2026-06-27T00:00:00Z",
        analysis={"total_meshes": 3, "total_payload_groups": 1},
        sub_assets=[
            SimpleNamespace(
                id="asset_a",
                name="AssetA",
                prim_path="/World/A",
                parent_group=None,
                source_classification="object",
                mesh_count=2,
                vertex_count=20,
                instance_group=None,
                split_context=None,
                extracted_usd=None,
                status="pending",
            ),
            SimpleNamespace(
                id="asset_b",
                name="AssetB",
                prim_path="/World/B",
                parent_group=None,
                source_classification="object",
                mesh_count=2,
                vertex_count=20,
                instance_group="inst_A",
                split_context={"siblings": ["AssetA"]},
                extracted_usd=None,
                status="pending",
            ),
        ],
        instance_groups=[
            SimpleNamespace(
                group_name="inst_A",
                source_file=None,
                instance_count=2,
                member_paths=["/World/A", "/World/B"],
                representative_id="asset_a",
            ),
            SimpleNamespace(
                group_name="filtered_group",
                source_file=None,
                instance_count=2,
                member_paths=["/World/FilteredA", "/World/FilteredB"],
                representative_id=None,
            ),
        ],
        payload_groups=[
            SimpleNamespace(
                id="payload_box",
                group_name="box",
                payload_file="/tmp/box.usd",
                instance_count=3,
                instance_paths=["/World/Box_1", "/World/Box_2"],
                depth=0,
                child_payload_files=[],
                parent_payload_files=[],
                representative_path=None,
                modified_input_path=None,
                output_usd_path=None,
                status="pending",
            ),
            SimpleNamespace(
                id="proto_bolt",
                group_name="bolt",
                payload_file="/tmp/bolt_proto.usd",
                instance_count=4,
                instance_paths=["/World/Bolt_1", "/World/Bolt_2"],
                status="pending",
            ),
        ],
    )


def test_scene_manifest_rejects_duplicate_asset_ids() -> None:
    with pytest.raises(ValueError, match="asset_id values must be unique"):
        SceneDecompositionManifest(
            scene_id="scene",
            original_usd_path="source.usda",
            assets=[
                DecomposedAsset(
                    asset_id="asset",
                    label="Asset A",
                    original_root_path="/World/A",
                ),
                DecomposedAsset(
                    asset_id="asset",
                    label="Asset B",
                    original_root_path="/World/B",
                ),
            ],
        )


def test_scene_manifest_rejects_empty_group_ids() -> None:
    with pytest.raises(ValueError, match="instance_groups.group_id"):
        SceneDecompositionManifest(
            scene_id="scene",
            original_usd_path="source.usda",
            instance_groups=[SceneInstanceGroup(group_id="", label="empty")],
        )


def test_scene_manifest_rejects_unknown_instance_representative() -> None:
    with pytest.raises(ValueError, match="references unknown representative asset"):
        SceneDecompositionManifest(
            scene_id="scene",
            original_usd_path="source.usda",
            assets=[
                DecomposedAsset(
                    asset_id="asset",
                    label="Asset",
                    original_root_path="/World/A",
                )
            ],
            instance_groups=[
                SceneInstanceGroup(
                    group_id="instances",
                    label="Instances",
                    representative_asset_id="missing",
                )
            ],
        )


def test_manifest_catalog_rejects_duplicate_manifest_ids() -> None:
    with pytest.raises(ValueError, match="manifest_id values must be unique"):
        ManifestCatalog(
            original_usd_path="source.usda",
            source_identity_digest="source-digest",
            structural_analysis_id="analysis",
            manifests=[
                ManifestCatalogEntry(
                    manifest_id="material",
                    intent="material_processing",
                    path="manifest_a.json",
                    manifest_digest="digest-a",
                ),
                ManifestCatalogEntry(
                    manifest_id="material",
                    intent="physics_processing",
                    path="manifest_b.json",
                    manifest_digest="digest-b",
                ),
            ],
        )


def test_convert_material_agent_manifest_marks_non_representative_assets() -> None:
    manifest = convert_material_agent_manifest(
        _fake_material_manifest(),
        policy=DecompositionPolicy(root_prim_path="/World"),
    )

    assert manifest.scene_id == "factory"
    assert manifest.original_usd_path == "/tmp/factory.usd"
    assert [asset.asset_id for asset in manifest.processable_assets] == ["asset_a"]
    asset_b = next(asset for asset in manifest.assets if asset.asset_id == "asset_b")
    assert not asset_b.processable
    assert asset_b.skip_reason == "non_representative_instance_member"
    assert asset_b.representative_asset_id == "asset_a"
    assert len(manifest.instance_groups) == 1
    assert manifest.instance_groups[0].group_id == "inst_A"
    assert len(manifest.payload_groups) == 1
    assert len(manifest.prototype_groups) == 1
    assert manifest.prototype_groups[0].representative_usd_path == "/tmp/bolt_proto.usd"


def test_run_scene_decomposition_writes_generic_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_analyze_scene(**kwargs):
        calls["analyze"] = kwargs
        manifest = _fake_material_manifest()
        manifest.scene_usd_path = str(kwargs["scene_usd_path"])
        return manifest

    def fake_extract_all(**kwargs):
        calls["extract"] = kwargs
        manifest = kwargs["manifest"]
        for asset in manifest.sub_assets:
            extracted_path = tmp_path / f"{asset.id}.usda"
            extracted_path.write_text("#usda 1.0\n", encoding="utf-8")
            asset.extracted_usd = str(extracted_path)
            asset.status = "extracted"
        return manifest

    monkeypatch.setattr(
        scene_decomp,
        "_load_material_agent_scene_functions",
        lambda: (fake_analyze_scene, fake_extract_all),
    )

    source_path = tmp_path / "factory.usd"
    source_path.write_text("#usda 1.0\n", encoding="utf-8")
    result = run_scene_decomposition(
        SceneDecompositionRequest(
            usd_path=source_path,
            output_dir=tmp_path / "run",
            root_prim_path="/World",
            detect_structural_duplicates=True,
            extract_assets=True,
        )
    )

    assert result.success
    assert result.asset_count == 2
    assert result.processable_asset_count == 1
    assert result.instance_group_count == 1
    assert result.payload_group_count == 1
    assert result.prototype_group_count == 1
    assert calls["analyze"]["filters"]["include_paths"] == ["/World"]
    assert calls["analyze"]["filters"]["detect_structural_duplicates"] is True
    assert calls["analyze"]["working_dir"] == tmp_path / "run" / "analysis_working"
    assert calls["extract"]["output_dir"] == tmp_path / "run" / "extracted"

    manifest_path = Path(result.manifest_path or "")
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["schema_version"].endswith("scene-decomposition-manifest.v1")
    assert manifest_data["assets"][0]["working_usd_path"] == str(
        tmp_path / "asset_a.usda"
    )
    assert Path(result.material_agent_manifest_path or "").exists()
    catalog_data = json.loads(
        Path(result.manifest_catalog_path or "").read_text(encoding="utf-8")
    )
    assert catalog_data["manifests"][0]["manifest_id"] == "default"
    phase_result_data = json.loads(
        Path(result.phase_result_path or "").read_text(encoding="utf-8")
    )
    assert phase_result_data["completion_policy_satisfied"] is True
    assert phase_result_data["output_digest"] == result.output_digest


def test_standalone_decomposition_input_digest_ignores_output_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_analyze_scene(**kwargs):
        manifest = _fake_material_manifest()
        manifest.scene_usd_path = str(kwargs["scene_usd_path"])
        return manifest

    monkeypatch.setattr(
        scene_decomp,
        "_load_material_agent_scene_functions",
        lambda: (fake_analyze_scene, lambda **kwargs: kwargs["manifest"]),
    )

    source_path = tmp_path / "factory.usd"
    source_path.write_text("#usda 1.0\n", encoding="utf-8")
    first = run_scene_decomposition(
        SceneDecompositionRequest(
            usd_path=source_path,
            output_dir=tmp_path / "run-a",
            root_prim_path="/World",
            llm_config={"provider": "ignored-a"},
        )
    )
    second = run_scene_decomposition(
        SceneDecompositionRequest(
            usd_path=source_path,
            output_dir=tmp_path / "run-b",
            root_prim_path="/World",
            llm_config={"provider": "ignored-b"},
        )
    )

    assert first.input_digest == second.input_digest


def test_scene_decomposition_failure_records_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_loader():
        raise RuntimeError("analysis backend unavailable")

    monkeypatch.setattr(
        scene_decomp, "_load_material_agent_scene_functions", fail_loader
    )

    source_path = tmp_path / "factory.usd"
    source_path.write_text("#usda 1.0\n", encoding="utf-8")

    result = run_scene_decomposition(
        SceneDecompositionRequest(
            usd_path=source_path,
            output_dir=tmp_path / "run-failed",
            root_prim_path="/World",
        )
    )

    assert not result.success
    phase_result_data = json.loads(
        Path(result.phase_result_path or "").read_text(encoding="utf-8")
    )
    assert phase_result_data["diagnostics"][0]["exception_type"] == "RuntimeError"
    assert (
        "analysis backend unavailable"
        in phase_result_data["diagnostics"][0]["traceback"]
    )


def test_scene_decomposition_cli_prints_result(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    captured_requests: list[SceneDecompositionRequest] = []

    def fake_run(request: SceneDecompositionRequest) -> SceneDecompositionResult:
        captured_requests.append(request)
        return SceneDecompositionResult(
            success=True,
            output_dir=str(request.output_dir),
            manifest_path=str(request.output_dir / "scene_manifest.json"),
            asset_count=1,
            processable_asset_count=1,
        )

    monkeypatch.setattr(scene_cli, "run_scene_decomposition", fake_run)

    rc = scene_cli.main(
        [
            str(tmp_path / "scene.usd"),
            "--output-dir",
            str(tmp_path / "run"),
            "--root-prim-path",
            "/World",
            "--extract-assets",
        ]
    )

    assert rc == 0
    assert captured_requests[0].root_prim_path == "/World"
    assert captured_requests[0].extract_assets is True
    output = json.loads(capsys.readouterr().out)
    assert output["success"] is True
    assert output["asset_count"] == 1
