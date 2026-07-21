# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional tests for material_agent.scene.validate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pxr import Sdf, Usd, UsdGeom, UsdShade

import material_agent.scene.validate as validate_mod
from material_agent.scene.validate import (
    AssetReport,
    PayloadReport,
    _check_topology_and_instances,
    _count_predictions,
    _validate_payload_group,
    format_asset_report,
    validate_asset,
    validate_scene,
    validate_scene_outputs,
)


def _make_stage(path: Path, meshes: list[str] | None = None) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    for mesh in meshes or []:
        UsdGeom.Mesh.Define(stage, mesh)
    stage.GetRootLayer().Save()
    return stage


def _make_materialized_scene(
    path: Path,
    *,
    bound_meshes: list[str] | None = None,
    unbound_meshes: list[str] | None = None,
) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    material = UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    for mesh_path in bound_meshes or []:
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    for mesh_path in unbound_meshes or []:
        UsdGeom.Mesh.Define(stage, mesh_path)
    stage.GetRootLayer().Save()
    return stage


def _write_predictions(
    working_dir: Path,
    *,
    relative_path: str = "predictions/predictions.jsonl",
    lines: list[str] | None = None,
) -> None:
    path = working_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines or ['{"id": "mesh-1"}']) + "\n")


def _write_state(
    working_dir: Path,
    *,
    completed_steps: list[str] | None = None,
    failed_steps: list[str] | None = None,
) -> None:
    (working_dir / ".pipeline_state.json").write_text(
        json.dumps(
            {
                "completed_steps": completed_steps or [],
                "failed_steps": failed_steps or [],
            }
        )
    )


class FakeRel:
    def __init__(self, targets: list[str]) -> None:
        self._targets = targets

    def GetTargets(self) -> list[str]:
        return self._targets


class FakePrim:
    def __init__(
        self, prim_type: str, target: str | None = None, *, instance_proxy: bool = True
    ) -> None:
        self._prim_type = prim_type
        self._target = target
        self._instance_proxy = instance_proxy

    def IsInstanceProxy(self) -> bool:
        return self._instance_proxy

    def IsA(self, schema: object) -> bool:
        if schema is UsdGeom.Mesh:
            return self._prim_type == "Mesh"
        if schema is UsdGeom.Subset:
            return self._prim_type == "GeomSubset"
        return False

    def GetTypeName(self) -> str:
        return self._prim_type

    def GetRelationship(self, name: str) -> FakeRel | None:
        assert name == "material:binding"
        if self._target is None:
            return None
        return FakeRel([self._target])


class FakeStage:
    def __init__(self, prims: list[FakePrim]) -> None:
        self._prims = prims

    def Traverse(self, *_args: object) -> list[FakePrim]:
        if _args:
            return self._prims
        return []


class TestCountPredictions:
    def test_skips_invalid_json_and_missing_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "predictions.jsonl"
        path.write_text('{"id": "one"}\n{"material": "steel"}\nnot-json\n\n')

        assert _count_predictions(path) == 1


class TestValidateAssetAdditional:
    def test_simulate_only_asset_is_treated_as_completed(self, tmp_path: Path) -> None:
        working_dir = tmp_path / ".asset"
        working_dir.mkdir()
        (working_dir / ".simulate").touch()
        _write_state(working_dir)
        _write_predictions(working_dir)

        report = validate_asset(working_dir)

        assert report.status == "completed"
        assert report.predictions_count == 1
        assert any("simulate mode" in warning for warning in report.warnings)

    def test_simulate_asset_without_state_uses_predictions(
        self, tmp_path: Path
    ) -> None:
        working_dir = tmp_path / ".asset"
        working_dir.mkdir()
        (working_dir / ".simulate").touch()
        _write_predictions(working_dir)

        report = validate_asset(working_dir)

        assert report.status == "completed"
        assert report.errors == []
        assert report.predictions_count == 1
        assert report.has_predictions is True
        assert any("simulate mode" in warning for warning in report.warnings)

    def test_simulate_asset_without_state_warns_for_empty_predictions(
        self, tmp_path: Path
    ) -> None:
        working_dir = tmp_path / ".asset"
        working_dir.mkdir()
        (working_dir / ".simulate").touch()
        _write_predictions(working_dir, lines=[""])

        report = validate_asset(working_dir)

        assert report.status == "completed"
        assert report.predictions_count == 0
        assert any("0 entries" in warning for warning in report.warnings)

    def test_reports_cannot_open_output_layer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        working_dir = tmp_path / ".asset"
        working_dir.mkdir()
        _write_state(working_dir, completed_steps=["predict", "apply"])
        _write_predictions(working_dir)
        output_path = working_dir / "output" / "output.usd"
        output_path.parent.mkdir()
        output_path.touch()

        monkeypatch.setattr(Sdf.Layer, "FindOrOpen", staticmethod(lambda _path: None))

        report = validate_asset(working_dir)

        assert "Cannot open output layer" in report.errors

    def test_reports_optimized_sublayers_and_stage_open_failures(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        working_dir = tmp_path / ".asset"
        working_dir.mkdir()
        _write_state(working_dir, completed_steps=["predict", "apply"])
        _write_predictions(working_dir)

        optimized_path = working_dir / "output" / "optimized_mesh.usda"
        _make_stage(optimized_path)
        output_path = working_dir / "output" / "output.usd"
        stage = _make_stage(output_path, ["/World/MeshA"])
        stage.GetRootLayer().subLayerPaths.append("./optimized_mesh.usda")
        stage.GetRootLayer().Save()

        original_open = Usd.Stage.Open

        def _raise_for_output(path: str) -> object:
            if path == str(output_path):
                raise RuntimeError("boom")
            return original_open(path)

        monkeypatch.setattr(Usd.Stage, "Open", staticmethod(_raise_for_output))

        report = validate_asset(working_dir)

        assert report.sublayers_optimized is True
        assert any("optimized USD" in error for error in report.errors)
        assert any(
            "Could not open output stage" in warning for warning in report.warnings
        )

    def test_runs_topology_check_when_neighbor_config_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        working_dir = tmp_path / ".asset"
        working_dir.mkdir()
        _write_state(working_dir, completed_steps=["predict", "apply"])
        _write_predictions(working_dir)

        output_path = working_dir / "output" / "output.usd"
        output_path.parent.mkdir()
        _make_stage(output_path, ["/World/MeshA"])

        config_path = tmp_path / "asset.yaml"
        config_path.write_text("input:\n  usd_path: input.usd\n")

        called: list[tuple[Path, Path, str]] = []

        def _record(config: Path, output: Path, report: AssetReport) -> None:
            called.append((config, output, report.name))

        monkeypatch.setattr(validate_mod, "_check_topology_and_instances", _record)

        validate_asset(working_dir)

        assert called == [(config_path, output_path, "asset")]


class TestCheckTopologyAndInstances:
    def test_records_mismatches_and_instance_regressions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        input_path = tmp_path / "input.usd"
        output_path = tmp_path / "output.usd"
        input_path.touch()
        output_path.touch()
        config_path = tmp_path / "asset.yaml"
        config_path.write_text(yaml.safe_dump({"input": {"usd_path": "input.usd"}}))
        report = AssetReport(name="asset")

        monkeypatch.setattr(
            validate_mod, "check_topology_match", lambda *_args: (False, 5, 3)
        )
        monkeypatch.setattr(
            validate_mod,
            "check_hierarchy_match",
            lambda *_args: (False, ["/World/OnlyIn"], ["/World/OnlyOut"]),
        )
        monkeypatch.setattr(
            validate_mod,
            "check_instances",
            lambda *_args: (4, 6, 3, 1, ["/World/InstA"]),
        )

        _check_topology_and_instances(config_path, output_path, report)

        assert report.input_meshes == 5
        assert report.topology_match is False
        assert report.hierarchy_match is False
        assert report.input_instances == 4
        assert report.output_instances == 6
        assert report.instances_kept == 3
        assert report.instances_deinstanced == 1
        assert any("Topology mismatch" in warning for warning in report.warnings)
        assert any("only in input" in warning for warning in report.warnings)
        assert any("only in output" in warning for warning in report.warnings)
        assert any("unexpected new instances" in warning for warning in report.warnings)
        assert any("instances lost" in error for error in report.errors)

    def test_records_exception_as_warning(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        input_path = tmp_path / "input.usd"
        output_path = tmp_path / "output.usd"
        input_path.touch()
        output_path.touch()
        config_path = tmp_path / "asset.yaml"
        config_path.write_text(yaml.safe_dump({"input": {"usd_path": "input.usd"}}))
        report = AssetReport(name="asset")

        monkeypatch.setattr(
            validate_mod,
            "check_topology_match",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("bad topology")),
        )

        _check_topology_and_instances(config_path, output_path, report)

        assert any("Could not check topology" in warning for warning in report.warnings)

    def test_missing_input_usd_skips_topology_checks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "output.usd"
        output_path.touch()
        config_path = tmp_path / "asset.yaml"
        config_path.write_text(
            yaml.safe_dump({"input": {"usd_path": "missing_input.usd"}})
        )
        report = AssetReport(name="asset")

        monkeypatch.setattr(
            validate_mod,
            "check_topology_match",
            lambda *_args: (_ for _ in ()).throw(AssertionError("not called")),
        )

        _check_topology_and_instances(config_path, output_path, report)

        assert report.warnings == []
        assert report.errors == []


class TestValidatePayloadGroup:
    def test_completed_payload_without_predictions_or_output_reports_problems(
        self, tmp_path: Path
    ) -> None:
        working_dir = tmp_path / ".payload"
        working_dir.mkdir()

        report = _validate_payload_group(
            {
                "group_name": "payload-a",
                "status": "completed",
                "working_dir": str(working_dir),
            },
            tmp_path,
        )

        assert report.name == "payload-a"
        assert "Status is completed but no predictions file found" in report.errors
        assert any("No output.usd found" in warning for warning in report.warnings)

    def test_completed_payload_reports_empty_predictions_and_bad_output(
        self, tmp_path: Path
    ) -> None:
        working_dir = tmp_path / ".payload"
        working_dir.mkdir()
        (working_dir / ".simulate").touch()
        _write_predictions(working_dir, lines=[""])

        output_path = tmp_path / "payload_output.usda"
        stage = _make_stage(output_path, ["/World/MeshA"])
        stage.GetPrimAtPath("/World/MeshA").SetInstanceable(False)
        stage.GetRootLayer().Save()

        report = _validate_payload_group(
            {
                "group_name": "payload-b",
                "status": "completed",
                "working_dir": str(working_dir),
                "output_usd_path": str(output_path),
                "instance_count": 2,
                "depth": 1,
            },
            tmp_path,
        )

        assert report.has_predictions is True
        assert report.predictions_count == 0
        assert report.has_output_usd is True
        assert report.instance_count == 2
        assert report.depth == 1
        assert any("simulate mode" in warning for warning in report.warnings)
        assert any("0 entries" in warning for warning in report.warnings)
        assert any("0 material bindings" in warning for warning in report.warnings)
        assert any("0 material definitions" in warning for warning in report.warnings)
        assert any("no sublayers" in warning for warning in report.warnings)
        assert any("de-instanced prims" in error for error in report.errors)


class TestValidateScene:
    def test_returns_error_when_manifest_is_missing(self, tmp_path: Path) -> None:
        scene_config = tmp_path / "scene.yaml"
        scene_config.write_text(yaml.safe_dump({"project": {"name": "demo"}}))

        report = validate_scene(scene_config)

        assert any("Manifest not found" in error for error in report.errors)

    def test_uses_session_id_for_manifest_directory(self, tmp_path: Path) -> None:
        scene_config = tmp_path / "scene.yaml"
        scene_config.write_text(
            yaml.safe_dump({"project": {"name": "demo", "session_id": "demo_run"}})
        )
        manifest_dir = tmp_path / ".demo_run_scene"
        manifest_dir.mkdir()
        (manifest_dir / "manifest.json").write_text(
            json.dumps({"sub_assets": [], "payload_groups": []})
        )

        report = validate_scene(scene_config)

        assert report.errors == []

    def test_rejects_non_mapping_scene_config(self, tmp_path: Path) -> None:
        scene_config = tmp_path / "scene.yaml"
        scene_config.write_text("- not\n- a\n- mapping\n")

        report = validate_scene(scene_config)

        assert any("YAML mapping" in error for error in report.errors)

    def test_non_mapping_project_uses_default_scene_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        scene_config = tmp_path / "scene.yaml"
        scene_config.write_text(yaml.safe_dump({"project": "not-a-mapping"}))
        captured: dict[str, Path] = {}

        def _fake_validate_scene_outputs(
            *,
            manifest_path: Path,
            working_dir: Path | None = None,
            composed_scene_path: Path | None = None,
            verbose: bool = False,
        ) -> validate_mod.SceneReport:
            captured["manifest_path"] = manifest_path
            captured["working_dir"] = working_dir or Path()
            assert composed_scene_path is None
            assert verbose is False
            return validate_mod.SceneReport()

        monkeypatch.setattr(
            validate_mod, "validate_scene_outputs", _fake_validate_scene_outputs
        )

        report = validate_scene(scene_config)

        assert report.errors == []
        assert captured["manifest_path"] == tmp_path / ".scene_scene" / "manifest.json"
        assert captured["working_dir"] == tmp_path / ".scene_scene"

    def test_validate_scene_outputs_resolves_relative_config_from_manifest_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        manifest_dir = tmp_path / "scene_work"
        manifest_dir.mkdir()
        config_path = manifest_dir / "configs" / "asset.yaml"
        config_path.parent.mkdir()
        config_path.write_text("project: {}\n")
        manifest_path = manifest_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "sub_assets": [
                        {
                            "id": "asset-a",
                            "name": "AssetA",
                            "config_path": "configs/asset.yaml",
                            "working_dir": "asset_work",
                            "status": "completed",
                        }
                    ],
                    "instance_groups": [],
                    "payload_groups": [],
                }
            )
        )
        captured: list[Path] = []

        def _fake_validate_asset(
            working_dir: Path, _verbose: bool = False
        ) -> AssetReport:
            captured.append(working_dir)
            assert _verbose is False
            return AssetReport(name="AssetA", status="completed")

        monkeypatch.setattr(validate_mod, "validate_asset", _fake_validate_asset)

        report = validate_scene_outputs(manifest_path=manifest_path)

        assert report.errors == []
        assert captured == [(manifest_dir / "configs" / "asset_work").resolve()]
        assert [asset.name for asset in report.assets] == ["AssetA"]

    def test_validate_scene_outputs_uses_manifest_working_dir_and_composed_path(
        self, tmp_path: Path
    ) -> None:
        working_dir = tmp_path / "scene_work"
        working_dir.mkdir()
        manifest_path = working_dir / "manifest.json"
        config_path = tmp_path / "configs" / "asset.yaml"
        config_path.parent.mkdir()
        config_path.write_text("project: {}\n")

        asset_work = tmp_path / "actual_asset_work"
        asset_work.mkdir()
        _write_state(asset_work, completed_steps=["predict"])
        _write_predictions(asset_work)

        composed_path = tmp_path / "output" / "scene_with_materials.usd"
        composed_path.parent.mkdir()
        _make_materialized_scene(
            composed_path,
            bound_meshes=["/Root/AssetA", "/Root/AssetB"],
        )

        manifest_path.write_text(
            json.dumps(
                {
                    "sub_assets": [
                        {
                            "id": "asset-a",
                            "name": "AssetA",
                            "prim_path": "/Root/AssetA",
                            "config_path": str(config_path),
                            "working_dir": str(asset_work),
                            "status": "completed",
                        }
                    ],
                    "instance_groups": [],
                    "payload_groups": [],
                }
            )
        )

        report = validate_scene_outputs(
            manifest_path=manifest_path,
            working_dir=working_dir,
            composed_scene_path=composed_path,
        )

        assert report.errors == []
        assert len(report.assets) == 1
        assert report.assets[0].status == "completed"
        assert report.assets[0].predictions_count == 1
        assert report.composed_scene_path == str(composed_path)
        assert report.composed_our == 2
        assert report.composed_none == 0

    def test_validate_scene_outputs_reports_missing_explicit_composed_scene(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps({"sub_assets": [], "instance_groups": [], "payload_groups": []})
        )

        report = validate_scene_outputs(
            manifest_path=manifest_path,
            composed_scene_path=tmp_path / "missing.usd",
        )

        assert any("Composed scene not found" in error for error in report.errors)

    def test_validate_scene_outputs_reports_unbound_composed_meshes(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps({"sub_assets": [], "instance_groups": [], "payload_groups": []})
        )
        composed_path = tmp_path / "composed.usd"
        _make_materialized_scene(
            composed_path,
            bound_meshes=["/Root/Bound"],
            unbound_meshes=["/Root/Unbound"],
        )

        report = validate_scene_outputs(
            manifest_path=manifest_path,
            composed_scene_path=composed_path,
        )

        assert report.composed_our == 1
        assert report.composed_none == 1
        assert any(
            "Mesh/GeomSubset prims missing our materials" in error
            for error in report.errors
        )

    def test_rejects_unsafe_session_id_before_manifest_lookup(
        self, tmp_path: Path
    ) -> None:
        scene_config = tmp_path / "scene.yaml"
        scene_config.write_text(
            yaml.safe_dump({"project": {"session_id": "safe/../../escape"}})
        )

        report = validate_scene(scene_config)

        assert any("Unsafe scene session_id/name" in error for error in report.errors)
        assert not any("Manifest not found" in error for error in report.errors)

    def test_validate_scene_honors_project_working_dir(self, tmp_path: Path) -> None:
        scene_config = tmp_path / "scene.yaml"
        scene_config.write_text(
            yaml.safe_dump(
                {"project": {"session_id": "../unsafe", "working_dir": "custom_work"}}
            )
        )
        working_dir = tmp_path / "custom_work"
        working_dir.mkdir()
        (working_dir / "manifest.json").write_text(
            json.dumps({"sub_assets": [], "instance_groups": [], "payload_groups": []})
        )

        report = validate_scene(scene_config)

        assert not any("Unsafe scene session_id/name" in e for e in report.errors)
        assert not any("Manifest not found" in e for e in report.errors)

    def test_validates_scene_assets_payloads_and_composed_scene(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        scene_config = tmp_path / "scene.yaml"
        scene_config.write_text(yaml.safe_dump({"project": {"name": "demo"}}))
        manifest_dir = tmp_path / ".demo_scene"
        manifest_dir.mkdir()

        rep_config = tmp_path / "rep_asset.yaml"
        rep_config.write_text("project: {}\n")
        failed_rep_config = tmp_path / "failed_rep.yaml"
        failed_rep_config.write_text("project: {}\n")
        fallback_config = tmp_path / "fallback_rep.yaml"
        fallback_config.write_text("project: {}\n")

        manifest = {
            "sub_assets": [
                {
                    "id": "rep1",
                    "name": "rep_asset",
                    "config_path": str(rep_config),
                    "status": "completed",
                    "prim_path": "/World/Rep",
                    "instance_group": "grp_rep",
                },
                {
                    "id": "rep_failed",
                    "name": "failed_rep",
                    "config_path": str(failed_rep_config),
                    "status": "failed",
                    "instance_group": "grp_failed",
                },
                {
                    "id": "missing_cfg",
                    "name": "missing_cfg",
                    "config_path": str(tmp_path / "missing.yaml"),
                    "status": "completed",
                },
                {
                    "id": "fallback_rep",
                    "name": "fallback_rep",
                    "config_path": str(fallback_config),
                    "status": "completed",
                    "instance_group": "grp_fallback",
                },
                {"id": "member1", "name": "member1", "instance_group": "grp_rep"},
                {
                    "id": "fallback_member",
                    "name": "fallback_member",
                    "instance_group": "grp_fallback",
                },
                {
                    "id": "orphan_ig",
                    "name": "orphan_ig",
                    "instance_group": "grp_missing",
                },
                {"id": "plain_no_config", "name": "plain_no_config"},
            ],
            "instance_groups": [
                {"group_name": "grp_no_rep", "instance_count": 2},
                {
                    "group_name": "grp_missing",
                    "representative_id": "missing-rep",
                    "instance_count": 3,
                },
                {
                    "group_name": "grp_failed",
                    "representative_id": "rep_failed",
                    "instance_count": 4,
                },
                {
                    "group_name": "grp_rep",
                    "representative_id": "rep1",
                    "instance_count": 5,
                },
            ],
            "payload_groups": [
                {
                    "group_name": "payload-group-1",
                    "status": "completed",
                    "instance_paths": ["/World/Rep"],
                }
            ],
        }
        (manifest_dir / "manifest.json").write_text(json.dumps(manifest))

        output_dir = manifest_dir / "output"
        output_dir.mkdir()
        composed_path = output_dir / "composed_scene.usd"
        _make_stage(composed_path, ["/World/Dummy"])

        def _fake_validate_asset(
            working_dir: Path, _verbose: bool = False
        ) -> AssetReport:
            if working_dir.name == ".rep_asset":
                return AssetReport(
                    name="rep_asset",
                    status="incomplete",
                    errors=["not done"],
                    warnings=[],
                    predictions_count=3,
                    has_predictions=True,
                    bindings_in_layer=2,
                    deinstanced=1,
                )
            if working_dir.name == ".failed_rep":
                return AssetReport(
                    name="failed_rep",
                    status="incomplete",
                    errors=["failed"],
                    bindings_in_layer=1,
                )
            return AssetReport(
                name="fallback_rep",
                status="completed",
                predictions_count=7,
                has_predictions=True,
                bindings_in_layer=5,
            )

        def _fake_validate_payload_group(
            payload_group: dict, _manifest_dir: Path, _verbose: bool = False
        ) -> PayloadReport:
            return PayloadReport(name=payload_group["group_name"], status="completed")

        monkeypatch.setattr(validate_mod, "validate_asset", _fake_validate_asset)
        monkeypatch.setattr(
            validate_mod, "_validate_payload_group", _fake_validate_payload_group
        )
        monkeypatch.setattr(
            validate_mod, "count_layer_bindings", lambda _layer: (0, 0, 0)
        )
        monkeypatch.setattr(
            validate_mod, "check_stage_bindings", lambda _stage: (1, 1, 1)
        )
        monkeypatch.setattr(
            Usd,
            "TraverseInstanceProxies",
            lambda: "instance-proxies",
        )
        monkeypatch.setattr(
            Usd.Stage,
            "Open",
            staticmethod(
                lambda _path: FakeStage(
                    [
                        FakePrim("Mesh", "/World/Looks/OurMesh"),
                        FakePrim("Mesh", "/World/Materials/OldMesh"),
                        FakePrim("Mesh"),
                        FakePrim("GeomSubset", "/World/Looks/OurSubset"),
                        FakePrim("GeomSubset", "/World/Materials/OldSubset"),
                        FakePrim("GeomSubset"),
                        FakePrim("Xform", "/World/Looks/Ignored"),
                        FakePrim("Mesh", "/World/Looks/NotProxy", instance_proxy=False),
                    ]
                )
            ),
        )

        report = validate_scene(scene_config)
        assets = {asset.name: asset for asset in report.assets}

        assert report.total_bindings == 8
        assert report.total_deinstanced == 1
        assert report.composed_scene_path == str(composed_path)
        assert report.composed_our == 1
        assert report.composed_old == 1
        assert report.composed_none == 1
        assert report.composed_instance_our == 1
        assert report.composed_instance_old == 1
        assert report.composed_instances_checked == 3
        assert report.composed_subset_our == 1
        assert report.composed_subset_old == 1
        assert report.composed_subsets_checked == 3
        assert assets["rep_asset"].status == "completed"
        assert assets["rep_asset"].errors == []
        assert any(
            "Processed via payload group 'payload-group-1'" in warning
            for warning in assets["rep_asset"].warnings
        )
        assert assets["failed_rep"].status == "failed"
        assert assets["member1"].status == "inherited"
        assert assets["member1"].predictions_count == 3
        assert assets["fallback_member"].status == "inherited"
        assert assets["fallback_member"].predictions_count == 7
        assert assets["orphan_ig"].status == "inherited"
        assert any(
            "representative not validated" in warning
            for warning in assets["orphan_ig"].warnings
        )
        assert assets["plain_no_config"].status == "no_config"
        assert assets["missing_cfg"].status == "no_config"
        assert [payload.name for payload in report.payloads] == ["payload-group-1"]
        assert any("grp_no_rep" in warning for warning in report.warnings)
        assert any(
            "Payload layers directory not found" in warning
            for warning in report.warnings
        )
        assert any(
            "Composed layer contains 0 material bindings" in warning
            for warning in report.warnings
        )
        assert any(
            "Composed layer contains 0 material definitions" in warning
            for warning in report.warnings
        )
        assert any("grp_missing" in error for error in report.errors)
        assert any("grp_failed" in error for error in report.errors)
        assert any(
            "instance proxies missing our materials" in error for error in report.errors
        )
        assert any(
            "GeomSubsets missing our materials" in error for error in report.errors
        )

    def test_warns_when_composed_scene_validation_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        scene_config = tmp_path / "scene.yaml"
        scene_config.write_text(yaml.safe_dump({"project": {"name": "demo"}}))
        manifest_dir = tmp_path / ".demo_scene"
        manifest_dir.mkdir()
        (manifest_dir / "manifest.json").write_text(
            json.dumps({"sub_assets": [], "instance_groups": [], "payload_groups": []})
        )
        output_dir = manifest_dir / "output"
        output_dir.mkdir()
        composed_path = output_dir / "composed_scene.usd"
        _make_stage(composed_path, ["/World/Dummy"])

        monkeypatch.setattr(
            validate_mod, "count_layer_bindings", lambda _layer: (1, 1, 0)
        )
        monkeypatch.setattr(
            Usd.Stage,
            "Open",
            staticmethod(
                lambda _path: (_ for _ in ()).throw(RuntimeError("bad scene"))
            ),
        )

        report = validate_scene(scene_config)

        assert any(
            "Could not validate composed scene" in warning
            for warning in report.warnings
        )


class TestFormatAssetReportAdditional:
    def test_verbose_success_report_includes_warnings(self) -> None:
        report = AssetReport(
            name="asset",
            status="completed",
            warnings=["predictions were simulated"],
        )

        lines = format_asset_report(report, verbose=True)

        assert any("WARN:  predictions were simulated" in line for line in lines)
