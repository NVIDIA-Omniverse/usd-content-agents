# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extra focused coverage for material_agent.scene.run helper functions."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pxr import Sdf, Usd, UsdGeom, UsdShade
from world_understanding.utils.credentials import InlineSecretError

from material_agent.scene.manifest import (
    InstanceGroup,
    PayloadGroup,
    SceneManifest,
    SubAsset,
)
from material_agent.scene.run import (
    _clean_working_dir_for_so_retry,
    _clear_pipeline_state_from_step,
    _copy_results_to_duplicates,
    _create_modified_parent_copy,
    _fix_output_material_scope,
    _fix_representative_sublayer,
    _generate_simulate_predictions,
    _run_parallel,
    _run_payload_worker,
    _run_payloads_parallel,
    _run_payloads_sequential,
    _run_sequential,
    _run_simulate,
    _run_sub_asset_worker,
    _set_payload_output_usd,
    _sub_asset_working_dir,
    _update_output_paths,
    _update_payload_output_paths,
    _write_sub_asset_harness_error,
    run_all,
    run_all_payloads_bottomup,
    run_payload,
    run_sub_asset,
)


@dataclass
class FakePipelineOutput:
    success: bool
    error: str | None = None
    step_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    raw_result: dict[str, Any] | None = None


def _make_sub_asset(
    name: str = "asset_a",
    *,
    status: str = "pending",
    config_path: str | None = None,
    working_dir: str | None = None,
    instance_group: str | None = None,
) -> SubAsset:
    return SubAsset(
        id=str(uuid.uuid4()),
        name=name,
        prim_path=f"/World/{name}",
        status=status,
        config_path=config_path,
        working_dir=working_dir,
        instance_group=instance_group,
    )


def _make_payload_group(
    group_name: str = "payload_a",
    *,
    depth: int = 0,
    status: str = "pending",
    config_path: str | None = None,
    working_dir: str | None = None,
    representative_path: str | None = None,
) -> PayloadGroup:
    return PayloadGroup(
        id=str(uuid.uuid4()),
        group_name=group_name,
        payload_file=f"/tmp/{group_name}.usd",
        depth=depth,
        status=status,
        config_path=config_path,
        working_dir=working_dir,
        representative_path=representative_path,
    )


def _write_config(
    path: Path,
    *,
    session_id: str | None = "test_session",
    extra: dict[str, Any] | None = None,
) -> None:
    cfg: dict[str, Any] = {"project": {}, "steps": {}}
    if session_id is not None:
        cfg["project"]["session_id"] = session_id
    if extra:
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _create_empty_layer(path: Path, *, sublayers: list[str] | None = None) -> Path:
    layer = Sdf.Layer.CreateNew(str(path))
    if sublayers is not None:
        layer.subLayerPaths = sublayers
    layer.Save()
    return path


def test_clean_working_dir_for_so_retry_removes_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, session_id="asset-1")
    working_dir = tmp_path / ".asset-1"

    for dirname in [
        "optimized",
        "dataset",
        "predictions",
        "restored",
        ".pipeline_temp",
    ]:
        _touch(working_dir / dirname / "artifact.txt")
    _touch(working_dir / ".pipeline_state.json", "{}")

    _clean_working_dir_for_so_retry(config_path)

    for dirname in [
        "optimized",
        "dataset",
        "predictions",
        "restored",
        ".pipeline_temp",
    ]:
        assert not (working_dir / dirname).exists()
    assert not (working_dir / ".pipeline_state.json").exists()


def test_clear_pipeline_state_from_step_removes_downstream_steps(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, session_id="asset-2")
    state_file = tmp_path / ".asset-2" / ".pipeline_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "completed_steps": [
            "validate_input",
            "optimize_usd",
            "build_dataset_usd",
            "predict",
            "apply",
        ],
        "failed_steps": ["apply"],
        "step_outputs": {
            "optimize_usd": {"optimized_usd_path": "optimized.usd"},
            "build_dataset_usd": {"output_dir": "dataset"},
            "predict": {"predictions_path": "predictions.jsonl"},
            "apply": {"output_usd_path": "output.usd"},
        },
    }
    state_file.write_text(json.dumps(state), encoding="utf-8")

    _clear_pipeline_state_from_step(config_path, "predict")

    updated = json.loads(state_file.read_text(encoding="utf-8"))
    assert updated["completed_steps"] == [
        "validate_input",
        "optimize_usd",
        "build_dataset_usd",
    ]
    assert updated["failed_steps"] == []
    assert "predict" not in updated["step_outputs"]
    assert "apply" not in updated["step_outputs"]
    assert "optimize_usd" in updated["step_outputs"]
    assert stat.S_IMODE(state_file.stat().st_mode) == 0o600


def test_clear_pipeline_state_from_step_persists_step_error_cleanup(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, session_id="asset-errors")
    state_file = tmp_path / ".asset-errors" / ".pipeline_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "completed_steps": ["validate_input", "optimize_usd"],
        "failed_steps": [],
        "step_outputs": {"optimize_usd": {"optimized_usd_path": "optimized.usd"}},
        "step_errors": {
            "optimize_usd": "keep this older error",
            "predict": "stale prediction failure",
            "apply": "stale apply failure",
        },
    }
    state_file.write_text(json.dumps(state), encoding="utf-8")

    _clear_pipeline_state_from_step(config_path, "predict")

    updated = json.loads(state_file.read_text(encoding="utf-8"))
    assert updated["completed_steps"] == ["validate_input", "optimize_usd"]
    assert updated["step_outputs"] == {
        "optimize_usd": {"optimized_usd_path": "optimized.usd"}
    }
    assert updated["step_errors"] == {"optimize_usd": "keep this older error"}


def test_clear_pipeline_state_rejects_inline_secrets_before_rewrite(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, session_id="asset-secret-state")
    state_file = tmp_path / ".asset-secret-state" / ".pipeline_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    secret = "never-rewrite-checkpoint-secret-727"
    state = {
        "completed_steps": ["validate_input", "predict"],
        "failed_steps": [],
        "step_outputs": {"predict": {"prediction": "stale"}},
        "api_key": secret,
    }
    original = json.dumps(state)
    state_file.write_text(original, encoding="utf-8")

    with pytest.raises(InlineSecretError) as exc_info:
        _clear_pipeline_state_from_step(config_path, "predict")

    assert state_file.read_text(encoding="utf-8") == original
    assert secret not in str(exc_info.value)


def test_copy_results_to_duplicates_copies_files_and_status(tmp_path: Path) -> None:
    rep_work = tmp_path / "rep"
    member_work = tmp_path / "member"
    predictions = _touch(rep_work / "predictions" / "predictions.jsonl", "predictions")
    material_layer = _touch(rep_work / "output" / "output.usd", "output")

    representative = _make_sub_asset(
        "rep",
        status="completed",
        working_dir=str(rep_work),
        instance_group="dup_group",
    )
    representative.predictions_path = str(predictions)
    representative.material_layer_path = str(material_layer)

    member = _make_sub_asset(
        "member",
        working_dir=str(member_work),
        instance_group="dup_group",
    )

    manifest = SceneManifest(
        sub_assets=[representative, member],
        instance_groups=[
            InstanceGroup(
                group_name="dup_group",
                representative_id=representative.id,
                member_paths=[representative.prim_path, member.prim_path],
            )
        ],
    )

    _copy_results_to_duplicates(manifest, [member])

    copied_predictions = member_work / "predictions" / predictions.name
    copied_output = member_work / "output" / material_layer.name
    assert copied_predictions.exists()
    assert copied_output.exists()
    assert member.predictions_path == str(copied_predictions)
    assert member.material_layer_path == str(copied_output)
    assert member.status == "completed"


def test_run_sequential_counts_completed_and_failures(tmp_path: Path) -> None:
    assets = [_make_sub_asset("a"), _make_sub_asset("b"), _make_sub_asset("c")]
    manifest = SceneManifest(sub_assets=assets)
    manifest.save = MagicMock()  # type: ignore[method-assign]

    def fake_run_sub_asset(sa, *args, **kwargs):
        if sa.name == "a":
            sa.status = "completed"
            return sa
        if sa.name == "b":
            sa.status = "failed"
            return sa
        raise RuntimeError("boom")

    with patch(
        "material_agent.scene.run.run_sub_asset", side_effect=fake_run_sub_asset
    ):
        completed, failed = _run_sequential(
            assets,
            manifest,
            tmp_path / "manifest.json",
            skip_steps=None,
            only_steps=None,
            verbose=False,
        )

    assert completed == 1
    assert failed == 2
    assert assets[2].status == "failed"
    assert manifest.save.call_count == 3


def test_run_sequential_cancel_checker_stops_between_assets(tmp_path: Path) -> None:
    assets = [_make_sub_asset("a"), _make_sub_asset("b")]
    manifest = SceneManifest(sub_assets=assets)
    manifest.save = MagicMock()  # type: ignore[method-assign]
    processed: list[str] = []

    def fake_run_sub_asset(sa, *args, **kwargs):
        processed.append(sa.name)
        sa.status = "completed"
        return sa

    with patch(
        "material_agent.scene.run.run_sub_asset", side_effect=fake_run_sub_asset
    ):
        with pytest.raises(asyncio.CancelledError):
            _run_sequential(
                assets,
                manifest,
                tmp_path / "manifest.json",
                skip_steps=None,
                only_steps=None,
                verbose=False,
                cancel_checker=lambda: bool(processed),
            )

    assert processed == ["a"]
    assert manifest.save.call_count == 1


def test_generate_simulate_predictions_prefers_optimized_usd(tmp_path: Path) -> None:
    config = {"input": {"usd_path": "scene.usd", "prim_path": "/Root"}}
    config_path = tmp_path / "config.yaml"
    working_dir = tmp_path / ".session"
    optimized = _touch(working_dir / "optimized" / "optimized_input.usd")

    with patch(
        "material_agent.scene.simulate.generate_mock_predictions", return_value=7
    ) as mock_generate:
        result = _generate_simulate_predictions(
            config,
            config_path,
            working_dir,
            ["Steel", "Plastic"],
        )

    assert result == 7
    kwargs = mock_generate.call_args.kwargs
    assert kwargs["usd_path"] == optimized
    assert kwargs["material_names"] == ["Steel", "Plastic"]
    assert kwargs["output_path"] == working_dir / "predictions" / "predictions.jsonl"
    assert kwargs["prim_path_scope"] == "/Root"


def test_update_output_paths_prefers_restored_predictions(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, session_id="asset-3")
    working_dir = tmp_path / ".asset-3"
    restored = _touch(working_dir / "restored" / "restored_predictions.jsonl")
    _touch(working_dir / "predictions" / "predictions.jsonl")
    output = _touch(working_dir / "output" / "output.usd")
    sub_asset = _make_sub_asset("Widget")

    _update_output_paths(sub_asset, config_path)

    assert sub_asset.working_dir == str(working_dir)
    assert sub_asset.predictions_path == str(restored)
    assert sub_asset.material_layer_path == str(output)


def test_update_output_paths_falls_back_to_safe_name_without_session(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, session_id=None)
    working_dir = tmp_path / ".my_asset"
    raw_predictions = _touch(working_dir / "predictions" / "predictions.jsonl")
    sub_asset = _make_sub_asset("My Asset")

    _update_output_paths(sub_asset, config_path)

    assert sub_asset.working_dir == str(working_dir)
    assert sub_asset.predictions_path == str(raw_predictions)


def test_run_simulate_short_circuits_when_no_predictions(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        session_id="simulate-1",
        extra={
            "input": {"usd_path": "scene.usd"},
            "steps": {"optimize_usd": {"enabled": True}},
        },
    )

    with (
        patch(
            "material_agent.api.pipeline.run_pipeline",
            return_value=FakePipelineOutput(
                success=True, completed_steps=["optimize_usd"]
            ),
        ) as mock_run,
        patch(
            "material_agent.scene.run._generate_simulate_predictions", return_value=0
        ),
    ):
        result = _run_simulate(config_path, ["Steel"], verbose=False)

    assert result.success is True
    assert result.completed_steps == ["optimize_usd"]
    assert mock_run.call_count == 1
    first_input = mock_run.call_args_list[0].args[0]
    assert first_input.only_steps == ["optimize_usd"]
    assert isinstance(first_input.config, dict)
    assert first_input.config_path == config_path
    marker = tmp_path / ".simulate-1" / ".simulate"
    assert marker.exists()


def test_run_simulate_skips_apply_when_apply_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        session_id="simulate-2",
        extra={
            "input": {"usd_path": "scene.usd"},
            "steps": {"apply": {"enabled": False}},
        },
    )

    with (
        patch("material_agent.api.pipeline.run_pipeline") as mock_run,
        patch(
            "material_agent.scene.run._generate_simulate_predictions", return_value=3
        ),
    ):
        result = _run_simulate(config_path, ["Steel"], verbose=False)

    assert result.success is True
    assert result.completed_steps == ["predict"]
    assert result.step_results["predict"]["predictions_count"] == 3
    mock_run.assert_not_called()
    marker = tmp_path / ".simulate-2" / ".simulate"
    assert marker.exists()


def test_run_sub_asset_simulate_uses_fast_prediction_path(tmp_path: Path) -> None:
    config_path = tmp_path / "asset.yaml"
    _write_config(config_path, session_id="asset-sim")
    predictions = _touch(
        tmp_path / ".asset-sim" / "predictions" / "predictions.jsonl",
        text='{"prim_path": "/World/asset_1", "material": "Steel"}\n',
    )

    sub_asset = _make_sub_asset("asset_1", config_path=str(config_path))
    with (
        patch(
            "material_agent.scene.run._run_simulate",
            return_value=FakePipelineOutput(success=True, completed_steps=["predict"]),
        ) as mock_simulate,
        patch("material_agent.api.pipeline.run_pipeline") as mock_run,
    ):
        result = run_sub_asset(
            sub_asset,
            simulate=True,
            material_names=["Steel"],
        )

    assert result.status == "completed"
    assert result.predictions_path == str(predictions)
    expected_simulation_config = yaml.safe_load(config_path.read_text())
    expected_simulation_config["scene"] = {}
    mock_simulate.assert_called_once_with(
        config_path,
        ["Steel"],
        verbose=False,
        cancel_checker=None,
        config_dict=expected_simulation_config,
    )
    mock_run.assert_not_called()


def test_run_sub_asset_forwards_cancel_checker(tmp_path: Path) -> None:
    config_path = tmp_path / "asset.yaml"
    _write_config(config_path, session_id="asset-1")
    seen_inputs: list[Any] = []

    def checker() -> bool:
        return False

    def fake_run_pipeline(params):
        seen_inputs.append(params)
        return FakePipelineOutput(success=True, completed_steps=["predict"])

    sub_asset = _make_sub_asset("asset_1", config_path=str(config_path))
    with patch(
        "material_agent.api.pipeline.run_pipeline", side_effect=fake_run_pipeline
    ):
        result = run_sub_asset(sub_asset, cancel_checker=checker)

    assert result.status == "completed"
    assert seen_inputs[0].cancel_checker is checker


def test_run_payload_forwards_cancel_checker(tmp_path: Path) -> None:
    config_path = tmp_path / "payload.yaml"
    _write_config(config_path, session_id="payload-1")
    seen_inputs: list[Any] = []

    def checker() -> bool:
        return False

    def fake_run_pipeline(params):
        seen_inputs.append(params)
        return FakePipelineOutput(success=True, completed_steps=["predict"])

    payload = _make_payload_group("payload_1", config_path=str(config_path))
    with patch(
        "material_agent.api.pipeline.run_pipeline", side_effect=fake_run_pipeline
    ):
        result = run_payload(payload, cancel_checker=checker)

    assert result.status == "completed"
    assert seen_inputs[0].cancel_checker is checker


def test_fix_output_material_scope_moves_materials_under_default_prim(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(output_path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Geom/Mesh").GetPrim()
    material = UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")
    UsdShade.MaterialBindingAPI(mesh).Bind(material)
    stage.GetRootLayer().Save()

    payload_group = _make_payload_group("scope_fix")
    payload_group.output_usd_path = str(output_path)

    _fix_output_material_scope(payload_group)

    layer = Sdf.Layer.FindOrOpen(str(output_path))
    assert layer is not None
    assert layer.GetPrimAtPath("/Asset/Looks").typeName == "Scope"
    assert layer.GetPrimAtPath("/Asset/Looks/TestMaterial") is not None
    mesh_spec = layer.GetPrimAtPath("/Asset/Geom/Mesh")
    assert mesh_spec is not None
    targets = mesh_spec.relationships["material:binding"].targetPathList.explicitItems
    assert targets == [Sdf.Path("/Asset/Looks/TestMaterial")]
    assert layer.GetPrimAtPath("/World/Looks/TestMaterial") is None


def test_fix_output_material_scope_guard_and_parent_creation_paths(
    tmp_path: Path,
) -> None:
    payload_group = _make_payload_group("scope_guards")
    _fix_output_material_scope(payload_group)

    payload_group.output_usd_path = str(tmp_path / "missing.usda")
    _fix_output_material_scope(payload_group)

    no_default = _create_empty_layer(tmp_path / "no_default.usda")
    payload_group.output_usd_path = str(no_default)
    _fix_output_material_scope(payload_group)

    world_path = tmp_path / "world_default.usda"
    world_stage = Usd.Stage.CreateNew(str(world_path))
    world = UsdGeom.Xform.Define(world_stage, "/World")
    world_stage.SetDefaultPrim(world.GetPrim())
    world_stage.GetRootLayer().Save()
    payload_group.output_usd_path = str(world_path)
    _fix_output_material_scope(payload_group)

    no_materials = tmp_path / "asset_no_materials.usda"
    no_mat_stage = Usd.Stage.CreateNew(str(no_materials))
    asset = UsdGeom.Xform.Define(no_mat_stage, "/Asset")
    no_mat_stage.SetDefaultPrim(asset.GetPrim())
    no_mat_stage.GetRootLayer().Save()
    payload_group.output_usd_path = str(no_materials)
    _fix_output_material_scope(payload_group)

    manual = tmp_path / "manual_scope.usda"
    layer = Sdf.Layer.CreateNew(str(manual))
    layer.defaultPrim = "MissingRoot"
    Sdf.CreatePrimInLayer(layer, Sdf.Path("/World"))
    Sdf.CreatePrimInLayer(layer, Sdf.Path("/World/Looks"))
    material_spec = Sdf.CreatePrimInLayer(layer, Sdf.Path("/World/Looks/Mat"))
    material_spec.typeName = "Material"
    mesh_spec = Sdf.CreatePrimInLayer(layer, Sdf.Path("/Geometry/Mesh"))
    Sdf.RelationshipSpec(mesh_spec, "not:material:relationship")
    outside_rel = Sdf.RelationshipSpec(mesh_spec, "material:binding:outside")
    outside_rel.targetPathList.explicitItems = [Sdf.Path("/Other/Looks/Mat")]
    layer.Save()

    payload_group.output_usd_path = str(manual)
    _fix_output_material_scope(payload_group)

    updated = Sdf.Layer.FindOrOpen(str(manual))
    assert updated.GetPrimAtPath("/MissingRoot/Looks/Mat") is not None
    updated_mesh = updated.GetPrimAtPath("/Geometry/Mesh")
    assert updated_mesh.relationships[
        "material:binding:outside"
    ].targetPathList.explicitItems == [Sdf.Path("/Other/Looks/Mat")]


def test_fix_representative_sublayer_swaps_to_original_payload(tmp_path: Path) -> None:
    original = _create_empty_layer(tmp_path / "original_payload.usda")
    representative = _create_empty_layer(tmp_path / "representative.usda")
    output_path = tmp_path / "out" / "output.usda"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateNew(str(output_path))
    layer.subLayerPaths = [os.path.relpath(representative, output_path.parent)]
    layer.Save()

    payload_group = _make_payload_group("payload")
    payload_group.payload_file = str(original)
    payload_group.representative_path = str(representative)
    payload_group.output_usd_path = str(output_path)

    _fix_representative_sublayer(payload_group)

    updated = Sdf.Layer.FindOrOpen(str(output_path))
    assert updated is not None
    assert updated.subLayerPaths == [os.path.relpath(original, output_path.parent)]


def test_fix_representative_sublayer_guard_and_noop_paths(tmp_path: Path) -> None:
    payload_group = _make_payload_group("rep_guard")
    _fix_representative_sublayer(payload_group)

    payload_group.output_usd_path = str(tmp_path / "missing_output.usda")
    payload_group.representative_path = str(tmp_path / "rep.usda")
    _fix_representative_sublayer(payload_group)

    output_path = tmp_path / "output.usda"
    layer = Sdf.Layer.CreateNew(str(output_path))
    layer.subLayerPaths = ["other.usda"]
    layer.Save()
    payload_group.output_usd_path = str(output_path)
    payload_group.payload_file = str(tmp_path / "original.usda")
    payload_group.representative_path = str(tmp_path / "representative.usda")
    _fix_representative_sublayer(payload_group)

    with patch("pxr.Sdf.Layer.FindOrOpen", return_value=None):
        _fix_representative_sublayer(payload_group)

    representative = _create_empty_layer(tmp_path / "representative.usda")
    original = _create_empty_layer(tmp_path / "original.usda")
    output_with_rep = tmp_path / "out_relpath" / "output.usda"
    output_with_rep.parent.mkdir()
    layer = Sdf.Layer.CreateNew(str(output_with_rep))
    layer.subLayerPaths = [os.path.relpath(representative, output_with_rep.parent)]
    layer.Save()
    with patch("material_agent.scene.run.os.path.relpath", side_effect=ValueError):
        payload_group.output_usd_path = str(output_with_rep)
        payload_group.payload_file = str(original)
        payload_group.representative_path = str(representative)
        _fix_representative_sublayer(payload_group)

    updated = Sdf.Layer.FindOrOpen(str(output_with_rep))
    assert updated.subLayerPaths == [str(original.resolve())]


def test_update_payload_output_paths_uses_group_name_when_session_missing(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "payload.yaml"
    _write_config(config_path, session_id=None)
    predictions = _touch(
        tmp_path / ".payload_a" / "predictions" / "predictions.jsonl", "predictions"
    )
    payload = _make_payload_group("payload_a")

    _update_payload_output_paths(payload, config_path)

    assert payload.working_dir == str(tmp_path / ".payload_a")
    assert payload.predictions_path == str(predictions)


def test_create_modified_parent_copy_rewrites_sublayers(tmp_path: Path) -> None:
    child_original = _create_empty_layer(tmp_path / "child.usda")
    child_output = _create_empty_layer(tmp_path / "child_output.usda")
    sublayer_original = _create_empty_layer(tmp_path / "parent_sub.usda")
    parent_original = _create_empty_layer(
        tmp_path / "parent.usda",
        sublayers=[os.path.relpath(sublayer_original, tmp_path)],
    )

    child_group = _make_payload_group("child", status="completed")
    child_group.payload_file = str(child_original)
    child_group.output_usd_path = str(child_output)

    parent_group = _make_payload_group("parent")
    parent_group.payload_file = str(parent_original)
    parent_group.child_payload_files = [str(child_original)]

    manifest = SceneManifest(payload_groups=[child_group, parent_group])

    def fake_rewrite_arcs_in_layer(layer, child_map, resolve_from):
        if Path(resolve_from) == parent_original:
            return 1
        if Path(resolve_from) == sublayer_original:
            return 1
        return 0

    with patch(
        "material_agent.scene.payload_dag_utils.rewrite_arcs_in_layer",
        side_effect=fake_rewrite_arcs_in_layer,
    ):
        _create_modified_parent_copy(parent_group, manifest, tmp_path / "work")

    modified = Path(parent_group.modified_input_path)
    copied_sublayer = modified.parent / sublayer_original.name
    assert modified.exists()
    assert copied_sublayer.exists()
    modified_layer = Sdf.Layer.FindOrOpen(str(modified))
    assert modified_layer is not None
    assert modified_layer.subLayerPaths == [str(copied_sublayer)]


def test_create_modified_parent_copy_child_map_edge_paths(tmp_path: Path) -> None:
    skipped_child = _make_payload_group("skipped", status="skipped")
    skipped_child_file = _create_empty_layer(tmp_path / "skipped.usda")
    skipped_child.payload_file = str(skipped_child_file)
    missing_child_file = _create_empty_layer(tmp_path / "missing_child.usda")
    parent_original = _create_empty_layer(tmp_path / "parent_no_map.usda")
    parent_group = _make_payload_group("parent_no_map")
    parent_group.payload_file = str(parent_original)
    parent_group.child_payload_files = [
        str(skipped_child_file),
        str(missing_child_file),
    ]
    manifest = SceneManifest(payload_groups=[skipped_child, parent_group])

    _create_modified_parent_copy(parent_group, manifest, tmp_path / "work")

    assert Path(parent_group.modified_input_path).exists()

    modified_child = _make_payload_group("modified_child")
    modified_child_file = _create_empty_layer(tmp_path / "modified_child.usda")
    modified_target = _create_empty_layer(tmp_path / "modified_target.usda")
    modified_child.payload_file = str(modified_child_file)
    modified_child.modified_input_path = str(modified_target)
    parent_with_missing_sublayer = _create_empty_layer(
        tmp_path / "parent_missing_sublayer.usda",
        sublayers=["missing_sublayer.usda"],
    )
    parent_group = _make_payload_group("parent_missing_sublayer")
    parent_group.payload_file = str(parent_with_missing_sublayer)
    parent_group.child_payload_files = [str(modified_child_file)]
    manifest = SceneManifest(payload_groups=[modified_child, parent_group])

    with patch(
        "material_agent.scene.payload_dag_utils.rewrite_arcs_in_layer",
        return_value=1,
    ):
        _create_modified_parent_copy(parent_group, manifest, tmp_path / "work2")

    assert Path(parent_group.modified_input_path).exists()


def test_run_payloads_sequential_counts_and_saves(tmp_path: Path) -> None:
    payloads = [
        _make_payload_group("a"),
        _make_payload_group("b"),
        _make_payload_group("c"),
    ]
    manifest = SceneManifest(payload_groups=payloads)
    manifest.save = MagicMock()  # type: ignore[method-assign]

    def fake_run_payload(pg, *args, **kwargs):
        if pg.group_name == "a":
            pg.status = "completed"
            return pg
        if pg.group_name == "b":
            pg.status = "failed"
            return pg
        raise RuntimeError("payload boom")

    with patch("material_agent.scene.run.run_payload", side_effect=fake_run_payload):
        completed, failed = _run_payloads_sequential(
            payloads,
            manifest,
            tmp_path / "manifest.json",
            skip_steps=None,
            only_steps=None,
            verbose=False,
        )

    assert completed == 1
    assert failed == 2
    assert payloads[2].status == "failed"
    assert manifest.save.call_count == 3


def test_run_payloads_sequential_cancel_checker_stops_between_payloads(
    tmp_path: Path,
) -> None:
    payloads = [_make_payload_group("a"), _make_payload_group("b")]
    manifest = SceneManifest(payload_groups=payloads)
    manifest.save = MagicMock()  # type: ignore[method-assign]
    processed: list[str] = []

    def fake_run_payload(pg, *args, **kwargs):
        processed.append(pg.group_name)
        pg.status = "completed"
        return pg

    with patch("material_agent.scene.run.run_payload", side_effect=fake_run_payload):
        with pytest.raises(asyncio.CancelledError):
            _run_payloads_sequential(
                payloads,
                manifest,
                tmp_path / "manifest.json",
                skip_steps=None,
                only_steps=None,
                verbose=False,
                cancel_checker=lambda: bool(processed),
            )

    assert processed == ["a"]
    assert manifest.save.call_count == 1


def test_run_payloads_parallel_updates_manifest(tmp_path: Path) -> None:
    payload_a = _make_payload_group("a")
    payload_b = _make_payload_group("b")
    manifest = SceneManifest(payload_groups=[payload_a, payload_b])
    manifest.save = MagicMock()  # type: ignore[method-assign]

    def fake_worker(pg, *args, **kwargs):
        pg.status = "completed" if pg.group_name == "a" else "failed"
        return pg

    with patch("material_agent.scene.run._run_payload_worker", side_effect=fake_worker):
        completed, failed = _run_payloads_parallel(
            [payload_a, payload_b],
            manifest,
            tmp_path / "manifest.json",
            skip_steps=None,
            only_steps=None,
            verbose=False,
            max_workers=2,
        )

    assert completed == 1
    assert failed == 1
    assert manifest.payload_groups[0].status in {"completed", "failed"}
    assert manifest.payload_groups[1].status in {"completed", "failed"}
    assert manifest.save.call_count == 2


def test_run_payloads_parallel_cancellation_and_worker_error_branches(
    tmp_path: Path,
) -> None:
    payload = _make_payload_group("cancel")
    manifest = SceneManifest(payload_groups=[payload])
    manifest.save = MagicMock()  # type: ignore[method-assign]
    cancel_calls = {"count": 0}

    def cancel_after_submit() -> bool:
        cancel_calls["count"] += 1
        return cancel_calls["count"] >= 2

    with patch("material_agent.scene.run._run_payload_worker", return_value=payload):
        with pytest.raises(asyncio.CancelledError):
            _run_payloads_parallel(
                [payload],
                manifest,
                tmp_path / "manifest.json",
                skip_steps=None,
                only_steps=None,
                verbose=False,
                max_workers=1,
                cancel_checker=cancel_after_submit,
            )

    def raise_cancel(pg, *args, **kwargs):
        raise asyncio.CancelledError("worker cancelled")

    with patch(
        "material_agent.scene.run._run_payload_worker", side_effect=raise_cancel
    ):
        with pytest.raises(asyncio.CancelledError):
            _run_payloads_parallel(
                [payload],
                manifest,
                tmp_path / "manifest.json",
                skip_steps=None,
                only_steps=None,
                verbose=False,
                max_workers=1,
                cancel_checker=lambda: False,
            )

    with patch(
        "material_agent.scene.run._run_payload_worker",
        side_effect=RuntimeError("worker failed"),
    ):
        completed, failed = _run_payloads_parallel(
            [payload],
            manifest,
            tmp_path / "manifest.json",
            skip_steps=None,
            only_steps=None,
            verbose=False,
            max_workers=1,
        )

    assert (completed, failed) == (0, 1)
    assert manifest.payload_groups[0].status == "failed"


def test_run_all_payloads_bottomup_forwards_source_config_at_each_depth(
    tmp_path: Path,
) -> None:
    configs_dir = tmp_path / "configs"
    leaf = _make_payload_group(
        "leaf",
        depth=0,
        config_path=str(configs_dir / "payloads" / "leaf.yaml"),
        working_dir=str(configs_dir / "payloads" / ".leaf"),
    )
    parent = _make_payload_group(
        "parent",
        depth=1,
        representative_path=str(tmp_path / "representative.usda"),
    )
    manifest = SceneManifest(payload_groups=[leaf, parent])
    manifest.save = MagicMock()  # type: ignore[method-assign]
    scene_config = {
        "project": {"name": "scene"},
        "steps": {"predict": {"vlm": {"api_key": "q7Z9"}}},
    }
    forwarded_scene_configs: list[dict[str, object]] = []

    def fake_run_payloads_sequential(
        payloads, manifest, manifest_path, *args, **kwargs
    ):
        forwarded_scene_configs.append(kwargs["scene_config"])
        for pg in payloads:
            if not pg.working_dir:
                continue
            output = _touch(Path(pg.working_dir) / "output" / "output.usd")
            pg.output_usd_path = str(output)
            pg.status = "completed"
        return len(payloads), 0

    with (
        patch(
            "material_agent.scene.run._run_payloads_sequential",
            side_effect=fake_run_payloads_sequential,
        ) as mock_run,
        patch(
            "material_agent.scene.run._create_modified_parent_copy"
        ) as mock_create_parent,
        patch("material_agent.scene.run._fix_output_material_scope") as mock_fix_scope,
        patch("material_agent.scene.run._fix_representative_sublayer") as mock_fix_rep,
        patch(
            "material_agent.scene.config_gen.generate_payload_config"
        ) as mock_generate_config,
    ):
        result = run_all_payloads_bottomup(
            manifest,
            tmp_path / "manifest.json",
            scene_config=scene_config,
            configs_dir=configs_dir,
            max_workers=1,
        )

    assert result is manifest
    assert mock_run.call_count == 2
    mock_create_parent.assert_called_once()
    mock_generate_config.assert_called_once()
    assert forwarded_scene_configs == [scene_config, scene_config]
    assert parent.config_path == str(configs_dir / "payloads" / "parent.yaml")
    assert parent.working_dir == str(configs_dir / "payloads" / ".parent")
    assert leaf.output_usd_path is not None
    assert parent.output_usd_path is not None
    assert mock_fix_scope.call_count == 2
    mock_fix_rep.assert_called_once_with(parent)


def test_sub_asset_working_dir_variants_and_harness_error_fallback(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "Asset Config.yaml"
    assert (
        _sub_asset_working_dir({"project": "bad"}, config_path)
        == (tmp_path / ".asset_config").resolve()
    )
    assert (
        _sub_asset_working_dir(
            {"project": {"working_dir": "relative_work"}},
            config_path,
        )
        == (tmp_path / "relative_work").resolve()
    )

    no_config_asset = _make_sub_asset("no_config")
    _write_sub_asset_harness_error(no_config_asset, RuntimeError("ignored"))
    assert no_config_asset.working_dir is None

    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("- not\n- a mapping\n", encoding="utf-8")
    asset = _make_sub_asset("bad_config", config_path=str(bad_config))
    sentinel = "harness-error-secret-713"
    _write_sub_asset_harness_error(asset, RuntimeError(sentinel))

    error_path = tmp_path / ".bad" / "harness_scene_adapter_error.json"
    assert asset.working_dir == str(tmp_path / ".bad")
    assert json.loads(error_path.read_text(encoding="utf-8")) == {
        "schema": "world-understanding-durable-diagnostic-v1",
        "code": "material_scene_adapter_failed",
        "phase": "pipeline_execution",
        "retryable": False,
    }
    assert sentinel not in error_path.read_text(encoding="utf-8")


def test_clear_pipeline_state_noops_for_missing_state_and_unknown_step(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, session_id="state-noop")
    _clear_pipeline_state_from_step(config_path, "predict")

    state_file = tmp_path / ".state-noop" / ".pipeline_state.json"
    state_file.parent.mkdir(parents=True)
    original_state = {
        "completed_steps": ["validate_input"],
        "failed_steps": [],
        "step_outputs": {},
    }
    state_file.write_text(json.dumps(original_state), encoding="utf-8")

    _clear_pipeline_state_from_step(config_path, "not_a_step")
    assert json.loads(state_file.read_text(encoding="utf-8")) == original_state


def test_run_sub_asset_simulate_retry_clears_state_from_step(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, session_id="asset-sim")
    asset = _make_sub_asset("asset-sim", config_path=str(config_path))

    first = FakePipelineOutput(success=False, completed_steps=["optimize_usd"])
    second = FakePipelineOutput(success=True, completed_steps=["predict"])

    with (
        patch(
            "material_agent.scene.run._run_simulate", side_effect=[first, second]
        ) as mock_simulate,
        patch("material_agent.scene.run._clear_pipeline_state_from_step") as mock_clear,
        patch("material_agent.scene.run._clean_working_dir_for_so_retry") as mock_clean,
        patch("material_agent.scene.run._update_output_paths") as mock_update,
    ):
        result = run_sub_asset(
            asset,
            simulate=True,
            material_names=["Steel"],
            resume=True,
            from_step="predict",
        )

    assert result.status == "completed"
    mock_clear.assert_called_once_with(config_path, "predict")
    assert mock_simulate.call_count == 2
    mock_clean.assert_called_once_with(config_path)
    mock_update.assert_called_once()


def test_run_sub_asset_worker_success_path() -> None:
    asset = _make_sub_asset("worker")

    def fake_selected(sa, *args, **kwargs):
        sa.status = "completed"
        return sa

    with patch(
        "material_agent.scene.run._run_sub_asset_selected",
        side_effect=fake_selected,
    ):
        result = _run_sub_asset_worker(
            asset,
            skip_steps=None,
            only_steps=None,
            verbose=False,
        )

    assert result.status == "completed"


def test_run_all_skips_duplicate_members_and_copies_results(tmp_path: Path) -> None:
    rep = _make_sub_asset("rep", config_path=str(tmp_path / "rep.yaml"))
    member = _make_sub_asset(
        "member",
        config_path=str(tmp_path / "member.yaml"),
        instance_group="dup_group",
    )
    manifest = SceneManifest(
        sub_assets=[rep, member],
        instance_groups=[
            InstanceGroup(group_name="dup_group", representative_id=rep.id)
        ],
    )

    def fake_run_sequential(*args, **kwargs):
        rep.status = "completed"
        return 1, 0

    with (
        patch(
            "material_agent.scene.run._run_sequential", side_effect=fake_run_sequential
        ) as mock_seq,
        patch("material_agent.scene.run._copy_results_to_duplicates") as mock_copy,
    ):
        run_all(manifest, tmp_path / "manifest.json")

    assert mock_seq.call_args.args[0] == [rep]
    mock_copy.assert_called_once_with(manifest, [member])


def test_copy_results_to_duplicates_skips_missing_or_incomplete_representatives() -> (
    None
):
    missing_member = _make_sub_asset("missing", instance_group="missing_group")
    failed_rep = _make_sub_asset("failed_rep", status="failed", instance_group="dup")
    failed_member = _make_sub_asset("failed_member", instance_group="dup")
    manifest = SceneManifest(
        sub_assets=[failed_rep, failed_member],
        instance_groups=[
            InstanceGroup(group_name="dup", representative_id=failed_rep.id)
        ],
    )

    _copy_results_to_duplicates(manifest, [missing_member, failed_member])

    assert missing_member.status == "pending"
    assert failed_member.status == "pending"


def test_run_sequential_emits_progress_callback(tmp_path: Path) -> None:
    asset = _make_sub_asset("progress")
    manifest = SceneManifest(sub_assets=[asset])
    manifest.save = MagicMock()  # type: ignore[method-assign]
    progress: list[dict[str, Any]] = []

    def fake_selected(sa, *args, **kwargs):
        sa.status = "completed"
        return sa

    with patch(
        "material_agent.scene.run._run_sub_asset_selected",
        side_effect=fake_selected,
    ):
        completed, failed = _run_sequential(
            [asset],
            manifest,
            tmp_path / "manifest.json",
            skip_steps=None,
            only_steps=None,
            verbose=False,
            progress_callback=progress.append,
        )

    assert (completed, failed) == (1, 0)
    assert progress == [
        {
            "current": 1,
            "total": 1,
            "completed": 1,
            "failed": 0,
            "asset_id": asset.id,
            "asset_name": "progress",
            "asset_status": "completed",
        }
    ]


def test_run_parallel_empty_and_failed_status_progress(tmp_path: Path) -> None:
    manifest = SceneManifest(sub_assets=[])
    assert _run_parallel(
        [],
        manifest,
        tmp_path / "manifest.json",
        skip_steps=None,
        only_steps=None,
        verbose=False,
        max_workers=2,
    ) == (0, 0)

    asset = _make_sub_asset("failed", status="pending")
    manifest = SceneManifest(sub_assets=[asset])
    manifest.save = MagicMock()  # type: ignore[method-assign]
    progress: list[dict[str, Any]] = []

    def fake_worker(sa, *args, **kwargs):
        sa.status = "failed"
        return sa

    with patch(
        "material_agent.scene.run._run_sub_asset_worker", side_effect=fake_worker
    ):
        completed, failed = _run_parallel(
            [asset],
            manifest,
            tmp_path / "manifest.json",
            skip_steps=None,
            only_steps=None,
            verbose=False,
            max_workers=1,
            progress_callback=progress.append,
        )

    assert (completed, failed) == (0, 1)
    assert progress[-1]["asset_status"] == "failed"


def test_run_parallel_cancellation_branches(tmp_path: Path) -> None:
    asset = _make_sub_asset("cancel")
    manifest = SceneManifest(sub_assets=[asset])
    manifest.save = MagicMock()  # type: ignore[method-assign]
    cancel_calls = {"count": 0}

    def cancel_after_submit() -> bool:
        cancel_calls["count"] += 1
        return cancel_calls["count"] >= 2

    with patch("material_agent.scene.run._run_sub_asset_worker", return_value=asset):
        with pytest.raises(asyncio.CancelledError):
            _run_parallel(
                [asset],
                manifest,
                tmp_path / "manifest.json",
                skip_steps=None,
                only_steps=None,
                verbose=False,
                max_workers=1,
                cancel_checker=cancel_after_submit,
            )

    def raise_cancel(sa, *args, **kwargs):
        raise asyncio.CancelledError("worker cancelled")

    with patch(
        "material_agent.scene.run._run_sub_asset_worker", side_effect=raise_cancel
    ):
        with pytest.raises(asyncio.CancelledError):
            _run_parallel(
                [asset],
                manifest,
                tmp_path / "manifest.json",
                skip_steps=None,
                only_steps=None,
                verbose=False,
                max_workers=1,
                cancel_checker=lambda: False,
            )


def test_run_simulate_so_failure_runs_apply_without_restore(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        session_id="simulate-fail-so",
        extra={
            "steps": {
                "optimize_usd": {"enabled": True},
                "apply": {"material_library": "materials.yaml"},
            }
        },
    )
    calls: list[Any] = []

    def fake_run_pipeline(params):
        calls.append(params)
        if len(calls) == 1:
            return FakePipelineOutput(success=False, completed_steps=[])
        return FakePipelineOutput(success=True, completed_steps=["apply"])

    with (
        patch(
            "material_agent.api.pipeline.run_pipeline", side_effect=fake_run_pipeline
        ),
        patch(
            "material_agent.scene.run._generate_simulate_predictions", return_value=2
        ),
    ):
        result = _run_simulate(config_path, ["Steel"], verbose=True)

    assert result.success
    assert calls[0].only_steps == ["optimize_usd"]
    assert calls[1].only_steps == ["apply"]
    assert all(isinstance(call.config, dict) for call in calls)
    assert all(call.config_path == config_path for call in calls)


def test_run_simulate_absolute_input_and_prediction_paths(tmp_path: Path) -> None:
    absolute_usd = _touch(tmp_path / "absolute_asset.usd", "#usda 1.0\n")
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        session_id="simulate-absolute",
        extra={
            "input": {"usd_path": str(absolute_usd)},
            "steps": {
                "optimize_usd": {"enabled": True},
                "apply": {"enabled": True},
            },
        },
    )

    with (
        patch(
            "material_agent.api.pipeline.run_pipeline",
            side_effect=[
                FakePipelineOutput(success=True, completed_steps=["optimize_usd"]),
                FakePipelineOutput(
                    success=True, completed_steps=["restore_usd", "apply"]
                ),
            ],
        ),
        patch(
            "material_agent.scene.run._generate_simulate_predictions", return_value=1
        ),
        patch(
            "material_agent.scene.simulate.generate_mock_predictions_append",
            return_value=0,
        ) as mock_append,
    ):
        _run_simulate(config_path, ["Steel"], verbose=False)

    assert mock_append.call_args.kwargs["usd_path"] == absolute_usd

    with patch(
        "material_agent.scene.simulate.generate_mock_predictions", return_value=1
    ) as mock_generate:
        _generate_simulate_predictions(
            {"input": {"usd_path": str(absolute_usd)}},
            config_path,
            tmp_path / ".simulate-absolute",
            ["Steel"],
        )

    assert mock_generate.call_args.kwargs["usd_path"] == absolute_usd


def test_run_simulate_so_success_appends_original_predictions(tmp_path: Path) -> None:
    input_usd = _touch(tmp_path / "asset.usd", "#usda 1.0\n")
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        session_id="simulate-so",
        extra={
            "input": {"usd_path": input_usd.name, "prim_path": "/Root"},
            "steps": {
                "optimize_usd": {"enabled": True},
                "apply": {"enabled": True},
            },
        },
    )
    restored = tmp_path / ".simulate-so" / "restored" / "restored_predictions.jsonl"
    _touch(restored, "")

    with (
        patch(
            "material_agent.api.pipeline.run_pipeline",
            side_effect=[
                FakePipelineOutput(success=True, completed_steps=["optimize_usd"]),
                FakePipelineOutput(
                    success=True, completed_steps=["restore_usd", "apply"]
                ),
            ],
        ),
        patch(
            "material_agent.scene.run._generate_simulate_predictions", return_value=2
        ),
        patch(
            "material_agent.scene.simulate.generate_mock_predictions_append",
            return_value=3,
        ) as mock_append,
    ):
        result = _run_simulate(config_path, ["Steel"], verbose=False)

    assert result.success
    assert mock_append.call_args.kwargs["usd_path"] == input_usd.resolve()
    assert mock_append.call_args.kwargs["output_path"] == restored
    assert mock_append.call_args.kwargs["prim_path_scope"] == "/Root"


def test_generate_simulate_predictions_uses_original_relative_path(
    tmp_path: Path,
) -> None:
    config = {"input": {"usd_path": "asset.usd"}}
    config_path = tmp_path / "config.yaml"
    working_dir = tmp_path / ".session"

    with patch(
        "material_agent.scene.simulate.generate_mock_predictions", return_value=1
    ) as mock_generate:
        result = _generate_simulate_predictions(
            config,
            config_path,
            working_dir,
            ["Steel"],
        )

    assert result == 1
    assert (
        mock_generate.call_args.kwargs["usd_path"] == (tmp_path / "asset.usd").resolve()
    )


def test_run_payload_patch_resume_and_simulate_retry_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "payload.yaml"
    _write_config(config_path, session_id="payload")
    persisted = config_path.read_text()
    payload = _make_payload_group("payload", config_path=str(config_path))

    with (
        patch("material_agent.scene.run._clear_pipeline_state_from_step") as mock_clear,
        patch(
            "material_agent.api.pipeline.run_pipeline",
            return_value=FakePipelineOutput(success=True, completed_steps=["predict"]),
        ) as mock_run,
        patch("material_agent.scene.run._update_payload_output_paths") as mock_update,
    ):
        result = run_payload(
            payload,
            resume=True,
            from_step="predict",
            predict_max_workers=3,
        )

    assert result.status == "completed"
    pipeline_input = mock_run.call_args.args[0]
    assert pipeline_input.config["steps"]["predict"]["max_workers"] == 3
    assert pipeline_input.config_path == config_path
    assert config_path.read_text() == persisted
    mock_clear.assert_called_once_with(config_path, "predict")
    mock_update.assert_called_once_with(payload, config_path)

    with pytest.raises(ValueError, match="material_names"):
        run_payload(payload, simulate=True)

    first = FakePipelineOutput(success=False, completed_steps=["optimize_usd"])
    second = FakePipelineOutput(success=True, completed_steps=["predict"])
    with (
        patch(
            "material_agent.scene.run._run_simulate", side_effect=[first, second]
        ) as mock_simulate,
        patch("material_agent.scene.run._clean_working_dir_for_so_retry") as mock_clean,
        patch("material_agent.scene.run._update_payload_output_paths"),
    ):
        result = run_payload(payload, simulate=True, material_names=["Steel"])

    assert result.status == "completed"
    assert mock_simulate.call_count == 2
    mock_clean.assert_called_once_with(config_path)


def test_run_payload_worker_success_and_exception(tmp_path: Path) -> None:
    config_path = tmp_path / "payload.yaml"
    _write_config(config_path, session_id="payload")
    payload = _make_payload_group("payload", config_path=str(config_path))

    def fake_run_payload(pg, *args, **kwargs):
        pg.status = "completed"
        return pg

    with patch("material_agent.scene.run.run_payload", side_effect=fake_run_payload):
        result = _run_payload_worker(
            payload,
            skip_steps=None,
            only_steps=None,
            verbose=False,
        )
    assert result.status == "completed"

    with patch(
        "material_agent.scene.run.run_payload", side_effect=RuntimeError("boom")
    ):
        result = _run_payload_worker(
            payload,
            skip_steps=None,
            only_steps=None,
            verbose=False,
        )
    assert result.status == "failed"


def test_run_all_payloads_bottomup_edge_branches(tmp_path: Path) -> None:
    empty_manifest = SceneManifest(payload_groups=[])
    assert (
        run_all_payloads_bottomup(
            empty_manifest,
            tmp_path / "manifest.json",
            scene_config={},
            configs_dir=tmp_path / "configs",
        )
        is empty_manifest
    )

    configs_dir = tmp_path / "configs"
    completed = _make_payload_group(
        "done",
        depth=0,
        status="completed",
        config_path=str(configs_dir / "done.yaml"),
        working_dir=str(configs_dir / ".done"),
    )
    _touch(Path(completed.working_dir) / "output" / "output.usd")
    no_config = _make_payload_group("no_config", depth=2, config_path=None)
    parent = _make_payload_group("parent", depth=2, config_path=None)
    completed_parent = _make_payload_group(
        "done_parent",
        depth=2,
        status="completed",
        config_path=str(configs_dir / "done_parent.yaml"),
        working_dir=str(configs_dir / ".done_parent"),
    )
    _touch(Path(completed_parent.working_dir) / "output" / "output.usd")
    manifest = SceneManifest(
        payload_groups=[completed, no_config, parent, completed_parent]
    )
    manifest.save = MagicMock()  # type: ignore[method-assign]

    with (
        patch(
            "material_agent.scene.run._create_modified_parent_copy",
            side_effect=RuntimeError("parent failed"),
        ),
        patch("material_agent.scene.run._run_payloads_parallel") as mock_parallel,
    ):
        result = run_all_payloads_bottomup(
            manifest,
            tmp_path / "manifest.json",
            scene_config={},
            configs_dir=configs_dir,
            skip_existing=True,
            max_workers=2,
        )

    assert result is manifest
    assert completed.output_usd_path == str(
        Path(completed.working_dir) / "output" / "output.usd"
    )
    assert completed_parent.output_usd_path == str(
        Path(completed_parent.working_dir) / "output" / "output.usd"
    )
    assert no_config.status == "failed"
    assert parent.status == "failed"
    mock_parallel.assert_not_called()

    runnable = _make_payload_group(
        "runnable",
        depth=0,
        config_path=str(configs_dir / "runnable.yaml"),
    )
    manifest = SceneManifest(payload_groups=[runnable])
    manifest.save = MagicMock()  # type: ignore[method-assign]
    with patch(
        "material_agent.scene.run._run_payloads_parallel",
        return_value=(1, 0),
    ) as mock_parallel:
        run_all_payloads_bottomup(
            manifest,
            tmp_path / "manifest.json",
            scene_config={},
            configs_dir=configs_dir,
            max_workers=2,
        )

    mock_parallel.assert_called_once()


def test_payload_output_path_guards(tmp_path: Path) -> None:
    payload = _make_payload_group("payload")
    _set_payload_output_usd(payload)
    assert payload.output_usd_path is None

    payload.working_dir = str(tmp_path / "missing_work")
    _set_payload_output_usd(payload)
    assert payload.output_usd_path is None
