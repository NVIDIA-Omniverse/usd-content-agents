# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the public large-scene Python API."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from material_agent.api.scene_pipeline import (
    ScenePipelineInput,
    ScenePipelineOutput,
    _build_output,
    _emit_safe_event,
    _get_working_dir,
    _load_material_names,
    _load_render_camera_config,
    _load_scene_config,
    _reject_retired_scene_harness_config,
    _report_to_dict,
    _report_validation_passed,
    _resolve_or_materialize_material_library_yaml,
    _resolve_output_path,
    _resolve_usd_path,
    arun_scene_pipeline,
    ascene_pipeline,
    run_scene_pipeline,
    scene_pipeline,
)
from material_agent.scene.manifest import PayloadGroup, SceneManifest, SubAsset
from material_agent.scene.validate import AssetReport, PayloadReport, SceneReport

scene_pipeline_module = importlib.import_module("material_agent.api.scene_pipeline")


class RecordingListener:
    """Minimal listener for API event assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.warnings: list[str] = []

    def event(self, event_type: str, data: dict[str, Any], **kwargs: Any) -> None:
        self.events.append((event_type, data))

    def info(self, message: str, **kwargs: Any) -> None:
        pass

    def debug(self, message: str, **kwargs: Any) -> None:
        pass

    def warning(self, message: str, **kwargs: Any) -> None:
        self.warnings.append(message)

    def error(self, message: str, **kwargs: Any) -> None:
        pass


def test_scene_harness_api_flags_are_retired(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="content-workflow-cli materials assign"):
        ScenePipelineInput(
            config=_scene_config(tmp_path),
            config_base_dir=tmp_path,
            harness_sub_assets=True,
        )


def test_scene_harness_config_section_is_retired(tmp_path: Path) -> None:
    config = _scene_config(tmp_path)
    config["scene"]["harness"] = {
        "enabled": True,
        "sub_assets": {
            "max_iterations": 2,
        },
    }

    params = ScenePipelineInput(
        config=config,
        config_base_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="content-workflow-cli materials assign"):
        _reject_retired_scene_harness_config(config, params)


def test_scene_harness_config_rejects_any_legacy_harness_section(
    tmp_path: Path,
) -> None:
    config = _scene_config(tmp_path)
    config["scene"]["harness"] = {"max_iterations": 2}

    params = ScenePipelineInput(
        config=config,
        config_base_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="content-workflow-cli materials assign"):
        _reject_retired_scene_harness_config(config, params)


def test_scene_harness_config_allows_explicitly_disabled_harness_section(
    tmp_path: Path,
) -> None:
    config = _scene_config(tmp_path)
    config["scene"]["harness"] = {
        "enabled": False,
        "max_iterations": 2,
    }

    params = ScenePipelineInput(
        config=config,
        config_base_dir=tmp_path,
    )

    _reject_retired_scene_harness_config(config, params)


def test_scene_harness_config_rejects_legacy_scene_keys(tmp_path: Path) -> None:
    config = _scene_config(tmp_path)
    config["scene"]["harness_request"] = {"prompt": "assign materials"}

    params = ScenePipelineInput(
        config=config,
        config_base_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="content-workflow-cli materials assign"):
        _reject_retired_scene_harness_config(config, params)


def _scene_config(tmp_path: Path) -> dict[str, Any]:
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text(
        """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
}
"""
    )
    material_lib = tmp_path / "materials.usda"
    material_lib.write_text("#usda 1.0\n")

    return {
        "project": {
            "name": "scene_api",
            "working_dir": str(tmp_path / "scene_work"),
        },
        "input": {"usd_path": str(usd_path)},
        "materials": {
            "library_path": str(material_lib),
            "entries": [
                {
                    "name": "Steel",
                    "description": "Test steel",
                    "prim_path": "/World/Looks/Steel",
                }
            ],
        },
        "scene": {
            "extract": {"flatten": True, "max_workers": 1},
            "reconcile": {"enabled": False},
            "harmonize": {"enabled": False},
        },
        "steps": {
            "render": {"enabled": False},
        },
    }


def test_run_scene_pipeline_orchestrates_and_materializes_inline_materials(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = SceneManifest(
        scene_usd_path=str(tmp_path / "scene.usda"),
        sub_assets=[
            SubAsset(
                id="asset_a",
                name="AssetA",
                prim_path="/Root/AssetA",
                mesh_count=2,
            )
        ],
    )
    called: dict[str, Any] = {}

    def fake_analyze_scene(**kwargs: Any) -> SceneManifest:
        called["analyze"] = kwargs
        return manifest

    def fake_extract_all(**kwargs: Any) -> SceneManifest:
        called["extract"] = kwargs
        for sub_asset in manifest.sub_assets:
            sub_asset.extracted_usd = str(tmp_path / "asset.usda")
            sub_asset.status = "extracted"
        return manifest

    def fake_generate_all_configs(**kwargs: Any) -> SceneManifest:
        called["config_gen"] = kwargs
        config_path = tmp_path / "asset.yaml"
        config_path.write_text("project:\n  session_id: asset_a\n")
        for sub_asset in manifest.sub_assets:
            sub_asset.config_path = str(config_path)
            sub_asset.working_dir = str(tmp_path / ".asset_a")
        return manifest

    def fake_run_all(**kwargs: Any) -> SceneManifest:
        called["run_all"] = kwargs
        for sub_asset in manifest.sub_assets:
            sub_asset.status = "completed"
        progress_callback = kwargs.get("progress_callback")
        if progress_callback:
            progress_callback(
                {
                    "current": 1,
                    "total": 1,
                    "completed": 1,
                    "failed": 0,
                    "asset_id": "asset_a",
                    "asset_name": "AssetA",
                    "asset_status": "completed",
                }
            )
        return manifest

    def fake_apply_and_compose(**kwargs: Any) -> Path:
        called["collect"] = kwargs
        output_path = Path(kwargs["output_usd_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("#usda 1.0\n")
        return output_path

    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        fake_analyze_scene,
    )
    monkeypatch.setattr(
        "material_agent.scene.extract.extract_all",
        fake_extract_all,
    )
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_configs",
        fake_generate_all_configs,
    )
    monkeypatch.setattr(
        "material_agent.scene.run.run_all",
        fake_run_all,
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose",
        fake_apply_and_compose,
    )
    listener = RecordingListener()

    result = run_scene_pipeline(
        ScenePipelineInput(
            config=_scene_config(tmp_path),
            config_base_dir=tmp_path,
            no_render=True,
            validate_output=False,
            event_listener=listener,
        )
    )

    assert result.success
    assert result.completed_assets == 1
    assert result.failed_assets == 0
    assert Path(result.output_usd_path).exists()

    material_yaml = Path(called["collect"]["material_library_yaml"])
    material_data = yaml.safe_load(material_yaml.read_text())
    assert material_data["library_path"] == "../../materials.usda"
    assert material_data["entries"][0]["binding"] == "/World/Looks/Steel"
    assert "prim_path" in material_data["entries"][0]
    assert called["config_gen"]["scene_config"]["materials"] == {
        "path": str(material_yaml)
    }

    assert called["extract"]["flatten"] is True
    assert called["config_gen"]["scene_config_dir"] == tmp_path

    completed_scene_steps = [
        data["step_name"]
        for event_type, data in listener.events
        if event_type == "step.completed"
        and data.get("workflow_type") == "scene_pipeline"
    ]
    assert completed_scene_steps == [
        "scene_analyze",
        "scene_extract",
        "scene_run_assets",
        "scene_run_payloads",
        "scene_reconcile",
        "scene_harmonize",
        "scene_collect",
        "scene_render",
        "scene_validate",
    ]
    asset_progress_events = [
        data
        for event_type, data in listener.events
        if event_type == "step.progress" and data.get("step_name") == "scene_run_assets"
    ]
    assert asset_progress_events
    assert asset_progress_events[-1]["current"] == 1
    assert asset_progress_events[-1]["total"] == 1
    assert asset_progress_events[-1]["percent"] == 100
    assert asset_progress_events[-1]["asset_name"] == "AssetA"


def test_run_scene_pipeline_passes_render_camera_config_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = SceneManifest(
        scene_usd_path=str(tmp_path / "scene.usda"),
        sub_assets=[
            SubAsset(
                id="asset_a",
                name="AssetA",
                prim_path="/Root/AssetA",
                mesh_count=1,
                status="completed",
            )
        ],
    )
    camera_json = tmp_path / "scene_camera.json"
    camera_json.write_text(
        """
{
  "name": "scene_humanoid_camera",
  "focal_length_mm": 130.0,
  "image_width": 1920,
  "image_height": 1080
}
""",
        encoding="utf-8",
    )
    config = _scene_config(tmp_path)
    config["steps"]["render"] = {
        "enabled": True,
        "camera_config_path": camera_json.name,
        "image_width": 64,
        "image_height": 64,
    }
    called: dict[str, Any] = {}

    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.extract.extract_all",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_configs",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.run.run_all",
        lambda **kwargs: manifest,
    )

    def fake_apply_and_compose(**kwargs: Any) -> Path:
        output_path = Path(kwargs["output_usd_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("#usda 1.0\n")
        return output_path

    def fake_render_composed_scene(**kwargs: Any) -> list[Path]:
        called["render"] = kwargs
        render_path = tmp_path / "render.png"
        render_path.write_text("png")
        return [render_path]

    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose",
        fake_apply_and_compose,
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.render_composed_scene",
        fake_render_composed_scene,
    )

    result = run_scene_pipeline(
        ScenePipelineInput(
            config=config,
            config_base_dir=tmp_path,
            validate_output=False,
        )
    )

    assert result.success
    assert result.rendered_images == [str(tmp_path / "render.png")]
    assert called["render"]["camera_config"]["name"] == "scene_humanoid_camera"
    assert called["render"]["camera_config"]["_source_path"] == str(camera_json)


def test_run_scene_pipeline_validates_explicit_output_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = SceneManifest(
        scene_usd_path=str(tmp_path / "scene.usda"),
        sub_assets=[
            SubAsset(
                id="asset_a",
                name="AssetA",
                prim_path="/Root/AssetA",
                mesh_count=1,
            )
        ],
    )
    called: dict[str, Any] = {}

    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        lambda **kwargs: manifest,
    )

    def fake_extract_all(**kwargs: Any) -> SceneManifest:
        for sub_asset in manifest.sub_assets:
            sub_asset.extracted_usd = str(tmp_path / "asset.usda")
            sub_asset.status = "extracted"
        return manifest

    def fake_generate_all_configs(**kwargs: Any) -> SceneManifest:
        config_path = tmp_path / "asset.yaml"
        config_path.write_text("project:\n  session_id: asset_a\n")
        for sub_asset in manifest.sub_assets:
            sub_asset.config_path = str(config_path)
            sub_asset.working_dir = str(tmp_path / ".asset_a")
        return manifest

    def fake_run_all(**kwargs: Any) -> SceneManifest:
        for sub_asset in manifest.sub_assets:
            sub_asset.status = "completed"
        return manifest

    def fake_apply_and_compose(**kwargs: Any) -> Path:
        output_path = Path(kwargs["output_usd_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("#usda 1.0\n")
        return output_path

    def fake_validate_scene_outputs(**kwargs: Any) -> SceneReport:
        called["validate"] = kwargs
        return SceneReport()

    monkeypatch.setattr("material_agent.scene.extract.extract_all", fake_extract_all)
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_configs",
        fake_generate_all_configs,
    )
    monkeypatch.setattr("material_agent.scene.run.run_all", fake_run_all)
    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose",
        fake_apply_and_compose,
    )
    monkeypatch.setattr(
        "material_agent.scene.validate.validate_scene_outputs",
        fake_validate_scene_outputs,
    )

    result = run_scene_pipeline(
        ScenePipelineInput(
            config=_scene_config(tmp_path),
            config_base_dir=tmp_path,
            output_usd_path=Path("exports/scene_with_materials.usd"),
            no_render=True,
            validate_output=True,
        )
    )

    assert result.success
    assert result.validation_passed is True
    assert Path(called["validate"]["manifest_path"]) == Path(result.manifest_path)
    assert Path(called["validate"]["working_dir"]) == Path(result.working_dir)
    assert Path(called["validate"]["composed_scene_path"]) == Path(
        result.output_usd_path
    )
    assert Path(result.output_usd_path) == tmp_path / "exports" / (
        "scene_with_materials.usd"
    )


def test_run_scene_pipeline_rebases_relative_config_and_output_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "project"
    (base_dir / "assets").mkdir(parents=True)
    (base_dir / "libs").mkdir()
    usd_path = base_dir / "assets" / "scene.usda"
    usd_path.write_text(
        """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
}
"""
    )
    material_lib = base_dir / "libs" / "materials.usda"
    material_lib.write_text("#usda 1.0\n")

    config = {
        "project": {"name": "relative_scene", "working_dir": "work"},
        "input": {"usd_path": "assets/scene.usda"},
        "materials": {
            "library_path": "libs/materials.usda",
            "entries": [
                {
                    "name": "Steel",
                    "description": "Test steel",
                    "prim_path": "/World/Looks/Steel",
                }
            ],
        },
        "scene": {
            "extract": {"flatten": True, "max_workers": 1},
            "reconcile": {"enabled": False},
            "harmonize": {"enabled": False},
        },
        "steps": {"render": {"enabled": False}},
    }
    manifest = SceneManifest(
        scene_usd_path=str(usd_path),
        sub_assets=[SubAsset(id="asset_a", name="AssetA", prim_path="/Root/AssetA")],
    )
    called: dict[str, Any] = {}

    def fake_analyze_scene(**kwargs: Any) -> SceneManifest:
        called["analyze"] = kwargs
        return manifest

    def fake_extract_all(**kwargs: Any) -> SceneManifest:
        called["extract"] = kwargs
        for sub_asset in manifest.sub_assets:
            sub_asset.extracted_usd = str(base_dir / "asset.usda")
            sub_asset.status = "extracted"
        return manifest

    def fake_generate_all_configs(**kwargs: Any) -> SceneManifest:
        called["config_gen"] = kwargs
        for sub_asset in manifest.sub_assets:
            sub_asset.config_path = str(base_dir / "work" / "configs" / "asset.yaml")
            sub_asset.working_dir = str(base_dir / "work" / "configs" / ".asset")
        return manifest

    def fake_run_all(**kwargs: Any) -> SceneManifest:
        called["run_all"] = kwargs
        for sub_asset in manifest.sub_assets:
            sub_asset.status = "completed"
        return manifest

    def fake_apply_and_compose(**kwargs: Any) -> Path:
        called["collect"] = kwargs
        output_path = Path(kwargs["output_usd_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("#usda 1.0\n")
        return output_path

    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        fake_analyze_scene,
    )
    monkeypatch.setattr(
        "material_agent.scene.extract.extract_all",
        fake_extract_all,
    )
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_configs",
        fake_generate_all_configs,
    )
    monkeypatch.setattr(
        "material_agent.scene.run.run_all",
        fake_run_all,
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose",
        fake_apply_and_compose,
    )

    result = run_scene_pipeline(
        ScenePipelineInput(
            config=config,
            config_base_dir=base_dir,
            output_usd_path=Path("exports/scene_with_materials.usd"),
            no_render=True,
            validate_output=False,
        )
    )

    assert result.success
    assert Path(called["analyze"]["scene_usd_path"]) == usd_path
    assert called["config_gen"]["scene_config_dir"] == base_dir
    assert Path(called["collect"]["output_usd_path"]) == (
        base_dir / "exports" / "scene_with_materials.usd"
    )
    assert Path(result.output_usd_path) == (
        base_dir / "exports" / "scene_with_materials.usd"
    )
    material_yaml = Path(called["collect"]["material_library_yaml"])
    material_data = yaml.safe_load(material_yaml.read_text())
    assert material_yaml == base_dir / "work" / "materials" / "materials.yaml"
    assert material_data["library_path"] == "../../libs/materials.usda"


def test_run_scene_pipeline_from_step_sets_resume_and_skip_steps(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = SceneManifest(
        sub_assets=[SubAsset(id="asset_a", name="AssetA", prim_path="/Root/AssetA")],
    )
    called: dict[str, Any] = {}

    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.extract.extract_all",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_configs",
        lambda **kwargs: manifest,
    )

    def fake_run_all(**kwargs: Any) -> SceneManifest:
        called["run_all"] = kwargs
        for sub_asset in manifest.sub_assets:
            sub_asset.status = "completed"
        return manifest

    monkeypatch.setattr("material_agent.scene.run.run_all", fake_run_all)

    def fake_collect(**kwargs: Any) -> Path:
        output_path = Path(kwargs["output_usd_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("#usda 1.0\n")
        return output_path

    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose",
        fake_collect,
    )

    result = run_scene_pipeline(
        ScenePipelineInput(
            config=_scene_config(tmp_path),
            config_base_dir=tmp_path,
            from_step="predict",
            skip_steps=["render"],
            no_render=True,
            validate_output=False,
        )
    )

    assert result.success
    assert called["run_all"]["resume"] is True
    assert "build_dataset_prepare_dataset" in called["run_all"]["skip_steps"]
    assert "cluster_prims" in called["run_all"]["skip_steps"]
    assert "render" in called["run_all"]["skip_steps"]


def test_run_scene_pipeline_requires_default_root_prim(
    monkeypatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "never-replay-scene-validation-path-713"
    sensitive_dir = tmp_path / f"api_key={sentinel}"
    sensitive_dir.mkdir()
    config = _scene_config(sensitive_dir)
    Path(config["input"]["usd_path"]).write_text("#usda 1.0\n")

    def fail_analyze_scene(**kwargs: Any) -> SceneManifest:
        raise AssertionError("analyze_scene should not run for invalid input")

    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        fail_analyze_scene,
    )

    result = run_scene_pipeline(
        ScenePipelineInput(
            config=config,
            config_base_dir=tmp_path,
            no_render=True,
            validate_output=False,
        )
    )

    assert not result.success
    assert result.error == "Scene pipeline failed"
    assert sentinel not in caplog.text


def test_scene_status_events_project_credential_bearing_paths() -> None:
    listener = RecordingListener()
    sentinel = "never-emit-scene-status-path-713"

    _emit_safe_event(
        listener,
        "workflow.completed",
        {
            "output_usd_path": f"scene.usd?X-Amz-Signature={sentinel}",
            "validation_report": {"error": f"api_key={sentinel}"},
        },
    )

    assert sentinel not in repr(listener.events)
    assert "<redacted>" in repr(listener.events)


def test_scene_output_projects_public_metadata_without_mutating_manifest(
    tmp_path: Path,
) -> None:
    sentinel = "api_key=scene-public-result-secret-713"
    manifest = SceneManifest(
        analysis={
            "summary": "safe",
            "api_key": sentinel,
            "config_dict": {"vlm": {"api_key": sentinel}},
            "runtime_collaborator": object(),
        }
    )

    result = _build_output(
        success=True,
        error=None,
        working_dir=tmp_path / sentinel / "work",
        manifest_path=tmp_path / "manifest.json",
        output_path=tmp_path / "output.usd",
        rendered_images=[
            str(tmp_path / "safe.png"),
            str(tmp_path / sentinel / "secret.png"),
        ],
        manifest=manifest,
        validation_passed=True,
        validation_report={
            "passed": True,
            "api_key": sentinel,
            "runtime_collaborator": object(),
        },
        scene_harness_summary_path="",
        warnings=["safe warning", sentinel],
    )

    assert result.success is True
    assert result.working_dir == ""
    assert result.manifest_path == str(tmp_path / "manifest.json")
    assert result.output_usd_path == str(tmp_path / "output.usd")
    assert result.rendered_images == [str(tmp_path / "safe.png")]
    assert result.validation_report is not None
    assert result.validation_report["passed"] is True
    assert "runtime_collaborator" not in result.validation_report
    assert result.raw_result["analysis"]["summary"] == "safe"
    assert "config_dict" not in result.raw_result["analysis"]
    assert "runtime_collaborator" not in result.raw_result["analysis"]
    assert sentinel not in repr(result)
    assert manifest.analysis["api_key"] == sentinel
    assert manifest.analysis["config_dict"]["vlm"]["api_key"] == sentinel


def test_run_scene_pipeline_cancel_checker_stops_before_stages(
    tmp_path: Path,
) -> None:
    listener = RecordingListener()

    with pytest.raises(asyncio.CancelledError):
        run_scene_pipeline(
            ScenePipelineInput(
                config=_scene_config(tmp_path),
                config_base_dir=tmp_path,
                cancel_checker=lambda: True,
                no_render=True,
                validate_output=False,
                event_listener=listener,
            )
        )

    assert (
        "workflow.cancelled",
        {
            "workflow_type": "scene_pipeline",
            "step_name": "scene_pipeline",
            "message": "Scene pipeline cancellation requested",
        },
    ) in listener.events


def test_scene_pipeline_input_and_helper_edge_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="Config dictionary cannot be empty"):
        ScenePipelineInput(config={})
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        ScenePipelineInput(config=tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="max_workers"):
        ScenePipelineInput(config=_scene_config(tmp_path), max_workers=0)
    with pytest.raises(ValueError, match="predict_max_workers"):
        ScenePipelineInput(
            config=_scene_config(tmp_path),
            predict_max_workers=0,
        )

    config_path = tmp_path / "scene_config.yaml"
    config = _scene_config(tmp_path)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded, loaded_path, base_dir = _load_scene_config(ScenePipelineInput(config_path))
    assert loaded["project"]["name"] == "scene_api"
    assert loaded_path == config_path.resolve()
    assert base_dir == tmp_path

    bad_config_path = tmp_path / "bad_scene_config.yaml"
    bad_config_path.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Scene config must contain a mapping"):
        _load_scene_config(ScenePipelineInput(bad_config_path))

    assert _get_working_dir({"project": "not-a-mapping"}, tmp_path) == (
        tmp_path / ".scene_scene"
    )
    assert _get_working_dir({"project": {"session_id": "abc"}}, tmp_path) == (
        tmp_path / ".abc_scene"
    )
    absolute_output = tmp_path / "absolute.usd"
    assert (
        _resolve_output_path(absolute_output, tmp_path / "work", tmp_path)
        == absolute_output
    )

    with pytest.raises(ValueError, match="input section"):
        _resolve_usd_path({"input": []}, tmp_path)

    params = ScenePipelineInput(
        config=_scene_config(tmp_path), config_base_dir=tmp_path
    )
    _reject_retired_scene_harness_config({"scene": "not-a-mapping"}, params)
    with pytest.raises(ValueError, match="content-workflow-cli materials assign"):
        _reject_retired_scene_harness_config({"scene": {"harness": ["legacy"]}}, params)

    material_yaml = tmp_path / "existing_materials.yaml"
    material_yaml.write_text("library_path: materials.usda\nentries: []\n")
    material_path_config = {"materials": {"path": material_yaml.name}}
    assert (
        _resolve_or_materialize_material_library_yaml(
            material_path_config,
            tmp_path,
            tmp_path / "work",
        )
        == material_yaml.resolve()
    )
    assert material_path_config["materials"]["path"] == str(material_yaml.resolve())

    with pytest.raises(ValueError, match="materials section"):
        _resolve_or_materialize_material_library_yaml(
            {"materials": []},
            tmp_path,
            tmp_path / "work",
        )

    library_usd = tmp_path / "materials.usda"
    library_usd.write_text("#usda 1.0\n", encoding="utf-8")
    relpath_config = {
        "materials": {
            "library_path": str(library_usd),
            "entries": [{"name": "Steel", "prim_path": "/Looks/Steel"}],
        }
    }
    monkeypatch.setattr(
        scene_pipeline_module.os.path,
        "relpath",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("different drives")),
    )
    materialized = _resolve_or_materialize_material_library_yaml(
        relpath_config,
        tmp_path,
        tmp_path / "work",
    )
    materialized_data = yaml.safe_load(materialized.read_text())
    assert materialized_data["library_path"] == str(library_usd.resolve())

    assert _load_render_camera_config(
        {"camera_config": {"name": "inline"}}, tmp_path
    ) == {"name": "inline"}
    assert _load_render_camera_config({}, tmp_path) is None

    assert _report_to_dict({"ok": True}) == {"ok": True}
    assert "object" in _report_to_dict(object())["repr"]
    assert not _report_validation_passed(
        SceneReport(assets=[AssetReport(name="asset", errors=["bad"])])
    )
    assert not _report_validation_passed(
        SceneReport(payloads=[PayloadReport(name="payload", errors=["bad"])])
    )
    assert not _report_validation_passed(SceneReport(errors=["scene failed"]))


def test_load_material_names_accepts_flat_and_nested_yaml(tmp_path: Path) -> None:
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    assert _load_material_names(empty_yaml) == []

    flat_yaml = tmp_path / "flat.yaml"
    flat_yaml.write_text(
        yaml.safe_dump({"entries": [{"name": "Steel"}, {"name": ""}, "bad"]}),
        encoding="utf-8",
    )
    assert _load_material_names(flat_yaml) == ["Steel"]

    nested_yaml = tmp_path / "nested.yaml"
    nested_yaml.write_text(
        yaml.safe_dump({"materials": {"entries": [{"name": "Wood"}]}}),
        encoding="utf-8",
    )
    assert _load_material_names(nested_yaml) == ["Wood"]

    none_entries_yaml = tmp_path / "none_entries.yaml"
    none_entries_yaml.write_text("entries:\n", encoding="utf-8")
    assert _load_material_names(none_entries_yaml) == []

    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        _load_material_names(list_yaml)

    bad_materials_yaml = tmp_path / "bad_materials.yaml"
    bad_materials_yaml.write_text("materials:\n  - bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping at 'materials'"):
        _load_material_names(bad_materials_yaml)

    bad_entries_yaml = tmp_path / "bad_entries.yaml"
    bad_entries_yaml.write_text("entries: not-a-list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'entries' must be a list"):
        _load_material_names(bad_entries_yaml)


def test_run_scene_pipeline_resume_payloads_reconcile_harmonize_and_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _scene_config(tmp_path)
    config["steps"]["render"] = "not-a-mapping"
    config["scene"]["reconcile"] = {"enabled": True, "llm": {"backend": "mock"}}
    config["scene"]["harmonize"] = {"enabled": True, "llm": {"backend": "mock"}}
    working_dir = Path(config["project"]["working_dir"])
    manifest_path = working_dir / "manifest.json"
    (working_dir / "extracted").mkdir(parents=True)
    (working_dir / "configs").mkdir()

    manifest = SceneManifest(
        scene_usd_path=config["input"]["usd_path"],
        sub_assets=[
            SubAsset(
                id="asset_a",
                name="AssetA",
                prim_path="/Root/AssetA",
                status="completed",
            )
        ],
        payload_groups=[
            PayloadGroup(
                id="payload_a",
                group_name="PayloadA",
                payload_file=str(tmp_path / "payload.usda"),
                status="completed",
            )
        ],
    )
    manifest.save(manifest_path)

    called: dict[str, Any] = {}

    monkeypatch.setattr(
        "material_agent.api.simulate_config.patch_config_for_simulate",
        lambda scene_config, mock_analyze=False: scene_config,
    )
    monkeypatch.setattr(
        "material_agent.scene.simulate.load_material_names_from_config",
        lambda scene_config, config_ref: ["Steel"],
    )

    def fake_run_all(**kwargs: Any) -> SceneManifest:
        run_manifest = kwargs["manifest"]
        called["run_all"] = kwargs
        called["asset_status_before_run"] = run_manifest.sub_assets[0].status
        run_manifest.sub_assets[0].status = "completed"
        return run_manifest

    def fake_run_all_payloads_bottomup(**kwargs: Any) -> SceneManifest:
        run_manifest = kwargs["manifest"]
        called["run_payloads"] = kwargs
        called["payload_status_before_run"] = run_manifest.payload_groups[0].status
        run_manifest.payload_groups[0].status = "completed"
        return run_manifest

    def fake_reconcile_predictions(**kwargs: Any) -> dict[str, str]:
        called["reconcile"] = kwargs
        return {"Old": "Steel"}

    def fake_apply_remapping(manifest: SceneManifest, remap: dict[str, str]) -> None:
        called["remap"] = remap

    def fake_harmonize_scene_predictions(**kwargs: Any) -> None:
        called["harmonize"] = kwargs

    def fake_apply_and_compose(**kwargs: Any) -> Path:
        output_path = Path(kwargs["output_usd_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("#usda 1.0\n", encoding="utf-8")
        return output_path

    def fake_render_composed_scene(**kwargs: Any) -> list[Path]:
        called["render"] = kwargs
        render_path = tmp_path / "rendered.png"
        render_path.write_text("png", encoding="utf-8")
        return [render_path]

    def fake_validate_scene_outputs(**kwargs: Any) -> SceneReport:
        return SceneReport(assets=[AssetReport(name="AssetA", errors=["missing"])])

    def fake_write_scene_stats_report(**kwargs: Any) -> Path:
        called["stats"] = kwargs
        stats_path = tmp_path / "stats.json"
        stats_path.write_text("{}", encoding="utf-8")
        return stats_path

    monkeypatch.setattr("material_agent.scene.run.run_all", fake_run_all)
    monkeypatch.setattr(
        "material_agent.scene.run.run_all_payloads_bottomup",
        fake_run_all_payloads_bottomup,
    )
    monkeypatch.setattr(
        "material_agent.scene.reconcile.reconcile_predictions",
        fake_reconcile_predictions,
    )
    monkeypatch.setattr(
        "material_agent.scene.reconcile.apply_remapping",
        fake_apply_remapping,
    )
    monkeypatch.setattr(
        "material_agent.scene.harmonize.harmonize_scene_predictions",
        fake_harmonize_scene_predictions,
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose",
        fake_apply_and_compose,
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.render_composed_scene",
        fake_render_composed_scene,
    )
    monkeypatch.setattr(
        "material_agent.scene.validate.validate_scene_outputs",
        fake_validate_scene_outputs,
    )
    monkeypatch.setattr(
        "material_agent.scene.stats.write_scene_stats_report",
        fake_write_scene_stats_report,
    )

    listener = RecordingListener()
    result = run_scene_pipeline(
        ScenePipelineInput(
            config=config,
            config_base_dir=tmp_path,
            from_step="predict",
            resume=True,
            simulate=True,
            simulate_mock_analyze=True,
            validate_output=True,
            fail_on_validation_error=True,
            event_listener=listener,
        )
    )

    assert not result.success
    assert result.error == "Scene validation failed"
    assert result.validation_passed is False
    assert result.rendered_images == [str(tmp_path / "rendered.png")]
    assert result.stats_report_path == str(tmp_path / "stats.json")
    assert called["asset_status_before_run"] == "extracted"
    assert called["payload_status_before_run"] == "pending"
    assert called["run_all"]["material_names"] == ["Steel"]
    assert called["run_payloads"]["material_names"] == ["Steel"]
    assert called["remap"] == {"Old": "Steel"}
    assert called["harmonize"]["mode"] == "full"
    assert called["render"]["image_width"] == 1024
    assert any(event_type == "workflow.failed" for event_type, _ in listener.events)


def test_run_scene_pipeline_clean_non_mapping_options_and_payload_config_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _scene_config(tmp_path)
    config["scene"]["analyze"] = []
    config["scene"]["filters"] = []
    config["scene"]["extract"] = []
    config["scene"]["harmonize"] = []
    working_dir = Path(config["project"]["working_dir"])
    working_dir.mkdir(parents=True)
    (working_dir / "stale.txt").write_text("old", encoding="utf-8")

    manifest = SceneManifest(
        scene_usd_path=config["input"]["usd_path"],
        sub_assets=[SubAsset(id="asset_a", name="AssetA", prim_path="/Root/AssetA")],
        payload_groups=[
            PayloadGroup(
                id="payload_a",
                group_name="PayloadA",
                payload_file=str(tmp_path / "payload.usda"),
            )
        ],
    )
    called: dict[str, Any] = {}

    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.extract.extract_all",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_configs",
        lambda **kwargs: manifest,
    )

    def fake_generate_all_payload_configs(**kwargs: Any) -> SceneManifest:
        called["payload_config"] = kwargs
        return manifest

    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_payload_configs",
        fake_generate_all_payload_configs,
    )

    def fake_run_all(**kwargs: Any) -> SceneManifest:
        assert kwargs["cancel_checker"]() is False
        return manifest

    monkeypatch.setattr("material_agent.scene.run.run_all", fake_run_all)
    monkeypatch.setattr(
        "material_agent.scene.run.run_all_payloads_bottomup",
        lambda **kwargs: manifest,
    )

    def fake_collect(**kwargs: Any) -> Path:
        output_path = Path(kwargs["output_usd_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("#usda 1.0\n", encoding="utf-8")
        return output_path

    monkeypatch.setattr("material_agent.scene.collect.apply_and_compose", fake_collect)
    monkeypatch.setattr(
        "material_agent.scene.harmonize.harmonize_scene_predictions",
        lambda **kwargs: None,
    )

    result = run_scene_pipeline(
        ScenePipelineInput(
            config=config,
            config_base_dir=tmp_path,
            clean=True,
            no_render=True,
            validate_output=False,
        )
    )

    assert result.success
    assert not (working_dir / "stale.txt").exists()
    assert called["payload_config"]["configs_dir"] == working_dir / "configs"


def test_run_scene_pipeline_accepts_non_mapping_scene_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _scene_config(tmp_path)
    config["scene"] = "not-a-mapping"
    manifest = SceneManifest(
        scene_usd_path=config["input"]["usd_path"],
        sub_assets=[SubAsset(id="asset_a", name="AssetA", prim_path="/Root/AssetA")],
    )

    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.extract.extract_all",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_configs",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.run.run_all",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.harmonize.harmonize_scene_predictions",
        lambda **kwargs: None,
    )

    def fake_collect(**kwargs: Any) -> Path:
        output_path = Path(kwargs["output_usd_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("#usda 1.0\n", encoding="utf-8")
        return output_path

    monkeypatch.setattr("material_agent.scene.collect.apply_and_compose", fake_collect)

    result = run_scene_pipeline(
        ScenePipelineInput(
            config=config,
            config_base_dir=tmp_path,
            no_render=True,
            validate_output=False,
        )
    )

    assert result.success


def test_run_scene_pipeline_cancel_checker_stops_worker_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = SceneManifest(
        scene_usd_path=str(tmp_path / "scene.usda"),
        sub_assets=[SubAsset(id="asset_a", name="AssetA", prim_path="/Root/AssetA")],
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.extract.extract_all",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_configs",
        lambda **kwargs: manifest,
    )

    def fake_run_all(**kwargs: Any) -> SceneManifest:
        kwargs["cancel_checker"]()
        return manifest

    monkeypatch.setattr("material_agent.scene.run.run_all", fake_run_all)
    listener = RecordingListener()
    cancel_calls = {"count": 0}

    def cancel_during_worker() -> bool:
        cancel_calls["count"] += 1
        return cancel_calls["count"] >= 6

    with pytest.raises(asyncio.CancelledError):
        run_scene_pipeline(
            ScenePipelineInput(
                config=_scene_config(tmp_path),
                config_base_dir=tmp_path,
                cancel_checker=cancel_during_worker,
                no_render=True,
                validate_output=False,
                event_listener=listener,
            )
        )

    assert any(
        event_type == "step.cancelled" and data["step_name"] == "scene_run_assets"
        for event_type, data in listener.events
    )


def test_scene_pipeline_convenience_wrappers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = ScenePipelineOutput(success=True, working_dir=str(tmp_path))
    seen: list[ScenePipelineInput] = []

    def fake_run(params: ScenePipelineInput) -> ScenePipelineOutput:
        seen.append(params)
        return output

    monkeypatch.setattr(scene_pipeline_module, "run_scene_pipeline", fake_run)

    assert scene_pipeline(_scene_config(tmp_path), no_render=True) is output
    async_result = asyncio.run(
        arun_scene_pipeline(ScenePipelineInput(config=_scene_config(tmp_path)))
    )
    assert async_result is output
    async_convenience = asyncio.run(ascene_pipeline(_scene_config(tmp_path)))
    assert async_convenience is output
    assert len(seen) == 3
