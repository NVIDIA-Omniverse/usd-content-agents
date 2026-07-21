# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional tests for material_agent.scene.cli command wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import typer

import material_agent.scene.cli as scene_cli
from material_agent.api.defaults import DEFAULT_LLM_BACKEND


@dataclass
class FakeSubAsset:
    id: str
    name: str
    prim_path: str
    mesh_count: int = 1
    vertex_count: int = 10
    instance_group: str | None = None
    status: str = "pending"


@dataclass
class FakeInstanceGroup:
    group_name: str
    instance_count: int
    representative_id: str | None = None


@dataclass
class FakePayloadGroup:
    id: str
    group_name: str
    payload_file: str
    instance_count: int
    status: str = "pending"
    depth: int = 0


class FakeManifest:
    def __init__(
        self,
        sub_assets: list[FakeSubAsset] | None = None,
        instance_groups: list[FakeInstanceGroup] | None = None,
        payload_groups: list[FakePayloadGroup] | None = None,
    ) -> None:
        self.sub_assets = sub_assets or []
        self.instance_groups = instance_groups or []
        self.payload_groups = payload_groups or []
        self.saved_paths: list[Path] = []

    def save(self, path: Path) -> None:
        self.saved_paths.append(Path(path))

    def get_processable_assets(
        self, names_filter: list[str] | None = None
    ) -> list[FakeSubAsset]:
        assets = self.sub_assets
        if names_filter:
            assets = [asset for asset in assets if asset.name in names_filter]
        return [asset for asset in assets if asset.status != "skipped"]

    def get_payloads_by_depth(self) -> dict[int, list[FakePayloadGroup]]:
        grouped: dict[int, list[FakePayloadGroup]] = {}
        for payload in self.payload_groups:
            grouped.setdefault(payload.depth, []).append(payload)
        return grouped


def _patch_console(monkeypatch: pytest.MonkeyPatch) -> Mock:
    printer = Mock()
    monkeypatch.setattr(scene_cli.console, "print", printer)
    return printer


def _record(called: dict[str, object], key: str, value: object) -> None:
    called[key] = value


def _record_and_return_value(
    called: dict[str, object], key: str, value: object, ret: object
) -> object:
    called[key] = value
    return ret


def test_print_manifest_summary_and_validation_stats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    printer = _patch_console(monkeypatch)
    manifest = FakeManifest(
        sub_assets=[
            FakeSubAsset("a", "AssetA", "/World/A", status="completed"),
            FakeSubAsset(
                "b", "AssetB", "/World/B", instance_group="group-1", status="failed"
            ),
        ],
        instance_groups=[FakeInstanceGroup("group-1", 2, "a")],
        payload_groups=[
            FakePayloadGroup(
                "p1", "PayloadOne", "payload_one.usd", 3, status="completed"
            )
        ],
    )

    scene_cli._print_manifest_summary(manifest)
    assert printer.call_count == 3

    report_path = tmp_path / "configs" / ".asset_a" / "predictions"
    report_path.mkdir(parents=True)
    (report_path / "validate_report.json").write_text(
        """
        {
          "stats": {
            "total": 5,
            "valid": 2,
            "auto_corrected": 1,
            "llm_repaired": 1,
            "failed": 1,
            "no_material": 0
          },
          "auto_corrected": [{"old": "stel", "new": "Steel"}],
          "llm_repaired": [{"old": "plasik", "new": "Plastic"}],
          "failed": [{"name": "mystery"}]
        }
        """
    )
    bad_report = tmp_path / "configs" / ".asset_b" / "predictions"
    bad_report.mkdir(parents=True)
    (bad_report / "validate_report.json").write_text("{bad json")

    printer.reset_mock()
    scene_cli._print_validation_stats(tmp_path)
    printed = "\n".join(str(call.args[0]) for call in printer.call_args_list)
    assert "Validation: 5 predictions checked" in printed
    assert "auto" in printed
    assert "LLM" in printed
    assert "UNFIXED" in printed

    printer.reset_mock()
    scene_cli._print_validation_stats(tmp_path / "empty")
    printer.assert_not_called()


def test_analyze_runs_scene_analysis_and_uses_default_llm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    printer = _patch_console(monkeypatch)
    manifest = FakeManifest(
        sub_assets=[
            FakeSubAsset("a", "AssetA", "/World/A", status="completed"),
            FakeSubAsset("b", "AssetB", "/World/B", status="skipped"),
        ],
        instance_groups=[FakeInstanceGroup("group-1", 2, "a")],
    )
    working_dir = tmp_path / ".demo_scene"
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("usd")
    called: dict[str, object] = {}

    monkeypatch.setattr(scene_cli, "_setup_logging", lambda verbose: None)
    monkeypatch.setattr(scene_cli, "_load_scene_config", lambda config: {"scene": {}})
    monkeypatch.setattr(
        scene_cli, "_resolve_usd_path", lambda scene_config, config: usd_path
    )
    monkeypatch.setattr(
        scene_cli, "_get_working_dir", lambda scene_config, config: working_dir
    )
    monkeypatch.setattr(
        scene_cli,
        "_print_manifest_summary",
        lambda manifest: _record(called, "summary", manifest),
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        lambda **kwargs: _record_and_return_value(called, "kwargs", kwargs, manifest),
    )

    scene_cli.analyze(tmp_path / "scene.yaml", verbose=True)

    kwargs = called["kwargs"]
    assert kwargs["scene_usd_path"] == usd_path
    assert kwargs["llm_config"]["backend"] == DEFAULT_LLM_BACKEND
    assert manifest.saved_paths == [working_dir / "manifest.json"]
    assert printer.call_count >= 3


def test_analyze_uses_configured_llm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_console(monkeypatch)
    working_dir = tmp_path / ".demo_scene"
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("usd")
    manifest = FakeManifest()
    called: dict[str, object] = {}

    monkeypatch.setattr(scene_cli, "_setup_logging", lambda verbose: None)
    monkeypatch.setattr(
        scene_cli,
        "_load_scene_config",
        lambda config: {"scene": {"analyze": {"llm": {"backend": "configured"}}}},
    )
    monkeypatch.setattr(
        scene_cli, "_resolve_usd_path", lambda scene_config, config: usd_path
    )
    monkeypatch.setattr(
        scene_cli, "_get_working_dir", lambda scene_config, config: working_dir
    )
    monkeypatch.setattr(
        scene_cli,
        "_print_manifest_summary",
        lambda manifest: _record(called, "summary", manifest),
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        lambda **kwargs: _record_and_return_value(called, "kwargs", kwargs, manifest),
    )

    scene_cli.analyze(tmp_path / "scene.yaml")

    assert called["kwargs"]["llm_config"] == {"backend": "configured"}


def test_extract_errors_without_manifest_and_runs_payload_config_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    printer = _patch_console(monkeypatch)
    working_dir = tmp_path / ".demo_scene"
    manifest_path = working_dir / "manifest.json"
    manifest = FakeManifest(
        sub_assets=[FakeSubAsset("a", "AssetA", "/World/A", status="extracted")],
        payload_groups=[FakePayloadGroup("p1", "PayloadOne", "payload.usd", 2)],
    )
    called: dict[str, object] = {}

    monkeypatch.setattr(scene_cli, "_setup_logging", lambda verbose: None)
    monkeypatch.setattr(
        scene_cli, "_load_scene_config", lambda config: {"scene": {"extract": {}}}
    )
    monkeypatch.setattr(
        scene_cli,
        "_resolve_usd_path",
        lambda scene_config, config: tmp_path / "scene.usd",
    )
    monkeypatch.setattr(
        scene_cli, "_get_working_dir", lambda scene_config, config: working_dir
    )

    with pytest.raises(typer.Exit):
        scene_cli.extract(tmp_path / "scene.yaml")
    assert "Manifest not found" in str(printer.call_args_list[-1].args[0])

    working_dir.mkdir()
    manifest_path.write_text("{}")
    monkeypatch.setattr(
        scene_cli, "SceneManifest", SimpleNamespace(load=lambda path: manifest)
    )
    monkeypatch.setattr(scene_cli, "_parse_assets_filter", lambda assets: ["AssetA"])
    monkeypatch.setattr(
        "material_agent.scene.extract.extract_all",
        lambda **kwargs: _record_and_return_value(called, "extract", kwargs, manifest),
    )
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_configs",
        lambda **kwargs: _record_and_return_value(called, "configs", kwargs, manifest),
    )
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_payload_configs",
        lambda **kwargs: _record_and_return_value(
            called, "payload_configs", kwargs, manifest
        ),
    )
    monkeypatch.setattr(scene_cli, "_print_manifest_summary", lambda manifest: None)

    scene_cli.extract(tmp_path / "scene.yaml", assets="AssetA")

    assert called["extract"]["names_filter"] == ["AssetA"]
    assert called["configs"]["configs_dir"] == working_dir / "configs"
    assert called["payload_configs"]["configs_dir"] == working_dir / "configs"
    assert manifest.saved_paths[-1] == manifest_path


def test_collect_handles_missing_library_and_successful_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_console(monkeypatch)
    working_dir = tmp_path / ".demo_scene"
    manifest_path = working_dir / "manifest.json"
    working_dir.mkdir()
    manifest_path.write_text("{}")
    manifest = FakeManifest(
        sub_assets=[FakeSubAsset("a", "AssetA", "/World/A", status="completed")],
        payload_groups=[FakePayloadGroup("p1", "Payload", "payload.usd", 2)],
    )
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("usd")
    material_yaml = tmp_path / "materials.yaml"
    material_yaml.write_text("entries: []\n")
    called: dict[str, object] = {}

    monkeypatch.setattr(scene_cli, "_setup_logging", lambda verbose: None)
    monkeypatch.setattr(
        scene_cli,
        "_load_scene_config",
        lambda config: {
            "scene": {"harmonize": {"enabled": True}},
            "steps": {
                "render": {
                    "enabled": True,
                    "image_width": 64,
                    "image_height": 32,
                    "camera_corners": ["+x"],
                    "camera_margin": 1.5,
                    "background_color": [0.1, 0.2, 0.3],
                }
            },
        },
    )
    monkeypatch.setattr(
        scene_cli, "_resolve_usd_path", lambda scene_config, config: usd_path
    )
    monkeypatch.setattr(
        scene_cli, "_get_working_dir", lambda scene_config, config: working_dir
    )
    monkeypatch.setattr(
        scene_cli, "SceneManifest", SimpleNamespace(load=lambda path: manifest)
    )
    monkeypatch.setattr(scene_cli, "_parse_assets_filter", lambda assets: ["AssetA"])
    monkeypatch.setattr(
        scene_cli, "_resolve_material_library_yaml", lambda scene_config, config: None
    )

    with pytest.raises(typer.Exit):
        scene_cli.collect(tmp_path / "scene.yaml")

    monkeypatch.setattr(
        scene_cli,
        "_resolve_material_library_yaml",
        lambda scene_config, config: material_yaml,
    )
    monkeypatch.setattr(
        "material_agent.scene.harmonize.harmonize_scene_predictions",
        lambda **kwargs: _record_and_return_value(
            called, "harmonize", kwargs, {"old": "new"}
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose",
        lambda **kwargs: _record(called, "compose", kwargs),
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.render_composed_scene",
        lambda **kwargs: _record_and_return_value(
            called, "render", kwargs, [tmp_path / "a.png", tmp_path / "b.png"]
        ),
    )

    scene_cli.collect(tmp_path / "scene.yaml", clear_materials=True)

    assert called["harmonize"]["mode"] == "simple"
    assert called["compose"]["material_library_yaml"] == material_yaml
    assert called["render"]["background_color"] == (0.1, 0.2, 0.3)
    assert called["render"]["clear_materials"] is True


def test_collect_and_validate_report_missing_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_console(monkeypatch)
    working_dir = tmp_path / ".demo_scene"
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("usd")

    monkeypatch.setattr(scene_cli, "_setup_logging", lambda verbose: None)
    monkeypatch.setattr(scene_cli, "_load_scene_config", lambda config: {"scene": {}})
    monkeypatch.setattr(
        scene_cli, "_resolve_usd_path", lambda scene_config, config: usd_path
    )
    monkeypatch.setattr(
        scene_cli, "_get_working_dir", lambda scene_config, config: working_dir
    )

    with pytest.raises(typer.Exit):
        scene_cli.collect(tmp_path / "scene.yaml")
    with pytest.raises(typer.Exit):
        scene_cli.validate(tmp_path / "scene.yaml")


def test_collect_handles_no_harmonize_changes_or_renders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    printer = _patch_console(monkeypatch)
    working_dir = tmp_path / ".demo_scene"
    manifest_path = working_dir / "manifest.json"
    working_dir.mkdir()
    manifest_path.write_text("{}")
    manifest = FakeManifest(
        sub_assets=[FakeSubAsset("a", "AssetA", "/World/A", status="completed")]
    )
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("usd")
    material_yaml = tmp_path / "materials.yaml"
    material_yaml.write_text("entries: []\n")

    monkeypatch.setattr(scene_cli, "_setup_logging", lambda verbose: None)
    monkeypatch.setattr(
        scene_cli,
        "_load_scene_config",
        lambda config: {"scene": {"harmonize": {"enabled": True}}},
    )
    monkeypatch.setattr(
        scene_cli, "_resolve_usd_path", lambda scene_config, config: usd_path
    )
    monkeypatch.setattr(
        scene_cli, "_get_working_dir", lambda scene_config, config: working_dir
    )
    monkeypatch.setattr(
        scene_cli, "SceneManifest", SimpleNamespace(load=lambda path: manifest)
    )
    monkeypatch.setattr(
        scene_cli,
        "_resolve_material_library_yaml",
        lambda scene_config, config: material_yaml,
    )
    monkeypatch.setattr(
        "material_agent.scene.harmonize.harmonize_scene_predictions",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.render_composed_scene", lambda **kwargs: []
    )

    scene_cli.collect(tmp_path / "scene.yaml")

    printed = "\n".join(str(call.args[0]) for call in printer.call_args_list)
    assert "No cross-asset conflicts" in printed
    assert "No renders produced" in printed


def test_run_validation_and_validate_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    printer = _patch_console(monkeypatch)
    report = SimpleNamespace(
        assets=[
            SimpleNamespace(ok=True, status="completed", name="ok"),
            SimpleNamespace(ok=True, status="inherited", name="ig"),
            SimpleNamespace(ok=False, status="failed", name="bad"),
        ],
        payloads=[
            SimpleNamespace(
                ok=False,
                name="payload-1",
                depth=1,
                predictions_count=2,
                instance_count=3,
                has_output_usd=False,
                errors=["missing predictions"],
                warnings=["warn"],
            )
        ],
        total_bindings=4,
        total_deinstanced=1,
        composed_scene_path="/tmp/composed.usd",
        composed_our=2,
        composed_old=1,
        composed_none=1,
        composed_instance_our=1,
        composed_instance_old=1,
        composed_instances_checked=3,
        composed_subset_our=1,
        composed_subset_old=0,
        composed_subsets_checked=2,
        errors=["scene error"],
        warnings=["scene warn"],
    )

    monkeypatch.setattr(
        "material_agent.scene.validate.validate_scene", lambda config, verbose: report
    )
    monkeypatch.setattr(
        "material_agent.scene.validate.format_asset_report",
        lambda asset, verbose: [f"asset:{asset.name}:{verbose}"],
    )
    monkeypatch.setattr(
        scene_cli, "_load_scene_config", lambda config: {"project": {"name": "demo"}}
    )
    monkeypatch.setattr(
        scene_cli,
        "_get_working_dir",
        lambda scene_config, config: tmp_path / ".demo_scene",
    )
    stats = Mock()
    monkeypatch.setattr(scene_cli, "_print_validation_stats", stats)

    exit_code = scene_cli._run_validation(tmp_path / "scene.yaml", verbose=True)
    assert exit_code == 1
    stats.assert_called_once()
    printed = "\n".join(str(call.args[0]) for call in printer.call_args_list)
    assert "RESULTS: 1/3 assets passed, 1 inherited, 1 failed" in printed
    assert "PAYLOADS: 0/1 passed, 1 failed" in printed
    assert "SCENE ERROR: scene error" in printed
    assert "SCENE WARN:  scene warn" in printed

    manifest_path = tmp_path / ".demo_scene" / "manifest.json"
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text("{}")
    monkeypatch.setattr(scene_cli, "_setup_logging", lambda verbose: None)
    monkeypatch.setattr(scene_cli, "_run_validation", lambda config, verbose: 3)
    with pytest.raises(typer.Exit) as exc:
        scene_cli.validate(tmp_path / "scene.yaml")
    assert exc.value.exit_code == 3


def test_run_cmd_legacy_resume_simulate_reconcile_harmonize_collect_and_validate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    printer = _patch_console(monkeypatch)
    working_dir = tmp_path / ".demo_scene"
    manifest_path = working_dir / "manifest.json"
    extracted_dir = working_dir / "extracted"
    configs_dir = working_dir / "configs"
    output_dir = working_dir / "output"
    for path in [working_dir, extracted_dir, configs_dir, output_dir]:
        path.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}")
    material_yaml = tmp_path / "materials.yaml"
    material_yaml.write_text("entries:\n  - name: Steel\n")

    manifest = FakeManifest(
        sub_assets=[
            FakeSubAsset("a", "AssetA", "/World/A", status="completed"),
            FakeSubAsset("b", "AssetB", "/World/B", status="completed"),
        ],
        payload_groups=[
            FakePayloadGroup(
                "p1", "PayloadOne", "payload.usd", 2, status="completed", depth=2
            )
        ],
    )
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("usd")
    called: dict[str, object] = {}
    config_path = tmp_path / "scene.yaml"
    config_path.write_text("project:\n  name: demo\n")

    scene_config = {
        "scene": {
            "reconcile": {"enabled": True},
            "harmonize": {"enabled": True},
        },
        "steps": {"render": {"enabled": True}},
    }

    monkeypatch.setattr(scene_cli, "_setup_logging", lambda verbose: None)
    monkeypatch.setattr(scene_cli, "_load_scene_config", lambda config: scene_config)
    monkeypatch.setattr(
        scene_cli, "_resolve_usd_path", lambda scene_config, config: usd_path
    )
    monkeypatch.setattr(
        scene_cli, "_get_working_dir", lambda scene_config, config: working_dir
    )
    monkeypatch.setattr(scene_cli, "_parse_assets_filter", lambda assets: ["AssetA"])
    monkeypatch.setattr(
        scene_cli, "_steps_before", lambda step_name: ["optimize_usd", "render_preview"]
    )
    monkeypatch.setattr(
        scene_cli,
        "_resolve_material_library_yaml",
        lambda scene_config, config: material_yaml,
    )
    monkeypatch.setattr(
        scene_cli,
        "_print_manifest_summary",
        lambda manifest: _record(called, "summary", True),
    )
    monkeypatch.setattr(
        scene_cli,
        "_print_validation_stats",
        lambda working_dir: _record(called, "validation_stats", working_dir),
    )
    monkeypatch.setattr(scene_cli, "_run_validation", lambda config, verbose: 0)
    monkeypatch.setattr(
        scene_cli, "SceneManifest", SimpleNamespace(load=lambda path: manifest)
    )
    monkeypatch.setattr(
        "material_agent.api.simulate_config.patch_config_for_simulate",
        lambda config, mock_analyze=False: dict(config, simulated=mock_analyze),
    )
    monkeypatch.setattr(
        "material_agent.scene.simulate.load_material_names_from_config",
        lambda scene_config, config: ["Steel", "Plastic"],
    )
    monkeypatch.setattr(
        "material_agent.scene.run.run_all",
        lambda **kwargs: _record_and_return(
            called, "run_all", kwargs, manifest, set_assets_status="completed"
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.run.run_all_payloads_bottomup",
        lambda **kwargs: _record_and_return(
            called, "run_payloads", kwargs, manifest, set_payloads_status="completed"
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.reconcile.reconcile_predictions",
        lambda **kwargs: _record_and_return_value(
            called, "reconcile", kwargs, {"old": "new"}
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.reconcile.apply_remapping",
        lambda manifest, remap: _record_and_return_value(
            called, "remap_applied", remap, 2
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.harmonize.harmonize_scene_predictions",
        lambda **kwargs: _record_and_return_value(
            called, "harmonize", kwargs, {"old": "new"}
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose",
        lambda **kwargs: _record(called, "compose", kwargs),
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.render_composed_scene",
        lambda **kwargs: _record_and_return_value(
            called, "render", kwargs, [output_dir / "a.png"]
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.stats.write_scene_stats_report",
        lambda **kwargs: _record_and_return_value(
            called, "stats_report", kwargs, output_dir / "stats.html"
        ),
    )

    scene_cli._run_cmd_legacy(
        config_path,
        assets="AssetA",
        skip="validate_predictions",
        only="predict,apply",
        from_step="predict",
        workers=2,
        skip_existing=True,
        simulate=True,
        simulate_mock_analyze=True,
        predict_max_workers=5,
        resume=True,
    )

    assert called["run_all"]["names_filter"] == ["AssetA"]
    assert called["run_all"]["only_steps"] == ["predict", "apply"]
    assert set(called["run_all"]["skip_steps"]) == {
        "validate_predictions",
        "optimize_usd",
        "render_preview",
    }
    assert called["run_all"]["skip_existing"] is True
    assert called["run_all"]["max_workers"] == 2
    assert called["run_all"]["resume"] is True
    assert called["run_all"]["from_step"] == "predict"
    assert called["run_all"]["predict_max_workers"] == 5
    assert called["run_all"]["material_names"] == ["Steel", "Plastic"]
    assert called["run_payloads"]["configs_dir"] == configs_dir
    assert called["run_payloads"]["scene_config_dir"] == tmp_path
    assert called["reconcile"]["materials_list"] == ["Steel"]
    assert called["remap_applied"] == {"old": "new"}
    assert called["harmonize"]["llm_config"] is None
    assert called["compose"]["output_usd_path"] == output_dir / "composed_scene.usd"
    assert called["compose"]["material_library_yaml"] == material_yaml
    assert called["render"]["output_dir"] == output_dir
    assert called["render"]["clear_materials"] is False
    assert called["validation_stats"] == working_dir
    assert called["stats_report"]["output_dir"] == output_dir
    printed = "\n".join(str(call.args[0]) for call in printer.call_args_list)
    assert "Validation passed" in printed
    assert "Scene pipeline complete" in printed


def test_run_cmd_prints_optional_results_and_simulate_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    printer = _patch_console(monkeypatch)
    config_path = tmp_path / "scene.yaml"
    config_path.write_text("project:\n  name: demo\n")

    from material_agent.api import ScenePipelineOutput

    outputs = [
        ScenePipelineOutput(
            success=True,
            output_usd_path=str(tmp_path / "out.usd"),
            rendered_images=[str(tmp_path / "a.png")],
            stats_report_path=str(tmp_path / "stats.html"),
            completed_assets=2,
            failed_assets=1,
            completed_payloads=1,
            failed_payloads=0,
            validation_passed=False,
            warnings=["careful now"],
        ),
        ScenePipelineOutput(success=True, completed_assets=0, failed_assets=0),
    ]
    calls: list[object] = []

    def fake_run_scene_pipeline(params: object) -> ScenePipelineOutput:
        calls.append(params)
        return outputs.pop(0)

    monkeypatch.setattr(
        "material_agent.api.run_scene_pipeline",
        fake_run_scene_pipeline,
    )

    scene_cli.run_cmd(config_path, simulate=True, simulate_mock_analyze=True)
    scene_cli.run_cmd(config_path, simulate=True, simulate_mock_analyze=False)

    assert calls[0].simulate_mock_analyze is True
    assert calls[1].simulate_mock_analyze is False
    printed = "\n".join(str(call.args[0]) for call in printer.call_args_list)
    assert "including analyze LLM" in printed
    assert "analyze LLM kept real" in printed
    assert "Rendered 1 images" in printed
    assert "Scene stats report" in printed
    assert "Payloads" in printed
    assert "careful now" in printed


def test_run_agent_resets_assets_and_runs_payloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    printer = _patch_console(monkeypatch)
    working_dir = tmp_path / ".demo_scene"
    working_dir.mkdir()
    manifest_path = working_dir / "manifest.json"
    manifest_path.write_text("{}")
    config_path = tmp_path / "scene.yaml"
    config_path.write_text("project:\n  name: demo\n")
    scene_config = {"project": {"name": "demo"}}
    manifest = FakeManifest(
        sub_assets=[
            FakeSubAsset("a", "AssetA", "/World/A", status="completed"),
            FakeSubAsset("b", "AssetB", "/World/B", status="failed"),
        ],
        payload_groups=[
            FakePayloadGroup("p1", "PayloadOne", "payload.usd", 2, depth=1)
        ],
    )
    called: dict[str, object] = {}
    simulate_patch_calls: list[tuple[dict[str, object], bool]] = []

    def patch_for_simulate(
        config: dict[str, object], mock_analyze: bool = False
    ) -> dict[str, object]:
        simulate_patch_calls.append((config, mock_analyze))
        return dict(config, mock_analyze=mock_analyze)

    monkeypatch.setattr(scene_cli, "_setup_logging", lambda verbose: None)
    monkeypatch.setattr(scene_cli, "_load_scene_config", lambda config: scene_config)
    monkeypatch.setattr(
        scene_cli, "_get_working_dir", lambda scene_config, config: working_dir
    )
    monkeypatch.setattr(scene_cli, "_parse_assets_filter", lambda assets: ["AssetA"])
    monkeypatch.setattr(
        scene_cli, "_steps_before", lambda step_name: ["render_preview"]
    )
    monkeypatch.setattr(
        scene_cli,
        "_print_manifest_summary",
        lambda manifest: _record(called, "summary", manifest),
    )
    monkeypatch.setattr(
        scene_cli, "SceneManifest", SimpleNamespace(load=lambda path: manifest)
    )
    monkeypatch.setattr(
        "material_agent.api.simulate_config.patch_config_for_simulate",
        patch_for_simulate,
    )
    monkeypatch.setattr(
        "material_agent.scene.simulate.load_material_names_from_config",
        lambda scene_config, config: ["Steel", "Plastic"],
    )
    monkeypatch.setattr(
        "material_agent.scene.run.run_all",
        lambda **kwargs: _record_and_return(
            called,
            "run_all",
            kwargs,
            manifest,
            set_assets_status="completed",
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.run.run_all_payloads_bottomup",
        lambda **kwargs: _record_and_return(
            called,
            "payloads",
            kwargs,
            manifest,
            set_payloads_status="completed",
        ),
    )

    scene_cli.run_agent_cmd(
        config_path,
        assets="AssetA",
        skip="validate_predictions",
        only="predict,apply",
        from_step="predict",
        skip_existing=True,
        workers=2,
        simulate=True,
        simulate_mock_analyze=True,
        predict_max_workers=4,
        verbose=True,
    )

    assert called["run_all"]["names_filter"] == ["AssetA"]
    assert "render_preview" in called["run_all"]["skip_steps"]
    assert called["run_all"]["only_steps"] == ["predict", "apply"]
    assert called["run_all"]["resume"] is True
    assert called["run_all"]["simulate"] is True
    assert called["run_all"]["material_names"] == ["Steel", "Plastic"]
    assert called["run_all"]["predict_max_workers"] == 4
    assert called["run_all"]["scene_config"] is scene_config
    assert called["payloads"]["configs_dir"] == working_dir / "configs"
    assert called["payloads"]["max_workers"] == 2
    assert called["payloads"]["scene_config"] is scene_config
    assert simulate_patch_calls == [(scene_config, True)]
    assert called["summary"] is manifest
    assert manifest.saved_paths
    printed = "\n".join(str(call.args[0]) for call in printer.call_args_list)
    assert "payloads" in printed
    assert "Simulate mode" in printed


def test_run_agent_cmd_exits_when_manifest_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_console(monkeypatch)
    working_dir = tmp_path / ".demo_scene"
    config_path = tmp_path / "scene.yaml"
    config_path.write_text("project:\n  name: demo\n")

    monkeypatch.setattr(scene_cli, "_setup_logging", lambda verbose: None)
    monkeypatch.setattr(scene_cli, "_load_scene_config", lambda config: {})
    monkeypatch.setattr(
        scene_cli, "_get_working_dir", lambda scene_config, config: working_dir
    )

    with pytest.raises(typer.Exit):
        scene_cli.run_agent_cmd(config_path)


def test_run_cmd_legacy_fresh_run_warning_validation_and_empty_renders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    printer = _patch_console(monkeypatch)
    working_dir = tmp_path / ".demo_scene"
    working_dir.mkdir()
    (working_dir / "stale.txt").write_text("old")
    output_dir = working_dir / "output"
    material_yaml = tmp_path / "materials.yaml"
    material_yaml.write_text("entries:\n  - name: Steel\n")
    config_path = tmp_path / "scene.yaml"
    config_path.write_text("project:\n  name: demo\n")
    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("usd")
    manifest = FakeManifest(
        sub_assets=[
            FakeSubAsset("a", "AssetA", "/World/A", status="extracted"),
            FakeSubAsset("b", "AssetB", "/World/B", status="skipped"),
        ],
        instance_groups=[FakeInstanceGroup("instances", 2, "a")],
        payload_groups=[
            FakePayloadGroup("p1", "PayloadOne", "payload.usd", 2, depth=1)
        ],
    )
    scene_config = {
        "scene": {
            "analyze": {
                "skip_geometry": True,
                "building_block_min_reuse": 7,
                "llm": {"backend": "custom"},
            },
            "filters": {"kind": "mesh"},
            "extract": {"flatten": False, "max_workers": 3},
            "reconcile": {"enabled": True, "llm": {"backend": "judge"}},
            "harmonize": {"enabled": True},
        },
        "steps": {"render": {"enabled": True}},
    }
    called: dict[str, object] = {}

    monkeypatch.setattr(scene_cli, "_setup_logging", lambda verbose: None)
    monkeypatch.setattr(scene_cli, "_load_scene_config", lambda config: scene_config)
    monkeypatch.setattr(
        scene_cli, "_resolve_usd_path", lambda scene_config, config: usd_path
    )
    monkeypatch.setattr(
        scene_cli, "_get_working_dir", lambda scene_config, config: working_dir
    )
    monkeypatch.setattr(scene_cli, "_parse_assets_filter", lambda assets: ["AssetA"])
    monkeypatch.setattr(
        scene_cli,
        "_resolve_material_library_yaml",
        lambda scene_config, config: material_yaml,
    )
    monkeypatch.setattr(
        scene_cli,
        "_print_manifest_summary",
        lambda manifest: _record(called, "summary", manifest),
    )
    monkeypatch.setattr(
        scene_cli,
        "_print_validation_stats",
        lambda working_dir: _record(called, "validation_stats", working_dir),
    )
    monkeypatch.setattr(scene_cli, "_run_validation", lambda config, verbose: 2)
    monkeypatch.setattr(
        "material_agent.api.simulate_config.patch_config_for_simulate",
        lambda config, mock_analyze=False: config,
    )
    monkeypatch.setattr(
        "material_agent.scene.simulate.load_material_names_from_config",
        lambda scene_config, config: ["Steel"],
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        lambda **kwargs: _record_and_return_value(called, "analyze", kwargs, manifest),
    )
    monkeypatch.setattr(
        "material_agent.scene.extract.extract_all",
        lambda **kwargs: _record_and_return_value(called, "extract", kwargs, manifest),
    )
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_configs",
        lambda **kwargs: _record_and_return_value(called, "configs", kwargs, manifest),
    )
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_payload_configs",
        lambda **kwargs: _record_and_return_value(
            called, "payload_configs", kwargs, manifest
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.run.run_all",
        lambda **kwargs: _record_and_return(
            called, "run_all", kwargs, manifest, set_assets_status="completed"
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.run.run_all_payloads_bottomup",
        lambda **kwargs: _record_and_return(
            called, "payloads", kwargs, manifest, set_payloads_status="completed"
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.reconcile.reconcile_predictions",
        lambda **kwargs: _record_and_return_value(called, "reconcile", kwargs, {}),
    )
    monkeypatch.setattr(
        "material_agent.scene.harmonize.harmonize_scene_predictions",
        lambda **kwargs: _record_and_return_value(called, "harmonize", kwargs, {}),
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose",
        lambda **kwargs: _record(called, "compose", kwargs),
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.render_composed_scene",
        lambda **kwargs: _record_and_return_value(called, "render", kwargs, []),
    )
    monkeypatch.setattr(
        "material_agent.scene.stats.write_scene_stats_report",
        lambda **kwargs: _record_and_return_value(
            called, "stats_report", kwargs, output_dir / "stats.html"
        ),
    )

    scene_cli._run_cmd_legacy(
        config_path,
        assets="AssetA",
        resume=True,
        clean=True,
        simulate=True,
        simulate_mock_analyze=False,
        fail_on_validation=False,
    )

    assert not (working_dir / "stale.txt").exists()
    assert called["analyze"]["skip_geometry"] is True
    assert called["analyze"]["building_block_min_reuse"] == 7
    assert called["analyze"]["filters"] == {"kind": "mesh"}
    assert called["extract"]["flatten"] is False
    assert called["extract"]["max_workers"] == 3
    assert called["configs"]["configs_dir"] == working_dir / "configs"
    assert called["payload_configs"]["configs_dir"] == working_dir / "configs"
    assert called["run_all"]["material_names"] == ["Steel"]
    assert called["reconcile"]["llm_config"] == {"backend": "judge"}
    assert called["harmonize"]["mode"] == "full"
    assert called["compose"]["names_filter"] == ["AssetA"]
    assert called["render"]["output_dir"] == output_dir
    assert called["validation_stats"] == working_dir
    printed = "\n".join(str(call.args[0]) for call in printer.call_args_list)
    assert "no existing state found" in printed
    assert "No renders produced" in printed
    assert "Validation failed" in printed
    assert "No reconciliation needed" in printed
    assert "No cross-asset conflicts" in printed


def test_run_cmd_legacy_exits_on_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_console(monkeypatch)
    working_dir = tmp_path / ".demo_scene"
    working_dir.mkdir()
    (working_dir / "manifest.json").write_text("{}")
    (working_dir / "extracted").mkdir()
    (working_dir / "configs").mkdir()
    material_yaml = tmp_path / "materials.yaml"
    material_yaml.write_text("entries: []\n")
    config_path = tmp_path / "scene.yaml"
    config_path.write_text("project:\n  name: demo\n")
    manifest = FakeManifest(
        sub_assets=[FakeSubAsset("a", "AssetA", "/World/A", status="completed")]
    )

    monkeypatch.setattr(scene_cli, "_setup_logging", lambda verbose: None)
    monkeypatch.setattr(
        scene_cli,
        "_load_scene_config",
        lambda config: {"scene": {"harmonize": {"enabled": False}}},
    )
    monkeypatch.setattr(
        scene_cli,
        "_resolve_usd_path",
        lambda scene_config, config: tmp_path / "scene.usd",
    )
    monkeypatch.setattr(
        scene_cli, "_get_working_dir", lambda scene_config, config: working_dir
    )
    monkeypatch.setattr(
        scene_cli, "SceneManifest", SimpleNamespace(load=lambda path: manifest)
    )
    monkeypatch.setattr(
        scene_cli,
        "_resolve_material_library_yaml",
        lambda scene_config, config: material_yaml,
    )
    monkeypatch.setattr(
        scene_cli,
        "_print_manifest_summary",
        lambda manifest: None,
    )
    monkeypatch.setattr(scene_cli, "_print_validation_stats", lambda working_dir: None)
    monkeypatch.setattr(scene_cli, "_run_validation", lambda config, verbose: 5)
    monkeypatch.setattr("material_agent.scene.run.run_all", lambda **kwargs: manifest)
    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "material_agent.scene.stats.write_scene_stats_report",
        lambda **kwargs: tmp_path / "stats.html",
    )

    with pytest.raises(typer.Exit) as exc:
        scene_cli._run_cmd_legacy(config_path, resume=True, no_render=True)

    assert exc.value.exit_code == 5


def test_run_cmd_legacy_exits_when_material_library_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_console(monkeypatch)
    working_dir = tmp_path / ".demo_scene"
    working_dir.mkdir()
    (working_dir / "manifest.json").write_text("{}")
    (working_dir / "extracted").mkdir()
    (working_dir / "configs").mkdir()
    config_path = tmp_path / "scene.yaml"
    config_path.write_text("project:\n  name: demo\n")
    manifest = FakeManifest(
        sub_assets=[FakeSubAsset("a", "AssetA", "/World/A", status="completed")]
    )

    monkeypatch.setattr(scene_cli, "_setup_logging", lambda verbose: None)
    monkeypatch.setattr(
        scene_cli,
        "_load_scene_config",
        lambda config: {"scene": {"harmonize": {"enabled": False}}},
    )
    monkeypatch.setattr(
        scene_cli,
        "_resolve_usd_path",
        lambda scene_config, config: tmp_path / "scene.usd",
    )
    monkeypatch.setattr(
        scene_cli, "_get_working_dir", lambda scene_config, config: working_dir
    )
    monkeypatch.setattr(
        scene_cli, "SceneManifest", SimpleNamespace(load=lambda path: manifest)
    )
    monkeypatch.setattr(
        scene_cli,
        "_resolve_material_library_yaml",
        lambda scene_config, config: None,
    )
    monkeypatch.setattr("material_agent.scene.run.run_all", lambda **kwargs: manifest)

    with pytest.raises(typer.Exit):
        scene_cli._run_cmd_legacy(config_path, resume=True)


def _record_and_return(
    called: dict[str, object],
    key: str,
    kwargs: dict[str, object],
    manifest: FakeManifest,
    set_assets_status: str | None = None,
    set_payloads_status: str | None = None,
) -> FakeManifest:
    called[key] = kwargs
    if set_assets_status is not None:
        for asset in manifest.sub_assets:
            asset.status = set_assets_status
    if set_payloads_status is not None:
        for payload in manifest.payload_groups:
            payload.status = set_payloads_status
    return manifest


def test_bundle_handles_missing_inputs_and_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    printer = _patch_console(monkeypatch)
    working_dir = tmp_path / ".demo_scene"
    output_dir = working_dir / "output"
    output_dir.mkdir(parents=True)
    composed = output_dir / "composed_scene.usd"
    material_yaml = tmp_path / "materials.yaml"
    material_yaml.write_text("entries: []\n")
    called: dict[str, object] = {}

    monkeypatch.setattr(scene_cli, "_setup_logging", lambda verbose: None)
    monkeypatch.setattr(
        scene_cli, "_load_scene_config", lambda config: {"project": {"name": "demo"}}
    )
    monkeypatch.setattr(
        scene_cli, "_get_working_dir", lambda scene_config, config: working_dir
    )
    monkeypatch.setattr(
        scene_cli,
        "_resolve_material_library_yaml",
        lambda scene_config, config: material_yaml,
    )

    with pytest.raises(typer.Exit):
        scene_cli.bundle(tmp_path / "scene.yaml")

    composed.write_text("usd")
    monkeypatch.setattr(
        scene_cli,
        "_resolve_material_library_yaml",
        lambda scene_config, config: None,
    )
    with pytest.raises(typer.Exit):
        scene_cli.bundle(tmp_path / "scene.yaml")

    monkeypatch.setattr(
        scene_cli,
        "_resolve_material_library_yaml",
        lambda scene_config, config: material_yaml,
    )
    monkeypatch.setattr(
        "material_agent.scene.bundle.create_bundle",
        lambda **kwargs: _record_and_return_value(
            called,
            "bundle",
            kwargs,
            {
                "usd_file": output_dir / "bundle" / "scene.usdc",
                "usd_size_mb": 10,
                "library_files": 4,
                "total_size_mb": 25,
                "verified_paths": 9,
                "missing_paths": 1,
            },
        ),
    )

    scene_cli.bundle(tmp_path / "scene.yaml", format="usda")

    assert called["bundle"]["output_format"] == ".usda"
    printed = "\n".join(str(call.args[0]) for call in printer.call_args_list)
    assert "Bundle created" in printed
    assert "WARNING: 1 unresolved paths" in printed
