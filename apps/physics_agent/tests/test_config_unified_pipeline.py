# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import traceback
from pathlib import Path

import pytest

from physics_agent.config import (
    STEP_ORDER,
    STEP_OUTPUT_DIRS,
    ConfigValidator,
    ProjectPathResolver,
    UnifiedPipelineConfigTask,
    get_default_config,
    get_step_defaults,
)
from physics_agent.workflows import create_unified_pipeline_workflow


def _path_endswith(path: str, *parts: str) -> bool:
    return Path(path).parts[-len(parts) :] == parts


def test_schema_helpers_and_unified_workflow_exports():
    defaults = get_default_config()

    assert defaults["project"]["name"] == "physics_agent_project"
    assert defaults["input"]["reference_images"] == []
    assert STEP_ORDER[0] == "optimize_usd"
    assert STEP_OUTPUT_DIRS["predict"] == "predictions"
    assert get_step_defaults("unknown_step") == {"enabled": True}
    assert get_step_defaults("predict")["allow_empty_predictions"] is False
    assert get_step_defaults("apply_physics")["mass_scale_policy"] == "skip_mass"
    assert get_step_defaults("apply_physics")["allow_empty_predictions"] is False

    workflow = create_unified_pipeline_workflow()
    assert [task.name for task in workflow.tasks] == [
        "UnifiedConfigLoading",
        "UnifiedPipelineExecutor",
    ]


def test_project_path_resolver_resolves_paths_and_summarizes(tmp_path: Path):
    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n")
    ref_image = tmp_path / "ref.png"
    ref_image.write_bytes(b"ref")
    config_path = tmp_path / "configs" / "pipeline.yaml"
    config_path.parent.mkdir()
    config_path.write_text("project:\n  name: demo\n")

    resolver = ProjectPathResolver(
        config={
            "project": {
                "name": "demo",
                "working_dir": "work",
                "session_id": "session-123",
            },
            "input": {
                "usd_path": "../asset.usd",
                "reference_images": ["../ref.png"],
            },
            "steps": {},
            "advanced": {},
        },
        config_file_path=config_path,
    )

    resolver.create_working_directories()
    resolver.validate_input_paths()
    summary = resolver.get_path_summary()

    assert resolver.input_usd == usd_path.resolve()
    assert resolver.reference_images == [ref_image.resolve()]
    assert (
        resolver.get_step_output_dir("predict") == resolver.working_dir / "predictions"
    )
    assert resolver.get_step_dataset_file("predict").name == "dataset.jsonl"
    assert resolver.get_step_predictions_file().name == "predictions.jsonl"
    assert resolver.get_usd_dataset_dir() == resolver.working_dir / "dataset" / "usd"
    assert resolver.get_dataset_dir() == resolver.working_dir / "dataset"
    assert summary["input"]["usd_path"] == str(usd_path.resolve())
    assert summary["step_outputs"]["predict"] == str(
        resolver.working_dir / "predictions"
    )


def test_project_path_resolver_redacts_missing_input_diagnostics(
    tmp_path: Path,
) -> None:
    secret = "physics-input-path-secret-713"
    config_dir = tmp_path / f"user:{secret}@assets.example.test"
    config_dir.mkdir()
    resolver = ProjectPathResolver(
        config={
            "project": {"name": "safe", "working_dir": "work"},
            "input": {"usd_path": "missing.usd"},
            "steps": {},
            "advanced": {},
        },
        config_file_path=config_dir / "config.yaml",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        resolver.validate_input_paths()

    observable = f"{exc_info.value}\n{resolver!r}\n{resolver.get_path_summary()!r}"
    assert secret not in observable
    assert str(exc_info.value) == "Input USD file not found: <redacted>"


def test_config_validator_handles_required_fields_and_warns(
    monkeypatch: pytest.MonkeyPatch,
):
    validator = ConfigValidator()

    with pytest.raises(ValueError, match="Missing required section"):
        validator.validate({"project": {"name": "demo"}})

    with pytest.raises(ValueError, match="project.name"):
        validator.validate({"project": {}, "input": {"usd_path": "/tmp/a.usd"}})

    warnings: list[str] = []

    def record_warning(message: str, *args: object) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr("physics_agent.config.validator.logger.warning", record_warning)
    validator.validate(
        {
            "project": {"name": "demo"},
            "input": {"usd_path": "/tmp/a.usd"},
            "steps": {"unknown_step": {"enabled": True}},
        }
    )

    assert warnings
    assert "Unknown step 'unknown_step'" in warnings[0]

    with pytest.raises(ValueError, match="predict.output_key must be a string"):
        validator.validate_step_requirements(
            "predict",
            {"output_key": 123},
            {"project": {"name": "demo"}, "input": {"usd_path": "/tmp/a.usd"}},
        )

    with pytest.raises(ValueError, match="predict.allow_empty_predictions"):
        validator.validate_step_requirements(
            "predict",
            {"allow_empty_predictions": "yes"},
            {"project": {"name": "demo"}, "input": {"usd_path": "/tmp/a.usd"}},
        )

    with pytest.raises(ValueError, match="apply_physics.mass_scale_policy"):
        validator.validate_step_requirements(
            "apply_physics",
            {"mass_scale_policy": "bad"},
            {"project": {"name": "demo"}, "input": {"usd_path": "/tmp/a.usd"}},
        )

    with pytest.raises(ValueError, match="apply_physics.allow_empty_predictions"):
        validator.validate_step_requirements(
            "apply_physics",
            {"allow_empty_predictions": "yes"},
            {"project": {"name": "demo"}, "input": {"usd_path": "/tmp/a.usd"}},
        )


def test_validator_unknown_step_log_redacts_credential_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "physics-validator-key-secret-713"
    target = logging.getLogger("physics_agent.config.validator")
    previous_propagate = target.propagate
    target.addHandler(caplog.handler)
    target.propagate = False
    try:
        with caplog.at_level(logging.WARNING, logger=target.name):
            ConfigValidator()._validate_steps(
                {f"https://user:{secret}@config.example.test": {}}
            )
    finally:
        target.removeHandler(caplog.handler)
        target.propagate = previous_propagate

    assert secret not in caplog.text
    assert "<redacted>" in caplog.text


def test_unified_validation_errors_do_not_echo_invalid_credential_values(
    tmp_path: Path,
) -> None:
    secret = "physics-validator-secret-713"
    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        UnifiedPipelineConfigTask().run(
            {
                "config_dict": {
                    "project": {"name": "demo", "working_dir": "runs/demo"},
                    "input": {"usd_path": str(usd_path)},
                    "steps": {
                        "apply_physics": {
                            "enabled": True,
                            "collision_approx": (
                                f"https://user:{secret}@physics.example.test"
                            ),
                        }
                    },
                },
                "only_steps": ["apply_physics"],
            }
        )

    assert secret not in str(exc_info.value)
    assert "unsupported value" in str(exc_info.value)


def test_unified_renderer_validation_error_is_value_free(tmp_path: Path) -> None:
    secret = "physics-renderer-validation-secret-713"
    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        UnifiedPipelineConfigTask().run(
            {
                "config_dict": {
                    "project": {"name": "demo", "working_dir": "runs/demo"},
                    "input": {"usd_path": str(usd_path)},
                    "steps": {
                        "build_dataset_usd": {
                            "enabled": True,
                            "renderer": {
                                "rendering_modes": {"beauty": {}},
                                "cull_style": f"Bearer {secret}",
                            },
                        }
                    },
                },
                "only_steps": ["build_dataset_usd"],
            }
        )

    observable = "".join(traceback.format_exception(exc_info.value))
    assert secret not in observable
    assert str(exc_info.value) == "Invalid renderer configuration"
    assert exc_info.value.__cause__ is None


def test_unified_pipeline_config_task_builds_autowired_step_configs(tmp_path: Path):
    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n")
    reference = tmp_path / "ref.png"
    reference.write_bytes(b"ref")

    config = {
        "project": {"name": "demo", "working_dir": "runs/demo"},
        "input": {
            "usd_path": str(usd_path),
            "reference_images": [str(reference)],
        },
        "steps": {
            "build_dataset_usd": {"enabled": True},
            "identify_asset": {"enabled": True},
            "build_dataset_prepare_dataset": {"enabled": True},
            "predict": {
                "enabled": True,
                "vlm": {"model": "demo-model"},
            },
            "restore_usd": {"enabled": True},
        },
    }

    context = {
        "config_dict": config,
        "session_id": "session-xyz",
        "only_steps": [
            "build_dataset_usd",
            "identify_asset",
            "build_dataset_prepare_dataset",
            "predict",
            "restore_usd",
        ],
    }

    result = UnifiedPipelineConfigTask().run(context)

    assert result["config"] is not config
    assert "session_id" not in config["project"]
    assert result["project_name"] == "demo"
    assert result["session_id"] == "session-xyz"
    assert result["config"]["project"]["session_id"] == "session-xyz"
    assert result["steps_to_run"] == [
        step_name for step_name in STEP_ORDER if step_name in context["only_steps"]
    ]

    build_dataset_usd = result["step_configs"]["build_dataset_usd"]
    assert build_dataset_usd["usd_path"] == str(usd_path)
    assert _path_endswith(build_dataset_usd["output_dir"], "dataset", "usd")
    assert build_dataset_usd["renderer"]["rgb_rendering_modes"]
    assert "composition" in build_dataset_usd["renderer"]["rgb_rendering_modes"]
    assert (
        build_dataset_usd["renderer"]["rendering_modes"]["composition"][
            "use_original_materials"
        ]
        is True
    )

    identify_asset = result["step_configs"]["identify_asset"]
    assert identify_asset["usd_path"] == str(usd_path)
    assert _path_endswith(identify_asset["output_dir"], "identification")

    prepare_dataset = result["step_configs"]["build_dataset_prepare_dataset"]
    assert _path_endswith(prepare_dataset["usd_dir"], "dataset", "usd")
    assert _path_endswith(prepare_dataset["dataset"], "dataset")
    assert prepare_dataset["models"] == ["."]
    assert prepare_dataset["reference_images"] == [str(reference)]

    predict = result["step_configs"]["predict"]
    assert _path_endswith(predict["dataset"], "dataset", "dataset.jsonl")
    assert _path_endswith(predict["output_dir"], "predictions")
    assert predict["output_key"] == "classification"
    assert predict["vlm"]["model"] == "demo-model"
    assert predict["allow_empty_predictions"] is False

    restore = result["step_configs"]["restore_usd"]
    assert restore["original_usd_path"] == str(usd_path)
    assert _path_endswith(
        restore["output_predictions_path"], "restored_predictions.jsonl"
    )


def test_unified_config_dict_uses_source_path_only_as_anchor(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    source_config_path = config_dir / "pipeline.yaml"
    source_config_path.write_text("# source anchor\n", encoding="utf-8")
    usd_path = config_dir / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    source = {
        "project": {"name": "demo", "working_dir": "work"},
        "input": {"usd_path": "asset.usda"},
        "steps": {"build_dataset_usd": {"enabled": True}},
    }

    result = UnifiedPipelineConfigTask().run(
        {"config_dict": source, "config_path": source_config_path}
    )

    assert result["path_resolver"].input_usd == usd_path.resolve()
    assert result["config_path"] == source_config_path
    assert result["config"] is not source
    assert source["project"] == {"name": "demo", "working_dir": "work"}


@pytest.mark.parametrize(
    ("input_suffix", "expected_suffix"),
    [
        (".usd", ".usd"),
        (".usda", ".usda"),
        (".usdc", ".usdc"),
        (".usdz", ".usda"),
    ],
)
def test_unified_pipeline_config_task_derives_apply_output_suffix(
    tmp_path: Path,
    input_suffix: str,
    expected_suffix: str,
):
    usd_path = tmp_path / f"asset{input_suffix}"
    usd_path.write_bytes(b"fake-usd")

    result = UnifiedPipelineConfigTask().run(
        {
            "config_dict": {
                "project": {"name": "demo", "working_dir": "runs/demo"},
                "input": {"usd_path": str(usd_path)},
                "steps": {"apply_physics": {"enabled": True}},
            },
            "only_steps": ["apply_physics"],
        }
    )

    apply_physics = result["step_configs"]["apply_physics"]
    assert _path_endswith(
        apply_physics["output_usd_path"],
        "runs",
        "demo",
        "physics",
        f"asset_physics{expected_suffix}",
    )


def test_unified_pipeline_config_task_respects_explicit_apply_output_path(
    tmp_path: Path,
):
    usd_path = tmp_path / "asset.usdz"
    usd_path.write_bytes(b"fake-usd")
    output_path = tmp_path / "custom_physics.usdz"

    result = UnifiedPipelineConfigTask().run(
        {
            "config_dict": {
                "project": {"name": "demo", "working_dir": "runs/demo"},
                "input": {"usd_path": str(usd_path)},
                "steps": {
                    "apply_physics": {
                        "enabled": True,
                        "output_usd_path": str(output_path),
                    }
                },
            },
            "only_steps": ["apply_physics"],
        }
    )

    assert result["step_configs"]["apply_physics"]["output_usd_path"] == str(
        output_path
    )


def test_unified_pipeline_config_task_nests_optimize_options(tmp_path: Path):
    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n")

    result = UnifiedPipelineConfigTask().run(
        {
            "config_dict": {
                "project": {"name": "demo", "working_dir": "runs/demo"},
                "input": {"usd_path": str(usd_path)},
                "steps": {
                    "optimize_usd": {
                        "enabled": True,
                        "backend": "local",
                        "flatten_prototypes": False,
                        "scene_optimizer_settings": {
                            "enable_deinstance": True,
                            "enable_split_meshes": False,
                            "enable_deduplicate": False,
                        },
                    }
                },
            },
            "only_steps": ["optimize_usd"],
        }
    )

    optimize = result["step_configs"]["optimize_usd"]
    optimization_config = optimize["optimization_config"]
    settings = optimization_config["scene_optimizer_settings"]

    assert "backend" not in optimize
    assert optimization_config["backend"] == "local"
    assert optimization_config["flatten_prototypes"] is False
    assert settings["enable_deinstance"] is True
    assert settings["enable_split_meshes"] is False
    assert settings["enable_deduplicate"] is False


def test_unified_pipeline_config_task_merges_nested_optimize_options(tmp_path: Path):
    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n")

    result = UnifiedPipelineConfigTask().run(
        {
            "config_dict": {
                "project": {"name": "demo", "working_dir": "runs/demo"},
                "input": {"usd_path": str(usd_path)},
                "steps": {
                    "optimize_usd": {
                        "enabled": True,
                        "optimization_config": {
                            "scene_optimizer_settings": {
                                "generate_report": False,
                                "deinstance": {"prim_paths": ["/KeepMe"]},
                            }
                        },
                        "scene_optimizer_settings": {
                            "enable_deinstance": True,
                            "enable_split_meshes": True,
                            "deinstance": {"prim_paths": ["/OverrideMe"]},
                        },
                    }
                },
            },
            "only_steps": ["optimize_usd"],
        }
    )

    settings = result["step_configs"]["optimize_usd"]["optimization_config"][
        "scene_optimizer_settings"
    ]

    assert settings["generate_report"] is False
    assert settings["enable_deinstance"] is True
    assert settings["enable_split_meshes"] is True
    assert settings["deinstance"]["prim_paths"] == ["/OverrideMe"]


def test_unified_pipeline_config_task_rejects_empty_enabled_steps(tmp_path: Path):
    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n")

    with pytest.raises(ValueError, match="No steps enabled"):
        UnifiedPipelineConfigTask().run(
            {
                "config_dict": {
                    "project": {"name": "demo"},
                    "input": {"usd_path": str(usd_path)},
                    "steps": {},
                }
            }
        )
