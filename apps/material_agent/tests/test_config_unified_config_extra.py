# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for unified configuration helper branches."""

from __future__ import annotations

import logging
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from material_agent.materials import FALLBACK_MATERIAL_NAME


class _Resolver:
    def __init__(self, tmp_path: Path) -> None:
        self.base = tmp_path.resolve()
        self.config_dir = self.base / "configs"
        self.working_dir = self.base / "work"
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = "session-1"
        self.input_usd = self.base / "input.usd"
        self.input_usd.touch()
        self.output_usd = self.base / "output" / "output.usd"
        self.output_usd.parent.mkdir(parents=True, exist_ok=True)
        self.layer_only = True
        self.flatten_output = False
        self.prim_path = "/Root/Part"

        refs_dir = self.base / "refs"
        refs_dir.mkdir(exist_ok=True)
        self.reference_images = [refs_dir / "ref.png"]
        self.reference_images[0].touch()
        self.reference_pdfs = [refs_dir / "spec.pdf"]
        self.reference_pdfs[0].touch()

    def _resolve_path(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else (self.base / path).resolve()

    def get_step_output_dir(self, step_name: str) -> Path:
        path = self.working_dir / step_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_usd_dataset_dir(self) -> Path:
        return self.get_step_output_dir("build_dataset_usd")

    def get_vectorstore_dir(self) -> Path:
        return self.get_step_output_dir("build_dataset_pdf_vectorstore")

    def get_dataset_dir(self) -> Path:
        return self.get_step_output_dir("build_dataset_prepare_dataset")

    def get_predictions_dir(self) -> Path:
        return self.get_step_output_dir("predict")

    def get_step_dataset_file(self, step_name: str) -> Path:
        return self.get_step_output_dir(step_name) / "dataset.jsonl"

    def get_step_predictions_file(self, step_name: str = "predict") -> Path:
        return self.get_step_output_dir(step_name) / "predictions.jsonl"


def _materials_data() -> dict[str, Any]:
    return {
        "library_path": "/materials/library.usd",
        "entries": [
            {
                "name": "Steel",
                "binding": "/World/Looks/Steel",
                "description": "Brushed metal",
            },
            {
                "name": "Wood",
                "binding": "/World/Looks/Wood",
            },
        ],
    }


def _load_unified_config():
    try:
        import material_agent.config.unified_config as unified_config
    except ImportError:
        import material_agent.config.unified_config as unified_config

    return unified_config


def test_run_loads_config_from_file_and_injects_session_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    config_file = tmp_path / "config.yaml"
    (tmp_path / "input.usd").touch()
    config_file.write_text(
        """
project:
  name: demo
input:
  usd_path: input.usd
steps:
  predict:
    enabled: true
""".strip(),
        encoding="utf-8",
    )

    task = UnifiedPipelineConfigTask()
    monkeypatch.setattr(task, "_merge_with_defaults", lambda config: config)
    monkeypatch.setattr(task.validator, "validate", lambda config: None)
    monkeypatch.setattr(task, "_parse_materials", lambda *args: None)
    monkeypatch.setattr(task, "_determine_steps", lambda *args: ["predict"])

    def build_step_configs(_steps, config, *_args):
        config["project"]["name"] = "demo"
        return {"predict": {}}

    monkeypatch.setattr(task, "_build_step_configs", build_step_configs)
    monkeypatch.setattr(task, "_log_summary", lambda *args: None)

    result = task.run({"config_path": str(config_file), "session_id": "override-1"})

    assert result["config_path"] == config_file
    assert result["config"]["project"]["session_id"] == "override-1"


def test_run_uses_config_dict_with_original_config_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "input.usd").touch()

    task = UnifiedPipelineConfigTask()
    monkeypatch.setattr(task.validator, "validate", lambda config: None)
    monkeypatch.setattr(task, "_parse_materials", lambda *args: None)
    monkeypatch.setattr(task, "_determine_steps", lambda *args: ["predict"])
    monkeypatch.setattr(task, "_build_step_configs", lambda *args: {"predict": {}})
    monkeypatch.setattr(task, "_log_summary", lambda *args: None)

    config_dict = {
        "project": {"name": "demo"},
        "input": {"usd_path": "input.usd"},
        "steps": {
            "predict": {
                "enabled": True,
                "vlm": {
                    "backend": "nim",
                    "api_key": "ZXCV713SECRETQWER",
                },
            }
        },
    }
    original = deepcopy(config_dict)
    run_context = {
        "config_dict": config_dict,
        "config_path": str(config_dir / "config.yaml"),
    }

    result = task.run(run_context.copy())
    result["config"]["steps"]["predict"]["vlm"]["api_key"] = "mutated"
    second_result = task.run(run_context.copy())

    assert config_dict == original
    assert (
        second_result["config"]["steps"]["predict"]["vlm"]["api_key"]
        == "ZXCV713SECRETQWER"
    )
    assert result["config"] is not second_result["config"]
    assert (
        second_result["path_resolver"].input_usd == (config_dir / "input.usd").resolve()
    )


def test_run_injects_session_id_when_project_section_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    config_file = tmp_path / "config.yaml"
    (tmp_path / "input.usd").touch()
    config_file.write_text(
        """
input:
  usd_path: input.usd
steps:
  predict:
    enabled: true
""".strip(),
        encoding="utf-8",
    )

    task = UnifiedPipelineConfigTask()
    monkeypatch.setattr(task, "_merge_with_defaults", lambda config: config)
    monkeypatch.setattr(task.validator, "validate", lambda config: None)
    monkeypatch.setattr(task, "_parse_materials", lambda *args: None)
    monkeypatch.setattr(task, "_determine_steps", lambda *args: ["predict"])

    def build_step_configs(_steps, config, *_args):
        config["project"]["name"] = "demo"
        return {"predict": {}}

    monkeypatch.setattr(task, "_build_step_configs", build_step_configs)
    monkeypatch.setattr(task, "_log_summary", lambda *args: None)

    result = task.run({"config_path": str(config_file), "session_id": "session-x"})

    assert result["config"]["project"]["session_id"] == "session-x"


def test_run_wraps_yaml_parse_errors(tmp_path: Path) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    sentinel = "ZXCV713MaterialUnifiedAliasSecretQWER"
    config_file = tmp_path / "bad.yaml"
    config_file.write_text(f"vlm:\n  api_key: *{sentinel}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to parse YAML configuration") as exc:
        UnifiedPipelineConfigTask().run({"config_path": str(config_file)})

    observable = "".join(traceback.format_exception(exc.value))
    assert sentinel not in str(exc.value)
    assert sentinel not in observable
    assert exc.value.__cause__ is None


def test_merge_with_defaults_handles_external_materials_and_missing_steps() -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()

    merged = task._merge_with_defaults({"materials": {"path": "materials.yaml"}})

    assert merged["materials"]["path"] == "materials.yaml"
    assert merged["materials"]["library_path"] is None
    assert "entries" not in merged["materials"]
    assert merged["steps"] == {}


def test_parse_materials_handles_none_inline_and_external_sources(
    tmp_path: Path,
) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)

    assert task._parse_materials({}, resolver) is None

    inline = task._parse_materials(
        {
            "materials": {
                "library_path": "libs/materials.usd",
                "entries": [{"name": "Steel", "binding": "/Steel"}],
                "simready": {"library_id": "inline-simready"},
            }
        },
        resolver,
    )
    assert inline["library_path"] == str(
        (tmp_path / "libs" / "materials.usd").resolve()
    )
    assert inline["simready"] == {"library_id": "inline-simready"}

    materials_file = tmp_path / "materials.yaml"
    materials_file.write_text(
        """
library_path: library/materials.usd
entries:
  - name: Steel
    binding: /World/Looks/Steel
""".strip(),
        encoding="utf-8",
    )
    external = task._parse_materials(
        {"materials": {"path": "materials.yaml"}}, resolver
    )
    assert external["library_path"] == str(
        (tmp_path / "library" / "materials.usd").resolve()
    )
    assert external["entries"][0]["name"] == "Steel"


def test_parse_materials_supports_simready_shortcut(tmp_path: Path) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)
    manifest_path = tmp_path / "manifests" / "simready.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        """
release_tag: test-release
categories:
  Plastic: {}
materials:
  - id: plastic-1
    name: Sim Plastic
    binding: /World/Looks/SimPlastic
    category: Plastic
    source_path: materials/plastic.usd
libraries:
  simready-light:
    material_ids:
      - plastic-1
""".strip(),
        encoding="utf-8",
    )

    materials = task._parse_materials(
        {
            "materials": {
                "simready": {
                    "library_id": "simready-light",
                    "manifest_path": str(manifest_path.relative_to(tmp_path)),
                    "cache_dir": "cache/simready",
                    "allowed_categories": ["Plastic"],
                }
            }
        },
        resolver,
    )

    assert materials["library_path"] == ""
    assert len(materials["entries"]) == 1
    assert {entry["simready_category"] for entry in materials["entries"]} == {"Plastic"}
    assert materials["simready"]["library_id"] == "simready-light"
    assert materials["simready"]["manifest_path"] == str(
        (tmp_path / "manifests" / "simready.json").resolve()
    )
    assert materials["simready"]["cache_dir"] == str(
        (tmp_path / "cache" / "simready").resolve()
    )
    assert task._parse_simready_allowed_categories("Plastic, Metal, ") == {
        "Plastic",
        "Metal",
    }


def test_run_supports_simready_shortcut_after_defaults_merge(tmp_path: Path) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    config_file = tmp_path / "config.yaml"
    (tmp_path / "input.usd").touch()
    config_file.write_text(
        """
project:
  name: demo
input:
  usd_path: input.usd
materials:
  simready: simready-light
steps:
  predict:
    enabled: true
""".strip(),
        encoding="utf-8",
    )

    result = UnifiedPipelineConfigTask().run({"config_path": str(config_file)})

    materials = result["materials_data"]
    assert len(materials["entries"]) == 265
    assert materials["library_path"] == ""
    assert materials["simready"]["library_id"] == "simready-light"
    assert result["step_configs"]["predict"]["dataset"].endswith("dataset.jsonl")


@pytest.mark.parametrize(
    ("filename", "contents", "error_type", "message"),
    [
        ("missing.yaml", None, FileNotFoundError, "Materials file not found"),
        ("broken.yaml", "[", ValueError, "Failed to parse materials file"),
        ("empty.yaml", "", ValueError, "Materials file is empty"),
    ],
)
def test_parse_materials_error_paths(
    tmp_path: Path,
    filename: str,
    contents: str | None,
    error_type: type[Exception],
    message: str,
) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)
    target = tmp_path / filename
    if contents is not None:
        target.write_text(contents, encoding="utf-8")

    with pytest.raises(error_type, match=message):
        task._parse_materials({"materials": {"path": filename}}, resolver)


def test_determine_steps_applies_skip_and_only_filters() -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    config = {
        "steps": {
            "build_dataset_usd": {"enabled": True},
            "predict": {"enabled": True},
            "apply": {"enabled": True},
        }
    }

    steps = task._determine_steps(
        config,
        {"skip_steps": ["build_dataset_usd"], "only_steps": ["predict"]},
    )

    assert steps == ["predict"]


def test_validator_rejects_non_boolean_apply_fail_on_unknown_material() -> None:
    from material_agent.config.validator import ConfigValidator

    with pytest.raises(ValueError, match="apply.fail_on_unknown_material"):
        ConfigValidator().validate_step_requirements(
            "apply",
            {"fail_on_unknown_material": "yes"},
            {"materials": {"entries": [{"name": "Steel", "binding": "/Looks/Steel"}]}},
        )


def test_validator_rejects_non_boolean_refine_apply_fail_on_unknown_material() -> None:
    from material_agent.config.validator import ConfigValidator

    with pytest.raises(ValueError, match="refine.apply.fail_on_unknown_material"):
        ConfigValidator().validate_step_requirements(
            "refine",
            {"apply": {"fail_on_unknown_material": "yes"}},
            {"materials": {"entries": [{"name": "Steel", "binding": "/Looks/Steel"}]}},
        )


def test_validator_allows_generated_material_library_as_material_source() -> None:
    from material_agent.config.validator import ConfigValidator

    ConfigValidator().validate_step_requirements(
        "predict",
        {},
        {"steps": {"generate_material_library": {"enabled": True}}},
    )


def test_validator_allows_create_materials_for_apply_without_materials() -> None:
    from material_agent.config.validator import ConfigValidator

    validator = ConfigValidator()
    config = {"steps": {"create_materials": {"enabled": True}}}
    validator.validate_step_requirements(
        "apply",
        {},
        config,
    )
    validator.validate_step_requirements(
        "refine",
        {},
        config,
    )


def test_validator_rejects_unscheduled_create_materials_source() -> None:
    from material_agent.config.validator import ConfigValidator

    validator = ConfigValidator()
    config = {"steps": {"create_materials": {"enabled": True}}}
    with pytest.raises(ValueError, match="Step 'apply' requires materials"):
        validator.validate_step_requirements("apply", {}, config, ["apply"])

    validator.validate_step_requirements(
        "apply",
        {},
        config,
        ["create_materials", "apply"],
    )
    validator.validate_step_requirements(
        "refine",
        {},
        config,
        ["create_materials", "refine"],
    )


def test_validator_logs_unknown_sections_and_accepts_allowed_path_like_keys(
    caplog,
) -> None:
    from material_agent.config.validator import ConfigValidator

    config = {
        "project": {"name": "demo"},
        "input": {"usd_path": "input.usd"},
        "output": {},
        "unexpected": {},
        "steps": {
            "unknown_step": {"enabled": True},
            "build_dataset_pdf_vectorstore": {
                "enabled": True,
                "source": "docs",
            },
            "generate_material_library": {
                "material_generation_plan_path": "plan.yaml",
            },
            "predict": {
                "include_prim_path_context": True,
            },
        },
    }

    with caplog.at_level("WARNING"):
        ConfigValidator().validate(config)

    assert "Unknown config section" in caplog.text
    assert "No 'materials' section found" in caplog.text
    assert "Unknown step" in caplog.text


def test_validator_rejects_user_supplied_create_materials_source_usd() -> None:
    from material_agent.config.validator import ConfigValidator

    config = {
        "project": {"name": "demo"},
        "input": {"usd_path": "input.usd"},
        "output": {},
        "materials": {"entries": [{"name": "Steel", "binding": "/Looks/Steel"}]},
        "steps": {"create_materials": {"source_usd": "override.usd"}},
    }

    with pytest.raises(ValueError, match="source_usd"):
        ConfigValidator().validate(config)


def test_validator_materials_external_and_simready_branches() -> None:
    from material_agent.config.validator import ConfigValidator

    validator = ConfigValidator()

    validator._validate_materials({"materials": {"path": "materials.yaml"}})
    validator._validate_materials({"materials": {"simready": "simready-light"}})
    validator._validate_materials(
        {"materials": {"simready": {"library_id": "simready-category:metal"}}}
    )

    with pytest.raises(ValueError, match="materials.path"):
        validator._validate_materials({"materials": {"path": 123}})
    with pytest.raises(ValueError, match="both 'materials.path'"):
        validator._validate_materials(
            {"materials": {"path": "materials.yaml", "entries": []}}
        )
    with pytest.raises(ValueError, match="must not be empty"):
        validator._validate_materials({"materials": {"simready": " "}})
    with pytest.raises(ValueError, match="string or mapping"):
        validator._validate_materials({"materials": {"simready": 3}})
    with pytest.raises(ValueError, match="library_id"):
        validator._validate_materials({"materials": {"simready": {}}})


def test_validator_redacts_external_materials_path_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from material_agent.config.validator import ConfigValidator

    secret = "material-validator-path-secret-713"
    caplog.set_level(logging.INFO, logger="material_agent.config.validator")
    ConfigValidator()._validate_materials(
        {
            "materials": {
                "path": f"https://user:{secret}@materials.example.test/library.yaml"
            }
        }
    )

    assert secret not in caplog.text
    assert "<redacted>" in caplog.text


def test_validator_diagnostics_do_not_echo_untrusted_keys_or_names(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from material_agent.config.validator import ConfigValidator

    validator = ConfigValidator()
    secret = "material-validator-value-secret-713"
    credential_uri = f"https://user:{secret}@config.example.test"
    caplog.set_level(logging.WARNING, logger="material_agent.config.validator")

    validator._validate_structure(
        {
            "project": {},
            "input": {},
            "output": {},
            credential_uri: {},
        }
    )
    validator._validate_steps({"steps": {credential_uri: {}}})
    with pytest.raises(ValueError) as name_error:
        validator._validate_materials(
            {"materials": {"entries": [{"name": credential_uri}]}}
        )
    with pytest.raises(ValueError) as key_error:
        validator._validate_no_path_keys(
            "predict",
            {f"credential_path_{credential_uri}": "ordinary"},
        )

    observable = "\n".join(
        (
            caplog.text,
            "".join(traceback.format_exception(name_error.value)),
            "".join(traceback.format_exception(key_error.value)),
        )
    )
    assert secret not in observable
    assert "<redacted>" in observable


def test_simready_catalog_errors_and_logs_do_not_echo_library_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unified_config = _load_unified_config()
    task = unified_config.UnifiedPipelineConfigTask()
    secret = "simready-library-secret-713"
    library_id = f"https://user:{secret}@catalog.example.test"

    def fail_catalog(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise unified_config.SimReadyCatalogError(
            f"Unknown SimReady material library: {library_id}"
        )

    monkeypatch.setattr(unified_config, "load_manifest", lambda path: {})
    monkeypatch.setattr(unified_config, "build_material_entries", fail_catalog)

    with pytest.raises(ValueError) as exc_info:
        task._parse_simready_materials(library_id, _Resolver(tmp_path))

    observable = "".join(traceback.format_exception(exc_info.value))
    assert secret not in observable
    assert str(exc_info.value) == (
        "Unable to load the configured SimReady material library"
    )
    assert exc_info.value.__cause__ is None


def test_validator_step_specific_missing_requirements() -> None:
    from material_agent.config.validator import ConfigValidator

    validator = ConfigValidator()

    with pytest.raises(ValueError, match="missing required field 'source'"):
        validator.validate_step_requirements(
            "build_dataset_pdf_vectorstore",
            {"enabled": True},
            {},
        )

    with pytest.raises(ValueError, match="requires 'llm_judge'"):
        validator.validate_step_requirements(
            "benchmark",
            {"enabled": True},
            {"materials": {"entries": [{"name": "Steel", "binding": "/Looks/Steel"}]}},
        )


def test_build_step_configs_updates_resolver_after_optimize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)
    task.validator = type(
        "_Validator",
        (),
        {"validate_step_requirements": staticmethod(lambda *args: None)},
    )()

    monkeypatch.setattr(task, "_merge_step_config", lambda step_name, user_config: {})

    def fake_autowire(
        step_name, step_config, path_resolver, materials_data, full_config
    ):
        if step_name == "optimize_usd":
            return {
                "output_usd_path": str(
                    path_resolver.get_step_output_dir("optimize_usd")
                    / "optimized_input.usd"
                )
            }
        return {"observed_input_usd": str(path_resolver.input_usd)}

    monkeypatch.setattr(task, "_autowire_paths", fake_autowire)

    built = task._build_step_configs(
        ["optimize_usd", "build_dataset_usd"],
        {"steps": {}},
        resolver,
        None,
    )

    assert built["build_dataset_usd"]["observed_input_usd"].endswith(
        "optimized_input.usd"
    )
    assert resolver.input_usd.name == "optimized_input.usd"


def test_deep_merge_recurses_nested_dictionaries() -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()

    merged = task._deep_merge(
        {"predict": {"vlm": {"model": "base", "temperature": 1.0}}, "enabled": True},
        {"predict": {"vlm": {"model": "override"}}},
    )

    assert merged["predict"]["vlm"] == {"model": "override", "temperature": 1.0}
    assert merged["enabled"] is True


def test_build_step_configs_carries_custom_prepare_prompt_into_predict_only(
    tmp_path: Path,
) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)
    custom_template = "Trusted material policy:\n{materials_list}"
    config = {
        "materials": {"entries": _materials_data()["entries"]},
        "steps": {
            "build_dataset_prepare_dataset": {
                "prompts": {"vlm_system": custom_template}
            },
            "predict": {"enabled": True},
        },
    }

    built = task._build_step_configs(
        ["predict"],
        config,
        resolver,
        _materials_data(),
    )

    assert "system_prompt" not in built["predict"]
    assert (
        built["predict"]["_trusted_prepare_system_prompt_template"] == custom_template
    )


def test_build_step_configs_leaves_default_predict_only_prompt_to_dataset(
    tmp_path: Path,
) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()

    built = task._build_step_configs(
        ["predict"],
        {
            "materials": {"entries": _materials_data()["entries"]},
            "steps": {
                "build_dataset_prepare_dataset": {"enabled": True},
                "predict": {"enabled": True},
            },
        },
        _Resolver(tmp_path),
        _materials_data(),
    )

    assert "system_prompt" not in built["predict"]
    assert "_trusted_prepare_system_prompt_template" not in built["predict"]


def test_build_step_configs_renders_prepare_prompt_for_benchmark_only(
    tmp_path: Path,
) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    custom_template = "Trusted benchmark material policy:\n{materials_list}"

    built = task._build_step_configs(
        ["benchmark"],
        {
            "materials": {"entries": _materials_data()["entries"]},
            "steps": {
                "build_dataset_prepare_dataset": {
                    "prompts": {"vlm_system": custom_template}
                },
                "benchmark": {"enabled": True},
            },
        },
        _Resolver(tmp_path),
        _materials_data(),
    )

    benchmark_prompt = built["benchmark"]["system_prompt"]
    assert "Trusted benchmark material policy:" in benchmark_prompt
    assert '"Steel"' in benchmark_prompt
    assert '"Wood"' in benchmark_prompt
    assert FALLBACK_MATERIAL_NAME in benchmark_prompt
    assert "Brushed metal" not in benchmark_prompt


def test_autowire_validation_and_basic_setup_steps(tmp_path: Path) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)

    validate_input = task._autowire_paths("validate_input", {}, resolver, None, {})
    assert validate_input["input_usd_path"] == str(resolver.input_usd)
    assert validate_input["validation_config"] == {}

    validate_output = task._autowire_paths("validate_output", {}, resolver, None, {})
    assert validate_output["input_usd_path"] == str(resolver.output_usd)
    assert validate_output["validation_config"] == {}

    with pytest.raises(ValueError, match="not supported for validate_output"):
        task._autowire_paths(
            "validate_output", {"on_failure": "fix"}, resolver, None, {}
        )

    optimize = task._autowire_paths("optimize_usd", {}, resolver, None, {})
    assert optimize["input_usd_path"] == str(resolver.input_usd)
    assert optimize["optimization_config"] == {}

    preview = task._autowire_paths("render_preview", {}, resolver, None, {})
    assert preview["usd_path"] == str(resolver.input_usd)

    with pytest.raises(ValueError, match="steps.identify_asset.vlm is required"):
        task._autowire_paths("identify_asset", {}, resolver, None, {})

    identify = task._autowire_paths(
        "identify_asset",
        {"vlm": {"backend": "nim", "model": "qwen/qwen3.5-397b-a17b"}},
        resolver,
        None,
        {},
    )
    assert identify["usd_path"] == str(resolver.input_usd)
    assert identify["output_dir"] == str(resolver.get_step_output_dir("identify_asset"))
    assert identify["vlm_config"] == {
        "backend": "nim",
        "model": "qwen/qwen3.5-397b-a17b",
    }
    assert "vlm" not in identify

    resolver.reference_images = [resolver.reference_images[0]]
    identify_with_refs = task._autowire_paths(
        "identify_asset",
        {"vlm_config": {"backend": "nim", "model": "qwen/qwen3.5-397b-a17b"}},
        resolver,
        None,
        {},
    )
    assert identify_with_refs["reference_images"] == [str(resolver.reference_images[0])]

    reference = task._autowire_paths("generate_reference_image", {}, resolver, None, {})
    assert reference["reference_images"] == [str(resolver.reference_images[0])]

    plan_path = tmp_path / "plans" / "material_generation_plan.yaml"
    plan_path.parent.mkdir()
    plan_path.touch()
    generated = task._autowire_paths(
        "generate_material_library",
        {"material_generation_plan_path": "plans/material_generation_plan.yaml"},
        resolver,
        None,
        {},
    )
    assert generated["input_usd_path"] == str(resolver.input_usd)
    assert generated["output_dir"] == str(
        resolver.get_step_output_dir("generate_material_library")
    )
    assert generated["material_generation_plan_path"] == str(plan_path.resolve())
    assert generated["reference_images"] == [str(resolver.reference_images[0])]

    generated_explicit_refs = task._autowire_paths(
        "generate_material_library",
        {"reference_images": ["refs/ref.png"]},
        resolver,
        None,
        {},
    )
    assert generated_explicit_refs["reference_images"] == [
        str(resolver.reference_images[0].resolve())
    ]


def test_autowire_build_dataset_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unified_config = _load_unified_config()
    UnifiedPipelineConfigTask = unified_config.UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)

    class _RendererConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def get_rendering_modes_config(
            self, rendering_modes_raw: dict[str, Any]
        ) -> dict[str, Any]:
            assert "beauty" in rendering_modes_raw
            return {"beauty": {}, "linear_depth": {}}

    monkeypatch.setattr(unified_config, "RendererConfig", _RendererConfig)

    usd_cfg = task._autowire_paths(
        "build_dataset_usd",
        {"renderer": {"rendering_modes": {"beauty": {}, "linear_depth": {}}}},
        resolver,
        None,
        {},
    )
    assert usd_cfg["prim_filters"]["root_prim"] == resolver.prim_path
    assert usd_cfg["renderer"]["rgb_rendering_modes"] == ["beauty"]
    assert usd_cfg["renderer"]["sensor_rendering_modes"] == ["linear_depth"]

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_cfg = task._autowire_paths(
        "build_dataset_pdf_vectorstore",
        {"source": "docs"},
        resolver,
        None,
        {},
    )
    assert pdf_cfg["source"] == str(docs_dir.resolve())
    assert pdf_cfg["output_dir"] == str(resolver.get_vectorstore_dir())

    prep_cfg = task._autowire_paths(
        "build_dataset_prepare_dataset",
        {"prompts": {}},
        resolver,
        _materials_data(),
        {"steps": {"build_dataset_pdf_vectorstore": {"enabled": True}}},
    )
    assert prep_cfg["usd_dir"] == str(resolver.get_usd_dataset_dir())
    assert prep_cfg["dataset"] == str(resolver.get_dataset_dir())
    assert prep_cfg["models"] == ["."]
    assert prep_cfg["vector_store"].endswith("vector_store")
    assert prep_cfg["reference_images"] == [str(resolver.reference_images[0])]
    assert prep_cfg["reference_pdfs"] == [str(resolver.reference_pdfs[0])]
    assert prep_cfg["materials_list"] == [
        "Steel",
        "Wood",
        FALLBACK_MATERIAL_NAME,
    ]
    assert "Brushed metal" not in prep_cfg["_materials_formatted"]
    assert '"Wood"' in prep_cfg["_materials_formatted"]


def test_autowire_renderer_validation_error_is_value_free(tmp_path: Path) -> None:
    unified_config = _load_unified_config()
    task = unified_config.UnifiedPipelineConfigTask()
    secret = "material-renderer-validation-secret-713"

    with pytest.raises(ValueError) as exc_info:
        task._autowire_paths(
            "build_dataset_usd",
            {
                "renderer": {
                    "rendering_modes": {"beauty": {}},
                    "cull_style": f"Bearer {secret}",
                }
            },
            _Resolver(tmp_path),
            None,
            {},
        )

    observable = "".join(traceback.format_exception(exc_info.value))
    assert secret not in observable
    assert str(exc_info.value) == "Invalid renderer configuration"
    assert exc_info.value.__cause__ is None


def test_autowire_prediction_validation_and_apply_steps(tmp_path: Path) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)
    materials_data = _materials_data()

    predict_cfg = task._autowire_paths("predict", {}, resolver, None, {})
    assert predict_cfg["dataset"].endswith("dataset.jsonl")
    assert predict_cfg["output_dir"] == str(resolver.get_predictions_dir())

    validate_cfg = task._autowire_paths(
        "validate_predictions", {}, resolver, materials_data, {}
    )
    assert validate_cfg["material_names"] == ["Steel", "Wood", FALLBACK_MATERIAL_NAME]

    harmonize_cfg = task._autowire_paths(
        "harmonize_predictions", {}, resolver, materials_data, {}
    )
    assert harmonize_cfg["material_names"] == ["Steel", "Wood", FALLBACK_MATERIAL_NAME]

    create_cfg = task._autowire_paths(
        "create_materials",
        {},
        resolver,
        None,
        {"output": {"shader_target": "usd_preview_surface"}},
    )
    create_output_dir = resolver.get_step_output_dir("create_materials")
    assert create_cfg["source_usd"] == str(resolver.input_usd)
    assert create_cfg["predictions_path"] == str(resolver.get_step_predictions_file())
    assert create_cfg["output_dir"] == str(create_output_dir)
    assert create_cfg["output_predictions_path"] == str(
        create_output_dir / "created_predictions.jsonl"
    )
    assert create_cfg["_config_dir"] == str(resolver.config_dir)
    assert create_cfg["material_profile"] == "usd_preview_surface"
    explicit_create_cfg = task._autowire_paths(
        "create_materials",
        {"material_profile": "openpbr_materialx"},
        resolver,
        None,
        {"output": {"shader_target": "usd_preview_surface"}},
    )
    assert explicit_create_cfg["material_profile"] == "openpbr_materialx"

    apply_cfg = task._autowire_paths("apply", {}, resolver, materials_data, {})
    assert apply_cfg["input_usd_path"] == str(resolver.input_usd)
    assert apply_cfg["output_usd_path"] == str(resolver.output_usd)
    assert apply_cfg["layer_only"] is True
    assert apply_cfg["flatten_output"] is False
    assert apply_cfg["material_profile"] == "auto"
    assert (
        apply_cfg["materials_mapping"]["material_library_path"]
        == "/materials/library.usd"
    )
    assert apply_cfg["materials_mapping"]["Steel"] == "/World/Looks/Steel"
    assert FALLBACK_MATERIAL_NAME in apply_cfg["materials_mapping"]

    openpbr_apply_cfg = task._autowire_paths(
        "apply",
        {},
        resolver,
        materials_data,
        {"output": {"material_profile": "openpbr_materialx"}},
    )
    assert openpbr_apply_cfg["material_profile"] == "openpbr_materialx"


def test_autowire_refine_with_minimal_defaults(tmp_path: Path) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)

    refine_with_refs = task._autowire_paths(
        "refine", {}, resolver, _materials_data(), {}
    )
    assert refine_with_refs["judge"]["reference_images"] == [
        str(resolver.reference_images[0])
    ]

    resolver.reference_images = []

    refine_cfg = task._autowire_paths("refine", {}, resolver, _materials_data(), {})

    assert refine_cfg["dataset"].endswith("dataset.jsonl")
    assert "system_prompt_file" not in refine_cfg["predict"]
    assert "vlm" in refine_cfg["predict"]
    assert refine_cfg["judge"]["backend"]
    assert refine_cfg["llm_judge"] == refine_cfg["judge"]
    assert refine_cfg["apply"]["materials_mapping"]["Wood"] == "/World/Looks/Wood"
    assert FALLBACK_MATERIAL_NAME in refine_cfg["apply"]["materials_mapping"]


def test_autowire_refine_with_custom_predict_and_judge(tmp_path: Path) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)

    refine_cfg = task._autowire_paths(
        "refine",
        {
            "predict": {
                "vlm": {"backend": "custom-vlm"},
                "max_workers": 7,
                "system_prompt_file": "legacy.txt",
            },
            "iteration": {"max_iterations": 5, "save_intermediate": False},
            "judge": {"vlm": {"backend": "judge-vlm"}},
        },
        resolver,
        _materials_data(),
        {},
    )

    assert "system_prompt_file" not in refine_cfg["predict"]
    assert refine_cfg["vlm"] == {"backend": "custom-vlm"}
    assert refine_cfg["max_workers"] == 7
    assert refine_cfg["max_iterations"] == 5
    assert refine_cfg["save_intermediate"] is False
    assert refine_cfg["vlm_judge"] == {"backend": "judge-vlm"}
    assert refine_cfg["judge"]["reference_images"] == [
        str(resolver.reference_images[0])
    ]


def test_autowire_restore_and_render_steps(tmp_path: Path) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)

    restore_cfg = task._autowire_paths("restore_usd", {}, resolver, None, {})
    assert restore_cfg["input_usd_path"] == str(resolver.output_usd)
    assert restore_cfg["output_usd_path"].endswith("restored_output.usd")
    assert restore_cfg["restore_config"] == {}

    render_cfg = task._autowire_paths("render", {}, resolver, None, {})
    assert render_cfg["input_usd_path"] == str(resolver.output_usd)
    assert render_cfg["prim_path"] == resolver.prim_path
    assert render_cfg["output_path"] == str(resolver.output_usd.parent)

    custom_render_cfg = task._autowire_paths(
        "render",
        {"input_usd_path": "custom/input.usd", "output_path": "renders"},
        resolver,
        None,
        {},
    )
    assert custom_render_cfg["input_usd_path"] == str(
        (tmp_path / "custom" / "input.usd").resolve()
    )
    assert custom_render_cfg["output_path"] == str((tmp_path / "renders").resolve())


def test_log_summary_includes_optional_description_and_library(tmp_path: Path) -> None:
    unified_config = _load_unified_config()
    UnifiedPipelineConfigTask = unified_config.UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)

    with patch.object(unified_config.logger, "info") as info:
        task._log_summary(
            {"project": {"name": "demo", "description": "Detailed project"}},
            resolver,
            _materials_data(),
            ["predict", "apply"],
        )

    logged = " ".join(" ".join(map(str, call.args)) for call in info.call_args_list)
    assert "Detailed project" in logged
    assert "/materials/library.usd" in logged
