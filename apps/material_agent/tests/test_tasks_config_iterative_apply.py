# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material_agent.tasks.config_iterative_apply."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

import material_agent.tasks.config_iterative_apply as iterative_mod
from material_agent.materials import FALLBACK_MATERIAL_NAME
from material_agent.tasks.config_iterative_apply import IterativeApplyConfigTask


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def _patch_listener(monkeypatch: pytest.MonkeyPatch) -> Mock:
    listener = Mock()
    monkeypatch.setattr(
        iterative_mod,
        "get_listener",
        lambda context, logger_name=None: listener,
    )
    return listener


class TestIterativeApplyConfigTask:
    def test_run_validates_config_path_and_file(self, tmp_path: Path) -> None:
        task = IterativeApplyConfigTask()

        with pytest.raises(ValueError, match="config_path"):
            task.run({})

        with pytest.raises(FileNotFoundError):
            task.run({"config_path": str(tmp_path / "missing.yaml")})

        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            task.run({"config_path": str(empty)})

    def test_run_with_inline_materials_and_vlm_judge(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_listener(monkeypatch)
        config_path = _write_yaml(
            tmp_path / "iterative.yaml",
            {
                "input_usd_path": "input.usd",
                "output_usd_path": "output.usd",
                "dataset": "dataset.jsonl",
                "iteration": {
                    "max_iterations": 7,
                    "save_intermediate": False,
                    "intermediate_dir": "iters",
                },
                "predict": {
                    "vlm": {"model": "custom-vlm"},
                    "llm": {"model": "custom-llm"},
                    "max_workers": 8,
                    "prediction_batch_size": 3,
                    "allow_empty_predictions": True,
                    "system_prompt": "Use the strict prompt",
                    "report": {
                        "image_max_size": 512,
                        "image_format": "jpeg",
                        "image_quality": 75,
                    },
                },
                "apply": {
                    "layer_only": True,
                    "flatten_output": False,
                    "aws_profile": "dev",
                    "usd_search": {"region": "us-east-2"},
                    "materials_mapping": {"Steel": "/Looks/Steel"},
                    "allow_empty_predictions": True,
                    "fail_on_unknown_material": True,
                },
                "render": {"enabled": True, "backend": "remote"},
                "judge": {
                    "vlm": {"model": "judge-vlm"},
                    "reference_images": ["ref.png"],
                },
            },
        )

        context = IterativeApplyConfigTask().run({"config_path": str(config_path)})

        assert context["input_usd_path"] == str((tmp_path / "input.usd").resolve())
        assert context["output_usd_path"] == str((tmp_path / "output.usd").resolve())
        assert context["final_output_usd_path"] == str(
            (tmp_path / "output.usd").resolve()
        )
        assert context["dataset_path"] == str((tmp_path / "dataset.jsonl").resolve())
        assert context["max_iterations"] == 7
        assert context["save_intermediate"] is False
        assert context["intermediate_output_dir"] == str((tmp_path / "iters").resolve())
        assert context["iterations_dir"] == str((tmp_path / "iters").resolve())
        assert context["vlm_config"]["model"] == "custom-vlm"
        assert context["vlm_config"]["backend"]  # default injected
        assert context["llm_config"]["model"] == "custom-llm"
        assert context["max_workers"] == 8
        assert context["prediction_batch_size"] == 3
        assert context["allow_empty_predictions"] is True
        assert context["system_prompt"] == "Use the strict prompt"
        assert context["config"]["system_prompt"] == "Use the strict prompt"
        assert context["report_image_max_size"] == 512
        assert context["report_image_format"] == "jpeg"
        assert context["report_image_quality"] == 75
        assert context["layer_only"] is True
        assert context["flatten_output"] is False
        assert context["apply_allow_empty_predictions"] is True
        assert context["apply_fail_on_unknown_material"] is True
        assert context["aws_profile"] == "dev"
        assert context["usd_search_config"] == {"region": "us-east-2"}
        assert context["materials_mapping"] == {"Steel": "/Looks/Steel"}
        assert context["render_enabled"] is True
        assert context["render_config"] == {"enabled": True, "backend": "remote"}
        assert context["judge_config"]["vlm"]["model"] == "judge-vlm"
        assert context["reference_images"] == [str((tmp_path / "ref.png").resolve())]
        assert context["config"]["vlm"]["model"] == "custom-vlm"
        assert context["config"]["llm"]["model"] == "custom-llm"
        assert context["config"]["vlm_judge"]["model"] == "judge-vlm"
        assert "llm_judge" not in context["config"]

    def test_run_validates_allow_empty_predictions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_listener(monkeypatch)
        config_path = _write_yaml(
            tmp_path / "iterative.yaml",
            {
                "input_usd_path": "input.usd",
                "output_usd_path": "output.usd",
                "dataset": "dataset.jsonl",
                "predict": {"allow_empty_predictions": "yes"},
            },
        )

        with pytest.raises(ValueError, match="allow_empty_predictions"):
            IterativeApplyConfigTask().run({"config_path": str(config_path)})

    def test_run_validates_apply_fail_on_unknown_material(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_listener(monkeypatch)
        config_path = _write_yaml(
            tmp_path / "iterative.yaml",
            {
                "input_usd_path": "input.usd",
                "output_usd_path": "output.usd",
                "dataset": "dataset.jsonl",
                "apply": {"fail_on_unknown_material": "yes"},
            },
        )

        with pytest.raises(ValueError, match="fail_on_unknown_material"):
            IterativeApplyConfigTask().run({"config_path": str(config_path)})

    def test_run_loads_prompt_file_and_external_materials_yaml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        listener = _patch_listener(monkeypatch)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Prompt from file")

        materials_dir = tmp_path / "materials"
        materials_dir.mkdir()
        materials_yaml = _write_yaml(
            materials_dir / "materials.yaml",
            {
                "library_path": "library.usd",
                "entries": [
                    {"name": "Steel", "binding": "/Looks/Steel"},
                    {"name": "Plastic", "binding": "/Looks/Plastic"},
                    {"name": "Ignored", "binding": ""},
                ],
            },
        )

        config_path = _write_yaml(
            tmp_path / "iterative.yaml",
            {
                "input_usd_path": "input.usd",
                "output_usd_path": "output.usd",
                "dataset": "dataset.jsonl",
                "max_iterations": 4,
                "iterations_dir": "legacy-iters",
                "predict": {"system_prompt_file": str(prompt_file)},
                "materials": {"path": str(materials_yaml.relative_to(tmp_path))},
                "judge": {"backend": "nim", "model": "judge-llm"},
            },
        )

        context = IterativeApplyConfigTask().run({"config_path": str(config_path)})

        assert context["max_iterations"] == 4
        assert context["save_intermediate"] is True
        assert context["intermediate_output_dir"] == str(
            (tmp_path / "legacy-iters").resolve()
        )
        assert context["iterations_dir"] == str((tmp_path / "legacy-iters").resolve())
        assert context["system_prompt"] == "Prompt from file"
        assert context["config"]["system_prompt"] == "Prompt from file"
        assert context["materials_mapping"]["material_library_path"] == str(
            (materials_dir / "library.usd").resolve()
        )
        assert context["materials_mapping"]["Steel"] == "/Looks/Steel"
        assert context["materials_mapping"]["Plastic"] == "/Looks/Plastic"
        assert "Ignored" not in context["materials_mapping"]
        assert context["config"]["llm_judge"]["model"] == "judge-llm"
        assert "vlm_judge" not in context["config"]
        listener.info.assert_any_call(f"Loaded system prompt from: {prompt_file}")

    def test_run_warns_for_missing_prompt_and_materials_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        listener = _patch_listener(monkeypatch)
        config_path = _write_yaml(
            tmp_path / "iterative.yaml",
            {
                "input_usd_path": "input.usd",
                "output_usd_path": "output.usd",
                "dataset": "dataset.jsonl",
                "predict": {"system_prompt_file": "missing.txt"},
                "materials": {"path": "missing-materials.yaml"},
            },
        )

        context = IterativeApplyConfigTask().run({"config_path": str(config_path)})

        assert context["system_prompt"] is None
        assert context["materials_mapping"] == {}
        assert context["render_enabled"] is False
        assert context["reference_images"] == []
        assert "llm_judge" in context["config"]
        assert "vlm_judge" not in context["config"]
        assert listener.warning.call_count >= 2

    def test_preserves_file_and_dict_anchor_semantics_from_another_cwd(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _patch_listener(monkeypatch)
        config_dir = tmp_path / "source"
        config_dir.mkdir()
        prompt_path = config_dir / "prompts" / "system.txt"
        prompt_path.parent.mkdir()
        prompt_path.write_text("anchored refine prompt", encoding="utf-8")
        absolute_reference = tmp_path / "absolute-reference.png"
        source = {
            "input_usd_path": "assets/input.usda",
            "output_usd_path": "outputs/final.usda",
            "dataset": "dataset/data.jsonl",
            "iteration": {"intermediate_dir": "iterations"},
            "predict": {"system_prompt_file": "prompts/system.txt"},
            "judge": {
                "reference_images": [
                    "references/front.png",
                    str(absolute_reference),
                ]
            },
        }
        config_path = _write_yaml(config_dir / "refine.yaml", source)
        unrelated_cwd = tmp_path / "cwd"
        unrelated_cwd.mkdir()
        monkeypatch.chdir(unrelated_cwd)

        file_context = IterativeApplyConfigTask().run({"config_path": str(config_path)})
        dict_context = IterativeApplyConfigTask().run(
            {
                "config_dict": source,
                "config_path": str(config_path),
            }
        )

        expected = {
            "input_usd_path": str((config_dir / "assets/input.usda").resolve()),
            "output_usd_path": str((config_dir / "outputs/final.usda").resolve()),
            "dataset_path": str((config_dir / "dataset/data.jsonl").resolve()),
            "iterations_dir": str((config_dir / "iterations").resolve()),
            "reference_images": [
                str((config_dir / "references/front.png").resolve()),
                str(absolute_reference),
            ],
        }
        for context in (file_context, dict_context):
            assert context["input_usd_path"] == expected["input_usd_path"]
            assert context["output_usd_path"] == expected["output_usd_path"]
            assert context["final_output_usd_path"] == expected["output_usd_path"]
            assert context["dataset_path"] == expected["dataset_path"]
            assert context["intermediate_output_dir"] == expected["iterations_dir"]
            assert context["iterations_dir"] == expected["iterations_dir"]
            assert context["system_prompt"] == "anchored refine prompt"
            assert context["reference_images"] == expected["reference_images"]
            assert context["config"]["predict"]["system_prompt_file"] == str(
                prompt_path.resolve()
            )

        assert source == {
            "input_usd_path": "assets/input.usda",
            "output_usd_path": "outputs/final.usda",
            "dataset": "dataset/data.jsonl",
            "iteration": {"intermediate_dir": "iterations"},
            "predict": {"system_prompt_file": "prompts/system.txt"},
            "judge": {
                "reference_images": [
                    "references/front.png",
                    str(absolute_reference),
                ]
            },
        }

    def test_load_materials_mapping_inline_and_edge_cases(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        listener = _patch_listener(monkeypatch)
        task = IterativeApplyConfigTask()
        config_path = tmp_path / "iterative.yaml"

        assert task._load_materials_mapping({}, config_path, listener) == {}
        assert (
            task._load_materials_mapping(
                {"materials": {"entries": []}}, config_path, listener
            )
            == {}
        )
        assert (
            task._load_materials_mapping(
                {
                    "materials": {
                        "library_path": None,
                        "entries": [{"name": "Steel", "binding": "/Looks/Steel"}],
                    }
                },
                config_path,
                listener,
            )
            == {}
        )

        mapping = task._load_materials_mapping(
            {
                "materials": {
                    "library_path": "library.usd",
                    "entries": [
                        {"name": "Steel", "binding": "/Looks/Steel"},
                        {"name": "", "binding": "/Looks/Bad"},
                        {"name": "Bad", "binding": ""},
                    ],
                }
            },
            config_path,
            listener,
        )
        assert mapping == {
            "material_library_path": str(tmp_path / "library.usd"),
            "Steel": "/Looks/Steel",
            FALLBACK_MATERIAL_NAME: "/World/Looks/Fallback_Neutral_Gray_Matte_Plastic",
        }
