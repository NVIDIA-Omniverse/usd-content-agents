# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage tests for USD dataset consolidation."""

import json
from pathlib import Path

import pytest

from world_understanding.agentic.usd_tasks.consolidate_dataset import (
    ConsolidateDatasetTask,
)


class _Listener:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.debugs = []

    def info(self, message):
        self.infos.append(str(message))

    def warning(self, message):
        self.warnings.append(str(message))

    def debug(self, message):
        self.debugs.append(str(message))


def _prim_entry(path="/World/Chair/Seat", *, renders=True):
    entry = {
        "prim_path": path,
        "metadata": {"extent": [0, 1], "material": "/Looks/OldMaterial"},
        "hierarchy": {"parent_path": "/World/Chair", "ancestors": ["World", "Chair"]},
        "display_color": [0.1, 0.2, 0.3],
        "world_bbox": {
            "min": [0, 0, 0],
            "max": [1, 2, 3],
            "center": [0.5, 1, 1.5],
            "size": [1, 2, 3],
        },
        "world_bbox_meters": {"size": [0.1, 0.2, 0.3]},
        "material_bindings": {"resolved": "/RootNode/Materials/Fabric_Blue"},
    }
    if renders:
        entry["renders"] = [
            {
                "path": "usd/renders/World/Chair/Seat/front.png",
                "view": "front",
                "camera": "/World/Camera",
                "render_mode": "prim_only",
            }
        ]
    return entry


def _write_prims(path: Path, entries):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n")
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def test_run_writes_v02_dataset_and_keeps_intermediate(tmp_path):
    task = ConsolidateDatasetTask()
    assert task.name == "ConsolidateDataset"
    assert task.description == "Consolidate intermediate data into v0.2 format"

    output_dir = tmp_path / "dataset"
    usd_dir = output_dir / "usd"
    usd_dir.mkdir(parents=True)
    (usd_dir / "renders" / "nested").mkdir(parents=True)
    (usd_dir / "renders" / "nested" / "front.png").write_bytes(b"png")
    _write_prims(
        usd_dir / "prims.jsonl",
        [
            _prim_entry(),
            _prim_entry("/World/Invalid", renders=False),
        ],
    )
    (usd_dir / "dataset.json").write_text(
        json.dumps({"metadata": {"source_usd": "from-phase-one.usd"}}),
        encoding="utf-8",
    )

    context = {
        "output_dir": str(output_dir),
        "usd_dir": str(usd_dir),
        "system_prompt": "System prompt",
        "task_type": "detection",
        "task_description": "Detect object role",
        "creator_name": "unit-test",
        "dataset_description": "A tiny dataset",
        "classification_steps": [
            {
                "name": "role",
                "prompt": "Choose the role",
                "classes": ["joint", "body"],
            }
        ],
        "prompts": {
            "vlm_user": "Prompt:\n{step_prompt}\n\nContext:\n{context}",
        },
        "vlm_image_prompts": {"prim_only": "Inspect the isolated prim."},
        "reference_images": ["refs/reference.png"],
        "include_prim_path_context": True,
        "include_geometric_context": True,
        "model_number": "model-7",
        "temperature": 0.2,
        "max_tokens": 128,
    }

    result = task.run(context)

    assert result is context
    assert result["num_entries"] == 1
    assert result["renders_flattened"] is False
    assert Path(result["dataset_config_path"]).exists()
    assert Path(result["dataset_entries_path"]).exists()

    config = json.loads(Path(result["dataset_config_path"]).read_text())
    assert config["schema_version"] == "0.2"
    assert config["metadata"]["creator"] == "unit-test"
    assert config["metadata"]["source_usd"] == "from-phase-one.usd"
    assert config["metadata"]["num_entries"] == 2
    assert config["task"]["type"] == "detection"
    assert config["inference"]["prompts"][0]["system_prompt"] == "System prompt"

    entry = json.loads(Path(result["dataset_entries_path"]).read_text().strip())
    assert entry["id"] == "/World/Chair/Seat"
    assert entry["source"]["model_number"] == "model-7"
    assert "Choose the role" in entry["user_prompt"]
    assert "Is this part a 'joint' or 'body'?" in entry["user_prompt"]
    assert "Bounding box dimensions" in entry["user_prompt"]
    assert entry["media"]["images"][0]["path"] == (
        "usd/renders/World/Chair/Seat/front.png"
    )
    assert entry["media"]["images"][0]["metadata"]["vlm_prompt"] == (
        "Inspect the isolated prim."
    )
    assert entry["media"]["reference_images"][0]["path"] == "refs/reference.png"
    assert entry["ground_truth"]["material"] == "Fabric_Blue"
    assert entry["usd_metadata"]["geometry"]["extent"] == [0, 1]
    assert (usd_dir / "prims.jsonl").exists()


def test_run_missing_prims_and_cleanup_intermediate(tmp_path):
    task = ConsolidateDatasetTask()
    missing_dir = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="Phase 1 intermediate file"):
        task.run({"output_dir": str(tmp_path / "out"), "usd_dir": str(missing_dir)})

    output_dir = tmp_path / "cleaned"
    usd_dir = output_dir / "usd"
    usd_dir.mkdir(parents=True)
    _write_prims(usd_dir / "prims.jsonl", [_prim_entry("/World/Cube")])
    for name in ("dataset.json", "vlm_system_prompt.txt", "spec.txt"):
        (usd_dir / name).write_text("{}", encoding="utf-8")
    (usd_dir / "renders" / "empty").mkdir(parents=True)
    (usd_dir / "renders" / "nonempty").mkdir(parents=True)
    (usd_dir / "renders" / "nonempty" / "keep.txt").write_text("x", encoding="utf-8")

    result = task.run(
        {
            "output_dir": str(output_dir),
            "usd_dir": str(usd_dir),
            "usd_path": tmp_path / "source.usd",
            "system_prompt": "System",
            "keep_intermediate": False,
            "user_prompt_template": "Analyze {context}",
        }
    )

    assert result["num_entries"] == 1
    assert not (usd_dir / "prims.jsonl").exists()
    assert not (usd_dir / "dataset.json").exists()
    assert not (usd_dir / "vlm_system_prompt.txt").exists()
    assert not (usd_dir / "spec.txt").exists()
    assert not (usd_dir / "renders" / "empty").exists()
    assert (usd_dir / "renders" / "nonempty").exists()

    config = json.loads(Path(result["dataset_config_path"]).read_text())
    assert config["metadata"]["source_usd"].endswith("source.usd")


def test_dataset_config_variants_and_invalid_task_type():
    task = ConsolidateDatasetTask()

    with pytest.raises(ValueError, match="Invalid task_type"):
        task._create_dataset_config(
            num_entries=1,
            system_prompt="System",
            task_type="bad",
            task_description="Bad",
            creator_name="tester",
            source_usd="",
            context={},
        )

    config = task._create_dataset_config(
        num_entries=2,
        system_prompt="Default system",
        task_type="iterative_classification",
        task_description="Two steps",
        creator_name="tester",
        source_usd="scene.usd",
        context={
            "classification_steps": [
                {"name": "coarse", "classes": ["a"], "system_prompt": "Coarse"},
                {"name": "fine", "classes": ["b"]},
            ],
            "temperature": 0.4,
            "max_tokens": 99,
        },
    )

    assert config.task.type == "iterative_classification"
    assert [prompt.step_name for prompt in config.inference.prompts] == [
        "coarse",
        "fine",
    ]
    assert config.inference.prompts[0].system_prompt == "Coarse"
    assert config.inference.prompts[1].system_prompt == "Default system"


def test_convert_entry_prompt_variants_and_helpers():
    task = ConsolidateDatasetTask()
    prim = _prim_entry("/World/Robot/Arm")

    iterative_entry = task._convert_prim_to_dataset_entry(
        prim_entry=prim,
        vlm_image_prompts={},
        reference_images=[],
        context={
            "task_type": "iterative_classification",
            "classification_steps": [
                {"prompt": "First step", "classes": ["a", "b"]},
                {"prompt": "Second step", "classes": ["c"]},
            ],
            "prompts": {"vlm_user": "Context only:\n{context}"},
            "include_prim_path_context": True,
        },
    )
    assert iterative_entry.user_prompt is None
    assert len(iterative_entry.user_prompts) == 2
    assert iterative_entry.user_prompts[0].startswith("First step")
    assert "Context only:" in iterative_entry.user_prompts[0]

    fallback_detection = task._convert_prim_to_dataset_entry(
        prim_entry=prim,
        vlm_image_prompts={},
        reference_images=[],
        context={
            "task_type": "detection",
            "classification_steps": [
                {"prompt": "Pick", "classes": ["hinge", "handle"]}
            ],
        },
    )
    assert fallback_detection.user_prompt == "Is this part a 'hinge' or 'handle'?"

    appended_detection = task._convert_prim_to_dataset_entry(
        prim_entry=prim,
        vlm_image_prompts={},
        reference_images=[],
        context={
            "task_type": "detection",
            "classification_steps": [{"prompt": "Pick", "classes": ["hinge"]}],
            "prompts": {"vlm_user": "Context without explicit step:\n{context}"},
            "include_prim_path_context": True,
        },
    )
    assert appended_detection.user_prompt.startswith("Pick")
    assert "Context without explicit step" in appended_detection.user_prompt

    direct_prompt = task._generate_user_prompt(
        prim,
        {"user_prompts": {"/World/Robot/Arm": "Precomputed prompt"}},
    )
    assert direct_prompt == "Precomputed prompt"

    templated = task._generate_user_prompt(
        prim,
        {
            "user_prompt_template": "Analyze with:\n{context}",
            "include_prim_path_context": True,
        },
    )
    assert "prim_path: /World/Robot/Arm" in templated

    assert task._generate_user_prompt({}, {}) == "Analyze this component."
    assert task._generate_user_prompts_multi_step(
        prim,
        [{"prompt": "Only step", "classes": ["x"]}],
        {"prompts": {"vlm_user": "{step_prompt}"}},
    ) == ["Only step\n\nIs this part a 'x'?"]
    assert task._generate_user_prompts_multi_step(
        prim,
        [{"prompt": "Contextual step", "classes": ["y"]}],
        {
            "prompts": {"vlm_user": "{step_prompt}\nContext:\n{context}"},
            "include_prim_path_context": True,
        },
    )[0].startswith("Contextual step")


def test_prompt_context_formatting_and_small_helpers():
    task = ConsolidateDatasetTask()
    prim = _prim_entry("/World/Robot/Arm")

    context = task._build_prompt_context(
        prim,
        {
            "include_prim_path_context": True,
            "include_geometric_context": True,
        },
    )
    assert context["prim_path"] == "/World/Robot/Arm"
    assert context["parent_path"] == "/World/Chair"
    assert context["ancestors"] == ["World", "Chair"]
    assert context["bounding_box_meters"]["width"] == 0.1
    assert context["extent"] == [0, 1]
    assert context["material_binding"] == "/Looks/OldMaterial"

    fallback_context = task._build_prompt_context(
        {
            "prim_path": "/World/Box",
            "world_bbox": {
                "min": [0],
                "max": [1],
                "center": [1, 2, 3],
                "size": [4, 5, 6],
            },
        },
        {"include_geometric_context": True},
    )
    formatted = task._format_context_string(
        {
            "bounding_box_meters": {"width": 1, "height": 2, "depth": 3},
            "bounding_box": fallback_context["bounding_box"],
            "ancestors": ["A", "B"],
            "other": "value",
        }
    )
    assert "width=1.000m" in formatted
    assert "center: (1.00, 2.00, 3.00)" in formatted
    assert "size: (4.00, 5.00, 6.00)" in formatted
    assert "hierarchy: A -> B" in formatted
    assert "other: value" in formatted

    assert task._format_context_string({}) == "No additional context available."
    assert task._extract_material_name("") is None
    assert task._extract_material_name("plain_name") is None
    assert task._extract_material_name("/Looks/Plastic") == "Plastic"
    assert task._flatten_image_path("renders/A/B.png") == "renders/A/B.png"


def test_render_directory_checks_and_cleanup_error_path(monkeypatch, tmp_path):
    task = ConsolidateDatasetTask()
    listener = _Listener()

    assert task._flatten_renders_directory(tmp_path / "none", listener) is False
    assert "No renders/ directory found" in listener.infos[-1]

    renders_root = tmp_path / "renders-root"
    (renders_root / "renders").mkdir(parents=True)
    assert task._flatten_renders_directory(renders_root, listener) is False
    assert "Renders directory is empty" in listener.infos[-1]

    (renders_root / "renders" / "a.jpg").write_bytes(b"jpg")
    (renders_root / "renders" / "b.png").write_bytes(b"png")
    assert task._flatten_renders_directory(renders_root, listener) is False
    assert "2 images" in listener.infos[-1]

    cleanup_dir = tmp_path / "cleanup"
    cleanup_dir.mkdir()
    (cleanup_dir / "renders").mkdir()
    original_iterdir = Path.iterdir

    def raising_iterdir(path):
        if path == cleanup_dir / "renders":
            raise OSError("cannot list")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", raising_iterdir)
    task._cleanup_intermediate_files(cleanup_dir, listener)
