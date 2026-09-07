# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared agentic image-generation skill."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_script() -> ModuleType:
    path = (
        _repo_root()
        / "agentic/.agents/skills/image-generation/scripts/generate_image.py"
    )
    spec = importlib.util.spec_from_file_location("agentic_generate_image", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_prompt(path: Path) -> None:
    path.write_text("Generate a seamless oxidized copper albedo.\n", encoding="utf-8")


def test_record_companion_writes_common_manifest(tmp_path: Path) -> None:
    module = _load_script()
    prompt = tmp_path / "prompt.txt"
    source = tmp_path / "companion.webp"
    output = tmp_path / "attempt" / "raw.png"
    manifest_path = tmp_path / "attempt" / "image_generation.json"
    _write_prompt(prompt)
    Image.new("RGB", (12, 8), (10, 20, 30)).save(source)

    returncode = module.main(
        [
            "record-companion",
            "--prompt-file",
            str(prompt),
            "--source-image",
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifest_path),
            "--tool-id",
            "codex-imagegen",
            "--model",
            "reported-model",
        ]
    )

    assert returncode == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "agentic-image-generation-result.v1"
    assert manifest["status"] == "completed"
    assert manifest["mode"] == "coding_agent_companion"
    assert manifest["provider"] == {
        "tool_id": "codex-imagegen",
        "backend": None,
        "model": "reported-model",
        "base_url": None,
        "api_key_env": None,
    }
    assert manifest["request"]["prompt_sha256"]
    assert manifest["request"]["conditioning_images"] == []
    assert manifest["output"]["path"] == str(output.resolve())
    assert manifest["output"]["width"] == 12
    assert manifest["output"]["height"] == 8
    assert manifest["output"]["sha256"]


def test_explicit_backend_uses_world_understanding_model_and_records_identity(
    tmp_path: Path,
) -> None:
    module = _load_script()
    prompt = tmp_path / "prompt.txt"
    conditioning = tmp_path / "reference.png"
    output = tmp_path / "raw.png"
    manifest_path = tmp_path / "image_generation.json"
    _write_prompt(prompt)
    Image.new("RGB", (6, 5), (1, 2, 3)).save(conditioning)
    calls: list[tuple[str, list[Path] | None]] = []

    class FakeModel:
        backend_name = "fake-backend"
        model_name = "fake-model"
        supports_image_conditioning = True

        def generate(
            self, request_prompt: str, images: list[Path] | None = None
        ) -> Image.Image:
            calls.append((request_prompt, images))
            return Image.new("RGB", (9, 7), (40, 50, 60))

    module._create_world_understanding_model = lambda args: FakeModel()

    returncode = module.main(
        [
            "backend",
            "--prompt-file",
            str(prompt),
            "--conditioning-image",
            str(conditioning),
            "--output",
            str(output),
            "--manifest",
            str(manifest_path),
            "--backend",
            "configured-backend",
            "--model",
            "requested-model",
        ]
    )

    assert returncode == 0
    assert calls == [
        (
            prompt.read_text(encoding="utf-8"),
            [conditioning.resolve()],
        )
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["mode"] == "world_understanding_backend"
    assert manifest["provider"]["backend"] == "fake-backend"
    assert manifest["provider"]["model"] == "fake-model"
    assert manifest["request"]["conditioning_images"][0]["sha256"]
    assert manifest["output"]["width"] == 9
    assert manifest["output"]["height"] == 7


def test_explicit_backend_failure_is_recorded_without_provider_switch(
    tmp_path: Path,
) -> None:
    module = _load_script()
    prompt = tmp_path / "prompt.txt"
    output = tmp_path / "raw.png"
    manifest_path = tmp_path / "image_generation.json"
    _write_prompt(prompt)

    def fail(_args: object) -> object:
        raise RuntimeError("configured backend unavailable")

    module._create_world_understanding_model = fail

    returncode = module.main(
        [
            "backend",
            "--prompt-file",
            str(prompt),
            "--output",
            str(output),
            "--manifest",
            str(manifest_path),
            "--backend",
            "configured-backend",
        ]
    )

    assert returncode == 1
    assert not output.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["mode"] == "world_understanding_backend"
    assert manifest["provider"]["backend"] == "configured-backend"
    assert manifest["diagnostic"] == {
        "type": "RuntimeError",
        "message": "configured backend unavailable",
    }


def test_backend_failure_redacts_credential_bearing_provider_metadata(
    tmp_path: Path,
) -> None:
    module = _load_script()
    prompt = tmp_path / "prompt.txt"
    output = tmp_path / "raw.png"
    manifest_path = tmp_path / "image_generation.json"
    _write_prompt(prompt)

    returncode = module.main(
        [
            "backend",
            "--prompt-file",
            str(prompt),
            "--output",
            str(output),
            "--manifest",
            str(manifest_path),
            "--backend",
            "configured-backend",
            "--base-url",
            "https://user:secret@host.example/v1",
        ]
    )

    assert returncode == 1
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["provider"]["base_url"] is None
    assert "secret" not in manifest_text


def test_backend_failure_does_not_persist_retrieved_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    prompt = tmp_path / "prompt.txt"
    output = tmp_path / "raw.png"
    manifest_path = tmp_path / "image_generation.json"
    _write_prompt(prompt)
    api_key = "sk-secret-value"
    monkeypatch.setenv("TEST_IMAGE_API_KEY", api_key)

    def fail_after_key_retrieval(args: object) -> object:
        module._validate_base_url(args.base_url)
        retrieved = module._explicit_api_key(args.api_key_env)
        raise RuntimeError(f"provider rejected API key {retrieved}")

    module._create_world_understanding_model = fail_after_key_retrieval

    returncode = module.main(
        [
            "backend",
            "--prompt-file",
            str(prompt),
            "--output",
            str(output),
            "--manifest",
            str(manifest_path),
            "--backend",
            "configured-backend",
            "--base-url",
            "https://host.example/v1",
            "--api-key-env",
            "TEST_IMAGE_API_KEY",
        ]
    )

    assert returncode == 1
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["provider"]["api_key_env"] == "TEST_IMAGE_API_KEY"
    assert api_key not in manifest_text


def test_conditioning_requires_explicit_backend_capability(tmp_path: Path) -> None:
    module = _load_script()
    prompt = tmp_path / "prompt.txt"
    conditioning = tmp_path / "reference.png"
    output = tmp_path / "raw.png"
    manifest_path = tmp_path / "image_generation.json"
    _write_prompt(prompt)
    Image.new("RGB", (2, 2), "white").save(conditioning)

    class UndeclaredModel:
        backend_name = "undeclared"
        model_name = "test"

    module._create_world_understanding_model = lambda args: UndeclaredModel()

    returncode = module.main(
        [
            "backend",
            "--prompt-file",
            str(prompt),
            "--conditioning-image",
            str(conditioning),
            "--output",
            str(output),
            "--manifest",
            str(manifest_path),
            "--backend",
            "configured-backend",
        ]
    )

    assert returncode == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["diagnostic"]["type"] == "ValueError"
    assert "does not support conditioning images" in manifest["diagnostic"]["message"]


def test_backend_exception_after_manifest_write_does_not_escape(tmp_path: Path) -> None:
    module = _load_script()
    prompt = tmp_path / "prompt.txt"
    output = tmp_path / "raw.png"
    manifest_path = tmp_path / "image_generation.json"
    _write_prompt(prompt)

    class PartialModel:
        backend_name = "partial"
        model_name = "test"

        def generate(self, request_prompt: str, images: object = None) -> Image.Image:
            manifest_path.write_text('{"status":"partial"}\n', encoding="utf-8")
            raise RuntimeError("secret backend detail")

    module._create_world_understanding_model = lambda args: PartialModel()

    assert (
        module.main(
            [
                "backend",
                "--prompt-file",
                str(prompt),
                "--output",
                str(output),
                "--manifest",
                str(manifest_path),
                "--backend",
                "configured-backend",
            ]
        )
        == 1
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "status": "partial"
    }


def test_failure_with_missing_inputs_still_writes_manifest(tmp_path: Path) -> None:
    module = _load_script()
    missing_prompt = tmp_path / "missing-prompt.txt"
    manifest_path = tmp_path / "attempt" / "image_generation.json"

    returncode = module.main(
        [
            "record-failure",
            "--prompt-file",
            str(missing_prompt),
            "--manifest",
            str(manifest_path),
            "--mode",
            "coding_agent_companion",
            "--reason",
            "companion generator was unavailable",
        ]
    )

    assert returncode == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["request"]["prompt_sha256"] is None
    assert manifest["request"]["conditioning_images"] is None
    assert manifest["request"]["request_inspection_error"] == "FileNotFoundError"
    assert manifest["diagnostic"]["message"] == ("companion generator was unavailable")


def test_recorded_companion_image_feeds_material_packaging(tmp_path: Path) -> None:
    pytest.importorskip("material_agent")
    from material_agent.material_library_generation import (
        MaterialGenerationPlan,
        MaterialRecipe,
        TextureGenerationSettings,
        build_generated_material_library,
        validate_generated_material_library,
    )

    module = _load_script()
    prompt = tmp_path / "prompt.txt"
    companion_output = tmp_path / "companion.png"
    attempt_dir = tmp_path / "attempt"
    raw_image = attempt_dir / "raw.png"
    manifest_path = attempt_dir / "image_generation.json"
    _write_prompt(prompt)
    Image.new("RGB", (8, 8), (21, 42, 63)).save(companion_output)

    assert (
        module.main(
            [
                "record-companion",
                "--prompt-file",
                str(prompt),
                "--source-image",
                str(companion_output),
                "--output",
                str(raw_image),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    image_result = json.loads(manifest_path.read_text(encoding="utf-8"))
    recipe = MaterialRecipe(
        name="Recorded Copper",
        description="Copper material generated through the shared image skill.",
        appearance_prompt="Seamless oxidized copper.",
    )
    package = build_generated_material_library(
        MaterialGenerationPlan(materials=(recipe,)),
        tmp_path / "package",
        texture_settings=TextureGenerationSettings(
            texture_size=8,
            color_correct_albedo=False,
        ),
        source_albedo_paths={
            recipe.material_id: image_result["output"]["path"],
        },
        material_profile="preview_surface",
    )

    validation = validate_generated_material_library(package.materials_manifest_path)
    assert validation.ok, validation.errors
    with Image.open(package.materials[0].textures.albedo) as packaged_albedo:
        assert packaged_albedo.getpixel((0, 0)) == (21, 42, 63)


def test_consuming_skills_delegate_image_generation() -> None:
    root = _repo_root() / "agentic/.agents/skills"
    image_skill = (root / "image-generation/SKILL.md").read_text(encoding="utf-8")
    material_skill = (root / "material-generation/SKILL.md").read_text(encoding="utf-8")
    mesh_skill = (root / "content-workflow-mesh-segmentation/SKILL.md").read_text(
        encoding="utf-8"
    )
    mesh_adapter = (
        root
        / "content-workflow-mesh-segmentation/scripts/generate_semantic_overlays.py"
    ).read_text(encoding="utf-8")

    assert "name: image-generation" in image_skill
    normalized_image_skill = image_skill.lower()
    assert "coding-agent companion" in normalized_image_skill
    assert "world understanding backend" in normalized_image_skill
    assert "image-generation" in material_skill
    assert "image-generation" in mesh_skill
    assert "image-generation/scripts/generate_image.py" in mesh_adapter
    assert "OpenAICompatibleChatImageGenerationModel" not in mesh_adapter
