# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for USD generated-reference image task provisioning."""

from pathlib import Path
from typing import Any

import pytest

from world_understanding.agentic.usd_tasks.generate_reference_image import (
    GenerateReferenceImageTask,
)


class _FakeImage:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"fake-image")


class _FakeImageGenModel:
    model_name = "fake-image-gen"
    backend_name = "fake"

    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    def generate_with_image_prompt_pairs(self, **kwargs: Any) -> _FakeImage:
        self._captured["generate_kwargs"] = kwargs
        return _FakeImage()


def test_generate_reference_image_uses_gemini_default_and_model_kwargs(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, Any] = {}
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fake_create_image_generation_model(
        backend: str, **kwargs: Any
    ) -> _FakeImageGenModel:
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return _FakeImageGenModel(captured)

    monkeypatch.setattr(
        "world_understanding.agentic.usd_tasks.generate_reference_image."
        "create_image_generation_model",
        fake_create_image_generation_model,
    )

    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"fake-preview")
    output_dir = tmp_path / "generated"

    context = {
        "rendered_preview_paths": [str(preview_path)],
        "image_gen_config": {
            "model": "gemini-3-pro-image-preview",
            "base_url": "http://image-gen.local/v1",
            "timeout": 12,
        },
        "image_gen_prompt": "matte blue plastic",
        "output_dir": str(output_dir),
        "num_images": 1,
    }

    result = GenerateReferenceImageTask().run(context)

    assert captured["backend"] == "gemini"
    assert captured["kwargs"] == {
        "model": "gemini-3-pro-image-preview",
        "base_url": "http://image-gen.local/v1",
        "timeout": 12,
    }
    assert captured["generate_kwargs"]["image_prompt_pairs"] == [
        (
            "This is preview image 1 of a 3D scene rendered from a USD file.",
            str(preview_path),
        )
    ]
    assert result["generated_reference_image_paths"] == [
        str(output_dir / "generated_ref_0.png")
    ]


def test_generate_reference_image_replaces_placeholder_gemini_key_from_env(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, Any] = {}

    def fake_create_image_generation_model(
        backend: str, **kwargs: Any
    ) -> _FakeImageGenModel:
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return _FakeImageGenModel(captured)

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "real-gemini-key")
    monkeypatch.setattr(
        "world_understanding.agentic.usd_tasks.generate_reference_image."
        "create_image_generation_model",
        fake_create_image_generation_model,
    )

    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"fake-preview")

    context = {
        "rendered_preview_paths": [str(preview_path)],
        "image_gen_config": {
            "backend": "gemini",
            "api_key": "YOUR_GOOGLE_API_KEY",
        },
        "image_gen_prompt": "matte blue plastic",
        "output_dir": str(tmp_path / "generated"),
        "num_images": 1,
    }

    GenerateReferenceImageTask().run(context)

    assert captured["backend"] == "gemini"
    assert captured["kwargs"]["api_key"] == "real-gemini-key"


def test_generate_reference_image_uses_explicit_local_openai_dummy_key_before_env(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, Any] = {}

    def fake_create_image_generation_model(
        backend: str, **kwargs: Any
    ) -> _FakeImageGenModel:
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return _FakeImageGenModel(captured)

    monkeypatch.setenv("OPENAI_API_KEY", "real-hosted-openai-key")
    monkeypatch.setattr(
        "world_understanding.agentic.usd_tasks.generate_reference_image."
        "create_image_generation_model",
        fake_create_image_generation_model,
    )

    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"fake-preview")

    context = {
        "rendered_preview_paths": [str(preview_path)],
        "image_gen_config": {
            "backend": "openai",
            "base_url": "http://localhost:8000/v1",
            "api_key": "not-used",
        },
        "image_gen_prompt": "matte blue plastic",
        "output_dir": str(tmp_path / "generated"),
        "num_images": 1,
    }

    GenerateReferenceImageTask().run(context)

    assert captured["backend"] == "openai"
    assert captured["kwargs"]["api_key"] == "not-used"


def test_generate_reference_image_resolves_api_key_env(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, Any] = {}

    def fake_create_image_generation_model(
        backend: str, **kwargs: Any
    ) -> _FakeImageGenModel:
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return _FakeImageGenModel(captured)

    monkeypatch.setenv("IMAGE_GEN_API_KEY", "endpoint-image-key")
    monkeypatch.setattr(
        "world_understanding.agentic.usd_tasks.generate_reference_image."
        "create_image_generation_model",
        fake_create_image_generation_model,
    )

    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"fake-preview")

    context = {
        "rendered_preview_paths": [str(preview_path)],
        "image_gen_config": {
            "backend": "openai",
            "base_url": "https://api.openai-compatible.example/v1",
            "api_key_env": "IMAGE_GEN_API_KEY",
        },
        "image_gen_prompt": "matte blue plastic",
        "output_dir": str(tmp_path / "generated"),
        "num_images": 1,
    }

    GenerateReferenceImageTask().run(context)

    assert captured["backend"] == "openai"
    assert captured["kwargs"]["api_key"] == "endpoint-image-key"


def test_generate_reference_image_forwards_custom_backend_explicit_key(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, Any] = {}

    def fake_create_image_generation_model(
        backend: str, **kwargs: Any
    ) -> _FakeImageGenModel:
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return _FakeImageGenModel(captured)

    monkeypatch.setattr(
        "world_understanding.agentic.usd_tasks.generate_reference_image."
        "create_image_generation_model",
        fake_create_image_generation_model,
    )

    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"fake-preview")

    context = {
        "rendered_preview_paths": [str(preview_path)],
        "image_gen_config": {
            "backend": "internal_image_backend",
            "api_key": "custom-backend-key",
        },
        "image_gen_prompt": "matte blue plastic",
        "output_dir": str(tmp_path / "generated"),
        "num_images": 1,
    }

    GenerateReferenceImageTask().run(context)

    assert captured["backend"] == "internal_image_backend"
    assert captured["kwargs"]["api_key"] == "custom-backend-key"


def test_generate_reference_image_does_not_forward_custom_backend_placeholder(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, Any] = {}

    def fake_create_image_generation_model(
        backend: str, **kwargs: Any
    ) -> _FakeImageGenModel:
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return _FakeImageGenModel(captured)

    monkeypatch.setattr(
        "world_understanding.agentic.usd_tasks.generate_reference_image."
        "create_image_generation_model",
        fake_create_image_generation_model,
    )

    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"fake-preview")

    context = {
        "rendered_preview_paths": [str(preview_path)],
        "image_gen_config": {
            "backend": "internal_image_backend",
            "api_key": "YOUR_API_KEY",
        },
        "image_gen_prompt": "matte blue plastic",
        "output_dir": str(tmp_path / "generated"),
        "num_images": 1,
    }

    GenerateReferenceImageTask().run(context)

    assert captured["backend"] == "internal_image_backend"
    assert "api_key" not in captured["kwargs"]


def test_generate_reference_image_auto_prompt_nim_references_and_chaining(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, Any] = {"generate_calls": []}

    class FakeChainedModel:
        model_name = "nim-image"
        backend_name = "nim"

        def generate_with_image_prompt_pairs(self, **kwargs: Any) -> _FakeImage:
            captured["generate_calls"].append(kwargs)
            return _FakeImage()

    def fake_create_image_generation_model(
        backend: str, **kwargs: Any
    ) -> FakeChainedModel:
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return FakeChainedModel()

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
    monkeypatch.setattr(
        "world_understanding.agentic.usd_tasks.generate_reference_image."
        "create_image_generation_model",
        fake_create_image_generation_model,
    )
    preview_a = tmp_path / "preview-a.png"
    preview_b = tmp_path / "preview-b.png"
    reference = tmp_path / "reference.png"
    for path in (preview_a, preview_b, reference):
        path.write_bytes(b"image")

    result = GenerateReferenceImageTask().run(
        {
            "rendered_preview_paths": [str(preview_a), str(preview_b)],
            "reference_images": [str(reference)],
            "image_gen_config": {
                "backend": "nim",
                "base_url": "http://image-gen-nim:8000/v1",
            },
            "identification": {
                "asset_type": "vehicle",
                "asset_subtype": "forklift",
                "asset_description": "compact warehouse lift",
                "expected_colors": "yellow body, black mast",
            },
            "additional_prompt": "use studio lighting",
            "output_dir": str(tmp_path / "generated"),
            "num_images": 2,
        }
    )

    assert captured["backend"] == "nim"
    assert captured["kwargs"]["api_key"] == "not-used"
    assert len(captured["generate_calls"]) == 2
    first_call = captured["generate_calls"][0]
    second_call = captured["generate_calls"][1]
    assert first_call["image_prompt_pairs"][0] == (
        "This is reference image 1 showing the desired look / style to match.",
        str(reference),
    )
    assert len(first_call["image_prompt_pairs"]) == 3
    assert "forklift" in first_call["final_prompt"]
    assert "use studio lighting" in first_call["final_prompt"]
    assert second_call["image_prompt_pairs"][0][1] == str(reference)
    assert (
        second_call["image_prompt_pairs"][1][1]
        == result["generated_reference_image_paths"][0]
    )
    assert second_call["image_prompt_pairs"][2][1] == str(preview_b)
    assert "EXACT SAME colors" in second_call["final_prompt"]
    assert result["generated_reference_image_paths"] == [
        str(tmp_path / "generated" / "generated_ref_0.png"),
        str(tmp_path / "generated" / "generated_ref_1.png"),
    ]


def test_generate_reference_image_requires_prompt_without_identification(
    tmp_path,
) -> None:
    preview_path = tmp_path / "preview.png"
    preview_path.write_bytes(b"fake-preview")

    with pytest.raises(ValueError, match="image_gen_prompt is required"):
        GenerateReferenceImageTask().run(
            {
                "rendered_preview_paths": [str(preview_path)],
                "identification": {"asset_type": "unknown"},
            }
        )


def test_prompt_from_identification_fallback_variants() -> None:
    assert "a robot" in GenerateReferenceImageTask._prompt_from_identification(
        {"asset_type": "robot", "asset_subtype": "unknown"}
    )
    assert "the object shown" in GenerateReferenceImageTask._prompt_from_identification(
        {"asset_type": "unknown", "asset_subtype": "unknown"}
    )
