# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the wu image-gen CLI command credential resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

import world_understanding.cli as cli
from world_understanding.cli import app

runner = CliRunner()


@pytest.fixture
def tmp_output_path(tmp_path: Path) -> Path:
    return tmp_path / "out.png"


def _stub_image_model() -> MagicMock:
    """Return a fake image generation model with a model_name attribute."""
    fake = MagicMock()
    fake.model_name = "fake-model"
    fake.generate.return_value = MagicMock()
    return fake


def test_wu_image_gen_openai_local_base_url_injects_no_auth_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_output_path: Path
) -> None:
    """``wu image-gen --backend openai --base-url http://localhost:8000/v1``
    must keep working without ``OPENAI_API_KEY`` and without ``--api-key``.

    The endpoint-aware credential resolver requires an explicit ``not-used``
    opt-in for local OpenAI-compatible endpoints (it does not silently
    forward a hosted ``OPENAI_API_KEY`` to a local URL anymore). The CLI
    must inject the placeholder so the documented locally-hosted image-gen
    flow does not regress.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    fake = _stub_image_model()
    fake.generate.return_value.save = lambda path: Path(path).write_bytes(b"\x89PNG")

    with patch(
        "world_understanding.functions.models.image_generation_models."
        "create_image_generation_model",
        return_value=fake,
    ) as mock_create:
        result = runner.invoke(
            app,
            [
                "image-gen",
                "test prompt",
                "--backend",
                "openai",
                "--base-url",
                "http://localhost:8000/v1",
                "--output",
                str(tmp_output_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert mock_create.call_count == 1
    backend_arg = mock_create.call_args.args[0]
    kwargs = mock_create.call_args.kwargs
    assert backend_arg == "openai"
    assert kwargs["base_url"] == "http://localhost:8000/v1"
    assert kwargs["api_key"] == "not-used"


def test_wu_image_gen_openai_no_key_no_base_url_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_output_path: Path
) -> None:
    """Without ``OPENAI_API_KEY`` and without ``--base-url``, the command
    still errors with a clear message — the new placeholder injection is
    scoped to the local-base-url path."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    result = runner.invoke(
        app,
        [
            "image-gen",
            "test prompt",
            "--backend",
            "openai",
            "--output",
            str(tmp_output_path),
        ],
    )

    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in result.output


def test_wu_image_gen_openai_custom_base_url_does_not_forward_hosted_key(
    monkeypatch: pytest.MonkeyPatch, tmp_output_path: Path
) -> None:
    """``OPENAI_API_KEY`` must not be promoted as an explicit endpoint key
    when ``--base-url`` points at a non-provider URL. Otherwise the hosted
    key would be forwarded to an arbitrary OpenAI-compatible endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    with patch(
        "world_understanding.functions.models.image_generation_models."
        "create_image_generation_model",
    ) as mock_create:
        result = runner.invoke(
            app,
            [
                "image-gen",
                "test prompt",
                "--backend",
                "openai",
                "--base-url",
                "https://api.openai-compatible.example/v1",
                "--output",
                str(tmp_output_path),
            ],
        )

    assert result.exit_code != 0
    assert mock_create.call_count == 0


def test_wu_image_gen_openai_resolved_key_model_and_conditioning_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tmp_output_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    conditioning = tmp_path / "conditioning.png"
    conditioning.write_bytes(b"\x89PNG")
    fake = _stub_image_model()
    fake.generate.return_value.save = lambda path: Path(path).write_bytes(b"\x89PNG")

    with patch(
        "world_understanding.functions.models.image_generation_models."
        "create_image_generation_model",
        return_value=fake,
    ) as mock_create:
        result = runner.invoke(
            app,
            [
                "image-gen",
                "test prompt",
                "--backend",
                "openai",
                "--model",
                "gpt-image-1",
                "--image",
                str(conditioning),
                "--output",
                str(tmp_output_path),
                "--verbose",
            ],
        )

    assert result.exit_code == 0, result.output
    kwargs = mock_create.call_args.kwargs
    assert kwargs["model"] == "gpt-image-1"
    assert kwargs["api_key"] == "sk-test"
    fake.generate.assert_called_once_with(
        prompt="test prompt",
        images=[str(conditioning)],
    )


def test_wu_image_gen_missing_conditioning_image_errors(
    tmp_path: Path, tmp_output_path: Path
) -> None:
    fake = _stub_image_model()
    with patch(
        "world_understanding.functions.models.image_generation_models."
        "create_image_generation_model",
        return_value=fake,
    ):
        result = runner.invoke(
            app,
            [
                "image-gen",
                "test prompt",
                "--backend",
                "gemini",
                "--image",
                str(tmp_path / "missing.png"),
                "--output",
                str(tmp_output_path),
            ],
        )

    assert result.exit_code != 0
    assert "Conditioning image not found" in result.output


def test_wu_image_gen_nim_local_base_url_injects_no_auth_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_output_path: Path
) -> None:
    """Local NIM image-gen flow must work without ``NVIDIA_API_KEY``."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    fake = _stub_image_model()
    fake.generate.return_value.save = lambda path: Path(path).write_bytes(b"\x89PNG")

    with patch(
        "world_understanding.functions.models.image_generation_models."
        "create_image_generation_model",
        return_value=fake,
    ) as mock_create:
        result = runner.invoke(
            app,
            [
                "image-gen",
                "test prompt",
                "--backend",
                "nim",
                "--base-url",
                "http://localhost:8000/v1",
                "--output",
                str(tmp_output_path),
            ],
        )

    assert result.exit_code == 0, result.output
    kwargs = mock_create.call_args.kwargs
    assert kwargs["api_key"] == "not-used"


def test_wu_image_gen_nim_resolved_key_and_no_key_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_output_path: Path
) -> None:
    fake = _stub_image_model()
    fake.generate.return_value.save = lambda path: Path(path).write_bytes(b"\x89PNG")
    with (
        patch(
            "world_understanding.utils.credentials.get_nim_api_key_for_base_url",
            return_value="nim-key",
        ),
        patch(
            "world_understanding.functions.models.image_generation_models."
            "create_image_generation_model",
            return_value=fake,
        ) as mock_create,
    ):
        result = runner.invoke(
            app,
            [
                "image-gen",
                "test prompt",
                "--backend",
                "nim",
                "--output",
                str(tmp_output_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert mock_create.call_args.kwargs["api_key"] == "nim-key"

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    result = runner.invoke(
        app,
        [
            "image-gen",
            "test prompt",
            "--backend",
            "nim",
            "--output",
            str(tmp_output_path),
        ],
    )
    assert result.exit_code != 0
    assert "NVIDIA_API_KEY" in result.output


def test_wu_image_gen_nim_custom_base_url_does_not_forward_hosted_key(
    monkeypatch: pytest.MonkeyPatch, tmp_output_path: Path
) -> None:
    """``NVIDIA_API_KEY`` must not be promoted as an explicit endpoint key
    when ``--base-url`` points at a non-NVIDIA NIM URL."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-real-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    with patch(
        "world_understanding.functions.models.image_generation_models."
        "create_image_generation_model",
    ) as mock_create:
        result = runner.invoke(
            app,
            [
                "image-gen",
                "test prompt",
                "--backend",
                "nim",
                "--base-url",
                "https://nim.example.com/v1",
                "--output",
                str(tmp_output_path),
            ],
        )

    assert result.exit_code != 0
    assert mock_create.call_count == 0


def test_wu_image_gen_plugin_backend_uses_registered_credential_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_output_path: Path
) -> None:
    from world_understanding.functions.models.backends import registry
    from world_understanding.utils import credentials

    backend = "test-image-provider"
    monkeypatch.setattr(registry, "_image_gen_backends", {backend: object()})
    monkeypatch.setattr(
        registry,
        "_image_gen_backend_requires_api_key",
        {backend: True},
    )
    monkeypatch.setattr(
        credentials,
        "API_KEY_ENV_VAR_MAP",
        {backend: ("TEST_IMAGE_PROVIDER_KEY",)},
    )
    monkeypatch.delenv("TEST_IMAGE_PROVIDER_KEY", raising=False)

    with pytest.raises(typer.Exit):
        cli.image_gen(
            "test prompt",
            output=str(tmp_output_path),
            images=None,
            backend=backend,
            model=None,
            base_url=None,
            verbose=False,
        )

    monkeypatch.setenv("TEST_IMAGE_PROVIDER_KEY", "provider-key")
    fake = _stub_image_model()
    fake.generate.return_value.save = lambda path: Path(path).write_bytes(b"\x89PNG")
    with patch(
        "world_understanding.functions.models.image_generation_models."
        "create_image_generation_model",
        return_value=fake,
    ) as mock_create:
        cli.image_gen(
            "test prompt",
            output=str(tmp_output_path),
            images=None,
            backend=backend,
            model=None,
            base_url=None,
            verbose=False,
        )

    assert mock_create.call_args.kwargs["api_key"] == "provider-key"

    registry._image_gen_backend_requires_api_key[backend] = False
    with patch(
        "world_understanding.functions.models.image_generation_models."
        "create_image_generation_model",
        return_value=fake,
    ) as mock_create:
        cli.image_gen(
            "test prompt",
            output=str(tmp_output_path),
            images=None,
            backend=backend,
            model=None,
            base_url=None,
            verbose=False,
        )

    assert "api_key" not in mock_create.call_args.kwargs
