# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for physics_agent.api.defaults service defaults."""

import importlib

import pytest
from world_understanding.utils.credentials import ensure_no_inline_secrets

from physics_agent.api import defaults


def test_default_render_backend_is_remote():
    """The public staging default should target the remote renderer service."""
    assert defaults.DEFAULT_RENDER_BACKEND == "remote"


def test_remote_default_pipeline_uses_remote_rendering_tuning():
    """Remote rendering defaults should use the HTTP-renderer worker tuning."""
    config = defaults.build_default_pipeline_config(
        session_id="session-123",
        usd_path="/tmp/scene.usd",
        working_dir="/tmp/work",
    )

    assert config["steps"]["identify_asset"]["renderer"]["backend"] == "remote"
    assert config["steps"]["build_dataset_usd"]["renderer"]["backend"] == "remote"
    assert config["steps"]["build_dataset_usd"]["batch_size"] == 4
    assert config["steps"]["build_dataset_usd"]["num_workers"] == 32
    assert config["steps"]["optimize_usd"] == {"enabled": False}
    assert config["steps"]["restore_usd"] == {"enabled": False}
    assert config["steps"]["predict"]["allow_empty_predictions"] is False
    assert config["steps"]["apply_physics"]["mass_scale_policy"] == "skip_mass"
    assert config["steps"]["apply_physics"]["allow_empty_predictions"] is False


def test_default_pipeline_accepts_mock_rendering_backend() -> None:
    config = defaults.build_default_pipeline_config(
        session_id="session-123",
        usd_path="/tmp/scene.usd",
        working_dir="/tmp/work",
        render_backend="mock",
    )

    assert config["steps"]["identify_asset"]["renderer"]["backend"] == "mock"
    assert config["steps"]["build_dataset_usd"]["renderer"]["backend"] == "mock"


def test_default_pipeline_rejects_unknown_rendering_backend() -> None:
    with pytest.raises(ValueError, match="Unknown rendering backend: typo"):
        defaults.build_default_pipeline_config(
            session_id="session-123",
            usd_path="/tmp/scene.usd",
            working_dir="/tmp/work",
            render_backend="typo",
        )


def test_default_pipeline_can_enable_deinstance_optimizer_path():
    """Service defaults should opt into deinstance + restore as a pair."""
    config = defaults.build_default_pipeline_config(
        session_id="session-123",
        usd_path="/tmp/scene.usd",
        working_dir="/tmp/work",
        optimize_usd=True,
    )

    optimize = config["steps"]["optimize_usd"]
    assert optimize["enabled"] is True
    assert optimize["backend"] == "local"
    assert optimize["flatten_prototypes"] is False
    assert optimize["scene_optimizer_settings"] == {
        "enable_deinstance": True,
        "enable_split_meshes": False,
        "enable_deduplicate": False,
        "generate_report": True,
        "capture_stats": True,
    }
    assert config["steps"]["restore_usd"] == {"enabled": True}


def test_default_pipeline_preserves_optimizer_operation_flags():
    """The service form flags should flow directly into Scene Optimizer settings."""
    config = defaults.build_default_pipeline_config(
        session_id="session-123",
        usd_path="/tmp/scene.usd",
        working_dir="/tmp/work",
        optimize_usd=True,
        enable_deinstance=False,
        enable_split=True,
        enable_deduplicate=True,
    )

    settings = config["steps"]["optimize_usd"]["scene_optimizer_settings"]
    assert settings["enable_deinstance"] is False
    assert settings["enable_split_meshes"] is True
    assert settings["enable_deduplicate"] is True


def test_vlm_model_can_be_overridden_from_env(monkeypatch):
    """Service env overrides should flow into generated pipeline configs."""
    monkeypatch.setenv("PA_VLM_MODEL", "nvidia/cosmos-reason2-8b")

    reloaded_defaults = importlib.reload(defaults)
    try:
        config = reloaded_defaults.build_default_pipeline_config(
            session_id="session-123",
            usd_path="/tmp/scene.usd",
            working_dir="/tmp/work",
        )

        assert reloaded_defaults.DEFAULT_VLM_MODEL == "nvidia/cosmos-reason2-8b"
        assert config["steps"]["predict"]["vlm"]["model"] == "nvidia/cosmos-reason2-8b"
        assert (
            reloaded_defaults.IDENTIFY_ASSET_DEFAULTS["vlm"]["model"]
            == "nvidia/cosmos-reason2-8b"
        )
    finally:
        monkeypatch.delenv("PA_VLM_MODEL", raising=False)
        importlib.reload(reloaded_defaults)


def test_vlm_openai_base_url_can_be_overridden_from_env(monkeypatch):
    resolved_secret = "physics-default-runtime-only-key"
    monkeypatch.setenv("PA_VLM_BACKEND", "openai")
    monkeypatch.setenv("PA_VLM_MODEL", "my-custom-vlm")
    monkeypatch.setenv("PA_VLM_BASE_URL", "https://api.openai-compatible.example/v1")
    monkeypatch.setenv("PA_VLM_API_KEY_ENV", "OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", resolved_secret)

    reloaded_defaults = importlib.reload(defaults)
    try:
        config = reloaded_defaults.build_default_pipeline_config(
            session_id="session-123",
            usd_path="/tmp/scene.usd",
            working_dir="/tmp/work",
        )

        vlm_config = config["steps"]["predict"]["vlm"]
        assert vlm_config["backend"] == "openai"
        assert vlm_config["model"] == "my-custom-vlm"
        assert vlm_config["base_url"] == "https://api.openai-compatible.example/v1"
        assert vlm_config["api_key_env"] == "${OPENAI_API_KEY}"
        ensure_no_inline_secrets(config, context="physics default publication")
        assert resolved_secret not in repr(config)
    finally:
        monkeypatch.delenv("PA_VLM_BACKEND", raising=False)
        monkeypatch.delenv("PA_VLM_MODEL", raising=False)
        monkeypatch.delenv("PA_VLM_BASE_URL", raising=False)
        monkeypatch.delenv("PA_VLM_API_KEY_ENV", raising=False)
        importlib.reload(reloaded_defaults)


def test_vlm_temperature_can_be_overridden_from_env(monkeypatch):
    """All generated Physics VLM defaults should honor PA_VLM_TEMPERATURE."""
    monkeypatch.setenv("PA_VLM_BACKEND", "openai")
    monkeypatch.setenv("PA_VLM_MODEL", "gpt-5.5")
    monkeypatch.setenv("PA_VLM_TEMPERATURE", "1.0")

    reloaded_defaults = importlib.reload(defaults)
    try:
        config = reloaded_defaults.build_default_pipeline_config(
            session_id="session-123",
            usd_path="/tmp/scene.usd",
            working_dir="/tmp/work",
        )

        assert reloaded_defaults.DEFAULT_VLM_TEMPERATURE == 1.0
        assert reloaded_defaults.IDENTIFY_ASSET_DEFAULTS["vlm"] == {
            "backend": "openai",
            "model": "gpt-5.5",
            "temperature": 1.0,
            "max_tokens": 4096,
        }
        assert config["steps"]["predict"]["vlm"] == {
            "backend": "openai",
            "model": "gpt-5.5",
            "temperature": 1.0,
            "max_tokens": 24576,
        }
    finally:
        monkeypatch.delenv("PA_VLM_BACKEND", raising=False)
        monkeypatch.delenv("PA_VLM_MODEL", raising=False)
        monkeypatch.delenv("PA_VLM_TEMPERATURE", raising=False)
        importlib.reload(reloaded_defaults)


def test_vlm_max_tokens_override_reaches_default_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PA_VLM_MAX_TOKENS", "8192")

    reloaded_defaults = importlib.reload(defaults)
    try:
        config = reloaded_defaults.build_default_pipeline_config(
            session_id="session-123",
            usd_path="/tmp/scene.usd",
            working_dir="/tmp/work",
        )

        assert reloaded_defaults.DEFAULT_VLM_MAX_TOKENS == 8192
        assert config["steps"]["predict"]["vlm"]["max_tokens"] == 8192
    finally:
        monkeypatch.delenv("PA_VLM_MAX_TOKENS", raising=False)
        importlib.reload(reloaded_defaults)


def test_malformed_judge_env_defaults_do_not_break_import(monkeypatch):
    monkeypatch.setenv("PA_JUDGE_TEMPERATURE", "high")
    monkeypatch.setenv("PA_JUDGE_MAX_TOKENS", "2k")

    reloaded_defaults = importlib.reload(defaults)
    try:
        assert reloaded_defaults.DEFAULT_JUDGE_TEMPERATURE == 0.0
        assert reloaded_defaults.DEFAULT_JUDGE_MAX_TOKENS == 2048
    finally:
        monkeypatch.delenv("PA_JUDGE_TEMPERATURE", raising=False)
        monkeypatch.delenv("PA_JUDGE_MAX_TOKENS", raising=False)
        importlib.reload(reloaded_defaults)


def test_predict_defaults_do_not_inject_internal_llm_backend():
    """Predict defaults should rely on llm=vlm fallback unless explicitly configured."""
    config = defaults.get_predict_config_with_defaults(
        {
            "dataset": "/tmp/dataset.jsonl",
            "vlm": {"model": "nvidia/nemotron-nano-12b-v2-vl"},
        }
    )

    assert "llm" not in config
    assert config["vlm"]["backend"] == defaults.DEFAULT_VLM_BACKEND
    assert config["vlm"]["model"] == "nvidia/nemotron-nano-12b-v2-vl"
