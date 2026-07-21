# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for service model routing configuration."""

from __future__ import annotations

from world_understanding.utils.credentials import ensure_no_inline_secrets

from ...service.routers import pipeline_router


def test_predict_model_routing_applies_service_token_limits(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_router.config, "vlm_temperature", 0.2)
    monkeypatch.setattr(pipeline_router.config, "vlm_max_tokens", 512)
    monkeypatch.setattr(pipeline_router.config, "llm_temperature", 0.1)
    monkeypatch.setattr(pipeline_router.config, "llm_max_tokens", 256)

    config = {
        "steps": {
            "predict": {
                "vlm": {"max_tokens": 9999},
                "llm": {"max_tokens": 9999},
            }
        }
    }
    routing = pipeline_router._ModelRouting(
        vlm_backend="nim",
        vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
        vlm_nim_base_url="http://vlm-nim:8000/v1",
        llm_backend="nim",
        llm_model="Qwen/Qwen2.5-VL-7B-Instruct",
        llm_nim_base_url="http://vlm-nim:8000/v1",
        llm_uses_vlm_sidecar=True,
    )

    pipeline_router._configure_predict_model_routing(config, routing)

    predict_config = config["steps"]["predict"]
    assert predict_config["vlm"]["temperature"] == 0.2
    assert predict_config["vlm"]["max_tokens"] == 512
    assert predict_config["llm"]["temperature"] == 0.1
    assert predict_config["llm"]["max_tokens"] == 256
    assert predict_config["llm"]["base_url"] == "http://vlm-nim:8000/v1"


def test_predict_model_routing_preserves_openai_base_url_and_key_env(
    monkeypatch,
) -> None:
    resolved_secret = "material-routing-runtime-only-key"
    monkeypatch.delenv("MA_VLM_NIM_BASE_URL", raising=False)
    monkeypatch.delenv("MA_LLM_NIM_BASE_URL", raising=False)
    monkeypatch.setattr(pipeline_router.config, "vlm_backend", "openai")
    monkeypatch.setattr(pipeline_router.config, "vlm_model", "my-custom-vlm")
    monkeypatch.setattr(
        pipeline_router.config,
        "vlm_base_url",
        "https://api.openai-compatible.example/v1",
    )
    monkeypatch.setattr(pipeline_router.config, "vlm_api_key", None)
    monkeypatch.setattr(pipeline_router.config, "vlm_api_key_env", "OPENAI_API_KEY")
    monkeypatch.setattr(pipeline_router.config, "llm_backend", "openai")
    monkeypatch.setattr(pipeline_router.config, "llm_model", "my-custom-llm")
    monkeypatch.setattr(
        pipeline_router.config,
        "llm_base_url",
        "https://api.openai-compatible.example/v1",
    )
    monkeypatch.setattr(pipeline_router.config, "llm_api_key", None)
    monkeypatch.setattr(pipeline_router.config, "llm_api_key_env", "OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", resolved_secret)

    config = {
        "steps": {
            "predict": {
                "llm": {
                    "api_key": "stale-key",
                    "api_key_env": "STALE_KEY_ENV",
                }
            }
        }
    }
    routing = pipeline_router._resolve_pipeline_model_routing()

    pipeline_router._configure_predict_model_routing(config, routing)

    predict_config = config["steps"]["predict"]
    assert predict_config["vlm"]["backend"] == "openai"
    assert predict_config["vlm"]["base_url"] == (
        "https://api.openai-compatible.example/v1"
    )
    assert predict_config["vlm"]["api_key_env"] == "${OPENAI_API_KEY}"
    assert predict_config["llm"]["backend"] == "openai"
    assert predict_config["llm"]["base_url"] == (
        "https://api.openai-compatible.example/v1"
    )
    assert predict_config["llm"]["api_key_env"] == "${OPENAI_API_KEY}"
    assert "api_key" not in predict_config["llm"]
    ensure_no_inline_secrets(config, context="material service publication")
    assert resolved_secret not in repr(config)


def test_predict_model_routing_preserves_config_only_keys_in_memory(
    monkeypatch,
) -> None:
    sentinel_vlm_key = "sentinel-vlm-key"
    sentinel_llm_key = "sentinel-llm-key"
    monkeypatch.delenv("MA_VLM_NIM_BASE_URL", raising=False)
    monkeypatch.delenv("MA_LLM_NIM_BASE_URL", raising=False)
    monkeypatch.delenv("MA_VLM_API_KEY", raising=False)
    monkeypatch.delenv("MA_LLM_API_KEY", raising=False)
    monkeypatch.setattr(pipeline_router.config, "vlm_backend", "openai")
    monkeypatch.setattr(pipeline_router.config, "vlm_model", "custom-vlm")
    monkeypatch.setattr(pipeline_router.config, "vlm_base_url", "https://vlm.test/v1")
    monkeypatch.setattr(pipeline_router.config, "vlm_api_key", sentinel_vlm_key)
    monkeypatch.setattr(pipeline_router.config, "vlm_api_key_env", None)
    monkeypatch.setattr(pipeline_router.config, "llm_backend", "openai")
    monkeypatch.setattr(pipeline_router.config, "llm_model", "custom-llm")
    monkeypatch.setattr(pipeline_router.config, "llm_base_url", "https://llm.test/v1")
    monkeypatch.setattr(pipeline_router.config, "llm_api_key", sentinel_llm_key)
    monkeypatch.setattr(pipeline_router.config, "llm_api_key_env", None)

    config = {"steps": {"predict": {"llm": {}}}}
    routing = pipeline_router._resolve_pipeline_model_routing()
    pipeline_router._configure_predict_model_routing(config, routing)

    predict_config = config["steps"]["predict"]
    assert predict_config["vlm"]["api_key"] == sentinel_vlm_key
    assert predict_config["llm"]["api_key"] == sentinel_llm_key
    assert "api_key_env" not in predict_config["vlm"]
    assert "api_key_env" not in predict_config["llm"]
    assert (
        pipeline_router.resolve_endpoint_api_key(
            api_key=predict_config["vlm"]["api_key"],
            api_key_env=predict_config["vlm"].get("api_key_env"),
        )
        == sentinel_vlm_key
    )
    assert (
        pipeline_router.resolve_endpoint_api_key(
            api_key=predict_config["llm"]["api_key"],
            api_key_env=predict_config["llm"].get("api_key_env"),
        )
        == sentinel_llm_key
    )


def test_request_nim_vlm_model_clears_service_endpoint_config(monkeypatch) -> None:
    monkeypatch.delenv("MA_VLM_NIM_BASE_URL", raising=False)
    monkeypatch.delenv("MA_LLM_NIM_BASE_URL", raising=False)
    monkeypatch.setattr(pipeline_router.config, "vlm_backend", "openai")
    monkeypatch.setattr(pipeline_router.config, "vlm_model", "my-custom-vlm")
    monkeypatch.setattr(
        pipeline_router.config,
        "vlm_base_url",
        "https://api.openai-compatible.example/v1",
    )
    monkeypatch.setattr(pipeline_router.config, "vlm_api_key", None)
    monkeypatch.setattr(pipeline_router.config, "vlm_api_key_env", "OPENAI_API_KEY")

    config = {"steps": {"predict": {"llm": {}}}}
    routing = pipeline_router._resolve_pipeline_model_routing(
        vlm_model="nim/Qwen/Qwen2.5-VL-7B-Instruct"
    )

    pipeline_router._configure_predict_model_routing(config, routing)

    vlm_config = config["steps"]["predict"]["vlm"]
    assert vlm_config["backend"] == "nim"
    assert vlm_config["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert "base_url" not in vlm_config
    assert "api_key_env" not in vlm_config


def test_cluster_step_config_preserves_embedding_api_key_env(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_router.config, "cluster_embedding_backend", "nim")
    monkeypatch.setattr(
        pipeline_router.config,
        "cluster_embedding_model",
        "nvidia/llama-nemotron-embed-vl-1b-v2",
    )
    monkeypatch.setattr(
        pipeline_router.config,
        "cluster_embedding_base_url",
        "http://embed-nim:8000/v1",
    )
    monkeypatch.setattr(pipeline_router.config, "cluster_embedding_api_key", None)
    monkeypatch.setattr(
        pipeline_router.config,
        "cluster_embedding_api_key_env",
        "CLUSTER_EMBEDDING_API_KEY",
    )
    monkeypatch.setenv("CLUSTER_EMBEDDING_API_KEY", "endpoint-embedding-key")

    cluster_config = pipeline_router._build_cluster_prims_step_config(
        cluster_min_prims=None,
        cluster_embedding_backend=None,
        cluster_embedding_model=None,
        cluster_embedding_base_url=None,
        cluster_embedding_max_workers=None,
        cluster_embedding_batch_size=None,
        cluster_max_size=None,
        cluster_similarity_threshold_low=None,
        cluster_similarity_threshold_medium=None,
        cluster_similarity_threshold_high=None,
        cluster_report="true",
    )

    assert cluster_config["base_url"] == "http://embed-nim:8000/v1"
    assert cluster_config["api_key_env"] == "${CLUSTER_EMBEDDING_API_KEY}"
    assert "api_key" not in cluster_config
    ensure_no_inline_secrets(cluster_config, context="cluster config publication")
    assert "endpoint-embedding-key" not in repr(cluster_config)
