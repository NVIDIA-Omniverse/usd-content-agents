# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from material_agent.api.simulate_config import patch_config_for_simulate


def test_patch_config_for_simulate_patches_expected_backends_without_mutating_input():
    original = {
        "steps": {
            "predict": {
                "vlm": {"backend": "nim"},
                "llm": {"backend": "nim"},
            },
            "validate_predictions": {"llm": {"backend": "nim"}},
            "harmonize_predictions": {"llm": {"backend": "nim"}},
            "build_dataset_usd": {"renderer": {"backend": "remote"}},
            "render": {"backend": "remote"},
            "cluster_prims": {"embedding_service": "nim"},
            "optimize_usd": {"backend": "remote"},
        },
        "scene": {
            "analyze": {"llm": {"backend": "real"}},
            "reconcile": {"llm": {"backend": "real"}},
            "harmonize": {"llm": {"backend": "real"}},
        },
    }

    patched = patch_config_for_simulate(original)

    assert patched is not original
    assert original["steps"]["predict"]["vlm"]["backend"] == "nim"
    assert patched["steps"]["predict"]["vlm"]["backend"] == "mock"
    assert patched["steps"]["predict"]["vlm"]["api_key"] == "not-used"
    assert patched["steps"]["predict"]["llm"]["backend"] == "mock"
    assert patched["steps"]["validate_predictions"]["llm"]["backend"] == "mock"
    assert patched["steps"]["harmonize_predictions"]["llm"]["backend"] == "mock"
    assert patched["steps"]["build_dataset_usd"]["renderer"]["backend"] == "mock"
    assert patched["steps"]["render"]["backend"] == "mock"
    assert patched["steps"]["cluster_prims"]["embedding_service"] == "mock"
    assert patched["scene"]["reconcile"]["llm"]["backend"] == "mock"
    assert patched["scene"]["harmonize"]["llm"]["backend"] == "mock"
    assert patched["scene"]["analyze"]["llm"]["backend"] == "real"


def test_patch_config_for_simulate_can_mock_scene_analyze():
    patched = patch_config_for_simulate(
        {"scene": {"analyze": {"llm": {"backend": "real"}}}},
        mock_analyze=True,
    )

    assert patched["scene"]["analyze"]["llm"]["backend"] == "mock"
    assert patched["scene"]["analyze"]["llm"]["api_key"] == "not-used"


def test_patch_config_for_simulate_sets_missing_cluster_embedding_service():
    patched = patch_config_for_simulate({"steps": {"cluster_prims": {"enabled": True}}})

    assert patched["steps"]["cluster_prims"]["embedding_service"] == "mock"
    assert patched["steps"]["cluster_prims"]["api_key"] == "not-used"


def test_patch_config_for_simulate_creates_missing_mock_scene_analyze():
    patched = patch_config_for_simulate({"steps": {}}, mock_analyze=True)

    assert patched["scene"]["analyze"]["llm"]["backend"] == "mock"
    assert patched["scene"]["analyze"]["llm"]["api_key"] == "not-used"


def test_patch_config_for_simulate_replaces_non_mapping_scene() -> None:
    patched = patch_config_for_simulate({"scene": "not-a-mapping"}, mock_analyze=True)

    assert patched["scene"]["analyze"]["llm"]["backend"] == "mock"


def test_patch_config_for_simulate_skips_non_analyze_missing_llm() -> None:
    patched = patch_config_for_simulate({"scene": {"reconcile": {}}})

    assert patched["scene"]["reconcile"] == {}


def test_patch_config_for_simulate_clears_real_provider_credentials() -> None:
    """Switching a section to the mock backend must drop any prior provider
    api_key/base_url. Otherwise a real credential would persist on a config
    that now claims to be a mock-only run, e.g. if the patched config is
    later serialized for inspection."""
    original = {
        "steps": {
            "predict": {
                "vlm": {
                    "backend": "openai",
                    "model": "example-vlm-model",
                    "api_key": "sk-real-openai-key",
                    "base_url": "https://api.openai.com/v1",
                },
                "llm": {
                    "backend": "nim",
                    "api_key": "nvidia-real-key",
                    "base_url": "https://integrate.api.nvidia.com/v1",
                },
            }
        },
        "scene": {
            "reconcile": {
                "llm": {
                    "backend": "openai",
                    "api_key": "sk-reconcile-openai-key",
                    "base_url": "https://api.openai.com/v1",
                }
            }
        },
    }

    patched = patch_config_for_simulate(original)

    vlm = patched["steps"]["predict"]["vlm"]
    assert vlm["backend"] == "mock"
    assert vlm["api_key"] == "not-used"
    assert "base_url" not in vlm

    llm = patched["steps"]["predict"]["llm"]
    assert llm["backend"] == "mock"
    assert llm["api_key"] == "not-used"
    assert "base_url" not in llm

    reconcile = patched["scene"]["reconcile"]["llm"]
    assert reconcile["backend"] == "mock"
    assert reconcile["api_key"] == "not-used"
    assert "base_url" not in reconcile

    # Original config must remain untouched.
    assert original["steps"]["predict"]["vlm"]["api_key"] == "sk-real-openai-key"
    assert original["steps"]["predict"]["llm"]["api_key"] == "nvidia-real-key"
    assert original["scene"]["reconcile"]["llm"]["api_key"] == "sk-reconcile-openai-key"
