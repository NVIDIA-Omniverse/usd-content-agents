# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config patching for simulate mode.

Rewrites all VLM / LLM / renderer / embedding backend fields to ``"mock"``
so that the real pipeline runs with instant, deterministic, no-network
backends.
"""

from __future__ import annotations

import logging
from typing import Any

from world_understanding.agentic.config import clone_config_containers
from world_understanding.utils.credentials import (
    LOCAL_NIM_API_KEY_PLACEHOLDER,
    drop_stale_endpoint_credentials,
)

logger = logging.getLogger(__name__)


def patch_config_for_simulate(
    config: dict[str, Any],
    *,
    mock_analyze: bool = False,
) -> dict[str, Any]:
    """Return a deep copy of *config* with all backends set to ``"mock"``.

    Args:
        config: The original config dict (will not be mutated).
        mock_analyze: If ``False`` (default), ``scene.analyze.llm`` is left
            unchanged so the real LLM drives scene decomposition.  Set to
            ``True`` to mock it as well (faster but produces worse splits).

    The scene optimizer (``optimize_usd``) is always left unchanged.
    """
    cfg = clone_config_containers(config)
    if not isinstance(cfg, dict):  # pragma: no cover - input type contract
        raise AssertionError("simulation configuration clone must be a dictionary")
    patched: list[str] = []

    steps = cfg.setdefault("steps", {})

    # Switching a section to the mock backend voids its previous endpoint
    # credentials/URL; clear them so a real provider key cannot persist on a
    # config that now claims to be a mock-only run.
    def _switch_to_mock(section: dict[str, Any]) -> None:
        drop_stale_endpoint_credentials(section)
        section["backend"] = "mock"
        section["api_key"] = LOCAL_NIM_API_KEY_PLACEHOLDER

    # -- predict step --
    predict = steps.get("predict", {})
    if isinstance(predict.get("vlm"), dict):
        _switch_to_mock(predict["vlm"])
        patched.append("steps.predict.vlm.backend")
    if isinstance(predict.get("llm"), dict):
        _switch_to_mock(predict["llm"])
        patched.append("steps.predict.llm.backend")

    # -- validate_predictions step --
    validate = steps.get("validate_predictions", {})
    if isinstance(validate.get("llm"), dict):
        _switch_to_mock(validate["llm"])
        patched.append("steps.validate_predictions.llm.backend")

    # -- harmonize_predictions step --
    harmonize = steps.get("harmonize_predictions", {})
    if isinstance(harmonize.get("llm"), dict):
        _switch_to_mock(harmonize["llm"])
        patched.append("steps.harmonize_predictions.llm.backend")

    # -- build_dataset_usd renderer --
    bdu = steps.get("build_dataset_usd", {})
    renderer = bdu.get("renderer", {})
    if isinstance(renderer, dict) and "backend" in renderer:
        renderer["backend"] = "mock"
        patched.append("steps.build_dataset_usd.renderer.backend")

    # -- render step --
    render = steps.get("render", {})
    if isinstance(render, dict) and "backend" in render:
        render["backend"] = "mock"
        patched.append("steps.render.backend")

    # -- cluster_prims embedding service --
    cluster = steps.get("cluster_prims", {})
    if isinstance(cluster, dict):
        cluster["embedding_service"] = "mock"
        cluster["api_key"] = "not-used"
        patched.append("steps.cluster_prims.embedding_service")

    # -- scene-level LLM configs --
    scene = cfg.setdefault("scene", {})
    if not isinstance(scene, dict):
        scene = {}
        cfg["scene"] = scene
    scene_llm_sections = ["reconcile", "harmonize"]
    if mock_analyze:
        scene_llm_sections.insert(0, "analyze")
    for section_name in scene_llm_sections:
        section = scene.get(section_name)
        if not isinstance(section, dict):
            if section_name != "analyze" or not mock_analyze:
                continue
            section = {}
            scene[section_name] = section

        llm = section.get("llm")
        if not isinstance(llm, dict):
            if section_name != "analyze" or not mock_analyze:
                continue
            llm = {}
            section["llm"] = llm

        _switch_to_mock(llm)
        patched.append(f"scene.{section_name}.llm.backend")

    if not mock_analyze:
        logger.info(
            "simulate: scene.analyze.llm kept real (use --simulate-mock-analyze to mock)"
        )

    if patched:
        logger.info("simulate: patched backends to 'mock': %s", ", ".join(patched))

    return cfg
