# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified config loader for the texture agent pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from world_understanding.utils.credentials import (
    LOCAL_NIM_API_KEY_PLACEHOLDER,
    is_local_base_url,
)

from texture_agent.config.rendering_backends import validate_texture_rendering_steps
from texture_agent.config.schema import DEFAULTS, STEP_ORDER, STEP_OUTPUT_DIRS
from texture_agent.functions.detail_policy import normalize_detail_policy

logger = logging.getLogger(__name__)


def load_config(
    config_path: str | Path,
    session_id: str | None = None,
    *,
    config_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate a texture pipeline config file.

    Resolves relative paths against the config file's directory,
    applies defaults, and creates the working directory structure.

    Args:
        config_path: Path to the YAML config file.
        session_id: Optional session ID override used to reuse an existing
            working directory.

    Returns:
        Validated and resolved config dict.
    """
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent

    if config_data is None:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(
                "Config file must contain a YAML mapping, "
                f"got {type(config).__name__}: {config_path}"
            )
    else:
        config = config_data

    # Project defaults
    project = config.setdefault("project", {})
    project.setdefault("name", config_path.stem)
    if session_id:
        project["session_id"] = session_id
    else:
        project.setdefault("session_id", project["name"])

    # Resolve working directory
    working_dir = project.get("working_dir")
    if working_dir:
        working_dir = (config_dir / working_dir).resolve()
    else:
        working_dir = config_dir / f".{project['session_id']}"
    project["working_dir"] = str(working_dir)

    # Resolve input paths
    input_cfg = config.get("input", {})
    if "usd_path" in input_cfg:
        usd_path = Path(input_cfg["usd_path"])
        if not usd_path.is_absolute():
            usd_path = (config_dir / usd_path).resolve()
        input_cfg["usd_path"] = str(usd_path)

    # Apply defaults for texture config
    texture = config.setdefault("texture", {})
    for key, val in DEFAULTS["texture"].items():
        texture.setdefault(key, val)
    texture["detail_policy"] = normalize_detail_policy(
        texture.get("detail_policy"),
        config_key="texture.detail_policy",
    )
    _validate_material_detail_policies(
        config.get("material_textures"),
        default_policy=texture["detail_policy"],
    )
    apply_runtime_endpoint_overrides(config)

    # Apply defaults for variations
    variations = config.setdefault("variations", {})
    for key, val in DEFAULTS["variations"].items():
        variations.setdefault(key, val)

    # Apply defaults for optional auto-prompting
    auto_prompt = config.setdefault("auto_prompt", {})
    for key, val in DEFAULTS["auto_prompt"].items():
        auto_prompt.setdefault(key, val)

    # Apply defaults for steps
    steps = config.setdefault("steps", {})
    for step_name in STEP_ORDER:
        step_cfg = steps.setdefault(step_name, {})
        defaults = DEFAULTS["steps"].get(step_name, {})
        for key, val in defaults.items():
            step_cfg.setdefault(key, val)

    # Rendering selectors are validated before the working directory tree is
    # created. This keeps typos and recognized-but-unsupported backends from
    # leaving partial output state behind.
    validate_texture_rendering_steps(steps)

    # Validate required fields
    if not input_cfg.get("usd_path"):
        raise ValueError("Config missing required field: input.usd_path")

    usd_path = Path(input_cfg["usd_path"])
    if not usd_path.exists():
        raise FileNotFoundError(f"Input USD file does not exist: {usd_path}")

    # Create working directory structure
    working_dir_path = Path(working_dir)
    working_dir_path.mkdir(parents=True, exist_ok=True)
    for _step_name, dir_name in STEP_OUTPUT_DIRS.items():
        (working_dir_path / dir_name).mkdir(parents=True, exist_ok=True)

    logger.info("Loaded config: %s", config_path)
    logger.info("  Project: %s", project["name"])
    logger.info("  Input: %s", input_cfg.get("usd_path"))
    logger.info("  Working dir: %s", working_dir)

    return config


def apply_runtime_endpoint_overrides(config: dict[str, Any]) -> None:
    """Apply Texture Agent local-pipeline endpoint overrides from the environment."""
    image_gen_overrides = {
        "backend": _env_value("TA_IMAGE_GEN_BACKEND"),
        "model": _env_value("TA_IMAGE_GEN_MODEL"),
        "base_url": _env_value("TA_IMAGE_GEN_BASE_URL"),
        "api_key": _env_value("TA_IMAGE_GEN_API_KEY"),
        "api_key_env": _env_value("TA_IMAGE_GEN_API_KEY_ENV"),
    }
    if any(image_gen_overrides.values()):
        texture = config.setdefault("texture", {})
        image_gen = texture.setdefault("image_gen", {})
    else:
        image_gen = None

    if image_gen is not None:
        endpoint_changed = _endpoint_changed(image_gen, image_gen_overrides)
        backend_changed = _override_changed(image_gen, image_gen_overrides, "backend")
        if backend := image_gen_overrides["backend"]:
            image_gen["backend"] = backend
        if model := image_gen_overrides["model"]:
            image_gen["model"] = model
        if base_url := image_gen_overrides["base_url"]:
            image_gen["base_url"] = base_url
        elif backend_changed:
            image_gen.pop("base_url", None)
        resolved_base_url = image_gen.get("base_url")
        api_key_env = image_gen_overrides["api_key_env"]
        if api_key_env:
            _apply_endpoint_api_key_override(
                image_gen,
                api_key=None,
                endpoint_changed=endpoint_changed,
                resolved_base_url=resolved_base_url,
            )
            image_gen["api_key_env"] = api_key_env
            image_gen.pop("api_key", None)
        else:
            _apply_endpoint_api_key_override(
                image_gen,
                api_key=image_gen_overrides["api_key"],
                endpoint_changed=endpoint_changed,
                resolved_base_url=resolved_base_url,
            )

    llm_base_url = _env_value("TA_LLM_BASE_URL") or _env_value("TA_LLM_NIM_BASE_URL")
    llm_overrides = {
        "backend": _env_value("TA_LLM_BACKEND"),
        "model": _env_value("TA_LLM_MODEL"),
        "base_url": llm_base_url,
        "api_key": _env_value("TA_LLM_API_KEY"),
        "api_key_env": _env_value("TA_LLM_API_KEY_ENV"),
        "nim_api_key": _env_value("TA_NIM_API_KEY"),
    }
    if any(llm_overrides.values()):
        auto_prompt = config.setdefault("auto_prompt", {})
        llm = auto_prompt.setdefault("llm", {})
    else:
        llm = None

    if llm is not None:
        endpoint_changed = _endpoint_changed(llm, llm_overrides)
        backend_changed = _override_changed(llm, llm_overrides, "backend")
        if backend := llm_overrides["backend"]:
            llm["backend"] = backend
        if model := llm_overrides["model"]:
            llm["model"] = model
        if base_url := llm_overrides["base_url"]:
            llm["base_url"] = base_url
        elif backend_changed:
            llm.pop("base_url", None)
        resolved_base_url = llm.get("base_url")
        api_key = llm_overrides["api_key"]
        api_key_env = llm_overrides["api_key_env"]
        if api_key_env:
            _apply_endpoint_api_key_override(
                llm,
                api_key=None,
                endpoint_changed=endpoint_changed,
                resolved_base_url=resolved_base_url,
            )
            llm["api_key_env"] = api_key_env
            llm.pop("api_key", None)
        else:
            if not api_key and _uses_nim_llm_credentials(llm):
                api_key = llm_overrides["nim_api_key"]
            _apply_endpoint_api_key_override(
                llm,
                api_key=api_key,
                endpoint_changed=endpoint_changed,
                resolved_base_url=resolved_base_url,
            )


def _validate_material_detail_policies(
    material_textures: Any,
    *,
    default_policy: str,
) -> None:
    """Normalize detail_policy fields in material texture config, if present."""
    if material_textures is None:
        return
    if not isinstance(material_textures, dict):
        return
    for material_name, spec in material_textures.items():
        if not isinstance(spec, dict):
            continue
        key = f"material_textures.{material_name}.detail_policy"
        material_policy = default_policy
        if "detail_policy" in spec:
            material_policy = normalize_detail_policy(
                spec.get("detail_policy"),
                config_key=key,
                default=default_policy,
            )
            spec["detail_policy"] = material_policy
        per_prim = spec.get("per_prim")
        if not isinstance(per_prim, dict):
            continue
        for prim_path, override in per_prim.items():
            if not isinstance(override, dict):
                continue
            if "detail_policy" in override:
                override["detail_policy"] = normalize_detail_policy(
                    override.get("detail_policy"),
                    config_key=(
                        f"material_textures.{material_name}.per_prim."
                        f"{prim_path}.detail_policy"
                    ),
                    default=material_policy,
                )


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    return None


def _endpoint_changed(
    section: dict[str, Any], overrides: dict[str, str | None]
) -> bool:
    return bool(
        _override_changed(section, overrides, "backend")
        or _override_changed(section, overrides, "base_url")
    )


def _override_changed(
    section: dict[str, Any], overrides: dict[str, str | None], key: str
) -> bool:
    return overrides.get(key) is not None and overrides[key] != section.get(key)


def _uses_nim_llm_credentials(llm: dict[str, Any]) -> bool:
    return llm.get("backend", "nim") == "nim"


def _apply_endpoint_api_key_override(
    section: dict[str, Any],
    *,
    api_key: str | None,
    endpoint_changed: bool,
    resolved_base_url: Any,
) -> None:
    """Update endpoint-scoped API keys after runtime endpoint overrides."""
    if api_key:
        section["api_key"] = api_key
        section.pop("api_key_env", None)
        return

    if endpoint_changed:
        section.pop("api_key", None)
        section.pop("api_key_env", None)

    if (
        isinstance(resolved_base_url, str)
        and is_local_base_url(resolved_base_url)
        and (endpoint_changed or not section.get("api_key"))
    ):
        section["api_key"] = LOCAL_NIM_API_KEY_PLACEHOLDER


def config_to_context(config: dict[str, Any]) -> dict[str, Any]:
    """Convert a loaded config dict into a pipeline context dict.

    Maps config sections to the context keys expected by pipeline tasks.

    Args:
        config: Loaded config from load_config().

    Returns:
        Context dict for pipeline execution.
    """
    return {
        "usd_path": config["input"]["usd_path"],
        "prim_paths": config["input"].get("prim_paths"),
        "working_dir": config["project"]["working_dir"],
        "texture_config": config.get("texture", {}),
        "material_textures": config.get("material_textures", {}),
        "blend_config": config["steps"].get("blend_textures", {}),
        "render_preview_config": config["steps"].get("render_previews", {}),
        "render_config": config["steps"].get("render", {}),
        "auto_prompt_config": config.get("auto_prompt", {}),
        "planning_config": config.get("planning", {}),
        "variations_config": config.get("variations", {}),
        "steps": config.get("steps", {}),
        "config": config,
    }
