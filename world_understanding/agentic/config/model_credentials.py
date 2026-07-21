# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selected-step model credential preflight shared by agent CLIs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from world_understanding.utils.credentials import (
    API_KEY_ENV_VAR_MAP,
    OPENAI_ENV_REDIRECT_CREDENTIAL_MESSAGE,
    get_env_api_key_for_backend,
    get_nim_api_key_for_base_url,
    get_openai_api_key_for_base_url,
    is_local_base_url,
    is_nvidia_provider_base_url,
    is_openai_provider_base_url,
    redact_sensitive_config,
    redact_sensitive_path,
    resolve_effective_openai_base_url,
    resolve_endpoint_api_key,
)

_MODEL_CONFIG_KEYS = {"vlm", "llm", "vlm_judge", "llm_judge", "image_gen"}
ModelConfigIterator = Callable[
    [str, dict[str, Any], str], list[tuple[str, dict[str, Any]]]
]
ModelConfigTransform = Callable[[str, str, dict[str, Any]], dict[str, Any]]


def _deep_merge(
    defaults: dict[str, Any], user_config: dict[str, Any]
) -> dict[str, Any]:
    merged = defaults.copy()
    for key, value in user_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _iter_model_configs(
    config: dict[str, Any], path: str
) -> list[tuple[str, dict[str, Any]]]:
    model_configs: list[tuple[str, dict[str, Any]]] = []
    for key, value in config.items():
        child_path = f"{path}.{key}" if path else key
        if key in _MODEL_CONFIG_KEYS and isinstance(value, dict):
            model_configs.append((child_path, value))
        if isinstance(value, dict):
            model_configs.extend(_iter_model_configs(value, child_path))
    return model_configs


def _default_model_config_iterator(
    get_step_defaults: Callable[[str], dict[str, Any]],
) -> ModelConfigIterator:
    def iterator(
        step_name: str,
        step_config: dict[str, Any],
        path: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        return _iter_model_configs(
            _deep_merge(get_step_defaults(step_name), step_config), path
        )

    return iterator


def _is_selected(
    step_name: str,
    step_config: dict[str, Any],
    skip_steps: Iterable[str],
    only_steps: Iterable[str],
) -> bool:
    skip = set(skip_steps)
    only = set(only_steps)
    if step_name in skip or (only and step_name not in only):
        return False
    enabled = step_config.get("enabled")
    if enabled is not None:
        return bool(enabled)
    return any(key != "enabled" for key in step_config)


def _selector_path(model_path: str, model_config: dict[str, Any]) -> str:
    if model_config.get("provider") and not model_config.get("backend"):
        return f"{model_path}.provider"
    return f"{model_path}.backend"


def _resolved_endpoint_key(
    model_config: dict[str, Any],
) -> tuple[str | None, bool]:
    try:
        return (
            resolve_endpoint_api_key(
                model_config.get("api_key"),
                model_config.get("api_key_env"),
                require_env=model_config.get("api_key") is None,
            ),
            False,
        )
    except ValueError:
        return None, True


def _missing_requirement(
    backend: str,
    model_config: dict[str, Any],
) -> tuple[tuple[str, ...], bool] | None:
    env_vars = API_KEY_ENV_VAR_MAP.get(backend)
    if not env_vars:
        return None

    endpoint_key, missing_configured_env = _resolved_endpoint_key(model_config)
    if missing_configured_env:
        return (("the configured api_key_env to be set and non-empty",), False)
    base_url = model_config.get("base_url")
    if backend == "openai":
        if get_openai_api_key_for_base_url(base_url, endpoint_key):
            return None
        effective_base_url = resolve_effective_openai_base_url(base_url)
        if effective_base_url and not is_openai_provider_base_url(effective_base_url):
            env_redirected = not (isinstance(base_url, str) and bool(base_url.strip()))
            if is_local_base_url(effective_base_url):
                return (
                    (
                        "endpoint-scoped api_key or api_key_env paired with base_url",
                        "api_key: not-used for a documented local no-auth endpoint",
                    ),
                    env_redirected,
                )
            return (
                ("endpoint-scoped api_key or api_key_env paired with base_url",),
                env_redirected,
            )
        return (env_vars, False)

    if backend == "nim":
        if get_nim_api_key_for_base_url(base_url, endpoint_key):
            return None
        if base_url and not is_nvidia_provider_base_url(base_url):
            hints: tuple[str, ...] = ("endpoint-scoped api_key or api_key_env",)
            if is_local_base_url(base_url):
                hints += ("api_key: not-used for a documented local no-auth endpoint",)
            return (
                hints,
                False,
            )
        return (env_vars, False)

    if endpoint_key or get_env_api_key_for_backend(backend):
        return None
    return (env_vars, False)


def validate_selected_model_credentials(
    raw_config: dict[str, Any],
    config_path: Path,
    skip_steps: Iterable[str],
    only_steps: Iterable[str],
    *,
    get_step_defaults: Callable[[str], dict[str, Any]] | None = None,
    model_config_iterator: ModelConfigIterator | None = None,
    transform_model_config: ModelConfigTransform | None = None,
    completed_steps: Iterable[str] = (),
    guidance_lines: Iterable[str] = (),
) -> None:
    """Fail before model/pipeline construction when selected credentials are absent."""
    steps = raw_config.get("steps") if "project" in raw_config else raw_config
    if not isinstance(steps, dict):
        return
    if model_config_iterator is None:
        if get_step_defaults is None:
            raise TypeError("get_step_defaults or model_config_iterator is required")
        model_config_iterator = _default_model_config_iterator(get_step_defaults)

    completed = set(completed_steps)
    missing: list[tuple[str, str, tuple[str, ...]]] = []
    env_redirected_openai = False
    for step_name, step_config in steps.items():
        if (
            not isinstance(step_config, dict)
            or step_name in completed
            or not _is_selected(step_name, step_config, skip_steps, only_steps)
        ):
            continue
        prefix = f"steps.{step_name}" if "project" in raw_config else step_name
        for model_path, model_config in model_config_iterator(
            step_name, step_config, prefix
        ):
            if transform_model_config is not None:
                model_config = transform_model_config(
                    step_name, model_path, model_config
                )
            backend = model_config.get("backend") or model_config.get("provider")
            if not isinstance(backend, str) or not backend.strip():
                continue
            backend = backend.strip().lower()
            requirement = _missing_requirement(backend, model_config)
            if requirement is None:
                continue
            hints, was_env_redirected = requirement
            env_redirected_openai = env_redirected_openai or was_env_redirected
            missing.append((_selector_path(model_path, model_config), backend, hints))

    if not missing:
        return

    lines = [
        f"Config '{redact_sensitive_path(config_path)}' selects model "
        "backend(s) without required API keys:"
    ]
    for model_path, backend, hints in missing:
        safe_path = redact_sensitive_config(model_path, _path_context=True)
        safe_backend = redact_sensitive_config(backend)
        lines.append(f"- {safe_path}={safe_backend!r} requires {' or '.join(hints)}")
    if env_redirected_openai:
        lines.append(OPENAI_ENV_REDIRECT_CREDENTIAL_MESSAGE)
    lines.extend(guidance_lines)
    raise ValueError("\n".join(lines))
