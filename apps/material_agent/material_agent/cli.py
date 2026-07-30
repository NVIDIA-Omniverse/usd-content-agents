# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material Agent CLI interface using Typer and Rich."""

# ruff: noqa: E402

import atexit
import io
import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any

# Ensure stdout/stderr use UTF-8 on Windows (avoids charmap errors with Unicode
# characters such as arrows printed by Rich tables).
if hasattr(sys.stdout, "buffer") and (
    sys.stdout.encoding or ""
).lower() not in (  # pragma: no cover - interpreter encoding guard
    "utf-8",
    "utf8",
):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
if hasattr(sys.stderr, "buffer") and (
    sys.stderr.encoding or ""
).lower() not in (  # pragma: no cover - interpreter encoding guard
    "utf-8",
    "utf8",
):
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

import typer
import yaml
from dotenv import find_dotenv, load_dotenv


# Load environment variables before importing modules with env-derived constants.
def _load_cli_dotenv() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    load_dotenv(dotenv_path=dotenv_path or Path.cwd() / ".env")


_load_cli_dotenv()

from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from world_understanding.agentic.cli import (
    load_cli_config_mapping,
    normalize_cli_step_filters,
    sever_cli_exception_graph,
)
from world_understanding.agentic.config import (
    API_KEY_ENV_VAR_MAP,
    is_local_base_url,
    is_local_nim_api_key_placeholder,
    is_placeholder_api_key,
)
from world_understanding.agentic.events import get_listener

# Import telemetry initialization functions
from world_understanding.telemetry import (
    TelemetryConfig,
    get_tracer,
    initialize_telemetry,
    shutdown_telemetry,
)
from world_understanding.telemetry.attributes import MAAttributes
from world_understanding.utils.credentials import (
    OPENAI_ENV_REDIRECT_CREDENTIAL_MESSAGE,
    drop_stale_endpoint_credentials,
    get_nim_api_key_for_base_url,
    get_openai_api_key_for_base_url,
    is_nvidia_provider_base_url,
    is_openai_provider_base_url,
    parse_env_reference,
    path_exists_with_safe_diagnostics,
    redact_sensitive_config,
    redact_sensitive_path,
    resolve_effective_openai_base_url,
    resolve_endpoint_api_key,
    resolve_path_with_safe_diagnostics,
)
from world_understanding.utils.model_auth import (
    MODEL_AUTHENTICATION_FAILURE_MESSAGE,
    is_model_authentication_error,
    public_model_failure_message,
)

from .scene.cli import scene_app  # noqa: E402
from .utils import get_version  # noqa: E402

__version__ = get_version()

# Initialize Typer app and Rich console
app = typer.Typer(
    name="material-agent",
    help="Material Agent - VLM-based material assignment for 3D objects",
    add_completion=False,
    rich_markup_mode="rich",
    # Config and path locals can hold runtime credentials. Unexpected errors
    # may render a traceback, but never its frame locals.
    pretty_exceptions_show_locals=False,
)
console = Console()


def _get_cli_user_email() -> str | None:
    """Get optional user email for telemetry from environment."""
    user_email = os.getenv("MA_USER_EMAIL", "").strip()
    return user_email or None


_ENV_OVERRIDE_VLM_BACKEND = "MA_VLM_BACKEND"
_ENV_OVERRIDE_VLM_MODEL = "MA_VLM_MODEL"
_ENV_OVERRIDE_LLM_BACKEND = "MA_LLM_BACKEND"
_ENV_OVERRIDE_LLM_MODEL = "MA_LLM_MODEL"

_MODEL_BACKEND_ENV_VARS = API_KEY_ENV_VAR_MAP
_LOCAL_NIM_CREDENTIAL_HINTS = ("MA_NIM_API_KEY", "api_key: not-used")
_LOCAL_OPENAI_CREDENTIAL_HINTS = (
    "endpoint-scoped api_key or api_key_env paired with base_url",
    "api_key: not-used for a documented local no-auth endpoint",
)
_MODEL_CONFIG_KEYS = {"vlm", "llm", "vlm_judge", "llm_judge", "image_gen"}
_EMBEDDING_SERVICE_KEYS = {"embedding_service"}


def _is_truthy_config_value(value: Any) -> bool:
    """Return True when a config/env value is present and non-empty."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _is_step_selected(
    step_name: str,
    step_config: Any,
    skip_steps: list[str],
    only_steps: list[str],
) -> bool:
    if skip_steps and step_name in skip_steps:
        return False
    if only_steps and step_name not in only_steps:
        return False
    if not isinstance(step_config, dict):
        return False
    enabled = step_config.get("enabled")
    if enabled is not None:
        return bool(enabled)
    return any(key != "enabled" for key in step_config)


def _deep_merge_config(
    defaults: dict[str, Any], user_config: dict[str, Any]
) -> dict[str, Any]:
    merged = defaults.copy()
    for key, value in user_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_step_defaults(step_name: str, step_config: dict[str, Any]) -> dict[str, Any]:
    from material_agent.config.schema import get_step_defaults

    return _deep_merge_config(get_step_defaults(step_name), step_config)


def _iter_model_configs(
    config: dict[str, Any],
    path: str,
) -> list[tuple[str, dict[str, Any]]]:
    model_configs: list[tuple[str, dict[str, Any]]] = []
    for key, value in config.items():
        child_path = f"{path}.{key}" if path else key
        if key in _MODEL_CONFIG_KEYS and isinstance(value, dict):
            model_configs.append((child_path, value))
        if isinstance(value, dict):
            model_configs.extend(_iter_model_configs(value, child_path))
    return model_configs


def _iter_embedding_service_configs(
    config: dict[str, Any],
    path: str,
) -> list[tuple[str, dict[str, Any]]]:
    model_configs: list[tuple[str, dict[str, Any]]] = []
    for key, value in config.items():
        child_path = f"{path}.{key}" if path else key
        if key in _EMBEDDING_SERVICE_KEYS and isinstance(value, str):
            model_configs.append(
                (
                    child_path,
                    {
                        "backend": value,
                        "api_key": config.get("api_key"),
                    },
                )
            )
        elif key == "embedding" and isinstance(value, dict):
            service = value.get("service")
            if isinstance(service, str):
                model_configs.append(
                    (
                        f"{child_path}.service",
                        {
                            "backend": service,
                            "api_key": value.get("api_key"),
                            "base_url": value.get("base_url"),
                        },
                    )
                )
        if isinstance(value, dict):
            model_configs.extend(_iter_embedding_service_configs(value, child_path))
    return model_configs


def _resolve_config_relative_path(
    value: str | Path | None, config_path: Path
) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return resolve_path_with_safe_diagnostics(
        config_path.parent / path,
        label="configuration-relative path",
    )


def _load_preflight_config(
    config: Path | dict[str, Any], source_config_path: Path | None = None
) -> tuple[Any, Path]:
    """Return effective config data and the path used for relative resolution."""
    if isinstance(config, dict):
        anchor = source_config_path or (Path.cwd() / "config_dict.yaml")
        return config, Path(anchor)

    config_path = Path(config)
    safe_config_path = redact_sensitive_path(config_path)
    failure_kind: str | None = None
    os_error_type: type[OSError] = OSError
    os_error_errno: int | None = None
    fh = None
    try:
        with open(config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}, config_path
    except yaml.YAMLError:
        # Parser diagnostics can echo source lines, including credentials.
        failure_kind = "yaml"
    except UnicodeError:
        failure_kind = "unicode"
    except OSError as error:
        failure_kind = "os"
        os_error_type = type(error)
        os_error_errno = error.errno

    # Raise only after the rejected parser/I/O exception has left the active
    # handler, and remove raw path/file references from the public traceback.
    del config
    del config_path
    del source_config_path
    del fh
    if failure_kind == "yaml":
        raise ValueError(f"Unable to parse configuration file: {safe_config_path}")
    if failure_kind == "unicode":
        raise ValueError(f"Unable to read configuration file: {safe_config_path}")
    if os_error_errno is None:
        raise OSError(f"Unable to read configuration file: {safe_config_path}")
    raise os_error_type(
        os_error_errno,
        "Unable to read configuration file",
        safe_config_path,
    )


def _get_resume_completed_steps(
    raw_config: dict[str, Any],
    config_path: Path,
    resume: bool,
    clean: bool,
    session_id: str | None,
) -> set[str]:
    if not resume or clean:
        return set()

    project = raw_config.get("project") or {}
    if not isinstance(project, dict):
        return set()

    working_dir = _resolve_config_relative_path(project.get("working_dir"), config_path)
    effective_session_id = session_id or project.get("session_id")
    if working_dir is None and isinstance(effective_session_id, str):
        working_dir = _resolve_config_relative_path(
            f".{effective_session_id}",
            config_path,
        )
    if working_dir is None:
        return set()

    state_file = working_dir / ".pipeline_state.json"
    try:
        with open(state_file, encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return set()

    completed_steps = state.get("completed_steps")
    if not isinstance(completed_steps, list):
        return set()
    return {step for step in completed_steps if isinstance(step, str)}


def _get_optimize_usd_backend(step_config: dict[str, Any]) -> str:
    effective_config = _merge_step_defaults("optimize_usd", step_config)
    optimization_config = effective_config.get("optimization_config")
    if isinstance(optimization_config, dict):
        backend = optimization_config.get("backend")
        if isinstance(backend, str) and backend.strip():
            return backend.strip().lower()
    return "local"


def _scene_optimizer_package_issue() -> str | None:
    from world_understanding.functions.graphics.scene_optimizer_local import (
        SO_PACKAGE_SUBDIRS,
        _default_so_package_dir,
        _is_valid_so_package_dir,
    )

    package_dir_env = os.environ.get("WU_SO_PACKAGE_DIR")
    if package_dir_env:
        package_dir = Path(package_dir_env)
        safe_package_dir = redact_sensitive_path(package_dir)
        try:
            missing = [
                sub for sub in SO_PACKAGE_SUBDIRS if not (package_dir / sub).is_dir()
            ]
        except OSError:
            return (
                f"Unable to inspect Scene Optimizer Core package at {safe_package_dir}."
            )
        if missing:
            return (
                "Scene Optimizer Core package at WU_SO_PACKAGE_DIR="
                f"{safe_package_dir} "
                f"is missing expected subdirectories: {', '.join(missing)}."
            )
        return None

    default_dir = _default_so_package_dir()
    if _is_valid_so_package_dir(default_dir):
        return None
    return (
        "Scene Optimizer Core package not found at "
        f"{redact_sensitive_path(default_dir)}. Run "
        "`./scripts/fetch_build_resources.sh` from the repo root inside WSL/Linux, "
        "or set WU_SO_PACKAGE_DIR to an unpacked scene_optimizer_core package."
    )


def _validate_run_config_windows_prerequisites(
    config_path: Path | dict[str, Any],
    skip_steps: list[str],
    only_steps: list[str],
    *,
    resume: bool = False,
    clean: bool = False,
    session_id: str | None = None,
    source_config_path: Path | None = None,
) -> None:
    """Fail fast for native-Windows full-pipeline prerequisites."""
    if sys.platform != "win32":
        return
    if os.getenv("NVCF_OPTIMIZER_FUNCTION_ID") or os.getenv("OPTIMIZER_ENDPOINT"):
        return

    raw_config, effective_config_path = _load_preflight_config(
        config_path, source_config_path
    )
    if not isinstance(raw_config, dict):
        return

    steps = raw_config.get("steps") if "project" in raw_config else raw_config
    if not isinstance(steps, dict):
        return

    step_config = steps.get("optimize_usd")
    if not isinstance(step_config, dict):
        return

    completed_steps = _get_resume_completed_steps(
        raw_config,
        effective_config_path,
        resume=resume,
        clean=clean,
        session_id=session_id,
    )
    if "optimize_usd" in completed_steps:
        return
    if not _is_step_selected("optimize_usd", step_config, skip_steps, only_steps):
        return
    if _get_optimize_usd_backend(step_config) != "local":
        return

    issues: list[str] = []
    if shutil.which("wsl") is None:
        issues.append("WSL launcher `wsl.exe` was not found on PATH.")
    if shutil.which("bash") is None:
        issues.append("`bash` was not found on PATH.")
    package_issue = _scene_optimizer_package_issue()
    if package_issue:
        issues.append(package_issue)

    safe_config_path = redact_sensitive_path(effective_config_path)
    lines = [
        f"Config '{safe_config_path}' selects optimize_usd with the local Scene "
        "Optimizer backend on native Windows.",
        "The full Material Agent CLI pipeline must run inside WSL/Linux for "
        "this path, because the local Scene Optimizer package is a Linux "
        "runtime dependency.",
    ]
    if issues:
        lines.append("Prerequisite check:")
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append(
            "WSL/bash and Scene Optimizer appear present, but this process is "
            "still native Windows; start the command from inside WSL/Linux."
        )
    lines.extend(
        [
            "Fix one of:",
            "- Run the command inside WSL/Linux after installing the repo there.",
            "- Skip the step: `material-agent run CONFIG --skip optimize_usd` "
            "or set `steps.optimize_usd.enabled: false`.",
            "- Use the remote optimizer: set "
            "`steps.optimize_usd.optimization_config.backend: remote` and "
            "configure NVCF_OPTIMIZER_FUNCTION_ID or OPTIMIZER_ENDPOINT.",
        ]
    )
    raise ValueError("\n".join(lines))


def _iter_selected_model_configs(
    step_name: str,
    step_config: dict[str, Any],
    model_path_prefix: str,
) -> list[tuple[str, dict[str, Any]]]:
    effective_config = _merge_step_defaults(step_name, step_config)
    model_configs = _iter_model_configs(effective_config, model_path_prefix)
    model_configs.extend(
        _iter_embedding_service_configs(effective_config, model_path_prefix)
    )
    if step_name == "generate_reference_image" and not any(
        path.endswith(".image_gen") for path, _ in model_configs
    ):
        model_configs.append((f"{model_path_prefix}.image_gen", {"backend": "gemini"}))
    return model_configs


def _model_backend_selector_path(model_path: str, model_config: dict[str, Any]) -> str:
    if model_path.endswith(".embedding_service") or model_path.endswith(
        ".embedding.service"
    ):
        return model_path
    if model_config.get("provider") and not model_config.get("backend"):
        return f"{model_path}.provider"
    return f"{model_path}.backend"


def _uses_local_nim_placeholder(model_path: str) -> bool:
    return model_path.endswith(
        (".vlm", ".llm", ".vlm_judge", ".llm_judge", ".image_gen")
    )


def _apply_runtime_model_overrides(
    step_name: str, model_path: str, model_config: dict[str, Any]
) -> dict[str, Any]:
    config = model_config.copy()

    # Mirror runtime override scope so preflight does not reject configs that
    # runtime would actually re-route:
    # - VLM env override (``MA_VLM_NIM_BASE_URL``) is applied at runtime only
    #   by ``PredictConfigTask``, so preflight applies it only in ``predict``.
    # - LLM env override (``MA_LLM_NIM_BASE_URL`` / ``MA_VLM_NIM_BASE_URL``)
    #   is applied by ``create_chat_model_from_config`` for *every* LLM call,
    #   so preflight applies it to any step's LLM section.
    if model_path.endswith((".vlm", ".vlm_judge")):
        nim_base_url = (
            os.getenv("MA_VLM_NIM_BASE_URL") if step_name == "predict" else None
        )
    elif model_path.endswith((".llm", ".llm_judge")):
        nim_base_url = os.getenv("MA_LLM_NIM_BASE_URL") or os.getenv(
            "MA_VLM_NIM_BASE_URL"
        )
    else:
        nim_base_url = None

    if nim_base_url:
        # Don't pierce mock/echo configs — the override is a runtime routing
        # hint, not a way to retarget deliberately-mocked simulate runs.
        current_backend = (
            (config.get("backend") or config.get("provider") or "").strip().lower()
        )
        if current_backend in ("", "echo", "mock"):
            return config
        drop_stale_endpoint_credentials(config, preserve_local_nim_placeholder=True)
        config["backend"] = "nim"
        config["base_url"] = nim_base_url
    return config


def _has_config_api_key(
    backend: str, model_path: str, model_config: dict[str, Any]
) -> bool:
    api_key = model_config.get("api_key")
    api_key_env = model_config.get("api_key_env")
    if not _is_truthy_config_value(api_key) and _is_truthy_config_value(api_key_env):
        env_name = parse_env_reference(api_key_env, allow_legacy_bare=True)
        api_key = os.getenv(env_name) if env_name else None
    if not _is_truthy_config_value(api_key):
        return False
    if not is_placeholder_api_key(api_key):
        return True
    if (
        backend == "openai"
        and _accepts_local_openai_placeholder_key(backend, model_path, model_config)
        and is_local_nim_api_key_placeholder(api_key)
    ):
        return True
    return (
        backend == "nim"
        and _uses_local_nim_placeholder(model_path)
        and is_local_nim_api_key_placeholder(api_key)
        and is_local_base_url(model_config.get("base_url"))
    )


def _has_env_api_key(env_vars: tuple[str, ...]) -> bool:
    return any(
        os.getenv(env_var) and not is_placeholder_api_key(os.getenv(env_var))
        for env_var in env_vars
    )


def _accepts_local_openai_placeholder_key(
    backend: str, model_path: str, model_config: dict[str, Any]
) -> bool:
    if backend != "openai" or not model_path.endswith(
        (".vlm", ".llm", ".vlm_judge", ".llm_judge", ".image_gen")
    ):
        return False
    if not is_local_base_url(model_config.get("base_url")):
        return False

    return is_local_nim_api_key_placeholder(model_config.get("api_key"))


def _has_local_nim_api_key(model_config: dict[str, Any]) -> bool:
    return bool(
        get_nim_api_key_for_base_url(
            model_config.get("base_url"),
            model_config.get("api_key"),
        )
    )


def _has_local_openai_api_key(model_config: dict[str, Any]) -> bool:
    api_key = resolve_endpoint_api_key(
        model_config.get("api_key"),
        model_config.get("api_key_env"),
    )
    return bool(
        get_openai_api_key_for_base_url(
            model_config.get("base_url"),
            api_key,
        )
    )


def _validate_run_config_model_credentials(
    config_path: Path | dict[str, Any],
    skip_steps: list[str],
    only_steps: list[str],
    *,
    resume: bool = False,
    clean: bool = False,
    session_id: str | None = None,
    source_config_path: Path | None = None,
) -> None:
    """Fail fast when selected model backends lack their required API keys."""
    raw_config, effective_config_path = _load_preflight_config(
        config_path, source_config_path
    )
    if not isinstance(raw_config, dict):
        return

    steps = raw_config.get("steps") if "project" in raw_config else raw_config
    if not isinstance(steps, dict):
        return

    completed_steps = _get_resume_completed_steps(
        raw_config,
        effective_config_path,
        resume=resume,
        clean=clean,
        session_id=session_id,
    )

    missing: list[tuple[str, str, tuple[str, ...]]] = []
    env_redirected_openai = False
    for step_name, step_config in steps.items():
        if not isinstance(step_config, dict):
            continue
        if step_name in completed_steps:
            continue
        if not _is_step_selected(step_name, step_config, skip_steps, only_steps):
            continue

        model_path_prefix = (
            f"steps.{step_name}" if "project" in raw_config else step_name
        )
        for model_path, model_config in _iter_selected_model_configs(
            step_name, step_config, model_path_prefix
        ):
            model_config = _apply_runtime_model_overrides(
                step_name, model_path, model_config
            )
            backend = model_config.get("backend") or model_config.get("provider")
            if not isinstance(backend, str) or not backend.strip():
                continue
            backend = backend.strip()
            env_vars = _MODEL_BACKEND_ENV_VARS.get(backend)
            if not env_vars:
                continue
            if not _is_truthy_config_value(
                model_config.get("api_key")
            ) and _is_truthy_config_value(model_config.get("api_key_env")):
                env_name = parse_env_reference(
                    model_config.get("api_key_env"), allow_legacy_bare=True
                )
                if not env_name or not os.getenv(env_name):
                    missing.append(
                        (
                            _model_backend_selector_path(model_path, model_config),
                            backend,
                            ("the configured api_key_env to be set and non-empty",),
                        )
                    )
                    continue
            if backend == "openai":
                # Route OpenAI through the same endpoint-aware resolver the
                # runtime factory uses so preflight cannot greenlight a
                # config that runtime will then reject — including when
                # ``OPENAI_BASE_URL`` / ``OPENAI_API_BASE`` redirects an
                # otherwise-hosted config to a custom endpoint.
                if _has_local_openai_api_key(model_config):
                    continue
                effective_base_url = resolve_effective_openai_base_url(
                    model_config.get("base_url")
                )
                if is_local_base_url(effective_base_url):
                    credential_hints = _LOCAL_OPENAI_CREDENTIAL_HINTS
                elif effective_base_url:
                    credential_hints = (
                        env_vars
                        if is_openai_provider_base_url(effective_base_url)
                        else (
                            "endpoint-scoped api_key or api_key_env paired "
                            "with base_url",
                        )
                    )
                else:
                    credential_hints = env_vars
                configured_base_url = model_config.get("base_url")
                has_explicit_base_url = isinstance(configured_base_url, str) and bool(
                    configured_base_url.strip()
                )
                env_redirected_openai = env_redirected_openai or bool(
                    effective_base_url
                    and not has_explicit_base_url
                    and not is_openai_provider_base_url(effective_base_url)
                )
                missing.append(
                    (
                        _model_backend_selector_path(model_path, model_config),
                        backend,
                        credential_hints,
                    )
                )
                continue
            if _has_config_api_key(backend, model_path, model_config):
                continue
            if backend == "nim" and is_local_base_url(model_config.get("base_url")):
                if _has_local_nim_api_key(model_config):
                    continue
                missing.append(
                    (
                        _model_backend_selector_path(model_path, model_config),
                        backend,
                        _LOCAL_NIM_CREDENTIAL_HINTS,
                    )
                )
                continue
            if backend == "nim" and model_config.get("base_url"):
                if _has_local_nim_api_key(model_config):
                    continue
                credential_hints = (
                    env_vars
                    if is_nvidia_provider_base_url(model_config.get("base_url"))
                    else ("explicit api_key in config",)
                )
                missing.append(
                    (
                        _model_backend_selector_path(model_path, model_config),
                        backend,
                        credential_hints,
                    )
                )
                continue
            if _has_env_api_key(env_vars):
                continue
            missing.append(
                (
                    _model_backend_selector_path(model_path, model_config),
                    backend,
                    env_vars,
                )
            )

    if not missing:
        return

    safe_config_path = redact_sensitive_path(effective_config_path)
    lines = [
        f"Config '{safe_config_path}' selects model backend(s) without required API keys:"
    ]
    for model_path, backend, env_vars in missing:
        env_text = " or ".join(env_vars)
        safe_model_path = redact_sensitive_config(model_path, _path_context=True)
        safe_backend = redact_sensitive_config(backend)
        lines.append(f"- {safe_model_path}={safe_backend!r} requires {env_text}")

    if env_redirected_openai:
        lines.append(OPENAI_ENV_REDIRECT_CREDENTIAL_MESSAGE)

    if any(
        backend == "nim" and "NVIDIA_API_KEY" in env_vars
        for _, backend, env_vars in missing
    ):
        lines.append(
            "The shipped unified_example.yaml defaults to backend: nim, so an "
            "unedited run requires NVIDIA_API_KEY."
        )
    lines.append(
        "Set the required key, edit the YAML backend/model fields, or set "
        "MA_VLM_BACKEND/MA_VLM_MODEL and MA_LLM_BACKEND/MA_LLM_MODEL before "
        "running."
    )
    lines.append(
        "OpenAI example: MA_VLM_BACKEND=openai MA_VLM_MODEL=example-vlm-model "
        "MA_LLM_BACKEND=openai MA_LLM_MODEL=example-vlm-model"
    )
    raise ValueError("\n".join(lines))


def _maybe_apply_backend_env_overrides(
    config_path: Path,
    config_data: dict[str, Any] | None = None,
) -> Path | dict[str, Any]:
    """Apply MA_VLM_* / MA_LLM_* env overrides to the pipeline config.

    CI jobs and local users commonly prepend ``MA_VLM_BACKEND=…`` to
    ``material-agent run`` expecting the config's VLM/LLM backend+model
    to be overridden at runtime (the service already honors these env
    vars). If any override is set, load the YAML, patch every step's
    ``vlm`` and ``llm`` subsections, and return the patched dictionary.
    Callers retain ``config_path`` as the relative-path anchor. When a caller
    already passed validated ``config_data``, return that mapping unchanged if
    no overrides are present so the CLI does not parse the same file twice.
    """
    overrides: dict[str, str] = {}
    for env_var, field in (
        (_ENV_OVERRIDE_VLM_BACKEND, ("vlm", "backend")),
        (_ENV_OVERRIDE_VLM_MODEL, ("vlm", "model")),
        (_ENV_OVERRIDE_LLM_BACKEND, ("llm", "backend")),
        (_ENV_OVERRIDE_LLM_MODEL, ("llm", "model")),
    ):
        value = os.environ.get(env_var, "").strip()
        if value:
            overrides[f"{field[0]}.{field[1]}"] = value

    if not overrides:
        return config_data if config_data is not None else config_path

    raw = config_data
    if raw is None:
        raw, _ = _load_preflight_config(config_path)
    if not isinstance(raw, dict):
        raise ValueError("Pipeline configuration must be a mapping")

    if "steps" not in raw:
        # Legacy pipeline configs store step mappings at the document root.
        steps = raw
    else:
        steps = raw["steps"]
    if steps is None:
        steps = {}
        if "steps" in raw:
            raw["steps"] = steps
    elif not isinstance(steps, dict):
        raise ValueError("Pipeline configuration 'steps' must be a mapping")
    for step_config in steps.values():
        if not isinstance(step_config, dict):
            continue
        for section in ("vlm", "llm"):
            section_config = step_config.get(section)
            if not isinstance(section_config, dict):
                continue
            original_backend = section_config.get("backend")
            for key in ("backend", "model"):
                env_key = f"{section}.{key}"
                if env_key in overrides:
                    section_config[key] = overrides[env_key]
            new_backend = section_config.get("backend")
            if new_backend and new_backend != original_backend:
                # Backend changed via env override; previous endpoint-scoped
                # fields belonged to the prior backend.
                drop_stale_endpoint_credentials(section_config)

    summary = ", ".join(
        f"{key}={redact_sensitive_config(value, _path_context=True)}"
        for key, value in overrides.items()
    )
    print(
        f"[yellow][run] Applied backend env overrides:[/yellow] {summary} "
        "[dim](in-memory config)[/dim]"
    )
    return raw


def _prepare_cli_config_payload(
    config_path: Path,
    logger: logging.Logger,
    config_data: dict[str, Any] | None = None,
) -> Path | dict[str, Any]:
    """Apply runtime overrides or terminate with a value-free CLI diagnostic."""
    try:
        return _maybe_apply_backend_env_overrides(config_path, config_data)
    except (OSError, ValueError) as exc:
        logger.error("Unable to load configuration: %s", exc)
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None


def _require_cli_config_file(
    config_path: Path,
    logger: logging.Logger,
    *,
    label: str = "Configuration",
) -> str:
    """Require a CLI config while keeping its runtime path out of diagnostics."""
    return _require_cli_path(
        config_path,
        logger,
        display_label=f"{label} file",
        inspection_label=f"{label.lower()} file",
    )


def _require_cli_path(
    path: Path,
    logger: logging.Logger,
    *,
    display_label: str,
    inspection_label: str | None = None,
) -> str:
    """Require a runtime path through one value-safe CLI diagnostic boundary."""
    safe_path = redact_sensitive_path(path)
    inspection_failed = False
    path_exists = False
    try:
        path_exists = path_exists_with_safe_diagnostics(
            path,
            label=inspection_label or display_label,
        )
    except OSError:
        message = f"Unable to inspect {inspection_label or display_label}: {safe_path}"
        logger.error("%s", message)
        console.print(f"[red]Error:[/red] {message}")
        inspection_failed = True

    if inspection_failed:
        # Raise outside the handler so a third-party path implementation
        # cannot retain credential-bearing exception context.
        raise typer.Exit(1) from None

    if not path_exists:
        message = f"{display_label} not found: {safe_path}"
        logger.error("%s", message)
        console.print(f"[red]Error:[/red] {message}")
        raise typer.Exit(1)
    return safe_path


def _report_cli_operation_failure(
    logger: logging.Logger,
    message: str,
) -> None:
    """Emit a fixed, value-free operation failure at the CLI boundary."""
    logger.error("%s", message)
    console.print(f"[red]Error:[/red] {message}")


def _get_cli_telemetry_session_id(session_id: str | None) -> str:
    """Get session identifier used for CLI telemetry tagging."""
    if session_id:
        return session_id
    return str(uuid.uuid4())


def setup_logging(
    verbose: bool = False,
    log_file: Path | None = None,
    log_level: str = "INFO",
) -> logging.Logger:
    """Setup logging configuration with Rich handler.

    This function now delegates to the shared logging utility.

    Args:
        verbose: Enable verbose output (sets DEBUG level)
        log_file: Optional path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    from world_understanding.agentic.cli import setup_logging as shared_setup_logging

    return shared_setup_logging(
        agent_name="material_agent",
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        print(
            f"[bold blue]Material Agent[/bold blue] version [green]{__version__}[/green]"
        )
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Material Agent - VLM-based material assignment for 3D objects.

    Use [bold]material-agent --help[/bold] to see available commands.
    """
    # Setup logging
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    # Initialize telemetry (reads from env vars via TelemetryConfig)
    # Telemetry is optional - failures are logged but don't crash the app
    telemetry_config = TelemetryConfig()
    tracer_provider = initialize_telemetry(telemetry_config)
    if tracer_provider is not None:
        logger.info(
            f"Telemetry initialized: enabled={telemetry_config.enabled}, "
            f"service={telemetry_config.service_name}, "
            f"exporters={telemetry_config.exporters}"
        )
        # Register shutdown handler for clean telemetry shutdown
        atexit.register(shutdown_telemetry)
    elif telemetry_config.enabled:
        logger.warning("Telemetry enabled but failed to initialize (check logs above)")
    else:
        logger.debug("Telemetry disabled via OTEL_ENABLED=false")

    # Store logger in app context for use in commands
    if not hasattr(app, "state"):
        app.state = {}
    app.state["logger"] = logger
    app.state["verbose"] = verbose

    if verbose:
        logger.debug("Verbose mode enabled")
        logger.debug(f"Log level: {log_level}")
        if log_file:
            logger.debug("Logging to file: %s", redact_sensitive_path(log_file))


@app.command()
@sever_cli_exception_graph
def benchmark(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to YAML configuration file",
        ),
    ],
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Override dataset path from config",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Override output directory from config",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume", help="Resume from existing predictions.jsonl"
        ),
    ] = False,
    stream_predictions: Annotated[
        bool,
        typer.Option(
            "--stream-predictions/--no-stream-predictions",
            help="Append predictions to predictions.jsonl as they are produced",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Run benchmarks to evaluate Material Agent performance.

    This command runs the Material Agent on a dataset of test cases
    and generates performance metrics including Functional Correctness Score (FCS).

    Example usage:
    ```bash
    # Using config file
    material-agent benchmark path/to/benchmark_config.yaml

    # Override dataset from command line
    material-agent benchmark path/to/benchmark_config.yaml --dataset data/custom.jsonl

    # Override output directory
    material-agent benchmark path/to/benchmark_config.yaml --output results/
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Material Agent Benchmark")

    safe_config_path = _require_cli_config_file(config, logger)

    config_path = config
    config_payload = _prepare_cli_config_payload(config_path, logger)

    console.print(
        Panel.fit(
            "[bold]Material Agent Benchmark[/bold]\n\n"
            f"Configuration: {safe_config_path}\n"
            f"Dataset override: {redact_sensitive_path(dataset)}\n"
            f"Output override: {redact_sensitive_path(output)}\n"
            f"Verbose mode: {'ON' if verbose else 'OFF'}",
            border_style="blue",
        )
    )

    logger.info("Configuration file: %s", safe_config_path)
    if dataset:
        logger.info("Dataset override: %s", redact_sensitive_path(dataset))
    if output:
        logger.info(
            "Output directory override: %s",
            redact_sensitive_path(output),
        )

    if verbose:
        logger.debug("Verbose mode enabled - detailed logging active")

    # Use API instead of directly creating workflow
    from material_agent.api import BenchmarkInput, run_benchmark

    try:
        # Create API parameters
        api_params = BenchmarkInput(
            config=config_payload,
            config_path=config_path if isinstance(config_payload, dict) else None,
            dataset_override=dataset,
            output_dir_override=output,
            resume=resume,
            stream_predictions=stream_predictions,
            verbose=verbose,
        )

        # Run benchmark via API
        logger.info("Running benchmark workflow...")
        console.print(
            "\n[cyan]Loading config, provisioning models, and running benchmark...[/cyan]"
        )

        result = run_benchmark(api_params)

        # Check if successful
        if result.success and result.metrics:
            logger.info("Benchmark completed successfully")

            # Display results using same format as evaluate command
            console.print("\n[bold green]Benchmark Results[/bold green]")
            console.print("=" * 50)

            # Create metrics table
            table = Table(title="Performance Metrics", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            # Use the metrics from API result
            metrics = result.metrics
            table.add_row(
                "Functional Correctness Score (FCS)",
                f"{metrics.functional_correctness_score}/5.0",
            )
            table.add_row(
                "Success Rate (Judge)",
                f"{metrics.success_rate}%",
            )
            table.add_row(
                "Exact Match Rate",
                f"{metrics.exact_match_rate}%",
            )
            table.add_row("Total Cases", str(metrics.total_cases))
            table.add_row("Valid Cases", str(metrics.valid_cases))
            table.add_row("Successful Cases (Judge)", str(metrics.successful_cases))
            table.add_row("Exact Matches", str(metrics.exact_matches))
            table.add_row("Failed Cases", str(metrics.failure_count))

            console.print(table)

            # Show score distribution if available
            if metrics.score_distribution:
                console.print("\n[cyan]Score Distribution:[/cyan]")
                for score, count in sorted(metrics.score_distribution.items()):
                    bar = "█" * count
                    console.print(f"  Score {score}: {bar} ({count})")

            console.print(
                "\n[bold green]✨ Benchmark completed successfully![/bold green]"
            )

            # Get output paths from API result
            if result.evaluation_path:
                safe_evaluation_path = redact_sensitive_path(result.evaluation_path)
                logger.info(
                    "Evaluation results saved to: %s",
                    safe_evaluation_path,
                )
                console.print(
                    f"[dim]Evaluation results saved to: {safe_evaluation_path}[/dim]"
                )
            if result.predictions_path:
                safe_predictions_path = redact_sensitive_path(result.predictions_path)
                logger.info("Predictions saved to: %s", safe_predictions_path)
                console.print(
                    f"[dim]Predictions saved to: {safe_predictions_path}[/dim]"
                )
        else:
            message = (
                MODEL_AUTHENTICATION_FAILURE_MESSAGE
                if is_model_authentication_error(result.error)
                else "Benchmark failed"
            )
            _report_cli_operation_failure(logger, message)
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as error:
        _report_cli_operation_failure(
            logger,
            public_model_failure_message(error, "Benchmark failed"),
        )
        raise typer.Exit(1) from None


@app.command()
def predict(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to unified YAML configuration file",
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Run material predictions on a dataset without evaluation.

    This is equivalent to: material-agent pipeline CONFIG --only predict

    Uses the unified configuration format where all paths are auto-derived from
    project.working_dir. The predict step will run VLM inference to predict materials.

    Example usage:
    ```bash
    material-agent predict apps/material_agent/configs/unified_example.yaml
    ```

    Output:
    - {working_dir}/predictions/predictions.jsonl: Material predictions with reasoning
    - {working_dir}/predictions/report.html: HTML report with visualizations
    """
    # This is just an alias for: pipeline --only predict
    return pipeline(
        config=config,
        skip=None,
        only="predict",
        resume=False,
        dry_run=False,
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


@app.command()
@sever_cli_exception_graph
def evaluate(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to evaluation configuration YAML file",
        ),
    ],
    predictions: Annotated[
        Path | None,
        typer.Argument(
            help="Path to predictions JSONL file to evaluate (overrides config)",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Evaluate existing predictions using an LLM judge.

    This command loads an evaluation configuration file and evaluates predictions
    against ground truth using the configured LLM judge. It calculates
    metrics including Functional Correctness Score (FCS) and success rate.

    The configuration file must specify:
    - predictions_path: Path to predictions JSONL file
    - llm_judge: LLM configuration for evaluation
    - dataset_path: Optional path to dataset for ground truth

    The predictions file must contain:
    - id: Entry identifier
    - materials: Predicted material assignments
    - ground_truth: Expected material assignments (or loaded from dataset)

    Example usage:
    ```bash
    # Evaluate using config file
    material-agent evaluate path/to/evaluation_config.yaml

    # Evaluate with predictions override
    material-agent evaluate path/to/evaluation_config.yaml output/predictions.jsonl
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Material Agent Evaluation")

    safe_config_path = _require_cli_config_file(config, logger)

    config_path = config
    config_payload = _prepare_cli_config_payload(config_path, logger)

    # Display evaluation info
    panel_content = "[bold]Material Agent Evaluation[/bold]\n\n"
    panel_content += f"Configuration: {safe_config_path}\n"
    if predictions:
        panel_content += f"Predictions override: {redact_sensitive_path(predictions)}\n"
    panel_content += f"Verbose mode: {'ON' if verbose else 'OFF'}"

    console.print(
        Panel.fit(
            panel_content,
            border_style="blue",
        )
    )

    logger.info("Configuration file: %s", safe_config_path)
    if predictions:
        # Validate predictions file exists if provided as override
        safe_predictions_path = _require_cli_path(
            predictions,
            logger,
            display_label="Predictions file",
            inspection_label="predictions file",
        )
        logger.info(
            "Predictions override: %s",
            safe_predictions_path,
        )

    if verbose:
        logger.debug("Verbose mode enabled - detailed logging active")

    # Use API instead of directly creating workflow
    from material_agent.api import EvaluateInput, run_evaluate

    try:
        # Create API parameters
        api_params = EvaluateInput(
            config=config_payload,
            config_path=config_path if isinstance(config_payload, dict) else None,
            predictions_override=predictions,
            verbose=verbose,
        )

        # Run evaluation via API
        logger.info("Running evaluation...")
        console.print(
            "\n[cyan]Loading config, provisioning LLM judge, and evaluating predictions...[/cyan]"
        )

        result = run_evaluate(api_params)

        # Check if evaluation was successful
        if result.success and result.metrics:
            metrics = result.metrics

            # Display results
            console.print("\n[bold green]Evaluation Results[/bold green]")
            console.print("=" * 50)

            # Create metrics table
            table = Table(title="Performance Metrics", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            # Use the metrics from API result
            table.add_row(
                "Functional Correctness Score (FCS)",
                f"{metrics.functional_correctness_score}/5.0",
            )
            table.add_row(
                "Success Rate (Judge)",
                f"{metrics.success_rate}%",
            )
            table.add_row(
                "Exact Match Rate",
                f"{metrics.exact_match_rate}%",
            )
            table.add_row("Total Cases", str(metrics.total_cases))
            table.add_row("Valid Cases", str(metrics.valid_cases))
            table.add_row("Successful Cases (Judge)", str(metrics.successful_cases))
            table.add_row("Exact Matches", str(metrics.exact_matches))
            table.add_row("Failed Cases", str(metrics.failure_count))

            console.print(table)

            # Show score distribution if available
            if metrics.score_distribution:
                console.print("\n[cyan]Score Distribution:[/cyan]")
                for score, count in sorted(metrics.score_distribution.items()):
                    bar = "█" * count
                    console.print(f"  Score {score}: {bar} ({count})")

            console.print(
                "\n[bold green]✨ Evaluation completed successfully![/bold green]"
            )

            if result.evaluation_path:
                safe_evaluation_path = redact_sensitive_path(result.evaluation_path)
                logger.info(
                    "Evaluation results saved to: %s",
                    safe_evaluation_path,
                )
                console.print(f"Evaluation results saved to: {safe_evaluation_path}")

            # Display HTML report path if generated
            if result.html_report_path:
                safe_html_report_path = redact_sensitive_path(result.html_report_path)
                console.print(
                    f"[cyan]HTML report generated: {safe_html_report_path}[/cyan]"
                )
        else:
            message = (
                MODEL_AUTHENTICATION_FAILURE_MESSAGE
                if is_model_authentication_error(result.error)
                else "Evaluation failed"
            )
            _report_cli_operation_failure(logger, message)
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as error:
        _report_cli_operation_failure(
            logger,
            public_model_failure_message(error, "Evaluation failed"),
        )
        raise typer.Exit(1) from None


# Create a sub-app for build-dataset commands
build_dataset_app = typer.Typer(
    name="build-dataset",
    help="Commands for building datasets from various sources",
    rich_markup_mode="rich",
)

# Add build-dataset as a command group to the main app
app.add_typer(build_dataset_app, name="build-dataset")


@build_dataset_app.command(name="pdf_vectorstore")
@sever_cli_exception_graph
def build_pdf_vectorstore(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to YAML configuration file",
        ),
    ],
    source: Annotated[
        Path | None,
        typer.Option(
            "--source",
            "-s",
            help="Override source path from config (PDF file or directory)",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Override output directory from config",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Build a multimodal vector store from PDF documents.

    This command processes PDF files to extract content (text, images, tables),
    splits them by type, and creates a searchable vector store.

    Example usage:
    ```bash
    # Using config file
    material-agent build-dataset pdf_vectorstore path/to/pdf_vectorstore_config.yaml

    # Override source path
    material-agent build-dataset pdf_vectorstore path/to/pdf_vectorstore_config.yaml --source docs/

    # Override output directory
    material-agent build-dataset pdf_vectorstore path/to/pdf_vectorstore_config.yaml --output ./vectorstore/
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting PDF to VectorStore workflow")

    safe_config_path = _require_cli_config_file(config, logger)

    # Display configuration info
    console.print(
        Panel.fit(
            "[bold]PDF to VectorStore Builder[/bold]\n\n"
            f"Configuration: {safe_config_path}\n"
            f"Source override: {redact_sensitive_path(source)}\n"
            f"Output override: {redact_sensitive_path(output)}\n"
            f"Verbose mode: {'ON' if verbose else 'OFF'}",
            border_style="blue",
        )
    )

    logger.info("Configuration file: %s", safe_config_path)
    if source:
        logger.info("Source override: %s", redact_sensitive_path(source))
    if output:
        logger.info(
            "Output directory override: %s",
            redact_sensitive_path(output),
        )

    if verbose:
        logger.debug("Verbose mode enabled - detailed logging active")

    # Use API for PDF vectorstore building
    from material_agent.api import (
        BuildDatasetPdfVectorstoreInput,
        build_dataset_pdf_vectorstore,
    )

    try:
        # Create API parameters
        api_params = BuildDatasetPdfVectorstoreInput(
            config=config,
            source_override=source,
            output_dir_override=output,
            verbose=verbose,
        )

        # Run the workflow via API
        console.print("\n[cyan]Processing PDFs and building vector store...[/cyan]")

        result = build_dataset_pdf_vectorstore(api_params)

        # Check if workflow completed successfully
        if result.success:
            logger.info("PDF vectorstore workflow completed successfully")

            # Display results
            console.print(
                "\n[bold green]✨ Vector store created successfully![/bold green]"
            )

            # Show extraction results if available
            if result.extraction_result:
                extraction = result.extraction_result
                console.print("\n[bold]Extraction Results:[/bold]")
                console.print(
                    f"  • Documents processed: {extraction.get('document_count', 0)}"
                )
                if "content_types" in extraction:
                    console.print(
                        "  • Content types: "
                        f"{redact_sensitive_config(extraction['content_types'])}"
                    )

            # Show split results if available
            if result.split_result:
                split = result.split_result
                console.print("\n[bold]Content Split Results:[/bold]")
                console.print(
                    f"  • Files created: {split.get('total_files_created', 0)}"
                )
                if "content_type_distribution" in split:
                    console.print(
                        "  • Distribution: "
                        f"{redact_sensitive_config(split['content_type_distribution'])}"
                    )

            # Show vectorstore results
            console.print("\n[bold]Vector Store Results:[/bold]")
            console.print(f"  • Documents indexed: {result.num_documents_indexed}")
            console.print(f"  • Text documents: {result.num_texts}")
            console.print(f"  • Image documents: {result.num_images}")
            console.print(f"  • Embedding dimension: {result.embedding_dimension}")
            if result.vectorstore_path:
                console.print(
                    f"  • Saved to: {redact_sensitive_path(result.vectorstore_path)}"
                )
        else:
            _report_cli_operation_failure(
                logger,
                "PDF vectorstore workflow failed",
            )
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception:
        _report_cli_operation_failure(
            logger,
            "PDF vectorstore workflow failed",
        )
        raise typer.Exit(1) from None


@build_dataset_app.command(name="prepare-dataset")
@sever_cli_exception_graph
def prepare_dataset(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to YAML configuration file",
        ),
    ],
    vector_store: Annotated[
        Path | None,
        typer.Option(
            "--vector-store",
            help="Override vector store path from config",
        ),
    ] = None,
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Override dataset path from config",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Prepare dataset with CMF specifications for benchmark or prediction.

    This command prepares datasets by extracting CMF specifications
    for model numbers using a document vector store. It can prepare either
    benchmark datasets (with ground truth) or prediction datasets (without ground truth).

    Example usage:
    ```bash
    # Using config file
    material-agent build-dataset prepare-dataset path/to/prepare_dataset.yaml

    # Override vector store and dataset paths
    material-agent build-dataset prepare-dataset path/to/prepare_dataset.yaml \
      --vector-store ./vectorstore --dataset ./data/prepared_dataset
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting prepare dataset workflow")

    safe_config_path = _require_cli_config_file(config, logger)

    # Display configuration info
    console.print(
        Panel.fit(
            "[bold]Prepare Dataset[/bold]\n\n"
            f"Configuration: {safe_config_path}\n"
            f"Vector Store Override: {redact_sensitive_path(vector_store)}\n"
            f"Dataset Override: {redact_sensitive_path(dataset)}\n"
            f"Models: Auto-discovered from dataset\n"
            f"Output: dataset.jsonl saved to dataset directory\n"
            f"Verbose mode: {'ON' if verbose else 'OFF'}",
            border_style="blue",
        )
    )

    logger.info("Configuration file: %s", safe_config_path)
    if vector_store:
        logger.info(
            "Vector store override: %s",
            redact_sensitive_path(vector_store),
        )
    if dataset:
        logger.info("Dataset override: %s", redact_sensitive_path(dataset))

    if verbose:
        logger.debug("Verbose mode enabled - detailed logging active")

    # Use API for dataset preparation
    from material_agent.api import (
        BuildDatasetPrepareDatasetInput,
        build_dataset_prepare_dataset,
    )

    try:
        # Create API parameters
        api_params = BuildDatasetPrepareDatasetInput(
            config=config,
            vector_store_override=vector_store,
            dataset_override=dataset,
            verbose=verbose,
        )

        # Run the workflow via API
        logger.info("Running prepare dataset workflow...")
        console.print(
            "\n[cyan]Loading config, provisioning LLM, and preparing benchmark data...[/cyan]"
        )

        result = build_dataset_prepare_dataset(api_params)

        # Check if workflow completed successfully
        if result.success:
            dataset_entries = result.dataset_entries
            failed_models = result.failed_models
            dataset_jsonl_path = result.dataset_jsonl_path

            console.print(
                "\n[bold green]✨ Dataset preparation completed![/bold green]"
            )
            console.print(f"  • Dataset entries: {len(dataset_entries)}")
            console.print(f"  • Failed models: {len(failed_models)}")
            console.print(
                f"  • Dataset saved to: {redact_sensitive_path(dataset_jsonl_path)}"
            )

            if failed_models:
                safe_failed_models = redact_sensitive_config(failed_models)
                console.print(
                    f"[yellow]Failed models: {', '.join(safe_failed_models)}[/yellow]"
                )
                logger.info("Failed models: %s", safe_failed_models)
        else:
            _report_cli_operation_failure(logger, "Prepare dataset failed")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception:
        _report_cli_operation_failure(logger, "Prepare dataset failed")
        raise typer.Exit(1) from None


@build_dataset_app.command(name="usd")
@sever_cli_exception_graph
def usd(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to the data preparation configuration file.",
        ),
    ],
    source: Annotated[
        Path | None,
        typer.Option(
            "--source",
            "-s",
            help="Path to the USD file or directory (overrides config).",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for dataset (overrides config).",
        ),
    ] = None,
    extract_metadata: Annotated[
        bool,
        typer.Option(
            "--extract-metadata/--no-extract-metadata",
            help="Extract prim metadata (materials, transforms, etc.).",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Build a dataset from USD file(s) by rendering views of each prim.

    This command will intelligently handle both single file and batch processing:
    - If config has 'usd_path': processes a single USD file
    - If config has 'usd_dir': processes all USD files in that directory

    For batch processing, subdirectories will be created for each USD file.

    Example usage:
    ```bash
    # Single file config (with usd_path)
    material-agent build-dataset usd path/to/single_usd.yaml

    # Batch processing config (with usd_dir)
    material-agent build-dataset usd path/to/usd_batch.yaml

    # Override source (file or directory)
    material-agent build-dataset usd path/to/data_prep.yaml \\
        --source path/to/file_or_dir

    # With metadata extraction
    material-agent build-dataset usd path/to/data_prep.yaml \\
        --extract-metadata
    ```
    """
    # Setup logging
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Material Agent Dataset Build Workflow")

    _require_cli_config_file(config, logger)

    # Load config to determine if it's single file or batch processing
    try:
        config_data, _ = _load_preflight_config(config)
    except (OSError, ValueError) as error:
        logger.error("Failed to load configuration: %s", error)
        raise typer.Exit(code=1) from None
    if not isinstance(config_data, dict):
        logger.error("Configuration must contain a mapping")
        raise typer.Exit(code=1) from None

    # Determine if source override points to a directory or file
    is_batch_mode = False
    source_is_directory = False
    if source:
        source = Path(source)
        try:
            source_is_directory = source.is_dir()
        except OSError:
            logger.error(
                "Unable to inspect USD source override: %s",
                redact_sensitive_path(source),
            )
            raise typer.Exit(code=1) from None
        if source_is_directory:
            is_batch_mode = True
    elif "usd_dir" in config_data:
        is_batch_mode = True
    elif "usd_path" not in config_data:
        # Neither usd_path nor usd_dir specified
        logger.error(
            "Configuration must contain either 'usd_path' (for single file) "
            "or 'usd_dir' (for batch processing)"
        )
        raise typer.Exit(code=1)

    # Handle batch processing
    if is_batch_mode:
        logger.info("Detected batch processing mode")

        # Get USD directory
        if source and source_is_directory:
            usd_dir = source
            logger.info(
                "Using USD directory override: %s",
                redact_sensitive_path(usd_dir),
            )
        elif "usd_dir" in config_data:
            # Resolve path relative to config file location
            usd_dir = _resolve_config_relative_path(config_data["usd_dir"], config)
            logger.info(
                "Using usd_dir from config: %s",
                redact_sensitive_path(usd_dir),
            )
        else:  # pragma: no cover - unreachable defensive branch
            logger.error("Batch mode requires usd_dir in config or --source directory")
            raise typer.Exit(code=1)

        # Get output directory
        if output_dir:
            batch_output_dir = output_dir
        elif "output_dir" in config_data:
            batch_output_dir = _resolve_config_relative_path(
                config_data["output_dir"],
                config,
            )
        else:
            batch_output_dir = Path("output")

        # Check if USD directory exists
        _require_cli_path(
            usd_dir,
            logger,
            display_label="USD directory",
        )

        # Use API for batch processing
        from material_agent.api import BuildDatasetUsdInput, build_dataset_usd

        try:
            # Create API parameters
            api_params = BuildDatasetUsdInput(
                config=config,
                source_override=usd_dir,
                output_dir_override=batch_output_dir,
                extract_metadata=extract_metadata,
                verbose=verbose,
            )

            # Run via API
            api_result = build_dataset_usd(api_params)

            if not api_result.success:
                raise RuntimeError("Batch processing failed")

            results = api_result.batch_results
            successful_builds = sum(
                1 for r in results.values() if r.get("status") == "success"
            )
            failed_builds = sum(
                1 for r in results.values() if r.get("status") != "success"
            )

        except Exception:
            logger.error("Batch processing failed")
            console.print("[red]Error:[/red] Batch processing failed")
            raise typer.Exit(code=1) from None

        # Display batch results
        table = Table(title="Batch Dataset Build Results", show_header=True)
        table.add_column("USD File", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Prims", justify="right")
        table.add_column("Images", justify="right")
        table.add_column("Output Directory", style="dim")

        for usd_name, result in results.items():
            status = "✓ Success" if result["status"] == "success" else "✗ Failed"
            status_style = "green" if result["status"] == "success" else "red"

            prims = str(redact_sensitive_config(result.get("num_prims", "N/A")))
            images = str(redact_sensitive_config(result.get("num_images", "N/A")))
            output_path = redact_sensitive_path(Path(result["output_dir"]).name)

            table.add_row(
                redact_sensitive_path(usd_name),
                f"[{status_style}]{status}[/{status_style}]",
                prims,
                images,
                output_path,
            )

        console.print("\n")
        console.print(table)
        console.print("\n")

        if failed_builds == 0:
            console.print(
                Panel.fit(
                    "[bold green]✓[/bold green] All datasets built successfully!",
                    border_style="green",
                )
            )
        elif successful_builds > 0:
            console.print(
                Panel.fit(
                    f"[bold yellow]⚠[/bold yellow] Completed with {failed_builds} failures",
                    border_style="yellow",
                )
            )
        else:
            console.print(
                Panel.fit(
                    "[bold red]✗[/bold red] All builds failed",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

    else:
        # Single file processing
        logger.info("Processing single USD file")

        try:
            from material_agent.api import BuildDatasetUsdInput, build_dataset_usd

            # Create API parameters
            api_params = BuildDatasetUsdInput(
                config=config,
                source_override=source,
                output_dir_override=output_dir,
                extract_metadata=extract_metadata,
                verbose=verbose,
            )

            # Run workflow via API
            logger.info("Executing dataset build workflow")
            result = build_dataset_usd(api_params)

            if not result.success:
                raise RuntimeError("Dataset build failed")

            # Create results table
            table = Table(title="Dataset Build Results", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row(
                "Dataset Manifest",
                (
                    redact_sensitive_path(result.dataset_path)
                    if result.dataset_path
                    else "N/A"
                ),
            )
            table.add_row(
                "Total Prims",
                str(redact_sensitive_config(result.num_prims)),
            )
            table.add_row(
                "Total Images",
                str(redact_sensitive_config(result.num_images)),
            )

            console.print("\n")
            console.print(table)
            console.print("\n")

            console.print(
                Panel.fit(
                    "[bold green]✓[/bold green] Dataset build completed successfully!",
                    border_style="green",
                )
            )

        except Exception:
            logger.error("Dataset build failed")
            console.print("[red]Error:[/red] Dataset build failed")
            raise typer.Exit(code=1) from None


@app.command()
def apply(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to unified YAML configuration file",
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Apply predicted materials to a USD file.

    This is equivalent to: material-agent pipeline CONFIG --only apply

    Uses the unified configuration format where all paths are auto-derived from
    project.working_dir. The apply step will apply predicted materials to the USD file.

    Example usage:
    ```bash
    material-agent apply apps/material_agent/configs/unified_example.yaml
    ```

    Output:
    - output.usd_path: USD file with materials applied (as specified in config)
    """
    # This is just an alias for: pipeline --only apply
    return pipeline(
        config=config,
        skip=None,
        only="apply",
        resume=False,
        dry_run=False,
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


# Keep old functions below for reference during migration
# These will be deleted after migration is complete
def _legacy_apply(
    config: Path,
    input_usd: Path | None,
    predictions: Path | None,
    output: Path | None,
    layer_only: bool,
    render: bool,
    verbose: bool,
    log_file: Path | None,
    log_level: str,
) -> None:
    """Legacy apply implementation - for migration reference only."""
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Material Agent Apply")

    safe_config_path = _require_cli_config_file(config, logger)

    # Display configuration info
    console.print(
        Panel.fit(
            "[bold]Material Agent Apply[/bold]\n\n"
            f"Configuration: {safe_config_path}\n"
            f"Input USD override: {redact_sensitive_path(input_usd)}\n"
            f"Predictions override: {redact_sensitive_path(predictions)}\n"
            f"Output override: {redact_sensitive_path(output)}\n"
            f"Output mode: {'Layer only' if layer_only else 'Full stage'}\n"
            f"Verbose mode: {'ON' if verbose else 'OFF'}",
            border_style="blue",
        )
    )

    logger.info("Configuration file: %s", safe_config_path)
    if input_usd:
        logger.info("Input USD override: %s", redact_sensitive_path(input_usd))
    if predictions:
        logger.info(
            "Predictions override: %s",
            redact_sensitive_path(predictions),
        )
    if output:
        logger.info("Output override: %s", redact_sensitive_path(output))

    if verbose:
        logger.debug("Verbose mode enabled - detailed logging active")

    # Import workflow factory
    from material_agent.workflows.factory import create_apply_workflow_from_config

    # Create config-driven apply workflow
    try:
        logger.info("Creating config-driven apply workflow...")
        workflow = create_apply_workflow_from_config()
        console.print("[green]✓ Config-driven apply workflow created[/green]")
    except Exception:
        _report_cli_operation_failure(logger, "Unable to create apply workflow")
        raise typer.Exit(1) from None

    # Run the apply workflow
    try:
        logger.info("Running material application...")
        console.print(
            "\n[cyan]Loading config, identifying materials, and applying to USD...[/cyan]"
        )

        # Prepare initial context with config path and overrides
        initial_context = {
            "config_path": str(config),
            "input_usd_override": str(input_usd) if input_usd else None,
            "predictions_override": str(predictions) if predictions else None,
            "output_usd_override": str(output) if output else None,
            "layer_only": layer_only,  # Pass layer_only flag
            "render_enabled": render,  # Pass render flag
            "verbose": verbose,
        }

        result = workflow.run(initial_context=initial_context)

        # Check if application was successful
        if result.get("application_complete"):
            unique_materials = result.get("unique_materials", [])
            matched_materials = result.get("matched_materials", {})
            materials_applied = result.get("materials_applied", {})
            assignment_stats = result.get("assignment_stats", {})
            output_path = result.get("output_usd_path")
            layer_only = result.get("layer_only", False)
            result.get("rendered_image_path")
            rendered_images = result.get("rendered_image_paths", [])
            rendering_skipped = result.get("rendering_skipped", True)

            output_message = (
                f"\n[bold green]✨ Material application complete![/bold green]\n"
                f"  • Unique materials found: {len(unique_materials)}\n"
                f"  • Materials matched via USD Search: {len(matched_materials)}\n"
                f"  • Materials applied to USD: {len(materials_applied)}\n"
                f"  • Prims with materials: {assignment_stats.get('total_prims', 0)}\n"
                f"  • Output mode: {'Layer only' if layer_only else 'Full stage'}\n"
                f"  • Output USD file: {redact_sensitive_path(output_path)}"
            )

            # Add rendering information if enabled
            if not rendering_skipped and rendered_images:
                if len(rendered_images) == 1:
                    output_message += (
                        "\n  • Rendered image: "
                        f"{redact_sensitive_path(rendered_images[0])}"
                    )
                else:
                    output_message += (
                        f"\n  • Rendered images ({len(rendered_images)} views):"
                    )
                    for img_path in rendered_images:
                        output_message += f"\n    - {redact_sensitive_path(img_path)}"

            console.print(output_message)

            # Display material search results
            if matched_materials:
                console.print("\n[cyan]Material Search Results:[/cyan]")
                for material, path_infos in matched_materials.items():
                    safe_material = redact_sensitive_config(material)
                    console.print(
                        f"  • {safe_material}: {len(path_infos)} matches found"
                    )
                    # Always show first match details if available
                    if path_infos and len(path_infos) > 0:
                        path_info = path_infos[0]
                        if isinstance(path_info, dict):
                            if path_info.get("source_path"):
                                console.print(
                                    "    Source: "
                                    f"{redact_sensitive_path(path_info['source_path'])}"
                                )
                            if path_info.get("s3_path"):
                                console.print(
                                    "    S3:     "
                                    f"{redact_sensitive_path(path_info['s3_path'])}"
                                )
                        else:
                            # Fallback for old format
                            console.print(f"    - {redact_sensitive_path(path_info)}")
                    # Show more matches in verbose mode
                    if verbose and len(path_infos) > 1:
                        for i, path_info in enumerate(
                            path_infos[1:3], start=2
                        ):  # Show next 2 paths
                            console.print(f"    [{i}]")
                            if isinstance(path_info, dict):
                                if path_info.get("source_path"):
                                    console.print(
                                        "        Source: "
                                        f"{redact_sensitive_path(path_info['source_path'])}"
                                    )
                                if path_info.get("s3_path"):
                                    console.print(
                                        "        S3:     "
                                        f"{redact_sensitive_path(path_info['s3_path'])}"
                                    )
                            else:
                                # Fallback for old format
                                console.print(
                                    f"        - {redact_sensitive_path(path_info)}"
                                )
                        if len(path_infos) > 3:
                            console.print(f"    ... and {len(path_infos) - 3} more")

            # Display resolved material files
            resolved_materials = result.get("resolved_materials", {})
            download_stats = result.get("download_stats", {})

            if resolved_materials:
                console.print("\n[cyan]Resolved Material Files:[/cyan]")
                for material, local_path in resolved_materials.items():
                    safe_material = redact_sensitive_config(material)
                    # Check if it's a local file or S3 path
                    if local_path.startswith("s3://"):
                        console.print(
                            f"  • {safe_material}: [yellow]S3[/yellow] "
                            f"{redact_sensitive_path(local_path)}"
                        )
                    else:
                        console.print(
                            f"  • {safe_material}: [green]Local[/green] "
                            f"{redact_sensitive_path(local_path)}"
                        )

                # Show download statistics
                if download_stats:
                    console.print("\n[cyan]Resolution Statistics:[/cyan]")
                    console.print(
                        f"  • Found locally: {download_stats.get('found_local', 0)}"
                    )
                    console.print(
                        f"  • Downloaded from S3: {download_stats.get('downloaded', 0)}"
                    )
                    console.print(f"  • Failed: {download_stats.get('failed', 0)}")
                    console.print(f"  • Skipped: {download_stats.get('skipped', 0)}")

            # Display USD assignment results
            if materials_applied:
                console.print("\n[cyan]USD Material Assignment:[/cyan]")
                console.print(
                    f"  • Materials created: {assignment_stats.get('materials_created', 0)}"
                )
                console.print(
                    f"  • Materials applied: {assignment_stats.get('materials_applied', 0)}"
                )
                console.print(
                    f"  • Prims updated: {assignment_stats.get('total_prims', 0)}"
                )
                console.print(
                    f"  • Failed assignments: {assignment_stats.get('failed', 0)}"
                )

            logger.info(
                "Material application saved to: %s",
                redact_sensitive_path(output_path),
            )
        else:
            logger.error("Apply workflow did not complete successfully")
            console.print("[red]Error:[/red] Apply workflow did not complete")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception:
        _report_cli_operation_failure(logger, "Material apply workflow failed")
        raise typer.Exit(1) from None


@app.command()
@sever_cli_exception_graph
def refine(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to YAML configuration file",
        ),
    ],
    max_iterations: Annotated[
        int | None,
        typer.Option(
            "--max-iterations",
            "-n",
            help="Override maximum number of iterations from config",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Refine materials on USD with VLM-based iterative refinement.

    This command executes a predict-apply-judge loop repeatedly until the judge
    approves the results or maximum iterations is reached. It uses VLM to predict
    materials, applies them to USD, renders the result, and has a VLM judge evaluate
    quality by comparing against reference images.

    The configuration file must specify:
    - dataset: Path to the dataset JSONL file
    - input_usd_path: Path to the input USD file
    - output_usd_path: Path for the final output (optional)
    - iteration: Iteration settings (max_iterations, save_intermediate, etc.)
    - judge: Judge configuration (reference_images, vlm settings, etc.)

    Example usage:
    ```bash
    # Run material refinement with iterative predict-apply-judge loop
    material-agent refine path/to/refine_config.yaml

    # Override max iterations
    material-agent refine path/to/refine_config.yaml --max-iterations 3
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Material Agent Material Refinement")

    _require_cli_config_file(config, logger)

    config_path = config
    config_payload = _prepare_cli_config_payload(config_path, logger)

    # Run the material refinement workflow using API
    try:
        from material_agent.api import RefineInput, run_refine

        logger.info("Running material refinement with iterative loop...")
        console.print(
            "\n[cyan]Starting iterative predict-apply-judge workflow...[/cyan]"
        )

        # Create API parameters
        api_params = RefineInput(
            config=config_payload,
            config_path=config_path if isinstance(config_payload, dict) else None,
            max_iterations_override=max_iterations,
            verbose=verbose,
        )

        result = run_refine(api_params)

        # Check if successful
        if result.success and result.iteration_count > 0:
            iteration_count = result.iteration_count
            termination_reason = result.termination_reason
            final_score = result.final_judge_score
            final_output_path = result.final_output_path

            final_score_str = f"{final_score:.2f}" if final_score is not None else "N/A"

            # Get materials info from last iteration
            final_materials_applied = 0
            final_prims_with_materials = 0
            if result.iteration_results:
                last_iter = result.iteration_results[-1]
                final_materials_applied = last_iter.materials_applied_count
                final_prims_with_materials = last_iter.prims_with_materials

            console.print(
                f"\n[bold green]Iterative material refinement complete![/bold green]\n"
                f"  • Total iterations: {iteration_count}\n"
                f"  • Termination reason: {termination_reason}\n"
                f"  • Final judge score: {final_score_str}\n"
                f"  • Final materials applied: {final_materials_applied}\n"
                f"  • Final prims with materials: {final_prims_with_materials}"
            )

            if final_output_path:
                console.print(
                    "\n[bold cyan]Final Output:[/bold cyan]\n  "
                    f"{redact_sensitive_path(final_output_path)}"
                )

            if result.all_iteration_outputs:
                console.print("\n[cyan]Iteration Outputs:[/cyan]")
                for i, output_path in enumerate(result.all_iteration_outputs, 1):
                    console.print(f"  [{i}] {redact_sensitive_path(output_path)}")

            if result.iteration_results:
                console.print("\n[cyan]Iteration Summary:[/cyan]")
                for iter_result in result.iteration_results:
                    iter_num = iter_result.iteration
                    score = iter_result.judge_score
                    decision = (
                        "CONTINUE" if iter_result.continue_iteration else "APPROVE"
                    )
                    score_str = f"{score:.2f}" if score is not None else "N/A"
                    console.print(
                        f"  • Iteration {iter_num}: Score={score_str}, Decision={decision}"
                    )

            logger.info(
                f"Material refinement completed after {iteration_count} iterations"
            )
        else:
            _report_cli_operation_failure(logger, "Material refinement failed")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception:
        _report_cli_operation_failure(logger, "Material refinement failed")
        raise typer.Exit(1) from None


@app.command()
@sever_cli_exception_graph
def run(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to unified YAML configuration file",
        ),
    ],
    skip: Annotated[
        str | None,
        typer.Option(
            "--skip",
            help="Comma-separated list of steps to skip",
        ),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(
            "--only",
            help="Comma-separated list of steps to run exclusively",
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="Reuse existing session ID instead of generating a new one",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Resume from last successful checkpoint",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show pipeline plan without executing",
        ),
    ] = False,
    clean: Annotated[
        bool,
        typer.Option(
            "--clean",
            help="Clean (delete) working directory and output files (USD + renders) before starting",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Execute a multi-step material agent pipeline.

    Uses the unified configuration format where all paths are auto-derived from
    project.working_dir, input.usd_path, and output.usd_path.

    A typical pipeline includes:
    1. build_dataset_usd: Build dataset from USD files
    2. build_dataset_pdf_vectorstore: Build vector store from PDFs (optional)
    3. build_dataset_prepare_dataset: Prepare dataset with specifications
    4. predict/benchmark: Run VLM inference
    5. apply: Apply predicted materials to USD

    The pipeline automatically connects outputs from one step to inputs of the next.

    Example usage:
    ```bash
    # Run complete pipeline
    material-agent run apps/material_agent/configs/unified_example.yaml

    # Skip USD dataset building (already exists)
    material-agent run apps/material_agent/configs/unified_example.yaml --skip build_dataset_usd

    # Run only prediction and apply steps
    material-agent run apps/material_agent/configs/unified_example.yaml --only predict,apply

    # Dry run to see execution plan
    material-agent run apps/material_agent/configs/unified_example.yaml --dry-run
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    # Get event listener for CLI output
    listener = get_listener({}, logger_name="material_agent.cli")

    logger.info("Starting Material Agent Pipeline")

    safe_config_path = _require_cli_config_file(
        config,
        logger,
        label="Pipeline configuration",
    )

    # Reject invalid filters before config tasks, output creation, credential
    # probes, or backend work.
    try:
        from material_agent.config.schema import STEP_ORDER

        skip_steps, only_steps = normalize_cli_step_filters(
            skip=skip,
            only=only,
            valid_steps=STEP_ORDER,
        )
    except ValueError as error:
        logger.error("Pipeline step filter validation failed: %s", error)
        console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(1) from None

    try:
        config_data = load_cli_config_mapping(config)
    except (OSError, ValueError) as error:
        logger.error("Pipeline configuration validation failed: %s", error)
        console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(1) from None

    # Apply MA_VLM_* / MA_LLM_* env-var overrides if any are set. The service
    # honours these env vars via its own config path; doing the same here
    # keeps CLI and service behaviour in sync and lets CI jobs redirect a
    # public-defaults config to another backend without editing YAML.
    config_path = config
    config_payload = _prepare_cli_config_payload(config_path, logger, config_data)

    # Display configuration info via event system
    listener.event(
        "pipeline.config.display",
        {
            "config": safe_config_path,
            "skip_steps": skip_steps,
            "only_steps": only_steps,
            "resume": resume,
            "dry_run": dry_run,
            "clean": clean,
        },
    )

    if not dry_run:
        try:
            _validate_run_config_windows_prerequisites(
                config_payload,
                skip_steps,
                only_steps,
                resume=resume,
                clean=clean,
                session_id=session_id,
                source_config_path=config_path,
            )
            _validate_run_config_model_credentials(
                config_payload,
                skip_steps,
                only_steps,
                resume=resume,
                clean=clean,
                session_id=session_id,
                source_config_path=config_path,
            )
        except ValueError as e:
            logger.error("Pipeline configuration validation failed: %s", e)
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from e

    if dry_run:
        # Load config and display plan without executing
        try:
            pipeline_config, _ = _load_preflight_config(
                config_payload,
                source_config_path=config_path,
            )

            console.print("\n[bold cyan]Pipeline Execution Plan:[/bold cyan]\n")

            # Detect config format (unified vs old)
            is_unified = "project" in pipeline_config

            if is_unified:
                # Unified config format
                project_name = pipeline_config.get("project", {}).get("name", "unknown")
                working_dir = pipeline_config.get("project", {}).get(
                    "working_dir", f".{project_name}"
                )

                console.print(
                    "[cyan]Project:[/cyan] "
                    f"{redact_sensitive_config(project_name, _path_context=True)}"
                )
                console.print(
                    "[cyan]Working Directory:[/cyan] "
                    f"{redact_sensitive_path(working_dir)}"
                )
                console.print(
                    "[cyan]Input USD:[/cyan] "
                    f"{redact_sensitive_path((pipeline_config.get('input') or {}).get('usd_path', 'N/A'))}"
                )
                console.print(
                    "[cyan]Output USD:[/cyan] "
                    f"{redact_sensitive_path((pipeline_config.get('output') or {}).get('usd_path', 'N/A'))}\n"
                )

                steps_section = pipeline_config.get("steps", {})
            else:
                # Old config format
                steps_section = pipeline_config

            # Use centralized step names
            from material_agent.api.defaults import PIPELINE_STEP_NAMES

            step_names = PIPELINE_STEP_NAMES

            table = Table(title="Steps", show_header=True)
            table.add_column("Step", style="cyan")
            table.add_column("Status", style="yellow")
            table.add_column("Enabled", style="green")

            for step in step_names:
                if step not in steps_section:
                    continue

                step_config = steps_section[step]

                # Check if enabled (for unified format)
                if is_unified:
                    enabled = step_config.get("enabled")
                    if enabled is None:
                        # Implicitly enable if step has any configuration besides 'enabled'
                        has_config = any(k != "enabled" for k in step_config.keys())
                        enabled = has_config
                    if not enabled:
                        continue

                if skip_steps and step in skip_steps:
                    status = "⊘ Skipped"
                    style_name = "dim"
                elif only_steps and step not in only_steps:
                    status = "⊘ Excluded"
                    style_name = "dim"
                else:
                    status = "→ Will Run"
                    style_name = "green"

                enabled = "Yes" if step_config.get("enabled", True) else "No"

                table.add_row(
                    f"[{style_name}]{step}[/{style_name}]",
                    f"[{style_name}]{status}[/{style_name}]",
                    f"[{style_name}]{enabled}[/{style_name}]",
                )

            console.print(table)
            console.print("\n[bold green]✓ Dry run complete[/bold green]")
            logger.info("Dry run completed successfully")
            return

        except typer.Exit:
            raise
        except Exception:
            _report_cli_operation_failure(
                logger,
                "Unable to render pipeline execution plan",
            )
            raise typer.Exit(1) from None

    # Execute unified pipeline using API
    try:
        from material_agent.api import PipelineInput, run_pipeline

        logger.info("Creating unified pipeline workflow")

        # Create CLI event listener with Rich formatting
        from material_agent.api import CLIEventListener

        cli_listener = CLIEventListener(
            logger=logger, console=console, show_events=False
        )

        # Create API parameters
        api_params = PipelineInput(
            config=config_payload,
            config_path=config_path if isinstance(config_payload, dict) else None,
            skip_steps=skip_steps,
            only_steps=only_steps,
            session_id=session_id,
            resume=resume,
            dry_run=False,
            clean=clean,
            verbose=False,  # Logging already set up
            event_listener=cli_listener,
        )

        logger.info("Running unified pipeline workflow")
        console.print()

        user_email = _get_cli_user_email()
        if user_email:
            telemetry_session_id = _get_cli_telemetry_session_id(session_id)
            tracer = get_tracer(__name__)
            with tracer.start_as_current_span("maa.pipeline.execution") as span:
                span.set_attribute(MAAttributes.PIPELINE_USER_EMAIL, user_email)
                span.set_attribute(MAAttributes.LANGFUSE_USER_ID, user_email)
                span.set_attribute(
                    MAAttributes.PIPELINE_SESSION_ID, telemetry_session_id
                )
                span.set_attribute(
                    MAAttributes.LANGFUSE_SESSION_ID, telemetry_session_id
                )
                try:
                    result = run_pipeline(api_params)
                except Exception:
                    span.set_attribute(MAAttributes.PIPELINE_STATUS, "failed")
                    raise
                span.set_attribute(
                    MAAttributes.PIPELINE_STATUS,
                    "completed" if result.success else "failed",
                )
        else:
            result = run_pipeline(api_params)

        # Display results
        if result.success:
            console.print()

            # Display summary of each step
            if result.step_results:
                console.print("[bold cyan]Pipeline Results Summary:[/bold cyan]\n")
                safe_step_results = redact_sensitive_config(result.step_results)
                if not isinstance(safe_step_results, dict):
                    console.print("  • <redacted>")
                else:
                    for step_name, step_output in safe_step_results.items():
                        console.print(f"[green]✓[/green] {step_name}")
                        if isinstance(step_output, dict):
                            for key, value in step_output.items():
                                if value is not None:
                                    console.print(f"  • {key}: {value}")

            logger.info("Pipeline completed successfully")
        else:
            message = (
                MODEL_AUTHENTICATION_FAILURE_MESSAGE
                if is_model_authentication_error(result.error)
                else "Pipeline execution failed"
            )
            _report_cli_operation_failure(logger, message)
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as error:
        _report_cli_operation_failure(
            logger,
            public_model_failure_message(error, "Pipeline execution failed"),
        )
        if resume:
            console.print(
                "\n[yellow]Tip:[/yellow] Pipeline checkpoint saved. Use --resume to continue."
            )
        raise typer.Exit(1) from None


@app.command()
@sever_cli_exception_graph
def pipeline(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to unified YAML configuration file",
        ),
    ],
    skip: Annotated[
        str | None,
        typer.Option(
            "--skip",
            help="Comma-separated list of steps to skip",
        ),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(
            "--only",
            help="Comma-separated list of steps to run exclusively",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Resume from last successful checkpoint",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show pipeline plan without executing",
        ),
    ] = False,
    clean: Annotated[
        bool,
        typer.Option(
            "--clean",
            help="Clean (delete) working directory and output files (USD + renders) before starting",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    [DEPRECATED] Execute a multi-step material agent pipeline.

    **This command is deprecated. Please use 'material-agent run' instead.**

    This is an alias for the 'run' command and will be removed in a future version.
    """
    # Print deprecation warning
    console.print(
        "[yellow]⚠ Warning:[/yellow] The 'pipeline' command is deprecated and will be removed in a future version."
    )
    console.print(
        "[yellow]           Please use 'material-agent run' instead.[/yellow]\n"
    )

    # Call the run command with the same arguments
    run(
        config=config,
        skip=skip,
        only=only,
        resume=resume,
        dry_run=dry_run,
        clean=clean,
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


@app.command()
def configure(
    output_config: Annotated[
        Path,
        typer.Argument(
            help="Path to output YAML configuration file to create",
        ),
    ],
    materials_manifest: Annotated[
        Path | None,
        typer.Option(
            "--materials-manifest",
            "-m",
            help="Path to materials manifest YAML file (contains library_path and entries)",
        ),
    ] = None,
    reference_images: Annotated[
        list[Path] | None,
        typer.Option(
            "--reference-image",
            "-r",
            help="Reference image path (can be specified multiple times)",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing configuration file",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Create a new pipeline configuration file interactively.

    This command guides you through creating a pipeline configuration
    by asking for essential parameters and auto-populating the rest
    with sensible defaults.

    Example usage:
    ```bash
    # Create a new configuration file
    material-agent configure my_pipeline.yaml

    # Create with a materials manifest
    material-agent configure my_pipeline.yaml -m data/materials/material_libs_new/materials.yaml

    # Create with reference images
    material-agent configure my_pipeline.yaml -m materials.yaml -r ref1.jpg -r ref2.jpg

    # Overwrite existing file
    material-agent configure my_pipeline.yaml --force
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Material Agent Configuration")

    console.print(
        Panel.fit(
            "[bold]Material Agent Configuration Wizard[/bold]\n\n"
            "This wizard will help you create a pipeline configuration file.\n"
            "You'll be asked a few questions, and the rest will be auto-populated.",
            border_style="blue",
        )
    )

    # Use API for configuration creation
    from material_agent.api import ConfigureInput, run_configure

    # Run the workflow using API
    try:
        logger.info("Running configuration wizard...")

        # Create API parameters
        api_params = ConfigureInput(
            output_config_path=output_config,
            materials_manifest=materials_manifest,
            reference_images=[str(p) for p in reference_images]
            if reference_images
            else None,
            force=force,
            verbose=verbose,
        )

        result = run_configure(api_params)

        # Check if configuration was created successfully
        if result.success:
            console.print("\n[bold green]✓ Configuration file created!")
            safe_result_config_path = redact_sensitive_path(result.config_path)
            console.print(
                f"\n[cyan]Configuration saved to:[/cyan] {safe_result_config_path}"
            )

            # Display summary
            console.print("\n[bold]Configuration Summary:[/bold]")
            safe_pipeline_name = redact_sensitive_config(
                result.pipeline_name,
                _path_context=True,
            )
            console.print(f"  • Pipeline name: {safe_pipeline_name}")
            console.print(
                f"  • Input USD: {redact_sensitive_path(result.input_usd_path)}"
            )
            if result.materials_library_path:
                console.print(
                    "  • Materials library: "
                    f"{redact_sensitive_path(result.materials_library_path)}"
                )
                console.print(
                    "  • Materials approach: Unified (library_path + entries)"
                )
            else:
                console.print("  • Materials library: Not specified")
                console.print("  • Materials approach: Legacy (materials_mapping)")
            console.print(
                f"  • Session ID: {safe_pipeline_name}"
                f" (working dir: .{safe_pipeline_name}/)"
            )

            console.print("\n[yellow]Next steps:[/yellow]")
            console.print(f"  1. Review and customize: {safe_result_config_path}")
            if result.materials_library_path:
                console.print(
                    "  2. Update the materials.entries section with your materials"
                )
            else:
                console.print(
                    "  2. Update the materials_list and materials_mapping sections"
                )
            console.print(
                "  3. Run the pipeline: material-agent pipeline "
                f"{safe_result_config_path}"
            )

            logger.info("Configuration wizard completed successfully")
        else:
            _report_cli_operation_failure(logger, "Configuration creation failed")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except FileExistsError:
        safe_output_config = redact_sensitive_path(output_config)
        logger.error(
            "Configuration file already exists: %s",
            safe_output_config,
        )
        console.print(
            f"[red]Error:[/red] Configuration file already exists: {safe_output_config}"
        )
        console.print("[yellow]Use --force to overwrite[/yellow]")
        raise typer.Exit(1) from None
    except Exception:
        _report_cli_operation_failure(logger, "Configuration creation failed")
        raise typer.Exit(1) from None


@app.command("generate-manifest")
@sever_cli_exception_graph
def generate_manifest(
    usd_file: Annotated[
        Path,
        typer.Argument(help="Path to the USD material library file"),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Output directory for materials.yaml and thumbs/"),
    ],
    image_size: Annotated[
        int,
        typer.Option("--image-size", help="Thumbnail size in pixels"),
    ] = 256,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing",
            help="Skip rendering thumbnails that already exist in the output dir",
        ),
    ] = False,
    library_path: Annotated[
        str | None,
        typer.Option(
            "--library-path",
            help="Value for library_path in materials.yaml (default: uses usd-file path)",
        ),
    ] = None,
    template: Annotated[
        Path | None,
        typer.Option(
            "--template",
            help="Path to the thumbnail template USD file (default: built-in template)",
        ),
    ] = None,
    max_workers: Annotated[
        int,
        typer.Option("--max-workers", help="Number of parallel NVCF render workers"),
    ] = 4,
    skip_descriptions: Annotated[
        bool,
        typer.Option(
            "--skip-descriptions",
            help="Skip VLM description generation (leave descriptions empty)",
        ),
    ] = False,
    vlm_backend: Annotated[
        str,
        typer.Option("--vlm-backend", help="VLM backend"),
    ] = "nim",
    vlm_model: Annotated[
        str | None,
        typer.Option("--vlm-model", help="VLM model name"),
    ] = "google/gemma-4-31b-it",
    vlm_workers: Annotated[
        int,
        typer.Option("--vlm-workers", help="Number of parallel VLM workers"),
    ] = 8,
    list_materials: Annotated[
        bool,
        typer.Option(
            "--list-materials",
            help="List all material prims in the USD file and exit",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Generate materials.yaml manifest and thumbnails from a USD material library.

    Discovers all Material prims in a USD file, renders thumbnails via NVCF
    cloud rendering, optionally generates VLM descriptions, and outputs a
    complete materials.yaml with a thumbs/ directory.

    Example usage:
    ```bash
    # Generate manifest with thumbnails and descriptions
    material-agent generate-manifest materials.usd output/

    # Skip VLM descriptions
    material-agent generate-manifest materials.usd output/ --skip-descriptions

    # Use a custom template and larger thumbnails
    material-agent generate-manifest materials.usd output/ --template my_template.usd --image-size 512

    # List materials without generating anything
    material-agent generate-manifest materials.usd output/ --list-materials

    # Resume (skip already-rendered thumbnails)
    material-agent generate-manifest materials.usd output/ --skip-existing
    ```
    """
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    from material_agent.manifest import (
        GenerateManifestInput,
        run_generate_manifest,
    )

    # Build params, using default template if not specified
    params_kwargs: dict = {
        "usd_file": usd_file,
        "output_dir": output_dir,
        "image_size": image_size,
        "skip_existing": skip_existing,
        "library_path": library_path,
        "max_workers": max_workers,
        "skip_descriptions": skip_descriptions,
        "vlm_backend": vlm_backend,
        "vlm_model": vlm_model,
        "vlm_workers": vlm_workers,
        "list_materials": list_materials,
        "verbose": verbose,
    }
    if template is not None:
        params_kwargs["template"] = template

    try:
        params = GenerateManifestInput(**params_kwargs)
        result = run_generate_manifest(params)

        if not result.success:
            _report_cli_operation_failure(logger, "Manifest generation failed")
            raise typer.Exit(1)

        if list_materials:
            console.print(
                f"\n[bold]Materials in {redact_sensitive_path(usd_file.name)}[/bold] "
                f"({result.materials_count} found):\n"
            )
            from material_agent.manifest import prim_path_to_name

            for pp in result.material_paths:
                console.print(
                    f"  {redact_sensitive_path(pp)}  ->  "
                    f"{redact_sensitive_path(prim_path_to_name(pp))}"
                )
            return

        # Summary
        console.print("\n[bold green]Manifest generated successfully[/bold green]")
        table = Table(show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Materials discovered", str(result.materials_count))
        table.add_row("Thumbnails rendered", str(result.thumbnails_count))
        table.add_row("Descriptions generated", str(result.descriptions_count))
        table.add_row("Output", redact_sensitive_path(result.yaml_path))
        table.add_row(
            "Thumbnails",
            redact_sensitive_path(output_dir / "thumbs" / f"{image_size}x{image_size}"),
        )
        console.print(table)

    except typer.Exit:
        raise
    except Exception:
        _report_cli_operation_failure(logger, "Manifest generation failed")
        raise typer.Exit(1) from None


# Register scene subcommand for large-scene multi-asset pipeline
app.add_typer(scene_app, name="scene")

if __name__ == "__main__":
    app()
