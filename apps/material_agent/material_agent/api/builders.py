# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config builder utilities for Material Agent API.

These helpers make it easy to build configuration dictionaries programmatically
without needing to remember all required fields.
"""

import copy
from pathlib import Path
from typing import Any

from material_agent.api.defaults import (
    DEFAULT_CLUSTER_BATCH_SIZE,
    DEFAULT_CLUSTER_COMPLEXITY_THRESHOLDS,
    DEFAULT_CLUSTER_EMBEDDING_BACKEND,
    DEFAULT_CLUSTER_EMBEDDING_MODEL,
    DEFAULT_CLUSTER_EMBEDDING_RETRIES,
    DEFAULT_CLUSTER_EMBEDDING_RETRY_BACKOFF,
    DEFAULT_CLUSTER_EMBEDDING_RETRY_INITIAL_DELAY,
    DEFAULT_CLUSTER_MAX_SIZE,
    DEFAULT_CLUSTER_MAX_WORKERS,
    DEFAULT_CLUSTER_MIN_PRIMS_TO_ACTIVATE,
    DEFAULT_CLUSTER_NIM_EMBEDDING_MODEL,
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_MODEL,
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_MAX_TOKENS,
    DEFAULT_VLM_MAX_WORKERS,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_TEMPERATURE,
)


def build_vlm_config(
    backend: str = DEFAULT_VLM_BACKEND,
    model: str = DEFAULT_VLM_MODEL,
    temperature: float | None = DEFAULT_VLM_TEMPERATURE,
    max_tokens: int | None = DEFAULT_VLM_MAX_TOKENS,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build VLM configuration dict.

    Args:
        backend: VLM backend service (default: from defaults.py)
        model: Model name (default: from defaults.py)
        temperature: Sampling temperature (default: from defaults.py)
        max_tokens: Maximum tokens (default: from defaults.py)
        **kwargs: Additional VLM parameters

    Returns:
        VLM configuration dictionary
    """
    config: dict[str, Any] = {
        "backend": backend,
        "model": model,
    }

    if temperature is not None:
        config["temperature"] = temperature
    if max_tokens is not None:
        config["max_tokens"] = max_tokens

    config.update(kwargs)
    return config


def build_predict_config(
    dataset_path: str | Path,  # REQUIRED - no default
    vlm_backend: str = DEFAULT_VLM_BACKEND,  # Uses centralized default
    vlm_model: str = DEFAULT_VLM_MODEL,  # Uses centralized default
    output_dir: str | Path | None = None,
    llm_backend: str = DEFAULT_LLM_BACKEND,  # Uses centralized default
    llm_model: str = DEFAULT_LLM_MODEL,  # Uses centralized default
    system_prompt: str | None = None,
    system_prompt_file: str | Path | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_workers: int = DEFAULT_VLM_MAX_WORKERS,  # Uses centralized default
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a minimal prediction configuration.

    Args:
        vlm_backend: VLM backend (e.g., "nim")
        vlm_model: VLM model name (e.g., "example-vlm-model")
        dataset_path: Path to dataset JSONL file
        output_dir: Output directory (optional)
        llm_backend: LLM backend for parsing (optional)
        llm_model: LLM model name (optional)
        system_prompt: Custom system prompt (optional)
        system_prompt_file: Path to system prompt file (optional)
        temperature: Sampling temperature (optional)
        max_tokens: Maximum tokens (optional)
        max_workers: Number of parallel workers (default: 16)
        **kwargs: Additional config parameters

    Returns:
        Configuration dictionary ready to use with predict API

    Example:
        >>> config = build_predict_config(
        ...     vlm_backend="nim",
        ...     vlm_model="example-vlm-model",
        ...     dataset_path="data/dataset.jsonl",
        ... )
        >>> result = predict(config)
    """
    if not dataset_path:
        raise ValueError("dataset_path is required")

    config: dict[str, Any] = {
        "vlm": build_vlm_config(vlm_backend, vlm_model, temperature, max_tokens),
        "llm": build_vlm_config(llm_backend, llm_model, temperature, max_tokens),
        "dataset": str(dataset_path),
        "max_workers": max_workers,
    }

    if output_dir:
        config["output_dir"] = str(output_dir)

    if system_prompt:
        config["system_prompt"] = system_prompt

    if system_prompt_file:
        config["system_prompt_file"] = str(system_prompt_file)

    config.update(kwargs)
    return config


def build_benchmark_config(
    dataset_path: str | Path,  # REQUIRED - no default
    vlm_backend: str = DEFAULT_VLM_BACKEND,  # Uses centralized default
    vlm_model: str = DEFAULT_VLM_MODEL,  # Uses centralized default
    judge_backend: str | None = None,  # Defaults to vlm_backend
    judge_model: str | None = None,  # Defaults to vlm_model
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a minimal benchmark configuration.

    Args:
        vlm_backend: VLM backend for predictions
        vlm_model: VLM model name
        dataset_path: Path to dataset JSONL file
        judge_backend: Judge backend (defaults to vlm_backend)
        judge_model: Judge model (defaults to vlm_model)
        output_dir: Output directory (optional)
        **kwargs: Additional config parameters

    Returns:
        Configuration dictionary ready to use with benchmark API

    Example:
        >>> config = build_benchmark_config(
        ...     vlm_backend="nim",
        ...     vlm_model="example-vlm-model",
        ...     dataset_path="data/dataset.jsonl",
        ... )
        >>> result = benchmark(config)
    """
    # Use same model for judge if not specified
    judge_backend = judge_backend or vlm_backend
    judge_model = judge_model or vlm_model

    config: dict[str, Any] = {
        "vlm": build_vlm_config(vlm_backend, vlm_model),
        "llm": build_vlm_config(vlm_backend, vlm_model),
        "judge": build_vlm_config(judge_backend, judge_model),
        "dataset": str(dataset_path),
    }

    if output_dir:
        config["output_dir"] = str(output_dir)

    config.update(kwargs)
    return config


def build_apply_config(
    input_usd_path: str | Path,
    predictions_path: str | Path,
    output_usd_path: str | Path,
    materials_library_path: str | Path,
    materials_entries: list[dict[str, str]],
    layer_only: bool = False,
    flatten: bool = True,
    material_profile: str = "auto",
    render_enabled: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a minimal apply configuration.

    Args:
        input_usd_path: Path to input USD file
        predictions_path: Path to predictions JSONL
        output_usd_path: Path to output USD file
        materials_library_path: Path to materials library USD
        materials_entries: List of material definitions
        layer_only: Output layer only (default: False)
        flatten: Flatten output USD (default: True)
        material_profile: Material authoring profile (default: "auto")
        render_enabled: Enable rendering (default: False)
        **kwargs: Additional config parameters

    Returns:
        Configuration dictionary ready to use with apply API

    Example:
        >>> config = build_apply_config(
        ...     input_usd_path="input.usd",
        ...     predictions_path="predictions.jsonl",
        ...     output_usd_path="output.usd",
        ...     materials_library_path="materials.usd",
        ...     materials_entries=[
        ...         {"name": "Steel", "prim_path": "/Materials/Steel"}
        ...     ],
        ... )
        >>> result = apply(config)
    """
    config: dict[str, Any] = {
        "input_usd_path": str(input_usd_path),
        "predictions_path": str(predictions_path),
        "output_usd_path": str(output_usd_path),
        "materials": {
            "library_path": str(materials_library_path),
            "entries": materials_entries,
        },
        "layer_only": layer_only,
        "flatten": flatten,
        "material_profile": material_profile,
    }

    if render_enabled:
        config["render"] = {"enabled": True}

    config.update(kwargs)
    return config


def build_cluster_prims_config(
    *,
    enabled: bool = True,
    embedding_service: str = DEFAULT_CLUSTER_EMBEDDING_BACKEND,
    embedding_model: str | None = None,
    min_prims_to_activate: int = DEFAULT_CLUSTER_MIN_PRIMS_TO_ACTIVATE,
    batch_size: int = DEFAULT_CLUSTER_BATCH_SIZE,
    max_workers: int = DEFAULT_CLUSTER_MAX_WORKERS,
    max_cluster_size: int | None = DEFAULT_CLUSTER_MAX_SIZE,
    base_url: str | None = None,
    api_key: str | None = None,
    report: bool | dict[str, Any] = True,
    complexity_thresholds: dict[str, list[float]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a prim clustering step configuration."""
    default_embedding_model = (
        DEFAULT_CLUSTER_NIM_EMBEDDING_MODEL
        if embedding_service == "nim"
        else DEFAULT_CLUSTER_EMBEDDING_MODEL
    )
    config: dict[str, Any] = {
        "enabled": enabled,
        "embedding_service": embedding_service,
        "embedding_model": embedding_model or default_embedding_model,
        "min_prims_to_activate": min_prims_to_activate,
        "batch_size": batch_size,
        "max_workers": max_workers,
        "max_cluster_size": max_cluster_size,
        "complexity_thresholds": copy.deepcopy(
            complexity_thresholds
            if complexity_thresholds is not None
            else DEFAULT_CLUSTER_COMPLEXITY_THRESHOLDS
        ),
        "embedding_retries": DEFAULT_CLUSTER_EMBEDDING_RETRIES,
        "embedding_retry_initial_delay": DEFAULT_CLUSTER_EMBEDDING_RETRY_INITIAL_DELAY,
        "embedding_retry_backoff": DEFAULT_CLUSTER_EMBEDDING_RETRY_BACKOFF,
    }
    if base_url:
        config["base_url"] = base_url
    if api_key:
        config["api_key"] = api_key
    if isinstance(report, bool):
        config["report"] = {"enabled": report}
    else:
        config["report"] = report
    config.update(kwargs)
    return config


def build_unified_pipeline_config(
    project_name: str,  # REQUIRED
    input_usd_path: str | Path,  # REQUIRED
    materials_library_path: str | Path,  # REQUIRED
    materials_entries: list[dict[str, str]],  # REQUIRED
    vlm_backend: str = DEFAULT_VLM_BACKEND,  # Uses centralized default
    vlm_model: str = DEFAULT_VLM_MODEL,  # Uses centralized default
    llm_backend: str = DEFAULT_LLM_BACKEND,  # Uses centralized default
    llm_model: str = DEFAULT_LLM_MODEL,  # Uses centralized default
    user_prompt: str | None = None,  # User prompt for the prepare_dataset step
    enabled_steps: list[str] | None = None,
    enable_prim_clustering: bool = False,
    cluster_prims_config: dict[str, Any] | None = None,
    session_id: str | None = None,
    working_dir: str | None = None,
    output_usd_path: str
    | Path
    | None = None,  # DEPRECATED - auto-derived from session_id
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a minimal unified pipeline configuration.

    Args:
        project_name: Project name
        input_usd_path: Path to input USD file
        materials_library_path: Path to materials library
        materials_entries: List of material definitions
        vlm_backend: VLM backend (default: nim)
        vlm_model: VLM model (default: example-vlm-model)
        enabled_steps: Steps to enable (default: all)
        enable_prim_clustering: Whether to insert visual prim clustering before
            prediction.
        cluster_prims_config: Optional overrides for the cluster_prims step.
        session_id: Session ID for tracking runs (auto-generated if None)
        working_dir: Working directory (default: .{session_id})
        output_usd_path: DEPRECATED - Output path is now auto-derived as .{session_id}/output/output.usd
        **kwargs: Additional config parameters

    Returns:
        Unified configuration dictionary

    Example:
        >>> config = build_unified_pipeline_config(
        ...     project_name="my_project",
        ...     input_usd_path="models/input.usd",
        ...     materials_library_path="materials/lib.usd",
        ...     materials_entries=[{"name": "Steel", "prim_path": "/Materials/Steel"}],
        ... )
        >>> result = pipeline(config)
    """
    from material_agent.config.schema import get_step_defaults

    enabled_steps = enabled_steps or [
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
        "apply",
    ]
    enabled_steps = list(enabled_steps)
    if enable_prim_clustering and "cluster_prims" not in enabled_steps:
        if "predict" in enabled_steps:
            insert_at = enabled_steps.index("predict")
        elif "benchmark" in enabled_steps:
            insert_at = enabled_steps.index("benchmark")
        else:
            insert_at = len(enabled_steps)
        enabled_steps.insert(insert_at, "cluster_prims")

    project_config: dict[str, Any] = {
        "name": project_name,
    }

    # Add session_id if provided
    if session_id is not None:
        project_config["session_id"] = session_id

    # Add working_dir if provided (otherwise will be auto-derived)
    if working_dir is not None:
        project_config["working_dir"] = working_dir

    # Build output config - only include usd_path if explicitly provided (for backward compat)
    output_config: dict[str, Any] = {}
    if output_usd_path is not None:
        output_config["usd_path"] = str(output_usd_path)

    config: dict[str, Any] = {
        "project": project_config,
        "input": {
            "usd_path": str(input_usd_path),
        },
        "output": output_config,  # Empty dict or with explicit usd_path
        "materials": {
            "library_path": str(materials_library_path),
            "entries": materials_entries,
        },
        "steps": {},
    }

    # Add enabled steps with model configs, merging with defaults
    # Path-like keys that should be excluded from unified pipeline step configs
    # (these get auto-wired by the unified config system)
    path_like_keys = {
        "local_material_source_dir",
        "aws_profile",  # Also path-like in nature
        "source",  # Used in pdf_vectorstore
        "usd_path",
        "usd_dir",
        "output_dir",
        "dataset",
        "dataset_path",
        "vector_store",
        "predictions_path",
    }

    for step_name in enabled_steps:
        step_defaults = get_step_defaults(step_name)
        # Filter out path-like keys that shouldn't be in unified pipeline configs
        step_config = {
            k: v for k, v in step_defaults.items() if k not in path_like_keys
        }

        if step_name == "predict":
            # Override with specific VLM/LLM configs for predict step
            step_config.update(
                {
                    "enabled": True,
                    "vlm": build_vlm_config(vlm_backend, vlm_model),
                    "llm": build_vlm_config(llm_backend, llm_model),  # For parsing
                }
            )
        elif step_name == "build_dataset_prepare_dataset":
            step_config["enabled"] = True
            if user_prompt is not None:
                step_config["prompts"]["vlm_user"] = user_prompt
        elif step_name == "cluster_prims":
            step_config.update(
                build_cluster_prims_config(
                    **(cluster_prims_config or {}),
                )
            )
        else:
            # For other steps, ensure they're enabled
            step_config["enabled"] = True

        config["steps"][step_name] = step_config

    config.update(kwargs)
    return config


def get_required_fields(api_name: str) -> dict[str, list[str]]:
    """Get required configuration fields for each API.

    Args:
        api_name: Name of API (benchmark, predict, evaluate, apply, pipeline, refine)

    Returns:
        Dictionary with required and optional field lists
    """
    required_fields = {
        "predict": {
            "required": ["vlm.backend", "vlm.model", "dataset"],
            "optional": ["llm", "output_dir", "system_prompt", "max_workers"],
        },
        "benchmark": {
            "required": [
                "vlm.backend",
                "vlm.model",
                "llm.backend",
                "llm.model",
                "judge.backend",
                "judge.model",
                "dataset",
            ],
            "optional": ["output_dir", "max_workers", "temperature"],
        },
        "evaluate": {
            "required": ["judge.backend", "judge.model", "predictions_path"],
            "optional": ["dataset_path", "output_dir"],
        },
        "apply": {
            "required": [
                "input_usd_path",
                "predictions_path",
                "output_usd_path",
                "materials.library_path",
                "materials.entries",
            ],
            "optional": ["layer_only", "flatten", "render.enabled"],
        },
        "pipeline": {
            "required": [
                "project.name",
                "input.usd_path",
                "output.usd_path",
                "materials.library_path",
                "materials.entries",
            ],
            "optional": ["project.working_dir", "steps.*"],
        },
        "refine": {
            "required": [
                "input_usd_path",
                "dataset",
                "materials.library_path",
                "materials.entries",
                "predict.vlm.backend",
                "predict.vlm.model",
                "judge.vlm.backend",
                "judge.vlm.model",
                "judge.reference_images",
            ],
            "optional": ["iteration.max_iterations", "output_usd_path"],
        },
    }

    return required_fields.get(api_name, {"required": [], "optional": []})
