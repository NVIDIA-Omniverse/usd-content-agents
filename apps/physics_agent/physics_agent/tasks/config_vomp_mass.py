# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading for the concrete VoMP mass-inference step."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import load_config_mapping_from_context
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import resolve_path_with_safe_diagnostics
from world_understanding.utils.object_store import ObjectStore

from physics_agent.integrations.vomp_defaults import (
    DEFAULT_VOMP_ARTIFACT_SHA256,
    DEFAULT_VOMP_REVISION,
)


def _resolve(path: str | Path, anchor: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = anchor / candidate
    return resolve_path_with_safe_diagnostics(candidate, label=label)


class VompMassConfigTask(Task):
    """Resolve the VoMP step without importing its external runtime."""

    def __init__(self) -> None:
        self.name = "VompMassConfig"
        self.description = "Load pinned VoMP mass-inference configuration"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        config, _ = load_config_mapping_from_context(
            context,
            allow_empty=True,
            missing_path_message="No config_path or config_dict in context",
            missing_file_message="Config not found: {config_path}",
            parse_error_message="Unable to parse vomp_mass configuration: {config_path}",
            config_dict_non_mapping_message=(
                "vomp_mass config_dict must be a mapping, got {type_name}"
            ),
            file_non_mapping_message=(
                "vomp_mass config must be a YAML mapping, got {type_name}: {config_path}"
            ),
        )
        config_dir = (
            Path(context["config_path"]).expanduser().resolve().parent
            if context.get("config_path")
            else Path.cwd()
        )
        required = ("usd_path", "output_usd_path", "target_prim", "runtime_root")
        missing = [name for name in required if not config.get(name)]
        if missing:
            raise ValueError(
                "vomp_mass is missing required configuration: " + ", ".join(missing)
            )

        runtime_root = _resolve(
            config["runtime_root"], config_dir, label="VoMP runtime root"
        )
        python_value = config.get("python_executable", ".venv/bin/python")
        python_candidate = Path(python_value).expanduser()
        if not python_candidate.is_absolute():
            python_candidate = runtime_root / python_candidate
        python_executable = Path(os.path.abspath(python_candidate))
        inference_config = _resolve(
            config.get("config_path", "weights/inference.json"),
            runtime_root,
            label="VoMP inference config",
        )
        output_usd = _resolve(
            config["output_usd_path"], config_dir, label="VoMP output USD"
        )
        work_dir = _resolve(
            config.get("work_dir", output_usd.parent / "vomp_artifacts"),
            config_dir,
            label="VoMP artifact directory",
        )
        provenance = config.get("provenance_path")
        render = config.get("render")
        if render is None:
            render = {}
        if not isinstance(render, dict):
            raise ValueError("vomp_mass.render must be a mapping")
        render = dict(render)
        ovrtx_venv_dir = render.get("ovrtx_venv_dir")
        if ovrtx_venv_dir:
            render["ovrtx_venv_dir"] = str(
                _resolve(
                    ovrtx_venv_dir,
                    config_dir,
                    label="OVRTX environment directory",
                )
            )

        context.update(
            {
                "usd_path": str(
                    _resolve(config["usd_path"], config_dir, label="VoMP input USD")
                ),
                "output_usd_path": str(output_usd),
                "target_prim": str(config["target_prim"]),
                "work_dir": str(work_dir),
                "runtime_root": str(runtime_root),
                "python_executable": str(python_executable),
                "vomp_config_path": str(inference_config),
                "expected_revision": str(
                    config.get("expected_revision", DEFAULT_VOMP_REVISION)
                ),
                "expected_artifact_sha256": dict(
                    config.get("expected_artifact_sha256")
                    or DEFAULT_VOMP_ARTIFACT_SHA256
                ),
                "attention_backend": str(config.get("attention_backend", "xformers")),
                "timeout_seconds": float(config.get("timeout_seconds", 3600.0)),
                "max_complete_voxels": int(config.get("max_complete_voxels", 262_144)),
                "render": render,
            }
        )
        if provenance:
            context["provenance_path"] = str(
                _resolve(provenance, config_dir, label="VoMP provenance output")
            )
        return context


__all__ = ["VompMassConfigTask"]
