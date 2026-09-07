# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run calibrated OVRTX evidence through VoMP and author USD mass properties."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore

from physics_agent.integrations.vomp_defaults import DEFAULT_VOMP_RENDER_CONFIG
from physics_agent.integrations.vomp_pipeline import (
    VompRenderConfig,
    run_vomp_mass_pipeline,
)
from physics_agent.integrations.vomp_runtime import VompRuntimeConfig


class VompMassTask(Task):
    """Execute the concrete OVRTX -> VoMP -> MassAPI blueprint."""

    def __init__(self) -> None:
        self.name = "VompMass"
        self.description = "Infer and author rigid-body mass properties with VoMP"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        render_values = {
            **DEFAULT_VOMP_RENDER_CONFIG,
            **dict(context.get("render") or {}),
        }
        render_config = VompRenderConfig(
            num_views=int(render_values["num_views"]),
            image_width=int(render_values["image_width"]),
            image_height=int(render_values["image_height"]),
            radius=float(render_values["radius"]),
            fov_degrees=float(render_values["fov_degrees"]),
            seed=int(render_values["seed"]),
            render_mode=str(render_values["render_mode"]),
            num_sensor_updates=int(render_values["num_sensor_updates"]),
            material_target=str(render_values["material_target"]),
            ovrtx_venv_dir=(
                str(render_values["ovrtx_venv_dir"])
                if render_values["ovrtx_venv_dir"] is not None
                else None
            ),
        )
        runtime = VompRuntimeConfig(
            runtime_root=Path(context["runtime_root"]),
            python_executable=Path(context["python_executable"]),
            config_path=Path(context["vomp_config_path"]),
            expected_revision=str(context["expected_revision"]),
            expected_artifact_sha256=dict(context["expected_artifact_sha256"]),
            attention_backend=str(context["attention_backend"]),
            timeout_seconds=float(context["timeout_seconds"]),
            max_complete_voxels=int(context["max_complete_voxels"]),
        )
        result = run_vomp_mass_pipeline(
            context["usd_path"],
            context["output_usd_path"],
            target_prim_path=context["target_prim"],
            work_dir=context["work_dir"],
            runtime_config=runtime,
            render_config=render_config,
            provenance_path=context.get("provenance_path"),
        )
        context.update(
            {
                "output_usd_path": str(result.apply_result.output_usd_path),
                "provenance_path": str(result.apply_result.provenance_path),
                "vomp_npz_path": str(result.vomp_npz_path),
                "vomp_artifact_dir": str(result.evidence.artifact_dir),
                "vomp_worker_manifest_path": str(result.worker_manifest_path),
                "vomp_worker_log_path": str(result.worker_log_path),
                "vomp_sample_count": result.apply_result.sample_count,
            }
        )
        return context


__all__ = ["VompMassTask"]
