# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: execute only the immutable units approved by a Texture Plan."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Lock
from typing import Any

from world_understanding.agentic.tasks import Task

from texture_agent.execution import (
    BoundedTextureExecutor,
    CancellationToken,
    FileTextureExecutionCheckpointStore,
    TextureArtifactRef,
    TextureUnitExecutionContext,
    TextureUnitExecutionResult,
    bind_prim_texture_units_to_plan,
)
from texture_agent.functions.artifact_manifest import redact_sensitive
from texture_agent.functions.material_discovery import PrimTextureUnit
from texture_agent.functions.texture_generation import GeneratedTextures
from texture_agent.planning import (
    TexturePlan,
    TexturePlanUnit,
    validate_texture_plan_payload,
)
from texture_agent.tasks.generate_textures import GenerateTexturesTask
from texture_agent.tasks.thresholds import (
    raise_if_failure_threshold_exceeded,
    validate_failure_threshold,
)

_CHECKPOINT_RELATIVE_PATH = Path("execution") / "texture_execution_checkpoint.json"


class ExecuteTexturePlanTask(Task):
    """Bridge the shared plan executor to the existing generation task.

    Integration supplies ``texture_plan`` (or ``texture_plan_path``) after
    prompt expansion. Runtime units are matched by canonical USD identity,
    filtered to the exact approved set, and rekeyed to stable plan unit IDs.
    Blend/apply tasks can then consume the accepted results without any
    display-name ambiguity.
    """

    def __init__(
        self,
        unit_runner: Callable[
            [
                TexturePlanUnit,
                PrimTextureUnit,
                dict[str, Any],
                TextureUnitExecutionContext,
            ],
            TextureUnitExecutionResult,
        ]
        | None = None,
    ) -> None:
        self.name = "ExecuteTexturePlan"
        self.description = "Execute immutable selected texture units"
        self._unit_runner = unit_runner

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        plan = self._load_plan(context)
        runtime_units: list[PrimTextureUnit] = context.get("prim_texture_units", [])
        bound_units = bind_prim_texture_units_to_plan(plan, runtime_units)
        runtime_by_id = {unit.key: unit for unit in bound_units}
        context["prim_texture_units"] = bound_units

        working_dir = Path(context["working_dir"])
        checkpoint_store = context.get("texture_execution_checkpoint_store")
        if checkpoint_store is None:
            checkpoint_path = working_dir / _CHECKPOINT_RELATIVE_PATH
            checkpoint_store = FileTextureExecutionCheckpointStore(checkpoint_path)
            context["texture_execution_checkpoint_path"] = str(checkpoint_path)
        # Validate the protocol structurally without coupling this public task
        # to a service checkpoint implementation.
        if not callable(getattr(checkpoint_store, "load", None)) or not callable(
            getattr(checkpoint_store, "save", None)
        ):
            raise TypeError(
                "texture_execution_checkpoint_store must provide load() and save()"
            )

        cancellation_token = context.get("texture_execution_cancellation_token")
        if cancellation_token is None:
            cancellation_token = CancellationToken()
        if not isinstance(cancellation_token, CancellationToken):
            raise TypeError(
                "texture_execution_cancellation_token must be a CancellationToken"
            )

        external_cancel = context.get("texture_execution_is_cancelled")
        if external_cancel is not None and not callable(external_cancel):
            raise TypeError("texture_execution_is_cancelled must be callable")

        base_context = dict(context)
        texture_config = context.get("texture_config") or {}
        failure_threshold = validate_failure_threshold(
            texture_config.get("failure_threshold", 1.0),
            config_key="texture_config.failure_threshold",
        )
        observation_lock = Lock()
        unit_observations: dict[str, dict[str, Any]] = {}

        def _record_observation(
            unit_id: str,
            observation: dict[str, Any],
        ) -> None:
            with observation_lock:
                unit_observations[unit_id] = observation

        def _execute_unit(
            plan_unit: TexturePlanUnit,
            execution_context: TextureUnitExecutionContext,
        ) -> TextureUnitExecutionResult:
            runtime_unit = runtime_by_id[plan_unit.unit_id]
            unit_context = dict(base_context)
            if self._unit_runner is not None:
                return self._unit_runner(
                    plan_unit,
                    runtime_unit,
                    unit_context,
                    execution_context,
                )
            return self._generate_one(
                plan,
                plan_unit,
                runtime_unit,
                unit_context,
                execution_context,
                _record_observation,
            )

        progress_callback = context.get("texture_execution_progress_callback")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("texture_execution_progress_callback must be callable")

        executor = BoundedTextureExecutor(
            plan=plan,
            checkpoint_store=checkpoint_store,
            unit_runner=_execute_unit,
            cancellation_token=cancellation_token,
            external_cancellation_check=external_cancel,
            progress_callback=progress_callback,
        )
        regenerate_ids = self._regenerate_ids(context)
        summary = executor.execute(
            resume=bool(
                context.get("resume")
                or context.get("texture_execution_resume")
                or (context.get("planning_config") or {}).get("resume_execution")
            ),
            regenerate_unit_ids=regenerate_ids,
        )

        generated: dict[str, GeneratedTextures] = {}
        projection_results: dict[str, Any] = {}
        diagnostics: list[dict[str, Any]] = []
        for record in summary.records:
            result = record.accepted_result
            if result is None:
                continue
            artifact_by_name = {
                artifact.name: artifact.uri for artifact in result.artifacts
            }
            if {"albedo", "normal", "orm"}.issubset(artifact_by_name):
                generated[record.unit_id] = GeneratedTextures(
                    albedo=artifact_by_name["albedo"],
                    normal=artifact_by_name["normal"],
                    orm=artifact_by_name["orm"],
                )
            backend_result = result.metadata.get("projection_backend_result")
            if isinstance(backend_result, dict):
                projection_results[record.unit_id] = backend_result

        for plan_unit in plan.selected_units:
            observation = unit_observations.get(plan_unit.unit_id, {})
            backend_result = observation.get("projection_backend_result")
            if isinstance(backend_result, dict):
                projection_results[plan_unit.unit_id] = backend_result
            unit_diagnostics = observation.get("diagnostics")
            if isinstance(unit_diagnostics, list):
                diagnostics.extend(
                    item for item in unit_diagnostics if isinstance(item, dict)
                )

        context["generated_textures"] = generated
        context["texture_execution"] = summary.model_dump(mode="json")
        context["texture_execution_status"] = summary.status.value
        context["texture_execution_accepted_unit_ids"] = list(summary.accepted_unit_ids)
        context["texture_execution_remaining_unit_ids"] = list(
            summary.remaining_unit_ids
        )
        errors: list[dict[str, Any]] = []
        for record in summary.records:
            if record.unit_id not in summary.failed_unit_ids:
                continue
            observed_errors = unit_observations.get(record.unit_id, {}).get("errors")
            if isinstance(observed_errors, list):
                errors.extend(
                    item for item in observed_errors if isinstance(item, dict)
                )
            if not observed_errors:
                errors.append(
                    {
                        "material": record.unit_id,
                        "type": "TextureUnitExecutionError",
                        "status": None,
                        "message": record.last_error or "Texture unit execution failed",
                    }
                )
        context["generate_textures_errors"] = errors
        context["generate_textures_failed_count"] = len(errors)
        context["generate_textures_attempted_count"] = len(summary.executed_unit_ids)
        if projection_results:
            context["projection_backend_results"] = projection_results
        if diagnostics:
            context["generate_textures_diagnostics"] = diagnostics
        raise_if_failure_threshold_exceeded(
            attempted_count=len(summary.requested_unit_ids),
            errors=errors,
            backend_label=plan.execution.backend,
            failure_threshold=failure_threshold,
        )
        return context

    @staticmethod
    def _load_plan(context: dict[str, Any]) -> TexturePlan:
        payload = context.get("texture_plan")
        if payload is None:
            path_value = context.get("texture_plan_path")
            if not isinstance(path_value, str) or not path_value.strip():
                raise ValueError(
                    "ExecuteTexturePlanTask requires texture_plan or texture_plan_path"
                )
            payload = Path(path_value).read_text(encoding="utf-8")
        return validate_texture_plan_payload(payload)

    @staticmethod
    def _regenerate_ids(context: dict[str, Any]) -> tuple[str, ...]:
        planning_config = context.get("planning_config") or {}
        raw = context.get(
            "texture_regenerate_unit_ids",
            context.get(
                "regenerate_texture_unit_ids",
                planning_config.get("regenerate_unit_ids", ()),
            ),
        )
        if raw is None:
            return ()
        if isinstance(raw, str) or not isinstance(raw, Sequence):
            raise TypeError("texture_regenerate_unit_ids must be a sequence of IDs")
        return tuple(str(unit_id) for unit_id in raw)

    @staticmethod
    def _generate_one(
        plan: TexturePlan,
        plan_unit: TexturePlanUnit,
        runtime_unit: PrimTextureUnit,
        base_context: dict[str, Any],
        execution_context: TextureUnitExecutionContext,
        observation_callback: Callable[[str, dict[str, Any]], None],
    ) -> TextureUnitExecutionResult:
        execution_context.raise_if_cancelled()
        unit_context = dict(base_context)
        texture_config = dict(unit_context.get("texture_config", {}))
        texture_config.update(
            {
                "backend": plan.execution.backend,
                "size": plan.execution.texture_size,
                "workers": 1,
                "skip_existing": False,
                "job_timeout_sec": plan.execution.unit_timeout_seconds,
            }
        )
        unit_context.update(
            {
                "prim_texture_units": [runtime_unit],
                "texture_config": texture_config,
                "resume": False,
                "generated_textures": {},
            }
        )
        try:
            result_context = GenerateTexturesTask().run(unit_context)
        except Exception:
            ExecuteTexturePlanTask._record_generation_observation(
                unit_context,
                plan_unit.unit_id,
                observation_callback,
            )
            raise
        ExecuteTexturePlanTask._record_generation_observation(
            result_context,
            plan_unit.unit_id,
            observation_callback,
        )
        backend_result = (result_context.get("projection_backend_results") or {}).get(
            plan_unit.unit_id
        )
        execution_context.raise_if_timed_out()
        generated = result_context.get("generated_textures", {}).get(plan_unit.unit_id)
        if not isinstance(generated, GeneratedTextures):
            raise RuntimeError(
                f"Texture backend produced no accepted result for {plan_unit.unit_id}"
            )
        metadata: dict[str, Any] = {}
        if isinstance(backend_result, dict):
            redacted_backend_result = redact_sensitive(backend_result)
            if isinstance(redacted_backend_result, dict):
                metadata["projection_backend_result"] = redacted_backend_result
        return TextureUnitExecutionResult(
            unit_id=plan_unit.unit_id,
            artifacts=(
                ExecuteTexturePlanTask._artifact_ref("albedo", generated.albedo),
                ExecuteTexturePlanTask._artifact_ref("normal", generated.normal),
                ExecuteTexturePlanTask._artifact_ref("orm", generated.orm),
            ),
            metadata=metadata,
        )

    @staticmethod
    def _record_generation_observation(
        result_context: dict[str, Any],
        unit_id: str,
        observation_callback: Callable[[str, dict[str, Any]], None],
    ) -> None:
        backend_result = (result_context.get("projection_backend_results") or {}).get(
            unit_id
        )
        observation_callback(
            unit_id,
            {
                "projection_backend_result": backend_result,
                "diagnostics": list(
                    result_context.get("generate_textures_diagnostics") or []
                ),
                "errors": list(result_context.get("generate_textures_errors") or []),
            },
        )

    @staticmethod
    def _artifact_ref(name: str, uri: str) -> TextureArtifactRef:
        path = Path(uri)
        digest = hashlib.sha256()
        with path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return TextureArtifactRef(name=name, uri=uri, sha256=digest.hexdigest())
