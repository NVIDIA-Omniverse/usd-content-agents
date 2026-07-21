# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified pipeline executor task for Joint Agent.

This executor works with step configs that have already been prepared by
UnifiedPipelineConfigTask, so it doesn't need to create temporary config files
or load configs again.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.base_pipeline_executor import (
    BasePipelineExecutor,
    remove_legacy_pipeline_temp_with_safe_diagnostics,
    safe_diagnostic_steps,
    safe_diagnostic_text,
    safe_step_failure_message,
)
from world_understanding.agentic.config import normalize_yaml_config_value
from world_understanding.agentic.events import get_listener
from world_understanding.utils.credentials import (
    create_directory_with_safe_diagnostics,
    redact_sensitive_config,
)

from joint_agent.joint_rigger_options import (
    PREDICTION_FREE_JOINT_RIGGER_ADAPTERS,
    PREDICTION_OPTIONAL_JOINT_RIGGER_ADAPTERS,
)

logger = logging.getLogger(__name__)


def _raise_if_cancelled(
    context: dict[str, Any],
    listener: Any,
    step_name: str | None = None,
) -> None:
    """Stop at a pipeline step boundary when cancellation is requested."""

    cancel_checker = context.get("cancel_checker")
    if not callable(cancel_checker) or not cancel_checker():
        return
    cancelled_step = step_name or context.get("current_step") or "pipeline"
    listener.event(
        "step.cancelled",
        {
            "step_name": safe_diagnostic_text(cancelled_step),
            "message": "Pipeline cancellation requested",
        },
    )
    raise asyncio.CancelledError("Pipeline cancellation requested")


class UnifiedPipelineExecutorTask(BasePipelineExecutor):
    """Execute pipeline steps with pre-configured, auto-wired step configs.

    This executor works with the unified config system where:
    - All paths are already resolved by UnifiedPipelineConfigTask
    - Step configs are complete and ready to use
    - No additional config loading needed

    Input context keys:
        - steps_to_run: List of step names to execute
        - step_configs: Dictionary of pre-configured step configs
        - path_resolver: ProjectPathResolver instance
        - working_dir: Working directory
        - resume: Optional flag to resume from checkpoint

    Output context keys:
        - pipeline_results: Dictionary of results from each step
        - pipeline_state: Final pipeline state
    """

    def __init__(self):
        """Initialize the unified pipeline executor."""
        self.name = "UnifiedPipelineExecutor"
        self.description = "Execute pipeline with auto-wired configs"

    # ========== Required Abstract Method Implementations ==========

    def _get_step_list_key(self) -> str:
        """Return context key for step list."""
        return "steps_to_run"

    def _get_required_context_keys(self) -> list[str]:
        """Return required context keys."""
        return ["steps_to_run", "step_configs"]

    def _get_state_file(self, context: dict[str, Any]) -> Path:
        """Return path to pipeline state file."""
        working_dir = context.get("working_dir", Path.cwd())
        return Path(working_dir) / ".pipeline_state.json"

    # ========== Pipeline Execution Logic ==========

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Execute pipeline steps in sequence.

        Args:
            context: Workflow context
            object_store: Optional object store

        Returns:
            Updated context with pipeline results
        """
        listener = get_listener(context, logger_name=__name__)
        _raise_if_cancelled(context, listener)

        steps_to_run = context.get("steps_to_run", [])
        step_configs = context.get("step_configs", {})
        working_dir = context.get("working_dir", Path.cwd())
        resume = context.get("resume", False)
        clean = context.get("clean", False)

        if not steps_to_run:
            raise ValueError("No steps to run in pipeline")

        # Clean working directory if requested. The base implementation keeps
        # both safety-check and filesystem diagnostics value-safe.
        if clean:
            self._clean_directories(context)

        # Ensure working directory exists
        create_directory_with_safe_diagnostics(
            working_dir,
            label="pipeline working directory",
        )
        if remove_legacy_pipeline_temp_with_safe_diagnostics(working_dir):
            logger.warning(
                "Removed retained legacy .pipeline_temp before pipeline startup"
            )

        # Get session info
        session_id = context.get("session_id")
        project_name = context.get("project_name")

        state_file = Path(working_dir) / ".pipeline_state.json"
        pipeline_state = self._initialize_pipeline_state(context, resume)

        safe_session_id = safe_diagnostic_text(session_id)
        safe_project_name = safe_diagnostic_text(project_name)
        safe_working_dir = safe_diagnostic_text(working_dir)
        safe_steps = safe_diagnostic_steps(steps_to_run)

        # Display pipeline start
        logger.info("=" * 80)
        logger.info("PIPELINE STARTING")
        logger.info("=" * 80)
        logger.info("Session ID: %s", safe_session_id)
        logger.info("Project: %s", safe_project_name)
        logger.info("Working Directory: %s", safe_working_dir)
        logger.info("Steps: %s", ", ".join(safe_steps))
        logger.info("=" * 80)

        # Emit pipeline start event
        listener.event(
            "pipeline.started",
            {
                "session_id": safe_session_id,
                "project_name": safe_project_name,
                "working_dir": safe_working_dir,
                "steps": safe_steps,
            },
        )

        # Execute each step
        for i, step_name in enumerate(steps_to_run, 1):
            safe_step_name = safe_diagnostic_text(step_name)
            _raise_if_cancelled(context, listener, step_name)
            # Skip if already completed (resume mode)
            if resume and step_name in pipeline_state["completed_steps"]:
                logger.info(
                    "[%d/%d] Skipping %s (already completed)",
                    i,
                    len(steps_to_run),
                    safe_step_name,
                )
                continue

            pipeline_state["current_step"] = step_name
            event_listener = context.get("event_listener")

            try:
                logger.info(
                    "\n[%d/%d] Executing step: %s",
                    i,
                    len(steps_to_run),
                    safe_step_name,
                )

                if event_listener:
                    event_listener.event(
                        "step.started",
                        {
                            "step_name": safe_step_name,
                            "step_index": i,
                            "total_steps": len(steps_to_run),
                        },
                    )

                # Execute the step
                step_config = step_configs[step_name]
                outputs = self._execute_step(
                    step_name, step_config, context, object_store, pipeline_state
                )

                # Mark step as completed
                pipeline_state["completed_steps"].append(step_name)
                pipeline_state["step_outputs"][step_name] = outputs
                pipeline_state["current_step"] = None

                # Save checkpoint
                self._save_checkpoint(pipeline_state, state_file)

                logger.info("Step '%s' completed successfully", safe_step_name)

                if event_listener:
                    event_listener.event(
                        "step.completed",
                        {
                            "step_name": safe_step_name,
                            "outputs": redact_sensitive_config(outputs),
                        },
                    )
                _raise_if_cancelled(context, listener)

            except Exception as error:
                safe_error = safe_step_failure_message(error)
                logger.error("Step '%s' failed: %s", safe_step_name, safe_error)
                pipeline_state["failed_steps"].append(step_name)
                pipeline_state["current_step"] = None

                self._save_checkpoint(pipeline_state, state_file)

                if event_listener:
                    event_listener.event(
                        "step.failed",
                        {"step_name": safe_step_name, "error": safe_error},
                    )

                raise RuntimeError(
                    f"Pipeline failed at step '{safe_step_name}': {safe_error}"
                ) from None

        # Pipeline completed
        pipeline_state["current_step"] = None
        self._save_checkpoint(pipeline_state, state_file)

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Session ID: %s", safe_session_id)
        logger.info(
            "Completed Steps: %s",
            ", ".join(safe_diagnostic_steps(pipeline_state["completed_steps"])),
        )
        logger.info("=" * 80)

        listener.event(
            "pipeline.completed",
            {
                "session_id": safe_session_id,
                "project_name": safe_project_name,
                "completed_steps": safe_diagnostic_steps(
                    pipeline_state["completed_steps"]
                ),
            },
        )

        context["pipeline_results"] = pipeline_state["step_outputs"]
        context["pipeline_state"] = "completed"

        return context

    def _execute_step(
        self,
        step_name: str,
        step_config: dict[str, Any],
        context: dict[str, Any],
        object_store: Any,
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single pipeline step.

        Args:
            step_name: Name of the step
            step_config: Pre-configured step configuration
            context: Workflow context
            object_store: Optional object store
            pipeline_state: Current pipeline state

        Returns:
            Dictionary with relevant outputs
        """
        # Each workflow receives its own YAML-equivalent copy. Auto-wiring must
        # never mutate the shared step_configs mapping because API callers can
        # run multiple pipelines concurrently with different credentials.
        step_config = self._prepare_runtime_config(step_config)

        # Auto-wire outputs from previous steps if needed
        step_outputs = pipeline_state.get("step_outputs", {})
        working_dir = Path(context.get("working_dir", Path.cwd()))

        # Auto-wire optimized USD for identify_asset and build_dataset_usd steps
        # When optimize_usd has run, use the optimized USD for downstream steps
        if step_name in ("identify_asset", "build_dataset_usd"):
            if "optimize_usd" in step_outputs:
                optimized_usd_path = step_outputs["optimize_usd"].get(
                    "optimized_usd_path"
                )
                if optimized_usd_path:
                    logger.info(
                        "Auto-wired usd_path for %s from optimize_usd: %s",
                        step_name,
                        optimized_usd_path,
                    )
                    step_config["usd_path"] = str(optimized_usd_path)

        # Auto-wire identification results into analyze_structure step
        if step_name == "analyze_structure":
            if "identify_asset" in step_outputs:
                identification = step_outputs["identify_asset"].get("identification")
                if identification:
                    step_config["asset_type"] = identification.get("asset_type", "")
                    step_config["asset_subtype"] = identification.get(
                        "asset_subtype", ""
                    )
                    step_config["asset_confidence"] = identification.get(
                        "confidence", ""
                    )
                # Also pass identification_path so analyze_structure can find
                # the preview images rendered by identify_asset
                id_path = step_outputs["identify_asset"].get("identification_path")
                if id_path:
                    step_config["identification_path"] = str(id_path)
                    logger.info(
                        "Auto-wired identification for %s: %s / %s (path: %s)",
                        step_name,
                        step_config.get("asset_type", ""),
                        step_config.get("asset_subtype", ""),
                        id_path,
                    )

        # Auto-wire identification results into prepare_dataset step
        if step_name == "build_dataset_prepare_dataset":
            if "identify_asset" in step_outputs:
                identification = step_outputs["identify_asset"].get("identification")
                if identification:
                    asset_type = identification.get("asset_type", "")
                    asset_subtype = identification.get("asset_subtype", "")
                    if asset_type:
                        # Inject asset type into prompts context
                        prompts = step_config.get("prompts", {})
                        system_prompt = prompts.get("system", "")
                        if system_prompt and "{asset_type}" not in system_prompt:
                            # Prepend identification context
                            id_context = f"This is a {asset_type}"
                            if asset_subtype:
                                id_context += f" ({asset_subtype})"
                            id_context += ". "
                            prompts["system"] = id_context + system_prompt
                            step_config["prompts"] = prompts
                        logger.info(
                            "Auto-wired identification for %s: %s / %s",
                            step_name,
                            asset_type,
                            asset_subtype,
                        )

        # Auto-wire structure assignments into prepare_dataset step
        if step_name == "build_dataset_prepare_dataset":
            if "analyze_structure" in step_outputs:
                structure_path = step_outputs["analyze_structure"].get(
                    "structure_assignments_path"
                )
                if structure_path:
                    step_config["structure_assignments_path"] = str(structure_path)
                    logger.info(
                        "Auto-wired structure_assignments for %s: %s",
                        step_name,
                        structure_path,
                    )

        # Auto-wire prediction output into the consistency pass
        if step_name == "consistency_pass":
            if "predict" in step_outputs:
                predictions_path = step_outputs["predict"].get("predictions_path")
                if predictions_path:
                    step_config["predictions_path"] = str(predictions_path)
                    predict_output_key = step_outputs["predict"].get("output_key")
                    if predict_output_key and "output_key" not in step_config:
                        step_config["output_key"] = predict_output_key
                    logger.info(
                        "Auto-wired predictions_path for %s: %s",
                        step_name,
                        predictions_path,
                    )

        # Auto-wire post-consistency predictions into articulation candidate inference.
        if step_name == "infer_articulation_candidates":
            if "build_dataset_prepare_dataset" in step_outputs:
                dataset_jsonl_path = step_outputs["build_dataset_prepare_dataset"].get(
                    "dataset_jsonl_path"
                )
                if dataset_jsonl_path and "dataset_path" not in step_config:
                    step_config["dataset_path"] = str(dataset_jsonl_path)
                    logger.info(
                        "Auto-wired dataset_path for %s from "
                        "build_dataset_prepare_dataset dataset_jsonl_path: %s",
                        step_name,
                        dataset_jsonl_path,
                    )
            predictions_path = None
            if "consistency_pass" in step_outputs:
                predictions_path = step_outputs["consistency_pass"].get(
                    "consistent_predictions_path"
                )
            if predictions_path:
                step_config["predictions_path"] = str(predictions_path)
                logger.info(
                    "Auto-wired predictions_path for %s from consistency_pass: %s",
                    step_name,
                    predictions_path,
                )
            elif "predict" in step_outputs:
                predictions_path = step_outputs["predict"].get("predictions_path")
                if predictions_path:
                    step_config["predictions_path"] = str(predictions_path)
                    predict_output_key = step_outputs["predict"].get("output_key")
                    if predict_output_key and "output_key" not in step_config:
                        step_config["output_key"] = predict_output_key
                    logger.info(
                        "Auto-wired predictions_path for %s from predict: %s",
                        step_name,
                        predictions_path,
                    )

        # Auto-wire restore_usd inputs from optimize_usd and prediction steps
        if step_name == "restore_usd":
            if "optimize_usd" in step_outputs:
                opt_outputs = step_outputs["optimize_usd"]
                original_usd_path = opt_outputs.get("original_usd_path")
                optimization_metadata = opt_outputs.get("optimization_metadata")
                if original_usd_path:
                    step_config["original_usd_path"] = str(original_usd_path)
                    logger.info(
                        "Auto-wired original_usd_path for restore_usd: %s",
                        original_usd_path,
                    )
                if optimization_metadata:
                    step_config["optimization_metadata"] = optimization_metadata
                    logger.info("Auto-wired optimization_metadata for restore_usd")
            else:
                logger.warning(
                    "restore_usd requested but optimize_usd has not run - "
                    "skipping restoration (no optimization metadata available)"
                )
                return {"restore_skipped": True, "reason": "no optimization metadata"}

            if "consistency_pass" in step_outputs:
                predictions_path = step_outputs["consistency_pass"].get(
                    "consistent_predictions_path"
                )
                if predictions_path:
                    step_config["predictions_path"] = str(predictions_path)
                    logger.info(
                        "Auto-wired predictions_path for restore_usd from "
                        "consistency_pass: %s",
                        predictions_path,
                    )
            elif "predict" in step_outputs:
                predictions_path = step_outputs["predict"].get("predictions_path")
                if predictions_path:
                    step_config["predictions_path"] = str(predictions_path)
                    logger.info(
                        "Auto-wired predictions_path for restore_usd: %s",
                        predictions_path,
                    )

            # Set output path in working dir
            if "output_predictions_path" not in step_config:
                step_config["output_predictions_path"] = str(
                    working_dir / "restored_predictions.jsonl"
                )

        # Auto-wire Joint Rigger inputs from namespace-compatible predictions.
        if step_name == "apply_joint_rigger":
            adapter = step_config.get("adapter", "mock")
            optimize_outputs = step_outputs.get("optimize_usd", {})
            optimized_usd_path = optimize_outputs.get("optimized_usd_path")
            original_usd_path = optimize_outputs.get("original_usd_path")
            selected_prediction_source = None
            prediction_sources = (
                (
                    ("consistency_pass", "consistent_predictions_path"),
                    ("predict", "predictions_path"),
                )
                if adapter in PREDICTION_OPTIONAL_JOINT_RIGGER_ADAPTERS
                else (
                    ("restore_usd", "restored_predictions_path"),
                    ("consistency_pass", "consistent_predictions_path"),
                    ("predict", "predictions_path"),
                )
            )
            has_explicit_optional_predictions = (
                adapter in PREDICTION_OPTIONAL_JOINT_RIGGER_ADAPTERS
                and bool(step_config.get("predictions_path"))
            )
            if (
                adapter not in PREDICTION_FREE_JOINT_RIGGER_ADAPTERS
                and not has_explicit_optional_predictions
            ):
                for source_step, output_key in prediction_sources:
                    if source_step not in step_outputs:
                        continue
                    predictions_path = step_outputs[source_step].get(output_key)
                    if predictions_path:
                        selected_prediction_source = source_step
                        step_config["predictions_path"] = str(predictions_path)
                        if source_step == "restore_usd" and original_usd_path:
                            step_config["input_usd_path"] = str(original_usd_path)
                        elif source_step != "restore_usd" and optimized_usd_path:
                            step_config["input_usd_path"] = str(optimized_usd_path)
                            logger.info(
                                "Auto-wired input_usd_path for %s from "
                                "optimize_usd: %s",
                                step_name,
                                optimized_usd_path,
                            )
                        logger.info(
                            "Auto-wired predictions_path for %s from %s: %s",
                            step_name,
                            source_step,
                            predictions_path,
                        )
                        break

            if selected_prediction_source == "restore_usd":
                removed_candidates_path = step_config.pop(
                    "articulation_candidates_path",
                    None,
                )
                if removed_candidates_path:
                    logger.info(
                        "Cleared articulation_candidates_path for %s because "
                        "restored predictions use the original USD namespace",
                        step_name,
                    )
            elif "infer_articulation_candidates" in step_outputs:
                candidates_path = step_outputs["infer_articulation_candidates"].get(
                    "articulation_candidates_path"
                )
                if (
                    candidates_path
                    and "articulation_candidates_path" not in step_config
                ):
                    step_config["articulation_candidates_path"] = str(candidates_path)
                    logger.info(
                        "Auto-wired articulation_candidates_path for %s: %s",
                        step_name,
                        candidates_path,
                    )
                using_inferred_candidates = bool(candidates_path) and str(
                    step_config.get("articulation_candidates_path")
                ) == str(candidates_path)
                if (
                    (
                        adapter in PREDICTION_FREE_JOINT_RIGGER_ADAPTERS
                        or (
                            adapter in PREDICTION_OPTIONAL_JOINT_RIGGER_ADAPTERS
                            and selected_prediction_source is None
                        )
                    )
                    and using_inferred_candidates
                    and optimized_usd_path
                ):
                    step_config["input_usd_path"] = str(optimized_usd_path)
                    logger.info(
                        "Paired inferred Stage 2 candidates with optimized USD "
                        "namespace for %s: %s",
                        step_name,
                        optimized_usd_path,
                    )

        if step_name == "author_physics_schemas":
            rigger_outputs = step_outputs.get("apply_joint_rigger", {})
            rigged_usd_path = rigger_outputs.get("rigged_usd_path")
            if rigged_usd_path:
                step_config["input_usd_path"] = str(rigged_usd_path)
                logger.info(
                    "Auto-wired input_usd_path for %s from apply_joint_rigger: %s",
                    step_name,
                    rigged_usd_path,
                )
            diagnostics_path = rigger_outputs.get("joint_rigger_diagnostics_path")
            if diagnostics_path:
                step_config["stage2_diagnostics_path"] = str(diagnostics_path)
            validation_path = rigger_outputs.get("joint_rigger_validation_path")
            if validation_path:
                step_config["stage2_validation_path"] = str(validation_path)

        # Import workflows
        from joint_agent.workflows import (
            create_analyze_structure_workflow_from_config,
            create_apply_joint_rigger_workflow_from_config,
            create_articulation_candidates_workflow_from_config,
            create_author_physics_schemas_workflow_from_config,
            create_consistency_pass_workflow_from_config,
            create_identify_asset_workflow_from_config,
            create_optimize_usd_workflow_from_config,
            create_prediction_workflow_from_config,
            create_prepare_dataset_workflow_from_config,
            create_restore_usd_workflow_from_config,
            create_usd_data_preparation_workflow_from_config,
        )

        # Map step names to workflow factories
        workflow_map = {
            "optimize_usd": create_optimize_usd_workflow_from_config,
            "build_dataset_usd": create_usd_data_preparation_workflow_from_config,
            "identify_asset": create_identify_asset_workflow_from_config,
            "analyze_structure": create_analyze_structure_workflow_from_config,
            "build_dataset_prepare_dataset": create_prepare_dataset_workflow_from_config,
            "predict": create_prediction_workflow_from_config,
            "consistency_pass": create_consistency_pass_workflow_from_config,
            "infer_articulation_candidates": (
                create_articulation_candidates_workflow_from_config
            ),
            "restore_usd": create_restore_usd_workflow_from_config,
            "apply_joint_rigger": create_apply_joint_rigger_workflow_from_config,
            "author_physics_schemas": (
                create_author_physics_schemas_workflow_from_config
            ),
        }

        if step_name not in workflow_map:
            raise ValueError(f"Unknown step: {step_name}")

        # Create workflow
        workflow = workflow_map[step_name]()

        # Keep the original config path only as a relative-path anchor. Runtime
        # values, including credentials, stay in memory and are never serialized
        # to .pipeline_temp.
        step_context: dict[str, Any] = {"config_dict": step_config}
        if config_path := context.get("config_path"):
            step_context["config_path"] = str(config_path)

        if "event_listener" in context:
            step_context["event_listener"] = context["event_listener"]

        # Extract and pass report compression configuration if present
        if "report" in step_config:
            report_config = step_config["report"]
            if isinstance(report_config, dict):
                # Map report config keys to context keys
                if "image_max_size" in report_config:
                    step_context["report_image_max_size"] = report_config[
                        "image_max_size"
                    ]
                if "image_format" in report_config:
                    step_context["report_image_format"] = report_config["image_format"]
                if "image_quality" in report_config:
                    step_context["report_image_quality"] = report_config[
                        "image_quality"
                    ]

        # Execute workflow
        logger.debug("Running workflow for %s", step_name)
        result = workflow.run(step_context)

        if not result:
            raise RuntimeError(f"Step '{step_name}' returned empty result")

        if result.get("error") or result.get("workflow_terminated"):
            failed_task = result.get("failed_task", "unknown")
            error_msg = result.get("error", "Workflow terminated")
            raise RuntimeError(
                f"Step '{step_name}' failed at '{failed_task}': {error_msg}"
            )

        # Extract outputs
        return self._extract_step_outputs(step_name, result)

    def _prepare_runtime_config(self, step_config: dict[str, Any]) -> dict[str, Any]:
        """Return an isolated, YAML-equivalent runtime configuration."""
        runtime_config: dict[str, Any] = {}
        for key, value in step_config.items():
            if key == "renderer" and isinstance(value, dict):
                value = {k: v for k, v in value.items() if not k.startswith("_")}
            runtime_config[key] = self._make_yaml_safe(value)
        return runtime_config

    def _make_yaml_safe(self, value: Any) -> Any:
        """Normalize explicit YAML-equivalent values without rendering objects."""
        return normalize_yaml_config_value(value)

    def _extract_step_outputs(
        self, step_name: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract relevant outputs from step result.

        Args:
            step_name: Name of the step
            result: Step execution result

        Returns:
            Dictionary with relevant outputs
        """
        outputs = {}

        if step_name == "optimize_usd":
            outputs["optimized_usd_path"] = result.get("optimized_usd_path")
            outputs["optimization_metadata"] = result.get("optimization_metadata")
            outputs["optimization_success"] = result.get("optimization_success")
            outputs["original_usd_path"] = result.get("original_usd_path")

        elif step_name == "identify_asset":
            outputs["identification"] = result.get("identification")
            outputs["identification_path"] = result.get("identification_path")

        elif step_name == "analyze_structure":
            outputs["structure_assignments"] = result.get("structure_assignments")
            outputs["structure_assignments_path"] = result.get(
                "structure_assignments_path"
            )
            outputs["structure_metadata"] = result.get("structure_metadata")

        elif step_name == "build_dataset_usd":
            outputs["output_dir"] = result.get("output_dir")
            outputs["usd_dataset_dir"] = result.get("output_dir")

        elif step_name == "build_dataset_prepare_dataset":
            outputs["dataset_path"] = result.get("dataset_path")
            outputs["dataset_jsonl_path"] = result.get("dataset_jsonl_path")

        elif step_name == "predict":
            outputs["predictions_path"] = result.get("predictions_path")
            outputs["predictions_count"] = result.get("predictions_count")
            outputs["output_key"] = result.get("output_key")

        elif step_name == "consistency_pass":
            outputs["consistent_predictions_path"] = result.get(
                "consistent_predictions_path"
            )
            outputs["consistency_stats_path"] = result.get("consistency_stats_path")
            outputs["consistency_stats"] = result.get("consistency_stats")
            outputs["predictions_count"] = result.get("predictions_count")

        elif step_name == "infer_articulation_candidates":
            outputs["articulation_candidates_path"] = result.get(
                "articulation_candidates_path"
            )
            outputs["articulation_report_path"] = result.get("articulation_report_path")
            outputs["articulation_adjudications_path"] = result.get(
                "articulation_adjudications_path"
            )
            outputs["articulation_topology_reconciliation_status"] = result.get(
                "articulation_topology_reconciliation_status"
            )
            outputs["articulation_summary"] = result.get("articulation_summary")
            outputs["articulation_candidate_count"] = result.get(
                "articulation_candidate_count"
            )

        elif step_name == "restore_usd":
            outputs["restored_predictions_path"] = result.get(
                "restored_predictions_path"
            )
            outputs["restore_success"] = result.get("restore_success")
            outputs["predictions_count"] = result.get("predictions_count")
            outputs["restore_stats"] = result.get("restore_stats")

        elif step_name == "apply_joint_rigger":
            outputs["joint_rigger_status"] = result.get("joint_rigger_status")
            outputs["rigged_usd_path"] = result.get("rigged_usd_path")
            outputs["joint_rigger_diagnostics_path"] = result.get(
                "joint_rigger_diagnostics_path"
            )
            outputs["joint_rigger_validation_path"] = result.get(
                "joint_rigger_validation_path"
            )
            outputs["joint_rigger_warnings"] = result.get("joint_rigger_warnings")
            outputs["joint_rigger_errors"] = result.get("joint_rigger_errors")
            outputs["joint_rigger_candidate_readiness"] = result.get(
                "joint_rigger_candidate_readiness"
            )
            outputs["authored_joint_count"] = result.get("authored_joint_count")
            outputs["apply_joint_rigger_skipped"] = result.get(
                "apply_joint_rigger_skipped"
            )

        elif step_name == "author_physics_schemas":
            outputs["physics_authoring_status"] = result.get("physics_authoring_status")
            outputs["physics_ready_usd_path"] = result.get("physics_ready_usd_path")
            outputs["physics_authoring_diagnostics_path"] = result.get(
                "physics_authoring_diagnostics_path"
            )
            outputs["physics_authoring_validation_path"] = result.get(
                "physics_authoring_validation_path"
            )
            outputs["physics_authoring_identity"] = result.get(
                "physics_authoring_identity"
            )
            outputs["physics_authoring_schema_deltas"] = result.get(
                "physics_authoring_schema_deltas"
            )
            outputs["physics_authoring_rule_deltas"] = result.get(
                "physics_authoring_rule_deltas"
            )
            outputs["physics_authoring_expected_residuals"] = result.get(
                "physics_authoring_expected_residuals"
            )

        return outputs
