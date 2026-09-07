# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Iteration task for executing sub-workflows repeatedly until termination."""

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.agentic.workflows import Workflow
from world_understanding.optimization import (
    RefinementDecision,
    RefinementIteration,
    run_refinement,
)

logger = logging.getLogger(__name__)


class IterationTask(Task):
    """Task that executes a sub-workflow iteratively until termination.

    This task wraps a sub-workflow and executes it repeatedly in a loop until:
    - A task in the sub-workflow sets continue_iteration=False
    - Maximum iterations is reached
    - An error occurs

    The task manages iteration state, preserves history, and handles
    intermediate output directories for each iteration.

    **Feedback Loop Support:**
    The task automatically passes judge_critique from iteration N to iteration N+1
    as previous_judge_critique, enabling the VLM to improve based on feedback.

    Input context keys:
        - max_iterations: Maximum iterations (default: 10)
        - save_intermediate: Save per-iteration outputs (default: True)
        - intermediate_output_dir: Base dir for iteration outputs
          (default: "_iterations")
        - continue_iteration_key: Context key for continuation
          (default: "continue_iteration")

    Output context keys:
        - iteration_count: Total iterations executed
        - iteration_results: List of per-iteration results
        - final_iteration: Results from last iteration
        - termination_reason: Why stopped
          ("approved", "max_iterations", "error")
        - all_iteration_outputs: Output paths from all iterations

    Example:
        ```python
        # Create a sub-workflow to iterate
        iteration_sub_workflow = Workflow(
            tasks=[
                PredictTask(),
                ApplyTask(),
                JudgeTask(),  # Sets continue_iteration flag
            ],
            name="Predict-Apply-Judge Iteration"
        )

        # Wrap in iteration task
        iteration_task = IterationTask(
            sub_workflow=iteration_sub_workflow,
            max_iterations=5,
        )

        # Use in main workflow
        main_workflow = Workflow(
            tasks=[
                ConfigTask(),
                iteration_task,
                CompletionTask(),
            ]
        )
        ```
    """

    def __init__(
        self,
        sub_workflow: Workflow,
        max_iterations: int = 10,
        save_intermediate: bool = True,
        continue_iteration_key: str = "continue_iteration",
    ):
        """Initialize the iteration task.

        Args:
            sub_workflow: The workflow to execute in each iteration
            max_iterations: Maximum number of iterations to execute
            save_intermediate: Whether to save intermediate outputs per iteration
            continue_iteration_key: Context key that controls iteration continuation
        """
        self.name = "Iteration"
        self.description = "Execute sub-workflow iteratively until termination"
        self.sub_workflow = sub_workflow
        self.max_iterations = max_iterations
        self.save_intermediate = save_intermediate
        self.continue_iteration_key = continue_iteration_key

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Execute the sub-workflow iteratively.

        Args:
            context: Workflow context
            object_store: Optional object store

        Returns:
            Updated context with iteration results
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Get iteration configuration from context (can override constructor params)
        max_iterations = context.get("max_iterations", self.max_iterations)
        save_intermediate = context.get("save_intermediate", self.save_intermediate)
        intermediate_base_dir = Path(
            context.get("intermediate_output_dir", "_iterations")
        )

        listener.info(f"Starting iterative execution (max {max_iterations} iterations)")
        listener.info(f"  Sub-workflow: {self.sub_workflow.name}")
        listener.info(f"  Save intermediate: {save_intermediate}")

        iteration_count = 0
        iteration_results = []
        all_iteration_outputs = []
        iteration_setup_error: Exception | None = None

        # Preserve original context values that shouldn't change between iterations
        original_context = self._preserve_original_context(context)

        def evaluate_iteration(
            iteration: RefinementIteration[dict[str, Any]],
        ) -> dict[str, Any]:
            nonlocal iteration_count, iteration_setup_error
            iteration_count = iteration.iteration
            listener.info("")
            listener.info("=" * 80)
            listener.info(f"ITERATION {iteration_count}/{max_iterations}")
            listener.info("=" * 80)

            try:
                iteration_context = self._prepare_iteration_context(
                    context=iteration.state,
                    original_context=original_context,
                    iteration_num=iteration_count,
                    intermediate_base_dir=intermediate_base_dir,
                    save_intermediate=save_intermediate,
                )
            except Exception as error:
                iteration_setup_error = error
                raise

            listener.info(f"Executing sub-workflow: {self.sub_workflow.name}")
            iteration_result_context = self.sub_workflow.run(
                initial_context=iteration_context
            )
            iteration_result = self._extract_iteration_results(
                iteration_result_context, iteration_count
            )
            iteration_results.append(iteration_result)
            if "output_usd_path" in iteration_result_context:
                all_iteration_outputs.append(
                    str(iteration_result_context["output_usd_path"])
                )

            iteration.state.update(iteration_result_context)
            self._log_iteration_summary(iteration.state, iteration_result)
            return iteration_result_context

        def decide_iteration(
            _iteration: RefinementIteration[dict[str, Any]],
            _result: dict[str, Any],
            should_continue: bool,
        ) -> RefinementDecision:
            if should_continue:
                listener.info("")
                listener.info("→ Continuing to next iteration...")
                return RefinementDecision.continue_()

            listener.info("")
            listener.info(f"✓ Iteration stopped after {iteration_count} iteration(s)")
            reason = context.get("judge_reasoning", "Judge approved termination")
            listener.info(f"  Reason: {reason}")
            return RefinementDecision.approve()

        if max_iterations <= 0:
            context["termination_reason"] = "max_iterations"
            listener.warning(f"Reached maximum iterations ({max_iterations})")
        else:
            refinement = run_refinement(
                initial_state=context,
                max_iterations=max_iterations,
                evaluate=evaluate_iteration,
                judge=lambda _iteration, result: bool(
                    result.get(self.continue_iteration_key, False)
                ),
                decide=decide_iteration,
                revise=lambda iteration, _result, _judgment: iteration.state,
                caught_exceptions=(Exception,),
            )

            if iteration_setup_error is not None:
                raise iteration_setup_error

            context["termination_reason"] = refinement.termination_reason
            if refinement.error is not None:
                listener.error(
                    f"Error in iteration {iteration_count}: {refinement.error}"
                )
                context["iteration_error"] = str(refinement.error)
            elif refinement.termination_reason == "max_iterations":
                listener.warning(f"Reached maximum iterations ({max_iterations})")

        # Store final iteration results
        context["iteration_count"] = iteration_count
        context["iteration_results"] = iteration_results
        context["final_iteration"] = (
            iteration_results[-1] if iteration_results else None
        )
        context["all_iteration_outputs"] = all_iteration_outputs

        listener.info("")
        listener.info("=" * 80)
        listener.info("ITERATION SUMMARY")
        listener.info("=" * 80)
        listener.info(f"  Total iterations: {iteration_count}")
        listener.info(f"  Termination reason: {context.get('termination_reason')}")
        if iteration_results:
            final_score = iteration_results[-1].get("judge_score", "N/A")
            listener.info(f"  Final judge score: {final_score}")

        return context

    def _preserve_original_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Preserve original context values that shouldn't change between iterations.

        Args:
            context: Original context

        Returns:
            Dictionary with preserved values
        """
        # Keys to preserve across iterations
        preserve_keys = [
            "config_path",
            "config",  # Preserve config (includes system_prompt with critique)
            "dataset",
            "dataset_path",
            "dataset_size",
            "image_base_dir",
            "input_usd_path_original",  # Renamed to avoid confusion
            "vlm",
            "llm",
            "vlm_judge",
            "llm_judge",
            "vlm_config",
            "llm_config",
            "vlm_judge_config",
            "llm_judge_config",
            "judge_config",
            "max_workers",
            "usd_search_config",
            "aws_profile",
            "materials_mapping",
        ]

        preserved = {}
        for key in preserve_keys:
            if key in context:
                preserved[key] = context[key]

        # Save the original input USD path
        if "input_usd_path" in context and "input_usd_path_original" not in preserved:
            preserved["input_usd_path_original"] = context["input_usd_path"]

        return preserved

    def _prepare_iteration_context(
        self,
        context: dict[str, Any],
        original_context: dict[str, Any],
        iteration_num: int,
        intermediate_base_dir: Path,
        save_intermediate: bool,
    ) -> dict[str, Any]:
        """Prepare context for a specific iteration.

        Args:
            context: Current context
            original_context: Preserved original values
            iteration_num: Current iteration number
            intermediate_base_dir: Base directory for iteration outputs
            save_intermediate: Whether to save intermediate outputs

        Returns:
            Updated context for this iteration
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Start with current context (shallow copy to avoid
        # pickling issues with VLM/LLM objects)
        iteration_context = context.copy()

        # Restore original values that shouldn't change
        iteration_context.update(original_context)

        # Add iteration metadata
        iteration_context["iteration_count"] = iteration_num
        iteration_context["is_first_iteration"] = iteration_num == 1
        iteration_context["iteration_results_history"] = context.get(
            "iteration_results", []
        )

        # Pass previous judge critique to next iteration (for feedback loop)
        if iteration_num > 1 and "judge_critique" in context:
            iteration_context["previous_judge_critique"] = context["judge_critique"]
            listener.info("Passing previous judge critique to next iteration")

        # Pass per-prim feedback from prediction analysis (for targeted feedback)
        if iteration_num > 1 and "previous_prim_feedback" in context:
            # Keep the key name — it flows directly to VLMInferenceTask
            iteration_context["previous_prim_feedback"] = context[
                "previous_prim_feedback"
            ]
            listener.info(
                f"Passing per-prim feedback for "
                f"{len(context['previous_prim_feedback'])} prims to next iteration"
            )

            # Pass previous predictions path so VLMInferenceTask can carry
            # forward good predictions and only re-predict flagged prims
            prev_predictions = context.get("predictions_path")
            if prev_predictions:
                iteration_context["previous_predictions_path"] = str(prev_predictions)
                listener.info(
                    f"Passing previous predictions for selective re-prediction: "
                    f"{prev_predictions}"
                )

        # Pass resolved assignments (deterministic fixes from prediction analyzer)
        if iteration_num > 1 and "resolved_assignments" in context:
            iteration_context["resolved_assignments"] = context["resolved_assignments"]
            listener.info(
                f"Passing {len(context['resolved_assignments'])} resolved assignments "
                f"(will be applied directly without VLM)"
            )

        # Set up iteration-specific output paths if saving intermediate results
        if save_intermediate:
            iteration_dir = intermediate_base_dir / f"iteration_{iteration_num}"
            iteration_dir.mkdir(parents=True, exist_ok=True)

            # Update output paths for this iteration
            iteration_context["iteration_output_dir"] = iteration_dir
            iteration_context["output_dir"] = iteration_dir  # For report generation
            iteration_context["predictions_path"] = iteration_dir / "predictions.jsonl"
            iteration_context["output_usd_path"] = iteration_dir / "output.usd"

            # If rendering is enabled, set iteration-specific render directory
            if iteration_context.get("render_enabled"):
                iteration_context["render_output_dir"] = iteration_dir / "renders"

            listener.info(f"Iteration output directory: {iteration_dir}")
        else:
            # Overwrite same outputs each iteration
            listener.info("Intermediate outputs disabled (overwriting each iteration)")

        # For iteration 2+, use the output from previous iteration as input
        if iteration_num > 1:
            prev_iteration = context.get("iteration_results", [])
            if prev_iteration:
                prev_result = prev_iteration[-1]
                if "output_usd_path" in prev_result:
                    iteration_context["input_usd_path"] = prev_result["output_usd_path"]
                    listener.info(
                        f"Using output from iteration {iteration_num - 1} as input: "
                        f"{prev_result['output_usd_path']}"
                    )

        return iteration_context

    def _extract_iteration_results(
        self, context: dict[str, Any], iteration_num: int
    ) -> dict[str, Any]:
        """Extract relevant results from an iteration.

        Args:
            context: Context after iteration execution
            iteration_num: Iteration number

        Returns:
            Dictionary with iteration results
        """
        return {
            "iteration": iteration_num,
            "predictions_count": context.get("total_predictions", 0),
            "predictions_path": str(context.get("predictions_path", "")),
            "materials_applied_count": len(context.get("materials_applied", {})),
            "prims_with_materials": context.get("assignment_stats", {}).get(
                "total_prims", 0
            ),
            "output_usd_path": str(context.get("output_usd_path", "")),
            "rendered_images": context.get("rendered_image_paths", []),
            "judge_score": context.get("judge_score"),
            "judge_reasoning": context.get("judge_reasoning", ""),
            "continue_iteration": context.get(self.continue_iteration_key, False),
            "suggested_improvements": context.get("suggested_improvements", {}),
        }

    def _log_iteration_summary(
        self, context: dict[str, Any], iteration_result: dict[str, Any]
    ) -> None:
        """Log summary of an iteration.

        Args:
            context: Current context
            iteration_result: Results from the iteration
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        listener.info("")
        listener.info(f"Iteration {iteration_result['iteration']} Summary:")
        listener.info(f"  • Predictions: {iteration_result['predictions_count']}")
        listener.info(
            f"  • Materials applied: {iteration_result['materials_applied_count']}"
        )
        listener.info(
            f"  • Prims with materials: {iteration_result['prims_with_materials']}"
        )

        if iteration_result.get("judge_score") is not None:
            listener.info(f"  • Judge score: {iteration_result['judge_score']:.2f}")

        if iteration_result.get("judge_reasoning"):
            listener.info(f"  • Judge reasoning: {iteration_result['judge_reasoning']}")

        listener.info(
            f"  • Continue: {iteration_result.get('continue_iteration', False)}"
        )
