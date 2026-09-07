# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for evaluating predictions with LLM judge."""

import json
import logging
import re
import statistics
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore
from world_understanding.utils.response_content import extract_text_content

from material_agent.utils import calculate_metrics

logger = logging.getLogger(__name__)


class EvaluationTask(Task):
    """Evaluate predictions with LLM judge and calculate metrics."""

    def __init__(
        self,
        llm_judge: Any = None,
        dataset_path: Path | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        success_threshold: float = 4.0,
    ):
        """Initialize the evaluation task.

        Args:
            llm_judge: LLM instance for evaluation (None to use from context)
            dataset_path: Path to dataset with ground truth (None to use from context)
            temperature: Temperature for judge (None to use from context)
            max_tokens: Maximum tokens for judge response (None to use from context)
            success_threshold: Score threshold for success (default: 4.0)
        """
        self.llm_judge = llm_judge
        self.dataset_path = dataset_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.success_threshold = success_threshold
        self.name = "Evaluation"
        self.description = "Evaluate predictions with LLM judge"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Run judge evaluation and calculate metrics.

        Args:
            context: Workflow context with predictions path
            object_store: Storage for evaluations

        Returns:
            Updated context with evaluation results and metrics
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Resolve parameters from constructor or context
        llm_judge = (
            self.llm_judge if self.llm_judge is not None else context.get("llm_judge")
        )
        if llm_judge is None:
            raise ValueError("llm_judge not provided in constructor or context")

        # Get config values from context if not provided in constructor
        llm_judge_config = context.get("llm_judge_config", {})
        temperature = (
            self.temperature
            if self.temperature is not None
            else llm_judge_config.get("temperature")
        )
        max_tokens = (
            self.max_tokens
            if self.max_tokens is not None
            else llm_judge_config.get("max_tokens")
        )

        # Load predictions from file
        predictions_path = Path(context["predictions_path"])
        with open(predictions_path, encoding="utf-8") as f:
            predictions = [json.loads(line) for line in f if line.strip()]

        # Check if we need to load ground truth from dataset
        dataset_path = (
            self.dataset_path
            if self.dataset_path is not None
            else context.get("dataset_path")
        )

        if dataset_path:
            dataset_path = Path(dataset_path)
            if dataset_path.exists():
                # Load dataset for ground truth
                listener.info(f"Loading ground truth from dataset: {dataset_path}")
                with open(dataset_path, encoding="utf-8") as f:
                    dataset = [json.loads(line) for line in f if line.strip()]

                # Create dataset mapping for both ground truth and prompts
                ground_truth_map = {
                    entry["id"]: entry.get("ground_truth", "")
                    for entry in dataset
                    if "id" in entry
                }

                # Also create a complete dataset map for other fields like prompt
                dataset_map = {entry["id"]: entry for entry in dataset if "id" in entry}

                # Enrich predictions with ground truth and prompts from dataset
                for pred in predictions:
                    pred_id = pred.get("id")
                    if pred_id in ground_truth_map:
                        # Dataset ground truth takes precedence
                        pred["ground_truth"] = ground_truth_map[pred_id]
                    elif "ground_truth" not in pred:
                        listener.warning(
                            f"No ground truth found for {pred_id} in dataset"
                        )

                    # Also enrich with prompt from dataset if not present or is "N/A"
                    if pred_id in dataset_map:
                        dataset_entry = dataset_map[pred_id]
                        if not pred.get("prompt") or pred.get("prompt") == "N/A":
                            pred["prompt"] = dataset_entry.get("text", "N/A")

                listener.info(
                    f"✓ Loaded ground truth from dataset for "
                    f"{len(ground_truth_map)} entries"
                )
            else:
                listener.warning(
                    f"Dataset file not found: {dataset_path} - "
                    "using ground truth from predictions if available"
                )

        # Display evaluation start info
        # Removed redundant console message - already have listener.info below
        listener.info(f"Starting evaluation of {len(predictions)} predictions")

        if dataset_path and dataset_path.exists():
            listener.debug(f"Using ground truth from: {dataset_path}")

        listener.debug(
            f"Judge: {llm_judge.__class__.__name__} "
            f"(temp={temperature:.1f}, max_tokens={max_tokens})"
        )

        # Evaluate each prediction
        evaluations = []
        scores = []

        for idx, pred in enumerate(predictions, 1):
            if "ground_truth" not in pred:
                listener.warning(f"Skipping {pred['id']}: no ground truth")
                # Already have listener.warning above
                continue

            # Show which entry we're evaluating
            listener.debug(f"Evaluating {idx}/{len(predictions)}: {pred['id']}")

            # Evaluate this prediction
            eval_result = self._evaluate_single(
                pred, llm_judge, temperature, max_tokens, listener
            )
            evaluations.append(eval_result)
            scores.append(eval_result["score"])

            # Show result for this entry
            match_indicator = "✓" if eval_result.get("exact_match", False) else "✗"
            (
                "green"
                if eval_result["score"] >= 4
                else "yellow"
                if eval_result["score"] >= 3
                else "red"
            )
            listener.debug(
                f"→ Score: {eval_result['score']}/5 | Exact match: {match_indicator}"
            )

            # Show detailed progress every 10 evaluations
            if idx % 10 == 0 or idx == len(predictions):
                avg_score = statistics.mean(scores) if scores else 0
                listener.info(
                    f"Progress: {idx}/{len(predictions)} complete "
                    f"(avg score: {avg_score:.1f}/5.0)"
                )

        # Calculate metrics
        metrics = self._calculate_metrics(scores, evaluations)

        # Save evaluation results
        # Check if output_dir is provided in context, otherwise use predictions parent dir
        if "output_dir" in context and context["output_dir"]:
            output_dir = Path(context["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = predictions_path.parent
        results_file = output_dir / "evaluation_results.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {"evaluations": evaluations, "metrics": metrics},
                f,
                indent=2,
            )

        # Removed redundant message - already have listener.info below
        listener.info(f"Evaluation complete: {len(evaluations)} evaluations")

        # Store in object store if available
        if object_store:
            object_store.set("evaluations", evaluations)
            object_store.set("metrics", metrics)

        # Update context
        context["evaluation_complete"] = True
        context["metrics"] = metrics
        context["evaluation_path"] = str(results_file)

        return context

    def _evaluate_single(
        self,
        prediction: dict,
        llm_judge: Any,
        temperature: float | None,
        max_tokens: int | None,
        listener: Any,
    ) -> dict:
        """Evaluate a single prediction with the judge.

        Args:
            prediction: Prediction entry with materials and ground_truth
            llm_judge: LLM instance for evaluation
            temperature: Temperature for judge
            max_tokens: Maximum tokens for judge response

        Returns:
            Evaluation result with score and explanation
        """
        # Extract predicted material from new format
        materials_data = prediction.get("materials", {})
        if isinstance(materials_data, dict):
            predicted_material = materials_data.get("material", "")
            original_response = materials_data.get("original_response", "")
        else:
            # Fallback for old format or string format
            predicted_material = str(materials_data)
            original_response = ""

        # Create judge prompt
        judge_prompt = f"""Evaluate this material assignment:

Predicted Material: '{predicted_material}'
Ground Truth: '{prediction.get("ground_truth", "")}'
{f"VLM Full Response: {original_response[:500]}..." if original_response else ""}

Provide a single score (1-5) based on:
- Functional correctness: Did the VLM choose the correct material for the identified part?

Score guide:
5 - Perfect match with ground truth
4 - Very close material choice (e.g., similar material type)
3 - Reasonable alternative material
2 - Incorrect but plausible material choice
1 - Completely incorrect material

Respond with a JSON object containing:
- "score": integer from 1 to 5
- "explanation": brief explanation of the score"""

        try:
            messages = [
                SystemMessage(
                    content=(
                        "You are an expert judge evaluating material "
                        "assignments for 3D objects. Be fair but strict "
                        "in your evaluation."
                    )
                ),
                HumanMessage(content=judge_prompt),
            ]

            invoke_kwargs = {}
            if temperature is not None:
                invoke_kwargs["temperature"] = temperature
            if max_tokens is not None:
                invoke_kwargs["max_tokens"] = max_tokens

            response = llm_judge.invoke(messages, **invoke_kwargs)

            # Parse judge response
            judge_text = extract_text_content(response.content)

            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", judge_text, re.DOTALL)
            if json_match:
                judge_result = json.loads(json_match.group())
            else:
                # Fallback parsing
                score_match = re.search(r'"?score"?\s*:\s*(\d+)', judge_text)
                if score_match:
                    score = int(score_match.group(1))
                    judge_result = {
                        "score": score,
                        "explanation": judge_text,
                    }
                else:
                    raise ValueError("Could not parse judge response")

            # Check for exact match
            ground_truth = prediction.get("ground_truth", "")
            exact_match = predicted_material == ground_truth

            return {
                "id": prediction["id"],
                "predicted_material": predicted_material,
                "ground_truth": ground_truth,
                "exact_match": exact_match,
                "score": judge_result["score"],
                "explanation": judge_result.get("explanation", ""),
            }

        except Exception as e:
            listener.error(f"Error evaluating {prediction['id']}: {str(e)}")
            ground_truth = prediction.get("ground_truth", "")
            predicted_mat = (
                predicted_material if "predicted_material" in locals() else ""
            )
            return {
                "id": prediction["id"],
                "predicted_material": predicted_mat,
                "ground_truth": ground_truth,
                "exact_match": (
                    predicted_mat == ground_truth if predicted_mat else False
                ),
                "score": 0,
                "explanation": f"Evaluation error: {str(e)}",
            }

    def _calculate_metrics(
        self, scores: list[int], evaluations: list[dict]
    ) -> dict[str, Any]:
        """Calculate evaluation metrics.

        Args:
            scores: List of judge scores
            evaluations: List of evaluation results

        Returns:
            Dictionary with calculated metrics
        """
        # Use the utility function with the configured success threshold
        return calculate_metrics(scores, evaluations, self.success_threshold)
