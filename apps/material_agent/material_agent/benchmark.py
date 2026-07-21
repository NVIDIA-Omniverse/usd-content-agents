# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark implementations for Material Agent evaluation."""

import json
import logging
import re
import statistics
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.table import Table

from material_agent.functions.inference import batch_assign_materials

# Load environment variables
load_dotenv()

console = Console()
logger = logging.getLogger("material_agent.benchmark")


class BaseBenchmark(ABC):
    """Base class for all Material Agent benchmarks."""

    def __init__(self, name: str = "base"):
        """Initialize base benchmark.

        Args:
            name: Name of the benchmark type
        """
        self.name = name
        self.logger = logging.getLogger(f"material_agent.benchmark.{name}")

    @abstractmethod
    def run_inference(self, dataset_path: Path, output_dir: Path | None = None) -> Path:
        """Run inference on dataset and save predictions.

        Args:
            dataset_path: Path to dataset file
            output_dir: Directory to save predictions

        Returns:
            Path to predictions file
        """
        pass  # pragma: no cover - abstract interface

    @abstractmethod
    def evaluate_with_judge(
        self, predictions_path: Path, output_dir: Path | None = None
    ) -> dict[str, Any]:
        """Evaluate predictions using judge.

        Args:
            predictions_path: Path to predictions file
            output_dir: Directory to save evaluation results

        Returns:
            Dictionary with evaluation metrics
        """
        pass  # pragma: no cover - abstract interface

    def run_full_benchmark(
        self, dataset_path: Path, output_dir: Path | None = None
    ) -> dict[str, Any]:
        """Run complete benchmark pipeline: inference + evaluation.

        Args:
            dataset_path: Path to dataset file
            output_dir: Directory for all outputs

        Returns:
            Dictionary with evaluation metrics
        """
        console.print(
            f"[bold magenta]Starting {self.name.upper()} Benchmark[/bold magenta]\n"
        )

        # Step 1: Run inference
        console.print("[bold]Step 1: Running Inference[/bold]")
        predictions_path = self.run_inference(dataset_path, output_dir)

        # Step 2: Evaluate with judge
        console.print("\n[bold]Step 2: Evaluating with Judge[/bold]")
        metrics = self.evaluate_with_judge(predictions_path, output_dir)

        console.print("\n[bold green]✓ Benchmark Complete![/bold green]")
        return metrics


class VLMBenchmark(BaseBenchmark):
    """VLM Benchmark runner for basic part identification and material selection."""

    def __init__(
        self,
        vlm: Any,
        llm_judge: Any,
        vlm_temperature: float | None = 0.7,
        vlm_max_tokens: int | None = 1024,
        llm_judge_temperature: float | None = 0.7,
        llm_judge_max_tokens: int | None = 1024,
        llm: Any | None = None,
        system_prompt: str | None = None,
    ):
        """Initialize VLM benchmark with VLM and LLM instances.

        Args:
            vlm: VLM instance for material assignment
            llm_judge: LLM instance for evaluation judge
            vlm_temperature: Temperature for VLM inference (default: 0.7)
            vlm_max_tokens: Maximum tokens for VLM response (default: 1024)
            llm_judge_temperature: Temperature for LLM judge (default: 0.7)
            llm_judge_max_tokens: Maximum tokens for LLM judge response (default: 1024)
            llm: Optional LLM for parsing and other tasks (uses llm_judge if not provided)
            system_prompt: Optional custom system prompt for VLM
        """
        super().__init__(name="vlm")
        self.vlm = vlm
        self.llm_judge = llm_judge
        self.vlm_temperature = vlm_temperature
        self.vlm_max_tokens = vlm_max_tokens
        self.llm_judge_temperature = llm_judge_temperature
        self.llm_judge_max_tokens = llm_judge_max_tokens
        self.llm = llm or llm_judge  # Use judge LLM if not provided
        self.system_prompt = system_prompt

        self.logger.info(
            "VLM Benchmark initialized with provided VLM and LLM instances"
        )
        self.logger.info(
            "Structured output is enabled (mandatory) for material assignments"
        )

    def run_inference(self, dataset_path: Path, output_dir: Path | None = None) -> Path:
        """Run VLM inference on dataset and save predictions.

        Args:
            dataset_path: Path to JSONL dataset file
            output_dir: Directory to save predictions (default: same as dataset)

        Returns:
            Path to predictions file
        """
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        # Setup output directory
        if output_dir is None:
            output_dir = dataset_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Output file path
        predictions_file = output_dir / "predictions.jsonl"

        # Clear existing predictions file
        if predictions_file.exists():
            predictions_file.unlink()

        # Load dataset
        console.print(f"[blue]Loading dataset from {dataset_path}[/blue]")
        with open(dataset_path, encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f]

        console.print(f"[green]Found {len(dataset)} test cases[/green]")
        self.logger.info(f"Starting VLM inference for {len(dataset)} test cases")

        # Progress callback
        def on_progress(entry_id: str, response: str) -> None:
            """Log progress after processing each entry."""
            self.logger.debug(f"Completed processing entry: {entry_id}")

        # Error callback
        def on_error(entry_id: str, error: str) -> None:
            """Handle errors during processing."""
            self.logger.error(f"Error processing {entry_id}: {error}")
            console.print(f"[red]Error processing {entry_id}: {error}[/red]")

        # Use the core batch processing function
        console.print("[cyan]Running VLM inference...[/cyan]")
        results = batch_assign_materials(
            vlm=self.vlm,
            entries=dataset,
            llm=self.llm,
            image_base_dir=dataset_path.parent,
            system_prompt=self.system_prompt,
            temperature=self.vlm_temperature,
            max_tokens=self.vlm_max_tokens,
            on_progress=on_progress,
            on_error=on_error,
        )

        # Save predictions to file
        predictions = []
        for result in results:
            if result["status"] == "success":
                # Find the original entry to get ground truth
                original_entry = next(
                    (e for e in dataset if e["id"] == result["id"]), {}
                )
                prediction = {
                    "id": result["id"],
                    "vlm_response": result["vlm_response"],
                    "ground_truth": original_entry.get("ground_truth", ""),
                }
                predictions.append(prediction)

                # Write incrementally to file
                with open(predictions_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(prediction) + "\n")

        console.print(
            f"[green]✓ Inference complete. Predictions saved to {predictions_file}[/green]"
        )
        console.print(
            f"[dim]Processed {len(predictions)}/{len(dataset)} entries successfully[/dim]"
        )
        self.logger.info(
            f"VLM inference complete: {len(predictions)}/{len(dataset)} successful"
        )
        return predictions_file

    def evaluate_with_judge(
        self, predictions_path: Path, output_dir: Path | None = None
    ) -> dict[str, Any]:
        """Evaluate predictions using LLM-as-judge.

        Args:
            predictions_path: Path to predictions JSONL file
            output_dir: Directory to save evaluation results

        Returns:
            Dictionary with evaluation metrics
        """
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

        # Setup output directory
        if output_dir is None:
            output_dir = predictions_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load predictions
        console.print(f"[blue]Loading predictions from {predictions_path}[/blue]")
        with open(predictions_path, encoding="utf-8") as f:
            predictions = [json.loads(line) for line in f]

        # Evaluate each prediction
        evaluations = []
        scores = []

        console.print("[cyan]Running LLM judge evaluation...[/cyan]")
        self.logger.info(
            f"Starting LLM judge evaluation for {len(predictions)} predictions"
        )

        for idx, pred in enumerate(predictions, 1):
            # Create judge prompt
            judge_prompt = f"""Evaluate this material assignment:

VLM Response: '{pred["vlm_response"]}'
Ground Truth: '{pred["ground_truth"]}'

Provide a single score (1-5) based on:
- Functional correctness: Did the VLM choose the correct material for the identified part?
- Reasoning quality: Is the logic connecting the visual observation to material choice sound?

Score guide:
5 - Correct material with excellent reasoning
4 - Correct material with good reasoning
3 - Correct material with weak reasoning OR reasonable alternative with excellent reasoning
2 - Incorrect but plausible material choice
1 - Incorrect material with poor reasoning

Respond with a JSON object containing:
- "score": integer from 1 to 5
- "explanation": brief explanation of the score"""

            # Run LLM judge
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
                if self.llm_judge_temperature is not None:
                    invoke_kwargs["temperature"] = self.llm_judge_temperature
                if self.llm_judge_max_tokens is not None:
                    invoke_kwargs["max_tokens"] = self.llm_judge_max_tokens

                response = self.llm_judge.invoke(messages, **invoke_kwargs)

                # Parse judge response
                judge_text = response.content
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

                # Store evaluation
                evaluation = {
                    "id": pred["id"],
                    "vlm_response": pred["vlm_response"],
                    "ground_truth": pred["ground_truth"],
                    "judge_score": judge_result["score"],
                    "judge_explanation": judge_result.get("explanation", ""),
                }
                evaluations.append(evaluation)
                scores.append(judge_result["score"])

                self.logger.debug(
                    f"Evaluated {idx}/{len(predictions)}: "
                    f"{pred['id']} - Score: {judge_result['score']}"
                )

            except Exception as e:
                self.logger.error(f"Error evaluating {pred['id']}: {str(e)}")
                console.print(f"[red]Error evaluating {pred['id']}: {str(e)}[/red]")
                continue

        # Calculate metrics
        if scores:
            metrics = self._calculate_metrics(scores, evaluations)

            # Save evaluation results
            eval_file = output_dir / "evaluation_results.json"
            with open(eval_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "evaluations": evaluations,
                        "metrics": metrics,
                    },
                    f,
                    indent=2,
                )

            console.print(
                f"[green]✓ Evaluation complete. Results saved to {eval_file}[/green]"
            )
            self.logger.info(
                f"LLM judge evaluation complete: {len(evaluations)} evaluations"
            )

            # Display metrics
            self._display_metrics(metrics)

            return metrics
        else:
            self.logger.warning("No valid evaluations to calculate metrics")
            console.print("[red]No valid evaluations to calculate metrics[/red]")
            return {}

    def _calculate_metrics(
        self, scores: list[int], evaluations: list[dict]
    ) -> dict[str, Any]:
        """Calculate benchmark metrics.

        Args:
            scores: List of judge scores
            evaluations: List of evaluation results

        Returns:
            Dictionary with calculated metrics
        """
        # Functional Correctness Score (average)
        fcs = statistics.mean(scores)

        # Success Rate (percentage scoring >= 4)
        success_count = sum(1 for s in scores if s >= 4)
        success_rate = (success_count / len(scores)) * 100

        # Score distribution
        score_dist = {i: scores.count(i) for i in range(1, 6)}

        # Failure analysis
        failures = [e for e in evaluations if e["judge_score"] < 4]

        metrics = {
            "functional_correctness_score": round(fcs, 2),
            "success_rate": round(success_rate, 1),
            "total_cases": len(scores),
            "successful_cases": success_count,
            "score_distribution": score_dist,
            "failure_count": len(failures),
        }

        return metrics

    def _display_metrics(self, metrics: dict[str, Any]) -> None:
        """Display metrics in a formatted table."""
        console.print("\n[bold cyan]Benchmark Results[/bold cyan]")
        console.print("=" * 50)

        table = Table(title="Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row(
            "Functional Correctness Score (FCS)",
            f"{metrics['functional_correctness_score']}/5.0",
        )
        table.add_row("Success Rate", f"{metrics['success_rate']}%")
        table.add_row("Total Cases", str(metrics["total_cases"]))
        table.add_row("Successful Cases", str(metrics["successful_cases"]))
        table.add_row("Failed Cases", str(metrics["failure_count"]))

        console.print(table)

        # Score distribution
        console.print("\n[bold]Score Distribution:[/bold]")
        for score, count in sorted(metrics["score_distribution"].items()):
            bar = "█" * count
            console.print(f"  Score {score}: {bar} ({count})")


def create_benchmark(
    benchmark_type: str,
    vlm: Any,
    llm_judge: Any,
    vlm_temperature: float | None = 0.7,
    vlm_max_tokens: int | None = 1024,
    llm_judge_temperature: float | None = 0.7,
    llm_judge_max_tokens: int | None = 1024,
    llm: Any | None = None,
    system_prompt: str | None = None,
    **kwargs,
) -> BaseBenchmark:
    """Factory function to create the appropriate benchmark based on type.

    DEPRECATED: This function is deprecated. Please use
    material_agent.workflows.factory.create_benchmark_workflow_from_config instead,
    which provides a more flexible config-driven workflow approach.

    Args:
        benchmark_type: Type of benchmark ('vlm', 'consistency', 'cmf', etc.)
        vlm: VLM instance for inference
        llm_judge: LLM instance for evaluation
        vlm_temperature: Temperature for VLM inference
        vlm_max_tokens: Maximum tokens for VLM response
        llm_judge_temperature: Temperature for LLM judge
        llm_judge_max_tokens: Maximum tokens for judge response
        llm: Optional LLM for parsing and other tasks
        system_prompt: Optional custom system prompt for VLM
        **kwargs: Additional benchmark-specific parameters

    Returns:
        Benchmark instance of the appropriate type

    Raises:
        ValueError: If benchmark_type is not recognized
    """
    import warnings

    warnings.warn(
        "create_benchmark is deprecated. Use "
        "material_agent.workflows.factory.create_benchmark_workflow_from_config instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    benchmark_type = benchmark_type.lower()

    if benchmark_type == "vlm":
        return VLMBenchmark(
            vlm=vlm,
            llm_judge=llm_judge,
            vlm_temperature=vlm_temperature,
            vlm_max_tokens=vlm_max_tokens,
            llm_judge_temperature=llm_judge_temperature,
            llm_judge_max_tokens=llm_judge_max_tokens,
            llm=llm,
            system_prompt=system_prompt,
        )
    # Future benchmark types can be added here:
    # elif benchmark_type == "consistency":
    #     return ConsistencyBenchmark(...)
    # elif benchmark_type == "cmf":
    #     return CMFBenchmark(...)
    else:
        raise ValueError(
            f"Unknown benchmark type: {benchmark_type}. Supported types: 'vlm'"
        )
