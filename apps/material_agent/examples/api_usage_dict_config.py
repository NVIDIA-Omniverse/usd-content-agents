#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example: Using Material Agent API with in-memory dict configs.

This example demonstrates how to use the Material Agent API with
dynamically generated configuration dictionaries instead of YAML files.
"""

from material_agent.api import (
    BenchmarkInput,
    PipelineInput,
    run_benchmark,
    run_pipeline,
)


def example_benchmark_with_dict_config():
    """Run benchmark using an in-memory config dictionary."""
    print("=" * 70)
    print("Example 1: Benchmark with dict config")
    print("=" * 70)

    # Build config dictionary dynamically
    config = {
        "model": {
            "service": "azure",
            "name": "example-vlm-model",
            "deployment": "example-vlm-model-deployment",
            "api_key_env": "${AZURE_API_KEY}",
        },
        "judge": {
            "service": "azure",
            "name": "example-vlm-model",
            "deployment": "example-vlm-model-deployment",
        },
        "dataset_path": "data/benchmark_dataset.jsonl",
        "output_dir": "output/benchmark_results",
    }

    # Create input with dict config
    params = BenchmarkInput(config=config, verbose=True)

    # Run benchmark
    print("\nRunning benchmark with in-memory config...")
    result = run_benchmark(params)

    if result.success:
        print("\n✓ Benchmark completed successfully!")
        print(f"  FCS: {result.metrics.functional_correctness_score}")
        print(f"  Success Rate: {result.metrics.success_rate}%")
    else:
        print(f"\n✗ Benchmark failed: {result.error}")


def example_multi_model_benchmark():
    """Run benchmarks for multiple models using dict configs."""
    print("\n" + "=" * 70)
    print("Example 2: Multi-model benchmark with dynamic configs")
    print("=" * 70)

    models = ["example-vlm-model", "example-vlm-model-small", "claude-3-5-sonnet"]
    results = {}

    for model_name in models:
        print(f"\n→ Testing {model_name}...")

        # Generate config for this model
        config = {
            "model": {
                "service": "azure",
                "name": model_name,
                "deployment": f"{model_name}-deployment",
            },
            "dataset_path": "data/benchmark_dataset.jsonl",
            "output_dir": f"output/{model_name}",
        }

        # Run benchmark
        params = BenchmarkInput(config=config, verbose=False)
        result = run_benchmark(params)

        if result.success:
            results[model_name] = result.metrics.functional_correctness_score
            print(f"  ✓ FCS: {result.metrics.functional_correctness_score}")
        else:
            print(f"  ✗ Failed: {result.error}")

    # Display comparison
    if results:
        print("\n" + "-" * 70)
        print("Model Comparison:")
        for model, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model:30} {score:.2f}")


def example_pipeline_with_dict_config():
    """Run pipeline with dynamically constructed config."""
    print("\n" + "=" * 70)
    print("Example 3: Pipeline with dict config")
    print("=" * 70)

    # Build unified pipeline config
    # NEW: No need to specify working_dir or output.usd_path!
    # They are auto-derived from session ID
    config = {
        "project": {
            "name": "dynamic_pipeline",
            # working_dir: auto-derived as .{session_id}
        },
        "input": {
            "usd_path": "models/ladder.usd",
        },
        "output": {
            # usd_path: auto-derived as .{session_id}/output/output.usd
            # Only specify output OPTIONS here:
        },
        "steps": {
            "predict": {
                "enabled": True,
                "model": {
                    "service": "azure",
                    "name": "example-vlm-model",
                },
            },
            "apply": {
                "enabled": True,
                "layer_only": False,
                "render": {"enabled": True},
            },
        },
    }

    # Run pipeline with only predict and apply steps
    params = PipelineInput(config=config, only_steps=["predict", "apply"], verbose=True)

    print("\nRunning pipeline...")
    result = run_pipeline(params)

    if result.success:
        print("\n✓ Pipeline completed successfully!")
        print(f"  Completed steps: {', '.join(result.completed_steps)}")
    else:
        print(f"\n✗ Pipeline failed: {result.error}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Material Agent API - Dict Config Examples")
    print("=" * 70)

    # Example 1: Single benchmark with dict config
    example_benchmark_with_dict_config()

    # Example 2: Multi-model benchmark
    # example_multi_model_benchmark()  # Uncomment to run

    # Example 3: Pipeline with dict config
    # example_pipeline_with_dict_config()  # Uncomment to run

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
