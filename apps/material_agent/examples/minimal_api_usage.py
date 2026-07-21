#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example: Minimal Material Agent API Usage.

This example demonstrates the simplest way to use Material Agent APIs
using convenience functions with minimal parameters.

Replace every ``path/to/...`` placeholder with a configuration you authored
for that individual API. The pipeline example uses the shipped public config.
"""

from pathlib import Path

from material_agent.api import (
    benchmark,
    pipeline,
)


def example_minimal_benchmark():
    """Run benchmark with just one line."""
    print("=" * 70)
    print("Example 1: Minimal Benchmark")
    print("=" * 70)

    # Just pass the config - that's it!
    result = benchmark(Path("path/to/benchmark_config.yaml"))

    if result.success:
        print(f"✓ FCS: {result.metrics.functional_correctness_score}")
        print(f"✓ Success Rate: {result.metrics.success_rate}%")
    else:
        print(f"✗ Error: {result.error}")


def example_minimal_pipeline():
    """Run complete pipeline with one line."""
    print("\n" + "=" * 70)
    print("Example 2: Minimal Pipeline")
    print("=" * 70)

    # Run complete pipeline with defaults
    # Note: No need to specify output paths - auto-derived from session ID!
    result = pipeline(Path("apps/material_agent/configs/unified_example.yaml"))

    if result.success:
        print(f"✓ Completed {len(result.completed_steps)} steps")
        for step in result.completed_steps:
            print(f"  - {step}")

        # Show session ID and output location
        session_id = result.raw_result.get("session_id", "unknown")
        print(f"\n🔑 Session ID: {session_id}")
        print(f"📁 Find outputs in: .{session_id}/output/")
    else:
        print(f"✗ Error: {result.error}")


def example_minimal_with_overrides():
    """Run with minimal code + optional overrides."""
    print("\n" + "=" * 70)
    print("Example 3: Minimal with Optional Overrides")
    print("=" * 70)

    # Pass config + any optional kwargs you need
    result = benchmark(
        Path("path/to/benchmark_config.yaml"),
        verbose=True,  # Optional
        resume=True,  # Optional
    )

    if result.success:
        print("✓ Benchmark completed")
    else:
        print(f"✗ Error: {result.error}")


def example_minimal_dict_config():
    """Run with in-memory dict config."""
    print("\n" + "=" * 70)
    print("Example 4: Minimal with Dict Config")
    print("=" * 70)

    # Build config programmatically
    config = {
        "model": {"service": "azure", "name": "example-vlm-model"},
        "dataset_path": "data/benchmark_dataset.jsonl",
        "output_dir": "output/",
    }

    # Run with dict config
    result = benchmark(config)

    if result.success:
        print("✓ Benchmark with dict config completed")
    else:
        print(f"✗ Error: {result.error}")


def example_all_apis_minimal():
    """Show minimal usage for all APIs."""
    print("\n" + "=" * 70)
    print("Example 5: All APIs Minimal Usage")
    print("=" * 70)

    # All APIs can be called with just config parameter:

    # Benchmark
    # result = benchmark(Path("config.yaml"))

    # Predict
    # result = predict(Path("config.yaml"))

    # Evaluate
    # result = evaluate(Path("config.yaml"))

    # Apply
    # result = apply(Path("config.yaml"))

    # Pipeline
    # result = pipeline(Path("config.yaml"))

    # Assign (iterative)
    # result = assign(Path("config.yaml"))

    # Configure (needs output path)
    # result = configure(Path("new_config.yaml"))

    print("✓ All APIs support minimal single-parameter usage!")


def example_comparison():
    """Compare old vs new style."""
    print("\n" + "=" * 70)
    print("Example 6: Old vs New Style Comparison")
    print("=" * 70)

    # OLD STYLE (still works!)
    from material_agent.api import BenchmarkInput, run_benchmark

    params = BenchmarkInput(
        config=Path("config.yaml"),
        dataset_override=None,
        output_dir_override=None,
        resume=False,
        stream_predictions=True,
        verbose=False,
    )
    result_old = run_benchmark(params)

    # NEW STYLE (much simpler!)
    result_new = benchmark(Path("config.yaml"))

    # Both produce the same result type
    assert type(result_old) is type(result_new)
    print("✓ Old and new styles produce identical results")
    print("  Old: 7 lines of code")
    print("  New: 1 line of code")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Material Agent API - Minimal Usage Examples")
    print("=" * 70)

    # Run examples (most commented out to avoid actual execution)
    # example_minimal_benchmark()
    # example_minimal_pipeline()
    # example_minimal_with_overrides()
    # example_minimal_dict_config()
    example_all_apis_minimal()
    # example_comparison()

    print("\n" + "=" * 70)
    print("Key Takeaway: All APIs need just ONE required parameter!")
    print("=" * 70)
    print()
    print("  result = benchmark(Path('config.yaml'))")
    print("  result = predict(Path('config.yaml'))")
    print("  result = pipeline(Path('config.yaml'))")
    print()
    print("Everything else has sensible defaults!")
    print("=" * 70)
