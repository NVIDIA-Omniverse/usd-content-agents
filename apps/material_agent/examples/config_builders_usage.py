#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example: Using Config Builders for Material Agent API.

This example demonstrates how to use config builders to create
configuration dictionaries programmatically without needing to
remember all required fields.
"""

from material_agent.api import (
    build_benchmark_config,
    build_predict_config,
    build_unified_pipeline_config,
    get_required_fields,
)


def example_predict_with_builder():
    """Build predict config programmatically."""
    print("=" * 70)
    print("Example 1: Build Predict Config")
    print("=" * 70)

    # Build config with required fields only
    config = build_predict_config(
        vlm_backend="nim",
        vlm_model="example-vlm-model",
        dataset_path="data/dataset.jsonl",
    )

    print("\nGenerated config:")
    print(config)

    # Use with predict API
    # result = predict(config)
    print("\n✓ Config ready to use: predict(config)")


def example_predict_with_all_options():
    """Build predict config with all options."""
    print("\n" + "=" * 70)
    print("Example 2: Build Predict Config with Options")
    print("=" * 70)

    config = build_predict_config(
        vlm_backend="nim",
        vlm_model="example-vlm-model",
        dataset_path="data/dataset.jsonl",
        # Optional parameters
        llm_backend="nim",
        llm_model="example-vlm-model",
        output_dir="output/predictions",
        temperature=0.7,
        max_tokens=1024,
        max_workers=8,
    )

    print("\nGenerated config with options:")
    print(config)


def example_benchmark_with_builder():
    """Build benchmark config."""
    print("\n" + "=" * 70)
    print("Example 3: Build Benchmark Config")
    print("=" * 70)

    # Minimal benchmark config
    config = build_benchmark_config(
        vlm_backend="nim",
        vlm_model="example-vlm-model",
        dataset_path="data/benchmark.jsonl",
    )

    print("\nGenerated config:")
    print(config)
    print("\n✓ All required models use same VLM by default")


def example_pipeline_with_builder():
    """Build unified pipeline config."""
    print("\n" + "=" * 70)
    print("Example 4: Build Unified Pipeline Config")
    print("=" * 70)

    # Define materials
    materials = [
        {"name": "Steel", "prim_path": "/Materials/Steel"},
        {"name": "Plastic", "prim_path": "/Materials/Plastic"},
    ]

    # NEW: No need to specify output_usd_path!
    # It will be auto-derived as .{session_id}/output/output.usd
    config = build_unified_pipeline_config(
        project_name="my_project",
        input_usd_path="models/input.usd",
        materials_library_path="materials/library.usd",
        materials_entries=materials,
    )

    print("\nGenerated config:")
    import json

    print(json.dumps(config, indent=2))
    print("\n✓ Output paths auto-managed via session ID!")
    print("  - working_dir: .{session_id}")
    print("  - output_usd: .{session_id}/output/output.usd")


def example_get_required_fields():
    """Check what fields are required for each API."""
    print("\n" + "=" * 70)
    print("Example 5: Check Required Fields")
    print("=" * 70)

    apis = ["predict", "benchmark", "evaluate", "apply", "pipeline", "refine"]

    for api in apis:
        fields = get_required_fields(api)
        print(f"\n{api.upper()}:")
        print(f"  Required: {', '.join(fields['required'][:3])}...")
        print(f"  Optional: {len(fields['optional'])} fields")


def example_comparison():
    """Compare manual vs builder approach."""
    print("\n" + "=" * 70)
    print("Example 6: Manual vs Builder Comparison")
    print("=" * 70)

    # MANUAL (error-prone, need to remember structure)
    manual_config = {
        "vlm": {
            "backend": "nim",
            "model": "example-vlm-model",
        },
        "dataset": "data/dataset.jsonl",
    }

    # BUILDER (guided, can't forget required fields)
    builder_config = build_predict_config(
        vlm_backend="nim",
        vlm_model="example-vlm-model",
        dataset_path="data/dataset.jsonl",
    )

    print("\nManual config:")
    print(manual_config)
    print("\nBuilder config:")
    print(builder_config)
    print("\n✓ Builder ensures all required fields are present!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Material Agent API - Config Builders")
    print("=" * 70)

    example_predict_with_builder()
    example_predict_with_all_options()
    example_benchmark_with_builder()
    # example_pipeline_with_builder()  # Commented - produces long output
    example_get_required_fields()
    example_comparison()

    print("\n" + "=" * 70)
    print("Key Takeaway:")
    print("=" * 70)
    print()
    print("Config builders help you create valid configurations")
    print("without needing to remember all required fields!")
    print()
    print("  config = build_predict_config(")
    print("      vlm_backend='nim',")
    print("      vlm_model='example-vlm-model',")
    print("      dataset_path='data.jsonl',")
    print("  )")
    print("  result = predict(config)")
    print()
    print("=" * 70)
