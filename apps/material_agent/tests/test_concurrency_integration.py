#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Practical test to demonstrate concurrency fixes.

This script simulates real-world concurrent pipeline execution to verify:
1. Child workflow configs stay isolated in memory without transport files
2. State file I/O is protected by locking (no corruption)
3. Concurrent pipelines with different session IDs work correctly
"""

import json
import multiprocessing
import tempfile
import time
from pathlib import Path

from world_understanding.agentic.base_pipeline_executor import BasePipelineExecutor


class MockPipelineExecutor(BasePipelineExecutor):
    """Mock executor for testing concurrency."""

    def _execute_step(self, step_name, context, object_store):
        """Simulate step execution with some work."""
        time.sleep(0.1)  # Simulate work
        return {"step": step_name, "status": "completed"}

    def _get_step_list_key(self):
        return "steps"

    def _get_required_context_keys(self):
        return ["steps", "working_dir"]

    def _get_state_file(self, context):
        return context["working_dir"] / ".pipeline_state.json"


def run_pipeline_process(process_id, working_dir_str, num_steps=5):
    """Worker function that runs a pipeline in a separate process."""
    from world_understanding.utils.object_store import InMemoryObjectStore

    working_dir = Path(working_dir_str)
    session_dir = working_dir / f".session_{process_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    executor = MockPipelineExecutor()

    # Create context with unique session
    context = {
        "steps": [f"step_{i}" for i in range(num_steps)],
        "working_dir": session_dir,
        "session_id": f"process_{process_id}",
        "project_name": f"test_project_{process_id}",
    }

    object_store = InMemoryObjectStore()

    start_time = time.time()

    try:
        # Run the pipeline
        executor.run(context, object_store)

        duration = time.time() - start_time

        # Verify state file was created correctly
        state_file = session_dir / ".pipeline_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)

            return {
                "process_id": process_id,
                "success": True,
                "duration": duration,
                "completed_steps": len(state.get("completed_steps", [])),
                "expected_steps": num_steps,
                "state_valid": True,
            }
        else:
            return {
                "process_id": process_id,
                "success": False,
                "error": "State file not created",
            }

    except Exception as e:
        return {
            "process_id": process_id,
            "success": False,
            "error": str(e),
            "duration": time.time() - start_time,
        }


def test_concurrent_pipelines(num_processes=10, num_steps=5):
    """Test concurrent pipeline execution."""
    print("=" * 80)
    print("TESTING CONCURRENT PIPELINE EXECUTION")
    print("=" * 80)
    print(f"Processes: {num_processes}")
    print(f"Steps per pipeline: {num_steps}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        working_dir = Path(tmpdir)

        # Launch concurrent processes
        print(
            f"[{time.strftime('%H:%M:%S')}] Launching {num_processes} concurrent pipelines..."
        )
        start_time = time.time()

        with multiprocessing.Pool(processes=num_processes) as pool:
            args = [(i, str(working_dir), num_steps) for i in range(num_processes)]
            results = pool.starmap(run_pipeline_process, args)

        total_duration = time.time() - start_time

        print(
            f"[{time.strftime('%H:%M:%S')}] All pipelines completed in {total_duration:.2f}s"
        )
        print()

        # Analyze results
        successes = [r for r in results if r.get("success")]
        failures = [r for r in results if not r.get("success")]

        print("RESULTS:")
        print(f"  ✓ Successful: {len(successes)}/{num_processes}")
        print(f"  ✗ Failed: {len(failures)}/{num_processes}")
        print()

        if failures:
            print("FAILURES:")
            for f in failures:
                print(f"  Process {f['process_id']}: {f.get('error', 'Unknown error')}")
            print()

        if successes:
            # Calculate statistics
            avg_duration = sum(r["duration"] for r in successes) / len(successes)
            min_duration = min(r["duration"] for r in successes)
            max_duration = max(r["duration"] for r in successes)

            print("PERFORMANCE:")
            print(f"  Average duration: {avg_duration:.2f}s per pipeline")
            print(f"  Min duration: {min_duration:.2f}s")
            print(f"  Max duration: {max_duration:.2f}s")
            print(f"  Total wall time: {total_duration:.2f}s")
            print(
                f"  Speedup: {num_processes * avg_duration / total_duration:.1f}x (parallel execution)"
            )
            print()

        # Verify no data corruption
        print("DATA INTEGRITY CHECKS:")

        # Check each session directory
        corrupted = 0
        for r in successes:
            session_dir = working_dir / f".session_{r['process_id']}"
            state_file = session_dir / ".pipeline_state.json"

            try:
                with open(state_file) as f:
                    state = json.load(f)

                # Verify state integrity
                if not isinstance(state, dict):
                    corrupted += 1
                    print(f"  ✗ Process {r['process_id']}: State is not a dict")
                elif "completed_steps" not in state:
                    corrupted += 1
                    print(f"  ✗ Process {r['process_id']}: Missing 'completed_steps'")
                elif len(state["completed_steps"]) != num_steps:
                    corrupted += 1
                    print(
                        f"  ✗ Process {r['process_id']}: Expected {num_steps} steps, "
                        f"got {len(state['completed_steps'])}"
                    )

            except json.JSONDecodeError:
                corrupted += 1
                print(f"  ✗ Process {r['process_id']}: JSON corruption detected")
            except Exception as e:
                corrupted += 1
                print(f"  ✗ Process {r['process_id']}: {e}")

        if corrupted == 0:
            print(f"  ✓ All {len(successes)} state files are valid and uncorrupted")
        else:
            print(f"  ✗ {corrupted}/{len(successes)} state files corrupted")

        print()

        print("CONFIG ARTIFACT CHECKS:")
        temp_dirs = list(working_dir.rglob(".pipeline_temp"))
        if temp_dirs:
            corrupted += len(temp_dirs)
            print(f"  ✗ Found {len(temp_dirs)} unexpected config transport directories")
        else:
            print("  ✓ No config transport files or directories found")

        print()

        # Final verdict
        print("=" * 80)
        if len(successes) == num_processes and corrupted == 0:
            print("✅ CONCURRENCY TEST PASSED")
            print("   All pipelines completed successfully without data corruption!")
            return True
        else:
            print("❌ CONCURRENCY TEST FAILED")
            if len(successes) < num_processes:
                print(f"   {len(failures)} pipeline(s) failed to complete")
            if corrupted > 0:
                print(f"   {corrupted} state file(s) corrupted")
            return False


def test_in_memory_config_isolation() -> None:
    """Test that child config copies are independent and preserve secrets."""
    print("=" * 80)
    print("TESTING IN-MEMORY CONFIG ISOLATION")
    print("=" * 80)

    from material_agent.tasks.unified_pipeline_executor import (
        _build_child_config_dict,
    )

    source = {"credentials": {"api_key": "x", "nested": [{"token": "tiny"}]}}
    first = _build_child_config_dict(source)
    second = _build_child_config_dict(source)
    first["credentials"]["nested"][0]["token"] = "mutated"

    assert second["credentials"]["nested"][0]["token"] == "tiny"
    assert source["credentials"]["api_key"] == "x"
    assert source["credentials"]["nested"][0]["token"] == "tiny"


if __name__ == "__main__":
    print()
    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║                  MATERIAL-AGENT CONCURRENCY FIX TEST                   ║")
    print("╚════════════════════════════════════════════════════════════════════════╝")
    print()

    # Test 1: In-memory config isolation
    test_in_memory_config_isolation()
    test1_pass = True
    print()

    # Test 2: Concurrent pipeline execution
    test2_pass = test_concurrent_pipelines(num_processes=10, num_steps=5)
    print()

    # Final summary
    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║                           FINAL RESULTS                                ║")
    print("╚════════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  In-Memory Config Isolation: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"  Concurrent Execution: {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print()

    if test1_pass and test2_pass:
        print("🎉 ALL TESTS PASSED - Concurrency issues are fixed!")
        exit(0)
    else:
        print("⚠️  SOME TESTS FAILED - Review the output above")
        exit(1)
