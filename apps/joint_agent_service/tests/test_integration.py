# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for Joint Agent Service."""

import os
import time
from pathlib import Path

import pytest
import requests

# Configuration
BASE_URL = os.getenv("SERVICE_BASE_URL", "http://localhost:8000")
TEST_USD_FILE = Path(__file__).parent / "test_data" / "simple_cube.usda"
MAX_WAIT_SECONDS = 300  # 5 minutes max
POLL_INTERVAL = 5  # Poll every 5 seconds

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SERVICE_INTEGRATION") != "1",
    reason="Set RUN_SERVICE_INTEGRATION=1 to run live service integration tests",
)


def _run_pipeline_workflow() -> bool:
    """Run the end-to-end workflow against a live service instance."""

    print("\n" + "=" * 80)
    print("Joint Agent Service - Integration Test")
    print("=" * 80)

    # 1. Check service health
    print("\n[1/7] Checking service health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        health_data = response.json()
        print(f"  Service healthy: {health_data['service']} v{health_data['version']}")
    except Exception as e:
        print(f"  Service health check failed: {e}")
        print("  Make sure service is running: uvicorn service.main:app --port 8000")
        return False

    # 2. Prepare test data
    print("\n[2/7] Preparing test data...")
    if not TEST_USD_FILE.exists():
        print(f"  USD file not found: {TEST_USD_FILE}")
        return False

    print(f"  USD file: {TEST_USD_FILE}")

    # 3. Create pipeline
    print("\n[3/7] Creating pipeline (uploading USD file)...")
    try:
        with open(TEST_USD_FILE, "rb") as usd_file:
            files = [
                (
                    "usd_file",
                    (
                        "scene.usda",
                        usd_file,
                        "application/octet-stream",
                    ),
                )
            ]

            response = requests.post(f"{BASE_URL}/pipeline", files=files)
            response.raise_for_status()

        result = response.json()
        session_id = result["session_id"]
        print(f"  Pipeline created: session_id = {session_id}")
        print(f"  Status: {result['status']}")
        print(f"  Message: {result['message']}")

    except Exception as e:
        print(f"  Failed to create pipeline: {e}")
        if hasattr(e, "response") and e.response:
            print(f"  Response: {e.response.text}")
        return False

    # 4. Monitor progress
    print(f"\n[4/7] Monitoring pipeline progress (session: {session_id})...")
    print("  (This may take several minutes...)\n")

    start_time = time.time()
    last_step = None
    last_percent = -1

    while True:
        try:
            elapsed = int(time.time() - start_time)

            if elapsed > MAX_WAIT_SECONDS:
                print(f"  Timeout after {elapsed}s")
                return False

            response = requests.get(f"{BASE_URL}/pipeline/{session_id}/status")
            response.raise_for_status()
            status = response.json()

            pipeline_status = status["status"]
            overall_progress = status.get("overall_progress", {})
            current_step = status.get("current_step")
            previews = status.get("preview_images", [])

            percent = overall_progress.get("percent", 0)
            if percent != last_percent or (
                current_step and current_step.get("name") != last_step
            ):
                print(f"  [{elapsed:4d}s] Overall: {percent}% | ", end="")

                if current_step:
                    step_name = current_step.get("display_name", "Unknown")
                    step_progress = current_step.get("progress", {})
                    step_percent = step_progress.get("percent", 0)
                    step_msg = step_progress.get("message", "")

                    print(f"{step_name}: {step_percent}% - {step_msg}")
                    last_step = current_step.get("name")
                else:
                    print(f"Status: {pipeline_status}")

                last_percent = percent

            if previews and len(previews) > 0:
                print(f"       Previews available: {len(previews)}")

            if pipeline_status == "completed":
                print(f"\n  Pipeline completed in {elapsed}s")
                break
            elif pipeline_status == "failed":
                print("\n  Pipeline failed")
                print(f"  Error: {status.get('error_message', 'Unknown error')}")
                return False
            elif pipeline_status == "cancelled":
                print("\n  Pipeline was cancelled")
                return False

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"  Error checking status: {e}")
            time.sleep(POLL_INTERVAL)
            continue

    # 5. Get results
    print("\n[5/7] Getting final results...")
    try:
        response = requests.get(f"{BASE_URL}/pipeline/{session_id}/results")
        response.raise_for_status()
        results = response.json()

        stats = results.get("stats", {})
        print("  Results retrieved:")
        print(f"    - Prims processed: {stats.get('prims_processed', 0)}")
        print(f"    - Images generated: {stats.get('images_generated', 0)}")
        print(f"    - Predictions made: {stats.get('predictions_made', 0)}")
        print(f"    - Duration: {results.get('duration_seconds', 0)}s")

    except Exception as e:
        print(f"  Failed to get results: {e}")
        return False

    # 6. Check artifacts
    print("\n[6/7] Verifying artifacts...")
    try:
        # Check predictions
        resp = requests.get(f"{BASE_URL}/artifacts/{session_id}/predictions")
        if resp.status_code == 200:
            print(f"  Predictions available: {len(resp.content)} bytes")
        else:
            print("  Predictions not available yet")

        # Check dataset
        resp = requests.get(f"{BASE_URL}/artifacts/{session_id}/dataset")
        if resp.status_code == 200:
            print(f"  Dataset available: {len(resp.content)} bytes")
        else:
            print("  Dataset not available yet")

    except Exception as e:
        print(f"  Warning checking artifacts: {e}")

    # 7. Cleanup
    print("\n[7/7] Cleaning up session...")
    try:
        response = requests.delete(f"{BASE_URL}/sessions/{session_id}")
        if response.status_code == 204:
            print(f"  Session deleted: {session_id}")
        else:
            print(f"  Session delete returned: {response.status_code}")

    except Exception as e:
        print(f"  Cleanup warning: {e}")

    # Success!
    print("\n" + "=" * 80)
    print("INTEGRATION TEST PASSED")
    print("=" * 80)

    return True


def test_pipeline_workflow() -> None:
    """Test complete pipeline workflow from upload to download."""
    assert _run_pipeline_workflow()


if __name__ == "__main__":
    success = _run_pipeline_workflow()
    exit(0 if success else 1)
