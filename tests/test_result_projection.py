# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for credential-safe published result projections."""

from pathlib import Path

from world_understanding.utils.result_projection import (
    project_result_metadata,
    retain_safe_result_path,
    retain_safe_result_text,
)


def test_result_projection_detaches_redacts_and_strips_runtime_objects() -> None:
    sentinel = "result-projection-credential-713"
    listener = object()

    def cancel_checker() -> bool:
        return False

    class UnsafeRuntimeObject:
        def __init__(self, secret: str) -> None:
            self.secret = secret

        def __repr__(self) -> str:
            raise AssertionError("runtime object repr must not be invoked")

        def __str__(self) -> str:
            raise AssertionError("runtime object str must not be invoked")

    runtime_object = UnsafeRuntimeObject(sentinel)
    raw_result = {
        "config_dict": {"vlm": {"api_key": sentinel}},
        "event_listener": listener,
        "cancel_checker": cancel_checker,
        "path_resolver": object(),
        "pipeline_results": {
            "predict": {
                "api_key": sentinel,
                "num_predictions": 1,
            }
        },
        "nested": {
            "config_dict": {"api_key": sentinel},
            "listener": runtime_object,
            "callback": cancel_checker,
            "providerError": f"opaque provider failure {sentinel}",
            "HTTPError": f"opaque HTTP failure {sentinel}",
            "NVCFExceptions": [f"opaque NVCF failure {sentinel}"],
            "provider.error": f"opaque dotted failure {sentinel}",
            "provider error": f"opaque spaced failure {sentinel}",
            "error_msg": f"opaque error message {sentinel}",
            "errorDetails": f"opaque error details {sentinel}",
            "stack_trace": f"opaque stack trace {sentinel}",
            "failure_message": f"opaque failure message {sentinel}",
            "reason": "visual evidence",
            "failure": False,
            "errors": 0,
            "material_profile_errors": [],
            "failure_detail": {},
            "provider_exceptions": (),
            "provider_failures": set(),
            "provider_tracebacks": frozenset(),
            "diagnostic": {
                "error_code": "provider_failed",
                "detail": f"opaque provider detail {sentinel}",
            },
            "items": [
                {"api_key": sentinel},
                runtime_object,
                cancel_checker,
            ],
        },
    }

    projected = project_result_metadata(raw_result)

    assert raw_result["config_dict"]["vlm"]["api_key"] == sentinel
    assert raw_result["pipeline_results"]["predict"]["api_key"] == sentinel
    assert projected is not raw_result
    assert projected["pipeline_results"] is not raw_result["pipeline_results"]
    assert projected["pipeline_results"]["predict"]["num_predictions"] == 1
    assert "config_dict" not in projected
    assert "event_listener" not in projected
    assert "cancel_checker" not in projected
    assert "path_resolver" not in projected
    assert "config_dict" not in projected["nested"]
    assert "listener" not in projected["nested"]
    assert "callback" not in projected["nested"]
    assert "providerError" not in projected["nested"]
    assert "HTTPError" not in projected["nested"]
    assert "NVCFExceptions" not in projected["nested"]
    assert "provider.error" not in projected["nested"]
    assert "provider error" not in projected["nested"]
    assert "error_msg" not in projected["nested"]
    assert "errorDetails" not in projected["nested"]
    assert "stack_trace" not in projected["nested"]
    assert "failure_message" not in projected["nested"]
    assert projected["nested"]["reason"] == "visual evidence"
    assert projected["nested"]["failure"] is False
    assert projected["nested"]["errors"] == 0
    assert projected["nested"]["material_profile_errors"] == []
    assert projected["nested"]["failure_detail"] == {}
    assert projected["nested"]["provider_exceptions"] == ()
    assert projected["nested"]["provider_failures"] == set()
    assert projected["nested"]["provider_tracebacks"] == frozenset()
    assert projected["nested"]["diagnostic"] == {"error_code": "provider_failed"}
    assert projected["nested"]["items"] == [{"api_key": "<redacted>"}]
    assert sentinel not in repr(projected)


def test_safe_result_scalars_preserve_only_unchanged_values(tmp_path: Path) -> None:
    safe_path = tmp_path / "output.jsonl"
    sentinel = "result-path-credential-713"
    unsafe_path = tmp_path / f"api_key={sentinel}" / "output.jsonl"

    assert retain_safe_result_path(safe_path) == safe_path
    assert retain_safe_result_path(unsafe_path) is None
    assert retain_safe_result_text("session-1", path_context=True) == "session-1"
    assert (
        retain_safe_result_text(
            f"https://user:{sentinel}@session.example.test/id",
            path_context=True,
        )
        is None
    )
