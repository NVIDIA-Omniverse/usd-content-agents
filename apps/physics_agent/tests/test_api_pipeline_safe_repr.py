# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Secret-safe representation contracts for Physics PipelineOutput."""

from __future__ import annotations

from pathlib import Path

from physics_agent.api.pipeline import PipelineOutput


class _Opaque:
    def __repr__(self) -> str:
        raise AssertionError("result representation invoked opaque repr")

    def __str__(self) -> str:
        raise AssertionError("result representation invoked opaque str")


def test_pipeline_output_repr_is_safe_after_construction_and_mutation() -> None:
    sentinel = "physics-output-secret-759"
    raw_result = {"model": {"api_key": sentinel}}
    output = PipelineOutput(
        success=False,
        error=f"https://user:{sentinel}@example.test/error",
        step_results={"predict": {"api_key": sentinel}},
        completed_steps=[sentinel],
        skipped_steps=[sentinel],
        session_id=sentinel,
        working_dir=Path(f"/tmp/{sentinel}"),
        raw_result=raw_result,
    )

    assert repr(output) == "PipelineOutput(<redacted>)"
    assert str(output) == "PipelineOutput(<redacted>)"
    assert sentinel not in repr(output)
    assert output.raw_result is raw_result

    opaque = _Opaque()
    output.error = opaque  # type: ignore[assignment]
    output.step_results = {"opaque": opaque}  # type: ignore[dict-item]
    output.session_id = opaque  # type: ignore[assignment]
    output.working_dir = opaque  # type: ignore[assignment]
    output.raw_result = {"opaque": opaque}
    assert repr(output) == "PipelineOutput(<redacted>)"
    assert str(output) == "PipelineOutput(<redacted>)"
