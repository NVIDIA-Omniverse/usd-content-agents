# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Secret-safe representation contracts for public Material API results."""

from __future__ import annotations

from dataclasses import fields

import pytest
from world_understanding.utils.safe_repr import SecretSafeReprMixin

from material_agent.api.apply import ApplyOutput
from material_agent.api.benchmark import BenchmarkOutput
from material_agent.api.build_dataset import (
    BuildDatasetPdfVectorstoreOutput,
    BuildDatasetPrepareDatasetOutput,
    BuildDatasetUsdOutput,
)
from material_agent.api.configure import ConfigureOutput
from material_agent.api.evaluate import EvaluateOutput
from material_agent.api.pipeline import PipelineOutput
from material_agent.api.predict import PredictOutput
from material_agent.api.refine import RefineOutput
from material_agent.api.scene_pipeline import ScenePipelineOutput

RAW_RESULT_OUTPUT_TYPES = (
    ApplyOutput,
    BenchmarkOutput,
    BuildDatasetPdfVectorstoreOutput,
    BuildDatasetPrepareDatasetOutput,
    BuildDatasetUsdOutput,
    ConfigureOutput,
    EvaluateOutput,
    PipelineOutput,
    PredictOutput,
    RefineOutput,
    ScenePipelineOutput,
)


class _Opaque:
    def __repr__(self) -> str:
        raise AssertionError("result representation invoked opaque repr")

    def __str__(self) -> str:
        raise AssertionError("result representation invoked opaque str")


class _UnsafeReprResult(SecretSafeReprMixin):
    def __repr__(self) -> str:
        return "material-output-secret-759"


def test_safe_str_does_not_dispatch_to_an_overridden_repr() -> None:
    output = _UnsafeReprResult()

    assert repr(output) == "material-output-secret-759"
    assert str(output) == "_UnsafeReprResult(<redacted>)"


def test_pipeline_output_repr_redacts_every_mutable_public_field() -> None:
    sentinel = "material-output-secret-759"
    raw_result = {"config_dict": {"api_key": sentinel}}
    output = PipelineOutput(
        success=False,
        error=f"https://user:{sentinel}@example.test/error",
        step_results={"predict": {"api_key": sentinel}},
        completed_steps=[sentinel],
        skipped_steps=[sentinel],
        raw_result=raw_result,
    )

    assert repr(output) == "PipelineOutput(<redacted>)"
    assert str(output) == "PipelineOutput(<redacted>)"
    assert sentinel not in repr(output)
    assert output.raw_result is raw_result

    opaque = _Opaque()
    output.error = opaque  # type: ignore[assignment]
    output.step_results = {"opaque": opaque}  # type: ignore[dict-item]
    output.raw_result = {"opaque": opaque}
    assert repr(output) == "PipelineOutput(<redacted>)"
    assert str(output) == "PipelineOutput(<redacted>)"


@pytest.mark.parametrize("output_type", RAW_RESULT_OUTPUT_TYPES)
def test_every_material_raw_result_output_uses_secret_safe_repr(output_type) -> None:
    assert "raw_result" in {field.name for field in fields(output_type)}
    assert issubclass(output_type, SecretSafeReprMixin)

    output = output_type(success=True)
    raw_result = {"api_key": "material-audit-secret-759"}
    output.raw_result = raw_result
    assert repr(output) == f"{output_type.__name__}(<redacted>)"
    assert str(output) == f"{output_type.__name__}(<redacted>)"
    assert output.raw_result is raw_result
