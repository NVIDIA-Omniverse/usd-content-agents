# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SimReady Benchmark runtime-validation adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, field_validator
from world_understanding.validation import (
    TemplateStatus,
    ValidationEvidence,
    ValidationIssue,
    ValidationTemplateResult,
)

from content_agent_workflows.common.artifacts import atomic_write_json, file_sha256
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.validation.verified_operations import (
    VerifiedOperationError,
    execution_artifact_binding,
)

from .asset_identity import (
    AssetDependencyIdentityError,
    asset_dependency_identity_errors,
    asset_dependency_root_sha256,
    build_asset_dependency_manifest,
)

SIMREADY_BENCHMARK_VERSION = "2026.6.5"
SIMREADY_BENCHMARK_ENGINE_KIT_VERSION = "2026.6.5"
SIMREADY_BENCHMARK_NATIVE_REPORT_SCHEMA_VERSION = "5.0"
SIMREADY_RUNTIME_VALIDATION_SCHEMA_VERSION = (
    "content-agent-workflows.simready-runtime-validation.v1"
)
SIMREADY_RUNTIME_VERIFIED_OPERATIONS_DIRNAME = "simready-runtime-verified-operations"
SIMREADY_RUNTIME_PUBLICATION_INVALIDATED_NAME = ".publication-invalidated.json"
SIMREADY_RUNTIME_PUBLICATION_VALIDITY_NAME = ".publication-valid.json"
SIMREADY_BENCHMARK_EXECUTABLE_ENV = "CONTENT_WORKFLOW_SIMREADY_BENCHMARK_EXECUTABLE"
SIMREADY_BENCHMARK_VENV_ENV = "CONTENT_WORKFLOW_SIMREADY_BENCHMARK_VENV"

_VERSION_PATTERN = re.compile(r"SimReady Benchmark \(v([^\s)]+)\)")
_NATIVE_TEST_STATUSES = frozenset({"pass", "fail", "skipped", "incomplete", "blocked"})
_NATIVE_PLAN_ONLY_FEATURE_STATUSES = frozenset({"neutral", "validation_failed"})
_RUNTIME_ERROR_EVENTS = frozenset(
    {"engine_error", "engine_crash", "engine_stuck", "run_error"}
)
_STATIC_VALIDATION_BLOCK_ERROR = (
    "Static validation prevented one or more runtime features from running."
)

RuntimeDisposition = Literal[
    "pass",
    "warn",
    "fail",
    "not_evaluated",
    "blocked",
    "error",
]


class SimReadyBenchmarkRuntimeInfo(BaseModel):
    """Resolved external benchmark executable and package identity."""

    model_config = ConfigDict(extra="forbid")

    executable: str | None = None
    executable_sha256: str | None = None
    benchmark_version: str | None = None
    engine_kit_version: str | None = None
    expected_benchmark_version: str = SIMREADY_BENCHMARK_VERSION
    expected_engine_kit_version: str = SIMREADY_BENCHMARK_ENGINE_KIT_VERSION
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.executable is not None and not self.errors


class SimReadyRuntimeValidationInput(BaseModel):
    """Input for one local-asset SimReady Benchmark invocation."""

    model_config = ConfigDict(extra="forbid")

    asset_path: str
    output_dir: str
    sr_specs_path: str
    engines_toml_path: str
    project_config_path: str | None = None
    tests_paths: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    runtimes: tuple[str, ...] = ()
    benchmark_executable: str | None = None
    report_path: str | None = None
    max_concurrent: int = Field(default=1, ge=1)
    timeout_s: float = Field(default=3600.0, gt=0.0)
    max_stdout_log_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    max_stderr_log_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    max_json_bytes: int = Field(default=64 * 1024 * 1024, gt=0)
    stamp_results: bool = False

    @field_validator("tests_paths", "features", "tests", "runtimes", mode="before")
    @classmethod
    def _normalize_string_tuple(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str | Path):
            return (str(value),)
        if isinstance(value, Sequence):
            return tuple(dict.fromkeys(str(item) for item in value))
        raise ValueError("Expected a string or sequence of strings")

    @field_validator("tests_paths")
    @classmethod
    def _validate_tests_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("tests_paths must not contain empty paths")
        return normalized

    @field_validator("features", "tests", "runtimes")
    @classmethod
    def _validate_filter_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item or item.startswith("-") for item in normalized):
            raise ValueError(
                "benchmark filter values must be non-empty and must not begin with '-'"
            )
        return tuple(dict.fromkeys(normalized))


class SimReadyRuntimeCheckResult(BaseModel):
    """One native benchmark test preserved with a WU disposition."""

    model_config = ConfigDict(extra="forbid")

    asset_key: str
    profile_id: str
    profile_version: str
    feature_id: str
    feature_version: str
    test_name: str
    test_version: str
    native_status: str
    disposition: RuntimeDisposition
    duration_s: float | None = None
    engine: str | None = None
    engine_version: str | None = None
    message: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    media: list[dict[str, Any]] = Field(default_factory=list)
    kit_logs: list[Any] = Field(default_factory=list)


class SimReadyRuntimeInputBinding(BaseModel):
    """Exact file roster and digest identity for one Benchmark runtime input."""

    model_config = ConfigDict(extra="forbid")

    label: str
    kind: Literal["file", "directory"]
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    directories: tuple[str, ...] = ()
    files: tuple[ExecutionArtifactBinding, ...] = ()


class SimReadyRuntimeValidationReport(BaseModel):
    """Durable WU envelope around an unmodified native benchmark report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SIMREADY_RUNTIME_VALIDATION_SCHEMA_VERSION
    passed: bool
    status: RuntimeDisposition
    asset_path: str
    asset_sha256: str | None = None
    asset_dependency_manifest: dict[str, Any] = Field(default_factory=dict)
    benchmark_version: str | None = None
    engine_kit_version: str | None = None
    benchmark_executable: str | None = None
    benchmark_executable_sha256: str | None = None
    runtime_input_bindings: list[SimReadyRuntimeInputBinding] = Field(
        default_factory=list
    )
    command: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    checks: list[SimReadyRuntimeCheckResult] = Field(default_factory=list)
    native_report_schema_version: str | None = None
    native_output_dir: str | None = None
    native_plan_path: str | None = None
    native_report_path: str | None = None
    run_summary_path: str | None = None
    events_path: str | None = None
    stdout_log_path: str | None = None
    stderr_log_path: str | None = None
    report_path: str | None = None
    validation_template_result_path: str | None = None
    verified_operation_publication_path: str | None = None
    runtime_error_events: list[dict[str, Any]] = Field(default_factory=list)
    artifact_sha256: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    next_step: str = "complete"


def resolve_simready_benchmark_runtime(
    executable: str | Path | None = None,
) -> SimReadyBenchmarkRuntimeInfo:
    """Resolve and attest the separately installed benchmark executable."""

    warnings: list[str] = []
    errors: list[str] = []
    candidate = _benchmark_executable_candidate(executable)
    if candidate is None:
        return SimReadyBenchmarkRuntimeInfo(
            errors=[
                "simready-benchmark is unavailable. Install "
                f"simready-benchmark[kit]=={SIMREADY_BENCHMARK_VERSION} in a "
                "separate Python 3.12 environment, then pass "
                "--benchmark-executable or set "
                f"{SIMREADY_BENCHMARK_EXECUTABLE_ENV}."
            ]
        )

    try:
        executable_sha256 = file_sha256(candidate)
    except (OSError, ValueError) as exc:
        executable_sha256 = None
        errors.append(f"Could not hash simready-benchmark executable: {exc}")

    version, version_error = _benchmark_version(candidate)
    if version_error:
        errors.append(version_error)
    elif version != SIMREADY_BENCHMARK_VERSION:
        errors.append(
            "Unsupported simready-benchmark version: expected "
            f"{SIMREADY_BENCHMARK_VERSION}, observed {version or 'unknown'}."
        )

    engine_kit_version = _distribution_version_from_sibling_python(
        candidate, "simready-benchmark-engine-kit"
    )
    if engine_kit_version is None:
        warnings.append(
            "simready-benchmark-engine-kit is not visible in the benchmark "
            "environment. Local Kit execution will remain blocked; remote workers "
            "may still be usable."
        )
    elif engine_kit_version != SIMREADY_BENCHMARK_ENGINE_KIT_VERSION:
        errors.append(
            "Unsupported simready-benchmark-engine-kit version: expected "
            f"{SIMREADY_BENCHMARK_ENGINE_KIT_VERSION}, observed "
            f"{engine_kit_version}."
        )

    return SimReadyBenchmarkRuntimeInfo(
        executable=str(candidate),
        executable_sha256=executable_sha256,
        benchmark_version=version,
        engine_kit_version=engine_kit_version,
        warnings=warnings,
        errors=errors,
    )


def build_simready_benchmark_command(
    *,
    params: SimReadyRuntimeValidationInput,
    runtime: SimReadyBenchmarkRuntimeInfo,
    asset_path: Path,
    native_output_dir: Path,
) -> list[str]:
    """Build one explicit, non-interactive benchmark invocation."""

    if runtime.executable is None:
        raise RuntimeError("SimReady Benchmark executable is not resolved.")
    command = [
        runtime.executable,
        "--assets",
        str(asset_path),
        "--sr-specs",
        str(Path(params.sr_specs_path).expanduser().resolve()),
        "--engines-toml",
        str(Path(params.engines_toml_path).expanduser().resolve()),
        "--output",
        str(native_output_dir / "plan.json"),
        "--output-dir",
        str(native_output_dir),
        "--max-concurrent",
        str(params.max_concurrent),
        "--format",
        "json",
    ]
    if params.project_config_path is not None:
        command.extend(
            [
                "--project-config",
                str(Path(params.project_config_path).expanduser().resolve()),
            ]
        )
    if params.tests_paths:
        command.extend(
            [
                "--tests-path",
                *(
                    str(Path(path).expanduser().resolve())
                    for path in params.tests_paths
                ),
            ]
        )
    if params.features:
        command.extend(["--features", *params.features])
    if params.tests:
        command.extend(["--tests", *params.tests])
    if params.runtimes:
        command.extend(["--runtime", *params.runtimes])
    if not params.stamp_results:
        command.append("--no-stamp")
    return command


def run_simready_runtime_validation(
    params: SimReadyRuntimeValidationInput,
) -> SimReadyRuntimeValidationReport:
    """Run SimReady Benchmark and project its native per-test results."""

    asset_path = Path(params.asset_path).expanduser().resolve()
    output_dir = Path(params.output_dir).expanduser().resolve()
    native_output_dir = output_dir / "simready-benchmark"
    verified_operations_dir = output_dir / SIMREADY_RUNTIME_VERIFIED_OPERATIONS_DIRNAME
    template_result_path = output_dir / "validation-template-result.json"
    stdout_log_path = output_dir / "simready-benchmark.stdout.log"
    stderr_log_path = output_dir / "simready-benchmark.stderr.log"
    default_report_path = output_dir / "simready-runtime-validation.json"
    protected_input_paths = _configured_input_paths(params, asset_path=asset_path)
    protected_output_errors = _protected_output_tree_errors(
        output_dir, protected_input_paths
    )
    if protected_output_errors:
        return _base_report(
            asset_path=asset_path,
            status="blocked",
            errors=protected_output_errors,
            next_step="configure-simready-benchmark",
        )
    verified_tree_is_configured_input = any(
        path == verified_operations_dir
        or _is_relative_to(path, verified_operations_dir)
        for path in protected_input_paths.values()
    )
    requested_report_path = (
        Path(params.report_path).expanduser().resolve()
        if params.report_path is not None
        else default_report_path
    )
    report_path = requested_report_path
    report_path_errors: list[str] = []
    report_path_is_directory = report_path.is_dir()
    report_path_non_directory_ancestor = _non_directory_ancestor(report_path)
    if (
        not _is_relative_to(report_path, output_dir)
        or _is_relative_to(report_path, native_output_dir)
        or _is_relative_to(report_path, verified_operations_dir)
        or report_path == asset_path
        or report_path == output_dir
        or report_path_is_directory
        or report_path_non_directory_ancestor is not None
        or report_path in {template_result_path, stdout_log_path, stderr_log_path}
        or _generated_path_conflicts_with_inputs(
            report_path, protected_input_paths.values()
        )
    ):
        report_path_errors.append(
            "Runtime report path must be a file under output_dir, outside the "
            "managed benchmark and verified-operation directories, and distinct "
            "from source/input and generated output paths: "
            f"{requested_report_path}"
        )
        if report_path_is_directory:
            report_path_errors.append(
                f"Generated runtime report path is a directory: {requested_report_path}"
            )
        if report_path_non_directory_ancestor is not None:
            report_path_errors.append(
                "Generated runtime report path has a non-directory ancestor: "
                f"{report_path_non_directory_ancestor}"
            )
        fallback_report_path = _fallback_runtime_report_path(
            output_dir,
            forbidden_paths={
                *protected_input_paths.values(),
                template_result_path,
                stdout_log_path,
                stderr_log_path,
            },
        )
        if fallback_report_path is None:
            return _base_report(
                asset_path=asset_path,
                status="blocked",
                errors=[
                    *report_path_errors,
                    "No safe runtime report path is available under output_dir.",
                ],
                next_step="configure-simready-benchmark",
            )
        report_path = fallback_report_path
    generated_file_paths = {
        "runtime report": report_path,
        "validation template result": template_result_path,
        "benchmark stdout log": stdout_log_path,
        "benchmark stderr log": stderr_log_path,
    }
    input_errors = [
        *report_path_errors,
        *_validate_input_paths(
            params=params,
            asset_path=asset_path,
            native_output_dir=native_output_dir,
            verified_operations_dir=verified_operations_dir,
            generated_file_paths=generated_file_paths,
        ),
    ]

    try:
        asset_dependency_manifest = build_asset_dependency_manifest(asset_path)
        asset_sha256 = asset_dependency_root_sha256(asset_dependency_manifest)
    except AssetDependencyIdentityError as exc:
        # The reserved marker revokes old canonical evidence without overwriting
        # an existing dependency. Compatibility reports are refreshed only when
        # dependency discovery produced a complete path inventory proving that
        # their selected destinations do not collide with the failed asset.
        invalidation_errors: list[str] = []
        if not verified_tree_is_configured_input:
            invalidation_error = _invalidate_previous_publication(
                verified_operations_dir
            )
            if invalidation_error is not None:
                invalidation_errors.append(invalidation_error)
        identity_error = f"Could not bind asset dependency identity: {exc}"
        report = _base_report(
            asset_path=asset_path,
            status="blocked",
            errors=[
                *input_errors,
                *invalidation_errors,
                identity_error,
            ],
            next_step="resolve-usd-dependencies",
        )
        if not exc.path_inventory_complete:
            return report

        dependency_paths = set(exc.dependency_paths)
        dependency_output_errors = _managed_dependency_output_errors(
            dependency_paths,
            native_output_dir=native_output_dir,
            verified_operations_dir=verified_operations_dir,
            generated_file_paths=generated_file_paths,
        )
        if report_path in dependency_paths:
            fallback_report_path = _fallback_runtime_report_path(
                output_dir,
                forbidden_paths={
                    *protected_input_paths.values(),
                    *dependency_paths,
                    template_result_path,
                    stdout_log_path,
                    stderr_log_path,
                },
            )
            if fallback_report_path is None:
                report.errors = _dedupe(
                    [
                        *report.errors,
                        *dependency_output_errors,
                        "No safe runtime report path is available under output_dir.",
                    ]
                )
                return report
            report_path = fallback_report_path
            generated_file_paths["runtime report"] = report_path
        write_template_result = (
            not template_result_path.is_dir()
            and _non_directory_ancestor(template_result_path) is None
            and template_result_path not in dependency_paths
            and not _generated_path_conflicts_with_inputs(
                template_result_path,
                protected_input_paths.values(),
            )
        )
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as output_exc:
            report.errors = _dedupe(
                [
                    *report.errors,
                    *dependency_output_errors,
                    f"Could not prepare runtime output directory {output_dir}: "
                    f"{output_exc}",
                ]
            )
            return report
        report.errors = _dedupe([*report.errors, *dependency_output_errors])
        return _write_runtime_report(
            report,
            report_path=report_path,
            template_result_path=template_result_path,
            write_template_result=write_template_result,
        )

    dependency_paths, dependency_path_errors = _asset_dependency_paths(
        asset_dependency_manifest
    )
    if dependency_path_errors:
        report = _base_report(
            asset_path=asset_path,
            status="blocked",
            errors=[*input_errors, *dependency_path_errors],
            next_step="resolve-usd-dependencies",
        )
        report.asset_sha256 = asset_sha256
        report.asset_dependency_manifest = asset_dependency_manifest
        return report
    dependency_output_errors = [
        *_managed_dependency_output_errors(
            dependency_paths,
            native_output_dir=native_output_dir,
            verified_operations_dir=verified_operations_dir,
            generated_file_paths=generated_file_paths,
        ),
    ]
    if report_path in dependency_paths:
        fallback_report_path = _fallback_runtime_report_path(
            output_dir,
            forbidden_paths={
                *protected_input_paths.values(),
                *dependency_paths,
                template_result_path,
                stdout_log_path,
                stderr_log_path,
            },
        )
        if fallback_report_path is None:
            report = _base_report(
                asset_path=asset_path,
                status="blocked",
                errors=[
                    *dependency_output_errors,
                    "No safe runtime report path is available under output_dir.",
                ],
                next_step="configure-simready-benchmark",
            )
            report.asset_sha256 = asset_sha256
            report.asset_dependency_manifest = asset_dependency_manifest
            return report
        report_path = fallback_report_path
        generated_file_paths["runtime report"] = report_path

    write_template_result = (
        not template_result_path.is_dir()
        and template_result_path not in dependency_paths
        and not _generated_path_conflicts_with_inputs(
            template_result_path, protected_input_paths.values()
        )
    )
    verified_tree_is_protected = verified_tree_is_configured_input or any(
        path == verified_operations_dir
        or _is_relative_to(path, verified_operations_dir)
        for path in dependency_paths
    )
    runtime_input_bindings, runtime_input_errors = _capture_runtime_input_bindings(
        params
    )
    preflight_errors = [
        *input_errors,
        *dependency_output_errors,
        *runtime_input_errors,
    ]

    # Dependency identity and all output collisions are resolved before this first
    # filesystem mutation. This keeps blocked runs from overwriting source inputs.
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        report = _base_report(
            asset_path=asset_path,
            status="blocked",
            errors=[
                *preflight_errors,
                f"Could not prepare runtime output directory {output_dir}: {exc}",
            ],
            next_step="prepare-simready-benchmark-output",
        )
        report.asset_sha256 = asset_sha256
        report.asset_dependency_manifest = asset_dependency_manifest
        report.runtime_input_bindings = runtime_input_bindings
        return report

    if not verified_tree_is_protected:
        publication_invalidation_error = _invalidate_previous_publication(
            verified_operations_dir
        )
        if publication_invalidation_error is not None:
            preflight_errors.append(publication_invalidation_error)

    if preflight_errors:
        report = _base_report(
            asset_path=asset_path,
            status="blocked",
            errors=_dedupe(preflight_errors),
            next_step="configure-simready-benchmark",
        )
        report.asset_sha256 = asset_sha256
        report.asset_dependency_manifest = asset_dependency_manifest
        report.runtime_input_bindings = runtime_input_bindings
        return _write_runtime_report(
            report,
            report_path=report_path,
            template_result_path=template_result_path,
            write_template_result=write_template_result,
        )

    publication_clear_error = _clear_managed_output_dir(verified_operations_dir)
    if publication_clear_error is not None:
        report = _base_report(
            asset_path=asset_path,
            status="blocked",
            errors=[publication_clear_error],
            next_step="prepare-simready-benchmark-output",
        )
        report.asset_sha256 = asset_sha256
        report.asset_dependency_manifest = asset_dependency_manifest
        report.runtime_input_bindings = runtime_input_bindings
        return _write_runtime_report(
            report,
            report_path=report_path,
            template_result_path=template_result_path,
        )

    runtime = resolve_simready_benchmark_runtime(params.benchmark_executable)
    if not runtime.passed:
        report = _base_report(
            asset_path=asset_path,
            status="blocked",
            errors=runtime.errors,
            warnings=runtime.warnings,
            next_step="install-simready-benchmark-runtime",
        )
        report.asset_sha256 = asset_sha256
        report.asset_dependency_manifest = asset_dependency_manifest
        report.benchmark_version = runtime.benchmark_version
        report.engine_kit_version = runtime.engine_kit_version
        report.benchmark_executable = runtime.executable
        report.benchmark_executable_sha256 = runtime.executable_sha256
        report.runtime_input_bindings = runtime_input_bindings
        return _write_runtime_report(
            report,
            report_path=report_path,
            template_result_path=template_result_path,
        )

    preparation_error = _prepare_native_output_dir(native_output_dir)
    if preparation_error:
        report = _base_report(
            asset_path=asset_path,
            status="error",
            errors=[preparation_error],
            warnings=runtime.warnings,
            next_step="prepare-simready-benchmark-output",
        )
        report.asset_sha256 = asset_sha256
        report.asset_dependency_manifest = asset_dependency_manifest
        report.runtime_input_bindings = runtime_input_bindings
        return _write_runtime_report(
            report,
            report_path=report_path,
            template_result_path=template_result_path,
        )

    assert runtime.executable is not None
    command = build_simready_benchmark_command(
        params=params,
        runtime=runtime,
        asset_path=asset_path,
        native_output_dir=native_output_dir,
    )
    exit_code, process_error, log_warnings = _run_bounded_benchmark_process(
        command,
        cwd=output_dir,
        env=_benchmark_subprocess_environment(Path(runtime.executable)),
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        timeout_s=params.timeout_s,
        stdout_limit_bytes=params.max_stdout_log_bytes,
        stderr_limit_bytes=params.max_stderr_log_bytes,
    )

    native_plan_path = native_output_dir / "plan.json"
    native_report_path = native_output_dir / "report" / "test_results_index.json"
    run_summary_path = native_output_dir / "run_summary.json"
    events_path = native_output_dir / "state" / "events.jsonl"
    native_plan, native_plan_error = _load_json_object(
        native_plan_path, max_bytes=params.max_json_bytes
    )
    native_report, native_report_error = _load_json_object(
        native_report_path, max_bytes=params.max_json_bytes
    )
    run_summary, run_summary_error = _load_json_object(
        run_summary_path, max_bytes=params.max_json_bytes
    )
    runtime_error_events, events_error = _load_runtime_error_events(
        events_path, max_bytes=params.max_json_bytes
    )
    identity_errors = asset_dependency_identity_errors(
        asset_path, asset_dependency_manifest
    )
    runtime_input_identity_errors = _runtime_input_binding_errors(
        runtime_input_bindings
    )
    status, checks, normalization_warnings, normalization_errors = (
        _normalize_benchmark_result(
            exit_code=exit_code,
            process_error=process_error,
            expected_asset_path=asset_path,
            native_plan=native_plan,
            native_plan_error=native_plan_error,
            native_report=native_report,
            native_report_error=native_report_error,
            run_summary=run_summary,
            run_summary_error=run_summary_error,
            runtime_error_events=runtime_error_events,
            events_error=events_error,
        )
    )
    all_errors = [
        *identity_errors,
        *runtime_input_identity_errors,
        *normalization_errors,
    ]
    if identity_errors or runtime_input_identity_errors:
        status = "error"
    artifact_sha256, artifact_warnings = _artifact_digests(
        {
            "native_plan": native_plan_path,
            "native_report": native_report_path,
            "run_summary": run_summary_path,
            "events": events_path,
            "stdout": stdout_log_path,
            "stderr": stderr_log_path,
        }
    )

    metadata = native_report.get("metadata") if native_report else None
    native_schema = (
        str(metadata.get("schema_version"))
        if isinstance(metadata, Mapping) and metadata.get("schema_version") is not None
        else None
    )
    report = SimReadyRuntimeValidationReport(
        passed=status == "pass" and not all_errors,
        status=status,
        asset_path=str(asset_path),
        asset_sha256=asset_sha256,
        asset_dependency_manifest=asset_dependency_manifest,
        benchmark_version=runtime.benchmark_version,
        engine_kit_version=runtime.engine_kit_version,
        benchmark_executable=runtime.executable,
        benchmark_executable_sha256=runtime.executable_sha256,
        runtime_input_bindings=runtime_input_bindings,
        command=command,
        exit_code=exit_code,
        checks=checks,
        native_report_schema_version=native_schema,
        native_output_dir=str(native_output_dir),
        native_plan_path=str(native_plan_path) if native_plan_path.exists() else None,
        native_report_path=str(native_report_path)
        if native_report_path.exists()
        else None,
        run_summary_path=str(run_summary_path) if run_summary_path.exists() else None,
        events_path=str(events_path) if events_path.exists() else None,
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        runtime_error_events=runtime_error_events,
        artifact_sha256=artifact_sha256,
        warnings=_dedupe(
            [
                *runtime.warnings,
                *log_warnings,
                *normalization_warnings,
                *artifact_warnings,
            ]
        ),
        errors=_dedupe(all_errors),
        next_step=(
            "simready-conform-profile"
            if _STATIC_VALIDATION_BLOCK_ERROR in normalization_errors
            else _next_step(status)
        ),
    )
    return _write_runtime_report(
        report,
        report_path=report_path,
        template_result_path=template_result_path,
    )


def simready_runtime_validation_template_result(
    report: SimReadyRuntimeValidationReport,
) -> ValidationTemplateResult:
    """Project a native runtime result into ``physical_behavior`` evidence."""

    status_map: dict[RuntimeDisposition, TemplateStatus] = {
        "pass": "passed",
        "warn": "warn",
        "fail": "failed",
        "not_evaluated": "skipped",
        "blocked": "error",
        "error": "error",
    }
    issues: list[ValidationIssue] = []
    for check in report.checks:
        if check.disposition == "pass":
            continue
        severity: Literal["info", "warn", "fail"] = (
            "fail" if check.disposition in {"fail", "error"} else "warn"
        )
        issues.append(
            ValidationIssue(
                code=f"physical_behavior.simready_runtime_{check.disposition}",
                severity=severity,
                message=check.message
                or f"SimReady runtime test {check.test_name} was {check.disposition}.",
                template_name="physical_behavior",
                subject=check.asset_key,
                details=check.model_dump(mode="json"),
            )
        )
    for error in report.errors:
        issues.append(
            ValidationIssue(
                code="physical_behavior.simready_runtime_error",
                severity="fail",
                message=error,
                template_name="physical_behavior",
                subject=report.asset_path,
            )
        )
    for event in report.runtime_error_events:
        event_type = str(event.get("type") or "runtime_error")
        event_detail = str(event.get("detail") or event.get("message") or "")
        issues.append(
            ValidationIssue(
                code="physical_behavior.simready_runtime_error",
                severity="fail",
                message=(
                    f"SimReady Benchmark reported {event_type}: {event_detail}"
                    if event_detail
                    else f"SimReady Benchmark reported {event_type}."
                ),
                template_name="physical_behavior",
                subject=report.asset_path,
                details=event,
            )
        )

    evidence_items = [
        ValidationEvidence(
            kind="simulation_json",
            path=report.native_report_path,
            subject=report.asset_path,
            summary="Native SimReady Benchmark per-test report.",
            metadata={
                "schema_version": report.native_report_schema_version,
                "framework_version": report.benchmark_version,
            },
        ),
        ValidationEvidence(
            kind="simulation_json",
            path=report.report_path,
            subject=report.asset_path,
            summary="World Understanding SimReady runtime-validation envelope.",
        ),
        ValidationEvidence(
            kind="artifact_dir",
            path=report.native_output_dir,
            subject=report.asset_path,
            summary="Native SimReady Benchmark logs, results, and media.",
            metadata={"artifact_sha256": report.artifact_sha256},
        ),
    ]
    if report.verified_operation_publication_path is not None:
        evidence_items.append(
            ValidationEvidence(
                kind="simulation_json",
                path=report.verified_operation_publication_path,
                subject=report.asset_path,
                summary=(
                    "Per-test SimReady runtime evidence for non-executing "
                    "verified-operation ingestion."
                ),
            )
        )
    return ValidationTemplateResult(
        template_name="physical_behavior",
        status=status_map[report.status],
        issues=tuple(issues),
        metrics={
            "simready_runtime_check_count": len(report.checks),
            "simready_runtime_failed_check_count": sum(
                check.disposition == "fail" for check in report.checks
            ),
            "simready_runtime_error_event_count": len(report.runtime_error_events),
        },
        evidence={
            "simready_runtime_checks": [
                check.model_dump(mode="json") for check in report.checks
            ],
            "native_report_path": report.native_report_path,
            "runtime_report_path": report.report_path,
            "verified_operation_publication_path": (
                report.verified_operation_publication_path
            ),
        },
        evidence_items=tuple(evidence_items),
        metadata={
            "simready_runtime_status": report.status,
            "simready_benchmark_version": report.benchmark_version,
            "native_dispositions_preserved": True,
            "verified_operation_projection": "per_native_test",
        },
    )


def _benchmark_executable_candidate(
    executable: str | Path | None,
) -> Path | None:
    value = executable or os.getenv(SIMREADY_BENCHMARK_EXECUTABLE_ENV)
    if value is not None:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute() and candidate.parent == Path("."):
            resolved = shutil.which(str(candidate))
            return Path(resolved).resolve() if resolved else None
        resolved_candidate = candidate.resolve()
        return resolved_candidate if resolved_candidate.is_file() else None
    venv = os.getenv(SIMREADY_BENCHMARK_VENV_ENV)
    if venv:
        scripts = "Scripts" if os.name == "nt" else "bin"
        name = "simready-benchmark.exe" if os.name == "nt" else "simready-benchmark"
        candidate = Path(venv).expanduser().resolve() / scripts / name
        return candidate if candidate.is_file() else None
    resolved = shutil.which("simready-benchmark")
    return Path(resolved).resolve() if resolved else None


def _benchmark_version(executable: Path) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15.0,
            env=_benchmark_subprocess_environment(executable),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"Could not query simready-benchmark version: {exc}"
    output = f"{completed.stdout}\n{completed.stderr}"
    match = _VERSION_PATTERN.search(output)
    if completed.returncode != 0 or match is None:
        return None, "simready-benchmark --version did not return a supported identity."
    return match.group(1), None


def _distribution_version_from_sibling_python(
    executable: Path, distribution: str
) -> str | None:
    python_name = "python.exe" if os.name == "nt" else "python"
    python = executable.parent / python_name
    if not python.is_file():
        return None
    script = f"import importlib.metadata as m; print(m.version({distribution!r}))"
    try:
        completed = subprocess.run(
            [str(python), "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=15.0,
            env=_benchmark_subprocess_environment(executable),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _benchmark_subprocess_environment(executable: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    current_path = environment.get("PATH", os.defpath)
    environment["PATH"] = os.pathsep.join((str(executable.parent), current_path))
    python_name = "python.exe" if os.name == "nt" else "python"
    if (executable.parent / python_name).is_file():
        environment["VIRTUAL_ENV"] = str(executable.parent.parent)
    else:
        environment.pop("VIRTUAL_ENV", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _configured_input_paths(
    params: SimReadyRuntimeValidationInput, *, asset_path: Path
) -> dict[str, Path]:
    paths = {"asset_path": asset_path, **_runtime_configuration_paths(params)}
    benchmark_executable = _benchmark_executable_candidate(params.benchmark_executable)
    if benchmark_executable is not None:
        paths["benchmark_executable"] = benchmark_executable
    return paths


def _runtime_configuration_paths(
    params: SimReadyRuntimeValidationInput,
) -> dict[str, Path]:
    paths = {
        "sr_specs_path": Path(params.sr_specs_path).expanduser().resolve(),
        "engines_toml_path": Path(params.engines_toml_path).expanduser().resolve(),
    }
    if params.project_config_path is not None:
        paths["project_config_path"] = (
            Path(params.project_config_path).expanduser().resolve()
        )
    for index, path_value in enumerate(params.tests_paths):
        paths[f"tests_paths[{index}]"] = Path(path_value).expanduser().resolve()
    return paths


def _runtime_input_tree_roster(root: Path) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    if not root.is_dir():
        raise ValueError(f"runtime input directory is missing: {root}")
    directories: list[str] = []
    files: list[Path] = []
    for entry in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise ValueError(f"runtime input tree contains a symlink: {entry}")
        if entry.is_dir():
            directories.append(relative)
        elif entry.is_file():
            files.append(entry)
        else:
            raise ValueError(f"runtime input tree contains a special file: {entry}")
    return tuple(directories), tuple(files)


def _runtime_input_digest(
    *,
    kind: Literal["file", "directory"],
    root: Path,
    directories: Sequence[str],
    files: Sequence[ExecutionArtifactBinding],
) -> str:
    roster = {
        "schema_version": "content-agent-workflows.simready-runtime-input.v1",
        "kind": kind,
        "directories": list(directories),
        "files": [
            {
                "path": (
                    Path(binding.path).relative_to(root).as_posix()
                    if kind == "directory"
                    else Path(binding.path).name
                ),
                "sha256": binding.sha256,
                "size_bytes": binding.size_bytes,
            }
            for binding in files
        ],
    }
    return hashlib.sha256(
        json.dumps(roster, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _capture_runtime_input_binding(
    *,
    label: str,
    path: Path,
    kind: Literal["file", "directory"],
) -> SimReadyRuntimeInputBinding:
    files: tuple[ExecutionArtifactBinding, ...]
    if kind == "file":
        if not path.is_file():
            raise ValueError(f"runtime input file is missing: {path}")
        files = (execution_artifact_binding(path),)
        directories: tuple[str, ...] = ()
    else:
        directories, paths = _runtime_input_tree_roster(path)
        files = tuple(execution_artifact_binding(item) for item in paths)
        after_directories, after_paths = _runtime_input_tree_roster(path)
        if directories != after_directories or paths != after_paths:
            raise ValueError(f"runtime input tree changed while binding: {path}")
    return SimReadyRuntimeInputBinding(
        label=label,
        kind=kind,
        path=str(path),
        sha256=_runtime_input_digest(
            kind=kind,
            root=path,
            directories=directories,
            files=files,
        ),
        directories=directories,
        files=files,
    )


def _capture_runtime_input_bindings(
    params: SimReadyRuntimeValidationInput,
) -> tuple[list[SimReadyRuntimeInputBinding], list[str]]:
    bindings: list[SimReadyRuntimeInputBinding] = []
    errors: list[str] = []
    for label, path in _runtime_configuration_paths(params).items():
        kind: Literal["file", "directory"] = (
            "directory"
            if label == "sr_specs_path" or label.startswith("tests_paths[")
            else "file"
        )
        try:
            bindings.append(
                _capture_runtime_input_binding(label=label, path=path, kind=kind)
            )
        except (OSError, ValueError, VerifiedOperationError) as exc:
            errors.append(f"Could not bind runtime input {label} at {path}: {exc}")
    return bindings, _dedupe(errors)


def _runtime_input_binding_errors(
    expected: Sequence[SimReadyRuntimeInputBinding],
) -> list[str]:
    errors: list[str] = []
    labels = [binding.label for binding in expected]
    if len(labels) != len(set(labels)):
        errors.append("Runtime input binding labels are not unique.")
    for binding in expected:
        try:
            observed = _capture_runtime_input_binding(
                label=binding.label,
                path=Path(binding.path),
                kind=binding.kind,
            )
        except (OSError, ValueError, VerifiedOperationError) as exc:
            errors.append(
                f"Could not reverify runtime input {binding.label} at "
                f"{binding.path}: {exc}"
            )
            continue
        if observed != binding:
            errors.append(
                f"Runtime input {binding.label} changed during or after execution: "
                f"{binding.path}"
            )
    return _dedupe(errors)


def _generated_path_conflicts_with_inputs(
    generated_path: Path, input_paths: Iterable[Path]
) -> bool:
    return any(
        _is_relative_to(generated_path, input_path) for input_path in input_paths
    )


def _protected_output_tree_errors(
    output_dir: Path, protected_input_paths: Mapping[str, Path]
) -> list[str]:
    return _dedupe(
        [
            "output_dir must not be the same as or inside protected input "
            f"{label}: {path}"
            for label, path in protected_input_paths.items()
            if _is_relative_to(output_dir, path)
        ]
    )


def _fallback_runtime_report_path(
    output_dir: Path, *, forbidden_paths: set[Path]
) -> Path | None:
    for index in range(1000):
        suffix = "" if index == 0 else f"-{index}"
        candidate = output_dir / f"simready-runtime-validation{suffix}.json"
        if (
            not _generated_path_conflicts_with_inputs(candidate, forbidden_paths)
            and not candidate.is_dir()
        ):
            return candidate
    return None


def _non_directory_ancestor(path: Path) -> Path | None:
    """Return the nearest existing ancestor that cannot contain ``path``."""

    current = path.parent
    while current != current.parent:
        if current.exists() or current.is_symlink():
            return None if current.is_dir() else current
        current = current.parent
    return None


def _validate_input_paths(
    *,
    params: SimReadyRuntimeValidationInput,
    asset_path: Path,
    native_output_dir: Path,
    verified_operations_dir: Path,
    generated_file_paths: Mapping[str, Path],
) -> list[str]:
    errors: list[str] = []
    if not asset_path.is_file():
        errors.append(f"Asset path is not a file: {asset_path}")
    elif asset_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        errors.append(
            "SimReady Benchmark requires a local .usd, .usda, or .usdc asset."
        )

    protected_paths = _configured_input_paths(params, asset_path=asset_path)
    required_paths = {
        "sr_specs_path": protected_paths["sr_specs_path"],
        "engines_toml_path": protected_paths["engines_toml_path"],
    }
    if not required_paths["sr_specs_path"].is_dir():
        errors.append(
            f"SimReady sr_specs directory does not exist: {required_paths['sr_specs_path']}"
        )
    if not required_paths["engines_toml_path"].is_file():
        errors.append(
            "SimReady Benchmark engines.toml does not exist: "
            f"{required_paths['engines_toml_path']}"
        )
    if params.project_config_path is not None:
        project_config = protected_paths["project_config_path"]
        if not project_config.is_file():
            errors.append(f"Project config does not exist: {project_config}")
    for index, path_value in enumerate(params.tests_paths):
        tests_path = protected_paths[f"tests_paths[{index}]"]
        if not tests_path.is_dir():
            errors.append(f"Benchmark tests path does not exist: {tests_path}")
    for output_label, output_path in generated_file_paths.items():
        if output_path.is_dir():
            errors.append(
                f"Generated {output_label} path is a directory: {output_path}"
            )
        non_directory_ancestor = _non_directory_ancestor(output_path)
        if non_directory_ancestor is not None:
            errors.append(
                f"Generated {output_label} path has a non-directory ancestor: "
                f"{non_directory_ancestor}"
            )
    for label, path in protected_paths.items():
        if path == native_output_dir or _is_relative_to(path, native_output_dir):
            errors.append(
                f"{label} must not be inside the managed benchmark output directory: "
                f"{native_output_dir}"
            )
        if path == verified_operations_dir or _is_relative_to(
            path, verified_operations_dir
        ):
            errors.append(
                f"{label} must not be inside the managed verified-operation "
                f"output directory: {verified_operations_dir}"
            )
        for output_label, output_path in generated_file_paths.items():
            if _is_relative_to(output_path, path):
                errors.append(
                    f"{label} must not collide with generated {output_label}: "
                    f"{output_path}"
                )
    return _dedupe(errors)


def _asset_dependency_paths(
    manifest: Mapping[str, Any],
) -> tuple[set[Path], list[str]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        return set(), ["Asset dependency manifest has no file roster."]
    paths: set[Path] = set()
    errors: list[str] = []
    for item in files:
        path_value = item.get("path") if isinstance(item, Mapping) else None
        if not isinstance(path_value, str):
            errors.append("Asset dependency manifest contains an invalid path.")
            continue
        paths.add(Path(path_value))
    return paths, _dedupe(errors)


def _managed_dependency_output_errors(
    dependency_paths: Iterable[Path],
    *,
    native_output_dir: Path,
    verified_operations_dir: Path,
    generated_file_paths: Mapping[str, Path],
) -> list[str]:
    errors: list[str] = []
    for path in dependency_paths:
        for label, managed_dir in (
            ("benchmark", native_output_dir),
            ("verified-operation", verified_operations_dir),
        ):
            if path == managed_dir or _is_relative_to(path, managed_dir):
                errors.append(
                    "Asset dependency must not be inside the managed "
                    f"{label} output directory: {path}"
                )
        for output_label, output_path in generated_file_paths.items():
            if path == output_path:
                errors.append(
                    "Asset dependency must not collide with generated "
                    f"{output_label}: {path}"
                )
    return _dedupe(errors)


def _prepare_native_output_dir(path: Path) -> str | None:
    try:
        if path.is_symlink():
            return f"Refusing symlinked benchmark output directory: {path}"
        if path.exists():
            if not path.is_dir():
                return f"Benchmark output path is not a directory: {path}"
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        return f"Could not prepare benchmark output directory {path}: {exc}"
    return None


def _clear_managed_output_dir(path: Path) -> str | None:
    try:
        if path.is_symlink():
            return f"Refusing symlinked managed output directory: {path}"
        if path.exists():
            if not path.is_dir():
                return f"Managed output path is not a directory: {path}"
            shutil.rmtree(path)
    except OSError as exc:
        return f"Could not clear managed output directory {path}: {exc}"
    return None


def _invalidate_previous_publication(path: Path) -> str | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink():
        return f"Refusing symlinked managed output directory: {path}"
    if not path.is_dir():
        return f"Managed output path is not a directory: {path}"
    marker = path / SIMREADY_RUNTIME_PUBLICATION_INVALIDATED_NAME
    validity = path / SIMREADY_RUNTIME_PUBLICATION_VALIDITY_NAME
    if marker.is_symlink() or (marker.exists() and not marker.is_file()):
        return f"Publication invalidation marker is not a regular file: {marker}"
    if validity.is_symlink() or (validity.exists() and not validity.is_file()):
        return f"Publication validity token is not a regular file: {validity}"
    try:
        if validity.exists():
            atomic_write_json(
                validity,
                {
                    "schema_version": (
                        "content-agent-workflows.simready-publication-validity.v1"
                    ),
                    "status": "invalidated",
                    "generation": secrets.token_hex(32),
                },
            )
        else:
            # Publications created before validity tokens still need one of each
            # envelope's bound artifacts revoked. The generated projection is a
            # safe publication-local target and every legacy envelope binds it.
            for projection in path.rglob("verified_operation_projection.json"):
                if projection.is_symlink() or not projection.is_file():
                    return (
                        "Legacy publication projection is not a regular file: "
                        f"{projection}"
                    )
                projection.unlink()
        if not marker.exists():
            atomic_write_json(
                marker,
                {
                    "schema_version": (
                        "content-agent-workflows.simready-publication-invalidated.v1"
                    ),
                    "status": "invalidated",
                },
            )
    except (OSError, ValueError) as exc:
        return f"Could not invalidate previous SimReady publication: {exc}"
    return None


def _run_bounded_benchmark_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_log_path: Path,
    stderr_log_path: Path,
    timeout_s: float,
    stdout_limit_bytes: int,
    stderr_limit_bytes: int,
) -> tuple[int | None, str | None, list[str]]:
    warnings: list[str] = []
    for log_path in (stdout_log_path, stderr_log_path):
        try:
            if log_path.is_symlink() or log_path.is_file():
                log_path.unlink()
            elif log_path.exists():
                return (
                    None,
                    f"Benchmark log path is not a regular file: {log_path}",
                    warnings,
                )
        except OSError as exc:
            return None, f"Could not prepare benchmark log {log_path}: {exc}", warnings
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
    except OSError as exc:
        return None, f"Could not start simready-benchmark: {exc}", warnings
    process_group_id = process.pid
    assert process.stdout is not None
    assert process.stderr is not None

    truncation = {"stdout": False, "stderr": False}
    stdout_thread = threading.Thread(
        target=_drain_bounded_stream,
        args=(
            process.stdout,
            stdout_log_path,
            stdout_limit_bytes,
            truncation,
            "stdout",
        ),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain_bounded_stream,
        args=(
            process.stderr,
            stderr_log_path,
            stderr_limit_bytes,
            truncation,
            "stderr",
        ),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    process_error: str | None = None
    try:
        _wait_for_process_exit_without_reaping(process, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        process_error = f"simready-benchmark timed out after {timeout_s:g} seconds."
    finally:
        _terminate_process_tree(process, process_group_id=process_group_id)
        stdout_thread.join(timeout=10.0)
        stderr_thread.join(timeout=10.0)
    for stream_name, was_truncated in truncation.items():
        if was_truncated:
            warnings.append(
                f"simready-benchmark {stream_name} log exceeded its configured "
                "retention limit and was truncated."
            )
    return process.returncode, process_error, warnings


def _drain_bounded_stream(
    stream: Any,
    path: Path,
    limit_bytes: int,
    truncation: dict[str, bool],
    key: str,
) -> None:
    retained = 0
    with path.open("wb") as output:
        while chunk := stream.read(64 * 1024):
            remaining = max(0, limit_bytes - retained)
            if remaining:
                output.write(chunk[:remaining])
                retained += min(len(chunk), remaining)
            if len(chunk) > remaining:
                truncation[key] = True


def _wait_for_process_exit_without_reaping(
    process: subprocess.Popen[bytes], *, timeout_s: float
) -> None:
    if os.name == "nt":  # pragma: win32 cover
        process.wait(timeout=timeout_s)
        return

    deadline = time.monotonic() + timeout_s
    wait_options = os.WEXITED | os.WNOHANG | os.WNOWAIT
    while True:
        try:
            result = os.waitid(os.P_PID, process.pid, wait_options)
        except InterruptedError:
            continue
        if result is not None:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout_s)
        time.sleep(min(0.05, remaining))


def _terminate_process_tree(
    process: subprocess.Popen[bytes], *, process_group_id: int
) -> None:
    if os.name != "nt":
        # The leader is deliberately unreaped here, so its PID still owns this
        # process-group ID while both signals are sent.
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except OSError:
            pass
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except OSError:
            pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return

    if process.poll() is not None:  # pragma: win32 cover
        return
    process.terminate()  # pragma: win32 cover
    try:  # pragma: win32 cover
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:  # pragma: win32 cover
        process.kill()
        process.wait()


def _load_json_object(
    path: Path, *, max_bytes: int
) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"Expected benchmark artifact is missing: {path}"
    try:
        if path.stat().st_size > max_bytes:
            return None, f"Benchmark JSON artifact exceeds {max_bytes} bytes: {path}"
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"Could not read benchmark JSON artifact {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"Benchmark JSON artifact is not an object: {path}"
    return payload, None


def _parse_runtime_error_events(payload: str) -> list[dict[str, Any]]:
    """Parse the pinned Benchmark JSONL stream and retain runtime-failure events."""

    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError(
                f"benchmark event {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise ValueError(f"benchmark event {line_number} is not an object")
        if event.get("type") in _RUNTIME_ERROR_EVENTS:
            events.append(event)
    return events


def _load_runtime_error_events(
    path: Path, *, max_bytes: int
) -> tuple[list[dict[str, Any]], str | None]:
    if not path.is_file():
        return [], f"Expected benchmark event stream is missing: {path}"
    try:
        if path.stat().st_size > max_bytes:
            return [], f"Benchmark event stream exceeds {max_bytes} bytes: {path}"
        return _parse_runtime_error_events(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return [], f"Could not read benchmark event stream {path}: {exc}"


def _artifact_digests(
    paths: Mapping[str, Path],
) -> tuple[dict[str, str], list[str]]:
    digests: dict[str, str] = {}
    warnings: list[str] = []
    for name, path in paths.items():
        if not path.is_file():
            continue
        try:
            digests[name] = file_sha256(path)
        except (OSError, ValueError) as exc:
            warnings.append(f"Could not hash benchmark artifact {path}: {exc}")
    return digests, warnings


def _normalize_benchmark_result(
    *,
    exit_code: int | None,
    process_error: str | None,
    expected_asset_path: Path,
    native_plan: dict[str, Any] | None,
    native_plan_error: str | None,
    native_report: dict[str, Any] | None,
    native_report_error: str | None,
    run_summary: dict[str, Any] | None,
    run_summary_error: str | None,
    runtime_error_events: list[dict[str, Any]],
    events_error: str | None,
) -> tuple[
    RuntimeDisposition,
    list[SimReadyRuntimeCheckResult],
    list[str],
    list[str],
]:
    warnings: list[str] = []
    errors = [item for item in (process_error,) if item]
    checks: list[SimReadyRuntimeCheckResult] = []
    validation_failed = False
    if native_report is not None:
        metadata = native_report.get("metadata")
        if not isinstance(metadata, Mapping):
            errors.append("Native benchmark report metadata is missing or malformed.")
        else:
            schema_version = str(metadata.get("schema_version") or "")
            framework_version = str(metadata.get("framework_version") or "")
            if schema_version != SIMREADY_BENCHMARK_NATIVE_REPORT_SCHEMA_VERSION:
                errors.append(
                    "Unsupported native benchmark report schema: expected "
                    f"{SIMREADY_BENCHMARK_NATIVE_REPORT_SCHEMA_VERSION}, observed "
                    f"{schema_version or 'unknown'}."
                )
            if framework_version != SIMREADY_BENCHMARK_VERSION:
                errors.append(
                    "Native benchmark report framework version differs from the "
                    f"pinned runtime: {framework_version or 'unknown'}."
                )
        checks, check_errors, validation_failed = native_benchmark_checks(
            native_report,
            expected_asset_path=expected_asset_path,
            native_plan=native_plan,
        )
        errors.extend(check_errors)
    elif native_report_error:
        errors.append(native_report_error)

    if native_plan is None and native_plan_error:
        errors.append(native_plan_error)

    if run_summary is None:
        if run_summary_error:
            errors.append(run_summary_error)
    else:
        total_work_items = run_summary.get("total_work_items")
        valid_total_work_items = isinstance(total_work_items, int) and not isinstance(
            total_work_items, bool
        )
        if not valid_total_work_items:
            errors.append(
                "Benchmark run_summary total_work_items is missing or invalid."
            )
        elif total_work_items <= 0:
            errors.append(
                "SimReady Benchmark planned no work items; total_work_items must be "
                "positive. Check sr_specs, test packs, and feature overrides."
            )
        completed_work_items = run_summary.get("completed")
        if (
            not isinstance(completed_work_items, int)
            or isinstance(completed_work_items, bool)
            or completed_work_items < 0
        ):
            errors.append(
                "Benchmark run_summary completed count is missing or invalid."
            )
        elif valid_total_work_items and completed_work_items != total_work_items:
            errors.append(
                "SimReady Benchmark did not complete every planned work item: "
                f"completed={completed_work_items}, total={total_work_items}."
            )
        run_status = run_summary.get("status")
        if run_status != "completed":
            errors.append(
                "Benchmark run_summary status is not completed: "
                f"{run_status or 'missing'}."
            )
        failed_work_items = run_summary.get("failed")
        if (
            not isinstance(failed_work_items, int)
            or isinstance(failed_work_items, bool)
            or failed_work_items < 0
        ):
            errors.append("Benchmark run_summary failed count is missing or invalid.")
        elif failed_work_items:
            errors.append(
                "SimReady Benchmark reported failed orchestration work items: "
                f"{failed_work_items}."
            )
        readiness = run_summary.get("readiness")
        if not isinstance(readiness, Mapping) or not readiness.get("status"):
            errors.append("Benchmark run_summary readiness is missing.")
        elif readiness.get("status") not in {"ready", "not_ready"}:
            errors.append(
                "Benchmark run_summary readiness status is unsupported: "
                f"{readiness.get('status')}."
            )

    if events_error:
        errors.append(events_error)
    if process_error or runtime_error_events:
        return "error", checks, warnings, _dedupe(errors)
    if exit_code == 4 or _readiness_not_ready(run_summary):
        return "blocked", checks, warnings, _dedupe(errors)
    if validation_failed:
        errors.append(_STATIC_VALIDATION_BLOCK_ERROR)
        return "blocked", checks, warnings, _dedupe(errors)
    if exit_code not in {0, 1}:
        errors.append(f"simready-benchmark exited with code {exit_code}.")
    if errors:
        return "error", checks, warnings, _dedupe(errors)
    if not checks:
        return "error", checks, warnings, ["Native benchmark report contains no tests."]

    dispositions = {check.disposition for check in checks}
    if "blocked" in dispositions:
        status: RuntimeDisposition = "blocked"
    elif "error" in dispositions:
        status = "error"
    elif "fail" in dispositions:
        status = "fail"
    elif dispositions == {"not_evaluated"}:
        status = "not_evaluated"
    elif "not_evaluated" in dispositions:
        status = "warn"
        warnings.append(
            "Some runtime checks were skipped; their native dispositions remain "
            "available in checks."
        )
    else:
        status = "pass"

    expected_exit_code = 1 if status == "fail" else 0
    if (
        status in {"pass", "warn", "fail", "not_evaluated"}
        and exit_code != expected_exit_code
    ):
        return (
            "error",
            checks,
            warnings,
            [
                "Benchmark exit code is inconsistent with the native per-test "
                f"results: exit={exit_code}, status={status}."
            ],
        )
    return status, checks, warnings, []


def native_benchmark_checks(
    native_report: Mapping[str, Any],
    *,
    expected_asset_path: str | Path | None = None,
    native_plan: Mapping[str, Any] | None = None,
) -> tuple[list[SimReadyRuntimeCheckResult], list[str], bool]:
    """Return normalized native checks and fail-closed contract errors."""

    checks: list[SimReadyRuntimeCheckResult] = []
    errors: list[str] = []
    validation_failed = False
    native_result_paths: set[tuple[str, str]] = set()
    assets = native_report.get("assets")
    if not isinstance(assets, Mapping) or len(assets) != 1:
        return [], ["Native benchmark report must contain exactly one asset."], False
    asset_key, asset_payload = next(iter(assets.items()))
    if not isinstance(asset_key, str):
        return [], ["Native benchmark report asset key must be a string."], False
    if not isinstance(asset_payload, Mapping):
        return [], [f"Native asset result is malformed: {asset_key}"], False
    errors.extend(
        _native_asset_identity_errors(
            asset_key=asset_key,
            asset_payload=asset_payload,
            expected_asset_path=expected_asset_path,
            native_plan=native_plan,
        )
    )
    profiles = asset_payload.get("profiles")
    if not _is_mapping_sequence(profiles):
        errors.append(f"Native asset profiles are malformed: {asset_key}")
        return checks, errors, validation_failed
    for profile in profiles:
        profile_id = _required_native_identity_field(
            profile, field="id", scope="profile", errors=errors
        )
        profile_version = _required_native_identity_field(
            profile, field="version", scope="profile", errors=errors
        )
        profile_label = profile_id or "<invalid-profile>"
        profile_status = str(profile.get("status") or "")
        feature_statuses: list[str] = []
        features = profile.get("features")
        if not _is_mapping_sequence(features):
            errors.append(f"Native profile features are malformed: {profile_label}")
            continue
        for feature in features:
            feature_id = _required_native_identity_field(
                feature, field="id", scope="feature", errors=errors
            )
            feature_version = _required_native_identity_field(
                feature, field="version", scope="feature", errors=errors
            )
            feature_label = feature_id or "<invalid-feature>"
            feature_status = str(feature.get("status") or "")
            feature_statuses.append(feature_status)
            if feature_status == "validation_failed":
                validation_failed = True
            tests = feature.get("tests")
            if tests is None:
                tests = []
            if not _is_mapping_sequence(tests, allow_empty=True):
                errors.append(f"Native feature tests are malformed: {feature_label}")
                continue
            test_statuses: list[str] = []
            for test in tests:
                test_name = _required_native_identity_field(
                    test, field="name", scope="test", errors=errors
                )
                test_version = _required_native_identity_field(
                    test, field="version", scope="test", errors=errors
                )
                test_label = test_name or "<invalid-test>"
                if test_name is not None:
                    native_result_path = (asset_key, test_name)
                    if native_result_path in native_result_paths:
                        errors.append(
                            "Native benchmark report contains duplicate per-test "
                            "result identity for asset "
                            f"{asset_key!r} and test {test_name!r}; SimReady Benchmark "
                            f"{SIMREADY_BENCHMARK_VERSION} stores one result.json per "
                            "asset and test."
                        )
                    native_result_paths.add(native_result_path)
                native_status = str(test.get("status") or "")
                test_statuses.append(native_status)
                if native_status not in _NATIVE_TEST_STATUSES:
                    errors.append(
                        f"Unknown native status {native_status!r} for test {test_label}."
                    )
                errors.extend(_native_test_field_errors(test, test_name=test_label))
                if (
                    profile_id is None
                    or profile_version is None
                    or feature_id is None
                    or feature_version is None
                    or test_name is None
                    or test_version is None
                ):
                    continue
                media = test.get("media")
                checks.append(
                    SimReadyRuntimeCheckResult(
                        asset_key=asset_key,
                        profile_id=profile_id,
                        profile_version=profile_version,
                        feature_id=feature_id,
                        feature_version=feature_version,
                        test_name=test_name,
                        test_version=test_version,
                        native_status=native_status,
                        disposition=_native_disposition(native_status),
                        duration_s=_optional_float(test.get("duration")),
                        engine=_optional_string(test.get("engine")),
                        engine_version=_optional_string(test.get("engine_version")),
                        message=str(test.get("message") or ""),
                        metrics=dict(test.get("metrics") or {})
                        if isinstance(test.get("metrics"), Mapping)
                        else {},
                        media=(
                            [dict(item) for item in media]
                            if _is_mapping_sequence(media, allow_empty=True)
                            else []
                        ),
                        kit_logs=list(test.get("kit_logs") or [])
                        if isinstance(test.get("kit_logs"), list)
                        else [],
                    )
                )
            if feature_status in _NATIVE_PLAN_ONLY_FEATURE_STATUSES:
                if tests:
                    errors.append(
                        "Native plan-only feature status is inconsistent with its "
                        f"tests: feature={feature_label!r}, status={feature_status!r}."
                    )
            elif feature_status not in _NATIVE_TEST_STATUSES:
                errors.append(
                    f"Unknown native aggregate status {feature_status!r} for "
                    f"feature {feature_label}."
                )
            else:
                derived_feature_status = _native_aggregate_status(test_statuses)
                if feature_status != derived_feature_status:
                    errors.append(
                        "Native feature aggregate status is inconsistent with its "
                        f"tests: feature={feature_label!r}, "
                        f"reported={feature_status!r}, "
                        f"derived={derived_feature_status!r}."
                    )
        if profile_status not in _NATIVE_TEST_STATUSES:
            errors.append(
                f"Unknown native aggregate status {profile_status!r} for profile "
                f"{profile_label}."
            )
        else:
            derived_profile_status = _native_aggregate_status(feature_statuses)
            if profile_status != derived_profile_status:
                errors.append(
                    "Native profile aggregate status is inconsistent with its "
                    f"features: profile={profile_label!r}, "
                    f"reported={profile_status!r}, "
                    f"derived={derived_profile_status!r}."
                )
    return checks, errors, validation_failed


def _native_aggregate_status(statuses: Sequence[str]) -> str:
    """Mirror SimReady Benchmark 2026.6.5 reporter fail-dominates rollup."""

    if "fail" in statuses:
        return "fail"
    if "blocked" in statuses:
        return "blocked"
    if "incomplete" in statuses:
        return "incomplete"
    if statuses and all(status == "skipped" for status in statuses):
        return "skipped"
    if "pass" in statuses:
        return "pass"
    return "skipped"


def _native_asset_identity_errors(
    *,
    asset_key: str,
    asset_payload: Mapping[str, Any],
    expected_asset_path: str | Path | None,
    native_plan: Mapping[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    reported_asset_path = asset_payload.get("asset_path")
    if reported_asset_path != asset_key:
        errors.append(
            "Native benchmark report asset_path differs from its asset key: "
            f"key={asset_key!r}, asset_path={reported_asset_path!r}."
        )
    if expected_asset_path is None:
        return errors

    expected_path = Path(expected_asset_path).expanduser().resolve()
    expected_key = _benchmark_asset_key(str(expected_path), "")
    if native_plan is not None:
        plan_assets = native_plan.get("assets")
        if (
            not isinstance(plan_assets, list)
            or len(plan_assets) != 1
            or not isinstance(plan_assets[0], Mapping)
        ):
            errors.append("Native benchmark plan must contain exactly one asset.")
            return errors
        plan_asset_path = plan_assets[0].get("path")
        if not isinstance(plan_asset_path, str) or not plan_asset_path:
            errors.append("Native benchmark plan asset path is missing or malformed.")
            return errors
        content_root = native_plan.get("content_root")
        if not isinstance(content_root, str):
            errors.append("Native benchmark plan content_root is malformed.")
            return errors
        resolved_plan_path = Path(plan_asset_path).expanduser()
        if not resolved_plan_path.is_absolute() and content_root:
            resolved_plan_path = Path(content_root).expanduser() / resolved_plan_path
        if resolved_plan_path.resolve() != expected_path:
            errors.append(
                "Native benchmark plan asset differs from the requested asset: "
                f"plan={resolved_plan_path.resolve()}, requested={expected_path}."
            )
        expected_key = _benchmark_asset_key(plan_asset_path, content_root)

    if asset_key != expected_key:
        errors.append(
            "Native benchmark report names a different asset than requested: "
            f"report={asset_key!r}, expected={expected_key!r}."
        )
    return errors


def _benchmark_asset_key(asset_path: str, content_root: str) -> str:
    """Mirror ``simready_benchmark.core.asset_key.asset_rel_key``.

    This contract is pinned with SimReady Benchmark ``2026.6.5`` via
    :data:`SIMREADY_BENCHMARK_VERSION`.
    """

    if content_root:
        try:
            relative = os.path.relpath(asset_path, content_root)
            if not relative.startswith(os.pardir):
                return relative.replace("\\", "/")
        except ValueError:
            pass
    _drive, tail = os.path.splitdrive(asset_path)
    return tail.lstrip("/\\").replace("\\", "/")


def _native_disposition(status: str) -> RuntimeDisposition:
    dispositions: dict[str, RuntimeDisposition] = {
        "pass": "pass",
        "fail": "fail",
        "skipped": "not_evaluated",
        "incomplete": "error",
        "blocked": "blocked",
    }
    return dispositions.get(status, "error")


def _readiness_not_ready(run_summary: Mapping[str, Any] | None) -> bool:
    if run_summary is None:
        return False
    readiness = run_summary.get("readiness")
    return isinstance(readiness, Mapping) and readiness.get("status") == "not_ready"


def _is_mapping_sequence(
    value: Any, *, allow_empty: bool = False
) -> TypeGuard[list[Mapping[str, Any]]]:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, Mapping) for item in value)
    )


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value


def _required_native_identity_field(
    payload: Mapping[str, Any],
    *,
    field: str,
    scope: str,
    errors: list[str],
) -> str | None:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"Native {scope} {field} is missing or malformed.")
        return None
    return value


def _optional_float(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _native_test_field_errors(test: Mapping[str, Any], *, test_name: str) -> list[str]:
    errors: list[str] = []
    for field in ("engine", "engine_version"):
        value = test.get(field)
        if value is not None and not isinstance(value, str):
            errors.append(f"Native test {field} is malformed: {test_name}.")

    for field in ("message", "description", "expected_video", "scene_file"):
        value = test.get(field)
        if value is not None and not isinstance(value, str):
            errors.append(f"Native test {field} is malformed: {test_name}.")

    duration = test.get("duration")
    if duration is not None and _optional_float(duration) is None:
        errors.append(f"Native test duration is malformed: {test_name}.")

    metrics = test.get("metrics")
    if metrics is not None and not isinstance(metrics, Mapping):
        errors.append(f"Native test metrics are malformed: {test_name}.")

    media = test.get("media")
    if media is not None and not _is_mapping_sequence(media, allow_empty=True):
        errors.append(f"Native test media are malformed: {test_name}.")

    kit_logs = test.get("kit_logs")
    if kit_logs is not None and not isinstance(kit_logs, list):
        errors.append(f"Native test kit_logs are malformed: {test_name}.")
    return errors


def _base_report(
    *,
    asset_path: Path,
    status: RuntimeDisposition,
    errors: list[str],
    warnings: list[str] | None = None,
    next_step: str,
) -> SimReadyRuntimeValidationReport:
    return SimReadyRuntimeValidationReport(
        passed=False,
        status=status,
        asset_path=str(asset_path),
        warnings=warnings or [],
        errors=errors,
        next_step=next_step,
    )


def _write_runtime_report(
    report: SimReadyRuntimeValidationReport,
    *,
    report_path: Path,
    template_result_path: Path,
    write_template_result: bool = True,
) -> SimReadyRuntimeValidationReport:
    from content_agent_workflows.validation.verified_operations import (
        VerifiedOperationError,
    )

    from .runtime_verified_operations import (
        SIMREADY_RUNTIME_VERIFIED_OPERATIONS_NAME,
        project_simready_runtime_verified_operations,
    )

    publication_root = (
        template_result_path.parent / SIMREADY_RUNTIME_VERIFIED_OPERATIONS_DIRNAME
    )
    publication_path = publication_root / SIMREADY_RUNTIME_VERIFIED_OPERATIONS_NAME
    should_publish = bool(
        write_template_result
        and report.status in {"pass", "warn", "fail", "not_evaluated"}
        and report.checks
        and report.native_plan_path
        and report.native_report_path
        and report.asset_sha256
        and report.benchmark_executable
        and report.benchmark_executable_sha256
    )
    resolved = report.model_copy(
        update={
            "report_path": str(report_path),
            "validation_template_result_path": (
                str(template_result_path) if write_template_result else None
            ),
            "verified_operation_publication_path": (
                str(publication_path) if should_publish else None
            ),
        }
    )
    atomic_write_json(report_path, resolved)
    if write_template_result:
        atomic_write_json(
            template_result_path,
            simready_runtime_validation_template_result(resolved),
        )
    if not should_publish:
        return resolved
    try:
        project_simready_runtime_verified_operations(
            report_path,
            output_dir=publication_root,
        )
    except (OSError, ValueError, VerifiedOperationError) as exc:
        cleanup_error = _clear_managed_output_dir(publication_root)
        projection_errors = [
            *resolved.errors,
            f"Could not publish verified SimReady runtime operations: {exc}",
        ]
        if cleanup_error is not None:
            projection_errors.append(cleanup_error)
        resolved = resolved.model_copy(
            update={
                "passed": False,
                "status": "error",
                "verified_operation_publication_path": None,
                "errors": _dedupe(projection_errors),
                "next_step": "inspect-simready-benchmark-runtime",
            }
        )
        atomic_write_json(report_path, resolved)
        if write_template_result:
            atomic_write_json(
                template_result_path,
                simready_runtime_validation_template_result(resolved),
            )
    return resolved


def _next_step(status: RuntimeDisposition) -> str:
    return {
        "pass": "complete",
        "warn": "review-simready-runtime-warnings",
        "fail": "repair-asset-runtime-behavior",
        "not_evaluated": "select-simready-runtime-tests",
        "blocked": "configure-simready-benchmark",
        "error": "inspect-simready-benchmark-runtime",
    }[status]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _dedupe(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in items if item.strip()))


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def _benchmark_filter_value(raw: str) -> str:
    value = raw.strip()
    if not value or value.startswith("-"):
        raise argparse.ArgumentTypeError(
            "must be non-empty and must not begin with '-'"
        )
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SimReady Foundation runtime validation."
    )
    parser.add_argument("asset_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Single-run workspace; reuse invalidates and removes prior verified "
            "runtime evidence before preflight completes."
        ),
    )
    parser.add_argument("--sr-specs", dest="sr_specs_path", type=Path, required=True)
    parser.add_argument(
        "--engines-toml", dest="engines_toml_path", type=Path, required=True
    )
    parser.add_argument("--project-config", dest="project_config_path", type=Path)
    parser.add_argument("--tests-path", dest="tests_paths", action="append", default=[])
    parser.add_argument(
        "--feature",
        dest="features",
        action="append",
        type=_benchmark_filter_value,
        default=[],
    )
    parser.add_argument(
        "--test",
        dest="tests",
        action="append",
        type=_benchmark_filter_value,
        default=[],
    )
    parser.add_argument(
        "--runtime",
        dest="runtimes",
        action="append",
        type=_benchmark_filter_value,
        default=[],
    )
    parser.add_argument("--benchmark-executable", type=Path)
    parser.add_argument("--report", dest="report_path", type=Path)
    parser.add_argument("--max-concurrent", type=_positive_int, default=1)
    parser.add_argument(
        "--timeout", dest="timeout_s", type=_positive_float, default=3600.0
    )
    parser.add_argument("--stamp-results", action="store_true")
    args = parser.parse_args(argv)
    try:
        validation_input = SimReadyRuntimeValidationInput(
            asset_path=str(args.asset_path),
            output_dir=str(args.output_dir),
            sr_specs_path=str(args.sr_specs_path),
            engines_toml_path=str(args.engines_toml_path),
            project_config_path=str(args.project_config_path)
            if args.project_config_path is not None
            else None,
            tests_paths=tuple(args.tests_paths),
            features=tuple(args.features),
            tests=tuple(args.tests),
            runtimes=tuple(args.runtimes),
            benchmark_executable=str(args.benchmark_executable)
            if args.benchmark_executable is not None
            else None,
            report_path=str(args.report_path) if args.report_path is not None else None,
            max_concurrent=args.max_concurrent,
            timeout_s=args.timeout_s,
            stamp_results=args.stamp_results,
        )
    except ValueError as exc:
        parser.error(str(exc))
    report = run_simready_runtime_validation(validation_input)
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if report.passed:
        return 0
    return 4 if report.status == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
