# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SimReady Benchmark runtime-validation adapter."""

from __future__ import annotations

import json
import os
import signal
import stat
import textwrap
import time
from pathlib import Path, PurePosixPath

import pytest

import content_agent_workflows.simready.runtime_benchmark as runtime_benchmark_module
import content_agent_workflows.simready.runtime_verified_operations as runtime_verified_operations_module
import content_agent_workflows.validation.verified_operations as verified_operations_module
from content_agent_workflows.simready import (
    SIMREADY_BENCHMARK_NATIVE_REPORT_SCHEMA_VERSION,
    SIMREADY_BENCHMARK_VERSION,
    SIMREADY_RUNTIME_VERIFIED_OPERATIONS_DIRNAME,
    SimReadyRuntimeValidationInput,
    load_simready_runtime_verified_operations,
    project_simready_runtime_verified_operations,
    resolve_simready_benchmark_runtime,
    run_simready_runtime_validation,
    simready_runtime_validation_template_result,
)
from content_agent_workflows.validation import (
    collect_verified_operation_evidence,
    ingest_verified_operation_result,
)
from content_agent_workflows.validation.verified_operations import (
    VerifiedOperationError,
)


def _write_usda(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(
            """\
            #usda 1.0
            (
                defaultPrim = "World"
            )

            def Xform "World"
            {
            }
            """
        ),
        encoding="utf-8",
    )


def _write_fake_benchmark(
    tmp_path: Path,
    *,
    status: str = "pass",
    exit_code: int | None = None,
    readiness: str = "ready",
    event_type: str = "test_complete",
    total_work_items: int = 1,
    framework_version: str = SIMREADY_BENCHMARK_VERSION,
    native_framework_version: str | None = None,
    schema_version: str = SIMREADY_BENCHMARK_NATIVE_REPORT_SCHEMA_VERSION,
    feature_status: str | None = None,
    profile_status: str | None = None,
    additional_status: str | None = None,
    result_status: str | None = None,
    mutate_asset: bool = False,
    summary_total_work_items: int | None = None,
    summary_completed: int | None = None,
    summary_failed: int | None = None,
    summary_status: str = "completed",
    reported_asset_key: str | None = None,
    test_name: str = "rigid_body_falls_isaac",
    leave_descendant: bool = False,
    sleep_before_exit_s: float = 0.0,
    session_log_file: str | None = "logs/isaac_sim_1.log",
    malformed_native_field: str | None = None,
    mutate_runtime_input: str | None = None,
) -> Path:
    executable = tmp_path / "benchmark-bin" / "simready-benchmark"
    executable.parent.mkdir(parents=True, exist_ok=True)
    resolved_exit_code = exit_code if exit_code is not None else int(status == "fail")
    resolved_summary_failed = summary_failed if summary_failed is not None else 0
    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import json
        import os
        import pathlib
        import subprocess
        import sys
        import time

        if "--version" in sys.argv:
            print("SimReady Benchmark (v{framework_version})")
            raise SystemExit(0)

        def flag_value(flag):
            return sys.argv[sys.argv.index(flag) + 1]

        output_dir = pathlib.Path(flag_value("--output-dir"))
        asset_path = pathlib.Path(flag_value("--assets"))
        plan_path = pathlib.Path(flag_value("--output"))
        _drive, asset_tail = os.path.splitdrive(str(asset_path))
        asset_key = {reported_asset_key!r} or asset_tail.lstrip("/\\\\").replace("\\\\", "/")
        report_dir = output_dir / "report"
        state_dir = output_dir / "state"
        report_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        session_log_file = {session_log_file!r}
        sessions = []
        if session_log_file:
            session_log_path = output_dir / pathlib.Path(
                *pathlib.PurePosixPath(session_log_file).parts
            )
            session_log_path.parent.mkdir(parents=True, exist_ok=True)
            session_log_path.write_text("fake Kit session log\\n", encoding="utf-8")
            sessions.append({{
                "id": "isaac_sim_1",
                "engine": "isaac_sim",
                "log_file": session_log_file,
                "duration": 1.25,
                "tests_run": 1,
                "test_time": 1.25,
            }})

        test_status = {status!r}
        tests = []
        if {total_work_items}:
            tests.append({{
                "name": {test_name!r},
                "version": "1.2.0",
                "status": test_status,
                "duration": 1.25,
                "engine": "isaac_sim",
                "engine_version": "5.1.0",
                "message": "asset did not settle" if test_status == "fail" else "",
                "metrics": {{"settled_after_s": 1.0}},
                "media": [{{"kind": "video", "filename": "fall.mp4"}}],
                "kit_logs": [],
                "description": "Free-fall stability check.",
                "expected_video": "",
            }})
        additional_status = {additional_status!r}
        if {total_work_items} and additional_status is not None:
            tests.append({{
                "name": "rigid_body_stacks_isaac",
                "version": "2.0.0",
                "status": additional_status,
                "duration": 0.75,
                "engine": "isaac_sim",
                "engine_version": "5.1.0",
                "message": "not selected" if additional_status == "skipped" else "",
                "metrics": {{"retained_stack": True}},
                "media": [{{"kind": "video", "filename": "stack.mp4"}}],
                "kit_logs": [],
                "description": "Stack stability check.",
                "expected_video": "",
            }})

        feature_status = {feature_status!r} or test_status
        profile_status = {profile_status!r} or (
            "skipped" if feature_status == "validation_failed" else feature_status
        )
        report_tests = [dict(test) for test in tests]
        malformed_native_field = {malformed_native_field!r}
        if report_tests and malformed_native_field == "media":
            report_tests[0]["media"] = 1
        if report_tests and malformed_native_field == "engine":
            report_tests[0]["engine"] = ["isaac_sim"]
        plan = {{
            "plan_version": "1.1.0",
            "created": "2026-08-12T00:00:00Z",
            "sr_specs_path": flag_value("--sr-specs"),
            "content_root": "",
            "test_dirs": [],
            "profiles": [],
            "features": [],
            "tests": [],
            "assets": [{{"id": 1, "path": str(asset_path), "tests": [1]}}],
            "overrides": [],
            "engine_configs": {{}},
            "warnings": [],
            "engine_coverage": None,
        }}
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        report = {{
            "metadata": {{
                "schema_version": {schema_version!r},
                "framework_version": {native_framework_version or framework_version!r},
                "generated": "2026-08-12T00:00:00Z",
                "sessions": sessions,
            }},
            "assets": {{
                asset_key: {{
                    "asset_path": asset_key,
                    "profiles": [{{
                        "id": "Prop-Robotics-Neutral",
                        "version": "1.0.0",
                        "status": profile_status,
                        "features": [{{
                            "id": "FET003_BASE_NEUTRAL",
                            "version": "0.1.0",
                            "status": feature_status,
                            "tests": report_tests,
                        }}],
                    }}],
                    "stamping": {{}},
                }}
            }},
        }}
        profile_payload = report["assets"][asset_key]["profiles"][0]
        feature_payload = profile_payload["features"][0]
        test_payload = feature_payload["tests"][0] if feature_payload["tests"] else None
        if malformed_native_field == "profile_id":
            profile_payload["id"] = None
        if malformed_native_field == "profile_version":
            profile_payload["version"] = 7
        if malformed_native_field == "feature_id":
            feature_payload["id"] = None
        if malformed_native_field == "feature_version":
            feature_payload["version"] = 7
        if test_payload is not None and malformed_native_field == "test_name":
            test_payload["name"] = None
        if test_payload is not None and malformed_native_field == "test_version":
            test_payload["version"] = 7
        (report_dir / "test_results_index.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        summary_total_work_items = {summary_total_work_items!r}
        summary_completed = {summary_completed!r}
        run_summary = {{
            "total_work_items": summary_total_work_items if summary_total_work_items is not None else int(bool(tests)),
            "completed": summary_completed if summary_completed is not None else int(bool(tests)),
            "failed": {resolved_summary_failed!r},
            "status": {summary_status!r},
            "readiness": {{"status": {readiness!r}}},
        }}
        (output_dir / "run_summary.json").write_text(
            json.dumps(run_summary), encoding="utf-8"
        )
        events = [{{
                "type": {event_type!r},
                "status": test["status"],
                "test": test["name"],
                "timestamp": "2026-08-12T00:00:00Z",
            }} for test in tests]
        (state_dir / "events.jsonl").write_text(
            "".join(json.dumps(item) + "\\n" for item in events),
            encoding="utf-8",
        )
        for test in tests:
            test_name = test["name"]
            test_dir = (
                output_dir
                / "results"
                / pathlib.Path(*pathlib.PurePosixPath(asset_key).parent.parts)
                / ".simready"
                / "runtime"
                / test_name
            )
            test_dir.mkdir(parents=True, exist_ok=True)
            result = dict(test)
            if test_name == {test_name!r} and {result_status!r} is not None:
                result["status"] = {result_status!r}
            result.update({{
                "asset": asset_key,
                "test_name": test_name,
                "test_version": test["version"],
            }})
            (test_dir / "result.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
            (test_dir / test["media"][0]["filename"]).write_bytes(b"fake-video")
        print("fake benchmark completed")
        print("fake benchmark diagnostic", file=sys.stderr)
        if {leave_descendant!r}:
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            (output_dir / "descendant.pid").write_text(str(child.pid), encoding="utf-8")
        if {mutate_asset!r}:
            asset_path.write_text(asset_path.read_text() + "\\n# changed\\n")
        mutate_runtime_input = {mutate_runtime_input!r}
        if mutate_runtime_input:
            runtime_input_flag = {{
                "sr_specs_path": "--sr-specs",
                "engines_toml_path": "--engines-toml",
                "project_config_path": "--project-config",
                "tests_paths[0]": "--tests-path",
            }}[mutate_runtime_input]
            runtime_input_path = pathlib.Path(flag_value(runtime_input_flag))
            if runtime_input_path.is_dir():
                (runtime_input_path / "changed-during-run.txt").write_text(
                    "changed\\n", encoding="utf-8"
                )
            else:
                runtime_input_path.write_text(
                    runtime_input_path.read_text(encoding="utf-8")
                    + "\\n# changed\\n",
                    encoding="utf-8",
                )
        time.sleep({sleep_before_exit_s!r})
        raise SystemExit({resolved_exit_code})
        """
    )
    executable.write_text(script, encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def _runtime_input(
    tmp_path: Path,
    executable: Path,
) -> SimReadyRuntimeValidationInput:
    asset = tmp_path / "asset.usda"
    _write_usda(asset)
    sr_specs = tmp_path / "simready_foundations" / "nv_core" / "sr_specs"
    sr_specs.mkdir(parents=True)
    engines_toml = tmp_path / "engines.toml"
    engines_toml.write_text("[kit.isaac_sim]\nversion = '5.1.0'\n", encoding="utf-8")
    tests_path = tmp_path / "tests"
    tests_path.mkdir()
    return SimReadyRuntimeValidationInput(
        asset_path=str(asset),
        output_dir=str(tmp_path / "output"),
        sr_specs_path=str(sr_specs),
        engines_toml_path=str(engines_toml),
        tests_paths=(str(tests_path),),
        features=("FET003",),
        runtimes=("isaac_sim",),
        benchmark_executable=str(executable),
    )


def _native_test_result_path(
    report: runtime_benchmark_module.SimReadyRuntimeValidationReport,
    *,
    check_index: int = 0,
) -> Path:
    check = report.checks[check_index]
    asset_parent = PurePosixPath(check.asset_key).parent
    return (
        Path(report.native_output_dir or "")
        / "results"
        / Path(*asset_parent.parts)
        / ".simready"
        / "runtime"
        / check.test_name
        / "result.json"
    )


def test_runtime_benchmark_preserves_native_pass_and_validation_projection(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    source_before = Path(params.asset_path).read_bytes()

    report = run_simready_runtime_validation(params)

    assert report.passed is True
    assert report.status == "pass"
    assert report.exit_code == 0
    assert report.native_report_schema_version == "5.0"
    assert report.benchmark_version == SIMREADY_BENCHMARK_VERSION
    assert report.checks[0].native_status == "pass"
    assert report.checks[0].disposition == "pass"
    assert report.checks[0].engine == "isaac_sim"
    assert report.checks[0].metrics == {"settled_after_s": 1.0}
    assert report.checks[0].asset_key.endswith("/asset.usda")
    assert "--output" in report.command
    assert "--no-stamp" in report.command
    assert Path(params.asset_path).read_bytes() == source_before
    assert Path(report.native_report_path or "").is_file()
    assert Path(report.native_plan_path or "").is_file()
    assert Path(report.report_path or "").is_file()
    assert Path(report.validation_template_result_path or "").is_file()
    assert Path(report.verified_operation_publication_path or "").is_file()
    assert set(report.artifact_sha256) == {
        "events",
        "native_plan",
        "native_report",
        "run_summary",
        "stderr",
        "stdout",
    }

    template_result = simready_runtime_validation_template_result(report)
    assert template_result.template_name == "physical_behavior"
    assert template_result.status == "passed"
    assert template_result.issues == ()
    assert template_result.metadata["native_dispositions_preserved"] is True
    assert template_result.evidence["simready_runtime_checks"][0]["native_status"] == (
        "pass"
    )
    publication = load_simready_runtime_verified_operations(
        report.verified_operation_publication_path or ""
    )
    assert publication.projector_benchmark_invoked is False
    assert publication.projector_simulator_invoked is False
    assert len(publication.operations) == 1
    operation = publication.operations[0]
    assert operation.result.native_status == "pass"
    assert operation.result.gate_id.startswith("simready.runtime-gate.")
    assert operation.result.source == operation.result.output
    assert operation.result.backend is not None
    assert operation.result.backend.component_id == "simready-backend-isaac-sim"
    assert operation.result.backend.version == "5.1.0"
    assert any(
        Path(binding.path).name == "fall.mp4" for binding in operation.result.artifacts
    )
    assert any(
        Path(binding.path).name == "isaac_sim_1.log"
        for binding in operation.result.artifacts
    )

    ingest_dir = tmp_path / "provided-validation"
    ingest_verified_operation_result(operation.envelope.path, output_dir=ingest_dir)
    evidence = collect_verified_operation_evidence(ingest_dir)
    assert len(evidence.records) == 1
    assert evidence.records[0].native_status == "pass"
    assert evidence.nested_agent_launched is False


def test_runtime_benchmark_cli_runs_the_same_adapter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)

    exit_code = runtime_benchmark_module.main(
        [
            params.asset_path,
            "--output-dir",
            params.output_dir,
            "--sr-specs",
            params.sr_specs_path,
            "--engines-toml",
            params.engines_toml_path,
            "--tests-path",
            params.tests_paths[0],
            "--feature",
            params.features[0],
            "--runtime",
            params.runtimes[0],
            "--benchmark-executable",
            str(executable),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "pass"
    assert payload["checks"][0]["native_status"] == "pass"


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--max-concurrent", "0", "must be at least 1"),
        ("--timeout", "0", "must be greater than 0"),
        ("--feature", "-x", "must not begin with '-'"),
    ],
)
def test_runtime_benchmark_cli_rejects_invalid_values_without_traceback(
    flag: str,
    value: str,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        runtime_benchmark_module.main(
            [
                "asset.usda",
                "--output-dir",
                "runtime-output",
                "--sr-specs",
                "sr-specs",
                "--engines-toml",
                "engines.toml",
                f"{flag}={value}" if value.startswith("-") else flag,
                *([] if value.startswith("-") else [value]),
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_runtime_benchmark_maps_asset_failure_without_collapsing_check(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path, status="fail")

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.passed is False
    assert report.status == "fail"
    assert report.exit_code == 1
    assert report.checks[0].native_status == "fail"
    assert report.checks[0].disposition == "fail"
    template_result = simready_runtime_validation_template_result(report)
    assert template_result.status == "failed"
    assert template_result.issues[0].code == ("physical_behavior.simready_runtime_fail")
    assert template_result.issues[0].details["metrics"] == {"settled_after_s": 1.0}


def test_runtime_projector_accepts_pinned_reporter_status_normalization(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(
        tmp_path,
        status="fail",
        result_status="error",
    )

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    publication = load_simready_runtime_verified_operations(
        report.verified_operation_publication_path or ""
    )
    operation = publication.operations[0]
    assert operation.result.native_status == "fail"
    raw_result_path = next(
        Path(binding.path)
        for binding in operation.result.artifacts
        if Path(binding.path).name == "result.json"
    )
    raw_result = json.loads(raw_result_path.read_text(encoding="utf-8"))
    assert raw_result["status"] == "error"


def test_runtime_benchmark_distinguishes_engine_crash_from_asset_failure(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(
        tmp_path,
        status="pass",
        event_type="engine_crash",
    )

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "error"
    assert report.passed is False
    assert report.checks[0].native_status == "pass"
    assert report.runtime_error_events[0]["type"] == "engine_crash"
    assert report.next_step == "inspect-simready-benchmark-runtime"
    template_result = simready_runtime_validation_template_result(report)
    assert any(
        issue.code == "physical_behavior.simready_runtime_error"
        and "engine_crash" in issue.message
        for issue in template_result.issues
    )
    assert report.verified_operation_publication_path is None


def test_runtime_projector_rederives_status_from_retained_artifacts(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(
        tmp_path,
        status="pass",
        event_type="engine_crash",
    )
    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))
    report_path = Path(report.report_path or "")
    retained = json.loads(report_path.read_text(encoding="utf-8"))
    retained.update(
        {
            "passed": True,
            "status": "pass",
            "runtime_error_events": [],
            "next_step": "complete",
        }
    )
    report_path.write_text(json.dumps(retained), encoding="utf-8")

    with pytest.raises(
        VerifiedOperationError,
        match="status differs from the retained native evidence",
    ):
        project_simready_runtime_verified_operations(
            report_path,
            output_dir=tmp_path / "tampered-status-projection",
        )


def test_runtime_benchmark_maps_not_ready_to_blocked(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(
        tmp_path,
        status="blocked",
        exit_code=4,
        readiness="not_ready",
    )

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "blocked"
    assert report.exit_code == 4
    assert report.checks[0].native_status == "blocked"
    assert report.checks[0].disposition == "blocked"
    assert report.verified_operation_publication_path is None
    assert simready_runtime_validation_template_result(report).status == "error"


def test_runtime_benchmark_rejects_zero_test_plan(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path, total_work_items=0)

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "error"
    assert report.checks == []
    assert any("planned no work items" in error for error in report.errors)


def test_runtime_benchmark_rejects_failed_orchestration_summary(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path, summary_failed=3)

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "error"
    assert report.passed is False
    assert any("failed orchestration" in error for error in report.errors)


def test_runtime_benchmark_rejects_incomplete_work_summary(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(
        tmp_path,
        summary_total_work_items=2,
        summary_completed=1,
    )

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "error"
    assert report.passed is False
    assert report.verified_operation_publication_path is None
    assert any(
        "did not complete every planned work item" in error for error in report.errors
    )


def test_runtime_benchmark_rejects_nonterminal_work_summary(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path, summary_status="cancelled")

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "error"
    assert report.passed is False
    assert report.verified_operation_publication_path is None
    assert any("status is not completed" in error for error in report.errors)


def test_runtime_benchmark_rejects_report_for_different_asset(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(
        tmp_path,
        reported_asset_key="stale/other-asset.usda",
    )

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "error"
    assert report.passed is False
    assert any("different asset than requested" in error for error in report.errors)


def test_runtime_projector_accepts_truncated_test_name_separator(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(
        tmp_path,
        test_name=f"{'a' * 47}_long_runtime_test_name",
    )

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "pass"
    publication = load_simready_runtime_verified_operations(
        report.verified_operation_publication_path or ""
    )
    assert "--" not in publication.operations[0].result.operation_id


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group behavior")
def test_runtime_benchmark_reaps_descendants_after_leader_exit(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path, leave_descendant=True)

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    descendant_pid_path = Path(report.native_output_dir or "") / "descendant.pid"
    assert descendant_pid_path.is_file()
    descendant_pid = int(descendant_pid_path.read_text())
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                os.kill(descendant_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"benchmark descendant {descendant_pid} remained alive")
    finally:
        try:
            os.kill(descendant_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_runtime_benchmark_terminates_process_group_on_timeout(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path, sleep_before_exit_s=60.0)
    params = _runtime_input(tmp_path, executable).model_copy(update={"timeout_s": 0.1})

    report = run_simready_runtime_validation(params)

    assert report.status == "error"
    assert report.exit_code is not None
    assert any("timed out after 0.1 seconds" in error for error in report.errors)


def test_runtime_benchmark_rejects_report_schema_drift(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path, schema_version="6.0")

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "error"
    assert report.verified_operation_publication_path is None
    assert any(
        "Unsupported native benchmark report schema" in error for error in report.errors
    )
    with pytest.raises(VerifiedOperationError, match="schema differs"):
        project_simready_runtime_verified_operations(
            report.report_path or "",
            output_dir=tmp_path / "unsupported-schema-projection",
        )


def test_runtime_projector_rejects_native_framework_drift(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(
        tmp_path,
        native_framework_version="2026.6.4",
    )

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "error"
    assert report.verified_operation_publication_path is None
    assert any("framework version differs" in error for error in report.errors)
    with pytest.raises(VerifiedOperationError, match="framework differs"):
        project_simready_runtime_verified_operations(
            report.report_path or "",
            output_dir=tmp_path / "unsupported-framework-projection",
        )


def test_runtime_benchmark_rejects_static_validation_gate(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(
        tmp_path,
        status="skipped",
        exit_code=0,
        feature_status="validation_failed",
    )

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "blocked"
    assert report.checks[0].disposition == "not_evaluated"
    assert report.verified_operation_publication_path is None
    assert report.next_step == "simready-conform-profile"
    assert any("Static validation prevented" in error for error in report.errors)


def test_runtime_benchmark_detects_source_mutation(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path, mutate_asset=True)

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "error"
    assert any("changed" in error.lower() for error in report.errors)


@pytest.mark.parametrize(
    "input_label",
    (
        "sr_specs_path",
        "engines_toml_path",
        "project_config_path",
        "tests_paths[0]",
    ),
)
def test_runtime_benchmark_detects_runtime_input_mutation(
    tmp_path: Path,
    input_label: str,
) -> None:
    executable = _write_fake_benchmark(
        tmp_path,
        mutate_runtime_input=input_label,
    )
    params = _runtime_input(tmp_path, executable)
    Path(params.sr_specs_path, "profile.yaml").write_text(
        "profile: robotics\n", encoding="utf-8"
    )
    Path(params.tests_paths[0], "test.yaml").write_text(
        "test: rigid-body\n", encoding="utf-8"
    )
    project_config = tmp_path / "project.toml"
    project_config.write_text("[project]\nname = 'runtime'\n", encoding="utf-8")
    params = params.model_copy(update={"project_config_path": str(project_config)})

    report = run_simready_runtime_validation(params)

    assert report.status == "error"
    assert report.verified_operation_publication_path is None
    assert any(input_label in error and "changed" in error for error in report.errors)


def test_runtime_publication_binds_every_runtime_input(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    spec = Path(params.sr_specs_path, "profile.yaml")
    spec.write_text("profile: robotics\n", encoding="utf-8")
    custom_test = Path(params.tests_paths[0], "test.yaml")
    custom_test.write_text("test: rigid-body\n", encoding="utf-8")
    project_config = tmp_path / "project.toml"
    project_config.write_text("[project]\nname = 'runtime'\n", encoding="utf-8")
    params = params.model_copy(update={"project_config_path": str(project_config)})

    report = run_simready_runtime_validation(params)
    publication = load_simready_runtime_verified_operations(
        report.verified_operation_publication_path or ""
    )

    assert report.status == "pass"
    assert {binding.label for binding in report.runtime_input_bindings} == {
        "sr_specs_path",
        "engines_toml_path",
        "project_config_path",
        "tests_paths[0]",
    }
    published_artifacts = {
        binding.path for binding in publication.operations[0].result.artifacts
    }
    for runtime_input in report.runtime_input_bindings:
        assert {binding.path for binding in runtime_input.files} <= published_artifacts
    assert any(
        Path(path).name == "runtime_input_bindings.json" for path in published_artifacts
    )
    assert any(
        Path(path).name == ".publication-valid.json" for path in published_artifacts
    )

    Path(params.engines_toml_path).write_text(
        "[kit.isaac_sim]\nversion = 'changed'\n", encoding="utf-8"
    )
    with pytest.raises(VerifiedOperationError, match="changed"):
        load_simready_runtime_verified_operations(
            report.verified_operation_publication_path or ""
        )


def test_runtime_benchmark_enforces_pinned_framework_version(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path, framework_version="2026.6.4")

    runtime = resolve_simready_benchmark_runtime(executable)

    assert runtime.passed is False
    assert runtime.benchmark_version == "2026.6.4"
    assert any(
        "Unsupported simready-benchmark version" in error for error in runtime.errors
    )


def test_runtime_adapter_matches_recorded_pinned_benchmark_contract(
    tmp_path: Path,
) -> None:
    contract_path = (
        Path(__file__).with_name("fixtures")
        / "simready_benchmark_2026_6_5_contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    distribution = contract["distribution"]
    native_report_contract = contract["native_report"]
    assert distribution["version"] == SIMREADY_BENCHMARK_VERSION
    assert set(distribution["public_releases"]) == {
        "simready-benchmark",
        "simready-benchmark-engine-kit",
    }
    assert all(
        release["url"].startswith("https://pypi.org/project/")
        and len(release["wheel_sha256"]) == 64
        for release in distribution["public_releases"].values()
    )
    assert {
        "simready_benchmark_engine_kit/execution.py",
        "simready_benchmark_engine_kit/kit_runner.py",
    } <= set(distribution["capture_sources_sha256"])
    assert (
        runtime_benchmark_module._VERSION_PATTERN.fullmatch(
            distribution["version_banner"]
        ).group(1)
        == SIMREADY_BENCHMARK_VERSION
    )
    assert (
        native_report_contract["schema_version"]
        == SIMREADY_BENCHMARK_NATIVE_REPORT_SCHEMA_VERSION
    )
    assert set(native_report_contract["test_statuses"]) == (
        runtime_benchmark_module._NATIVE_TEST_STATUSES
    )
    assert set(native_report_contract["per_test_result_statuses"]) == set(
        native_report_contract["test_statuses"]
    )
    assert set(native_report_contract["plan_only_feature_statuses"]) == (
        runtime_benchmark_module._NATIVE_PLAN_ONLY_FEATURE_STATUSES
    )
    assert contract["stamping"] == {
        "default_enabled": True,
        "source_directory_written": False,
        "target": "results/<asset-mirror>",
    }

    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    second_tests_path = tmp_path / "second-tests"
    second_tests_path.mkdir()
    project_config = tmp_path / "project.toml"
    project_config.write_text("[project]\nname = 'contract-test'\n", encoding="utf-8")
    params = params.model_copy(
        update={
            "tests_paths": (*params.tests_paths, str(second_tests_path)),
            "features": ("FET003", "FET004"),
            "tests": ("rigid_body_falls_isaac", "rigid_body_stacks_isaac"),
            "runtimes": ("isaac_sim", "newton"),
            "project_config_path": str(project_config),
            "max_concurrent": 2,
        }
    )
    runtime = runtime_benchmark_module.SimReadyBenchmarkRuntimeInfo(
        executable=str(executable),
        benchmark_version=SIMREADY_BENCHMARK_VERSION,
    )
    native_output_dir = tmp_path / "native-output"
    command = runtime_benchmark_module.build_simready_benchmark_command(
        params=params,
        runtime=runtime,
        asset_path=Path(params.asset_path).resolve(),
        native_output_dir=native_output_dir,
    )

    def flag_values(flag: str) -> list[str]:
        start = command.index(flag) + 1
        end = next(
            (
                index
                for index in range(start, len(command))
                if command[index].startswith("--")
            ),
            len(command),
        )
        return command[start:end]

    expected_values = {
        "--assets": [str(Path(params.asset_path).resolve())],
        "--tests-path": [str(Path(path).resolve()) for path in params.tests_paths],
        "--features": list(params.features),
        "--tests": list(params.tests),
        "--runtime": list(params.runtimes),
    }
    assert contract["cli"]["multi_value_flags"] == list(expected_values)
    for flag, values in expected_values.items():
        assert flag_values(flag) == values
    expected_single_values = {
        "--output": [str(native_output_dir / "plan.json")],
        "--output-dir": [str(native_output_dir)],
        "--sr-specs": [str(Path(params.sr_specs_path).resolve())],
        "--engines-toml": [str(Path(params.engines_toml_path).resolve())],
        "--project-config": [str(project_config.resolve())],
        "--max-concurrent": ["2"],
        "--format": ["json"],
    }
    assert contract["cli"]["single_value_flags"] == list(expected_single_values)
    for flag, values in expected_single_values.items():
        assert flag_values(flag) == values
    no_stamp_flag = contract["cli"]["disable_stamping_flag"]
    assert no_stamp_flag in command
    stamped_command = runtime_benchmark_module.build_simready_benchmark_command(
        params=params.model_copy(update={"stamp_results": True}),
        runtime=runtime,
        asset_path=Path(params.asset_path).resolve(),
        native_output_dir=tmp_path / "stamped-native-output",
    )
    assert no_stamp_flag not in stamped_command
    assert contract["runtime_exit_codes"] == {
        "success": 0,
        "test_failure": 1,
        "not_ready": 4,
    }
    report = run_simready_runtime_validation(params)
    native_root = Path(report.native_output_dir or "")
    native_plan_path = Path(report.native_plan_path or "")
    assert (
        native_plan_path.relative_to(native_root).as_posix()
        == (native_report_contract["plan_path"])
    )
    native_plan = json.loads(native_plan_path.read_text(encoding="utf-8"))
    assert native_report_contract["required_plan_identity_fields"] == [
        "assets[0].path",
        "content_root",
    ]
    assert isinstance(native_plan["assets"][0]["path"], str)
    assert isinstance(native_plan["content_root"], str)
    assert (
        Path(report.native_report_path or "").relative_to(native_root).as_posix()
        == native_report_contract["report_path"]
    )
    assert (
        Path(report.run_summary_path or "").relative_to(native_root).as_posix()
        == native_report_contract["run_summary_path"]
    )
    assert (
        Path(report.events_path or "").relative_to(native_root).as_posix()
        == native_report_contract["events_path"]
    )
    native_report = json.loads(
        Path(report.native_report_path or "").read_text(encoding="utf-8")
    )
    session_log_file = native_report["metadata"]["sessions"][0]["log_file"]
    expected_session_log_file = native_report_contract["session_log_file_path"].replace(
        "<engine-config>", report.checks[0].engine or ""
    )
    expected_session_log_file = expected_session_log_file.replace("<spawn-number>", "1")
    assert session_log_file == expected_session_log_file
    assert (native_root / Path(*PurePosixPath(session_log_file).parts)).is_file()
    result_relative = (
        _native_test_result_path(report).relative_to(native_root).as_posix()
    )
    asset_parent = PurePosixPath(report.checks[0].asset_key).parent.as_posix()
    asset_prefix = f"results/{asset_parent}/" if asset_parent != "." else "results/"
    normalized_result_path = result_relative.replace(
        asset_prefix,
        "results/<asset-dir>/",
        1,
    ).replace(
        f"/{report.checks[0].test_name}/",
        "/<test-name>/",
        1,
    )
    assert normalized_result_path == native_report_contract["per_test_result_path"]
    media_filename = report.checks[0].media[0]["filename"]
    media_path = _native_test_result_path(report).parent / Path(
        *PurePosixPath(media_filename).parts
    )
    media_relative = media_path.relative_to(native_root).as_posix()
    normalized_media_path = media_relative.replace(
        asset_prefix,
        "results/<asset-dir>/",
        1,
    ).replace(
        f"/{report.checks[0].test_name}/",
        "/<test-name>/",
        1,
    )
    normalized_media_path = normalized_media_path.removesuffix(media_filename)
    normalized_media_path += "<media-filename>"
    assert media_path.is_file()
    assert normalized_media_path == native_report_contract["per_test_media_path"]
    for example in native_report_contract["rollup_examples"]:
        assert (
            runtime_benchmark_module._native_aggregate_status(example["statuses"])
            == example["derived"]
        )
    for native_status, normalized in native_report_contract[
        "test_status_normalization"
    ].items():
        assert (
            runtime_verified_operations_module._aggregate_native_status(native_status)
            == normalized
        )


@pytest.mark.parametrize(
    ("benchmark_kwargs", "aggregate_level"),
    [
        ({"feature_status": "fail"}, "feature"),
        ({"profile_status": "fail"}, "profile"),
    ],
)
def test_runtime_benchmark_rejects_inconsistent_native_aggregate_status(
    tmp_path: Path,
    benchmark_kwargs: dict[str, str],
    aggregate_level: str,
) -> None:
    executable = _write_fake_benchmark(tmp_path, **benchmark_kwargs)

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "error"
    assert report.verified_operation_publication_path is None
    assert any(
        f"Native {aggregate_level} aggregate status is inconsistent" in error
        for error in report.errors
    )


def test_runtime_benchmark_isolates_python_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    monkeypatch.setenv("PYTHONHOME", "/caller/python-home")
    monkeypatch.setenv("PYTHONPATH", "/caller/python-path")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    environment = runtime_benchmark_module._benchmark_subprocess_environment(executable)

    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert "VIRTUAL_ENV" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_runtime_benchmark_blocks_missing_configuration(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable).model_copy(
        update={"engines_toml_path": str(tmp_path / "missing-engines.toml")}
    )

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert report.command == []
    assert any("engines.toml does not exist" in error for error in report.errors)


@pytest.mark.parametrize(
    ("target_name", "generated_label"),
    [
        ("runtime-report.json", "generated runtime report"),
        ("validation-template-result.json", "generated validation template result"),
    ],
)
def test_runtime_benchmark_resolves_dependencies_before_blocked_writes(
    tmp_path: Path,
    target_name: str,
    generated_label: str,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    dependency = Path(params.output_dir) / target_name
    _write_usda(dependency)
    dependency_before = dependency.read_bytes()
    relative_dependency = dependency.relative_to(Path(params.asset_path).parent)
    Path(params.asset_path).write_text(
        textwrap.dedent(
            f"""\
            #usda 1.0

            def Xform "World" (
                references = @{relative_dependency.as_posix()}@</World>
            )
            {{
            }}
            """
        ),
        encoding="utf-8",
    )
    updates = {
        "engines_toml_path": str(tmp_path / "missing-engines.toml"),
    }
    if target_name == "runtime-report.json":
        updates["report_path"] = str(dependency)
    params = params.model_copy(update=updates)

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert report.command == []
    assert dependency.read_bytes() == dependency_before
    assert Path(report.report_path or "").is_file()
    assert report.report_path != str(dependency)
    if target_name == "validation-template-result.json":
        assert report.validation_template_result_path is None
    assert any("engines.toml does not exist" in error for error in report.errors)
    assert any(generated_label in error for error in report.errors)


@pytest.mark.parametrize("input_name", ["sr_specs_path", "tests_paths"])
@pytest.mark.parametrize("nested", [False, True])
def test_runtime_benchmark_rejects_protected_output_tree_before_writing(
    tmp_path: Path,
    input_name: str,
    nested: bool,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    input_root = (
        Path(params.sr_specs_path)
        if input_name == "sr_specs_path"
        else Path(params.tests_paths[0])
    )
    output_dir = input_root / "runtime-output" if nested else input_root
    sentinel = input_root / "simready-runtime-validation.json"
    sentinel.write_text("preserve-input\n", encoding="utf-8")
    prior_publication = output_dir / SIMREADY_RUNTIME_VERIFIED_OPERATIONS_DIRNAME
    if not nested:
        prior_publication.mkdir()
        (prior_publication / "existing.json").write_text(
            "preserve-publication\n", encoding="utf-8"
        )
    params = params.model_copy(update={"output_dir": str(output_dir)})

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert report.report_path is None
    assert sentinel.read_text(encoding="utf-8") == "preserve-input\n"
    if nested:
        assert not output_dir.exists()
    else:
        assert (prior_publication / "existing.json").read_text(
            encoding="utf-8"
        ) == "preserve-publication\n"
        assert not (
            prior_publication
            / runtime_benchmark_module.SIMREADY_RUNTIME_PUBLICATION_INVALIDATED_NAME
        ).exists()
    assert any(f"protected input {input_name}" in error for error in report.errors)


def test_runtime_benchmark_does_not_create_missing_protected_input_tree(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    missing_tests = tmp_path / "missing-tests"
    params = params.model_copy(
        update={
            "tests_paths": (str(missing_tests),),
            "output_dir": str(missing_tests / "runtime-output"),
        }
    )

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert report.report_path is None
    assert not missing_tests.exists()
    assert any("protected input tests_paths[0]" in error for error in report.errors)


def test_runtime_report_fallback_never_returns_forbidden_default(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"

    report_path = runtime_benchmark_module._fallback_runtime_report_path(
        output_dir,
        forbidden_paths={output_dir},
    )

    assert report_path is None
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("target_name", "expected_error"),
    [
        ("custom-report", "Generated runtime report path is a directory"),
        (
            "validation-template-result.json",
            "Generated validation template result path is a directory",
        ),
    ],
)
def test_runtime_benchmark_rejects_directory_generated_file_targets(
    tmp_path: Path,
    target_name: str,
    expected_error: str,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    target = Path(params.output_dir) / target_name
    target.mkdir(parents=True)
    sentinel = target / "preserve.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    if target_name == "custom-report":
        params = params.model_copy(update={"report_path": str(target)})

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert report.command == []
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"
    assert Path(report.report_path or "").is_file()
    assert report.report_path != str(target)
    if target_name == "validation-template-result.json":
        assert report.validation_template_result_path is None
    assert any(expected_error in error for error in report.errors)


def test_runtime_benchmark_rejects_non_directory_report_ancestor(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    non_directory_ancestor = Path(params.output_dir) / "existing-file"
    non_directory_ancestor.parent.mkdir(parents=True)
    non_directory_ancestor.write_text("preserve\n", encoding="utf-8")
    requested_report = non_directory_ancestor / "report.json"
    params = params.model_copy(update={"report_path": str(requested_report)})

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert report.command == []
    assert non_directory_ancestor.read_text(encoding="utf-8") == "preserve\n"
    assert report.report_path != str(requested_report)
    assert Path(report.report_path or "").is_file()
    assert any("non-directory ancestor" in error for error in report.errors)


@pytest.mark.parametrize(
    ("malformed_native_field", "expected_error"),
    [
        ("media", "Native test media are malformed"),
        ("engine", "Native test engine is malformed"),
    ],
)
def test_runtime_benchmark_persists_malformed_native_fields_as_errors(
    tmp_path: Path,
    malformed_native_field: str,
    expected_error: str,
) -> None:
    executable = _write_fake_benchmark(
        tmp_path,
        malformed_native_field=malformed_native_field,
    )

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "error"
    assert report.passed is False
    assert report.verified_operation_publication_path is None
    assert Path(report.report_path or "").is_file()
    assert any(expected_error in error for error in report.errors)


@pytest.mark.parametrize(
    ("malformed_native_field", "expected_error"),
    [
        ("profile_id", "Native profile id is missing or malformed"),
        ("profile_version", "Native profile version is missing or malformed"),
        ("feature_id", "Native feature id is missing or malformed"),
        ("feature_version", "Native feature version is missing or malformed"),
        ("test_name", "Native test name is missing or malformed"),
        ("test_version", "Native test version is missing or malformed"),
    ],
)
def test_runtime_benchmark_rejects_malformed_native_identity_fields(
    tmp_path: Path,
    malformed_native_field: str,
    expected_error: str,
) -> None:
    executable = _write_fake_benchmark(
        tmp_path,
        malformed_native_field=malformed_native_field,
    )

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "error"
    assert report.passed is False
    assert report.checks == []
    assert report.verified_operation_publication_path is None
    assert any(expected_error in error for error in report.errors)


def test_runtime_benchmark_rejects_option_like_filter_values(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    payload = params.model_dump()
    payload["runtimes"] = ("--workers",)

    with pytest.raises(ValueError, match="must not begin"):
        SimReadyRuntimeValidationInput.model_validate(payload)


def test_runtime_benchmark_never_writes_report_over_source_asset(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    source_path = Path(params.asset_path)
    source_before = source_path.read_bytes()
    params = params.model_copy(update={"report_path": str(source_path)})

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert source_path.read_bytes() == source_before
    assert report.report_path != str(source_path)
    assert Path(report.report_path or "").is_file()
    assert any("Runtime report path must be" in error for error in report.errors)


def test_runtime_benchmark_preserves_input_at_fixed_log_path(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    engines_toml = Path(params.output_dir) / "simready-benchmark.stdout.log"
    engines_toml.parent.mkdir(parents=True, exist_ok=True)
    engines_toml.write_text("[kit.isaac_sim]\nversion = '5.1.0'\n", encoding="utf-8")
    before = engines_toml.read_bytes()
    params = params.model_copy(update={"engines_toml_path": str(engines_toml)})

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert report.command == []
    assert engines_toml.read_bytes() == before
    assert any("benchmark stdout log" in error for error in report.errors)


def test_runtime_benchmark_persists_report_when_template_path_is_input(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    engines_toml = Path(params.output_dir) / "validation-template-result.json"
    engines_toml.parent.mkdir(parents=True, exist_ok=True)
    engines_toml.write_text("[kit.isaac_sim]\nversion = '5.1.0'\n", encoding="utf-8")
    before = engines_toml.read_bytes()
    params = params.model_copy(update={"engines_toml_path": str(engines_toml)})

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert report.validation_template_result_path is None
    assert Path(report.report_path or "").is_file()
    assert engines_toml.read_bytes() == before
    assert any("validation template result" in error for error in report.errors)


def test_runtime_benchmark_falls_back_when_report_collides_with_input(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    engines_toml = Path(params.engines_toml_path)
    before = engines_toml.read_bytes()
    params = params.model_copy(update={"report_path": str(engines_toml)})

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert report.report_path != str(engines_toml)
    assert engines_toml.read_bytes() == before
    assert any("Runtime report path must be" in error for error in report.errors)


def test_runtime_benchmark_does_not_overwrite_invalid_output_file(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    source_path = Path(params.asset_path)
    source_before = source_path.read_bytes()
    params = params.model_copy(update={"output_dir": str(source_path)})

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert report.report_path is None
    assert source_path.read_bytes() == source_before
    assert any("protected input asset_path" in error for error in report.errors)


def test_runtime_benchmark_keeps_inputs_outside_managed_output(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    nested_asset = Path(params.output_dir) / "simready-benchmark" / "asset.usda"
    _write_usda(nested_asset)
    params = params.model_copy(update={"asset_path": str(nested_asset)})

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert any("managed benchmark output" in error for error in report.errors)


def test_runtime_benchmark_does_not_delete_managed_projection_input(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    nested_executable = (
        Path(params.output_dir)
        / SIMREADY_RUNTIME_VERIFIED_OPERATIONS_DIRNAME
        / "simready-benchmark"
    )
    nested_executable.parent.mkdir(parents=True)
    nested_executable.write_bytes(executable.read_bytes())
    nested_executable.chmod(executable.stat().st_mode)
    executable_before = nested_executable.read_bytes()
    params = params.model_copy(update={"benchmark_executable": str(nested_executable)})

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert nested_executable.read_bytes() == executable_before
    assert any("managed verified-operation" in error for error in report.errors)


def test_runtime_benchmark_does_not_delete_managed_projection_dependency(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    dependency = (
        Path(params.output_dir)
        / SIMREADY_RUNTIME_VERIFIED_OPERATIONS_DIRNAME
        / "dependency.usda"
    )
    _write_usda(dependency)
    dependency_before = dependency.read_bytes()
    relative_dependency = dependency.relative_to(Path(params.asset_path).parent)
    Path(params.asset_path).write_text(
        textwrap.dedent(
            f"""\
            #usda 1.0

            def Xform "World" (
                references = @{relative_dependency.as_posix()}@</World>
            )
            {{
            }}
            """
        ),
        encoding="utf-8",
    )

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert dependency.read_bytes() == dependency_before
    assert any("Asset dependency must not be" in error for error in report.errors)


def test_runtime_benchmark_never_writes_report_over_asset_dependency(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    dependency = Path(params.output_dir) / "dependency.usda"
    _write_usda(dependency)
    dependency_before = dependency.read_bytes()
    relative_dependency = dependency.relative_to(Path(params.asset_path).parent)
    Path(params.asset_path).write_text(
        textwrap.dedent(
            f"""\
            #usda 1.0

            def Xform "World" (
                references = @{relative_dependency.as_posix()}@</World>
            )
            {{
            }}
            """
        ),
        encoding="utf-8",
    )
    params = params.model_copy(update={"report_path": str(dependency)})

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert report.report_path != str(dependency)
    assert dependency.read_bytes() == dependency_before
    assert any("generated runtime report" in error for error in report.errors)


def test_runtime_benchmark_persists_report_when_template_path_is_dependency(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    dependency = Path(params.output_dir) / "validation-template-result.json"
    _write_usda(dependency)
    dependency_before = dependency.read_bytes()
    relative_dependency = dependency.relative_to(Path(params.asset_path).parent)
    Path(params.asset_path).write_text(
        textwrap.dedent(
            f"""\
            #usda 1.0

            def Xform "World" (
                references = @{relative_dependency.as_posix()}@</World>
            )
            {{
            }}
            """
        ),
        encoding="utf-8",
    )

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert report.validation_template_result_path is None
    assert Path(report.report_path or "").is_file()
    assert dependency.read_bytes() == dependency_before
    assert any("validation template result" in error for error in report.errors)


def test_runtime_benchmark_invalidates_publication_before_input_failure(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    initial = _runtime_input(tmp_path, executable).model_copy(
        update={"report_path": str(tmp_path / "output" / "old-report.json")}
    )
    passed = run_simready_runtime_validation(initial)
    old_publication_path = Path(passed.verified_operation_publication_path or "")
    assert old_publication_path.is_file()
    old_publication = load_simready_runtime_verified_operations(old_publication_path)
    copied_envelope_path = tmp_path / "copied-old-envelope.json"
    copied_envelope_path.write_bytes(
        Path(old_publication.operations[0].envelope.path).read_bytes()
    )

    blocked = initial.model_copy(
        update={
            "engines_toml_path": str(tmp_path / "missing-engines.toml"),
            "report_path": str(tmp_path / "output" / "new-report.json"),
        }
    )
    blocked_report = run_simready_runtime_validation(blocked)

    assert blocked_report.status == "blocked"
    with pytest.raises(VerifiedOperationError, match="invalidated by a newer run"):
        load_simready_runtime_verified_operations(old_publication_path)
    with pytest.raises(VerifiedOperationError, match="stale"):
        ingest_verified_operation_result(
            copied_envelope_path,
            output_dir=tmp_path / "revoked-envelope-ingest",
        )


@pytest.mark.parametrize("asset_failure", ["missing", "unresolved-dependency"])
def test_runtime_benchmark_invalidates_publication_before_asset_identity_failure(
    tmp_path: Path,
    asset_failure: str,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    initial = _runtime_input(tmp_path, executable).model_copy(
        update={"report_path": str(tmp_path / "output" / "old-report.json")}
    )
    passed = run_simready_runtime_validation(initial)
    old_publication_path = Path(passed.verified_operation_publication_path or "")
    assert old_publication_path.is_file()

    if asset_failure == "missing":
        failed_asset = tmp_path / "missing.usda"
    else:
        failed_asset = tmp_path / "unresolved.usda"
        failed_asset.write_text(
            textwrap.dedent(
                """\
                #usda 1.0

                def Xform "World" (
                    references = @missing-dependency.usda@</World>
                )
                {
                }
                """
            ),
            encoding="utf-8",
        )
    blocked = initial.model_copy(update={"asset_path": str(failed_asset)})

    blocked_report = run_simready_runtime_validation(blocked)

    assert blocked_report.status == "blocked"
    assert blocked_report.report_path is not None
    assert (
        json.loads(Path(blocked_report.report_path).read_text(encoding="utf-8"))[
            "status"
        ]
        == "blocked"
    )
    assert blocked_report.validation_template_result_path is not None
    assert (
        json.loads(
            Path(blocked_report.validation_template_result_path).read_text(
                encoding="utf-8"
            )
        )["status"]
        == "error"
    )
    with pytest.raises(VerifiedOperationError, match="invalidated by a newer run"):
        load_simready_runtime_verified_operations(old_publication_path)


def test_runtime_benchmark_does_not_write_an_unresolved_dependency_as_report(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    params = _runtime_input(tmp_path, executable)
    default_report = Path(params.output_dir) / "simready-runtime-validation.json"
    Path(params.asset_path).write_text(
        textwrap.dedent(
            """\
            #usda 1.0

            def Xform "World" (
                references = @output/simready-runtime-validation.json@</World>
            )
            {
            }
            """
        ),
        encoding="utf-8",
    )

    report = run_simready_runtime_validation(params)

    assert report.status == "blocked"
    assert report.report_path is not None
    assert Path(report.report_path) != default_report
    assert Path(report.report_path).is_file()
    assert not default_report.exists()
    assert any(
        "Asset dependency must not collide with generated runtime report" in error
        for error in report.errors
    )


@pytest.mark.parametrize(
    ("native_status", "expected_status"),
    [
        ("skipped", "not_evaluated"),
        ("incomplete", "error"),
    ],
)
def test_runtime_benchmark_preserves_nonterminal_native_dispositions(
    tmp_path: Path,
    native_status: str,
    expected_status: str,
) -> None:
    executable = _write_fake_benchmark(tmp_path, status=native_status, exit_code=0)

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == expected_status
    assert report.checks[0].native_status == native_status
    assert report.checks[0].disposition == expected_status
    if expected_status == "error":
        assert report.verified_operation_publication_path is None
        return
    publication = load_simready_runtime_verified_operations(
        report.verified_operation_publication_path or ""
    )
    assert publication.operations[0].result.native_status == expected_status
    payload = json.loads(
        Path(publication.operations[0].payload.path).read_text(encoding="utf-8")
    )
    assert payload["check"]["native_status"] == native_status


def test_runtime_projector_keeps_sibling_checks_independent_through_ingress(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path, additional_status="skipped")

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "warn"
    publication = load_simready_runtime_verified_operations(
        report.verified_operation_publication_path or ""
    )
    assert publication.projector_provider_invoked is False
    assert publication.projector_nested_agent_launched is False
    assert len(publication.operations) == 2
    assert len({item.result.operation_id for item in publication.operations}) == 2
    assert len({item.result.gate_id for item in publication.operations}) == 2

    status_by_test = {}
    ingest_dir = tmp_path / "provided-mixed-validation"
    for item in publication.operations:
        payload = json.loads(Path(item.payload.path).read_text(encoding="utf-8"))
        status_by_test[payload["check"]["test_name"]] = item.result.native_status
        ingest_verified_operation_result(item.envelope.path, output_dir=ingest_dir)

    assert status_by_test == {
        "rigid_body_falls_isaac": "pass",
        "rigid_body_stacks_isaac": "not_evaluated",
    }
    evidence = collect_verified_operation_evidence(ingest_dir)
    assert {record.native_status for record in evidence.records} == {
        "pass",
        "not_evaluated",
    }


def test_runtime_publication_hashes_shared_source_once_per_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _write_fake_benchmark(tmp_path, additional_status="skipped")
    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))
    source = Path(report.asset_path)
    source_hashes = 0
    original_verify = (
        verified_operations_module._verify_execution_artifact_binding_streaming
    )

    def tracking_verify(binding, *, label: str):
        nonlocal source_hashes
        if Path(binding.path) == source:
            source_hashes += 1
        return original_verify(binding, label=label)

    monkeypatch.setattr(
        verified_operations_module,
        "_verify_execution_artifact_binding_streaming",
        tracking_verify,
    )

    publication = load_simready_runtime_verified_operations(
        report.verified_operation_publication_path or ""
    )

    assert len(publication.operations) == 2
    assert all(
        operation.result.source.path == str(source)
        for operation in publication.operations
    )
    assert source_hashes == 1


def test_runtime_operation_identity_distinguishes_engine_contracts(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))
    base_check = report.checks[0]
    checks = [
        base_check,
        base_check.model_copy(update={"engine": "newton", "engine_version": "1.0.0"}),
        base_check.model_copy(update={"engine_version": "5.2.0"}),
    ]

    identities = {
        runtime_verified_operations_module._operation_identity(check)
        for check in checks
    }

    assert len(identities) == 3


def test_runtime_benchmark_rejects_duplicate_native_asset_test_storage_identity(
    tmp_path: Path,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))
    native_report = json.loads(
        Path(report.native_report_path or "").read_text(encoding="utf-8")
    )
    native_plan = json.loads(
        Path(report.native_plan_path or "").read_text(encoding="utf-8")
    )
    asset_payload = next(iter(native_report["assets"].values()))
    tests = asset_payload["profiles"][0]["features"][0]["tests"]
    second_engine = dict(tests[0])
    second_engine.update({"engine": "newton", "engine_version": "1.0.0"})
    tests.append(second_engine)

    checks, errors, _validation_failed = (
        runtime_benchmark_module.native_benchmark_checks(
            native_report,
            expected_asset_path=report.asset_path,
            native_plan=native_plan,
        )
    )

    assert len(checks) == 2
    assert (
        len(
            {
                runtime_verified_operations_module._operation_identity(check)
                for check in checks
            }
        )
        == 2
    )
    assert any("duplicate per-test result identity" in error for error in errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration", 9.5),
        ("message", "different result message"),
        ("metrics", {"settled_after_s": 99.0}),
        ("media", []),
        ("kit_logs", [{"path": "different.log"}]),
        ("description", "different description"),
        ("expected_video", "different expected video"),
        ("scene_file", "different-scene.usda"),
    ],
)
def test_runtime_projector_rejects_shared_per_test_field_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))
    result_path = _native_test_result_path(report)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result[field] = value
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(
        VerifiedOperationError,
        match="per-test result differs from the aggregate benchmark report",
    ):
        project_simready_runtime_verified_operations(
            report.report_path or "",
            output_dir=tmp_path / f"drift-{field}",
        )


def test_runtime_projector_reuses_retained_results_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _write_fake_benchmark(tmp_path)
    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    def unexpected_execution(*_args, **_kwargs):
        raise AssertionError("projector must not launch SimReady Benchmark")

    monkeypatch.setattr(
        runtime_benchmark_module,
        "_run_bounded_benchmark_process",
        unexpected_execution,
    )
    publication = project_simready_runtime_verified_operations(
        report.report_path or "",
        output_dir=tmp_path / "reprojected",
    )

    assert publication.operations[0].result.native_status == "pass"
    assert publication.projector_benchmark_invoked is False
    assert publication.projector_simulator_invoked is False


@pytest.mark.parametrize("source_layout", ["hardlink", "symlink-ancestor"])
def test_runtime_projector_snapshots_installed_source_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_layout: str,
) -> None:
    projector_source = Path(runtime_verified_operations_module.__file__).resolve()
    adapter_source = projector_source.with_name("runtime_benchmark.py")
    if source_layout == "hardlink":
        module_root = tmp_path / "hardlinked-package"
        link_source_root = tmp_path / "hardlink-source"
        module_root.mkdir()
        link_source_root.mkdir()
        for source in (projector_source, adapter_source):
            link_source = link_source_root / source.name
            link_source.write_bytes(source.read_bytes())
            os.link(link_source, module_root / source.name)
        assert (module_root / projector_source.name).stat().st_nlink > 1
    else:
        module_root = tmp_path / "symlinked-package"
        module_root.symlink_to(projector_source.parent, target_is_directory=True)
    monkeypatch.setattr(
        runtime_verified_operations_module,
        "__file__",
        str(module_root / projector_source.name),
    )
    executable = _write_fake_benchmark(tmp_path)

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    publication = load_simready_runtime_verified_operations(
        report.verified_operation_publication_path or ""
    )
    operation = publication.operations[0]
    assert Path(operation.result.producer.contract.path).parent.name == (
        "execution-contracts"
    )
    assert operation.result.producer.contract == operation.result.verifier.contract
    assert operation.result.projector.contract != operation.result.producer.contract


def test_runtime_projector_rejects_native_report_drift(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path)
    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))
    native_report_path = Path(report.native_report_path or "")
    native_report_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(VerifiedOperationError, match="differs from runtime report"):
        project_simready_runtime_verified_operations(
            report.report_path or "",
            output_dir=tmp_path / "tampered-projection",
        )


def test_runtime_adapter_fails_closed_when_projection_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _write_fake_benchmark(tmp_path)

    def reject_projection(*_args, **_kwargs):
        raise VerifiedOperationError("projection failed")

    monkeypatch.setattr(
        runtime_verified_operations_module,
        "project_simready_runtime_verified_operations",
        reject_projection,
    )

    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))

    assert report.status == "error"
    assert report.passed is False
    assert report.verified_operation_publication_path is None
    assert any("projection failed" in error for error in report.errors)
    publication_root = (
        Path(report.report_path or "").parent
        / SIMREADY_RUNTIME_VERIFIED_OPERATIONS_DIRNAME
    )
    assert not publication_root.exists()


def test_runtime_publication_rejects_omitted_native_check(tmp_path: Path) -> None:
    executable = _write_fake_benchmark(tmp_path, additional_status="skipped")
    report = run_simready_runtime_validation(_runtime_input(tmp_path, executable))
    publication_path = Path(report.verified_operation_publication_path or "")
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    publication["operations"].pop()
    publication_path.write_text(json.dumps(publication), encoding="utf-8")

    with pytest.raises(
        VerifiedOperationError, match="cover every native runtime check"
    ):
        load_simready_runtime_verified_operations(publication_path)
