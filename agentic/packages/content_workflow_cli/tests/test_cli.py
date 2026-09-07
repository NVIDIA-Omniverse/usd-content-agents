# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the content-workflow-cli CLI wrapper."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import textwrap
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from content_workflow_cli import cli
from content_workflow_cli.cli import main
from content_workflow_cli.controlled_artifact_cli import (
    CONTROLLED_ARTIFACT_ROOT_ENV,
)
from content_workflow_cli.runner import RunResult


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_building_cli_parser_does_not_eagerly_import_texture_runtime() -> None:
    script = """
import json
import sys
from content_workflow_cli import cli

cli.build_parser()
modules = (
    "content_agent_workflows.texture",
    "world_understanding.functions.models.vision_language_models",
    "pxr",
)
print(json.dumps({name: name in sys.modules for name in modules}))
"""
    repo_root = _repo_root()
    python_paths = [
        repo_root / "agentic" / "packages" / "content_workflow_cli",
        repo_root / "agentic" / "packages" / "content_agent_workflows",
        repo_root,
    ]
    env = os.environ.copy()
    if existing_pythonpath := env.get("PYTHONPATH"):
        python_paths.extend(
            Path(path) for path in existing_pythonpath.split(os.pathsep)
        )
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert json.loads(completed.stdout) == {
        "content_agent_workflows.texture": False,
        "world_understanding.functions.models.vision_language_models": False,
        "pxr": False,
    }


def test_copy_text_artifact_replaces_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("approved report\n", encoding="utf-8")
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged\n", encoding="utf-8")
    destination = tmp_path / "report.txt"
    destination.symlink_to(victim)

    cli._copy_text_artifact(source, destination)

    assert victim.read_text(encoding="utf-8") == "unchanged\n"
    assert not destination.is_symlink()
    assert destination.read_text(encoding="utf-8") == "approved report\n"


def test_material_authoring_route_is_hidden_without_internal_skills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "material_authoring_skills_available",
        lambda: False,
    )

    parser = cli.build_parser()
    root_subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    materials_parser = root_subparsers.choices["materials"]
    material_subparsers = next(
        action
        for action in materials_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert "author" not in material_subparsers.choices
    assert "generate" not in material_subparsers.choices
    assert {"assign", "apply"}.issubset(material_subparsers.choices)


def test_building_cli_parser_does_not_discover_git_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_find_repo_root(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("parser construction must not invoke Git root discovery")

    monkeypatch.setattr(cli, "find_repo_root", fail_find_repo_root)

    cli.build_parser()


def test_controlled_json_writer_is_confined_and_non_clobbering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(CONTROLLED_ARTIFACT_ROOT_ENV, str(tmp_path))

    assert (
        main(
            [
                "artifact",
                "write-json",
                "--output",
                "nested/result.json",
                "--json",
                '{"answer": 42}',
            ]
        )
        == 0
    )
    output = tmp_path / "nested" / "result.json"
    assert json.loads(output.read_text(encoding="utf-8")) == {"answer": 42}

    monkeypatch.setattr(sys, "stdin", StringIO('{"answer": 99}'))
    assert main(["artifact", "write-json", "--output", "nested/result.json"]) == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {"answer": 42}

    decision_patch = tmp_path / "raw" / "physics_decision_patch.json"
    decision_patch.parent.mkdir()
    decision_patch.write_text('{"revision": 1}\n', encoding="utf-8")
    assert (
        main(
            [
                "artifact",
                "write-json",
                "--replace",
                "--output",
                "raw/physics_decision_patch.json",
                "--json",
                '{"revision": 2}',
            ]
        )
        == 0
    )
    assert json.loads(decision_patch.read_text(encoding="utf-8")) == {"revision": 2}

    assert (
        main(
            [
                "artifact",
                "write-json",
                "--replace",
                "--output",
                "nested/result.json",
                "--json",
                '{"answer": 100}',
            ]
        )
        == 2
    )
    assert json.loads(output.read_text(encoding="utf-8")) == {"answer": 42}

    monkeypatch.setattr(sys, "stdin", StringIO('{"escape": true}'))
    assert main(["artifact", "write-json", "--output", "../escape.json"]) == 2
    assert not (tmp_path.parent / "escape.json").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows startup fast path")
def test_console_entrypoint_fast_paths_controlled_json_writer(
    tmp_path: Path,
) -> None:
    script = """
import json
import sys
from content_workflow_cli.entrypoint import main

returncode = main([
    "artifact",
    "write-json",
    "--output",
    "nested/result.json",
    "--json",
    '{"answer": 42}',
])
print(json.dumps({
    "returncode": returncode,
    "full_cli_imported": "content_workflow_cli.cli" in sys.modules,
}))
"""
    environment = {
        **os.environ,
        CONTROLLED_ARTIFACT_ROOT_ENV: str(tmp_path),
    }

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=True,
        text=True,
    )

    lines = completed.stdout.splitlines()
    assert json.loads(lines[0]) == {
        "output": "nested/result.json",
        "size_bytes": 19,
    }
    assert json.loads(lines[1]) == {
        "returncode": 0,
        "full_cli_imported": False,
    }
    assert json.loads((tmp_path / "nested" / "result.json").read_text()) == {
        "answer": 42
    }


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell argv")
def test_powershell_controlled_json_guidance_preserves_native_argument(
    tmp_path: Path,
) -> None:
    environment = {
        **os.environ,
        CONTROLLED_ARTIFACT_ROOT_ENV: str(tmp_path),
        "PATH": os.pathsep.join(
            [str(Path(sys.executable).parent), os.environ.get("PATH", "")]
        ),
    }
    command = (
        "content-workflow-cli artifact write-json --output native/result.json "
        r"--json '{\"note\":\"two\u0020words\"}'"
    )

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads((tmp_path / "native" / "result.json").read_text()) == {
        "note": "two words"
    }


def _write_fake_simready_foundation(tmp_path: Path) -> tuple[Path, Path]:
    foundation_root = tmp_path / "simready-foundation"
    spec_root = foundation_root / "nv_core" / "sr_specs" / "docs"
    (spec_root / "capabilities").mkdir(parents=True)
    (spec_root / "features").mkdir(parents=True)
    (spec_root / "profiles").mkdir(parents=True)
    (spec_root / "profiles" / "profiles.toml").write_text(
        '[Prop-Robotics-Neutral]\n"1.0.0" = { features = [] }\n',
        encoding="utf-8",
    )
    (foundation_root / "requirements.txt").write_text(
        "simready-foundation-test==1.0\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(foundation_root)], check=True)
    subprocess.run(
        ["git", "-C", str(foundation_root), "add", "."],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(foundation_root),
            "-c",
            "user.name=SimReady Test",
            "-c",
            "user.email=simready-test@example.invalid",
            "commit",
            "-qm",
            "test fixture",
        ],
        check=True,
    )
    venv = tmp_path / "simready-venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    site_packages = (
        venv / "Lib" / "site-packages"
        if os.name == "nt"
        else venv
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages.mkdir(parents=True)
    executable = bin_dir / (
        "simready-validate.exe" if os.name == "nt" else "simready-validate"
    )
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"passed": True, "status": "PASS"}), encoding="utf-8")
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return foundation_root, venv


def _write_fake_simready_runtime_benchmark(tmp_path: Path) -> Path:
    executable = tmp_path / "runtime-benchmark" / "simready-benchmark"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            import pathlib
            import sys

            if "--version" in sys.argv:
                print("SimReady Benchmark (v2026.6.5)")
                raise SystemExit(0)

            def flag_value(flag):
                return sys.argv[sys.argv.index(flag) + 1]

            output_dir = pathlib.Path(flag_value("--output-dir"))
            asset_path = pathlib.Path(flag_value("--assets"))
            plan_path = pathlib.Path(flag_value("--output"))
            _drive, asset_tail = os.path.splitdrive(str(asset_path))
            asset_key = asset_tail.lstrip("/\\\\").replace("\\\\", "/")
            report_dir = output_dir / "report"
            state_dir = output_dir / "state"
            test_dir = (
                output_dir
                / "results"
                / pathlib.Path(*pathlib.PurePosixPath(asset_key).parent.parts)
                / ".simready"
                / "runtime"
                / "rigid_body_falls_isaac"
            )
            report_dir.mkdir(parents=True, exist_ok=True)
            state_dir.mkdir(parents=True, exist_ok=True)
            test_dir.mkdir(parents=True, exist_ok=True)
            session_log = output_dir / "logs" / "isaac_sim_1.log"
            session_log.parent.mkdir(parents=True, exist_ok=True)
            session_log.write_text("fake Kit session log\\n", encoding="utf-8")
            plan_path.write_text(
                json.dumps(
                    {
                        "plan_version": "1.1.0",
                        "created": "2026-08-12T00:00:00Z",
                        "sr_specs_path": flag_value("--sr-specs"),
                        "content_root": "",
                        "test_dirs": [],
                        "profiles": [],
                        "features": [],
                        "tests": [],
                        "assets": [{"id": 1, "path": str(asset_path), "tests": [1]}],
                        "overrides": [],
                        "engine_configs": {},
                        "warnings": [],
                        "engine_coverage": None,
                    }
                ),
                encoding="utf-8",
            )
            test = {
                "name": "rigid_body_falls_isaac",
                "version": "1.2.0",
                "status": "pass",
                "duration": 1.25,
                "engine": "isaac_sim",
                "engine_version": "5.1.0",
                "message": "",
                "metrics": {"settled_after_s": 1.0},
                "media": [
                    {"kind": "video", "role": "summary", "filename": "fall.mp4"}
                ],
                "kit_logs": [],
                "description": "Free-fall stability check.",
                "expected_video": "Asset settles.",
            }
            report = {
                "metadata": {
                    "schema_version": "5.0",
                    "framework_version": "2026.6.5",
                    "sessions": [
                        {
                            "id": "isaac_sim_1",
                            "engine": "isaac_sim",
                            "log_file": "logs/isaac_sim_1.log",
                            "duration": 1.25,
                            "tests_run": 1,
                            "test_time": 1.25,
                        }
                    ],
                },
                "assets": {
                    asset_key: {
                        "asset_path": asset_key,
                        "profiles": [
                            {
                                "id": "Prop-Robotics-Neutral",
                                "version": "1.0.0",
                                "status": "pass",
                                "features": [
                                    {
                                        "id": "FET003_BASE_NEUTRAL",
                                        "version": "0.1.0",
                                        "status": "pass",
                                        "tests": [test],
                                    }
                                ],
                            }
                        ],
                        "stamping": {},
                    }
                },
            }
            (report_dir / "test_results_index.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            (output_dir / "run_summary.json").write_text(
                json.dumps(
                    {
                        "total_work_items": 1,
                        "completed": 1,
                        "failed": 0,
                        "status": "completed",
                        "readiness": {"status": "ready"},
                    }
                ),
                encoding="utf-8",
            )
            (state_dir / "events.jsonl").write_text(
                json.dumps(
                    {
                        "type": "test_complete",
                        "status": "pass",
                        "test": "rigid_body_falls_isaac",
                    }
                )
                + "\\n",
                encoding="utf-8",
            )
            result = dict(test)
            result.update(
                {
                    "asset": asset_key,
                    "test_name": "rigid_body_falls_isaac",
                    "test_version": "1.2.0",
                }
            )
            (test_dir / "result.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
            (test_dir / "fall.mp4").write_bytes(b"fake-video")
            raise SystemExit(0)
            """
        ),
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def _write_error_status_simready_foundation(tmp_path: Path) -> tuple[Path, Path]:
    foundation_root, venv = _write_fake_simready_foundation(tmp_path)
    executable = (
        venv
        / ("Scripts" if os.name == "nt" else "bin")
        / ("simready-validate.exe" if os.name == "nt" else "simready-validate")
    )
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    "passed": False,
    "status": "ERROR",
    "issues": [{"requirement_id": "NP.006", "severity": "ERROR"}],
}), encoding="utf-8")
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return foundation_root, venv


def _write_failed_simready_foundation(tmp_path: Path) -> tuple[Path, Path]:
    foundation_root, venv = _write_fake_simready_foundation(tmp_path)
    executable = (
        venv
        / ("Scripts" if os.name == "nt" else "bin")
        / ("simready-validate.exe" if os.name == "nt" else "simready-validate")
    )
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    "passed": False,
    "status": "FAIL",
    "issues": [{"requirement_id": "NP.006", "severity": "ERROR"}],
}), encoding="utf-8")
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return foundation_root, venv


def test_simready_preflight_cli_reports_json(tmp_path: Path, capsys) -> None:
    foundation_root, venv = _write_fake_simready_foundation(tmp_path)

    code = main(
        [
            "preflight",
            "simready-foundation",
            "--foundation-root",
            str(foundation_root),
            "--venv",
            str(venv),
            "--no-install-missing",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["passed"] is True
    assert payload["available_profiles"] == ["Prop-Robotics-Neutral"]


def test_simready_preflight_cli_expands_report_path(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    foundation_root, venv = _write_fake_simready_foundation(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(cwd)

    code = main(
        [
            "preflight",
            "simready-foundation",
            "--foundation-root",
            str(foundation_root),
            "--venv",
            str(venv),
            "--no-install-missing",
            "--report",
            "~/simready-preflight.json",
        ]
    )

    capsys.readouterr()
    assert code == 0
    assert (home / "simready-preflight.json").exists()
    assert not (cwd / "~" / "simready-preflight.json").exists()


def test_simready_validate_profile_cli_reports_json(tmp_path: Path, capsys) -> None:
    foundation_root, venv = _write_fake_simready_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")

    code = main(
        [
            "simready",
            "validate-profile",
            str(asset),
            "--foundation-root",
            str(foundation_root),
            "--venv",
            str(venv),
            "--no-install-missing",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["profile_target"] == "Prop-Robotics-Neutral@1.0.0"
    assert payload["passed"] is True


def test_simready_validate_runtime_cli_runs_canonical_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from content_agent_workflows.simready import (
        SimReadyRuntimeCheckPayload,
        load_simready_runtime_verified_operations,
        runtime_benchmark,
    )

    executable = _write_fake_simready_runtime_benchmark(tmp_path)
    if os.name == "nt":
        real_run_bounded_process = runtime_benchmark._run_bounded_benchmark_process

        monkeypatch.setattr(
            runtime_benchmark,
            "_benchmark_version",
            lambda _executable: (runtime_benchmark.SIMREADY_BENCHMARK_VERSION, None),
        )

        def run_python_fixture(command, **kwargs):
            return real_run_bounded_process(
                [sys.executable, *command],
                **kwargs,
            )

        monkeypatch.setattr(
            runtime_benchmark,
            "_run_bounded_benchmark_process",
            run_python_fixture,
        )
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")
    sr_specs = tmp_path / "simready_foundations" / "nv_core" / "sr_specs"
    sr_specs.mkdir(parents=True)
    engines_toml = tmp_path / "engines.toml"
    engines_toml.write_text("[kit.isaac_sim]\nversion = '5.1.0'\n", encoding="utf-8")
    tests_path = tmp_path / "runtime-tests"
    tests_path.mkdir()
    output_dir = tmp_path / "runtime-output"

    code = main(
        [
            "simready",
            "validate-runtime",
            str(asset),
            "--output-dir",
            str(output_dir),
            "--sr-specs",
            str(sr_specs),
            "--engines-toml",
            str(engines_toml),
            "--tests-path",
            str(tests_path),
            "--feature",
            "FET003",
            "--runtime",
            "isaac_sim",
            "--benchmark-executable",
            str(executable),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "pass"
    assert payload["checks"][0]["native_status"] == "pass"
    publication_path = Path(payload["verified_operation_publication_path"])
    assert publication_path.is_file()
    publication = load_simready_runtime_verified_operations(publication_path)
    assert len(publication.operations) == 1
    operation = publication.operations[0]
    check_payload = SimReadyRuntimeCheckPayload.model_validate_json(
        Path(operation.payload.path).read_text(encoding="utf-8")
    )
    assert check_payload.check.test_name == "rigid_body_falls_isaac"
    _drive, asset_tail = os.path.splitdrive(str(asset.resolve()))
    assert check_payload.check.asset_key == asset_tail.lstrip("/\\").replace("\\", "/")
    assert Path(operation.result.source.path) == asset.resolve()
    assert any(
        Path(binding.path).name == "isaac_sim_1.log"
        for binding in operation.result.artifacts
    )


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--max-concurrent", "0", "must be at least 1"),
        ("--timeout", "0", "must be greater than 0"),
    ],
)
def test_simready_validate_runtime_cli_rejects_nonpositive_limits(
    flag: str,
    value: str,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "simready",
            "validate-runtime",
            "asset.usda",
            "--output-dir",
            "runtime-output",
            "--sr-specs",
            "sr-specs",
            "--engines-toml",
            "engines.toml",
            flag,
            value,
        ]
    )

    assert exit_code == 2
    assert message in capsys.readouterr().err


def test_simready_validate_profile_cli_non_strict_allows_profile_error_status(
    tmp_path: Path,
    capsys,
) -> None:
    foundation_root, venv = _write_error_status_simready_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")

    code = main(
        [
            "simready",
            "validate-profile",
            str(asset),
            "--foundation-root",
            str(foundation_root),
            "--venv",
            str(venv),
            "--no-install-missing",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["status"] == "ERROR"
    assert payload["next_step"] == "simready-conform-profile"


def test_simready_conform_profile_cli_reports_json(tmp_path: Path, capsys) -> None:
    foundation_root, venv = _write_failed_simready_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")
    validation_report = tmp_path / "simready-profile.json"

    validation_code = main(
        [
            "simready",
            "validate-profile",
            str(asset),
            "--foundation-root",
            str(foundation_root),
            "--venv",
            str(venv),
            "--no-install-missing",
            "--report",
            str(validation_report),
            "--json",
        ]
    )
    validation_payload = json.loads(capsys.readouterr().out)
    assert validation_code == 0
    assert validation_payload["status"] == "FAIL"
    assert validation_payload["validator_runtime_verified"] is True

    code = main(
        [
            "simready",
            "conform-profile",
            str(asset),
            "--foundation-root",
            str(foundation_root),
            "--venv",
            str(venv),
            "--validation-report",
            str(validation_report),
            "--output-dir",
            str(tmp_path / "conform"),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["status"] == "PASS"
    assert payload["failed_requirements"] == ["NP.006"]
    assert payload["requirements_repaired"] == ["NP.006"]


def test_simready_conform_profile_cli_forwards_physics_fingerprint(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import content_agent_workflows.simready as simready_module

    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "Asset" {}\n', encoding="utf-8")
    fingerprint = "a" * 64
    captured = {}

    def fake_conformance(params):
        captured["params"] = params
        payload = {
            "passed": True,
            "status": "PASS",
            "output_usd_path": str(asset),
            "report_path": None,
            "requirements_blocked": [],
            "errors": [],
        }
        return SimpleNamespace(**payload, model_dump=lambda **_kwargs: payload)

    monkeypatch.setattr(
        simready_module,
        "run_simready_profile_conformance",
        fake_conformance,
    )

    code = main(
        [
            "simready",
            "conform-profile",
            str(asset),
            "--output-dir",
            str(tmp_path / "conform"),
            "--repair",
            "G3A.HYG.001",
            "--expected-physics-inventory-sha256",
            fingerprint,
            "--json",
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert captured["params"].expected_physics_inventory_sha256 == fingerprint


def test_physics_apply_cli_wrapper_runs_fake_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from content_agent_workflows.physics import usd_cli_ops

    asset = (
        _repo_root()
        / "apps"
        / "physics_agent_service"
        / "tests"
        / "test_data"
        / "simple_cube.usda"
    )
    real_run_json = usd_cli_ops._run_json

    def run_with_attested_ovrtx_probe(**kwargs: object) -> dict[str, object]:
        arguments = kwargs.get("arguments")
        if isinstance(arguments, list) and arguments[:1] == ["render-probe"]:
            output_dir = Path(arguments[arguments.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            render_path = output_dir / "probe.png"
            Image.new("RGB", (64, 64), "red").save(render_path)
            return {
                "schema_version": "usd-cli.render-probe.v1",
                "engine": "ovrtx",
                "resolved_renderer": "ovrtx",
                "transport": "local",
                "ready": True,
                "render": {
                    "backend": "ovrtx",
                    "path": str(render_path),
                    "width": 64,
                    "height": 64,
                    "size_bytes": render_path.stat().st_size,
                },
            }
        return real_run_json(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(usd_cli_ops, "_run_json", run_with_attested_ovrtx_probe)

    code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(asset),
            "--output-dir",
            str(tmp_path),
            "--simulation-engine",
            "fake",
            "--duration-s",
            "0.2",
            "--sample-fps",
            "10",
            "--drop-height-m",
            "0.1",
            "--direct-executor",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0, payload.get("error")
    assert payload["success"] is True
    assert Path(payload["physics_usd_path"]).exists()
    assert Path(payload["validation_evidence_path"]).exists()


@pytest.mark.parametrize("direct_executor", [False, True])
def test_physics_apply_rejects_explicit_video_reference_before_dispatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    direct_executor: bool,
) -> None:
    asset = tmp_path / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    video = tmp_path / "motion.mp4"
    video.write_bytes(b"video")
    argv = [
        "physics",
        "apply",
        "--usd",
        str(asset),
        "--output-dir",
        str(tmp_path / "out"),
        "--reference-image",
        str(video),
    ]
    if direct_executor:
        argv.append("--direct-executor")

    code = main(argv)

    assert code == 2
    assert "Video references are unsupported" in capsys.readouterr().err


def test_physics_apply_deterministic_workflow_forwards_the_penetration_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The agentic path honours --revalidation-max-penetration-m at the
    apply-phase runtime gate; the deterministic branch builds its own
    PhysicsApplyWorkflowInput and must forward the same override instead of
    silently falling back to the scene-tool default."""

    import content_agent_workflows.physics as physics_module
    from content_agent_workflows.physics import PhysicsApplyWorkflowInput

    captured: dict[str, PhysicsApplyWorkflowInput] = {}

    def fake_apply(params: PhysicsApplyWorkflowInput) -> SimpleNamespace:
        captured["params"] = params
        payload = {
            "success": True,
            "validation_status": "pass",
            "physics_usd_path": None,
            "validation_evidence_path": None,
        }
        return SimpleNamespace(**payload, model_dump=lambda **_kwargs: payload)

    monkeypatch.setattr(physics_module, "run_physics_apply_workflow", fake_apply)

    asset = tmp_path / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")

    code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(asset),
            "--output-dir",
            str(tmp_path / "out"),
            "--simulation-engine",
            "fake",
            "--deterministic-workflow",
            "--revalidation-max-penetration-m",
            "0.05",
            "--scene-tool-timeout",
            "1800",
            "--json",
        ]
    )

    assert code == 0
    assert (
        "WARNING: --deterministic-workflow is deprecated; use --direct-executor instead."
        in capsys.readouterr().err
    )
    assert captured["params"].max_ground_penetration_m == 0.05
    # On this subparser --scene-tool-timeout IS the documented per-request
    # budget for usd-cli physics operations (default 300 s); the
    # deterministic branch must keep forwarding it as the command timeout, or
    # a user-provided budget for heavy assets silently stops applying.
    assert captured["params"].scene_tool_timeout_seconds == 1800.0


def test_physics_apply_deterministic_workflow_forwards_coordinator_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import content_agent_workflows.physics as physics_module
    from content_agent_workflows.physics import PhysicsApplyWorkflowInput

    captured: dict[str, PhysicsApplyWorkflowInput] = {}

    def fake_apply(params: PhysicsApplyWorkflowInput) -> SimpleNamespace:
        captured["params"] = params
        payload = {
            "success": True,
            "validation_status": "pass",
            "physics_usd_path": None,
            "validation_evidence_path": None,
        }
        return SimpleNamespace(**payload, model_dump=lambda **_kwargs: payload)

    monkeypatch.setattr(physics_module, "run_physics_apply_workflow", fake_apply)
    asset = tmp_path / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    decision = tmp_path / "physics-decision.json"
    decision.write_text('{"schema_version":"test"}\n', encoding="utf-8")
    topology = tmp_path / "physics-topology.json"
    topology.write_text('{"schema_version":"test"}\n', encoding="utf-8")

    code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(asset),
            "--output-dir",
            str(tmp_path / "out"),
            "--decision-patch",
            str(decision),
            "--topology-plan",
            str(topology),
            "--deterministic-workflow",
            "--json",
        ]
    )

    assert code == 0
    assert captured["params"].decision_patch_path == decision.resolve()
    assert captured["params"].topology_plan_path == topology.resolve()


def test_physics_apply_rejects_coordinator_patch_on_nested_agent_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    asset = tmp_path / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    decision = tmp_path / "physics-decision.json"
    decision.write_text("{}\n", encoding="utf-8")

    code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(asset),
            "--output-dir",
            str(tmp_path / "out"),
            "--decision-patch",
            str(decision),
        ]
    )

    assert code == 2
    assert "require --direct-executor" in capsys.readouterr().err


def test_physics_refine_external_cli_handler(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """Pins the operator-facing CLI contract: missing --runtime-config exits
    2 before any run, --child-timeout derives from the sweep budget when
    omitted, and the JSON `status` field is the only way to tell an honest
    stop from a real failure (both exit 1)."""

    from content_workflow_cli import external_refine_runner

    code = main(
        [
            "physics",
            "refine-external",
            "--runtime-config",
            str(tmp_path / "missing.yaml"),
            "--user-prompt",
            "keep the gear grasped",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    capsys.readouterr()
    assert code == 2

    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(config, **_kwargs):
        captured["config"] = config
        return external_refine_runner.ExternalRefineRunResult(
            run_dir=tmp_path / "run",
            status="stopped",
            validated=False,
            returncode=1,
            reasons=("objective cannot express the goal",),
        )

    monkeypatch.setattr(external_refine_runner, "run_physics_external_refine", fake_run)
    code = main(
        [
            "physics",
            "refine-external",
            "--runtime-config",
            str(runtime),
            "--user-prompt",
            "keep the gear grasped",
            "--output-dir",
            str(tmp_path / "run"),
            "--max-iterations",
            "3",
            "--sweep-deadline-seconds",
            "100",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "stopped"
    assert payload["validated"] is False
    config = captured["config"]
    assert config.child_timeout_seconds == 3 * 100 + 3600.0


def _physics_apply_failure_result(
    tmp_path: Path, record: dict[str, str] | None
) -> RunResult:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    if record is not None:
        (raw_dir / "physics_finalize_result_1.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    return RunResult(
        run_dir=run_dir,
        prompt_path=run_dir / "agent_prompt.md",
        request_path=run_dir / "request.json",
        child_output_path=run_dir / "child-output.log",
        child_final_path=run_dir / "child-final.md",
        returncode=1,
        trace_paths={},
    )


def test_physics_apply_agentic_failure_prints_error_line_with_finalize_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed agentic run must print an explicit error line naming the
    finalize error and the record artifact — the child's final message is
    often a success-looking summary (regression: a failed run read as
    passed with the cause buried in raw/)."""
    result = _physics_apply_failure_result(
        tmp_path, {"error": "usd-cli session reload guard tripped"}
    )
    from content_workflow_cli import cli as cli_module

    monkeypatch.setattr(cli_module, "run_physics_apply", lambda config: result)
    asset = tmp_path / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")

    code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(asset),
            "--output-dir",
            str(tmp_path / "out"),
            "--repo-root",
            str(_repo_root()),
        ]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert "error: physics apply failed with exit code 1" in err
    assert "finalize failed: usd-cli session reload guard tripped" in err
    assert "finalize report at" in err
    assert "physics_finalize_result_1.json" in err


def test_physics_apply_agentic_failure_points_at_record_without_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A finalize record without a recorded error is still the closest
    artifact to the failure; the error line must point at it."""
    result = _physics_apply_failure_result(tmp_path, {"verdict": "conditional"})
    from content_workflow_cli import cli as cli_module

    monkeypatch.setattr(cli_module, "run_physics_apply", lambda config: result)
    asset = tmp_path / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")

    code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(asset),
            "--output-dir",
            str(tmp_path / "out"),
            "--repo-root",
            str(_repo_root()),
        ]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert "error: physics apply failed with exit code 1" in err
    assert "finalize failed" not in err
    assert ": finalize report at" in err
    assert "physics_finalize_result_1.json" in err


def test_physics_apply_agentic_failure_json_stdout_stays_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--json stdout must stay parseable on a failed agentic run: the
    diagnostic error line goes to stderr only."""
    result = _physics_apply_failure_result(
        tmp_path, {"error": "usd-cli session reload guard tripped"}
    )
    from content_workflow_cli import cli as cli_module

    monkeypatch.setattr(cli_module, "run_physics_apply", lambda config: result)
    asset = tmp_path / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")

    code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(asset),
            "--output-dir",
            str(tmp_path / "out"),
            "--repo-root",
            str(_repo_root()),
            "--json",
        ]
    )

    assert code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["returncode"] == 1
    assert "error: physics apply failed" not in captured.out
    assert "error: physics apply failed with exit code 1" in captured.err


def test_physics_apply_tune_failure_points_at_tuning_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """On a --tune failure the finalize phase succeeded before tuning, so the
    latest finalize record is clean; the error line must point at the tuning
    summary (which carries the real cause), not a clean finalize record."""
    result = _physics_apply_failure_result(tmp_path, {"verdict": "pass"})
    (tmp_path / "run" / "raw" / "physics_agentic_tuning_result.json").write_text(
        json.dumps({"status": "tool_failure", "error": "sweep budget exhausted"}),
        encoding="utf-8",
    )
    from content_workflow_cli import cli as cli_module

    monkeypatch.setattr(cli_module, "run_physics_apply", lambda config: result)
    asset = tmp_path / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")

    code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(asset),
            "--output-dir",
            str(tmp_path / "out"),
            "--repo-root",
            str(_repo_root()),
            "--tune",
        ]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert "tuning failed: sweep budget exhausted" in err
    assert "physics_agentic_tuning_result.json" in err
    assert "finalize report at" not in err
