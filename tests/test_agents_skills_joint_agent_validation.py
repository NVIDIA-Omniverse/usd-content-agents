# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPO_ROOT / ".agents/skills/joint-agent-validation"
RUNNER_PATH = SKILL_ROOT / "scripts/run_gate3a.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "joint_agent_validation_gate3a", RUNNER_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_joint_agent_validation_skill_has_runnable_gate_commands() -> None:
    skill_text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert "name: joint-agent-validation" in skill_text
    assert "run_gate3a.py" in skill_text
    assert "content-workflow-simready-validate-profile" in skill_text
    assert "Prop-Robotics-Isaac" in skill_text
    assert "Research Preview" in skill_text
    assert RUNNER_PATH.is_file()


def test_gate3a_runner_help_does_not_require_isaac_sim() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Gate 3A" in completed.stdout


def test_gate3a_runner_writes_blocked_report_for_missing_asset(tmp_path: Path) -> None:
    report_path = tmp_path / "gate3a.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            str(tmp_path / "missing.usdz"),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert "unavailable" in report["errors"][0]


def test_gate3a_runner_requires_explicit_eula_acceptance(tmp_path: Path) -> None:
    asset_path = tmp_path / "asset.usda"
    asset_path.write_text("#usda 1.0\n", encoding="utf-8")
    report_path = tmp_path / "gate3a.json"
    environment = os.environ.copy()
    environment.pop("OMNI_KIT_ACCEPT_EULA", None)

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            str(asset_path),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert "OMNI_KIT_ACCEPT_EULA=YES" in report["errors"][0]
    assert report["asset_sha256"]
    assert report["asset_sha256_before"] == report["asset_sha256_after"]


def test_gate3a_runner_finishes_when_initial_hash_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    asset_path = tmp_path / "asset.usda"
    asset_path.write_text("#usda 1.0\n", encoding="utf-8")
    report_path = tmp_path / "gate3a.json"

    def fail_sha256(_path: Path) -> str:
        raise OSError("initial hash failed")

    monkeypatch.setattr(runner, "sha256_file", fail_sha256)

    result = runner.run(
        runner.parse_args([str(asset_path), "--report", str(report_path)])
    )

    assert result == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert report["asset_sha256"] is None
    assert report["asset_sha256_before"] is None
    assert report["asset_sha256_after"] is None
    assert "before Gate 3A validation" in report["errors"][0]


def test_gate3a_runner_hashes_after_validation_runtime_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    asset_path = tmp_path / "asset.usda"
    asset_path.write_text("#usda 1.0\n", encoding="utf-8")
    report_path = tmp_path / "gate3a.json"
    fake_isaacsim = ModuleType("isaacsim")

    def simulation_app(_launch_config: dict[str, Any]) -> object:
        return object()

    fake_isaacsim.__dict__["SimulationApp"] = simulation_app
    monkeypatch.setitem(sys.modules, "isaacsim", fake_isaacsim)
    monkeypatch.setitem(sys.modules, "omni", None)
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    real_finish = runner.finish
    finish_called = False

    def finish_without_process_exit(
        path: Path,
        payload: dict[str, Any],
        *,
        strict: bool,
        kit_started: bool,
    ) -> int:
        nonlocal finish_called
        finish_called = True
        assert kit_started is True
        return int(
            real_finish(
                path,
                payload,
                strict=strict,
                kit_started=False,
            )
        )

    monkeypatch.setattr(runner, "finish", finish_without_process_exit)

    result = runner.run(
        runner.parse_args([str(asset_path), "--report", str(report_path)])
    )

    assert result == 2
    assert finish_called
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert report["asset_sha256_before"] == report["asset_sha256_after"]
    assert report["asset_sha256"] == report["asset_sha256_before"]
    assert "ModuleNotFoundError" in report["errors"][0]


def test_gate3a_runner_finishes_when_post_run_hash_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    asset_path = tmp_path / "asset.usda"
    asset_path.write_text("#usda 1.0\n", encoding="utf-8")
    report_path = tmp_path / "gate3a.json"
    digest = "a" * 64
    hash_calls = 0

    def fail_post_run_sha256(_path: Path) -> str:
        nonlocal hash_calls
        hash_calls += 1
        if hash_calls == 1:
            return digest
        raise OSError("post-run hash failed")

    monkeypatch.setattr(runner, "sha256_file", fail_post_run_sha256)
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)

    result = runner.run(
        runner.parse_args([str(asset_path), "--report", str(report_path)])
    )

    assert result == 2
    assert hash_calls == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert report["asset_sha256"] == digest
    assert report["asset_sha256_before"] == digest
    assert report["asset_sha256_after"] is None
    assert "OMNI_KIT_ACCEPT_EULA=YES" in report["errors"][0]
    assert "after Gate 3A validation" in report["errors"][1]


def test_gate3a_runner_refuses_asset_alias_and_existing_report(tmp_path: Path) -> None:
    asset_path = tmp_path / "asset.usda"
    asset_text = "#usda 1.0\n"
    asset_path.write_text(asset_text, encoding="utf-8")

    alias = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            str(asset_path),
            "--report",
            str(asset_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert alias.returncode == 2
    assert "must not alias" in alias.stderr
    assert asset_path.read_text(encoding="utf-8") == asset_text

    report_path = tmp_path / "gate3a.json"
    report_path.write_text("prior evidence\n", encoding="utf-8")
    existing = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            str(asset_path),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert existing.returncode == 2
    assert "already exists" in existing.stderr
    assert report_path.read_text(encoding="utf-8") == "prior evidence\n"


def test_gate3a_exit_policy_preserves_non_strict_findings() -> None:
    runner = _load_runner()

    assert runner.exit_code("pass", strict=False) == 0
    assert runner.exit_code("warning", strict=False) == 0
    assert runner.exit_code("fail", strict=False) == 0
    assert runner.exit_code("fail", strict=True) == 1
    assert runner.exit_code("blocked", strict=False) == 2
    assert runner.exit_code("error", strict=True) == 2
