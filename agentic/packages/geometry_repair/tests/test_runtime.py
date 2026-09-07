# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for Geometry Repair's usd-cli runtime boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import geometry_repair.runtime as runtime_module
from geometry_repair.runtime import (
    _collision_runtime_policy,
    _usd_cli_runtime_validation,
    validate_collision_runtime,
)


def test_ovphysx_runtime_uses_raw_usd_cli_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from usd_core.session import Session

    proxy = tmp_path / "proxy.usda"
    proxy.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "runtime"
    calls: dict[str, object] = {}

    def fake_author_scenario(
        _physics_usd: Path, output_scene: Path, **_kwargs: object
    ) -> dict[str, object]:
        output_scene.parent.mkdir(parents=True, exist_ok=True)
        output_scene.write_text("#usda 1.0\n", encoding="utf-8")
        return {
            "body_prim_path": "/GeometryRepairRuntimeProxy",
            "rest_position": [0.0, 0.0, 0.0],
            "world_up": [0.0, 0.0, 1.0],
        }

    class FakeSession:
        def physics_simulate(self, **kwargs: object) -> SimpleNamespace:
            calls.update(kwargs)
            return SimpleNamespace(
                data={
                    "engine": "ovphysx",
                    "executor": "remote",
                    "report_path": str(output / "runtime_validation_report.json"),
                },
                issues=[],
            )

    monkeypatch.setattr(Session, "open", lambda path: FakeSession())
    monkeypatch.setattr(runtime_module, "_author_drop_settle_scenario", fake_author_scenario)

    report = _usd_cli_runtime_validation(
        proxy,
        output,
        engine="ovphysx",
        duration_s=1.5,
        dt=1.0 / 240.0,
        sample_fps=30,
        drop_height_m=0.02,
    )

    assert report["executor"] == "remote"
    assert calls == {
        "scene": str(output / "drop_settle_scene.usda"),
        "body": "/GeometryRepairRuntimeProxy",
        "rest_position": [0.0, 0.0, 0.0],
        "world_up": [0.0, 0.0, 1.0],
        "engine": "ovphysx",
        "duration": 1.5,
        "dt": 1.0 / 240.0,
        "fps": 30,
        "output": str(output),
        "relax_ovphysx_address_space_limit": True,
    }


def test_collision_runtime_policy_preserves_evidence_and_requires_gravity(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "frame": 0,
                "t": 0.0,
                "pose": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                "vel": [0.0] * 6,
            }
        )
        + "\n"
        + json.dumps(
            {
                "frame": 1,
                "t": 1.0,
                "pose": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                "vel": [0.0] * 6,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "runtime_validation_report.json"

    report = _collision_runtime_policy(
        {
            "engine": "ovphysx",
            "executor": "local",
            "n_bodies": 1,
            "trajectory_jsonl": str(trajectory),
            "drop_height_m": 0.02,
            "scene_info": {"world_up": [0.0, 0.0, 1.0]},
            "recording_usda": str(tmp_path / "recording.usda"),
            "report_path": str(report_path),
            "failures": [],
            "warnings": [],
        },
        runtime_root=tmp_path,
    )

    assert report["ok"] is False
    assert report["runtime_report"] == str(report_path)
    assert report["recording_usda"] == str(tmp_path / "recording.usda")
    assert report["acceptance"] == {
        "expected_body_count": 1,
        "max_ground_penetration_m": None,
        "require_gravity_response": True,
    }
    assert any("gravity response" in failure for failure in report["failures"])
    assert json.loads(report_path.read_text(encoding="utf-8"))["executor"] == "local"


def test_collision_runtime_policy_rejects_trajectory_outside_runtime_root(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    outside = tmp_path / "trajectory.jsonl"
    outside.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="outside its output root"):
        _collision_runtime_policy(
            {
                "n_bodies": 1,
                "trajectory_jsonl": str(outside),
                "report_path": str(runtime_root / "runtime_validation_report.json"),
            },
            runtime_root=runtime_root,
        )


def test_collision_runtime_policy_rejects_symlinked_trajectory(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    (runtime_root / "trajectory.jsonl").symlink_to(outside)

    with pytest.raises(RuntimeError, match="unsafe trajectory"):
        _collision_runtime_policy(
            {
                "n_bodies": 1,
                "trajectory_jsonl": str(runtime_root / "trajectory.jsonl"),
                "report_path": str(runtime_root / "runtime_validation_report.json"),
            },
            runtime_root=runtime_root,
        )


def test_collision_runtime_writes_new_receipts_under_usd_cli_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    collision = tmp_path / "collision.usda"
    collision.write_text("#usda 1.0\n", encoding="utf-8")
    observed_outputs: list[Path] = []

    def fake_author(_source: Path, output: Path, **_kwargs: object) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("#usda 1.0\n", encoding="utf-8")
        return output

    def fake_runtime(_proxy: Path, output: Path, **_kwargs: object) -> dict:
        observed_outputs.append(output)
        report = output / "runtime_validation_report.json"
        return {
            "runtime_report": str(report),
            "failures": [],
            "warnings": [],
        }

    monkeypatch.setattr(
        runtime_module,
        "_RIGID_ORIENTATIONS",
        (("identity", (0.0, 0.0, 0.0, 1.0)),),
    )
    monkeypatch.setattr(runtime_module, "_author_temporary_proxy", fake_author)
    monkeypatch.setattr(runtime_module, "_usd_cli_runtime_validation", fake_runtime)
    monkeypatch.setattr(
        runtime_module,
        "_collision_runtime_policy",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        runtime_module,
        "_exact_proxy_ground_penetration",
        lambda *_args, **_kwargs: {"exact_maximum_ground_penetration_m": 0.0},
    )

    result = validate_collision_runtime(
        collision,
        tmp_path / "runtime",
        engine="fake",
    )

    assert result["status"] == "pass"
    assert observed_outputs == [(tmp_path / "runtime" / "identity" / "usd_cli_runtime").resolve()]
