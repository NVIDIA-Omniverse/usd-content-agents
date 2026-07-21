# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner tests for content-workflow-cli."""

from __future__ import annotations

import ast
import errno
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from content_agent_workflows.common import physics_validation_evidence
from PIL import Image, ImageDraw

from content_workflow_cli import descendant_reaper, runner
from content_workflow_cli.prompts import (
    build_material_refinement_prompt,
    build_skill_routed_material_assignment_prompt,
)
from content_workflow_cli.runner import (
    ChildProcessInterrupted,
    MaterialAssignConfig,
    PhysicsApplyConfig,
    WatchdogFailure,
    _assignment_uncertainty,
    _ensure_material_assignment_artifacts,
    _fallback_coverage_review_and_assignments,
    _install_child_signal_handlers,
    _run_child_agent,
    _run_subprocess_with_timeout,
    _validate_config,
    _visual_quality_from_assignments_or_fallback,
    run_physics_apply,
)
from content_workflow_cli.trace import TraceWriter, UnsafeRunArtifactError
from content_workflow_cli.workbench_tools import (
    material_finalize,
    material_grounding,
    material_run_packet,
)
from content_workflow_cli.workbench_tools.material_finalize import (
    MaterialFinalizeConfig,
    finalize_material_decisions,
)


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for bridge tests")
    return node


def test_load_material_optimizer_decision_validates_and_normalizes(
    tmp_path: Path,
) -> None:
    decision_path = tmp_path / "optimizer_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.optimizer-decision.v1",
                "task": "material_assignment",
                "optimize": False,
                "flatten_prototypes": True,
                "enable_deinstance": True,
                "enable_split": False,
                "enable_deduplicate": False,
                "rationale": "The source already exposes stable material targets.",
                "evidence": ["Six distinct source-space mesh candidates."],
            }
        ),
        encoding="utf-8",
    )

    decision = runner._load_material_optimizer_decision(decision_path)

    assert decision["optimize"] is False
    assert decision["flatten_prototypes"] is None
    assert decision["enable_deinstance"] is None
    assert decision["enable_split"] is None
    assert decision["enable_deduplicate"] is None
    with pytest.raises(ValueError, match="task must be 'physics_authoring'"):
        runner._load_optimizer_decision(
            decision_path,
            expected_task="physics_authoring",
        )


def test_load_material_optimizer_decision_accepts_flatten_only(
    tmp_path: Path,
) -> None:
    decision_path = tmp_path / "optimizer_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.optimizer-decision.v1",
                "task": "material_assignment",
                "optimize": True,
                "flatten_prototypes": True,
                "enable_deinstance": False,
                "enable_split": False,
                "enable_deduplicate": False,
                "rationale": "Flattening exposes stable material targets.",
                "evidence": ["The source uses prototype-authored meshes."],
            }
        ),
        encoding="utf-8",
    )

    decision = runner._load_material_optimizer_decision(decision_path)

    assert decision["flatten_prototypes"] is True
    assert decision["enable_deinstance"] is False
    assert decision["enable_split"] is False
    assert decision["enable_deduplicate"] is False


def test_material_optimizer_selection_uses_unoptimized_analysis_then_agent_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    observed_packet_configs: list[Any] = []
    closed_sessions: list[str] = []

    def fake_prepare(config: Any) -> dict[str, Any]:
        observed_packet_configs.append(config)
        (config.run_dir / "raw").mkdir(parents=True, exist_ok=True)
        (config.run_dir / "raw" / "material_run_packet.json").write_text(
            json.dumps({"session_id": "analysis-session"}),
            encoding="utf-8",
        )
        return {"session_id": "analysis-session", "initial_evidence_renders": []}

    def fake_child(**kwargs: Any) -> int:
        assert "unoptimized Workbench evidence" in kwargs["prompt"]
        (run_dir / "raw" / "optimizer_decision.json").write_text(
            json.dumps(
                {
                    "schema_version": "content-agents.optimizer-decision.v1",
                    "task": "material_assignment",
                    "optimize": True,
                    "flatten_prototypes": True,
                    "enable_deinstance": True,
                    "enable_split": False,
                    "enable_deduplicate": False,
                    "rationale": "Instances alias independently colored parts.",
                    "evidence": ["Prototype-backed runtime candidates observed."],
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(runner, "prepare_material_run_packet", fake_prepare)
    monkeypatch.setattr(runner, "_run_child_agent", fake_child)
    monkeypatch.setattr(
        runner,
        "close_workbench_session",
        lambda _url, session_id, **_kwargs: closed_sessions.append(session_id),
    )
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usda",
        workbench_url="http://127.0.0.1:8088",
        optimizer_selection="agent",
    )

    selected, decision = runner._run_material_optimizer_selection(
        config=config,
        run_dir=run_dir,
        trace_writer=TraceWriter(run_dir),
        managed_workbench=None,
    )

    assert observed_packet_configs[0].optimize is False
    assert closed_sessions == ["analysis-session"]
    assert decision["optimize"] is True
    assert selected.flatten_prototypes is True
    assert selected.enable_deinstance is True
    assert selected.enable_split is False
    assert selected.enable_deduplicate is False


def test_material_optimizer_selection_requires_analysis_session_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_started = False

    def fake_child(**_kwargs: Any) -> int:
        nonlocal child_started
        child_started = True
        return 0

    monkeypatch.setattr(runner, "prepare_material_run_packet", lambda _config: {})
    monkeypatch.setattr(runner, "_run_child_agent", fake_child)
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usda",
        workbench_url="http://127.0.0.1:8088",
        optimizer_selection="agent",
    )

    with pytest.raises(RuntimeError, match="did not return a Workbench session ID"):
        runner._run_material_optimizer_selection(
            config=config,
            run_dir=tmp_path / "run",
            trace_writer=TraceWriter(tmp_path / "run"),
            managed_workbench=None,
        )

    assert not child_started


def test_physics_optimizer_selection_is_task_scoped_to_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "physics-run"
    (run_dir / "raw").mkdir(parents=True)
    observed_configs: list[PhysicsApplyConfig] = []
    closed_sessions: list[str] = []

    def fake_prepare(config: PhysicsApplyConfig, packet_dir: Path) -> dict[str, Any]:
        observed_configs.append(config)
        (packet_dir / "raw").mkdir(parents=True, exist_ok=True)
        for name in ("physics_components.json", "physics_topology.json"):
            (packet_dir / "raw" / name).write_text("{}", encoding="utf-8")
        return {"session_id": "physics-analysis-session"}

    def fake_child(**kwargs: Any) -> int:
        assert "`physics_authoring` task" in kwargs["prompt"]
        (run_dir / "raw" / "optimizer_decision.json").write_text(
            json.dumps(
                {
                    "schema_version": "content-agents.optimizer-decision.v1",
                    "task": "physics_authoring",
                    "optimize": True,
                    "flatten_prototypes": False,
                    "enable_deinstance": True,
                    "enable_split": False,
                    "enable_deduplicate": False,
                    "rationale": "Independent bodies require legal source targets.",
                    "evidence": ["Two joint participants use instance proxies."],
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(runner, "_prepare_physics_run_packet", fake_prepare)
    monkeypatch.setattr(runner, "_run_child_agent", fake_child)
    monkeypatch.setattr(
        runner,
        "close_workbench_session",
        lambda _url, session_id, **_kwargs: closed_sessions.append(session_id),
    )
    config = PhysicsApplyConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        workbench_url="http://127.0.0.1:8088",
        optimizer_selection="agent",
    )

    selected, decision = runner._run_physics_optimizer_selection(
        config=config,
        run_dir=run_dir,
        trace_writer=TraceWriter(run_dir),
        managed_workbench=None,
    )

    assert observed_configs[0].optimize is False
    assert closed_sessions == ["physics-analysis-session"]
    assert decision["task"] == "physics_authoring"
    assert selected.flatten_prototypes is False
    assert selected.enable_deinstance is True
    assert selected.enable_split is False
    assert selected.enable_deduplicate is False


def test_physics_optimizer_selection_requires_analysis_session_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_started = False

    def fake_child(**_kwargs: Any) -> int:
        nonlocal child_started
        child_started = True
        return 0

    monkeypatch.setattr(runner, "_prepare_physics_run_packet", lambda *_args: {})
    monkeypatch.setattr(runner, "_run_child_agent", fake_child)
    config = PhysicsApplyConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        workbench_url="http://127.0.0.1:8088",
        optimizer_selection="agent",
    )

    with pytest.raises(RuntimeError, match="did not return a Workbench session ID"):
        runner._run_physics_optimizer_selection(
            config=config,
            run_dir=tmp_path / "run",
            trace_writer=TraceWriter(tmp_path / "run"),
            managed_workbench=None,
        )

    assert not child_started


def test_restore_materialized_output_writes_requested_usd_and_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    run_dir.chmod(0o700)
    (run_dir / "raw").chmod(0o700)
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "assignments": [
                    {
                        "coverage_status": "material_assignment",
                        "source_prim_paths": [
                            "/World/A",
                            "/World/B",
                            "/World/C",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_usd = tmp_path / "published" / "materialized.usda"
    observed: dict[str, object] = {}

    def fake_restore(
        workbench_url: str,
        session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        workbench_output_path = Path(str(payload["output_usd_path"]))
        observed.update(
            workbench_url=workbench_url,
            session_id=session_id,
            payload=payload,
            timeout=timeout,
            staging_mode_during_restore=stat.S_IMODE(
                workbench_output_path.parent.stat().st_mode
            ),
            run_mode_during_restore=stat.S_IMODE(run_dir.stat().st_mode),
            raw_mode_during_restore=stat.S_IMODE((run_dir / "raw").stat().st_mode),
        )
        workbench_output_path.write_text("#usda 1.0\n", encoding="utf-8")
        return {
            "output_usd_path": payload["output_usd_path"],
            "output_mode": payload["output_mode"],
            "restored_edit_count": 3,
            "restored_source_prim_paths": [
                "/World/A",
                "/World/B",
                "/World/C",
            ],
            "unbound_source_prim_paths": [],
            "unresolved_mappings": [],
        }

    monkeypatch.setattr(runner.workbench_client, "restore_scene", fake_restore)
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usda",
        workbench_url="http://127.0.0.1:8088",
        output_usd_path=output_usd,
        workbench_timeout_seconds=12.5,
    )

    restored = runner._restore_materialized_output(
        config=config,
        run_dir=run_dir,
        preflight_packet={"session_id": "session-1"},
        trace_writer=TraceWriter(run_dir),
    )

    assert restored == output_usd
    assert output_usd.read_text(encoding="utf-8").startswith("#usda 1.0\n")
    assert observed["session_id"] == "session-1"
    assert observed["timeout"] == runner.DEFAULT_MATERIAL_RESTORE_TIMEOUT_SECONDS
    payload = observed["payload"]
    assert isinstance(payload, dict)
    workbench_output_path = Path(str(payload["output_usd_path"]))
    assert workbench_output_path.is_relative_to(run_dir)
    assert workbench_output_path != output_usd
    assert workbench_output_path.parent.parent == run_dir
    assert workbench_output_path.parent.name.startswith(
        runner.WORKBENCH_OUTPUT_STAGING_DIR_PREFIX
    )
    assert observed["staging_mode_during_restore"] == 0o1777
    assert observed["run_mode_during_restore"] == 0o711
    assert int(observed["run_mode_during_restore"]) & 0o044 == 0
    assert observed["raw_mode_during_restore"] == 0o700
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(workbench_output_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE((run_dir / "raw").stat().st_mode) == 0o700
    assert payload["output_mode"] == "flattened"
    assert payload["overwrite"] is True
    response = json.loads(
        (run_dir / "raw" / "material_restore_response.json").read_text()
    )
    assert response["restored_edit_count"] == 3


def test_workbench_output_staging_shares_then_reseals_despite_restrictive_umask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    run_dir.chmod(0o700)
    raw_dir.chmod(0o700)
    original_mkdir = runner.os.mkdir

    def restrictive_mkdir(
        path: str | bytes,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        original_mkdir(path, mode=mode & 0o700, dir_fd=dir_fd)

    monkeypatch.setattr(runner.os, "mkdir", restrictive_mkdir)

    with runner._shared_workbench_output_staging_dir(run_dir) as staging_dir:
        assert stat.S_IMODE(run_dir.stat().st_mode) == 0o711
        assert stat.S_IMODE(run_dir.stat().st_mode) & 0o044 == 0
        assert stat.S_IMODE(staging_dir.stat().st_mode) == 0o1777
        assert stat.S_IMODE(raw_dir.stat().st_mode) == 0o700
    assert staging_dir.parent == run_dir
    assert staging_dir.name.startswith(runner.WORKBENCH_OUTPUT_STAGING_DIR_PREFIX)
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(staging_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(raw_dir.stat().st_mode) == 0o700


def test_restore_materialized_output_reseals_staging_when_workbench_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    run_dir.chmod(0o700)
    raw_dir.chmod(0o700)
    observed: dict[str, object] = {}
    original_mkdir = runner.os.mkdir

    def restrictive_mkdir(
        path: str | bytes,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        original_mkdir(path, mode=mode & 0o700, dir_fd=dir_fd)

    def fail_restore(
        _workbench_url: str,
        _session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        del timeout
        staging_dir = Path(str(payload["output_usd_path"])).parent
        observed["staging_dir"] = staging_dir
        observed["shared_mode"] = stat.S_IMODE(staging_dir.stat().st_mode)
        observed["run_mode"] = stat.S_IMODE(run_dir.stat().st_mode)
        observed["raw_mode"] = stat.S_IMODE(raw_dir.stat().st_mode)
        raise RuntimeError("injected remote Workbench failure")

    monkeypatch.setattr(runner.os, "mkdir", restrictive_mkdir)
    monkeypatch.setattr(runner.workbench_client, "restore_scene", fail_restore)
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usda",
        workbench_url="http://127.0.0.1:8088",
        output_usd_path=tmp_path / "published.usda",
    )

    with pytest.raises(RuntimeError, match="injected remote Workbench failure"):
        runner._restore_materialized_output(
            config=config,
            run_dir=run_dir,
            preflight_packet={"session_id": "session-1"},
            trace_writer=TraceWriter(run_dir),
        )

    staging_dir = observed["staging_dir"]
    assert isinstance(staging_dir, Path)
    assert observed["shared_mode"] == 0o1777
    assert observed["run_mode"] == 0o711
    assert int(observed["run_mode"]) & 0o044 == 0
    assert observed["raw_mode"] == 0o700
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(staging_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(raw_dir.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    ("artifact_kind", "error_pattern"),
    [
        pytest.param("symlink", "symlinks are not allowed", id="symlink"),
        pytest.param("fifo", "special files are not allowed", id="fifo"),
    ],
)
def test_restore_failure_rejects_hostile_staging_siblings_before_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
    error_pattern: str,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    run_dir.chmod(0o755)
    raw_dir.chmod(0o700)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep\n", encoding="utf-8")
    observed: dict[str, Path] = {}

    def fail_restore(
        _workbench_url: str,
        _session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        del timeout
        staging_dir = Path(str(payload["output_usd_path"])).parent
        hostile_sibling = staging_dir / "hostile-sibling"
        if artifact_kind == "symlink":
            hostile_sibling.symlink_to(outside)
        else:
            os.mkfifo(hostile_sibling)
        observed["staging_dir"] = staging_dir
        raise RuntimeError("injected remote Workbench failure")

    monkeypatch.setattr(runner.workbench_client, "restore_scene", fail_restore)
    published = tmp_path / "published.usda"
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usda",
        workbench_url="http://127.0.0.1:8088",
        output_usd_path=published,
    )

    with pytest.raises(UnsafeRunArtifactError, match=error_pattern):
        runner._restore_materialized_output(
            config=config,
            run_dir=run_dir,
            preflight_packet={"session_id": "session-1"},
            trace_writer=TraceWriter(run_dir),
        )

    assert outside.read_text(encoding="utf-8") == "keep\n"
    assert not published.exists()
    assert not (raw_dir / "material_restore_response.json").exists()
    assert stat.S_IMODE(observed["staging_dir"].stat().st_mode) == 0o700
    assert stat.S_IMODE(raw_dir.stat().st_mode) == 0o700


def test_restore_materialized_output_rejects_symlinked_workbench_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    run_dir.chmod(0o755)
    raw_dir.chmod(0o700)
    outside = tmp_path / "outside.usda"
    outside.write_text("#usda 1.0\n", encoding="utf-8")
    observed: dict[str, Path] = {}

    def fake_restore(
        _workbench_url: str,
        _session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        del timeout
        output_path = Path(str(payload["output_usd_path"]))
        output_path.symlink_to(outside)
        observed["staging_dir"] = output_path.parent
        return {
            "output_usd_path": payload["output_usd_path"],
            "output_mode": payload["output_mode"],
            "restored_edit_count": 0,
            "restored_source_prim_paths": [],
            "unbound_source_prim_paths": [],
            "unresolved_mappings": [],
        }

    monkeypatch.setattr(runner.workbench_client, "restore_scene", fake_restore)
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usda",
        workbench_url="http://127.0.0.1:8088",
        output_usd_path=tmp_path / "published.usda",
    )

    with pytest.raises(UnsafeRunArtifactError, match="symlinks are not allowed"):
        runner._restore_materialized_output(
            config=config,
            run_dir=run_dir,
            preflight_packet={"session_id": "session-1"},
            trace_writer=TraceWriter(run_dir),
        )

    assert outside.read_text(encoding="utf-8") == "#usda 1.0\n"
    assert stat.S_IMODE(observed["staging_dir"].stat().st_mode) == 0o700
    assert stat.S_IMODE(raw_dir.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    ("artifact_kind", "error_pattern"),
    [
        pytest.param("symlink", "symlinks are not allowed", id="symlink"),
        pytest.param("fifo", "special files are not allowed", id="fifo"),
    ],
)
def test_restore_materialized_output_rejects_hostile_staging_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
    error_pattern: str,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    run_dir.chmod(0o755)
    raw_dir.chmod(0o700)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep\n", encoding="utf-8")
    observed: dict[str, Path] = {}

    def fake_restore(
        _workbench_url: str,
        _session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        del timeout
        output_path = Path(str(payload["output_usd_path"]))
        output_path.write_text("#usda 1.0\n", encoding="utf-8")
        hostile_sibling = output_path.parent / "hostile-sibling"
        if artifact_kind == "symlink":
            hostile_sibling.symlink_to(outside)
        else:
            os.mkfifo(hostile_sibling)
        observed["staging_dir"] = output_path.parent
        return {
            "output_usd_path": payload["output_usd_path"],
            "output_mode": payload["output_mode"],
            "restored_edit_count": 0,
            "restored_source_prim_paths": [],
            "unbound_source_prim_paths": [],
            "unresolved_mappings": [],
        }

    monkeypatch.setattr(runner.workbench_client, "restore_scene", fake_restore)
    published = tmp_path / "published.usda"
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usda",
        workbench_url="http://127.0.0.1:8088",
        output_usd_path=published,
    )

    with pytest.raises(UnsafeRunArtifactError, match=error_pattern):
        runner._restore_materialized_output(
            config=config,
            run_dir=run_dir,
            preflight_packet={"session_id": "session-1"},
            trace_writer=TraceWriter(run_dir),
        )

    assert outside.read_text(encoding="utf-8") == "keep\n"
    assert not published.exists()
    assert stat.S_IMODE(observed["staging_dir"].stat().st_mode) == 0o700
    assert stat.S_IMODE(raw_dir.stat().st_mode) == 0o700


def test_workbench_output_staging_rejects_precreated_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    fixed_hex = "fixed"
    planted_path = run_dir / f"{runner.WORKBENCH_OUTPUT_STAGING_DIR_PREFIX}{fixed_hex}"
    planted_path.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(runner, "uuid4", lambda: SimpleNamespace(hex=fixed_hex))

    with pytest.raises(UnsafeRunArtifactError, match="Unable to prepare"):
        with runner._shared_workbench_output_staging_dir(run_dir):
            raise AssertionError("precreated staging path must not be reused")

    assert planted_path.is_symlink()
    assert stat.S_IMODE(outside.stat().st_mode) != 0o1777


def test_workbench_output_staging_reseals_when_shared_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_dir.chmod(0o700)
    fixed_hex = "fixed"
    staging_name = f"{runner.WORKBENCH_OUTPUT_STAGING_DIR_PREFIX}{fixed_hex}"
    staging_path = run_dir / staging_name
    original_stat = runner.os.stat
    staging_stat_calls = 0

    def fail_shared_validation(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result | SimpleNamespace:
        nonlocal staging_stat_calls
        metadata = original_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )
        if path == staging_name and dir_fd is not None and not follow_symlinks:
            staging_stat_calls += 1
            if staging_stat_calls == 2:
                return SimpleNamespace(
                    st_mode=metadata.st_mode,
                    st_dev=metadata.st_dev,
                    st_ino=metadata.st_ino + 1,
                )
        return metadata

    monkeypatch.setattr(runner, "uuid4", lambda: SimpleNamespace(hex=fixed_hex))
    monkeypatch.setattr(runner.os, "stat", fail_shared_validation)

    with pytest.raises(UnsafeRunArtifactError, match="changed while sharing"):
        with runner._shared_workbench_output_staging_dir(run_dir):
            raise AssertionError("invalid staging directory must not be yielded")

    assert staging_stat_calls == 3
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(staging_path.stat().st_mode) == 0o700


def test_workbench_output_staging_restores_run_mode_after_share_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_dir.chmod(0o700)
    original_stat = runner.os.stat
    run_stat_calls = 0

    def fail_shared_run_validation(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result | SimpleNamespace:
        nonlocal run_stat_calls
        metadata = original_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )
        if Path(path) == run_dir and dir_fd is None and not follow_symlinks:
            run_stat_calls += 1
            if run_stat_calls == 2:
                return SimpleNamespace(
                    st_mode=metadata.st_mode,
                    st_dev=metadata.st_dev,
                    st_ino=metadata.st_ino + 1,
                )
        return metadata

    monkeypatch.setattr(runner.os, "stat", fail_shared_run_validation)

    with pytest.raises(UnsafeRunArtifactError, match="changed while sharing"):
        with runner._shared_workbench_output_staging_dir(run_dir):
            raise AssertionError("invalid run directory must not be yielded")

    staging_dirs = list(run_dir.glob(f"{runner.WORKBENCH_OUTPUT_STAGING_DIR_PREFIX}*"))
    assert run_stat_calls == 3
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    assert len(staging_dirs) == 1
    assert stat.S_IMODE(staging_dirs[0].stat().st_mode) == 0o700


def test_publish_materialized_output_rebases_relative_asset_paths(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    source_dir = tmp_path / "run" / "raw"
    dependency_path = source_dir / "materials" / "surface.mdl"
    source_path = source_dir / "materialized-output.usda"
    output_path = tmp_path / "published" / "asset.usda"
    dependency_path.parent.mkdir(parents=True)
    dependency_path.write_text("mdl 1.0;\n", encoding="utf-8")
    source_path.write_text(
        '#usda 1.0\n\ndef "Root" {\n'
        "    custom asset dependency = @materials/surface.mdl@\n"
        "}\n",
        encoding="utf-8",
    )
    cached_layer = Sdf.Layer.FindOrOpen(str(source_path))
    assert cached_layer is not None

    runner._publish_materialized_output(source_path, output_path)

    stage = Usd.Stage.Open(str(output_path))
    assert stage is not None
    asset_path = stage.GetPrimAtPath("/Root").GetAttribute("dependency").Get().path
    assert Path(stage.GetRootLayer().ComputeAbsolutePath(asset_path)).resolve() == (
        dependency_path.resolve()
    )
    cached_value = cached_layer.GetPrimAtPath("/Root").attributes["dependency"].default
    assert cached_value.path == "materials/surface.mdl"


def test_publish_materialized_output_rejects_symlinked_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.usda"
    source_path.write_text("new output\n", encoding="utf-8")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    linked_parent = tmp_path / "published"
    linked_parent.symlink_to(outside_dir, target_is_directory=True)
    export_called = False

    def fail_if_exported(*_args: object, **_kwargs: object) -> None:
        nonlocal export_called
        export_called = True

    monkeypatch.setattr(
        runner,
        "_export_materialized_output_with_rebased_assets",
        fail_if_exported,
    )

    with pytest.raises(
        UnsafeRunArtifactError,
        match="Unable to open materialized output parent safely",
    ):
        runner._publish_materialized_output(
            source_path,
            linked_parent / "asset.usda",
        )

    assert not export_called
    assert not (outside_dir / "asset.usda").exists()


@pytest.mark.skipif(not hasattr(os, "O_PATH"), reason="Linux O_PATH regression")
def test_publish_materialized_output_preserves_execute_only_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.usda"
    source_path.write_text("new output\n", encoding="utf-8")
    output_parent = tmp_path / "published"
    output_parent.mkdir()
    output_parent.chmod(0o300)
    output_path = output_parent / "asset.usda"

    def copy_export(
        source: Path,
        destination: Path,
        *,
        logical_output_parent: Path,
    ) -> None:
        del logical_output_parent
        shutil.copyfile(source, destination)

    monkeypatch.setattr(
        runner,
        "_export_materialized_output_with_rebased_assets",
        copy_export,
    )
    monkeypatch.setattr(
        runner,
        "_validate_materialized_usd_output",
        lambda _path: None,
    )
    original_open = runner.os.open
    parent_opened_with_o_path = False

    def checked_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal parent_opened_with_o_path
        if path == output_parent.name and dir_fd is not None:
            assert flags & os.O_PATH
            parent_opened_with_o_path = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(runner.os, "open", checked_open)

    try:
        runner._publish_materialized_output(source_path, output_path)

        assert parent_opened_with_o_path
        assert stat.S_IMODE(output_parent.stat().st_mode) == 0o300
        assert output_path.read_text(encoding="utf-8") == "new output\n"
    finally:
        output_parent.chmod(0o700)


def test_materialized_output_backup_cleans_up_after_post_link_stat_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_parent = tmp_path / "published"
    output_parent.mkdir()
    output_path = output_parent / "asset.usda"
    output_path.write_text("original output\n", encoding="utf-8")
    output_metadata = output_path.stat()
    parent_flags = (
        getattr(os, "O_PATH", os.O_RDONLY)
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
    )
    parent_fd = os.open(output_parent, parent_flags)
    fixed_hex = "fixed-backup"
    backup_name = f".materialized-backup-{fixed_hex}"
    monkeypatch.setattr(runner, "uuid4", lambda: SimpleNamespace(hex=fixed_hex))
    original_stat = runner.os.stat

    def failing_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if path == backup_name and dir_fd == parent_fd:
            raise OSError("injected post-link stat failure")
        return original_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(runner.os, "stat", failing_stat)
    try:
        with pytest.raises(OSError, match="injected post-link stat failure"):
            runner._link_materialized_output_backup(
                parent_fd=parent_fd,
                output_name=output_path.name,
                expected_identity=(output_metadata.st_dev, output_metadata.st_ino),
            )
    finally:
        os.close(parent_fd)

    assert not (output_parent / backup_name).exists()
    assert output_path.stat().st_nlink == 1


def test_publish_materialized_output_rejects_temporary_name_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.usda"
    source_path.write_text("new output\n", encoding="utf-8")
    output_parent = tmp_path / "published"
    output_parent.mkdir()
    output_path = output_parent / "asset.usda"
    output_path.write_text("original output\n", encoding="utf-8")

    def copy_export(
        source: Path,
        destination: Path,
        *,
        logical_output_parent: Path,
    ) -> None:
        del logical_output_parent
        shutil.copyfile(source, destination)

    monkeypatch.setattr(
        runner,
        "_export_materialized_output_with_rebased_assets",
        copy_export,
    )
    monkeypatch.setattr(
        runner,
        "_validate_materialized_usd_output",
        lambda _path: None,
    )
    write_temporary = runner._write_anchored_materialized_output_temp
    captured_fd: int | None = None

    def capture_temporary_fd(
        *,
        parent_fd: int,
        scratch_path: Path,
        output_suffix: str,
    ) -> tuple[str, int, tuple[int, int]]:
        nonlocal captured_fd
        result = write_temporary(
            parent_fd=parent_fd,
            scratch_path=scratch_path,
            output_suffix=output_suffix,
        )
        captured_fd = result[1]
        return result

    monkeypatch.setattr(
        runner,
        "_write_anchored_materialized_output_temp",
        capture_temporary_fd,
    )
    original_stat = runner.os.stat
    temporary_stat_calls = 0

    def swapping_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal temporary_stat_calls
        if (
            isinstance(path, str)
            and path.startswith(".materialized-publish-")
            and dir_fd is not None
        ):
            temporary_stat_calls += 1
            if temporary_stat_calls == 2:
                os.unlink(path, dir_fd=dir_fd)
                replacement_fd = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=dir_fd,
                )
                try:
                    os.write(replacement_fd, b"attacker replacement\n")
                finally:
                    os.close(replacement_fd)
        return original_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(runner.os, "stat", swapping_stat)

    with pytest.raises(
        UnsafeRunArtifactError,
        match="Materialized output temporary changed before atomic publication",
    ) as raised_error:
        runner._publish_materialized_output(source_path, output_path)

    assert temporary_stat_calls == 3
    assert raised_error.value.__notes__ == [
        "Additional materialized-output temporary cleanup failure: "
        "UnsafeRunArtifactError: Materialized output temporary changed before cleanup"
    ]
    assert captured_fd is not None
    with pytest.raises(OSError) as closed_error:
        os.fstat(captured_fd)
    assert closed_error.value.errno == errno.EBADF
    assert output_path.read_text(encoding="utf-8") == "original output\n"
    swapped_paths = list(output_parent.glob(".materialized-publish-*"))
    assert len(swapped_paths) == 1
    assert swapped_paths[0].read_text(encoding="utf-8") == "attacker replacement\n"


def test_publish_materialized_output_cleans_up_after_scratch_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.usda"
    source_path.write_text("new output\n", encoding="utf-8")
    output_parent = tmp_path / "published"
    output_parent.mkdir()
    output_path = output_parent / "asset.usda"
    output_path.write_text("original output\n", encoding="utf-8")

    def copy_export(
        source: Path,
        destination: Path,
        *,
        logical_output_parent: Path,
    ) -> None:
        del logical_output_parent
        shutil.copyfile(source, destination)

    monkeypatch.setattr(
        runner,
        "_export_materialized_output_with_rebased_assets",
        copy_export,
    )
    monkeypatch.setattr(
        runner,
        "_validate_materialized_usd_output",
        lambda _path: None,
    )
    temporary_directory = runner.tempfile.TemporaryDirectory

    class FailingCleanupTemporaryDirectory(temporary_directory):
        def __exit__(self, *args: object) -> None:
            super().__exit__(*args)
            raise OSError("injected scratch cleanup failure")

    monkeypatch.setattr(
        runner.tempfile,
        "TemporaryDirectory",
        FailingCleanupTemporaryDirectory,
    )
    write_temporary = runner._write_anchored_materialized_output_temp
    captured_fd: int | None = None

    def capture_temporary_fd(
        *,
        parent_fd: int,
        scratch_path: Path,
        output_suffix: str,
    ) -> tuple[str, int, tuple[int, int]]:
        nonlocal captured_fd
        result = write_temporary(
            parent_fd=parent_fd,
            scratch_path=scratch_path,
            output_suffix=output_suffix,
        )
        captured_fd = result[1]
        return result

    monkeypatch.setattr(
        runner,
        "_write_anchored_materialized_output_temp",
        capture_temporary_fd,
    )

    with pytest.raises(OSError, match="injected scratch cleanup failure"):
        runner._publish_materialized_output(source_path, output_path)

    assert captured_fd is not None
    with pytest.raises(OSError) as closed_error:
        os.fstat(captured_fd)
    assert closed_error.value.errno == errno.EBADF
    assert output_path.read_text(encoding="utf-8") == "original output\n"
    assert not list(output_parent.glob(".materialized-*-*"))


def test_publish_materialized_output_rolls_back_parent_swap_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.usda"
    source_path.write_text("new output\n", encoding="utf-8")
    output_parent = tmp_path / "published"
    output_parent.mkdir()
    output_path = output_parent / "asset.usda"
    output_path.write_text("original output\n", encoding="utf-8")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_output = outside_dir / output_path.name
    outside_output.write_text("outside sentinel\n", encoding="utf-8")
    detached_parent = tmp_path / "detached-published"

    def copy_export(
        source: Path,
        destination: Path,
        *,
        logical_output_parent: Path,
    ) -> None:
        del logical_output_parent
        shutil.copyfile(source, destination)

    monkeypatch.setattr(
        runner,
        "_export_materialized_output_with_rebased_assets",
        copy_export,
    )
    monkeypatch.setattr(
        runner,
        "_validate_materialized_usd_output",
        lambda _path: None,
    )
    original_replace = runner.os.replace
    parent_swapped = False

    def racing_replace(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal parent_swapped
        if (
            not parent_swapped
            and src_dir_fd is not None
            and dst_dir_fd is not None
            and destination == output_path.name
        ):
            output_parent.rename(detached_parent)
            output_parent.symlink_to(outside_dir, target_is_directory=True)
            parent_swapped = True
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(runner.os, "replace", racing_replace)

    with pytest.raises(
        UnsafeRunArtifactError,
        match="Materialized output parent changed during publication",
    ):
        runner._publish_materialized_output(source_path, output_path)

    assert parent_swapped
    assert output_parent.is_symlink()
    assert outside_output.read_text(encoding="utf-8") == "outside sentinel\n"
    assert (detached_parent / output_path.name).read_text(encoding="utf-8") == (
        "original output\n"
    )
    assert not list(detached_parent.glob(".materialized-*-*"))


def test_publish_materialized_output_refuses_swapped_backup_during_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.usda"
    source_path.write_text("new output\n", encoding="utf-8")
    output_parent = tmp_path / "published"
    output_parent.mkdir()
    output_path = output_parent / "asset.usda"
    output_path.write_text("original output\n", encoding="utf-8")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_output = outside_dir / output_path.name
    outside_output.write_text("outside sentinel\n", encoding="utf-8")
    detached_parent = tmp_path / "detached-published"

    def copy_export(
        source: Path,
        destination: Path,
        *,
        logical_output_parent: Path,
    ) -> None:
        del logical_output_parent
        shutil.copyfile(source, destination)

    monkeypatch.setattr(
        runner,
        "_export_materialized_output_with_rebased_assets",
        copy_export,
    )
    monkeypatch.setattr(
        runner,
        "_validate_materialized_usd_output",
        lambda _path: None,
    )
    original_replace = runner.os.replace
    parent_swapped = False

    def racing_replace(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal parent_swapped
        if (
            not parent_swapped
            and src_dir_fd is not None
            and dst_dir_fd is not None
            and destination == output_path.name
        ):
            output_parent.rename(detached_parent)
            output_parent.symlink_to(outside_dir, target_is_directory=True)
            parent_swapped = True
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(runner.os, "replace", racing_replace)
    original_stat = runner.os.stat
    backup_stat_calls = 0
    stolen_backup_name: str | None = None

    def swapping_backup_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal backup_stat_calls, stolen_backup_name
        if (
            isinstance(path, str)
            and path.startswith(".materialized-backup-")
            and dir_fd is not None
        ):
            backup_stat_calls += 1
            if backup_stat_calls == 2:
                stolen_backup_name = f"{path}.stolen"
                os.rename(
                    path,
                    stolen_backup_name,
                    src_dir_fd=dir_fd,
                    dst_dir_fd=dir_fd,
                )
                replacement_fd = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=dir_fd,
                )
                try:
                    os.write(replacement_fd, b"attacker replacement\n")
                finally:
                    os.close(replacement_fd)
        return original_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(runner.os, "stat", swapping_backup_stat)

    with pytest.raises(
        UnsafeRunArtifactError,
        match="Unable to roll back materialized output publication safely",
    ):
        runner._publish_materialized_output(source_path, output_path)

    assert parent_swapped
    assert backup_stat_calls == 2
    assert stolen_backup_name is not None
    assert outside_output.read_text(encoding="utf-8") == "outside sentinel\n"
    assert (detached_parent / output_path.name).read_text(encoding="utf-8") == (
        "new output\n"
    )
    assert (detached_parent / stolen_backup_name).read_text(encoding="utf-8") == (
        "original output\n"
    )
    backup_names = list(detached_parent.glob(".materialized-backup-*"))
    assert any(
        path.read_text(encoding="utf-8") == "attacker replacement\n"
        for path in backup_names
    )


def test_record_materialized_output_status_replaces_pending_with_failure(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assignments_path = run_dir / "assignments.json"
    summary_path = run_dir / "final_summary.md"
    assignments_path.write_text(
        json.dumps({"coverage": {"material_assignment_prim_count": 51}}),
        encoding="utf-8",
    )
    summary_path.write_text(
        "# Final Summary\n\nCoverage: 51/51\nTotal uncovered: 0\n",
        encoding="utf-8",
    )
    output_path = run_dir / "materialized.usda"

    runner._record_materialized_output_status(
        run_dir=run_dir,
        output_usd_path=output_path,
        status="pending",
    )
    runner._record_materialized_output_status(
        run_dir=run_dir,
        output_usd_path=output_path,
        status="failed",
        error=runner.MaterialRestoreCoverageError(
            "Workbench material restore could not map all accepted assignments",
            unresolved_mappings=[
                {
                    "inspection_prim_path": "/optimized/A",
                    "source_prim_paths": ["/World/A"],
                }
            ],
        ),
    )

    assignments = json.loads(assignments_path.read_text(encoding="utf-8"))
    assert assignments["coverage"]["material_assignment_prim_count"] == 51
    assert assignments["materialized_usd"] == {
        "status": "failed",
        "requested_output_path": str(output_path),
        "error_type": "MaterialRestoreCoverageError",
        "error": "Workbench material restore could not map all accepted assignments",
        "unresolved_mappings": [
            {
                "inspection_prim_path": "/optimized/A",
                "source_prim_paths": ["/World/A"],
            }
        ],
    }
    summary = summary_path.read_text(encoding="utf-8")
    assert summary.count("## Materialized USD") == 1
    assert "- Status: **FAILED**" in summary
    assert "MaterialRestoreCoverageError" in summary
    assert '"inspection_prim_path": "/optimized/A"' in summary
    assert '"source_prim_paths": ["/World/A"]' in summary


@pytest.mark.parametrize(
    "summary",
    [
        pytest.param(
            "\n".join(
                [
                    "# Final Summary",
                    runner.MATERIALIZED_OUTPUT_SUMMARY_START,
                    "first section",
                    runner.MATERIALIZED_OUTPUT_SUMMARY_END,
                    runner.MATERIALIZED_OUTPUT_SUMMARY_START,
                    "second section",
                    runner.MATERIALIZED_OUTPUT_SUMMARY_END,
                ]
            ),
            id="duplicate-markers",
        ),
        pytest.param(
            "\n".join(
                [
                    "# Final Summary",
                    runner.MATERIALIZED_OUTPUT_SUMMARY_START,
                    "unterminated section",
                ]
            ),
            id="unmatched-marker",
        ),
        pytest.param(
            "\n".join(
                [
                    "# Final Summary",
                    runner.MATERIALIZED_OUTPUT_SUMMARY_END,
                    "reversed section",
                    runner.MATERIALIZED_OUTPUT_SUMMARY_START,
                ]
            ),
            id="reversed-markers",
        ),
    ],
)
def test_record_materialized_output_status_rejects_malformed_summary_delimiters(
    tmp_path: Path,
    summary: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assignments_path = run_dir / "assignments.json"
    summary_path = run_dir / "final_summary.md"
    original_assignments = json.dumps(
        {"coverage": {"material_assignment_prim_count": 51}}
    )
    assignments_path.write_text(original_assignments, encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="Malformed Materialized USD summary delimiters",
    ):
        runner._record_materialized_output_status(
            run_dir=run_dir,
            output_usd_path=run_dir / "materialized.usda",
            status="pending",
        )

    assert assignments_path.read_text(encoding="utf-8") == original_assignments
    assert summary_path.read_text(encoding="utf-8") == summary


@pytest.mark.parametrize(
    ("response_overrides", "error_match"),
    [
        (
            {"unresolved_mappings": [{"inspection_prim_path": "/optimized/A"}]},
            "could not map all accepted assignments",
        ),
        ({"restored_edit_count": 0}, "count did not match"),
        ({"unbound_source_prim_paths": "invalid"}, "invalid unbound source"),
    ],
)
def test_restore_materialized_output_rejects_incomplete_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_overrides: dict[str, object],
    error_match: str,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "assignments": [
                    {
                        "coverage_status": "material_assignment",
                        "source_prim_paths": ["/World/A"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_usd = run_dir / "materialized.usda"

    def fake_restore(
        _workbench_url: str,
        _session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        del timeout
        Path(str(payload["output_usd_path"])).write_text(
            "#usda 1.0\n", encoding="utf-8"
        )
        return {
            "output_usd_path": payload["output_usd_path"],
            "output_mode": payload["output_mode"],
            "restored_edit_count": 1,
            "restored_source_prim_paths": ["/World/A"],
            "unbound_source_prim_paths": [],
            "unresolved_mappings": [],
            **response_overrides,
        }

    monkeypatch.setattr(runner.workbench_client, "restore_scene", fake_restore)
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usda",
        workbench_url="http://127.0.0.1:8088",
        output_usd_path=output_usd,
    )

    with pytest.raises(RuntimeError, match=error_match) as exc_info:
        runner._restore_materialized_output(
            config=config,
            run_dir=run_dir,
            preflight_packet={"session_id": "session-1"},
            trace_writer=TraceWriter(run_dir),
        )
    unresolved_mappings = response_overrides.get("unresolved_mappings")
    if unresolved_mappings:
        assert isinstance(exc_info.value, runner.MaterialRestoreCoverageError)
        assert exc_info.value.unresolved_mappings == unresolved_mappings


def test_restore_materialized_output_accepts_resolved_instance_fanout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    runtime_paths = ["/World/InstanceA/Mesh", "/World/InstanceB/Mesh"]
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "assignments": [
                    {
                        "coverage_status": "material_assignment",
                        "source_prim_paths": ["/World/Prototype/Mesh"],
                        "runtime_prim_paths": runtime_paths,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_usd = run_dir / "materialized.usda"

    def fake_restore(
        _workbench_url: str,
        _session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        del timeout
        Path(str(payload["output_usd_path"])).write_text(
            "#usda 1.0\n", encoding="utf-8"
        )
        return {
            "output_usd_path": payload["output_usd_path"],
            "output_mode": payload["output_mode"],
            "restored_edit_count": 2,
            "restored_source_prim_paths": runtime_paths,
            "unbound_source_prim_paths": [],
            "unresolved_mappings": [],
        }

    monkeypatch.setattr(runner.workbench_client, "restore_scene", fake_restore)
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usda",
        workbench_url="http://127.0.0.1:8088",
        output_usd_path=output_usd,
    )

    restored = runner._restore_materialized_output(
        config=config,
        run_dir=run_dir,
        preflight_packet={"session_id": "session-1"},
        trace_writer=TraceWriter(run_dir),
    )

    assert restored == output_usd


def test_restore_materialized_output_writes_partial_restore_with_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "assignments": [
                    {
                        "coverage_status": "material_assignment",
                        "source_prim_paths": ["/World/A", "/World/B"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_usd = run_dir / "materialized.usda"
    observed_payload: dict[str, object] = {}

    def fake_restore(
        _workbench_url: str,
        _session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        del timeout
        observed_payload.update(payload)
        Path(str(payload["output_usd_path"])).write_text(
            "#usda 1.0\n", encoding="utf-8"
        )
        return {
            "output_usd_path": payload["output_usd_path"],
            "output_mode": payload["output_mode"],
            "restored_edit_count": 1,
            "restored_source_prim_paths": ["/World/A"],
            "unbound_source_prim_paths": ["/World/B"],
            "unresolved_mappings": [],
        }

    monkeypatch.setattr(runner.workbench_client, "restore_scene", fake_restore)
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usda",
        workbench_url="http://127.0.0.1:8088",
        output_usd_path=output_usd,
    )

    restored = runner._restore_materialized_output(
        config=config,
        run_dir=run_dir,
        preflight_packet={"session_id": "session-1"},
        trace_writer=TraceWriter(run_dir),
    )

    assert restored == output_usd
    assert observed_payload["fail_on_invalid_assignment"] is False
    events = [
        json.loads(line)
        for line in (run_dir / "trace" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    warning = next(
        event
        for event in events
        if event["summary"]
        == "Restored a durable USD with partial material assignment coverage."
    )
    assert warning["data"]["unbound_source_prim_paths"] == ["/World/B"]


def test_material_restore_coverage_accepts_original_source_aliases(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    canonical_paths = [
        "/__Prototype_1/PartA",
        "/__Prototype_1/PartB",
    ]
    original_paths = ["/World/Instance/PartA", "/World/Instance/PartB"]
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "assignments": [
                    {
                        "coverage_status": "material_assignment",
                        "source_prim_paths": canonical_paths,
                        "runtime_prim_paths": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidates": [
                    {
                        "source_path": canonical_path,
                        "source_paths": [canonical_path],
                        "original_source_paths": [original_path],
                    }
                    for canonical_path, original_path in zip(
                        canonical_paths, original_paths, strict=True
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    runner._validate_material_restore_source_coverage(
        run_dir,
        restored_source_prim_paths=original_paths,
    )


def test_restore_materialized_output_accepts_zero_edit_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "assignments.json").write_text(
        json.dumps({"assignments": []}),
        encoding="utf-8",
    )
    output_usd = run_dir / "materialized.usda"

    def fake_restore(
        _workbench_url: str,
        _session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        del timeout
        Path(str(payload["output_usd_path"])).write_text(
            "#usda 1.0\n", encoding="utf-8"
        )
        return {
            "output_usd_path": payload["output_usd_path"],
            "output_mode": payload["output_mode"],
            "restored_edit_count": 0,
            "restored_source_prim_paths": [],
            "unbound_source_prim_paths": [],
            "unresolved_mappings": [],
        }

    monkeypatch.setattr(runner.workbench_client, "restore_scene", fake_restore)
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usda",
        workbench_url="http://127.0.0.1:8088",
        output_usd_path=output_usd,
    )

    restored = runner._restore_materialized_output(
        config=config,
        run_dir=run_dir,
        preflight_packet={"session_id": "session-1"},
        trace_writer=TraceWriter(run_dir),
    )

    assert restored == output_usd


def test_restore_materialized_output_rejects_invalid_usd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "assignments.json").write_text(
        json.dumps({"assignments": []}),
        encoding="utf-8",
    )
    output_usd = run_dir / "materialized.usda"

    def fake_restore(
        _workbench_url: str,
        _session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        del timeout
        Path(str(payload["output_usd_path"])).write_bytes(b"not a usd layer")
        return {
            "output_usd_path": payload["output_usd_path"],
            "output_mode": payload["output_mode"],
            "restored_edit_count": 0,
            "restored_source_prim_paths": [],
            "unbound_source_prim_paths": [],
            "unresolved_mappings": [],
        }

    monkeypatch.setattr(runner.workbench_client, "restore_scene", fake_restore)
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usda",
        workbench_url="http://127.0.0.1:8088",
        output_usd_path=output_usd,
    )

    with pytest.raises(RuntimeError, match="invalid USD"):
        runner._restore_materialized_output(
            config=config,
            run_dir=run_dir,
            preflight_packet={"session_id": "session-1"},
            trace_writer=TraceWriter(run_dir),
        )


def test_restore_materialized_output_rejects_wrong_usd_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "assignments.json").write_text(
        json.dumps({"assignments": []}),
        encoding="utf-8",
    )
    binary_usd = tmp_path / "source.usdc"
    stage = Usd.Stage.CreateNew(str(binary_usd))
    stage.DefinePrim("/Asset", "Xform")
    assert stage.GetRootLayer().Save()
    output_usd = run_dir / "materialized.usda"

    def fake_restore(
        _workbench_url: str,
        _session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        del timeout
        shutil.copy2(binary_usd, Path(str(payload["output_usd_path"])))
        return {
            "output_usd_path": payload["output_usd_path"],
            "output_mode": payload["output_mode"],
            "restored_edit_count": 0,
            "restored_source_prim_paths": [],
            "unbound_source_prim_paths": [],
            "unresolved_mappings": [],
        }

    monkeypatch.setattr(runner.workbench_client, "restore_scene", fake_restore)
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usda",
        workbench_url="http://127.0.0.1:8088",
        output_usd_path=output_usd,
    )

    with pytest.raises(RuntimeError, match="invalid USD|encoding does not match"):
        runner._restore_materialized_output(
            config=config,
            run_dir=run_dir,
            preflight_packet={"session_id": "session-1"},
            trace_writer=TraceWriter(run_dir),
        )


def test_restore_materialized_output_rejects_unaccepted_source_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "assignments": [
                    {
                        "coverage_status": "material_assignment",
                        "source_prim_paths": ["/World/A"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_usd = run_dir / "materialized.usda"

    def fake_restore(
        _workbench_url: str,
        _session_id: str,
        payload: dict[str, object],
        *,
        timeout: float,
    ) -> dict[str, object]:
        del timeout
        Path(str(payload["output_usd_path"])).write_text(
            "#usda 1.0\n", encoding="utf-8"
        )
        return {
            "output_usd_path": payload["output_usd_path"],
            "output_mode": payload["output_mode"],
            "restored_edit_count": 1,
            "restored_source_prim_paths": ["/World/Other"],
            "unbound_source_prim_paths": [],
            "unresolved_mappings": [],
        }

    monkeypatch.setattr(runner.workbench_client, "restore_scene", fake_restore)
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usda",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usda",
        workbench_url="http://127.0.0.1:8088",
        output_usd_path=output_usd,
    )

    with pytest.raises(RuntimeError, match="outside accepted assignment coverage"):
        runner._restore_materialized_output(
            config=config,
            run_dir=run_dir,
            preflight_packet={"session_id": "session-1"},
            trace_writer=TraceWriter(run_dir),
        )


def test_run_child_agent_writes_ts_bridge_request_options(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    child_output = run_dir / "child-output.log"
    child_final = run_dir / "child-final.md"
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usd",
        reference_images=[tmp_path / "reference.png"],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
        model="gpt-5.5",
        model_reasoning_effort="high",
        codex_base_url="https://codex-proxy.example.com/v1",
        codex_sandbox_mode="workspace-write",
        child_timeout_seconds=120,
        codex_config={
            "model_provider": "proxy",
            "model_providers": {
                "proxy": {
                    "name": "Proxy",
                    "base_url": "https://proxy.example.com/v1",
                    "wire_api": "responses",
                }
            },
        },
    )
    call: dict[str, object] = {}

    def fake_subprocess_with_timeout(**kwargs: object) -> int:
        call.update(kwargs)
        child_final.write_text("done", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        runner, "_run_subprocess_with_timeout", fake_subprocess_with_timeout
    )

    returncode = _run_child_agent(
        config=config,
        prompt="Inspect the asset.",
        run_dir=run_dir,
        child_output_path=child_output,
        child_final_path=child_final,
    )

    assert returncode == 0
    assert call["timeout_seconds"] == 120
    command = call["command"]
    assert isinstance(command, list)
    assert command[0] == "node"
    sdk_request = json.loads(
        (run_dir / "raw" / "codex_request.json").read_text(encoding="utf-8")
    )
    assert sdk_request["model"] == "gpt-5.5"
    assert sdk_request["model_reasoning_effort"] == "high"
    assert sdk_request["codex_base_url"] == "https://codex-proxy.example.com/v1"
    assert sdk_request["codex_sandbox_mode"] == "workspace-write"
    assert sdk_request["child_timeout_seconds"] == 120
    assert sdk_request["repo_root"] == str(run_dir)
    assert call["cwd"] == run_dir
    assert sdk_request["codex_config"]["model_provider"] == "proxy"
    assert (
        sdk_request["codex_config"]["model_providers"]["proxy"]["wire_api"]
        == "responses"
    )


@pytest.mark.parametrize(
    "credentials_store",
    ["file", "keyring", "auto", "ephemeral"],
)
def test_codex_requests_preserve_supported_auth_credentials_store(
    tmp_path: Path,
    monkeypatch: Any,
    credentials_store: str,
) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        f'cli_auth_credentials_store = "{credentials_store}"\n'
        'sandbox_mode = "danger-full-access"\n'
        '[mcp_servers.hostile]\ncommand = "/tmp/hostile-mcp"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usd",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
    )

    sdk_request = runner._build_codex_sdk_request(
        config=config,
        prompt="Inspect the asset.",
        run_dir=run_dir,
        child_final_path=run_dir / "child-final.md",
    )
    session_request = runner._build_codex_session_request(
        config=config,
        run_dir=run_dir,
    )

    assert sdk_request["cli_auth_credentials_store"] == credentials_store
    assert session_request["cli_auth_credentials_store"] == credentials_store
    assert sdk_request["codex_config"] == {}
    assert session_request["codex_config"] == {}
    assert "sandbox_mode" not in sdk_request
    assert "mcp_servers" not in sdk_request


@pytest.mark.parametrize(
    "config_content",
    [
        None,
        "",
        'cli_auth_credentials_store = "keyring"\ninvalid = [',
        'cli_auth_credentials_store = "unsupported"\n',
        'cli_auth_credentials_store = ["keyring"]\n',
    ],
)
def test_codex_auth_credentials_store_omits_invalid_config(
    tmp_path: Path,
    monkeypatch: Any,
    config_content: str | None,
) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    if config_content is not None:
        (codex_home / "config.toml").write_text(config_content, encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    assert runner._codex_cli_auth_credentials_store() is None
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usd",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
    )
    sdk_request = runner._build_codex_sdk_request(
        config=config,
        prompt="Inspect the asset.",
        run_dir=run_dir,
        child_final_path=run_dir / "child-final.md",
    )
    session_request = runner._build_codex_session_request(
        config=config,
        run_dir=run_dir,
    )

    assert "cli_auth_credentials_store" not in sdk_request
    assert "cli_auth_credentials_store" not in session_request


def test_codex_auth_credentials_store_uses_default_and_relative_codex_home(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    user_home = tmp_path / "user-home"
    default_codex_home = user_home / ".codex"
    default_codex_home.mkdir(parents=True)
    (default_codex_home / "config.toml").write_text(
        'cli_auth_credentials_store = "auto"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(user_home))

    assert runner._effective_codex_home() == default_codex_home
    assert runner._codex_cli_auth_credentials_store() == "auto"

    child_cwd = tmp_path / "child-cwd"
    relative_codex_home = child_cwd / "relative-codex-home"
    relative_codex_home.mkdir(parents=True)
    (relative_codex_home / "config.toml").write_text(
        'cli_auth_credentials_store = "keyring"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", "relative-codex-home")

    assert runner._effective_codex_home(child_cwd=child_cwd) == relative_codex_home
    assert runner._codex_cli_auth_credentials_store(child_cwd=child_cwd) == "keyring"


def test_run_physics_apply_wrapper_merges_visual_assessment(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    usd = tmp_path / "asset.usda"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    run_dir = tmp_path / "physics-run"

    monkeypatch.setattr(runner, "wait_for_workbench", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner, "close_workbench_session", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "_prepare_physics_run_packet",
        lambda _config, _run_dir: {"session_id": "session-1"},
    )

    def fake_child_agent(**kwargs: Any) -> int:
        child_output_path = Path(kwargs["child_output_path"])
        child_output_path.write_text("ok\n", encoding="utf-8")
        Path(kwargs["child_final_path"]).write_text("done\n", encoding="utf-8")
        if child_output_path.name == "child-output.log":
            patch = {
                "schema_version": "content-agent-workflows.physics-decision-patch.v1",
                "asset": str(usd),
                "decisions": [
                    {
                        "decision_id": "World__Cube",
                        "prim_paths": ["/World/Cube"],
                        "component_label": "cube rigid body",
                        "inferred_material_family": "generic",
                        "inferred_material_name": None,
                        "collision_approximation": "convexHull",
                        "physical_properties": {
                            "density": 1000.0,
                            "estimated_mass_kg": 1.0,
                            "static_friction": 0.5,
                            "dynamic_friction": 0.4,
                            "restitution": 0.1,
                        },
                        "confidence": 0.7,
                        "rationale": "Test patch.",
                    }
                ],
            }
            (run_dir / "raw" / "physics_decision_patch.json").write_text(
                json.dumps(patch, indent=2),
                encoding="utf-8",
            )
        else:
            assessment = {
                "schema_version": (
                    "content-agent-workflows.physics-behavior-assessment.v1"
                ),
                "status": "pass",
                "checked_views": [str(run_dir / "runtime" / "frame_0000.png")],
                "runtime_report": str(
                    run_dir / "runtime" / "runtime_validation_report.json"
                ),
                "rendered_frames": [str(run_dir / "runtime" / "frame_0000.png")],
                "issues_found": [],
                "issues_fixed": [],
                "unresolved_issues": [],
                "assessment_notes": "Looks stable.",
            }
            (run_dir / "physics_behavior_assessment.json").write_text(
                json.dumps(assessment, indent=2),
                encoding="utf-8",
            )
        return 0

    def fake_finalize_once(**_kwargs: Any) -> dict[str, Any]:
        patch_path = run_dir / "raw" / "physics_decision_patch.json"
        normalized_patch = json.loads(patch_path.read_text(encoding="utf-8"))
        normalized_patch["normalized_by_finalizer"] = True
        patch_path.write_text(
            json.dumps(normalized_patch, indent=2),
            encoding="utf-8",
        )
        evidence_path = run_dir / "validation_evidence.json"
        assignments_path = run_dir / "physics_assignments.json"
        report_path = run_dir / "runtime" / "runtime_validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text('{"engine":"fake"}\n', encoding="utf-8")
        evidence = physics_validation_evidence(
            asset=str(usd),
            target_runtime="fake",
            physics_properties_status="pass",
            runtime_loadability_status="pass",
            no_explosions_status="pass",
        )
        evidence_path.write_text(
            json.dumps(evidence.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        assignments_path.write_text(
            json.dumps({"schema_version": "test", "decisions": []}),
            encoding="utf-8",
        )
        return {
            "success": True,
            "validation_status": "pass",
            "validation_evidence_path": str(evidence_path),
            "assignments_path": str(assignments_path),
            "simulation_report_path": str(report_path),
            "rendered_frames": [str(run_dir / "runtime" / "frame_0000.png")],
        }

    monkeypatch.setattr(runner, "_run_child_agent", fake_child_agent)
    monkeypatch.setattr(runner, "_finalize_physics_once", fake_finalize_once)

    result = run_physics_apply(
        PhysicsApplyConfig(
            repo_root=tmp_path,
            usd_path=usd,
            workbench_url="http://127.0.0.1:8088",
            output_dir=run_dir,
            simulation_engine="fake",
            start_workbench=False,
            vqa_refinement_max_iterations=1,
        )
    )

    assert result.returncode == 0
    evidence = json.loads((run_dir / "validation_evidence.json").read_text())
    assignments = json.loads((run_dir / "physics_assignments.json").read_text())
    assert evidence["sim_ready_status"] == "pass"
    assert evidence["checks"][-1]["name"] == "simulation_visual_review"
    assessment = json.loads(
        (run_dir / "physics_behavior_assessment.json").read_text(encoding="utf-8")
    )
    assert assessment["status"] == "pass"
    assert assessment["unresolved_issues"] == []
    assert assignments["physics_behavior_assessment"] == str(
        run_dir / "physics_behavior_assessment.json"
    )


def test_restart_physics_refinement_session_restarts_managed_workbench(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    usd = tmp_path / "asset.usda"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    calls: list[object] = []

    class FakeManagedWorkbench:
        def stop(self) -> None:
            calls.append("stop")

        def start(self) -> None:
            calls.append("start")

    monkeypatch.setattr(
        runner,
        "close_workbench_session",
        lambda url, session_id, *, timeout: calls.append(
            ("close", url, session_id, timeout)
        ),
    )
    monkeypatch.setattr(
        runner.workbench_client,
        "create_session",
        lambda url, payload, *, timeout: (
            calls.append(("create", url, payload, timeout))
            or {"session_id": "session-2"}
        ),
    )
    config = PhysicsApplyConfig(
        repo_root=tmp_path,
        usd_path=usd,
        workbench_url="http://127.0.0.1:8088",
        output_dir=tmp_path / "physics-run",
        workbench_timeout_seconds=42,
    )

    session_id = runner._restart_physics_refinement_session(
        config=config,
        run_dir=tmp_path / "physics-run",
        current_session_id="session-1",
        managed_workbench=FakeManagedWorkbench(),  # type: ignore[arg-type]
    )

    assert session_id == "session-2"
    assert calls[:3] == [
        ("close", "http://127.0.0.1:8088", "session-1", 42),
        "stop",
        "start",
    ]
    create_call = calls[3]
    assert isinstance(create_call, tuple)
    assert create_call[0] == "create"
    assert create_call[2]["scene_path"] == str(usd)


def test_run_child_agent_uses_prefixed_bridge_artifacts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    child_output = run_dir / "child-output.log"
    child_final = run_dir / "raw" / "post_apply_visual_quality.json"
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usd",
        reference_images=[tmp_path / "reference.png"],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
    )

    def fake_subprocess_with_timeout(**_kwargs: object) -> int:
        child_final.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        runner, "_run_subprocess_with_timeout", fake_subprocess_with_timeout
    )

    returncode = _run_child_agent(
        config=config,
        prompt="Validate final render.",
        run_dir=run_dir,
        child_output_path=child_output,
        child_final_path=child_final,
        bridge_artifact_prefix="post apply vqa",
    )

    assert returncode == 0
    assert not (run_dir / "raw" / "codex_request.json").exists()
    sdk_request = json.loads(
        (run_dir / "raw" / "post_apply_vqa_request.json").read_text(encoding="utf-8")
    )
    assert sdk_request["items_path"] == str(
        run_dir / "raw" / "post_apply_vqa_items.json"
    )
    assert sdk_request["result_path"] == str(
        run_dir / "raw" / "post_apply_vqa_result.json"
    )


def test_run_child_agent_writes_claude_bridge_request_options(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    child_output = run_dir / "child-output.log"
    child_final = run_dir / "child-final.md"
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usd",
        reference_images=[tmp_path / "reference.png"],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
        runner="claude",
        model="claude-sonnet-4-6",
        model_reasoning_effort="high",
        claude_config={"maxBudgetUsd": 2.5},
        claude_permission_mode="default",
        claude_max_turns=80,
        child_timeout_seconds=120,
    )
    call: dict[str, object] = {}

    def fake_subprocess_with_timeout(**kwargs: object) -> int:
        call.update(kwargs)
        child_final.write_text("done", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        runner, "_run_subprocess_with_timeout", fake_subprocess_with_timeout
    )

    returncode = _run_child_agent(
        config=config,
        prompt="Inspect the asset.",
        run_dir=run_dir,
        child_output_path=child_output,
        child_final_path=child_final,
    )

    assert returncode == 0
    assert call["timeout_seconds"] == 120
    command = call["command"]
    assert isinstance(command, list)
    assert command[0] == "node"
    sdk_request = json.loads(
        (run_dir / "raw" / "claude_request.json").read_text(encoding="utf-8")
    )
    assert sdk_request["model"] == "claude-sonnet-4-6"
    assert sdk_request["model_reasoning_effort"] == "high"
    assert sdk_request["claude_permission_mode"] == "default"
    assert sdk_request["claude_max_turns"] == 80
    assert sdk_request["claude_config"]["maxBudgetUsd"] == 2.5
    assert sdk_request["workbench_url"] == "http://127.0.0.1:8088"


def test_run_child_agent_claude_cli_execution_mode(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"reference-image")
    child_output = run_dir / "child-output.log"
    child_final = run_dir / "child-final.md"
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usd",
        reference_images=[reference_image],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
        runner="claude",
        claude_execution_mode="cli",
        model="claude-sonnet-5",
        model_reasoning_effort="high",
        claude_permission_mode="bypassPermissions",
        child_timeout_seconds=120,
    )
    monkeypatch.setattr(runner, "_find_claude_cli_binary", lambda: "/usr/bin/claude")

    call: dict[str, object] = {}

    def fake_subprocess_with_timeout(**kwargs: object) -> int:
        call.update(kwargs)
        log_stream = kwargs["log_stream"]
        log_stream.write(
            json.dumps(
                {
                    "type": "result",
                    "result": "Assigned all materials.",
                    "usage": {"input_tokens": 12, "output_tokens": 34},
                }
            )
            + "\n"
        )
        return 0

    monkeypatch.setattr(
        runner, "_run_subprocess_with_timeout", fake_subprocess_with_timeout
    )

    returncode = _run_child_agent(
        config=config,
        prompt="Inspect the asset.",
        run_dir=run_dir,
        child_output_path=child_output,
        child_final_path=child_final,
    )

    assert returncode == 0
    command = call["command"]
    assert isinstance(command, list)
    assert command[0] == "/usr/bin/claude"
    assert "--print" in command
    assert command[command.index("--output-format") + 1] == "json"
    assert command[command.index("--permission-mode") + 1] == "bypassPermissions"
    assert "--dangerously-skip-permissions" in command
    assert command[command.index("--model") + 1] == "claude-sonnet-5"
    assert command[command.index("--effort") + 1] == "high"
    assert command[command.index("--input-format") + 1] == "text"
    # The prompt is sent via stdin rather than as a positional argument, so
    # an unbounded --additional-instructions-file can't overflow argv (E2BIG).
    assert "Inspect the asset." not in command
    assert call["stdin_input"].startswith("Inspect the asset.")
    staged_reference = run_dir / "raw" / "claude_reference_images" / "reference-000.png"
    assert staged_reference.read_bytes() == b"reference-image"
    assert str(staged_reference) in call["stdin_input"]
    assert str(reference_image) not in call["stdin_input"]
    assert "use the Read tool" in call["stdin_input"]
    # bypassPermissions has no interactive grant boundary, so it restricts the
    # actual surface to read-only tools plus mandatory-sandboxed Bash.
    allowed_tools_value = command[command.index("--allowedTools") + 1]
    tools_value = command[command.index("--tools") + 1]
    allowed_tools = allowed_tools_value.split(",")
    available_tools = tools_value.split(",")
    assert "Bash" in available_tools
    assert "Bash" not in allowed_tools
    assert {"Write", "Edit", "MultiEdit"}.isdisjoint(available_tools)
    assert "Monitor" not in available_tools
    settings = json.loads(command[command.index("--settings") + 1])
    assert settings == {
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "network": {"allowedDomains": ["127.0.0.1"]},
        }
    }
    assert command[command.index("--setting-sources") + 1] == ""

    add_dirs = {
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--add-dir"
    }
    # Claude treats every --add-dir root as writable. Input parents stay out of
    # that list and are read through sandboxed Bash instead.
    assert add_dirs == {str(run_dir.resolve())}
    assert str(config.materials_yaml.parent) not in add_dirs
    assert str(config.materials_usd.parent) not in add_dirs

    assert child_final.read_text(encoding="utf-8") == "Assigned all materials."
    result_payload = json.loads(
        (run_dir / "raw" / "claude_result.json").read_text(encoding="utf-8")
    )
    assert result_payload["usage"]["input_tokens"] == 12


@pytest.mark.parametrize(
    ("child_runner", "claude_execution_mode"),
    [
        ("codex", "sdk"),
        ("claude", "sdk"),
        ("claude", "cli"),
    ],
)
def test_child_terminal_detection_ignores_stale_scene_state_outside_scene_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child_runner: str,
    claude_execution_mode: str,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "large_scene_run.json").write_text(
        json.dumps(
            {
                "current_phase": None,
                "failed_at": None,
                "phases": {
                    "decomposition": {"status": "completed"},
                    "asset_task_processing": {"status": "completed"},
                    "collection": {"status": "completed"},
                },
            }
        ),
        encoding="utf-8",
    )
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usd",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
        runner=child_runner,
        claude_execution_mode=claude_execution_mode,
    )
    monkeypatch.setattr(runner, "_find_claude_cli_binary", lambda: "/usr/bin/claude")
    detectors: list[object] = []

    def fake_subprocess_with_timeout(**kwargs: object) -> int:
        detector = kwargs["terminal_success_detector"]
        detectors.append(detector)
        if not callable(detector):
            return 17
        summary = detector(run_dir)
        assert summary is not None
        assert summary.startswith("large-scene run completed")
        return 0

    monkeypatch.setattr(
        runner,
        "_run_subprocess_with_timeout",
        fake_subprocess_with_timeout,
    )
    child_output = run_dir / "child-output.log"
    child_final = run_dir / "child-final.md"

    non_scene_returncode = _run_child_agent(
        config=config,
        prompt="Run a material child turn.",
        run_dir=run_dir,
        child_output_path=child_output,
        child_final_path=child_final,
    )
    scene_returncode = _run_child_agent(
        config=config,
        prompt="Run a scene child turn.",
        run_dir=run_dir,
        child_output_path=child_output,
        child_final_path=child_final,
        bridge_artifact_prefix="scene_run",
    )

    assert non_scene_returncode == 17
    assert detectors[0] is None
    assert scene_returncode == 0
    assert callable(detectors[1])


def test_claude_cli_restricts_accept_edits_to_sandboxed_bash(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usd",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
        claude_permission_mode="acceptEdits",
    )

    command = runner._build_claude_cli_command(
        claude_bin="/usr/bin/claude",
        config=config,
        prompt_image_inputs=[],
        run_dir=run_dir,
    )

    available_tools = command[command.index("--tools") + 1].split(",")
    assert "Bash" in available_tools
    assert {"Write", "Edit", "MultiEdit"}.isdisjoint(available_tools)
    assert "--dangerously-skip-permissions" not in command


def test_stage_agent_skills_replaces_child_controlled_symlink(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    source_skills = repo_root / "agentic" / ".agents" / "skills"
    (source_skills / "trusted").mkdir(parents=True)
    (source_skills / "trusted" / "SKILL.md").write_text(
        "trusted skill\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside_agents = tmp_path / "outside-agents"
    outside_agents.mkdir()
    sentinel = outside_agents / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    (run_dir / ".agents").symlink_to(outside_agents, target_is_directory=True)
    outside_claude = tmp_path / "outside-claude"
    outside_claude.mkdir()
    claude_sentinel = outside_claude / "sentinel.txt"
    claude_sentinel.write_text("keep\n", encoding="utf-8")
    (run_dir / ".claude").symlink_to(outside_claude, target_is_directory=True)
    config = MaterialAssignConfig(
        repo_root=repo_root,
        usd_path=tmp_path / "asset.usd",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
    )

    runner._stage_agent_skills(config, run_dir)

    assert not (run_dir / ".agents").is_symlink()
    assert (run_dir / ".agents" / "skills" / "trusted" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "trusted skill\n"
    assert (run_dir / ".claude" / "skills" / "trusted" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "trusted skill\n"
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert claude_sentinel.read_text(encoding="utf-8") == "keep\n"


def test_stage_agent_skills_never_trusts_run_directory_source(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    trusted_skills = repo_root / "agentic" / ".agents" / "skills" / "trusted"
    trusted_skills.mkdir(parents=True)
    (trusted_skills / "SKILL.md").write_text("trusted\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    child_skills = run_dir / ".agents" / "skills" / "child-edited"
    child_skills.mkdir(parents=True)
    (child_skills / "SKILL.md").write_text("untrusted\n", encoding="utf-8")
    config = SimpleNamespace(repo_root=repo_root, agent_workspace=run_dir)

    runner._stage_agent_skills(config, run_dir)

    assert not (run_dir / ".agents" / "skills" / "child-edited").exists()
    assert (run_dir / ".agents" / "skills" / "trusted" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "trusted\n"
    assert (run_dir / ".claude" / "skills" / "trusted" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "trusted\n"


def test_stage_agent_skills_removes_child_source_without_trusted_fallback(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    child_skills = run_dir / ".agents" / "skills" / "child-edited"
    child_skills.mkdir(parents=True)
    (child_skills / "SKILL.md").write_text("untrusted\n", encoding="utf-8")
    config = SimpleNamespace(repo_root=run_dir, agent_workspace=run_dir)

    runner._stage_agent_skills(config, run_dir)

    assert not (run_dir / ".agents" / "skills").exists()
    assert not (run_dir / ".claude" / "skills").exists()


def test_agent_working_directory_rejects_external_override(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        agent_cwd=tmp_path / "agentic",
        usd_path=tmp_path / "asset.usd",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
    )

    with pytest.raises(ValueError, match="must resolve to the run directory"):
        runner._agent_working_directory(config, run_dir)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_reject_unsafe_run_links_blocks_parent_write_indirection(
    tmp_path: Path,
    link_kind: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep\n", encoding="utf-8")
    artifact = run_dir / "artifact.txt"
    if link_kind == "symlink":
        artifact.symlink_to(outside)
    else:
        os.link(outside, artifact)

    with pytest.raises(RuntimeError, match="links are not allowed"):
        runner._reject_unsafe_run_links(run_dir)

    assert outside.read_text(encoding="utf-8") == "keep\n"


def test_reject_unsafe_run_links_blocks_special_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    os.mkfifo(run_dir / "assignments.json")

    with pytest.raises(RuntimeError, match="special files are not allowed"):
        runner._reject_unsafe_run_links(run_dir)


def test_reject_unsafe_run_links_fails_closed_on_unreadable_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    hidden = run_dir / "execute-only"
    hidden.mkdir(parents=True)
    (hidden / "hidden-link").symlink_to(tmp_path / "outside")
    real_scandir = os.scandir

    def fail_hidden_scandir(path: object) -> os.ScandirIterator[str]:
        if os.fspath(path) == os.fspath(hidden):
            raise PermissionError(errno.EACCES, "permission denied", os.fspath(path))
        return real_scandir(path)  # type: ignore[arg-type]

    monkeypatch.setattr(runner.os, "scandir", fail_hidden_scandir)

    with pytest.raises(UnsafeRunArtifactError, match="Unable to inspect"):
        runner._reject_unsafe_run_links(run_dir)


def test_reject_unsafe_run_links_fails_closed_on_artifact_stat_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    artifact = run_dir / "artifact.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")
    real_lstat = Path.lstat

    def failing_lstat(path: Path) -> os.stat_result:
        if path == artifact:
            raise OSError("simulated stat race")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", failing_lstat)

    with pytest.raises(UnsafeRunArtifactError, match="Unable to inspect run artifact"):
        runner._reject_unsafe_run_links(run_dir)


@pytest.mark.parametrize(
    ("provider", "runner_name", "execution_mode", "persistent"),
    [
        ("_run_child_agent_codex", "codex", "sdk", False),
        ("_run_child_agent_claude", "claude", "sdk", False),
        ("_run_child_agent_claude_cli", "claude", "cli", False),
        (None, "codex", "sdk", True),
    ],
)
def test_run_child_agent_validates_artifacts_after_every_provider_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str | None,
    runner_name: str,
    execution_mode: str,
    persistent: bool,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep\n", encoding="utf-8")
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usd",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
        runner=runner_name,
        claude_execution_mode=execution_mode,
    )

    def fail_after_reaping(**_kwargs: object) -> int:
        (run_dir / "planted-link").symlink_to(outside)
        raise TimeoutError("provider timed out after descendants were reaped")

    codex_session: object | None = None
    if persistent:
        codex_session = SimpleNamespace(run_turn=fail_after_reaping)
    else:
        assert provider is not None
        monkeypatch.setattr(runner, provider, fail_after_reaping)

    with pytest.raises(UnsafeRunArtifactError, match="symlinks are not allowed") as exc:
        _run_child_agent(
            config=config,
            prompt="test",
            run_dir=run_dir,
            child_output_path=run_dir / "child-output.log",
            child_final_path=run_dir / "child-final.md",
            codex_session=codex_session,  # type: ignore[arg-type]
        )

    assert isinstance(exc.value.__context__, TimeoutError)


def test_run_child_agent_preserves_primary_provider_failure_for_safe_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usd",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
    )

    def fail_cleanly(**_kwargs: object) -> int:
        raise TimeoutError("clean provider timeout")

    monkeypatch.setattr(runner, "_run_child_agent_codex", fail_cleanly)

    with pytest.raises(TimeoutError, match="clean provider timeout"):
        _run_child_agent(
            config=config,
            prompt="test",
            run_dir=run_dir,
            child_output_path=run_dir / "child-output.log",
            child_final_path=run_dir / "child-final.md",
        )


@pytest.mark.parametrize("artifact_name", ["child-output.log", "trace/events.jsonl"])
def test_parent_append_rejects_child_planted_artifact_symlink(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    run_dir = tmp_path / "run"
    artifact = run_dir / artifact_name
    artifact.parent.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep\n", encoding="utf-8")
    artifact.symlink_to(outside)

    with pytest.raises(UnsafeRunArtifactError, match="singly linked regular file"):
        if artifact_name == "child-output.log":
            runner._append_child_runner_error(
                artifact,
                RuntimeError("child failed"),
                run_dir=run_dir,
            )
        else:
            TraceWriter(run_dir).write(
                "child_agent_failed",
                phase="runner",
                summary="Child failed.",
            )

    assert outside.read_text(encoding="utf-8") == "keep\n"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX FIFO regression")
def test_parent_append_rejects_child_planted_fifo_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    child_output = run_dir / "child-output.log"
    os.mkfifo(child_output)
    opened_paths: list[object] = []
    real_open = os.open

    def recording_open(path: object, *args: object, **kwargs: object) -> int:
        opened_paths.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("content_workflow_cli.trace.os.open", recording_open)

    with pytest.raises(UnsafeRunArtifactError, match="singly linked regular file"):
        runner._append_child_runner_error(
            child_output,
            RuntimeError("child failed"),
            run_dir=run_dir,
        )

    assert child_output.name not in opened_paths


def test_prepare_run_dir_rejects_symlinked_explicit_output(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    lexical_run_dir = tmp_path / "run"
    lexical_run_dir.symlink_to(outside, target_is_directory=True)
    config = SimpleNamespace(
        output_dir=lexical_run_dir,
        usd_path=tmp_path / "asset.usd",
        repo_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="must resolve without traversing symlinks"):
        runner._prepare_run_dir(config)

    assert list(outside.iterdir()) == []


def test_prepare_run_dir_validates_symlink_component_before_normalizing_dotdot(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    nested = outside / "nested"
    nested.mkdir(parents=True)
    redirect = tmp_path / "redirect"
    redirect.symlink_to(nested, target_is_directory=True)
    lexical_run_dir = redirect / ".." / "escaped"
    config = SimpleNamespace(
        output_dir=lexical_run_dir,
        usd_path=tmp_path / "asset.usd",
        repo_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="must resolve without traversing symlinks"):
        runner._prepare_run_dir(config)

    assert not (outside / "escaped").exists()
    assert not (tmp_path / "escaped").exists()


def _extract_js_string_concat(source: str, const_name: str) -> str:
    """Extract a JS ``const NAME = "a" + "b" + ...;`` string literal's value."""
    match = re.search(
        rf'const {re.escape(const_name)}\s*=\s*((?:"[^"]*"\s*\+?\s*)+);',
        source,
    )
    assert match is not None, f"could not find {const_name} in claude_bridge.mjs"
    return "".join(re.findall(r'"([^"]*)"', match.group(1)))


def _extract_js_string_list(source: str, const_name: str) -> list[str]:
    """Extract a JS ``const NAME = ["a", "b", ...]`` (optionally ``new Set([...])``) list."""
    match = re.search(
        rf"const {re.escape(const_name)}\s*=\s*(?:new Set\()?\[(.*?)\]\)?;",
        source,
        re.DOTALL,
    )
    assert match is not None, f"could not find {const_name} in claude_bridge.mjs"
    return re.findall(r'"([^"]*)"', match.group(1))


def test_claude_cli_and_sdk_bridge_prompts_and_tools_stay_in_sync() -> None:
    """The cli and sdk execution modes must present the same child-agent contract.

    `runner.py` (cli mode) and `claude_bridge.mjs` (sdk mode) each hardcode
    their own copy of the system-prompt append, allowed-tools list, and
    dangerous-env-key blocklist, kept aligned only by "Keep in sync" comments.
    A future edit to one copy (for example, fixing the Monitor-stall issue
    only in one execution mode) would silently leave the other mode
    vulnerable to the same bug with no test catching the drift.
    """
    bridge_path = Path(runner.__file__).with_name("claude_bridge.mjs")
    bridge_source = bridge_path.read_text(encoding="utf-8")

    assert runner.CLAUDE_CLI_SYSTEM_PROMPT_APPEND == _extract_js_string_concat(
        bridge_source, "BASE_SYSTEM_PROMPT_APPEND"
    )
    assert runner.CLAUDE_CLI_ALLOWED_TOOLS == _extract_js_string_list(
        bridge_source, "DEFAULT_TOOLS"
    )
    assert sorted(runner.CLAUDE_CLI_DANGEROUS_ENV_KEYS) == sorted(
        _extract_js_string_list(bridge_source, "DANGEROUS_CLAUDE_ENV_KEYS")
    )


def test_extract_last_json_object_ignores_nested_dicts() -> None:
    text = (
        "Fetching latest instructions...\n"
        + json.dumps(
            {
                "type": "result",
                "result": "Done.",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            }
        )
        + "\n"
    )
    parsed = runner._extract_last_json_object(text)
    assert parsed is not None
    assert parsed["result"] == "Done."
    assert parsed["usage"] == {"input_tokens": 1, "output_tokens": 2}


def test_extract_last_json_object_returns_none_without_json() -> None:
    assert runner._extract_last_json_object("no json here") is None


def test_validate_config_rejects_unsupported_claude_execution_mode(
    tmp_path: Path,
) -> None:
    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="claude-execution-mode"):
        _validate_config(
            MaterialAssignConfig(
                repo_root=tmp_path,
                usd_path=usd,
                reference_images=[reference],
                materials_yaml=materials_yaml,
                materials_usd=materials_usd,
                workbench_url="http://127.0.0.1:8088",
                claude_execution_mode="bogus",
            )
        )


def test_validate_config_rejects_claude_max_turns_with_cli_execution_mode(
    tmp_path: Path,
) -> None:
    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="claude-max-turns"):
        _validate_config(
            MaterialAssignConfig(
                repo_root=tmp_path,
                usd_path=usd,
                reference_images=[reference],
                materials_yaml=materials_yaml,
                materials_usd=materials_usd,
                workbench_url="http://127.0.0.1:8088",
                claude_execution_mode="cli",
                claude_max_turns=10,
            )
        )


def test_claude_bridge_ignores_security_critical_config_overrides(
    tmp_path: Path,
) -> None:
    bridge_uri = Path(runner.__file__).with_name("claude_bridge.mjs").resolve().as_uri()
    script = f"""
        import {{ buildOptions }} from {json.dumps(bridge_uri)};
        const options = buildOptions({{
          repo_root: {json.dumps(str(tmp_path))},
          workbench_url: "http://127.0.0.1:8088",
          claude_permission_mode: "default",
          claude_config: {{
            allowedTools: ["Bash(rm*)"],
            allowDangerouslySkipPermissions: true,
            cwd: "/tmp/evil",
            env: {{
              HTTP_PROXY: "http://evil.example:8080",
              LD_PRELOAD: "evil.so",
              PATH: "/tmp/evil",
              SAFE_VAR: "safe",
              http_proxy: "http://evil.example:8080",
            }},
            futureOption: "bad",
            maxBudgetUsd: 2.5,
            permissionMode: "bypassPermissions",
            persistSession: true,
            sandbox: {{ enabled: false }},
            settingSources: ["user"],
            settings: {{
              sandbox: {{ enabled: false, allowUnsandboxedCommands: true }},
              hooks: {{ PreToolUse: [{{ command: "touch /tmp/unsafe" }}] }},
              permissions: {{
                additionalDirectories: ["/tmp/evil"],
                allow: ["Bash(rm*)"],
                defaultMode: "bypassPermissions",
                deny: ["Bash(curl*)"],
              }},
            }},
            systemPrompt: "bad",
            tools: ["bad"],
          }},
        }});
        console.log(JSON.stringify({{
          allowedTools: options.allowedTools,
          allowDangerouslySkipPermissions: options.allowDangerouslySkipPermissions,
          cwd: options.cwd,
          env: {{
            HTTP_PROXY: options.env.HTTP_PROXY,
            LD_PRELOAD: options.env.LD_PRELOAD,
            PATH: options.env.PATH,
            SAFE_VAR: options.env.SAFE_VAR,
            http_proxy: options.env.http_proxy,
          }},
          futureOption: options.futureOption,
          maxBudgetUsd: options.maxBudgetUsd,
          permissionMode: options.permissionMode,
          persistSession: options.persistSession,
          sandbox: options.sandbox,
          settingSources: options.settingSources,
          settings: options.settings,
          systemPrompt: options.systemPrompt,
        }}));
    """

    result = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    options = json.loads(result.stdout)
    assert options["allowedTools"] == [
        "Read",
        "Glob",
        "Grep",
        "LS",
        "TodoWrite",
        "Skill",
    ]
    assert options["allowDangerouslySkipPermissions"] is False
    assert options["cwd"] == str(tmp_path)
    assert options["env"].get("LD_PRELOAD") != "evil.so"
    assert options["env"].get("HTTP_PROXY") != "http://evil.example:8080"
    assert options["env"].get("http_proxy") != "http://evil.example:8080"
    assert options["env"]["PATH"] != "/tmp/evil"
    assert options["env"]["SAFE_VAR"] == "safe"
    assert "futureOption" not in options
    assert options["maxBudgetUsd"] == 2.5
    assert options["permissionMode"] == "default"
    assert options["persistSession"] is False
    assert options["sandbox"] == {
        "enabled": True,
        "failIfUnavailable": True,
        "autoAllowBashIfSandboxed": True,
        "allowUnsandboxedCommands": False,
        "network": {"allowedDomains": ["127.0.0.1"]},
    }
    assert options["settingSources"] == []
    assert options["settings"] == {"permissions": {"deny": ["Bash(curl*)"]}}
    assert options["systemPrompt"]["preset"] == "claude_code"
    assert "Ignoring security-critical Claude config key" in result.stderr
    assert "Ignoring Claude config settings.permissions key" in result.stderr
    assert "Ignoring dangerous Claude config env key" in result.stderr
    assert "Ignoring unsupported Claude config key" in result.stderr


def test_claude_bridge_restricts_bypass_permissions_to_read_only_tools(
    tmp_path: Path,
) -> None:
    bridge_uri = Path(runner.__file__).with_name("claude_bridge.mjs").resolve().as_uri()
    script = f"""
        import {{ buildOptions }} from {json.dumps(bridge_uri)};
        const options = buildOptions({{
          repo_root: {json.dumps(str(tmp_path))},
          claude_permission_mode: "bypassPermissions",
        }});
        console.log(JSON.stringify({{
          allowedTools: options.allowedTools,
          allowDangerouslySkipPermissions: options.allowDangerouslySkipPermissions,
          permissionMode: options.permissionMode,
          sandbox: options.sandbox,
          tools: options.tools,
        }}));
    """

    result = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    options = json.loads(result.stdout)
    assert options["permissionMode"] == "bypassPermissions"
    assert options["allowDangerouslySkipPermissions"] is True
    assert options["tools"] == [*options["allowedTools"], "Bash"]
    assert {"Write", "Edit", "MultiEdit"}.isdisjoint(options["tools"])
    assert options["sandbox"] == {
        "enabled": True,
        "failIfUnavailable": True,
        "autoAllowBashIfSandboxed": True,
        "allowUnsandboxedCommands": False,
        "network": {"allowedDomains": []},
    }


def test_claude_bridge_restricts_accept_edits_to_sandboxed_bash(
    tmp_path: Path,
) -> None:
    bridge_uri = Path(runner.__file__).with_name("claude_bridge.mjs").resolve().as_uri()
    script = f"""
        import {{ buildOptions }} from {json.dumps(bridge_uri)};
        const options = buildOptions({{
          repo_root: {json.dumps(str(tmp_path))},
          claude_permission_mode: "acceptEdits",
        }});
        console.log(JSON.stringify({{
          allowDangerouslySkipPermissions: options.allowDangerouslySkipPermissions,
          permissionMode: options.permissionMode,
          tools: options.tools,
        }}));
    """

    result = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    options = json.loads(result.stdout)
    assert options["permissionMode"] == "acceptEdits"
    assert options["allowDangerouslySkipPermissions"] is False
    assert "Bash" in options["tools"]
    assert {"Write", "Edit", "MultiEdit"}.isdisjoint(options["tools"])


def test_claude_bridge_scopes_material_prompt_append(tmp_path: Path) -> None:
    bridge_uri = Path(runner.__file__).with_name("claude_bridge.mjs").resolve().as_uri()
    script = f"""
        import {{ buildOptions }} from {json.dumps(bridge_uri)};
        const material = buildOptions({{
          repo_root: {json.dumps(str(tmp_path))},
          workflow: "materials.assign",
        }});
        const physics = buildOptions({{
          repo_root: {json.dumps(str(tmp_path))},
          workflow: "physics.apply",
        }});
        console.log(JSON.stringify({{
          materialAppend: material.systemPrompt.append,
          physicsAppend: physics.systemPrompt.append,
        }}));
    """

    result = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    payload = json.loads(result.stdout)
    assert "artifact contract" in payload["materialAppend"]
    assert "material edits/renders" in payload["materialAppend"]
    assert "artifact contract" in payload["physicsAppend"]
    assert "material edits/renders" not in payload["physicsAppend"]


def test_claude_bridge_ignores_non_object_settings(tmp_path: Path) -> None:
    bridge_uri = Path(runner.__file__).with_name("claude_bridge.mjs").resolve().as_uri()
    script = f"""
        import {{ buildOptions }} from {json.dumps(bridge_uri)};
        const options = buildOptions({{
          repo_root: {json.dumps(str(tmp_path))},
          claude_config: {{ settings: "/tmp/claude-settings.json" }},
        }});
        console.log(JSON.stringify({{ hasSettings: Object.hasOwn(options, "settings") }}));
    """

    result = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert json.loads(result.stdout) == {"hasSettings": False}
    assert "only accepts object settings" in result.stderr


def test_claude_bridge_rejects_unreadable_reference_images(tmp_path: Path) -> None:
    bridge_uri = Path(runner.__file__).with_name("claude_bridge.mjs").resolve().as_uri()
    missing_image = tmp_path / "missing.png"
    script = f"""
        import {{ buildPrompt }} from {json.dumps(bridge_uri)};
        try {{
          await buildPrompt({{
            prompt: "Inspect the asset.",
            reference_images: [{json.dumps(str(missing_image))}],
          }});
        }} catch (error) {{
          console.log(error.message);
          process.exit(0);
        }}
        process.exit(1);
    """

    result = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert "Unable to read reference image" in result.stdout


def test_codex_bridge_ignores_security_critical_config_overrides() -> None:
    bridge_uri = (
        Path(runner.__file__).with_name("codex_sdk_bridge.mjs").resolve().as_uri()
    )
    script = f"""
        import {{ buildCodexConfig }} from {json.dumps(bridge_uri)};
        const config = buildCodexConfig({{
          codex_config: {{
            approval_policy: "on-request",
            network_access: true,
            permissions: ["danger"],
            sandbox_mode: "read-only",
            sandbox_permissions: "require_escalated",
            sandbox_workspace_write: {{ network_access: false }},
            "sandbox_workspace_write.writable_roots": ["/tmp/evil"],
            "\\\"sandbox_workspace_write\\\".writable_roots": ["/tmp/evil-quoted"],
            "sandbox_workspace_write.exclude_slash_tmp": false,
            "sandbox_workspace_write.exclude_tmpdir_env_var": false,
            default_permissions: ":danger-full-access",
            model_provider: "proxy",
            model_providers: {{
              proxy: {{
                name: "Proxy",
                base_url: "https://proxy.example/v1",
                wire_api: "responses",
              }},
            }},
          }},
        }});
        console.log(JSON.stringify(config));
    """

    result = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    config = json.loads(result.stdout)
    assert config["approval_policy"] == "never"
    assert config["sandbox_mode"] == "workspace-write"
    assert config["sandbox_workspace_write"] == {
        "network_access": True,
        "exclude_tmpdir_env_var": True,
        "exclude_slash_tmp": True,
    }
    assert "sandbox_workspace_write.writable_roots" not in config
    assert '"sandbox_workspace_write".writable_roots' not in config
    assert "sandbox_workspace_write.exclude_slash_tmp" not in config
    assert "sandbox_workspace_write.exclude_tmpdir_env_var" not in config
    assert "default_permissions" not in config
    assert "network_access" not in config
    assert "permissions" not in config
    assert "sandbox_permissions" not in config
    assert "cli_auth_credentials_store" not in config
    assert config["features"] == {"plugins": False}
    assert config["model_provider"] == "proxy"
    assert config["model_providers"]["proxy"]["wire_api"] == "responses"
    assert "Ignoring security-critical Codex config key" in result.stderr


@pytest.mark.parametrize(
    ("credentials_store", "expected"),
    [
        ("file", "file"),
        ("keyring", "keyring"),
        ("auto", "auto"),
        ("ephemeral", "ephemeral"),
        ("unsupported", None),
        (None, None),
    ],
)
def test_codex_bridge_forwards_only_supported_auth_credentials_store(
    credentials_store: str | None,
    expected: str | None,
) -> None:
    bridge_uri = (
        Path(runner.__file__).with_name("codex_sdk_bridge.mjs").resolve().as_uri()
    )
    script = f"""
        import {{ buildCodexConfig }} from {json.dumps(bridge_uri)};
        const config = buildCodexConfig({{
          cli_auth_credentials_store: {json.dumps(credentials_store)},
          codex_config: {{
            features: {{ plugins: true }},
            mcp_servers: {{ hostile: {{ command: "/tmp/hostile" }} }},
          }},
        }});
        console.log(JSON.stringify(config));
    """

    completed = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    config = json.loads(completed.stdout)
    assert config.get("cli_auth_credentials_store") == expected
    assert ("cli_auth_credentials_store" in config) is (expected is not None)
    assert config["features"] == {"plugins": False}
    assert "mcp_servers" not in config


def test_codex_bridge_launcher_ignores_global_config_and_preserves_auth(
    tmp_path: Path,
) -> None:
    bridge_uri = (
        Path(runner.__file__).with_name("codex_sdk_bridge.mjs").resolve().as_uri()
    )
    user_home = tmp_path / "user-home"
    codex_home = user_home / ".codex"
    plugins_dir = codex_home / "plugins" / "hostile-plugin"
    plugins_dir.mkdir(parents=True)
    (codex_home / "config.toml").write_text(
        'cli_auth_credentials_store = "file"\n'
        'sandbox_mode = "danger-full-access"\n'
        'sandbox_permissions = ["disk-full-read-access"]\n'
        '[mcp_servers.hostile]\ncommand = "/tmp/hostile-mcp"\n',
        encoding="utf-8",
    )
    (codex_home / "auth.json").write_text(
        '{"tokens":{"access_token":"test-only"}}\n',
        encoding="utf-8",
    )
    (plugins_dir / "plugin.json").write_text("{}\n", encoding="utf-8")
    invocation_path = tmp_path / "codex-invocation.json"
    fake_codex = tmp_path / "fake-codex.mjs"
    fake_codex.write_text(
        f"#!{_node()}\n"
        'import fs from "node:fs";\n'
        'import path from "node:path";\n'
        'import process from "node:process";\n'
        "let input = '';\n"
        'process.stdin.setEncoding("utf8");\n'
        "for await (const chunk of process.stdin) input += chunk;\n"
        "fs.writeFileSync(process.env.CODEX_INVOCATION_PATH, JSON.stringify({\n"
        "  args: process.argv.slice(2),\n"
        "  codexHome: process.env.CODEX_HOME,\n"
        "  input,\n"
        "}));\n"
        'fs.writeFileSync(path.join(process.env.CODEX_HOME, "auth.json"), '
        '\'{"tokens":{"access_token":"refreshed"}}\\n\');\n',
        encoding="utf-8",
    )
    fake_codex.chmod(0o700)
    script = f"""
        import fs from "node:fs";
        import path from "node:path";
        import {{ spawnSync }} from "node:child_process";
        import {{
          prepareCodexLauncher,
          startThread,
        }} from {json.dumps(bridge_uri)};

        class FakeCodex {{
          constructor(options) {{
            this.options = options;
          }}
          startThread(threadOptions) {{
            return {{ codexOptions: this.options, threadOptions }};
          }}
        }}

        const sourceEnv = {{
          HOME: {json.dumps(str(user_home))},
          CODEX_HOME: {json.dumps(str(codex_home))},
          PATH: "/usr/bin",
          CODEX_API_KEY: "test-api-key",
          CODEX_INVOCATION_PATH: {json.dumps(str(invocation_path))},
        }};
        const launcher = prepareCodexLauncher(
          sourceEnv,
          {json.dumps(str(fake_codex))},
        );
        const thread = startThread(
          FakeCodex,
          {{
            repo_root: "/workspace/run",
            model: "gpt-test",
            codex_config: {{ model_provider: "openai" }},
            cli_auth_credentials_store: "file",
          }},
          launcher,
        );
        const launched = spawnSync(
          thread.codexOptions.codexPathOverride,
          [
            "exec",
            "--experimental-json",
            "--config",
            'cli_auth_credentials_store="file"',
          ],
          {{
            env: thread.codexOptions.env,
            input: "inspect this workspace",
            encoding: "utf8",
          }},
        );
        const rejected = spawnSync(
          thread.codexOptions.codexPathOverride,
          ["login"],
          {{ env: thread.codexOptions.env, encoding: "utf8" }},
        );
        const launcherPath = launcher.codexPathOverride;
        const launcherDir = path.dirname(launcherPath);
        const launcherMode = fs.statSync(launcherPath).mode & 0o777;
        const launcherDirMode = fs.statSync(launcherDir).mode & 0o777;
        const originalRmSync = fs.rmSync;
        let removalAttempts = 0;
        fs.rmSync = (target, options) => {{
          if (String(target) === launcherDir && removalAttempts++ === 0) {{
            throw new Error("simulated cleanup failure");
          }}
          return originalRmSync(target, options);
        }};
        let cleanupError = "";
        try {{
          launcher.cleanup();
        }} catch (error) {{
          cleanupError = error.message;
        }}
        const existsAfterFailedCleanup = fs.existsSync(launcherDir);
        fs.rmSync = originalRmSync;
        launcher.cleanup();
        launcher.cleanup();
        const result = {{
          sameEnv: thread.codexOptions.env === sourceEnv,
          launcherPath,
          launcherMode,
          launcherDirMode,
          env: thread.codexOptions.env,
          config: thread.codexOptions.config,
          threadOptions: thread.threadOptions,
          launchedStatus: launched.status,
          launchedStderr: launched.stderr,
          rejectedStatus: rejected.status,
          rejectedStderr: rejected.stderr,
          cleanupError,
          existsAfterFailedCleanup,
          removalAttempts,
          existsAfterCleanup: fs.existsSync(launcherDir),
        }};
        console.log(JSON.stringify(result));
    """

    completed = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    result = json.loads(completed.stdout)
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    assert invocation == {
        "args": [
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--experimental-json",
            "--config",
            'cli_auth_credentials_store="file"',
        ],
        "codexHome": str(codex_home),
        "input": "inspect this workspace",
    }
    assert json.loads((codex_home / "auth.json").read_text(encoding="utf-8")) == {
        "tokens": {"access_token": "refreshed"}
    }
    assert (codex_home / "config.toml").exists()
    assert (plugins_dir / "plugin.json").exists()
    assert result["sameEnv"] is True
    assert Path(result["launcherPath"]).suffix == ".mjs"
    assert result["launcherMode"] == 0o700
    assert result["launcherDirMode"] == 0o700
    assert result["env"]["CODEX_HOME"] == str(codex_home)
    assert result["env"]["CODEX_API_KEY"] == "test-api-key"
    assert result["env"]["PATH"] == "/usr/bin"
    assert result["config"] == {
        "model_provider": "openai",
        "approval_policy": "never",
        "sandbox_mode": "workspace-write",
        "cli_auth_credentials_store": "file",
        "features": {"plugins": False},
        "sandbox_workspace_write": {
            "network_access": True,
            "exclude_tmpdir_env_var": True,
            "exclude_slash_tmp": True,
        },
    }
    assert result["threadOptions"] == {
        "workingDirectory": "/workspace/run",
        "skipGitRepoCheck": True,
        "model": "gpt-test",
    }
    assert result["launchedStatus"] == 0, result["launchedStderr"]
    assert result["rejectedStatus"] == 2
    assert "accepts only exec --experimental-json" in result["rejectedStderr"]
    assert result["cleanupError"] == "simulated cleanup failure"
    assert result["existsAfterFailedCleanup"] is True
    assert result["removalAttempts"] == 1
    assert result["existsAfterCleanup"] is False


@pytest.mark.parametrize("server_mode", [False, True])
def test_codex_bridge_cleans_launcher_for_process_lifetime(
    tmp_path: Path,
    server_mode: bool,
) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    bridge_path = app_dir / "codex_sdk_bridge.mjs"
    shutil.copy2(
        Path(runner.__file__).with_name("codex_sdk_bridge.mjs"),
        bridge_path,
    )
    node_modules = app_dir / "node_modules" / "@openai"
    codex_package = node_modules / "codex"
    codex_package.mkdir(parents=True)
    (codex_package / "package.json").write_text(
        json.dumps(
            {
                "name": "@openai/codex",
                "version": "0.139.0",
                "bin": {"codex": "bin/codex.mjs"},
            }
        ),
        encoding="utf-8",
    )
    fake_executable = codex_package / "bin" / "codex.mjs"
    fake_executable.parent.mkdir()
    fake_executable.write_text(f"#!{_node()}\nprocess.exit(0);\n", encoding="utf-8")
    fake_executable.chmod(0o700)
    sdk_package = node_modules / "codex-sdk"
    sdk_package.mkdir()
    (sdk_package / "package.json").write_text(
        json.dumps(
            {
                "name": "@openai/codex-sdk",
                "version": "0.139.0",
                "type": "module",
                "exports": "./index.mjs",
            }
        ),
        encoding="utf-8",
    )
    (sdk_package / "index.mjs").write_text(
        "export class Codex {\n"
        "  constructor(options) { this.options = options; }\n"
        "  startThread() {\n"
        "    return { run: async () => ({ finalResponse: 'ok', items: [] }) };\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    request_path = run_dir / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "repo_root": str(tmp_path),
                "run_dir": str(run_dir),
                "prompt": "test",
                "child_final_path": str(raw_dir / "child-final.md"),
                "items_path": str(raw_dir / "items.json"),
            }
        ),
        encoding="utf-8",
    )
    launcher_tmp = tmp_path / "launcher-tmp"
    launcher_tmp.mkdir()
    env = {**os.environ, "TMPDIR": str(launcher_tmp)}
    command = [_node(), str(bridge_path)]
    input_text = None
    if server_mode:
        command.extend(["--server", str(request_path)])
        input_text = json.dumps({"type": "shutdown"}) + "\n"
    else:
        command.append(str(request_path))

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        input=input_text,
        env=env,
        timeout=10,
    )

    if server_mode:
        messages = [json.loads(line) for line in completed.stdout.splitlines()]
        assert [message["type"] for message in messages] == ["ready", "shutdown_ack"]
    else:
        assert completed.stdout == "ok\n"
    assert list(launcher_tmp.iterdir()) == []


@pytest.mark.parametrize(
    "bridge_name",
    ["codex_sdk_bridge.mjs", "claude_bridge.mjs"],
)
def test_sdk_bridge_artifact_write_replaces_symlink_without_following(
    tmp_path: Path,
    bridge_name: str,
) -> None:
    bridge_uri = Path(runner.__file__).with_name(bridge_name).resolve().as_uri()
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep\n", encoding="utf-8")
    output = raw_dir / "result.json"
    output.symlink_to(outside)
    script = f"""
        import {{ writeRunArtifact }} from {json.dumps(bridge_uri)};
        writeRunArtifact(
          {{ run_dir: {json.dumps(str(run_dir))} }},
          {json.dumps(str(output))},
          "safe\\n",
        );
    """

    subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert not output.is_symlink()
    assert output.read_text(encoding="utf-8") == "safe\n"
    assert outside.read_text(encoding="utf-8") == "keep\n"


@pytest.mark.parametrize(
    "bridge_name",
    ["codex_sdk_bridge.mjs", "claude_bridge.mjs"],
)
def test_sdk_bridge_artifact_write_rejects_symlinked_parent(
    tmp_path: Path,
    bridge_name: str,
) -> None:
    bridge_uri = Path(runner.__file__).with_name(bridge_name).resolve().as_uri()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (run_dir / "raw").symlink_to(outside_dir, target_is_directory=True)
    output = run_dir / "raw" / "result.json"
    script = f"""
        import {{ writeRunArtifact }} from {json.dumps(bridge_uri)};
        try {{
          writeRunArtifact(
            {{ run_dir: {json.dumps(str(run_dir))} }},
            {json.dumps(str(output))},
            "unsafe\\n",
          );
        }} catch (error) {{
          console.log(error.message);
          process.exit(0);
        }}
        process.exit(1);
    """

    result = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert "must not contain symlinks" in result.stdout
    assert not (outside_dir / "result.json").exists()


@pytest.mark.parametrize(
    "bridge_name",
    ["codex_sdk_bridge.mjs", "claude_bridge.mjs"],
)
def test_sdk_bridge_preopened_artifact_resists_parent_swap(
    tmp_path: Path,
    bridge_name: str,
) -> None:
    bridge_uri = Path(runner.__file__).with_name(bridge_name).resolve().as_uri()
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    held_dir = run_dir / "raw-held"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_result = outside_dir / "result.json"
    outside_result.write_text("keep\n", encoding="utf-8")
    output = raw_dir / "result.json"
    script = f"""
        import fs from "node:fs";
        import {{ prepareRunArtifact, writePreparedRunArtifact }} from {json.dumps(bridge_uri)};
        const artifact = prepareRunArtifact(
          {{ run_dir: {json.dumps(str(run_dir))} }},
          {json.dumps(str(output))},
        );
        fs.renameSync({json.dumps(str(raw_dir))}, {json.dumps(str(held_dir))});
        fs.symlinkSync({json.dumps(str(outside_dir))}, {json.dumps(str(raw_dir))}, "dir");
        try {{
          writePreparedRunArtifact(artifact, "unsafe\\n");
        }} catch (error) {{
          console.log(error.message);
          fs.closeSync(artifact.fd);
          process.exit(0);
        }}
        fs.closeSync(artifact.fd);
        process.exit(1);
    """

    result = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert "changed during child turn" in result.stdout
    assert outside_result.read_text(encoding="utf-8") == "keep\n"


def test_codex_bridge_rejects_full_access_sandbox_mode() -> None:
    bridge_uri = (
        Path(runner.__file__).with_name("codex_sdk_bridge.mjs").resolve().as_uri()
    )
    script = f"""
        import {{ buildCodexConfig }} from {json.dumps(bridge_uri)};
        const config = buildCodexConfig({{
          codex_sandbox_mode: "danger-full-access",
        }});
        console.log(JSON.stringify(config));
    """

    result = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "Unsupported Codex sandbox mode: danger-full-access" in result.stderr


def test_codex_bridge_passes_reasoning_effort_as_thread_option(
    tmp_path: Path,
) -> None:
    bridge_uri = (
        Path(runner.__file__).with_name("codex_sdk_bridge.mjs").resolve().as_uri()
    )
    script = f"""
        import {{ buildThreadOptions, buildTurnOptions }} from {json.dumps(bridge_uri)};
        const request = {{
          repo_root: {json.dumps(str(tmp_path))},
          model: "gpt-5.6-sol",
          model_reasoning_effort: "ultra",
          output_schema: {{ type: "object" }},
        }};
        console.log(JSON.stringify({{
          thread: buildThreadOptions(request),
          turn: buildTurnOptions(request),
        }}));
    """

    result = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    options = json.loads(result.stdout)
    assert options["thread"] == {
        "workingDirectory": str(tmp_path),
        "skipGitRepoCheck": True,
        "model": "gpt-5.6-sol",
        "modelReasoningEffort": "ultra",
    }
    assert options["turn"] == {"outputSchema": {"type": "object"}}


def test_prepare_run_dir_uses_unique_names_and_private_raw_dir(tmp_path: Path) -> None:
    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    working_dir = tmp_path / "agentic"
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=usd,
        reference_images=[reference],
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        workbench_url="http://127.0.0.1:8088",
        default_output_root=working_dir,
    )

    first = runner._prepare_run_dir(config)
    second = runner._prepare_run_dir(config)
    request_path = first / "raw" / "request.json"
    runner._write_private_json(request_path, {"api_key": "secret"})

    assert first != second
    assert first.parent == working_dir / "runs"
    assert second.parent == working_dir / "runs"
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    assert stat.S_IMODE(second.stat().st_mode) == 0o700
    assert stat.S_IMODE((first / "raw").stat().st_mode) == 0o700
    assert stat.S_IMODE(request_path.stat().st_mode) == 0o600


def test_prepare_run_dir_privatizes_existing_run_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "existing-run"
    run_dir.mkdir(mode=0o755)
    run_dir.chmod(0o755)
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usd",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
        output_dir=run_dir,
    )

    prepared = runner._prepare_run_dir(config)

    assert prepared == run_dir
    assert stat.S_IMODE(prepared.stat().st_mode) == 0o700


def test_prepare_run_dir_removes_stale_physics_topology_plan(tmp_path: Path) -> None:
    run_dir = tmp_path / "physics-run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    stale_plan = raw_dir / "physics_topology_plan.json"
    stale_plan.write_text("{}", encoding="utf-8")
    stale_apply_patch = raw_dir / "physics_decision_patch_apply.json"
    stale_apply_patch.write_text("{}", encoding="utf-8")
    keep_patch = raw_dir / "physics_decision_patch.json"
    keep_patch.write_text("{}", encoding="utf-8")
    config = PhysicsApplyConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usd",
        workbench_url="http://127.0.0.1:8088",
        output_dir=run_dir,
    )

    prepared = runner._prepare_run_dir(config)  # type: ignore[arg-type]

    assert prepared == run_dir
    assert not stale_plan.exists()
    assert not stale_apply_patch.exists()
    assert keep_patch.exists()


def test_material_assignment_prints_run_location_at_start(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text("entries: []\n", encoding="utf-8")

    result = runner.run_material_assignment(
        MaterialAssignConfig(
            repo_root=tmp_path,
            usd_path=usd,
            reference_images=[reference],
            materials_yaml=materials_yaml,
            materials_usd=materials_usd,
            workbench_url="http://127.0.0.1:8088",
            dry_run=True,
            output_dir=tmp_path / "run",
        )
    )

    output = capsys.readouterr().out
    assert f"content-workflow-cli: run id: {result.run_dir.name}" in output
    assert f"content-workflow-cli: run directory: {result.run_dir}" in output
    assert f"content-workflow-cli: request: {result.request_path}" in output
    assert f"content-workflow-cli: child output: {result.child_output_path}" in output


def test_physics_apply_prints_run_location_at_start(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    usd = tmp_path / "asset.usd"
    usd.write_text("placeholder", encoding="utf-8")

    result = runner.run_physics_apply(
        PhysicsApplyConfig(
            repo_root=tmp_path,
            usd_path=usd,
            workbench_url="http://127.0.0.1:8088",
            dry_run=True,
            output_dir=tmp_path / "physics-run",
            simulation_engine="fake",
        )
    )

    output = capsys.readouterr().out
    assert f"content-workflow-cli: run id: {result.run_dir.name}" in output
    assert f"content-workflow-cli: run directory: {result.run_dir}" in output
    assert f"content-workflow-cli: request: {result.request_path}" in output
    assert f"content-workflow-cli: child output: {result.child_output_path}" in output


def test_codex_bridge_rejects_invalid_base_url() -> None:
    bridge_uri = (
        Path(runner.__file__).with_name("codex_sdk_bridge.mjs").resolve().as_uri()
    )
    script = f"""
        import {{ buildCodexConfig }} from {json.dumps(bridge_uri)};
        try {{
          buildCodexConfig({{
            codex_sandbox_mode: "workspace-write",
            codex_base_url: "file:///tmp/codex.sock",
          }});
        }} catch (error) {{
          console.log(error.message);
          process.exit(0);
        }}
        process.exit(1);
    """

    result = subprocess.run(
        [_node(), "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert "Invalid codex_base_url" in result.stdout


@pytest.mark.parametrize(
    ("bridge_name", "expected"),
    [
        ("claude_bridge.mjs", "Invalid Claude bridge request file"),
        ("codex_sdk_bridge.mjs", "Invalid Codex SDK bridge request file"),
    ],
)
def test_agent_bridges_report_invalid_request_files_with_context(
    tmp_path: Path,
    bridge_name: str,
    expected: str,
) -> None:
    bridge_path = Path(runner.__file__).with_name(bridge_name)
    request_path = tmp_path / "request.json"
    request_path.write_text("{not-json", encoding="utf-8")

    result = subprocess.run(
        [_node(), str(bridge_path), str(request_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 1
    assert expected in result.stderr
    assert str(request_path) in result.stderr


def test_run_material_assignment_uses_configured_unmanaged_workbench_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )
    observed_waits: list[tuple[float, Path | None]] = []

    def fake_wait_for_workbench(
        _url: str,
        *,
        timeout_seconds: float,
        output_root: Path | None = None,
    ) -> None:
        observed_waits.append((timeout_seconds, output_root))

    def fake_run_child_agent(**kwargs: object) -> int:
        child_final_path = kwargs["child_final_path"]
        assert isinstance(child_final_path, Path)
        child_final_path.write_text("done", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "wait_for_workbench", fake_wait_for_workbench)
    monkeypatch.setattr(runner, "_run_child_agent", fake_run_child_agent)
    monkeypatch.setattr(
        runner,
        "_ensure_material_assignment_artifacts",
        lambda **_kwargs: True,
    )

    result = runner.run_material_assignment(
        MaterialAssignConfig(
            repo_root=tmp_path,
            usd_path=usd,
            reference_images=[reference],
            materials_yaml=materials_yaml,
            materials_usd=materials_usd,
            workbench_url="http://127.0.0.1:8088",
            start_workbench=False,
            preflight=False,
            workbench_timeout_seconds=42.0,
        )
    )

    assert observed_waits == [(42.0, result.run_dir)]


def test_persistent_codex_session_is_disabled_for_confined_mode(tmp_path: Path) -> None:
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=tmp_path / "asset.usd",
        reference_images=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
        codex_persistent_refinement=True,
        codex_sandbox_mode=runner.CODEX_SANDBOX_WORKSPACE_WRITE,
    )

    assert not runner._should_start_codex_thread_session(config)


def test_run_material_assignment_runs_structured_finalizer_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)

    def fake_prepare_material_run_packet(config: Any) -> dict[str, Any]:
        raw_dir = config.run_dir / "raw"
        (config.run_dir / "trace").mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "material_run_packet.json").write_text(
            json.dumps(
                {
                    "session_id": "session-1",
                    "respect_existing_material_bindings": False,
                    "operation_counts_so_far": {
                        "render_calls_total": 0,
                        "workbench_api_calls_total": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        (raw_dir / "material_palette.json").write_text(
            json.dumps(
                {
                    "materials": [
                        {
                            "name": "Rubber Black Matte",
                            "material_path": "/World/Looks/Rubber_Black_Matte",
                            "tags": ["rubber", "black"],
                        },
                        {
                            "name": "Paint White Satin",
                            "material_path": "/World/Looks/Paint_White_Satin",
                            "tags": ["paint", "white"],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        (raw_dir / "visible_candidate_prims.json").write_text(
            json.dumps(
                {
                    "candidates": [
                        {"source_path": "/World/Foot", "shape_hint": "mesh"},
                        {"source_path": "/World/Torso", "shape_hint": "thin_panel"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        (raw_dir / "material_assignment_seed.json").write_text(
            json.dumps(
                {
                    "coverage": {"candidate_visible_prim_count": 2},
                    "assignments": [
                        {
                            "family": "Seed: foot ankle",
                            "coverage_status": "ambiguous_unassigned",
                            "material_name": None,
                            "material_path": None,
                            "prim_paths": ["/World/Foot"],
                        },
                        {
                            "family": "Seed: torso shell",
                            "coverage_status": "ambiguous_unassigned",
                            "material_name": None,
                            "material_path": None,
                            "prim_paths": ["/World/Torso"],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {"session_id": "session-1", "initial_evidence_renders": []}

    def fake_run_child_agent(**kwargs: object) -> int:
        prompt = kwargs["prompt"]
        child_final_path = kwargs["child_final_path"]
        child_run_dir = kwargs["run_dir"]
        assert isinstance(prompt, str)
        assert "Patch-only handoff" in prompt
        assert "Do not write `assignments.json`" in prompt
        assert "Wrapper-owned final artifacts" in prompt
        assert "Required final artifacts:" not in prompt
        assert "`assignments.json` must include" not in prompt
        assert "Pick uses the current session camera" in prompt
        assert "Workbench rejects extra fields" in prompt
        assert "matching camera/view fields" not in prompt
        assert "closest opaque/surface-compatible visual proxy" in prompt
        assert "Lack of an exact fabric/textile/paint subtype" in prompt
        assert isinstance(child_final_path, Path)
        assert isinstance(child_run_dir, Path)
        (child_run_dir / "raw" / "material_decision_patch.json").write_text(
            json.dumps(
                {
                    "material_assignments": [
                        {
                            "family": "foot ankle",
                            "material_name": "Rubber Black Matte",
                            "material_path": "/World/Looks/Rubber_Black_Matte",
                            "prim_paths": ["/World/Foot"],
                            "rationale": "Reference shows black foot hardware.",
                        },
                        {
                            "family": "torso shell",
                            "material_name": "Paint White Satin",
                            "material_path": "/World/Looks/Paint_White_Satin",
                            "prim_paths": ["/World/Torso"],
                            "rationale": "Reference shows a light torso shell.",
                        },
                    ],
                    "reviewed_no_override": [],
                    "visual_quality_assessment": {
                        "status": "fixed",
                        "issues_found": [],
                        "issues_fixed": [],
                        "unresolved_issues": [],
                        "assessment_notes": "Reviewed final render.",
                    },
                    "final_review_notes": "done",
                }
            ),
            encoding="utf-8",
        )
        child_final_path.write_text("done", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "wait_for_workbench", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "prepare_material_run_packet",
        fake_prepare_material_run_packet,
    )
    monkeypatch.setattr(runner, "packet_image_inputs", lambda _packet: [])
    monkeypatch.setattr(runner, "_run_child_agent", fake_run_child_agent)
    monkeypatch.setattr(runner, "close_workbench_session", lambda *_args: None)

    result = runner.run_material_assignment(
        MaterialAssignConfig(
            repo_root=tmp_path,
            usd_path=usd,
            reference_images=[reference],
            materials_yaml=materials_yaml,
            materials_usd=materials_usd,
            workbench_url="http://127.0.0.1:8088",
            start_workbench=False,
            preflight=True,
            output_dir=run_dir,
        )
    )

    assert result.returncode == 0
    assert posted_commands[0]["payload"]["prim_path"] == "/World/Foot"
    assignments = json.loads((run_dir / "assignments.json").read_text())
    assert assignments["generated_by"] == ("content-workflow-cli material finalizer")
    assert isinstance(assignments["assignments"], list)
    assert assignments["coverage"]["material_assignment_prim_count"] == 2
    assert assignments["coverage"]["preserved_existing_prim_count"] == 0
    assert (run_dir / "raw" / "material_decision_patch.json").exists()
    events = [
        json.loads(line)
        for line in (run_dir / "trace" / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert any(event["event_type"] == "material_decision_finalized" for event in events)


def test_run_material_assignment_refines_unresolved_vqa_until_fixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    _stub_structured_finalizer_workbench(monkeypatch)
    child_prompts: list[str] = []

    def fake_prepare_material_run_packet(config: Any) -> dict[str, Any]:
        raw_dir = config.run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "material_run_packet.json").write_text(
            json.dumps({"session_id": "session-1"}),
            encoding="utf-8",
        )
        (raw_dir / "material_palette.json").write_text(
            json.dumps(
                {
                    "materials": [
                        {
                            "name": "Rubber Black Matte",
                            "material_path": "/World/Looks/Rubber_Black_Matte",
                            "tags": ["black"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (raw_dir / "visible_candidate_prims.json").write_text(
            json.dumps(
                {
                    "candidates": [
                        {"source_path": "/World/Foot", "shape_hint": "mesh"},
                        {"source_path": "/World/Torso", "shape_hint": "panel"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        (raw_dir / "material_assignment_seed.json").write_text(
            json.dumps(
                {
                    "coverage": {"candidate_visible_prim_count": 2},
                    "assignments": [
                        {
                            "family": "Seed: foot",
                            "coverage_status": "ambiguous_unassigned",
                            "prim_paths": ["/World/Foot"],
                        },
                        {
                            "family": "Seed: torso",
                            "coverage_status": "ambiguous_unassigned",
                            "prim_paths": ["/World/Torso"],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        (raw_dir / "material_authoring_context.md").write_text(
            "foot and torso candidates",
            encoding="utf-8",
        )
        return {"session_id": "session-1", "initial_evidence_renders": []}

    def write_patch(child_run_dir: Path, *, fixed: bool) -> None:
        material_assignments = [
            {
                "family": "foot",
                "material_name": "Rubber Black Matte",
                "material_path": "/World/Looks/Rubber_Black_Matte",
                "prim_paths": ["/World/Foot"],
                "rationale": "Reference shows dark foot hardware.",
            }
        ]
        reviewed_no_override: list[dict[str, object]] = []
        vqa = {
            "status": "unresolved_issues",
            "issues_found": [
                {
                    "description": "Torso is too green.",
                    "affected_prim_paths": ["/World/Torso"],
                }
            ],
            "issues_fixed": [],
            "unresolved_issues": [
                {
                    "description": "Torso is too green.",
                    "affected_prim_paths": ["/World/Torso"],
                }
            ],
            "assessment_notes": "Needs repair.",
        }
        if fixed:
            material_assignments.append(
                {
                    "family": "torso",
                    "material_name": "Rubber Black Matte",
                    "material_path": "/World/Looks/Rubber_Black_Matte",
                    "prim_paths": ["/World/Torso"],
                    "rationale": "Refinement fixed the recorded torso mismatch.",
                }
            )
            reviewed_no_override = []
            vqa = {
                "status": "fixed",
                "issues_found": ["Torso is too green."],
                "issues_fixed": ["Changed torso material after VQA repair."],
                "unresolved_issues": [],
                "assessment_notes": "Recorded VQA issue was repaired.",
            }
        (child_run_dir / "raw" / "material_decision_patch.json").write_text(
            json.dumps(
                {
                    "material_assignments": material_assignments,
                    "reviewed_no_override": reviewed_no_override,
                    "visual_quality_assessment": vqa,
                    "final_review_issues_found": ["Torso is too green."]
                    if not fixed
                    else ["Torso is too green."],
                    "final_review_issues_fixed": []
                    if not fixed
                    else ["Changed torso material after VQA repair."],
                    "final_review_notes": "refined" if fixed else "needs repair",
                }
            ),
            encoding="utf-8",
        )

    def fake_run_child_agent(**kwargs: object) -> int:
        prompt = kwargs["prompt"]
        child_final_path = kwargs["child_final_path"]
        child_run_dir = kwargs["run_dir"]
        assert isinstance(prompt, str)
        assert isinstance(child_final_path, Path)
        assert isinstance(child_run_dir, Path)
        child_prompts.append(prompt)
        write_patch(child_run_dir, fixed="Active issue packet:" in prompt)
        child_final_path.write_text("done", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "wait_for_workbench", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "prepare_material_run_packet",
        fake_prepare_material_run_packet,
    )
    monkeypatch.setattr(runner, "packet_image_inputs", lambda _packet: [])
    monkeypatch.setattr(runner, "_run_child_agent", fake_run_child_agent)
    monkeypatch.setattr(runner, "close_workbench_session", lambda *_args: None)

    result = runner.run_material_assignment(
        MaterialAssignConfig(
            repo_root=tmp_path,
            usd_path=usd,
            reference_images=[reference],
            materials_yaml=materials_yaml,
            materials_usd=materials_usd,
            workbench_url="http://127.0.0.1:8088",
            start_workbench=False,
            preflight=True,
            output_dir=run_dir,
        )
    )

    assert result.returncode == 0
    assert len(child_prompts) == 2
    assert "Torso is too green" in child_prompts[1]
    assert "Active issue packet:" in child_prompts[1]
    assert "Do not redo the material plan" in child_prompts[1]
    assert "Do not read skill docs, repository source" in child_prompts[1]
    assert "decision_patch_signature" not in child_prompts[1]
    assert "Read `/" not in child_prompts[1]
    visual_quality = json.loads(
        (run_dir / "visual_quality_assessment.json").read_text(encoding="utf-8")
    )
    assert visual_quality["status"] == "fixed"
    history = json.loads(
        (run_dir / "raw" / "vqa_refinement_history.json").read_text(encoding="utf-8")
    )
    assert history["status"] == "satisfied"
    assert history["iterations"][0]["iteration"] == 2


def test_vqa_refinement_assessment_uses_compact_decision_summary(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    long_paths = [
        f"/Root/Very/Deep/Assembly/Component_{index}/Mesh" for index in range(25)
    ]
    vqa = {
        "status": "unresolved_issues",
        "issues_found": ["Panel is too green."],
        "issues_fixed": [],
        "unresolved_issues": ["Panel is too green."],
        "checked_views": [str(run_dir / "final_renders" / "final_oblique.png")],
        "assessment_notes": "Human-readable notes should not define convergence.",
    }
    (run_dir / "visual_quality_assessment.json").write_text(
        json.dumps(vqa),
        encoding="utf-8",
    )
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "candidate_visible_prim_count": 25,
                    "material_decision_prim_count": 25,
                    "ambiguous_unassigned_prim_count": 0,
                },
                "final_review": {"unresolved_issues": []},
                "visual_quality_assessment": vqa,
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_decision_patch.json").write_text(
        json.dumps(
            {
                "material_assignments": [
                    {
                        "family": "green panel",
                        "material_name": "Plastic Green",
                        "material_path": "/World/Looks/Plastic_Green",
                        "runtime_prim_paths": long_paths,
                        "source_prim_paths": long_paths,
                        "rationale": "Current accepted decision.",
                    }
                ],
                "reviewed_no_override": [],
                "visual_quality_assessment": vqa,
            }
        ),
        encoding="utf-8",
    )

    assessment = runner._current_vqa_refinement_assessment(run_dir)
    rendered = json.dumps(assessment)

    assert "decision_patch_signature" not in assessment
    assert len(str(assessment["signature"])) == 64
    assert "/Root/Very/Deep" not in rendered
    assert assessment["current_material_decisions"] == [
        {
            "kind": "material_assignments",
            "family": "green panel",
            "material_name": "Plastic Green",
            "material_path": "/World/Looks/Plastic_Green",
            "target_prim_count": 25,
            "rationale": "Current accepted decision.",
        }
    ]


def test_vqa_refinement_assessment_ignores_fixed_issues_found(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    vqa = {
        "status": "fixed",
        "issues_found": ["Initial clean-slate render lacked blue plastic."],
        "issues_fixed": ["Blue plastic was assigned."],
        "unresolved_issues": [],
        "checked_views": [str(run_dir / "final_renders" / "final_oblique.png")],
    }
    (run_dir / "visual_quality_assessment.json").write_text(
        json.dumps(vqa),
        encoding="utf-8",
    )
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "candidate_visible_prim_count": 1,
                    "material_decision_prim_count": 1,
                    "ambiguous_unassigned_prim_count": 0,
                },
                "final_review": {"unresolved_issues": []},
                "visual_quality_assessment": vqa,
            }
        ),
        encoding="utf-8",
    )

    assessment = runner._current_vqa_refinement_assessment(run_dir)

    assert assessment["status"] == "fixed"
    assert assessment["active_issues"] == []
    assert assessment["unresolved_vqa_issues"] == []
    assert assessment["vqa_issues_found"] == [
        "Initial clean-slate render lacked blue plastic."
    ]
    assert runner._vqa_refinement_satisfied(assessment) is True


def test_vqa_refinement_rejected_assignment_prevents_satisfied_gate(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    vqa = {
        "status": "fixed",
        "issues_found": [],
        "issues_fixed": [],
        "unresolved_issues": [],
    }
    (run_dir / "visual_quality_assessment.json").write_text(
        json.dumps(vqa),
        encoding="utf-8",
    )
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "candidate_visible_prim_count": 2,
                    "material_decision_prim_count": 2,
                    "ambiguous_unassigned_prim_count": 0,
                },
                "final_review": {"unresolved_issues": []},
                "visual_quality_assessment": vqa,
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "rejected_material_assignments.json").write_text(
        json.dumps(
            [
                {
                    "family": "broad white frame",
                    "coverage_status": "material_assignment",
                    "prim_paths": ["/World/A", "/World/B"],
                    "rejection_reason": "Rejected material assignment was too broad.",
                }
            ]
        ),
        encoding="utf-8",
    )

    assessment = runner._current_vqa_refinement_assessment(run_dir)

    assert assessment["rejected_assignment_issues"] == [
        "Rejected material decision for broad white frame: Rejected material assignment was too broad."
    ]
    assert runner._vqa_refinement_satisfied(assessment) is False


def test_vqa_refinement_assessment_treats_coverage_gap_as_actionable(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    vqa = {
        "status": "pass",
        "issues_found": [],
        "issues_fixed": [],
        "unresolved_issues": [],
    }
    (run_dir / "visual_quality_assessment.json").write_text(
        json.dumps(vqa),
        encoding="utf-8",
    )
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "candidate_visible_prim_count": 4,
                    "material_decision_prim_count": 2,
                    "ambiguous_unassigned_prim_count": 2,
                    "missing_assignment_prim_count": 2,
                    "rejected_assignment_prim_count": 0,
                },
                "final_review": {"unresolved_issues": []},
                "visual_quality_assessment": vqa,
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_decision_patch.json").write_text(
        json.dumps({"material_assignments": [], "reviewed_no_override": []}),
        encoding="utf-8",
    )

    assessment = runner._current_vqa_refinement_assessment(run_dir)

    assert assessment["active_issues"] == [
        {
            "id": "issue-1",
            "source": "coverage",
            "description": (
                "Coverage gap: 2 visible candidate prim(s) have no proposed material assignment."
            ),
            "affected_prim_paths": [],
            "actionable": True,
            "systematic_limitation": False,
        }
    ]
    assert runner._vqa_refinement_systematic_unfixable(assessment) is False


def test_vqa_refinement_loop_stops_when_issue_signature_converges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    (run_dir / "trace").mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    for path in [
        tmp_path / "asset.usd",
        tmp_path / "materials.yaml",
        tmp_path / "materials.usd",
        tmp_path / "reference.png",
    ]:
        path.write_text("placeholder", encoding="utf-8")
    vqa = {
        "status": "unresolved_issues",
        "issues_found": [
            {
                "description": "Panel is too dark.",
                "affected_prim_paths": ["/World/Panel"],
            }
        ],
        "issues_fixed": [],
        "unresolved_issues": [
            {
                "description": "Panel is too dark.",
                "affected_prim_paths": ["/World/Panel"],
            }
        ],
    }
    (run_dir / "visual_quality_assessment.json").write_text(
        json.dumps(vqa),
        encoding="utf-8",
    )
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "candidate_visible_prim_count": 1,
                    "material_decision_prim_count": 1,
                },
                "assignments": [],
                "final_review": {"unresolved_issues": []},
                "visual_quality_assessment": vqa,
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_decision_patch.json").write_text(
        json.dumps(
            {
                "material_assignments": [],
                "reviewed_no_override": [],
                "visual_quality_assessment": vqa,
            }
        ),
        encoding="utf-8",
    )
    child_calls = 0

    def fake_run_child_agent(**kwargs: object) -> int:
        nonlocal child_calls
        child_calls += 1
        child_final_path = kwargs["child_final_path"]
        assert isinstance(child_final_path, Path)
        child_final_path.write_text("unchanged", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "_run_child_agent", fake_run_child_agent)
    monkeypatch.setattr(
        runner,
        "_finalize_structured_material_decisions",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        runner,
        "_ensure_material_assignment_artifacts",
        lambda **_kwargs: True,
    )

    returncode = runner._run_vqa_refinement_loop(
        config=MaterialAssignConfig(
            repo_root=tmp_path,
            usd_path=tmp_path / "asset.usd",
            reference_images=[tmp_path / "reference.png"],
            materials_yaml=tmp_path / "materials.yaml",
            materials_usd=tmp_path / "materials.usd",
            workbench_url="http://127.0.0.1:8088",
            output_dir=run_dir,
            vqa_refinement_max_iterations=5,
        ),
        run_dir=run_dir,
        request={},
        preflight_packet={"session_id": "session-1"},
        trace_writer=TraceWriter(run_dir),
        managed_workbench=None,
        initial_child_output_path=run_dir / "child-output.log",
        initial_child_final_path=run_dir / "child-final.md",
        prompt_image_inputs=[],
    )

    assert returncode == 0
    assert child_calls == 1
    history = json.loads((raw_dir / "vqa_refinement_history.json").read_text())
    assert history["status"] == "converged_unresolved"
    assert history["iterations"][0]["converged"] is True
    artifact_index = json.loads(
        (raw_dir / "vqa_refinement_artifact_index_2.json").read_text()
    )
    issue_packet = json.loads(
        (raw_dir / "vqa_refinement_issue_packet_2.json").read_text()
    )
    assert artifact_index["schema_version"].endswith("artifact-index.v1")
    assert issue_packet["schema_version"].endswith("issue-packet.v1")
    assert issue_packet["active_issues"][0]["affected_prim_paths"] == ["/World/Panel"]


def test_vqa_refinement_runs_visual_issue_without_known_prims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    (run_dir / "trace").mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    for path in [
        tmp_path / "asset.usd",
        tmp_path / "materials.yaml",
        tmp_path / "materials.usd",
        tmp_path / "reference.png",
    ]:
        path.write_text("placeholder", encoding="utf-8")
    vqa = {
        "status": "unresolved_issues",
        "issues_found": ["Blue tray is still missing."],
        "issues_fixed": [],
        "unresolved_issues": ["Blue tray is still missing."],
        "assessment_notes": "Visual review did not identify affected prim paths.",
    }
    fixed_vqa = {
        "status": "fixed",
        "issues_found": ["Blue tray was missing."],
        "issues_fixed": ["Blue tray material was repaired after picking."],
        "unresolved_issues": [],
    }
    (run_dir / "visual_quality_assessment.json").write_text(
        json.dumps(vqa),
        encoding="utf-8",
    )
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "candidate_visible_prim_count": 4,
                    "material_decision_prim_count": 4,
                },
                "assignments": [],
                "final_review": {"unresolved_issues": []},
                "visual_quality_assessment": vqa,
            }
        ),
        encoding="utf-8",
    )
    child_calls = 0

    def fake_run_child_agent(**kwargs: object) -> int:
        nonlocal child_calls
        child_calls += 1
        prompt = kwargs["prompt"]
        child_final_path = kwargs["child_final_path"]
        assert isinstance(prompt, str)
        assert "Blue tray is still missing." in prompt
        assert isinstance(child_final_path, Path)
        (run_dir / "visual_quality_assessment.json").write_text(
            json.dumps(fixed_vqa),
            encoding="utf-8",
        )
        (run_dir / "assignments.json").write_text(
            json.dumps(
                {
                    "coverage": {
                        "candidate_visible_prim_count": 4,
                        "material_decision_prim_count": 4,
                    },
                    "assignments": [],
                    "final_review": {"unresolved_issues": []},
                    "visual_quality_assessment": fixed_vqa,
                }
            ),
            encoding="utf-8",
        )
        child_final_path.write_text("fixed", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "_run_child_agent", fake_run_child_agent)
    monkeypatch.setattr(
        runner,
        "_finalize_structured_material_decisions",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        runner,
        "_ensure_material_assignment_artifacts",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        runner,
        "_snapshot_material_step_artifacts",
        lambda **_kwargs: True,
    )

    returncode = runner._run_vqa_refinement_loop(
        config=MaterialAssignConfig(
            repo_root=tmp_path,
            usd_path=tmp_path / "asset.usd",
            reference_images=[tmp_path / "reference.png"],
            materials_yaml=tmp_path / "materials.yaml",
            materials_usd=tmp_path / "materials.usd",
            workbench_url="http://127.0.0.1:8088",
            output_dir=run_dir,
            vqa_refinement_max_iterations=5,
        ),
        run_dir=run_dir,
        request={},
        preflight_packet={"session_id": "session-1"},
        trace_writer=TraceWriter(run_dir),
        managed_workbench=None,
        initial_child_output_path=run_dir / "child-output.log",
        initial_child_final_path=run_dir / "child-final.md",
        prompt_image_inputs=[],
    )

    assert returncode == 0
    assert child_calls == 1
    history = json.loads((raw_dir / "vqa_refinement_history.json").read_text())
    assert history["status"] == "satisfied"
    assessment = history["initial_assessment"]
    assert assessment["active_issues"][0]["affected_prim_paths"] == []
    assert assessment["active_issues"][0]["actionable"] is True
    assert assessment["active_issues"][0]["systematic_limitation"] is False


def test_vqa_refinement_loop_stops_on_systematic_unfixable_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    (run_dir / "trace").mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    for path in [
        tmp_path / "asset.usd",
        tmp_path / "materials.yaml",
        tmp_path / "materials.usd",
        tmp_path / "reference.png",
    ]:
        path.write_text("placeholder", encoding="utf-8")
    initial_vqa = {
        "status": "unresolved_issues",
        "issues_found": [
            {
                "description": "Seat cushion is the wrong material.",
                "affected_prim_paths": ["/World/SeatCushion"],
            }
        ],
        "issues_fixed": [],
        "unresolved_issues": [
            {
                "description": "Seat cushion is the wrong material.",
                "affected_prim_paths": ["/World/SeatCushion"],
            }
        ],
    }
    final_vqa = {
        "status": "unresolved_issues",
        "issues_found": ["Seat cushion is the wrong material."],
        "issues_fixed": [],
        "unresolved_issues": [
            "Seat cushion remains unresolved because the material library has no blue fabric upholstery material."
        ],
    }

    def write_artifacts(vqa: dict[str, object]) -> None:
        (run_dir / "visual_quality_assessment.json").write_text(
            json.dumps(vqa),
            encoding="utf-8",
        )
        (run_dir / "assignments.json").write_text(
            json.dumps(
                {
                    "coverage": {
                        "candidate_visible_prim_count": 1,
                        "material_decision_prim_count": 1,
                    },
                    "assignments": [],
                    "final_review": {"unresolved_issues": vqa["unresolved_issues"]},
                    "visual_quality_assessment": vqa,
                }
            ),
            encoding="utf-8",
        )
        (raw_dir / "material_decision_patch.json").write_text(
            json.dumps(
                {
                    "material_assignments": [],
                    "reviewed_no_override": [],
                    "visual_quality_assessment": vqa,
                }
            ),
            encoding="utf-8",
        )

    write_artifacts(initial_vqa)
    child_calls = 0

    def fake_run_child_agent(**kwargs: object) -> int:
        nonlocal child_calls
        child_calls += 1
        prompt = kwargs["prompt"]
        child_final_path = kwargs["child_final_path"]
        assert isinstance(prompt, str)
        assert "Active issue packet:" in prompt
        assert "Do not read skill docs, repository source" in prompt
        assert isinstance(child_final_path, Path)
        write_artifacts(final_vqa)
        child_final_path.write_text("unfixable", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "_run_child_agent", fake_run_child_agent)
    monkeypatch.setattr(
        runner,
        "_finalize_structured_material_decisions",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        runner,
        "_ensure_material_assignment_artifacts",
        lambda **_kwargs: True,
    )

    returncode = runner._run_vqa_refinement_loop(
        config=MaterialAssignConfig(
            repo_root=tmp_path,
            usd_path=tmp_path / "asset.usd",
            reference_images=[tmp_path / "reference.png"],
            materials_yaml=tmp_path / "materials.yaml",
            materials_usd=tmp_path / "materials.usd",
            workbench_url="http://127.0.0.1:8088",
            output_dir=run_dir,
            vqa_refinement_max_iterations=5,
        ),
        run_dir=run_dir,
        request={},
        preflight_packet={"session_id": "session-1"},
        trace_writer=TraceWriter(run_dir),
        managed_workbench=None,
        initial_child_output_path=run_dir / "child-output.log",
        initial_child_final_path=run_dir / "child-final.md",
        prompt_image_inputs=[],
    )

    assert returncode == 0
    assert child_calls == 1
    history = json.loads((raw_dir / "vqa_refinement_history.json").read_text())
    assert history["status"] == "systematic_unfixable"
    assert (
        history["iterations"][0]["after"]["active_issues"][0]["systematic_limitation"]
        is True
    )


def test_vqa_refinement_loop_skips_initial_systematic_unfixable_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    (run_dir / "trace").mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    issue = (
        "No exact wood or pale laminate material exists in the supplied "
        "palette for the tan desktop/shelf slab; preserving its current tan "
        "appearance is a better visual match than any available library proxy."
    )
    vqa = {
        "status": "unresolved_issues",
        "issues_found": [issue],
        "issues_fixed": [],
        "unresolved_issues": [issue],
        "assessment_notes": (
            "The remaining limitation is material-library granularity, not an "
            "unfixed high-contrast mismatch."
        ),
    }
    (run_dir / "visual_quality_assessment.json").write_text(
        json.dumps(vqa),
        encoding="utf-8",
    )
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "candidate_visible_prim_count": 1,
                    "material_decision_prim_count": 1,
                },
                "assignments": [],
                "final_review": {"unresolved_issues": [issue]},
                "visual_quality_assessment": vqa,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "_run_child_agent",
        lambda **_kwargs: pytest.fail("systematic issue should not refine"),
    )

    returncode = runner._run_vqa_refinement_loop(
        config=MaterialAssignConfig(
            repo_root=tmp_path,
            usd_path=tmp_path / "asset.usd",
            reference_images=[tmp_path / "reference.png"],
            materials_yaml=tmp_path / "materials.yaml",
            materials_usd=tmp_path / "materials.usd",
            workbench_url="http://127.0.0.1:8088",
            output_dir=run_dir,
            vqa_refinement_max_iterations=5,
        ),
        run_dir=run_dir,
        request={},
        preflight_packet={"session_id": "session-1"},
        trace_writer=TraceWriter(run_dir),
        managed_workbench=None,
        initial_child_output_path=run_dir / "child-output.log",
        initial_child_final_path=run_dir / "child-final.md",
        prompt_image_inputs=[],
    )

    assert returncode == 0
    history = json.loads((raw_dir / "vqa_refinement_history.json").read_text())
    assert history["status"] == "systematic_unfixable_initial"
    assert not (raw_dir / "vqa_refinement_2_request.json").exists()


def test_vqa_refinement_loop_skips_after_targeted_picks_prove_limitation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    (run_dir / "trace").mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    for path in [
        tmp_path / "asset.usd",
        tmp_path / "materials.yaml",
        tmp_path / "materials.usd",
        tmp_path / "reference.png",
    ]:
        path.write_text("placeholder", encoding="utf-8")
    vqa = {
        "status": "unresolved_issues",
        "issues_found": [
            "Iteration 2 targeted picks confirmed the limitation: red-edge picks "
            "resolved broad tray/base geometry."
        ],
        "issues_fixed": [],
        "unresolved_issues": [
            "Minor accent colors remain approximate or unassigned.",
            (
                "82 visible material candidate prim(s) were left without explicit "
                "library material assignments in a clean-slate session."
            ),
        ],
        "assessment_notes": (
            "Iteration 2 left the remaining accent and coverage issues unresolved "
            "because targeted picks did not provide safe isolated prims."
        ),
    }
    (run_dir / "visual_quality_assessment.json").write_text(
        json.dumps(vqa),
        encoding="utf-8",
    )
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "candidate_visible_prim_count": 97,
                    "material_decision_prim_count": 97,
                    "ambiguous_unassigned_prim_count": 82,
                },
                "assignments": [],
                "final_review": {"unresolved_issues": vqa["unresolved_issues"]},
                "visual_quality_assessment": vqa,
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_decision_patch.json").write_text(
        json.dumps({"material_assignments": [], "reviewed_no_override": []}),
        encoding="utf-8",
    )

    def fail_run_child_agent(**_kwargs: object) -> int:
        raise AssertionError("systematic limitation should skip child refinement")

    monkeypatch.setattr(runner, "_run_child_agent", fail_run_child_agent)

    returncode = runner._run_vqa_refinement_loop(
        config=MaterialAssignConfig(
            repo_root=tmp_path,
            usd_path=tmp_path / "asset.usd",
            reference_images=[tmp_path / "reference.png"],
            materials_yaml=tmp_path / "materials.yaml",
            materials_usd=tmp_path / "materials.usd",
            workbench_url="http://127.0.0.1:8088",
            output_dir=run_dir,
            vqa_refinement_max_iterations=5,
        ),
        run_dir=run_dir,
        request={},
        preflight_packet={"session_id": "session-1"},
        trace_writer=TraceWriter(run_dir),
        managed_workbench=None,
        initial_child_output_path=run_dir / "child-output.log",
        initial_child_final_path=run_dir / "child-final.md",
        prompt_image_inputs=[],
    )

    assert returncode == 0
    history = json.loads((raw_dir / "vqa_refinement_history.json").read_text())
    assert history["status"] == "systematic_unfixable_initial"
    assert not (raw_dir / "vqa_refinement_2_request.json").exists()


def test_vqa_refinement_exposes_rejected_assignment_paths(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    vqa = {
        "status": "unresolved_issues",
        "issues_found": ["Some roller bars are still orange."],
        "issues_fixed": [],
        "unresolved_issues": ["Some roller bars are still orange."],
    }
    (run_dir / "visual_quality_assessment.json").write_text(
        json.dumps(vqa),
        encoding="utf-8",
    )
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "candidate_visible_prim_count": 4,
                    "material_decision_prim_count": 4,
                },
                "assignments": [],
                "final_review": {"unresolved_issues": []},
                "visual_quality_assessment": vqa,
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "rejected_material_assignments.json").write_text(
        json.dumps(
            [
                {
                    "family": "top silver roller cylinders",
                    "coverage_status": "material_assignment",
                    "prim_paths": ["/World/Lift/Roller_A", "/World/Lift/Roller_B"],
                    "rejection_reason": (
                        "Rejected material assignment was too broad and must be "
                        "split into smaller mixed groups."
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )

    assessment = runner._current_vqa_refinement_assessment(run_dir)

    rejected_issue = next(
        issue
        for issue in assessment["active_issues"]
        if issue["source"] == "rejected_assignment"
    )
    assert rejected_issue["affected_prim_paths"] == [
        "/World/Lift/Roller_A",
        "/World/Lift/Roller_B",
    ]
    assert rejected_issue["actionable"] is True
    assert rejected_issue["systematic_limitation"] is False
    assert runner._vqa_refinement_systematic_unfixable(assessment) is False


def test_vqa_refinement_marks_coupled_granularity_as_systematic(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    issue = (
        "The arm pads remain blue because Workbench candidate granularity "
        "couples them with the seat family, and the prepared packet did not "
        "expose a safe separate visible family."
    )
    vqa = {
        "status": "unresolved_issues",
        "issues_found": [issue],
        "issues_fixed": [],
        "unresolved_issues": [issue],
    }
    (run_dir / "visual_quality_assessment.json").write_text(
        json.dumps(vqa),
        encoding="utf-8",
    )
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "candidate_visible_prim_count": 1,
                    "material_decision_prim_count": 1,
                },
                "assignments": [],
                "final_review": {"unresolved_issues": []},
                "visual_quality_assessment": vqa,
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_decision_patch.json").write_text(
        json.dumps({"material_assignments": [], "reviewed_no_override": []}),
        encoding="utf-8",
    )

    assessment = runner._current_vqa_refinement_assessment(run_dir)

    assert assessment["active_issues"] == [
        {
            "id": "issue-1",
            "source": "visual_quality",
            "description": issue,
            "affected_prim_paths": [],
            "actionable": False,
            "systematic_limitation": True,
        }
    ]
    assert runner._vqa_refinement_systematic_unfixable(assessment) is True
    assert runner._issue_looks_systematic_unfixable(
        "Reference-blue top/tray coverage is incomplete because Workbench "
        "exposes that surface as part of the aluminum prim and it is not "
        "separately authorable from the rungs/frame."
    )
    assert runner._issue_looks_systematic_unfixable(
        "Reference-blue top/tray coverage is incomplete at the visible "
        "broad-surface level because Workbench exposes that surface as part "
        "of the aluminum prim; overriding it blue would also recolor metal "
        "rungs/frame and is visually worse."
    )
    assert runner._issue_looks_systematic_unfixable(
        "Fine papers, pegboard holes, and small tool colors from the photo "
        "are not exposed as individually assignable visible candidate prims "
        "in this Workbench session."
    )
    assert runner._issue_looks_systematic_unfixable(
        "Black bin/tool accents remain gray/white because applying Plastic "
        "Black to the picked bin/tool paths also blackens large desk body and "
        "shelf surfaces; they are not independently assignable."
    )
    assert runner._issue_looks_systematic_unfixable(
        "Black bin/tool accents share Workbench target paths with large white "
        "body/shelf surfaces; Plastic Black was rejected because it blackened "
        "the body."
    )
    assert runner._issue_looks_systematic_unfixable(
        "Car Paint Beige is a visual proxy rather than a true wood/laminate "
        "substance for the desktop because the supplied material palette has "
        "no wood material."
    )
    assert runner._issue_looks_systematic_unfixable(
        "Missing source geometry for tools, papers, pegboard perforations, "
        "and room context cannot be fixed with material material assignments."
    )
    assert runner._issue_looks_systematic_unfixable(
        "No wood or laminate material is present in the supplied palette; "
        "retaining the current tan tabletop is visually closer than using "
        "glossy beige car paint, brown painted steel, or orange plastic."
    )
    assert runner._issue_looks_systematic_unfixable(
        "Small reference-photo details such as papers, tools, pegboard holes, "
        "and blue/orange accents are not separately material-addressable in "
        "the prepared visible candidate set."
    )
    assert runner._issue_looks_systematic_unfixable(
        "Remaining candidates are mixed generic-geometry buckets, not safe exact "
        "material targets."
    )
    assert runner._issue_looks_systematic_unfixable(
        "Small central underside sensor core is still brown/orange; no safe "
        "separate material path was identified in the allowed pick budget."
    )
    assert not runner._issue_looks_systematic_unfixable(
        "82 visible material candidate prim(s) were left without explicit library "
        "material assignments in a clean-slate session."
    )
    assert runner._assessment_notes_claim_systematic_limitations(
        "Iteration 2 left the remaining accent and coverage issues unresolved "
        "because targeted picks did not provide safe isolated prims."
    )
    assert runner._assessment_notes_claim_systematic_limitations(
        "Remaining issues are the central underside sensor core, which is not "
        "safely separable by pick, and the systematic coverage limitation."
    )


def test_build_material_refinement_prompt_is_compact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prompt = build_material_refinement_prompt(
        run_dir=run_dir,
        usd_path=tmp_path / "asset.usd",
        reference_images=[tmp_path / "reference.png"],
        reference_files=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
        session_id="session-1",
        iteration=2,
        max_iterations=5,
        issue_summary={
            "status": "unresolved_issues",
            "active_issues": [
                {
                    "id": "issue-1",
                    "source": "visual_quality",
                    "description": "Blue plastic is too dark.",
                    "systematic_limitation": False,
                }
            ],
            "vqa_issues_fixed": ["Frame is already fixed."],
            "current_material_decisions": [
                {
                    "family": "blue plastic",
                    "material_name": "Plastic Dark Blue",
                    "target_prim_count": 2,
                }
            ],
        },
        history_path=run_dir / "raw" / "vqa_refinement_history.json",
        repair_attempt_ledger=[],
        previous_child_artifacts=[
            {
                "child_output": str(run_dir / "child-output.log"),
                "child_final": str(run_dir / "child-final.md"),
            }
        ],
    )

    assert len(prompt) < 8000
    assert "Active issue packet:" in prompt
    assert "Do not read skill docs, repository source" in prompt
    assert "Current unresolved issue summary:" not in prompt
    assert "Repair procedure:" not in prompt
    assert "child-output.log" not in prompt
    assert "`None`" not in prompt

    prompt_with_artifacts = build_material_refinement_prompt(
        run_dir=run_dir,
        usd_path=tmp_path / "asset.usd",
        reference_images=[tmp_path / "reference.png"],
        reference_files=[],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
        session_id="session-1",
        iteration=2,
        max_iterations=5,
        issue_summary={"status": "unresolved_issues", "active_issues": []},
        history_path=run_dir / "raw" / "vqa_refinement_history.json",
        artifact_index_path=run_dir / "raw" / "vqa_refinement_artifact_index_2.json",
        issue_packet_path=run_dir / "raw" / "vqa_refinement_issue_packet_2.json",
    )
    assert "vqa_refinement_artifact_index_2.json" in prompt_with_artifacts
    assert "vqa_refinement_issue_packet_2.json" in prompt_with_artifacts


def test_build_skill_routed_material_assignment_prompt_is_compact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    prompt = build_skill_routed_material_assignment_prompt(
        repo_root=tmp_path,
        run_dir=run_dir,
        usd_path=tmp_path / "asset.usd",
        reference_images=[tmp_path / "reference.png"],
        reference_files=[tmp_path / "reference.pdf"],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
        preflight_packet={
            "initial_evidence_renders": [
                {"name": "initial_oblique", "image_path": str(tmp_path / "view.png")}
            ]
        },
    )

    assert "`content-workbench`" in prompt
    assert "`content-workflow-material`" in prompt
    assert '"workflow": "materials.assign"' in prompt
    assert '"asset_path":' in prompt
    assert '"reference_files":' in prompt
    assert "validation_evidence.json" in prompt
    assert "Workbench API quick contract" not in prompt
    assert "POST /sessions/<session_id>/commands" not in prompt


def test_vqa_refinement_image_inputs_use_current_final_views_only(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    final_dir = run_dir / "final_renders"
    evidence_dir = run_dir / "evidence_renders"
    final_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    for name in [
        "final_front_py.png",
        "final_oblique.png",
        "final_side_px.png",
        "final_top.png",
    ]:
        (final_dir / name).write_bytes(b"placeholder")
    initial_path = evidence_dir / "initial_oblique.png"
    initial_path.write_bytes(b"placeholder")

    inputs = runner._vqa_refinement_image_inputs(
        run_dir,
        [{"label": "initial", "path": str(initial_path)}],
    )

    assert [Path(item["path"]).name for item in inputs] == [
        "final_oblique.png",
        "final_front_py.png",
        "final_top.png",
    ]


def test_decision_patch_from_status_keyed_assignments_uses_source_paths_for_optimized(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "candidates": [
                    {
                        "runtime_path": "/Optimized/Conveyor/Roller_1",
                        "source_paths": ["/World/Conveyor/Roller_1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    patch = runner._decision_patch_from_assignments(
        tmp_path,
        {
            "assignments": {
                "material_assignment": [
                    {
                        "family": "rollers",
                        "material_name": "Brushed Steel",
                        "material_path": "/World/Looks/Brushed_Steel",
                        "prim_paths": ["/World/Conveyor/Roller_1"],
                        "rationale": "Reference rollers are metallic.",
                    }
                ],
                "preserved_existing": [
                    {
                        "family": "belt",
                        "prim_paths": ["/World/Conveyor/Belt"],
                        "rationale": "Existing dark belt matches.",
                    }
                ],
            }
        },
    )

    assert patch is not None
    assert patch["material_assignments"][0]["source_prim_paths"] == [
        "/World/Conveyor/Roller_1"
    ]
    assert "prim_paths" not in patch["material_assignments"][0]
    assert patch["reviewed_no_override"][0]["source_prim_paths"] == [
        "/World/Conveyor/Belt"
    ]


def test_material_grounding_samples_dark_defect_region(tmp_path: Path) -> None:
    image_path = tmp_path / "render.png"
    image = Image.new("RGB", (160, 120), (214, 214, 214))
    draw = ImageDraw.Draw(image)
    draw.rectangle((28, 24, 132, 94), fill=(8, 8, 8))
    draw.rectangle((52, 42, 70, 60), fill=(180, 0, 0))
    image.save(image_path)

    points = material_grounding.sample_grounding_pixels(
        image_path,
        issue_text="The broad black enclosure is too dark.",
        max_points=6,
    )

    assert points
    assert all(28 <= point["x"] <= 132 for point in points)
    assert all(24 <= point["y"] <= 94 for point in points)
    assert {point["mode"] for point in points} == {"dark"}


def test_material_grounding_diagnostics_ground_vqa_issue_to_source_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    final_dir = run_dir / "final_renders"
    trace_dir = run_dir / "trace"
    raw_dir.mkdir(parents=True)
    final_dir.mkdir()
    trace_dir.mkdir()

    final_image = final_dir / "final_oblique.png"
    image = Image.new("RGB", (160, 120), (214, 214, 214))
    ImageDraw.Draw(image).rectangle((26, 22, 136, 96), fill=(5, 5, 5))
    image.save(final_image)
    camera_json = final_dir / "final_oblique_camera.json"
    camera_json.write_text(
        json.dumps(
            {
                "camera_state": {
                    "eye": [3.0, -3.0, 2.0],
                    "target": [0.0, 0.0, 0.0],
                    "up": [0.0, 0.0, 1.0],
                },
                "image_width": 160,
                "image_height": 120,
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "final_render_records.json").write_text(
        json.dumps(
            [
                {
                    "name": "final_oblique",
                    "image_path": str(final_image),
                    "camera_json_path": str(camera_json),
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "visual_quality_assessment.json").write_text(
        json.dumps(
            {
                "status": "unresolved_issues",
                "issues_found": ["Wrong visible body part is black."],
                "issues_fixed": [],
                "unresolved_issues": ["Broad enclosure is incorrectly black."],
                "assessment_notes": "Needs grounded repair.",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "assignments": [
                    {
                        "family": "enclosure",
                        "coverage_status": "material_assignment",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": ["/World/Panel"],
                        "rationale": "Mistakenly treated as dark rubber.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "source_path": "/World/Panel",
                        "type_name": "Mesh",
                        "shape_hint": "blocky",
                        "inspection_paths": ["/World/Panel/Geometry"],
                        "bounds_samples": [{"size": [10.0, 8.0, 2.0]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_authoring_context.json").write_text(
        json.dumps(
            {
                "candidate_groups": [
                    {
                        "authoring_family": "enclosure",
                        "source_paths": ["/World/Panel"],
                        "size_hints": {"large": 1},
                        "shape_hints": {"blocky": 1},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    posted: list[tuple[str, dict[str, Any]]] = []

    def fake_post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
        posted.append((url, body))
        if url.endswith("/pick"):
            return {"prim_paths": ["/World/Panel/Geometry"]}
        if url.endswith("/paths/translate:batch"):
            return {
                "results": [
                    {
                        "inspection_paths": ["/World/Panel/Geometry"],
                        "source_paths": ["/World/Panel"],
                    }
                ]
            }
        if url.endswith("/material-binding:batch"):
            return {
                "results": [
                    {
                        "prim_path": "/World/Panel/Geometry",
                        "material_name": "Rubber Black Matte",
                    }
                ]
            }
        if url.endswith("/render"):
            return {
                "image_url": "/artifacts/outline.png",
                "camera_json_url": "/artifacts/outline_camera.json",
            }
        return {"ok": True}

    def fake_download_to_file(_url: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".png":
            Image.new("RGB", (160, 120), (80, 80, 80)).save(path)
        else:
            path.write_text(
                json.dumps({"camera_state": {"eye": [1, 1, 1]}}),
                encoding="utf-8",
            )

    monkeypatch.setattr(material_grounding, "_post_json", fake_post_json)
    monkeypatch.setattr(material_grounding, "_download_to_file", fake_download_to_file)

    paths = material_grounding.run_material_grounding_diagnostics(
        material_grounding.MaterialGroundingConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
        )
    )

    assert paths is not None
    aggregate = json.loads(
        (raw_dir / "material_grounding_diagnostics.json").read_text(encoding="utf-8")
    )
    issue = aggregate["latest"]["issues"][0]
    assert issue["grounded_source_paths"] == ["/World/Panel"]
    evidence = issue["views"][0]["source_path_evidence"][0]
    assert evidence["current_assignment"]["material_name"] == "Rubber Black Matte"
    assert evidence["candidate"]["relative_size"] == "very_large"
    assert any(
        "do not treat it as a small fastener" in hint
        for hint in evidence["diagnosis_hints"]
    )
    assert Path(issue["views"][0]["outline_render_path"]).is_file()

    counts = json.loads((run_dir / "api_operation_counts.json").read_text())
    assert counts["pick_calls"] > 0
    assert counts["grounding_pick_calls"] == counts["pick_calls"]
    assert any(url.endswith("/pick") for url, _body in posted)


def test_material_grounding_records_failed_view_without_aborting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        material_grounding,
        "_ground_view",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("render unavailable")),
    )

    issue = material_grounding._ground_issue(
        config=material_grounding.MaterialGroundingConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=tmp_path,
            session_id="session-1",
        ),
        issue_index=1,
        issue_text="Wrong color.",
        final_render_records=[{"name": "final_oblique", "image_path": "missing.png"}],
        grounding_dir=tmp_path,
        assignment_by_path={},
        candidate_by_path={},
        authoring_group_by_path={},
        size_rank_by_path={},
    )

    assert issue["views"][0]["skip_reason"] == "grounding_view_failed"
    assert issue["views"][0]["error"] == "render unavailable"


def test_run_material_assignment_preserves_child_failure_exit_code_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )

    def fake_run_child_agent(**kwargs: object) -> int:
        child_final_path = kwargs["child_final_path"]
        assert isinstance(child_final_path, Path)
        child_final_path.write_text("recovered", encoding="utf-8")
        return 7

    monkeypatch.setattr(runner, "wait_for_workbench", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_run_child_agent", fake_run_child_agent)
    monkeypatch.setattr(
        runner,
        "_ensure_material_assignment_artifacts",
        lambda **_kwargs: True,
    )
    monkeypatch.delenv(runner.ALLOW_FALLBACK_SUCCESS_ENV, raising=False)
    monkeypatch.delenv(runner.DISABLE_FALLBACK_SUCCESS_ENV, raising=False)

    result = runner.run_material_assignment(
        MaterialAssignConfig(
            repo_root=tmp_path,
            usd_path=usd,
            reference_images=[reference],
            materials_yaml=materials_yaml,
            materials_usd=materials_usd,
            workbench_url="http://127.0.0.1:8088",
            start_workbench=False,
            preflight=False,
            output_dir=tmp_path / "run",
        )
    )

    assert result.returncode == 7
    metrics = json.loads(
        (result.run_dir / "run_cost_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["wall_time_seconds"] >= 0
    assert metrics["repeated_file_reads_total"] is None
    assert metrics["failed_api_calls"] is None
    assert metrics["retried_api_calls"] is None
    assert metrics["context"]["asset"] == str(usd)
    events = [
        json.loads(line)
        for line in (result.run_dir / "trace" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    recovery_events = [
        event
        for event in events
        if event["phase"] == "runner" and event["event_type"] == "warning"
    ]
    assert recovery_events[-1]["data"]["fallback_success_enabled"] is False
    assert recovery_events[-1]["data"]["allow_env"] == runner.ALLOW_FALLBACK_SUCCESS_ENV
    assert recovery_events[-1]["data"]["effective_returncode"] == 7


@pytest.mark.parametrize(
    "failure_mode",
    ["pending_annotation", "restore", "succeeded_annotation"],
)
def test_run_material_assignment_distinguishes_restore_and_status_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    usd = tmp_path / "asset.usda"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usda"
    output_usd = tmp_path / "materialized.usda"
    run_dir = tmp_path / "run"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usda"\nentries: []\n',
        encoding="utf-8",
    )

    def fake_run_child_agent(**kwargs: object) -> int:
        child_final_path = kwargs["child_final_path"]
        assert isinstance(child_final_path, Path)
        child_final_path.write_text("done", encoding="utf-8")
        return 0

    def fake_ensure_artifacts(**kwargs: object) -> bool:
        artifact_run_dir = kwargs["run_dir"]
        assert isinstance(artifact_run_dir, Path)
        (artifact_run_dir / "assignments.json").write_text(
            json.dumps(
                {
                    "coverage": {
                        "candidate_visible_prim_count": 51,
                        "material_assignment_prim_count": 51,
                        "ambiguous_unassigned_prim_count": 0,
                    },
                    "assignments": [],
                }
            ),
            encoding="utf-8",
        )
        (artifact_run_dir / "final_summary.md").write_text(
            "# Final Summary\n\nCoverage: 51/51\nTotal uncovered: 0\n",
            encoding="utf-8",
        )
        return True

    restore_calls = 0

    def fake_restore_materialized_output(**_kwargs: object) -> Path:
        nonlocal restore_calls
        restore_calls += 1
        if failure_mode == "restore":
            raise RuntimeError("HTTP 500: Internal server error")
        output_usd.write_text("#usda 1.0\n", encoding="utf-8")
        return output_usd

    original_record_status = runner._record_materialized_output_status

    def fake_record_status(
        *,
        run_dir: Path,
        output_usd_path: Path,
        status: str,
        error: Exception | None = None,
    ) -> None:
        if failure_mode == f"{status}_annotation":
            raise RuntimeError("status artifact is read-only")
        original_record_status(
            run_dir=run_dir,
            output_usd_path=output_usd_path,
            status=status,
            error=error,
        )

    monkeypatch.setattr(runner, "wait_for_workbench", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "prepare_material_run_packet",
        lambda _config: {"session_id": "session-1", "initial_evidence_renders": []},
    )
    monkeypatch.setattr(runner, "packet_image_inputs", lambda _packet: [])
    monkeypatch.setattr(
        runner,
        "_build_material_assignment_child_prompt",
        lambda **_kwargs: "prompt",
    )
    monkeypatch.setattr(runner, "_run_child_agent", fake_run_child_agent)
    monkeypatch.setattr(
        runner,
        "_finalize_structured_material_decisions",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        runner,
        "_ensure_material_assignment_artifacts",
        fake_ensure_artifacts,
    )
    monkeypatch.setattr(
        runner,
        "_snapshot_material_step_artifacts",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "_run_vqa_refinement_loop",
        lambda **_kwargs: 0,
    )
    monkeypatch.setattr(
        runner,
        "_restore_materialized_output",
        fake_restore_materialized_output,
    )
    monkeypatch.setattr(
        runner,
        "_record_materialized_output_status",
        fake_record_status,
    )
    monkeypatch.setattr(
        runner,
        "close_workbench_session",
        lambda *_args, **_kwargs: None,
    )

    result = runner.run_material_assignment(
        MaterialAssignConfig(
            repo_root=tmp_path,
            usd_path=usd,
            reference_images=[reference],
            materials_yaml=materials_yaml,
            materials_usd=materials_usd,
            workbench_url="http://127.0.0.1:8088",
            start_workbench=False,
            preflight=True,
            output_dir=run_dir,
            output_usd_path=output_usd,
        )
    )

    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["coverage"]["material_assignment_prim_count"] == 51
    summary = (run_dir / "final_summary.md").read_text(encoding="utf-8")
    assert "Coverage: 51/51" in summary
    events = [
        json.loads(line)
        for line in (run_dir / "trace" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if failure_mode == "pending_annotation":
        assert result.returncode == 2
        assert restore_calls == 0
        assert not output_usd.exists()
        assert "materialized_usd" not in assignments
        assert "## Materialized USD" not in summary
        pending_event = next(
            event
            for event in events
            if event["summary"]
            == "Failed to record pending materialized USD status before restore."
        )
        assert pending_event["event_type"] == "error"
        assert pending_event["data"] == {
            "error_type": "RuntimeError",
            "error": "status artifact is read-only",
        }
    elif failure_mode == "restore":
        assert result.returncode == 2
        assert restore_calls == 1
        assert assignments["materialized_usd"]["status"] == "failed"
        assert assignments["materialized_usd"]["error"] == (
            "HTTP 500: Internal server error"
        )
        assert "- Status: **FAILED**" in summary
        assert "RuntimeError: HTTP 500: Internal server error" in summary
        restore_event = next(
            event
            for event in events
            if event["summary"]
            == "Failed to restore accepted materials to durable USD."
        )
        assert restore_event["data"] == {
            "error_type": "RuntimeError",
            "error": "HTTP 500: Internal server error",
        }
    else:
        assert result.returncode == 2
        assert restore_calls == 1
        assert output_usd.is_file()
        assert assignments["materialized_usd"]["status"] == "pending"
        assert "- Status: **PENDING**" in summary
        assert not any(
            event["summary"] == "Failed to restore accepted materials to durable USD."
            for event in events
        )
        annotation_event = next(
            event
            for event in events
            if event["summary"]
            == (
                "Restored durable USD but failed to record its successful "
                "materialization status."
            )
        )
        assert annotation_event["data"] == {
            "error_type": "RuntimeError",
            "error": "status artifact is read-only",
        }
        assert annotation_event["event_type"] == "error"


def test_fallback_success_can_be_enabled_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(runner.ALLOW_FALLBACK_SUCCESS_ENV, "1")

    assert runner._fallback_success_enabled() is True


def test_unrecognized_fallback_success_env_logs_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(runner.ALLOW_FALLBACK_SUCCESS_ENV, "maybe")

    with caplog.at_level("WARNING", logger=runner.__name__):
        assert runner._fallback_success_enabled() is False

    assert runner.ALLOW_FALLBACK_SUCCESS_ENV in caplog.text
    assert "Ignoring unrecognized" in caplog.text


def test_fallback_material_path_sanitizes_usd_name() -> None:
    assert (
        runner._fallback_material_path("Painted/Steel.v2 - 01")
        == "/World/Looks/Painted_Steel_v2_01"
    )


def test_run_cost_helpers_aggregate_multiple_child_turns(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "codex_result.json").write_text(
        json.dumps(
            {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cached_input_tokens": 3,
                    "total_tokens": 15,
                }
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "post_apply_vqa_result.json").write_text(
        json.dumps(
            {
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 4,
                    "cached_input_tokens": 2,
                    "total_tokens": 11,
                }
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "vqa_refinement_1_result.json").write_text(
        json.dumps({"usage": {"input_tokens": 8, "output_tokens": 3}}),
        encoding="utf-8",
    )
    (raw_dir / "codex_items.json").write_text(
        json.dumps([{"command": "python plan.py", "exit_code": 0}]),
        encoding="utf-8",
    )
    (raw_dir / "post_apply_vqa_items.json").write_text(
        json.dumps({"items": [{"command": "jq . assignments.json", "exit_code": 0}]}),
        encoding="utf-8",
    )

    assert runner._load_model_usage(tmp_path) == {
        "cached_input_tokens": 5,
        "input_tokens": 25,
        "output_tokens": 12,
        "total_tokens": 37,
    }
    assert runner._model_turn_count(tmp_path) == 3
    assert runner._load_child_command_records(tmp_path) == [
        {"command": "python plan.py", "exit_code": 0},
        {"command": "jq . assignments.json", "exit_code": 0},
    ]


def test_legacy_fallback_success_disable_env_false_enables_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(runner.ALLOW_FALLBACK_SUCCESS_ENV, raising=False)
    monkeypatch.setenv(runner.DISABLE_FALLBACK_SUCCESS_ENV, "0")

    assert runner._fallback_success_enabled() is True


def test_unrecognized_legacy_fallback_success_env_logs_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv(runner.ALLOW_FALLBACK_SUCCESS_ENV, raising=False)
    monkeypatch.setenv(runner.DISABLE_FALLBACK_SUCCESS_ENV, "maybe")

    with caplog.at_level("WARNING", logger=runner.__name__):
        assert runner._fallback_success_enabled() is False

    assert runner.DISABLE_FALLBACK_SUCCESS_ENV in caplog.text
    assert "Ignoring unrecognized" in caplog.text


def test_validate_config_rejects_output_usd_without_preflight(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--output-usd requires material preflight"):
        _validate_config(
            MaterialAssignConfig(
                repo_root=tmp_path,
                usd_path=tmp_path / "asset.usd",
                reference_images=[],
                materials_yaml=tmp_path / "materials.yaml",
                materials_usd=tmp_path / "materials.usd",
                workbench_url="http://127.0.0.1:8088",
                output_usd_path=tmp_path / "materialized.usd",
                preflight=False,
            )
        )


@pytest.mark.parametrize("use_symlink", [False, True])
def test_validate_config_rejects_output_usd_aliasing_source(
    tmp_path: Path,
    use_symlink: bool,
) -> None:
    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, reference, materials_yaml, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    output_usd = usd
    if use_symlink:
        output_usd = tmp_path / "materialized.usd"
        output_usd.symlink_to(usd)

    with pytest.raises(ValueError, match="must differ from --usd"):
        _validate_config(
            MaterialAssignConfig(
                repo_root=tmp_path,
                usd_path=usd,
                reference_images=[reference],
                materials_yaml=materials_yaml,
                materials_usd=materials_usd,
                workbench_url="http://127.0.0.1:8088",
                output_usd_path=output_usd,
            )
        )


def test_validate_config_rejects_non_loopback_auto_start(tmp_path: Path) -> None:
    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="loopback"):
        _validate_config(
            MaterialAssignConfig(
                repo_root=tmp_path,
                usd_path=usd,
                reference_images=[reference],
                materials_yaml=materials_yaml,
                materials_usd=materials_usd,
                workbench_url="http://0.0.0.0:8088",
                start_workbench=True,
            )
        )


def test_validate_config_rejects_workbench_url_with_path(tmp_path: Path) -> None:
    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="root http"):
        _validate_config(
            MaterialAssignConfig(
                repo_root=tmp_path,
                usd_path=usd,
                reference_images=[reference],
                materials_yaml=materials_yaml,
                materials_usd=materials_usd,
                workbench_url="http://127.0.0.1:8088/somepath",
            )
        )


def test_validate_config_rejects_negative_vqa_refinement_iterations(
    tmp_path: Path,
) -> None:
    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="vqa-refinement-max-iterations"):
        _validate_config(
            MaterialAssignConfig(
                repo_root=tmp_path,
                usd_path=usd,
                reference_images=[reference],
                materials_yaml=materials_yaml,
                materials_usd=materials_usd,
                workbench_url="http://127.0.0.1:8088",
                vqa_refinement_max_iterations=-1,
            )
        )


def test_validate_config_warns_when_timeout_disabled_without_managed_workbench(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )

    with caplog.at_level("WARNING", logger=runner.__name__):
        _validate_config(
            MaterialAssignConfig(
                repo_root=tmp_path,
                usd_path=usd,
                reference_images=[reference],
                materials_yaml=materials_yaml,
                materials_usd=materials_usd,
                workbench_url="http://127.0.0.1:8088",
                start_workbench=False,
                child_timeout_seconds=0,
            )
        )

    assert "--no-start-workbench leaves no managed Workbench watchdog" in caplog.text


def test_run_material_assignment_scopes_managed_workbench_material_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    usd_dir = repo_root / "assets"
    materials_dir = repo_root / "materials"
    usd_dir.mkdir(parents=True)
    materials_dir.mkdir()
    usd = usd_dir / "asset.usd"
    reference = repo_root / "reference.png"
    materials_yaml = materials_dir / "materials.yaml"
    materials_usd = materials_dir / "materials.usd"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )
    observed_roots: list[Path] = []

    class CapturingWorkbench:
        def __init__(
            self,
            *,
            material_library_roots: list[Path],
            **_kwargs: object,
        ) -> None:
            observed_roots.extend(material_library_roots)

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    def fake_run_child_agent(**kwargs: object) -> int:
        child_final_path = kwargs["child_final_path"]
        assert isinstance(child_final_path, Path)
        child_final_path.write_text("done", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "ManagedWorkbench", CapturingWorkbench)
    monkeypatch.setattr(runner, "_run_child_agent", fake_run_child_agent)
    monkeypatch.setattr(
        runner,
        "_ensure_material_assignment_artifacts",
        lambda **_kwargs: True,
    )

    result = runner.run_material_assignment(
        MaterialAssignConfig(
            repo_root=repo_root,
            usd_path=usd,
            reference_images=[reference],
            materials_yaml=materials_yaml,
            materials_usd=materials_usd,
            workbench_url="http://127.0.0.1:8088",
            start_workbench=True,
            preflight=False,
            output_dir=tmp_path / "run",
        )
    )

    assert result.returncode == 0
    assert observed_roots == [usd_dir, materials_dir]


def test_run_material_assignment_explains_material_library_root_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_dir = tmp_path / "asset"
    materials_dir = tmp_path / "materials"
    usd_dir.mkdir()
    materials_dir.mkdir()
    usd = usd_dir / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = materials_dir / "materials.yaml"
    materials_usd = materials_dir / "materials.usd"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )

    def fake_prepare_material_run_packet(_config: object) -> dict[str, object]:
        raise RuntimeError(
            "Workbench request failed for http://127.0.0.1:8088/sessions: "
            "HTTP 400: Material library path is outside "
            "CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS"
        )

    monkeypatch.setattr(runner, "wait_for_workbench", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "prepare_material_run_packet",
        fake_prepare_material_run_packet,
    )

    with pytest.raises(RuntimeError) as exc_info:
        runner.run_material_assignment(
            MaterialAssignConfig(
                repo_root=tmp_path,
                usd_path=usd,
                reference_images=[reference],
                materials_yaml=materials_yaml,
                materials_usd=materials_usd,
                workbench_url="http://127.0.0.1:8088",
                start_workbench=False,
                preflight=True,
                output_dir=tmp_path / "run",
            )
        )

    message = str(exc_info.value)
    assert "Content Workbench rejected the material library" in message
    assert "existing Workbench endpoint" in message
    assert "CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS" in message
    assert str(materials_dir.resolve()) in message
    assert str(usd_dir.resolve()) in message
    assert "Original Workbench error" in message
    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "trace" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[-1]["event_type"] == "preflight_failed"


def test_run_material_assignment_traces_workbench_stop_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )

    class FailingWorkbench:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            raise RuntimeError("SIGKILL failed")

        def returncode(self) -> None:
            return None

    def fake_run_child_agent(**kwargs: object) -> int:
        child_final_path = kwargs["child_final_path"]
        assert isinstance(child_final_path, Path)
        child_final_path.write_text("done", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "ManagedWorkbench", FailingWorkbench)
    monkeypatch.setattr(runner, "_run_child_agent", fake_run_child_agent)
    monkeypatch.setattr(
        runner,
        "_ensure_material_assignment_artifacts",
        lambda **_kwargs: True,
    )

    result = runner.run_material_assignment(
        MaterialAssignConfig(
            repo_root=tmp_path,
            usd_path=usd,
            reference_images=[reference],
            materials_yaml=materials_yaml,
            materials_usd=materials_usd,
            workbench_url="http://127.0.0.1:8088",
            start_workbench=True,
            preflight=False,
            output_dir=tmp_path / "run",
        )
    )

    assert result.returncode == 0
    events = [
        json.loads(line)
        for line in (result.run_dir / "trace" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    stop_events = [
        event for event in events if event["event_type"] == "workbench_stopped"
    ]
    assert stop_events[-1]["data"] == {
        "error_type": "RuntimeError",
        "error": "SIGKILL failed",
    }


def test_managed_workbench_sets_material_library_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_env: dict[str, str] = {}
    observed_kwargs: dict[str, object] = {}

    class FakeProcess:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            return 0

        def kill(self) -> None:
            return None

    def fake_popen(*_args: object, **kwargs: object) -> FakeProcess:
        observed_kwargs.update(kwargs)
        env = kwargs["env"]
        assert isinstance(env, dict)
        observed_env.update(env)
        return FakeProcess()

    monkeypatch.setattr(runner, "is_workbench_healthy", lambda _url: False)
    monkeypatch.setattr(runner, "wait_for_workbench", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setenv(
        "CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS", str(tmp_path / "pre")
    )
    monkeypatch.setenv("CONTENT_WORKBENCH_OUTPUT_ROOTS", str(tmp_path))

    workbench = runner.ManagedWorkbench(
        repo_root=tmp_path,
        workbench_url="http://127.0.0.1:8088",
        run_dir=tmp_path / "run",
        timeout_seconds=1.0,
        material_library_roots=[tmp_path / "materials", tmp_path / "asset"],
    )
    workbench.start()
    workbench.stop()

    roots = observed_env["CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS"].split(",")
    assert roots == [
        str(tmp_path / "pre"),
        str((tmp_path / "materials").resolve()),
        str((tmp_path / "asset").resolve()),
    ]
    assert observed_env["CONTENT_WORKBENCH_OUTPUT_ROOTS"] == str(
        (tmp_path / "run").resolve()
    )
    assert observed_kwargs["start_new_session"] is True


def test_managed_workbench_reused_service_requires_exact_run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_wait(
        workbench_url: str,
        *,
        timeout_seconds: float,
        output_root: Path | None = None,
    ) -> None:
        observed.update(
            workbench_url=workbench_url,
            timeout_seconds=timeout_seconds,
            output_root=output_root,
        )

    monkeypatch.setattr(runner, "is_workbench_healthy", lambda _url: True)
    monkeypatch.setattr(runner, "wait_for_workbench", fake_wait)
    run_dir = tmp_path / "run"
    workbench = runner.ManagedWorkbench(
        repo_root=tmp_path,
        workbench_url="http://127.0.0.1:8088",
        run_dir=run_dir,
        timeout_seconds=3.0,
    )

    workbench.start()

    assert observed == {
        "workbench_url": "http://127.0.0.1:8088",
        "timeout_seconds": 3.0,
        "output_root": run_dir,
    }
    assert workbench.process is None


def test_workbench_watchdog_fails_closed_if_output_root_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"

    class FakeResponse:
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "status": "healthy",
                    "service": "content-workbench",
                    "output_roots": [str(tmp_path.resolve())],
                }
            ).encode()

    monkeypatch.setattr(
        runner.workbench_client,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    watchdog = runner._make_workbench_watchdog(
        "http://127.0.0.1:8088",
        None,
        output_root=run_dir,
    )

    failure = watchdog()

    assert failure is not None
    assert failure.fatal
    assert "configuration changed" in failure.reason
    assert "run-scoped output root" in failure.reason


def test_every_child_workbench_watchdog_is_scoped_to_run_directory() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_make_workbench_watchdog"
    ]

    assert len(calls) == 4
    for call in calls:
        output_root = next(
            (
                keyword.value
                for keyword in call.keywords
                if keyword.arg == "output_root"
            ),
            None,
        )
        assert isinstance(output_root, ast.Name)
        assert output_root.id == "run_dir"


def test_managed_workbench_uses_preview_package_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agentic_root = tmp_path / "agentic"
    preview_workbench = agentic_root / "packages" / "content_workbench"
    preview_workbench.mkdir(parents=True)
    observed_env: dict[str, str] = {}

    class FakeProcess:
        pid = 1234

        def poll(self) -> int | None:
            return 0

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            return 0

        def kill(self) -> None:
            return None

    def fake_popen(*_args: object, **kwargs: object) -> FakeProcess:
        env = kwargs["env"]
        assert isinstance(env, dict)
        observed_env.update(env)
        return FakeProcess()

    monkeypatch.setattr(runner, "is_workbench_healthy", lambda _url: False)
    monkeypatch.setattr(runner, "wait_for_workbench", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    workbench = runner.ManagedWorkbench(
        repo_root=agentic_root,
        workbench_url="http://127.0.0.1:8088",
        run_dir=tmp_path / "run",
        timeout_seconds=1.0,
    )
    workbench.start()
    workbench.stop()

    pythonpath = observed_env["PYTHONPATH"].split(os.pathsep)
    assert str(preview_workbench.resolve()) in pythonpath


def test_codex_bridge_env_uses_preview_package_roots(tmp_path: Path) -> None:
    agentic_root = tmp_path / "agentic"
    expected_roots = [
        agentic_root / "packages" / "content_workflow_cli",
        agentic_root / "packages" / "content_workbench",
        agentic_root / "packages" / "content_workbench_agent_client",
        agentic_root / "packages" / "content_agent_workflows",
    ]
    for expected_root in expected_roots:
        expected_root.mkdir(parents=True)
    config = MaterialAssignConfig(
        repo_root=agentic_root,
        usd_path=tmp_path / "asset.usd",
        reference_images=[tmp_path / "reference.png"],
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        workbench_url="http://127.0.0.1:8088",
    )

    env = runner._codex_bridge_env(config)
    pythonpath = env["PYTHONPATH"].split(os.pathsep)

    for expected_root in expected_roots:
        assert str(expected_root.resolve()) in pythonpath


def test_managed_workbench_refuses_non_loopback_auto_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "is_workbench_healthy", lambda _url: False)
    workbench = runner.ManagedWorkbench(
        repo_root=tmp_path,
        workbench_url="http://0.0.0.0:8088",
        run_dir=tmp_path / "run",
        timeout_seconds=1.0,
    )

    with pytest.raises(ValueError, match="loopback"):
        workbench.start()


def test_is_workbench_healthy_requires_content_workbench_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"status":"healthy","service":"not-content-workbench"}'

    monkeypatch.setattr(
        runner.workbench_client,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )

    assert not runner.is_workbench_healthy("http://127.0.0.1:8088")


def test_is_workbench_healthy_accepts_content_workbench_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        status = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"status":"healthy","service":"content-workbench"}'

    monkeypatch.setattr(
        runner.workbench_client,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )

    assert runner.is_workbench_healthy("http://127.0.0.1:8088")


def test_close_workbench_session_uses_cleanup_timeout_and_encodes_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request: object, *, timeout: float) -> FakeResponse:
        observed["url"] = getattr(request, "full_url")
        observed["method"] = request.get_method()
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(runner.workbench_client, "urlopen", fake_urlopen)

    runner.close_workbench_session("http://127.0.0.1:8088", "session/one")

    assert observed["url"] == "http://127.0.0.1:8088/sessions/session%2Fone"
    assert observed["method"] == "DELETE"
    assert observed["timeout"] == 300.0


def test_managed_workbench_stop_terminates_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class FakeProcess:
        pid = 4321

        def poll(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            return 0

    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    workbench = runner.ManagedWorkbench(
        repo_root=tmp_path,
        workbench_url="http://127.0.0.1:8088",
        run_dir=tmp_path / "run",
        timeout_seconds=1.0,
    )
    workbench.process = FakeProcess()  # type: ignore[assignment]
    workbench.log_stream = StringIO()

    workbench.stop()

    assert calls == [(4321, signal.SIGTERM)]


def test_terminate_subprocess_surfaces_stubborn_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class StubbornProcess:
        pid = 1234

        def poll(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            raise subprocess.TimeoutExpired(cmd="child", timeout=timeout)

    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: calls.append((pid, sig)))

    with pytest.raises(RuntimeError, match="did not exit after SIGKILL"):
        runner._terminate_subprocess(StubbornProcess())  # type: ignore[arg-type]

    assert calls == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]


def test_terminate_subprocess_wraps_fallback_kill_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubbornProcess:
        pid = 1234

        def poll(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            raise subprocess.TimeoutExpired(cmd="child", timeout=timeout)

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            raise OSError("kill failed")

    def fake_killpg(_pid: int, _sig: int) -> None:
        raise OSError("group kill failed")

    monkeypatch.setattr(runner.os, "killpg", fake_killpg)

    with pytest.raises(RuntimeError, match="Failed to kill process 1234"):
        runner._terminate_subprocess(StubbornProcess())  # type: ignore[arg-type]


def test_terminate_subprocess_wraps_fallback_kill_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubbornProcess:
        pid = 1234
        killed = False

        def poll(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            raise subprocess.TimeoutExpired(cmd="child", timeout=timeout)

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

    def fake_killpg(_pid: int, _sig: int) -> None:
        raise OSError("group kill failed")

    process = StubbornProcess()
    monkeypatch.setattr(runner.os, "killpg", fake_killpg)

    with pytest.raises(RuntimeError, match="did not exit after SIGKILL"):
        runner._terminate_subprocess(process)  # type: ignore[arg-type]

    assert process.killed is True


def test_run_subprocess_with_timeout_does_not_signal_group_after_clean_exit(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls: list[tuple[int, int]] = []

    def fake_killpg(pid: int, sig: int) -> None:
        calls.append((pid, sig))

    monkeypatch.setattr(runner.os, "killpg", fake_killpg)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    log_stream = StringIO()

    returncode = _run_subprocess_with_timeout(
        command=[sys.executable, "-c", "print('done')"],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5,
        log_stream=log_stream,
        timeout_label="test child",
    )

    assert returncode == 0
    assert "done" in log_stream.getvalue()
    assert calls == []


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper regression")
@pytest.mark.parametrize("timeout_seconds", [0.0, 0.3])
def test_run_subprocess_reaps_detached_descendants_before_return(
    tmp_path: Path,
    timeout_seconds: float,
) -> None:
    descendant_pid_path = tmp_path / "descendant.pid"
    descendant_script = "import time; time.sleep(30)"
    target_lines = [
        "from pathlib import Path",
        "import subprocess, sys, time",
        (
            "child = subprocess.Popen("
            "[sys.executable, '-c', sys.argv[2]], start_new_session=True, "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
        ),
        "Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')",
    ]
    if timeout_seconds > 0:
        target_lines.append("time.sleep(30)")
    target_script = "\n".join(target_lines)

    def invoke() -> int:
        return _run_subprocess_with_timeout(
            command=[
                sys.executable,
                "-c",
                target_script,
                str(descendant_pid_path),
                descendant_script,
            ],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_seconds=timeout_seconds,
            log_stream=StringIO(),
            timeout_label="detached descendant test",
            heartbeat_interval_seconds=0,
            watchdog_interval_seconds=0,
        )

    if timeout_seconds > 0:
        with pytest.raises(TimeoutError, match="exceeded"):
            invoke()
    else:
        assert invoke() == 0

    descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
    assert not Path(f"/proc/{descendant_pid}").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper regression")
def test_run_subprocess_reaps_double_forked_descendant(tmp_path: Path) -> None:
    descendant_pid_path = tmp_path / "double-fork.pid"
    target_script = "\n".join(
        [
            "from pathlib import Path",
            "import os, sys, time",
            "first = os.fork()",
            "if first == 0:",
            "    os.setsid()",
            "    second = os.fork()",
            "    if second != 0:",
            "        Path(sys.argv[1]).write_text(str(second), encoding='utf-8')",
            "        os._exit(0)",
            "    time.sleep(30)",
            "    os._exit(0)",
            "os.waitpid(first, 0)",
        ]
    )

    returncode = _run_subprocess_with_timeout(
        command=[sys.executable, "-c", target_script, str(descendant_pid_path)],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5,
        log_stream=StringIO(),
        timeout_label="double-fork descendant test",
        heartbeat_interval_seconds=0,
        watchdog_interval_seconds=0,
    )

    assert returncode == 0
    descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
    assert not Path(f"/proc/{descendant_pid}").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper regression")
def test_run_subprocess_reaps_descendants_forked_during_cleanup(
    tmp_path: Path,
) -> None:
    descendant_pids_path = tmp_path / "fork-burst.pids"
    ready_path = tmp_path / "fork-burst.ready"
    target_script = "\n".join(
        [
            "from pathlib import Path",
            "import os, sys, time",
            "spawner = os.fork()",
            "if spawner == 0:",
            "    os.setsid()",
            "    pid_fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)",
            "    for index in range(64):",
            "        child = os.fork()",
            "        if child == 0:",
            "            time.sleep(30)",
            "            os._exit(0)",
            "        os.write(pid_fd, f'{child}\\n'.encode('ascii'))",
            "        if index == 7:",
            "            Path(sys.argv[2]).write_text('ready', encoding='ascii')",
            "        time.sleep(0.002)",
            "    os.close(pid_fd)",
            "    time.sleep(30)",
            "    os._exit(0)",
            "deadline = time.monotonic() + 5",
            "while not Path(sys.argv[2]).exists():",
            "    if time.monotonic() >= deadline:",
            "        raise SystemExit(2)",
            "    time.sleep(0.005)",
        ]
    )

    returncode = _run_subprocess_with_timeout(
        command=[
            sys.executable,
            "-c",
            target_script,
            str(descendant_pids_path),
            str(ready_path),
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=10,
        log_stream=StringIO(),
        timeout_label="fork-burst descendant test",
        heartbeat_interval_seconds=0,
        watchdog_interval_seconds=0,
    )

    assert returncode == 0
    descendant_pids = {
        int(value)
        for value in descendant_pids_path.read_text(encoding="ascii").splitlines()
    }
    assert len(descendant_pids) >= 8
    assert not [pid for pid in descendant_pids if Path(f"/proc/{pid}").exists()]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper regression")
def test_run_subprocess_blocks_child_from_killing_descendant_reaper(
    tmp_path: Path,
) -> None:
    descendant_pid_path = tmp_path / "descendant.pid"
    target_script = "\n".join(
        [
            "from pathlib import Path",
            "import os, signal, subprocess, sys",
            (
                "child = subprocess.Popen("
                "[sys.executable, '-c', 'import time; time.sleep(30)'], "
                "start_new_session=True, stdout=subprocess.DEVNULL, "
                "stderr=subprocess.DEVNULL)"
            ),
            "Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')",
            "try:",
            "    os.kill(os.getppid(), signal.SIGKILL)",
            "except PermissionError:",
            "    print('CONTROL_SIGNAL_BLOCKED', flush=True)",
            "else:",
            "    raise SystemExit(2)",
            "for target in (-1, -os.getppid()):",
            "    try:",
            "        os.kill(target, 0)",
            "    except PermissionError:",
            "        continue",
            "    raise SystemExit(3)",
            "print('CONTROL_GROUP_SIGNALS_BLOCKED', flush=True)",
        ]
    )
    log_stream = StringIO()

    returncode = _run_subprocess_with_timeout(
        command=[sys.executable, "-c", target_script, str(descendant_pid_path)],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5,
        log_stream=log_stream,
        timeout_label="descendant reaper signal guard test",
        heartbeat_interval_seconds=0,
        watchdog_interval_seconds=0,
    )

    assert returncode == 0
    assert "CONTROL_SIGNAL_BLOCKED" in log_stream.getvalue()
    assert "CONTROL_GROUP_SIGNALS_BLOCKED" in log_stream.getvalue()
    descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
    assert not Path(f"/proc/{descendant_pid}").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper regression")
def test_run_subprocess_masks_high_pid_bits_in_signal_guard(tmp_path: Path) -> None:
    target_script = "\n".join(
        [
            "import ctypes, errno, os",
            "seccomp = ctypes.CDLL('libseccomp.so.2')",
            "seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]",
            "seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int",
            "syscall_number = seccomp.seccomp_syscall_resolve_name(b'kill')",
            "if syscall_number < 0:",
            "    raise SystemExit(3)",
            "supervisor_pid = os.getppid()",
            "targets = [",
            "    supervisor_pid | (1 << 32),",
            "    ctypes.c_uint32(-supervisor_pid).value | (1 << 32),",
            "]",
            "libc = ctypes.CDLL(None, use_errno=True)",
            "for target in targets:",
            "    ctypes.set_errno(0)",
            (
                "    result = libc.syscall("
                "syscall_number, ctypes.c_uint64(target), ctypes.c_int(0))"
            ),
            "    error_number = ctypes.get_errno()",
            "    if result != -1 or error_number != errno.EPERM:",
            "        raise SystemExit(2)",
            "print('HIGH_PID_BITS_MASKED', flush=True)",
        ]
    )
    log_stream = StringIO()

    returncode = _run_subprocess_with_timeout(
        command=[sys.executable, "-c", target_script],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5,
        log_stream=log_stream,
        timeout_label="descendant reaper masked pid guard test",
        heartbeat_interval_seconds=0,
        watchdog_interval_seconds=0,
    )

    assert returncode == 0
    assert "HIGH_PID_BITS_MASKED" in log_stream.getvalue()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper regression")
def test_run_subprocess_masks_high_command_bits_in_signal_guard(
    tmp_path: Path,
) -> None:
    target_script = "\n".join(
        [
            "import ctypes, errno, os, socket",
            "seccomp = ctypes.CDLL('libseccomp.so.2')",
            "seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]",
            "seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int",
            "fcntl_number = seccomp.seccomp_syscall_resolve_name(b'fcntl')",
            "ioctl_number = seccomp.seccomp_syscall_resolve_name(b'ioctl')",
            "if fcntl_number < 0 or ioctl_number < 0:",
            "    raise SystemExit(3)",
            "read_fd, write_fd = os.pipe()",
            "left, right = socket.socketpair()",
            "libc = ctypes.CDLL(None, use_errno=True)",
            "owner = ctypes.c_int(os.getpid())",
            "calls = [",
            "    (fcntl_number, read_fd, (1 << 32) | 8, owner),",
            (
                "    (ioctl_number, left.fileno(), (1 << 32) | 0x8901, "
                "ctypes.pointer(owner)),"
            ),
            "]",
            "try:",
            "    for syscall_number, descriptor, command, argument in calls:",
            "        ctypes.set_errno(0)",
            "        result = libc.syscall(",
            "            syscall_number,",
            "            descriptor,",
            "            ctypes.c_uint64(command),",
            "            argument,",
            "        )",
            "        if result != -1 or ctypes.get_errno() != errno.EPERM:",
            "            raise SystemExit(2)",
            "finally:",
            "    os.close(read_fd)",
            "    os.close(write_fd)",
            "    left.close()",
            "    right.close()",
            "print('HIGH_COMMAND_BITS_MASKED', flush=True)",
        ]
    )
    log_stream = StringIO()

    returncode = _run_subprocess_with_timeout(
        command=[sys.executable, "-c", target_script],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5,
        log_stream=log_stream,
        timeout_label="descendant reaper masked command guard test",
        heartbeat_interval_seconds=0,
        watchdog_interval_seconds=0,
    )

    assert returncode == 0
    assert "HIGH_COMMAND_BITS_MASKED" in log_stream.getvalue()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper regression")
def test_run_subprocess_blocks_pidfd_signal_attack_on_reaper(
    tmp_path: Path,
) -> None:
    target_script = "\n".join(
        [
            "import ctypes, errno, os",
            "seccomp = ctypes.CDLL('libseccomp.so.2')",
            "seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]",
            "seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int",
            (
                "syscall_number = "
                "seccomp.seccomp_syscall_resolve_name(b'pidfd_send_signal')"
            ),
            "if syscall_number < 0:",
            "    raise SystemExit(3)",
            "target_fd = os.open(f'/proc/{os.getppid()}', os.O_RDONLY)",
            "try:",
            "    libc = ctypes.CDLL(None, use_errno=True)",
            "    result = libc.syscall(syscall_number, target_fd, 0, 0, 0)",
            "    error_number = ctypes.get_errno()",
            "finally:",
            "    os.close(target_fd)",
            "if result == -1 and error_number == errno.EPERM:",
            "    print('PIDFD_SIGNAL_BLOCKED', flush=True)",
            "    raise SystemExit(0)",
            "raise SystemExit(2)",
        ]
    )
    log_stream = StringIO()

    returncode = _run_subprocess_with_timeout(
        command=[sys.executable, "-c", target_script],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5,
        log_stream=log_stream,
        timeout_label="descendant reaper pidfd signal guard test",
        heartbeat_interval_seconds=0,
        watchdog_interval_seconds=0,
    )

    assert returncode == 0
    assert "PIDFD_SIGNAL_BLOCKED" in log_stream.getvalue()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper regression")
def test_run_subprocess_blocks_async_io_signal_attack_on_reaper(
    tmp_path: Path,
) -> None:
    target_pid_path = tmp_path / "target.pid"
    target_script = "\n".join(
        [
            "from pathlib import Path",
            "import fcntl, os, signal, sys, time",
            "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8')",
            "read_fd, write_fd = os.pipe()",
            "try:",
            "    fcntl.fcntl(read_fd, fcntl.F_SETOWN, os.getppid())",
            "    fcntl.fcntl(read_fd, fcntl.F_SETSIG, signal.SIGKILL)",
            ("    fcntl.fcntl(read_fd, fcntl.F_SETFL, os.O_ASYNC | os.O_NONBLOCK)"),
            "    os.write(write_fd, b'x')",
            "except PermissionError:",
            "    print('ASYNC_SIGNAL_BLOCKED', flush=True)",
            "    raise SystemExit(0)",
            "time.sleep(30)",
        ]
    )
    log_stream = StringIO()
    target_pid: int | None = None
    try:
        returncode = _run_subprocess_with_timeout(
            command=[sys.executable, "-c", target_script, str(target_pid_path)],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_seconds=5,
            log_stream=log_stream,
            timeout_label="descendant reaper async signal guard test",
            heartbeat_interval_seconds=0,
            watchdog_interval_seconds=0,
        )
        target_pid = int(target_pid_path.read_text(encoding="utf-8"))
    finally:
        if target_pid is None and target_pid_path.exists():
            target_pid = int(target_pid_path.read_text(encoding="utf-8"))
        if target_pid is not None and Path(f"/proc/{target_pid}").exists():
            os.kill(target_pid, signal.SIGKILL)

    assert returncode == 0
    assert "ASYNC_SIGNAL_BLOCKED" in log_stream.getvalue()
    assert target_pid is not None
    assert not Path(f"/proc/{target_pid}").exists()


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bwrap") is None,
    reason="Linux bubblewrap compatibility regression",
)
def test_descendant_reaper_guard_preserves_native_sandbox_primitives(
    tmp_path: Path,
) -> None:
    returncode = _run_subprocess_with_timeout(
        command=[
            shutil.which("bwrap") or "bwrap",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--",
            "/bin/true",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5,
        log_stream=StringIO(),
        timeout_label="native sandbox compatibility test",
        heartbeat_interval_seconds=0,
        watchdog_interval_seconds=0,
    )

    assert returncode == 0
    assert {
        "bpf",
        "clone",
        "clone3",
        "io_uring_enter",
        "mount",
        "socket",
        "socketpair",
        "unshare",
    }.isdisjoint(descendant_reaper._CONTROL_SYSCALL_RULES)
    assert {
        "clone",
        "io_uring_enter",
        "mount",
        "socket",
        "unshare",
    }.isdisjoint(descendant_reaper._CONTROL_COMMAND_RULES)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper regression")
def test_descendant_reaper_guard_failure_does_not_exec_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "provider-started"

    def fail_filter(**_kwargs: object) -> None:
        raise OSError("missing libseccomp")

    monkeypatch.setattr(
        descendant_reaper,
        "_load_control_plane_filter",
        fail_filter,
    )
    parent_pid = os.getppid()
    try:
        returncode = descendant_reaper._exec_guarded_child(
            [
                str(parent_pid),
                str(parent_pid),
                "--",
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).touch()",
            ]
        )
    finally:
        libc = descendant_reaper.ctypes.CDLL(None, use_errno=True)
        libc.prctl(descendant_reaper._PR_SET_PDEATHSIG, 0, 0, 0, 0)

    assert returncode == 125
    assert not marker.exists()


def test_run_subprocess_with_timeout_does_not_signal_group_after_exited_exception(
    tmp_path: Path, monkeypatch: Any
) -> None:
    class FakeProcess:
        pid = 12345
        stdout = None
        returncode = 0

        def poll(self) -> int:
            return 0

    drain_calls = 0
    group_calls: list[int] = []

    def fake_popen(*_args: Any, **_kwargs: Any) -> FakeProcess:
        return FakeProcess()

    def fake_drain(
        _output_queue: Any,
        _log_stream: Any,
        _console_stream: Any | None = None,
    ) -> None:
        nonlocal drain_calls
        drain_calls += 1
        if drain_calls == 2:
            raise RuntimeError("drain failed")

    def fake_terminate_group(pid: int | None) -> None:
        if pid is not None:
            group_calls.append(pid)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner, "_drain_subprocess_output", fake_drain)
    monkeypatch.setattr(runner, "_terminate_subprocess_group", fake_terminate_group)

    with pytest.raises(RuntimeError, match="drain failed"):
        _run_subprocess_with_timeout(
            command=[sys.executable, "-c", "pass"],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_seconds=5,
            log_stream=StringIO(),
            timeout_label="test child",
        )

    assert group_calls == []


def test_run_subprocess_with_timeout_rejects_missing_returncode(
    tmp_path: Path, monkeypatch: Any
) -> None:
    class FakeProcess:
        pid = 12345
        stdout = None
        returncode = None

        def poll(self) -> int:
            return 0

    def fake_popen(*_args: Any, **_kwargs: Any) -> FakeProcess:
        return FakeProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="test child exited without a return code"):
        _run_subprocess_with_timeout(
            command=[sys.executable, "-c", "pass"],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_seconds=5,
            log_stream=StringIO(),
            timeout_label="test child",
        )


@pytest.mark.parametrize("supervisor_returncode", [125, -signal.SIGKILL, None])
def test_run_subprocess_exception_fails_closed_on_unproven_reaping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supervisor_returncode: int | None,
) -> None:
    class FakeProcess:
        pid = 12345
        stdin = None
        stdout = None
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            if self.returncode is None:
                raise subprocess.TimeoutExpired(cmd="child", timeout=timeout)
            return self.returncode

    process = FakeProcess()

    def fake_terminate(_process: FakeProcess) -> None:
        process.returncode = supervisor_returncode

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(runner, "_terminate_subprocess", fake_terminate)
    monkeypatch.setattr(
        runner,
        "_drain_subprocess_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TimeoutError("primary timeout")
        ),
    )

    with pytest.raises(
        UnsafeRunArtifactError,
        match="could not prove that all child descendants were reaped",
    ):
        _run_subprocess_with_timeout(
            command=[sys.executable, "-c", "pass"],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_seconds=5,
            log_stream=StringIO(),
            timeout_label="test child",
        )


def test_assignment_uncertainty_detects_structured_no_hit_pick() -> None:
    uncertainty = _assignment_uncertainty(
        events=[
            {"event_type": "pick", "data": {"prim_paths": []}},
            {"event_type": "pick", "data": {"prim_paths": ["/World/NoneMesh"]}},
        ],
        child_returncode=0,
        final_render_count=1,
    )

    assert uncertainty == [
        "Some pixel-pick events returned no prim path; fallback used successful override records and hierarchy evidence."
    ]


def test_validate_config_rejects_reference_image_directory(tmp_path: Path) -> None:
    usd = tmp_path / "asset.usd"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, materials_yaml, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=usd,
        reference_images=[tmp_path],
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        workbench_url="http://127.0.0.1:8088",
    )

    with pytest.raises(ValueError, match="reference image 1 is not a file"):
        _validate_config(config)


def test_validate_config_accepts_reference_file_without_images(tmp_path: Path) -> None:
    usd = tmp_path / "asset.usd"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    reference_pdf = tmp_path / "reference.pdf"
    for path in [usd, materials_yaml, materials_usd, reference_pdf]:
        path.write_text("placeholder", encoding="utf-8")
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=usd,
        reference_images=[],
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        workbench_url="http://127.0.0.1:8088",
        reference_files=[reference_pdf],
    )

    _validate_config(config)


def test_validate_config_rejects_missing_references(tmp_path: Path) -> None:
    usd = tmp_path / "asset.usd"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, materials_yaml, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=usd,
        reference_images=[],
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        workbench_url="http://127.0.0.1:8088",
        reference_files=[],
    )

    with pytest.raises(ValueError, match="--reference-image or --reference"):
        _validate_config(config)


def test_run_subprocess_with_timeout_emits_heartbeat(tmp_path: Path) -> None:
    log_stream = StringIO()

    returncode = _run_subprocess_with_timeout(
        command=[
            sys.executable,
            "-c",
            "import time; time.sleep(0.4); print('done')",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5,
        log_stream=log_stream,
        timeout_label="test child",
        heartbeat_interval_seconds=0.1,
        watchdog_interval_seconds=0,
    )

    output = log_stream.getvalue()
    assert returncode == 0
    assert "test child still running" in output
    assert "done" in output


def test_run_subprocess_with_timeout_emits_progress_heartbeat(tmp_path: Path) -> None:
    log_stream = StringIO()

    returncode = _run_subprocess_with_timeout(
        command=[
            sys.executable,
            "-c",
            "import time; time.sleep(0.4); print('done')",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5,
        log_stream=log_stream,
        timeout_label="test child",
        heartbeat_interval_seconds=0.1,
        watchdog_interval_seconds=0,
        progress_reporter=lambda run_dir: f"run_dir={run_dir.name}; item=running",
        run_dir=tmp_path,
    )

    output = log_stream.getvalue()
    assert returncode == 0
    assert "test child still running" in output
    assert "progress: run_dir=" in output
    assert "item=running" in output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX artifact types regression")
@pytest.mark.parametrize("artifact_kind", ["fifo", "symlink", "devzero", "oversized"])
def test_terminal_progress_read_rejects_blocking_or_unbounded_artifacts(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "large_scene_run.json"
    completed = json.dumps(
        {
            "current_phase": None,
            "failed_at": None,
            "phases": {
                "decomposition": {"status": "completed"},
                "asset_task_processing": {"status": "completed"},
                "collection": {"status": "completed"},
            },
        }
    )
    if artifact_kind == "fifo":
        os.mkfifo(artifact)
    elif artifact_kind == "symlink":
        outside = tmp_path / "outside.json"
        outside.write_text(completed, encoding="utf-8")
        artifact.symlink_to(outside)
    elif artifact_kind == "devzero":
        if not Path("/dev/zero").exists():
            pytest.skip("/dev/zero is unavailable")
        artifact.symlink_to("/dev/zero")
    else:
        artifact.write_bytes(b" " * (runner.LIVE_PROGRESS_MAX_BYTES + 1))

    started = time.monotonic()
    assert runner._terminal_success_summary_for_log(run_dir) is None
    assert time.monotonic() - started < 1.0


def test_terminal_progress_read_rejects_leaf_swap_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "large_scene_run.json"
    artifact.write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps(
            {
                "current_phase": None,
                "failed_at": None,
                "phases": {
                    "decomposition": {"status": "completed"},
                    "asset_task_processing": {"status": "completed"},
                    "collection": {"status": "completed"},
                },
            }
        ),
        encoding="utf-8",
    )
    real_open = os.open
    swapped = False

    def racing_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == artifact.name and dir_fd is not None and not swapped:
            swapped = True
            artifact.unlink()
            artifact.symlink_to(outside)
        return real_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

    monkeypatch.setattr(runner.os, "open", racing_open)

    assert runner._terminal_success_summary_for_log(run_dir) is None
    assert swapped


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX FIFO regression")
def test_progress_summary_rejects_fifo_jsonl_without_blocking(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    trace_dir = run_dir / "trace"
    trace_dir.mkdir(parents=True)
    os.mkfifo(trace_dir / "events.jsonl")

    started = time.monotonic()
    assert runner._run_progress_summary_for_log(run_dir) is None
    assert time.monotonic() - started < 1.0


@pytest.mark.parametrize(
    "exception_type",
    [ValueError, RecursionError],
)
def test_terminal_progress_json_parser_failures_are_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[Exception],
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "large_scene_run.json").write_text("{}", encoding="utf-8")

    def fail_loads(_raw: object) -> object:
        raise exception_type("adversarial JSON")

    monkeypatch.setattr(runner.json, "loads", fail_loads)

    assert runner._terminal_success_summary_for_log(run_dir) is None


@pytest.mark.parametrize(
    "exception_type",
    [ValueError, RecursionError],
)
def test_progress_jsonl_parser_failures_are_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[Exception],
) -> None:
    run_dir = tmp_path / "run"
    trace_dir = run_dir / "trace"
    trace_dir.mkdir(parents=True)
    (trace_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")

    def fail_loads(_raw: object) -> object:
        raise exception_type("adversarial JSONL")

    monkeypatch.setattr(runner.json, "loads", fail_loads)

    summary = runner._run_progress_summary_for_log(run_dir)
    assert summary is not None
    assert "last_event=" not in summary


def test_progress_discovery_ignores_linked_asset_dir_and_latest_file(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside_dir = tmp_path / "outside-asset-tasks"
    outside_dir.mkdir()
    (run_dir / "02-asset-tasks").symlink_to(outside_dir, target_is_directory=True)
    outside_file = tmp_path / "outside.log"
    outside_file.write_text("outside", encoding="utf-8")
    (run_dir / "child-output.log").symlink_to(outside_file)

    assert runner._active_asset_task_dir(run_dir) is None
    assert runner._latest_progress_artifact(run_dir) is None


def test_active_asset_task_discovery_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    yielded = 0

    class FakeEntry:
        name = "child-noise"

    def entries() -> Iterator[FakeEntry]:
        nonlocal yielded
        for _index in range(runner.LIVE_PROGRESS_MAX_DIRECTORY_ENTRIES + 1):
            yielded += 1
            yield FakeEntry()

    class FakeScandir:
        def __enter__(self) -> Iterator[FakeEntry]:
            return entries()

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(runner.os, "scandir", lambda _fd: FakeScandir())

    assert runner._active_asset_task_dir(run_dir) is None
    assert yielded == runner.LIVE_PROGRESS_MAX_DIRECTORY_ENTRIES + 1


def test_run_subprocess_with_timeout_stops_on_terminal_scene_success(
    tmp_path: Path,
) -> None:
    (tmp_path / "large_scene_run.json").write_text(
        json.dumps(
            {
                "current_phase": None,
                "failed_at": None,
                "phases": {
                    "decomposition": {"status": "completed"},
                    "asset_task_processing": {"status": "completed"},
                    "collection": {"status": "completed"},
                },
            }
        ),
        encoding="utf-8",
    )
    log_stream = StringIO()

    returncode = _run_subprocess_with_timeout(
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5,
        log_stream=log_stream,
        timeout_label="test child",
        heartbeat_interval_seconds=0,
        watchdog_interval_seconds=0,
        terminal_success_detector=runner._terminal_success_summary_for_log,
        terminal_success_grace_seconds=0.1,
        run_dir=tmp_path,
    )

    output = log_stream.getvalue()
    assert returncode == 0
    assert "terminal run state detected" in output
    assert "terminating child and treating run as successful" in output


@pytest.mark.parametrize(
    ("supervisor_returncode", "expected_error"),
    [
        (125, "descendant supervisor failed"),
        (-signal.SIGKILL, "descendant supervisor died from signal 9"),
    ],
)
def test_run_subprocess_with_timeout_rejects_supervisor_failure_after_terminal_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supervisor_returncode: int,
    expected_error: str,
) -> None:
    class FakeProcess:
        pid = 12345
        stdin = None
        stdout = None
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            if self.returncode is None:
                raise subprocess.TimeoutExpired(cmd="child", timeout=timeout)
            return self.returncode

    process = FakeProcess()

    def fake_popen(*_args: Any, **_kwargs: Any) -> FakeProcess:
        return process

    def fake_terminate(_process: FakeProcess) -> None:
        process.returncode = supervisor_returncode

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner, "_terminate_subprocess", fake_terminate)

    with pytest.raises(RuntimeError, match=expected_error):
        _run_subprocess_with_timeout(
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_seconds=5,
            log_stream=StringIO(),
            timeout_label="test child",
            heartbeat_interval_seconds=0,
            watchdog_interval_seconds=0,
            terminal_success_detector=lambda _run_dir: "completed",
            terminal_success_grace_seconds=0,
            run_dir=tmp_path,
        )


def test_run_progress_summary_for_log_reports_large_scene_state(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    asset_dir = run_dir / "02-asset-tasks"
    (run_dir / "01-decomposition").mkdir(parents=True)
    (run_dir / "trace").mkdir()
    asset_dir.mkdir()
    (asset_dir / "agent_plan").mkdir()

    (run_dir / "large_scene_run.json").write_text(
        json.dumps(
            {
                "current_phase": "asset_task_processing",
                "phases": {
                    "decomposition": {"status": "completed"},
                    "asset_task_processing": {"status": "running"},
                    "collection": {"status": "pending"},
                },
            }
        ),
        encoding="utf-8",
    )
    (asset_dir / "asset_task_run_state.json").write_text(
        json.dumps(
            {
                "work_items": [
                    {"status": "completed"},
                    {"status": "running"},
                    {"status": "planned"},
                    {"status": "planned"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (asset_dir / "asset_task_results_index.json").write_text(
        json.dumps({"entries": [{"work_item_id": "material:obj_001"}]}),
        encoding="utf-8",
    )
    (asset_dir / "agent_plan" / "revision-0001.md").write_text(
        "plan",
        encoding="utf-8",
    )

    summary = runner._run_progress_summary_for_log(run_dir)

    assert summary is not None
    assert "scene phase=asset_task_processing" in summary
    assert "decomposition=completed" in summary
    assert "asset_task_processing=running" in summary
    assert "collection=pending" in summary
    assert "workflow2 dir=02-asset-tasks, results=1/4" in summary
    assert "completed=1" in summary
    assert "planned=2" in summary
    assert "running=1" in summary
    assert "latest=" in summary


def test_run_progress_summary_for_log_uses_latest_asset_task_retry_dir(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    stale_dir = run_dir / "02-asset-tasks"
    active_dir = run_dir / "02-asset-tasks-r3"
    stale_dir.mkdir(parents=True)
    active_dir.mkdir()

    (stale_dir / "asset_task_run_state.json").write_text(
        json.dumps({"work_items": [{"status": "planned"}]}),
        encoding="utf-8",
    )
    (stale_dir / "asset_task_results_index.json").write_text(
        json.dumps({"entries": []}),
        encoding="utf-8",
    )
    (active_dir / "asset_task_run_state.json").write_text(
        json.dumps(
            {
                "work_items": [
                    {"status": "completed"},
                    {"status": "completed"},
                    {"status": "running"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (active_dir / "asset_task_results_index.json").write_text(
        json.dumps({"entries": [{}, {}]}),
        encoding="utf-8",
    )
    shared_mtime = 1_700_000_000
    for path in (
        stale_dir,
        active_dir,
        stale_dir / "asset_task_run_state.json",
        stale_dir / "asset_task_results_index.json",
        active_dir / "asset_task_run_state.json",
        active_dir / "asset_task_results_index.json",
    ):
        os.utime(path, (shared_mtime, shared_mtime))

    summary = runner._run_progress_summary_for_log(run_dir)

    assert summary is not None
    assert "workflow2 dir=02-asset-tasks-r3, results=2/3" in summary
    assert "completed=2" in summary
    assert "running=1" in summary


def test_run_progress_summary_for_log_reports_single_material_state(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "material-run"
    raw_dir = run_dir / "raw"
    trace_dir = run_dir / "trace"
    raw_dir.mkdir(parents=True)
    trace_dir.mkdir()
    (run_dir / "request.json").write_text(
        json.dumps({"workflow": "materials.assign"}),
        encoding="utf-8",
    )
    (raw_dir / "material_decision_patch.json").write_text(
        json.dumps(
            {
                "material_assignments": [{}, {}, {}],
                "reviewed_no_override": [{}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "assignments.json").write_text(
        json.dumps(
            {
                "assignments": [{}, {}],
                "coverage": {
                    "material_decision_prim_count": 4,
                    "missing_assignment_prim_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "events.jsonl").write_text(
        json.dumps({"event_type": "prompt_written", "phase": "setup"}) + "\n",
        encoding="utf-8",
    )

    summary = runner._run_progress_summary_for_log(run_dir)

    assert summary is not None
    assert "workflow=materials.assign" in summary
    assert "material assignments=2" in summary
    assert "decision_prims=4" in summary
    assert "missing_prims=1" in summary
    assert "patch_assignments=3" in summary
    assert "patch_reviewed_no_override=1" in summary
    assert "last_event=prompt_written/setup" in summary


def test_run_progress_summary_for_log_reports_single_physics_state(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "physics-run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    (run_dir / "request.json").write_text(
        json.dumps({"workflow": "physics.apply"}),
        encoding="utf-8",
    )
    (raw_dir / "physics_decision_patch.json").write_text(
        json.dumps({"decisions": [{}, {}]}),
        encoding="utf-8",
    )
    (run_dir / "physics_assignments.json").write_text(
        json.dumps({"decisions": [{}], "validation_status": "pass"}),
        encoding="utf-8",
    )
    (run_dir / "physics_behavior_assessment.json").write_text(
        json.dumps({"status": "pass"}),
        encoding="utf-8",
    )
    (raw_dir / "physics_visual_validation_history.json").write_text(
        json.dumps({"status": "satisfied", "iterations": [{}, {}]}),
        encoding="utf-8",
    )

    summary = runner._run_progress_summary_for_log(run_dir)

    assert summary is not None
    assert "workflow=physics.apply" in summary
    assert "physics patch_decisions=2" in summary
    assert "assignments=1" in summary
    assert "validation=pass" in summary
    assert "assessment=pass" in summary
    assert "visual_history=satisfied" in summary
    assert "visual_iterations=2" in summary


def test_run_subprocess_with_timeout_stops_child_on_fatal_workbench_watchdog(
    tmp_path: Path,
) -> None:
    log_stream = StringIO()

    def watchdog() -> WatchdogFailure:
        return WatchdogFailure("Content Workbench process exited.", fatal=True)

    with pytest.raises(RuntimeError, match="Content Workbench process exited"):
        _run_subprocess_with_timeout(
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_seconds=5,
            log_stream=log_stream,
            timeout_label="test child",
            workbench_watchdog=watchdog,
            heartbeat_interval_seconds=0,
            watchdog_interval_seconds=0.1,
        )

    output = log_stream.getvalue()
    assert "Workbench watchdog warning" in output
    assert (
        "RuntimeError: test child stopped because Content Workbench process exited."
        in output
    )


def test_child_signal_handler_terminates_child_process(monkeypatch: Any) -> None:
    process = object()
    log_stream = StringIO()
    registered: dict[int, Any] = {}
    termination_requests: list[object] = []

    def fake_signal(signum: int, handler: Any) -> Any:
        registered[signum] = handler
        return signal.SIG_DFL

    monkeypatch.setattr(runner.signal, "getsignal", lambda _signum: signal.SIG_DFL)
    monkeypatch.setattr(runner.signal, "signal", fake_signal)
    monkeypatch.setattr(
        runner,
        "_request_subprocess_termination",
        lambda child_process: termination_requests.append(child_process),
    )

    restore = _install_child_signal_handlers(
        process=process,  # type: ignore[arg-type]
        log_stream=log_stream,
        timeout_label="test child",
    )

    with pytest.raises(ChildProcessInterrupted):
        registered[signal.SIGTERM](signal.SIGTERM, None)

    restore()
    assert termination_requests == [process]
    assert "received signal" in log_stream.getvalue()


def test_structured_finalizer_flags_unreviewed_high_salience_group(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    _stub_structured_finalizer_workbench(monkeypatch)

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "Feet still need review.",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": [],
                    "issues_fixed": [],
                    "unresolved_issues": [],
                    "assessment_notes": "Incorrectly passed.",
                },
                "final_review_notes": "done",
            },
        )
    )

    visual_quality = json.loads(
        (run_dir / "visual_quality_assessment.json").read_text(encoding="utf-8")
    )
    assert visual_quality["status"] == "unresolved_issues"
    assert any(
        "visible material candidate prim(s) have no proposed material-library assignment"
        in issue
        for issue in visual_quality["unresolved_issues"]
    )
    validation_evidence = json.loads(
        (run_dir / "validation_evidence.json").read_text(encoding="utf-8")
    )
    assert validation_evidence["workflow"] == "material_assignment"
    assert validation_evidence["validation_tier"] == "T1_basic_stability"
    assert validation_evidence["sim_ready_status"] == "fail"
    assert validation_evidence["checks"][0]["name"] == "visual_materials"


def test_structured_finalizer_requires_explicit_clean_slate_assignments(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    _stub_structured_finalizer_workbench(monkeypatch)

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "foot ankle",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": ["/World/Foot"],
                        "rationale": "Reference shows black foot hardware.",
                    }
                ],
                "reviewed_no_override": [],
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Foot was too light."],
                    "issues_fixed": ["Assigned black rubber to foot."],
                    "unresolved_issues": [],
                    "assessment_notes": "Incorrectly accepted one missing mesh.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    coverage = assignments["coverage"]
    assert coverage["candidate_visible_prim_count"] == 2
    assert coverage["material_decision_prim_count"] == 1
    assert coverage["material_assignment_prim_count"] == 1
    assert coverage["missing_assignment_prim_count"] == 1
    assert coverage["rejected_assignment_prim_count"] == 0
    assert coverage["unassigned_visible_prim_count"] == 1
    assert assignments["visual_quality_assessment"]["status"] == "unresolved_issues"
    assert assignments["assignments"][-1]["coverage_status"] == (
        "missing_material_assignment"
    )


def test_structured_finalizer_distinguishes_rejected_from_missing_assignments(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    _stub_structured_finalizer_workbench(monkeypatch)

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "foot ankle",
                        "material_name": "Not In Palette",
                        "material_path": "/World/Looks/Not_In_Palette",
                        "prim_paths": ["/World/Foot"],
                        "rationale": "Reference shows black foot hardware.",
                    }
                ],
                "reviewed_no_override": [],
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": [],
                    "issues_fixed": [],
                    "unresolved_issues": [],
                    "assessment_notes": "Rejected and missing rows must be separate.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    coverage = assignments["coverage"]
    statuses = [
        group["coverage_status"]
        for group in assignments["assignments"]
        if group["coverage_status"]
        in {
            "missing_material_assignment",
            "rejected_material_assignment",
        }
    ]
    assert coverage["candidate_visible_prim_count"] == 2
    assert coverage["material_decision_prim_count"] == 0
    assert coverage["missing_assignment_prim_count"] == 1
    assert coverage["rejected_assignment_prim_count"] == 1
    assert coverage["unassigned_visible_prim_count"] == 2
    assert statuses == ["rejected_material_assignment", "missing_material_assignment"]


def test_unresolved_high_salience_families_include_current_unassigned_statuses() -> (
    None
):
    assert material_finalize._unresolved_high_salience_families(
        [
            {
                "coverage_status": "missing_material_assignment",
                "semantic_hints": {"hand_gripper": 1.0},
                "authoring_family": "hand gripper",
            },
            {
                "coverage_status": "rejected_material_assignment",
                "semantic_hints": {"logo_marking": 0.9},
                "family": "logo marking",
            },
            {
                "coverage_status": "material_assignment",
                "semantic_hints": {"torso_shell": 0.8},
                "family": "resolved torso",
            },
        ]
    ) == ["hand gripper", "logo marking"]


def test_structured_finalizer_large_group_guardrail_requires_mixed_shape_hints() -> (
    None
):
    paths = [f"/World/Part_{index}" for index in range(20)]
    group = {
        "prim_paths": paths,
        "material_tags": ["rubber", "black"],
    }

    assert material_finalize._rejection_reason(group, {}) is None
    assert (
        material_finalize._rejection_reason(
            group,
            {paths[0]: "mesh", paths[1]: "thin_panel"},
        )
        is not None
    )


def test_structured_finalizer_accepts_reviewed_no_override_group(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = _write_structured_finalizer_inputs(
        tmp_path,
        respect_existing_material_bindings=True,
    )
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "foot ankle",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": ["/World/Foot"],
                        "rationale": "Reference shows black foot hardware.",
                    }
                ],
                "reviewed_no_override": [
                    {
                        "family": "torso shell",
                        "prim_paths": ["/World/Torso"],
                        "rationale": (
                            "Current light torso shell matches the reference body panel."
                        ),
                    }
                ],
                "preserved_existing_rationale": (
                    "Torso shell was reviewed visually and needs no command."
                ),
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Feet were too light."],
                    "issues_fixed": ["Assigned black rubber to feet."],
                    "unresolved_issues": [],
                    "assessment_notes": "High-salience families were reviewed.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["coverage"]["material_assignment_prim_count"] == 1
    assert assignments["coverage"]["preserved_existing_prim_count"] == 1
    assert assignments["coverage"]["ambiguous_unassigned_prim_count"] == 0
    assert assignments["visual_quality_assessment"]["status"] == "fixed"
    assert assignments["assignments"][1]["coverage_status"] == "preserved_existing"
    assert assignments["assignments"][1]["prim_paths"] == ["/World/Torso"]
    assert (
        assignments["assignments"][0]["material_description"]
        == "Dark gray rubber with non-reflective matte finish"
    )
    assert assignments["assignments"][0]["material_manifest_semantics"][
        "substances"
    ] == ["rubber"]
    assert len(posted_commands) == 1
    assert posted_commands[0]["payload"]["prim_path"] == "/World/Foot"


def test_structured_finalizer_uses_runtime_paths_for_optimized_candidates(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    runtime_path = "/World/BoltPrototype/Geometry"
    source_paths = ["/World/BoltA", "/World/BoltB"]
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "candidate_visible_prim_count": 1,
                "candidates": [
                    {
                        "runtime_path": runtime_path,
                        "runtime_paths": [runtime_path],
                        "inspection_path": runtime_path,
                        "inspection_paths": [runtime_path],
                        "source_path": source_paths[0],
                        "source_paths": source_paths,
                        "shape_hint": "blocky",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "inspection_usd": str(tmp_path / "optimized.usd"),
                "coverage": {"candidate_visible_prim_count": 1},
                "assignments": [
                    {
                        "family": "Seed: repeated bolts",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [runtime_path],
                        "source_prim_paths": source_paths,
                        "prim_paths": [runtime_path],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "repeated bolts",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "runtime_prim_paths": [runtime_path],
                        "source_prim_paths": source_paths,
                        "prim_paths": [runtime_path],
                        "rationale": "One deduplicated runtime mesh controls both bolts.",
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Bolts were too light."],
                    "issues_fixed": ["Assigned one runtime bolt prototype."],
                    "unresolved_issues": [],
                    "assessment_notes": "Deduplicated runtime path was used.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert len(posted_commands) == 1
    assert posted_commands[0]["payload"]["prim_path"] == runtime_path
    assert posted_commands[0]["payload"]["space"] == "inspection"
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["path_space"] == "inspection"
    assert assignments["inspection_usd"] == str(tmp_path / "optimized.usd")
    assert assignments["coverage"]["candidate_visible_prim_count"] == 1
    assert assignments["coverage"]["material_assignment_prim_count"] == 1
    assert assignments["assignments"][0]["prim_paths"] == [runtime_path]
    assert assignments["assignments"][0]["runtime_prim_paths"] == [runtime_path]
    assert assignments["assignments"][0]["source_prim_paths"] == source_paths


def test_structured_finalizer_translates_optimized_candidate_alias_targets(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    prototype_source_path = "/World/Prototypes/_prototype_hash_0/Mesh_0"
    prototype_runtime_path = "/World/Geometry/_prototype_hash_0/Mesh_0"
    split_source_path = "/P5M001028269_001/P5M001028269_001/Mesh_25_part"
    split_original_source_path = "/P5M001028269_001/P5M001028269_001/Mesh_25_part_0"
    split_runtime_path = "/P5M001028269_001/P5M001028269_001/Mesh_25_part"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "candidate_visible_prim_count": 2,
                "candidates": [
                    {
                        "runtime_path": prototype_runtime_path,
                        "runtime_paths": [prototype_runtime_path],
                        "inspection_path": prototype_runtime_path,
                        "inspection_paths": [prototype_runtime_path],
                        "source_path": prototype_source_path,
                        "source_paths": [prototype_source_path],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_path": split_runtime_path,
                        "runtime_paths": [split_runtime_path],
                        "inspection_path": split_runtime_path,
                        "inspection_paths": [split_runtime_path],
                        "source_path": split_source_path,
                        "source_paths": [split_source_path],
                        "original_source_paths": [split_original_source_path],
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "inspection_usd": str(tmp_path / "optimized.usd"),
                "coverage": {"candidate_visible_prim_count": 2},
                "assignments": [
                    {
                        "family": "Seed: prototype mesh",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [prototype_runtime_path],
                        "source_prim_paths": [prototype_source_path],
                        "prim_paths": [prototype_runtime_path],
                    },
                    {
                        "family": "Seed: split mesh",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [split_runtime_path],
                        "source_prim_paths": [
                            split_source_path,
                            split_original_source_path,
                        ],
                        "prim_paths": [split_runtime_path],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "prototype mesh",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "runtime_prim_paths": [prototype_source_path],
                        "prim_paths": [prototype_source_path],
                        "rationale": "Patch used a prototype/source alias.",
                    },
                    {
                        "family": "split mesh",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [split_original_source_path],
                        "rationale": "Patch used the original pre-split source alias.",
                    },
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Source aliases needed live runtime targets."],
                    "issues_fixed": ["Applied overrides through translated paths."],
                    "unresolved_issues": [],
                    "assessment_notes": "Aliases were translated to runtime targets.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert [command["payload"]["prim_path"] for command in posted_commands] == [
        prototype_runtime_path,
        split_runtime_path,
    ]
    assert {command["payload"]["space"] for command in posted_commands} == {
        "inspection"
    }
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["path_space"] == "inspection"
    assert assignments["coverage"]["material_assignment_prim_count"] == 2
    assert assignments["assignments"][0]["prim_paths"] == [prototype_runtime_path]
    assert assignments["assignments"][1]["prim_paths"] == [split_runtime_path]
    assert assignments["assignments"][0]["source_prim_paths"] == [prototype_source_path]
    assert assignments["assignments"][1]["source_prim_paths"] == [
        split_source_path,
        split_original_source_path,
    ]


def test_structured_finalizer_prefers_exact_target_over_shared_alias(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    exact_runtime_target = "/World/Original/Mesh_A"
    other_runtime_target = "/World/Optimized/Mesh_B"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "candidate_visible_prim_count": 2,
                "candidates": [
                    {
                        "runtime_path": exact_runtime_target,
                        "runtime_paths": [exact_runtime_target],
                        "inspection_path": exact_runtime_target,
                        "inspection_paths": [exact_runtime_target],
                        "source_path": "/World/Source/Mesh_A",
                        "source_paths": ["/World/Source/Mesh_A"],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_path": other_runtime_target,
                        "runtime_paths": [other_runtime_target],
                        "inspection_path": other_runtime_target,
                        "inspection_paths": [other_runtime_target],
                        "source_path": exact_runtime_target,
                        "source_paths": [exact_runtime_target],
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "inspection_usd": str(tmp_path / "optimized.usd"),
                "coverage": {"candidate_visible_prim_count": 2},
                "assignments": [
                    {
                        "family": "Seed: exact target",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [exact_runtime_target],
                        "source_prim_paths": ["/World/Source/Mesh_A"],
                        "prim_paths": [exact_runtime_target],
                    },
                    {
                        "family": "Seed: alias collision",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [other_runtime_target],
                        "source_prim_paths": [exact_runtime_target],
                        "prim_paths": [other_runtime_target],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "exact target",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "runtime_prim_paths": [exact_runtime_target],
                        "rationale": (
                            "The exact active target should not expand through "
                            "another candidate's source alias."
                        ),
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Exact target collided with source evidence."],
                    "issues_fixed": ["Applied only the exact runtime target."],
                    "unresolved_issues": [],
                    "assessment_notes": "Alias collision did not expand coverage.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert len(posted_commands) == 1
    assert posted_commands[0]["payload"]["prim_path"] == exact_runtime_target
    assert posted_commands[0]["payload"]["space"] == "inspection"
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["path_space"] == "inspection"
    assert assignments["assignments"][0]["prim_paths"] == [exact_runtime_target]
    assert assignments["coverage"]["material_assignment_prim_count"] == 1


def test_structured_finalizer_fans_out_flattened_source_targets(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    canonical_path = "/Flattened_Prototype_48/left_hip_pitch_link/mesh"
    original_path = (
        "/g1_29dof_with_hand_rev_1_0/left_hip_pitch_link/visuals/"
        "left_hip_pitch_link/mesh"
    )
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidate_visible_prim_count": 1,
                "candidates": [
                    {
                        "runtime_path": original_path,
                        "runtime_paths": [original_path],
                        "inspection_path": original_path,
                        "inspection_paths": [original_path],
                        "runtime_space": "inspection",
                        "source_path": canonical_path,
                        "source_paths": [canonical_path],
                        "original_source_paths": [original_path],
                        "shape_hint": "mesh",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "coverage": {"candidate_visible_prim_count": 1},
                "assignments": [
                    {
                        "family": "Seed: hip shell",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [original_path],
                        "source_prim_paths": [canonical_path],
                        "prim_paths": [canonical_path],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "hip shell",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [canonical_path],
                        "rationale": "Use the canonical source target for coverage.",
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": [],
                    "issues_fixed": [],
                    "unresolved_issues": [],
                    "assessment_notes": "Flattened target was applied through source fan-out.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert len(posted_commands) == 1
    assert posted_commands[0]["payload"]["prim_path"] == original_path
    assert posted_commands[0]["payload"]["space"] == "inspection"
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["path_space"] == "source"
    assert assignments["assignments"][0]["prim_paths"] == [canonical_path]
    assert assignments["assignments"][0]["source_prim_paths"] == [canonical_path]
    assert assignments["coverage"]["material_assignment_prim_count"] == 1


def test_structured_finalizer_fans_out_single_runtime_source_aliases(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    prototype_source_path = "/World/Prototypes/_prototype_hash_0/Mesh_31"
    prototype_runtime_path = "/World/Geometry/_prototype_hash_0/Mesh_31"
    split_source_path = "/Flattened_Prototype_1/Mesh_25_part_2"
    split_original_source_path = (
        "/P5M001028269_001/P5M001028269_001/Mesh_25_part_2_stale"
    )
    split_runtime_path = "/P5M001028269_001/P5M001028269_001/Mesh_25_part_2"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidate_visible_prim_count": 2,
                "candidates": [
                    {
                        "runtime_path": prototype_runtime_path,
                        "runtime_paths": [prototype_runtime_path],
                        "inspection_paths": [prototype_runtime_path],
                        "runtime_space": "inspection",
                        "source_path": prototype_source_path,
                        "source_paths": [prototype_source_path],
                        "original_source_paths": [prototype_runtime_path],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_path": split_runtime_path,
                        "runtime_paths": [split_runtime_path],
                        "inspection_paths": [split_runtime_path],
                        "runtime_space": "inspection",
                        "source_path": split_source_path,
                        "source_paths": [split_source_path],
                        "original_source_paths": [split_original_source_path],
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "coverage": {"candidate_visible_prim_count": 2},
                "assignments": [
                    {
                        "family": "Seed: prototype mesh",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [prototype_runtime_path],
                        "source_prim_paths": [prototype_source_path],
                        "prim_paths": [prototype_source_path],
                    },
                    {
                        "family": "Seed: split mesh",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [split_runtime_path],
                        "source_prim_paths": [split_source_path],
                        "prim_paths": [split_source_path],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "prototype mesh",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [prototype_source_path],
                        "rationale": "Patch used canonical prototype source path.",
                    },
                    {
                        "family": "split mesh",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [split_source_path],
                        "rationale": "Patch used flattened split source alias.",
                    },
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Source aliases needed resolvable commands."],
                    "issues_fixed": ["Applied original source targets."],
                    "unresolved_issues": [],
                    "assessment_notes": "Runtime aliases were used for command fan-out.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert [command["payload"]["prim_path"] for command in posted_commands] == [
        prototype_runtime_path,
        split_runtime_path,
    ]
    assert {command["payload"]["space"] for command in posted_commands} == {
        "inspection"
    }
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["path_space"] == "source"
    assert assignments["coverage"]["material_assignment_prim_count"] == 2
    assert assignments["assignments"][0]["prim_paths"] == [prototype_source_path]
    assert assignments["assignments"][1]["prim_paths"] == [split_source_path]


def test_structured_finalizer_keeps_multi_runtime_flattened_source_targets_on_source(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    source_path = "/Flattened_Prototype_1/Shelf/Mesh_part_1"
    runtime_paths = [
        "/World/Asset/Shelf_A/Mesh_part_1/Geometry",
        "/World/Asset/Shelf_B/Mesh_part_1/Geometry",
    ]
    original_source_path = "/World/Asset/Shelf_A/Mesh_part_1"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidate_visible_prim_count": 1,
                "candidates": [
                    {
                        "runtime_path": runtime_paths[0],
                        "runtime_paths": runtime_paths,
                        "inspection_paths": runtime_paths,
                        "runtime_space": "inspection",
                        "source_path": source_path,
                        "source_paths": [source_path],
                        "original_source_paths": [original_source_path],
                        "translation_ambiguous": True,
                        "shape_hint": "thin_panel",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "coverage": {"candidate_visible_prim_count": 1},
                "assignments": [
                    {
                        "family": "Seed: instanced shelf",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": runtime_paths,
                        "source_prim_paths": [source_path],
                        "prim_paths": [source_path],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "instanced shelf",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [source_path],
                        "rationale": (
                            "Multi-runtime flattened source target must stay atomic."
                        ),
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": [],
                    "issues_fixed": [],
                    "unresolved_issues": [],
                    "assessment_notes": "Multi-runtime flattened target was applied once through source space.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert len(posted_commands) == 1
    assert posted_commands[0]["payload"]["prim_path"] == original_source_path
    assert posted_commands[0]["payload"]["space"] == "source"
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["path_space"] == "source"
    assert assignments["coverage"]["material_assignment_prim_count"] == 1
    assert assignments["assignments"][0]["prim_paths"] == [source_path]


def test_structured_finalizer_keeps_collapsed_source_candidates_on_source_target(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    prototype_source_path = "/World/Prototypes/_prototype_hash_0/Mesh_0"
    instance_runtime_paths = [
        "/World/Instances/instance_0/Mesh_0",
        "/World/Instances/instance_1/Mesh_0",
    ]
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidate_visible_prim_count": 1,
                "candidates": [
                    {
                        "runtime_paths": instance_runtime_paths,
                        "runtime_space": "source",
                        "source_path": prototype_source_path,
                        "source_paths": [prototype_source_path],
                        "original_source_paths": [instance_runtime_paths[0]],
                        "shape_hint": "mesh",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "coverage": {"candidate_visible_prim_count": 1},
                "assignments": [
                    {
                        "family": "Seed: collapsed prototype mesh",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "source",
                        "runtime_prim_paths": instance_runtime_paths,
                        "source_prim_paths": [prototype_source_path],
                        "prim_paths": [prototype_source_path],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "collapsed prototype mesh",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [prototype_source_path],
                        "rationale": (
                            "One source/prototype assignment should cover all "
                            "instance proxies."
                        ),
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Collapsed prototype needed one source bind."],
                    "issues_fixed": ["Applied one source-space command."],
                    "unresolved_issues": [],
                    "assessment_notes": "Runtime instance aliases were not expanded.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert len(posted_commands) == 1
    assert posted_commands[0]["payload"]["prim_path"] == prototype_source_path
    assert posted_commands[0]["payload"]["space"] == "source"
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["path_space"] == "source"
    assert assignments["assignments"][0]["prim_paths"] == [prototype_source_path]
    assert assignments["assignments"][0]["source_prim_paths"] == [prototype_source_path]
    assert assignments["coverage"]["material_assignment_prim_count"] == 1


def test_structured_finalizer_fans_out_optimized_collapsed_instances(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    prototype_source_path = "/World/Prototypes/_prototype_hash_0/Mesh_0"
    instance_runtime_paths = [
        "/World/Instances/instance_0/Mesh_0",
        "/World/Instances/instance_1/Mesh_0",
    ]
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidate_visible_prim_count": 1,
                "candidates": [
                    {
                        "runtime_paths": instance_runtime_paths,
                        "inspection_paths": instance_runtime_paths,
                        "runtime_space": "inspection",
                        "source_path": prototype_source_path,
                        "source_paths": [prototype_source_path],
                        "original_source_paths": instance_runtime_paths,
                        "instance_collapsed": True,
                        "shape_hint": "mesh",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "coverage": {"candidate_visible_prim_count": 1},
                "assignments": [
                    {
                        "family": "Seed: collapsed prototype mesh",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": instance_runtime_paths,
                        "source_prim_paths": [prototype_source_path],
                        "prim_paths": [prototype_source_path],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "collapsed prototype mesh",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [prototype_source_path],
                        "rationale": "Apply the prototype material to each live instance.",
                    }
                ],
                "reviewed_no_override": [],
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": [],
                    "issues_fixed": [],
                    "unresolved_issues": [],
                    "assessment_notes": "Collapsed instances were applied.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert [command["payload"]["prim_path"] for command in posted_commands] == (
        instance_runtime_paths
    )
    assert all(
        command["payload"]["space"] == "inspection" for command in posted_commands
    )
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["assignments"][0]["prim_paths"] == [prototype_source_path]
    assert assignments["coverage"]["material_assignment_prim_count"] == 1


def test_structured_finalizer_keeps_multi_runtime_flattened_source_candidate_atomic(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    canonical_source_path = "/Flattened_Prototype_1/Mesh_25_part"
    original_source_path = "/World/Source/Mesh_25_part"
    other_original_source_path = "/World/Source/Mesh_25_part_duplicate"
    split_runtime_paths = [
        "/World/Optimized/Mesh_25_part_1",
        "/World/Optimized/Mesh_25_part_2",
    ]
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidate_visible_prim_count": 1,
                "candidates": [
                    {
                        "runtime_paths": split_runtime_paths,
                        "runtime_space": "inspection",
                        "source_path": canonical_source_path,
                        "source_paths": [canonical_source_path],
                        "original_source_paths": [
                            original_source_path,
                            other_original_source_path,
                        ],
                        "shape_hint": "mesh",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "coverage": {"candidate_visible_prim_count": 1},
                "assignments": [
                    {
                        "family": "Seed: optimized split mesh",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": split_runtime_paths,
                        "source_prim_paths": [canonical_source_path],
                        "prim_paths": [canonical_source_path],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "optimized split mesh",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [canonical_source_path],
                        "rationale": (
                            "Flattened source bind must atomically cover every "
                            "optimized runtime fragment."
                        ),
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Split mesh fragments need an atomic bind."],
                    "issues_fixed": ["Applied the canonical logical source target."],
                    "unresolved_issues": [],
                    "assessment_notes": "Runtime fragments were not applied separately.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert len(posted_commands) == 1
    assert posted_commands[0]["payload"]["prim_path"] == canonical_source_path
    assert posted_commands[0]["payload"]["space"] == "source"
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["path_space"] == "source"
    assert assignments["assignments"][0]["prim_paths"] == [canonical_source_path]
    assert assignments["assignments"][0]["source_prim_paths"] == [canonical_source_path]
    assert assignments["coverage"]["material_assignment_prim_count"] == 1


def test_structured_finalizer_keeps_shared_runtime_aliases_on_source_targets(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    shared_runtime_path = "/World/Optimized/DedupedMesh"
    source_paths = [
        "/World/BoltA",
        "/World/BoltB",
    ]
    source_alias_paths = [
        "/World/Source/Mesh_A",
        "/World/Source/Mesh_B",
    ]
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidate_visible_prim_count": 2,
                "candidates": [
                    {
                        "runtime_paths": [shared_runtime_path],
                        "runtime_space": "inspection",
                        "source_path": source_paths[0],
                        "source_paths": [source_paths[0]],
                        "original_source_paths": [source_alias_paths[0]],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_paths": [shared_runtime_path],
                        "runtime_space": "inspection",
                        "source_path": source_paths[1],
                        "source_paths": [source_paths[1]],
                        "original_source_paths": [source_alias_paths[1]],
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "coverage": {"candidate_visible_prim_count": 2},
                "assignments": [
                    {
                        "family": "Seed: mesh A",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [shared_runtime_path],
                        "source_prim_paths": [source_paths[0]],
                        "prim_paths": [source_paths[0]],
                    },
                    {
                        "family": "Seed: mesh B",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [shared_runtime_path],
                        "source_prim_paths": [source_paths[1]],
                        "prim_paths": [source_paths[1]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "mesh A",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [source_paths[0]],
                        "rationale": "Keep source A separate from source B.",
                    },
                    {
                        "family": "mesh B",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [source_paths[1]],
                        "rationale": "Keep source B separate from source A.",
                    },
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Shared runtime alias needed source binds."],
                    "issues_fixed": ["Applied each canonical source target."],
                    "unresolved_issues": [],
                    "assessment_notes": "Shared runtime alias was not reused.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert len(posted_commands) == 1
    assert posted_commands[0]["payload"]["prim_path"] == shared_runtime_path
    assert posted_commands[0]["payload"]["space"] == "inspection"
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["path_space"] == "source"
    assert assignments["coverage"]["material_assignment_prim_count"] == 2
    assert assignments["assignments"][0]["prim_paths"] == source_paths
    counts = json.loads(
        (run_dir / "api_operation_counts.json").read_text(encoding="utf-8")
    )
    assert counts["material_override_commands"] == 1


def test_structured_finalizer_binds_overlapping_shared_runtime_aliases_atomically(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    source_paths = ["/World/A", "/World/B", "/World/C"]
    runtime_a = "/World/Optimized/R1"
    runtime_b = "/World/Optimized/R2"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidate_visible_prim_count": 3,
                "candidates": [
                    {
                        "runtime_paths": [runtime_a],
                        "runtime_space": "inspection",
                        "source_path": source_paths[0],
                        "source_paths": [source_paths[0]],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_paths": [runtime_a, runtime_b],
                        "runtime_space": "inspection",
                        "source_path": source_paths[1],
                        "source_paths": [source_paths[1]],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_paths": [runtime_b],
                        "runtime_space": "inspection",
                        "source_path": source_paths[2],
                        "source_paths": [source_paths[2]],
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "coverage": {"candidate_visible_prim_count": 3},
                "assignments": [
                    {
                        "family": "Seed: overlapping aliases",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [runtime_a, runtime_b],
                        "source_prim_paths": source_paths,
                        "prim_paths": source_paths,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "overlapping aliases",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": source_paths,
                        "rationale": (
                            "No single runtime alias can cover the transitive "
                            "overlap atomically."
                        ),
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Overlapping runtime aliases need one material."],
                    "issues_fixed": [],
                    "unresolved_issues": [],
                    "assessment_notes": "Unsafe overlap was rejected.",
                },
                "final_review_notes": "done",
            },
        )
    )

    # A candidate that transitively touches multiple shared runtime aliases
    # (B, a member of both the R1 and R2 components) is bound atomically on
    # its own source path instead of being folded into either alias's merge;
    # the remaining single-target members of each alias (A, C) bind directly
    # to their own runtime alias. Session-level override tracking narrows
    # coverage on overlap instead of deleting it wholesale, so none of these
    # three commands can clobber another's coverage.
    assert len(posted_commands) == 3
    posted_by_prim_path = {
        command["payload"]["prim_path"]: command for command in posted_commands
    }
    assert posted_by_prim_path[runtime_a]["payload"]["space"] == "inspection"
    assert posted_by_prim_path[runtime_b]["payload"]["space"] == "inspection"
    assert posted_by_prim_path[source_paths[1]]["payload"]["space"] == "source"
    rejected = json.loads(
        (raw_dir / "rejected_material_assignments.json").read_text(encoding="utf-8")
    )
    assert rejected == []
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["coverage"]["rejected_assignment_prim_count"] == 0
    assert assignments["coverage"]["material_assignment_prim_count"] == 3


def test_shared_runtime_alias_rejection_group_dedupes_runtime_paths() -> None:
    result = material_finalize._shared_runtime_alias_rejection_group(
        {
            "family": "duplicate paths",
            "material_name": "Rubber Black Matte",
            "material_path": "/World/Looks/Rubber_Black_Matte",
            "rationale": "reject unsafe duplicate paths",
        },
        ["/World/A", "/World/A", "/World/B"],
        {},
        path_space="inspection",
        reason="unsafe",
    )

    assert result["prim_paths"] == ["/World/A", "/World/B"]
    assert result["runtime_prim_paths"] == ["/World/A", "/World/B"]
    assert result["source_prim_paths"] == ["/World/A", "/World/B"]


def test_structured_finalizer_rejects_instance_collapsed_shared_runtime_aliases(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    prototype_paths = [
        "/World/Prototypes/_prototype_hash_0/Mesh_0",
        "/World/Prototypes/_prototype_hash_1/Mesh_0",
    ]
    instance_paths = [
        "/World/Instances/instance_0/Mesh_0",
        "/World/Instances/instance_1/Mesh_0",
    ]
    shared_runtime_path = "/World/Optimized/SharedCollapsedMesh"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidate_visible_prim_count": 2,
                "candidates": [
                    {
                        "runtime_paths": [shared_runtime_path],
                        "runtime_space": "inspection",
                        "source_path": prototype_paths[0],
                        "source_paths": [prototype_paths[0]],
                        "original_source_paths": [instance_paths[0]],
                        "instance_collapsed": True,
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_paths": [shared_runtime_path],
                        "runtime_space": "inspection",
                        "source_path": prototype_paths[1],
                        "source_paths": [prototype_paths[1]],
                        "original_source_paths": [instance_paths[1]],
                        "instance_collapsed": True,
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "coverage": {"candidate_visible_prim_count": 2},
                "assignments": [
                    {
                        "family": "Seed: collapsed shared alias",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [shared_runtime_path],
                        "source_prim_paths": prototype_paths,
                        "prim_paths": prototype_paths,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "collapsed shared alias",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": prototype_paths,
                        "rationale": (
                            "The shared optimized alias may resolve to instance "
                            "proxies instead of canonical prototypes."
                        ),
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Collapsed shared alias needs review."],
                    "issues_fixed": [],
                    "unresolved_issues": [],
                    "assessment_notes": "Unsafe collapsed shared alias was rejected.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert posted_commands == []
    rejected = json.loads(
        (raw_dir / "rejected_material_assignments.json").read_text(encoding="utf-8")
    )
    assert rejected
    assert all("instance-collapsed" in item["rejection_reason"] for item in rejected)
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["coverage"]["rejected_assignment_prim_count"] == 2
    assert assignments["coverage"]["material_assignment_prim_count"] == 0


def test_structured_finalizer_binds_mixed_shared_and_unique_runtime_aliases(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    source_paths = ["/World/BoltA", "/World/BoltB"]
    shared_runtime_path = "/World/Optimized/SharedBolt"
    unique_runtime_path = "/World/Optimized/BoltA_Detail"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidate_visible_prim_count": 2,
                "candidates": [
                    {
                        "runtime_paths": [shared_runtime_path, unique_runtime_path],
                        "runtime_space": "inspection",
                        "source_path": source_paths[0],
                        "source_paths": [source_paths[0]],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_paths": [shared_runtime_path],
                        "runtime_space": "inspection",
                        "source_path": source_paths[1],
                        "source_paths": [source_paths[1]],
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "coverage": {"candidate_visible_prim_count": 2},
                "assignments": [
                    {
                        "family": "Seed: mixed shared alias",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [
                            shared_runtime_path,
                            unique_runtime_path,
                        ],
                        "source_prim_paths": source_paths,
                        "prim_paths": source_paths,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "mixed shared alias",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": source_paths,
                        "rationale": (
                            "The shared alias and unique fragment need one "
                            "atomic representation."
                        ),
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Mixed shared and unique aliases need review."],
                    "issues_fixed": [],
                    "unresolved_issues": [],
                    "assessment_notes": "Unsafe mixed alias was rejected.",
                },
                "final_review_notes": "done",
            },
        )
    )

    # BoltA has an extra runtime fragment beyond the alias it shares with
    # BoltB, so it is bound atomically on its own source path instead of
    # being folded into the shared alias's merge; BoltB binds directly to the
    # (now BoltA-free) shared alias. Session-level override tracking narrows
    # the shared alias's coverage to BoltB instead of deleting it wholesale
    # when BoltA's command is applied, so both prims end up covered.
    assert len(posted_commands) == 2
    posted_by_prim_path = {
        command["payload"]["prim_path"]: command for command in posted_commands
    }
    assert posted_by_prim_path[source_paths[0]]["payload"]["space"] == "source"
    assert posted_by_prim_path[shared_runtime_path]["payload"]["space"] == "inspection"
    rejected = json.loads(
        (raw_dir / "rejected_material_assignments.json").read_text(encoding="utf-8")
    )
    assert rejected == []
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["coverage"]["rejected_assignment_prim_count"] == 0
    assert assignments["coverage"]["material_assignment_prim_count"] == 2


def test_structured_finalizer_rejects_shared_alias_with_undecided_sibling(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """An undecided extra-edge sibling must block the shared alias, not be ignored.

    Regression test: BoltA maps to [shared, unique] (an "extra edge") while
    BoltB maps only to [shared]. If only BoltB gets a material decision and
    BoltA is left completely undecided (no material_assignments entry at
    all), BoltB's command still binds the same "shared" runtime prim that
    represents part of BoltA's geometry too. Applying it would silently
    change BoltA's rendered appearance in final renders even though coverage
    reports BoltA as unassigned and never ran any safety check for it. The
    whole component must be rejected instead of letting BoltB proceed alone.
    """
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    source_paths = ["/World/BoltA", "/World/BoltB"]
    shared_runtime_path = "/World/Optimized/SharedBolt"
    unique_runtime_path = "/World/Optimized/BoltA_Detail"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidate_visible_prim_count": 2,
                "candidates": [
                    {
                        "runtime_paths": [shared_runtime_path, unique_runtime_path],
                        "runtime_space": "inspection",
                        "source_path": source_paths[0],
                        "source_paths": [source_paths[0]],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_paths": [shared_runtime_path],
                        "runtime_space": "inspection",
                        "source_path": source_paths[1],
                        "source_paths": [source_paths[1]],
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "coverage": {"candidate_visible_prim_count": 2},
                "assignments": [
                    {
                        "family": "Seed: mixed shared alias",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [
                            shared_runtime_path,
                            unique_runtime_path,
                        ],
                        "source_prim_paths": source_paths,
                        "prim_paths": source_paths,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "bolt b only",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        # Only BoltB is decided; BoltA is left out entirely.
                        "prim_paths": [source_paths[1]],
                        "rationale": "BoltB alone gets a material decision.",
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "BoltA needs more evidence.",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": [],
                    "issues_fixed": [],
                    "unresolved_issues": [],
                    "assessment_notes": "BoltB was rejected pending BoltA.",
                },
                "final_review_notes": "done",
            },
        )
    )

    # BoltB's command must never be posted: it shares the runtime alias with
    # completely-undecided BoltA, so applying it would silently change
    # BoltA's appearance without any coverage or safety check.
    assert posted_commands == []
    rejected = json.loads(
        (raw_dir / "rejected_material_assignments.json").read_text(encoding="utf-8")
    )
    rejected_paths = {
        path for group in rejected for path in group.get("prim_paths", [])
    }
    assert rejected_paths == {source_paths[1]}
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["coverage"]["material_assignment_prim_count"] == 0


def test_structured_finalizer_rejects_conflicting_extra_edge_alias_members(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A material conflict must be caught even when the mergeable set is small.

    Regression test: BoltA maps to [shared, unique] runtime paths (an "extra
    edge") while BoltB maps only to [shared]. BoltA is excluded from the
    shared alias's coalesce merge and bound atomically on its own, leaving
    only BoltB mergeable -- too few members to run the old merge-time
    conflict check. But BoltA's own command still resolves onto the same
    shared runtime prim, so assigning it a different material than BoltB
    must still be rejected instead of silently applied.
    """
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    source_paths = ["/World/BoltA", "/World/BoltB"]
    shared_runtime_path = "/World/Optimized/SharedBolt"
    unique_runtime_path = "/World/Optimized/BoltA_Detail"
    # Both materials referenced by the decision_patch below must be in the
    # palette, or the second one is rejected earlier for an unrelated reason
    # (unknown material) before it ever reaches the shared-alias conflict
    # check this test targets.
    (raw_dir / "material_palette.json").write_text(
        json.dumps(
            {
                "materials": [
                    {
                        "name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "tags": ["rubber", "black"],
                        "description": "Dark rubber",
                        "manifest_semantics": {},
                    },
                    {
                        "name": "Steel Carbon",
                        "material_path": "/World/Looks/Steel_Carbon",
                        "tags": ["steel"],
                        "description": "Carbon steel",
                        "manifest_semantics": {},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidate_visible_prim_count": 2,
                "candidates": [
                    {
                        "runtime_paths": [shared_runtime_path, unique_runtime_path],
                        "runtime_space": "inspection",
                        "source_path": source_paths[0],
                        "source_paths": [source_paths[0]],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_paths": [shared_runtime_path],
                        "runtime_space": "inspection",
                        "source_path": source_paths[1],
                        "source_paths": [source_paths[1]],
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "coverage": {"candidate_visible_prim_count": 2},
                "assignments": [
                    {
                        "family": "Seed: mixed shared alias",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [
                            shared_runtime_path,
                            unique_runtime_path,
                        ],
                        "source_prim_paths": source_paths,
                        "prim_paths": source_paths,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "bolt a",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [source_paths[0]],
                        "rationale": "BoltA gets a different material than BoltB.",
                    },
                    {
                        "family": "bolt b",
                        "material_name": "Steel Carbon",
                        "material_path": "/World/Looks/Steel_Carbon",
                        "prim_paths": [source_paths[1]],
                        "rationale": "BoltB gets a different material than BoltA.",
                    },
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Mixed shared and unique aliases need review."],
                    "issues_fixed": [],
                    "unresolved_issues": [],
                    "assessment_notes": "Conflicting materials were rejected.",
                },
                "final_review_notes": "done",
            },
        )
    )

    # Neither command may be posted: BoltA and BoltB disagree on material
    # while sharing the same underlying runtime prim, so applying either
    # command would silently overwrite the other's coverage.
    assert posted_commands == []
    rejected = json.loads(
        (raw_dir / "rejected_material_assignments.json").read_text(encoding="utf-8")
    )
    rejected_paths = {
        path for group in rejected for path in group.get("prim_paths", [])
    }
    assert rejected_paths == set(source_paths)
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["coverage"]["material_assignment_prim_count"] == 0
    assert assignments["coverage"]["rejected_assignment_prim_count"] == 2


def test_structured_finalizer_rejects_all_candidates_covered_by_a_failed_shared_command(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A failed shared command must reject every candidate it represents.

    Regression test: when two canonical candidates map onto one shared
    runtime alias with no extra edges, they collapse into a single
    `material_override` command. If that command fails, both candidates must
    be counted as rejected -- not just the first one recorded against the
    command -- since neither actually received its material.
    """
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    source_paths = ["/World/BoltA", "/World/BoltB"]
    shared_runtime_path = "/World/Optimized/SharedBolt"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidate_visible_prim_count": 2,
                "candidates": [
                    {
                        "runtime_paths": [shared_runtime_path],
                        "runtime_space": "inspection",
                        "source_path": source_paths[0],
                        "source_paths": [source_paths[0]],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_paths": [shared_runtime_path],
                        "runtime_space": "inspection",
                        "source_path": source_paths[1],
                        "source_paths": [source_paths[1]],
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "coverage": {"candidate_visible_prim_count": 2},
                "assignments": [
                    {
                        "family": "Seed: shared alias",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [shared_runtime_path],
                        "source_prim_paths": source_paths,
                        "prim_paths": source_paths,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_post_json(_url: str, body: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(
            f"Workbench request failed: HTTP 404: Prim not found: {shared_runtime_path}"
        )

    monkeypatch.setattr(material_finalize, "_post_json", fake_post_json)
    monkeypatch.setattr(
        material_finalize,
        "_render_view",
        lambda **kwargs: {
            "name": kwargs["name"],
            "image_path": str(Path(kwargs["output_dir"]) / f"{kwargs['name']}.png"),
            "direction": kwargs["direction"],
        },
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "shared alias",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": source_paths,
                        "rationale": "Both bolts share one optimized runtime prim.",
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "pass",
                    "issues_found": [],
                    "issues_fixed": [],
                    "unresolved_issues": [],
                    "assessment_notes": "Looks correct.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["coverage"]["material_assignment_prim_count"] == 0
    assert assignments["coverage"]["rejected_assignment_prim_count"] == 2
    assigned_paths = {
        path
        for group in assignments["assignments"]
        if group.get("coverage_status") == "material_assignment"
        for path in group.get("prim_paths", [])
    }
    assert assigned_paths == set()
    rejected_placeholder_paths = {
        path
        for group in assignments["assignments"]
        if group.get("coverage_status") == "rejected_material_assignment"
        for path in group.get("prim_paths", [])
    }
    assert rejected_placeholder_paths == set(source_paths)


def test_structured_finalizer_survives_a_single_failed_override_command(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """One Workbench command failure must not sink the whole run.

    Regression test: a `material_override` command can fail for reasons
    outside content-workflow-cli's control (e.g. a Content Workbench path
    translation edge case for a specific optimized mesh). Previously this
    exception propagated out of `finalize_material_decisions` entirely,
    discarding every other group's already-successful commands and falling
    back to reporting 0% coverage. The failure must instead be scoped to the
    specific prim it affects.
    """
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    source_paths = ["/World/Good", "/World/Bad"]
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "candidate_visible_prim_count": 2,
                "candidates": [
                    {
                        "runtime_paths": [source_paths[0]],
                        "runtime_space": "source",
                        "source_path": source_paths[0],
                        "source_paths": [source_paths[0]],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_paths": [source_paths[1]],
                        "runtime_space": "source",
                        "source_path": source_paths[1],
                        "source_paths": [source_paths[1]],
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "source",
                "coverage": {"candidate_visible_prim_count": 2},
                "assignments": [
                    {
                        "family": "Seed: two prims",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "source",
                        "runtime_prim_paths": [],
                        "source_prim_paths": source_paths,
                        "prim_paths": source_paths,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_post_json(_url: str, body: dict[str, Any]) -> dict[str, Any]:
        if body["payload"]["prim_path"] == "/World/Bad":
            raise RuntimeError(
                "Workbench request failed: HTTP 404: Prim not found: "
                "/World/Bad/Geometry/Geometry"
            )
        return {"ok": True}

    monkeypatch.setattr(material_finalize, "_post_json", fake_post_json)
    monkeypatch.setattr(
        material_finalize,
        "_render_view",
        lambda **kwargs: {
            "name": kwargs["name"],
            "image_path": str(Path(kwargs["output_dir"]) / f"{kwargs['name']}.png"),
            "direction": kwargs["direction"],
        },
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "two prims",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": source_paths,
                        "rationale": "Both prims get the same material.",
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "pass",
                    "issues_found": [],
                    "issues_fixed": [],
                    "unresolved_issues": [],
                    "assessment_notes": "Looks correct.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["coverage"]["material_assignment_prim_count"] == 1
    assert assignments["coverage"]["rejected_assignment_prim_count"] == 1
    assigned_paths = {
        path
        for group in assignments["assignments"]
        if group.get("coverage_status") == "material_assignment"
        for path in group.get("prim_paths", [])
    }
    assert assigned_paths == {"/World/Good"}
    rejected_placeholder_paths = {
        path
        for group in assignments["assignments"]
        if group.get("coverage_status") == "rejected_material_assignment"
        for path in group.get("prim_paths", [])
    }
    assert rejected_placeholder_paths == {"/World/Bad"}
    rejected = json.loads(
        (raw_dir / "rejected_material_assignments.json").read_text(encoding="utf-8")
    )
    assert any("/World/Bad" in group.get("prim_paths", []) for group in rejected)


def test_structured_finalizer_prefers_runtime_targets_over_source_evidence(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    runtime_path = "/World/Optimized/shared_source_part_0"
    other_runtime_path = "/World/Optimized/shared_source_part_1"
    shared_source_path = "/World/Source/SharedMesh"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "candidate_visible_prim_count": 2,
                "candidates": [
                    {
                        "runtime_paths": [runtime_path],
                        "inspection_paths": [runtime_path],
                        "source_path": shared_source_path,
                        "source_paths": [shared_source_path],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_paths": [other_runtime_path],
                        "inspection_paths": [other_runtime_path],
                        "source_path": shared_source_path,
                        "source_paths": [shared_source_path],
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "inspection_usd": str(tmp_path / "optimized.usd"),
                "coverage": {"candidate_visible_prim_count": 2},
                "assignments": [
                    {
                        "family": "Seed: shared source part 0",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [runtime_path],
                        "source_prim_paths": [shared_source_path],
                        "prim_paths": [runtime_path],
                    },
                    {
                        "family": "Seed: shared source part 1",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [other_runtime_path],
                        "source_prim_paths": [shared_source_path],
                        "prim_paths": [other_runtime_path],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "shared source part 0",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "runtime_prim_paths": [runtime_path],
                        "source_prim_paths": [shared_source_path],
                        "rationale": (
                            "Runtime target is authoritative; source path is "
                            "companion evidence."
                        ),
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Only one optimized fragment needed material."],
                    "issues_fixed": ["Applied only the requested runtime target."],
                    "unresolved_issues": [],
                    "assessment_notes": "Source evidence did not expand the group.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert len(posted_commands) == 1
    assert posted_commands[0]["payload"]["prim_path"] == runtime_path
    assert posted_commands[0]["payload"]["space"] == "inspection"
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["path_space"] == "inspection"
    assert assignments["assignments"][0]["prim_paths"] == [runtime_path]
    assert assignments["coverage"]["material_assignment_prim_count"] == 1


def test_structured_finalizer_rejects_ambiguous_shared_alias_targets(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    runtime_paths = [
        "/World/Optimized/shared_source_part_0",
        "/World/Optimized/shared_source_part_1",
    ]
    shared_source_path = "/World/Source/SharedMesh"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "candidate_visible_prim_count": 2,
                "candidates": [
                    {
                        "runtime_paths": [runtime_paths[0]],
                        "inspection_paths": [runtime_paths[0]],
                        "source_path": shared_source_path,
                        "source_paths": [shared_source_path],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_paths": [runtime_paths[1]],
                        "inspection_paths": [runtime_paths[1]],
                        "source_path": shared_source_path,
                        "source_paths": [shared_source_path],
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "inspection_usd": str(tmp_path / "optimized.usd"),
                "coverage": {"candidate_visible_prim_count": 2},
                "assignments": [
                    {
                        "family": "Seed: shared source part 0",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [runtime_paths[0]],
                        "source_prim_paths": [shared_source_path],
                        "prim_paths": [runtime_paths[0]],
                    },
                    {
                        "family": "Seed: shared source part 1",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [runtime_paths[1]],
                        "source_prim_paths": [shared_source_path],
                        "prim_paths": [runtime_paths[1]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "ambiguous shared source",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [shared_source_path],
                        "rationale": "Shared source alias should not fan out.",
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Shared source alias was ambiguous."],
                    "issues_fixed": ["Rejected ambiguous alias assignment."],
                    "unresolved_issues": [],
                    "assessment_notes": "Ambiguous alias did not expand coverage.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert posted_commands == []
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    rejected = json.loads(
        (run_dir / "raw" / "rejected_material_assignments.json").read_text(
            encoding="utf-8"
        )
    )
    assert assignments["path_space"] == "inspection"
    assert assignments["coverage"]["material_assignment_prim_count"] == 0
    assert assignments["coverage"]["missing_assignment_prim_count"] == 2
    assert assignments["coverage"]["rejected_assignment_prim_count"] == 0
    assert rejected[0]["prim_paths"] == [shared_source_path]
    assert "ambiguous aliases" in rejected[0]["rejection_reason"]


def test_structured_finalizer_rejects_ambiguous_reviewed_alias_targets(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(
        tmp_path,
        respect_existing_material_bindings=True,
    )
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    runtime_paths = [
        "/World/Optimized/reviewed_shared_part_0",
        "/World/Optimized/reviewed_shared_part_1",
    ]
    shared_source_path = "/World/Source/ReviewedSharedMesh"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "candidate_visible_prim_count": 2,
                "candidates": [
                    {
                        "runtime_paths": [runtime_paths[0]],
                        "inspection_paths": [runtime_paths[0]],
                        "source_path": shared_source_path,
                        "source_paths": [shared_source_path],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_paths": [runtime_paths[1]],
                        "inspection_paths": [runtime_paths[1]],
                        "source_path": shared_source_path,
                        "source_paths": [shared_source_path],
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "inspection_usd": str(tmp_path / "optimized.usd"),
                "coverage": {"candidate_visible_prim_count": 2},
                "assignments": [
                    {
                        "family": "Seed: reviewed shared part 0",
                        "coverage_status": "preserved_existing",
                        "material_name": "Existing material",
                        "material_path": "/World/Looks/Existing",
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [runtime_paths[0]],
                        "source_prim_paths": [shared_source_path],
                        "prim_paths": [runtime_paths[0]],
                    },
                    {
                        "family": "Seed: reviewed shared part 1",
                        "coverage_status": "preserved_existing",
                        "material_name": "Existing material",
                        "material_path": "/World/Looks/Existing",
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [runtime_paths[1]],
                        "source_prim_paths": [shared_source_path],
                        "prim_paths": [runtime_paths[1]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [],
                "reviewed_no_override": [
                    {
                        "family": "ambiguous reviewed shared source",
                        "prim_paths": [shared_source_path],
                        "rationale": "Shared alias should not be accepted.",
                    }
                ],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Shared reviewed alias was ambiguous."],
                    "issues_fixed": ["Rejected ambiguous reviewed alias."],
                    "unresolved_issues": [],
                    "assessment_notes": "Ambiguous reviewed alias did not resolve.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert posted_commands == []
    rejected = json.loads(
        (run_dir / "raw" / "rejected_material_assignments.json").read_text(
            encoding="utf-8"
        )
    )
    assert rejected[0]["coverage_status"] == "preserved_existing"
    assert rejected[0]["prim_paths"] == [shared_source_path]
    assert "ambiguous aliases" in rejected[0]["rejection_reason"]


def test_structured_finalizer_accepts_inspection_candidate_prim_path_fallback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    fallback_target = "/World/InspectionOnly/Mesh"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "candidate_visible_prim_count": 1,
                "candidates": [
                    {
                        "prim_path": fallback_target,
                        "prim_paths": [fallback_target],
                        "shape_hint": "mesh",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "inspection_usd": str(tmp_path / "optimized.usd"),
                "coverage": {"candidate_visible_prim_count": 1},
                "assignments": [
                    {
                        "family": "Seed: prim-path-only candidate",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [fallback_target],
                        "source_prim_paths": [fallback_target],
                        "prim_paths": [fallback_target],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "prim-path-only candidate",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [fallback_target],
                        "rationale": "Candidate only exposed prim_path.",
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Candidate only exposed prim_path."],
                    "issues_fixed": ["Accepted prim_path fallback."],
                    "unresolved_issues": [],
                    "assessment_notes": "prim_path fallback resolved.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert len(posted_commands) == 1
    assert posted_commands[0]["payload"]["prim_path"] == fallback_target
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["coverage"]["material_assignment_prim_count"] == 1


def test_structured_finalizer_accepts_shared_alias_with_single_target(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    runtime_target = "/World/Optimized/shared_alias_single_target"
    shared_source_path = "/World/Source/SharedAliasSingleTarget"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "candidate_visible_prim_count": 1,
                "candidates": [
                    {
                        "runtime_paths": [runtime_target],
                        "inspection_paths": [runtime_target],
                        "source_path": shared_source_path,
                        "source_paths": [shared_source_path],
                        "shape_hint": "mesh",
                    },
                    {
                        "runtime_paths": [runtime_target],
                        "inspection_paths": [runtime_target],
                        "source_path": shared_source_path,
                        "source_paths": [shared_source_path],
                        "shape_hint": "mesh",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "inspection_usd": str(tmp_path / "optimized.usd"),
                "coverage": {"candidate_visible_prim_count": 1},
                "assignments": [
                    {
                        "family": "Seed: shared alias single target",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [runtime_target],
                        "source_prim_paths": [shared_source_path],
                        "prim_paths": [runtime_target],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "shared alias single target",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [shared_source_path],
                        "rationale": "Shared alias resolves to one target.",
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Shared alias looked ambiguous by row count."],
                    "issues_fixed": ["Resolved because target set was unique."],
                    "unresolved_issues": [],
                    "assessment_notes": "Shared alias resolved to a single target.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert len(posted_commands) == 1
    assert posted_commands[0]["payload"]["prim_path"] == runtime_target
    rejected = json.loads(
        (run_dir / "raw" / "rejected_material_assignments.json").read_text(
            encoding="utf-8"
        )
    )
    assert rejected == []


def test_structured_finalizer_rejects_duplicate_canonical_assignment_targets(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    raw_dir = run_dir / "raw"
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)
    runtime_target = "/World/Optimized/duplicate_target_mesh"
    source_alias = "/World/Source/DuplicateTargetMesh"
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "candidate_visible_prim_count": 1,
                "candidates": [
                    {
                        "runtime_paths": [runtime_target],
                        "inspection_paths": [runtime_target],
                        "source_path": source_alias,
                        "source_paths": [source_alias],
                        "shape_hint": "mesh",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "path_space": "inspection",
                "inspection_usd": str(tmp_path / "optimized.usd"),
                "coverage": {"candidate_visible_prim_count": 1},
                "assignments": [
                    {
                        "family": "Seed: duplicate target",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "runtime_space": "inspection",
                        "runtime_prim_paths": [runtime_target],
                        "source_prim_paths": [source_alias],
                        "prim_paths": [runtime_target],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "duplicate exact target",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "runtime_prim_paths": [runtime_target],
                        "rationale": "First assignment claims the target.",
                    },
                    {
                        "family": "duplicate alias target",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": [source_alias],
                        "rationale": "Alias resolves to the same target.",
                    },
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Duplicate target was proposed twice."],
                    "issues_fixed": ["Rejected the duplicate assignment."],
                    "unresolved_issues": [],
                    "assessment_notes": "Only one command should be posted.",
                },
                "final_review_notes": "done",
            },
        )
    )

    assert len(posted_commands) == 1
    assert posted_commands[0]["payload"]["prim_path"] == runtime_target
    rejected = json.loads(
        (run_dir / "raw" / "rejected_material_assignments.json").read_text(
            encoding="utf-8"
        )
    )
    assert rejected[0]["family"] == "duplicate alias target"
    assert rejected[0]["prim_paths"] == [runtime_target]
    assert "duplicate material assignment" in rejected[0]["rejection_reason"]
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["coverage"]["material_assignment_prim_count"] == 1


def test_structured_finalizer_preserves_grounding_operation_counts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(
        tmp_path,
        respect_existing_material_bindings=True,
    )
    _stub_structured_finalizer_workbench(monkeypatch)
    (run_dir / "raw" / "material_grounding_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": ("content-agents.material-grounding-diagnostics.v1"),
                "runs": [
                    {
                        "validation_iteration": 0,
                        "operation_counts": {
                            "pick_calls": 3,
                            "render_calls": 1,
                            "render_artifact_downloads": 2,
                            "workbench_api_calls": 9,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "foot ankle",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": ["/World/Foot"],
                        "rationale": "Reference shows black foot hardware.",
                    }
                ],
                "reviewed_no_override": [
                    {
                        "family": "torso shell",
                        "prim_paths": ["/World/Torso"],
                        "rationale": "Torso shell was reviewed.",
                    }
                ],
                "preserved_existing_rationale": "Torso shell was reviewed.",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Feet were too light."],
                    "issues_fixed": ["Assigned black rubber to feet."],
                    "unresolved_issues": [],
                    "assessment_notes": "High-salience families were reviewed.",
                },
                "final_review_notes": "done",
            },
        )
    )

    counts = json.loads(
        (run_dir / "api_operation_counts.json").read_text(encoding="utf-8")
    )
    assert counts["pick_calls"] == 3
    assert counts["grounding_diagnostic_runs"] == 1
    assert counts["grounding_pick_calls"] == 3
    assert counts["grounding_render_calls"] == 1
    assert counts["render_count_total"] == 29
    assert counts["final_render_calls"] == 28
    assert counts["final_renders"] == 5
    assert counts["render_artifact_downloads"] == 31
    assert "grounding diagnostics" in counts["count_basis"]


def test_structured_finalizer_merges_post_apply_visual_quality(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(
        tmp_path,
        respect_existing_material_bindings=True,
    )
    _stub_structured_finalizer_workbench(monkeypatch)

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "foot ankle",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": ["/World/Foot"],
                        "rationale": "Reference shows black foot hardware.",
                    }
                ],
                "reviewed_no_override": [
                    {
                        "family": "torso shell",
                        "prim_paths": ["/World/Torso"],
                        "rationale": "Torso shell was reviewed.",
                    }
                ],
                "preserved_existing_rationale": "Torso shell was reviewed.",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": ["Feet were too light."],
                    "issues_fixed": ["Assigned black rubber to feet."],
                    "unresolved_issues": [],
                    "assessment_notes": "Planner accepted final state.",
                },
                "final_review_notes": "done",
            },
        )
    )

    paths = material_finalize.apply_post_apply_visual_quality(
        run_dir=run_dir,
        visual_quality={
            "status": "unresolved_issues",
            "issues_found": ["Wrong visible part received a dark material."],
            "issues_fixed": [],
            "unresolved_issues": ["Final render still has the wrong target part."],
            "assessment_notes": "Final render mismatch.",
        },
        validator_artifact=run_dir / "raw" / "post_apply_visual_quality.json",
    )

    visual_quality = json.loads(
        Path(paths["visual_quality_assessment"]).read_text(encoding="utf-8")
    )
    assignments = json.loads(Path(paths["assignments"]).read_text(encoding="utf-8"))
    counts = json.loads(Path(paths["api_operation_counts"]).read_text(encoding="utf-8"))
    final_summary = Path(paths["final_summary"]).read_text(encoding="utf-8")
    assert visual_quality["status"] == "unresolved_issues"
    assert visual_quality["issues_found"] == [
        "Feet were too light.",
        "Wrong visible part received a dark material.",
    ]
    assert visual_quality["unresolved_issues"] == [
        "Final render still has the wrong target part."
    ]
    assert assignments["visual_quality_assessment"] == visual_quality
    assert assignments["final_review"]["issues_found"] == [
        "Wrong visible part received a dark material."
    ]
    assert assignments["final_review"]["unresolved_issues"] == [
        "Final render still has the wrong target part."
    ]
    assert counts["final_review_issues_found"] == 1
    assert counts["post_apply_vqa_issues_found"] == 1
    assert counts["post_apply_vqa_unresolved_issues"] == 1
    assert "unresolved_issues" in final_summary


def test_post_apply_visual_quality_can_clear_planner_self_assessment(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = _write_structured_finalizer_inputs(
        tmp_path,
        respect_existing_material_bindings=True,
    )
    _stub_structured_finalizer_workbench(monkeypatch)

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "foot ankle",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": ["/World/Foot"],
                        "rationale": "Reference shows black foot hardware.",
                    }
                ],
                "reviewed_no_override": [
                    {
                        "family": "torso shell",
                        "prim_paths": ["/World/Torso"],
                        "rationale": "Torso shell was reviewed.",
                    }
                ],
                "preserved_existing_rationale": "Torso shell was reviewed.",
                "ambiguous_unassigned_rationale": "",
                "visual_quality_assessment": {
                    "status": "unresolved_issues",
                    "issues_found": ["Planner was uncertain."],
                    "issues_fixed": [],
                    "unresolved_issues": ["Planner-only uncertainty."],
                    "assessment_notes": "Planner self-assessment was conservative.",
                },
                "final_review_notes": "done",
            },
        )
    )

    material_finalize.apply_post_apply_visual_quality(
        run_dir=run_dir,
        visual_quality={
            "status": "fixed",
            "issues_found": [],
            "issues_fixed": ["Final render was accepted."],
            "unresolved_issues": [],
            "assessment_notes": "Independent VQA accepted the final render.",
        },
        validator_artifact=run_dir / "raw" / "post_apply_visual_quality.json",
    )

    visual_quality = json.loads(
        (run_dir / "visual_quality_assessment.json").read_text(encoding="utf-8")
    )
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert visual_quality["status"] == "fixed"
    assert visual_quality["unresolved_issues"] == []
    assert assignments["final_review"]["unresolved_issues"] == []


def test_structured_finalizer_rejects_assignment_paths_outside_visible_candidates(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "hidden part",
                        "material_name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "prim_paths": ["/World/Hidden"],
                        "rationale": "Invalid hidden target.",
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "Invalid target rejected.",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": [],
                    "issues_fixed": [],
                    "unresolved_issues": [],
                    "assessment_notes": "Incorrectly passed.",
                },
                "final_review_notes": "done",
            },
        )
    )

    rejected = json.loads(
        (run_dir / "raw" / "rejected_material_assignments.json").read_text(
            encoding="utf-8"
        )
    )
    visual_quality = json.loads(
        (run_dir / "visual_quality_assessment.json").read_text(encoding="utf-8")
    )
    assert posted_commands == []
    assert rejected[0]["prim_paths"] == ["/World/Hidden"]
    assert "not visible material candidates" in rejected[0]["rejection_reason"]
    assert visual_quality["status"] == "unresolved_issues"
    assert any(
        "Rejected material decision for hidden part" in issue
        for issue in visual_quality["unresolved_issues"]
    )


def test_structured_finalizer_rejects_assignment_without_material_path(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    posted_commands = _stub_structured_finalizer_workbench(monkeypatch)

    finalize_material_decisions(
        MaterialFinalizeConfig(
            workbench_url="http://127.0.0.1:8088",
            run_dir=run_dir,
            session_id="session-1",
            source_usd=tmp_path / "asset.usd",
            materials_usd=tmp_path / "materials.usd",
            reference_images=[tmp_path / "reference.png"],
            decision_patch={
                "material_assignments": [
                    {
                        "family": "foot ankle",
                        "material_name": "Not In Palette",
                        "material_path": None,
                        "prim_paths": ["/World/Foot"],
                        "rationale": "Model chose an unavailable material.",
                    }
                ],
                "reviewed_no_override": [],
                "preserved_existing_rationale": "",
                "ambiguous_unassigned_rationale": "Missing material path.",
                "visual_quality_assessment": {
                    "status": "fixed",
                    "issues_found": [],
                    "issues_fixed": [],
                    "unresolved_issues": [],
                    "assessment_notes": "Incorrectly passed.",
                },
                "final_review_notes": "done",
            },
        )
    )

    rejected = json.loads(
        (run_dir / "raw" / "rejected_material_assignments.json").read_text(
            encoding="utf-8"
        )
    )
    visual_quality = json.loads(
        (run_dir / "visual_quality_assessment.json").read_text(encoding="utf-8")
    )
    assert posted_commands == []
    assert rejected[0]["prim_paths"] == ["/World/Foot"]
    assert "material_name not found" in rejected[0]["rejection_reason"]
    assert visual_quality["status"] == "unresolved_issues"
    assert any(
        "Rejected material decision for foot ankle" in issue
        for issue in visual_quality["unresolved_issues"]
    )


def test_structured_finalizer_requires_seed_artifact(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = _write_structured_finalizer_inputs(tmp_path)
    _stub_structured_finalizer_workbench(monkeypatch)
    (run_dir / "raw" / "material_assignment_seed.json").unlink()

    with pytest.raises(RuntimeError, match="material_assignment_seed.json"):
        finalize_material_decisions(
            MaterialFinalizeConfig(
                workbench_url="http://127.0.0.1:8088",
                run_dir=run_dir,
                session_id="session-1",
                source_usd=tmp_path / "asset.usd",
                materials_usd=tmp_path / "materials.usd",
                reference_images=[tmp_path / "reference.png"],
                decision_patch={
                    "material_assignments": [],
                    "reviewed_no_override": [],
                    "preserved_existing_rationale": "",
                    "ambiguous_unassigned_rationale": "",
                    "visual_quality_assessment": {
                        "status": "fixed",
                        "issues_found": [],
                        "issues_fixed": [],
                        "unresolved_issues": [],
                        "assessment_notes": "no-op",
                    },
                    "final_review_notes": "done",
                },
            )
        )


def test_material_run_packet_render_delegates_to_client(
    tmp_path: Path, monkeypatch: Any
) -> None:
    observed: dict[str, object] = {}

    def fake_render_view(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {
            "name": kwargs["name"],
            "image_path": str(tmp_path / "initial_top.png"),
            "camera_json_path": None,
            "response_path": str(tmp_path / "initial_top_response.json"),
            "artifact_download_count": 1,
        }

    monkeypatch.setattr(material_run_packet, "_client_render_view", fake_render_view)

    result = material_run_packet._render_view(
        workbench_url="http://127.0.0.1:8088",
        session_id="session/one",
        output_dir=tmp_path,
        name="initial_top",
        direction="+z",
        width=64,
        height=64,
        render_quality="inspection",
    )

    assert observed == {
        "workbench_url": "http://127.0.0.1:8088",
        "session_id": "session/one",
        "output_dir": tmp_path,
        "name": "initial_top",
        "direction": "+z",
        "width": 64,
        "height": 64,
        "render_quality": "inspection",
    }
    assert result["name"] == "initial_top"


def test_material_run_packet_closes_session_when_snapshot_fails(
    tmp_path: Path, monkeypatch: Any
) -> None:
    closed_sessions: list[tuple[str, str]] = []
    optimization_urls: list[str] = []
    for path in [tmp_path / "asset.usd", tmp_path / "materials.usd"]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml = tmp_path / "materials.yaml"
    materials_yaml.write_text("entries: []\n", encoding="utf-8")

    monkeypatch.setattr(
        material_run_packet,
        "_fetch_docs",
        lambda _workbench_url, _raw_dir: {},
    )
    monkeypatch.setattr(
        material_run_packet,
        "_create_session",
        lambda _config: {"session_id": "session/one"},
    )

    def fake_fetch_optional_json(url: str) -> None:
        optimization_urls.append(url)
        return None

    monkeypatch.setattr(
        material_run_packet, "_fetch_optional_json", fake_fetch_optional_json
    )

    def fail_snapshot(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(material_run_packet, "fetch_snapshot", fail_snapshot)
    monkeypatch.setattr(
        material_run_packet,
        "_delete_session",
        lambda workbench_url, session_id: closed_sessions.append(
            (workbench_url, session_id)
        ),
    )

    with pytest.raises(RuntimeError, match="snapshot failed"):
        material_run_packet.prepare_material_run_packet(
            material_run_packet.MaterialRunPacketConfig(
                workbench_url="http://127.0.0.1:8088",
                run_dir=tmp_path / "run",
                usd_path=tmp_path / "asset.usd",
                materials_yaml=materials_yaml,
                materials_usd=tmp_path / "materials.usd",
            )
        )

    assert optimization_urls == [
        "http://127.0.0.1:8088/sessions/session%2Fone/optimization"
    ]
    assert closed_sessions == [("http://127.0.0.1:8088", "session/one")]


def test_material_run_packet_create_session_passes_optimizer_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = material_run_packet.MaterialRunPacketConfig(
        workbench_url="http://127.0.0.1:8088",
        run_dir=tmp_path / "run",
        usd_path=tmp_path / "asset.usd",
        materials_yaml=tmp_path / "materials.yaml",
        materials_usd=tmp_path / "materials.usd",
        optimize=True,
        respect_existing_material_bindings=True,
        flatten_prototypes=True,
        enable_deinstance=False,
        enable_split=True,
        enable_deduplicate=False,
    )

    payloads: list[dict[str, Any]] = []

    def fake_create_session(
        workbench_url: str, payload: dict[str, Any]
    ) -> dict[str, str]:
        payloads.append(payload)
        assert workbench_url == "http://127.0.0.1:8088"
        return {"session_id": "session-1"}

    monkeypatch.setattr(
        material_run_packet,
        "_client_create_session",
        fake_create_session,
    )
    assert material_run_packet._create_session(config) == {"session_id": "session-1"}

    assert payloads == [
        {
            "scene_path": str(tmp_path / "asset.usd"),
            "optimize": True,
            "clear_materials": False,
            "width": 640,
            "height": 480,
            "flatten_prototypes": True,
            "enable_deinstance": False,
            "enable_split": True,
            "enable_deduplicate": False,
        }
    ]


def _write_structured_finalizer_inputs(
    tmp_path: Path,
    *,
    respect_existing_material_bindings: bool = False,
) -> Path:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    (run_dir / "trace").mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    for path in [
        tmp_path / "asset.usd",
        tmp_path / "materials.usd",
        tmp_path / "reference.png",
    ]:
        path.write_text("placeholder", encoding="utf-8")
    (raw_dir / "material_run_packet.json").write_text(
        json.dumps(
            {
                "session_id": "session-1",
                "respect_existing_material_bindings": respect_existing_material_bindings,
                "operation_counts_so_far": {
                    "render_calls_total": 0,
                    "workbench_api_calls_total": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_palette.json").write_text(
        json.dumps(
            {
                "materials": [
                    {
                        "name": "Rubber Black Matte",
                        "material_path": "/World/Looks/Rubber_Black_Matte",
                        "tags": ["rubber", "black"],
                        "description": (
                            "Dark gray rubber with non-reflective matte finish"
                        ),
                        "manifest_semantics": {
                            "colors": ["black", "gray"],
                            "substances": ["rubber"],
                            "finishes": ["matte"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "visible_candidate_prims.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {"source_path": "/World/Foot", "shape_hint": "mesh"},
                    {"source_path": "/World/Torso", "shape_hint": "thin_panel"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "material_assignment_seed.json").write_text(
        json.dumps(
            {
                "coverage": {"candidate_visible_prim_count": 2},
                "assignments": [
                    {
                        "family": "Seed: foot ankle",
                        "authoring_family": "foot ankle",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "prim_paths": ["/World/Foot"],
                        "semantic_hints": {"foot_ankle": 1},
                    },
                    {
                        "family": "Seed: torso shell",
                        "authoring_family": "torso shell",
                        "coverage_status": "ambiguous_unassigned",
                        "material_name": None,
                        "material_path": None,
                        "prim_paths": ["/World/Torso"],
                        "semantic_hints": {"torso_shell": 1},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _stub_structured_finalizer_workbench(monkeypatch: Any) -> list[dict[str, Any]]:
    posted_commands: list[dict[str, Any]] = []

    def fake_post_json(_url: str, body: dict[str, Any]) -> dict[str, Any]:
        posted_commands.append(body)
        return {"ok": True}

    def fake_render_view(**kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"{kwargs['name']}.png"
        image_path.write_text("png", encoding="utf-8")
        return {
            "name": kwargs["name"],
            "image_path": str(image_path),
            "direction": kwargs["direction"],
        }

    def fake_turntable_view(**kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "final_turntable.gif"
        image_path.write_text("gif", encoding="utf-8")
        return {
            "name": "final_turntable",
            "image_path": str(image_path),
            "direction": "turntable_up_axis",
            "render_call_count": 24,
            "artifact_download_count": 25,
        }

    monkeypatch.setattr(material_finalize, "_post_json", fake_post_json)
    monkeypatch.setattr(material_finalize, "_render_view", fake_render_view)
    monkeypatch.setattr(
        material_finalize, "_render_turntable_view", fake_turntable_view
    )
    return posted_commands


def test_final_turntable_gif_missing_ffmpeg_is_best_effort(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from PIL import Image

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for index, color in enumerate(("red", "blue")):
        Image.new("RGB", (8, 6), color=color).save(
            frames_dir / f"frame_{index:03d}.png"
        )
    output_path = tmp_path / "final_turntable.gif"

    monkeypatch.setattr(material_finalize.shutil, "which", lambda _name: None)

    result = material_finalize._encode_turntable_gif(
        frames_dir=frames_dir,
        output_path=output_path,
        width=512,
    )

    assert result["encoded"] is True
    assert result["encoder"] == "pillow"
    assert result["ffmpeg_skip_reason"] == "ffmpeg_not_found"
    assert result["image_path"] == str(output_path)
    assert output_path.exists()


def test_final_turntable_gif_keeps_primary_skip_reason_when_fallback_fails(
    tmp_path: Path, monkeypatch: Any
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    output_path = tmp_path / "final_turntable.gif"

    monkeypatch.setattr(material_finalize.shutil, "which", lambda _name: None)

    result = material_finalize._encode_turntable_gif(
        frames_dir=frames_dir,
        output_path=output_path,
        width=512,
    )

    assert result["encoded"] is False
    assert result["skip_reason"] == "ffmpeg_not_found"
    assert result["pillow_skip_reason"] == "pillow_no_frames"


def test_final_turntable_gif_removes_stale_partial_when_fallback_fails(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    output_path = tmp_path / "final_turntable.gif"
    output_path.write_bytes(b"partial gif")

    def fake_run(*_args: Any, **_kwargs: Any) -> None:
        raise material_finalize.subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="ffmpeg failed",
        )

    monkeypatch.setattr(material_finalize.shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(material_finalize.subprocess, "run", fake_run)

    result = material_finalize._encode_turntable_gif(
        frames_dir=frames_dir,
        output_path=output_path,
        width=512,
    )

    assert result["encoded"] is False
    assert result["skip_reason"] == "ffmpeg_failed"
    assert result["pillow_skip_reason"] == "pillow_no_frames"
    assert not output_path.exists()


def test_final_turntable_batched_render_counts_one_api_call(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fake_download(_url: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    monkeypatch.setattr(
        material_finalize,
        "_post_json",
        lambda *_args, **_kwargs: {
            "frame_urls": ["/renders/frame_000.png", "/renders/frame_001.png"],
            "camera_json_urls": [],
            "elapsed_seconds": 1.25,
        },
    )
    monkeypatch.setattr(material_finalize, "_download_to_file", fake_download)
    monkeypatch.setattr(
        material_finalize,
        "_encode_turntable_gif",
        lambda **_kwargs: {"encoded": False, "skip_reason": "ffmpeg_not_found"},
    )

    result = material_finalize._render_turntable_view(
        workbench_url="http://127.0.0.1:8088",
        session_id="session-1",
        output_dir=tmp_path,
        width=64,
        height=64,
        render_quality="final",
    )

    assert result["frame_count"] == 2
    assert result["render_call_count"] == 1
    assert result["artifact_download_count"] == 2


def test_final_turntable_records_rescued_ffmpeg_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fake_download(_url: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    monkeypatch.setattr(
        material_finalize,
        "_post_json",
        lambda *_args, **_kwargs: {
            "frame_urls": ["/renders/frame_000.png"],
            "camera_json_urls": [],
        },
    )
    monkeypatch.setattr(material_finalize, "_download_to_file", fake_download)
    monkeypatch.setattr(
        material_finalize,
        "_encode_turntable_gif",
        lambda **kwargs: {
            "encoded": True,
            "encoder": "pillow",
            "image_path": str(kwargs["output_path"]),
            "ffmpeg_skip_reason": "ffmpeg_failed",
            "ffmpeg_error": "ffmpeg exited with status 1",
        },
    )

    result = material_finalize._render_turntable_view(
        workbench_url="http://127.0.0.1:8088",
        session_id="session-1",
        output_dir=tmp_path,
        width=64,
        height=64,
        render_quality="final",
    )

    assert result["gif_encoded"] is True
    assert result["gif_skip_reason"] is None
    assert result["gif_error"] is None
    assert result["gif_fallback_reason"] == "ffmpeg_failed"
    assert result["gif_fallback_error"] == "ffmpeg exited with status 1"


def test_final_turntable_extra_batched_frames_fall_back_to_sequential(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    directions = material_finalize._turntable_directions()

    monkeypatch.setattr(
        material_finalize,
        "_post_json",
        lambda *_args, **_kwargs: {
            "frame_urls": [
                f"/renders/frame_{index:03d}.png"
                for index in range(len(directions) + 1)
            ],
            "camera_json_urls": [],
        },
    )

    def fake_render_view(**kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"{kwargs['name']}.png"
        image_path.write_bytes(b"png")
        return {
            "name": kwargs["name"],
            "direction": kwargs["direction"],
            "image_path": str(image_path),
            "artifact_download_count": 1,
        }

    monkeypatch.setattr(material_finalize, "_render_view", fake_render_view)
    monkeypatch.setattr(
        material_finalize,
        "_encode_turntable_gif",
        lambda **_kwargs: {"encoded": False, "skip_reason": "ffmpeg_not_found"},
    )

    result = material_finalize._render_turntable_view(
        workbench_url="http://127.0.0.1:8088",
        session_id="session-1",
        output_dir=tmp_path,
        width=64,
        height=64,
        render_quality="final",
    )

    assert result["frame_count"] == len(directions)
    assert result["render_call_count"] == len(directions)
    assert "more frames than requested" in result["batched_render_error"]
    assert [record["direction"] for record in result["frame_records"]] == directions


def test_step_artifact_snapshot_preserves_overwritten_final_renders(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    final_dir = run_dir / "final_renders"
    evidence_dir = run_dir / "evidence_renders"
    for path in (raw_dir, final_dir, evidence_dir, run_dir / "trace"):
        path.mkdir(parents=True)

    prompt = run_dir / "agent_prompt.md"
    child_output = run_dir / "child-output.log"
    child_final = run_dir / "child-final.md"
    for path, text in [
        (prompt, "initial prompt"),
        (child_output, "initial output"),
        (child_final, "initial final"),
        (
            run_dir / "request.json",
            json.dumps(
                {
                    "inputs": {
                        "usd": "/assets/ladder.usd",
                        "reference_images": ["/refs/ladder.png"],
                        "reference_files": ["/refs/ladder.pdf"],
                        "materials_yaml": "/materials/materials.yaml",
                        "materials_usd": "/materials/materials.usd",
                    },
                    "clear_materials": True,
                    "respect_existing_material_bindings": False,
                    "workbench_optimize": True,
                }
            ),
        ),
        (
            raw_dir / "material_run_packet.json",
            json.dumps(
                {
                    "session_id": "session-1",
                    "source_usd": "/assets/ladder.usd",
                    "materials_yaml": "/materials/materials.yaml",
                    "materials_usd": "/materials/materials.usd",
                    "clear_materials": True,
                    "respect_existing_material_bindings": False,
                    "optimize": True,
                    "docs": {"openapi_json": "raw/openapi.json"},
                    "session": {"optimization": "raw/optimization.json"},
                }
            ),
        ),
        (
            raw_dir / "material_palette.json",
            json.dumps(
                {
                    "materials": [
                        {
                            "name": "Brushed Aluminum",
                            "material_path": "/World/Looks/Brushed_Aluminum",
                        }
                    ]
                }
            ),
        ),
        (
            raw_dir / "visible_candidate_prims.json",
            json.dumps(
                {
                    "path_space": "inspection",
                    "candidates": [
                        {
                            "runtime_path": "/World/initial",
                            "source_path": "/World/initial_source",
                            "source_paths": ["/World/initial_source"],
                        },
                        {
                            "runtime_path": "/World/initial_detail",
                            "source_path": "/World/initial_detail_source",
                            "source_paths": ["/World/initial_detail_source"],
                        },
                    ],
                }
            ),
        ),
        (raw_dir / "codex_request.json", "{}"),
        (raw_dir / "codex_items.json", "[]"),
        (raw_dir / "codex_result.json", "{}"),
        (evidence_dir / "initial_oblique.png", "input render"),
        (evidence_dir / "child_iter1_oblique.png", "child preview"),
    ]:
        path.write_text(text, encoding="utf-8")

    def write_canonical(label: str) -> None:
        (run_dir / "assignments.json").write_text(
            json.dumps(
                {
                    "coverage": {
                        "candidate_visible_prim_count": 2,
                        "material_decision_prim_count": 2,
                        "material_assignment_prim_count": 2,
                        "ambiguous_unassigned_prim_count": 0,
                    },
                    "assignments": [
                        {
                            "coverage_status": "material_assignment",
                            "family": f"{label} rails",
                            "material_name": "Brushed Aluminum",
                            "material_path": "/World/Looks/Brushed_Aluminum",
                            "path_space": "inspection",
                            "runtime_space": "inspection",
                            "prim_paths": [
                                f"/World/{label}",
                                f"/World/{label}_detail",
                            ],
                            "runtime_prim_paths": [
                                f"/World/{label}",
                                f"/World/{label}_detail",
                            ],
                            "source_prim_paths": [
                                f"/World/{label}_source",
                                f"/World/{label}_detail_source",
                            ],
                            "rationale": "matches reference metal rails",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "api_operation_counts.json").write_text(
            json.dumps({"material_assignment_target_prims": 2}),
            encoding="utf-8",
        )
        (run_dir / "visual_quality_assessment.json").write_text(
            json.dumps(
                {
                    "status": "fixed",
                    "unresolved_issues": [],
                    "assessment_notes": label,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "final_summary.md").write_text(label, encoding="utf-8")
        (raw_dir / "material_decision_patch.json").write_text(
            json.dumps({"label": label}), encoding="utf-8"
        )
        (raw_dir / "rejected_material_assignments.json").write_text(
            "[]", encoding="utf-8"
        )
        (raw_dir / "final_render_records.json").write_text(
            json.dumps([{"image_path": str(final_dir / "final_oblique.png")}]),
            encoding="utf-8",
        )
        (final_dir / "final_oblique.png").write_text(
            f"{label} render", encoding="utf-8"
        )

    write_canonical("initial")
    runner._snapshot_material_step_artifacts(
        run_dir=run_dir,
        trace_writer=TraceWriter(run_dir),
        step_id="01_initial_prediction",
        step_role="initial_prediction",
        iteration=1,
        prompt_path=prompt,
        child_output_path=child_output,
        child_final_path=child_final,
        bridge_artifact_prefix="codex",
        summary="Initial step snapshot.",
    )

    refinement_prompt = raw_dir / "vqa_refinement_prompt_2.md"
    refinement_output = raw_dir / "vqa_refinement_2_child-output.log"
    refinement_final = raw_dir / "vqa_refinement_2_child-final.md"
    for path, text in [
        (refinement_prompt, "refine prompt"),
        (refinement_output, "refine output"),
        (refinement_final, "refine final"),
        (raw_dir / "vqa_refinement_2_request.json", "{}"),
        (raw_dir / "vqa_refinement_2_items.json", "[]"),
        (raw_dir / "vqa_refinement_2_result.json", "{}"),
        (raw_dir / "vqa_refinement_iter2_summary.json", "{}"),
    ]:
        path.write_text(text, encoding="utf-8")
    write_canonical("iteration2")
    runner._snapshot_material_step_artifacts(
        run_dir=run_dir,
        trace_writer=TraceWriter(run_dir),
        step_id="02_vqa_refinement_iter2",
        step_role="vqa_refinement",
        iteration=2,
        prompt_path=refinement_prompt,
        child_output_path=refinement_output,
        child_final_path=refinement_final,
        bridge_artifact_prefix="vqa_refinement_2",
        summary="Iteration 2 snapshot.",
    )

    assert (
        run_dir / "steps" / "01_initial_prediction" / "renders" / "final_oblique.png"
    ).read_text(encoding="utf-8") == "initial render"
    assert (
        run_dir / "steps" / "02_vqa_refinement_iter2" / "renders" / "final_oblique.png"
    ).read_text(encoding="utf-8") == "iteration2 render"
    manifest = json.loads(
        (run_dir / "steps" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["latest_step_id"] == "02_vqa_refinement_iter2"
    assert [step["step_id"] for step in manifest["steps"]] == [
        "01_initial_prediction",
        "02_vqa_refinement_iter2",
    ]
    assert manifest["steps"][1]["material_assignment_groups"] == 1
    assert manifest["steps"][1]["material_assignment_target_prims"] == 2
    assert manifest["steps"][1]["context"]["source_usd"] == "/assets/ladder.usd"
    assert manifest["steps"][1]["context"]["reference_image_count"] == 1
    assert manifest["steps"][1]["context"]["candidate_path_space"] == "inspection"
    assert manifest["steps"][1]["rejected_assignment_count"] == 0
    assert not (run_dir / "steps" / "step_dataset.jsonl").exists()
    step_record = json.loads(
        (run_dir / "steps" / "02_vqa_refinement_iter2" / "step.json").read_text(
            encoding="utf-8"
        )
    )
    assert step_record["artifacts"]["agent"]["runner_result"].endswith(
        "steps/02_vqa_refinement_iter2/agent/vqa_refinement_2_result.json"
    )
    assert step_record["material_assignment_groups"] == 1
    assert step_record["material_assignment_target_prims"] == 2
    assert step_record["context"]["clean_slate"]["clear_materials"] is True
    assert step_record["context"]["optimizer"]["optimize"] is True
    assert step_record["context"]["materials"]["palette_material_count"] == 1
    assert step_record["decision_group_summaries"][0]["family"] == "iteration2 rails"
    assert step_record["decision_group_summaries"][0]["runtime_prim_count"] == 2
    assert step_record["rejected_assignment_summaries"] == []


def test_fallback_finalizer_writes_required_material_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    final_dir = run_dir / "final_renders"
    trace_dir = run_dir / "trace"
    raw_dir.mkdir(parents=True)
    final_dir.mkdir()
    trace_dir.mkdir()

    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    child_output = run_dir / "child-output.log"
    child_final = run_dir / "child-final.md"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        """
entries:
  - name: "Car Paint Orange"
    binding: "/World/Looks/Car_Paint_Orange"
""".strip(),
        encoding="utf-8",
    )
    (final_dir / "final_top.png").write_text("png", encoding="utf-8")
    (raw_dir / "applied_override_groups.json").write_text(
        json.dumps(
            [
                {
                    "name": "orange",
                    "material_name": "Car Paint Orange",
                    "prim_path": "/World/Body",
                    "source_paths": ["/World/Body"],
                    "rationale": "Visible orange body.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (raw_dir / "hierarchy_flat.json").write_text(
        json.dumps(
            [
                {
                    "path": "/World/Body",
                    "type_name": "Mesh",
                    "active": True,
                },
                {
                    "path": "/World/Wheel",
                    "type_name": "Mesh",
                    "active": True,
                },
                {
                    "path": "/World/collisions/BodyCollision",
                    "type_name": "Mesh",
                    "active": True,
                },
            ]
        ),
        encoding="utf-8",
    )

    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=usd,
        reference_images=[reference],
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        workbench_url="http://127.0.0.1:8088",
    )

    finalized = _ensure_material_assignment_artifacts(
        config=config,
        run_dir=run_dir,
        request={"workflow": "materials.assign"},
        trace_writer=TraceWriter(run_dir),
        child_output_path=child_output,
        child_final_path=child_final,
        child_returncode=2,
    )

    assert finalized is True
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert (
        assignments["assignments"][0]["material_path"]
        == "/World/Looks/Car_Paint_Orange"
    )
    assert assignments["assignments"][0]["coverage_status"] == "material_assignment"
    assert assignments["assignments"][0]["prim_paths"] == ["/World/Body"]
    assert assignments["coverage"] == {
        "candidate_visible_prim_count": 2,
        "material_decision_prim_count": 2,
        "material_assignment_prim_count": 1,
        "preserved_existing_prim_count": 0,
        "ambiguous_unassigned_prim_count": 1,
        "coverage_notes": assignments["coverage"]["coverage_notes"],
    }
    assert assignments["assignments"][-1]["coverage_status"] == "ambiguous_unassigned"
    assert assignments["assignments"][-1]["prim_paths"] == ["/World/Wheel"]
    assert assignments["final_review"]["issues_found"] == 2
    counts = json.loads(
        (run_dir / "api_operation_counts.json").read_text(encoding="utf-8")
    )
    assert counts["coverage_candidate_visible_prims"] == 2
    assert counts["coverage_material_decision_prims"] == 2
    assert counts["final_review_issues_found"] == 2
    assert counts["visual_quality_issues_found"] >= 1
    visual_quality = json.loads(
        (run_dir / "visual_quality_assessment.json").read_text(encoding="utf-8")
    )
    assert visual_quality["status"] == "unresolved_issues"
    assert assignments["visual_quality_assessment"]["status"] == "unresolved_issues"
    final_summary = (run_dir / "final_summary.md").read_text(encoding="utf-8")
    assert "## Coverage Summary" in final_summary
    assert "## Visual Quality Assessment" in final_summary
    assert "[ambiguous_unassigned]" in final_summary
    assert child_final.exists()


def test_fallback_coverage_marks_all_visible_candidates_ambiguous(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "hierarchy_flat.json").write_text(
        json.dumps(
            [
                {"path": "/World/Body", "type_name": "Mesh", "active": True},
                {"path": "/World/Wheel", "type_name": "Mesh", "active": True},
            ]
        ),
        encoding="utf-8",
    )

    coverage, final_review, assignments = _fallback_coverage_review_and_assignments(
        run_dir=tmp_path,
        groups=[],
        child_returncode=2,
    )

    assert coverage["candidate_visible_prim_count"] == 2
    assert coverage["material_assignment_prim_count"] == 0
    assert coverage["ambiguous_unassigned_prim_count"] == 2
    assert assignments == [
        {
            "family": "fallback-unreviewed-visible-candidates",
            "coverage_status": "ambiguous_unassigned",
            "material_name": None,
            "material_path": None,
            "prim_paths": ["/World/Body", "/World/Wheel"],
            "rationale": assignments[0]["rationale"],
        }
    ]
    assert final_review["issues_found"] == 2


def test_visual_quality_fallback_deduplicates_child_failure_reason() -> None:
    visual_quality = _visual_quality_from_assignments_or_fallback(
        assignments={},
        final_review={"unresolved_issues": []},
        final_render_paths=[],
        reference_images=[],
        child_returncode=2,
    )

    unresolved = visual_quality["unresolved_issues"]
    assert isinstance(unresolved, list)
    assert len(unresolved) == 2
    assert any("return code 2" in issue for issue in unresolved)
    assert all(
        "did not produce visual_quality_assessment" not in issue for issue in unresolved
    )


def test_fallback_finalizer_traces_corrupted_json_recovery(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    final_dir = run_dir / "final_renders"
    trace_dir = run_dir / "trace"
    raw_dir.mkdir(parents=True)
    final_dir.mkdir()
    trace_dir.mkdir()

    usd = tmp_path / "asset.usd"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    child_output = run_dir / "child-output.log"
    child_final = run_dir / "child-final.md"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        """
entries:
  - name: "Car Paint Orange"
    binding: "/World/Looks/Car_Paint_Orange"
""".strip(),
        encoding="utf-8",
    )
    (final_dir / "final_top.png").write_text("png", encoding="utf-8")
    (raw_dir / "applied_override_groups.json").write_text("{not-json", encoding="utf-8")
    (raw_dir / "hierarchy_flat.json").write_text(
        json.dumps(
            [
                {
                    "path": "/World/visuals/Body",
                    "type_name": "Mesh",
                    "active": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    trace_writer = TraceWriter(run_dir)
    trace_writer.write(
        "assignment",
        phase="material assignment",
        summary="Recovered orange body override.",
        data={
            "material_names": ["Car Paint Orange"],
            "prim_paths": ["/World/visuals/Body"],
        },
    )
    config = MaterialAssignConfig(
        repo_root=tmp_path,
        usd_path=usd,
        reference_images=[reference],
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        workbench_url="http://127.0.0.1:8088",
    )

    finalized = _ensure_material_assignment_artifacts(
        config=config,
        run_dir=run_dir,
        request={"workflow": "materials.assign"},
        trace_writer=trace_writer,
        child_output_path=child_output,
        child_final_path=child_final,
        child_returncode=1,
    )

    assert finalized is True
    events = [
        json.loads(line)
        for line in (trace_dir / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    warnings = [event for event in events if event["event_type"] == "warning"]
    assert warnings
    assert warnings[0]["artifacts"] == [str(raw_dir / "applied_override_groups.json")]
    assignments = json.loads((run_dir / "assignments.json").read_text(encoding="utf-8"))
    assert assignments["assignments"][0]["prim_paths"] == ["/World/visuals/Body"]


def test_visible_candidate_filter_uses_path_components(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "hierarchy_flat.json").write_text(
        json.dumps(
            [
                {
                    "path": "/World/visuals/MyLooksPanel",
                    "type_name": "Mesh",
                    "active": True,
                },
                {
                    "path": "/World/visuals_legacy/Body",
                    "type_name": "Mesh",
                    "active": True,
                },
                {
                    "path": "/World/visuals/Body",
                    "type_name": "Mesh",
                    "active": True,
                },
                {
                    "path": "/World/Looks/MaterialMesh",
                    "type_name": "Mesh",
                    "active": True,
                },
                {
                    "path": "/World/collisions/BodyCollision",
                    "type_name": "Mesh",
                    "active": True,
                },
            ]
        ),
        encoding="utf-8",
    )

    assert runner._visible_candidate_prim_paths(run_dir) == [
        "/World/visuals/MyLooksPanel",
        "/World/visuals/Body",
    ]
