# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subprocess parity coverage for secret-safe unified step transport."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "fresh-process-resume"])
@pytest.mark.parametrize("agent", ["material", "physics", "joint"])
def test_subprocess_step_receives_secret_in_memory_only(
    tmp_path: Path, agent: str, resume: bool
) -> None:
    workdir = tmp_path / agent
    if resume:
        # Model a failed prior process using only the durable, secret-free
        # checkpoint contract. The subprocess below starts with no inherited
        # Python state and must receive credentials again from its current
        # source configuration rather than a process-local registry.
        workdir.mkdir(parents=True)
        (workdir / ".pipeline_state.json").write_text(
            json.dumps(
                {
                    "completed_steps": [],
                    "failed_steps": ["predict"],
                    "step_outputs": {},
                    "current_step": None,
                }
            ),
            encoding="utf-8",
        )
    repo_root = Path(__file__).resolve().parents[1]
    env = {
        name: os.environ[name]
        for name in ("HOME", "LD_LIBRARY_PATH", "PATH", "PYTHONPATH", "VIRTUAL_ENV")
        if name in os.environ
    }
    python_paths = [
        str(repo_root / "apps" / "material_agent"),
        str(repo_root / "apps" / "physics_agent"),
        str(repo_root / "apps" / "joint_agent"),
    ]
    if existing_python_path := env.get("PYTHONPATH"):
        python_paths.append(existing_python_path)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.update(
        {
            "WU_TEST_AGENT": agent,
            "WU_TEST_SECRET": f"{agent}-subprocess-secret",
            "WU_TEST_SHORT_SECRET": "xy",
            "WU_TEST_WORKDIR": str(workdir),
            "WU_TEST_RESUME": "1" if resume else "0",
        }
    )
    code = textwrap.dedent(
        """
        import os
        from pathlib import Path

        agent = os.environ["WU_TEST_AGENT"]
        secret = os.environ["WU_TEST_SECRET"]
        short_secret = os.environ["WU_TEST_SHORT_SECRET"]
        workdir = Path(os.environ["WU_TEST_WORKDIR"])
        resume = os.environ["WU_TEST_RESUME"] == "1"
        workdir.mkdir(parents=True, exist_ok=True)
        source = workdir / "source.yaml"
        source.write_text("steps: {}\\n", encoding="utf-8")
        captured = {}

        class Workflow:
            def run(self, context):
                captured.update(context)
                return {
                    "predictions_path": "predictions.jsonl",
                    "predictions_count": 1,
                    "output_key": "classification",
                }

        if agent == "material":
            import material_agent.workflows as workflows
            from material_agent.tasks.unified_pipeline_executor import (
                UnifiedPipelineExecutorTask,
            )
        elif agent == "physics":
            import physics_agent.workflows as workflows
            from physics_agent.tasks.unified_pipeline_executor import (
                UnifiedPipelineExecutorTask,
            )
        else:
            import joint_agent.workflows as workflows
            from joint_agent.tasks.unified_pipeline_executor import (
                UnifiedPipelineExecutorTask,
            )

        workflows.create_prediction_workflow_from_config = lambda: Workflow()
        executor = UnifiedPipelineExecutorTask()
        step_config = {
            "vlm": {"api_key": secret, "nested": [{"token": short_secret}]}
        }
        if resume:
            executor.run(
                {
                    "steps_to_run": ["predict"],
                    "step_configs": {"predict": step_config},
                    "working_dir": workdir,
                    "config_path": source,
                    "resume": True,
                }
            )
        else:
            executor._execute_step(
                "predict",
                step_config,
                {"working_dir": workdir, "config_path": source},
                None,
                {"step_outputs": {}},
            )
        assert captured["config_dict"]["vlm"]["api_key"] == secret
        assert captured["config_dict"]["vlm"]["nested"][0]["token"] == short_secret
        assert str(captured["config_path"]) == str(source)
        assert not (workdir / ".pipeline_temp").exists()
        print("ok")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    artifact_text = "".join(
        path.read_text(encoding="utf-8")
        for path in workdir.rglob("*")
        if path.is_file()
    )
    assert f"{agent}-subprocess-secret" not in artifact_text
    assert "xy" not in artifact_text
