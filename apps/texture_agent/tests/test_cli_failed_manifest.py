# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from texture_agent.cli import app
from texture_agent.config import unified_config
from texture_agent.workflows import factory as workflow_factory


def test_run_failure_after_partial_output_writes_manifest_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("input: {}\n", encoding="utf-8")
    working_dir = tmp_path / "work"

    class _PrepareTask:
        name = "prepare_uvs"
        description = "prepare"

        def run(self, context: dict[str, Any]) -> dict[str, Any]:
            return context

    class _FailingTask:
        name = "generate_textures"
        description = "generate"

        def run(self, context: dict[str, Any]) -> dict[str, Any]:
            partial = Path(context["working_dir"]) / "generated" / "partial.png"
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b"partial")
            raise RuntimeError("provider response body with token=FAKESECRET")

    monkeypatch.setattr(
        unified_config,
        "load_config",
        lambda path, session_id=None, *, config_data: {},
    )
    monkeypatch.setattr(
        unified_config,
        "config_to_context",
        lambda config: {"working_dir": str(working_dir)},
    )
    monkeypatch.setattr(
        workflow_factory,
        "create_texture_pipeline_workflow",
        lambda context, skip=None, only=None: [_PrepareTask(), _FailingTask()],
    )

    result = CliRunner().invoke(app, ["run", str(config_path)])

    assert result.exit_code == 1
    manifest = json.loads(
        (working_dir / "artifacts_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"]["state"] == "failed"
    assert manifest["status"]["completed_steps"] == ["prepare_uvs"]
    assert manifest["status"]["failed_step"] == "generate_textures"
    assert manifest["status"]["error_code"] == "TEXTURE_PIPELINE_STEP_FAILED"
    assert manifest["status"]["error"] == "Texture Agent pipeline step failed."
    assert manifest["status"]["partial_artifacts"] == [
        {"path": "generated/partial.png", "size_bytes": 7}
    ]
    serialized = json.dumps(manifest, sort_keys=True)
    assert "FAKESECRET" not in serialized
    assert "provider response body" not in serialized
