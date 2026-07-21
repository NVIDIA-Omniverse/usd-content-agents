# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import texture_agent.workflows.factory as factory


@dataclass
class _FakeTask:
    name: str
    description: str
    run_marker: str

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        context.setdefault("executed", []).append(self.run_marker)
        return context


@dataclass
class _FailingAfterArtifactTask:
    name: str
    description: str
    error: Exception

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        partial = Path(context["working_dir"]) / "generated" / "partial.png"
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"partial artifact")
        context["generate_textures_errors"] = [
            {
                "message": (
                    "provider body Authorization: Bearer "
                    "sk-FAKESECRET12345678 traceback details"
                )
            }
        ]
        raise self.error


def test_create_texture_pipeline_workflow_respects_order_and_filters(
    monkeypatch,
) -> None:
    monkeypatch.setattr(factory, "STEP_ORDER", ["one", "two", "three"])
    monkeypatch.setattr(
        factory,
        "_STEP_TASKS",
        {
            "one": lambda: _FakeTask("one", "first", "one"),
            "two": lambda: _FakeTask("two", "second", "two"),
            "three": lambda: _FakeTask("three", "third", "three"),
        },
    )

    tasks = factory.create_texture_pipeline_workflow(
        {"steps": {"two": {"enabled": False}}},
        skip=["three"],
    )

    assert [task.name for task in tasks] == ["one"]


def test_create_texture_pipeline_workflow_only_filter(monkeypatch) -> None:
    monkeypatch.setattr(factory, "STEP_ORDER", ["one", "two", "three"])
    monkeypatch.setattr(
        factory,
        "_STEP_TASKS",
        {
            "one": lambda: _FakeTask("one", "first", "one"),
            "two": lambda: _FakeTask("two", "second", "two"),
            "three": lambda: _FakeTask("three", "third", "three"),
        },
    )

    tasks = factory.create_texture_pipeline_workflow({}, only=["two", "three"])

    assert [task.name for task in tasks] == ["two", "three"]


def test_create_texture_pipeline_workflow_trims_step_filters(monkeypatch) -> None:
    monkeypatch.setattr(factory, "STEP_ORDER", ["one", "two", "three"])
    monkeypatch.setattr(
        factory,
        "_STEP_TASKS",
        {
            "one": lambda: _FakeTask("one", "first", "one"),
            "two": lambda: _FakeTask("two", "second", "two"),
            "three": lambda: _FakeTask("three", "third", "three"),
        },
    )

    tasks = factory.create_texture_pipeline_workflow({}, only=[" two ", "three"])

    assert [task.name for task in tasks] == ["two", "three"]


def test_create_texture_pipeline_workflow_rejects_unknown_step(monkeypatch) -> None:
    monkeypatch.setattr(factory, "STEP_ORDER", ["one", "two", "three"])

    with pytest.raises(ValueError, match="Invalid --only step name"):
        factory.create_texture_pipeline_workflow({}, only=["two", "bogus"])


def test_create_texture_pipeline_workflow_rejects_empty_step_name() -> None:
    with pytest.raises(ValueError, match="empty step name"):
        factory.create_texture_pipeline_workflow({}, skip=["render", " "])


def test_create_texture_pipeline_workflow_rejects_skip_and_only() -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        factory.create_texture_pipeline_workflow(
            {},
            skip=["render"],
            only=["apply_textures"],
        )


def test_full_workflow_cannot_skip_plan_before_backend_work() -> None:
    with pytest.raises(ValueError, match="plan_textures cannot be skipped"):
        factory.create_texture_pipeline_workflow({}, skip=["plan_textures"])


def test_run_pipeline_dry_run_does_not_execute(monkeypatch) -> None:
    tasks = [_FakeTask("one", "first", "one"), _FakeTask("two", "second", "two")]
    monkeypatch.setattr(
        factory,
        "create_texture_pipeline_workflow",
        lambda context, skip=None, only=None: tasks,
    )

    context = {"executed": []}
    result = factory.run_pipeline(context, dry_run=True)

    assert result is context
    assert context["executed"] == []


def test_run_pipeline_executes_tasks_in_sequence(monkeypatch) -> None:
    tasks = [_FakeTask("one", "first", "one"), _FakeTask("two", "second", "two")]
    monkeypatch.setattr(
        factory,
        "create_texture_pipeline_workflow",
        lambda context, skip=None, only=None: tasks,
    )

    context = {"executed": []}
    result = factory.run_pipeline(context)

    assert result["executed"] == ["one", "two"]


def test_run_pipeline_writes_failure_safe_manifest_after_partial_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_error = RuntimeError(
        "provider body Authorization: Bearer sk-FAKESECRET12345678 traceback"
    )
    tasks = [
        _FakeTask("prepare_uvs", "prepare", "prepare_uvs"),
        _FailingAfterArtifactTask("generate_textures", "generate", original_error),
    ]
    monkeypatch.setattr(
        factory,
        "create_texture_pipeline_workflow",
        lambda context, skip=None, only=None: tasks,
    )

    with pytest.raises(RuntimeError) as exc_info:
        factory.run_pipeline(
            {
                "working_dir": str(tmp_path),
                "executed": [],
                "output_portability": {"diagnostics": ["portability token=FAKESECRET"]},
            }
        )

    assert exc_info.value is original_error
    manifest = json.loads(
        (tmp_path / "artifacts_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"]["state"] == "failed"
    assert manifest["status"]["completed_steps"] == ["prepare_uvs"]
    assert manifest["status"]["failed_step"] == "generate_textures"
    assert manifest["status"]["error_code"] == "TEXTURE_PIPELINE_STEP_FAILED"
    assert manifest["status"]["error"] == "Texture Agent pipeline step failed."
    assert manifest["status"]["partial_artifacts"] == [
        {"path": "generated/partial.png", "size_bytes": 16}
    ]
    serialized = json.dumps(manifest, sort_keys=True)
    assert "FAKESECRET" not in serialized
    assert "provider body" not in serialized
    assert "traceback" not in serialized
    assert manifest["textures"]["generation_errors"] == []
    assert manifest["outputs"]["portability"]["diagnostics"] == []


def test_run_pipeline_manifest_failure_does_not_mask_original_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from texture_agent.functions import artifact_manifest

    original_error = RuntimeError("original pipeline failure")
    task = _FailingAfterArtifactTask("generate_textures", "generate", original_error)
    monkeypatch.setattr(
        factory,
        "create_texture_pipeline_workflow",
        lambda context, skip=None, only=None: [task],
    )
    monkeypatch.setattr(
        artifact_manifest,
        "write_failed_artifacts_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("secondary manifest failure")
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        factory.run_pipeline({"working_dir": str(tmp_path)})

    assert exc_info.value is original_error


def test_run_pipeline_requires_persisted_plan_for_backend_only_workflow(
    monkeypatch,
    tmp_path,
) -> None:
    tasks = [_FakeTask("GenerateTextures", "backend", "generate")]
    monkeypatch.setattr(
        factory,
        "create_texture_pipeline_workflow",
        lambda context, skip=None, only=None: tasks,
    )

    with pytest.raises(RuntimeError, match="requires texture_plan.json"):
        factory.run_pipeline({"working_dir": str(tmp_path)})


@pytest.mark.parametrize(
    "context",
    [
        {"cached_apply_only": True},
        {"planning_config": {"resume_apply_textures": True}},
    ],
)
def test_cached_apply_prompt_hydration_does_not_require_plan(
    monkeypatch,
    context,
) -> None:
    tasks = [_FakeTask("GeneratePrompts", "cache hydration", "prompts")]
    monkeypatch.setattr(
        factory,
        "create_texture_pipeline_workflow",
        lambda current, skip=None, only=None: tasks,
    )

    result = factory.run_pipeline(context)

    assert result["executed"] == ["prompts"]


def test_cached_apply_does_not_bypass_image_generation_plan_gate(
    monkeypatch,
    tmp_path,
) -> None:
    tasks = [_FakeTask("ExecuteTexturePlan", "backend", "generate")]
    monkeypatch.setattr(
        factory,
        "create_texture_pipeline_workflow",
        lambda context, skip=None, only=None: tasks,
    )

    with pytest.raises(RuntimeError, match="requires texture_plan.json"):
        factory.run_pipeline(
            {
                "working_dir": str(tmp_path),
                "cached_apply_only": True,
            }
        )
