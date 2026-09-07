# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visually grounded physics classification through usd-cli/OVRTX."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from content_workflow_cli import runner
from content_workflow_cli.prompts import build_physics_apply_prompt


def _apply_prompt(tmp_path: Path, **overrides: object) -> str:
    kwargs: dict[str, object] = {
        "repo_root": tmp_path,
        "run_dir": tmp_path / "run",
        "usd_path": tmp_path / "asset.usd",
    }
    kwargs.update(overrides)
    return build_physics_apply_prompt(**kwargs)  # type: ignore[arg-type]


def test_prompt_grounds_classification_in_attached_ovrtx_views(
    tmp_path: Path,
) -> None:
    prompt = _apply_prompt(
        tmp_path,
        classification_view_labels=[
            "usd-cli OVRTX render: classification_oblique",
            "usd-cli OVRTX render: classification_top",
        ],
    )

    assert "ATTACHED ASSET VIEWS" in prompt
    assert "classification_oblique" in prompt
    assert "classification_top" in prompt
    assert "trust the image" in prompt
    assert "classification_views_attached" in prompt


def test_prompt_omits_visual_claim_without_views(tmp_path: Path) -> None:
    prompt = _apply_prompt(tmp_path, classification_view_labels=[])

    assert "ATTACHED ASSET VIEWS" not in prompt


class _FakeSession:
    session_id = "workflow-physics"

    def __init__(self, *, fail_names: set[str] | None = None) -> None:
        self.fail_names = fail_names or set()
        self.opened: list[Path] = []

    def open(self, path: Path) -> dict[str, Any]:
        self.opened.append(path)
        return {"ok": True}

    def render_view(self, **kwargs: Any) -> dict[str, Any]:
        name = str(kwargs["name"])
        if name in self.fail_names:
            raise RuntimeError("render backend unavailable")
        return {
            "name": name,
            "direction": kwargs["direction"],
            "image_path": str(Path(kwargs["output_dir"]) / f"{name}.png"),
            "renderer": "ovrtx",
        }


def test_classification_render_failure_is_recorded_not_raised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSession(
        fail_names={
            "classification_oblique",
            "classification_top",
            "classification_bottom",
        }
    )
    staged = tmp_path / "staged.usda"
    staged.write_text("#usda 1.0\n", encoding="utf-8")

    renders, error, session_id = runner._render_physics_classification_views(
        SimpleNamespace(),  # type: ignore[arg-type]
        tmp_path,
        staged,
        usd_cli_session=fake,  # type: ignore[arg-type]
    )

    assert renders == []
    assert error is not None and "render backend unavailable" in error
    assert session_id == "workflow-physics"


def test_classification_renders_keep_partial_views_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSession(fail_names={"classification_top"})
    staged = tmp_path / "staged.usda"
    staged.write_text("#usda 1.0\n", encoding="utf-8")

    renders, error, session_id = runner._render_physics_classification_views(
        SimpleNamespace(),  # type: ignore[arg-type]
        tmp_path,
        staged,
        usd_cli_session=fake,  # type: ignore[arg-type]
    )

    assert [record["name"] for record in renders] == [
        "classification_oblique",
        "classification_bottom",
    ]
    assert error is not None and "classification_top" in error
    assert session_id == "workflow-physics"


def test_prepare_packet_records_usd_cli_classification_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "run" / "inputs" / "source" / "asset.usda"
    staged.parent.mkdir(parents=True)
    staged.write_text("#usda 1.0\n", encoding="utf-8")
    staged_source = SimpleNamespace(
        staged_usd_path=staged,
        source_usd_path=tmp_path / "asset.usda",
        source_sha256="a" * 64,
        contract_metadata=lambda: {
            "staged_usd_path": str(staged),
            "source_usd_path": str(tmp_path / "asset.usda"),
            "source_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        runner,
        "_stage_usd_cli_input_tree",
        lambda **_kwargs: staged_source,
    )
    from content_agent_workflows.physics import scene_ops

    monkeypatch.setattr(
        scene_ops,
        "inspect_components",
        lambda *_args, **_kwargs: {"component_count": 1, "components": []},
    )
    monkeypatch.setattr(
        scene_ops,
        "inspect_topology",
        lambda *_args, **_kwargs: {"nodes": [], "edges": []},
    )
    monkeypatch.setattr(
        runner,
        "_render_physics_classification_views",
        lambda *_args, **_kwargs: (
            [
                {
                    "name": "classification_oblique",
                    "image_path": str(tmp_path / "view.png"),
                    "renderer": "ovrtx",
                }
            ],
            None,
            "workflow-physics",
        ),
    )
    config = SimpleNamespace(optimize=True, usd_path=tmp_path / "asset.usda")

    packet = runner._prepare_usd_cli_physics_run_packet(
        config,  # type: ignore[arg-type]
        tmp_path / "run",
        usd_cli_session=_FakeSession(),  # type: ignore[arg-type]
    )

    assert packet["scene_backend"] == "usd-cli"
    assert packet["session_id"] == "workflow-physics"
    assert packet["initial_evidence_renders"][0]["renderer"] == "ovrtx"
