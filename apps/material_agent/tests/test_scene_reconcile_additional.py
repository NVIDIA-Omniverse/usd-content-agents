# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional tests for material_agent.scene.reconcile orchestration paths."""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import material_agent.scene.reconcile as reconcile
from material_agent.scene.reconcile import (
    _gather_predictions,
    _llm_reconcile,
    _remap_predictions_file,
    apply_remapping,
    reconcile_predictions,
)


def _write_predictions(path: Path, predictions: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(pred) for pred in predictions) + "\n")


def _read_predictions(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_gather_predictions_uses_best_file_and_skips_bad_rows(tmp_path: Path) -> None:
    asset_dir = tmp_path / "asset_a"
    raw_path = asset_dir / "predictions" / "predictions.jsonl"
    restored_path = asset_dir / "restored" / "restored_predictions.jsonl"
    _write_predictions(raw_path, [{"id": "/raw", "materials": {"material": "Raw"}}])
    restored_path.parent.mkdir(parents=True, exist_ok=True)
    restored_path.write_text(
        json.dumps({"id": "/restored", "materials": {"material": "Steel"}})
        + "\n"
        + "\n"
        + "{bad json}\n"
    )

    manifest = SimpleNamespace(
        sub_assets=[
            SimpleNamespace(
                status="completed",
                working_dir=str(asset_dir),
                name="AssetA",
            ),
            SimpleNamespace(
                status="failed", working_dir=str(tmp_path / "ignored"), name="Ignored"
            ),
            SimpleNamespace(status="completed", working_dir=None, name="NoDir"),
            SimpleNamespace(
                status="completed",
                working_dir=str(tmp_path / "missing_predictions"),
                name="MissingPredictions",
            ),
        ],
        payload_groups=[],
    )

    gathered = _gather_predictions(manifest)

    assert gathered == [
        {
            "id": "/restored",
            "materials": {"material": "Steel"},
            "_asset_name": "AssetA",
        }
    ]


def test_reconcile_predictions_short_circuits_and_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = SimpleNamespace()

    monkeypatch.setattr(reconcile, "_gather_predictions", lambda manifest: [])
    assert reconcile_predictions(manifest, {"backend": "mock"}) == {}

    monkeypatch.setattr(
        reconcile,
        "_gather_predictions",
        lambda manifest: [
            {"_asset_name": "AssetA", "materials": {"material": "Steel"}}
        ],
    )
    monkeypatch.setattr(reconcile, "_detect_ambiguous_pairs", lambda distributions: {})
    assert reconcile_predictions(manifest, {"backend": "mock"}) == {}

    captured: dict[str, object] = {}
    ambiguous = {
        "orange_group": {
            "materials": {"Car Paint Orange": 2, "Steel Painted Orange": 3},
            "asset_count": 1,
        }
    }
    monkeypatch.setattr(
        reconcile,
        "_gather_predictions",
        lambda manifest: [
            {"_asset_name": "AssetA", "materials": {"material": "Car Paint Orange"}},
            {
                "_asset_name": "AssetA",
                "materials": {"material": "Steel Painted Orange"},
            },
        ],
    )
    monkeypatch.setattr(
        reconcile, "_detect_ambiguous_pairs", lambda distributions: ambiguous
    )
    monkeypatch.setattr(
        reconcile,
        "_llm_reconcile",
        lambda ambiguous_groups, llm_config, materials_list, token_tracker=None: (
            captured.update(
                {
                    "ambiguous": ambiguous_groups,
                    "llm_config": llm_config,
                    "materials_list": materials_list,
                    "token_tracker": token_tracker,
                }
            )
            or {"Car Paint Orange": "Steel Painted Orange"}
        ),
    )

    result = reconcile_predictions(
        manifest,
        {"backend": "mock", "model": "fake"},
        ["Steel Painted Orange"],
    )

    assert result == {"Car Paint Orange": "Steel Painted Orange"}
    assert captured["ambiguous"] == ambiguous
    assert captured["llm_config"] == {"backend": "mock", "model": "fake"}
    assert captured["materials_list"] == ["Steel Painted Orange"]
    assert captured["token_tracker"] is None


def test_apply_remapping_updates_sub_assets_and_payloads(tmp_path: Path) -> None:
    asset_dir = tmp_path / "asset_a"
    asset_predictions = asset_dir / "predictions" / "predictions.jsonl"
    _write_predictions(
        asset_predictions,
        [{"id": "/a", "materials": {"material": "Car Paint Orange"}}],
    )

    payload_config = tmp_path / "configs" / "payload.yaml"
    payload_config.parent.mkdir(parents=True)
    payload_config.write_text("config")
    payload_working_dir = payload_config.parent / ".payload"
    payload_predictions = (
        payload_working_dir / "restored" / "restored_predictions.jsonl"
    )
    _write_predictions(
        payload_predictions,
        [{"id": "/payload", "materials": {"material": "Car Paint Orange"}}],
    )

    manifest = SimpleNamespace(
        sub_assets=[
            SimpleNamespace(
                status="completed", working_dir=str(asset_dir), name="AssetA"
            ),
            SimpleNamespace(
                status="completed",
                working_dir=str(tmp_path / "asset_without_predictions"),
                name="NoPredictions",
            ),
            SimpleNamespace(
                status="failed", working_dir=str(tmp_path / "ignored"), name="Ignored"
            ),
        ],
        payload_groups=[
            SimpleNamespace(status="completed", config_path=str(payload_config)),
            SimpleNamespace(
                status="completed",
                config_path=str(tmp_path / "configs" / "empty_payload.yaml"),
            ),
            SimpleNamespace(status="completed", config_path=None),
            SimpleNamespace(status="failed", config_path=str(tmp_path / "skip.yaml")),
        ],
    )

    remap = {"Car Paint Orange": "Steel Painted Orange"}
    assert apply_remapping(manifest, {}) == 0
    updated = apply_remapping(manifest, remap)

    assert updated == 2
    assert (
        _read_predictions(asset_predictions)[0]["materials"]["material"]
        == "Steel Painted Orange"
    )
    assert (
        _read_predictions(payload_predictions)[0]["materials"]["material"]
        == "Steel Painted Orange"
    )


def _install_llm_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    llm_factory: object,
    load_dotenv: object | None = None,
) -> None:
    class FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    monkeypatch.setitem(
        sys.modules,
        "langchain_core.messages",
        SimpleNamespace(HumanMessage=FakeMessage, SystemMessage=FakeMessage),
    )
    monkeypatch.setitem(
        sys.modules,
        "world_understanding.functions.models.chat_models",
        SimpleNamespace(create_chat_model_from_config=llm_factory),
    )
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        SimpleNamespace(load_dotenv=load_dotenv or (lambda: None)),
    )


def test_llm_reconcile_handles_missing_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_llm_modules(monkeypatch, llm_factory=lambda config: None)

    result = _llm_reconcile(
        {
            "orange_group": {
                "materials": {"Car Paint Orange": 2, "Steel Painted Orange": 3},
                "asset_count": 1,
            }
        },
        {"backend": "mock"},
        ["Steel Painted Orange"],
    )

    assert result == {}


def test_llm_reconcile_builds_prompt_and_parses_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_dotenv = Mock()

    class FakeLLM:
        def __init__(self) -> None:
            self.messages: list[object] | None = None

        def invoke(self, messages: list[object]) -> object:
            self.messages = messages
            return SimpleNamespace(
                content='{"Car Paint Orange": "Steel Painted Orange"}'
            )

    llm = FakeLLM()
    _install_llm_modules(
        monkeypatch,
        llm_factory=lambda config: llm,
        load_dotenv=load_dotenv,
    )

    result = _llm_reconcile(
        {
            "orange_group": {
                "materials": {"Car Paint Orange": 2, "Steel Painted Orange": 3},
                "asset_count": 1,
            }
        },
        {"backend": "mock", "model": "fake"},
        ["Steel Painted Orange", "Car Paint Orange"],
    )

    assert result == {"Car Paint Orange": "Steel Painted Orange"}
    load_dotenv.assert_called_once()
    assert llm.messages is not None
    assert "orange_group" in llm.messages[1].content
    assert "Valid materials in the library" in llm.messages[1].content


def test_llm_reconcile_skips_missing_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def invoke(self, _messages: list[object]) -> object:
            return SimpleNamespace(
                content='{"remap": {"Car Paint Orange": "Steel Painted Orange"}}'
            )

    _install_llm_modules(
        monkeypatch,
        llm_factory=lambda config: FakeLLM(),
    )
    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "dotenv":
            raise ImportError("no dotenv")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert _llm_reconcile(
        {
            "orange_group": {
                "materials": {"Car Paint Orange": 2, "Steel Painted Orange": 3},
                "asset_count": 1,
            }
        },
        {"backend": "mock", "model": "fake"},
    ) == {"Car Paint Orange": "Steel Painted Orange"}


def test_remap_predictions_file_keeps_blank_and_invalid_lines(tmp_path: Path) -> None:
    pred_file = tmp_path / "predictions.jsonl"
    pred_file.write_text(
        json.dumps({"id": "/a", "materials": {"material": "Old"}})
        + "\n\n"
        + "{not-json}\n"
    )

    assert _remap_predictions_file(pred_file, {"Old": "New"}) == 1
    assert pred_file.read_text().splitlines()[1] == ""
    assert pred_file.read_text().splitlines()[2] == "{not-json}"
