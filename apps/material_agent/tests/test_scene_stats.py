# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for scene-level statistics reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from material_agent.scene.manifest import PayloadGroup, SceneManifest, SubAsset
from material_agent.scene.stats import (
    _find_member_token_stats,
    _load_token_usage_json,
    _parse_prediction_report_html,
    normalise_token_stats,
    record_model_response_usage,
    write_scene_stats_report,
)


def test_write_scene_stats_report_aggregates_token_usage_json(
    tmp_path: Path,
) -> None:
    working_dir = tmp_path / "scene"
    asset_dir = working_dir / "configs" / ".asset_a"
    predictions_dir = asset_dir / "predictions"
    predictions_dir.mkdir(parents=True)
    predictions_path = predictions_dir / "predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")
    (predictions_dir / "token_usage.json").write_text(
        json.dumps(
            {
                "scope": "asset_predict",
                "token_usage": {
                    "total_input_tokens": 10,
                    "total_output_tokens": 5,
                    "total_tokens": 15,
                    "invocation_count": 2,
                    "by_model": {
                        "vlm-model": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                            "count": 2,
                        }
                    },
                    "by_type": {
                        "vlm": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                            "count": 2,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = SceneManifest(
        scene_usd_path="/tmp/scene.usd",
        sub_assets=[
            SubAsset(
                id="asset_a",
                name="Asset A",
                prim_path="/World/AssetA",
                working_dir=str(asset_dir),
                predictions_path=str(predictions_path),
                status="completed",
            )
        ],
    )
    report_path = write_scene_stats_report(
        manifest=manifest,
        working_dir=working_dir,
        scene_operation_token_stats={
            "total_input_tokens": 3,
            "total_output_tokens": 2,
            "total_tokens": 5,
            "invocation_count": 1,
            "by_model": {
                "llm-model": {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                    "count": 1,
                }
            },
            "by_type": {
                "scene_reconcile_llm": {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                    "count": 1,
                }
            },
        },
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["token_usage"]["asset_predict"]["total_tokens"] == 15
    assert report["token_usage"]["scene_operations"]["total_tokens"] == 5
    assert report["token_usage"]["total"]["total_tokens"] == 20
    assert report["token_usage"]["total"]["invocation_count"] == 3
    assert report["assets"][0]["token_usage_source"] == "token_usage_json"
    assert report["missing_token_usage"] == []
    assert Path(report["output_usd_path"]).parts[-2:] == (
        "output",
        "composed_scene.usd",
    )


def test_write_scene_stats_report_finds_restored_prediction_token_usage(
    tmp_path: Path,
) -> None:
    working_dir = tmp_path / "scene"
    asset_dir = working_dir / "configs" / ".asset_a"
    predictions_dir = asset_dir / "predictions"
    restored_dir = predictions_dir / "restored"
    restored_dir.mkdir(parents=True)
    predictions_path = restored_dir / "restored_predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")
    (predictions_dir / "token_usage.json").write_text(
        json.dumps(
            {
                "token_usage": {
                    "total_input_tokens": 7,
                    "total_output_tokens": 3,
                    "total_tokens": 10,
                    "invocation_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = SceneManifest(
        scene_usd_path="/tmp/scene.usd",
        sub_assets=[
            SubAsset(
                id="asset_a",
                name="Asset A",
                prim_path="/World/AssetA",
                predictions_path=str(predictions_path),
                status="completed",
            )
        ],
    )
    report_path = write_scene_stats_report(manifest=manifest, working_dir=working_dir)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["token_usage"]["asset_predict"]["total_tokens"] == 10
    assert report["assets"][0]["token_usage_source"] == "token_usage_json"
    assert report["missing_token_usage"] == []


def test_write_scene_stats_report_recovers_prediction_report_html(
    tmp_path: Path,
) -> None:
    working_dir = tmp_path / "scene"
    payload_dir = working_dir / "configs" / ".payload_a"
    predictions_dir = payload_dir / "predictions"
    predictions_dir.mkdir(parents=True)
    predictions_path = predictions_dir / "predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")
    (predictions_dir / "prediction_report.html").write_text(
        """
        <html><body>
        <h2>Token Usage</h2>
        <div class="metric-label">Total Tokens</div>
        <div class="metric-value">1,234</div>
        <div class="metric-label">Input Tokens</div>
        <div class="metric-value">1,000</div>
        <div class="metric-label">Output Tokens</div>
        <div class="metric-value">234</div>
        <div class="metric-label">VLM Calls</div>
        <div class="metric-value">7</div>
        <h3>By Model</h3>
        <table>
          <tr><th>Model</th><th>Calls</th><th>Input Tokens</th>
              <th>Output Tokens</th><th>Total Tokens</th></tr>
          <tr><td>vlm-html</td><td>7</td><td>1,000</td><td>234</td>
              <td>1,234</td></tr>
        </table>
        </body></html>
        """,
        encoding="utf-8",
    )

    manifest = SceneManifest(
        scene_usd_path="/tmp/scene.usd",
        payload_groups=[
            PayloadGroup(
                id="payload_a",
                group_name="Payload A",
                payload_file="/tmp/payload.usd",
                working_dir=str(payload_dir),
                predictions_path=str(predictions_path),
                status="completed",
            )
        ],
    )
    report_path = write_scene_stats_report(manifest=manifest, working_dir=working_dir)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    asset_usage = report["token_usage"]["asset_predict"]
    assert asset_usage["total_input_tokens"] == 1000
    assert asset_usage["total_output_tokens"] == 234
    assert asset_usage["total_tokens"] == 1234
    assert asset_usage["invocation_count"] == 7
    assert asset_usage["by_model"]["vlm-html"]["count"] == 7
    assert report["assets"][0]["token_usage_source"] == "prediction_report_html"


def test_parse_prediction_report_html_warns_when_metrics_missing(
    tmp_path: Path,
    caplog,
) -> None:
    report_path = tmp_path / "prediction_report.html"
    report_path.write_text(
        """
        <html><body>
        <h2>Token Usage</h2>
        <table><tr><th>Model</th><th>Calls</th></tr></table>
        </body></html>
        """,
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        assert _parse_prediction_report_html(report_path) is None

    assert "did not contain parseable token usage metrics" in caplog.text


def test_write_scene_stats_report_records_custom_output_usd_path(
    tmp_path: Path,
) -> None:
    working_dir = tmp_path / "scene"
    custom_output = tmp_path / "exports" / "client_scene.usd"
    manifest = SceneManifest(scene_usd_path="/tmp/scene.usd")

    report_path = write_scene_stats_report(
        manifest=manifest,
        working_dir=working_dir,
        output_dir=working_dir / "output",
        output_usd_path=custom_output,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["output_usd_path"] == str(custom_output)


def test_write_scene_stats_report_preserves_existing_report_on_write_error(
    tmp_path: Path,
) -> None:
    working_dir = tmp_path / "scene"
    output_dir = working_dir / "output"
    output_dir.mkdir(parents=True)
    report_path = output_dir / "scene_stats_report.json"
    report_path.write_text('{"existing": true}\n', encoding="utf-8")

    with (
        patch("material_agent.scene.stats.json.dump", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        write_scene_stats_report(
            manifest=SceneManifest(scene_usd_path="/tmp/scene.usd"),
            working_dir=working_dir,
            output_dir=output_dir,
        )

    assert json.loads(report_path.read_text(encoding="utf-8")) == {"existing": True}


def test_token_stats_normalise_handles_loose_values_and_breakdowns() -> None:
    stats = normalise_token_stats(
        {
            "input_tokens": "1,000",
            "output_tokens": "250",
            "invocation_count": None,
            "by_model": {
                "ignored": "not-a-dict",
                "vlm": {
                    "input_tokens": "",
                    "output_tokens": None,
                    "total_tokens": "-",
                    "count": "2 calls",
                },
            },
        }
    )

    assert stats["total_tokens"] == 1250
    assert stats["invocation_count"] == 0
    assert stats["by_model"]["vlm"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "count": 2,
    }


def test_load_token_usage_json_accepts_direct_dict_and_ignores_other_shapes(
    tmp_path: Path,
) -> None:
    direct = tmp_path / "direct.json"
    direct.write_text(
        json.dumps({"input_tokens": 4, "output_tokens": 5}),
        encoding="utf-8",
    )
    other = tmp_path / "other.json"
    other.write_text("[1, 2, 3]", encoding="utf-8")

    assert _load_token_usage_json(direct)["total_tokens"] == 9
    assert _load_token_usage_json(other) is None


def test_parse_prediction_report_html_skips_empty_model_rows(tmp_path: Path) -> None:
    report_path = tmp_path / "prediction_report.html"
    report_path.write_text(
        """
        <html><body>
        <div>Total Tokens</div><div>12</div>
        <div>Input Tokens</div><div>10</div>
        <div>Output Tokens</div><div>2</div>
        <div>VLM Calls</div><div>1</div>
        <table>
          <tr><th>Model</th><th>Calls</th><th>Input Tokens</th>
              <th>Output Tokens</th><th>Total Tokens</th></tr>
          <tr><td>empty-model</td><td>-</td><td></td><td></td><td></td></tr>
        </table>
        </body></html>
        """,
        encoding="utf-8",
    )

    stats = _parse_prediction_report_html(report_path)

    assert stats["total_tokens"] == 12
    assert "empty-model" not in stats["by_model"]


def test_find_member_token_stats_skips_duplicate_candidate_dirs(tmp_path: Path) -> None:
    asset_dir = tmp_path / "asset"
    predictions_dir = asset_dir / "predictions"
    predictions_dir.mkdir(parents=True)
    predictions_path = predictions_dir / "predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")
    member = SubAsset(
        id="asset",
        name="Asset",
        prim_path="/World/Asset",
        working_dir=str(asset_dir),
        predictions_path=str(predictions_path),
        status="completed",
    )

    assert _find_member_token_stats(member) == (None, None, None)


def test_record_model_response_usage_forwards_langchain_usage() -> None:
    class Tracker:
        def __init__(self) -> None:
            self.usages = []

        def add_usage(self, usage) -> None:
            self.usages.append(usage)

    tracker = Tracker()
    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 7,
            "output_tokens": 3,
            "total_tokens": 10,
        }
    )

    record_model_response_usage(tracker, response, "vlm-model", "scene")

    assert len(tracker.usages) == 1
    assert tracker.usages[0].model_name == "vlm-model"
    assert tracker.usages[0].invocation_type == "scene"
    assert tracker.usages[0].total_tokens == 10
