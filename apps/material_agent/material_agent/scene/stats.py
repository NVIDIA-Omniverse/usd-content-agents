# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene-level statistics and token usage reports."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .manifest import PayloadGroup, SceneManifest, SubAsset

logger = logging.getLogger(__name__)


class _ReportTextParser(HTMLParser):
    """Extract text tokens and table rows from generated prediction reports."""

    def __init__(self) -> None:
        super().__init__()
        self.text_items: list[str] = []
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        self.text_items.append(text)
        if self._cell is not None:
            self._cell.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None:
            if self._row is not None:
                self._row.append(" ".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _empty_token_stats() -> dict[str, Any]:
    return {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "invocation_count": 0,
        "by_model": {},
        "by_type": {},
    }


def _parse_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    cleaned = re.sub(r"[^\d-]", "", text)
    if not cleaned or cleaned == "-":
        return 0
    return int(cleaned)


def _normalise_breakdown(raw: dict[str, Any]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        result[str(key)] = {
            "input_tokens": _parse_int(
                value.get("input_tokens", value.get("total_input_tokens"))
            ),
            "output_tokens": _parse_int(
                value.get("output_tokens", value.get("total_output_tokens"))
            ),
            "total_tokens": _parse_int(value.get("total_tokens")),
            "count": _parse_int(value.get("count", value.get("invocation_count"))),
        }
    return result


def normalise_token_stats(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Return token stats in the shape produced by TokenTracker.get_stats()."""
    if not raw:
        return _empty_token_stats()

    input_tokens = _parse_int(raw.get("total_input_tokens", raw.get("input_tokens")))
    output_tokens = _parse_int(raw.get("total_output_tokens", raw.get("output_tokens")))
    total_tokens = _parse_int(raw.get("total_tokens"))
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens

    return {
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "invocation_count": _parse_int(
            raw.get("invocation_count", raw.get("count", raw.get("calls")))
        ),
        "by_model": _normalise_breakdown(raw.get("by_model", {})),
        "by_type": _normalise_breakdown(raw.get("by_type", {})),
    }


def aggregate_token_stats(stats_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple token stat dictionaries."""
    total = _empty_token_stats()

    for raw_stats in stats_items:
        stats = normalise_token_stats(raw_stats)
        total["total_input_tokens"] += stats["total_input_tokens"]
        total["total_output_tokens"] += stats["total_output_tokens"]
        total["total_tokens"] += stats["total_tokens"]
        total["invocation_count"] += stats["invocation_count"]
        _merge_breakdown(total["by_model"], stats.get("by_model", {}))
        _merge_breakdown(total["by_type"], stats.get("by_type", {}))

    return total


def _merge_breakdown(
    target: dict[str, dict[str, int]], source: dict[str, dict[str, int]]
) -> None:
    for key, stats in source.items():
        bucket = target.setdefault(
            key,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "count": 0,
            },
        )
        bucket["input_tokens"] += _parse_int(stats.get("input_tokens"))
        bucket["output_tokens"] += _parse_int(stats.get("output_tokens"))
        bucket["total_tokens"] += _parse_int(stats.get("total_tokens"))
        bucket["count"] += _parse_int(stats.get("count"))


def _token_usage_path_from_working_dir(working_dir: Path) -> Path:
    return working_dir / "predictions" / "token_usage.json"


def _prediction_report_path_from_working_dir(working_dir: Path) -> Path:
    return working_dir / "predictions" / "prediction_report.html"


def _load_token_usage_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except OSError:
        return None
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse token usage JSON %s: %s", path, exc)
        return None

    if isinstance(payload, dict) and isinstance(payload.get("token_usage"), dict):
        return normalise_token_stats(payload["token_usage"])
    if isinstance(payload, dict):
        return normalise_token_stats(payload)
    return None


def _metric_after_label(items: list[str], label: str) -> int:
    for index, item in enumerate(items[:-1]):
        if item == label:
            return _parse_int(items[index + 1])
    return 0


def _parse_prediction_report_html(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    parser = _ReportTextParser()
    parser.feed(text)

    total_tokens = _metric_after_label(parser.text_items, "Total Tokens")
    input_tokens = _metric_after_label(parser.text_items, "Input Tokens")
    output_tokens = _metric_after_label(parser.text_items, "Output Tokens")
    invocation_count = _metric_after_label(parser.text_items, "VLM Calls")

    if not any((total_tokens, input_tokens, output_tokens, invocation_count)):
        logger.warning(
            "Prediction report %s did not contain parseable token usage metrics",
            path,
        )
        return None

    by_model: dict[str, dict[str, int]] = {}
    for row in parser.rows:
        if len(row) != 5 or row[0] == "Model":
            continue
        calls = _parse_int(row[1])
        row_input = _parse_int(row[2])
        row_output = _parse_int(row[3])
        row_total = _parse_int(row[4])
        if not any((calls, row_input, row_output, row_total)):
            continue
        by_model[row[0]] = {
            "count": calls,
            "input_tokens": row_input,
            "output_tokens": row_output,
            "total_tokens": row_total,
        }

    by_type = {
        "vlm": {
            "count": invocation_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
    }

    return normalise_token_stats(
        {
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "invocation_count": invocation_count,
            "by_model": by_model,
            "by_type": by_type,
        }
    )


def _find_member_token_stats(
    member: SubAsset | PayloadGroup,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    working_dir = getattr(member, "working_dir", None)
    candidate_dirs: list[Path] = []
    if working_dir:
        candidate_dirs.append(Path(working_dir))

    predictions_path = getattr(member, "predictions_path", None)
    if predictions_path:
        pred_path = Path(predictions_path)
        if pred_path.parent.name == "predictions":
            candidate_dirs.append(pred_path.parent.parent)
        elif (
            pred_path.parent.name == "restored"
            and pred_path.parent.parent.name == "predictions"
        ):
            candidate_dirs.append(pred_path.parent.parent.parent)

    seen: set[Path] = set()
    for candidate_dir in candidate_dirs:
        if candidate_dir in seen:
            continue
        seen.add(candidate_dir)

        token_path = _token_usage_path_from_working_dir(candidate_dir)
        stats = _load_token_usage_json(token_path)
        if stats is not None:
            return stats, "token_usage_json", str(token_path)

        report_path = _prediction_report_path_from_working_dir(candidate_dir)
        stats = _parse_prediction_report_html(report_path)
        if stats is not None:
            return stats, "prediction_report_html", str(report_path)

    return None, None, None


def _member_name(member: SubAsset | PayloadGroup) -> str:
    return member.name if isinstance(member, SubAsset) else member.group_name


def _member_status_counts(
    members: list[SubAsset] | list[PayloadGroup],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for member in members:
        counts[member.status] = counts.get(member.status, 0) + 1
    return counts


def collect_scene_stats(
    manifest: SceneManifest,
    working_dir: Path,
    output_usd_path: Path | None = None,
    scene_operation_token_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable scene statistics report."""
    asset_reports: list[dict[str, Any]] = []
    missing_token_usage: list[dict[str, str]] = []
    asset_token_stats: list[dict[str, Any]] = []

    members: list[tuple[str, SubAsset | PayloadGroup]] = [
        *[("sub_asset", sa) for sa in manifest.sub_assets],
        *[("payload_group", pg) for pg in manifest.payload_groups],
    ]

    for kind, member in members:
        stats, source, source_path = _find_member_token_stats(member)
        token_stats = normalise_token_stats(stats)
        if stats is not None:
            asset_token_stats.append(token_stats)
        elif member.status == "completed":
            missing_token_usage.append(
                {
                    "kind": kind,
                    "id": member.id,
                    "name": _member_name(member),
                }
            )

        asset_reports.append(
            {
                "kind": kind,
                "id": member.id,
                "name": _member_name(member),
                "status": member.status,
                "working_dir": getattr(member, "working_dir", None),
                "predictions_path": getattr(member, "predictions_path", None),
                "token_usage_source": source,
                "token_usage_source_path": source_path,
                "token_usage": token_stats,
            }
        )

    asset_predict = aggregate_token_stats(asset_token_stats)
    scene_operations = normalise_token_stats(scene_operation_token_stats)
    total = aggregate_token_stats([asset_predict, scene_operations])

    output_usd = output_usd_path or working_dir / "output" / "composed_scene.usd"
    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "scene_usd_path": manifest.scene_usd_path,
        "working_dir": str(working_dir),
        "output_usd_path": str(output_usd),
        "asset_counts": {
            "sub_assets": len(manifest.sub_assets),
            "payload_groups": len(manifest.payload_groups),
            "sub_asset_status": _member_status_counts(manifest.sub_assets),
            "payload_group_status": _member_status_counts(manifest.payload_groups),
        },
        "token_usage": {
            "total": total,
            "asset_predict": asset_predict,
            "scene_operations": scene_operations,
        },
        "assets": asset_reports,
        "missing_token_usage": missing_token_usage,
        "notes": [
            "asset_predict aggregates per-asset predict-step token usage.",
            "scene_operations aggregates scene-level LLM calls tracked during this run.",
            "Older runs may recover asset_predict usage from prediction_report.html.",
        ],
    }


def write_scene_stats_report(
    manifest: SceneManifest,
    working_dir: Path,
    output_dir: Path | None = None,
    output_usd_path: Path | None = None,
    scene_operation_token_stats: dict[str, Any] | None = None,
) -> Path:
    """Write the final scene statistics report and return its path."""
    output_dir = output_dir or working_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "scene_stats_report.json"
    report = collect_scene_stats(
        manifest=manifest,
        working_dir=working_dir,
        output_usd_path=output_usd_path or output_dir / "composed_scene.usd",
        scene_operation_token_stats=scene_operation_token_stats,
    )
    tmp_path = report_path.with_name(f".{report_path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    tmp_path.replace(report_path)
    logger.info("Scene stats report written to %s", report_path)
    return report_path


def record_model_response_usage(
    token_tracker: Any | None,
    response: Any,
    model_name: str | None,
    invocation_type: str,
) -> None:
    """Record usage metadata from a LangChain model response if available."""
    if token_tracker is None:
        return

    from world_understanding.utils.token_tracking import TokenUsage

    usage = TokenUsage.from_langchain_response(
        response,
        model_name=model_name,
        invocation_type=invocation_type,
    )
    token_tracker.add_usage(usage)
