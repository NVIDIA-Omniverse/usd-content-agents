# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from world_understanding.agentic.usd_tasks import restore_usd as ru


class _Listener:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.debugs: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def debug(self, message: str) -> None:
        self.debugs.append(message)


class _Child:
    def __init__(self, path: str, *, is_subset: bool) -> None:
        self.path = path
        self.is_subset = is_subset

    def IsA(self, schema: object) -> bool:
        return self.is_subset and schema == "subset"

    def GetPath(self) -> str:
        return self.path


class _Prim:
    def __init__(
        self, children: list[_Child] | None = None, *, valid: bool = True
    ) -> None:
        self.children = children or []
        self.valid = valid

    def IsValid(self) -> bool:
        return self.valid

    def GetChildren(self) -> list[_Child]:
        return self.children


class _Stage:
    def __init__(self, prims: dict[str, _Prim]) -> None:
        self.prims = prims

    def GetPrimAtPath(self, path: str) -> _Prim | None:
        return self.prims.get(path)


def _write_predictions(path: Path, rows: list[dict[str, Any] | str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, str):
                f.write(row + "\n")
            else:
                f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_restoration_stats_to_dict() -> None:
    stats = ru.RestorationStats(
        total_originals=4,
        identity_count=1,
        dedup_count=1,
        split_count=1,
        split_dedup_count=1,
        predictions_consumed={"/A", "/B"},
        predictions_written=3,
        uncovered_originals=["/Missing"],
        unconsumed_predictions=["/Unused"],
    )

    assert stats.to_dict() == {
        "total_originals": 4,
        "identity_count": 1,
        "dedup_count": 1,
        "split_count": 1,
        "split_dedup_count": 1,
        "predictions_consumed": 2,
        "predictions_written": 3,
        "uncovered_originals": ["/Missing"],
        "unconsumed_predictions": ["/Unused"],
        "restored_prim_sources": {},
        "expected_target_count": 0,
        "mapping_complete": True,
        "mapping_warnings": [],
    }


def test_restore_task_requires_inputs_and_prediction_file(tmp_path: Path) -> None:
    task = ru.RestoreUSDTask()

    with pytest.raises(ValueError, match="original_usd_path is required"):
        task.run({})
    with pytest.raises(ValueError, match="predictions_path is required"):
        task.run({"original_usd_path": "scene.usd"})
    with pytest.raises(ValueError, match="output_predictions_path is required"):
        task.run({"original_usd_path": "scene.usd", "predictions_path": "pred.jsonl"})
    with pytest.raises(FileNotFoundError):
        task.run(
            {
                "original_usd_path": "scene.usd",
                "predictions_path": tmp_path / "missing.jsonl",
                "output_predictions_path": tmp_path / "out.jsonl",
            }
        )


def test_transform_predictions_copies_without_mapping(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "restored.jsonl"
    _write_predictions(
        predictions_path,
        [{"id": "/A", "material": "red"}, "", {"id": "/B", "material": "blue"}],
    )
    listener = _Listener()

    count, stats = ru.RestoreUSDTask()._transform_predictions(
        predictions_path,
        output_path,
        Path("scene.usd"),
        {},
        listener,
    )

    assert count == 2
    assert stats.predictions_written == 2
    assert stats.mapping_complete is False
    assert stats.restored_prim_sources == {}
    assert _read_jsonl(output_path) == [
        {"id": "/A", "material": "red"},
        {"id": "/B", "material": "blue"},
    ]
    assert any("copied as-is" in message for message in listener.warnings)


def test_restore_task_transforms_identity_dedup_split_and_split_dedup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "restored.jsonl"
    _write_predictions(
        predictions_path,
        [
            {"id": "/A", "material": "identity"},
            {"id": "/Proto", "material": "dedup"},
            {"id": "/Part1", "material": "split-a"},
            {"id": "/Part2", "material": "split-b"},
            {"id": "/DedupPart", "material": "split-dedup"},
            {"id": "/Unused", "material": "unused"},
            "",
            "{not json",
            {"material": "missing id"},
        ],
    )
    listener = _Listener()
    monkeypatch.setattr(ru, "get_listener", lambda context: listener)
    task = ru.RestoreUSDTask()
    monkeypatch.setattr(task, "_open_stage", lambda path, listener: object())

    def fake_geomsubsets(
        stage: object, original_path: str, listener: _Listener
    ) -> list[str]:
        return {
            "/C": ["/C/Subset0"],
            "/D": ["/D/Subset0", "/D/Subset1"],
        }.get(original_path, [])

    monkeypatch.setattr(task, "_get_geomsubset_paths", fake_geomsubsets)
    metadata = {
        "correspondence_map": {
            "summary": {"operations_run": {"split": True, "deduplicate": True}},
            "full_mapping": {
                "original_to_prototype": {
                    "/A": ["/A"],
                    "/B": ["/Proto"],
                    "/C": ["/Part1", "/Part2"],
                    "/D": ["/DedupPart", "/DedupPart"],
                    "/E": "/Missing",
                }
            },
            "split_mapping": {"/C": ["/Part1", "/Part2"], "/D": ["/DedupPart"]},
        }
    }

    context = task.run(
        {
            "original_usd_path": "scene.usd",
            "predictions_path": predictions_path,
            "output_predictions_path": output_path,
            "optimization_metadata": metadata,
        }
    )

    rows = _read_jsonl(output_path)
    assert [row["id"] for row in rows] == [
        "/A",
        "/B",
        "/C/Subset0",
        "/C_part_1",
        "/D/Subset0",
        "/D/Subset1",
    ]
    assert context["restore_success"] is True
    assert context["predictions_count"] == 6
    assert context["restore_stats"]["identity_count"] == 1
    assert context["restore_stats"]["dedup_count"] == 1
    assert context["restore_stats"]["split_count"] == 1
    assert context["restore_stats"]["split_dedup_count"] == 1
    assert context["restore_stats"]["uncovered_originals"] == ["/E"]
    assert context["restore_stats"]["unconsumed_predictions"] == ["/Unused"]
    assert context["restore_stats"]["restored_prim_sources"] == {
        "/A": "/A",
        "/B": "/Proto",
        "/C/Subset0": "/Part1",
        "/C_part_1": "/Part2",
        "/D/Subset0": "/DedupPart",
        "/D/Subset1": "/DedupPart",
        "/E": "/Missing",
    }
    assert context["restore_stats"]["expected_target_count"] == 7
    assert context["restore_stats"]["mapping_complete"] is False
    assert any("Skipping invalid JSON" in message for message in listener.warnings)
    assert any("not consumed" in message for message in listener.warnings)
    assert any("not covered" in message for message in listener.warnings)


def test_restore_mapping_is_complete_even_when_prediction_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "restored.jsonl"
    _write_predictions(
        predictions_path,
        [
            {"id": "/Identity", "material": "identity"},
            {"id": "/Shared", "material": "dedup"},
            {"id": "/Split0", "material": "split-a"},
            {"id": "/SplitDedup", "material": "split-dedup"},
        ],
    )
    task = ru.RestoreUSDTask()
    monkeypatch.setattr(task, "_open_stage", lambda path, listener: object())
    monkeypatch.setattr(
        task,
        "_get_geomsubset_paths",
        lambda stage, original_path, listener: {
            "/Split": ["/Split/Left", "/Split/Right"],
            "/SplitDedupOriginal": [
                "/SplitDedupOriginal/Left",
                "/SplitDedupOriginal/Right",
            ],
        }.get(original_path, []),
    )
    metadata = {
        "correspondence_map": {
            "full_mapping": {
                "original_to_prototype": {
                    "/Identity": ["/Identity"],
                    "/DedupA": ["/Shared"],
                    "/DedupB": ["/Shared"],
                    "/Split": ["/Split0", "/Split1"],
                    "/SplitDedupOriginal": ["/SplitDedup", "/SplitDedup"],
                }
            },
            "split_mapping": {
                "/Split": ["/Split0", "/Split1"],
                "/SplitDedupOriginal": ["/SplitPart0", "/SplitPart1"],
            },
        }
    }

    _, stats = task._transform_predictions(
        predictions_path,
        output_path,
        Path("scene.usd"),
        metadata,
        _Listener(),
    )

    assert stats.mapping_complete is True
    assert stats.expected_target_count == 7
    assert stats.restored_prim_sources == {
        "/Identity": "/Identity",
        "/DedupA": "/Shared",
        "/DedupB": "/Shared",
        "/Split/Left": "/Split0",
        "/Split/Right": "/Split1",
        "/SplitDedupOriginal/Left": "/SplitDedup",
        "/SplitDedupOriginal/Right": "/SplitDedup",
    }
    assert "/Split" not in stats.uncovered_originals
    assert {row["id"] for row in _read_jsonl(output_path)} == {
        "/Identity",
        "/DedupA",
        "/DedupB",
        "/Split/Left",
        "/SplitDedupOriginal/Left",
        "/SplitDedupOriginal/Right",
    }


def test_restore_mapping_marks_malformed_correspondence_incomplete(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "restored.jsonl"
    _write_predictions(predictions_path, [{"id": "/A", "material": "red"}])

    _, stats = ru.RestoreUSDTask()._transform_predictions(
        predictions_path,
        output_path,
        Path("scene.usd"),
        {"correspondence_map": {"full_mapping": {"original_to_prototype": []}}},
        _Listener(),
    )

    assert stats.mapping_complete is False
    assert stats.restored_prim_sources == {}
    assert any("original_to_prototype" in warning for warning in stats.mapping_warnings)


@pytest.mark.parametrize(
    "optimization_metadata",
    [
        [],
        {"correspondence_map": []},
        {"correspondence_map": {"full_mapping": []}},
    ],
)
def test_restore_mapping_treats_malformed_metadata_containers_as_unmapped(
    tmp_path: Path,
    optimization_metadata: Any,
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "restored.jsonl"
    prediction = {"id": "/A", "material": "red"}
    _write_predictions(predictions_path, [prediction])

    count, stats = ru.RestoreUSDTask()._transform_predictions(
        predictions_path,
        output_path,
        Path("scene.usd"),
        optimization_metadata,
        _Listener(),
    )

    assert count == 1
    assert stats.mapping_complete is False
    assert _read_jsonl(output_path) == [prediction]
    assert any("original_to_prototype" in warning for warning in stats.mapping_warnings)


def test_restore_mapping_rejects_malformed_paths_and_split_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "restored.jsonl"
    _write_predictions(
        predictions_path,
        [
            {"id": "/A", "material": "a"},
            {"id": "/B", "material": "b"},
            {"id": "/Good", "material": "good"},
        ],
    )
    task = ru.RestoreUSDTask()
    monkeypatch.setattr(task, "_open_stage", lambda path, listener: None)
    metadata = {
        "correspondence_map": {
            "full_mapping": {
                "original_to_prototype": {
                    "relative-original": ["/A"],
                    "/RelativePrototype": ["relative-prototype"],
                    "/NoSplitMetadata": ["/A", "/B"],
                    "/Good": ["/Good"],
                }
            },
            "split_mapping": [],
            "summary": [],
        }
    }

    count, stats = task._transform_predictions(
        predictions_path,
        output_path,
        Path("scene.usd"),
        metadata,
        _Listener(),
    )

    assert count == 3
    assert stats.mapping_complete is False
    assert stats.restored_prim_sources == {
        "/NoSplitMetadata_part_0": "/A",
        "/NoSplitMetadata_part_1": "/B",
        "/Good": "/Good",
    }
    assert {row["id"] for row in _read_jsonl(output_path)} == {
        "/NoSplitMetadata_part_0",
        "/NoSplitMetadata_part_1",
        "/Good",
    }
    assert any(
        "split_mapping was malformed" in warning for warning in stats.mapping_warnings
    )
    assert any(
        "Invalid original prim path" in warning for warning in stats.mapping_warnings
    )
    assert any(
        "Invalid prototype mapping" in warning for warning in stats.mapping_warnings
    )
    assert any("lacked split metadata" in warning for warning in stats.mapping_warnings)


def test_restore_mapping_detects_incomplete_duplicate_target_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "restored.jsonl"
    _write_predictions(
        predictions_path,
        [
            {"id": "/PartA", "material": "a"},
            {"id": "/PartB", "material": "b"},
        ],
    )
    task = ru.RestoreUSDTask()
    monkeypatch.setattr(task, "_open_stage", lambda path, listener: object())
    monkeypatch.setattr(
        task,
        "_get_geomsubset_paths",
        lambda stage, original_path, listener: [
            "/Original/Repeated",
            "/Original/Repeated",
        ],
    )
    metadata = {
        "correspondence_map": {
            "full_mapping": {
                "original_to_prototype": {
                    "/Original": ["/PartA", "/PartB"],
                }
            },
            "split_mapping": {"/Original": ["/PartA", "/PartB"]},
        }
    }

    count, stats = task._transform_predictions(
        predictions_path,
        output_path,
        Path("scene.usd"),
        metadata,
        _Listener(),
    )

    assert count == 1
    assert stats.expected_target_count == 2
    assert stats.restored_prim_sources == {"/Original/Repeated": "/PartA"}
    assert stats.mapping_complete is False
    assert _read_jsonl(output_path) == [{"id": "/Original/Repeated", "material": "a"}]
    assert any(
        "Duplicate restored target" in warning for warning in stats.mapping_warnings
    )
    assert any(
        "did not contain every expected target" in warning
        for warning in stats.mapping_warnings
    )


def test_rejected_restored_mapping_does_not_emit_prediction() -> None:
    task = ru.RestoreUSDTask()
    stats = ru.RestorationStats()
    output = io.StringIO()
    listener = _Listener()

    task._handle_single_prototype(
        "/Original",
        "/PrototypeA",
        {"/PrototypeA": {"id": "/PrototypeA", "material": "Steel"}},
        stats,
        output,
        listener,
    )
    task._handle_single_prototype(
        "/Original",
        "/PrototypeB",
        {"/PrototypeB": {"id": "/PrototypeB", "material": "Wood"}},
        stats,
        output,
        listener,
    )
    task._handle_single_prototype(
        "relative",
        "/PrototypeC",
        {"/PrototypeC": {"id": "/PrototypeC", "material": "Plastic"}},
        stats,
        output,
        listener,
    )

    assert [json.loads(line) for line in output.getvalue().splitlines()] == [
        {"id": "/Original", "material": "Steel"}
    ]
    assert stats.predictions_written == 1
    assert stats.mapping_complete is False
    assert stats.uncovered_originals == ["/Original", "relative"]
    assert any(
        "Duplicate restored target" in warning for warning in stats.mapping_warnings
    )
    assert any(
        "Invalid restored/source" in warning for warning in stats.mapping_warnings
    )


def test_split_rejected_duplicate_target_does_not_emit_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ru.UsdGeom, "Subset", "subset")
    task = ru.RestoreUSDTask()
    stats = ru.RestorationStats()
    output = io.StringIO()

    task._handle_split(
        "/Original",
        ["/PrototypeA", "/PrototypeB"],
        {
            "/PrototypeA": {"id": "/PrototypeA", "material": "Steel"},
            "/PrototypeB": {"id": "/PrototypeB", "material": "Wood"},
        },
        _Stage(
            {
                "/Original": _Prim(
                    [
                        _Child("/Original/Subset", is_subset=True),
                        _Child("/Original/Subset", is_subset=True),
                    ]
                )
            }
        ),
        stats,
        output,
        _Listener(),
    )

    assert [json.loads(line) for line in output.getvalue().splitlines()] == [
        {"id": "/Original/Subset", "material": "Steel"}
    ]
    assert stats.predictions_written == 1
    assert stats.mapping_complete is False
    assert any(
        "Duplicate restored target" in warning for warning in stats.mapping_warnings
    )


def test_restore_task_records_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text('{"id": "/A"}\n', encoding="utf-8")
    task = ru.RestoreUSDTask()
    monkeypatch.setattr(ru, "get_listener", lambda context: _Listener())
    monkeypatch.setattr(
        task,
        "_transform_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    context = {
        "original_usd_path": "scene.usd",
        "predictions_path": predictions_path,
        "output_predictions_path": tmp_path / "out.jsonl",
    }

    with pytest.raises(RuntimeError, match="boom"):
        task.run(context)

    assert context["restore_success"] is False
    assert context["restore_error"] == "boom"


def test_handle_single_and_split_helpers() -> None:
    task = ru.RestoreUSDTask()
    listener = _Listener()
    stats = ru.RestorationStats()
    out = io.StringIO()

    task._handle_single_prototype("/Original", "/Missing", {}, stats, out, listener)
    assert stats.uncovered_originals == ["/Original"]

    task._handle_split(
        "/Split",
        ["/Missing"],
        {},
        None,
        stats,
        out,
        listener,
    )
    assert stats.split_count == 1
    assert stats.uncovered_originals == ["/Original", "/Split"]
    assert any("No prediction found" in message for message in listener.warnings)


def test_open_stage_success_none_and_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    task = ru.RestoreUSDTask()
    listener = _Listener()
    stage = object()

    monkeypatch.setattr(ru.Usd.Stage, "Open", lambda path: stage)
    assert task._open_stage(Path("scene.usd"), listener) is stage

    monkeypatch.setattr(ru.Usd.Stage, "Open", lambda path: None)
    assert task._open_stage(Path("missing.usd"), listener) is None

    monkeypatch.setattr(
        ru.Usd.Stage,
        "Open",
        lambda path: (_ for _ in ()).throw(RuntimeError("bad open")),
    )
    assert task._open_stage(Path("bad.usd"), listener) is None
    assert any("Failed to open" in message for message in listener.warnings)
    assert any("Error opening" in message for message in listener.warnings)


def test_get_geomsubset_paths_found_invalid_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ru.UsdGeom, "Subset", "subset")
    task = ru.RestoreUSDTask()
    listener = _Listener()
    stage = _Stage(
        {
            "/Mesh": _Prim(
                [
                    _Child("/Mesh/SubsetA", is_subset=True),
                    _Child("/Mesh/NotSubset", is_subset=False),
                ]
            ),
            "/Invalid": _Prim(valid=False),
        }
    )

    assert task._get_geomsubset_paths(stage, "/Mesh", listener) == ["/Mesh/SubsetA"]
    assert task._get_geomsubset_paths(stage, "/Missing", listener) == []
    assert task._get_geomsubset_paths(stage, "/Invalid", listener) == []
    assert any("Found 1 GeomSubsets" in message for message in listener.debugs)

    class _FailingStage:
        def GetPrimAtPath(self, path: str) -> None:
            raise RuntimeError("bad prim")

    assert task._get_geomsubset_paths(_FailingStage(), "/Boom", listener) == []
    assert any("Error inspecting" in message for message in listener.warnings)
