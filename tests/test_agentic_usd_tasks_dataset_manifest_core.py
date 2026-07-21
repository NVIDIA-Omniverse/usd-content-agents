# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from world_understanding.agentic.usd_tasks import dataset_manifest as dm


class _Listener:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _Store:
    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = values or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


class _FakeUSDModel:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.saved_path: Path | None = None

    def save_json(self, path: Path, *, include_hierarchy: bool, indent: int) -> None:
        if self.fail:
            raise RuntimeError("save failed")
        self.saved_path = path
        path.write_text(
            json.dumps({"include_hierarchy": include_hierarchy, "indent": indent}),
            encoding="utf-8",
        )


def _prim_data() -> list[dict[str, Any]]:
    return [
        {
            "prim_path": "/World/A",
            "images": [
                {"view": "front", "path": "a_front.png"},
                {
                    "view": "side",
                    "path": "a_side.png",
                    "camera": "/Camera",
                    "render_mode": "beauty",
                },
            ],
            "metadata": {"type": "Mesh"},
            "display_color": [0.12345, 0.5, 0.5],
            "material_bindings": [{"material": "/Looks/Red"}],
            "hierarchy": {
                "collections": [{"prim_path": "/World", "name": "assets"}],
            },
            "world_bbox": [[0, 0, 0], [1, 1, 1]],
            "world_bbox_meters": [[0, 0, 0], [0.01, 0.01, 0.01]],
            "relative_metrics": {"volume_ratio": 0.5},
        },
        {
            "prim_path": "/World/B",
            "images": [{"view": "front", "path": "b_front.png"}],
            "metadata": {"type": "Xform"},
            "display_color": [0.12344, 0.5, 0.5],
            "hierarchy": {
                "collections": [{"prim_path": "/World", "name": "assets"}],
            },
        },
    ]


def test_dataset_manifest_run_writes_dataset_and_prims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _Listener()
    monkeypatch.setattr(dm, "get_listener", lambda context, logger_name=None: listener)
    monkeypatch.setattr(dm, "USDModel", _FakeUSDModel)
    fake_model = _FakeUSDModel()
    store = _Store(
        {
            "prim_data": _prim_data(),
            "usd_model": fake_model,
            "stage_up_axis": "Z",
            "meters_per_unit": 0.01,
            "stage_world_bbox": [[0, 0, 0], [1, 2, 3]],
            "stage_world_bbox_meters": [[0, 0, 0], [0.01, 0.02, 0.03]],
        }
    )

    context = dm.USDDatasetManifestTask().run(
        {
            "output_dir": tmp_path,
            "usd_path": "scene.usd",
            "include_display_color_statistics": True,
            "renderer_config": {
                "image_width": 256,
                "image_height": 128,
                "camera_view_type": "side",
                "backend": "warp",
            },
        },
        store,
    )

    dataset = json.loads((tmp_path / "dataset.json").read_text(encoding="utf-8"))
    prims = [
        json.loads(line)
        for line in (tmp_path / "prims.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert context["dataset_path"] == str(tmp_path / "dataset.json")
    assert context["prims_path"] == str(tmp_path / "prims.jsonl")
    assert context["num_prims"] == 2
    assert context["num_images"] == 3
    assert context["usd_model_path"] == str(tmp_path / "usd_model.json")
    assert dataset["usd_model_file"] == "usd_model.json"
    assert dataset["stage_up_axis"] == "Z"
    assert dataset["meters_per_unit"] == 0.01
    assert dataset["stage_world_bbox"] == [[0, 0, 0], [1, 2, 3]]
    assert dataset["stage_world_bbox_meters"] == [[0, 0, 0], [0.01, 0.02, 0.03]]
    assert dataset["render_settings"] == {
        "image_width": 256,
        "image_height": 128,
        "camera_type": "side",
        "backend": "warp",
    }
    assert dataset["statistics"]["display_color_stats"] == {
        "unique_colors": [[0.123, 0.5, 0.5]],
        "total_unique_colors": 1,
        "prims_with_color": 2,
    }
    assert prims[0]["renders"][0]["camera"] == "default"
    assert prims[0]["renders"][0]["render_mode"] == "unknown"
    assert prims[0]["material_bindings"] == [{"material": "/Looks/Red"}]
    assert fake_model.saved_path == tmp_path / "usd_model.json"


def test_dataset_manifest_warns_when_usd_model_export_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _Listener()
    monkeypatch.setattr(dm, "get_listener", lambda context, logger_name=None: listener)
    monkeypatch.setattr(dm, "USDModel", _FakeUSDModel)
    store = _Store({"prim_data": _prim_data(), "usd_model": _FakeUSDModel(fail=True)})

    context = dm.USDDatasetManifestTask().run({"output_dir": tmp_path}, store)
    dataset = json.loads((tmp_path / "dataset.json").read_text(encoding="utf-8"))

    assert "usd_model_path" not in context
    assert "usd_model_file" not in dataset
    assert any("Could not save USD model" in message for message in listener.warnings)


def test_dataset_manifest_raises_when_no_images(tmp_path: Path) -> None:
    store = _Store({"prim_data": [{"prim_path": "/World/A", "images": []}]})

    with pytest.raises(RuntimeError, match="Dataset has 0 images"):
        dm.USDDatasetManifestTask().run({"output_dir": tmp_path}, store)


def test_dataset_manifest_helpers_and_clean_for_json() -> None:
    task = dm.USDDatasetManifestTask()

    entries = task._create_prim_entries(
        [
            {
                "prim_path": "/World/A",
                "images": [],
                "metadata": {"type": ""},
            }
        ]
    )
    assert entries == [
        {"prim_path": "/World/A", "renders": [], "metadata": {"type": ""}}
    ]
    assert task._calculate_statistics(
        entries + [{"prim_path": "/World/B", "renders": []}],
        include_display_color_stats=True,
    ) == {
        "total_prims": 2,
        "total_images": 0,
        "total_collections": 0,
        "type_distribution": {"unknown": 1},
    }
    assert task._create_dataset_json(
        "scene.usd",
        {"total_prims": 0, "total_images": 0},
        {},
        _Store(),
        usd_model_exported=False,
    )["render_settings"] == {
        "image_width": 512,
        "image_height": 512,
        "camera_type": "corner",
        "backend": "remote",
    }

    class _ArrayLike:
        def __array__(self) -> np.ndarray:
            return np.array([4, 5])

    class _BadString:
        def __str__(self) -> str:
            raise TypeError("no string")

    assert task._clean_for_json(None) is None
    assert task._clean_for_json({"a": (1, np.array([2, 3]))}) == {"a": [1, [2, 3]]}
    assert task._clean_for_json(_ArrayLike()) == [4, 5]
    assert task._clean_for_json(_BadString()) is None
