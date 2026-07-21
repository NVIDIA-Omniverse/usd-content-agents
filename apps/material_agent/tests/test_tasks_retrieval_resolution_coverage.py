# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional coverage for material retrieval, resolution, and prediction saving."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from world_understanding.utils.object_store import InMemoryObjectStore

import material_agent.tasks.material_retrieval as retrieval_module
from material_agent.materials import FALLBACK_MATERIAL_BINDING, FALLBACK_MATERIAL_NAME
from material_agent.tasks.material_retrieval import MaterialRetrievalTask
from material_agent.tasks.predictions import SavePredictionsTask
from material_agent.tasks.resolve_materials import ResolveMaterialFilesTask


class _FakeLLM:
    def __init__(self, *responses: str | Exception) -> None:
        self._responses = list(responses)

    def invoke(self, messages):
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=response)


class _FakeUSDSearchClient:
    instances: list[_FakeUSDSearchClient] = []

    def __init__(self, host: str | None = None) -> None:
        self.host = host
        self.calls: list[dict[str, object]] = []
        self.instances.append(self)

    def search(self, **kwargs):
        self.calls.append(kwargs)
        query = kwargs["query"]
        if query == "Raises":
            raise RuntimeError("search failed")
        if query == "Empty":
            return []
        return [
            {
                "source": {
                    "path": f"/materials/{query}.mdl",
                    "base_key": f"s3://assets/{query}.mdl",
                },
                "metadata": {
                    "dependencies": ["dep.usd"],
                    "resources": {"normal": "normal.png"},
                    "textures": "albedo.png",
                },
            }
        ]


def _listener() -> MagicMock:
    listener = MagicMock()
    listener.event = MagicMock()
    return listener


def test_material_retrieval_handles_empty_and_legacy_direct_mappings() -> None:
    listener = _listener()
    task = MaterialRetrievalTask()

    empty = task.run({"unique_materials": []})
    assert empty["search_stats"] == {
        "total_queries": 0,
        "total_matches": 0,
        "failed_queries": 0,
    }

    context: dict[str, object] = {}
    result = task._use_materials_mapping(
        context,
        ["Steel", "Missing"],
        [{"Steel": "s3://bucket-name/path/to/steel.mdl"}, "ignored"],
        listener,
    )

    steel = result["matched_materials"]["Steel"][0]
    assert steel["source_path"] == "/path/to/steel.mdl"
    assert steel["s3_path"] == "s3://bucket-name/path/to/steel.mdl"
    assert result["unresolved_materials"] == ["Missing"]
    assert result["search_stats"] == {
        "total_queries": 2,
        "total_matches": 1,
        "failed_queries": 1,
    }


def test_material_retrieval_library_mapping_success_and_missing(
    tmp_path: Path,
) -> None:
    listener = _listener()
    library_path = tmp_path / "materials.usda"
    library_path.write_text("#usda 1.0\n", encoding="utf-8")

    result = MaterialRetrievalTask()._use_materials_mapping(
        {},
        ["Glass", "Missing"],
        {
            "material_library_path": str(library_path),
            "Glass": "/World/Looks/Glass",
        },
        listener,
    )

    assert result["is_library_based_mapping"] is True
    assert result["matched_materials"]["Glass"][0]["metadata"] == {
        "source": "materials_library",
        "library_path": str(library_path),
        "is_library_material": True,
    }
    assert result["unresolved_materials"] == ["Missing"]


def test_material_retrieval_library_mapping_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Material library file not found"):
        MaterialRetrievalTask()._use_library_based_mapping(
            {},
            ["Glass"],
            {"Glass": "/World/Looks/Glass"},
            str(tmp_path / "missing.usda"),
            _listener(),
        )


def test_material_retrieval_standard_search_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _listener()
    _FakeUSDSearchClient.instances = []
    monkeypatch.setattr(retrieval_module, "USDSearchClient", _FakeUSDSearchClient)

    with patch(
        "material_agent.tasks.material_retrieval.get_listener",
        return_value=listener,
    ):
        result = MaterialRetrievalTask().run(
            {
                "unique_materials": [
                    "__UNKNOWN__",
                    "Steel",
                    "Empty",
                    "Raises",
                ],
                "unknown_material_predictions": "bad",
                "usd_search_config": {
                    "limit": 3,
                    "host": "search-host",
                    "file_extension_include": ["mdl", "usd"],
                },
            }
        )

    assert _FakeUSDSearchClient.instances[0].host == "search-host"
    assert result["unknown_material_predictions"] == 1
    assert result["matched_materials"][FALLBACK_MATERIAL_NAME][0] == {
        "source_path": FALLBACK_MATERIAL_BINDING,
        "s3_path": None,
        "dependencies": [],
        "metadata": {"source": "canonical_fallback"},
    }
    assert result["matched_materials"]["Steel"][0]["dependencies"] == [
        "dep.usd",
        "normal.png",
        "albedo.png",
    ]
    assert result["matched_materials"]["Raises"] == []
    assert result["search_stats"]["failed_queries"] == 1
    assert "Empty" in result["unresolved_materials"]


def test_material_retrieval_run_uses_llm_enhanced_and_debug_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _listener()
    monkeypatch.setattr(retrieval_module, "USDSearchClient", _FakeUSDSearchClient)

    def fake_create_llm(self, llm_config):
        return _FakeLLM("{}")

    def fake_llm_enhanced_retrieval(
        self, material, client, llm, limit, file_extensions, listener
    ):
        return [{"source_path": f"/llm/{material}.mdl"}], [{"id": material}]

    monkeypatch.setattr(MaterialRetrievalTask, "_create_llm", fake_create_llm)
    monkeypatch.setattr(
        MaterialRetrievalTask,
        "_llm_enhanced_retrieval",
        fake_llm_enhanced_retrieval,
    )

    with patch(
        "material_agent.tasks.material_retrieval.get_listener",
        return_value=listener,
    ):
        result = MaterialRetrievalTask().run(
            {
                "unique_materials": ["Steel"],
                "llm_config": {"service": "nim"},
                "usd_search_config": {"use_llm_enhanced_search": True},
            }
        )

    assert result["matched_materials"]["Steel"] == [{"source_path": "/llm/Steel.mdl"}]

    old_level = retrieval_module.logger.level
    retrieval_module.logger.setLevel(logging.DEBUG)
    try:
        with patch(
            "material_agent.tasks.material_retrieval.get_listener",
            return_value=listener,
        ):
            MaterialRetrievalTask().run({"unique_materials": ["Steel"]})
        MaterialRetrievalTask()._extract_path_from_result(
            {"metadata": {"path": "/debug.mdl"}},
            listener,
        )
        MaterialRetrievalTask()._log_retrieval_summary(
            {
                "Steel": [
                    {"source_path": f"/mat/{index}.mdl", "s3_path": None}
                    for index in range(6)
                ]
            },
            listener,
        )
    finally:
        retrieval_module.logger.setLevel(old_level)


def test_material_retrieval_llm_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    listener = _listener()
    task = MaterialRetrievalTask()
    created: dict[str, object] = {}

    def fake_create_chat_model(**kwargs):
        created.update(kwargs)
        return "llm"

    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
    monkeypatch.setattr(retrieval_module, "create_chat_model", fake_create_chat_model)
    assert task._create_llm({"service": "nim", "model": "m"}) == "llm"
    assert created["api_key"] == "env-key"

    parsed = task._parse_material_with_llm(
        "brushed black steel",
        _FakeLLM('prefix {"material": "steel", "color": "black", "finish": "brushed"}'),
        listener,
    )
    assert parsed == {"material": "steel", "color": "black", "finish": "brushed"}
    assert task._parse_material_with_llm(
        "mystery",
        _FakeLLM('{"color": "blue"}'),
        listener,
    ) == {"material": "mystery", "color": "none", "finish": "none"}
    assert task._parse_material_with_llm(
        "error",
        _FakeLLM(RuntimeError("boom")),
        listener,
    ) == {"material": "error", "color": "none", "finish": "none"}

    results = [
        {"source": {"path": "/candidate/a.mdl"}},
        {"id": "/candidate/b.mdl"},
    ]
    assert (
        task._select_best_match_with_llm(
            "black steel",
            parsed,
            results,
            _FakeLLM('{"best_match_index": 1, "reasoning": "closer"}'),
            listener,
        )
        == 1
    )
    assert (
        task._select_best_match_with_llm(
            "black steel",
            parsed,
            results,
            _FakeLLM('{"best_match_index": null, "reasoning": "none"}'),
            listener,
        )
        is None
    )
    assert (
        task._select_best_match_with_llm(
            "black steel",
            parsed,
            results,
            _FakeLLM("not json"),
            listener,
        )
        == 0
    )
    assert (
        task._select_best_match_with_llm(
            "black steel",
            parsed,
            results,
            _FakeLLM(RuntimeError("bad")),
            listener,
        )
        == 0
    )
    assert (
        task._select_best_match_with_llm(
            "black steel",
            parsed,
            [],
            _FakeLLM("{}"),
            listener,
        )
        is None
    )


def test_material_retrieval_llm_enhanced_paths() -> None:
    listener = _listener()
    task = MaterialRetrievalTask()
    client = _FakeUSDSearchClient()

    paths, results = task._llm_enhanced_retrieval(
        "black steel",
        client,
        _FakeLLM(
            '{"material": "Steel", "color": "black", "finish": "matte"}',
            '{"best_match_index": null, "reasoning": "weak"}',
        ),
        limit=2,
        file_extensions=["mdl"],
        listener=listener,
    )
    assert paths[0]["source_path"] == "/materials/Steel.mdl"
    assert results[0]["source"]["base_key"] == "s3://assets/Steel.mdl"
    assert client.calls[0]["limit"] == 10

    empty_paths, empty_results = task._llm_enhanced_retrieval(
        "empty",
        client,
        _FakeLLM('{"material": "Empty", "color": "none", "finish": "none"}'),
        limit=20,
        file_extensions=["mdl"],
        listener=listener,
    )
    assert empty_paths == []
    assert empty_results == []


def test_material_retrieval_extracts_paths_from_many_result_shapes() -> None:
    listener = _listener()
    task = MaterialRetrievalTask()

    assert (
        task._extract_path_from_result(
            {"path": "/direct.mdl"},
            listener,
        )["source_path"]
        == "/direct.mdl"
    )
    assert (
        task._extract_path_from_result(
            {"metadata": {"uri": "s3://metadata/path.mdl"}},
            listener,
        )["source_path"]
        == "s3://metadata/path.mdl"
    )
    assert (
        task._extract_path_from_result(
            {"data": {"location": "/nested/data.mdl"}},
            listener,
        )["source_path"]
        == "/nested/data.mdl"
    )
    assert (
        task._extract_path_from_result(
            {"document": {"url": "https://example.com/mat.mdl"}},
            listener,
        )["source_path"]
        == "https://example.com/mat.mdl"
    )
    assert (
        task._extract_path_from_result(
            {"item": {"file_path": "/nested/item.mdl"}},
            listener,
        )["source_path"]
        == "/nested/item.mdl"
    )
    assert (
        task._extract_path_from_result(
            {"no_paths": True},
            listener,
        )["source_path"]
        is None
    )

    with patch.object(
        task,
        "_extract_path_from_result",
        side_effect=[RuntimeError("bad"), {"source_path": "/ok.mdl"}],
    ):
        assert task._extract_paths_from_results([{}, {}], listener) == [
            {"source_path": "/ok.mdl"}
        ]


def test_resolve_material_files_handles_empty_and_path_variants() -> None:
    listener = _listener()
    context = {
        "matched_materials": {
            "S3": [{"s3_path": "s3://bucket/path/mat.mdl"}],
            "Local": [{"source_path": "/local/mat.mdl"}],
            "Empty": [],
            "Weird": ["not a dict"],
            "Missing": [{"dependencies": []}],
        }
    }

    with (
        patch(
            "material_agent.tasks.resolve_materials.get_listener",
            return_value=listener,
        ),
        patch("material_agent.tasks.resolve_materials.WU_S3_BUCKET", "special"),
        patch(
            "material_agent.tasks.resolve_materials.WU_S3_REGION",
            "us-west-2",
        ),
    ):
        context["matched_materials"]["Regional"] = [
            {"s3_path": "s3://special-assets/path/regional.mdl"}
        ]
        result = ResolveMaterialFilesTask().run(context)

    assert result["resolved_materials"]["S3"] == (
        "https://bucket.s3.amazonaws.com/path/mat.mdl"
    )
    assert result["resolved_materials"]["Regional"] == (
        "https://special-assets.s3.us-west-2.amazonaws.com/path/regional.mdl"
    )
    assert result["resolved_materials"]["Local"] == "/local/mat.mdl"
    assert result["download_stats"] == {"resolved": 3, "failed": 1, "skipped": 2}

    empty = ResolveMaterialFilesTask().run({"matched_materials": {}})
    assert empty["download_stats"] == {"resolved": 0, "failed": 0, "skipped": 0}


def test_resolve_material_files_handles_library_mappings() -> None:
    listener = _listener()
    context = {
        "is_library_based_mapping": True,
        "material_library_path": "/tmp/library.usda",
        "matched_materials": {
            "Glass": [{"source_path": "/World/Looks/Glass"}],
            "Empty": [],
            "Weird": ["bad"],
            "Missing": [{"s3_path": None}],
        },
    }

    with patch(
        "material_agent.tasks.resolve_materials.get_listener",
        return_value=listener,
    ):
        result = ResolveMaterialFilesTask().run(context)

    assert result["resolved_materials"] == {"Glass": "/World/Looks/Glass"}
    assert result["download_stats"] == {"resolved": 1, "failed": 1, "skipped": 2}


def test_save_predictions_writes_and_enriches_streamed_predictions(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset = [
        {"id": "/A", "ground_truth": {"material": "Steel"}},
        {"id": "/B", "ground_truth": {"material": "Rubber"}},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in dataset) + "\n",
        encoding="utf-8",
    )

    store = InMemoryObjectStore()
    store.set(
        "predictions",
        [
            {
                "id": "/A",
                "vlm_response": {"material": "Steel"},
                "image_path": "a.png",
                "confidence": 0.9,
            },
            {
                "id": "/B",
                "vlm_response": {"material": "Rubber"},
                "images": ["b.png"],
            },
        ],
    )
    store.set("dataset", dataset)
    result = SavePredictionsTask(include_ground_truth=True).run(
        {"dataset_path": str(dataset_path), "output_dir": tmp_path / "out"},
        store,
    )

    output_path = Path(result["predictions_path"])
    saved = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert saved[0]["ground_truth"] == {"material": "Steel"}
    assert saved[0]["confidence"] == 0.9
    assert saved[1]["images"] == ["b.png"]

    streamed = tmp_path / "streamed.jsonl"
    streamed.write_text(
        json.dumps({"id": "/A", "materials": {"material": "Steel"}}) + "\n\nnot-json\n",
        encoding="utf-8",
    )
    streamed_store = InMemoryObjectStore()
    streamed_store.set("dataset", dataset)
    streamed_result = SavePredictionsTask(include_ground_truth=True).run(
        {
            "predictions_path": str(streamed),
            "output_dir": tmp_path / "streamed-out",
            "stream_predictions": True,
        },
        streamed_store,
    )

    assert streamed_result["predictions_path"] == str(streamed)
    enriched = [
        json.loads(line) for line in streamed.read_text(encoding="utf-8").splitlines()
    ]
    assert enriched == [
        {
            "id": "/A",
            "materials": {"material": "Steel"},
            "ground_truth": {"material": "Steel"},
        }
    ]

    file_dataset_store = InMemoryObjectStore()
    file_dataset_store.set(
        "predictions",
        [{"id": "/A", "vlm_response": {"material": "Steel"}}],
    )
    file_dataset_result = SavePredictionsTask(include_ground_truth=True).run(
        {
            "dataset_path": str(dataset_path),
            "output_dir": tmp_path / "file-dataset-out",
        },
        file_dataset_store,
    )
    assert Path(file_dataset_result["predictions_path"]).exists()

    streamed_from_file = tmp_path / "streamed_from_file.jsonl"
    streamed_from_file.write_text(
        json.dumps({"id": "/A", "materials": {"material": "Steel"}}) + "\n",
        encoding="utf-8",
    )
    SavePredictionsTask(include_ground_truth=True).run(
        {
            "dataset_path": str(dataset_path),
            "predictions_path": str(streamed_from_file),
            "output_dir": tmp_path / "streamed-file-out",
            "stream_predictions": True,
        },
        InMemoryObjectStore(),
    )
    assert "ground_truth" in streamed_from_file.read_text(encoding="utf-8")


def test_save_predictions_uses_dataset_default_output_and_errors(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "data" / "dataset.jsonl"
    dataset_path.parent.mkdir()
    dataset_path.write_text(json.dumps({"id": "/A"}) + "\n", encoding="utf-8")

    store = InMemoryObjectStore()
    store.set("predictions", [{"id": "/A", "vlm_response": {"material": "Steel"}}])

    result = SavePredictionsTask().run({"dataset_path": str(dataset_path)}, store)
    assert Path(result["predictions_path"]).parent == dataset_path.parent / "output"

    with pytest.raises(ValueError, match="output_dir not provided"):
        SavePredictionsTask().run({}, store)

    with pytest.raises(ValueError, match="No predictions found"):
        SavePredictionsTask(output_dir=tmp_path / "out").run({}, InMemoryObjectStore())
