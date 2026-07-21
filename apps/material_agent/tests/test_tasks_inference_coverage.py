# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional branch coverage for material-agent inference tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from world_understanding.utils.object_store import InMemoryObjectStore

import material_agent.tasks.inference as inference_module
from material_agent.tasks.inference import (
    VLMInferenceTask,
    _merge_token_usage_stats,
)


def test_token_usage_merging_tolerates_bad_bucket_values() -> None:
    merged = _merge_token_usage_stats(
        {
            "total_input_tokens": "bad",
            "total_output_tokens": 2,
            "total_tokens": None,
            "invocation_count": 1,
            "by_model": {"vlm": {"input_tokens": "bad", "count": "2"}},
            "by_type": "not-a-dict",
            "all_usages": "not-a-list",
        },
        {
            "total_input_tokens": 3,
            "total_output_tokens": "4",
            "total_tokens": "bad",
            "invocation_count": "bad",
            "by_model": {
                "vlm": {
                    "input_tokens": 5,
                    "output_tokens": 6,
                    "total_tokens": 11,
                    "count": 1,
                },
                "ignored": "not-a-dict",
            },
            "by_type": {"vlm": {"input_tokens": 1}},
            "all_usages": [{"model": "vlm"}],
        },
    )

    assert merged["total_input_tokens"] == 3
    assert merged["total_output_tokens"] == 6
    assert merged["total_tokens"] == 0
    assert merged["invocation_count"] == 1
    assert merged["by_model"]["vlm"]["input_tokens"] == 5
    assert merged["by_model"]["vlm"]["output_tokens"] == 6
    assert merged["by_model"]["vlm"]["total_tokens"] == 11
    assert merged["by_model"]["vlm"]["count"] == 3
    assert merged["by_type"]["vlm"]["input_tokens"] == 1
    assert merged["all_usages"] == [{"model": "vlm"}]


def test_selective_reprediction_attaches_visual_context_and_carries_forward() -> None:
    dataset = [
        {"id": "/Resolved", "text": "old", "images": ["resolved.png"]},
        {
            "id": "/Feedback",
            "text": "base prompt",
            "media": "not-a-dict",
        },
        {"id": "/Carried", "text": "keep"},
        {"id": "/New", "text": "new"},
    ]
    prev_preds = {
        "/Resolved": {"id": "/Resolved", "materials": {"material": "Old"}},
        "/Carried": {"id": "/Carried", "materials": {"material": "Steel"}},
    }

    carried, repredict, resolved_count = (
        VLMInferenceTask._classify_entries_for_selective_reprediction(
            dataset,
            resolved_assignments={"/Resolved": "Rubber"},
            prim_feedback={"/Feedback": "Use the highlighted crop."},
            prev_preds=prev_preds,
            visual_refinement_context_by_prim={
                "/Feedback": {
                    "text": "The crop shows a dark rubber gasket.",
                    "images": [
                        {"path": "crop.png", "caption": "dark gasket crop"},
                        {"path": "crop.png", "caption": "duplicate"},
                        {"path": "", "caption": "empty"},
                        "bad",
                    ],
                }
            },
        )
    )

    assert resolved_count == 1
    assert carried[0]["materials"] == {"material": "Rubber"}
    assert carried[0]["images"] == ["resolved.png"]
    assert carried[1] == prev_preds["/Carried"]
    assert [entry["id"] for entry in repredict] == ["/Feedback", "/New"]
    feedback = repredict[0]
    assert "Use the highlighted crop." in feedback["text"]
    assert "TARGETED VISUAL REFINEMENT EVIDENCE" in feedback["text"]
    assert feedback["media"]["images"] == [
        {
            "path": "crop.png",
            "type": "render",
            "metadata": {
                "render_mode": "visual_refinement",
                "view": "targeted_full_scene_visual_evidence",
                "camera": "full_scene",
                "vlm_prompt": "dark gasket crop",
            },
        }
    ]


def test_attach_visual_refinement_images_handles_existing_and_invalid_media() -> None:
    entry = {
        "media": {
            "images": [
                {"path": "existing.png"},
            ]
        }
    }
    VLMInferenceTask._attach_visual_refinement_images(
        entry,
        {
            "images": [
                {"path": "existing.png", "caption": "duplicate"},
                {"path": "new.png"},
            ]
        },
    )
    assert [image["path"] for image in entry["media"]["images"]] == [
        "existing.png",
        "new.png",
    ]
    assert entry["media"]["images"][1]["metadata"]["vlm_prompt"].startswith(
        "Targeted full-scene"
    )

    untouched = {"media": {"images": "not-a-list"}}
    VLMInferenceTask._attach_visual_refinement_images(untouched, {"images": []})
    assert untouched == {"media": {"images": "not-a-list"}}


def test_multi_prim_prompt_handles_multiple_reference_and_render_ranges(
    tmp_path: Path,
) -> None:
    for name in ["ref_a.png", "ref_b.png", "a.png", "b.png", "c.png"]:
        (tmp_path / name).write_bytes(b"png")
    group = [
        {
            "id": "/World/A",
            "text": "Part A",
            "images": ["ref_a.png", "ref_b.png", "a.png"],
            "image_metadata": [
                {"render_mode": "reference_image", "vlm_prompt": "ref a"},
                {"render_mode": "reference_image", "vlm_prompt": "ref b"},
                {"render_mode": "highlighted", "vlm_prompt": "view a"},
            ],
        },
        {
            "id": "/World/B",
            "text": "Part B",
            "images": ["b.png", str(tmp_path / "c.png")],
            "image_metadata": [
                {"render_mode": "highlighted", "vlm_prompt": "view b"},
                {"render_mode": "highlighted", "vlm_prompt": "view c"},
            ],
        },
    ]

    merged_images, prompts, prim_ids, user_prompt = (
        VLMInferenceTask._build_multi_prim_images_and_prompt(group, tmp_path)
    )

    assert merged_images[:4] == [
        tmp_path / "ref_a.png",
        tmp_path / "ref_b.png",
        tmp_path / "a.png",
        tmp_path / "b.png",
    ]
    assert prompts == [
        "ref a",
        "ref b",
        "[Part: /World/A] view a",
        "[Part: /World/B] view b",
        "[Part: /World/B] view c",
    ]
    assert prim_ids == ["/World/A", "/World/B"]
    assert "Images [0" in user_prompt
    assert "Images [3" in user_prompt


def test_carried_forward_predictions_and_empty_guard(tmp_path: Path) -> None:
    results = VLMInferenceTask._carried_forward_predictions_as_results(
        [
            {"id": "/A", "materials": {"material": "Steel"}},
            {"id": "", "materials": {"material": "Missing Id"}},
            {"id": "/B"},
        ]
    )
    assert results == [
        {
            "id": "/A",
            "vlm_response": {"material": "Steel"},
            "status": "success",
        }
    ]

    VLMInferenceTask._fail_if_predictions_empty(
        0,
        [],
        [],
        tmp_path / "predictions.jsonl",
        allow_empty_predictions=False,
    )
    VLMInferenceTask._fail_if_predictions_empty(
        1,
        [],
        [],
        tmp_path / "predictions.jsonl",
        allow_empty_predictions=True,
    )
    with pytest.raises(RuntimeError, match="1 dataset entry"):
        VLMInferenceTask._fail_if_predictions_empty(
            1,
            [],
            [{"id": "/A", "status": "error"}],
            tmp_path / "predictions.jsonl",
            allow_empty_predictions=False,
        )


def test_run_injects_feedback_without_previous_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps({"id": "/A", "text": "base", "image_path": "a.png"}) + "\n",
        encoding="utf-8",
    )
    listener = MagicMock()
    captured: dict[str, object] = {}

    def fake_batch_assign_materials(**kwargs):
        captured.update(kwargs)
        kwargs["on_progress"]("/A", "done")
        kwargs["on_prediction"](
            "/A",
            {
                "material": "Rubber",
                "confidence": 0.7,
                "original_response": "rubber gasket",
            },
        )
        kwargs["on_result"](
            {
                "id": "/A",
                "vlm_response": {"material": "Rubber", "confidence": 0.7},
                "status": "success",
            },
            kwargs["entries"][0],
        )
        return [
            {
                "id": "/A",
                "vlm_response": {"material": "Rubber", "confidence": 0.7},
                "status": "success",
            }
        ]

    monkeypatch.setattr(
        inference_module, "batch_assign_materials", fake_batch_assign_materials
    )
    monkeypatch.setattr(
        inference_module, "get_listener", lambda *args, **kwargs: listener
    )
    monkeypatch.setattr(
        inference_module,
        "_write_token_usage_artifact",
        lambda *args, **kwargs: tmp_path / "token_usage.json",
    )

    result = VLMInferenceTask(vlm=object()).run(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "previous_prim_feedback": {"/A": "Use rubber."},
            "visual_refinement_context_by_prim": {
                "/A": {
                    "text": "Crop shows black rubber.",
                    "images": [{"path": "crop.png"}],
                }
            },
            "output_dir": tmp_path / "predictions",
            "config": {"system_prompt": "Available materials:\nRubber"},
            "max_workers": "2",
        },
        InMemoryObjectStore(),
    )

    assert result["predictions_count"] == 1
    assert "Use rubber." in captured["entries"][0]["text"]
    assert captured["entries"][0]["media"]["images"][0]["path"] == "crop.png"
    streamed = [
        json.loads(line)
        for line in Path(result["predictions_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert streamed == [
        {
            "id": "/A",
            "materials": {"material": "Rubber", "confidence": 0.7},
            "image_path": "a.png",
            "confidence": 0.7,
        }
    ]


def _write_jsonl(path: Path, entries: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )
    return path


def _zero_token_artifact(*_args: Any, **_kwargs: Any) -> Path | None:
    return None


def test_attach_visual_refinement_images_replaces_non_list_bucket() -> None:
    entry = {"media": {"images": "not-a-list"}}

    VLMInferenceTask._attach_visual_refinement_images(
        entry,
        {"images": [{"path": "crop.png", "caption": "crop"}]},
    )

    assert entry["media"]["images"][0]["path"] == "crop.png"


def test_multi_prim_inference_parallel_retry_streams_deduped_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = VLMInferenceTask()
    listener = MagicMock()
    predictions_path = tmp_path / "predictions.jsonl"
    dataset = [
        {"id": "/A", "text": "a", "images": ["a.png"]},
        {"id": "/B", "text": "b", "images": ["b.png"]},
        {"id": "/B", "text": "duplicate"},
    ]
    captured: dict[str, Any] = {}

    def fake_build(group: list[dict[str, Any]], _image_base_dir: Path):
        return (
            [tmp_path / "missing.png", object()],
            [],
            [e["id"] for e in group],
            "prompt",
        )

    def fake_assign_materials_multi_prim(**_kwargs: Any) -> dict[str, Any]:
        return {}

    def fake_batch_assign_materials(**kwargs: Any) -> list[dict[str, Any]]:
        captured["retry_ids"] = [entry["id"] for entry in kwargs["entries"]]
        results = []
        for entry in kwargs["entries"]:
            result = {
                "id": entry["id"],
                "vlm_response": {"material": "Steel"},
                "status": "success",
            }
            kwargs["on_result"]({"id": entry["id"], "status": "error"}, entry)
            kwargs["on_result"](result, entry)
            results.append(result)
        return results

    monkeypatch.setattr(task, "_build_multi_prim_images_and_prompt", fake_build)
    monkeypatch.setattr(
        inference_module,
        "assign_materials_multi_prim",
        fake_assign_materials_multi_prim,
    )
    monkeypatch.setattr(
        inference_module, "batch_assign_materials", fake_batch_assign_materials
    )

    results = task._run_multi_prim_inference(
        dataset=dataset,
        context={
            "image_base_dir": str(tmp_path),
            "config": {"materials_list": ["Steel", "Rubber"]},
            "max_workers": 2,
        },
        prediction_batch_size=2,
        vlm=object(),
        llm=object(),
        system_prompt=None,
        vlm_invoke_kwargs={},
        max_retries=1,
        predictions_path=predictions_path,
        stream_predictions=True,
        listener=listener,
        token_tracker=MagicMock(),
    )

    streamed = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sorted(captured["retry_ids"]) == ["/A", "/B"]
    assert sorted(result["id"] for result in results) == ["/A", "/B"]
    assert sorted(entry["id"] for entry in streamed) == ["/A", "/B"]
    assert any("images" in entry for entry in streamed)


def test_multi_prim_retry_callback_returns_when_streaming_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = VLMInferenceTask()

    def fake_build(group: list[dict[str, Any]], _image_base_dir: Path):
        return [], [], [entry["id"] for entry in group], "prompt"

    def fake_assign_materials_multi_prim(**_kwargs: Any) -> dict[str, Any]:
        return {}

    def fake_batch_assign_materials(**kwargs: Any) -> list[dict[str, Any]]:
        result = {
            "id": kwargs["entries"][0]["id"],
            "vlm_response": {"material": "Steel"},
            "status": "success",
        }
        kwargs["on_result"](result, kwargs["entries"][0])
        return [result]

    monkeypatch.setattr(task, "_build_multi_prim_images_and_prompt", fake_build)
    monkeypatch.setattr(
        inference_module,
        "assign_materials_multi_prim",
        fake_assign_materials_multi_prim,
    )
    monkeypatch.setattr(
        inference_module, "batch_assign_materials", fake_batch_assign_materials
    )

    results = task._run_multi_prim_inference(
        dataset=[{"id": "/A", "text": "a"}, {"id": "/B", "text": "b"}],
        context={"image_base_dir": str(tmp_path), "config": {}},
        prediction_batch_size=2,
        vlm=object(),
        llm=object(),
        system_prompt=None,
        vlm_invoke_kwargs={},
        max_retries=1,
        predictions_path=tmp_path / "predictions.jsonl",
        stream_predictions=False,
        listener=MagicMock(),
        token_tracker=MagicMock(),
    )

    assert results == [
        {
            "id": "/A",
            "vlm_response": {"material": "Steel"},
            "status": "success",
        }
    ]


def test_multi_prim_inference_invalid_max_workers_uses_sequential_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = VLMInferenceTask()
    listener = MagicMock()

    def fake_build(group: list[dict[str, Any]], _image_base_dir: Path):
        return [], [], [entry["id"] for entry in group], "prompt"

    def fake_assign_materials_multi_prim(**kwargs: Any) -> dict[str, Any]:
        return {prim_id: {"material": "Steel"} for prim_id in kwargs["prim_ids"]}

    monkeypatch.setattr(task, "_build_multi_prim_images_and_prompt", fake_build)
    monkeypatch.setattr(
        inference_module,
        "assign_materials_multi_prim",
        fake_assign_materials_multi_prim,
    )

    results = task._run_multi_prim_inference(
        dataset=[{"id": "/A", "text": "a"}, {"id": "/B", "text": "b"}],
        context={
            "image_base_dir": str(tmp_path),
            "config": {"_materials_formatted": "Steel\nRubber"},
            "max_workers": "bad",
        },
        prediction_batch_size=2,
        vlm=object(),
        llm=object(),
        system_prompt=None,
        vlm_invoke_kwargs={},
        max_retries=1,
        predictions_path=tmp_path / "predictions.jsonl",
        stream_predictions=False,
        listener=listener,
        token_tracker=MagicMock(),
    )

    assert [result["id"] for result in results] == ["/A", "/B"]


def test_run_standard_path_covers_prompt_callbacks_and_stream_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entries = [
        {"id": f"/P{i}", "text": f"part {i}", "images": [f"p{i}.png"]}
        for i in range(10)
    ]
    dataset_path = _write_jsonl(tmp_path / "dataset.jsonl", entries)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "predictions.jsonl").write_text("stale\n", encoding="utf-8")
    listener = MagicMock()

    def fake_batch_assign_materials(**kwargs: Any) -> list[dict[str, Any]]:
        results = []
        for idx, entry in enumerate(kwargs["entries"]):
            kwargs["on_progress"](entry["id"], "done")
            kwargs["on_prediction"](
                entry["id"],
                {
                    "material": "Steel",
                    "confidence": 0.9,
                    "original_response": "steel",
                },
            )
            status = "error" if idx == 0 else "success"
            result = {
                "id": entry["id"],
                "vlm_response": {"material": "Steel", "confidence": 0.9},
                "status": status,
            }
            kwargs["on_result"](result, entry)
            results.append(result)
        return results

    monkeypatch.setattr(
        inference_module, "batch_assign_materials", fake_batch_assign_materials
    )
    monkeypatch.setattr(
        inference_module, "get_listener", lambda *args, **kwargs: listener
    )
    monkeypatch.setattr(
        inference_module,
        "_write_token_usage_artifact",
        _zero_token_artifact,
    )

    store = InMemoryObjectStore()
    result = VLMInferenceTask(vlm=object()).run(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "previous_judge_critique": "The last pass was too broad.",
            "iteration_count": 2,
            "resolved_assignments": {"/Missing": "Steel"},
            "previous_predictions_path": str(tmp_path / "missing_predictions.jsonl"),
            "visual_refinement_context_by_prim": "not-a-dict",
            "config": {"system_prompt": "Base prompt"},
            "max_workers": 4,
        },
        store,
    )

    streamed = [
        json.loads(line)
        for line in Path(result["predictions_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result["predictions_count"] == 9
    assert result["failed_count"] == 1
    assert "Base prompt" in result["actual_system_prompt_used"]
    assert "FEEDBACK FROM PREVIOUS ITERATION" in result["actual_system_prompt_used"]
    assert all(entry["id"] != "stale" for entry in streamed)
    assert streamed[0]["images"] == ["p1.png"]
    assert store.get("predictions")


def test_run_sync_multi_prim_dispatch_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [{"id": "/A"}, {"id": "/B"}],
    )
    listener = MagicMock()

    def fake_multi_prim(
        self: VLMInferenceTask,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": "/A",
                "vlm_response": {"material": "Steel"},
                "status": "success",
            }
        ]

    monkeypatch.setattr(VLMInferenceTask, "_run_multi_prim_inference", fake_multi_prim)
    monkeypatch.setattr(
        inference_module, "get_listener", lambda *args, **kwargs: listener
    )
    monkeypatch.setattr(
        inference_module,
        "_write_token_usage_artifact",
        _zero_token_artifact,
    )

    result = VLMInferenceTask(vlm=object()).run(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "prediction_batch_size": 2,
            "stream_predictions": False,
            "output_dir": tmp_path / "out",
            "config": {},
        }
    )

    assert result["predictions_count"] == 1


def test_run_selective_reprediction_skips_invalid_carried_forward_prediction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [{"id": "/Carry"}, {"id": "/Resolved"}, {"id": "/New"}],
    )
    previous_path = _write_jsonl(
        tmp_path / "previous.jsonl",
        [
            {"id": "/Carry"},
            {"id": "/Resolved", "materials": {"material": "Old"}},
        ],
    )
    listener = MagicMock()

    def fake_batch_assign_materials(**kwargs: Any) -> list[dict[str, Any]]:
        result = {
            "id": "/New",
            "vlm_response": {"material": "Plastic"},
            "status": "success",
        }
        kwargs["on_result"](result, kwargs["entries"][0])
        return [result]

    monkeypatch.setattr(
        inference_module, "batch_assign_materials", fake_batch_assign_materials
    )
    monkeypatch.setattr(
        inference_module, "get_listener", lambda *args, **kwargs: listener
    )
    monkeypatch.setattr(
        inference_module,
        "_write_token_usage_artifact",
        _zero_token_artifact,
    )

    result = VLMInferenceTask(vlm=object()).run(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "previous_predictions_path": str(previous_path),
            "resolved_assignments": {"/Resolved": "Rubber"},
            "previous_judge_critique": "No base prompt branch.",
            "output_dir": tmp_path / "out",
            "config": {},
        }
    )

    streamed = [
        json.loads(line)
        for line in Path(result["predictions_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [entry["id"] for entry in streamed] == ["/Resolved", "/New"]


def test_run_resume_parses_existing_predictions_and_ignores_bad_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _write_jsonl(tmp_path / "dataset.jsonl", [{"id": "/Done"}])
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"id": "/Done", "materials": {"material": "Steel"}}) + "\n{bad\n",
        encoding="utf-8",
    )
    listener = MagicMock()
    captured: dict[str, Any] = {}

    def fake_batch_assign_materials(**kwargs: Any) -> list[dict[str, Any]]:
        captured["processed_ids"] = kwargs["processed_ids"]
        return []

    monkeypatch.setattr(
        inference_module, "batch_assign_materials", fake_batch_assign_materials
    )
    monkeypatch.setattr(
        inference_module, "get_listener", lambda *args, **kwargs: listener
    )
    monkeypatch.setattr(
        inference_module,
        "_write_token_usage_artifact",
        _zero_token_artifact,
    )

    result = VLMInferenceTask(vlm=object()).run(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "predictions_path": str(predictions_path),
            "resume": True,
            "allow_empty_predictions": True,
            "config": {},
        }
    )

    assert captured["processed_ids"] == {"/Done"}
    assert result["predictions_count"] == 0


def test_run_stream_false_includes_carried_forward_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [{"id": "/Carry"}, {"id": "/New", "text": "new"}],
    )
    previous_path = _write_jsonl(
        tmp_path / "previous.jsonl",
        [{"id": "/Carry", "materials": {"material": "Steel"}}],
    )
    listener = MagicMock()

    def fake_batch_assign_materials(**kwargs: Any) -> list[dict[str, Any]]:
        result = {
            "id": "/New",
            "vlm_response": {"material": "Rubber"},
            "status": "success",
        }
        kwargs["on_result"](result, kwargs["entries"][0])
        return [result]

    monkeypatch.setattr(
        inference_module, "batch_assign_materials", fake_batch_assign_materials
    )
    monkeypatch.setattr(
        inference_module, "get_listener", lambda *args, **kwargs: listener
    )
    monkeypatch.setattr(
        inference_module,
        "_write_token_usage_artifact",
        _zero_token_artifact,
    )

    result = VLMInferenceTask(vlm=object()).run(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "previous_predictions_path": str(previous_path),
            "previous_prim_feedback": {"/New": "Predict this one again."},
            "stream_predictions": False,
            "output_dir": tmp_path / "out",
            "config": {},
        }
    )

    assert result["predictions_count"] == 2


@pytest.mark.asyncio
async def test_arun_standard_path_covers_prompt_callbacks_and_stream_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entries = [{"id": "/A0", "text": "part 0", "image_path": "a0.png"}] + [
        {"id": f"/A{i}", "text": f"part {i}", "images": [f"a{i}.png"]}
        for i in range(1, 10)
    ]
    dataset_path = _write_jsonl(tmp_path / "dataset.jsonl", entries)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "predictions.jsonl").write_text("stale\n", encoding="utf-8")
    listener = MagicMock()

    async def fake_async_batch_assign_materials(**kwargs: Any) -> list[dict[str, Any]]:
        results = []
        for entry in kwargs["entries"]:
            kwargs["on_progress"](entry["id"], "done")
            kwargs["on_prediction"](
                entry["id"],
                {
                    "material": "Steel",
                    "confidence": 0.8,
                    "original_response": "steel",
                },
            )
            result = {
                "id": entry["id"],
                "vlm_response": {"material": "Steel", "confidence": 0.8},
                "status": "success",
            }
            if entry["id"] == "/A0":
                kwargs["on_result"]({"id": entry["id"], "status": "error"}, entry)
            kwargs["on_result"](result, entry)
            results.append(result)
        return results

    monkeypatch.setattr(
        inference_module,
        "async_batch_assign_materials",
        fake_async_batch_assign_materials,
    )
    monkeypatch.setattr(
        inference_module, "get_listener", lambda *args, **kwargs: listener
    )
    monkeypatch.setattr(
        inference_module,
        "_write_token_usage_artifact",
        _zero_token_artifact,
    )

    result = await VLMInferenceTask(vlm=object()).arun(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "previous_judge_critique": "Fix only specific mistakes.",
            "iteration_count": 3,
            "resolved_assignments": {"/Missing": "Steel"},
            "previous_predictions_path": str(tmp_path / "missing_predictions.jsonl"),
            "visual_refinement_context_by_prim": "bad",
            "config": {"system_prompt": "Base prompt"},
            "max_workers": 2,
        },
        InMemoryObjectStore(),
    )

    streamed = [
        json.loads(line)
        for line in Path(result["predictions_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result["predictions_count"] == 10
    assert "Base prompt" in result["actual_system_prompt_used"]
    assert streamed[0]["image_path"] == "a0.png"
    assert streamed[0]["confidence"] == 0.8


@pytest.mark.asyncio
async def test_arun_feedback_without_previous_predictions_stream_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [{"id": "/A", "text": "base"}],
    )
    listener = MagicMock()
    captured: dict[str, Any] = {}

    async def fake_async_batch_assign_materials(**kwargs: Any) -> list[dict[str, Any]]:
        captured["entry"] = kwargs["entries"][0]
        result = {
            "id": "/A",
            "vlm_response": {"material": "Rubber"},
            "status": "success",
        }
        kwargs["on_result"](result, kwargs["entries"][0])
        return [result]

    monkeypatch.setattr(
        inference_module,
        "async_batch_assign_materials",
        fake_async_batch_assign_materials,
    )
    monkeypatch.setattr(
        inference_module, "get_listener", lambda *args, **kwargs: listener
    )
    monkeypatch.setattr(
        inference_module,
        "_write_token_usage_artifact",
        _zero_token_artifact,
    )

    result = await VLMInferenceTask(vlm=object()).arun(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "previous_prim_feedback": {"/A": "Use the close crop."},
            "previous_judge_critique": "The previous answer missed one part.",
            "visual_refinement_context_by_prim": {
                "/A": {
                    "text": "The crop is dark rubber.",
                    "images": [{"path": "crop.png"}],
                }
            },
            "stream_predictions": False,
            "output_dir": tmp_path / "out",
            "config": {},
        }
    )

    assert "Use the close crop." in captured["entry"]["text"]
    assert captured["entry"]["media"]["images"][0]["path"] == "crop.png"
    assert "FEEDBACK FROM PREVIOUS ITERATION" in result["actual_system_prompt_used"]
    assert result["predictions_count"] == 1


@pytest.mark.asyncio
async def test_arun_selective_reprediction_skips_invalid_carried_forward_prediction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [{"id": "/Carry"}, {"id": "/Resolved"}, {"id": "/New", "text": "new"}],
    )
    previous_path = _write_jsonl(
        tmp_path / "previous.jsonl",
        [
            {"id": "/Carry"},
            {"id": "/Resolved", "materials": {"material": "Old"}},
        ],
    )
    listener = MagicMock()

    async def fake_async_batch_assign_materials(**kwargs: Any) -> list[dict[str, Any]]:
        result = {
            "id": "/New",
            "vlm_response": {"material": "Plastic"},
            "status": "success",
        }
        kwargs["on_result"](result, kwargs["entries"][0])
        return [result]

    monkeypatch.setattr(
        inference_module,
        "async_batch_assign_materials",
        fake_async_batch_assign_materials,
    )
    monkeypatch.setattr(
        inference_module, "get_listener", lambda *args, **kwargs: listener
    )
    monkeypatch.setattr(
        inference_module,
        "_write_token_usage_artifact",
        _zero_token_artifact,
    )

    result = await VLMInferenceTask(vlm=object()).arun(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "previous_predictions_path": str(previous_path),
            "resolved_assignments": {"/Resolved": "Rubber"},
            "previous_prim_feedback": {"/New": "Predict this one again."},
            "output_dir": tmp_path / "out",
            "config": {},
        }
    )

    streamed = [
        json.loads(line)
        for line in Path(result["predictions_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [entry["id"] for entry in streamed] == ["/Resolved", "/New"]


@pytest.mark.asyncio
async def test_arun_resume_parses_existing_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _write_jsonl(tmp_path / "dataset.jsonl", [{"id": "/Done"}])
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"id": "/Done", "materials": {"material": "Steel"}}) + "\n{bad\n",
        encoding="utf-8",
    )
    listener = MagicMock()
    captured: dict[str, Any] = {}

    async def fake_async_batch_assign_materials(**kwargs: Any) -> list[dict[str, Any]]:
        captured["processed_ids"] = kwargs["processed_ids"]
        return []

    monkeypatch.setattr(
        inference_module,
        "async_batch_assign_materials",
        fake_async_batch_assign_materials,
    )
    monkeypatch.setattr(
        inference_module, "get_listener", lambda *args, **kwargs: listener
    )
    monkeypatch.setattr(
        inference_module,
        "_write_token_usage_artifact",
        _zero_token_artifact,
    )

    result = await VLMInferenceTask(vlm=object()).arun(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "predictions_path": str(predictions_path),
            "resume": True,
            "allow_empty_predictions": True,
            "config": {},
        }
    )

    assert captured["processed_ids"] == {"/Done"}
    assert result["predictions_count"] == 0


@pytest.mark.asyncio
async def test_arun_multi_prim_path_runs_in_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = _write_jsonl(
        tmp_path / "dataset.jsonl",
        [{"id": "/A"}, {"id": "/B"}],
    )
    listener = MagicMock()

    def fake_multi_prim(
        self: VLMInferenceTask,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": "/A",
                "vlm_response": {"material": "Steel"},
                "status": "success",
            }
        ]

    monkeypatch.setattr(
        VLMInferenceTask,
        "_run_multi_prim_inference",
        fake_multi_prim,
    )
    monkeypatch.setattr(
        inference_module, "get_listener", lambda *args, **kwargs: listener
    )
    monkeypatch.setattr(
        inference_module,
        "_write_token_usage_artifact",
        _zero_token_artifact,
    )

    result = await VLMInferenceTask(vlm=object()).arun(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "prediction_batch_size": 2,
            "output_dir": tmp_path / "out",
            "config": {},
            "stream_predictions": False,
        }
    )

    assert result["predictions_count"] == 1
