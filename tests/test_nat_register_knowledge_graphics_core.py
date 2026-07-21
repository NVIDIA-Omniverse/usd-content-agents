# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for NAT knowledge and graphics registration wrappers."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from PIL import Image


class FakeFunctionBaseConfig:
    def __init_subclass__(cls, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.name = name


class FakeFunctionInfo:
    def __init__(self, fn: Any, description: str) -> None:
        self.fn = fn
        self.description = description

    @classmethod
    def from_fn(cls, fn: Any, description: str) -> FakeFunctionInfo:
        return cls(fn, description)


class FakeStore:
    num_documents = 3
    dimension = 8

    def save(self, output_path: str) -> None:
        Path(output_path).write_text("store", encoding="utf-8")


class FakeDocument:
    def __init__(
        self,
        *,
        document_id: str = "doc-1",
        text_content: str = "content",
        image_path: str = "image.png",
    ) -> None:
        self.document_id = document_id
        self.text_content = text_content
        self.image_path = image_path
        self.metadata = {"kind": "demo"}

    def get_content_type(self) -> str:
        return "image" if self.image_path else "text"


class FakeResult:
    def __init__(self, rank: int = 0, score: float = 0.98765) -> None:
        self.rank = rank
        self.score = score
        self.document = FakeDocument()


def _install_fake_nat(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        "nat": types.ModuleType("nat"),
        "nat.builder": types.ModuleType("nat.builder"),
        "nat.builder.builder": types.ModuleType("nat.builder.builder"),
        "nat.builder.function_info": types.ModuleType("nat.builder.function_info"),
        "nat.cli": types.ModuleType("nat.cli"),
        "nat.cli.register_workflow": types.ModuleType("nat.cli.register_workflow"),
        "nat.data_models": types.ModuleType("nat.data_models"),
        "nat.data_models.function": types.ModuleType("nat.data_models.function"),
    }
    modules["nat.builder.builder"].Builder = type("Builder", (), {})
    modules["nat.builder.function_info"].FunctionInfo = FakeFunctionInfo
    modules["nat.cli.register_workflow"].register_function = lambda **_kwargs: (
        lambda fn: fn
    )
    modules["nat.data_models.function"].FunctionBaseConfig = FakeFunctionBaseConfig
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _load_register_module(monkeypatch: pytest.MonkeyPatch, module_name: str) -> Any:
    _install_fake_nat(monkeypatch)
    full_name = f"world_understanding.nat.{module_name}"
    sys.modules.pop(full_name, None)
    module = importlib.import_module(full_name)
    return importlib.reload(module)


async def _function_info(module: Any, function_name: str, config_name: str) -> Any:
    generator = getattr(module, function_name)(getattr(module, config_name)(), object())
    return await anext(generator)


def _set_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> None:
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    monkeypatch.setitem(sys.modules, name, module)


def _raise(exc: Exception) -> Any:
    raise exc


def _capture_return(
    captured: dict[str, Any], key: str, value: Any, **kwargs: Any
) -> Any:
    captured[key] = kwargs
    return value


def test_nat_register_knowledge_success_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_register_module(monkeypatch, "register_knowledge")
    assert module.BuildImageVectorStoreConfig.name == "build_image_vector_store"

    captured: dict[str, Any] = {}
    _set_fake_module(
        monkeypatch,
        "world_understanding.functions.knowledge.image_vector_store",
        build_image_vector_store=lambda **kwargs: _capture_return(
            captured, "build_image", FakeStore(), **kwargs
        ),
        find_similar_images_from_vector_store=lambda **kwargs: _capture_return(
            captured, "find_image", [FakeResult()], **kwargs
        ),
    )
    _set_fake_module(
        monkeypatch,
        "world_understanding.functions.knowledge.text_vector_store",
        build_text_vector_store=lambda **kwargs: _capture_return(
            captured, "build_text", FakeStore(), **kwargs
        ),
        find_similar_texts_from_vector_store=lambda **kwargs: _capture_return(
            captured, "find_text", [FakeResult()], **kwargs
        ),
    )
    _set_fake_module(
        monkeypatch,
        "world_understanding.functions.knowledge.multimodal_vector_store",
        build_multimodal_vector_store=lambda **kwargs: _capture_return(
            captured, "build_multimodal", FakeStore(), **kwargs
        ),
        find_similar_documents_from_vector_store=lambda **kwargs: _capture_return(
            captured, "find_documents", [FakeResult()], **kwargs
        ),
        collect_documents_from_vector_store=lambda **kwargs: _capture_return(
            captured, "collect_documents", [FakeDocument()], **kwargs
        ),
    )

    info = asyncio.run(
        _function_info(
            module, "build_image_vector_store", "BuildImageVectorStoreConfig"
        )
    )
    payload = json.loads(
        asyncio.run(
            info.fn(
                str(tmp_path),
                str(tmp_path / "image.store"),
                image_extensions="jpg,png",
            )
        )
    )
    assert payload["status"] == "success"
    assert captured["build_image"]["image_extensions"] == (".jpg", ".png")

    info = asyncio.run(
        _function_info(module, "find_similar_images", "FindSimilarImagesConfig")
    )
    payload = json.loads(
        asyncio.run(
            info.fn("query.png", "image.store", filter_key="kind", filter_value="demo")
        )
    )
    assert payload["results"][0]["rank"] == 1
    assert payload["results"][0]["similarity_score"] == "0.9877"
    assert captured["find_image"]["filter_metadata"] == {"kind": "demo"}

    info = asyncio.run(
        _function_info(module, "build_text_vector_store", "BuildTextVectorStoreConfig")
    )
    payload = json.loads(
        asyncio.run(
            info.fn(
                text_sources=str(tmp_path),
                image_sources=str(tmp_path),
                output_path=str(tmp_path / "text.store"),
                text_extensions="txt,md",
                image_extensions="jpg",
                vlm_model="vlm",
                vlm_api_key="key",
            )
        )
    )
    assert payload["num_documents"] == 3
    assert captured["build_text"]["vlm_model"] == "vlm"

    info = asyncio.run(
        _function_info(module, "find_similar_texts", "FindSimilarTextsConfig")
    )
    payload = json.loads(
        asyncio.run(
            info.fn("query", "text", "text.store", filter_key="", filter_value="")
        )
    )
    assert payload["results"][0]["text_id"] == "doc-1"
    assert captured["find_text"]["filter_metadata"] is None
    json.loads(
        asyncio.run(
            info.fn(
                "query", "text", "text.store", filter_key="kind", filter_value="demo"
            )
        )
    )
    assert captured["find_text"]["filter_metadata"] == {"kind": "demo"}

    info = asyncio.run(
        _function_info(
            module,
            "build_multimodal_vector_store",
            "BuildMultimodalVectorStoreConfig",
        )
    )
    payload = json.loads(
        asyncio.run(
            info.fn(
                text_sources="",
                image_sources="",
                output_path=str(tmp_path / "multi.store"),
                vlm_model="vlm",
                vlm_api_key="key",
            )
        )
    )
    assert payload["text_sources"] is None
    assert captured["build_multimodal"]["text_source"] is None

    info = asyncio.run(
        _function_info(module, "find_similar_documents", "FindSimilarDocumentsConfig")
    )
    payload = json.loads(
        asyncio.run(
            info.fn(
                "query.png",
                "image",
                "multi.store",
                filter_key="kind",
                filter_value="demo",
                embedding_type="text",
            )
        )
    )
    assert payload["results"][0]["content_type"] == "image"
    assert captured["find_documents"]["embedding_type"] == "text"

    info = asyncio.run(
        _function_info(module, "collect_documents", "CollectDocumentsConfig")
    )
    payload = json.loads(
        asyncio.run(info.fn("multi.store", filter_key="kind", filter_value="demo"))
    )
    assert payload["filter_applied"] is True
    assert payload["documents"][0]["document_id"] == "doc-1"


@pytest.mark.parametrize(
    (
        "function_name",
        "config_name",
        "module_name",
        "attr_name",
        "call_args",
        "prefixes",
    ),
    [
        (
            "build_image_vector_store",
            "BuildImageVectorStoreConfig",
            "world_understanding.functions.knowledge.image_vector_store",
            "build_image_vector_store",
            ("src", "out.store"),
            ["Error:", "Invalid parameter:", "Failed to build vector store:"],
        ),
        (
            "find_similar_images",
            "FindSimilarImagesConfig",
            "world_understanding.functions.knowledge.image_vector_store",
            "find_similar_images_from_vector_store",
            ("query.png", "store"),
            ["Error:", "Invalid parameter:", "Failed to search for similar images:"],
        ),
        (
            "build_text_vector_store",
            "BuildTextVectorStoreConfig",
            "world_understanding.functions.knowledge.text_vector_store",
            "build_text_vector_store",
            (),
            ["Error:", "Invalid parameter:", "Failed to build vector store:"],
        ),
        (
            "find_similar_texts",
            "FindSimilarTextsConfig",
            "world_understanding.functions.knowledge.text_vector_store",
            "find_similar_texts_from_vector_store",
            ("query", "text", "store"),
            ["Error:", "Invalid parameter:", "Failed to search for similar texts:"],
        ),
        (
            "build_multimodal_vector_store",
            "BuildMultimodalVectorStoreConfig",
            "world_understanding.functions.knowledge.multimodal_vector_store",
            "build_multimodal_vector_store",
            (),
            ["Error:", "Invalid parameter:", "Failed to build vector store:"],
        ),
        (
            "find_similar_documents",
            "FindSimilarDocumentsConfig",
            "world_understanding.functions.knowledge.multimodal_vector_store",
            "find_similar_documents_from_vector_store",
            ("query", "text", "store"),
            ["Error:", "Invalid parameter:", "Failed to search for similar documents:"],
        ),
        (
            "collect_documents",
            "CollectDocumentsConfig",
            "world_understanding.functions.knowledge.multimodal_vector_store",
            "collect_documents_from_vector_store",
            ("store",),
            ["Error:", "Invalid parameter:", "Failed to collect documents:"],
        ),
    ],
)
def test_nat_register_knowledge_error_paths(
    monkeypatch: pytest.MonkeyPatch,
    function_name: str,
    config_name: str,
    module_name: str,
    attr_name: str,
    call_args: tuple[Any, ...],
    prefixes: list[str],
) -> None:
    module = _load_register_module(monkeypatch, "register_knowledge")
    for exc, prefix in zip(
        [FileNotFoundError("missing"), ValueError("bad"), RuntimeError("boom")],
        prefixes,
        strict=True,
    ):
        _set_fake_module(
            monkeypatch,
            module_name,
            **{attr_name: lambda _exc=exc, **_kwargs: _raise(_exc)},
        )
        info = asyncio.run(_function_info(module, function_name, config_name))
        assert asyncio.run(info.fn(*call_args)).startswith(prefix)


def test_nat_register_graphics_stage_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_register_module(monkeypatch, "register_graphics")
    assert module.LoadUsdStageConfig.name == "load_usd_stage"

    from world_understanding.utils.usd import stage as stage_utils

    monkeypatch.setattr(stage_utils, "load_stage", lambda path: {"stage": path})
    monkeypatch.setattr(
        stage_utils,
        "get_stage_info",
        lambda _stage: {
            "prim_count": 4,
            "default_prim": "/World",
            "up_axis": "Z",
            "meters_per_unit": 1.0,
            "time_codes_per_second": 24,
            "start_time_code": 0,
            "end_time_code": 1,
        },
    )
    info = asyncio.run(_function_info(module, "load_usd_stage", "LoadUsdStageConfig"))
    payload = asyncio.run(info.fn("scene.usda"))
    assert payload["success"] is True
    assert payload["prim_count"] == 4

    monkeypatch.setattr(
        stage_utils,
        "load_stage",
        lambda path: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    info = asyncio.run(_function_info(module, "load_usd_stage", "LoadUsdStageConfig"))
    assert asyncio.run(info.fn("missing.usda"))["error"].startswith("File not found")

    monkeypatch.setattr(
        stage_utils,
        "load_stage",
        lambda path: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    info = asyncio.run(_function_info(module, "load_usd_stage", "LoadUsdStageConfig"))
    assert asyncio.run(info.fn("bad.usda"))["error"].startswith("Failed to load")

    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0", encoding="utf-8")
    monkeypatch.setattr(stage_utils, "load_stage", lambda path: {"stage": path})

    def fake_save_stage(_stage: Any, output_file: str) -> str:
        Path(output_file).write_text("saved", encoding="utf-8")
        return output_file

    monkeypatch.setattr(stage_utils, "save_stage", fake_save_stage)
    info = asyncio.run(_function_info(module, "save_usd_stage", "SaveUsdStageConfig"))
    payload = asyncio.run(info.fn(str(source), str(tmp_path / "out" / "saved.usda")))
    assert payload["success"] is True
    assert payload["file_size"] == 5

    payload = asyncio.run(
        info.fn(str(tmp_path / "missing.usda"), str(tmp_path / "x.usda"))
    )
    assert payload["success"] is False
    assert "Source file not found" in payload["error"]


def test_nat_register_graphics_render_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_register_module(monkeypatch, "register_graphics")

    from pxr import Usd

    from world_understanding.functions.graphics import rendering

    monkeypatch.setattr(Usd.Stage, "Open", lambda _path: "stage")
    monkeypatch.setattr(
        "world_understanding.utils.usd.stage.sanitize_name_for_filesystem",
        lambda name: str(name).replace("/", "_"),
    )

    class FakeBackend:
        def render(
            self, stage: Any, cameras: Any, image_width: int, frames: str
        ) -> dict[str, Any]:
            assert stage == "stage"
            if cameras is None:
                return {
                    "results": [
                        {
                            "camera": "/Cam A",
                            "images": [Image.new("RGB", (1, 1), "red")],
                            "frame_count": 1,
                        },
                        {
                            "camera": "/Cam B",
                            "images": [
                                Image.new("RGB", (1, 1), "green"),
                                Image.new("RGB", (1, 1), "blue"),
                            ],
                            "frame_count": 2,
                        },
                        {"camera": "/Broken", "error": "bad"},
                    ],
                    "total_cameras": 3,
                    "successful_cameras": 2,
                    "failed_cameras": 1,
                    "total_render_time": 1.5,
                }
            return {
                "results": [
                    {
                        "camera": cameras[0],
                        "images": [
                            Image.new("RGB", (1, 1), "red"),
                            Image.new("RGB", (1, 1), "blue"),
                        ],
                        "frame_count": 2,
                        "error": None,
                    }
                ],
                "total_render_time": 2.5,
            }

    monkeypatch.setattr(rendering, "RemoteRenderingBackend", FakeBackend)
    info = asyncio.run(
        _function_info(module, "render_single_camera", "RenderSingleCameraConfig")
    )
    payload = asyncio.run(
        info.fn(
            "scene.usda", "/Cam A", output_dir=str(tmp_path / "single"), frames="0,1"
        )
    )
    assert payload["success"] is True
    assert len(payload["images"]) == 2
    assert Path(payload["images"][0]).name == "scene__Cam A_frame0.png"

    class SingleImageBackend:
        def render(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "results": [
                    {
                        "camera": kwargs["cameras"][0],
                        "images": [Image.new("RGB", (1, 1), "red")],
                        "frame_count": 1,
                        "error": None,
                    }
                ],
                "total_render_time": 0.5,
            }

    monkeypatch.setattr(rendering, "RemoteRenderingBackend", SingleImageBackend)
    info = asyncio.run(
        _function_info(module, "render_single_camera", "RenderSingleCameraConfig")
    )
    payload = asyncio.run(
        info.fn("scene.usda", "/Cam A", output_dir=str(tmp_path / "single-one"))
    )
    assert Path(payload["images"][0]).name == "scene__Cam A.png"

    monkeypatch.setattr(rendering, "RemoteRenderingBackend", FakeBackend)
    info = asyncio.run(
        _function_info(module, "render_all_cameras", "RenderAllCamerasConfig")
    )
    payload = asyncio.run(info.fn("scene.usda", output_dir=str(tmp_path / "all")))
    assert payload["success"] is True
    assert payload["total_cameras"] == 3
    assert payload["cameras"][0]["images"][0].endswith("scene__Cam A.png")
    assert payload["cameras"][1]["images"][0].endswith("scene__Cam B_frame0.png")

    class RaisingBackend:
        def render(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("render failed")

    monkeypatch.setattr(rendering, "RemoteRenderingBackend", RaisingBackend)
    info = asyncio.run(
        _function_info(module, "render_single_camera", "RenderSingleCameraConfig")
    )
    assert asyncio.run(info.fn("scene.usda", "/Cam"))["success"] is False
    info = asyncio.run(
        _function_info(module, "render_all_cameras", "RenderAllCamerasConfig")
    )
    assert asyncio.run(info.fn("scene.usda"))["success"] is False
