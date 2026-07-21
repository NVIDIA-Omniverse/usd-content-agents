# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for the shared FAISS vector store base class."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image as PILImage

from world_understanding.functions.knowledge import base_vector_store as bvs


class TinyEmbeddingModel:
    AVAILABLE_MODELS = ["tiny"]
    DEFAULT_MODEL = "tiny"

    def __init__(self, model: str = "tiny") -> None:
        self.model = model
        self.base_url = "https://example.test"
        self.timeout = 3.0
        self.embedding_dimension = 2

    def embed_text(self, text: str | Path, **_kwargs: Any) -> np.ndarray:
        content = text.read_text(encoding="utf-8") if isinstance(text, Path) else text
        if "alpha" in content:
            return np.array([0.0, 0.0], dtype=np.float32)
        if "caption" in content:
            return np.array([0.0, 5.0], dtype=np.float32)
        if "beta" in content:
            return np.array([10.0, 0.0], dtype=np.float32)
        return np.array([float(len(content) % 7), 1.0], dtype=np.float32)

    def embed_image(
        self, image: str | Path | PILImage.Image | np.ndarray, **_kwargs: Any
    ) -> np.ndarray:
        if isinstance(image, PILImage.Image):
            return np.array([0.0, 5.0], dtype=np.float32)
        if isinstance(image, np.ndarray):
            return np.array([float(np.mean(image)), 5.0], dtype=np.float32)
        return np.array([float(len(str(image)) % 7), 5.0], dtype=np.float32)


class NIMTinyEmbeddingModel(TinyEmbeddingModel):
    pass


class OpenAITinyEmbeddingModel(TinyEmbeddingModel):
    pass


class ModulePathTinyEmbeddingModel(TinyEmbeddingModel):
    pass


class FakeTrainIndex:
    def __init__(self, vectors: np.ndarray | None = None) -> None:
        self._vectors = vectors
        self.ntotal = 0 if vectors is None else len(vectors)
        self.trained_with: np.ndarray | None = None

    def train(self, vectors: np.ndarray) -> None:
        self.trained_with = vectors

    def reconstruct_n(self, _start: int, size: int) -> np.ndarray:
        assert self._vectors is not None
        return self._vectors[:size]


class SparseSearchIndex:
    ntotal = 3

    def search(self, _query: np.ndarray, _k: int) -> tuple[np.ndarray, np.ndarray]:
        distances = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
        indices = np.array([[-1, 99, 0]], dtype=np.int64)
        return distances, indices


def test_base_document_content_type_edges() -> None:
    assert bvs.BaseDocument(document_id="empty").get_content_type() == "none"
    assert (
        bvs.BaseDocument(text_content="hello", document_id="text").get_content_type()
        == "text"
    )
    assert (
        bvs.BaseDocument(image_path="image.png", document_id="image").get_content_type()
        == "image"
    )
    assert (
        bvs.BaseDocument(
            text_path="text.txt",
            image_data=PILImage.new("RGB", (1, 1)),
            document_id="both",
        ).get_content_type()
        == "multimodal"
    )


def test_add_documents_ivf_training_and_search_caption_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ivf_store = bvs.BaseVectorStore(
        TinyEmbeddingModel(), index_type="IndexIVFFlat", nlist=1
    )
    assert ivf_store.add_text("alpha") == 0
    assert ivf_store._needs_training is False

    store = bvs.BaseVectorStore(TinyEmbeddingModel(), normalize_embeddings=True)
    docs = [
        bvs.BaseDocument(text_content="alpha document", document_id="alpha"),
        bvs.BaseDocument(text_content="beta document", document_id="beta"),
    ]
    assert store.add_documents(docs, "text") == [0, 1]
    assert store.add_documents(
        [bvs.BaseDocument(text_content="alpha again", document_id="alpha-2")],
        ["text"],
    ) == [2]
    with pytest.raises(ValueError, match="Number of embedding types"):
        store.add_documents(docs, ["text"])

    import world_understanding.functions.cv.vlm as vlm

    monkeypatch.setattr(vlm, "get_image_caption", lambda *_args, **_kwargs: "caption")
    image_id = store.add_image(PILImage.new("RGB", (2, 2)), embedding_type="text")
    assert store.metadata_store[image_id].document.text_content == "caption"

    text_results = store.search_by_text("alpha", k=2)
    assert text_results[0].document.document_id == "alpha"
    image_results = store.search_by_image(
        PILImage.new("RGB", (2, 2)),
        k=1,
        embedding_type="text",
        filter_metadata={"missing": None},
    )
    assert image_results[0].document.document_id == f"image_{image_id}"
    embedding_results = store.search_by_embedding(np.array([3.0, 4.0]), k=1)
    assert len(embedding_results) == 1
    assert (
        store.find_similar_documents("alpha", query_type="text", k=1)[
            0
        ].document.document_id
        == "alpha"
    )
    assert (
        len(
            store.find_similar_documents(
                np.array([3.0, 4.0]),
                query_type="embedding",
                k=1,
            )
        )
        == 1
    )


def test_search_skips_sparse_faiss_results_and_metadata_filter_none() -> None:
    store = bvs.BaseVectorStore(TinyEmbeddingModel())
    store.index = SparseSearchIndex()  # type: ignore[assignment]
    store.metadata_store = {
        0: bvs.BaseMetadata(
            document=bvs.BaseDocument(
                text_content="kept",
                document_id="kept",
                metadata={"category": "Alpha Team", "optional": None},
            ),
            embedding_id=0,
        )
    }

    assert (
        store.search(query_embedding=np.array([1.0, 0.0]), k=2)[0].document.document_id
        == "kept"
    )
    assert store.collect_documents({"category": "alpha"})[0].document_id == "kept"
    assert store.collect_documents({"optional": None})[0].document_id == "kept"
    assert store.collect_documents({"optional": "value"}) == []


def test_serialize_service_detection_clear_and_manual_training() -> None:
    assert (
        bvs.BaseVectorStore(NIMTinyEmbeddingModel())._serialize_embedding_model()[
            "service"
        ]
        == "nim"
    )
    assert (
        bvs.BaseVectorStore(OpenAITinyEmbeddingModel())._serialize_embedding_model()[
            "service"
        ]
        == "openai"
    )

    ModulePathTinyEmbeddingModel.__module__ = "pkg.nim.embedding"
    assert (
        bvs.BaseVectorStore(
            ModulePathTinyEmbeddingModel()
        )._serialize_embedding_model()["service"]
        == "nim"
    )
    ModulePathTinyEmbeddingModel.__module__ = "pkg.openai.embedding"
    assert (
        bvs.BaseVectorStore(
            ModulePathTinyEmbeddingModel()
        )._serialize_embedding_model()["service"]
        == "openai"
    )

    info = bvs.BaseVectorStore(TinyEmbeddingModel())._serialize_embedding_model()
    assert info["available_models"] == ["tiny"]
    assert info["default_model"] == "tiny"

    for index_type in ("IndexFlatIP", "IndexIVFFlat", "IndexHNSWFlat"):
        store = bvs.BaseVectorStore(
            TinyEmbeddingModel(),
            index_type=index_type,
            nlist=1,
            M=4,
        )
        store.add_text("alpha")
        store.clear()
        assert store.num_documents == 0
        assert store.index.ntotal == 0

    store = bvs.BaseVectorStore(TinyEmbeddingModel())
    initial_index = FakeTrainIndex()
    store.index = initial_index  # type: ignore[assignment]
    store._train_index([np.array([[1.0, 2.0]], dtype=np.float32)])
    assert initial_index.trained_with is not None
    assert initial_index.trained_with.shape == (1, 2)

    sampled_index = FakeTrainIndex(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    store.index = sampled_index  # type: ignore[assignment]
    store._train_index()
    assert sampled_index.trained_with is not None
    assert sampled_index.trained_with.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_build_vector_store_text_and_image_sources(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    text_dir = tmp_path / "texts"
    nested = text_dir / "nested"
    nested.mkdir(parents=True)
    root_text = text_dir / "root.txt"
    nested_text = nested / "nested.md"
    ignored = nested / "ignored.bin"
    root_text.write_text("alpha file", encoding="utf-8")
    nested_text.write_text("beta file", encoding="utf-8")
    ignored.write_text("ignored", encoding="utf-8")

    def metadata(path: str | Path) -> dict[str, Any]:
        return {"name": Path(path).name}

    recursive_store = bvs.BaseVectorStore.build_vector_store(
        TinyEmbeddingModel(),
        text_source=str(text_dir),
        metadata_extractor=metadata,
        recursive=True,
    )
    assert recursive_store.num_documents == 2
    assert {
        m.document.metadata["name"] for m in recursive_store.metadata_store.values()
    } == {
        "root.txt",
        "nested.md",
    }

    shallow_store = bvs.BaseVectorStore.build_vector_store(
        TinyEmbeddingModel(),
        text_source=[text_dir],
        metadata_extractor=metadata,
        recursive=False,
    )
    assert shallow_store.num_documents == 1

    file_store = bvs.BaseVectorStore.build_vector_store(
        TinyEmbeddingModel(),
        text_source=[root_text],
        metadata_extractor=metadata,
    )
    assert file_store.metadata_store[0].document.text_path == str(root_text)

    inline_store = bvs.BaseVectorStore.build_vector_store(
        TinyEmbeddingModel(),
        text_source="inline alpha text",
    )
    assert inline_store.metadata_store[0].document.text_content == "inline alpha text"

    caplog.set_level(logging.WARNING, logger=bvs.__name__)
    warning_store = bvs.BaseVectorStore.build_vector_store(
        TinyEmbeddingModel(),
        text_source=[root_text],
        metadata_extractor=lambda _path: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert warning_store.num_documents == 1
    assert "Failed to extract metadata" in caplog.text

    image = PILImage.new("RGB", (2, 2), "red")
    image_store = bvs.BaseVectorStore.build_vector_store(
        TinyEmbeddingModel(),
        image_source=image,
    )
    assert image_store.metadata_store[0].document.image_data is image
