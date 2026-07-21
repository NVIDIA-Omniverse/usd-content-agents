# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cluster_prims and expand_cluster_predictions tasks."""

from __future__ import annotations

import builtins
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from PIL import Image

import material_agent.tasks.cluster_prims as cluster_prims_module
from material_agent.tasks.cluster_prims import (
    DEFAULT_COMPLEXITY_THRESHOLDS,
    ClusterPrimsTask,
    ExpandClusterPredictionsTask,
    _cluster_by_tier,
    _complexity_tier,
    _default_embedding_model_for_service,
    _edge_density,
    _normalize_thresholds,
    _select_representatives,
    _split_large_clusters,
    _validate_optional_non_negative_int,
    _validate_optional_positive_int,
)
from material_agent.tasks.config_cluster_prims import (
    ClusterPrimsConfigTask,
    ExpandClusterPredictionsConfigTask,
)

# ---------------------------------------------------------------------------
# _edge_density
# ---------------------------------------------------------------------------


class TestEdgeDensity:
    def test_missing_file_returns_zero(self, tmp_path: Path) -> None:
        assert _edge_density(tmp_path / "nonexistent.png") == 0.0

    def test_valid_image_returns_float_in_range(self, tmp_path: Path) -> None:
        from PIL import Image

        # Create a simple gradient image that will have some edges
        img = Image.new("RGB", (64, 64), color="white")
        # Draw a black rectangle to create edges
        pixels = img.load()
        for x in range(20, 44):
            for y in range(20, 44):
                pixels[x, y] = (0, 0, 0)
        path = tmp_path / "test.png"
        img.save(path)

        result = _edge_density(path)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_blank_image_has_low_density(self, tmp_path: Path) -> None:
        from PIL import Image

        img = Image.new("RGB", (64, 64), color="white")
        path = tmp_path / "blank.png"
        img.save(path)

        result = _edge_density(path)
        assert result == 0.0

    def test_import_error_warns_once_and_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        original_import = builtins.__import__

        def import_without_cv2(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("cv2 unavailable")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(cluster_prims_module, "_cv2_warned", False)
        monkeypatch.setattr(builtins, "__import__", import_without_cv2)
        caplog.set_level(logging.WARNING)

        assert _edge_density("image.png") == 0.0
        assert _edge_density("image.png") == 0.0
        assert caplog.text.count("cv2 (opencv-python-headless)") == 1

    def test_cv2_exception_returns_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class BrokenCv2:
            def imread(self, path):
                raise RuntimeError("bad decoder")

        monkeypatch.setitem(__import__("sys").modules, "cv2", BrokenCv2())

        assert _edge_density("broken.png") == 0.0


# ---------------------------------------------------------------------------
# _complexity_tier
# ---------------------------------------------------------------------------


class TestComplexityTier:
    def test_low_tier(self) -> None:
        assert _complexity_tier(0.01, DEFAULT_COMPLEXITY_THRESHOLDS) == "low"

    def test_medium_tier(self) -> None:
        assert _complexity_tier(0.05, DEFAULT_COMPLEXITY_THRESHOLDS) == "medium"

    def test_high_tier(self) -> None:
        assert _complexity_tier(0.10, DEFAULT_COMPLEXITY_THRESHOLDS) == "high"

    def test_boundary_low_medium(self) -> None:
        # 0.02 is >= low.hi, so should be medium
        assert _complexity_tier(0.02, DEFAULT_COMPLEXITY_THRESHOLDS) == "medium"

    def test_boundary_medium_high(self) -> None:
        assert _complexity_tier(0.08, DEFAULT_COMPLEXITY_THRESHOLDS) == "high"

    def test_out_of_range_negative_falls_back_to_high(self) -> None:
        # Negative is not in any [lo, hi) range
        assert _complexity_tier(-1.0, DEFAULT_COMPLEXITY_THRESHOLDS) == "high"

    def test_out_of_range_above_falls_back_to_high(self) -> None:
        # The high tier goes up to 1.0 inclusive (due to max_hi logic in
        # _cluster_by_tier), but _complexity_tier uses strict < for hi.
        # Score of 1.5 is above all tiers.
        assert _complexity_tier(1.5, DEFAULT_COMPLEXITY_THRESHOLDS) == "high"


class TestNormalizeThresholds:
    def test_returns_ordered_cover(self) -> None:
        thresholds = _normalize_thresholds(
            {
                "medium": [0.2, 0.8, 0.95],
                "low": [0.0, 0.2, 0.98],
                "high": [0.8, 1.0, 0.90],
            }
        )

        assert list(thresholds) == ["low", "medium", "high"]

    def test_rejects_gap(self) -> None:
        with pytest.raises(ValueError, match="Gap in complexity_thresholds"):
            _normalize_thresholds(
                {
                    "low": [0.0, 0.2, 0.98],
                    "high": [0.3, 1.0, 0.90],
                }
            )

    def test_rejects_overlap(self) -> None:
        with pytest.raises(ValueError, match="Overlapping complexity_thresholds"):
            _normalize_thresholds(
                {
                    "low": [0.0, 0.5, 0.98],
                    "high": [0.4, 1.0, 0.90],
                }
            )

    def test_rejects_incomplete_upper_bound(self) -> None:
        with pytest.raises(ValueError, match="cover edge density values up to 1.0"):
            _normalize_thresholds({"low": [0.0, 0.9, 0.98]})


class TestClusterConfigHelpers:
    def test_default_embedding_model_for_unknown_service_uses_mock_default(
        self,
    ) -> None:
        assert _default_embedding_model_for_service("other") == (
            cluster_prims_module.DEFAULT_EMBEDDING_MODEL
        )

    @pytest.mark.parametrize("value", [None, ""])
    def test_optional_positive_int_accepts_empty_values(self, value: object) -> None:
        assert _validate_optional_positive_int("max_cluster_size", value) is None

    @pytest.mark.parametrize("value", [None, ""])
    def test_optional_non_negative_int_accepts_empty_values(
        self, value: object
    ) -> None:
        assert (
            _validate_optional_non_negative_int("report.max_singletons", value) is None
        )


# ---------------------------------------------------------------------------
# _cluster_by_tier
# ---------------------------------------------------------------------------


class TestClusterByTier:
    def test_identical_vectors_cluster_together(self) -> None:
        # 4 identical vectors should form one cluster
        vec = np.random.default_rng(42).random(128)
        vec /= np.linalg.norm(vec)
        embeddings = np.tile(vec, (4, 1))
        complexities = np.array([0.01, 0.01, 0.01, 0.01])  # all low tier

        labels = _cluster_by_tier(
            embeddings, complexities, DEFAULT_COMPLEXITY_THRESHOLDS
        )
        assert len(np.unique(labels)) == 1

    def test_different_vectors_separate_clusters(self) -> None:
        rng = np.random.default_rng(42)
        # Two very different unit vectors in low-complexity tier
        v1 = rng.random(128)
        v1 /= np.linalg.norm(v1)
        v2 = -v1  # opposite direction -> cosine distance = 2.0
        embeddings = np.stack([v1, v2])
        complexities = np.array([0.01, 0.01])

        labels = _cluster_by_tier(
            embeddings, complexities, DEFAULT_COMPLEXITY_THRESHOLDS
        )
        assert labels[0] != labels[1]

    def test_single_prim_gets_own_cluster(self) -> None:
        rng = np.random.default_rng(42)
        vec = rng.random(128)
        vec /= np.linalg.norm(vec)
        embeddings = vec.reshape(1, -1)
        complexities = np.array([0.05])  # medium tier

        labels = _cluster_by_tier(
            embeddings, complexities, DEFAULT_COMPLEXITY_THRESHOLDS
        )
        assert labels[0] >= 0

    def test_different_tiers_separate_clusters(self) -> None:
        # Two identical vectors in different complexity tiers should be in
        # different clusters (they are clustered independently per tier).
        rng = np.random.default_rng(42)
        vec = rng.random(128)
        vec /= np.linalg.norm(vec)
        embeddings = np.tile(vec, (2, 1))
        complexities = np.array([0.01, 0.10])  # low vs high tier

        labels = _cluster_by_tier(
            embeddings, complexities, DEFAULT_COMPLEXITY_THRESHOLDS
        )
        assert labels[0] != labels[1]

    def test_all_labels_assigned(self) -> None:
        rng = np.random.default_rng(42)
        embeddings = rng.random((5, 64))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= norms
        complexities = np.array([0.01, 0.03, 0.05, 0.09, 0.5])

        labels = _cluster_by_tier(
            embeddings, complexities, DEFAULT_COMPLEXITY_THRESHOLDS
        )
        assert np.all(labels >= 0)


class TestSplitLargeClusters:
    def test_splits_oversized_clusters_into_stable_chunks(self) -> None:
        labels = _split_large_clusters(np.array([0, 0, 0, 0, 0]), 2)

        assert len(np.unique(labels)) == 3
        assert [int((labels == cid).sum()) for cid in np.unique(labels)] == [2, 2, 1]

    def test_none_max_cluster_size_returns_original_labels(self) -> None:
        labels = np.array([0, 0, 1])
        assert _split_large_clusters(labels, None) is labels


# ---------------------------------------------------------------------------
# _select_representatives
# ---------------------------------------------------------------------------


class TestSelectRepresentatives:
    def test_picks_closest_to_centroid(self) -> None:
        # 3 vectors in one cluster; the centroid-closest should be chosen
        v_center = np.array([1.0, 0.0, 0.0])
        v_near = np.array([0.99, 0.1, 0.0])
        v_near /= np.linalg.norm(v_near)
        v_far = np.array([0.5, 0.5, 0.5])
        v_far /= np.linalg.norm(v_far)

        embeddings = np.stack([v_far, v_center, v_near])
        labels = np.array([0, 0, 0])

        reps = _select_representatives(embeddings, labels)
        assert 0 in reps
        # The centroid of these vectors should be closest to v_center or v_near
        rep_idx = reps[0]
        assert rep_idx in (1, 2)  # v_center or v_near

    def test_singleton_cluster(self) -> None:
        embeddings = np.array([[1.0, 0.0, 0.0]])
        labels = np.array([5])

        reps = _select_representatives(embeddings, labels)
        assert reps[5] == 0

    def test_multiple_clusters(self) -> None:
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        labels = np.array([0, 0, 1, 1])

        reps = _select_representatives(embeddings, labels)
        assert set(reps.keys()) == {0, 1}
        assert reps[0] in (0, 1)
        assert reps[1] in (2, 3)


# ---------------------------------------------------------------------------
# ClusterPrimsTask.run — skip path
# ---------------------------------------------------------------------------


def _make_dataset_jsonl(path: Path, n: int) -> Path:
    """Write a minimal dataset.jsonl with n entries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"prim_{i}", "images": {"prim_only": []}}) + "\n")
    return path


class TestClusterPrimsTaskRun:
    def test_skips_when_below_min_prims(self, tmp_path: Path) -> None:
        dataset_path = _make_dataset_jsonl(tmp_path / "dataset" / "dataset.jsonl", n=5)
        context: dict[str, Any] = {
            "dataset_path": str(dataset_path),
            "working_dir": str(tmp_path / "work"),
            "cluster_prims_config": {"min_prims_to_activate": 10},
        }

        task = ClusterPrimsTask()
        result = task.run(context)

        assert result["cluster_prims_ran"] is False
        assert "cluster_map_path" not in result

    def test_skips_when_below_string_min_prims(self, tmp_path: Path) -> None:
        dataset_path = _make_dataset_jsonl(tmp_path / "dataset" / "dataset.jsonl", n=5)
        context: dict[str, Any] = {
            "dataset_path": str(dataset_path),
            "working_dir": str(tmp_path / "work"),
            "cluster_prims_config": {"min_prims_to_activate": "10"},
        }

        result = ClusterPrimsTask().run(context)

        assert result["cluster_prims_ran"] is False
        assert "cluster_map_path" not in result

    def test_all_no_image_prims_do_not_create_embedding_model(
        self, tmp_path: Path
    ) -> None:
        dataset_path = _make_dataset_jsonl(tmp_path / "dataset" / "dataset.jsonl", n=3)
        context: dict[str, Any] = {
            "dataset_path": str(dataset_path),
            "working_dir": str(tmp_path / "work"),
            "cluster_prims_config": {
                "min_prims_to_activate": 1,
                "embedding_service": "nim",
                "report": False,
            },
        }

        with patch(
            "world_understanding.functions.models.image_embedding_models.create_image_embedding_model"
        ) as create_model:
            result = ClusterPrimsTask().run(context)

        create_model.assert_not_called()
        assert result["cluster_prims_ran"] is True
        assert result["cluster_count"] == 3
        rows = _read_jsonl(Path(result["cluster_map_path"]))
        assert len(rows) == 3
        assert all(row["cluster_size"] == 1 for row in rows)

    def test_no_image_path_absolutizes_v02_media_and_copies_dataset_config(
        self, tmp_path: Path
    ) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        preview = dataset_dir / "preview.png"
        Image.new("RGB", (8, 8), color="blue").save(preview)
        dataset_jsonl = dataset_dir / "dataset.jsonl"
        dataset_jsonl.write_text(
            json.dumps(
                {
                    "id": "media_prim",
                    "media": {
                        "images": [
                            {
                                "path": "preview.png",
                                "metadata": {"render_mode": "beauty"},
                            }
                        ]
                    },
                }
            )
            + "\n"
            + json.dumps({"id": "legacy_prim", "images": {"prim_only": "not-a-list"}})
            + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "dataset.json").write_text(
            json.dumps({"system_prompt": "preserved"}),
            encoding="utf-8",
        )
        context: dict[str, Any] = {
            "dataset_path": str(dataset_jsonl),
            "working_dir": str(tmp_path / "work"),
            "cluster_prims_config": {"min_prims_to_activate": 1, "report": False},
        }

        result = ClusterPrimsTask().run(context)

        assert result["cluster_prims_ran"] is True
        copied_config = Path(result["cluster_summary_path"]).parent / "dataset.json"
        assert json.loads(copied_config.read_text()) == {"system_prompt": "preserved"}
        reps = _read_jsonl(Path(result["dataset_representatives_path"]))
        reps_by_id = {row["id"]: row for row in reps}
        assert reps_by_id["media_prim"]["media"]["images"][0]["path"] == str(
            preview.resolve()
        )
        assert reps_by_id["legacy_prim"]["images"]["prim_only"] == "not-a-list"

    def test_progress_listener_errors_are_debug_only(self, tmp_path: Path) -> None:
        dataset_path = _make_dataset_jsonl(tmp_path / "dataset" / "dataset.jsonl", n=1)
        listener = MagicMock()
        listener.event.side_effect = RuntimeError("listener down")

        with patch(
            "material_agent.tasks.cluster_prims.get_listener",
            return_value=listener,
        ):
            result = ClusterPrimsTask().run(
                {
                    "dataset_path": str(dataset_path),
                    "working_dir": str(tmp_path / "work"),
                    "cluster_prims_config": {"min_prims_to_activate": 5},
                }
            )

        assert result["cluster_prims_ran"] is False
        listener.event.assert_called_once()

    def test_copies_dataset_json_to_clusters_dir(self, tmp_path: Path) -> None:
        """Verify that dataset.json is copied into the clusters/ directory."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        # Write dataset.jsonl with enough prims and prim_only images
        n = 3
        entries = []
        for i in range(n):
            img_path = dataset_dir / f"prim_{i}.png"
            # Create a tiny image
            from PIL import Image

            Image.new("RGB", (8, 8), color="red").save(img_path)
            entries.append(
                {
                    "id": f"prim_{i}",
                    "images": {"prim_only": [str(img_path)]},
                }
            )

        dataset_jsonl = dataset_dir / "dataset.jsonl"
        with open(dataset_jsonl, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        # Write a dataset.json config file
        dataset_json = dataset_dir / "dataset.json"
        dataset_json.write_text(json.dumps({"system_prompt": "test"}))

        working_dir = tmp_path / "work"
        context: dict[str, Any] = {
            "dataset_path": str(dataset_jsonl),
            "working_dir": str(working_dir),
            "cluster_prims_config": {
                "min_prims_to_activate": 1,
                "report": False,  # skip HTML report generation
            },
        }

        # Mock the embedding model
        mock_model = MagicMock()
        mock_model.embedding_dimension = 8
        mock_model.embed_images = MagicMock(
            side_effect=lambda imgs: [np.random.default_rng(42).random(8) for _ in imgs]
        )

        with patch(
            "world_understanding.functions.models.image_embedding_models.create_image_embedding_model",
            return_value=mock_model,
        ):
            task = ClusterPrimsTask()
            result = task.run(context)

        assert result["cluster_prims_ran"] is True
        # Check dataset.json was copied
        copied = working_dir / "clusters" / "dataset.json"
        assert copied.exists()
        assert json.loads(copied.read_text()) == {"system_prompt": "test"}

    def test_passes_embedding_base_url_to_model_factory(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        img_path = dataset_dir / "prim_0.png"
        Image.new("RGB", (8, 8), color="red").save(img_path)
        dataset_jsonl = dataset_dir / "dataset.jsonl"
        dataset_jsonl.write_text(
            json.dumps({"id": "prim_0", "images": {"prim_only": [str(img_path)]}})
            + "\n",
            encoding="utf-8",
        )

        mock_model = MagicMock()
        mock_model.embedding_dimension = 8
        mock_model.embed_images = MagicMock(return_value=[np.ones(8)])

        with patch(
            "world_understanding.functions.models.image_embedding_models.create_image_embedding_model",
            return_value=mock_model,
        ) as create_model:
            ClusterPrimsTask().run(
                {
                    "dataset_path": str(dataset_jsonl),
                    "working_dir": str(tmp_path / "work"),
                    "cluster_prims_config": {
                        "min_prims_to_activate": 1,
                        "embedding_service": "nim",
                        "embedding_model": "nvidia/llama-nemotron-embed-vl-1b-v2",
                        "base_url": "http://embed-nim:8000/v1",
                        "api_key": "not-used",
                        "report": False,
                    },
                }
            )

        create_model.assert_called_once()
        assert create_model.call_args.kwargs["base_url"] == "http://embed-nim:8000/v1"

    def test_empty_cluster_api_key_env_falls_back_to_nvidia_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        img_path = dataset_dir / "prim_0.png"
        Image.new("RGB", (8, 8), color="red").save(img_path)
        dataset_jsonl = dataset_dir / "dataset.jsonl"
        dataset_jsonl.write_text(
            json.dumps({"id": "prim_0", "images": {"prim_only": [str(img_path)]}})
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("MA_CLUSTER_EMBEDDING_API_KEY", "")
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-real")

        mock_model = MagicMock()
        mock_model.embedding_dimension = 8
        mock_model.embed_images = MagicMock(return_value=[np.ones(8)])

        with patch(
            "world_understanding.functions.models.image_embedding_models.create_image_embedding_model",
            return_value=mock_model,
        ) as create_model:
            ClusterPrimsTask().run(
                {
                    "dataset_path": str(dataset_jsonl),
                    "working_dir": str(tmp_path / "work"),
                    "cluster_prims_config": {
                        "min_prims_to_activate": 1,
                        "embedding_service": "nim",
                        "embedding_model": "nvidia/llama-nemotron-embed-vl-1b-v2",
                        "report": False,
                    },
                }
            )

        create_model.assert_called_once()
        assert create_model.call_args.kwargs["api_key"] == "nvapi-real"

    def test_cluster_api_key_env_resolves_endpoint_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MA_CLUSTER_EMBEDDING_API_KEY", raising=False)
        monkeypatch.delenv("MA_CLUSTER_EMBEDDING_API_KEY_ENV", raising=False)
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        img_path = dataset_dir / "prim_0.png"
        Image.new("RGB", (8, 8), color="red").save(img_path)
        dataset_jsonl = dataset_dir / "dataset.jsonl"
        dataset_jsonl.write_text(
            json.dumps({"id": "prim_0", "images": {"prim_only": [str(img_path)]}})
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("CLUSTER_EMBEDDING_API_KEY", "endpoint-embedding-key")

        mock_model = MagicMock()
        mock_model.embedding_dimension = 8
        mock_model.embed_images = MagicMock(return_value=[np.ones(8)])

        with patch(
            "world_understanding.functions.models.image_embedding_models.create_image_embedding_model",
            return_value=mock_model,
        ) as create_model:
            ClusterPrimsTask().run(
                {
                    "dataset_path": str(dataset_jsonl),
                    "working_dir": str(tmp_path / "work"),
                    "cluster_prims_config": {
                        "min_prims_to_activate": 1,
                        "embedding_service": "nim",
                        "embedding_model": "nvidia/llama-nemotron-embed-vl-1b-v2",
                        "base_url": "http://embed-nim:8000/v1",
                        "api_key_env": "CLUSTER_EMBEDDING_API_KEY",
                        "report": False,
                    },
                }
            )

        create_model.assert_called_once()
        assert create_model.call_args.kwargs["api_key"] == "endpoint-embedding-key"

    def test_retries_transient_embedding_batch_failure(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        from PIL import Image

        entries = []
        for i in range(2):
            img_path = dataset_dir / f"prim_{i}.png"
            Image.new("RGB", (8, 8), color="red").save(img_path)
            entries.append(
                {"id": f"prim_{i}", "images": {"prim_only": [str(img_path)]}}
            )

        dataset_jsonl = dataset_dir / "dataset.jsonl"
        with open(dataset_jsonl, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        mock_model = MagicMock()
        mock_model.embedding_dimension = 8
        mock_model.embed_images = MagicMock(
            side_effect=[
                RuntimeError("temporary embedding outage"),
                [np.ones(8), np.ones(8)],
            ]
        )

        context: dict[str, Any] = {
            "dataset_path": str(dataset_jsonl),
            "working_dir": str(tmp_path / "work"),
            "cluster_prims_config": {
                "min_prims_to_activate": 1,
                "batch_size": 2,
                "max_workers": 1,
                "embedding_retries": 2,
                "embedding_retry_initial_delay": 0,
                "report": False,
            },
        }

        with (
            patch(
                "world_understanding.functions.models.image_embedding_models.create_image_embedding_model",
                return_value=mock_model,
            ),
            patch("material_agent.tasks.cluster_prims.time.sleep"),
        ):
            result = ClusterPrimsTask().run(context)

        assert result["cluster_prims_ran"] is True
        assert mock_model.embed_images.call_count == 2

    def test_splits_large_visual_cluster_by_max_cluster_size(
        self, tmp_path: Path
    ) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        entries = []
        for i in range(5):
            img_path = dataset_dir / f"prim_{i}.png"
            Image.new("RGB", (8, 8), color="red").save(img_path)
            entries.append(
                {"id": f"prim_{i}", "images": {"prim_only": [str(img_path)]}}
            )

        dataset_jsonl = dataset_dir / "dataset.jsonl"
        with open(dataset_jsonl, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        mock_model = MagicMock()
        mock_model.embedding_dimension = 4
        mock_model.embed_images = MagicMock(
            side_effect=lambda imgs: [np.ones(4) for _ in imgs]
        )

        context: dict[str, Any] = {
            "dataset_path": str(dataset_jsonl),
            "working_dir": str(tmp_path / "work"),
            "cluster_prims_config": {
                "min_prims_to_activate": 1,
                "embedding_service": "mock",
                "batch_size": 5,
                "max_workers": 1,
                "max_cluster_size": 2,
                "complexity_thresholds": {
                    "low": [0.0, 1.0, 0.0],
                    "medium": [1.0, 2.0, 0.0],
                    "high": [2.0, 3.0, 0.0],
                },
                "report": False,
            },
        }

        with patch(
            "world_understanding.functions.models.image_embedding_models.create_image_embedding_model",
            return_value=mock_model,
        ):
            result = ClusterPrimsTask().run(context)

        assert result["cluster_prims_ran"] is True
        assert result["cluster_count"] == 3
        assert result["cluster_representative_count"] == 3
        assert result["cluster_max_size"] == 2
        assert result["cluster_capped_count"] == 1
        rows = _read_jsonl(Path(result["cluster_map_path"]))
        assert max(row["cluster_size"] for row in rows) == 2
        assert sum(1 for row in rows if row["is_representative"]) == 3
        summary = json.loads(Path(result["cluster_summary_path"]).read_text())
        assert summary["cluster_count"] == 3
        assert summary["observed_max_cluster_size"] == 2
        assert summary["capped_cluster_count"] == 1

    def test_cluster_report_respects_size_limits(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        entries = []
        for i in range(4):
            img_path = dataset_dir / f"prim_{i}.png"
            Image.new("RGB", (8, 8), color="red").save(img_path)
            entries.append(
                {"id": f"prim_{i}", "images": {"prim_only": [str(img_path)]}}
            )

        dataset_jsonl = dataset_dir / "dataset.jsonl"
        with open(dataset_jsonl, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        mock_model = MagicMock()
        mock_model.embedding_dimension = 4
        mock_model.embed_images = MagicMock(
            side_effect=lambda imgs: [np.ones(4) for _ in imgs]
        )

        context: dict[str, Any] = {
            "dataset_path": str(dataset_jsonl),
            "working_dir": str(tmp_path / "work"),
            "cluster_prims_config": {
                "min_prims_to_activate": 1,
                "embedding_service": "mock",
                "batch_size": 4,
                "max_workers": 1,
                "complexity_thresholds": {
                    "low": [0.0, 1.0, 0.0],
                    "medium": [1.0, 2.0, 0.0],
                    "high": [2.0, 3.0, 0.0],
                },
                "report": {
                    "enabled": True,
                    "max_multi_member_clusters": 1,
                    "max_members_per_cluster": 2,
                    "max_singletons": 0,
                },
            },
        }

        with patch(
            "world_understanding.functions.models.image_embedding_models.create_image_embedding_model",
            return_value=mock_model,
        ):
            result = ClusterPrimsTask().run(context)

        report_html = Path(result["cluster_report_path"]).read_text()
        assert "2 additional members omitted by report limit" in report_html
        summary = json.loads(Path(result["cluster_summary_path"]).read_text())
        assert summary["report_limits"]["max_members_per_cluster"] == 2

    def test_v02_media_images_and_mixed_no_image_prims_are_clustered(
        self, tmp_path: Path
    ) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        entries = []
        for i in range(2):
            img_path = dataset_dir / f"prim_{i}.png"
            Image.new("RGB", (8, 8), color="green").save(img_path)
            entries.append(
                {
                    "id": f"prim_{i}",
                    "media": {
                        "images": [
                            {
                                "path": img_path.name,
                                "metadata": {"render_mode": "prim_only"},
                            }
                        ]
                    },
                }
            )
        entries.append({"id": "no_image", "images": {"prim_only": []}})
        dataset_jsonl = dataset_dir / "dataset.jsonl"
        dataset_jsonl.write_text(
            "".join(json.dumps(entry) + "\n" for entry in entries),
            encoding="utf-8",
        )

        mock_model = MagicMock()
        mock_model.embedding_dimension = 4
        mock_model.embed_images = MagicMock(return_value=[np.ones(4), np.ones(4)])

        with patch(
            "world_understanding.functions.models.image_embedding_models.create_image_embedding_model",
            return_value=mock_model,
        ):
            result = ClusterPrimsTask().run(
                {
                    "dataset_path": str(dataset_jsonl),
                    "working_dir": str(tmp_path / "work"),
                    "cluster_prims_config": {
                        "min_prims_to_activate": 1,
                        "batch_size": 2,
                        "max_workers": 1,
                        "report": False,
                    },
                }
            )

        rows = _read_jsonl(Path(result["cluster_map_path"]))
        no_image_row = next(row for row in rows if row["id"] == "no_image")
        assert no_image_row["is_representative"] is True
        assert no_image_row["cluster_size"] == 1
        reps = _read_jsonl(Path(result["dataset_representatives_path"]))
        assert len(reps) == 2

    def test_logs_embedding_progress_every_twentieth_batch(
        self, tmp_path: Path
    ) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        entries = []
        for i in range(20):
            img_path = dataset_dir / f"prim_{i}.png"
            Image.new("RGB", (4, 4), color="red").save(img_path)
            entries.append(
                {"id": f"prim_{i}", "images": {"prim_only": [str(img_path)]}}
            )
        dataset_jsonl = dataset_dir / "dataset.jsonl"
        dataset_jsonl.write_text(
            "".join(json.dumps(entry) + "\n" for entry in entries),
            encoding="utf-8",
        )

        mock_model = MagicMock()
        mock_model.embedding_dimension = 4
        mock_model.embed_images = MagicMock(return_value=[np.ones(4)])
        listener = MagicMock()

        with (
            patch(
                "world_understanding.functions.models.image_embedding_models.create_image_embedding_model",
                return_value=mock_model,
            ),
            patch(
                "material_agent.tasks.cluster_prims.get_listener",
                return_value=listener,
            ),
        ):
            ClusterPrimsTask().run(
                {
                    "dataset_path": str(dataset_jsonl),
                    "working_dir": str(tmp_path / "work"),
                    "cluster_prims_config": {
                        "min_prims_to_activate": 1,
                        "batch_size": 1,
                        "max_workers": 1,
                        "report": False,
                    },
                }
            )

        assert any(
            "Embedded 20/20 images" in str(call)
            for call in listener.info.call_args_list
        )

    def test_encode_image_handles_alpha_palette_and_failures(
        self, tmp_path: Path
    ) -> None:
        rgba_path = tmp_path / "rgba.png"
        Image.new("RGBA", (8, 8), color=(255, 0, 0, 128)).save(rgba_path)
        palette_path = tmp_path / "palette.png"
        Image.new("P", (8, 8)).save(palette_path)

        assert ClusterPrimsTask._encode_image(str(rgba_path), fmt="jpeg")
        assert ClusterPrimsTask._encode_image(str(palette_path), fmt="jpeg")
        assert ClusterPrimsTask._encode_image(str(tmp_path / "missing.png")) is None

    def test_generate_html_report_with_all_limits_visible_and_singletons(
        self, tmp_path: Path
    ) -> None:
        image_path = tmp_path / "prim.png"
        Image.new("RGB", (8, 8), color="purple").save(image_path)
        report_path = tmp_path / "report.html"

        ClusterPrimsTask()._generate_html_report(
            report_path=report_path,
            dataset=[
                {"id": "rep<&>"},
                {"id": "member"},
                {"id": "singleton"},
            ],
            labels=np.array([0, 0, 1]),
            reps={0: 0, 1: 2},
            complexities=np.array([0.01, 0.01, 0.9]),
            thresholds={"low": (0.0, 0.5, 0.9), "wild tier!": (0.5, 1.0, 0.8)},
            prim_image_paths=[[str(image_path)], [str(image_path)], [str(image_path)]],
            n_clusters=2,
            reduction_pct=33.3,
            image_max_size=16,
            image_format="png",
            image_quality=75,
            max_multi_member_clusters=None,
            max_members_per_cluster=None,
            max_singletons=None,
            listener=MagicMock(),
        )

        html = report_path.read_text()
        assert "rep&lt;&amp;&gt;" in html
        assert "Singletons (1)" in html
        assert "data:image/png;base64" in html

    def test_generate_html_report_notes_omitted_clusters_and_singletons(
        self, tmp_path: Path
    ) -> None:
        report_path = tmp_path / "limited.html"

        ClusterPrimsTask()._generate_html_report(
            report_path=report_path,
            dataset=[
                {"id": "a"},
                {"id": "b"},
                {"id": "c"},
                {"id": "d"},
                {"id": "e"},
                {"id": "f"},
            ],
            labels=np.array([0, 0, 1, 1, 2, 3]),
            reps={0: 0, 1: 2, 2: 4, 3: 5},
            complexities=np.zeros(6),
            thresholds={"low": (0.0, 1.0, 0.9)},
            prim_image_paths=[[] for _ in range(6)],
            n_clusters=4,
            reduction_pct=33.3,
            image_max_size=16,
            image_format="jpeg",
            image_quality=75,
            max_multi_member_clusters=1,
            max_members_per_cluster=1,
            max_singletons=1,
            listener=MagicMock(),
        )

        html = report_path.read_text()
        assert "additional multi-member clusters omitted" in html
        assert "additional singletons omitted" in html


# ---------------------------------------------------------------------------
# ExpandClusterPredictionsTask.run
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestExpandClusterPredictionsTaskRun:
    def test_skips_when_cluster_prims_not_ran(self) -> None:
        context: dict[str, Any] = {"cluster_prims_ran": False}
        task = ExpandClusterPredictionsTask()
        result = task.run(context)
        assert result is context

    def test_expands_predictions(self, tmp_path: Path) -> None:
        predictions_path = tmp_path / "predictions" / "predictions.jsonl"
        cluster_map_path = tmp_path / "clusters" / "cluster_map.jsonl"

        # Rep prim_0 predicted "metal"; prim_1 is a member of the same cluster
        _write_jsonl(
            predictions_path,
            [
                {"id": "prim_0", "material": "metal", "confidence": 0.9},
            ],
        )
        _write_jsonl(
            cluster_map_path,
            [
                {
                    "id": "prim_0",
                    "cluster_id": 0,
                    "is_representative": True,
                    "cluster_representative_id": "prim_0",
                    "cluster_size": 2,
                    "complexity_score": 0.01,
                    "complexity_tier": "low",
                },
                {
                    "id": "prim_1",
                    "cluster_id": 0,
                    "is_representative": False,
                    "cluster_representative_id": "prim_0",
                    "cluster_size": 2,
                    "complexity_score": 0.01,
                    "complexity_tier": "low",
                },
            ],
        )

        context: dict[str, Any] = {
            "cluster_prims_ran": True,
            "predictions_path": str(predictions_path),
            "cluster_map_path": str(cluster_map_path),
        }

        task = ExpandClusterPredictionsTask()
        result = task.run(context)

        preds = _read_jsonl(Path(result["predictions_path"]))
        assert len(preds) == 2

        # Representative keeps its own prediction
        rep = next(p for p in preds if p["id"] == "prim_0")
        assert rep["material"] == "metal"
        assert "prediction_source" not in rep

        # Member gets propagated prediction
        member = next(p for p in preds if p["id"] == "prim_1")
        assert member["material"] == "metal"
        assert member["prediction_source"] == "cluster_representative"
        assert member["cluster_representative_id"] == "prim_0"
        assert member["cluster_id"] == 0

    def test_missing_representative_predictions_are_skipped_with_warning(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions" / "predictions.jsonl"
        cluster_map_path = tmp_path / "clusters" / "cluster_map.jsonl"
        _write_jsonl(
            predictions_path,
            [
                {"id": "predicted_rep", "material": "metal"},
            ],
        )
        _write_jsonl(
            cluster_map_path,
            [
                {
                    "id": "predicted_rep",
                    "cluster_id": 0,
                    "is_representative": True,
                    "cluster_representative_id": "predicted_rep",
                    "cluster_size": 1,
                },
                {
                    "id": "missing_rep",
                    "cluster_id": 1,
                    "is_representative": True,
                    "cluster_representative_id": "missing_rep",
                    "cluster_size": 2,
                },
                {
                    "id": "member_missing_rep",
                    "cluster_id": 1,
                    "is_representative": False,
                    "cluster_representative_id": "missing_rep",
                    "cluster_size": 2,
                },
            ],
        )
        listener = MagicMock()
        listener.event.side_effect = RuntimeError("listener down")

        with patch(
            "material_agent.tasks.cluster_prims.get_listener",
            return_value=listener,
        ):
            result = ExpandClusterPredictionsTask().run(
                {
                    "cluster_prims_ran": True,
                    "predictions_path": str(predictions_path),
                    "cluster_map_path": str(cluster_map_path),
                }
            )

        assert result["predictions_path"] == str(predictions_path)
        assert _read_jsonl(predictions_path) == [
            {"id": "predicted_rep", "material": "metal"}
        ]
        listener.warning.assert_called_once()
        assert listener.event.call_count == 3

    def test_raises_for_missing_prediction_or_cluster_map_files(
        self, tmp_path: Path
    ) -> None:
        cluster_map_path = tmp_path / "clusters" / "cluster_map.jsonl"
        _write_jsonl(cluster_map_path, [])

        with pytest.raises(FileNotFoundError, match="predictions_path not found"):
            ExpandClusterPredictionsTask().run(
                {
                    "cluster_prims_ran": True,
                    "predictions_path": str(tmp_path / "missing.jsonl"),
                    "cluster_map_path": str(cluster_map_path),
                }
            )

        predictions_path = tmp_path / "predictions" / "predictions.jsonl"
        _write_jsonl(predictions_path, [])
        with pytest.raises(FileNotFoundError, match="cluster_map_path not found"):
            ExpandClusterPredictionsTask().run(
                {
                    "cluster_prims_ran": True,
                    "predictions_path": str(predictions_path),
                    "cluster_map_path": str(tmp_path / "missing-map.jsonl"),
                }
            )


# ---------------------------------------------------------------------------
# Config tasks
# ---------------------------------------------------------------------------


class TestClusterPrimsConfigTask:
    def test_loads_config(self, tmp_path: Path) -> None:
        config = {
            "dataset_path": "/some/dataset.jsonl",
            "working_dir": "/some/workdir",
            "batch_size": 100,
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ClusterPrimsConfigTask()
        result = task.run(context)

        assert result["dataset_path"] == "/some/dataset.jsonl"
        assert result["working_dir"] == "/some/workdir"
        assert result["cluster_prims_config"]["batch_size"] == 100

    def test_resolves_relative_paths_against_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "configs" / "cluster.yaml"
        config_path.parent.mkdir()
        config_path.write_text(
            yaml.safe_dump(
                {
                    "dataset_path": "data/dataset.jsonl",
                    "working_dir": "run",
                }
            ),
            encoding="utf-8",
        )

        result = ClusterPrimsConfigTask().run({"config_path": str(config_path)})

        assert result["dataset_path"] == str(
            config_path.parent / "data" / "dataset.jsonl"
        )
        assert result["working_dir"] == str(config_path.parent / "run")

    def test_raises_when_both_config_sources_are_missing(self) -> None:
        with pytest.raises(ValueError, match="config_dict or config_path"):
            ClusterPrimsConfigTask().run({})

    def test_logs_redact_sensitive_config_values(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = {
            "dataset_path": "/some/dataset.jsonl",
            "working_dir": "/some/workdir",
            "api_key": "super-secret-key",
            "nested": {
                "embedding_api_key": "nested-secret-key",
                "list": [{"token": "list-secret-token"}],
            },
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        caplog.set_level(logging.INFO)

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ClusterPrimsConfigTask()
        result = task.run(context)

        assert result["cluster_prims_config"]["api_key"] == "super-secret-key"
        assert result["cluster_prims_config"]["nested"]["embedding_api_key"] == (
            "nested-secret-key"
        )
        assert "super-secret-key" not in caplog.text
        assert "nested-secret-key" not in caplog.text
        assert "list-secret-token" not in caplog.text
        assert "<redacted>" in caplog.text

    def test_logs_redact_credential_bearing_dataset_path(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        sentinel = "api_key=cluster-dataset-path-713"
        dataset_path = tmp_path / sentinel / "dataset.jsonl"
        caplog.set_level(logging.INFO)

        result = ClusterPrimsConfigTask().run(
            {
                "config_path": str(tmp_path / "config.yaml"),
                "config_dict": {
                    "dataset_path": str(dataset_path),
                    "working_dir": str(tmp_path / "work"),
                },
            }
        )

        assert result["dataset_path"] == str(dataset_path)
        assert sentinel not in caplog.text
        assert "[cluster_prims] dataset: <redacted>" in caplog.text

    def test_raises_on_missing_dataset_path(self, tmp_path: Path) -> None:
        config = {"working_dir": "/some/workdir"}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ClusterPrimsConfigTask()
        with pytest.raises(ValueError, match="dataset_path is required"):
            task.run(context)

    def test_raises_on_missing_working_dir(self, tmp_path: Path) -> None:
        config = {"dataset_path": "/some/dataset.jsonl"}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ClusterPrimsConfigTask()
        with pytest.raises(ValueError, match="working_dir is required"):
            task.run(context)

    def test_raises_on_empty_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ClusterPrimsConfigTask()
        with pytest.raises(ValueError, match="Empty config"):
            task.run(context)

    def test_raises_on_missing_config_file(self, tmp_path: Path) -> None:
        context: dict[str, Any] = {"config_path": str(tmp_path / "nope.yaml")}
        task = ClusterPrimsConfigTask()
        with pytest.raises(FileNotFoundError):
            task.run(context)


class TestExpandClusterPredictionsConfigTask:
    def test_loads_config_when_cluster_ran(self, tmp_path: Path) -> None:
        config = {
            "cluster_prims_ran": True,
            "predictions_path": "/pred.jsonl",
            "cluster_map_path": "/map.jsonl",
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ExpandClusterPredictionsConfigTask()
        result = task.run(context)

        assert result["predictions_path"] == "/pred.jsonl"
        assert result["cluster_map_path"] == "/map.jsonl"

    def test_resolves_relative_paths_against_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "configs" / "expand.yaml"
        config_path.parent.mkdir()
        config_path.write_text(
            yaml.safe_dump(
                {
                    "cluster_prims_ran": True,
                    "predictions_path": "predictions/out.jsonl",
                    "cluster_map_path": "clusters/map.jsonl",
                }
            ),
            encoding="utf-8",
        )

        result = ExpandClusterPredictionsConfigTask().run(
            {"config_path": str(config_path)}
        )

        assert result["predictions_path"] == str(
            config_path.parent / "predictions" / "out.jsonl"
        )
        assert result["cluster_map_path"] == str(
            config_path.parent / "clusters" / "map.jsonl"
        )

    def test_logs_redact_credential_bearing_result_paths(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        predictions_sentinel = "api_key=cluster-predictions-path-713"
        cluster_map_sentinel = "token=cluster-map-path-713"
        predictions_path = tmp_path / predictions_sentinel / "predictions.jsonl"
        cluster_map_path = tmp_path / cluster_map_sentinel / "cluster-map.jsonl"
        caplog.set_level(logging.INFO)

        result = ExpandClusterPredictionsConfigTask().run(
            {
                "config_path": str(tmp_path / "config.yaml"),
                "config_dict": {
                    "cluster_prims_ran": True,
                    "predictions_path": str(predictions_path),
                    "cluster_map_path": str(cluster_map_path),
                },
            }
        )

        assert result["predictions_path"] == str(predictions_path)
        assert result["cluster_map_path"] == str(cluster_map_path)
        assert predictions_sentinel not in caplog.text
        assert cluster_map_sentinel not in caplog.text
        assert caplog.text.count("<redacted>") == 2

    def test_raises_when_both_config_sources_are_missing(self) -> None:
        with pytest.raises(ValueError, match="config_dict or config_path"):
            ExpandClusterPredictionsConfigTask().run({})

    def test_skips_when_cluster_not_ran(self, tmp_path: Path) -> None:
        config = {"cluster_prims_ran": False}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ExpandClusterPredictionsConfigTask()
        result = task.run(context)

        assert result["cluster_prims_ran"] is False
        assert "predictions_path" not in result

    def test_raises_on_missing_predictions_path(self, tmp_path: Path) -> None:
        config = {
            "cluster_prims_ran": True,
            "cluster_map_path": "/map.jsonl",
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ExpandClusterPredictionsConfigTask()
        with pytest.raises(ValueError, match="predictions_path is required"):
            task.run(context)

    def test_raises_on_missing_cluster_map_path(self, tmp_path: Path) -> None:
        config = {
            "cluster_prims_ran": True,
            "predictions_path": "/pred.jsonl",
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ExpandClusterPredictionsConfigTask()
        with pytest.raises(ValueError, match="cluster_map_path is required"):
            task.run(context)
