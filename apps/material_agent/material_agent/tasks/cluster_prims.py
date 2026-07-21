# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prim-level visual deduplication via image embedding clustering.

Inserted between build_dataset_prepare_dataset and predict.

ClusterPrimsTask:
    - Reads dataset.jsonl (all prims)
    - Embeds prim_only images via the configured image embedding model
    - Computes edge-density image complexity score
    - Clusters prims with complexity-aware cosine thresholds
    - Writes clusters/cluster_map.jsonl (every prim → cluster metadata)
    - Writes clusters/dataset_representatives.jsonl (representatives only)

ExpandClusterPredictionsTask:
    - Reads predictions/predictions.jsonl (representatives only)
    - Reads clusters/cluster_map.jsonl
    - Propagates each representative's prediction to all cluster members
    - Writes predictions/predictions.jsonl (all prims)
"""

from __future__ import annotations

import concurrent.futures
import copy
import io
import json
import logging
import os
import re
import time
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import (
    get_nim_api_key_for_base_url,
    resolve_endpoint_api_key,
)

from material_agent.api.defaults import (
    DEFAULT_CLUSTER_BATCH_SIZE,
    DEFAULT_CLUSTER_COMPLEXITY_THRESHOLDS,
    DEFAULT_CLUSTER_EMBEDDING_BACKEND,
    DEFAULT_CLUSTER_EMBEDDING_MODEL,
    DEFAULT_CLUSTER_EMBEDDING_RETRIES,
    DEFAULT_CLUSTER_EMBEDDING_RETRY_BACKOFF,
    DEFAULT_CLUSTER_EMBEDDING_RETRY_INITIAL_DELAY,
    DEFAULT_CLUSTER_MAX_SIZE,
    DEFAULT_CLUSTER_MAX_WORKERS,
    DEFAULT_CLUSTER_MIN_PRIMS_TO_ACTIVATE,
    DEFAULT_CLUSTER_NIM_EMBEDDING_MODEL,
    DEFAULT_CLUSTER_REPORT_MAX_MEMBERS_PER_CLUSTER,
    DEFAULT_CLUSTER_REPORT_MAX_MULTI_MEMBER_CLUSTERS,
    DEFAULT_CLUSTER_REPORT_MAX_SINGLETONS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Complexity tiers: (edge_density_min, edge_density_max, cosine_sim_threshold)
# ---------------------------------------------------------------------------
DEFAULT_COMPLEXITY_THRESHOLDS: dict[str, tuple[float, float, float]] = {
    tier: (float(values[0]), float(values[1]), float(values[2]))
    for tier, values in DEFAULT_CLUSTER_COMPLEXITY_THRESHOLDS.items()
}

DEFAULT_EMBEDDING_MODEL = DEFAULT_CLUSTER_EMBEDDING_MODEL
DEFAULT_NIM_EMBEDDING_MODEL = DEFAULT_CLUSTER_NIM_EMBEDDING_MODEL
DEFAULT_EMBEDDING_SERVICE = DEFAULT_CLUSTER_EMBEDDING_BACKEND
DEFAULT_BATCH_SIZE = DEFAULT_CLUSTER_BATCH_SIZE
DEFAULT_MAX_WORKERS = DEFAULT_CLUSTER_MAX_WORKERS
DEFAULT_MIN_PRIMS_TO_ACTIVATE = DEFAULT_CLUSTER_MIN_PRIMS_TO_ACTIVATE
DEFAULT_MAX_CLUSTER_SIZE = DEFAULT_CLUSTER_MAX_SIZE
DEFAULT_REPORT_MAX_MULTI_MEMBER_CLUSTERS = (
    DEFAULT_CLUSTER_REPORT_MAX_MULTI_MEMBER_CLUSTERS
)
DEFAULT_REPORT_MAX_MEMBERS_PER_CLUSTER = DEFAULT_CLUSTER_REPORT_MAX_MEMBERS_PER_CLUSTER
DEFAULT_REPORT_MAX_SINGLETONS = DEFAULT_CLUSTER_REPORT_MAX_SINGLETONS
DEFAULT_EMBEDDING_RETRIES = DEFAULT_CLUSTER_EMBEDDING_RETRIES
DEFAULT_EMBEDDING_RETRY_INITIAL_DELAY = DEFAULT_CLUSTER_EMBEDDING_RETRY_INITIAL_DELAY
DEFAULT_EMBEDDING_RETRY_BACKOFF = DEFAULT_CLUSTER_EMBEDDING_RETRY_BACKOFF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_cv2_warned = False


def _edge_density(image_path: str | Path) -> float:
    """Fraction of pixels that are Canny edges (complexity proxy)."""
    global _cv2_warned  # noqa: PLW0603
    try:
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(np.count_nonzero(edges)) / edges.size
    except ImportError:
        if not _cv2_warned:
            logger.warning(
                "cv2 (opencv-python-headless) is not installed — "
                "edge density will be 0 and all prims will be classified "
                "as 'low' complexity tier"
            )
            _cv2_warned = True
        return 0.0
    except Exception:
        return 0.0


def _complexity_tier(
    score: float,
    thresholds: dict[str, tuple[float, float, float]],
) -> str:
    for tier, (lo, hi, _) in thresholds.items():
        if lo <= score < hi:
            return tier
    return "high"


def _tier_css_slug(tier: str) -> str:
    """Return a safe suffix for tier-specific CSS classes."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", tier.lower()).strip("-")
    return slug or "unknown"


def _embed_batch(
    model: Any,
    image_paths: list[str],
) -> list[np.ndarray]:
    """Embed a batch of images, return list of 1-D numpy arrays."""
    from PIL import Image as PILImage

    images = [PILImage.open(p).convert("RGB") for p in image_paths]
    return model.embed_images(images)


def _default_embedding_model_for_service(service: str) -> str:
    """Return a valid default model for the selected embedding service."""
    if service == "nim":
        return DEFAULT_NIM_EMBEDDING_MODEL
    if service == "mock":
        return DEFAULT_EMBEDDING_MODEL
    return DEFAULT_EMBEDDING_MODEL


def _validate_positive_int(name: str, value: Any) -> int:
    """Parse and validate a positive integer config value."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _validate_optional_positive_int(name: str, value: Any) -> int | None:
    """Parse an optional positive integer config value."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    parsed = _validate_positive_int(name, value)
    return parsed


def _validate_optional_non_negative_int(name: str, value: Any) -> int | None:
    """Parse an optional non-negative integer config value."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _normalize_thresholds(
    thresholds_cfg: dict[str, Any],
) -> dict[str, tuple[float, float, float]]:
    """Validate and normalize complexity thresholds.

    Expected shape is {tier: [edge_min, edge_max, cosine_similarity]}.
    """
    eps = 1e-9
    thresholds: dict[str, tuple[float, float, float]] = {}
    for tier, raw in thresholds_cfg.items():
        if not isinstance(raw, list | tuple) or len(raw) != 3:
            raise ValueError(
                "complexity_thresholds values must be "
                "[edge_min, edge_max, cosine_similarity]"
            )
        lo, hi, sim = (float(raw[0]), float(raw[1]), float(raw[2]))
        if lo < 0.0 or hi <= lo:
            raise ValueError(
                f"Invalid complexity range for tier {tier!r}: edge_min={lo}, edge_max={hi}"
            )
        if not 0.0 <= sim <= 1.0:
            raise ValueError(
                f"Invalid similarity threshold for tier {tier!r}: {sim}. "
                "Expected a value in [0.0, 1.0]."
            )
        thresholds[str(tier)] = (lo, hi, sim)
    if not thresholds:
        raise ValueError("complexity_thresholds must define at least one tier")

    ordered = sorted(thresholds.items(), key=lambda item: item[1][0])
    first_tier, (first_lo, _, _) = ordered[0]
    if abs(first_lo) > eps:
        raise ValueError(
            "complexity_thresholds must start at 0.0; "
            f"tier {first_tier!r} starts at {first_lo}"
        )

    prev_tier, (_, prev_hi, _) = ordered[0]
    for tier, (lo, hi, _) in ordered[1:]:
        if lo > prev_hi + eps:
            raise ValueError(
                "Gap in complexity_thresholds between "
                f"tier {prev_tier!r} ending at {prev_hi} and "
                f"tier {tier!r} starting at {lo}"
            )
        if lo < prev_hi - eps:
            raise ValueError(
                "Overlapping complexity_thresholds between "
                f"tier {prev_tier!r} ending at {prev_hi} and "
                f"tier {tier!r} starting at {lo}"
            )
        prev_tier, prev_hi = tier, hi

    if prev_hi < 1.0 - eps:
        raise ValueError(
            "complexity_thresholds must cover edge density values up to 1.0; "
            f"last tier {prev_tier!r} ends at {prev_hi}"
        )

    return dict(ordered)


def _cluster_by_tier(
    embeddings: np.ndarray,
    complexities: np.ndarray,
    thresholds: dict[str, tuple[float, float, float]],
) -> np.ndarray:
    """Cluster prims within each complexity tier; return global cluster labels."""
    from sklearn.cluster import AgglomerativeClustering

    labels = np.full(len(embeddings), -1, dtype=int)
    cluster_id = 0
    max_hi = max(hi for _, hi, _ in thresholds.values())

    for _tier, (lo, hi, sim_thresh) in thresholds.items():
        # Use inclusive upper bound for the highest tier
        if hi >= max_hi:
            mask = (complexities >= lo) & (complexities <= hi)
        else:
            mask = (complexities >= lo) & (complexities < hi)
        n = int(mask.sum())
        if n == 0:
            continue

        tier_emb = embeddings[mask]
        dist_thresh = 1.0 - sim_thresh

        if n == 1:
            # Single prim — its own cluster
            labels[np.where(mask)[0][0]] = cluster_id
            cluster_id += 1
            continue

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_thresh,
            metric="cosine",
            linkage="average",
        )
        tier_labels = clustering.fit_predict(tier_emb)

        for local_id in np.unique(tier_labels):
            global_idx = np.where(mask)[0][tier_labels == local_id]
            labels[global_idx] = cluster_id
            cluster_id += 1

    return labels


def _split_large_clusters(
    labels: np.ndarray, max_cluster_size: int | None
) -> np.ndarray:
    """Split clusters larger than max_cluster_size into deterministic chunks."""
    if max_cluster_size is None:
        return labels

    new_labels = np.full(len(labels), -1, dtype=int)
    next_cluster_id = 0
    for cid in np.unique(labels):
        idxs = np.where(labels == cid)[0]
        for start in range(0, len(idxs), max_cluster_size):
            chunk = idxs[start : start + max_cluster_size]
            new_labels[chunk] = next_cluster_id
            next_cluster_id += 1
    return new_labels


def _select_representatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[int, int]:
    """Return {cluster_id: index_of_representative}."""
    from sklearn.metrics.pairwise import cosine_similarity

    reps: dict[int, int] = {}
    for cid in np.unique(labels):
        mask = labels == cid
        idxs = np.where(mask)[0]
        if len(idxs) == 1:
            reps[int(cid)] = int(idxs[0])
            continue
        centroid = embeddings[mask].mean(axis=0, keepdims=True)
        sims = cosine_similarity(embeddings[mask], centroid).ravel()
        reps[int(cid)] = int(idxs[np.argmax(sims)])
    return reps


# ---------------------------------------------------------------------------
# ClusterPrimsTask
# ---------------------------------------------------------------------------


class ClusterPrimsTask(Task):
    """Cluster prims visually; write cluster map and representative dataset.

    Input context keys:
        - dataset_path: Path to dataset.jsonl (all prims)
        - cluster_prims_config: Config dict with embedding + threshold settings
        - working_dir: Pipeline working directory

    Output context keys:
        - cluster_map_path: Path to clusters/cluster_map.jsonl
        - dataset_representatives_path: Path to clusters/dataset_representatives.jsonl
        - cluster_prims_ran: True (so downstream steps know to use representative dataset)
    """

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        def _emit_progress(
            current: int,
            total: int,
            message: str,
            **extra: Any,
        ) -> None:
            try:
                listener.event(
                    "task.progress",
                    {
                        "task_name": "cluster_prims",
                        "step_name": "cluster_prims",
                        "current": current,
                        "total": total,
                        "percent": int(current * 100 / total),
                        "message": message,
                        **extra,
                    },
                )
            except Exception:
                logger.debug("Failed to emit cluster_prims progress", exc_info=True)

        config: dict[str, Any] = context.get("cluster_prims_config", {})
        dataset_path = Path(context["dataset_path"])
        working_dir = Path(context.get("working_dir", Path.cwd()))
        clusters_dir = working_dir / "clusters"
        clusters_dir.mkdir(parents=True, exist_ok=True)

        cluster_map_path = clusters_dir / "cluster_map.jsonl"
        representatives_path = clusters_dir / "dataset_representatives.jsonl"
        summary_path = clusters_dir / "cluster_summary.json"

        # --- Load dataset ---
        dataset: list[dict[str, Any]] = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    dataset.append(json.loads(line))

        n = len(dataset)
        _emit_progress(1, 6, "Loaded clustering dataset", prim_count=n)
        min_prims = _validate_positive_int(
            "min_prims_to_activate",
            config.get("min_prims_to_activate", DEFAULT_MIN_PRIMS_TO_ACTIVATE),
        )

        if n < min_prims:
            listener.info(
                f"[cluster_prims] {n} prims < min_prims_to_activate={min_prims} — skipping"
            )
            context["cluster_prims_ran"] = False
            return context

        listener.info(f"[cluster_prims] Clustering {n} prims...")

        # Dataset images may be relative to the dataset directory
        dataset_dir = dataset_path.parent

        def _absolutize_paths(entry: dict[str, Any]) -> dict[str, Any]:
            """Return a copy with image paths absolute for relocated datasets."""
            # v0.2 schema: media.images[*].path
            media = entry.get("media")
            if isinstance(media, dict):
                images = media.get("images", [])
                if images:
                    entry = copy.deepcopy(entry)
                    for img in entry["media"]["images"]:
                        raw = img.get("path", "")
                        if raw and not Path(raw).is_absolute():
                            img["path"] = str((dataset_dir / raw).resolve())
                return entry

            # v0.1 schema: images.<render_mode> → list[str]
            images_dict = entry.get("images")
            if isinstance(images_dict, dict):
                entry = copy.deepcopy(entry)
                for key, paths in entry["images"].items():
                    if not isinstance(paths, list):
                        continue
                    entry["images"][key] = [
                        str((dataset_dir / p).resolve())
                        if p and not Path(p).is_absolute()
                        else p
                        for p in paths
                    ]
            return entry

        def _resolve_prim_only_paths(entry: dict[str, Any]) -> list[str]:
            """Extract prim_only image paths from either v0.1 or v0.2 schema."""
            paths: list[str] = []

            # v0.2 schema: media.images[*] with metadata.render_mode
            media = entry.get("media")
            if isinstance(media, dict):
                for img in media.get("images", []):
                    if img.get("metadata", {}).get("render_mode") == "prim_only":
                        raw = img.get("path", "")
                        p = Path(raw) if Path(raw).is_absolute() else dataset_dir / raw
                        if p.exists():
                            paths.append(str(p))
                return paths

            # v0.1 schema: images.prim_only list of paths
            images = entry.get("images", {})
            for raw in images.get("prim_only", []):
                p = Path(raw) if Path(raw).is_absolute() else dataset_dir / raw
                if p.exists():
                    paths.append(str(p))
            return paths

        # --- Collect prim_only image paths (average across views per prim) ---
        prim_image_paths: list[list[str]] = [
            _resolve_prim_only_paths(entry) for entry in dataset
        ]

        # --- Split: prims with images vs without ---
        has_images = [bool(paths) for paths in prim_image_paths]
        img_indices = [i for i, h in enumerate(has_images) if h]
        no_img_indices = [i for i, h in enumerate(has_images) if not h]
        listener.info(
            f"[cluster_prims] {len(img_indices)} prims have prim_only images, "
            f"{len(no_img_indices)} have none (will be singletons)"
        )

        thresholds_cfg = config.get(
            "complexity_thresholds", DEFAULT_COMPLEXITY_THRESHOLDS
        )
        thresholds = _normalize_thresholds(thresholds_cfg)
        max_cluster_size = _validate_optional_positive_int(
            "max_cluster_size", config.get("max_cluster_size", DEFAULT_MAX_CLUSTER_SIZE)
        )
        if max_cluster_size is not None:
            listener.info(
                f"[cluster_prims] Max cluster size cap: {max_cluster_size} prims"
            )

        if not img_indices:
            listener.info(
                "[cluster_prims] No prim_only images found; writing singleton "
                "clusters without embedding"
            )
            _emit_progress(2, 6, "No prim-only images; using singleton clusters")
            with open(cluster_map_path, "w", encoding="utf-8") as fout:
                for i, entry in enumerate(dataset):
                    row = {
                        "id": entry["id"],
                        "cluster_id": i,
                        "is_representative": True,
                        "cluster_representative_id": entry["id"],
                        "cluster_size": 1,
                        "complexity_score": 0.0,
                        "complexity_tier": _complexity_tier(0.0, thresholds),
                    }
                    fout.write(json.dumps(row) + "\n")

            with open(representatives_path, "w", encoding="utf-8") as fout:
                for entry in dataset:
                    fout.write(json.dumps(_absolutize_paths(entry)) + "\n")

            import shutil

            dataset_config = dataset_dir / "dataset.json"
            if dataset_config.exists():
                shutil.copy2(dataset_config, clusters_dir / "dataset.json")

            self._write_cluster_summary(
                summary_path=summary_path,
                labels=np.arange(n, dtype=int),
                complexities=np.zeros(n, dtype=float),
                thresholds=thresholds,
                total_prims=n,
                representative_count=n,
                reduction_pct=0.0,
                max_cluster_size=max_cluster_size,
                capped_cluster_count=0,
                report_limits=None,
            )
            context["cluster_map_path"] = str(cluster_map_path)
            context["dataset_representatives_path"] = str(representatives_path)
            context["cluster_summary_path"] = str(summary_path)
            context["cluster_prims_ran"] = True
            context["cluster_total_prims"] = n
            context["cluster_count"] = n
            context["cluster_representative_count"] = n
            context["cluster_reduction_percent"] = 0.0
            context["cluster_multi_member_count"] = 0
            context["cluster_singleton_count"] = n
            context["cluster_max_size"] = max_cluster_size
            context["cluster_capped_count"] = 0
            listener.info(
                f"[cluster_prims] Singleton cluster map → {cluster_map_path} "
                f"| Representatives ({n}) → {representatives_path}"
            )
            _emit_progress(6, 6, "Wrote singleton cluster artifacts")
            return context

        # --- Compute edge density complexity (image prims only) ---
        listener.info("[cluster_prims] Computing image complexity (edge density)...")
        _emit_progress(2, 6, "Computing image complexity")
        complexities_all = np.zeros(n, dtype=float)
        for i in img_indices:
            scores = [_edge_density(p) for p in prim_image_paths[i]]
            complexities_all[i] = float(np.mean(scores))

        # --- Build embedding model ---
        service = str(
            config.get("embedding_service", DEFAULT_EMBEDDING_SERVICE)
        ).lower()
        model_name = config.get(
            "embedding_model"
        ) or _default_embedding_model_for_service(service)
        base_url = config.get("base_url") or config.get("embedding_base_url")
        explicit_api_key = resolve_endpoint_api_key(
            (
                config.get("api_key")
                or config.get("embedding_api_key")
                or os.environ.get("MA_CLUSTER_EMBEDDING_API_KEY")
            ),
            (
                config.get("api_key_env")
                or config.get("embedding_api_key_env")
                or os.environ.get("MA_CLUSTER_EMBEDDING_API_KEY_ENV")
            ),
        )
        api_key = (
            get_nim_api_key_for_base_url(base_url, explicit_api_key)
            if service == "nim"
            else explicit_api_key
        )
        batch_size = _validate_positive_int(
            "batch_size", config.get("batch_size", DEFAULT_BATCH_SIZE)
        )
        max_workers = _validate_positive_int(
            "max_workers", config.get("max_workers", DEFAULT_MAX_WORKERS)
        )
        embedding_retries = int(
            config.get("embedding_retries", DEFAULT_EMBEDDING_RETRIES)
        )
        retry_delay = float(
            config.get(
                "embedding_retry_initial_delay",
                DEFAULT_EMBEDDING_RETRY_INITIAL_DELAY,
            )
        )
        retry_backoff = float(
            config.get("embedding_retry_backoff", DEFAULT_EMBEDDING_RETRY_BACKOFF)
        )

        listener.info(
            f"[cluster_prims] Embedding {len(img_indices)} prims with {model_name} "
            f"(batch={batch_size}, workers={max_workers})..."
        )
        _emit_progress(
            3,
            6,
            "Embedding prim-only images",
            embedding_model=model_name,
            embedding_backend=service,
            image_prim_count=len(img_indices),
        )

        from world_understanding.functions.models.image_embedding_models import (
            create_image_embedding_model,
        )

        embed_model = create_image_embedding_model(
            backend=service,
            api_key=api_key,
            model=model_name,
            **({"base_url": base_url} if base_url else {}),
        )

        # --- Embed: average prim_only views per image-prim ---
        flat_requests: list[tuple[int, str]] = []
        for i in img_indices:
            for p in prim_image_paths[i]:
                flat_requests.append((i, p))

        batches = [
            flat_requests[s : s + batch_size]
            for s in range(0, len(flat_requests), batch_size)
        ]

        prim_embeddings: dict[int, list[np.ndarray]] = {i: [] for i in img_indices}

        def _embed_batch_worker(
            batch: list[tuple[int, str]],
        ) -> list[tuple[int, np.ndarray]]:
            idxs = [b[0] for b in batch]
            paths = [b[1] for b in batch]
            delay = retry_delay
            attempts = max(1, embedding_retries)
            for attempt in range(1, attempts + 1):
                try:
                    vecs = _embed_batch(embed_model, paths)
                    return list(zip(idxs, vecs, strict=False))
                except Exception as exc:
                    if attempt >= attempts:
                        raise
                    listener.warning(
                        f"[cluster_prims] Embedding batch failed on attempt "
                        f"{attempt}/{attempts}: {exc}; retrying after {delay:.1f}s"
                    )
                    time.sleep(delay)
                    delay *= retry_backoff

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_embed_batch_worker, b) for b in batches]
            for done_i, fut in enumerate(concurrent.futures.as_completed(futures)):
                for prim_idx, vec in fut.result():
                    prim_embeddings[prim_idx].append(vec)
                if (done_i + 1) % 20 == 0:
                    listener.info(
                        f"[cluster_prims] Embedded {(done_i + 1) * batch_size}"
                        f"/{len(flat_requests)} images"
                    )

        # Average views per prim → unit-norm embedding matrix
        dim = embed_model.embedding_dimension
        embeddings_img = np.zeros((len(img_indices), dim), dtype=float)
        for row, i in enumerate(img_indices):
            vecs = prim_embeddings[i]
            if vecs:
                embeddings_img[row] = np.mean(vecs, axis=0)
            norm = np.linalg.norm(embeddings_img[row])
            if norm > 0:
                embeddings_img[row] /= norm

        complexities_img = complexities_all[img_indices]

        # --- Cluster image-prims ---
        listener.info("[cluster_prims] Clustering embeddings by complexity tier...")
        _emit_progress(4, 6, "Clustering image embeddings")
        labels_img = _cluster_by_tier(embeddings_img, complexities_img, thresholds)
        capped_cluster_count = 0
        if max_cluster_size is not None:
            capped_cluster_count = sum(
                1
                for cid in np.unique(labels_img)
                if int((labels_img == cid).sum()) > max_cluster_size
            )
            if capped_cluster_count:
                before = len(np.unique(labels_img))
                labels_img = _split_large_clusters(labels_img, max_cluster_size)
                after = len(np.unique(labels_img))
                listener.info(
                    f"[cluster_prims] Split {capped_cluster_count} oversized "
                    f"clusters at max_cluster_size={max_cluster_size}: "
                    f"{before} → {after} image clusters"
                )
        reps_img = _select_representatives(embeddings_img, labels_img)

        # --- Merge: assign global labels ---
        # image prims → clustered labels (offset from 0)
        # no-image prims → each gets a unique singleton cluster
        labels = np.full(n, -1, dtype=int)
        for row, i in enumerate(img_indices):
            labels[i] = int(labels_img[row])

        n_img_clusters = int(labels_img.max()) + 1 if len(labels_img) > 0 else 0
        complexities = complexities_all  # alias for downstream code
        for j, i in enumerate(no_img_indices):
            labels[i] = n_img_clusters + j

        # Remap reps to global indices
        reps: dict[int, int] = {}
        for cid, row in reps_img.items():
            reps[cid] = img_indices[row]
        for j, i in enumerate(no_img_indices):
            reps[n_img_clusters + j] = i

        n_clusters = len(reps)
        reduction_pct = 100.0 * (1 - n_clusters / n) if n > 0 else 0.0
        listener.info(
            f"[cluster_prims] {n} prims → {n_clusters} clusters "
            f"({reduction_pct:.1f}% reduction)"
        )

        # Log per-tier breakdown
        max_hi = max(hi for _, hi, _ in thresholds.values())
        for tier, (lo, hi, sim_thresh) in thresholds.items():
            if hi >= max_hi:
                mask = (complexities >= lo) & (complexities <= hi)
            else:
                mask = (complexities >= lo) & (complexities < hi)
            tier_n = int(mask.sum())
            tier_clusters = len(np.unique(labels[mask])) if tier_n > 0 else 0
            listener.info(
                f"  {tier:8s}: {tier_n:5d} prims → {tier_clusters:4d} clusters "
                f"(sim_threshold={sim_thresh})"
            )

        # --- Build cluster map ---
        n_total_reps = len(set(reps.values()))
        _emit_progress(
            5,
            6,
            "Writing cluster artifacts",
            cluster_count=n_clusters,
            representative_count=n_total_reps,
            reduction_percent=round(reduction_pct, 3),
        )
        rep_prim_id: dict[int, str] = {
            cid: dataset[idx]["id"] for cid, idx in reps.items()
        }

        with open(cluster_map_path, "w", encoding="utf-8") as fout:
            for i, entry in enumerate(dataset):
                cid = int(labels[i])
                is_rep = reps.get(cid) == i
                tier = _complexity_tier(float(complexities[i]), thresholds)
                row = {
                    "id": entry["id"],
                    "cluster_id": cid,
                    "is_representative": is_rep,
                    "cluster_representative_id": rep_prim_id[cid],
                    "cluster_size": int((labels == cid).sum()),
                    "complexity_score": round(float(complexities[i]), 6),
                    "complexity_tier": tier,
                }
                fout.write(json.dumps(row) + "\n")

        # --- Write representative-only dataset ---
        # Absolutize image paths so predict can resolve them regardless of
        # where dataset_representatives.jsonl lives (clusters/ vs dataset/).
        # Write ALL representatives (image-prim and no-image singletons) so
        # that predict can generate predictions for every cluster.  No-image
        # singletons still carry the system prompt / reference images which
        # is enough for mock and some real VLM backends.
        all_rep_indices = set(reps.values())
        with open(representatives_path, "w", encoding="utf-8") as fout:
            for i, entry in enumerate(dataset):
                if i in all_rep_indices:
                    fout.write(json.dumps(_absolutize_paths(entry)) + "\n")

        n_total_reps = len(all_rep_indices)

        # Copy dataset.json into clusters/ so the predict config task can find
        # the system prompt (it looks for dataset.json next to the .jsonl file).
        import shutil

        dataset_config = dataset_dir / "dataset.json"
        if dataset_config.exists():
            shutil.copy2(dataset_config, clusters_dir / "dataset.json")

        listener.info(
            f"[cluster_prims] Cluster map → {cluster_map_path} "
            f"| Representatives ({n_total_reps}) → {representatives_path}"
        )

        report_config_raw = config.get("report", {})
        report_config = report_config_raw if isinstance(report_config_raw, dict) else {}
        report_limits: dict[str, int | None] | None = None
        if report_config_raw is not False and report_config_raw is not None:
            report_limits = {
                "max_multi_member_clusters": _validate_optional_non_negative_int(
                    "report.max_multi_member_clusters",
                    report_config.get(
                        "max_multi_member_clusters",
                        DEFAULT_REPORT_MAX_MULTI_MEMBER_CLUSTERS,
                    ),
                ),
                "max_members_per_cluster": _validate_optional_non_negative_int(
                    "report.max_members_per_cluster",
                    report_config.get(
                        "max_members_per_cluster",
                        DEFAULT_REPORT_MAX_MEMBERS_PER_CLUSTER,
                    ),
                ),
                "max_singletons": _validate_optional_non_negative_int(
                    "report.max_singletons",
                    report_config.get(
                        "max_singletons",
                        DEFAULT_REPORT_MAX_SINGLETONS,
                    ),
                ),
            }

        self._write_cluster_summary(
            summary_path=summary_path,
            labels=labels,
            complexities=complexities,
            thresholds=thresholds,
            total_prims=n,
            representative_count=n_total_reps,
            reduction_pct=reduction_pct,
            max_cluster_size=max_cluster_size,
            capped_cluster_count=int(capped_cluster_count),
            report_limits=report_limits,
        )

        # --- Generate HTML report ---
        if report_config_raw is not False and report_config_raw is not None:
            if report_config.get("enabled", True):
                report_path = clusters_dir / "cluster_report.html"
                self._generate_html_report(
                    report_path=report_path,
                    dataset=dataset,
                    labels=labels,
                    reps=reps,
                    complexities=complexities,
                    thresholds=thresholds,
                    prim_image_paths=prim_image_paths,
                    n_clusters=n_clusters,
                    reduction_pct=reduction_pct,
                    image_max_size=report_config.get("image_max_size", 128),
                    image_format=report_config.get("image_format", "jpeg"),
                    image_quality=report_config.get("image_quality", 75),
                    max_multi_member_clusters=(report_limits or {}).get(
                        "max_multi_member_clusters"
                    ),
                    max_members_per_cluster=(report_limits or {}).get(
                        "max_members_per_cluster"
                    ),
                    max_singletons=(report_limits or {}).get("max_singletons"),
                    listener=listener,
                )
                context["cluster_report_path"] = str(report_path)

        multi_member_count = sum(
            1 for cid in np.unique(labels) if (labels == cid).sum() > 1
        )
        context["cluster_map_path"] = str(cluster_map_path)
        context["dataset_representatives_path"] = str(representatives_path)
        context["cluster_summary_path"] = str(summary_path)
        context["cluster_prims_ran"] = True
        context["cluster_total_prims"] = n
        context["cluster_count"] = n_clusters
        context["cluster_representative_count"] = n_total_reps
        context["cluster_reduction_percent"] = round(reduction_pct, 3)
        context["cluster_multi_member_count"] = int(multi_member_count)
        context["cluster_singleton_count"] = int(n_clusters - multi_member_count)
        context["cluster_max_size"] = max_cluster_size
        context["cluster_capped_count"] = int(capped_cluster_count)
        _emit_progress(
            6,
            6,
            "Completed prim clustering",
            cluster_count=n_clusters,
            representative_count=n_total_reps,
            reduction_percent=round(reduction_pct, 3),
            capped_cluster_count=int(capped_cluster_count),
        )
        return context

    @staticmethod
    def _encode_image(
        img_path: str,
        max_size: int = 128,
        fmt: str = "jpeg",
        quality: int = 75,
    ) -> str | None:
        """Load, resize, and base64-encode an image."""
        try:
            from PIL import Image as PILImage

            img = PILImage.open(img_path)
            if fmt == "jpeg" and img.mode in ("RGBA", "LA", "P"):
                bg = PILImage.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = bg
            img.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)
            buf = io.BytesIO()
            save_fmt = "JPEG" if fmt == "jpeg" else "PNG"
            img.save(buf, format=save_fmt, quality=quality, optimize=True)
            buf.seek(0)
            import base64

            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return None

    @staticmethod
    def _write_cluster_summary(
        *,
        summary_path: Path,
        labels: np.ndarray,
        complexities: np.ndarray,
        thresholds: dict[str, tuple[float, float, float]],
        total_prims: int,
        representative_count: int,
        reduction_pct: float,
        max_cluster_size: int | None,
        capped_cluster_count: int,
        report_limits: dict[str, int | None] | None,
    ) -> None:
        """Write a lightweight JSON summary for API consumers."""
        cluster_ids = [int(cid) for cid in np.unique(labels)]
        cluster_sizes = [int((labels == cid).sum()) for cid in cluster_ids]
        multi_member_count = sum(1 for size in cluster_sizes if size > 1)
        max_hi = max(hi for _, hi, _ in thresholds.values())
        tiers: dict[str, dict[str, int | float]] = {}
        for tier, (lo, hi, sim_thresh) in thresholds.items():
            if hi >= max_hi:
                mask = (complexities >= lo) & (complexities <= hi)
            else:
                mask = (complexities >= lo) & (complexities < hi)
            tier_n = int(mask.sum())
            tiers[tier] = {
                "prim_count": tier_n,
                "cluster_count": len(np.unique(labels[mask])) if tier_n else 0,
                "similarity_threshold": float(sim_thresh),
            }

        summary = {
            "total_prims": int(total_prims),
            "cluster_count": len(cluster_ids),
            "representative_count": int(representative_count),
            "reduction_percent": round(float(reduction_pct), 3),
            "multi_member_count": int(multi_member_count),
            "singleton_count": int(len(cluster_ids) - multi_member_count),
            "max_cluster_size": max_cluster_size,
            "observed_max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "capped_cluster_count": int(capped_cluster_count),
            "tiers": tiers,
            "report_limits": report_limits or {},
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def _generate_html_report(
        self,
        report_path: Path,
        dataset: list[dict[str, Any]],
        labels: np.ndarray,
        reps: dict[int, int],
        complexities: np.ndarray,
        thresholds: dict[str, tuple[float, float, float]],
        prim_image_paths: list[list[str]],
        n_clusters: int,
        reduction_pct: float,
        image_max_size: int,
        image_format: str,
        image_quality: int,
        max_multi_member_clusters: int | None,
        max_members_per_cluster: int | None,
        max_singletons: int | None,
        listener: Any,
    ) -> None:
        """Generate a self-contained HTML report of clustering results."""
        from collections import defaultdict

        n = len(dataset)
        mime = "image/jpeg" if image_format == "jpeg" else "image/png"

        # Group prims by cluster
        clusters: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            clusters[int(labels[i])].append(i)

        # Sort clusters: multi-member first (largest first), then singletons
        sorted_cids = sorted(
            clusters.keys(),
            key=lambda c: (-len(clusters[c]), c),
        )
        multi_member_cids = [cid for cid in sorted_cids if len(clusters[cid]) >= 2]
        if max_multi_member_clusters is None:
            visible_multi_member_cids = multi_member_cids
        else:
            visible_multi_member_cids = multi_member_cids[:max_multi_member_clusters]
        omitted_multi_member_clusters = len(multi_member_cids) - len(
            visible_multi_member_cids
        )
        singleton_cids = [cid for cid in sorted_cids if len(clusters[cid]) == 1]
        if max_singletons is None:
            visible_singleton_cids = singleton_cids
        else:
            visible_singleton_cids = singleton_cids[:max_singletons]
        omitted_singletons = len(singleton_cids) - len(visible_singleton_cids)

        # Build HTML
        lines: list[str] = []
        lines.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        lines.append("<title>Cluster Prims Report</title>")
        lines.append("<style>")
        lines.append("""
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 20px; background: #fafafa; color: #333; }
h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { margin-top: 32px; }
.summary { background: #fff; border: 1px solid #ddd; border-radius: 8px;
           padding: 16px; margin: 16px 0; display: inline-block; }
.summary td { padding: 4px 16px 4px 0; }
.summary .label { font-weight: 600; }
.cluster { background: #fff; border: 1px solid #ddd; border-radius: 8px;
           padding: 16px; margin: 12px 0; }
.cluster-header { font-weight: 600; font-size: 1.1em; margin-bottom: 8px; }
.cluster-meta { color: #666; font-size: 0.9em; margin-bottom: 12px; }
.prim-row { display: flex; align-items: center; gap: 12px; padding: 6px 0;
            border-bottom: 1px solid #f0f0f0; }
.prim-row:last-child { border-bottom: none; }
.prim-images { display: flex; gap: 4px; flex-shrink: 0; }
.prim-images img { border-radius: 4px; border: 1px solid #ddd; }
.prim-id { font-family: monospace; font-size: 0.85em; word-break: break-all; }
.prim-meta { color: #888; font-size: 0.8em; }
.rep-badge { background: #2563eb; color: #fff; font-size: 0.7em;
             padding: 2px 6px; border-radius: 3px; margin-left: 6px; }
.singleton-section { margin-top: 32px; }
.singleton-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.singleton-card { background: #fff; border: 1px solid #eee; border-radius: 6px;
                  padding: 8px; width: 180px; font-size: 0.8em; }
.singleton-card img { border-radius: 3px; margin-bottom: 4px; }
.tier-low { color: #16a34a; } .tier-medium { color: #ca8a04; } .tier-high { color: #dc2626; }
.collapse-toggle { cursor: pointer; color: #2563eb; font-size: 0.85em; }
""")
        lines.append("</style></head><body>")
        lines.append("<h1>Prim Clustering Report</h1>")

        # Summary table
        lines.append("<div class='summary'><table>")
        lines.append(f"<tr><td class='label'>Total prims</td><td>{n}</td></tr>")
        lines.append(f"<tr><td class='label'>Clusters</td><td>{n_clusters}</td></tr>")
        lines.append(
            f"<tr><td class='label'>Reduction</td><td>{reduction_pct:.1f}%</td></tr>"
        )
        multi_count = sum(1 for c in clusters.values() if len(c) > 1)
        singleton_count = sum(1 for c in clusters.values() if len(c) == 1)
        lines.append(
            f"<tr><td class='label'>Multi-member clusters</td>"
            f"<td>{multi_count}</td></tr>"
        )
        lines.append(
            f"<tr><td class='label'>Singletons</td><td>{singleton_count}</td></tr>"
        )
        # Per-tier stats
        max_hi_report = max(hi for _, hi, _ in thresholds.values())
        for tier, (lo, hi, sim) in thresholds.items():
            if hi >= max_hi_report:
                mask = (complexities >= lo) & (complexities <= hi)
            else:
                mask = (complexities >= lo) & (complexities < hi)
            tier_n = int(mask.sum())
            tier_c = len({int(labels[i]) for i in range(n) if mask[i]})
            tier_label = escape(str(tier))
            lines.append(
                f"<tr><td class='label'>Tier: {tier_label}</td>"
                f"<td>{tier_n} prims → {tier_c} clusters (threshold={sim})</td></tr>"
            )
        lines.append(
            f"<tr><td class='label'>Image settings</td>"
            f"<td>{image_max_size}px, {image_format}, q={image_quality}</td></tr>"
        )
        lines.append(
            f"<tr><td class='label'>Report limits</td>"
            f"<td>multi-member clusters={max_multi_member_clusters if max_multi_member_clusters is not None else 'all'}, "
            f"members/cluster={max_members_per_cluster if max_members_per_cluster is not None else 'all'}, "
            f"singletons={max_singletons if max_singletons is not None else 'all'}</td></tr>"
        )
        lines.append("</table></div>")

        # Multi-member clusters
        lines.append("<h2>Multi-Member Clusters</h2>")
        for cid in visible_multi_member_cids:
            members = clusters[cid]
            rep_idx = reps[cid]
            if max_members_per_cluster is None:
                visible_members = members
            else:
                visible_members = members[:max_members_per_cluster]
            omitted_members = len(members) - len(visible_members)

            def _prim_id(idx: int) -> str:
                return escape(str(dataset[idx]["id"]))

            rep_tier = _complexity_tier(float(complexities[rep_idx]), thresholds)
            rep_tier_label = escape(rep_tier)
            rep_tier_slug = _tier_css_slug(rep_tier)
            lines.append("<div class='cluster'>")
            lines.append(
                f"<div class='cluster-header'>Cluster {cid} "
                f"({len(members)} prims)</div>"
            )
            lines.append(
                f"<div class='cluster-meta'>"
                f"Complexity tier: <span class='tier-{rep_tier_slug}'>"
                f"{rep_tier_label}</span>"
                f"</div>"
            )

            for idx in visible_members:
                is_rep = idx == rep_idx
                paths = prim_image_paths[idx][:2]  # Show up to 2 images
                imgs_html = ""
                for p in paths:
                    b64 = self._encode_image(
                        p, image_max_size, image_format, image_quality
                    )
                    if b64:
                        imgs_html += (
                            f"<img src='data:{mime};base64,{b64}' "
                            f"width='{image_max_size}' height='{image_max_size}'>"
                        )
                badge = "<span class='rep-badge'>REP</span>" if is_rep else ""
                lines.append(
                    f"<div class='prim-row'>"
                    f"<div class='prim-images'>{imgs_html}</div>"
                    f"<div>"
                    f"<span class='prim-id'>{_prim_id(idx)}</span>{badge}<br>"
                    f"<span class='prim-meta'>"
                    f"edge_density={complexities[idx]:.4f}"
                    f"</span></div></div>"
                )
            if omitted_members > 0:
                lines.append(
                    f"<div class='prim-meta'>"
                    f"{omitted_members} additional members omitted by report limit."
                    f"</div>"
                )
            lines.append("</div>")
        if omitted_multi_member_clusters > 0:
            lines.append(
                f"<p class='prim-meta'>{omitted_multi_member_clusters} additional "
                f"multi-member clusters omitted by report limit. Use "
                f"cluster_map.jsonl or cluster_summary.json for complete data.</p>"
            )

        # Singletons (collapsed)
        if singleton_count > 0:
            lines.append("<div class='singleton-section'>")
            lines.append(
                f"<h2><span class='collapse-toggle' "
                f'onclick="this.parentElement.nextElementSibling.style.display='
                f"this.parentElement.nextElementSibling.style.display==='none'?"
                f"'flex':'none'\">▶</span> "
                f"Singletons ({singleton_count})</h2>"
            )
            lines.append("<div class='singleton-grid' style='display:none'>")
            for cid in visible_singleton_cids:
                members = clusters[cid]
                idx = members[0]
                paths = prim_image_paths[idx][:1]
                img_html = ""
                for p in paths:
                    b64 = self._encode_image(
                        p, image_max_size, image_format, image_quality
                    )
                    if b64:
                        img_html = (
                            f"<img src='data:{mime};base64,{b64}' "
                            f"width='{image_max_size}'>"
                        )
                prim_id = escape(str(dataset[idx]["id"]))
                lines.append(
                    f"<div class='singleton-card'>{img_html}"
                    f"<div class='prim-id'>{prim_id}</div>"
                    f"<div class='prim-meta'>"
                    f"edge={complexities[idx]:.4f}</div></div>"
                )
            if omitted_singletons > 0:
                lines.append(
                    f"<div class='singleton-card'>"
                    f"{omitted_singletons} additional singletons omitted by "
                    f"report limit.</div>"
                )
            lines.append("</div></div>")

        lines.append("</body></html>")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        listener.info(f"[cluster_prims] HTML report → {report_path}")


# ---------------------------------------------------------------------------
# ExpandClusterPredictionsTask
# ---------------------------------------------------------------------------


class ExpandClusterPredictionsTask(Task):
    """Expand representative predictions to all cluster members.

    Input context keys:
        - predictions_path: Path to predictions.jsonl (representatives only)
        - cluster_map_path: Path to clusters/cluster_map.jsonl

    Output context keys:
        - predictions_path: Path to expanded predictions.jsonl (all prims, in-place overwrite)
    """

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        def _emit_progress(
            current: int,
            total: int,
            message: str,
            **extra: Any,
        ) -> None:
            try:
                listener.event(
                    "task.progress",
                    {
                        "task_name": "expand_cluster_predictions",
                        "step_name": "expand_cluster_predictions",
                        "current": current,
                        "total": total,
                        "percent": int(current * 100 / total),
                        "message": message,
                        **extra,
                    },
                )
            except Exception:
                logger.debug(
                    "Failed to emit expand_cluster_predictions progress",
                    exc_info=True,
                )

        # If cluster_prims was skipped (too few prims), nothing to expand
        if not context.get("cluster_prims_ran", False):
            listener.info(
                "[expand_cluster_predictions] cluster_prims did not run — skipping"
            )
            return context

        predictions_path = Path(context["predictions_path"])
        cluster_map_path = Path(context["cluster_map_path"])

        if not predictions_path.exists():
            raise FileNotFoundError(f"predictions_path not found: {predictions_path}")
        if not cluster_map_path.exists():
            raise FileNotFoundError(f"cluster_map_path not found: {cluster_map_path}")

        # Load predictions (representative predictions, keyed by id)
        pred_by_id: dict[str, dict[str, Any]] = {}
        with open(predictions_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    p = json.loads(line)
                    pred_by_id[p["id"]] = p

        # Load cluster map
        cluster_map: list[dict[str, Any]] = []
        with open(cluster_map_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cluster_map.append(json.loads(line))

        n_total = len(cluster_map)
        n_reps = sum(1 for row in cluster_map if row["is_representative"])
        n_pred = len(pred_by_id)

        listener.info(
            f"[expand_cluster_predictions] {n_pred} VLM predictions → "
            f"{n_total} prims (via {n_reps} cluster representatives)"
        )
        _emit_progress(
            1,
            3,
            "Loaded representative predictions and cluster map",
            prediction_count=n_pred,
            prim_count=n_total,
            representative_count=n_reps,
        )

        # Expand: for each prim, copy the representative's prediction
        expanded: list[dict[str, Any]] = []
        n_no_pred = 0  # representatives with no VLM result (no-image prims)
        missing_reps: list[str] = []  # non-representatives missing rep prediction

        for row in cluster_map:
            prim_id = row["id"]
            rep_id = row["cluster_representative_id"]

            if row["is_representative"]:
                # This prim is a representative — use its own prediction.
                # Missing = no-image prim that predict skipped; silently omit.
                pred = pred_by_id.get(prim_id)
                if pred is None:
                    n_no_pred += 1
                    continue
                expanded.append(pred)
            else:
                # Non-representative — copy from its cluster representative
                rep_pred = pred_by_id.get(rep_id)
                if rep_pred is None:
                    # Representative prediction missing — unexpected, warn
                    missing_reps.append(prim_id)
                    continue
                # Clone prediction with this prim's id (deep copy to avoid
                # sharing nested dicts like "materials" across members)
                member_pred = copy.deepcopy(rep_pred)
                member_pred["id"] = prim_id
                member_pred["prediction_source"] = "cluster_representative"
                member_pred["cluster_representative_id"] = rep_id
                member_pred["cluster_id"] = row["cluster_id"]
                expanded.append(member_pred)

        _emit_progress(
            2,
            3,
            "Expanded cluster representative predictions",
            expanded_prediction_count=len(expanded),
            missing_representative_count=len(missing_reps),
            skipped_no_image_count=n_no_pred,
        )

        if n_no_pred:
            listener.info(
                f"[expand_cluster_predictions] {n_no_pred} no-image prims skipped "
                f"(no VLM prediction, expected)"
            )
        if missing_reps:
            listener.warning(
                f"[expand_cluster_predictions] {len(missing_reps)} non-representative "
                f"prims missing their rep's prediction (first 5: {missing_reps[:5]})"
            )

        # Overwrite predictions.jsonl with expanded predictions
        with open(predictions_path, "w", encoding="utf-8") as fout:
            for pred in expanded:
                fout.write(json.dumps(pred) + "\n")

        listener.info(
            f"[expand_cluster_predictions] Wrote {len(expanded)} predictions → {predictions_path}"
        )
        _emit_progress(
            3,
            3,
            "Wrote expanded predictions",
            expanded_prediction_count=len(expanded),
        )

        context["predictions_path"] = str(predictions_path)
        return context
