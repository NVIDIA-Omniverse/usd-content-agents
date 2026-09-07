#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provide deterministic affine registration for semantic overlays."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage, optimize


def foreground_mask(rgb: np.ndarray, *, threshold: float = 14.0) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32)[..., :3]
    corners = np.stack((image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]))
    background = np.median(corners, axis=0)
    distance = np.linalg.norm(image - background, axis=-1)
    mask = distance > float(threshold)
    return ndimage.binary_closing(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    )


def chroma_semantic_mask(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.int16)[..., :3]
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    return (
        (red >= 145)
        & (blue >= 115)
        & ((red - green) >= 55)
        & ((blue - green) >= 35)
        & (((red + blue) // 2) >= green + 55)
    )


def _bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 16:
        raise ValueError("Foreground mask is too small for affine registration")
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def bbox_affine(source_mask: np.ndarray, generated_mask: np.ndarray) -> np.ndarray:
    sx0, sy0, sx1, sy1 = _bbox(source_mask)
    gx0, gy0, gx1, gy1 = _bbox(generated_mask)
    scale_x = (gx1 - gx0) / max(sx1 - sx0, 1.0)
    scale_y = (gy1 - gy0) / max(sy1 - sy0, 1.0)
    return np.asarray(
        [
            [scale_x, 0.0, gx0 - scale_x * sx0],
            [0.0, scale_y, gy0 - scale_y * sy0],
        ],
        dtype=np.float32,
    )


def _registration_feature(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.float32)
    edges = mask ^ ndimage.binary_erosion(mask)
    silhouette = ndimage.gaussian_filter(binary, sigma=2.0)
    edge_field = ndimage.gaussian_filter(edges.astype(np.float32), sigma=1.5)
    return np.clip(0.72 * silhouette + 0.28 * edge_field, 0.0, 1.0)


def _resize_feature(values: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(values, dtype=np.float32))
    return np.asarray(
        image.resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = np.asarray(first, dtype=np.float64) - float(np.mean(first))
    second_centered = np.asarray(second, dtype=np.float64) - float(np.mean(second))
    denominator = float(
        np.linalg.norm(first_centered.ravel()) * np.linalg.norm(second_centered.ravel())
    )
    if denominator <= 1.0e-12:
        return -1.0
    return float(np.dot(first_centered.ravel(), second_centered.ravel()) / denominator)


def _fit_feature_affine(
    source: np.ndarray,
    generated: np.ndarray,
    initial: np.ndarray,
    *,
    iterations: int,
    epsilon: float,
) -> tuple[np.ndarray, float]:
    height, width = source.shape

    def objective(parameters: np.ndarray) -> float:
        matrix = np.asarray(parameters, dtype=np.float64).reshape(2, 3)
        candidate = _inverse_warp(
            generated,
            matrix,
            (width, height),
            interpolation=1,
        )
        score = _correlation(source, candidate)
        return 1.0 - score if math.isfinite(score) else 2.0

    result = optimize.minimize(
        objective,
        np.asarray(initial, dtype=np.float64).reshape(-1),
        method="Powell",
        bounds=(
            (-2.0, 2.0),
            (-1.5, 1.5),
            (-float(width), float(width)),
            (-1.5, 1.5),
            (-2.0, 2.0),
            (-float(height), float(height)),
        ),
        options={
            "maxiter": int(iterations),
            "xtol": float(epsilon),
            "ftol": float(epsilon),
        },
    )
    if not np.all(np.isfinite(result.x)) or not math.isfinite(float(result.fun)):
        raise ValueError("affine correlation fitting produced non-finite output")
    return np.asarray(result.x, dtype=np.float32).reshape(2, 3), float(1.0 - result.fun)


def fit_affine_ecc(
    source_mask: np.ndarray,
    generated_mask: np.ndarray,
    *,
    scales: tuple[float, ...] = (0.25, 0.5, 1.0),
    iterations: int = 300,
    epsilon: float = 1.0e-7,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    source_feature = _registration_feature(source_mask)
    generated_feature = _registration_feature(generated_mask)
    warp = bbox_affine(source_mask, generated_mask)
    trace: list[dict[str, float]] = []
    for scale in scales:
        width = max(32, int(round(source_feature.shape[1] * scale)))
        height = max(32, int(round(source_feature.shape[0] * scale)))
        source_scaled = _resize_feature(source_feature, width, height)
        generated_scaled = _resize_feature(generated_feature, width, height)
        scaled_warp = warp.copy()
        scaled_warp[:, 2] *= float(scale)
        try:
            scaled_warp, correlation = _fit_feature_affine(
                source_scaled,
                generated_scaled,
                scaled_warp,
                iterations=iterations,
                epsilon=epsilon,
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"affine correlation fitting failed at scale {scale:g}: {exc}"
            ) from exc
        warp = scaled_warp
        warp[:, 2] /= float(scale)
        trace.append({"scale": float(scale), "correlation": float(correlation)})
    return warp.astype(np.float32), trace


def _inverse_warp(
    values: np.ndarray,
    source_to_generated: np.ndarray,
    output_size: tuple[int, int],
    *,
    interpolation: int,
) -> np.ndarray:
    matrix = np.asarray(source_to_generated, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise ValueError("source_to_generated must be a 2x3 affine matrix")
    width, height = output_size
    input_from_output = np.asarray(
        [
            [matrix[1, 1], matrix[1, 0]],
            [matrix[0, 1], matrix[0, 0]],
        ],
        dtype=np.float64,
    )
    offset = np.asarray([matrix[1, 2], matrix[0, 2]], dtype=np.float64)

    def warp_channel(channel: np.ndarray) -> np.ndarray:
        return ndimage.affine_transform(
            channel,
            input_from_output,
            offset=offset,
            output_shape=(height, width),
            order=int(interpolation),
            mode="constant",
            cval=0,
            prefilter=False,
        )

    array = np.asarray(values)
    if array.ndim == 2:
        return warp_channel(array)
    if array.ndim == 3:
        return np.stack(
            [warp_channel(array[..., channel]) for channel in range(array.shape[2])],
            axis=-1,
        )
    raise ValueError("affine warp input must be a 2D or 3D array")


def _silhouette_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.count_nonzero(first & second)
    union = np.count_nonzero(first | second)
    return float(intersection / max(union, 1))


def _edge_f_score(
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    tolerance_pixels: int = 2,
) -> float:
    reference_edges = reference_mask ^ ndimage.binary_erosion(reference_mask)
    candidate_edges = candidate_mask ^ ndimage.binary_erosion(candidate_mask)
    if not np.any(reference_edges) or not np.any(candidate_edges):
        return 0.0
    size = 2 * int(tolerance_pixels) + 1
    kernel = np.ones((size, size), dtype=bool)
    reference_neighborhood = ndimage.binary_dilation(reference_edges, structure=kernel)
    candidate_neighborhood = ndimage.binary_dilation(candidate_edges, structure=kernel)
    precision = np.count_nonzero(candidate_edges & reference_neighborhood) / max(
        np.count_nonzero(candidate_edges), 1
    )
    recall = np.count_nonzero(reference_edges & candidate_neighborhood) / max(
        np.count_nonzero(reference_edges), 1
    )
    return float(2.0 * precision * recall / max(precision + recall, 1.0e-12))


def _affine_plausibility(matrix: np.ndarray) -> dict[str, Any]:
    linear = np.asarray(matrix, dtype=np.float64)[:, :2]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    determinant = float(np.linalg.det(linear))
    anisotropy = float(singular_values.max() / singular_values.min())
    accepted = bool(
        determinant > 0.0
        and 0.55 <= float(singular_values.min())
        and float(singular_values.max()) <= 1.8
        and anisotropy <= 1.45
    )
    return {
        "accepted": accepted,
        "determinant": determinant,
        "singular_values": [float(value) for value in singular_values],
        "maximum_anisotropy": anisotropy,
    }


def register_overlay(
    source_rgb: np.ndarray,
    generated_rgb: np.ndarray,
) -> dict[str, Any]:
    source = np.asarray(source_rgb, dtype=np.uint8)[..., :3]
    generated_image = Image.fromarray(
        np.asarray(generated_rgb, dtype=np.uint8)[..., :3], mode="RGB"
    )
    generated_resized = np.asarray(
        generated_image.resize(
            (source.shape[1], source.shape[0]), Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )
    source_silhouette = foreground_mask(source)
    generated_silhouette = foreground_mask(generated_resized)
    generated_semantic = chroma_semantic_mask(generated_resized)
    matrix, trace = fit_affine_ecc(source_silhouette, generated_silhouette)
    output_size = (source.shape[1], source.shape[0])
    aligned_rgb = _inverse_warp(
        generated_resized,
        matrix,
        output_size,
        interpolation=1,
    )
    aligned_silhouette = (
        _inverse_warp(
            generated_silhouette.astype(np.uint8),
            matrix,
            output_size,
            interpolation=0,
        )
        > 0
    )
    aligned_semantic = (
        _inverse_warp(
            generated_semantic.astype(np.uint8),
            matrix,
            output_size,
            interpolation=0,
        )
        > 0
    )
    return {
        "source_to_generated_affine": matrix,
        "ecc_trace": trace,
        "aligned_rgb": aligned_rgb,
        "aligned_semantic_mask": aligned_semantic,
        "silhouette_iou": _silhouette_iou(source_silhouette, aligned_silhouette),
        "edge_f_score_2px": _edge_f_score(
            source_silhouette, aligned_silhouette, tolerance_pixels=2
        ),
        "plausibility": _affine_plausibility(matrix),
    }
