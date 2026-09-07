# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = (
    REPO_ROOT / "agentic/.agents/skills/content-workflow-mesh-segmentation/scripts"
)
REGISTRATION_SCRIPT = SCRIPT_DIR / "semantic_overlay_registration.py"
COMMAND_SCRIPT = SCRIPT_DIR / "register_semantic_overlays.py"


def _load_registration_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "mesh_segmentation_semantic_overlay_registration",
        REGISTRATION_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_semantic_registration_has_no_opencv_dependency() -> None:
    for path in (REGISTRATION_SCRIPT, COMMAND_SCRIPT):
        source = path.read_text(encoding="utf-8")
        assert "import cv2" not in source
        assert "opencv" not in source.casefold()


def test_public_semantic_registration_aligns_scaled_image_evidence() -> None:
    registration = _load_registration_module()
    source = np.zeros((64, 64, 3), dtype=np.uint8)
    source[16:48, 18:46] = (185, 185, 185)

    generated = np.zeros_like(source)
    generated[13:53, 13:51] = (185, 185, 185)
    generated[25:42, 24:39] = (220, 40, 180)

    result = registration.register_overlay(source, generated)

    assert result["silhouette_iou"] >= 0.99
    assert result["edge_f_score_2px"] >= 0.99
    assert result["plausibility"]["accepted"] is True
    assert np.count_nonzero(result["aligned_semantic_mask"]) > 0
    assert len(result["ecc_trace"]) == 3


def test_public_semantic_registration_aligns_rotated_noisy_evidence() -> None:
    registration = _load_registration_module()
    source = np.zeros((96, 96, 3), dtype=np.uint8)
    source[20:72, 24:40] = (185, 185, 185)
    source[55:72, 24:70] = (185, 185, 185)
    source[30:48, 52:68] = (185, 185, 185)

    rotated = ndimage.rotate(
        source.astype(np.float32),
        11.0,
        axes=(1, 0),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    shifted = ndimage.shift(
        rotated,
        shift=(-3.0, 4.0, 0.0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    generated = np.clip(shifted, 0, 255).astype(np.uint8)
    ys, xs = np.indices(generated.shape[:2])
    semantic_region = (generated[..., 0] > 100) & (ys > 43) & (xs > 38)
    generated[semantic_region] = (220, 40, 180)
    noise = np.random.default_rng(73).normal(0.0, 2.0, generated.shape)
    generated = np.clip(generated.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    result = registration.register_overlay(source, generated)

    assert result["silhouette_iou"] >= 0.94
    assert result["edge_f_score_2px"] >= 0.98
    assert result["plausibility"]["accepted"] is True
    assert np.count_nonzero(result["aligned_semantic_mask"]) > 0
    assert all(step["correlation"] >= 0.98 for step in result["ecc_trace"])
