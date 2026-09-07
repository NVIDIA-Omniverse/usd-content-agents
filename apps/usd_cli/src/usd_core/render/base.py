# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The render-backend contract shared by local and remote OVRTX."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


def prepare_render_input(stage, out_dir) -> tuple[Path, bool]:
    """Write a USD file an external renderer can open, preserving payload/texture
    resolution AND every composed edit — root-layer and session-layer alike.
    Returns (path, is_temp).

    A SimReady asset references its payload/textures by paths relative to the original
    layer's directory; `Layer.Export` to a different directory breaks them (the renderer
    then loads no geometry → a blank image). So we export the working stage — which
    carries in-memory edits like the managed camera — *next to the original* where those
    relatives still resolve. For in-memory stages (no on-disk original) we `Flatten()`
    into out_dir, which inlines payloads.

    The session layer is part of the evidence contract: `appearance clear` deliberately
    moves the edit target to the SESSION layer, so it and every accepted material edit
    after it live there. Exporting only the root layer silently reopened the pre-clear
    asset — final renders showed the OLD materials while `save --flatten` wrote the new
    ones (wu review P1: wrong final evidence). When the session layer carries opinions,
    the export is a single field-level merge of session-over-root
    (`Sdf.Layer.TransferContent` + `UsdUtils.StitchLayers`): session opinions win,
    root supplies everything else (geometry, layer metadata like defaultPrim/upAxis,
    sublayer lists), and references/payloads stay composition arcs — nothing is
    inlined, and relative asset paths keep resolving because the merged file sits in
    the original's directory, exactly like the plain root export.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = stage.GetRootLayer()
    session = stage.GetSessionLayer()
    session_carries_edits = session is not None and not session.empty
    root_carries_edits = bool(getattr(root, "dirty", False))
    real = getattr(root, "realPath", "") or ""
    tok = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"  # unique: avoid collisions under concurrency
    if real.endswith(".usdz"):
        if not session_carries_edits and not root_carries_edits:
            # An untouched USDZ root resolves its references only inside the package.
            # Hand the exact package to the renderer rather than exporting a loose
            # layer with dangling package-internal paths.
            return Path(real), False
        # Package layers cannot be saved, but USD still accepts in-memory root-layer
        # opinions (for example usd-cli's managed render camera) and marks the layer
        # dirty. Session or dirty-root opinions therefore require the flattened
        # fallback, which composes both and anchors asset paths into the package
        # (@…scene.usdz[textures/x.png]@). Reusing the original package would silently
        # drop the camera, leaving OVRTX with no valid RenderProduct to discover.
    elif real:
        orig_dir = Path(real).parent
        if os.access(orig_dir, os.W_OK):
            p = orig_dir / f".usd-cli_render_{tok}.usdc"
            if session_carries_edits:
                from pxr import Sdf, UsdUtils
                merged = Sdf.Layer.CreateNew(str(p))
                merged.TransferContent(session)
                # Field-level merge: opinions already in `merged` (the session copy)
                # win; the root layer fills in everything it doesn't override.
                UsdUtils.StitchLayers(merged, root)
                merged.Save()
            else:
                root.Export(str(p))
            return p, True
    # fallback: inline payloads (textures may not resolve). Flatten() composes the
    # full stage, session layer included. Binary .usdc, not .usda — ASCII flattening
    # is ~3× larger, which alone can push remote uploads past the service body limit
    # (HTTP 413).
    p = out_dir / f"_render_stage_{tok}.usdc"
    stage.Flatten().Export(str(p))
    return p, False


@dataclass
class RenderResult:
    path: str  # PNG written to disk
    camera: str  # camera prim path rendered
    width: int
    height: int
    backend: str
    render_time: float = 0.0
    # For remote OVRTX pools, bind the exact verified endpoint that produced
    # this image. Local renderers leave this unset.
    renderer_identity: dict[str, object] | None = None
    # advisory blank-frame heuristic: True when the image looks featureless (the camera
    # likely missed its subject). Set by the remote backend; serialized by the session.
    blank_suspect: bool = False
    # Factual OVRTX settings used for this image. These are reported by the
    # executing backend rather than inferred from the caller's quality label.
    ovrtx_render_mode: str | None = None
    ovrtx_num_sensor_updates: int | None = None
    active_aov: str | None = None
    camera_world_transform: list[list[float]] | None = None
    camera_pos: list[float] | None = None
    camera_dir: list[float] | None = None


@runtime_checkable
class RenderBackend(Protocol):
    """Single-modality (RGB/beauty) render of one or more cameras already on the stage.

    The camera transform is authored onto the camera prim (see usd_core.camera), so the
    backend only needs the stage + camera prim paths — matching OVRTX's contract.
    """

    name: str

    def render(
        self,
        stage,  # pxr.Usd.Stage
        cameras: list[str],
        width: int,
        height: int,
        out_dir,  # pathlib.Path
        mode: str = "quality",
        names: list[str] | None = None,  # output file stems (no ext), parallel to cameras
        frame: float | None = None,  # USD time code to render at (None = default/earliest)
    ) -> list[RenderResult]: ...
