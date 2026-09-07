# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image post-processing for the render command: annotated labels, render-diff overlay,
and base64 inlining. Backend-agnostic — operates on the PNGs the render backend produced.
"""

from __future__ import annotations

import base64
from pathlib import Path


def annotate(beauty_path: str, labels) -> str | None:
    """Overlay `@ref` labels on the beauty render at pre-computed screen anchors.

    `labels` is a list of (ref, (px_u, px_v)) — screen-space anchors on each prim's
    visible-pixel centroid (the caller does occlusion culling via a segmentation pass, so
    only prims actually visible from the camera appear here). Writes
    `<stem>__annotated.png` and returns its path, or None when there is nothing to label.

    Layout: each label box is placed to the side of its anchor (right for anchors in the
    left half, left for the right half, so labels fan outward toward the margins), then
    nudged vertically until it clears every already-placed box — a leader line + dot ties
    it back to the true anchor. This spreads labels out instead of stacking them.
    """
    from PIL import Image, ImageDraw

    anchors = [(ref, int(u), int(v)) for ref, (u, v) in labels if ref]
    if not anchors:
        return None

    img = Image.open(beauty_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    line_h, placed = 14, []

    def _overlaps(box):
        return any(not (box[2] < p[0] or box[0] > p[2] or box[3] < p[1] or box[1] > p[3])
                   for p in placed)

    # Place densest-first (top-to-bottom, then left-to-right) for stable layout.
    for ref, ax, ay in sorted(anchors, key=lambda a: (a[2], a[1])):
        tw = 7 * len(ref) + 6
        right = ax < w // 2  # anchors on the left get labels to their right, and vice-versa
        lx = ax + 8 if right else ax - 8 - tw
        ly = ay - line_h // 2
        lx = max(1, min(lx, w - tw - 1))
        tries = 0
        while _overlaps((lx, ly, lx + tw, ly + line_h)) and tries < 80:
            ly += line_h + 2
            if ly + line_h >= h:            # ran off the bottom — reset high and shove sideways
                ly = 2
                lx += (tw + 10) if right else -(tw + 10)
                lx = max(1, min(lx, w - tw - 1))
            tries += 1
        cx = lx if right else lx + tw       # leader attaches to the box edge nearest the anchor
        draw.line([ax, ay, cx, ly + line_h // 2], fill=(255, 230, 90), width=1)
        draw.ellipse([ax - 3, ay - 3, ax + 3, ay + 3], fill=(255, 230, 90))
        draw.rectangle([lx, ly, lx + tw, ly + line_h], fill=(0, 0, 0))
        draw.text((lx + 2, ly + 2), ref, fill=(255, 230, 90))
        placed.append((lx, ly, lx + tw, ly + line_h))

    out = str(Path(beauty_path).with_name(Path(beauty_path).stem + "__annotated.png"))
    img.save(out)
    return out


def render_diff(before_path: str, after_path: str, *, threshold: int = 12) -> tuple[str, dict]:
    """Directional diff of `after` vs `before`, overlaid on a dimmed base.

    Changed pixels are colored by direction (via luminance): **green = after/added** (now
    brighter — appeared / moved-in) and **red = before/removed** (now darker — disappeared /
    moved-out). For the geometry (segmentation) diff this reads as the object's new footprint
    in green and its old footprint in red; for the beauty diff, brighter vs darker regions.

    Returns (diff_image_path, stats). `stats` carries changed/added/removed fractions and the
    mean absolute difference — a cheap, quantitative verify signal.
    """
    import numpy as np
    from PIL import Image

    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")
    if before.size != after.size:
        before = before.resize(after.size)
    a = np.asarray(after, dtype=np.float32)
    b = np.asarray(before, dtype=np.float32)
    delta = np.abs(a - b).sum(axis=2)
    changed = delta > threshold
    lum_a = a @ (0.299, 0.587, 0.114)
    lum_b = b @ (0.299, 0.587, 0.114)
    added = changed & (lum_a >= lum_b)   # after brighter → appeared / moved here
    removed = changed & (lum_a < lum_b)  # after darker → disappeared / moved away
    base = (a * 0.3).astype(np.uint8)
    base[added] = [60, 230, 90]
    base[removed] = [235, 60, 60]
    out = str(Path(after_path).with_name(Path(after_path).stem + "__diff.png"))
    Image.fromarray(base, "RGB").save(out)
    stats = {"changed_fraction": round(float(changed.mean()), 5),
             "added_fraction": round(float(added.mean()), 5),
             "removed_fraction": round(float(removed.mean()), 5),
             "mean_abs_diff": round(float(delta.mean() / 3.0), 3),
             "legend": "green=after/added, red=before/removed",
             "width": after.size[0], "height": after.size[1]}
    return out, stats


def is_blank_suspect(path: str, *, std_threshold: float = 2.0,
                     max_unique_colors: int = 8) -> bool:
    """True when an image looks featureless — a render that likely missed its subject.

    A camera pointed at nothing produces a plausible-looking flat/gradient frame the
    agent only catches by actually viewing it. Two cheap signals: near-zero grayscale
    standard deviation (flat or barely-graded frame) or a tiny unique-color count
    (solid background, unlit silhouette fills). Advisory — errors read as "not blank"
    so a heuristic hiccup can never fail a successful render.
    """
    from PIL import Image, ImageStat

    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((128, 128))  # heuristic-scale: cheap and noise-averaging
            if ImageStat.Stat(img.convert("L")).stddev[0] < std_threshold:
                return True
            colors = img.getcolors(maxcolors=max_unique_colors)
            return colors is not None  # non-None: at most max_unique_colors distinct
    except Exception:  # noqa: BLE001 — advisory heuristic only
        return False


def blank_suspects(paths: list[str], **kwargs) -> list[str]:
    """The subset of `paths` whose images look blank/featureless (see is_blank_suspect)."""
    return [p for p in paths if is_blank_suspect(p, **kwargs)]


def to_base64(path: str) -> str:
    """Read an image file and return its base64-encoded contents (for `--inline`)."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def assemble_gif(frame_paths: list[str], out_path: str, fps: float = 24.0) -> str:
    """Assemble an animated GIF from frame PNGs (dependency-free playback for notebooks)."""
    from PIL import Image

    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    if not frames:
        raise ValueError("no frames to assemble")
    duration_ms = max(1, int(round(1000.0 / max(1e-3, fps))))
    frames[0].save(out_path, save_all=True, append_images=frames[1:], loop=0,
                   duration=duration_ms, optimize=False)
    return out_path
