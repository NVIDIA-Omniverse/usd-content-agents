# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render time-sampled USD files into ordered image evidence."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image

from world_understanding.functions.graphics.rendering import RenderingBackend
from world_understanding.functions.graphics.rendering_backend_factory import (
    create_rendering_backend,
    validate_rendering_backend_name,
)

logger = logging.getLogger(__name__)


def render_time_sampled_usd(
    time_sampled_usd: Path | str,
    output_dir: Path | str,
    *,
    renderer: str = "ovrtx",
    frames: str | None = None,
    fps: int | None = None,
    cameras: list[str] | None = None,
    image_width: int = 512,
    image_height: int = 512,
    make_mp4: bool = True,
    max_duration_seconds: float = 2.0,
    num_sensor_updates: int | None = None,
    render_mode: str | None = None,
) -> list[Path]:
    """Render an existing time-sampled USD into chronological PNG files.

    The input USD is treated as read-only. This helper opens the stage,
    resolves a frame range, delegates rendering to an existing graphics
    backend, and writes the returned images to ``output_dir``.

    The selected backend is responsible for evaluating authored USD time
    samples at each frame. This driver does not bake animation or add
    time-sample support to backends that only partially implement it.

    Supported animation channels per backend (v1 contract):

    * **OvRTX** — time-sampled `xformOp:*` transforms,
      `primvars:displayColor` (replayed via per-frame displayColor overlays
      until native OVRTX 0.3 sampling is validated), visibility (replayed via
      per-frame static overlay layers around an historical OvRTX 0.2.0
      visibility crash until 0.3 GPU validation proves native visibility
      safe), and time-sampled camera transforms. The remote backend inherits
      this when backed by the same OvRTX service path.
    * **Warp** — preview-quality only; mesh geometry is sampled at the
      first frame and not re-evaluated per timecode.
    * **Mock** — deterministic CPU-only per-frame images for tests and
      simulation. These are not production visual evidence.

    Anything *outside* the channels listed above (e.g. animated
    `points`/topology, animated UsdShade materials, animated lights,
    UsdGeomPointInstancer, UsdSkel) is not part of the v1 contract and
    may render as the first-frame state regardless of authored
    time samples. Rigid-body simulation evidence — translating bodies
    plus camera motion — is the intended primary use case.
    """

    renderer = validate_rendering_backend_name(renderer)

    usd_path = Path(time_sampled_usd)
    if not usd_path.exists():
        raise FileNotFoundError(usd_path)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    if max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be positive")

    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    fps_value = _resolve_fps(stage, fps)
    frame_list = _resolve_frame_list(stage, frames)
    _validate_frame_caps(frame_list, fps_value, max_duration_seconds)
    frames_arg = _format_frames(frame_list)

    # The remote frame parser only accepts ``"N"`` and ``"start:end"``
    # (see render_remote.render_single_camera_from_url). A sparse spec
    # like ``"0,5,10"`` would raise ``int(...)`` mid-request and produce
    # an opaque error — surface a precise message up front so callers
    # can switch renderer or expand the selection.
    if renderer == "remote" and "," in frames_arg:
        raise ValueError(
            f"renderer='remote' does not support sparse frame selections "
            f"(got {frames_arg!r}). Use a contiguous range "
            "(e.g. '0:10'), a single frame, or switch to "
            "renderer='ovrtx'/'warp'/'mock'."
        )

    # Override only the in-memory stage metadata. The source USD file is not
    # saved or exported, but renderers that consult stage fps see this value.
    if fps is not None:
        stage.SetTimeCodesPerSecond(float(fps_value))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    backend = _make_backend(renderer)
    render_kwargs: dict[str, Any] = {
        "cameras": cameras,
        "image_width": image_width,
        "image_height": image_height,
        "frames": frames_arg,
    }
    if num_sensor_updates is not None:
        render_kwargs["num_sensor_updates"] = num_sensor_updates
    if render_mode is not None:
        render_kwargs["render_mode"] = render_mode
    result = backend.render(stage, **render_kwargs)
    png_paths, image_sequences = _save_rendered_images(result, frame_list, output_path)
    if not png_paths:
        raise RuntimeError("Renderer produced no images")

    if make_mp4:
        _write_mp4_sequences(image_sequences, output_path, fps_value)

    return png_paths


def _resolve_fps(stage: Any, fps: int | None) -> float:
    if fps is not None:
        fps_value = float(fps)
    else:
        fps_value = float(stage.GetTimeCodesPerSecond() or 24.0)

    if fps_value <= 0:
        raise ValueError("fps must be positive")
    if fps_value > 60:
        raise ValueError("fps must be <= 60")
    return fps_value


def _resolve_frame_list(stage: Any, frames: str | None) -> list[int]:
    if frames is not None:
        frame_list = _parse_frames(frames)
    else:
        frame_list = _infer_frame_list(stage)

    if not frame_list:
        raise ValueError("No frames selected for rendering")
    return frame_list


def _parse_frames(frames: str) -> list[int]:
    spec = frames.strip()
    if not spec:
        raise ValueError("frames must not be empty")

    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid frame range: {frames!r}")
        start = int(parts[0])
        end = int(parts[1])
        if end < start:
            raise ValueError("frame range end must be >= start")
        return list(range(start, end + 1))

    if "," in spec:
        values = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
        if not values:
            raise ValueError("frames must select at least one frame")
        return values

    return [int(spec)]


def _infer_frame_list(stage: Any) -> list[int]:
    # An authored ``startTimeCode == endTimeCode`` is a valid single-frame
    # range per the USD spec, so check ``HasAuthoredTimeCodeRange()``
    # rather than relying on ``end > start``. Without this, callers who
    # author a single-frame range explicitly would silently fall through
    # to scanning every prim's authored time samples (slow on big stages
    # and incorrect when no animated attributes exist).
    if stage.HasAuthoredTimeCodeRange():
        start = _as_integral_frame(stage.GetStartTimeCode(), "startTimeCode")
        end = _as_integral_frame(stage.GetEndTimeCode(), "endTimeCode")
        if end < start:
            raise ValueError(
                f"Authored time-code range is reversed (start={start}, end={end})"
            )
        return list(range(start, end + 1))

    authored = _authored_time_sample_frames(stage)
    if authored:
        return list(range(min(authored), max(authored) + 1))

    return [_as_integral_frame(stage.GetStartTimeCode(), "startTimeCode")]


def _authored_time_sample_frames(stage: Any) -> list[int]:
    frames: set[int] = set()
    for prim in stage.TraverseAll():
        if prim.IsInstanceProxy():
            continue
        for attr in prim.GetAttributes():
            for time_code in attr.GetTimeSamples():
                frames.add(_as_integral_frame(time_code, attr.GetPath().pathString))
    return sorted(frames)


def _as_integral_frame(value: Any, label: str) -> int:
    number = float(value)
    rounded = round(number)
    if abs(number - rounded) > 1e-6:
        raise ValueError(
            f"{label} uses non-integer time code {number}; "
            "current render backends accept integer frame selections"
        )
    return int(rounded)


def _validate_frame_caps(
    frame_list: list[int], fps_value: float, max_duration_seconds: float
) -> None:
    """Cap on rendered frame count with closed-interval tolerance.

    The recorder writes ``int(fps × duration_s) + 1`` samples for
    closed-interval trajectories, which trips a strict cap by exactly
    1. We therefore split the check into two bands:

    * ``len(frame_list) > max_frames``: warn-only. This is the
      fence-post boundary — legitimate closed-interval recordings
      land here and should render, but the discrepancy is logged so
      the user can act if unintended.
    * ``len(frame_list) > max_frames + 1``: hard cap. Raise
      ``ValueError`` rather than letting unbounded evidence renders
      proceed. This preserves the original guard against accidental
      huge renders that ``max_duration_seconds`` advertises.

    Where ``max_frames = max(1, int(max_duration_seconds * fps_value))``.
    """
    max_frames = max(1, int(max_duration_seconds * fps_value))
    if len(frame_list) > max_frames + 1:
        raise ValueError(
            f"frame count {len(frame_list)} exceeds cap of "
            f"{max_frames + 1} (= max_duration_seconds {max_duration_seconds} "
            f"× sample_fps {fps_value} + 1-frame closed-interval tolerance). "
            "Reduce the frame selection or raise max_duration_seconds."
        )
    if len(frame_list) > max_frames:
        logger.warning(
            "frame-count discrepancy: recording has %d time samples, "
            "validator's expected cap is %d (%g s × %g fps). "
            "Rendering all %d frames; if unexpected, adjust "
            "max_duration_seconds or sample_fps.",
            len(frame_list),
            max_frames,
            max_duration_seconds,
            fps_value,
            len(frame_list),
        )


def _format_frames(frame_list: list[int]) -> str:
    ordered = sorted(frame_list)
    if len(ordered) == 1:
        return str(ordered[0])
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}:{ordered[-1]}"
    return ",".join(str(frame) for frame in ordered)


def _make_backend(renderer: str) -> RenderingBackend:
    """Delegate time-sampled rendering to the canonical backend factory."""
    return create_rendering_backend(renderer)


def _save_rendered_images(
    result: dict[str, Any], frame_list: list[int], output_dir: Path
) -> tuple[list[Path], list[tuple[str, list[Path]]]]:
    render_results = result.get("results")
    if not isinstance(render_results, list):
        raise RuntimeError("Renderer result missing results list")

    single_camera = len(render_results) == 1
    written: list[Path] = []
    image_sequences: list[tuple[str, list[Path]]] = []
    # Track final slugs already used in this pass — not just base slugs —
    # so a third camera whose own normalized name happens to equal a
    # previously-disambiguated suffix (e.g. ``Cam A`` then ``Cam_A``
    # then a real ``Cam_A_1``, all mapping to the same family) still
    # ends up with a unique slug. Walking the suffix counter until
    # the candidate is free guarantees no two cameras share filenames
    # regardless of how the input names happen to normalize.
    used_final_slugs: set[str] = set()
    for camera_result in render_results:
        images = camera_result.get("images") or []
        if len(images) != len(frame_list):
            camera_name = camera_result.get("camera", "<unknown>")
            raise RuntimeError(
                f"Renderer returned {len(images)} image(s) for camera {camera_name!r}, "
                f"expected {len(frame_list)}"
            )

        base_slug = _slug(camera_result.get("camera"))
        candidate = base_slug
        suffix = 0
        while candidate in used_final_slugs:
            suffix += 1
            candidate = f"{base_slug}_{suffix}"
        used_final_slugs.add(candidate)
        camera_slug = candidate
        camera_prefix = "" if single_camera else f"{camera_slug}__"
        camera_paths: list[Path] = []
        for frame, image in zip(frame_list, images, strict=True):
            if not isinstance(image, Image.Image):
                raise RuntimeError(
                    "Renderer returned a non-PIL image; expected normalized backend output"
                )
            frame_path = output_dir / f"{camera_prefix}frame_{frame:04d}.png"
            image.save(frame_path)
            written.append(frame_path)
            camera_paths.append(frame_path)

        image_sequences.append((camera_slug, camera_paths))

    return written, image_sequences


def _slug(value: Any) -> str:
    text = str(value or "camera").strip().strip("/") or "camera"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _write_mp4_sequences(
    image_sequences: list[tuple[str, list[Path]]], output_dir: Path, fps_value: float
) -> None:
    if len(image_sequences) == 1:
        _maybe_write_mp4(image_sequences[0][1], output_dir / "render.mp4", fps_value)
        return

    for camera_slug, png_paths in image_sequences:
        _maybe_write_mp4(
            png_paths, output_dir / f"{camera_slug}__render.mp4", fps_value
        )


def _maybe_write_mp4(
    png_paths: list[Path], output_path: Path, fps_value: float
) -> None:
    try:
        import imageio.v3 as iio  # type: ignore[import-not-found,unused-ignore]
        import numpy as np
    except ImportError:
        logger.info("imageio is unavailable; skipping mp4 creation")
        return

    try:
        frames = [np.asarray(Image.open(path).convert("RGB")) for path in png_paths]
        iio.imwrite(output_path, frames, fps=fps_value)
    except Exception as exc:  # pragma: no cover - depends on optional codecs
        logger.warning("Failed to write mp4 render %s: %s", output_path, exc)
