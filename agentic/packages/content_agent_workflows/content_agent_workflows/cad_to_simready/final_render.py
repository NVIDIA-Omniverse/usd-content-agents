# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Digest-bound OVRTX presentation evidence for CAD-to-SimReady outputs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, ImageSequence

from content_agent_workflows.common.artifacts import atomic_write_json, file_sha256

CAD_TO_SIMREADY_FINAL_RENDER_SCHEMA_VERSION = (
    "content-agent-workflows.cad-to-simready-final-render.v3"
)

_HERO_DIRECTION = "+x-y+z"
_MULTIVIEW_DIRECTIONS = (
    ("px", "+x"),
    ("nx", "-x"),
    ("py", "+y"),
    ("ny", "-y"),
    ("pz", "+z"),
    ("nz", "-z"),
)
_OVRTX_TRANSPORTS = frozenset({"ovrtx", "remote"})


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read final validation report: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"final validation report must be a JSON object: {path}")
    return value


def _file_record(path: Path, *, media_type: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "media_type": media_type,
    }


def _verified_final_asset(final_usd: Path, validation_report: Path) -> dict[str, Any]:
    report = _read_json_object(validation_report)
    try:
        reported_asset = Path(str(report.get("asset_path"))).expanduser().resolve()
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("final validation report has no valid asset_path") from exc
    digest = file_sha256(final_usd)
    status = str(report.get("status", "")).upper()
    passed = report.get("passed")
    if not (
        report.get("schema_version")
        == "content-agent-workflows.simready-profile-validation.v3"
        and status in {"PASS", "FAIL"}
        and isinstance(passed, bool)
        and passed == (status == "PASS")
        and reported_asset == final_usd
        and report.get("asset_sha256") == digest
        and report.get("foundation_checkout_verified") is True
        and report.get("validator_runtime_verified") is True
        and report.get("errors") == []
    ):
        raise ValueError(
            "final render input must be the exact USD from a verified SimReady "
            "validation report"
        )
    return report


def _image_metadata(path: Path) -> dict[str, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return {"width": image.width, "height": image.height}
    except (OSError, ValueError) as exc:
        raise ValueError(f"final render is not a readable image: {path}") from exc


def _gif_metadata(path: Path) -> dict[str, int]:
    try:
        with Image.open(path) as image:
            frame_count = sum(1 for _ in ImageSequence.Iterator(image))
            return {
                "width": image.width,
                "height": image.height,
                "frame_count": frame_count,
            }
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"turntable render is not a readable animation: {path}"
        ) from exc


def _compose_multiview(images: list[Path], output: Path) -> dict[str, int]:
    frames: list[Image.Image] = []
    for path in images:
        with Image.open(path) as image:
            frames.append(image.convert("RGB").copy())
    if len(frames) != 6 or len({frame.size for frame in frames}) != 1:
        raise ValueError("six-view evidence requires six equal-sized OVRTX images")
    width, height = frames[0].size
    sheet = Image.new("RGB", (width * 3, height * 2))
    for index, frame in enumerate(frames):
        sheet.paste(frame, ((index % 3) * width, (index // 3) * height))
    sheet.save(output, format="PNG")
    return {"width": sheet.width, "height": sheet.height, "view_count": 6}


def _encode_turntable_gif(
    frames: list[Path], output: Path, *, fps: float
) -> dict[str, int | float]:
    images: list[Image.Image] = []
    for path in frames:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    if not images or len({image.size for image in images}) != 1:
        raise ValueError("turntable frames are missing or have inconsistent dimensions")
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000.0 / fps)),
        loop=0,
        optimize=False,
        disposal=2,
    )
    metadata = _gif_metadata(output)
    if metadata.get("frame_count") != len(images):
        raise RuntimeError("encoded turntable GIF failed frame-count verification")
    return {**metadata, "fps": fps}


def _sealed_render_record(record: dict[str, Any], source: Path) -> dict[str, Any]:
    if record.get("renderer") not in _OVRTX_TRANSPORTS:
        raise RuntimeError("usd-cli final render did not use a verified OVRTX backend")
    sealed = dict(record)
    sealed["rendered_usd"] = _file_record(source, media_type="model/vnd.usd")
    for field, media_type in (
        ("image_path", "image/png"),
        ("response_path", "application/json"),
        ("camera_json_path", "application/json"),
    ):
        value = sealed.get(field)
        if not isinstance(value, str):
            raise RuntimeError(f"usd-cli final render omitted {field}")
        artifact = Path(value).expanduser().resolve()
        if not artifact.is_file():
            raise RuntimeError(f"usd-cli final render artifact is missing: {artifact}")
        sealed[f"{field}_binding"] = _file_record(artifact, media_type=media_type)
    return sealed


def render_final_simready_evidence(
    final_usd: Path | str,
    validation_report: Path | str,
    output_dir: Path | str,
    report_path: Path | str,
    *,
    width: int = 768,
    height: int = 576,
    turntable_frame_count: int = 24,
    turntable_fps: float = 12.5,
    scene_tool_timeout_seconds: float = 300.0,
    session_factory: Callable[..., Any] | None = None,
    dependency_paths_fn: Callable[[Path], tuple[Path, ...]] | None = None,
) -> dict[str, Any]:
    """Render the exact verified USD through one receipt-bound usd-cli session."""

    source = Path(final_usd).expanduser().resolve()
    validation = Path(validation_report).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    report = Path(report_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"final SimReady USD is missing: {source}")
    if not validation.is_file():
        raise FileNotFoundError(
            f"final SimReady validation report is missing: {validation}"
        )
    if width < 64 or height < 64 or width % 2 or height % 2:
        raise ValueError("final render width and height must be even integers >= 64")
    if turntable_frame_count < 2:
        raise ValueError("turntable frame count must be at least 2")
    if turntable_fps <= 0:
        raise ValueError("turntable fps must be positive")
    if scene_tool_timeout_seconds <= 0:
        raise ValueError("scene tool timeout must be positive")
    validation_result = _verified_final_asset(source, validation)

    destination.mkdir(parents=True, exist_ok=True)
    source_before = _file_record(source, media_type="model/vnd.usd")
    validation_before = _file_record(validation, media_type="application/json")
    if dependency_paths_fn is None:
        from content_agent_workflows.validation.workflow import (
            _usd_dependency_paths,
        )

        dependency_paths_fn = _usd_dependency_paths
    dependency_paths = dependency_paths_fn(source)
    if not dependency_paths or source not in {
        path.resolve() for path in dependency_paths
    }:
        raise ValueError("USD dependency closure does not contain the final USD")
    dependencies_before = [
        _file_record(path.resolve(), media_type="application/octet-stream")
        for path in dependency_paths
    ]
    identity_payload = {
        "source": source_before,
        "validation": validation_before,
        "usd_dependencies": dependencies_before,
        "width": width,
        "height": height,
        "turntable_frame_count": turntable_frame_count,
        "turntable_fps": turntable_fps,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if session_factory is None:
        from content_agent_workflows.common.usd_cli_session import (
            WorkflowUsdCliSession,
        )

        session_factory = WorkflowUsdCliSession.create
    session = session_factory(
        owner_root=destination.parent,
        project_dir=destination,
        identity=identity,
        workflow="cad-to-simready-final-render",
        input_roots=(source, validation, *dependency_paths),
    )

    render_records: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    primary_error = False
    try:
        probe = session.require_ovrtx(destination / "probe")
        probe_path = destination / "ovrtx_probe_response.json"
        atomic_write_json(probe_path, probe, within=destination)
        session.open(source, read_only=False)
        up_axis_y = session.stage_up_axis_is_y(source)

        def render(
            name: str, direction: str, output: Path = destination
        ) -> dict[str, Any]:
            record = _sealed_render_record(
                session.render_view(
                    output_dir=output,
                    name=name,
                    direction=direction,
                    width=width,
                    height=height,
                    timeout_seconds=scene_tool_timeout_seconds,
                    up_axis_y=up_axis_y,
                ),
                source,
            )
            all_records.append(record)
            return record

        render_records["hero"] = render("final_hero", _HERO_DIRECTION)
        multiview_records = [
            render(f"final_six_view_{name}", direction)
            for name, direction in _MULTIVIEW_DIRECTIONS
        ]
        multiview_path = destination / "final_six_view.png"
        multiview_metadata = _compose_multiview(
            [Path(record["image_path"]) for record in multiview_records],
            multiview_path,
        )
        render_records["multiview"] = {
            "composition": _file_record(multiview_path, media_type="image/png"),
            "views": multiview_records,
        }

        frame_dir = destination / "turntable_frames"
        turntable_records: list[dict[str, Any]] = []
        for index in range(turntable_frame_count):
            angle = 2.0 * math.pi * index / turntable_frame_count
            direction = f"{math.cos(angle):+.8f}x{math.sin(angle):+.8f}y+0.35000000z"
            turntable_records.append(
                render(f"final_turntable_{index:03d}", direction, frame_dir)
            )
        frame_paths = [Path(record["image_path"]) for record in turntable_records]
        gif_path = destination / "final_turntable.gif"
        gif_metadata = _encode_turntable_gif(frame_paths, gif_path, fps=turntable_fps)
        render_records["turntable"] = {
            "frames": turntable_records,
            "gif": {
                **_file_record(gif_path, media_type="image/gif"),
                **gif_metadata,
            },
        }
    except Exception:
        primary_error = True
        raise
    finally:
        try:
            session.close()
        except Exception as exc:
            if not primary_error:
                raise RuntimeError(
                    f"could not close final-render usd-cli session: {exc}"
                ) from exc

    if (
        _file_record(source, media_type="model/vnd.usd") != source_before
        or _file_record(validation, media_type="application/json") != validation_before
        or [
            _file_record(path.resolve(), media_type="application/octet-stream")
            for path in dependency_paths_fn(source)
        ]
        != dependencies_before
    ):
        raise RuntimeError(
            "final USD, dependency closure, or validation report changed while rendering"
        )

    receipt = _file_record(session.receipt_file, media_type="application/x-ndjson")
    checkpoint = _file_record(
        session.receipt_checkpoint_file, media_type="application/json"
    )
    probe_record = _file_record(probe_path, media_type="application/json")
    records_path = destination / "final_render_records.json"
    atomic_write_json(
        records_path,
        {
            "schema_version": "content-agent-workflows.cad-to-simready-usd-cli-renders.v1",
            "scene_tool": "usd-cli",
            "render_engine": "ovrtx",
            "session_id": session.session_id,
            "usd_cli_source_revision": session.route.source_revision,
            "source_usd": source_before,
            "usd_dependencies": dependencies_before,
            "render_transports": sorted(
                {str(record["renderer"]) for record in all_records}
            ),
            "probe_response": probe_record,
            "usd_cli_command_receipt": receipt,
            "usd_cli_receipt_checkpoint": checkpoint,
            "renders": render_records,
        },
        within=destination,
    )
    records_binding = _file_record(records_path, media_type="application/json")

    hero_path = destination / "final_hero.png"
    multiview_path = destination / "final_six_view.png"
    gif_path = destination / "final_turntable.gif"
    payload: dict[str, Any] = {
        "schema_version": CAD_TO_SIMREADY_FINAL_RENDER_SCHEMA_VERSION,
        "render_status": "PASS",
        "scene_tool": "usd-cli",
        "render_engine": "ovrtx",
        "render_transports": sorted(
            {str(record["renderer"]) for record in all_records}
        ),
        "source_usd_path": str(source),
        "source_usd_sha256": source_before["sha256"],
        "validation_report_path": str(validation),
        "validation_report_sha256": validation_before["sha256"],
        "simready_validation": {
            "status": validation_result["status"],
            "passed": validation_result["passed"],
            "profile_name": validation_result.get("profile_name"),
            "profile_version": validation_result.get("profile_version"),
        },
        "usd_dependencies": dependencies_before,
        "render_settings": {
            "width": width,
            "height": height,
            "turntable_frame_count": turntable_frame_count,
            "turntable_fps": turntable_fps,
            "scene_tool_timeout_seconds": scene_tool_timeout_seconds,
        },
        "usd_cli": {
            "session_id": session.session_id,
            "source_revision": session.route.source_revision,
            "probe_response": probe_record,
            "command_receipt": receipt,
            "receipt_checkpoint": checkpoint,
            "render_records": records_binding,
        },
        "renderer_records": render_records,
        "artifacts": {
            "hero": {
                **_file_record(hero_path, media_type="image/png"),
                **_image_metadata(hero_path),
            },
            "multiview": {
                **_file_record(multiview_path, media_type="image/png"),
                **multiview_metadata,
            },
            "turntable_image": {
                **_file_record(gif_path, media_type="image/gif"),
                **gif_metadata,
            },
        },
        "errors": [],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, payload, within=report.parent)
    return payload
