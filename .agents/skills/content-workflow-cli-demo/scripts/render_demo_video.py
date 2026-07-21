#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a polished demo video from content-workflow-cli run artifacts."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_RUN_DIR = ".local-runs/content-workflow-cli/ladder-product-demo"
DEFAULT_REFERENCE_IMAGE = (
    "apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg"
)
DEFAULT_OUTPUT = (
    ".local-runs/content-workflow-cli-demo/ladder-product-demo-video/"
    "content_workflow_cli_ladder_demo.mp4"
)
DEFAULT_RENDER_ASSETS_DIR = (
    ".local-runs/content-workflow-cli-demo/ladder-product-demo-rerender/assets"
)
DEFAULT_USD = "apps/material_agent/data/examples/ladder/sources/usd/ladder.usd"
DEFAULT_PROMPT = "author reference-matched materials"
DEFAULT_TARGET_DESCRIPTION = (
    "brushed aluminum rails and steps, blue top and tray, and black rubber feet"
)
DEFAULT_ASSIGNED_USD_LABEL = "ladder_material_assignments.usda"

WIDTH = 1920
HEIGHT = 1080
FPS = 24
LEFT_WIDTH = 760
MARGIN = 32

BG = (17, 19, 24)
PANEL = (28, 32, 39)
PANEL_2 = (11, 13, 17)
TERM_BG = (11, 13, 17)
TEXT = (248, 250, 252)
MUTED = (148, 163, 184)
GREEN = (118, 185, 0)
GREEN_HOVER = (139, 211, 15)
BLUE = (96, 165, 250)
YELLOW = (245, 209, 112)
LINE = (52, 58, 69)
LINE_STRONG = (89, 97, 113)
FIT_IMAGE_CACHE: dict[tuple[str, int, int], Image.Image] = {}


@dataclass(frozen=True)
class Scene:
    title: str
    eyebrow: str
    image: Path
    lines: list[str]
    duration: float


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compose an artifact-backed product demo video from a completed "
            "content-workflow-cli run."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=Path(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--reference-image", type=Path, default=Path(DEFAULT_REFERENCE_IMAGE)
    )
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument(
        "--render-assets-dir",
        type=Path,
        default=Path(DEFAULT_RENDER_ASSETS_DIR),
    )
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument(
        "--width",
        type=int,
        default=WIDTH,
        help="Canvas width. The composed layout currently supports 1920 only.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=HEIGHT,
        help="Canvas height. The composed layout currently supports 1080 only.",
    )
    parser.add_argument("--runner-label", default="agent runner")
    parser.add_argument("--model-label", default="configured model")
    parser.add_argument("--effort-label", default="configured effort")
    parser.add_argument(
        "--usd",
        default=DEFAULT_USD,
        help="USD path to show in the composed terminal overlay.",
    )
    parser.add_argument(
        "--workflow-runner",
        default="codex",
        help="Workflow runner to show in the fallback terminal command.",
    )
    parser.add_argument(
        "--workflow-command",
        default=None,
        help=(
            "Exact command text to show in the terminal overlay. If omitted, "
            "a compact materials assign command is generated from the other inputs."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Workflow goal text to show in the baseline scene.",
    )
    parser.add_argument(
        "--target-description",
        default=DEFAULT_TARGET_DESCRIPTION,
        help="Reference-target description to show in the opening scene.",
    )
    parser.add_argument(
        "--assigned-usd-label",
        default=DEFAULT_ASSIGNED_USD_LABEL,
        help="Durable apply output filename to show in the final artifact list.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    validate_video_dimensions(args.width, args.height)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    assets = load_assets(run_dir, args.render_assets_dir, args.reference_image)
    scenes = build_scenes(
        run_dir,
        assets,
        runner_label=args.runner_label,
        model_label=args.model_label,
        effort_label=args.effort_label,
        usd=args.usd,
        reference_image=args.reference_image,
        workflow_runner=args.workflow_runner,
        workflow_command=args.workflow_command,
        prompt=args.prompt,
        target_description=args.target_description,
        assigned_usd_label=args.assigned_usd_label,
    )
    render_video(scenes, args.output, args.width, args.height, args.fps)
    print(args.output)
    return 0


def load_assets(
    run_dir: Path, render_assets_dir: Path, reference_image: Path
) -> dict[str, Path]:
    if render_assets_dir.exists():
        candidates = {
            "reference": reference_image,
            "initial": render_assets_dir / "baseline_contact_sheet.png",
            "initial_front": render_assets_dir / "baseline_front_plus_y.png",
            "initial_side": render_assets_dir / "baseline_right_plus_x.png",
            "logo": render_assets_dir / "final_front_plus_x.png",
            "preview_front": render_assets_dir / "final_front_plus_x.png",
            "preview_side": render_assets_dir / "final_side_plus_y.png",
            "final": render_assets_dir / "final_contact_sheet.png",
            "final_front": render_assets_dir / "final_front_plus_x.png",
            "final_oblique": render_assets_dir
            / "final_oblique_plus_x_minus_y_plus_z.png",
        }
        missing = [path for path in candidates.values() if not path.exists()]
        if not missing:
            return candidates

    evidence = run_dir / "evidence_renders"
    final = run_dir / "final_renders"
    candidates = {
        "reference": reference_image,
        "initial": evidence / "initial_contact_sheet.png",
        "initial_front": evidence / "initial_front_plus_y.png",
        "initial_side": evidence / "initial_right_plus_x.png",
        "logo": evidence / "preview_test_logo_dark_gray_front.png",
        "preview_front": evidence / "preview_overrides_front_plus_x.png",
        "preview_side": evidence / "preview_overrides_side_plus_y.png",
        "final": final / "final_contact_sheet.png",
        "final_front": final / "final_front_plus_x.png",
        "final_oblique": final / "final_oblique_plus_x_minus_y_plus_z.png",
    }
    fallback = next(run_dir.glob("**/*.png"), None)
    if fallback is None:
        raise FileNotFoundError(f"No PNG artifacts found in {run_dir}")
    return {
        key: path if path.exists() else fallback for key, path in candidates.items()
    }


def load_events(run_dir: Path) -> list[tuple[str, str]]:
    path = run_dir / "trace" / "events.jsonl"
    if not path.exists():
        return []
    events: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        phase = str(data.get("phase") or "trace")
        summary = str(data.get("summary") or "")
        events.append((phase, summary))
    return events


def summary_table_metric(line: str, label: str) -> str | None:
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    if len(cells) < 2:
        return None
    value = cells[1]
    return f"{value} {label}" if value else None


def load_summary_lines(run_dir: Path) -> list[str]:
    path = run_dir / "final_summary.md"
    if not path.exists():
        return ["Final summary not found."]
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("Coverage invariant"):
            lines.append(clean_markup(line))
        elif line.startswith("Status:"):
            lines.append(clean_markup(line))
        elif line.startswith("| Visible/renderable"):
            metric = summary_table_metric(line, "visible/renderable candidates")
            if metric:
                lines.append(metric)
        elif line.startswith("| Material decision"):
            metric = summary_table_metric(line, "explicit material decisions")
            if metric:
                lines.append(metric)
        elif line.startswith("| Preview override"):
            metric = summary_table_metric(line, "preview override prims")
            if metric:
                lines.append(metric)
    return lines[:6] or ["Final artifacts generated."]


def load_stats_lines(
    run_dir: Path, *, runner_label: str, model_label: str, effort_label: str
) -> list[str]:
    api_counts = read_json(run_dir / "api_operation_counts.json")
    runner_result = read_runner_result(run_dir)
    raw_usage = runner_result.get("usage", {})
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    trace_path = run_dir / "trace" / "events.jsonl"
    trace_events = (
        sum(
            1
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if trace_path.exists()
        else 0
    )

    return [
        "$ cat workflow_stats.json",
        f"Environment: {runner_label} / {model_label} / {effort_label}",
        f"Input tokens: {format_count(usage.get('input_tokens'))}",
        f"Cached input tokens: {format_count(usage.get('cached_input_tokens'))}",
        f"Output tokens: {format_count(usage.get('output_tokens'))}",
        f"Reasoning output tokens: {format_count(usage.get('reasoning_output_tokens'))}",
        (
            "Content Authoring Tool API queries: "
            f"{format_count(api_counts.get('api_operation_count_total'))} total / "
            f"{format_count(api_counts.get('api_operation_count_successful_workflow'))} successful"
        ),
        (
            f"Render queries: {format_count(api_counts.get('render_count_total'))} / "
            f"final renders: {format_count(api_counts.get('final_renders'))}"
        ),
        (
            f"Pick queries: {format_count(api_counts.get('pick_calls'))} / "
            f"preview overrides: {format_count(api_counts.get('preview_override_commands'))}"
        ),
        (
            "Material coverage: "
            f"{format_count(api_counts.get('coverage_candidate_visible_prims'))} candidates / "
            f"{format_count(api_counts.get('coverage_material_decision_prims'))} decisions"
        ),
        (
            "Issues fixed: "
            f"{format_count(api_counts.get('final_review_issues_fixed'))} final review / "
            f"{format_count(api_counts.get('visual_quality_issues_fixed'))} visual"
        ),
        f"Trace events: {format_count(trace_events)}",
    ]


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def read_runner_result(run_dir: Path) -> dict[str, object]:
    raw_dir = run_dir / "raw"
    for filename in ("codex_result.json", "claude_result.json", "runner_result.json"):
        data = read_json(raw_dir / filename)
        if data:
            return data
    return {}


def clean_markup(line: str) -> str:
    return (
        line.replace("**", "")
        .replace("`", "")
        .replace("Workbench", "Content Authoring Tool")
    )


def format_count(value: object) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return str(value) if value not in (None, "") else "n/a"


def build_scenes(
    run_dir: Path,
    assets: dict[str, Path],
    *,
    runner_label: str,
    model_label: str,
    effort_label: str,
    usd: str,
    reference_image: Path,
    workflow_runner: str,
    workflow_command: str | None,
    prompt: str,
    target_description: str,
    assigned_usd_label: str,
) -> list[Scene]:
    events = load_events(run_dir)
    summary = load_summary_lines(run_dir)
    stats = load_stats_lines(
        run_dir,
        runner_label=runner_label,
        model_label=model_label,
        effort_label=effort_label,
    )
    command = terminal_command_lines(
        workflow_command,
        usd=usd,
        reference_image=reference_image,
        workflow_runner=workflow_runner,
        output_dir=run_dir,
        output_usd=run_dir / assigned_usd_label,
    )

    def event_lines(start: int, end: int) -> list[str]:
        return [
            f"[{phase}] {clean_markup(summary)}" for phase, summary in events[start:end]
        ]

    return [
        Scene(
            title="Reference Target",
            eyebrow="REFERENCE",
            image=assets["reference"],
            duration=4.0,
            lines=(
                [
                    f"$ open {reference_image.name}",
                    "",
                    "Target material read:",
                ]
                + textwrap.wrap(
                    target_description,
                    width=42,
                    break_long_words=False,
                    break_on_hyphens=False,
                )[:4]
            ),
        ),
        Scene(
            title="Unmaterialized Baseline",
            eyebrow="BASELINE",
            image=assets["initial_front"],
            duration=4.5,
            lines=command
            + [
                "",
                f"Start with neutral asset: {Path(usd).name}",
                "Goal:",
            ]
            + textwrap.wrap(
                prompt,
                width=52,
                break_long_words=False,
                break_on_hyphens=False,
            )[:4],
        ),
        Scene(
            title="Open The Content Authoring Tool",
            eyebrow="SESSION",
            image=assets["initial"],
            duration=4.5,
            lines=["$ trace"] + event_lines(0, 3),
        ),
        Scene(
            title="Inspect Visible Material Candidates",
            eyebrow="INSPECTION",
            image=assets["initial_side"],
            duration=5.5,
            lines=["$ trace"] + event_lines(2, 6),
        ),
        Scene(
            title="Preview A Precise Material Override",
            eyebrow="AUTHORING",
            image=assets["logo"],
            duration=4.5,
            lines=["$ trace"] + event_lines(5, 8),
        ),
        Scene(
            title="Review And Iterate In Render",
            eyebrow="VALIDATION",
            image=assets["preview_front"],
            duration=5.5,
            lines=["$ trace"] + event_lines(7, 10),
        ),
        Scene(
            title="Verify Final Looks",
            eyebrow="FINAL RENDERS",
            image=assets["final"],
            duration=5.5,
            lines=["$ trace"] + event_lines(9, 12),
        ),
        Scene(
            title="Apply Accepted Materials",
            eyebrow="DURABLE OUTPUT",
            image=assets["final_front"],
            duration=4.5,
            lines=[
                f"$ ls {run_dir.name}",
                "assignments.json",
                "visual_quality_assessment.json",
                assigned_usd_label,
                "trace/operation_trace.md",
            ]
            + event_lines(max(0, len(events) - 2), len(events)),
        ),
        Scene(
            title="Evidence Package Ready",
            eyebrow="RESULT",
            image=assets["final_oblique"],
            duration=5.0,
            lines=["$ cat final_summary.md"] + summary,
        ),
        Scene(
            title="Run Stats And Environment",
            eyebrow="STATS",
            image=assets["final_front"],
            duration=6.0,
            lines=stats,
        ),
    ]


def terminal_command_lines(
    workflow_command: str | None,
    *,
    usd: str,
    reference_image: Path,
    workflow_runner: str,
    output_dir: Path,
    output_usd: Path,
) -> list[str]:
    if workflow_command:
        lines = [line.rstrip() for line in workflow_command.strip().splitlines()]
        return [
            line if index > 0 or line.lstrip().startswith("$") else f"$ {line}"
            for index, line in enumerate(lines)
            if line.strip()
        ]
    return [
        "$ content-workflow-cli materials assign",
        f"  --usd {usd}",
        f"  --reference-image {reference_image.name}",
        "  --materials-yaml material_libs_default/materials.yaml",
        f"  --output-dir {output_dir}",
        f"  --output-usd {output_usd}",
        f"  --runner {workflow_runner} --keep-workbench",
    ]


def render_video(
    scenes: list[Scene], output: Path, width: int, height: int, fps: int
) -> None:
    validate_video_dimensions(width, height)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output}")

    fonts = Fonts()
    frame_index = 0
    total_frames = sum(max(1, int(scene.duration * fps)) for scene in scenes)
    for scene_index, scene in enumerate(scenes):
        scene_frames = max(1, int(scene.duration * fps))
        for local_frame in range(scene_frames):
            progress = frame_index / max(1, total_frames - 1)
            reveal = ease_out(local_frame / max(1, scene_frames - 1))
            frame = draw_frame(
                scene=scene,
                scene_index=scene_index,
                scene_count=len(scenes),
                progress=progress,
                reveal=reveal,
                blink=frame_index % fps < fps // 2,
                fonts=fonts,
                width=width,
                height=height,
            )
            writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
            frame_index += 1
    writer.release()


class Fonts:
    def __init__(self) -> None:
        sans = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        sans_bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        mono = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        self.title = load_font(sans_bold, 46)
        self.heading = load_font(sans_bold, 34)
        self.body = load_font(sans, 25)
        self.small = load_font(sans, 20)
        self.mono = load_font(mono, 21)
        self.mono_small = load_font(mono, 18)


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def validate_video_dimensions(width: int, height: int) -> None:
    if (width, height) != (WIDTH, HEIGHT):
        raise ValueError(
            f"Demo video layout is fixed at {WIDTH}x{HEIGHT}; got {width}x{height}. "
            "Render at the default size, then scale the MP4 as a post-process step."
        )


def draw_frame(
    *,
    scene: Scene,
    scene_index: int,
    scene_count: int,
    progress: float,
    reveal: float,
    blink: bool,
    fonts: Fonts,
    width: int,
    height: int,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw_header(draw, scene, scene_index, scene_count, progress, fonts, width)
    draw_terminal(draw, scene.lines, reveal, blink, fonts, height)
    draw_workbench(canvas, draw, scene, fonts, width, height)
    draw_footer(draw, fonts, width, height)
    return canvas


def draw_header(
    draw: ImageDraw.ImageDraw,
    scene: Scene,
    scene_index: int,
    scene_count: int,
    progress: float,
    fonts: Fonts,
    width: int,
) -> None:
    draw.text(
        (MARGIN, 24), "Agentic Content Authoring Workflow", font=fonts.title, fill=TEXT
    )
    draw.text(
        (MARGIN, 78),
        "Content authoring with the Content Authoring Tool",
        font=fonts.small,
        fill=MUTED,
    )
    pill = f"{scene.eyebrow}  {scene_index + 1}/{scene_count}"
    x0 = width - 330
    draw.rounded_rectangle(
        (x0, 36, width - MARGIN, 78), radius=6, fill=PANEL_2, outline=LINE
    )
    draw.text((x0 + 22, 45), pill, font=fonts.small, fill=GREEN_HOVER)
    bar_x0, bar_y0 = MARGIN, 124
    bar_x1 = width - MARGIN
    draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y0 + 8), radius=4, fill=PANEL)
    draw.rounded_rectangle(
        (bar_x0, bar_y0, bar_x0 + int((bar_x1 - bar_x0) * progress), bar_y0 + 8),
        radius=4,
        fill=GREEN,
    )


def draw_terminal(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    reveal: float,
    blink: bool,
    fonts: Fonts,
    height: int,
) -> None:
    x0, y0, x1, y1 = MARGIN, 160, LEFT_WIDTH - MARGIN, height - 58
    draw.rounded_rectangle(
        (x0, y0, x1, y1), radius=8, fill=TERM_BG, outline=LINE, width=2
    )
    draw.text((x0 + 24, y0 + 20), "Terminal", font=fonts.small, fill=MUTED)
    all_lines: list[str] = []
    for line in lines:
        all_lines.extend(wrap_line(line, 54))
    visible_count = max(1, math.ceil(len(all_lines) * reveal))
    visible = all_lines[:visible_count][-25:]
    y = y0 + 62
    for line in visible:
        color = TEXT
        if line.startswith("$"):
            color = GREEN
        elif line.startswith("["):
            color = BLUE
        elif "unresolved" in line.lower() or "granularity" in line.lower():
            color = YELLOW
        draw.text((x0 + 24, y), line, font=fonts.mono_small, fill=color)
        y += 29
    if blink and visible_count >= len(all_lines):
        draw.rectangle(
            (x0 + 24, min(y, y1 - 36), x0 + 37, min(y + 22, y1 - 14)), fill=GREEN
        )


def draw_workbench(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    scene: Scene,
    fonts: Fonts,
    width: int,
    height: int,
) -> None:
    x0, y0 = LEFT_WIDTH, 160
    x1, y1 = width - MARGIN, height - 58
    draw.rounded_rectangle(
        (x0, y0, x1, y1), radius=8, fill=PANEL, outline=LINE, width=2
    )
    draw.text(
        (x0 + 30, y0 + 22), "Content Authoring Tool", font=fonts.small, fill=MUTED
    )
    draw.text((x0 + 30, y0 + 58), scene.title, font=fonts.heading, fill=TEXT)
    image_box = (x0 + 30, y0 + 116, x1 - 30, y1 - 78)
    paste_fit(canvas, scene.image, image_box)
    draw.rounded_rectangle(
        (x0 + 30, y1 - 52, x1 - 30, y1 - 18),
        radius=6,
        fill=PANEL_2,
        outline=LINE,
        width=1,
    )
    draw.text(
        (x0 + 48, y1 - 46),
        "Representative replay composed from real run artifacts",
        font=fonts.small,
        fill=MUTED,
    )


def draw_footer(
    draw: ImageDraw.ImageDraw, fonts: Fonts, width: int, height: int
) -> None:
    draw.text(
        (MARGIN, height - 34),
        "Observable API calls, render queries, preview overrides, and final artifacts",
        font=fonts.small,
        fill=MUTED,
    )
    draw.text(
        (width - 258, height - 34),
        "Content authoring demo",
        font=fonts.small,
        fill=MUTED,
    )


def paste_fit(canvas: Image.Image, path: Path, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    max_w = x1 - x0
    max_h = y1 - y0
    cache_key = (str(path), max_w, max_h)
    image = FIT_IMAGE_CACHE.get(cache_key)
    if image is None:
        source = Image.open(path).convert("RGB")
        scale = min(max_w / source.width, max_h / source.height)
        size = (max(1, int(source.width * scale)), max(1, int(source.height * scale)))
        image = source.resize(size, Image.Resampling.LANCZOS)
        FIT_IMAGE_CACHE[cache_key] = image
    px = x0 + (max_w - image.width) // 2
    py = y0 + (max_h - image.height) // 2
    canvas.paste(image, (px, py))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (px, py, px + image.width, py + image.height),
        outline=LINE_STRONG,
        width=2,
    )


def wrap_line(text: str, width: int) -> list[str]:
    if not text:
        return [""]
    if len(text) <= width:
        return [text]
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def ease_out(value: float) -> float:
    return 1 - (1 - value) * (1 - value)


if __name__ == "__main__":
    raise SystemExit(main())
