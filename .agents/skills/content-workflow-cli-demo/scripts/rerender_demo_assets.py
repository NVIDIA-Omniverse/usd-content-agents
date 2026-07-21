#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rerender demo assets with real Content Authoring Tool lighting."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

DISPLAY_COLOR_ATTRS = (
    "primvars:displayColor",
    "primvars:displayColor:indices",
    "primvars:displayOpacity",
)

DEFAULT_SOURCE_USD = "apps/material_agent/data/examples/ladder/sources/usd/ladder.usd"
DEFAULT_ASSIGNED_USD = (
    ".local-runs/content-workflow-cli/ladder-product-demo/"
    "ladder_material_assignments.usda"
)
DEFAULT_OUTPUT_DIR = (
    ".local-runs/content-workflow-cli-demo/ladder-product-demo-rerender"
)


@dataclass(frozen=True)
class RenderSpec:
    filename: str
    direction: str
    label: str


BASELINE_RENDERS = [
    RenderSpec("baseline_front_plus_y.png", "+y", "baseline +y"),
    RenderSpec("baseline_right_plus_x.png", "+x", "baseline +x"),
    RenderSpec(
        "baseline_oblique_plus_x_minus_y_plus_z.png", "+x-y+z", "baseline oblique"
    ),
]

ASSIGNED_RENDERS = [
    RenderSpec("final_front_plus_x.png", "+x", "final +x"),
    RenderSpec("final_side_plus_y.png", "+y", "final +y"),
    RenderSpec("final_oblique_plus_x_minus_y_plus_z.png", "+x-y+z", "final oblique"),
    RenderSpec("final_back_minus_x.png", "-x", "final -x"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create no-authored-material clay baseline renders and final "
            "material renders through the Content Authoring Tool render API."
        )
    )
    parser.add_argument(
        "--workbench-url",
        default="http://127.0.0.1:8088",
        help=(
            "Local Content Authoring Tool endpoint. Rerendering posts local "
            "scene paths, so the service must be able to read this filesystem."
        ),
    )
    parser.add_argument("--source-usd", type=Path, default=Path(DEFAULT_SOURCE_USD))
    parser.add_argument("--assigned-usd", type=Path, default=Path(DEFAULT_ASSIGNED_USD))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--hdri-light", type=float, default=600.0)
    parser.add_argument("--dome-light", type=float, default=None)
    parser.add_argument("--distant-light", type=float, default=None)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=576)
    args = parser.parse_args()

    source_usd = args.source_usd.expanduser().resolve()
    assigned_usd = args.assigned_usd.expanduser().resolve()
    if not assigned_usd.exists():
        raise FileNotFoundError(
            "Assigned USD does not exist. Pass --assigned-usd with the durable "
            f"material apply output path from the workflow run: {assigned_usd}"
        )
    output_dir = args.output_dir.expanduser().resolve()
    assets_dir = output_dir / "assets"
    scenes_dir = output_dir / "scenes"
    responses_dir = output_dir / "responses"
    for directory in (assets_dir, scenes_dir, responses_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unmaterialized_usd, clay_baseline_usd = scene_overlay_paths(scenes_dir, source_usd)
    blocked_counts = write_unmaterialized_overlay(source_usd, unmaterialized_usd)
    write_clay_baseline_overlay(unmaterialized_usd, clay_baseline_usd)

    baseline_paths = render_set(
        workbench_url=args.workbench_url,
        scene_path=clay_baseline_usd,
        specs=BASELINE_RENDERS,
        assets_dir=assets_dir,
        responses_dir=responses_dir / "baseline",
        width=args.width,
        height=args.height,
        hdri_light=args.hdri_light,
        dome_light=args.dome_light,
        distant_light=args.distant_light,
    )
    assigned_paths = render_set(
        workbench_url=args.workbench_url,
        scene_path=assigned_usd,
        specs=ASSIGNED_RENDERS,
        assets_dir=assets_dir,
        responses_dir=responses_dir / "assigned",
        width=args.width,
        height=args.height,
        hdri_light=args.hdri_light,
        dome_light=args.dome_light,
        distant_light=args.distant_light,
    )

    make_contact_sheet(
        [
            (baseline_paths["baseline_front_plus_y.png"], "no material +y"),
            (baseline_paths["baseline_right_plus_x.png"], "no material +x"),
            (
                baseline_paths["baseline_oblique_plus_x_minus_y_plus_z.png"],
                "no material oblique",
            ),
        ],
        assets_dir / "baseline_contact_sheet.png",
    )
    make_contact_sheet(
        [
            (assigned_paths["final_front_plus_x.png"], "final +x"),
            (assigned_paths["final_side_plus_y.png"], "final +y"),
            (
                assigned_paths["final_oblique_plus_x_minus_y_plus_z.png"],
                "final oblique",
            ),
            (assigned_paths["final_back_minus_x.png"], "final -x"),
        ],
        assets_dir / "final_contact_sheet.png",
    )
    metadata = {
        "source_usd": str(source_usd),
        "assigned_usd": str(assigned_usd),
        "unmaterialized_usd": str(unmaterialized_usd),
        "clay_baseline_usd": str(clay_baseline_usd),
        "blocked_material_binding_relationships": blocked_counts["material_bindings"],
        "blocked_display_color_attributes": blocked_counts["display_colors"],
        "instance_prims_not_traversed": blocked_counts["instances"],
        "baseline_render": (
            "neutral matte clay material bound over source with authored "
            "material bindings and display colors blocked"
        ),
        "warnings": (
            [
                "Source stage contains instance prims; prototype-internal "
                "materials or display colors may require explicit un-instancing."
            ]
            if blocked_counts["instances"]
            else []
        ),
        "hdri_light": args.hdri_light,
        "dome_light": args.dome_light,
        "distant_light": args.distant_light,
        "assets_dir": str(assets_dir),
    }
    (output_dir / "rerender_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(assets_dir)
    return 0


def write_unmaterialized_overlay(source_usd: Path, output_usd: Path) -> dict[str, int]:
    source_stage = Usd.Stage.Open(str(source_usd))
    if source_stage is None:
        raise RuntimeError(f"Failed to open source USD: {source_usd}")

    binding_relationships: list[tuple[str, list[str]]] = []
    display_color_attrs: list[
        tuple[str, list[tuple[str, Sdf.ValueTypeName, bool]]]
    ] = []
    instance_count = 0
    for prim in source_stage.Traverse():
        if prim.IsInstance():
            instance_count += 1
        names = [
            rel.GetName()
            for rel in prim.GetRelationships()
            if rel.GetName().startswith("material:binding")
        ]
        if names:
            binding_relationships.append((str(prim.GetPath()), names))
        attrs = []
        for name in DISPLAY_COLOR_ATTRS:
            attr = prim.GetAttribute(name)
            if attr and attr.HasAuthoredValueOpinion():
                attrs.append((name, attr.GetTypeName(), attr.IsCustom()))
        if attrs:
            display_color_attrs.append((str(prim.GetPath()), attrs))

    output_usd.unlink(missing_ok=True)
    layer = Sdf.Layer.CreateNew(str(output_usd))
    layer.subLayerPaths.append(str(source_usd))
    stage = Usd.Stage.Open(layer.identifier)
    if stage is None:
        raise RuntimeError(f"Failed to create unmaterialized overlay: {output_usd}")
    for prim_path, relationship_names in binding_relationships:
        prim = stage.OverridePrim(prim_path)
        for name in relationship_names:
            prim.CreateRelationship(name, custom=False).SetTargets([])
    for prim_path, attrs in display_color_attrs:
        prim = stage.OverridePrim(prim_path)
        for name, type_name, is_custom in attrs:
            prim.CreateAttribute(name, type_name, custom=is_custom).Block()
    stage.GetRootLayer().Save()

    check_stage = Usd.Stage.Open(str(output_usd))
    if check_stage is None:
        raise RuntimeError(f"Failed to verify unmaterialized overlay: {output_usd}")
    remaining = []
    for prim in check_stage.Traverse():
        material, _relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial()
        if material:
            remaining.append((str(prim.GetPath()), str(material.GetPath())))
    if remaining:
        raise RuntimeError(
            "Unmaterialized overlay still has bound materials: "
            + ", ".join(f"{prim}->{material}" for prim, material in remaining[:8])
        )
    remaining_display_colors = []
    for prim in check_stage.Traverse():
        for name in DISPLAY_COLOR_ATTRS:
            attr = prim.GetAttribute(name)
            if attr and (attr.Get() is not None or attr.GetNumTimeSamples() > 0):
                remaining_display_colors.append((str(prim.GetPath()), name))
    if remaining_display_colors:
        raise RuntimeError(
            "Unmaterialized overlay still has authored display colors: "
            + ", ".join(f"{prim}.{name}" for prim, name in remaining_display_colors[:8])
        )
    return {
        "material_bindings": sum(
            len(names) for _prim_path, names in binding_relationships
        ),
        "display_colors": sum(len(attrs) for _prim_path, attrs in display_color_attrs),
        "instances": instance_count,
    }


def write_clay_baseline_overlay(unmaterialized_usd: Path, output_usd: Path) -> None:
    output_usd.unlink(missing_ok=True)
    layer = Sdf.Layer.CreateNew(str(output_usd))
    layer.subLayerPaths.append(str(unmaterialized_usd))
    stage = Usd.Stage.Open(layer.identifier)
    if stage is None:
        raise RuntimeError(f"Failed to create clay baseline overlay: {output_usd}")

    root = stage.GetDefaultPrim()
    if not root:
        children = list(stage.GetPseudoRoot().GetChildren())
        if not children:
            raise RuntimeError(f"Clay baseline overlay has no root prim: {output_usd}")
        root = children[0]

    material = UsdShade.Material.Define(stage, "/ClayBaseline/NeutralGray")
    shader = UsdShade.Shader.Define(stage, "/ClayBaseline/NeutralGray/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.34, 0.34, 0.34)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.88)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(root).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
    stage.GetRootLayer().Save()


def render_set(
    *,
    workbench_url: str,
    scene_path: Path,
    specs: list[RenderSpec],
    assets_dir: Path,
    responses_dir: Path,
    width: int,
    height: int,
    hdri_light: float,
    dome_light: float | None,
    distant_light: float | None,
) -> dict[str, Path]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    session = post_json(
        workbench_url,
        "/sessions",
        {
            "scene_path": str(scene_path),
            "optimize": False,
            "width": width,
            "height": height,
        },
    )
    session_id = require_response_string(session, "/sessions", "session_id")
    outputs: dict[str, Path] = {}
    try:
        for spec in specs:
            request = {
                "width": width,
                "height": height,
                "use_session_camera": False,
                "direction": spec.direction,
                "margin": 1.22,
                "render_quality": "final",
                "hdri_light": hdri_light,
                "dome_light": dome_light,
                "distant_light": distant_light,
                "save_camera_json": True,
            }
            render_path = f"/sessions/{session_id}/render"
            response = post_json(
                workbench_url,
                render_path,
                request,
                timeout_seconds=240,
            )
            image_path = require_response_string(response, render_path, "image_path")
            if not Path(image_path).exists():
                raise RuntimeError(
                    f"Render response image_path does not exist for {render_path}: "
                    f"{image_path}"
                )
            output_path = assets_dir / spec.filename
            shutil.copy2(image_path, output_path)
            (responses_dir / f"{output_path.stem}.json").write_text(
                json.dumps({"request": request, "response": response}, indent=2),
                encoding="utf-8",
            )
            outputs[spec.filename] = output_path
    finally:
        delete_session(workbench_url, session_id)
    return outputs


def scene_overlay_paths(scenes_dir: Path, source_usd: Path) -> tuple[Path, Path]:
    source_stem = source_usd.stem or "asset"
    return (
        scenes_dir / f"{source_stem}_unmaterialized.usda",
        scenes_dir / f"{source_stem}_clay_baseline.usda",
    )


def require_response_string(response: dict[str, object], path: str, key: str) -> str:
    value = response.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            f"Unexpected response from {path}: missing string `{key}` in {response!r}"
        )
    return value


def delete_session(base_url: str, session_id: str) -> None:
    url = f"{base_url.rstrip('/')}/sessions/{session_id}"
    request = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(request, timeout=30):
            pass
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        pass


def post_json(
    base_url: str,
    path: str,
    payload: dict[str, object],
    *,
    timeout_seconds: int = 120,
) -> dict[str, object]:
    url = base_url.rstrip("/") + path
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to {url}: {exc.reason}") from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}: {body[:200]!r}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response from {path}: {data!r}")
    return data


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def make_contact_sheet(items: list[tuple[Path, str]], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path, _label in items]
    thumb_width = 384
    thumb_height = 288
    label_height = 30
    cols = 2
    rows = (len(items) + cols - 1) // cols
    font = load_font(18)
    sheet = Image.new(
        "RGB", (thumb_width * cols, rows * (thumb_height + label_height)), (20, 22, 25)
    )
    draw = ImageDraw.Draw(sheet)
    for index, (image, (_path, label)) in enumerate(zip(images, items, strict=False)):
        col = index % cols
        row = index // cols
        x = col * thumb_width
        y = row * (thumb_height + label_height)
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        px = x + (thumb_width - image.width) // 2
        py = y + label_height + (thumb_height - image.height) // 2
        draw.text((x + 10, y + 5), label, font=font, fill=(235, 238, 242))
        sheet.paste(image, (px, py))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


if __name__ == "__main__":
    raise SystemExit(main())
