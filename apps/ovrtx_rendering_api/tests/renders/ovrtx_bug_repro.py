#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression reproduction for a historical ovrtx visibility segfault.

ovrtx 0.2.0.280040 crashed with SIGSEGV (exit code -11) when rendering a USD
stage that contains time-sampled visibility attributes on 3+ prims. The crash
occurred inside libovrtx.dylib.so in std::vector::_M_realloc_insert.

Reproduction:
    REPRO_DIR="$(cd "$(dirname apps/ovrtx_rendering_api/tests/renders/ovrtx_bug_repro.py)" && pwd)"
    REPO_ROOT="$(git -C "$REPRO_DIR" rev-parse --show-toplevel)"
    OVRTX_VENV="${WU_OVRTX_VENV_DIR:-$HOME/.cache/wu/ovrtx_venv}"
    RUNTIME_LOCK="$REPO_ROOT/world_understanding/functions/graphics/pylock.ovrtx-runtime.toml"
    uv venv --python 3.12 "$OVRTX_VENV"
    uv pip install --python "$OVRTX_VENV/bin/python" --require-hashes --no-deps \
      -r "$RUNTIME_LOCK" --no-config --no-sources
    "$OVRTX_VENV/bin/python" "$REPRO_DIR/ovrtx_bug_repro.py" --mode both

The USD file (ovrtx_bug_repro.usda) contains:
- 3 cubes at different positions
- Time-sampled visibility: each cube visible at one frame, invisible at others
  - Part0: visible at frame 0, invisible at frame 1+
  - Part1: visible at frame 1, invisible at frame 0 and 2
  - Part2: visible at frame 2, invisible at frame 0 and 1
- A fixed camera and key light

Expected: renders 3 frames (one per visible cube)
Historical actual on ovrtx 0.2.0.280040: SIGSEGV in libovrtx.dylib.so

Notes:
- 2 prims with visibility toggling works fine
- 3+ prims crashes
- Rotation animation (no visibility changes) works fine with any number of prims
- Single-frame render of a stage containing visibility keyframes works at frame 0
  but crashes at frame 1+

Workaround: production strips time-sampled visibility from the base USD and
loads frame-specific static visibility overlays. Direct write_attribute
visibility updates also crashed on some ovrtx 0.2.0 runtimes.
"""

import logging
import os
import sys
import tempfile
from argparse import ArgumentParser
from importlib import metadata

logger = logging.getLogger(__name__)

EXPECTED_VISIBLE_PRIMS = {
    0: "/World/Part0",
    1: "/World/Part1",
    2: "/World/Part2",
}

STATIC_VISIBILITY_SCENE = """#usda 1.0
(
    endTimeCode = 2
    startTimeCode = 0
    timeCodesPerSecond = 24
    upAxis = "Y"
)

def "World"
{
    def Cube "Part0"
    {
        color3f[] primvars:displayColor = [(0.5, 0.3, 0.8)]
        double size = 0.5
        token visibility = "inherited"
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Cube "Part1"
    {
        color3f[] primvars:displayColor = [(0.5, 0.3, 0.8)]
        double size = 0.5
        token visibility = "inherited"
        double3 xformOp:translate = (1.5, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Cube "Part2"
    {
        color3f[] primvars:displayColor = [(0.5, 0.3, 0.8)]
        double size = 0.5
        token visibility = "inherited"
        double3 xformOp:translate = (3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def SphereLight "KeyLight"
    {
        float inputs:intensity = 30000
        float inputs:radius = 1
        double3 xformOp:translate = (1.5, 3, 4)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}

def "Cameras"
{
    def Camera "Cam"
    {
        float2 clippingRange = (25.427465, 28.562498)
        float focalLength = 50
        float horizontalAperture = 36
        float verticalAperture = 36
        matrix4d xformOp:transform = ( (0.7071067811865475, 0, -0.7071067811865476, 0), (-0.40824829046386313, 0.8164965809277263, -0.4082482904638631, 0), (0.5773502691896258, 0.5773502691896257, 0.5773502691896257, 0), (17.06197389230382, 15.561973892303817, 15.561973892303817, 1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
"""


def _usda_asset_path(path: str) -> str:
    """Return a simple USDA asset path for local repro files."""
    return f"@{path.replace('@', '%40')}@"


def _write_render_products(tmp_dir: str) -> str:
    # Build render products USDA
    render_products = """#usda 1.0

def Scope "Render"
{
    def RenderProduct "Product_Cameras_Cam"
    {
        rel camera = </Cameras/Cam>
        rel orderedVars = [</Render/Vars/LdrColor>]
        uniform int2 resolution = (512, 512)
    }

    def Scope "Vars"
    {
        def RenderVar "LdrColor"
        {
            uniform string sourceName = "LdrColor"
        }
    }
}
"""

    rp_path = os.path.join(tmp_dir, "render_products.usda")
    with open(rp_path, "w", encoding="utf-8") as f:
        f.write(render_products)
    return rp_path


def _write_combined_scene(
    scene_path: str,
    render_products_path: str,
    tmp_dir: str,
    name: str,
) -> str:
    # Sublayer scene + render products without requiring external pxr bindings
    # in the isolated OVRTX repro environment.
    combined_path = os.path.join(tmp_dir, f"{name}.usda")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(
            "#usda 1.0\n"
            "(\n"
            "    subLayers = [\n"
            f"        {_usda_asset_path(scene_path)},\n"
            f"        {_usda_asset_path(render_products_path)}\n"
            "    ]\n"
            ")\n"
        )
    return combined_path


def _save_frame_products(
    ovrtx,
    products,
    tmp_dir: str,
    mode: str,
    frame: int,
) -> list[str]:
    import numpy as np
    from PIL import Image

    output_paths = []
    product_items = list(products.items())
    for product_index, (_name, product) in enumerate(product_items):
        frame_items = list(product.frames)
        for product_frame_index, product_frame in enumerate(frame_items):
            if "LdrColor" not in product_frame.render_vars:
                continue
            with product_frame.render_vars["LdrColor"].map(
                device=ovrtx.Device.CPU
            ) as var:
                pixels = np.from_dlpack(var).copy()
            suffix = (
                ""
                if len(product_items) == 1 and len(frame_items) == 1
                else f"_p{product_index}_r{product_frame_index}"
            )
            out = os.path.join(tmp_dir, f"{mode}_frame_{frame}{suffix}.png")
            Image.fromarray(pixels).save(out)
            logger.info("Saved %s", out)
            output_paths.append(out)
    if not output_paths:
        raise RuntimeError(f"frame {frame} did not produce LdrColor")
    return output_paths


def _assert_no_blank_or_duplicate_frames(image_paths: list[str]) -> None:
    import numpy as np
    from PIL import Image

    arrays = []
    for image_path in image_paths:
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        brightness = rgb.max(axis=2)
        if float(brightness.max()) < 5.0 or int((brightness > 5.0).sum()) < 16:
            raise RuntimeError(f"blank render detected: {image_path}")
        arrays.append(rgb)

    diffs = []
    for left_idx in range(len(arrays)):
        for right_idx in range(left_idx + 1, len(arrays)):
            diff = float(np.abs(arrays[left_idx] - arrays[right_idx]).mean())
            diffs.append((left_idx, right_idx, diff))
            if diff <= 0.01:
                raise RuntimeError(
                    "frames are unexpectedly identical: "
                    f"{left_idx} vs {right_idx}, diff={diff}"
                )
    logger.info("Validation passed: no blank frames; frame RGB diffs=%s", diffs)


def _render_native_visibility(ovrtx, combined_path: str, tmp_dir: str) -> list[str]:
    logger.info("Loading native time-sampled visibility scene: %s", combined_path)
    renderer = ovrtx.Renderer()
    renderer.open_usd(combined_path)

    product_paths = {"/Render/Product_Cameras_Cam"}
    fps = 24.0
    outputs = []

    for frame in range(3):
        logger.info(
            "Rendering native frame %s; expected visible prim: %s",
            frame,
            EXPECTED_VISIBLE_PRIMS[frame],
        )
        renderer.update_from_usd_time(float(frame) / fps)
        renderer.reset()
        products = renderer.step(render_products=product_paths, delta_time=1.0 / 60.0)

        if not products:
            raise RuntimeError(f"native frame {frame} produced no render products")
        outputs.extend(_save_frame_products(ovrtx, products, tmp_dir, "native", frame))

    del renderer
    return outputs


def _render_write_attribute_visibility(
    ovrtx, combined_path: str, tmp_dir: str
) -> list[str]:
    logger.info(
        "Loading static visibility scene for write_attribute probe: %s",
        combined_path,
    )
    renderer = ovrtx.Renderer()
    renderer.open_usd(combined_path)

    product_paths = {"/Render/Product_Cameras_Cam"}
    outputs = []

    for frame in range(3):
        logger.info(
            "Rendering write_attribute frame %s; expected visible prim: %s",
            frame,
            EXPECTED_VISIBLE_PRIMS[frame],
        )
        renderer.update_from_usd_time(float(frame) / 24.0)
        renderer.reset()
        for prim_frame, prim_path in EXPECTED_VISIBLE_PRIMS.items():
            token = "inherited" if prim_frame == frame else "invisible"
            renderer.write_attribute([prim_path], "visibility", [token])
        products = renderer.step(render_products=product_paths, delta_time=1.0 / 60.0)

        if not products:
            raise RuntimeError(
                f"write_attribute frame {frame} produced no render products"
            )
        outputs.extend(
            _save_frame_products(ovrtx, products, tmp_dir, "write_attribute", frame)
        )

    del renderer
    return outputs


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("native", "write-attribute", "both"),
        default="native",
        help=(
            "native validates authored time-sampled visibility; write-attribute "
            "validates the legacy direct visibility update path"
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="save frames without checking for blank or duplicate output",
    )
    args = parser.parse_args()

    import ovrtx

    logger.info("ovrtx version: %s", metadata.version("ovrtx"))

    usd_path = os.path.join(os.path.dirname(__file__), "ovrtx_bug_repro.usda")
    if not os.path.exists(usd_path):
        logger.error("%s not found", usd_path)
        sys.exit(1)

    tmp_dir = tempfile.mkdtemp(prefix="ovrtx_bug_")
    rp_path = _write_render_products(tmp_dir)
    native_combined_path = _write_combined_scene(
        usd_path, rp_path, tmp_dir, "combined_native"
    )

    static_scene_path = os.path.join(tmp_dir, "static_visibility_scene.usda")
    with open(static_scene_path, "w", encoding="utf-8") as f:
        f.write(STATIC_VISIBILITY_SCENE)
    static_combined_path = _write_combined_scene(
        static_scene_path, rp_path, tmp_dir, "combined_write_attribute"
    )

    if args.mode in ("native", "both"):
        native_outputs = _render_native_visibility(ovrtx, native_combined_path, tmp_dir)
        if not args.no_validate:
            _assert_no_blank_or_duplicate_frames(native_outputs)

    if args.mode in ("write-attribute", "both"):
        write_outputs = _render_write_attribute_visibility(
            ovrtx, static_combined_path, tmp_dir
        )
        if not args.no_validate:
            _assert_no_blank_or_duplicate_frames(write_outputs)

    logger.info("All requested visibility probes rendered (bug not triggered)")


if __name__ == "__main__":
    main()
