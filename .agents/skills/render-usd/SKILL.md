---
name: render-usd
description: Render USD files using the wu CLI with a remote rendering service or the local OVRTX subprocess backend. Use when the user wants to render a USD scene, generate images from .usd/.usda/.usdc/.usdz files, render all cameras or frame ranges, produce remote depth or segmentation sensors, focus or isolate prims, or test a render endpoint.
version: "0.2.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - usd
  - rendering
  - cli
  - ovrtx
tools:
  - Shell
  - Filesystem
  - wu
compatibility: Requires the wu CLI and readable USD input files. The remote backend requires a configured RENDER_ENDPOINT plus any service authentication or asset-transfer credentials; the local ovrtx backend requires its isolated OVRTX runtime.
---

# Render USD

Render USD assets through `wu render-usd` using either a configured remote
render service or the local OVRTX subprocess backend.

## When to Use

- Use when the user wants rendered images from `.usd`, `.usda`, `.usdc`, or
  `.usdz` files.
- Use when the user asks for depth, instance segmentation, all-camera renders,
  frame ranges, focused prim renders, isolated prim renders, or camera JSON.
- Use `print-usd` first when the user needs to discover cameras or prim paths.
- Use `deploy-ovrtx-docker` first when no render endpoint is running.

## Limitations

- Canonical CLI backends are `remote` and `ovrtx`. Other canonical rendering
  backends, including `warp` and `mock`, are not supported by this CLI surface.
- The compatibility alias `local-ovrtx` is deprecated and will be removed in
  0.6.0; use `ovrtx`.
- Sensor outputs are supported only by `remote`; `ovrtx` currently renders
  color images only.
- Single-frame single-camera runs must use exactly one of `--output` or
  `--output-dir`.
- Multi-frame or all-camera runs require `--output-dir` and cannot use
  `--output`.
- Large scenes can take time to flatten, upload, render, and download.
- Missing lights or unsupported material shaders can produce dark output even
  when geometry renders correctly.

## Prerequisites

- Activate the repo Python environment and confirm `wu` is on `PATH`.
- For `--backend remote`, set `RENDER_ENDPOINT` to the render service URL and
  configure any authentication or asset-transfer credentials required by the
  deployment.
- For `--backend ovrtx`, ensure the isolated OVRTX runtime can be provisioned
  or pass `--ovrtx-venv-dir` to an existing environment.
- Ensure the USD input and referenced assets are readable.

## Instructions

1. Inspect the input path and choose single image, multi-frame, all-camera, or
   focused/isolation mode.
2. Use `wu print-usd <file> --show-types --max-depth 3` when camera or prim
   paths are unknown.
3. Select `remote` for a REST renderer or `ovrtx` for local color rendering.
4. Choose output flags according to the output rules.
5. Add `--focus` to auto-frame a prim, `--isolate` to hide everything except
   listed prims, or both for an object-only render.
6. Add sensor outputs only with `remote` and when the render service supports
   them.
7. Report the canonical backend, output files, camera JSON, and render warnings.

## Command Reference

```bash
wu render-usd <usd_path> [OPTIONS]
```

| Option | Description |
|---|---|
| `--output`, `-o` | Output path for a single frame and camera. |
| `--output-dir` | Directory for multi-camera, multi-frame, or directory-based single renders. |
| `--width`, `-w` | Image width. Default is `1920`. |
| `--height` | Image height. Defaults to width. |
| `--camera`, `-c` | Camera name or prim path. Default is `Camera`. |
| `--frames`, `-f` | Frame selector such as `0`, `0:10`, or comma-separated values. |
| `--backend`, `-b` | Canonical backend: `remote` (default) or `ovrtx`. The deprecated `local-ovrtx` alias warns and will be removed in 0.6.0. |
| `--sensors` | Remote-only comma-separated sensors such as `linear_depth`, `depth`, or `instance_id_segmentation`. |
| `--all-cameras` | Render every camera. Requires `--output-dir`. |
| `--save-camera-json` | Save camera parameters next to rendered images. |
| `--focus` | Prim path to auto-frame with the camera. |
| `--isolate` | Comma-separated prim paths to render while hiding other geometry. |
| `--hide` | Comma-separated prim paths or subtrees to hide before rendering. |
| `--direction` | Camera direction such as `+x+y+z` or `+x-0.5y+z`. |
| `--margin` | Camera distance margin multiplier. |
| `--focal-length` | Camera focal length in millimeters. |
| `--aperture` | Horizontal aperture in millimeters. |
| `--cam-x`, `--cam-y`, `--cam-z` | Override camera position. |
| `--target-x`, `--target-y`, `--target-z` | Override look-at target. |
| `--near-clip` | Override camera near clipping plane distance. |
| `--far-clip` | Override camera far clipping plane distance. |
| `--dome-light` | Replace scene lights with a dome light intensity. |
| `--distant-light` | Replace scene lights with a distant light intensity. |
| `--ovrtx-log-level` | Local OVRTX log level. |
| `--ovrtx-venv-dir` | Override the isolated local OVRTX virtualenv directory. |
| `--ovrtx-num-sensor-updates` | Local OVRTX progressive render steps per frame. |
| `--ovrtx-render-mode` | Local OVRTX render mode: `rt1`, `rt2`, or `pt`. |
| `--verbose`, `-v` | Enable debug logging. |

## Common Workflows

```bash
# Single image.
wu render-usd scene.usd --output render.png

# Square thumbnail.
wu render-usd scene.usd --output render.png --width 512 --height 512

# All cameras.
wu render-usd scene.usd --all-cameras --output-dir renders/

# Frame range.
wu render-usd scene.usd --frames 0:10 --output-dir frames/

# Depth sensor.
wu render-usd scene.usd --output render.png --sensors linear_depth

# Local OVRTX color render.
wu render-usd scene.usd --backend ovrtx --output local.png

# Focus and isolate one object.
wu render-usd scene.usd --focus /World/Chair --isolate /World/Chair \
  --output chair.png

# Add simple lighting when a scene is dark.
wu render-usd scene.usd --output lit.png --dome-light 1500
```

## Output Format

Report:

- Command executed and canonical render backend used, without printing
  credentials. Include the render endpoint only for `remote`.
- Input USD path, camera selection, frame selection, and output path or
  directory.
- Any focus, isolate, light, sensor, or camera override options.
- Rendered image files, sensor files, and camera JSON files when present.
- Any failure cause, such as invalid output flag combination, missing USD file,
  missing prim/camera, endpoint error, timeout, or dark render caveat.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Missing `RENDER_ENDPOINT` | Remote backend has no service URL. | Start or configure a render service, then export `RENDER_ENDPOINT`. |
| Deprecated `local-ovrtx` warning | A compatibility alias was used. | Replace it with `--backend ovrtx`; the alias is removed in 0.6.0. |
| Local OVRTX setup error | The isolated OVRTX runtime could not be provisioned or started. | Verify the OVRTX installation or pass the correct `--ovrtx-venv-dir`. |
| OVRTX sensor error | `--sensors` was combined with local `ovrtx`. | Use `remote` for depth or segmentation sensors, or omit `--sensors`. |
| Multi-output flag error | Multi-frame or all-camera run used `--output`. | Use `--output-dir` for multi-output runs. |
| Single-output flag error | Single-frame, single-camera run used both `--output` and `--output-dir`, or neither. | Use exactly one output flag for single-output runs. |
| Camera path error | The named camera is absent or misspelled. | Inspect with `wu print-usd scene.usd --show-types` or omit `--camera` for auto-created view. |
| Dark or black image | Scene lacks lights or uses unsupported shaders. | Try `--dome-light 1500` or `--distant-light 800`; inspect the USD materials separately. |
| Timeout or 504 | Large scene, cold render service, or slow asset transfer. | Reduce resolution, render a focused prim, or retry after endpoint warm-up. |
