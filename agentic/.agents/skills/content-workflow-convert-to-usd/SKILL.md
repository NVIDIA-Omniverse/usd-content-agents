---
name: content-workflow-convert-to-usd
description: Use before USD-based content workflows when a source asset must be routed to OpenUSD through the agentic convert-to-usd workflow, including existing USD passthrough, CAD/mesh conversion, URDF conversion, MuJoCo/MJCF conversion, blocked reports, and handoff artifacts.
---

# content-workflow-convert-to-usd

Use this skill before `content-workbench` when a source asset is not yet a USD
asset and must be converted before material, physics, articulation, geometry, or
runtime-validation workflows.

This skill owns source-to-USD routing and conversion reports. Workbench owns
scene loading, inspection, rendering, editing, and validation after a USD asset
exists.

## Active Routes

Use only these routes unless the user explicitly asks to expand converter
coverage:

- existing USD passthrough;
- `convert-to-usd` router semantics;
- `urdf-usd-converter` for `.urdf` sources;
- `mujoco-usd-converter` for `.mjcf` sources and `.xml` sources with a
  `<mujoco>` root;
- `usd-convert-cad` for CAD and mesh sources it supports, including STEP/IGES,
  STL, OBJ, FBX, GLTF/GLB, 3MF, JT, DGN, Parasolid, SolidWorks, CATIA,
  Inventor, Revit, IFC, DWG/DXF, and related CAD formats.

Do not route through `usd-convert-asset`, `usd-convert-gsplat`, hand-authored
USD, or substitute mesh converters when `usd-convert-cad` supports the source.

## Workflow

1. Resolve the source asset path from the user request or workflow manifest.
2. Resolve the requested output USD file path. If the user did not specify one,
   use the current working directory:

   ```text
   ./<source-stem>.usda
   ```

   The output format may be configured as `usd`, `usda`, `usdc`, or `usdz`.
   When no output path is provided, use `--output-format FORMAT` to choose the
   inferred suffix. When an explicit output path is provided, its suffix must
   match the requested format.

3. Run dependency preflight. This installs only the converter package required
   by the source format:

   ```bash
   content-workflow-convert-to-usd-preflight SOURCE_ASSET
   ```

   The equivalent module command is:

   ```bash
   python -m content_agent_workflows.convert_to_usd.preflight SOURCE_ASSET
   ```

   For check-only mode, pass `--no-install-missing`.

   Preflight routes installs as follows:

   ```bash
   uv pip install urdf-usd-converter      # .urdf
   uv pip install mujoco-usd-converter    # .mjcf, or .xml with <mujoco> root
   uv pip install git+https://github.com/NVIDIA-Omniverse/usd-convert-cad.git@4226fd49c06420adf193f821e2ddee805bb38eef --extra-index-url https://pypi.nvidia.com
   ```

   Do not install a converter for existing USD files. Do not treat arbitrary
   `.xml` as MuJoCo; inspect the file and require MuJoCo evidence before
   installing `mujoco-usd-converter`. Use `usd-convert-cad` for CAD/mesh
   extensions rather than inventing a conversion path.
   For CAD conversion, prefer an existing `usd-convert-cad` checkout when the
   environment provides one:

   ```bash
   export USD_CONVERT_CAD_ROOT=/path/to/usd-convert-cad
   python "$USD_CONVERT_CAD_ROOT/install.py"
   ```

   The workflow will call `$USD_CONVERT_CAD_ROOT/convert.py` when that variable
   points at a checkout; otherwise it expects the `usd-convert-cad` console
   script to be installed.

4. Run the file-oriented conversion script. It also performs the same missing
   dependency install by default:

   ```bash
   python -m content_agent_workflows.convert_to_usd.cli SOURCE_ASSET [OUTPUT_USD]
   ```

   To infer a non-default output suffix:

   ```bash
   python -m content_agent_workflows.convert_to_usd.cli SOURCE_ASSET --output-format usdc
   ```

   For dependency-free probe/convert attempts, pass `--no-install-missing`:

   ```bash
   python -m content_agent_workflows.convert_to_usd.cli SOURCE_ASSET [OUTPUT_USD] --no-install-missing
   ```

5. Inspect the JSON report printed by the script, or pass `--report` and
   `--markdown-report` when durable report files are needed.
6. If `status` is not `passed` or `errors` is non-empty, report the blocked condition
   and do not invent a USD replacement.
7. If conversion succeeds, hand `output_usd_path` to the next workflow.
8. Load the produced USD through Workbench only after the conversion report
   identifies a concrete USD path.

## Required Artifacts

For durable workflow runs, preserve:

- `request.json`
- `converter_probe.json`
- `conversion_report.json`
- `conversion_report.md`
- `validation_report.json`
- `manifest.json`
- generated USD files when conversion succeeds

## Input And Output Paths

Use the source asset path, output USD path, and run/output directory provided by
the user, workflow manifest, or calling CLI. Do not assume any repository-local
path convention in this skill.

Treat source assets as immutable inputs. Write generated USD files, reports, and
temporary artifacts only to caller-provided output or run directories.

## Boundaries

Conversion is a pre-Workbench workflow. Do not start Workbench just to decide
whether a source can be converted. Use Workbench after conversion for loading,
preview renders, inspection, edits, runtime validation, and downstream workflow
evidence.
