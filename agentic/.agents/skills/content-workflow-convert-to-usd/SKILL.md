---
name: content-workflow-convert-to-usd
description: Use before USD-based content workflows when a source asset must be routed to OpenUSD through the existing content-workflow-cli conversion workflow, including dependency preflight, existing USD passthrough, CAD/mesh, URDF, MuJoCo/MJCF, durable reports, blocked results, and downstream USD handoff.
metadata:
  author: NVIDIA Omniverse
---

# content-workflow-convert-to-usd

Use this skill when a source asset is not yet USD and must be converted before
material, physics, articulation, geometry, or runtime-validation work.

This skill and `content_agent_workflows.convert_to_usd` own source-format
routing, converter dependency preflight, converter execution, validation,
durable reports, and the output-USD handoff. Conversion is not a usd-cli scene
operation and is not a usd-cli parity requirement.

## Active Routes

Use only the existing workflow routes unless the user explicitly expands
converter coverage:

- existing USD-family passthrough;
- `urdf-usd-converter` for `.urdf`;
- `mujoco-usd-converter` for `.mjcf` and `.xml` with a verified `<mujoco>` root;
- `usd-convert-cad` for its supported CAD and mesh formats, including
  STEP/IGES, STL, OBJ, FBX, GLTF/GLB, 3MF, JT, DGN, Parasolid, SolidWorks,
  CATIA, Inventor, Revit, IFC, and DWG/DXF.

Do not route through `usd-convert-asset`, `usd-convert-gsplat`, hand-authored
replacement USD, or a substitute mesh converter when `usd-convert-cad`
supports the source. Do not treat arbitrary XML as MuJoCo.

## Workflow

1. Resolve the source, requested output format/path, and durable run directory.
   Treat the source as immutable.
2. Run the existing dependency preflight. It selects and checks only the
   converter implied by the source:

   ```bash
   content-workflow-cli preflight convert-to-usd SOURCE_ASSET \
     --report RUN/converter_preflight.json
   ```

   Use `--no-install-missing` only when installation is prohibited. Do not ask
   the agent or user to recreate the workflow's dependency routing manually.
3. Run the durable conversion workflow:

   ```bash
   content-workflow-cli convert-to-usd SOURCE_ASSET OUTPUT.usd \
     --output-dir RUN \
     --report RUN/conversion_report.json \
     --markdown-report RUN/conversion_report.md
   ```

   `--converter-timeout` accepts a positive finite number of seconds and
   defaults to 120. The exact value is frozen in `request.json` and the workflow
   run manifest. A resumed run must use the recorded value; choose a new run
   directory when intentionally changing the timeout after a failed attempt.
   For a supported large input that needs a larger bound, select it explicitly:

   ```bash
   content-workflow-cli convert-to-usd SOURCE_ASSET OUTPUT.usd \
     --output-dir RUN \
     --converter-timeout 600
   ```

   When the output path is omitted, use `--output-format usd|usda|usdc|usdz`
   to select the inferred suffix.
4. Inspect the normalized result and generated artifacts. A failed or blocked
   conversion must retain its report and must not invent a replacement USD.
5. On success, take `output_usd_path` from the workflow result and give that
   exact file to the next workflow.
6. Only after conversion succeeds may the next workflow open the output with
   usd-cli for low-level scene operations.

Use `content-workflow-cli preflight convert-to-usd --help` and
`content-workflow-cli convert-to-usd --help` for the exact option surface.

## Required Artifacts

For a durable run preserve:

- `request.json`;
- `converter_probe.json`;
- `conversion_report.json`;
- `conversion_report.md`;
- `validation_report.json`;
- `manifest.json`;
- the durable `workflow_run_manifest.json` record at the workflow result's
  `workflow_run_manifest_path`, including its sealed checkpoint history;
- the generated USD-family output on success;
- the concrete blocked/failure reason on failure.

These artifacts are produced by the workflow. Do not replace them with
manually assembled notes or raw converter stdout.

## Input and Output Paths

Use caller-provided paths when present. Otherwise the CLI resolves the default
output beside the current working directory and places durable workflow
artifacts under the explicit `--output-dir`.

Never convert in place. Keep temporary files and reports under the workflow
run/output directory.

## Boundaries

- Converter selection, installation policy, execution, validation, and
  reporting remain in `content_agent_workflows.convert_to_usd`.
- Conversion completes before any stateful scene session begins.
- usd-cli may inspect, render, edit, or validate the
  resulting USD only after the conversion workflow returns a concrete,
  successful output path.
