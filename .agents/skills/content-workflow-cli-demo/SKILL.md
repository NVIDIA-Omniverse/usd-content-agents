---
name: content-workflow-cli-demo
description: Prepare polished representative demo videos for content-workflow-cli workflows. Use when the user wants an Apple/Google-style software demo, recording plan, terminal demo script, storyboard, or artifact-backed video showing the CLI, observable agent trace, Content Authoring Tool API/material authoring calls, render review, iteration, and final evidence.
version: "0.1.1"
author: NVIDIA Content Agents
tags:
  - content-authoring
  - demo
  - video
  - content-workflow-cli
tools:
  - Shell
  - Python
  - content-workflow-cli
  - Content Authoring Tool
compatibility: Requires Codex or another agent runner with repo-local .agents skills, Python 3.12, content-workflow-cli, and a reachable Content Authoring Tool render API.
---

# Agentic Content Authoring Demo

Use this skill to stage a representative product demo where
`content-workflow-cli` is the terminal entrypoint and the Content
Authoring Tool is the visible authoring surface. The final deliverable should feel like a
curated software demo, not a raw recording of every slow render or log line.
Use real workflow artifacts as evidence, then compose the clearest moments into
a short video.

## When to Use

- Use when the user wants a representative software-demo video for an agentic
  content authoring workflow.
- Use when the demo should show the terminal command, observable agent trace,
  Content Authoring Tool API/material authoring calls, render review,
  iteration, and final artifacts.
- Use when the deliverable can be an artifact-backed replay from real run
  outputs instead of a literal desktop recording.

## Guardrails

- Show observable trace, commands, API calls, artifacts, and rendered evidence.
  Do not describe hidden chain-of-thought as "reasoning"; call it an agent
  trace, decision log, or workflow trace.
- Keep secrets, account state, API keys, and auth URLs out of the captured
  terminal and generated files.
- Use real `content-workflow-cli` output as the source of truth, but edit
  for clarity: skip waits, collapse repetitive logs, and highlight decisions.
- Prefer a composed artifact-backed video when no live desktop recorder is
  available. Label it as a generated replay from real run artifacts.
- Keep terminal text readable on video: large font, short command lines, and
  concise trace panes. The demo should be understandable without pausing.
- For material assignment demos, prove the start state is clean: clear or block
  source material bindings, `primvars:displayColor`,
  `primvars:displayColor:indices`, and `primvars:displayOpacity` before showing
  baseline inspection renders. Do not use colored source diagnostic renders as
  the apparent "before" state.
- Match the material-agent-service visual system when composing UI: dark
  surface tokens (`#111318`, `#1c2027`, `#0b0d11`), NVIDIA green
  (`#76b900` / `#8bd30f`) for active accents, subdued gray borders, compact
  typography, and 6-8px rounded rectangles for panels, labels, buttons, and
  progress tracks. Avoid elliptical labels or decorative oval treatments.

## Limitations

- Generated videos are composed product-demo replays unless a live recorder is
  explicitly used.
- Render fidelity depends on the available Content Authoring Tool render API,
  lighting configuration, and completed workflow artifacts.
- The rerender helper posts local scene paths to the Content Authoring Tool, so
  that service must run on the same host/filesystem or otherwise be able to read
  those paths.
- Large USD assets or final-quality renders can take several minutes per
  rerender pass.
- USD instances are not un-instanced by the rerender helper; if instance prims
  are detected, the metadata warns that prototype-internal materials or display
  colors may need explicit handling.
- The provided defaults are tuned for the staged Ladder material authoring demo;
  other assets may need different camera directions and scene timing.

## Prerequisites

- Python 3.12 with the repository development environment available.
- `content-workflow-cli` installed and authenticated for the workflow
  runner being demonstrated.
- A reachable Content Authoring Tool service, usually
  `http://127.0.0.1:8088`, that can read local scene paths used by the rerender
  helper.
- A USD asset, a reference image, and a material library YAML.
- Completed or runnable workflow artifacts when composing the final video.

## Instructions

1. Pick the asset, reference image, prompt, runner, Content Authoring Tool
   endpoint, and run directory. The defaults use public assets that remain
   available after the staging copy:

```bash
--usd apps/material_agent/data/examples/ladder/sources/usd/ladder.usd
--reference-image apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg
--materials-yaml apps/material_agent/data/materials/material_libs_default/materials.yaml
--output-dir .local-runs/content-workflow-cli/ladder-product-demo
--output-usd .local-runs/content-workflow-cli/ladder-product-demo/ladder_material_assignments.usda
```

2. Generate the product-demo plan and run command:

```bash
python3 .agents/skills/content-workflow-cli-demo/scripts/build_recording_demo.py \
  --write-dir .local-runs/content-workflow-cli-demo/ladder-product-demo
```

`--write-dir` stores only the generated `demo_plan.md` and `run_demo.sh`
launcher. The generated workflow command writes its run artifacts to
`--output-dir` and writes the durable material result to `--output-usd` under
that same workflow run. The launcher creates the workflow run directory before
invoking the CLI because `--output-usd` requires its parent to exist. Do not
treat the plan/launcher directory as the workflow run directory.

Use `--dry-run` only to rehearse the command; the rerender and video steps need
durable apply output from a real workflow run.

The generated command defaults to `--no-optimize` so it runs from a staged
checkout without the optional Scene Optimizer package. Pass `--optimize` to the
plan builder only when a local or remote optimizer backend is configured.

3. Preflight before running the workflow:

```bash
uv pip install -e agentic/packages/content_workflow_cli
npm ci --prefix agentic/packages/content_workflow_cli
content-workflow-cli auth status
curl -fsS http://127.0.0.1:8088/healthz || true
```

4. Run the generated `run_demo.sh`. Its command includes both the canonical
   workflow `--output-dir` and the exact durable `--output-usd`. The workflow
   should produce
   `assignments.json`, final renders, `visual_quality_assessment.json`, durable
   apply output, and `trace/operation_trace.md`. Keep downstream rerender and
   video steps pointed at those exact paths; do not rename or copy the durable
   output between steps.

5. Rerender the demo visual assets through the Content Authoring Tool. This
   creates an unmaterialized USD by blocking source material bindings and
   authored display color/opacity primvars, renders the opening from a neutral
   clay baseline overlay, then uses the Content Authoring Tool render API with
   actual lighting:

```bash
python3 .agents/skills/content-workflow-cli-demo/scripts/rerender_demo_assets.py \
  --source-usd apps/material_agent/data/examples/ladder/sources/usd/ladder.usd \
  --assigned-usd .local-runs/content-workflow-cli/ladder-product-demo/ladder_material_assignments.usda \
  --hdri-light 600
```

6. Render a polished representative demo video from the completed run and the
   rerendered visual assets:

```bash
python3 .agents/skills/content-workflow-cli-demo/scripts/render_demo_video.py \
  --run-dir .local-runs/content-workflow-cli/ladder-product-demo \
  --reference-image apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg \
  --render-assets-dir .local-runs/content-workflow-cli-demo/ladder-product-demo-rerender/assets \
  --output .local-runs/content-workflow-cli-demo/ladder-product-demo-video/content_workflow_cli_ladder_demo.mp4
```

For non-Ladder demos, pass `--usd`, `--workflow-command`, `--prompt`,
`--target-description`, and `--assigned-usd-label` so the composed terminal and
reference text matches the actual workflow inputs.

7. For a live capture, record a two-window layout:
   - Left: terminal running the generated command.
   - Right: Content Authoring Tool viewport/API surface.
   - Optional: a small reference image window for the Ladder image.

8. Capture or compose these beats:
   - start with the reference image so the target look is visible immediately;
   - then show the unmaterialized gray baseline asset to make the material
     improvement obvious;
   - user launches the CLI with the Ladder asset, reference image, material library, and
     short extra instruction prompt;
   - CLI creates the run directory and opens/uses an authoring session;
   - terminal trace shows material inspection, preview material overrides,
     render requests, issue detection, and accepted apply output;
   - Content Authoring Tool window shows the asset, preview material changes,
     final render, and one correction pass if an issue is visible;
   - final frame shows `final_renders/`, `assignments.json`,
     `visual_quality_assessment.json`, and `trace/operation_trace.md`;
   - close with run stats such as token counts, API/query counts, renders,
     picks, material coverage, and environment labels.

9. After the run, rebuild or inspect trace artifacts if the demo needs a
   cleaner summary:

```bash
content-workflow-cli trace build \
  --run-dir .local-runs/content-workflow-cli/<run-id>
```

## Output Format

When reporting back, include:

- the generated demo plan path;
- the exact CLI command used for the workflow run;
- whether the video is a live capture or artifact-backed representative replay;
- the generated video path;
- final run directory and key render/trace artifacts;
- any parts that still need manual capture in the Content Authoring Tool window.

## Troubleshooting

- If `content-workflow-cli auth status` fails, refresh the runner
  authentication before recording.
- If the Content Authoring Tool health check fails, start or reconnect the
  service and retry the rerender step.
- If video rendering cannot find PNG assets, run `rerender_demo_assets.py`
  against the completed workflow output directory first.
- If a generated replay looks inconsistent with the final asset, rerender
  through the Content Authoring Tool instead of compositing over prior images.
