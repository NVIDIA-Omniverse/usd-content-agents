# Basic Texture Companion Workflow

Native Windows execution is unsupported in the 0.6 release. On a Windows host,
run the Bash commands inside WSL2. PowerShell commands below are retained only
for development diagnostics and future qualification.

This example starts one skill-routed Texture workflow on the shipped UV-ready
ladder. It selects only the top plastic grip/tray assembly and asks the outer
coding-agent companion for one clearly visible texture-generation attempt.

## What changes

The target prim is
`/RootNode/Geometry/M_AluminumStepLadder_B01_Plastic2`:

- **Before:** the top plastic grip/tray assembly is dark blue.
- **After:** only that assembly is smooth high-visibility safety orange rubber
  with an even single-color matte finish.
- **Preserved:** every silver rail and step plus all black hardware remains
  unchanged.

This deliberately high-contrast target makes the Texture result visually
distinct from the Material example and easy to verify in matched OVRTX views.
The launchers pass the same appearance text to the overall request and selected
texture unit, binding the agent-authored plan to the exact provider-neutral
preparation record. The workflow's non-target preservation gate protects the
rest of the ladder.

## Prerequisites

Complete the Agentic setup from [`../../../README.md`](../../../README.md),
configure the coding-agent runner and OVRTX backend described there, and run
the example from a coding-agent session with image generation enabled. No
Texture service or generation-provider configuration is required for a Codex
session that can satisfy the companion handoff itself.

`--runner claude` controls the Texture planning and review children; it does
not add image generation to the outer Claude environment. If that environment
has no compatible image-generation tool, start a fresh run with both a
concrete Texture backend and service URL. A backend or URL by itself is not a
complete provider selection.

## Run

From the repository root, ask the coding agent to run:

```bash
agentic/examples/texture/basic/run.sh
```

From PowerShell:

```powershell
agentic/examples/texture/basic/run.ps1
```

The launchers intentionally default to Codex. To use the authenticated Claude
CLI for planning and review while routing generation through a configured
Texture service:

```bash
agentic/examples/texture/basic/run.sh \
  --runner claude \
  --claude-execution-mode cli \
  --texture-backend <name> \
  --texture-agent-url <url>
```

Without both service arguments, a Claude run may still freeze a valid plan and
pause at the companion handoff below, but it cannot claim end-to-end success
until an outer image generator records the bound result.

The launcher normally pauses with exit code `3` after writing
`texture_companion_generation_handoff.json`. That pause is part of the durable
workflow, not a failure. The outer coding agent must read the exact handoff,
perform only its requested image-generation attempt, and write the bound result
manifest at the path named by the handoff. It then resumes the same selected
run directory.

From Bash:

```bash
run_dir="${CONTENT_WORKFLOW_OUTPUT_DIR:-runs/texture-ladder}"
content-workflow-cli texture resume --run-dir "$run_dir"
```

From PowerShell:

```powershell
$runDir = if ($env:CONTENT_WORKFLOW_OUTPUT_DIR) {
    $env:CONTENT_WORKFLOW_OUTPUT_DIR
} else {
    "runs/texture-ladder"
}
content-workflow-cli texture resume --run-dir $runDir
```

Relative `CONTENT_WORKFLOW_OUTPUT_DIR` values are resolved from the repository
root by both launchers.

## Expected outcome

The first expected result is `awaiting_companion_generation`, accompanied by
exit code `3`. It is a resumable handoff, not terminal success. After the outer
coding agent writes the exact requested PNG and bound result manifest, resume
must finish with both of these conditions:

- `workflow_result.json` has `status: "published"`;
- `texture_terminal_receipt.json` has `disposition: "published"`.

The published asset is `published/textured_asset.usdz`. In its matched source
and candidate views, the top plastic grip/tray assembly must change from dark
blue to the smooth safety orange finish described above. Every non-target
scoped unit must retain its original material state. The publication is
self-contained and is bound to
deterministic saved-stage readback, current-run OVRTX images, and an accepting
review through:

- `readback/texture_saved_stage_readback.json`;
- `evidence/texture_agentic_evidence.json`;
- `texture_review.json`;
- `published/publication_verification.json` and
  `published/texture_publication.json`.

A `blocked`, `rejected`, or `failed` terminal disposition is a preserved result,
but it is not a successful example result and must not publish the textured
asset. Generated textures and OVRTX renders are runtime results and are not
checked into this example.

Set `CONTENT_WORKFLOW_OUTPUT_DIR` to a new path before starting another run. Do
not reuse or edit a completed run directory.

This example uses the coding-agent companion path. It does not configure a
Texture Agent service, generation provider, or fixed-pipeline fallback.
