# Mini Drawer Articulation

Native Windows execution is unsupported in the 0.6 release. On a Windows host,
run the Bash launcher inside WSL2. PowerShell commands below are retained only
for development diagnostics and future qualification.

This example gives the checked-in mini cabinet to the sole long-running asset
coordinator. The default route uses deterministic scene evidence to author one
prismatic drawer joint; it does not configure or require a Joint proposal
provider.

An external proposal provider is an optional advisory leaf. Opt into one only
for a task that explicitly needs it, following the provider-selection steps in
the [Articulation workflow skill](../../../.agents/skills/content-workflow-articulation/SKILL.md#instructions).
The provider never replaces outer review, deterministic authoring, saved-stage
readback, or canonical OVRTX evidence.

## What changes

- **Before:** `/MiniCabinet/Frame` and `/MiniCabinet/Drawer` are separate
  unarticulated components with no physics joint.
- **After:** the authored result adds exactly one prismatic joint that moves
  the blue drawer along the Y axis relative to the fixed gray frame.
- **Preserved:** the source geometry, display colors, component hierarchy, and
  every non-target property remain unchanged. No masses or colliders are added.

## Prerequisites

Complete the Agentic setup from [`../../../README.md`](../../../README.md),
then configure the coding-agent runner and an OVRTX backend as described there.
No Joint service URL, NIM key, or Joint Agent YAML is needed for the default
example.

Descriptor-sealed Joint Rigger authoring requires Linux, a Linux container, or
WSL2. The native PowerShell launcher runs the canonical platform preflight and
exits before creating workflow state; it cannot author Joint Rigger artifacts.

## Run

From the repository root:

```bash
agentic/examples/articulation/drawer/run.sh
```

Development-only native PowerShell diagnostic:

```powershell
agentic/examples/articulation/drawer/run.ps1
```

The launchers intentionally default to Codex. From an authenticated Claude Code
environment, override that default explicitly; the CLI preflights the existing
Claude login before launching the child:

```bash
agentic/examples/articulation/drawer/run.sh \
  --runner claude \
  --claude-execution-mode cli
```

```powershell
agentic/examples/articulation/drawer/run.ps1 `
  --runner claude `
  --claude-execution-mode cli
```

Set `CONTENT_WORKFLOW_OUTPUT_DIR` before launching to use another run directory.
Relative values are resolved from the repository root. Resume an interrupted
non-terminal run with the same selected directory:

```bash
run_dir="${CONTENT_WORKFLOW_OUTPUT_DIR:-runs/articulation-mini-drawer}"
content-workflow-cli asset resume --run-dir "$run_dir"
```

From PowerShell:

```powershell
$runDir = if ($env:CONTENT_WORKFLOW_OUTPUT_DIR) {
    $env:CONTENT_WORKFLOW_OUTPUT_DIR
} else {
    "runs/articulation-mini-drawer"
}
content-workflow-cli asset resume --run-dir $runDir
```

The launchers freeze the exact prompt-specific leaf scope. For this
provider-free scenario, `execution_graph.json` must omit
`articulation.proposal-provider.v1` and every unrelated catalog leaf, select
only the Articulation and canonical-evidence leaves needed by the drawer task,
and explain each selection with a prompt-relevance rationale. Graph freeze
rejects over-selection. The coordinator writes the complete decision patch
itself; no nested Joint reasoning agent is launched.
If evidence is ambiguous, the workflow may fail closed or require explicit
human review instead of guessing.

## Expected outcome

Only a passing `graph_terminal_receipt.json` together with
`terminal_validation.json` is success. The terminal artifacts must bind an
authored USD result whose Articulation author receipt binds
`joint_rigger/rigged.usdz` with exactly one prismatic joint that moves
`/MiniCabinet/Drawer` along the Y axis relative to the fixed
`/MiniCabinet/Frame`. The source `drawer.usda` remains unchanged, and no masses
or colliders are added.

The principal durable artifacts are:

- `request.json` and `execution_graph.json`, which bind the sole coordinator
  and show that the optional proposal-provider leaf was omitted;
- `asset_run.json` and the Articulation leaf receipts;
- canonical current-run OVRTX output evidence and saved-stage readback;
- `graph_terminal_receipt.json` and `terminal_validation.json`.

The canonical OVRTX images are static post-authoring evidence: they prove that
the saved articulated asset loads and renders with the drawer and frame still
in the expected places. They do not by themselves prove motion. To qualify or
demonstrate the drawer's runtime movement, use the published rig as input to a
separate Physics workflow and inspect its trajectory metrics,
`recording.usda`, and OVRTX-rendered frame sequence. That derivative runtime
recording is visual motion evidence, not a checked-in input or an Articulation
terminal artifact.

Preserve failed or cancelled attempts as evidence. After correcting the cause,
resume only when the workflow reports that recovery is valid; otherwise start
a new output directory. The workflow never falls back to fixed-pipeline
execution.
