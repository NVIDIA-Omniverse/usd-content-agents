# Basic Agentic Validation

Native Windows execution is unsupported in the 0.6 release. On a Windows host,
run the Bash commands inside WSL2. PowerShell commands below are retained only
for development diagnostics and future qualification.

This example is owned end to end by one long-running coding agent. The launcher
starts one decision-only Validation planning child to select and run
evidence-backed checks over the checked-in smoke bracket. The same outer agent
then inspects the exact results, authors the independent assessment and review,
and seals the terminal receipt. Validation is non-mutating and uses an
explicitly selected local or remote OVRTX backend for current render evidence.

## What changes

- **Before:** the checked-in smoke bracket has no current-run release-readiness
  assessment.
- **After:** the source USD is byte-for-byte unchanged and the run publishes an
  accepted assessment of its static integrity, current OVRTX rendering, and
  package integrity.
- **Preserved:** geometry, material, hierarchy, dependencies, and every source
  byte remain unchanged. Validation does not publish another USD.

## Prerequisites

Complete the Agentic setup from [`../../../README.md`](../../../README.md),
then configure Codex or Claude and an OVRTX backend described there. The
launchers default to the local `ovrtx` backend. WSL2 cannot use local OVRTX;
select the already configured remote backend there as shown below.
The [smoke bracket](../../geometry/quickstart/smoke_bracket.usda) is a small,
self-contained public asset already checked into the repository, so it needs no
download or asset credentials. No Validation service or advisory VLM provider
is required for this example.

## Run

From the repository root, ask the long-running coding agent to run:

```bash
agentic/examples/validation/basic/run.sh
```

From PowerShell:

```powershell
agentic/examples/validation/basic/run.ps1
```

To use a configured remote OVRTX endpoint, including on WSL2, select it
explicitly before running the same launcher. Keep endpoint credentials in the
environment as described by the root Agentic setup guide.

```bash
export CONTENT_WORKFLOW_RENDER_BACKEND=remote
```

```powershell
$env:CONTENT_WORKFLOW_RENDER_BACKEND = "remote"
```

The launchers intentionally default to Codex. From an authenticated Claude Code
environment, select the CLI execution path and an explicit Claude model:

```bash
agentic/examples/validation/basic/run.sh \
  --runner claude \
  --claude-execution-mode cli \
  --model sonnet
```

```powershell
agentic/examples/validation/basic/run.ps1 `
  --runner claude `
  --claude-execution-mode cli `
  --model sonnet
```

The planning child can only propose a preparation-bound check plan. Trusted
outer code validates that plan and executes the exact selected adapters. A
successful launcher exit is therefore not terminal Validation success. The
same outer coding agent must inspect the current-run evidence, write
`outer-assessment.json` and `outer-review.json` inside the selected run
directory, and continue the independent decision chain:

```bash
run_dir="${CONTENT_WORKFLOW_OUTPUT_DIR:-runs/validation-smoke-bracket}"
content-workflow-cli validate collect-evidence \
  --output-dir "$run_dir"
content-workflow-cli validate assess \
  --output-dir "$run_dir" \
  --assessment "$run_dir/outer-assessment.json"
content-workflow-cli validate review-assessment \
  --output-dir "$run_dir" \
  --review "$run_dir/outer-review.json"
```

From PowerShell:

```powershell
$runDir = if ($env:CONTENT_WORKFLOW_OUTPUT_DIR) {
    $env:CONTENT_WORKFLOW_OUTPUT_DIR
} else {
    "runs/validation-smoke-bracket"
}
content-workflow-cli validate collect-evidence `
  --output-dir $runDir
content-workflow-cli validate assess `
  --output-dir $runDir `
  --assessment (Join-Path $runDir "outer-assessment.json")
content-workflow-cli validate review-assessment `
  --output-dir $runDir `
  --review (Join-Path $runDir "outer-review.json")
```

The assessment and review files must be authored from the exact current-run
evidence; do not copy example verdicts or reuse an earlier run. Default agentic
coordinator runs are not resumable. If interrupted, preserve the run and start
a fresh value of `CONTENT_WORKFLOW_OUTPUT_DIR`.
`content-workflow-cli validate resume --output-dir "$run_dir"` (or
`--output-dir $runDir` in PowerShell) records
`safe_restart_required` instead of launching another coordinator.

## Expected outcome

For the intended passing case, the planning child selects preparation-bound
static USD and current-render checks, and trusted outer code records their exact
results in `validation_coordinator_accepted_plan.json`,
`validation_operation_index.json`, and
`validation_coordinator_execution_receipt.json`. The execution receipt must say
`source_mutated: false`; successful execution alone is not semantic acceptance.

The independent decision chain then produces:

- `standalone_validation_evidence.json`, which binds the operation results and
  current selected OVRTX render evidence;
- `canonical_validation_assessment.json`, only when the evidence supports a
  passing assessment;
- `validation_terminal_receipt.json`, the authoritative result readback.

A successful terminal receipt has `receipt_status: "completed"`,
`terminal_disposition: "pass"`, `review_disposition: "accept"`,
`publication_kind: "validation_assessment"`, and `source_mutated: false`. The
result is an evidence-backed validation assessment; this non-mutating workflow
does not publish a modified USD. A failed required check, non-passing
assessment, or non-accepting review is a useful preserved result, but not a
successful validation outcome. OVRTX renders are generated at runtime and are
not checked into this example. The rendered bracket should look like the input:
the visible result is evidence of successful load and rendering, while the
meaningful output is the accepted assessment and its exact receipt chain. An
optional advisory critique may remain `not_requested` or `not_evaluated`; it is
never a required provider or a substitute for the outer agent's direct image
inspection.
