# Long-Running Asset Workflow Quickstart

The asset workflow owns one durable run from an admitted geometry source through
the final validated USDZ package. It starts with an existing CAD, mesh, USD,
URDF, MJCF, or immutable `geometry.source.v1` bundle and then runs Geometry,
Articulation, Material, Texture, Physics, Validation, and Finalization.

Every stage consumes the accepted, digest-bound output of its predecessor.
Rejected Geometry never falls through to the original source. Completed stages
are immutable and are not rerun on resume.

Text- or image-conditioned authoring is a separate provider step. Use the
authenticated Geometry Agent service with an explicitly selected provider, then
pass its exported artifact into this workflow. The asset runner does not install
or execute authoring systems. See
[Geometry Agent Quickstart](geometry_quickstart.md#external-authoring).

## Install

From the repository root on a Linux NVIDIA GPU host:

```bash
./scripts/setup_content_agent.sh
source .venv/bin/activate

# Codex CLI authentication
content-workflow-cli auth login
content-workflow-cli auth status

# Claude CLI authentication, when using --claude-execution-mode cli
claude auth status
```

Node.js 20 or newer is required for model-backed Codex or Claude child runners.
For Claude Agent SDK execution, export `ANTHROPIC_API_KEY`. For Claude CLI
execution, run `claude auth login` once; the workflow checks that login before
launch. Pass both `--runner claude` and `--claude-execution-mode sdk|cli`.
Provision the isolated OVRTX runtime by following
[Geometry Workflow Quickstart](geometry_quickstart.md#requirements). Accepted
Geometry visual evidence is rendered through OVRTX.

## Author Then Continue

Generate from text or images through the selected provider and request an
immediate Geometry run:

```bash
geometry-agent generate \
  --provider build123d-http \
  --prompt "A parameterized wall bracket with two M6 clearance holes" \
  --image path/to/reference.png \
  --parameter shelf_depth_mm=70 \
  --format step \
  --run --render-evidence \
  --output runs/generated-bracket.json
```

The authoring handoff is an immutable source specification. Geometry validates
its exported representations independently and carries semantic parts,
parameters, provider assertions, lineage, and rights into downstream evidence.
Provider-native source remains opaque and is never executed by public Geometry.

For semantic families, request each explicit parameter row from the same
provider revision and retain each returned source bundle. A downstream claim is
valid only when every requested row has its own digest-bound representation and
Geometry validation evidence.

## Process An Existing Asset

Provide any supported CAD, mesh, USD, URDF, or MJCF source to start at Geometry:

```bash
content-workflow-cli asset run \
  --source path/to/source.step \
  --prompt-file requirements.md \
  --compatibility-fixed-order \
  --joint-config apps/joint_agent/configs/byoa_joint_rigger.yaml \
  --materials-yaml apps/material_agent/data/materials/material_libs_default/materials.yaml \
  --geometry-repair-mode diagnose \
  --geometry-repair-profile rigid_pick_place \
  --output-dir agentic/runs/imported-part-001
```

`--usd` remains a compatibility alias for `--source`.

## Review And Resume

The articulation stage always pauses for an explicit human decision. A paused
run exits with code `3` and prints `Needs review: true`. Inspect the frozen
candidate evidence, create a JSON object containing exactly one `accept` or
`reject` decision for every candidate ID, and continue the same run:

```bash
content-workflow-cli asset review \
  --run-dir agentic/runs/wall-bracket-001 \
  --decisions-json articulation-decisions.json \
  --reviewer "$USER"
```

Resume an interrupted active run without replaying accepted stages:

```bash
content-workflow-cli asset resume \
  --run-dir agentic/runs/wall-bracket-001
```

A failed or cancelled current stage requires an explicit recovery reason:

```bash
content-workflow-cli asset resume \
  --run-dir agentic/runs/wall-bracket-001 \
  --recover "Backend restored; retry the current bounded attempt."
```

Use `--dry-run` on `asset run`, `asset review`, or `asset resume` to validate
and freeze the requested transition without launching a child model.

## Durable Evidence

The run directory contains immutable inputs, `request.json`, `asset_run.json`,
agent prompts and logs, numbered stage attempts, accepted handoffs, and
`terminal_validation.json`. A first source-free CAD attempt additionally
contains:

```text
stages/01-cad_modeling/
  cad_modeling_stage_result.json
  domain-run/
    compose_request.json
    compose_job.json
    selected_cad_program.json
    selected_cad_evaluation.json
    generated_asset.usd*
    cad_source_manifest.json
    program_pipeline_evidence.json
```

Refinements use `stages/01-cad_modeling/attempts/02`, `attempts/03`, and so
on. The same numbered-attempt rule applies to every later workflow stage.

Artifact names may vary by selected representation, but their typed bindings
and SHA-256 identities are authoritative. The terminal result is valid only
after every stage has accepted its own native evidence and Finalization has
produced the package and combined report.

## Exit Codes

| Exit | Meaning |
| --- | --- |
| `0` | The requested transition completed; for `asset run`, the whole run is terminal unless `Needs review` says otherwise. |
| `1` | The child stopped cleanly without a valid terminal workflow result. |
| `2` | Setup, contract, coordinator, or child execution failed. The durable state is retained. |
| `3` | Articulation requires human review. Use `asset review`. |
| `130` | Execution was interrupted. Use `asset resume`; add `--recover` if the current stage was cancelled. |

## Scope

The workflow composes ownership; it does not collapse it. CAD owns semantic
design intent and evaluated representations. Geometry owns source fidelity,
optimization/repair evidence, segmentation metadata, and the canonical handoff.
Articulation, Material, Texture, Physics, Validation, and Finalization own their
respective claims. A Geometry pass alone is not a claim of final SimReady,
runtime, material, joint, or serviceability readiness.
