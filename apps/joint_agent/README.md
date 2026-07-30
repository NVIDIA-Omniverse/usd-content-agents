# Joint Agent

> **Status: Research Preview (0.5).** Joint Agent can infer articulation
> candidates and publish a self-contained USDZ whose authored joint graph
> matches accepted structured input. Model inference can still be wrong, and
> successful publication does not prove simulation readiness or dynamic
> behavior.

Articulation analysis and topology authoring for USD-family assets.

## Overview

Joint Agent renders USD files through a remote renderer, classifies articulated
components with a public NIM model, infers Stage 2 joint candidates, and can
author accepted topology with the repo-owned `owned_core` adapter.

Set the public hosted-model credential and renderer endpoint before running:

```bash
export NVIDIA_API_KEY="YOUR_NVIDIA_API_KEY"
export RENDER_ENDPOINT="http://renderer.example:8000"
```

The public template defaults to `nim`, `google/gemma-4-31b-it`, and remote
rendering through `RENDER_ENDPOINT`.

The prediction VLM requests up to 24,576 output tokens by default. Set
`JA_VLM_MAX_TOKENS` to override that default for CLI/config-driven runs. Known
lower OpenAI-family provider caps are applied at the final request boundary, so
the override can reduce a request but cannot force a request above its model's
supported output limit.

For supported structured Stage 2 inputs, successful publication guarantees
that the package opens and readback matches the accepted joint graph. This is a
package-integrity guarantee, not a guarantee that model-inferred joints are
semantically correct or physically complete.

The frozen 17-asset release candidate passes the pinned static Gate 3A and
Gate 3B profiles on the exact published package bytes. Treat this as bounded
release evidence, not as a guarantee for arbitrary assets or dynamic behavior.

## Usage

```bash
# Copy the public bring-your-own-asset template, then edit input.usd_path.
cp apps/joint_agent/configs/byoa_joint_rigger.yaml my_joint_asset.yaml

# Verify the plan before running the classification and candidate pipeline.
joint-agent run my_joint_asset.yaml --dry-run
joint-agent run my_joint_asset.yaml

# Validate Stage 2 candidates against an authored rigged USD reference
joint-agent validate-rigged-reference \
  path/to/rigged_reference.usdz \
  path/to/articulation_candidates.json
```

The public template enables `infer_articulation_candidates` and configures the
topology-only `owned_core` adapter, but keeps `apply_joint_rigger` disabled.
Review the generated Stage 2 candidate document before opting into package
authoring. Then set `steps.apply_joint_rigger.enabled: true` and resume:

```bash
joint-agent run my_joint_asset.yaml --resume
```

The owned-core output is `.joint-agent-byoa/joint_rigger/rigged.usdz`.

The rigged-reference validation command expects a USD/USDZ with authored
`PhysicsXxxJoint` prims and an existing Stage 2 `articulation_candidates.json`
artifact. It writes a reference manifest, validation JSON, and HTML report that
compare candidate recall, joint type, body0/body1 paths, world-frame axis, and
authored limits. Unlimited joints compare axes up to sign, while limited joints
preserve the candidate's signed axis direction so inverted limit ranges surface
as validation failures.

### Exact Prediction Coverage

The prediction step requires exactly one successful row for every unique
dataset prim. By default, an isolated failed or omitted row gets up to three
bounded single-worker completion retries (`completion_retries: 3`). Persistent
gaps, null success payloads, duplicate or unexpected IDs, and invalid resume
rows stop the pipeline before consistency, Stage 2, or rigging. Resume normalizes an
unterminated valid final row, or discards a crash-truncated final append and
reprocesses that ID. A successful run atomically seals `predictions.jsonl` in
dataset order; path media are serialized as strings and non-persistable
in-memory images are omitted from the checkpoint row.

### Asset-Wide Topology Reconciliation

The public template enables source-image VLM topology reconciliation. It is
triggered when Stage 1 contains an unknown role or strict Stage 2 finds a
structural topology conflict other than an isolated compound-edge conflict. A
triggered reconciliation normally makes one additional asset-wide VLM call. An
unparseable or validation-invalid response gets one corrective compact-JSON
retry with the same evidence and a bounded larger output budget, so plan for at
most two calls and the corresponding model latency and token/image cost.

Only a high-confidence, complete fixed/moving-link partition that passes strict
Stage 2 reinference is accepted. The partition may contain at most one fixed
link; multi-member fixed links must be direct siblings beneath one non-root
parent, with no fixed-member namespace overlapping another link member by
ancestry in either direction, so the owned-core projection can consume them.
For a flat multi-member moving link, the member with the shortest final path
component is the deterministic anchor/body1 representative; exact path order
breaks equal-length ties. This is an identity-only choice after membership is
validated, not semantic path-name inference. A fully resolved raw edge to
another member is retained exactly as superseded evidence, while an unknown-axis
partial edge remains an authoritative endpoint constraint.
Missing images,
incomplete or ambiguous model output after that retry, or any remaining
structural conflict fails closed: the original state is preserved, unresolved
candidates stay review-required, and no partial topology correction is
accepted.
An accepted decision is recorded in
`.joint-agent-byoa/articulation_candidates/articulation_candidate_adjudications.json`
with schema `joint-agent-articulation-adjudication-artifact-v1`. On a later run,
disabling adjudication or setting `reconcile_topology: false` validates that
receipt and restores the exact pre-reconciliation prediction payloads. Disabling
adjudication removes the stale artifact; keeping adjudication enabled writes the
current run's legacy v1 envelope. An incomplete or modified receipt blocks
restoration.

## Owned-Core Publication

`apply_joint_rigger` remains disabled by default. When enabled, the public
`owned_core` adapter consumes accepted `joint-agent-stage2-v0` candidates. The
service path also supplies Stage 1 prediction JSONL: owned-core projects both
artifacts into the first-class articulation contract and authors its exact
versioned request. Aggregate or multi-root contracts select V2 and author
aggregate rigid-link membership plus articulation roots; one-root existing-link
contracts retain V1. Direct or CLI/YAML candidate-only calls may omit predictions
and retain the transitional Stage 2-only behavior. During a normal pipeline run,
consistent or raw predictions are auto-wired with optimized-namespace Stage 2
candidates and enter the first-class V1/V2 contract path. Restored prediction
outputs are never auto-wired into that pairing.
Without an explicit prediction path or consistency/raw output, owned-core
retains candidate-only behavior; an explicit optimized-namespace prediction
path is preserved. Neither path falls back to an external rigger.

Enabled CLI/YAML configurations must set `adapter` explicitly. The REST service
uses a separate request contract where omitting the adapter selects
`owned_core`.

```yaml
steps:
  apply_joint_rigger:
    enabled: true
    adapter: owned_core
    # Optional: omit for transitional candidate-only authoring.
    predictions_path: path/to/predictions.jsonl
    articulation_candidates_path: path/to/articulation_candidates.json
    output_usd_path: path/to/rigged.usdz
    diagnostics_path: path/to/joint_rigger_diagnostics.json
    validation_path: path/to/joint_rigger_validation.json
    apply_masses: false
    apply_collision: false
```

Both contract-derived paths author accepted joint topology and source-backed
limits. For aggregate or multi-root contracts, V2 additionally authors exact
aggregate rigid-link membership and articulation roots; ordinary one-root
existing-link contracts retain V1. The owned path does not author rigid bodies,
masses, colliders, drives, joint state, or mimic schemas, and does not prove
simulation readiness. USD/USDZ readback is validated against the exact request
before its shared diagnostics and result artifacts are published.
Empty, all-unready, or policy-blocked inputs publish compatibility
diagnostics without a generated package. The 0.5 owned bridge supports
revolute and prismatic candidates; spherical candidates fail closed.

## Gate 3A And Gate 3B

The bundled `joint-agent-validation` skill runs two optional static checks on
the final USDZ. Gate 3A uses Isaac Sim Asset Validator:

```bash
mkdir -p ./joint-validation
"$ISAAC_SIM_PYTHON" \
  .agents/skills/joint-agent-validation/scripts/run_gate3a.py \
  .joint-agent-byoa/joint_rigger/rigged.usdz \
  --report ./joint-validation/gate3a.json
```

Gate 3B uses the explicit SimReady Foundation profile:

```bash
content-workflow-simready-validate-profile \
  .joint-agent-byoa/joint_rigger/rigged.usdz \
  --profile Prop-Robotics-Isaac \
  --profile-version 1.0.0 \
  --report ./joint-validation/gate3b.json \
  --stdout-log ./joint-validation/gate3b.stdout.log \
  --stderr-log ./joint-validation/gate3b.stderr.log
```

These gates inspect static package/schema conformance. They do not run dynamic
simulation or prove contact behavior, joint motion, containment, or stability.
Use `validation-agent-cli` separately for visual or behavior-evidence checks.
