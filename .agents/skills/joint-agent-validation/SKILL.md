---
name: joint-agent-validation
description: Validate Joint Agent USD or USDZ output with Gate 3A Isaac Sim Asset Validator and Gate 3B SimReady Foundation. Use when the user asks to validate joint output, run Gate 3A or Gate 3B, check SimReady readiness, inspect physics-schema failures, or prepare Joint Agent 0.5 Research Preview validation evidence.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - joint-agent
  - validation
  - isaac-sim
  - simready
  - usd
tools:
  - Shell
  - Filesystem
  - Python
compatibility: Requires Linux, a Joint Agent USD-family output, an Isaac Sim Python runtime for Gate 3A, and network access or an existing SimReady Foundation checkout for Gate 3B.
---

# Joint Agent Validation

Run the two optional static validation gates against a Joint Agent output and
preserve their raw reports. These checks help users understand whether a 0.5
Research Preview package is ready for a particular simulation workflow.

## When to Use

- Use when the user asks to validate a Joint Agent `.usd`, `.usda`, `.usdc`, or
  `.usdz` output.
- Use when the user mentions Gate 3A, Isaac Sim Asset Validator, Gate 3B,
  SimReady Foundation, SimReady readiness, or physics-schema findings.
- Use after Joint Agent has produced a final package. Do not validate a
  temporary staging layer when a published USDZ exists.
- Use `validation-agent-cli` for render, visual, prompt-match, or behavior
  validation that is outside the Gate 3A/3B static profiles.

## Limitations

- Joint Agent 0.5 is a Research Preview. A package can open and match the
  accepted joint graph while still failing these validation profiles.
- Gate 3A requires a compatible Isaac Sim Python runtime containing
  `omni.asset_validator.core` and `isaacsim.asset.validation`. This skill does
  not download Isaac Sim or accept its EULA for the user.
- Gate 3B may clone the public SimReady Foundation repository and create a
  managed validation environment when dependencies are missing.
- Neither gate repairs the USD. Preserve findings and route fixes back to the
  appropriate authoring workflow.
- Gate 3A and Gate 3B are static checks. They do not run dynamic simulation or
  prove contact behavior, joint motion, containment, or long-run stability.
- Gate 3A hashes all bytes in a USDZ archive before and after validation. For a
  loose USD layer, the hash covers only the root layer, so that evidence is
  diagnostic-only when external dependencies exist.
- Missing runtime dependencies are `BLOCKED`, not validation failures.
- By default, completed validation with profile findings exits successfully so
  an agent can collect the report. Add `--strict` when findings must fail the
  shell command.

## Prerequisites

Run from the repository root and activate the project environment:

```bash
source .venv/bin/activate
uv pip install -e . -e apps/joint_agent \
  -e agentic/packages/content_agent_workflows
```

For Gate 3A, locate the Python launcher for an installed Isaac Sim runtime.
Review and accept the applicable Isaac Sim license terms before setting the
EULA variable:

```bash
export ISAAC_SIM_PYTHON=/absolute/path/to/isaac-sim/python.sh
export OMNI_KIT_ACCEPT_EULA=YES
```

The 0.5 validation baseline is Linux with GLIBC 2.35 or newer, Python 3.12,
and Isaac Sim `6.0.0.1`. If a compatible runtime is not already installed,
create an isolated environment only after reviewing its license terms:

```bash
ISAAC_VENV="$HOME/.cache/content-agents/isaacsim-validator-6.0"
uv venv --python 3.12 --seed "$ISAAC_VENV"
"$ISAAC_VENV/bin/python" -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  'isaacsim[all,extscache]==6.0.0.1'
export ISAAC_SIM_PYTHON="$ISAAC_VENV/bin/python"
```

Do not print credentials or license tokens. Gate 3A and Gate 3B do not require
VLM API keys.

## Instructions

1. Resolve the final Joint Agent USD-family output to an absolute path.
2. Create a new validation output directory. Gate 3A reports are create-only;
   choose a fresh path and do not mix reports from different package identities.
3. Run Gate 3A when an Isaac Sim runtime is available. If it is unavailable,
   preserve the generated `blocked` report and continue with Gate 3B.
4. Run the SimReady Foundation preflight, then Gate 3B with the explicit
   `Prop-Robotics-Isaac@1.0.0` profile.
5. Read each JSON report. Distinguish `fail` from `blocked` or `error`.
6. Return the package path, both statuses, top hard findings, and report paths.
7. Do not describe the package as simulation-ready unless the requested
   profiles pass and the intended runtime behavior is separately verified.

## Gate 3A

Run the bundled diagnostic runner through the Isaac Sim Python launcher:

```bash
mkdir -p ./joint-validation

"$ISAAC_SIM_PYTHON" \
  .agents/skills/joint-agent-validation/scripts/run_gate3a.py \
  /absolute/path/to/joint-output.usdz \
  --report ./joint-validation/gate3a.json
```

Use `--strict` only when `fail` or `warning` must produce a non-zero exit code:

```bash
"$ISAAC_SIM_PYTHON" \
  .agents/skills/joint-agent-validation/scripts/run_gate3a.py \
  /absolute/path/to/joint-output.usdz \
  --report ./joint-validation/gate3a.json \
  --strict
```

The runner uses the `articulated-prop-v1` profile. It enables the installed
Basic, schema, geometry, material, layout, skeleton, USD Physics, Omni
SimReady, Isaac physics, and Isaac SimReady rule categories while excluding
robot-only packaging rules.

## Gate 3B

Preflight the managed Foundation runtime:

```bash
content-workflow-simready-preflight
```

Run the explicit Isaac-oriented prop profile. USDZ inputs are safely staged for
the upstream validator without changing the published package:

```bash
content-workflow-simready-validate-profile \
  /absolute/path/to/joint-output.usdz \
  --profile Prop-Robotics-Isaac \
  --profile-version 1.0.0 \
  --report ./joint-validation/gate3b.json \
  --stdout-log ./joint-validation/gate3b.stdout.log \
  --stderr-log ./joint-validation/gate3b.stderr.log
```

Add `--no-install-missing` for an offline check-only preflight or `--strict`
when profile findings must fail the shell command.

## Output Format

Return a concise result with:

- absolute input package path;
- input SHA-256 and its before/after identity check;
- Gate 3A status and `gate3a.json` path;
- Gate 3B status and normalized/raw report paths;
- hard finding counts grouped by rule or requirement;
- any `BLOCKED` runtime prerequisite and the exact next step;
- a clear statement that validation ran without modifying the package.

Interpret statuses as follows:

| Status | Meaning |
|---|---|
| `pass` / `PASS` | The selected static profile reported no blocking findings. |
| `warning` / `WARN` | Validation completed with warnings. |
| `fail` / `FAIL` | Validation completed and found profile violations. |
| `blocked` / `BLOCKED` | A runtime, extension, profile, or dependency was unavailable. |
| `error` / `ERROR` | The validator could not complete reliably. |

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| Gate 3A reports `blocked` before Kit starts | Isaac Sim Python or EULA acceptance is missing. | Set `ISAAC_SIM_PYTHON`, review the license, set `OMNI_KIT_ACCEPT_EULA=YES`, and rerun. |
| Gate 3A lists missing categories | The Isaac Sim runtime does not contain the required validator extensions/rules. | Use a complete compatible Isaac Sim installation. Do not treat a partial category run as Gate 3A. |
| Gate 3A reports joint state, drive, rigid-body, collider, mass, or articulation-root failures | The Research Preview output lacks complete simulation schemas. | Preserve the raw findings and report the 0.5 limitation; do not silently add guessed physics. |
| Gate 3B is `BLOCKED` | Foundation specs or the validator environment are unavailable. | Rerun preflight with network access, or pass `--foundation-root` and `--venv`. |
| Gate 3B rejects an existing staging workspace | A prior deterministic USDZ staging path remains. | Choose a fresh report path/output directory and rerun. |
| A gate fails but the USDZ opens | Package integrity and simulation readiness are separate contracts. | Report both facts without converting one into the other. |
