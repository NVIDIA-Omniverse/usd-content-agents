# Container Slide Refinement

This standalone Physics example reuses the normalized public
Container_Gray_C04 USD. It gives the container a gentle horizontal velocity,
tunes dynamic friction, and asks the coding agent to select a candidate that
slides forward, decelerates smoothly, and settles without bouncing, reversing,
tipping, separating its lid, or penetrating the floor.

The source asset is intentionally used as-is. This launcher does not run
Material Agent and does not make container color part of the behavior verdict.

The pre-tuning visual pass checks baseline authoring and runtime safety. It does
not require the untuned baseline to slide. The subsequent agent-owned tuning
loop applies the scenario's initial velocity and judges rendered candidate
motion against the slide goal.

The scenario keeps expensive all-trial rendering off. The tuning agent exports
top candidates and their exact scenario recordings into the confined run, then
renders those recordings through OVRTX before acceptance. The wrapper replays
the exact promoted bytes under the accepted scenario, performs a final
goal-level OVRTX review, and rolls back a failed publication.

## Prerequisites

Complete the Agentic setup from [`../../../README.md`](../../../README.md),
install the production Physics tuning extra, and provision the isolated
OvPhysX runtime described in
[`../../../../apps/physics_agent/docs/tuning.md`](../../../../apps/physics_agent/docs/tuning.md):

```bash
./scripts/setup_content_agent.sh
source .venv/bin/activate
uv pip install -e "apps/physics_agent[tuning]"
```

The setup script installs the in-tree `usd-cli` backend with the required
usd-exchange override.

## Run

From the repository root:

```bash
agentic/examples/physics/container-slide/run-refine.sh
```

The launcher permits up to three sweeps of eight trials. Pass trailing CLI
options to change those budgets or select a fresh `--output-dir`. A successful
run promotes the tuned USD to the canonical output path and keeps the scenario,
sweep, decision, recording, render, validation, and summary evidence under the
run directory.
