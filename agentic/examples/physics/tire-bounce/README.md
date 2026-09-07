# Tire Bounce Tuning and Refinement

This example reuses the public Tire_B01 USD packaged with Physics Agent. It
demonstrates two agent-owned behavior loops:

- `run-tune.sh` starts from a text goal and lets the coding agent author and
  revise the tuning scenario.
- `run-refine.sh` starts from an explicit bounce scenario and asks the coding
  agent to produce a visible airborne rebound, natural tipping, and final rest.

Both launchers run Physics authoring first, then request budgeted BoTorch
sweeps through `content-workflow-cli physics apply`. They do not invoke the
fixed `physics-agent refine` loop.

## Prerequisites

Complete the Agentic setup from [`../../../README.md`](../../../README.md),
then install the production Physics tuning extra:

```bash
./scripts/setup_content_agent.sh
source .venv/bin/activate
uv pip install -e "apps/physics_agent[tuning]"
```

The setup script installs the in-tree `apps/usd_cli[cli,server]` backend with
the required usd-exchange override. To refresh that backend independently in
an existing environment, run this from the repository root:

```bash
uv pip install -e "apps/usd_cli[cli,server]" \
  --overrides apps/usd_cli/requirements/usd-exchange-override.txt
```

Provision the isolated OvPhysX runtime described in
[`../../../../apps/physics_agent/docs/tuning.md`](../../../../apps/physics_agent/docs/tuning.md).
The workflow also requires a configured coding-agent runner, OVRTX readiness,
and the model credentials used by that runner.

## Run

From the repository root, run the text-only tuning example:

```bash
agentic/examples/physics/tire-bounce/run-tune.sh
```

Run behavior-guided refinement with:

```bash
agentic/examples/physics/tire-bounce/run-refine.sh
```

Each launcher permits up to 12 sweeps of 30 trials. Pass trailing CLI options
to lower those budgets or choose a fresh `--output-dir`. Successful runs keep
the promoted tuned USD at the canonical output path and preserve scenario,
sweep, decision, recording, render, validation, and summary evidence under the
run directory.

The scenario keeps per-trial rendering off. During each decision turn, the
agent exports the top candidates and their exact scenario recordings into the
confined run, then reviews OVRTX frames from those recordings before acceptance.
The wrapper replays the exact promoted bytes under the accepted scenario and
performs a separate OVRTX review; a failed final review rolls the publication
back.
