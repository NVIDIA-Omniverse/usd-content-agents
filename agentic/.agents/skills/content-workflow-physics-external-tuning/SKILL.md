---
name: content-workflow-physics-external-tuning
description: Use when a long-running coding agent owns the outer loop of external-runtime (BYOR) physics refinement — requesting budgeted qualification-gated tune-external sweeps through the wrapper broker, visually judging rendered rollout evidence against a behavior goal, revising the active parameter search, and writing digest-bound accept/revise/stop decisions.
metadata:
  author: NVIDIA Omniverse
---

# content-workflow-physics-external-tuning

Use this skill when a wrapper-launched coding agent runs the agent-owned
external-runtime (BYOR) refinement loop: the customer's own simulator runtime
evaluates every trial through the qualified adapter, the wrapper-owned broker
runs budgeted judge-free `tune-external` sweeps, and YOU replace the built-in
VLM judge and LLM refiner with your own evidence review and search decisions.

This skill owns workflow method. The broker owns budget, qualification
enforcement, engine invocation, and evidence publication. The engine
(`physics-agent tune-external`) owns fixed-seed optimization, the adapter
protocol, recording integrity, and winner rendering.

## When to Use

- The session prompt carries a
  `content-agents.physics-external-tuning-session-task.v1` task block with a
  sweep broker endpoint and an external-runtime parameter catalog.
- The tuning target is a customer repository task (for example an IsaacLab
  environment), not a standalone physics USD — there is no usd-cli session
  and no USD candidate to materialize.

## Limitations

- The scalar objective is adapter-owned and fixed. You choose *where to
  search* (active parameter subset and bounds), never *what is measured*.
  Changing the objective requires changing the customer adapter and
  re-qualifying — that is outside this session.
- Qualification approval is wrapper/operator-owned. You never approve, edit,
  or re-run qualification; a fingerprint mismatch fails sweeps closed.
- Sweeps are sequential and budgeted. Parameters omitted from your next
  active search are pinned by the broker to the previous sweep's winning
  values (nominal values before the first sweep).
- The rollout recording renders the adapter's recorded proxy geometry, not
  the customer renderer. Judge motion and configuration, not appearance.
- Final review evidence must render through the OVRTX backend. The engine's
  `evidence.playback_renderer: remote` option is not supported here: a
  generic REST endpoint carries no verifiable OVRTX provenance, so the
  wrapper refuses such a runtime config before qualification and the broker
  fails evidence publication closed if one slips through.

## Instructions

1. Read the structured task block and the external contract file it names.
   Understand the behavior goal, the objective (name, unit, direction), the
   parameter catalog, and the budget before requesting anything.
2. Request one sweep per iteration with the absolute sweep client path from
   the task block (`run` subcommand). Omit `--active-search` on the first
   sweep to use the runtime config's declared search. Never wrap the client
   in `uv run` or any package manager, and never invoke `run_external_tune`
   or the `physics-agent` CLI directly — unbrokered work cannot be verified
   and is rejected at conclusion.
3. Review the evidence the broker published under the iteration directory:
   `evidence.json` plus `frames/*.png`. Read the frames as images — they are
   the behavior ground truth. Use `distinct_frames` when your runner limits
   byte-identical images (a fully settled body renders identical frames).
   Weigh `best_objective` and the per-trial `history` for search guidance.
4. Decide and record. Write one digest-bound decision file per iteration
   (`accept` / `revise_search` / `stop`) and a terminal result file, exactly
   as the session prompt specifies: hash-chain each decision to the previous
   decision file's SHA-256, copy `evidence_sha256`, `recording_sha256`, and
   frame digests from the sweep record, and for `accept` list the frames you
   actually reviewed in `reviewed_frames` (`selected.recording_sha256` is
   mandatory on accept). The wrapper rehashes everything at conclusion; a
   mismatch fails the run. `accept` and `stop` are terminal: they must be
   the final decision, and the result's status must agree (`accept` →
   `accepted`; `stop` → `stopped`, or `budget_exhausted` when the next
   reservation was refused).
5. Search revision heuristics:
   - Best value pinned at a bound (a rail) → widen that bound.
   - Objective improved but behavior is wrong in the frames → the active
     subset is probably wrong; activate a different catalog parameter.
   - Trials mostly failed or timed out → shrink the search toward the
     nominal values; the runtime may be unstable far from them.
   - The first sweep can land entirely worse than the runtime's hand-tuned
     defaults (the optimizer is not seeded with nominal values) — judge
     iteration 2+ before concluding the goal is unreachable.
6. Stop honestly. `stop` with a clear rationale (or `budget_exhausted` after
   a refused reservation) beats a forced `accept`. Only `accepted` publishes
   a final result, and the wrapper independently verifies every digest.

## Output Format

- `raw/physics_external_tuning_decision_<i>.json` — one per iteration,
  schema `content-agents.physics-external-tuning-decision.v1`.
- `raw/physics_external_tuning_result.json` — terminal, schema
  `content-agents.physics-external-tuning-result.v1`.
- A short final response: sweeps used, final status, best parameters and
  objective, what the frames showed, remaining limitations.

## Troubleshooting

- Sweep client exit 3 — budget or phase deadline refused the reservation.
  Write the result artifact (`budget_exhausted` if you needed more sweeps)
  and stop; do not retry in a loop.
- Sweep client exit 7 — the previous sweep's runtime is still terminating
  (common right after a deadline-exceeded sweep). This is transient and
  consumes no budget: wait roughly a minute and retry the same command.
- Your blocking `run` call was killed (timeout, crash) — the reserved sweep
  still exists and must be cited. Run the sweep client's `list` subcommand
  to recover every reserved sweep's `sweep_id`, state, and record without
  reserving new budget, then continue the decision chain from it.
- Sweep client exit 4/5 — deadline or engine/runtime failure. The failed
  sweep is still a valid decision reference: cite it in a `revise_search` or
  `stop` decision with what you learned. External trials are minutes each
  (simulator startup dominates); a timeout usually means the deadline is too
  tight for the trial count, so request fewer `--max-trials` next sweep.
- `fingerprint_changed` in a sweep error — the customer runtime was modified
  after qualification. Stop with `tool_failure`; the operator must re-qualify.
- Frames all black or identical when motion is expected — report it in the
  rationale and prefer `revise_search`/`stop` over accepting numbers you
  cannot visually confirm; the adapter's evidence camera may be broken, which
  is an adapter defect the operator must fix.
