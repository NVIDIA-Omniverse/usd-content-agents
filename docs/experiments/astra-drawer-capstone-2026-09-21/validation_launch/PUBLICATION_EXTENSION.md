# Bounded public-builder extension points

The current curated bundle preserves Geometry04/Joint02, Physics03/06/07/08,
Physics08 task0/5, code through e640b8d, and an explicitly pending Validation slot.
Do not regenerate a final outcome until the final independent evidence is retained.

`publication_tools/build_public_bundle.py` has separate allowlist blocks for:

- Native physical runs: currently03/06/07/08. Add09 and later candidates explicitly;
  retain each typed assessment, native ValidationEvidence, terminal manifest,
  authored USD, packaged source, complete measured native trajectory, runtime
  reports and both bounded visual-review frame/receipt sets when present. Scan
  decoded USD/dependencies before adding exact bytes. Never overwrite an earlier
  candidate's projected or exact receipt.
- Task outputs: currently `evaluations/native08_source_clear_v1` and canonical
  `task_plots/native08_source_clear_v1`. Add separate candidate directories;
  verify actual report/USD/bindings/spec hashes and five prescribed seeds. Use
  root's deterministic complete `trace.jsonl.gz`, recording both compressed and
  original uncompressed digests. Keep files below100MB. Preserve task failures.
- Independent diagnostics: explicitly review and select
  `evidence/native09_startup_review/review.json`, the later source-bound Joint03
  endpoint repair/helper/qualification, and any subsequent source/cooking checks.
  Their measured conclusions do not prove native/task acceptance.
- Presentation: review `render_task_replay.py`, its source/pose readback receipt,
  real final renderer receipt and images. Label replay as measured-trace
  presentation, not a new task simulation or substitute native Validation evidence.
- Validation: add exact or explicitly projected request/preparation, selected plan,
  accepted plan, execution/operation index, actual per-check records, collected
  evidence index, outer typed assessment, independent typed review and terminal
  receipt. Exact original hashes remain distinct from projected public hashes.
  Exclude private stage logs, model reasoning, rollouts, raw tool streams, session
  identifiers, credentials and service/GPU identities. `run` exit0 alone is never
  the final Validation result.
- Code qualification: preserve `public_repo_required_tests_e640.log` (47passed,
  14skipped,3existing warnings) and empty successful `public_repo_skill_sync_e640.log`
  with timestamp/commit and retained digests. Add later test/qualification records
  explicitly, retaining prior failures and warning limitations.

`outcome.json` needs per-attempt actual native workflow/ValidationEvidence/typed
assessment statuses, task counts and same-asset digest. Final conjunction is true
only when native terminal assessment/review and all five physical task seeds pass
on the exact same final bytes; earlier Geometry conditional and failed attempts
remain disclosed. Do not fold these unscored repairs into frozen pilot scores.

Continue using generated `.gitattributes` (`* -text`, binary evidence patterns,
CSV CRLF and original USD EOF allowances), adding a narrow SVG whitespace rule for
a new plot path. Every inclusion, including attributes/scripts, must be manifested.
Run negative controls, full compressed/decompressed privacy/hash checks, links,
decoded USD review, dependency relocation and deterministic fresh rebuild. Preserve
unprovided intermediate digest references as such rather than silently substituting
final files at reused paths. No current public bundle changes are made by this plan.
