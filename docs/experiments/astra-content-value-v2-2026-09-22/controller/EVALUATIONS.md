# Post-author independent evaluation

`run_evaluations.py` is a new controller, not an author or model launcher. Deploy
the same qualified controller, frozen evaluator/reference bytes, source closure,
and `evaluator_runtime/rootfs` to both workers before use. Do not use an author
rootfs for evaluation. Native runtime remains the qualified `/tools/ovphysx-venv`.

From the local experiment root, first validate the complete author batch:

```sh
python3 controller/run_evaluations.py \
  --config controller/private/MEASURED_BATCH/controller.json \
  --jobs controller/private/MEASURED_BATCH/jobs.json \
  --batch measured-v2 --validate-only
```

Use the identical command with a fresh `--output` directory and without
`--validate-only` only after the entire author queue is terminal. It refuses
missing assignments, unreaped ledger rows, mismatched reap receipts, unverified
archives, incomplete author cleanup, changed sources or changed frozen files.
The twenty assignments retain their original four lane queues and pair order.

The authoritative submission declaration is the exact regular JSON file
`private/author_delivery.json` in each fully verified author archive. The runner
never substitutes an author log, inferred filename, or a repaired candidate.
`pack_workspace` transfers workspace data only; home, authorization and sessions
are excluded. No authored Python, shell script or command is executed.

Each fresh evaluation ID gets root-owned `0600` config and request files and a
bounded tmpfs input mount. The compressed workspace limit is 9 GiB; the extracted
workspace limit remains 8 GiB. Input pages, restoration, native evaluation, and
retention share the lane's maximum 32 GiB cgroup. The input tmpfs is populated by
a process inside that lane, so it does not consume the worker's limited `/opt`
disk allocation. The existing evaluator allocates 8 GiB each for workspace and
home/evidence. Actual combined use can therefore reach the cgroup cap before all
individual tmpfs limits are reached; such resource failure is infrastructure
`INCONCLUSIVE`, not an asset failure or permission to enlarge the budget.

An atomic root-owned lane reservation remains in place between SSH operations.
Only the controller's complete, locally verified and fsynced evaluator archive
permits exact run cleanup and reservation release. The retained archive includes
the complete restored workspace, evaluator home/evidence, native logs, requests,
configuration and terminal receipts. It supports early submission rejection even
when the existing evaluator emitted no `launch.json`. Retention uses the explicit
trusted lane and `batch_evaluation.json`; it does not invoke the author-specific
retention CLI or infer a lane from a missing launch field.

Any transport, retention, mount, cleanup or lane ambiguity stops that sequential
queue. Partial archives and remote staging stay in place. There is no automatic
retry, resumed evaluation or timeout-based lane release. Inspect and retain the
exact failed operation before an explicitly identified recovery attempt.

Local qualification passed 26 controller, restore and retention tests, covering
the batch gate, source identity, no authored execution, early rejection,
occupied/reserved lanes and failed retention. Agent07 separately qualified the
exact `evaluate_run.execute` primitive. The new orchestration then passed a full
synthetic round trip on node1 lane2: workspace packing, bounded input tmpfs,
native CPU gravity/zero controls, complete 30-member archive verification,
fsync, exact cleanup and reservation release. See
`qualification/evaluation_batch_agent06_v1/lifecycle_qualification.json` and its
retained archive. The twenty-job dispatch gate itself was tested with local
fixtures; the live lifecycle used one synthetic run. No scored evaluation was
launched while implementing or testing this controller.
