# Independent review of pilot interpretation and accounting

Reviewed the retained drawer pair, `METHOD.md`, `summarize_results.py`, `reconcile_usage.py`, `extract_tool_audit.py`, and `export_completed.py`. This review does not change author outputs, frozen evaluators or installed dependencies.

This review predates the final accounting repairs. The original findings below are retained; their resolutions are recorded at the end.

## Runtime completeness changes the causal claim

The matched tools statement is accurate as a statement about the configured common environment. It does not show that every workflow-specific repair backend was provisioned. Case02's preserved native repair receipts show that `scene_optimizer_deinstance` and `geogram_local_repair` both returned `unavailable`, with no output.

Scene Optimizer Core is explicitly included in the public Content Workflow CLI setup sequence: its README lines106–110 instruct fetching build resources for optimized workflows. It is a prerequisite for the route that Case02 actually attempted. The native resolver checks `WU_SO_PACKAGE_DIR`, then the current directory's `.build-resources/scene_optimizer_core`; merely placing the resources in a different repository directory would not guarantee discovery from a scored run directory. This is a documented setup/configuration gap for that path, not evidence that a fully provisioned Scene Optimizer could not perform the operation.

Geogram is a separately qualified native repair dependency. Geometry Repair's README lines246–251 requires vorpalite1.10.0 and an approved executable digest. Its package metadata describes other kernels as separate, explicitly qualified optional dependencies. This means it is optional for installing the base package, but required for executing the selected Geogram repair route. The scored receipt reports it was not installed or on PATH.

The source topology issue and native gate rejection are observed facts. The unavailable repair routes are also observed facts. Their combination supports a configured-runtime failure to deliver the task. It does not identify whether completing the documented setup would have made repair pass, nor isolate the effect of workflow logic from runtime completeness.

Suggested METHOD addition:

> The common environment did not include every workflow-specific native repair backend. In Case02, the selected Scene Optimizer deinstancing route could not discover its documented build resources, and the selected Geogram route could not find its separately qualified vorpalite executable. These are preserved setup/capability limitations of this configuration. They count toward task non-delivery in this deployment pilot, but cannot establish that a fully provisioned workflow would fail, or that workflow logic alone caused the paired difference. No dependency was installed or changed after observing the scored result.

Evidence excerpts, original file hashes and scored attempt reasons are in `optional_backend_review_evidence.json`.

## Accounting and privacy review

- The aggregation correctly uses independent acceptance, withholds a pass pending a positive protocol audit, treats concrete false passes separately from inconclusive evidence, includes failed authoring effort, and leaves cost null with missing usage or rates. Five physical seeds are correctly described as repeat trials of one authored mechanism.
- `summarize_results.py` currently checks that the four numeric rate fields are present but does not require the template's `rate_source` or `effective_date`. This does not affect current null costs, but a later numeric-only rates file could produce a cost without the provenance promised in METHOD. Gate dollar calculation on nonempty provenance and finite nonnegative rates, and validate `0 <= cached_input_tokens <= input_tokens` before pricing.
- `reconcile_usage.py` extracts only model/effort and numerical accounting metadata. It filters response records by the owning session ID, deduplicates response IDs within that session, checks the final counter, and requires completion after final usage. Missing or interrupted evidence stays incomplete. These are client-side accounting checks, not provider attestation. Session deduplication uses the session filename; using the extracted session ID would make the promised distinct-session policy robust to renamed copies. No duplicate-count error is established for the retained pilot files by this code review.
- `extract_tool_audit.py` explicitly creates private tool-request files. `export_completed.py` is a local-retention archive, not a public sanitizer: it selects entire run/audit trees and excludes certain private paths, but does not inspect arbitrary embedded credentials or private text. Its scope warning is appropriate. A public bundle must use a separate explicit allowlist and exclude `tool_calls.private.json`, raw events/rollouts, author stdout/stderr and tool-operation metadata unless specifically reviewed and redacted. Do not treat local archive inclusion as publication approval.
- Public code and receipts can contain internal hostnames, service addresses or absolute infrastructure paths even when they contain no credentials. Apply the publication boundary to file contents, not only filenames, and preserve an internal original plus a separate reviewed public copy.

## Drawer-specific interpretation

The plain drawer artifact passed the frozen physical task on all five independent seeds. The Content Agents arm preserved a native failed handoff and submitted no final physical mechanism; its contractual rejection is distinct from the raw diagnostic evaluator's null-binding `INCONCLUSIVE` report. Neither arm demonstrated a false pass. Both protocol audits are evidence-bound valid, not OS-level access enforcement. The plain repair counter convention is ambiguous for a constructor failure before the first complete candidate, but both reasonable counts remain within the two-repair limit.

The accepted drawer result supports bounded force-driven articulation with a free payload in native physics. It does not establish hardware fidelity, robot manipulation, or a general workflow advantage. Replay images are explanatory views of independently recorded trajectories and cannot replace the five-seed acceptance evidence.

## Accounting review resolutions

The final aggregator requires rate provenance, an effective date, finite nonnegative rates and allocation time, and nonnegative integer token counts with cached input no greater than total input. Missing information leaves dollars unknown. The usage reconciler deduplicates sessions by their recorded identity, including renamed copies, while retaining the per-response and final-counter checks. Ten regression tests cover these changes, conservative evaluator reviews, numerical corrections, and separation of physical acceptance from protocol eligibility.

Publication uses a separate explicit allowlist with content scanning and per-file original/published digests. Private retention archives remain excluded. The method now states the missing repair dependencies, missing Git provenance, observed cross-case process exposure, and confirmed global process-cap breach. These repairs improve the reporting; they do not retroactively repair the experimental protocol.
