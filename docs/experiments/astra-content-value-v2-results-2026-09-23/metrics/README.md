# Quantitative comparison metadata

The manifest labels this bundle as either a synthetic fixture or final assembled
results. A synthetic fixture is qualification data, not an experimental outcome.
Actual result production is authorized only after all twenty blinded protocol
audits and independent evaluations are complete. This builder neither performs
nor substitutes for those audits and physical tests.

Use **CPython 3.9.6**, the original controller/aggregation runtime. Run
`python3.9 -B verify_results_metadata.py --verify .` to verify membership and
recompute the exact comparison and per-request prices using the supplied,
unchanged, digest-pinned accounting.py and compare_results.py. Exact runtime is
enforced: Python 3.12 changed float summation, producing last-bit aggregate cost
differences in synthetic qualification. No tolerance or rounding substitutes for
exact reproduction. See https://docs.python.org/3.12/library/functions.html#sum .
This pin applies to quantitative postprocessing, not the simulation runtime. Python -O is
unsupported because the frozen calculators use assertions. The fixed historical
rates are API-equivalent reference prices, not invoices or current price advice.

rows.json retains all twenty attempts, the explicit eligibility/verdicts/claims,
public assigned case/arm/lane identifiers, input hashes and measured times. Private
model request/response/session/thread/turn identities are absent. Request ordinals
preserve original list order only; they do not identify an account or session.
Human interventions are count placeholders, without private IDs, text or dates.
Unmeasured human minutes remain null. No zero effort or human savings are inferred.

Token counts retain observed input/output totals and available cache/read/write
or reasoning subsets. Missing usage stays missing, unknown cache partitions stay
unknown, and missing terminal cost has an unknown upper bound. Reasoning tokens
are not added again to output totals. All-attempt and eligible-pair calculations,
false positives versus unresolved positive claims, and zero-acceptance undefined
unit costs are reproduced without dropping rows or requests.

per_request_accounting.json omits textual reason/scope fields while preserving
every quantitative accounting field. comparison.json contains only the exact
frozen comparator's output, including its fixed public scope caveats. No author
reason text, prompts, transcripts, adjudication prose, geometry or executable
authored code is copied. Provenance contains original input digests, not private
paths; hashes are attestations to retained originals, not a claim those private
inputs are included here. Projection equality does not establish the truth of
the underlying audit or physics result.
