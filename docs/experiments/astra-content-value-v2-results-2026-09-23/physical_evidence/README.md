# Independent physical evidence

Read evidence_index.json first. A development bundle with complete_twenty_attempts
false is incomplete and explicitly lists pending attempts. A final edition covers
all twenty author attempts, preserving PASS, FAIL, INCONCLUSIVE and individual
checks without selecting favorable seeds or suppressing failed metrics.

Each evaluation_summary.json is the trusted evaluator wrapper summary. The exact
native acceptance.json or report.json, parser/structural/preflight reports, and
per-seed reports are separate evidence. A summary FAIL for an immutable scene or
bindings that were not submitted legitimately has a submission-rejection report
and no native physical report. Missing reports never become synthetic PASS data.
Native report status and protocol eligibility answer different questions; this
companion does not adjudicate eligibility or comparative benefit.

Reports retain all original numerical thresholds, metrics, checks, and any embedded
trace arrays. Standalone numeric JSONL traces are supplied as lossless gzip derivatives.
Decompression returns their exact original bytes and format; no CSV conversion,
rounding, arithmetic, or reserialization is performed. The manifest separately
binds supplied compressed bytes and exact decompressed original-member bytes.
Decompression is bounded by that retained member length. Trace
sample counts do not independently prove task acceptance. Drawer scene-wide
contact samples do not identify contact actors; sensor reports and their original
limits must be read as supplied. No new contact provenance is inferred.

Run python3 -B verify_physical_evidence.py --verify . to verify provided membership,
hashes, privacy and summary/report bindings. No simulator or author code is run.
The bundled privacy scanner has the exact preregistered digest. Verification of
this companion is an integrity check, not a replay of the physical evaluation.

The manifest distinguishes exact retained reports, lossless gzip trace derivatives,
and generated index/documentation. The gzip bytes are newly generated and are not
falsely described as original supplied bytes. Each original member is bound to its evaluation archive/manifest
and original author run. The full archives and private retention proofs are not
provided: their hashes attest separately retained originals. Extraction never
follows links and selects only bounded regular evaluator reports/traces. Requests,
runtime configurations, solver/process logs, authored programs, model histories,
and geometry are excluded. Cases 05 and 06 contain no geometry payload here.
