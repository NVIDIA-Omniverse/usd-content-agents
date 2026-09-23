# Sealed outcome-blinded protocol audits

This companion preserves all 20 sealed reviewer decisions: 19 eligible runs and
9 eligible pairs. These are protocol decisions, not physical task results.
The reviews were sealed at 2026-09-23 04:44:27 UTC before independent outcomes
were disclosed. See `index.json` for the exact seal digest and each original
adjudication digest. No failed or incomplete author attempt was removed.

The JSON files preserve reviewer reasons, observed stage chronology, author
claims, mechanical facts, billing gaps, limitations, and the nine original
AUTHOR evidence hashes. A native failed/blocked stage or an absent later stage
does not itself violate the treatment. Author/native success statements are
not independent physical acceptance. Missing numerical billing remains unknown
accounting; it does not by itself exclude a run. Automated audit effort is not
measured human review time.

The plain printer run remains **ineligible / insufficient_evidence** because
its frozen UI association was incomplete. The retained U+0085 / LF parsing
diagnosis is a posthoc explanation only. It neither repairs the frozen audit
nor proves an observed non-Ultra call nor promotes eligibility.

## Evidence boundary

Every audit here is a **projection**. Its original digest identifies retained
private bytes that are not distributed. The original seal is also not supplied.
Public file digests and original retained digests are separate fields.
Safe archive member locators, script names, line numbers and original hashes
are reviewer attestations, not the referenced evidence itself. No capture,
script body, tool/model content, request/session/call identifier, credential,
author geometry or independent evaluation result is included. Reviewer prose
and selected native status fields are summaries of the private review.

Public verification checks supplied bytes, all 20 experiment identities,
original-vs-projected bindings, statuses and privacy rules. It cannot re-open
private citations or independently establish the original compliance decisions.

```sh
python3 tools/verify_public.py --bundle .
python3 -m unittest discover -s tools -p test_projection.py -v
```

Reprojection and independent field-preservation checking require authorized
access to the exact private originals. Neither command reads captures,
archives, author scripts or evaluation files. The output must be new.

```sh
python3 tools/project_audits.py --adjudications PRIVATE_ADJUDICATIONS \
  --seal PRIVATE_SEAL --output NEW_PUBLIC_DIRECTORY
python3 tools/check_preservation.py --adjudications PRIVATE_ADJUDICATIONS \
  --seal PRIVATE_SEAL --bundle NEW_PUBLIC_DIRECTORY --output NEW_CHECK_JSON
```

The separately retained preservation receipt is an attestation from the
publisher's private-original comparison. A public-only check must not relabel
it as verification of unavailable original evidence.
