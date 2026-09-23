# Author control evidence

This publication contains privacy projections of trusted control records for all
20 author attempts. These records preserve input and frozen tool/profile hashes,
fixed budgets, measured author and allocated-lane time, source preservation,
process reap, UI association completeness, observed billing completeness, and
verified retention hashes. They do not decide protocol eligibility, treatment
compliance, physical acceptance, or comparative benefit.

All 20 attempts were reaped, their archives verified, and their sources unchanged.
UI association audits are complete for 19 of 20; printer plain remains incomplete.
Billing records are complete for 12 of 20. The ledger retains 5,354 completed
requests and 13 requests with unknown terminal usage; unknown usage is not zero.
Capture error counts represent entries in final metadata, not distinct incidents.
Allocated-lane durations retain trusted monotonic measurements rather than being
reconstructed from displayed Unix timestamps.

The Linux receipt covers one Joint configuration-load regression, with 228 source
files checked against its recorded publication commit. It is not a simulation
acceptance test or a claim that the entire suite ran on Linux.

Transport health records contain 881 checks for node0 and 885 for node1, with one
recovered node0 timeout and no reconnects in the declared window. The 13.357-second
interval between failed and recovered health records is not a measured author
outage. No author impact is inferred. Twenty failures on each node after the
intentional gateway shutdown are reported separately.

`publication_manifest.json` hashes the supplied bytes and distinguishes projected
evidence from this generated explanatory file. Original receipt digests are
attestations to separately retained private records: those original bytes and
archive payloads are not supplied here. The 23 previously reviewed control
projections are copied byte-for-byte; the transport summary is a new allowlisted
projection. No private model identifiers, raw histories, worker configuration,
credentials, authored programs, geometry, or independent evaluation outputs are
included. Hash verification establishes integrity, not the truth of an underlying
physical result. No physical or eligibility results are asserted in this bundle.
