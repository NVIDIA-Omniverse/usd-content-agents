# Additional exact submitted assets

[Download the 95.31 MB experiment companion](https://github.com/NVIDIA-Omniverse/usd-content-agents/releases/download/astra-content-value-v2-20260923-data/additional_publication_v2.tar.gz).
This is a data-only experiment prerelease, not a product software release.

The archive contains 26 unchanged submitted payload files (212,177,755 bytes)
and 19 notices, review records, verifier/manifest files: 45 ordinary files total.
It covers both arms of refrigerator, gripper, Thor arm, excavator and Dextra hand.
The Content Agents refrigerator has an explicit no-submission record; the Content
Agents gripper has its incomplete USD scene with absent bindings. No unsuccessful
artifact is replaced with a repaired version.

- Archive SHA256: `a58fe0ba18b75f1917a17d5dfa12655e4d7d24ef035a3aab2b2ba20a203d6e30`.
- Archive bytes: `95306294`.
- Contained manifest SHA256: `7e6b0d27b703899a92a3df5aea6613a1d3fe6247c1f5543373e803e7880898cf`.

Inspect [attempt records](attempts.json), the [exact contained manifest](bundle_manifest.json)
and [packing receipt](delivery_receipt.json) before downloading. The receipt's
`VERIFIED_LOCAL_DELIVERY_NOT_UPLOADED` status records the earlier packaging step;
publication and the release download are subsequent events. The manifest describes
files inside the archive, not geometry present in this small Git directory.

After verifying the archive digest, unpack into a fresh directory, enter the
extracted `additional_publication_v2` directory, and run `python3 -B verify.py .`.
The verifier rejects Python optimization, checks exact
membership/hashes and dependency bindings, and can additionally recheck decoded
USD identity with `--usd-review` and `usd-core==25.5`. It executes no submitted
programs and runs no model, renderer or physics solver.

The archive preserves upstream attribution and notices, including CC BY 4.0 for
the refrigerator, MIT for gripper/excavator and CC BY-SA 4.0 for Thor/Dextra
adaptations. Native all-rights-reserved metadata and separately unverified vendor
provenance remain explicitly documented alongside the repository license grants.
No payload byte or authored path was rewritten to hide those caveats.

Cases05/06 remain withheld under recorded provenance uncertainty; case09's
modified-source distribution scope remains unresolved. The original source,
frozen reference geometry and runtime dependencies for these five cases are not
included. This output companion is not a complete ten-case replay or a new task
acceptance claim. See [the independent physical evidence](../physical_evidence/README.md).
