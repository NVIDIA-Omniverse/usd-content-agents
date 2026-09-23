#!/usr/bin/env python3
"""Verify supplied public bytes, never unavailable private originals."""
import argparse
import collections
import hashlib
import json
import pathlib
from project_audits import CASES, ARMS, AUTHOR_HASHES, EXPECTED_SEAL, COMPLIANCE_STATUSES, privacy_check, sha


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(bundle):
    root = pathlib.Path(bundle)
    manifest = json.loads((root / "manifest.json").read_bytes())
    listed = set()
    for item in manifest["files"]:
        name = item["relative_path"]
        p = pathlib.PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or "\\" in name or name in listed:
            raise ValueError("unsafe/duplicate manifest path")
        listed.add(name)
        q = root / name
        if q.is_symlink() or not q.is_file() or q.stat().st_size != item["bytes"] or digest(q) != item["sha256"]:
            raise ValueError("manifest bytes changed")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    if actual != listed | {"manifest.json"}:
        raise ValueError("unlisted/missing public files")
    index = json.loads((root / "index.json").read_bytes())
    privacy_check(index)
    expected = {f"v2_{case}_{arm}" for case in CASES for arm in ARMS}
    if len(index["runs"]) != 20 or {r["run_id"] for r in index["runs"]} != expected:
        raise ValueError("not all twenty identities")
    if index["original_seal_sha256"] != EXPECTED_SEAL or manifest["original_seal_sha256"] != EXPECTED_SEAL:
        raise ValueError("wrong original seal binding")
    by_case = collections.defaultdict(list)
    statuses = collections.Counter()
    for row in index["runs"]:
        if row["public_file"] != f"audits/{row['run_id']}.json" or row["projection"] is not True or row["original_provided"] is not False:
            raise ValueError("invalid projection linkage")
        p = root / row["public_file"]
        if digest(p) != row["published_sha256"]:
            raise ValueError("published audit digest mismatch")
        d = json.loads(p.read_bytes())
        privacy_check(d)
        if any(d[k] != row[k] for k in ("run_id", "protocol_eligible", "compliance_status")) or d["outcome_blinded"] is not True:
            raise ValueError("decision mismatch")
        if d["compliance_status"] not in COMPLIANCE_STATUSES:
            raise ValueError("unknown frozen compliance status")
        if d["run_id"] != f"v2_{d['case_id']}_{d['arm']}" or type(d["protocol_eligible"]) is not bool:
            raise ValueError("invalid identity/decision")
        pub = d["publication"]
        if pub["original_adjudication_sha256"] != row["original_retained_sha256"] or pub["original_seal_sha256"] != EXPECTED_SEAL or pub["private_original_provided"] is not False:
            raise ValueError("invalid original attestation")
        sha(row["original_retained_sha256"])
        if set(d["evidence_sha256"]) != AUTHOR_HASHES:
            raise ValueError("missing AUTHOR hashes")
        for h in d["evidence_sha256"].values():
            sha(h)
        by_case[d["case_id"]].append(d["protocol_eligible"])
        statuses[d["compliance_status"]] += 1
        if "ui_evidence_boundary" in d:
            if d["protocol_eligible"] or d["compliance_status"] != "insufficient_evidence" or pub.get("ui_diagnostic_scope") != "posthoc_explanation_only_no_eligibility_change":
                raise ValueError("printer diagnostic promoted eligibility")
    eligible = sum(sum(v) for v in by_case.values())
    pairs = sum(len(v) == 2 and all(v) for v in by_case.values())
    if index["run_count"] != 20 or index["eligible_run_count"] != eligible or index["eligible_pair_count"] != pairs:
        raise ValueError("summary changed")
    preservation = json.loads((root / "preservation_checks.json").read_bytes())
    privacy_check(preservation)
    if preservation["original_seal_sha256"] != EXPECTED_SEAL or preservation["records_checked"] != 20 or preservation["checker_sha256"] != digest(root / "tools/check_preservation.py"):
        raise ValueError("preservation attestation binding mismatch")
    if {r["run_id"]: (r["original_retained_sha256"], r["protocol_eligible"], r["compliance_status"]) for r in preservation["runs"]} != {r["run_id"]: (r["original_retained_sha256"], r["protocol_eligible"], r["compliance_status"]) for r in index["runs"]}:
        raise ValueError("preservation status mismatch")
    return {"status": "PASS", "public_files_checked": len(listed), "runs": 20,
            "eligible_runs": eligible, "eligible_pairs": pairs, "status_counts": dict(statuses),
            "private_originals_reverified": False, "physical_acceptance_assessed": False,
            "boundary": "supplied bytes and projection consistency only"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    print(json.dumps(verify(p.parse_args().bundle), sort_keys=True, indent=2))
