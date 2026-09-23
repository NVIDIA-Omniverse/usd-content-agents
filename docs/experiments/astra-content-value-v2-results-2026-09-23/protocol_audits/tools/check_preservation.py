#!/usr/bin/env python3
"""Compare original and projected leaves without calling the projector.

This needs the private originals; its result is an attestation in public use.
Only original adjudication JSONs, the seal and public projections are read.
"""
import argparse
import hashlib
import json
import pathlib


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def leaves(value, path=()):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from leaves(v, path + (k,))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from leaves(v, path + (i,))
        if not value:
            yield path, []
    else:
        yield path, value


def omitted(path):
    # Only these already-reviewed privacy omissions are allowed. Unknown
    # omissions, including numeric/status/claim/null changes, fail comparison.
    return (path[0] == "reviewer" or
            (path[-1] == "path" and path[0] in {"citations", "native_terminal_reference_integrity"}) or
            (path[-1] in {"call_id", "observed_call_created_unix", "tool_name"} and path[0] == "citations"))


def compare_records(originals, public, seal_sha):
    if len(originals) != 20 or len(public) != 20:
        raise ValueError("must retain twenty records")
    projected = {d["run_id"]: d for d in public}
    if len(projected) != 20:
        raise ValueError("duplicate projected identity")
    rows = []
    for original, digest in sorted(originals, key=lambda row: row[0]["run_id"]):
        d = projected[original["run_id"]]
        actual = dict(leaves(d))
        counts = {"preserved_numeric_leaves": 0, "preserved_boolean_leaves": 0,
                  "preserved_null_leaves": 0, "preserved_other_leaves": 0, "privacy_omitted_leaves": 0}
        for path, value in leaves(original):
            if omitted(path):
                counts["privacy_omitted_leaves"] += 1
                if path in actual:
                    raise ValueError("private leaf survived")
                continue
            if path not in actual or type(actual[path]) is not type(value) or actual[path] != value:
                raise ValueError("semantic leaf changed: " + "/".join(map(str, path)))
            kind = ("boolean" if type(value) is bool else "numeric" if type(value) in {int, float}
                    else "null" if value is None else "other")
            counts[f"preserved_{kind}_leaves"] += 1
        p = d["publication"]
        if p["original_adjudication_sha256"] != digest or p["original_seal_sha256"] != seal_sha or p["private_original_provided"] is not False:
            raise ValueError("original digest/boundary mismatch")
        if original.get("ui_evidence_boundary") and p.get("ui_diagnostic_scope") != "posthoc_explanation_only_no_eligibility_change":
            raise ValueError("missing printer posthoc boundary")
        rows.append({"run_id": original["run_id"], "original_retained_sha256": digest,
                     "protocol_eligible": original["protocol_eligible"],
                     "compliance_status": original["compliance_status"], **counts,
                     "all_nonprivate_semantic_leaves_exact": True})
    return {"schema_version": "private-original-to-public-preservation-check.v1", "status": "PASS",
            "source_access_scope": "seal_and_twenty_adjudication_JSON_files_only",
            "public_interpretation": "publisher_attestation_private_originals_not_supplied",
            "original_seal_sha256": seal_sha, "records_checked": 20,
            "eligible_runs": sum(row["protocol_eligible"] for row in rows),
            "independent_physical_outcomes_accessed": False,
            "checker_sha256": sha(pathlib.Path(__file__).read_bytes()), "runs": rows}


def check(adjudications, seal_path, bundle):
    seal_raw = pathlib.Path(seal_path).read_bytes()
    seal_sha = sha(seal_raw)
    index = json.loads((pathlib.Path(bundle) / "index.json").read_bytes())
    if index["original_seal_sha256"] != seal_sha:
        raise ValueError("seal mismatch")
    seal = json.loads(seal_raw)
    originals, public = [], []
    for row in seal["runs"]:
        run_id = row["run_id"]
        if "/" in run_id or "\\" in run_id or ".." in run_id:
            raise ValueError("unsafe run identity")
        raw = (pathlib.Path(adjudications) / (run_id + ".json")).read_bytes()
        if sha(raw) != row["adjudication"]["sha256"]:
            raise ValueError("stale original")
        original = json.loads(raw)
        if any(original[k] != row[k] for k in ("run_id", "compliance_status", "protocol_eligible")):
            raise ValueError("seal decision mismatch")
        originals.append((original, sha(raw)))
        public.append(json.loads((pathlib.Path(bundle) / "audits" / (run_id + ".json")).read_bytes()))
    return compare_records(originals, public, seal_sha)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adjudications", required=True)
    p.add_argument("--seal", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = check(a.adjudications, a.seal, a.bundle)
    with pathlib.Path(a.output).open("x") as f:
        json.dump(result, f, sort_keys=True, indent=2)
        f.write("\n")
    print(json.dumps({"status": result["status"], "records_checked": result["records_checked"]}))
