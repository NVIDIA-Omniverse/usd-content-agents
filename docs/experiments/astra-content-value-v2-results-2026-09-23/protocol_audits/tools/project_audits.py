#!/usr/bin/env python3
"""Create a public, explicitly non-evidentiary projection of sealed audit records.

Only the seal and twenty adjudication JSON files are opened. Capture, archive,
author, evaluation, and model files are never opened or executed.
"""
import argparse
import hashlib
import json
import math
import pathlib
import re

EXPECTED_SEAL = "ec9ceb5efc6e231df1bda550aa0386e0923a6f534a9a8634b96726535fe30d48"
CASES = ("01_drawer", "02_conveyor", "03_hinge", "04_gripper", "05_vise",
         "06_engine", "07_robot_arm", "08_excavator", "09_printer", "10_complex")
ARMS = ("plain_astra", "content_agents")
COMPLIANCE_STATUSES = {"observed_compliance", "observed_noncompliance", "insufficient_evidence"}
IDENTITIES = "run_id case_id arm lane_id protocol_sha256 task_sha256 input_sha256 source_sha256".split()
AUTHOR_HASHES = set("launch reap declaration timing usage ui_metadata ui_association author_retention author_manifest".split())
POLICIES = "protocol_sha256 analysis_plan_sha256 treatment_adjudication_sha256".split()
SCALARS = set(("schema_version outcome_blinded protocol_eligible compliance_status reason started_utc completed_utc "
               "analysis_plan_sha256 treatment_adjudication_sha256 model_request_count ui_association_complete "
               "terminal_billing_complete independent_physical_evaluation scored_result_data_modified frozen_files_modified").split()) | set(IDENTITIES)
MAP_FIELDS = {
    "mechanical_checks": "all9_reference_bytes association_reproduced captured_bytes drained_usage fresh_model job_identity limits metadata_reproduced prompt_exact protocol_exact reaped retained snapshot_complete source_inventory_binding source_unchanged task_exact time ui_complete",
    "mechanical_findings": "admitted_requests all_observed_ui_matches allocation_released billing_complete budget_s cgroup_empty cpu_quota_millicores exact_identity_prompt_and_lease_verified fresh_context_verified frozen_usage_within_deadline late_admitted_requests memory_bytes namespace_reaped pids_max recorded_late_finishes source_unchanged swap_bytes terminal_billing_records ui_association_complete",
    "review_coverage": "all_original_archive_bytes_rehashed all_original_capture_and_navigation_hashes_verified capture_count copied_ancestor_context_not_counted_as_new_invocation navigation_not_used_as_verdict review_navigation_sha256 script_inventory_count script_inventory_sha256 unique_navigation_tool_calls",
    "author_budget_observation": "cgroup_populated elapsed_seconds namespace_init_exited timed_out",
    "author_claim_only": "claimed_accepted declared_repairs_used physical_acceptance task_outcome",
    "ui_evidence_boundary": "admission_identity_associations all_observed_ui_matches archive_member associated_requests capture_sha256 disposition frozen_parser_file frozen_parser_sha256 frozen_str_splitlines_invalid_fragment_count frozen_ui_complete lf_record_line lf_record_sha256 literal_lf_json_records_valid parser_lines record_type request_count retained_capture_error_count separator separator_count_in_lf_record_117 terminal_response_id_associations unassociated_request_count",
}
NATIVE_FIELDS = "claimed_accepted gate_dispositions receipt_status returncode schema_version status success verdict workflow".split()
GATES = "cross_stage_integrity package_integrity runtime_validation static_validation visual_quality".split()
PEER_CITATION = "archive_member archive_sha256 kind line line_delimiter member_sha256 navigation_line observation observed_fields scope".split()
CAPTURE_CITATION = "capture_archive_member capture_line capture_sha256 kind navigation_ordinal call_input_sha256".split()
SCRIPT_CITATION = "archive_member kind reviewed_lines scope sha256".split()
SCRIPT_INVENTORY = "archive_member line_count review_scope sha256".split()
RECORD_CITATION = "archive_member kind scope sha256".split()
BOUNDARY = ("This is a privacy projection of sealed reviewer attestations. Private originals, captures, "
            "scripts, archives and native contracts are not supplied here. Their original digests and "
            "safe member locators identify retained private bytes; this bundle cannot reverify those bytes "
            "or independently establish treatment compliance or physical acceptance.")
FORBIDDEN = re.compile(
    r"(?:/Users/|/opt/|/home/(?!agent(?:/|\b)|evaluator(?:/|\b))|"
    r"\b(?:call_|resp_|sess_|thread_)[A-Za-z0-9-]{8,}|"
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|"
    r"https?://|\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,}|"
    r"[A-Za-z0-9.-]+\.(?:teleport\.sh|horde\.nvidia\.com)|"
    r"\b(?:api_key|access_token|authorization)\s*[=:])", re.I)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return digest_bytes(pathlib.Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()


def sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid SHA256")
    return value


def member(value):
    p = pathlib.PurePosixPath(value)
    if not isinstance(value, str) or p.is_absolute() or ".." in p.parts or "\\" in value or str(p) != value:
        raise ValueError("unsafe archive member")
    if not value.startswith(("workspace/", "private/session_captures/", "harness/")):
        raise ValueError("archive member outside reviewed roots")
    if FORBIDDEN.search(value) or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("unsafe identifier in member")
    return value


def scalar(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("unexpected nested or non-finite scalar")


def pick(data, keys, dropped=()):
    if not isinstance(data, dict) or set(data) - set(keys) - set(dropped):
        raise ValueError("unknown schema fields")
    return {k: data[k] for k in keys if k in data}


def flat(data, keys, dropped=()):
    result = pick(data, keys, dropped)
    for k, v in result.items():
        if k in ("parser_lines", "reviewed_lines"):
            if not isinstance(v, list) or any(type(n) is not int or n < 1 for n in v):
                raise ValueError("invalid line numbers")
        else:
            scalar(v)
        if k.endswith("sha256"):
            sha(v)
        if k in ("archive_member", "capture_archive_member", "frozen_parser_file"):
            member(v)
    return result


def citation(data):
    result = pick(data, PEER_CITATION, ("call_id",))
    for k, v in list(result.items()):
        if k == "observed_fields":
            fields = pick(v, NATIVE_FIELDS)
            for name, value in fields.items():
                if name == "gate_dispositions":
                    fields[name] = flat(value, GATES)
                else:
                    scalar(value)
            result[k] = fields
        else:
            flat({k: v}, (k,))
    return result


def privacy_check(data):
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"call_id", "session_id", "request_id", "response_id", "thread_id", "agent", "path", "prompt", "arguments", "tool_output", "raw", "authorization", "api_key", "observed_call_created_unix"}:
                    raise ValueError("private field in public projection")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and FORBIDDEN.search(value):
            raise ValueError("private locator/identity/secret pattern in projection")
    walk(data)


def project(data, original_sha, seal_sha):
    allowed = SCALARS | set(MAP_FIELDS) | {"reviewer", "evidence_sha256", "treatment_findings", "limitations", "explicit_unresolved_eligibility_questions", "citations", "script_review_inventory", "native_terminal_reference_integrity"}
    pick(data, allowed)
    result = {k: scalar(v) for k, v in data.items() if k in SCALARS}
    for k in IDENTITIES + POLICIES:
        if k not in data:
            raise ValueError("missing identity/policy")
        if k.endswith("sha256"):
            sha(data[k])
    if data["case_id"] not in CASES or data["arm"] not in ARMS or data["run_id"] != f"v2_{data['case_id']}_{data['arm']}":
        raise ValueError("invalid experiment identity")
    if data["outcome_blinded"] is not True or type(data["protocol_eligible"]) is not bool:
        raise ValueError("not an explicit blinded decision")
    if data["compliance_status"] not in COMPLIANCE_STATUSES:
        raise ValueError("unknown compliance status")
    if set(data["evidence_sha256"]) != AUTHOR_HASHES:
        raise ValueError("not exactly nine AUTHOR references")
    result["evidence_sha256"] = {k: sha(v) for k, v in data["evidence_sha256"].items()}
    for k, keys in MAP_FIELDS.items():
        if k in data:
            result[k] = flat(data[k], keys.split())
    for key in ("limitations", "explicit_unresolved_eligibility_questions"):
        if key in data:
            if not isinstance(data[key], list) or any(not isinstance(x, str) for x in data[key]):
                raise ValueError("invalid reviewer prose")
            result[key] = list(data[key])
    value = data["treatment_findings"]
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        result["treatment_findings"] = list(value)
    else:
        result["treatment_findings"] = flat(value, "author_claimed_accepted independent_physical_acceptance scope status".split())
    if isinstance(data["citations"], list):
        result["citations"] = [citation(x) for x in data["citations"]]
    else:
        parts = pick(data["citations"], "author_archive review_package mechanical_review static_route_review original_capture_calls retained_script_members native_or_author_record_members".split())
        result["citations"] = {}
        for key, value in parts.items():
            if key in ("author_archive", "review_package", "mechanical_review", "static_route_review"):
                result["citations"][key] = flat(value, ("sha256", "bytes"), ("path",))
            else:
                keys, dropped = {
                    "original_capture_calls": (CAPTURE_CITATION, ("call_id", "observed_call_created_unix", "tool_name")),
                    "retained_script_members": (SCRIPT_CITATION, ()),
                    "native_or_author_record_members": (RECORD_CITATION, ()),
                }[key]
                result["citations"][key] = [flat(x, keys, dropped) for x in value]
    if "script_review_inventory" in data:
        result["script_review_inventory"] = [flat(x, SCRIPT_INVENTORY) for x in data["script_review_inventory"]]
    if "native_terminal_reference_integrity" in data:
        result["native_terminal_reference_integrity"] = flat(data["native_terminal_reference_integrity"], "all_match reference_count sha256".split(), ("path",))
    result["publication"] = {
        "schema_version": "sealed-adjudication-public-projection.v1",
        "projection": True, "private_original_provided": False,
        "original_adjudication_sha256": sha(original_sha), "original_seal_sha256": sha(seal_sha),
        "reviewer_role": "automated independent outcome-blinded protocol reviewer",
        "citation_status": "attestations_to_private_original_bytes_not_supplied_evidence",
        "scope": BOUNDARY,
        "omitted_categories": ["private filesystem paths", "individual reviewer/model/session/request/call identities", "call timestamps and tool labels", "raw captures, scripts, tool/model content and credentials"],
    }
    if "ui_evidence_boundary" in data:
        result["publication"]["ui_diagnostic_scope"] = "posthoc_explanation_only_no_eligibility_change"
    privacy_check(result)
    return result


def load_sealed(adjudications, seal_path, expected_seal):
    seal_path = pathlib.Path(seal_path)
    seal_bytes = seal_path.read_bytes()
    seal_sha = digest_bytes(seal_bytes)
    if seal_sha != sha(expected_seal):
        raise ValueError("seal digest mismatch")
    seal = json.loads(seal_bytes)
    if seal.get("all_twenty_audits_validated") is not True or seal.get("independent_outcomes_not_opened_by_root_or_supplied_to_reviewers_before_seal") is not True:
        raise ValueError("seal lacks outcome-blind validation")
    expected = {f"v2_{case}_{arm}" for case in CASES for arm in ARMS}
    if len(seal["runs"]) != 20 or {r["run_id"] for r in seal["runs"]} != expected:
        raise ValueError("seal must cover all twenty runs exactly")
    records = []
    for row in sorted(seal["runs"], key=lambda r: r["run_id"]):
        # Deliberately ignore the seal's private absolute locator.
        p = pathlib.Path(adjudications) / (row["run_id"] + ".json")
        if p.is_symlink():
            raise ValueError("adjudication symlinks are not permitted")
        raw = p.read_bytes()
        if digest_bytes(raw) != sha(row["adjudication"]["sha256"]):
            raise ValueError("adjudication digest mismatch")
        data = json.loads(raw)
        if any(data[k] != row[k] for k in ("run_id", "protocol_eligible", "compliance_status")):
            raise ValueError("sealed decision mismatch")
        records.append((data, row["adjudication"]["sha256"]))
    return seal, seal_sha, records


def build(adjudications, seal_path, output, expected_seal=EXPECTED_SEAL):
    output = pathlib.Path(output)
    if output.exists():
        raise FileExistsError("output must be a new directory")
    seal, seal_sha, records = load_sealed(adjudications, seal_path, expected_seal)
    files = {}
    rows = []
    for data, original_sha in records:
        name = "audits/" + data["run_id"] + ".json"
        files[name] = encoded(project(data, original_sha, seal_sha))
        rows.append({"run_id": data["run_id"], "protocol_eligible": data["protocol_eligible"],
                     "compliance_status": data["compliance_status"], "public_file": name,
                     "published_sha256": digest_bytes(files[name]), "original_retained_sha256": original_sha,
                     "projection": True, "original_provided": False})
    eligible_pairs = sum(all(d["protocol_eligible"] for d, _ in records if d["case_id"] == case) for case in CASES)
    summary = {"schema_version": "sealed-audit-companion.v1", "sealed_at": seal["sealed_at"],
               "original_seal_sha256": seal_sha, "original_seal_provided": False,
               "projection": True, "outcome_blinded": True, "scope": BOUNDARY,
               "run_count": 20, "eligible_run_count": sum(d["protocol_eligible"] for d, _ in records),
               "eligible_pair_count": eligible_pairs, "physical_outcomes": "NOT_ACCESSED", "runs": rows}
    privacy_check(summary)
    files["index.json"] = encoded(summary)
    from check_preservation import compare_records
    files["preservation_checks.json"] = encoded(compare_records(
        records, [json.loads(files[row["public_file"]]) for row in rows], seal_sha))
    files["README.md"] = README.encode()
    for name in ("project_audits.py", "verify_public.py", "check_preservation.py", "test_projection.py"):
        files["tools/" + name] = pathlib.Path(__file__).with_name(name).read_bytes()
    manifest = {"schema_version": "public-audit-companion-manifest.v1", "scope": BOUNDARY,
                "original_seal_sha256": seal_sha,
                "files": [{"relative_path": name, "sha256": digest_bytes(raw), "bytes": len(raw)} for name, raw in sorted(files.items())]}
    files["manifest.json"] = encoded(manifest)
    # All source parsing/privacy validation precedes the first output mutation.
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in sorted(files.items()):
        p = output / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("xb") as f:
            f.write(raw)
    return {"manifest_sha256": digest(output / "manifest.json"), "files_in_manifest": len(manifest["files"]),
            "run_count": 20, "eligible_run_count": summary["eligible_run_count"], "eligible_pair_count": eligible_pairs}


README = """# Sealed outcome-blinded protocol audits

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
python3 tools/project_audits.py --adjudications PRIVATE_ADJUDICATIONS \\
  --seal PRIVATE_SEAL --output NEW_PUBLIC_DIRECTORY
python3 tools/check_preservation.py --adjudications PRIVATE_ADJUDICATIONS \\
  --seal PRIVATE_SEAL --bundle NEW_PUBLIC_DIRECTORY --output NEW_CHECK_JSON
```

The separately retained preservation receipt is an attestation from the
publisher's private-original comparison. A public-only check must not relabel
it as verification of unavailable original evidence.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudications", required=True)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-seal-sha256", default=EXPECTED_SEAL)
    args = parser.parse_args()
    print(json.dumps(build(args.adjudications, args.seal, args.output, args.expected_seal_sha256), sort_keys=True))
