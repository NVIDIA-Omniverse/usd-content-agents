"""Synthetic only: no author archives, private real records or model calls."""
import copy
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import project_audits as p
import check_preservation as c
import verify_public as v

H = "a" * 64


def fixture(case="01_drawer", arm="plain_astra", peer=False):
    d = dict(schema_version="synthetic-original.v1", run_id=f"v2_{case}_{arm}", case_id=case,
             arm=arm, lane_id=0, protocol_sha256=H, task_sha256=H, input_sha256=H, source_sha256=H,
             analysis_plan_sha256=H, treatment_adjudication_sha256=H, outcome_blinded=True,
             protocol_eligible=True, compliance_status="observed_compliance", reason="Native failure is retained.",
             evidence_sha256={k: H for k in p.AUTHOR_HASHES}, reviewer={"agent": "/private/reviewer", "model": "private-model-instance"},
             completed_utc="2026-09-23T04:00:00Z", limitations=["Physical acceptance was not accessed."],
             terminal_billing_complete=False, model_request_count=12,
             author_claim_only={"claimed_accepted": None, "declared_repairs_used": None, "physical_acceptance": "NOT_ASSESSED", "task_outcome": "incomplete"})
    if peer:
        d["mechanical_findings"] = {"billing_complete": False, "ui_association_complete": True, "admitted_requests": 12, "terminal_billing_records": 11}
        d["treatment_findings"] = {"status": "observed_compliance", "scope": "Reached stages only.", "author_claimed_accepted": False, "independent_physical_acceptance": "not_accessed"}
        d["citations"] = [{"kind": "original_tool_call", "archive_member": "private/session_captures/capture_000000_000.jsonl", "archive_sha256": H,
                           "member_sha256": H, "line": 12, "navigation_line": 14, "call_id": "call_secret12345", "observation": "Native invocation."},
                          {"kind": "retained_author_artifact", "archive_member": "workspace/native/receipt.json", "member_sha256": H,
                           "observed_fields": {"status": "failed", "success": False, "returncode": 1, "gate_dispositions": {"runtime_validation": "fail"}}}]
        d["script_review_inventory"] = [{"archive_member": "workspace/check.py", "sha256": H, "line_count": 31, "review_scope": "Static only."}]
    else:
        d["mechanical_checks"] = {"source_unchanged": True, "ui_complete": True}
        d["author_budget_observation"] = {"elapsed_seconds": 2399.125, "timed_out": False, "namespace_init_exited": True, "cgroup_populated": 0}
        d["treatment_findings"] = ["Native attempted; blocked later phase retained."]
        d["citations"] = {"author_archive": {"path": "/private/home/archive.tgz", "sha256": H, "bytes": 5321},
                          "original_capture_calls": [{"kind": "retained_tool_call", "capture_archive_member": "private/session_captures/capture_000000_000.jsonl",
                                                       "capture_sha256": H, "capture_line": 17, "navigation_ordinal": 9, "call_input_sha256": H,
                                                       "call_id": "call_secret12345", "observed_call_created_unix": 100000.125, "tool_name": "exec"}],
                          "retained_script_members": [{"kind": "retained_script", "archive_member": "workspace/main.py", "sha256": H, "reviewed_lines": [1, 40], "scope": "Static only."}]}
    return d


def twenty():
    result = []
    for i, case in enumerate(p.CASES):
        for arm in p.ARMS:
            d = fixture(case, arm, peer=bool(i % 2))
            if case == "09_printer" and arm == "plain_astra":
                d.update(protocol_eligible=False, compliance_status="insufficient_evidence")
                d["ui_evidence_boundary"] = {"separator": "U+0085", "frozen_ui_complete": False, "request_count": 288, "associated_requests": 288,
                                              "archive_member": "private/session_captures/capture_000001_000.jsonl", "capture_sha256": H,
                                              "lf_record_line": 117, "parser_lines": [79, 83], "disposition": "No posthoc eligibility promotion."}
            result.append(d)
    return result


class ProjectionTests(unittest.TestCase):
    def projected(self, d=None):
        return p.project(d or fixture(), H, p.EXPECTED_SEAL)

    def test_private_id_path_reviewer_removed_hashes_preserved(self):
        d = self.projected()
        self.assertNotIn("reviewer", d)
        self.assertNotIn("path", d["citations"]["author_archive"])
        self.assertEqual(d["citations"]["author_archive"]["sha256"], H)
        x = d["citations"]["original_capture_calls"][0]
        self.assertNotIn("call_id", x)
        self.assertNotIn("observed_call_created_unix", x)
        self.assertEqual(x["capture_line"], 17)
        self.assertFalse(d["publication"]["private_original_provided"])

    def test_other_reviewer_shape_native_failure_retained(self):
        d = self.projected(fixture(peer=True))
        self.assertNotIn("call_id", d["citations"][0])
        self.assertEqual(d["citations"][1]["observed_fields"]["status"], "failed")
        self.assertEqual(d["citations"][1]["observed_fields"]["returncode"], 1)

    def test_missing_billing_does_not_exclude_or_fill_null_claim(self):
        d = self.projected()
        self.assertTrue(d["protocol_eligible"])
        self.assertFalse(d["terminal_billing_complete"])
        self.assertIsNone(d["author_claim_only"]["claimed_accepted"])

    def test_frozen_noncompliance_preserved_without_dropping_row(self):
        ds = twenty()
        ds[0].update(protocol_eligible=False, compliance_status="observed_noncompliance",
                     reason="Synthetic prohibited workflow invocation was observed.")
        outs = [self.projected(d) for d in ds]
        result = c.compare_records([(d, H) for d in ds], outs, p.EXPECTED_SEAL)
        self.assertEqual(result["records_checked"], 20)
        self.assertEqual(result["eligible_runs"], 18)
        self.assertEqual(outs[0]["compliance_status"], "observed_noncompliance")
        self.assertFalse(outs[0]["protocol_eligible"])
        self.assertEqual(outs[0]["reason"], ds[0]["reason"])

    def test_invented_violation_enum_rejected(self):
        d = fixture(); d.update(protocol_eligible=False, compliance_status="observed_violation")
        with self.assertRaises(ValueError): self.projected(d)

    def test_printer_frozen_false_and_posthoc_scope(self):
        d = next(x for x in twenty() if x["run_id"] == "v2_09_printer_plain_astra")
        out = self.projected(d)
        self.assertFalse(out["protocol_eligible"])
        self.assertEqual(out["ui_evidence_boundary"], d["ui_evidence_boundary"])
        self.assertEqual(out["publication"]["ui_diagnostic_scope"], "posthoc_explanation_only_no_eligibility_change")

    def test_unknown_raw_content_field_rejected(self):
        d = fixture(); d["raw_capture"] = "synthetic private content"
        with self.assertRaises(ValueError): self.projected(d)

    def test_secret_in_allowed_reason_rejected(self):
        for value in ("Bearer syntheticsecret", "https://private.example", "/Users/example/private", "resp_synthetic12345"):
            d = fixture(); d["reason"] = value
            with self.subTest(value=value), self.assertRaises(ValueError): self.projected(d)

    def test_unsafe_member_paths_and_identifiers_rejected(self):
        for value in ("../secret", "/absolute/member", "workspace/a/../b", "workspace/12345678-abcd-abcd-abcd-123456789012.py"):
            d = fixture(); d["citations"]["retained_script_members"][0]["archive_member"] = value
            with self.subTest(value=value), self.assertRaises(ValueError): self.projected(d)

    def test_unknown_nested_field_rejected(self):
        d = fixture(peer=True); d["citations"][1]["observed_fields"]["model_output"] = "private"
        with self.assertRaises(ValueError): self.projected(d)

    def test_nan_rejected(self):
        d = fixture(); d["author_budget_observation"]["elapsed_seconds"] = float("nan")
        with self.assertRaises(ValueError): self.projected(d)

    def test_exact_nine_hashes_required(self):
        d = fixture(); d["evidence_sha256"].pop("usage")
        with self.assertRaises(ValueError): self.projected(d)

    def test_independent_leaf_preservation_both_shapes(self):
        ds = twenty(); outs = [self.projected(d) for d in ds]
        result = c.compare_records([(d, H) for d in ds], outs, p.EXPECTED_SEAL)
        self.assertEqual(result["records_checked"], 20)
        self.assertEqual(result["eligible_runs"], 19)
        self.assertGreater(result["runs"][0]["preserved_null_leaves"], 0)

    def test_independent_checker_rejects_status_numeric_null_changes(self):
        for field, value in (("protocol_eligible", False), ("model_request_count", 13), ("reason", "changed")):
            ds = twenty(); outs = [self.projected(d) for d in ds]; outs[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): c.compare_records([(d, H) for d in ds], outs, p.EXPECTED_SEAL)
        ds = twenty(); outs = [self.projected(d) for d in ds]; outs[0]["author_claim_only"]["claimed_accepted"] = False
        with self.assertRaises(ValueError): c.compare_records([(d, H) for d in ds], outs, p.EXPECTED_SEAL)

    def test_independent_checker_rejects_missing_row(self):
        ds = twenty()
        with self.assertRaises(ValueError): c.compare_records([(d, H) for d in ds], [self.projected(d) for d in ds[:-1]], p.EXPECTED_SEAL)

    def write_sources(self, root):
        source = root / "originals"; source.mkdir()
        rows = []
        for d in twenty():
            raw = p.encoded(d); (source / (d["run_id"] + ".json")).write_bytes(raw)
            rows.append({"run_id": d["run_id"], "protocol_eligible": d["protocol_eligible"], "compliance_status": d["compliance_status"],
                         "adjudication": {"path": "/never/open/this/path", "sha256": p.digest_bytes(raw)}})
        seal = root / "seal.json"
        seal.write_bytes(p.encoded({"all_twenty_audits_validated": True, "independent_outcomes_not_opened_by_root_or_supplied_to_reviewers_before_seal": True,
                                   "sealed_at": "2026-09-23T04:44:27Z", "runs": rows}))
        return source, seal

    def test_complete_create_only_roundtrip_and_public_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); source, seal = self.write_sources(root); output = root / "public"
            result = p.build(source, seal, output, p.digest(seal))
            self.assertEqual(result["eligible_pair_count"], 9)
            with patch.object(v, "EXPECTED_SEAL", p.digest(seal)):
                self.assertEqual(v.verify(output)["eligible_runs"], 19)
            checked = c.check(source, seal, output)
            self.assertEqual(checked, json.loads((output / "preservation_checks.json").read_bytes()))
            original_manifest = (output / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError): p.build(source, seal, output, p.digest(seal))
            self.assertEqual(original_manifest, (output / "manifest.json").read_bytes())

    def test_stale_seal_or_original_refused_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); source, seal = self.write_sources(root)
            with self.assertRaises(ValueError): p.build(source, seal, root / "new1", H)
            self.assertFalse((root / "new1").exists())
            next(source.glob("*.json")).write_text("{}")
            with self.assertRaises(ValueError): p.build(source, seal, root / "new2", p.digest(seal))
            self.assertFalse((root / "new2").exists())

    def test_public_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); source, seal = self.write_sources(root); output = root / "public"
            p.build(source, seal, output, p.digest(seal))
            next((output / "audits").glob("*.json")).write_text("{}")
            with patch.object(v, "EXPECTED_SEAL", p.digest(seal)), self.assertRaises(ValueError): v.verify(output)


if __name__ == "__main__":
    unittest.main()
