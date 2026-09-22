"""Provider-free negative controls; synthetic fixtures only, no native launches."""

import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from build_public_bundle import Builder, GITATTRIBUTES, FINAL_ASSESSMENT_SCHEMA
from verify_public_bundle import verify


def digest(data):
    return hashlib.sha256(data).hexdigest()


class PublicationChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.rows = []
        for name, value in [
            ("source_clear_freeze.json", {"files_sha256": {}}),
            ("original_drawer_freeze.json", {"files": {}}),
        ]:
            self.add("evaluator/source_clear_v1/" + name, json.dumps(value).encode())
        self.add(".gitattributes", GITATTRIBUTES.encode())
        self.add("evidence.json", b'{"status":"fail"}\n')
        self.save()

    def add(self, rel, data, mode="exact", original=None):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        self.rows.append(
            {
                "path": rel,
                "published_sha256": digest(data),
                "published_bytes": len(data),
                "mode": mode,
                "original_retained_sha256": original or digest(data),
            }
        )

    def save(self):
        (self.root / "publication_manifest.json").write_text(
            json.dumps({"files": self.rows})
        )

    def errors(self):
        return {e["error"] for e in verify(self.root)["errors"]}

    def test_exact_positive_and_byte_preserving_attributes(self):
        self.assertTrue(verify(self.root)["passed"])
        self.assertIn("* -text\n", GITATTRIBUTES)
        for suffix in ["png", "jpg", "usdc", "usdz", "gz"]:
            self.assertIn(f"*.{suffix} binary", GITATTRIBUTES)

    def test_changed_evidence_rejected(self):
        (self.root / "evidence.json").write_bytes(b'{"status":"pass"}\n')
        self.assertIn("published_digest_or_size_mismatch", self.errors())

    def test_false_exact_label_rejected(self):
        self.rows[-1]["original_retained_sha256"] = "0" * 64
        self.save()
        self.assertIn("false_exact_label", self.errors())

    def test_projection_is_not_claimed_exact(self):
        old_digest = "1" * 64
        self.rows[-1].update(
            mode="field_projection", original_retained_sha256=old_digest
        )
        self.add(
            "reference.json",
            json.dumps({"path": "evidence.json", "sha256": old_digest}).encode(),
        )
        self.save()
        result = verify(self.root)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["reference_bindings"][0]["binding"],
            "ORIGINAL_RETAINED_BYTES_TARGET_IS_PROJECTION",
        )

    def test_declared_missing_intermediate_is_not_exact_verification(self):
        old_digest = "a" * 64
        self.add(
            "reference.json",
            json.dumps({"path": "evidence.json", "sha256": old_digest}).encode(),
        )
        self.rows[-1]["unprovided_intermediate_references"] = [
            {"path": "evidence.json", "sha256": old_digest, "bytes_provided": False}
        ]
        self.save()
        result = verify(self.root)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["reference_bindings"][0]["binding"],
            "RECEIPT_ATTESTED_INTERMEDIATE_DIGEST_BYTES_NOT_PROVIDED",
        )
        self.rows[-1]["unprovided_intermediate_references"][0]["sha256"] = "b" * 64
        self.save()
        self.assertIn("reference_digest_mismatch", self.errors())

    def test_forged_reference_rejected(self):
        self.add(
            "reference.json",
            json.dumps({"path": "evidence.json", "sha256": "f" * 64}).encode(),
        )
        self.save()
        self.assertIn("reference_digest_mismatch", self.errors())

    def test_manifest_escape_and_duplicate_rejected(self):
        duplicate = copy.deepcopy(self.rows[-1])
        self.rows.append(duplicate)
        escaped = copy.deepcopy(duplicate)
        escaped["path"] = "../not-readable.json"
        self.rows.append(escaped)
        self.save()
        self.assertTrue(
            {"unsafe_manifest_path", "duplicate_manifest_path"} <= self.errors()
        )

    def test_unmanifested_and_private_field_rejected(self):
        self.add("bad.json", b'{"api_key":"synthetic-negative-control"}')
        (self.root / "unlisted.txt").write_text("synthetic")
        self.save()
        self.assertTrue(
            {"disallowed_private_field", "unmanifested_files"} <= self.errors()
        )

    def test_source_freeze_mismatch_rejected(self):
        p = self.root / "evaluator/source_clear_v1/source_clear_freeze.json"
        data = json.dumps({"files_sha256": {"absent.py": "0" * 64}}).encode()
        p.write_bytes(data)
        self.rows[0].update(
            published_sha256=digest(data),
            published_bytes=len(data),
            original_retained_sha256=digest(data),
        )
        self.save()
        self.assertIn("frozen_reference_digest_mismatch", self.errors())

    def test_full_gzip_hash_and_privacy_checked_after_decode(self):
        raw = b'{"phase":"open","position":0.2}\n'
        self.add("trace.jsonl.gz", gzip.compress(raw, mtime=0))
        self.rows[-1]["compression"] = {
            "uncompressed_sha256": digest(raw),
            "uncompressed_bytes": len(raw),
        }
        self.save()
        self.assertTrue(verify(self.root)["passed"])
        self.rows[-1]["compression"]["uncompressed_sha256"] = "0" * 64
        self.save()
        self.assertIn("uncompressed_digest_or_size_mismatch", self.errors())

    def test_qualification_relative_binding_resolves_published_artifact(self):
        self.add("nested/bound.json", b'{"status":"fail"}\n')
        self.add(
            "nested/qualification.json",
            json.dumps(
                {
                    "path": "bound.json",
                    "sha256": digest(b'{"status":"fail"}\n'),
                }
            ).encode(),
        )
        self.save()
        result = verify(self.root)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(
            result["reference_bindings"][0]["binding"], "EXACT_PUBLISHED_BYTES"
        )

    def test_explicit_qualification_reference_base_avoids_root_name_collision(self):
        raw = b'{"status":"nested"}\n'
        self.add("nested/evidence.json", raw)
        self.add(
            "nested/qualification.json",
            json.dumps({"path": "evidence.json", "sha256": digest(raw)}).encode(),
        )
        self.rows[-1]["reference_base"] = "nested"
        self.save()
        result = verify(self.root)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(
            result["reference_bindings"][0]["binding"], "EXACT_PUBLISHED_BYTES"
        )

    def test_historical_exact_reference_redirect_is_hash_checked(self):
        old = b'{"status":"historical"}\n'
        self.add("history/evidence.json", old)
        self.add(
            "history/qualification.json",
            json.dumps({"path": "evidence.json", "sha256": digest(old)}).encode(),
        )
        self.rows[-1]["reference_redirects"] = [
            {
                "path": "evidence.json",
                "sha256": digest(old),
                "published_path": "history/evidence.json",
            }
        ]
        self.save()
        result = verify(self.root)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(
            result["reference_bindings"][0]["binding"],
            "EXACT_PUBLISHED_HISTORICAL_BYTES",
        )
        self.rows[-1]["reference_redirects"][0]["published_path"] = "evidence.json"
        self.save()
        self.assertIn("historical_redirect_digest_mismatch", self.errors())

    def test_nested_workflow_session_is_projected_and_detected(self):
        b = Builder(self.root, self.root / "unused", self.root)
        private = "workflow-" + "0123456789abcdefabcd"
        value = {
            "camera": {
                "_workflow_command": {"argv": ["--session", private]},
                "opaque": private,
            }
        }
        projected = b.project(value, True)
        self.assertNotIn("_workflow_command", projected["camera"])
        self.assertEqual(projected["camera"]["opaque"], "[private identifier removed]")
        self.add("private_workflow.json", json.dumps({"opaque": private}).encode())
        self.save()
        self.assertIn("private_identifier_pattern", self.errors())

    def test_reviewed_json_preserves_safe_exact_native_receipt(self):
        cap = self.root / "capstone"
        cap.mkdir()
        raw = b'{"status":"accepted","endpoint_state":"static_mesh","path":"/opt/astra-content-value-20260921/capstone/asset.usdz"}\n'
        (cap / "terminal.json").write_bytes(raw)
        b = Builder(cap, self.root / "published", self.root)
        b.reviewed_json("terminal.json")
        self.assertEqual((b.output / "terminal.json").read_bytes(), raw)
        self.assertEqual(b.records[0]["mode"], "exact")
        self.assertEqual(b.records[0]["original_retained_sha256"], digest(raw))

    def test_reviewed_json_projects_private_metadata_and_preserves_original_hash(self):
        cap = self.root / "capstone"
        cap.mkdir()
        raw = b'{"status":"accepted","session_id":"private","service_endpoint":"private","endpoint_state":"static_mesh"}\n'
        (cap / "terminal.json").write_bytes(raw)
        b = Builder(cap, self.root / "published", self.root)
        b.reviewed_json("terminal.json")
        value = json.loads((b.output / "terminal.json").read_text())
        self.assertEqual(value["endpoint_state"], "static_mesh")
        self.assertNotIn("session_id", value)
        self.assertNotIn("service_endpoint", value)
        self.assertEqual(b.records[0]["mode"], "field_projection")
        self.assertEqual(b.records[0]["original_retained_sha256"], digest(raw))
        self.assertNotEqual(b.records[0]["published_sha256"], digest(raw))

    def test_generic_original_locator_is_bound_to_actual_published_bytes(self):
        self.add(
            "reference.json",
            json.dumps(
                {
                    "path": "/opt/astra-content-value-20260921/capstone/evidence.json",
                    "sha256": digest(b'{"status":"fail"}\n'),
                    "endpoint_state": "static_mesh",
                }
            ).encode(),
        )
        self.save()
        result = verify(self.root)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(
            result["reference_bindings"][0]["binding"], "EXACT_PUBLISHED_BYTES"
        )
        (self.root / "evidence.json").write_bytes(b'{"status":"pass"}\n')
        self.assertIn("published_digest_or_size_mismatch", self.errors())

    def final_record(self, value):
        self.add("assessment.json", json.dumps(value).encode())
        self.rows[-1]["public_typed_final_assessment_rationale_paths"] = [
            "/gates/0/rationale"
        ]
        self.save()

    def test_reviewed_typed_final_assessment_keeps_only_declared_explanations(self):
        value = {
            "schema_version": FINAL_ASSESSMENT_SCHEMA,
            "gates": [{"rationale": "Measured evidence supports this final gate."}],
        }
        self.final_record(value)
        self.assertTrue(verify(self.root)["passed"])
        with tempfile.TemporaryDirectory() as temporary:
            cap = Path(temporary) / "capstone"
            cap.mkdir()
            raw = json.dumps(value).encode()
            (cap / "assessment.json").write_bytes(raw)
            b = Builder(cap, Path(temporary) / "published", cap)
            b.reviewed_json("assessment.json", typed_final_assessment=True)
            self.assertEqual((b.output / "assessment.json").read_bytes(), raw)
            self.assertEqual(b.records[0]["mode"], "exact")
            self.assertEqual(
                b.records[0]["public_typed_final_assessment_rationale_paths"],
                ["/gates/0/rationale"],
            )

    def test_wrong_schema_cannot_opt_into_final_explanations(self):
        self.final_record(
            {"schema_version": "raw.untrusted", "gates": [{"rationale": "private"}]}
        )
        self.assertIn("invalid_final_assessment_privacy_scope", self.errors())
        b = Builder(self.root, self.root / "unused", self.root)
        with self.assertRaises(ValueError):
            b.reviewed_json("assessment.json", typed_final_assessment=True)

    def test_ordinary_and_nested_rationale_remain_private(self):
        self.final_record(
            {
                "schema_version": FINAL_ASSESSMENT_SCHEMA,
                "gates": [
                    {
                        "rationale": "Final explanation",
                        "nested": {"rationale": "private"},
                    }
                ],
                "rationale": "private",
            }
        )
        self.assertIn("disallowed_private_field", self.errors())
        self.rows[-1]["public_typed_final_assessment_rationale_paths"].append(
            "/rationale"
        )
        self.save()
        self.assertIn("disallowed_private_field", self.errors())

    def test_private_identifiers_inside_final_explanation_are_rejected(self):
        private = "workflow-" + "0123456789abcdefabcd"
        self.final_record(
            {
                "schema_version": FINAL_ASSESSMENT_SCHEMA,
                "gates": [{"rationale": private}],
            }
        )
        self.assertIn("private_identifier_pattern", self.errors())

    def test_configuration_presence_boolean_is_safe_but_string_is_not(self):
        self.add("boolean.json", b'{"base_url_configured":true}')
        self.save()
        self.assertTrue(verify(self.root)["passed"])
        self.add(
            "string.json", b'{"base_url_configured":"https://example.invalid/private"}'
        )
        self.save()
        self.assertIn("disallowed_private_field", self.errors())
        b = Builder(self.root, self.root / "unused", self.root)
        self.assertEqual(
            b.clean({"base_url_configured": True}), {"base_url_configured": True}
        )
        self.assertEqual(b.clean({"base_url_configured": "private"}), {})

    def test_schema_property_definition_does_not_allow_assessment_values(self):
        self.add("schema.json", b'{"properties":{"rationale":{"type":"string"}}}')
        self.rows[-1]["public_json_schema_definition"] = True
        self.save()
        self.assertTrue(verify(self.root)["passed"])
        self.add("raw.json", b'{"rationale":"private"}')
        self.rows[-1]["public_json_schema_definition"] = True
        self.save()
        self.assertIn("disallowed_private_field", self.errors())

    def test_explicit_projection_removes_private_fields(self):
        b = Builder(self.root, self.root / "unused", self.root)
        value = {
            "status": "fail",
            "session_id": "synthetic",
            "analysis": "not-public",
            "nested": {"credentials": "synthetic", "measured": 3},
        }
        projected = b.project(value, {"status": True, "nested": True})
        self.assertEqual(projected, {"status": "fail", "nested": {"measured": 3}})


if __name__ == "__main__":
    unittest.main()
