"""Reconstruct public original inputs and rerun the source-only fixture builder."""

import argparse, hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
bundle = a.bundle.resolve()
output = a.output.resolve()
assert not output.exists()
fixture = bundle / "evaluator/source_clear_v1"
historical = json.loads((fixture / "original_drawer_freeze.json").read_text())
historical.pop("_publication", None)
with tempfile.TemporaryDirectory(prefix="public-source-input-") as tmp:
    original = Path(tmp)
    for name, expected in historical["files"].items():
        source = fixture / (
            "original_spec/drawer_acceptance.json"
            if name == "drawer_acceptance.json"
            else name
        )
        assert sha(source) == expected, (name, "original input hash mismatch")
        destination = original / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (original / "drawer_freeze.json").write_text(
        json.dumps(historical, indent=2) + "\n"
    )
    subprocess.run(
        [
            sys.executable,
            str(bundle / "prepare_source_clear_fixture.py"),
            "--input-evaluator",
            str(original),
            "--output",
            str(output),
        ],
        check=True,
    )
expected = json.loads((fixture / "source_clear_freeze.json").read_text())
checks = {
    name: sha(output / name) == digest
    for name, digest in expected["files_sha256"].items()
    if name != "original_drawer_freeze.json"
}
result = {
    "schema_version": "public-source-clear-rebuild.v1",
    "source_and_spec_and_evaluator_bytes_match": all(checks.values()),
    "file_checks": checks,
    "original_spec_sha256": sha(fixture / "original_spec/drawer_acceptance.json"),
    "rebuilt_spec_sha256": sha(output / "drawer_acceptance.json"),
    "original_recorded_freeze_sha256": sha(fixture / "source_clear_freeze.json"),
    "rebuilt_freeze_sha256": sha(output / "source_clear_freeze.json"),
    "overall_freeze_equality_claimed": False,
    "historical_receipt_exact_bytes_provided": False,
    "scope": "Source-only regeneration; timestamps/historical receipt projection alter freeze bytes. No native simulation or acceptance.",
}
(output / "public_rebuild_review.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(result, indent=2))
raise SystemExit(0 if all(checks.values()) else 1)
