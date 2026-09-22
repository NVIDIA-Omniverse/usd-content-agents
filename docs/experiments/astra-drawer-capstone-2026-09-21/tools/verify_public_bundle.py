"""Read-only public manifest, privacy, links and optional decoded USD review."""

from __future__ import annotations
import argparse, hashlib, json, re, zipfile, sys, gzip

sys.dont_write_bytecode = True
from pathlib import Path
from urllib.parse import unquote
from build_public_bundle import (
    public_boolean_flag,
    FINAL_ASSESSMENT_SCHEMA,
    PRIVATE_VALUE,
    PRIVATE_KEY,
    BANNED_NAME,
    private_key,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(root, decode_usd=False):
    manifest = json.loads((root / "publication_manifest.json").read_text())
    records = {}
    errors = []
    usd = []
    images = []
    references = []
    for record in manifest["files"]:
        rel = record["path"]
        path = root / rel
        if (
            Path(rel).is_absolute()
            or ".." in Path(rel).parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
        ):
            errors.append({"path": rel, "error": "unsafe_manifest_path"})
            continue
        if rel in records:
            errors.append({"path": rel, "error": "duplicate_manifest_path"})
            continue
        records[rel] = record

    def check_text(text, rel):
        # Cheap necessary substrings avoid expensive domain-regex searches over
        # hundreds of MB of numeric machine traces; the UUID alternative stays exact.
        lowered = text.lower()
        candidates = (
            "/users/",
            "/home/",
            "teleport.",
            ".horde",
            "astra-value-pair",
            "codex-2gpu",
            "bearer",
            "-----begin",
            "workflow-",
        )
        has_named_pattern = any(marker in lowered for marker in candidates)
        has_uuid = (
            re.search(
                r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                text,
            )
            is not None
        )
        if has_uuid or (has_named_pattern and PRIVATE_VALUE.search(text)):
            errors.append({"path": rel, "error": "private_identifier_pattern"})

    def scan_references(value, source, pointer=""):
        if isinstance(value, dict):
            allowed_final = records[source].get(
                "public_typed_final_assessment_rationale_paths", []
            )
            if (
                pointer == ""
                and allowed_final
                and value.get("schema_version") != FINAL_ASSESSMENT_SCHEMA
            ):
                errors.append(
                    {"path": source, "error": "invalid_final_assessment_privacy_scope"}
                )
            for key, child in value.items():
                child_pointer = pointer + "/" + key
                final_explanation = (
                    child_pointer in allowed_final
                    and isinstance(child, str)
                    and re.fullmatch(
                        r"/(?:gates|findings)/[0-9]+/rationale", child_pointer
                    )
                )
                schema_definition = (
                    records[source].get("public_json_schema_definition") is True
                    and child_pointer.endswith("/properties/rationale")
                    and isinstance(child, dict)
                    and child.get("type") == "string"
                )
                if (
                    private_key(key)
                    and not public_boolean_flag(key, child)
                    and not final_explanation
                    and not schema_definition
                    and child not in (None, "", [], {})
                ):
                    errors.append(
                        {
                            "path": source,
                            "error": "disallowed_private_field",
                            "field": key,
                        }
                    )
                scan_references(child, source, child_pointer)
            if isinstance(value.get("path"), str) and re.fullmatch(
                "[0-9a-f]{64}", str(value.get("sha256", ""))
            ):
                path = value["path"]
                digest = value["sha256"]
                redirect = next(
                    (
                        r
                        for r in records[source].get("reference_redirects", [])
                        if r.get("path") == path and r.get("sha256") == digest
                    ),
                    None,
                )
                base = records[source].get("reference_base")
                if base and not Path(path).is_absolute():
                    if (
                        Path(base).is_absolute()
                        or ".." in Path(base).parts
                        or ".." in Path(path).parts
                    ):
                        errors.append(
                            {"path": source, "error": "unsafe_reference_base"}
                        )
                        target = None
                    else:
                        target = records.get(str(Path(base) / path))
                else:
                    target = records.get(path)
                for prefix in (
                    "capstone/",
                    "/opt/astra-content-value-20260921/capstone/",
                ):
                    if target is None and path.startswith(prefix):
                        target = records.get(path[len(prefix) :])
                if (
                    target is None
                    and not Path(path).is_absolute()
                    and ".." not in Path(path).parts
                ):
                    target = records.get(str(Path(source).parent / path))
                if redirect:
                    redirected_path = redirect.get("published_path", "")
                    target = records.get(redirected_path)
                    if not target or target["published_sha256"] != digest:
                        errors.append(
                            {
                                "path": source,
                                "error": "historical_redirect_digest_mismatch",
                            }
                        )
                if (
                    records[source].get("reference_namespace")
                    == "portable_derivative_not_in_bundle"
                ):
                    state = "SEPARATE_PORTABLE_DERIVATIVE_BYTES_NOT_INCLUDED"
                elif target:
                    if digest == target["published_sha256"]:
                        state = (
                            "EXACT_PUBLISHED_HISTORICAL_BYTES"
                            if redirect
                            else "EXACT_PUBLISHED_BYTES"
                        )
                    elif digest == target.get("original_retained_sha256"):
                        state = "ORIGINAL_RETAINED_BYTES_TARGET_IS_PROJECTION"
                    elif any(
                        r.get("path") == path
                        and r.get("sha256") == digest
                        and r.get("bytes_provided") is False
                        for r in records[source].get(
                            "unprovided_intermediate_references", []
                        )
                    ):
                        state = (
                            "RECEIPT_ATTESTED_INTERMEDIATE_DIGEST_BYTES_NOT_PROVIDED"
                        )
                    else:
                        state = "UNMATCHED_DIGEST"
                        errors.append(
                            {
                                "path": source,
                                "error": "reference_digest_mismatch",
                                "target": path,
                            }
                        )
                else:
                    state = "RETAINED_ONLY_NOT_CLAIMED_PUBLISHED"
                references.append(
                    {
                        "source": source,
                        "target": path,
                        "sha256": digest,
                        "binding": state,
                    }
                )
        elif isinstance(value, list):
            for i, child in enumerate(value):
                scan_references(child, source, pointer + f"/{i}")

    def layer_review(path, rel):
        from pxr import Sdf, Usd  # Initialize USD file-format plugins before Sdf opens.

        paths = [str(path)]
        if path.suffix == ".usdz":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if Path(name).is_absolute() or ".." in Path(name).parts:
                        errors.append({"path": rel, "error": "unsafe_archive_member"})
                    if Path(name).suffix in {".usd", ".usda", ".usdc"}:
                        paths.append(str(path) + "[" + name + "]")
        for locator in sorted(set(paths)):
            layer = Sdf.Layer.FindOrOpen(locator)
            if not layer:
                errors.append({"path": rel, "error": "usd_layer_open_failed"})
                continue
            text = layer.ExportToString()
            check_text(text, rel)
            refs = sorted(layer.GetExternalReferences())
            usd.append(
                {
                    "path": rel,
                    "layer": Path(locator).name,
                    "decoded_sha256": sha(text.encode()),
                    "external_asset_paths": refs,
                    "absolute_locators_require_relocation": any(
                        p.startswith("/") for p in refs
                    ),
                    "privacy_scan": (
                        "PASS" if not PRIVATE_VALUE.search(text) else "FAIL"
                    ),
                }
            )

    for rel, record in records.items():
        path = root / rel
        if not path.is_file():
            errors.append({"path": rel, "error": "missing"})
            continue
        if BANNED_NAME.search(rel):
            errors.append({"path": rel, "error": "blocked_filename"})
        data = path.read_bytes()
        if (
            sha(data) != record["published_sha256"]
            or len(data) != record["published_bytes"]
        ):
            errors.append({"path": rel, "error": "published_digest_or_size_mismatch"})
        if record["mode"] == "exact" and sha(data) != record.get(
            "original_retained_sha256"
        ):
            errors.append({"path": rel, "error": "false_exact_label"})
        if (
            path.suffix.lower()
            in {
                ".json",
                ".jsonl",
                ".md",
                ".py",
                ".txt",
                ".log",
                ".patch",
                ".usda",
                ".gltf",
                ".svg",
                ".csv",
            }
            or path.name == ".gitattributes"
        ):
            text = data.decode()
            check_text(text, rel)
            if path.suffix == ".json":
                scan_references(json.loads(text), rel)
            if path.suffix == ".md":
                for link in re.findall(r"!?\[[^\]]*\]\(([^\n)]+)\)", text):
                    link = unquote(link.split(" ")[0]).strip("<>")
                    if link.startswith(("http:", "https:", "#", "mailto:")):
                        continue
                    target = (path.parent / link.split("#")[0]).resolve()
                    if not target.is_relative_to(root) or not target.exists():
                        errors.append(
                            {
                                "path": rel,
                                "error": "broken_markdown_link",
                                "target": link,
                            }
                        )
        if path.suffix == ".gz":
            compression = record.get("compression", {})
            h = hashlib.sha256()
            size = 0
            try:
                if data[4:8] != bytes(4):
                    errors.append({"path": rel, "error": "nonzero_gzip_mtime"})
                with gzip.open(path, "rb") as stream:
                    for line in stream:
                        h.update(line)
                        size += len(line)
                        check_text(line.decode(), rel)
                        json.loads(line)
                if (
                    compression.get("uncompressed_sha256") != h.hexdigest()
                    or compression.get("uncompressed_bytes") != size
                ):
                    errors.append(
                        {"path": rel, "error": "uncompressed_digest_or_size_mismatch"}
                    )
            except (OSError, EOFError, UnicodeError, ValueError) as error:
                errors.append(
                    {
                        "path": rel,
                        "error": "invalid_gzip_jsonl",
                        "error_type": type(error).__name__,
                    }
                )
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            from PIL import Image

            with Image.open(path) as im:
                im.load()
                metadata = json.dumps(im.info, default=str)
                check_text(metadata, rel)
                images.append(
                    {
                        "path": rel,
                        "size": list(im.size),
                        "metadata_scan": (
                            "PASS" if not PRIVATE_VALUE.search(metadata) else "FAIL"
                        ),
                    }
                )
        if path.suffix in {".usd", ".usda", ".usdc", ".usdz"} and decode_usd:
            layer_review(path, rel)
    for freeze_name, field in [
        ("source_clear_freeze.json", "files_sha256"),
        ("original_drawer_freeze.json", "files"),
    ]:
        freeze_rel = "evaluator/source_clear_v1/" + freeze_name
        freeze = json.loads((root / freeze_rel).read_text())
        for name, digest in freeze[field].items():
            target = "evaluator/source_clear_v1/" + (
                "original_spec/drawer_acceptance.json"
                if freeze_name == "original_drawer_freeze.json"
                and name == "drawer_acceptance.json"
                else name
            )
            record = records.get(target)
            if record and digest == record["published_sha256"]:
                state = "EXACT_PUBLISHED_BYTES"
            elif record and digest == record.get("original_retained_sha256"):
                state = "ORIGINAL_DIGEST_ATTESTED_PROJECTION_PROVIDED"
            else:
                state = "UNMATCHED_DIGEST"
                errors.append(
                    {
                        "path": freeze_rel,
                        "error": "frozen_reference_digest_mismatch",
                        "target": target,
                    }
                )
            references.append(
                {
                    "source": freeze_rel,
                    "target": target,
                    "sha256": digest,
                    "binding": state,
                }
            )
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    extras = actual - set(records) - {"publication_manifest.json"}
    if extras:
        errors.append({"error": "unmanifested_files", "paths": sorted(extras)})
    return {
        "schema_version": "capstone-public-bundle-review.v1",
        "manifest_sha256": sha((root / "publication_manifest.json").read_bytes()),
        "file_count": len(records),
        "byte_count": sum(r["published_bytes"] for r in records.values()),
        "passed": not errors,
        "decoded_usd_review_completed": decode_usd,
        "errors": errors,
        "images": images,
        "decoded_usd": usd,
        "reference_bindings": references,
        "limitations": [
            "Source references marked retained-only are not independently revalidated from absent files.",
            "Projected native receipts cannot be substituted for original native digest-bound inputs.",
            "Decoded USD privacy/dependency review is not simulation or geometry-fidelity acceptance.",
        ],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--decode-usd", action="store_true")
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    r = verify(a.root.resolve(), a.decode_usd)
    if a.report:
        assert not a.report.resolve().is_relative_to(
            a.root.resolve()
        ), "Write review outside the manifested bundle."
        a.report.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in r.items()
                if k not in ["images", "decoded_usd", "reference_bindings"]
            },
            indent=2,
        )
    )
    raise SystemExit(0 if r["passed"] else 1)
