#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed license admission for an OpenVDB native wheel."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import platform
import re
import subprocess
import sys
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from email.parser import BytesParser
from typing import Any

POLICY_SCHEMA = "world-understanding.sdf-native-license-policy.v2"
PREPROMOTION_REPORT_SCHEMA = "world-understanding.sdf-native-prepromotion-verification.v2"
RUNTIME_CLOSURE_SCHEMA = "world-understanding.sdf-native-runtime-closure.v2"
REPORT_SCHEMA = "world-understanding.sdf-native-license-admission.v1"
RELEASE_LOCK_SCHEMA = "world-understanding.sdf-native-release-lock.v1"
PLATFORM_RECORD_KEYS = {
    "status",
    "source_lock_sha256",
    "build_recipe_sha256",
    "license_policy_sha256",
    "license_verifier_sha256",
    "native_dependency_provenance_sha256",
    "prepromotion_report_sha256",
    "ldd_closure_sha256",
    "wheel",
    "wheel_runtime_members",
    "wheel_native_members",
    "build_artifacts",
}
_ROOT_KEYS = {
    "schema",
    "scope",
    "artifact_modes",
    "allowed_selected_licenses",
    "forbidden_introduced_license_families",
    "runtime_pseudo_sonames",
    "components",
    "platform_abi",
}
_COMPONENT_KEYS = {
    "id",
    "version",
    "relationship",
    "selected_license",
    "license_selection_note",
    "runtime_location",
    "soname_patterns",
    "elf_without_soname_member_patterns",
    "archive_members",
    "source_ids",
}
_PLATFORM_KEYS = {
    "id",
    "relationship",
    "license",
    "must_be_external",
    "required",
    "rationale",
    "soname_patterns",
    "provider_packages",
}
_RELATIONSHIPS = {
    "bundled",
    "compiled_in",
    "external_runtime",
    "metadata",
    "native_runtime",
}
_RUNTIME_LOCATIONS = {"bundled", "external"}
_KNOWN_ALLOWED_LICENSES = {"Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "MIT", "Zlib"}
_NATIVE_NAME = re.compile(r"[.]so(?:[.]|$)")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LDD_LINK = re.compile(r"^\s*(?P<name>\S+)\s+=>\s+(?P<path>\S+)\s+\(0x[0-9a-fA-F]+\)\s*$")
_LDD_DIRECT = re.compile(r"^\s*(?P<token>\S+)\s+\(0x[0-9a-fA-F]+\)\s*$")


class AdmissionError(RuntimeError):
    """The artifact or policy failed native license admission."""


@dataclass(frozen=True)
class ElfRecord:
    """ELF metadata for one archive member."""

    member: str
    sha256: str
    soname: str | None
    needed: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeRecord:
    """One library resolved by the clean-environment ldd closure."""

    requested_soname: str
    resolved_path: str | None
    sha256: str | None
    actual_soname: str | None
    needed: tuple[str, ...]
    pseudo: bool = False
    platform_provider: str | None = None


@dataclass(frozen=True)
class SonameClassification:
    """The policy owner of a dynamic-library name."""

    kind: str
    component_id: str


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AdmissionError(f"duplicate JSON key in license policy: {key!r}")
        result[key] = value
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdmissionError(f"{label} must be a non-empty string")
    return value


def _required_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise AdmissionError(f"{label} must be a non-empty list")
    result = [_required_text(item, f"{label}[]") for item in value]
    if len(result) != len(set(result)):
        raise AdmissionError(f"{label} contains duplicate values")
    return result


def _compile(pattern: str, label: str) -> None:
    try:
        re.compile(pattern)
    except re.error as exc:
        raise AdmissionError(f"invalid regex in {label}: {pattern!r}: {exc}") from exc


def load_policy(path: pathlib.Path) -> dict[str, Any]:
    """Load a policy while rejecting duplicate JSON object keys."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"could not load native license policy {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdmissionError("native license policy root must be an object")
    validate_policy(value)
    return value


def _load_json_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"could not load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdmissionError(f"{label} root must be an object")
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise AdmissionError("release evidence is not canonical JSON data") from exc


def _normalized_architecture(value: str) -> str:
    aliases = {
        "amd64": "x86_64",
        "arm64": "aarch64",
    }
    normalized = aliases.get(value.lower(), value.lower())
    if normalized not in {"x86_64", "aarch64"}:
        raise AdmissionError(f"unsupported native release architecture: {value!r}")
    return normalized


def _runtime_relevant_wheel_members(
    archive_digests: Mapping[str, str],
    wheel_elf: Sequence[ElfRecord],
) -> dict[str, str]:
    """Return the exact installed members that participate in runtime trust."""

    native_members = {record.member for record in wheel_elf}
    categories: dict[str, set[str]] = {
        "python wrapper": set(),
        "embedded source lock": set(),
        "native ELF": set(),
        "METADATA": set(),
        "WHEEL": set(),
        "native dependency provenance": set(),
    }
    selected: dict[str, str] = {}
    for name, digest in archive_digests.items():
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            raise AdmissionError("wheel archive digest inventory is malformed")
        path = pathlib.PurePosixPath(name)
        matched = False
        if path.suffix == ".py" and not any(part.endswith(".dist-info") for part in path.parts):
            categories["python wrapper"].add(name)
            matched = True
        if path.name == "_source_lock.json":
            categories["embedded source lock"].add(name)
            matched = True
        if name in native_members:
            categories["native ELF"].add(name)
            matched = True
        if path.name in {"METADATA", "WHEEL"} and path.parent.name.endswith(".dist-info"):
            categories[path.name].add(name)
            matched = True
        if path.name == "NATIVE_DEPENDENCY_PROVENANCE.json":
            categories["native dependency provenance"].add(name)
            matched = True
        if matched:
            selected[name] = digest

    missing = sorted(label for label, names in categories.items() if not names)
    if missing:
        raise AdmissionError(f"wheel is missing runtime-relevant release members: {missing}")
    singleton_categories = {
        "embedded source lock",
        "METADATA",
        "WHEEL",
        "native dependency provenance",
    }
    ambiguous = sorted(label for label in singleton_categories if len(categories[label]) != 1)
    if ambiguous:
        raise AdmissionError(f"wheel has ambiguous runtime-relevant release members: {ambiguous}")
    return dict(sorted(selected.items()))


def _validate_release_lock(
    release_lock: Mapping[str, Any],
    *,
    architecture: str,
    source_lock: Mapping[str, Any],
    source_lock_bytes: bytes,
    policy_bytes: bytes,
    verifier_bytes: bytes,
    provenance: Mapping[str, Any],
    provenance_bytes: bytes,
    prepromotion_report: Mapping[str, Any],
    prepromotion_report_bytes: bytes,
    wheel_path: pathlib.Path,
    wheel_bytes: bytes,
    archive_digests: Mapping[str, str],
    wheel_elf: Sequence[ElfRecord],
) -> Mapping[str, Any]:
    """Bind generated evidence to one independently reviewed release record."""

    if set(release_lock) != {"schema", "trust_model", "platforms"}:
        raise AdmissionError("native release lock has unexpected keys")
    if release_lock.get("schema") != RELEASE_LOCK_SCHEMA:
        raise AdmissionError("unexpected native release-lock schema")
    _required_text(release_lock.get("trust_model"), "release_lock.trust_model")
    platforms = release_lock.get("platforms")
    if not isinstance(platforms, dict):
        raise AdmissionError("native release-lock platforms must be an object")
    normalized_architecture = _normalized_architecture(architecture)
    selected = platforms.get(normalized_architecture)
    if not isinstance(selected, dict):
        raise AdmissionError(
            f"native wheel architecture {normalized_architecture!r} has no promoted release lock"
        )
    if set(selected) != PLATFORM_RECORD_KEYS:
        raise AdmissionError("native release-lock platform record has unexpected keys")
    if selected.get("status") != "promoted":
        raise AdmissionError(
            f"native wheel architecture {normalized_architecture!r} is not promoted"
        )

    recipe = source_lock.get("build_recipe")
    if not isinstance(recipe, dict):
        raise AdmissionError("native source lock has no build recipe")
    prepromotion_ldd = prepromotion_report.get("ldd")
    if (
        not isinstance(prepromotion_ldd, dict)
        or set(prepromotion_ldd) != {"schema", "sha256"}
        or prepromotion_ldd.get("schema") != RUNTIME_CLOSURE_SCHEMA
    ):
        raise AdmissionError("native prepromotion report has no ldd evidence")
    prepromotion_wheel = prepromotion_report.get("wheel")
    if not isinstance(prepromotion_wheel, dict) or prepromotion_wheel != {
        "filename": wheel_path.name,
        "sha256": _sha256(wheel_bytes),
    }:
        raise AdmissionError("native prepromotion report changed wheel identity")
    expected_report_inputs = {
        "policy": _sha256(policy_bytes),
        "source_lock": _sha256(source_lock_bytes),
        "native_dependency_provenance": _sha256(provenance_bytes),
        "verifier": _sha256(verifier_bytes),
    }
    for key, expected_digest in expected_report_inputs.items():
        report_identity = prepromotion_report.get(key)
        if (
            not isinstance(report_identity, dict)
            or report_identity.get("sha256") != expected_digest
        ):
            raise AdmissionError(f"native prepromotion report changed {key} identity")
    expected_digests = {
        "source_lock_sha256": _sha256(source_lock_bytes),
        "build_recipe_sha256": recipe.get("sha256"),
        "license_policy_sha256": _sha256(policy_bytes),
        "license_verifier_sha256": _sha256(verifier_bytes),
        "native_dependency_provenance_sha256": _sha256(provenance_bytes),
        "prepromotion_report_sha256": _sha256(prepromotion_report_bytes),
        "ldd_closure_sha256": prepromotion_ldd.get("sha256"),
    }
    for key, expected in expected_digests.items():
        actual = selected.get(key)
        if not isinstance(actual, str) or not _SHA256.fullmatch(actual):
            raise AdmissionError(f"native release lock has an invalid {key}")
        if actual != expected:
            raise AdmissionError(f"native release lock changed {key}")

    wheel = selected.get("wheel")
    if not isinstance(wheel, dict) or set(wheel) != {"filename", "sha256"}:
        raise AdmissionError("native release-lock wheel identity is malformed")
    if wheel.get("filename") != wheel_path.name or wheel.get("sha256") != _sha256(wheel_bytes):
        raise AdmissionError("native wheel does not match the promoted release digest")

    actual_members = {record.member: record.sha256 for record in wheel_elf}
    prepromotion_elf = prepromotion_report.get("wheel_elf")
    if not isinstance(prepromotion_elf, list) or any(
        not isinstance(row, dict) for row in prepromotion_elf
    ):
        raise AdmissionError("native prepromotion report has no ELF member inventory")
    prepromotion_native_members = {row.get("member"): row.get("sha256") for row in prepromotion_elf}
    if prepromotion_native_members != actual_members:
        raise AdmissionError("native prepromotion report changed ELF member inventory")
    locked_members = selected.get("wheel_native_members")
    if not isinstance(locked_members, dict) or not locked_members:
        raise AdmissionError("native release lock has no native-member inventory")
    if _canonical_json(locked_members) != _canonical_json(actual_members):
        raise AdmissionError("native wheel members do not match the promoted release digests")
    if any(
        not isinstance(path, str) or not isinstance(digest, str) or not _SHA256.fullmatch(digest)
        for path, digest in locked_members.items()
    ):
        raise AdmissionError("native release-lock member inventory is malformed")

    actual_runtime_members = _runtime_relevant_wheel_members(archive_digests, wheel_elf)
    if prepromotion_report.get("wheel_runtime_members") != actual_runtime_members:
        raise AdmissionError("native prepromotion report changed runtime-member inventory")
    locked_runtime_members = selected.get("wheel_runtime_members")
    if not isinstance(locked_runtime_members, dict) or not locked_runtime_members:
        raise AdmissionError("native release lock has no runtime-member inventory")
    if _canonical_json(locked_runtime_members) != _canonical_json(actual_runtime_members):
        raise AdmissionError(
            "runtime-relevant wheel members do not match the promoted release digests"
        )
    if any(
        not isinstance(path, str) or not isinstance(digest, str) or not _SHA256.fullmatch(digest)
        for path, digest in locked_runtime_members.items()
    ):
        raise AdmissionError("native release-lock runtime-member inventory is malformed")

    components = provenance.get("components")
    if not isinstance(components, list):
        raise AdmissionError("native dependency provenance components must be a list")
    actual_artifacts = {
        component["id"]: component["artifacts"]
        for component in components
        if isinstance(component, dict) and component.get("artifacts")
    }
    locked_artifacts = selected.get("build_artifacts")
    if not isinstance(locked_artifacts, dict) or not locked_artifacts:
        raise AdmissionError("native release lock has no source-built artifact inventory")
    if _canonical_json(locked_artifacts) != _canonical_json(actual_artifacts):
        raise AdmissionError("source-built artifacts do not match the promoted release digests")
    return selected


def _source_index(source_lock: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            component_id = value.get("id")
            if isinstance(component_id, str) and "version" in value and "license" in value:
                if component_id in records:
                    raise AdmissionError(f"duplicate source-lock component id: {component_id}")
                records[component_id] = value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(source_lock)
    if not records:
        raise AdmissionError("source lock contains no component source records")
    return records


def _validate_source_record(component_id: str, record: Mapping[str, Any]) -> None:
    version = _required_text(record.get("version"), f"source[{component_id}].version")
    if any(token in version for token in (">", "<", "*")):
        raise AdmissionError(f"source component {component_id} uses a version range")
    license_id = _required_text(record.get("license"), f"source[{component_id}].license")
    if license_id not in _KNOWN_ALLOWED_LICENSES:
        raise AdmissionError(
            f"source component {component_id} has an unapproved license: {license_id}"
        )
    if record.get("compiled_into_artifact") is False:
        return
    license_files = record.get("license_files")
    if not isinstance(license_files, list) or not license_files:
        raise AdmissionError(f"source component {component_id} has no license-file evidence")
    for index, item in enumerate(license_files):
        if not isinstance(item, dict):
            raise AdmissionError(
                f"source component {component_id} license_files[{index}] must be an object"
            )
        _required_text(
            item.get("source_path"),
            f"source[{component_id}].license_files[{index}].source_path",
        )
        _required_text(
            item.get("redistribution_path"),
            f"source[{component_id}].license_files[{index}].redistribution_path",
        )
        digest = _required_text(
            item.get("sha256"), f"source[{component_id}].license_files[{index}].sha256"
        )
        if not _SHA256.fullmatch(digest):
            raise AdmissionError(f"source component {component_id} has an invalid license digest")
    direct_identity_keys = (
        "repository",
        "commit",
        "archive_url",
        "archive_sha256",
        "git_tree_inventory_sha256",
    )
    containing_source = record.get("containing_source_id")
    if containing_source is not None:
        _required_text(containing_source, f"source[{component_id}].containing_source_id")
        _required_text(record.get("containing_path"), f"source[{component_id}].containing_path")
        if not any(key in record for key in direct_identity_keys):
            return
    for key in direct_identity_keys:
        value = _required_text(record.get(key), f"source[{component_id}].{key}")
        if key.endswith("sha256") and not _SHA256.fullmatch(value):
            raise AdmissionError(f"source component {component_id} has an invalid {key}")


def _verify_locked_license_files(
    source_lock: Mapping[str, Any], archive_digests: Mapping[str, str]
) -> None:
    expected: dict[str, tuple[str, str]] = {}
    for component_id, record in _source_index(source_lock).items():
        if record.get("compiled_into_artifact") is False:
            continue
        for item in record.get("license_files", []):
            redistribution_path = _required_text(
                item.get("redistribution_path"),
                f"source[{component_id}].license_files.redistribution_path",
            )
            filename = pathlib.PurePosixPath(redistribution_path).name
            if not filename or filename in expected:
                raise AdmissionError(
                    f"source-lock license redistribution filename is ambiguous: {filename!r}"
                )
            expected[filename] = (component_id, str(item["sha256"]))

    for filename, (component_id, expected_digest) in expected.items():
        members = [
            (member, digest)
            for member, digest in archive_digests.items()
            if ".dist-info/licenses/" in member
            and pathlib.PurePosixPath(member).name == filename
        ]
        if len(members) != 1:
            raise AdmissionError(
                f"wheel must contain exactly one locked {component_id} license file {filename!r}"
            )
        if members[0][1] != expected_digest:
            raise AdmissionError(
                f"wheel {component_id} license file {filename!r} differs from source lock"
            )


def _verify_no_declared_wheel_dependencies(archive: zipfile.ZipFile) -> None:
    metadata_members = [
        name
        for name in archive.namelist()
        if pathlib.PurePosixPath(name).name == "METADATA"
        and pathlib.PurePosixPath(name).parent.name.endswith(".dist-info")
    ]
    if len(metadata_members) != 1:
        raise AdmissionError("wheel must contain exactly one distribution METADATA file")
    metadata = BytesParser().parsebytes(archive.read(metadata_members[0]))
    requirements = metadata.get_all("Requires-Dist", [])
    if requirements:
        raise AdmissionError(
            "native OpenVDB wheel must not declare Python dependencies: "
            f"{sorted(requirements)}"
        )


def _validate_source_materials(
    policy: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if source_lock.get("schema") != "world-understanding.openvdb-native-source-lock.v1":
        raise AdmissionError("unexpected native source-lock schema")
    source_records = _source_index(source_lock)
    for component_id, record in source_records.items():
        _validate_source_record(component_id, record)

    expected_source_ids = {
        component_id
        for component_id, record in source_records.items()
        if record.get("compiled_into_artifact") is not False
    }
    policy_source_ids: set[str] = set()
    for component in _components(policy):
        for source_id in component["source_ids"]:
            source = source_records.get(source_id)
            if source is None:
                raise AdmissionError(
                    f"policy component {component['id']} references unknown source {source_id}"
                )
            policy_source_ids.add(source_id)
            if component["relationship"] != "metadata":
                if component["version"] != source["version"]:
                    raise AdmissionError(
                        f"policy/source version mismatch for {component['id']}: "
                        f"{component['version']} != {source['version']}"
                    )
                if component["selected_license"] != source["license"]:
                    raise AdmissionError(f"policy/source license mismatch for {component['id']}")
    if policy_source_ids != expected_source_ids:
        raise AdmissionError(
            "policy source coverage mismatch: "
            f"missing={sorted(expected_source_ids - policy_source_ids)}, "
            f"unknown={sorted(policy_source_ids - expected_source_ids)}"
        )

    if provenance.get("schema") != ("world-understanding.openvdb-native-dependency-provenance.v1"):
        raise AdmissionError("unexpected native dependency provenance schema")
    components = provenance.get("components")
    if not isinstance(components, list):
        raise AdmissionError("native dependency provenance components must be a list")
    provenance_by_id: dict[str, Mapping[str, Any]] = {}
    for item in components:
        if not isinstance(item, dict):
            raise AdmissionError("native dependency provenance component must be an object")
        component_id = _required_text(item.get("id"), "provenance component id")
        if component_id in provenance_by_id:
            raise AdmissionError(f"duplicate provenance component id: {component_id}")
        provenance_by_id[component_id] = item
    introduced_sources = {
        component_id: source
        for component_id, source in source_records.items()
        if source.get("compiled_into_artifact") is not False
    }
    if set(provenance_by_id) != set(introduced_sources):
        raise AdmissionError("native dependency provenance introduced-source inventory mismatch")
    for component_id, source in introduced_sources.items():
        recorded = provenance_by_id[component_id]
        for key in ("version", "license"):
            if recorded.get(key) != source.get(key):
                raise AdmissionError(f"native dependency provenance changed {component_id} {key}")
        if recorded.get("license_files") != source.get("license_files"):
            raise AdmissionError(
                f"native dependency provenance changed {component_id} license evidence"
            )
        if "commit" in source:
            expected_source = {
                key: source[key]
                for key in (
                    "repository",
                    "commit",
                    "archive_sha256",
                    "git_tree_inventory_sha256",
                )
            }
            if source.get("patches"):
                expected_source["patches"] = source["patches"]
            if recorded.get("source") != expected_source:
                raise AdmissionError(
                    f"native dependency provenance changed {component_id} source identity"
                )
        else:
            expected_vendored_source = {
                "containing_source_id": source.get("containing_source_id"),
                "containing_path": source.get("containing_path"),
                "source_files": source.get("source_files"),
            }
            for key, expected in expected_vendored_source.items():
                if recorded.get(key) != expected:
                    raise AdmissionError(
                        f"native dependency provenance changed {component_id} {key}"
                    )
    return provenance_by_id


def validate_policy(policy: Mapping[str, Any]) -> None:
    """Validate the policy schema and reject permissive-policy escape hatches."""

    if set(policy) != _ROOT_KEYS:
        missing = sorted(_ROOT_KEYS - set(policy))
        unknown = sorted(set(policy) - _ROOT_KEYS)
        raise AdmissionError(
            f"native license policy keys mismatch: missing={missing}, unknown={unknown}"
        )
    if policy.get("schema") != POLICY_SCHEMA:
        raise AdmissionError(f"unexpected native license policy schema: {policy.get('schema')!r}")

    scope = policy.get("scope")
    if not isinstance(scope, dict) or set(scope) != {"subject", "claim", "excludes"}:
        raise AdmissionError("scope must contain subject, claim, and excludes")
    _required_text(scope["subject"], "scope.subject")
    _required_text(scope["claim"], "scope.claim")
    _required_string_list(scope["excludes"], "scope.excludes")

    modes = _required_string_list(policy.get("artifact_modes"), "artifact_modes")
    if modes != ["repaired"]:
        raise AdmissionError("artifact_modes must contain exactly repaired")
    allowed_licenses = set(
        _required_string_list(policy.get("allowed_selected_licenses"), "allowed_selected_licenses")
    )
    unknown_allowed_licenses = allowed_licenses - _KNOWN_ALLOWED_LICENSES
    if unknown_allowed_licenses:
        raise AdmissionError(
            "allowed_selected_licenses contains unknown licenses: "
            f"{sorted(unknown_allowed_licenses)}"
        )
    forbidden = _required_string_list(
        policy.get("forbidden_introduced_license_families"),
        "forbidden_introduced_license_families",
    )
    if set(forbidden) != {"LGPL", "GPL", "AGPL"}:
        raise AdmissionError("the native policy must forbid LGPL, GPL, and AGPL")
    pseudo_patterns = _required_string_list(
        policy.get("runtime_pseudo_sonames"), "runtime_pseudo_sonames"
    )
    for pattern in pseudo_patterns:
        _compile(pattern, "runtime_pseudo_sonames")

    components = policy.get("components")
    if not isinstance(components, list) or not components:
        raise AdmissionError("components must be a non-empty list")
    component_ids: set[str] = set()
    archive_patterns: set[str] = set()
    for index, component in enumerate(components):
        label = f"components[{index}]"
        if not isinstance(component, dict):
            raise AdmissionError(f"{label} must be an object")
        unknown = set(component) - _COMPONENT_KEYS
        required = {
            "id",
            "version",
            "relationship",
            "selected_license",
            "archive_members",
            "source_ids",
        }
        missing = required - set(component)
        if missing or unknown:
            raise AdmissionError(
                f"{label} keys mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        component_id = _required_text(component["id"], f"{label}.id")
        if component_id in component_ids:
            raise AdmissionError(f"duplicate component id: {component_id}")
        component_ids.add(component_id)
        _required_text(component["version"], f"{label}.version")
        relationship = _required_text(component["relationship"], f"{label}.relationship")
        if relationship not in _RELATIONSHIPS:
            raise AdmissionError(f"unsupported relationship for {component_id}: {relationship}")
        selected_license = _required_text(
            component["selected_license"], f"{label}.selected_license"
        )
        if selected_license not in allowed_licenses:
            raise AdmissionError(
                f"unapproved or unknown selected license for {component_id}: {selected_license}"
            )
        upper_license = selected_license.upper()
        for family in forbidden:
            if family in upper_license:
                raise AdmissionError(
                    f"forbidden {family} selected license for introduced component "
                    f"{component_id}: {selected_license}"
                )
        _required_string_list(component["source_ids"], f"{label}.source_ids")

        member_rules = component["archive_members"]
        if not isinstance(member_rules, list) or not member_rules:
            raise AdmissionError(f"{label}.archive_members must be a non-empty list")
        for rule_index, rule in enumerate(member_rules):
            rule_label = f"{label}.archive_members[{rule_index}]"
            if not isinstance(rule, dict) or set(rule) != {"pattern", "required_in"}:
                raise AdmissionError(f"{rule_label} must contain pattern and required_in")
            pattern = _required_text(rule["pattern"], f"{rule_label}.pattern")
            if pattern in archive_patterns:
                raise AdmissionError(f"duplicate archive member pattern: {pattern}")
            archive_patterns.add(pattern)
            _compile(pattern, rule_label)
            required_in = rule["required_in"]
            if not isinstance(required_in, list) or not set(required_in).issubset(modes):
                raise AdmissionError(f"{rule_label}.required_in contains an unknown mode")
            if len(required_in) != len(set(required_in)):
                raise AdmissionError(f"{rule_label}.required_in contains duplicates")

        soname_patterns = component.get("soname_patterns", [])
        if not isinstance(soname_patterns, list):
            raise AdmissionError(f"{label}.soname_patterns must be a list")
        for pattern in soname_patterns:
            _compile(_required_text(pattern, f"{label}.soname_patterns[]"), label)
        no_soname_patterns = component.get("elf_without_soname_member_patterns", [])
        if not isinstance(no_soname_patterns, list):
            raise AdmissionError(f"{label}.elf_without_soname_member_patterns must be a list")
        for pattern in no_soname_patterns:
            _compile(
                _required_text(pattern, f"{label}.elf_without_soname_member_patterns[]"), label
            )

        location = component.get("runtime_location")
        if soname_patterns:
            if not isinstance(location, dict) or set(location) != set(modes):
                raise AdmissionError(
                    f"{component_id} has SONAMEs and needs a runtime_location for every mode"
                )
            if not set(location.values()).issubset(_RUNTIME_LOCATIONS):
                raise AdmissionError(f"{component_id} has an invalid runtime location")
        elif location is not None:
            raise AdmissionError(f"{component_id} has runtime_location but no SONAME patterns")

    platform = policy.get("platform_abi")
    if not isinstance(platform, list) or not platform:
        raise AdmissionError("platform_abi must be a non-empty list")
    platform_ids: set[str] = set()
    for index, item in enumerate(platform):
        label = f"platform_abi[{index}]"
        if not isinstance(item, dict) or set(item) != _PLATFORM_KEYS:
            raise AdmissionError(f"{label} has missing or unknown keys")
        component_id = _required_text(item["id"], f"{label}.id")
        if component_id in component_ids or component_id in platform_ids:
            raise AdmissionError(f"duplicate native policy id: {component_id}")
        platform_ids.add(component_id)
        if item["relationship"] != "platform_abi" or item["must_be_external"] is not True:
            raise AdmissionError(f"{component_id} must be an external platform_abi entry")
        if not isinstance(item["required"], bool):
            raise AdmissionError(f"{component_id}.required must be boolean")
        _required_text(item["license"], f"{label}.license")
        _required_text(item["rationale"], f"{label}.rationale")
        patterns = _required_string_list(item["soname_patterns"], f"{label}.soname_patterns")
        for pattern in patterns:
            _compile(pattern, label)
        _required_string_list(item["provider_packages"], f"{label}.provider_packages")


def _components(policy: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(policy["components"])


def _platform(policy: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(policy["platform_abi"])


def _platform_by_id(policy: Mapping[str, Any], component_id: str) -> Mapping[str, Any]:
    for component in _platform(policy):
        if component["id"] == component_id:
            return component
    raise AdmissionError(f"native policy references unknown platform ABI: {component_id}")


def _component_by_id(policy: Mapping[str, Any], component_id: str) -> Mapping[str, Any]:
    for component in _components(policy):
        if component["id"] == component_id:
            return component
    raise AdmissionError(f"native policy references unknown component: {component_id}")


def classify_archive_member(policy: Mapping[str, Any], name: str) -> tuple[str, str]:
    """Return component id and matched rule; reject unknown or ambiguous files."""

    matches: list[tuple[str, str]] = []
    for component in _components(policy):
        for rule in component["archive_members"]:
            if re.fullmatch(rule["pattern"], name):
                matches.append((component["id"], rule["pattern"]))
    if not matches:
        raise AdmissionError(f"unknown wheel archive member: {name}")
    if len(matches) != 1:
        raise AdmissionError(f"ambiguous wheel archive member {name}: {matches}")
    return matches[0]


def classify_soname(policy: Mapping[str, Any], soname: str) -> SonameClassification:
    """Classify a SONAME into one introduced component or the external ABI."""

    matches: list[SonameClassification] = []
    for pattern in policy["runtime_pseudo_sonames"]:
        if re.fullmatch(pattern, soname):
            matches.append(SonameClassification("pseudo", "runtime-pseudo"))
    for component in _components(policy):
        for pattern in component.get("soname_patterns", []):
            if re.fullmatch(pattern, soname):
                matches.append(SonameClassification("component", component["id"]))
    for item in _platform(policy):
        for pattern in item["soname_patterns"]:
            if re.fullmatch(pattern, soname):
                matches.append(SonameClassification("platform_abi", item["id"]))
    unique = {(match.kind, match.component_id) for match in matches}
    if not unique:
        raise AdmissionError(f"unknown ELF dependency or SONAME: {soname}")
    if len(unique) != 1:
        raise AdmissionError(f"ambiguous ELF dependency or SONAME {soname}: {sorted(unique)}")
    kind, component_id = next(iter(unique))
    return SonameClassification(kind, component_id)


def _elf_metadata(data: bytes, label: str) -> tuple[str | None, tuple[str, ...]]:
    try:
        from elftools.elf.elffile import ELFFile
    except ImportError as exc:
        raise AdmissionError("pyelftools is required for native wheel license admission") from exc
    try:
        elf = ELFFile(io.BytesIO(data))
        dynamic = elf.get_section_by_name(".dynamic")
        if dynamic is None:
            raise AdmissionError(f"ELF has no dynamic section: {label}")
        sonames: list[str] = []
        needed: list[str] = []
        for tag in dynamic.iter_tags():
            if tag.entry.d_tag == "DT_SONAME":
                sonames.append(str(tag.soname))
            elif tag.entry.d_tag == "DT_NEEDED":
                needed.append(str(tag.needed))
    except AdmissionError:
        raise
    except Exception as exc:
        raise AdmissionError(f"could not inspect ELF {label}: {exc}") from exc
    if len(sonames) > 1:
        raise AdmissionError(f"ELF has multiple SONAME entries: {label}: {sonames}")
    if len(needed) != len(set(needed)):
        raise AdmissionError(f"ELF has duplicate DT_NEEDED entries: {label}: {needed}")
    return (sonames[0] if sonames else None, tuple(sorted(needed)))


def inspect_wheel(
    wheel: pathlib.Path,
) -> tuple[dict[str, str], list[ElfRecord]]:
    """Inventory every wheel file and parse every ELF payload."""

    archive_digests: dict[str, str] = {}
    elf_records: list[ElfRecord] = []
    try:
        archive = zipfile.ZipFile(wheel)
    except (OSError, zipfile.BadZipFile) as exc:
        raise AdmissionError(f"could not open wheel {wheel}: {exc}") from exc
    with archive:
        seen: set[str] = set()
        for info in archive.infolist():
            name = info.filename
            if name in seen:
                raise AdmissionError(f"wheel contains a duplicate archive member: {name}")
            seen.add(name)
            if "\\" in name:
                raise AdmissionError(f"wheel archive member uses a backslash: {name}")
            path = pathlib.PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise AdmissionError(f"unsafe wheel archive member: {name}")
            if info.is_dir():
                continue
            data = archive.read(info)
            digest = _sha256(data)
            archive_digests[name] = digest
            is_elf = data.startswith(b"\x7fELF")
            native_name = bool(_NATIVE_NAME.search(path.name))
            if native_name and not is_elf:
                raise AdmissionError(f"native-looking wheel member is not ELF: {name}")
            if is_elf and not native_name:
                raise AdmissionError(
                    f"unexpected ELF wheel member without a shared-library name: {name}"
                )
            if is_elf:
                soname, needed = _elf_metadata(data, name)
                elf_records.append(ElfRecord(name, digest, soname, needed))
    if not archive_digests:
        raise AdmissionError("wheel contains no files")
    if not elf_records:
        raise AdmissionError("wheel contains no ELF members")
    return archive_digests, elf_records


def _debian_platform_provider(
    provider_packages: Sequence[str],
    resolved_path: pathlib.Path,
) -> str:
    """Attest a platform DSO to an installed base-system Debian package."""

    try:
        canonical = resolved_path.resolve(strict=True)
    except OSError as exc:
        raise AdmissionError(
            f"could not resolve platform ABI provider {resolved_path}: {exc}"
        ) from exc

    for package_name in provider_packages:
        try:
            status = subprocess.run(
                ["/usr/bin/dpkg-query", "--show", "--showformat=${db:Status-Abbrev}", package_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AdmissionError(
                f"could not inspect platform ABI package status for {package_name}: {exc}"
            ) from exc
        if status.returncode != 0 or status.stdout != "ii ":
            continue
        try:
            manifest = subprocess.run(
                ["/usr/bin/dpkg-query", "--listfiles", package_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AdmissionError(
                f"could not inspect platform ABI package manifest for {package_name}: {exc}"
            ) from exc
        if manifest.returncode != 0:
            continue
        for manifest_entry in manifest.stdout.splitlines():
            try:
                if pathlib.Path(manifest_entry).samefile(canonical):
                    return f"debian-package:{package_name}"
            except OSError:
                continue

    raise AdmissionError(
        f"platform ABI provider is not owned by an installed trusted base package: "
        f"{resolved_path}"
    )


def inspect_ldd(policy: Mapping[str, Any], ldd_path: pathlib.Path) -> list[RuntimeRecord]:
    """Parse and inspect the complete ldd closure captured in a clean environment."""

    try:
        lines = ldd_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AdmissionError(f"could not read ldd closure {ldd_path}: {exc}") from exc
    if not lines:
        raise AdmissionError("ldd closure is empty")
    records: list[RuntimeRecord] = []
    seen: set[str] = set()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if re.search(r"=>\s+not found(?:\s|$)", line):
            raise AdmissionError(f"unresolved library in ldd closure: {line}")
        link = _LDD_LINK.fullmatch(raw_line)
        direct = _LDD_DIRECT.fullmatch(raw_line)
        if link:
            requested = link.group("name")
            resolved = pathlib.Path(link.group("path"))
        elif direct:
            token = direct.group("token")
            if token.startswith("/"):
                resolved = pathlib.Path(token)
                requested = resolved.name
            else:
                classification = classify_soname(policy, token)
                if classification.kind != "pseudo":
                    raise AdmissionError(f"ldd emitted an unresolved non-pseudo entry: {line}")
                requested = token
                resolved = None
        else:
            raise AdmissionError(f"unrecognized ldd closure line: {raw_line!r}")
        if requested in seen:
            raise AdmissionError(f"ldd closure contains duplicate SONAME: {requested}")
        seen.add(requested)
        classification = classify_soname(policy, requested)
        if resolved is None:
            records.append(RuntimeRecord(requested, None, None, None, (), pseudo=True))
            continue
        try:
            data = resolved.read_bytes()
        except OSError as exc:
            raise AdmissionError(f"could not read resolved library {resolved}: {exc}") from exc
        if not data.startswith(b"\x7fELF"):
            raise AdmissionError(f"resolved runtime library is not ELF: {resolved}")
        digest = _sha256(data)
        actual_soname, needed = _elf_metadata(data, str(resolved))
        platform_provider = None
        if classification.kind == "platform_abi":
            # The platform ABI is the explicit recursion boundary. Its identity is
            # still recorded, but its own dependency graph belongs to the base image.
            platform_provider = _debian_platform_provider(
                _platform_by_id(policy, classification.component_id)["provider_packages"],
                resolved,
            )
            needed = ()
        records.append(
            RuntimeRecord(
                requested_soname=requested,
                resolved_path=str(resolved.resolve()),
                sha256=digest,
                actual_soname=actual_soname,
                needed=needed,
                platform_provider=platform_provider,
            )
        )
    return records


def _allows_missing_soname(component: Mapping[str, Any], member: str) -> bool:
    return any(
        re.fullmatch(pattern, member)
        for pattern in component.get("elf_without_soname_member_patterns", [])
    )


def _target_location(
    policy: Mapping[str, Any], classification: SonameClassification, mode: str
) -> str:
    if classification.kind in {"platform_abi", "pseudo"}:
        return "external"
    component = _component_by_id(policy, classification.component_id)
    location = component.get("runtime_location")
    if not isinstance(location, dict) or mode not in location:
        raise AdmissionError(
            f"ELF component {classification.component_id} has no runtime location for {mode}"
        )
    return str(location[mode])


def verify_inventory(
    policy: Mapping[str, Any],
    *,
    mode: str,
    archive_digests: Mapping[str, str],
    wheel_elf: Sequence[ElfRecord],
    runtime: Sequence[RuntimeRecord],
) -> dict[str, Any]:
    """Apply the policy to a pre-inspected wheel and resolved runtime closure."""

    validate_policy(policy)
    if mode not in policy["artifact_modes"]:
        raise AdmissionError(f"unknown artifact mode: {mode}")

    member_rows: list[dict[str, str]] = []
    rule_counts: dict[tuple[str, str], int] = {}
    member_components: dict[str, str] = {}
    for name, digest in sorted(archive_digests.items()):
        component_id, pattern = classify_archive_member(policy, name)
        member_components[name] = component_id
        rule_counts[(component_id, pattern)] = rule_counts.get((component_id, pattern), 0) + 1
        member_rows.append({"path": name, "sha256": digest, "component": component_id})
    for component in _components(policy):
        for rule in component["archive_members"]:
            if mode in rule["required_in"] and not rule_counts.get(
                (component["id"], rule["pattern"])
            ):
                raise AdmissionError(
                    f"wheel is missing required {mode} archive member for {component['id']}: "
                    f"{rule['pattern']}"
                )

    wheel_rows: list[dict[str, Any]] = []
    wheel_providers: dict[str, list[tuple[str, str, str]]] = {}
    wheel_components_with_soname: set[str] = set()
    for record in wheel_elf:
        if record.member not in archive_digests:
            raise AdmissionError(f"ELF inventory references a non-member: {record.member}")
        component_id = member_components[record.member]
        component = _component_by_id(policy, component_id)
        if record.soname is None:
            if not _allows_missing_soname(component, record.member):
                raise AdmissionError(f"ELF member is missing an approved SONAME: {record.member}")
        else:
            classification = classify_soname(policy, record.soname)
            if classification.kind == "platform_abi":
                raise AdmissionError(
                    f"platform ABI {record.soname} must remain external, but is bundled in "
                    f"{record.member}"
                )
            if classification.kind != "component" or classification.component_id != component_id:
                raise AdmissionError(
                    f"wheel ELF component mismatch for {record.member}: archive={component_id}, "
                    f"soname={classification}"
                )
            wheel_components_with_soname.add(component_id)
            wheel_providers.setdefault(record.soname, []).append(
                (component_id, record.sha256, record.member)
            )
        wheel_rows.append(
            {
                **asdict(record),
                "needed": list(record.needed),
                "component": component_id,
            }
        )
    for soname, providers in wheel_providers.items():
        identities = {(component_id, digest) for component_id, digest, _ in providers}
        if len(identities) != 1:
            raise AdmissionError(f"wheel has conflicting providers for {soname}: {providers}")

    for component in _components(policy):
        if not component.get("soname_patterns"):
            continue
        expected = component["runtime_location"][mode]
        bundled = component["id"] in wheel_components_with_soname
        if expected == "bundled" and not bundled:
            raise AdmissionError(
                f"{component['id']} must be bundled in a {mode} wheel but has no ELF provider"
            )
        if expected == "external" and bundled:
            raise AdmissionError(
                f"{component['id']} must remain external in a {mode} wheel but is bundled"
            )

    runtime_by_requested: dict[str, RuntimeRecord] = {}
    runtime_rows: list[dict[str, Any]] = []
    runtime_component_ids: set[str] = set()
    platform_ids: set[str] = set()
    for record in runtime:
        if record.requested_soname in runtime_by_requested:
            raise AdmissionError(f"duplicate runtime closure record: {record.requested_soname}")
        runtime_by_requested[record.requested_soname] = record
        classification = classify_soname(policy, record.requested_soname)
        row = {
            "requested_soname": record.requested_soname,
            "sha256": record.sha256,
            "actual_soname": record.actual_soname,
            "needed": sorted(record.needed),
            "pseudo": record.pseudo,
            "classification": classification.kind,
            "component": classification.component_id,
        }
        if classification.kind == "pseudo":
            if (
                not record.pseudo
                or record.resolved_path is not None
                or record.platform_provider is not None
            ):
                raise AdmissionError(
                    f"runtime pseudo-library unexpectedly resolved: {record.requested_soname}"
                )
            row["normalized_location"] = f"pseudo:{record.requested_soname}"
            runtime_rows.append(row)
            continue
        if record.pseudo or not record.resolved_path or not record.sha256:
            raise AdmissionError(
                f"runtime library did not resolve to an inspected ELF: {record.requested_soname}"
            )
        if (
            classification.kind == "platform_abi"
            and record.actual_soname != record.requested_soname
        ):
            raise AdmissionError(
                f"resolved platform ABI SONAME differs from request: "
                f"{record.requested_soname} -> {record.actual_soname}"
            )
        if record.actual_soname:
            actual = classify_soname(policy, record.actual_soname)
            if actual != classification:
                raise AdmissionError(
                    f"resolved SONAME changed component for {record.requested_soname}: "
                    f"{record.actual_soname}"
                )
        if classification.kind == "platform_abi":
            platform_ids.add(classification.component_id)
            platform_policy = _platform_by_id(policy, classification.component_id)
            trusted_providers = {
                f"debian-package:{package_name}"
                for package_name in platform_policy["provider_packages"]
            }
            if record.platform_provider not in trusted_providers:
                raise AdmissionError(
                    f"platform ABI provider has no trusted base-package identity: "
                    f"{record.requested_soname}"
                )
            if record.needed:
                raise AdmissionError(
                    f"platform ABI recursion boundary contains inspected edges: "
                    f"{record.requested_soname}"
                )
            # Host ABI bytes can vary across patched base images. Normalize only
            # after binding the ELF to its exact SONAME and trusted package owner.
            row["sha256"] = None
            row["platform_provider"] = record.platform_provider
            row["normalized_location"] = f"platform:{record.requested_soname}"
            runtime_rows.append(row)
            continue
        if record.platform_provider is not None:
            raise AdmissionError(
                f"non-platform runtime library has a platform provider identity: "
                f"{record.requested_soname}"
            )
        runtime_component_ids.add(classification.component_id)
        expected = _target_location(policy, classification, mode)
        requested_providers = [
            (digest, member)
            for component_id, digest, member in wheel_providers.get(record.requested_soname, [])
            if component_id == classification.component_id
        ]
        provider_hashes = {digest for digest, _ in requested_providers}
        if expected == "bundled" and record.sha256 not in provider_hashes:
            raise AdmissionError(
                f"runtime loaded external or modified {classification.component_id}: "
                f"{record.resolved_path}"
            )
        if expected == "external" and provider_hashes:
            raise AdmissionError(
                f"runtime component {classification.component_id} must remain external"
            )
        if expected == "bundled":
            matching_members = sorted(
                member for digest, member in requested_providers if digest == record.sha256
            )
            if not matching_members:
                raise AdmissionError(
                    f"runtime has no exact wheel identity for {record.requested_soname}"
                )
            row["normalized_location"] = f"wheel:{matching_members[0]}"
        else:
            row["normalized_location"] = f"external:{record.requested_soname}"
        runtime_rows.append(row)

    for component in _components(policy):
        if component.get("soname_patterns") and component["id"] not in runtime_component_ids:
            raise AdmissionError(
                f"resolved runtime closure is missing component: {component['id']}"
            )
    for item in _platform(policy):
        if item["required"] and item["id"] not in platform_ids:
            raise AdmissionError(f"resolved runtime closure is missing platform ABI: {item['id']}")

    edges: list[dict[str, str]] = []
    edge_sources: Iterable[tuple[str, Sequence[str]]] = [
        (f"wheel:{record.member}", record.needed) for record in wheel_elf
    ]
    edge_sources = list(edge_sources) + [
        (f"runtime:{record.requested_soname}", record.needed)
        for record in runtime
        if not record.pseudo
    ]
    for source, needed_values in edge_sources:
        for needed in needed_values:
            classification = classify_soname(policy, needed)
            if needed not in runtime_by_requested:
                raise AdmissionError(
                    f"ELF edge is absent from resolved ldd closure: {source} -> {needed}"
                )
            expected = _target_location(policy, classification, mode)
            providers = wheel_providers.get(needed, [])
            if expected == "bundled" and not providers:
                raise AdmissionError(
                    f"bundled ELF edge has no wheel provider: {source} -> {needed}"
                )
            if expected == "external" and providers:
                raise AdmissionError(
                    f"external or platform ELF edge was bundled: {source} -> {needed}"
                )
            edges.append(
                {
                    "source": source,
                    "needed": needed,
                    "classification": classification.kind,
                    "component": classification.component_id,
                    "expected_location": expected,
                }
            )

    return {
        "archive_members": member_rows,
        "wheel_elf": sorted(wheel_rows, key=lambda row: row["member"]),
        "runtime_libraries": sorted(runtime_rows, key=lambda row: row["requested_soname"]),
        "dependency_edges": sorted(edges, key=lambda row: (row["source"], row["needed"])),
    }


def _canonical_runtime_closure_sha256(inventory: Mapping[str, Any]) -> str:
    runtime_libraries = inventory.get("runtime_libraries")
    if not isinstance(runtime_libraries, list):
        raise AdmissionError("verified runtime-library evidence is malformed")
    return _sha256(
        _canonical_json(
            {
                "schema": RUNTIME_CLOSURE_SCHEMA,
                "runtime_libraries": runtime_libraries,
            }
        ).encode("utf-8")
    )


def _verify_source_built_tbb(
    inventory: Mapping[str, Any],
    provenance_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    tbb = provenance_by_id.get("onetbb")
    if tbb is None:
        raise AdmissionError("native dependency provenance is missing oneTBB")
    tbb_artifacts = tbb.get("artifacts")
    if (
        not isinstance(tbb_artifacts, list)
        or len(tbb_artifacts) != 1
        or not isinstance(tbb_artifacts[0], dict)
    ):
        raise AdmissionError("oneTBB provenance must identify exactly one source-built artifact")
    source_tbb_sha256 = tbb_artifacts[0].get("sha256")
    if not isinstance(source_tbb_sha256, str) or not _SHA256.fullmatch(source_tbb_sha256):
        raise AdmissionError("oneTBB provenance artifact digest is invalid")
    wheel_elf = inventory.get("wheel_elf")
    if not isinstance(wheel_elf, list):
        raise AdmissionError("wheel ELF inventory is malformed")
    bundled_tbb = [row for row in wheel_elf if row.get("component") == "onetbb"]
    if len(bundled_tbb) != 1 or f"-{source_tbb_sha256[:8]}.so" not in bundled_tbb[0].get(
        "member", ""
    ):
        raise AdmissionError("bundled oneTBB does not carry the source-built auditwheel identity")


def verify_prepromotion(
    *,
    policy_path: pathlib.Path,
    source_lock_path: pathlib.Path,
    provenance_path: pathlib.Path,
    wheel_path: pathlib.Path,
    ldd_path: pathlib.Path,
    mode: str,
) -> dict[str, Any]:
    """Verify a wheel completely without claiming that it is release-admitted."""

    policy_bytes = policy_path.read_bytes()
    policy = load_policy(policy_path)
    source_lock_bytes = source_lock_path.read_bytes()
    provenance_bytes = provenance_path.read_bytes()
    ldd_bytes = ldd_path.read_bytes()
    wheel_bytes = wheel_path.read_bytes()
    source_lock = _load_json_object(source_lock_path, "native source lock")
    provenance = _load_json_object(provenance_path, "native dependency provenance")
    provenance_by_id = _validate_source_materials(policy, source_lock, provenance)
    if provenance.get("source_lock_sha256") != _sha256(source_lock_bytes):
        raise AdmissionError("native dependency provenance source-lock digest mismatch")

    recipe = source_lock.get("build_recipe")
    if not isinstance(recipe, dict) or set(recipe) != {"path", "sha256"}:
        raise AdmissionError("source lock must bind the exact native build recipe")
    recipe_value = _required_text(recipe["path"], "build_recipe.path")
    recipe_relative = pathlib.PurePosixPath(recipe_value)
    if recipe_relative.is_absolute() or ".." in recipe_relative.parts or "\\" in recipe_value:
        raise AdmissionError("build_recipe.path must be a safe relative path")
    recipe_path = source_lock_path.parent.joinpath(*recipe_relative.parts)
    if not _SHA256.fullmatch(str(recipe["sha256"])):
        raise AdmissionError("source-lock build recipe digest is invalid")
    try:
        recipe_digest = _sha256(recipe_path.read_bytes())
    except OSError as exc:
        raise AdmissionError(f"could not inspect native build recipe {recipe_path}: {exc}") from exc
    if recipe_digest != recipe["sha256"]:
        raise AdmissionError("native build recipe digest differs from source lock")

    admission = source_lock.get("license_admission")
    if not isinstance(admission, dict):
        raise AdmissionError("source lock does not bind native license admission inputs")
    verifier_bytes = pathlib.Path(__file__).resolve().read_bytes()
    expected_admission = {
        "policy": {
            "path": policy_path.name,
            "schema": POLICY_SCHEMA,
            "sha256": _sha256(policy_bytes),
        },
        "verifier": {
            "path": pathlib.Path(__file__).name,
            "report_schema": REPORT_SCHEMA,
            "sha256": _sha256(verifier_bytes),
        },
    }
    if admission != expected_admission:
        raise AdmissionError("source-lock native policy or verifier binding mismatch")
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            _verify_no_declared_wheel_dependencies(archive)
            if archive.read("openvdb/_source_lock.json") != source_lock_bytes:
                raise AdmissionError("wheel embedded source lock differs from admitted input")
            provenance_members = [
                name
                for name in archive.namelist()
                if pathlib.PurePosixPath(name).name == "NATIVE_DEPENDENCY_PROVENANCE.json"
            ]
            if len(provenance_members) != 1:
                raise AdmissionError(
                    "wheel must contain exactly one native dependency provenance document"
                )
            if archive.read(provenance_members[0]) != provenance_bytes:
                raise AdmissionError("wheel embedded native dependency provenance differs")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise AdmissionError(f"could not verify embedded source materials: {exc}") from exc

    archive_digests, wheel_elf = inspect_wheel(wheel_path)
    _verify_locked_license_files(source_lock, archive_digests)
    runtime = inspect_ldd(policy, ldd_path)
    inventory = verify_inventory(
        policy,
        mode=mode,
        archive_digests=archive_digests,
        wheel_elf=wheel_elf,
        runtime=runtime,
    )
    _verify_source_built_tbb(inventory, provenance_by_id)
    runtime_members = _runtime_relevant_wheel_members(archive_digests, wheel_elf)
    if ldd_path.read_bytes() != ldd_bytes:
        raise AdmissionError("ldd closure changed during native verification")
    if wheel_path.read_bytes() != wheel_bytes:
        raise AdmissionError("wheel changed during native verification")
    return {
        "schema": PREPROMOTION_REPORT_SCHEMA,
        "prepromotion": {
            "passed": True,
            "admitted": False,
            "deployable": False,
            "requires_promoted_release_lock": True,
            "artifact_mode": mode,
            "unknown_components_allowed": False,
            "unknown_licenses_allowed": False,
            "platform_abi_must_be_external": True,
            "forbidden_introduced_license_families": list(
                policy["forbidden_introduced_license_families"]
            ),
        },
        "scope": policy["scope"],
        "verifier": {
            "filename": pathlib.Path(__file__).name,
            "sha256": _sha256(verifier_bytes),
        },
        "policy": {
            "filename": policy_path.name,
            "sha256": _sha256(policy_bytes),
        },
        "source_lock": {
            "filename": source_lock_path.name,
            "sha256": _sha256(source_lock_bytes),
        },
        "native_dependency_provenance": {
            "filename": provenance_path.name,
            "sha256": _sha256(provenance_bytes),
            "source_ids": sorted(provenance_by_id),
        },
        "ldd": {
            "schema": RUNTIME_CLOSURE_SCHEMA,
            "sha256": _canonical_runtime_closure_sha256(inventory),
        },
        "wheel": {
            "filename": wheel_path.name,
            "sha256": _sha256(wheel_bytes),
        },
        "wheel_runtime_members": runtime_members,
        "components": [
            {
                "id": component["id"],
                "version": component["version"],
                "relationship": component["relationship"],
                "selected_license": component["selected_license"],
                "runtime_location": component.get("runtime_location", {}).get(mode),
            }
            for component in _components(policy)
        ],
        **inventory,
    }


def load_verified_prepromotion_report(
    path: pathlib.Path,
    *,
    expected: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    """Load a non-deployable report and require an exact recomputed match."""

    report_bytes = path.read_bytes()
    report = _load_json_object(path, "native prepromotion verification report")
    marker = report.get("prepromotion")
    if report.get("schema") != PREPROMOTION_REPORT_SCHEMA or not isinstance(marker, dict):
        raise AdmissionError("unexpected native prepromotion report schema")
    required_marker = {
        "passed": True,
        "admitted": False,
        "deployable": False,
        "requires_promoted_release_lock": True,
    }
    if any(marker.get(key) != value for key, value in required_marker.items()):
        raise AdmissionError("prepromotion report falsely claims release admission")
    if _canonical_json(report) != _canonical_json(expected):
        raise AdmissionError("prepromotion report differs from recomputed verification")
    return report, report_bytes


def verify_wheel(
    *,
    policy_path: pathlib.Path,
    release_lock_path: pathlib.Path,
    prepromotion_report_path: pathlib.Path,
    source_lock_path: pathlib.Path,
    provenance_path: pathlib.Path,
    wheel_path: pathlib.Path,
    ldd_path: pathlib.Path,
    mode: str,
    architecture: str | None = None,
) -> dict[str, Any]:
    """Recompute prepromotion verification and admit one promoted wheel."""

    recomputed = verify_prepromotion(
        policy_path=policy_path,
        source_lock_path=source_lock_path,
        provenance_path=provenance_path,
        wheel_path=wheel_path,
        ldd_path=ldd_path,
        mode=mode,
    )
    prepromotion_report, prepromotion_report_bytes = load_verified_prepromotion_report(
        prepromotion_report_path, expected=recomputed
    )
    policy_bytes = policy_path.read_bytes()
    policy = load_policy(policy_path)
    release_lock_bytes = release_lock_path.read_bytes()
    release_lock = _load_json_object(release_lock_path, "native release lock")
    source_lock_bytes = source_lock_path.read_bytes()
    provenance_bytes = provenance_path.read_bytes()
    source_lock = _load_json_object(source_lock_path, "native source lock")
    provenance = _load_json_object(provenance_path, "native dependency provenance")
    _validate_source_materials(policy, source_lock, provenance)
    archive_digests, wheel_elf = inspect_wheel(wheel_path)
    wheel_bytes = wheel_path.read_bytes()
    selected_release = _validate_release_lock(
        release_lock,
        architecture=architecture or platform.machine(),
        source_lock=source_lock,
        source_lock_bytes=source_lock_bytes,
        policy_bytes=policy_bytes,
        verifier_bytes=pathlib.Path(__file__).resolve().read_bytes(),
        provenance=provenance,
        provenance_bytes=provenance_bytes,
        prepromotion_report=prepromotion_report,
        prepromotion_report_bytes=prepromotion_report_bytes,
        wheel_path=wheel_path,
        wheel_bytes=wheel_bytes,
        archive_digests=archive_digests,
        wheel_elf=wheel_elf,
    )
    shared_evidence = dict(recomputed)
    del shared_evidence["schema"]
    del shared_evidence["prepromotion"]
    return {
        "schema": REPORT_SCHEMA,
        "admission": {
            "passed": True,
            "artifact_mode": mode,
            "unknown_components_allowed": False,
            "unknown_licenses_allowed": False,
            "platform_abi_must_be_external": True,
            "forbidden_introduced_license_families": list(
                policy["forbidden_introduced_license_families"]
            ),
        },
        "scope": policy["scope"],
        "prepromotion_report": {
            "filename": prepromotion_report_path.name,
            "sha256": _sha256(prepromotion_report_bytes),
            "schema": PREPROMOTION_REPORT_SCHEMA,
            "passed": True,
            "admitted": False,
            "deployable": False,
        },
        "release_lock": {
            "filename": release_lock_path.name,
            "sha256": _sha256(release_lock_bytes),
            "architecture": _normalized_architecture(architecture or platform.machine()),
            "status": selected_release["status"],
        },
        **shared_evidence,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--release-lock", type=pathlib.Path)
    parser.add_argument("--prepromotion-report", type=pathlib.Path)
    parser.add_argument("--prepromotion-only", action="store_true")
    parser.add_argument("--source-lock", required=True, type=pathlib.Path)
    parser.add_argument("--provenance", required=True, type=pathlib.Path)
    parser.add_argument("--wheel", required=True, type=pathlib.Path)
    parser.add_argument("--ldd", required=True, type=pathlib.Path)
    parser.add_argument("--artifact-mode", required=True, choices=("repaired",))
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.prepromotion_only:
            if args.release_lock is not None or args.prepromotion_report is not None:
                raise AdmissionError(
                    "prepromotion verification cannot consume release-admission inputs"
                )
            report = verify_prepromotion(
                policy_path=args.policy,
                source_lock_path=args.source_lock,
                provenance_path=args.provenance,
                wheel_path=args.wheel,
                ldd_path=args.ldd,
                mode=args.artifact_mode,
            )
        else:
            if args.release_lock is None or args.prepromotion_report is None:
                raise AdmissionError(
                    "final admission requires --release-lock and --prepromotion-report"
                )
            report = verify_wheel(
                policy_path=args.policy,
                release_lock_path=args.release_lock,
                prepromotion_report_path=args.prepromotion_report,
                source_lock_path=args.source_lock,
                provenance_path=args.provenance,
                wheel_path=args.wheel,
                ldd_path=args.ldd,
                mode=args.artifact_mode,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (AdmissionError, OSError) as exc:
        print(f"native wheel license admission failed: {exc}", file=sys.stderr)
        return 1
    if args.prepromotion_only:
        print(
            "Native wheel prepromotion verification passed; artifact remains "
            f"unadmitted and non-deployable ({args.artifact_mode}): {args.wheel}",
            file=sys.stdout,
        )
    else:
        print(
            f"Native wheel license admission passed ({args.artifact_mode}): {args.wheel}",
            file=sys.stdout,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
