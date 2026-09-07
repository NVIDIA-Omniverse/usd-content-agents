# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

PACKAGE_ROOT = Path(__file__).parents[1]
NATIVE_ROOT = PACKAGE_ROOT / "native"
POLICY_PATH = NATIVE_ROOT / "license-policy.json"
VERIFIER_PATH = NATIVE_ROOT / "verify_wheel_license_policy.py"
SOURCE_LOCK_PATH = NATIVE_ROOT / "source-lock.json"

EMBEDDED_COMPONENTS = {
    "openexr-half": {
        "version": "openvdb-13.0.0-embedded",
        "license": "BSD-3-Clause",
        "containing_source_id": "openvdb",
        "source_files": {
            "openvdb/openvdb/math/Half.cc": (
                "07a515b1bddd6ad51f7a918ca988461f9ad25a379b37da2e728e2e986051d293"
            ),
            "openvdb/openvdb/math/Half.h": (
                "5c6ca5d229150ceff5112bd1f63bff28578a8a1fea23956184f0c19193c6f6b4"
            ),
        },
        "license_path": "licenses/openexr-half-BSD-3-Clause.txt",
        "license_sha256": "c20236d3b39fd20eba8e3d1fb3b892a5483df2e7d8d61bf43f165d3fac22f601",
    },
    "libdivsufsort-lite": {
        "version": "zstd-1.5.6-vendored",
        "license": "MIT",
        "containing_source_id": "c-blosc",
        "source_files": {
            "internal-complibs/zstd-1.5.6/dictBuilder/divsufsort.c": (
                "2081acb08865f623857d2c0dcb0e79fce9489f01416528c30cfee7097915c616"
            ),
            "internal-complibs/zstd-1.5.6/dictBuilder/divsufsort.h": (
                "f8312544f98feb695e611c68d5191b5ae5302e01a3613e2a6f8e6cd365dd3b2d"
            ),
        },
        "license_path": "licenses/libdivsufsort-lite-MIT.txt",
        "license_sha256": "a801f489b279d91b8afc67d41e3769ec4df476fda218d63f4326c005db881541",
    },
}


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("openvdb_native_license_verifier", VERIFIER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


def _policy() -> dict[str, Any]:
    return {
        "schema": verifier.POLICY_SCHEMA,
        "scope": {
            "subject": "synthetic native driver",
            "claim": "no forbidden introduced native licenses",
            "excludes": ["external platform ABI"],
        },
        "artifact_modes": ["repaired"],
        "allowed_selected_licenses": ["Apache-2.0", "MIT"],
        "forbidden_introduced_license_families": ["LGPL", "GPL", "AGPL"],
        "runtime_pseudo_sonames": [r"^linux-vdso[.]so[.]1$"],
        "components": [
            {
                "id": "driver",
                "version": "1.0",
                "relationship": "bundled",
                "selected_license": "Apache-2.0",
                "source_ids": ["driver"],
                "runtime_location": {"repaired": "bundled"},
                "soname_patterns": [r"^libdriver[.]so[.]1$"],
                "archive_members": [
                    {
                        "pattern": r"^pkg/libdriver[.]so[.]1$",
                        "required_in": ["repaired"],
                    }
                ],
            },
            {
                "id": "helper",
                "version": "1.0",
                "relationship": "native_runtime",
                "selected_license": "MIT",
                "source_ids": ["helper"],
                "runtime_location": {"repaired": "bundled"},
                "soname_patterns": [r"^libhelper[.]so[.]1$"],
                "archive_members": [
                    {
                        "pattern": r"^pkg[.]libs/libhelper[.]so[.]1$",
                        "required_in": ["repaired"],
                    }
                ],
            },
            {
                "id": "metadata",
                "version": "1.0",
                "relationship": "metadata",
                "selected_license": "Apache-2.0",
                "source_ids": ["driver"],
                "archive_members": [
                    {
                        "pattern": r"^pkg/METADATA$",
                        "required_in": ["repaired"],
                    }
                ],
            },
        ],
        "platform_abi": [
            {
                "id": "glibc",
                "relationship": "platform_abi",
                "license": "LGPL-2.1-or-later",
                "must_be_external": True,
                "required": True,
                "rationale": "external Linux ABI",
                "provider_packages": ["libc6"],
                "soname_patterns": [r"^libc[.]so[.]6$"],
            }
        ],
    }


def _repaired_inventory():
    archive = {
        "pkg/libdriver.so.1": "d" * 64,
        "pkg.libs/libhelper.so.1": "h" * 64,
        "pkg/METADATA": "m" * 64,
    }
    wheel_elf = [
        verifier.ElfRecord(
            "pkg/libdriver.so.1",
            "d" * 64,
            "libdriver.so.1",
            ("libc.so.6", "libhelper.so.1"),
        ),
        verifier.ElfRecord(
            "pkg.libs/libhelper.so.1",
            "h" * 64,
            "libhelper.so.1",
            ("libc.so.6",),
        ),
    ]
    runtime = [
        verifier.RuntimeRecord(
            "libdriver.so.1",
            "/installed/pkg/libdriver.so.1",
            "d" * 64,
            "libdriver.so.1",
            ("libc.so.6", "libhelper.so.1"),
        ),
        verifier.RuntimeRecord(
            "libhelper.so.1",
            "/installed/pkg.libs/libhelper.so.1",
            "h" * 64,
            "libhelper.so.1",
            ("libc.so.6",),
        ),
        verifier.RuntimeRecord(
            "libc.so.6",
            "/lib/libc.so.6",
            "c" * 64,
            "libc.so.6",
            (),
            platform_provider="debian-package:libc6",
        ),
    ]
    return archive, wheel_elf, runtime


def _source_record(component_id: str, license_id: str) -> dict[str, Any]:
    return {
        "id": component_id,
        "name": component_id,
        "version": "1.0",
        "repository": f"https://example.test/{component_id}.git",
        "commit": "a" * 40,
        "archive_url": f"https://example.test/{component_id}.tar.gz",
        "archive_sha256": "b" * 64,
        "git_tree_inventory_sha256": "c" * 64,
        "license": license_id,
        "license_files": [
            {
                "source_path": "LICENSE",
                "redistribution_path": f"licenses/{component_id}.txt",
                "sha256": "d" * 64,
            }
        ],
    }


def _source_materials() -> tuple[dict[str, Any], dict[str, Any]]:
    driver = _source_record("driver", "Apache-2.0")
    helper = _source_record("helper", "MIT")
    source_lock = {
        "schema": "world-understanding.openvdb-native-source-lock.v1",
        "source": driver,
        "native_dependencies": [helper],
    }
    provenance = {
        "schema": "world-understanding.openvdb-native-dependency-provenance.v1",
        "source_lock_sha256": "e" * 64,
        "components": [
            {
                "id": source["id"],
                "version": source["version"],
                "relationship": "compiled_in",
                "license": source["license"],
                "license_files": source["license_files"],
                "source": {
                    key: source[key]
                    for key in (
                        "repository",
                        "commit",
                        "archive_sha256",
                        "git_tree_inventory_sha256",
                    )
                },
                "artifacts": [],
            }
            for source in (driver, helper)
        ],
    }
    return source_lock, provenance


def _release_materials(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_lock, provenance = _source_materials()
    source_lock["build_recipe"] = {"path": "build.sh", "sha256": "f" * 64}
    provenance["components"][0]["artifacts"] = [
        {"path": "lib/libdriver.a", "sha256": "1" * 64, "linkage": "static"}
    ]
    wheel = tmp_path / "driver-1.0-cp312-cp312-manylinux_x86_64.whl"
    wheel.write_bytes(b"promoted wheel")
    wheel_elf = [
        verifier.ElfRecord(
            record.member,
            ("e" * 64 if record.sha256 == "h" * 64 else record.sha256),
            record.soname,
            record.needed,
        )
        for record in _repaired_inventory()[1]
    ]
    source_lock_bytes = json.dumps(source_lock, sort_keys=True).encode()
    policy_bytes = json.dumps(_policy(), sort_keys=True).encode()
    verifier_bytes = b"reviewed verifier"
    provenance_bytes = json.dumps(provenance, sort_keys=True).encode()
    archive_digests = {
        "openvdb/__init__.py": "a" * 64,
        "openvdb/_source_lock.json": hashlib.sha256(source_lock_bytes).hexdigest(),
        "pkg/libdriver.so.1": wheel_elf[0].sha256,
        "pkg.libs/libhelper.so.1": wheel_elf[1].sha256,
        "driver-1.0.dist-info/METADATA": "b" * 64,
        "driver-1.0.dist-info/WHEEL": "c" * 64,
        "driver-1.0.dist-info/licenses/NATIVE_DEPENDENCY_PROVENANCE.json": (
            hashlib.sha256(provenance_bytes).hexdigest()
        ),
    }
    prepromotion_report = {
        "schema": verifier.PREPROMOTION_REPORT_SCHEMA,
        "prepromotion": {
            "passed": True,
            "admitted": False,
            "deployable": False,
            "requires_promoted_release_lock": True,
        },
        "ldd": {
            "schema": verifier.RUNTIME_CLOSURE_SCHEMA,
            "sha256": "9" * 64,
        },
        "policy": {"sha256": hashlib.sha256(policy_bytes).hexdigest()},
        "source_lock": {"sha256": hashlib.sha256(source_lock_bytes).hexdigest()},
        "native_dependency_provenance": {"sha256": hashlib.sha256(provenance_bytes).hexdigest()},
        "verifier": {"sha256": hashlib.sha256(verifier_bytes).hexdigest()},
        "wheel": {
            "filename": wheel.name,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        },
        "wheel_runtime_members": verifier._runtime_relevant_wheel_members(
            archive_digests, wheel_elf
        ),
        "wheel_elf": [
            {
                "member": record.member,
                "sha256": record.sha256,
            }
            for record in wheel_elf
        ],
    }
    prepromotion_report_bytes = json.dumps(prepromotion_report, sort_keys=True).encode()
    release_lock = {
        "schema": verifier.RELEASE_LOCK_SCHEMA,
        "trust_model": "reviewed promotion fixture",
        "platforms": {
            "x86_64": {
                "status": "promoted",
                "source_lock_sha256": hashlib.sha256(source_lock_bytes).hexdigest(),
                "build_recipe_sha256": "f" * 64,
                "license_policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
                "license_verifier_sha256": hashlib.sha256(verifier_bytes).hexdigest(),
                "native_dependency_provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
                "prepromotion_report_sha256": hashlib.sha256(prepromotion_report_bytes).hexdigest(),
                "ldd_closure_sha256": "9" * 64,
                "wheel": {
                    "filename": wheel.name,
                    "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                },
                "wheel_runtime_members": verifier._runtime_relevant_wheel_members(
                    archive_digests, wheel_elf
                ),
                "wheel_native_members": {record.member: record.sha256 for record in wheel_elf},
                "build_artifacts": {
                    "driver": provenance["components"][0]["artifacts"],
                },
            }
        },
    }
    inputs = {
        "architecture": "x86_64",
        "source_lock": source_lock,
        "source_lock_bytes": source_lock_bytes,
        "policy_bytes": policy_bytes,
        "verifier_bytes": verifier_bytes,
        "provenance": provenance,
        "provenance_bytes": provenance_bytes,
        "prepromotion_report": prepromotion_report,
        "prepromotion_report_bytes": prepromotion_report_bytes,
        "wheel_path": wheel,
        "wheel_bytes": wheel.read_bytes(),
        "archive_digests": archive_digests,
        "wheel_elf": wheel_elf,
    }
    return release_lock, inputs


def test_checked_in_policy_and_verifier_are_source_locked():
    policy = verifier.load_policy(POLICY_PATH)
    source_lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))

    assert policy["schema"] == verifier.POLICY_SCHEMA
    assert source_lock["license_admission"] == {
        "policy": {
            "path": POLICY_PATH.name,
            "schema": verifier.POLICY_SCHEMA,
            "sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
        },
        "verifier": {
            "path": VERIFIER_PATH.name,
            "report_schema": verifier.REPORT_SCHEMA,
            "sha256": hashlib.sha256(VERIFIER_PATH.read_bytes()).hexdigest(),
        },
    }


def test_checked_in_embedded_component_evidence_is_exact_and_redistributed() -> None:
    policy = verifier.load_policy(POLICY_PATH)
    source_lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    source_records = verifier._source_index(source_lock)
    policy_by_id = {component["id"]: component for component in policy["components"]}
    redistribution = {item["path"]: item["sha256"] for item in source_lock["redistribution_files"]}

    for component_id, expected in EMBEDDED_COMPONENTS.items():
        source = source_records[component_id]
        assert source["version"] == expected["version"]
        assert source["license"] == expected["license"]
        assert source["containing_source_id"] == expected["containing_source_id"]
        assert {item["path"]: item["sha256"] for item in source["source_files"]} == expected[
            "source_files"
        ]

        license_record = source["license_files"][0]
        assert license_record["redistribution_path"] == expected["license_path"]
        assert license_record["sha256"] == expected["license_sha256"]
        assert license_record["source_sha256"] in expected["source_files"].values()
        license_path = NATIVE_ROOT / expected["license_path"]
        assert hashlib.sha256(license_path.read_bytes()).hexdigest() == expected["license_sha256"]
        assert redistribution[expected["license_path"]] == expected["license_sha256"]

        component = policy_by_id[component_id]
        assert component["version"] == expected["version"]
        assert component["selected_license"] == expected["license"]
        assert component["source_ids"] == [component_id]
        assert component["archive_members"] == [
            {
                "pattern": (
                    "^openvdb-13[.]0[.]0[+]wu[.]3[.]dist-info/licenses/LICENSES/"
                    f"{Path(expected['license_path']).name.replace('.', '[.]')}$"
                ),
                "required_in": ["repaired"],
            }
        ]


@pytest.mark.parametrize("component_id", sorted(EMBEDDED_COMPONENTS))
def test_embedded_component_cannot_be_omitted_from_policy(component_id: str) -> None:
    policy = verifier.load_policy(POLICY_PATH)
    source_lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    source_records = verifier._source_index(source_lock)
    provenance_components = []
    for source in source_records.values():
        component = {
            "id": source["id"],
            "version": source["version"],
            "relationship": source.get("relationship", "compiled_in"),
            "license": source["license"],
            "license_files": source.get("license_files", []),
            "artifacts": [],
        }
        if "commit" in source:
            component["source"] = {
                key: source[key]
                for key in (
                    "repository",
                    "commit",
                    "archive_sha256",
                    "git_tree_inventory_sha256",
                )
            }
        else:
            component["containing_source_id"] = source.get("containing_source_id")
        provenance_components.append(component)
    provenance = {
        "schema": "world-understanding.openvdb-native-dependency-provenance.v1",
        "source_lock_sha256": "0" * 64,
        "components": provenance_components,
    }
    policy["components"] = [
        component for component in policy["components"] if component["id"] != component_id
    ]

    with pytest.raises(verifier.AdmissionError, match="policy source coverage mismatch"):
        verifier._validate_source_materials(policy, source_lock, provenance)


def test_source_materials_bind_every_policy_component() -> None:
    source_lock, provenance = _source_materials()

    records = verifier._validate_source_materials(_policy(), source_lock, provenance)

    assert set(records) == {"driver", "helper"}


def test_preexisting_application_dependency_is_not_part_of_introduced_provenance() -> None:
    source_lock, provenance = _source_materials()
    source_lock["application_dependencies"] = [
        {
            "id": "array-runtime",
            "name": "application-owned array runtime",
            "version": "2.0",
            "compiled_into_artifact": False,
            "license": "BSD-3-Clause",
        }
    ]

    records = verifier._validate_source_materials(_policy(), source_lock, provenance)

    assert set(records) == {"driver", "helper"}


def test_preexisting_dependency_cannot_be_smuggled_into_introduced_provenance() -> None:
    source_lock, provenance = _source_materials()
    source_lock["application_dependencies"] = [
        {
            "id": "array-runtime",
            "name": "application-owned array runtime",
            "version": "2.0",
            "compiled_into_artifact": False,
            "license": "BSD-3-Clause",
        }
    ]
    provenance["components"].append(
        {
            "id": "array-runtime",
            "version": "2.0",
            "relationship": "bundled",
            "license": "LGPL-2.1-only",
            "license_files": [],
            "artifacts": [{"path": "libquadmath.so.0", "sha256": "1" * 64}],
        }
    )

    with pytest.raises(verifier.AdmissionError, match="introduced-source inventory mismatch"):
        verifier._validate_source_materials(_policy(), source_lock, provenance)


def test_unknown_policy_source_is_rejected() -> None:
    source_lock, provenance = _source_materials()
    policy = _policy()
    policy["components"][1]["source_ids"] = ["renamed-lgpl-binary"]

    with pytest.raises(verifier.AdmissionError, match="unknown source"):
        verifier._validate_source_materials(policy, source_lock, provenance)


def test_forbidden_or_unknown_source_license_is_rejected() -> None:
    source_lock, provenance = _source_materials()
    source_lock["native_dependencies"][0]["license"] = "LGPL-2.1-only"

    with pytest.raises(verifier.AdmissionError, match="unapproved license"):
        verifier._validate_source_materials(_policy(), source_lock, provenance)


def test_platform_abi_policy_requires_provider_packages() -> None:
    policy = _policy()
    del policy["platform_abi"][0]["provider_packages"]

    with pytest.raises(verifier.AdmissionError, match="missing or unknown keys"):
        verifier.validate_policy(policy)


def test_changed_source_provenance_is_rejected() -> None:
    source_lock, provenance = _source_materials()
    provenance["components"][1]["source"]["archive_sha256"] = "f" * 64

    with pytest.raises(verifier.AdmissionError, match="changed helper source identity"):
        verifier._validate_source_materials(_policy(), source_lock, provenance)


def test_source_patch_identity_is_required_and_tamper_evident() -> None:
    source_lock, provenance = _source_materials()
    patches = [{"path": "patches/helper.patch", "sha256": "1" * 64}]
    source_lock["native_dependencies"][0]["patches"] = patches
    provenance["components"][1]["source"]["patches"] = deepcopy(patches)

    verifier._validate_source_materials(_policy(), source_lock, provenance)

    provenance["components"][1]["source"]["patches"][0]["sha256"] = "2" * 64
    with pytest.raises(verifier.AdmissionError, match="changed helper source identity"):
        verifier._validate_source_materials(_policy(), source_lock, provenance)


def test_wheel_license_files_must_match_locked_redistribution_bytes() -> None:
    source_lock, _ = _source_materials()
    archive_digests = {
        "driver-1.0.dist-info/licenses/LICENSES/driver.txt": "d" * 64,
        "driver-1.0.dist-info/licenses/LICENSES/helper.txt": "d" * 64,
    }

    verifier._verify_locked_license_files(source_lock, archive_digests)

    archive_digests["driver-1.0.dist-info/licenses/LICENSES/helper.txt"] = "e" * 64
    with pytest.raises(verifier.AdmissionError, match="helper license file .* differs"):
        verifier._verify_locked_license_files(source_lock, archive_digests)


def test_wheel_license_file_must_be_unique() -> None:
    source_lock, _ = _source_materials()
    archive_digests = {
        "driver-1.0.dist-info/licenses/driver.txt": "d" * 64,
        "driver-1.0.dist-info/licenses/LICENSES/driver.txt": "d" * 64,
        "driver-1.0.dist-info/licenses/LICENSES/helper.txt": "d" * 64,
    }

    with pytest.raises(verifier.AdmissionError, match="exactly one locked driver license"):
        verifier._verify_locked_license_files(source_lock, archive_digests)


def _vendored_source_materials() -> tuple[dict[str, Any], dict[str, Any]]:
    source_lock, provenance = _source_materials()
    source = source_lock["native_dependencies"][0]
    for key in (
        "repository",
        "commit",
        "archive_url",
        "archive_sha256",
        "git_tree_inventory_sha256",
    ):
        source.pop(key)
    source.update(
        {
            "containing_source_id": "driver",
            "containing_path": "vendor/helper",
            "source_files": [{"path": "vendor/helper/helper.cc", "sha256": "1" * 64}],
        }
    )
    recorded = provenance["components"][1]
    recorded.pop("source")
    recorded.update(
        {
            "containing_source_id": source["containing_source_id"],
            "containing_path": source["containing_path"],
            "source_files": deepcopy(source["source_files"]),
        }
    )
    return source_lock, provenance


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "https://example.invalid/helper.git"),
        ("commit", "a" * 40),
        ("archive_url", "https://example.invalid/helper.tar.gz"),
        ("archive_sha256", "b" * 64),
        ("git_tree_inventory_sha256", "c" * 64),
    ],
)
def test_vendored_source_with_partial_direct_identity_fails_closed(field: str, value: str) -> None:
    source_lock, provenance = _vendored_source_materials()
    source_lock["native_dependencies"][0][field] = value

    with pytest.raises(verifier.AdmissionError, match=r"source\[helper\]\.(?:repository|commit)"):
        verifier._validate_source_materials(_policy(), source_lock, provenance)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("containing_source_id", "renamed-parent", "containing_source_id"),
        ("containing_path", "vendor/other", "containing_path"),
        (
            "source_files",
            [{"path": "vendor/helper/helper.cc", "sha256": "2" * 64}],
            "source_files",
        ),
        ("version", "1.0-tampered", "helper version"),
        ("license", "Apache-2.0", "helper license"),
        ("license_files", [], "helper license evidence"),
    ],
)
def test_vendored_provenance_rejects_exact_field_tampering(
    field: str, replacement: Any, message: str
) -> None:
    source_lock, provenance = _vendored_source_materials()
    provenance["components"][1][field] = replacement

    with pytest.raises(verifier.AdmissionError, match=message):
        verifier._validate_source_materials(_policy(), source_lock, provenance)


@pytest.mark.parametrize(
    "field",
    [
        "containing_source_id",
        "containing_path",
        "source_files",
        "version",
        "license",
        "license_files",
    ],
)
def test_vendored_provenance_rejects_exact_field_omission(field: str) -> None:
    source_lock, provenance = _vendored_source_materials()
    provenance["components"][1].pop(field)

    with pytest.raises(verifier.AdmissionError, match="changed helper"):
        verifier._validate_source_materials(_policy(), source_lock, provenance)


def test_release_lock_binds_promoted_wheel_and_source_built_artifacts(tmp_path: Path) -> None:
    release_lock, inputs = _release_materials(tmp_path)

    selected = verifier._validate_release_lock(release_lock, **inputs)

    assert selected["status"] == "promoted"


@pytest.mark.parametrize(
    ("section", "replacement", "message"),
    [
        ("wheel", "2" * 64, "promoted release digest"),
        (
            "wheel_runtime_members",
            "3" * 64,
            "runtime-relevant wheel members",
        ),
        ("wheel_native_members", "3" * 64, "native wheel members"),
        ("prepromotion_report_sha256", "4" * 64, "prepromotion_report_sha256"),
        ("ldd_closure_sha256", "5" * 64, "ldd_closure_sha256"),
        ("build_artifacts", "4" * 64, "source-built artifacts"),
    ],
)
def test_release_lock_rejects_self_attested_artifact_changes(
    tmp_path: Path,
    section: str,
    replacement: str,
    message: str,
) -> None:
    release_lock, inputs = _release_materials(tmp_path)
    changed = deepcopy(release_lock)
    platform_record = changed["platforms"]["x86_64"]
    if section == "wheel":
        platform_record["wheel"]["sha256"] = replacement
    elif section == "wheel_runtime_members":
        first = next(iter(platform_record["wheel_runtime_members"]))
        platform_record["wheel_runtime_members"][first] = replacement
    elif section == "wheel_native_members":
        first = next(iter(platform_record["wheel_native_members"]))
        platform_record["wheel_native_members"][first] = replacement
    elif section in {"prepromotion_report_sha256", "ldd_closure_sha256"}:
        platform_record[section] = replacement
    else:
        platform_record["build_artifacts"]["driver"][0]["sha256"] = replacement

    with pytest.raises(verifier.AdmissionError, match=message):
        verifier._validate_release_lock(changed, **inputs)


def test_release_lock_rejects_unpromoted_architecture(tmp_path: Path) -> None:
    release_lock, inputs = _release_materials(tmp_path)
    release_lock["platforms"]["x86_64"]["status"] = "candidate"

    with pytest.raises(verifier.AdmissionError, match="not promoted"):
        verifier._validate_release_lock(release_lock, **inputs)


def test_bundled_tbb_must_match_source_built_auditwheel_identity() -> None:
    provenance = {
        "onetbb": {
            "artifacts": [
                {
                    "path": "lib/libtbb.so.12.16",
                    "sha256": "01234567" + "a" * 56,
                    "linkage": "shared",
                }
            ]
        }
    }
    inventory = {
        "wheel_elf": [
            {
                "component": "onetbb",
                "member": "openvdb.libs/libtbb-deadbeef.so.12.16",
            }
        ]
    }

    with pytest.raises(verifier.AdmissionError, match="source-built auditwheel identity"):
        verifier._verify_source_built_tbb(inventory, provenance)


@pytest.mark.parametrize("license_id", ["LGPL-2.1-only", "GPL-3.0-only", "AGPL-3.0-only"])
def test_forbidden_introduced_license_family_is_rejected(license_id):
    policy = _policy()
    policy["allowed_selected_licenses"].append(license_id)
    policy["components"][0]["selected_license"] = license_id

    with pytest.raises(verifier.AdmissionError, match="forbidden|unapproved|unknown"):
        verifier.validate_policy(policy)


def test_unknown_archive_member_is_rejected():
    archive, wheel_elf, runtime = _repaired_inventory()
    archive["pkg/unclassified.txt"] = "u" * 64

    with pytest.raises(verifier.AdmissionError, match="unknown wheel archive member"):
        verifier.verify_inventory(
            _policy(),
            mode="repaired",
            archive_digests=archive,
            wheel_elf=wheel_elf,
            runtime=runtime,
        )


def test_native_looking_non_elf_archive_member_is_rejected(tmp_path):
    wheel = tmp_path / "broken.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pkg/libdriver.so.1", b"not an ELF")

    with pytest.raises(verifier.AdmissionError, match="native-looking wheel member is not ELF"):
        verifier.inspect_wheel(wheel)


def test_platform_abi_temp_copy_is_not_a_trusted_package_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted = tmp_path / "base-image/libc.so.6"
    vendor = tmp_path / "vendor/libc.so.6"
    trusted.parent.mkdir()
    vendor.parent.mkdir()
    elf_bytes = Path("/bin/true").read_bytes()
    trusted.write_bytes(elf_bytes)
    vendor.write_bytes(elf_bytes)
    monkeypatch.setattr(verifier, "_elf_metadata", lambda data, label: ("libc.so.6", ()))

    def fake_dpkg_query(command: list[str], **kwargs: Any) -> SimpleNamespace:
        del kwargs
        if "--show" in command:
            return SimpleNamespace(returncode=0, stdout="ii ", stderr="")
        if "--listfiles" in command:
            return SimpleNamespace(returncode=0, stdout=f"{trusted}\n", stderr="")
        raise AssertionError(f"unexpected dpkg-query command: {command}")

    monkeypatch.setattr(verifier.subprocess, "run", fake_dpkg_query)
    trusted_ldd = tmp_path / "trusted-ldd.txt"
    trusted_ldd.write_text(f"libc.so.6 => {trusted} (0x1)\n", encoding="utf-8")
    [trusted_record] = verifier.inspect_ldd(_policy(), trusted_ldd)
    assert trusted_record.platform_provider == "debian-package:libc6"

    vendor_ldd = tmp_path / "vendor-ldd.txt"
    vendor_ldd.write_text(f"libc.so.6 => {vendor} (0x1)\n", encoding="utf-8")
    with pytest.raises(verifier.AdmissionError, match="not owned by an installed trusted"):
        verifier.inspect_ldd(_policy(), vendor_ldd)


def test_platform_abi_actual_soname_must_match_request() -> None:
    archive, wheel_elf, runtime = _repaired_inventory()
    runtime[-1] = verifier.RuntimeRecord(
        "libc.so.6",
        "/lib/libc.so.6",
        "c" * 64,
        "libm.so.6",
        (),
        platform_provider="debian-package:libc6",
    )

    with pytest.raises(verifier.AdmissionError, match="SONAME differs from request"):
        verifier.verify_inventory(
            _policy(),
            mode="repaired",
            archive_digests=archive,
            wheel_elf=wheel_elf,
            runtime=runtime,
        )


def test_unknown_elf_dependency_edge_is_rejected():
    archive, wheel_elf, runtime = _repaired_inventory()
    wheel_elf[0] = verifier.ElfRecord(
        wheel_elf[0].member,
        wheel_elf[0].sha256,
        wheel_elf[0].soname,
        ("libc.so.6", "libsurprise.so.9"),
    )

    with pytest.raises(verifier.AdmissionError, match="unknown ELF dependency or SONAME"):
        verifier.verify_inventory(
            _policy(),
            mode="repaired",
            archive_digests=archive,
            wheel_elf=wheel_elf,
            runtime=runtime,
        )


def test_platform_abi_cannot_be_bundled():
    archive, wheel_elf, runtime = _repaired_inventory()
    wheel_elf[1] = verifier.ElfRecord(
        wheel_elf[1].member,
        wheel_elf[1].sha256,
        "libc.so.6",
        (),
    )

    with pytest.raises(
        verifier.AdmissionError, match="platform ABI libc[.]so[.]6 must remain external"
    ):
        verifier.verify_inventory(
            _policy(),
            mode="repaired",
            archive_digests=archive,
            wheel_elf=wheel_elf,
            runtime=runtime,
        )


def test_missing_required_archive_member_is_rejected():
    archive, wheel_elf, runtime = _repaired_inventory()
    del archive["pkg/METADATA"]

    with pytest.raises(verifier.AdmissionError, match="missing required repaired archive member"):
        verifier.verify_inventory(
            _policy(),
            mode="repaired",
            archive_digests=archive,
            wheel_elf=wheel_elf,
            runtime=runtime,
        )


def test_repaired_inventory_is_admitted_with_complete_evidence():
    archive, wheel_elf, runtime = _repaired_inventory()

    evidence = verifier.verify_inventory(
        _policy(),
        mode="repaired",
        archive_digests=archive,
        wheel_elf=wheel_elf,
        runtime=runtime,
    )

    assert len(evidence["archive_members"]) == 3
    assert len(evidence["wheel_elf"]) == 2
    assert len(evidence["runtime_libraries"]) == 3
    assert {edge["component"] for edge in evidence["dependency_edges"]} == {
        "glibc",
        "helper",
    }
    runtime_by_soname = {row["requested_soname"]: row for row in evidence["runtime_libraries"]}
    assert runtime_by_soname["libc.so.6"]["sha256"] is None
    assert runtime_by_soname["libdriver.so.1"]["sha256"] == "d" * 64
    assert runtime_by_soname["libhelper.so.1"]["sha256"] == "h" * 64


def test_platform_abi_patch_does_not_change_promoted_runtime_evidence():
    archive, wheel_elf, runtime = _repaired_inventory()
    first = verifier.verify_inventory(
        _policy(),
        mode="repaired",
        archive_digests=archive,
        wheel_elf=wheel_elf,
        runtime=runtime,
    )
    runtime[-1] = verifier.RuntimeRecord(
        runtime[-1].requested_soname,
        runtime[-1].resolved_path,
        "f" * 64,
        runtime[-1].actual_soname,
        runtime[-1].needed,
        platform_provider=runtime[-1].platform_provider,
    )

    patched_platform = verifier.verify_inventory(
        _policy(),
        mode="repaired",
        archive_digests=archive,
        wheel_elf=wheel_elf,
        runtime=runtime,
    )

    assert patched_platform == first
    assert verifier._canonical_runtime_closure_sha256(patched_platform) == (
        verifier._canonical_runtime_closure_sha256(first)
    )


def test_raw_artifacts_are_not_admitted():
    archive, wheel_elf, runtime = _repaired_inventory()

    with pytest.raises(verifier.AdmissionError, match="unknown artifact mode"):
        verifier.verify_inventory(
            _policy(),
            mode="raw",
            archive_digests=archive,
            wheel_elf=wheel_elf,
            runtime=runtime,
        )
