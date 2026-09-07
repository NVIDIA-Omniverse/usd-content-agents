# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import shutil
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

PACKAGE_ROOT = Path(__file__).parents[1]
NATIVE_ROOT = PACKAGE_ROOT / "native"
VERIFIER_PATH = NATIVE_ROOT / "verify_wheel_license_policy.py"
CANDIDATE_PATH = NATIVE_ROOT / "generate_release_candidate.py"
RELEASE_LOCK_PATH = NATIVE_ROOT / "release-lock.json"


def _load_candidate() -> ModuleType:
    sys.path.insert(0, str(NATIVE_ROOT))
    try:
        spec = importlib.util.spec_from_file_location(
            "openvdb_native_release_candidate", CANDIDATE_PATH
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


candidate = _load_candidate()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> bytes:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return data


def _replace_wheel_member(path: Path, member: str, payload: bytes | None) -> None:
    with zipfile.ZipFile(path) as archive:
        members = {
            info.filename: archive.read(info) for info in archive.infolist() if not info.is_dir()
        }
    assert member in members
    if payload is None:
        del members[member]
    else:
        members[member] = payload
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def _host_architecture() -> str:
    aliases = {"amd64": "x86_64", "arm64": "aarch64"}
    return aliases.get(platform.machine().lower(), platform.machine().lower())


@pytest.fixture
def candidate_materials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    architecture = _host_architecture()
    if architecture not in {"aarch64", "x86_64"}:
        pytest.skip(f"native candidate fixture does not support {architecture}")
    elf_sample = Path("/bin/true")
    if (
        sys.platform != "linux"
        or not elf_sample.is_file()
        or elf_sample.read_bytes()[:4] != b"\x7fELF"
    ):
        pytest.skip("native candidate fixture requires a Linux ELF sample binary")

    recipe_path = tmp_path / "build.sh"
    recipe_path.write_bytes(b"#!/bin/sh\nset -eu\n")
    policy_path = tmp_path / "license-policy.json"
    policy = {
        "schema": candidate.admission.POLICY_SCHEMA,
        "scope": {
            "subject": "synthetic candidate driver",
            "claim": "no forbidden introduced native licenses",
            "excludes": ["external platform ABI"],
        },
        "artifact_modes": ["repaired"],
        "allowed_selected_licenses": ["Apache-2.0"],
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
                "archive_members": [{"pattern": r"^.+$", "required_in": ["repaired"]}],
            }
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
    policy_bytes = _write_json(policy_path, policy)

    license_payload = b"Synthetic Apache-2.0 driver license notice.\n"
    license_member = "driver-1.0.dist-info/licenses/driver.txt"
    license_files = [
        {
            "source_path": "LICENSE",
            "redistribution_path": "licenses/driver.txt",
            "sha256": _sha256(license_payload),
        }
    ]
    source = {
        "id": "driver",
        "name": "driver",
        "version": "1.0",
        "repository": "https://example.test/driver.git",
        "commit": "a" * 40,
        "archive_url": "https://example.test/driver.tar.gz",
        "archive_sha256": "b" * 64,
        "git_tree_inventory_sha256": "c" * 64,
        "license": "Apache-2.0",
        "license_files": license_files,
    }
    source_lock = {
        "schema": "world-understanding.openvdb-native-source-lock.v1",
        "source": source,
        "build_recipe": {
            "path": recipe_path.name,
            "sha256": _sha256(recipe_path.read_bytes()),
        },
        "license_admission": {
            "policy": {
                "path": policy_path.name,
                "schema": candidate.admission.POLICY_SCHEMA,
                "sha256": _sha256(policy_bytes),
            },
            "verifier": {
                "path": VERIFIER_PATH.name,
                "report_schema": candidate.admission.REPORT_SCHEMA,
                "sha256": _sha256(VERIFIER_PATH.read_bytes()),
            },
        },
    }
    source_lock_path = tmp_path / "source-lock.json"
    source_lock_bytes = _write_json(source_lock_path, source_lock)

    provenance = {
        "schema": "world-understanding.openvdb-native-dependency-provenance.v1",
        "source_lock_sha256": _sha256(source_lock_bytes),
        "components": [
            {
                "id": "driver",
                "version": "1.0",
                "relationship": "bundled",
                "license": "Apache-2.0",
                "license_files": license_files,
                "source": {
                    key: source[key]
                    for key in (
                        "repository",
                        "commit",
                        "archive_sha256",
                        "git_tree_inventory_sha256",
                    )
                },
                "artifacts": [
                    {
                        "path": "lib/libdriver.a",
                        "sha256": "1" * 64,
                        "linkage": "static",
                    }
                ],
            }
        ],
    }
    provenance_path = tmp_path / "native-dependency-provenance.json"
    provenance_bytes = _write_json(provenance_path, provenance)
    wheel_path = tmp_path / f"driver-1.0-cp312-cp312-manylinux_2_34_{architecture}.whl"
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("openvdb/__init__.py", b"from .driver import *\n")
        archive.writestr("openvdb/_source_lock.json", source_lock_bytes)
        archive.writestr("driver-1.0.dist-info/METADATA", b"Name: driver\n")
        archive.writestr("driver-1.0.dist-info/WHEEL", b"Wheel-Version: 1.0\n")
        archive.writestr(
            "driver-1.0.dist-info/licenses/NATIVE_DEPENDENCY_PROVENANCE.json",
            provenance_bytes,
        )
        archive.writestr(license_member, license_payload)
        archive.writestr("pkg/libdriver.so.1", elf_sample.read_bytes())

    def inspect_wheel(path: Path) -> tuple[dict[str, str], list[Any]]:
        with zipfile.ZipFile(path) as archive:
            digests = {
                name: _sha256(archive.read(name))
                for name in archive.namelist()
                if not name.endswith("/")
            }
        member = "pkg/libdriver.so.1"
        return digests, [
            candidate.admission.ElfRecord(
                member,
                digests[member],
                "libdriver.so.1",
                ("libc.so.6",),
            )
        ]

    monkeypatch.setattr(candidate.admission, "inspect_wheel", inspect_wheel)
    ldd_path = tmp_path / "ldd-closure.json"
    ldd_evidence = {
        "root": str(tmp_path / "installed-a"),
        "driver_sha256": inspect_wheel(wheel_path)[1][0].sha256,
        "driver_actual_soname": "libdriver.so.1",
        "driver_needed": ["libc.so.6"],
        "platform_sha256": "c" * 64,
        "platform_actual_soname": "libc.so.6",
    }
    _write_json(ldd_path, ldd_evidence)

    def inspect_ldd(policy: dict[str, Any], path: Path) -> list[Any]:
        del policy
        evidence = json.loads(path.read_text(encoding="utf-8"))
        root = Path(evidence["root"])
        return [
            candidate.admission.RuntimeRecord(
                "libdriver.so.1",
                str(root / "openvdb/libdriver.so.1"),
                evidence["driver_sha256"],
                evidence["driver_actual_soname"],
                tuple(evidence["driver_needed"]),
            ),
            candidate.admission.RuntimeRecord(
                "libc.so.6",
                "/lib/libc.so.6",
                evidence["platform_sha256"],
                evidence["platform_actual_soname"],
                (),
                platform_provider="debian-package:libc6",
            ),
        ]

    monkeypatch.setattr(candidate.admission, "inspect_ldd", inspect_ldd)
    monkeypatch.setattr(
        candidate.admission, "_verify_source_built_tbb", lambda inventory, provenance: None
    )
    prepromotion_report_path = tmp_path / "prepromotion-report.json"
    prepromotion_report = candidate.admission.verify_prepromotion(
        policy_path=policy_path,
        source_lock_path=source_lock_path,
        provenance_path=provenance_path,
        wheel_path=wheel_path,
        ldd_path=ldd_path,
        mode="repaired",
    )
    prepromotion_report_bytes = _write_json(prepromotion_report_path, prepromotion_report)

    return {
        "architecture": architecture,
        "policy_path": policy_path,
        "verifier_path": VERIFIER_PATH,
        "source_lock_path": source_lock_path,
        "provenance_path": provenance_path,
        "prepromotion_report_path": prepromotion_report_path,
        "wheel_path": wheel_path,
        "ldd_path": ldd_path,
        "ldd_evidence": ldd_evidence,
        "inspect_ldd": inspect_ldd,
        "source_lock": source_lock,
        "source_lock_bytes": source_lock_bytes,
        "policy_bytes": policy_bytes,
        "provenance": provenance,
        "provenance_bytes": provenance_bytes,
        "license_member": license_member,
        "license_payload": license_payload,
        "prepromotion_report": prepromotion_report,
        "prepromotion_report_bytes": prepromotion_report_bytes,
    }


def _build(materials: dict[str, Any]) -> dict[str, Any]:
    return candidate.build_candidate_record(
        architecture=materials["architecture"],
        policy_path=materials["policy_path"],
        verifier_path=materials["verifier_path"],
        source_lock_path=materials["source_lock_path"],
        provenance_path=materials["provenance_path"],
        prepromotion_report_path=materials["prepromotion_report_path"],
        wheel_path=materials["wheel_path"],
        ldd_path=materials["ldd_path"],
    )


def test_candidate_record_is_exact_deterministic_and_never_promoted(
    candidate_materials: dict[str, Any],
) -> None:
    record = _build(candidate_materials)

    assert record["status"] == "candidate"
    assert candidate.canonical_candidate_json(record) == candidate.canonical_candidate_json(
        _build(candidate_materials)
    )
    promoted = deepcopy(record)
    promoted["status"] = "promoted"
    with pytest.raises(candidate.admission.AdmissionError, match="only unpromoted"):
        candidate.canonical_candidate_json(promoted)

    runtime_members = record["wheel_runtime_members"]
    assert "openvdb/__init__.py" in runtime_members
    assert "openvdb/_source_lock.json" in runtime_members
    assert "pkg/libdriver.so.1" in runtime_members
    assert "driver-1.0.dist-info/METADATA" in runtime_members
    assert "driver-1.0.dist-info/WHEEL" in runtime_members
    assert any(path.endswith("NATIVE_DEPENDENCY_PROVENANCE.json") for path in runtime_members)
    runtime_libraries = candidate_materials["prepromotion_report"]["runtime_libraries"]
    assert all("resolved_path" not in row for row in runtime_libraries)
    assert {row["normalized_location"] for row in runtime_libraries} == {
        "wheel:pkg/libdriver.so.1",
        "platform:libc.so.6",
    }


def test_candidate_rejects_tampered_or_admission_claiming_prepromotion_report(
    candidate_materials: dict[str, Any],
) -> None:
    changed = deepcopy(candidate_materials["prepromotion_report"])
    changed["prepromotion"]["deployable"] = True
    _write_json(candidate_materials["prepromotion_report_path"], changed)

    with pytest.raises(
        candidate.admission.AdmissionError,
        match="falsely claims release admission",
    ):
        _build(candidate_materials)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        pytest.param(None, "must contain exactly one locked driver license", id="missing"),
        pytest.param(b"tampered license notice\n", "differs from source lock", id="tampered"),
    ],
)
def test_prepromotion_rejects_missing_or_tampered_locked_license_file(
    candidate_materials: dict[str, Any], replacement: bytes | None, message: str
) -> None:
    _replace_wheel_member(
        candidate_materials["wheel_path"],
        candidate_materials["license_member"],
        replacement,
    )

    with pytest.raises(candidate.admission.AdmissionError, match=message):
        candidate.admission.verify_prepromotion(
            policy_path=candidate_materials["policy_path"],
            source_lock_path=candidate_materials["source_lock_path"],
            provenance_path=candidate_materials["provenance_path"],
            wheel_path=candidate_materials["wheel_path"],
            ldd_path=candidate_materials["ldd_path"],
            mode="repaired",
        )


def test_prepromotion_rejects_declared_python_dependency(
    candidate_materials: dict[str, Any],
) -> None:
    _replace_wheel_member(
        candidate_materials["wheel_path"],
        "driver-1.0.dist-info/METADATA",
        b"Name: driver\nRequires-Dist: numpy>=1.26,<3\n",
    )

    with pytest.raises(candidate.admission.AdmissionError, match="must not declare"):
        candidate.admission.verify_prepromotion(
            policy_path=candidate_materials["policy_path"],
            source_lock_path=candidate_materials["source_lock_path"],
            provenance_path=candidate_materials["provenance_path"],
            wheel_path=candidate_materials["wheel_path"],
            ldd_path=candidate_materials["ldd_path"],
            mode="repaired",
        )


def test_candidate_is_identical_for_equivalent_closure_at_different_paths(
    candidate_materials: dict[str, Any], tmp_path: Path
) -> None:
    first = _build(candidate_materials)
    equivalent_evidence = {
        **candidate_materials["ldd_evidence"],
        "root": str(tmp_path / "completely-different-install-root"),
    }
    equivalent_ldd = tmp_path / "different-closure-filename.json"
    _write_json(equivalent_ldd, equivalent_evidence)
    equivalent_report = candidate.admission.verify_prepromotion(
        policy_path=candidate_materials["policy_path"],
        source_lock_path=candidate_materials["source_lock_path"],
        provenance_path=candidate_materials["provenance_path"],
        wheel_path=candidate_materials["wheel_path"],
        ldd_path=equivalent_ldd,
        mode="repaired",
    )
    equivalent_report_path = tmp_path / "equivalent-prepromotion.json"
    _write_json(equivalent_report_path, equivalent_report)
    equivalent_materials = {
        **candidate_materials,
        "ldd_path": equivalent_ldd,
        "prepromotion_report_path": equivalent_report_path,
    }

    assert equivalent_report == candidate_materials["prepromotion_report"]
    assert _build(equivalent_materials) == first


def test_candidate_is_identical_across_platform_abi_patch_updates(
    candidate_materials: dict[str, Any], tmp_path: Path
) -> None:
    first = _build(candidate_materials)
    patched_evidence = {
        **candidate_materials["ldd_evidence"],
        "platform_sha256": "f" * 64,
    }
    patched_ldd = tmp_path / "patched-platform-closure.json"
    _write_json(patched_ldd, patched_evidence)
    patched_report = candidate.admission.verify_prepromotion(
        policy_path=candidate_materials["policy_path"],
        source_lock_path=candidate_materials["source_lock_path"],
        provenance_path=candidate_materials["provenance_path"],
        wheel_path=candidate_materials["wheel_path"],
        ldd_path=patched_ldd,
        mode="repaired",
    )
    patched_report_path = tmp_path / "patched-platform-prepromotion.json"
    _write_json(patched_report_path, patched_report)

    assert patched_report == candidate_materials["prepromotion_report"]
    assert (
        _build(
            {
                **candidate_materials,
                "ldd_path": patched_ldd,
                "prepromotion_report_path": patched_report_path,
            }
        )
        == first
    )


def test_candidate_rejects_semantically_changed_ldd_evidence(
    candidate_materials: dict[str, Any],
) -> None:
    changed = {
        **candidate_materials["ldd_evidence"],
        "driver_sha256": "f" * 64,
    }
    _write_json(candidate_materials["ldd_path"], changed)

    with pytest.raises(
        candidate.admission.AdmissionError,
        match="runtime loaded external or modified driver",
    ):
        _build(candidate_materials)


def test_prepromotion_rejects_changed_ldd_edges(
    candidate_materials: dict[str, Any],
) -> None:
    changed = {
        **candidate_materials["ldd_evidence"],
        "driver_needed": ["libunknown.so.9"],
    }
    _write_json(candidate_materials["ldd_path"], changed)

    with pytest.raises(
        candidate.admission.AdmissionError,
        match="unknown ELF dependency or SONAME",
    ):
        _build(candidate_materials)


def test_prepromotion_detects_raw_ldd_mutation_during_inspection(
    candidate_materials: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    inspect_ldd = candidate_materials["inspect_ldd"]

    def mutating_inspection(policy: dict[str, Any], path: Path) -> list[Any]:
        records = inspect_ldd(policy, path)
        changed = {
            **candidate_materials["ldd_evidence"],
            "root": "/mutated-during-inspection",
        }
        _write_json(path, changed)
        return records

    monkeypatch.setattr(candidate.admission, "inspect_ldd", mutating_inspection)

    with pytest.raises(
        candidate.admission.AdmissionError,
        match="ldd closure changed during native verification",
    ):
        candidate.admission.verify_prepromotion(
            policy_path=candidate_materials["policy_path"],
            source_lock_path=candidate_materials["source_lock_path"],
            provenance_path=candidate_materials["provenance_path"],
            wheel_path=candidate_materials["wheel_path"],
            ldd_path=candidate_materials["ldd_path"],
            mode="repaired",
        )


def test_candidate_is_rejected_until_independently_promoted(
    candidate_materials: dict[str, Any],
) -> None:
    record = _build(candidate_materials)
    _, wheel_elf = candidate.admission.inspect_wheel(candidate_materials["wheel_path"])
    inputs = {
        "architecture": candidate_materials["architecture"],
        "source_lock": candidate_materials["source_lock"],
        "source_lock_bytes": candidate_materials["source_lock_bytes"],
        "policy_bytes": candidate_materials["policy_bytes"],
        "verifier_bytes": VERIFIER_PATH.read_bytes(),
        "provenance": candidate_materials["provenance"],
        "provenance_bytes": candidate_materials["provenance_bytes"],
        "prepromotion_report": candidate_materials["prepromotion_report"],
        "prepromotion_report_bytes": candidate_materials["prepromotion_report_bytes"],
        "wheel_path": candidate_materials["wheel_path"],
        "wheel_bytes": candidate_materials["wheel_path"].read_bytes(),
        "archive_digests": candidate.admission.inspect_wheel(candidate_materials["wheel_path"])[0],
        "wheel_elf": wheel_elf,
    }
    release_lock = {
        "schema": candidate.admission.RELEASE_LOCK_SCHEMA,
        "trust_model": "synthetic review",
        "platforms": {candidate_materials["architecture"]: record},
    }

    with pytest.raises(candidate.admission.AdmissionError, match="not promoted"):
        candidate.admission._validate_release_lock(release_lock, **inputs)

    reviewed = deepcopy(release_lock)
    reviewed["platforms"][candidate_materials["architecture"]]["status"] = "promoted"
    assert candidate.admission._validate_release_lock(reviewed, **inputs)["status"] == "promoted"


def test_final_admission_recomputes_exact_prepromotion_evidence(
    candidate_materials: dict[str, Any], tmp_path: Path
) -> None:
    record = _build(candidate_materials)
    record["status"] = "promoted"
    release_lock_path = tmp_path / "reviewed-release-lock.json"
    _write_json(
        release_lock_path,
        {
            "schema": candidate.admission.RELEASE_LOCK_SCHEMA,
            "trust_model": "synthetic independent review",
            "platforms": {candidate_materials["architecture"]: record},
        },
    )

    report = candidate.admission.verify_wheel(
        policy_path=candidate_materials["policy_path"],
        release_lock_path=release_lock_path,
        prepromotion_report_path=candidate_materials["prepromotion_report_path"],
        source_lock_path=candidate_materials["source_lock_path"],
        provenance_path=candidate_materials["provenance_path"],
        wheel_path=candidate_materials["wheel_path"],
        ldd_path=candidate_materials["ldd_path"],
        mode="repaired",
        architecture=candidate_materials["architecture"],
    )

    assert report["schema"] == candidate.admission.REPORT_SCHEMA
    assert report["admission"]["passed"] is True
    assert report["release_lock"]["status"] == "promoted"
    assert report["prepromotion_report"]["admitted"] is False

    changed = {
        **candidate_materials["ldd_evidence"],
        "driver_actual_soname": "libunexpected.so.1",
    }
    _write_json(candidate_materials["ldd_path"], changed)
    with pytest.raises(
        candidate.admission.AdmissionError,
        match="unknown ELF dependency or SONAME",
    ):
        candidate.admission.verify_wheel(
            policy_path=candidate_materials["policy_path"],
            release_lock_path=release_lock_path,
            prepromotion_report_path=candidate_materials["prepromotion_report_path"],
            source_lock_path=candidate_materials["source_lock_path"],
            provenance_path=candidate_materials["provenance_path"],
            wheel_path=candidate_materials["wheel_path"],
            ldd_path=candidate_materials["ldd_path"],
            mode="repaired",
            architecture=candidate_materials["architecture"],
        )


def test_candidate_rejects_unsupported_or_mislabeled_architecture(
    candidate_materials: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(candidate.admission.AdmissionError, match="must be one of"):
        candidate.build_candidate_record(
            architecture="amd64",
            policy_path=candidate_materials["policy_path"],
            verifier_path=candidate_materials["verifier_path"],
            source_lock_path=candidate_materials["source_lock_path"],
            provenance_path=candidate_materials["provenance_path"],
            prepromotion_report_path=candidate_materials["prepromotion_report_path"],
            wheel_path=candidate_materials["wheel_path"],
            ldd_path=candidate_materials["ldd_path"],
        )

    wrong_architecture = "aarch64" if candidate_materials["architecture"] == "x86_64" else "x86_64"
    mislabeled = tmp_path / f"driver-1.0-cp312-cp312-manylinux_2_34_{wrong_architecture}.whl"
    shutil.copyfile(candidate_materials["wheel_path"], mislabeled)
    changed = {**candidate_materials, "architecture": wrong_architecture, "wheel_path": mislabeled}
    with pytest.raises(candidate.admission.AdmissionError, match="ELF architecture mismatch"):
        _build(changed)


def test_cli_emits_only_canonical_candidate_and_does_not_touch_release_lock(
    candidate_materials: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    release_lock_before = RELEASE_LOCK_PATH.read_bytes()
    result = candidate.main(
        [
            "--architecture",
            candidate_materials["architecture"],
            "--policy",
            str(candidate_materials["policy_path"]),
            "--verifier",
            str(candidate_materials["verifier_path"]),
            "--source-lock",
            str(candidate_materials["source_lock_path"]),
            "--provenance",
            str(candidate_materials["provenance_path"]),
            "--prepromotion-report",
            str(candidate_materials["prepromotion_report_path"]),
            "--wheel",
            str(candidate_materials["wheel_path"]),
            "--ldd",
            str(candidate_materials["ldd_path"]),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert json.loads(captured.out)["status"] == "candidate"
    assert captured.out == candidate.canonical_candidate_json(json.loads(captured.out))
    assert RELEASE_LOCK_PATH.read_bytes() == release_lock_before


def test_cli_reports_a_corrupt_wheel_without_a_traceback(
    candidate_materials: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    candidate_materials["wheel_path"].write_bytes(b"not a zip archive")

    result = candidate.main(
        [
            "--architecture",
            candidate_materials["architecture"],
            "--policy",
            str(candidate_materials["policy_path"]),
            "--verifier",
            str(candidate_materials["verifier_path"]),
            "--source-lock",
            str(candidate_materials["source_lock_path"]),
            "--provenance",
            str(candidate_materials["provenance_path"]),
            "--prepromotion-report",
            str(candidate_materials["prepromotion_report_path"]),
            "--wheel",
            str(candidate_materials["wheel_path"]),
            "--ldd",
            str(candidate_materials["ldd_path"]),
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err.startswith("native release candidate generation failed:")
    assert "Traceback" not in captured.err


def test_verifier_cli_keeps_prepromotion_visibly_non_deployable(
    candidate_materials: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "prepromotion.json"
    common = [
        "--policy",
        str(candidate_materials["policy_path"]),
        "--source-lock",
        str(candidate_materials["source_lock_path"]),
        "--provenance",
        str(candidate_materials["provenance_path"]),
        "--wheel",
        str(candidate_materials["wheel_path"]),
        "--ldd",
        str(candidate_materials["ldd_path"]),
        "--artifact-mode",
        "repaired",
        "--output",
        str(output),
    ]

    assert candidate.admission.main(["--prepromotion-only", *common]) == 0
    captured = capsys.readouterr()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert "remains unadmitted and non-deployable" in captured.out
    assert report["schema"] == candidate.admission.PREPROMOTION_REPORT_SCHEMA
    assert report["prepromotion"]["passed"] is True
    assert report["prepromotion"]["admitted"] is False
    assert report["prepromotion"]["deployable"] is False

    missing_admission_output = tmp_path / "must-not-exist.json"
    failed_common = [
        *common[:-1],
        str(missing_admission_output),
    ]
    assert candidate.admission.main(failed_common) == 1
    captured = capsys.readouterr()
    assert "final admission requires" in captured.err
    assert not missing_admission_output.exists()
