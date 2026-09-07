#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit read-only candidate evidence for native release review."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any

import verify_wheel_license_policy as admission

_SUPPORTED_ARCHITECTURES = {"aarch64", "x86_64"}
_ELF_MACHINE_ARCHITECTURES = {
    62: "x86_64",
    183: "aarch64",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact_architecture(value: str) -> str:
    if value not in _SUPPORTED_ARCHITECTURES:
        raise admission.AdmissionError(
            f"candidate architecture must be one of {sorted(_SUPPORTED_ARCHITECTURES)}: {value!r}"
        )
    return value


def _safe_relative_path(value: Any, label: str) -> pathlib.PurePosixPath:
    text = admission._required_text(value, label)
    path = pathlib.PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text:
        raise admission.AdmissionError(f"{label} must be a safe relative path")
    return path


def _validate_bound_inputs(
    *,
    policy_path: pathlib.Path,
    verifier_path: pathlib.Path,
    source_lock_path: pathlib.Path,
    provenance_path: pathlib.Path,
) -> tuple[
    dict[str, Any],
    bytes,
    bytes,
    bytes,
    dict[str, Any],
    bytes,
]:
    checked_verifier = pathlib.Path(admission.__file__).resolve()
    if verifier_path.resolve() != checked_verifier:
        raise admission.AdmissionError(
            "candidate generation must use the verifier executing this inspection"
        )

    policy_bytes = policy_path.read_bytes()
    policy = admission.load_policy(policy_path)
    verifier_bytes = verifier_path.read_bytes()
    source_lock_bytes = source_lock_path.read_bytes()
    source_lock = admission._load_json_object(source_lock_path, "native source lock")
    provenance_bytes = provenance_path.read_bytes()
    provenance = admission._load_json_object(provenance_path, "native dependency provenance")
    admission._validate_source_materials(policy, source_lock, provenance)

    if provenance.get("source_lock_sha256") != _sha256(source_lock_bytes):
        raise admission.AdmissionError("native dependency provenance source-lock digest mismatch")

    recipe = source_lock.get("build_recipe")
    if not isinstance(recipe, dict) or set(recipe) != {"path", "sha256"}:
        raise admission.AdmissionError("source lock must bind the exact native build recipe")
    recipe_relative = _safe_relative_path(recipe.get("path"), "build_recipe.path")
    recipe_path = source_lock_path.parent.joinpath(*recipe_relative.parts)
    recipe_digest = recipe.get("sha256")
    if (
        not isinstance(recipe_digest, str)
        or not admission._SHA256.fullmatch(recipe_digest)
        or _sha256(recipe_path.read_bytes()) != recipe_digest
    ):
        raise admission.AdmissionError("native build recipe differs from source lock")

    bindings = source_lock.get("license_admission")
    if not isinstance(bindings, dict) or set(bindings) != {"policy", "verifier"}:
        raise admission.AdmissionError("source lock must bind the native policy and verifier")
    expected = {
        "policy": {
            "path": policy_path.name,
            "schema": admission.POLICY_SCHEMA,
            "sha256": _sha256(policy_bytes),
        },
        "verifier": {
            "path": verifier_path.name,
            "report_schema": admission.REPORT_SCHEMA,
            "sha256": _sha256(verifier_bytes),
        },
    }
    if bindings != expected:
        raise admission.AdmissionError("source-lock native policy or verifier binding mismatch")
    return (
        source_lock,
        source_lock_bytes,
        policy_bytes,
        verifier_bytes,
        provenance,
        provenance_bytes,
    )


def _validate_repaired_wheel_name(wheel_path: pathlib.Path, architecture: str) -> None:
    if wheel_path.suffix != ".whl":
        raise admission.AdmissionError("candidate input must be a wheel")
    platform_tag = wheel_path.name.removesuffix(".whl").rsplit("-", 1)[-1]
    tags = platform_tag.split(".")
    expected_suffix = f"_{architecture}"
    if not tags or any(
        not tag.startswith("manylinux_") or not tag.endswith(expected_suffix) for tag in tags
    ):
        raise admission.AdmissionError(
            f"candidate input is not a repaired manylinux wheel for {architecture}"
        )


def _validate_wheel_architecture(
    wheel_path: pathlib.Path,
    wheel_elf: Sequence[admission.ElfRecord],
    architecture: str,
) -> None:
    machines: set[str] = set()
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            for record in wheel_elf:
                data = archive.read(record.member)
                if len(data) < 20 or data[:4] != b"\x7fELF" or data[5] not in {1, 2}:
                    raise admission.AdmissionError(
                        f"could not read ELF architecture from {record.member}"
                    )
                byteorder = "little" if data[5] == 1 else "big"
                machine = int.from_bytes(data[18:20], byteorder=byteorder)
                actual = _ELF_MACHINE_ARCHITECTURES.get(machine)
                if actual is None:
                    raise admission.AdmissionError(
                        f"unsupported ELF architecture {machine!r} in {record.member}"
                    )
                machines.add(actual)
    except (KeyError, zipfile.BadZipFile) as exc:
        raise admission.AdmissionError(f"could not verify wheel ELF architecture: {exc}") from exc
    if machines != {architecture}:
        raise admission.AdmissionError(
            f"wheel ELF architecture mismatch: expected {architecture}, found {sorted(machines)}"
        )


def _validate_embedded_inputs(
    wheel_path: pathlib.Path,
    source_lock_bytes: bytes,
    provenance_bytes: bytes,
) -> None:
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            if archive.read("openvdb/_source_lock.json") != source_lock_bytes:
                raise admission.AdmissionError(
                    "wheel embedded source lock differs from candidate input"
                )
            names = [
                name
                for name in archive.namelist()
                if pathlib.PurePosixPath(name).name == "NATIVE_DEPENDENCY_PROVENANCE.json"
            ]
            if len(names) != 1 or archive.read(names[0]) != provenance_bytes:
                raise admission.AdmissionError(
                    "wheel embedded native dependency provenance differs from candidate input"
                )
    except (KeyError, zipfile.BadZipFile) as exc:
        raise admission.AdmissionError(
            f"could not verify wheel embedded candidate inputs: {exc}"
        ) from exc


def _build_artifacts(provenance: Mapping[str, Any]) -> dict[str, Any]:
    components = provenance.get("components")
    if not isinstance(components, list):
        raise admission.AdmissionError("native dependency provenance components must be a list")
    result: dict[str, Any] = {}
    for component in components:
        if not isinstance(component, dict):
            raise admission.AdmissionError(
                "native dependency provenance component must be an object"
            )
        artifacts = component.get("artifacts")
        if artifacts:
            component_id = admission._required_text(component.get("id"), "provenance component id")
            if component_id in result:
                raise admission.AdmissionError(
                    f"duplicate provenance artifact owner: {component_id}"
                )
            if not isinstance(artifacts, list) or any(
                not isinstance(artifact, dict) or not artifact for artifact in artifacts
            ):
                raise admission.AdmissionError(
                    f"provenance artifacts for {component_id} must be non-empty objects"
                )
            result[component_id] = artifacts
    if not result:
        raise admission.AdmissionError("native dependency provenance contains no build artifacts")
    return result


def build_candidate_record(
    *,
    architecture: str,
    policy_path: pathlib.Path,
    verifier_path: pathlib.Path,
    source_lock_path: pathlib.Path,
    provenance_path: pathlib.Path,
    prepromotion_report_path: pathlib.Path,
    wheel_path: pathlib.Path,
    ldd_path: pathlib.Path,
    mode: str = "repaired",
) -> dict[str, Any]:
    """Build an exact, unadmitted platform record for human review."""

    architecture = _exact_architecture(architecture)
    _validate_repaired_wheel_name(wheel_path, architecture)
    try:
        archive_digests, wheel_elf = admission.inspect_wheel(wheel_path)
    except zipfile.BadZipFile as exc:
        raise admission.AdmissionError(f"could not inspect candidate wheel: {exc}") from exc
    _validate_wheel_architecture(wheel_path, wheel_elf, architecture)
    (
        source_lock,
        source_lock_bytes,
        policy_bytes,
        verifier_bytes,
        provenance,
        provenance_bytes,
    ) = _validate_bound_inputs(
        policy_path=policy_path,
        verifier_path=verifier_path,
        source_lock_path=source_lock_path,
        provenance_path=provenance_path,
    )
    recomputed = admission.verify_prepromotion(
        policy_path=policy_path,
        source_lock_path=source_lock_path,
        provenance_path=provenance_path,
        wheel_path=wheel_path,
        ldd_path=ldd_path,
        mode=mode,
    )
    prepromotion_report, prepromotion_report_bytes = admission.load_verified_prepromotion_report(
        prepromotion_report_path, expected=recomputed
    )
    _validate_embedded_inputs(wheel_path, source_lock_bytes, provenance_bytes)
    archive_digests, wheel_elf = admission.inspect_wheel(wheel_path)
    _validate_wheel_architecture(wheel_path, wheel_elf, architecture)
    wheel_bytes = wheel_path.read_bytes()
    if prepromotion_report.get("wheel") != {
        "filename": wheel_path.name,
        "sha256": _sha256(wheel_bytes),
    }:
        raise admission.AdmissionError("candidate wheel changed after prepromotion verification")
    runtime_members = admission._runtime_relevant_wheel_members(archive_digests, wheel_elf)
    if prepromotion_report.get("wheel_runtime_members") != runtime_members:
        raise admission.AdmissionError(
            "candidate runtime-member inventory changed after prepromotion verification"
        )
    recipe = source_lock["build_recipe"]
    record = {
        "status": "candidate",
        "source_lock_sha256": _sha256(source_lock_bytes),
        "build_recipe_sha256": recipe["sha256"],
        "license_policy_sha256": _sha256(policy_bytes),
        "license_verifier_sha256": _sha256(verifier_bytes),
        "native_dependency_provenance_sha256": _sha256(provenance_bytes),
        "prepromotion_report_sha256": _sha256(prepromotion_report_bytes),
        "ldd_closure_sha256": prepromotion_report["ldd"]["sha256"],
        "wheel": {
            "filename": wheel_path.name,
            "sha256": _sha256(wheel_bytes),
        },
        "wheel_runtime_members": runtime_members,
        "wheel_native_members": {
            item.member: item.sha256 for item in sorted(wheel_elf, key=lambda item: item.member)
        },
        "build_artifacts": _build_artifacts(provenance),
    }
    if set(record) != admission.PLATFORM_RECORD_KEYS or record["status"] != "candidate":
        raise admission.AdmissionError("candidate platform record is malformed")
    return record


def canonical_candidate_json(record: Mapping[str, Any]) -> str:
    """Serialize candidate evidence without permitting a promoted status."""

    if set(record) != admission.PLATFORM_RECORD_KEYS or record.get("status") != "candidate":
        raise admission.AdmissionError("only unpromoted candidate records may be emitted")
    return admission._canonical_json(record) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=sorted(_SUPPORTED_ARCHITECTURES))
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--verifier", required=True, type=pathlib.Path)
    parser.add_argument("--source-lock", required=True, type=pathlib.Path)
    parser.add_argument("--provenance", required=True, type=pathlib.Path)
    parser.add_argument("--prepromotion-report", required=True, type=pathlib.Path)
    parser.add_argument("--wheel", required=True, type=pathlib.Path)
    parser.add_argument("--ldd", required=True, type=pathlib.Path)
    parser.add_argument("--artifact-mode", choices=("repaired",), default="repaired")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        record = build_candidate_record(
            architecture=args.architecture,
            policy_path=args.policy,
            verifier_path=args.verifier,
            source_lock_path=args.source_lock,
            provenance_path=args.provenance,
            prepromotion_report_path=args.prepromotion_report,
            wheel_path=args.wheel,
            ldd_path=args.ldd,
            mode=args.artifact_mode,
        )
        sys.stdout.write(canonical_candidate_json(record))
    except (admission.AdmissionError, OSError) as exc:
        print(f"native release candidate generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
