# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish exact-plan OpenUSD collision-filter derivatives.

Plans bind exact source and evidence bytes. Authoring runs in a private package
copy, preserves unrelated source opinions and dependency bytes, then publishes a
content-addressed tree plus a plan-scoped receipt. Expected input and OpenUSD
failures return ``BLOCKED``; programming and control-flow exceptions propagate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from world_understanding.utils.usd.package import safe_usdz_member_parts

COLLISION_FILTER_PLAN_SCHEMA_VERSION = (
    "content-agent-workflows.simready-collision-filter-plan.v1"
)
COLLISION_FILTER_RECEIPT_SCHEMA_VERSION = (
    "content-agent-workflows.simready-collision-filter-receipt.v1"
)
COLLISION_FILTER_OUTPUT_DIR = "collision-filtered"
COLLISION_FILTER_REQUIREMENT = "COL.FILTER.001"

_USD_LAYER_SUFFIXES = {".usd", ".usda", ".usdc"}
_FILTER_API_NAME = "PhysicsFilteredPairsAPI"
_FILTER_RELATIONSHIP_NAME = "physics:filteredPairs"
_MAX_PACKAGE_PATH_DEPTH = 256


class _Blocked(ValueError):
    """Expected failure mapped to ``BLOCKED`` by the authoring entrypoint."""


_StrictNonEmptyString = Annotated[
    str,
    StringConstraints(strict=True, min_length=1),
]
_ReadableStrictString = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1),
]
_Sha256 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]


class CollisionFilterEvidence(BaseModel):
    """One immutable machine-evidence artifact referenced by a filter plan."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    kind: Literal["machine_collision_preflight", "gate3a_validation"]
    artifact_path: _StrictNonEmptyString
    artifact_sha256: _Sha256

    @field_validator("artifact_path")
    @classmethod
    def validate_absolute_artifact_path(cls, value: str) -> str:
        """Require evidence to identify exact local machine-output bytes."""

        if not Path(value).is_absolute():
            raise _Blocked("artifact_path must be absolute")
        return value


class CollisionFilterPlanProvenance(BaseModel):
    """Machine evidence and the owner approval required to suppress collisions."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    approved_by: _ReadableStrictString
    approval_reference: _ReadableStrictString
    evidence: Annotated[list[CollisionFilterEvidence], Field(min_length=1)]


class CollisionFilterPair(BaseModel):
    """One unordered pair of exact rigid-body collider prim paths."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    body_a_path: _StrictNonEmptyString
    body_b_path: _StrictNonEmptyString

    @model_validator(mode="after")
    def validate_distinct_paths(self) -> CollisionFilterPair:
        """Reject self-pairs before opening the USD stage."""

        if self.body_a_path == self.body_b_path:
            raise _Blocked("collision-filter pair cannot reference one body twice")
        return self

    def canonical_paths(self) -> tuple[str, str]:
        """Return the one-way OpenUSD representation for this unordered pair."""

        first, second = sorted((self.body_a_path, self.body_b_path))
        return first, second


class CollisionFilterPlan(BaseModel):
    """Strict collision-filter plan bound to exact source bytes and a source path."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["content-agent-workflows.simready-collision-filter-plan.v1"]
    source_asset_path: _StrictNonEmptyString
    source_asset_sha256: _Sha256
    provenance: CollisionFilterPlanProvenance
    pairs: Annotated[list[CollisionFilterPair], Field(min_length=1)]

    @field_validator("source_asset_path")
    @classmethod
    def validate_absolute_source_path(cls, value: str) -> str:
        """Require plans to name one exact filesystem source."""

        if not Path(value).is_absolute():
            raise _Blocked("source_asset_path must be absolute")
        return value

    @model_validator(mode="after")
    def validate_unique_pairs(self) -> CollisionFilterPlan:
        """Reject duplicate unordered pairs, including reversed duplicates."""

        pairs = [pair.canonical_paths() for pair in self.pairs]
        if len(pairs) != len(set(pairs)):
            raise _Blocked("pairs contains a duplicate unordered body pair")
        return self


@dataclass(frozen=True)
class CollisionFilterResult:
    """Result of applying an exact collision-filter plan."""

    status: str
    passed: bool
    reason: str
    output_path: Path
    receipt_path: Path | None
    report: dict[str, Any]


@dataclass(frozen=True)
class _ListOpState:
    is_explicit: bool
    explicit: tuple[str, ...]
    prepended: tuple[str, ...]
    appended: tuple[str, ...]
    added: tuple[str, ...]
    deleted: tuple[str, ...]
    ordered: tuple[str, ...]


def author_collision_filter_derivative(
    *,
    asset_path: Path,
    plan_path: Path,
    output_dir: Path,
    package_root: Path | None = None,
) -> CollisionFilterResult:
    """Publish a deterministic package derivative containing exact filter pairs.

    ``package_root`` identifies the complete package tree for non-USDZ input. It
    defaults to the source layer's parent. USDZ inputs are streamed to a private
    extraction tree without an arbitrary member-size or total-size limit.

    The OpenUSD ``pxr`` modules are imported when this function is invoked so
    importing the workflow contracts does not initialize OpenUSD. A missing
    OpenUSD installation raises ``ImportError`` instead of returning ``BLOCKED``.
    """

    from pxr import Sdf, Tf, Usd, UsdPhysics, UsdUtils

    report: dict[str, Any] = {
        "schema_version": COLLISION_FILTER_RECEIPT_SCHEMA_VERSION,
        "requirement": COLLISION_FILTER_REQUIREMENT,
        "asset_path": str(asset_path),
        "plan_path": str(plan_path),
        "changes": [],
    }
    source_tree: Path | None = None
    source_root: Path | None = None
    extraction_dir: Path | None = None
    build_dir: Path | None = None
    try:
        requested_output_dir = Path(output_dir).expanduser().absolute()
        plan_path = _regular_file(plan_path, label="collision-filter plan")
        plan, plan_sha256 = _load_plan(plan_path)
        asset_path = _regular_file(asset_path, label="source asset")
        report.update(
            {
                "asset_path": str(asset_path),
                "plan_path": str(plan_path),
            }
        )
        if str(asset_path) != plan.source_asset_path:
            raise _Blocked(
                "Plan source_asset_path does not match the exact source path: "
                f"expected {asset_path}, received {plan.source_asset_path}."
            )
        source_asset_sha256 = _file_sha256(asset_path)
        if source_asset_sha256 != plan.source_asset_sha256:
            raise _Blocked(
                "Plan source_asset_sha256 is stale: expected "
                f"{source_asset_sha256}, received {plan.source_asset_sha256}."
            )
        evidence_identities = _validate_evidence_artifacts(plan.provenance.evidence)
        if asset_path.suffix.lower() != ".usdz":
            regular_package_root = _resolve_package_root(
                asset_path=asset_path,
                package_root=package_root,
            )
            if _relative_to(requested_output_dir, regular_package_root) is not None:
                raise _Blocked(
                    "Collision-filter output cannot be located inside the source "
                    "package."
                )
            package_root = regular_package_root
        output_dir = _prepare_output_dir(requested_output_dir)

        source_tree, source_root, extraction_dir = _source_package(
            asset_path=asset_path,
            package_root=package_root,
            output_dir=output_dir,
        )
        publish_root = output_dir / COLLISION_FILTER_OUTPUT_DIR
        if _relative_to(publish_root, source_tree) is not None:
            raise _Blocked(
                "Collision-filter output cannot be located inside the source package."
            )

        source_tree_sha256 = _tree_sha256(source_tree)
        source_inventory = _tree_file_sha256(source_tree)
        source_relative_root = source_root.relative_to(source_tree)
        source_was_usdz = extraction_dir is not None
        source_package_identity = (
            str(asset_path) if source_was_usdz else str(source_tree)
        )
        source_root_identity = (
            f"{asset_path}[{source_relative_root.as_posix()}]"
            if source_was_usdz
            else str(source_root)
        )
        source_stage = Usd.Stage.Open(str(source_root), load=Usd.Stage.LoadAll)
        if source_stage is None:
            raise _Blocked(f"Unable to open collision-filter source USD: {source_root}")
        ignored_identity_paths = _validate_dependency_closure(
            stage=source_stage,
            source_root=source_root,
            source_tree=source_tree,
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        canonical_pairs = _validate_plan_against_stage(
            stage=source_stage,
            plan=plan,
            Sdf=Sdf,
            UsdPhysics=UsdPhysics,
        )
        endpoint_paths = tuple(
            sorted({path for pair in canonical_pairs for path in pair})
        )
        initial_targets = _filtered_targets(
            stage=source_stage,
            endpoint_paths=endpoint_paths,
            Sdf=Sdf,
            UsdPhysics=UsdPhysics,
        )
        desired_targets = _canonical_target_map(
            initial_targets=initial_targets,
            canonical_pairs=canonical_pairs,
        )
        source_root_layer_fingerprint = _root_layer_fingerprint_without_filters(
            stage=source_stage,
            endpoint_paths=endpoint_paths,
            Sdf=Sdf,
        )
        report.update(
            {
                "plan_schema_version": plan.schema_version,
                "plan_sha256": plan_sha256,
                "source_asset_path": str(asset_path),
                "source_asset_sha256": source_asset_sha256,
                "source_package_root": source_package_identity,
                "source_root": source_root_identity,
                "source_root_relative_path": source_relative_root.as_posix(),
                "source_tree_sha256": source_tree_sha256,
                "source_was_usdz": source_was_usdz,
                "canonical_representation": "one_way_lexicographic",
                "canonical_pairs": [
                    {"source_body_path": source, "target_body_path": target}
                    for source, target in canonical_pairs
                ],
                "provenance": plan.provenance.model_dump(mode="json"),
                "ignored_unresolved_asset_identity_paths": list(ignored_identity_paths),
            }
        )
        # OpenUSD has no explicit Stage.close(); delete the final Python
        # reference and collect wrapper cycles before copying package files.
        del source_stage
        gc.collect()

        publish_root.mkdir(parents=True, exist_ok=True)
        if publish_root.is_symlink() or not publish_root.is_dir():
            raise _Blocked(
                f"Collision-filter publish root is not a regular directory: {publish_root}"
            )
        build_dir = Path(
            tempfile.mkdtemp(prefix=".collision-filter-build-", dir=publish_root)
        )
        _copy_package_tree(source_tree=source_tree, build_dir=build_dir)
        if _tree_sha256(build_dir) != source_tree_sha256:
            raise _Blocked("Collision-filter package copy changed source identity.")

        build_root = build_dir / source_relative_root
        _make_owner_writable(build_root)
        build_stage = Usd.Stage.Open(str(build_root), load=Usd.Stage.LoadAll)
        if build_stage is None:
            raise _Blocked(f"Unable to open collision-filter build USD: {build_root}")
        build_pairs = _validate_plan_against_stage(
            stage=build_stage,
            plan=plan,
            Sdf=Sdf,
            UsdPhysics=UsdPhysics,
        )
        if build_pairs != canonical_pairs:
            raise _Blocked("Collision-filter package copy changed the planned bodies.")
        build_initial_targets = _filtered_targets(
            stage=build_stage,
            endpoint_paths=endpoint_paths,
            Sdf=Sdf,
            UsdPhysics=UsdPhysics,
        )
        if build_initial_targets != initial_targets:
            raise _Blocked(
                "Collision-filter package copy changed existing filter relationships."
            )
        changes, expected_api_list_ops, expected_target_list_ops = (
            _author_filtered_pairs(
                stage=build_stage,
                endpoint_paths=endpoint_paths,
                canonical_pairs=canonical_pairs,
                desired_targets=desired_targets,
                Sdf=Sdf,
                UsdPhysics=UsdPhysics,
            )
        )
        if not build_stage.GetRootLayer().Save():
            raise OSError(f"Could not save collision-filter layer: {build_root}")
        del build_stage
        gc.collect()

        output_stage = Usd.Stage.Open(str(build_root), load=Usd.Stage.LoadAll)
        if output_stage is None:
            raise _Blocked(
                f"Unable to open authored collision-filter USD: {build_root}"
            )
        _validate_dependency_closure(
            stage=output_stage,
            source_root=build_root,
            source_tree=build_dir,
            Sdf=Sdf,
            UsdUtils=UsdUtils,
        )
        _verify_filtered_pairs(
            stage=output_stage,
            endpoint_paths=endpoint_paths,
            canonical_pairs=canonical_pairs,
            desired_targets=desired_targets,
            expected_api_list_ops=expected_api_list_ops,
            expected_target_list_ops=expected_target_list_ops,
            Sdf=Sdf,
            UsdPhysics=UsdPhysics,
        )
        output_root_layer_fingerprint = _root_layer_fingerprint_without_filters(
            stage=output_stage,
            endpoint_paths=endpoint_paths,
            Sdf=Sdf,
        )
        if output_root_layer_fingerprint != source_root_layer_fingerprint:
            raise _Blocked(
                "Collision-filter authoring changed root-layer content outside the exact "
                "PhysicsFilteredPairsAPI plan."
            )
        del output_stage
        gc.collect()

        output_inventory = _tree_file_sha256(build_dir)
        if set(output_inventory) != set(source_inventory):
            raise _Blocked("Collision-filter authoring changed package file inventory.")
        changed_unowned_files = sorted(
            relative
            for relative, digest in source_inventory.items()
            if relative != source_relative_root.as_posix()
            and output_inventory[relative] != digest
        )
        if changed_unowned_files:
            raise _Blocked(
                "Collision-filter authoring changed package dependencies: "
                + ", ".join(changed_unowned_files[:5])
            )

        _verify_source_unchanged(
            asset_path=asset_path,
            source_asset_sha256=source_asset_sha256,
            source_tree=source_tree,
            source_tree_sha256=source_tree_sha256,
            plan_path=plan_path,
            plan_sha256=plan_sha256,
            evidence_identities=evidence_identities,
        )
        output_asset_sha256 = _file_sha256(build_root)
        output_tree_sha256 = _tree_sha256(build_dir)
        final_tree, publication_outcome = _publish_tree(
            build_dir=build_dir,
            publish_root=publish_root,
            tree_sha256=output_tree_sha256,
        )
        build_dir = None
        if _tree_sha256(final_tree) != output_tree_sha256:
            raise _Blocked(
                "Published collision-filter output failed its content identity check."
            )
        final_root = final_tree / source_relative_root

        persistent_report = {
            key: value
            for key, value in report.items()
            if key not in {"asset_path", "plan_path"}
        }
        receipt = {
            **persistent_report,
            "changes": changes,
            "output_root_relative_path": source_relative_root.as_posix(),
            "output_asset_sha256": output_asset_sha256,
            "output_tree_sha256": output_tree_sha256,
            "source_unrelated_root_layer_sha256": source_root_layer_fingerprint,
            "output_unrelated_root_layer_sha256": output_root_layer_fingerprint,
            "dependencies_preserved": True,
            "evidence_artifact_integrity_verified": True,
            "geometry_and_topology_preserved": True,
            "readback_verified": True,
            "source_identity_verified": True,
            "status": "AUTHORED",
            "passed": True,
            "reason": (
                "Published an exact evidence-backed collision-filter derivative."
            ),
        }
        receipt_path = _publish_receipt(
            output_dir=output_dir,
            output_tree_sha256=output_tree_sha256,
            plan_sha256=plan_sha256,
            receipt=receipt,
        )
        runtime_report = {
            **receipt,
            "asset_path": str(asset_path),
            "plan_path": str(plan_path),
            "output_path": str(final_root),
            "receipt_path": str(receipt_path),
            "publication_outcome": publication_outcome,
            "reused_output": publication_outcome != "published",
        }
        return CollisionFilterResult(
            status="AUTHORED",
            passed=True,
            reason=receipt["reason"],
            output_path=final_root,
            receipt_path=receipt_path,
            report=runtime_report,
        )
    except (
        Tf.ErrorException,
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError,
        ValidationError,
        _Blocked,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        report.update(
            {
                "status": "BLOCKED",
                "passed": False,
                "reason": str(exc),
                "failure": str(exc),
                "changes": [],
            }
        )
        return CollisionFilterResult(
            status="BLOCKED",
            passed=False,
            reason=str(exc),
            output_path=Path(asset_path),
            receipt_path=None,
            report=report,
        )
    finally:
        if build_dir is not None:
            _remove_tree(build_dir, ignore_missing=True)
        if extraction_dir is not None:
            _remove_tree(extraction_dir, ignore_missing=True)


def filtered_pair_is_authored(stage: Any, body_a_path: str, body_b_path: str) -> bool:
    """Return whether a pair is represented in either OpenUSD direction."""

    if body_a_path == body_b_path:
        return False
    for source, target in (
        (body_a_path, body_b_path),
        (body_b_path, body_a_path),
    ):
        prim = stage.GetPrimAtPath(source)
        if not prim or _FILTER_API_NAME not in prim.GetAppliedSchemas():
            continue
        relation = prim.GetRelationship(_FILTER_RELATIONSHIP_NAME)
        if relation and target in {str(path) for path in relation.GetTargets()}:
            return True
    return False


def _load_plan(path: Path) -> tuple[CollisionFilterPlan, str]:
    payload = path.read_bytes()
    plan_sha256 = hashlib.sha256(payload).hexdigest()
    decoded = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json_number,
    )
    if not isinstance(decoded, dict):
        raise _Blocked("Collision-filter plan must be a JSON object.")
    return CollisionFilterPlan.model_validate(decoded), plan_sha256


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _Blocked(f"Collision-filter plan contains duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> None:
    raise _Blocked(f"Collision-filter plan contains non-finite JSON number: {value}")


def _prepare_output_dir(path: Path) -> Path:
    path = Path(path).expanduser().absolute()
    if path.is_symlink():
        raise _Blocked(f"Output directory cannot be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise _Blocked(f"Output path is not a directory: {path}")
    return path


def _regular_file(path: Path, *, label: str) -> Path:
    path = Path(path).expanduser().absolute()
    if path.is_symlink():
        raise _Blocked(f"{label.capitalize()} cannot be a symlink: {path}")
    try:
        path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise _Blocked(f"{label.capitalize()} does not exist: {path}") from exc
    if not path.is_file():
        raise _Blocked(f"{label.capitalize()} is not a regular file: {path}")
    return path


def _validate_evidence_artifacts(
    evidence: list[CollisionFilterEvidence],
) -> tuple[tuple[Path, str], ...]:
    identities: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for item in evidence:
        path = _regular_file(
            Path(item.artifact_path),
            label="collision-filter evidence artifact",
        )
        if str(path) != item.artifact_path:
            raise _Blocked(
                "Evidence artifact_path is not its exact canonical path: "
                f"expected {path}, received {item.artifact_path}."
            )
        if path in seen:
            raise _Blocked(f"Evidence artifact is listed more than once: {path}")
        seen.add(path)
        digest = _file_sha256(path)
        if digest != item.artifact_sha256:
            raise _Blocked(
                "Evidence artifact_sha256 is stale for "
                f"{path}: expected {digest}, received {item.artifact_sha256}."
            )
        identities.append((path, digest))
    return tuple(identities)


def _source_package(
    *, asset_path: Path, package_root: Path | None, output_dir: Path
) -> tuple[Path, Path, Path | None]:
    if asset_path.suffix.lower() == ".usdz":
        extraction_dir = Path(
            tempfile.mkdtemp(prefix=".collision-filter-source-", dir=output_dir)
        )
        extraction_complete = False
        try:
            root_relative = _extract_usdz_without_size_limit(
                asset_path=asset_path,
                extraction_dir=extraction_dir,
            )
            root_path = extraction_dir / root_relative
            if not root_path.is_file() or root_path.is_symlink():
                raise _Blocked(
                    "USDZ root layer was not safely extracted: "
                    f"{root_relative.as_posix()}"
                )
            extraction_complete = True
            return extraction_dir, root_path, extraction_dir
        finally:
            if not extraction_complete:
                _remove_tree(extraction_dir, ignore_missing=True)

    if asset_path.suffix.lower() not in _USD_LAYER_SUFFIXES:
        raise _Blocked(f"Unsupported source asset type: {asset_path.suffix}")
    package_root = _resolve_package_root(
        asset_path=asset_path,
        package_root=package_root,
    )
    return package_root, asset_path, None


def _resolve_package_root(*, asset_path: Path, package_root: Path | None) -> Path:
    package_root = Path(package_root or asset_path.parent).expanduser().absolute()
    if package_root.is_symlink():
        raise _Blocked(f"Package root cannot be a symlink: {package_root}")
    try:
        package_root = package_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise _Blocked(f"Package root does not exist: {package_root}") from exc
    if not package_root.is_dir():
        raise _Blocked(f"Package root is not a directory: {package_root}")
    if _relative_to(asset_path, package_root) is None:
        raise _Blocked(f"Source asset is outside package root: {asset_path}")
    return package_root


def _extract_usdz_without_size_limit(*, asset_path: Path, extraction_dir: Path) -> Path:
    """Stream every safe USDZ member without an arbitrary byte ceiling."""

    root_relative: Path | None = None
    with zipfile.ZipFile(asset_path) as archive:
        normalized_entries: dict[tuple[str, ...], zipfile.ZipInfo] = {}
        validated_members: list[tuple[zipfile.ZipInfo, Path, str]] = []
        for info in archive.infolist():
            parts = safe_usdz_member_parts(info.filename)
            if parts is None:
                raise _Blocked(f"USDZ contains an unsafe entry: {info.filename}")
            if len(parts) > _MAX_PACKAGE_PATH_DEPTH:
                raise _Blocked(
                    "USDZ entry exceeds the maximum package path depth of "
                    f"{_MAX_PACKAGE_PATH_DEPTH}: {info.filename}"
                )
            relative = Path(*parts)
            normalized = "/".join(parts)
            previous = normalized_entries.get(parts)
            if previous is not None:
                if previous.is_dir() != info.is_dir():
                    names = sorted((previous.filename, info.filename))
                    raise _Blocked(
                        "USDZ contains a file/directory collision: "
                        f"{names[0]} and {names[1]}"
                    )
                raise _Blocked(f"USDZ contains a duplicate entry: {normalized}")
            normalized_entries[parts] = info
            if _zip_info_is_symlink(info):
                raise _Blocked(f"USDZ contains a symlink entry: {normalized}")
            if info.flag_bits & 0x1:
                raise _Blocked(f"USDZ contains an encrypted entry: {normalized}")
            validated_members.append((info, relative, normalized))

        file_paths = {
            parts for parts, info in normalized_entries.items() if not info.is_dir()
        }
        for parts in sorted(normalized_entries):
            for depth in range(1, len(parts)):
                ancestor = parts[:depth]
                if ancestor in file_paths:
                    raise _Blocked(
                        "USDZ contains a file/member ancestor collision: "
                        f"{'/'.join(ancestor)} and {'/'.join(parts)}"
                    )

        for info, relative, normalized in validated_members:
            if not info.is_dir() and root_relative is None:
                if relative.suffix.lower() not in _USD_LAYER_SUFFIXES:
                    raise _Blocked(
                        "USDZ first file is not a supported USD root layer: "
                        f"{normalized}"
                    )
                root_relative = relative
        if root_relative is None:
            raise _Blocked(f"USDZ package has no file entries: {asset_path}")

        for info, relative, _normalized in validated_members:
            destination = extraction_dir / relative
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    return root_relative


def _zip_info_is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def _validate_dependency_closure(
    *, stage: Any, source_root: Path, source_tree: Path, Sdf: Any, UsdUtils: Any
) -> tuple[str, ...]:
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(source_root))
    metadata_only_paths = _metadata_only_asset_identity_paths(stage, Sdf=Sdf)
    unresolved_paths = tuple(sorted(str(item) for item in unresolved))
    blocking_unresolved = tuple(
        path for path in unresolved_paths if path not in metadata_only_paths
    )
    if blocking_unresolved:
        raise _Blocked(
            "USD dependency closure is unresolved: "
            + ", ".join(blocking_unresolved[:5])
        )

    dependency_paths: list[Path] = []
    for layer in layers:
        identifier = str(
            getattr(layer, "realPath", "") or getattr(layer, "identifier", "") or ""
        )
        if not identifier or identifier.startswith("anon:"):
            raise _Blocked(
                f"USD dependency layer has no stable local path: {identifier or layer}"
            )
        dependency_paths.append(Path(identifier))
    for asset in assets:
        authored_identifier = str(getattr(asset, "path", "") or asset)
        if authored_identifier in metadata_only_paths:
            continue
        identifier = str(
            getattr(asset, "resolvedPath", "") or getattr(asset, "path", "") or asset
        )
        if not identifier:
            continue
        dependency = Path(identifier)
        if not dependency.is_absolute():
            raise _Blocked(
                "USD dependency has no stable resolved local path: "
                f"{authored_identifier}"
            )
        dependency_paths.append(dependency)

    source_tree = source_tree.resolve(strict=True)
    for dependency in dependency_paths:
        dependency = dependency.resolve(strict=True)
        if _relative_to(dependency, source_tree) is None:
            raise _Blocked(f"USD dependency resolves outside package: {dependency}")
        if dependency.is_symlink() or not dependency.is_file():
            raise _Blocked(f"USD dependency is not a regular file: {dependency}")
    return tuple(sorted(metadata_only_paths))


def _metadata_only_asset_identity_paths(stage: Any, *, Sdf: Any) -> set[str]:
    identity_paths: set[str] = set()
    authored_dependency_paths: set[str] = set()
    for prim in stage.TraverseAll():
        identifier = prim.GetAssetInfo().get("identifier")
        if isinstance(identifier, Sdf.AssetPath) and identifier.path:
            identity_paths.add(str(identifier.path))
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() not in (
                Sdf.ValueTypeNames.Asset,
                Sdf.ValueTypeNames.AssetArray,
            ):
                continue
            values = [attribute.Get()]
            values.extend(attribute.Get(time) for time in attribute.GetTimeSamples())
            for value in values:
                if isinstance(value, Sdf.AssetPath) and value.path:
                    authored_dependency_paths.add(str(value.path))
                    continue
                if attribute.GetTypeName() == Sdf.ValueTypeNames.AssetArray and value:
                    authored_dependency_paths.update(
                        str(item.path)
                        for item in value
                        if isinstance(item, Sdf.AssetPath) and item.path
                    )
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        authored_dependency_paths.update(str(path) for path in layer.subLayerPaths)

        def collect_composition_arcs(path: Any) -> None:
            spec = layer.GetObjectAtPath(path)
            if not isinstance(spec, Sdf.PrimSpec):
                return
            authored_dependency_paths.update(
                str(item.assetPath)
                for item in spec.referenceList.GetAppliedItems()
                if item.assetPath
            )
            authored_dependency_paths.update(
                str(item.assetPath)
                for item in spec.payloadList.GetAppliedItems()
                if item.assetPath
            )

        layer.Traverse(Sdf.Path.absoluteRootPath, collect_composition_arcs)
    return identity_paths - authored_dependency_paths


def _validate_plan_against_stage(
    *, stage: Any, plan: CollisionFilterPlan, Sdf: Any, UsdPhysics: Any
) -> tuple[tuple[str, str], ...]:
    canonical_pairs: list[tuple[str, str]] = []
    for pair in plan.pairs:
        paths: list[str] = []
        for text in (pair.body_a_path, pair.body_b_path):
            path = Sdf.Path(text)
            if not path.IsAbsolutePath() or not path.IsPrimPath() or str(path) != text:
                raise _Blocked(
                    f"Collision-filter body path is not a canonical absolute prim path: {text}"
                )
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise _Blocked(f"Collision-filter body prim does not exist: {text}")
            if not prim.IsActive() or not prim.IsDefined() or prim.IsAbstract():
                raise _Blocked(
                    f"Collision-filter body is not an active defined prim: {text}"
                )
            if (
                prim.IsInstance()
                or prim.IsInstanceProxy()
                or prim.IsPrototype()
                or prim.IsInPrototype()
            ):
                raise _Blocked(
                    f"Collision-filter plans cannot edit an instance or prototype: {text}"
                )
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise _Blocked(
                    f"Collision-filter body lacks PhysicsRigidBodyAPI: {text}"
                )
            if not _body_owns_collision_prim(
                stage=stage,
                body=prim,
                UsdPhysics=UsdPhysics,
            ):
                raise _Blocked(
                    "Collision-filter body owns no active PhysicsCollisionAPI "
                    f"prim: {text}"
                )
            can_apply, why_not = UsdPhysics.FilteredPairsAPI.CanApply(prim)
            if not can_apply and not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                raise _Blocked(
                    f"PhysicsFilteredPairsAPI cannot be applied to {text}: {why_not}"
                )
            paths.append(text)
        first, second = sorted(paths)
        canonical_pairs.append((first, second))

    result = tuple(sorted(canonical_pairs))
    joint_pairs = _direct_joint_pairs(stage=stage, UsdPhysics=UsdPhysics)
    adjacent = sorted(set(result) & joint_pairs)
    if adjacent:
        raise _Blocked(
            "Collision-filter plan contains directly joint-adjacent bodies: "
            + ", ".join(f"{source} <-> {target}" for source, target in adjacent)
        )
    return result


def _body_owns_collision_prim(*, stage: Any, body: Any, UsdPhysics: Any) -> bool:
    """Return whether a rigid body owns a composed collider in its subtree."""

    for candidate in stage.TraverseAll():
        if (
            not candidate.IsActive()
            or not candidate.IsDefined()
            or candidate.IsAbstract()
            or not candidate.HasAPI(UsdPhysics.CollisionAPI)
        ):
            continue
        owner = candidate
        while owner and owner.IsValid() and not owner.IsPseudoRoot():
            if owner.HasAPI(UsdPhysics.RigidBodyAPI):
                if owner == body:
                    return True
                break
            owner = owner.GetParent()
    return False


def _direct_joint_pairs(*, stage: Any, UsdPhysics: Any) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = [path for path in joint.GetBody0Rel().GetTargets() if path.IsPrimPath()]
        body1 = [path for path in joint.GetBody1Rel().GetTargets() if path.IsPrimPath()]
        for first in body0:
            for second in body1:
                if first != second:
                    source, target = sorted((str(first), str(second)))
                    pairs.add((source, target))
    return pairs


def _filtered_targets(
    *,
    stage: Any,
    endpoint_paths: tuple[str, ...],
    Sdf: Any,
    UsdPhysics: Any,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for body_path in endpoint_paths:
        prim = stage.GetPrimAtPath(body_path)
        has_api = prim.HasAPI(UsdPhysics.FilteredPairsAPI)
        relation = prim.GetRelationship(_FILTER_RELATIONSHIP_NAME)
        if relation and not has_api:
            raise _Blocked(
                "physics:filteredPairs exists without PhysicsFilteredPairsAPI at "
                f"{body_path}."
            )
        targets: list[str] = []
        if relation:
            for target in relation.GetTargets():
                if not target.IsAbsolutePath() or not target.IsPrimPath():
                    raise _Blocked(
                        f"Invalid existing filtered-pair target at {body_path}: {target}"
                    )
                target_prim = stage.GetPrimAtPath(target)
                if not target_prim:
                    raise _Blocked(
                        f"Missing existing filtered-pair target at {body_path}: {target}"
                    )
                targets.append(str(Sdf.Path(target)))
        result[body_path] = tuple(sorted(set(targets)))
    return result


def _canonical_target_map(
    *,
    initial_targets: dict[str, tuple[str, ...]],
    canonical_pairs: tuple[tuple[str, str], ...],
) -> dict[str, tuple[str, ...]]:
    endpoints = {path for pair in canonical_pairs for path in pair}
    missing_endpoints = sorted(endpoints.difference(initial_targets))
    if missing_endpoints:
        raise _Blocked(
            "Cannot canonicalize collision-filter pairs because initial targets are "
            "missing endpoints: " + ", ".join(missing_endpoints)
        )
    desired = {path: set(targets) for path, targets in initial_targets.items()}
    for source, target in canonical_pairs:
        desired[target].discard(source)
        desired[source].add(target)
    return {path: tuple(sorted(targets)) for path, targets in desired.items()}


def _planned_targets_by_endpoint(
    *,
    endpoint_paths: tuple[str, ...],
    canonical_pairs: tuple[tuple[str, str], ...],
) -> dict[str, set[str]]:
    planned: dict[str, set[str]] = {path: set() for path in endpoint_paths}
    for source, target in canonical_pairs:
        planned[source].add(target)
        planned[target].add(source)
    return planned


def _root_api_list_ops(
    *, root_layer: Any, endpoint_paths: tuple[str, ...], Sdf: Any
) -> dict[str, _ListOpState | None]:
    result: dict[str, _ListOpState | None] = {}
    for body_path in endpoint_paths:
        prim_spec = root_layer.GetPrimAtPath(Sdf.Path(body_path))
        operation = (
            prim_spec.GetInfo("apiSchemas")
            if prim_spec is not None and prim_spec.HasInfo("apiSchemas")
            else None
        )
        result[body_path] = _list_op_state(operation)
    return result


def _root_target_list_ops(
    *, root_layer: Any, endpoint_paths: tuple[str, ...], Sdf: Any
) -> dict[str, _ListOpState | None]:
    result: dict[str, _ListOpState | None] = {}
    for body_path in endpoint_paths:
        prim_spec = root_layer.GetPrimAtPath(Sdf.Path(body_path))
        relationship = (
            prim_spec.GetPropertyAtPath(
                Sdf.Path(body_path).AppendProperty(_FILTER_RELATIONSHIP_NAME)
            )
            if prim_spec is not None
            else None
        )
        operation = (
            relationship.GetInfo("targetPaths")
            if relationship is not None and relationship.HasInfo("targetPaths")
            else None
        )
        result[body_path] = _list_op_state(operation)
    return result


def _list_op_state(operation: Any | None) -> _ListOpState | None:
    if operation is None:
        return None
    return _ListOpState(
        is_explicit=bool(operation.isExplicit),
        explicit=tuple(str(item) for item in operation.explicitItems),
        prepended=tuple(str(item) for item in operation.prependedItems),
        appended=tuple(str(item) for item in operation.appendedItems),
        added=tuple(str(item) for item in operation.addedItems),
        deleted=tuple(str(item) for item in operation.deletedItems),
        ordered=tuple(str(item) for item in operation.orderedItems),
    )


def _require_list_op_items_preserved(
    *,
    before: _ListOpState | None,
    after: _ListOpState | None,
    allowed_items: set[str],
    label: str,
) -> None:
    empty = _ListOpState(False, (), (), (), (), (), ())
    normalized_before = _list_op_without_items(before, allowed_items) or empty
    normalized_after = _list_op_without_items(after, allowed_items) or empty
    if normalized_after != normalized_before:
        raise _Blocked(f"Collision-filter authoring changed unrelated {label} items.")


def _list_op_without_items(
    state: _ListOpState | None, ignored_items: set[str]
) -> _ListOpState | None:
    if state is None:
        return None

    def kept(items: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(item for item in items if item not in ignored_items)

    return _ListOpState(
        is_explicit=state.is_explicit,
        explicit=kept(state.explicit),
        prepended=kept(state.prepended),
        appended=kept(state.appended),
        added=kept(state.added),
        deleted=kept(state.deleted),
        ordered=kept(state.ordered),
    )


def _author_filtered_pairs(
    *,
    stage: Any,
    endpoint_paths: tuple[str, ...],
    canonical_pairs: tuple[tuple[str, str], ...],
    desired_targets: dict[str, tuple[str, ...]],
    Sdf: Any,
    UsdPhysics: Any,
) -> tuple[
    list[dict[str, Any]],
    dict[str, _ListOpState | None],
    dict[str, _ListOpState | None],
]:
    canonical_sources = {source for source, _target in canonical_pairs}
    planned_targets = _planned_targets_by_endpoint(
        endpoint_paths=endpoint_paths,
        canonical_pairs=canonical_pairs,
    )
    root_layer = stage.GetRootLayer()
    before_api_list_ops = _root_api_list_ops(
        root_layer=root_layer,
        endpoint_paths=endpoint_paths,
        Sdf=Sdf,
    )
    before_target_list_ops = _root_target_list_ops(
        root_layer=root_layer,
        endpoint_paths=endpoint_paths,
        Sdf=Sdf,
    )
    changes: list[dict[str, Any]] = []
    for body_path in endpoint_paths:
        prim = stage.GetPrimAtPath(body_path)
        api = UsdPhysics.FilteredPairsAPI(prim)
        had_api = bool(api)
        relation = prim.GetRelationship(_FILTER_RELATIONSHIP_NAME)
        before = (
            tuple(sorted({str(path) for path in relation.GetTargets()}))
            if relation
            else ()
        )
        needs_api = body_path in canonical_sources
        applied_api = False
        if needs_api and not api:
            api = UsdPhysics.FilteredPairsAPI.Apply(prim)
            if not api:
                raise _Blocked(
                    f"Could not apply PhysicsFilteredPairsAPI to {body_path}."
                )
            applied_api = True
        after = desired_targets[body_path]
        if before != after:
            if not api:
                api = UsdPhysics.FilteredPairsAPI.Apply(prim)
                if not api:
                    raise _Blocked(
                        f"Could not apply PhysicsFilteredPairsAPI to {body_path}."
                    )
                applied_api = True
            relation = api.CreateFilteredPairsRel()
            for target in sorted(set(before).difference(after)):
                if not relation.RemoveTarget(Sdf.Path(target)):
                    raise _Blocked(
                        "Could not remove physics:filteredPairs target at "
                        f"{body_path}: {target}."
                    )
            for target in sorted(set(after).difference(before)):
                if not relation.AddTarget(Sdf.Path(target)):
                    raise _Blocked(
                        "Could not add physics:filteredPairs target at "
                        f"{body_path}: {target}."
                    )
        if applied_api or before != after:
            changes.append(
                {
                    "action": "author_canonical_filtered_pairs",
                    "body_path": body_path,
                    "before_targets": list(before),
                    "after_targets": list(after),
                    "applied_api": applied_api and not had_api,
                }
            )

    after_api_list_ops = _root_api_list_ops(
        root_layer=root_layer,
        endpoint_paths=endpoint_paths,
        Sdf=Sdf,
    )
    after_target_list_ops = _root_target_list_ops(
        root_layer=root_layer,
        endpoint_paths=endpoint_paths,
        Sdf=Sdf,
    )
    for body_path in endpoint_paths:
        _require_list_op_items_preserved(
            before=before_api_list_ops[body_path],
            after=after_api_list_ops[body_path],
            allowed_items={_FILTER_API_NAME},
            label=f"apiSchemas at {body_path}",
        )
        _require_list_op_items_preserved(
            before=before_target_list_ops[body_path],
            after=after_target_list_ops[body_path],
            allowed_items=planned_targets[body_path],
            label=f"physics:filteredPairs at {body_path}",
        )
    return changes, after_api_list_ops, after_target_list_ops


def _verify_filtered_pairs(
    *,
    stage: Any,
    endpoint_paths: tuple[str, ...],
    canonical_pairs: tuple[tuple[str, str], ...],
    desired_targets: dict[str, tuple[str, ...]],
    expected_api_list_ops: dict[str, _ListOpState | None],
    expected_target_list_ops: dict[str, _ListOpState | None],
    Sdf: Any,
    UsdPhysics: Any,
) -> None:
    actual = _filtered_targets(
        stage=stage,
        endpoint_paths=endpoint_paths,
        Sdf=Sdf,
        UsdPhysics=UsdPhysics,
    )
    if actual != desired_targets:
        raise _Blocked(
            "Collision-filter relationship readback does not match the plan."
        )
    root_layer = stage.GetRootLayer()
    if (
        _root_api_list_ops(
            root_layer=root_layer,
            endpoint_paths=endpoint_paths,
            Sdf=Sdf,
        )
        != expected_api_list_ops
    ):
        raise _Blocked("Collision-filter apiSchemas list-op changed on reopen.")
    if (
        _root_target_list_ops(
            root_layer=root_layer,
            endpoint_paths=endpoint_paths,
            Sdf=Sdf,
        )
        != expected_target_list_ops
    ):
        raise _Blocked(
            "Collision-filter relationship target list-op changed on reopen."
        )
    for source, target in canonical_pairs:
        source_prim = stage.GetPrimAtPath(source)
        if not source_prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            raise _Blocked(
                f"Canonical filter source lacks API after authoring: {source}"
            )
        if target not in actual[source] or source in actual[target]:
            raise _Blocked(
                f"Collision-filter pair is not canonical one-way: {source} <-> {target}"
            )
        if not filtered_pair_is_authored(stage, source, target):
            raise _Blocked(
                f"Collision-filter pair is absent after authoring: {source} <-> {target}"
            )


def _root_layer_fingerprint_without_filters(
    *,
    stage: Any,
    endpoint_paths: tuple[str, ...],
    Sdf: Any,
) -> str:
    root_layer_text = stage.GetRootLayer().ExportToString()
    normalized = Sdf.Layer.CreateAnonymous("collision-filter-normalized.usda")
    if not normalized.ImportFromString(root_layer_text):
        raise _Blocked("Could not clone the collision-filter root layer.")
    for body_path in endpoint_paths:
        prim_spec = normalized.GetPrimAtPath(Sdf.Path(body_path))
        if prim_spec is None:
            continue
        relationship = prim_spec.GetPropertyAtPath(
            Sdf.Path(body_path).AppendProperty(_FILTER_RELATIONSHIP_NAME)
        )
        if relationship is not None:
            prim_spec.RemoveProperty(relationship)
        if prim_spec.HasInfo("apiSchemas"):
            api_schemas = prim_spec.GetInfo("apiSchemas")
            _remove_filter_api_from_list_op(api_schemas)
            if _list_op_has_items(api_schemas):
                prim_spec.SetInfo("apiSchemas", api_schemas)
            else:
                prim_spec.ClearInfo("apiSchemas")
        if prim_spec.IsInert():
            normalized.ScheduleRemoveIfInert(prim_spec)
    return hashlib.sha256(normalized.ExportToString().encode("utf-8")).hexdigest()


def _remove_filter_api_from_list_op(list_op: Any) -> None:
    if list_op.isExplicit:
        list_op.explicitItems = [
            item for item in list_op.explicitItems if str(item) != _FILTER_API_NAME
        ]
        return

    values = {
        attribute_name: [
            item
            for item in getattr(list_op, attribute_name)
            if str(item) != _FILTER_API_NAME
        ]
        for attribute_name in (
            "prependedItems",
            "appendedItems",
            "addedItems",
            "deletedItems",
            "orderedItems",
        )
    }
    for attribute_name, items in values.items():
        setattr(list_op, attribute_name, items)


def _list_op_has_items(list_op: Any) -> bool:
    return bool(list_op.isExplicit) or any(
        getattr(list_op, attribute_name)
        for attribute_name in (
            "explicitItems",
            "prependedItems",
            "appendedItems",
            "addedItems",
            "deletedItems",
            "orderedItems",
        )
    )


def _copy_package_tree(*, source_tree: Path, build_dir: Path) -> None:
    _validate_tree_layout(source_tree)
    shutil.copytree(
        source_tree, build_dir, dirs_exist_ok=True, copy_function=shutil.copy2
    )
    _make_owner_writable(build_dir)
    _validate_tree_layout(build_dir)


def _validate_tree_layout(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise _Blocked(f"Package tree is not a regular directory: {root}")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if len(relative.parts) > _MAX_PACKAGE_PATH_DEPTH:
            raise _Blocked(
                "Package tree entry exceeds the maximum path depth of "
                f"{_MAX_PACKAGE_PATH_DEPTH}: {relative.as_posix()}"
            )
        if path.is_symlink():
            raise _Blocked(f"Package tree contains a symlink: {path}")
        if not path.is_dir() and not path.is_file():
            raise _Blocked(f"Package tree contains a non-file entry: {path}")


def _tree_file_sha256(root: Path) -> dict[str, str]:
    _validate_tree_layout(root)
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(
            root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
        )
        if path.is_file()
    }


def _tree_sha256(root: Path) -> str:
    _validate_tree_layout(root)
    digest = hashlib.sha256()
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        digest.update(b"D\0" if path.is_dir() else b"F\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_unchanged(
    *,
    asset_path: Path,
    source_asset_sha256: str,
    source_tree: Path,
    source_tree_sha256: str,
    plan_path: Path,
    plan_sha256: str,
    evidence_identities: tuple[tuple[Path, str], ...],
) -> None:
    if _file_sha256(asset_path) != source_asset_sha256:
        raise _Blocked("Source asset changed while collision filters were authored.")
    if _tree_sha256(source_tree) != source_tree_sha256:
        raise _Blocked("Source package changed while collision filters were authored.")
    if _file_sha256(plan_path) != plan_sha256:
        raise _Blocked("Collision-filter plan changed while output was authored.")
    for evidence_path, evidence_sha256 in evidence_identities:
        if _file_sha256(evidence_path) != evidence_sha256:
            raise _Blocked(
                "Collision-filter evidence changed while output was authored: "
                f"{evidence_path}"
            )


def _publish_tree(
    *, build_dir: Path, publish_root: Path, tree_sha256: str
) -> tuple[Path, str]:
    if build_dir.parent != publish_root:
        raise _Blocked("Atomic publication requires a same-parent build directory.")
    final_tree = publish_root / tree_sha256
    if final_tree.exists() or final_tree.is_symlink():
        _verify_existing_tree(final_tree=final_tree, tree_sha256=tree_sha256)
        _remove_tree(build_dir)
        return final_tree, "cache_hit"
    try:
        _atomic_rename(build_dir, final_tree)
    except OSError:
        if not final_tree.exists() and not final_tree.is_symlink():
            raise
        _verify_existing_tree(final_tree=final_tree, tree_sha256=tree_sha256)
        _remove_tree(build_dir)
        return final_tree, "concurrent_reuse"
    return final_tree, "published"


def _atomic_rename(source: Path, target: Path) -> None:
    source.rename(target)


def _verify_existing_tree(*, final_tree: Path, tree_sha256: str) -> None:
    if final_tree.is_symlink() or not final_tree.is_dir():
        raise _Blocked(
            f"Collision-filter content-addressed output is invalid: {final_tree}"
        )
    if _tree_sha256(final_tree) != tree_sha256:
        raise _Blocked(
            f"Existing collision-filter output failed identity check: {final_tree}"
        )


def _publish_receipt(
    *,
    output_dir: Path,
    output_tree_sha256: str,
    plan_sha256: str,
    receipt: dict[str, Any],
) -> Path:
    receipt_root = output_dir / "receipts"
    if receipt_root.is_symlink():
        raise _Blocked(f"Receipt root cannot be a symlink: {receipt_root}")
    receipt_root.mkdir(parents=True, exist_ok=True)
    if not receipt_root.is_dir():
        raise _Blocked(f"Receipt root is not a directory: {receipt_root}")
    receipt_dir = receipt_root / "collision-filter"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    if receipt_dir.is_symlink() or not receipt_dir.is_dir():
        raise _Blocked(f"Receipt directory is not a regular directory: {receipt_dir}")
    receipt_path = receipt_dir / f"{output_tree_sha256}-{plan_sha256}.json"
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if receipt_path.exists() or receipt_path.is_symlink():
        _verify_existing_receipt(receipt_path, payload)
        return receipt_path

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".collision-filter-receipt-", dir=receipt_dir
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, receipt_path)
        except FileExistsError:
            _verify_existing_receipt(receipt_path, payload)
    finally:
        temporary_path.unlink(missing_ok=True)
    return receipt_path


def _verify_existing_receipt(path: Path, expected: bytes) -> None:
    if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
        raise _Blocked(
            f"Existing collision-filter receipt has conflicting bytes: {path}"
        )


def _make_owner_writable(path: Path) -> None:
    mode = path.stat(follow_symlinks=False).st_mode
    if stat.S_ISLNK(mode):
        raise _Blocked(f"Cannot make a symlink writable: {path}")
    writable_mode = mode | stat.S_IRUSR | stat.S_IWUSR
    if stat.S_ISDIR(mode):
        writable_mode |= stat.S_IXUSR
    os.chmod(path, writable_mode, follow_symlinks=False)
    if path.is_dir():
        for child in path.iterdir():
            _make_owner_writable(child)


def _remove_tree(path: Path, *, ignore_missing: bool = False) -> None:
    if not path.exists() and not path.is_symlink():
        if ignore_missing:
            return
        raise FileNotFoundError(path)
    _make_owner_writable(path)
    shutil.rmtree(path)


def _relative_to(path: Path, root: Path) -> Path | None:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Author exact owner-approved OpenUSD collision-filter pairs."
    )
    parser.add_argument("asset", type=Path, help="Local USD/USDZ source asset.")
    parser.add_argument("plan", type=Path, help="Strict collision-filter plan JSON.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--package-root",
        type=Path,
        help="Complete source package root for non-USDZ assets.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run exact collision-filter authoring from the command line."""

    args = _build_parser().parse_args(argv)
    try:
        result = author_collision_filter_derivative(
            asset_path=args.asset,
            plan_path=args.plan,
            output_dir=args.output_dir,
            package_root=args.package_root,
        )
    except ImportError as exc:
        print(
            json.dumps(
                {"error": str(exc), "passed": False, "status": "BLOCKED"},
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    payload = {
        **result.report,
        "output_path": str(result.output_path),
        "receipt_path": (
            str(result.receipt_path) if result.receipt_path is not None else None
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
